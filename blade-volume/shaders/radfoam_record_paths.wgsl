// GPU path-recording for differentiable RadFoam training.
//
// Traces one ray per output slot through the foam — same Voronoi /
// power-diagram traversal as `radfoam_trace.wgsl` — but instead of
// accumulating colour it writes a `(previous_cell, cell, next_cell, dt,
// mask)` tuple for every visible segment. Weighted paths additionally write
// the exact local derivative of `dt` with respect to each role's
// `(x, y, z, radius)`, the selected site's oriented surface normal, and the ray-relative reference tangent
// `J * (geometry_ref - ray_origin)`. The cell sequence and active min/max
// branch remain discrete; the training graph consumes the recorded
// linearization as `dt_ref + tangent_actual - tangent_ref`.
//
// `next_cell_indices` enables differentiable position optimisation:
// with `cell` and `next_cell` known per step, the graph can compute
// `dt` from `positions[cell]` and `positions[next_cell]` (the bisector
// plane between them) and the ray geometry — making `positions` an
// optimisable parameter while the cell-sequence decision (an argmin
// over face intersections) stays in this non-differentiable shader.
//
// Output layout: row-major `[num_pixels, max_steps]` flat buffers.
// The PowerFoam gather workgroup initializes the index/mask row before the
// record pass writes active steps. The unweighted walker leaves trailing
// slots at their pre-dispatch value.

// #include "common.wgsl" — required for `qrot`
// #include "surface_detail.wgsl"
// #include "radfoam.wgsl" partial: we redeclare accessors below so this
//   file is self-contained for codegen.

struct RecordParams {
    /// Starting cell index (same for every ray in this dispatch).
    start_point: u32,
    /// Maximum path entries per pixel (= L in `[P, L]` tensors).
    max_steps: u32,
    /// Pixels in this dispatch (= P).
    num_pixels: u32,
    /// First pixel/output row owned by this dispatch.
    pixel_offset: u32,
    /// Source image width (for converting pixel index → (ix, iy)).
    image_width: u32,
    image_height: u32,
    /// Saturating cap for `dt` to keep the sigmoid surrogate finite
    /// when the ray escapes through a giant cell. Matches the CPU
    /// path's `MAX_PATH_DT` clamp.
    max_path_dt: f32,
    /// Far plane of the camera frustum.
    depth: f32,
    /// Non-zero for bounded PowerFoam support spheres.
    power_foam: u32,
    /// Number of sites in `g_points`.
    num_points: u32,
    /// Scratch slots reserved per sampled ray for sphere candidates.
    candidate_capacity: u32,
    /// 0 = intervals only, 1 = complete geometry, 2 = oriented surface only.
    jacobian_mode: u32,
    /// Projected 16x16 screen-tile layout. `tile_capacity == 0` selects the
    /// exhaustive point scan.
    tile_width: u32,
    tile_height: u32,
    tile_capacity: u32,
    /// Non-zero when cells are split by learned oriented surface planes.
    oriented: u32,
};

struct Camera {
    position: vec3<f32>,
    _pad0: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    principal: vec2<f32>,
};

var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_surface_normals: array<vec4<f32>>;
var<storage, read> g_surface_details: array<vec4<f32>>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;
var<storage, read> g_pixel_indices: array<u32>;
var<storage, read_write> g_previous_cells_out: array<u32>;
var<storage, read_write> g_cells_out: array<u32>;
var<storage, read_write> g_next_cells_out: array<u32>;
var<storage, read_write> g_dts_out: array<f32>;
var<storage, read_write> g_mask_out: array<f32>;
// Low 31 bits: recorded entries. High bit: another valid segment remained
// after the fixed path row filled.
var<storage, read_write> g_path_status_out: array<u32>;
var<storage, read_write> g_dt_reference_tangents_out: array<f32>;
var<storage, read_write> g_dt_grad_previous_out: array<vec4<f32>>;
var<storage, read_write> g_dt_grad_current_out: array<vec4<f32>>;
var<storage, read_write> g_dt_grad_next_out: array<vec4<f32>>;
var<storage, read_write> g_dt_grad_surface_normal_out: array<vec4<f32>>;
var<storage, read_write> g_surface_queries_out: array<vec2<f32>>;
var<storage, read_write> g_surface_query_grad_previous_out: array<vec4<f32>>;
var<storage, read_write> g_surface_query_grad_current_out: array<vec4<f32>>;
var<storage, read_write> g_candidate_counts: array<u32>;
var<storage, read_write> g_candidates: array<u32>;
var<storage, read_write> g_candidate_depths: array<f32>;
var<storage, read_write> g_candidate_faces: array<vec2<f32>>;
var<storage, read_write> g_candidate_neighbors: array<vec2<u32>>;
var<storage, read_write> g_projected_bounds: array<vec4<u32>>;
var<storage, read_write> g_tile_counts: array<u32>;
var<storage, read_write> g_tile_candidates: array<u32>;
var<uniform> g_camera: Camera;
var<uniform> g_params: RecordParams;

var<workgroup> w_candidate_count: atomic<u32>;
var<workgroup> w_tile_count: atomic<u32>;
var<workgroup> w_parallel_cells: array<u32, 64>;
var<workgroup> w_parallel_depths: array<f32, 64>;
var<workgroup> w_parallel_faces: array<vec2<f32>, 64>;
var<workgroup> w_parallel_neighbors: array<vec2<u32>, 64>;
var<workgroup> w_parallel_valid: array<u32, 64>;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn ray_dir_for_pixel(pidx: u32) -> vec3<f32> {
    let ix = pidx % g_params.image_width;
    let iy = pidx / g_params.image_width;
    let px = (f32(ix) + 0.5) / f32(g_params.image_width);
    let py = (f32(iy) + 0.5) / f32(g_params.image_height);
    let ndc_x = px * 2.0 - 1.0;
    let ndc_y = py * 2.0 - 1.0;
    let tan_half = vec2<f32>(tan(0.5 * g_camera.fov.x), tan(0.5 * g_camera.fov.y));
    let local = vec3<f32>(
        (ndc_x - g_camera.principal.x) * tan_half.x,
        (ndc_y - g_camera.principal.y) * tan_half.y,
        1.0,
    );
    return normalize(qrot(g_camera.orientation, local));
}

struct FaceJacobians {
    current: vec4<f32>,
    adjacent: vec4<f32>,
};

struct SphereIntersections {
    near_t: f32,
    far_t: f32,
    root: f32,
    near_jacobian: vec4<f32>,
    far_jacobian: vec4<f32>,
    valid: u32,
};

struct IntervalDifferential {
    dt: f32,
    dt_d_previous: vec4<f32>,
    dt_d_current: vec4<f32>,
    dt_d_next: vec4<f32>,
    dt_d_surface_normal: vec4<f32>,
    surface_query_d_previous: vec4<f32>,
    surface_query_d_current: vec4<f32>,
    surface_query: vec2<f32>,
    surface_offset: f32,
    valid: u32,
};

struct PowerInterval {
    face_near: f32,
    face_far: f32,
    effective_near: f32,
    previous: u32,
    next: u32,
    valid: u32,
};

fn has_surface_detail() -> bool {
    return (g_params.oriented & 2u) != 0u;
}

fn effective_surface_offset(
    cell: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    query_near: f32,
) -> f32 {
    let surface = g_surface_normals[cell];
    if (!has_surface_detail()) {
        return surface.w;
    }
    var sites: array<vec4<f32>, SURFACE_DETAIL_SITES>;
    for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
        sites[site] = g_surface_details[cell * SURFACE_DETAIL_SITES + site];
    }
    let point = g_points[cell];
    return surface_detail_height(
        point.xyz,
        point.w,
        surface.xyz,
        surface.w,
        ray_origin,
        ray_dir,
        query_near,
        sites,
    );
}

struct ProjectedTileBounds {
    min_tile: vec2<u32>,
    max_tile: vec2<u32>,
    valid: u32,
};

fn inside_camera_exclusion(delta: vec3<f32>, radius: f32) -> bool {
    let exclusion_radius = 4.0 * radius;
    return dot(delta, delta) < exclusion_radius * exclusion_radius;
}

fn face_intersection_jacobians(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    current: vec4<f32>,
    adjacent: vec4<f32>,
    t: f32,
) -> FaceJacobians {
    let normal = adjacent.xyz - current.xyz;
    let denominator = dot(ray_dir, normal);
    if (abs(denominator) <= 1e-20) {
        return FaceJacobians(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let current_xyz = (ray_origin - current.xyz + t * ray_dir) / denominator;
    let adjacent_xyz = (adjacent.xyz - ray_origin - t * ray_dir) / denominator;
    return FaceJacobians(
        vec4<f32>(current_xyz, current.w / denominator),
        vec4<f32>(adjacent_xyz, -adjacent.w / denominator),
    );
}

fn sphere_intersections(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    sphere: vec4<f32>,
) -> SphereIntersections {
    let oc = ray_origin - sphere.xyz;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere.w * sphere.w;
    let discriminant = b * b - c;
    if (discriminant <= 0.0) {
        return SphereIntersections(
            0.0, 0.0, 0.0, vec4<f32>(0.0), vec4<f32>(0.0), 0u,
        );
    }
    let root = sqrt(discriminant);
    let perpendicular = oc - b * ray_dir;
    let root_d_center = perpendicular / root;
    let root_d_radius = sphere.w / root;
    return SphereIntersections(
        -b - root,
        -b + root,
        root,
        vec4<f32>(ray_dir - root_d_center, -root_d_radius),
        vec4<f32>(ray_dir + root_d_center, root_d_radius),
        1u,
    );
}

fn interval_differential(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    previous_idx: u32,
    current_idx: u32,
    next_idx: u32,
    t0: f32,
    t1: f32,
) -> IntervalDifferential {
    if (g_params.power_foam == 0u) {
        return IntervalDifferential(
            min(t1 - t0, g_params.max_path_dt),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec2<f32>(t0, 0.0), 0.0,
            select(0u, 1u, t1 > t0),
        );
    }

    let current = g_points[current_idx];
    let sphere = sphere_intersections(ray_origin, ray_dir, current);
    if (sphere.valid == 0u) {
        return IntervalDifferential(
            0.0, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec2<f32>(t0, 0.0), 0.0, 0u,
        );
    }

    var start = t0;
    var end = t1;
    var start_from_sphere = false;
    var end_from_sphere = false;
    var start_from_surface = false;
    var end_from_surface = false;
    var surface_query = vec2<f32>(start, 0.0);
    var effective_offset = 0.0;
    var surface_center_jacobian = vec3<f32>(0.0);
    var surface_plane_jacobian = vec4<f32>(0.0);
    if (sphere.near_t > start) {
        start = sphere.near_t;
        start_from_sphere = true;
    }
    if (sphere.far_t < end) {
        end = sphere.far_t;
        end_from_sphere = true;
    }
    surface_query.x = start;
    var surface_query_d_previous = vec4<f32>(0.0);
    var surface_query_d_current = vec4<f32>(0.0);
    if (start_from_sphere) {
        surface_query_d_current = sphere.near_jacobian;
    } else if (previous_idx != current_idx) {
        let query_face = face_intersection_jacobians(
            ray_origin, ray_dir, current, g_points[previous_idx], t0,
        );
        surface_query_d_current = query_face.current;
        surface_query_d_previous = query_face.adjacent;
    }
    if ((g_params.oriented & 1u) != 0u) {
        let surface_data = g_surface_normals[current_idx];
        let normal = surface_data.xyz;
        let offset = effective_surface_offset(current_idx, ray_origin, ray_dir, start);
        effective_offset = offset;
        let denominator = dot(ray_dir, normal);
        surface_query = vec2<f32>(start, select(0.0, 1.0, denominator < -1e-20));
        if (abs(denominator) <= 1e-20) {
            if (dot(ray_origin - current.xyz, normal) > offset) {
                return IntervalDifferential(
                    0.0, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
                    vec4<f32>(0.0), surface_query_d_previous,
                    surface_query_d_current, surface_query, effective_offset, 0u,
                );
            }
        } else {
            let relative_center = current.xyz - ray_origin;
            let numerator = dot(relative_center, normal) + offset;
            let surface_t = numerator / denominator;
            surface_center_jacobian = normal / denominator;
            surface_plane_jacobian = vec4<f32>(
                (relative_center * denominator - numerator * ray_dir) /
                    (denominator * denominator),
                1.0 / denominator,
            );
            if (denominator > 0.0 && surface_t < end) {
                end = surface_t;
                end_from_surface = true;
            } else if (denominator < 0.0 && surface_t > start) {
                start = surface_t;
                start_from_surface = true;
            }
        }
    }
    if (end <= start) {
        return IntervalDifferential(
            0.0, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), surface_query_d_previous,
            surface_query_d_current, surface_query, effective_offset, 0u,
        );
    }

    var dt_d_previous = vec4<f32>(0.0);
    var dt_d_current = vec4<f32>(0.0);
    var dt_d_next = vec4<f32>(0.0);
    var dt_d_surface_normal = vec4<f32>(0.0);
    if (start_from_surface) {
        dt_d_current -= vec4<f32>(surface_center_jacobian, 0.0);
        dt_d_surface_normal -= surface_plane_jacobian;
    } else {
        dt_d_current -= surface_query_d_current;
        dt_d_previous -= surface_query_d_previous;
    }
    if (end_from_surface) {
        dt_d_current += vec4<f32>(surface_center_jacobian, 0.0);
        dt_d_surface_normal += surface_plane_jacobian;
    } else if (end_from_sphere) {
        dt_d_current += sphere.far_jacobian;
    } else if (next_idx != current_idx) {
        let face = face_intersection_jacobians(
            ray_origin, ray_dir, current, g_points[next_idx], t1,
        );
        dt_d_current += face.current;
        dt_d_next += face.adjacent;
    }

    var dt = end - start;
    if (dt > g_params.max_path_dt) {
        // The forward clamp is locally constant on its saturated branch.
        dt = g_params.max_path_dt;
        dt_d_previous = vec4<f32>(0.0);
        dt_d_current = vec4<f32>(0.0);
        dt_d_next = vec4<f32>(0.0);
        dt_d_surface_normal = vec4<f32>(0.0);
    }
    return IntervalDifferential(
        dt, dt_d_previous, dt_d_current, dt_d_next, dt_d_surface_normal,
        surface_query_d_previous, surface_query_d_current,
        surface_query, effective_offset, 1u,
    );
}

// Reconstruct one gathered support-sphere interval from its cached square root,
// then clip it against every radical plane in its Cech neighborhood. A
// non-overlapping sphere cannot win power distance anywhere inside the current
// support, so overlapping-ball neighbors are the complete clipping set.
fn power_interval(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    cell: u32,
    sphere_root: f32,
) -> PowerInterval {
    let current = g_points[cell];
    let sphere_b = dot(ray_origin - current.xyz, ray_dir);
    let sphere_near = -sphere_b - sphere_root;
    let sphere_far = -sphere_b + sphere_root;

    var face_near = 0.0;
    var face_far = g_params.depth;
    var previous = cell;
    var next = cell;
    let begin = g_adjacency_offsets[cell];
    let end = g_adjacency_offsets[cell + 1u];
    for (var j = begin; j < end; j += 1u) {
        let adjacent_idx = g_adjacency[j];
        let adjacent = g_points[adjacent_idx];
        let normal = adjacent.xyz - current.xyz;
        let dsq = max(dot(normal, normal), 1e-20);
        let shift = 0.5 + 0.5 * (current.w * current.w - adjacent.w * adjacent.w) / dsq;
        let face_origin = current.xyz + shift * normal;
        let numerator = dot(face_origin - ray_origin, normal);
        let denominator = dot(ray_dir, normal);

        if (denominator > 1e-20) {
            let t = numerator / denominator;
            if (t < face_far) {
                face_far = t;
                next = adjacent_idx;
            }
        } else if (denominator < -1e-20) {
            let t = numerator / denominator;
            if (t > face_near) {
                face_near = t;
                previous = adjacent_idx;
            }
        } else if (numerator < 0.0) {
            // The ray is parallel to this face and lies outside the cell.
            return PowerInterval(0.0, 0.0, 0.0, cell, cell, 0u);
        }
    }

    var effective_near = max(face_near, sphere_near);
    var effective_far = min(face_far, sphere_far);
    let query_near = effective_near;
    if ((g_params.oriented & 1u) != 0u) {
        let surface_data = g_surface_normals[cell];
        let surface_normal = surface_data.xyz;
        let surface_offset = effective_surface_offset(cell, ray_origin, ray_dir, query_near);
        let denominator = dot(ray_dir, surface_normal);
        if (abs(denominator) <= 1e-20) {
            if (dot(ray_origin - current.xyz, surface_normal) > surface_offset) {
                return PowerInterval(0.0, 0.0, 0.0, cell, cell, 0u);
            }
        } else {
            let surface_t =
                (dot(current.xyz - ray_origin, surface_normal) + surface_offset) /
                denominator;
            if (denominator > 0.0) {
                effective_far = min(effective_far, surface_t);
            } else {
                effective_near = max(effective_near, surface_t);
            }
        }
    }
    if (effective_far <= effective_near) {
        return PowerInterval(0.0, 0.0, 0.0, cell, cell, 0u);
    }
    return PowerInterval(face_near, face_far, effective_near, previous, next, 1u);
}

// Conservative perspective bounds for one support sphere. The camera-space
// axis-aligned cube encloses the sphere, and all eight x/z and y/z corner
// ratios bound its projection while the cube stays in front of the camera.
// A sphere crossing the camera plane is assigned to the whole image. Exact
// ray/sphere testing in the gather pass removes all false positives.
fn projected_tile_bounds(sphere_data: vec4<f32>) -> ProjectedTileBounds {
    let inverse_orientation = vec4<f32>(
        -g_camera.orientation.xyz,
        g_camera.orientation.w,
    );
    let center = qrot(inverse_orientation, sphere_data.xyz - g_camera.position);
    let radius = sphere_data.w;
    if (radius <= 0.0 ||
        inside_camera_exclusion(center, radius) ||
        center.z + radius <= 0.0) {
        return ProjectedTileBounds(vec2<u32>(0u), vec2<u32>(0u), 0u);
    }

    var ndc_min = vec2<f32>(-1.0);
    var ndc_max = vec2<f32>(1.0);
    let z_near = center.z - radius;
    if (z_near > 1e-6) {
        let z_far = center.z + radius;
        let x_min = center.x - radius;
        let x_max = center.x + radius;
        let y_min = center.y - radius;
        let y_max = center.y + radius;
        let ratio_min = vec2<f32>(
            min(min(x_min / z_near, x_min / z_far), min(x_max / z_near, x_max / z_far)),
            min(min(y_min / z_near, y_min / z_far), min(y_max / z_near, y_max / z_far)),
        );
        let ratio_max = vec2<f32>(
            max(max(x_min / z_near, x_min / z_far), max(x_max / z_near, x_max / z_far)),
            max(max(y_min / z_near, y_min / z_far), max(y_max / z_near, y_max / z_far)),
        );
        let tan_half = tan(0.5 * g_camera.fov);
        ndc_min = ratio_min / tan_half + g_camera.principal - vec2<f32>(1e-5);
        ndc_max = ratio_max / tan_half + g_camera.principal + vec2<f32>(1e-5);
    }
    if (ndc_max.x < -1.0 || ndc_min.x > 1.0 ||
        ndc_max.y < -1.0 || ndc_min.y > 1.0) {
        return ProjectedTileBounds(vec2<u32>(0u), vec2<u32>(0u), 0u);
    }

    let image_size = vec2<f32>(
        f32(g_params.image_width),
        f32(g_params.image_height),
    );
    let last_pixel = image_size - vec2<f32>(1.0);
    let pixel_min = vec2<u32>(floor(min(
        0.5 * (clamp(ndc_min, vec2<f32>(-1.0), vec2<f32>(1.0)) + vec2<f32>(1.0)) * image_size,
        last_pixel,
    )));
    let pixel_max = vec2<u32>(floor(min(
        0.5 * (clamp(ndc_max, vec2<f32>(-1.0), vec2<f32>(1.0)) + vec2<f32>(1.0)) * image_size,
        last_pixel,
    )));
    return ProjectedTileBounds(pixel_min / 16u, pixel_max / 16u, 1u);
}

// Project every sphere once in parallel. The following tile-centric pass only
// needs integer bounds tests rather than repeating camera transforms and
// perspective divisions for every tile.
@compute @workgroup_size(64)
fn project_powerfoam_candidates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= g_params.num_points) {
        return;
    }
    let bounds = projected_tile_bounds(g_points[cell]);
    if (bounds.valid == 0u) {
        g_projected_bounds[cell] = vec4<u32>(0xffffffffu);
    } else {
        g_projected_bounds[cell] = vec4<u32>(bounds.min_tile, bounds.max_tile);
    }
}

// One workgroup owns one conservative 16x16 tile and scans the projected site
// bounds in parallel. The counter stays in workgroup memory, avoiding
// device-scope atomics while lanes write uniquely allocated storage slots.
// Tile rows are bounded; their counters retain true occupancy, so gather can
// detect overflow and fall back to the exact exhaustive scan.
@compute @workgroup_size(64)
fn bin_powerfoam_candidates(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tile_count = g_params.tile_width * g_params.tile_height;
    let tile = workgroup_id.x;
    if (tile >= tile_count) {
        return;
    }
    if (local_id.x == 0u) {
        atomicStore(&w_tile_count, 0u);
    }
    workgroupBarrier();

    let tile_xy = vec2<u32>(
        tile % g_params.tile_width,
        tile / g_params.tile_width,
    );
    for (var cell = local_id.x; cell < g_params.num_points; cell += 64u) {
        let bounds = g_projected_bounds[cell];
        if (bounds.x != 0xffffffffu &&
            all(tile_xy >= bounds.xy) &&
            all(tile_xy <= bounds.zw)) {
            let slot = atomicAdd(&w_tile_count, 1u);
            if (slot < g_params.tile_capacity) {
                g_tile_candidates[tile * g_params.tile_capacity + slot] = cell;
            }
        }
    }
    workgroupBarrier();

    if (local_id.x == 0u) {
        g_tile_counts[tile] = atomicLoad(&w_tile_count);
    }
}

// One workgroup owns one sampled ray. Its lanes scan disjoint point ranges
// and atomically append the support spheres intersected by that ray. A bounded
// projected tile row replaces the exhaustive scan when available; overflowing
// rows take the exhaustive path. Candidate order is intentionally irrelevant:
// `record_powerfoam_splats` selects the next interval by exact clipped entry
// depth with an index tie-break. 256 lanes is guaranteed across WebGPU
// adapters and minimizes the measured exhaustive production scan.
@compute @workgroup_size(256)
fn gather_powerfoam_candidates(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let p_id = workgroup_id.x;
    if (p_id >= g_params.num_pixels) {
        return;
    }
    let output_pixel = g_params.pixel_offset + p_id;
    let row_start = output_pixel * g_params.max_steps;
    for (var step = local_id.x; step < g_params.max_steps; step += 256u) {
        let slot = row_start + step;
        g_cells_out[slot] = 0u;
        g_next_cells_out[slot] = 0u;
        g_mask_out[slot] = 0.0;
        if (g_params.jacobian_mode == 1u) {
            g_previous_cells_out[slot] = 0u;
        }
    }
    if (local_id.x == 0u) {
        atomicStore(&w_candidate_count, 0u);
    }
    workgroupBarrier();

    let pixel_idx = g_pixel_indices[output_pixel];
    let ray_dir = ray_dir_for_pixel(pixel_idx);
    let ray_origin = g_camera.position;
    var use_tile = false;
    var tile = 0u;
    var scan_count = g_params.num_points;
    if (g_params.tile_capacity != 0u) {
        let ix = pixel_idx % g_params.image_width;
        let iy = pixel_idx / g_params.image_width;
        tile = (iy / 16u) * g_params.tile_width + ix / 16u;
        let tile_count = g_tile_counts[tile];
        if (tile_count <= g_params.tile_capacity) {
            use_tile = true;
            scan_count = tile_count;
        }
    }
    for (var scan_index = local_id.x; scan_index < scan_count; scan_index += 256u) {
        var cell = scan_index;
        if (use_tile) {
            cell = g_tile_candidates[tile * g_params.tile_capacity + scan_index];
        }
        let sphere_data = g_points[cell];
        if (inside_camera_exclusion(sphere_data.xyz - ray_origin, sphere_data.w)) {
            continue;
        }
        let sphere = sphere_intersections(ray_origin, ray_dir, sphere_data);
        if (sphere.valid == 0u || sphere.far_t <= 0.0 || sphere.near_t >= g_params.depth) {
            continue;
        }
        let slot = atomicAdd(&w_candidate_count, 1u);
        if (slot < g_params.candidate_capacity) {
            let candidate_slot = output_pixel * g_params.candidate_capacity + slot;
            g_candidates[candidate_slot] = cell;
            g_candidate_depths[candidate_slot] = sphere.root;
        }
    }
    workgroupBarrier();
    if (local_id.x == 0u) {
        g_candidate_counts[output_pixel] = atomicLoad(&w_candidate_count);
    }
}

fn candidate_after(left: u32, right: u32) -> bool {
    let left_depth = g_candidate_depths[left];
    let right_depth = g_candidate_depths[right];
    let left_cell = g_candidates[left];
    let right_cell = g_candidates[right];
    return left_depth > right_depth ||
        (left_depth == right_depth && left_cell > right_cell);
}

fn swap_candidates(left: u32, right: u32) {
    let left_cell = g_candidates[left];
    let left_depth = g_candidate_depths[left];
    let left_faces = g_candidate_faces[left];
    let left_neighbors = g_candidate_neighbors[left];
    let right_cell = g_candidates[right];
    let right_depth = g_candidate_depths[right];
    let right_faces = g_candidate_faces[right];
    let right_neighbors = g_candidate_neighbors[right];
    g_candidates[left] = right_cell;
    g_candidate_depths[left] = right_depth;
    g_candidate_faces[left] = right_faces;
    g_candidate_neighbors[left] = right_neighbors;
    g_candidates[right] = left_cell;
    g_candidate_depths[right] = left_depth;
    g_candidate_faces[right] = left_faces;
    g_candidate_neighbors[right] = left_neighbors;
}

fn sift_candidate_heap(begin: u32, root: u32, count: u32) {
    var current = root;
    loop {
        let left = 2u * current + 1u;
        if (left >= count) {
            break;
        }
        var child = left;
        let right = left + 1u;
        if (right < count && candidate_after(begin + right, begin + left)) {
            child = right;
        }
        if (!candidate_after(begin + child, begin + current)) {
            break;
        }
        swap_candidates(begin + current, begin + child);
        current = child;
    }
}

fn sort_candidate_row(begin: u32, count: u32) {
    var root = count / 2u;
    while (root > 0u) {
        root -= 1u;
        sift_candidate_heap(begin, root, count);
    }
    var end = count;
    while (end > 1u) {
        end -= 1u;
        swap_candidates(begin, begin + end);
        sift_candidate_heap(begin, 0u, end);
    }
}

fn emit_powerfoam_splats(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    output_pixel: u32,
    candidate_begin: u32,
    valid_count: u32,
) {
    let row_start = output_pixel * g_params.max_steps;
    sort_candidate_row(candidate_begin, valid_count);

    var candidate_index = 0u;
    var output_step = 0u;
    while (candidate_index < valid_count && output_step < g_params.max_steps) {
        let candidate_slot = candidate_begin + candidate_index;
        let cell = g_candidates[candidate_slot];
        let faces = g_candidate_faces[candidate_slot];
        let neighbors = g_candidate_neighbors[candidate_slot];
        candidate_index += 1u;
        let differential = interval_differential(
            ray_origin,
            ray_dir,
            neighbors.x,
            cell,
            neighbors.y,
            faces.x,
            faces.y,
        );
        if (differential.valid == 0u) {
            continue;
        }

        let output_slot = row_start + output_step;
        g_cells_out[output_slot] = cell;
        g_next_cells_out[output_slot] = neighbors.y;
        g_dts_out[output_slot] = differential.dt;
        g_mask_out[output_slot] = 1.0;
        if (has_surface_detail()) {
            g_surface_queries_out[output_slot] = differential.surface_query;
            if (g_params.jacobian_mode == 1u) {
                g_surface_query_grad_previous_out[output_slot] =
                    differential.surface_query_d_previous;
                g_surface_query_grad_current_out[output_slot] =
                    differential.surface_query_d_current;
            }
        }
        if (g_params.jacobian_mode != 0u) {
            var reference_tangent = 0.0;
            if (g_params.jacobian_mode == 1u) {
                g_previous_cells_out[output_slot] = neighbors.x;
                let previous_geometry = vec4<f32>(
                    g_points[neighbors.x].xyz - ray_origin,
                    g_points[neighbors.x].w,
                );
                let current_geometry = vec4<f32>(
                    g_points[cell].xyz - ray_origin,
                    g_points[cell].w,
                );
                let next_geometry = vec4<f32>(
                    g_points[neighbors.y].xyz - ray_origin,
                    g_points[neighbors.y].w,
                );
                reference_tangent =
                    dot(differential.dt_d_previous, previous_geometry) +
                    dot(differential.dt_d_current, current_geometry) +
                    dot(differential.dt_d_next, next_geometry);
                g_dt_grad_previous_out[output_slot] = differential.dt_d_previous;
                g_dt_grad_current_out[output_slot] = differential.dt_d_current;
                g_dt_grad_next_out[output_slot] = differential.dt_d_next;
            }
            if ((g_params.oriented & 1u) != 0u) {
                let surface = g_surface_normals[cell];
                reference_tangent += dot(
                    differential.dt_d_surface_normal.xyz,
                    surface.xyz,
                ) + differential.dt_d_surface_normal.w * differential.surface_offset;
                g_dt_grad_surface_normal_out[output_slot] =
                    differential.dt_d_surface_normal;
            }
            g_dt_reference_tangents_out[output_slot] = reference_tangent;
        }
        output_step += 1u;
    }

    let truncated = output_step == g_params.max_steps && candidate_index < valid_count;
    g_path_status_out[output_pixel] = output_step | select(0u, 0x80000000u, truncated);
}

// Independently clipped PowerFoam cells are disjoint along a ray. Select them
// in front-to-back order and write the same fixed-size path/Jacobian streams
// consumed by the differentiable integration graph.
@compute @workgroup_size(64)
fn record_powerfoam_splats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_id = gid.x;
    if (p_id >= g_params.num_pixels) {
        return;
    }

    let output_pixel = g_params.pixel_offset + p_id;
    let pixel_idx = g_pixel_indices[output_pixel];
    let ray_dir = ray_dir_for_pixel(pixel_idx);
    let ray_origin = g_camera.position;
    let candidate_begin = output_pixel * g_params.candidate_capacity;
    let candidate_count = min(g_candidate_counts[output_pixel], g_params.candidate_capacity);
    let row_start = output_pixel * g_params.max_steps;

    // Clip each candidate once and compact the valid intervals. A heap sort
    // then establishes the same depth/index order as the old repeated minimum
    // scan, without rescanning the complete candidate row for every segment.
    var valid_count = 0u;
    for (var slot = 0u; slot < candidate_count; slot += 1u) {
        let source_slot = candidate_begin + slot;
        let cell = g_candidates[source_slot];
        let sphere_root = g_candidate_depths[source_slot];
        let interval = power_interval(ray_origin, ray_dir, cell, sphere_root);
        if (interval.valid == 0u) {
            continue;
        }
        let compact_slot = candidate_begin + valid_count;
        g_candidates[compact_slot] = cell;
        g_candidate_depths[compact_slot] = interval.effective_near;
        g_candidate_faces[compact_slot] = vec2<f32>(interval.face_near, interval.face_far);
        g_candidate_neighbors[compact_slot] = vec2<u32>(interval.previous, interval.next);
        valid_count += 1u;
    }
    sort_candidate_row(candidate_begin, valid_count);

    var candidate_index = 0u;
    var output_step = 0u;
    while (candidate_index < valid_count && output_step < g_params.max_steps) {
        let candidate_slot = candidate_begin + candidate_index;
        let cell = g_candidates[candidate_slot];
        let faces = g_candidate_faces[candidate_slot];
        let neighbors = g_candidate_neighbors[candidate_slot];
        candidate_index += 1u;
        let differential = interval_differential(
            ray_origin,
            ray_dir,
            neighbors.x,
            cell,
            neighbors.y,
            faces.x,
            faces.y,
        );
        if (differential.valid == 0u) {
            continue;
        }

        let output_slot = row_start + output_step;
        g_cells_out[output_slot] = cell;
        g_next_cells_out[output_slot] = neighbors.y;
        g_dts_out[output_slot] = differential.dt;
        g_mask_out[output_slot] = 1.0;
        if (has_surface_detail()) {
            g_surface_queries_out[output_slot] = differential.surface_query;
            if (g_params.jacobian_mode == 1u) {
                g_surface_query_grad_previous_out[output_slot] =
                    differential.surface_query_d_previous;
                g_surface_query_grad_current_out[output_slot] =
                    differential.surface_query_d_current;
            }
        }
        if (g_params.jacobian_mode != 0u) {
            var reference_tangent = 0.0;
            if (g_params.jacobian_mode == 1u) {
                g_previous_cells_out[output_slot] = neighbors.x;
                let previous_geometry = vec4<f32>(
                    g_points[neighbors.x].xyz - ray_origin,
                    g_points[neighbors.x].w,
                );
                let current_geometry = vec4<f32>(
                    g_points[cell].xyz - ray_origin,
                    g_points[cell].w,
                );
                let next_geometry = vec4<f32>(
                    g_points[neighbors.y].xyz - ray_origin,
                    g_points[neighbors.y].w,
                );
                reference_tangent =
                    dot(differential.dt_d_previous, previous_geometry) +
                    dot(differential.dt_d_current, current_geometry) +
                    dot(differential.dt_d_next, next_geometry);
                g_dt_grad_previous_out[output_slot] = differential.dt_d_previous;
                g_dt_grad_current_out[output_slot] = differential.dt_d_current;
                g_dt_grad_next_out[output_slot] = differential.dt_d_next;
            }
            if ((g_params.oriented & 1u) != 0u) {
                let surface = g_surface_normals[cell];
                reference_tangent += dot(
                    differential.dt_d_surface_normal.xyz,
                    surface.xyz,
                ) + differential.dt_d_surface_normal.w * differential.surface_offset;
                g_dt_grad_surface_normal_out[output_slot] =
                    differential.dt_d_surface_normal;
            }
            g_dt_reference_tangents_out[output_slot] = reference_tangent;
        }
        output_step += 1u;
    }

    let truncated = output_step == g_params.max_steps && candidate_index < valid_count;
    g_path_status_out[output_pixel] = output_step | select(0u, 0x80000000u, truncated);
}

// Dense Cech graphs instead use all 64 lanes to clip one ray's candidates.
// Each chunk stays in workgroup memory until lane zero compacts it into the
// candidate row, avoiding a device-scope storage barrier. Lane zero then
// sorts and emits the deterministic path.
@compute @workgroup_size(64)
fn record_powerfoam_splats_parallel(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let groups_y = 1u + (g_params.num_pixels - 1u) / 65535u;
    let groups_x = 1u + (g_params.num_pixels - 1u) / groups_y;
    let p_id = workgroup_id.y * groups_x + workgroup_id.x;
    if (p_id >= g_params.num_pixels) {
        return;
    }

    let output_pixel = g_params.pixel_offset + p_id;
    let pixel_idx = g_pixel_indices[output_pixel];
    let ray_dir = ray_dir_for_pixel(pixel_idx);
    let ray_origin = g_camera.position;
    let candidate_begin = output_pixel * g_params.candidate_capacity;
    let candidate_count = min(g_candidate_counts[output_pixel], g_params.candidate_capacity);

    var valid_count = 0u;
    for (var chunk_begin = 0u; chunk_begin < candidate_count; chunk_begin += 64u) {
        let source_slot = chunk_begin + local_id.x;
        w_parallel_valid[local_id.x] = 0u;
        if (source_slot < candidate_count) {
            let candidate_slot = candidate_begin + source_slot;
            let cell = g_candidates[candidate_slot];
            let sphere_root = g_candidate_depths[candidate_slot];
            let interval = power_interval(ray_origin, ray_dir, cell, sphere_root);
            if (interval.valid != 0u) {
                w_parallel_cells[local_id.x] = cell;
                w_parallel_depths[local_id.x] = interval.effective_near;
                w_parallel_faces[local_id.x] =
                    vec2<f32>(interval.face_near, interval.face_far);
                w_parallel_neighbors[local_id.x] =
                    vec2<u32>(interval.previous, interval.next);
                w_parallel_valid[local_id.x] = 1u;
            }
        }
        workgroupBarrier();

        if (local_id.x == 0u) {
            let chunk_count = min(64u, candidate_count - chunk_begin);
            for (var lane = 0u; lane < chunk_count; lane += 1u) {
                if (w_parallel_valid[lane] == 0u) {
                    continue;
                }
                let destination = candidate_begin + valid_count;
                g_candidates[destination] = w_parallel_cells[lane];
                g_candidate_depths[destination] = w_parallel_depths[lane];
                g_candidate_faces[destination] = w_parallel_faces[lane];
                g_candidate_neighbors[destination] = w_parallel_neighbors[lane];
                valid_count += 1u;
            }
        }
        workgroupBarrier();
    }

    if (local_id.x != 0u) {
        return;
    }
    emit_powerfoam_splats(ray_origin, ray_dir, output_pixel, candidate_begin, valid_count);
}

@compute @workgroup_size(64)
fn record_paths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_id = gid.x;
    if (p_id >= g_params.num_pixels) {
        return;
    }

    let output_pixel = g_params.pixel_offset + p_id;
    let pixel_idx = g_pixel_indices[output_pixel];
    let ray_dir = ray_dir_for_pixel(pixel_idx);
    let ray_origin = g_camera.position;

    var t0: f32 = 0.0;
    var current: u32 = g_params.start_point;
    var previous: u32 = current;
    var current_pos = g_points[current].xyz;
    var current_radius = g_points[current].w;

    let row_start = output_pixel * g_params.max_steps;
    var output_step: u32 = 0u;
    var truncated = false;

    // The CPU `record_path` walks until t1 >= depth, no more faces,
    // or it has emitted `max_steps` entries. Mirror that here.
    for (var step: u32 = 0u; step < g_params.max_steps; step += 1u) {
        let begin = g_adjacency_offsets[current];
        let end = g_adjacency_offsets[current + 1u];

        var t1: f32 = g_params.depth;
        var next_face: u32 = 0xffffffffu;
        let r_i_sq = current_radius * current_radius;

        for (var j = begin; j < end; j += 1u) {
            let next_idx = g_adjacency[j];
            let next_pos = g_points[next_idx].xyz;
            let r_j = g_points[next_idx].w;
            let offset_vec = next_pos - current_pos;

            // Radical plane between power spheres (degenerates to the
            // Voronoi bisector when both radii are zero).
            let dsq = max(dot(offset_vec, offset_vec), 1e-20);
            let shift = 0.5 + 0.5 * (r_i_sq - r_j * r_j) / dsq;
            let face_origin = current_pos + shift * offset_vec;
            let face_normal = offset_vec;

            let dp = dot(face_normal, ray_dir);
            if (dp > 0.0) {
                let t = dot(face_origin - ray_origin, face_normal) / dp;
                if (t > t0 && t < t1) {
                    t1 = t;
                    next_face = j;
                }
            }
        }

        if (t1 <= t0) {
            break;
        }

        var next_idx = current;
        if (next_face != 0xffffffffu) {
            next_idx = g_adjacency[next_face];
        }
        let interval = interval_differential(
            ray_origin, ray_dir, previous, current, next_idx, t0, t1,
        );
        if (interval.valid != 0u) {
            g_cells_out[row_start + output_step] = current;
            g_next_cells_out[row_start + output_step] = next_idx;
            g_dts_out[row_start + output_step] = interval.dt;
            g_mask_out[row_start + output_step] = 1.0;
            if (has_surface_detail()) {
                g_surface_queries_out[row_start + output_step] = interval.surface_query;
                if (g_params.jacobian_mode == 1u) {
                    g_surface_query_grad_previous_out[row_start + output_step] =
                        interval.surface_query_d_previous;
                    g_surface_query_grad_current_out[row_start + output_step] =
                        interval.surface_query_d_current;
                }
            }
            if (g_params.power_foam != 0u && g_params.jacobian_mode != 0u) {
                var reference_tangent = 0.0;
                if (g_params.jacobian_mode == 1u) {
                    g_previous_cells_out[row_start + output_step] = previous;
                    let previous_geometry = vec4<f32>(
                        g_points[previous].xyz - ray_origin,
                        g_points[previous].w,
                    );
                    let current_geometry = vec4<f32>(
                        g_points[current].xyz - ray_origin,
                        g_points[current].w,
                    );
                    let next_geometry = vec4<f32>(
                        g_points[next_idx].xyz - ray_origin,
                        g_points[next_idx].w,
                    );
                    reference_tangent =
                        dot(interval.dt_d_previous, previous_geometry) +
                        dot(interval.dt_d_current, current_geometry) +
                        dot(interval.dt_d_next, next_geometry);
                    g_dt_grad_previous_out[row_start + output_step] =
                        interval.dt_d_previous;
                    g_dt_grad_current_out[row_start + output_step] =
                        interval.dt_d_current;
                    g_dt_grad_next_out[row_start + output_step] = interval.dt_d_next;
                }
                if ((g_params.oriented & 1u) != 0u) {
                    let surface = g_surface_normals[current];
                    reference_tangent += dot(
                        interval.dt_d_surface_normal.xyz,
                        surface.xyz,
                    ) + interval.dt_d_surface_normal.w * interval.surface_offset;
                    g_dt_grad_surface_normal_out[row_start + output_step] =
                        interval.dt_d_surface_normal;
                }
                g_dt_reference_tangents_out[row_start + output_step] = reference_tangent;
            }
            output_step += 1u;
        }

        if (next_face == 0xffffffffu) {
            break;
        }

        if (step + 1u == g_params.max_steps) {
            truncated = true;
        }

        t0 = t1;
        previous = current;
        current = next_idx;
        current_pos = g_points[next_idx].xyz;
        current_radius = g_points[next_idx].w;
    }
    g_path_status_out[output_pixel] = output_step | select(0u, 0x80000000u, truncated);
}
