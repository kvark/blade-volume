// GPU path-recording for differentiable RadFoam training.
//
// Traces one ray per output slot through the foam — same Voronoi /
// power-diagram traversal as `radfoam_trace.wgsl` — but instead of
// accumulating colour it writes a `(previous_cell, cell, next_cell, dt,
// mask)` tuple for every visible segment. Weighted paths additionally write
// the exact local derivative of `dt` with respect to each role's
// `(x, y, z, radius)` and the ray-relative reference tangent
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
// Pre-zero the output buffers before dispatch — the shader only writes
// the steps that were actually taken, leaving the trailing slots at the
// pre-dispatch value (which is canonically `mask = 0`).

// #include "common.wgsl" — required for `qrot`
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
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct Camera {
    position: vec3<f32>,
    _pad0: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    principal: vec2<f32>,
};

var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;
var<storage, read> g_pixel_indices: array<u32>;
var<storage, read_write> g_previous_cells_out: array<u32>;
var<storage, read_write> g_cells_out: array<u32>;
var<storage, read_write> g_next_cells_out: array<u32>;
var<storage, read_write> g_dts_out: array<f32>;
var<storage, read_write> g_mask_out: array<f32>;
var<storage, read_write> g_dt_reference_tangents_out: array<f32>;
var<storage, read_write> g_dt_grad_previous_out: array<vec4<f32>>;
var<storage, read_write> g_dt_grad_current_out: array<vec4<f32>>;
var<storage, read_write> g_dt_grad_next_out: array<vec4<f32>>;
var<uniform> g_camera: Camera;
var<uniform> g_params: RecordParams;

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
    near_jacobian: vec4<f32>,
    far_jacobian: vec4<f32>,
    valid: u32,
};

struct IntervalDifferential {
    dt: f32,
    dt_d_previous: vec4<f32>,
    dt_d_current: vec4<f32>,
    dt_d_next: vec4<f32>,
    valid: u32,
};

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
        return SphereIntersections(0.0, 0.0, vec4<f32>(0.0), vec4<f32>(0.0), 0u);
    }
    let root = sqrt(discriminant);
    let perpendicular = oc - b * ray_dir;
    let root_d_center = perpendicular / root;
    let root_d_radius = sphere.w / root;
    return SphereIntersections(
        -b - root,
        -b + root,
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
            select(0u, 1u, t1 > t0),
        );
    }

    let current = g_points[current_idx];
    let sphere = sphere_intersections(ray_origin, ray_dir, current);
    if (sphere.valid == 0u) {
        return IntervalDifferential(
            0.0, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), 0u,
        );
    }

    var start = t0;
    var end = t1;
    var start_from_sphere = false;
    var end_from_sphere = false;
    if (sphere.near_t > start) {
        start = sphere.near_t;
        start_from_sphere = true;
    }
    if (sphere.far_t < end) {
        end = sphere.far_t;
        end_from_sphere = true;
    }
    if (end <= start) {
        return IntervalDifferential(
            0.0, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), 0u,
        );
    }

    var dt_d_previous = vec4<f32>(0.0);
    var dt_d_current = vec4<f32>(0.0);
    var dt_d_next = vec4<f32>(0.0);
    if (start_from_sphere) {
        dt_d_current -= sphere.near_jacobian;
    } else if (previous_idx != current_idx) {
        let face = face_intersection_jacobians(
            ray_origin, ray_dir, current, g_points[previous_idx], t0,
        );
        dt_d_current -= face.current;
        dt_d_previous -= face.adjacent;
    }
    if (end_from_sphere) {
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
    }
    return IntervalDifferential(
        dt, dt_d_previous, dt_d_current, dt_d_next, 1u,
    );
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
            if (g_params.power_foam != 0u) {
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
                let reference_tangent =
                    dot(interval.dt_d_previous, previous_geometry) +
                    dot(interval.dt_d_current, current_geometry) +
                    dot(interval.dt_d_next, next_geometry);
                g_dt_reference_tangents_out[row_start + output_step] = reference_tangent;
                g_dt_grad_previous_out[row_start + output_step] = interval.dt_d_previous;
                g_dt_grad_current_out[row_start + output_step] = interval.dt_d_current;
                g_dt_grad_next_out[row_start + output_step] = interval.dt_d_next;
            }
            output_step += 1u;
        }

        if (next_face == 0xffffffffu) {
            break;
        }

        t0 = t1;
        previous = current;
        current = next_idx;
        current_pos = g_points[next_idx].xyz;
        current_radius = g_points[next_idx].w;
    }
}
