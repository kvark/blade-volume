// GPU path-recording for differentiable RadFoam training.
//
// Traces one ray per output slot through the foam — same Voronoi /
// power-diagram traversal as `radfoam_trace.wgsl` — but instead of
// accumulating colour it writes a `(cell, next_cell, dt, mask)` tuple
// for every step along the path. The training pipeline reads those
// tuples into meganeura as `cell_indices` (u32), `next_cell_indices`
// (u32), `dt` (f32), and `mask` (f32) tensors, and the differentiable
// forward is built around them.
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
    /// Source image width (for converting pixel index → (ix, iy)).
    image_width: u32,
    image_height: u32,
    /// Saturating cap for `dt` to keep the sigmoid surrogate finite
    /// when the ray escapes through a giant cell. Matches the CPU
    /// path's `MAX_PATH_DT` clamp.
    max_path_dt: f32,
    /// Far plane of the camera frustum.
    depth: f32,
    _pad: u32,
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
var<storage, read_write> g_cells_out: array<u32>;
var<storage, read_write> g_next_cells_out: array<u32>;
var<storage, read_write> g_dts_out: array<f32>;
var<storage, read_write> g_mask_out: array<f32>;
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

@compute @workgroup_size(64)
fn record_paths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_id = gid.x;
    if (p_id >= g_params.num_pixels) {
        return;
    }

    let pixel_idx = g_pixel_indices[p_id];
    let ray_dir = ray_dir_for_pixel(pixel_idx);
    let ray_origin = g_camera.position;

    var t0: f32 = 0.0;
    var current: u32 = g_params.start_point;
    var current_pos = g_points[current].xyz;
    var current_radius = g_points[current].w;

    let row_start = p_id * g_params.max_steps;

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

        var dt = t1 - t0;
        if (dt > g_params.max_path_dt) {
            dt = g_params.max_path_dt;
        }
        var next_idx = current;
        if (next_face != 0xffffffffu) {
            next_idx = g_adjacency[next_face];
        }
        g_cells_out[row_start + step] = current;
        g_next_cells_out[row_start + step] = next_idx;
        g_dts_out[row_start + step] = dt;
        g_mask_out[row_start + step] = 1.0;

        if (next_face == 0xffffffffu) {
            break;
        }

        t0 = t1;
        current = next_idx;
        current_pos = g_points[next_idx].xyz;
        current_radius = g_points[next_idx].w;
    }
}
