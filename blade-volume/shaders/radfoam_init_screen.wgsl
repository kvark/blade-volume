// RadFoam init-screen pass (wavefront preparation)
//
// This compute shader runs a *fixed* number of Voronoi traversal steps per pixel and writes out:
// - `g_alive[pixel]`: 0/1 for whether the ray is still alive after init_steps
// - `g_state[pixel]`: a dense RayState array (one per pixel), sufficient to resume traversal later
//
// Design goals:
// - Keep this pass screen-ordered (coherent).
// - Match the legacy camera model and traversal step logic from `radfoam_trace_legacy.wgsl`.
// - Do not integrate radiance here (that happens in later wavefront passes).
//
// Important:
// - Dimensions are derived from `g_out` (a storage texture). The actual color output is unused;
//   we write a dummy value so the binding is "used" and dimensions are available.
//
// Bindings are designed to be driven by `blade_graphics` ShaderData.

struct Camera {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>, // quaternion (x,y,z,w)
    fov: vec2<f32>,         // (fov_x, fov_y) where local_dir = (ndc * tan(0.5*fov), 1)
    pad: vec2<u32>,
};

struct Params {
    // must match scene
    sh_degree: u32,
    // tracing controls
    init_steps: u32,
    weight_threshold: f32, // stop when transmittance <= threshold
    max_steps: u32,        // maximum cell transitions
    start_point: u32,      // starting cell index for all rays (MVP)
    // debug mode (ignored here, but kept for layout compatibility)
    debug_mode: u32,
    // padding
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
var<uniform> g_camera: Camera;
var<uniform> g_params: Params;

// Storage buffers (match legacy RadFoam traversal inputs).
// Points are packed as array<vec4<f32>> for alignment simplicity: xyz = position, w unused.
var<storage, read> g_points: array<vec4<f32>>;
// Packed attributes are unused in this pass, but kept for potential future heuristics.
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

// Outputs:
// One u32 per pixel (0/1).
var<storage, read_write> g_alive: array<u32>;

// Dense ray state: one RayState per pixel.
struct RayState {
    // Pixel coordinates
    pixel_x: u32,
    pixel_y: u32,

    // Traversal control/state
    steps: u32,
    current: u32,

    // Ray marching state
    t0: f32,
    transmittance: f32,

    // Direction as vec4 for alignment (xyz used, w unused)
    ray_dir: vec4<f32>,

    // Accumulation placeholders for later passes (xyz used, w unused)
    accum_rgb: vec4<f32>,

    // Stats/debug
    cells_visited: u32,
    terminated: u32, // 0 = alive, 1 = terminated
    pad: vec2<u32>,
};

var<storage, read_write> g_state: array<RayState>;

// Dummy output texture: used to query dimensions; color output is unused.
var g_out: texture_storage_2d<rgba16float, write>;

// ---- Quaternion helpers (matches legacy) ----

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn pixel_index(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

// One Voronoi cell transition step (ported from legacy trace loop).
// Returns:
// - next_t0
// - next_current
// - advanced flag (false means "no next face" => terminate)
fn step_voronoi(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    current: u32,
    t0: f32,
) -> vec3<f32> {
    // Pack results into vec3:
    // x = next_t0
    // y = f32(next_current)
    // z = 1.0 if advanced else 0.0
    let current_pos = g_points[current].xyz;

    let begin = g_adjacency_offsets[current];
    let end = g_adjacency_offsets[current + 1u];
    let num_faces = end - begin;

    // WGSL/naga doesn't allow abstract-float inf; use large finite.
    var t1: f32 = 3.402823466e+38; // ~f32::MAX
    var next_face: u32 = 0xffffffffu;

    // Scan neighbors
    for (var j = 0u; j < num_faces; j += 1u) {
        let next_idx = g_adjacency[begin + j];
        let next_pos = g_points[next_idx].xyz;
        let offset = next_pos - current_pos;

        // Bisector plane
        let face_origin = current_pos + 0.5 * offset;
        let face_normal = offset;

        let dp = dot(face_normal, ray_dir);
        if (dp > 0.0) {
            let t = dot(face_origin - ray_origin, face_normal) / dp;
            if (t < t1) {
                t1 = t;
                next_face = j;
            }
        }
    }

    if (next_face == 0xffffffffu) {
        return vec3<f32>(t0, f32(current), 0.0);
    }

    let next_idx = g_adjacency[begin + next_face];
    let next_t0 = max(t0, t1);
    return vec3<f32>(next_t0, f32(next_idx), 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_i = textureDimensions(g_out);
    let w = u32(dims_i.x);
    let h = u32(dims_i.y);

    if (gid.x >= w || gid.y >= h) {
        return;
    }

    // NDC in [-1, 1]
    let px = (f32(gid.x) + 0.5) / f32(w);
    let py = (f32(gid.y) + 0.5) / f32(h);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);

    // Match the legacy camera model:
    // local_dir = (ndc * tan(0.5*fov), 1)
    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>(ndc * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));
    let ray_origin = g_camera.position;

    let idx = pixel_index(gid.x, gid.y, w);

    // Initialize state similar to legacy `trace_ray`.
    var state: RayState;
    state.pixel_x = gid.x;
    state.pixel_y = gid.y;
    state.steps = 0u;
    state.current = g_params.start_point;
    state.t0 = 0.0;
    state.transmittance = 1.0;
    state.ray_dir = vec4<f32>(ray_dir, 0.0);
    state.accum_rgb = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    state.cells_visited = 0u;
    state.terminated = 0u;
    state.pad = vec2<u32>(0u, 0u);

    // Run a bounded number of steps.
    let init_steps = g_params.init_steps;
    let max_init_steps = min(init_steps, g_params.max_steps);
    for (var i = 0u; i < max_init_steps; i += 1u) {
        // Termination checks (match legacy loop guards).
        if (state.t0 >= g_camera.depth) {
            state.terminated = 1u;
            break;
        }
        if (state.transmittance <= g_params.weight_threshold) {
            state.terminated = 1u;
            break;
        }
        if (state.steps >= g_params.max_steps) {
            state.terminated = 1u;
            break;
        }

        let packed = step_voronoi(ray_origin, ray_dir, state.current, state.t0);
        let advanced = packed.z > 0.5;
        if (!advanced) {
            state.terminated = 1u;
            break;
        }

        // Legacy increments "steps" per cell transition.
        state.steps += 1u;

        // Legacy increments "cells_visited" when t1 > t0 and integration happens.
        // In init pass we don't integrate; we still track potential segment transitions for stats.
        // Approximate: count every successful step as a visited cell.
        state.cells_visited += 1u;

        state.t0 = packed.x;
        state.current = u32(packed.y);
    }

    // Alive if not terminated and still within legacy loop conditions.
    // (Even if not terminated, the next pass will re-check loop conditions.)
    let alive = select(
        0u,
        1u,
        state.terminated == 0u
            && state.t0 < g_camera.depth
            && state.steps < g_params.max_steps
            && state.transmittance > g_params.weight_threshold,
    );

    g_alive[idx] = alive;
    g_state[idx] = state;

    // Dummy write so `g_out` is used (and thus `textureDimensions` is well-defined).
    // This output is not used for display.
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}
