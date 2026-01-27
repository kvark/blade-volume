// RadFoam init-screen pass (wavefront preparation)
//
// This compute shader runs a *fixed* number of Voronoi traversal steps per pixel and writes out:
// - `g_alive[pixel]`: 0/1 for whether the ray is still alive after init_steps
// - `g_state[pixel]`: a dense RayState array (one per pixel), sufficient to resume traversal later
//
// Design goals:
// - Keep this pass screen-ordered (coherent).
// - Match the legacy camera model and traversal step logic from `radfoam_trace_legacy.wgsl`.
// - Integrate radiance + transmittance for the stepped segments so the combined output of
//   (init-screen + wavefront) matches the legacy tracer.
// - IMPORTANT: If a ray terminates during init, write its final HDR output here so every pixel
//   gets a value even if it is not queued for the wavefront pass.
//
// Important:
// - Dimensions are derived from `g_out` (a storage texture).
//
// Bindings are designed to be driven by `blade_graphics` ShaderData.

// #include "common.wgsl"
// #include "sh_eval.wgsl"

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
};

var<uniform> g_camera: Camera;
var<uniform> g_params: Params;

// Storage buffers (match legacy RadFoam traversal inputs).
// Points are packed as array<vec4<f32>> for alignment simplicity: xyz = position, w unused.
var<storage, read> g_points: array<vec4<f32>>;
// Packed attributes are used for density + SH evaluation.
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

    // Accumulated RGB (xyz used, w unused)
    accum_rgb: vec4<f32>,

    // Stats/debug
    cells_visited: u32,
    terminated: u32, // 0 = alive, 1 = terminated
    pad: vec2<u32>,
};

var<storage, read_write> g_state: array<RayState>;

// Output HDR image (also used to query dimensions).
var g_out: texture_storage_2d<rgba16float, write>;

fn pixel_index(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

// ---- SH evaluation (ported from legacy) ----

fn eval_sh_rgb(point_idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let deg = g_params.sh_degree;
    let comps = sh_component_count(deg);
    let clamped_comps = min(comps, MAX_SH_COMPONENTS);

    let SH = sh_basis_constants();
    let d2 = dir * dir;

    let sh_dim = 3u * clamped_comps;
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;

    // L0
    var color = vec3<f32>(
        SH[0] * g_attributes[base + 0u],
        SH[0] * g_attributes[base + 1u],
        SH[0] * g_attributes[base + 2u]
    );

    if (deg >= 1u && clamped_comps >= 4u) {
        let y = dir.y;
        let z = dir.z;
        let x = dir.x;

        color += vec3<f32>(
            SH[1] * g_attributes[base + 3u + 0u] * y,
            SH[1] * g_attributes[base + 3u + 1u] * y,
            SH[1] * g_attributes[base + 3u + 2u] * y
        );
        color += vec3<f32>(
            SH[2] * g_attributes[base + 6u + 0u] * z,
            SH[2] * g_attributes[base + 6u + 1u] * z,
            SH[2] * g_attributes[base + 6u + 2u] * z
        );
        color += vec3<f32>(
            SH[3] * g_attributes[base + 9u + 0u] * x,
            SH[3] * g_attributes[base + 9u + 1u] * x,
            SH[3] * g_attributes[base + 9u + 2u] * x
        );
    }

    if (deg >= 2u && clamped_comps >= 9u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        color += vec3<f32>(
            SH[4] * g_attributes[base + 12u + 0u] * x * y,
            SH[4] * g_attributes[base + 12u + 1u] * x * y,
            SH[4] * g_attributes[base + 12u + 2u] * x * y
        );
        color += vec3<f32>(
            SH[5] * g_attributes[base + 15u + 0u] * y * z,
            SH[5] * g_attributes[base + 15u + 1u] * y * z,
            SH[5] * g_attributes[base + 15u + 2u] * y * z
        );
        color += vec3<f32>(
            SH[6] * g_attributes[base + 18u + 0u] * (3.0 * zz - 1.0),
            SH[6] * g_attributes[base + 18u + 1u] * (3.0 * zz - 1.0),
            SH[6] * g_attributes[base + 18u + 2u] * (3.0 * zz - 1.0)
        );
        color += vec3<f32>(
            SH[7] * g_attributes[base + 21u + 0u] * x * z,
            SH[7] * g_attributes[base + 21u + 1u] * x * z,
            SH[7] * g_attributes[base + 21u + 2u] * x * z
        );
        color += vec3<f32>(
            SH[8] * g_attributes[base + 24u + 0u] * (xx - yy),
            SH[8] * g_attributes[base + 24u + 1u] * (xx - yy),
            SH[8] * g_attributes[base + 24u + 2u] * (xx - yy)
        );
    }

    if (deg >= 3u && clamped_comps >= 16u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        color += vec3<f32>(
            SH[9] * g_attributes[base + 27u + 0u] * y * (3.0 * xx - yy),
            SH[9] * g_attributes[base + 27u + 1u] * y * (3.0 * xx - yy),
            SH[9] * g_attributes[base + 27u + 2u] * y * (3.0 * xx - yy)
        );
        color += vec3<f32>(
            SH[10] * g_attributes[base + 30u + 0u] * x * y * z,
            SH[10] * g_attributes[base + 30u + 1u] * x * y * z,
            SH[10] * g_attributes[base + 30u + 2u] * x * y * z
        );
        color += vec3<f32>(
            SH[11] * g_attributes[base + 33u + 0u] * y * (5.0 * zz - 1.0),
            SH[11] * g_attributes[base + 33u + 1u] * y * (5.0 * zz - 1.0),
            SH[11] * g_attributes[base + 33u + 2u] * y * (5.0 * zz - 1.0)
        );
        color += vec3<f32>(
            SH[12] * g_attributes[base + 36u + 0u] * z * (5.0 * zz - 3.0),
            SH[12] * g_attributes[base + 36u + 1u] * z * (5.0 * zz - 3.0),
            SH[12] * g_attributes[base + 36u + 2u] * z * (5.0 * zz - 3.0)
        );
        color += vec3<f32>(
            SH[13] * g_attributes[base + 39u + 0u] * x * (5.0 * zz - 1.0),
            SH[13] * g_attributes[base + 39u + 1u] * x * (5.0 * zz - 1.0),
            SH[13] * g_attributes[base + 39u + 2u] * x * (5.0 * zz - 1.0)
        );
        color += vec3<f32>(
            SH[14] * g_attributes[base + 42u + 0u] * z * (xx - yy),
            SH[14] * g_attributes[base + 42u + 1u] * z * (xx - yy),
            SH[14] * g_attributes[base + 42u + 2u] * z * (xx - yy)
        );
        color += vec3<f32>(
            SH[15] * g_attributes[base + 45u + 0u] * x * (xx - 3.0 * yy),
            SH[15] * g_attributes[base + 45u + 1u] * x * (xx - 3.0 * yy),
            SH[15] * g_attributes[base + 45u + 2u] * x * (xx - 3.0 * yy)
        );
    }

    // Upstream adds 0.5 bias to color; keep similar behavior for visibility.
    return 0.5 + color;
}

fn load_density(point_idx: u32) -> f32 {
    let deg = g_params.sh_degree;
    let comps = min(sh_component_count(deg), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;
    return g_attributes[base + sh_dim];
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
    // y = f32(next_idx)
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

        let next_t0 = packed.x;
        let next_idx = u32(packed.y);

        // Integrate segment [t0, next_t0] in current cell (matches legacy).
        if (next_t0 > state.t0) {
            state.cells_visited += 1u;
            let s = load_density(state.current);
            if (s > 1e-6) {
                let dt = max(next_t0 - state.t0, 0.0);
                let alpha = 1.0 - exp(-s * dt);
                let wgt = state.transmittance * alpha;
                let rgb = eval_sh_rgb(state.current, ray_dir);
                state.accum_rgb = vec4<f32>(state.accum_rgb.xyz + wgt * rgb, state.accum_rgb.w);
                state.transmittance = state.transmittance * (1.0 - alpha);
            }
        }

        // Legacy increments "steps" per cell transition.
        state.steps += 1u;

        state.t0 = max(state.t0, next_t0);
        state.current = next_idx;
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

    // Write partial accumulated HDR output for ALL rays.
    //
    // Rationale:
    // - If a ray is (incorrectly) dropped from the queue after init, the pixel should still show
    //   the partial result instead of becoming black.
    // - If the ray is queued, the wavefront pass will overwrite this with the final result.
    let out_rgb = state.accum_rgb.xyz;
    let out_a = 1.0 - state.transmittance;
    textureStore(
        g_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(out_rgb, out_a),
    );
}
