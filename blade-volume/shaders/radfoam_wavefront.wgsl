//// RadFoam wavefront traversal pass (queue-driven)
////
//// This shader continues RadFoam Voronoi traversal starting from a compacted ray queue.
//// It is intended to be used after:
////  - `radfoam_init_screen.wgsl` (dense state + alive flags)
////  - `wavefront/compact_alive.wgsl` + `wavefront/scan_block_sums.wgsl` (queue + queue_count)
////
//// Responsibilities:
////  - For each queued ray, resume traversal from `RayState` and integrate radiance/opacity.
////  - Write final HDR color into `g_out` at the pixel coordinate stored in the RayState.
////
//// Notes:
////  - This is a first implementation: one thread = one ray, full continuation until termination.
////  - Uses the same camera model and SH basis constants as the legacy tracer.
////  - RayState layout MUST match the producer (`radfoam_init_screen.wgsl`) and compaction shader.
////
//// Bindings are designed to be driven by `blade_graphics` ShaderData.
////
//// Output:
////  - `g_out`: storage texture (rgba16float), written once per ray/pixel.
////
//// Debug:
////  - If `g_params.debug_mode == 1`, output a heatmap based on `cells_visited`.
////
//// WGSL limitations:
////  - Avoid `let x = if (...) { ... } else { ... }` because naga rejects it in some configs.

// #include "common.wgsl"
// #include "sh_eval.wgsl"

// ----------------------------------------------------------------------------
// Camera / Params (must match Rust layouts used in the viewer)
// ----------------------------------------------------------------------------

struct Params {
    // must match scene
    sh_degree: u32,

    // tracing controls
    init_steps: u32,        // unused here (set to 0 for this pass)
    weight_threshold: f32,  // stop when transmittance <= threshold
    max_steps: u32,         // maximum cell transitions
    start_point: u32,       // unused here (RayState.current is authoritative)

    // debug mode
    debug_mode: u32,        // 0 = off, 1 = cell density visualization

    // padding
    pad0: u32,
    pad1: u32,
};

struct WavefrontPhaseParams {
    // Number of traversal steps to execute per phase.
    phase_steps: u32,
    // Padding
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

var<uniform> g_camera: Camera;
var<uniform> g_params: Params;
var<uniform> g_phase: WavefrontPhaseParams;

// ----------------------------------------------------------------------------
// Scene buffers (match legacy)
// ----------------------------------------------------------------------------

// Points packed as vec4; xyz = position, w unused.
var<storage, read> g_points: array<vec4<f32>>;

// Packed attributes: f32 array with row size = attr_dim = 1 + 3*(1+deg)^2.
// Layout per point row:
//   coeffs: sh_dim = 3 * sh_components
//   density: last scalar
var<storage, read> g_attributes: array<f32>;

// CSR adjacency.
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

// ----------------------------------------------------------------------------
// Queue input (from compaction)
// ----------------------------------------------------------------------------

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

    // Direction (xyz used, w unused)
    ray_dir: vec4<f32>,

    // Accumulated RGB (xyz used, w unused)
    accum_rgb: vec4<f32>,

    // Stats/debug
    cells_visited: u32,
    terminated: u32, // 0 = alive, 1 = terminated
    pad: vec2<u32>,
};

var<storage, read> g_queue: array<RayState>;
var<storage, read> g_queue_count: atomic<u32>;
var<storage, read_write> g_next_queue: array<RayState>;
var<storage, read_write> g_next_queue_count: atomic<u32>;

// Output HDR image.
var g_out: texture_storage_2d<rgba16float, write>;

// Workgroup shared state for early-exit decision.
var<workgroup> wg_exit: u32;
var<workgroup> wg_alive: atomic<u32>;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

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

// One Voronoi traversal step: choose next cell boundary intersection and neighbor.
fn step_voronoi(ray_origin: vec3<f32>, ray_dir: vec3<f32>, current: u32, t0: f32) -> vec3<f32> {
    // Pack (next_t0, next_current as f32, advanced flag)
    let current_pos = g_points[current].xyz;

    let begin = g_adjacency_offsets[current];
    let end = g_adjacency_offsets[current + 1u];
    let num_faces = end - begin;

    var t1: f32 = 3.402823466e+38;
    var next_face: u32 = 0xffffffffu;

    for (var j = 0u; j < num_faces; j += 1u) {
        let next_idx = g_adjacency[begin + j];
        let next_pos = g_points[next_idx].xyz;
        let offset = next_pos - current_pos;

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

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn wavefront_phase_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let lid = lid3.x;
    let qi = gid.x;
    let queue_count = atomicLoad(&g_queue_count);
    if (qi >= queue_count) {
        return;
    }

    let dims_i = textureDimensions(g_out);
    let w = u32(dims_i.x);
    let h = u32(dims_i.y);

    var state = g_queue[qi];

    // Bounds check pixel coords (defensive)
    if (state.pixel_x >= w || state.pixel_y >= h) {
        return;
    }

    // If already terminated, just write what we have (likely black).
    if (state.terminated != 0u) {
        let out_rgb = state.accum_rgb.xyz;
        let out_a = 1.0 - state.transmittance;
        textureStore(g_out, vec2<i32>(i32(state.pixel_x), i32(state.pixel_y)), vec4<f32>(out_rgb, out_a));
        return;
    }

    let ray_origin = g_camera.position;
    let ray_dir = normalize(state.ray_dir.xyz);

    let phase_steps = g_phase.phase_steps;
    var alive = true;
    let exit_threshold = 128u; // half of workgroup size (256)

    // Continue traversal for a bounded number of steps.
    for (var step = 0u; step < phase_steps; step += 1u) {
        if (lid == 0u) {
            atomicStore(&wg_alive, 0u);
            wg_exit = 0u;
        }
        workgroupBarrier();

        if (alive) {
            if (!(state.t0 < g_camera.depth && state.steps < g_params.max_steps && state.transmittance > g_params.weight_threshold)) {
                alive = false;
            } else {
                let packed = step_voronoi(ray_origin, ray_dir, state.current, state.t0);
                let advanced = packed.z > 0.5;
                if (!advanced) {
                    alive = false;
                } else {
                    let next_t0 = packed.x;
                    let next_idx = u32(packed.y);

                    // Integrate segment [t0, next_t0] in current cell.
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

                    state.steps += 1u;
                    state.t0 = max(state.t0, next_t0);
                    state.current = next_idx;
                }
            }
        }

        if (alive) {
            atomicAdd(&wg_alive, 1u);
        }
        workgroupBarrier();

        if (lid == 0u) {
            let alive_count = atomicLoad(&wg_alive);
            if (alive_count <= exit_threshold) {
                wg_exit = 1u;
            }
        }
        workgroupBarrier();

        if (wg_exit != 0u) {
            break;
        }
    }

    if (alive && state.t0 < g_camera.depth && state.steps < g_params.max_steps && state.transmittance > g_params.weight_threshold) {
        let dst = atomicAdd(&g_next_queue_count, 1u);
        g_next_queue[dst] = state;
        return;
    }

    // Debug mode: visualize traversal effort
    if (g_params.debug_mode == 1u) {
        // Normalize cell count (0-100 -> 0-1) as in legacy
        let density = f32(state.cells_visited) / 100.0;
        let debug_color = heatmap_color(density);
        textureStore(
            g_out,
            vec2<i32>(i32(state.pixel_x), i32(state.pixel_y)),
            vec4<f32>(debug_color, 1.0)
        );
        return;
    }

    let out_rgb = state.accum_rgb.xyz;
    let out_a = 1.0 - state.transmittance;

    textureStore(
        g_out,
        vec2<i32>(i32(state.pixel_x), i32(state.pixel_y)),
        vec4<f32>(out_rgb, out_a)
    );
}
