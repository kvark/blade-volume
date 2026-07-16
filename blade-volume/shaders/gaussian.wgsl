// Gaussian splatting renderer using hardware ray tracing.
//
// Output is rendered directly to the screen via rasterization (fullscreen triangle).

enable wgpu_ray_query;

// #gaussian_query scalar g_gaussian_tlas

// #include "common.wgsl"
// #include "sh_eval.wgsl"

var<uniform> g_camera: Camera;

struct Parameters {
    min_opacity: f32,
    min_transmittance: f32,
    sh_degree: u32,
    debug_mode: u32,
    pad: vec4<u32>,
}
var<uniform> g_params: Parameters;

// TLAS for hardware RT (required name for gaussian_trace.wgsl)
var g_gaussian_tlas: acceleration_structure;

struct Gaussian {
    mean: vec3f,
    pad1: f32,
    rotation: vec4f,
    scale: vec3f,
    opacity: f32,
    harmonics: array<vec4f, MAX_SH_COMPONENTS>,
}

// Gaussian data buffer
var<storage> g_data: array<Gaussian>;

// Debug mode constants
const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_PARTICLE_DENSITY: u32 = 1u;

// ============================================================================
// Gaussian accessor functions (interface for gaussian_trace.wgsl)
// ============================================================================

fn gs_get_gaussian(idx: u32) -> Gaussian {
    return g_data[idx];
}

fn gs_get_sh_degree() -> u32 {
    return g_params.sh_degree;
}

fn gs_get_weight_threshold() -> f32 {
    return g_params.min_transmittance;
}

// #include "gaussian_trace.wgsl"

// ============================================================================
// Vertex shader (fullscreen triangle)
// ============================================================================

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) direction: vec3<f32>,
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo = VertexOutput();
    let tc = vec2f(vec2u(vi) & vec2u(1u, 2u)) * vec2f(1.0, 0.5);
    let ndc = 4.0 * tc - 1.0;
    let local_dir = vec3f((ndc - g_camera.principal) * tan(0.5 * g_camera.fov), 1.0);
    vo.clip_pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
    vo.direction = qrot(g_camera.orientation, local_dir);
    return vo;
}

// ============================================================================
// Fragment shader
// ============================================================================

const BACKGROUND: vec3f = vec3f(0.0);

@fragment
fn draw_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let ray_pos = g_camera.position;
    let ray_dir = normalize(vo.direction);

    var params: GaussianTraceParams;
    params.t_start = 0.0;
    params.t_end = g_camera.depth;
    params.query_t_start = params.t_start;
    params.query_t_end = params.t_end;

    let result = gaussian_trace(ray_pos, ray_dir, params);

    // Debug mode: particle density visualization
    if (g_params.debug_mode == DEBUG_MODE_PARTICLE_DENSITY) {
        let density = f32(result.hits_total) / 50.0;
        return vec4f(heatmap_color(density), 1.0);
    }

    // Add background contribution
    let transmittance = 1.0 - result.color.w;
    let radiance = result.color.xyz + transmittance * BACKGROUND;
    return vec4f(radiance, 1.0);
}
