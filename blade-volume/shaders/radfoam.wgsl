// RadFoam compute tracer.
//
// Voronoi cell traversal using CSR adjacency list with volumetric integration:
//     alpha = 1 - exp(-s * dt)
//     weight = T * alpha
//     rgb += weight * cell_rgb(dir)
//     T *= (1 - alpha)
//
// Output is HDR to a storage texture (rgba16f).

// #include "common.wgsl"
// #include "sh_eval.wgsl"

struct Params {
    sh_degree: u32,
    weight_threshold: f32,
    max_steps: u32,
    start_point: u32,
    debug_mode: u32,
    pad: vec3<u32>,
};

var<uniform> g_camera: Camera;
var<uniform> g_params: Params;

var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

var g_out: texture_storage_2d<rgba16float, write>;

// Debug mode constants
const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_CELL_DENSITY: u32 = 1u;

// ============================================================================
// RadFoam accessor functions (interface for radfoam_trace.wgsl)
// ============================================================================

fn rf_compute_attr_dim() -> u32 {
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    return 3u * comps + 1u;  // RGB per component + density
}

fn rf_get_point(idx: u32) -> vec3<f32> {
    return g_points[idx].xyz;
}

// Per-point radius for Power Foam. The CPU side packs `radius` into the .w
// channel of `g_points` (density lives in `g_attributes`). Unweighted clouds
// upload zero here, which makes the radical plane in radfoam_trace.wgsl
// degenerate to the standard Voronoi bisector.
fn rf_get_radius(idx: u32) -> f32 {
    return g_points[idx].w;
}

fn rf_get_density(idx: u32) -> f32 {
    let attr_dim = rf_compute_attr_dim();
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    return g_attributes[idx * attr_dim + sh_dim];
}

fn rf_get_color(idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let attr_dim = rf_compute_attr_dim();
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let base = idx * attr_dim;

    var coeffs: array<vec3<f32>, MAX_SH_COMPONENTS>;
    for (var i = 0u; i < comps; i += 1u) {
        let offset = base + i * 3u;
        coeffs[i] = vec3<f32>(
            g_attributes[offset + 0u],
            g_attributes[offset + 1u],
            g_attributes[offset + 2u]
        );
    }
    return 0.5 + sh_eval_color(coeffs, dir, g_params.sh_degree);
}

fn rf_adjacency_begin(idx: u32) -> u32 {
    return g_adjacency_offsets[idx];
}

fn rf_adjacency_end(idx: u32) -> u32 {
    return g_adjacency_offsets[idx + 1u];
}

fn rf_get_neighbor(adj_idx: u32) -> u32 {
    return g_adjacency[adj_idx];
}

// #include "radfoam_trace.wgsl"

// ============================================================================
// Main entry points
// ============================================================================

fn trace_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var params: RadFoamTraceParams;
    params.start_point = g_params.start_point;
    params.max_steps = g_params.max_steps;
    params.weight_threshold = g_params.weight_threshold;

    let result = radfoam_trace(
        ray_origin,
        ray_dir,
        0.0,
        g_camera.depth,
        params
    );

    if (g_params.debug_mode == DEBUG_MODE_CELL_DENSITY) {
        let density = f32(result.cells_visited) / 100.0;
        return vec4<f32>(heatmap_color(density), 1.0);
    }

    return result.color;
}

@compute @workgroup_size(8, 8, 1)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }

    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);

    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>(ndc * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));
    let ray_origin = g_camera.position;

    let rgba = trace_ray(ray_origin, ray_dir);
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
