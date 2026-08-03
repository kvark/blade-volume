// Bindings and accessors shared by the standalone RadFoam colour and depth
// tracers. Include common.wgsl, sh_eval.wgsl, surface_color.wgsl, and
// spherical_voronoi.wgsl before this fragment.

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
var<storage, read> g_surface_normals: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

fn rf_compute_attr_dim() -> u32 {
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    return 3u * comps + 1u
        + select(0u, 3u * SURFACE_COLOR_COMPONENTS, rf_has_surface_color())
        + select(0u, 6u * SPHERICAL_VORONOI_SITES, rf_has_spherical_voronoi());
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

fn rf_is_bounded() -> bool {
    return g_params.pad.x != 0u;
}

fn rf_is_oriented() -> bool {
    return g_params.pad.y != 0u;
}

fn rf_has_surface_color() -> bool {
    return (g_params.pad.z & 1u) != 0u;
}

fn rf_has_spherical_voronoi() -> bool {
    return (g_params.pad.z & 2u) != 0u;
}

fn rf_get_surface_normal(idx: u32) -> vec3<f32> {
    return g_surface_normals[idx].xyz;
}

fn rf_get_surface_offset(idx: u32) -> f32 {
    return g_surface_normals[idx].w;
}

fn rf_get_density(idx: u32) -> f32 {
    let attr_dim = rf_compute_attr_dim();
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    return g_attributes[idx * attr_dim + sh_dim];
}

fn rf_get_color(
    idx: u32,
    ray_origin: vec3<f32>,
    dir: vec3<f32>,
) -> vec3<f32> {
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
    var color = 0.5 + sh_eval_color(coeffs, dir, g_params.sh_degree);
    if (rf_has_surface_color()) {
        let surface_base = base + 3u * comps + 1u;
        let basis = surface_color_basis(
            rf_get_point(idx),
            rf_get_radius(idx),
            rf_get_surface_normal(idx),
            rf_get_surface_offset(idx),
            ray_origin,
            dir,
        );
        for (var component = 0u; component < SURFACE_COLOR_COMPONENTS; component += 1u) {
            let offset = surface_base + 3u * component;
            color += basis[component] * vec3<f32>(
                g_attributes[offset],
                g_attributes[offset + 1u],
                g_attributes[offset + 2u],
            );
        }
    }
    if (rf_has_spherical_voronoi()) {
        let surface_length = select(
            0u,
            3u * SURFACE_COLOR_COMPONENTS,
            rf_has_surface_color(),
        );
        let spherical_base = base + 3u * comps + 1u + surface_length;
        var axes: array<vec3<f32>, SPHERICAL_VORONOI_SITES>;
        var colors: array<vec3<f32>, SPHERICAL_VORONOI_SITES>;
        for (var site = 0u; site < SPHERICAL_VORONOI_SITES; site += 1u) {
            let axis_offset = spherical_base + 3u * site;
            axes[site] = vec3<f32>(
                g_attributes[axis_offset],
                g_attributes[axis_offset + 1u],
                g_attributes[axis_offset + 2u],
            );
            let color_offset = spherical_base + 3u * SPHERICAL_VORONOI_SITES + 3u * site;
            colors[site] = vec3<f32>(
                g_attributes[color_offset],
                g_attributes[color_offset + 1u],
                g_attributes[color_offset + 2u],
            );
        }
        color += spherical_voronoi_evaluate(axes, colors, dir);
    }
    return max(vec3<f32>(0.0), color);
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

fn trace_radfoam_camera(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    record_depth: bool,
) -> RadFoamTraceResult {
    var params: RadFoamTraceParams;
    params.start_point = g_params.start_point;
    params.max_steps = g_params.max_steps;
    params.weight_threshold = g_params.weight_threshold;
    params.integration_start = 0.0;
    params.record_depth = record_depth;

    return radfoam_trace(
        ray_origin,
        ray_dir,
        0.0,
        g_camera.depth,
        params
    );
}
