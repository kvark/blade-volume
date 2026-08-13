// Bindings and accessors shared by the standalone RadFoam colour and depth
// tracers. Include common.wgsl, sh_eval.wgsl, surface_color.wgsl,
// surface_detail.wgsl, and spherical_voronoi.wgsl before this fragment.

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
var<storage, read> g_surface_details: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

fn rf_compute_attr_dim() -> u32 {
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    return 3u * comps + 1u
        + select(0u, 3u * SURFACE_COLOR_COMPONENTS, rf_has_surface_color())
        + select(0u, 3u * SURFACE_DETAIL_SITES, rf_has_surface_detail())
        + select(0u, SURFACE_DETAIL_SITES, rf_has_surface_detail_density())
        + select(
            0u,
            6u * SURFACE_DETAIL_SITES * SURFACE_DETAIL_DIRECTIONS,
            rf_has_surface_detail_directional(),
        )
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

fn rf_has_surface_detail() -> bool {
    return (g_params.pad.z & 4u) != 0u;
}

fn rf_has_surface_detail_density() -> bool {
    return (g_params.pad.z & 8u) != 0u;
}

fn rf_has_surface_detail_directional() -> bool {
    return (g_params.pad.z & 16u) != 0u;
}

fn rf_get_surface_normal(idx: u32) -> vec3<f32> {
    return g_surface_normals[idx].xyz;
}

fn rf_get_surface_offset(
    idx: u32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
) -> f32 {
    let surface = g_surface_normals[idx];
    if (!rf_has_surface_detail()) {
        return surface.w;
    }
    var sites: array<vec4<f32>, SURFACE_DETAIL_SITES>;
    for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
        sites[site] = g_surface_details[idx * SURFACE_DETAIL_SITES + site];
    }
    return surface_detail_height(
        rf_get_point(idx),
        rf_get_radius(idx),
        surface.xyz,
        surface.w,
        ray_origin,
        ray_direction,
        query_near,
        sites,
    );
}

fn rf_get_density(
    idx: u32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
    surface_offset: f32,
) -> f32 {
    let attr_dim = rf_compute_attr_dim();
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    let base = idx * attr_dim;
    let density = g_attributes[base + sh_dim];
    if (!rf_has_surface_detail_density()) {
        return density;
    }
    let compact_length = select(
        0u,
        3u * SURFACE_COLOR_COMPONENTS,
        rf_has_surface_color(),
    );
    let density_base = base + sh_dim + 1u + compact_length + 3u * SURFACE_DETAIL_SITES;
    var sites: array<vec4<f32>, SURFACE_DETAIL_SITES>;
    var logits: array<f32, SURFACE_DETAIL_SITES>;
    for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
        sites[site] = g_surface_details[idx * SURFACE_DETAIL_SITES + site];
        logits[site] = g_attributes[density_base + site];
    }
    let surface = g_surface_normals[idx];
    return density * surface_detail_density_scale(
        rf_get_point(idx),
        rf_get_radius(idx),
        surface.xyz,
        surface_offset,
        ray_origin,
        ray_direction,
        query_near,
        sites,
        logits,
    );
}

fn rf_get_density_color(
    idx: u32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
    surface_offset: f32,
) -> vec4<f32> {
    let attr_dim = rf_compute_attr_dim();
    let comps = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let base = idx * attr_dim;
    let density = g_attributes[base + 3u * comps];
    let density_scale_bound = select(
        1.0,
        f32(SURFACE_DETAIL_SITES),
        rf_has_surface_detail_density(),
    );
    if (density * density_scale_bound <= 1e-6) {
        return vec4<f32>(0.0, 0.0, 0.0, density);
    }
    var coeffs: array<vec3<f32>, MAX_SH_COMPONENTS>;
    for (var i = 0u; i < comps; i += 1u) {
        let offset = base + i * 3u;
        coeffs[i] = vec3<f32>(
            g_attributes[offset + 0u],
            g_attributes[offset + 1u],
            g_attributes[offset + 2u]
        );
    }
    var color = 0.5 + sh_eval_color(coeffs, ray_direction, g_params.sh_degree);
    if (rf_has_surface_color()) {
        let surface_base = base + 3u * comps + 1u;
        let basis = surface_color_basis(
            rf_get_point(idx),
            rf_get_radius(idx),
            rf_get_surface_normal(idx),
            surface_offset,
            ray_origin,
            ray_direction,
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
    var density_scale = 1.0;
    let compact_length = select(
        0u,
        3u * SURFACE_COLOR_COMPONENTS,
        rf_has_surface_color(),
    );
    if (rf_has_surface_detail()) {
        let detail_base = base + 3u * comps + 1u + compact_length;
        let density_length = select(
            0u,
            SURFACE_DETAIL_SITES,
            rf_has_surface_detail_density(),
        );
        let directional_base =
            detail_base + 3u * SURFACE_DETAIL_SITES + density_length;
        var sites: array<vec4<f32>, SURFACE_DETAIL_SITES>;
        var colors: array<vec3<f32>, SURFACE_DETAIL_SITES>;
        for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
            sites[site] = g_surface_details[idx * SURFACE_DETAIL_SITES + site];
            let color_offset = detail_base + 3u * site;
            colors[site] = vec3<f32>(
                g_attributes[color_offset],
                g_attributes[color_offset + 1u],
                g_attributes[color_offset + 2u],
            );
            if (rf_has_surface_detail_directional()) {
                var axes: array<vec3<f32>, SURFACE_DETAIL_DIRECTIONS>;
                var directional_colors: array<vec3<f32>, SURFACE_DETAIL_DIRECTIONS>;
                let site_base = site * SURFACE_DETAIL_DIRECTIONS;
                for (var direction = 0u; direction < SURFACE_DETAIL_DIRECTIONS; direction += 1u) {
                    let axis_offset = directional_base + 3u * (site_base + direction);
                    axes[direction] = vec3<f32>(
                        g_attributes[axis_offset],
                        g_attributes[axis_offset + 1u],
                        g_attributes[axis_offset + 2u],
                    );
                    let direction_color_offset = directional_base
                        + 3u * SURFACE_DETAIL_SITES * SURFACE_DETAIL_DIRECTIONS
                        + 3u * (site_base + direction);
                    directional_colors[direction] = vec3<f32>(
                        g_attributes[direction_color_offset],
                        g_attributes[direction_color_offset + 1u],
                        g_attributes[direction_color_offset + 2u],
                    );
                }
                let point = rf_get_point(idx);
                let tangent_site = surface_detail_project_site(
                    sites[site].xyz,
                    rf_get_surface_normal(idx),
                );
                colors[site] = surface_detail_released_residual(
                    colors[site],
                    surface_detail_directional_color(
                        point + rf_get_radius(idx) * tangent_site,
                        ray_origin,
                        axes,
                        directional_colors,
                    ),
                );
            }
        }
        let surface = g_surface_normals[idx];
        if (rf_has_surface_detail_density()) {
            let density_base = detail_base + 3u * SURFACE_DETAIL_SITES;
            var logits: array<f32, SURFACE_DETAIL_SITES>;
            for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
                logits[site] = g_attributes[density_base + site];
            }
            let detail = surface_detail_color_density_scale(
                rf_get_point(idx),
                rf_get_radius(idx),
                surface.xyz,
                surface_offset,
                ray_origin,
                ray_direction,
                query_near,
                sites,
                colors,
                logits,
            );
            color += detail.xyz;
            density_scale = detail.w;
        } else {
            color += surface_detail_color(
                rf_get_point(idx),
                rf_get_radius(idx),
                surface.xyz,
                surface.w,
                surface_offset,
                ray_origin,
                ray_direction,
                query_near,
                sites,
                colors,
            );
        }
    }
    if (rf_has_spherical_voronoi()) {
        let detail_length = select(
            0u,
            3u * SURFACE_DETAIL_SITES,
            rf_has_surface_detail(),
        );
        let density_length = select(
            0u,
            SURFACE_DETAIL_SITES,
            rf_has_surface_detail_density(),
        );
        let directional_length = select(
            0u,
            6u * SURFACE_DETAIL_SITES * SURFACE_DETAIL_DIRECTIONS,
            rf_has_surface_detail_directional(),
        );
        let spherical_base =
            base + 3u * comps + 1u + compact_length + detail_length
                + density_length + directional_length;
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
        color += spherical_voronoi_evaluate(axes, colors, ray_direction);
    }
    return vec4<f32>(max(vec3<f32>(0.0), color), density * density_scale);
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
