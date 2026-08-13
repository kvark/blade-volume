// Front-to-back integration of independently clipped PowerFoam path rows.
// Candidate discovery and radical-plane clipping live in
// `radfoam_record_paths.wgsl`; this pass only evaluates density and SH color.

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "surface_color.wgsl"
// #include "surface_detail.wgsl"
// #include "spherical_voronoi.wgsl"

struct SplatIntegrateParams {
    sh_degree: u32,
    max_steps: u32,
    width: u32,
    height: u32,
    weight_threshold: f32,
    appearance_flags: u32,
    _padding0: f32,
    _padding1: f32,
};

var<uniform> g_camera: Camera;
var<uniform> g_params: SplatIntegrateParams;
var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_surface_normals: array<vec4<f32>>;
var<storage, read> g_surface_details: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_cells: array<u32>;
var<storage, read> g_dts: array<f32>;
var<storage, read> g_mask: array<f32>;
var<storage, read> g_surface_queries: array<vec2<f32>>;
var g_out: texture_storage_2d<rgba16float, write>;

fn attribute_dimension() -> u32 {
    return 3u * min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS) + 1u
        + select(0u, 3u * SURFACE_COLOR_COMPONENTS, (g_params.appearance_flags & 1u) != 0u)
        + select(0u, 3u * SURFACE_DETAIL_SITES, (g_params.appearance_flags & 4u) != 0u)
        + select(0u, SURFACE_DETAIL_SITES, (g_params.appearance_flags & 8u) != 0u)
        + select(
            0u,
            6u * SURFACE_DETAIL_SITES * SURFACE_DETAIL_DIRECTIONS,
            (g_params.appearance_flags & 16u) != 0u,
        )
        + select(0u, 6u * SPHERICAL_VORONOI_SITES, (g_params.appearance_flags & 2u) != 0u);
}

fn cell_density_color(
    cell: u32,
    ray_origin: vec3<f32>,
    direction: vec3<f32>,
    query_near: f32,
) -> vec4<f32> {
    let components = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let base = cell * attribute_dimension();
    let density = g_attributes[base + 3u * components];
    let has_detail_density = (g_params.appearance_flags & 8u) != 0u;
    let density_scale_bound = select(1.0, f32(SURFACE_DETAIL_SITES), has_detail_density);
    if (density * density_scale_bound <= 1e-6) {
        return vec4<f32>(0.0, 0.0, 0.0, density);
    }
    var coefficients: array<vec3<f32>, MAX_SH_COMPONENTS>;
    for (var component = 0u; component < components; component += 1u) {
        let offset = base + 3u * component;
        coefficients[component] = vec3<f32>(
            g_attributes[offset],
            g_attributes[offset + 1u],
            g_attributes[offset + 2u],
        );
    }
    var color = 0.5 + sh_eval_color(coefficients, direction, g_params.sh_degree);
    let compact_length = select(
        0u,
        3u * SURFACE_COLOR_COMPONENTS,
        (g_params.appearance_flags & 1u) != 0u,
    );
    var effective_offset = g_surface_normals[cell].w;
    var detail_sites: array<vec4<f32>, SURFACE_DETAIL_SITES>;
    if ((g_params.appearance_flags & 4u) != 0u) {
        for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
            detail_sites[site] = g_surface_details[cell * SURFACE_DETAIL_SITES + site];
        }
        let point = g_points[cell];
        let surface = g_surface_normals[cell];
        effective_offset = surface_detail_height(
            point.xyz,
            point.w,
            surface.xyz,
            surface.w,
            ray_origin,
            direction,
            query_near,
            detail_sites,
        );
    }
    if ((g_params.appearance_flags & 1u) != 0u) {
        let point = g_points[cell];
        let surface = g_surface_normals[cell];
        let basis = surface_color_basis(
            point.xyz,
            point.w,
            surface.xyz,
            effective_offset,
            ray_origin,
            direction,
        );
        let surface_base = base + 3u * components + 1u;
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
    if ((g_params.appearance_flags & 4u) != 0u) {
        let detail_base = base + 3u * components + 1u + compact_length;
        let density_length = select(
            0u,
            SURFACE_DETAIL_SITES,
            (g_params.appearance_flags & 8u) != 0u,
        );
        let directional_base =
            detail_base + 3u * SURFACE_DETAIL_SITES + density_length;
        var detail_colors: array<vec3<f32>, SURFACE_DETAIL_SITES>;
        for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
            let color_offset = detail_base + 3u * site;
            detail_colors[site] = vec3<f32>(
                g_attributes[color_offset],
                g_attributes[color_offset + 1u],
                g_attributes[color_offset + 2u],
            );
            if ((g_params.appearance_flags & 16u) != 0u) {
                var axes: array<vec3<f32>, SURFACE_DETAIL_DIRECTIONS>;
                var directional_colors: array<vec3<f32>, SURFACE_DETAIL_DIRECTIONS>;
                let site_base = site * SURFACE_DETAIL_DIRECTIONS;
                for (var direction_index = 0u; direction_index < SURFACE_DETAIL_DIRECTIONS; direction_index += 1u) {
                    let axis_offset = directional_base + 3u * (site_base + direction_index);
                    axes[direction_index] = vec3<f32>(
                        g_attributes[axis_offset],
                        g_attributes[axis_offset + 1u],
                        g_attributes[axis_offset + 2u],
                    );
                    let direction_color_offset = directional_base
                        + 3u * SURFACE_DETAIL_SITES * SURFACE_DETAIL_DIRECTIONS
                        + 3u * (site_base + direction_index);
                    directional_colors[direction_index] = vec3<f32>(
                        g_attributes[direction_color_offset],
                        g_attributes[direction_color_offset + 1u],
                        g_attributes[direction_color_offset + 2u],
                    );
                }
                let tangent_site = surface_detail_project_site(
                    detail_sites[site].xyz,
                    g_surface_normals[cell].xyz,
                );
                detail_colors[site] = surface_detail_released_residual(
                    detail_colors[site],
                    surface_detail_directional_color(
                        g_points[cell].xyz + g_points[cell].w * tangent_site,
                        ray_origin,
                        axes,
                        directional_colors,
                    ),
                );
            }
        }
        let point = g_points[cell];
        let surface = g_surface_normals[cell];
        if ((g_params.appearance_flags & 8u) != 0u) {
            let density_base = detail_base + 3u * SURFACE_DETAIL_SITES;
            var logits: array<f32, SURFACE_DETAIL_SITES>;
            for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
                logits[site] = g_attributes[density_base + site];
            }
            let detail = surface_detail_color_density_scale(
                point.xyz,
                point.w,
                surface.xyz,
                effective_offset,
                ray_origin,
                direction,
                query_near,
                detail_sites,
                detail_colors,
                logits,
            );
            color += detail.xyz;
            density_scale = detail.w;
        } else {
            color += surface_detail_color(
                point.xyz,
                point.w,
                surface.xyz,
                surface.w,
                effective_offset,
                ray_origin,
                direction,
                query_near,
                detail_sites,
                detail_colors,
            );
        }
    }
    if ((g_params.appearance_flags & 2u) != 0u) {
        let detail_length = select(
            0u,
            3u * SURFACE_DETAIL_SITES,
            (g_params.appearance_flags & 4u) != 0u,
        );
        let density_length = select(
            0u,
            SURFACE_DETAIL_SITES,
            (g_params.appearance_flags & 8u) != 0u,
        );
        let directional_length = select(
            0u,
            6u * SURFACE_DETAIL_SITES * SURFACE_DETAIL_DIRECTIONS,
            (g_params.appearance_flags & 16u) != 0u,
        );
        let spherical_base =
            base + 3u * components + 1u + compact_length + detail_length
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
        color += spherical_voronoi_evaluate(axes, colors, direction);
    }
    return vec4<f32>(max(vec3<f32>(0.0), color), density * density_scale);
}

@compute @workgroup_size(8, 8, 1)
fn integrate_powerfoam_splats(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= g_params.width || gid.y >= g_params.height) {
        return;
    }
    let pixel = gid.y * g_params.width + gid.x;
    let row = pixel * g_params.max_steps;
    let px = (f32(gid.x) + 0.5) / f32(g_params.width);
    let py = (f32(gid.y) + 0.5) / f32(g_params.height);
    let ndc = vec2<f32>(2.0 * px - 1.0, 2.0 * py - 1.0);
    let local_direction = vec3<f32>(
        (ndc - g_camera.principal) * tan(0.5 * g_camera.fov),
        1.0,
    );
    let direction = normalize(qrot(g_camera.orientation, local_direction));

    var transmittance = 1.0;
    var color = vec3<f32>(0.0);
    for (var step = 0u; step < g_params.max_steps; step += 1u) {
        let slot = row + step;
        if (g_mask[slot] <= 0.0 || transmittance <= g_params.weight_threshold) {
            break;
        }
        let cell = g_cells[slot];
        var query_near = 0.0;
        if ((g_params.appearance_flags & 4u) != 0u) {
            query_near = g_surface_queries[slot].x;
        }
        let sample = cell_density_color(cell, g_camera.position, direction, query_near);
        let density = sample.w;
        if (density > 1e-6) {
            let alpha = 1.0 - exp(-density * g_dts[slot]);
            color += transmittance * alpha * sample.xyz;
            transmittance *= 1.0 - alpha;
        }
    }
    textureStore(
        g_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(color, 1.0 - transmittance),
    );
}
