// Front-to-back integration of independently clipped PowerFoam path rows.
// Candidate discovery and radical-plane clipping live in
// `radfoam_record_paths.wgsl`; this pass only evaluates density and SH color.

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "surface_color.wgsl"
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
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_cells: array<u32>;
var<storage, read> g_dts: array<f32>;
var<storage, read> g_mask: array<f32>;
var g_out: texture_storage_2d<rgba16float, write>;

fn attribute_dimension() -> u32 {
    return 3u * min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS) + 1u
        + select(0u, 3u * SURFACE_COLOR_COMPONENTS, (g_params.appearance_flags & 1u) != 0u)
        + select(0u, 6u * SPHERICAL_VORONOI_SITES, (g_params.appearance_flags & 2u) != 0u);
}

fn cell_density(cell: u32) -> f32 {
    let components = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    return g_attributes[cell * attribute_dimension() + 3u * components];
}

fn cell_color(cell: u32, ray_origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let components = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let base = cell * attribute_dimension();
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
    if ((g_params.appearance_flags & 1u) != 0u) {
        let point = g_points[cell];
        let surface = g_surface_normals[cell];
        let basis = surface_color_basis(
            point.xyz,
            point.w,
            surface.xyz,
            surface.w,
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
    if ((g_params.appearance_flags & 2u) != 0u) {
        let surface_length = select(
            0u,
            3u * SURFACE_COLOR_COMPONENTS,
            (g_params.appearance_flags & 1u) != 0u,
        );
        let spherical_base = base + 3u * components + 1u + surface_length;
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
    return max(vec3<f32>(0.0), color);
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
        let density = cell_density(cell);
        if (density > 1e-6) {
            let alpha = 1.0 - exp(-density * g_dts[slot]);
            color += transmittance * alpha * cell_color(cell, g_camera.position, direction);
            transmittance *= 1.0 - alpha;
        }
    }
    textureStore(
        g_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(color, 1.0 - transmittance),
    );
}
