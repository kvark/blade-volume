// Full-precision depth statistics from independently clipped PowerFoam
// supports. Candidate discovery and interval clipping live in
// `radfoam_record_paths.wgsl`.

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "surface_color.wgsl"
// #include "surface_detail.wgsl"
// #include "spherical_voronoi.wgsl"

struct SplatDepthParams {
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
var<uniform> g_params: SplatDepthParams;
var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_surface_normals: array<vec4<f32>>;
var<storage, read> g_surface_details: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_cells: array<u32>;
var<storage, read> g_entry_depths: array<u32>;
var<storage, read> g_dts: array<f32>;
var<storage, read> g_mask: array<f32>;
var<storage, read> g_surface_queries: array<vec2<f32>>;
var g_out: texture_storage_2d<rgba32float, write>;

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

fn cell_density(
    cell: u32,
    ray_origin: vec3<f32>,
    direction: vec3<f32>,
    query_near: f32,
) -> f32 {
    let components = min(sh_component_count(g_params.sh_degree), MAX_SH_COMPONENTS);
    let base = cell * attribute_dimension();
    let density = g_attributes[base + 3u * components];
    if ((g_params.appearance_flags & 8u) == 0u || density <= 1e-6) {
        return density;
    }

    var sites: array<vec4<f32>, SURFACE_DETAIL_SITES>;
    for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
        sites[site] = g_surface_details[cell * SURFACE_DETAIL_SITES + site];
    }
    let point = g_points[cell];
    let surface = g_surface_normals[cell];
    let effective_offset = surface_detail_height(
        point.xyz,
        point.w,
        surface.xyz,
        surface.w,
        ray_origin,
        direction,
        query_near,
        sites,
    );
    let compact_length = select(
        0u,
        3u * SURFACE_COLOR_COMPONENTS,
        (g_params.appearance_flags & 1u) != 0u,
    );
    let detail_base = base + 3u * components + 1u + compact_length;
    let density_base = detail_base + 3u * SURFACE_DETAIL_SITES;
    var logits: array<f32, SURFACE_DETAIL_SITES>;
    for (var site = 0u; site < SURFACE_DETAIL_SITES; site += 1u) {
        logits[site] = g_attributes[density_base + site];
    }
    return density * surface_detail_density_scale(
        point.xyz,
        point.w,
        surface.xyz,
        effective_offset,
        ray_origin,
        direction,
        query_near,
        sites,
        logits,
    );
}

@compute @workgroup_size(8, 8, 1)
fn integrate_powerfoam_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    var depth_mode = 0.0;
    var peak_weight = 0.0;
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
        let density = cell_density(cell, g_camera.position, direction, query_near);
        if (density > 1e-6) {
            let dt = g_dts[slot];
            let alpha = 1.0 - exp(-density * dt);
            let weight = transmittance * alpha;
            if (weight > peak_weight) {
                peak_weight = weight;
                depth_mode = bitcast<f32>(g_entry_depths[slot]) + 0.5 * dt;
            }
            transmittance *= 1.0 - alpha;
        }
    }
    textureStore(
        g_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(depth_mode, 1.0 - transmittance, peak_weight, 1.0),
    );
}
