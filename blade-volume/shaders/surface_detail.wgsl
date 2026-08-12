// Shared spatial detail-site evaluation for oriented PowerFoam cells.
//
// Site xyz values are radius-normalized object-space offsets. They are
// projected onto the current tangent plane so learned normal changes remain
// continuous without storing an implicit tangent frame. Site w values are
// radius-normalized signed heights. The temperature and squared-distance
// kernel match the released PowerFoam implementation.

const SURFACE_DETAIL_SITES: u32 = 8u;
const SURFACE_DETAIL_DIRECTIONS: u32 = 8u;
const SURFACE_DETAIL_TEMPERATURE: f32 = 10.0;

fn surface_detail_project_site(raw_site: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return raw_site - dot(raw_site, normal) * normal;
}

fn surface_detail_directional_color(
    site_position: vec3<f32>,
    camera_origin: vec3<f32>,
    axes: array<vec3<f32>, SURFACE_DETAIL_DIRECTIONS>,
    colors: array<vec3<f32>, SURFACE_DETAIL_DIRECTIONS>,
) -> vec3<f32> {
    let view = site_position - camera_origin;
    let direction = view / max(length(view), 1e-20);
    var max_logit = -1e30;
    for (var i = 0u; i < SURFACE_DETAIL_DIRECTIONS; i += 1u) {
        let temperature = length(axes[i]);
        let axis = axes[i] / max(temperature, 1e-20);
        max_logit = max(max_logit, -temperature * length(direction - axis));
    }
    var weight_sum = 0.0;
    var color_sum = vec3<f32>(0.0);
    for (var i = 0u; i < SURFACE_DETAIL_DIRECTIONS; i += 1u) {
        let temperature = length(axes[i]);
        let axis = axes[i] / max(temperature, 1e-20);
        let logit = -temperature * length(direction - axis);
        let weight = exp(logit - max_logit);
        weight_sum += weight;
        color_sum += weight * colors[i];
    }
    return color_sum / max(weight_sum, 1e-20);
}

fn surface_detail_query_t(
    center: vec3<f32>,
    normal: vec3<f32>,
    offset: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
) -> f32 {
    let denominator = dot(ray_direction, normal);
    if (denominator >= -1e-20) {
        return query_near;
    }
    let plane_t = (dot(center - ray_origin, normal) + offset) / denominator;
    return max(query_near, plane_t);
}

fn surface_detail_height(
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    base_offset: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
    sites: array<vec4<f32>, SURFACE_DETAIL_SITES>,
) -> f32 {
    let safe_radius = max(radius, 1e-6);
    let query_t = surface_detail_query_t(
        center, normal, base_offset, ray_origin, ray_direction, query_near,
    );
    let query = (ray_origin + query_t * ray_direction - center) / safe_radius;
    var height_sum = 0.0;
    var weight_sum = 0.0;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        let site = sites[site_index];
        let tangent_site = surface_detail_project_site(site.xyz, normal);
        let delta = query - tangent_site;
        let weight = exp(-SURFACE_DETAIL_TEMPERATURE * dot(delta, delta));
        height_sum += weight * site.w;
        weight_sum += weight;
    }
    let normalized_height = height_sum / max(weight_sum, 1e-20);
    return base_offset + safe_radius * normalized_height;
}

fn surface_detail_color(
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    base_offset: f32,
    effective_offset: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
    sites: array<vec4<f32>, SURFACE_DETAIL_SITES>,
    colors: array<vec3<f32>, SURFACE_DETAIL_SITES>,
) -> vec3<f32> {
    let safe_radius = max(radius, 1e-6);
    let query_t = surface_detail_query_t(
        center, normal, effective_offset, ray_origin, ray_direction, query_near,
    );
    let query = (ray_origin + query_t * ray_direction - center) / safe_radius;
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        let tangent_site = surface_detail_project_site(sites[site_index].xyz, normal);
        let delta = query - tangent_site;
        let weight = exp(-SURFACE_DETAIL_TEMPERATURE * dot(delta, delta));
        color_sum += weight * colors[site_index];
        weight_sum += weight;
    }
    return color_sum / max(weight_sum, 1e-20);
}

fn surface_detail_density_scale(
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    effective_offset: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
    sites: array<vec4<f32>, SURFACE_DETAIL_SITES>,
    logits: array<f32, SURFACE_DETAIL_SITES>,
) -> f32 {
    let safe_radius = max(radius, 1e-6);
    let query_t = surface_detail_query_t(
        center, normal, effective_offset, ray_origin, ray_direction, query_near,
    );
    let query = (ray_origin + query_t * ray_direction - center) / safe_radius;
    var max_logit = -1e30;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        max_logit = max(max_logit, logits[site_index]);
    }
    var logit_sum = 0.0;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        logit_sum += exp(logits[site_index] - max_logit);
    }
    var spatial_sum = 0.0;
    var residual_sum = 0.0;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        let tangent_site = surface_detail_project_site(sites[site_index].xyz, normal);
        let delta = query - tangent_site;
        let spatial_weight = exp(-SURFACE_DETAIL_TEMPERATURE * dot(delta, delta));
        let density_residual = f32(SURFACE_DETAIL_SITES)
            * exp(logits[site_index] - max_logit) / max(logit_sum, 1e-20) - 1.0;
        spatial_sum += spatial_weight;
        residual_sum += spatial_weight * density_residual;
    }
    return max(0.0, 1.0 + residual_sum / max(spatial_sum, 1e-20));
}

// Evaluate colour and normalized density from one displaced-plane query. The
// standalone, scene, and splat colour renderers consume both values together;
// sharing the spatial kernel avoids a second set of eight projections and
// exponentials per integrated cell. Depth-only rendering retains the scalar
// density helper above and does not pay for colour.
fn surface_detail_color_density_scale(
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    effective_offset: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    query_near: f32,
    sites: array<vec4<f32>, SURFACE_DETAIL_SITES>,
    colors: array<vec3<f32>, SURFACE_DETAIL_SITES>,
    logits: array<f32, SURFACE_DETAIL_SITES>,
) -> vec4<f32> {
    let safe_radius = max(radius, 1e-6);
    let query_t = surface_detail_query_t(
        center, normal, effective_offset, ray_origin, ray_direction, query_near,
    );
    let query = (ray_origin + query_t * ray_direction - center) / safe_radius;
    var max_logit = -1e30;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        max_logit = max(max_logit, logits[site_index]);
    }
    var logit_sum = 0.0;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        logit_sum += exp(logits[site_index] - max_logit);
    }
    var color_sum = vec3<f32>(0.0);
    var spatial_sum = 0.0;
    var residual_sum = 0.0;
    for (var site_index = 0u; site_index < SURFACE_DETAIL_SITES; site_index += 1u) {
        let tangent_site = surface_detail_project_site(sites[site_index].xyz, normal);
        let delta = query - tangent_site;
        let spatial_weight = exp(-SURFACE_DETAIL_TEMPERATURE * dot(delta, delta));
        let density_residual = f32(SURFACE_DETAIL_SITES)
            * exp(logits[site_index] - max_logit) / max(logit_sum, 1e-20) - 1.0;
        color_sum += spatial_weight * colors[site_index];
        spatial_sum += spatial_weight;
        residual_sum += spatial_weight * density_residual;
    }
    let denominator = max(spatial_sum, 1e-20);
    let density_scale = max(0.0, 1.0 + residual_sum / denominator);
    return vec4<f32>(color_sum / denominator, density_scale);
}
