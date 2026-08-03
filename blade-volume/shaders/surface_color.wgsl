// Shared spatial appearance basis for oriented PowerFoam sites.
//
// The query is intersected with the site's displaced surface plane. Its
// tangent displacement is normalized by the support radius and clamped to a
// finite cube. This avoids storing a per-site tangent frame: q lies in the
// two-dimensional tangent plane, while its three object-space coordinates
// transform consistently with the cloud.

const SURFACE_COLOR_COMPONENTS: u32 = 4u;

fn surface_color_basis(
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    offset: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
) -> vec4<f32> {
    let denominator = dot(ray_direction, normal);
    let numerator = dot(center - ray_origin, normal) + offset;
    // Bounded reciprocal matching the differentiable graph. It approaches
    // 1/d away from grazing incidence and remains finite at d == 0.
    let t = numerator * denominator / (denominator * denominator + 1e-12);
    let hit = ray_origin + t * ray_direction;
    let relative = hit - (center + offset * normal);
    let tangent = relative - dot(relative, normal) * normal;
    let q = clamp(tangent / max(radius, 1e-6), vec3<f32>(-1.0), vec3<f32>(1.0));
    return vec4<f32>(q, min(dot(q, q), 1.0));
}
