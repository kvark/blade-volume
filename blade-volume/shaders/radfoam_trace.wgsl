// RadFoam / Power Foam cell traversal algorithm.
//
// This shared module provides the core traversal logic. The including file must
// define accessor functions BEFORE including this file:
//
//   fn rf_get_point(idx: u32) -> vec3<f32>;
//   fn rf_get_radius(idx: u32) -> f32;
//   fn rf_is_bounded() -> bool;
//   fn rf_get_density(idx: u32) -> f32;
//   fn rf_get_color(idx: u32, dir: vec3<f32>) -> vec3<f32>;
//   fn rf_adjacency_begin(idx: u32) -> u32;
//   fn rf_adjacency_end(idx: u32) -> u32;
//   fn rf_get_neighbor(adj_idx: u32) -> u32;
//
// The face between sites i and j is the *radical plane* of the two power spheres:
//   face_origin = midpoint + ((r_i^2 - r_j^2) / (2 |p_j - p_i|^2)) * (p_j - p_i)
//   face_normal = p_j - p_i
// When both radii are zero this reduces to the standard Voronoi bisector, so
// unweighted clouds (rf_get_radius returns 0) traverse identically to the
// original RadFoam.
//
// For scene traversal with multiple objects, use a var<private> to store the
// current object index and reference it in the accessor functions.

struct RadFoamTraceParams {
    start_point: u32,
    max_steps: u32,
    weight_threshold: f32,
    integration_start: f32,
    record_depth: bool,
}

struct RadFoamTraceResult {
    color: vec4<f32>,    // rgb + alpha
    cells_visited: u32,
    depth_mode: f32,
    peak_weight: f32,
}

fn rf_support_interval(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    radius: f32,
    t0: f32,
    t1: f32,
) -> vec2<f32> {
    if (!rf_is_bounded()) {
        return vec2<f32>(t0, t1);
    }
    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    if (a <= 0.0) {
        return vec2<f32>(t0, t0);
    }
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - a * c;
    if (discriminant <= 0.0) {
        return vec2<f32>(t0, t0);
    }
    let root = sqrt(discriminant);
    return vec2<f32>(max(t0, (-b - root) / a), min(t1, (-b + root) / a));
}

// Core Voronoi cell traversal
fn radfoam_trace(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t_start: f32,
    t_end: f32,
    params: RadFoamTraceParams,
) -> RadFoamTraceResult {
    var t0 = t_start;
    var transmittance = 1.0;
    var accum_rgb = vec3<f32>(0.0);
    var cells_visited: u32 = 0u;
    var depth_mode = 0.0;
    var peak_weight = 0.0;

    var current = params.start_point;
    var current_pos = rf_get_point(current);
    var current_radius = rf_get_radius(current);

    var steps: u32 = 0u;
    while (t0 < t_end && steps < params.max_steps && transmittance > params.weight_threshold) {
        steps += 1u;

        let begin = rf_adjacency_begin(current);
        let end = rf_adjacency_end(current);
        let num_faces = end - begin;

        var t1: f32 = t_end;
        var next_face: u32 = 0xffffffffu;

        let r_i_sq = current_radius * current_radius;
        for (var j = 0u; j < num_faces; j += 1u) {
            let next_idx = rf_get_neighbor(begin + j);
            let next_pos = rf_get_point(next_idx);
            let r_j = rf_get_radius(next_idx);
            let offset_vec = next_pos - current_pos;

            // Radical plane between power spheres (degenerates to bisector when radii are zero).
            let dsq = max(dot(offset_vec, offset_vec), 1e-20);
            let shift = 0.5 + 0.5 * (r_i_sq - r_j * r_j) / dsq;
            let face_origin = current_pos + shift * offset_vec;
            let face_normal = offset_vec;

            let dp = dot(face_normal, ray_dir);
            if (dp > 0.0) {
                let t = dot(face_origin - ray_origin, face_normal) / dp;
                if (t > t0 && t < t1) {
                    t1 = t;
                    next_face = j;
                }
            }
        }

        let support = rf_support_interval(
            ray_origin, ray_dir, current_pos, current_radius, t0, t1,
        );
        let integration_begin = max(support.x, params.integration_start);
        if (support.y > integration_begin) {
            cells_visited += 1u;
            let s = rf_get_density(current);
            if (s > 1e-6) {
                let dt = support.y - integration_begin;
                let alpha = 1.0 - exp(-s * dt);
                let w = transmittance * alpha;
                if (params.record_depth) {
                    if (w > peak_weight) {
                        peak_weight = w;
                        depth_mode = 0.5 * (integration_begin + support.y);
                    }
                } else {
                    let rgb = rf_get_color(current, normalize(ray_dir));
                    accum_rgb += w * rgb;
                }
                transmittance *= (1.0 - alpha);
            }
        }

        if (next_face == 0xffffffffu) {
            t0 = t1;
            break;
        }

        let next_idx = rf_get_neighbor(begin + next_face);
        let next_pos = rf_get_point(next_idx);

        t0 = max(t0, t1);
        current = next_idx;
        current_pos = next_pos;
        current_radius = rf_get_radius(next_idx);
    }

    var result: RadFoamTraceResult;
    result.color = vec4<f32>(accum_rgb, 1.0 - transmittance);
    result.cells_visited = cells_visited;
    result.depth_mode = depth_mode;
    result.peak_weight = peak_weight;
    return result;
}
