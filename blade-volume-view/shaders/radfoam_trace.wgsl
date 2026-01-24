// RadFoam Voronoi cell traversal algorithm.
//
// This shared module provides the core traversal logic. The including file must
// define accessor functions BEFORE including this file:
//
//   fn rf_get_point(idx: u32) -> vec3<f32>;
//   fn rf_get_density(idx: u32) -> f32;
//   fn rf_get_color(idx: u32, dir: vec3<f32>) -> vec3<f32>;
//   fn rf_adjacency_begin(idx: u32) -> u32;
//   fn rf_adjacency_end(idx: u32) -> u32;
//   fn rf_get_neighbor(adj_idx: u32) -> u32;
//
// For scene traversal with multiple objects, use a var<private> to store the
// current object index and reference it in the accessor functions.

struct RadFoamTraceParams {
    start_point: u32,
    max_steps: u32,
    weight_threshold: f32,
    pad: u32,
}

struct RadFoamTraceResult {
    color: vec4<f32>,    // rgb + alpha
    cells_visited: u32,
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

    var current = params.start_point;
    var current_pos = rf_get_point(current);

    var steps: u32 = 0u;
    while (t0 < t_end && steps < params.max_steps && transmittance > params.weight_threshold) {
        steps += 1u;

        let begin = rf_adjacency_begin(current);
        let end = rf_adjacency_end(current);
        let num_faces = end - begin;

        var t1: f32 = t_end;
        var next_face: u32 = 0xffffffffu;

        for (var j = 0u; j < num_faces; j += 1u) {
            let next_idx = rf_get_neighbor(begin + j);
            let next_pos = rf_get_point(next_idx);
            let offset_vec = next_pos - current_pos;

            let face_origin = current_pos + 0.5 * offset_vec;
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

        if (next_face == 0xffffffffu) {
            break;
        }

        let next_idx = rf_get_neighbor(begin + next_face);
        let next_pos = rf_get_point(next_idx);

        if (t1 > t0) {
            cells_visited += 1u;
            let s = rf_get_density(current);
            if (s > 1e-6) {
                let dt = max(t1 - t0, 0.0);
                let alpha = 1.0 - exp(-s * dt);
                let w = transmittance * alpha;
                let rgb = rf_get_color(current, ray_dir);
                accum_rgb += w * rgb;
                transmittance *= (1.0 - alpha);
            }
        }

        t0 = max(t0, t1);
        current = next_idx;
        current_pos = next_pos;
    }

    var result: RadFoamTraceResult;
    result.color = vec4<f32>(accum_rgb, 1.0 - transmittance);
    result.cells_visited = cells_visited;
    return result;
}
