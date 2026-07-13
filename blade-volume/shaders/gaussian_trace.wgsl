// Gaussian splatting ray traversal algorithm using hardware RT.
//
// This shared module provides the core traversal logic. The including file must:
// 1. Define `var g_gaussian_tlas: acceleration_structure;` (required name)
// 2. Define accessor functions BEFORE including this file:
//
//    fn gs_get_gaussian(idx: u32) -> Gaussian;
//    fn gs_get_sh_degree() -> u32;
//    fn gs_get_weight_threshold() -> f32;
//
// The Gaussian struct must also be defined before including this file.

// NOTE: `sh_eval.wgsl` must be included by the parent shader before this file.

struct GaussianTraceParams {
    t_start: f32,
    t_end: f32,
    pad: vec2<u32>,
}

struct GaussianTraceResult {
    color: vec4<f32>,    // rgb + alpha
    hits_total: u32,
}

const GAUSSIAN_HIT_WINDOW: u32 = 5u;

struct GaussianHit {
    t: f32,
    idx: u32,
}

fn gaussian_hit_less(a: GaussianHit, b: GaussianHit) -> bool {
    return a.t < b.t || (a.t == b.t && a.idx < b.idx);
}

// Evaluate SH color for a Gaussian
fn gaussian_eval_sh(gs: Gaussian, dir: vec3<f32>) -> vec3<f32> {
    var coeffs: array<vec3<f32>, MAX_SH_COMPONENTS>;
    for (var i = 0u; i < MAX_SH_COMPONENTS; i += 1u) {
        coeffs[i] = gs.harmonics[i].xyz;
    }
    return 0.5 + sh_eval_color(coeffs, dir, gs_get_sh_degree());
}

// Evaluate the Gaussian's maximum-response depth and reject conservative
// triangle-proxy hits outside its finite ellipsoidal support. Unlike a custom
// procedural intersection, a triangle ray query reports the proxy face's t;
// that value is not a valid particle sort key when scales overlap.
fn gaussian_max_response_hit(intersection: RayIntersection,
                             ray_pos: vec3<f32>, ray_dir: vec3<f32>) -> GaussianHit {
    let gs = gs_get_gaussian(intersection.instance_index);
    let object_ray_pos = intersection.world_to_object * vec4f(ray_pos, 1.0);
    let object_ray_dir = intersection.world_to_object * vec4f(ray_dir, 0.0);
    let effective_t = -dot(object_ray_pos, object_ray_dir) / dot(object_ray_dir, object_ray_dir);
    let object_pos = object_ray_pos + effective_t * object_ray_dir;
    if (dot(object_pos, object_pos) <= 1.0) {
        return GaussianHit(effective_t, intersection.instance_index);
    }
    return GaussianHit(0.0, 0xFFFFFFFFu);
}

// Core Gaussian traversal using hardware RT
fn gaussian_trace(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    params: GaussianTraceParams,
) -> GaussianTraceResult {
    var transmittance = 1.0;
    var radiance = vec3<f32>(0.0);
    var hits_total: u32 = 0u;
    let weight_threshold = gs_get_weight_threshold();
    var cursor_valid = false;
    var cursor = GaussianHit(params.t_start, 0u);

    while (transmittance > weight_threshold) {
        var rq: ray_query;
        // Scan the complete interval on every batch. Advancing triangle t_min
        // to the previous proxy face can omit a broad Gaussian whose proxy
        // starts early but whose maximum response belongs to a later batch.
        // Front and back faces are both enabled so rays beginning inside a
        // proxy still produce a candidate; instance-index deduplication below
        // collapses the two faces.
        let ray_flags = RAY_FLAG_FORCE_NO_OPAQUE;
        let desc = RayDesc(ray_flags, 0xFFu, params.t_start, params.t_end, ray_origin, ray_dir);
        rayQueryInitialize(&rq, g_gaussian_tlas, desc);

        var hit_count = 0u;
        var hits: array<GaussianHit, GAUSSIAN_HIT_WINDOW>;
        var in_progress = true;

        while (in_progress) {
            in_progress = rayQueryProceed(&rq);
            let intersection = rayQueryGetCandidateIntersection(&rq);
            if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
                continue;
            }
            let candidate = gaussian_max_response_hit(intersection, ray_origin, ray_dir);
            if (candidate.idx == 0xFFFFFFFFu ||
                candidate.t <= params.t_start || candidate.t >= params.t_end) {
                continue;
            }
            if (cursor_valid && !gaussian_hit_less(cursor, candidate)) {
                continue;
            }

            var duplicate = false;
            for (var i = 0u; i < hit_count; i += 1u) {
                duplicate = duplicate || hits[i].idx == candidate.idx;
            }
            if (duplicate) {
                continue;
            }

            var hit = candidate;
            // Insertion sort to retain the next K particles in exact
            // maximum-response order, independent of BVH candidate order.
            for (var i = 0u; i < hit_count; i += 1u) {
                let other = hits[i];
                if (gaussian_hit_less(hit, other)) {
                    hits[i] = hit;
                    hit = other;
                }
            }
            if (hit_count < GAUSSIAN_HIT_WINDOW) {
                hits[hit_count] = hit;
                hit_count += 1u;
            }
        }

        if (hit_count == 0u) {
            break;
        }

        // Accumulate contributions from sorted hits
        for (var i = 0u; i < hit_count; i += 1u) {
            if (transmittance <= weight_threshold) {
                break;
            }
            let hit = hits[i];
            let gs = gs_get_gaussian(hit.idx);

            // Transform ray to Gaussian's local coordinate system
            let g_origin = qrot(qinv(gs.rotation), ray_origin - gs.mean) / gs.scale;
            let g_dir = qrot(qinv(gs.rotation), ray_dir) / gs.scale;
            let effective_t = -dot(g_origin, g_dir) / dot(g_dir, g_dir);
            let g_pos = g_origin + effective_t * g_dir;
            let alpha = gs.opacity * exp(-0.5 * dot(g_pos, g_pos));

            let color = gaussian_eval_sh(gs, ray_dir);
            radiance += alpha * transmittance * color;
            transmittance *= 1.0 - alpha;
            hits_total += 1u;
        }

        cursor = hits[hit_count - 1u];
        cursor_valid = true;
    }

    var result: GaussianTraceResult;
    result.color = vec4<f32>(radiance, 1.0 - transmittance);
    result.hits_total = hits_total;
    return result;
}
