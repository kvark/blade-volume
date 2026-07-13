// Shared software-TLAS traversal and RadFoam backend support.
// The parent shader defines g_out and scene_trace_object().

const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_BOUNDS: u32 = 1u;
const DEBUG_MODE_OBJECT_TYPE: u32 = 2u;
const DEBUG_MODE_BACKEND_DENSITY: u32 = 3u;

struct SphereHit {
    hit: bool,
    t_near: f32,
    t_far: f32,
}

fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                        center: vec3<f32>, radius: f32) -> SphereHit {
    var result: SphereHit;
    result.hit = false;
    result.t_near = 0.0;
    result.t_far = 0.0;

    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return result;
    }

    let sqrt_disc = sqrt(discriminant);
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    if (t2 < 0.0) {
        return result;
    }

    result.hit = true;
    result.t_near = max(t1, 0.0);
    result.t_far = t2;
    return result;
}

struct ObjectHit {
    t_near: f32,
    t_far: f32,
    object_idx: u32,
}

fn object_hit_less(a: ObjectHit, b: ObjectHit) -> bool {
    return a.t_near < b.t_near ||
        (a.t_near == b.t_near && a.object_idx < b.object_idx);
}

fn find_next_object_hit(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                        cursor_valid: bool, cursor: ObjectHit) -> ObjectHit {
    var best = ObjectHit(0.0, 0.0, 0xFFFFFFFFu);
    for (var i = 0u; i < g_scene_params.object_count; i += 1u) {
        let bounds = g_bounds[i];
        let sphere_hit = ray_sphere_intersect(ray_origin, ray_dir, bounds.center, bounds.radius);
        if (!sphere_hit.hit) {
            continue;
        }

        let candidate = ObjectHit(sphere_hit.t_near, sphere_hit.t_far, i);
        if (cursor_valid && !object_hit_less(cursor, candidate)) {
            continue;
        }
        if (best.object_idx == 0xFFFFFFFFu || object_hit_less(candidate, best)) {
            best = candidate;
        }
    }
    return best;
}

var<private> g_rf_obj: u32;
var<private> g_rf_bounded: bool;
var<private> g_rf_sh_degree: u32;
var<private> g_rf_attribute_stride: u32;

fn rf_get_point(idx: u32) -> vec3<f32> {
    return g_radfoam_points[g_rf_obj].data[idx].xyz;
}

fn rf_get_radius(idx: u32) -> f32 {
    return g_radfoam_points[g_rf_obj].data[idx].w;
}

fn rf_is_bounded() -> bool {
    return g_rf_bounded;
}

fn rf_get_density(idx: u32) -> f32 {
    let attr_dim = g_rf_attribute_stride;
    let comps = min(sh_component_count(g_rf_sh_degree), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    return g_radfoam_attributes[g_rf_obj].data[idx * attr_dim + sh_dim];
}

fn rf_get_color(idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let attr_dim = g_rf_attribute_stride;
    let comps = min(sh_component_count(g_rf_sh_degree), MAX_SH_COMPONENTS);
    let base = idx * attr_dim;

    var coeffs: array<vec3<f32>, MAX_SH_COMPONENTS>;
    for (var i = 0u; i < comps; i += 1u) {
        let offset = base + i * 3u;
        coeffs[i] = vec3<f32>(
            g_radfoam_attributes[g_rf_obj].data[offset + 0u],
            g_radfoam_attributes[g_rf_obj].data[offset + 1u],
            g_radfoam_attributes[g_rf_obj].data[offset + 2u]
        );
    }
    return 0.5 + sh_eval_color(coeffs, dir, g_rf_sh_degree);
}

fn rf_adjacency_begin(idx: u32) -> u32 {
    return g_radfoam_adjacency_offsets[g_rf_obj].data[idx];
}

fn rf_adjacency_end(idx: u32) -> u32 {
    return g_radfoam_adjacency_offsets[g_rf_obj].data[idx + 1u];
}

fn rf_get_neighbor(adj_idx: u32) -> u32 {
    return g_radfoam_adjacency[g_rf_obj].data[adj_idx];
}

// #include "radfoam_trace.wgsl"

fn scene_trace_radfoam(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                       t_start: f32, t_end: f32,
                       bounds: ObjectBounds) -> vec4<f32> {
    g_rf_obj = bounds.data_index;
    g_rf_bounded = (bounds.flags & 1u) != 0u;
    g_rf_sh_degree = bounds.sh_degree;
    g_rf_attribute_stride = bounds.attribute_stride;

    var params: RadFoamTraceParams;
    params.start_point = bounds.start_point;
    params.max_steps = g_scene_params.max_steps;
    params.weight_threshold = g_scene_params.weight_threshold;
    params.integration_start = t_start;

    // Start in the camera-containing cell for a topologically valid walk, but
    // clip optical integration to the software-TLAS interval.
    let result = radfoam_trace(ray_origin, ray_dir, 0.0, t_end, params);
    return result.color;
}

fn trace_scene(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var hit_count: u32 = 0u;
    for (var i = 0u; i < g_scene_params.object_count; i += 1u) {
        let bounds = g_bounds[i];
        let sphere_hit = ray_sphere_intersect(ray_origin, ray_dir, bounds.center, bounds.radius);
        if (sphere_hit.hit) {
            hit_count += 1u;
        }
    }

    if (hit_count == 0u) {
        return vec4<f32>(0.0);
    }

    let first_hit = find_next_object_hit(
        ray_origin, ray_dir, false, ObjectHit(0.0, 0.0, 0u));

    if (g_scene_params.debug_mode == DEBUG_MODE_BOUNDS) {
        let density = f32(hit_count) / f32(max(g_scene_params.object_count, 1u));
        return vec4<f32>(heatmap_color(density), 1.0);
    }

    if (g_scene_params.debug_mode == DEBUG_MODE_OBJECT_TYPE) {
        let bounds = g_bounds[first_hit.object_idx];
        var type_color: vec3<f32>;
        switch (bounds.object_type) {
            case OBJECT_TYPE_GAUSSIAN: { type_color = vec3<f32>(1.0, 0.3, 0.3); }
            case OBJECT_TYPE_RADFOAM: { type_color = vec3<f32>(0.3, 1.0, 0.3); }
            case OBJECT_TYPE_SDF: { type_color = vec3<f32>(0.3, 0.3, 1.0); }
            case OBJECT_TYPE_MESH: { type_color = vec3<f32>(1.0, 1.0, 0.3); }
            default: { type_color = vec3<f32>(0.5); }
        }
        return vec4<f32>(type_color, 1.0);
    }

    var total_radiance = vec3<f32>(0.0);
    var total_transmittance = 1.0;
    var cursor_valid = false;
    var cursor = ObjectHit(0.0, 0.0, 0u);

    for (var i = 0u; i < hit_count; i += 1u) {
        if (total_transmittance < g_scene_params.weight_threshold) {
            break;
        }

        let hit = find_next_object_hit(ray_origin, ray_dir, cursor_valid, cursor);
        cursor = hit;
        cursor_valid = true;
        let bounds = g_bounds[hit.object_idx];
        let transform = g_transforms[hit.object_idx];
        let obj_ray_origin = (transform.world_to_object * vec4<f32>(ray_origin, 1.0)).xyz;
        // Keeping this direction unnormalized preserves the world-ray t under
        // rotations and nonuniform object scale.
        let obj_ray_dir = (transform.world_to_object * vec4<f32>(ray_dir, 0.0)).xyz;
        let result = scene_trace_object(obj_ray_origin, obj_ray_dir, hit, bounds);

        total_radiance += total_transmittance * result.xyz;
        total_transmittance *= 1.0 - result.w;
    }

    return vec4<f32>(total_radiance, 1.0 - total_transmittance);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }

    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);
    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>((ndc - g_camera.principal) * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));
    let rgba = trace_scene(g_camera.position, ray_dir);
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
