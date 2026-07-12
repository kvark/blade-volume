// Unified scene traversal shader for software TLAS.
//
// This compute shader implements a software top-level acceleration structure (TLAS)
// that traverses bounding volumes and dispatches to backend-specific renderers:
// - Gaussian: hardware RT query (each object has its own TLAS)
// - RadFoam: Voronoi cell traversal (each object has its own buffers via binding arrays)
// - SDF: sphere tracing (future)
// - Mesh: hardware RT query (future)
//
// The traversal is unified across all object types - the same bounding sphere
// intersection code handles everything, with backend-specific handling inside
// each object's bounds.
//
// LIMITATIONS:
// - RadFoam uses fixed start point (proper entry point search is future work)

enable wgpu_ray_query;

// #include "common.wgsl"
// #include "sh_eval.wgsl"

// ============================================================================
// Object Type Constants (must match Rust ObjectType enum)
// ============================================================================

const OBJECT_TYPE_GAUSSIAN: u32 = 0u;
const OBJECT_TYPE_RADFOAM: u32 = 1u;
const OBJECT_TYPE_SDF: u32 = 2u;
const OBJECT_TYPE_MESH: u32 = 3u;

// ============================================================================
// GPU Structures (must match Rust scene.rs)
// ============================================================================

struct ObjectBounds {
    center: vec3<f32>,
    radius: f32,
    object_type: u32,
    data_index: u32,
    pad: vec2<u32>,
}

struct GpuTransform {
    object_to_world: mat4x4<f32>,
    world_to_object: mat4x4<f32>,
}

// ============================================================================
// Scene Parameters
// ============================================================================

struct SceneParams {
    object_count: u32,
    sh_degree: u32,
    weight_threshold: f32,
    max_steps: u32,
    debug_mode: u32,
    radfoam_attr_dim: u32,
    gaussian_min_opacity: f32,
    pad: u32,
}

var<uniform> g_camera: Camera;
var<uniform> g_scene_params: SceneParams;

// Scene buffers
var<storage, read> g_bounds: array<ObjectBounds>;
var<storage, read> g_transforms: array<GpuTransform>;

// ============================================================================
// RadFoam Backend Buffers (binding arrays - one buffer per object)
// ============================================================================

// Wrapper structs for binding arrays (required by WGSL)
struct RadFoamPointsBuffer {
    data: array<vec4<f32>>,
}
struct RadFoamAttributesBuffer {
    data: array<f32>,
}
struct RadFoamAdjacencyBuffer {
    data: array<u32>,
}
struct RadFoamAdjacencyOffsetsBuffer {
    data: array<u32>,
}

var<storage, read> g_radfoam_points: binding_array<RadFoamPointsBuffer>;
var<storage, read> g_radfoam_attributes: binding_array<RadFoamAttributesBuffer>;
var<storage, read> g_radfoam_adjacency: binding_array<RadFoamAdjacencyBuffer>;
var<storage, read> g_radfoam_adjacency_offsets: binding_array<RadFoamAdjacencyOffsetsBuffer>;

// ============================================================================
// Gaussian Backend Buffers (MVP: single object)
// ============================================================================

struct Gaussian {
    mean: vec3f,
    pad1: f32,
    rotation: vec4f,
    scale: vec3f,
    opacity: f32,
    harmonics: array<vec4f, MAX_SH_COMPONENTS>,
}

var g_gaussian_tlas: acceleration_structure;
var<storage, read> g_gaussian_data: array<Gaussian>;

// Output HDR image
var g_out: texture_storage_2d<rgba16float, write>;

// ============================================================================
// Debug Mode Constants
// ============================================================================

const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_BOUNDS: u32 = 1u;          // Show bounding sphere intersections
const DEBUG_MODE_OBJECT_TYPE: u32 = 2u;     // Color by object type
const DEBUG_MODE_BACKEND_DENSITY: u32 = 3u; // Backend-specific density

// ============================================================================
// Ray-Sphere Intersection
// ============================================================================

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

    // We want the segment [t_near, t_far] that's in front of the ray
    if (t2 < 0.0) {
        return result;
    }

    result.hit = true;
    result.t_near = max(t1, 0.0);
    result.t_far = t2;
    return result;
}

// ============================================================================
// Sorting helpers for object hits
// ============================================================================

const MAX_SCENE_HITS: u32 = 16u;

struct ObjectHit {
    t_near: f32,
    t_far: f32,
    object_idx: u32,
}

// Insertion sort for small arrays
fn sort_hits(hits: ptr<function, array<ObjectHit, MAX_SCENE_HITS>>, count: u32) {
    for (var i = 1u; i < count; i += 1u) {
        let key = (*hits)[i];
        var j = i;
        while (j > 0u && (*hits)[j - 1u].t_near > key.t_near) {
            (*hits)[j] = (*hits)[j - 1u];
            j -= 1u;
        }
        (*hits)[j] = key;
    }
}

// ============================================================================
// RadFoam Backend (accessor functions for shared radfoam_trace module)
// ============================================================================

// Current RadFoam object index for binding array access
var<private> g_rf_obj: u32;
var<private> g_rf_bounded: bool;

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
    let attr_dim = g_scene_params.radfoam_attr_dim;
    let comps = min(sh_component_count(g_scene_params.sh_degree), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    return g_radfoam_attributes[g_rf_obj].data[idx * attr_dim + sh_dim];
}

fn rf_get_color(idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let attr_dim = g_scene_params.radfoam_attr_dim;
    let comps = min(sh_component_count(g_scene_params.sh_degree), MAX_SH_COMPONENTS);
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
    return 0.5 + sh_eval_color(coeffs, dir, g_scene_params.sh_degree);
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
                       data_index: u32, bounded: bool) -> vec4<f32> {
    // Set the current object for accessor functions
    g_rf_obj = data_index;
    g_rf_bounded = bounded;

    var params: RadFoamTraceParams;
    params.start_point = 0u;  // Could be improved with proper entry point search
    params.max_steps = g_scene_params.max_steps;
    params.weight_threshold = g_scene_params.weight_threshold;

    let result = radfoam_trace(ray_origin, ray_dir, t_start, t_end, params);
    return result.color;
}

// ============================================================================
// Gaussian Backend (accessor functions for shared gaussian_trace module)
// ============================================================================

fn gs_get_gaussian(idx: u32) -> Gaussian {
    return g_gaussian_data[idx];
}

fn gs_get_sh_degree() -> u32 {
    return g_scene_params.sh_degree;
}

fn gs_get_weight_threshold() -> f32 {
    return g_scene_params.weight_threshold;
}

// #include "gaussian_trace.wgsl"

fn scene_trace_gaussian(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                        t_start: f32, t_end: f32) -> vec4<f32> {
    var params: GaussianTraceParams;
    params.t_start = t_start;
    params.t_end = t_end;

    let result = gaussian_trace(ray_origin, ray_dir, params);
    return result.color;
}

// ============================================================================
// Unified Scene Traversal
// ============================================================================

fn trace_scene(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    // Collect all object bounds intersections
    var hits: array<ObjectHit, MAX_SCENE_HITS>;
    var hit_count: u32 = 0u;

    for (var i = 0u; i < g_scene_params.object_count && i < MAX_SCENE_HITS; i += 1u) {
        let bounds = g_bounds[i];
        let sphere_hit = ray_sphere_intersect(ray_origin, ray_dir, bounds.center, bounds.radius);

        if (sphere_hit.hit) {
            hits[hit_count] = ObjectHit(sphere_hit.t_near, sphere_hit.t_far, i);
            hit_count += 1u;
        }
    }

    if (hit_count == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Sort by t_near
    sort_hits(&hits, hit_count);

    // Debug mode: bounds visualization
    if (g_scene_params.debug_mode == DEBUG_MODE_BOUNDS) {
        let density = f32(hit_count) / f32(MAX_SCENE_HITS);
        return vec4<f32>(heatmap_color(density), 1.0);
    }

    // Debug mode: object type visualization
    if (g_scene_params.debug_mode == DEBUG_MODE_OBJECT_TYPE) {
        let bounds = g_bounds[hits[0].object_idx];
        var type_color: vec3<f32>;
        switch (bounds.object_type) {
            case OBJECT_TYPE_GAUSSIAN: { type_color = vec3<f32>(1.0, 0.3, 0.3); }  // Red
            case OBJECT_TYPE_RADFOAM: { type_color = vec3<f32>(0.3, 1.0, 0.3); }   // Green
            case OBJECT_TYPE_SDF: { type_color = vec3<f32>(0.3, 0.3, 1.0); }       // Blue
            case OBJECT_TYPE_MESH: { type_color = vec3<f32>(1.0, 1.0, 0.3); }      // Yellow
            default: { type_color = vec3<f32>(0.5, 0.5, 0.5); }
        }
        return vec4<f32>(type_color, 1.0);
    }

    // Traverse objects in order, accumulating radiance
    var total_radiance = vec3<f32>(0.0);
    var total_transmittance = 1.0;

    for (var i = 0u; i < hit_count; i += 1u) {
        if (total_transmittance < g_scene_params.weight_threshold) {
            break;
        }

        let hit = hits[i];
        let bounds = g_bounds[hit.object_idx];
        let transform = g_transforms[hit.object_idx];

        // Transform ray to object space
        let obj_ray_origin = (transform.world_to_object * vec4<f32>(ray_origin, 1.0)).xyz;
        let obj_ray_dir = normalize((transform.world_to_object * vec4<f32>(ray_dir, 0.0)).xyz);

        var result: vec4<f32>;

        switch (bounds.object_type) {
            case OBJECT_TYPE_GAUSSIAN: {
                // Gaussian uses world-space coordinates (transform is baked into instances)
                result = scene_trace_gaussian(ray_origin, ray_dir, hit.t_near, hit.t_far);
            }
            case OBJECT_TYPE_RADFOAM: {
                // RadFoam uses object-space coordinates
                // data_index identifies which RadFoam object's buffers to use
                result = scene_trace_radfoam(obj_ray_origin, obj_ray_dir,
                                             hit.t_near, hit.t_far, bounds.data_index,
                                             bounds.pad.x != 0u);
            }
            default: {
                // Unsupported object type - skip
                result = vec4<f32>(0.0);
            }
        }

        // Composite result (front-to-back blending)
        let alpha = result.w;
        total_radiance += total_transmittance * result.xyz;
        total_transmittance *= (1.0 - alpha);
    }

    return vec4<f32>(total_radiance, 1.0 - total_transmittance);
}

// ============================================================================
// Entry Point
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }

    // NDC in [-1, 1]
    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);

    // Generate ray using camera model (matches existing shaders)
    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>((ndc - g_camera.principal) * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));
    let ray_origin = g_camera.position;

    let rgba = trace_scene(ray_origin, ray_dir);

    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
