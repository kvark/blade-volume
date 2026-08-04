// Mixed point-cloud scene traversal: RadFoam/PowerFoam plus ray-traced Gaussians.

enable wgpu_ray_query;
enable wgpu_binding_array;

// Naga 29 cannot dynamically index acceleration-structure binding arrays, so
// the composer expands this to a constant-index switch.
// #gaussian_query array g_gaussian_tlas g_gs_obj 64

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "surface_color.wgsl"
// #include "surface_detail.wgsl"
// #include "spherical_voronoi.wgsl"
// #include "scene_bindings.wgsl"

struct Gaussian {
    mean: vec3f,
    pad1: f32,
    rotation: vec4f,
    scale: vec3f,
    opacity: f32,
    harmonics: array<vec4f, MAX_SH_COMPONENTS>,
}

struct GaussianBuffer {
    data: array<Gaussian>,
}

var g_gaussian_tlas: binding_array<acceleration_structure>;
var<storage, read> g_gaussian_data: binding_array<GaussianBuffer>;
var g_out: texture_storage_2d<rgba16float, write>;

// #include "scene_trace_core.wgsl"

var<private> g_gs_obj: u32;
var<private> g_gs_sh_degree: u32;

fn gs_get_gaussian(idx: u32) -> Gaussian {
    return g_gaussian_data[g_gs_obj].data[idx];
}

fn gs_get_sh_degree() -> u32 {
    return g_gs_sh_degree;
}

fn gs_get_weight_threshold() -> f32 {
    return g_scene_params.weight_threshold;
}

// #include "gaussian_trace.wgsl"

fn scene_trace_gaussian(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                        t_start: f32, t_end: f32,
                        bounds: ObjectBounds) -> vec4<f32> {
    g_gs_obj = bounds.data_index;
    g_gs_sh_degree = bounds.sh_degree;

    var params: GaussianTraceParams;
    params.t_start = t_start;
    params.t_end = t_end;
    // The icosahedron is a conservative proxy whose triangle faces can be
    // outside the semantic Gaussian-support interval. Query the complete
    // forward local TLAS, then retain only maximum-response depths inside the
    // software-bound interval through t_start/t_end above.
    params.query_t_start = 0.0;
    params.query_t_end = 1.0e30;
    return gaussian_trace(ray_origin, ray_dir, params).color;
}

fn scene_trace_object(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                      hit: ObjectHit, bounds: ObjectBounds) -> vec4<f32> {
    switch (bounds.object_type) {
        case OBJECT_TYPE_GAUSSIAN: {
            return scene_trace_gaussian(ray_origin, ray_dir, hit.t_near, hit.t_far, bounds);
        }
        case OBJECT_TYPE_RADFOAM: {
            return scene_trace_radfoam(ray_origin, ray_dir, hit.t_near, hit.t_far, bounds);
        }
        default: {
            return vec4<f32>(0.0);
        }
    }
}
