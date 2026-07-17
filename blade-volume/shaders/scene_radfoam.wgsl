// RadFoam/PowerFoam-only scene traversal. This pipeline intentionally has no
// ray-query extension or acceleration-structure bindings.

enable wgpu_binding_array;

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "scene_bindings.wgsl"

var g_out: texture_storage_2d<rgba16float, write>;

// #include "scene_trace_core.wgsl"

fn scene_trace_object(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
                      hit: ObjectHit, bounds: ObjectBounds) -> vec4<f32> {
    if (bounds.object_type == OBJECT_TYPE_RADFOAM) {
        return scene_trace_radfoam(ray_origin, ray_dir, hit.t_near, hit.t_far, bounds);
    }
    return vec4<f32>(0.0);
}
