// Full-precision surface statistics from the production RadFoam / PowerFoam
// traversal. Reconstruction consumes mode depth, alpha, and peak weight; it
// never evaluates the radiance spherical harmonics.

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "surface_color.wgsl"
// #include "radfoam_model.wgsl"

var g_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn trace_depth_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let ray_origin = g_camera.position;

    let result = trace_radfoam_camera(ray_origin, ray_dir, true);
    let value = vec4<f32>(
        result.depth_mode,
        result.color.a,
        result.peak_weight,
        1.0,
    );
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), value);
}
