// RadFoam compute tracer.
//
// Voronoi cell traversal using CSR adjacency list with volumetric integration:
//     alpha = 1 - exp(-s * dt)
//     weight = T * alpha
//     rgb += weight * cell_rgb(dir)
//     T *= (1 - alpha)
//
// Output is HDR to a storage texture (rgba16f).

// #include "common.wgsl"
// #include "sh_eval.wgsl"
// #include "surface_color.wgsl"
// #include "surface_detail.wgsl"
// #include "spherical_voronoi.wgsl"
// #include "radfoam_model.wgsl"

var g_out: texture_storage_2d<rgba16float, write>;

// Debug mode constants
const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_CELL_DENSITY: u32 = 1u;

// ============================================================================
// Main entry points
// ============================================================================

fn trace_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    let result = trace_radfoam_camera(ray_origin, ray_dir, false);

    if (g_params.debug_mode == DEBUG_MODE_CELL_DENSITY) {
        let density = f32(result.cells_visited) / 100.0;
        return vec4<f32>(heatmap_color(density), 1.0);
    }
    return result.color;
}

@compute @workgroup_size(8, 8, 1)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let rgba = trace_ray(ray_origin, ray_dir);
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
