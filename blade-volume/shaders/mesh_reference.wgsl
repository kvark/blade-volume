// Ground-truth renderer for the source triangle mesh of an offline conversion.
//
// This exists so mesh -> cloud conversion can be scored on rendered images
// instead of on structural proxies. It deliberately implements the *same*
// shading the converter bakes into cloud samples — flat per-triangle albedo,
// ambient gain applied in linear light, encoded once to display sRGB — so that
// any difference between this image and a cloud render is attributable to the
// representation (sampling rate, cell structure, density profile) rather than
// to two different shading models.
//
// Opaque closest-hit only: no transparency, no lighting model, no shadows.

enable wgpu_ray_query;

// #include "common.wgsl"

var<uniform> g_camera: Camera;

// vec3 alignment pads each field to 16 bytes, giving a 32-byte struct that
// matches `MeshRefParams` on the Rust side; a trailing explicit pad would
// round the WGSL struct up to 48 and break the binding.
struct MeshRefParams {
    // Linear-light multiplier matching `ConvertOptions::ambient`.
    ambient: vec3<f32>,
    // Background is composited the same way the volume backends do it.
    background: vec3<f32>,
}
var<uniform> g_params: MeshRefParams;

var g_mesh_tlas: acceleration_structure;

// One entry per triangle: rgb = linear base colour, a = unused. Indexed by
// the ray query's primitive index, so it must be built in the same order as
// the index buffer handed to the acceleration structure.
var<storage> g_triangle_colors: array<vec4<f32>>;

var g_out: texture_storage_2d<rgba16float, write>;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c <= vec3<f32>(0.0031308);
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, cutoff);
}

fn trace_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var rq: ray_query;
    let desc = RayDesc(
        RAY_FLAG_FORCE_OPAQUE,
        0xFFu,
        0.0,
        g_camera.depth,
        ray_origin,
        ray_dir,
    );
    rayQueryInitialize(&rq, g_mesh_tlas, desc);
    // Drive traversal to completion. Even with opaque triangles, where the
    // implementation commits hits for us, Proceed has to be pumped until it
    // reports there is nothing left, or nothing is committed.
    while (rayQueryProceed(&rq)) {}

    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return vec4<f32>(linear_to_srgb(g_params.background), 0.0);
    }

    // The converter multiplies base colour by the ambient gain in linear
    // light, then encodes once. Mirror that exactly.
    let base = g_triangle_colors[hit.primitive_index].xyz;
    let shaded = base * g_params.ambient;
    return vec4<f32>(linear_to_srgb(clamp(shaded, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }

    // Identical ray generation to radfoam.wgsl, so a pose produces the same
    // rays for the reference and for every cloud backend.
    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);

    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>((ndc - g_camera.principal) * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));

    let rgba = trace_ray(g_camera.position, ray_dir);
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
