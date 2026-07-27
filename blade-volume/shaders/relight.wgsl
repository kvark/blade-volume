// Forward renderer for relightable surfels.
//
// Each primitive is an oriented disc carrying a material index rather than a
// colour, so the radiance it sends towards the camera is computed here from
// whatever environment is bound, instead of having been baked in when the
// model was made.
//
// Direct lighting only: no shadow rays and no indirect term. That is a
// measured choice rather than a simplification to be fixed later — leaving out
// visibility makes the image too bright and leaving out interreflection makes
// it too dark by about as much, so a renderer with neither lands closer to a
// path traced reference than one with only the first. The two have to be added
// together or not at all.

enable wgpu_ray_query;

// #include "common.wgsl"

var<uniform> g_camera: Camera;

struct Parameters {
    // Nine coefficients of diffuse irradiance, with the Lambertian
    // convolution and the BRDF's 1/PI already folded in, so a unit albedo
    // shades to the outgoing radiance directly.
    irradiance: array<vec4<f32>, 9>,
    background: vec3<f32>,
    // Highest index in the prefiltered ladder, as a float, since every use of
    // it is arithmetic rather than indexing.
    max_specular_level: f32,
}
var<uniform> g_params: Parameters;

struct Surfel {
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    material: u32,
}

struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    specular_f0: vec3<f32>,
    pad: f32,
}

var g_tlas: acceleration_structure;
var<storage> g_surfels: array<Surfel>;
var<storage> g_materials: array<Material>;
// The environment convolved with the GGX lobe, roughness ascending by layer.
var g_specular: texture_2d_array<f32>;
var g_sampler: sampler;
var g_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265;

// Matches `equirect_direction` on the host, inverted.
fn direction_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let yaw = asin(clamp(dir.y, -1.0, 1.0));
    let pitch = atan2(dir.x, dir.z);
    let u = fract(pitch / (2.0 * PI) + 0.5);
    let v = clamp(0.5 - yaw / PI, 0.0, 1.0);
    return vec2<f32>(u, v);
}

fn sh9_dot_irradiance(n: vec3<f32>) -> vec3<f32> {
    var total = g_params.irradiance[0].xyz * 0.282095;
    total += g_params.irradiance[1].xyz * (0.488603 * n.y);
    total += g_params.irradiance[2].xyz * (0.488603 * n.z);
    total += g_params.irradiance[3].xyz * (0.488603 * n.x);
    total += g_params.irradiance[4].xyz * (1.092548 * n.x * n.y);
    total += g_params.irradiance[5].xyz * (1.092548 * n.y * n.z);
    total += g_params.irradiance[6].xyz * (0.315392 * (3.0 * n.z * n.z - 1.0));
    total += g_params.irradiance[7].xyz * (1.092548 * n.x * n.z);
    total += g_params.irradiance[8].xyz * (0.546274 * (n.x * n.x - n.y * n.y));
    return total;
}

// Lazarov's analytic fit to the environment BRDF, so no lookup table is needed
// for what is already an approximation.
fn specular_scale(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * max(n_dot_v, 0.0))) * r.x + r.y;
    return f0 * (a004 * -1.04 + r.z) + vec3<f32>(a004 * 1.04 + r.w);
}

fn shade(normal: vec3<f32>, view: vec3<f32>, material: Material) -> vec3<f32> {
    var out = material.albedo * sh9_dot_irradiance(normal);

    let n_dot_v = dot(normal, view);
    if (n_dot_v > 0.0) {
        let reflection = normalize(2.0 * n_dot_v * normal - view);
        let uv = direction_to_equirect(reflection);
        // The ladder is sampled at a continuous level, so a roughness between
        // two of them blends rather than steps.
        let level = clamp(material.roughness, 0.0, 1.0) * g_params.max_specular_level;
        let low = floor(level);
        let high = min(low + 1.0, g_params.max_specular_level);
        let a = textureSampleLevel(g_specular, g_sampler, uv, i32(low), 0.0).xyz;
        let b = textureSampleLevel(g_specular, g_sampler, uv, i32(high), 0.0).xyz;
        let prefiltered = mix(a, b, level - low);
        out += prefiltered * specular_scale(material.specular_f0, material.roughness, n_dot_v);
    }
    return out;
}

// How many overlapping surfels are resolved in one traversal before the walk
// has to be resumed from the last of them. Deep enough that a single surface
// layer is always covered in one pass.
const HIT_WINDOW: u32 = 12u;
// Below this the remaining surfels cannot change the pixel.
const MIN_TRANSMITTANCE: f32 = 0.003;
// Matches `relight::SURFACE_BAND`.
const SURFACE_BAND: f32 = 2.0;

struct Hit {
    t: f32,
    index: u32,
    // Fraction of the ray this surfel covers, from the falloff towards its rim.
    coverage: f32,
}

fn hit_less(a: Hit, b: Hit) -> bool {
    return a.t < b.t || (a.t == b.t && a.index < b.index);
}

// Matches `relight::coverage` on the host.
fn coverage_of(normalized_radius_squared: f32) -> f32 {
    let t = clamp((normalized_radius_squared - 0.4) / 0.6, 0.0, 1.0);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

// Where a ray meets a surfel's disc, and how much of it that disc covers.
fn intersect_surfel(index: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> Hit {
    var miss = Hit(0.0, index, 0.0);
    let surfel = g_surfels[index];
    let denominator = dot(ray_dir, surfel.normal);
    if (abs(denominator) < 1.0e-8) {
        return miss;
    }
    let t = dot(surfel.center - ray_origin, surfel.normal) / denominator;
    if (t <= 0.0) {
        return miss;
    }
    let offset = ray_origin + t * ray_dir - surfel.center;
    let normalized = dot(offset, offset) / (surfel.radius * surfel.radius);
    return Hit(t, index, coverage_of(normalized));
}

fn shade_surfel(index: u32, ray_dir: vec3<f32>) -> vec3<f32> {
    let surfel = g_surfels[index];
    // A disc has two sides and a ray may arrive at either; shading the one it
    // actually met keeps a surface lit rather than black when the conversion
    // wound it the other way.
    var normal = surfel.normal;
    if (dot(normal, ray_dir) > 0.0) {
        normal = -normal;
    }
    return shade(normal, -ray_dir, g_materials[surfel.material]);
}

// Composite every surfel the ray passes through, nearest first.
//
// The acceleration structure holds a triangle circumscribing each disc, so a
// candidate is a candidate rather than an answer, and the ray query hands them
// over in whatever order the traversal reached them. Compositing is not
// commutative, so they have to be sorted; the window below keeps the next few
// in order and the walk resumes from the last one when more remain.
fn trace_blended(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var radiance = vec3<f32>(0.0);
    var transmittance = 1.0;
    var cursor = Hit(0.0, 0u, 0.0);
    var cursor_valid = false;

    // Bounded so a pathological cloud cannot spin here forever.
    for (var pass_index = 0u; pass_index < 8u; pass_index += 1u) {
        var rq: ray_query;
        rayQueryInitialize(&rq, g_tlas, RayDesc(
            0u, 0xFFu, 0.0, g_camera.depth, ray_origin, ray_dir
        ));

        var hit_count = 0u;
        var hits: array<Hit, HIT_WINDOW>;
        var in_progress = true;
        while (in_progress) {
            in_progress = rayQueryProceed(&rq);
            let intersection = rayQueryGetCandidateIntersection(&rq);
            if (intersection.kind != RAY_QUERY_INTERSECTION_TRIANGLE) {
                continue;
            }
            var candidate = intersect_surfel(intersection.instance_index, ray_origin, ray_dir);
            if (candidate.coverage <= 0.0) {
                continue;
            }
            if (cursor_valid && !hit_less(cursor, candidate)) {
                continue;
            }
            // Insertion sort, so the window holds the nearest few in order
            // whatever order the traversal produced them in.
            for (var i = 0u; i < hit_count; i += 1u) {
                let other = hits[i];
                if (hit_less(candidate, other)) {
                    hits[i] = candidate;
                    candidate = other;
                }
            }
            if (hit_count < HIT_WINDOW) {
                hits[hit_count] = candidate;
                hit_count += 1u;
            }
        }

        if (hit_count == 0u) {
            break;
        }
        // Surfels close together in depth are one surface and get averaged;
        // the next group along the ray is behind it and occludes instead.
        // Compositing all of them alike lets whichever disc happens to be in
        // front carry the pixel, which is exactly what makes its flat normal
        // visible as a facet.
        var i = 0u;
        while (i < hit_count && transmittance > MIN_TRANSMITTANCE) {
            let band = SURFACE_BAND * g_surfels[hits[i].index].radius;
            let limit = hits[i].t + band;
            var sum_color = vec3<f32>(0.0);
            var sum_weight = 0.0;
            var j = i;
            while (j < hit_count && hits[j].t <= limit) {
                let hit = hits[j];
                sum_color += hit.coverage * shade_surfel(hit.index, ray_dir);
                sum_weight += hit.coverage;
                j += 1u;
            }
            // Coverage weights the average; it does not decide opacity.
            // Compositing the weights instead leaves the interior of a surface
            // partly transparent wherever a ray happens to pass near the rim
            // of every disc covering it, and the background bleeding through
            // there is far worse than the facets this is meant to remove.
            // Saturating the sum keeps the inside solid and still lets a
            // silhouette, where only a sliver of one disc is left, go soft.
            let alpha = min(1.0, sum_weight);
            radiance += transmittance * alpha * sum_color / max(sum_weight, 1.0e-6);
            transmittance *= 1.0 - alpha;
            i = j;
        }
        if (transmittance <= MIN_TRANSMITTANCE || hit_count < HIT_WINDOW) {
            break;
        }
        cursor = hits[hit_count - 1u];
        cursor_valid = true;
    }

    return vec4<f32>(radiance, transmittance);
}

@compute @workgroup_size(8, 8, 1)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }

    // Identical ray generation to the other backends, so a pose produces the
    // same rays here as it does for the reference tracer.
    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);
    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>((ndc - g_camera.principal) * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));

    let traced = trace_blended(g_camera.position, ray_dir);
    // Whatever the surfels did not cover shows the environment behind them.
    let radiance = traced.xyz + traced.w * g_params.background;
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(radiance, 1.0));
}
