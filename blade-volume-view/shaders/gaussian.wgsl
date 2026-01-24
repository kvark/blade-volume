//Must match `MAX_SH_COMPONENTS`
const MAX_HARMONICS: u32 = 16;

struct Camera {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    pad: vec2<u32>,
}

var<uniform> g_camera: Camera;

struct Parameters {
    min_opacity: f32,
    min_transmittance: f32,
    sh_degree: u32,
    debug_mode: u32,
    pad: vec4<u32>,
}
var<uniform> g_params: Parameters;
var g_acc_struct: acceleration_structure;

struct Gaussian {
    mean: vec3f,
    pad1: f32,
    rotation: vec4f,
    scale: vec3f,
    opacity: f32,
    harmonics: array<vec4f, MAX_HARMONICS>,
}
var<storage> g_data: array<Gaussian>;

fn qmake(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    return vec4<f32>(axis * sin(angle), cos(angle));
}
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz,q.w);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) direction: vec3<f32>,
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo = VertexOutput();
    let tc = vec2f(vec2u(vi) & vec2u(1u, 2u)) * vec2f(1.0, 0.5);
    let ndc = 4.0 * tc - 1.0;
    let local_dir = vec3f(ndc * tan(0.5 * g_camera.fov), 1.0);
    vo.clip_pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
    vo.direction = qrot(g_camera.orientation, local_dir);
    return vo;
}

fn evaluate_spherical_harmonics(gs: Gaussian, dir: vec3f) -> vec3f {
    const SH = array<f32, MAX_HARMONICS>(
        0.28209479177387814,
        -0.4886025119029199,
        0.4886025119029199,
        -0.4886025119029199,
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    );

    let d2 = dir * dir;
    var color = 0.5 + SH[0] * gs.harmonics[0].xyz;
    if (g_params.sh_degree >= 1u) {
        color += SH[1] * gs.harmonics[1].xyz * dir.y;
        color += SH[2] * gs.harmonics[2].xyz * dir.z;
        color += SH[3] * gs.harmonics[3].xyz * dir.x;
    }
    if (g_params.sh_degree >= 2u) {
        color += SH[4] * gs.harmonics[4].xyz * dir.x * dir.y;
        color += SH[5] * gs.harmonics[5].xyz * dir.y * dir.z;
        color += SH[6] * gs.harmonics[6].xyz * (3.0 * d2.z - 1.0);
        color += SH[7] * gs.harmonics[7].xyz * dir.x * dir.z;
        color += SH[8] * gs.harmonics[8].xyz * (d2.x - d2.y);
    }
    if (g_params.sh_degree >= 3u) {
        color += SH[9] * gs.harmonics[9].xyz * dir.y * (3.0 * d2.x - d2.y);
        color += SH[10] * gs.harmonics[10].xyz * dir.x * dir.y * dir.z;
        color += SH[11] * gs.harmonics[11].xyz * dir.y * (5.0 * d2.z - 1.0);
        color += SH[12] * gs.harmonics[12].xyz * dir.z * (5.0 * d2.z - 3.0);
        color += SH[13] * gs.harmonics[13].xyz * dir.x * (5.0 * d2.z - 1.0);
        color += SH[14] * gs.harmonics[14].xyz * dir.z * (d2.x - d2.y);
        color += SH[15] * gs.harmonics[15].xyz * dir.x * (d2.x - 3.0 * d2.y);
    }
    return color;
}

fn check_intersection(intersection: RayIntersection, ray_pos: vec3f, ray_dir: vec3f) -> bool {
    let gs = g_data[intersection.instance_index];
    let object_ray_pos = intersection.world_to_object * vec4f(ray_pos, 1.0);
    let object_ray_dir = intersection.world_to_object * vec4f(ray_dir, 0.0);
    let effective_t = -dot(object_ray_pos, object_ray_dir) / dot(object_ray_dir, object_ray_dir);
    let object_pos = object_ray_pos + effective_t * object_ray_dir;
    return dot(object_pos, object_pos) <= 1.0;
}

struct Hit {
    t: f32,
    i: u32,
}

const hit_window: u32 = 5;
const BACKGROUND: vec3f = vec3f(0.0);

// Debug mode constants
const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_PARTICLE_DENSITY: u32 = 1u;

// Heatmap color ramp for debug visualization
fn heatmap_color(t: f32) -> vec3f {
    // Blue -> Cyan -> Green -> Yellow -> Red
    let t_clamped = clamp(t, 0.0, 1.0);
    if (t_clamped < 0.25) {
        let s = t_clamped / 0.25;
        return vec3f(0.0, s, 1.0);
    } else if (t_clamped < 0.5) {
        let s = (t_clamped - 0.25) / 0.25;
        return vec3f(0.0, 1.0, 1.0 - s);
    } else if (t_clamped < 0.75) {
        let s = (t_clamped - 0.5) / 0.25;
        return vec3f(s, 1.0, 0.0);
    } else {
        let s = (t_clamped - 0.75) / 0.25;
        return vec3f(1.0, 1.0 - s, 0.0);
    }
}

@fragment
fn draw_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let ray_pos = g_camera.position;
    let ray_dir = normalize(vo.direction);
    var t_start = 0.0;
    var transmittance = 1.0;
    var radiance = vec3f(0.0);
    var hit_count_total = 0u;

    while (transmittance > g_params.min_transmittance) {
        var rq: ray_query;
        let ray_flags = RAY_FLAG_CULL_BACK_FACING | RAY_FLAG_FORCE_NO_OPAQUE;
        let desc = RayDesc(ray_flags, 0xFFu, t_start, g_camera.depth, ray_pos, ray_dir);
        rayQueryInitialize(&rq, g_acc_struct, desc);
        var hit_count = 0u;
        var hits: array<Hit, hit_window>;
        var in_progress = true;

        while (in_progress) {
            in_progress = rayQueryProceed(&rq);
            let intersection = rayQueryGetCandidateIntersection(&rq);
            if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
                continue;
            }
            if (!check_intersection(intersection, ray_pos, ray_dir)) {
                continue;
            }

            hit_count_total += 1u;
            var hit = Hit(intersection.t, intersection.instance_index);
            for (var i = 0u; i < hit_count; i += 1u) {
                let other = hits[i];
                if (hit.t < other.t) {
                    hits[i] = hit;
                    hit = other;
                }
            }
            if (hit_count < hit_window) {
                hits[hit_count] = hit;
                hit_count += 1;
            }
            if (hit_count == hit_window && intersection.t >= hits[hit_window - 1u].t) {
                rayQueryConfirmIntersection(&rq);
            }
        }

        for (var i = 0u; i < hit_count; i += 1u) {
            let hit = hits[i];
            let gs = g_data[hit.i];

            let g_origin = qrot(qinv(gs.rotation), ray_pos - gs.mean) / gs.scale;
            let g_dir = qrot(qinv(gs.rotation), ray_dir) / gs.scale;
            let effective_t = -dot(g_origin, g_dir) / dot(g_dir, g_dir);
            let g_pos = g_origin + effective_t * g_dir;
            let alpha = gs.opacity * exp(-0.5 * dot(g_pos, g_pos));

            //TODO: evaluate spherical harmonics here
            let color = evaluate_spherical_harmonics(gs, ray_dir);
            radiance += alpha * transmittance * color;
            transmittance *= 1.0 - alpha;
            t_start = hit.t;
        }

        t_start *= 1.00001; // avoid hitting the same primitive
        if (hit_count < hit_window) {
            break;
        }
    }

    // Debug mode: particle density visualization
    if (g_params.debug_mode == DEBUG_MODE_PARTICLE_DENSITY) {
        // Normalize hit count to a reasonable range (0-50 hits -> 0-1)
        let density = f32(hit_count_total) / 50.0;
        let debug_color = heatmap_color(density);
        return vec4f(debug_color, 1.0);
    }

    radiance += transmittance * BACKGROUND;
    return vec4f(radiance, 1.0);
}
