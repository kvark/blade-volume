// RadFoam-style compute tracer (MVP)
//
// This is a compute-only WGSL port of the upstream RadFoam forward pass traversal idea:
// - Voronoi cell traversal using a CSR adjacency list.
// - Segment-wise volumetric integration with per-cell density s:
//     alpha = 1 - exp(-s * dt)
//     weight = T * alpha
//     rgb += weight * cell_rgb(dir)
//     T *= (1 - alpha)
//
// TEMPORARY / MVP NOTES:
// - DC (L0) SH coefficients are approximated from PLY RGB preview in the loader (not exact).
// - We do NOT prefetch half-precision adjacent_diffs; we compute neighbor offsets from points.
// - This shader assumes attributes are packed per point as:
//     [R_l0, G_l0, B_l0, R_l1m?, G_l1m?, B_l1m?, ..., density]
//   i.e. interleaved RGB per SH basis component, and density is the last scalar.
//
// Bindings are designed to be driven by `blade_graphics` ShaderData.
//
// Output is HDR to a storage texture (rgba16f).

const MAX_SH_COMPONENTS: u32 = 16u; // (1+3)^2

struct Camera {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>, // quaternion (x,y,z,w)
    fov: vec2<f32>,         // (fov_x, fov_y) where local_dir = (ndc * tan(0.5*fov), 1)
    pad: vec2<u32>,
};

struct Params {
    // must match scene
    sh_degree: u32,

    // tracing controls
    weight_threshold: f32,   // stop when transmittance <= threshold
    max_steps: u32,          // maximum cell transitions
    start_point: u32,        // starting cell index for all rays (MVP)
};

var<uniform> g_camera: Camera;
var<uniform> g_params: Params;

// Storage buffers.
// Points are packed as float3 tightly (std430 rules will align to 16 for vec3 in struct;
// we avoid structs and store as array<vec4> style layout, but here we use array<vec4<f32>>
// with w unused to keep alignment simple.
var<storage, read> g_points: array<vec4<f32>>;

// Packed attributes: f32 array with row size = attr_dim = 1 + 3*(1+deg)^2.
// Layout per point row:
//   coeffs: sh_dim = 3 * sh_components
//   density: last scalar
var<storage, read> g_attributes: array<f32>;

// CSR adjacency.
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

// Output HDR image.
var g_out: texture_storage_2d<rgba16float, write>;

// ---- Quaternion helpers ----

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// ---- SH evaluation (matches existing gaussian shader basis constants) ----

fn sh_basis_constants() -> array<f32, MAX_SH_COMPONENTS> {
    // These constants match the ones used in `blade-gaussian/examples/shader.wgsl`.
    return array<f32, MAX_SH_COMPONENTS>(
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
        -0.5900435899266435
    );
}

fn sh_component_count(deg: u32) -> u32 {
    // (1+deg)^2
    let d = deg + 1u;
    return d * d;
}

fn eval_sh_rgb(point_idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let deg = g_params.sh_degree;
    let comps = sh_component_count(deg);
    // Clamp to MAX_SH_COMPONENTS for safety.
    let clamped_comps = min(comps, MAX_SH_COMPONENTS);

    let SH = sh_basis_constants();
    let d2 = dir * dir;

    // Attribute row layout:
    // base = point_idx * attr_dim
    // coefficient for component i, channel c is at base + 3*i + c
    // density at base + 3*clamped_comps
    let sh_dim = 3u * clamped_comps;
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;

    // L0
    var color = vec3<f32>(
        SH[0] * g_attributes[base + 0u],
        SH[0] * g_attributes[base + 1u],
        SH[0] * g_attributes[base + 2u]
    );

    if (deg >= 1u && clamped_comps >= 4u) {
        // Basis matches gaussian shader:
        // 1: y, 2: z, 3: x
        let y = dir.y;
        let z = dir.z;
        let x = dir.x;

        color += vec3<f32>(
            SH[1] * g_attributes[base + 3u + 0u] * y,
            SH[1] * g_attributes[base + 3u + 1u] * y,
            SH[1] * g_attributes[base + 3u + 2u] * y
        );
        color += vec3<f32>(
            SH[2] * g_attributes[base + 6u + 0u] * z,
            SH[2] * g_attributes[base + 6u + 1u] * z,
            SH[2] * g_attributes[base + 6u + 2u] * z
        );
        color += vec3<f32>(
            SH[3] * g_attributes[base + 9u + 0u] * x,
            SH[3] * g_attributes[base + 9u + 1u] * x,
            SH[3] * g_attributes[base + 9u + 2u] * x
        );
    }

    if (deg >= 2u && clamped_comps >= 9u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        // component 4: x*y
        color += vec3<f32>(
            SH[4] * g_attributes[base + 12u + 0u] * x * y,
            SH[4] * g_attributes[base + 12u + 1u] * x * y,
            SH[4] * g_attributes[base + 12u + 2u] * x * y
        );
        // component 5: y*z
        color += vec3<f32>(
            SH[5] * g_attributes[base + 15u + 0u] * y * z,
            SH[5] * g_attributes[base + 15u + 1u] * y * z,
            SH[5] * g_attributes[base + 15u + 2u] * y * z
        );
        // component 6: (3z^2 - 1)
        color += vec3<f32>(
            SH[6] * g_attributes[base + 18u + 0u] * (3.0 * zz - 1.0),
            SH[6] * g_attributes[base + 18u + 1u] * (3.0 * zz - 1.0),
            SH[6] * g_attributes[base + 18u + 2u] * (3.0 * zz - 1.0)
        );
        // component 7: x*z
        color += vec3<f32>(
            SH[7] * g_attributes[base + 21u + 0u] * x * z,
            SH[7] * g_attributes[base + 21u + 1u] * x * z,
            SH[7] * g_attributes[base + 21u + 2u] * x * z
        );
        // component 8: (x^2 - y^2)
        color += vec3<f32>(
            SH[8] * g_attributes[base + 24u + 0u] * (xx - yy),
            SH[8] * g_attributes[base + 24u + 1u] * (xx - yy),
            SH[8] * g_attributes[base + 24u + 2u] * (xx - yy)
        );
    }

    if (deg >= 3u && clamped_comps >= 16u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        // component 9
        color += vec3<f32>(
            SH[9] * g_attributes[base + 27u + 0u] * y * (3.0 * xx - yy),
            SH[9] * g_attributes[base + 27u + 1u] * y * (3.0 * xx - yy),
            SH[9] * g_attributes[base + 27u + 2u] * y * (3.0 * xx - yy)
        );
        // component 10
        color += vec3<f32>(
            SH[10] * g_attributes[base + 30u + 0u] * x * y * z,
            SH[10] * g_attributes[base + 30u + 1u] * x * y * z,
            SH[10] * g_attributes[base + 30u + 2u] * x * y * z
        );
        // component 11
        color += vec3<f32>(
            SH[11] * g_attributes[base + 33u + 0u] * y * (5.0 * zz - 1.0),
            SH[11] * g_attributes[base + 33u + 1u] * y * (5.0 * zz - 1.0),
            SH[11] * g_attributes[base + 33u + 2u] * y * (5.0 * zz - 1.0)
        );
        // component 12
        color += vec3<f32>(
            SH[12] * g_attributes[base + 36u + 0u] * z * (5.0 * zz - 3.0),
            SH[12] * g_attributes[base + 36u + 1u] * z * (5.0 * zz - 3.0),
            SH[12] * g_attributes[base + 36u + 2u] * z * (5.0 * zz - 3.0)
        );
        // component 13
        color += vec3<f32>(
            SH[13] * g_attributes[base + 39u + 0u] * x * (5.0 * zz - 1.0),
            SH[13] * g_attributes[base + 39u + 1u] * x * (5.0 * zz - 1.0),
            SH[13] * g_attributes[base + 39u + 2u] * x * (5.0 * zz - 1.0)
        );
        // component 14
        color += vec3<f32>(
            SH[14] * g_attributes[base + 42u + 0u] * z * (xx - yy),
            SH[14] * g_attributes[base + 42u + 1u] * z * (xx - yy),
            SH[14] * g_attributes[base + 42u + 2u] * z * (xx - yy)
        );
        // component 15
        color += vec3<f32>(
            SH[15] * g_attributes[base + 45u + 0u] * x * (xx - 3.0 * yy),
            SH[15] * g_attributes[base + 45u + 1u] * x * (xx - 3.0 * yy),
            SH[15] * g_attributes[base + 45u + 2u] * x * (xx - 3.0 * yy)
        );
    }

    // Upstream adds 0.5 bias to color; keep similar behavior for visibility.
    return 0.5 + color;
}

fn load_density(point_idx: u32) -> f32 {
    let deg = g_params.sh_degree;
    let comps = min(sh_component_count(deg), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;
    return g_attributes[base + sh_dim];
}

// ---- Voronoi traversal ----

fn trace_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var t0 = 0.0;
    var transmittance = 1.0;
    var accum_rgb = vec3<f32>(0.0);

    var current = g_params.start_point;
    var current_pos = g_points[current].xyz;

    // Main traversal loop
    var steps: u32 = 0u;
    loop {
        steps += 1u;
        if (steps > g_params.max_steps) {
            break;
        }
        if (transmittance <= g_params.weight_threshold) {
            break;
        }

        let begin = g_adjacency_offsets[current];
        let end = g_adjacency_offsets[current + 1u];
        let num_faces = end - begin;

        // WGSL/naga doesn't allow abstract-float `inf` here; use a large finite value.
        var t1: f32 = 3.402823466e+38; // ~f32::MAX
        var next_face: u32 = 0xffffffffu;

        // Scan neighbors
        var j: u32 = 0u;
        loop {
            if (j >= num_faces) { break; }
            let next_idx = g_adjacency[begin + j];
            let next_pos = g_points[next_idx].xyz;
            let offset = next_pos - current_pos;

            // Bisector plane
            let face_origin = current_pos + 0.5 * offset;
            let face_normal = offset;

            let dp = dot(face_normal, ray_dir);
            if (dp > 0.0) {
                let t = dot(face_origin - ray_origin, face_normal) / dp;
                if (t < t1) {
                    t1 = t;
                    next_face = j;
                }
            }
            j += 1u;
        }

        if (next_face == 0xffffffffu) {
            break;
        }

        let next_idx = g_adjacency[begin + next_face];
        let next_pos = g_points[next_idx].xyz;

        if (t1 > t0) {
            let s = load_density(current);
            if (s > 1e-6) {
                let dt = max(t1 - t0, 0.0);
                let alpha = 1.0 - exp(-s * dt);
                let w = transmittance * alpha;
                let rgb = eval_sh_rgb(current, ray_dir);
                accum_rgb += w * rgb;
                transmittance *= (1.0 - alpha);
            }
        }

        t0 = max(t0, t1);
        current = next_idx;
        current_pos = next_pos;

        // Optional camera depth clamp
        if (t0 > g_camera.depth) {
            break;
        }
    }

    return vec4<f32>(accum_rgb, 1.0 - transmittance);
}

// ---- Entry point ----

@compute @workgroup_size(8, 8, 1)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }

    // NDC in [-1, 1]
    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);

    // Match the gaussian shader camera model:
    // local_dir = (ndc * tan(0.5*fov), 1)
    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>(ndc * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));
    let ray_origin = g_camera.position;

    let rgba = trace_ray(ray_origin, ray_dir);

    // Write HDR output (no tonemap here; tonemap in a later pass).
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
