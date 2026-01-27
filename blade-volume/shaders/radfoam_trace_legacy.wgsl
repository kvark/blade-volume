// RadFoam-style compute tracer (MVP)
//
// Baseline implementation:
// - Keep this file easy to read and self-contained at the *entry point* level.
// - Reuse the shared `radfoam_trace.wgsl` module for the traversal loop/integration math,
//   via small accessor functions defined below.
//
// NOTE: This file intentionally preserves the existing bindings/uniform layout used by the viewer.
// It is not meant to be the fastest path; it is meant to be a stable baseline.
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
    init_steps: u32,         // unused by legacy path, but kept for shared layout
    weight_threshold: f32,   // stop when transmittance <= threshold
    max_steps: u32,          // maximum cell transitions
    start_point: u32,        // starting cell index for all rays (MVP)
    // debug mode
    debug_mode: u32,         // 0 = off, 1 = cell density visualization
    // padding
    pad0: u32,
    pad1: u32,
};

var<uniform> g_camera: Camera;
var<uniform> g_params: Params;

// Storage buffers.
var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

// Output HDR image.
var g_out: texture_storage_2d<rgba16float, write>;

// ---- Quaternion helpers ----

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// ---- SH evaluation (baseline keeps its own SH constants) ----

fn sh_basis_constants() -> array<f32, MAX_SH_COMPONENTS> {
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
    let d = deg + 1u;
    return d * d;
}

fn eval_sh_rgb(point_idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let deg = g_params.sh_degree;
    let comps = sh_component_count(deg);
    let clamped_comps = min(comps, MAX_SH_COMPONENTS);

    let SH = sh_basis_constants();
    let d2 = dir * dir;

    let sh_dim = 3u * clamped_comps;
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;

    var color = vec3<f32>(
        SH[0] * g_attributes[base + 0u],
        SH[0] * g_attributes[base + 1u],
        SH[0] * g_attributes[base + 2u]
    );

    if (deg >= 1u && clamped_comps >= 4u) {
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

        color += vec3<f32>(
            SH[4] * g_attributes[base + 12u + 0u] * x * y,
            SH[4] * g_attributes[base + 12u + 1u] * x * y,
            SH[4] * g_attributes[base + 12u + 2u] * x * y
        );
        color += vec3<f32>(
            SH[5] * g_attributes[base + 15u + 0u] * y * z,
            SH[5] * g_attributes[base + 15u + 1u] * y * z,
            SH[5] * g_attributes[base + 15u + 2u] * y * z
        );
        color += vec3<f32>(
            SH[6] * g_attributes[base + 18u + 0u] * (3.0 * zz - 1.0),
            SH[6] * g_attributes[base + 18u + 1u] * (3.0 * zz - 1.0),
            SH[6] * g_attributes[base + 18u + 2u] * (3.0 * zz - 1.0)
        );
        color += vec3<f32>(
            SH[7] * g_attributes[base + 21u + 0u] * x * z,
            SH[7] * g_attributes[base + 21u + 1u] * x * z,
            SH[7] * g_attributes[base + 21u + 2u] * x * z
        );
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

        color += vec3<f32>(
            SH[9] * g_attributes[base + 27u + 0u] * y * (3.0 * xx - yy),
            SH[9] * g_attributes[base + 27u + 1u] * y * (3.0 * xx - yy),
            SH[9] * g_attributes[base + 27u + 2u] * y * (3.0 * xx - yy)
        );
        color += vec3<f32>(
            SH[10] * g_attributes[base + 30u + 0u] * x * y * z,
            SH[10] * g_attributes[base + 30u + 1u] * x * y * z,
            SH[10] * g_attributes[base + 30u + 2u] * x * y * z
        );
        color += vec3<f32>(
            SH[11] * g_attributes[base + 33u + 0u] * y * (5.0 * zz - 1.0),
            SH[11] * g_attributes[base + 33u + 1u] * y * (5.0 * zz - 1.0),
            SH[11] * g_attributes[base + 33u + 2u] * y * (5.0 * zz - 1.0)
        );
        color += vec3<f32>(
            SH[12] * g_attributes[base + 36u + 0u] * z * (5.0 * zz - 3.0),
            SH[12] * g_attributes[base + 36u + 1u] * z * (5.0 * zz - 3.0),
            SH[12] * g_attributes[base + 36u + 2u] * z * (5.0 * zz - 3.0)
        );
        color += vec3<f32>(
            SH[13] * g_attributes[base + 39u + 0u] * x * (5.0 * zz - 1.0),
            SH[13] * g_attributes[base + 39u + 1u] * x * (5.0 * zz - 1.0),
            SH[13] * g_attributes[base + 39u + 2u] * x * (5.0 * zz - 1.0)
        );
        color += vec3<f32>(
            SH[14] * g_attributes[base + 42u + 0u] * z * (xx - yy),
            SH[14] * g_attributes[base + 42u + 1u] * z * (xx - yy),
            SH[14] * g_attributes[base + 42u + 2u] * z * (xx - yy)
        );
        color += vec3<f32>(
            SH[15] * g_attributes[base + 45u + 0u] * x * (xx - 3.0 * yy),
            SH[15] * g_attributes[base + 45u + 1u] * x * (xx - 3.0 * yy),
            SH[15] * g_attributes[base + 45u + 2u] * x * (xx - 3.0 * yy)
        );
    }

    return 0.5 + color;
}
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

// Debug mode constants
const DEBUG_MODE_OFF: u32 = 0u;
const DEBUG_MODE_CELL_DENSITY: u32 = 1u;

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

// ---- RadFoam accessors for shared traversal module ----

// NOTE: This baseline maps the legacy bindings into the accessor API expected by `radfoam_trace.wgsl`.

fn rf_get_point(idx: u32) -> vec3<f32> {
    return g_points[idx].xyz;
}

fn rf_get_density(idx: u32) -> f32 {
    return load_density(idx);
}

fn rf_get_color(idx: u32, dir: vec3<f32>) -> vec3<f32> {
    return eval_sh_rgb(idx, dir);
}

fn rf_adjacency_begin(idx: u32) -> u32 {
    return g_adjacency_offsets[idx];
}

fn rf_adjacency_end(idx: u32) -> u32 {
    return g_adjacency_offsets[idx + 1u];
}

fn rf_get_neighbor(adj_idx: u32) -> u32 {
    return g_adjacency[adj_idx];
}

// Shared traversal module (expects the accessors above).
// #include "radfoam_trace.wgsl"

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

    // Map legacy uniform into shared-trace params.
    var tp: RadFoamTraceParams;
    tp.start_point = g_params.start_point;
    tp.max_steps = g_params.max_steps;
    tp.weight_threshold = g_params.weight_threshold;
    tp.pad = 0u;

    let result = radfoam_trace(ray_origin, ray_dir, 0.0, g_camera.depth, tp);

    // Debug mode: cell density visualization
    if (g_params.debug_mode == DEBUG_MODE_CELL_DENSITY) {
        let density = f32(result.cells_visited) / 100.0;
        let debug_color = heatmap_color(density);
        textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(debug_color, 1.0));
        return;
    }

    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), result.color);
}
