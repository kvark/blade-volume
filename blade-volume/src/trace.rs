//! CPU forward tracer for RadFoam / Power Foam point clouds.
//!
//! This is the reference implementation that mirrors `shaders/radfoam_trace.wgsl`
//! step-for-step. It exists for three purposes:
//!
//! 1. **Correctness reference** for the GPU compute tracer. The GPU-vs-CPU test
//!    in this repo asserts pixel-level agreement on synthetic fixtures.
//! 2. **Forward pass for training**. `blade-volume-train` evaluates the trace
//!    on the current point cloud to produce predicted pixels for loss
//!    computation against ground-truth photographs.
//! 3. **Sanity-check renderer** when no GPU is available (CI, headless tools).
//!
//! What it implements (forward-only, deterministic, single-precision):
//! - Voronoi / power-diagram cell traversal over points with CSR adjacency.
//! - Plane selection identical to upstream RadFoam / Power Foam: the *radical
//!   plane* between weighted spheres degenerates to the standard bisector when
//!   both radii are zero.
//! - Segment-wise volumetric integration per cell with piecewise constant
//!   density (`alpha = 1 - exp(-s * dt)`).
//! - Termination on weight threshold / step cap / no exit face / `t0 > depth`.
//!
//! SH evaluation supports degrees 0..=3 with the same basis constants as the
//! WGSL shader. The renderer's `0.5 +` bias is applied here too so the CPU and
//! GPU paths produce identical pixels.

use crate::PointCloudModel;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ray {
    pub origin: glam::Vec3,
    pub direction: glam::Vec3,
}

/// Controls how RGB is produced per traversed cell.
#[derive(Clone, Copy, Debug)]
pub enum EvalMode {
    /// Ignore SH coefficients; use a constant RGB for every visited cell.
    ConstantRgb(glam::Vec3),
    /// Evaluate the model's packed SH coefficients in the ray direction.
    Sh,
}

#[derive(Clone, Copy, Debug)]
pub struct TraceSettings {
    pub weight_threshold: f32,
    pub max_steps: u32,
    pub start_point: u32,
    pub depth: f32,
    pub eval_mode: EvalMode,
}

impl Default for TraceSettings {
    fn default() -> Self {
        Self {
            weight_threshold: 0.001,
            max_steps: 1024,
            start_point: 0,
            depth: 10_000.0,
            eval_mode: EvalMode::ConstantRgb(glam::Vec3::splat(1.0)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TraceResult {
    /// RGB plus accumulated alpha (`1 - transmittance`).
    pub rgba: glam::Vec4,
    /// Number of traversal steps actually executed.
    pub steps: u32,
    /// Index of the last cell entered (useful for "warm-start" follow-up rays).
    pub last_point: u32,
    /// Parametric `t` at which traversal stopped.
    pub t_end: f32,
}

fn sh_component_count(deg: u32) -> u32 {
    let d = deg + 1;
    d * d
}

fn read_density(model: &PointCloudModel, point_idx: u32) -> f32 {
    model.points[point_idx as usize].w
}

/// Per-point radius; `0` when the model has no `radii` (plain RadFoam, in
/// which case the radical plane degenerates to the standard Voronoi bisector).
fn read_radius(model: &PointCloudModel, point_idx: u32) -> f32 {
    model
        .radii
        .as_deref()
        .map_or(0.0, |r| r[point_idx as usize])
}

/// SH basis constants for degrees up to 3 (16 components). These match the
/// constants in `shaders/sh_eval.wgsl`.
fn sh_basis_constants() -> [f32; 16] {
    [
        0.282_094_8,
        -0.488_602_52,
        0.488_602_52,
        -0.488_602_52,
        1.092_548_5,
        -1.092_548_5,
        0.315_391_57,
        -1.092_548_5,
        0.546_274_24,
        -0.590_043_6,
        2.890_611_4,
        -0.457_045_8,
        0.373_176_34,
        -0.457_045_8,
        1.445_305_7,
        -0.590_043_6,
    ]
}

/// Evaluate SH RGB for a single point. Mirrors the WGSL implementation.
/// For degree < 3 the extra coefficients are ignored. The `0.5 +` bias matches
/// `shaders/radfoam.wgsl::rf_get_color`.
pub fn eval_rgb_sh(model: &PointCloudModel, point_idx: u32, dir: glam::Vec3) -> glam::Vec3 {
    let deg = (model.sh_degree as u32).min(3);
    let comps = sh_component_count(deg).min(16);

    let sh = sh_basis_constants();
    let d2 = dir * dir;

    let sh_dim = 3 * comps;
    let base = (point_idx as usize) * (sh_dim as usize);

    let mut color = glam::Vec3::ZERO;

    if comps >= 1 {
        let c0 = sh[0];
        color.x += c0 * model.sh_coefficients[base];
        color.y += c0 * model.sh_coefficients[base + 1];
        color.z += c0 * model.sh_coefficients[base + 2];
    }

    if deg >= 1 && comps >= 4 {
        let y = dir.y;
        let z = dir.z;
        let x = dir.x;

        let c1 = sh[1];
        color.x += c1 * model.sh_coefficients[base + 3] * y;
        color.y += c1 * model.sh_coefficients[base + 4] * y;
        color.z += c1 * model.sh_coefficients[base + 5] * y;

        let c2 = sh[2];
        color.x += c2 * model.sh_coefficients[base + 6] * z;
        color.y += c2 * model.sh_coefficients[base + 7] * z;
        color.z += c2 * model.sh_coefficients[base + 8] * z;

        let c3 = sh[3];
        color.x += c3 * model.sh_coefficients[base + 9] * x;
        color.y += c3 * model.sh_coefficients[base + 10] * x;
        color.z += c3 * model.sh_coefficients[base + 11] * x;
    }

    if deg >= 2 && comps >= 9 {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        let c4 = sh[4];
        color.x += c4 * model.sh_coefficients[base + 12] * x * y;
        color.y += c4 * model.sh_coefficients[base + 13] * x * y;
        color.z += c4 * model.sh_coefficients[base + 14] * x * y;

        let c5 = sh[5];
        color.x += c5 * model.sh_coefficients[base + 15] * y * z;
        color.y += c5 * model.sh_coefficients[base + 16] * y * z;
        color.z += c5 * model.sh_coefficients[base + 17] * y * z;

        let c6 = sh[6];
        let t6 = 3.0 * zz - 1.0;
        color.x += c6 * model.sh_coefficients[base + 18] * t6;
        color.y += c6 * model.sh_coefficients[base + 19] * t6;
        color.z += c6 * model.sh_coefficients[base + 20] * t6;

        let c7 = sh[7];
        color.x += c7 * model.sh_coefficients[base + 21] * x * z;
        color.y += c7 * model.sh_coefficients[base + 22] * x * z;
        color.z += c7 * model.sh_coefficients[base + 23] * x * z;

        let c8 = sh[8];
        let t8 = xx - yy;
        color.x += c8 * model.sh_coefficients[base + 24] * t8;
        color.y += c8 * model.sh_coefficients[base + 25] * t8;
        color.z += c8 * model.sh_coefficients[base + 26] * t8;
    }

    if deg >= 3 && comps >= 16 {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        let c9 = sh[9];
        let t9 = y * (3.0 * xx - yy);
        color.x += c9 * model.sh_coefficients[base + 27] * t9;
        color.y += c9 * model.sh_coefficients[base + 28] * t9;
        color.z += c9 * model.sh_coefficients[base + 29] * t9;

        let c10 = sh[10];
        let t10 = x * y * z;
        color.x += c10 * model.sh_coefficients[base + 30] * t10;
        color.y += c10 * model.sh_coefficients[base + 31] * t10;
        color.z += c10 * model.sh_coefficients[base + 32] * t10;

        let c11 = sh[11];
        let t11 = y * (5.0 * zz - 1.0);
        color.x += c11 * model.sh_coefficients[base + 33] * t11;
        color.y += c11 * model.sh_coefficients[base + 34] * t11;
        color.z += c11 * model.sh_coefficients[base + 35] * t11;

        let c12 = sh[12];
        let t12 = z * (5.0 * zz - 3.0);
        color.x += c12 * model.sh_coefficients[base + 36] * t12;
        color.y += c12 * model.sh_coefficients[base + 37] * t12;
        color.z += c12 * model.sh_coefficients[base + 38] * t12;

        let c13 = sh[13];
        let t13 = x * (5.0 * zz - 1.0);
        color.x += c13 * model.sh_coefficients[base + 39] * t13;
        color.y += c13 * model.sh_coefficients[base + 40] * t13;
        color.z += c13 * model.sh_coefficients[base + 41] * t13;

        let c14 = sh[14];
        let t14 = z * (xx - yy);
        color.x += c14 * model.sh_coefficients[base + 42] * t14;
        color.y += c14 * model.sh_coefficients[base + 43] * t14;
        color.z += c14 * model.sh_coefficients[base + 44] * t14;

        let c15 = sh[15];
        let t15 = x * (xx - 3.0 * yy);
        color.x += c15 * model.sh_coefficients[base + 45] * t15;
        color.y += c15 * model.sh_coefficients[base + 46] * t15;
        color.z += c15 * model.sh_coefficients[base + 47] * t15;
    }

    // Match `rf_get_color` in radfoam.wgsl which adds a 0.5 visibility bias.
    0.5 + color
}

/// Trace a single ray through the point cloud, returning the integrated
/// colour, traversal step count, last cell entered, and stop parameter.
pub fn trace_one_ray(model: &PointCloudModel, ray: Ray, settings: TraceSettings) -> TraceResult {
    assert!(!model.points.is_empty(), "model has no points");
    assert!(
        (settings.start_point as usize) < model.points.len(),
        "start_point out of bounds"
    );

    let adjacency = model
        .adjacency
        .as_ref()
        .expect("trace_one_ray requires adjacency");

    let mut dir = ray.direction;
    let dir_len = dir.length();
    if dir_len <= 0.0 || !dir_len.is_finite() {
        return TraceResult {
            rgba: glam::Vec4::ZERO,
            steps: 0,
            last_point: settings.start_point,
            t_end: 0.0,
        };
    }
    dir /= dir_len;

    let mut t0 = 0.0f32;
    let mut transmittance = 1.0f32;
    let mut accum_rgb = glam::Vec3::ZERO;

    let mut current = settings.start_point;
    let p = model.points[current as usize];
    let mut current_pos = glam::Vec3::new(p.x, p.y, p.z);
    let mut current_radius = read_radius(model, current);

    let mut steps = 0u32;

    while steps < settings.max_steps {
        steps += 1;
        if transmittance <= settings.weight_threshold || t0 > settings.depth {
            break;
        }

        let begin = adjacency.offsets[current as usize] as usize;
        let end = adjacency.offsets[current as usize + 1] as usize;

        let mut best_t1 = f32::MAX;
        let mut next_face: Option<usize> = None;
        let r_i_sq = current_radius * current_radius;

        for (j, &next_idx_u32) in adjacency.neighbors[begin..end].iter().enumerate() {
            let next_idx = next_idx_u32 as usize;
            let next_p = model.points[next_idx];
            let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);
            let r_j = read_radius(model, next_idx_u32);
            let offset = next_pos - current_pos;

            // Radical plane between weighted spheres; reduces to the bisector
            // when both radii are zero. Matches shaders/radfoam_trace.wgsl.
            let dsq = offset.length_squared().max(1e-20);
            let shift = 0.5 + 0.5 * (r_i_sq - r_j * r_j) / dsq;
            let face_origin = current_pos + shift * offset;
            let face_normal = offset;

            let dp = face_normal.dot(dir);
            if dp > 0.0 {
                let t = (face_origin - ray.origin).dot(face_normal) / dp;
                if t.is_finite() && t < best_t1 {
                    best_t1 = t;
                    next_face = Some(j);
                }
            }
        }

        let Some(j) = next_face else {
            break;
        };

        let next_idx_u32 = adjacency.neighbors[begin + j];
        let next_idx = next_idx_u32;
        let next_p = model.points[next_idx as usize];
        let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);

        if best_t1 > t0 {
            let s = read_density(model, current);
            if s > 1e-6 {
                let dt = (best_t1 - t0).max(0.0);
                let alpha = 1.0 - (-s * dt).exp();
                let w = transmittance * alpha;
                let rgb = match settings.eval_mode {
                    EvalMode::ConstantRgb(c) => c,
                    EvalMode::Sh => eval_rgb_sh(model, current, dir),
                };
                accum_rgb += w * rgb;
                transmittance *= 1.0 - alpha;
            }
        }

        t0 = t0.max(best_t1);
        current = next_idx;
        current_pos = next_pos;
        current_radius = read_radius(model, next_idx);
    }

    TraceResult {
        rgba: glam::Vec4::new(accum_rgb.x, accum_rgb.y, accum_rgb.z, 1.0 - transmittance),
        steps,
        last_point: current,
        t_end: t0,
    }
}
