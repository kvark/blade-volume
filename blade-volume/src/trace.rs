//! CPU forward tracers for point-cloud rendering backends.
//!
//! The RadFoam / PowerFoam implementation mirrors
//! `shaders/radfoam_trace.wgsl` step-for-step. The Gaussian implementation is
//! an exhaustive maximum-response-depth oracle for
//! `shaders/gaussian_trace.wgsl`. They exist for three purposes:
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
//! - PowerFoam segments are clipped to each site's support sphere; an absent
//!   `radii` array retains RadFoam's unbounded Voronoi cells.
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
    /// Skip colour evaluation and accumulation. Alpha and depth statistics are
    /// still integrated normally.
    Opacity,
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
    /// Distance to the single segment that absorbed the most light, and how
    /// much of the ray that segment took.
    ///
    /// Where the surface is, as far as a density field has one — and the mode
    /// rather than the mean or the median, which were both tried and are both
    /// worse. A field trained to reproduce photographs has long thin tails: two
    /// views of one wall agree about the wall and disagree about where their
    /// tails end, so a cloud fused from mean depths comes out many sheets
    /// thick. Fusing modes instead took the share of surfels that were ever the
    /// frontmost thing in any view from 13% to 75%.
    ///
    /// `peak_weight` is the part neither of the others has. A room's field is
    /// genuinely foggy — every ray reaches full absorption eventually — so mean,
    /// median and mode alike return a confident position in mid-air for a ray
    /// that passed through haze and met nothing. A low peak is that ray
    /// admitting it, and is the only available signal that tells a wall from a
    /// volume of dust.
    pub depth_mode: f32,
    pub peak_weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GaussianTraceSettings {
    /// Discard particles and ray responses at or below this opacity. This also
    /// defines each Gaussian's finite ellipsoidal support, matching the GPU
    /// acceleration proxy construction.
    pub min_opacity: f32,
    /// Stop front-to-back compositing once transmittance reaches this value.
    pub min_transmittance: f32,
    pub t_start: f32,
    pub t_end: f32,
}

impl Default for GaussianTraceSettings {
    fn default() -> Self {
        Self {
            min_opacity: 0.01,
            min_transmittance: 0.01,
            t_start: 0.0,
            t_end: 10_000.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GaussianTraceResult {
    /// RGB plus accumulated alpha (`1 - transmittance`).
    pub rgba: glam::Vec4,
    /// Number of Gaussian responses composited before early termination.
    pub hits: u32,
}

#[derive(Clone, Copy, Debug)]
struct GaussianRayHit {
    t: f32,
    index: u32,
    alpha: f32,
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

struct ExitQuery {
    current_pos: glam::Vec3,
    current_radius: f32,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    t0: f32,
    depth: f32,
}

#[inline]
fn find_exit_face<const WEIGHTED: bool>(
    points: &[glam::Vec4],
    radii: &[f32],
    neighbours: &[u32],
    query: &ExitQuery,
) -> (f32, Option<usize>) {
    // The const branch keeps plain RadFoam's hot neighbour loop free of the
    // radius loads, squared distance, and radical-plane division.
    let mut best_t1 = query.depth;
    let mut next_face = None;
    let r_i_sq = query.current_radius * query.current_radius;
    for (face, &next_idx_u32) in neighbours.iter().enumerate() {
        let next_idx = next_idx_u32 as usize;
        let next_p = points[next_idx];
        let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);
        let offset = next_pos - query.current_pos;
        let shift = if WEIGHTED {
            let r_j = radii[next_idx];
            let dsq = offset.length_squared().max(1e-20);
            0.5 + 0.5 * (r_i_sq - r_j * r_j) / dsq
        } else {
            0.5
        };
        let face_origin = query.current_pos + shift * offset;
        let dp = offset.dot(query.ray_dir);
        if dp > 0.0 {
            let t = (face_origin - query.ray_origin).dot(offset) / dp;
            if t.is_finite() && t > query.t0 && t < best_t1 {
                best_t1 = t;
                next_face = Some(face);
            }
        }
    }
    (best_t1, next_face)
}

/// Intersect a power-cell interval with its site's support sphere. Plain
/// RadFoam has no radii and therefore keeps the complete cell interval.
fn support_interval(
    bounded: bool,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    center: glam::Vec3,
    radius: f32,
    t0: f32,
    t1: f32,
) -> Option<(f32, f32)> {
    if !bounded {
        return (t1 > t0).then_some((t0, t1));
    }
    let oc = ray_origin - center;
    let a = ray_dir.length_squared();
    if !a.is_finite() || a <= 0.0 {
        return None;
    }
    let b = oc.dot(ray_dir);
    let c = oc.length_squared() - radius * radius;
    let discriminant = b * b - a * c;
    if discriminant <= 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let near = (-b - root) / a;
    let far = (-b + root) / a;
    let clipped_start = t0.max(near);
    let clipped_end = t1.min(far);
    (clipped_end > clipped_start).then_some((clipped_start, clipped_end))
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

    // RadFoam clamps the evaluated radiance before compositing. This is a
    // per-cell clamp, so applying it only to the final pixel is not
    // equivalent when several cells contribute to a ray.
    (0.5 + color).max(glam::Vec3::ZERO)
}

/// Exhaustively trace Gaussian ellipsoids and composite them in the 3DGRT
/// reference order: the depth of maximum response along the ray, with point
/// index as a deterministic tie-breaker.
///
/// This is intentionally an O(N log N) correctness oracle, not a CPU renderer.
/// The GPU uses finite support proxies and batched ray queries, but must produce
/// the same response set and ordering.
pub fn trace_gaussians(
    model: &PointCloudModel,
    ray: Ray,
    settings: GaussianTraceSettings,
) -> GaussianTraceResult {
    model
        .validate()
        .unwrap_or_else(|error| panic!("trace_gaussians: invalid model: {error}"));
    let transforms = model
        .transforms
        .as_ref()
        .expect("trace_gaussians requires Gaussian transforms");
    assert!(
        settings.min_opacity.is_finite() && settings.min_opacity > 0.0,
        "min_opacity must be finite and positive"
    );
    assert!(
        settings.min_transmittance.is_finite()
            && settings.min_transmittance >= 0.0
            && settings.min_transmittance <= 1.0,
        "min_transmittance must be finite and in [0, 1]"
    );
    assert!(
        settings.t_start.is_finite()
            && settings.t_end.is_finite()
            && settings.t_end > settings.t_start,
        "Gaussian trace interval must be finite and non-empty"
    );
    let direction_length = ray.direction.length();
    assert!(
        direction_length.is_finite() && direction_length > 0.0,
        "Gaussian ray direction must be finite and non-zero"
    );
    let direction = ray.direction / direction_length;

    let mut hits = Vec::new();
    for (index, point) in model.points.iter().copied().enumerate() {
        if point.w <= settings.min_opacity {
            continue;
        }
        let inverse_rotation = transforms.rotations[index].inverse();
        let mean = point.truncate();
        let local_origin = inverse_rotation * (ray.origin - mean) / transforms.scales[index];
        let local_direction = inverse_rotation * direction / transforms.scales[index];
        let denominator = local_direction.length_squared();
        let t = -local_origin.dot(local_direction) / denominator;
        if !(t > settings.t_start && t < settings.t_end) {
            continue;
        }
        let closest = local_origin + t * local_direction;
        let alpha = point.w * (-0.5 * closest.length_squared()).exp();
        if alpha >= settings.min_opacity {
            hits.push(GaussianRayHit {
                t,
                index: index as u32,
                alpha,
            });
        }
    }
    hits.sort_by(|a, b| a.t.total_cmp(&b.t).then(a.index.cmp(&b.index)));

    let mut transmittance = 1.0_f32;
    let mut radiance = glam::Vec3::ZERO;
    let mut hit_count = 0;
    for hit in hits {
        if transmittance <= settings.min_transmittance {
            break;
        }
        let color = eval_rgb_sh(model, hit.index, direction);
        radiance += hit.alpha * transmittance * color;
        transmittance *= 1.0 - hit.alpha;
        hit_count += 1;
    }

    GaussianTraceResult {
        rgba: radiance.extend(1.0 - transmittance),
        hits: hit_count,
    }
}

#[cfg(test)]
mod gaussian_tests {
    use super::*;

    fn dc(color: glam::Vec3) -> [f32; 3] {
        let coefficient = (color - 0.5) / 0.282_094_8;
        coefficient.to_array()
    }

    #[test]
    fn sh_color_clamps_negative_channels_before_compositing() {
        let mut sh_coefficients = Vec::new();
        sh_coefficients.extend_from_slice(&dc(glam::Vec3::new(-1.0, 0.25, 2.0)));
        let model = PointCloudModel {
            points: vec![glam::Vec4::new(0.0, 0.0, 1.0, 1.0)],
            sh_coefficients,
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
        };

        let color = eval_rgb_sh(&model, 0, glam::Vec3::Z);
        assert_eq!(color.x, 0.0);
        assert!((color.y - 0.25).abs() < 1.0e-6);
        assert!((color.z - 2.0).abs() < 1.0e-6);
    }

    fn two_gaussians() -> PointCloudModel {
        // Index order is deliberately far, near. The far particle is broad
        // enough that its support proxy begins before the near particle's,
        // while its maximum response is still later along the ray.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 4.0, 0.5),
            glam::Vec4::new(0.0, 0.0, 2.0, 0.5),
        ];
        let mut sh_coefficients = Vec::new();
        sh_coefficients.extend_from_slice(&dc(glam::Vec3::new(0.0, 0.0, 1.0)));
        sh_coefficients.extend_from_slice(&dc(glam::Vec3::new(1.0, 0.0, 0.0)));
        PointCloudModel {
            points,
            sh_coefficients,
            sh_degree: 0,
            transforms: Some(crate::Transforms {
                rotations: vec![glam::Quat::IDENTITY; 2],
                scales: vec![
                    glam::Vec3::new(1.0, 1.0, 1.0),
                    glam::Vec3::new(1.0, 1.0, 0.1),
                ],
            }),
            adjacency: None,
            radii: None,
        }
    }

    #[test]
    fn gaussian_trace_orders_maximum_response_not_proxy_entry() {
        let model = two_gaussians();
        let settings = GaussianTraceSettings {
            min_opacity: 0.01,
            min_transmittance: 0.0,
            t_start: 0.0,
            t_end: 10.0,
        };
        let support_radius = (2.0 * (0.5_f32 / settings.min_opacity).ln()).sqrt();
        let far_proxy_entry = 4.0 - support_radius;
        let near_proxy_entry = 2.0 - 0.1 * support_radius;
        assert!(far_proxy_entry < near_proxy_entry);

        let result = trace_gaussians(
            &model,
            Ray {
                origin: glam::Vec3::ZERO,
                direction: glam::Vec3::Z,
            },
            settings,
        );

        // Correct maximum-response order is near red at t=2, then far blue
        // at t=4: 0.5R + (1-0.5)*0.5B.
        let expected = glam::Vec4::new(0.5, 0.0, 0.25, 0.75);
        assert!((result.rgba - expected).abs().max_element() < 1.0e-6);
        assert_eq!(result.hits, 2);
    }

    fn hit_less(a: GaussianRayHit, b: GaussianRayHit) -> bool {
        a.t < b.t || (a.t == b.t && a.index < b.index)
    }

    fn next_batch(
        candidates: &[GaussianRayHit],
        cursor: Option<GaussianRayHit>,
        window: usize,
    ) -> Vec<GaussianRayHit> {
        let mut selected = Vec::new();
        for &candidate in candidates {
            if cursor.is_some_and(|value| !hit_less(value, candidate))
                || selected
                    .iter()
                    .any(|hit: &GaussianRayHit| hit.index == candidate.index)
            {
                continue;
            }
            let position = selected
                .iter()
                .position(|&hit| hit_less(candidate, hit))
                .unwrap_or(selected.len());
            if position < window {
                selected.insert(position, candidate);
                selected.truncate(window);
            }
        }
        selected
    }

    #[test]
    fn gaussian_batch_cursor_keeps_more_than_window_and_equal_depth_hits() {
        let expected: Vec<GaussianRayHit> = (0..13)
            .map(|index| GaussianRayHit {
                t: 1.0 + (index / 3) as f32,
                index,
                alpha: 0.1,
            })
            .collect();
        let mut candidates = Vec::new();
        for hit in expected.iter().rev() {
            // Front/back triangle candidates for the same instance arrive in
            // an arbitrary order and must collapse to one particle response.
            candidates.push(*hit);
            candidates.push(*hit);
        }

        let mut actual = Vec::new();
        let mut cursor = None;
        loop {
            let batch = next_batch(&candidates, cursor, 5);
            if batch.is_empty() {
                break;
            }
            cursor = batch.last().copied();
            actual.extend(batch);
        }

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.t, expected.t);
            assert_eq!(actual.index, expected.index);
        }
    }
}

/// One step of the trace: the cell we were *in* during the segment and the
/// segment's length along the ray.
#[derive(Clone, Copy, Debug, Default)]
pub struct PathEntry {
    pub cell: u32,
    pub dt: f32,
}

/// One recorded segment together with the local derivative of its length.
///
/// The three `dt_d_*` vectors differentiate `dt` with respect to the selected
/// site's `(x, y, z, radius)` and the two sites defining its entry and exit
/// radical planes. A repeated index is a sentinel for a fixed boundary:
/// `previous_cell == cell` means the ray starts at `t = 0`, while
/// `next_cell == cell` means it terminates at the configured depth. The
/// corresponding derivative is zero unless the site's support sphere is the
/// active clip.
#[derive(Clone, Copy, Debug, Default)]
pub struct PathJacobianEntry {
    pub previous_cell: u32,
    pub cell: u32,
    pub next_cell: u32,
    pub dt: f32,
    pub dt_d_previous: glam::Vec4,
    pub dt_d_current: glam::Vec4,
    pub dt_d_next: glam::Vec4,
}

/// Record-only result of a ray trace: the sequence of `(cell, dt)` segments
/// the ray covers, plus the normalised ray direction (used for view-dependent
/// SH evaluation downstream). Unlike [`TraceResult`] this performs no
/// volumetric integration — all colour/density/alpha computation is deferred
/// to the consumer.
///
/// Used as the input to the differentiable forward in `blade-volume-train`:
/// path traversal stays in non-differentiable native code, the per-segment
/// alpha/transmittance/colour integration runs through autodiff.
#[derive(Clone, Debug)]
pub struct PathResult {
    pub entries: Vec<PathEntry>,
    pub ray_dir: glam::Vec3,
}

/// Record-only path with exact local segment-length Jacobians.
///
/// The cell sequence remains discrete. Derivatives are for the active branch
/// of that fixed sequence: radical-plane entry/exit faces and PowerFoam
/// support-sphere clips. Re-record after geometry changes enough to alter a
/// face choice or clipping branch.
#[derive(Clone, Debug)]
pub struct PathJacobianResult {
    pub entries: Vec<PathJacobianEntry>,
    pub ray_dir: glam::Vec3,
}

fn vec4_xyz_w(xyz: glam::Vec3, w: f32) -> glam::Vec4 {
    glam::Vec4::new(xyz.x, xyz.y, xyz.z, w)
}

/// Derivative of one radical-plane intersection `t` with respect to the
/// current and adjacent sites. This is the closed form used by PowerFoam's
/// reference backward pass.
fn face_intersection_jacobians(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    current_pos: glam::Vec3,
    current_radius: f32,
    adjacent_pos: glam::Vec3,
    adjacent_radius: f32,
    t: f32,
) -> (glam::Vec4, glam::Vec4) {
    let normal = adjacent_pos - current_pos;
    let denominator = ray_dir.dot(normal);
    if !denominator.is_finite() || denominator.abs() <= 1e-20 {
        return (glam::Vec4::ZERO, glam::Vec4::ZERO);
    }
    let current_xyz = (ray_origin - current_pos + t * ray_dir) / denominator;
    let adjacent_xyz = (adjacent_pos - ray_origin - t * ray_dir) / denominator;
    (
        vec4_xyz_w(current_xyz, current_radius / denominator),
        vec4_xyz_w(adjacent_xyz, -adjacent_radius / denominator),
    )
}

/// Near/far sphere intersections and their derivatives with respect to the
/// sphere's `(center, radius)`. `ray_dir` must be normalized.
fn sphere_intersection_jacobians(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    center: glam::Vec3,
    radius: f32,
) -> Option<(f32, f32, glam::Vec4, glam::Vec4)> {
    let oc = ray_origin - center;
    let b = oc.dot(ray_dir);
    let discriminant = b * b - (oc.length_squared() - radius * radius);
    if discriminant <= 0.0 || !discriminant.is_finite() {
        return None;
    }
    let root = discriminant.sqrt();
    let perpendicular = oc - b * ray_dir;
    let root_d_center = perpendicular / root;
    let root_d_radius = radius / root;
    let base_d_center = ray_dir;
    Some((
        -b - root,
        -b + root,
        vec4_xyz_w(base_d_center - root_d_center, -root_d_radius),
        vec4_xyz_w(base_d_center + root_d_center, root_d_radius),
    ))
}

#[allow(clippy::too_many_arguments)]
fn path_interval_jacobian(
    model: &PointCloudModel,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    bounded: bool,
    previous_cell: u32,
    cell: u32,
    next_cell: u32,
    t0: f32,
    t1: f32,
) -> Option<PathJacobianEntry> {
    let current_pos = model.points[cell as usize].truncate();
    let current_radius = read_radius(model, cell);
    let mut start = t0;
    let mut end = t1;
    let mut start_sphere_jacobian = None;
    let mut end_sphere_jacobian = None;

    if bounded {
        let (sphere_near, sphere_far, near_jacobian, far_jacobian) =
            sphere_intersection_jacobians(ray_origin, ray_dir, current_pos, current_radius)?;
        if sphere_near > start {
            start = sphere_near;
            start_sphere_jacobian = Some(near_jacobian);
        }
        if sphere_far < end {
            end = sphere_far;
            end_sphere_jacobian = Some(far_jacobian);
        }
    }
    if end <= start || !start.is_finite() || !end.is_finite() {
        return None;
    }

    let mut dt_d_previous = glam::Vec4::ZERO;
    let mut dt_d_current = glam::Vec4::ZERO;
    let mut dt_d_next = glam::Vec4::ZERO;

    if let Some(jacobian) = start_sphere_jacobian {
        dt_d_current -= jacobian;
    } else if previous_cell != cell {
        let previous_pos = model.points[previous_cell as usize].truncate();
        let previous_radius = read_radius(model, previous_cell);
        let (current_jacobian, previous_jacobian) = face_intersection_jacobians(
            ray_origin,
            ray_dir,
            current_pos,
            current_radius,
            previous_pos,
            previous_radius,
            t0,
        );
        dt_d_current -= current_jacobian;
        dt_d_previous -= previous_jacobian;
    }

    if let Some(jacobian) = end_sphere_jacobian {
        dt_d_current += jacobian;
    } else if next_cell != cell {
        let next_pos = model.points[next_cell as usize].truncate();
        let next_radius = read_radius(model, next_cell);
        let (current_jacobian, next_jacobian) = face_intersection_jacobians(
            ray_origin,
            ray_dir,
            current_pos,
            current_radius,
            next_pos,
            next_radius,
            t1,
        );
        dt_d_current += current_jacobian;
        dt_d_next += next_jacobian;
    }

    Some(PathJacobianEntry {
        previous_cell,
        cell,
        next_cell,
        dt: end - start,
        dt_d_previous,
        dt_d_current,
        dt_d_next,
    })
}

/// Clip one PowerFoam support against all of its Cech-neighbor radical planes.
/// Returns the effective entry depth together with the exact local interval
/// derivative. Non-overlapping supports cannot win power distance inside this
/// support, so the Cech row contains every required clipping constraint.
fn powerfoam_splat_interval(
    model: &PointCloudModel,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    depth: f32,
    cell: u32,
) -> Option<(f32, PathJacobianEntry)> {
    let adjacency = model.adjacency.as_ref().unwrap();
    let current = model.points[cell as usize].truncate();
    let current_radius = read_radius(model, cell);
    if (current - ray_origin).length() < 4.0 * current_radius {
        return None;
    }
    let (sphere_near, sphere_far, _, _) =
        sphere_intersection_jacobians(ray_origin, ray_dir, current, current_radius)?;
    if sphere_far <= 0.0 || sphere_near >= depth {
        return None;
    }

    let mut face_near = 0.0_f32;
    let mut face_far = depth;
    let mut previous = cell;
    let mut next = cell;
    let begin = adjacency.offsets[cell as usize] as usize;
    let end = adjacency.offsets[cell as usize + 1] as usize;
    for &adjacent in &adjacency.neighbors[begin..end] {
        let adjacent_point = model.points[adjacent as usize].truncate();
        let adjacent_radius = read_radius(model, adjacent);
        let normal = adjacent_point - current;
        let distance_squared = normal.length_squared().max(1.0e-20);
        let shift = 0.5
            + 0.5 * (current_radius * current_radius - adjacent_radius * adjacent_radius)
                / distance_squared;
        let face_origin = current + shift * normal;
        let numerator = (face_origin - ray_origin).dot(normal);
        let denominator = ray_dir.dot(normal);
        if denominator > 1.0e-20 {
            let t = numerator / denominator;
            if t < face_far {
                face_far = t;
                next = adjacent;
            }
        } else if denominator < -1.0e-20 {
            let t = numerator / denominator;
            if t > face_near {
                face_near = t;
                previous = adjacent;
            }
        } else if numerator < 0.0 {
            return None;
        }
    }

    let effective_near = face_near.max(sphere_near);
    let effective_far = face_far.min(sphere_far);
    if effective_far <= effective_near || !effective_near.is_finite() || !effective_far.is_finite()
    {
        return None;
    }
    path_interval_jacobian(
        model, ray_origin, ray_dir, true, previous, cell, next, face_near, face_far,
    )
    .map(|entry| (effective_near, entry))
}

fn sorted_powerfoam_splat_intervals(
    model: &PointCloudModel,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    depth: f32,
    max_steps: u32,
) -> Vec<(f32, u32, PathJacobianEntry)> {
    let mut intervals = (0..model.points.len() as u32)
        .filter_map(|cell| {
            powerfoam_splat_interval(model, ray_origin, ray_dir, depth, cell)
                .map(|(near, entry)| (near, cell, entry))
        })
        .collect::<Vec<_>>();
    intervals.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    intervals.truncate(max_steps as usize);
    intervals
}

/// Record PowerFoam cells by compute-splat semantics rather than a global walk.
///
/// Every support sphere hit by the ray is clipped independently against its
/// Cech-neighbor radical planes, then the surviving disjoint intervals are
/// sorted front-to-back. This remains correct when the overlapping-ball graph
/// has multiple connected components; a camera-seeded adjacency walk cannot
/// discover those components. This CPU implementation is the correctness
/// oracle for the GPU training recorder.
pub fn record_powerfoam_splats_jacobians(
    model: &PointCloudModel,
    ray: Ray,
    settings: TraceSettings,
) -> PathJacobianResult {
    assert!(model.radii.is_some(), "PowerFoam splats require radii");
    assert!(
        model.adjacency.is_some(),
        "PowerFoam splats require adjacency"
    );
    let mut ray_dir = ray.direction;
    let direction_length = ray_dir.length();
    if direction_length <= 0.0 || !direction_length.is_finite() {
        return PathJacobianResult {
            entries: Vec::new(),
            ray_dir: glam::Vec3::ZERO,
        };
    }
    ray_dir /= direction_length;

    let intervals = sorted_powerfoam_splat_intervals(
        model,
        ray.origin,
        ray_dir,
        settings.depth,
        settings.max_steps,
    );
    PathJacobianResult {
        entries: intervals.into_iter().map(|(_, _, entry)| entry).collect(),
        ray_dir,
    }
}

/// Trace a PowerFoam ray with independent support splats and volumetric
/// front-to-back integration.
pub fn trace_powerfoam_splats(
    model: &PointCloudModel,
    ray: Ray,
    settings: TraceSettings,
) -> TraceResult {
    assert!(model.radii.is_some(), "PowerFoam splats require radii");
    assert!(
        model.adjacency.is_some(),
        "PowerFoam splats require adjacency"
    );
    let direction_length = ray.direction.length();
    if direction_length <= 0.0 || !direction_length.is_finite() {
        return TraceResult {
            rgba: glam::Vec4::ZERO,
            steps: 0,
            last_point: settings.start_point,
            t_end: 0.0,
            depth_mode: 0.0,
            peak_weight: 0.0,
        };
    }
    let ray_dir = ray.direction / direction_length;
    let intervals = sorted_powerfoam_splat_intervals(
        model,
        ray.origin,
        ray_dir,
        settings.depth,
        settings.max_steps,
    );
    let mut transmittance = 1.0_f32;
    let mut accum_rgb = glam::Vec3::ZERO;
    let mut depth_mode = 0.0_f32;
    let mut peak_weight = 0.0_f32;
    let mut steps = 0_u32;
    let mut last_point = settings.start_point;
    let mut t_end = 0.0_f32;
    for (near, cell, entry) in intervals {
        if transmittance <= settings.weight_threshold {
            break;
        }
        steps += 1;
        last_point = cell;
        t_end = near + entry.dt;
        let density = read_density(model, cell);
        if density <= 1.0e-6 {
            continue;
        }
        let alpha = 1.0 - (-density * entry.dt).exp();
        let weight = transmittance * alpha;
        match settings.eval_mode {
            EvalMode::Opacity => {}
            EvalMode::ConstantRgb(color) => accum_rgb += weight * color,
            EvalMode::Sh => accum_rgb += weight * eval_rgb_sh(model, cell, ray_dir),
        }
        if weight > peak_weight {
            peak_weight = weight;
            depth_mode = near + 0.5 * entry.dt;
        }
        transmittance *= 1.0 - alpha;
    }
    TraceResult {
        rgba: accum_rgb.extend(1.0 - transmittance),
        steps,
        last_point,
        t_end,
        depth_mode,
        peak_weight,
    }
}

/// Trace a ray and record per-segment `(cell, dt)` pairs without integrating.
/// Termination rules match [`trace_one_ray`] but the weight-threshold early-
/// out is disabled — the consumer decides when transmittance has decayed
/// enough to stop, so it needs the whole path.
pub fn record_path(model: &PointCloudModel, ray: Ray, settings: TraceSettings) -> PathResult {
    let result = record_path_jacobians(model, ray, settings);
    PathResult {
        entries: result
            .entries
            .into_iter()
            .map(|entry| PathEntry {
                cell: entry.cell,
                dt: entry.dt,
            })
            .collect(),
        ray_dir: result.ray_dir,
    }
}

/// Trace a ray and record exact local `dt` derivatives for its fixed cell walk.
///
/// This is the CPU correctness oracle for the differentiable GPU path
/// recorder. In particular, `previous_cell` tracks the actual entry face even
/// when one or more traversed power cells have no support-sphere overlap and
/// therefore emit no segment.
pub fn record_path_jacobians(
    model: &PointCloudModel,
    ray: Ray,
    settings: TraceSettings,
) -> PathJacobianResult {
    assert!(!model.points.is_empty(), "model has no points");
    assert!(
        (settings.start_point as usize) < model.points.len(),
        "start_point out of bounds"
    );

    let adjacency = model
        .adjacency
        .as_ref()
        .expect("record_path_jacobians requires adjacency");

    let mut entries = Vec::new();
    let mut dir = ray.direction;
    let dir_len = dir.length();
    if dir_len <= 0.0 || !dir_len.is_finite() {
        return PathJacobianResult {
            entries,
            ray_dir: glam::Vec3::ZERO,
        };
    }
    dir /= dir_len;

    let mut t0 = 0.0f32;
    let mut current = settings.start_point;
    let mut previous = current;
    let p = model.points[current as usize];
    let mut current_pos = glam::Vec3::new(p.x, p.y, p.z);
    let mut current_radius = read_radius(model, current);
    let bounded = model.radii.is_some();

    let mut steps = 0u32;
    while steps < settings.max_steps {
        steps += 1;
        if t0 > settings.depth {
            break;
        }

        let begin = adjacency.offsets[current as usize] as usize;
        let end = adjacency.offsets[current as usize + 1] as usize;

        let mut best_t1 = settings.depth;
        let mut next_face: Option<usize> = None;
        let r_i_sq = current_radius * current_radius;

        for (j, &next_idx_u32) in adjacency.neighbors[begin..end].iter().enumerate() {
            let next_idx = next_idx_u32 as usize;
            let next_p = model.points[next_idx];
            let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);
            let r_j = read_radius(model, next_idx_u32);
            let offset = next_pos - current_pos;
            let dsq = offset.length_squared().max(1e-20);
            let shift = 0.5 + 0.5 * (r_i_sq - r_j * r_j) / dsq;
            let face_origin = current_pos + shift * offset;
            let face_normal = offset;
            let dp = face_normal.dot(dir);
            if dp > 0.0 {
                let t = (face_origin - ray.origin).dot(face_normal) / dp;
                if t.is_finite() && t > t0 && t < best_t1 {
                    best_t1 = t;
                    next_face = Some(j);
                }
            }
        }

        let next_idx_u32 = next_face
            .map(|j| adjacency.neighbors[begin + j])
            .unwrap_or(current);
        if let Some(entry) = path_interval_jacobian(
            model,
            ray.origin,
            dir,
            bounded,
            previous,
            current,
            next_idx_u32,
            t0,
            best_t1,
        ) {
            entries.push(entry);
        }

        if next_face.is_none() {
            break;
        }

        t0 = t0.max(best_t1);
        previous = current;
        current = next_idx_u32;
        let next_p = model.points[next_idx_u32 as usize];
        let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);
        current_pos = next_pos;
        current_radius = read_radius(model, next_idx_u32);
    }

    PathJacobianResult {
        entries,
        ray_dir: dir,
    }
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
            depth_mode: 0.0,
            peak_weight: 0.0,
        };
    }
    dir /= dir_len;

    let mut t0 = 0.0f32;
    let mut transmittance = 1.0f32;
    let mut accum_rgb = glam::Vec3::ZERO;
    let mut depth_mode = 0.0f32;
    let mut peak_weight = 0.0f32;

    let mut current = settings.start_point;
    let p = model.points[current as usize];
    let mut current_pos = glam::Vec3::new(p.x, p.y, p.z);
    let mut current_radius = read_radius(model, current);
    let bounded = model.radii.is_some();

    let mut steps = 0u32;

    while steps < settings.max_steps {
        steps += 1;
        if transmittance <= settings.weight_threshold || t0 > settings.depth {
            break;
        }

        let begin = adjacency.offsets[current as usize] as usize;
        let end = adjacency.offsets[current as usize + 1] as usize;

        let neighbours = &adjacency.neighbors[begin..end];
        let exit_query = ExitQuery {
            current_pos,
            current_radius,
            ray_origin: ray.origin,
            ray_dir: dir,
            t0,
            depth: settings.depth,
        };
        let (best_t1, next_face) = match model.radii.as_deref() {
            Some(radii) => find_exit_face::<true>(&model.points, radii, neighbours, &exit_query),
            None => find_exit_face::<false>(&model.points, &[], neighbours, &exit_query),
        };

        if let Some((segment_start, segment_end)) = support_interval(
            bounded,
            ray.origin,
            dir,
            current_pos,
            current_radius,
            t0,
            best_t1,
        ) {
            let s = read_density(model, current);
            if s > 1e-6 {
                let dt = segment_end - segment_start;
                let alpha = 1.0 - (-s * dt).exp();
                let w = transmittance * alpha;
                match settings.eval_mode {
                    EvalMode::Opacity => {}
                    EvalMode::ConstantRgb(c) => accum_rgb += w * c,
                    EvalMode::Sh => accum_rgb += w * eval_rgb_sh(model, current, dir),
                }
                if w > peak_weight {
                    peak_weight = w;
                    depth_mode = 0.5 * (segment_start + segment_end);
                }
                // The middle of the segment, which is where a uniform-density
                // cell puts its mass. Using the entry point instead biases
                // every surface towards the camera by half a cell.
                transmittance *= 1.0 - alpha;
            }
        }

        let Some(j) = next_face else {
            t0 = best_t1;
            break;
        };

        let next_idx_u32 = neighbours[j];
        let next_idx = next_idx_u32;
        let next_p = model.points[next_idx as usize];
        let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);

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
        depth_mode,
        peak_weight,
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;

    /// Tetrahedron of unit-density cells. Adjacency is the Delaunay edges
    /// computed from positions.
    fn tetra_model() -> PointCloudModel {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 5.0),
            glam::Vec4::new(0.5, 0.0, 0.0, 5.0),
            glam::Vec4::new(0.0, 0.5, 0.0, 5.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 5.0),
        ];
        let n = points.len();
        let mut m = PointCloudModel {
            points,
            sh_coefficients: vec![0.0; n * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
        };
        m.compute_adjacency_default();
        m
    }

    fn settings_for(start_point: u32) -> TraceSettings {
        TraceSettings {
            start_point,
            max_steps: 32,
            weight_threshold: 0.0, // record_path ignores this anyway
            depth: 100.0,
            eval_mode: EvalMode::ConstantRgb(glam::Vec3::ONE),
        }
    }

    #[test]
    fn opacity_trace_preserves_every_non_colour_result() {
        let model = tetra_model();
        let ray = Ray {
            origin: glam::Vec3::new(-1.0, 0.1, 0.1),
            direction: glam::Vec3::X,
        };
        let settings = settings_for(0);
        let colour = trace_one_ray(&model, ray, settings);
        let opacity = trace_one_ray(
            &model,
            ray,
            TraceSettings {
                eval_mode: EvalMode::Opacity,
                ..settings
            },
        );
        assert_eq!(opacity.rgba.truncate(), glam::Vec3::ZERO);
        assert_eq!(opacity.rgba.w, colour.rgba.w);
        assert_eq!(opacity.steps, colour.steps);
        assert_eq!(opacity.last_point, colour.last_point);
        assert_eq!(opacity.t_end, colour.t_end);
        assert_eq!(opacity.depth_mode, colour.depth_mode);
        assert_eq!(opacity.peak_weight, colour.peak_weight);
    }

    #[test]
    fn record_path_yields_finite_dts_and_nonempty_entries() {
        let model = tetra_model();
        let ray = Ray {
            origin: glam::Vec3::new(0.1, 0.1, -1.0),
            direction: glam::Vec3::new(0.0, 0.0, 1.0),
        };
        let path = record_path(&model, ray, settings_for(0));
        assert!(!path.entries.is_empty(), "path is empty");
        for e in &path.entries {
            assert!(e.dt.is_finite() && e.dt >= 0.0, "bad dt {}", e.dt);
            assert!(
                (e.cell as usize) < model.points.len(),
                "bad cell {}",
                e.cell
            );
        }
        // Ray direction is unit-length.
        assert!((path.ray_dir.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn record_path_agrees_with_trace_one_ray_on_path() {
        // Integrate over the recorded path manually with the same formulas
        // the GPU shader uses and check that the answer matches trace_one_ray.
        let model = tetra_model();
        let ray = Ray {
            origin: glam::Vec3::new(0.1, 0.1, -1.0),
            direction: glam::Vec3::new(0.0, 0.0, 1.0),
        };
        let settings = settings_for(0);
        let path = record_path(&model, ray, settings);
        let integrated = trace_one_ray(&model, ray, settings);

        let mut transmittance = 1.0f32;
        let mut alpha_accum = 0.0f32;
        for e in &path.entries {
            let s = model.points[e.cell as usize].w;
            if s <= 1e-6 {
                continue;
            }
            let alpha = 1.0 - (-s * e.dt).exp();
            alpha_accum += transmittance * alpha;
            transmittance *= 1.0 - alpha;
        }
        // Reconstructed alpha matches `1 - transmittance` from trace_one_ray.
        let trace_alpha = integrated.rgba.w;
        assert!(
            (alpha_accum - trace_alpha).abs() < 1e-4,
            "path-integrated alpha {alpha_accum} disagrees with trace {trace_alpha}"
        );
    }

    #[test]
    fn record_path_respects_max_steps() {
        let model = tetra_model();
        let ray = Ray {
            origin: glam::Vec3::new(0.1, 0.1, -1.0),
            direction: glam::Vec3::new(0.0, 0.0, 1.0),
        };
        let mut settings = settings_for(0);
        settings.max_steps = 2;
        let path = record_path(&model, ray, settings);
        assert!(path.entries.len() <= 2);
    }

    #[test]
    fn terminal_cell_integrates_to_depth_without_an_exit_face() {
        let model = PointCloudModel {
            points: vec![glam::Vec4::new(0.0, 0.0, 0.0, 2.0)],
            sh_coefficients: vec![0.0; 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: Vec::new(),
                offsets: vec![0, 0],
            }),
            radii: None,
        };
        let ray = Ray {
            origin: glam::Vec3::ZERO,
            direction: glam::Vec3::Z,
        };
        let settings = TraceSettings {
            depth: 3.0,
            ..settings_for(0)
        };
        let path = record_path(&model, ray, settings);
        assert_eq!(path.entries.len(), 1);
        assert_eq!(path.entries[0].cell, 0);
        assert_eq!(path.entries[0].dt, 3.0);

        let traced = trace_one_ray(&model, ray, settings);
        let expected_alpha = 1.0 - (-6.0_f32).exp();
        assert!((traced.rgba.w - expected_alpha).abs() < 1e-6);
        assert_eq!(traced.t_end, 3.0);
    }

    #[test]
    fn power_foam_clips_cell_integration_to_support_sphere() {
        let model = PointCloudModel {
            points: vec![glam::Vec4::new(0.0, 0.0, 0.0, 2.0)],
            sh_coefficients: vec![0.0; 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: Vec::new(),
                offsets: vec![0, 0],
            }),
            radii: Some(vec![1.0]),
        };
        let ray = Ray {
            origin: glam::Vec3::new(0.0, 0.0, -2.0),
            direction: glam::Vec3::Z,
        };
        let settings = TraceSettings {
            depth: 10.0,
            ..settings_for(0)
        };
        let path = record_path(&model, ray, settings);
        assert_eq!(path.entries.len(), 1);
        assert!((path.entries[0].dt - 2.0).abs() < 1e-6);

        let traced = trace_one_ray(&model, ray, settings);
        let expected_alpha = 1.0 - (-4.0_f32).exp();
        assert!((traced.rgba.w - expected_alpha).abs() < 1e-6);
        assert_eq!(traced.t_end, 10.0);
    }

    #[test]
    fn powerfoam_splats_discover_disconnected_supports_in_depth_order() {
        let model = PointCloudModel {
            points: vec![
                glam::Vec4::new(0.0, 0.0, 3.0, 1.0),
                glam::Vec4::new(0.0, 0.0, 6.0, 1.0),
            ],
            sh_coefficients: vec![0.0; 6],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: Vec::new(),
                offsets: vec![0, 0, 0],
            }),
            radii: Some(vec![0.5, 0.5]),
        };
        let ray = Ray {
            origin: glam::Vec3::ZERO,
            direction: glam::Vec3::Z,
        };
        let settings = TraceSettings {
            depth: 10.0,
            ..settings_for(0)
        };

        let walked = record_path_jacobians(&model, ray, settings);
        assert_eq!(walked.entries.len(), 1);
        let splatted = record_powerfoam_splats_jacobians(&model, ray, settings);
        assert_eq!(
            splatted
                .entries
                .iter()
                .map(|entry| entry.cell)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert!(splatted
            .entries
            .iter()
            .all(|entry| (entry.dt - 1.0).abs() < 1.0e-6));
    }

    #[test]
    fn support_sphere_preserves_non_unit_ray_parameterization() {
        let origin = glam::Vec3::new(0.0, 0.0, -2.0);
        let center = glam::Vec3::ZERO;
        let unit = support_interval(true, origin, glam::Vec3::Z, center, 1.0, 0.0, 10.0).unwrap();
        let scaled =
            support_interval(true, origin, 2.0 * glam::Vec3::Z, center, 1.0, 0.0, 10.0).unwrap();
        assert_eq!(unit, (1.0, 3.0));
        assert_eq!(scaled, (0.5, 1.5));
        assert_eq!(
            origin + unit.0 * glam::Vec3::Z,
            origin + scaled.0 * 2.0 * glam::Vec3::Z
        );
        assert_eq!(
            origin + unit.1 * glam::Vec3::Z,
            origin + scaled.1 * 2.0 * glam::Vec3::Z
        );
    }

    #[test]
    fn power_foam_walk_matches_brute_force_bounded_cells() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(2.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(4.0, 0.0, 0.0, 1.0),
        ];
        let model = PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: vec![1, 2, 0, 2, 0, 1],
                offsets: vec![0, 2, 4, 6],
            }),
            radii: Some(vec![1.4, 0.9, 1.6]),
            points,
        };
        let ray = Ray {
            origin: glam::Vec3::new(-2.0, 0.0, 0.0),
            direction: glam::Vec3::X,
        };
        let settings = TraceSettings {
            depth: 8.0,
            ..settings_for(0)
        };
        let walked = record_path(&model, ray, settings);

        // Independent O(N²) oracle: intersect each site's full power cell
        // against every radical half-space, then against its support sphere.
        let mut oracle = Vec::new();
        for (i, point_i) in model.points.iter().enumerate() {
            let center_i = point_i.truncate();
            let radius_i = model.radii.as_ref().unwrap()[i];
            let mut near = 0.0_f32;
            let mut far = settings.depth;
            for (j, point_j) in model.points.iter().enumerate() {
                if i == j {
                    continue;
                }
                let center_j = point_j.truncate();
                let radius_j = model.radii.as_ref().unwrap()[j];
                let offset = center_j - center_i;
                let shift = 0.5
                    + 0.5 * (radius_i * radius_i - radius_j * radius_j) / offset.length_squared();
                let face = center_i + shift * offset;
                let denominator = offset.dot(ray.direction);
                let t = (face - ray.origin).dot(offset) / denominator;
                if denominator > 0.0 {
                    far = far.min(t);
                } else if denominator < 0.0 {
                    near = near.max(t);
                }
            }
            if let Some((start, end)) = support_interval(
                true,
                ray.origin,
                ray.direction,
                center_i,
                radius_i,
                near,
                far,
            ) {
                oracle.push((
                    start,
                    PathEntry {
                        cell: i as u32,
                        dt: end - start,
                    },
                ));
            }
        }
        oracle.sort_by(|a, b| a.0.total_cmp(&b.0));

        assert_eq!(walked.entries.len(), oracle.len());
        for pair in walked.entries.iter().zip(oracle.iter()) {
            let walked_entry = pair.0;
            let oracle_entry = &pair.1 .1;
            assert_eq!(walked_entry.cell, oracle_entry.cell);
            assert!((walked_entry.dt - oracle_entry.dt).abs() < 1e-5);
        }
    }

    fn perturb_site(model: &mut PointCloudModel, cell: u32, component: usize, delta: f32) {
        let index = cell as usize;
        match component {
            0 => model.points[index].x += delta,
            1 => model.points[index].y += delta,
            2 => model.points[index].z += delta,
            3 => model.radii.as_mut().unwrap()[index] += delta,
            _ => panic!("invalid site component {component}"),
        }
    }

    fn recorded_dt_for_cell(
        model: &PointCloudModel,
        ray: Ray,
        settings: TraceSettings,
        cell: u32,
    ) -> f32 {
        let path = record_path_jacobians(model, ray, settings);
        path.entries
            .iter()
            .find(|entry| entry.cell == cell)
            .unwrap_or_else(|| panic!("perturbation removed cell {cell} from the fixed path"))
            .dt
    }

    fn assert_site_jacobian_matches_finite_difference(
        model: &PointCloudModel,
        ray: Ray,
        settings: TraceSettings,
        target_cell: u32,
        parameter_cell: u32,
        analytical: glam::Vec4,
    ) {
        const EPSILON: f32 = 1.0e-3;
        for component in 0..4 {
            let mut plus = model.clone();
            perturb_site(&mut plus, parameter_cell, component, EPSILON);
            let plus_dt = recorded_dt_for_cell(&plus, ray, settings, target_cell);
            let mut minus = model.clone();
            perturb_site(&mut minus, parameter_cell, component, -EPSILON);
            let minus_dt = recorded_dt_for_cell(&minus, ray, settings, target_cell);
            let numerical = (plus_dt - minus_dt) / (2.0 * EPSILON);
            let expected = analytical[component];
            let absolute = (expected - numerical).abs();
            let scale = expected.abs().max(numerical.abs()).max(1.0e-4);
            assert!(
                absolute < 2.0e-3 || absolute / scale < 1.0e-2,
                "cell {target_cell}, parameter cell {parameter_cell}, component {component}: \
                 analytical={expected}, numerical={numerical}, absolute={absolute}",
            );
        }
    }

    #[test]
    fn power_foam_face_jacobians_match_central_finite_differences() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(2.0, 0.0, 0.0, 1.0),
        ];
        let model = PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: vec![1, 0, 2, 1],
                offsets: vec![0, 1, 3, 4],
            }),
            radii: Some(vec![2.0, 2.0, 2.0]),
            points,
        };
        let ray = Ray {
            origin: glam::Vec3::new(-3.0, 0.2, -0.1),
            direction: glam::Vec3::new(1.0, 0.05, 0.02),
        };
        let settings = TraceSettings {
            depth: 10.0,
            ..settings_for(0)
        };
        let result = record_path_jacobians(&model, ray, settings);
        let entry = *result.entries.iter().find(|entry| entry.cell == 1).unwrap();
        assert_eq!(entry.previous_cell, 0);
        assert_eq!(entry.next_cell, 2);
        assert_site_jacobian_matches_finite_difference(
            &model,
            ray,
            settings,
            entry.cell,
            entry.previous_cell,
            entry.dt_d_previous,
        );
        assert_site_jacobian_matches_finite_difference(
            &model,
            ray,
            settings,
            entry.cell,
            entry.cell,
            entry.dt_d_current,
        );
        assert_site_jacobian_matches_finite_difference(
            &model,
            ray,
            settings,
            entry.cell,
            entry.next_cell,
            entry.dt_d_next,
        );
    }

    #[test]
    fn power_foam_sphere_jacobian_matches_central_finite_differences() {
        let model = PointCloudModel {
            points: vec![glam::Vec4::new(0.0, 0.0, 0.0, 1.0)],
            sh_coefficients: vec![0.0; 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: Vec::new(),
                offsets: vec![0, 0],
            }),
            radii: Some(vec![1.0]),
        };
        let ray = Ray {
            origin: glam::Vec3::new(0.2, 0.1, -2.0),
            direction: glam::Vec3::Z,
        };
        let settings = TraceSettings {
            depth: 10.0,
            ..settings_for(0)
        };
        let entry = record_path_jacobians(&model, ray, settings).entries[0];
        assert_eq!(entry.previous_cell, entry.cell);
        assert_eq!(entry.next_cell, entry.cell);
        assert_eq!(entry.dt_d_previous, glam::Vec4::ZERO);
        assert_eq!(entry.dt_d_next, glam::Vec4::ZERO);
        assert_site_jacobian_matches_finite_difference(
            &model,
            ray,
            settings,
            entry.cell,
            entry.cell,
            entry.dt_d_current,
        );
    }

    #[test]
    fn path_jacobian_keeps_true_entry_neighbor_across_skipped_cells() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(2.0, 0.0, 0.0, 1.0),
        ];
        let model = PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(crate::Adjacency {
                neighbors: vec![1, 0, 2, 1],
                offsets: vec![0, 1, 3, 4],
            }),
            radii: Some(vec![0.95, 0.1, 0.95]),
            points,
        };
        let ray = Ray {
            origin: glam::Vec3::new(-2.0, 0.2, 0.0),
            direction: glam::Vec3::X,
        };
        let result = record_path_jacobians(
            &model,
            ray,
            TraceSettings {
                depth: 6.0,
                ..settings_for(0)
            },
        );
        let cells: Vec<u32> = result.entries.iter().map(|entry| entry.cell).collect();
        assert_eq!(cells, vec![0, 2]);
        assert_eq!(result.entries[1].previous_cell, 1);
        assert_ne!(result.entries[1].previous_cell, result.entries[0].cell);
    }
}
