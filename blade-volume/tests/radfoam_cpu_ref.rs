//! CPU reference tracer for RadFoam traversal and integration.
//!
//! Goal: provide a deterministic, dependency-light reference implementation of the
//! core forward traversal used by our WGSL compute shader (and upstream RadFoam),
//! so we can write correctness tests without relying on visual inspection.
//!
//! This is intentionally small and "obviously correct" rather than fast.
//!
//! What it implements (forward-only):
//! - Voronoi cell traversal over points with CSR adjacency.
//! - Plane selection identical to upstream `trace()` logic:
//!   offset = next - current
//!   face_origin = current + 0.5 * offset
//!   face_normal = offset
//!   dp = dot(face_normal, ray_dir)
//!   t = dot(face_origin - ray_origin, face_normal) / dp
//!   Pick the smallest `t` among faces with `dp > 0`.
//! - Segment-wise alpha integration per cell (piecewise constant density):
//!   dt = max(t1 - t0, 0)
//!   alpha = 1 - exp(-s * dt)
//!   w = T * alpha
//!   rgb += w * cell_rgb
//!   T *= (1 - alpha)
//! - Termination:
//!   - transmittance <= weight_threshold
//!   - steps >= max_steps
//!   - no valid exit face
//!   - t0 > depth (camera far)
//!
//! Debugging:
//! - Set `RADFOAM_CPU_TRACE=1` to print per-step traversal information (stderr).
//!   This is useful when a GPU-vs-CPU test mismatch indicates traversal differences.
//!
//! Notes:
//! - This module uses the unified `PointCloudModel`:
//!   - Position (xyz) + density (w) in `points`
//!   - SH coefficients in `sh_coefficients` (3 * sh_components per point)
//! - For now, SH evaluation is intentionally minimal:
//!   - If you pass `EvalMode::ConstantRgb`, it ignores SH and uses a fixed rgb.
//!   - If you pass `EvalMode::Sh`, it evaluates the same SH basis constants as our shader
//!     for degrees 0..=3, assuming coefficient layout matches our WGSL.
//! - This is placed under `tests/` so it can be used by integration tests directly.
//!
//! Usage pattern (in other tests):
//! - Load a PointCloudModel fixture
//! - Call `trace_one_ray(...)` for a few rays and compare to GPU output.

use blade_volume as vol;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ray {
    pub origin: glam::Vec3,
    pub direction: glam::Vec3,
}

/// Controls how RGB is produced per cell.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum EvalMode {
    /// Ignore SH coefficients, use a constant RGB for every visited cell.
    ConstantRgb(glam::Vec3),
    /// Evaluate SH coefficients in the packed attribute buffer.
    Sh,
}

#[derive(Clone, Copy, Debug)]
pub struct TraceSettings {
    pub weight_threshold: f32,
    pub max_steps: u32,
    pub start_point: u32,
    pub depth: f32,
    pub eval_mode: EvalMode,

    /// Debug: limit how many steps to print when `RADFOAM_CPU_TRACE=1`.
    /// If `None`, prints all steps.
    pub debug_max_print_steps: Option<u32>,
}

impl Default for TraceSettings {
    fn default() -> Self {
        Self {
            weight_threshold: 0.001,
            max_steps: 1024,
            start_point: 0,
            depth: 10_000.0,
            eval_mode: EvalMode::ConstantRgb(glam::Vec3::splat(1.0)),
            debug_max_print_steps: Some(32),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub struct TraceResult {
    pub rgba: glam::Vec4,
    pub steps: u32,
    pub last_point: u32,
    pub t_end: f32,
}

fn sh_component_count(deg: u32) -> u32 {
    let d = deg + 1;
    d * d
}

fn read_density(model: &vol::PointCloudModel, point_idx: u32) -> f32 {
    model.points[point_idx as usize].w
}

/// Per-point radius. Defaults to zero when the model has no `radii` (plain
/// RadFoam, in which case the radical plane in [`trace_one_ray`] degenerates to
/// the standard Voronoi bisector).
fn read_radius(model: &vol::PointCloudModel, point_idx: u32) -> f32 {
    model
        .radii
        .as_deref()
        .map_or(0.0, |r| r[point_idx as usize])
}

fn eval_rgb_constant(constant: glam::Vec3) -> glam::Vec3 {
    constant
}

/// SH basis constants for degrees up to 3 (16 components).
/// These match `blade-gaussian/examples/shader.wgsl` and our `examples/radfoam.wgsl`.
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

/// Evaluate SH RGB for one point using the packed coefficient layout:
/// coeff(component i, channel c) is at base + 3*i + c.
///
/// This mirrors the WGSL implementation and supports degree 0..=3.
/// For degree < 3, extra coefficients are ignored.
fn eval_rgb_sh(model: &vol::PointCloudModel, point_idx: u32, dir: glam::Vec3) -> glam::Vec3 {
    let deg = model.sh_degree as u32;
    let deg = deg.min(3);
    let comps = sh_component_count(deg).min(16);

    let sh = sh_basis_constants();
    let d2 = dir * dir;

    let sh_dim = 3 * comps;
    let base = (point_idx as usize) * (sh_dim as usize);

    let mut color = glam::Vec3::ZERO;

    // L0
    if comps >= 1 {
        let c0 = sh[0];
        color.x += c0 * model.sh_coefficients[base];
        color.y += c0 * model.sh_coefficients[base + 1];
        color.z += c0 * model.sh_coefficients[base + 2];
    }

    if deg >= 1 && comps >= 4 {
        // 1: y, 2: z, 3: x
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

        // 4: x*y
        let c4 = sh[4];
        color.x += c4 * model.sh_coefficients[base + 12] * x * y;
        color.y += c4 * model.sh_coefficients[base + 13] * x * y;
        color.z += c4 * model.sh_coefficients[base + 14] * x * y;

        // 5: y*z
        let c5 = sh[5];
        color.x += c5 * model.sh_coefficients[base + 15] * y * z;
        color.y += c5 * model.sh_coefficients[base + 16] * y * z;
        color.z += c5 * model.sh_coefficients[base + 17] * y * z;

        // 6: (3z^2 - 1)
        let c6 = sh[6];
        let t6 = 3.0 * zz - 1.0;
        color.x += c6 * model.sh_coefficients[base + 18] * t6;
        color.y += c6 * model.sh_coefficients[base + 19] * t6;
        color.z += c6 * model.sh_coefficients[base + 20] * t6;

        // 7: x*z
        let c7 = sh[7];
        color.x += c7 * model.sh_coefficients[base + 21] * x * z;
        color.y += c7 * model.sh_coefficients[base + 22] * x * z;
        color.z += c7 * model.sh_coefficients[base + 23] * x * z;

        // 8: (x^2 - y^2)
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

        // 9
        let c9 = sh[9];
        let t9 = y * (3.0 * xx - yy);
        color.x += c9 * model.sh_coefficients[base + 27] * t9;
        color.y += c9 * model.sh_coefficients[base + 28] * t9;
        color.z += c9 * model.sh_coefficients[base + 29] * t9;

        // 10
        let c10 = sh[10];
        let t10 = x * y * z;
        color.x += c10 * model.sh_coefficients[base + 30] * t10;
        color.y += c10 * model.sh_coefficients[base + 31] * t10;
        color.z += c10 * model.sh_coefficients[base + 32] * t10;

        // 11
        let c11 = sh[11];
        let t11 = y * (5.0 * zz - 1.0);
        color.x += c11 * model.sh_coefficients[base + 33] * t11;
        color.y += c11 * model.sh_coefficients[base + 34] * t11;
        color.z += c11 * model.sh_coefficients[base + 35] * t11;

        // 12
        let c12 = sh[12];
        let t12 = z * (5.0 * zz - 3.0);
        color.x += c12 * model.sh_coefficients[base + 36] * t12;
        color.y += c12 * model.sh_coefficients[base + 37] * t12;
        color.z += c12 * model.sh_coefficients[base + 38] * t12;

        // 13
        let c13 = sh[13];
        let t13 = x * (5.0 * zz - 1.0);
        color.x += c13 * model.sh_coefficients[base + 39] * t13;
        color.y += c13 * model.sh_coefficients[base + 40] * t13;
        color.z += c13 * model.sh_coefficients[base + 41] * t13;

        // 14
        let c14 = sh[14];
        let t14 = z * (xx - yy);
        color.x += c14 * model.sh_coefficients[base + 42] * t14;
        color.y += c14 * model.sh_coefficients[base + 43] * t14;
        color.z += c14 * model.sh_coefficients[base + 44] * t14;

        // 15
        let c15 = sh[15];
        let t15 = x * (xx - 3.0 * yy);
        color.x += c15 * model.sh_coefficients[base + 45] * t15;
        color.y += c15 * model.sh_coefficients[base + 46] * t15;
        color.z += c15 * model.sh_coefficients[base + 47] * t15;
    }

    // For visibility parity with our WGSL: add bias.
    0.5 + color
}

/// Trace a single ray through the RadFoam point set starting from `settings.start_point`.
///
/// This is the CPU reference equivalent of the traversal and forward integration.
pub fn trace_one_ray(
    model: &vol::PointCloudModel,
    ray: Ray,
    settings: TraceSettings,
) -> TraceResult {
    assert!(!model.points.is_empty(), "model has no points");
    assert!(
        (settings.start_point as usize) < model.points.len(),
        "start_point out of bounds"
    );

    let adjacency = model
        .adjacency
        .as_ref()
        .expect("trace_one_ray requires adjacency");

    let debug_enabled = std::env::var("RADFOAM_CPU_TRACE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);

    let mut dir = ray.direction;
    let dir_len = dir.length();
    if dir_len <= 0.0 || !dir_len.is_finite() {
        return TraceResult {
            rgba: glam::Vec4::new(0.0, 0.0, 0.0, 0.0),
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

        if transmittance <= settings.weight_threshold {
            if debug_enabled {
                eprintln!(
                    "[radfoam_cpu_ref] stop: transmittance {} <= threshold {} at step {}",
                    transmittance, settings.weight_threshold, steps
                );
            }
            break;
        }
        if t0 > settings.depth {
            if debug_enabled {
                eprintln!(
                    "[radfoam_cpu_ref] stop: t0 {} > depth {} at step {}",
                    t0, settings.depth, steps
                );
            }
            break;
        }

        let begin = adjacency.offsets[current as usize] as usize;
        let end = adjacency.offsets[current as usize + 1] as usize;

        // Match shader semantics:
        // - initialize to a large finite value
        // - consider all faces with dp > 0
        // - allow negative/zero t; integration happens only if t1 > t0.
        let mut best_t1 = f32::MAX;
        let mut next_face: Option<usize> = None;
        let mut dp_pos_count: u32 = 0;
        let mut considered_count: u32 = 0;
        let mut nan_t_count: u32 = 0;

        let r_i_sq = current_radius * current_radius;
        for (j, &next_idx_u32) in adjacency.neighbors[begin..end].iter().enumerate() {
            let next_idx = next_idx_u32 as usize;
            let next_p = model.points[next_idx];
            let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);
            let r_j = read_radius(model, next_idx_u32);
            let offset = next_pos - current_pos;

            // Radical plane between weighted spheres; reduces to the bisector when
            // both radii are zero. Matches shaders/radfoam_trace.wgsl.
            let dsq = offset.length_squared().max(1e-20);
            let shift = 0.5 + 0.5 * (r_i_sq - r_j * r_j) / dsq;
            let face_origin = current_pos + shift * offset;
            let face_normal = offset;

            let dp = face_normal.dot(dir);
            if dp > 0.0 {
                dp_pos_count += 1;

                let t = (face_origin - ray.origin).dot(face_normal) / dp;
                if !t.is_finite() {
                    nan_t_count += 1;
                    continue;
                }
                considered_count += 1;
                if t < best_t1 {
                    best_t1 = t;
                    next_face = Some(j);
                }
            }
        }

        if debug_enabled {
            let do_print = settings
                .debug_max_print_steps
                .map(|m| steps <= m)
                .unwrap_or(true);
            if do_print {
                eprintln!(
                    "[radfoam_cpu_ref] step {}: cell={} t0={} T={} faces={} dp>0={} considered={} nan_t={} best_t1={} has_next={}",
                    steps,
                    current,
                    t0,
                    transmittance,
                    (end - begin),
                    dp_pos_count,
                    considered_count,
                    nan_t_count,
                    best_t1,
                    next_face.is_some()
                );
            }
        }

        let Some(j) = next_face else {
            if debug_enabled {
                eprintln!(
                    "[radfoam_cpu_ref] stop: no next face found at step {} (cell={})",
                    steps, current
                );
            }
            break;
        };

        let next_idx_u32 = adjacency.neighbors[begin + j];
        let next_idx = next_idx_u32;
        let next_p = model.points[next_idx as usize];
        let next_pos = glam::Vec3::new(next_p.x, next_p.y, next_p.z);

        // Integrate only if the segment is forward in parametric distance.
        if best_t1 > t0 {
            let s = read_density(model, current);
            if s > 1e-6 {
                let dt = (best_t1 - t0).max(0.0);
                let alpha = 1.0 - (-s * dt).exp();
                let w = transmittance * alpha;

                let rgb = match settings.eval_mode {
                    EvalMode::ConstantRgb(c) => eval_rgb_constant(c),
                    EvalMode::Sh => eval_rgb_sh(model, current, dir),
                };

                accum_rgb += w * rgb;
                transmittance *= 1.0 - alpha;

                if debug_enabled {
                    let do_print = settings
                        .debug_max_print_steps
                        .map(|m| steps <= m)
                        .unwrap_or(true);
                    if do_print {
                        eprintln!(
                            "[radfoam_cpu_ref] integrate: cell={} t0={} t1={} dt={} s={} alpha={} w={} new_T={}",
                            current, t0, best_t1, dt, s, alpha, w, transmittance
                        );
                    }
                }
            }
        }

        // Match shader: t0 = max(t0, t1) (negative t1 keeps t0 at 0).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_traces_without_nan_on_tiny_fixture() {
        let model = vol::io::load_radfoam("tests/data/radfoam_tiny_ascii.ply");

        // Ray from slightly above the square, pointing down + forward-ish.
        let ray = Ray {
            origin: glam::Vec3::new(0.5, 0.5, -1.0),
            direction: glam::Vec3::new(0.0, 0.0, 1.0),
        };

        let settings = TraceSettings {
            start_point: 0,
            max_steps: 16,
            depth: 100.0,
            weight_threshold: 1e-4,
            eval_mode: EvalMode::ConstantRgb(glam::Vec3::splat(1.0)),
            // Avoid noisy logs in unit tests by default.
            debug_max_print_steps: Some(0),
        };

        let out = trace_one_ray(&model, ray, settings);

        assert!(out.rgba.x.is_finite());
        assert!(out.rgba.y.is_finite());
        assert!(out.rgba.z.is_finite());
        assert!(out.rgba.w.is_finite());
        assert!(out.rgba.w >= 0.0 && out.rgba.w <= 1.0);
        assert!(out.steps > 0);
    }
}
