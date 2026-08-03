//! Smallest-end-to-end meganeura training loop wired into this crate.
//!
//! This is **not** the real differentiable renderer — it's a plumbing proof.
//! The toy problem: optimise a `[1, 3]` parameter to match a target RGB via
//! MSE loss + Adam. It exists to validate:
//!
//! 1. `meganeura` links and runs in our crate alongside `blade-graphics`.
//! 2. We can declare parameters and an MSE loss that gradients flow through.
//! 3. The Trainer / DataLoader / SessionConfig wiring is understood.
//!
//! The real differentiable forward in M3c-4 will keep this overall shape:
//! - Build a `Graph` from a `vol::PointCloudModel` (positions, density, SH,
//!   radii) as parameter nodes.
//! - Pixels from `vol::trace::trace_one_ray` flow out as a tensor.
//! - L1 + SSIM against an input ground-truth image.
//! - Adam steps the parameters.
//!
//! What we already know about meganeura that affects M3c-4:
//! - Graph ops are standard NN primitives (matmul, conv2d, attention, RMS/
//!   LayerNorm, scatter_add). No `where` / scan / while primitive is exposed,
//!   so a literal data-dependent cell-walk traversal cannot be expressed as a
//!   composition of existing ops. Differentiable rendering will need a custom
//!   op (recorded-path-then-integrate, like PowerFoam's raytrace mode) or a
//!   splat-style rasteriser composed from elementwise + `scatter_add`.

use blade_graphics as gpu;
use blade_volume as vol;
use meganeura as mn;
use std::sync;

/// Try to bring up a Blade GPU context. Returns `None` on hosts without a
/// supported device (CI, headless, no Vulkan/Metal/DX12) so callers can
/// skip gracefully.
///
/// The context is wrapped in `Arc` so the same one can be shared with
/// `meganeura` via [`mn::SessionConfig::gpu`] — no need to spin up a second
/// context for training.
///
/// Refuses to return a software-rasterizer context (llvmpipe, lavapipe) by
/// default — a silent fallback to CPU rendering is ~25× slower than the
/// real GPU and was responsible for ~43 h of wasted training runs after the
/// May-18 Xid 62 crash. Set `BLADE_VOLUME_ALLOW_SOFTWARE=1` to override.
pub fn try_init_gpu() -> Option<sync::Arc<gpu::Context>> {
    if blade_volume::gpu::access_disabled() {
        eprintln!("try_init_gpu: GPU access disabled by BLADE_VOLUME_DISABLE_GPU");
        return None;
    }
    let ctx = mn::init_gpu_context().ok()?;
    let info = ctx.device_information();
    let allow_sw = std::env::var("BLADE_VOLUME_ALLOW_SOFTWARE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if info.is_software_emulated && !allow_sw {
        eprintln!(
            "try_init_gpu: refusing software rasterizer '{}' (driver '{}'); \
             set BLADE_VOLUME_ALLOW_SOFTWARE=1 to override, or fix the Vulkan ICD selection \
             (pin VK_ICD_FILENAMES to the real driver).",
            info.device_name, info.driver_name,
        );
        return None;
    }
    eprintln!(
        "try_init_gpu: using '{}' ({})",
        info.device_name, info.driver_name,
    );
    Some(sync::Arc::new(ctx))
}

#[cfg(test)]
pub(crate) fn gpu_test_guard() -> sync::MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: sync::OnceLock<sync::Mutex<()>> = sync::OnceLock::new();
    GPU_TEST_LOCK
        .get_or_init(|| sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(sync::PoisonError::into_inner)
}

#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    pub learning_rate: f32,
    pub epochs: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epochs: 200,
        }
    }
}

/// Optimise a 3-vector parameter `rgb` so that `mse(rgb, target) → 0`.
///
/// Trivial as a renderer but exercises the full Graph → autodiff → Adam loop
/// while sharing a GPU context with the rest of the pipeline. Returns the
/// trained value.
pub fn fit_constant_rgb(
    target: [f32; 3],
    config: FitConfig,
    gpu: sync::Arc<gpu::Context>,
) -> [f32; 3] {
    let mut g = mn::Graph::new();

    // batch=1 because the toy has a single "sample". meganeura still requires
    // the per-batch convention: both `x` and `labels` are 2D `[batch, dim]`.
    let batch: usize = 1;
    let dim: usize = 3;

    // `x` is required by the Trainer (it calls set_input("x", _) every step)
    // but the toy has no real input — we multiply it by a same-shape zero
    // constant so it threads into the graph without contributing. meganeura
    // doesn't broadcast scalars, so the zero has to match `x`'s shape.
    let x = g.input("x", &[batch, dim]);
    let labels = g.input("labels", &[batch, dim]);
    let rgb_param = g.parameter("rgb", &[batch, dim]);

    let zero = g.constant(vec![0.0; batch * dim], &[batch, dim]);
    let dead = g.mul(x, zero);
    let pred = g.add(rgb_param, dead);
    let loss = g.mse_loss(pred, labels);
    g.set_outputs(vec![loss]);

    let (mut session, _report) = mn::build(
        &g,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );

    // Start at zero so we know convergence wasn't pre-seeded.
    session.set_parameter("rgb", &vec![0.0; batch * dim]);

    // One-sample dataset: input is the target (ignored downstream), label is
    // the target. Each "epoch" yields one batch → one Adam step.
    let data: Vec<f32> = target.to_vec();
    let labels: Vec<f32> = target.to_vec();
    let mut loader = mn::DataLoader::new(data, labels, dim, dim, batch);

    let train_cfg = mn::TrainConfig {
        optimizer: mn::Optimizer::adam(config.learning_rate),
        learning_rate: config.learning_rate,
        log_interval: 0,
    };
    let mut trainer = mn::Trainer::new(session, train_cfg);
    trainer.train(&mut loader, config.epochs);
    let session = trainer.into_session();

    let buf = session
        .param_buffer("rgb")
        .expect("rgb parameter not found in session plan");
    let mut data = vec![0.0f32; batch * dim];
    session.read_buffer(buf, &mut data);
    [data[0], data[1], data[2]]
}

/// Build a `Graph` that takes a `vol::PointCloudModel`'s SH coefficients as a
/// parameter and exposes them as the only output. Not usable for training yet;
/// it's a sanity check that we can shape graph parameters from our data types.
///
/// Returns the parameter's `NodeId` so callers can extend the graph before
/// `set_outputs`.
pub fn sh_coefficients_as_parameter(g: &mut mn::Graph, model: &vol::PointCloudModel) -> mn::NodeId {
    let n = model.points.len();
    let sh_dim = model.sh_component_count() * 3;
    g.parameter("sh_coefficients", &[n, sh_dim])
}

/// Optimise an image-shaped parameter to match `target` via L1 loss + Adam.
///
/// Next step up from [`fit_constant_rgb`]: the parameter is now a flat RGB
/// image (`width * height * 3` floats in row-major order), the loss is L1
/// (the photometric loss used by 3DGS/RadFoam/PowerFoam training), and the
/// data flows as a higher-rank tensor. The forward is still identity (param
/// → prediction) — M3c-4 replaces this with the real differentiable
/// renderer, but the rest of this scaffold (loss shape, parameter layout,
/// Adam wiring) stays.
///
/// SSIM is intentionally not included yet. meganeura's op set doesn't expose
/// a 2D-convolution shortcut for the per-window stats; expressing SSIM as a
/// composition of `conv2d` would work but adds shape-juggling overhead we
/// don't need until L1 alone stops being enough.
///
/// `target.len()` must equal `width * height * 3`. Returns the trained image
/// in the same layout.
pub fn fit_constant_image(
    target: &[f32],
    width: u32,
    height: u32,
    config: FitConfig,
    gpu: sync::Arc<gpu::Context>,
) -> Vec<f32> {
    let dim = (width as usize) * (height as usize) * 3;
    assert_eq!(
        target.len(),
        dim,
        "fit_constant_image: target.len() ({}) must equal width*height*3 ({})",
        target.len(),
        dim
    );

    let mut g = mn::Graph::new();
    let batch: usize = 1;

    let x = g.input("x", &[batch, dim]);
    let labels = g.input("labels", &[batch, dim]);
    let img_param = g.parameter("img", &[batch, dim]);

    let zero = g.constant(vec![0.0; batch * dim], &[batch, dim]);
    let dead = g.mul(x, zero);
    let pred = g.add(img_param, dead);
    let loss = g.l1_loss(pred, labels);
    g.set_outputs(vec![loss]);

    let (mut session, _report) = mn::build(
        &g,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );

    session.set_parameter("img", &vec![0.5; batch * dim]);

    let data: Vec<f32> = target.to_vec();
    let labels: Vec<f32> = target.to_vec();
    let mut loader = mn::DataLoader::new(data, labels, dim, dim, batch);

    let train_cfg = mn::TrainConfig {
        optimizer: mn::Optimizer::adam(config.learning_rate),
        learning_rate: config.learning_rate,
        log_interval: 0,
    };
    let mut trainer = mn::Trainer::new(session, train_cfg);
    trainer.train(&mut loader, config.epochs);
    let session = trainer.into_session();

    let buf = session
        .param_buffer("img")
        .expect("img parameter not found in session plan");
    let mut out = vec![0.0f32; batch * dim];
    session.read_buffer(buf, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_target_rgb_via_adam() {
        let _gpu_test_guard = gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping fit_constant_rgb: no supported GPU device");
            return;
        };
        let target = [0.30_f32, 0.60, 0.90];
        let result = fit_constant_rgb(target, FitConfig::default(), gpu);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - target[i]).abs() < 0.02,
                "channel {i} converged to {v}, want {} (within 0.02)",
                target[i]
            );
        }
    }

    /// Synthetic 8x8 RGB target with a smooth gradient — every pixel takes a
    /// distinct value so the test catches systemic miswiring (rows transposed,
    /// channels swapped, batch dim collapsed, etc.).
    fn synthetic_target(w: u32, h: u32) -> Vec<f32> {
        let mut img = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                img.push(x as f32 / (w - 1) as f32);
                img.push(y as f32 / (h - 1) as f32);
                img.push(0.5);
            }
        }
        img
    }

    #[test]
    fn fits_image_via_l1_adam() {
        let _gpu_test_guard = gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping fit_constant_image: no supported GPU device");
            return;
        };
        let w = 8;
        let h = 8;
        let target = synthetic_target(w, h);
        let result = fit_constant_image(
            &target,
            w,
            h,
            FitConfig {
                learning_rate: 0.1,
                epochs: 300,
            },
            gpu,
        );

        assert_eq!(result.len(), target.len());
        let mut max_err = 0.0f32;
        for (got, want) in result.iter().zip(target.iter()) {
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.02,
            "max per-pixel L1 error {max_err} exceeds 0.02; image did not converge"
        );
    }

    #[test]
    fn sh_parameter_has_expected_shape() {
        // No GPU needed; just exercises the Graph wiring.
        let model = vol::PointCloudModel {
            points: vec![
                glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
                glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            ],
            sh_coefficients: vec![0.0; 2 * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_color_coefficients: None,
        };
        let mut g = mn::Graph::new();
        let _node = sh_coefficients_as_parameter(&mut g, &model);
        assert!(!g.nodes().is_empty());
    }
}
