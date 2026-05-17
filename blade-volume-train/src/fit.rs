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

/// Probe whether a Blade GPU context can be initialised on this host.
///
/// meganeura 0.2 builds its own GPU context inside [`build_session`], so we
/// can't inject one — the best we can do is *try* to initialise the same
/// configuration first and bail if it fails, mirroring what
/// `meganeura::runtime::Session::new` does internally. If this returns `false`,
/// callers should skip; calling [`fit_constant_rgb`] anyway would panic.
pub fn gpu_available() -> bool {
    unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: true,
            capture: false,
            overlay: false,
            ray_tracing: false,
            xr: None,
            device_id: None,
        })
    }
    .is_ok()
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
/// Trivial as a renderer but exercises the full Graph → autodiff → Adam loop.
/// Returns the trained value.
///
/// Panics if meganeura fails to build/run the graph (e.g. no supported GPU).
/// Callers running this without a guaranteed GPU should probe with
/// [`gpu_available`] first.
pub fn fit_constant_rgb(target: [f32; 3], config: FitConfig) -> [f32; 3] {
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

    let mut session = mn::train::build_session(&g);

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
        data_input: "x".into(),
        label_input: "labels".into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_target_rgb_via_adam() {
        if !gpu_available() {
            eprintln!("skipping fit_constant_rgb: no supported GPU device");
            return;
        }
        let target = [0.30_f32, 0.60, 0.90];
        let result = fit_constant_rgb(target, FitConfig::default());
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - target[i]).abs() < 0.02,
                "channel {i} converged to {v}, want {} (within 0.02)",
                target[i]
            );
        }
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
        };
        let mut g = mn::Graph::new();
        let _node = sh_coefficients_as_parameter(&mut g, &model);
        // Graph should now contain the parameter node; we just exercised the
        // declaration without panicking.
        assert!(!g.nodes().is_empty());
    }
}
