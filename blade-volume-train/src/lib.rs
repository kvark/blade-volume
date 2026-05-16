//! Training pipeline for blade-volume point clouds.
//!
//! This crate sits on top of [`blade_volume`] (geometry, I/O, rendering) and uses
//! [`meganeura`] for gradient-based optimisation. The data boundary between the
//! trainer and the rest of the pipeline is [`blade_volume::PointCloudModel`].
//!
//! # Status
//!
//! This is the M3a skeleton. It defines the durable trainer state and the
//! `PointCloudModel ⇆ TrainerState` conversion only. Gradient-bearing tensors,
//! the differentiable forward, and the optimisation loop arrive in M3c.
//!
//! # Why mirror `PointCloudModel` instead of using it directly
//!
//! Training works in a transformed parameter space:
//! - Density and radius must be strictly positive in the rendered output, so we
//!   store their pre-images and apply `softplus` to recover the rendered value.
//!   This matches what PowerFoam does (`F.softplus(self.density, beta=100)`).
//! - Positions are stored flat (`Vec<glam::Vec3>`) rather than mixed with the
//!   density channel as in `PointCloudModel.points`, so the optimiser can treat
//!   them as a homogeneous parameter group.
//!
//! `PointCloudModel` stays the renderer-facing shape; `TrainerState` is the
//! optimiser-facing one. The conversions in this module are the one place where
//! the softplus / inverse-softplus transforms live.

#![allow(irrefutable_let_patterns)]

use blade_volume as vol;

// Re-export so downstream crates only need to add one dep.
pub use meganeura;

/// Pre-image of `softplus` used to map a strictly-positive rendered value
/// back to its (unconstrained) trainable parameter. Matches the inverse of
/// `softplus_beta(x, beta=1)`:
///
/// ```text
/// inverse_softplus(y) = ln(exp(y) - 1)
/// ```
///
/// `y` must be > 0. For `y ≫ 1` the formula collapses to `y`; for `y ≪ 1`
/// we fall back to `ln(y)` to avoid `exp(y) - 1` underflowing to zero.
fn inverse_softplus(y: f32) -> f32 {
    debug_assert!(y > 0.0, "softplus output must be strictly positive");
    if y > 20.0 {
        // exp(y) - 1 ≈ exp(y); ln(exp(y)) = y.
        y
    } else if y < 1e-6 {
        // exp(y) - 1 ≈ y; ln(y).
        y.ln()
    } else {
        (y.exp() - 1.0).ln()
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Optimiser-facing point cloud state.
///
/// Each `Vec` has length `points.len()` (or a multiple thereof for SH).
/// Positions and SH coefficients are free parameters; density/radius are stored
/// in their `softplus` pre-image so the optimiser doesn't have to fight the
/// positivity constraint. Use [`to_model`](Self::to_model) to materialise a
/// renderer-ready [`PointCloudModel`].
#[derive(Clone, Debug)]
pub struct TrainerState {
    /// Position per point. Free parameter.
    pub positions: Vec<glam::Vec3>,

    /// `inverse_softplus(density)` per point. Apply [`softplus`] to recover
    /// the value the renderer consumes.
    pub log_density: Vec<f32>,

    /// Packed SH coefficients, same layout as [`vol::PointCloudModel::sh_coefficients`].
    pub sh_coefficients: Vec<f32>,

    /// SH basis degree (0..=`vol::MAX_SH_DEGREE`).
    pub sh_degree: usize,

    /// `inverse_softplus(radius)` per point, when this cloud is a Power-Foam
    /// model. `None` for plain Voronoi.
    pub log_radii: Option<Vec<f32>>,
}

impl TrainerState {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Lift a renderer-ready [`vol::PointCloudModel`] into trainer parameter space.
    ///
    /// Density (`point.w`) and per-point radius are passed through
    /// `inverse_softplus`. Gaussian-only attributes (`transforms`) and the
    /// renderer's precomputed `adjacency` are dropped — the trainer rebuilds
    /// adjacency on demand from the current positions+radii.
    pub fn from_model(model: &vol::PointCloudModel) -> Self {
        let n = model.points.len();
        let mut positions = Vec::with_capacity(n);
        let mut log_density = Vec::with_capacity(n);
        for p in &model.points {
            positions.push(glam::Vec3::new(p.x, p.y, p.z));
            log_density.push(inverse_softplus(p.w.max(1e-12)));
        }
        let log_radii = model
            .radii
            .as_ref()
            .map(|r| r.iter().map(|&v| inverse_softplus(v.max(1e-12))).collect());
        Self {
            positions,
            log_density,
            sh_coefficients: model.sh_coefficients.clone(),
            sh_degree: model.sh_degree,
            log_radii,
        }
    }

    /// Materialise a renderer-ready [`vol::PointCloudModel`] from current parameters.
    ///
    /// Applies `softplus` to density and radius. Does not compute adjacency —
    /// the caller is responsible for calling
    /// [`vol::PointCloudModel::compute_adjacency_default`] before rendering.
    pub fn to_model(&self) -> vol::PointCloudModel {
        let n = self.positions.len();
        assert_eq!(self.log_density.len(), n);
        if let Some(ref r) = self.log_radii {
            assert_eq!(r.len(), n);
        }
        let mut points = Vec::with_capacity(n);
        for (i, p) in self.positions.iter().enumerate() {
            points.push(glam::Vec4::new(
                p.x,
                p.y,
                p.z,
                softplus(self.log_density[i]),
            ));
        }
        let radii = self
            .log_radii
            .as_ref()
            .map(|r| r.iter().map(|&v| softplus(v)).collect());
        vol::PointCloudModel {
            points,
            sh_coefficients: self.sh_coefficients.clone(),
            sh_degree: self.sh_degree,
            transforms: None,
            adjacency: None,
            radii,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_inverse_softplus_round_trip() {
        for &y in &[1e-3f32, 0.01, 0.1, 1.0, 5.0, 50.0] {
            let x = inverse_softplus(y);
            let back = softplus(x);
            assert!(
                (back - y).abs() / y < 1e-4,
                "softplus({}) = {}, expected {}",
                x,
                back,
                y
            );
        }
    }

    fn tiny_model(with_radii: bool) -> vol::PointCloudModel {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 0.50),
            glam::Vec4::new(1.0, 0.0, 0.0, 0.25),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.50),
        ];
        let n = points.len();
        vol::PointCloudModel {
            sh_coefficients: (0..n * 3).map(|i| (i as f32) * 0.01).collect(),
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: if with_radii {
                Some(vec![0.10, 0.25, 0.50])
            } else {
                None
            },
            points,
        }
    }

    #[test]
    fn trainer_state_round_trip_without_radii() {
        let model = tiny_model(false);
        let state = TrainerState::from_model(&model);
        let back = state.to_model();

        assert_eq!(back.points.len(), model.points.len());
        for (a, b) in back.points.iter().zip(model.points.iter()) {
            assert!((a.x - b.x).abs() < 1e-4);
            assert!((a.y - b.y).abs() < 1e-4);
            assert!((a.z - b.z).abs() < 1e-4);
            // softplus(inverse_softplus(w)) ≈ w
            assert!((a.w - b.w).abs() / b.w < 1e-3, "{} vs {}", a.w, b.w);
        }
        assert_eq!(back.sh_coefficients, model.sh_coefficients);
        assert_eq!(back.sh_degree, model.sh_degree);
        assert!(back.radii.is_none());
        assert!(back.adjacency.is_none());
        assert!(back.transforms.is_none());
    }

    #[test]
    fn trainer_state_round_trip_with_radii() {
        let model = tiny_model(true);
        let state = TrainerState::from_model(&model);
        assert!(state.log_radii.is_some());

        let back = state.to_model();
        let expected = model.radii.as_ref().unwrap();
        let actual = back.radii.as_ref().expect("radii preserved");
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() / b < 1e-3, "{} vs {}", a, b);
        }
    }

    #[test]
    fn round_trip_preserves_length_for_empty_radii_vs_some() {
        let mut model = tiny_model(true);
        let state_a = TrainerState::from_model(&model);
        model.radii = None;
        let state_b = TrainerState::from_model(&model);

        assert!(state_a.log_radii.is_some());
        assert!(state_b.log_radii.is_none());
        assert_eq!(state_a.len(), state_b.len());
    }
}
