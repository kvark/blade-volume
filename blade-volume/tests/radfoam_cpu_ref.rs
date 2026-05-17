//! Thin re-export of the public CPU tracer.
//!
//! The implementation now lives in `blade_volume::trace`. This file exists only
//! to let the sibling integration tests keep their `mod radfoam_cpu_ref; use
//! radfoam_cpu_ref as cpu;` pattern, and to host a smoke test against a fixture
//! PLY shipped under `tests/data/`.

#![allow(dead_code)]

pub use blade_volume::trace::{trace_one_ray, EvalMode, Ray, TraceSettings};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_traces_without_nan_on_tiny_fixture() {
        let model = blade_volume::io::load_radfoam("tests/data/radfoam_tiny_ascii.ply");

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
