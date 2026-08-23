//! CPU screen-space renderer used as the training forward pass.
//!
//! This is a thin wrapper that maps a `vol::CameraParams` + pixel grid into
//! rays and delegates each ray to the matching unweighted walk or weighted
//! compute-splat CPU oracle. Output is a
//! flat `Vec<f32>` of RGBA pixels in row-major order, matching the format the
//! photometric loss in M3c-2 will consume.
//!
//! Why a CPU forward pass for training:
//! - Differentiable. The GPU compute path uses storage textures and binding
//!   arrays meganeura can't currently autodiff through.
//! - Identical math to the WGSL shader (the trace lives in `blade_volume::trace`
//!   and is asserted GPU-equivalent in the regression suite).
//! - Slow but fine at training resolutions of 64–256 px.

use blade_volume as vol;

/// Settings shared across all rays of one frame. Per-pixel rays are derived
/// from the camera + image extent.
#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    /// Width of the rendered image in pixels.
    pub width: u32,
    /// Height of the rendered image in pixels.
    pub height: u32,
    /// Index of the cell each ray should *start* in. Future work: per-pixel
    /// entry-point search; for now the caller picks one cell that's a valid
    /// starting region for the camera (e.g. the cell containing the camera).
    pub start_point: u32,
    /// Maximum traversal steps per ray.
    pub max_steps: u32,
    /// Transmittance threshold for early-out.
    pub weight_threshold: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            start_point: 0,
            max_steps: 1024,
            weight_threshold: 0.001,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TraversalDiagnostics {
    pub rays: usize,
    pub total_steps: u64,
    pub max_steps_used: u32,
    /// Rays which exhausted `RenderSettings::max_steps` while still before
    /// the far plane and above the transmittance early-out threshold.
    pub truncated_rays: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderResult {
    pub rgba: Vec<f32>,
    pub traversal: TraversalDiagnostics,
}

/// Render `model` from `camera` at the resolution in `settings`. Returns
/// `width * height * 4` RGBA floats in row-major order (top-left first).
///
/// Camera mapping matches the WGSL shader:
/// ```text
/// px = (x + 0.5) / W;   py = (y + 0.5) / H
/// ndc = (px*2 - 1, py*2 - 1)
/// local_dir = (ndc.x * tan(0.5 * fov_x), ndc.y * tan(0.5 * fov_y), 1)
/// ray_dir   = normalize(rotate_by(orientation, local_dir))
/// ```
pub fn render_cpu(
    model: &vol::PointCloudModel,
    camera: &vol::CameraParams,
    settings: RenderSettings,
) -> Vec<f32> {
    render_cpu_with_diagnostics(model, camera, settings).rgba
}

/// Construct the camera ray through the centre of one output pixel.
pub fn camera_ray(
    camera: &vol::CameraParams,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> vol::trace::Ray {
    let tan_half = glam::Vec2::new((0.5 * camera.fov[0]).tan(), (0.5 * camera.fov[1]).tan());
    let orientation = glam::Quat::from_xyzw(
        camera.cam_orientation[0],
        camera.cam_orientation[1],
        camera.cam_orientation[2],
        camera.cam_orientation[3],
    );
    let px = (x as f32 + 0.5) / width as f32;
    let py = (y as f32 + 0.5) / height as f32;
    let local_dir = glam::Vec3::new(
        (px * 2.0 - 1.0 - camera.principal[0]) * tan_half.x,
        (py * 2.0 - 1.0 - camera.principal[1]) * tan_half.y,
        1.0,
    );
    vol::trace::Ray {
        origin: glam::Vec3::from_array(camera.cam_position),
        direction: (orientation * local_dir).normalize(),
    }
}

/// Render like [`render_cpu`] and retain aggregate traversal diagnostics.
/// A ray is counted as truncated only when the step cap—not opacity, the far
/// plane, or a terminal cell—stopped it.
pub fn render_cpu_with_diagnostics(
    model: &vol::PointCloudModel,
    camera: &vol::CameraParams,
    settings: RenderSettings,
) -> RenderResult {
    let w = settings.width as usize;
    let h = settings.height as usize;
    let mut out = vec![0.0f32; w * h * 4];
    let mut traversal = TraversalDiagnostics {
        rays: w * h,
        ..TraversalDiagnostics::default()
    };

    let trace_settings = vol::trace::TraceSettings {
        weight_threshold: settings.weight_threshold,
        max_steps: settings.max_steps,
        start_point: settings.start_point,
        depth: camera.depth,
        eval_mode: vol::trace::EvalMode::Sh,
    };

    for iy in 0..h {
        for ix in 0..w {
            let ray = camera_ray(
                camera,
                settings.width,
                settings.height,
                ix as u32,
                iy as u32,
            );
            let res = if model.radii.is_some() {
                vol::trace::trace_powerfoam_splats(model, ray, trace_settings)
            } else {
                vol::trace::trace_one_ray(model, ray, trace_settings)
            };
            traversal.total_steps += res.steps as u64;
            traversal.max_steps_used = traversal.max_steps_used.max(res.steps);
            let remaining_transmittance = 1.0 - res.rgba.w;
            if res.steps >= settings.max_steps
                && res.t_end < camera.depth
                && remaining_transmittance > settings.weight_threshold
            {
                traversal.truncated_rays += 1;
            }
            let base = (iy * w + ix) * 4;
            out[base] = res.rgba.x;
            out[base + 1] = res.rgba.y;
            out[base + 2] = res.rgba.z;
            out[base + 3] = res.rgba.w;
        }
    }

    RenderResult {
        rgba: out,
        traversal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere_model() -> vol::PointCloudModel {
        // Tiny tetrahedron of dense, red cells; small enough that the CPU
        // tracer finishes in milliseconds at 16x16.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 5.0),
            glam::Vec4::new(0.5, 0.0, 0.0, 5.0),
            glam::Vec4::new(0.0, 0.5, 0.0, 5.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 5.0),
        ];
        let n = points.len();
        // SH degree 0, red-ish DC: solve color = 0.5 + SH_C0 * coeff for
        // ~(1, 0, 0) target → coeff = (1 - 0.5)/SH_C0 for red, -0.5/SH_C0 elsewhere.
        const SH_C0: f32 = 0.282_094_8;
        let red = 0.5 / SH_C0;
        let dark = -0.5 / SH_C0;
        let mut sh = Vec::with_capacity(n * 3);
        for _ in 0..n {
            sh.extend_from_slice(&[red, dark, dark]);
        }
        let mut model = vol::PointCloudModel {
            points,
            sh_coefficients: sh,
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
        };
        model.compute_adjacency_default();
        model
    }

    fn camera_at(z: f32) -> vol::CameraParams {
        vol::CameraParams {
            cam_position: [0.1, 0.1, z],
            depth: 100.0,
            cam_orientation: [0.0, 0.0, 0.0, 1.0], // identity, +Z forward
            fov: [1.0, 1.0],
            principal: [0.0, 0.0],
        }
    }

    #[test]
    fn render_cpu_produces_finite_pixels() {
        let model = unit_sphere_model();
        let cam = camera_at(-1.0);
        let pixels = render_cpu(
            &model,
            &cam,
            RenderSettings {
                width: 16,
                height: 16,
                ..Default::default()
            },
        );
        assert_eq!(pixels.len(), 16 * 16 * 4);
        assert!(pixels.iter().all(|p| p.is_finite()));
        // Alpha must be in [0, 1].
        for px in pixels.as_chunks::<4>().0 {
            assert!((0.0..=1.0).contains(&px[3]));
        }
    }

    #[test]
    fn render_cpu_hits_at_least_one_cell() {
        // With a dense tetrahedron and a camera staring at it, alpha must be
        // > 0 for at least some pixels — otherwise traversal never integrated.
        let model = unit_sphere_model();
        let cam = camera_at(-1.0);
        let pixels = render_cpu(
            &model,
            &cam,
            RenderSettings {
                width: 16,
                height: 16,
                ..Default::default()
            },
        );
        let any_alpha = pixels.as_chunks::<4>().0.iter().any(|px| px[3] > 0.0);
        assert!(any_alpha, "render produced an entirely transparent image");
    }

    #[test]
    fn traversal_diagnostics_distinguish_step_cap_from_terminal_exit() {
        let model = unit_sphere_model();
        let cam = camera_at(-1.0);
        let capped = render_cpu_with_diagnostics(
            &model,
            &cam,
            RenderSettings {
                width: 16,
                height: 16,
                max_steps: 1,
                weight_threshold: 0.0,
                ..RenderSettings::default()
            },
        );
        assert_eq!(capped.traversal.rays, 16 * 16);
        assert!(capped.traversal.truncated_rays > 0);
        assert_eq!(capped.traversal.max_steps_used, 1);

        let complete = render_cpu_with_diagnostics(
            &model,
            &cam,
            RenderSettings {
                width: 16,
                height: 16,
                max_steps: 32,
                weight_threshold: 0.0,
                ..RenderSettings::default()
            },
        );
        assert_eq!(complete.traversal.truncated_rays, 0);
    }
}
