//! Mask-supervised PowerFoam continuation of an established Gaussian surface.
//!
//! Weighted cells failed as an initializer because a volume lattice has no
//! surface correspondence. Once multi-view fusion has established particles,
//! the same centers can remain fixed while PowerFoam learns support radii,
//! oriented planes, opacity, and view-dependent appearance. The learned radii
//! and normals then return to the Gaussian surface; no polygonal intermediate
//! is constructed and no second runtime representation is retained.

use crate::{diff_render, inverse::capture};
use blade_volume as vol;
use std::sync;

const SH_C0: f32 = 0.282_094_8;
const SH_DEGREE: usize = 2;
const SH_COMPONENTS: usize = (SH_DEGREE + 1) * (SH_DEGREE + 1);

/// The selected fixed-surface continuation schedule.
#[derive(Clone, Copy, Debug)]
pub struct ContinueOptions {
    /// Adam updates per training view. The selected synthetic gate uses 300.
    pub steps_per_view: usize,
    /// Maximum weighted cells retained along one sampled ray.
    pub max_steps: usize,
}

impl Default for ContinueOptions {
    fn default() -> Self {
        Self {
            steps_per_view: 300,
            max_steps: 192,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContinueStats {
    pub updates: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
}

pub struct ContinueOutcome {
    pub stats: ContinueStats,
    /// Trained static light field. The caller may save it alongside the
    /// relightable Gaussian surface.
    pub light_field: vol::PointCloudModel,
}

/// Continue an established Gaussian surface through oriented PowerFoam.
///
/// All selected views must carry independent foreground masks. Centers and
/// surface offsets remain fixed; density, SH, support radii, and normals are
/// optimized using the existing differentiable PowerFoam path.
pub fn continue_surface(
    surface: &mut vol::relight::RelightModel,
    capture: &capture::Capture,
    views: &[usize],
    options: ContinueOptions,
    gpu: sync::Arc<blade_graphics::Context>,
) -> Result<ContinueOutcome, String> {
    if surface.kernel != vol::relight::ParticleKernel::Gaussian {
        return Err("PowerFoam continuation requires Gaussian surface particles".to_string());
    }
    if surface.surfels.len() < 4 {
        return Err("PowerFoam continuation requires at least four particles".to_string());
    }
    if views.len() < 3 {
        return Err("PowerFoam continuation requires at least three training views".to_string());
    }
    if options.steps_per_view == 0 {
        return Err("PowerFoam continuation needs at least one update per view".to_string());
    }
    surface.validate()?;

    let supervision = supervision(capture, views)?;
    let mut cloud = cloud_from_surface(surface);
    cloud.compute_adjacency_default();
    cloud.validate()?;
    let losses = diff_render::fit_appearance_multi_view(
        &mut cloud,
        &supervision,
        capture.width as u32,
        capture.height as u32,
        options.max_steps,
        diff_render::AppearanceFitConfig {
            learning_rate: 0.1,
            pixel_batch: Some(2048),
            views_per_batch: views.len().min(6),
            steps_per_view: options.steps_per_view,
            sh_degree: SH_DEGREE,
            color_loss: diff_render::ColorLoss::SmoothL1,
            opacity_weight: 1.0,
            quantile_weight: 1.0e-4,
            softplus_beta: 10.0,
            radius_lr_ratio: 0.01,
            surface_normal_lr_ratio: 0.1,
            surface_normal_weight: 0.1,
            geometry_rebuild_every: 100,
            ..diff_render::AppearanceFitConfig::default()
        },
        gpu,
    );
    cloud.validate().map_err(|error| {
        format!("PowerFoam continuation produced an invalid light field: {error}")
    })?;
    if losses.iter().any(|loss| !loss.is_finite()) {
        return Err("PowerFoam continuation produced a non-finite loss".to_string());
    }
    update_surface(surface, &cloud);
    Ok(ContinueOutcome {
        stats: ContinueStats {
            updates: losses.len(),
            initial_loss: losses.first().copied().unwrap_or(f32::NAN),
            final_loss: losses.last().copied().unwrap_or(f32::NAN),
        },
        light_field: cloud,
    })
}

fn supervision(
    capture: &capture::Capture,
    views: &[usize],
) -> Result<Vec<diff_render::ViewSupervision>, String> {
    let mut result = Vec::with_capacity(views.len());
    for &index in views {
        let view = capture
            .views
            .get(index)
            .ok_or_else(|| format!("capture has no view {index}"))?;
        let mask = view
            .mask
            .as_ref()
            .ok_or_else(|| format!("view {} has no foreground mask", view.name))?;
        if mask.len() != capture.width * capture.height {
            return Err(format!(
                "view {} has a malformed foreground mask",
                view.name
            ));
        }
        let target_rgb = view
            .pixels
            .iter()
            .flat_map(|rgb| rgb.map(capture::linear_to_srgb))
            .collect();
        result.push(diff_render::ViewSupervision {
            camera: view.camera,
            target_rgb,
            target_alpha: Some(mask.to_vec()),
            width: capture.width as u32,
            height: capture.height as u32,
        });
    }
    Ok(result)
}

fn cloud_from_surface(surface: &vol::relight::RelightModel) -> vol::PointCloudModel {
    let mut points = Vec::with_capacity(surface.surfels.len());
    let mut radii = Vec::with_capacity(surface.surfels.len());
    let mut normals = Vec::with_capacity(surface.surfels.len());
    let mut sh = Vec::with_capacity(surface.surfels.len() * SH_COMPONENTS * 3);
    for surfel in &surface.surfels {
        points.push(glam::Vec3::from(surfel.center).extend(1.0));
        radii.push(surfel.radius);
        normals.push(glam::Vec3::from(surfel.normal));
        let albedo = surface.materials[surfel.material as usize].albedo;
        sh.extend(albedo.map(|value| (capture::linear_to_srgb(value) - 0.5) / SH_C0));
        sh.resize(sh.len() + (SH_COMPONENTS - 1) * 3, 0.0);
    }
    vol::PointCloudModel {
        points,
        sh_coefficients: sh,
        sh_degree: SH_DEGREE,
        transforms: None,
        adjacency: None,
        radii: Some(radii),
        surface_normals: Some(normals),
        surface_offsets: Some(vec![0.0; surface.surfels.len()]),
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    }
}

fn update_surface(surface: &mut vol::relight::RelightModel, cloud: &vol::PointCloudModel) {
    let radii = cloud.radii.as_ref().unwrap();
    let normals = cloud.surface_normals.as_ref().unwrap();
    assert_eq!(surface.surfels.len(), cloud.points.len());
    for (index, surfel) in surface.surfels.iter_mut().enumerate() {
        surfel.center = cloud.points[index].truncate().to_array();
        surfel.radius = radii[index];
        surfel.normal = normals[index].to_array();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_continuation_defaults_are_bounded() {
        let options = ContinueOptions::default();
        assert_eq!(options.steps_per_view, 300);
        assert_eq!(options.max_steps, 192);
    }

    fn surface() -> vol::relight::RelightModel {
        let material = vol::relight::Material {
            albedo: [0.522_697_3, 0.523_159_44, 0.523_708_94],
            ..Default::default()
        };
        vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![vol::relight::Surfel {
                center: [-7.044_026, -3.180_242_8, -1.539_512_4],
                radius: 0.502_746_05,
                normal: [0.373_487_26, -0.528_455_6, 0.762_392_2],
                material: 0,
            }],
            materials: vec![material],
        }
    }

    #[test]
    fn surface_initializer_matches_the_selected_gate_oracle() {
        let surface = surface();
        let cloud = cloud_from_surface(&surface);
        assert_eq!(cloud.sh_degree, 2);
        assert_eq!(cloud.points[0].w, 1.0);
        assert_eq!(cloud.radii.as_deref(), Some(&[0.502_746_05][..]));
        assert_eq!(
            cloud.surface_normals.as_deref(),
            Some(&[glam::Vec3::new(0.373_487_26, -0.528_455_6, 0.762_392_2)][..])
        );
        let expected = [0.886_626_84, 0.887_677_85, 0.888_926_8];
        for (&actual, expected) in cloud.sh_coefficients[..3].iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert!(cloud.sh_coefficients[3..].iter().all(|&value| value == 0.0));
        cloud.validate().unwrap();
    }

    #[test]
    fn supervision_requires_independent_masks_and_encodes_linear_rgb() {
        let camera = vol::CameraParams::default();
        let mut capture = capture::Capture {
            width: 1,
            height: 1,
            views: vec![capture::View {
                name: "one".to_string(),
                camera,
                pixels: vec![[0.25, 0.5, 1.0]],
                mask: None,
            }],
        };
        assert!(supervision(&capture, &[0]).is_err());
        capture.views[0].mask = Some(vec![0.75].into());
        let views = supervision(&capture, &[0]).unwrap();
        assert_eq!(views[0].target_alpha.as_deref(), Some(&[0.75][..]));
        for (actual, linear) in views[0].target_rgb.iter().zip([0.25, 0.5, 1.0]) {
            assert!((*actual - capture::linear_to_srgb(linear)).abs() < 1.0e-7);
        }
    }

    #[test]
    fn masked_surface_continuation_runs_on_the_physical_gpu() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping masked surface continuation test: no GPU");
            return;
        };
        let centers = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ];
        let mut surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: centers
                .iter()
                .map(|&center| vol::relight::Surfel {
                    center,
                    radius: 0.8,
                    normal: [0.0, 0.0, -1.0],
                    material: 0,
                })
                .collect(),
            materials: vec![vol::relight::Material::default()],
        };
        let camera = |position: glam::Vec3| vol::CameraParams {
            cam_position: position.to_array(),
            depth: 10.0,
            cam_orientation: glam::Quat::from_rotation_arc(
                glam::Vec3::Z,
                (glam::Vec3::splat(0.1) - position).normalize(),
            )
            .to_array(),
            fov: [0.5; 2],
            principal: [0.0; 2],
        };
        let capture = capture::Capture {
            width: 1,
            height: 1,
            views: [
                glam::Vec3::new(0.05, 0.05, -1.0),
                glam::Vec3::new(-0.25, 0.05, -1.0),
                glam::Vec3::new(0.35, 0.05, -1.0),
            ]
            .into_iter()
            .enumerate()
            .map(|(index, position)| capture::View {
                name: format!("masked-{index}"),
                camera: camera(position),
                pixels: vec![[0.5; 3]],
                mask: Some(vec![1.0].into()),
            })
            .collect(),
        };
        let outcome = continue_surface(
            &mut surface,
            &capture,
            &[0, 1, 2],
            ContinueOptions {
                steps_per_view: 1,
                max_steps: 16,
            },
            gpu,
        )
        .unwrap();
        assert_eq!(outcome.stats.updates, 3);
        assert!(outcome.stats.initial_loss.is_finite());
        assert_eq!(
            surface
                .surfels
                .iter()
                .map(|surfel| surfel.center)
                .collect::<Vec<_>>(),
            centers
        );
        surface.validate().unwrap();
        outcome.light_field.validate().unwrap();
    }
}
