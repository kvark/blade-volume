//! Multi-view refinement of an oriented point cloud.
//!
//! A fused depth map is only an initializer: averaging nearby rays does not
//! make their shared surface precise. This stage keeps the representation as
//! oriented particles and searches along each particle normal. For every
//! candidate position it projects one world-space tangent patch into several
//! photographs. The right depth is the one whose reprojected patches agree.

use crate::inverse::{capture, decompose, depth, score};
use blade_volume as vol;
use std::thread;

const MAX_RENDERED_ALBEDO_COORDINATES: usize = 96;
const RENDERED_NORMAL_COVERAGE_WEIGHT: f32 = 0.1;

/// Whether the final Gaussian material pass is the bounded, independently
/// validated large-table transfer rather than the shared-palette solver.
pub fn automatic_gaussian_material_transfer(
    material_count: usize,
    individual_materials: bool,
) -> bool {
    individual_materials && material_count > MAX_RENDERED_ALBEDO_COORDINATES / 3
}

/// One photograph used by [`refine`].
#[derive(Clone, Copy)]
pub struct RefinementView<'a> {
    pub capture_index: usize,
    /// Optional source-foam depth. When present, it rejects views in which the
    /// particle was behind a different surface.
    pub depth: Option<&'a depth::DepthMap>,
}

/// Controls for normal-direction plane sweep.
#[derive(Clone, Copy, Debug)]
pub struct RefineOptions {
    /// Search half-width as a fraction of each surfel radius.
    pub search_radius_factor: f32,
    /// Odd number of uniformly spaced normal offsets, including zero.
    pub candidates: usize,
    /// Odd tangent-patch width. Three means a 3×3 world-space patch.
    pub patch_side: usize,
    /// Tangent-patch half-width as a fraction of the surfel radius.
    pub patch_radius_factor: f32,
    /// At least this many photographs must carry a textured patch.
    pub min_views: usize,
    /// Keep at most this many, selected by square-on facing.
    pub max_views: usize,
    /// Minimum cosine between the normal and direction to the camera.
    pub min_facing: f32,
    /// Normalized patch RMS below this is textureless and carries no depth.
    pub min_texture: f32,
    /// Relative cost reduction required before moving a surfel.
    pub min_improvement: f32,
    /// Small normalized quadratic preference for the initializer.
    pub offset_prior: f32,
    /// Extra source-depth tolerance, beyond the search half-width, as a
    /// fraction of surfel radius.
    pub visibility_radius_factor: f32,
    pub visibility_min_alpha: f32,
    pub visibility_min_peak: f32,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            search_radius_factor: 0.5,
            candidates: 9,
            patch_side: 3,
            patch_radius_factor: 0.5,
            min_views: 4,
            max_views: 8,
            min_facing: 0.15,
            min_texture: 0.01,
            min_improvement: 0.02,
            offset_prior: 0.001,
            visibility_radius_factor: 0.0,
            visibility_min_alpha: 0.5,
            visibility_min_peak: 0.05,
        }
    }
}

/// What one refinement pass changed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefineStats {
    pub surfels: usize,
    pub scored: usize,
    pub moved: usize,
    pub mean_absolute_offset: f32,
    pub mean_relative_improvement: f32,
    pub mean_views: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderedStats {
    pub particles: usize,
    /// Observed particles perturbed together in each simultaneous round.
    pub simultaneous_particles: usize,
    /// Requested simultaneous rounds.
    pub simultaneous_rounds: usize,
    /// Rounds whose localized or whole-frame proposal reduced the objective.
    pub simultaneous_accepted: usize,
    pub tested: usize,
    pub moved: usize,
    pub radii_moved: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderedMaterialStats {
    /// Scalar diffuse-albedo coordinates in scope.
    pub coordinates: usize,
    /// Distinct material-table proposals scored.
    pub proposals: usize,
    /// Coordinates whose final albedo differs from its input by more than
    /// numerical noise.
    pub changed: usize,
    /// Accepted global diffuse gain for a large table, otherwise `None`.
    pub global_gain: Option<f32>,
    /// Accepted diffuse-to-specular allocation for a large rough-dielectric
    /// table, otherwise `None`.
    pub global_specular_allocation: Option<f32>,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderedMaterialAssignmentStats {
    pub particles: usize,
    pub candidates: usize,
    pub proposals: usize,
    pub changed: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub seconds: f64,
}

/// One known-light capture used to refine Gaussian surface normals against
/// complete production renders.
#[derive(Clone, Copy)]
pub struct RenderedNormalEvidence<'a> {
    pub capture: &'a capture::Capture,
    pub indices: &'a [usize],
    pub environment: &'a vol::relight::Environment,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderedNormalStats {
    pub normals: usize,
    pub rounds: usize,
    pub accepted: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderedRadiusStats {
    pub radii: usize,
    pub rounds: usize,
    pub accepted: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub seconds: f64,
}

fn rendered_loss(
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
) -> f64 {
    tracer.reset_sampling();
    renderer.prepared_srgb_loss(tracer, capture, indices, cameras)
}

fn rendered_errors(
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
) -> Vec<f32> {
    tracer.reset_sampling();
    renderer.prepared_srgb_errors(tracer, capture, indices, cameras, 0.0)
}

fn rendered_evidence_errors(
    renderer: &mut score::Renderer,
    tracers: &mut [vol::gpu::RelightTracer],
    evidence: &[RenderedNormalEvidence<'_>],
    cameras: &[Vec<vol::CameraParams>],
) -> Vec<f32> {
    let capacity = evidence
        .iter()
        .map(|entry| entry.indices.len() * entry.capture.width * entry.capture.height)
        .sum();
    let mut errors = Vec::with_capacity(capacity);
    for ((tracer, entry), cameras) in tracers.iter_mut().zip(evidence).zip(cameras) {
        tracer.reset_sampling();
        errors.extend(renderer.prepared_srgb_errors(
            tracer,
            entry.capture,
            entry.indices,
            cameras,
            RENDERED_NORMAL_COVERAGE_WEIGHT,
        ));
    }
    errors
}

fn update_rendered_geometry(
    renderer: &mut score::Renderer,
    tracers: &mut [vol::gpu::RelightTracer],
    surfels: &[vol::relight::Surfel],
) {
    for tracer in tracers {
        renderer.update_prepared_surfel_geometry(tracer, surfels);
    }
}

fn perturbation_hash(index: usize, round: usize) -> u64 {
    let mut value = index as u64 ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn normal_perturbation(normal: glam::Vec3, index: usize, round: usize, angle: f32) -> glam::Vec3 {
    let tangent = tangent_to(normal);
    let bitangent = normal.cross(tangent);
    let phase = perturbation_hash(index, round) as f64 / u64::MAX as f64;
    let phase = (phase * std::f64::consts::TAU) as f32;
    let axis = phase.cos() * tangent + phase.sin() * bitangent;
    (angle.cos() * normal + angle.sin() * axis).normalize()
}

#[derive(Clone, Copy, Debug, Default)]
struct RenderedParameterStats {
    parameters: usize,
    accepted: usize,
    initial_loss: f64,
    final_loss: f64,
    seconds: f64,
}

fn refine_rendered_parameter(
    scene: &mut score::Scene,
    evidence: &[RenderedNormalEvidence<'_>],
    observations: &decompose::Observations,
    diffuse_samples: u32,
    rounds: usize,
    parameter: &str,
    perturb: impl Fn(&vol::relight::Surfel, usize, usize, f32) -> vol::relight::Surfel,
) -> Result<RenderedParameterStats, String> {
    if observations.surfels() != scene.model.surfels.len() {
        return Err(format!(
            "rendered {parameter} observation count does not match the model"
        ));
    }
    if evidence.is_empty() || rounds == 0 || scene.model.surfels.is_empty() {
        return Ok(RenderedParameterStats::default());
    }
    let width = evidence[0].capture.width;
    let height = evidence[0].capture.height;
    for entry in evidence {
        if entry.capture.width != width || entry.capture.height != height {
            return Err(format!(
                "rendered {parameter} captures must have one resolution"
            ));
        }
        for &index in entry.indices {
            if index >= entry.capture.views.len() {
                return Err(format!(
                    "rendered {parameter} view {index} is out of bounds"
                ));
            }
        }
    }

    let cameras: Vec<Vec<vol::CameraParams>> = evidence
        .iter()
        .map(|entry| {
            entry
                .indices
                .iter()
                .map(|&index| entry.capture.views[index].camera)
                .collect()
        })
        .collect();
    let flat_cameras: Vec<vol::CameraParams> = cameras.iter().flatten().copied().collect();
    let candidates: Vec<usize> = (0..scene.model.surfels.len())
        .filter(|&index| !observations.of(index).is_empty())
        .collect();
    if candidates.is_empty() {
        return Ok(RenderedParameterStats::default());
    }

    let mut renderer = score::Renderer::new(width, height)?;
    let mut tracers: Vec<_> = evidence
        .iter()
        .map(|entry| {
            let prepared_scene = if scene.environment().width == entry.environment.width
                && scene.environment().height == entry.environment.height
                && scene.environment().texels == entry.environment.texels
            {
                scene.clone()
            } else {
                score::Scene::new(scene.model.clone(), entry.environment.clone())
            };
            renderer.prepare_scene(&prepared_scene, diffuse_samples, false)
        })
        .collect();
    let started = std::time::Instant::now();
    let initial_errors = rendered_evidence_errors(&mut renderer, &mut tracers, evidence, &cameras);
    let mut loss = mean_error(&initial_errors);
    let initial_loss = loss;
    let mut accepted = 0usize;
    for round in 0..rounds {
        let current = scene.model.surfels.clone();
        let mut plus = current.clone();
        let mut minus = current.clone();
        for &index in &candidates {
            plus[index] = perturb(&current[index], index, round, 1.0);
            minus[index] = perturb(&current[index], index, round, -1.0);
        }

        update_rendered_geometry(&mut renderer, &mut tracers, &plus);
        let plus_errors = rendered_evidence_errors(&mut renderer, &mut tracers, evidence, &cameras);
        let plus_loss = mean_error(&plus_errors);
        update_rendered_geometry(&mut renderer, &mut tracers, &minus);
        let minus_errors =
            rendered_evidence_errors(&mut renderer, &mut tracers, evidence, &cameras);
        let minus_loss = mean_error(&minus_errors);
        let difference = ErrorDifference::new(&plus_errors, &minus_errors, width, height);

        let mut localized = current.clone();
        for &index in &candidates {
            match projected_error_difference(
                &difference,
                evidence[0].capture,
                &flat_cameras,
                &current[index],
            ) {
                Some(value) if value < 0.0 => localized[index] = plus[index],
                Some(value) if value > 0.0 => localized[index] = minus[index],
                _ => {}
            }
        }
        update_rendered_geometry(&mut renderer, &mut tracers, &localized);
        let localized_errors =
            rendered_evidence_errors(&mut renderer, &mut tracers, evidence, &cameras);
        let localized_loss = mean_error(&localized_errors);

        let (next, next_loss) =
            if localized_loss < loss && localized_loss <= plus_loss && localized_loss <= minus_loss
            {
                (localized, localized_loss)
            } else if plus_loss < loss && plus_loss <= minus_loss {
                (plus, plus_loss)
            } else if minus_loss < loss {
                (minus, minus_loss)
            } else {
                (current, loss)
            };
        if next_loss < loss {
            accepted += 1;
            loss = next_loss;
        }
        scene.model.surfels = next;
        update_rendered_geometry(&mut renderer, &mut tracers, &scene.model.surfels);
    }
    for tracer in tracers {
        renderer.destroy_prepared_scene(tracer);
    }
    renderer.destroy();
    Ok(RenderedParameterStats {
        parameters: candidates.len(),
        accepted,
        initial_loss,
        final_loss: loss,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Refine observed Gaussian normals against complete renders from known-light
/// captures. Each round renders one antithetic perturbation pair, chooses a
/// direction per projected particle from its local error difference, and only
/// keeps a proposal that lowers the full multi-light objective. Centers,
/// radii, materials, assignments, and illumination stay fixed.
pub fn refine_rendered_normals(
    scene: &mut score::Scene,
    evidence: &[RenderedNormalEvidence<'_>],
    observations: &decompose::Observations,
    diffuse_samples: u32,
    rounds: usize,
    step_degrees: f32,
) -> Result<RenderedNormalStats, String> {
    if !step_degrees.is_finite() || step_degrees <= 0.0 || step_degrees >= 45.0 {
        return Err(
            "rendered normal step must be finite and between zero and 45 degrees".to_string(),
        );
    }
    let step = step_degrees.to_radians();
    let stats = refine_rendered_parameter(
        scene,
        evidence,
        observations,
        diffuse_samples,
        rounds,
        "normal",
        |surfel, index, round, direction| {
            let mut perturbed = *surfel;
            let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
            let scale = if round < rounds.div_ceil(2) { 1.0 } else { 0.5 };
            perturbed.normal =
                normal_perturbation(normal, index, round, direction * scale * step).to_array();
            perturbed
        },
    )?;
    Ok(RenderedNormalStats {
        normals: stats.parameters,
        rounds,
        accepted: stats.accepted,
        initial_loss: stats.initial_loss,
        final_loss: stats.final_loss,
        seconds: stats.seconds,
    })
}

/// Refine observed Gaussian support radii against complete production renders.
/// Centers, normals, materials, assignments, and illumination stay fixed.
pub fn refine_rendered_radii(
    scene: &mut score::Scene,
    evidence: &[RenderedNormalEvidence<'_>],
    observations: &decompose::Observations,
    diffuse_samples: u32,
    rounds: usize,
    step_fraction: f32,
) -> Result<RenderedRadiusStats, String> {
    if !step_fraction.is_finite() || step_fraction <= 0.0 || step_fraction >= 0.5 {
        return Err(
            "rendered radius step must be finite and between zero and one half".to_string(),
        );
    }
    let stats = refine_rendered_parameter(
        scene,
        evidence,
        observations,
        diffuse_samples,
        rounds,
        "radius",
        |surfel, _index, round, direction| {
            let mut perturbed = *surfel;
            let scale = if round < rounds.div_ceil(2) { 1.0 } else { 0.5 };
            perturbed.radius *= 1.0 + direction * scale * step_fraction;
            perturbed
        },
    )?;
    Ok(RenderedRadiusStats {
        radii: stats.parameters,
        rounds,
        accepted: stats.accepted,
        initial_loss: stats.initial_loss,
        final_loss: stats.final_loss,
        seconds: stats.seconds,
    })
}

/// Refine a small shared diffuse-material table against complete production
/// renders of the training photographs. Larger tables are left unchanged here;
/// their final Gaussian output receives a bounded global transfer instead.
pub fn refine_rendered_materials(
    scene: &mut score::Scene,
    capture: &capture::Capture,
    indices: &[usize],
    diffuse_samples: u32,
    step: f32,
) -> Result<RenderedMaterialStats, String> {
    refine_rendered_materials_impl(
        scene,
        None,
        capture,
        indices,
        diffuse_samples,
        step,
        true,
        false,
    )
}

/// Re-polish a rendered material fit after geometry or particle support moved.
///
/// Unlike [`refine_rendered_materials`], this keeps the existing material
/// estimate as its initializer. The linear initializer is useful for the first
/// fit, but repeating it after a small support update can needlessly replace a
/// solution that was already selected through complete renders.
pub fn polish_rendered_materials(
    scene: &mut score::Scene,
    capture: &capture::Capture,
    indices: &[usize],
    diffuse_samples: u32,
    step: f32,
) -> Result<RenderedMaterialStats, String> {
    refine_rendered_materials_impl(
        scene,
        None,
        capture,
        indices,
        diffuse_samples,
        step,
        false,
        false,
    )
}

/// Reassign a conservative prefix of particles to existing shared materials.
///
/// Observation-space error only ranks candidates. Complete production renders
/// decide whether any prefix is retained, then the shared table is re-polished
/// at `step` after an accepted assignment.
pub fn refine_rendered_material_assignments(
    scene: &mut score::Scene,
    capture: &capture::Capture,
    indices: &[usize],
    observations: &decompose::Observations,
    diffuse_samples: u32,
    step: f32,
) -> Result<RenderedMaterialAssignmentStats, String> {
    if observations.surfels() != scene.model.surfels.len() {
        return Err("rendered material assignment requires matching observations".to_string());
    }
    for &index in indices {
        if index >= capture.views.len() {
            return Err(format!("rendered material view {index} is out of bounds"));
        }
    }
    let one_material_per_particle = scene.model.materials.len() == scene.model.surfels.len()
        && scene
            .model
            .surfels
            .iter()
            .enumerate()
            .all(|(index, surfel)| surfel.material as usize == index);
    if scene.model.materials.len() < 2 || one_material_per_particle || indices.is_empty() {
        return Ok(RenderedMaterialAssignmentStats {
            particles: scene.model.surfels.len(),
            ..RenderedMaterialAssignmentStats::default()
        });
    }

    let cameras: Vec<_> = indices
        .iter()
        .map(|&index| capture.views[index].camera)
        .collect();
    let mut renderer = score::Renderer::new(capture.width, capture.height)?;
    let mut tracer = renderer.prepare_scene(scene, diffuse_samples, false);
    let started = std::time::Instant::now();
    let initial_loss = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
    let irradiance = scene.environment().diffuse_irradiance();
    let mut candidates = Vec::new();
    for (index, surfel) in scene.model.surfels.iter().enumerate() {
        let samples = observations.of(index);
        if samples.is_empty() {
            continue;
        }
        let current = surfel.material as usize;
        let mut best = current;
        let mut best_error = material_observation_error(
            surfel,
            &scene.model.materials[current],
            samples,
            &irradiance,
            scene.specular(),
        );
        let current_error = best_error;
        for (material, candidate) in scene.model.materials.iter().enumerate() {
            if material == current {
                continue;
            }
            let error = material_observation_error(
                surfel,
                candidate,
                samples,
                &irradiance,
                scene.specular(),
            );
            if error < best_error {
                best = material;
                best_error = error;
            }
        }
        if best != current {
            candidates.push((current_error - best_error, index, best as u32));
        }
    }
    candidates.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let original = scene.model.surfels.clone();
    let mut best_surfels = original.clone();
    let mut final_loss = initial_loss;
    let mut proposals = 0;
    let mut count = candidates.len();
    while count > 0 {
        scene.model.surfels.clone_from(&original);
        for &(_, index, material) in &candidates[..count] {
            scene.model.surfels[index].material = material;
        }
        renderer.update_prepared_surfels(&scene.model.surfels);
        let loss = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
        proposals += 1;
        if loss < final_loss {
            final_loss = loss;
            best_surfels.clone_from(&scene.model.surfels);
        }
        count /= 2;
    }
    scene.model.surfels = best_surfels;
    let changed = original
        .iter()
        .zip(&scene.model.surfels)
        .filter(|&(before, after)| before.material != after.material)
        .count();
    renderer.destroy_prepared_scene(tracer);
    renderer.destroy();
    if changed > 0 {
        final_loss =
            polish_rendered_materials(scene, capture, indices, diffuse_samples, step)?.final_loss;
    }
    Ok(RenderedMaterialAssignmentStats {
        particles: scene.model.surfels.len(),
        candidates: candidates.len(),
        proposals,
        changed,
        initial_loss,
        final_loss,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn material_observation_error(
    surfel: &vol::relight::Surfel,
    material: &vol::relight::Material,
    samples: &[decompose::Sample],
    irradiance: &[[f32; 4]; 9],
    specular: &vol::relight::SpecularEnvironment,
) -> f64 {
    let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
    samples
        .iter()
        .map(|sample| {
            let predicted =
                vol::relight::shade(normal, sample.towards, material, irradiance, specular);
            predicted
                .iter()
                .zip(sample.radiance)
                .map(|(&predicted, observed)| {
                    let difference =
                        capture::linear_to_srgb(predicted) - capture::linear_to_srgb(observed);
                    f64::from(sample.facing * difference * difference)
                })
                .sum::<f64>()
        })
        .sum()
}

/// Re-polish diffuse materials through the exact learned-Gaussian PBR renderer
/// after its geometry and support have reached their final values. Large
/// individual-mode tables receive bounded global proposals instead of an
/// unbounded coordinate pass. `transfer_specular` opts a still-rough-dielectric
/// table into one conservative diffuse-to-specular response proposal.
pub fn polish_gaussian_materials(
    scene: &score::Scene,
    gaussian: &mut vol::PointCloudModel,
    capture: &capture::Capture,
    indices: &[usize],
    step: f32,
    transfer_specular: bool,
) -> Result<RenderedMaterialStats, String> {
    gaussian.validate()?;
    let material_count = gaussian
        .transforms
        .as_ref()
        .and_then(|transforms| transforms.pbr.as_ref())
        .ok_or_else(|| "Gaussian material polish requires PBR attributes".to_string())?
        .materials
        .len();
    if material_count != scene.model.materials.len() {
        return Err("Gaussian and surface material tables must have matching lengths".to_string());
    }
    let mut gaussian_scene = scene.clone();
    let stats = refine_rendered_materials_impl(
        &mut gaussian_scene,
        Some(gaussian),
        capture,
        indices,
        0,
        step,
        false,
        transfer_specular,
    )?;
    gaussian
        .transforms
        .as_mut()
        .unwrap()
        .pbr
        .as_mut()
        .unwrap()
        .materials
        .clone_from(&gaussian_scene.model.materials);
    Ok(stats)
}

fn refine_rendered_materials_impl(
    scene: &mut score::Scene,
    gaussian: Option<&vol::PointCloudModel>,
    capture: &capture::Capture,
    indices: &[usize],
    diffuse_samples: u32,
    step: f32,
    initialize_linear: bool,
    transfer_specular: bool,
) -> Result<RenderedMaterialStats, String> {
    if !step.is_finite() || step <= 0.0 || step >= 1.0 {
        return Err("rendered material step must be finite and between zero and one".to_string());
    }
    for &index in indices {
        if index >= capture.views.len() {
            return Err(format!("rendered material view {index} is out of bounds"));
        }
    }
    if scene.model.materials.is_empty() || indices.is_empty() {
        return Ok(RenderedMaterialStats::default());
    }
    if gaussian.is_none() && scene.model.materials.len() * 3 > MAX_RENDERED_ALBEDO_COORDINATES {
        return Ok(RenderedMaterialStats {
            coordinates: scene.model.materials.len() * 3,
            ..RenderedMaterialStats::default()
        });
    }
    let cameras: Vec<_> = indices
        .iter()
        .map(|&index| capture.views[index].camera)
        .collect();
    let mut renderer = score::Renderer::new(capture.width, capture.height)?;
    let mut tracer = if let Some(gaussian) = gaussian {
        renderer.prepare_gaussian_scene(scene, gaussian, false)
    } else {
        renderer.prepare_scene(scene, diffuse_samples, false)
    };
    let started = std::time::Instant::now();
    let mut loss = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
    let initial_loss = loss;
    let initial_materials = scene.model.materials.clone();
    let basis = build_rendered_albedo_basis(
        scene,
        &mut renderer,
        &mut tracer,
        capture,
        indices,
        &cameras,
    );
    scene.model.materials.clone_from(&initial_materials);
    renderer.update_prepared_materials(&mut tracer, &scene.model.materials);
    if initialize_linear {
        if let Some(ref basis) = basis {
            if let Some(candidate_loss) = fit_rendered_linear_albedo(
                scene,
                &mut renderer,
                &mut tracer,
                capture,
                indices,
                &cameras,
                basis,
                loss,
            ) {
                loss = candidate_loss;
            }
        }
    }
    let coordinates;
    let mut proposals;
    let global_gain;
    let global_specular_allocation;
    if let Some(ref basis) = basis {
        global_gain = None;
        global_specular_allocation = None;
        let before_coordinates = scene.model.materials.clone();
        let (next_coordinates, next_proposals) = refine_albedo_from_basis(scene, basis, step);
        coordinates = next_coordinates;
        proposals = next_proposals;
        renderer.update_prepared_materials(&mut tracer, &scene.model.materials);
        let candidate_loss = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
        if candidate_loss < loss {
            loss = candidate_loss;
        } else {
            scene.model.materials = before_coordinates;
            renderer.update_prepared_materials(&mut tracer, &scene.model.materials);
        }
    } else {
        let result = refine_rendered_global_albedo(
            scene,
            &mut renderer,
            &mut tracer,
            capture,
            indices,
            &cameras,
            loss,
        );
        loss = result.0;
        coordinates = result.1;
        proposals = result.2;
        global_gain = result.3;
        if transfer_specular {
            let result = refine_rendered_global_specular_allocation(
                scene,
                &mut renderer,
                &mut tracer,
                capture,
                indices,
                &cameras,
                loss,
            );
            loss = result.0;
            proposals += result.1;
            global_specular_allocation = result.2;
        } else {
            global_specular_allocation = None;
        }
    }
    let changed = initial_materials
        .iter()
        .zip(&scene.model.materials)
        .flat_map(|(before, after)| before.albedo.iter().zip(after.albedo))
        .filter(|&(before, after)| (*before - after).abs() > 1.0e-6)
        .count();
    renderer.destroy_prepared_scene(tracer);
    renderer.destroy();
    Ok(RenderedMaterialStats {
        coordinates,
        proposals,
        changed,
        global_gain,
        global_specular_allocation,
        initial_loss,
        final_loss: loss,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn rendered_linear_rgb(
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    cameras: &[vol::CameraParams],
) -> Vec<f32> {
    tracer.reset_sampling();
    renderer
        .render_prepared_flat(tracer, cameras)
        .iter()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect()
}

struct RenderedAlbedoBasis {
    base: Vec<f32>,
    responses: Vec<Vec<f32>>,
    target: Vec<f32>,
    target_srgb: Vec<f32>,
}

impl RenderedAlbedoBasis {
    fn render(&self, materials: &[vol::relight::Material]) -> Vec<f32> {
        let mut rendered = self.base.clone();
        for (material, response) in self.responses.iter().enumerate() {
            for (index, (value, &basis)) in rendered.iter_mut().zip(response).enumerate() {
                *value += materials[material].albedo[index % 3] * basis;
            }
        }
        rendered
    }

    fn squared_error(&self, rendered: f32, index: usize) -> f64 {
        let difference = capture::linear_to_srgb(rendered) - self.target_srgb[index];
        f64::from(difference * difference)
    }

    fn error_sum(&self, rendered: &[f32]) -> f64 {
        debug_assert_eq!(rendered.len(), self.target.len());
        rendered
            .iter()
            .enumerate()
            .map(|(index, &rendered)| self.squared_error(rendered, index))
            .sum::<f64>()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rendered_albedo_basis(
    scene: &mut score::Scene,
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
) -> Option<RenderedAlbedoBasis> {
    let coordinates = scene.model.materials.len() * 3;
    if coordinates == 0 || coordinates > MAX_RENDERED_ALBEDO_COORDINATES {
        return None;
    }
    for material in &mut scene.model.materials {
        material.albedo = [0.0; 3];
    }
    renderer.update_prepared_materials(tracer, &scene.model.materials);
    let base = rendered_linear_rgb(renderer, tracer, cameras);
    let mut responses = Vec::with_capacity(scene.model.materials.len());
    for material in 0..scene.model.materials.len() {
        // Diffuse transport is diagonal in RGB, so one white-albedo render
        // supplies the three independent coordinate responses.
        scene.model.materials[material].albedo = [1.0; 3];
        renderer.update_prepared_materials(tracer, &scene.model.materials);
        let rendered = rendered_linear_rgb(renderer, tracer, cameras);
        responses.push(
            rendered
                .iter()
                .zip(&base)
                .map(|(value, base)| value - base)
                .collect(),
        );
        scene.model.materials[material].albedo = [0.0; 3];
    }
    let target: Vec<f32> = indices
        .iter()
        .flat_map(|&index| capture.views[index].pixels.iter())
        .flat_map(|pixel| *pixel)
        .collect();
    let target_srgb = target
        .iter()
        .map(|&value| capture::linear_to_srgb(value))
        .collect();
    debug_assert_eq!(target.len(), base.len());
    Some(RenderedAlbedoBasis {
        base,
        responses,
        target,
        target_srgb,
    })
}

fn refine_albedo_from_basis(
    scene: &mut score::Scene,
    basis: &RenderedAlbedoBasis,
    step: f32,
) -> (usize, usize) {
    let mut rendered = basis.render(&scene.model.materials);
    let mut error_sum = basis.error_sum(&rendered);
    let mut coordinates = 0;
    let mut proposals = 0;
    for step in [step, 0.5 * step] {
        for material in 0..scene.model.materials.len() {
            for channel in 0..3 {
                coordinates += 1;
                let original = scene.model.materials[material].albedo[channel];
                let candidates = [(original - step).max(0.0), (original + step).min(1.0)];
                let mut best = original;
                let mut best_error_sum = error_sum;
                let mut candidate_error_sums = [error_sum; 2];
                for index in (channel..rendered.len()).step_by(3) {
                    let current_error = basis.squared_error(rendered[index], index);
                    for (slot, &candidate) in candidates.iter().enumerate() {
                        if candidate == original {
                            continue;
                        }
                        candidate_error_sums[slot] -= current_error;
                        let candidate_rendered = rendered[index]
                            + (candidate - original) * basis.responses[material][index];
                        candidate_error_sums[slot] +=
                            basis.squared_error(candidate_rendered, index);
                    }
                }
                for (slot, candidate) in candidates.into_iter().enumerate() {
                    if candidate == original {
                        continue;
                    }
                    proposals += 1;
                    let candidate_error_sum = candidate_error_sums[slot];
                    if candidate_error_sum < best_error_sum {
                        best_error_sum = candidate_error_sum;
                        best = candidate;
                    }
                }
                if best != original {
                    let delta = best - original;
                    for index in (channel..rendered.len()).step_by(3) {
                        rendered[index] += delta * basis.responses[material][index];
                    }
                    scene.model.materials[material].albedo[channel] = best;
                    error_sum = best_error_sum;
                }
            }
        }
    }
    (coordinates, proposals)
}

fn fit_global_albedo_gain(base: &[f32], response: &[f32], target: &[f32], maximum: f32) -> f32 {
    debug_assert_eq!(base.len(), response.len());
    debug_assert_eq!(base.len(), target.len());
    if base.is_empty() {
        return 1.0;
    }
    let error = |gain: f32| {
        base.iter()
            .zip(response)
            .zip(target)
            .map(|((&base, &response), &target)| {
                let rendered = capture::linear_to_srgb(base + gain * response);
                let difference = rendered - capture::linear_to_srgb(target);
                f64::from(difference * difference)
            })
            .sum::<f64>()
    };
    let ratio = 0.5 * (5.0f32.sqrt() - 1.0);
    let mut low = 0.0f32;
    let mut high = maximum;
    let mut left = high - ratio * (high - low);
    let mut right = low + ratio * (high - low);
    let mut left_error = error(left);
    let mut right_error = error(right);
    for _ in 0..48 {
        if left_error <= right_error {
            high = right;
            right = left;
            right_error = left_error;
            left = high - ratio * (high - low);
            left_error = error(left);
        } else {
            low = left;
            left = right;
            left_error = right_error;
            right = low + ratio * (high - low);
            right_error = error(right);
        }
    }
    0.5 * (low + high)
}

fn conservative_global_albedo_gain(fitted: f32) -> f32 {
    0.5 * (1.0 + fitted)
}

#[allow(clippy::too_many_arguments)]
fn refine_rendered_global_albedo(
    scene: &mut score::Scene,
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
    loss: f64,
) -> (f64, usize, usize, Option<f32>) {
    let original = scene.model.materials.clone();
    tracer.reset_sampling();
    let current = renderer.render_prepared_flat(tracer, cameras).to_vec();
    for material in &mut scene.model.materials {
        material.albedo = [0.0; 3];
    }
    renderer.update_prepared_materials(tracer, &scene.model.materials);
    tracer.reset_sampling();
    let zero = renderer.render_prepared_flat(tracer, cameras).to_vec();

    let frame_pixels = capture.width * capture.height;
    let mut base = Vec::new();
    let mut response = Vec::new();
    let mut target = Vec::new();
    for ((current, zero), &index) in current
        .chunks(frame_pixels)
        .zip(zero.chunks(frame_pixels))
        .zip(indices)
    {
        let view = &capture.views[index];
        for (pixel, (&current, &zero)) in current.iter().zip(zero).enumerate() {
            let foreground = view.mask.as_ref().is_none_or(|mask| mask[pixel] >= 0.5);
            if !foreground || current[3] < 0.5 {
                continue;
            }
            for channel in 0..3 {
                base.push(zero[channel]);
                response.push(current[channel] - zero[channel]);
                target.push(view.pixels[pixel][channel]);
            }
        }
    }
    let maximum_albedo = original
        .iter()
        .flat_map(|material| material.albedo)
        .fold(0.0f32, f32::max);
    let maximum_gain = if maximum_albedo > 0.0 {
        (1.0 / maximum_albedo).min(2.0)
    } else {
        1.0
    };
    let fitted_gain = fit_global_albedo_gain(&base, &response, &target, maximum_gain);
    // A one-dimensional training optimum can still over-darken one unseen
    // camera. Apply only half of the correction toward that optimum; the exact
    // complete-render loss below retains it only when the construction images
    // improve.
    let gain = conservative_global_albedo_gain(fitted_gain);
    scene.model.materials.clone_from(&original);
    for material in &mut scene.model.materials {
        for channel in &mut material.albedo {
            *channel *= gain;
        }
    }
    renderer.update_prepared_materials(tracer, &scene.model.materials);
    let candidate_loss = rendered_loss(renderer, tracer, capture, indices, cameras);
    if candidate_loss < loss {
        (candidate_loss, original.len() * 3, 1, Some(gain))
    } else {
        scene.model.materials = original;
        renderer.update_prepared_materials(tracer, &scene.model.materials);
        (loss, scene.model.materials.len() * 3, 1, None)
    }
}

const CONSERVATIVE_SPECULAR_ALLOCATION: f32 = 0.25;

fn allocate_global_specular(materials: &mut [vol::relight::Material], amount: f32) {
    for material in materials {
        for channel in 0..3 {
            let albedo = material.albedo[channel];
            material.albedo[channel] = albedo * (1.0 - amount);
            material.specular_f0[channel] =
                material.specular_f0[channel] * (1.0 - amount) + albedo * amount;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_rendered_global_specular_allocation(
    scene: &mut score::Scene,
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
    loss: f64,
) -> (f64, usize, Option<f32>) {
    let is_rough_dielectric = scene.model.materials.iter().all(|material| {
        (material.roughness - 1.0).abs() <= 1.0e-6
            && material
                .specular_f0
                .iter()
                .all(|&value| (value - 0.04).abs() <= 1.0e-6)
    });
    if !is_rough_dielectric {
        return (loss, 0, None);
    }

    let original = scene.model.materials.clone();
    allocate_global_specular(&mut scene.model.materials, CONSERVATIVE_SPECULAR_ALLOCATION);
    renderer.update_prepared_materials(tracer, &scene.model.materials);
    let candidate_loss = rendered_loss(renderer, tracer, capture, indices, cameras);
    if candidate_loss < loss {
        (candidate_loss, 1, Some(CONSERVATIVE_SPECULAR_ALLOCATION))
    } else {
        scene.model.materials = original;
        renderer.update_prepared_materials(tracer, &scene.model.materials);
        (loss, 1, None)
    }
}

fn solve_dense(matrix: &mut [f64], right: &mut [f64]) -> bool {
    let size = right.len();
    debug_assert_eq!(matrix.len(), size * size);
    for column in 0..size {
        let Some(pivot) = (column..size).max_by(|&left, &right_row| {
            matrix[left * size + column]
                .abs()
                .total_cmp(&matrix[right_row * size + column].abs())
        }) else {
            return false;
        };
        if !matrix[pivot * size + column].is_finite()
            || matrix[pivot * size + column].abs() < 1.0e-12
        {
            return false;
        }
        if pivot != column {
            for entry in 0..size {
                matrix.swap(column * size + entry, pivot * size + entry);
            }
            right.swap(column, pivot);
        }
        let diagonal = matrix[column * size + column];
        for entry in column..size {
            matrix[column * size + entry] /= diagonal;
        }
        right[column] /= diagonal;
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row * size + column];
            if factor == 0.0 {
                continue;
            }
            for entry in column..size {
                matrix[row * size + entry] -= factor * matrix[column * size + entry];
            }
            right[row] -= factor * right[column];
        }
    }
    right.iter().all(|value| value.is_finite())
}

#[allow(clippy::too_many_arguments)]
/// Solve shared albedos from a linear-response surrogate of the production
/// renderer. Geometry, material assignments, light, and every non-albedo
/// parameter stay fixed. The ridge keeps correlated palette entries near their
/// observation-based initializer; an exact production render decides
/// acceptance because sampled one-bounce transport is not strictly affine.
fn fit_rendered_linear_albedo(
    scene: &mut score::Scene,
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
    basis: &RenderedAlbedoBasis,
    current_loss: f64,
) -> Option<f64> {
    const RIDGE_RATIO: f64 = 1.0e-4;

    let original = scene.model.materials.clone();
    let coordinates = original.len() * 3;
    let original_albedo: Vec<f64> = original
        .iter()
        .flat_map(|material| material.albedo)
        .map(f64::from)
        .collect();
    let residual: Vec<f32> = basis
        .target
        .iter()
        .zip(&basis.base)
        .map(|(target, base)| target - base)
        .collect();
    let mut matrix = vec![0.0f64; coordinates * coordinates];
    let mut right = vec![0.0f64; coordinates];
    for row in 0..coordinates {
        let row_material = row / 3;
        let row_channel = row % 3;
        right[row] = (row_channel..residual.len())
            .step_by(3)
            .map(|index| {
                f64::from(basis.responses[row_material][index]) * f64::from(residual[index])
            })
            .sum();
        for column in 0..=row {
            let column_material = column / 3;
            let column_channel = column % 3;
            let product = if row_channel == column_channel {
                (row_channel..residual.len())
                    .step_by(3)
                    .map(|index| {
                        f64::from(basis.responses[row_material][index])
                            * f64::from(basis.responses[column_material][index])
                    })
                    .sum()
            } else {
                0.0
            };
            matrix[row * coordinates + column] = product;
            matrix[column * coordinates + row] = product;
        }
    }
    let trace = (0..coordinates)
        .map(|index| matrix[index * coordinates + index])
        .sum::<f64>();
    let ridge = (RIDGE_RATIO * trace / coordinates as f64).max(1.0e-9);
    for coordinate in 0..coordinates {
        matrix[coordinate * coordinates + coordinate] += ridge;
        right[coordinate] += ridge * original_albedo[coordinate];
    }
    if !solve_dense(&mut matrix, &mut right) {
        scene.model.materials = original;
        renderer.update_prepared_materials(tracer, &scene.model.materials);
        return None;
    }
    for (coordinate, value) in right.into_iter().enumerate() {
        scene.model.materials[coordinate / 3].albedo[coordinate % 3] = value.clamp(0.0, 1.0) as f32;
    }
    renderer.update_prepared_materials(tracer, &scene.model.materials);
    let candidate_loss = rendered_loss(renderer, tracer, capture, indices, cameras);
    if candidate_loss < current_loss {
        Some(candidate_loss)
    } else {
        scene.model.materials = original;
        renderer.update_prepared_materials(tracer, &scene.model.materials);
        None
    }
}

fn perturbation_sign(index: usize, round: usize) -> f32 {
    let value = perturbation_hash(index, round);
    if value & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

const SIMULTANEOUS_STEP_FACTOR: f32 = 0.025;
const SIMULTANEOUS_OFFSET_PRIOR: f64 = 0.001;

fn apply_offsets(
    scene: &mut score::Scene,
    candidates: &[usize],
    anchors: &[[f32; 3]],
    offsets: &[f32],
) {
    for ((&index, &anchor), &offset) in candidates.iter().zip(anchors).zip(offsets) {
        let surfel = &mut scene.model.surfels[index];
        let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
        surfel.center = (glam::Vec3::from(anchor) + offset * surfel.radius * normal).to_array();
    }
}

fn offset_prior(offsets: &[f32]) -> f64 {
    offsets
        .iter()
        .map(|value| (*value * *value) as f64)
        .sum::<f64>()
        / offsets.len().max(1) as f64
}

fn mean_error(errors: &[f32]) -> f64 {
    errors.iter().map(|&value| value as f64).sum::<f64>() / errors.len().max(1) as f64
}

struct ErrorDifference {
    width: usize,
    height: usize,
    sums: Vec<f64>,
}

impl ErrorDifference {
    fn new(plus: &[f32], minus: &[f32], width: usize, height: usize) -> Self {
        assert_eq!(plus.len(), minus.len());
        let frame_pixels = width * height;
        assert_eq!(plus.len() % frame_pixels, 0);
        let stride = width + 1;
        let frame_sums = stride * (height + 1);
        let mut sums = vec![0.0f64; plus.len() / frame_pixels * frame_sums];
        for (frame, (plus, minus)) in plus
            .chunks(frame_pixels)
            .zip(minus.chunks(frame_pixels))
            .enumerate()
        {
            let base = frame * frame_sums;
            for y in 0..height {
                let mut row = 0.0f64;
                for x in 0..width {
                    let pixel = y * width + x;
                    row += (plus[pixel] - minus[pixel]) as f64;
                    sums[base + (y + 1) * stride + x + 1] = sums[base + y * stride + x + 1] + row;
                }
            }
        }
        Self {
            width,
            height,
            sums,
        }
    }

    fn sum(&self, frame: usize, min_x: usize, max_x: usize, min_y: usize, max_y: usize) -> f64 {
        let stride = self.width + 1;
        let base = frame * stride * (self.height + 1);
        let at = |x: usize, y: usize| self.sums[base + y * stride + x];
        at(max_x, max_y) + at(min_x, min_y) - at(min_x, max_y) - at(max_x, min_y)
    }
}

fn projected_error_difference(
    errors: &ErrorDifference,
    capture: &capture::Capture,
    cameras: &[vol::CameraParams],
    surfel: &vol::relight::Surfel,
) -> Option<f32> {
    debug_assert_eq!(errors.width, capture.width);
    debug_assert_eq!(errors.height, capture.height);
    let center = glam::Vec3::from(surfel.center);
    let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
    let tangent = tangent_to(normal);
    let bitangent = normal.cross(tangent);
    let mut difference = 0.0f64;
    let mut samples = 0usize;
    for (frame, camera) in cameras.iter().enumerate() {
        let Some((pixel, _)) = capture::project(camera, capture.width, capture.height, center)
        else {
            continue;
        };
        let Some((pixel_tangent, _)) = capture::project(
            camera,
            capture.width,
            capture.height,
            center + surfel.radius * tangent,
        ) else {
            continue;
        };
        let Some((pixel_bitangent, _)) = capture::project(
            camera,
            capture.width,
            capture.height,
            center + surfel.radius * bitangent,
        ) else {
            continue;
        };
        let extent_x = (pixel_tangent[0] - pixel[0])
            .abs()
            .max((pixel_bitangent[0] - pixel[0]).abs())
            .max(0.5);
        let extent_y = (pixel_tangent[1] - pixel[1])
            .abs()
            .max((pixel_bitangent[1] - pixel[1]).abs())
            .max(0.5);
        let min_x = (pixel[0] - extent_x).floor().max(0.0) as usize;
        let max_x = (pixel[0] + extent_x).ceil().min(capture.width as f32) as usize;
        let min_y = (pixel[1] - extent_y).floor().max(0.0) as usize;
        let max_y = (pixel[1] + extent_y).ceil().min(capture.height as f32) as usize;
        if min_x >= max_x || min_y >= max_y {
            continue;
        }
        difference += errors.sum(frame, min_x, max_x, min_y, max_y);
        samples += (max_x - min_x) * (max_y - min_y);
    }
    (samples > 0).then_some((difference / samples as f64) as f32)
}

fn refine_simultaneous(
    scene: &mut score::Scene,
    renderer: &mut score::Renderer,
    tracer: &mut vol::gpu::RelightTracer,
    capture: &capture::Capture,
    indices: &[usize],
    cameras: &[vol::CameraParams],
    candidates: &[usize],
    rounds: usize,
    mut loss: f64,
) -> (f64, usize) {
    if candidates.is_empty() || rounds == 0 {
        return (loss, 0);
    }
    let anchors: Vec<[f32; 3]> = candidates
        .iter()
        .map(|&index| scene.model.surfels[index].center)
        .collect();
    let mut offsets = vec![0.0f32; candidates.len()];
    let mut objective = loss;
    let mut accepted = 0usize;
    for round in 0..rounds {
        let plus_offsets: Vec<f32> = offsets
            .iter()
            .zip(candidates)
            .map(|(&offset, &index)| {
                offset + perturbation_sign(index, round) * SIMULTANEOUS_STEP_FACTOR
            })
            .collect();
        apply_offsets(scene, candidates, &anchors, &plus_offsets);
        renderer.update_prepared_surfels(&scene.model.surfels);
        let plus_errors = rendered_errors(renderer, tracer, capture, indices, cameras);
        let plus = mean_error(&plus_errors);
        let plus_objective = plus + SIMULTANEOUS_OFFSET_PRIOR * offset_prior(&plus_offsets);

        let minus_offsets: Vec<f32> = offsets
            .iter()
            .zip(candidates)
            .map(|(&offset, &index)| {
                offset - perturbation_sign(index, round) * SIMULTANEOUS_STEP_FACTOR
            })
            .collect();
        apply_offsets(scene, candidates, &anchors, &minus_offsets);
        renderer.update_prepared_surfels(&scene.model.surfels);
        let minus_errors = rendered_errors(renderer, tracer, capture, indices, cameras);
        let minus = mean_error(&minus_errors);
        let minus_objective = minus + SIMULTANEOUS_OFFSET_PRIOR * offset_prior(&minus_offsets);
        let error_difference =
            ErrorDifference::new(&plus_errors, &minus_errors, capture.width, capture.height);

        // A whole-frame SPSA difference gives every particle the same scalar
        // direction, although one particle affects only a small screen patch.
        // Correlate each antithetic sign with the error difference inside its
        // projected Gaussian footprint instead. The proposal is still scored
        // by complete production renders, with the two original whole-frame
        // perturbations retained as fallbacks, so this changes neither the
        // renderer nor the acceptance objective.
        let localized_offsets: Vec<f32> = offsets
            .iter()
            .zip(candidates)
            .enumerate()
            .map(|(candidate, (&offset, &index))| {
                let mut surfel = scene.model.surfels[index];
                let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
                surfel.center = (glam::Vec3::from(anchors[candidate])
                    + offset * surfel.radius * normal)
                    .to_array();
                projected_error_difference(&error_difference, capture, cameras, &surfel)
                    .filter(|gradient| *gradient != 0.0)
                    .map_or(offset, |gradient| {
                        offset
                            - gradient.signum()
                                * perturbation_sign(index, round)
                                * SIMULTANEOUS_STEP_FACTOR
                    })
            })
            .collect();
        apply_offsets(scene, candidates, &anchors, &localized_offsets);
        renderer.update_prepared_surfels(&scene.model.surfels);
        let localized = rendered_loss(renderer, tracer, capture, indices, cameras);
        let localized_objective =
            localized + SIMULTANEOUS_OFFSET_PRIOR * offset_prior(&localized_offsets);

        if localized_objective < objective
            && localized_objective <= plus_objective
            && localized_objective <= minus_objective
        {
            offsets = localized_offsets;
            loss = localized;
            objective = localized_objective;
            accepted += 1;
        } else if plus_objective < objective && plus_objective <= minus_objective {
            offsets = plus_offsets;
            apply_offsets(scene, candidates, &anchors, &offsets);
            renderer.update_prepared_surfels(&scene.model.surfels);
            loss = plus;
            objective = plus_objective;
            accepted += 1;
        } else if minus_objective < objective {
            offsets = minus_offsets;
            apply_offsets(scene, candidates, &anchors, &offsets);
            renderer.update_prepared_surfels(&scene.model.surfels);
            loss = minus;
            objective = minus_objective;
            accepted += 1;
        } else {
            apply_offsets(scene, candidates, &anchors, &offsets);
            renderer.update_prepared_surfels(&scene.model.surfels);
        }
    }
    (loss, accepted)
}

/// Move particles along their current normals against the complete runtime
/// PBR render of every training photograph.
///
/// Simultaneous rounds test one deterministic paired perturbation of every
/// observed particle, derive a localized direction from the per-pixel
/// antithetic error inside each projected Gaussian footprint, and keep the
/// best of that proposal and the original two whole-frame directions. Every
/// proposal refreshes the production TLAS and all training renders. A small
/// radius-normalized anchor prior limits drift. The exact pass then tests up
/// to `max_particles` independently in both directions.
/// Materials and illumination stay fixed throughout. The held-out cameras
/// never enter this function; callers decide which indices are training
/// evidence.
pub fn refine_rendered(
    scene: &mut score::Scene,
    capture: &capture::Capture,
    indices: &[usize],
    observations: &decompose::Observations,
    diffuse_samples: u32,
    simultaneous_rounds: usize,
    refine_radii: bool,
    max_particles: usize,
) -> Result<RenderedStats, String> {
    if observations.surfels() != scene.model.surfels.len() {
        return Err("rendered refinement observation count does not match the model".to_string());
    }
    for &index in indices {
        if index >= capture.views.len() {
            return Err(format!("rendered refinement view {index} is out of bounds"));
        }
    }
    if scene.model.surfels.is_empty()
        || indices.is_empty()
        || (simultaneous_rounds == 0 && max_particles == 0)
    {
        return Ok(RenderedStats {
            particles: scene.model.surfels.len(),
            ..RenderedStats::default()
        });
    }

    let cameras: Vec<vol::CameraParams> = indices
        .iter()
        .map(|&index| capture.views[index].camera)
        .collect();
    let mut renderer = score::Renderer::new(capture.width, capture.height)?;
    let mut tracer = renderer.prepare_scene(scene, diffuse_samples, false);
    let started = std::time::Instant::now();
    let mut loss = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
    let initial_loss = loss;
    let radius_fraction = refine_radii.then_some(0.2);
    let observed: Vec<usize> = (0..scene.model.surfels.len())
        .filter(|&index| !observations.of(index).is_empty())
        .collect();
    let (next_loss, simultaneous_accepted) = refine_simultaneous(
        scene,
        &mut renderer,
        &mut tracer,
        capture,
        indices,
        &cameras,
        &observed,
        simultaneous_rounds,
        loss,
    );
    loss = next_loss;
    let simultaneous_particles = if simultaneous_rounds > 0 {
        observed.len()
    } else {
        0
    };
    let candidates: Vec<usize> = observed.into_iter().take(max_particles).collect();
    let tested = candidates.len();
    let mut moved = 0usize;
    let mut radii_moved = 0usize;
    for index in candidates {
        let original = scene.model.surfels[index].center;
        let normal = glam::Vec3::from(scene.model.surfels[index].normal).normalize_or_zero();
        let step = 0.25 * scene.model.surfels[index].radius;
        let plus = (glam::Vec3::from(original) + step * normal).to_array();
        let mut best_loss = loss;
        let mut best = original;
        for sign in [-1.0f32, 1.0] {
            scene.model.surfels[index].center =
                (glam::Vec3::from(original) + sign * step * normal).to_array();
            renderer.update_prepared_surfels(&scene.model.surfels);
            let candidate = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
            if candidate < best_loss {
                best_loss = candidate;
                best = scene.model.surfels[index].center;
                // This coordinate pass is a bounded final polish. Once the
                // first direction improves its full rendered objective, keep
                // it instead of paying another TLAS rebuild for a marginally
                // better step on the same coordinate.
                if sign < 0.0 {
                    break;
                }
            }
        }
        scene.model.surfels[index].center = best;
        if best != plus {
            renderer.update_prepared_surfels(&scene.model.surfels);
        }
        if best != original {
            moved += 1;
            loss = best_loss;
        }
        if let Some(fraction) = radius_fraction {
            let original = scene.model.surfels[index].radius;
            let plus = original * (1.0 + fraction);
            let mut best_loss = loss;
            let mut best = original;
            for scale in [1.0 + fraction, 1.0 - fraction] {
                scene.model.surfels[index].radius = original * scale;
                renderer.update_prepared_surfels(&scene.model.surfels);
                let candidate =
                    rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
                if candidate < best_loss {
                    best_loss = candidate;
                    best = scene.model.surfels[index].radius;
                    if scale > 1.0 {
                        break;
                    }
                }
            }
            scene.model.surfels[index].radius = best;
            if best != plus {
                renderer.update_prepared_surfels(&scene.model.surfels);
            }
            if best != original {
                radii_moved += 1;
                loss = best_loss;
            }
        }
    }
    renderer.destroy_prepared_scene(tracer);
    renderer.destroy();
    Ok(RenderedStats {
        particles: scene.model.surfels.len(),
        simultaneous_particles,
        simultaneous_rounds,
        simultaneous_accepted,
        tested,
        moved,
        radii_moved,
        initial_loss,
        final_loss: loss,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct Choice {
    offset: f32,
    improvement: f32,
    views: usize,
    scored: bool,
}

struct SelectedView<'a> {
    view: &'a capture::View,
    depth: Option<&'a depth::DepthMap>,
    facing: f32,
}

/// Refine surfel centers against all supplied photographs.
///
/// Only centers move; normals, radii and material indices remain attached to
/// their particles. Every surfel is independent during this pass, which makes
/// the result deterministic and permits a bounded worker pool.
pub fn refine(
    surfels: &mut [vol::relight::Surfel],
    capture: &capture::Capture,
    views: &[RefinementView<'_>],
    options: RefineOptions,
) -> RefineStats {
    validate_options(options);
    if surfels.is_empty() || views.is_empty() {
        return RefineStats {
            surfels: surfels.len(),
            ..RefineStats::default()
        };
    }
    for view in views {
        assert!(
            view.capture_index < capture.views.len(),
            "refinement view index is out of bounds"
        );
        if let Some(map) = view.depth {
            assert_eq!(map.width, capture.width, "depth/capture width mismatch");
            assert_eq!(map.height, capture.height, "depth/capture height mismatch");
        }
    }

    let mut choices = vec![Choice::default(); surfels.len()];
    let workers = thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(surfels.len());
    let chunk_size = surfels.len().div_ceil(workers);
    thread::scope(|scope| {
        for (block, choice_chunk) in choices.chunks_mut(chunk_size).enumerate() {
            let begin = block * chunk_size;
            let surfel_chunk = &surfels[begin..begin + choice_chunk.len()];
            scope.spawn(move || {
                for (choice, surfel) in choice_chunk.iter_mut().zip(surfel_chunk) {
                    *choice = choose_offset(surfel, capture, views, options);
                }
            });
        }
    });
    regularize_choices(surfels, &mut choices);

    let mut stats = RefineStats {
        surfels: surfels.len(),
        ..RefineStats::default()
    };
    let mut absolute_offset = 0.0f32;
    let mut improvement = 0.0f32;
    let mut view_count = 0usize;
    for (surfel, choice) in surfels.iter_mut().zip(&choices) {
        if !choice.scored {
            continue;
        }
        stats.scored += 1;
        improvement += choice.improvement;
        view_count += choice.views;
        if choice.offset == 0.0 {
            continue;
        }
        let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
        surfel.center = (glam::Vec3::from(surfel.center) + choice.offset * normal).to_array();
        stats.moved += 1;
        absolute_offset += choice.offset.abs();
    }
    if stats.moved > 0 {
        stats.mean_absolute_offset = absolute_offset / stats.moved as f32;
    }
    if stats.scored > 0 {
        stats.mean_relative_improvement = improvement / stats.scored as f32;
        stats.mean_views = view_count as f32 / stats.scored as f32;
    }
    stats
}

fn regularize_choices(surfels: &[vol::relight::Surfel], choices: &mut [Choice]) {
    if surfels.len() < 4 {
        return;
    }
    let positions: Vec<[f32; 3]> = surfels.iter().map(|surfel| surfel.center).collect();
    let tree = kiddo::ImmutableKdTree::new_from_slice(&positions);
    let source = choices.to_vec();
    let count = std::num::NonZero::new(9.min(surfels.len())).unwrap();
    for (index, choice) in choices.iter_mut().enumerate() {
        if choice.offset == 0.0 {
            continue;
        }
        let surfel = &surfels[index];
        let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
        let distance_limit = 2.0 * surfel.radius;
        let mut offsets = Vec::new();
        for hit in tree.nearest_n::<kiddo::SquaredEuclidean>(&surfel.center, count) {
            let neighbor = hit.item as usize;
            let neighbor_normal = glam::Vec3::from(surfels[neighbor].normal).normalize_or_zero();
            if source[neighbor].offset == 0.0
                || hit.distance > distance_limit * distance_limit
                || normal.dot(neighbor_normal) < 0.8
            {
                continue;
            }
            offsets.push(source[neighbor].offset);
        }
        if offsets.len() < 4 {
            continue;
        }
        offsets.sort_by(f32::total_cmp);
        let median = offsets[offsets.len() / 2];
        choice.offset = if choice.offset.signum() == median.signum() {
            median
        } else {
            0.0
        };
    }
}

fn validate_options(options: RefineOptions) {
    assert!(
        options.search_radius_factor.is_finite() && options.search_radius_factor > 0.0,
        "search radius factor must be finite and positive"
    );
    assert!(
        options.candidates >= 3 && options.candidates % 2 == 1,
        "candidate count must be odd and at least three"
    );
    assert!(
        options.patch_side >= 3 && options.patch_side % 2 == 1,
        "patch side must be odd and at least three"
    );
    assert!(
        options.patch_radius_factor.is_finite() && options.patch_radius_factor > 0.0,
        "patch radius factor must be finite and positive"
    );
    assert!(
        options.min_views >= 2,
        "refinement needs at least two views"
    );
    assert!(
        options.max_views >= options.min_views,
        "max views must be at least min views"
    );
    assert!(
        options.min_facing.is_finite() && (0.0..=1.0).contains(&options.min_facing),
        "minimum facing must be in [0, 1]"
    );
    assert!(
        options.min_texture.is_finite() && options.min_texture >= 0.0,
        "minimum texture must be finite and non-negative"
    );
    assert!(
        options.min_improvement.is_finite() && options.min_improvement >= 0.0,
        "minimum improvement must be finite and non-negative"
    );
    assert!(
        options.offset_prior.is_finite() && options.offset_prior >= 0.0,
        "offset prior must be finite and non-negative"
    );
    assert!(
        options.visibility_radius_factor.is_finite() && options.visibility_radius_factor >= 0.0,
        "visibility radius factor must be finite and non-negative"
    );
}

fn choose_offset(
    surfel: &vol::relight::Surfel,
    capture: &capture::Capture,
    views: &[RefinementView<'_>],
    options: RefineOptions,
) -> Choice {
    if !surfel.radius.is_finite() || surfel.radius <= 0.0 {
        return Choice::default();
    }
    let center = glam::Vec3::from(surfel.center);
    let Some(normal) = glam::Vec3::from(surfel.normal).try_normalize() else {
        return Choice::default();
    };
    let tangent = tangent_to(normal);
    let bitangent = normal.cross(tangent);
    let search_radius = surfel.radius * options.search_radius_factor;
    let patch_radius = surfel.radius * options.patch_radius_factor;
    let visibility_tolerance = options.visibility_radius_factor * surfel.radius + search_radius;

    let mut selected = Vec::new();
    for input in views {
        let view = &capture.views[input.capture_index];
        let towards = (glam::Vec3::from(view.camera.cam_position) - center).normalize_or_zero();
        let facing = normal.dot(towards);
        if facing < options.min_facing
            || !patch_fits(
                center - search_radius * normal,
                tangent,
                bitangent,
                patch_radius,
                view,
                capture.width,
                capture.height,
                options.patch_side,
            )
            || !patch_fits(
                center + search_radius * normal,
                tangent,
                bitangent,
                patch_radius,
                view,
                capture.width,
                capture.height,
                options.patch_side,
            )
            || !source_depth_visible(surfel, view, input.depth, capture, search_radius, options)
        {
            continue;
        }
        selected.push(SelectedView {
            view,
            depth: input.depth,
            facing,
        });
    }
    selected.sort_by(|left, right| right.facing.total_cmp(&left.facing));
    selected.truncate(options.max_views);
    if selected.len() < options.min_views {
        return Choice::default();
    }

    let middle = options.candidates / 2;
    let Some((baseline, baseline_views)) = candidate_cost(
        center,
        tangent,
        bitangent,
        patch_radius,
        visibility_tolerance,
        &selected,
        capture,
        options,
    ) else {
        return Choice::default();
    };
    let mut best_cost = baseline;
    let mut best_offset = 0.0f32;
    let mut best_views = baseline_views;
    for candidate in 0..options.candidates {
        if candidate == middle {
            continue;
        }
        let phase = candidate as f32 / (options.candidates - 1) as f32;
        let offset = search_radius * (2.0 * phase - 1.0);
        let Some((raw_cost, used_views)) = candidate_cost(
            center + offset * normal,
            tangent,
            bitangent,
            patch_radius,
            visibility_tolerance,
            &selected,
            capture,
            options,
        ) else {
            continue;
        };
        let normalized_offset = offset / search_radius;
        let cost = raw_cost + options.offset_prior * normalized_offset * normalized_offset;
        if cost.total_cmp(&best_cost).is_lt()
            || (cost == best_cost && offset.abs() < best_offset.abs())
        {
            best_cost = cost;
            best_offset = offset;
            best_views = used_views;
        }
    }

    let improvement = ((baseline - best_cost) / baseline.max(1.0e-6)).max(0.0);
    if improvement < options.min_improvement {
        best_offset = 0.0;
        best_views = baseline_views;
    }
    Choice {
        offset: best_offset,
        improvement,
        views: best_views,
        scored: true,
    }
}

fn tangent_to(normal: glam::Vec3) -> glam::Vec3 {
    let axis = if normal.z.abs() < 0.9 {
        glam::Vec3::Z
    } else {
        glam::Vec3::X
    };
    normal.cross(axis).normalize()
}

#[allow(clippy::too_many_arguments)]
fn patch_fits(
    center: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    radius: f32,
    view: &capture::View,
    width: usize,
    height: usize,
    side: usize,
) -> bool {
    patch_points(center, tangent, bitangent, radius, side).all(|point| {
        capture::project(&view.camera, width, height, point)
            .is_some_and(|(pixel, _)| sample_coordinates(pixel, width, height).is_some())
    })
}

fn patch_points(
    center: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    radius: f32,
    side: usize,
) -> impl Iterator<Item = glam::Vec3> {
    let half = 0.5 * (side - 1) as f32;
    (0..side * side).map(move |index| {
        let x = index % side;
        let y = index / side;
        let u = (x as f32 - half) / half;
        let v = (y as f32 - half) / half;
        center + radius * (u * tangent + v * bitangent)
    })
}

fn candidate_cost(
    center: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    radius: f32,
    visibility_tolerance: f32,
    views: &[SelectedView<'_>],
    capture: &capture::Capture,
    options: RefineOptions,
) -> Option<(f32, usize)> {
    let mut patches = Vec::with_capacity(views.len());
    for selected in views {
        if !candidate_depth_visible(center, visibility_tolerance, selected, capture, options) {
            continue;
        }
        if let Some(patch) = normalized_patch(
            center,
            tangent,
            bitangent,
            radius,
            selected.view,
            capture.width,
            capture.height,
            options.patch_side,
            options.min_texture,
        ) {
            patches.push(patch);
        }
    }
    if patches.len() < options.min_views {
        return None;
    }
    let mut pair_costs = Vec::with_capacity(patches.len() * (patches.len() - 1) / 2);
    for left in 0..patches.len() {
        for right in left + 1..patches.len() {
            let cost = patches[left]
                .iter()
                .zip(&patches[right])
                .map(|(a, b)| {
                    let difference = a - b;
                    difference * difference
                })
                .sum::<f32>()
                / patches[left].len() as f32;
            pair_costs.push(cost);
        }
    }
    pair_costs.sort_by(f32::total_cmp);
    Some((pair_costs[pair_costs.len() / 2], patches.len()))
}

fn candidate_depth_visible(
    center: glam::Vec3,
    tolerance: f32,
    selected: &SelectedView<'_>,
    capture: &capture::Capture,
    options: RefineOptions,
) -> bool {
    let Some(map) = selected.depth else {
        return true;
    };
    let Some((pixel, _)) =
        capture::project(&selected.view.camera, capture.width, capture.height, center)
    else {
        return false;
    };
    let Some(alpha) = sample_scalar(&map.alpha, map.width, map.height, pixel) else {
        return false;
    };
    let Some(peak) = sample_scalar(&map.peak, map.width, map.height, pixel) else {
        return false;
    };
    let Some(source_distance) = sample_scalar(&map.distance, map.width, map.height, pixel) else {
        return false;
    };
    if alpha < options.visibility_min_alpha || peak < options.visibility_min_peak {
        return false;
    }
    let candidate_distance = center.distance(glam::Vec3::from(selected.view.camera.cam_position));
    (candidate_distance - source_distance).abs() <= tolerance
}

#[allow(clippy::too_many_arguments)]
fn normalized_patch(
    center: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    radius: f32,
    view: &capture::View,
    width: usize,
    height: usize,
    side: usize,
    min_texture: f32,
) -> Option<Vec<f32>> {
    let mut pixels = Vec::with_capacity(side * side);
    for point in patch_points(center, tangent, bitangent, radius, side) {
        let (pixel, _) = capture::project(&view.camera, width, height, point)?;
        if let Some(ref mask) = view.mask {
            if sample_scalar(mask, width, height, pixel)? <= 0.5 {
                return None;
            }
        }
        pixels.push(sample_rgb(&view.pixels, width, height, pixel)?);
    }
    let mut mean = [0.0f32; 3];
    for pixel in &pixels {
        for channel in 0..3 {
            mean[channel] += pixel[channel];
        }
    }
    for value in &mut mean {
        *value /= pixels.len() as f32;
    }
    let energy = pixels
        .iter()
        .flat_map(|pixel| [pixel[0] - mean[0], pixel[1] - mean[1], pixel[2] - mean[2]])
        .map(|value| value * value)
        .sum::<f32>()
        / (pixels.len() * 3) as f32;
    let rms = energy.sqrt();
    if !rms.is_finite() || rms < min_texture {
        return None;
    }
    Some(
        pixels
            .into_iter()
            .flat_map(|pixel| {
                [
                    (pixel[0] - mean[0]) / rms,
                    (pixel[1] - mean[1]) / rms,
                    (pixel[2] - mean[2]) / rms,
                ]
            })
            .collect(),
    )
}

fn sample_coordinates(pixel: [f32; 2], width: usize, height: usize) -> Option<(f32, f32)> {
    let x = pixel[0] - 0.5;
    let y = pixel[1] - 0.5;
    (x >= 0.0 && y >= 0.0 && x <= (width - 1) as f32 && y <= (height - 1) as f32).then_some((x, y))
}

fn sample_rgb(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    pixel: [f32; 2],
) -> Option<[f32; 3]> {
    let (x, y) = sample_coordinates(pixel, width, height)?;
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let a = pixels[y0 * width + x0];
    let b = pixels[y0 * width + x1];
    let c = pixels[y1 * width + x0];
    let d = pixels[y1 * width + x1];
    Some(std::array::from_fn(|channel| {
        let top = a[channel] + tx * (b[channel] - a[channel]);
        let bottom = c[channel] + tx * (d[channel] - c[channel]);
        top + ty * (bottom - top)
    }))
}

fn sample_scalar(values: &[f32], width: usize, height: usize, pixel: [f32; 2]) -> Option<f32> {
    let (x, y) = sample_coordinates(pixel, width, height)?;
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let top = values[y0 * width + x0] + tx * (values[y0 * width + x1] - values[y0 * width + x0]);
    let bottom = values[y1 * width + x0] + tx * (values[y1 * width + x1] - values[y1 * width + x0]);
    Some(top + ty * (bottom - top))
}

fn source_depth_visible(
    surfel: &vol::relight::Surfel,
    view: &capture::View,
    depth: Option<&depth::DepthMap>,
    capture: &capture::Capture,
    search_radius: f32,
    options: RefineOptions,
) -> bool {
    let Some(map) = depth else {
        return true;
    };
    let center = glam::Vec3::from(surfel.center);
    let Some((pixel, _)) = capture::project(&view.camera, capture.width, capture.height, center)
    else {
        return false;
    };
    let Some(alpha) = sample_scalar(&map.alpha, map.width, map.height, pixel) else {
        return false;
    };
    let Some(peak) = sample_scalar(&map.peak, map.width, map.height, pixel) else {
        return false;
    };
    let Some(source_distance) = sample_scalar(&map.distance, map.width, map.height, pixel) else {
        return false;
    };
    if alpha < options.visibility_min_alpha || peak < options.visibility_min_peak {
        return false;
    }
    let candidate_distance = center.distance(glam::Vec3::from(view.camera.cam_position));
    let tolerance = options.visibility_radius_factor * surfel.radius + search_radius;
    (candidate_distance - source_distance).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_solver_pivots_to_the_known_solution() {
        let mut matrix = [0.0, 2.0, 1.0, 3.0];
        let mut right = [4.0, 7.0];
        assert!(solve_dense(&mut matrix, &mut right));
        assert!((right[0] - 1.0).abs() < 1.0e-12);
        assert!((right[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn global_albedo_gain_recovers_an_affine_target() {
        let base = [0.05, 0.1, 0.2, 0.15, 0.05, 0.1];
        let response = [0.4, 0.3, 0.2, 0.2, 0.5, 0.4];
        let target: Vec<_> = base
            .iter()
            .zip(response)
            .map(|(&base, response)| base + 0.375 * response)
            .collect();
        let gain = fit_global_albedo_gain(&base, &response, &target, 2.0);
        assert!((gain - 0.375).abs() < 1.0e-4, "recovered {gain}");
        assert!((conservative_global_albedo_gain(gain) - 0.6875).abs() < 1.0e-4);
    }

    #[test]
    fn global_specular_allocation_blends_base_color_into_f0() {
        let mut materials = [vol::relight::Material {
            albedo: [0.8, 0.4, 0.2],
            specular_f0: [0.04; 3],
            ..vol::relight::Material::default()
        }];
        allocate_global_specular(&mut materials, 0.25);
        assert_eq!(materials[0].albedo, [0.6, 0.3, 0.15]);
        for (&actual, expected) in materials[0].specular_f0.iter().zip([0.23, 0.13, 0.08]) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn only_a_large_individual_table_transfers_automatically() {
        assert!(automatic_gaussian_material_transfer(33, true));
        assert!(!automatic_gaussian_material_transfer(32, true));
        assert!(!automatic_gaussian_material_transfer(33, false));
    }

    #[test]
    fn encoded_basis_descent_recovers_a_material() {
        let mut scene = score::Scene::new(
            vol::relight::RelightModel {
                kernel: vol::relight::ParticleKernel::Gaussian,
                surfels: Vec::new(),
                materials: vec![vol::relight::Material {
                    albedo: [0.4; 3],
                    ..vol::relight::Material::default()
                }],
            },
            vol::relight::Environment::uniform([1.0; 3], 2, 1),
        );
        let target = vec![0.55; 6];
        let target_srgb = target
            .iter()
            .map(|&value| capture::linear_to_srgb(value))
            .collect();
        let basis = RenderedAlbedoBasis {
            base: vec![0.0; 6],
            responses: vec![vec![1.0; 6]],
            target,
            target_srgb,
        };

        assert_eq!(refine_albedo_from_basis(&mut scene, &basis, 0.1), (6, 12));
        assert_eq!(scene.model.materials[0].albedo, [0.55; 3]);
        assert!(basis.error_sum(&basis.render(&scene.model.materials)) < 1.0e-12);
    }

    #[test]
    fn error_difference_matches_direct_rectangle_sums() {
        const WIDTH: usize = 4;
        const HEIGHT: usize = 3;
        let plus: Vec<f32> = (0..2 * WIDTH * HEIGHT)
            .map(|index| 0.1 * index as f32)
            .collect();
        let minus: Vec<f32> = (0..2 * WIDTH * HEIGHT)
            .map(|index| 0.03 * (index % 5) as f32)
            .collect();
        let field = ErrorDifference::new(&plus, &minus, WIDTH, HEIGHT);
        for frame in 0..2 {
            for min_y in 0..HEIGHT {
                for max_y in min_y + 1..=HEIGHT {
                    for min_x in 0..WIDTH {
                        for max_x in min_x + 1..=WIDTH {
                            let mut expected = 0.0f64;
                            for y in min_y..max_y {
                                for x in min_x..max_x {
                                    let index = frame * WIDTH * HEIGHT + y * WIDTH + x;
                                    expected += (plus[index] - minus[index]) as f64;
                                }
                            }
                            let actual = field.sum(frame, min_x, max_x, min_y, max_y);
                            assert!((actual - expected).abs() < 1.0e-12);
                        }
                    }
                }
            }
        }
    }

    fn camera(position: glam::Vec3) -> vol::CameraParams {
        let orientation = glam::Quat::from_rotation_arc(glam::Vec3::Z, (-position).normalize());
        vol::CameraParams {
            cam_position: position.to_array(),
            depth: 4.0,
            cam_orientation: orientation.to_array(),
            fov: [0.9, 0.9],
            principal: [0.0, 0.0],
        }
    }

    fn texture(point: glam::Vec3) -> [f32; 3] {
        [
            0.5 + 0.3 * (19.0 * point.x + 7.0 * point.y).sin(),
            0.5 + 0.3 * (11.0 * point.x - 17.0 * point.y).sin(),
            0.5 + 0.3 * (23.0 * point.x + 13.0 * point.y).cos(),
        ]
    }

    fn plane_fixture() -> (capture::Capture, Vec<depth::DepthMap>) {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let cameras = [
            camera(glam::Vec3::new(0.0, 0.0, -1.2)),
            camera(glam::Vec3::new(-0.35, 0.0, -1.2)),
            camera(glam::Vec3::new(0.35, 0.0, -1.2)),
            camera(glam::Vec3::new(0.0, -0.3, -1.2)),
            camera(glam::Vec3::new(0.0, 0.3, -1.2)),
        ];
        let mut depths = Vec::with_capacity(cameras.len());
        let views = cameras
            .iter()
            .enumerate()
            .map(|(index, camera)| {
                let origin = glam::Vec3::from(camera.cam_position);
                let mut pixels = Vec::with_capacity(WIDTH * HEIGHT);
                let mut distance = Vec::with_capacity(WIDTH * HEIGHT);
                for y in 0..HEIGHT {
                    for x in 0..WIDTH {
                        let direction = capture::pixel_direction(camera, WIDTH, HEIGHT, x, y);
                        let along = -origin.z / direction.z;
                        pixels.push(texture(origin + along * direction));
                        distance.push(along);
                    }
                }
                depths.push(depth::DepthMap {
                    width: WIDTH,
                    height: HEIGHT,
                    distance,
                    alpha: vec![1.0; WIDTH * HEIGHT],
                    peak: vec![1.0; WIDTH * HEIGHT],
                });
                capture::View {
                    name: format!("synthetic-{index}"),
                    camera: *camera,
                    pixels,
                    mask: None,
                }
            })
            .collect();
        (
            capture::Capture {
                width: WIDTH,
                height: HEIGHT,
                views,
            },
            depths,
        )
    }

    #[test]
    fn normalized_patches_do_not_cross_the_foreground_mask() {
        let (mut capture, _) = plane_fixture();
        let view = &capture.views[0];
        assert!(normalized_patch(
            glam::Vec3::ZERO,
            glam::Vec3::X,
            glam::Vec3::Y,
            0.1,
            view,
            capture.width,
            capture.height,
            3,
            1.0e-4,
        )
        .is_some());

        capture.views[0].mask = Some(vec![0.0; capture.width * capture.height]);
        assert!(normalized_patch(
            glam::Vec3::ZERO,
            glam::Vec3::X,
            glam::Vec3::Y,
            0.1,
            &capture.views[0],
            capture.width,
            capture.height,
            3,
            1.0e-4,
        )
        .is_none());
    }

    fn plane_surfels(offset: f32) -> Vec<vol::relight::Surfel> {
        let mut surfels = Vec::new();
        for y in 0..7 {
            for x in 0..7 {
                surfels.push(vol::relight::Surfel {
                    center: [(x as f32 - 3.0) * 0.08, (y as f32 - 3.0) * 0.08, offset],
                    radius: 0.12,
                    normal: [0.0, 0.0, -1.0],
                    material: surfels.len() as u32,
                });
            }
        }
        surfels
    }

    fn sphere_fixture() -> (capture::Capture, Vec<depth::DepthMap>) {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        const RADIUS: f32 = 0.5;
        let cameras = [
            camera(glam::Vec3::new(0.0, 0.0, -1.5)),
            camera(glam::Vec3::new(-0.35, 0.0, -1.5)),
            camera(glam::Vec3::new(0.35, 0.0, -1.5)),
            camera(glam::Vec3::new(0.0, -0.3, -1.5)),
            camera(glam::Vec3::new(0.0, 0.3, -1.5)),
        ];
        let mut depths = Vec::with_capacity(cameras.len());
        let views = cameras
            .iter()
            .enumerate()
            .map(|(index, camera)| {
                let origin = glam::Vec3::from(camera.cam_position);
                let mut pixels = Vec::with_capacity(WIDTH * HEIGHT);
                let mut distance = Vec::with_capacity(WIDTH * HEIGHT);
                let mut alpha = Vec::with_capacity(WIDTH * HEIGHT);
                for y in 0..HEIGHT {
                    for x in 0..WIDTH {
                        let direction = capture::pixel_direction(camera, WIDTH, HEIGHT, x, y);
                        let middle = -origin.dot(direction);
                        let discriminant =
                            middle * middle - origin.length_squared() + RADIUS * RADIUS;
                        if discriminant < 0.0 {
                            pixels.push([0.0; 3]);
                            distance.push(0.0);
                            alpha.push(0.0);
                            continue;
                        }
                        let along = middle - discriminant.sqrt();
                        let point = origin + along * direction;
                        pixels.push(texture(point));
                        distance.push(along);
                        alpha.push(1.0);
                    }
                }
                let peak = alpha.clone();
                depths.push(depth::DepthMap {
                    width: WIDTH,
                    height: HEIGHT,
                    distance,
                    alpha,
                    peak,
                });
                capture::View {
                    name: format!("sphere-{index}"),
                    camera: *camera,
                    pixels,
                    mask: None,
                }
            })
            .collect();
        (
            capture::Capture {
                width: WIDTH,
                height: HEIGHT,
                views,
            },
            depths,
        )
    }

    fn sphere_surfels(offset: f32) -> Vec<vol::relight::Surfel> {
        const RADIUS: f32 = 0.5;
        let mut surfels = Vec::new();
        for y in 0..7 {
            for x in 0..7 {
                let px = (x as f32 - 3.0) * 0.06;
                let py = (y as f32 - 3.0) * 0.06;
                let pz = -(RADIUS * RADIUS - px * px - py * py).sqrt();
                let truth = glam::Vec3::new(px, py, pz);
                let normal = truth.normalize();
                surfels.push(vol::relight::Surfel {
                    center: (truth + offset * normal).to_array(),
                    radius: 0.1,
                    normal: normal.to_array(),
                    material: surfels.len() as u32,
                });
            }
        }
        surfels
    }

    fn rendered_test_scene(offset: f32) -> score::Scene {
        score::Scene::new(
            vol::relight::RelightModel {
                kernel: vol::relight::ParticleKernel::Gaussian,
                surfels: vec![vol::relight::Surfel {
                    center: [0.0, 0.0, offset],
                    radius: 0.3,
                    normal: [0.0, 0.0, -1.0],
                    material: 0,
                }],
                materials: vec![vol::relight::Material {
                    albedo: [0.7, 0.3, 0.2],
                    roughness: 1.0,
                    specular_f0: [0.04; 3],
                    _padding: 0.0,
                }],
            },
            vol::relight::Environment {
                width: 8,
                height: 4,
                texels: vec![[1.0; 3]; 32],
            },
        )
    }

    #[test]
    fn plane_sweep_recovers_displacement_and_preserves_exact_surface() {
        let (capture, depths) = plane_fixture();
        let views: Vec<RefinementView<'_>> = (0..capture.views.len())
            .map(|capture_index| RefinementView {
                capture_index,
                depth: Some(&depths[capture_index]),
            })
            .collect();
        let options = RefineOptions {
            search_radius_factor: 0.5,
            candidates: 13,
            ..RefineOptions::default()
        };

        let mut displaced = plane_surfels(0.05);
        let displaced_stats = refine(&mut displaced, &capture, &views, options);
        let displaced_error = displaced
            .iter()
            .map(|surfel| surfel.center[2].abs())
            .sum::<f32>()
            / displaced.len() as f32;
        assert!(
            displaced_error < 0.012,
            "known 0.05 displacement remained {displaced_error}; stats={displaced_stats:?}"
        );
        assert!(
            displaced_stats.moved >= displaced.len() * 4 / 5,
            "too few displaced surfels moved: {displaced_stats:?}"
        );

        let mut exact = plane_surfels(0.0);
        let exact_stats = refine(&mut exact, &capture, &views, options);
        let exact_error = exact
            .iter()
            .map(|surfel| surfel.center[2].abs())
            .sum::<f32>()
            / exact.len() as f32;
        assert!(
            exact_error < 1.0e-6,
            "an exact surface moved by {exact_error}; stats={exact_stats:?}"
        );
    }

    #[test]
    fn sphere_sweep_repairs_supported_curvature_without_moving_exact_surface() {
        const RADIUS: f32 = 0.5;
        let (capture, depths) = sphere_fixture();
        let views: Vec<RefinementView<'_>> = (0..capture.views.len())
            .map(|capture_index| RefinementView {
                capture_index,
                depth: Some(&depths[capture_index]),
            })
            .collect();
        let options = RefineOptions::default();

        let mut displaced = sphere_surfels(0.0375);
        let displaced_stats = refine(&mut displaced, &capture, &views, options);
        let displaced_error = displaced
            .iter()
            .map(|surfel| (glam::Vec3::from(surfel.center).length() - RADIUS).abs())
            .sum::<f32>()
            / displaced.len() as f32;
        assert!(
            displaced_error < 0.028,
            "known 0.0375 displacement remained {displaced_error}; stats={displaced_stats:?}"
        );
        let corrected = displaced
            .iter()
            .filter(|surfel| (glam::Vec3::from(surfel.center).length() - RADIUS).abs() < 0.002)
            .count();
        assert!(
            displaced_stats.moved >= displaced.len() / 3,
            "too few displaced surfels moved: {displaced_stats:?}"
        );
        assert_eq!(corrected, displaced_stats.moved);

        let mut exact = sphere_surfels(0.0);
        let exact_stats = refine(&mut exact, &capture, &views, options);
        let exact_error = exact
            .iter()
            .map(|surfel| (glam::Vec3::from(surfel.center).length() - RADIUS).abs())
            .sum::<f32>()
            / exact.len() as f32;
        assert!(
            exact_error < 1.0e-6 && exact_stats.moved == 0,
            "an exact sphere moved by {exact_error}; stats={exact_stats:?}"
        );
    }

    #[test]
    fn runtime_render_loss_recovers_known_surface_and_material_offsets() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        const SIZE: usize = 32;
        let cameras = [
            camera(glam::Vec3::new(0.0, 0.0, -1.2)),
            camera(glam::Vec3::new(0.25, 0.0, -1.2)),
        ];
        let make_scene = rendered_test_scene;

        let Ok(mut truth_renderer) = score::Renderer::new(SIZE, SIZE) else {
            eprintln!("skipping runtime render refinement test: no ray-tracing GPU");
            return;
        };
        let truth = make_scene(0.0);
        let rendered = truth_renderer.render_views(&truth, &cameras, 0, false);
        for (index, &camera) in cameras.iter().enumerate() {
            let single = truth_renderer.render_views(&truth, &[camera], 0, false);
            assert_eq!(rendered[index], single[0]);
        }
        let mut sampled = truth_renderer.prepare_scene(&truth, 2, false);
        let first_sampled = truth_renderer.render_prepared_views(&mut sampled, &cameras);
        sampled.reset_sampling();
        let second_sampled = truth_renderer.render_prepared_views(&mut sampled, &cameras);
        assert_eq!(first_sampled, second_sampled);
        truth_renderer.destroy_prepared_scene(sampled);
        let capture = capture::Capture {
            width: SIZE,
            height: SIZE,
            views: cameras
                .iter()
                .enumerate()
                .map(|(index, &camera)| capture::View {
                    name: format!("truth-{index}"),
                    camera,
                    pixels: rendered[index]
                        .iter()
                        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                        .collect(),
                    mask: None,
                })
                .collect(),
        };
        let mut scored = truth_renderer.prepare_scene(&truth, 2, false);
        scored.reset_sampling();
        let scalar_loss =
            truth_renderer.prepared_srgb_loss(&mut scored, &capture, &[0, 1], &cameras);
        scored.reset_sampling();
        let pixel_errors =
            truth_renderer.prepared_srgb_errors(&mut scored, &capture, &[0, 1], &cameras, 0.0);
        assert!((scalar_loss - mean_error(&pixel_errors)).abs() < 1.0e-8);
        truth_renderer.destroy_prepared_scene(scored);
        truth_renderer.destroy();

        let mut displaced = make_scene(0.075);
        let observations = decompose::observe(&displaced.model, &capture, &[0, 1], 0.1);
        assert_eq!(observations.seen(), 1);
        let stats = refine_rendered(
            &mut displaced,
            &capture,
            &[0, 1],
            &observations,
            0,
            0,
            false,
            1,
        )
        .unwrap();
        assert_eq!(stats.tested, 1);
        assert_eq!(stats.moved, 1);
        assert!(stats.final_loss < stats.initial_loss);
        assert!(displaced.model.surfels[0].center[2].abs() < 1.0e-6);

        let mut undersized = make_scene(0.0);
        undersized.model.surfels[0].radius = 0.24;
        let observations = decompose::observe(&undersized.model, &capture, &[0, 1], 0.1);
        let stats = refine_rendered(
            &mut undersized,
            &capture,
            &[0, 1],
            &observations,
            0,
            0,
            true,
            1,
        )
        .unwrap();
        assert_eq!(stats.radii_moved, 1);
        assert!(undersized.model.surfels[0].radius > 0.24);

        let mut recolored = make_scene(0.0);
        recolored.model.materials[0].albedo = [0.6, 0.4, 0.2];
        let stats = refine_rendered_materials(&mut recolored, &capture, &[0, 1], 0, 0.1)
            .expect("material refinement");
        assert_eq!(stats.coordinates, 6);
        assert_eq!(stats.proposals, 12);
        assert_eq!(stats.changed, 2);
        assert!(stats.final_loss < stats.initial_loss);
        for (actual, expected) in recolored.model.materials[0]
            .albedo
            .iter()
            .zip([0.7, 0.3, 0.2])
        {
            assert!(
                (actual - expected).abs() < 2.0e-5,
                "albedo {actual} != {expected}; stats={stats:?}"
            );
        }
        recolored.model.materials[0].albedo = [0.65, 0.35, 0.2];
        let stats = polish_rendered_materials(&mut recolored, &capture, &[0, 1], 0, 0.05)
            .expect("material re-polish");
        assert_eq!(stats.coordinates, 6);
        assert_eq!(stats.proposals, 12);
        assert_eq!(stats.changed, 2);
        assert!(stats.final_loss < stats.initial_loss);
        for (actual, expected) in recolored.model.materials[0]
            .albedo
            .iter()
            .zip([0.7, 0.3, 0.2])
        {
            assert!((actual - expected).abs() < 2.0e-5);
        }

        let gaussian_truth = make_scene(0.0);
        let mut truth_gaussian =
            crate::gaussian_splat::from_surface(&gaussian_truth.model).unwrap();
        crate::gaussian_splat::attach_pbr(&mut truth_gaussian, &gaussian_truth.model).unwrap();
        let mut gaussian_renderer = score::Renderer::new(SIZE, SIZE).unwrap();
        let mut gaussian_tracer =
            gaussian_renderer.prepare_gaussian_scene(&gaussian_truth, &truth_gaussian, false);
        let rendered = gaussian_renderer.render_prepared_views(&mut gaussian_tracer, &cameras);
        gaussian_renderer.destroy_prepared_scene(gaussian_tracer);
        gaussian_renderer.destroy();
        let gaussian_capture = capture::Capture {
            width: SIZE,
            height: SIZE,
            views: cameras
                .iter()
                .enumerate()
                .map(|(index, &camera)| capture::View {
                    name: format!("gaussian-truth-{index}"),
                    camera,
                    pixels: rendered[index]
                        .iter()
                        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                        .collect(),
                    mask: None,
                })
                .collect(),
        };
        let mut gaussian_recolored = make_scene(0.0);
        gaussian_recolored.model.materials[0].albedo = [0.6, 0.4, 0.2];
        let mut gaussian = crate::gaussian_splat::from_surface(&gaussian_recolored.model).unwrap();
        crate::gaussian_splat::attach_pbr(&mut gaussian, &gaussian_recolored.model).unwrap();
        let stats = polish_gaussian_materials(
            &gaussian_recolored,
            &mut gaussian,
            &gaussian_capture,
            &[0, 1],
            0.1,
            false,
        )
        .expect("Gaussian material re-polish");
        assert_eq!(stats.coordinates, 6);
        assert_eq!(stats.changed, 2);
        assert!(stats.final_loss < stats.initial_loss);
        assert_eq!(
            gaussian_recolored.model.materials[0].albedo,
            [0.6, 0.4, 0.2]
        );
        for (actual, expected) in gaussian
            .transforms
            .as_ref()
            .unwrap()
            .pbr
            .as_ref()
            .unwrap()
            .materials[0]
            .albedo
            .iter()
            .zip([0.7, 0.3, 0.2])
        {
            assert!((actual - expected).abs() < 2.0e-5);
        }

        let mut large_scene = gaussian_truth.clone();
        large_scene.model.materials[0].albedo = [0.35, 0.15, 0.1];
        large_scene
            .model
            .materials
            .resize(33, vol::relight::Material::default());
        let mut large_gaussian = crate::gaussian_splat::from_surface(&large_scene.model).unwrap();
        crate::gaussian_splat::attach_pbr(&mut large_gaussian, &large_scene.model).unwrap();
        let stats = polish_gaussian_materials(
            &large_scene,
            &mut large_gaussian,
            &gaussian_capture,
            &[0, 1],
            0.1,
            false,
        )
        .expect("large Gaussian material transfer");
        assert_eq!(stats.coordinates, 99);
        assert_eq!(stats.proposals, 1);
        assert!(stats.changed > 0);
        assert!(stats.final_loss < stats.initial_loss);
        assert!((stats.global_gain.unwrap() - 1.5).abs() < 1.0e-4);
        assert_eq!(stats.global_specular_allocation, None);

        let mut specular_truth = gaussian_truth.clone();
        allocate_global_specular(
            &mut specular_truth.model.materials,
            CONSERVATIVE_SPECULAR_ALLOCATION,
        );
        let mut truth_gaussian =
            crate::gaussian_splat::from_surface(&specular_truth.model).unwrap();
        crate::gaussian_splat::attach_pbr(&mut truth_gaussian, &specular_truth.model).unwrap();
        let mut renderer = score::Renderer::new(SIZE, SIZE).unwrap();
        let mut tracer = renderer.prepare_gaussian_scene(&specular_truth, &truth_gaussian, false);
        let rendered = renderer.render_prepared_views(&mut tracer, &cameras);
        renderer.destroy_prepared_scene(tracer);
        renderer.destroy();
        let specular_capture = capture::Capture {
            width: SIZE,
            height: SIZE,
            views: cameras
                .iter()
                .enumerate()
                .map(|(index, &camera)| capture::View {
                    name: format!("specular-truth-{index}"),
                    camera,
                    pixels: rendered[index]
                        .iter()
                        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                        .collect(),
                    mask: None,
                })
                .collect(),
        };
        let mut rough_scene = gaussian_truth.clone();
        let rough_material = rough_scene.model.materials[0];
        rough_scene.model.materials.resize(33, rough_material);
        let mut rough_gaussian = crate::gaussian_splat::from_surface(&rough_scene.model).unwrap();
        crate::gaussian_splat::attach_pbr(&mut rough_gaussian, &rough_scene.model).unwrap();
        let stats = polish_gaussian_materials(
            &rough_scene,
            &mut rough_gaussian,
            &specular_capture,
            &[0, 1],
            0.1,
            true,
        )
        .expect("large Gaussian specular allocation");
        assert_eq!(
            stats.global_specular_allocation,
            Some(CONSERVATIVE_SPECULAR_ALLOCATION)
        );
        assert!(stats.final_loss < stats.initial_loss);

        let mut first = make_scene(0.075);
        let mut second = first.clone();
        let observations = decompose::observe(&first.model, &capture, &[0, 1], 0.1);
        let first_stats = refine_rendered(
            &mut first,
            &capture,
            &[0, 1],
            &observations,
            0,
            64,
            false,
            0,
        )
        .unwrap();
        let second_stats = refine_rendered(
            &mut second,
            &capture,
            &[0, 1],
            &observations,
            0,
            64,
            false,
            0,
        )
        .unwrap();
        assert_eq!(first_stats.simultaneous_particles, 1);
        assert_eq!(first_stats.simultaneous_rounds, 64);
        assert!(first_stats.simultaneous_accepted > 0);
        assert!(first_stats.final_loss < first_stats.initial_loss);
        assert!(first.model.surfels[0].center[2].abs() < 0.075);
        assert_eq!(
            first.model.surfels[0].center,
            second.model.surfels[0].center
        );
        assert_eq!(first_stats.final_loss, second_stats.final_loss);
        assert_eq!(
            first_stats.simultaneous_accepted,
            second_stats.simultaneous_accepted
        );

        let mut final_renderer = score::Renderer::new(SIZE, SIZE).unwrap();
        let mut final_tracer = final_renderer.prepare_scene(&first, 0, false);
        let final_loss = rendered_loss(
            &mut final_renderer,
            &mut final_tracer,
            &capture,
            &[0, 1],
            &cameras,
        );
        final_renderer.destroy_prepared_scene(final_tracer);
        final_renderer.destroy();
        assert!((first_stats.final_loss - final_loss).abs() < 1.0e-8);
    }

    #[test]
    fn rendered_material_assignment_recovers_and_rolls_back_labels() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        const SIZE: usize = 32;
        let camera = camera(glam::Vec3::new(0.0, 0.0, -1.2));
        let truth = rendered_test_scene(0.0);
        let Ok(mut renderer) = score::Renderer::new(SIZE, SIZE) else {
            eprintln!("skipping rendered material assignment test: no ray-tracing GPU");
            return;
        };
        let rendered = renderer.render_views(&truth, &[camera], 0, false);
        renderer.destroy();
        let capture = capture::Capture {
            width: SIZE,
            height: SIZE,
            views: vec![capture::View {
                name: "material-truth".to_string(),
                camera,
                pixels: rendered[0]
                    .iter()
                    .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect(),
                mask: None,
            }],
        };
        let wrong = vol::relight::Material {
            albedo: [0.2, 0.7, 0.4],
            roughness: 1.0,
            specular_f0: [0.04; 3],
            _padding: 0.0,
        };

        let mut misassigned = truth.clone();
        misassigned.model.materials.push(wrong);
        misassigned.model.surfels[0].material = 1;
        let observations = decompose::observe(&misassigned.model, &capture, &[0], 0.1);
        let stats = refine_rendered_material_assignments(
            &mut misassigned,
            &capture,
            &[0],
            &observations,
            0,
            0.05,
        )
        .unwrap();
        assert_eq!(stats.candidates, 1);
        assert_eq!(stats.changed, 1);
        assert!(stats.final_loss < stats.initial_loss);
        assert_eq!(misassigned.model.surfels[0].material, 0);

        let mut protected = truth;
        protected.model.materials.push(wrong);
        let normal = glam::Vec3::from(protected.model.surfels[0].normal);
        let towards = (glam::Vec3::from(camera.cam_position)
            - glam::Vec3::from(protected.model.surfels[0].center))
        .normalize();
        let irradiance = protected.environment().diffuse_irradiance();
        let specular = vol::relight::SpecularEnvironment::prefilter(
            protected.environment(),
            protected.environment().width,
            protected.environment().height,
        );
        let misleading = decompose::Observations {
            samples: vec![decompose::Sample {
                view: 0,
                radiance: vol::relight::shade(normal, towards, &wrong, &irradiance, &specular),
                towards,
                facing: normal.dot(towards),
            }],
            offsets: vec![0, 1],
        };
        let stats = refine_rendered_material_assignments(
            &mut protected,
            &capture,
            &[0],
            &misleading,
            0,
            0.05,
        )
        .unwrap();
        assert_eq!(stats.candidates, 1);
        assert_eq!(stats.changed, 0);
        assert_eq!(stats.final_loss, stats.initial_loss);
        assert_eq!(protected.model.surfels[0].material, 0);
    }

    #[test]
    fn complete_render_refinement_moves_a_normal_towards_known_truth() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        const SIZE: usize = 32;
        let cameras = [
            camera(glam::Vec3::new(0.0, 0.0, -1.2)),
            camera(glam::Vec3::new(0.25, 0.0, -1.2)),
        ];
        let environment = vol::relight::Environment::sky(
            glam::Vec3::new(-0.5, 0.3, -1.0),
            [100.0, 80.0, 60.0],
            0.2,
            32,
            16,
        );
        let truth_normal = -glam::Vec3::Z;
        let truth = score::Scene::new(
            vol::relight::RelightModel {
                kernel: vol::relight::ParticleKernel::Gaussian,
                surfels: vec![vol::relight::Surfel {
                    center: [0.0; 3],
                    radius: 0.3,
                    normal: truth_normal.to_array(),
                    material: 0,
                }],
                materials: vec![vol::relight::Material {
                    albedo: [0.7, 0.3, 0.2],
                    roughness: 1.0,
                    specular_f0: [0.04; 3],
                    _padding: 0.0,
                }],
            },
            environment.clone(),
        );
        let Ok(mut truth_renderer) = score::Renderer::new(SIZE, SIZE) else {
            eprintln!("skipping rendered normal refinement test: no ray-tracing GPU");
            return;
        };
        let rendered = truth_renderer.render_views(&truth, &cameras, 0, false);
        truth_renderer.destroy();
        let capture = capture::Capture {
            width: SIZE,
            height: SIZE,
            views: cameras
                .iter()
                .enumerate()
                .map(|(index, &camera)| capture::View {
                    name: format!("normal-truth-{index}"),
                    camera,
                    pixels: rendered[index]
                        .iter()
                        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                        .collect(),
                    mask: None,
                })
                .collect(),
        };
        let mut candidate = truth.clone();
        let tilted = glam::Quat::from_rotation_y(15.0_f32.to_radians()) * truth_normal;
        candidate.model.surfels[0].normal = tilted.to_array();
        let observations = decompose::observe(&candidate.model, &capture, &[0, 1], -1.0);
        let evidence = [RenderedNormalEvidence {
            capture: &capture,
            indices: &[0, 1],
            environment: &environment,
        }];
        let stats =
            refine_rendered_normals(&mut candidate, &evidence, &observations, 0, 24, 5.0).unwrap();
        let refined = glam::Vec3::from(candidate.model.surfels[0].normal);
        assert!(stats.accepted > 0, "stats={stats:?}");
        assert!(stats.final_loss < stats.initial_loss, "stats={stats:?}");
        assert!(
            refined.angle_between(truth_normal) < tilted.angle_between(truth_normal),
            "tilted={tilted:?}, refined={refined:?}, stats={stats:?}"
        );
        let mut final_renderer = score::Renderer::new(SIZE, SIZE).unwrap();
        let mut final_tracer = final_renderer.prepare_scene(&candidate, 0, false);
        let final_loss = rendered_loss(
            &mut final_renderer,
            &mut final_tracer,
            &capture,
            &[0, 1],
            &cameras,
        );
        final_renderer.destroy_prepared_scene(final_tracer);
        final_renderer.destroy();
        assert!(
            (stats.final_loss - final_loss).abs() < 1.0e-7,
            "stats loss {}, rebuilt loss {final_loss}",
            stats.final_loss,
        );

        let mut undersized = truth.clone();
        undersized.model.surfels[0].radius = 0.24;
        let observations = decompose::observe(&undersized.model, &capture, &[0, 1], -1.0);
        let stats =
            refine_rendered_radii(&mut undersized, &evidence, &observations, 0, 24, 0.05).unwrap();
        assert!(stats.accepted > 0, "stats={stats:?}");
        assert!(stats.final_loss < stats.initial_loss, "stats={stats:?}");
        assert!(
            (undersized.model.surfels[0].radius - 0.3).abs() < 0.06,
            "radius={}, stats={stats:?}",
            undersized.model.surfels[0].radius,
        );
    }
}
