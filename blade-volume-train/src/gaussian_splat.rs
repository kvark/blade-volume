//! Direct image formation for Gaussian point clouds.
//!
//! The continuous response follows 3DGUT/3DGRT: project sigma points only to
//! find finite screen-space candidates, then evaluate each candidate at
//! its exact maximum-response point on the camera ray. Candidate indices are
//! discrete host inputs; all Gaussian parameters in the response and alpha
//! compositing remain differentiable Meganeura graph values.

use blade_graphics as gpu;
use blade_volume as vol;
use meganeura as mn;
use std::{sync, thread};

const SH_C0: f32 = 0.282_094_8;
const SH_C1: f32 = 0.488_602_52;
const MIN_SCALE: f32 = 1.0e-4;
const MAX_ALPHA: f32 = 0.999;
const TILE_SIZE: usize = 8;
const TILE_SUPPORT_MARGIN: f32 = 1.25;
const MIN_SH1_VIEWS: usize = 8;

/// Screen-space mean and covariance estimated from the seven 3DGUT sigma
/// points. The covariance columns follow `glam::Mat2`'s column-major layout.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedConic {
    pub mean: glam::Vec2,
    pub covariance: glam::Mat2,
}

/// Maximum-response sample of one Gaussian along a ray.
#[derive(Clone, Copy, Debug)]
pub struct RayResponse {
    /// Ray parameter at the maximum response. This is also the exact per-ray
    /// sort key used by the ray-traced Gaussian backend.
    pub depth: f32,
    /// Unit-opacity Gaussian kernel response at `depth`.
    pub response: f32,
}

/// Fixed-width candidate table consumed by [`build_graph`].
#[derive(Clone, Debug)]
pub struct FlatCandidates {
    pub indices: Vec<u32>,
    pub mask: Vec<f32>,
    pub pixel_indices: Vec<u32>,
    pub depths: Vec<f32>,
    pub pixels: usize,
    pub candidates_per_pixel: usize,
}

struct CandidateSlices<'a> {
    indices: &'a mut [u32],
    mask: &'a mut [f32],
    pixel_indices: &'a mut [u32],
    depths: &'a mut [f32],
}

struct CandidateGrid {
    width: usize,
    tiles_x: usize,
    tiles: Vec<Vec<u32>>,
}

struct CandidateIndex {
    views: Vec<Option<CandidateGrid>>,
    transforms: Vec<CandidateTransform>,
}

#[derive(Clone, Copy)]
struct CandidateTransform {
    mean: glam::Vec3,
    inverse_rotation: glam::Quat,
    scale: glam::Vec3,
}

impl CandidateIndex {
    fn new(
        model: &vol::PointCloudModel,
        capture: &crate::inverse::capture::Capture,
        view_indices: &[usize],
        min_alpha: f32,
    ) -> Self {
        let mut views: Vec<Option<CandidateGrid>> = std::iter::repeat_with(|| None)
            .take(capture.views.len())
            .collect();
        let transforms = model.transforms.as_ref().unwrap();
        let transforms: Vec<CandidateTransform> = model
            .points
            .iter()
            .zip(&transforms.rotations)
            .zip(&transforms.scales)
            .map(|((point, rotation), scale)| CandidateTransform {
                mean: point.truncate(),
                inverse_rotation: rotation.normalize().inverse(),
                scale: *scale,
            })
            .collect();
        if min_alpha == 0.0 {
            return Self { views, transforms };
        }
        for &view_index in view_indices {
            views[view_index] = Some(CandidateGrid::new(
                model,
                &capture.views[view_index].camera,
                capture.width,
                capture.height,
                min_alpha,
            ));
        }
        Self { views, transforms }
    }

    fn candidates(&self, view: usize, pixel: usize) -> Option<&[u32]> {
        let grid = self.views[view].as_ref()?;
        let x = pixel % grid.width;
        let y = pixel / grid.width;
        let tile = (y / TILE_SIZE) * grid.tiles_x + x / TILE_SIZE;
        Some(&grid.tiles[tile])
    }
}

impl CandidateGrid {
    fn new(
        model: &vol::PointCloudModel,
        camera: &vol::CameraParams,
        width: usize,
        height: usize,
        min_alpha: f32,
    ) -> Self {
        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        let mut tiles = vec![Vec::new(); tiles_x * tiles_y];
        let transforms = model.transforms.as_ref().unwrap();
        for (index, point) in model.points.iter().enumerate() {
            if point.w < min_alpha {
                continue;
            }
            let Some((min, max)) = projected_support_bounds(
                camera,
                width,
                height,
                point.truncate(),
                transforms.rotations[index],
                transforms.scales[index],
                point.w,
                min_alpha,
            ) else {
                continue;
            };
            let min_x = min.x.floor().max(0.0) as usize / TILE_SIZE;
            let min_y = min.y.floor().max(0.0) as usize / TILE_SIZE;
            let max_x = max.x.ceil().min(width.saturating_sub(1) as f32) as usize / TILE_SIZE;
            let max_y = max.y.ceil().min(height.saturating_sub(1) as f32) as usize / TILE_SIZE;
            if min_x > max_x || min_y > max_y || min_x >= tiles_x || min_y >= tiles_y {
                continue;
            }
            for tile_y in min_y..=max_y.min(tiles_y - 1) {
                for tile_x in min_x..=max_x.min(tiles_x - 1) {
                    tiles[tile_y * tiles_x + tile_x].push(index as u32);
                }
            }
        }
        Self {
            width,
            tiles_x,
            tiles,
        }
    }
}

/// Differentiable outputs and parameters of the Gaussian image-formation
/// graph.
#[derive(Clone, Copy, Debug)]
pub struct GaussianGraph {
    pub loss: mn::NodeId,
    pub pixels: [mn::NodeId; 3],
    pub opacity: mn::NodeId,
    pub positions: mn::NodeId,
    pub log_scales: mn::NodeId,
    pub opacity_logits: mn::NodeId,
    pub sh: [mn::NodeId; 3],
}

/// Minimal optimizer settings for direct Gaussian reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    pub steps: usize,
    pub batch_size: usize,
    pub candidates_per_pixel: usize,
    pub candidate_min_alpha: f32,
    pub geometry_sync_every: usize,
    pub position_learning_rate: f32,
    pub scale_learning_rate: f32,
    pub opacity_learning_rate: f32,
    pub sh_learning_rate: f32,
    pub opacity_loss_weight: f32,
    pub background: [f32; 3],
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            steps: 1_000,
            batch_size: 512,
            candidates_per_pixel: 64,
            candidate_min_alpha: 1.0e-4,
            geometry_sync_every: 25,
            position_learning_rate: 0.0,
            scale_learning_rate: 5.0e-3,
            opacity_learning_rate: 5.0e-2,
            sh_learning_rate: 2.5e-3,
            opacity_loss_weight: 0.0,
            background: [0.0; 3],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FitStats {
    pub steps: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
}

/// Statistics for the selected appearance-then-support training schedule.
#[derive(Clone, Copy, Debug)]
pub struct StagedFitStats {
    pub appearance: FitStats,
    pub support: FitStats,
}

fn staged_step_counts(steps: usize) -> Option<(usize, usize)> {
    (steps >= 2).then(|| {
        let appearance = (steps / 3).max(1);
        (appearance, steps - appearance)
    })
}

fn use_directional_appearance(view_count: usize) -> bool {
    view_count >= MIN_SH1_VIEWS
}

/// Apply the selected direct-Gaussian schedule to an established cloud.
///
/// The first third learns only view-independent appearance. The remaining
/// updates also learn opacity and three anisotropic scales, while keeping the
/// reconstructed centres fixed. With enough distinct training views they also
/// promote appearance to SH-1; smaller captures remain compact SH-0 because
/// their directional terms did not generalize in held-view gates.
/// Foreground opacity is supervised when every selected view carries a mask;
/// ordinary scene captures continue without that optional term.
pub fn fit_staged(
    model: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    gpu: sync::Arc<gpu::Context>,
) -> Result<StagedFitStats, String> {
    if model.sh_degree != 0 {
        return Err("staged direct Gaussian fitting requires an SH-0 input model".to_string());
    }
    let Some((appearance_steps, support_steps)) = staged_step_counts(steps) else {
        return Err("staged direct Gaussian fitting requires at least two updates".to_string());
    };
    let opacity_loss_weight = if view_indices.iter().all(|&index| {
        capture
            .views
            .get(index)
            .is_some_and(|view| view.mask.is_some())
    }) {
        1.5
    } else {
        0.0
    };
    let appearance_options = FitOptions {
        steps: appearance_steps,
        batch_size: 512,
        candidates_per_pixel: 64,
        candidate_min_alpha: 1.0e-5,
        geometry_sync_every: 10,
        position_learning_rate: 0.0,
        scale_learning_rate: 0.0,
        opacity_learning_rate: 0.0,
        sh_learning_rate: 0.002,
        opacity_loss_weight,
        background: [0.0; 3],
    };
    let appearance = fit(
        model,
        capture,
        view_indices,
        appearance_options,
        gpu.clone(),
    )?;
    if use_directional_appearance(view_indices.len()) {
        promote_to_sh_degree_one(model);
    }
    let support = fit(
        model,
        capture,
        view_indices,
        FitOptions {
            steps: support_steps,
            scale_learning_rate: 0.005,
            opacity_learning_rate: 0.05,
            ..appearance_options
        },
        gpu,
    )?;
    Ok(StagedFitStats {
        appearance,
        support,
    })
}

fn promote_to_sh_degree_one(model: &mut vol::PointCloudModel) {
    assert_eq!(model.sh_degree, 0);
    let mut coefficients = vec![0.0; model.points.len() * 4 * 3];
    for (index, dc) in model.sh_coefficients.chunks_exact(3).enumerate() {
        coefficients[index * 12..index * 12 + 3].copy_from_slice(dc);
    }
    model.sh_coefficients = coefficients;
    model.sh_degree = 1;
}

/// Convert an established relightable Gaussian surface into the neutral
/// `PointCloudModel` consumed by direct light-field training.
///
/// A relight surfel stores its finite support at three standard deviations;
/// the Gaussian backend stores one standard deviation. The surface normal is
/// kept as local Y, matching the extracted tangent frame. Appearance starts at
/// neutral grey instead of leaking the PBR material or its training light into
/// the static field.
pub fn from_surface(surface: &vol::relight::RelightModel) -> Result<vol::PointCloudModel, String> {
    surface.validate()?;
    if surface.kernel != vol::relight::ParticleKernel::Gaussian {
        return Err("direct Gaussian training requires a Gaussian surface".to_string());
    }
    if surface.is_empty() {
        return Err("direct Gaussian training requires at least one surface particle".to_string());
    }
    let count = surface.surfels.len();
    let model = vol::PointCloudModel {
        points: surface
            .surfels
            .iter()
            .map(|surfel| glam::Vec3::from(surfel.center).extend(0.5))
            .collect(),
        sh_coefficients: vec![0.0; count * 3],
        sh_degree: 0,
        transforms: Some(vol::Transforms {
            rotations: surface
                .surfels
                .iter()
                .map(|surfel| {
                    glam::Quat::from_rotation_arc(glam::Vec3::Y, glam::Vec3::from(surfel.normal))
                })
                .collect(),
            scales: surface
                .surfels
                .iter()
                .map(|surfel| glam::Vec3::splat(surfel.radius / 3.0))
                .collect(),
        }),
        adjacency: None,
        radii: None,
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    };
    model.validate()?;
    Ok(model)
}

struct RayBatch {
    origins: Vec<glam::Vec3>,
    directions: Vec<glam::Vec3>,
    views: Vec<usize>,
    pixels: Vec<usize>,
    labels: Vec<f32>,
    alpha: Vec<f32>,
}

/// Estimate a Gaussian's projected conic with the fixed 3DGUT parameters
/// `alpha=1`, `beta=2`, `kappa=0`.
///
/// Returns `None` when any sigma point crosses the camera plane. Near-plane
/// clipping needs a separate conservative bound; pretending the partial set is
/// a conic would allow false-negative culling.
pub fn project_conic(
    camera: &vol::CameraParams,
    width: usize,
    height: usize,
    mean: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) -> Option<ProjectedConic> {
    if !mean.is_finite()
        || !rotation.is_finite()
        || rotation.length_squared() <= 0.0
        || !scale.is_finite()
        || scale.min_element() <= 0.0
    {
        return None;
    }

    // lambda = 0, so sqrt(3 + lambda) = sqrt(3). R*S is a valid square
    // root of R*S*S^T*R^T and its columns are the sigma-point axes.
    let rotation = rotation.normalize();
    let root = 3.0_f32.sqrt();
    let axes = [
        rotation * (glam::Vec3::X * scale.x * root),
        rotation * (glam::Vec3::Y * scale.y * root),
        rotation * (glam::Vec3::Z * scale.z * root),
    ];
    let mut projected = [glam::Vec2::ZERO; 7];
    projected[0] =
        glam::Vec2::from(crate::inverse::capture::project(camera, width, height, mean)?.0);
    for (axis_index, axis) in axes.into_iter().enumerate() {
        projected[1 + axis_index] = glam::Vec2::from(
            crate::inverse::capture::project(camera, width, height, mean + axis)?.0,
        );
        projected[4 + axis_index] = glam::Vec2::from(
            crate::inverse::capture::project(camera, width, height, mean - axis)?.0,
        );
    }

    // With lambda=0 the centre has zero mean weight and every outer point has
    // weight 1/6. Its covariance weight is 2; outer weights remain 1/6.
    let projected_mean = projected[1..].iter().copied().sum::<glam::Vec2>() / 6.0;
    let mut covariance = glam::Mat2::ZERO;
    for (index, point) in projected.into_iter().enumerate() {
        let delta = point - projected_mean;
        let weight = if index == 0 { 2.0 } else { 1.0 / 6.0 };
        covariance += glam::Mat2::from_cols(delta * delta.x, delta * delta.y) * weight;
    }
    Some(ProjectedConic {
        mean: projected_mean,
        covariance,
    })
}

fn projected_support_bounds(
    camera: &vol::CameraParams,
    width: usize,
    height: usize,
    mean: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
    opacity: f32,
    min_alpha: f32,
) -> Option<(glam::Vec2, glam::Vec2)> {
    let response_threshold = (min_alpha / opacity).clamp(f32::MIN_POSITIVE, 1.0);
    let gaussian_radius = (-2.0 * response_threshold.ln()).sqrt() * TILE_SUPPORT_MARGIN;
    let world_radius = gaussian_radius * scale.max_element();
    let camera_rotation = glam::Quat::from_array(camera.cam_orientation);
    let local_mean = camera_rotation.inverse() * (mean - glam::Vec3::from(camera.cam_position));
    if local_mean.z + world_radius <= 1.0e-6 {
        return None;
    }
    if local_mean.z - world_radius <= 1.0e-6 {
        return Some((
            glam::Vec2::ZERO,
            glam::Vec2::new(
                width.saturating_sub(1) as f32,
                height.saturating_sub(1) as f32,
            ),
        ));
    }

    let conic = project_conic(camera, width, height, mean, rotation, scale)?;
    let conic_extent = glam::Vec2::new(
        (gaussian_radius * gaussian_radius * conic.covariance.x_axis.x.max(0.0)).sqrt(),
        (gaussian_radius * gaussian_radius * conic.covariance.y_axis.y.max(0.0)).sqrt(),
    );
    let min = conic.mean - conic_extent;
    let max = conic.mean + conic_extent;
    if max.x < 0.0 || max.y < 0.0 || min.x >= width as f32 || min.y >= height as f32 {
        None
    } else {
        Some((min, max))
    }
}

/// Evaluate the exact maximum of a 3D anisotropic Gaussian along a ray.
pub fn ray_response(
    ray_origin: glam::Vec3,
    ray_direction: glam::Vec3,
    mean: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) -> Option<RayResponse> {
    if !ray_origin.is_finite()
        || !ray_direction.is_finite()
        || ray_direction.length_squared() <= 0.0
        || !mean.is_finite()
        || !rotation.is_finite()
        || rotation.length_squared() <= 0.0
        || !scale.is_finite()
        || scale.min_element() <= 0.0
    {
        return None;
    }
    ray_response_transformed(
        ray_origin,
        ray_direction,
        CandidateTransform {
            mean,
            inverse_rotation: rotation.normalize().inverse(),
            scale,
        },
    )
}

fn ray_response_transformed(
    ray_origin: glam::Vec3,
    ray_direction: glam::Vec3,
    transform: CandidateTransform,
) -> Option<RayResponse> {
    let gaussian_origin =
        (transform.inverse_rotation * (ray_origin - transform.mean)) / transform.scale;
    let gaussian_direction = (transform.inverse_rotation * ray_direction) / transform.scale;
    let direction_squared = gaussian_direction.length_squared();
    if !direction_squared.is_finite() || direction_squared <= 0.0 {
        return None;
    }
    let depth = -gaussian_origin.dot(gaussian_direction) / direction_squared;
    let closest = gaussian_origin + depth * gaussian_direction;
    let response = (-0.5 * closest.length_squared()).exp();
    (depth.is_finite() && response.is_finite()).then_some(RayResponse { depth, response })
}

/// Record the closest exact Gaussian candidates for a ray batch.
///
/// This deliberately uses an `O(rays * particles)` CPU oracle. It is the
/// correctness path for small training gates and for checking the private tile
/// index without changing the differentiable graph.
pub fn record_candidates(
    model: &vol::PointCloudModel,
    ray_origins: &[glam::Vec3],
    ray_directions: &[glam::Vec3],
    candidates_per_pixel: usize,
    min_alpha: f32,
) -> FlatCandidates {
    assert_eq!(ray_origins.len(), ray_directions.len());
    assert!(candidates_per_pixel > 0);
    assert!(min_alpha.is_finite() && min_alpha >= 0.0);
    let transforms = model
        .transforms
        .as_ref()
        .expect("Gaussian candidate recording requires model transforms");
    assert_eq!(transforms.rotations.len(), model.points.len());
    assert_eq!(transforms.scales.len(), model.points.len());

    let pixels = ray_origins.len();
    let entries = pixels * candidates_per_pixel;
    let mut indices = vec![0_u32; entries];
    let mut mask = vec![0.0_f32; entries];
    let mut pixel_indices = vec![0_u32; entries];
    let mut depths = vec![0.0_f32; entries];
    let worker_count = if pixels < 1_024 {
        1
    } else {
        thread::available_parallelism()
            .map_or(1, |count| count.get())
            .min(8)
            .min(pixels)
    };
    let chunk_pixels = pixels.div_ceil(worker_count);
    thread::scope(|scope| {
        let mut remaining = CandidateSlices {
            indices: &mut indices,
            mask: &mut mask,
            pixel_indices: &mut pixel_indices,
            depths: &mut depths,
        };
        for worker in 0..worker_count {
            let begin = worker * chunk_pixels;
            let end = (begin + chunk_pixels).min(pixels);
            if begin == end {
                break;
            }
            let chunk_entries = (end - begin) * candidates_per_pixel;
            let (worker_indices, rest) = remaining.indices.split_at_mut(chunk_entries);
            remaining.indices = rest;
            let (worker_mask, rest) = remaining.mask.split_at_mut(chunk_entries);
            remaining.mask = rest;
            let (worker_pixel_indices, rest) = remaining.pixel_indices.split_at_mut(chunk_entries);
            remaining.pixel_indices = rest;
            let (worker_depths, rest) = remaining.depths.split_at_mut(chunk_entries);
            remaining.depths = rest;
            let origins = &ray_origins[begin..end];
            let directions = &ray_directions[begin..end];
            scope.spawn(move || {
                record_candidate_range(
                    model,
                    origins,
                    directions,
                    begin,
                    candidates_per_pixel,
                    min_alpha,
                    CandidateSlices {
                        indices: worker_indices,
                        mask: worker_mask,
                        pixel_indices: worker_pixel_indices,
                        depths: worker_depths,
                    },
                );
            });
        }
    });
    FlatCandidates {
        indices,
        mask,
        pixel_indices,
        depths,
        pixels,
        candidates_per_pixel,
    }
}

fn record_candidate_range(
    model: &vol::PointCloudModel,
    ray_origins: &[glam::Vec3],
    ray_directions: &[glam::Vec3],
    first_pixel: usize,
    candidates_per_pixel: usize,
    min_alpha: f32,
    output: CandidateSlices<'_>,
) {
    let mut hits = Vec::with_capacity(model.points.len());
    for (local_pixel, (&origin, &direction)) in ray_origins.iter().zip(ray_directions).enumerate() {
        collect_candidate_hits(
            model,
            origin,
            direction,
            0..model.points.len(),
            min_alpha,
            &mut hits,
        );
        hits.sort_unstable_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        for (slot, &(depth, index)) in hits.iter().take(candidates_per_pixel).enumerate() {
            let flat = local_pixel * candidates_per_pixel + slot;
            output.indices[flat] = index;
            output.mask[flat] = 1.0;
            output.depths[flat] = depth;
        }
        let pixel = first_pixel + local_pixel;
        let pixel = u32::try_from(pixel).expect("ray batch exceeds u32 pixel indices");
        output.pixel_indices
            [local_pixel * candidates_per_pixel..(local_pixel + 1) * candidates_per_pixel]
            .fill(pixel);
    }
}

fn collect_candidate_hits(
    model: &vol::PointCloudModel,
    origin: glam::Vec3,
    direction: glam::Vec3,
    indices: impl Iterator<Item = usize>,
    min_alpha: f32,
    hits: &mut Vec<(f32, u32)>,
) {
    hits.clear();
    let transforms = model.transforms.as_ref().unwrap();
    for index in indices {
        let point = model.points[index];
        let Some(hit) = ray_response(
            origin,
            direction,
            point.truncate(),
            transforms.rotations[index],
            transforms.scales[index],
        ) else {
            continue;
        };
        let alpha = point.w * hit.response;
        if hit.depth > 0.0 && alpha.is_finite() && alpha >= min_alpha {
            hits.push((hit.depth, index as u32));
        }
    }
}

fn collect_indexed_candidate_hits(
    model: &vol::PointCloudModel,
    index: &CandidateIndex,
    origin: glam::Vec3,
    direction: glam::Vec3,
    indices: &[u32],
    min_alpha: f32,
    hits: &mut Vec<(f32, u32)>,
) {
    hits.clear();
    for &particle in indices {
        let Some(hit) =
            ray_response_transformed(origin, direction, index.transforms[particle as usize])
        else {
            continue;
        };
        let alpha = model.points[particle as usize].w * hit.response;
        if hit.depth > 0.0 && alpha.is_finite() && alpha >= min_alpha {
            hits.push((hit.depth, particle));
        }
    }
}

fn record_indexed_candidates(
    model: &vol::PointCloudModel,
    batch: &RayBatch,
    index: &CandidateIndex,
    candidates_per_pixel: usize,
    min_alpha: f32,
) -> FlatCandidates {
    let pixels = batch.origins.len();
    let entries = pixels * candidates_per_pixel;
    let mut indices = vec![0_u32; entries];
    let mut mask = vec![0.0_f32; entries];
    let mut pixel_indices = vec![0_u32; entries];
    let mut depths = vec![0.0_f32; entries];
    let worker_count = thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(8)
        .min(pixels.div_ceil(64).max(1));
    let chunk_pixels = pixels.div_ceil(worker_count);
    thread::scope(|scope| {
        let mut remaining = CandidateSlices {
            indices: &mut indices,
            mask: &mut mask,
            pixel_indices: &mut pixel_indices,
            depths: &mut depths,
        };
        for worker in 0..worker_count {
            let begin = worker * chunk_pixels;
            let end = (begin + chunk_pixels).min(pixels);
            if begin == end {
                break;
            }
            let chunk_entries = (end - begin) * candidates_per_pixel;
            let (worker_indices, rest) = remaining.indices.split_at_mut(chunk_entries);
            remaining.indices = rest;
            let (worker_mask, rest) = remaining.mask.split_at_mut(chunk_entries);
            remaining.mask = rest;
            let (worker_pixel_indices, rest) = remaining.pixel_indices.split_at_mut(chunk_entries);
            remaining.pixel_indices = rest;
            let (worker_depths, rest) = remaining.depths.split_at_mut(chunk_entries);
            remaining.depths = rest;
            scope.spawn(move || {
                record_indexed_candidate_range(
                    model,
                    &batch.origins[begin..end],
                    &batch.directions[begin..end],
                    &batch.views[begin..end],
                    &batch.pixels[begin..end],
                    index,
                    begin,
                    candidates_per_pixel,
                    min_alpha,
                    CandidateSlices {
                        indices: worker_indices,
                        mask: worker_mask,
                        pixel_indices: worker_pixel_indices,
                        depths: worker_depths,
                    },
                );
            });
        }
    });
    FlatCandidates {
        indices,
        mask,
        pixel_indices,
        depths,
        pixels,
        candidates_per_pixel,
    }
}

#[allow(clippy::too_many_arguments)]
fn record_indexed_candidate_range(
    model: &vol::PointCloudModel,
    origins: &[glam::Vec3],
    directions: &[glam::Vec3],
    views: &[usize],
    pixels: &[usize],
    index: &CandidateIndex,
    first_pixel: usize,
    candidates_per_pixel: usize,
    min_alpha: f32,
    output: CandidateSlices<'_>,
) {
    let mut hits = Vec::new();
    for (local_pixel, (((&origin, &direction), &view), &pixel)) in origins
        .iter()
        .zip(directions)
        .zip(views)
        .zip(pixels)
        .enumerate()
    {
        match index.candidates(view, pixel) {
            Some(candidates) => collect_indexed_candidate_hits(
                model, index, origin, direction, candidates, min_alpha, &mut hits,
            ),
            None => collect_candidate_hits(
                model,
                origin,
                direction,
                0..model.points.len(),
                min_alpha,
                &mut hits,
            ),
        }
        hits.sort_unstable_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        for (slot, &(depth, particle)) in hits.iter().take(candidates_per_pixel).enumerate() {
            let flat = local_pixel * candidates_per_pixel + slot;
            output.indices[flat] = particle;
            output.mask[flat] = 1.0;
            output.depths[flat] = depth;
        }
        let pixel_index =
            u32::try_from(first_pixel + local_pixel).expect("ray batch exceeds u32 pixel indices");
        output.pixel_indices
            [local_pixel * candidates_per_pixel..(local_pixel + 1) * candidates_per_pixel]
            .fill(pixel_index);
    }
}

/// Render an SH-0 or SH-1 Gaussian model with the exact CPU response oracle.
///
/// This is the quality reference for the differentiable graph and future tile
/// culling. `candidates_per_pixel` is still a deliberate approximation to the
/// full sorted particle list and should be raised until the score converges.
/// RGB follows [`vol::PointCloudModel`]'s display-referred sRGB convention.
pub fn render_rays(
    model: &vol::PointCloudModel,
    ray_origins: &[glam::Vec3],
    ray_directions: &[glam::Vec3],
    candidates_per_pixel: usize,
    min_alpha: f32,
    background: [f32; 3],
) -> Vec<glam::Vec4> {
    assert!(
        model.sh_degree <= 1,
        "Gaussian CPU oracle supports SH degrees 0 and 1"
    );
    let candidates = record_candidates(
        model,
        ray_origins,
        ray_directions,
        candidates_per_pixel,
        min_alpha,
    );
    render_recorded(model, ray_origins, ray_directions, &candidates, background)
}

fn render_recorded(
    model: &vol::PointCloudModel,
    ray_origins: &[glam::Vec3],
    ray_directions: &[glam::Vec3],
    candidates: &FlatCandidates,
    background: [f32; 3],
) -> Vec<glam::Vec4> {
    let transforms = model
        .transforms
        .as_ref()
        .expect("Gaussian CPU oracle requires model transforms");
    let mut output = Vec::with_capacity(ray_origins.len());
    for pixel in 0..ray_origins.len() {
        let mut radiance = glam::Vec3::ZERO;
        let mut transmittance = 1.0_f32;
        for slot in 0..candidates.candidates_per_pixel {
            let flat = pixel * candidates.candidates_per_pixel + slot;
            if candidates.mask[flat] == 0.0 {
                continue;
            }
            let index = candidates.indices[flat] as usize;
            let point = model.points[index];
            let response = ray_response(
                ray_origins[pixel],
                ray_directions[pixel],
                point.truncate(),
                transforms.rotations[index],
                transforms.scales[index],
            )
            .unwrap();
            let alpha = (point.w * response.response).clamp(0.0, MAX_ALPHA);
            let color = vol::trace::eval_rgb_sh(model, index as u32, ray_directions[pixel]);
            radiance += transmittance * alpha * color;
            transmittance *= 1.0 - alpha;
        }
        radiance += transmittance * glam::Vec3::from(background);
        output.push(radiance.extend(1.0 - transmittance));
    }
    output
}

/// Score complete capture views with the exact CPU response oracle.
///
/// The returned values are display-referred sRGB PSNR, one per requested
/// view, matching the colour domain used by direct Gaussian fitting.
pub fn evaluate_views(
    model: &vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    candidates_per_pixel: usize,
    min_alpha: f32,
    background: [f32; 3],
) -> Result<Vec<f32>, String> {
    model.validate()?;
    if model.sh_degree > 1 || model.transforms.is_none() {
        return Err(
            "direct Gaussian evaluation requires transformed SH-0 or SH-1 Gaussians".to_string(),
        );
    }
    if candidates_per_pixel == 0 {
        return Err("direct Gaussian evaluation needs at least one candidate".to_string());
    }
    if capture.width == 0 || capture.height == 0 {
        return Err("direct Gaussian evaluation needs a non-empty capture extent".to_string());
    }
    if view_indices.is_empty() {
        return Err("direct Gaussian evaluation needs at least one view".to_string());
    }
    if !min_alpha.is_finite() || min_alpha < 0.0 {
        return Err("candidate minimum alpha must be finite and non-negative".to_string());
    }
    if let Some(&index) = view_indices
        .iter()
        .find(|&&index| index >= capture.views.len())
    {
        return Err(format!(
            "capture view {index} is outside {} available views",
            capture.views.len()
        ));
    }

    let pixels = capture.width * capture.height;
    let index = CandidateIndex::new(model, capture, view_indices, min_alpha);
    let mut scores = Vec::with_capacity(view_indices.len());
    for &view_index in view_indices {
        let view = &capture.views[view_index];
        let origin = glam::Vec3::from(view.camera.cam_position);
        let origins = vec![origin; pixels];
        let directions: Vec<_> = (0..capture.height)
            .flat_map(|y| {
                (0..capture.width).map(move |x| {
                    crate::inverse::capture::pixel_direction(
                        &view.camera,
                        capture.width,
                        capture.height,
                        x,
                        y,
                    )
                })
            })
            .collect();
        let batch = RayBatch {
            origins,
            directions,
            views: vec![view_index; pixels],
            pixels: (0..pixels).collect(),
            labels: Vec::new(),
            alpha: Vec::new(),
        };
        let candidates =
            record_indexed_candidates(model, &batch, &index, candidates_per_pixel, min_alpha);
        let rendered = render_recorded(
            model,
            &batch.origins,
            &batch.directions,
            &candidates,
            background,
        );
        let squared_error: f64 = rendered
            .iter()
            .zip(&view.pixels)
            .map(|(actual, expected)| {
                let expected =
                    glam::Vec3::from(*expected).map(crate::inverse::capture::linear_to_srgb);
                (actual.truncate() - expected).length_squared() as f64
            })
            .sum();
        let mse = squared_error / (3 * pixels) as f64;
        scores.push((-10.0 * mse.log10()) as f32);
    }
    Ok(scores)
}

/// Build a direct, differentiable anisotropic-Gaussian renderer using only
/// Meganeura's existing graph operations.
///
/// Candidate order is a discrete input and must be refreshed by the caller as
/// particles move. Positions, scales, opacity, SH colour, exact ray response,
/// and front-to-back compositing are all inside the graph. SH-1 reuses the
/// existing fused gather-times-basis reduction instead of a separate graph or
/// shader variant.
pub fn build_graph(
    g: &mut mn::Graph,
    particles: usize,
    pixels: usize,
    candidates_per_pixel: usize,
    sh_degree: usize,
    opacity_loss_weight: f32,
    background: [f32; 3],
) -> GaussianGraph {
    assert!(particles > 0);
    assert!(pixels > 0);
    assert!(candidates_per_pixel > 0);
    assert!(sh_degree <= 1);
    assert!(opacity_loss_weight.is_finite() && opacity_loss_weight >= 0.0);
    let rows = pixels * candidates_per_pixel;
    let sh_components = vol::get_sh_component_count(sh_degree);

    let candidate_indices = g.input_u32("candidate_indices", &[rows]);
    let pixel_indices = g.input_u32("candidate_pixel_indices", &[rows]);
    let mask = g.input("candidate_mask", &[rows, 1]);
    let ray_origins = g.input("ray_origins", &[pixels, 3]);
    let ray_directions = g.input("ray_directions", &[pixels, 3]);
    let sh_basis = g.input("sh_basis", &[pixels, sh_components]);
    let labels = g.input("labels", &[pixels, 3]);
    let target_alpha = g.input("target_alpha", &[pixels, 1]);
    let rotation_rows =
        ["rotation_x", "rotation_y", "rotation_z"].map(|name| g.input(name, &[particles, 3]));

    let positions = g.parameter("positions", &[particles, 3]);
    let log_scales = g.parameter("log_scales", &[particles, 3]);
    let opacity_logits = g.parameter("opacity_logits", &[particles, 1]);
    let sh = ["sh_r", "sh_g", "sh_b"].map(|name| g.parameter(name, &[particles, sh_components]));

    let centers = g.embedding(candidate_indices, positions);
    let origins = g.embedding(pixel_indices, ray_origins);
    let directions = g.embedding(pixel_indices, ray_directions);
    let negative_centers = g.neg(centers);
    let relative_origins = g.add(origins, negative_centers);

    let raw_scales = g.embedding(candidate_indices, log_scales);
    let scales = g.softplus(raw_scales, 1.0);
    let scale_floor = g.constant(vec![MIN_SCALE; rows * 3], &[rows, 3]);
    let scales = g.add(scales, scale_floor);
    let scale_x = g.split_a(scales, rows as u32, 1, 2, 1);
    let scale_x = g.reshape(scale_x, &[rows, 1]);
    let scale_yz = g.split_b(scales, rows as u32, 1, 2, 1);
    let scale_y = g.split_a(scale_yz, rows as u32, 1, 1, 1);
    let scale_y = g.reshape(scale_y, &[rows, 1]);
    let scale_z = g.split_b(scale_yz, rows as u32, 1, 1, 1);
    let scale_z = g.reshape(scale_z, &[rows, 1]);

    let mut gaussian_origins = Vec::with_capacity(3);
    let mut gaussian_directions = Vec::with_capacity(3);
    for (rotation_row, scale) in rotation_rows.into_iter().zip([scale_x, scale_y, scale_z]) {
        let basis = g.embedding(candidate_indices, rotation_row);
        let origin_product = g.mul(relative_origins, basis);
        let local_origin = g.sum_inner(origin_product);
        gaussian_origins.push(g.div(local_origin, scale));
        let direction_product = g.mul(directions, basis);
        let local_direction = g.sum_inner(direction_product);
        gaussian_directions.push(g.div(local_direction, scale));
    }
    let mut numerator_terms = Vec::with_capacity(3);
    let mut denominator_terms = Vec::with_capacity(3);
    for (&origin, &direction) in gaussian_origins.iter().zip(&gaussian_directions) {
        numerator_terms.push(g.mul(origin, direction));
        denominator_terms.push(g.mul(direction, direction));
    }
    let numerator_xy = g.add(numerator_terms[0], numerator_terms[1]);
    let numerator = g.add(numerator_xy, numerator_terms[2]);
    let numerator = g.neg(numerator);
    let denominator_xy = g.add(denominator_terms[0], denominator_terms[1]);
    let denominator = g.add(denominator_xy, denominator_terms[2]);
    let depth = g.div(numerator, denominator);
    let mut distance_terms = Vec::with_capacity(3);
    for (&origin, &direction) in gaussian_origins.iter().zip(&gaussian_directions) {
        let along_ray = g.mul(depth, direction);
        let closest = g.add(origin, along_ray);
        distance_terms.push(g.mul(closest, closest));
    }
    let distance_xy = g.add(distance_terms[0], distance_terms[1]);
    let normalized_distance_squared = g.add(distance_xy, distance_terms[2]);
    let negative_half = g.constant(vec![-0.5_f32; rows], &[rows, 1]);
    let exponent = g.mul(normalized_distance_squared, negative_half);
    let response = g.exp(exponent);

    let raw_opacity = g.embedding(candidate_indices, opacity_logits);
    let opacity = g.sigmoid(raw_opacity);
    let alpha_cap = g.constant(vec![MAX_ALPHA; rows], &[rows, 1]);
    let alpha = g.mul(opacity, response);
    let alpha = g.mul(alpha, mask);
    let alpha = g.mul(alpha, alpha_cap);
    let alpha_2d = g.reshape(alpha, &[pixels, candidates_per_pixel]);
    let ones = g.constant(vec![1.0_f32; rows], &[pixels, candidates_per_pixel]);
    let negative_alpha = g.neg(alpha_2d);
    let one_minus_alpha = g.add(ones, negative_alpha);
    let log_transmission = g.log(one_minus_alpha);
    let log_prefix = g.exclusive_cumsum(log_transmission, false);
    let transmittance = g.exp(log_prefix);
    let weight = g.mul(transmittance, alpha_2d);
    let accumulated_opacity = g.sum_inner(weight);

    let basis = g.embedding(pixel_indices, sh_basis);
    let bias = g.constant(vec![0.5_f32; rows], &[rows, 1]);
    let remaining = {
        let ones = g.constant(vec![1.0_f32; pixels], &[pixels, 1]);
        let negative_opacity = g.neg(accumulated_opacity);
        g.add(ones, negative_opacity)
    };
    let mut rendered = Vec::with_capacity(3);
    let mut losses = Vec::with_capacity(3);
    for (channel, background) in sh.into_iter().zip(background) {
        let coefficients = g.embedding(candidate_indices, channel);
        let terms = g.mul(coefficients, basis);
        let value = g.sum_inner(terms);
        let color = g.add(value, bias);
        let color = g.relu(color);
        let color_2d = g.reshape(color, &[pixels, candidates_per_pixel]);
        let weighted = g.mul(weight, color_2d);
        let pixel = g.sum_inner(weighted);
        let pixel = if background == 0.0 {
            pixel
        } else {
            let background = g.constant(vec![background; pixels], &[pixels, 1]);
            let uncovered = g.mul(remaining, background);
            g.add(pixel, uncovered)
        };
        rendered.push(pixel);
    }
    let label_r = g.split_a(labels, pixels as u32, 1, 2, 1);
    let label_r = g.reshape(label_r, &[pixels, 1]);
    let label_gb = g.split_b(labels, pixels as u32, 1, 2, 1);
    let label_gb = g.reshape(label_gb, &[pixels, 2]);
    let label_g = g.split_a(label_gb, pixels as u32, 1, 1, 1);
    let label_g = g.reshape(label_g, &[pixels, 1]);
    let label_b = g.split_b(label_gb, pixels as u32, 1, 1, 1);
    let label_b = g.reshape(label_b, &[pixels, 1]);
    for (&pixel, target) in rendered.iter().zip([label_r, label_g, label_b]) {
        losses.push(g.l1_loss(pixel, target));
    }
    let rg_loss = g.add(losses[0], losses[1]);
    let color_loss = g.add(rg_loss, losses[2]);
    let loss = if opacity_loss_weight == 0.0 {
        color_loss
    } else {
        let opacity_loss = g.mse_loss(accumulated_opacity, target_alpha);
        let scale = g.constant(vec![opacity_loss_weight], &[1]);
        let scaled_opacity_loss = g.mul(opacity_loss, scale);
        g.add(color_loss, scaled_opacity_loss)
    };
    let pixels: [mn::NodeId; 3] = rendered.try_into().expect("three rendered channels");
    g.set_outputs(vec![
        loss,
        pixels[0],
        pixels[1],
        pixels[2],
        accumulated_opacity,
    ]);

    GaussianGraph {
        loss,
        pixels,
        opacity: accumulated_opacity,
        positions,
        log_scales,
        opacity_logits,
        sh,
    }
}

fn inverse_softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < 1.0e-6 {
        value.ln()
    } else {
        value.exp_m1().ln()
    }
}

fn validate_fit(
    model: &vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    options: FitOptions,
) -> Result<(), String> {
    model.validate()?;
    if model.sh_degree > 1 {
        return Err("direct Gaussian fitting currently supports SH degrees 0 and 1".to_string());
    }
    if let Some((index, point)) = model
        .points
        .iter()
        .enumerate()
        .find(|&(_, point)| point.w > MAX_ALPHA)
    {
        return Err(format!(
            "Gaussian {index} has opacity {}; direct fitting requires opacity <= {MAX_ALPHA}",
            point.w
        ));
    }
    model
        .transforms
        .as_ref()
        .ok_or_else(|| "direct Gaussian fitting requires model transforms".to_string())?;
    if view_indices.is_empty() {
        return Err("direct Gaussian fitting needs at least one view".to_string());
    }
    if capture.width == 0 || capture.height == 0 {
        return Err("direct Gaussian fitting needs a non-empty capture extent".to_string());
    }
    if let Some(&index) = view_indices
        .iter()
        .find(|&&index| index >= capture.views.len())
    {
        return Err(format!(
            "capture view {index} is outside {} available views",
            capture.views.len()
        ));
    }
    if options.opacity_loss_weight > 0.0
        && view_indices
            .iter()
            .any(|&index| capture.views[index].mask.is_none())
    {
        return Err("opacity loss requires a foreground mask on every selected view".into());
    }
    if options.steps == 0
        || options.batch_size == 0
        || options.candidates_per_pixel == 0
        || options.geometry_sync_every == 0
    {
        return Err("fit steps, batch size, candidates, and sync interval must be non-zero".into());
    }
    if !options.candidate_min_alpha.is_finite() || options.candidate_min_alpha < 0.0 {
        return Err("candidate minimum alpha must be finite and non-negative".into());
    }
    let rates = [
        options.position_learning_rate,
        options.scale_learning_rate,
        options.opacity_learning_rate,
        options.sh_learning_rate,
    ];
    if rates.iter().any(|rate| !rate.is_finite() || *rate < 0.0) {
        return Err("Gaussian learning rates must be finite and non-negative".into());
    }
    if !options.opacity_loss_weight.is_finite() || options.opacity_loss_weight < 0.0 {
        return Err("opacity loss weight must be finite and non-negative".into());
    }
    Ok(())
}

fn model_parameters(model: &vol::PointCloudModel) -> [Vec<f32>; 6] {
    let transforms = model.transforms.as_ref().unwrap();
    let positions = model
        .points
        .iter()
        .flat_map(|point| point.truncate().to_array())
        .collect();
    let log_scales = transforms
        .scales
        .iter()
        .flat_map(|scale| {
            scale
                .to_array()
                .map(|value| inverse_softplus((value - MIN_SCALE).max(1.0e-8)))
        })
        .collect();
    let opacity_logits = model
        .points
        .iter()
        .map(|point| {
            let activated = (point.w / MAX_ALPHA).clamp(1.0e-6, 1.0 - 1.0e-6);
            (activated / (1.0 - activated)).ln()
        })
        .collect();
    let sh_components = model.sh_component_count();
    let mut sh: [Vec<f32>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(model.points.len() * sh_components));
    for coefficients in model.sh_coefficients.chunks_exact(3) {
        for channel in 0..3 {
            sh[channel].push(coefficients[channel]);
        }
    }
    [
        positions,
        log_scales,
        opacity_logits,
        std::mem::take(&mut sh[0]),
        std::mem::take(&mut sh[1]),
        std::mem::take(&mut sh[2]),
    ]
}

fn set_model_parameters(session: &mut mn::Session, model: &vol::PointCloudModel) {
    let [positions, log_scales, opacity_logits, sh_r, sh_g, sh_b] = model_parameters(model);
    session.set_parameter("positions", &positions);
    session.set_parameter("log_scales", &log_scales);
    session.set_parameter("opacity_logits", &opacity_logits);
    session.set_parameter("sh_r", &sh_r);
    session.set_parameter("sh_g", &sh_g);
    session.set_parameter("sh_b", &sh_b);
}

fn set_rotation_inputs(session: &mut mn::Session, model: &vol::PointCloudModel) {
    let rotations = &model.transforms.as_ref().unwrap().rotations;
    let mut rows: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(rotations.len() * 3));
    for &rotation in rotations {
        for (values, axis) in rows
            .iter_mut()
            .zip([glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z])
        {
            values.extend((rotation.normalize() * axis).to_array());
        }
    }
    for (name, values) in ["rotation_x", "rotation_y", "rotation_z"]
        .into_iter()
        .zip(rows)
    {
        session.set_input(name, &values);
    }
}

fn download_model(session: &mn::Session, model: &mut vol::PointCloudModel) {
    let count = model.points.len();
    let sh_components = model.sh_component_count();
    let mut positions = vec![0.0_f32; count * 3];
    let mut log_scales = vec![0.0_f32; count * 3];
    let mut opacity_logits = vec![0.0_f32; count];
    let mut sh: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0_f32; count * sh_components]);
    session.read_param("positions", &mut positions);
    session.read_param("log_scales", &mut log_scales);
    session.read_param("opacity_logits", &mut opacity_logits);
    for (name, values) in ["sh_r", "sh_g", "sh_b"].into_iter().zip(&mut sh) {
        session.read_param(name, values);
    }
    let transforms = model.transforms.as_mut().unwrap();
    for index in 0..count {
        let position = glam::Vec3::from_slice(&positions[3 * index..3 * index + 3]);
        let opacity = MAX_ALPHA / (1.0 + (-opacity_logits[index]).exp());
        model.points[index] = position.extend(opacity);
        let scale = glam::Vec3::from_array(std::array::from_fn(|axis| {
            let value = log_scales[3 * index + axis];
            MIN_SCALE
                + if value > 20.0 {
                    value
                } else {
                    value.exp().ln_1p()
                }
        }));
        transforms.scales[index] = scale;
        for component in 0..sh_components {
            for (channel, values) in sh.iter().enumerate() {
                let source = index * sh_components + component;
                let target = (index * sh_components + component) * 3 + channel;
                model.sh_coefficients[target] = values[source];
            }
        }
    }
}

fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn sample_rays(
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    count: usize,
    sequence: u32,
) -> RayBatch {
    let mut origins = Vec::with_capacity(count);
    let mut directions = Vec::with_capacity(count);
    let mut views = Vec::with_capacity(count);
    let mut pixels = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count * 3);
    let mut alpha = Vec::with_capacity(count);
    let pixel_count = capture.width * capture.height;
    for lane in 0..count {
        let key = hash(sequence.wrapping_mul(0x9e37_79b9) ^ lane as u32);
        let view_index = view_indices[hash(key ^ 0xa511_e9b3) as usize % view_indices.len()];
        let pixel = hash(key ^ 0x63d8_3595) as usize % pixel_count;
        let x = pixel % capture.width;
        let y = pixel / capture.width;
        let view = &capture.views[view_index];
        origins.push(glam::Vec3::from(view.camera.cam_position));
        directions.push(crate::inverse::capture::pixel_direction(
            &view.camera,
            capture.width,
            capture.height,
            x,
            y,
        ));
        views.push(view_index);
        pixels.push(pixel);
        labels.extend(
            view.pixels[pixel]
                .into_iter()
                .map(crate::inverse::capture::linear_to_srgb),
        );
        alpha.push(view.mask.as_ref().map_or(0.0, |mask| mask[pixel]));
    }
    RayBatch {
        origins,
        directions,
        views,
        pixels,
        labels,
        alpha,
    }
}

fn set_batch(
    session: &mut mn::Session,
    model: &vol::PointCloudModel,
    batch: &RayBatch,
    index: &CandidateIndex,
    options: FitOptions,
) {
    let candidates = record_indexed_candidates(
        model,
        batch,
        index,
        options.candidates_per_pixel,
        options.candidate_min_alpha,
    );
    let origins: Vec<f32> = batch
        .origins
        .iter()
        .flat_map(|origin| origin.to_array())
        .collect();
    let directions: Vec<f32> = batch
        .directions
        .iter()
        .flat_map(|direction| direction.to_array())
        .collect();
    let sh_components = model.sh_component_count();
    let sh_basis: Vec<f32> = batch
        .directions
        .iter()
        .flat_map(|direction| {
            [
                SH_C0,
                -SH_C1 * direction.y,
                SH_C1 * direction.z,
                -SH_C1 * direction.x,
            ]
            .into_iter()
            .take(sh_components)
        })
        .collect();
    session.set_input_u32("candidate_indices", &candidates.indices);
    session.set_input_u32("candidate_pixel_indices", &candidates.pixel_indices);
    session.set_input("candidate_mask", &candidates.mask);
    session.set_input("ray_origins", &origins);
    session.set_input("ray_directions", &directions);
    session.set_input("sh_basis", &sh_basis);
    session.set_input("labels", &batch.labels);
    session.set_input("target_alpha", &batch.alpha);
}

fn read_loss(session: &mn::Session) -> f32 {
    let mut loss = [0.0_f32];
    session.read_output_by_index(0, &mut loss);
    loss[0]
}

fn changes_candidate_geometry(options: FitOptions) -> bool {
    options.position_learning_rate > 0.0
        || options.scale_learning_rate > 0.0
        || options.opacity_learning_rate > 0.0
}

/// Fit a Gaussian light field directly to posed RGB images.
///
/// This is intentionally the small baseline: fixed particle count, SH-0/1, and
/// exact CPU response testing after private screen-tile culling. It establishes
/// whether direct Gaussian image formation improves reconstruction before
/// learned rotations, higher-order SH, or densification add implementation
/// surface area. Capture radiance is converted from linear light to the
/// display-referred sRGB convention stored by [`vol::PointCloudModel`].
pub fn fit(
    model: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    options: FitOptions,
    gpu: sync::Arc<gpu::Context>,
) -> Result<FitStats, String> {
    validate_fit(model, capture, view_indices, options)?;
    let mut graph = mn::Graph::new();
    build_graph(
        &mut graph,
        model.points.len(),
        options.batch_size,
        options.candidates_per_pixel,
        model.sh_degree,
        options.opacity_loss_weight,
        options.background,
    );
    let (mut session, _report) = mn::build(
        &graph,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );
    set_model_parameters(&mut session, model);
    set_rotation_inputs(&mut session, model);
    let mut candidate_index =
        CandidateIndex::new(model, capture, view_indices, options.candidate_min_alpha);

    let audit = sample_rays(capture, view_indices, options.batch_size, u32::MAX);
    set_batch(&mut session, model, &audit, &candidate_index, options);
    session.step();
    session.wait();
    let initial_loss = read_loss(&session);

    session.set_adam(1.0, 0.9, 0.999, 1.0e-8);
    session.set_lr_multiplier("positions", options.position_learning_rate);
    session.set_lr_multiplier("log_scales", options.scale_learning_rate);
    session.set_lr_multiplier("opacity_logits", options.opacity_learning_rate);
    session.set_lr_multiplier("sh_", options.sh_learning_rate);
    let sync_candidate_geometry = changes_candidate_geometry(options);
    for step in 0..options.steps {
        let batch = sample_rays(capture, view_indices, options.batch_size, step as u32);
        set_batch(&mut session, model, &batch, &candidate_index, options);
        session.step();
        session.wait();
        if sync_candidate_geometry && (step + 1) % options.geometry_sync_every == 0 {
            download_model(&session, model);
            candidate_index =
                CandidateIndex::new(model, capture, view_indices, options.candidate_min_alpha);
        }
    }
    download_model(&session, model);

    session.clear_optimizer();
    candidate_index =
        CandidateIndex::new(model, capture, view_indices, options.candidate_min_alpha);
    set_batch(&mut session, model, &audit, &candidate_index, options);
    session.step();
    session.wait();
    let final_loss = read_loss(&session);
    if !initial_loss.is_finite() || !final_loss.is_finite() {
        return Err(format!(
            "direct Gaussian fit produced a non-finite loss ({initial_loss} -> {final_loss})"
        ));
    }
    Ok(FitStats {
        steps: options.steps,
        initial_loss,
        final_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(points: Vec<glam::Vec4>) -> vol::PointCloudModel {
        let count = points.len();
        vol::PointCloudModel {
            points,
            sh_coefficients: vec![0.0; count * 3],
            sh_degree: 0,
            transforms: Some(vol::Transforms {
                rotations: vec![glam::Quat::IDENTITY; count],
                scales: vec![glam::Vec3::ONE; count],
            }),
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
        }
    }

    #[test]
    fn gaussian_surface_conversion_preserves_geometry_without_material_leakage() {
        let surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![
                vol::relight::Surfel {
                    center: [1.0, 2.0, 3.0],
                    radius: 0.6,
                    normal: glam::Vec3::Z.to_array(),
                    material: 0,
                },
                vol::relight::Surfel {
                    center: [-1.0, 0.0, 4.0],
                    radius: 0.3,
                    normal: (-glam::Vec3::Y).to_array(),
                    material: 1,
                },
            ],
            materials: vec![
                vol::relight::Material::default(),
                vol::relight::Material {
                    albedo: [1.0, 0.0, 0.0],
                    ..vol::relight::Material::default()
                },
            ],
        };
        let model = from_surface(&surface).unwrap();
        assert_eq!(model.points[0], glam::Vec4::new(1.0, 2.0, 3.0, 0.5));
        assert_eq!(model.points[1], glam::Vec4::new(-1.0, 0.0, 4.0, 0.5));
        assert_eq!(model.sh_coefficients, [0.0; 6]);
        let transforms = model.transforms.unwrap();
        assert_close(transforms.scales[0].x, 0.2, 1.0e-6);
        assert_close(transforms.scales[1].x, 0.1, 1.0e-6);
        for (rotation, surfel) in transforms.rotations.iter().zip(&surface.surfels) {
            let normal = glam::Vec3::from(surfel.normal);
            assert!((*rotation * glam::Vec3::Y).dot(normal) > 0.9999);
        }
    }

    #[test]
    fn selected_fit_defaults_keep_established_centres_fixed() {
        assert_eq!(FitOptions::default().position_learning_rate, 0.0);
    }

    #[test]
    fn appearance_only_updates_do_not_rebuild_candidate_geometry() {
        let appearance = FitOptions {
            position_learning_rate: 0.0,
            scale_learning_rate: 0.0,
            opacity_learning_rate: 0.0,
            ..FitOptions::default()
        };
        assert!(!changes_candidate_geometry(appearance));
        assert!(changes_candidate_geometry(FitOptions {
            scale_learning_rate: 0.01,
            ..appearance
        }));
    }

    #[test]
    fn staged_budget_prioritizes_support_and_keeps_both_stages_nonempty() {
        assert_eq!(staged_step_counts(1), None);
        assert_eq!(staged_step_counts(2), Some((1, 1)));
        assert_eq!(staged_step_counts(1_500), Some((500, 1_000)));
        assert!(!use_directional_appearance(MIN_SH1_VIEWS - 1));
        assert!(use_directional_appearance(MIN_SH1_VIEWS));
    }

    fn synthetic_capture(model: &vol::PointCloudModel) -> crate::inverse::capture::Capture {
        let width = 9;
        let height = 7;
        let target = glam::Vec3::new(0.0, 0.0, 4.0);
        let origins = [
            glam::Vec3::new(-1.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.3, 0.0),
        ];
        let transforms = model.transforms.as_ref().unwrap();
        let point = model.points[0];
        let views = origins
            .into_iter()
            .enumerate()
            .map(|(index, origin)| {
                let orientation =
                    glam::Quat::from_rotation_arc(glam::Vec3::Z, (target - origin).normalize());
                let camera = vol::CameraParams {
                    cam_position: origin.into(),
                    cam_orientation: orientation.into(),
                    fov: [50.0_f32.to_radians(), 40.0_f32.to_radians()],
                    depth: 100.0,
                    principal: [0.0; 2],
                };
                let pixels = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let direction = crate::inverse::capture::pixel_direction(
                                &camera, width, height, x, y,
                            );
                            let response = ray_response(
                                origin,
                                direction,
                                point.truncate(),
                                transforms.rotations[0],
                                transforms.scales[0],
                            )
                            .unwrap();
                            let color = vol::trace::eval_rgb_sh(model, 0, direction);
                            let radiance = if response.depth > 0.0 {
                                point.w * response.response * color.x
                            } else {
                                0.0
                            };
                            [crate::inverse::capture::srgb_to_linear(radiance); 3]
                        })
                    })
                    .collect();
                crate::inverse::capture::View {
                    name: format!("synthetic-{index}"),
                    camera,
                    pixels,
                    mask: None,
                }
            })
            .collect();
        crate::inverse::capture::Capture {
            width,
            height,
            views,
        }
    }

    #[test]
    fn exact_view_evaluation_scores_its_source_model() {
        let bright = (0.8 - 0.5) / SH_C0;
        let mut truth = model(vec![glam::Vec4::new(0.0, 0.0, 4.0, 0.8)]);
        truth.sh_coefficients.fill(bright);
        truth.transforms.as_mut().unwrap().scales[0] = glam::Vec3::splat(0.5);
        let capture = synthetic_capture(&truth);
        let scores = evaluate_views(&truth, &capture, &[0, 1], 1, 0.0, [0.0; 3]).unwrap();
        assert!(scores.iter().all(|score| *score > 100.0), "{scores:?}");
    }

    #[test]
    fn sh_degree_one_evaluation_scores_its_source_model() {
        let bright = (0.7 - 0.5) / SH_C0;
        let directional = 0.15 / SH_C1;
        let mut truth = model(vec![glam::Vec4::new(0.0, 0.0, 4.0, 0.8)]);
        truth.sh_degree = 1;
        truth.sh_coefficients = vec![
            bright,
            bright,
            bright,
            0.0,
            0.0,
            0.0,
            directional,
            directional,
            directional,
            0.0,
            0.0,
            0.0,
        ];
        truth.transforms.as_mut().unwrap().scales[0] = glam::Vec3::splat(0.5);
        let capture = synthetic_capture(&truth);
        let scores = evaluate_views(&truth, &capture, &[0, 1], 1, 0.0, [0.0; 3]).unwrap();
        assert!(scores.iter().all(|score| *score > 100.0), "{scores:?}");
    }

    #[test]
    fn degree_one_promotion_preserves_dc_and_zeros_directional_terms() {
        let mut model = model(vec![glam::Vec4::W, glam::Vec4::ONE]);
        model.sh_coefficients = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        promote_to_sh_degree_one(&mut model);
        assert_eq!(model.sh_degree, 1);
        assert_eq!(
            model.sh_coefficients,
            [
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
    }

    #[test]
    fn degree_one_model_parameters_are_channel_major_tables() {
        let mut model = model(vec![glam::Vec4::W, glam::Vec4::ONE]);
        model.sh_degree = 1;
        model.sh_coefficients = (0..24).map(|value| value as f32).collect();
        let [_, _, _, red, green, blue] = model_parameters(&model);
        assert_eq!(red, [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]);
        assert_eq!(green, [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]);
        assert_eq!(blue, [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0]);
    }

    fn empty_capture(width: usize, height: usize) -> crate::inverse::capture::Capture {
        let cameras = [
            vol::CameraParams::looking_at(
                glam::Vec3::ZERO,
                glam::Vec3::new(0.0, 0.0, 4.0),
                60.0_f32.to_radians(),
                width as f32 / height as f32,
                20.0,
            ),
            vol::CameraParams::looking_at(
                glam::Vec3::new(1.0, 0.5, 0.0),
                glam::Vec3::new(0.0, 0.0, 4.0),
                60.0_f32.to_radians(),
                width as f32 / height as f32,
                20.0,
            ),
        ];
        crate::inverse::capture::Capture {
            width,
            height,
            views: cameras
                .into_iter()
                .enumerate()
                .map(|(index, camera)| crate::inverse::capture::View {
                    name: format!("empty-{index}"),
                    camera,
                    pixels: vec![[0.0; 3]; width * height],
                    mask: Some(vec![0.0; width * height]),
                })
                .collect(),
        }
    }

    fn camera() -> vol::CameraParams {
        vol::CameraParams {
            cam_position: [0.0, 0.0, 0.0],
            cam_orientation: glam::Quat::IDENTITY.into(),
            fov: [90.0_f32.to_radians(); 2],
            depth: 100.0,
            principal: [0.0; 2],
        }
    }

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn centered_isotropic_projection_matches_pinhole_jacobian() {
        let conic = project_conic(
            &camera(),
            200,
            200,
            glam::Vec3::new(0.0, 0.0, 10.0),
            glam::Quat::IDENTITY,
            glam::Vec3::splat(0.5),
        )
        .unwrap();
        assert_close(conic.mean.x, 100.0, 1.0e-5);
        assert_close(conic.mean.y, 100.0, 1.0e-5);
        // fx = width / (2*tan(fov/2)) = 100, so (fx*s/z)^2 = 25.
        assert_close(conic.covariance.x_axis.x, 25.0, 1.0e-3);
        assert_close(conic.covariance.y_axis.y, 25.0, 1.0e-3);
        assert_close(conic.covariance.x_axis.y, 0.0, 1.0e-4);
        assert_close(conic.covariance.y_axis.x, 0.0, 1.0e-4);
    }

    #[test]
    fn nonlinear_projection_produces_finite_positive_covariance() {
        let conic = project_conic(
            &camera(),
            320,
            180,
            glam::Vec3::new(1.0, -0.5, 4.0),
            glam::Quat::from_rotation_y(0.4),
            glam::Vec3::new(0.2, 0.4, 0.7),
        )
        .unwrap();
        assert!(conic.mean.is_finite());
        assert!(conic.covariance.is_finite());
        assert!(conic.covariance.determinant() > 0.0);
        assert!(conic.covariance.x_axis.x > 0.0);
        assert!(conic.covariance.y_axis.y > 0.0);
    }

    #[test]
    fn ray_response_is_exact_in_gaussian_space() {
        let through_center = ray_response(
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Quat::IDENTITY,
            glam::Vec3::splat(0.5),
        )
        .unwrap();
        assert_close(through_center.depth, 5.0, 1.0e-6);
        assert_close(through_center.response, 1.0, 1.0e-6);

        let offset = ray_response(
            glam::Vec3::new(0.5, 0.0, 0.0),
            glam::Vec3::Z,
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Quat::IDENTITY,
            glam::Vec3::splat(0.5),
        )
        .unwrap();
        assert_close(offset.depth, 5.0, 1.0e-6);
        assert_close(offset.response, (-0.5_f32).exp(), 1.0e-6);
    }

    #[test]
    fn cached_candidate_transform_is_bit_exact() {
        let origin = glam::Vec3::new(0.7, -0.2, -3.0);
        let direction = glam::Vec3::new(0.1, 0.05, 1.0).normalize();
        let mean = glam::Vec3::new(0.2, 0.4, 2.0);
        let rotation = glam::Quat::from_rotation_y(0.4);
        let scale = glam::Vec3::new(0.3, 0.8, 1.1);
        let direct = ray_response(origin, direction, mean, rotation, scale).unwrap();
        let cached = ray_response_transformed(
            origin,
            direction,
            CandidateTransform {
                mean,
                inverse_rotation: rotation.normalize().inverse(),
                scale,
            },
        )
        .unwrap();

        assert_eq!(cached.depth.to_bits(), direct.depth.to_bits());
        assert_eq!(cached.response.to_bits(), direct.response.to_bits());
    }

    #[test]
    fn anisotropic_response_is_rotation_invariant() {
        let rotation = glam::Quat::from_rotation_z(0.5);
        let origin = glam::Vec3::new(0.7, -0.2, -3.0);
        let direction = glam::Vec3::new(0.1, 0.05, 1.0).normalize();
        let mean = glam::Vec3::new(0.2, 0.4, 2.0);
        let scale = glam::Vec3::new(0.3, 0.8, 1.1);
        let reference = ray_response(origin, direction, mean, rotation, scale).unwrap();
        let world_rotation = glam::Quat::from_rotation_y(-0.8);
        let transformed = ray_response(
            world_rotation * origin,
            world_rotation * direction,
            world_rotation * mean,
            world_rotation * rotation,
            scale,
        )
        .unwrap();
        assert_close(transformed.depth, reference.depth, 1.0e-5);
        assert_close(transformed.response, reference.response, 1.0e-5);
    }

    #[test]
    fn candidate_recorder_sorts_by_maximum_response_depth() {
        let model = model(vec![
            glam::Vec4::new(0.0, 0.0, 4.0, 0.8),
            glam::Vec4::new(0.0, 0.0, 2.0, 0.8),
            glam::Vec4::new(0.0, 0.0, 3.0, 0.8),
            glam::Vec4::new(0.0, 0.0, -1.0, 0.8),
        ]);
        let recorded = record_candidates(&model, &[glam::Vec3::ZERO], &[glam::Vec3::Z], 3, 0.01);
        assert_eq!(recorded.indices, [1, 2, 0]);
        assert_eq!(recorded.mask, [1.0, 1.0, 1.0]);
        assert_eq!(recorded.depths, [2.0, 3.0, 4.0]);
        assert_eq!(recorded.pixel_indices, [0, 0, 0]);
    }

    #[test]
    fn tiled_candidates_match_exhaustive_recording() {
        let points: Vec<glam::Vec4> = (0..25)
            .map(|index| {
                let x = (index % 5) as f32 * 0.7 - 1.4;
                let y = (index / 5) as f32 * 0.55 - 1.1;
                glam::Vec4::new(x, y, 3.5 + 0.05 * index as f32, 0.8)
            })
            .collect();
        let mut model = model(points);
        model
            .transforms
            .as_mut()
            .unwrap()
            .scales
            .fill(glam::Vec3::splat(0.12));
        let capture = empty_capture(64, 48);
        let batch = sample_rays(&capture, &[0, 1], 1_024, 17);
        let exhaustive = record_candidates(&model, &batch.origins, &batch.directions, 8, 1.0e-4);
        let index = CandidateIndex::new(&model, &capture, &[0, 1], 1.0e-4);
        let tiled = record_indexed_candidates(&model, &batch, &index, 8, 1.0e-4);
        assert_eq!(tiled.indices, exhaustive.indices);
        assert_eq!(tiled.mask, exhaustive.mask);
        assert_eq!(tiled.depths, exhaustive.depths);
        assert_eq!(tiled.pixel_indices, exhaustive.pixel_indices);

        let candidate_entries: usize = batch
            .views
            .iter()
            .zip(&batch.pixels)
            .map(|(&view, &pixel)| index.candidates(view, pixel).unwrap().len())
            .sum();
        let exhaustive_entries = batch.origins.len() * model.points.len();
        assert!(candidate_entries < exhaustive_entries / 2);
    }

    #[test]
    fn anisotropic_graph_matches_oracle_and_has_position_gradient() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping anisotropic Gaussian graph test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        build_graph(&mut graph, 2, 1, 2, 1, 0.0, [0.0; 3]);
        let (mut session, _report) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );

        session.set_parameter("positions", &[0.25, 0.0, 2.0, 0.0, 0.0, 4.0]);
        let front_scale = glam::Vec3::new(0.5, 1.5, 2.0);
        let back_scale = glam::Vec3::ONE;
        let log_scales: Vec<f32> = [front_scale, back_scale]
            .into_iter()
            .flat_map(|scale| {
                scale
                    .to_array()
                    .map(|value| inverse_softplus(value - MIN_SCALE))
            })
            .collect();
        session.set_parameter("log_scales", &log_scales);
        session.set_parameter("opacity_logits", &[0.0; 2]);
        let bright = 0.5 / SH_C0;
        let directional = 0.2 / SH_C1;
        session.set_parameter(
            "sh_r",
            &[bright, 0.0, directional, 0.0, -bright, 0.0, 0.0, 0.0],
        );
        session.set_parameter(
            "sh_g",
            &[-bright, 0.0, 0.0, 0.0, bright, 0.0, -directional, 0.0],
        );
        session.set_parameter("sh_b", &[-bright, 0.0, 0.0, 0.0, -bright, 0.0, 0.0, 0.0]);
        session.set_input_u32("candidate_indices", &[0, 1]);
        session.set_input_u32("candidate_pixel_indices", &[0, 0]);
        session.set_input("candidate_mask", &[1.0, 1.0]);
        session.set_input("ray_origins", &[0.0, 0.0, 0.0]);
        session.set_input("ray_directions", &[0.0, 0.0, 1.0]);
        session.set_input("sh_basis", &[SH_C0, 0.0, SH_C1, 0.0]);
        session.set_input("labels", &[0.2, 0.2, 0.1]);
        session.set_input("target_alpha", &[0.0]);
        let front_rotation = glam::Quat::from_rotation_y(0.6);
        for (name, axis) in [
            ("rotation_x", glam::Vec3::X),
            ("rotation_y", glam::Vec3::Y),
            ("rotation_z", glam::Vec3::Z),
        ] {
            let values: Vec<f32> = [front_rotation * axis, axis]
                .into_iter()
                .flat_map(|value| value.to_array())
                .collect();
            session.set_input(name, &values);
        }
        session.step();
        session.wait();

        let response = ray_response(
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            glam::Vec3::new(0.25, 0.0, 2.0),
            front_rotation,
            front_scale,
        )
        .unwrap();
        let front_alpha = 0.5 * response.response * MAX_ALPHA;
        let back_alpha = 0.5 * MAX_ALPHA;
        let expected = [
            1.2 * front_alpha,
            0.8 * (1.0 - front_alpha) * back_alpha,
            0.0,
            front_alpha + (1.0 - front_alpha) * back_alpha,
        ];
        for (output, expected) in (1..=4).zip(expected) {
            let mut value = [0.0_f32];
            session.read_output_by_index(output, &mut value);
            assert_close(value[0], expected, 2.0e-5);
        }

        let mut position_gradient = [0.0_f32; 6];
        session.read_param_grad("positions", &mut position_gradient);
        assert!(position_gradient.iter().all(|value| value.is_finite()));
        assert!(position_gradient[0].abs() > 1.0e-5);
        let mut scale_gradient = [0.0_f32; 6];
        session.read_param_grad("log_scales", &mut scale_gradient);
        assert!(scale_gradient.iter().all(|value| value.is_finite()));
        assert!(scale_gradient[..3].iter().any(|value| value.abs() > 1.0e-5));
        let mut sh_gradient = [0.0_f32; 8];
        session.read_param_grad("sh_r", &mut sh_gradient);
        assert!(sh_gradient.iter().all(|value| value.is_finite()));
        assert!(sh_gradient[2].abs() > 1.0e-5);
    }

    #[test]
    fn direct_fit_recovers_multiview_position() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping direct Gaussian fit test: no GPU");
            return;
        };
        let bright = (0.8 - 0.5) / SH_C0;
        let mut truth = model(vec![glam::Vec4::new(0.0, 0.0, 4.0, 0.8)]);
        truth.sh_coefficients.fill(bright);
        let transforms = truth.transforms.as_mut().unwrap();
        transforms.rotations[0] = glam::Quat::from_rotation_y(0.4);
        transforms.scales[0] = glam::Vec3::new(0.35, 0.55, 0.45);
        let capture = synthetic_capture(&truth);

        let mut fitted = truth.clone();
        fitted.points[0] = glam::Vec4::new(0.45, -0.2, 3.6, 0.8);
        let initial_distance = fitted.points[0]
            .truncate()
            .distance(truth.points[0].truncate());
        let stats = fit(
            &mut fitted,
            &capture,
            &[0, 1],
            FitOptions {
                steps: 200,
                batch_size: 64,
                candidates_per_pixel: 1,
                candidate_min_alpha: 0.0,
                geometry_sync_every: 1,
                position_learning_rate: 0.02,
                scale_learning_rate: 0.0,
                opacity_learning_rate: 0.0,
                sh_learning_rate: 0.0,
                opacity_loss_weight: 0.0,
                background: [0.0; 3],
            },
            gpu,
        )
        .unwrap();
        let final_distance = fitted.points[0]
            .truncate()
            .distance(truth.points[0].truncate());
        assert!(
            stats.final_loss < 0.4 * stats.initial_loss,
            "loss did not converge: {} -> {}",
            stats.initial_loss,
            stats.final_loss
        );
        assert!(
            final_distance < 0.25 * initial_distance,
            "position did not converge: {initial_distance} -> {final_distance}"
        );
    }
}
