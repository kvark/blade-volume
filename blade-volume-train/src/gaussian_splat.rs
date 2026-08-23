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
use std::{mem, sync, thread};

const SH_C0: f32 = 0.282_094_8;
const SH_C1: f32 = 0.488_602_52;
const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
const MIN_SCALE: f32 = 1.0e-4;
const MAX_ALPHA: f32 = 0.999;
const TILE_SIZE: usize = 8;
const TILE_SUPPORT_MARGIN: f32 = 1.25;
const MIN_SH1_VIEWS: usize = 8;
const MIN_SH2_VIEWS: usize = 18;
const LIGHT_FIELD_POSITION_LEARNING_RATE: f32 = 1.0e-4;
const MULTI_LIGHT_GEOMETRY_STEPS: usize = 50;
const MULTI_LIGHT_GEOMETRY_POSITION_LEARNING_RATE: f32 = 4.0e-4;
const MULTI_LIGHT_NORMAL_LEARNING_RATE: f32 = 5.0e-4;
const MULTI_LIGHT_NORMAL_CONTRAST_STEPS: usize = 25;
const MULTI_LIGHT_OPACITY_CONDITIONED_COLOR_CEILING: f32 = 0.5;
const BACKGROUND_ONLY_OPACITY_SCALE: f32 = 2.0 / 3.0;
// Match the selected support geometry-refresh cadence so one preparation pass
// never crosses a point at which candidate transforms could change.
const PREPARED_CANDIDATE_BATCHES: usize = 20;
// Preserve learned anisotropy while carrying a small, cross-representation
// residual from the final production-render support refinement. Full scalar
// radius agreement over-expands the Gaussian; this log-space fraction is the
// smallest tested joint-safe setting across five clouds and two real gates.
const PBR_SUPPORT_FEEDBACK: f32 = 0.025;
const PBR_MIN_PERSISTED_OPACITY: f32 = 0.05;
const STATIC_CONTINUATION_MIN_VALIDATION_GAIN_DB: f32 = 0.05;

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

struct CandidateRows {
    indices: Vec<u32>,
    mask: Vec<f32>,
}

struct CandidateRowSlices<'a> {
    indices: &'a mut [u32],
    mask: &'a mut [f32],
}

struct CandidateGrid {
    width: usize,
    tiles_x: usize,
    tiles: Vec<Vec<u32>>,
    gaussian_origins: Vec<glam::Vec3>,
}

struct CandidateIndex {
    views: Vec<Option<CandidateGrid>>,
    transforms: Vec<CandidateTransform>,
    max_distance_squared: Vec<f32>,
}

#[derive(Clone, Copy)]
struct CandidateTransform {
    mean: glam::Vec3,
    world_to_gaussian: glam::Mat3,
}

struct ProjectionSupport {
    mean: glam::Vec3,
    axes: [glam::Vec3; 3],
    gaussian_radius: f32,
    world_radius: f32,
}

fn candidate_transform(
    mean: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) -> CandidateTransform {
    let inverse_rotation = glam::Mat3::from_quat(rotation.normalize().inverse());
    CandidateTransform {
        mean,
        world_to_gaussian: glam::Mat3::from_diagonal(scale.recip()) * inverse_rotation,
    }
}

fn candidate_max_distance_squared(opacity: f32, min_alpha: f32) -> f32 {
    if min_alpha == 0.0 {
        f32::INFINITY
    } else if opacity >= min_alpha {
        -2.0 * (min_alpha / opacity).ln()
    } else {
        -1.0
    }
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
        let model_transforms = model.transforms.as_ref().unwrap();
        let transforms: Vec<CandidateTransform> = model
            .points
            .iter()
            .zip(&model_transforms.rotations)
            .zip(&model_transforms.scales)
            .map(|((point, rotation), scale)| {
                candidate_transform(point.truncate(), *rotation, *scale)
            })
            .collect();
        let max_distance_squared: Vec<f32> = model
            .points
            .iter()
            .map(|point| candidate_max_distance_squared(point.w, min_alpha))
            .collect();
        if min_alpha == 0.0 {
            return Self {
                views,
                transforms,
                max_distance_squared,
            };
        }
        let projection_supports: Vec<_> = model
            .points
            .iter()
            .zip(&model_transforms.rotations)
            .zip(&model_transforms.scales)
            .map(|((point, rotation), scale)| {
                ProjectionSupport::new(*point, *rotation, *scale, min_alpha)
            })
            .collect();
        let worker_count = thread::available_parallelism()
            .map_or(1, |count| count.get())
            .min(view_indices.len().max(1));
        let chunk_views = view_indices.len().div_ceil(worker_count).max(1);
        thread::scope(|scope| {
            let handles: Vec<_> = view_indices
                .chunks(chunk_views)
                .map(|chunk| {
                    scope.spawn(|| {
                        chunk
                            .iter()
                            .map(|&view_index| {
                                (
                                    view_index,
                                    CandidateGrid::new(
                                        &capture.views[view_index].camera,
                                        capture.width,
                                        capture.height,
                                        &transforms,
                                        &projection_supports,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for handle in handles {
                for (view_index, grid) in handle.join().unwrap() {
                    views[view_index] = Some(grid);
                }
            }
        });
        Self {
            views,
            transforms,
            max_distance_squared,
        }
    }

    fn candidates(&self, view: usize, pixel: usize) -> Option<(&[u32], &[glam::Vec3])> {
        let grid = self.views[view].as_ref()?;
        let x = pixel % grid.width;
        let y = pixel / grid.width;
        let tile = (y / TILE_SIZE) * grid.tiles_x + x / TILE_SIZE;
        Some((&grid.tiles[tile], &grid.gaussian_origins))
    }

    fn locality_key(&self, view: usize, pixel: usize) -> (usize, usize) {
        let Some(ref grid) = self.views[view] else {
            return (view, pixel);
        };
        let x = pixel % grid.width;
        let y = pixel / grid.width;
        (view, (y / TILE_SIZE) * grid.tiles_x + x / TILE_SIZE)
    }
}

impl CandidateGrid {
    fn new(
        camera: &vol::CameraParams,
        width: usize,
        height: usize,
        candidate_transforms: &[CandidateTransform],
        projection_supports: &[Option<ProjectionSupport>],
    ) -> Self {
        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        let mut tiles = vec![Vec::new(); tiles_x * tiles_y];
        let camera_origin = glam::Vec3::from(camera.cam_position);
        let gaussian_origins = candidate_transforms
            .iter()
            .map(|transform| transform.world_to_gaussian * (camera_origin - transform.mean))
            .collect();
        let projection = crate::inverse::capture::PixelProjection::new(camera, width, height);
        for (index, support) in projection_supports.iter().enumerate() {
            let Some(ref support) = *support else {
                continue;
            };
            let Some((min, max)) = projected_support_bounds(&projection, width, height, support)
            else {
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
            gaussian_origins,
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

/// One calibrated image capture used to separate Gaussian geometry from
/// illumination-dependent appearance.
#[derive(Clone, Copy)]
pub struct KnownLightCapture<'a> {
    pub capture: &'a crate::inverse::capture::Capture,
    pub environment: &'a vol::relight::Environment,
}

/// Statistics for the selected appearance-then-support training schedule.
#[derive(Clone, Copy, Debug)]
pub struct StagedFitStats {
    pub appearance: FitStats,
    pub support: FitStats,
}

/// Statistics for two Gaussian outputs that share appearance initialization.
#[derive(Clone, Copy, Debug)]
pub struct SharedStagedFitStats {
    pub appearance: FitStats,
    pub pbr_support: FitStats,
    pub light_field_support: FitStats,
}

/// Training-only evidence for choosing an optional static-surface continuation.
#[derive(Clone, Copy, Debug)]
pub struct StaticSurfaceSelection {
    pub use_continued: bool,
    pub baseline_validation_psnr: f32,
    pub continued_validation_psnr: f32,
}

/// Statistics for a paired PBR support and static light-field fit.
pub type OutputFitStats = SharedStagedFitStats;

fn static_continuation_wins(baseline: f32, continued: f32) -> bool {
    continued >= baseline + STATIC_CONTINUATION_MIN_VALIDATION_GAIN_DB
}

/// Choose between two static Gaussian surfaces on one withheld training camera.
///
/// Both probes learn appearance and support from `fit_views`; the validation
/// camera is never passed to either optimizer. The continued surface must win
/// by a small margin so atomic-gradient noise cannot select a more expensive
/// path. The caller can then fit the selected surface normally on every
/// training view.
pub fn validate_static_surface_continuation(
    baseline: &vol::relight::RelightModel,
    continued: &vol::relight::RelightModel,
    capture: &crate::inverse::capture::Capture,
    fit_views: &[usize],
    validation_view: usize,
    steps: usize,
    gpu: sync::Arc<gpu::Context>,
) -> Result<StaticSurfaceSelection, String> {
    if fit_views.contains(&validation_view) {
        return Err("static continuation validation view must be withheld from fitting".into());
    }
    if validation_view >= capture.views.len() {
        return Err(format!(
            "static continuation validation view {validation_view} is outside {} available views",
            capture.views.len()
        ));
    }
    let mut baseline_gaussian = from_surface(baseline)?;
    let mut continued_gaussian = from_surface(continued)?;
    fit_staged_light_field(
        &mut baseline_gaussian,
        capture,
        fit_views,
        steps,
        gpu.clone(),
    )?;
    fit_staged_light_field(&mut continued_gaussian, capture, fit_views, steps, gpu)?;
    let baseline_validation_psnr = evaluate_views(
        &baseline_gaussian,
        capture,
        &[validation_view],
        64,
        1.0e-5,
        [0.0; 3],
    )?[0];
    let continued_validation_psnr = evaluate_views(
        &continued_gaussian,
        capture,
        &[validation_view],
        64,
        1.0e-5,
        [0.0; 3],
    )?[0];
    Ok(StaticSurfaceSelection {
        use_continued: static_continuation_wins(
            baseline_validation_psnr,
            continued_validation_psnr,
        ),
        baseline_validation_psnr,
        continued_validation_psnr,
    })
}

struct MultilightFit {
    stats: FitStats,
    normals: Option<Vec<[f32; 3]>>,
}

#[derive(Clone, Copy)]
enum OpacityLoss {
    Mask,
    BackgroundOnly,
}

fn staged_step_counts(steps: usize) -> Option<(usize, usize)> {
    (steps >= 2).then(|| {
        let appearance = (steps / 3).max(1);
        (appearance, steps - appearance)
    })
}

fn selected_sh_degree(view_count: usize) -> usize {
    if view_count >= MIN_SH2_VIEWS {
        2
    } else if view_count >= MIN_SH1_VIEWS {
        1
    } else {
        0
    }
}

fn sh_basis(direction: glam::Vec3) -> [f32; 9] {
    let squared = direction * direction;
    [
        SH_C0,
        -SH_C1 * direction.y,
        SH_C1 * direction.z,
        -SH_C1 * direction.x,
        SH_C2[0] * direction.x * direction.y,
        SH_C2[1] * direction.y * direction.z,
        SH_C2[2] * (3.0 * squared.z - 1.0),
        SH_C2[3] * direction.x * direction.z,
        SH_C2[4] * (squared.x - squared.y),
    ]
}

/// Apply the fixed-centre direct-Gaussian schedule used to learn PBR support.
///
/// The first third learns only view-independent appearance. The remaining
/// updates also learn opacity and three anisotropic scales, while keeping the
/// reconstructed centres fixed. With enough distinct training views they also
/// promote appearance to SH-1 or SH-2 according to the available view count;
/// smaller captures remain compact because higher directional terms did not
/// generalize in held-view gates.
/// Known background rays penalize contradictory PBR support when every selected
/// view carries a mask; ordinary scene captures continue without that optional
/// term. Static light fields retain full-mask supervision because foreground
/// opacity is directly part of that output.
pub fn fit_staged(
    model: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    gpu: sync::Arc<gpu::Context>,
) -> Result<StagedFitStats, String> {
    fit_staged_impl(
        model,
        capture,
        view_indices,
        steps,
        0.0,
        OpacityLoss::BackgroundOnly,
        gpu,
    )
}

/// Apply the direct-Gaussian schedule while also learning static light-field centres.
pub fn fit_staged_light_field(
    model: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    gpu: sync::Arc<gpu::Context>,
) -> Result<StagedFitStats, String> {
    fit_staged_impl(
        model,
        capture,
        view_indices,
        steps,
        LIGHT_FIELD_POSITION_LEARNING_RATE,
        OpacityLoss::Mask,
        gpu,
    )
}

/// Fit fixed-centre PBR support and a learned-centre static light field.
///
/// The light field may append particles after the PBR prefix. Appearance-only
/// initialization is shared because neither geometry nor support changes in
/// that stage; the two outputs then receive independent support optimization.
pub fn fit_staged_outputs(
    pbr: &mut vol::PointCloudModel,
    light_field: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    gpu: sync::Arc<gpu::Context>,
) -> Result<SharedStagedFitStats, String> {
    validate_shared_prefix(pbr, light_field)?;
    let (appearance_options, light_field_options, sh_degree) = staged_options(
        capture,
        view_indices,
        steps,
        LIGHT_FIELD_POSITION_LEARNING_RATE,
    )?;
    let appearance = fit(
        light_field,
        capture,
        view_indices,
        appearance_options,
        gpu.clone(),
    )?;
    copy_appearance_prefix(pbr, light_field);
    if sh_degree > 0 {
        promote_to_sh_degree(pbr, sh_degree);
        promote_to_sh_degree(light_field, sh_degree);
    }
    let pbr_support = fit_with_opacity_loss(
        pbr,
        capture,
        view_indices,
        FitOptions {
            position_learning_rate: 0.0,
            ..light_field_options
        },
        OpacityLoss::BackgroundOnly,
        gpu.clone(),
    )?;
    let light_field_support = fit(light_field, capture, view_indices, light_field_options, gpu)?;
    Ok(SharedStagedFitStats {
        appearance,
        pbr_support,
        light_field_support,
    })
}

/// Fit PBR support and a static light field whose input geometry differs.
pub fn fit_staged_independent_outputs(
    pbr: &mut vol::PointCloudModel,
    light_field: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    gpu: sync::Arc<gpu::Context>,
) -> Result<OutputFitStats, String> {
    let pbr_stats = fit_staged(pbr, capture, view_indices, steps, gpu.clone())?;
    let light_field_stats = fit_staged_light_field(light_field, capture, view_indices, steps, gpu)?;
    Ok(OutputFitStats {
        appearance: light_field_stats.appearance,
        pbr_support: pbr_stats.support,
        light_field_support: light_field_stats.support,
    })
}

fn fit_staged_impl(
    model: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    position_learning_rate: f32,
    opacity_loss: OpacityLoss,
    gpu: sync::Arc<gpu::Context>,
) -> Result<StagedFitStats, String> {
    if model.sh_degree != 0 {
        return Err("staged direct Gaussian fitting requires an SH-0 input model".to_string());
    }
    let (appearance_options, support_options, sh_degree) =
        staged_options(capture, view_indices, steps, position_learning_rate)?;
    let appearance = fit_with_opacity_loss(
        model,
        capture,
        view_indices,
        appearance_options,
        opacity_loss,
        gpu.clone(),
    )?;
    if sh_degree > 0 {
        promote_to_sh_degree(model, sh_degree);
    }
    let support = fit_with_opacity_loss(
        model,
        capture,
        view_indices,
        support_options,
        opacity_loss,
        gpu,
    )?;
    Ok(StagedFitStats {
        appearance,
        support,
    })
}

fn staged_options(
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    steps: usize,
    position_learning_rate: f32,
) -> Result<(FitOptions, FitOptions, usize), String> {
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
        geometry_sync_every: 20,
        position_learning_rate: 0.0,
        scale_learning_rate: 0.0,
        opacity_learning_rate: 0.0,
        sh_learning_rate: 0.002,
        opacity_loss_weight,
        background: [0.0; 3],
    };
    let sh_degree = selected_sh_degree(view_indices.len());
    Ok((
        appearance_options,
        FitOptions {
            steps: support_steps,
            position_learning_rate,
            scale_learning_rate: 0.005,
            opacity_learning_rate: 0.05,
            ..appearance_options
        },
        sh_degree,
    ))
}

fn validate_shared_prefix(
    pbr: &vol::PointCloudModel,
    light_field: &vol::PointCloudModel,
) -> Result<(), String> {
    pbr.validate()?;
    light_field.validate()?;
    if pbr.sh_degree != 0 || light_field.sh_degree != 0 {
        return Err("shared staged Gaussian fitting requires SH-0 input models".to_string());
    }
    if pbr.points.len() > light_field.points.len()
        || light_field.points.get(..pbr.points.len()) != Some(&pbr.points)
        || light_field.sh_coefficients.get(..pbr.sh_coefficients.len())
            != Some(&pbr.sh_coefficients)
    {
        return Err("the PBR Gaussian must be an exact prefix of the light field".to_string());
    }
    let Some(ref pbr_transforms) = pbr.transforms else {
        return Err("the PBR Gaussian requires transforms".to_string());
    };
    let Some(ref light_field_transforms) = light_field.transforms else {
        return Err("the light-field Gaussian requires transforms".to_string());
    };
    if light_field_transforms
        .rotations
        .get(..pbr_transforms.rotations.len())
        != Some(&pbr_transforms.rotations)
        || light_field_transforms
            .scales
            .get(..pbr_transforms.scales.len())
            != Some(&pbr_transforms.scales)
    {
        return Err("the PBR Gaussian transforms must prefix the light field".to_string());
    }
    Ok(())
}

fn copy_appearance_prefix(pbr: &mut vol::PointCloudModel, light_field: &vol::PointCloudModel) {
    let coefficient_count = pbr.sh_coefficients.len();
    pbr.sh_coefficients
        .copy_from_slice(&light_field.sh_coefficients[..coefficient_count]);
}

fn promote_to_sh_degree(model: &mut vol::PointCloudModel, degree: usize) {
    assert_eq!(model.sh_degree, 0);
    assert!((1..=2).contains(&degree));
    let components = vol::get_sh_component_count(degree);
    let mut coefficients = vec![0.0; model.points.len() * components * 3];
    for (index, dc) in model.sh_coefficients.chunks_exact(3).enumerate() {
        let base = index * components * 3;
        coefficients[base..base + 3].copy_from_slice(dc);
    }
    model.sh_coefficients = coefficients;
    model.sh_degree = degree;
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
            pbr: None,
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

/// Transfer learned Gaussian support back to the corresponding PBR surfels.
///
/// The direct field learns three one-sigma scales while the relightable
/// surface stores one finite three-sigma radius. Their volume-equivalent
/// scalar preserves the learned support budget without inventing an
/// orientation-dependent PBR footprint.
pub fn update_surface_radii(
    surface: &mut vol::relight::RelightModel,
    gaussian: &vol::PointCloudModel,
) -> Result<(), String> {
    surface.validate()?;
    gaussian.validate()?;
    if surface.surfels.len() != gaussian.points.len() {
        return Err("Gaussian radius feedback requires matching particle counts".to_string());
    }
    let transforms = gaussian
        .transforms
        .as_ref()
        .ok_or_else(|| "Gaussian radius feedback requires learned scales".to_string())?;
    for (surfel, scale) in surface.surfels.iter_mut().zip(&transforms.scales) {
        surfel.radius = 3.0 * (scale.x * scale.y * scale.z).cbrt();
    }
    surface.validate()
}

/// Apply the selected residual of a scalar support refinement to corresponding
/// Gaussian ellipsoids.
///
/// The Gaussian fit remains authoritative for orientation and aspect ratio.
/// This carries only a conservative log-space fraction of the later
/// production-render radius correction into the persisted Gaussian asset.
pub fn apply_surface_radius_feedback(
    gaussian: &mut vol::PointCloudModel,
    surface: &vol::relight::RelightModel,
) -> Result<(), String> {
    gaussian.validate()?;
    surface.validate()?;
    if gaussian.points.len() != surface.surfels.len() {
        return Err("Gaussian support feedback requires matching particle counts".to_string());
    }
    let transforms = gaussian
        .transforms
        .as_mut()
        .ok_or_else(|| "Gaussian support feedback requires transforms".to_string())?;
    for (scale, surfel) in transforms.scales.iter_mut().zip(&surface.surfels) {
        let gaussian_radius = 3.0 * (scale.x * scale.y * scale.z).cbrt();
        *scale *= (surfel.radius / gaussian_radius).powf(PBR_SUPPORT_FEEDBACK);
    }
    gaussian.validate()
}

/// Attach the final explicit normals and shared materials to corresponding
/// learned Gaussian geometry.
pub fn attach_pbr(
    gaussian: &mut vol::PointCloudModel,
    surface: &vol::relight::RelightModel,
) -> Result<(), String> {
    gaussian.validate()?;
    surface.validate()?;
    if gaussian.points.len() != surface.surfels.len() {
        return Err("PBR attachment requires matching particle counts".to_string());
    }
    let transforms = gaussian
        .transforms
        .as_mut()
        .ok_or_else(|| "PBR attachment requires Gaussian transforms".to_string())?;
    transforms.pbr = Some(vol::PbrAttributes {
        normals: surface
            .surfels
            .iter()
            .map(|surfel| glam::Vec3::from(surfel.normal))
            .collect(),
        material_indices: surface
            .surfels
            .iter()
            .map(|surfel| surfel.material)
            .collect(),
        materials: surface.materials.clone(),
    });
    gaussian.validate()
}

/// Remove learned Gaussian particles whose peak opacity is negligible.
///
/// This is deliberately a final model-compaction pass: topology is fixed while
/// fitting, then every point-indexed Gaussian and PBR attribute is remapped in
/// one place before persistence. Other point-cloud semantics are rejected
/// rather than silently detached from their indices.
pub fn prune_low_opacity(gaussian: &mut vol::PointCloudModel) -> Result<usize, String> {
    gaussian.validate()?;
    if gaussian.adjacency.is_some()
        || gaussian.radii.is_some()
        || gaussian.surface_normals.is_some()
        || gaussian.surface_offsets.is_some()
        || gaussian.surface_detail.is_some()
        || gaussian.surface_color_coefficients.is_some()
        || gaussian.spherical_voronoi.is_some()
    {
        return Err("Gaussian opacity pruning does not accept surface-cloud semantics".to_string());
    }
    let Some(ref transforms) = gaussian.transforms else {
        return Err("Gaussian opacity pruning requires transforms".to_string());
    };
    let keep: Vec<_> = gaussian
        .points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| (point.w >= PBR_MIN_PERSISTED_OPACITY).then_some(index))
        .collect();
    if keep.is_empty() {
        return Err("Gaussian opacity pruning would remove every particle".to_string());
    }
    let removed = gaussian.points.len() - keep.len();
    if removed == 0 {
        return Ok(0);
    }

    let sh_stride = gaussian.sh_component_count() * 3;
    let points = keep.iter().map(|&index| gaussian.points[index]).collect();
    let sh_coefficients = keep
        .iter()
        .flat_map(|&index| {
            gaussian.sh_coefficients[index * sh_stride..(index + 1) * sh_stride]
                .iter()
                .copied()
        })
        .collect();
    let rotations = keep
        .iter()
        .map(|&index| transforms.rotations[index])
        .collect();
    let scales = keep.iter().map(|&index| transforms.scales[index]).collect();
    let pbr = transforms.pbr.as_ref().map(|pbr| vol::PbrAttributes {
        normals: keep.iter().map(|&index| pbr.normals[index]).collect(),
        material_indices: keep
            .iter()
            .map(|&index| pbr.material_indices[index])
            .collect(),
        materials: pbr.materials.clone(),
    });
    gaussian.points = points;
    gaussian.sh_coefficients = sh_coefficients;
    gaussian.transforms = Some(vol::Transforms {
        rotations,
        scales,
        pbr,
    });
    gaussian.validate()?;
    Ok(removed)
}

fn set_diffuse_appearance(
    gaussian: &mut vol::PointCloudModel,
    surface: &vol::relight::RelightModel,
    environment: &vol::relight::Environment,
    encode_srgb: bool,
) {
    let irradiance = environment.diffuse_irradiance();
    gaussian.sh_degree = 0;
    gaussian.sh_coefficients = surface
        .surfels
        .iter()
        .flat_map(|surfel| {
            let material = surface.materials[surfel.material as usize];
            let basis = vol::relight::sh9(glam::Vec3::from(surfel.normal));
            let mut color = [0.0f32; 3];
            for (coefficient, weight) in irradiance.iter().zip(basis) {
                for channel in 0..3 {
                    color[channel] += coefficient[channel] * weight * material.albedo[channel];
                }
            }
            color.map(|value| {
                let value = if encode_srgb {
                    crate::inverse::capture::linear_to_srgb(value)
                } else {
                    value
                };
                (value - 0.5) / SH_C0
            })
        })
        .collect();
}

fn capture_has_masks(capture: &crate::inverse::capture::Capture, view_indices: &[usize]) -> bool {
    view_indices
        .iter()
        .all(|&view_index| capture.views[view_index].mask.is_some())
}

fn multilight_sample(light_count: usize, step: usize) -> (usize, u32) {
    assert!(light_count != 0);
    let period = 2 * light_count;
    let slot = step % period;
    let light = if slot < light_count {
        slot
    } else {
        period - 1 - slot
    };
    (light, (step / light_count) as u32)
}

/// Continue corresponding Gaussian positions under two or more calibrated
/// lights with exactly aligned cameras while keeping support and appearance
/// fixed.
///
/// Each pass predicts a diffuse color from the current PBR surface and one
/// measured environment. The color is scratch supervision: the Gaussian's
/// learned coefficients are restored before returning. Fully masked captures
/// also update explicit diffuse shading normals in the same graph; maskless
/// captures retain their established position-only path. On masked foreground
/// rays, a weak opacity-conditioned
/// target keeps center motion from compensating for frozen transmittance;
/// background and maskless captures retain the ordinary color residual. One
/// optimizer compares masked captures in linear radiance, while maskless or
/// mixed captures retain their display-referred residual for low-radiance
/// coverage. It interleaves paired rays from the lights in forward/reverse
/// order, avoiding an order prior while retaining joint Adam state and
/// compiling the image-formation graph only once. A short normals-only tail
/// subtracts aligned light pairs, cancelling their shared opacity response
/// without moving geometry.
pub fn fit_multilight_geometry(
    gaussian: &mut vol::PointCloudModel,
    surface: &mut vol::relight::RelightModel,
    lights: &[KnownLightCapture<'_>],
    view_indices: &[usize],
    gpu: sync::Arc<gpu::Context>,
) -> Result<Vec<FitStats>, String> {
    gaussian.validate()?;
    surface.validate()?;
    if gaussian.points.len() != surface.surfels.len() {
        return Err("multi-light Gaussian geometry requires matching particle counts".to_string());
    }
    if lights.len() < 2 {
        return Err("multi-light Gaussian geometry requires at least two known lights".to_string());
    }

    let saved_degree = gaussian.sh_degree;
    let saved_appearance = mem::take(&mut gaussian.sh_coefficients);
    let saved_positions = gaussian.points.clone();
    let learn_normals = lights
        .iter()
        .all(|light| capture_has_masks(light.capture, view_indices));
    let result =
        fit_joint_multilight_positions(gaussian, surface, lights, view_indices, learn_normals, gpu);
    gaussian.sh_degree = saved_degree;
    gaussian.sh_coefficients = saved_appearance;
    match result {
        Ok(fit) => {
            if let Some(normals) = fit.normals {
                for (surfel, normal) in surface.surfels.iter_mut().zip(normals) {
                    surfel.normal = normal;
                }
            }
            Ok(vec![fit.stats])
        }
        Err(error) => {
            gaussian.points = saved_positions;
            Err(error)
        }
    }
}

fn fit_joint_multilight_positions(
    gaussian: &mut vol::PointCloudModel,
    surface: &vol::relight::RelightModel,
    lights: &[KnownLightCapture<'_>],
    view_indices: &[usize],
    learn_normals: bool,
    gpu: sync::Arc<gpu::Context>,
) -> Result<MultilightFit, String> {
    let options = FitOptions {
        steps: MULTI_LIGHT_GEOMETRY_STEPS * 2 * lights.len(),
        batch_size: 512,
        candidates_per_pixel: 64,
        candidate_min_alpha: 1.0e-5,
        geometry_sync_every: 20,
        position_learning_rate: MULTI_LIGHT_GEOMETRY_POSITION_LEARNING_RATE,
        scale_learning_rate: 0.0,
        opacity_learning_rate: 0.0,
        sh_learning_rate: 0.0,
        opacity_loss_weight: 0.0,
        background: [0.0; 3],
    };
    set_diffuse_appearance(gaussian, surface, lights[0].environment, true);
    for light in lights {
        validate_fit(gaussian, light.capture, view_indices, options)?;
        if !captures_are_aligned(lights[0].capture, light.capture) {
            return Err(
                "multi-light Gaussian geometry requires captures with aligned cameras".to_string(),
            );
        }
    }
    let encode_srgb = !lights
        .iter()
        .all(|light| capture_has_masks(light.capture, view_indices));
    if learn_normals && encode_srgb {
        return Err(
            "joint multi-light Gaussian normals require masks on every selected view".to_string(),
        );
    }
    set_diffuse_appearance(gaussian, surface, lights[0].environment, encode_srgb);

    let appearances = if learn_normals {
        Vec::new()
    } else {
        let mut appearances = Vec::with_capacity(lights.len());
        for light in lights {
            set_diffuse_appearance(gaussian, surface, light.environment, encode_srgb);
            appearances.push(sh_parameters(gaussian));
        }
        set_diffuse_appearance(gaussian, surface, lights[0].environment, encode_srgb);
        appearances
    };
    let normal_albedo = learn_normals.then(|| diffuse_albedo(surface));
    let normal_irradiance = learn_normals.then(|| {
        lights
            .iter()
            .map(|light| diffuse_irradiance(light.environment))
            .collect::<Vec<_>>()
    });
    let normal_contrasts = normal_irradiance.as_ref().map(|irradiance| {
        irradiance
            .iter()
            .enumerate()
            .map(|(index, values)| {
                let reference = &irradiance[(index + 1) % irradiance.len()];
                values
                    .iter()
                    .zip(reference)
                    .map(|(value, reference)| value - reference)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });

    let mut graph = mn::Graph::new();
    build_graph_with_losses(
        &mut graph,
        gaussian.points.len(),
        options.batch_size,
        options.candidates_per_pixel,
        gaussian.sh_degree,
        0.0,
        options.background,
        OpacityLoss::Mask,
        MULTI_LIGHT_OPACITY_CONDITIONED_COLOR_CEILING,
        learn_normals,
    );
    let (mut session, _report) = mn::build(
        &graph,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );
    let pixel_indices: Vec<u32> = (0..options.batch_size as u32)
        .flat_map(|pixel| std::iter::repeat_n(pixel, options.candidates_per_pixel))
        .collect();
    session.set_input_u32("candidate_pixel_indices", &pixel_indices);
    if learn_normals {
        let [positions, log_scales, opacity_logits, _, _, _] = model_parameters(gaussian);
        session.set_parameter("positions", &positions);
        session.set_parameter("log_scales", &log_scales);
        session.set_parameter("opacity_logits", &opacity_logits);
        set_surface_normal_parameters(&mut session, surface);
        session.set_input("diffuse_albedo", normal_albedo.as_ref().unwrap());
        session.set_input(
            "diffuse_irradiance",
            &normal_irradiance.as_ref().unwrap()[0],
        );
    } else {
        set_model_parameters(&mut session, gaussian);
    }
    set_rotation_inputs(&mut session, gaussian);

    let pixel_rays = capture_pixel_rays(lights[0].capture);
    let mut candidate_index = CandidateIndex::new(
        gaussian,
        lights[0].capture,
        view_indices,
        options.candidate_min_alpha,
    );
    let audit = sample_rays_with_color_space(
        lights[0].capture,
        &pixel_rays,
        view_indices,
        options.batch_size,
        u32::MAX,
        encode_srgb,
    );
    set_batch(&mut session, gaussian, &audit, &candidate_index, options);
    session.step();
    session.wait();
    let initial_loss = read_loss(&session);

    session.set_adam(1.0, 0.9, 0.999, 1.0e-8);
    session.set_lr_multiplier("positions", options.position_learning_rate);
    session.set_lr_multiplier("log_scales", 0.0);
    session.set_lr_multiplier("opacity_logits", 0.0);
    session.set_lr_multiplier("sh_", 0.0);
    if learn_normals {
        session.set_lr_multiplier("surface_normals", MULTI_LIGHT_NORMAL_LEARNING_RATE);
    }
    for step in 0..options.steps {
        if step != 0 && step.is_multiple_of(options.geometry_sync_every) {
            download_candidate_geometry(&session, gaussian);
            candidate_index = CandidateIndex::new(
                gaussian,
                lights[0].capture,
                view_indices,
                options.candidate_min_alpha,
            );
        }
        let (light_index, sequence) = multilight_sample(lights.len(), step);
        if learn_normals {
            session.set_input(
                "diffuse_irradiance",
                &normal_irradiance.as_ref().unwrap()[light_index],
            );
        } else {
            set_sh_parameters(&mut session, &appearances[light_index]);
        }
        let batch = sample_rays_with_color_space(
            lights[light_index].capture,
            &pixel_rays,
            view_indices,
            options.batch_size,
            sequence,
            encode_srgb,
        );
        set_batch(&mut session, gaussian, &batch, &candidate_index, options);
        session.step();
        session.wait();
    }
    let contrast_steps = if learn_normals {
        MULTI_LIGHT_NORMAL_CONTRAST_STEPS * lights.len()
    } else {
        0
    };
    if learn_normals {
        // With aligned cameras, subtracting two rendered diffuse fields also
        // subtracts their labels while leaving the same Gaussian weights.
        // Freeze centers so this tail can only debias shading normals.
        session.set_lr_multiplier("positions", 0.0);
        for step in 0..contrast_steps {
            let (light_index, sequence) = multilight_sample(lights.len(), step);
            let reference_index = (light_index + 1) % lights.len();
            session.set_input(
                "diffuse_irradiance",
                &normal_contrasts.as_ref().unwrap()[light_index],
            );
            let mut batch = sample_rays_with_color_space(
                lights[light_index].capture,
                &pixel_rays,
                view_indices,
                options.batch_size,
                sequence + 2 * MULTI_LIGHT_GEOMETRY_STEPS as u32,
                false,
            );
            let reference = sample_rays_with_color_space(
                lights[reference_index].capture,
                &pixel_rays,
                view_indices,
                options.batch_size,
                sequence + 2 * MULTI_LIGHT_GEOMETRY_STEPS as u32,
                false,
            );
            debug_assert_eq!(batch.pixels, reference.pixels);
            debug_assert_eq!(batch.alpha, reference.alpha);
            for (value, reference) in batch.labels.iter_mut().zip(&reference.labels) {
                *value -= reference;
            }
            set_batch(&mut session, gaussian, &batch, &candidate_index, options);
            session.step();
            session.wait();
        }
    }
    let normals = if learn_normals {
        let [positions, log_scales, opacity_logits, normals] = session
            .read_params(&[
                "positions",
                "log_scales",
                "opacity_logits",
                "surface_normals",
            ])
            .try_into()
            .unwrap();
        apply_model_geometry(gaussian, &positions, &log_scales, &opacity_logits);
        Some(
            normals
                .chunks_exact(3)
                .zip(&surface.surfels)
                .map(|(normal, surfel)| {
                    glam::Vec3::from_slice(normal)
                        .try_normalize()
                        .unwrap_or_else(|| glam::Vec3::from(surfel.normal))
                        .to_array()
                })
                .collect(),
        )
    } else {
        download_model(&session, gaussian);
        None
    };

    session.clear_optimizer();
    let final_index = CandidateIndex::new(
        gaussian,
        lights[0].capture,
        view_indices,
        options.candidate_min_alpha,
    );
    if learn_normals {
        session.set_input(
            "diffuse_irradiance",
            &normal_irradiance.as_ref().unwrap()[0],
        );
    } else {
        set_sh_parameters(&mut session, &appearances[0]);
    }
    set_batch(&mut session, gaussian, &audit, &final_index, options);
    session.step();
    session.wait();
    let final_loss = read_loss(&session);
    if !initial_loss.is_finite() || !final_loss.is_finite() {
        return Err(format!(
            "joint multi-light Gaussian fit produced a non-finite loss ({initial_loss} -> {final_loss})"
        ));
    }
    Ok(MultilightFit {
        stats: FitStats {
            steps: options.steps + contrast_steps,
            initial_loss,
            final_loss,
        },
        normals,
    })
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
    let projection = crate::inverse::capture::PixelProjection::new(camera, width, height);
    project_conic_with(&projection, mean, rotation, scale)
}

fn project_conic_with(
    projection: &crate::inverse::capture::PixelProjection,
    mean: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) -> Option<ProjectedConic> {
    project_sigma_axes(
        projection,
        mean,
        gaussian_sigma_axes(mean, rotation, scale)?,
    )
}

fn gaussian_sigma_axes(
    mean: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) -> Option<[glam::Vec3; 3]> {
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
    Some(axes)
}

fn project_sigma_axes(
    projection: &crate::inverse::capture::PixelProjection,
    mean: glam::Vec3,
    axes: [glam::Vec3; 3],
) -> Option<ProjectedConic> {
    let local_mean = projection.camera_space(mean);
    let local_axes = axes.map(|axis| projection.camera_vector(axis));
    project_camera_sigma_axes(projection, local_mean, local_axes)
}

fn project_camera_sigma_axes(
    projection: &crate::inverse::capture::PixelProjection,
    mean: glam::Vec3,
    axes: [glam::Vec3; 3],
) -> Option<ProjectedConic> {
    let mut projected = [glam::Vec2::ZERO; 7];
    projected[0] = glam::Vec2::from(projection.project_camera_space(mean)?.0);
    for (axis_index, axis) in axes.into_iter().enumerate() {
        projected[1 + axis_index] =
            glam::Vec2::from(projection.project_camera_space(mean + axis)?.0);
        projected[4 + axis_index] =
            glam::Vec2::from(projection.project_camera_space(mean - axis)?.0);
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

impl ProjectionSupport {
    fn new(
        point: glam::Vec4,
        rotation: glam::Quat,
        scale: glam::Vec3,
        min_alpha: f32,
    ) -> Option<Self> {
        if point.w < min_alpha {
            return None;
        }
        let mean = point.truncate();
        let axes = gaussian_sigma_axes(mean, rotation, scale)?;
        let response_threshold = (min_alpha / point.w).clamp(f32::MIN_POSITIVE, 1.0);
        let gaussian_radius = (-2.0 * response_threshold.ln()).sqrt() * TILE_SUPPORT_MARGIN;
        Some(Self {
            mean,
            axes,
            gaussian_radius,
            world_radius: gaussian_radius * scale.max_element(),
        })
    }
}

fn projected_support_bounds(
    projection: &crate::inverse::capture::PixelProjection,
    width: usize,
    height: usize,
    support: &ProjectionSupport,
) -> Option<(glam::Vec2, glam::Vec2)> {
    let local_mean = projection.camera_space(support.mean);
    if local_mean.z + support.world_radius <= 1.0e-6 {
        return None;
    }
    if local_mean.z - support.world_radius <= 1.0e-6 {
        return Some((
            glam::Vec2::ZERO,
            glam::Vec2::new(
                width.saturating_sub(1) as f32,
                height.saturating_sub(1) as f32,
            ),
        ));
    }

    let local_axes = support.axes.map(|axis| projection.camera_vector(axis));
    let conic = project_camera_sigma_axes(projection, local_mean, local_axes)?;
    let conic_extent = glam::Vec2::new(
        (support.gaussian_radius * support.gaussian_radius * conic.covariance.x_axis.x.max(0.0))
            .sqrt(),
        (support.gaussian_radius * support.gaussian_radius * conic.covariance.y_axis.y.max(0.0))
            .sqrt(),
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
        candidate_transform(mean, rotation, scale),
    )
}

fn ray_response_transformed(
    ray_origin: glam::Vec3,
    ray_direction: glam::Vec3,
    transform: CandidateTransform,
) -> Option<RayResponse> {
    let gaussian_origin = transform.world_to_gaussian * (ray_origin - transform.mean);
    ray_response_from_gaussian_origin(gaussian_origin, ray_direction, transform)
}

fn ray_response_from_gaussian_origin(
    gaussian_origin: glam::Vec3,
    ray_direction: glam::Vec3,
    transform: CandidateTransform,
) -> Option<RayResponse> {
    let (depth, distance_squared) =
        ray_distance_squared_from_gaussian_origin(gaussian_origin, ray_direction, transform)?;
    let response = (-0.5 * distance_squared).exp();
    response
        .is_finite()
        .then_some(RayResponse { depth, response })
}

fn ray_distance_squared_from_gaussian_origin(
    gaussian_origin: glam::Vec3,
    ray_direction: glam::Vec3,
    transform: CandidateTransform,
) -> Option<(f32, f32)> {
    let gaussian_direction = transform.world_to_gaussian * ray_direction;
    let direction_squared = gaussian_direction.length_squared();
    if !direction_squared.is_finite() || direction_squared <= 0.0 {
        return None;
    }
    let depth = -gaussian_origin.dot(gaussian_direction) / direction_squared;
    let closest = gaussian_origin + depth * gaussian_direction;
    let distance_squared = closest.length_squared();
    (depth.is_finite() && distance_squared.is_finite()).then_some((depth, distance_squared))
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
    index: &CandidateIndex,
    direction: glam::Vec3,
    indices: &[u32],
    gaussian_origins: &[glam::Vec3],
    hits: &mut Vec<(f32, u32)>,
) {
    hits.clear();
    for &particle in indices {
        let particle = particle as usize;
        let Some((depth, distance_squared)) = ray_distance_squared_from_gaussian_origin(
            gaussian_origins[particle],
            direction,
            index.transforms[particle],
        ) else {
            continue;
        };
        if depth > 0.0 && distance_squared <= index.max_distance_squared[particle] {
            hits.push((depth, particle as u32));
        }
    }
}

fn record_indexed_candidates(
    model: &vol::PointCloudModel,
    batch: &RayBatch,
    index: &CandidateIndex,
    candidates_per_pixel: usize,
    min_alpha: f32,
) -> CandidateRows {
    let pixels = batch.origins.len();
    let entries = pixels * candidates_per_pixel;
    let mut indices = vec![0_u32; entries];
    let mut mask = vec![0.0_f32; entries];
    let worker_count = thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(pixels.div_ceil(64).max(1));
    let chunk_pixels = pixels.div_ceil(worker_count);
    thread::scope(|scope| {
        let mut remaining = CandidateRowSlices {
            indices: &mut indices,
            mask: &mut mask,
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
            scope.spawn(move || {
                record_indexed_candidate_range(
                    model,
                    &batch.origins[begin..end],
                    &batch.directions[begin..end],
                    &batch.views[begin..end],
                    &batch.pixels[begin..end],
                    index,
                    candidates_per_pixel,
                    min_alpha,
                    CandidateRowSlices {
                        indices: worker_indices,
                        mask: worker_mask,
                    },
                );
            });
        }
    });
    CandidateRows { indices, mask }
}

#[allow(clippy::too_many_arguments)]
fn record_indexed_candidate_range(
    model: &vol::PointCloudModel,
    origins: &[glam::Vec3],
    directions: &[glam::Vec3],
    views: &[usize],
    pixels: &[usize],
    index: &CandidateIndex,
    candidates_per_pixel: usize,
    min_alpha: f32,
    output: CandidateRowSlices<'_>,
) {
    let mut hits = Vec::new();
    let mut order: Vec<_> = (0..origins.len()).collect();
    if index.views.iter().any(Option::is_some) {
        order.sort_unstable_by_key(|&local| index.locality_key(views[local], pixels[local]));
    }
    for local_pixel in order {
        let origin = origins[local_pixel];
        let direction = directions[local_pixel];
        let view = views[local_pixel];
        let pixel = pixels[local_pixel];
        match index.candidates(view, pixel) {
            Some((candidates, gaussian_origins)) => collect_indexed_candidate_hits(
                index,
                direction,
                candidates,
                gaussian_origins,
                &mut hits,
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
        for (slot, &(_, particle)) in hits.iter().take(candidates_per_pixel).enumerate() {
            let flat = local_pixel * candidates_per_pixel + slot;
            output.indices[flat] = particle;
            output.mask[flat] = 1.0;
        }
    }
}

/// Render an SH-0, SH-1, or SH-2 Gaussian model with the exact CPU response oracle.
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
        model.sh_degree <= 2,
        "Gaussian CPU oracle supports SH degrees 0 through 2"
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
    render_candidate_rows(
        model,
        ray_origins,
        ray_directions,
        &candidates.indices,
        &candidates.mask,
        candidates.candidates_per_pixel,
        background,
    )
}

fn render_candidate_rows(
    model: &vol::PointCloudModel,
    ray_origins: &[glam::Vec3],
    ray_directions: &[glam::Vec3],
    candidate_indices: &[u32],
    candidate_mask: &[f32],
    candidates_per_pixel: usize,
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
        for slot in 0..candidates_per_pixel {
            let flat = pixel * candidates_per_pixel + slot;
            if candidate_mask[flat] == 0.0 {
                continue;
            }
            let index = candidate_indices[flat] as usize;
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
    if model.sh_degree > 2 || model.transforms.is_none() {
        return Err(
            "direct Gaussian evaluation requires transformed SH-0 through SH-2 Gaussians"
                .to_string(),
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
        let rays =
            crate::inverse::capture::PixelRays::new(&view.camera, capture.width, capture.height);
        let directions: Vec<_> = (0..capture.height)
            .flat_map(|y| (0..capture.width).map(move |x| rays.direction(x, y)))
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
        let rendered = render_candidate_rows(
            model,
            &batch.origins,
            &batch.directions,
            &candidates.indices,
            &candidates.mask,
            candidates_per_pixel,
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
/// and front-to-back compositing are all inside the graph. SH-1 and SH-2 reuse the
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
    build_graph_with_losses(
        g,
        particles,
        pixels,
        candidates_per_pixel,
        sh_degree,
        opacity_loss_weight,
        background,
        OpacityLoss::Mask,
        0.0,
        false,
    )
}

fn build_diffuse_normal_colors(g: &mut mn::Graph, particles: usize) -> [mn::NodeId; 3] {
    let normals = g.parameter("surface_normals", &[particles, 3]);
    let unit_scale = 1.0 / 3.0_f32.sqrt();
    let weight = g.constant(vec![unit_scale; 3], &[3]);
    let unit = g.rms_norm(normals, weight, 1.0e-12);
    let unit = g.reshape(unit, &[particles * 3]);

    let x = g.split_a(unit, particles as u32, 1, 2, 1);
    let yz = g.split_b(unit, particles as u32, 1, 2, 1);
    let y = g.split_a(yz, particles as u32, 1, 1, 1);
    let z = g.split_b(yz, particles as u32, 1, 1, 1);
    let constant = |g: &mut mn::Graph, value: f32| g.constant(vec![value; particles], &[particles]);
    let b0 = constant(g, 0.282_095);
    let c1 = constant(g, 0.488_603);
    let b1 = g.mul(y, c1);
    let b2 = g.mul(z, c1);
    let b3 = g.mul(x, c1);
    let xy = g.mul(x, y);
    let yz = g.mul(y, z);
    let xz = g.mul(x, z);
    let x2 = g.mul(x, x);
    let y2 = g.mul(y, y);
    let z2 = g.mul(z, z);
    let c2 = constant(g, 1.092_548);
    let b4 = g.mul(xy, c2);
    let b5 = g.mul(yz, c2);
    let three = constant(g, 3.0);
    let one = constant(g, 1.0);
    let three_z2 = g.mul(z2, three);
    let negative_one = g.neg(one);
    let b6_inner = g.add(three_z2, negative_one);
    let b6_scale = constant(g, 0.315_392);
    let b6 = g.mul(b6_inner, b6_scale);
    let b7 = g.mul(xz, c2);
    let negative_y2 = g.neg(y2);
    let b8_inner = g.add(x2, negative_y2);
    let b8_scale = constant(g, 0.546_274);
    let b8 = g.mul(b8_inner, b8_scale);
    let basis01 = g.concat(b0, b1, particles as u32, 1, 1, 1);
    let basis23 = g.concat(b2, b3, particles as u32, 1, 1, 1);
    let basis03 = g.concat(basis01, basis23, particles as u32, 2, 2, 1);
    let basis45 = g.concat(b4, b5, particles as u32, 1, 1, 1);
    let basis67 = g.concat(b6, b7, particles as u32, 1, 1, 1);
    let basis47 = g.concat(basis45, basis67, particles as u32, 2, 2, 1);
    let basis07 = g.concat(basis03, basis47, particles as u32, 4, 4, 1);
    let basis = g.concat(basis07, b8, particles as u32, 8, 1, 1);
    let basis = g.reshape(basis, &[particles, 9]);

    let irradiance = g.input("diffuse_irradiance", &[9, 3]);
    let albedo = g.input("diffuse_albedo", &[particles, 3]);
    let lighting = g.matmul(basis, irradiance);
    let color = g.mul(lighting, albedo);
    let color = g.reshape(color, &[particles * 3]);
    let r = g.split_a(color, particles as u32, 1, 2, 1);
    let r = g.reshape(r, &[particles, 1]);
    let gb = g.split_b(color, particles as u32, 1, 2, 1);
    let gb = g.reshape(gb, &[particles, 2]);
    let g_channel = g.split_a(gb, particles as u32, 1, 1, 1);
    let g_channel = g.reshape(g_channel, &[particles, 1]);
    let b = g.split_b(gb, particles as u32, 1, 1, 1);
    let b = g.reshape(b, &[particles, 1]);
    [r, g_channel, b]
}

#[allow(clippy::too_many_arguments)]
fn build_graph_with_losses(
    g: &mut mn::Graph,
    particles: usize,
    pixels: usize,
    candidates_per_pixel: usize,
    sh_degree: usize,
    opacity_loss_weight: f32,
    background: [f32; 3],
    opacity_loss: OpacityLoss,
    opacity_conditioned_color_ceiling: f32,
    diffuse_normals: bool,
) -> GaussianGraph {
    assert!(particles > 0);
    assert!(pixels > 0);
    assert!(candidates_per_pixel > 0);
    assert!(sh_degree <= 2);
    assert!(opacity_loss_weight.is_finite() && opacity_loss_weight >= 0.0);
    assert!(opacity_conditioned_color_ceiling.is_finite());
    assert!((0.0..=1.0).contains(&opacity_conditioned_color_ceiling));
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
    let diffuse_colors = diffuse_normals.then(|| build_diffuse_normal_colors(g, particles));

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
    for (channel_index, (&channel, background)) in sh.iter().zip(background).enumerate() {
        let color = if let Some(ref diffuse_colors) = diffuse_colors {
            g.embedding(candidate_indices, diffuse_colors[channel_index])
        } else {
            let coefficients = g.embedding(candidate_indices, channel);
            let terms = g.mul(coefficients, basis);
            let value = g.sum_inner(terms);
            let color = g.add(value, bias);
            g.relu(color)
        };
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
    let opacity_condition = (opacity_conditioned_color_ceiling > 0.0).then(|| {
        let detached_opacity = g.stop_gradient(accumulated_opacity);
        let conditioned_weight = g.constant(
            vec![opacity_conditioned_color_ceiling; pixels],
            &[pixels, 1],
        );
        let conditioned_weight = g.mul(target_alpha, conditioned_weight);
        let conditioned_weight = g.mul(detached_opacity, conditioned_weight);
        let negative_conditioned_weight = g.neg(conditioned_weight);
        let ones = g.constant(vec![1.0_f32; pixels], &[pixels, 1]);
        let ordinary_weight = g.add(ones, negative_conditioned_weight);
        (detached_opacity, conditioned_weight, ordinary_weight)
    });
    for (&pixel, target) in rendered.iter().zip([label_r, label_g, label_b]) {
        if let Some((detached_opacity, conditioned_weight, ordinary_weight)) = opacity_condition {
            let foreground_target = g.mul(target, detached_opacity);
            let negative_foreground_target = g.neg(foreground_target);
            let foreground_error = g.add(pixel, negative_foreground_target);
            let foreground_error = g.abs(foreground_error);

            let negative_target = g.neg(target);
            let ordinary_error = g.add(pixel, negative_target);
            let ordinary_error = g.abs(ordinary_error);
            let conditioned_error = g.mul(foreground_error, conditioned_weight);
            let ordinary_error = g.mul(ordinary_error, ordinary_weight);
            let error = g.add(conditioned_error, ordinary_error);
            losses.push(g.mean_all(error));
        } else {
            losses.push(g.l1_loss(pixel, target));
        }
    }
    let rg_loss = g.add(losses[0], losses[1]);
    let color_loss = g.add(rg_loss, losses[2]);
    let loss = if opacity_loss_weight == 0.0 {
        color_loss
    } else {
        let opacity_loss = match opacity_loss {
            OpacityLoss::Mask => g.mse_loss(accumulated_opacity, target_alpha),
            OpacityLoss::BackgroundOnly => {
                let ones = g.constant(vec![1.0_f32; pixels], &[pixels, 1]);
                let negative_target_alpha = g.neg(target_alpha);
                let background_weight = g.add(ones, negative_target_alpha);
                let opacity_squared = g.mul(accumulated_opacity, accumulated_opacity);
                let background_error = g.mul(opacity_squared, background_weight);
                g.mean_all(background_error)
            }
        };
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
    if model.sh_degree > 2 {
        return Err(
            "direct Gaussian fitting currently supports SH degrees 0 through 2".to_string(),
        );
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
    let [sh_r, sh_g, sh_b] = sh_parameters(model);
    [positions, log_scales, opacity_logits, sh_r, sh_g, sh_b]
}

fn sh_parameters(model: &vol::PointCloudModel) -> [Vec<f32>; 3] {
    let sh_components = model.sh_component_count();
    let mut sh: [Vec<f32>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(model.points.len() * sh_components));
    for coefficients in model.sh_coefficients.chunks_exact(3) {
        for channel in 0..3 {
            sh[channel].push(coefficients[channel]);
        }
    }
    sh
}

fn set_sh_parameters(session: &mut mn::Session, sh: &[Vec<f32>; 3]) {
    for (name, values) in ["sh_r", "sh_g", "sh_b"].into_iter().zip(sh) {
        session.set_parameter(name, values);
    }
}

fn set_surface_normal_parameters(session: &mut mn::Session, surface: &vol::relight::RelightModel) {
    let values: Vec<f32> = surface
        .surfels
        .iter()
        .flat_map(|surfel| surfel.normal)
        .collect();
    session.set_parameter("surface_normals", &values);
}

fn diffuse_albedo(surface: &vol::relight::RelightModel) -> Vec<f32> {
    surface
        .surfels
        .iter()
        .flat_map(|surfel| surface.materials[surfel.material as usize].albedo)
        .collect()
}

fn diffuse_irradiance(environment: &vol::relight::Environment) -> Vec<f32> {
    environment
        .diffuse_irradiance()
        .into_iter()
        .flat_map(|coefficient| coefficient.into_iter().take(3))
        .collect()
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

fn apply_model_geometry(
    model: &mut vol::PointCloudModel,
    positions: &[f32],
    log_scales: &[f32],
    opacity_logits: &[f32],
) {
    let count = model.points.len();
    assert_eq!(positions.len(), count * 3);
    assert_eq!(log_scales.len(), count * 3);
    assert_eq!(opacity_logits.len(), count);
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
    }
}

fn download_candidate_geometry(session: &mn::Session, model: &mut vol::PointCloudModel) {
    let [positions, log_scales, opacity_logits] = session
        .read_params(&["positions", "log_scales", "opacity_logits"])
        .try_into()
        .unwrap();
    apply_model_geometry(model, &positions, &log_scales, &opacity_logits);
}

fn download_model(session: &mn::Session, model: &mut vol::PointCloudModel) {
    let sh_components = model.sh_component_count();
    let [positions, log_scales, opacity_logits, sh_r, sh_g, sh_b] = session
        .read_params(&[
            "positions",
            "log_scales",
            "opacity_logits",
            "sh_r",
            "sh_g",
            "sh_b",
        ])
        .try_into()
        .unwrap();
    let sh = [sh_r, sh_g, sh_b];
    apply_model_geometry(model, &positions, &log_scales, &opacity_logits);
    for index in 0..model.points.len() {
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

fn capture_pixel_rays(
    capture: &crate::inverse::capture::Capture,
) -> Vec<crate::inverse::capture::PixelRays> {
    capture
        .views
        .iter()
        .map(|view| {
            crate::inverse::capture::PixelRays::new(&view.camera, capture.width, capture.height)
        })
        .collect()
}

fn cameras_are_aligned(first: &vol::CameraParams, second: &vol::CameraParams) -> bool {
    first.cam_position == second.cam_position
        && first.depth == second.depth
        && first.cam_orientation == second.cam_orientation
        && first.fov == second.fov
        && first.principal == second.principal
}

fn captures_are_aligned(
    first: &crate::inverse::capture::Capture,
    second: &crate::inverse::capture::Capture,
) -> bool {
    first.width == second.width
        && first.height == second.height
        && first.views.len() == second.views.len()
        && first
            .views
            .iter()
            .zip(&second.views)
            .all(|(first, second)| cameras_are_aligned(&first.camera, &second.camera))
}

fn sample_rays_with_color_space(
    capture: &crate::inverse::capture::Capture,
    pixel_rays: &[crate::inverse::capture::PixelRays],
    view_indices: &[usize],
    count: usize,
    sequence: u32,
    encode_srgb: bool,
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
        directions.push(pixel_rays[view_index].direction(x, y));
        views.push(view_index);
        pixels.push(pixel);
        labels.extend(view.pixels[pixel].into_iter().map(|value| {
            if encode_srgb {
                crate::inverse::capture::linear_to_srgb(value)
            } else {
                value
            }
        }));
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

fn sample_rays(
    capture: &crate::inverse::capture::Capture,
    pixel_rays: &[crate::inverse::capture::PixelRays],
    view_indices: &[usize],
    count: usize,
    sequence: u32,
) -> RayBatch {
    sample_rays_with_color_space(capture, pixel_rays, view_indices, count, sequence, true)
}

fn prepare_candidates(
    model: &vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    pixel_rays: &[crate::inverse::capture::PixelRays],
    view_indices: &[usize],
    first_step: usize,
    batch_count: usize,
    index: &CandidateIndex,
    options: FitOptions,
) -> CandidateRows {
    let ray_count = batch_count * options.batch_size;
    let mut combined = RayBatch {
        origins: Vec::with_capacity(ray_count),
        directions: Vec::with_capacity(ray_count),
        views: Vec::with_capacity(ray_count),
        pixels: Vec::with_capacity(ray_count),
        labels: Vec::with_capacity(3 * ray_count),
        alpha: Vec::with_capacity(ray_count),
    };
    for step in first_step..first_step + batch_count {
        let batch = sample_rays(
            capture,
            pixel_rays,
            view_indices,
            options.batch_size,
            step as u32,
        );
        combined.origins.extend_from_slice(&batch.origins);
        combined.directions.extend_from_slice(&batch.directions);
        combined.views.extend_from_slice(&batch.views);
        combined.pixels.extend_from_slice(&batch.pixels);
        combined.labels.extend_from_slice(&batch.labels);
        combined.alpha.extend_from_slice(&batch.alpha);
    }
    record_indexed_candidates(
        model,
        &combined,
        index,
        options.candidates_per_pixel,
        options.candidate_min_alpha,
    )
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
    set_batch_inputs(session, model, batch, &candidates.indices, &candidates.mask);
}

fn set_batch_inputs(
    session: &mut mn::Session,
    model: &vol::PointCloudModel,
    batch: &RayBatch,
    candidate_indices: &[u32],
    candidate_mask: &[f32],
) {
    debug_assert_eq!(candidate_indices.len(), candidate_mask.len());
    debug_assert_eq!(candidate_indices.len() % batch.origins.len(), 0);
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
        .flat_map(|&direction| sh_basis(direction).into_iter().take(sh_components))
        .collect();
    session.set_input_u32("candidate_indices", candidate_indices);
    session.set_input("candidate_mask", candidate_mask);
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

fn geometry_synced_by_final_step(options: FitOptions) -> bool {
    changes_candidate_geometry(options) && options.steps.is_multiple_of(options.geometry_sync_every)
}

/// Fit a Gaussian light field directly to posed RGB images.
///
/// This is intentionally the small baseline: fixed particle count, SH-0/1/2, and
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
    fit_with_opacity_loss(
        model,
        capture,
        view_indices,
        options,
        OpacityLoss::Mask,
        gpu,
    )
}

fn fit_with_opacity_loss(
    model: &mut vol::PointCloudModel,
    capture: &crate::inverse::capture::Capture,
    view_indices: &[usize],
    options: FitOptions,
    opacity_loss: OpacityLoss,
    gpu: sync::Arc<gpu::Context>,
) -> Result<FitStats, String> {
    validate_fit(model, capture, view_indices, options)?;
    let opacity_loss_weight = match opacity_loss {
        OpacityLoss::Mask => options.opacity_loss_weight,
        OpacityLoss::BackgroundOnly => options.opacity_loss_weight * BACKGROUND_ONLY_OPACITY_SCALE,
    };
    let mut graph = mn::Graph::new();
    build_graph_with_losses(
        &mut graph,
        model.points.len(),
        options.batch_size,
        options.candidates_per_pixel,
        model.sh_degree,
        opacity_loss_weight,
        options.background,
        opacity_loss,
        0.0,
        false,
    );
    let (mut session, _report) = mn::build(
        &graph,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );
    let pixel_indices: Vec<u32> = (0..options.batch_size as u32)
        .flat_map(|pixel| std::iter::repeat_n(pixel, options.candidates_per_pixel))
        .collect();
    session.set_input_u32("candidate_pixel_indices", &pixel_indices);
    set_model_parameters(&mut session, model);
    set_rotation_inputs(&mut session, model);
    let pixel_rays = capture_pixel_rays(capture);
    let mut candidate_index =
        CandidateIndex::new(model, capture, view_indices, options.candidate_min_alpha);

    let audit = sample_rays(
        capture,
        &pixel_rays,
        view_indices,
        options.batch_size,
        u32::MAX,
    );
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
    let mut step = 0;
    while step < options.steps {
        let until_geometry_sync = if sync_candidate_geometry {
            options.geometry_sync_every - step % options.geometry_sync_every
        } else {
            PREPARED_CANDIDATE_BATCHES
        };
        let batch_count = PREPARED_CANDIDATE_BATCHES
            .min(until_geometry_sync)
            .min(options.steps - step);
        let candidates = prepare_candidates(
            model,
            capture,
            &pixel_rays,
            view_indices,
            step,
            batch_count,
            &candidate_index,
            options,
        );
        let entries = options.batch_size * options.candidates_per_pixel;
        for batch_index in 0..batch_count {
            let batch = sample_rays(
                capture,
                &pixel_rays,
                view_indices,
                options.batch_size,
                (step + batch_index) as u32,
            );
            let start = batch_index * entries;
            let end = start + entries;
            set_batch_inputs(
                &mut session,
                model,
                &batch,
                &candidates.indices[start..end],
                &candidates.mask[start..end],
            );
            session.step();
            session.wait();
        }
        step += batch_count;
        if sync_candidate_geometry && step % options.geometry_sync_every == 0 {
            // Candidate grids read only position, support, and opacity. Keep
            // the much larger SH table device-local until the final model.
            if step == options.steps {
                download_model(&session, model);
            } else {
                download_candidate_geometry(&session, model);
            }
            candidate_index =
                CandidateIndex::new(model, capture, view_indices, options.candidate_min_alpha);
        }
    }
    let geometry_synced = geometry_synced_by_final_step(options);
    if !geometry_synced {
        download_model(&session, model);
    }

    session.clear_optimizer();
    if sync_candidate_geometry && !geometry_synced {
        candidate_index =
            CandidateIndex::new(model, capture, view_indices, options.candidate_min_alpha);
    }
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
                pbr: None,
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
        let mut surface = vol::relight::RelightModel {
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
        let mut model = from_surface(&surface).unwrap();
        assert_eq!(model.points[0], glam::Vec4::new(1.0, 2.0, 3.0, 0.5));
        assert_eq!(model.points[1], glam::Vec4::new(-1.0, 0.0, 4.0, 0.5));
        assert_eq!(model.sh_coefficients, [0.0; 6]);
        let transforms = model.transforms.as_ref().unwrap();
        assert_close(transforms.scales[0].x, 0.2, 1.0e-6);
        assert_close(transforms.scales[1].x, 0.1, 1.0e-6);
        for (rotation, surfel) in transforms.rotations.iter().zip(&surface.surfels) {
            let normal = glam::Vec3::from(surfel.normal);
            assert!((*rotation * glam::Vec3::Y).dot(normal) > 0.9999);
        }
        model.transforms.as_mut().unwrap().scales[0] = glam::Vec3::new(0.2, 0.4, 0.8);
        let original = surface.surfels.clone();
        update_surface_radii(&mut surface, &model).unwrap();
        assert_close(surface.surfels[0].radius, 1.2, 1.0e-6);
        assert_close(surface.surfels[1].radius, 0.3, 1.0e-6);
        for (before, after) in original.iter().zip(&surface.surfels) {
            assert_eq!(before.center, after.center);
            assert_eq!(before.normal, after.normal);
            assert_eq!(before.material, after.material);
        }
        surface.surfels[0].radius *= 0.5;
        let scale_before = model.transforms.as_ref().unwrap().scales[0];
        apply_surface_radius_feedback(&mut model, &surface).unwrap();
        attach_pbr(&mut model, &surface).unwrap();
        let transforms = model.transforms.unwrap();
        let multiplier = 0.5_f32.powf(PBR_SUPPORT_FEEDBACK);
        assert_close(transforms.scales[0].x, multiplier * scale_before.x, 1.0e-6);
        assert_close(transforms.scales[0].y, multiplier * scale_before.y, 1.0e-6);
        assert_close(transforms.scales[0].z, multiplier * scale_before.z, 1.0e-6);
        assert_close(
            transforms.scales[0].y / transforms.scales[0].x,
            scale_before.y / scale_before.x,
            1.0e-6,
        );
        let pbr = transforms.pbr.unwrap();
        assert_eq!(pbr.normals, [glam::Vec3::Z, -glam::Vec3::Y]);
        assert_eq!(pbr.material_indices, [0, 1]);
        assert_eq!(pbr.materials, surface.materials);
    }

    #[test]
    fn opacity_pruning_remaps_every_gaussian_pbr_attribute() {
        let mut gaussian = model(vec![
            glam::Vec4::new(1.0, 0.0, 0.0, 0.5),
            glam::Vec4::new(2.0, 0.0, 0.0, 0.04),
            glam::Vec4::new(3.0, 0.0, 0.0, 0.8),
        ]);
        gaussian.sh_degree = 1;
        gaussian.sh_coefficients = (0..36).map(|value| value as f32).collect();
        let transforms = gaussian.transforms.as_mut().unwrap();
        transforms.rotations = vec![
            glam::Quat::IDENTITY,
            glam::Quat::from_rotation_x(0.5),
            glam::Quat::from_rotation_y(0.5),
        ];
        transforms.scales = vec![
            glam::Vec3::splat(1.0),
            glam::Vec3::splat(2.0),
            glam::Vec3::splat(3.0),
        ];
        transforms.pbr = Some(vol::PbrAttributes {
            normals: vec![glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z],
            material_indices: vec![1, 0, 1],
            materials: vec![
                vol::relight::Material::default(),
                vol::relight::Material {
                    albedo: [0.2, 0.3, 0.4],
                    ..vol::relight::Material::default()
                },
            ],
        });
        gaussian.validate().unwrap();

        assert_eq!(prune_low_opacity(&mut gaussian).unwrap(), 1);

        assert_eq!(gaussian.points.len(), 2);
        assert_eq!(gaussian.points[0].x, 1.0);
        assert_eq!(gaussian.points[1].x, 3.0);
        assert_eq!(
            gaussian.sh_coefficients,
            (0..12)
                .chain(24..36)
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        );
        let transforms = gaussian.transforms.unwrap();
        assert_eq!(
            transforms.scales,
            [glam::Vec3::splat(1.0), glam::Vec3::splat(3.0)],
        );
        assert_eq!(
            transforms.rotations,
            [glam::Quat::IDENTITY, glam::Quat::from_rotation_y(0.5)],
        );
        let pbr = transforms.pbr.unwrap();
        assert_eq!(pbr.normals, [glam::Vec3::X, glam::Vec3::Z]);
        assert_eq!(pbr.material_indices, [1, 1]);
        assert_eq!(pbr.materials.len(), 2);
    }

    #[test]
    fn selected_fit_defaults_keep_established_centres_fixed() {
        assert_eq!(FitOptions::default().position_learning_rate, 0.0);
    }

    #[test]
    fn multilight_geometry_uses_paired_diffuse_scratch_appearance() {
        let surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![vol::relight::Surfel {
                center: [0.0; 3],
                radius: 0.3,
                normal: glam::Vec3::Z.to_array(),
                material: 0,
            }],
            materials: vec![vol::relight::Material {
                albedo: [0.5, 0.25, 1.0],
                ..vol::relight::Material::default()
            }],
        };
        let mut gaussian = from_surface(&surface).unwrap();
        let environment = vol::relight::Environment::uniform([0.4, 0.8, 0.2], 64, 32);
        set_diffuse_appearance(&mut gaussian, &surface, &environment, false);
        for (actual, expected) in gaussian.sh_coefficients.iter().zip([0.2; 3]) {
            assert_close(0.5 + SH_C0 * actual, expected, 5.0e-3);
        }
        set_diffuse_appearance(&mut gaussian, &surface, &environment, true);
        let expected = [0.2, 0.2, 0.2].map(crate::inverse::capture::linear_to_srgb);
        for (actual, expected) in gaussian.sh_coefficients.iter().zip(expected) {
            assert_close(0.5 + SH_C0 * actual, expected, 5.0e-3);
        }
        let schedule: Vec<_> = (0..12).map(|step| multilight_sample(3, step)).collect();
        assert_eq!(
            schedule,
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (2, 1),
                (1, 1),
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (2, 3),
                (1, 3),
                (0, 3),
            ]
        );
    }

    #[test]
    fn multilight_capture_alignment_compares_image_geometry() {
        let truth = model(vec![glam::Vec4::new(0.0, 0.0, 4.0, 0.8)]);
        let first = synthetic_capture(&truth);
        let mut second = synthetic_capture(&truth);
        assert!(captures_are_aligned(&first, &second));
        assert!(!capture_has_masks(&second, &[0, 1]));
        for view in &mut second.views {
            view.mask = Some(vec![1.0; second.width * second.height]);
        }
        assert!(capture_has_masks(&second, &[0, 1]));

        second.views[1].camera.principal[0] += 0.01;
        assert!(!captures_are_aligned(&first, &second));
    }

    #[test]
    fn shared_appearance_requires_and_updates_only_the_pbr_prefix() {
        let first = glam::Vec4::new(1.0, 2.0, 3.0, 0.5);
        let second = glam::Vec4::new(4.0, 5.0, 6.0, 0.5);
        let mut pbr = model(vec![first]);
        let mut light_field = model(vec![first, second]);
        validate_shared_prefix(&pbr, &light_field).unwrap();

        light_field
            .sh_coefficients
            .copy_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        copy_appearance_prefix(&mut pbr, &light_field);
        assert_eq!(pbr.sh_coefficients, [0.1, 0.2, 0.3]);
        assert_eq!(light_field.sh_coefficients[3..], [0.4, 0.5, 0.6]);

        light_field.points[0].x += 1.0;
        assert!(validate_shared_prefix(&pbr, &light_field).is_err());
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
    fn candidate_geometry_readback_preserves_appearance_and_rotations() {
        let mut model = model(vec![glam::Vec4::new(0.0, 0.0, 1.0, 0.2)]);
        model.sh_coefficients = vec![0.1, 0.2, 0.3];
        model.transforms.as_mut().unwrap().rotations[0] = glam::Quat::from_rotation_x(0.2);
        let appearance = model.sh_coefficients.clone();
        let rotation = model.transforms.as_ref().unwrap().rotations[0];
        let scale = glam::Vec3::new(0.2, 0.3, 0.4);
        let log_scales = scale
            .to_array()
            .map(|value| inverse_softplus(value - MIN_SCALE));

        apply_model_geometry(&mut model, &[1.0, 2.0, 3.0], &log_scales, &[1.0]);

        assert_eq!(model.points[0].truncate(), glam::Vec3::new(1.0, 2.0, 3.0));
        assert_close(
            model.points[0].w,
            MAX_ALPHA / (1.0 + (-1.0_f32).exp()),
            1.0e-6,
        );
        let transform = model.transforms.as_ref().unwrap();
        assert!((transform.scales[0] - scale).abs().max_element() < 1.0e-6);
        assert_eq!(transform.rotations[0], rotation);
        assert_eq!(model.sh_coefficients, appearance);
    }

    #[test]
    fn final_geometry_sync_is_reused_only_at_an_exact_refresh_boundary() {
        let support = FitOptions {
            steps: 100,
            geometry_sync_every: 20,
            scale_learning_rate: 0.01,
            ..FitOptions::default()
        };
        assert!(geometry_synced_by_final_step(support));
        assert!(!geometry_synced_by_final_step(FitOptions {
            steps: 99,
            ..support
        }));
        assert!(!geometry_synced_by_final_step(FitOptions {
            position_learning_rate: 0.0,
            scale_learning_rate: 0.0,
            opacity_learning_rate: 0.0,
            ..support
        }));
    }

    #[test]
    fn staged_budget_prioritizes_support_and_keeps_both_stages_nonempty() {
        assert_eq!(staged_step_counts(1), None);
        assert_eq!(staged_step_counts(2), Some((1, 1)));
        assert_eq!(staged_step_counts(1_500), Some((500, 1_000)));
        assert_eq!(selected_sh_degree(MIN_SH1_VIEWS - 1), 0);
        assert_eq!(selected_sh_degree(MIN_SH1_VIEWS), 1);
        assert_eq!(selected_sh_degree(MIN_SH2_VIEWS - 1), 1);
        assert_eq!(selected_sh_degree(MIN_SH2_VIEWS), 2);
    }

    #[test]
    fn static_continuation_requires_a_validation_margin() {
        assert!(!static_continuation_wins(24.0, 24.049));
        assert!(static_continuation_wins(24.0, 24.05));
        assert!(!static_continuation_wins(24.0, 23.9));
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
    fn joint_multilight_fit_restores_durable_appearance() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping joint multi-light Gaussian fit test: no GPU");
            return;
        };
        let mut surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![vol::relight::Surfel {
                center: [0.0, 0.0, 4.0],
                radius: 0.5,
                normal: glam::Vec3::Z.to_array(),
                material: 0,
            }],
            materials: vec![vol::relight::Material {
                albedo: [0.7, 0.5, 0.3],
                ..vol::relight::Material::default()
            }],
        };
        let environment = vol::relight::Environment::uniform([0.8, 0.6, 0.4], 64, 32);
        let mut truth = from_surface(&surface).unwrap();
        set_diffuse_appearance(&mut truth, &surface, &environment, true);
        let mut capture = synthetic_capture(&truth);
        for view in &mut capture.views {
            view.mask = Some(
                view.pixels
                    .iter()
                    .map(|pixel| {
                        if pixel.iter().any(|value| *value > 0.0) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
        }
        let mut candidate = truth.clone();
        candidate.sh_coefficients.fill(-0.25);
        let durable_appearance = candidate.sh_coefficients.clone();
        let lights = [
            KnownLightCapture {
                capture: &capture,
                environment: &environment,
            },
            KnownLightCapture {
                capture: &capture,
                environment: &environment,
            },
        ];

        let stats =
            fit_multilight_geometry(&mut candidate, &mut surface, &lights, &[0, 1], gpu).unwrap();

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].steps, 250);
        assert!(stats[0].initial_loss.is_finite());
        assert!(stats[0].final_loss.is_finite());
        assert_eq!(candidate.sh_degree, 0);
        assert_eq!(candidate.sh_coefficients, durable_appearance);
        candidate.validate().unwrap();
    }

    #[test]
    fn diffuse_normal_graph_has_a_finite_gradient() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping diffuse-normal graph test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let colors = build_diffuse_normal_colors(&mut graph, 1);
        let loss = graph.mean_all(colors[0]);
        graph.set_outputs(vec![loss, colors[0], colors[1], colors[2]]);
        let (mut session, _report) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_parameter("surface_normals", &[0.0, 0.0, 1.0]);
        session.set_input("diffuse_albedo", &[0.7, 0.5, 0.3]);
        session.set_input("diffuse_irradiance", &[0.1; 27]);
        session.set_adam(1.0, 0.9, 0.999, 1.0e-8);
        session.set_lr_multiplier("surface_normals", 1.0e-3);
        session.step();
        session.wait();
        assert!(session.read_output(1)[0].is_finite());
        let irradiance = 0.1 * vol::relight::sh9(glam::Vec3::Z).into_iter().sum::<f32>();
        for (output, albedo) in (1..=3).zip([0.7, 0.5, 0.3]) {
            let mut actual = [0.0];
            session.read_output_by_index(output, &mut actual);
            assert_close(actual[0], irradiance * albedo, 1.0e-5);
        }
        let updated = session.read_params(&["surface_normals"]);
        assert_ne!(updated[0], [0.0, 0.0, 1.0]);
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
        promote_to_sh_degree(&mut model, 1);
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
    fn degree_two_promotion_preserves_dc_and_zeros_directional_terms() {
        let mut model = model(vec![glam::Vec4::W, glam::Vec4::ONE]);
        model.sh_coefficients = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        promote_to_sh_degree(&mut model, 2);
        assert_eq!(model.sh_degree, 2);
        assert_eq!(model.sh_coefficients.len(), 54);
        assert_eq!(&model.sh_coefficients[..3], &[1.0, 2.0, 3.0]);
        assert!(model.sh_coefficients[3..27]
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(&model.sh_coefficients[27..30], &[4.0, 5.0, 6.0]);
        assert!(model.sh_coefficients[30..]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn degree_two_basis_matches_the_point_cloud_oracle() {
        let mut model = model(vec![glam::Vec4::W]);
        model.sh_degree = 2;
        model.sh_coefficients = (0..27).map(|index| 0.01 * (index as f32 - 13.0)).collect();
        let direction = glam::Vec3::new(0.3, -0.4, 0.5).normalize();
        let basis = sh_basis(direction);
        let mut expected = glam::Vec3::splat(0.5);
        for (component, value) in basis.into_iter().enumerate() {
            for channel in 0..3 {
                expected[channel] += value * model.sh_coefficients[component * 3 + channel];
            }
        }
        let actual = vol::trace::eval_rgb_sh(&model, 0, direction);
        for (actual, expected) in actual.to_array().into_iter().zip(expected.to_array()) {
            assert_close(actual, expected.max(0.0), 1.0e-6);
        }
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
    fn candidate_distance_cutoff_matches_the_alpha_threshold() {
        for (opacity, min_alpha) in [(0.8, 1.0e-4), (0.2, 0.05), (0.05, 0.05)] {
            let cutoff = candidate_max_distance_squared(opacity, min_alpha);
            for distance_squared in [0.0, (cutoff - 1.0e-3).max(0.0), cutoff + 1.0e-3] {
                let alpha = opacity * (-0.5 * distance_squared).exp();
                assert_eq!(distance_squared <= cutoff, alpha >= min_alpha);
            }
        }
        assert_eq!(candidate_max_distance_squared(0.0, 0.0), f32::INFINITY);
        assert!(candidate_max_distance_squared(0.01, 0.02) < 0.0);
    }

    #[test]
    fn cached_candidate_transform_matches_quaternion_response() {
        let origin = glam::Vec3::new(0.7, -0.2, -3.0);
        let direction = glam::Vec3::new(0.1, 0.05, 1.0).normalize();
        let mean = glam::Vec3::new(0.2, 0.4, 2.0);
        let rotation = glam::Quat::from_rotation_y(0.4);
        let scale = glam::Vec3::new(0.3, 0.8, 1.1);
        let direct = ray_response(origin, direction, mean, rotation, scale).unwrap();
        let cached = ray_response_transformed(
            origin,
            direction,
            candidate_transform(mean, rotation, scale),
        )
        .unwrap();

        assert_eq!(cached.depth.to_bits(), direct.depth.to_bits());
        assert_eq!(cached.response.to_bits(), direct.response.to_bits());
        let inverse_rotation = rotation.normalize().inverse();
        let gaussian_origin = (inverse_rotation * (origin - mean)) / scale;
        let gaussian_direction = (inverse_rotation * direction) / scale;
        let expected_depth =
            -gaussian_origin.dot(gaussian_direction) / gaussian_direction.length_squared();
        let closest = gaussian_origin + expected_depth * gaussian_direction;
        let expected_response = (-0.5 * closest.length_squared()).exp();
        assert_close(cached.depth, expected_depth, 1.0e-5);
        assert_close(cached.response, expected_response, 1.0e-6);
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
        let pixel_rays = capture_pixel_rays(&capture);
        let batch = sample_rays(&capture, &pixel_rays, &[0, 1], 1_024, 17);
        let exhaustive = record_candidates(&model, &batch.origins, &batch.directions, 8, 1.0e-4);
        let index = CandidateIndex::new(&model, &capture, &[0, 1], 1.0e-4);
        let tiled = record_indexed_candidates(&model, &batch, &index, 8, 1.0e-4);
        assert_eq!(tiled.indices, exhaustive.indices);
        assert_eq!(tiled.mask, exhaustive.mask);

        let candidate_entries: usize = batch
            .views
            .iter()
            .zip(&batch.pixels)
            .map(|(&view, &pixel)| index.candidates(view, pixel).unwrap().0.len())
            .sum();
        let exhaustive_entries = batch.origins.len() * model.points.len();
        assert!(candidate_entries < exhaustive_entries / 2);
    }

    #[test]
    fn prepared_candidate_batches_match_individual_recording() {
        let model = model(vec![
            glam::Vec4::new(-0.3, 0.0, 3.0, 0.8),
            glam::Vec4::new(0.2, 0.1, 3.5, 0.7),
            glam::Vec4::new(0.0, -0.4, 4.0, 0.9),
        ]);
        let capture = empty_capture(32, 24);
        let pixel_rays = capture_pixel_rays(&capture);
        let view_indices = [0, 1];
        let options = FitOptions {
            batch_size: 73,
            candidates_per_pixel: 3,
            candidate_min_alpha: 1.0e-4,
            ..FitOptions::default()
        };
        let index =
            CandidateIndex::new(&model, &capture, &view_indices, options.candidate_min_alpha);
        let combined_candidates = prepare_candidates(
            &model,
            &capture,
            &pixel_rays,
            &view_indices,
            11,
            3,
            &index,
            options,
        );
        let entries = options.batch_size * options.candidates_per_pixel;
        for offset in 0..3 {
            let rays = sample_rays(
                &capture,
                &pixel_rays,
                &view_indices,
                options.batch_size,
                11 + offset as u32,
            );
            let individual = record_indexed_candidates(
                &model,
                &rays,
                &index,
                options.candidates_per_pixel,
                options.candidate_min_alpha,
            );
            let start = offset * entries;
            let end = start + entries;
            assert_eq!(&combined_candidates.indices[start..end], individual.indices);
            assert_eq!(&combined_candidates.mask[start..end], individual.mask);
        }
    }

    #[test]
    fn anisotropic_sh2_graph_matches_oracle_and_has_position_gradient() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping anisotropic Gaussian graph test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        build_graph_with_losses(
            &mut graph,
            2,
            1,
            2,
            2,
            1.5,
            [0.0; 3],
            OpacityLoss::BackgroundOnly,
            0.0,
            false,
        );
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
        let mut sh_r = [0.0; 18];
        sh_r[0] = bright;
        sh_r[2] = directional;
        sh_r[9] = -bright;
        session.set_parameter("sh_r", &sh_r);
        let mut sh_g = [0.0; 18];
        sh_g[0] = -bright;
        sh_g[9] = bright;
        sh_g[11] = -directional;
        session.set_parameter("sh_g", &sh_g);
        let mut sh_b = [0.0; 18];
        sh_b[0] = -bright;
        sh_b[9] = -bright;
        session.set_parameter("sh_b", &sh_b);
        session.set_input_u32("candidate_indices", &[0, 1]);
        session.set_input_u32("candidate_pixel_indices", &[0, 0]);
        session.set_input("candidate_mask", &[1.0, 1.0]);
        session.set_input("ray_origins", &[0.0, 0.0, 0.0]);
        session.set_input("ray_directions", &[0.0, 0.0, 1.0]);
        session.set_input(
            "sh_basis",
            &[SH_C0, 0.0, SH_C1, 0.0, 0.0, 0.0, 2.0 * SH_C2[2], 0.0, 0.0],
        );
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
        let mut sh_gradient = [0.0_f32; 18];
        session.read_param_grad("sh_r", &mut sh_gradient);
        assert!(sh_gradient.iter().all(|value| value.is_finite()));
        assert!(sh_gradient[2].abs() > 1.0e-5);
        assert!(sh_gradient[6].abs() > 1.0e-5);

        let mut background_loss = [0.0_f32];
        let mut opacity = [0.0_f32];
        session.read_output_by_index(0, &mut background_loss);
        session.read_output_by_index(4, &mut opacity);
        session.set_input("target_alpha", &[1.0]);
        session.step();
        session.wait();
        let mut foreground_loss = [0.0_f32];
        session.read_output_by_index(0, &mut foreground_loss);
        assert_close(
            background_loss[0] - foreground_loss[0],
            1.5 * opacity[0] * opacity[0],
            2.0e-5,
        );
    }

    #[test]
    fn opacity_conditioned_color_loss_matches_formula() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping opacity-conditioned Gaussian loss test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        build_graph_with_losses(
            &mut graph,
            1,
            1,
            1,
            0,
            0.0,
            [0.0; 3],
            OpacityLoss::Mask,
            MULTI_LIGHT_OPACITY_CONDITIONED_COLOR_CEILING,
            false,
        );
        let (mut session, _report) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_parameter("positions", &[0.0, 0.0, 2.0]);
        let scale = inverse_softplus(1.0 - MIN_SCALE);
        session.set_parameter("log_scales", &[scale; 3]);
        session.set_parameter("opacity_logits", &[0.0]);
        let coefficient = (0.8 - 0.5) / SH_C0;
        for name in ["sh_r", "sh_g", "sh_b"] {
            session.set_parameter(name, &[coefficient]);
        }
        session.set_input_u32("candidate_indices", &[0]);
        session.set_input_u32("candidate_pixel_indices", &[0]);
        session.set_input("candidate_mask", &[1.0]);
        session.set_input("ray_origins", &[0.0, 0.0, 0.0]);
        session.set_input("ray_directions", &[0.0, 0.0, 1.0]);
        session.set_input("sh_basis", &[SH_C0]);
        session.set_input("labels", &[0.6; 3]);
        for (name, axis) in [
            ("rotation_x", glam::Vec3::X),
            ("rotation_y", glam::Vec3::Y),
            ("rotation_z", glam::Vec3::Z),
        ] {
            session.set_input(name, &axis.to_array());
        }

        let alpha = 0.5 * MAX_ALPHA;
        let ordinary = (alpha * 0.8 - 0.6).abs();
        let conditioned = alpha * (0.8 - 0.6);
        let conditioned_weight = MULTI_LIGHT_OPACITY_CONDITIONED_COLOR_CEILING * alpha;
        let expected_foreground =
            3.0 * (conditioned_weight * conditioned + (1.0 - conditioned_weight) * ordinary);
        session.set_input("target_alpha", &[1.0]);
        session.step();
        session.wait();
        let mut loss = [0.0];
        session.read_output_by_index(0, &mut loss);
        assert_close(loss[0], expected_foreground, 2.0e-5);

        session.set_input("target_alpha", &[0.0]);
        session.step();
        session.wait();
        session.read_output_by_index(0, &mut loss);
        assert_close(loss[0], 3.0 * ordinary, 2.0e-5);
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
