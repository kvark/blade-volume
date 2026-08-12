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
    pub tested: usize,
    pub moved: usize,
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

/// Move particles along their current normals against the complete runtime
/// PBR render of every training photograph.
///
/// Materials and illumination stay fixed during the pass. Each coordinate is
/// tested independently in both directions, and a move is retained only when
/// it reduces the joint image loss. The held-out cameras never enter this
/// function; callers decide which indices are training evidence.
pub fn refine_rendered(
    scene: &mut score::Scene,
    capture: &capture::Capture,
    indices: &[usize],
    observations: &decompose::Observations,
    diffuse_samples: u32,
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
    if scene.model.surfels.is_empty() || indices.is_empty() || max_particles == 0 {
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
    let candidates: Vec<usize> = (0..scene.model.surfels.len())
        .filter(|&index| !observations.of(index).is_empty())
        .take(max_particles)
        .collect();
    let tested = candidates.len();
    let mut moved = 0usize;
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
            renderer.update_prepared_surfels(&mut tracer, &scene.model.surfels);
            let candidate = rendered_loss(&mut renderer, &mut tracer, capture, indices, &cameras);
            if candidate < best_loss {
                best_loss = candidate;
                best = scene.model.surfels[index].center;
            }
        }
        scene.model.surfels[index].center = best;
        if best != plus {
            renderer.update_prepared_surfels(&mut tracer, &scene.model.surfels);
        }
        if best != original {
            moved += 1;
            loss = best_loss;
        }
    }
    renderer.destroy_prepared_scene(tracer);
    renderer.destroy();
    Ok(RenderedStats {
        particles: scene.model.surfels.len(),
        tested,
        moved,
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
    fn runtime_render_loss_recovers_a_known_surface_offset() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        const SIZE: usize = 32;
        let cameras = [
            camera(glam::Vec3::new(0.0, 0.0, -1.2)),
            camera(glam::Vec3::new(0.25, 0.0, -1.2)),
        ];
        let make_scene = |offset: f32| score::Scene {
            model: vol::relight::RelightModel {
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
            environment: vol::relight::Environment {
                width: 8,
                height: 4,
                texels: vec![[1.0; 3]; 32],
            },
        };

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
        truth_renderer.destroy();
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
                })
                .collect(),
        };

        let mut displaced = make_scene(0.075);
        let observations = decompose::observe(&displaced.model, &capture, &[0, 1], 0.1);
        assert_eq!(observations.seen(), 1);
        let stats =
            refine_rendered(&mut displaced, &capture, &[0, 1], &observations, 0, 1).unwrap();
        assert_eq!(stats.tested, 1);
        assert_eq!(stats.moved, 1);
        assert!(stats.final_loss < stats.initial_loss);
        assert!(displaced.model.surfels[0].center[2].abs() < 1.0e-6);
    }
}
