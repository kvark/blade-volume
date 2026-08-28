//! Sparse cross-view tracks for foreground the current cloud does not cover.
//!
//! A current Gaussian may provide a depth interval, but it cannot decide that
//! two missing pixels are the same surface point. This module starts from the
//! independent image observations: it matches compact response patches along
//! calibrated epipolar lines, requires a mutual match, and triangulates only
//! camera-ray groups that agree in several views.

use crate::inverse::capture;
use std::collections;

const DESCRIPTOR_SIDE: usize = 3;
const DESCRIPTOR_VALUES: usize = DESCRIPTOR_SIDE * DESCRIPTOR_SIDE * 3;

/// Finite world-space search region for a track.
#[derive(Clone, Copy, Debug)]
pub struct WorldBounds {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

impl WorldBounds {
    pub fn from_points(points: &[glam::Vec3], padding: f32) -> Option<Self> {
        if points.is_empty() || !padding.is_finite() || padding < 0.0 {
            return None;
        }
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        for &point in points {
            if !point.is_finite() {
                return None;
            }
            min = min.min(point);
            max = max.max(point);
        }
        let extent = max - min;
        let margin = padding * extent.max_element().max(1.0e-6);
        Some(Self {
            min: min - margin,
            max: max + margin,
        })
    }

    fn is_valid(self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && (self.max - self.min).cmpgt(glam::Vec3::ZERO).all()
    }

    fn contains(self, point: glam::Vec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }
}

/// One selected camera and its independently measured missing-foreground map.
pub struct TrackView<'a> {
    pub capture_index: usize,
    /// Row-major, `capture.width * capture.height`. A true entry is eligible
    /// to become one observation of a new track.
    pub missing: &'a [bool],
}

#[derive(Clone, Copy, Debug)]
pub struct TrackOptions {
    /// Only every Nth missing pixel in each direction starts a track. Target
    /// observations are still searched at full resolution.
    pub anchor_stride: usize,
    /// Cameras considered for one anchor, including the anchor camera.
    pub max_views: usize,
    pub min_views: usize,
    /// Radius around the rasterized epipolar line, in pixels.
    pub epipolar_radius: usize,
    /// Minimum normalized patch RMS.
    pub min_texture: f32,
    /// Maximum mean squared distance between normalized response patches.
    pub max_descriptor_error: f32,
    /// Best/second-best descriptor-cost ratio. One disables the ratio check.
    pub match_ratio: f32,
    pub min_parallax_degrees: f32,
    pub max_reprojection_error: f32,
}

impl Default for TrackOptions {
    fn default() -> Self {
        Self {
            anchor_stride: 2,
            max_views: 12,
            min_views: 3,
            epipolar_radius: 1,
            min_texture: 0.02,
            max_descriptor_error: 0.25,
            match_ratio: 0.8,
            min_parallax_degrees: 1.0,
            max_reprojection_error: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub view: usize,
    pub pixel: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub position: glam::Vec3,
    pub observations: Vec<Observation>,
    pub mean_reprojection_error: f32,
    pub mean_descriptor_error: f32,
    pub parallax_degrees: f32,
}

/// Counts at successive rejection boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackStats {
    pub eligible_descriptors: usize,
    pub anchors: usize,
    pub one_way_matches: usize,
    pub mutual_matches: usize,
    pub multiview_groups: usize,
    pub triangulated: usize,
    pub accepted: usize,
}

#[derive(Clone, Copy)]
struct Descriptor {
    values: [f32; DESCRIPTOR_VALUES],
}

#[derive(Clone, Copy)]
struct DescriptorSample {
    pixel: [usize; 2],
    descriptor: Descriptor,
}

struct PreparedView<'a> {
    input: &'a TrackView<'a>,
    descriptor_at: Vec<usize>,
    descriptors: Vec<DescriptorSample>,
}

impl PreparedView<'_> {
    fn at(&self, pixel: usize) -> Option<DescriptorSample> {
        let index = self.descriptor_at[pixel];
        (index != usize::MAX).then(|| self.descriptors[index])
    }
}

#[derive(Clone, Copy)]
struct PairMatch {
    target: usize,
    pixel: [usize; 2],
    descriptor_error: f32,
}

/// Match and triangulate missing-foreground observations.
///
/// The capture is normally the output of [`capture::photometric_response`].
/// Only `views` participate; a caller can therefore exclude held cameras by
/// construction rather than by a score-time convention.
pub fn find(
    response: &capture::Capture,
    views: &[TrackView<'_>],
    bounds: WorldBounds,
    options: TrackOptions,
) -> Result<(Vec<Track>, TrackStats), String> {
    validate(response, views, bounds, options)?;
    let prepared: Vec<_> = views
        .iter()
        .map(|view| prepare_view(response, view, options.min_texture))
        .collect();
    let mut stats = TrackStats {
        eligible_descriptors: prepared.iter().map(|view| view.descriptors.len()).sum(),
        ..TrackStats::default()
    };
    let mut candidates = Vec::new();
    for (source_index, source) in prepared.iter().enumerate() {
        let mut targets: Vec<usize> = (0..prepared.len())
            .filter(|&target| target != source_index)
            .collect();
        let source_origin = glam::Vec3::from(
            response.views[source.input.capture_index]
                .camera
                .cam_position,
        );
        targets.sort_by(|&left, &right| {
            let distance = |index: usize| {
                let origin = glam::Vec3::from(
                    response.views[prepared[index].input.capture_index]
                        .camera
                        .cam_position,
                );
                origin.distance_squared(source_origin)
            };
            distance(left).total_cmp(&distance(right))
        });
        targets.truncate(options.max_views.saturating_sub(1));

        for source_sample in source.descriptors.iter().filter(|sample| {
            sample.pixel[0] % options.anchor_stride == 0
                && sample.pixel[1] % options.anchor_stride == 0
        }) {
            stats.anchors += 1;
            let mut matches = Vec::new();
            for &target_index in &targets {
                let target = &prepared[target_index];
                let Some(pair) = best_match(
                    response,
                    source,
                    *source_sample,
                    target,
                    target_index,
                    bounds,
                    options,
                ) else {
                    continue;
                };
                stats.one_way_matches += 1;
                let target_sample = target
                    .at(pair.pixel[1] * response.width + pair.pixel[0])
                    .expect("a match always names a prepared descriptor");
                let Some(reverse) = best_match(
                    response,
                    target,
                    target_sample,
                    source,
                    source_index,
                    bounds,
                    options,
                ) else {
                    continue;
                };
                let dx = reverse.pixel[0].abs_diff(source_sample.pixel[0]);
                let dy = reverse.pixel[1].abs_diff(source_sample.pixel[1]);
                if dx > options.epipolar_radius || dy > options.epipolar_radius {
                    continue;
                }
                stats.mutual_matches += 1;
                matches.push(pair);
            }
            if matches.len() + 1 < options.min_views {
                continue;
            }
            stats.multiview_groups += 1;
            if let Some(track) = triangulate_group(
                response,
                &prepared,
                source_index,
                source_sample.pixel,
                &matches,
                bounds,
                options,
            ) {
                stats.triangulated += 1;
                candidates.push(track);
            }
        }
    }

    let tracks = unique_tracks(candidates, response.width);
    stats.accepted = tracks.len();
    Ok((tracks, stats))
}

fn validate(
    response: &capture::Capture,
    views: &[TrackView<'_>],
    bounds: WorldBounds,
    options: TrackOptions,
) -> Result<(), String> {
    if response.width < DESCRIPTOR_SIDE || response.height < DESCRIPTOR_SIDE {
        return Err("track matching needs at least a 3x3 capture".to_string());
    }
    if !bounds.is_valid() {
        return Err("track matching needs finite non-empty world bounds".to_string());
    }
    if views.len() < options.min_views || options.min_views < 2 {
        return Err("track matching has fewer selected cameras than min_views".to_string());
    }
    if options.anchor_stride == 0 || options.max_views < options.min_views {
        return Err("track strides and view limits are inconsistent".to_string());
    }
    if options.epipolar_radius > 4 {
        return Err("epipolar radius must not exceed four pixels".to_string());
    }
    if !options.min_texture.is_finite()
        || options.min_texture < 0.0
        || !options.max_descriptor_error.is_finite()
        || options.max_descriptor_error < 0.0
        || !options.match_ratio.is_finite()
        || !(0.0..=1.0).contains(&options.match_ratio)
        || !options.min_parallax_degrees.is_finite()
        || options.min_parallax_degrees < 0.0
        || !options.max_reprojection_error.is_finite()
        || options.max_reprojection_error <= 0.0
    {
        return Err("track thresholds are invalid".to_string());
    }
    let pixels = response.width * response.height;
    let mut selected = collections::HashSet::new();
    for view in views {
        if view.capture_index >= response.views.len() {
            return Err(format!(
                "track view {} is outside {} capture views",
                view.capture_index,
                response.views.len()
            ));
        }
        if view.missing.len() != pixels {
            return Err(format!(
                "track view {} has {} missing entries, expected {pixels}",
                view.capture_index,
                view.missing.len()
            ));
        }
        if !selected.insert(view.capture_index) {
            return Err(format!("track view {} is repeated", view.capture_index));
        }
    }
    Ok(())
}

fn prepare_view<'a>(
    response: &capture::Capture,
    input: &'a TrackView<'a>,
    min_texture: f32,
) -> PreparedView<'a> {
    let view = &response.views[input.capture_index];
    let mut descriptor_at = vec![usize::MAX; response.width * response.height];
    let mut descriptors = Vec::new();
    for y in 1..response.height - 1 {
        for x in 1..response.width - 1 {
            let pixel = y * response.width + x;
            if !input.missing[pixel] {
                continue;
            }
            let Some(descriptor) = descriptor(view, response.width, x, y, min_texture) else {
                continue;
            };
            descriptor_at[pixel] = descriptors.len();
            descriptors.push(DescriptorSample {
                pixel: [x, y],
                descriptor,
            });
        }
    }
    PreparedView {
        input,
        descriptor_at,
        descriptors,
    }
}

fn descriptor(
    view: &capture::View,
    width: usize,
    x: usize,
    y: usize,
    min_texture: f32,
) -> Option<Descriptor> {
    let mut values = [0.0f32; DESCRIPTOR_VALUES];
    let mut cursor = 0;
    for patch_y in y - 1..=y + 1 {
        for patch_x in x - 1..=x + 1 {
            let pixel = patch_y * width + patch_x;
            if !view.is_foreground(pixel) {
                return None;
            }
            for value in view.pixels[pixel] {
                values[cursor] = value;
                cursor += 1;
            }
        }
    }
    let mean = values.iter().sum::<f32>() / DESCRIPTOR_VALUES as f32;
    let rms = (values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / DESCRIPTOR_VALUES as f32)
        .sqrt();
    if !rms.is_finite() || rms < min_texture {
        return None;
    }
    for value in &mut values {
        *value = (*value - mean) / rms;
    }
    Some(Descriptor { values })
}

fn descriptor_error(left: Descriptor, right: Descriptor) -> f32 {
    left.values
        .iter()
        .zip(right.values)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f32>()
        / DESCRIPTOR_VALUES as f32
}

#[allow(clippy::too_many_arguments)]
fn best_match(
    response: &capture::Capture,
    source: &PreparedView<'_>,
    source_sample: DescriptorSample,
    target: &PreparedView<'_>,
    target_index: usize,
    bounds: WorldBounds,
    options: TrackOptions,
) -> Option<PairMatch> {
    let source_camera = &response.views[source.input.capture_index].camera;
    let target_camera = &response.views[target.input.capture_index].camera;
    let origin = glam::Vec3::from(source_camera.cam_position);
    let direction = capture::pixel_direction(
        source_camera,
        response.width,
        response.height,
        source_sample.pixel[0],
        source_sample.pixel[1],
    );
    let [near, far] = ray_bounds(origin, direction, bounds)?;
    let start = origin + near * direction;
    let end = origin + far * direction;
    let (start_pixel, _) = capture::project(target_camera, response.width, response.height, start)?;
    let (end_pixel, _) = capture::project(target_camera, response.width, response.height, end)?;
    let slots = epipolar_slots(
        start_pixel,
        end_pixel,
        response.width,
        response.height,
        options.epipolar_radius,
    );
    let mut best = None;
    let mut second_error = f32::INFINITY;
    for slot in slots {
        let Some(candidate) = target.at(slot) else {
            continue;
        };
        let error = descriptor_error(source_sample.descriptor, candidate.descriptor);
        match best {
            None => best = Some((candidate, error)),
            Some((_, best_error)) if error < best_error => {
                second_error = best_error;
                best = Some((candidate, error));
            }
            Some(_) if error < second_error => second_error = error,
            Some(_) => {}
        }
    }
    let (sample, error) = best?;
    if error > options.max_descriptor_error
        || (second_error.is_finite() && error >= options.match_ratio * second_error)
    {
        return None;
    }
    Some(PairMatch {
        target: target_index,
        pixel: sample.pixel,
        descriptor_error: error,
    })
}

fn ray_bounds(origin: glam::Vec3, direction: glam::Vec3, bounds: WorldBounds) -> Option<[f32; 2]> {
    let mut near = 0.0f32;
    let mut far = f32::INFINITY;
    for axis in 0..3 {
        let coordinate = origin[axis];
        let delta = direction[axis];
        if delta.abs() < 1.0e-8 {
            if coordinate < bounds.min[axis] || coordinate > bounds.max[axis] {
                return None;
            }
            continue;
        }
        let left = (bounds.min[axis] - coordinate) / delta;
        let right = (bounds.max[axis] - coordinate) / delta;
        near = near.max(left.min(right));
        far = far.min(left.max(right));
    }
    (far.is_finite() && far > near.max(0.0)).then_some([near.max(0.0), far])
}

fn epipolar_slots(
    start: [f32; 2],
    end: [f32; 2],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<usize> {
    let delta = glam::Vec2::from(end) - glam::Vec2::from(start);
    let steps = delta.abs().max_element().ceil().max(1.0) as usize;
    let mut slots = Vec::new();
    for step in 0..=steps {
        let phase = step as f32 / steps as f32;
        let pixel = glam::Vec2::from(start) + phase * delta;
        let center_x = (pixel.x - 0.5).round() as isize;
        let center_y = (pixel.y - 0.5).round() as isize;
        for y in center_y - radius as isize..=center_y + radius as isize {
            for x in center_x - radius as isize..=center_x + radius as isize {
                if x >= 0 && y >= 0 && x < width as isize && y < height as isize {
                    slots.push(y as usize * width + x as usize);
                }
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

#[allow(clippy::too_many_arguments)]
fn triangulate_group(
    response: &capture::Capture,
    views: &[PreparedView<'_>],
    source: usize,
    source_pixel: [usize; 2],
    matches: &[PairMatch],
    bounds: WorldBounds,
    options: TrackOptions,
) -> Option<Track> {
    let source_view = views[source].input.capture_index;
    let source_camera = &response.views[source_view].camera;
    let source_origin = glam::Vec3::from(source_camera.cam_position);
    let source_direction = capture::pixel_direction(
        source_camera,
        response.width,
        response.height,
        source_pixel[0],
        source_pixel[1],
    );
    let source_observation = Observation {
        view: source_view,
        pixel: [source_pixel[0] as f32 + 0.5, source_pixel[1] as f32 + 0.5],
    };
    let min_parallax = options.min_parallax_degrees.to_radians();
    let mut best_inliers = Vec::new();
    let mut best_error = f32::INFINITY;
    for hypothesis in matches {
        let (target_origin, target_direction, _) = match_ray(response, views, *hypothesis);
        if source_direction
            .dot(target_direction)
            .clamp(-1.0, 1.0)
            .acos()
            < min_parallax
        {
            continue;
        }
        let Some(point) = triangulate_pair(
            source_origin,
            source_direction,
            target_origin,
            target_direction,
        ) else {
            continue;
        };
        if !bounds.contains(point) {
            continue;
        }
        let mut inliers = Vec::new();
        let mut error = reprojection_error(response, source_observation, point)?;
        for (index, pair) in matches.iter().enumerate() {
            let observation = match_observation(views, *pair);
            let Some(reprojection) = reprojection_error(response, observation, point) else {
                continue;
            };
            if reprojection <= options.max_reprojection_error {
                inliers.push(index);
                error += reprojection;
            }
        }
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && error < best_error)
        {
            best_inliers = inliers;
            best_error = error;
        }
    }
    if best_inliers.len() + 1 < options.min_views {
        return None;
    }

    let mut observations = vec![source_observation];
    let mut rays = vec![(source_origin, source_direction)];
    let mut descriptor_sum = 0.0f32;
    for &index in &best_inliers {
        let pair = matches[index];
        observations.push(match_observation(views, pair));
        let (origin, direction, _) = match_ray(response, views, pair);
        rays.push((origin, direction));
        descriptor_sum += pair.descriptor_error;
    }
    let point = triangulate_rays(&rays)?;
    if !bounds.contains(point) {
        return None;
    }
    let mut reprojection_sum = 0.0f32;
    for &observation in &observations {
        let error = reprojection_error(response, observation, point)?;
        if error > options.max_reprojection_error {
            return None;
        }
        reprojection_sum += error;
    }
    let parallax = ray_parallax(&rays);
    if parallax < min_parallax {
        return None;
    }
    observations.sort_by_key(|entry| entry.view);
    Some(Track {
        position: point,
        mean_reprojection_error: reprojection_sum / observations.len() as f32,
        mean_descriptor_error: descriptor_sum / best_inliers.len() as f32,
        parallax_degrees: parallax.to_degrees(),
        observations,
    })
}

fn match_ray(
    response: &capture::Capture,
    views: &[PreparedView<'_>],
    pair: PairMatch,
) -> (glam::Vec3, glam::Vec3, usize) {
    let capture_index = views[pair.target].input.capture_index;
    let camera = &response.views[capture_index].camera;
    (
        glam::Vec3::from(camera.cam_position),
        capture::pixel_direction(
            camera,
            response.width,
            response.height,
            pair.pixel[0],
            pair.pixel[1],
        ),
        capture_index,
    )
}

fn match_observation(views: &[PreparedView<'_>], pair: PairMatch) -> Observation {
    Observation {
        view: views[pair.target].input.capture_index,
        pixel: [pair.pixel[0] as f32 + 0.5, pair.pixel[1] as f32 + 0.5],
    }
}

fn triangulate_pair(
    left_origin: glam::Vec3,
    left_direction: glam::Vec3,
    right_origin: glam::Vec3,
    right_direction: glam::Vec3,
) -> Option<glam::Vec3> {
    let offset = left_origin - right_origin;
    let cross = left_direction.dot(right_direction);
    let denominator = 1.0 - cross * cross;
    if denominator.abs() < 1.0e-8 {
        return None;
    }
    let left_distance =
        (cross * right_direction.dot(offset) - left_direction.dot(offset)) / denominator;
    let right_distance =
        (right_direction.dot(offset) - cross * left_direction.dot(offset)) / denominator;
    if left_distance <= 0.0 || right_distance <= 0.0 {
        return None;
    }
    let left = left_origin + left_distance * left_direction;
    let right = right_origin + right_distance * right_direction;
    Some(0.5 * (left + right))
}

fn triangulate_rays(rays: &[(glam::Vec3, glam::Vec3)]) -> Option<glam::Vec3> {
    let mut matrix = glam::Mat3::ZERO;
    let mut right = glam::Vec3::ZERO;
    for &(origin, direction) in rays {
        let outer = glam::Mat3::from_cols(
            direction * direction.x,
            direction * direction.y,
            direction * direction.z,
        );
        let projector = glam::Mat3::IDENTITY - outer;
        matrix += projector;
        right += projector * origin;
    }
    let determinant = matrix.determinant();
    if !determinant.is_finite() || determinant.abs() < 1.0e-8 {
        return None;
    }
    let point = matrix.inverse() * right;
    point.is_finite().then_some(point)
}

fn reprojection_error(
    response: &capture::Capture,
    observation: Observation,
    point: glam::Vec3,
) -> Option<f32> {
    let (pixel, _) = capture::project(
        &response.views[observation.view].camera,
        response.width,
        response.height,
        point,
    )?;
    Some(glam::Vec2::from(pixel).distance(glam::Vec2::from(observation.pixel)))
}

fn ray_parallax(rays: &[(glam::Vec3, glam::Vec3)]) -> f32 {
    let mut maximum = 0.0f32;
    for left in 0..rays.len() {
        for right in left + 1..rays.len() {
            maximum = maximum.max(rays[left].1.dot(rays[right].1).clamp(-1.0, 1.0).acos());
        }
    }
    maximum
}

fn unique_tracks(mut candidates: Vec<Track>, width: usize) -> Vec<Track> {
    candidates.sort_by(|left, right| {
        right
            .observations
            .len()
            .cmp(&left.observations.len())
            .then_with(|| {
                left.mean_reprojection_error
                    .total_cmp(&right.mean_reprojection_error)
            })
            .then_with(|| {
                left.mean_descriptor_error
                    .total_cmp(&right.mean_descriptor_error)
            })
            .then_with(|| {
                left.position
                    .to_array()
                    .partial_cmp(&right.position.to_array())
                    .unwrap()
            })
    });
    let mut used = collections::HashSet::new();
    let mut tracks = Vec::new();
    for track in candidates {
        let observations: Vec<_> = track
            .observations
            .iter()
            .map(|observation| {
                let x = (observation.pixel[0] - 0.5).round() as usize;
                let y = (observation.pixel[1] - 0.5).round() as usize;
                (observation.view, y * width + x)
            })
            .collect();
        if observations.iter().any(|key| used.contains(key)) {
            continue;
        }
        used.extend(observations);
        tracks.push(track);
    }
    tracks.sort_by(|left, right| {
        left.position
            .to_array()
            .partial_cmp(&right.position.to_array())
            .unwrap()
    });
    tracks
}

#[cfg(test)]
mod tests {
    use super::*;
    use blade_volume as vol;

    const WIDTH: usize = 9;

    fn fixture(camera_positions: &[glam::Vec3]) -> (capture::Capture, Vec<Vec<bool>>) {
        let views = camera_positions
            .iter()
            .enumerate()
            .map(|(view_index, &position)| {
                let mut pixels = vec![[0.0; 3]; WIDTH * WIDTH];
                for y in 0..WIDTH {
                    for x in 0..WIDTH {
                        let wave = ((x * 7 + y * 11) % 13) as f32 / 13.0;
                        pixels[y * WIDTH + x] = [
                            0.1 * x as f32 + wave,
                            0.07 * y as f32 - wave,
                            0.03 * (x + y) as f32,
                        ];
                    }
                }
                capture::View {
                    name: format!("view-{view_index}"),
                    camera: vol::CameraParams::looking_at(
                        position,
                        glam::Vec3::ZERO,
                        0.9,
                        1.0,
                        100.0,
                    ),
                    pixels,
                    mask: Some(vec![1.0; WIDTH * WIDTH]),
                }
            })
            .collect();
        let mut missing = vec![vec![false; WIDTH * WIDTH]; camera_positions.len()];
        for map in &mut missing {
            map[4 * WIDTH + 4] = true;
        }
        (
            capture::Capture {
                width: WIDTH,
                height: WIDTH,
                views,
            },
            missing,
        )
    }

    fn options() -> TrackOptions {
        TrackOptions {
            anchor_stride: 1,
            max_views: 4,
            min_views: 3,
            epipolar_radius: 0,
            min_texture: 0.001,
            max_descriptor_error: 1.0e-5,
            match_ratio: 0.8,
            min_parallax_degrees: 1.0,
            max_reprojection_error: 0.01,
        }
    }

    fn bounds() -> WorldBounds {
        WorldBounds {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        }
    }

    #[test]
    fn mutually_matches_and_triangulates_a_point() {
        let cameras = [
            glam::Vec3::new(0.0, 0.0, -4.0),
            glam::Vec3::new(1.0, 0.0, -4.0),
            glam::Vec3::new(-1.0, 0.5, -4.0),
            glam::Vec3::new(0.0, -1.0, -4.0),
            glam::Vec3::new(4.0, 0.0, 0.0),
        ];
        let (capture, missing) = fixture(&cameras);
        let selected: Vec<_> = (0..4)
            .map(|index| TrackView {
                capture_index: index,
                missing: &missing[index],
            })
            .collect();

        let (tracks, stats) = find(&capture, &selected, bounds(), options()).unwrap();

        assert_eq!(tracks.len(), 1);
        assert!(tracks[0].position.length() < 1.0e-4);
        assert!(tracks[0].observations.len() >= 3);
        assert!(tracks[0]
            .observations
            .iter()
            .all(|observation| observation.view < 4));
        assert!(stats.mutual_matches >= 2);
        assert_eq!(stats.accepted, 1);
    }

    #[test]
    fn rejects_parallel_camera_rays() {
        let cameras = [glam::Vec3::new(0.0, 0.0, -4.0); 3];
        let (capture, missing) = fixture(&cameras);
        let selected: Vec<_> = (0..3)
            .map(|index| TrackView {
                capture_index: index,
                missing: &missing[index],
            })
            .collect();

        let (tracks, stats) = find(&capture, &selected, bounds(), options()).unwrap();

        assert!(tracks.is_empty());
        assert_eq!(stats.triangulated, 0);
    }

    #[test]
    fn requires_missing_observations_in_several_views() {
        let cameras = [
            glam::Vec3::new(0.0, 0.0, -4.0),
            glam::Vec3::new(1.0, 0.0, -4.0),
            glam::Vec3::new(-1.0, 0.5, -4.0),
        ];
        let (capture, mut missing) = fixture(&cameras);
        missing[1].fill(false);
        missing[2].fill(false);
        let selected: Vec<_> = (0..3)
            .map(|index| TrackView {
                capture_index: index,
                missing: &missing[index],
            })
            .collect();

        let (tracks, stats) = find(&capture, &selected, bounds(), options()).unwrap();

        assert!(tracks.is_empty());
        assert_eq!(stats.multiview_groups, 0);
    }

    #[test]
    fn rejects_inconsistent_response_descriptors() {
        let cameras = [
            glam::Vec3::new(0.0, 0.0, -4.0),
            glam::Vec3::new(1.0, 0.0, -4.0),
            glam::Vec3::new(-1.0, 0.5, -4.0),
        ];
        let (mut capture, missing) = fixture(&cameras);
        for (view_index, view) in capture.views.iter_mut().enumerate() {
            for y in 3..=5 {
                for x in 3..=5 {
                    let phase = (view_index + 1) as f32;
                    view.pixels[y * WIDTH + x] = [
                        phase * x as f32 + y as f32,
                        x as f32 - phase * y as f32,
                        phase * (x * y) as f32,
                    ];
                }
            }
        }
        let selected: Vec<_> = (0..3)
            .map(|index| TrackView {
                capture_index: index,
                missing: &missing[index],
            })
            .collect();

        let (tracks, stats) = find(&capture, &selected, bounds(), options()).unwrap();

        assert!(tracks.is_empty());
        assert_eq!(stats.mutual_matches, 0);
    }

    #[test]
    fn point_bounds_expand_by_the_requested_fraction() {
        let bounds =
            WorldBounds::from_points(&[glam::Vec3::ZERO, glam::Vec3::new(2.0, 1.0, 0.5)], 0.25)
                .unwrap();
        assert_eq!(bounds.min, glam::Vec3::splat(-0.5));
        assert_eq!(bounds.max, glam::Vec3::new(2.5, 1.5, 1.0));
    }
}
