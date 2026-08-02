//! Surfaces out of a density field.
//!
//! A trained foam is a cloud of transparency, not a surface: it says how much
//! light every point in space absorbs, and nowhere does it say "here is a
//! wall". The surface a photograph sees is where the ray was absorbed, and the
//! expected value of that position along the ray is a depth — one per pixel,
//! per view. Unprojecting those and merging them across views is how a
//! volumetric fit becomes something a surfel renderer can draw.
//!
//! Normals come from the depth map rather than from the merged cloud. The
//! screen-space derivatives of a depth map already know which way the surface
//! faces, because they know which side the camera was on; a covariance fitted
//! to the merged points has to guess, and guesses wrong on every thin
//! structure — which in a room is most of them.

use crate::inverse::capture;
use blade_volume as vol;
use std::{collections::HashMap, thread};

/// Where the surface is, from one point of view.
pub struct DepthMap {
    pub width: usize,
    pub height: usize,
    /// Expected distance from the camera, in world units. Meaningless where
    /// `alpha` is small.
    pub distance: Vec<f32>,
    /// How much of the ray was absorbed. Low values are empty space seen
    /// through, and produce no surface.
    pub alpha: Vec<f32>,
    /// The largest share of the ray any one segment absorbed. Low values are
    /// haze: absorbed, but not by anything that is a surface.
    pub peak: Vec<f32>,
}

/// How a density field is turned into discs.
#[derive(Clone, Copy, Debug)]
pub struct DepthOptions {
    /// Rays below this absorption are treated as having hit nothing.
    pub min_alpha: f32,
    /// Weight the single strongest segment of a ray must carry for that ray to
    /// count as having met a surface.
    ///
    /// This is what separates a wall from haze. Without it, a field trained on
    /// a room reports a confident depth for every pixel, because every ray is
    /// eventually absorbed by fog, and the fused result is a cloud of dust
    /// filling the room rather than the surfaces around it.
    pub min_peak: f32,
    /// Traversal steps per ray. The far end of a truncated ray is missing,
    /// which shows up as a hole rather than as a wrong depth.
    pub max_steps: u32,
    /// Merge cell size, as a multiple of the pixel footprint at the median
    /// depth. One means about one surfel per pixel of the extraction, which
    /// is as fine as the evidence goes.
    pub voxel_factor: f32,
    /// Disc radius as a multiple of the merge cell.
    ///
    /// Has to clear the half-diagonal of a cell face, `sqrt(2)/2 = 0.71`, or
    /// the discs do not meet and every pixel is part background. That reads as
    /// a uniformly dark render rather than as visible holes, because the gaps
    /// are a fraction of a pixel wide.
    pub disc_radius: f32,
    /// Reject a pixel whose neighbours disagree about depth by more than this
    /// fraction of its own. These are silhouettes, where a normal taken from
    /// the derivatives spans two surfaces and belongs to neither.
    pub max_depth_step: f32,
    /// Distinct camera views that must contribute to a merge cell.
    ///
    /// One preserves single-view surfaces. Raising this rejects density-field
    /// layers that only one depth map supports, at the cost of surfaces seen
    /// from just one training camera.
    pub min_views: usize,
}

impl Default for DepthOptions {
    fn default() -> Self {
        Self {
            min_alpha: 0.5,
            min_peak: 0.15,
            max_steps: 512,
            voxel_factor: 1.0,
            disc_radius: 1.0,
            max_depth_step: 0.02,
            min_views: 2,
        }
    }
}

/// Trace one view and keep where each ray was absorbed.
pub fn trace_depth(
    model: &vol::PointCloudModel,
    camera: &vol::CameraParams,
    width: usize,
    height: usize,
    start_point: u32,
    max_steps: u32,
) -> DepthMap {
    let settings = vol::trace::TraceSettings {
        weight_threshold: 0.001,
        max_steps,
        start_point,
        depth: camera.depth,
        eval_mode: vol::trace::EvalMode::ConstantRgb(glam::Vec3::ONE),
    };
    let origin = glam::Vec3::from(camera.cam_position);
    let mut distance = vec![0.0f32; width * height];
    let mut alpha = vec![0.0f32; width * height];
    let mut peak = vec![0.0f32; width * height];

    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let rows = height.div_ceil(threads).max(1);
    thread::scope(|scope| {
        for (block, ((distance, alpha), peak)) in distance
            .chunks_mut(rows * width)
            .zip(alpha.chunks_mut(rows * width))
            .zip(peak.chunks_mut(rows * width))
            .enumerate()
        {
            scope.spawn(move || {
                for local in 0..distance.len() / width {
                    let y = block * rows + local;
                    for x in 0..width {
                        let direction = capture::pixel_direction(camera, width, height, x, y);
                        let result = vol::trace::trace_one_ray(
                            model,
                            vol::trace::Ray { origin, direction },
                            settings,
                        );
                        let slot = local * width + x;
                        alpha[slot] = result.rgba.w;
                        peak[slot] = result.peak_weight;
                        // The mode, not the mean or the median: it is the only
                        // one that comes with evidence that a surface was met
                        // at all.
                        distance[slot] = result.depth_mode;
                    }
                }
            });
        }
    });
    DepthMap {
        width,
        height,
        distance,
        alpha,
        peak,
    }
}

/// One oriented sample of a surface, before merging.
struct Sample {
    position: glam::Vec3,
    normal: glam::Vec3,
}

/// Unproject a depth map into oriented world-space samples.
///
/// The normal is the cross product of the two screen-space derivatives, which
/// faces the camera by construction. Pixels on a silhouette are dropped: their
/// derivatives straddle a depth discontinuity, and the normal that comes out
/// is perpendicular to nothing in the scene.
fn samples_from(map: &DepthMap, camera: &vol::CameraParams, options: DepthOptions) -> Vec<Sample> {
    let (width, height) = (map.width, map.height);
    let origin = glam::Vec3::from(camera.cam_position);
    let point_at = |x: usize, y: usize| -> Option<glam::Vec3> {
        let slot = y * width + x;
        if map.alpha[slot] < options.min_alpha || map.peak[slot] < options.min_peak {
            return None;
        }
        let direction = capture::pixel_direction(camera, width, height, x, y);
        Some(origin + map.distance[slot] * direction)
    };

    let mut samples = Vec::new();
    for y in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            let (Some(here), Some(right), Some(down)) =
                (point_at(x, y), point_at(x + 1, y), point_at(x, y + 1))
            else {
                continue;
            };
            let distance = map.distance[y * width + x];
            let step = options.max_depth_step * distance;
            if (map.distance[y * width + x + 1] - distance).abs() > step
                || (map.distance[(y + 1) * width + x] - distance).abs() > step
            {
                continue;
            }
            let Some(normal) = (right - here).cross(down - here).try_normalize() else {
                continue;
            };
            // Towards the camera: the cross product's sign depends on the
            // handedness of the pixel grid, which is not worth reasoning about
            // when the answer is one dot product away.
            let normal = if normal.dot(origin - here) < 0.0 {
                -normal
            } else {
                normal
            };
            samples.push(Sample {
                position: here,
                normal,
            });
        }
    }
    samples
}

/// Merge oriented samples from many views into one set of discs.
///
/// Merging is a voxel average rather than a nearest-neighbour cluster: every
/// view sees the same wall and contributes a sample to it, and without the
/// merge the same surface is drawn thirty times over. Normals are summed
/// rather than averaged as unit vectors, so a cell whose samples disagree ends
/// up with a short vector and is dropped instead of being given a normal that
/// no view supports.
pub fn surfels_from_depth(
    maps: &[(vol::CameraParams, DepthMap)],
    options: DepthOptions,
) -> (Vec<vol::relight::Surfel>, f32) {
    let mut distances = Vec::new();
    for entry in maps {
        let (_, ref map) = *entry;
        for (slot, &alpha) in map.alpha.iter().enumerate() {
            if alpha >= options.min_alpha && map.peak[slot] >= options.min_peak {
                distances.push(map.distance[slot]);
            }
        }
    }
    if distances.is_empty() {
        return (Vec::new(), 0.0);
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_distance = distances[distances.len() / 2];

    // A pixel's world footprint at the typical depth, which is the finest a
    // surface can be resolved from these images.
    let (ref camera, ref map) = maps[0];
    let footprint = median_distance * (0.5 * camera.fov[0]).tan() * 2.0 / map.width as f32;
    let voxel = (footprint * options.voxel_factor).max(1.0e-6);

    struct Cell {
        positions: glam::Vec3,
        normals: glam::Vec3,
        samples: u32,
        views: usize,
        last_view: usize,
    }

    let mut cells: HashMap<[i32; 3], Cell> = HashMap::new();
    for (view, entry) in maps.iter().enumerate() {
        let (ref camera, ref map) = *entry;
        for sample in samples_from(map, camera, options) {
            let key = [
                (sample.position.x / voxel).floor() as i32,
                (sample.position.y / voxel).floor() as i32,
                (sample.position.z / voxel).floor() as i32,
            ];
            let entry = cells.entry(key).or_insert(Cell {
                positions: glam::Vec3::ZERO,
                normals: glam::Vec3::ZERO,
                samples: 0,
                views: 0,
                last_view: usize::MAX,
            });
            entry.positions += sample.position;
            entry.normals += sample.normal;
            entry.samples += 1;
            if entry.last_view != view {
                entry.views += 1;
                entry.last_view = view;
            }
        }
    }

    // A disc has to reach its neighbours in the grid, and the worst case is
    // the diagonal of a cell face.
    let radius = voxel * options.disc_radius.max(0.1);
    let mut surfels = Vec::with_capacity(cells.len());
    for (_, entry) in cells {
        if entry.views < options.min_views.max(1) {
            continue;
        }
        // Half the samples pointing one way and half the other cancel, and a
        // cell like that is two surfaces sharing a voxel rather than one.
        let Some(normal) = (entry.normals / entry.samples as f32).try_normalize() else {
            continue;
        };
        if entry.normals.length() < 0.3 * entry.samples as f32 {
            continue;
        }
        surfels.push(vol::relight::Surfel {
            center: (entry.positions / entry.samples as f32).to_array(),
            radius,
            normal: normal.to_array(),
            material: surfels.len() as u32,
        });
    }
    // Deterministic order: a hash map iterates differently every run, and the
    // material table is indexed by position in this list.
    surfels.sort_by(|a, b| a.center.partial_cmp(&b.center).unwrap());
    for (index, surfel) in surfels.iter_mut().enumerate() {
        surfel.material = index as u32;
    }
    (surfels, voxel)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera_looking_at_a_wall(distance: f32) -> vol::CameraParams {
        vol::CameraParams {
            cam_position: [0.0, 0.0, -distance],
            depth: 100.0,
            cam_orientation: glam::Quat::IDENTITY.to_array(),
            fov: [1.0, 1.0],
            principal: [0.0, 0.0],
        }
    }

    /// A flat wall at `z = 0`, as a depth map would see it.
    fn wall_map(camera: &vol::CameraParams, size: usize) -> DepthMap {
        let origin = glam::Vec3::from(camera.cam_position);
        let mut distance = vec![0.0f32; size * size];
        let alpha = vec![1.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let direction = capture::pixel_direction(camera, size, size, x, y);
                // Distance to the plane z = 0 along this ray.
                distance[y * size + x] = -origin.z / direction.z;
            }
        }
        DepthMap {
            width: size,
            height: size,
            distance,
            alpha,
            peak: vec![1.0f32; size * size],
        }
    }

    #[test]
    fn a_flat_wall_comes_back_flat_and_facing_the_camera() {
        let camera = camera_looking_at_a_wall(3.0);
        let map = wall_map(&camera, 64);
        let options = DepthOptions {
            min_views: 1,
            ..Default::default()
        };
        let (surfels, voxel) = surfels_from_depth(&[(camera, map)], options);
        assert!(!surfels.is_empty());
        assert!(voxel > 0.0);
        for surfel in &surfels {
            assert!(
                surfel.center[2].abs() < 0.05,
                "a point of the wall landed at z = {}",
                surfel.center[2]
            );
            let normal = glam::Vec3::from(surfel.normal);
            assert!(
                normal.dot(glam::Vec3::NEG_Z) > 0.99,
                "normal {normal:?} does not face the camera"
            );
        }
    }

    #[test]
    fn two_views_of_one_wall_do_not_make_two_walls() {
        // The point of merging: every view sees the same surface, and without
        // the voxel average the surfel count grows with the number of views
        // while the geometry does not.
        let first = camera_looking_at_a_wall(3.0);
        let second = camera_looking_at_a_wall(3.2);
        let options = DepthOptions {
            min_views: 1,
            ..Default::default()
        };
        let alone = surfels_from_depth(&[(first, wall_map(&first, 64))], options)
            .0
            .len();
        let together = surfels_from_depth(
            &[
                (first, wall_map(&first, 64)),
                (second, wall_map(&second, 64)),
            ],
            options,
        )
        .0
        .len();
        assert!(
            together < 2 * alone,
            "{together} surfels from two views against {alone} from one"
        );
    }

    #[test]
    fn a_ray_that_hit_nothing_contributes_no_surface() {
        let camera = camera_looking_at_a_wall(3.0);
        let mut map = wall_map(&camera, 32);
        map.alpha.fill(0.01);
        let (surfels, _) = surfels_from_depth(&[(camera, map)], DepthOptions::default());
        assert!(
            surfels.is_empty(),
            "{} surfels from empty space",
            surfels.len()
        );
    }

    #[test]
    fn view_consensus_counts_cameras_not_samples() {
        let camera = camera_looking_at_a_wall(3.0);
        let map = wall_map(&camera, 32);
        let options = DepthOptions {
            min_views: 2,
            ..Default::default()
        };
        let (alone, _) = surfels_from_depth(&[(camera, map)], options);
        assert!(alone.is_empty(), "one camera satisfied two-view consensus");

        let second = camera_looking_at_a_wall(3.2);
        let (together, _) = surfels_from_depth(
            &[
                (camera, wall_map(&camera, 32)),
                (second, wall_map(&second, 32)),
            ],
            options,
        );
        assert!(
            !together.is_empty(),
            "two cameras produced no consensus surface"
        );
    }
}
