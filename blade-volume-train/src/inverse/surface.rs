//! Turning a point cloud into oriented surface elements.
//!
//! A surfel needs three things a bare point does not have: a normal, a side to
//! face, and a radius. The normal comes from the local covariance, the side
//! from the cameras that can see the point, and the radius from how far apart
//! the neighbours are — a disc smaller than the spacing leaves holes, and one
//! much larger turns a thin structure into a slab.

use crate::inverse::capture;
use blade_volume as vol;

/// How the cloud is turned into discs.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceOptions {
    /// Neighbours used for the covariance and the spacing estimate.
    pub neighbours: usize,
    /// Disc radius as a multiple of the mean neighbour distance. Above one
    /// the discs overlap, which is what closes the gaps between them.
    pub radius_factor: f32,
    /// Discard a point whose neighbours are further away than this multiple
    /// of the cloud's median spacing. Sparse reconstruction leaves outliers
    /// floating in the middle of a room and each one becomes a visible disc.
    pub outlier_factor: f32,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        Self {
            neighbours: 12,
            radius_factor: 1.4,
            outlier_factor: 6.0,
        }
    }
}

/// The direction of least variance among a set of offsets.
///
/// Power iteration on `trace(C) I - C`, whose largest eigenvector is the
/// smallest of `C`. Going at it this way avoids an eigen decomposition and is
/// well conditioned exactly when the patch is planar, which is when the normal
/// means anything.
fn least_variance_direction(offsets: &[glam::Vec3]) -> Option<glam::Vec3> {
    if offsets.len() < 3 {
        return None;
    }
    // Fit the plane through the query (offset zero) and its neighbours around
    // their centroid. Covariance around the query instead biases a curved or
    // one-sided neighbourhood towards whichever side happened to contain more
    // samples.
    let mean = offsets.iter().sum::<glam::Vec3>() / (offsets.len() + 1) as f32;
    let mut covariance = glam::Mat3::ZERO;
    for offset in offsets {
        let offset = *offset - mean;
        covariance +=
            glam::Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
    }
    let query = -mean;
    covariance += glam::Mat3::from_cols(query * query.x, query * query.y, query * query.z);
    let trace = covariance.x_axis.x + covariance.y_axis.y + covariance.z_axis.z;
    if trace <= 0.0 || !trace.is_finite() {
        return None;
    }
    let shifted = glam::Mat3::IDENTITY * trace - covariance;
    // Not an axis, so a patch lying in a coordinate plane still converges.
    let mut vector = glam::Vec3::new(0.577_35, 0.211_32, 0.788_67);
    for _ in 0..64 {
        let next = shifted * vector;
        let length = next.length();
        if length < 1.0e-20 {
            return None;
        }
        vector = next / length;
    }
    vector.try_normalize()
}

/// Local surface inferred around one point without any polygonal intermediate.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceEstimate {
    pub normal: glam::Vec3,
    pub spacing: f32,
}

/// Blend camera-derived surface normals towards a local density-field slope.
///
/// Depth derivatives estimate each view independently. The density field was
/// trained through all views at once, so its local slope supplies a weak
/// cross-view orientation signal. Its sign is not independently meaningful;
/// preserve the camera-facing side of the established normal and use only a
/// caller-selected trust region.
pub fn refine_normals_from_density(
    surfels: &mut [vol::relight::Surfel],
    foam: &vol::PointCloudModel,
    blend: f32,
) -> usize {
    const NEIGHBOURS: usize = 32;

    if surfels.is_empty() || foam.points.len() < 4 || blend <= 0.0 {
        return 0;
    }
    let positions: Vec<[f32; 3]> = foam
        .points
        .iter()
        .map(|point| point.truncate().to_array())
        .collect();
    let tree = kiddo::ImmutableKdTree::new_from_slice(&positions);
    let want = std::num::NonZero::new(NEIGHBOURS.min(foam.points.len())).unwrap();
    let mut changed = 0;
    for surfel in surfels {
        let hits = tree.nearest_n::<kiddo::SquaredEuclidean>(&surfel.center, want);
        let mut mean_position = glam::Vec3::ZERO;
        let mut mean_density = 0.0f32;
        let mut weight = 0.0f32;
        for hit in &hits {
            let point = foam.points[hit.item as usize];
            let sample_weight = 1.0 / hit.distance.max(1.0e-6).sqrt();
            mean_position += point.truncate() * sample_weight;
            mean_density += point.w * sample_weight;
            weight += sample_weight;
        }
        mean_position /= weight;
        mean_density /= weight;

        let mut covariance = glam::Mat3::ZERO;
        let mut right = glam::Vec3::ZERO;
        for hit in &hits {
            let point = foam.points[hit.item as usize];
            let sample_weight = 1.0 / hit.distance.max(1.0e-6).sqrt();
            let offset = point.truncate() - mean_position;
            covariance +=
                glam::Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z)
                    * sample_weight;
            right += offset * ((point.w - mean_density) * sample_weight);
        }
        let trace = covariance.x_axis.x + covariance.y_axis.y + covariance.z_axis.z;
        covariance += glam::Mat3::IDENTITY * (trace * 1.0e-4).max(1.0e-8);
        if covariance.determinant().abs() <= 1.0e-12 {
            continue;
        }
        let Some(mut gradient) = (covariance.inverse() * right).try_normalize() else {
            continue;
        };
        let prior = glam::Vec3::from(surfel.normal);
        if gradient.dot(prior) < 0.0 {
            gradient = -gradient;
        }
        let Some(normal) = prior.lerp(gradient, blend).try_normalize() else {
            continue;
        };
        surfel.normal = normal.to_array();
        changed += 1;
    }
    changed
}

/// Estimate an oriented local surface around every point.
///
/// A missing entry is an outlier or a neighbourhood that does not define a
/// plane. The vector remains aligned with `points`, which lets callers retain
/// their own material and observation indexing while dropping unsupported
/// particles.
pub fn estimate_surfaces(
    points: &[glam::Vec3],
    cameras: &[glam::Vec3],
    options: SurfaceOptions,
) -> Vec<Option<SurfaceEstimate>> {
    if points.len() < options.neighbours + 1 {
        return vec![None; points.len()];
    }
    let positions: Vec<[f32; 3]> = points.iter().map(|p| p.to_array()).collect();
    let tree = kiddo::ImmutableKdTree::new_from_slice(&positions);
    let want = std::num::NonZero::new((options.neighbours + 1).min(points.len()))
        .expect("the cloud is not empty");

    // Two passes: the first measures the spacing everywhere so the second can
    // say what "too spread out" means for this cloud rather than for a scale
    // chosen in advance.
    let mut spacings = Vec::with_capacity(points.len());
    let mut neighbourhoods = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let hits = tree.nearest_n::<kiddo::SquaredEuclidean>(&point.to_array(), want);
        let mut offsets = Vec::with_capacity(hits.len());
        let mut total = 0.0f32;
        let mut counted = 0usize;
        for hit in &hits {
            if hit.item as usize == index {
                continue;
            }
            offsets.push(points[hit.item as usize] - *point);
            total += hit.distance.max(0.0).sqrt();
            counted += 1;
        }
        let spacing = if counted > 0 {
            total / counted as f32
        } else {
            f32::INFINITY
        };
        spacings.push(spacing);
        neighbourhoods.push(offsets);
    }
    let mut sorted: Vec<f32> = spacings.iter().copied().filter(|s| s.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.is_empty() {
        1.0
    } else {
        sorted[sorted.len() / 2]
    };
    let spacing_limit = median * options.outlier_factor;

    let mut estimates = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let spacing = spacings[index];
        if !spacing.is_finite() || spacing <= 0.0 || spacing > spacing_limit {
            estimates.push(None);
            continue;
        }
        let normal = match least_variance_direction(&neighbourhoods[index]) {
            Some(n) => n,
            None => {
                estimates.push(None);
                continue;
            }
        };
        // A covariance has no sign. Face the cameras: the nearest one is the
        // one whose observation of this point is least likely to be occluded.
        let facing = nearest_camera(cameras, *point).map_or(normal, |camera| camera - *point);
        let normal = if normal.dot(facing) < 0.0 {
            -normal
        } else {
            normal
        };
        estimates.push(Some(SurfaceEstimate { normal, spacing }));
    }
    estimates
}

/// Build discs from a cloud, facing whichever cameras can see them.
///
/// Every surfel is given material zero; the caller decides what that material
/// is. Points whose neighbourhood is too spread out to be a surface are
/// dropped, and the count of those is returned so the caller can say how much
/// of the cloud survived.
pub fn surfels_from_points(
    points: &[glam::Vec3],
    cameras: &[glam::Vec3],
    options: SurfaceOptions,
) -> (Vec<vol::relight::Surfel>, usize) {
    let estimates = estimate_surfaces(points, cameras, options);
    let mut surfels = Vec::with_capacity(points.len());
    let mut dropped = 0usize;
    for (point, estimate) in points.iter().zip(estimates) {
        let Some(estimate) = estimate else {
            dropped += 1;
            continue;
        };
        // One material per surviving surfel, numbered in the order they come
        // out: the caller builds a table of exactly this length.
        surfels.push(vol::relight::Surfel {
            center: point.to_array(),
            radius: estimate.spacing * options.radius_factor,
            normal: estimate.normal.to_array(),
            material: surfels.len() as u32,
        });
    }
    (surfels, dropped)
}

/// Build discs from points that already carry independently measured normals.
///
/// Local covariance still establishes spacing and rejects isolated points, but
/// it does not replace the normal supplied by dense multi-view stereo.
pub fn surfels_from_oriented_points(
    points: &[(glam::Vec3, glam::Vec3)],
    options: SurfaceOptions,
) -> (Vec<vol::relight::Surfel>, usize) {
    let positions: Vec<_> = points.iter().map(|entry| entry.0).collect();
    let estimates = estimate_surfaces(&positions, &[], options);
    let mut surfels = Vec::with_capacity(points.len());
    let mut dropped = 0usize;
    for (point, estimate) in points.iter().zip(estimates) {
        let (Some(estimate), Some(normal)) = (estimate, point.1.try_normalize()) else {
            dropped += 1;
            continue;
        };
        surfels.push(vol::relight::Surfel {
            center: point.0.to_array(),
            radius: estimate.spacing * options.radius_factor,
            normal: normal.to_array(),
            material: surfels.len() as u32,
        });
    }
    (surfels, dropped)
}

fn nearest_camera(cameras: &[glam::Vec3], point: glam::Vec3) -> Option<glam::Vec3> {
    cameras.iter().copied().min_by(|a, b| {
        (*a - point)
            .length_squared()
            .partial_cmp(&(*b - point).length_squared())
            .unwrap()
    })
}

/// Flip normals that face away from every camera that can see the point.
///
/// The covariance gives an axis and the nearest camera gives a side, but the
/// nearest camera can be on the wrong side of a wall. Counting how many views
/// actually contain the point, weighted by how square-on they are, is a better
/// vote — and it costs one projection per view per surfel.
pub fn orient_towards_views(
    surfels: &mut [vol::relight::Surfel],
    capture: &capture::Capture,
) -> usize {
    let mut flipped = 0usize;
    for surfel in surfels.iter_mut() {
        let center = glam::Vec3::from(surfel.center);
        let normal = glam::Vec3::from(surfel.normal);
        let mut vote = 0.0f32;
        for view in &capture.views {
            if capture::project(&view.camera, capture.width, capture.height, center).is_none() {
                continue;
            }
            let towards = (glam::Vec3::from(view.camera.cam_position) - center).normalize_or_zero();
            vote += normal.dot(towards);
        }
        if vote < 0.0 {
            surfel.normal = (-normal).to_array();
            flipped += 1;
        }
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn density_lattice(constant: bool) -> vol::PointCloudModel {
        let mut points = Vec::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let position =
                        0.1 * glam::Vec3::new(x as f32 - 1.5, y as f32 - 1.5, z as f32 - 1.5);
                    let density = if constant { 1.0 } else { 1.0 + position.y };
                    points.push(position.extend(density));
                }
            }
        }
        vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
            points,
        }
    }

    #[test]
    fn oriented_points_keep_the_independent_normal() {
        let points: Vec<_> = (0..4)
            .flat_map(|y| {
                (0..4).map(move |x| (glam::Vec3::new(x as f32, y as f32, 0.0), glam::Vec3::Y))
            })
            .collect();
        let (surfels, dropped) = surfels_from_oriented_points(&points, SurfaceOptions::default());
        assert_eq!(dropped, 0);
        assert_eq!(surfels.len(), points.len());
        assert!(surfels
            .iter()
            .all(|surfel| glam::Vec3::from(surfel.normal) == glam::Vec3::Y));
    }

    fn plane_cloud(normal: glam::Vec3, extent: usize) -> Vec<glam::Vec3> {
        let tangent = if normal.z.abs() < 0.9 {
            glam::Vec3::Z.cross(normal).normalize()
        } else {
            glam::Vec3::X.cross(normal).normalize()
        };
        let bitangent = normal.cross(tangent);
        let mut points = Vec::new();
        for i in 0..extent {
            for j in 0..extent {
                let u = i as f32 - extent as f32 * 0.5;
                let v = j as f32 - extent as f32 * 0.5;
                points.push(0.1 * (u * tangent + v * bitangent));
            }
        }
        points
    }

    #[test]
    fn a_flat_patch_gets_the_plane_normal_and_faces_the_camera() {
        let truth = glam::Vec3::new(0.3, 0.9, -0.2).normalize();
        let points = plane_cloud(truth, 12);
        // Deliberately on the negative side, so a normal that ignored the
        // camera would come back pointing the wrong way half the time.
        let camera = -3.0 * truth;
        let (surfels, dropped) = surfels_from_points(&points, &[camera], SurfaceOptions::default());
        assert_eq!(dropped, 0);
        assert_eq!(surfels.len(), points.len());
        for surfel in &surfels {
            let normal = glam::Vec3::from(surfel.normal);
            assert!(
                normal.dot(-truth) > 0.99,
                "normal {normal:?} against plane {truth:?}"
            );
            assert!(surfel.radius > 0.0);
        }
    }

    #[test]
    fn density_gradient_supplies_a_conservative_normal_correction() {
        let initial = glam::Vec3::new(0.6, 0.8, 0.0).normalize();
        let mut surfels = vec![vol::relight::Surfel {
            center: [0.0; 3],
            radius: 0.2,
            normal: initial.to_array(),
            material: 0,
        }];
        let unchanged = surfels[0].normal;
        assert_eq!(
            refine_normals_from_density(&mut surfels, &density_lattice(false), 0.0),
            0
        );
        assert_eq!(surfels[0].normal, unchanged);
        assert_eq!(
            refine_normals_from_density(&mut surfels, &density_lattice(false), 0.1),
            1
        );
        let refined = glam::Vec3::from(surfels[0].normal);
        assert!(refined.dot(glam::Vec3::Y) > initial.dot(glam::Vec3::Y));
        assert!(refined.dot(initial) > 0.99);

        let before = surfels[0].normal;
        assert_eq!(
            refine_normals_from_density(&mut surfels, &density_lattice(true), 0.1),
            0
        );
        assert_eq!(surfels[0].normal, before);
    }

    #[test]
    fn a_disc_is_wide_enough_to_meet_its_neighbours() {
        // Discs narrower than the spacing render a sieve. The spacing here is
        // 0.1 by construction, so the radius has to clear it.
        let points = plane_cloud(glam::Vec3::Y, 10);
        let (surfels, _) =
            surfels_from_points(&points, &[glam::Vec3::Y * 3.0], SurfaceOptions::default());
        let smallest = surfels
            .iter()
            .map(|s| s.radius)
            .fold(f32::INFINITY, f32::min);
        assert!(smallest > 0.1, "smallest radius {smallest}");
    }

    #[test]
    fn a_point_alone_in_the_middle_of_the_room_is_dropped() {
        let mut points = plane_cloud(glam::Vec3::Y, 10);
        points.push(glam::Vec3::new(0.0, 40.0, 0.0));
        let (surfels, dropped) =
            surfels_from_points(&points, &[glam::Vec3::Y * 3.0], SurfaceOptions::default());
        assert!(dropped >= 1, "the outlier survived");
        for surfel in &surfels {
            assert!(surfel.center[1] < 10.0, "the outlier became a disc");
        }
    }

    #[test]
    fn surface_estimates_stay_aligned_when_an_outlier_is_dropped() {
        let mut points = plane_cloud(glam::Vec3::Y, 10);
        points.push(glam::Vec3::new(0.0, 40.0, 0.0));
        let estimates =
            estimate_surfaces(&points, &[glam::Vec3::Y * 3.0], SurfaceOptions::default());
        assert_eq!(estimates.len(), points.len());
        assert!(estimates[..points.len() - 1].iter().all(Option::is_some));
        assert!(estimates.last().unwrap().is_none());
    }
}
