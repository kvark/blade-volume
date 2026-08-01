//! A capture whose answer is known.
//!
//! Re-rendering accuracy on a photograph cannot tell a decomposition from a
//! disguise: paint the illumination onto the surfaces and the images come back
//! perfect. The only way to know whether the material and the light are
//! separately anything is to build a scene, photograph it, and check what
//! comes back against what went in.
//!
//! The photographs are taken with **shadows and a bounce** and the fit is done
//! with neither. That asymmetry is the point. If the capture were rendered
//! with the same analytic model the solver assumes, the solver would be
//! inverting its own forward pass and would say more about arithmetic than
//! about reconstruction. Rendering with a strictly richer model means what the
//! fit cannot explain is the same kind of thing it cannot explain about a
//! room: light that arrived by a path the model has no term for.
//!
//! # What can be checked, and what cannot
//!
//! Two gauges are chosen rather than recovered — overall scale and per-channel
//! white balance — so a bare albedo comparison would be measuring the choice.
//! Everything here therefore reports the gauge separately: the per-channel
//! factor that best aligns the recovered albedo with the truth is fitted and
//! quoted, and the error is what survives it.

use crate::inverse::{capture, score};
use blade_volume as vol;

/// A scene built to be recovered from.
///
/// Spheres on a floor in front of a wall, in five plainly different diffuse
/// colours. Every part of it is there for a reason a converted asset cannot
/// supply:
///
///   - **Diffuse albedo that is not zero.** A glTF authored fully metallic —
///     which `police.glb` is, on all nine of its materials — has no diffuse
///     response at all, and a test of a diffuse decomposition against it is
///     measuring nothing.
///   - **Curved surfaces.** A sphere presents every normal, which is what
///     leaves the light as the only explanation for how its shading varies.
///     A scene of flat panels would let almost any light fit.
///   - **Things that occlude each other.** The spheres shadow the floor and
///     bounce onto it, so the photographs contain light the fit has no term
///     for. That is the whole point of the exercise.
///   - **Gloss that varies, and one metal.** Roughness and F0 are recoverable
///     only from how a surface changes between views, so a scene where every
///     material is a rough dielectric would be passed by a solver that always
///     answers "rough dielectric". The ladder from 0.9 down to 0.15, plus a
///     gold sphere with no diffuse response at all, is what makes those two
///     parameters answerable.
pub fn studio(spacing: f32) -> vol::relight::RelightModel {
    /// Albedo, roughness, and reflectance at normal incidence.
    ///
    /// Metalness is not a field here for the same reason it is not one in the
    /// fit: a metal is what zero albedo and a high coloured F0 *mean*.
    const MATERIALS: [([f32; 3], f32, [f32; 3]); 5] = [
        ([0.72, 0.70, 0.66], 0.90, [0.04; 3]), // floor and wall, near neutral
        ([0.66, 0.18, 0.14], 0.55, [0.04; 3]), // red, slightly glossy
        ([0.16, 0.42, 0.20], 0.30, [0.04; 3]), // green, glossier
        ([0.20, 0.30, 0.70], 0.15, [0.04; 3]), // blue, nearly a mirror
        ([0.0; 3], 0.25, [1.0, 0.72, 0.29]),   // gold: a metal
    ];
    let spacing = spacing.max(1.0e-4);
    let radius = spacing * 0.75;
    let mut surfels = Vec::new();

    let plane = |origin: glam::Vec3,
                 across: glam::Vec3,
                 down: glam::Vec3,
                 normal: glam::Vec3,
                 material: u32,
                 surfels: &mut Vec<vol::relight::Surfel>| {
        let steps_across = (across.length() / spacing).ceil() as usize;
        let steps_down = (down.length() / spacing).ceil() as usize;
        for i in 0..=steps_across {
            for j in 0..=steps_down {
                let u = i as f32 / steps_across as f32;
                let v = j as f32 / steps_down as f32;
                surfels.push(vol::relight::Surfel {
                    center: (origin + u * across + v * down).to_array(),
                    radius,
                    normal: normal.to_array(),
                    material,
                });
            }
        }
    };
    plane(
        glam::Vec3::new(-4.0, 0.0, -4.0),
        glam::Vec3::new(8.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 0.0, 8.0),
        glam::Vec3::Y,
        0,
        &mut surfels,
    );
    plane(
        glam::Vec3::new(-4.0, 0.0, 4.0),
        glam::Vec3::new(8.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 4.0, 0.0),
        glam::Vec3::NEG_Z,
        0,
        &mut surfels,
    );

    let centres = [
        (glam::Vec3::new(-2.0, 1.0, 0.0), 1.0f32, 1u32),
        (glam::Vec3::new(0.4, 1.3, 1.4), 1.3, 2),
        (glam::Vec3::new(2.4, 0.9, -0.4), 0.9, 3),
        (glam::Vec3::new(-0.6, 0.7, -2.2), 0.7, 4),
    ];
    for &(centre, size, material) in &centres {
        // Fibonacci, so the samples are even rather than crowded at the poles.
        let count = ((4.0 * std::f32::consts::PI * size * size) / (spacing * spacing)) as usize;
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        for index in 0..count.max(4) {
            let y = 1.0 - 2.0 * (index as f32 + 0.5) / count.max(4) as f32;
            let ring = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * index as f32;
            let normal = glam::Vec3::new(ring * theta.cos(), y, ring * theta.sin());
            surfels.push(vol::relight::Surfel {
                center: (centre + size * normal).to_array(),
                radius,
                normal: normal.to_array(),
                material,
            });
        }
    }

    vol::relight::RelightModel {
        surfels,
        materials: MATERIALS
            .iter()
            .map(|entry| {
                let (albedo, roughness, specular_f0) = *entry;
                vol::relight::Material {
                    albedo,
                    roughness,
                    specular_f0,
                    _padding: 0.0,
                }
            })
            .collect(),
    }
}

/// Poses on a ring around a bounding box, looking at its centre.
///
/// A ring rather than a sphere: a capture is what someone walking around a
/// thing produces, and the top and bottom of an object are the parts a real
/// reconstruction never sees. Including them would flatter the result.
pub fn orbit_poses(
    min: glam::Vec3,
    max: glam::Vec3,
    count: usize,
    elevation: f32,
    fov_y: f32,
    aspect: f32,
) -> Vec<vol::CameraParams> {
    let center = 0.5 * (min + max);
    let radius = (0.5 * (max - min)).length().max(1.0e-6);
    let distance = 1.3 * radius / (0.5 * fov_y).sin().max(1.0e-3);
    (0..count)
        .map(|index| {
            let angle = std::f32::consts::TAU * index as f32 / count as f32;
            let offset = glam::Vec3::new(
                angle.cos() * elevation.cos(),
                elevation.sin(),
                angle.sin() * elevation.cos(),
            );
            vol::CameraParams::looking_at(
                center + distance * offset,
                center,
                fov_y,
                aspect,
                distance + 8.0 * radius,
            )
        })
        .collect()
}

/// Photograph a known scene.
///
/// The environment is deliberately not shown behind the geometry. A background
/// that is the light itself would let a fit read the illumination straight off
/// the sky rather than off the shading, which is not the situation a room is
/// in and would make the measurement worthless.
pub fn photograph(
    renderer: &mut score::Renderer,
    scene: &score::Scene,
    cameras: &[vol::CameraParams],
    width: usize,
    height: usize,
    diffuse_samples: u32,
) -> capture::Capture {
    let frames = renderer.render_views(scene, cameras, diffuse_samples, false);
    let views = frames
        .into_iter()
        .zip(cameras)
        .enumerate()
        .map(|(index, (frame, camera))| capture::View {
            name: format!("truth{index:03}"),
            camera: *camera,
            pixels: frame.iter().map(|t| [t[0], t[1], t[2]]).collect(),
        })
        .collect();
    capture::Capture {
        width,
        height,
        views,
    }
}

/// How far a recovered quantity is from the one it was built from.
#[derive(Clone, Copy, Debug)]
pub struct Error {
    /// The per-channel factor that best aligns the recovery with the truth.
    /// One means the gauge came out right; anything else is the assumption
    /// being wrong by that much, and it is not an error in the fit.
    pub gauge: [f32; 3],
    /// Root mean square difference after the gauge, relative to the mean of
    /// the truth.
    pub relative_rms: f64,
    /// The worst single element, on the same scale.
    pub worst: f64,
}

/// Compare recovered albedo against what the scene was made of.
///
/// `surfel_of` maps each truth surfel onto the recovered material that covers
/// it, which for a fit over the truth's own geometry is just the assignment
/// the clustering produced.
pub fn compare_albedo(
    truth: &vol::relight::RelightModel,
    recovered: &vol::relight::RelightModel,
) -> Error {
    let pairs: Vec<([f32; 3], [f32; 3])> = truth
        .surfels
        .iter()
        .zip(&recovered.surfels)
        .map(|(a, b)| {
            (
                truth.materials[a.material as usize].albedo,
                recovered.materials[b.material as usize].albedo,
            )
        })
        .collect();
    compare_triples(&pairs)
}

/// Compare a recovered light against the real one, through what a Lambertian
/// surface can actually see of it: nine coefficients per channel.
///
/// Comparing the maps texel by texel would score a difference nothing in the
/// scene could have detected, and would fail a perfectly good recovery for
/// putting a window in a slightly different place behind the same amount of
/// light.
pub fn compare_environment(
    truth: &vol::relight::Environment,
    recovered: &vol::relight::Environment,
) -> Error {
    let a = truth.diffuse_irradiance();
    let b = recovered.diffuse_irradiance();
    let pairs: Vec<([f32; 3], [f32; 3])> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| ([x[0], x[1], x[2]], [y[0], y[1], y[2]]))
        .collect();
    compare_triples(&pairs)
}

/// Compare recovered reflectance at normal incidence.
///
/// F0 shares the albedo's gauge problem only partly: it is not traded against
/// the light in the same way, because the lobe and the diffuse term scale
/// together. It is compared through the same machinery for consistency, and the
/// gauge it reports should be near one.
pub fn compare_f0(
    truth: &vol::relight::RelightModel,
    recovered: &vol::relight::RelightModel,
) -> Error {
    let pairs: Vec<([f32; 3], [f32; 3])> = truth
        .surfels
        .iter()
        .zip(&recovered.surfels)
        .map(|(a, b)| {
            (
                truth.materials[a.material as usize].specular_f0,
                recovered.materials[b.material as usize].specular_f0,
            )
        })
        .collect();
    compare_triples(&pairs)
}

/// What came back for one of the materials the scene was built from.
pub struct MaterialReport {
    pub truth: vol::relight::Material,
    pub surfels: usize,
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub specular_f0: [f32; 3],
    /// Mean share of the observed radiance that the lobe accounts for, using
    /// the truth material and the truth light.
    ///
    /// This is the identifiability of the gloss, and it is the number that
    /// makes a roughness error readable. A rough dielectric puts about four
    /// per cent of its brightness in the lobe; no solver recovers a parameter
    /// that moves four per cent of the signal, and reporting its error next to
    /// a metal's is comparing a measurement with a guess.
    pub lobe_share: f32,
}

/// Break the recovery down by the material it was supposed to find.
///
/// An aggregate over a scene whose floor is most of its surfels reports the
/// floor and calls it the scene.
pub fn per_material(
    truth: &vol::relight::RelightModel,
    recovered: &vol::relight::RelightModel,
    environment: &vol::relight::Environment,
    observations: &crate::inverse::decompose::Observations,
) -> Vec<MaterialReport> {
    let count = truth.materials.len();
    let mut surfels = vec![0usize; count];
    let mut albedo = vec![[0.0f64; 3]; count];
    let mut roughness = vec![0.0f64; count];
    let mut specular_f0 = vec![[0.0f64; 3]; count];
    for (a, b) in truth.surfels.iter().zip(&recovered.surfels) {
        let slot = a.material as usize;
        let found = &recovered.materials[b.material as usize];
        surfels[slot] += 1;
        roughness[slot] += found.roughness as f64;
        for channel in 0..3 {
            albedo[slot][channel] += found.albedo[channel] as f64;
            specular_f0[slot][channel] += found.specular_f0[channel] as f64;
        }
    }

    let shares = lobe_shares(truth, environment, observations, count);
    (0..count)
        .map(|slot| {
            let n = surfels[slot].max(1) as f64;
            MaterialReport {
                truth: truth.materials[slot],
                surfels: surfels[slot],
                albedo: [
                    (albedo[slot][0] / n) as f32,
                    (albedo[slot][1] / n) as f32,
                    (albedo[slot][2] / n) as f32,
                ],
                roughness: (roughness[slot] / n) as f32,
                specular_f0: [
                    (specular_f0[slot][0] / n) as f32,
                    (specular_f0[slot][1] / n) as f32,
                    (specular_f0[slot][2] / n) as f32,
                ],
                lobe_share: shares[slot],
            }
        })
        .collect()
}

/// How much of each material's brightness the lobe actually accounts for.
fn lobe_shares(
    truth: &vol::relight::RelightModel,
    environment: &vol::relight::Environment,
    observations: &crate::inverse::decompose::Observations,
    count: usize,
) -> Vec<f32> {
    let specular = vol::relight::SpecularEnvironment::prefilter(
        environment,
        environment.width,
        environment.height,
    );
    let irradiance = environment.diffuse_irradiance();
    let mut lobe = vec![0.0f64; count];
    let mut total = vec![0.0f64; count];
    for (index, surfel) in truth.surfels.iter().enumerate() {
        let slot = surfel.material as usize;
        let material = &truth.materials[slot];
        let normal = glam::Vec3::from(surfel.normal);
        let mut diffuse_only = *material;
        diffuse_only.specular_f0 = [0.0; 3];
        for sample in observations.of(index) {
            let full =
                vol::relight::shade(normal, sample.towards, material, &irradiance, &specular);
            let without = vol::relight::shade(
                normal,
                sample.towards,
                &diffuse_only,
                &irradiance,
                &specular,
            );
            for channel in 0..3 {
                lobe[slot] += (full[channel] - without[channel]).max(0.0) as f64;
                total[slot] += full[channel].max(0.0) as f64;
            }
        }
    }
    (0..count)
        .map(|slot| {
            if total[slot] > 1.0e-9 {
                (lobe[slot] / total[slot]) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Mean absolute roughness error, weighted by nothing.
///
/// Roughness has no gauge: it is not traded off against the light the way
/// albedo is, so it can be compared directly. It is also the parameter with the
/// least evidence behind it — a surface seen from one direction says nothing
/// about its own gloss — so read it next to the share of materials that had any
/// angular spread at all.
pub fn compare_roughness(
    truth: &vol::relight::RelightModel,
    recovered: &vol::relight::RelightModel,
) -> f64 {
    let mut error = 0.0f64;
    let mut count = 0usize;
    for (a, b) in truth.surfels.iter().zip(&recovered.surfels) {
        let expected = truth.materials[a.material as usize].roughness;
        let found = recovered.materials[b.material as usize].roughness;
        error += (found - expected).abs() as f64;
        count += 1;
    }
    error / count.max(1) as f64
}

/// Least-squares per-channel gauge, then the error that survives it.
fn compare_triples(pairs: &[([f32; 3], [f32; 3])]) -> Error {
    let mut gauge = [1.0f32; 3];
    for (channel, factor) in gauge.iter_mut().enumerate() {
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for &(truth, recovered) in pairs {
            numerator += truth[channel] as f64 * recovered[channel] as f64;
            denominator += (recovered[channel] as f64).powi(2);
        }
        if denominator > 1.0e-12 {
            *factor = (numerator / denominator) as f32;
        }
    }
    let mut error = 0.0f64;
    let mut scale = 0.0f64;
    let mut worst = 0.0f64;
    for &(truth, recovered) in pairs {
        for channel in 0..3 {
            let difference = (gauge[channel] * recovered[channel] - truth[channel]).abs() as f64;
            error += difference * difference;
            worst = worst.max(difference);
            scale += truth[channel].abs() as f64;
        }
    }
    let count = (pairs.len() * 3).max(1) as f64;
    let mean = (scale / count).max(1.0e-9);
    Error {
        gauge,
        relative_rms: (error / count).sqrt() / mean,
        worst: worst / mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_recovery_up_to_scale_reports_the_scale_and_no_error() {
        // The gauge is an assumption, so a recovery that differs only by it
        // has to come back as correct with the factor stated, not as wrong.
        let pairs: Vec<([f32; 3], [f32; 3])> = (0..32)
            .map(|index| {
                let value = 0.1 + 0.02 * index as f32;
                ([value, 0.5 * value, 0.25 * value], [2.0 * value; 3])
            })
            .collect();
        let error = compare_triples(&pairs);
        assert!((error.gauge[0] - 0.5).abs() < 1.0e-4, "{:?}", error.gauge);
        assert!((error.gauge[1] - 0.25).abs() < 1.0e-4);
        assert!((error.gauge[2] - 0.125).abs() < 1.0e-4);
        assert!(error.relative_rms < 1.0e-4, "{}", error.relative_rms);
    }

    #[test]
    fn a_recovery_that_is_wrong_in_shape_is_not_rescued_by_the_gauge() {
        // The gauge is one number per channel, so it can absorb a brightness
        // but not a pattern. Two materials swapped stay swapped.
        let pairs = vec![
            ([0.8, 0.1, 0.1], [0.1, 0.8, 0.1]),
            ([0.1, 0.8, 0.1], [0.8, 0.1, 0.1]),
        ];
        let error = compare_triples(&pairs);
        assert!(error.relative_rms > 0.5, "{}", error.relative_rms);
    }

    #[test]
    fn the_orbit_looks_at_what_it_goes_round() {
        let min = glam::Vec3::new(-1.0, -0.5, -2.0);
        let max = glam::Vec3::new(1.0, 1.5, 2.0);
        let center = 0.5 * (min + max);
        for camera in orbit_poses(min, max, 12, 0.4, 0.9, 1.5) {
            let position = glam::Vec3::from(camera.cam_position);
            let forward = glam::Quat::from_array(camera.cam_orientation) * glam::Vec3::Z;
            assert!(
                forward.dot((center - position).normalize()) > 0.999,
                "a pose is not pointed at the centre"
            );
            assert!(
                (position - center).length() > (0.5 * (max - min)).length(),
                "a pose is inside the object"
            );
        }
    }
}
