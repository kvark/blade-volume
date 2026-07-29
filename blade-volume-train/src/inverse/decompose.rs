//! Splitting what was observed into a material and a light.
//!
//! # Why this cannot be done by fitting alone
//!
//! The renderer's diffuse term is `albedo * E(normal)`. With one free albedo
//! per surfel the product is under-determined at every point: pick any light
//! you like, set `albedo = observed / E(normal)`, and the fit is exact. That
//! solution re-renders the capture perfectly and has recovered nothing — it is
//! the identity in disguise, with the illumination painted onto the surface.
//!
//! What breaks the tie is **sharing**. When many surfels are made of the same
//! material, one albedo has to explain observations under many different
//! normals, and only a light close to the real one can do that: a wall facing
//! a window is brighter than the same wall facing away, and the ratio between
//! them is a fact about the light rather than about the paint.
//!
//! So the number of materials is the knob that trades re-rendering accuracy
//! against having decomposed anything at all. It is exposed rather than
//! chosen, because the sweep across it is the measurement.
//!
//! # What is identifiable
//!
//! A Lambertian fit sees the environment only through nine spherical-harmonic
//! coefficients per channel. Everything else about the map — where exactly the
//! window is, how hard its edge is — is invisible to it. The map produced here
//! is therefore the smoothest non-negative one consistent with what was seen,
//! and its resolution is a rendering convenience rather than a claim.

use crate::inverse::{capture, score};
use blade_volume as vol;
use std::thread;

/// How the split is set up.
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    /// Materials the surfels are clustered into. Zero gives every surfel its
    /// own, which fits the capture best and decomposes nothing.
    pub materials: usize,
    /// Equirectangular size of the recovered environment.
    pub environment_width: usize,
    /// Alternations between solving for albedo and solving for light.
    pub iterations: usize,
    /// Views closer to the surface than this cosine are treated as grazing and
    /// ignored: at a glancing angle one surfel covers a sliver of a pixel and
    /// the colour read from it is mostly its neighbour's.
    pub min_facing: f32,
    /// The brightest recovered albedo, which anchors the global scale.
    ///
    /// Albedo and light trade off exactly: doubling one and halving the other
    /// renders the same image. Something has to fix it, and "the most
    /// reflective thing in the scene is about this bright" is the assumption
    /// with the fewest moving parts.
    pub brightest_albedo: f32,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            materials: 256,
            environment_width: 32,
            iterations: 24,
            min_facing: 0.15,
            brightest_albedo: 0.8,
        }
    }
}

/// What each surfel was seen to be, and how well it was seen.
pub struct Observations {
    /// Mean observed radiance, weighted by how square-on the view was.
    pub mean: Vec<[f32; 3]>,
    /// Total weight behind that mean. Zero means the surfel was never seen,
    /// and nothing about it is recoverable.
    pub weight: Vec<f32>,
}

impl Observations {
    pub fn seen(&self) -> usize {
        self.weight.iter().filter(|&&w| w > 0.0).count()
    }
}

/// Gather what every surfel looked like, from the views that could see it.
///
/// Visibility is decided with a depth buffer per view rather than by ray
/// casting: each disc is splatted, the nearest wins, and a surfel further back
/// than the winner along its own pixel is occluded. Without this a surfel
/// behind a wall is fitted against the colour of the wall, which produces a
/// material for a surface nobody ever saw.
pub fn observe(
    model: &vol::relight::RelightModel,
    capture: &capture::Capture,
    views: &[usize],
    min_facing: f32,
) -> Observations {
    let count = model.surfels.len();
    let mut total = vec![[0.0f32; 3]; count];
    let mut weight = vec![0.0f32; count];
    let (width, height) = (capture.width, capture.height);
    let mut depth = vec![f32::INFINITY; width * height];

    for &view_index in views {
        let view = &capture.views[view_index];
        let camera = &view.camera;
        // Pixels per world unit at unit distance, for the disc's footprint.
        let focal = 0.5 * height as f32 / (0.5 * camera.fov[1]).tan();
        depth.fill(f32::INFINITY);
        for surfel in &model.surfels {
            let center = glam::Vec3::from(surfel.center);
            let Some((pixel, distance)) = capture::project(camera, width, height, center) else {
                continue;
            };
            splat(
                &mut depth,
                width,
                height,
                pixel,
                (surfel.radius * focal / distance).clamp(0.5, 64.0),
                distance,
            );
        }

        for (index, surfel) in model.surfels.iter().enumerate() {
            let center = glam::Vec3::from(surfel.center);
            let normal = glam::Vec3::from(surfel.normal);
            let Some((pixel, distance)) = capture::project(camera, width, height, center) else {
                continue;
            };
            let towards = (glam::Vec3::from(camera.cam_position) - center).normalize_or_zero();
            let facing = normal.dot(towards);
            if facing < min_facing {
                continue;
            }
            let x = pixel[0] as usize;
            let y = pixel[1] as usize;
            if pixel[0] < 0.0 || pixel[1] < 0.0 || x >= width || y >= height {
                continue;
            }
            // The disc has depth of its own, so being behind the winner by
            // less than its radius still counts as being the winner.
            if distance > depth[y * width + x] + surfel.radius {
                continue;
            }
            let observed = view.pixels[y * width + x];
            for channel in 0..3 {
                total[index][channel] += observed[channel] * facing;
            }
            weight[index] += facing;
        }
    }

    let mean = total
        .iter()
        .zip(&weight)
        .map(|(sum, &w)| {
            if w > 0.0 {
                [sum[0] / w, sum[1] / w, sum[2] / w]
            } else {
                [0.0; 3]
            }
        })
        .collect();
    Observations { mean, weight }
}

fn splat(
    depth: &mut [f32],
    width: usize,
    height: usize,
    pixel: [f32; 2],
    radius: f32,
    distance: f32,
) {
    let min_x = ((pixel[0] - radius).floor() as isize).max(0) as usize;
    let min_y = ((pixel[1] - radius).floor() as isize).max(0) as usize;
    let max_x = ((pixel[0] + radius).ceil() as isize).min(width as isize - 1);
    let max_y = ((pixel[1] + radius).ceil() as isize).min(height as isize - 1);
    if max_x < 0 || max_y < 0 {
        return;
    }
    let radius_squared = radius * radius;
    for y in min_y..=max_y as usize {
        for x in min_x..=max_x as usize {
            let dx = x as f32 + 0.5 - pixel[0];
            let dy = y as f32 + 0.5 - pixel[1];
            if dx * dx + dy * dy > radius_squared {
                continue;
            }
            let slot = &mut depth[y * width + x];
            if distance < *slot {
                *slot = distance;
            }
        }
    }
}

// ------------------------------------------------------------------ materials

/// Group surfels by the colour they were seen to be, ignoring how bright.
///
/// Chromaticity and not intensity, because intensity is what the light is
/// supposed to explain. Two walls of the same paint under different
/// illumination must land in the same cluster or there is nothing for the
/// light to account for.
fn cluster_by_chromaticity(
    observations: &Observations,
    clusters: usize,
    seed: u64,
) -> (Vec<u32>, usize) {
    let count = observations.mean.len();
    if clusters == 0 || clusters >= count {
        return ((0..count as u32).collect(), count);
    }
    let chromaticity: Vec<[f32; 3]> = observations
        .mean
        .iter()
        .map(|rgb| {
            let sum = rgb[0] + rgb[1] + rgb[2];
            if sum > 1.0e-6 {
                [rgb[0] / sum, rgb[1] / sum, rgb[2] / sum]
            } else {
                [1.0 / 3.0; 3]
            }
        })
        .collect();

    // Seeded stride rather than random draws: the same cloud has to cluster
    // the same way twice or a sweep over cluster counts is measuring noise.
    let mut centres: Vec<[f32; 3]> = Vec::with_capacity(clusters);
    let stride = (count / clusters).max(1);
    let mut cursor = (seed as usize) % count;
    for _ in 0..clusters {
        centres.push(chromaticity[cursor]);
        cursor = (cursor + stride) % count;
    }

    let mut assignment = vec![0u32; count];
    for _ in 0..12 {
        let mut sums = vec![[0.0f64; 3]; clusters];
        let mut counts = vec![0u32; clusters];
        for (index, colour) in chromaticity.iter().enumerate() {
            let mut best = 0usize;
            let mut best_distance = f32::INFINITY;
            for (slot, centre) in centres.iter().enumerate() {
                let distance = (0..3).map(|c| (colour[c] - centre[c]).powi(2)).sum::<f32>();
                if distance < best_distance {
                    best_distance = distance;
                    best = slot;
                }
            }
            assignment[index] = best as u32;
            counts[best] += 1;
            for channel in 0..3 {
                sums[best][channel] += colour[channel] as f64;
            }
        }
        for (slot, centre) in centres.iter_mut().enumerate() {
            if counts[slot] > 0 {
                for channel in 0..3 {
                    centre[channel] = (sums[slot][channel] / counts[slot] as f64) as f32;
                }
            }
        }
    }
    (assignment, clusters)
}

// -------------------------------------------------------------------- the fit

/// The Lambertian response of one equirectangular texel, per unit radiance.
///
/// `max(0, n . d) * solid_angle / PI` — the cosine lobe with the BRDF's
/// `1 / PI` folded in, so the sum over texels is directly the outgoing
/// radiance of a unit albedo, matching what the shader computes from its nine
/// coefficients. The exact lobe is used rather than its band-limited version:
/// it is non-negative, which is what keeps the update below from ever asking
/// for a negative light.
struct Kernel {
    directions: Vec<glam::Vec3>,
    solid_angles: Vec<f32>,
    width: usize,
    height: usize,
}

impl Kernel {
    fn new(width: usize, height: usize) -> Self {
        let mut directions = Vec::with_capacity(width * height);
        let mut solid_angles = Vec::with_capacity(width * height);
        let base =
            (2.0 * std::f32::consts::PI / width as f32) * (std::f32::consts::PI / height as f32);
        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            let row = base * (std::f32::consts::PI * v).sin();
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                directions.push(vol::relight::equirect_direction(u, v));
                solid_angles.push(row);
            }
        }
        Self {
            directions,
            solid_angles,
            width,
            height,
        }
    }

    fn texels(&self) -> usize {
        self.directions.len()
    }

    /// Weights of every texel for one normal, written into `out`.
    fn weights(&self, normal: glam::Vec3, out: &mut [f32]) {
        for (index, direction) in self.directions.iter().enumerate() {
            let cosine = normal.dot(*direction);
            out[index] = if cosine > 0.0 {
                cosine * self.solid_angles[index] * std::f32::consts::FRAC_1_PI
            } else {
                0.0
            };
        }
    }
}

/// A recovered scene, and what it cost to recover.
pub struct Decomposition {
    pub scene: score::Scene,
    /// Material index per surfel, before the table was built.
    pub assignment: Vec<u32>,
    /// Root mean square residual of the fit, in linear radiance.
    pub residual: f32,
    /// Surfels no view ever saw. They keep a default material and are the
    /// honest measure of how much of the model is unsupported by evidence.
    pub unseen: usize,
}

/// Alternate between the albedo that best explains the observations under the
/// current light, and the light that best explains them under the current
/// albedo.
///
/// Both halves are convex in their own variable. The light step is a
/// multiplicative update, which keeps every texel non-negative without a
/// constraint solver and cannot step past zero.
pub fn fit(
    model: &vol::relight::RelightModel,
    observations: &Observations,
    options: FitOptions,
) -> Decomposition {
    let kernel = Kernel::new(options.environment_width, options.environment_width / 2);
    let texels = kernel.texels();
    let count = model.surfels.len();
    let (assignment, clusters) = cluster_by_chromaticity(observations, options.materials, 0x5EED);

    // Only surfels that were actually seen constrain anything.
    let active: Vec<usize> = (0..count)
        .filter(|&index| observations.weight[index] > 0.0)
        .collect();

    // The cosine weights depend only on the normal, and are reused every
    // iteration by both halves of the alternation.
    let mut response = vec![0.0f32; active.len() * texels];
    for (slot, &index) in active.iter().enumerate() {
        kernel.weights(
            glam::Vec3::from(model.surfels[index].normal),
            &mut response[slot * texels..(slot + 1) * texels],
        );
    }

    // Start from a uniform sky bright enough that a mid albedo explains the
    // mean observation. Starting uniform is also the prior: the multiplicative
    // update moves every texel away from the same place, so it only puts
    // structure where the observations asked for it.
    let mean_observed = mean_radiance(observations, &active);
    let mut light = vec![[0.0f32; 3]; texels];
    for texel in light.iter_mut() {
        for channel in 0..3 {
            texel[channel] = (mean_observed[channel] / 0.5).max(1.0e-4);
        }
    }

    let mut albedo = vec![[0.5f32; 3]; clusters];
    let mut shade = vec![[0.0f32; 3]; active.len()];
    let mut residual = 0.0f32;
    for _ in 0..options.iterations.max(1) {
        evaluate_shade(&response, &light, texels, &mut shade);
        solve_albedo(
            &shade,
            observations,
            &active,
            &assignment,
            clusters,
            options.brightest_albedo,
            &mut albedo,
        );
        residual = update_light(
            &response,
            &shade,
            observations,
            &active,
            &assignment,
            &albedo,
            texels,
            &mut light,
        );
    }

    // Two gauges have to be fixed by assumption, because no image can fix
    // them: one scale per channel, and one overall. Both are applied to the
    // albedo and undone in the light, so the product — the only thing that was
    // ever observed — is untouched by either.
    neutralize_light(&mut albedo, &mut light, &kernel);
    let brightest = albedo
        .iter()
        .flat_map(|rgb| rgb.iter().copied())
        .fold(0.0f32, f32::max);
    if brightest > 1.0e-6 {
        let scale = options.brightest_albedo / brightest;
        for value in albedo.iter_mut().flat_map(|rgb| rgb.iter_mut()) {
            *value *= scale;
        }
        for value in light.iter_mut().flat_map(|texel| texel.iter_mut()) {
            *value /= scale;
        }
    }

    let materials = albedo
        .iter()
        .map(|rgb| vol::relight::Material {
            albedo: *rgb,
            // A diffuse fit says nothing about gloss. Claiming a roughness
            // would be inventing evidence; the dielectric default is the
            // smallest thing that is not a claim.
            roughness: 1.0,
            specular_f0: [0.04; 3],
            _padding: 0.0,
        })
        .collect();
    let mut surfels = model.surfels.clone();
    for (surfel, &material) in surfels.iter_mut().zip(&assignment) {
        surfel.material = material;
    }

    Decomposition {
        scene: score::Scene {
            model: vol::relight::RelightModel { surfels, materials },
            environment: vol::relight::Environment {
                width: kernel.width,
                height: kernel.height,
                texels: light,
            },
        },
        assignment,
        residual,
        unseen: count - active.len(),
    }
}

/// Move the colour of the illuminant into the materials.
///
/// A red wall under white light and a white wall under red light produce the
/// same photograph. Nothing in the images distinguishes them, so one of the
/// two has to be assumed, and this assumes the light: its mean over the sphere
/// is made neutral and the per-channel gain that took it there is handed to
/// the albedo.
///
/// The assumption is wrong in exactly the case you would expect — a room lit
/// by tungsten returns materials that are too warm — and it is stated here
/// rather than buried because it is not recoverable, only chosen.
fn neutralize_light(albedo: &mut [[f32; 3]], light: &mut [[f32; 3]], kernel: &Kernel) {
    let mut mean = [0.0f64; 3];
    let mut weight = 0.0f64;
    for (texel, &solid_angle) in kernel.solid_angles.iter().enumerate() {
        weight += solid_angle as f64;
        for channel in 0..3 {
            mean[channel] += light[texel][channel] as f64 * solid_angle as f64;
        }
    }
    if weight <= 0.0 {
        return;
    }
    for value in mean.iter_mut() {
        *value /= weight;
    }
    let neutral = (mean[0] + mean[1] + mean[2]) / 3.0;
    if neutral <= 1.0e-9 {
        return;
    }
    for channel in 0..3 {
        if mean[channel] <= 1.0e-9 {
            continue;
        }
        // Scale the light to neutral; the albedo takes up the slack.
        let gain = (neutral / mean[channel]) as f32;
        for texel in light.iter_mut() {
            texel[channel] *= gain;
        }
        for rgb in albedo.iter_mut() {
            rgb[channel] /= gain;
        }
    }
}

fn mean_radiance(observations: &Observations, active: &[usize]) -> [f32; 3] {
    let mut total = [0.0f64; 3];
    for &index in active {
        for (sum, value) in total.iter_mut().zip(observations.mean[index]) {
            *sum += value as f64;
        }
    }
    let count = active.len().max(1) as f64;
    [
        (total[0] / count) as f32,
        (total[1] / count) as f32,
        (total[2] / count) as f32,
    ]
}

fn evaluate_shade(response: &[f32], light: &[[f32; 3]], texels: usize, shade: &mut [[f32; 3]]) {
    let chunk = (shade.len() / thread::available_parallelism().map_or(1, |n| n.get())).max(1);
    thread::scope(|scope| {
        for (block, rows) in shade.chunks_mut(chunk).enumerate() {
            let response = &response[block * chunk * texels..];
            scope.spawn(move || {
                for (slot, out) in rows.iter_mut().enumerate() {
                    let weights = &response[slot * texels..(slot + 1) * texels];
                    let mut total = [0.0f32; 3];
                    for (texel, &weight) in weights.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        for channel in 0..3 {
                            total[channel] += weight * light[texel][channel];
                        }
                    }
                    *out = total;
                }
            });
        }
    });
}

fn solve_albedo(
    shade: &[[f32; 3]],
    observations: &Observations,
    active: &[usize],
    assignment: &[u32],
    clusters: usize,
    ceiling: f32,
    albedo: &mut [[f32; 3]],
) {
    let mut numerator = vec![[0.0f64; 3]; clusters];
    let mut denominator = vec![[0.0f64; 3]; clusters];
    for (slot, &index) in active.iter().enumerate() {
        let cluster = assignment[index] as usize;
        let weight = observations.weight[index] as f64;
        for channel in 0..3 {
            let e = shade[slot][channel] as f64;
            numerator[cluster][channel] += weight * e * observations.mean[index][channel] as f64;
            denominator[cluster][channel] += weight * e * e;
        }
    }
    for cluster in 0..clusters {
        for channel in 0..3 {
            let value = if denominator[cluster][channel] > 1.0e-12 {
                numerator[cluster][channel] / denominator[cluster][channel]
            } else {
                0.5
            };
            albedo[cluster][channel] = (value as f32).clamp(0.005, ceiling);
        }
    }
}

/// One multiplicative step towards the light that best explains the residual.
///
/// `L <- L * (sum of what was seen) / (sum of what is predicted)`, both sums
/// weighted by how strongly each texel reaches each surfel. Every quantity is
/// non-negative, so the light stays a light. Returns the root mean square
/// residual before the step.
#[allow(clippy::too_many_arguments)]
fn update_light(
    response: &[f32],
    shade: &[[f32; 3]],
    observations: &Observations,
    active: &[usize],
    assignment: &[u32],
    albedo: &[[f32; 3]],
    texels: usize,
    light: &mut [[f32; 3]],
) -> f32 {
    /// What one thread contributes: the light each texel was seen to give,
    /// the light it is currently predicted to give, and the squared error.
    type Partial = (Vec<[f64; 3]>, Vec<[f64; 3]>, f64);

    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = (active.len() / threads).max(1);
    let mut partials: Vec<Partial> = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (block, slots) in active.chunks(chunk).enumerate() {
            let response = &response[block * chunk * texels..];
            let shade = &shade[block * chunk..];
            handles.push(scope.spawn(move || {
                let mut seen = vec![[0.0f64; 3]; texels];
                let mut predicted = vec![[0.0f64; 3]; texels];
                let mut error = 0.0f64;
                for (slot, &index) in slots.iter().enumerate() {
                    let weights = &response[slot * texels..(slot + 1) * texels];
                    let rho = albedo[assignment[index] as usize];
                    let confidence = observations.weight[index] as f64;
                    for channel in 0..3 {
                        let difference = (rho[channel] * shade[slot][channel]
                            - observations.mean[index][channel])
                            as f64;
                        error += confidence * difference * difference;
                    }
                    for (texel, &weight) in weights.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        let weight = weight as f64 * confidence;
                        for channel in 0..3 {
                            let rho = rho[channel] as f64;
                            seen[texel][channel] +=
                                weight * rho * observations.mean[index][channel] as f64;
                            predicted[texel][channel] +=
                                weight * rho * rho * shade[slot][channel] as f64;
                        }
                    }
                }
                (seen, predicted, error)
            }));
        }
        for handle in handles {
            partials.push(handle.join().unwrap());
        }
    });

    let mut seen = vec![[0.0f64; 3]; texels];
    let mut predicted = vec![[0.0f64; 3]; texels];
    let mut error = 0.0f64;
    let mut confidence = 0.0f64;
    for partial in &partials {
        let (ref partial_seen, ref partial_predicted, partial_error) = *partial;
        for (total, part) in seen.iter_mut().zip(partial_seen) {
            for (channel, value) in total.iter_mut().zip(part) {
                *channel += value;
            }
        }
        for (total, part) in predicted.iter_mut().zip(partial_predicted) {
            for (channel, value) in total.iter_mut().zip(part) {
                *channel += value;
            }
        }
        error += partial_error;
    }
    for &index in active {
        confidence += observations.weight[index] as f64;
    }

    for texel in 0..texels {
        for channel in 0..3 {
            let ratio = if predicted[texel][channel] > 1.0e-14 {
                seen[texel][channel] / predicted[texel][channel]
            } else {
                1.0
            };
            // A step of more than a few times per iteration is the update
            // chasing a texel nothing constrains, and it oscillates.
            let ratio = ratio.clamp(0.25, 4.0) as f32;
            light[texel][channel] = (light[texel][channel] * ratio).max(0.0);
        }
    }
    ((error / (3.0 * confidence.max(1.0e-6))).max(0.0) as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere of surfels, one material, under a light from one side.
    fn sphere(count: usize, albedo: [f32; 3]) -> vol::relight::RelightModel {
        let mut surfels = Vec::with_capacity(count);
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        for index in 0..count {
            let y = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
            let radius = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * index as f32;
            let normal = glam::Vec3::new(radius * theta.cos(), y, radius * theta.sin()).normalize();
            surfels.push(vol::relight::Surfel {
                center: normal.to_array(),
                radius: 0.05,
                normal: normal.to_array(),
                material: 0,
            });
        }
        vol::relight::RelightModel {
            surfels,
            materials: vec![vol::relight::Material {
                albedo,
                roughness: 1.0,
                specular_f0: [0.04; 3],
                _padding: 0.0,
            }],
        }
    }

    /// What the renderer would show, computed the same way the fit models it.
    fn render(
        model: &vol::relight::RelightModel,
        environment: &vol::relight::Environment,
        albedo: [f32; 3],
    ) -> Observations {
        let kernel = Kernel::new(environment.width, environment.height);
        let mut weights = vec![0.0f32; kernel.texels()];
        let mut mean = Vec::with_capacity(model.surfels.len());
        for surfel in &model.surfels {
            kernel.weights(glam::Vec3::from(surfel.normal), &mut weights);
            let mut radiance = [0.0f32; 3];
            for (texel, &weight) in weights.iter().enumerate() {
                for (value, source) in radiance.iter_mut().zip(environment.texels[texel]) {
                    *value += weight * source;
                }
            }
            for (value, gain) in radiance.iter_mut().zip(albedo) {
                *value *= gain;
            }
            mean.push(radiance);
        }
        let weight = vec![1.0f32; model.surfels.len()];
        Observations { mean, weight }
    }

    #[test]
    fn one_shared_material_recovers_where_the_light_came_from() {
        // The whole claim of this module in one test: a single albedo over
        // many normals leaves the light as the only free explanation, so it
        // has to be found rather than absorbed.
        let truth_albedo = [0.6, 0.5, 0.4];
        let model = sphere(4096, truth_albedo);
        let mut environment = vol::relight::Environment::uniform([0.05; 3], 32, 16);
        // A bright patch to the east, which is what has to be recovered.
        let kernel = Kernel::new(32, 16);
        let east = glam::Vec3::X;
        for (texel, direction) in kernel.directions.iter().enumerate() {
            if direction.dot(east) > 0.9 {
                environment.texels[texel] = [4.0; 3];
            }
        }
        let observations = render(&model, &environment, truth_albedo);

        let fitted = fit(
            &model,
            &observations,
            FitOptions {
                materials: 1,
                environment_width: 32,
                iterations: 60,
                min_facing: 0.0,
                brightest_albedo: 0.6,
            },
        );

        // The recovered light is brightest towards the east.
        let recovered = &fitted.scene.environment;
        let mut brightest = glam::Vec3::ZERO;
        let mut best = f32::NEG_INFINITY;
        for (texel, direction) in kernel.directions.iter().enumerate() {
            let luminance: f32 = recovered.texels[texel].iter().sum();
            if luminance > best {
                best = luminance;
                brightest = *direction;
            }
        }
        assert!(
            brightest.dot(east) > 0.7,
            "the light was found at {brightest:?}, not to the east"
        );

        // And the albedo is close to what it was, not to the shading.
        let recovered_albedo = fitted.scene.model.materials[0].albedo;
        for channel in 0..3 {
            assert!(
                (recovered_albedo[channel] / truth_albedo[channel] - 1.0).abs() < 0.25,
                "albedo came back {recovered_albedo:?} against {truth_albedo:?}"
            );
        }
    }

    #[test]
    fn the_ratio_between_two_materials_survives_the_gauge() {
        // Neutralising the light is an assumption, and a test that leans on it
        // is testing the assumption. The ratio of two albedos does not: both
        // are scaled by the same per-channel gain, whatever it turns out to
        // be, so it cancels. This is the part of the decomposition that is
        // recoverable rather than chosen.
        let first = [0.7, 0.35, 0.2];
        let second = [0.2, 0.4, 0.6];
        let mut model = sphere(4096, first);
        let mut environment = vol::relight::Environment::uniform([0.4; 3], 16, 8);
        let kernel = Kernel::new(16, 8);
        for (texel, direction) in kernel.directions.iter().enumerate() {
            if direction.dot(glam::Vec3::new(0.0, 1.0, 0.0)) > 0.8 {
                // A coloured source, so the gauge really is exercised.
                environment.texels[texel] = [3.0, 2.0, 1.0];
            }
        }
        // Half the sphere is the other material.
        let mut mean = Vec::with_capacity(model.surfels.len());
        let mut weights = vec![0.0f32; kernel.texels()];
        for (index, surfel) in model.surfels.iter_mut().enumerate() {
            let albedo = if index % 2 == 0 { first } else { second };
            surfel.material = (index % 2) as u32;
            kernel.weights(glam::Vec3::from(surfel.normal), &mut weights);
            let mut radiance = [0.0f32; 3];
            for (texel, &weight) in weights.iter().enumerate() {
                for (value, source) in radiance.iter_mut().zip(environment.texels[texel]) {
                    *value += weight * source;
                }
            }
            for (value, gain) in radiance.iter_mut().zip(albedo) {
                *value *= gain;
            }
            mean.push(radiance);
        }
        let observations = Observations {
            weight: vec![1.0f32; mean.len()],
            mean,
        };

        let fitted = fit(
            &model,
            &observations,
            FitOptions {
                materials: 2,
                environment_width: 16,
                iterations: 80,
                min_facing: 0.0,
                brightest_albedo: 0.8,
            },
        );
        assert_eq!(fitted.scene.model.materials.len(), 2);
        // Whichever cluster is which, the two ratios have to match the truth.
        let recovered: Vec<[f32; 3]> = fitted
            .scene
            .model
            .materials
            .iter()
            .map(|m| m.albedo)
            .collect();
        let ordered = if recovered[0][0] > recovered[1][0] {
            [recovered[0], recovered[1]]
        } else {
            [recovered[1], recovered[0]]
        };
        for channel in 0..3 {
            let truth = first[channel] / second[channel];
            let found = ordered[0][channel] / ordered[1][channel];
            assert!(
                (found / truth - 1.0).abs() < 0.15,
                "channel {channel}: ratio {found:.3} against {truth:.3}, \
                 from {ordered:?}"
            );
        }
    }

    #[test]
    fn a_material_per_surfel_explains_everything_and_recovers_nothing() {
        // The degenerate end, stated as a test so it cannot be mistaken for a
        // result: with albedo free everywhere the residual goes to nothing and
        // the light stays wherever it started.
        let truth_albedo = [0.6, 0.5, 0.4];
        let model = sphere(1024, truth_albedo);
        let mut environment = vol::relight::Environment::uniform([0.05; 3], 16, 8);
        let kernel = Kernel::new(16, 8);
        for (texel, direction) in kernel.directions.iter().enumerate() {
            if direction.dot(glam::Vec3::X) > 0.9 {
                environment.texels[texel] = [4.0; 3];
            }
        }
        let observations = render(&model, &environment, truth_albedo);
        let fitted = fit(
            &model,
            &observations,
            FitOptions {
                materials: 0,
                environment_width: 16,
                iterations: 30,
                min_facing: 0.0,
                brightest_albedo: 0.8,
            },
        );
        assert!(
            fitted.residual < 1.0e-3,
            "an unconstrained albedo left a residual of {}",
            fitted.residual
        );
        let recovered = &fitted.scene.environment;
        let east: f32 = kernel
            .directions
            .iter()
            .enumerate()
            .filter(|&(_, d)| d.dot(glam::Vec3::X) > 0.9)
            .map(|(t, _)| recovered.texels[t][0])
            .sum();
        let west: f32 = kernel
            .directions
            .iter()
            .enumerate()
            .filter(|&(_, d)| d.dot(glam::Vec3::X) < -0.9)
            .map(|(t, _)| recovered.texels[t][0])
            .sum();
        assert!(
            east < 3.0 * west,
            "the free fit found the light anyway: east {east}, west {west}"
        );
    }
}
