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

use crate::inverse::{capture, score, visibility};
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
    /// Rounds in which roughness and F0 are re-chosen against the current
    /// light. Zero leaves every surface a rough dielectric, which is the
    /// albedo-only fit.
    pub specular_rounds: usize,
    /// How much better a lobe hypothesis has to be before it is believed.
    ///
    /// A fraction of the residual the default explains. Nothing here is a
    /// tuning knob for accuracy — it is the threshold below which the data
    /// cannot tell the difference, and calling a coin toss either way is worse
    /// than declining to call it.
    ///
    /// A rough dielectric puts about four per cent of its brightness in the
    /// lobe. At that level a metal and a dielectric fit a surfel equally well,
    /// and without a margin roughly a third of a matte floor came back as
    /// metal — each of those surfels losing its albedo completely, which cost
    /// twice as much as the metals it was trying to find were worth.
    pub lobe_margin: f64,
    /// Rounds of indirect light folded into the fit.
    ///
    /// Only meaningful with shadowing, and close to mandatory with it: a patch
    /// the model says is fully shadowed is not black in any real photograph,
    /// because whatever shadows it also lights it. Zero leaves that light
    /// unexplained, and the fit puts it in the sky instead.
    pub bounces: usize,
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
            materials: 0,
            environment_width: 32,
            iterations: 24,
            // One material per surfel. Sharing is a prior about the scene
            // rather than something the photographs said, and it costs both
            // the albedo and the re-rendering to buy a better-shaped light.
            specular_rounds: 3,
            lobe_margin: 0.15,
            bounces: 3,
            min_facing: 0.15,
            brightest_albedo: 0.8,
        }
    }
}

/// One surfel, seen once.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Observed radiance, linear.
    pub radiance: [f32; 3],
    /// Unit direction from the surface towards the camera.
    pub towards: glam::Vec3,
    /// How square-on the view was, `n . towards`. Used as the weight: a
    /// glancing view of a surfel reads mostly its neighbour.
    pub facing: f32,
}

/// What every surfel was seen to be, kept per view rather than averaged.
///
/// The average was the earlier design and it is fatal to anything beyond a
/// Lambertian albedo. A diffuse surface looks the same from every direction, so
/// for albedo alone the mean is a sufficient statistic and nothing is lost. A
/// lobe is not: roughness and F0 are recoverable *only* from how one surfel's
/// appearance changes between views, and averaging destroys exactly that signal
/// before the fit sees it.
///
/// Stored as CSR by surfel, because every stage wants one surfel's samples
/// together and the count per surfel varies from zero to the whole capture.
pub struct Observations {
    pub samples: Vec<Sample>,
    /// `offsets[i]..offsets[i + 1]` are surfel `i`'s samples.
    pub offsets: Vec<u32>,
}

impl Observations {
    pub fn of(&self, surfel: usize) -> &[Sample] {
        let begin = self.offsets[surfel] as usize;
        let end = self.offsets[surfel + 1] as usize;
        &self.samples[begin..end]
    }

    pub fn surfels(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn seen(&self) -> usize {
        (0..self.surfels())
            .filter(|&index| !self.of(index).is_empty())
            .count()
    }

    /// Total weight behind one surfel; zero when nothing saw it.
    pub fn weight(&self, surfel: usize) -> f32 {
        self.of(surfel).iter().map(|s| s.facing).sum()
    }

    /// The weighted mean of what a surfel was seen to be.
    ///
    /// Still the right sufficient statistic for the view-independent half of
    /// the model — but it is derived here rather than being all that was kept.
    pub fn mean(&self, surfel: usize) -> [f32; 3] {
        let mut total = [0.0f32; 3];
        let mut weight = 0.0f32;
        for sample in self.of(surfel) {
            for (sum, value) in total.iter_mut().zip(sample.radiance) {
                *sum += value * sample.facing;
            }
            weight += sample.facing;
        }
        if weight > 0.0 {
            for value in total.iter_mut() {
                *value /= weight;
            }
        }
        total
    }

    /// How many views saw a surfel from meaningfully different angles.
    ///
    /// A lobe cannot be fitted from one direction, however many times it was
    /// photographed from there. Reported so a claim about roughness can be
    /// qualified by how much of the model had any evidence for it.
    pub fn angular_spread(&self, surfel: usize) -> f32 {
        let samples = self.of(surfel);
        if samples.len() < 2 {
            return 0.0;
        }
        let mut widest = 0.0f32;
        for (index, first) in samples.iter().enumerate() {
            for second in &samples[index + 1..] {
                widest = widest.max(1.0 - first.towards.dot(second.towards));
            }
        }
        widest
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
    let (width, height) = (capture.width, capture.height);
    let mut depth = vec![f32::INFINITY; width * height];
    let mut per_surfel: Vec<Vec<Sample>> = vec![Vec::new(); count];

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
            per_surfel[index].push(Sample {
                radiance: view.pixels[y * width + x],
                towards,
                facing,
            });
        }
    }

    let mut offsets = Vec::with_capacity(count + 1);
    let mut samples = Vec::new();
    offsets.push(0);
    for list in &per_surfel {
        samples.extend_from_slice(list);
        offsets.push(samples.len() as u32);
    }
    Observations { samples, offsets }
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
    let count = observations.surfels();
    if clusters == 0 || clusters >= count {
        return ((0..count as u32).collect(), count);
    }
    let chromaticity: Vec<[f32; 3]> = (0..count)
        .map(|index| {
            let rgb = observations.mean(index);
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
pub struct Kernel {
    pub directions: Vec<glam::Vec3>,
    pub solid_angles: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl Kernel {
    pub fn new(width: usize, height: usize) -> Self {
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

    pub fn texels(&self) -> usize {
        self.directions.len()
    }

    /// Which texel a direction falls in.
    ///
    /// The inverse of `equirect_direction`, so a mirror direction can be asked
    /// the same occlusion question the diffuse half asks of every texel.
    pub fn texel_of(&self, direction: glam::Vec3) -> usize {
        let yaw = direction.y.clamp(-1.0, 1.0).asin();
        let pitch = direction.x.atan2(direction.z);
        let u = (pitch / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
        let v = (0.5 - yaw / std::f32::consts::PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f32) as usize).min(self.width - 1);
        let y = ((v * self.height as f32) as usize).min(self.height - 1);
        y * self.width + x
    }

    /// Weights of every texel for one normal, written into `out`.
    pub fn weights(&self, normal: glam::Vec3, out: &mut [f32]) {
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

/// The directions the environment texels of a fit look in.
///
/// Exposed so a caller can compute visibility against the same layout the fit
/// will use. Getting the two out of step would shadow each texel with a
/// different one's occlusion, which is worse than no shadowing at all.
pub fn environment_directions(width: usize) -> Vec<glam::Vec3> {
    Kernel::new(width, width / 2).directions
}

/// The sky a fit of this width works against.
pub fn environment_kernel(width: usize) -> Kernel {
    Kernel::new(width, width / 2)
}

/// Direct shading for every surfel, computed rather than cached.
///
/// The cached table in the fit covers only the surfels that were seen. A
/// bounce needs every surfel, because a surface nobody photographed still
/// blocks light and still reflects it.
fn shade_all(
    model: &vol::relight::RelightModel,
    kernel: &Kernel,
    light: &[[f32; 3]],
    visibility: Option<&visibility::Visibility>,
) -> Vec<[f32; 3]> {
    let texels = kernel.texels();
    let count = model.surfels.len();
    let mut out = vec![[0.0f32; 3]; count];
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = count.div_ceil(threads).max(1);
    thread::scope(|scope| {
        for (block, rows) in out.chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                let mut weights = vec![0.0f32; texels];
                for (local, row) in rows.iter_mut().enumerate() {
                    let index = block * chunk + local;
                    kernel.weights(glam::Vec3::from(model.surfels[index].normal), &mut weights);
                    let mut total = [0.0f32; 3];
                    for (texel, &weight) in weights.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        if visibility.is_some_and(|v| !v.visible(index, texel)) {
                            continue;
                        }
                        for (value, source) in total.iter_mut().zip(light[texel]) {
                            *value += weight * source;
                        }
                    }
                    *row = total;
                }
            });
        }
    });
    out
}

/// How far the model is from the photographs, with a material already in hand.
///
/// The point of having this apart from the solver is that it can be pointed at
/// the truth. Given the real materials and the real light, whatever is left is
/// what the model cannot express — no solver gets under it, and one that
/// appears to is fitting the gap rather than the scene.
///
/// Measured per sample rather than per surfel, because the specular term is the
/// reason the samples were kept apart.
pub fn forward_error(
    model: &vol::relight::RelightModel,
    environment: &vol::relight::Environment,
    visibility: Option<&visibility::Visibility>,
    observations: &Observations,
) -> f64 {
    let kernel = Kernel::new(environment.width, environment.height);
    let specular = vol::relight::SpecularEnvironment::prefilter(
        environment,
        environment.width,
        environment.height,
    );
    let diffuse = shade_all(model, &kernel, &environment.texels, visibility);
    let mut error = 0.0f64;
    let mut scale = 0.0f64;
    let mut count = 0usize;
    for (index, surfel) in model.surfels.iter().enumerate() {
        let material = &model.materials[surfel.material as usize];
        let normal = glam::Vec3::from(surfel.normal);
        for sample in observations.of(index) {
            let lobe = specular_radiance(
                &specular,
                normal,
                sample,
                material.roughness,
                visibility.map(|v| (v, &kernel, index)),
            );
            let gain = vol::relight::specular_scale(
                material.specular_f0,
                material.roughness,
                sample.facing,
            );
            for channel in 0..3 {
                let predicted = material.albedo[channel] * diffuse[index][channel]
                    + lobe[channel] * gain[channel];
                let difference = (predicted - sample.radiance[channel]) as f64;
                error += difference * difference;
                scale += sample.radiance[channel] as f64;
            }
            count += 3;
        }
    }
    let mean = (scale / count.max(1) as f64).max(1.0e-9);
    (error / count.max(1) as f64).sqrt() / mean
}

/// The prefiltered environment in the mirror direction of one sample.
///
/// Occluded the same way the diffuse half is. Leaving the lobe unshadowed while
/// the diffuse term is shadowed is not a small inconsistency: a metal sphere
/// standing on a floor reflects that floor over much of its surface, and a fit
/// that thinks it is reflecting open sky there predicts far too much light,
/// decides the metal hypothesis fits badly, and puts the metal's colour into a
/// diffuse albedo instead.
fn specular_radiance(
    specular: &vol::relight::SpecularEnvironment,
    normal: glam::Vec3,
    sample: &Sample,
    roughness: f32,
    occlusion: Option<(&visibility::Visibility, &Kernel, usize)>,
) -> [f32; 3] {
    let n_dot_v = normal.dot(sample.towards);
    if n_dot_v <= 0.0 {
        return [0.0; 3];
    }
    let reflection = (2.0 * n_dot_v * normal - sample.towards).normalize();
    if let Some((visibility, kernel, surfel)) = occlusion {
        // The centre of the lobe, which is the whole of it for a sharp one and
        // a fair sample of it for a broad one.
        if !visibility.visible(surfel, kernel.texel_of(reflection)) {
            return [0.0; 3];
        }
    }
    vol::relight::sample_prefiltered(specular, reflection, roughness)
}

/// A recovered scene, and what it cost to recover.
pub struct Decomposition {
    pub scene: score::Scene,
    /// Material index per surfel.
    pub assignment: Vec<u32>,
    /// Root mean square residual of the fit, in linear radiance.
    pub residual: f32,
    /// Surfels no view ever saw. They keep a default material and are the
    /// honest measure of how much of the model is unsupported by evidence.
    pub unseen: usize,
    /// Fraction of the materials that were seen from angles far enough apart
    /// for a lobe to be identifiable at all.
    ///
    /// Roughness and F0 are recoverable only from how appearance changes with
    /// direction. A material seen from one angle has no evidence about either,
    /// whatever number the solver writes down, and this says how much of the
    /// result is in that position.
    pub with_lobe_evidence: f64,
}

/// What the fit is told rather than asked to work out.
#[derive(Clone, Copy, Default)]
pub struct Given<'a> {
    /// Which parts of the sky each surfel can see. Without it, a patch in
    /// shadow has only its material to explain being dark.
    pub visibility: Option<&'a visibility::Visibility>,
    /// A light that is known and held fixed.
    ///
    /// The reason to have this is diagnostic rather than practical: it
    /// separates "the material solver does not work" from "the light the
    /// material solver was given is too coarse to fit a lobe against", and
    /// those two want completely different fixes.
    pub light: Option<&'a vol::relight::Environment>,
}

/// Split what was observed into a material and a light.
///
/// Three things alternate, each convex in its own variable given the others:
///
/// 1. **the material** — albedo and F0 are jointly linear given the light and a
///    roughness, so they come out of a two-by-two per channel, and roughness is
///    chosen from the eight prefiltered levels by residual. Metalness is not a
///    parameter: it is what a fitted pair of low albedo and high coloured F0
///    means.
/// 2. **the light** — a multiplicative update, which keeps every texel
///    non-negative without a constraint solver and cannot step past zero.
/// 3. **the bounce** — the light arriving from whatever is shadowing each
///    surfel, refreshed a few times because it depends on the current answer.
///
/// One material per surfel is the default. Sharing them across surfels is
/// available and measured, and it buys a better-shaped light at the cost of the
/// albedo and of the re-rendering; per surfel is the better trade and the
/// honest one, since a shared material is a prior about the scene rather than
/// something the photographs said.
pub fn fit(
    model: &vol::relight::RelightModel,
    observations: &Observations,
    options: FitOptions,
    given: Given<'_>,
) -> Decomposition {
    let visibility = given.visibility;
    // A light handed over brings its own resolution with it. Resampling it to
    // the fit's would be answering a different question: the point of fixing
    // the light is usually to ask what the material fit can do when the sky is
    // not the limitation, and a coarser copy would put the limitation back.
    let kernel = match given.light {
        Some(light) => Kernel::new(light.width, light.height),
        None => Kernel::new(options.environment_width, options.environment_width / 2),
    };
    let texels = kernel.texels();
    let count = model.surfels.len();
    let (assignment, clusters) = cluster_by_chromaticity(observations, options.materials, 0x5EED);

    let active: Vec<usize> = (0..count)
        .filter(|&index| !observations.of(index).is_empty())
        .collect();

    // The cosine weights depend only on the normal and the occlusion, and are
    // reused by every stage of every iteration.
    let mut response = vec![0.0f32; active.len() * texels];
    for (slot, &index) in active.iter().enumerate() {
        let row = &mut response[slot * texels..(slot + 1) * texels];
        kernel.weights(glam::Vec3::from(model.surfels[index].normal), row);
        // A texel the surfel cannot see contributes nothing to it. This is the
        // whole of shadowing, and without it the only way for a patch in shadow
        // to be dark is for its material to be dark.
        if let Some(visibility) = visibility {
            for (texel, weight) in row.iter_mut().enumerate() {
                if !visibility.visible(index, texel) {
                    *weight = 0.0;
                }
            }
        }
    }
    // Which slot of the active list each surfel occupies, for the stages that
    // walk clusters rather than the active list.
    let mut slot_of = vec![u32::MAX; count];
    for (slot, &index) in active.iter().enumerate() {
        slot_of[index] = slot as u32;
    }

    // Start from a uniform sky bright enough that a mid albedo explains the
    // mean observation. Starting uniform is also the prior: the multiplicative
    // update moves every texel away from the same place, so it only puts
    // structure where the observations asked for it.
    let mut light = match given.light {
        Some(known) => known.texels.clone(),
        None => {
            let mean_observed = mean_radiance(observations, &active);
            let mut uniform = vec![[0.0f32; 3]; texels];
            for texel in uniform.iter_mut() {
                for channel in 0..3 {
                    texel[channel] = (mean_observed[channel] / 0.5).max(1.0e-4);
                }
            }
            uniform
        }
    };

    let mut albedo = vec![[0.5f32; 3]; clusters];
    let mut specular_f0 = vec![[0.04f32; 3]; clusters];
    let mut roughness = vec![1.0f32; clusters];
    let mut shade = vec![[0.0f32; 3]; active.len()];
    // Irradiance arriving from other surfels rather than from the sky. Zero
    // until there is something to bounce, and refreshed a few times: it depends
    // on the current answer, so it is a fixed point rather than a quantity that
    // can be computed once.
    let mut indirect = vec![[0.0f32; 3]; count];
    let mut total = vec![[0.0f32; 3]; active.len()];
    // What the lobe contributes to each sample, held fixed while the diffuse
    // half and the light are solved and refreshed alongside the bounce.
    let mut lobe = vec![[0.0f32; 3]; observations.samples.len()];
    let mut corrected = vec![[0.0f32; 3]; active.len()];

    let bounces = if visibility.is_some() {
        options.bounces
    } else {
        0
    };
    let refresh = (options.iterations.max(1) / (options.specular_rounds.max(bounces) + 1)).max(1);
    let mut residual = 0.0f32;
    for iteration in 0..options.iterations.max(1) {
        evaluate_shade(&response, &light, texels, &mut shade);
        for (slot, &index) in active.iter().enumerate() {
            for channel in 0..3 {
                total[slot][channel] = shade[slot][channel] + indirect[index][channel];
            }
            // What the diffuse half still has to explain, once the lobe has
            // been credited. The weighted mean is still the right sufficient
            // statistic for a view-independent term — it just has to be the
            // mean of the residual rather than of the observation.
            let mut sum = [0.0f32; 3];
            let mut weight = 0.0f32;
            let begin = observations.offsets[index] as usize;
            for (offset, sample) in observations.of(index).iter().enumerate() {
                for channel in 0..3 {
                    sum[channel] += (sample.radiance[channel] - lobe[begin + offset][channel])
                        .max(0.0)
                        * sample.facing;
                }
                weight += sample.facing;
            }
            for channel in 0..3 {
                corrected[slot][channel] = if weight > 0.0 {
                    sum[channel] / weight
                } else {
                    0.0
                };
            }
        }

        solve_albedo(
            &total,
            &corrected,
            observations,
            &active,
            &assignment,
            clusters,
            options.brightest_albedo,
            &mut albedo,
        );
        residual = update_light(
            &response,
            &total,
            &corrected,
            observations,
            &active,
            &assignment,
            &albedo,
            &indirect,
            texels,
            // A light that was handed over is not a variable. The residual is
            // still wanted, so the step is computed and discarded rather than
            // skipped.
            if given.light.is_some() {
                None
            } else {
                Some(&mut light)
            },
        );

        if iteration > 0 && iteration % refresh == 0 {
            if bounces > 0 {
                let direct = shade_all(model, &kernel, &light, visibility);
                let outgoing: Vec<[f32; 3]> = (0..count)
                    .map(|index| {
                        let rho = albedo[assignment[index] as usize];
                        let mut radiance = [0.0f32; 3];
                        for channel in 0..3 {
                            radiance[channel] =
                                rho[channel] * (direct[index][channel] + indirect[index][channel]);
                        }
                        radiance
                    })
                    .collect();
                indirect = visibility::bounce(
                    model,
                    &kernel,
                    &outgoing,
                    visibility::VisibilityOptions::default(),
                );
            }
            if options.specular_rounds > 0 {
                let environment = vol::relight::Environment {
                    width: kernel.width,
                    height: kernel.height,
                    texels: light.clone(),
                };
                let prefiltered = vol::relight::SpecularEnvironment::prefilter(
                    &environment,
                    kernel.width,
                    kernel.height,
                );
                solve_lobe(
                    model,
                    observations,
                    &prefiltered,
                    &total,
                    &slot_of,
                    &assignment,
                    clusters,
                    options.brightest_albedo,
                    options.lobe_margin,
                    visibility,
                    &kernel,
                    &mut albedo,
                    &mut specular_f0,
                    &mut roughness,
                    &mut lobe,
                );
            }
        }
    }

    // A light that was handed over is already in its own units, and moving it
    // would be answering with a scene that does not match the light it was
    // told about.
    if given.light.is_none() {
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
    }

    let materials = (0..clusters)
        .map(|cluster| vol::relight::Material {
            albedo: albedo[cluster],
            roughness: roughness[cluster],
            specular_f0: specular_f0[cluster],
            _padding: 0.0,
        })
        .collect();
    // A surfel nothing ever saw has no material of its own, and it still gets
    // drawn: the renderer averages every disc within a depth band, so an
    // unlit-grey neighbour dilutes a surface that was measured. Copying the
    // nearest measured surfel's material is local evidence rather than a
    // scene-wide prior, and it is what makes one material per surfel usable on
    // a capture where most of the geometry is buried.
    let mut assignment = assignment;
    if !active.is_empty() && active.len() < count {
        let seen: Vec<[f32; 3]> = active
            .iter()
            .map(|&index| model.surfels[index].center)
            .collect();
        let tree = kiddo::ImmutableKdTree::new_from_slice(&seen);
        for index in 0..count {
            if !observations.of(index).is_empty() {
                continue;
            }
            let hit = tree.nearest_one::<kiddo::SquaredEuclidean>(&model.surfels[index].center);
            assignment[index] = assignment[active[hit.item as usize]];
        }
    }

    let mut surfels = model.surfels.clone();
    for (surfel, &material) in surfels.iter_mut().zip(&assignment) {
        surfel.material = material;
    }

    // How much of the result had any evidence about a lobe. Two views of one
    // surfel a degree apart say nothing about roughness, however confident the
    // number that comes out is.
    const ENOUGH_SPREAD: f32 = 0.05;
    let with_evidence = active
        .iter()
        .filter(|&&index| observations.angular_spread(index) > ENOUGH_SPREAD)
        .count();

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
        with_lobe_evidence: with_evidence as f64 / active.len().max(1) as f64,
    }
}

/// Move the colour of the illuminant into the materials.
///
/// A red wall under white light and a white wall under red light produce the
/// same photograph. Nothing in the images distinguishes them, so one of the two
/// has to be assumed, and this assumes the light: its mean over the sphere is
/// made neutral and the per-channel gain that took it there is handed to the
/// albedo.
///
/// The assumption is wrong in exactly the case you would expect — a room lit by
/// tungsten returns materials that are too warm — and it is stated here rather
/// than buried because it is not recoverable, only chosen.
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
        for (sum, value) in total.iter_mut().zip(observations.mean(index)) {
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

#[allow(clippy::too_many_arguments)]
fn solve_albedo(
    shade: &[[f32; 3]],
    corrected: &[[f32; 3]],
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
        let weight = observations.weight(index) as f64;
        for channel in 0..3 {
            let e = shade[slot][channel] as f64;
            numerator[cluster][channel] += weight * e * corrected[slot][channel] as f64;
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

/// Choose a roughness and an F0 for every material, and re-solve its albedo
/// alongside them.
///
/// The two are separable only because their angular dependence differs: the
/// diffuse term is the same from every direction, while the lobe follows the
/// mirror direction and the Fresnel factor follows `n . v`. Where a material
/// was seen from one angle they are not separable at all, which is why F0 is
/// pulled towards the dielectric value and why the fraction of materials with
/// any angular spread is reported.
///
/// Roughness is picked from the eight prefiltered levels rather than optimised.
/// It enters the model through a texture lookup, so a continuous parameter
/// would be optimising through an interpolation of eight numbers; and eight
/// levels is exactly the resolution the renderer will use.
#[allow(clippy::too_many_arguments)]
fn solve_lobe(
    model: &vol::relight::RelightModel,
    observations: &Observations,
    specular: &vol::relight::SpecularEnvironment,
    total: &[[f32; 3]],
    slot_of: &[u32],
    assignment: &[u32],
    clusters: usize,
    ceiling: f32,
    margin: f64,
    occlusion: Option<&visibility::Visibility>,
    kernel: &Kernel,
    albedo: &mut [[f32; 3]],
    specular_f0: &mut [[f32; 3]],
    roughness: &mut [f32],
    lobe: &mut [[f32; 3]],
) {
    // Which surfels make up each material.
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); clusters];
    for (index, &cluster) in assignment.iter().enumerate() {
        if slot_of[index] != u32::MAX {
            members[cluster as usize].push(index as u32);
        }
    }

    /// Albedo, reflectance at normal incidence, and roughness.
    type Solved = ([f32; 3], [f32; 3], f32);

    let levels = vol::relight::SPECULAR_LEVELS as usize;
    let solved: Vec<Solved> = {
        let threads = thread::available_parallelism().map_or(1, |n| n.get());
        let chunk = clusters.div_ceil(threads).max(1);
        let mut parts: Vec<Vec<Solved>> = Vec::new();
        let order: Vec<usize> = (0..clusters).collect();
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for block in order.chunks(chunk) {
                let members = &members;
                handles.push(scope.spawn(move || {
                    block
                        .iter()
                        .map(|&cluster| {
                            // The default: a rough dielectric. Anything else has
                            // to beat it by a margin, because at four per cent
                            // of the brightness the residuals of every
                            // hypothesis are within noise of each other and
                            // the winner is whichever way the noise fell.
                            let default = solve_one_lobe(
                                model,
                                observations,
                                specular,
                                total,
                                slot_of,
                                &members[cluster],
                                1.0,
                                ceiling,
                                Hypothesis::Dielectric,
                                occlusion,
                                kernel,
                            );
                            let threshold = default.2 * (1.0 - margin);
                            let mut best = (default.0, default.1, 1.0f32);
                            let mut best_error = threshold;
                            for level in 0..levels {
                                let candidate = level as f32 / (levels - 1) as f32;
                                for hypothesis in [Hypothesis::Dielectric, Hypothesis::Metal] {
                                    let (rho, f0, error) = solve_one_lobe(
                                        model,
                                        observations,
                                        specular,
                                        total,
                                        slot_of,
                                        &members[cluster],
                                        candidate,
                                        ceiling,
                                        hypothesis,
                                        occlusion,
                                        kernel,
                                    );
                                    if error < best_error {
                                        best_error = error;
                                        best = (rho, f0, candidate);
                                    }
                                }
                            }
                            best
                        })
                        .collect::<Vec<Solved>>()
                }));
            }
            for handle in handles {
                parts.push(handle.join().unwrap());
            }
        });
        parts.concat()
    };

    for (cluster, entry) in solved.iter().enumerate() {
        let (rho, f0, chosen) = *entry;
        if members[cluster].is_empty() {
            continue;
        }
        albedo[cluster] = rho;
        specular_f0[cluster] = f0;
        roughness[cluster] = chosen;
    }

    // Record what the chosen lobe contributes to every sample, so the diffuse
    // half can be solved against what is left.
    for index in 0..model.surfels.len() {
        if slot_of[index] == u32::MAX {
            continue;
        }
        let cluster = assignment[index] as usize;
        let normal = glam::Vec3::from(model.surfels[index].normal);
        let begin = observations.offsets[index] as usize;
        for (offset, sample) in observations.of(index).iter().enumerate() {
            let radiance = specular_radiance(
                specular,
                normal,
                sample,
                roughness[cluster],
                occlusion.map(|v| (v, kernel, index)),
            );
            let gain = vol::relight::specular_scale(
                specular_f0[cluster],
                roughness[cluster],
                sample.facing,
            );
            for channel in 0..3 {
                lobe[begin + offset][channel] = radiance[channel] * gain[channel];
            }
        }
    }
}

/// The best material for one cluster at one roughness, and what it costs.
///
/// Two hypotheses rather than one free solve, because the free solve does not
/// work and the reason is physical rather than numerical. A dielectric and a
/// metal are not two points on a continuum the data can interpolate along:
///
///   - a **dielectric** reflects about 4 % at normal incidence whatever colour
///     it is, and carries its colour in the diffuse term;
///   - a **metal** has no diffuse term at all, and carries its colour in the
///     reflectance.
///
/// Left free, the two are close to collinear — a bright lobe and a bright
/// albedo both make a surfel brighter — so the solver splits the difference and
/// returns a half-metal that is neither. On the truth scene that showed up as a
/// gold sphere whose colour was recovered correctly and put entirely in the
/// wrong channel: albedo 0.60 0.51 0.29 against a true albedo of zero.
///
/// Solving each hypothesis separately makes both well conditioned — one unknown
/// per channel instead of two — and makes metalness a decision with a residual
/// behind it rather than a number that came out of a matrix.
fn solve_one_lobe(
    model: &vol::relight::RelightModel,
    observations: &Observations,
    specular: &vol::relight::SpecularEnvironment,
    total: &[[f32; 3]],
    slot_of: &[u32],
    members: &[u32],
    roughness: f32,
    ceiling: f32,
    hypothesis: Hypothesis,
    occlusion: Option<&visibility::Visibility>,
    kernel: &Kernel,
) -> ([f32; 3], [f32; 3], f64) {
    /// Reflectance of a dielectric at normal incidence. Not fitted: nothing in
    /// a photograph of a painted wall distinguishes 0.04 from 0.05, and letting
    /// it float is what lets a wall become a mirror.
    const DIELECTRIC: f32 = 0.04;

    // Per channel: `o = x * basis + offset`, one unknown.
    let mut fit = Accumulator::default();

    for &member in members {
        let index = member as usize;
        let slot = slot_of[index] as usize;
        let normal = glam::Vec3::from(model.surfels[index].normal);
        for sample in observations.of(index) {
            let radiance = specular_radiance(
                specular,
                normal,
                sample,
                roughness,
                occlusion.map(|v| (v, kernel, index)),
            );
            // `specular_scale` is affine in F0, so one evaluation at zero and
            // one at one recover its slope and offset exactly.
            let zero = vol::relight::specular_scale([0.0; 3], roughness, sample.facing);
            let one = vol::relight::specular_scale([1.0; 3], roughness, sample.facing);
            let weight = sample.facing as f64;
            for channel in 0..3 {
                let diffuse = total[slot][channel] as f64;
                let slope = (radiance[channel] * (one[channel] - zero[channel])) as f64;
                let floor = (radiance[channel] * zero[channel]) as f64;
                let observed = sample.radiance[channel] as f64;

                match hypothesis {
                    // The lobe is fixed, the albedo is the unknown.
                    Hypothesis::Dielectric => {
                        let fixed = floor + slope * DIELECTRIC as f64;
                        fit.add(channel, weight, diffuse, observed - fixed);
                    }
                    // No diffuse term at all; the reflectance is the unknown.
                    Hypothesis::Metal => {
                        fit.add(channel, weight, slope, observed - floor);
                    }
                }
            }
        }
    }

    let mut albedo = match hypothesis {
        Hypothesis::Dielectric => [0.5f32; 3],
        Hypothesis::Metal => [0.005f32; 3],
    };
    let mut specular_f0 = [DIELECTRIC; 3];
    let mut error = 0.0f64;
    for channel in 0..3 {
        match hypothesis {
            Hypothesis::Dielectric => {
                let (value, left) = fit.solve(channel, 0.5);
                albedo[channel] = (value as f32).clamp(0.005, ceiling);
                error += left;
            }
            Hypothesis::Metal => {
                let (value, left) = fit.solve(channel, DIELECTRIC as f64);
                specular_f0[channel] = (value as f32).clamp(0.02, 1.0);
                error += left;
            }
        }
    }
    (albedo, specular_f0, error)
}

/// What a surface is being supposed to be, while its parameters are solved.
#[derive(Clone, Copy)]
enum Hypothesis {
    /// Colour in the diffuse term, reflectance fixed at the dielectric value.
    Dielectric,
    /// Colour in the reflectance, no diffuse term at all.
    Metal,
}

/// Normal equations for one unknown per channel, and the residual it leaves.
#[derive(Default)]
struct Accumulator {
    basis_squared: [f64; 3],
    basis_target: [f64; 3],
    target_squared: [f64; 3],
}

impl Accumulator {
    fn add(&mut self, channel: usize, weight: f64, basis: f64, target: f64) {
        self.basis_squared[channel] += weight * basis * basis;
        self.basis_target[channel] += weight * basis * target;
        self.target_squared[channel] += weight * target * target;
    }

    /// The least-squares value and the squared error it leaves behind.
    fn solve(&self, channel: usize, fallback: f64) -> (f64, f64) {
        if self.basis_squared[channel] <= 1.0e-14 {
            return (fallback, self.target_squared[channel]);
        }
        let value = self.basis_target[channel] / self.basis_squared[channel];
        let error = self.target_squared[channel] - 2.0 * value * self.basis_target[channel]
            + value * value * self.basis_squared[channel];
        (value, error.max(0.0))
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
    corrected: &[[f32; 3]],
    observations: &Observations,
    active: &[usize],
    assignment: &[u32],
    albedo: &[[f32; 3]],
    indirect: &[[f32; 3]],
    texels: usize,
    light: Option<&mut [[f32; 3]]>,
) -> f32 {
    /// What one thread contributes: the light each texel was seen to give, the
    /// light it is currently predicted to give, and the squared error.
    type Partial = (Vec<[f64; 3]>, Vec<[f64; 3]>, f64);

    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = (active.len() / threads).max(1);
    let mut partials: Vec<Partial> = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (block, slots) in active.chunks(chunk).enumerate() {
            let response = &response[block * chunk * texels..];
            let shade = &shade[block * chunk..];
            let corrected = &corrected[block * chunk..];
            handles.push(scope.spawn(move || {
                let mut seen = vec![[0.0f64; 3]; texels];
                let mut predicted = vec![[0.0f64; 3]; texels];
                let mut error = 0.0f64;
                for (slot, &index) in slots.iter().enumerate() {
                    let weights = &response[slot * texels..(slot + 1) * texels];
                    let rho = albedo[assignment[index] as usize];
                    let confidence = observations.weight(index) as f64;
                    for channel in 0..3 {
                        let difference =
                            (rho[channel] * shade[slot][channel] - corrected[slot][channel]) as f64;
                        error += confidence * difference * difference;
                    }
                    for (texel, &weight) in weights.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        let weight = weight as f64 * confidence;
                        for channel in 0..3 {
                            let rho = rho[channel] as f64;
                            // What the sky still has to account for, once the
                            // bounce has been credited. Clamped at zero: a
                            // surfel already over-explained by indirect light
                            // asks the sky for nothing, not for less than
                            // nothing, and a negative would turn the
                            // multiplicative step into a sign flip.
                            let residual = (corrected[slot][channel] as f64
                                - rho * indirect[index][channel] as f64)
                                .max(0.0);
                            seen[texel][channel] += weight * rho * residual;
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
        confidence += observations.weight(index) as f64;
    }

    if let Some(light) = light {
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
    }
    ((error / (3.0 * confidence.max(1.0e-6))).max(0.0) as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One head-on observation per surfel, which is all a diffuse-only test
    /// needs and is deliberately not enough to identify a lobe.
    fn from_means(mean: &[[f32; 3]]) -> Observations {
        let samples = mean
            .iter()
            .map(|radiance| Sample {
                radiance: *radiance,
                towards: glam::Vec3::Y,
                facing: 1.0,
            })
            .collect();
        Observations {
            samples,
            offsets: (0..=mean.len() as u32).collect(),
        }
    }

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
        from_means(&mean)
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
                // Diffuse only: these tests are about the albedo/light split,
                // and a lobe cannot be identified from one view anyway.
                specular_rounds: 0,
                lobe_margin: 0.15,
                bounces: 0,
                brightest_albedo: 0.6,
            },
            Given::default(),
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

    /// Photograph a model with the renderer's own CPU shading, from many
    /// directions, with nothing else in the way.
    ///
    /// No geometry stage, no GPU, no visibility, no bounce, no clustering —
    /// just the forward model the solver assumes, so a failure here is the
    /// solver's and a failure only in the full pipeline is somewhere else.
    fn photograph_analytically(
        model: &vol::relight::RelightModel,
        environment: &vol::relight::Environment,
        views: usize,
    ) -> Observations {
        let specular = vol::relight::SpecularEnvironment::prefilter(
            environment,
            environment.width,
            environment.height,
        );
        let irradiance = environment.diffuse_irradiance();
        let mut samples = Vec::new();
        let mut offsets = vec![0u32];
        for surfel in &model.surfels {
            let normal = glam::Vec3::from(surfel.normal);
            let material = &model.materials[surfel.material as usize];
            for view in 0..views {
                // Camera directions spread over the sphere, so each surfel is
                // seen from a spread of angles rather than one.
                let angle = std::f32::consts::TAU * view as f32 / views as f32;
                let towards = glam::Vec3::new(angle.cos(), 0.35 * (2.0 * angle).sin(), angle.sin())
                    .normalize();
                let facing = normal.dot(towards);
                if facing < 0.2 {
                    continue;
                }
                samples.push(Sample {
                    radiance: vol::relight::shade(
                        normal,
                        towards,
                        material,
                        &irradiance,
                        &specular,
                    ),
                    towards,
                    facing,
                });
            }
            offsets.push(samples.len() as u32);
        }
        Observations { samples, offsets }
    }

    #[test]
    fn a_metal_is_recovered_as_a_metal_when_the_light_is_known() {
        // The narrowest possible statement of the lobe fit: one sphere, one
        // material, the real light handed over, and the renderer's own shading
        // used to make the photographs. If a gold sphere comes back with a
        // diffuse albedo, the solver has put the colour in the wrong channel.
        let gold = vol::relight::Material {
            albedo: [0.0; 3],
            roughness: 0.25,
            specular_f0: [1.0, 0.72, 0.29],
            _padding: 0.0,
        };
        let mut model = sphere(2048, [0.0; 3]);
        model.materials = vec![gold];
        let environment = vol::relight::Environment::sky(
            glam::Vec3::new(0.55, 0.62, -0.56),
            [22.0, 20.0, 17.0],
            0.09,
            64,
            32,
        );
        let observations = photograph_analytically(&model, &environment, 32);

        let fitted = fit(
            &model,
            &observations,
            FitOptions {
                materials: 1,
                iterations: 8,
                specular_rounds: 4,
                bounces: 0,
                min_facing: 0.0,
                ..Default::default()
            },
            Given {
                visibility: None,
                light: Some(&environment),
            },
        );
        let found = fitted.scene.model.materials[0];
        assert!(
            found.albedo.iter().sum::<f32>() < 0.15,
            "a metal came back with a diffuse albedo of {:?}",
            found.albedo
        );
        for channel in 0..3 {
            assert!(
                (found.specular_f0[channel] - gold.specular_f0[channel]).abs() < 0.2,
                "reflectance {:?} against {:?}",
                found.specular_f0,
                gold.specular_f0
            );
        }
        assert!(
            (found.roughness - gold.roughness).abs() < 0.2,
            "roughness {} against {}",
            found.roughness,
            gold.roughness
        );
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
        let observations = from_means(&mean);

        let fitted = fit(
            &model,
            &observations,
            FitOptions {
                materials: 2,
                environment_width: 16,
                iterations: 80,
                min_facing: 0.0,
                // Diffuse only: these tests are about the albedo/light split,
                // and a lobe cannot be identified from one view anyway.
                specular_rounds: 0,
                lobe_margin: 0.15,
                bounces: 0,
                brightest_albedo: 0.8,
            },
            Given::default(),
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
                // Diffuse only: these tests are about the albedo/light split,
                // and a lobe cannot be identified from one view anyway.
                specular_rounds: 0,
                lobe_margin: 0.15,
                bounces: 0,
                brightest_albedo: 0.8,
            },
            Given::default(),
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
