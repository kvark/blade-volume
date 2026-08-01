//! A relightable surface representation, and the lighting it is shaded with.
//!
//! Everything else in this crate stores appearance as spherical harmonics: the
//! radiance a point sent towards the camera under whatever light happened to be
//! there when it was captured. That cannot be relit, because the light is
//! already inside the number.
//!
//! This stores the material instead — an albedo, a specular reflectance and a
//! roughness — and computes the radiance at render time from whatever
//! environment it is given. The shading is the direct-lighting model that the
//! relighting study settled on:
//!
//! ```text
//! L = albedo * E(n) + prefiltered(reflect(v, n), roughness) * (F0 * A + B)
//! ```
//!
//! with `E` nine coefficients of diffuse irradiance and the specular term the
//! usual split sum. Three things about it are measured rather than chosen:
//!
//! - **Normals are exact.** A surfel is a disc, so its normal is a property of
//!   the primitive rather than something inferred from a covariance. Five
//!   degrees of normal error costs 1.3 dB and ten costs 3.6, which is more than
//!   any other approximation here.
//! - **No spherical harmonics.** With a material and a light, view dependence
//!   is derived rather than stored, and a primitive costs eleven floats instead
//!   of sixty-two.
//! - **No shadowing, and no indirect light.** Not for want of either: leaving
//!   out visibility makes the result too bright and leaving out interreflection
//!   makes it too dark, by about the same amount, so a model with neither sits
//!   closer to a path traced reference than one with only the first. They have
//!   to arrive together.
//!
//! Materials are shared rather than stored per primitive, which is what a patch
//! of surface can actually support: one patch does not determine a BRDF.

use std::mem;

/// The number of prefiltered roughness levels in a [`SpecularEnvironment`].
///
/// The ladder has to reach a mirror at one end, and the resolution of each
/// level has to resolve the source, or a small bright light smears into a dim
/// smudge and the highlight it should have produced is lost.
pub const SPECULAR_LEVELS: u32 = 8;

/// An oriented disc of surface.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Surfel {
    pub center: [f32; 3],
    pub radius: f32,
    /// Unit normal. The disc lies in the plane perpendicular to it.
    pub normal: [f32; 3],
    /// Index into the material table.
    pub material: u32,
}

/// What a surface is made of, shared by every surfel that points at it.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    /// Diffuse albedo. Zero for a metal, which has no diffuse response.
    pub albedo: [f32; 3],
    pub roughness: f32,
    /// Specular reflectance at normal incidence. About 0.04 for a dielectric,
    /// and the base colour for a metal.
    pub specular_f0: [f32; 3],
    pub _padding: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [0.5; 3],
            roughness: 0.5,
            specular_f0: [0.04; 3],
            _padding: 0.0,
        }
    }
}

/// A set of surfels and the materials they refer to.
#[derive(Clone, Debug, Default)]
pub struct RelightModel {
    pub surfels: Vec<Surfel>,
    pub materials: Vec<Material>,
}

impl RelightModel {
    pub fn is_empty(&self) -> bool {
        self.surfels.is_empty()
    }

    /// Check the things that would otherwise fail far from their cause.
    pub fn validate(&self) -> Result<(), String> {
        if self.materials.is_empty() {
            return Err("a relightable model needs at least one material".to_string());
        }
        for (index, surfel) in self.surfels.iter().enumerate() {
            if surfel.material as usize >= self.materials.len() {
                return Err(format!(
                    "surfel {index} refers to material {} of {}",
                    surfel.material,
                    self.materials.len()
                ));
            }
            if !surfel.radius.is_finite() || surfel.radius <= 0.0 {
                return Err(format!("surfel {index} has radius {}", surfel.radius));
            }
            let normal = glam::Vec3::from(surfel.normal);
            if !normal.is_finite() || (normal.length() - 1.0).abs() > 1.0e-3 {
                return Err(format!(
                    "surfel {index} has a normal of length {}",
                    normal.length()
                ));
            }
            if !surfel.center.iter().all(|c| c.is_finite()) {
                return Err(format!("surfel {index} has a non-finite centre"));
            }
        }
        for (index, material) in self.materials.iter().enumerate() {
            if !material.roughness.is_finite()
                || material.roughness < 0.0
                || material.roughness > 1.0
            {
                return Err(format!(
                    "material {index} has roughness {}",
                    material.roughness
                ));
            }
        }
        Ok(())
    }

    /// World-space bounds, for framing a camera on the thing.
    pub fn bounds(&self) -> Option<(glam::Vec3, glam::Vec3)> {
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        for surfel in &self.surfels {
            let center = glam::Vec3::from(surfel.center);
            let extent = glam::Vec3::splat(surfel.radius);
            min = min.min(center - extent);
            max = max.max(center + extent);
        }
        (min.x <= max.x).then_some((min, max))
    }
}

// ------------------------------------------------------------------- lighting

/// Real spherical harmonic basis for bands 0..=2.
pub fn sh9(n: glam::Vec3) -> [f32; 9] {
    [
        0.282_095,
        0.488_603 * n.y,
        0.488_603 * n.z,
        0.488_603 * n.x,
        1.092_548 * n.x * n.y,
        1.092_548 * n.y * n.z,
        0.315_392 * (3.0 * n.z * n.z - 1.0),
        1.092_548 * n.x * n.z,
        0.546_274 * (n.x * n.x - n.y * n.y),
    ]
}

/// Direction an equirectangular texel looks at.
///
/// Matches `map_equirect_uv_to_dir` in blade's renderer, so an environment can
/// be handed to either without reinterpreting it.
pub fn equirect_direction(u: f32, v: f32) -> glam::Vec3 {
    let yaw = std::f32::consts::PI * (0.5 - v);
    let pitch = 2.0 * std::f32::consts::PI * (u - 0.5);
    glam::Vec3::new(yaw.cos() * pitch.sin(), yaw.sin(), yaw.cos() * pitch.cos())
}

/// An equirectangular environment, as linear radiance.
#[derive(Clone, Debug)]
pub struct Environment {
    pub width: usize,
    pub height: usize,
    /// Row major, `width * height` texels.
    pub texels: Vec<[f32; 3]>,
}

impl Environment {
    /// A constant environment, which is the simplest thing that can be relit to.
    pub fn uniform(radiance: [f32; 3], width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            texels: vec![radiance; width * height],
        }
    }

    /// A sky with a sun in it, for when there is no captured environment.
    ///
    /// Not a model of anything: a gradient from the horizon to the zenith, a
    /// flat ground below, and a disc of concentrated radiance. What it is for
    /// is having the one property that matters to this renderer and that a
    /// uniform environment does not have — most of the energy in a small part
    /// of the sphere, which is what produces a shadow, a highlight and a
    /// visible difference when the light moves.
    ///
    /// `sun_direction` points from the scene towards the sun. `sun_radiance`
    /// is the radiance inside the disc, which for a plausible daylight sky is
    /// two or three orders of magnitude above the sky around it.
    pub fn sky(
        sun_direction: glam::Vec3,
        sun_radiance: [f32; 3],
        sun_angular_radius: f32,
        width: usize,
        height: usize,
    ) -> Self {
        const ZENITH: [f32; 3] = [0.10, 0.16, 0.32];
        const HORIZON: [f32; 3] = [0.34, 0.38, 0.44];
        const GROUND: [f32; 3] = [0.07, 0.06, 0.05];

        let sun = sun_direction.normalize_or_zero();
        let cos_sun = sun_angular_radius.max(0.0).cos();
        let mut texels = Vec::with_capacity(width * height);
        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let direction = equirect_direction(u, v);
                let elevation = direction.y;
                let mut radiance = if elevation >= 0.0 {
                    let t = elevation.sqrt();
                    let mut sky = [0.0f32; 3];
                    for channel in 0..3 {
                        sky[channel] = HORIZON[channel] * (1.0 - t) + ZENITH[channel] * t;
                    }
                    sky
                } else {
                    GROUND
                };
                if sun.length_squared() > 0.0 && direction.dot(sun) >= cos_sun {
                    radiance = sun_radiance;
                }
                texels.push(radiance);
            }
        }
        Self {
            width,
            height,
            texels,
        }
    }

    pub fn sample(&self, direction: glam::Vec3) -> [f32; 3] {
        // Inverse of `equirect_direction`.
        let yaw = direction.y.clamp(-1.0, 1.0).asin();
        let pitch = direction.x.atan2(direction.z);
        let u = (pitch / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
        let v = (0.5 - yaw / std::f32::consts::PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f32) as usize).min(self.width - 1);
        let y = ((v * self.height as f32) as usize).min(self.height - 1);
        self.texels[y * self.width + x]
    }

    /// The luminance an exposure should be built around.
    ///
    /// The log average, solid angle weighted, which is the photographic key
    /// rather than the arithmetic mean. The difference matters here because
    /// every environment worth relighting under has a sun in it: a source
    /// covering a thousandth of the sphere at four hundred times the sky
    /// carries a quarter of the energy, so an arithmetic mean halves the
    /// exposure and renders the sky — and everything lit by it — dark. The log
    /// average is what a light meter measures instead.
    ///
    /// Radiance is in whatever units the environment was captured or invented
    /// in, and nothing else says how bright the number one is meant to look.
    pub fn key_luminance(&self) -> f32 {
        // Rec. 709 luminance, the same weighting a display applies.
        const WEIGHT: [f32; 3] = [0.2126, 0.7152, 0.0722];
        // Keeps a black texel from taking the whole average to zero.
        const FLOOR: f32 = 1.0e-4;

        let mut total = 0.0f64;
        let mut solid_angle = 0.0f64;
        for y in 0..self.height {
            let v = (y as f32 + 0.5) / self.height as f32;
            let row = (std::f32::consts::PI * v).sin() as f64;
            for x in 0..self.width {
                let texel = self.texels[y * self.width + x];
                let luminance: f32 = texel
                    .iter()
                    .zip(WEIGHT)
                    .map(|(value, weight)| value.max(0.0) * weight)
                    .sum();
                total += (luminance.max(FLOOR)).ln() as f64 * row;
                solid_angle += row;
            }
        }
        if solid_angle > 0.0 {
            ((total / solid_angle) as f32).exp()
        } else {
            0.0
        }
    }

    /// Project onto the diffuse basis, with the Lambertian convolution and the
    /// BRDF's `1 / PI` folded in, so the result is the outgoing radiance of a
    /// unit albedo directly.
    pub fn diffuse_irradiance(&self) -> [[f32; 4]; 9] {
        const BAND_GAIN: [f32; 3] = [
            std::f32::consts::PI,
            2.0 * std::f32::consts::FRAC_PI_3,
            std::f32::consts::FRAC_PI_4,
        ];
        const BAND_OF: [usize; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

        let mut raw = [[0.0f32; 3]; 9];
        let base = (2.0 * std::f32::consts::PI / self.width as f32)
            * (std::f32::consts::PI / self.height as f32);
        for y in 0..self.height {
            let v = (y as f32 + 0.5) / self.height as f32;
            let solid_angle = base * (std::f32::consts::PI * v).sin();
            for x in 0..self.width {
                let u = (x as f32 + 0.5) / self.width as f32;
                let basis = sh9(equirect_direction(u, v));
                let texel = self.texels[y * self.width + x];
                for (index, weight) in basis.iter().enumerate() {
                    for channel in 0..3 {
                        raw[index][channel] += texel[channel] * weight * solid_angle;
                    }
                }
            }
        }

        let mut out = [[0.0f32; 4]; 9];
        for index in 0..9 {
            let gain = BAND_GAIN[BAND_OF[index]] / std::f32::consts::PI;
            for channel in 0..3 {
                out[index][channel] = raw[index][channel] * gain;
            }
        }
        out
    }
}

/// The environment convolved with the GGX lobe at a ladder of roughnesses.
///
/// The first half of the split sum. Prefiltering happens once per environment
/// rather than per pixel, which is the entire point of the approximation.
pub struct SpecularEnvironment {
    pub width: usize,
    pub height: usize,
    /// `SPECULAR_LEVELS` planes of `width * height` texels, roughness ascending.
    pub levels: Vec<Vec<[f32; 4]>>,
}

fn radical_inverse(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn importance_sample_ggx(u: glam::Vec2, roughness: f32) -> glam::Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * u.x;
    let cos_theta = ((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)).max(0.0).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    glam::Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

impl SpecularEnvironment {
    pub fn prefilter(environment: &Environment, width: usize, height: usize) -> Self {
        // Enough to find a concentrated source. A sun a few degrees across
        // covers well under a thousandth of the sphere, so a broad lobe drawing
        // a hundred samples usually misses it entirely and the level comes out
        // dark — which is most of the environment's energy thrown away, and it
        // showed up as several decibels against a path traced reference.
        const SAMPLES: u32 = 4096;

        let mut levels = Vec::with_capacity(SPECULAR_LEVELS as usize);
        for level in 0..SPECULAR_LEVELS {
            let roughness = level as f32 / (SPECULAR_LEVELS - 1) as f32;
            if roughness <= 0.0 {
                // A mirror reflects the environment, so this level is a copy
                // and has to keep every bit of the source's resolution.
                let mut plane = Vec::with_capacity(width * height);
                for y in 0..height {
                    let v = (y as f32 + 0.5) / height as f32;
                    for x in 0..width {
                        let u = (x as f32 + 0.5) / width as f32;
                        let radiance = environment.sample(equirect_direction(u, v));
                        plane.push([radiance[0], radiance[1], radiance[2], 1.0]);
                    }
                }
                levels.push(plane);
                continue;
            }

            // A rough level is smooth by construction, so it is computed small
            // and stretched back up. That is where the cost goes: the same
            // sample budget on a sixteenth of the texels is a far better
            // estimate of each of them, and nothing is lost because there was
            // no detail at this roughness to begin with.
            let small_width = (width >> level).max(8);
            let small_height = (height >> level).max(4);
            let mut small = vec![[0.0f32; 4]; small_width * small_height];
            // Rows are independent, and there are millions of samples behind
            // each level, so this is split across the machine. It is the one
            // thing standing between changing the light and seeing it change.
            let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
            let rows_per_chunk = small_height.div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                for (chunk_index, chunk) in
                    small.chunks_mut(rows_per_chunk * small_width).enumerate()
                {
                    scope.spawn(move || {
                        let first_row = chunk_index * rows_per_chunk;
                        for (row, texels) in chunk.chunks_mut(small_width).enumerate() {
                            let v = ((first_row + row) as f32 + 0.5) / small_height as f32;
                            for (x, texel) in texels.iter_mut().enumerate() {
                                let u = (x as f32 + 0.5) / small_width as f32;
                                // Split sum assumes the view, the normal and the
                                // reflection all agree, which is what lets one
                                // table serve every angle.
                                let normal = equirect_direction(u, v);
                                let up = if normal.z.abs() < 0.999 {
                                    glam::Vec3::Z
                                } else {
                                    glam::Vec3::X
                                };
                                let tangent = up.cross(normal).normalize();
                                let bitangent = normal.cross(tangent);

                                let mut total = [0.0f32; 3];
                                let mut weight = 0.0f32;
                                for index in 0..SAMPLES {
                                    let hammersley = glam::Vec2::new(
                                        index as f32 / SAMPLES as f32,
                                        radical_inverse(index),
                                    );
                                    let local = importance_sample_ggx(hammersley, roughness);
                                    let half =
                                        tangent * local.x + bitangent * local.y + normal * local.z;
                                    let light =
                                        (2.0 * normal.dot(half) * half - normal).normalize();
                                    let cosine = normal.dot(light);
                                    if cosine <= 0.0 {
                                        continue;
                                    }
                                    let radiance = environment.sample(light);
                                    for (accumulated, value) in total.iter_mut().zip(radiance) {
                                        *accumulated += value * cosine;
                                    }
                                    weight += cosine;
                                }
                                if weight > 0.0 {
                                    for accumulated in total.iter_mut() {
                                        *accumulated /= weight;
                                    }
                                }
                                *texel = [total[0], total[1], total[2], 1.0];
                            }
                        }
                    });
                }
            });

            // Back up to the size every level is stored at, so one array can
            // hold the ladder and the shader can blend between two of them.
            let mut plane = vec![[0.0f32; 4]; width * height];
            for y in 0..height {
                let fy = ((y as f32 + 0.5) / height as f32) * small_height as f32 - 0.5;
                let y0 = fy.floor();
                let ty = fy - y0;
                let y0 = (y0 as i64).clamp(0, small_height as i64 - 1) as usize;
                let y1 = (y0 + 1).min(small_height - 1);
                for x in 0..width {
                    let fx = ((x as f32 + 0.5) / width as f32) * small_width as f32 - 0.5;
                    let x0 = fx.floor();
                    let tx = fx - x0;
                    let x0 = (x0 as i64).rem_euclid(small_width as i64) as usize;
                    let x1 = (x0 + 1) % small_width;
                    let mut texel = [0.0f32; 4];
                    for channel in 0..3 {
                        let top = small[y0 * small_width + x0][channel] * (1.0 - tx)
                            + small[y0 * small_width + x1][channel] * tx;
                        let bottom = small[y1 * small_width + x0][channel] * (1.0 - tx)
                            + small[y1 * small_width + x1][channel] * tx;
                        texel[channel] = top * (1.0 - ty) + bottom * ty;
                    }
                    texel[3] = 1.0;
                    plane[y * width + x] = texel;
                }
            }
            levels.push(plane);
        }
        Self {
            width,
            height,
            levels,
        }
    }

    pub fn level_bytes(&self) -> usize {
        self.width * self.height * mem::size_of::<[f32; 4]>()
    }
}

/// One entry of the environment's alias table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AliasEntry {
    /// Probability of keeping this texel rather than taking the alias.
    pub threshold: f32,
    /// Texel to fall back to.
    pub alias: u32,
    /// Probability of this texel being chosen at all, before the solid angle
    /// is divided out to make it a density over directions.
    pub probability: f32,
    pub _padding: f32,
}

/// A way of drawing directions in proportion to the light coming from them.
///
/// Sampling the environment by the cosine and nothing else loses concentrated
/// light: a sun a few degrees across covers well under a thousandth of the
/// sphere, so almost no ray finds it and the estimate comes out dark however
/// many rays are cast. Choosing texels in proportion to what they emit puts
/// the samples where the light is.
///
/// The alias method rather than a search through a cumulative table, because
/// it draws in constant time from one lookup and one comparison, which is what
/// a shader wants.
pub struct EnvironmentSampler {
    pub width: usize,
    pub height: usize,
    pub entries: Vec<AliasEntry>,
}

impl EnvironmentSampler {
    pub fn build(environment: &Environment) -> Self {
        let (width, height) = (environment.width, environment.height);
        let count = width * height;

        // Weight by what a texel emits and how much of the sphere it covers,
        // since a texel near a pole stands for far less of it than one at the
        // equator.
        let base =
            (2.0 * std::f32::consts::PI / width as f32) * (std::f32::consts::PI / height as f32);
        let mut weights = Vec::with_capacity(count);
        let mut total = 0.0f64;
        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            let solid_angle = base * (std::f32::consts::PI * v).sin();
            for x in 0..width {
                let texel = environment.texels[y * width + x];
                let luminance = 0.2126 * texel[0] + 0.7152 * texel[1] + 0.0722 * texel[2];
                let weight = (luminance.max(0.0) * solid_angle) as f64;
                total += weight;
                weights.push(weight);
            }
        }
        if total <= 0.0 {
            // A black environment: draw uniformly rather than dividing by zero.
            total = count as f64;
            weights.iter_mut().for_each(|w| *w = 1.0);
        }

        let mut entries = vec![AliasEntry::default(); count];
        for (index, weight) in weights.iter().enumerate() {
            entries[index].probability = (*weight / total) as f32;
        }

        // Vose's construction: scale the probabilities so the mean is one, then
        // pair every texel that is under with one that is over until none are
        // left.
        let mut scaled: Vec<f64> = weights.iter().map(|w| w / total * count as f64).collect();
        let mut small = Vec::new();
        let mut large = Vec::new();
        for (index, value) in scaled.iter().enumerate() {
            if *value < 1.0 {
                small.push(index);
            } else {
                large.push(index);
            }
        }
        while let (Some(lean), Some(fat)) = (small.pop(), large.pop()) {
            entries[lean].threshold = scaled[lean] as f32;
            entries[lean].alias = fat as u32;
            scaled[fat] -= 1.0 - scaled[lean];
            if scaled[fat] < 1.0 {
                small.push(fat);
            } else {
                large.push(fat);
            }
        }
        // Whatever is left is exactly one by construction, up to rounding.
        for index in small.into_iter().chain(large) {
            entries[index].threshold = 1.0;
            entries[index].alias = index as u32;
        }

        Self {
            width,
            height,
            entries,
        }
    }

    /// Density over directions for the texel a direction falls in.
    pub fn pdf(&self, direction: glam::Vec3) -> f32 {
        let yaw = direction.y.clamp(-1.0, 1.0).asin();
        let pitch = direction.x.atan2(direction.z);
        let u = (pitch / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
        let v = (0.5 - yaw / std::f32::consts::PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f32) as usize).min(self.width - 1);
        let y = ((v * self.height as f32) as usize).min(self.height - 1);
        let solid_angle = (2.0 * std::f32::consts::PI / self.width as f32)
            * (std::f32::consts::PI / self.height as f32)
            * (std::f32::consts::PI * (y as f32 + 0.5) / self.height as f32).sin();
        if solid_angle <= 0.0 {
            return 0.0;
        }
        self.entries[y * self.width + x].probability / solid_angle
    }
}

/// The second half of the split sum: what scales `F0`.
///
/// Lazarov's analytic fit, which spares carrying a lookup table for something
/// that is an approximation to begin with.
pub fn specular_scale(f0: [f32; 3], roughness: f32, n_dot_v: f32) -> [f32; 3] {
    let c0 = [-1.0f32, -0.0275, -0.572, 0.022];
    let c1 = [1.0f32, 0.0425, 1.04, -0.04];
    let r: Vec<f32> = c0.iter().zip(c1).map(|(a, b)| roughness * a + b).collect();
    let a004 = (r[0] * r[0]).min((-9.28 * n_dot_v.max(0.0)).exp2()) * r[0] + r[1];
    let scale = a004 * -1.04 + r[2];
    let bias = a004 * 1.04 + r[3];
    [
        f0[0] * scale + bias,
        f0[1] * scale + bias,
        f0[2] * scale + bias,
    ]
}

/// Depth over which surfels are taken to be the same surface, in units of the
/// nearest one's radius.
///
/// Surfels that overlap on one surface have to be averaged rather than
/// composited: compositing lets the front one dominate, which is what leaves
/// each disc showing its own flat normal. Ones belonging to a *different*
/// surface further along the ray have to occlude rather than average. The band
/// is what separates the two cases, and a couple of radii is the width over
/// which the same sheet can wobble.
pub const SURFACE_BAND: f32 = 2.0;

/// How much of a ray a surfel covers, as a function of how far from its centre
/// the ray passed, in units of its radius squared.
///
/// Taking the nearest surfel and shading it opaquely makes every disc edge a
/// visible discontinuity, because two neighbours that overlap disagree about
/// their normals and the picture switches abruptly from one to the other.
/// Falling off towards the rim instead lets the overlap do what it is for:
/// neighbours blend where they meet, and a silhouette gets a soft edge rather
/// than a staircase.
///
/// The inner part of a disc counts fully and only the rim tapers. A falloff
/// that starts dropping immediately leaves the *interior* of a surface partly
/// transparent — between two disc centres neither one is contributing much —
/// and the background bleeding through there is far worse than the facets the
/// blending is meant to remove.
pub fn coverage(normalized_radius_squared: f32) -> f32 {
    let t = ((normalized_radius_squared - 0.4) / 0.6).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// The prefiltered environment in one direction, at one roughness.
///
/// The first half of the split sum, fetched the way the renderer's sampler
/// fetches it: bilinear, wrapping in longitude and clamping in latitude, and
/// interpolating between the two roughness levels either side. Taking the
/// nearest texel instead disagrees with the shader by more than the whole of
/// everything else wherever the environment has a bright feature in it.
///
/// Extracted so a solver fitting roughness against photographs evaluates the
/// same arithmetic the renderer will, rather than an approximation of it.
pub fn sample_prefiltered(
    specular: &SpecularEnvironment,
    direction: glam::Vec3,
    roughness: f32,
) -> [f32; 3] {
    let scaled = roughness.clamp(0.0, 1.0) * (SPECULAR_LEVELS - 1) as f32;
    let low = scaled.floor() as usize;
    let high = (low + 1).min(SPECULAR_LEVELS as usize - 1);
    let blend = scaled - low as f32;

    let yaw = direction.y.clamp(-1.0, 1.0).asin();
    let pitch = direction.x.atan2(direction.z);
    let u = (pitch / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
    let v = (0.5 - yaw / std::f32::consts::PI).clamp(0.0, 1.0);
    let fx = u * specular.width as f32 - 0.5;
    let fy = v * specular.height as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let wrap_x = |x: i64| x.rem_euclid(specular.width as i64) as usize;
    let clamp_y = |y: i64| y.clamp(0, specular.height as i64 - 1) as usize;
    let (x0, x1) = (wrap_x(x0 as i64), wrap_x(x0 as i64 + 1));
    let (y0, y1) = (clamp_y(y0 as i64), clamp_y(y0 as i64 + 1));
    let fetch = |level: usize| {
        let plane = &specular.levels[level];
        let mut out = [0.0f32; 3];
        for (channel, value) in out.iter_mut().enumerate() {
            let top = plane[y0 * specular.width + x0][channel] * (1.0 - tx)
                + plane[y0 * specular.width + x1][channel] * tx;
            let bottom = plane[y1 * specular.width + x0][channel] * (1.0 - tx)
                + plane[y1 * specular.width + x1][channel] * tx;
            *value = top * (1.0 - ty) + bottom * ty;
        }
        out
    };
    let a = fetch(low);
    let b = fetch(high);
    [
        a[0] + (b[0] - a[0]) * blend,
        a[1] + (b[1] - a[1]) * blend,
        a[2] + (b[2] - a[2]) * blend,
    ]
}

/// Shade one surface point, on the CPU.
///
/// The reference the GPU shader is checked against: the same arithmetic in a
/// place where it can be read and stepped through.
pub fn shade(
    normal: glam::Vec3,
    view: glam::Vec3,
    material: &Material,
    irradiance: &[[f32; 4]; 9],
    specular: &SpecularEnvironment,
) -> [f32; 3] {
    let basis = sh9(normal);
    let mut out = [0.0f32; 3];
    for (coefficient, weight) in irradiance.iter().zip(basis) {
        for channel in 0..3 {
            out[channel] += coefficient[channel] * weight * material.albedo[channel];
        }
    }

    let n_dot_v = normal.dot(view);
    if n_dot_v > 0.0 {
        let reflection = (2.0 * n_dot_v * normal - view).normalize();
        let prefiltered = sample_prefiltered(specular, reflection, material.roughness);
        let gain = specular_scale(material.specular_f0, material.roughness, n_dot_v);
        for (value, (radiance, gain)) in out.iter_mut().zip(prefiltered.into_iter().zip(gain)) {
            *value += radiance * gain;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_environment_lights_a_unit_albedo_to_itself() {
        // A Lambertian surface under constant radiance `L` reflects exactly
        // `albedo * L`, whatever way it faces.
        let environment = Environment::uniform([0.4, 0.5, 0.6], 64, 32);
        let irradiance = environment.diffuse_irradiance();
        let basis = sh9(glam::Vec3::new(0.3, 0.8, 0.5).normalize());
        let mut lit = [0.0f32; 3];
        for (coefficient, weight) in irradiance.iter().zip(basis) {
            for channel in 0..3 {
                lit[channel] += coefficient[channel] * weight;
            }
        }
        for (channel, expected) in [0.4f32, 0.5, 0.6].iter().enumerate() {
            assert!(
                (lit[channel] - expected).abs() < 5.0e-3,
                "expected the environment back, got {lit:?}"
            );
        }
    }

    #[test]
    fn the_procedural_sky_puts_the_sun_where_it_was_asked_to() {
        let direction = glam::Vec3::new(0.4, 0.7, -0.3).normalize();
        let sky = Environment::sky(direction, [200.0; 3], 0.05, 256, 128);

        // The sun is where it was put, and nowhere else.
        assert!(sky.sample(direction)[0] > 100.0);
        assert!(sky.sample(-direction)[0] < 1.0);
        // Below the horizon is ground rather than sky, and both are dim
        // against the sun: that contrast is the whole reason to prefer this
        // over a uniform environment.
        assert!(sky.sample(glam::Vec3::NEG_Y)[2] < sky.sample(glam::Vec3::Y)[2]);
        let bright = sky.texels.iter().filter(|texel| texel[0] > 100.0).count();
        let fraction = bright as f32 / sky.texels.len() as f32;
        assert!(
            fraction > 0.0 && fraction < 0.01,
            "the sun covers {bright} of {} texels",
            sky.texels.len()
        );
    }

    #[test]
    fn the_exposure_key_ignores_the_sun_and_follows_the_sky() {
        // A sun carrying a quarter of the energy must not halve the exposure,
        // or every environment with one in it renders dark.
        let plain = Environment::uniform([0.2; 3], 128, 64);
        let mut sunny = plain.clone();
        for index in 0..12 {
            sunny.texels[20 * 128 + 40 + index] = [400.0; 3];
        }
        let plain_key = plain.key_luminance();
        let sunny_key = sunny.key_luminance();
        assert!(
            (sunny_key / plain_key - 1.0).abs() < 0.1,
            "the sun moved the key from {plain_key:.4} to {sunny_key:.4}"
        );
        // And it still tracks the overall level, or it would not be an
        // exposure at all.
        let bright = Environment::uniform([2.0; 3], 128, 64);
        assert!(bright.key_luminance() > 5.0 * plain_key);
    }

    #[test]
    fn a_mirror_prefilters_to_the_environment_itself() {
        let mut environment = Environment::uniform([0.1; 3], 64, 32);
        environment.texels[10 * 64 + 20] = [8.0, 8.0, 8.0];
        let specular = SpecularEnvironment::prefilter(&environment, 64, 32);
        // Level zero is a perfect mirror, so it has to keep the bright texel
        // rather than average it away.
        let peak = specular.levels[0]
            .iter()
            .map(|t| t[0])
            .fold(0.0f32, f32::max);
        assert!(peak > 7.0, "the mirror level lost the highlight: {peak}");
        // The roughest level cannot, and should have spread it out.
        let rough_peak = specular.levels[SPECULAR_LEVELS as usize - 1]
            .iter()
            .map(|t| t[0])
            .fold(0.0f32, f32::max);
        assert!(
            rough_peak < peak,
            "the rough level kept as much as the mirror: {rough_peak} against {peak}"
        );
    }

    #[test]
    fn a_metal_has_no_diffuse_response() {
        let environment = Environment::uniform([1.0; 3], 32, 16);
        let irradiance = environment.diffuse_irradiance();
        let specular = SpecularEnvironment::prefilter(&environment, 32, 16);
        let metal = Material {
            albedo: [0.0; 3],
            roughness: 0.3,
            specular_f0: [0.9, 0.7, 0.3],
            _padding: 0.0,
        };
        let normal = glam::Vec3::Y;
        let lit = shade(normal, normal, &metal, &irradiance, &specular);
        // Everything it sends back is the reflection, which under a white
        // furnace is tinted by its reflectance rather than neutral.
        assert!(
            lit[0] > lit[1] && lit[1] > lit[2],
            "expected a tint: {lit:?}"
        );
    }

    #[test]
    fn validation_catches_a_dangling_material() {
        let model = RelightModel {
            surfels: vec![Surfel {
                center: [0.0; 3],
                radius: 1.0,
                normal: [0.0, 1.0, 0.0],
                material: 3,
            }],
            materials: vec![Material::default()],
        };
        assert!(model.validate().unwrap_err().contains("material 3"));
    }

    #[test]
    fn validation_catches_an_unnormalised_normal() {
        let model = RelightModel {
            surfels: vec![Surfel {
                center: [0.0; 3],
                radius: 1.0,
                normal: [0.0, 2.0, 0.0],
                material: 0,
            }],
            materials: vec![Material::default()],
        };
        assert!(model.validate().unwrap_err().contains("length"));
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;

    #[test]
    fn the_alias_table_reproduces_the_environment_it_was_built_from() {
        // A dim sky with one bright patch, which is the case cosine sampling
        // cannot handle and this exists for.
        let (width, height) = (64usize, 32usize);
        let mut environment = Environment::uniform([0.05; 3], width, height);
        for y in 12..15 {
            for x in 20..24 {
                environment.texels[y * width + x] = [50.0; 3];
            }
        }
        let sampler = EnvironmentSampler::build(&environment);

        // Drawing from the table has to reproduce the probabilities that went
        // into it, or the density the estimator divides by is a lie.
        //
        // The two draws per sample come from separate hashed streams. A linear
        // congruential generator's consecutive outputs are correlated, and
        // since one picks the texel and the other decides whether to keep it,
        // that correlation lands squarely on what is being measured.
        let mut counts = vec![0u32; width * height];
        let draws = 400_000u32;
        let hash = |mut x: u32| {
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b);
            x ^= x >> 16;
            x as f32 / u32::MAX as f32
        };
        for draw in 0..draws {
            let index = ((hash(draw.wrapping_mul(2)) * (width * height) as f32) as usize)
                .min(width * height - 1);
            let entry = sampler.entries[index];
            let chosen = if hash(draw.wrapping_mul(2).wrapping_add(1)) < entry.threshold {
                index
            } else {
                entry.alias as usize
            };
            counts[chosen] += 1;
        }

        // Only where enough draws are expected for the frequency to mean
        // something; a texel expecting four of them says nothing either way.
        let mut worst = 0.0f32;
        let mut checked = 0usize;
        for (index, entry) in sampler.entries.iter().enumerate() {
            if entry.probability * (draws as f32) < 200.0 {
                continue;
            }
            checked += 1;
            let observed = counts[index] as f32 / draws as f32;
            worst = worst.max((observed - entry.probability).abs() / entry.probability);
        }
        assert!(checked > 8, "too few texels were worth checking: {checked}");
        assert!(
            worst < 0.1,
            "drawn frequencies are {:.1} % off the table's own probabilities \
             over {checked} texels",
            100.0 * worst
        );
    }

    #[test]
    fn the_bright_patch_takes_most_of_the_probability() {
        let (width, height) = (64usize, 32usize);
        let mut environment = Environment::uniform([0.05; 3], width, height);
        for y in 12..15 {
            for x in 20..24 {
                environment.texels[y * width + x] = [50.0; 3];
            }
        }
        let sampler = EnvironmentSampler::build(&environment);
        let mut patch = 0.0f32;
        for y in 12..15 {
            for x in 20..24 {
                patch += sampler.entries[y * width + x].probability;
            }
        }
        // Twelve texels out of two thousand, carrying most of the light.
        assert!(
            patch > 0.5,
            "the patch should dominate the sampling, got {:.3}",
            patch
        );
    }

    #[test]
    fn the_density_integrates_to_one_over_the_sphere() {
        let (width, height) = (64usize, 32usize);
        let mut environment = Environment::uniform([0.2; 3], width, height);
        environment.texels[13 * width + 21] = [30.0; 3];
        let sampler = EnvironmentSampler::build(&environment);
        let base =
            (2.0 * std::f32::consts::PI / width as f32) * (std::f32::consts::PI / height as f32);
        let mut integral = 0.0f64;
        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            let solid_angle = base * (std::f32::consts::PI * v).sin();
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                integral += (sampler.pdf(equirect_direction(u, v)) * solid_angle) as f64;
            }
        }
        assert!(
            (integral - 1.0).abs() < 0.02,
            "a density has to integrate to one, got {integral:.4}"
        );
    }
}
