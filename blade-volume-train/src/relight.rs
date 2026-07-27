//! Reading the synthetic relighting dataset, and the bound it can establish.
//!
//! The dataset is written by blade's `relight_data` test: many views of one
//! scene under several environments, as linear radiance, next to the albedo,
//! roughness, normal and depth the renderer actually used.
//!
//! What this module is for is the question that has to be answered before any
//! solver is worth writing. A relightable model has to do two things — recover
//! the material, and re-render it under a light it has not seen — and only the
//! second one is evidence, because the first is satisfied exactly by baking all
//! the illumination into the albedo. So the useful measurement holds geometry
//! at ground truth, recovers albedo from a subset of the environments, relights
//! into one that was held out, and scores that. Whatever comes out is an upper
//! bound: a real reconstruction also has to estimate the geometry this hands
//! over for free.

use std::{fs, path};

/// Spherical harmonic coefficients of the diffuse response, per colour channel.
///
/// These are not the radiance coefficients of the environment. The Lambertian
/// convolution and the `1 / PI` of the BRDF are folded in, so that outgoing
/// radiance is just `albedo * shade(normal)` with no further constants.
#[derive(Clone, Copy, Debug, Default)]
pub struct Irradiance {
    pub coefficients: [[f32; 3]; 9],
}

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

impl Irradiance {
    /// Outgoing diffuse radiance for a unit albedo facing `n`.
    pub fn shade(&self, n: glam::Vec3) -> [f32; 3] {
        let basis = sh9(n);
        let mut out = [0.0f32; 3];
        for (coefficient, weight) in self.coefficients.iter().zip(basis) {
            for channel in 0..3 {
                out[channel] += coefficient[channel] * weight;
            }
        }
        out
    }

    /// Project an equirectangular environment onto the diffuse basis.
    ///
    /// The band gains are the Lambertian ones from Ramamoorthi and Hanrahan;
    /// dividing them by `PI` here is what lets `shade` be used directly as the
    /// outgoing radiance of a unit albedo.
    pub fn project(texels: &[[f32; 3]], width: usize, height: usize) -> Self {
        const BAND_GAIN: [f32; 3] = [
            std::f32::consts::PI,
            2.0 * std::f32::consts::FRAC_PI_3,
            std::f32::consts::FRAC_PI_4,
        ];
        const BAND_OF: [usize; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

        let mut raw = [[0.0f32; 3]; 9];
        // Solid angle of one texel, without the latitude term.
        let base =
            (2.0 * std::f32::consts::PI / width as f32) * (std::f32::consts::PI / height as f32);
        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            // The generator's mapping puts the polar angle at `PI * v`, so the
            // texels shrink towards the poles by its sine.
            let solid_angle = base * (std::f32::consts::PI * v).sin();
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let dir = equirect_direction(u, v);
                let basis = sh9(dir);
                let texel = texels[y * width + x];
                for (index, weight) in basis.iter().enumerate() {
                    for channel in 0..3 {
                        raw[index][channel] += texel[channel] * weight * solid_angle;
                    }
                }
            }
        }

        let mut coefficients = [[0.0f32; 3]; 9];
        for index in 0..9 {
            let gain = BAND_GAIN[BAND_OF[index]] / std::f32::consts::PI;
            for channel in 0..3 {
                coefficients[index][channel] = raw[index][channel] * gain;
            }
        }
        Self { coefficients }
    }
}

/// Direction an equirectangular texel looks at.
///
/// Has to match `map_equirect_uv_to_dir` in blade's `env-light.inc.wgsl`, and
/// `env_direction` in the generator, or the recovered light is not the light
/// that was rendered.
pub fn equirect_direction(u: f32, v: f32) -> glam::Vec3 {
    let yaw = std::f32::consts::PI * (0.5 - v);
    let pitch = 2.0 * std::f32::consts::PI * (u - 0.5);
    glam::Vec3::new(yaw.cos() * pitch.sin(), yaw.sin(), yaw.cos() * pitch.cos())
}

// ------------------------------------------------------------------- manifest

/// One camera and everything rendered from it.
#[derive(Clone, Debug)]
pub struct View {
    pub index: usize,
    pub position: glam::Vec3,
    pub orientation: glam::Quat,
    pub fov_y: f32,
    /// Radiance file per environment, in the dataset's environment order.
    pub radiance: Vec<path::PathBuf>,
    pub material: path::PathBuf,
    pub geometry: path::PathBuf,
    /// Present once the generator started writing the specular plane.
    pub specular: Option<path::PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub root: path::PathBuf,
    pub width: usize,
    pub height: usize,
    pub environments: Vec<String>,
    pub environment_files: Vec<path::PathBuf>,
    pub views: Vec<View>,
}

/// Pull `key = ...` out of a line, if that is what the line is.
fn key_of(line: &str) -> Option<(&str, &str)> {
    let (key, value) = line.split_once('=')?;
    Some((key.trim(), value.trim()))
}

fn unquote(value: &str) -> String {
    value.trim().trim_matches('"').to_string()
}

fn numbers(value: &str) -> Vec<f32> {
    value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|piece| piece.trim().parse::<f32>().ok())
        .collect()
}

impl Dataset {
    /// Read the manifest the generator writes.
    ///
    /// This understands the subset of TOML that generator emits rather than
    /// the format at large: tables, arrays of tables, and scalar, array and
    /// inline-table values on one line each. Pulling in a parser for a file
    /// this side also writes did not seem worth the dependency.
    pub fn load(root: &path::Path) -> Result<Self, String> {
        let text = fs::read_to_string(root.join("manifest.toml"))
            .map_err(|e| format!("cannot read the manifest in {}: {e}", root.display()))?;

        let mut width = 0usize;
        let mut height = 0usize;
        let mut environments = Vec::new();
        let mut environment_files = Vec::new();
        let mut views: Vec<View> = Vec::new();
        let mut section = String::new();
        let mut in_radiance = false;

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') {
                section = line.to_string();
                in_radiance = false;
                match section.as_str() {
                    "[[environment]]" => {
                        environments.push(String::new());
                        environment_files.push(path::PathBuf::new());
                    }
                    "[[view]]" => views.push(View {
                        index: views.len(),
                        position: glam::Vec3::ZERO,
                        orientation: glam::Quat::IDENTITY,
                        fov_y: 0.0,
                        radiance: Vec::new(),
                        material: path::PathBuf::new(),
                        geometry: path::PathBuf::new(),
                        specular: None,
                    }),
                    _ => {}
                }
                continue;
            }

            // Entries of the radiance array, one inline table per line.
            if in_radiance {
                if line.starts_with(']') {
                    in_radiance = false;
                    continue;
                }
                if let Some(view) = views.last_mut() {
                    for field in line
                        .trim_matches(|c| c == '{' || c == '}' || c == ',')
                        .split(',')
                    {
                        if let Some((key, value)) = key_of(field) {
                            if key == "file" {
                                view.radiance.push(root.join(unquote(value)));
                            }
                        }
                    }
                }
                continue;
            }

            let Some((key, value)) = key_of(line) else {
                continue;
            };
            match section.as_str() {
                "[dataset]" => match key {
                    "width" => width = value.parse().unwrap_or(0),
                    "height" => height = value.parse().unwrap_or(0),
                    _ => {}
                },
                "[[environment]]" => match key {
                    "name" => *environments.last_mut().unwrap() = unquote(value),
                    // `file` is what older datasets called the 8-bit map;
                    // `radiance` is the float one that replaced it.
                    "file" | "radiance" => {
                        *environment_files.last_mut().unwrap() = root.join(unquote(value))
                    }
                    _ => {}
                },
                "[[view]]" => {
                    let view = views.last_mut().unwrap();
                    match key {
                        "position" => {
                            let v = numbers(value);
                            if v.len() == 3 {
                                view.position = glam::Vec3::new(v[0], v[1], v[2]);
                            }
                        }
                        "orientation" => {
                            let v = numbers(value);
                            if v.len() == 4 {
                                view.orientation = glam::Quat::from_xyzw(v[0], v[1], v[2], v[3]);
                            }
                        }
                        "fov_y" => view.fov_y = value.parse().unwrap_or(0.0),
                        "material" => view.material = root.join(unquote(value)),
                        "geometry" => view.geometry = root.join(unquote(value)),
                        "specular" => view.specular = Some(root.join(unquote(value))),
                        "radiance" => in_radiance = true,
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        if width == 0 || height == 0 {
            return Err("the manifest carries no image size".to_string());
        }
        if views.is_empty() {
            return Err("the manifest lists no views".to_string());
        }
        if environments.is_empty() {
            return Err("the manifest lists no environments".to_string());
        }
        for view in &views {
            if view.radiance.len() != environments.len() {
                return Err(format!(
                    "view {} has {} radiance files for {} environments",
                    view.index,
                    view.radiance.len(),
                    environments.len()
                ));
            }
        }
        Ok(Self {
            root: root.to_path_buf(),
            width,
            height,
            environments,
            environment_files,
            views,
        })
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Read one of the raw `rgba32f` planes the generator writes.
    pub fn read_plane(&self, path: &path::Path) -> Result<Vec<[f32; 4]>, String> {
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let expected = self.pixel_count() * 16;
        if bytes.len() != expected {
            return Err(format!(
                "{} is {} bytes, expected {expected}",
                path.display(),
                bytes.len()
            ));
        }
        let mut out = Vec::with_capacity(self.pixel_count());
        for texel in bytes.chunks_exact(16) {
            let mut values = [0.0f32; 4];
            for (index, chunk) in texel.chunks_exact(4).enumerate() {
                values[index] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out.push(values);
        }
        Ok(out)
    }

    /// Prefilter every environment for the specular lobe.
    pub fn environment_specular(&self) -> Result<Vec<SpecularEnvironment>, String> {
        let mut out = Vec::with_capacity(self.environment_files.len());
        for file in &self.environment_files {
            let (texels, width, height) = read_environment_plane(file)?;
            out.push(SpecularEnvironment::prefilter(&texels, width, height));
        }
        Ok(out)
    }

    /// Project every environment onto the diffuse basis.
    ///
    /// The maps are stored as the 8-bit linear values the renderer sampled, so
    /// this decodes them the same way rather than applying a transfer curve.
    pub fn environment_irradiance(&self) -> Result<Vec<Irradiance>, String> {
        let mut out = Vec::with_capacity(self.environment_files.len());
        for file in &self.environment_files {
            let (texels, width, height) = read_environment_plane(file)?;
            out.push(Irradiance::project(&texels, width, height));
        }
        Ok(out)
    }
}

/// Read an environment map as linear radiance.
///
/// A float plane when the generator wrote one, and an 8-bit PNG otherwise, so
/// datasets made before the maps went floating point still load. The two are
/// not equivalent: a sun a hundred times brighter than its sky survives only
/// in the first, and that contrast is what a specular fit needs.
pub fn read_environment_plane(path: &path::Path) -> Result<(Vec<[f32; 3]>, usize, usize), String> {
    if path.extension().and_then(|e| e.to_str()) == Some("f32") {
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let texel_count = bytes.len() / 16;
        // Equirectangular maps are twice as wide as they are tall.
        let height = ((texel_count / 2) as f64).sqrt().round() as usize;
        let width = height * 2;
        if width * height != texel_count {
            return Err(format!(
                "{} holds {texel_count} texels, which is not a 2:1 map",
                path.display()
            ));
        }
        let mut texels = Vec::with_capacity(texel_count);
        for texel in bytes.chunks_exact(16) {
            let mut rgb = [0.0f32; 3];
            for (channel, chunk) in texel.chunks_exact(4).take(3).enumerate() {
                rgb[channel] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            texels.push(rgb);
        }
        return Ok((texels, width, height));
    }

    let decoded = image::open(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let texels = decoded
        .pixels()
        .map(|p| {
            [
                p.0[0] as f32 / 255.0,
                p.0[1] as f32 / 255.0,
                p.0[2] as f32 / 255.0,
            ]
        })
        .collect();
    Ok((texels, width as usize, height as usize))
}

// -------------------------------------------------------------------- specular

/// Roughness levels the specular environment is prefiltered at.
const SPECULAR_LEVELS: usize = 10;
/// Equirectangular resolution of each prefiltered level.
///
/// This has to resolve the source: a sun a few degrees across is sub-texel at
/// 64x32, so the mirror end of the ladder smears it into a dim smudge and the
/// model cannot render the very highlight that identifies a roughness.
const SPECULAR_SIZE: (usize, usize) = (256, 128);

/// The environment convolved with the GGX lobe at a ladder of roughnesses.
///
/// This is the first half of the split-sum approximation: the light a mirror
/// of the given roughness sees along a reflection direction, with the BRDF's
/// own scale left for [`specular_scale`] to supply.
pub struct SpecularEnvironment {
    levels: Vec<Vec<[f32; 3]>>,
}

/// Van der Corput radical inverse, for the Hammersley sequence.
fn radical_inverse(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

/// A half vector drawn from the GGX distribution around `+Z`.
fn importance_sample_ggx(u: glam::Vec2, roughness: f32) -> glam::Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * u.x;
    let cos_theta = ((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)).max(0.0).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    glam::Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

fn sample_equirect(texels: &[[f32; 3]], width: usize, height: usize, dir: glam::Vec3) -> [f32; 3] {
    // Inverse of `equirect_direction`.
    let yaw = dir.y.clamp(-1.0, 1.0).asin();
    let pitch = dir.x.atan2(dir.z);
    let u = (pitch / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
    let v = (0.5 - yaw / std::f32::consts::PI).clamp(0.0, 1.0);
    let x = ((u * width as f32) as usize).min(width - 1);
    let y = ((v * height as f32) as usize).min(height - 1);
    texels[y * width + x]
}

impl SpecularEnvironment {
    /// Convolve an environment with the GGX lobe at each roughness level.
    pub fn prefilter(texels: &[[f32; 3]], width: usize, height: usize) -> Self {
        const SAMPLES: u32 = 128;
        let (out_width, out_height) = SPECULAR_SIZE;
        let mut levels = Vec::with_capacity(SPECULAR_LEVELS);
        for level in 0..SPECULAR_LEVELS {
            let roughness = level as f32 / (SPECULAR_LEVELS - 1) as f32;
            let mut plane = vec![[0.0f32; 3]; out_width * out_height];
            for y in 0..out_height {
                let v = (y as f32 + 0.5) / out_height as f32;
                for x in 0..out_width {
                    let u = (x as f32 + 0.5) / out_width as f32;
                    // Split-sum assumes the view and the normal agree with the
                    // reflection direction, which is what makes one table
                    // serve every viewing angle.
                    let normal = equirect_direction(u, v);
                    if roughness <= 0.0 {
                        plane[y * out_width + x] = sample_equirect(texels, width, height, normal);
                        continue;
                    }
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
                        let hammersley =
                            glam::Vec2::new(index as f32 / SAMPLES as f32, radical_inverse(index));
                        let local = importance_sample_ggx(hammersley, roughness);
                        let half = tangent * local.x + bitangent * local.y + normal * local.z;
                        let light = (2.0 * normal.dot(half) * half - normal).normalize();
                        let cosine = normal.dot(light);
                        if cosine <= 0.0 {
                            continue;
                        }
                        let radiance = sample_equirect(texels, width, height, light);
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
                    plane[y * out_width + x] = total;
                }
            }
            levels.push(plane);
        }
        Self { levels }
    }

    /// Prefiltered radiance along `reflection` at `roughness`, interpolated
    /// between the two neighbouring levels.
    pub fn sample(&self, reflection: glam::Vec3, roughness: f32) -> [f32; 3] {
        let (width, height) = SPECULAR_SIZE;
        let scaled = roughness.clamp(0.0, 1.0) * (SPECULAR_LEVELS - 1) as f32;
        let low = scaled.floor() as usize;
        let high = (low + 1).min(SPECULAR_LEVELS - 1);
        let blend = scaled - low as f32;
        let a = sample_equirect(&self.levels[low], width, height, reflection);
        let b = sample_equirect(&self.levels[high], width, height, reflection);
        [
            a[0] + (b[0] - a[0]) * blend,
            a[1] + (b[1] - a[1]) * blend,
            a[2] + (b[2] - a[2]) * blend,
        ]
    }
}

/// The second half of the split sum: the scale and bias applied to `F0`.
///
/// Lazarov's analytic fit to the environment BRDF, which avoids carrying a
/// lookup table around for an approximation that is already one.
pub fn specular_scale(f0: [f32; 3], roughness: f32, n_dot_v: f32) -> [f32; 3] {
    let c0 = glam::Vec4::new(-1.0, -0.0275, -0.572, 0.022);
    let c1 = glam::Vec4::new(1.0, 0.0425, 1.04, -0.04);
    let r = glam::Vec4::new(
        roughness * c0.x + c1.x,
        roughness * c0.y + c1.y,
        roughness * c0.z + c1.z,
        roughness * c0.w + c1.w,
    );
    let a004 = (r.x * r.x).min((-9.28 * n_dot_v.max(0.0)).exp2()) * r.x + r.y;
    let scale = a004 * -1.04 + r.z;
    let bias = a004 * 1.04 + r.w;
    [
        f0[0] * scale + bias,
        f0[1] * scale + bias,
        f0[2] * scale + bias,
    ]
}

/// Outgoing specular radiance for one sample under one environment.
pub fn specular_radiance(sample: &Sample, environment: &SpecularEnvironment) -> [f32; 3] {
    let n_dot_v = sample.normal.dot(sample.view);
    if n_dot_v <= 0.0 {
        return [0.0; 3];
    }
    let reflection = (2.0 * n_dot_v * sample.normal - sample.view).normalize();
    let prefiltered = environment.sample(reflection, sample.roughness);
    let scale = specular_scale(sample.specular_f0, sample.roughness, n_dot_v);
    [
        prefiltered[0] * scale[0],
        prefiltered[1] * scale[1],
        prefiltered[2] * scale[2],
    ]
}

// --------------------------------------------------------------------- solving

/// One surface sample: everything a shading model needs about a pixel.
pub struct Sample {
    pub normal: glam::Vec3,
    pub albedo_truth: [f32; 3],
    /// Specular reflectance at normal incidence. A metal keeps its base colour
    /// here and has none in the diffuse channel, so the two together are what
    /// identify the material.
    pub specular_f0: [f32; 3],
    pub roughness: f32,
    /// Direction from the surface back towards the camera.
    pub view: glam::Vec3,
    /// Where the surface is in the world, so observations of the same point
    /// from different cameras can be recognised as such.
    pub position: glam::Vec3,
    /// Which view this came from, and where in it, so a prediction can be put
    /// back into image space to be looked at.
    pub view_index: usize,
    pub pixel: usize,
    /// Observed radiance per environment, in dataset order.
    pub radiance: Vec<[f32; 3]>,
}

/// Gather the foreground pixels of every view.
pub fn gather_samples(dataset: &Dataset) -> Result<Vec<Sample>, String> {
    let mut samples = Vec::new();
    for view in &dataset.views {
        let geometry = dataset.read_plane(&view.geometry)?;
        let material = dataset.read_plane(&view.material)?;
        let specular = match view.specular {
            Some(ref path) => Some(dataset.read_plane(path)?),
            None => None,
        };
        let mut radiance = Vec::with_capacity(dataset.environments.len());
        for file in &view.radiance {
            radiance.push(dataset.read_plane(file)?);
        }
        // The renderer widens the vertical field of view by the aspect ratio;
        // reproducing that exactly is what makes the reconstructed ray the one
        // that was actually traced.
        let tan_half_y = (0.5 * view.fov_y).tan();
        let tan_half_x = tan_half_y * dataset.width as f32 / dataset.height as f32;
        for pixel in 0..dataset.pixel_count() {
            // Negative distance is the generator's miss marker.
            if geometry[pixel][3] <= 0.0 {
                continue;
            }
            let normal =
                glam::Vec3::new(geometry[pixel][0], geometry[pixel][1], geometry[pixel][2]);
            if normal.length_squared() < 0.5 {
                continue;
            }
            let x = (pixel % dataset.width) as f32 + 0.5;
            let y = (pixel / dataset.width) as f32 + 0.5;
            let ndc = glam::Vec2::new(
                x / (0.5 * dataset.width as f32) - 1.0,
                y / (0.5 * dataset.height as f32) - 1.0,
            );
            // Camera space is right handed with -Z forward, and the film is
            // flipped vertically, matching `get_ray_direction_at`.
            let local = glam::Vec3::new(ndc.x * tan_half_x, -ndc.y * tan_half_y, -1.0);
            let direction = (view.orientation * local).normalize();
            // The generator stores distance along the ray rather than a
            // projected depth, so the world position is the ray walked out.
            let position = view.position + direction * geometry[pixel][3];
            samples.push(Sample {
                view_index: view.index,
                pixel,
                normal: normal.normalize(),
                albedo_truth: [material[pixel][0], material[pixel][1], material[pixel][2]],
                specular_f0: match specular {
                    Some(ref plane) => [plane[pixel][0], plane[pixel][1], plane[pixel][2]],
                    None => [0.0; 3],
                },
                roughness: material[pixel][3],
                view: -direction,
                position,
                radiance: radiance
                    .iter()
                    .map(|plane| [plane[pixel][0], plane[pixel][1], plane[pixel][2]])
                    .collect(),
            });
        }
    }
    Ok(samples)
}

/// Least-squares albedo for one sample, given the lights it was seen under.
///
/// With one environment this is just division, and it reproduces that
/// environment exactly whatever the albedo means — which is precisely the
/// failure the held-out score is there to expose.
pub fn solve_albedo(sample: &Sample, lights: &[(usize, Irradiance)]) -> [f32; 3] {
    let mut numerator = [0.0f32; 3];
    let mut denominator = [0.0f32; 3];
    for &(environment, ref irradiance) in lights {
        let shade = irradiance.shade(sample.normal);
        let observed = sample.radiance[environment];
        for channel in 0..3 {
            numerator[channel] += shade[channel] * observed[channel];
            denominator[channel] += shade[channel] * shade[channel];
        }
    }
    let mut albedo = [0.0f32; 3];
    for channel in 0..3 {
        albedo[channel] = if denominator[channel] > 1e-8 {
            (numerator[channel] / denominator[channel]).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    albedo
}

// ------------------------------------------------------------------- elements

/// A patch of surface, and every pixel that ever landed on it.
///
/// The per-pixel fit solves each observation on its own, so a surface point is
/// only ever seen once per environment and the specular lobe is
/// indistinguishable from a brighter albedo. Gathering the views that saw the
/// same point is what makes the two separable: the diffuse part is the same
/// from every direction and the lobe is not, so disagreement between views is
/// the signal that says how much of the radiance was which.
pub struct Element {
    /// Indices into the sample list, all landing on this patch.
    pub samples: Vec<u32>,
    pub albedo: [f32; 3],
    pub specular_f0: [f32; 3],
    pub roughness: f32,
}

/// Group samples that share a voxel of world space.
///
/// Voxels rather than a nearest-neighbour merge: a surface point is defined
/// here by where it is, and two cameras that disagree by less than the voxel
/// are looking at the same thing for our purposes. The size is the usual
/// trade — too large and distinct materials get averaged together, too small
/// and each element is seen once and nothing has been fused.
pub fn build_elements(samples: &[Sample], voxel: f32) -> Vec<Element> {
    use std::collections::HashMap;
    let mut by_cell: HashMap<[i32; 3], usize> = HashMap::new();
    let mut elements: Vec<Element> = Vec::new();
    let inverse = 1.0 / voxel.max(1e-6);
    for (index, sample) in samples.iter().enumerate() {
        let cell = [
            (sample.position.x * inverse).floor() as i32,
            (sample.position.y * inverse).floor() as i32,
            (sample.position.z * inverse).floor() as i32,
        ];
        let slot = *by_cell.entry(cell).or_insert_with(|| {
            elements.push(Element {
                samples: Vec::new(),
                // A mid grey dielectric, so the fit starts somewhere neutral
                // rather than at the answer.
                albedo: [0.5; 3],
                specular_f0: [0.04; 3],
                roughness: 0.5,
            });
            elements.len() - 1
        });
        elements[slot].samples.push(index as u32);
    }
    elements
}

/// Group samples by the material they were authored with, rather than by where
/// they are.
///
/// An oracle grouping, and deliberately so. If a fit that pools every
/// observation of one material still cannot recover its roughness, then the
/// information is not in the data and no amount of clustering or prior will
/// put it there. If it can, then what defeats the per-element fit is being
/// asked to determine a BRDF from one patch of surface, and the answer is to
/// share materials rather than to gather more views.
pub fn build_material_groups(samples: &[Sample]) -> Vec<Element> {
    use std::collections::HashMap;
    let mut by_material: HashMap<[i32; 4], usize> = HashMap::new();
    let mut elements: Vec<Element> = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        let key = [
            (sample.roughness * 100.0).round() as i32,
            (sample.specular_f0[0] * 100.0).round() as i32,
            (sample.specular_f0[1] * 100.0).round() as i32,
            (sample.specular_f0[2] * 100.0).round() as i32,
        ];
        let slot = *by_material.entry(key).or_insert_with(|| {
            elements.push(Element {
                samples: Vec::new(),
                albedo: [0.5; 3],
                specular_f0: [0.04; 3],
                roughness: 0.5,
            });
            elements.len() - 1
        });
        elements[slot].samples.push(index as u32);
    }
    elements
}

/// Seed every element's reflectance and roughness from the truth.
///
/// For the comparison against the per-pixel fit, which was handed both: frozen
/// at a neutral guess they would be plain wrong, and the resulting numbers
/// would say nothing about fusing views.
pub fn seed_elements_from_truth(elements: &mut [Element], samples: &[Sample]) {
    for element in elements.iter_mut() {
        let mut specular_f0 = [0.0f32; 3];
        let mut roughness = 0.0f32;
        for &index in &element.samples {
            let sample = &samples[index as usize];
            for (total, value) in specular_f0.iter_mut().zip(sample.specular_f0) {
                *total += value;
            }
            roughness += sample.roughness;
        }
        let count = element.samples.len().max(1) as f32;
        for (target, total) in element.specular_f0.iter_mut().zip(specular_f0) {
            *target = total / count;
        }
        element.roughness = roughness / count;
    }
}

/// Predicted radiance for one observation under one environment.
pub fn predict(
    sample: &Sample,
    albedo: [f32; 3],
    specular_f0: [f32; 3],
    roughness: f32,
    irradiance: &Irradiance,
    specular: &SpecularEnvironment,
) -> [f32; 3] {
    let shade = irradiance.shade(sample.normal);
    let n_dot_v = sample.normal.dot(sample.view);
    let mut out = [
        albedo[0] * shade[0],
        albedo[1] * shade[1],
        albedo[2] * shade[2],
    ];
    if n_dot_v > 0.0 {
        let reflection = (2.0 * n_dot_v * sample.normal - sample.view).normalize();
        let prefiltered = specular.sample(reflection, roughness);
        let scale = specular_scale(specular_f0, roughness, n_dot_v);
        for channel in 0..3 {
            out[channel] += prefiltered[channel] * scale[channel];
        }
    }
    out
}

/// Adam, with the moments kept alongside the parameter they belong to.
#[derive(Clone, Copy, Default)]
struct Moment {
    first: f32,
    second: f32,
}

impl Moment {
    fn step(&mut self, gradient: f32, rate: f32, iteration: i32) -> f32 {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        self.first = BETA1 * self.first + (1.0 - BETA1) * gradient;
        self.second = BETA2 * self.second + (1.0 - BETA2) * gradient * gradient;
        let corrected_first = self.first / (1.0 - BETA1.powi(iteration));
        let corrected_second = self.second / (1.0 - BETA2.powi(iteration));
        rate * corrected_first / (corrected_second.sqrt() + 1e-8)
    }
}

/// What the joint fit is allowed to move.
#[derive(Clone, Copy)]
pub struct JointConfig {
    pub iterations: usize,
    pub albedo_rate: f32,
    pub specular_rate: f32,
    pub roughness_rate: f32,
    /// Let the reflectance move. Off holds it at whatever it started as.
    pub fit_specular: bool,
    /// Let the roughness move.
    pub fit_roughness: bool,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            iterations: 120,
            albedo_rate: 0.02,
            specular_rate: 0.01,
            roughness_rate: 0.02,
            fit_specular: true,
            fit_roughness: true,
        }
    }
}

/// Fit every element against all of its observations at once.
///
/// Albedo and reflectance enter the prediction linearly, so their gradients are
/// exact. Roughness does not — it moves the prefiltered lookup and the BRDF
/// term together — so it takes a central difference, which is one extra pair of
/// evaluations for a single parameter.
pub fn optimize_elements(
    elements: &mut [Element],
    samples: &[Sample],
    environments: &[usize],
    irradiance: &[Irradiance],
    specular: &[SpecularEnvironment],
    config: JointConfig,
) -> Vec<f64> {
    const ROUGHNESS_EPSILON: f32 = 0.02;
    let mut history = Vec::with_capacity(config.iterations);

    for element in elements.iter_mut() {
        let mut albedo_moments = [Moment::default(); 3];
        let mut specular_moments = [Moment::default(); 3];
        let mut roughness_moment = Moment::default();

        for iteration in 1..=config.iterations as i32 {
            let mut albedo_gradient = [0.0f32; 3];
            let mut specular_gradient = [0.0f32; 3];
            let mut roughness_gradient = 0.0f32;
            let mut count = 0.0f32;

            for &index in &element.samples {
                let sample = &samples[index as usize];
                let n_dot_v = sample.normal.dot(sample.view);
                for &environment in environments {
                    let shade = irradiance[environment].shade(sample.normal);
                    let predicted = predict(
                        sample,
                        element.albedo,
                        element.specular_f0,
                        element.roughness,
                        &irradiance[environment],
                        &specular[environment],
                    );
                    let observed = sample.radiance[environment];
                    let mut residual = [0.0f32; 3];
                    for channel in 0..3 {
                        residual[channel] = predicted[channel] - observed[channel];
                    }

                    for channel in 0..3 {
                        albedo_gradient[channel] += residual[channel] * shade[channel];
                    }
                    if (config.fit_specular || config.fit_roughness) && n_dot_v > 0.0 {
                        let reflection = (2.0 * n_dot_v * sample.normal - sample.view).normalize();
                        let prefiltered =
                            specular[environment].sample(reflection, element.roughness);
                        // d(F0 * scale + bias) / d(F0) is the scale alone.
                        let unit = specular_scale([1.0; 3], element.roughness, n_dot_v);
                        let zero = specular_scale([0.0; 3], element.roughness, n_dot_v);
                        for channel in 0..3 {
                            specular_gradient[channel] += residual[channel]
                                * prefiltered[channel]
                                * (unit[channel] - zero[channel]);
                        }

                        let mut difference = 0.0f32;
                        for direction in [-1.0f32, 1.0] {
                            let shifted =
                                (element.roughness + direction * ROUGHNESS_EPSILON).clamp(0.0, 1.0);
                            let nudged = predict(
                                sample,
                                element.albedo,
                                element.specular_f0,
                                shifted,
                                &irradiance[environment],
                                &specular[environment],
                            );
                            for channel in 0..3 {
                                difference += direction * residual[channel] * nudged[channel];
                            }
                        }
                        roughness_gradient += difference / (2.0 * ROUGHNESS_EPSILON);
                    }
                    count += 1.0;
                }
            }

            if count == 0.0 {
                break;
            }
            let normalise = 1.0 / count;
            for channel in 0..3 {
                element.albedo[channel] = (element.albedo[channel]
                    - albedo_moments[channel].step(
                        albedo_gradient[channel] * normalise,
                        config.albedo_rate,
                        iteration,
                    ))
                .clamp(0.0, 1.0);
            }
            if config.fit_specular {
                for channel in 0..3 {
                    element.specular_f0[channel] = (element.specular_f0[channel]
                        - specular_moments[channel].step(
                            specular_gradient[channel] * normalise,
                            config.specular_rate,
                            iteration,
                        ))
                    .clamp(0.0, 1.0);
                }
            }
            if config.fit_roughness {
                element.roughness = (element.roughness
                    - roughness_moment.step(
                        roughness_gradient * normalise,
                        config.roughness_rate,
                        iteration,
                    ))
                .clamp(0.02, 1.0);
            }
        }
    }

    // One final pass for the loss, so the number reported is the one the
    // parameters actually produce rather than the one before the last step.
    let mut total = 0.0f64;
    let mut count = 0usize;
    for element in elements.iter() {
        for &index in &element.samples {
            let sample = &samples[index as usize];
            for &environment in environments {
                let predicted = predict(
                    sample,
                    element.albedo,
                    element.specular_f0,
                    element.roughness,
                    &irradiance[environment],
                    &specular[environment],
                );
                for (value, observed) in predicted.iter().zip(sample.radiance[environment]) {
                    let difference = (value - observed) as f64;
                    total += difference * difference;
                    count += 1;
                }
            }
        }
    }
    history.push(total / count.max(1) as f64);
    history
}

/// A deterministic hash, so a perturbation sweep repeats exactly.
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn unit_float(seed: u32) -> f32 {
    hash_u32(seed) as f32 / u32::MAX as f32
}

/// Tilt a normal by a fixed angle in a direction drawn from `seed`.
///
/// A reconstruction does not get the normals this dataset hands over, so how
/// fast the bound falls away from them is what says how accurate a primitive's
/// normals actually have to be.
pub fn perturb_normal(normal: glam::Vec3, angle: f32, seed: u32) -> glam::Vec3 {
    if angle <= 0.0 {
        return normal;
    }
    let up = if normal.z.abs() < 0.9 {
        glam::Vec3::Z
    } else {
        glam::Vec3::X
    };
    let tangent = up.cross(normal).normalize();
    let bitangent = normal.cross(tangent);
    let phi = 2.0 * std::f32::consts::PI * unit_float(seed);
    let axis = tangent * phi.cos() + bitangent * phi.sin();
    (normal * angle.cos() + axis * angle.sin()).normalize()
}

/// Least-squares albedo with the specular lobe accounted for.
///
/// The lobe does not depend on the albedo, so subtracting it leaves the same
/// linear problem as [`solve_albedo`] on what is left. Whether that helps is
/// the question worth asking: a model that explains more of the image should
/// leave less of it for the albedo to absorb.
pub fn solve_albedo_specular(
    sample: &Sample,
    lights: &[(usize, Irradiance)],
    specular: &[SpecularEnvironment],
) -> [f32; 3] {
    let mut numerator = [0.0f32; 3];
    let mut denominator = [0.0f32; 3];
    for &(environment, ref irradiance) in lights {
        let shade = irradiance.shade(sample.normal);
        let observed = sample.radiance[environment];
        let lobe = specular_radiance(sample, &specular[environment]);
        for channel in 0..3 {
            let residual = observed[channel] - lobe[channel];
            numerator[channel] += shade[channel] * residual;
            denominator[channel] += shade[channel] * shade[channel];
        }
    }
    let mut albedo = [0.0f32; 3];
    for channel in 0..3 {
        albedo[channel] = if denominator[channel] > 1e-8 {
            (numerator[channel] / denominator[channel]).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    albedo
}

/// Solve a small dense system by Gaussian elimination with partial pivoting.
fn solve_dense(mut a: [[f64; 9]; 9], mut b: [f64; 9]) -> Option<[f64; 9]> {
    for column in 0..9 {
        let mut pivot = column;
        for row in column + 1..9 {
            if a[row][column].abs() > a[pivot][column].abs() {
                pivot = row;
            }
        }
        if a[pivot][column].abs() < 1e-12 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        let pivot_row = a[column];
        let pivot_rhs = b[column];
        for row in column + 1..9 {
            let factor = a[row][column] / pivot_row[column];
            for (target, source) in a[row].iter_mut().zip(pivot_row).skip(column) {
                *target -= factor * source;
            }
            b[row] -= factor * pivot_rhs;
        }
    }
    let mut x = [0.0f64; 9];
    for row in (0..9).rev() {
        let mut sum = b[row];
        for (coefficient, solved) in a[row].iter().zip(x).skip(row + 1) {
            sum -= coefficient * solved;
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Least-squares light for one environment, given the albedos.
///
/// The other half of the alternation: with the material fixed, recovering nine
/// coefficients per channel is an ordinary linear problem over every sample.
pub fn solve_light(samples: &[Sample], albedos: &[[f32; 3]], environment: usize) -> Irradiance {
    let mut coefficients = [[0.0f32; 3]; 9];
    for channel in 0..3 {
        let mut ata = [[0.0f64; 9]; 9];
        let mut atb = [0.0f64; 9];
        for (sample, albedo) in samples.iter().zip(albedos) {
            let weight = albedo[channel] as f64;
            if weight <= 0.0 {
                continue;
            }
            let basis = sh9(sample.normal);
            let observed = sample.radiance[environment][channel] as f64;
            for i in 0..9 {
                let bi = weight * basis[i] as f64;
                atb[i] += bi * observed;
                for j in 0..9 {
                    ata[i][j] += bi * weight * basis[j] as f64;
                }
            }
        }
        if let Some(solution) = solve_dense(ata, atb) {
            for i in 0..9 {
                coefficients[i][channel] = solution[i] as f32;
            }
        }
    }
    Irradiance { coefficients }
}

/// Alternate between material and light until neither moves much.
///
/// Starting the albedos at a flat grey rather than at the truth keeps this
/// honest: the light has to come out of the images. What cannot come out of
/// them is the overall scale — multiplying every albedo by `k` and dividing
/// every light by `k` reproduces the images exactly — so the caller has to
/// quotient that out before comparing against anything.
pub fn alternate(
    samples: &[Sample],
    environments: &[usize],
    iterations: usize,
) -> (Vec<[f32; 3]>, Vec<Irradiance>) {
    let mut albedos = vec![[0.5f32; 3]; samples.len()];
    let mut lights: Vec<Irradiance> = Vec::new();
    for _ in 0..iterations {
        lights = environments
            .iter()
            .map(|environment| solve_light(samples, &albedos, *environment))
            .collect();
        let paired: Vec<(usize, Irradiance)> = environments
            .iter()
            .copied()
            .zip(lights.iter().copied())
            .collect();
        for (sample, albedo) in samples.iter().zip(albedos.iter_mut()) {
            *albedo = solve_albedo(sample, &paired);
        }
    }
    (albedos, lights)
}

/// Per-channel scale that best maps `fit` onto `truth`.
///
/// The material/light product is all the images determine, so a comparison
/// that does not remove this scale is measuring the gauge rather than the
/// decomposition.
pub fn optimal_scale(fit: &[[f32; 3]], truth: &[[f32; 3]]) -> [f32; 3] {
    let mut scale = [1.0f32; 3];
    for channel in 0..3 {
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for (a, b) in fit.iter().zip(truth) {
            numerator += a[channel] as f64 * b[channel] as f64;
            denominator += a[channel] as f64 * a[channel] as f64;
        }
        if denominator > 1e-9 {
            scale[channel] = (numerator / denominator) as f32;
        }
    }
    scale
}

/// Peak signal-to-noise ratio over a set of linear-radiance residuals.
///
/// Reported against a peak of 1.0, the value a display-referred white would
/// have, so it is comparable with the PSNR the rest of the pipeline quotes.
pub struct Accumulator {
    sum_squared: f64,
    count: usize,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Self {
            sum_squared: 0.0,
            count: 0,
        }
    }

    pub fn add(&mut self, predicted: [f32; 3], observed: [f32; 3]) {
        for channel in 0..3 {
            let difference = (predicted[channel] - observed[channel]) as f64;
            self.sum_squared += difference * difference;
            self.count += 1;
        }
    }

    pub fn psnr(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let mse = self.sum_squared / self.count as f64;
        if mse <= 0.0 {
            f64::INFINITY
        } else {
            10.0 * (1.0 / mse).log10()
        }
    }

    pub fn rmse(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        (self.sum_squared / self.count as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_environment_shades_every_normal_alike() {
        let texels = vec![[0.5f32, 0.5, 0.5]; 64 * 32];
        let irradiance = Irradiance::project(&texels, 64, 32);
        let up = irradiance.shade(glam::Vec3::Y);
        let side = irradiance.shade(glam::Vec3::X);
        for channel in 0..3 {
            assert!(
                (up[channel] - side[channel]).abs() < 1e-3,
                "uniform light has to be isotropic, got {up:?} and {side:?}"
            );
        }
        // A Lambertian surface under uniform radiance L reflects exactly
        // `albedo * L`, so a unit albedo gives the environment back.
        assert!(
            (up[0] - 0.5).abs() < 5e-3,
            "expected the environment radiance back, got {}",
            up[0]
        );
    }

    #[test]
    fn albedo_comes_back_when_the_light_is_known() {
        let texels = vec![[0.4f32, 0.6, 0.8]; 64 * 32];
        let irradiance = Irradiance::project(&texels, 64, 32);
        let normal = glam::Vec3::new(0.3, 0.8, 0.5).normalize();
        let truth = [0.7f32, 0.35, 0.15];
        let shade = irradiance.shade(normal);
        let sample = Sample {
            view_index: 0,
            pixel: 0,
            normal,
            albedo_truth: truth,
            specular_f0: [0.0; 3],
            view: normal,
            position: glam::Vec3::ZERO,
            roughness: 1.0,
            radiance: vec![[
                truth[0] * shade[0],
                truth[1] * shade[1],
                truth[2] * shade[2],
            ]],
        };
        let solved = solve_albedo(&sample, &[(0, irradiance)]);
        for channel in 0..3 {
            assert!(
                (solved[channel] - truth[channel]).abs() < 1e-4,
                "expected {truth:?}, got {solved:?}"
            );
        }
    }

    #[test]
    fn the_basis_is_orthonormal_enough_to_round_trip_a_direction() {
        // Y_1m are proportional to the direction itself, so recovering it from
        // the basis is a cheap check that the ordering has not drifted.
        let dir = glam::Vec3::new(-0.4, 0.5, 0.75).normalize();
        let basis = sh9(dir);
        let recovered = glam::Vec3::new(basis[3], basis[1], basis[2]) / 0.488_603;
        assert!((recovered - dir).length() < 1e-5, "got {recovered:?}");
    }

    #[test]
    fn equirect_mapping_agrees_with_the_renderer_at_the_poles() {
        assert!(equirect_direction(0.5, 0.0).y > 0.99, "v=0 has to look up");
        assert!(
            equirect_direction(0.5, 1.0).y < -0.99,
            "v=1 has to look down"
        );
        let forward = equirect_direction(0.5, 0.5);
        assert!(
            forward.z > 0.99,
            "the middle looks along +Z, got {forward:?}"
        );
    }
}
