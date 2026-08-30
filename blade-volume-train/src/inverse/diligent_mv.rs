//! Loader for the calibrated DiLiGenT-MV photometric-stereo capture.
//!
//! The release stores linear 16-bit RGB, masks, OpenCV camera extrinsics, and
//! 96 calibrated distant lights at each of 20 poses. Images are divided by the
//! published per-channel light intensities. Distant emitters reuse the existing
//! view-specific point-light path at a distance where falloff is negligible;
//! this avoids another renderer or training-graph variant.

use blade_volume as vol;
use flate2::read;
use std::{collections, fs, io, path};

pub const VIEW_COUNT: usize = 20;
pub const LIGHT_COUNT: usize = 96;
pub const TRAIN_VIEW_INDICES: [usize; 16] =
    [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19];
pub const HELD_VIEW_INDICES: [usize; 4] = [0, 5, 10, 15];
pub const TRAIN_LIGHT_INDICES: [usize; 24] = [
    3, 6, 9, 15, 18, 21, 27, 30, 33, 39, 42, 45, 51, 54, 57, 63, 66, 69, 75, 78, 81, 87, 90, 93,
];
pub const HELD_LIGHT_INDICES: [usize; 8] = [0, 12, 24, 36, 48, 60, 72, 84];

const SOURCE_WIDTH: usize = 612;
const SOURCE_HEIGHT: usize = 512;
const LIGHT_DISTANCE: f32 = 1.0e6;

/// Aligned image captures and their view-specific calibrated lights.
pub struct Dataset {
    pub captures: Vec<super::capture::Capture>,
    pub lights: Vec<Vec<vol::relight::PointLight>>,
    pub source_light_indices: Vec<usize>,
}

/// Load selected DiLiGenT-MV lights at a bounded working resolution.
pub fn load(object: &path::Path, width: usize, light_indices: &[usize]) -> Result<Dataset, String> {
    if width == 0 {
        return Err("DiLiGenT-MV working width must be non-zero".to_string());
    }
    if light_indices.is_empty() {
        return Err("DiLiGenT-MV needs at least one selected light".to_string());
    }
    if let Some(&index) = light_indices.iter().find(|&&index| index >= LIGHT_COUNT) {
        return Err(format!(
            "DiLiGenT-MV light {index} is outside 0..{LIGHT_COUNT}"
        ));
    }
    if light_indices
        .iter()
        .enumerate()
        .any(|(offset, index)| light_indices[..offset].contains(index))
    {
        return Err("DiLiGenT-MV selected lights must be unique".to_string());
    }

    let calibration = MatFile::load(&object.join("Calib_Results.mat"))?;
    let intrinsics = calibration.matrix("KK", 3, 3)?;
    let focal = [intrinsics[0] as f32, intrinsics[4] as f32];
    let principal = [intrinsics[6] as f32, intrinsics[7] as f32];
    if focal
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || !principal.iter().all(|value| value.is_finite())
    {
        return Err("DiLiGenT-MV has invalid camera intrinsics".to_string());
    }
    let height = (SOURCE_HEIGHT * width + SOURCE_WIDTH / 2) / SOURCE_WIDTH;
    let mut captures = light_indices
        .iter()
        .map(|_| super::capture::Capture {
            width,
            height,
            views: Vec::with_capacity(VIEW_COUNT),
        })
        .collect::<Vec<_>>();
    let mut lights = light_indices
        .iter()
        .map(|_| Vec::with_capacity(VIEW_COUNT))
        .collect::<Vec<_>>();

    for view_index in 0..VIEW_COUNT {
        let camera = load_camera(&calibration, view_index, focal, principal)?;
        let directory = object.join(format!("view_{:02}", view_index + 1));
        let mask = load_mask(&directory.join("mask.png"), width, height)?;
        let directions = load_vectors(&directory.join("light_directions.txt"), "directions")?;
        let intensities = load_vectors(&directory.join("light_intensities.txt"), "intensities")?;
        let orientation = glam::Quat::from_array(camera.cam_orientation);
        for (output, &light_index) in light_indices.iter().enumerate() {
            let intensity = intensities[light_index];
            if intensity.iter().any(|value| *value <= 0.0) {
                return Err(format!(
                    "{} light {} has non-positive intensity",
                    directory.display(),
                    light_index + 1,
                ));
            }
            let pixels = load_linear_image(
                &directory.join(format!("{:03}.png", light_index + 1)),
                width,
                height,
                intensity,
                &mask,
            )?;
            captures[output].views.push(super::capture::View {
                name: format!("view_{:02}-light_{:03}", view_index + 1, light_index + 1),
                camera,
                pixels,
                mask: Some(mask.clone()),
            });
            let direction = photometric_direction_world(orientation, directions[light_index]);
            if direction.length_squared() == 0.0 {
                return Err(format!(
                    "{} light {} has no direction",
                    directory.display(),
                    light_index + 1,
                ));
            }
            lights[output].push(distant_point_light(direction));
        }
    }

    Ok(Dataset {
        captures,
        lights,
        source_light_indices: light_indices.to_vec(),
    })
}

fn load_camera(
    calibration: &MatFile,
    index: usize,
    focal: [f32; 2],
    principal: [f32; 2],
) -> Result<vol::CameraParams, String> {
    let rotation = calibration.matrix(&format!("Rc_{}", index + 1), 3, 3)?;
    let translation = calibration.matrix(&format!("Tc_{}", index + 1), 3, 1)?;
    let camera_from_world =
        glam::DMat3::from_cols_array(rotation.try_into().expect("matrix shape was checked"));
    let identity_error = (camera_from_world.transpose() * camera_from_world
        - glam::DMat3::IDENTITY)
        .to_cols_array()
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let determinant = camera_from_world.determinant();
    if !camera_from_world.is_finite()
        || identity_error > 1.0e-3
        || (determinant - 1.0).abs() > 1.0e-3
    {
        return Err(format!(
            "DiLiGenT-MV camera {} has a non-rigid rotation (error {identity_error}, determinant {determinant})",
            index + 1,
        ));
    }
    let world_from_camera = camera_from_world.transpose();
    let translation = glam::DVec3::from_slice(translation);
    let position = world_from_camera * -translation;
    let orientation = glam::DQuat::from_mat3(&world_from_camera).normalize();
    Ok(vol::CameraParams {
        cam_position: position.as_vec3().to_array(),
        depth: 4_000.0,
        cam_orientation: orientation.as_quat().to_array(),
        fov: [
            2.0 * (0.5 * SOURCE_WIDTH as f32 / focal[0]).atan(),
            2.0 * (0.5 * SOURCE_HEIGHT as f32 / focal[1]).atan(),
        ],
        principal: [
            2.0 * principal[0] / SOURCE_WIDTH as f32 - 1.0,
            2.0 * principal[1] / SOURCE_HEIGHT as f32 - 1.0,
        ],
    })
}

fn distant_point_light(direction: glam::Vec3) -> vol::relight::PointLight {
    vol::relight::PointLight {
        position: (LIGHT_DISTANCE * direction).to_array(),
        direction: [0.0, 0.0, 1.0],
        intensity: [LIGHT_DISTANCE * LIGHT_DISTANCE; 3],
        exponent: 0.0,
    }
}

fn photometric_direction_world(orientation: glam::Quat, direction: [f32; 3]) -> glam::Vec3 {
    // The release's photometric coordinate has +Z pointing out of the image,
    // while CameraParams local +Z points into the scene.
    (orientation * -glam::Vec3::from(direction)).normalize_or_zero()
}

fn load_vectors(file: &path::Path, label: &str) -> Result<Vec<[f32; 3]>, String> {
    let text = fs::read_to_string(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let values = text
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f32>()
                .map_err(|error| format!("cannot parse {value:?} in {}: {error}", file.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let vectors = values.as_chunks::<3>();
    if !vectors.1.is_empty()
        || vectors.0.len() != LIGHT_COUNT
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "{} has {} rather than {LIGHT_COUNT} finite light {label}",
            file.display(),
            vectors.0.len(),
        ));
    }
    Ok(vectors.0.to_vec())
}

fn load_mask(file: &path::Path, width: usize, height: usize) -> Result<Vec<f32>, String> {
    let source = image::open(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?
        .into_luma8();
    if source.dimensions() != (SOURCE_WIDTH as u32, SOURCE_HEIGHT as u32) {
        return Err(format!("{} has unexpected dimensions", file.display()));
    }
    let resized = image::imageops::resize(
        &source,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    Ok(resized
        .pixels()
        .map(|pixel| pixel[0] as f32 / u8::MAX as f32)
        .collect())
}

fn load_linear_image(
    file: &path::Path,
    width: usize,
    height: usize,
    intensity: [f32; 3],
    mask: &[f32],
) -> Result<Vec<[f32; 3]>, String> {
    let source = image::open(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?
        .into_rgb16();
    if source.dimensions() != (SOURCE_WIDTH as u32, SOURCE_HEIGHT as u32) {
        return Err(format!("{} has unexpected dimensions", file.display()));
    }
    let resized = image::imageops::resize(
        &source,
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    Ok(resized
        .pixels()
        .zip(mask)
        .map(|(pixel, &coverage)| {
            std::array::from_fn(|channel| {
                coverage * pixel[channel] as f32 / u16::MAX as f32 / intensity[channel]
            })
        })
        .collect())
}

struct MatFile {
    matrices: collections::HashMap<String, Matrix>,
}

struct Matrix {
    dimensions: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Element<'a> {
    kind: u32,
    data: &'a [u8],
    next: usize,
}

impl MatFile {
    fn load(file: &path::Path) -> Result<Self, String> {
        let bytes =
            fs::read(file).map_err(|error| format!("cannot read {}: {error}", file.display()))?;
        if bytes.len() < 128 || &bytes[126..128] != b"IM" {
            return Err(format!(
                "{} is not a little-endian MATLAB v5 file",
                file.display()
            ));
        }
        let mut matrices = collections::HashMap::new();
        parse_elements(&bytes, 128, &mut matrices)?;
        Ok(Self { matrices })
    }

    fn matrix(&self, name: &str, rows: usize, columns: usize) -> Result<&[f64], String> {
        let matrix = self
            .matrices
            .get(name)
            .ok_or_else(|| format!("DiLiGenT-MV calibration has no {name}"))?;
        if matrix.dimensions != [rows, columns] || matrix.values.len() != rows * columns {
            return Err(format!(
                "DiLiGenT-MV calibration {name} is {:?}, expected [{rows}, {columns}]",
                matrix.dimensions,
            ));
        }
        Ok(&matrix.values)
    }
}

fn parse_elements(
    bytes: &[u8],
    mut offset: usize,
    matrices: &mut collections::HashMap<String, Matrix>,
) -> Result<(), String> {
    while offset + 8 <= bytes.len() {
        let entry = element(bytes, offset)?;
        match entry.kind {
            14 => {
                let (name, matrix) = parse_matrix(entry.data)?;
                if matrices.insert(name.clone(), matrix).is_some() {
                    return Err(format!("duplicate MATLAB matrix {name}"));
                }
            }
            15 => {
                let mut decoded = Vec::new();
                io::Read::read_to_end(&mut read::ZlibDecoder::new(entry.data), &mut decoded)
                    .map_err(|error| format!("cannot decompress MATLAB matrix: {error}"))?;
                parse_elements(&decoded, 0, matrices)?;
            }
            kind => return Err(format!("unsupported MATLAB element type {kind}")),
        }
        offset = entry.next;
    }
    Ok(())
}

fn parse_matrix(bytes: &[u8]) -> Result<(String, Matrix), String> {
    let flags = element(bytes, 0)?;
    let dimensions = element(bytes, flags.next)?;
    let name = element(bytes, dimensions.next)?;
    let values = element(bytes, name.next)?;
    if flags.kind != 6 || dimensions.kind != 5 || name.kind != 1 || values.kind != 9 {
        return Err("unsupported MATLAB numeric matrix layout".to_string());
    }
    let name = std::str::from_utf8(name.data)
        .map_err(|error| format!("invalid MATLAB matrix name: {error}"))?
        .to_string();
    let (dimension_values, dimension_tail) = dimensions.data.as_chunks::<4>();
    let (real_values, real_tail) = values.data.as_chunks::<8>();
    if !dimension_tail.is_empty() || !real_tail.is_empty() {
        return Err("misaligned MATLAB numeric matrix".to_string());
    }
    let dimensions = dimension_values
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .map(|value| {
            usize::try_from(value).map_err(|_| "negative MATLAB matrix dimension".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let values = real_values
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if values.len() != dimensions.iter().product::<usize>()
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!("MATLAB matrix {name} has invalid values"));
    }
    Ok((name, Matrix { dimensions, values }))
}

fn element(bytes: &[u8], offset: usize) -> Result<Element<'_>, String> {
    if offset + 8 > bytes.len() {
        return Err("truncated MATLAB element".to_string());
    }
    let tag = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let small_size = tag >> 16;
    if small_size != 0 {
        let size = small_size as usize;
        if size > 4 {
            return Err(format!("invalid small MATLAB element size {size}"));
        }
        return Ok(Element {
            kind: tag & 0xffff,
            data: &bytes[offset + 4..offset + 4 + size],
            next: offset + 8,
        });
    }
    let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
    let end = offset + 8 + size;
    if end > bytes.len() {
        return Err("MATLAB element payload is truncated".to_string());
    }
    Ok(Element {
        kind: tag,
        data: &bytes[offset + 8..end],
        next: if tag == 15 { end } else { (end + 7) & !7 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_element(kind: u32, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + data.len() + 7);
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes.resize((bytes.len() + 7) & !7, 0);
        bytes
    }

    fn compressed_matrix(name: &[u8], values: &[f64]) -> Vec<u8> {
        let mut payload = regular_element(6, &[6, 0, 0, 0, 0, 0, 0, 0]);
        let mut dimensions = Vec::new();
        dimensions.extend_from_slice(&(values.len() as i32).to_le_bytes());
        dimensions.extend_from_slice(&1_i32.to_le_bytes());
        payload.extend(regular_element(5, &dimensions));
        let mut small_name = Vec::new();
        small_name.extend_from_slice(&((name.len() as u32) << 16 | 1).to_le_bytes());
        small_name.extend_from_slice(name);
        small_name.resize(8, 0);
        payload.extend(small_name);
        let real = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        payload.extend(regular_element(9, &real));
        let matrix = regular_element(14, &payload);
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        io::Write::write_all(&mut encoder, &matrix).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut element = Vec::new();
        element.extend_from_slice(&15_u32.to_le_bytes());
        element.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        element.extend(compressed);
        element
    }

    #[test]
    fn split_is_disjoint_and_samples_the_light_range() {
        assert_eq!(
            TRAIN_VIEW_INDICES.len() + HELD_VIEW_INDICES.len(),
            VIEW_COUNT
        );
        assert!(TRAIN_VIEW_INDICES
            .iter()
            .all(|index| !HELD_VIEW_INDICES.contains(index)));
        assert_eq!(TRAIN_LIGHT_INDICES.len(), 24);
        assert_eq!(HELD_LIGHT_INDICES.len(), 8);
        assert!(TRAIN_LIGHT_INDICES
            .iter()
            .all(|index| !HELD_LIGHT_INDICES.contains(index)));
        let mut selected = TRAIN_LIGHT_INDICES.to_vec();
        selected.extend(HELD_LIGHT_INDICES);
        selected.sort_unstable();
        assert_eq!(selected, (0..LIGHT_COUNT).step_by(3).collect::<Vec<_>>());
    }

    #[test]
    fn far_point_light_matches_a_directional_light_near_the_origin() {
        let direction = glam::Vec3::new(0.3, -0.2, 0.93).normalize();
        let point = glam::Vec3::new(100.0, -50.0, 25.0);
        let sample = distant_point_light(direction).sample(point).unwrap();
        assert!(sample.towards.dot(direction) > 0.999_999);
        for value in sample.radiance {
            assert!((value - 1.0).abs() < 5.0e-4);
        }
    }

    #[test]
    fn photometric_positive_z_points_back_towards_the_camera() {
        let direction = photometric_direction_world(glam::Quat::IDENTITY, [0.0, 0.0, 1.0]);
        assert_eq!(direction, glam::Vec3::NEG_Z);
    }

    #[test]
    fn reads_unpadded_compressed_matlab_matrix() {
        let bytes = compressed_matrix(b"KK", &[1.0, 2.0, 3.0]);
        let mut matrices = collections::HashMap::new();
        parse_elements(&bytes, 0, &mut matrices).unwrap();
        let matrix = matrices.get("KK").unwrap();
        assert_eq!(matrix.dimensions, [3, 1]);
        assert_eq!(matrix.values, [1.0, 2.0, 3.0]);
    }
}
