//! Loader for the calibrated LUCES-MV near-field capture.
//!
//! The release stores linear 16-bit RGB, masks, world-to-camera extrinsics,
//! and one camera-local calibration for each LED. Images are divided by the
//! published per-channel brightness while the light intensity is normalized at
//! the published mean depth. That is the convention used by the reference
//! photometric-stereo code and leaves material albedo on a useful scale.

use blade_volume as vol;
use std::{fs, path};

const VIEW_IDS: [usize; 12] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66];
const LIGHT_COUNT: usize = 15;

#[derive(Clone, Copy)]
struct LocalLight {
    light: vol::relight::PointLight,
    brightness: [f32; 3],
}

struct Calibration {
    width: usize,
    height: usize,
    focal: f32,
    principal: [f32; 2],
    lights: [LocalLight; LIGHT_COUNT],
}

/// Aligned image captures and their view-specific finite lights.
///
/// `captures[index]` and `lights[index]` describe one selected LED over all
/// twelve camera poses. The inner light index is the matching capture view.
pub struct Dataset {
    pub captures: Vec<super::capture::Capture>,
    pub lights: Vec<Vec<vol::relight::PointLight>>,
    pub source_light_indices: Vec<usize>,
}

/// Load selected LUCES-MV LEDs at a bounded working resolution.
pub fn load(
    object: &path::Path,
    camera_one: &path::Path,
    camera_two: &path::Path,
    width: usize,
    light_indices: &[usize],
) -> Result<Dataset, String> {
    if width == 0 {
        return Err("LUCES-MV working width must be non-zero".to_string());
    }
    if light_indices.is_empty() {
        return Err("LUCES-MV needs at least one selected light".to_string());
    }
    if let Some(&index) = light_indices.iter().find(|&&index| index >= LIGHT_COUNT) {
        return Err(format!(
            "LUCES-MV light {index} is outside 0..{LIGHT_COUNT}"
        ));
    }
    if light_indices
        .iter()
        .enumerate()
        .any(|(offset, index)| light_indices[..offset].contains(index))
    {
        return Err("LUCES-MV selected lights must be unique".to_string());
    }

    let calibrations = [
        parse_calibration(camera_one)?,
        parse_calibration(camera_two)?,
    ];
    validate_cameras(&calibrations)?;
    let source_width = calibrations[0].width;
    let source_height = calibrations[0].height;
    let height = (source_height * width + source_width / 2) / source_width;
    let mut captures: Vec<_> = light_indices
        .iter()
        .map(|_| super::capture::Capture {
            width,
            height,
            views: Vec::with_capacity(VIEW_IDS.len()),
        })
        .collect();
    let mut lights: Vec<_> = light_indices
        .iter()
        .map(|_| Vec::with_capacity(VIEW_IDS.len()))
        .collect();

    for &view_id in &VIEW_IDS {
        let view_path = object.join(format!("view_{view_id:03}"));
        // The single released overlap view 036 uses camera two; camera one
        // supplies 000 through 030 and camera two supplies 036 through 066.
        let calibration = &calibrations[usize::from(view_id >= 36)];
        let camera = load_camera(
            &view_path.join("RT.npz"),
            calibration,
            source_width,
            source_height,
        )?;
        let mask = load_mask(
            &view_path.join("mask.png"),
            source_width,
            source_height,
            width,
            height,
        )?;
        let orientation = glam::Quat::from_array(camera.cam_orientation);
        let position = glam::Vec3::from(camera.cam_position);
        for (output, &light_index) in light_indices.iter().enumerate() {
            let local = calibration.lights[light_index];
            let pixels = load_linear_image(
                &view_path.join(format!("{:02}.png", light_index + 1)),
                source_width,
                source_height,
                width,
                height,
                local.brightness,
                &mask,
            )?;
            captures[output].views.push(super::capture::View {
                name: format!("view_{view_id:03}-light_{:02}", light_index + 1),
                camera,
                pixels,
                mask: Some(mask.clone()),
            });
            lights[output].push(local.light.to_world(orientation, position));
        }
    }

    Ok(Dataset {
        captures,
        lights,
        source_light_indices: light_indices.to_vec(),
    })
}

fn parse_calibration(file: &path::Path) -> Result<Calibration, String> {
    let text = fs::read_to_string(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let mut lines = text.lines();
    lines
        .next()
        .ok_or_else(|| format!("{} has no calibration header", file.display()))?;
    let header = parse_floats(
        lines
            .next()
            .ok_or_else(|| format!("{} has no camera calibration", file.display()))?,
    )?;
    if header.len() != 7 || header[0] as usize != LIGHT_COUNT {
        return Err(format!(
            "{} has an unexpected camera calibration: {header:?}",
            file.display()
        ));
    }
    let height = checked_integer(header[1], "image height")?;
    let width = checked_integer(header[2], "image width")?;
    let mean_distance = header[6];
    if !header[3..].iter().all(|value| value.is_finite())
        || header[3] <= 0.0
        || mean_distance <= 0.0
    {
        return Err(format!("{} has invalid camera values", file.display()));
    }
    let empty = LocalLight {
        light: vol::relight::PointLight {
            position: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            intensity: [1.0; 3],
            exponent: 0.0,
        },
        brightness: [1.0; 3],
    };
    let mut lights = [empty; LIGHT_COUNT];
    let mut count = 0;
    for (index, line) in lines.filter(|line| !line.trim().is_empty()).enumerate() {
        if index >= LIGHT_COUNT {
            return Err(format!("{} has too many lights", file.display()));
        }
        let values = parse_floats(line)?;
        if values.len() != 11 {
            return Err(format!(
                "{} light {index} has {} rather than 11 values",
                file.display(),
                values.len()
            ));
        }
        let position: [f32; 3] = values[0..3].try_into().unwrap();
        let direction: [f32; 3] = values[3..6].try_into().unwrap();
        let brightness: [f32; 3] = values[6..9].try_into().unwrap();
        if !values.iter().all(|value| value.is_finite())
            || glam::Vec3::from(direction).length_squared() <= f32::EPSILON
            || brightness.iter().any(|value| *value <= 0.0)
            || values[9] < 0.0
        {
            return Err(format!("{} light {index} is invalid", file.display()));
        }
        lights[index] = LocalLight {
            light: vol::relight::PointLight {
                position,
                direction,
                intensity: [mean_distance * mean_distance; 3],
                exponent: values[9],
            },
            brightness,
        };
        count += 1;
    }
    if count != LIGHT_COUNT {
        return Err(format!(
            "{} has {count} rather than {LIGHT_COUNT} lights",
            file.display()
        ));
    }
    Ok(Calibration {
        width,
        height,
        focal: header[3],
        principal: [header[4], header[5]],
        lights,
    })
}

fn parse_floats(line: &str) -> Result<Vec<f32>, String> {
    line.split_whitespace()
        .map(|value| {
            value
                .parse::<f32>()
                .map_err(|error| format!("cannot parse {value:?}: {error}"))
        })
        .collect()
}

fn checked_integer(value: f32, label: &str) -> Result<usize, String> {
    if value.is_finite() && value > 0.0 && value.fract() == 0.0 {
        Ok(value as usize)
    } else {
        Err(format!("invalid LUCES-MV {label}: {value}"))
    }
}

fn validate_cameras(calibrations: &[Calibration; 2]) -> Result<(), String> {
    let first = &calibrations[0];
    let second = &calibrations[1];
    if first.width != second.width
        || first.height != second.height
        || first.focal != second.focal
        || first.principal != second.principal
    {
        return Err("LUCES-MV camera calibrations have different wrapped intrinsics".to_string());
    }
    Ok(())
}

fn load_camera(
    file: &path::Path,
    calibration: &Calibration,
    width: usize,
    height: usize,
) -> Result<vol::CameraParams, String> {
    let bytes =
        fs::read(file).map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let rotation = npy_f32(stored_npz_member(&bytes, "R.npy")?, 9)?;
    let translation = npy_f32(stored_npz_member(&bytes, "T.npy")?, 3)?;
    let world_from_camera = glam::Mat3::from_cols_array(&rotation.try_into().unwrap());
    let determinant = world_from_camera.determinant();
    let identity_error = (world_from_camera.transpose() * world_from_camera - glam::Mat3::IDENTITY)
        .to_cols_array()
        .into_iter()
        .map(f32::abs)
        .fold(0.0, f32::max);
    if !world_from_camera.is_finite()
        || identity_error > 1.0e-3
        || (determinant - 1.0).abs() > 1.0e-3
    {
        return Err(format!(
            "{} has a non-rigid rotation (error {identity_error}, determinant {determinant})",
            file.display()
        ));
    }
    let translation = glam::Vec3::from_slice(&translation);
    let position = world_from_camera * -translation;
    let orientation = glam::Quat::from_mat3(&world_from_camera).normalize();
    Ok(vol::CameraParams {
        cam_position: position.to_array(),
        depth: 2_000.0,
        cam_orientation: orientation.to_array(),
        fov: [
            2.0 * (0.5 * width as f32 / calibration.focal).atan(),
            2.0 * (0.5 * height as f32 / calibration.focal).atan(),
        ],
        principal: [
            2.0 * calibration.principal[0] / width as f32 - 1.0,
            2.0 * calibration.principal[1] / height as f32 - 1.0,
        ],
    })
}

fn stored_npz_member<'a>(bytes: &'a [u8], expected: &str) -> Result<&'a [u8], String> {
    let mut offset = 0;
    while offset + 30 <= bytes.len() {
        if bytes[offset..offset + 4] != [0x50, 0x4b, 0x03, 0x04] {
            break;
        }
        let method = u16::from_le_bytes(bytes[offset + 8..offset + 10].try_into().unwrap());
        let size = u32::from_le_bytes(bytes[offset + 18..offset + 22].try_into().unwrap()) as usize;
        let name_length =
            u16::from_le_bytes(bytes[offset + 26..offset + 28].try_into().unwrap()) as usize;
        let extra_length =
            u16::from_le_bytes(bytes[offset + 28..offset + 30].try_into().unwrap()) as usize;
        let name_start = offset + 30;
        let data_start = name_start + name_length + extra_length;
        let data_end = data_start.saturating_add(size);
        if data_end > bytes.len() {
            return Err("truncated NPZ member".to_string());
        }
        let name = std::str::from_utf8(&bytes[name_start..name_start + name_length])
            .map_err(|error| format!("invalid NPZ member name: {error}"))?;
        if name == expected {
            if method != 0 {
                return Err(format!("NPZ member {expected} is compressed"));
            }
            return Ok(&bytes[data_start..data_end]);
        }
        offset = data_end;
    }
    Err(format!("NPZ has no {expected} member"))
}

fn npy_f32(bytes: &[u8], count: usize) -> Result<Vec<f32>, String> {
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\0" {
        return Err("expected a NumPy v1 array".to_string());
    }
    let header_length = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
    let data_start = 10 + header_length;
    let data_end = data_start.saturating_add(4 * count);
    if data_end != bytes.len() {
        return Err(format!(
            "NumPy array has {} data bytes, expected {}",
            bytes.len().saturating_sub(data_start),
            4 * count
        ));
    }
    let header = std::str::from_utf8(&bytes[10..data_start])
        .map_err(|error| format!("invalid NumPy header: {error}"))?;
    if !header.contains("'descr': '<f4'") || !header.contains("'fortran_order': False") {
        return Err(format!("unsupported NumPy array: {header}"));
    }
    Ok(bytes[data_start..data_end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect())
}

fn load_mask(
    file: &path::Path,
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
) -> Result<Vec<f32>, String> {
    let source = image::open(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?
        .into_luma8();
    if source.dimensions() != (source_width as u32, source_height as u32) {
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

#[allow(clippy::too_many_arguments)]
fn load_linear_image(
    file: &path::Path,
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    brightness: [f32; 3],
    mask: &[f32],
) -> Result<Vec<[f32; 3]>, String> {
    let source = image::open(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?
        .into_rgb16();
    if source.dimensions() != (source_width as u32, source_height as u32) {
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
                coverage * pixel[channel] as f32 / u16::MAX as f32 / brightness[channel]
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(values: &[f32]) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            values.len()
        );
        while !(10 + header.len() + 1).is_multiple_of(16) {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\0".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        bytes
    }

    fn npz_member(name: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0x50, 0x4b, 0x03, 0x04, 20, 0, 0, 0, 0, 0];
        bytes.extend([0; 8]);
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend((name.len() as u16).to_le_bytes());
        bytes.extend(0_u16.to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend(data);
        bytes
    }

    #[test]
    fn reads_stored_numpy_members_without_a_zip_dependency() {
        let r = npy(&[1.0, 2.0, 3.0]);
        let t = npy(&[4.0, 5.0]);
        let mut archive = npz_member("R.npy", &r);
        archive.extend(npz_member("T.npy", &t));

        assert_eq!(
            npy_f32(stored_npz_member(&archive, "R.npy").unwrap(), 3).unwrap(),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            npy_f32(stored_npz_member(&archive, "T.npy").unwrap(), 2).unwrap(),
            [4.0, 5.0]
        );
    }
}
