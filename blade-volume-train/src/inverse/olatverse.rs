//! Loader for the calibrated OLATverse object captures.
//!
//! The release contains 35 cameras by 331 individually controlled lights.
//! Its published benchmark split uses 24 construction and six held cameras,
//! excludes the five polarized cameras, and assigns disjoint thirds of the
//! first 310 lights to fitting and evaluation. Images are display-encoded AVIF
//! with a documented two-times brightness scale; this loader decodes sRGB and
//! removes that scale before fitting physical materials.

use blade_volume as vol;
use std::{collections, fs, path, thread};

pub const SOURCE_WIDTH: usize = 1500;
pub const SOURCE_HEIGHT: usize = 2844;
pub const LIGHT_COUNT: usize = 331;
pub const FULL_BRIGHT_FRAME: usize = 14;

pub const TRAIN_VIEW_INDICES: [usize; 24] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];
pub const HELD_VIEW_INDICES: [usize; 6] = [24, 25, 26, 27, 28, 29];

/// Published camera order: construction cameras first, then held cameras.
pub const VIEW_NAMES: [&str; 30] = [
    "Cam02", "Cam03", "Cam04", "Cam05", "Cam08", "Cam09", "Cam11", "Cam12", "Cam14", "Cam15",
    "Cam16", "Cam18", "Cam20", "Cam23", "Cam24", "Cam26", "Cam29", "Cam30", "Cam31", "Cam32",
    "Cam36", "Cam37", "Cam38", "Cam40", "Cam01", "Cam06", "Cam13", "Cam19", "Cam27", "Cam35",
];

#[derive(Clone)]
struct Camera {
    name: String,
    params: vol::CameraParams,
}

#[derive(Clone, PartialEq)]
struct SourceLight {
    frame: usize,
    light: vol::relight::PointLight,
}

/// The 104 construction lights from the official benchmark protocol.
pub fn train_light_indices() -> Vec<usize> {
    (0..310).step_by(3).collect()
}

/// The 103 untouched lights from the official benchmark protocol.
pub fn held_light_indices() -> Vec<usize> {
    (1..310).step_by(3).collect()
}

/// Load full-bright frame 14 for point-cloud geometry initialization.
pub fn load_full_bright(
    object: &path::Path,
    width: usize,
) -> Result<super::capture::Capture, String> {
    let cameras = load_cameras(object, width)?;
    let mut capture = empty_capture(width, cameras.len());
    for camera in cameras {
        let mask = load_mask(
            &object.join("mask").join(format!("{}.png", camera.name)),
            width,
        )?;
        let files = image_frames(&object.join("masked_olat").join(&camera.name))?;
        let file = files.get(&FULL_BRIGHT_FRAME).ok_or_else(|| {
            format!(
                "{} has no full-bright frame {FULL_BRIGHT_FRAME:06}",
                object.join("masked_olat").join(&camera.name).display(),
            )
        })?;
        let pixels = load_image(file, width, capture.height, &mask)?;
        capture.views.push(super::capture::View {
            name: format!("{}-full-bright", camera.name),
            camera: camera.params,
            pixels,
            mask: Some(mask),
        });
    }
    Ok(capture)
}

/// Load selected OLATs and their finite world-space lights.
pub fn load(
    object: &path::Path,
    lights_file: &path::Path,
    width: usize,
    light_indices: &[usize],
) -> Result<crate::calibrated::Dataset, String> {
    if light_indices.is_empty() {
        return Err("OLATverse needs at least one selected light".to_string());
    }
    if let Some(&index) = light_indices.iter().find(|&&index| index >= LIGHT_COUNT) {
        return Err(format!(
            "OLATverse light {index} is outside 0..{LIGHT_COUNT}"
        ));
    }
    if light_indices
        .iter()
        .enumerate()
        .any(|(offset, index)| light_indices[..offset].contains(index))
    {
        return Err("OLATverse selected lights must be unique".to_string());
    }

    let cameras = load_cameras(object, width)?;
    let source_lights = load_lights(lights_file)?;
    let height = output_height(width)?;
    let mut captures = light_indices
        .iter()
        .map(|_| super::capture::Capture {
            width,
            height,
            views: Vec::with_capacity(cameras.len()),
        })
        .collect::<Vec<_>>();

    let workers = thread::available_parallelism()
        .map_or(1, usize::from)
        .min(cameras.len());
    let mut decoded = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let cameras = &cameras;
            let source_lights = &source_lights;
            handles.push(scope.spawn(move || {
                let mut decoded = Vec::new();
                for camera_index in (worker..cameras.len()).step_by(workers) {
                    let camera = &cameras[camera_index];
                    let mask = load_mask(
                        &object.join("mask").join(format!("{}.png", camera.name)),
                        width,
                    )?;
                    let directory = object.join("masked_olat").join(&camera.name);
                    let files = image_frames(&directory)?;
                    let mut views = Vec::with_capacity(light_indices.len());
                    for &light_index in light_indices {
                        let source = &source_lights[light_index];
                        let file = files.get(&source.frame).ok_or_else(|| {
                            format!(
                                "{} has no OLAT frame {:06} for light {light_index}",
                                directory.display(),
                                source.frame,
                            )
                        })?;
                        let pixels = load_image(file, width, height, &mask)?;
                        views.push(super::capture::View {
                            name: format!("{}-light-{light_index:03}", camera.name),
                            camera: camera.params,
                            pixels,
                            mask: Some(mask.clone()),
                        });
                    }
                    decoded.push((camera_index, views));
                }
                Ok::<_, String>(decoded)
            }));
        }
        let mut decoded = Vec::with_capacity(cameras.len());
        for handle in handles {
            decoded.extend(
                handle
                    .join()
                    .map_err(|_| "OLATverse decode worker panicked".to_string())??,
            );
        }
        Ok::<_, String>(decoded)
    })?;
    decoded.sort_unstable_by_key(|&(camera_index, _)| camera_index);
    for (_, views) in decoded {
        for (capture, view) in captures.iter_mut().zip(views) {
            capture.views.push(view);
        }
    }

    let lights = light_indices
        .iter()
        .map(|&index| vec![source_lights[index].light; VIEW_NAMES.len()])
        .collect();
    Ok(crate::calibrated::Dataset {
        captures,
        lights,
        source_light_indices: light_indices.to_vec(),
    })
}

fn empty_capture(width: usize, views: usize) -> super::capture::Capture {
    super::capture::Capture {
        width,
        height: output_height(width).expect("width was validated by load_cameras"),
        views: Vec::with_capacity(views),
    }
}

fn output_height(width: usize) -> Result<usize, String> {
    if width == 0 {
        Err("OLATverse working width must be non-zero".to_string())
    } else {
        Ok((SOURCE_HEIGHT * width + SOURCE_WIDTH / 2) / SOURCE_WIDTH)
    }
}

fn load_cameras(object: &path::Path, width: usize) -> Result<Vec<Camera>, String> {
    output_height(width)?;
    let file = object.join("all_cam.json");
    let text = fs::read_to_string(&file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    parse_cameras(&text, &file)
}

fn parse_cameras(text: &str, source: &path::Path) -> Result<Vec<Camera>, String> {
    let root: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| format!("cannot parse {}: {error}", source.display()))?;
    let frames = root
        .get("frames")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} has no frame array", source.display()))?;
    let mut by_name = collections::HashMap::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let name = frame
            .get("cam_idx")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("camera frame {index} has no cam_idx"))?;
        let params = parse_camera(frame, name)?;
        if by_name.insert(name.to_string(), params).is_some() {
            return Err(format!("duplicate OLATverse camera {name}"));
        }
    }
    VIEW_NAMES
        .iter()
        .map(|&name| {
            by_name
                .remove(name)
                .map(|params| Camera {
                    name: name.to_string(),
                    params,
                })
                .ok_or_else(|| format!("{} has no camera {name}", source.display()))
        })
        .collect()
}

fn parse_camera(frame: &serde_json::Value, name: &str) -> Result<vol::CameraParams, String> {
    let matrix = parse_matrix(
        frame
            .get("transform_matrix")
            .ok_or_else(|| format!("camera {name} has no transform_matrix"))?,
        &format!("camera {name} transform_matrix"),
    )?;
    let intrinsic = parse_numbers::<4>(
        frame
            .get("camera_intrinsics")
            .ok_or_else(|| format!("camera {name} has no camera_intrinsics"))?,
        &format!("camera {name} camera_intrinsics"),
    )?;
    let [principal_x, principal_y, focal_x, focal_y] = intrinsic;
    if focal_x <= 0.0 || focal_y <= 0.0 {
        return Err(format!("camera {name} has non-positive focal length"));
    }

    let right = glam::DVec3::new(matrix[0][0], matrix[1][0], matrix[2][0]);
    let up = glam::DVec3::new(matrix[0][1], matrix[1][1], matrix[2][1]);
    let back = glam::DVec3::new(matrix[0][2], matrix[1][2], matrix[2][2]);
    // OLATverse stores OpenGL-style camera-to-world axes. Blade camera rays
    // use +Y down and +Z forward, so flip both axes and preserve handedness.
    let world_from_camera = glam::DMat3::from_cols(right, -up, -back);
    let identity_error = (world_from_camera.transpose() * world_from_camera
        - glam::DMat3::IDENTITY)
        .to_cols_array()
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let determinant = world_from_camera.determinant();
    if identity_error > 1.0e-3 || (determinant - 1.0).abs() > 1.0e-3 {
        return Err(format!(
            "camera {name} has a non-rigid rotation (error {identity_error}, determinant {determinant})"
        ));
    }
    let position = 0.001 * glam::DVec3::new(matrix[0][3], matrix[1][3], matrix[2][3]);
    let orientation = glam::DQuat::from_mat3(&world_from_camera).normalize();
    Ok(vol::CameraParams {
        cam_position: position.as_vec3().to_array(),
        cam_orientation: orientation.as_quat().to_array(),
        depth: 10.0,
        fov: [
            2.0 * (0.5 * SOURCE_WIDTH as f64 / focal_x).atan() as f32,
            2.0 * (0.5 * SOURCE_HEIGHT as f64 / focal_y).atan() as f32,
        ],
        principal: [
            (2.0 * principal_x / SOURCE_WIDTH as f64 - 1.0) as f32,
            (2.0 * principal_y / SOURCE_HEIGHT as f64 - 1.0) as f32,
        ],
    })
}

fn parse_matrix(value: &serde_json::Value, name: &str) -> Result<[[f64; 4]; 4], String> {
    let rows = value
        .as_array()
        .filter(|rows| rows.len() == 4)
        .ok_or_else(|| format!("{name} is not a 4x4 matrix"))?;
    let mut matrix = [[0.0f64; 4]; 4];
    for (index, row) in rows.iter().enumerate() {
        matrix[index] = parse_numbers::<4>(row, &format!("{name}[{index}]"))?;
    }
    if matrix[3]
        .iter()
        .zip([0.0, 0.0, 0.0, 1.0])
        .any(|(actual, expected)| (actual - expected).abs() > 1.0e-8)
    {
        return Err(format!("{name} has a non-affine final row"));
    }
    Ok(matrix)
}

fn parse_numbers<const N: usize>(
    value: &serde_json::Value,
    name: &str,
) -> Result<[f64; N], String> {
    let values = value
        .as_array()
        .filter(|values| values.len() == N)
        .ok_or_else(|| format!("{name} does not have {N} numbers"))?;
    let mut out = [0.0; N];
    for (index, value) in values.iter().enumerate() {
        out[index] = value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("{name}[{index}] is not finite"))?;
    }
    Ok(out)
}

fn load_lights(file: &path::Path) -> Result<Vec<SourceLight>, String> {
    let text = fs::read_to_string(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let root: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", file.display()))?;
    let frames = root
        .get("frames")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} has no frame array", file.display()))?;
    let mut by_index = collections::HashMap::with_capacity(LIGHT_COUNT);
    for (offset, frame) in frames.iter().enumerate() {
        let index = frame
            .get("light_idx")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value < LIGHT_COUNT)
            .ok_or_else(|| format!("light frame {offset} has an invalid light_idx"))?;
        let file_path = frame
            .get("file_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("light frame {offset} has no file_path"))?;
        let source_frame = file_path
            .rsplit_once('.')
            .and_then(|(_, suffix)| suffix.parse::<usize>().ok())
            .ok_or_else(|| format!("light {index} has invalid file_path {file_path:?}"))?;
        let position = parse_numbers::<3>(
            frame
                .get("pl_pos")
                .ok_or_else(|| format!("light {index} has no pl_pos"))?,
            &format!("light {index} pl_pos"),
        )?;
        let intensity = parse_numbers::<3>(
            frame
                .get("pl_intensity")
                .ok_or_else(|| format!("light {index} has no pl_intensity"))?,
            &format!("light {index} pl_intensity"),
        )?;
        if intensity.iter().any(|&value| value <= 0.0) {
            return Err(format!("light {index} has non-positive intensity"));
        }
        let source = SourceLight {
            frame: source_frame,
            light: vol::relight::PointLight {
                position: position.map(|value| value as f32),
                direction: [0.0, 0.0, 1.0],
                intensity: intensity.map(|value| value as f32),
                exponent: 0.0,
            },
        };
        if let Some(previous) = by_index.insert(index, source.clone()) {
            if previous != source {
                return Err(format!("light {index} has inconsistent repeated metadata"));
            }
        }
    }
    (0..LIGHT_COUNT)
        .map(|index| {
            by_index
                .remove(&index)
                .ok_or_else(|| format!("{} has no light {index}", file.display()))
        })
        .collect()
}

fn image_frames(
    directory: &path::Path,
) -> Result<collections::HashMap<usize, path::PathBuf>, String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    let mut frames = collections::HashMap::new();
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("avif") {
            continue;
        }
        let Some(frame) = path
            .file_stem()
            .and_then(|value| value.to_str())
            .and_then(|value| value.rsplit_once('.'))
            .and_then(|(_, suffix)| suffix.parse::<usize>().ok())
        else {
            return Err(format!("{} has an invalid OLAT filename", path.display()));
        };
        if frames.insert(frame, path.clone()).is_some() {
            return Err(format!(
                "{} contains duplicate frame {frame:06}",
                directory.display()
            ));
        }
    }
    Ok(frames)
}

fn load_mask(file: &path::Path, width: usize) -> Result<Vec<f32>, String> {
    let source = image::open(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?
        .into_luma8();
    if source.dimensions() != (SOURCE_WIDTH as u32, SOURCE_HEIGHT as u32) {
        return Err(format!("{} has unexpected dimensions", file.display()));
    }
    let height = output_height(width)?;
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

fn load_image(
    file: &path::Path,
    width: usize,
    height: usize,
    mask: &[f32],
) -> Result<Vec<[f32; 3]>, String> {
    let source = decode_avif(file)?;
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
                0.5 * coverage
                    * super::capture::srgb_to_linear(pixel[channel] as f32 / u8::MAX as f32)
            })
        })
        .collect())
}

fn decode_avif(file: &path::Path) -> Result<image::RgbaImage, String> {
    let decoded = avif_rust::image_from_file(file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    image::RgbaImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.rgba)
        .ok_or_else(|| format!("{} has inconsistent RGBA storage", file.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_split_is_disjoint_and_complete() {
        assert_eq!(TRAIN_VIEW_INDICES.len(), 24);
        assert_eq!(HELD_VIEW_INDICES.len(), 6);
        assert_eq!(VIEW_NAMES.len(), 30);
        assert!(TRAIN_VIEW_INDICES
            .iter()
            .all(|index| !HELD_VIEW_INDICES.contains(index)));
        let train = train_light_indices();
        let held = held_light_indices();
        assert_eq!(train.len(), 104);
        assert_eq!(held.len(), 103);
        assert!(train.iter().all(|index| !held.contains(index)));
        assert_eq!(train[0], 0);
        assert_eq!(train[103], 309);
        assert_eq!(held[0], 1);
        assert_eq!(held[102], 307);
        for polarized in ["Cam07", "Cam10", "Cam17", "Cam22", "Cam39"] {
            assert!(!VIEW_NAMES.contains(&polarized));
        }
    }

    #[test]
    fn opengl_camera_is_converted_to_blade_axes_and_meters() {
        let frame: serde_json::Value = serde_json::from_str(
            r#"{
                "transform_matrix": [
                    [1, 0, 0, 1000],
                    [0, 1, 0, 2000],
                    [0, 0, 1, 3000],
                    [0, 0, 0, 1]
                ],
                "camera_intrinsics": [750, 1422, 750, 1422]
            }"#,
        )
        .unwrap();
        let camera = parse_camera(&frame, "fixture").unwrap();
        let orientation = glam::Quat::from_array(camera.cam_orientation);

        assert_eq!(camera.cam_position, [1.0, 2.0, 3.0]);
        assert!((orientation * glam::Vec3::X - glam::Vec3::X).length() < 1.0e-6);
        assert!((orientation * glam::Vec3::Y + glam::Vec3::Y).length() < 1.0e-6);
        assert!((orientation * glam::Vec3::Z + glam::Vec3::Z).length() < 1.0e-6);
        assert!((camera.fov[0] - std::f32::consts::FRAC_PI_2).abs() < 1.0e-6);
        assert!((camera.fov[1] - std::f32::consts::FRAC_PI_2).abs() < 1.0e-6);
        assert_eq!(camera.principal, [0.0, 0.0]);
    }

    #[test]
    fn source_height_preserves_the_published_aspect() {
        assert_eq!(output_height(1500).unwrap(), 2844);
        assert_eq!(output_height(250).unwrap(), 474);
        assert!(output_height(0).is_err());
    }

    #[test]
    fn pure_rust_decoder_reads_avif_pixels() {
        let temporary = std::env::temp_dir().join(format!(
            "blade-volume-olatverse-avif-{}.avif",
            std::process::id(),
        ));
        let color = [128u8, 64, 32, 255];
        image::save_buffer_with_format(
            &temporary,
            &color.repeat(4),
            2,
            2,
            image::ColorType::Rgba8,
            image::ImageFormat::Avif,
        )
        .unwrap();

        let decoded = decode_avif(&temporary).unwrap();

        assert_eq!(decoded.dimensions(), (2, 2));
        for pixel in decoded.pixels() {
            for (actual, expected) in pixel.0.into_iter().zip(color) {
                assert!(actual.abs_diff(expected) <= 4);
            }
        }
        fs::remove_file(temporary).unwrap();
    }
}
