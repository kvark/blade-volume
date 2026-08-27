//! Prepare an OpenIllumination object for the existing COLMAP capture path.
//!
//! The public dataset already carries calibrated camera-to-world matrices but
//! no sparse SfM point cloud. This writes pose-only COLMAP binaries, mirrors
//! the masks under the photograph names, and converts selected OLAT directions
//! to normalized distant-light environments. Train the initial foam with
//! `train_colmap --initialization camera-lattice`.

use blade_volume as vol;
use std::{collections, fs, io, path};

const DEFAULT_LIGHTS: [&str; 4] = ["000", "062", "082", "092"];
const ENVIRONMENT_WIDTH: usize = 64;
const LIGHT_ANGULAR_RADIUS: f32 = 0.08;

#[derive(argh::FromArgs)]
/// Convert one downloaded OpenIllumination object to the training layout.
struct Args {
    /// object root containing `Lights/` and `output/transforms_*.json`
    #[argh(option)]
    input: String,

    /// new output directory for `sparse/0`, masks, and light environments
    #[argh(option)]
    output: String,

    /// official `tools/ps_recon/light_pos.npy`
    #[argh(option)]
    light_positions: String,

    /// light label, optionally followed by = and comma-separated LED indices;
    /// repeatable (default 000,062,082,092; use LABEL=all for every LED)
    #[argh(option)]
    light: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Light {
    label: String,
    indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct Frame {
    name: String,
    held_out: bool,
    fov_x: f64,
    calibrated_width: u32,
    world_from_camera: glam::DMat4,
}

struct PreparedFrame {
    frame: Frame,
    image_name: String,
    dimensions: (u32, u32),
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
}

fn value_number(value: &serde_json::Value, name: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{name} is not a finite number"))
}

fn parse_matrix(value: &serde_json::Value, name: &str) -> Result<glam::DMat4, String> {
    let rows = value
        .as_array()
        .filter(|rows| rows.len() == 4)
        .ok_or_else(|| format!("{name} is not a 4x4 matrix"))?;
    let mut matrix = [[0.0f64; 4]; 4];
    for (row_index, row) in rows.iter().enumerate() {
        let values = row
            .as_array()
            .filter(|values| values.len() == 4)
            .ok_or_else(|| format!("{name} row {row_index} does not have four entries"))?;
        for (column, value) in values.iter().enumerate() {
            matrix[row_index][column] =
                value_number(value, &format!("{name}[{row_index}][{column}]"))?;
        }
    }
    if matrix[3]
        .iter()
        .zip([0.0, 0.0, 0.0, 1.0])
        .any(|(actual, expected)| (actual - expected).abs() > 1.0e-8)
    {
        return Err(format!("{name} has a non-affine final row"));
    }
    let rotation = glam::DMat3::from_cols(
        glam::DVec3::new(matrix[0][0], matrix[1][0], matrix[2][0]),
        glam::DVec3::new(matrix[0][1], matrix[1][1], matrix[2][1]),
        glam::DVec3::new(matrix[0][2], matrix[1][2], matrix[2][2]),
    );
    let identity_error = (rotation.transpose() * rotation - glam::DMat3::IDENTITY)
        .to_cols_array()
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if identity_error > 1.0e-5 || (rotation.determinant() - 1.0).abs() > 1.0e-5 {
        return Err(format!("{name} does not carry a rigid camera rotation"));
    }
    Ok(glam::DMat4::from_cols_array_2d(&[
        [matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]],
        [matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]],
    ]))
}

fn parse_frames(text: &str, source: &path::Path, held_out: bool) -> Result<Vec<Frame>, String> {
    let root: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| format!("cannot parse {}: {error}", source.display()))?;
    let frames = root
        .get("frames")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{} has no frame object", source.display()))?;
    let mut out = Vec::with_capacity(frames.len());
    for (name, value) in frames {
        let object = value
            .as_object()
            .ok_or_else(|| format!("frame {name} is not an object"))?;
        let fov_x = value_number(
            object
                .get("camera_angle_x")
                .ok_or_else(|| format!("frame {name} has no camera_angle_x"))?,
            &format!("frame {name} camera_angle_x"),
        )?;
        if !(0.0..std::f64::consts::PI).contains(&fov_x) || fov_x == 0.0 {
            return Err(format!(
                "frame {name} has an invalid horizontal field of view"
            ));
        }
        let calibrated_width = object
            .get("calib_imgw")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| format!("frame {name} has no valid calib_imgw"))?;
        let world_from_camera = parse_matrix(
            object
                .get("transform_matrix")
                .ok_or_else(|| format!("frame {name} has no transform_matrix"))?,
            &format!("frame {name} transform_matrix"),
        )?;
        out.push(Frame {
            name: name.clone(),
            held_out,
            fov_x,
            calibrated_width,
            world_from_camera,
        });
    }
    out.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(out)
}

fn load_frames(input: &path::Path) -> Result<Vec<Frame>, String> {
    let mut by_name = collections::BTreeMap::new();
    for split in ["train", "test"] {
        let source = input.join(format!("output/transforms_{split}.json"));
        let text = fs::read_to_string(&source)
            .map_err(|error| format!("cannot read {}: {error}", source.display()))?;
        for frame in parse_frames(&text, &source, split == "test")? {
            let name = frame.name.clone();
            if by_name.insert(frame.name.clone(), frame).is_some() {
                return Err(format!(
                    "camera name occurs in both OpenIllumination splits: {name}"
                ));
            }
        }
    }
    if by_name.len() < 4 {
        return Err(format!(
            "the object provides only {} calibrated cameras",
            by_name.len()
        ));
    }
    Ok(by_name.into_values().collect())
}

fn image_for(light_directory: &path::Path, stem: &str) -> Result<path::PathBuf, String> {
    let mut matches = Vec::new();
    for extension in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"] {
        let candidate = light_directory.join(format!("{stem}.{extension}"));
        if candidate.is_file() {
            matches.push(candidate);
        }
    }
    match matches.len() {
        1 => Ok(matches.pop().unwrap()),
        0 => Err(format!(
            "{} contains no image for camera {stem}",
            light_directory.display()
        )),
        _ => Err(format!(
            "{} contains several images for camera {stem}",
            light_directory.display()
        )),
    }
}

fn write_u32(writer: &mut impl io::Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl io::Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_i32(writer: &mut impl io::Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_f64(writer: &mut impl io::Write, value: f64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_bytes(writer: &mut impl io::Write, bytes: &[u8]) -> io::Result<()> {
    writer.write_all(bytes)
}

fn colmap_pose(world_from_camera: glam::DMat4) -> (glam::DQuat, glam::DVec3) {
    let camera_from_world = world_from_camera.inverse();
    let rotation = glam::DMat3::from_mat4(camera_from_world);
    (
        glam::DQuat::from_mat3(&rotation).normalize(),
        camera_from_world.w_axis.truncate(),
    )
}

fn write_colmap_files(sparse: &path::Path, records: &[PreparedFrame]) -> io::Result<()> {
    let cameras_path = sparse.join("cameras.bin");
    let cameras_file = fs::File::create(cameras_path)?;
    let mut cameras = io::BufWriter::new(cameras_file);
    write_u64(&mut cameras, records.len() as u64)?;
    for (index, record) in records.iter().enumerate() {
        let id = index as u32 + 1;
        let focal = 0.5 * record.dimensions.0 as f64 / (0.5 * record.frame.fov_x).tan();
        write_u32(&mut cameras, id)?;
        write_i32(&mut cameras, 1)?;
        write_u64(&mut cameras, record.dimensions.0 as u64)?;
        write_u64(&mut cameras, record.dimensions.1 as u64)?;
        for value in [
            focal,
            focal,
            0.5 * record.dimensions.0 as f64,
            0.5 * record.dimensions.1 as f64,
        ] {
            write_f64(&mut cameras, value)?;
        }
    }
    io::Write::flush(&mut cameras)?;

    let images_path = sparse.join("images.bin");
    let images_file = fs::File::create(images_path)?;
    let mut images = io::BufWriter::new(images_file);
    write_u64(&mut images, records.len() as u64)?;
    for (index, record) in records.iter().enumerate() {
        let id = index as u32 + 1;
        let (orientation, translation) = colmap_pose(record.frame.world_from_camera);
        write_u32(&mut images, id)?;
        for value in [
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z,
            translation.x,
            translation.y,
            translation.z,
        ] {
            write_f64(&mut images, value)?;
        }
        write_u32(&mut images, id)?;
        write_bytes(&mut images, record.image_name.as_bytes())?;
        write_bytes(&mut images, &[0])?;
        write_u64(&mut images, 0)?;
    }
    io::Write::flush(&mut images)?;

    let points_path = sparse.join("points3D.bin");
    let mut points = io::BufWriter::new(fs::File::create(points_path)?);
    write_u64(&mut points, 0)?;
    io::Write::flush(&mut points)
}

fn write_colmap(
    input: &path::Path,
    output: &path::Path,
    frames: &[Frame],
    lights: &[Light],
) -> Result<(), String> {
    let sparse = output.join("sparse/0");
    let masks = output.join("masks");
    fs::create_dir_all(&sparse)
        .and_then(|()| fs::create_dir_all(&masks))
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    let primary = input.join(format!("Lights/{}/raw_undistorted", lights[0].label));
    let mut records = Vec::with_capacity(frames.len());
    for frame in frames {
        let primary_image = image_for(&primary, &frame.name)?;
        let image_name = primary_image
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let dimensions = image::image_dimensions(&primary_image)
            .map_err(|error| format!("cannot inspect {}: {error}", primary_image.display()))?;
        if dimensions.0 != frame.calibrated_width {
            return Err(format!(
                "{} is {} pixels wide but its calibration says {}",
                primary_image.display(),
                dimensions.0,
                frame.calibrated_width,
            ));
        }
        for light in lights.iter().skip(1) {
            let directory = input.join(format!("Lights/{}/raw_undistorted", light.label));
            let image = image_for(&directory, &frame.name)?;
            if image.file_name() != primary_image.file_name() {
                return Err(format!(
                    "camera {} has inconsistent filenames between selected lights",
                    frame.name
                ));
            }
            let other_dimensions = image::image_dimensions(&image)
                .map_err(|error| format!("cannot inspect {}: {error}", image.display()))?;
            if other_dimensions != dimensions {
                return Err(format!(
                    "camera {} changes resolution between selected lights",
                    frame.name
                ));
            }
        }
        let source_mask = input.join(format!("output/obj_masks/{}.png", frame.name));
        let destination_mask = masks.join(format!("{}.png", frame.name));
        fs::copy(&source_mask, &destination_mask).map_err(|error| {
            format!(
                "cannot copy {} to {}: {error}",
                source_mask.display(),
                destination_mask.display()
            )
        })?;
        records.push(PreparedFrame {
            frame: frame.clone(),
            image_name,
            dimensions,
        });
    }
    write_colmap_files(&sparse, &records)
        .map_err(|error| format!("cannot write {}: {error}", sparse.display()))?;
    let test_names = records
        .iter()
        .filter(|record| record.frame.held_out)
        .map(|record| record.image_name.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(output.join("test.txt"), format!("{test_names}\n"))
        .map_err(|error| format!("cannot write dataset split: {error}"))
}

fn parse_light_directions(bytes: &[u8]) -> Result<Vec<glam::Vec3>, String> {
    const COUNT: usize = 142;
    const COMPONENTS: usize = 3;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" || bytes[6..8] != [1, 0] {
        return Err("light positions are not a NumPy v1 array".to_string());
    }
    let header_length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10usize
        .checked_add(header_length)
        .ok_or_else(|| "light-position header length overflows".to_string())?;
    let header = std::str::from_utf8(
        bytes
            .get(10..data_start)
            .ok_or_else(|| "light-position header is truncated".to_string())?,
    )
    .map_err(|_| "light-position header is not UTF-8".to_string())?;
    if !header.contains("'descr': '<f8'")
        || !header.contains("'fortran_order': False")
        || !header.contains("'shape': (142, 3)")
    {
        return Err(format!("unsupported light-position array: {header}"));
    }
    let data = bytes
        .get(data_start..)
        .ok_or_else(|| "light-position body is missing".to_string())?;
    if data.len() != COUNT * COMPONENTS * 8 {
        return Err(format!(
            "light-position body is {} bytes, expected {}",
            data.len(),
            COUNT * COMPONENTS * 8
        ));
    }
    let mut directions = Vec::with_capacity(COUNT);
    let (rows, remainder) = data.as_chunks::<24>();
    debug_assert!(remainder.is_empty());
    for row in rows {
        let component = |index: usize| {
            let start = index * 8;
            f64::from_le_bytes(row[start..start + 8].try_into().unwrap()) as f32
        };
        let direction = glam::Vec3::new(component(0), component(1), component(2));
        if !direction.is_finite() || direction.length() <= f32::EPSILON {
            return Err("light-position array contains an invalid direction".to_string());
        }
        directions.push(direction.normalize());
    }
    Ok(directions)
}

fn write_environments(
    output: &path::Path,
    source: &path::Path,
    lights: &[Light],
) -> Result<(), String> {
    let bytes =
        fs::read(source).map_err(|error| format!("cannot read {}: {error}", source.display()))?;
    let directions = parse_light_directions(&bytes)?;
    for light in lights {
        let environment = environment_for(&directions, &light.indices);
        let destination = output.join(format!("light-{}.f32", light.label));
        vol::io::try_save_environment(&destination, &environment)
            .map_err(|error| format!("cannot write {}: {error}", destination.display()))?;
    }
    Ok(())
}

fn environment_for(directions: &[glam::Vec3], indices: &[usize]) -> vol::relight::Environment {
    let mut environment =
        vol::relight::Environment::uniform([0.0; 3], ENVIRONMENT_WIDTH, ENVIRONMENT_WIDTH / 2);
    for &index in indices {
        let emitter = vol::relight::Environment::directional(
            directions[index],
            [1.0; 3],
            LIGHT_ANGULAR_RADIUS,
            ENVIRONMENT_WIDTH,
            ENVIRONMENT_WIDTH / 2,
        );
        for (target, source) in environment.texels.iter_mut().zip(&emitter.texels) {
            for channel in 0..3 {
                target[channel] += source[channel];
            }
        }
    }
    environment
}

fn parse_light(value: &str) -> Result<Light, String> {
    let (label, members) = value
        .split_once('=')
        .map_or((value, None), |(label, members)| (label, Some(members)));
    if label.len() != 3 || !label.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("light label must be three digits: {label}"));
    }
    let indices = match members {
        None => vec![label
            .parse::<usize>()
            .ok()
            .filter(|&index| index < 142)
            .ok_or_else(|| format!("OLAT index must be between 000 and 141: {label}"))?],
        Some("all") => (0..142).collect(),
        Some(members) => {
            let mut unique = collections::BTreeSet::new();
            for member in members.split(',') {
                let index = member
                    .parse::<usize>()
                    .ok()
                    .filter(|&index| index < 142)
                    .ok_or_else(|| format!("LED index must be between 0 and 141: {member}"))?;
                if !unique.insert(index) {
                    return Err(format!(
                        "LED index occurs more than once in {label}: {index}"
                    ));
                }
            }
            if unique.is_empty() {
                return Err(format!("light group {label} has no LED indices"));
            }
            unique.into_iter().collect()
        }
    };
    Ok(Light {
        label: label.to_string(),
        indices,
    })
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let output = path::Path::new(&args.output);
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing output {}",
            output.display()
        ));
    }
    let light_values: Vec<String> = if args.light.is_empty() {
        DEFAULT_LIGHTS
            .iter()
            .map(|light| light.to_string())
            .collect()
    } else {
        args.light.clone()
    };
    let lights: Vec<_> = light_values
        .iter()
        .map(|light| parse_light(light))
        .collect::<Result<_, _>>()?;
    let mut unique_lights = collections::BTreeSet::new();
    for light in &lights {
        if !unique_lights.insert(&light.label) {
            return Err(format!(
                "light label was selected more than once: {}",
                light.label
            ));
        }
    }
    let frames = load_frames(input)?;
    let result = write_colmap(input, output, &frames, &lights)
        .and_then(|()| write_environments(output, path::Path::new(&args.light_positions), &lights));
    if let Err(message) = result {
        let _ = fs::remove_dir_all(output);
        return Err(message);
    }
    println!(
        "prepared {} cameras and {} lights in {}",
        frames.len(),
        lights.len(),
        output.display()
    );
    println!("masks: {}", output.join("masks").display());
    println!(
        "official held cameras: {}",
        output.join("test.txt").display()
    );
    for light in &lights {
        println!(
            "light {} ({} LEDs): images={} environment={}",
            light.label,
            light.indices.len(),
            input
                .join(format!("Lights/{}/raw_undistorted", light.label))
                .display(),
            output.join(format!("light-{}.f32", light.label)).display(),
        );
    }
    Ok(())
}

fn main() {
    let args: Args = argh::from_env();
    if let Err(message) = run(&args) {
        fail(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_pose_round_trips_through_colmap_conventions() {
        let position = glam::DVec3::new(0.4, -0.2, 1.3);
        let orientation = glam::DQuat::from_rotation_y(0.7) * glam::DQuat::from_rotation_x(-0.2);
        let world_from_camera = glam::DMat4::from_rotation_translation(orientation, position);
        let temporary = std::env::temp_dir().join(format!(
            "blade-volume-openillumination-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("pose")
        ));
        let _ = fs::remove_dir_all(&temporary);
        fs::create_dir_all(&temporary).unwrap();
        let records = [PreparedFrame {
            frame: Frame {
                name: "A1".to_string(),
                held_out: true,
                fov_x: 0.6,
                calibrated_width: 3_000,
                world_from_camera,
            },
            image_name: "A1.jpg".to_string(),
            dimensions: (3_000, 4_096),
        }];
        write_colmap_files(&temporary, &records).unwrap();
        let reconstruction =
            blade_volume_train::colmap::try_load_reconstruction(&temporary).unwrap();
        let image = &reconstruction.images[0];
        let camera_from_world = glam::DMat4::from_rotation_translation(
            glam::DQuat::from_xyzw(
                image.quat_wxyz[1],
                image.quat_wxyz[2],
                image.quat_wxyz[3],
                image.quat_wxyz[0],
            ),
            glam::DVec3::from(image.translation),
        );
        let recovered = camera_from_world.inverse();
        let recovered_rotation = glam::DQuat::from_mat4(&recovered).normalize();
        assert!((recovered.w_axis.truncate() - position).length() < 1.0e-12);
        assert!(recovered_rotation.abs_diff_eq(orientation, 1.0e-12));
        assert_eq!(reconstruction.cameras[&1].width, 3_000);
        assert_eq!(reconstruction.cameras[&1].height, 4_096);
        assert!(reconstruction.points.is_empty());
        fs::remove_dir_all(temporary).unwrap();
    }

    #[test]
    fn frame_parser_rejects_non_rigid_camera_matrices() {
        let source = path::Path::new("fixture.json");
        let invalid = r#"{"frames":{"A1":{"camera_angle_x":0.3,"calib_imgw":3000,
            "transform_matrix":[[2,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}}}"#;
        assert!(parse_frames(invalid, source, false).is_err());
    }

    #[test]
    fn light_spec_accepts_olat_groups_and_all_emitters() {
        assert_eq!(
            parse_light("062").unwrap(),
            Light {
                label: "062".to_string(),
                indices: vec![62],
            }
        );
        assert_eq!(
            parse_light("001=8,2,5").unwrap(),
            Light {
                label: "001".to_string(),
                indices: vec![2, 5, 8],
            }
        );
        let all = parse_light("013=all").unwrap();
        assert_eq!(all.indices.len(), 142);
        assert_eq!(all.indices[0], 0);
        assert_eq!(all.indices[141], 141);
    }

    #[test]
    fn light_spec_rejects_empty_duplicate_and_invalid_groups() {
        assert!(parse_light("01=2").is_err());
        assert!(parse_light("001=").is_err());
        assert!(parse_light("001=2,2").is_err());
        assert!(parse_light("001=142").is_err());
        assert!(parse_light("142").is_err());
    }

    #[test]
    fn grouped_environment_adds_individual_emitters() {
        let directions = [glam::Vec3::X, glam::Vec3::Y];
        let first = environment_for(&directions, &[0]);
        let second = environment_for(&directions, &[1]);
        let grouped = environment_for(&directions, &[0, 1]);
        for ((grouped, first), second) in
            grouped.texels.iter().zip(&first.texels).zip(&second.texels)
        {
            assert_eq!(grouped[0], first[0] + second[0]);
            assert_eq!(grouped[1], first[1] + second[1]);
            assert_eq!(grouped[2], first[2] + second[2]);
        }
    }
}
