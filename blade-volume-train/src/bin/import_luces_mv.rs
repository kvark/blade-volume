//! Prepare one LUCES-MV object for the pose-only point-cloud training path.
//!
//! The source is already geometrically and radiometrically calibrated. This
//! importer writes one shared COLMAP pose bundle, normalized sRGB images for
//! each LED, foreground masks, and fixed view/light lists. It does not read or
//! convert the released ground-truth mesh.

use blade_volume_train as train;
use std::{fs, io, path};

#[derive(argh::FromArgs)]
/// Convert a downloaded LUCES-MV object to the training layout.
struct Args {
    /// object directory containing `view_000` through `view_066`
    #[argh(option)]
    input: String,

    /// first official camera parameter text file
    #[argh(option)]
    camera_one: String,

    /// second official camera parameter text file
    #[argh(option)]
    camera_two: String,

    /// new output directory
    #[argh(option)]
    output: String,

    /// output image width (default 400; height preserves aspect)
    #[argh(option, default = "400")]
    width: usize,
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
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

fn view_name(index: usize) -> String {
    format!("view_{:03}.png", train::inverse::luces::VIEW_IDS[index])
}

fn write_colmap(sparse: &path::Path, capture: &train::inverse::capture::Capture) -> io::Result<()> {
    fs::create_dir_all(sparse)?;
    let reference = capture
        .views
        .first()
        .expect("LUCES-MV loader returned no views");
    let width = capture.width as f64;
    let height = capture.height as f64;
    let focal_x = 0.5 * width / (0.5 * reference.camera.fov[0] as f64).tan();
    let focal_y = 0.5 * height / (0.5 * reference.camera.fov[1] as f64).tan();
    let principal_x = 0.5 * width * (reference.camera.principal[0] as f64 + 1.0);
    let principal_y = 0.5 * height * (reference.camera.principal[1] as f64 + 1.0);

    let mut cameras = io::BufWriter::new(fs::File::create(sparse.join("cameras.bin"))?);
    write_u64(&mut cameras, 1)?;
    write_u32(&mut cameras, 1)?;
    write_i32(&mut cameras, 1)?;
    write_u64(&mut cameras, capture.width as u64)?;
    write_u64(&mut cameras, capture.height as u64)?;
    for value in [focal_x, focal_y, principal_x, principal_y] {
        write_f64(&mut cameras, value)?;
    }
    io::Write::flush(&mut cameras)?;

    let mut images = io::BufWriter::new(fs::File::create(sparse.join("images.bin"))?);
    write_u64(&mut images, capture.views.len() as u64)?;
    for (index, view) in capture.views.iter().enumerate() {
        let world_from_camera = glam::Quat::from_array(view.camera.cam_orientation).normalize();
        let camera_from_world = world_from_camera.inverse();
        let position = glam::Vec3::from(view.camera.cam_position);
        let translation = camera_from_world * -position;
        write_u32(&mut images, index as u32 + 1)?;
        for value in [
            camera_from_world.w,
            camera_from_world.x,
            camera_from_world.y,
            camera_from_world.z,
            translation.x,
            translation.y,
            translation.z,
        ] {
            write_f64(&mut images, value as f64)?;
        }
        write_u32(&mut images, 1)?;
        io::Write::write_all(&mut images, view_name(index).as_bytes())?;
        io::Write::write_all(&mut images, &[0])?;
        write_u64(&mut images, 0)?;
    }
    io::Write::flush(&mut images)?;

    let mut points = io::BufWriter::new(fs::File::create(sparse.join("points3D.bin"))?);
    write_u64(&mut points, 0)?;
    io::Write::flush(&mut points)
}

fn write_capture_images(
    directory: &path::Path,
    capture: &train::inverse::capture::Capture,
) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    for (index, view) in capture.views.iter().enumerate() {
        let pixels = view
            .pixels
            .iter()
            .flat_map(|rgb| {
                rgb.map(|value| {
                    (train::inverse::capture::linear_to_srgb(value) * u8::MAX as f32).round() as u8
                })
            })
            .collect();
        let image = image::RgbImage::from_raw(capture.width as u32, capture.height as u32, pixels)
            .expect("capture has inconsistent pixel dimensions");
        let output = directory.join(view_name(index));
        image
            .save(&output)
            .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    }
    Ok(())
}

fn write_masks(
    directory: &path::Path,
    capture: &train::inverse::capture::Capture,
) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    for (index, view) in capture.views.iter().enumerate() {
        let mask = view
            .mask
            .as_ref()
            .ok_or_else(|| format!("LUCES-MV view {index} has no mask"))?;
        let pixels = mask
            .iter()
            .map(|&coverage| (coverage.clamp(0.0, 1.0) * u8::MAX as f32).round() as u8)
            .collect();
        let image = image::GrayImage::from_raw(capture.width as u32, capture.height as u32, pixels)
            .expect("capture has inconsistent mask dimensions");
        let output = directory.join(view_name(index));
        image
            .save(&output)
            .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    }
    Ok(())
}

fn write_splits(output: &path::Path, views: usize) -> io::Result<()> {
    let view_list = |held: bool| {
        (0..views)
            .filter(|index| train::inverse::luces::HELD_VIEW_INDICES.contains(index) == held)
            .map(view_name)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    let light_list = |held: bool| {
        (0..train::inverse::luces::LIGHT_COUNT)
            .filter(|index| train::inverse::luces::HELD_LIGHT_INDICES.contains(index) == held)
            .map(|index| format!("{:02}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    fs::write(output.join("train-views.txt"), view_list(false))?;
    fs::write(output.join("test-views.txt"), view_list(true))?;
    fs::write(output.join("train-lights.txt"), light_list(false))?;
    fs::write(output.join("test-lights.txt"), light_list(true))
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let camera_one = path::Path::new(&args.camera_one);
    let camera_two = path::Path::new(&args.camera_two);
    let output = path::Path::new(&args.output);
    if output.exists() {
        return Err(format!("output {} already exists", output.display()));
    }
    fs::create_dir_all(output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;

    for light in 0..train::inverse::luces::LIGHT_COUNT {
        let dataset =
            train::inverse::luces::load(input, camera_one, camera_two, args.width, &[light])?;
        let capture = &dataset.captures[0];
        if light == 0 {
            write_colmap(&output.join("sparse/0"), capture)
                .map_err(|error| format!("cannot write COLMAP poses: {error}"))?;
            write_masks(&output.join("masks"), capture)?;
            write_splits(output, capture.views.len())
                .map_err(|error| format!("cannot write fixed splits: {error}"))?;
        }
        write_capture_images(
            &output.join(format!("light-{:02}/images", light + 1)),
            capture,
        )?;
        println!(
            "wrote LED {:02}/{}",
            light + 1,
            train::inverse::luces::LIGHT_COUNT
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
    fn fixed_splits_are_disjoint_and_complete() {
        let train_views: Vec<_> = (0..12)
            .filter(|index| !train::inverse::luces::HELD_VIEW_INDICES.contains(index))
            .collect();
        let test_views: Vec<_> = (0..12)
            .filter(|index| train::inverse::luces::HELD_VIEW_INDICES.contains(index))
            .collect();
        let train_lights: Vec<_> = (0..train::inverse::luces::LIGHT_COUNT)
            .filter(|index| !train::inverse::luces::HELD_LIGHT_INDICES.contains(index))
            .collect();
        assert_eq!(train_views.len(), 9);
        assert_eq!(test_views.len(), 3);
        assert_eq!(train_lights.len(), 12);
        assert_eq!(train::inverse::luces::HELD_LIGHT_INDICES.len(), 3);
        assert_eq!(
            train_views.as_slice(),
            train::inverse::luces::TRAIN_VIEW_INDICES
        );
        assert_eq!(
            train_lights.as_slice(),
            train::inverse::luces::TRAIN_LIGHT_INDICES
        );
        assert!(train_views.iter().all(|index| !test_views.contains(index)));
        assert!(train_lights
            .iter()
            .all(|index| !train::inverse::luces::HELD_LIGHT_INDICES.contains(index)));
    }
}
