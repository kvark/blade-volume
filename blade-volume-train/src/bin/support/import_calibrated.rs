//! Shared pose-only capture writer for calibrated dataset importers.

use blade_volume_train as train;
use std::{fs, io, path};

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

pub fn write_colmap(
    sparse: &path::Path,
    capture: &train::inverse::capture::Capture,
    view_name: fn(usize) -> String,
) -> io::Result<()> {
    fs::create_dir_all(sparse)?;
    let reference = capture
        .views
        .first()
        .expect("calibrated dataset loader returned no views");
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

pub fn write_capture_images(
    directory: &path::Path,
    capture: &train::inverse::capture::Capture,
    view_name: fn(usize) -> String,
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

pub fn write_masks(
    directory: &path::Path,
    capture: &train::inverse::capture::Capture,
    view_name: fn(usize) -> String,
) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    for (index, view) in capture.views.iter().enumerate() {
        let mask = view
            .mask
            .as_ref()
            .ok_or_else(|| format!("calibrated view {index} has no mask"))?;
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
