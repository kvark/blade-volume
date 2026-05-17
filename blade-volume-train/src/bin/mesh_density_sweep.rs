//! Measure how mesh→foam conversion quality changes with sample density.
//!
//! Converts a glTF mesh at several `density` settings and renders each via
//! the CPU RadFoam tracer (no GPU RT, no training). Each lower-density
//! render is PSNR-compared to the highest-density render — the highest
//! density is treated as a proxy "truth" since we don't have a triangle
//! rasteriser to render the mesh directly. This answers the question
//! "how good is the converter at preserving mesh appearance" without
//! adding new dependencies.
//!
//! Usage:
//!   cargo run --release -p blade-volume-train --bin mesh_density_sweep -- \
//!       --input blade-volume-test/data/police.glb \
//!       --output-dir /tmp/police_sweep --width 128 --height 128

use argh::FromArgs;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train::{metrics, pipeline, render};
use std::path;

/// Sweep blade-volume-convert density vs CPU-rendered foam PSNR.
#[derive(FromArgs)]
struct Args {
    /// path to the input glTF/glb file
    #[argh(option)]
    input: String,

    /// directory for output PNGs (one per density)
    #[argh(option)]
    output_dir: String,

    /// render width (default 128)
    #[argh(option, default = "128")]
    width: u32,

    /// render height (default 128)
    #[argh(option, default = "128")]
    height: u32,

    /// density values to sweep (comma-separated; default 10,40,160,640)
    #[argh(option, default = "String::from(\"10,40,160,640\")")]
    densities: String,

    /// max traversal steps per ray (default 256)
    #[argh(option, default = "256")]
    max_steps: u32,
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();
    let densities: Vec<f32> = args
        .densities
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(densities.len() >= 2, "need at least 2 density values");

    let out_dir = path::Path::new(&args.output_dir);
    std::fs::create_dir_all(out_dir).unwrap();

    let mut renders: Vec<(f32, usize, Vec<f32>)> = Vec::new();
    for &d in &densities {
        let options = convert::ConvertOptions {
            output: convert::OutputKind::RadFoam,
            density: d,
            surface_density_scale: 1.0,
            interior_density_scale: 0.25,
            surface_opacity: 0.95,
            interior_opacity: 0.6,
            ..Default::default()
        };
        let mut model = convert::convert_gltf(path::Path::new(&args.input), &options)
            .unwrap_or_else(|err| panic!("conversion failed at density {d}: {err:?}"));
        if model.adjacency.is_none() {
            model.compute_adjacency_default();
        }
        let n_points = model.len();
        let cam = camera_from_bounds(&model);
        // Cell-index 0 is a different point in space at each density, so
        // hard-coding it makes the per-density renders incomparable. Use
        // the kd-tree-based nearest-to-camera lookup so each render starts
        // in the cell that contains (or is closest to) the eye, consistent
        // across densities.
        let start_point =
            pipeline::pick_start_cell(&model, glam::Vec3::from_array(cam.cam_position));
        let settings = render::RenderSettings {
            width: args.width,
            height: args.height,
            start_point,
            max_steps: args.max_steps,
            weight_threshold: 1e-4,
        };
        let pixels = render::render_cpu(&model, &cam, settings);
        let rgb = metrics::rgba_to_rgb(&pixels);
        let png_path = out_dir.join(format!("police_d{}.png", d as u32));
        save_rgb(&png_path, &rgb, args.width, args.height);
        println!(
            "density {d:>6.1}: {n_points:>6} points → {}",
            png_path.display()
        );
        renders.push((d, n_points, rgb));
    }

    // Treat the highest-density render as the truth proxy.
    let truth_idx = renders
        .iter()
        .enumerate()
        .max_by(|&(_, a), &(_, b)| a.0.total_cmp(&b.0))
        .map(|(i, _)| i)
        .unwrap();
    let truth = renders[truth_idx].clone();
    println!(
        "\ntruth proxy = density {} ({} points). PSNR of other configs:",
        truth.0, truth.1
    );
    for entry in &renders {
        if std::ptr::eq(&entry.2, &truth.2) {
            continue;
        }
        let mut clamped = entry.2.clone();
        for p in clamped.iter_mut() {
            *p = p.clamp(0.0, 1.0);
        }
        let psnr = metrics::psnr(&clamped, &truth.2);
        let mae = metrics::mae_rgb(&clamped, &truth.2);
        println!(
            "  density {:>6.1} ({:>6} points): PSNR {psnr:6.2} dB, MAE {mae:.4}",
            entry.0, entry.1
        );
    }
}

fn save_rgb(path: &path::Path, rgb: &[f32], w: u32, h: u32) {
    let mut img = image::RgbImage::new(w, h);
    for (i, px) in img.pixels_mut().enumerate() {
        let r = (rgb[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (rgb[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (rgb[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
        *px = image::Rgb([r, g, b]);
    }
    img.save(path).unwrap();
}

fn camera_from_bounds(model: &vol::PointCloudModel) -> vol::CameraParams {
    let (center, radius) = compute_bounds(model);
    let view_dir = glam::Vec3::new(0.8, -0.4, -0.6).normalize();
    let distance = radius * 2.5 + 0.1;
    let position = center - view_dir * distance;
    let orientation = look_at_orientation(position, center, -glam::Vec3::Y);
    vol::CameraParams {
        cam_position: position.into(),
        depth: distance + radius * 2.0,
        cam_orientation: orientation.into(),
        fov: [1.0, 1.0],
        pad: [0, 0],
    }
}

fn compute_bounds(model: &vol::PointCloudModel) -> (glam::Vec3, f32) {
    if model.points.is_empty() {
        return (glam::Vec3::ZERO, 1.0);
    }
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for p in &model.points {
        let v = glam::Vec3::new(p.x, p.y, p.z);
        min = min.min(v);
        max = max.max(v);
    }
    let center = (min + max) * 0.5;
    let mut radius = 0.0f32;
    for p in &model.points {
        let v = glam::Vec3::new(p.x, p.y, p.z);
        radius = radius.max((v - center).length());
    }
    (center, radius.max(0.1))
}

fn look_at_orientation(position: glam::Vec3, target: glam::Vec3, up: glam::Vec3) -> glam::Quat {
    let forward = (target - position).normalize();
    let mut right = up.cross(forward).normalize_or_zero();
    if right.length_squared() < 1e-6 {
        right = glam::Vec3::X.cross(forward).normalize_or_zero();
    }
    let up = forward.cross(right);
    let basis = glam::Mat3::from_cols(right, up, forward);
    glam::Quat::from_mat3(&basis)
}
