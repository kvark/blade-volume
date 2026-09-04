//! Shared mechanics for calibrated multi-view/multi-light datasets.

use crate as train;
use blade_volume as vol;
use blade_volume_convert as convert;
use std::{fs, io, path};

pub struct Dataset {
    pub captures: Vec<train::inverse::capture::Capture>,
    pub lights: Vec<Vec<vol::relight::PointLight>>,
    pub source_light_indices: Vec<usize>,
}

pub struct FitOptions<'a> {
    pub surface: &'a path::Path,
    pub output: &'a path::Path,
    pub gaussian_output: Option<&'a path::Path>,
    pub dump: Option<&'a path::Path>,
    pub width: usize,
    pub rounds: usize,
    pub normal_candidates: usize,
    pub albedo_ceiling: f32,
    pub point_light_visibility: bool,
    pub train_views: &'a [usize],
    pub held_views: &'a [usize],
    pub light_digits: usize,
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

/// Write a pose-only COLMAP model for a calibrated capture.
pub fn write_colmap(
    sparse: &path::Path,
    capture: &train::inverse::capture::Capture,
    view_name: fn(usize) -> String,
) -> io::Result<()> {
    fs::create_dir_all(sparse)?;
    if capture.views.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibrated dataset loader returned no views",
        ));
    }
    let width = capture.width as f64;
    let height = capture.height as f64;

    let mut cameras = io::BufWriter::new(fs::File::create(sparse.join("cameras.bin"))?);
    write_u64(&mut cameras, capture.views.len() as u64)?;
    for (index, view) in capture.views.iter().enumerate() {
        let focal_x = 0.5 * width / (0.5 * view.camera.fov[0] as f64).tan();
        let focal_y = 0.5 * height / (0.5 * view.camera.fov[1] as f64).tan();
        let principal_x = 0.5 * width * (view.camera.principal[0] as f64 + 1.0);
        let principal_y = 0.5 * height * (view.camera.principal[1] as f64 + 1.0);
        write_u32(&mut cameras, index as u32 + 1)?;
        write_i32(&mut cameras, 1)?;
        write_u64(&mut cameras, capture.width as u64)?;
        write_u64(&mut cameras, capture.height as u64)?;
        for value in [focal_x, focal_y, principal_x, principal_y] {
            write_f64(&mut cameras, value)?;
        }
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
        write_u32(&mut images, index as u32 + 1)?;
        io::Write::write_all(&mut images, view_name(index).as_bytes())?;
        io::Write::write_all(&mut images, &[0])?;
        write_u64(&mut images, 0)?;
    }
    io::Write::flush(&mut images)?;

    let mut points = io::BufWriter::new(fs::File::create(sparse.join("points3D.bin"))?);
    write_u64(&mut points, 0)?;
    io::Write::flush(&mut points)
}

/// Encode a linear capture as ordinary sRGB images.
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

/// Encode calibrated foreground coverage as grayscale images.
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

/// Build an explicit nearest-camera PatchMatch graph from construction views.
pub fn patch_match_config(
    capture: &train::inverse::capture::Capture,
    views: &[usize],
    view_name: fn(usize) -> String,
    source_count: usize,
) -> Result<String, String> {
    if source_count == 0 || views.len() < 2 {
        return Err("PatchMatch needs at least two views and one source".to_string());
    }
    if views
        .iter()
        .enumerate()
        .any(|(offset, index)| *index >= capture.views.len() || views[..offset].contains(index))
    {
        return Err("PatchMatch view indices must be unique and in range".to_string());
    }
    let mut graph = String::new();
    for &index in views {
        let position = glam::Vec3::from(capture.views[index].camera.cam_position);
        let mut sources = views
            .iter()
            .copied()
            .filter(|&other| other != index)
            .map(|other| {
                (
                    position.distance_squared(glam::Vec3::from(
                        capture.views[other].camera.cam_position,
                    )),
                    other,
                )
            })
            .collect::<Vec<_>>();
        sources.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        graph.push_str(&view_name(index));
        graph.push('\n');
        graph.push_str(
            &sources
                .iter()
                .take(source_count)
                .map(|&(_, source)| view_name(source))
                .collect::<Vec<_>>()
                .join(", "),
        );
        graph.push('\n');
    }
    Ok(graph)
}

fn print_render_summaries(label: &str, summaries: &[train::inverse::score::Summary]) {
    let views = summaries.iter().map(|summary| summary.views).sum::<usize>();
    let mean = |get: fn(&train::inverse::score::Summary) -> f64| {
        summaries
            .iter()
            .map(|summary| get(summary) * summary.views as f64)
            .sum::<f64>()
            / views.max(1) as f64
    };
    let worst = summaries
        .iter()
        .map(|summary| summary.worst_srgb_psnr)
        .fold(f64::INFINITY, f64::min);
    let foreground_worst = summaries
        .iter()
        .filter_map(|summary| summary.worst_foreground_srgb_psnr)
        .fold(f64::INFINITY, f64::min);
    println!(
        "{label}: {views} views, {:.2} dB linear, {:.2}/{worst:.2} dB sRGB mean/worst, \
         {:.2}/{foreground_worst:.2} dB foreground, {:.1}% recall, {:.1}% precision, \
         {:.2} dB where-hit",
        mean(|summary| summary.linear_psnr),
        mean(|summary| summary.srgb_psnr),
        mean(|summary| summary.foreground_srgb_psnr.unwrap_or(0.0)),
        100.0 * mean(|summary| summary.mask_recall.unwrap_or(0.0)),
        100.0 * mean(|summary| summary.mask_precision.unwrap_or(0.0)),
        mean(|summary| summary.covered_srgb_psnr),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn render_dataset_cross(
    renderer: &mut train::inverse::score::Renderer,
    scene: &train::inverse::score::Scene,
    gaussian: Option<&vol::PointCloudModel>,
    captures: &[train::inverse::capture::Capture],
    lights: &[Vec<vol::relight::PointLight>],
    source_light_indices: &[usize],
    train_views: &[usize],
    held_views: &[usize],
    light_label: &str,
    light_digits: usize,
    point_light_visibility: bool,
    dump: Option<&path::Path>,
) -> Result<(), String> {
    let mut surface_fitted = Vec::with_capacity(captures.len());
    let mut surface_held = Vec::with_capacity(captures.len());
    let mut gaussian_fitted = Vec::with_capacity(captures.len());
    let mut gaussian_held = Vec::with_capacity(captures.len());
    for ((capture, lights), &source_light) in captures.iter().zip(lights).zip(source_light_indices)
    {
        let surface_dump = dump.map(|directory| {
            directory.join("surface").join(format!(
                "light-{:0width$}",
                source_light + 1,
                width = light_digits,
            ))
        });
        if let Some(ref directory) = surface_dump {
            fs::create_dir_all(directory)
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        let summaries = renderer.score_point_light_splits(
            scene,
            capture,
            lights,
            &[(train_views, None), (held_views, surface_dump.as_deref())],
            point_light_visibility,
        );
        surface_fitted.push(summaries[0]);
        surface_held.push(summaries[1]);

        if let Some(gaussian) = gaussian {
            let gaussian_dump = dump.map(|directory| {
                directory.join("gaussian").join(format!(
                    "light-{:0width$}",
                    source_light + 1,
                    width = light_digits,
                ))
            });
            if let Some(ref directory) = gaussian_dump {
                fs::create_dir_all(directory)
                    .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
            }
            let summaries = renderer.score_gaussian_point_light_splits(
                scene,
                gaussian,
                capture,
                lights,
                &[(train_views, None), (held_views, gaussian_dump.as_deref())],
                point_light_visibility,
            );
            gaussian_fitted.push(summaries[0]);
            gaussian_held.push(summaries[1]);
        }
    }
    print_render_summaries(
        &format!("surface {light_label} lights / fitted cameras"),
        &surface_fitted,
    );
    print_render_summaries(
        &format!("surface {light_label} lights / held cameras"),
        &surface_held,
    );
    if gaussian.is_some() {
        print_render_summaries(
            &format!("Gaussian {light_label} lights / fitted cameras"),
            &gaussian_fitted,
        );
        print_render_summaries(
            &format!("Gaussian {light_label} lights / held cameras"),
            &gaussian_held,
        );
    }
    Ok(())
}

pub fn refine_surface(
    surface: &mut vol::relight::RelightModel,
    captures: &[train::inverse::capture::Capture],
    lights: &[Vec<vol::relight::PointLight>],
    views: &[usize],
    rounds: usize,
    normal_candidates: usize,
    albedo_ceiling: f32,
) {
    for round in 0..rounds {
        let observations = captures
            .iter()
            .map(|capture| train::inverse::decompose::observe(surface, capture, views, 0.15))
            .collect::<Vec<_>>();
        let known = observations
            .iter()
            .zip(lights)
            .map(
                |(observations, lights)| train::inverse::decompose::KnownLightObservations {
                    light: train::inverse::decompose::CalibratedLight::Near(lights),
                    observations,
                },
            )
            .collect::<Vec<_>>();
        let normals = train::inverse::decompose::refine_normals_known_lights(
            surface,
            &known,
            normal_candidates,
        );
        let materials = train::inverse::decompose::refine_materials_known_lights(
            surface,
            &known,
            albedo_ceiling,
        );
        println!(
            "round {}: normals {}/{} changed, materials {} supported ({} channel updates)",
            round + 1,
            normals.changed,
            normals.supported,
            materials.supported,
            materials.changed,
        );
    }
}

fn refine_blended_surface(
    surface: &mut vol::relight::RelightModel,
    captures: &[train::inverse::capture::Capture],
    lights: &[Vec<vol::relight::PointLight>],
    views: &[usize],
    albedo_ceiling: f32,
) -> Result<(), String> {
    let blended = train::inverse::blended::refine_materials_point_lights(
        surface,
        captures,
        lights,
        views,
        albedo_ceiling,
    )?;
    println!(
        "blended materials: {}/{} supported, {} changed, {} equations / {} terms, linear RMSE {:.6} -> {:.6}",
        blended.supported,
        surface.materials.len(),
        blended.changed,
        blended.equations,
        blended.terms,
        blended.initial_loss.sqrt(),
        blended.final_loss.sqrt(),
    );
    Ok(())
}

pub fn save_gaussian(path: &path::Path, model: &vol::PointCloudModel) -> Result<(), String> {
    convert::save_ply_with_options(
        path,
        model,
        &convert::SaveOptions {
            format: convert::PlyFormat::Binary,
        },
    )
    .map_err(|error| format!("cannot write {}: {error:?}", path.display()))
}

pub fn create_output_parent(file: &path::Path) -> Result<(), String> {
    let Some(parent) = file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    fs::create_dir_all(parent)
        .map_err(|error| format!("cannot create {}: {error}", parent.display()))
}

pub fn fit(
    load_training: impl FnOnce() -> Result<Dataset, String>,
    load_held: impl FnOnce() -> Result<Dataset, String>,
    options: FitOptions<'_>,
) -> Result<(), String> {
    if options.width == 0 || options.rounds == 0 || options.normal_candidates == 0 {
        return Err("width, rounds, and normal candidates must be non-zero".to_string());
    }
    if !options.albedo_ceiling.is_finite() || options.albedo_ceiling <= 0.0 {
        return Err("albedo ceiling must be finite and positive".to_string());
    }
    create_output_parent(options.output)?;
    if let Some(output) = options.gaussian_output {
        create_output_parent(output)?;
    }
    let mut surface = vol::io::try_load_relight(options.surface)
        .map_err(|error| format!("cannot read {}: {error}", options.surface.display()))?;
    for material in &mut surface.materials {
        material.roughness = 1.0;
        material.specular_f0 = [0.0; 3];
    }
    let training = load_training()?;

    refine_surface(
        &mut surface,
        &training.captures,
        &training.lights,
        options.train_views,
        options.rounds,
        options.normal_candidates,
        options.albedo_ceiling,
    );
    let gaussian = if options.gaussian_output.is_some() {
        let mut gaussian =
            train::gaussian_splat::pbr_from_surface(&surface, options.train_views.len())?;
        let established = gaussian.clone();
        let lights = training
            .captures
            .iter()
            .zip(&training.lights)
            .map(
                |(capture, lights)| train::gaussian_splat::KnownLightCapture {
                    capture,
                    light: train::gaussian_splat::KnownLight::Near(lights),
                },
            )
            .collect::<Vec<_>>();
        let gpu = train::fit::try_init_gpu()
            .ok_or_else(|| "cannot initialize a supported GPU".to_string())?;
        let stats = train::gaussian_splat::fit_multilight_geometry(
            &mut gaussian,
            &mut surface,
            &lights,
            options.train_views,
            gpu,
        )?;
        let regression_restored = stats
            .last()
            .is_some_and(|stats| stats.final_loss > stats.initial_loss);
        let support = train::gaussian_splat::guard_pbr_support(&mut gaussian, &established)?;
        println!(
            "Gaussian: {} updates, audit loss {:.6} -> {:.6}, retained {}/{}{}",
            stats.iter().map(|stats| stats.steps).sum::<usize>(),
            stats.first().map_or(f32::NAN, |stats| stats.initial_loss),
            stats.last().map_or(f32::NAN, |stats| stats.final_loss),
            support.retained,
            support.particles,
            if regression_restored {
                " (regression restored)"
            } else if support.restored {
                " (support restored)"
            } else {
                ""
            },
        );
        refine_surface(
            &mut surface,
            &training.captures,
            &training.lights,
            options.train_views,
            1,
            options.normal_candidates,
            options.albedo_ceiling,
        );
        refine_blended_surface(
            &mut surface,
            &training.captures,
            &training.lights,
            options.train_views,
            options.albedo_ceiling,
        )?;
        train::gaussian_splat::attach_pbr(&mut gaussian, &surface)?;
        Some(gaussian)
    } else {
        refine_blended_surface(
            &mut surface,
            &training.captures,
            &training.lights,
            options.train_views,
            options.albedo_ceiling,
        )?;
        None
    };

    vol::io::try_save_relight(options.output, &surface)
        .map_err(|error| format!("cannot write {}: {error}", options.output.display()))?;
    println!("wrote {}", options.output.display());
    if let (Some(output), Some(model)) = (options.gaussian_output, gaussian.as_ref()) {
        save_gaussian(output, model)?;
        println!("wrote {}", output.display());
    }

    let black = vol::relight::Environment::uniform([0.0; 3], 8, 4);
    let scene = train::inverse::score::Scene::new(surface, black);
    let mut renderer =
        train::inverse::score::Renderer::new(options.width, training.captures[0].height)?;
    render_dataset_cross(
        &mut renderer,
        &scene,
        gaussian.as_ref(),
        &training.captures,
        &training.lights,
        &training.source_light_indices,
        options.train_views,
        options.held_views,
        "fitted",
        options.light_digits,
        options.point_light_visibility,
        None,
    )?;
    drop(training);
    let held = load_held()?;
    render_dataset_cross(
        &mut renderer,
        &scene,
        gaussian.as_ref(),
        &held.captures,
        &held.lights,
        &held.source_light_indices,
        options.train_views,
        options.held_views,
        "held",
        options.light_digits,
        options.point_light_visibility,
        options.dump,
    )?;
    renderer.destroy();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(
        position: [f32; 3],
        fov: [f32; 2],
        principal: [f32; 2],
    ) -> train::inverse::capture::View {
        train::inverse::capture::View {
            name: String::new(),
            camera: vol::CameraParams {
                cam_position: position,
                cam_orientation: glam::Quat::IDENTITY.to_array(),
                depth: 10.0,
                fov,
                principal,
            },
            pixels: vec![[0.0; 3]],
            mask: Some(vec![1.0]),
        }
    }

    fn view_name(index: usize) -> String {
        format!("view-{index}.png")
    }

    #[test]
    fn colmap_writer_preserves_per_view_intrinsics_and_poses() {
        let temporary =
            std::env::temp_dir().join(format!("blade-volume-calibrated-{}", std::process::id(),));
        let _ = fs::remove_dir_all(&temporary);
        let capture = train::inverse::capture::Capture {
            width: 100,
            height: 80,
            views: vec![
                view([0.0, 0.0, 0.0], [1.0, 0.8], [0.0, 0.0]),
                view([1.0, 2.0, 3.0], [0.5, 0.4], [0.2, -0.1]),
            ],
        };

        write_colmap(&temporary, &capture, view_name).unwrap();
        let reconstruction = train::colmap::try_load_reconstruction(&temporary).unwrap();

        assert_eq!(reconstruction.cameras.len(), 2);
        assert_eq!(reconstruction.images.len(), 2);
        assert_eq!(reconstruction.images[0].camera_id, 1);
        assert_eq!(reconstruction.images[1].camera_id, 2);
        assert_ne!(
            reconstruction.cameras[&1].params,
            reconstruction.cameras[&2].params
        );
        assert!((reconstruction.cameras[&2].params[2] - 60.0).abs() < 1.0e-5);
        assert!((reconstruction.cameras[&2].params[3] - 36.0).abs() < 1.0e-5);
        assert_eq!(reconstruction.images[1].translation, [-1.0, -2.0, -3.0]);
        fs::remove_dir_all(temporary).unwrap();
    }

    #[test]
    fn patch_match_uses_nearest_selected_views_only() {
        let capture = train::inverse::capture::Capture {
            width: 1,
            height: 1,
            views: vec![
                view([0.0, 0.0, 0.0], [1.0; 2], [0.0; 2]),
                view([10.0, 0.0, 0.0], [1.0; 2], [0.0; 2]),
                view([2.0, 0.0, 0.0], [1.0; 2], [0.0; 2]),
                view([1.0, 0.0, 0.0], [1.0; 2], [0.0; 2]),
            ],
        };

        let graph = patch_match_config(&capture, &[0, 1, 2], view_name, 2).unwrap();

        assert_eq!(
            graph,
            "view-0.png\nview-2.png, view-1.png\n\
             view-1.png\nview-2.png, view-0.png\n\
             view-2.png\nview-0.png, view-1.png\n"
        );
        assert!(!graph.contains("view-3"));
        assert!(patch_match_config(&capture, &[0, 0], view_name, 1).is_err());
        assert!(patch_match_config(&capture, &[0, 4], view_name, 1).is_err());
    }
}
