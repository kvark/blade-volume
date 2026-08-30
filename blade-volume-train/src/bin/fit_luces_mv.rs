//! Fit calibrated near-field surface properties on a reconstructed LUCES-MV cloud.
//!
//! Training uses the fixed 9-camera/12-light split. The three excluded LEDs
//! are not loaded until the fitted scalar and Gaussian point clouds have been
//! serialized, keeping the held-light cross-product out of model selection.

use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train as train;
use std::{fs, path};

#[derive(argh::FromArgs)]
/// Fit LUCES-MV normals, diffuse materials, and optional Gaussian geometry.
struct Args {
    /// object directory containing the calibrated views
    #[argh(option)]
    input: String,

    /// first official camera parameter text file
    #[argh(option)]
    camera_one: String,

    /// second official camera parameter text file
    #[argh(option)]
    camera_two: String,

    /// reconstructed relightable surfel cloud
    #[argh(option)]
    surface: String,

    /// fitted relightable surfel output
    #[argh(option)]
    output: String,

    /// optional fitted relightable Gaussian output
    #[argh(option)]
    gaussian_output: Option<String>,

    /// optional directory for production held-light/held-camera renders
    #[argh(option)]
    dump: Option<String>,

    /// fitting image width (default 128; height preserves aspect)
    #[argh(option, default = "128")]
    width: usize,

    /// alternating normal/material rounds (default 3)
    #[argh(option, default = "3")]
    rounds: usize,

    /// hemisphere candidates tested per normal update (default 1024)
    #[argh(option, default = "1024")]
    normal_candidates: usize,

    /// maximum diffuse albedo (default 1)
    #[argh(option, default = "1.0")]
    albedo_ceiling: f32,
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

fn render_dataset_cross(
    renderer: &mut train::inverse::score::Renderer,
    scene: &train::inverse::score::Scene,
    gaussian: Option<&vol::PointCloudModel>,
    dataset: &train::inverse::luces::Dataset,
    light_label: &str,
    dump: Option<&path::Path>,
) -> Result<(), String> {
    let mut surface_fitted = Vec::with_capacity(dataset.captures.len());
    let mut surface_held = Vec::with_capacity(dataset.captures.len());
    let mut gaussian_fitted = Vec::with_capacity(dataset.captures.len());
    let mut gaussian_held = Vec::with_capacity(dataset.captures.len());
    for ((capture, lights), &source_light) in dataset
        .captures
        .iter()
        .zip(&dataset.lights)
        .zip(&dataset.source_light_indices)
    {
        let surface_dump = dump.map(|directory| {
            directory
                .join("surface")
                .join(format!("light-{:02}", source_light + 1))
        });
        if let Some(ref directory) = surface_dump {
            fs::create_dir_all(directory)
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        let summaries = renderer.score_point_light_splits(
            scene,
            capture,
            lights,
            &[
                (&train::inverse::luces::TRAIN_VIEW_INDICES, None),
                (
                    &train::inverse::luces::HELD_VIEW_INDICES,
                    surface_dump.as_deref(),
                ),
            ],
        );
        surface_fitted.push(summaries[0]);
        surface_held.push(summaries[1]);

        if let Some(gaussian) = gaussian {
            let gaussian_dump = dump.map(|directory| {
                directory
                    .join("gaussian")
                    .join(format!("light-{:02}", source_light + 1))
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
                &[
                    (&train::inverse::luces::TRAIN_VIEW_INDICES, None),
                    (
                        &train::inverse::luces::HELD_VIEW_INDICES,
                        gaussian_dump.as_deref(),
                    ),
                ],
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

fn refine_surface(
    surface: &mut vol::relight::RelightModel,
    dataset: &train::inverse::luces::Dataset,
    views: &[usize],
    rounds: usize,
    normal_candidates: usize,
    albedo_ceiling: f32,
) {
    for round in 0..rounds {
        let observations = dataset
            .captures
            .iter()
            .map(|capture| train::inverse::decompose::observe(surface, capture, views, 0.15))
            .collect::<Vec<_>>();
        let known = observations
            .iter()
            .zip(&dataset.lights)
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

fn save_gaussian(path: &path::Path, model: &vol::PointCloudModel) -> Result<(), String> {
    convert::save_ply_with_options(
        path,
        model,
        &convert::SaveOptions {
            format: convert::PlyFormat::Binary,
        },
    )
    .map_err(|error| format!("cannot write {}: {error:?}", path.display()))
}

fn create_output_parent(file: &path::Path) -> Result<(), String> {
    let Some(parent) = file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    fs::create_dir_all(parent)
        .map_err(|error| format!("cannot create {}: {error}", parent.display()))
}

fn run(args: &Args) -> Result<(), String> {
    if args.width == 0 || args.rounds == 0 || args.normal_candidates == 0 {
        return Err("width, rounds, and normal candidates must be non-zero".to_string());
    }
    if !args.albedo_ceiling.is_finite() || args.albedo_ceiling <= 0.0 {
        return Err("albedo ceiling must be finite and positive".to_string());
    }
    let input = path::Path::new(&args.input);
    let camera_one = path::Path::new(&args.camera_one);
    let camera_two = path::Path::new(&args.camera_two);
    create_output_parent(path::Path::new(&args.output))?;
    if let Some(ref output) = args.gaussian_output {
        create_output_parent(path::Path::new(output))?;
    }
    let mut surface = vol::io::try_load_relight(path::Path::new(&args.surface))
        .map_err(|error| format!("cannot read {}: {error}", args.surface))?;
    let training = train::inverse::luces::load(
        input,
        camera_one,
        camera_two,
        args.width,
        &train::inverse::luces::TRAIN_LIGHT_INDICES,
    )?;

    refine_surface(
        &mut surface,
        &training,
        &train::inverse::luces::TRAIN_VIEW_INDICES,
        args.rounds,
        args.normal_candidates,
        args.albedo_ceiling,
    );
    let gaussian = if args.gaussian_output.is_some() {
        let mut gaussian = train::gaussian_splat::pbr_from_surface(
            &surface,
            train::inverse::luces::TRAIN_VIEW_INDICES.len(),
        )?;
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
            &train::inverse::luces::TRAIN_VIEW_INDICES,
            gpu,
        )?;
        let support = train::gaussian_splat::guard_pbr_support(&mut gaussian, &established)?;
        println!(
            "Gaussian: {} updates, loss {:.6} -> {:.6}, retained {}/{}{}",
            stats.iter().map(|stats| stats.steps).sum::<usize>(),
            stats.first().map_or(f32::NAN, |stats| stats.initial_loss),
            stats.last().map_or(f32::NAN, |stats| stats.final_loss),
            support.retained,
            support.particles,
            if support.restored {
                " (support restored)"
            } else {
                ""
            },
        );
        refine_surface(
            &mut surface,
            &training,
            &train::inverse::luces::TRAIN_VIEW_INDICES,
            1,
            args.normal_candidates,
            args.albedo_ceiling,
        );
        train::gaussian_splat::attach_pbr(&mut gaussian, &surface)?;
        Some(gaussian)
    } else {
        None
    };

    vol::io::try_save_relight(path::Path::new(&args.output), &surface)
        .map_err(|error| format!("cannot write {}: {error}", args.output))?;
    println!("wrote {}", args.output);
    if let (Some(output), Some(model)) = (args.gaussian_output.as_deref(), gaussian.as_ref()) {
        save_gaussian(path::Path::new(output), model)?;
        println!("wrote {output}");
    }

    let black = vol::relight::Environment::uniform([0.0; 3], 8, 4);
    let scene = train::inverse::score::Scene::new(surface, black);
    let mut renderer =
        train::inverse::score::Renderer::new(args.width, training.captures[0].height)?;
    render_dataset_cross(
        &mut renderer,
        &scene,
        gaussian.as_ref(),
        &training,
        "fitted",
        None,
    )?;
    drop(training);
    let held = train::inverse::luces::load(
        input,
        camera_one,
        camera_two,
        args.width,
        &train::inverse::luces::HELD_LIGHT_INDICES,
    )?;
    render_dataset_cross(
        &mut renderer,
        &scene,
        gaussian.as_ref(),
        &held,
        "held",
        args.dump.as_deref().map(path::Path::new),
    )?;
    renderer.destroy();
    Ok(())
}

fn main() {
    let args: Args = argh::from_env();
    if let Err(message) = run(&args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
