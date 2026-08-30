//! Shared mechanics for calibrated multi-view/multi-light fitting binaries.

use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train as train;
use std::{fs, path};

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
    pub train_views: &'a [usize],
    pub held_views: &'a [usize],
    pub light_digits: usize,
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
            &training.captures,
            &training.lights,
            options.train_views,
            1,
            options.normal_candidates,
            options.albedo_ceiling,
        );
        train::gaussian_splat::attach_pbr(&mut gaussian, &surface)?;
        refine_blended_surface(
            &mut surface,
            &training.captures,
            &training.lights,
            options.train_views,
            options.albedo_ceiling,
        )?;
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
        options.dump,
    )?;
    renderer.destroy();
    Ok(())
}
