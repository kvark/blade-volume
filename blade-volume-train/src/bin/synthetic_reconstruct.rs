//! Reconstruct and score a Gaussian PBR point cloud from Blade synthetic data.
//!
//! This is deliberately the geometry upper bound, not an RGB-only claim. It
//! fuses depth and normal truth from training cameras, never from held-out
//! cameras, then fits PBR materials only to selected training illumination.
//! The resulting point cloud is scored at unseen poses under both a seen light
//! and an unseen light. An oracle-material cloud with identical fused geometry
//! separates representation/shading error from material-fitting error.

use blade_volume as vol;
use blade_volume_train as train;
use std::{collections, path, time};

#[derive(argh::FromArgs)]
/// Reconstruct a Gaussian PBR point cloud from a Blade relighting dataset.
struct Args {
    /// directory written by Blade's relight_data test
    #[argh(option)]
    dataset: String,

    /// comma-separated training environments (default: first non-uniform light)
    #[argh(option)]
    train_environments: Option<String>,

    /// environment reserved for relighting evaluation (default: last)
    #[argh(option)]
    held_out_environment: Option<String>,

    /// reserve every nth camera for novel-pose evaluation
    #[argh(option, default = "4")]
    held_out_stride: usize,

    /// residue within the stride to reserve
    #[argh(option, default = "1")]
    held_out_offset: usize,

    /// world-space side length used to fuse training samples
    #[argh(option, default = "0.08")]
    voxel: f32,

    /// particle support radius divided by voxel size
    #[argh(option, default = "1.7")]
    radius_factor: f32,

    /// joint material-fit iterations per fused element
    #[argh(option, default = "120")]
    iterations: usize,

    /// maximum shared fitted materials, zero to keep one per particle
    #[argh(option, default = "64")]
    material_count: usize,

    /// deterministic material-palette clustering iterations
    #[argh(option, default = "8")]
    material_iterations: usize,

    /// direct-light visibility samples per rendered particle
    #[argh(option, default = "0")]
    samples: u32,

    /// use the legacy compact point kernel instead of a Gaussian
    #[argh(switch)]
    compact_kernel: bool,

    /// write the fitted point cloud to this .rply or .surfel file
    #[argh(option)]
    output: Option<String>,

    /// write held-out-pose comparisons into this directory
    #[argh(option)]
    dump: Option<String>,
}

#[derive(Default)]
struct ReconstructionError {
    position_rmse: f64,
    normal_rmse_degrees: f64,
    albedo_rmse: f64,
    specular_rmse: f64,
    roughness_rmse: f64,
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

fn environment_index(dataset: &train::relight::Dataset, name: &str) -> usize {
    dataset
        .environments
        .iter()
        .position(|candidate| candidate == name)
        .unwrap_or_else(|| fail(format!("no environment named '{name}'")))
}

fn training_environments(
    dataset: &train::relight::Dataset,
    specification: Option<&str>,
    held_out: usize,
) -> Vec<usize> {
    let indices = match specification {
        Some(names) => names
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| environment_index(dataset, name))
            .collect(),
        None => vec![dataset
            .environments
            .iter()
            .enumerate()
            .find(|&(index, name)| index != held_out && name != "uniform")
            .or_else(|| {
                dataset
                    .environments
                    .iter()
                    .enumerate()
                    .find(|&(index, _)| index != held_out)
            })
            .map_or_else(
                || fail("the dataset needs separate training and held-out environments"),
                |(index, _)| index,
            )],
    };
    if indices.is_empty() {
        fail("at least one training environment is required");
    }
    if indices.contains(&held_out) {
        fail("the held-out environment cannot also supervise material fitting");
    }
    indices
}

fn split_views(count: usize, stride: usize, offset: usize) -> (Vec<usize>, Vec<usize>) {
    if stride < 2 {
        fail("held-out stride must be at least two");
    }
    let residue = offset % stride;
    let mut training = Vec::new();
    let mut held_out = Vec::new();
    for index in 0..count {
        if index % stride == residue {
            held_out.push(index);
        } else {
            training.push(index);
        }
    }
    if training.is_empty() || held_out.is_empty() {
        fail(format!(
            "the {count}-view dataset and stride {stride}:{residue} do not produce both splits"
        ));
    }
    (training, held_out)
}

fn foreground_coverage(
    dataset: &train::relight::Dataset,
    indices: &[usize],
) -> Result<f64, String> {
    let mut foreground = 0usize;
    for &index in indices {
        let view = dataset
            .views
            .get(index)
            .ok_or_else(|| format!("no view {index}"))?;
        let geometry = dataset.read_plane(&view.geometry)?;
        foreground += geometry.iter().filter(|texel| texel[3] > 0.0).count();
    }
    Ok(foreground as f64 / (indices.len() * dataset.pixel_count()) as f64)
}

fn reconstruction_error(
    elements: &[train::relight::Element],
    samples: &[train::relight::Sample],
    palette: Option<&train::relight::MaterialPalette>,
) -> ReconstructionError {
    let mut position_squared = 0.0f64;
    let mut normal_squared = 0.0f64;
    let mut albedo_squared = 0.0f64;
    let mut specular_squared = 0.0f64;
    let mut roughness_squared = 0.0f64;
    let mut observations = 0usize;

    for (element_index, element) in elements.iter().enumerate() {
        let mut center = glam::Vec3::ZERO;
        let mut normal = glam::Vec3::ZERO;
        for &index in &element.samples {
            center += samples[index as usize].position;
            normal += samples[index as usize].normal;
        }
        let inverse_count = 1.0 / element.samples.len().max(1) as f32;
        center *= inverse_count;
        normal = normal.normalize_or_zero();
        let (albedo, specular_f0, roughness) = match palette {
            Some(palette) => {
                let material = palette.materials[palette.assignments[element_index] as usize];
                (material.albedo, material.specular_f0, material.roughness)
            }
            None => (element.albedo, element.specular_f0, element.roughness),
        };
        for &index in &element.samples {
            let sample = &samples[index as usize];
            position_squared += (center - sample.position).length_squared() as f64;
            let angle = normal.dot(sample.normal).clamp(-1.0, 1.0).acos();
            normal_squared += angle.to_degrees().powi(2) as f64;
            for channel in 0..3 {
                albedo_squared += (albedo[channel] - sample.albedo_truth[channel]).powi(2) as f64;
                specular_squared +=
                    (specular_f0[channel] - sample.specular_f0[channel]).powi(2) as f64;
            }
            roughness_squared += (roughness - sample.roughness).powi(2) as f64;
            observations += 1;
        }
    }
    ReconstructionError {
        position_rmse: (position_squared / observations.max(1) as f64).sqrt(),
        normal_rmse_degrees: (normal_squared / observations.max(1) as f64).sqrt(),
        albedo_rmse: (albedo_squared / (3 * observations).max(1) as f64).sqrt(),
        specular_rmse: (specular_squared / (3 * observations).max(1) as f64).sqrt(),
        roughness_rmse: (roughness_squared / observations.max(1) as f64).sqrt(),
    }
}

fn view_support(
    elements: &[train::relight::Element],
    samples: &[train::relight::Sample],
) -> (f64, f64) {
    let mut total = 0usize;
    let mut multiple = 0usize;
    for element in elements {
        let views: collections::HashSet<usize> = element
            .samples
            .iter()
            .map(|&index| samples[index as usize].view_index)
            .collect();
        total += views.len();
        multiple += (views.len() > 1) as usize;
    }
    (
        total as f64 / elements.len().max(1) as f64,
        multiple as f64 / elements.len().max(1) as f64,
    )
}

fn score_model(
    renderer: &mut train::inverse::score::Renderer,
    model: &vol::relight::RelightModel,
    environment: &vol::relight::Environment,
    capture: &train::inverse::capture::Capture,
    training_views: &[usize],
    held_out_views: &[usize],
    samples: u32,
    dump: Option<&path::Path>,
) -> Vec<train::inverse::score::Summary> {
    if let Some(directory) = dump {
        std::fs::create_dir_all(directory).unwrap();
    }
    renderer.score_splits(
        &train::inverse::score::Scene {
            model: model.clone(),
            environment: environment.clone(),
        },
        capture,
        &[(training_views, None), (held_out_views, dump)],
        samples,
    )
}

fn print_score(label: &str, summary: train::inverse::score::Summary) {
    println!(
        "{label:<34}{:>8.2}{:>9.2}{:>10.1}%{:>10.2}{:>10.2}",
        summary.srgb_psnr,
        summary.worst_srgb_psnr,
        100.0 * summary.coverage,
        summary.covered_srgb_psnr,
        summary.render_ms,
    );
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();
    let root = path::Path::new(&args.dataset);
    let dataset = train::relight::Dataset::load(root).unwrap_or_else(|error| fail(error));
    let held_out_environment = args
        .held_out_environment
        .as_deref()
        .map_or(dataset.environments.len() - 1, |name| {
            environment_index(&dataset, name)
        });
    let training_environments = training_environments(
        &dataset,
        args.train_environments.as_deref(),
        held_out_environment,
    );
    let (training_views, held_out_views) = split_views(
        dataset.views.len(),
        args.held_out_stride,
        args.held_out_offset,
    );
    println!(
        "{} views x {} environments at {}x{}",
        dataset.views.len(),
        dataset.environments.len(),
        dataset.width,
        dataset.height
    );
    println!(
        "training views {:?}; held-out views {:?}",
        training_views, held_out_views
    );
    println!(
        "training light {:?}; held-out light '{}'",
        training_environments
            .iter()
            .map(|&index| dataset.environments[index].as_str())
            .collect::<Vec<_>>(),
        dataset.environments[held_out_environment]
    );

    let started = time::Instant::now();
    let samples = train::relight::gather_samples_for_views(&dataset, &training_views)
        .unwrap_or_else(|error| fail(error));
    let sample_time = started.elapsed();
    let mut elements = train::relight::build_elements(&samples, args.voxel);
    let (mean_views, multi_view) = view_support(&elements, &samples);
    let training_coverage =
        foreground_coverage(&dataset, &training_views).unwrap_or_else(|error| fail(error));
    let held_out_coverage =
        foreground_coverage(&dataset, &held_out_views).unwrap_or_else(|error| fail(error));
    println!(
        "{} training foreground samples ({:.1}% of pixels) fused into {} elements in {:.3} s",
        samples.len(),
        100.0 * training_coverage,
        elements.len(),
        sample_time.as_secs_f64()
    );
    println!(
        "{mean_views:.2} distinct training views per element; {:.1}% multi-view; held-out truth covers {:.1}%",
        100.0 * multi_view,
        100.0 * held_out_coverage
    );

    let irradiance = dataset
        .environment_irradiance()
        .unwrap_or_else(|error| fail(error));
    let prefilter_started = time::Instant::now();
    let mut needed_environments = training_environments.clone();
    needed_environments.push(held_out_environment);
    let specular = dataset
        .environment_specular_for(&needed_environments)
        .unwrap_or_else(|error| fail(error));
    println!(
        "prefiltered {} material-fit environments in {:.3} s",
        needed_environments.len(),
        prefilter_started.elapsed().as_secs_f64()
    );
    let fit_started = time::Instant::now();
    let loss = train::relight::optimize_elements(
        &mut elements,
        &samples,
        &training_environments,
        &irradiance,
        &specular,
        train::relight::JointConfig {
            iterations: args.iterations,
            ..Default::default()
        },
    );
    let fit_time = fit_started.elapsed();
    let element_error = reconstruction_error(&elements, &samples, None);
    println!(
        "material fit in {:.3} s: loss {:.6}, albedo {:.4}, F0 {:.4}, roughness {:.4} RMSE",
        fit_time.as_secs_f64(),
        loss.last().copied().unwrap_or(f64::NAN),
        element_error.albedo_rmse,
        element_error.specular_rmse,
        element_error.roughness_rmse,
    );
    println!(
        "training-view fusion: position {:.5} world-unit RMSE, normal {:.3} degree RMSE",
        element_error.position_rmse, element_error.normal_rmse_degrees
    );

    let palette_started = time::Instant::now();
    let palette = (args.material_count > 0).then(|| {
        train::relight::fit_material_palette(
            &elements,
            args.material_count,
            args.material_iterations,
        )
    });
    if let Some(ref palette) = palette {
        let palette_error = reconstruction_error(&elements, &samples, Some(palette));
        println!(
            "clustered to {} shared materials in {:.3} s: albedo {:.4}, F0 {:.4}, roughness {:.4} RMSE",
            palette.materials.len(),
            palette_started.elapsed().as_secs_f64(),
            palette_error.albedo_rmse,
            palette_error.specular_rmse,
            palette_error.roughness_rmse,
        );
    }

    let kernel = if args.compact_kernel {
        vol::relight::ParticleKernel::Compact
    } else {
        vol::relight::ParticleKernel::Gaussian
    };
    let radius = args.voxel * args.radius_factor;
    let fitted = match palette {
        Some(ref palette) => train::relight::model_from_elements_with_palette(
            &elements, &samples, radius, kernel, palette,
        ),
        None => train::relight::model_from_elements(
            &elements,
            &samples,
            radius,
            kernel,
            train::relight::ElementMaterial::Fitted,
        ),
    };
    let oracle = train::relight::model_from_elements(
        &elements,
        &samples,
        radius,
        kernel,
        train::relight::ElementMaterial::Truth,
    );
    fitted.validate().unwrap_or_else(|error| fail(error));
    oracle.validate().unwrap_or_else(|error| fail(error));
    println!(
        "{} {:?} particles, radius {:.4}, {} fitted materials",
        fitted.surfels.len(),
        fitted.kernel,
        radius,
        fitted.materials.len()
    );
    if let Some(ref output) = args.output {
        let output_path = path::Path::new(output);
        if let Some(parent) = output_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).unwrap_or_else(|error| {
                fail(format!("cannot create {}: {error}", parent.display()))
            });
        }
        vol::io::try_save_relight(output_path, &fitted)
            .unwrap_or_else(|error| fail(format!("cannot write {output}: {error}")));
        println!("wrote {output}");
    }

    let training_environment =
        vol::io::try_load_environment(&dataset.environment_files[training_environments[0]])
            .unwrap_or_else(|error| fail(error));
    let relight_environment =
        vol::io::try_load_environment(&dataset.environment_files[held_out_environment])
            .unwrap_or_else(|error| fail(error));
    let training_capture = train::inverse::capture::Capture::from_relight_dataset(
        &dataset,
        training_environments[0],
        true,
    )
    .unwrap_or_else(|error| fail(error));
    let relight_capture = train::inverse::capture::Capture::from_relight_dataset(
        &dataset,
        held_out_environment,
        true,
    )
    .unwrap_or_else(|error| fail(error));

    let dump_root = args.dump.as_deref().map(path::Path::new);
    let mut renderer = train::inverse::score::Renderer::new(dataset.width, dataset.height)
        .unwrap_or_else(|error| fail(error));
    println!("GPU: {}", renderer.device_name());
    println!(
        "\n{:<34}{:>8}{:>9}{:>11}{:>10}{:>10}",
        "model/light/poses", "PSNR", "worst", "coverage", "on-hit", "ms/frame"
    );
    for (model_name, model) in [("fitted", &fitted), ("oracle material", &oracle)] {
        for (light_name, environment, capture) in [
            ("training light", &training_environment, &training_capture),
            ("held-out light", &relight_environment, &relight_capture),
        ] {
            let directory = dump_root.map(|root| {
                root.join(format!(
                    "{}-{}",
                    model_name.replace(' ', "-"),
                    light_name.replace(' ', "-")
                ))
            });
            let summaries = score_model(
                &mut renderer,
                model,
                environment,
                capture,
                &training_views,
                &held_out_views,
                args.samples,
                directory.as_deref(),
            );
            print_score(&format!("{model_name}, {light_name}, train"), summaries[0]);
            print_score(
                &format!("{model_name}, {light_name}, held-out"),
                summaries[1],
            );
        }
    }
    renderer.destroy();
}
