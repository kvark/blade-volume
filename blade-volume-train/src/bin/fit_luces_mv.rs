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

    /// optional directory for held-light/held-camera reference and diagnostic renders
    #[argh(option)]
    dump: Option<String>,

    /// fitting image width (default 128; height preserves aspect)
    #[argh(option, default = "128")]
    width: usize,

    /// alternating normal/material rounds (default 2)
    #[argh(option, default = "2")]
    rounds: usize,

    /// hemisphere candidates tested per normal update (default 1024)
    #[argh(option, default = "1024")]
    normal_candidates: usize,

    /// maximum diffuse albedo (default 1)
    #[argh(option, default = "1.0")]
    albedo_ceiling: f32,
}

#[derive(Clone, Copy)]
struct PhysicalScore {
    samples: usize,
    linear_psnr: f64,
    srgb_psnr: f64,
    black_srgb_psnr: f64,
}

fn psnr(error: f64, samples: usize) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    let mean = error / samples as f64;
    if mean <= 1.0e-12 {
        99.0
    } else {
        -10.0 * mean.log10()
    }
}

fn score_physical_samples(
    surface: &vol::relight::RelightModel,
    dataset: &train::inverse::luces::Dataset,
    views: &[usize],
) -> PhysicalScore {
    let mut linear_error = 0.0f64;
    let mut encoded_error = 0.0f64;
    let mut black_error = 0.0f64;
    let mut samples = 0usize;
    for (capture, lights) in dataset.captures.iter().zip(&dataset.lights) {
        let observations = train::inverse::decompose::observe(surface, capture, views, 0.15);
        for (surfel_index, surfel) in surface.surfels.iter().enumerate() {
            let material = surface.materials[surfel.material as usize];
            let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
            let point = glam::Vec3::from(surfel.center);
            for observation in observations.of(surfel_index) {
                let Some(light) = lights[observation.view as usize].sample(point) else {
                    continue;
                };
                let cosine = normal.dot(light.towards).max(0.0);
                for channel in 0..3 {
                    let prediction = material.albedo[channel] * light.radiance[channel] * cosine;
                    let reference = observation.radiance[channel];
                    linear_error += f64::from(prediction - reference).powi(2);
                    let encoded_prediction =
                        train::inverse::capture::linear_to_srgb(prediction) as f64;
                    let encoded_reference =
                        train::inverse::capture::linear_to_srgb(reference) as f64;
                    encoded_error += (encoded_prediction - encoded_reference).powi(2);
                    black_error += encoded_reference.powi(2);
                    samples += 1;
                }
            }
        }
    }
    PhysicalScore {
        samples: samples / 3,
        linear_psnr: psnr(linear_error, samples),
        srgb_psnr: psnr(encoded_error, samples),
        black_srgb_psnr: psnr(black_error, samples),
    }
}

fn print_score(label: &str, score: PhysicalScore) {
    println!(
        "{label}: {} surface/view samples, {:.2} dB linear, {:.2} dB sRGB (black {:.2} dB)",
        score.samples, score.linear_psnr, score.srgb_psnr, score.black_srgb_psnr,
    );
}

fn render_point_light(
    surface: &vol::relight::RelightModel,
    capture: &train::inverse::capture::Capture,
    lights: &[vol::relight::PointLight],
    view: usize,
) -> Vec<[f32; 4]> {
    let camera = capture.views[view].camera;
    let mut rendered = vec![[0.0; 4]; capture.width * capture.height];
    let mut depth_buffer = vec![f32::INFINITY; rendered.len()];
    let pixels_per_world = capture.width as f32 / (2.0 * (0.5 * camera.fov[0]).tan());
    for surfel in &surface.surfels {
        let point = glam::Vec3::from(surfel.center);
        let Some((pixel, depth)) =
            train::inverse::capture::project(&camera, capture.width, capture.height, point)
        else {
            continue;
        };
        let Some(light) = lights[view].sample(point) else {
            continue;
        };
        let normal = glam::Vec3::from(surfel.normal).normalize_or_zero();
        let cosine = normal.dot(light.towards).max(0.0);
        let material = surface.materials[surfel.material as usize];
        let color: [f32; 3] = std::array::from_fn(|channel| {
            material.albedo[channel] * light.radiance[channel] * cosine
        });
        let radius = (surfel.radius * pixels_per_world / depth).max(0.75);
        let x_min = (pixel[0] - radius).floor().max(0.0) as usize;
        let y_min = (pixel[1] - radius).floor().max(0.0) as usize;
        let x_max = (pixel[0] + radius)
            .ceil()
            .min(capture.width.saturating_sub(1) as f32) as usize;
        let y_max = (pixel[1] + radius)
            .ceil()
            .min(capture.height.saturating_sub(1) as f32) as usize;
        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let offset = glam::Vec2::new(x as f32 + 0.5 - pixel[0], y as f32 + 0.5 - pixel[1]);
                if offset.length_squared() > radius * radius {
                    continue;
                }
                let index = y * capture.width + x;
                if depth < depth_buffer[index] {
                    depth_buffer[index] = depth;
                    rendered[index] = [color[0], color[1], color[2], 1.0];
                }
            }
        }
    }
    rendered
}

fn save_linear_image(
    path: &path::Path,
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
) -> Result<(), String> {
    let encoded = pixels
        .iter()
        .flat_map(|rgb| {
            rgb.map(|value| {
                (train::inverse::capture::linear_to_srgb(value) * u8::MAX as f32).round() as u8
            })
        })
        .collect();
    image::RgbImage::from_raw(width as u32, height as u32, encoded)
        .expect("linear image has inconsistent dimensions")
        .save(path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn dump_held_cross(
    directory: &path::Path,
    surface: &vol::relight::RelightModel,
    held: &train::inverse::luces::Dataset,
) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    for (light_slot, (&source_light, (capture, lights))) in held
        .source_light_indices
        .iter()
        .zip(held.captures.iter().zip(&held.lights))
        .enumerate()
    {
        for &view in &train::inverse::luces::HELD_VIEW_INDICES {
            let rendered = render_point_light(surface, capture, lights, view);
            let reference_path = directory.join(format!(
                "light-{:02}-view-{:03}-reference.png",
                source_light + 1,
                train::inverse::luces::VIEW_IDS[view],
            ));
            let render_path = directory.join(format!(
                "light-{:02}-view-{:03}-render.png",
                source_light + 1,
                train::inverse::luces::VIEW_IDS[view],
            ));
            save_linear_image(
                &reference_path,
                &capture.views[view].pixels,
                capture.width,
                capture.height,
            )?;
            let linear = rendered
                .iter()
                .map(|rgba| [rgba[0], rgba[1], rgba[2]])
                .collect::<Vec<_>>();
            save_linear_image(&render_path, &linear, capture.width, capture.height)?;
        }
        println!(
            "dumped held LED {:02} ({}/{})",
            source_light + 1,
            light_slot + 1,
            held.captures.len(),
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
        let normals = train::inverse::decompose::refine_normals_known_lights_per_view(
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

    print_score(
        "fitted lights / fitted cameras",
        score_physical_samples(
            &surface,
            &training,
            &train::inverse::luces::TRAIN_VIEW_INDICES,
        ),
    );
    print_score(
        "fitted lights / held cameras",
        score_physical_samples(
            &surface,
            &training,
            &train::inverse::luces::HELD_VIEW_INDICES,
        ),
    );

    vol::io::try_save_relight(path::Path::new(&args.output), &surface)
        .map_err(|error| format!("cannot write {}: {error}", args.output))?;
    println!("wrote {}", args.output);
    if let (Some(output), Some(model)) = (args.gaussian_output.as_deref(), gaussian.as_ref()) {
        save_gaussian(path::Path::new(output), model)?;
        println!("wrote {output}");
    }

    drop(gaussian);
    drop(training);
    let held = train::inverse::luces::load(
        input,
        camera_one,
        camera_two,
        args.width,
        &train::inverse::luces::HELD_LIGHT_INDICES,
    )?;
    print_score(
        "held lights / fitted cameras",
        score_physical_samples(&surface, &held, &train::inverse::luces::TRAIN_VIEW_INDICES),
    );
    print_score(
        "held lights / held cameras",
        score_physical_samples(&surface, &held, &train::inverse::luces::HELD_VIEW_INDICES),
    );
    if let Some(ref dump) = args.dump {
        dump_held_cross(path::Path::new(dump), &surface, &held)?;
    }
    Ok(())
}

fn main() {
    let args: Args = argh::from_env();
    if let Err(message) = run(&args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_point_light_render_uses_calibrated_distance_and_normal() {
        let surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![vol::relight::Surfel {
                center: [0.0, 0.0, 2.0],
                radius: 0.5,
                normal: [0.0, 0.0, -1.0],
                material: 0,
            }],
            materials: vec![vol::relight::Material {
                albedo: [0.5; 3],
                ..Default::default()
            }],
        };
        let capture = train::inverse::capture::Capture {
            width: 8,
            height: 8,
            views: vec![train::inverse::capture::View {
                name: "fixture".to_string(),
                camera: vol::CameraParams {
                    cam_position: [0.0; 3],
                    depth: 10.0,
                    cam_orientation: glam::Quat::IDENTITY.to_array(),
                    fov: [std::f32::consts::FRAC_PI_2; 2],
                    principal: [0.0; 2],
                },
                pixels: vec![[0.0; 3]; 64],
                mask: Some(vec![0.0; 64]),
            }],
        };
        let light = vol::relight::PointLight {
            position: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            intensity: [4.0; 3],
            exponent: 0.0,
        };
        let rendered = render_point_light(&surface, &capture, &[light], 0);
        let center = rendered[4 * 8 + 4];
        assert_eq!(center, [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(rendered[0], [0.0; 4]);
    }
}
