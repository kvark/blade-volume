//! What can a relightable reconstruction reach if its geometry is perfect?
//!
//! The dataset hands over the normals the renderer used, so this solves only
//! the part that is actually ambiguous: splitting the observed radiance into a
//! material and a light. Holding geometry at ground truth makes the result an
//! upper bound — a real reconstruction has to estimate it as well — and if the
//! decomposition fails here, no amount of Gaussian or surfel machinery in front
//! of it will help.
//!
//! The model is the one a direct-lighting relightable primitive implements: a
//! Lambertian albedo against the environment's diffuse response, with no
//! shadowing and no interreflection. It is deliberately the simplest thing that
//! could work, so that what it cannot explain is legible.
//!
//! Three numbers come out of it, and the third is the only one that is
//! evidence:
//!
//!   - albedo error against the truth the scene was authored with;
//!   - training error, which one environment drives to zero by construction and
//!     which is therefore worth nothing on its own;
//!   - held-out relighting error, re-rendering under an environment the fit
//!     never saw, with that environment's light known — which is the situation
//!     relighting is actually in.
//!
//! Sweeping the number of training environments turns "how many lighting
//! conditions does this need" into a measurement.
//!
//! Usage:
//!   relight_bound --dataset <dir> [--held-out <name>]

use blade_volume_train as train;

#[derive(argh::FromArgs)]
/// Measure the relighting bound available with ground-truth geometry.
struct Args {
    /// directory written by blade's `relight_data` test
    #[argh(option)]
    dataset: String,

    /// environment to hold out of every fit (default: the last one)
    #[argh(option)]
    held_out: Option<String>,

    /// treat a sample as a metal, and report it apart, below this albedo
    #[argh(option, default = "0.02")]
    metal_albedo: f32,

    /// write per-view images of the fit into this directory
    #[argh(option)]
    dump: Option<String>,

    /// which view to write images of (default 6)
    #[argh(option, default = "6")]
    dump_view: usize,

    /// world-space size of a fused surface element
    #[argh(option, default = "0.08")]
    voxel: f32,

    /// joint optimiser iterations per element
    #[argh(option, default = "120")]
    iterations: usize,
}

/// The display transfer function, so linear radiance can be looked at.
fn encode_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let encoded = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

/// Write one plane of linear values as an image.
fn save_plane(path: &std::path::Path, pixels: &[[f32; 3]], width: usize, height: usize) {
    let mut buffer = image::RgbImage::new(width as u32, height as u32);
    for (index, texel) in pixels.iter().enumerate() {
        buffer.put_pixel(
            (index % width) as u32,
            (index / width) as u32,
            image::Rgb([
                encode_srgb(texel[0]),
                encode_srgb(texel[1]),
                encode_srgb(texel[2]),
            ]),
        );
    }
    let _ = buffer.save(path);
}

/// Outcome of one fit.
struct Score {
    albedo_rmse: f64,
    albedo_rmse_dielectric: f64,
    train_psnr: f64,
    held_out_psnr: f64,
    held_out_psnr_dielectric: f64,
}

fn evaluate(
    samples: &[train::relight::Sample],
    lights: &[(usize, train::relight::Irradiance)],
    held_out: usize,
    held_out_light: &train::relight::Irradiance,
    metal_albedo: f32,
    oracle: bool,
) -> Score {
    let mut albedo_error = train::relight::Accumulator::new();
    let mut albedo_error_dielectric = train::relight::Accumulator::new();
    let mut train_error = train::relight::Accumulator::new();
    let mut held_error = train::relight::Accumulator::new();
    let mut held_error_dielectric = train::relight::Accumulator::new();

    for sample in samples {
        let albedo = if oracle {
            sample.albedo_truth
        } else {
            train::relight::solve_albedo(sample, lights)
        };
        let dielectric = sample.albedo_truth.iter().any(|c| *c > metal_albedo);

        albedo_error.add(albedo, sample.albedo_truth);
        if dielectric {
            albedo_error_dielectric.add(albedo, sample.albedo_truth);
        }

        for &(environment, ref irradiance) in lights {
            let shade = irradiance.shade(sample.normal);
            let predicted = [
                albedo[0] * shade[0],
                albedo[1] * shade[1],
                albedo[2] * shade[2],
            ];
            train_error.add(predicted, sample.radiance[environment]);
        }

        // Relighting knows the light it is asked to render under; what it does
        // not know is the material, which is the whole point.
        let shade = held_out_light.shade(sample.normal);
        let predicted = [
            albedo[0] * shade[0],
            albedo[1] * shade[1],
            albedo[2] * shade[2],
        ];
        held_error.add(predicted, sample.radiance[held_out]);
        if dielectric {
            held_error_dielectric.add(predicted, sample.radiance[held_out]);
        }
    }

    Score {
        albedo_rmse: albedo_error.rmse(),
        albedo_rmse_dielectric: albedo_error_dielectric.rmse(),
        train_psnr: train_error.psnr(),
        held_out_psnr: held_error.psnr(),
        held_out_psnr_dielectric: held_error_dielectric.psnr(),
    }
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();
    let root = std::path::Path::new(&args.dataset);

    let dataset = match train::relight::Dataset::load(root) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!(
        "{} views x {} environments at {}x{}",
        dataset.views.len(),
        dataset.environments.len(),
        dataset.width,
        dataset.height
    );

    let irradiance = match dataset.environment_irradiance() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let held_out = match args.held_out {
        Some(ref name) => match dataset.environments.iter().position(|e| e == name) {
            Some(index) => index,
            None => {
                eprintln!("no environment named {name}");
                std::process::exit(1);
            }
        },
        None => dataset.environments.len() - 1,
    };
    let training: Vec<usize> = (0..dataset.environments.len())
        .filter(|index| *index != held_out)
        .collect();
    println!(
        "holding out '{}', training from {:?}\n",
        dataset.environments[held_out],
        training
            .iter()
            .map(|i| dataset.environments[*i].as_str())
            .collect::<Vec<_>>()
    );

    let samples = match train::relight::gather_samples(&dataset) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let metals = samples
        .iter()
        .filter(|s| s.albedo_truth.iter().all(|c| *c <= args.metal_albedo))
        .count();
    println!(
        "{} surface samples, {:.1} % of them metal\n",
        samples.len(),
        100.0 * metals as f64 / samples.len() as f64
    );

    println!(
        "{:<28}{:>10}{:>10}{:>12}{:>12}",
        "fit", "albedo", "train", "held-out", "held-out"
    );
    println!(
        "{:<28}{:>10}{:>10}{:>12}{:>12}",
        "", "rmse", "psnr", "psnr", "dielectric"
    );

    for count in 1..=training.len() {
        let lights: Vec<_> = training[..count]
            .iter()
            .map(|index| (*index, irradiance[*index]))
            .collect();
        let score = evaluate(
            &samples,
            &lights,
            held_out,
            &irradiance[held_out],
            args.metal_albedo,
            false,
        );
        let names: Vec<&str> = training[..count]
            .iter()
            .map(|i| dataset.environments[*i].as_str())
            .collect();
        println!(
            "{:<28}{:>10.4}{:>10.2}{:>12.2}{:>12.2}",
            format!("{count} env: {}", names.join("+")),
            score.albedo_rmse,
            score.train_psnr,
            score.held_out_psnr,
            score.held_out_psnr_dielectric
        );
    }

    // The same model handed the true albedo. Whatever it still cannot reach is
    // the model's own limit rather than the fit's: no shadowing, no
    // interreflection, no specular lobe, and only nine coefficients of light.
    let all: Vec<_> = training
        .iter()
        .map(|index| (*index, irradiance[*index]))
        .collect();
    let oracle = evaluate(
        &samples,
        &all,
        held_out,
        &irradiance[held_out],
        args.metal_albedo,
        true,
    );
    println!(
        "\n{:<28}{:>10.4}{:>10.2}{:>12.2}{:>12.2}",
        "oracle albedo",
        oracle.albedo_rmse,
        oracle.train_psnr,
        oracle.held_out_psnr,
        oracle.held_out_psnr_dielectric
    );
    // Where the model's own error lives. A specular lobe is a function of
    // roughness and vanishes as it goes to one; shadowing is not and does not.
    // So sorting the oracle's residual by roughness says which of the two the
    // model is actually missing, without having to implement either.
    // Adding the specular lobe the sweep above points at. The albedo, the
    // reflectance and the roughness are all exact here, so what changes is the
    // model rather than the fit.
    let specular = match dataset.environment_specular() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let mut with_train = train::relight::Accumulator::new();
    let mut with_held = train::relight::Accumulator::new();
    for sample in &samples {
        let albedo = sample.albedo_truth;
        for &index in &training {
            let shade = irradiance[index].shade(sample.normal);
            let lobe = train::relight::specular_radiance(sample, &specular[index]);
            with_train.add(
                [
                    albedo[0] * shade[0] + lobe[0],
                    albedo[1] * shade[1] + lobe[1],
                    albedo[2] * shade[2] + lobe[2],
                ],
                sample.radiance[index],
            );
        }
        let shade = irradiance[held_out].shade(sample.normal);
        let lobe = train::relight::specular_radiance(sample, &specular[held_out]);
        with_held.add(
            [
                albedo[0] * shade[0] + lobe[0],
                albedo[1] * shade[1] + lobe[1],
                albedo[2] * shade[2] + lobe[2],
            ],
            sample.radiance[held_out],
        );
    }
    println!(
        "\noracle albedo + specular lobe   {:>10}{:>10.2}{:>12.2}",
        "-",
        with_train.psnr(),
        with_held.psnr()
    );

    // The same sweep as the first table, but with the lobe subtracted before
    // the albedo is solved for. If the material error was the model's bias
    // being absorbed, this is where it comes back out.
    println!("\nfitted albedo, with the specular lobe in the model:");
    println!(
        "{:<28}{:>10}{:>10}{:>12}",
        "fit", "albedo", "train", "held-out"
    );
    for count in 1..=training.len() {
        let lights: Vec<_> = training[..count]
            .iter()
            .map(|index| (*index, irradiance[*index]))
            .collect();
        let mut albedo_error = train::relight::Accumulator::new();
        let mut train_error = train::relight::Accumulator::new();
        let mut held_error = train::relight::Accumulator::new();
        for sample in &samples {
            let albedo = train::relight::solve_albedo_specular(sample, &lights, &specular);
            albedo_error.add(albedo, sample.albedo_truth);
            for &(index, ref light) in &lights {
                let shade = light.shade(sample.normal);
                let lobe = train::relight::specular_radiance(sample, &specular[index]);
                train_error.add(
                    [
                        albedo[0] * shade[0] + lobe[0],
                        albedo[1] * shade[1] + lobe[1],
                        albedo[2] * shade[2] + lobe[2],
                    ],
                    sample.radiance[index],
                );
            }
            let shade = irradiance[held_out].shade(sample.normal);
            let lobe = train::relight::specular_radiance(sample, &specular[held_out]);
            held_error.add(
                [
                    albedo[0] * shade[0] + lobe[0],
                    albedo[1] * shade[1] + lobe[1],
                    albedo[2] * shade[2] + lobe[2],
                ],
                sample.radiance[held_out],
            );
        }
        let names: Vec<&str> = training[..count]
            .iter()
            .map(|i| dataset.environments[*i].as_str())
            .collect();
        println!(
            "{:<28}{:>10.4}{:>10.2}{:>12.2}",
            format!("{count} env: {}", names.join("+")),
            albedo_error.rmse(),
            train_error.psnr(),
            held_error.psnr()
        );
    }

    // Put the fit back into image space, so the numbers above can be looked at.
    if let Some(ref directory) = args.dump {
        let directory = std::path::Path::new(directory);
        let _ = std::fs::create_dir_all(directory);
        let width = dataset.width;
        let height = dataset.height;
        let count = width * height;
        let background = [0.0f32; 3];

        let lights: Vec<_> = training
            .iter()
            .map(|index| (*index, irradiance[*index]))
            .collect();
        let mut truth_plane = vec![background; count];
        let mut diffuse_plane = vec![background; count];
        let mut specular_plane = vec![background; count];
        let mut albedo_truth_plane = vec![background; count];
        let mut albedo_diffuse_plane = vec![background; count];
        let mut albedo_specular_plane = vec![background; count];
        let mut error_diffuse_plane = vec![background; count];
        let mut error_specular_plane = vec![background; count];

        for sample in samples.iter().filter(|s| s.view_index == args.dump_view) {
            let shade = irradiance[held_out].shade(sample.normal);
            let lobe = train::relight::specular_radiance(sample, &specular[held_out]);
            let observed = sample.radiance[held_out];

            let only_diffuse = train::relight::solve_albedo(sample, &lights);
            let with_specular = train::relight::solve_albedo_specular(sample, &lights, &specular);
            let predicted_diffuse = [
                only_diffuse[0] * shade[0],
                only_diffuse[1] * shade[1],
                only_diffuse[2] * shade[2],
            ];
            let predicted_specular = [
                with_specular[0] * shade[0] + lobe[0],
                with_specular[1] * shade[1] + lobe[1],
                with_specular[2] * shade[2] + lobe[2],
            ];

            let pixel = sample.pixel;
            truth_plane[pixel] = observed;
            diffuse_plane[pixel] = predicted_diffuse;
            specular_plane[pixel] = predicted_specular;
            albedo_truth_plane[pixel] = sample.albedo_truth;
            albedo_diffuse_plane[pixel] = only_diffuse;
            albedo_specular_plane[pixel] = with_specular;
            // Amplified, because the interesting differences are small.
            for channel in 0..3 {
                error_diffuse_plane[pixel][channel] =
                    4.0 * (predicted_diffuse[channel] - observed[channel]).abs();
                error_specular_plane[pixel][channel] =
                    4.0 * (predicted_specular[channel] - observed[channel]).abs();
            }
        }

        for (name, plane) in [
            ("held-out-truth", &truth_plane),
            ("relit-diffuse", &diffuse_plane),
            ("relit-specular", &specular_plane),
            ("albedo-truth", &albedo_truth_plane),
            ("albedo-diffuse", &albedo_diffuse_plane),
            ("albedo-specular", &albedo_specular_plane),
            ("error-diffuse-x4", &error_diffuse_plane),
            ("error-specular-x4", &error_specular_plane),
        ] {
            save_plane(&directory.join(format!("{name}.png")), plane, width, height);
        }
        println!(
            "\nwrote images of view {} under '{}' to {}",
            args.dump_view,
            dataset.environments[held_out],
            directory.display()
        );
    }

    // Everything above solves each pixel on its own, so a surface point is
    // seen once per environment and a specular lobe is indistinguishable from
    // a brighter albedo. Fusing the views that landed on the same point makes
    // them separable, because the diffuse part is the same from every
    // direction and the lobe is not.
    {
        let mut elements = train::relight::build_elements(&samples, args.voxel);
        let observations: usize = elements.iter().map(|e| e.samples.len()).sum();
        let multi = elements.iter().filter(|e| e.samples.len() > 1).count();
        println!(
            "\nfused {} samples into {} elements at {} world units: {:.1} views each, {:.0} % seen more than once",
            observations,
            elements.len(),
            args.voxel,
            observations as f64 / elements.len() as f64,
            100.0 * multi as f64 / elements.len() as f64
        );

        println!(
            "\n{:<34}{:>10}{:>10}{:>12}{:>12}",
            "joint fit", "albedo", "rough", "held-out", "held-out"
        );
        println!(
            "{:<34}{:>10}{:>10}{:>12}{:>12}",
            "", "rmse", "rmse", "psnr", "dielectric"
        );

        // Freezing one at the truth at a time says which of the two the fit
        // cannot identify, rather than only that it cannot identify both.
        for (label, seed_truth, fit_specular, fit_roughness) in [
            ("albedo only, true rough+F0", true, false, false),
            ("albedo + roughness, true F0", true, false, true),
            ("albedo + F0, true roughness", true, true, false),
            ("albedo + roughness + F0", true, true, true),
            ("all three, from a grey guess", false, true, true),
        ] {
            let mut fitted = train::relight::build_elements(&samples, args.voxel);
            if seed_truth {
                train::relight::seed_elements_from_truth(&mut fitted, &samples);
            }
            let config = train::relight::JointConfig {
                iterations: args.iterations,
                fit_specular,
                fit_roughness,
                ..Default::default()
            };
            train::relight::optimize_elements(
                &mut fitted,
                &samples,
                &training,
                &irradiance,
                &specular,
                config,
            );

            let mut albedo_error = train::relight::Accumulator::new();
            let mut held_error = train::relight::Accumulator::new();
            let mut held_dielectric = train::relight::Accumulator::new();
            let mut roughness_squared = 0.0f64;
            let mut roughness_count = 0usize;
            // Roughness is only identifiable where the lobe is narrow enough
            // to see. On a fully rough surface it barely changes the image, so
            // pooling it with the rest says more about the floor than about
            // whether the fit works.
            let mut glossy_squared = 0.0f64;
            let mut glossy_count = 0usize;
            let mut metal_f0_squared = 0.0f64;
            let mut metal_f0_count = 0usize;
            for element in &fitted {
                for &index in &element.samples {
                    let sample = &samples[index as usize];
                    albedo_error.add(element.albedo, sample.albedo_truth);
                    let difference = (element.roughness - sample.roughness) as f64;
                    roughness_squared += difference * difference;
                    roughness_count += 1;
                    if sample.roughness < 0.9 {
                        glossy_squared += difference * difference;
                        glossy_count += 1;
                    }
                    if sample.albedo_truth.iter().all(|c| *c <= args.metal_albedo) {
                        for channel in 0..3 {
                            let d =
                                (element.specular_f0[channel] - sample.specular_f0[channel]) as f64;
                            metal_f0_squared += d * d;
                            metal_f0_count += 1;
                        }
                    }
                    let predicted = train::relight::predict(
                        sample,
                        element.albedo,
                        element.specular_f0,
                        element.roughness,
                        &irradiance[held_out],
                        &specular[held_out],
                    );
                    held_error.add(predicted, sample.radiance[held_out]);
                    if sample.albedo_truth.iter().any(|c| *c > args.metal_albedo) {
                        held_dielectric.add(predicted, sample.radiance[held_out]);
                    }
                }
            }
            println!(
                "{:<34}{:>10.4}{:>10.4}{:>12.2}{:>12.2}",
                label,
                albedo_error.rmse(),
                (roughness_squared / roughness_count.max(1) as f64).sqrt(),
                held_error.psnr(),
                held_dielectric.psnr()
            );
            println!(
                "{:<34}rough rmse on glossy only {:.4}, metal F0 rmse {:.4}",
                "",
                (glossy_squared / glossy_count.max(1) as f64).sqrt(),
                (metal_f0_squared / metal_f0_count.max(1) as f64).sqrt(),
            );
            elements = fitted;
        }
        drop(elements);
    }

    // How accurate does a reconstruction's geometry have to be? The bound above
    // is handed exact normals; a real primitive estimates them. Tilting every
    // normal by a fixed angle, and using the tilted one both to fit and to
    // relight, is what a normal error of that size would actually cost.
    println!("\nsensitivity to normal error (diffuse + specular, all training environments):");
    println!(
        "{:<16}{:>12}{:>14}",
        "tilt (degrees)", "albedo rmse", "held-out psnr"
    );
    let lights: Vec<_> = training
        .iter()
        .map(|index| (*index, irradiance[*index]))
        .collect();
    for degrees in [0.0f32, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0] {
        let angle = degrees.to_radians();
        let mut albedo_error = train::relight::Accumulator::new();
        let mut held_error = train::relight::Accumulator::new();
        for (index, sample) in samples.iter().enumerate() {
            let tilted = train::relight::Sample {
                view_index: sample.view_index,
                pixel: sample.pixel,
                normal: train::relight::perturb_normal(sample.normal, angle, index as u32),
                albedo_truth: sample.albedo_truth,
                specular_f0: sample.specular_f0,
                roughness: sample.roughness,
                view: sample.view,
                position: sample.position,
                radiance: sample.radiance.clone(),
            };
            let albedo = train::relight::solve_albedo_specular(&tilted, &lights, &specular);
            albedo_error.add(albedo, sample.albedo_truth);
            let shade = irradiance[held_out].shade(tilted.normal);
            let lobe = train::relight::specular_radiance(&tilted, &specular[held_out]);
            held_error.add(
                [
                    albedo[0] * shade[0] + lobe[0],
                    albedo[1] * shade[1] + lobe[1],
                    albedo[2] * shade[2] + lobe[2],
                ],
                sample.radiance[held_out],
            );
        }
        println!(
            "{:<16.0}{:>12.4}{:>14.2}",
            degrees,
            albedo_error.rmse(),
            held_error.psnr()
        );
    }

    println!("\noracle residual by roughness (albedo is exact, so this is model error):");
    println!(
        "{:<14}{:>10}{:>12}{:>12}{:>14}",
        "roughness", "samples", "diffuse", "+specular", "held-out spec"
    );
    let mut buckets: Vec<f32> = samples.iter().map(|s| s.roughness).collect();
    buckets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    buckets.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
    for &roughness in &buckets {
        let selected: Vec<&train::relight::Sample> = samples
            .iter()
            .filter(|s| (s.roughness - roughness).abs() < 1e-3)
            .collect();
        let mut train_error = train::relight::Accumulator::new();
        let mut train_specular = train::relight::Accumulator::new();
        let mut held_specular = train::relight::Accumulator::new();
        for sample in &selected {
            let albedo = sample.albedo_truth;
            for &index in &training {
                let shade = irradiance[index].shade(sample.normal);
                let diffuse = [
                    albedo[0] * shade[0],
                    albedo[1] * shade[1],
                    albedo[2] * shade[2],
                ];
                train_error.add(diffuse, sample.radiance[index]);
                let lobe = train::relight::specular_radiance(sample, &specular[index]);
                train_specular.add(
                    [
                        diffuse[0] + lobe[0],
                        diffuse[1] + lobe[1],
                        diffuse[2] + lobe[2],
                    ],
                    sample.radiance[index],
                );
            }
            let shade = irradiance[held_out].shade(sample.normal);
            let lobe = train::relight::specular_radiance(sample, &specular[held_out]);
            held_specular.add(
                [
                    albedo[0] * shade[0] + lobe[0],
                    albedo[1] * shade[1] + lobe[1],
                    albedo[2] * shade[2] + lobe[2],
                ],
                sample.radiance[held_out],
            );
        }
        println!(
            "{:<14.2}{:>10}{:>12.2}{:>12.2}{:>14.2}",
            roughness,
            selected.len(),
            train_error.psnr(),
            train_specular.psnr(),
            held_specular.psnr()
        );
    }

    // The complement of the test above. Restricted to the roughest samples,
    // where there is no specular lobe left to miss, the residual is whatever
    // else the model leaves out - and shadowing is the term that separates a
    // uniform environment, which casts none, from a concentrated one.
    println!("\noracle residual on fully rough samples, per environment:");
    println!("{:<14}{:>12}", "environment", "train psnr");
    let rough: Vec<&train::relight::Sample> =
        samples.iter().filter(|s| s.roughness > 0.99).collect();
    for &index in &training {
        let mut error = train::relight::Accumulator::new();
        for sample in &rough {
            let albedo = sample.albedo_truth;
            let shade = irradiance[index].shade(sample.normal);
            error.add(
                [
                    albedo[0] * shade[0],
                    albedo[1] * shade[1],
                    albedo[2] * shade[2],
                ],
                sample.radiance[index],
            );
        }
        println!("{:<14}{:>12.2}", dataset.environments[index], error.psnr());
    }

    // With the lights known there is nothing ambiguous left to resolve, so
    // extra environments only average away noise. The claim that several
    // lighting conditions are what make the split well posed is about the case
    // where the light is unknown too, which is this one.
    println!("\nlights recovered from the images as well (scale quotiented out):");
    println!(
        "{:<28}{:>10}{:>10}{:>12}{:>12}",
        "fit", "albedo", "train", "held-out", "light"
    );
    println!(
        "{:<28}{:>10}{:>10}{:>12}{:>12}",
        "", "rmse", "psnr", "psnr", "rmse"
    );
    for count in 1..=training.len() {
        let subset: Vec<usize> = training[..count].to_vec();
        let (albedos, lights) = train::relight::alternate(&samples, &subset, 12);
        let truth: Vec<[f32; 3]> = samples.iter().map(|s| s.albedo_truth).collect();
        let scale = train::relight::optimal_scale(&albedos, &truth);

        let mut albedo_error = train::relight::Accumulator::new();
        let mut train_error = train::relight::Accumulator::new();
        let mut held_error = train::relight::Accumulator::new();
        for (index, sample) in samples.iter().enumerate() {
            let scaled = [
                albedos[index][0] * scale[0],
                albedos[index][1] * scale[1],
                albedos[index][2] * scale[2],
            ];
            albedo_error.add(scaled, sample.albedo_truth);
            for (slot, environment) in subset.iter().enumerate() {
                let shade = lights[slot].shade(sample.normal);
                // The recovered light carries the reciprocal of the scale, so
                // the product is unchanged here by construction.
                let predicted = [
                    albedos[index][0] * shade[0],
                    albedos[index][1] * shade[1],
                    albedos[index][2] * shade[2],
                ];
                train_error.add(predicted, sample.radiance[*environment]);
            }
            let shade = irradiance[held_out].shade(sample.normal);
            let predicted = [
                scaled[0] * shade[0],
                scaled[1] * shade[1],
                scaled[2] * shade[2],
            ];
            held_error.add(predicted, sample.radiance[held_out]);
        }

        // How far the recovered lights are from the ones that were used, after
        // undoing the same scale.
        let mut light_error = 0.0f64;
        let mut light_count = 0usize;
        for (slot, &environment) in subset.iter().enumerate() {
            for index in 0..9 {
                let recovered = lights[slot].coefficients[index];
                let expected = irradiance[environment].coefficients[index];
                for ((got, want), factor) in recovered.iter().zip(expected).zip(scale) {
                    let difference = (got / factor.max(1e-6) - want) as f64;
                    light_error += difference * difference;
                    light_count += 1;
                }
            }
        }
        let names: Vec<&str> = subset
            .iter()
            .map(|i| dataset.environments[*i].as_str())
            .collect();
        println!(
            "{:<28}{:>10.4}{:>10.2}{:>12.2}{:>12.4}",
            format!("{count} env: {}", names.join("+")),
            albedo_error.rmse(),
            train_error.psnr(),
            held_error.psnr(),
            (light_error / light_count as f64).sqrt(),
        );
    }

    println!(
        "\n  dielectric-only albedo rmse: fit {:.4}, oracle {:.4}",
        evaluate(
            &samples,
            &all,
            held_out,
            &irradiance[held_out],
            args.metal_albedo,
            false,
        )
        .albedo_rmse_dielectric,
        oracle.albedo_rmse_dielectric
    );
}
