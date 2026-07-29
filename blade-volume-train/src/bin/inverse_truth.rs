//! Does the split recover anything, or does it only re-render well?
//!
//! A photograph of a real room cannot answer that: nothing in it says what the
//! walls are made of. So this builds a scene whose answer is known, photographs
//! it, throws the answer away, and checks what the solver hands back.
//!
//! The photographs are rendered **with shadows and a bounce**; the fit assumes
//! neither. Without that asymmetry the solver would be inverting its own
//! forward pass, and would pass this test while failing every real one.
//!
//! Geometry is handed over at ground truth on purpose. Splitting radiance into
//! a material and a light is the part that is ambiguous, and it is worth
//! knowing what it can reach before a geometry error is added on top. What
//! comes out is an upper bound for the whole pipeline.
//!
//! Usage:
//!   inverse_truth --asset blade-volume-test/data/police.glb

use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train as train;
use std::path;

#[derive(argh::FromArgs)]
/// Measure what the material/light split recovers when the answer is known.
struct Args {
    /// a glTF asset or a `.surfel` file to build the truth from. Omit for the
    /// built-in studio scene, which is the one designed for this measurement.
    #[argh(option)]
    asset: Option<String>,

    /// surfel spacing of the built-in studio scene (default 0.04)
    #[argh(option, default = "0.04")]
    spacing: f32,

    /// sampling resolution when converting a glTF (default 300)
    #[argh(option, default = "300")]
    resolution: u32,

    /// views around the object (default 24)
    #[argh(option, default = "24")]
    views: usize,

    /// width of each photograph (default 320)
    #[argh(option, default = "320")]
    width: usize,

    /// height of each photograph (default 240)
    #[argh(option, default = "240")]
    height: usize,

    /// shadow rays per shading point when photographing (default 64). Zero
    /// makes the capture analytic, which is the same model the fit assumes,
    /// and turns this into a test of arithmetic.
    #[argh(option, default = "64")]
    samples: u32,

    /// material counts to sweep (default "1,16,256,4096,0"; 0 = one per surfel)
    #[argh(option, default = "String::from(\"1,16,256,4096,0\")")]
    materials: String,

    /// alternations between solving for albedo and for light (default 24)
    #[argh(option, default = "24")]
    iterations: usize,

    /// fit without shadowing, so a patch in shadow has only its material to
    /// explain being dark
    #[argh(switch)]
    no_shadows: bool,

    /// write photographs and re-renders here
    #[argh(option)]
    dump: Option<String>,
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();

    let model = match args.asset {
        None => train::inverse::truth::studio(args.spacing),
        Some(ref name) => {
            let asset = path::Path::new(name);
            if asset.extension().and_then(|e| e.to_str()) == Some(vol::io::SURFEL_EXTENSION) {
                match vol::io::try_load_relight(asset) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("cannot load {name}: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                let options = convert::ConvertOptions {
                    resolution: Some(args.resolution as f32),
                    ..Default::default()
                };
                match convert::relight_model_from_gltf(asset, &options) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("cannot convert {name}: {e:?}");
                        std::process::exit(1);
                    }
                }
            }
        }
    };
    let diffuse: f32 = model
        .materials
        .iter()
        .map(|m| m.albedo.iter().sum::<f32>())
        .sum();
    if diffuse <= 1.0e-6 {
        eprintln!(
            "every material in this asset is a metal, so it has no diffuse albedo to \
             recover and this measurement would be vacuous"
        );
        std::process::exit(1);
    }
    let Some((min, max)) = model.bounds() else {
        eprintln!("the asset has no geometry");
        std::process::exit(1);
    };

    // A sky with a sun in it. A uniform environment would make every normal
    // equally lit, and a fit against it could not be wrong about where the
    // light is because there would be nowhere for it to be.
    let environment = vol::relight::Environment::sky(
        glam::Vec3::new(0.55, 0.62, -0.56),
        [22.0, 20.0, 17.0],
        0.09,
        64,
        32,
    );
    let truth = train::inverse::score::Scene {
        model: model.clone(),
        environment,
    };
    println!(
        "truth: {} surfels over {} materials, {} views at {}x{}, {} shadow rays",
        model.surfels.len(),
        model.materials.len(),
        args.views,
        args.width,
        args.height,
        args.samples
    );

    let mut renderer = match train::inverse::score::Renderer::new(args.width, args.height) {
        Ok(r) => r,
        Err(message) => {
            eprintln!("cannot render: {message}");
            std::process::exit(1);
        }
    };
    println!("GPU: {}", renderer.device_name());

    let aspect = args.width as f32 / args.height as f32;
    let cameras = train::inverse::truth::orbit_poses(min, max, args.views, 0.35, 0.9, aspect);
    let started = std::time::Instant::now();
    let capture = train::inverse::truth::photograph(
        &mut renderer,
        &truth,
        &cameras,
        args.width,
        args.height,
        args.samples,
    );
    println!("photographed in {:.1} s\n", started.elapsed().as_secs_f64());

    if let Some(ref directory) = args.dump {
        let _ = std::fs::create_dir_all(directory);
        train::inverse::score::save_rgb(
            &path::Path::new(directory).join("truth-view0.png"),
            &capture.views[0].pixels,
            args.width,
            args.height,
        );
    }

    let all: Vec<usize> = (0..capture.views.len()).collect();
    let observations = train::inverse::decompose::observe(
        &model,
        &capture,
        &all,
        train::inverse::decompose::FitOptions::default().min_facing,
    );
    println!(
        "{} of {} surfels were seen by at least one view\n",
        observations.seen(),
        model.surfels.len()
    );

    // Computed once: it depends on the geometry and the sky's resolution, and
    // on neither the materials nor the light, so recomputing it per sweep
    // entry would only cost time.
    let options = train::inverse::decompose::FitOptions {
        iterations: args.iterations,
        ..Default::default()
    };
    let shadows = if args.no_shadows {
        None
    } else {
        let started = std::time::Instant::now();
        let directions =
            train::inverse::decompose::environment_directions(options.environment_width);
        let computed = train::inverse::visibility::compute(
            &model,
            &directions,
            train::inverse::visibility::VisibilityOptions::default(),
        );
        println!(
            "shadowing: {} directions, {:.0}% of the sky open on average, in {:.1} s\n",
            directions.len(),
            100.0 * computed.mean_openness(),
            started.elapsed().as_secs_f64()
        );
        Some(computed)
    };

    // The ceiling. With the real materials and the real light, whatever is
    // left is what the model cannot say — and no solver gets under it.
    // At the truth sky's own resolution, not the fit's: this is measuring the
    // model rather than the solver, so it should not inherit the solver's
    // coarser sky.
    let truth_shadows = train::inverse::visibility::compute(
        &model,
        &train::inverse::decompose::environment_directions(2 * truth.environment.height),
        train::inverse::visibility::VisibilityOptions::default(),
    );
    for (label, shadows) in [("open sky", None), ("shadowed", Some(&truth_shadows))] {
        let predicted = train::inverse::decompose::predict(&model, &truth.environment, shadows);
        let mut error = 0.0f64;
        let mut scale = 0.0f64;
        let mut count = 0usize;
        for (index, &weight) in observations.weight.iter().enumerate() {
            if weight <= 0.0 {
                continue;
            }
            for (value, truth) in predicted[index].iter().zip(observations.mean[index]) {
                let difference = (value - truth) as f64;
                error += difference * difference;
                scale += truth as f64;
            }
            count += 3;
        }
        let mean = (scale / count.max(1) as f64).max(1.0e-9);
        println!(
            "forward model with the truth in it, {label:>8}: {:.1}% off the photographs",
            100.0 * (error / count.max(1) as f64).sqrt() / mean
        );
    }
    println!();

    println!(
        "{:>10}{:>12}{:>10}{:>12}{:>10}{:>12}",
        "materials", "albedo err", "gauge", "light err", "psnr", "residual"
    );
    for count in args.materials.split(',') {
        let Ok(count) = count.trim().parse::<usize>() else {
            continue;
        };
        let fitted = train::inverse::decompose::fit(
            &model,
            &observations,
            train::inverse::decompose::FitOptions {
                materials: count,
                ..options
            },
            shadows.as_ref(),
        );
        let albedo = train::inverse::truth::compare_albedo(&model, &fitted.scene.model);
        let light = train::inverse::truth::compare_environment(
            &truth.environment,
            &fitted.scene.environment,
        );
        // Scored the way it was fitted. A fit that accounted for shadowing
        // has to be re-rendered with shadowing, or the comparison measures the
        // mismatch between two renderer settings rather than the scene.
        let scoring_samples = if shadows.is_some() { args.samples } else { 0 };
        let summary = renderer.score(&fitted.scene, &capture, &all, scoring_samples, None);
        let label = if count == 0 {
            "per surfel".to_string()
        } else {
            count.to_string()
        };
        println!(
            "{label:>10}{:>11.1}%{:>10.2}{:>11.1}%{:>12.2}{:>12.4}",
            100.0 * albedo.relative_rms,
            albedo.gauge[1],
            100.0 * light.relative_rms,
            summary.srgb_psnr,
            fitted.residual
        );

        if let Some(ref directory) = args.dump {
            let frames =
                renderer.render_views(&fitted.scene, &cameras[..1], scoring_samples, false);
            train::inverse::score::save_rgba(
                &path::Path::new(directory).join(format!("fit-{label}-view0.png")),
                &frames[0],
                args.width,
                args.height,
            );
        }
    }
    println!(
        "\nalbedo err and light err are what survives a per-channel gauge, which is\n\
         assumed rather than recovered; gauge is that factor on the green channel.\n\
         psnr re-renders the photographs the same way the fit modelled them."
    );
    renderer.destroy();
}
