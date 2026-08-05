#![allow(irrefutable_let_patterns)]

//! Score the relightable surfel renderer against a path traced reference.
//!
//! Everything checked so far has been internal: the GPU shading agrees with a
//! CPU implementation of the same formulas, and the sampled lighting agrees
//! with the analytic irradiance it replaces. None of that says the renderer
//! produces the *right* image — only that it is consistent with itself.
//!
//! This compares it against something that does not share its assumptions.
//! Blade's canonical path tracer renders the source glTF with a real GGX BRDF
//! under an environment; the same file is converted to surfels and rendered
//! here under the same environment from the same pose. What separates the two
//! images is the representation and the shading model, which is the thing
//! worth knowing.
//!
//! Both sides start from the same triangles and the same materials, so a
//! difference is not two scenes that merely resemble each other.
//!
//! Generate the reference with blade's `relight_data` test, pointing it at the
//! asset and framing it on the asset's own bounds, then:
//!
//! ```text
//!   relight_quality --dataset <dir> --asset <file.gltf>
//! ```

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train as train;

#[derive(argh::FromArgs)]
/// Score the surfel renderer against a path traced reference.
struct Args {
    /// reference dataset written by blade's `relight_data`
    #[argh(option)]
    dataset: String,

    /// the glTF the reference was rendered from, or a `.surfel` file already
    /// converted from it
    #[argh(option)]
    asset: String,

    /// surfels per unit of the asset's diagonal
    #[argh(option, default = "400.0")]
    resolution: f32,

    /// rays per shading point for shadowing and the bounce with it
    #[argh(option, default = "0")]
    samples: u32,

    /// write the rendered and reference images here
    #[argh(option)]
    dump: Option<String>,
}

/// Peak signal-to-noise ratio over linear radiance, against a peak of one.
///
/// The convention the rest of the relighting work reports, so the numbers sit
/// next to each other. It is the wrong measure on its own for a frame with a
/// light source in it: the error is squared and the peak is fixed, so a
/// handful of pixels at five times white can outweigh every other pixel in the
/// image put together. Read it next to [`tone_mapped_psnr`].
///
/// [`tone_mapped_psnr`]: fn.tone_mapped_psnr.html
fn psnr(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (x, y) in a.iter().zip(b) {
        for channel in 0..3 {
            let difference = (x[channel] - y[channel]) as f64;
            sum += difference * difference;
            count += 1;
        }
    }
    let mse = sum / count.max(1) as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// The same, after the curve a display would apply.
///
/// Compresses the highlights before differencing, so what is measured is how
/// far apart the two images look rather than how far apart their brightest
/// pixels are. On a frame with the light source in view the two disagree by
/// more than ten decibels, and this is the one that corresponds to looking.
fn tone_mapped_psnr(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let map = |v: f32| {
        let v = v.max(0.0);
        let reinhard = v / (1.0 + v);
        if reinhard <= 0.003_130_8 {
            12.92 * reinhard
        } else {
            1.055 * reinhard.powf(1.0 / 2.4) - 0.055
        }
    };
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (x, y) in a.iter().zip(b) {
        for channel in 0..3 {
            let difference = (map(x[channel]) - map(y[channel])) as f64;
            sum += difference * difference;
            count += 1;
        }
    }
    let mse = sum / count.max(1) as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

fn encode_srgb(value: f32) -> u8 {
    let mapped = value.max(0.0) / (1.0 + value.max(0.0));
    let encoded = if mapped <= 0.003_130_8 {
        12.92 * mapped
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn save(path: &std::path::Path, pixels: &[[f32; 4]], width: usize, height: usize) {
    let mut image = image::RgbImage::new(width as u32, height as u32);
    for (index, texel) in pixels.iter().enumerate() {
        image.put_pixel(
            (index % width) as u32,
            (index / width) as u32,
            image::Rgb([
                encode_srgb(texel[0]),
                encode_srgb(texel[1]),
                encode_srgb(texel[2]),
            ]),
        );
    }
    let _ = image.save(path);
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();

    let dataset = match train::relight::Dataset::load(std::path::Path::new(&args.dataset)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    // A converted file scores as itself rather than as the glTF it came from.
    // Any difference between the two would be the format losing something, and
    // this is where it would show: against a path traced reference, not against
    // another copy of the same numbers.
    let asset = std::path::Path::new(&args.asset);
    let started = std::time::Instant::now();
    let model = if asset.extension().and_then(|e| e.to_str()) == Some(vol::io::SURFEL_EXTENSION) {
        match vol::io::try_load_relight(asset) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("cannot load {}: {e}", args.asset);
                std::process::exit(1);
            }
        }
    } else {
        let options = convert::ConvertOptions {
            resolution: Some(args.resolution),
            ..Default::default()
        };
        match convert::relight_model_from_gltf(asset, &options) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("cannot convert {}: {e:?}", args.asset);
                std::process::exit(1);
            }
        }
    };
    println!("asset ready in {:.3} s", started.elapsed().as_secs_f64());
    println!(
        "{} surfels over {} materials, against {} views x {} environments at {}x{}",
        model.surfels.len(),
        model.materials.len(),
        dataset.views.len(),
        dataset.environments.len(),
        dataset.width,
        dataset.height
    );

    let context = match unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        })
    } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no ray tracing context: {e:?}");
            std::process::exit(1);
        }
    };
    let info = context.device_information();
    println!("GPU: {}", info.device_name);
    assert!(
        !info.is_software_emulated,
        "refusing to score on a software rasterizer"
    );

    let extent = gpu::Extent {
        width: dataset.width as u32,
        height: dataset.height as u32,
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba32Float;
    let texture = context.create_texture(gpu::TextureDesc {
        name: "relight-quality",
        format,
        size: extent,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let target_view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "relight-quality",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "relight-quality-readback",
        size: (dataset.width * dataset.height) as u64 * 16,
        memory: gpu::Memory::Download,
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "relight-quality",
        buffer_count: 1,
        manual_barriers: false,
    });

    if let Some(ref directory) = args.dump {
        let _ = std::fs::create_dir_all(directory);
    }

    println!(
        "\n{:<14}{:>10}{:>12}{:>12}{:>12}{:>12}",
        "environment", "psnr", "worst", "tonemapped", "worst", "render ms"
    );
    let mut overall = Vec::new();
    for (index, name) in dataset.environments.iter().enumerate() {
        let (texels, width, height) =
            match train::relight::read_environment_plane(&dataset.environment_files[index]) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            };
        let environment = vol::relight::Environment {
            width,
            height,
            texels,
        };
        let specular = vol::relight::SpecularEnvironment::prefilter(&environment, width, height);
        let mut tracer = vol::gpu::RelightTracer::new(
            &model,
            &environment,
            &specular,
            vol::gpu::RelightSettings {
                background_rgb: [0.0; 3],
                diffuse_samples: args.samples,
                // The reference shows the environment behind the object, so
                // this has to as well or the background alone would dominate.
                show_environment: true,
            },
            &context,
            &mut encoder,
        );

        let mut scores = Vec::new();
        let mut display_scores = Vec::new();
        let mut elapsed = std::time::Duration::ZERO;
        for view in &dataset.views {
            let reference = match dataset.read_plane(&view.radiance[index]) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            };

            let started = std::time::Instant::now();
            encoder.start();
            encoder.init_texture(texture);
            tracer.dispatch(
                &mut encoder,
                target_view,
                train::relight::camera_params(view, dataset.width, dataset.height),
                [dataset.width as u32, dataset.height as u32],
            );
            if let mut pass = encoder.transfer("relight-quality-readback") {
                pass.copy_texture_to_buffer(
                    texture.into(),
                    readback.into(),
                    dataset.width as u32 * 16,
                    extent,
                );
            }
            let sync_point = context.submit(&mut encoder);
            assert!(context.wait_for(&sync_point, 60_000).unwrap());
            elapsed += started.elapsed();

            let count = dataset.width * dataset.height;
            let rendered =
                unsafe { std::slice::from_raw_parts(readback.data() as *const [f32; 4], count) }
                    .to_vec();
            scores.push(psnr(&rendered, &reference));
            display_scores.push(tone_mapped_psnr(&rendered, &reference));

            if let Some(ref directory) = args.dump {
                let directory = std::path::Path::new(directory);
                save(
                    &directory.join(format!("{name}-view{}-surfels.png", view.index)),
                    &rendered,
                    dataset.width,
                    dataset.height,
                );
                save(
                    &directory.join(format!("{name}-view{}-reference.png", view.index)),
                    &reference,
                    dataset.width,
                    dataset.height,
                );
            }
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let worst = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let display_mean = display_scores.iter().sum::<f64>() / display_scores.len() as f64;
        let display_worst = display_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "{name:<14}{mean:>10.2}{worst:>12.2}{display_mean:>12.2}{display_worst:>12.2}{:>12.1}",
            elapsed.as_secs_f64() * 1000.0 / scores.len() as f64
        );
        overall.push((mean, display_mean));
        tracer.deinit(&context);
    }

    let count = overall.len() as f64;
    println!(
        "\nmean over environments: {:.2} dB linear, {:.2} dB tone mapped",
        overall.iter().map(|p| p.0).sum::<f64>() / count,
        overall.iter().map(|p| p.1).sum::<f64>() / count,
    );

    context.destroy_buffer(readback);
    context.destroy_texture_view(target_view);
    context.destroy_texture(texture);
    context.destroy_command_encoder(&mut encoder);
}
