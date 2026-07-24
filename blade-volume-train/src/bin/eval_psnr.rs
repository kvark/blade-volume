//! Evaluate PSNR (train + test) of a trained RadFoam PLY against a COLMAP
//! dataset, picking a sensible train/test split from the *files that
//! actually exist on disk* (the default `train_colmap` slicing assumes
//! COLMAP's image order matches filesystem order, which it doesn't on
//! datasets where only a subset of the images were ever provided).
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p blade-volume-train --bin eval_psnr -- \
//!     --ply /tmp/blade-train/bigres1/bonsai.ply \
//!     --sparse etc/data/bonsai/sparse/0 \
//!     --images etc/data/bonsai/images \
//!     --width 128 --height 128 --train 51 --test 8
//! ```
//!
//! Reports the per-view PSNR breakdown and the train/test averages.

use argh::FromArgs;
use blade_volume as vol;
use blade_volume_train::{colmap, diff_render, fit, metrics, pipeline, render};
use std::path;

/// Evaluate PSNR of a trained RadFoam PLY against existing COLMAP images.
#[derive(FromArgs)]
struct Args {
    /// path to the trained PLY
    #[argh(option)]
    ply: String,

    /// path to the COLMAP `sparse/0` directory
    #[argh(option)]
    sparse: String,

    /// path to the COLMAP images directory
    #[argh(option)]
    images: String,

    /// render width
    #[argh(option, default = "128")]
    width: u32,

    /// render height
    #[argh(option, default = "128")]
    height: u32,

    /// number of training views to evaluate (first N existing files in
    /// COLMAP order, after filtering)
    #[argh(option, default = "16")]
    train: usize,

    /// number of test views (next K existing files after the train block)
    #[argh(option, default = "8")]
    test: usize,

    /// far plane for camera params
    #[argh(option, default = "100.0")]
    far_plane: f32,

    /// max ray steps for the path-tracer
    #[argh(option, default = "96")]
    max_steps: usize,

    /// rebuild adjacency from the loaded points (ignore any adjacency
    /// stored in the PLY). Useful for PLYs where positions were
    /// optimised after the saved adjacency was computed.
    #[argh(switch)]
    rebuild_adjacency: bool,

    /// optional directory to dump per-test-view comparison PNGs
    /// (ground-truth | rendered, side by side). Created if absent.
    #[argh(option)]
    dump_dir: Option<String>,

    /// hold out every Nth image as the test set (standard NVS "llffhold"
    /// protocol; Mip-NeRF-360 uses 8), train slice = the rest. Default 0 =
    /// legacy contiguous slicing (--train then --test). Must match the
    /// split the PLY was trained with or train/test numbers are polluted.
    #[argh(option, default = "0")]
    test_every: usize,

    /// composite predictions on white instead of the default black background
    #[argh(switch)]
    white_background: bool,

    /// use the production GPU compute tracer instead of the CPU oracle
    #[argh(switch)]
    gpu: bool,
}

/// Write a side-by-side `[GT | rendered]` PNG. Both inputs are row-major
/// RGB f32 in `[0,1]` of size `w*h*3`.
fn dump_comparison(path: &path::Path, gt: &[f32], pred: &[f32], w: u32, h: u32) {
    let (wu, hu) = (w as usize, h as usize);
    let gap = 8usize;
    let out_w = wu * 2 + gap;
    let mut buf = image::RgbImage::from_pixel(out_w as u32, h, image::Rgb([20, 20, 20]));
    for y in 0..hu {
        for x in 0..wu {
            let i = (y * wu + x) * 3;
            let g = image::Rgb([
                (gt[i].clamp(0.0, 1.0) * 255.0) as u8,
                (gt[i + 1].clamp(0.0, 1.0) * 255.0) as u8,
                (gt[i + 2].clamp(0.0, 1.0) * 255.0) as u8,
            ]);
            let p = image::Rgb([
                (pred[i].clamp(0.0, 1.0) * 255.0) as u8,
                (pred[i + 1].clamp(0.0, 1.0) * 255.0) as u8,
                (pred[i + 2].clamp(0.0, 1.0) * 255.0) as u8,
            ]);
            buf.put_pixel(x as u32, y as u32, g);
            buf.put_pixel((x + wu + gap) as u32, y as u32, p);
        }
    }
    if let Err(e) = buf.save(path) {
        eprintln!("failed to write {}: {e}", path.display());
    }
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();

    let model = vol::io::load_radfoam(&args.ply);
    #[cfg(feature = "qhull")]
    let mut model = model;
    if args.rebuild_adjacency {
        #[cfg(feature = "qhull")]
        {
            eprintln!("rebuilding adjacency from points (Qhull)…");
            model.adjacency = Some(vol::compute_adjacency_qhull_default(&model.points));
        }
        #[cfg(not(feature = "qhull"))]
        {
            eprintln!(
                "--rebuild-adjacency requires building blade-volume-train with --features qhull"
            );
            std::process::exit(2);
        }
    }
    println!(
        "loaded {} ({} cells, sh_degree {}, adjacency {})",
        args.ply,
        model.points.len(),
        model.sh_degree,
        model
            .adjacency
            .as_ref()
            .map(|a| a.neighbors.len())
            .unwrap_or(0),
    );

    let sparse = path::Path::new(&args.sparse);
    let images_dir = path::Path::new(&args.images);
    let recon = colmap::load_reconstruction(sparse);

    // Split via the shared helper (existence-filtered COLMAP order):
    // interleaved every-Nth when --test-every > 0, else the legacy
    // contiguous --train / --test slicing.
    let (train_slice, test_slice) = pipeline::split_train_test(
        &recon,
        images_dir,
        Some(args.train),
        args.test,
        args.test_every,
    );
    println!(
        "split: {} train / {} test views (test_every={})",
        train_slice.len(),
        test_slice.len(),
        args.test_every,
    );
    if test_slice.is_empty() {
        eprintln!(
            "no test views available — train={} test={} test_every={}",
            args.train, args.test, args.test_every,
        );
        std::process::exit(2);
    }

    let config = pipeline::PipelineConfig {
        resolution: (args.width, args.height),
        max_steps: args.max_steps,
        max_views: Some(args.train),
        max_initial_points: Some(model.points.len()),
        far_plane: args.far_plane,
        fit: diff_render::AppearanceFitConfig {
            background_rgb: if args.white_background {
                [1.0; 3]
            } else {
                [0.0; 3]
            },
            ..diff_render::AppearanceFitConfig::default()
        },
        ..pipeline::PipelineConfig::default()
    };

    let train_views =
        pipeline::build_views_from(&recon, images_dir, &config, train_slice.iter().copied());
    let test_views =
        pipeline::build_views_from(&recon, images_dir, &config, test_slice.iter().copied());

    let mut gpu_evaluator = if args.gpu {
        let context = fit::try_init_gpu().unwrap_or_else(|| {
            eprintln!("no supported GPU device — cannot run GPU evaluation");
            std::process::exit(2);
        });
        Some(pipeline::GpuViewEvaluator::new(&model, &config, context))
    } else {
        None
    };
    let train_psnrs = match gpu_evaluator {
        Some(ref mut evaluator) => evaluator
            .evaluate(&train_views, config.fit.background_rgb)
            .unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(3);
            }),
        None => pipeline::evaluate_views(&model, &train_views, &config),
    };
    let test_psnrs = match gpu_evaluator {
        Some(ref mut evaluator) => evaluator
            .evaluate(&test_views, config.fit.background_rgb)
            .unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(3);
            }),
        None => pipeline::evaluate_views(&model, &test_views, &config),
    };

    let avg_train: f32 = train_psnrs.iter().copied().sum::<f32>() / train_psnrs.len() as f32;
    let avg_test: f32 = test_psnrs.iter().copied().sum::<f32>() / test_psnrs.len() as f32;
    println!(
        "PSNR train (avg over {} views): {avg_train:.2} dB",
        train_psnrs.len()
    );
    println!(
        "PSNR test  (avg over {} views): {avg_test:.2} dB",
        test_psnrs.len()
    );
    for (img, p) in test_slice.iter().zip(test_psnrs.iter()) {
        println!("  test {}: {:.2} dB", img.name, p);
    }

    if let Some(ref dir) = args.dump_dir {
        let dir = path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("could not create dump dir {}: {e}", dir.display());
        } else {
            for (img, view) in test_slice.iter().zip(test_views.iter()) {
                let rgba = match gpu_evaluator {
                    Some(ref mut evaluator) => {
                        evaluator.render_rgba(view.camera).unwrap_or_else(|err| {
                            eprintln!("{err}");
                            std::process::exit(3);
                        })
                    }
                    None => render::render_cpu(
                        &model,
                        &view.camera,
                        render::RenderSettings {
                            width: args.width,
                            height: args.height,
                            start_point: pipeline::pick_start_cell(
                                &model,
                                glam::Vec3::from_array(view.camera.cam_position),
                            ),
                            max_steps: args.max_steps as u32,
                            weight_threshold: 1e-4,
                        },
                    ),
                };
                let pred = metrics::rgba_over_background(&rgba, config.fit.background_rgb);
                let stem = path::Path::new(&img.name)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| img.name.clone());
                let out = dir.join(format!("cmp_{stem}.png"));
                dump_comparison(&out, &view.target_rgb, &pred, args.width, args.height);
                println!("  wrote {}", out.display());
            }
        }
    }
    if let Some(evaluator) = gpu_evaluator {
        evaluator.deinit();
    }
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn gpu_evaluation_is_opt_in() {
        let base = [
            "--ply",
            "model.ply",
            "--sparse",
            "sparse",
            "--images",
            "images",
        ];
        let default = <Args as argh::FromArgs>::from_args(&["eval_psnr"], &base).unwrap();
        assert!(!default.gpu);

        let mut explicit = base.to_vec();
        explicit.push("--gpu");
        let gpu = <Args as argh::FromArgs>::from_args(&["eval_psnr"], &explicit).unwrap();
        assert!(gpu.gpu);
    }
}
