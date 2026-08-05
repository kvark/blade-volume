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
use std::{collections, path};

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
    #[argh(option, default = "128")]
    max_steps: usize,

    /// minimum PowerFoam sphere candidates per ray (default 0 = automatic:
    /// max of four times --max-steps and 1024)
    #[argh(option, default = "0")]
    powerfoam_candidate_capacity: u32,

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

    /// trace and attribute this many worst held-out overprediction pixels and
    /// this many worst underprediction pixels (default 0 = off). Rendering can
    /// remain on the GPU; only the selected diagnostic rays use the CPU oracle.
    #[argh(option, default = "0")]
    diagnose_worst: usize,
}

#[derive(Clone, Copy, Debug)]
struct PixelError {
    view: usize,
    pixel: usize,
    score: f32,
    mse: f32,
    target: [f32; 3],
    prediction: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
enum ErrorKind {
    Overprediction,
    Underprediction,
}

fn top_error_pixels(
    target: &[f32],
    prediction: &[f32],
    view: usize,
    count: usize,
    kind: ErrorKind,
) -> Vec<PixelError> {
    assert_eq!(target.len(), prediction.len());
    assert_eq!(target.len() % 3, 0);
    let mut errors = Vec::with_capacity(target.len() / 3);
    for (pixel, (target_rgb, prediction_rgb)) in target
        .chunks_exact(3)
        .zip(prediction.chunks_exact(3))
        .enumerate()
    {
        let mut score = 0.0_f32;
        let mut mse = 0.0_f32;
        for channel in 0..3 {
            let delta = prediction_rgb[channel] - target_rgb[channel];
            let directional = match kind {
                ErrorKind::Overprediction => delta.max(0.0),
                ErrorKind::Underprediction => (-delta).max(0.0),
            };
            score += directional * directional;
            mse += delta * delta;
        }
        errors.push(PixelError {
            view,
            pixel,
            score: score / 3.0,
            mse: mse / 3.0,
            target: [target_rgb[0], target_rgb[1], target_rgb[2]],
            prediction: [prediction_rgb[0], prediction_rgb[1], prediction_rgb[2]],
        });
    }
    errors.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.pixel.cmp(&right.pixel))
    });
    errors.truncate(count.min(errors.len()));
    errors
}

fn merge_top_errors(errors: &mut Vec<PixelError>, next: Vec<PixelError>, count: usize) {
    errors.extend(next);
    errors.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.view.cmp(&right.view))
            .then_with(|| left.pixel.cmp(&right.pixel))
    });
    errors.truncate(count.min(errors.len()));
}

fn finite_quantile(values: impl Iterator<Item = f32>, quantile: f32) -> f32 {
    let mut sorted: Vec<f32> = values.filter(|value| value.is_finite()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_unstable_by(f32::total_cmp);
    let position = quantile * (sorted.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f32;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

fn support_radii(model: &vol::PointCloudModel) -> Vec<f32> {
    if let Some(ref radii) = model.radii {
        return radii.clone();
    }
    let mut radii = vec![0.0_f32; model.points.len()];
    let Some(ref adjacency) = model.adjacency else {
        return radii;
    };
    for (index, radius) in radii.iter_mut().enumerate() {
        let point = model.points[index].truncate();
        let start = adjacency.offsets[index] as usize;
        let end = adjacency.offsets[index + 1] as usize;
        *radius = adjacency.neighbors[start..end]
            .iter()
            .map(|&neighbor| (model.points[neighbor as usize].truncate() - point).length())
            .fold(0.0_f32, f32::max);
    }
    radii
}

fn trace_pixel(
    model: &vol::PointCloudModel,
    view: &diff_render::ViewSupervision,
    width: u32,
    height: u32,
    max_steps: usize,
    pixel: usize,
) -> vol::trace::TraceResult {
    let x = pixel as u32 % width;
    let y = pixel as u32 / width;
    let ray = render::camera_ray(&view.camera, width, height, x, y);
    let settings = vol::trace::TraceSettings {
        weight_threshold: 1.0e-4,
        max_steps: max_steps as u32,
        start_point: pipeline::pick_start_cell(model, ray.origin),
        depth: view.camera.depth,
        eval_mode: vol::trace::EvalMode::Sh,
    };
    if model.radii.is_some() {
        vol::trace::trace_powerfoam_splats(model, ray, settings)
    } else {
        vol::trace::trace_one_ray(model, ray, settings)
    }
}

fn print_error_diagnostics(
    label: &str,
    errors: &[PixelError],
    names: &[&str],
    views: &[diff_render::ViewSupervision],
    model: &vol::PointCloudModel,
    radii: &[f32],
    contribution: Option<&diff_render::PathContributionStats>,
    width: u32,
    height: u32,
    max_steps: usize,
) {
    let adjacency = model
        .adjacency
        .as_ref()
        .expect("reconstruction diagnostics require adjacency");
    let global_radius_p90 = finite_quantile(radii.iter().copied(), 0.9);
    let global_radius_p99 = finite_quantile(radii.iter().copied(), 0.99);
    let global_density_p90 = finite_quantile(model.points.iter().map(|point| point.w), 0.9);
    let global_density_p99 = finite_quantile(model.points.iter().map(|point| point.w), 0.99);
    let global_degree_p90 = finite_quantile(
        adjacency
            .offsets
            .windows(2)
            .map(|range| (range[1] - range[0]) as f32),
        0.9,
    );
    let global_degree_p99 = finite_quantile(
        adjacency
            .offsets
            .windows(2)
            .map(|range| (range[1] - range[0]) as f32),
        0.99,
    );
    let mut responsible = collections::BTreeSet::new();
    let mut pixel_radii = Vec::with_capacity(errors.len());
    let mut pixel_densities = Vec::with_capacity(errors.len());
    let mut pixel_degrees = Vec::with_capacity(errors.len());
    let mut peak_weights = Vec::with_capacity(errors.len());
    let mut opacities = Vec::with_capacity(errors.len());
    let mut training_support = Vec::with_capacity(errors.len());
    let mut large_radius = 0_usize;
    let mut high_degree = 0_usize;

    println!(
        "diagnostic_pixel\tkind\trank\timage\tx\ty\tdirectional_mse\ttotal_mse\t\
         target_rgb\tprediction_rgb\topacity\tpeak_weight\tdepth_mode\tsteps\tpeak_cell\t\
         radius\tdensity\tdegree\ttraining_max_weight\ttraining_views\tposition"
    );
    for (rank, error) in errors.iter().enumerate() {
        let result = trace_pixel(
            model,
            &views[error.view],
            width,
            height,
            max_steps,
            error.pixel,
        );
        let x = error.pixel as u32 % width;
        let y = error.pixel as u32 / width;
        let (cell, radius, density, degree, training_max, training_views, position) = match result
            .peak_point
        {
            Some(cell) => {
                let index = cell as usize;
                let start = adjacency.offsets[index];
                let end = adjacency.offsets[index + 1];
                let point = model.points[index];
                responsible.insert(cell);
                pixel_radii.push(radii[index]);
                pixel_densities.push(point.w);
                pixel_degrees.push((end - start) as f32);
                large_radius += usize::from(radii[index] >= global_radius_p99);
                high_degree += usize::from((end - start) as f32 >= global_degree_p99);
                let (training_max, training_views) = contribution.map_or((f32::NAN, 0), |stats| {
                    (stats.per_cell[index], stats.supporting_views[index])
                });
                if contribution.is_some() {
                    training_support.push(training_views as f32);
                }
                (
                    cell.to_string(),
                    radii[index],
                    point.w,
                    end - start,
                    training_max,
                    training_views,
                    format!("{:.6},{:.6},{:.6}", point.x, point.y, point.z),
                )
            }
            None => (
                String::from("none"),
                0.0,
                0.0,
                0,
                f32::NAN,
                0,
                String::from("none"),
            ),
        };
        peak_weights.push(result.peak_weight);
        opacities.push(result.rgba.w);
        println!(
            "diagnostic_pixel\t{label}\t{}\t{}\t{x}\t{y}\t{:.8}\t{:.8}\t\
             {:.6},{:.6},{:.6}\t{:.6},{:.6},{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t\
             {:.6}\t{:.6}\t{}\t{:.6}\t{}\t{}",
            rank + 1,
            names[error.view],
            error.score,
            error.mse,
            error.target[0],
            error.target[1],
            error.target[2],
            error.prediction[0],
            error.prediction[1],
            error.prediction[2],
            result.rgba.w,
            result.peak_weight,
            result.depth_mode,
            result.steps,
            cell,
            radius,
            density,
            degree,
            training_max,
            training_views,
            position,
        );
    }
    println!(
        "diagnostic_summary {label}: {} pixels, {} attributed, {} unique peak cells; \
         radius p50/p90/p99 {:.6}/{:.6}/{:.6} (global p90/p99 {:.6}/{:.6}; {} at global p99); \
         density p50/p90/p99 {:.6}/{:.6}/{:.6} (global p90/p99 {:.6}/{:.6}); \
         degree p50/p90/p99 {:.1}/{:.1}/{:.1} (global p90/p99 {:.1}/{:.1}; {} at global p99); \
         peak weight p50/p90 {:.4}/{:.4}; opacity p50/p90 {:.4}/{:.4}",
        errors.len(),
        pixel_radii.len(),
        responsible.len(),
        finite_quantile(pixel_radii.iter().copied(), 0.5),
        finite_quantile(pixel_radii.iter().copied(), 0.9),
        finite_quantile(pixel_radii.iter().copied(), 0.99),
        global_radius_p90,
        global_radius_p99,
        large_radius,
        finite_quantile(pixel_densities.iter().copied(), 0.5),
        finite_quantile(pixel_densities.iter().copied(), 0.9),
        finite_quantile(pixel_densities.iter().copied(), 0.99),
        global_density_p90,
        global_density_p99,
        finite_quantile(pixel_degrees.iter().copied(), 0.5),
        finite_quantile(pixel_degrees.iter().copied(), 0.9),
        finite_quantile(pixel_degrees.iter().copied(), 0.99),
        global_degree_p90,
        global_degree_p99,
        high_degree,
        finite_quantile(peak_weights.iter().copied(), 0.5),
        finite_quantile(peak_weights.iter().copied(), 0.9),
        finite_quantile(opacities.iter().copied(), 0.5),
        finite_quantile(opacities.iter().copied(), 0.9),
    );
    if let Some(stats) = contribution {
        let global_supported_p50 = finite_quantile(
            stats.supporting_views.iter().map(|&count| count as f32),
            0.5,
        );
        let global_supported_p90 = finite_quantile(
            stats.supporting_views.iter().map(|&count| count as f32),
            0.9,
        );
        let global_supported_p99 = finite_quantile(
            stats.supporting_views.iter().map(|&count| count as f32),
            0.99,
        );
        println!(
            "diagnostic_support_summary {label}: responsible training-view count p50/p90/p99 \
             {:.1}/{:.1}/{:.1}; global p50/p90/p99 {:.1}/{:.1}/{:.1}; global cells in 0/1/<4 views {}/{}/{}; \
             collector {} rays, {:.1} mean segments, max {}, {} truncated",
            finite_quantile(training_support.iter().copied(), 0.5),
            finite_quantile(training_support.iter().copied(), 0.9),
            finite_quantile(training_support.iter().copied(), 0.99),
            global_supported_p50,
            global_supported_p90,
            global_supported_p99,
            stats.supporting_views.iter().filter(|&&count| count == 0).count(),
            stats.supporting_views.iter().filter(|&&count| count == 1).count(),
            stats.supporting_views.iter().filter(|&&count| count < 4).count(),
            stats.rays,
            stats.segments as f32 / stats.rays.max(1) as f32,
            stats.max_steps_used,
            stats.truncated_rays,
        );
    }
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
            powerfoam_candidate_capacity: args.powerfoam_candidate_capacity,
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

    let gpu_context = if args.gpu {
        Some(fit::try_init_gpu().unwrap_or_else(|| {
            eprintln!("no supported GPU device — cannot run GPU evaluation");
            std::process::exit(2);
        }))
    } else {
        None
    };
    let mut gpu_evaluator = gpu_context
        .as_ref()
        .map(|context| pipeline::GpuViewEvaluator::new(&model, &config, context.clone()));
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
        "PSNR train (avg over {} views): {avg_train:.4} dB",
        train_psnrs.len()
    );
    println!(
        "PSNR test  (avg over {} views): {avg_test:.4} dB",
        test_psnrs.len()
    );
    for (img, p) in test_slice.iter().zip(test_psnrs.iter()) {
        println!("  test {}: {:.2} dB", img.name, p);
    }

    let dump_dir = args.dump_dir.as_deref().and_then(|dir| {
        let path = path::Path::new(dir);
        match std::fs::create_dir_all(path) {
            Ok(()) => Some(path),
            Err(err) => {
                eprintln!("could not create dump dir {}: {err}", path.display());
                None
            }
        }
    });
    let mut overprediction = Vec::new();
    let mut underprediction = Vec::new();
    if dump_dir.is_some() || args.diagnose_worst > 0 {
        for (view_index, (img, view)) in test_slice.iter().zip(&test_views).enumerate() {
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
            let mut prediction = metrics::rgba_over_background(&rgba, config.fit.background_rgb);
            for value in &mut prediction {
                *value = value.clamp(0.0, 1.0);
            }
            if let Some(dir) = dump_dir {
                let stem = path::Path::new(&img.name)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| img.name.clone());
                let out = dir.join(format!("cmp_{stem}.png"));
                dump_comparison(&out, &view.target_rgb, &prediction, args.width, args.height);
                println!("  wrote {}", out.display());
            }
            if args.diagnose_worst > 0 {
                merge_top_errors(
                    &mut overprediction,
                    top_error_pixels(
                        &view.target_rgb,
                        &prediction,
                        view_index,
                        args.diagnose_worst,
                        ErrorKind::Overprediction,
                    ),
                    args.diagnose_worst,
                );
                merge_top_errors(
                    &mut underprediction,
                    top_error_pixels(
                        &view.target_rgb,
                        &prediction,
                        view_index,
                        args.diagnose_worst,
                        ErrorKind::Underprediction,
                    ),
                    args.diagnose_worst,
                );
            }
        }
    }
    if let Some(evaluator) = gpu_evaluator.take() {
        evaluator.deinit();
    }
    if args.diagnose_worst > 0 {
        let contribution = gpu_context.as_ref().map(|context| {
            diff_render::measure_path_contributions(
                context,
                &model,
                &train_views,
                args.max_steps,
                args.powerfoam_candidate_capacity,
                0.01,
            )
        });
        let radii = support_radii(&model);
        let names: Vec<&str> = test_slice.iter().map(|image| image.name.as_str()).collect();
        print_error_diagnostics(
            "overprediction",
            &overprediction,
            &names,
            &test_views,
            &model,
            &radii,
            contribution.as_ref(),
            args.width,
            args.height,
            args.max_steps,
        );
        print_error_diagnostics(
            "underprediction",
            &underprediction,
            &names,
            &test_views,
            &model,
            &radii,
            contribution.as_ref(),
            args.width,
            args.height,
            args.max_steps,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{top_error_pixels, Args, ErrorKind};

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
        assert_eq!(default.max_steps, 128);
        assert_eq!(default.diagnose_worst, 0);

        let mut explicit = base.to_vec();
        explicit.push("--gpu");
        explicit.extend(["--diagnose-worst", "17"]);
        let gpu = <Args as argh::FromArgs>::from_args(&["eval_psnr"], &explicit).unwrap();
        assert!(gpu.gpu);
        assert_eq!(gpu.diagnose_worst, 17);
    }

    #[test]
    fn directional_errors_separate_floaters_from_holes() {
        let target = [0.1, 0.2, 0.3, 0.8, 0.7, 0.6];
        let prediction = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0];
        let over = top_error_pixels(&target, &prediction, 3, 1, ErrorKind::Overprediction);
        let under = top_error_pixels(&target, &prediction, 3, 1, ErrorKind::Underprediction);
        assert_eq!((over[0].view, over[0].pixel), (3, 0));
        assert_eq!((under[0].view, under[0].pixel), (3, 1));
        assert!(over[0].score > 0.0 && under[0].score > 0.0);
    }
}
