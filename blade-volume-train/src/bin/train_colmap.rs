//! Train a foam model from a COLMAP sparse reconstruction.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p blade-volume-train --bin train_colmap -- \
//!     --sparse etc/data/bonsai/sparse/0 \
//!     --images etc/data/bonsai/images \
//!     --output bonsai.ply \
//!     --novel-out bonsai_novel.png \
//!     --width 32 --height 32 --views 8 --epochs 100
//! ```
//!
//! What this does:
//! 1. Loads `cameras.bin`, `images.bin`, `points3D.bin` from `--sparse`.
//! 2. Builds an initial `PointCloudModel` from the sparse 3D points (SH
//!    degree 0 from COLMAP RGB, uniform initial density).
//! 3. Computes Delaunay adjacency.
//! 4. For up to `--views` images, loads the file under `--images/`,
//!    downsamples to `(--width, --height)`, and uses it as a training view.
//! 5. Runs multi-view point-cloud training (Adam, L1 against per-pixel RGB).
//!    Geometry is fixed unless position optimisation is explicitly enabled.
//! 6. Saves the trained model as a RadFoam PLY at `--output`.
//! 7. Renders the trained model from a *novel* pose interpolated between
//!    the first two training cameras and saves it as `--novel-out`.

use argh::FromArgs;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train::{diff_render, fit, pipeline};
use std::path;

/// Train a RadFoam from a COLMAP sparse reconstruction.
#[derive(FromArgs)]
struct Args {
    /// path to the COLMAP `sparse/0` directory
    #[argh(option)]
    sparse: String,

    /// path to the COLMAP images directory
    #[argh(option)]
    images: String,

    /// output PLY path for the trained foam
    #[argh(option)]
    output: String,

    /// optional output PNG for a novel-pose render
    #[argh(option)]
    novel_out: Option<String>,

    /// optional output prefix for a strip of novel poses (writes <prefix>_NN.png)
    #[argh(option)]
    novel_strip_prefix: Option<String>,

    /// how many novel poses to render in the strip (default 5)
    #[argh(option, default = "5")]
    novel_strip_count: u32,

    /// training image width (default 32)
    #[argh(option, default = "32")]
    width: u32,

    /// training image height (default 32)
    #[argh(option, default = "32")]
    height: u32,

    /// max number of training views (default 8)
    #[argh(option, default = "8")]
    views: usize,

    /// held-out images for PSNR evaluation (default 0 = no test set)
    #[argh(option, default = "0")]
    test_views: usize,

    /// training epochs (default 100)
    #[argh(option, default = "100")]
    epochs: usize,

    /// max traversal steps per ray (default 128). With 100K cells in a
    /// bonsai-scale scene, an average ray crosses ~50-80 cell boundaries
    /// at 64² resolution; higher resolution / more cells need more
    /// steps. Path truncation throws away the far end of the integral.
    #[argh(option, default = "128")]
    max_steps: usize,

    /// adam learning rate (default 0.1)
    #[argh(option, default = "0.1")]
    learning_rate: f32,

    /// cap on initial COLMAP points (default 2000; Delaunay scales poorly past this)
    #[argh(option, default = "2000")]
    max_points: usize,

    /// use Qhull instead of the default `simple_delaunay_lib` Delaunay
    /// backend. Qhull scales better on typical large clouds, though exact
    /// 3D Delaunay has quadratic worst-case output. Both produce exact
    /// Voronoi adjacency, so traversal remains correct.
    #[argh(switch)]
    qhull: bool,

    /// k for symmetric k-NN adjacency (default 0 = use Delaunay).
    /// Approximate; useful for non-radfoam workloads only — the
    /// path-tracer pays ~3 dB of PSNR for the wrong face partners.
    #[argh(option, default = "0")]
    knn: usize,

    /// radius factor for Čech adjacency (default 0 = off). Approximate;
    /// same caveat as `--knn` re: path-tracer correctness.
    #[argh(option, default = "0.0")]
    cech_radius: f32,

    /// pixels per Adam step (default 256). Random pixel sampling keeps the
    /// graph small regardless of image resolution. Set 0 to use every pixel.
    #[argh(option, default = "256")]
    pixel_batch: usize,

    /// adam steps per view in pixel-batched mode (default 200)
    #[argh(option, default = "200")]
    steps_per_view: usize,

    /// initial uniform per-cell density before training (default 1.0)
    #[argh(option, default = "1.0")]
    initial_density: f32,

    /// SH degree for view-dependent colour (0–3, default 0).
    /// Higher degrees give better PSNR on view-dependent surfaces
    /// (metallic, glossy, leafy) at the cost of `(1+deg)²` per-cell
    /// parameters per channel.
    #[argh(option, default = "0")]
    sh_degree: usize,

    /// adaptive densification: cells between splits (default 0 = off).
    /// Recommended 500–1000. After every cycle the top
    /// `--densify-fraction` cells by accumulated position-gradient magnitude
    /// times cell radius
    /// are split — each parent gets a sibling cell at `pos + jitter`
    /// with the same density and SH coefficients, and adjacency is
    /// rebuilt with the configured exact backend. Requires `--pixel-batch`.
    #[argh(option, default = "0")]
    densify_every: usize,

    /// per-round growth: each densify round adds `fraction × current`
    /// cells, parents drawn by weighted multinomial on
    /// `|grad(position)| × cell_radius` (RadFoam uses 0.15). Ignored
    /// when `--densify-every 0`.
    #[argh(option, default = "0.15")]
    densify_fraction: f32,

    /// point budget: stop growing once the cloud reaches this many cells
    /// (RadFoam bonsai ≈ 2,097,152). Default 2_000_000.
    #[argh(option, default = "2000000")]
    densify_target: usize,

    /// stop densifying after this step (refinement-only afterwards).
    /// Default 0 = densify until the end of training.
    #[argh(option, default = "0")]
    densify_until: usize,

    /// prune small near-transparent cells each densify round (RadFoam's
    /// floater remover). Off by default for backward compatibility.
    #[argh(switch)]
    prune: bool,

    /// prune a cell only if its post-activation density is below this
    /// (default 0.01). Requires `--prune`.
    #[argh(option, default = "0.01")]
    prune_density: f32,

    /// also require farthest-neighbour radius below this (default 0.1) to
    /// prune — only cull small dim cells, never large empty background ones.
    #[argh(option, default = "0.1")]
    prune_radius: f32,

    /// per-cell jitter for split siblings, in units of the parent's
    /// nearest-neighbour distance (default 0.5). A small fraction of
    /// the cell scale keeps the new cell inside the parent's Voronoi
    /// neighbourhood; too large and it pops out into unrelated
    /// territory.
    #[argh(option, default = "0.5")]
    densify_jitter: f32,

    /// steps to wait before the first densify cycle (default 500). Lets
    /// the per-cell gradient signal settle from random init before
    /// committing to splits.
    #[argh(option, default = "500")]
    densify_warmup: usize,

    /// learning-rate schedule: "constant" or "cosine" (default cosine).
    /// Cosine decays from `--learning-rate` to `learning_rate * --lr-min-ratio`
    /// over the full Adam-step budget.
    #[argh(option, default = "String::from(\"cosine\")")]
    lr_schedule: String,

    /// cosine-decay floor as a fraction of `--learning-rate` (default 0.01).
    /// At step `total_steps` the effective LR equals `learning_rate * lr_min_ratio`.
    #[argh(option, default = "0.01")]
    lr_min_ratio: f32,

    /// patch-based sampling + structural-gradient L1 loss. `0` (default) =
    /// random-pixel L1 only; `> 0` = each Adam step samples a single
    /// `--patch-size × --patch-size` contiguous patch (requires
    /// `--pixel-batch == --patch-size²`) and the loss becomes
    /// `(1 - --grad-loss-weight) * L1 + --grad-loss-weight * L1_grad`,
    /// where `L1_grad` is the L1 distance between rendered and target
    /// finite-difference gradients. Captures edge structure that
    /// random-pixel L1 misses.
    #[argh(option, default = "0")]
    patch_size: usize,

    /// weight on the structural-gradient L1 term when `--patch-size > 0`.
    #[argh(option, default = "0.2")]
    grad_loss_weight: f32,

    /// weight on the RadFoam opacity loss `mean((opacity-1)^2)` (default 0 =
    /// off). Pushes rays to full opacity and suppresses
    /// semi-transparent floaters independently of the background choice.
    #[argh(option, default = "0.0")]
    opacity_weight: f32,

    /// weight on smooth per-ray depth variance (default 0 = off). Small
    /// values such as 1e-4 discourage floaters and thick multi-surface rays.
    #[argh(option, default = "0.0")]
    distortion_weight: f32,

    /// composite training, held-out evaluation, and novel renders on white
    /// instead of the default black background
    #[argh(switch)]
    white_background: bool,

    /// softplus beta for density activation (default 0 = legacy ReLU).
    /// RadFoam uses 10. Keeps a gradient for negative log-density so cells
    /// recover instead of dying (dead-ReLU) — stabilises densification.
    #[argh(option, default = "0.0")]
    softplus_beta: f32,

    /// position learning rate as a fraction of the main learning rate
    /// (default 0 = fixed geometry). Experimental: validate on a held-out
    /// split before using for production checkpoints.
    #[argh(option, default = "0.0")]
    position_lr_ratio: f32,

    /// steps between adjacency/path rebuilds during position optimisation
    /// (default 100). Ignored when --position-lr-ratio is 0.
    #[argh(option, default = "100")]
    geometry_rebuild_every: usize,

    /// optional checkpoint PLY path written with an exact safetensors optimizer
    /// sidecar at every densify cycle. Defaults to <output>.ckpt.ply when
    /// --densify-every > 0; pass "none" to disable.
    #[argh(option)]
    checkpoint: Option<String>,

    /// resume from a checkpoint PLY. A sibling `.safetensors` file is loaded
    /// automatically when present to restore exact parameters and Adam state.
    #[argh(option)]
    init_ply: Option<String>,

    /// resume: absolute step to continue the LR/densify schedule from.
    /// When omitted and --init-ply is set, read it from `<init-ply>.step`.
    #[argh(option)]
    resume_step: Option<usize>,

    /// hold out every Nth image for testing (standard NVS "llffhold"
    /// protocol; Mip-NeRF-360 uses 8) and train on the rest. Default 0 =
    /// legacy contiguous split (first --views train, next --test-views
    /// test), which on filename-ordered captures tests on a tail arc of
    /// the trajectory (mostly extrapolation) and is not comparable to
    /// published numbers. --test-views caps the held-out count when > 0.
    #[argh(option, default = "0")]
    test_every: usize,
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();

    let Some(gpu) = fit::try_init_gpu() else {
        eprintln!("no supported GPU device — cannot train");
        std::process::exit(2);
    };

    let pixel_batch = if args.pixel_batch == 0 {
        None
    } else {
        Some(args.pixel_batch)
    };
    let adjacency = if args.qhull {
        pipeline::AdjacencyKind::DelaunayQhull
    } else if args.cech_radius > 0.0 {
        pipeline::AdjacencyKind::Cech {
            radius_factor: args.cech_radius,
        }
    } else if args.knn > 0 {
        pipeline::AdjacencyKind::Knn(args.knn)
    } else {
        pipeline::AdjacencyKind::Delaunay
    };
    let lr_schedule = match args.lr_schedule.as_str() {
        "constant" => diff_render::LrSchedule::Constant,
        "cosine" => diff_render::LrSchedule::Cosine,
        other => {
            eprintln!("unknown --lr-schedule '{other}' (use 'constant' or 'cosine')");
            std::process::exit(2);
        }
    };
    // Resume bookkeeping: when --init-ply is given, determine the step to
    // continue the schedule from — explicit --resume-step wins, else the
    // `<ply>.step` sidecar the checkpoint writer dropped, else 0.
    let init_ply = args.init_ply.as_deref().map(path::PathBuf::from);
    let resume_step = match args.resume_step {
        Some(s) => s,
        None => match init_ply {
            Some(ref ply) => {
                let sidecar = ply.with_extension("ply.step");
                match std::fs::read_to_string(&sidecar)
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                {
                    Some(s) => {
                        eprintln!("resume: read step {s} from {}", sidecar.display());
                        s
                    }
                    None => {
                        eprintln!(
                            "resume: no readable step sidecar at {}, starting schedule at 0",
                            sidecar.display()
                        );
                        0
                    }
                }
            }
            None => 0,
        },
    };
    let resume_state_path = init_ply.as_ref().and_then(|ply| {
        let state = ply.with_extension("safetensors");
        state.is_file().then_some(state)
    });

    let config = pipeline::PipelineConfig {
        resolution: (args.width, args.height),
        max_steps: args.max_steps,
        max_views: Some(args.views),
        max_initial_points: Some(args.max_points),
        fit: diff_render::AppearanceFitConfig {
            learning_rate: args.learning_rate,
            epochs: args.epochs,
            pixel_batch,
            steps_per_view: args.steps_per_view,
            sh_degree: args.sh_degree,
            lr_schedule,
            lr_min_ratio: args.lr_min_ratio,
            patch_size: args.patch_size,
            grad_loss_weight: args.grad_loss_weight,
            opacity_weight: args.opacity_weight,
            distortion_weight: args.distortion_weight,
            softplus_beta: args.softplus_beta,
            background_rgb: if args.white_background {
                [1.0; 3]
            } else {
                [0.0; 3]
            },
            position_lr_ratio: args.position_lr_ratio,
            geometry_rebuild_every: args.geometry_rebuild_every,
            rebuild_with_qhull: args.qhull,
            resume_step,
            resume_state_path,
            checkpoint_path: match args.checkpoint.as_deref() {
                Some("none") => None,
                Some(p) => Some(path::PathBuf::from(p)),
                // Default: checkpoint next to the output when densifying.
                None if args.densify_every > 0 => {
                    Some(path::Path::new(&args.output).with_extension("ckpt.ply"))
                }
                None => None,
            },
            densify: if args.densify_every > 0 {
                Some(diff_render::DensifyConfig {
                    every: args.densify_every,
                    fraction: args.densify_fraction,
                    jitter_scale: args.densify_jitter,
                    warmup: args.densify_warmup,
                    target_points: args.densify_target,
                    densify_until: if args.densify_until == 0 {
                        usize::MAX
                    } else {
                        args.densify_until
                    },
                    prune: args.prune,
                    prune_density: args.prune_density,
                    prune_radius: args.prune_radius,
                })
            } else {
                None
            },
            ..diff_render::AppearanceFitConfig::default()
        },
        far_plane: 100.0,
        initial_density: args.initial_density,
        adjacency,
        init_ply,
        test_every: args.test_every,
    };

    let sparse = path::Path::new(&args.sparse);
    let images = path::Path::new(&args.images);
    let outcome = pipeline::train_colmap_appearance_split(
        sparse,
        images,
        &config,
        args.test_views,
        gpu.clone(),
    );

    let output = path::Path::new(&args.output);
    convert::save_ply_with_options(
        output,
        &outcome.model,
        &convert::SaveOptions {
            format: convert::PlyFormat::Binary,
        },
    )
    .unwrap_or_else(|err| {
        eprintln!("failed to save PLY: {err:?}");
        std::process::exit(3);
    });
    println!(
        "wrote {} ({} points, {} adjacency edges)",
        output.display(),
        outcome.model.len(),
        outcome
            .model
            .adjacency
            .as_ref()
            .map(|a| a.neighbors.len())
            .unwrap_or(0),
    );

    // --- Evaluation on held-out test views ---
    if args.test_views > 0 || args.test_every > 0 {
        // Same split logic as training (existence-filtered COLMAP order):
        // interleaved every-Nth when --test-every > 0, else the legacy
        // contiguous first-train/next-test slicing.
        let (train_images, test_images) = pipeline::split_train_test(
            &outcome.reconstruction,
            images,
            config.max_views,
            args.test_views,
            args.test_every,
        );
        let test_views = pipeline::build_views_from(
            &outcome.reconstruction,
            images,
            &outcome.model,
            &config,
            test_images.iter().copied(),
        );
        if test_views.is_empty() {
            eprintln!("no usable test views (image files missing?)");
        } else {
            let psnrs = pipeline::evaluate_views(&outcome.model, &test_views, &config);
            // Train-set PSNR too, for comparison.
            let train_views = pipeline::build_views_from(
                &outcome.reconstruction,
                images,
                &outcome.model,
                &config,
                train_images.iter().copied(),
            );
            let train_psnrs = pipeline::evaluate_views(&outcome.model, &train_views, &config);
            let avg_train: f32 =
                train_psnrs.iter().copied().sum::<f32>() / train_psnrs.len() as f32;
            let avg_test: f32 = psnrs.iter().copied().sum::<f32>() / psnrs.len() as f32;
            println!(
                "PSNR train (avg over {} views): {avg_train:.2} dB",
                train_psnrs.len()
            );
            println!(
                "PSNR test  (avg over {} views): {avg_test:.2} dB",
                psnrs.len()
            );
            for (img, p) in test_images.iter().zip(psnrs.iter()) {
                println!("  test {}: {:.2} dB", img.name, p);
            }
        }
    }

    if let Some(ref novel_path) = args.novel_out {
        render_novel_at(&outcome.model, sparse, &config, 0.5, novel_path);
    }
    if let Some(ref prefix) = args.novel_strip_prefix {
        let n = args.novel_strip_count.max(2);
        for k in 0..n {
            let t = k as f32 / (n - 1) as f32;
            let path = format!("{prefix}_{k:02}.png");
            render_novel_at(&outcome.model, sparse, &config, t, &path);
        }
    }
}

/// Render the trained model from a pose interpolated between the first two
/// training cameras at parameter `t` in `[0, 1]`. `t = 0` reproduces the first
/// training view, `t = 1` reproduces the second, anything in between is a
/// genuinely novel pose.
fn render_novel_at(
    model: &vol::PointCloudModel,
    sparse: &path::Path,
    config: &pipeline::PipelineConfig,
    t: f32,
    novel_path: &str,
) {
    let recon = blade_volume_train::colmap::load_reconstruction(sparse);
    if recon.images.len() < 2 {
        eprintln!(
            "need at least 2 images to interpolate a novel view; got {}",
            recon.images.len()
        );
        return;
    }
    let cam_a = recon.camera_params_for(&recon.images[0], config.far_plane);
    let cam_b = recon.camera_params_for(&recon.images[1], config.far_plane);
    let novel = interp_camera(&cam_a, &cam_b, t);

    let settings = blade_volume_train::render::RenderSettings {
        width: config.resolution.0,
        height: config.resolution.1,
        start_point: pipeline::pick_start_cell(model, glam::Vec3::from_array(novel.cam_position)),
        max_steps: config.max_steps as u32,
        weight_threshold: 1e-4,
    };
    let pixels = blade_volume_train::render::render_cpu(model, &novel, settings);
    let rgb = blade_volume_train::metrics::rgba_over_background(&pixels, config.fit.background_rgb);

    let w = config.resolution.0;
    let h = config.resolution.1;
    let mut img = image::RgbImage::new(w, h);
    for (i, px) in img.pixels_mut().enumerate() {
        let r = (rgb[i * 3] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (rgb[i * 3 + 1] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (rgb[i * 3 + 2] * 255.0).clamp(0.0, 255.0) as u8;
        *px = image::Rgb([r, g, b]);
    }
    if let Err(err) = img.save(novel_path) {
        eprintln!("failed to save novel-view PNG: {err}");
    } else {
        println!("wrote {} ({}x{})", novel_path, w, h);
    }
}

fn interp_camera(a: &vol::CameraParams, b: &vol::CameraParams, t: f32) -> vol::CameraParams {
    let pa = glam::Vec3::from_array(a.cam_position);
    let pb = glam::Vec3::from_array(b.cam_position);
    let pos = pa.lerp(pb, t);
    let qa = glam::Quat::from_xyzw(
        a.cam_orientation[0],
        a.cam_orientation[1],
        a.cam_orientation[2],
        a.cam_orientation[3],
    );
    let qb = glam::Quat::from_xyzw(
        b.cam_orientation[0],
        b.cam_orientation[1],
        b.cam_orientation[2],
        b.cam_orientation[3],
    );
    let q = qa.slerp(qb, t);
    vol::CameraParams {
        cam_position: pos.into(),
        depth: a.depth,
        cam_orientation: [q.x, q.y, q.z, q.w],
        fov: [
            a.fov[0] * (1.0 - t) + b.fov[0] * t,
            a.fov[1] * (1.0 - t) + b.fov[1] * t,
        ],
        principal: [
            a.principal[0] * (1.0 - t) + b.principal[0] * t,
            a.principal[1] * (1.0 - t) + b.principal[1] * t,
        ],
    }
}
