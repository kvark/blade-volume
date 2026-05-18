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
//! 5. Runs `--epochs` epochs of multi-view appearance training (Adam, L1
//!    against per-pixel RGB) — geometry stays fixed.
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

    /// max traversal steps per ray (default 64)
    #[argh(option, default = "64")]
    max_steps: usize,

    /// adam learning rate (default 0.1)
    #[argh(option, default = "0.1")]
    learning_rate: f32,

    /// cap on initial COLMAP points (default 2000; Delaunay scales poorly past this)
    #[argh(option, default = "2000")]
    max_points: usize,

    /// k for symmetric k-NN adjacency (default 0 = use Delaunay).
    /// Set to e.g. 24 to scale past Delaunay's ~7K-point memory wall.
    /// Note: k-NN edges are a poor approximation of the true Voronoi
    /// adjacency and the path-tracer pays ~3 dB of PSNR for it. Prefer
    /// `--cech-radius` over `--knn` for quality.
    #[argh(option, default = "0")]
    knn: usize,

    /// radius factor for Čech adjacency (default 0 = off). When > 0,
    /// each point's radius = factor × distance to its nearest
    /// neighbour, and the adjacency is the intersection-graph of
    /// those balls. Better than k-NN for path-tracing because edges
    /// are anchored to the geometric scale of each cell, not raw
    /// distance rank. A factor of ~1.0 keeps balls just-touching.
    #[argh(option, default = "0.0")]
    cech_radius: f32,

    /// pixels per Adam step (default 0 = whole-image mode). Random
    /// pixel sampling per step keeps the matmul tile aligned regardless
    /// of image resolution.
    #[argh(option, default = "0")]
    pixel_batch: usize,

    /// adam steps per view in pixel-batched mode (default 200)
    #[argh(option, default = "200")]
    steps_per_view: usize,

    /// initial uniform per-cell density before training (default 1.0)
    #[argh(option, default = "1.0")]
    initial_density: f32,
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
    let adjacency = if args.cech_radius > 0.0 {
        pipeline::AdjacencyKind::Cech {
            radius_factor: args.cech_radius,
        }
    } else if args.knn > 0 {
        pipeline::AdjacencyKind::Knn(args.knn)
    } else {
        pipeline::AdjacencyKind::Delaunay
    };
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
            ..diff_render::AppearanceFitConfig::default()
        },
        far_plane: 100.0,
        initial_density: args.initial_density,
        adjacency,
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
    if args.test_views > 0 {
        let train_count = config.max_views.unwrap_or(0);
        let test_images: Vec<_> = outcome
            .reconstruction
            .images
            .iter()
            .skip(train_count)
            .take(args.test_views)
            .collect();
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
                outcome.reconstruction.images.iter().take(train_count),
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

    let w = config.resolution.0;
    let h = config.resolution.1;
    let mut img = image::RgbImage::new(w, h);
    for (i, px) in img.pixels_mut().enumerate() {
        let r = (pixels[i * 4] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (pixels[i * 4 + 1] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (pixels[i * 4 + 2] * 255.0).clamp(0.0, 255.0) as u8;
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
        pad: [0, 0],
    }
}
