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

    /// training image width (default 32)
    #[argh(option, default = "32")]
    width: u32,

    /// training image height (default 32)
    #[argh(option, default = "32")]
    height: u32,

    /// max number of training views (default 8)
    #[argh(option, default = "8")]
    views: usize,

    /// training epochs (default 100)
    #[argh(option, default = "100")]
    epochs: usize,

    /// max traversal steps per ray (default 64)
    #[argh(option, default = "64")]
    max_steps: usize,

    /// adam learning rate (default 0.1)
    #[argh(option, default = "0.1")]
    learning_rate: f32,
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();

    let Some(gpu) = fit::try_init_gpu() else {
        eprintln!("no supported GPU device — cannot train");
        std::process::exit(2);
    };

    let config = pipeline::PipelineConfig {
        resolution: (args.width, args.height),
        max_steps: args.max_steps,
        max_views: Some(args.views),
        fit: diff_render::AppearanceFitConfig {
            learning_rate: args.learning_rate,
            epochs: args.epochs,
            ..diff_render::AppearanceFitConfig::default()
        },
        far_plane: 100.0,
    };

    let sparse = path::Path::new(&args.sparse);
    let images = path::Path::new(&args.images);
    let trained = pipeline::train_colmap_appearance(sparse, images, &config, gpu.clone());

    let output = path::Path::new(&args.output);
    convert::save_ply_with_options(
        output,
        &trained,
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
        trained.len(),
        trained
            .adjacency
            .as_ref()
            .map(|a| a.neighbors.len())
            .unwrap_or(0),
    );

    if let Some(ref novel_path) = args.novel_out {
        render_novel(&trained, sparse, images, &config, novel_path);
    }
}

/// Render the trained model from a pose halfway between the first two
/// training cameras. Quick-and-dirty: we re-read the reconstruction to get
/// the first two image extrinsics, then interpolate position + spherical-
/// linearly interpolate the orientation quat.
fn render_novel(
    model: &vol::PointCloudModel,
    sparse: &path::Path,
    _images: &path::Path,
    config: &pipeline::PipelineConfig,
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
    let novel = interp_camera(&cam_a, &cam_b, 0.5);

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
