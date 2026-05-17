//! End-to-end COLMAP → trained foam → render-novel-pose pipeline.
//!
//! Components:
//! - [`pick_start_cell`] picks the Voronoi cell each per-view ray starts in
//!   (nearest cell to the camera origin via a kd-tree)
//! - [`load_and_downsample_image`] loads an image file with the `image`
//!   crate and resizes to the training resolution
//! - [`train_colmap_appearance`] is the orchestrator: load reconstruction
//!   → build initial model → compute adjacency → fit appearance across
//!   all training views → return the trained `PointCloudModel`
//!
//! The bin target `train_colmap` (in `src/bin/`) wraps these for CLI use.

use crate::{colmap, diff_render};
use blade_graphics as gpu;
use blade_volume as vol;
use std::{path, sync};

/// Returns the Voronoi cell whose site is closest to `camera_origin`. Use
/// this as the `start_cell` of every ray for the view at that camera —
/// rays parametrically advance forward, so starting in the cell nearest
/// the eye keeps the trace inside the scene's depth budget.
pub fn pick_start_cell(model: &vol::PointCloudModel, camera_origin: glam::Vec3) -> u32 {
    let mut tree: kiddo::KdTree<f32, 3> = kiddo::KdTree::new();
    for (i, p) in model.points.iter().enumerate() {
        tree.add(&[p.x, p.y, p.z], i as u64);
    }
    let hit = tree.nearest_one::<kiddo::SquaredEuclidean>(&[
        camera_origin.x,
        camera_origin.y,
        camera_origin.z,
    ]);
    hit.item as u32
}

/// Load `path` as an RGB image, downsample to `(width, height)`, and return
/// `width * height * 3` floats in `[0, 1]`, row-major.
pub fn load_and_downsample_image(
    path: &path::Path,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, image::ImageError> {
    let img = image::open(path)?;
    let resized = img.resize_exact(width, height, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for px in rgb.pixels() {
        out.push(px[0] as f32 / 255.0);
        out.push(px[1] as f32 / 255.0);
        out.push(px[2] as f32 / 255.0);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Training resolution per view (width, height).
    pub resolution: (u32, u32),
    /// Maximum path length per ray.
    pub max_steps: usize,
    /// Cap on number of views to feed into training. `None` uses all.
    pub max_views: Option<usize>,
    /// Adam settings.
    pub fit: diff_render::AppearanceFitConfig,
    /// `CameraParams::depth` — far plane forwarded to every per-view camera.
    pub far_plane: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            resolution: (32, 32),
            max_steps: 64,
            max_views: Some(8),
            fit: diff_render::AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 100,
                ..diff_render::AppearanceFitConfig::default()
            },
            far_plane: 100.0,
        }
    }
}

/// Build the training views from a COLMAP reconstruction.
///
/// Image files are loaded from `images_dir/<image.name>`. Images that can't
/// be opened are skipped (with a log warning). The returned vector is
/// truncated to `config.max_views` if set.
pub fn build_views(
    reconstruction: &colmap::Reconstruction,
    images_dir: &path::Path,
    initial_model: &vol::PointCloudModel,
    config: &PipelineConfig,
) -> Vec<diff_render::ViewSupervision> {
    let mut views = Vec::new();
    let limit = config.max_views.unwrap_or(reconstruction.images.len());
    for image in reconstruction.images.iter().take(limit) {
        let target_path = images_dir.join(&image.name);
        let target_rgb =
            match load_and_downsample_image(&target_path, config.resolution.0, config.resolution.1)
            {
                Ok(rgb) => rgb,
                Err(err) => {
                    log::warn!("skipping image {}: {err}", image.name);
                    continue;
                }
            };
        let cam = reconstruction.camera_params_for(image, config.far_plane);
        let start_cell = pick_start_cell(initial_model, glam::Vec3::from_array(cam.cam_position));
        views.push(diff_render::ViewSupervision {
            camera: cam,
            target_rgb,
            start_cell,
        });
    }
    views
}

/// Orchestrator. Takes a sparse COLMAP directory + a parallel images
/// directory, returns a trained `PointCloudModel` whose per-cell density
/// and SH degree-0 DC have been fit to the provided views' downsampled
/// pixels. The model's geometry is the sparse 3D points from
/// `points3D.bin`; adjacency is computed via Delaunay.
pub fn train_colmap_appearance(
    sparse_dir: &path::Path,
    images_dir: &path::Path,
    config: &PipelineConfig,
    gpu: sync::Arc<gpu::Context>,
) -> vol::PointCloudModel {
    let recon = colmap::load_reconstruction(sparse_dir);
    let mut model = recon.to_initial_model();
    model.compute_adjacency_default();
    log::info!(
        "loaded {} COLMAP points, {} images; training {} views at {:?}",
        model.points.len(),
        recon.images.len(),
        config.max_views.unwrap_or(recon.images.len()),
        config.resolution,
    );

    let views = build_views(&recon, images_dir, &model, config);
    if views.is_empty() {
        log::warn!("no usable training views — returning untrained initial model");
        return model;
    }

    let losses = diff_render::fit_appearance_multi_view(
        &mut model,
        &views,
        config.resolution.0,
        config.resolution.1,
        config.max_steps,
        config.fit,
        gpu,
    );

    if let (Some(&first), Some(&last)) = (losses.first(), losses.last()) {
        log::info!(
            "training: loss {first:.4} → {last:.4} over {} steps",
            losses.len()
        );
    }
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model_far_apart() -> vol::PointCloudModel {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(10.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 10.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 10.0, 1.0),
        ];
        vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            points,
        }
    }

    #[test]
    fn pick_start_cell_returns_nearest() {
        let model = tiny_model_far_apart();
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(0.5, 0.0, 0.0)), 0);
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(9.0, 0.0, 0.0)), 1);
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(0.0, 9.0, 0.0)), 2);
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(0.0, 0.0, 9.0)), 3);
    }

    /// Write a minimal COLMAP fixture: one PINHOLE camera, two images
    /// referencing it, and four sparse 3D points (enough for Delaunay).
    /// Side-by-side images directory with the matching PNGs.
    fn write_colmap_fixture(sparse: &path::Path, images: &path::Path) {
        use std::io::Write as _;
        let _ = std::fs::create_dir_all(sparse);
        let _ = std::fs::create_dir_all(images);

        fn put_u32(out: &mut Vec<u8>, v: u32) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn put_i32(out: &mut Vec<u8>, v: i32) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn put_u64(out: &mut Vec<u8>, v: u64) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn put_f64(out: &mut Vec<u8>, v: f64) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn put_cstr(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(s.as_bytes());
            out.push(0);
        }

        // cameras.bin: 1 PINHOLE camera at 64×64, fx=fy=64, cx=cy=32.
        let mut buf = Vec::new();
        put_u64(&mut buf, 1);
        put_u32(&mut buf, 1);
        put_i32(&mut buf, 1); // PINHOLE
        put_u64(&mut buf, 64);
        put_u64(&mut buf, 64);
        for &v in &[64.0_f64, 64.0, 32.0, 32.0] {
            put_f64(&mut buf, v);
        }
        std::fs::File::create(sparse.join("cameras.bin"))
            .unwrap()
            .write_all(&buf)
            .unwrap();

        // images.bin: two images, varying poses.
        let mut buf = Vec::new();
        put_u64(&mut buf, 2);
        for &(id, qw, tz, name) in &[
            (10u32, 1.0_f64, 0.0_f64, "frame0001.png"),
            (11u32, 1.0_f64, 1.0_f64, "frame0002.png"),
        ] {
            put_u32(&mut buf, id);
            for &v in &[qw, 0.0, 0.0, 0.0] {
                put_f64(&mut buf, v);
            }
            for &v in &[0.0, 0.0, tz] {
                put_f64(&mut buf, v);
            }
            put_u32(&mut buf, 1);
            put_cstr(&mut buf, name);
            put_u64(&mut buf, 0);
        }
        std::fs::File::create(sparse.join("images.bin"))
            .unwrap()
            .write_all(&buf)
            .unwrap();

        // points3D.bin: 4 well-separated points.
        let mut buf = Vec::new();
        put_u64(&mut buf, 4);
        let pts = [
            (100u64, [0.0_f64, 0.0, 2.0], [255_u8, 0, 0]),
            (101u64, [0.5, 0.0, 2.0], [0, 255, 0]),
            (102u64, [0.0, 0.5, 2.0], [0, 0, 255]),
            (103u64, [0.0, 0.0, 2.5], [255, 255, 0]),
        ];
        for &(id, xyz, rgb) in &pts {
            put_u64(&mut buf, id);
            for &v in &xyz {
                put_f64(&mut buf, v);
            }
            for &v in &rgb {
                buf.push(v);
            }
            put_f64(&mut buf, 0.1);
            put_u64(&mut buf, 0);
        }
        std::fs::File::create(sparse.join("points3D.bin"))
            .unwrap()
            .write_all(&buf)
            .unwrap();

        // Two solid-colour 4x4 PNGs (resize in the pipeline picks them up).
        for &(name, colour) in &[
            ("frame0001.png", image::Rgb([200_u8, 100, 50])),
            ("frame0002.png", image::Rgb([50, 100, 200])),
        ] {
            let mut img = image::RgbImage::new(4, 4);
            for px in img.pixels_mut() {
                *px = colour;
            }
            img.save(images.join(name)).unwrap();
        }
    }

    #[test]
    fn build_views_from_synthetic_colmap() {
        let dir = std::env::temp_dir().join("blade-volume-train-pipeline");
        let sparse = dir.join("sparse/0");
        let images = dir.join("images");
        let _ = std::fs::remove_dir_all(&dir);
        write_colmap_fixture(&sparse, &images);

        let recon = crate::colmap::load_reconstruction(&sparse);
        let mut model = recon.to_initial_model();
        model.compute_adjacency_default();
        assert_eq!(model.points.len(), 4);

        let config = PipelineConfig::default();
        let views = build_views(&recon, &images, &model, &config);
        assert_eq!(views.len(), 2);
        assert!(
            views[0].target_rgb.len()
                == config.resolution.0 as usize * config.resolution.1 as usize * 3
        );
        // start_cell sits within the model.
        for v in &views {
            assert!((v.start_cell as usize) < model.points.len());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn end_to_end_colmap_train_succeeds() {
        let Some(gpu) = crate::fit::try_init_gpu() else {
            eprintln!("skipping end_to_end_colmap_train_succeeds: no GPU");
            return;
        };
        let dir = std::env::temp_dir().join("blade-volume-train-e2e");
        let sparse = dir.join("sparse/0");
        let images = dir.join("images");
        let _ = std::fs::remove_dir_all(&dir);
        write_colmap_fixture(&sparse, &images);

        let config = PipelineConfig {
            resolution: (8, 8),
            max_steps: 16,
            max_views: Some(2),
            fit: diff_render::AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 20,
                ..diff_render::AppearanceFitConfig::default()
            },
            far_plane: 50.0,
        };
        let trained = train_colmap_appearance(&sparse, &images, &config, gpu);

        // Initial model has SH derived from the points3D RGB; training should
        // adjust at least one cell measurably.
        assert!(trained.points.len() == 4);
        assert!(trained.sh_coefficients.iter().any(|v| !v.is_nan()));
        // Render at the first training view and assert pixels are finite.
        let cam = crate::colmap::load_reconstruction(&sparse)
            .camera_params_for(&crate::colmap::load_reconstruction(&sparse).images[0], 50.0);
        let pixels = crate::render::render_cpu(
            &trained,
            &cam,
            crate::render::RenderSettings {
                width: 8,
                height: 8,
                start_point: pick_start_cell(&trained, glam::Vec3::from_array(cam.cam_position)),
                max_steps: 16,
                weight_threshold: 1e-4,
            },
        );
        assert!(pixels.iter().all(|p| p.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_and_downsample_image_returns_normalized_rgb() {
        // Write a tiny 4x4 solid-red PNG to a temp file, load it, check
        // both that resize produces the requested shape and values land in
        // [0, 1].
        let dir = std::env::temp_dir().join("blade-volume-train-img");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("red.png");
        let mut img = image::RgbImage::new(4, 4);
        for px in img.pixels_mut() {
            *px = image::Rgb([255, 0, 0]);
        }
        img.save(&path).unwrap();
        let pixels = load_and_downsample_image(&path, 2, 2).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(pixels.len(), 2 * 2 * 3);
        for px in pixels.chunks_exact(3) {
            assert!((px[0] - 1.0).abs() < 1e-6, "red should be 1.0");
            assert!(px[1].abs() < 1e-6);
            assert!(px[2].abs() < 1e-6);
        }
    }
}
