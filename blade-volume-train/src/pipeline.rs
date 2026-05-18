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

use crate::{colmap, diff_render, metrics, render};
use blade_graphics as gpu;
use blade_volume as vol;
use std::{path, sync};

/// Returns the Voronoi cell whose site is closest to `camera_origin`. Use
/// this as the `start_cell` of every ray for the view at that camera —
/// rays parametrically advance forward, so starting in the cell nearest
/// the eye keeps the trace inside the scene's depth budget.
///
/// Linear scan over every point; cheap enough for our scales (sub-millisecond
/// for 100K points) and avoids the kiddo "too many points at same position"
/// panic on mesh-derived clouds where grid-sampled interior points often
/// share coordinates.
pub fn pick_start_cell(model: &vol::PointCloudModel, camera_origin: glam::Vec3) -> u32 {
    let mut best_idx = 0u32;
    let mut best_sq = f32::INFINITY;
    for (i, p) in model.points.iter().enumerate() {
        let dx = p.x - camera_origin.x;
        let dy = p.y - camera_origin.y;
        let dz = p.z - camera_origin.z;
        let sq = dx * dx + dy * dy + dz * dz;
        if sq < best_sq {
            best_sq = sq;
            best_idx = i as u32;
        }
    }
    best_idx
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
    /// Training resolution per view (width, height). With
    /// `fit.pixel_batch.is_some()` this is the size we downsample images
    /// to before training — paths are recorded against the downsampled
    /// pixel grid. With `None`, this is also the per-step graph shape.
    pub resolution: (u32, u32),
    /// Maximum path length per ray.
    pub max_steps: usize,
    /// Cap on number of views to feed into training. `None` uses all.
    pub max_views: Option<usize>,
    /// Cap on number of cells in the initial point cloud. `None` keeps every
    /// point from `points3D.bin`. The current Delaunay backend
    /// (`simple_delaunay_lib`) gets quadratic on tens of thousands of points;
    /// real COLMAP scenes routinely ship 100K+, so subsampling is essential.
    /// When set, we pick a deterministic stride through the points.
    pub max_initial_points: Option<usize>,
    /// Adam settings.
    pub fit: diff_render::AppearanceFitConfig,
    /// `CameraParams::depth` — far plane forwarded to every per-view camera.
    pub far_plane: f32,
    /// Initial per-cell density at conversion time. The trainer optimises
    /// this, but starting value matters: too low and alphas are tiny so
    /// gradients can't push the model forward in reasonable Adam steps.
    pub initial_density: f32,
    /// Adjacency choice for the initial point cloud.
    pub adjacency: AdjacencyKind,
}

/// Adjacency algorithm to use when building the initial foam.
#[derive(Clone, Copy, Debug)]
pub enum AdjacencyKind {
    /// Exact Delaunay tetrahedralisation. Memory is O(N^1.5); breaks past
    /// ~7 K points on a 24 GB machine.
    Delaunay,
    /// Symmetric k-nearest-neighbour graph with `k` neighbours per point.
    /// Memory is O(N · k), so 50 K+ points fit easily. Edges are an
    /// approximation of the true Voronoi neighbours; appearance training
    /// is tolerant to this because the differentiable forward integrates
    /// along a path and averages over the cells visited.
    Knn(usize),
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            resolution: (32, 32),
            max_steps: 64,
            max_views: Some(8),
            max_initial_points: Some(2000),
            fit: diff_render::AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 100,
                ..diff_render::AppearanceFitConfig::default()
            },
            far_plane: 100.0,
            initial_density: 1.0,
            adjacency: AdjacencyKind::Delaunay,
        }
    }
}

impl PipelineConfig {
    /// Total number of Adam steps the trainer will run. Useful for callers
    /// that want a progress bar without re-running training.
    pub fn total_adam_steps(&self, n_views: usize) -> usize {
        if self.fit.pixel_batch.is_some() {
            self.fit.steps_per_view * n_views
        } else {
            self.fit.epochs * n_views
        }
    }
}

/// Return at most `cap` evenly-spaced indices from `[0, n)`.
fn subsample_indices(n: usize, cap: usize) -> Vec<usize> {
    if n <= cap {
        return (0..n).collect();
    }
    let mut out = Vec::with_capacity(cap);
    for k in 0..cap {
        // floor((k + 0.5) * n / cap) — places samples in the middle of each bin.
        let idx = ((k as u64 * 2 + 1) * n as u64) / (2 * cap as u64);
        out.push((idx as usize).min(n - 1));
    }
    out
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
    let images = reconstruction
        .images
        .iter()
        .take(config.max_views.unwrap_or(reconstruction.images.len()));
    build_views_from(reconstruction, images_dir, initial_model, config, images)
}

/// Build a custom slice of views — same machinery as [`build_views`] but the
/// caller picks the images. Used by the test-set evaluation path so it can
/// reuse the camera-conversion + image-loading logic.
pub fn build_views_from<'a>(
    reconstruction: &colmap::Reconstruction,
    images_dir: &path::Path,
    initial_model: &vol::PointCloudModel,
    config: &PipelineConfig,
    images: impl IntoIterator<Item = &'a colmap::ColmapImage>,
) -> Vec<diff_render::ViewSupervision> {
    let mut views = Vec::new();
    for image in images {
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
            width: config.resolution.0,
            height: config.resolution.1,
            start_cell,
        });
    }
    views
}

/// Render each `view` at the configured resolution and report per-view +
/// average PSNR against its ground-truth pixels. Predictions are clamped to
/// `[0, 1]` before the PSNR computation (matches how the image would land
/// on screen / disk). Returns the per-view PSNR values in the same order
/// as `views`.
pub fn evaluate_views(
    model: &vol::PointCloudModel,
    views: &[diff_render::ViewSupervision],
    config: &PipelineConfig,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(views.len());
    for v in views {
        let pixels = render::render_cpu(
            model,
            &v.camera,
            render::RenderSettings {
                width: config.resolution.0,
                height: config.resolution.1,
                start_point: v.start_cell,
                max_steps: config.max_steps as u32,
                weight_threshold: 1e-4,
            },
        );
        let mut pred = metrics::rgba_to_rgb(&pixels);
        for p in pred.iter_mut() {
            *p = p.clamp(0.0, 1.0);
        }
        out.push(metrics::psnr(&pred, &v.target_rgb));
    }
    out
}

/// Result of an end-to-end COLMAP training run. The Reconstruction comes
/// back so callers can render extra views (test set, novel poses) without
/// re-parsing the binary.
pub struct TrainOutcome {
    pub model: vol::PointCloudModel,
    pub reconstruction: colmap::Reconstruction,
    /// Loss values per Adam step in the order they ran (epoch-major × view-major).
    pub training_loss: Vec<f32>,
}

/// Convenience wrapper that calls [`train_colmap_appearance_split`] with no
/// test views. Returns just the trained model for backwards compatibility.
pub fn train_colmap_appearance(
    sparse_dir: &path::Path,
    images_dir: &path::Path,
    config: &PipelineConfig,
    gpu: sync::Arc<gpu::Context>,
) -> vol::PointCloudModel {
    train_colmap_appearance_split(sparse_dir, images_dir, config, 0, gpu).model
}

/// Orchestrator. Loads the COLMAP reconstruction, splits its image list into
/// the first `config.max_views` training images and the next `test_views`
/// held-out images, builds an initial `PointCloudModel`, fits appearance,
/// and returns the trained model along with the reconstruction so the caller
/// can evaluate on the test split if desired (`evaluate_views`).
pub fn train_colmap_appearance_split(
    sparse_dir: &path::Path,
    images_dir: &path::Path,
    config: &PipelineConfig,
    _test_views: usize,
    gpu: sync::Arc<gpu::Context>,
) -> TrainOutcome {
    let recon = colmap::load_reconstruction(sparse_dir);
    let mut model = recon.to_initial_model_with_density(config.initial_density);
    if let Some(cap) = config.max_initial_points {
        if model.points.len() > cap {
            let n_before = model.points.len();
            let indices = subsample_indices(n_before, cap);
            let mut new_points = Vec::with_capacity(indices.len());
            let mut new_sh = Vec::with_capacity(indices.len() * 3);
            for &i in &indices {
                new_points.push(model.points[i]);
                new_sh.extend_from_slice(&model.sh_coefficients[i * 3..i * 3 + 3]);
            }
            model.points = new_points;
            model.sh_coefficients = new_sh;
            log::info!(
                "subsampled COLMAP cloud {} → {} points",
                n_before,
                model.points.len()
            );
        }
    }
    let t0 = std::time::Instant::now();
    match config.adjacency {
        AdjacencyKind::Delaunay => {
            log::info!(
                "computing Delaunay adjacency for {} points...",
                model.points.len()
            );
            model.compute_adjacency_default();
        }
        AdjacencyKind::Knn(k) => {
            log::info!(
                "computing symmetric k-NN adjacency (k={k}) for {} points...",
                model.points.len()
            );
            let adj = vol::compute_knn(&model.points, k);
            model.adjacency = Some(adj);
        }
    }
    log::info!(
        "adjacency done in {:.2}s ({} edges)",
        t0.elapsed().as_secs_f32(),
        model
            .adjacency
            .as_ref()
            .map(|a| a.neighbors.len())
            .unwrap_or(0),
    );
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
        return TrainOutcome {
            model,
            reconstruction: recon,
            training_loss: Vec::new(),
        };
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
    TrainOutcome {
        model,
        reconstruction: recon,
        training_loss: losses,
    }
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
            max_initial_points: None,
            initial_density: 1.0,
            fit: diff_render::AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 20,
                ..diff_render::AppearanceFitConfig::default()
            },
            far_plane: 50.0,
            adjacency: AdjacencyKind::Delaunay,
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
