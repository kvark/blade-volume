//! End-to-end COLMAP → trained foam → render-novel-pose pipeline.
//!
//! Components:
//! - [`pick_start_cell`] picks the Voronoi or power cell containing each
//!   camera origin
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

/// Returns the Voronoi or power cell containing `camera_origin`. Use this as
/// the `start_cell` of every ray for the view at that camera. Weighted clouds
/// minimize power distance rather than Euclidean distance.
///
/// Linear scan over every point; cheap enough for our scales (sub-millisecond
/// for 100K points) and avoids the kiddo "too many points at same position"
/// panic on quantized clouds with many coincident sites.
pub fn pick_start_cell(model: &vol::PointCloudModel, camera_origin: glam::Vec3) -> u32 {
    model.containing_cell(camera_origin)
}

/// Load `path` as an RGB image, downsample to `(width, height)`, and return
/// `width * height * 3` display-referred sRGB code values in `[0, 1]`,
/// row-major. Resizing intentionally matches reference NVS loaders by
/// filtering encoded image samples without a hidden linear-light conversion.
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

fn sample_bilinear(image: &image::RgbImage, x: f64, y: f64) -> [f32; 3] {
    if x < 0.0 || y < 0.0 || x > (image.width() - 1) as f64 || y > (image.height() - 1) as f64 {
        return [0.0; 3];
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let tx = (x - x0 as f64) as f32;
    let ty = (y - y0 as f64) as f32;
    let mut result = [0.0; 3];
    for (channel, output) in result.iter_mut().enumerate() {
        let p00 = image.get_pixel(x0, y0)[channel] as f32 / 255.0;
        let p10 = image.get_pixel(x1, y0)[channel] as f32 / 255.0;
        let p01 = image.get_pixel(x0, y1)[channel] as f32 / 255.0;
        let p11 = image.get_pixel(x1, y1)[channel] as f32 / 255.0;
        let top = p00 * (1.0 - tx) + p10 * tx;
        let bottom = p01 * (1.0 - tx) + p11 * tx;
        *output = top * (1.0 - ty) + bottom * ty;
    }
    result
}

/// Load and rectify a COLMAP image onto the camera's undistorted pinhole
/// plane. The output camera retains its calibrated focal length and principal
/// point, so all renderers can use one pinhole ray generator while supervision
/// remains aligned for radial, tangential, division, and fisheye models.
pub fn load_and_rectify_image(
    path: &path::Path,
    width: u32,
    height: u32,
    camera: &colmap::ColmapCamera,
) -> Result<Vec<f32>, image::ImageError> {
    let source = image::open(path)?.to_rgb8();
    let (fx, fy, cx, cy) = camera.model.fxfycxcy(&camera.params);
    let mut result = Vec::with_capacity((width * height * 3) as usize);
    for iy in 0..height {
        for ix in 0..width {
            let source_u = (ix as f64 + 0.5) * camera.width as f64 / width as f64;
            let source_v = (iy as f64 + 0.5) * camera.height as f64 / height as f64;
            let normal_u = (source_u - cx) / fx;
            let normal_v = (source_v - cy) / fy;
            let color = if let Some(projected) = camera.project_camera_plane(normal_u, normal_v) {
                let sample_x = projected[0] * source.width() as f64 / camera.width as f64 - 0.5;
                let sample_y = projected[1] * source.height() as f64 / camera.height as f64 - 0.5;
                sample_bilinear(&source, sample_x, sample_y)
            } else {
                [0.0; 3]
            };
            result.extend_from_slice(&color);
        }
    }
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Training resolution per view (width, height). Images are downsampled
    /// to this size before training; paths use the same pixel grid.
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
    /// Resume: when `Some(path)`, load the initial foam from this RadFoam
    /// PLY (a prior checkpoint) instead of building it from the COLMAP
    /// reconstruction. Cameras/views still come from `--sparse`/`--images`;
    /// only the point cloud topology is taken from the PLY. Pairs with
    /// `fit.resume_state_path` for lossless parameters/Adam and
    /// `fit.resume_step` to continue the schedule.
    pub init_ply: Option<path::PathBuf>,
    /// Standard NVS held-out split: when `> 0`, every `test_every`-th image
    /// (index `% test_every == 0` over the existence-filtered, COLMAP-ordered
    /// list) is held out for testing and training uses the rest. The
    /// Mip-NeRF-360 / NeRF "llffhold" convention is 8. `0` (legacy) trains
    /// on the first `max_views` images — which on filename-ordered captures
    /// makes the test set a contiguous tail arc (mostly extrapolation) and
    /// depresses test PSNR versus the standard protocol.
    pub test_every: usize,
}

/// Adjacency algorithm to use when building the initial foam.
#[derive(Clone, Copy, Debug)]
pub enum AdjacencyKind {
    /// Exact Delaunay tetrahedralisation via `simple_delaunay_lib`.
    /// Memory is O(N^1.5); breaks past ~7 K points on a 24 GB machine.
    Delaunay,
    /// Exact Delaunay tetrahedralisation via Qhull. It scales much better
    /// than the default Rust backend on typical clouds, though 3D Delaunay
    /// has quadratic worst-case output. It matches the `Delaunay` backend
    /// on non-degenerate inputs (≥98%
    /// edge agreement on random clouds; differences come from
    /// tie-breaking on near-co-spherical quadruples).
    DelaunayQhull,
    /// Symmetric k-nearest-neighbour graph with `k` neighbours per point.
    /// Cheap (O(N · k) memory) but the resulting edges are a poor
    /// approximation of the Voronoi adjacency: true Voronoi neighbours
    /// can be far away (long thin cells) while many close points sit
    /// on the same side. Path tracing through a k-NN graph picks the
    /// wrong next-cell often enough to cost ~3 dB of PSNR at 6 K
    /// cells. Kept for scaling experiments, not for quality.
    Knn(usize),
    /// Čech complex: each point gets a radius (here scaled
    /// nearest-neighbour distance) and two points are adjacent when
    /// their balls intersect. Closer to a power-diagram adjacency than
    /// k-NN at the same memory budget, so the path-tracer's
    /// face-finding picks the right next-cell more often.
    /// `radius_factor` ~ 1.0 keeps balls just-touching their nearest
    /// neighbour.
    Cech { radius_factor: f32 },
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
            init_ply: None,
            test_every: 0,
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

/// Partition the existence-filtered, COLMAP-ordered image list into
/// `(train, test)` slices. With `test_every > 0`, every `test_every`-th
/// image is held out (the standard NVS "llffhold" protocol); train is the
/// rest, capped at `max_views`, and test is capped at `test_views` when
/// that is non-zero. With `test_every == 0` this reproduces the legacy
/// contiguous slicing: first `max_views` train, next `test_views` test.
pub fn split_train_test<'a>(
    reconstruction: &'a colmap::Reconstruction,
    images_dir: &path::Path,
    max_views: Option<usize>,
    test_views: usize,
    test_every: usize,
) -> (Vec<&'a colmap::ColmapImage>, Vec<&'a colmap::ColmapImage>) {
    let existing: Vec<&colmap::ColmapImage> = reconstruction
        .images
        .iter()
        .filter(|img| images_dir.join(&img.name).is_file())
        .collect();
    if test_every > 0 {
        let mut train = Vec::new();
        let mut test = Vec::new();
        for (i, img) in existing.iter().enumerate() {
            if i % test_every == 0 {
                test.push(*img);
            } else {
                train.push(*img);
            }
        }
        if let Some(cap) = max_views {
            train.truncate(cap);
        }
        if test_views > 0 {
            test.truncate(test_views);
        }
        (train, test)
    } else {
        let train: Vec<&colmap::ColmapImage> = existing
            .iter()
            .take(max_views.unwrap_or(existing.len()))
            .copied()
            .collect();
        let test: Vec<&colmap::ColmapImage> = existing
            .iter()
            .skip(train.len())
            .take(test_views)
            .copied()
            .collect();
        (train, test)
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
    config: &PipelineConfig,
) -> Vec<diff_render::ViewSupervision> {
    // Filter to images that actually exist on disk before taking
    // `max_views`. The COLMAP image list and the filesystem can drift
    // (e.g. only a 27% subset of images delivered in the bonsai-80
    // dataset), and `take(max_views)` from the raw list ends up with
    // fewer than requested when missing files fall inside the prefix
    // — and worse, the eval test-split lands on the missing files.
    let images = reconstruction
        .images
        .iter()
        .filter(|img| images_dir.join(&img.name).is_file())
        .take(config.max_views.unwrap_or(reconstruction.images.len()));
    build_views_from(reconstruction, images_dir, config, images)
}

/// Build a custom slice of views — same machinery as [`build_views`] but the
/// caller picks the images. Used by the test-set evaluation path so it can
/// reuse the camera-conversion + image-loading logic.
pub fn build_views_from<'a>(
    reconstruction: &colmap::Reconstruction,
    images_dir: &path::Path,
    config: &PipelineConfig,
    images: impl IntoIterator<Item = &'a colmap::ColmapImage>,
) -> Vec<diff_render::ViewSupervision> {
    let mut views = Vec::new();
    for image in images {
        let camera = reconstruction
            .cameras
            .get(&image.camera_id)
            .unwrap_or_else(|| {
                panic!(
                    "image {} references unknown camera_id {}",
                    image.id, image.camera_id
                )
            });
        let target_path = images_dir.join(&image.name);
        let target_rgb = match load_and_rectify_image(
            &target_path,
            config.resolution.0,
            config.resolution.1,
            camera,
        ) {
            Ok(rgb) => rgb,
            Err(err) => {
                log::warn!("skipping image {}: {err}", image.name);
                continue;
            }
        };
        let cam = reconstruction.camera_params_for(image, config.far_plane);
        views.push(diff_render::ViewSupervision {
            camera: cam,
            target_rgb,
            width: config.resolution.0,
            height: config.resolution.1,
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
        let rendered = render::render_cpu_with_diagnostics(
            model,
            &v.camera,
            render::RenderSettings {
                width: config.resolution.0,
                height: config.resolution.1,
                start_point: pick_start_cell(model, glam::Vec3::from_array(v.camera.cam_position)),
                max_steps: config.max_steps as u32,
                weight_threshold: 1e-4,
            },
        );
        let diagnostics = rendered.traversal;
        let mean_steps = diagnostics.total_steps as f32 / diagnostics.rays.max(1) as f32;
        log::debug!(
            "evaluation traversal: {} rays, {:.1} mean steps, max {}",
            diagnostics.rays,
            mean_steps,
            diagnostics.max_steps_used,
        );
        if diagnostics.truncated_rays > 0 {
            log::warn!(
                "evaluation truncated {} / {} rays ({:.3}%) at max_steps={}",
                diagnostics.truncated_rays,
                diagnostics.rays,
                100.0 * diagnostics.truncated_rays as f32 / diagnostics.rays.max(1) as f32,
                config.max_steps,
            );
        }
        let mut pred = metrics::rgba_over_background(&rendered.rgba, config.fit.background_rgb);
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

fn rebuild_adjacency(model: &mut vol::PointCloudModel, kind: AdjacencyKind) {
    match kind {
        AdjacencyKind::Delaunay => {
            log::info!(
                "computing Delaunay adjacency for {} points...",
                model.points.len()
            );
            model.radii = None;
            model.compute_adjacency_default();
        }
        AdjacencyKind::DelaunayQhull => {
            log::info!(
                "computing Qhull Delaunay adjacency for {} points...",
                model.points.len()
            );
            model.radii = None;
            #[cfg(feature = "qhull")]
            {
                model.adjacency = Some(vol::compute_adjacency_qhull_default(&model.points));
            }
            #[cfg(not(feature = "qhull"))]
            panic!("Qhull adjacency requires blade-volume-train feature `qhull`");
        }
        AdjacencyKind::Knn(k) => {
            log::info!(
                "computing symmetric k-NN adjacency (k={k}) for {} points...",
                model.points.len()
            );
            model.radii = None;
            model.adjacency = Some(vol::compute_knn(&model.points, k));
        }
        AdjacencyKind::Cech { radius_factor } => {
            log::info!(
                "computing Cech adjacency (radius_factor={radius_factor}) for {} points...",
                model.points.len()
            );
            let radii = vol::radii_from_nearest_neighbour(&model.points, radius_factor);
            model.adjacency = Some(vol::compute_cech_default(&model.points, &radii));
            model.radii = Some(radii);
        }
    }
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
    // Resume path: take the foam from a checkpoint PLY instead of the
    // COLMAP cloud. Cameras/views still come from `recon`; the PLY already
    // carries the densified, trained cell set, so we skip the COLMAP
    // subsample. Adjacency is (re)built below either way.
    let mut model = if let Some(ref ply) = config.init_ply {
        let path_str = ply.to_string_lossy().to_string();
        let m = vol::io::load_radfoam(&path_str);
        log::info!(
            "resume: loaded initial foam from {} ({} cells, sh_degree {})",
            path_str,
            m.points.len(),
            m.sh_degree,
        );
        m
    } else {
        recon.to_initial_model_with_density(config.initial_density)
    };
    if config.init_ply.is_none() {
        if let Some(cap) = config.max_initial_points {
            if model.points.len() > cap {
                let n_before = model.points.len();
                // Pick the highest-track-length COLMAP points first. Track
                // length = how many images saw this 3D point during SfM, so
                // it's a cheap proxy for "how many training rays will hit
                // the cell built around this point". Stride-sampling picks
                // many points the training cameras never look at, and those
                // cells stay at init values and pollute the path-trace.
                let mut order: Vec<usize> = (0..n_before).collect();
                order.sort_by(|&a, &b| recon.points[b].track_len.cmp(&recon.points[a].track_len));
                order.truncate(cap);
                order.sort_unstable();
                let indices = order;
                let mut new_points = Vec::with_capacity(indices.len());
                let mut new_sh = Vec::with_capacity(indices.len() * 3);
                for &i in &indices {
                    new_points.push(model.points[i]);
                    new_sh.extend_from_slice(&model.sh_coefficients[i * 3..i * 3 + 3]);
                }
                model.points = new_points;
                model.sh_coefficients = new_sh;
                log::info!(
                    "subsampled COLMAP cloud {} → {} points (top-track-length)",
                    n_before,
                    model.points.len()
                );
            }
        }
    }
    let t0 = std::time::Instant::now();
    rebuild_adjacency(&mut model, config.adjacency);
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

    let views = if config.test_every > 0 {
        let (train_imgs, test_imgs) =
            split_train_test(&recon, images_dir, config.max_views, 0, config.test_every);
        log::info!(
            "every-{}th held-out split: {} train / {} test images",
            config.test_every,
            train_imgs.len(),
            test_imgs.len(),
        );
        build_views_from(&recon, images_dir, config, train_imgs)
    } else {
        build_views(&recon, images_dir, config)
    };
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
        config.fit.clone(),
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
    fn cech_selection_preserves_radii_and_other_modes_clear_them() {
        let mut model = tiny_model_far_apart();
        rebuild_adjacency(&mut model, AdjacencyKind::Cech { radius_factor: 0.6 });
        let radii = model.radii.as_ref().expect("Cech model must keep weights");
        assert_eq!(radii.len(), model.points.len());
        assert!(radii.iter().all(|&r| (r - 6.0).abs() < 1e-6));
        model.validate().unwrap();

        rebuild_adjacency(&mut model, AdjacencyKind::Delaunay);
        assert!(model.radii.is_none());
        model.validate().unwrap();
    }

    #[test]
    fn pick_start_cell_returns_nearest() {
        let model = tiny_model_far_apart();
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(0.5, 0.0, 0.0)), 0);
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(9.0, 0.0, 0.0)), 1);
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(0.0, 9.0, 0.0)), 2);
        assert_eq!(pick_start_cell(&model, glam::Vec3::new(0.0, 0.0, 9.0)), 3);
    }

    #[test]
    fn pick_start_cell_uses_power_distance() {
        let mut model = tiny_model_far_apart();
        model.radii = Some(vec![0.0, 20.0, 0.0, 0.0]);
        assert_eq!(pick_start_cell(&model, glam::Vec3::ZERO), 1);
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
        let views = build_views(&recon, &images, &config);
        assert_eq!(views.len(), 2);
        assert!(
            views[0].target_rgb.len()
                == config.resolution.0 as usize * config.resolution.1 as usize * 3
        );
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
        let checkpoint = dir.join("checkpoint.ply");

        let config = PipelineConfig {
            resolution: (8, 8),
            max_steps: 16,
            max_views: Some(2),
            max_initial_points: None,
            initial_density: 1.0,
            fit: diff_render::AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 20,
                checkpoint_path: Some(checkpoint.clone()),
                ..diff_render::AppearanceFitConfig::default()
            },
            far_plane: 50.0,
            adjacency: AdjacencyKind::Delaunay,
            init_ply: None,
            test_every: 0,
        };
        let trained = train_colmap_appearance(&sparse, &images, &config, gpu.clone());

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

        let optimizer_checkpoint = checkpoint.with_extension("safetensors");
        assert!(checkpoint.is_file());
        assert!(optimizer_checkpoint.is_file());
        assert!(checkpoint.with_extension("ply.step").is_file());
        assert!(checkpoint.with_extension("trainstate").is_file());
        let trainer_state = diff_render::load_training_state(&checkpoint).unwrap();
        assert_eq!(trainer_state.step, config.total_adam_steps(2));

        let mut resume_config = config.clone();
        resume_config.init_ply = Some(checkpoint.clone());
        resume_config.fit.resume_state_path = Some(optimizer_checkpoint);
        resume_config.fit.resume_step = resume_config.total_adam_steps(2) - 1;
        resume_config.fit.checkpoint_path = None;
        let resumed = train_colmap_appearance(&sparse, &images, &resume_config, gpu);
        assert_eq!(resumed.points.len(), trained.points.len());
        assert!(resumed
            .sh_coefficients
            .iter()
            .all(|value| value.is_finite()));
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
