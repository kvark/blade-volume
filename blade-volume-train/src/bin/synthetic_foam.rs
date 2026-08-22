//! Train and score a RadFoam density field on Blade's synthetic capture.
//!
//! This developer benchmark deliberately starts without G-buffer positions or
//! normals. Its initial sites fill a camera-derived volume, while selected
//! training-view radiance and foreground masks provide all supervision.

use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train as train;
use std::path;

#[derive(argh::FromArgs)]
/// Train a camera-initialized RadFoam on a Blade relighting dataset.
struct Args {
    /// directory written by Blade's relight_data test
    #[argh(option)]
    dataset: String,

    /// output RadFoam PLY
    #[argh(option)]
    output: String,

    /// evaluate and extract an existing trained foam instead of training
    #[argh(option)]
    input: Option<String>,

    /// environment used as RGB supervision (default: sun-east)
    #[argh(option, default = "String::from(\"sun-east\")")]
    environment: String,

    /// environment reserved for relighting evaluation (default: studio)
    #[argh(option, default = "String::from(\"studio\")")]
    held_out_environment: String,

    /// reserve every nth camera for novel-pose evaluation
    #[argh(option, default = "4")]
    held_out_stride: usize,

    /// residue within the stride to reserve
    #[argh(option, default = "1")]
    held_out_offset: usize,

    /// training and evaluation width
    #[argh(option, default = "100")]
    width: u32,

    /// training and evaluation height
    #[argh(option, default = "75")]
    height: u32,

    /// camera-derived initial foam sites
    #[argh(option, default = "1024")]
    points: usize,

    /// final adaptively densified foam-site budget
    #[argh(option, default = "2048")]
    target_points: usize,

    /// initial volume radius divided by median camera distance to its focus
    #[argh(option, default = "0.7")]
    volume_radius: f32,

    /// initial activated density
    #[argh(option, default = "0.1")]
    initial_density: f32,

    /// spherical-harmonic appearance degree
    #[argh(option, default = "2")]
    sh_degree: usize,

    /// random pixels per Adam update
    #[argh(option, default = "2048")]
    pixel_batch: usize,

    /// cameras represented in every Adam update
    #[argh(option, default = "6")]
    views_per_batch: usize,

    /// adam updates per training camera
    #[argh(option, default = "300")]
    steps_per_view: usize,

    /// base Adam learning rate
    #[argh(option, default = "0.1")]
    learning_rate: f32,

    /// position learning rate divided by the base rate
    #[argh(option, default = "0.01")]
    position_lr_ratio: f32,

    /// updates between topology rebuilds when positions move
    #[argh(option, default = "100")]
    geometry_rebuild_every: usize,

    /// maximum cells traversed by one training ray
    #[argh(option, default = "256")]
    max_steps: usize,

    /// minimum total ray absorption used for surface extraction
    #[argh(option, default = "0.7")]
    min_alpha: f32,

    /// minimum single-segment ray weight used for surface extraction
    #[argh(option, default = "0.05")]
    min_peak: f32,

    /// fused surface cell size in pixel-footprint units
    #[argh(option, default = "5.0")]
    voxel_factor: f32,

    /// gaussian support radius divided by fused cell size
    #[argh(option, default = "1.4")]
    disc_radius: f32,

    /// distinct training depth maps required per fused surface cell
    #[argh(option, default = "2")]
    min_views: usize,

    /// shared PBR materials in the reconstructed surface cloud
    #[argh(option, default = "6")]
    materials: usize,

    /// hand the real capture light to the material fit as a diagnostic control
    #[argh(switch)]
    true_light: bool,

    /// brightest diffuse albedo assumed when recovering an unknown light
    #[argh(option, default = "0.8")]
    brightest_albedo: f32,

    /// fit Gaussian normals from every known light except the held-out one
    #[argh(switch)]
    photometric_normals: bool,

    /// masked PowerFoam updates per view after Gaussian surface extraction
    #[argh(option, default = "0")]
    surface_powerfoam_steps_per_view: usize,

    /// optional trained surface-PowerFoam light-field PLY output
    #[argh(option)]
    surface_powerfoam_output: Option<String>,

    /// optional directly trained anisotropic Gaussian light-field PLY output
    #[argh(option)]
    gaussian_output: Option<String>,

    /// direct Gaussian updates, one third appearance then support (default 1500)
    #[argh(option, default = "1500")]
    gaussian_steps: usize,

    /// retain fused depth centers instead of multi-view photo refinement
    #[argh(switch)]
    no_refine: bool,

    /// particles to refine against all training renders (default 0)
    #[argh(option, default = "0")]
    render_refine: usize,

    /// simultaneous full-cloud render refinement rounds (default 0; use 8 for
    /// the conservative final-quality schedule)
    #[argh(option, default = "0")]
    render_refine_rounds: usize,

    /// refine PBR Gaussian radii against complete renders
    #[argh(switch)]
    render_refine_radii: bool,

    /// refine the shared diffuse material table against complete renders
    #[argh(switch)]
    render_refine_materials: bool,

    /// refine Gaussian normals against complete renders from the known lights
    #[argh(switch)]
    render_refine_normals: bool,

    /// optional relightable Gaussian surface output
    #[argh(option)]
    surface_output: Option<String>,
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

fn split_views(count: usize, stride: usize, offset: usize) -> (Vec<usize>, Vec<usize>) {
    if stride < 2 {
        fail("held-out stride must be at least two");
    }
    let residue = offset % stride;
    let mut training = Vec::new();
    let mut held_out = Vec::new();
    for index in 0..count {
        if index % stride == residue {
            held_out.push(index);
        } else {
            training.push(index);
        }
    }
    if training.is_empty() || held_out.is_empty() {
        fail("the requested view split needs both training and held-out cameras");
    }
    (training, held_out)
}

fn camera_focus(cameras: &[vol::CameraParams]) -> glam::Vec3 {
    let mut system = glam::Mat3::ZERO;
    let mut right = glam::Vec3::ZERO;
    for camera in cameras {
        let origin = glam::Vec3::from(camera.cam_position);
        let direction = glam::Quat::from_array(camera.cam_orientation) * glam::Vec3::Z;
        let projector = glam::Mat3::IDENTITY
            - glam::Mat3::from_cols(
                direction * direction.x,
                direction * direction.y,
                direction * direction.z,
            );
        system += projector;
        right += projector * origin;
    }
    if system.determinant().abs() <= 1.0e-6 {
        cameras
            .iter()
            .map(|camera| glam::Vec3::from(camera.cam_position))
            .sum::<glam::Vec3>()
            / cameras.len().max(1) as f32
    } else {
        system.inverse() * right
    }
}

fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// Fill a jittered lattice from the back of the capture volume towards the
/// cameras. A non-cubic count leaves the camera-facing outer layer sparse,
/// avoiding an initial wall of absorbing sites directly in front of every
/// training ray.
fn layered_positions(count: usize, radius: f32, camera_side: glam::Vec3) -> Vec<glam::Vec3> {
    let side = (count as f64).cbrt().ceil() as usize;
    let orientation = glam::Quat::from_rotation_arc(glam::Vec3::Z, camera_side);
    (0..count)
        .map(|index| {
            let indices = [index % side, (index / side) % side, index / (side * side)];
            let local = glam::Vec3::from_array(std::array::from_fn(|axis| {
                let jitter = hash((3 * index + axis) as u32) as f64 / u32::MAX as f64 - 0.5;
                ((((indices[axis] as f64 + 0.5 + 0.6 * jitter) / side as f64) * 2.0 - 1.0)
                    * radius as f64) as f32
            }));
            orientation * local
        })
        .collect()
}

fn initial_model(
    cameras: &[vol::CameraParams],
    count: usize,
    radius_factor: f32,
    density: f32,
) -> vol::PointCloudModel {
    if count < 4 {
        fail("the initial cloud needs at least four sites");
    }
    let focus = camera_focus(cameras);
    let mut distances: Vec<f32> = cameras
        .iter()
        .map(|camera| (glam::Vec3::from(camera.cam_position) - focus).length())
        .collect();
    distances.sort_by(f32::total_cmp);
    let radius = distances[distances.len() / 2] * radius_factor;
    let camera_side = cameras
        .iter()
        .filter_map(|camera| (glam::Vec3::from(camera.cam_position) - focus).try_normalize())
        .sum::<glam::Vec3>()
        .try_normalize()
        .unwrap_or(glam::Vec3::Z);
    let points = layered_positions(count, radius, camera_side)
        .into_iter()
        .map(|position| (focus + position).extend(density))
        .collect();
    println!(
        "camera bundle focus ({:.3}, {:.3}, {:.3}), initial radius {:.3}",
        focus.x, focus.y, focus.z, radius
    );
    let mut model = vol::PointCloudModel {
        sh_coefficients: vec![0.0; count * 3],
        sh_degree: 0,
        transforms: None,
        adjacency: None,
        radii: None,
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
        points,
    };
    model.compute_adjacency_default();
    model
}

fn supervision(
    dataset: &train::relight::Dataset,
    environment: usize,
    indices: &[usize],
    width: u32,
    height: u32,
) -> Result<Vec<train::diff_render::ViewSupervision>, String> {
    let mut result = Vec::with_capacity(indices.len());
    for &index in indices {
        let view = dataset
            .views
            .get(index)
            .ok_or_else(|| format!("no view {index}"))?;
        let radiance = dataset.read_plane(&view.radiance[environment])?;
        let geometry = dataset.read_plane(&view.geometry)?;
        let source =
            image::Rgb32FImage::from_fn(dataset.width as u32, dataset.height as u32, |x, y| {
                let pixel = y as usize * dataset.width + x as usize;
                if geometry[pixel][3] <= 0.0 {
                    image::Rgb([0.0; 3])
                } else {
                    image::Rgb([
                        train::inverse::capture::linear_to_srgb(radiance[pixel][0]),
                        train::inverse::capture::linear_to_srgb(radiance[pixel][1]),
                        train::inverse::capture::linear_to_srgb(radiance[pixel][2]),
                    ])
                }
            });
        let source_alpha =
            image::ImageBuffer::from_fn(dataset.width as u32, dataset.height as u32, |x, y| {
                let pixel = y as usize * dataset.width + x as usize;
                image::Luma([(geometry[pixel][3] > 0.0) as u8 as f32])
            });
        let resized = image::imageops::resize(
            &source,
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        let resized_alpha = image::imageops::resize(
            &source_alpha,
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        result.push(train::diff_render::ViewSupervision {
            camera: train::relight::camera_params(view, width as usize, height as usize),
            target_rgb: resized.pixels().flat_map(|pixel| pixel.0).collect(),
            target_alpha: Some(resized_alpha.pixels().map(|pixel| pixel[0]).collect()),
            width,
            height,
        });
    }
    Ok(result)
}

fn describe_scores(name: &str, scores: &[f32]) {
    let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
    let worst = scores.iter().copied().fold(f32::INFINITY, f32::min);
    println!("{name}: {mean:.2} dB mean, {worst:.2} dB worst {scores:?}");
}

fn densify_config(
    initial_points: usize,
    target_points: usize,
) -> Result<Option<train::diff_render::DensifyConfig>, String> {
    if target_points < initial_points {
        return Err(format!(
            "target point budget {target_points} is smaller than {initial_points} initial sites"
        ));
    }
    Ok(
        (target_points > initial_points).then(|| train::diff_render::DensifyConfig {
            every: 150,
            fraction: 0.25,
            warmup: 300,
            target_points,
            densify_until: 1050,
            ..train::diff_render::DensifyConfig::default()
        }),
    )
}

fn describe_relight_score(name: &str, summary: train::inverse::score::Summary) {
    println!(
        "{name}: {:.2} dB, {:.2} worst, {:.1}% coverage, {:.2} where hit, {:.2} ms/frame",
        summary.srgb_psnr,
        summary.worst_srgb_psnr,
        100.0 * summary.coverage,
        summary.covered_srgb_psnr,
        summary.render_ms,
    );
}

fn quantile(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let index = ((sorted.len() - 1) as f32 * fraction).round() as usize;
    sorted[index]
}

fn describe_depths(
    name: &str,
    dataset: &train::relight::Dataset,
    indices: &[usize],
    maps: &[train::inverse::depth::DepthMap],
    options: train::inverse::depth::DepthOptions,
) -> Result<(), String> {
    let mut truth_hits = 0usize;
    let mut predicted_hits = 0usize;
    let mut shared_hits = 0usize;
    let mut squared_error = 0.0f64;
    let mut absolute_errors = Vec::new();
    let mut view_rmse = Vec::with_capacity(indices.len());
    let mut normal_errors = Vec::new();
    for (&index, map) in indices.iter().zip(maps) {
        let geometry = dataset.read_plane(&dataset.views[index].geometry)?;
        let camera =
            train::relight::camera_params(&dataset.views[index], dataset.width, dataset.height);
        let origin = glam::Vec3::from(camera.cam_position);
        let point_at = |x: usize, y: usize| {
            let slot = y * map.width + x;
            (map.alpha[slot] >= options.min_alpha && map.peak[slot] >= options.min_peak).then(
                || {
                    origin
                        + map.distance[slot]
                            * train::inverse::capture::pixel_direction(
                                &camera, map.width, map.height, x, y,
                            )
                },
            )
        };
        let mut view_squared_error = 0.0f64;
        let mut view_shared_hits = 0usize;
        for (pixel, truth) in geometry.iter().enumerate() {
            let truth_hit = truth[3] > 0.0;
            let predicted_hit =
                map.alpha[pixel] >= options.min_alpha && map.peak[pixel] >= options.min_peak;
            truth_hits += truth_hit as usize;
            predicted_hits += predicted_hit as usize;
            if truth_hit && predicted_hit {
                shared_hits += 1;
                view_shared_hits += 1;
                let error = (map.distance[pixel] - truth[3]).abs();
                squared_error += error.powi(2) as f64;
                view_squared_error += error.powi(2) as f64;
                absolute_errors.push(error);
            }
        }
        for y in 0..map.height.saturating_sub(1) {
            for x in 0..map.width.saturating_sub(1) {
                let slot = y * map.width + x;
                let truth = geometry[slot];
                let (Some(here), Some(right), Some(down)) =
                    (point_at(x, y), point_at(x + 1, y), point_at(x, y + 1))
                else {
                    continue;
                };
                let distance = map.distance[slot];
                let step = options.max_depth_step * distance;
                if truth[3] <= 0.0
                    || (map.distance[slot + 1] - distance).abs() > step
                    || (map.distance[slot + map.width] - distance).abs() > step
                {
                    continue;
                }
                let Some(normal) = (right - here).cross(down - here).try_normalize() else {
                    continue;
                };
                let normal = if normal.dot(origin - here) < 0.0 {
                    -normal
                } else {
                    normal
                };
                let Some(truth_normal) =
                    glam::Vec3::from_array([truth[0], truth[1], truth[2]]).try_normalize()
                else {
                    continue;
                };
                normal_errors.push(
                    normal
                        .dot(truth_normal)
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees(),
                );
            }
        }
        view_rmse.push((view_squared_error / view_shared_hits.max(1) as f64).sqrt() as f32);
    }
    let precision = shared_hits as f64 / predicted_hits.max(1) as f64;
    let recall = shared_hits as f64 / truth_hits.max(1) as f64;
    let rmse = (squared_error / shared_hits.max(1) as f64).sqrt();
    println!(
        "{name} depth: {:.1}% precision, {:.1}% recall, {rmse:.4} world-unit RMSE",
        100.0 * precision,
        100.0 * recall,
    );
    absolute_errors.sort_by(f32::total_cmp);
    view_rmse.sort_by(f32::total_cmp);
    normal_errors.sort_by(f32::total_cmp);
    println!(
        "{name} depth tails: absolute p50 {:.4}, p90 {:.4}, p99 {:.4}, max {:.4}; per-view RMSE p50 {:.4}, max {:.4}; finite-difference normal p50 {:.2}, p90 {:.2}, RMSE {:.2}",
        quantile(&absolute_errors, 0.5),
        quantile(&absolute_errors, 0.9),
        quantile(&absolute_errors, 0.99),
        quantile(&absolute_errors, 1.0),
        quantile(&view_rmse, 0.5),
        quantile(&view_rmse, 1.0),
        quantile(&normal_errors, 0.5),
        quantile(&normal_errors, 0.9),
        (normal_errors
            .iter()
            .map(|angle| angle.powi(2) as f64)
            .sum::<f64>()
            / normal_errors.len().max(1) as f64)
            .sqrt(),
    );
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct ErrorBucket {
    count: usize,
    position_squared: f64,
    normal_squared: f64,
}

impl ErrorBucket {
    fn add(&mut self, position: f32, normal: f32) {
        self.count += 1;
        self.position_squared += position.powi(2) as f64;
        self.normal_squared += normal.powi(2) as f64;
    }

    fn position_rmse(self) -> f64 {
        (self.position_squared / self.count.max(1) as f64).sqrt()
    }

    fn normal_rmse(self) -> f64 {
        (self.normal_squared / self.count.max(1) as f64).sqrt()
    }
}

fn depth_support(
    surfel: &vol::relight::Surfel,
    maps: &[(vol::CameraParams, train::inverse::depth::DepthMap)],
    options: train::inverse::depth::DepthOptions,
) -> usize {
    let center = glam::Vec3::from(surfel.center);
    maps.iter()
        .filter(|entry| {
            let entry = *entry;
            let camera = &entry.0;
            let map = &entry.1;
            let Some((pixel, _)) =
                train::inverse::capture::project(camera, map.width, map.height, center)
            else {
                return false;
            };
            let x = (pixel[0] - 0.5).round() as isize;
            let y = (pixel[1] - 0.5).round() as isize;
            if x < 0 || y < 0 || x >= map.width as isize || y >= map.height as isize {
                return false;
            }
            let slot = y as usize * map.width + x as usize;
            if map.alpha[slot] < options.min_alpha || map.peak[slot] < options.min_peak {
                return false;
            }
            let distance = center.distance(glam::Vec3::from(camera.cam_position));
            (distance - map.distance[slot]).abs() <= 0.5 * surfel.radius
        })
        .count()
}

fn describe_surface_error(
    name: &str,
    surfels: &[vol::relight::Surfel],
    dataset: &train::relight::Dataset,
    training_indices: &[usize],
    maps: &[(vol::CameraParams, train::inverse::depth::DepthMap)],
    options: train::inverse::depth::DepthOptions,
) -> Result<(), String> {
    let truth = train::relight::gather_samples_for_views(dataset, training_indices)?;
    let positions: Vec<[f32; 3]> = truth
        .iter()
        .map(|sample| sample.position.to_array())
        .collect();
    let tree = kiddo::ImmutableKdTree::new_from_slice(&positions);
    let mut position_squared = 0.0f64;
    let mut normal_squared = 0.0f64;
    let mut position_errors = Vec::with_capacity(surfels.len());
    let mut normal_errors = Vec::with_capacity(surfels.len());
    let mut support_buckets = [ErrorBucket::default(); 3];
    for surfel in surfels {
        let center = glam::Vec3::from(surfel.center);
        let hit = tree.nearest_one::<kiddo::SquaredEuclidean>(&surfel.center);
        let sample = &truth[hit.item as usize];
        let position_error = (center - sample.position).length();
        position_squared += position_error.powi(2) as f64;
        let angle = glam::Vec3::from(surfel.normal)
            .dot(sample.normal)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        normal_squared += angle.powi(2) as f64;
        position_errors.push(position_error);
        normal_errors.push(angle);
        let support = depth_support(surfel, maps, options);
        let bucket = if support < 2 {
            0
        } else if support < 4 {
            1
        } else {
            2
        };
        support_buckets[bucket].add(position_error, angle);
    }
    println!(
        "{name} nearest-truth error: position {:.4} world-unit RMSE, normal {:.2} degree RMSE",
        (position_squared / surfels.len().max(1) as f64).sqrt(),
        (normal_squared / surfels.len().max(1) as f64).sqrt(),
    );
    position_errors.sort_by(f32::total_cmp);
    normal_errors.sort_by(f32::total_cmp);
    println!(
        "{name} tails: position p50 {:.4}, p90 {:.4}, p99 {:.4}, max {:.4}; normal p50 {:.2}, p90 {:.2}, p99 {:.2}, max {:.2}",
        quantile(&position_errors, 0.5),
        quantile(&position_errors, 0.9),
        quantile(&position_errors, 0.99),
        quantile(&position_errors, 1.0),
        quantile(&normal_errors, 0.5),
        quantile(&normal_errors, 0.9),
        quantile(&normal_errors, 0.99),
        quantile(&normal_errors, 1.0),
    );
    println!(
        "{name} by refreshed depth support: 0-1 views {} / {:.4} / {:.2}, 2-3 views {} / {:.4} / {:.2}, 4+ views {} / {:.4} / {:.2} (count / position RMSE / normal RMSE)",
        support_buckets[0].count,
        support_buckets[0].position_rmse(),
        support_buckets[0].normal_rmse(),
        support_buckets[1].count,
        support_buckets[1].position_rmse(),
        support_buckets[1].normal_rmse(),
        support_buckets[2].count,
        support_buckets[2].position_rmse(),
        support_buckets[2].normal_rmse(),
    );
    Ok(())
}

fn refine_photometric_normals(
    model: &mut vol::relight::RelightModel,
    dataset: &train::relight::Dataset,
    views: &[usize],
    held_out_environment: usize,
) -> Result<train::inverse::decompose::NormalRefinement, String> {
    let mut observations = Vec::new();
    let mut irradiance = Vec::new();
    for environment in 0..dataset.environments.len() {
        if environment == held_out_environment {
            continue;
        }
        let capture =
            train::inverse::capture::Capture::from_relight_dataset(dataset, environment, true)?;
        observations.push(train::inverse::decompose::observe(
            model, &capture, views, -1.0,
        ));
        let light = vol::io::try_load_environment(&dataset.environment_files[environment])
            .map_err(|error| error.to_string())?;
        irradiance.push(train::relight::Irradiance::project(
            &light.texels,
            light.width,
            light.height,
        ));
    }
    let lights: Vec<_> = irradiance
        .into_iter()
        .zip(&observations)
        .map(
            |(irradiance, observations)| train::inverse::decompose::KnownLightObservations {
                irradiance,
                observations,
            },
        )
        .collect();
    Ok(train::inverse::decompose::refine_normals_known_lights_per_view(model, &lights, 512))
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();
    if args.surface_powerfoam_output.is_some() && args.surface_powerfoam_steps_per_view == 0 {
        fail("--surface-powerfoam-output requires --surface-powerfoam-steps-per-view");
    }
    if args.gaussian_output.is_some() && args.gaussian_steps < 2 {
        fail("--gaussian-output requires at least two --gaussian-steps");
    }
    let dataset = train::relight::Dataset::load(path::Path::new(&args.dataset))
        .unwrap_or_else(|error| fail(error));
    let environment = dataset
        .environments
        .iter()
        .position(|name| name == &args.environment)
        .unwrap_or_else(|| fail(format!("no environment named '{}'", args.environment)));
    let held_out_environment = dataset
        .environments
        .iter()
        .position(|name| name == &args.held_out_environment)
        .unwrap_or_else(|| {
            fail(format!(
                "no environment named '{}'",
                args.held_out_environment
            ))
        });
    let (training_indices, held_out_indices) = split_views(
        dataset.views.len(),
        args.held_out_stride,
        args.held_out_offset,
    );
    let training = supervision(
        &dataset,
        environment,
        &training_indices,
        args.width,
        args.height,
    )
    .unwrap_or_else(|error| fail(error));
    let held_out = supervision(
        &dataset,
        environment,
        &held_out_indices,
        args.width,
        args.height,
    )
    .unwrap_or_else(|error| fail(error));
    println!(
        "{} training and {} held-out views at {}x{}",
        training.len(),
        held_out.len(),
        args.width,
        args.height
    );

    let Some(gpu) = train::fit::try_init_gpu() else {
        fail("no supported GPU device");
    };
    let densify =
        densify_config(args.points, args.target_points).unwrap_or_else(|error| fail(error));
    let config = train::diff_render::AppearanceFitConfig {
        learning_rate: args.learning_rate,
        pixel_batch: Some(args.pixel_batch),
        views_per_batch: args.views_per_batch.min(training.len()),
        steps_per_view: args.steps_per_view,
        sh_degree: args.sh_degree,
        color_loss: train::diff_render::ColorLoss::SmoothL1,
        opacity_weight: 1.0,
        quantile_weight: 1.0e-4,
        softplus_beta: 10.0,
        position_lr_ratio: args.position_lr_ratio,
        geometry_rebuild_every: args.geometry_rebuild_every,
        densify,
        ..train::diff_render::AppearanceFitConfig::default()
    };
    let model = match args.input {
        Some(ref input) => vol::io::try_load(input)
            .unwrap_or_else(|error| fail(format!("cannot read {input}: {error}"))),
        None => {
            let adjacency_started = std::time::Instant::now();
            let mut model = initial_model(
                &training.iter().map(|view| view.camera).collect::<Vec<_>>(),
                args.points,
                args.volume_radius,
                args.initial_density,
            );
            println!(
                "{} initial sites, {} directed edges in {:.3} s",
                model.points.len(),
                model.adjacency.as_ref().unwrap().neighbors.len(),
                adjacency_started.elapsed().as_secs_f64()
            );
            let train_started = std::time::Instant::now();
            let losses = train::diff_render::fit_appearance_multi_view(
                &mut model,
                &training,
                args.width,
                args.height,
                args.max_steps,
                config.clone(),
                gpu.clone(),
            );
            println!(
                "trained {} updates in {:.3} s, loss {:.6} -> {:.6}",
                losses.len(),
                train_started.elapsed().as_secs_f64(),
                losses.first().copied().unwrap_or(f32::NAN),
                losses.last().copied().unwrap_or(f32::NAN)
            );
            model
        }
    };
    model.validate().unwrap_or_else(|error| fail(error));
    let mut densities: Vec<f32> = model.points.iter().map(|point| point.w).collect();
    densities.sort_by(f32::total_cmp);
    let density_at =
        |fraction: f32| densities[((densities.len() - 1) as f32 * fraction).round() as usize];
    println!(
        "density: min {:.4}, p50 {:.4}, p90 {:.4}, p95 {:.4}, p99 {:.4}, max {:.4}",
        densities[0],
        density_at(0.5),
        density_at(0.9),
        density_at(0.95),
        density_at(0.99),
        densities[densities.len() - 1],
    );
    println!(
        "density support: >.01 {} >.1 {} >.5 {} >1 {}",
        densities.iter().filter(|&&density| density > 0.01).count(),
        densities.iter().filter(|&&density| density > 0.1).count(),
        densities.iter().filter(|&&density| density > 0.5).count(),
        densities.iter().filter(|&&density| density > 1.0).count(),
    );

    let evaluator_config = train::pipeline::PipelineConfig {
        resolution: (args.width, args.height),
        max_steps: args.max_steps,
        fit: config,
        ..train::pipeline::PipelineConfig::default()
    };
    let mut evaluator =
        train::pipeline::GpuViewEvaluator::new(&model, &evaluator_config, gpu.clone());
    let training_scores = evaluator
        .evaluate(&training, [0.0; 3])
        .unwrap_or_else(|error| fail(error));
    let held_out_scores = evaluator
        .evaluate(&held_out, [0.0; 3])
        .unwrap_or_else(|error| fail(error));
    evaluator.deinit();
    describe_scores("training", &training_scores);
    describe_scores("held-out", &held_out_scores);

    let depth_options = train::inverse::depth::DepthOptions {
        min_alpha: args.min_alpha,
        min_peak: args.min_peak,
        max_steps: args.max_steps as u32,
        voxel_factor: args.voxel_factor,
        disc_radius: if args.photometric_normals {
            args.disc_radius.max(1.6)
        } else {
            args.disc_radius
        },
        min_views: args.min_views,
        ..train::inverse::depth::DepthOptions::default()
    };
    let cameras: Vec<_> = dataset
        .views
        .iter()
        .map(|view| train::relight::camera_params(view, dataset.width, dataset.height))
        .collect();
    let requests = |indices: &[usize]| {
        indices
            .iter()
            .map(|&index| {
                let camera = cameras[index];
                let start =
                    train::pipeline::pick_start_cell(&model, glam::Vec3::from(camera.cam_position));
                (camera, start)
            })
            .collect::<Vec<_>>()
    };
    let depth_started = std::time::Instant::now();
    let training_requests = requests(&training_indices);
    let training_maps = train::inverse::depth::trace_depths_gpu(
        &model,
        &training_requests,
        dataset.width,
        dataset.height,
        depth_options.max_steps,
        gpu.clone(),
    )
    .unwrap_or_else(|error| fail(error));
    let held_out_requests = requests(&held_out_indices);
    let held_out_maps = train::inverse::depth::trace_depths_gpu(
        &model,
        &held_out_requests,
        dataset.width,
        dataset.height,
        depth_options.max_steps,
        gpu,
    )
    .unwrap_or_else(|error| fail(error));
    println!(
        "traced {} depth maps at {}x{} in {:.3} s",
        training_maps.len() + held_out_maps.len(),
        dataset.width,
        dataset.height,
        depth_started.elapsed().as_secs_f64(),
    );
    describe_depths(
        "training",
        &dataset,
        &training_indices,
        &training_maps,
        depth_options,
    )
    .unwrap_or_else(|error| fail(error));
    describe_depths(
        "held-out",
        &dataset,
        &held_out_indices,
        &held_out_maps,
        depth_options,
    )
    .unwrap_or_else(|error| fail(error));
    let maps: Vec<_> = training_requests
        .into_iter()
        .zip(training_maps)
        .map(|((camera, _), map)| (camera, map))
        .collect();
    let fusion_started = std::time::Instant::now();
    let (mut surfels, voxel) = train::inverse::depth::surfels_from_depth(&maps, depth_options);
    println!(
        "fused {} Gaussian surface particles at voxel {:.4} in {:.3} s",
        surfels.len(),
        voxel,
        fusion_started.elapsed().as_secs_f64(),
    );

    let training_capture =
        train::inverse::capture::Capture::from_relight_dataset(&dataset, environment, true)
            .unwrap_or_else(|error| fail(error));
    if !args.no_refine {
        let refine_started = std::time::Instant::now();
        let refinement_views: Vec<_> = training_indices
            .iter()
            .zip(&maps)
            .map(
                |(&capture_index, entry)| train::inverse::refine::RefinementView {
                    capture_index,
                    depth: Some(&entry.1),
                },
            )
            .collect();
        let stats = train::inverse::refine::refine(
            &mut surfels,
            &training_capture,
            &refinement_views,
            train::inverse::refine::RefineOptions::default(),
        );
        println!(
            "refined {} of {} scored particles by {:.3} cells ({:.1}% lower cost) in {:.3} s",
            stats.moved,
            stats.scored,
            stats.mean_absolute_offset / voxel,
            100.0 * stats.mean_relative_improvement,
            refine_started.elapsed().as_secs_f64(),
        );
    }
    for surfel in surfels.iter_mut() {
        surfel.material = 0;
    }
    let mut geometry = vol::relight::RelightModel {
        kernel: vol::relight::ParticleKernel::Gaussian,
        surfels,
        materials: vec![vol::relight::Material::default()],
    };
    if args.surface_powerfoam_steps_per_view > 0 {
        let started = std::time::Instant::now();
        let Some(gpu) = train::fit::try_init_gpu() else {
            fail("no supported GPU device for surface PowerFoam continuation");
        };
        let outcome = train::inverse::powerfoam::continue_surface(
            &mut geometry,
            &training_capture,
            &training_indices,
            train::inverse::powerfoam::ContinueOptions {
                steps_per_view: args.surface_powerfoam_steps_per_view,
                ..Default::default()
            },
            gpu,
        )
        .unwrap_or_else(|error| fail(error));
        if let Some(ref output) = args.surface_powerfoam_output {
            let output = path::Path::new(output);
            if let Some(parent) = output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent).unwrap_or_else(|error| fail(error));
            }
            convert::save_ply(output, &outcome.light_field).unwrap_or_else(|error| {
                fail(format!("cannot write {}: {error:?}", output.display()))
            });
            println!("wrote {}", output.display());
        }
        println!(
            "surface PowerFoam: {} updates, loss {:.6} -> {:.6}, in {:.3} s",
            outcome.stats.updates,
            outcome.stats.initial_loss,
            outcome.stats.final_loss,
            started.elapsed().as_secs_f64(),
        );
    }
    if args.photometric_normals {
        let started = std::time::Instant::now();
        let stats = refine_photometric_normals(
            &mut geometry,
            &dataset,
            &training_indices,
            held_out_environment,
        )
        .unwrap_or_else(|error| fail(error));
        println!(
            "photometrically refined {} of {} supported normals ({} particles) in {:.3} s",
            stats.changed,
            stats.supported,
            geometry.surfels.len(),
            started.elapsed().as_secs_f64(),
        );
    }
    describe_surface_error(
        "extracted surface",
        &geometry.surfels,
        &dataset,
        &training_indices,
        &maps,
        depth_options,
    )
    .unwrap_or_else(|error| fail(error));
    let observe_started = std::time::Instant::now();
    let observations = train::inverse::decompose::observe(
        &geometry,
        &training_capture,
        &training_indices,
        train::inverse::decompose::FitOptions::default().min_facing,
    );
    println!(
        "observed {} of {} particles in {:.3} s",
        observations.seen(),
        geometry.surfels.len(),
        observe_started.elapsed().as_secs_f64(),
    );
    let training_light = vol::io::try_load_environment(&dataset.environment_files[environment])
        .unwrap_or_else(|error| fail(error));
    let given_light = args.true_light.then_some(&training_light);
    let decompose_started = std::time::Instant::now();
    let mut fitted = train::inverse::decompose::fit(
        &geometry,
        &observations,
        train::inverse::decompose::FitOptions {
            materials: args.materials,
            // Reconstructed normals are not yet accurate enough to identify a
            // specular lobe from one light. A false mirror fits that light and
            // fails catastrophically under the held-out one.
            specular_rounds: 0,
            brightest_albedo: args.brightest_albedo,
            ..train::inverse::decompose::FitOptions::default()
        },
        train::inverse::decompose::Given {
            visibility: None,
            light: given_light,
        },
    );
    println!(
        "fitted {} materials with {:.5} residual ({} unseen) in {:.3} s",
        fitted.scene.model.materials.len(),
        fitted.residual,
        fitted.unseen,
        decompose_started.elapsed().as_secs_f64(),
    );
    if args.render_refine_normals {
        let environment_indices = [environment];
        let normal_captures: Vec<_> = environment_indices
            .iter()
            .map(|&index| {
                train::inverse::capture::Capture::from_relight_dataset(&dataset, index, true)
                    .unwrap_or_else(|error| fail(error))
            })
            .collect();
        let normal_environments: Vec<_> = environment_indices
            .iter()
            .map(|&index| {
                vol::io::try_load_environment(&dataset.environment_files[index])
                    .unwrap_or_else(|error| fail(error))
            })
            .collect();
        let evidence: Vec<_> = normal_captures
            .iter()
            .zip(&normal_environments)
            .map(
                |(capture, environment)| train::inverse::refine::RenderedNormalEvidence {
                    capture,
                    indices: &training_indices,
                    environment,
                },
            )
            .collect();
        let stats = train::inverse::refine::refine_rendered_normals(
            &mut fitted.scene,
            &evidence,
            &observations,
            0,
            8,
            2.5,
        )
        .unwrap_or_else(|error| fail(error));
        println!(
            "render-refined {} normals in {} rounds ({} accepted), loss {:.7} -> {:.7}, in {:.3} s",
            stats.normals,
            stats.rounds,
            stats.accepted,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
        describe_surface_error(
            "normal-render-refined surface",
            &fitted.scene.model.surfels,
            &dataset,
            &training_indices,
            &maps,
            depth_options,
        )
        .unwrap_or_else(|error| fail(error));
    }
    if args.render_refine_materials {
        let stats = train::inverse::refine::refine_rendered_materials(
            &mut fitted.scene,
            &training_capture,
            &training_indices,
            0,
            0.025,
        )
        .unwrap_or_else(|error| fail(error));
        println!(
            "render-refined {} of {} material coordinates in {} proposals, loss {:.7} -> {:.7}, in {:.3} s",
            stats.changed,
            stats.coordinates,
            stats.proposals,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
    }
    if !args.true_light {
        let light_error =
            train::inverse::truth::compare_environment(&training_light, &fitted.scene.environment);
        println!(
            "recovered training light: {:.1}% relative RMS after gauge [{:.3}, {:.3}, {:.3}]",
            100.0 * light_error.relative_rms,
            light_error.gauge[0],
            light_error.gauge[1],
            light_error.gauge[2],
        );
    }
    if args.render_refine > 0 || args.render_refine_rounds > 0 {
        let stats = train::inverse::refine::refine_rendered(
            &mut fitted.scene,
            &training_capture,
            &training_indices,
            &observations,
            0,
            args.render_refine_rounds,
            args.render_refine_radii,
            args.render_refine,
        )
        .unwrap_or_else(|error| fail(error));
        println!(
            "rendered-surface refinement: {} particles in {} rounds ({} accepted); tested {}, moved {}, radii {}, loss {:.7} -> {:.7} in {:.3} s",
            stats.simultaneous_particles,
            stats.simultaneous_rounds,
            stats.simultaneous_accepted,
            stats.tested,
            stats.moved,
            stats.radii_moved,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
        describe_surface_error(
            "render-refined surface",
            &fitted.scene.model.surfels,
            &dataset,
            &training_indices,
            &maps,
            depth_options,
        )
        .unwrap_or_else(|error| fail(error));
    }
    if let Some(ref output) = args.gaussian_output {
        let mut gaussian = train::gaussian_splat::from_surface(&fitted.scene.model)
            .unwrap_or_else(|error| fail(error));
        let mut pbr_gaussian = gaussian.clone();
        let Some(gpu) = train::fit::try_init_gpu() else {
            fail("no supported GPU device for direct Gaussian training");
        };
        let started = std::time::Instant::now();
        let stats = train::gaussian_splat::fit_staged_outputs(
            &mut pbr_gaussian,
            &mut gaussian,
            &training_capture,
            &training_indices,
            args.gaussian_steps,
            gpu,
        )
        .unwrap_or_else(|error| fail(error));
        let fit_seconds = started.elapsed().as_secs_f64();
        let core_held_out_scores = train::gaussian_splat::evaluate_views(
            &pbr_gaussian,
            &training_capture,
            &held_out_indices,
            64,
            1.0e-5,
            [0.0; 3],
        )
        .unwrap_or_else(|error| fail(error));
        println!(
            "Gaussian outputs: {} shared appearance, {} PBR support, and {} light-field support updates in {:.3} s",
            stats.appearance.steps,
            stats.pbr_support.steps,
            stats.light_field_support.steps,
            fit_seconds,
        );
        describe_scores("fixed-center Gaussian held-out", &core_held_out_scores);
        let held_out_scores = train::gaussian_splat::evaluate_views(
            &gaussian,
            &training_capture,
            &held_out_indices,
            64,
            1.0e-5,
            [0.0; 3],
        )
        .unwrap_or_else(|error| fail(error));
        let output = path::Path::new(output);
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).unwrap_or_else(|error| fail(error));
        }
        convert::save_ply(output, &gaussian)
            .unwrap_or_else(|error| fail(format!("cannot write {}: {error:?}", output.display())));
        println!(
            "direct Gaussian: {} appearance updates {:.6} -> {:.6}, {} support updates {:.6} -> {:.6} (shared-output fit {:.3} s total)",
            stats.appearance.steps,
            stats.appearance.initial_loss,
            stats.appearance.final_loss,
            stats.light_field_support.steps,
            stats.light_field_support.initial_loss,
            stats.light_field_support.final_loss,
            fit_seconds,
        );
        describe_scores("direct Gaussian held-out", &held_out_scores);
        println!("wrote {}", output.display());
        train::gaussian_splat::update_surface_radii(&mut fitted.scene.model, &pbr_gaussian)
            .unwrap_or_else(|error| fail(error));
        println!("updated PBR radii from learned Gaussian support");
    }
    let mut pbr_controls = Vec::new();
    if args.render_refine_radii && args.gaussian_output.is_some() {
        pbr_controls.push(("radius", fitted.scene.model.clone()));
        let environment = fitted.scene.environment.clone();
        let evidence = [train::inverse::refine::RenderedNormalEvidence {
            capture: &training_capture,
            indices: &training_indices,
            environment: &environment,
        }];
        let stats = train::inverse::refine::refine_rendered_radii(
            &mut fitted.scene,
            &evidence,
            &observations,
            0,
            8,
            0.1,
        )
        .unwrap_or_else(|error| fail(error));
        println!(
            "render-refined {} radii in {} rounds ({} accepted), loss {:.7} -> {:.7}, in {:.3} s",
            stats.radii,
            stats.rounds,
            stats.accepted,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
        if args.render_refine_normals {
            pbr_controls.push(("post-support normal", fitted.scene.model.clone()));
            let stats = train::inverse::refine::refine_rendered_normals(
                &mut fitted.scene,
                &evidence,
                &observations,
                0,
                8,
                2.5,
            )
            .unwrap_or_else(|error| fail(error));
            println!(
                "post-support refined {} normals in {} rounds ({} accepted), loss {:.7} -> {:.7}, in {:.3} s",
                stats.normals,
                stats.rounds,
                stats.accepted,
                stats.initial_loss,
                stats.final_loss,
                stats.seconds,
            );
            describe_surface_error(
                "post-support-normal surface",
                &fitted.scene.model.surfels,
                &dataset,
                &training_indices,
                &maps,
                depth_options,
            )
            .unwrap_or_else(|error| fail(error));
        }
    }
    if let Some(ref surface_output) = args.surface_output {
        let surface_path = path::Path::new(surface_output);
        if let Some(parent) = surface_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).unwrap_or_else(|error| fail(error));
        }
        vol::io::try_save_relight(surface_path, &fitted.scene.model)
            .unwrap_or_else(|error| fail(format!("cannot write {surface_output}: {error}")));
        println!("wrote {surface_output}");
        let environment_path = surface_path.with_extension("f32");
        vol::io::try_save_environment(&environment_path, &fitted.scene.environment).unwrap_or_else(
            |error| {
                fail(format!(
                    "cannot write {}: {error}",
                    environment_path.display()
                ))
            },
        );
        println!("wrote {}", environment_path.display());
    }
    let held_out_light =
        vol::io::try_load_environment(&dataset.environment_files[held_out_environment])
            .unwrap_or_else(|error| fail(error));
    let held_out_capture = train::inverse::capture::Capture::from_relight_dataset(
        &dataset,
        held_out_environment,
        true,
    )
    .unwrap_or_else(|error| fail(error));
    let mut renderer = train::inverse::score::Renderer::new(dataset.width, dataset.height)
        .unwrap_or_else(|error| fail(error));
    for (name, model) in pbr_controls {
        let control = renderer.score_splits(
            &train::inverse::score::Scene {
                model,
                environment: held_out_light.clone(),
            },
            &held_out_capture,
            &[(&held_out_indices, None)],
            0,
        )[0];
        println!(
            "PBR {name} control / held-out light: {:.2} dB, {:.2} worst, {:.1}% coverage",
            control.srgb_psnr,
            control.worst_srgb_psnr,
            100.0 * control.coverage,
        );
    }
    let training_summary = renderer.score_splits(
        &train::inverse::score::Scene {
            model: fitted.scene.model.clone(),
            environment: fitted.scene.environment,
        },
        &training_capture,
        &[(&held_out_indices, None)],
        0,
    )[0];
    let relight_summary = renderer.score_splits(
        &train::inverse::score::Scene {
            model: fitted.scene.model,
            environment: held_out_light,
        },
        &held_out_capture,
        &[(&held_out_indices, None)],
        0,
    )[0];
    renderer.destroy();
    describe_relight_score("PBR training light / held-out poses", training_summary);
    describe_relight_score("PBR held-out light / held-out poses", relight_summary);

    let output = path::Path::new(&args.output);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).unwrap_or_else(|error| fail(error));
    }
    convert::save_ply_with_options(
        output,
        &model,
        &convert::SaveOptions {
            format: convert::PlyFormat::Binary,
        },
    )
    .unwrap_or_else(|error| fail(format!("cannot write {}: {error:?}", output.display())));
    println!("wrote {}", output.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_powerfoam_continuation_is_opt_in() {
        let continued = <Args as argh::FromArgs>::from_args(
            &["synthetic_foam"],
            &[
                "--dataset",
                "capture",
                "--output",
                "model.ply",
                "--surface-powerfoam-steps-per-view",
                "300",
                "--surface-powerfoam-output",
                "surface.ply",
            ],
        )
        .unwrap();
        assert_eq!(continued.surface_powerfoam_steps_per_view, 300);
        assert_eq!(
            continued.surface_powerfoam_output.as_deref(),
            Some("surface.ply")
        );
    }

    #[test]
    fn direct_gaussian_output_is_opt_in() {
        let args = <Args as argh::FromArgs>::from_args(
            &["synthetic_foam"],
            &[
                "--dataset",
                "capture",
                "--output",
                "model.ply",
                "--gaussian-output",
                "light-field.ply",
                "--gaussian-steps",
                "800",
            ],
        )
        .unwrap();
        assert_eq!(args.gaussian_output.as_deref(), Some("light-field.ply"));
        assert_eq!(args.gaussian_steps, 800);
    }

    fn camera(position: glam::Vec3, target: glam::Vec3) -> vol::CameraParams {
        vol::CameraParams {
            cam_position: position.to_array(),
            cam_orientation: glam::Quat::from_rotation_arc(
                glam::Vec3::Z,
                (target - position).normalize(),
            )
            .to_array(),
            fov: [1.0; 2],
            principal: [0.0; 2],
            depth: 100.0,
        }
    }

    #[test]
    fn view_split_reserves_only_the_requested_residue() {
        let (training, held_out) = split_views(8, 4, 1);
        assert_eq!(training, [0, 2, 3, 4, 6, 7]);
        assert_eq!(held_out, [1, 5]);
    }

    #[test]
    fn camera_focus_meets_converging_view_axes() {
        let target = glam::Vec3::new(0.2, -0.3, 0.7);
        let cameras = [
            camera(glam::Vec3::new(-3.0, 0.0, 1.0), target),
            camera(glam::Vec3::new(2.0, -2.0, 2.0), target),
            camera(glam::Vec3::new(1.0, 3.0, -1.0), target),
        ];
        let actual = camera_focus(&cameras);
        assert!((actual - target).length() < 1.0e-4, "focus {actual:?}");
    }

    #[test]
    fn partial_lattice_layer_faces_the_cameras() {
        let camera_side = glam::Vec3::new(0.4, -0.2, 0.9).normalize();
        let positions = layered_positions(2048, 1.0, camera_side);
        assert_eq!(positions.len(), 2048);
        let front = positions
            .iter()
            .filter(|position| position.dot(camera_side) > 0.85)
            .count();
        assert_eq!(front, 20, "the sparse outer layer moved off camera-side");
    }

    #[test]
    fn staged_point_budget_spends_the_middle_of_training_densifying() {
        let config = densify_config(1024, 2048).unwrap().unwrap();
        assert_eq!(config.every, 150);
        assert_eq!(config.warmup, 300);
        assert_eq!(config.densify_until, 1050);
        assert_eq!(config.target_points, 2048);
    }

    #[test]
    fn fixed_or_invalid_point_budgets_are_explicit() {
        assert!(densify_config(2048, 2048).unwrap().is_none());
        assert!(densify_config(2048, 1024).is_err());
    }

    #[test]
    fn selected_material_count_is_the_cli_default() {
        let args = <Args as argh::FromArgs>::from_args(
            &["synthetic_foam"],
            &["--dataset", "capture", "--output", "model.ply"],
        )
        .unwrap();
        assert_eq!(args.materials, 6);
        assert_eq!(args.surface_powerfoam_steps_per_view, 0);
        assert!(args.surface_powerfoam_output.is_none());
        assert!(args.gaussian_output.is_none());
        assert_eq!(args.gaussian_steps, 1_500);
    }
}
