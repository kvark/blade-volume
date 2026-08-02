//! Posed photographs in, a relightable scene out, and the score for it.
//!
//! Everything this prints is one of two kinds of number, and they are not
//! interchangeable:
//!
//!   - **re-rendering**, which says how close the scene comes to the images it
//!     was built from. This is what was asked for, and it can be reached by
//!     cheating: paint the illumination onto the surfaces and it goes up.
//!   - **decomposition**, which says whether the material and the light are
//!     separately anything. No photograph of a real room can measure that, so
//!     it is measured elsewhere, against a scene whose answer is known — see
//!     the `inverse_truth` binary.
//!
//! Usage:
//!   reconstruct --sparse etc/data/bonsai/sparse/0 --images etc/data/bonsai/images

use blade_volume as vol;
use blade_volume_train as train;
use std::path;

#[derive(argh::FromArgs)]
/// Reconstruct geometry, material and light from a posed capture.
struct Args {
    /// path to the COLMAP `sparse/0` directory
    #[argh(option)]
    sparse: String,

    /// path to the COLMAP images directory
    #[argh(option)]
    images: String,

    /// width to work at (default 320). Height follows the camera's aspect.
    #[argh(option, default = "320")]
    width: usize,

    /// keep every n-th image (default 4)
    #[argh(option, default = "4")]
    stride: usize,

    /// hold out every n-th kept image for testing (default 8, 0 = none)
    #[argh(option, default = "8")]
    test_every: usize,

    /// materials the surfels are clustered into (default 0 = one per surfel).
    /// Sharing is a prior about the scene rather than something the
    /// photographs said, and it costs both the albedo and the re-rendering.
    #[argh(option, default = "0")]
    materials: usize,

    /// equirectangular width of the recovered environment (default 32)
    #[argh(option, default = "32")]
    environment_width: usize,

    /// alternations between solving for albedo and for light (default 24)
    #[argh(option, default = "24")]
    iterations: usize,

    /// rounds in which roughness and reflectance are re-chosen (default 3;
    /// 0 leaves every surface a rough dielectric, which is the albedo-only fit)
    #[argh(option, default = "3")]
    specular_rounds: usize,

    /// disc radius as a multiple of the local point spacing (default 1.4).
    /// Applies to the sparse-cloud geometry only.
    #[argh(option, default = "1.4")]
    radius_factor: f32,

    /// a trained foam PLY to take the geometry from, instead of the sparse
    /// points. The surface is where each ray was absorbed, fused across views.
    #[argh(option)]
    foam: Option<String>,

    /// merge cell size when fusing depth, as a multiple of the pixel
    /// footprint at the median depth (default 5.0)
    #[argh(option, default = "5.0")]
    voxel_factor: f32,

    /// absorption below which a ray is treated as having hit nothing
    /// (default 0.5)
    #[argh(option, default = "0.5")]
    min_alpha: f32,

    /// weight the strongest segment of a ray must carry for it to count as
    /// having met a surface rather than haze (default 0.05)
    #[argh(option, default = "0.05")]
    min_peak: f32,

    /// disc radius as a multiple of the merge cell (default 1.7). Below 0.71
    /// the discs do not meet and the render is part background everywhere.
    #[argh(option, default = "1.7")]
    disc_radius: f32,

    /// distinct training views that must support a fused depth cell (default
    /// 2). Higher values reject view-specific layers but can remove surfaces
    /// seen by only one camera.
    #[argh(option, default = "2")]
    min_views: usize,

    /// shadow rays per shading point when scoring (default 32). A scene
    /// fitted with shadowing has to be rendered with it, or the comparison
    /// measures two renderer settings rather than the scene.
    #[argh(option, default = "32")]
    diffuse_samples: u32,

    /// fit without shadowing or indirect light
    #[argh(switch)]
    no_shadows: bool,

    /// measure how many visible disc footprints overlap each observation
    #[argh(switch)]
    observation_diagnostics: bool,

    /// trace foam depth on the CPU instead of the GPU
    #[argh(switch)]
    cpu_depth: bool,

    /// keep raw fused surfels instead of refining them across training views
    #[argh(switch)]
    no_multi_view_refine: bool,

    /// use the legacy compact particle footprint instead of a surface
    /// Gaussian. Intended for matched representation ablations.
    #[argh(switch)]
    compact_kernel: bool,

    /// write the reconstructed scene here
    #[argh(option)]
    output: Option<String>,

    /// write rendered and reference images for the test views here
    #[argh(option)]
    dump: Option<String>,
}

fn main() {
    env_logger::init();
    let args: Args = argh::from_env();

    let sparse = path::Path::new(&args.sparse);
    let images = path::Path::new(&args.images);

    // The working height follows the camera rather than being asked for: a
    // square render of a 3:2 photograph compares two different framings and
    // scores the difference as reconstruction error.
    let height = match aspect_height(sparse, args.width) {
        Ok(h) => h,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    let started = std::time::Instant::now();
    let (capture, reconstruction) = match train::inverse::capture::Capture::from_colmap(
        sparse,
        images,
        args.width,
        height,
        args.stride,
    ) {
        Ok(pair) => pair,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let (train_views, test_views) = capture.split(args.test_every);
    println!(
        "capture: {} views at {}x{} ({} train, {} test), {} sparse points, loaded in {:.1} s",
        capture.views.len(),
        capture.width,
        capture.height,
        train_views.len(),
        test_views.len(),
        reconstruction.points.len(),
        started.elapsed().as_secs_f64()
    );

    // ------------------------------------------------------------- geometry
    let started = std::time::Instant::now();
    let surfels = match args.foam {
        Some(ref foam) => surfels_from_foam(path::Path::new(foam), &capture, &train_views, &args),
        None => surfels_from_sparse(&reconstruction, &capture, &args),
    };
    let kernel = if args.compact_kernel {
        vol::relight::ParticleKernel::Compact
    } else {
        vol::relight::ParticleKernel::Gaussian
    };
    let mut geometry = vol::relight::RelightModel {
        kernel,
        surfels,
        materials: vec![vol::relight::Material::default()],
    };
    if args.foam.is_none() {
        let flipped =
            train::inverse::surface::orient_towards_views(&mut geometry.surfels, &capture);
        println!("geometry: {flipped} normals turned round to face the cameras");
    }
    println!(
        "geometry: {} {:?} particles in {:.1} s",
        geometry.surfels.len(),
        geometry.kernel,
        started.elapsed().as_secs_f64()
    );
    if geometry.surfels.is_empty() {
        eprintln!("no geometry survived; nothing to fit");
        std::process::exit(1);
    }

    // ---------------------------------------------------- material and light
    let started = std::time::Instant::now();
    let (observations, observation_diagnostics) = if args.observation_diagnostics {
        let (observations, diagnostics) = train::inverse::decompose::observe_with_diagnostics(
            &geometry,
            &capture,
            &train_views,
            train::inverse::decompose::FitOptions::default().min_facing,
        );
        (observations, Some(diagnostics))
    } else {
        (
            train::inverse::decompose::observe(
                &geometry,
                &capture,
                &train_views,
                train::inverse::decompose::FitOptions::default().min_facing,
            ),
            None,
        )
    };
    println!(
        "observed: {} of {} surfels seen by at least one training view, in {:.1} s",
        observations.seen(),
        geometry.surfels.len(),
        started.elapsed().as_secs_f64()
    );
    if let Some(observation_diagnostics) = observation_diagnostics {
        let shared = 100.0 * observation_diagnostics.samples_on_shared_pixels as f64
            / observation_diagnostics.samples.max(1) as f64;
        let blended = 100.0 * observation_diagnostics.samples_with_multiple_supports as f64
            / observation_diagnostics.samples.max(1) as f64;
        println!(
            "observed: {} samples over {} pixels, {shared:.1}% share a pixel (at most {} surfels)",
            observation_diagnostics.samples,
            observation_diagnostics.pixels,
            observation_diagnostics.max_samples_per_pixel,
        );
        println!(
            "observed: {blended:.1}% sample pixels blend disc footprints (mean {:.1}, at most {} supports; {} unsupported)",
            observation_diagnostics.mean_supports_per_sample(),
            observation_diagnostics.max_supports_per_sample,
            observation_diagnostics.samples_without_support,
        );
    }

    let shadows = if args.no_shadows {
        None
    } else {
        let started = std::time::Instant::now();
        let directions = train::inverse::decompose::environment_directions(args.environment_width);
        let computed = train::inverse::visibility::compute(
            &geometry,
            &directions,
            train::inverse::visibility::VisibilityOptions::default(),
        );
        println!(
            "shadowing: {} directions, {:.0}% of the sky open on average, in {:.1} s",
            directions.len(),
            100.0 * computed.mean_openness(),
            started.elapsed().as_secs_f64()
        );
        Some(computed)
    };

    let started = std::time::Instant::now();
    let fitted = train::inverse::decompose::fit(
        &geometry,
        &observations,
        train::inverse::decompose::FitOptions {
            materials: args.materials,
            environment_width: args.environment_width,
            iterations: args.iterations,
            specular_rounds: args.specular_rounds,
            ..Default::default()
        },
        train::inverse::decompose::Given {
            visibility: shadows.as_ref(),
            light: None,
        },
    );
    println!(
        "decomposed: {} materials, residual {:.4}, {} surfels unseen, in {:.1} s",
        fitted.scene.model.materials.len(),
        fitted.residual,
        fitted.unseen,
        started.elapsed().as_secs_f64()
    );
    describe_light(&fitted.scene.environment);

    if let Some(ref output) = args.output {
        let output = path::Path::new(output);
        if let Err(e) = vol::io::try_save_relight(output, &fitted.scene.model) {
            eprintln!("cannot write {}: {e}", output.display());
        } else {
            println!("wrote {}", output.display());
        }
        let environment = output.with_extension("f32");
        if let Err(e) = vol::io::try_save_environment(&environment, &fitted.scene.environment) {
            eprintln!("cannot write {}: {e}", environment.display());
        } else {
            println!("wrote {}", environment.display());
        }
    }

    // ---------------------------------------------------------------- score
    let mut renderer = match train::inverse::score::Renderer::new(capture.width, capture.height) {
        Ok(r) => r,
        Err(message) => {
            eprintln!("cannot score: {message}");
            std::process::exit(1);
        }
    };
    println!("\nGPU: {}", renderer.device_name());
    if let Some(ref directory) = args.dump {
        let _ = std::fs::create_dir_all(directory);
    }
    let dump = args.dump.as_deref().map(path::Path::new);

    println!(
        "\n{:<8}{:>8}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "split", "views", "psnr srgb", "worst", "psnr linear", "coverage", "where hit"
    );
    let splits = [
        (train_views.as_slice(), None),
        (test_views.as_slice(), dump),
    ];
    let summaries = renderer.score_splits(&fitted.scene, &capture, &splits, args.diffuse_samples);
    for ((name, indices), summary) in [("train", &train_views), ("test", &test_views)]
        .into_iter()
        .zip(summaries)
    {
        if indices.is_empty() {
            continue;
        }
        println!(
            "{name:<8}{:>8}{:>12.2}{:>12.2}{:>12.2}{:>11.1}%{:>12.2}",
            summary.views,
            summary.srgb_psnr,
            summary.worst_srgb_psnr,
            summary.linear_psnr,
            100.0 * summary.coverage,
            summary.covered_srgb_psnr
        );
    }
    renderer.destroy();
}

/// Discs from the COLMAP sparse points.
///
/// The cheap geometry, and the one that needs no training. It covers whatever
/// COLMAP could triangulate, which is the textured parts of the scene and not
/// the blank wall behind them.
fn surfels_from_sparse(
    reconstruction: &train::colmap::Reconstruction,
    capture: &train::inverse::capture::Capture,
    args: &Args,
) -> Vec<vol::relight::Surfel> {
    let points: Vec<glam::Vec3> = reconstruction
        .points
        .iter()
        .map(|p| glam::Vec3::new(p.xyz[0] as f32, p.xyz[1] as f32, p.xyz[2] as f32))
        .collect();
    let cameras: Vec<glam::Vec3> = capture
        .views
        .iter()
        .map(|v| glam::Vec3::from(v.camera.cam_position))
        .collect();
    let (surfels, dropped) = train::inverse::surface::surfels_from_points(
        &points,
        &cameras,
        train::inverse::surface::SurfaceOptions {
            radius_factor: args.radius_factor,
            ..Default::default()
        },
    );
    println!(
        "geometry: {} of {} sparse points dropped as outliers",
        dropped,
        points.len()
    );
    surfels
}

/// Discs from where a trained density field absorbed each ray.
fn surfels_from_foam(
    foam: &path::Path,
    capture: &train::inverse::capture::Capture,
    views: &[usize],
    args: &Args,
) -> Vec<vol::relight::Surfel> {
    let mut model = match vol::io::try_load(&foam.to_string_lossy()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot read {}: {e}", foam.display());
            std::process::exit(1);
        }
    };
    if model.adjacency.is_none() {
        println!("geometry: the foam carries no adjacency, building it");
        model.compute_adjacency_default();
    }
    println!("geometry: {} cells in the foam", model.points.len());

    let options = train::inverse::depth::DepthOptions {
        voxel_factor: args.voxel_factor,
        min_alpha: args.min_alpha,
        min_peak: args.min_peak,
        disc_radius: args.disc_radius,
        min_views: args.min_views,
        ..Default::default()
    };
    // Training views only. Tracing the held-out ones would build geometry from
    // the images the score is supposed to be surprised by, and the test number
    // would be measuring nothing.
    let trace_started = std::time::Instant::now();
    let requests: Vec<(vol::CameraParams, u32)> = views
        .iter()
        .map(|&index| {
            let camera = capture.views[index].camera;
            let start =
                train::pipeline::pick_start_cell(&model, glam::Vec3::from(camera.cam_position));
            (camera, start)
        })
        .collect();
    let depth_maps = if args.cpu_depth {
        train::inverse::depth::trace_depths(
            &model,
            &requests,
            capture.width,
            capture.height,
            options.max_steps,
        )
    } else {
        let Some(gpu) = train::fit::try_init_gpu() else {
            eprintln!("cannot initialize a supported GPU for depth tracing");
            std::process::exit(1);
        };
        match train::inverse::depth::trace_depths_gpu(
            &model,
            &requests,
            capture.width,
            capture.height,
            options.max_steps,
            gpu,
        ) {
            Ok(maps) => maps,
            Err(message) => {
                eprintln!("cannot trace depth on the GPU: {message}");
                std::process::exit(1);
            }
        }
    };
    let mut maps = Vec::with_capacity(views.len());
    let mut hit = 0usize;
    let mut total = 0usize;
    for ((camera, _), map) in requests.into_iter().zip(depth_maps) {
        hit += map
            .alpha
            .iter()
            .zip(&map.peak)
            .filter(|&(&a, &p)| a >= options.min_alpha && p >= options.min_peak)
            .count();
        total += map.alpha.len();
        maps.push((camera, map));
    }
    println!(
        "geometry: {:.1}% of traced rays met a surface in {:.1} s",
        100.0 * hit as f64 / total.max(1) as f64,
        trace_started.elapsed().as_secs_f64(),
    );
    let fuse_started = std::time::Instant::now();
    let (mut surfels, voxel) = train::inverse::depth::surfels_from_depth(&maps, options);
    println!(
        "geometry: merged at a cell size of {voxel:.4} world units in {:.1} s",
        fuse_started.elapsed().as_secs_f64(),
    );
    if !args.no_multi_view_refine {
        let refine_started = std::time::Instant::now();
        let refine_views: Vec<train::inverse::refine::RefinementView<'_>> = views
            .iter()
            .zip(&maps)
            .map(
                |(capture_index, entry)| train::inverse::refine::RefinementView {
                    capture_index: *capture_index,
                    depth: Some(&entry.1),
                },
            )
            .collect();
        let stats = train::inverse::refine::refine(
            &mut surfels,
            capture,
            &refine_views,
            train::inverse::refine::RefineOptions::default(),
        );
        println!(
            "geometry: multi-view patches scored {} and moved {} surfels by {:.3} cells on average ({:.1}% lower cost, {:.1} views) in {:.1} s",
            stats.scored,
            stats.moved,
            stats.mean_absolute_offset / voxel,
            100.0 * stats.mean_relative_improvement,
            stats.mean_views,
            refine_started.elapsed().as_secs_f64(),
        );
    }
    surfels
}

/// The height that keeps the camera's aspect ratio at the chosen width.
fn aspect_height(sparse: &path::Path, width: usize) -> Result<usize, String> {
    let cameras = train::colmap::try_load_cameras(&sparse.join("cameras.bin"))
        .map_err(|e| format!("cannot read cameras: {e}"))?;
    let camera = cameras
        .first()
        .ok_or_else(|| "the reconstruction has no cameras".to_string())?;
    Ok(((width * camera.height as usize) / camera.width as usize).max(1))
}

/// Say what the recovered light looks like, in terms that can be checked.
///
/// A single number for the whole sky hides the only thing worth knowing about
/// it — whether it has any structure at all, or whether the fit gave up and
/// returned the uniform it started from.
fn describe_light(environment: &vol::relight::Environment) {
    let mut brightest = (glam::Vec3::ZERO, f32::NEG_INFINITY);
    let mut total = 0.0f64;
    let mut weight = 0.0f64;
    for y in 0..environment.height {
        let v = (y as f32 + 0.5) / environment.height as f32;
        let row = (std::f32::consts::PI * v).sin() as f64;
        for x in 0..environment.width {
            let u = (x as f32 + 0.5) / environment.width as f32;
            let texel = environment.texels[y * environment.width + x];
            let luminance = 0.2126 * texel[0] + 0.7152 * texel[1] + 0.0722 * texel[2];
            if luminance > brightest.1 {
                brightest = (vol::relight::equirect_direction(u, v), luminance);
            }
            total += luminance as f64 * row;
            weight += row;
        }
    }
    let mean = (total / weight.max(1.0e-9)) as f32;
    let direction = brightest.0;
    println!(
        "light: mean {mean:.3}, brightest {:.3} towards ({:.2}, {:.2}, {:.2}), contrast {:.1}x",
        brightest.1,
        direction.x,
        direction.y,
        direction.z,
        brightest.1 / mean.max(1.0e-6)
    );
}
