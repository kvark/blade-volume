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

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_train as train;
use std::{collections, path, sync};

const PBR_SPARSE_CELL_POINTS: usize = 5;
const STATIC_GAUSSIAN_SPARSE_CELL_POINTS: usize = 3;
const STATIC_SPARSE_RADIUS_SCALE: f32 = 15.0 / 14.0;

fn require_compute_gpu(purpose: &str) -> sync::Arc<gpu::Context> {
    train::fit::try_init_gpu().unwrap_or_else(|| {
        eprintln!("cannot initialize a supported GPU for {purpose}");
        std::process::exit(1);
    })
}

#[derive(argh::FromArgs)]
/// Reconstruct geometry, material and light from a posed capture.
struct Args {
    /// path to the COLMAP `sparse/0` directory
    #[argh(option)]
    sparse: String,

    /// path to the COLMAP images directory
    #[argh(option)]
    images: String,

    /// optional foreground-mask directory mirroring the image paths
    #[argh(option)]
    masks: Option<String>,

    /// width to work at (default 320). Height follows the camera's aspect.
    #[argh(option, default = "320")]
    width: usize,

    /// keep every n-th image (default 4)
    #[argh(option, default = "4")]
    stride: usize,

    /// hold out every n-th kept image for testing (default 8, 0 = none)
    #[argh(option, default = "8")]
    test_every: usize,

    /// file containing exact COLMAP image names to hold out, one per line;
    /// takes precedence over --test-every
    #[argh(option)]
    test_list: Option<String>,

    /// materials the surfels are clustered into (default 0 = one per surfel).
    /// Sharing is a prior about the scene rather than something the
    /// photographs said, and it costs both the albedo and the re-rendering.
    #[argh(option, default = "0")]
    materials: usize,

    /// equirectangular width of the recovered environment (default 32)
    #[argh(option, default = "32")]
    environment_width: usize,

    /// known linear-radiance environment; when supplied, fit materials without
    /// trying to recover the capture light
    #[argh(option)]
    environment: Option<String>,

    /// additional aligned image directory used to fit normals under another
    /// known light; repeat together with --normal-environment
    #[argh(option)]
    normal_images: Vec<String>,

    /// measured environment for the corresponding --normal-images capture
    #[argh(option)]
    normal_environment: Vec<String>,

    /// aligned photographs under a light excluded from all fitting; repeat
    /// together with --held-out-environment to score several unseen lights
    #[argh(option)]
    held_out_images: Vec<String>,

    /// measured environment for each --held-out-images capture
    #[argh(option)]
    held_out_environment: Vec<String>,

    /// alternations between solving for albedo and for light (default 24)
    #[argh(option, default = "24")]
    iterations: usize,

    /// brightest diffuse albedo used to fix the material/light scale (default 0.8)
    #[argh(option, default = "0.8")]
    brightest_albedo: f32,

    /// rounds in which roughness and reflectance are re-chosen (default 0;
    /// nonzero is an experimental calibrated-material fit)
    #[argh(option, default = "0")]
    specular_rounds: usize,

    /// disc radius as a multiple of the local point spacing (default 1.4).
    /// Applies to sparse and dense input clouds.
    #[argh(option, default = "1.4")]
    radius_factor: f32,

    /// COLMAP stereo_fusion fused.ply to use instead of sparse geometry
    #[argh(option)]
    dense_cloud: Option<String>,

    /// maximum spatially averaged dense-cloud particles (default 50000)
    #[argh(option, default = "50_000")]
    dense_max_points: usize,

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

    /// shadow rays per shading point when scoring (default 0). A scene
    /// fitted with shadowing has to be rendered with it, or the comparison
    /// measures two renderer settings rather than the scene.
    #[argh(option, default = "0")]
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

    /// masked PowerFoam updates per view for the static light-field branch
    #[argh(option, default = "0")]
    surface_powerfoam_steps_per_view: usize,

    /// optional trained PowerFoam static-light-field PLY output
    #[argh(option)]
    surface_powerfoam_output: Option<String>,

    /// optional directly trained anisotropic Gaussian light-field PLY output
    #[argh(option)]
    gaussian_output: Option<String>,

    /// optional relightable anisotropic Gaussian PLY output
    #[argh(option)]
    pbr_gaussian_output: Option<String>,

    /// direct Gaussian updates, one third appearance then support (default 1500)
    #[argh(option, default = "1500")]
    gaussian_steps: usize,

    /// particles to refine against all training renders (default 0). This is
    /// an expensive final coordinate pass; a value larger than the observed
    /// particle count refines the complete visible surface.
    #[argh(option, default = "0")]
    render_refine: usize,

    /// simultaneous full-cloud render refinement rounds (default 0; use 8 for
    /// the conservative final-quality schedule)
    #[argh(option, default = "0")]
    render_refine_rounds: usize,

    /// refine PBR Gaussian radii against complete renders
    #[argh(switch)]
    render_refine_radii: bool,

    /// refine a small shared diffuse-material table against complete renders
    #[argh(switch)]
    render_refine_materials: bool,

    /// refine Gaussian normals against complete renders from the primary measured light
    #[argh(switch)]
    render_refine_normals: bool,

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
    let fit_gaussians = args.gaussian_output.is_some() || args.pbr_gaussian_output.is_some();

    if let Err(message) = validate_lit_captures(&args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    if let Err(message) = validate_geometry_source(&args) {
        eprintln!("{message}");
        std::process::exit(1);
    }

    let known_light = args.environment.as_ref().map(|file| {
        vol::io::try_load_environment(path::Path::new(file)).unwrap_or_else(|error| {
            eprintln!("cannot read known environment {file}: {error}");
            std::process::exit(1);
        })
    });
    if let Some(ref file) = args.environment {
        println!("light: fixed from {file}");
    }
    let environment_width = known_light
        .as_ref()
        .map_or(args.environment_width, |light| light.width);

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
    let (capture, reconstruction) = match train::inverse::capture::Capture::from_colmap_with_masks(
        sparse,
        images,
        args.masks.as_deref().map(path::Path::new),
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
    let test_names = args
        .test_list
        .as_deref()
        .map(path::Path::new)
        .map(train::pipeline::read_image_list)
        .transpose()
        .unwrap_or_else(|message| {
            eprintln!("{message}");
            std::process::exit(1);
        });
    let (train_views, test_views) = test_names
        .as_ref()
        .map_or_else(
            || Ok(capture.split(args.test_every)),
            |names| capture.split_named(names),
        )
        .unwrap_or_else(|message| {
            eprintln!("cannot split capture: {message}");
            std::process::exit(1);
        });
    if !args.held_out_images.is_empty() && test_views.is_empty() {
        eprintln!("held-out-light scoring needs at least one held camera view");
        std::process::exit(1);
    }
    if args.dense_cloud.is_some() && !test_views.is_empty() {
        eprintln!(
            "dense-cloud evaluation is valid only when fused.ply was built from the training cameras"
        );
    }
    if args.surface_powerfoam_steps_per_view > 0 && args.masks.is_none() {
        eprintln!("surface PowerFoam continuation requires --masks");
        std::process::exit(1);
    }
    if args.surface_powerfoam_steps_per_view > 0 && args.compact_kernel {
        eprintln!("surface PowerFoam continuation requires Gaussian particles");
        std::process::exit(1);
    }
    if args.surface_powerfoam_output.is_some() && args.surface_powerfoam_steps_per_view == 0 {
        eprintln!("--surface-powerfoam-output requires --surface-powerfoam-steps-per-view");
        std::process::exit(1);
    }
    if fit_gaussians && args.gaussian_steps < 2 {
        eprintln!("Gaussian output requires at least two --gaussian-steps");
        std::process::exit(1);
    }
    if fit_gaussians && args.compact_kernel {
        eprintln!("Gaussian output requires Gaussian particles");
        std::process::exit(1);
    }
    let mut compute_gpu =
        (args.foam.is_some() && !args.cpu_depth).then(|| require_compute_gpu("depth tracing"));
    let normal_captures =
        load_normal_captures(&capture, sparse, height, &args).unwrap_or_else(|message| {
            eprintln!("cannot load normal captures: {message}");
            std::process::exit(1);
        });
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
    let mut sparse_gaussian_surface = if uses_sparse_static_gaussian_surface(&args) {
        let surfels = surfels_from_training_sparse(&reconstruction, &capture, &train_views, &args);
        // Pose-only datasets have no sparse tracks. In that case the selected
        // dense cloud is also the only point-cloud initializer for the direct
        // output.
        (!surfels.is_empty()).then_some(surfels)
    } else {
        None
    };
    let (surfels, sparse_support, gaussian_sparse_support, mut density_normal_source) =
        match (args.foam.as_deref(), args.dense_cloud.as_deref()) {
            (Some(foam), None) => {
                let (surfels, sparse_support, gaussian_sparse_support, source) = surfels_from_foam(
                    path::Path::new(foam),
                    &capture,
                    &reconstruction,
                    &train_views,
                    &normal_captures,
                    &args,
                    compute_gpu.clone(),
                );
                (
                    surfels,
                    sparse_support,
                    gaussian_sparse_support,
                    Some(source),
                )
            }
            (None, Some(dense)) => (
                surfels_from_dense(path::Path::new(dense), &args),
                Vec::new(),
                Vec::new(),
                None,
            ),
            (None, None) => (
                surfels_from_sparse(&reconstruction, &capture, &args),
                Vec::new(),
                Vec::new(),
                None,
            ),
            (Some(_), Some(_)) => unreachable!("geometry source was validated"),
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
        if let Some(ref mut surfels) = sparse_gaussian_surface {
            train::inverse::surface::orient_towards_views(surfels, &capture);
        }
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

    let mut static_gaussian_surface = ((density_normal_source.is_some()
        && !args.normal_images.is_empty()
        && args.gaussian_output.is_some())
        || args.surface_powerfoam_steps_per_view > 0)
        .then(|| prepare_static_surface(geometry.clone(), density_normal_source.as_ref()));
    if !args.normal_images.is_empty() {
        let started = std::time::Instant::now();
        let stats = refine_normals_from_captures(
            &mut geometry,
            &capture,
            &train_views,
            known_light.as_ref().expect("known light validated above"),
            &normal_captures,
        );
        println!(
            "geometry: photometrically refined {} of {} supported normals from {} known lights in {:.1} s",
            stats.changed,
            stats.supported,
            args.normal_images.len() + 1,
            started.elapsed().as_secs_f64(),
        );
    }
    if let Some(source) = density_normal_source.take() {
        let blend = if args.normal_images.is_empty() {
            0.1
        } else {
            0.2
        };
        let refined = source.refine(&mut geometry, blend);
        println!("geometry: density-gradient refined {refined} foam-surface normals");
    }

    if args.surface_powerfoam_steps_per_view > 0 {
        let started = std::time::Instant::now();
        drop(compute_gpu.take());
        let powerfoam_gpu = require_compute_gpu("surface PowerFoam continuation");
        let options = train::inverse::powerfoam::ContinueOptions {
            steps_per_view: args.surface_powerfoam_steps_per_view,
            ..Default::default()
        };
        let baseline_surface = static_gaussian_surface
            .clone()
            .unwrap_or_else(|| geometry.clone());
        let selection = if args.gaussian_output.is_some()
            && sparse_gaussian_surface.is_none()
            && train_views.len() >= 4
        {
            let (&validation_view, fit_views) = train_views.split_last().unwrap();
            let mut probe_surface = baseline_surface.clone();
            train::inverse::powerfoam::continue_surface(
                &mut probe_surface,
                &capture,
                fit_views,
                options,
                powerfoam_gpu.clone(),
            )
            .unwrap_or_else(|message| {
                eprintln!("cannot validate the surface continuation: {message}");
                std::process::exit(1);
            });
            let baseline_probe =
                static_surface_with_sparse(baseline_surface.clone(), &gaussian_sparse_support);
            let continued_probe =
                static_surface_with_sparse(probe_surface, &gaussian_sparse_support);
            Some(
                train::gaussian_splat::validate_static_surface_continuation(
                    &baseline_probe,
                    &continued_probe,
                    &capture,
                    fit_views,
                    validation_view,
                    args.gaussian_steps,
                    powerfoam_gpu.clone(),
                )
                .unwrap_or_else(|message| {
                    eprintln!("cannot validate the static Gaussian continuation: {message}");
                    std::process::exit(1);
                }),
            )
        } else {
            None
        };
        let surface = static_continuation_surface(&mut static_gaussian_surface, &geometry);
        let outcome = train::inverse::powerfoam::continue_surface(
            surface,
            &capture,
            &train_views,
            options,
            powerfoam_gpu.clone(),
        )
        .unwrap_or_else(|message| {
            eprintln!("cannot continue the extracted surface: {message}");
            std::process::exit(1);
        });
        if let Some(ref output) = args.surface_powerfoam_output {
            convert::save_ply(path::Path::new(output), &outcome.light_field).unwrap_or_else(
                |error| {
                    eprintln!("cannot write surface PowerFoam {output}: {error:?}");
                    std::process::exit(1);
                },
            );
            println!("wrote {output}");
        }
        let stats = outcome.stats;
        println!(
            "geometry: continued through masked PowerFoam for {} updates, loss {:.6} -> {:.6}, in {:.1} s",
            stats.updates,
            stats.initial_loss,
            stats.final_loss,
            started.elapsed().as_secs_f64(),
        );
        compute_gpu = Some(powerfoam_gpu);
        if let Some(selection) = selection {
            println!(
                "static Gaussian continuation: {} at {:.2} -> {:.2} dB on withheld training view",
                if selection.use_continued {
                    "selected"
                } else {
                    "rejected"
                },
                selection.baseline_validation_psnr,
                selection.continued_validation_psnr,
            );
            if !selection.use_continued {
                static_gaussian_surface = Some(baseline_surface);
            }
        }
    }

    // ---------------------------------------------------- material and light
    let started = std::time::Instant::now();
    let (mut observations, observation_diagnostics) = if args.observation_diagnostics {
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
    let sparse_added = sparse_support.len();
    if sparse_added > 0 {
        let sparse_start = observations.surfels() - sparse_added;
        let sparse_seen = (sparse_start..observations.surfels())
            .filter(|&index| !observations.of(index).is_empty())
            .count();
        println!(
            "observed: {sparse_seen} of {sparse_added} sparse-track additions seen by at least one training view"
        );
    }
    let sparse_supplemented = supplement_sparse_track_observations(
        &mut observations,
        &geometry,
        &capture,
        &sparse_support,
        train::inverse::decompose::FitOptions::default().min_facing,
    );
    if sparse_supplemented > 0 {
        println!(
            "observed: supplemented {sparse_supplemented} otherwise-unseen sparse-track additions"
        );
    }
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

    let shadows = if args.no_shadows || args.diffuse_samples == 0 {
        None
    } else {
        let started = std::time::Instant::now();
        let directions = train::inverse::decompose::environment_directions(environment_width);
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
    let mut fitted = train::inverse::decompose::fit(
        &geometry,
        &observations,
        train::inverse::decompose::FitOptions {
            materials: args.materials,
            environment_width,
            iterations: args.iterations,
            specular_rounds: args.specular_rounds,
            brightest_albedo: args.brightest_albedo,
            ..Default::default()
        },
        train::inverse::decompose::Given {
            visibility: shadows.as_ref(),
            light: known_light.as_ref(),
        },
    );
    println!(
        "decomposed: {} materials, residual {:.4}, {} surfels unseen, in {:.1} s",
        fitted.scene.model.materials.len(),
        fitted.residual,
        fitted.unseen,
        started.elapsed().as_secs_f64()
    );
    describe_light(fitted.scene.environment());

    // Smaller light sets and shared palettes fail held-light transfer gates;
    // five broad directions with one material per particle pass both tails.
    if normal_captures.len() >= 4 && args.materials == 0 {
        let stats = refine_materials_from_captures(
            &mut fitted.scene.model,
            &observations,
            &train_views,
            known_light
                .as_ref()
                .expect("known light validated for additional captures"),
            &normal_captures,
            args.brightest_albedo,
        );
        println!(
            "materials: jointly fitted {} of {} entries across {} known lights ({} coordinates changed)",
            stats.supported,
            fitted.scene.model.materials.len(),
            normal_captures.len() + 1,
            stats.changed,
        );
    }

    if args.render_refine_normals {
        let evidence = [train::inverse::refine::RenderedNormalEvidence {
            capture: &capture,
            indices: &train_views,
            environment: known_light.as_ref().expect("known light validated above"),
        }];
        let stats = train::inverse::refine::refine_rendered_normals(
            &mut fitted.scene,
            &evidence,
            &observations,
            args.diffuse_samples,
            8,
            2.5,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot refine rendered normals: {error}");
            std::process::exit(1);
        });
        println!(
            "rendered normals: changed {} candidates in {} rounds ({} accepted), loss {:.7} -> {:.7}, in {:.1} s",
            stats.normals,
            stats.rounds,
            stats.accepted,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
    }

    if args.render_refine_materials {
        let stats = train::inverse::refine::refine_rendered_materials(
            &mut fitted.scene,
            &capture,
            &train_views,
            args.diffuse_samples,
            0.025,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot refine rendered materials: {error}");
            std::process::exit(1);
        });
        println!(
            "rendered materials: changed {} of {} coordinates in {} proposals, loss {:.7} -> {:.7}, in {:.1} s",
            stats.changed,
            stats.coordinates,
            stats.proposals,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
    }

    if args.render_refine > 0 || args.render_refine_rounds > 0 {
        let stats = train::inverse::refine::refine_rendered(
            &mut fitted.scene,
            &capture,
            &train_views,
            &observations,
            args.diffuse_samples,
            args.render_refine_rounds,
            args.render_refine_radii,
            args.render_refine,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot refine the rendered surface: {error}");
            std::process::exit(1);
        });
        println!(
            "rendered surface: {} particles in {} rounds ({} accepted); tested {}, moved {}, radii {}, loss {:.7} -> {:.7}, in {:.1} s",
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
    }

    let mut learned_pbr_gaussian = None;
    let mut pbr_support_baseline = None;
    if fit_gaussians && args.gaussian_output.is_none() {
        let gpu = compute_gpu
            .get_or_insert_with(|| require_compute_gpu("PBR Gaussian training"))
            .clone();
        let mut pbr_gaussian =
            train::gaussian_splat::pbr_from_surface(&fitted.scene.model, train_views.len())
                .unwrap_or_else(|error| {
                    eprintln!("cannot initialize PBR support Gaussian field: {error}");
                    std::process::exit(1);
                });
        let established_support = pbr_gaussian.clone();
        let started = std::time::Instant::now();
        let stats = train::gaussian_splat::fit_staged(
            &mut pbr_gaussian,
            &capture,
            &train_views,
            args.gaussian_steps,
            gpu,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot fit PBR Gaussian support: {error}");
            std::process::exit(1);
        });
        println!(
            "Gaussian PBR only: {} appearance and {} support updates in {:.1} s",
            stats.appearance.steps,
            stats.support.steps,
            started.elapsed().as_secs_f64(),
        );
        if guard_pbr_support(&mut pbr_gaussian, &established_support, "single-light fit") {
            match_pbr_surface_extent(&mut pbr_gaussian, &fitted.scene.model);
        } else {
            update_surface_radii(&mut fitted.scene.model, &pbr_gaussian);
        }
        pbr_support_baseline = Some(pbr_gaussian.clone());
        learned_pbr_gaussian = Some(pbr_gaussian);
    } else if fit_gaussians {
        let gaussian_surface = if let Some(ref surfels) = sparse_gaussian_surface {
            vol::relight::RelightModel {
                kernel: vol::relight::ParticleKernel::Gaussian,
                surfels: surfels
                    .iter()
                    .copied()
                    .map(|mut surfel| {
                        surfel.material = 0;
                        surfel
                    })
                    .collect(),
                materials: vec![vol::relight::Material::default()],
            }
        } else {
            static_surface_with_sparse(
                static_gaussian_surface
                    .clone()
                    .unwrap_or_else(|| fitted.scene.model.clone()),
                &gaussian_sparse_support,
            )
        };
        let mut gaussian =
            train::gaussian_splat::from_surface(&gaussian_surface).unwrap_or_else(|error| {
                eprintln!("cannot initialize direct Gaussian field: {error}");
                std::process::exit(1);
            });
        let initial_test =
            score_gaussian_test(&gaussian, &capture, &test_views).unwrap_or_else(|error| {
                eprintln!("cannot score direct Gaussian field: {error}");
                std::process::exit(1);
            });
        let gpu = compute_gpu
            .get_or_insert_with(|| require_compute_gpu("direct Gaussian training"))
            .clone();
        let independent_outputs =
            sparse_gaussian_surface.is_some() || static_gaussian_surface.is_some();
        let mut pbr_gaussian = if independent_outputs {
            train::gaussian_splat::pbr_from_surface(&fitted.scene.model, train_views.len())
        } else {
            train::gaussian_splat::from_surface(&fitted.scene.model)
        }
        .unwrap_or_else(|error| {
            eprintln!("cannot initialize PBR support Gaussian field: {error}");
            std::process::exit(1);
        });
        let established_support = pbr_gaussian.clone();
        let started = std::time::Instant::now();
        let (stats, shared_appearance) = if independent_outputs {
            let stats = train::gaussian_splat::fit_staged_independent_outputs(
                &mut pbr_gaussian,
                &mut gaussian,
                &capture,
                &train_views,
                args.gaussian_steps,
                gpu,
            )
            .unwrap_or_else(|error| {
                eprintln!("cannot fit independent Gaussian outputs: {error}");
                std::process::exit(1);
            });
            (stats, false)
        } else {
            let stats = train::gaussian_splat::fit_staged_outputs(
                &mut pbr_gaussian,
                &mut gaussian,
                &capture,
                &train_views,
                args.gaussian_steps,
                gpu,
            )
            .unwrap_or_else(|error| {
                eprintln!("cannot fit Gaussian outputs: {error}");
                std::process::exit(1);
            });
            (stats, true)
        };
        let fit_seconds = started.elapsed().as_secs_f64();
        if shared_appearance {
            println!(
                "Gaussian outputs: {} shared appearance, {} PBR support, and {} light-field support updates in {:.1} s",
                stats.appearance.steps,
                stats.pbr_support.steps,
                stats.light_field_support.steps,
                fit_seconds,
            );
        } else {
            println!(
                "Gaussian outputs: independent PBR and static light-field fits in {:.1} s",
                fit_seconds,
            );
        }
        if let Some(ref output) = args.gaussian_output {
            let output = path::Path::new(output);
            if let Some(parent) = output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent).unwrap_or_else(|error| {
                    eprintln!("cannot create {}: {error}", parent.display());
                    std::process::exit(1);
                });
            }
            convert::save_ply(output, &gaussian).unwrap_or_else(|error| {
                eprintln!(
                    "cannot write direct Gaussian {}: {error:?}",
                    output.display()
                );
                std::process::exit(1);
            });
            gaussian = vol::io::try_load_gaussian(output.to_str().unwrap_or_else(|| {
                eprintln!("output path is not UTF-8: {}", output.display());
                std::process::exit(1);
            }))
            .unwrap_or_else(|error| {
                eprintln!(
                    "cannot reload written direct Gaussian {}: {error}",
                    output.display()
                );
                std::process::exit(1);
            });
            println!("wrote and reloaded {}", output.display());
        }
        let test = score_gaussian_test(&gaussian, &capture, &test_views).unwrap_or_else(|error| {
            eprintln!("cannot score direct Gaussian field: {error}");
            std::process::exit(1);
        });
        println!(
            "direct Gaussian: {} appearance updates {:.6} -> {:.6}, {} support updates {:.6} -> {:.6} ({:.1} s total Gaussian fitting)",
            stats.appearance.steps,
            stats.appearance.initial_loss,
            stats.appearance.final_loss,
            stats.light_field_support.steps,
            stats.light_field_support.initial_loss,
            stats.light_field_support.final_loss,
            fit_seconds,
        );
        if let (Some((initial_mean, initial_worst)), Some((mean, worst))) = (initial_test, test) {
            println!(
                "light-field test: {initial_mean:.2} -> {mean:.2} dB mean, {initial_worst:.2} -> {worst:.2} dB worst"
            );
        }
        let support_restored =
            guard_pbr_support(&mut pbr_gaussian, &established_support, "single-light fit");
        if let Some(ref directory) = args.dump {
            dump_static_gaussian(
                &gaussian,
                &capture,
                &test_views,
                &path::Path::new(directory).join("static-light"),
            )
            .unwrap_or_else(|error| {
                eprintln!("cannot dump static Gaussian comparisons: {error}");
                std::process::exit(1);
            });
        }
        if support_restored {
            match_pbr_surface_extent(&mut pbr_gaussian, &fitted.scene.model);
        } else {
            update_surface_radii(&mut fitted.scene.model, &pbr_gaussian);
        }
        pbr_support_baseline = Some(pbr_gaussian.clone());
        learned_pbr_gaussian = Some(pbr_gaussian);
    }
    if args.render_refine_radii && fit_gaussians {
        let environment = fitted.scene.environment().clone();
        let evidence = [train::inverse::refine::RenderedNormalEvidence {
            capture: &capture,
            indices: &train_views,
            environment: &environment,
        }];
        let stats = train::inverse::refine::refine_rendered_radii(
            &mut fitted.scene,
            &evidence,
            &observations,
            args.diffuse_samples,
            8,
            0.1,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot refine rendered radii: {error}");
            std::process::exit(1);
        });
        println!(
            "rendered radii: changed {} candidates in {} rounds ({} accepted), loss {:.7} -> {:.7}, in {:.1} s",
            stats.radii,
            stats.rounds,
            stats.accepted,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
        if args.render_refine_normals {
            let stats = train::inverse::refine::refine_rendered_normals(
                &mut fitted.scene,
                &evidence,
                &observations,
                args.diffuse_samples,
                8,
                2.5,
            )
            .unwrap_or_else(|error| {
                eprintln!("cannot refine post-support normals: {error}");
                std::process::exit(1);
            });
            println!(
                "post-support normals: changed {} candidates in {} rounds ({} accepted), loss {:.7} -> {:.7}, in {:.1} s",
                stats.normals,
                stats.rounds,
                stats.accepted,
                stats.initial_loss,
                stats.final_loss,
                stats.seconds,
            );
        }
    }
    if args.render_refine_materials && fit_gaussians {
        let stats = train::inverse::refine::polish_rendered_materials(
            &mut fitted.scene,
            &capture,
            &train_views,
            args.diffuse_samples,
            0.0125,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot polish post-support materials: {error}");
            std::process::exit(1);
        });
        println!(
            "post-support materials: changed {} of {} coordinates in {} proposals, loss {:.7} -> {:.7}, in {:.1} s",
            stats.changed,
            stats.coordinates,
            stats.proposals,
            stats.initial_loss,
            stats.final_loss,
            stats.seconds,
        );
        let assignment_stats = train::inverse::refine::refine_rendered_material_assignments(
            &mut fitted.scene,
            &capture,
            &train_views,
            &observations,
            args.diffuse_samples,
            0.0125,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot refine rendered material assignments: {error}");
            std::process::exit(1);
        });
        println!(
            "rendered material assignments: {} candidates, {} proposals, {} of {} particles changed, loss {:.7} -> {:.7}, in {:.1} s",
            assignment_stats.candidates,
            assignment_stats.proposals,
            assignment_stats.changed,
            assignment_stats.particles,
            assignment_stats.initial_loss,
            assignment_stats.final_loss,
            assignment_stats.seconds,
        );
    }

    if let Some(ref mut gaussian) = learned_pbr_gaussian {
        let mut multilight_geometry_fitted = false;
        if !normal_captures.is_empty() && args.pbr_gaussian_output.is_some() {
            let mut lights = Vec::with_capacity(normal_captures.len() + 1);
            lights.push(train::gaussian_splat::KnownLightCapture {
                capture: &capture,
                environment: known_light
                    .as_ref()
                    .expect("known light validated for additional captures"),
            });
            lights.extend(normal_captures.iter().map(|entry| {
                train::gaussian_splat::KnownLightCapture {
                    capture: &entry.capture,
                    environment: &entry.environment,
                }
            }));
            let gpu = compute_gpu
                .get_or_insert_with(|| require_compute_gpu("multi-light Gaussian geometry"))
                .clone();
            let started = std::time::Instant::now();
            let stats = train::gaussian_splat::fit_multilight_geometry(
                gaussian,
                &mut fitted.scene.model,
                &lights,
                &train_views,
                gpu,
            )
            .unwrap_or_else(|error| {
                eprintln!("cannot fit multi-light Gaussian geometry: {error}");
                std::process::exit(1);
            });
            println!(
                "multi-light Gaussian geometry: {} calibrated lights, {} optimizer updates in {:.1} s",
                lights.len(),
                stats.iter().map(|stats| stats.steps).sum::<usize>(),
                started.elapsed().as_secs_f64(),
            );
            multilight_geometry_fitted = true;
        }
        guard_pbr_support(
            gaussian,
            pbr_support_baseline
                .as_ref()
                .expect("learned PBR Gaussian has an established support baseline"),
            "multi-light fit",
        );
        if args.render_refine_radii {
            train::gaussian_splat::apply_surface_radius_feedback(gaussian, &fitted.scene.model)
                .unwrap_or_else(|error| {
                    eprintln!("cannot apply final PBR support to Gaussian geometry: {error}");
                    std::process::exit(1);
                });
        }
        train::gaussian_splat::attach_pbr(gaussian, &fitted.scene.model).unwrap_or_else(|error| {
            eprintln!("cannot attach final PBR attributes to Gaussian geometry: {error}");
            std::process::exit(1);
        });
        if args.render_refine_materials && multilight_geometry_fitted {
            let stats = train::inverse::refine::polish_gaussian_materials(
                &fitted.scene,
                gaussian,
                &capture,
                &train_views,
                0.025,
            )
            .unwrap_or_else(|error| {
                eprintln!("cannot polish final Gaussian materials: {error}");
                std::process::exit(1);
            });
            println!(
                "final Gaussian materials: changed {} of {} coordinates in {} proposals, loss {:.7} -> {:.7}, in {:.1} s",
                stats.changed,
                stats.coordinates,
                stats.proposals,
                stats.initial_loss,
                stats.final_loss,
                stats.seconds,
            );
        }
        let removed = train::gaussian_splat::prune_low_opacity(gaussian).unwrap_or_else(|error| {
            eprintln!("cannot prune low-opacity PBR Gaussian particles: {error}");
            std::process::exit(1);
        });
        println!("pruned {removed} low-opacity PBR Gaussian particles");
        if let Some(ref output) = args.pbr_gaussian_output {
            let output = path::Path::new(output);
            if let Some(parent) = output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent).unwrap_or_else(|error| {
                    eprintln!("cannot create {}: {error}", parent.display());
                    std::process::exit(1);
                });
            }
            convert::save_ply(output, gaussian).unwrap_or_else(|error| {
                eprintln!(
                    "cannot write relightable Gaussian {}: {error:?}",
                    output.display()
                );
                std::process::exit(1);
            });
            *gaussian = vol::io::try_load_gaussian(output.to_str().unwrap_or_else(|| {
                eprintln!("output path is not UTF-8: {}", output.display());
                std::process::exit(1);
            }))
            .unwrap_or_else(|error| {
                eprintln!(
                    "cannot reload written relightable Gaussian {}: {error}",
                    output.display()
                );
                std::process::exit(1);
            });
            println!("wrote and reloaded {}", output.display());
            let environment = output.with_extension("f32");
            vol::io::try_save_environment(&environment, fitted.scene.environment()).unwrap_or_else(
                |error| {
                    eprintln!("cannot write {}: {error}", environment.display());
                    std::process::exit(1);
                },
            );
            println!("wrote {}", environment.display());
        }
    }

    if let Some(ref output) = args.output {
        let output = path::Path::new(output);
        if let Err(e) = vol::io::try_save_relight(output, &fitted.scene.model) {
            eprintln!("cannot write {}: {e}", output.display());
        } else {
            println!("wrote {}", output.display());
        }
        let environment = output.with_extension("f32");
        if let Err(e) = vol::io::try_save_environment(&environment, fitted.scene.environment()) {
            eprintln!("cannot write {}: {e}", environment.display());
        } else {
            println!("wrote {}", environment.display());
        }
    }

    // ---------------------------------------------------------------- score
    // Load the withheld light only after every fitting stage has finished, so
    // neither its pixels nor its environment can accidentally select a model.
    drop(normal_captures);
    let held_out_captures =
        load_held_out_captures(&capture, sparse, height, &args).unwrap_or_else(|message| {
            eprintln!("cannot load held-out capture: {message}");
            std::process::exit(1);
        });
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

    if !test_views.is_empty() {
        let baseline = capture_baseline_psnr(None, &capture, &test_views);
        println!(
            "held-camera black baseline: {:.2}/{:.2} dB whole-frame, {} foreground mean/worst",
            baseline.mean,
            baseline.worst,
            baseline.foreground_text(),
        );
    }
    for (index, held) in held_out_captures.iter().enumerate() {
        let black = capture_baseline_psnr(None, &held.capture, &test_views);
        let copy = capture_baseline_psnr(Some(&capture), &held.capture, &test_views);
        if held_out_captures.len() > 1 {
            println!("held-light {index}: {}", args.held_out_environment[index]);
        }
        let label = if held_out_captures.len() == 1 {
            String::new()
        } else {
            format!(" {index}")
        };
        println!(
            "held-light{label} baselines: black {:.2}/{:.2} dB whole-frame, {} foreground; \
             capture-light copy {:.2}/{:.2} dB whole-frame, {} foreground mean/worst",
            black.mean,
            black.worst,
            black.foreground_text(),
            copy.mean,
            copy.worst,
            copy.foreground_text(),
        );
    }

    println!(
        "\n{:<12}{:>8}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "split",
        "views",
        "psnr srgb",
        "worst",
        "psnr linear",
        "mask psnr",
        "mask worst",
        "coverage",
        "mask rec",
        "mask prec",
        "where hit"
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
        print_reconstruction_summary(name, summary);
    }
    if let Some(ref gaussian) = learned_pbr_gaussian {
        let summaries =
            renderer.score_gaussian_splits(&fitted.scene, gaussian, &capture, &splits, 0);
        for ((name, indices), summary) in [("g-train", &train_views), ("g-test", &test_views)]
            .into_iter()
            .zip(summaries)
        {
            if indices.is_empty() {
                continue;
            }
            print_reconstruction_summary(name, summary);
        }
    }
    for (index, held) in held_out_captures.iter().enumerate() {
        let held_scene =
            train::inverse::score::Scene::new(fitted.scene.model.clone(), held.environment.clone());
        let held_dump = dump.map(|directory| {
            if held_out_captures.len() == 1 {
                directory.join("held-light/scalar")
            } else {
                directory.join(format!("held-light/{index}/scalar"))
            }
        });
        if let Some(ref directory) = held_dump {
            let _ = std::fs::create_dir_all(directory);
        }
        let held_splits = [(test_views.as_slice(), held_dump.as_deref())];
        let summary = renderer.score_splits(
            &held_scene,
            &held.capture,
            &held_splits,
            args.diffuse_samples,
        )[0];
        let label = if held_out_captures.len() == 1 {
            "relight".to_string()
        } else {
            format!("relight-{index}")
        };
        print_reconstruction_summary(&label, summary);
        if let Some(ref gaussian) = learned_pbr_gaussian {
            let gaussian_dump = dump.map(|directory| {
                if held_out_captures.len() == 1 {
                    directory.join("held-light/gaussian")
                } else {
                    directory.join(format!("held-light/{index}/gaussian"))
                }
            });
            if let Some(ref directory) = gaussian_dump {
                let _ = std::fs::create_dir_all(directory);
            }
            let gaussian_splits = [(test_views.as_slice(), gaussian_dump.as_deref())];
            let summary = renderer.score_gaussian_splits(
                &held_scene,
                gaussian,
                &held.capture,
                &gaussian_splits,
                0,
            )[0];
            let label = if held_out_captures.len() == 1 {
                "g-relight".to_string()
            } else {
                format!("g-relight-{index}")
            };
            print_reconstruction_summary(&label, summary);
        }
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

/// Discs from COLMAP dense stereo fusion, retaining its independent normals.
fn surfels_from_dense(dense: &path::Path, args: &Args) -> Vec<vol::relight::Surfel> {
    let loaded = train::dense::try_load_colmap_fused(dense).unwrap_or_else(|error| {
        eprintln!("cannot read dense cloud {}: {error}", dense.display());
        std::process::exit(1);
    });
    let points = train::dense::downsample(&loaded, args.dense_max_points);
    println!(
        "geometry: spatially retained {} of {} COLMAP dense points",
        points.len(),
        loaded.len(),
    );
    let oriented: Vec<_> = points
        .iter()
        .map(|point| (point.position, point.normal))
        .collect();
    let (surfels, dropped) = train::inverse::surface::surfels_from_oriented_points(
        &oriented,
        train::inverse::surface::SurfaceOptions {
            radius_factor: args.radius_factor,
            ..Default::default()
        },
    );
    println!(
        "geometry: {dropped} of {} dense points dropped as unsupported outliers",
        points.len(),
    );
    surfels
}

fn surfels_from_training_sparse(
    reconstruction: &train::colmap::Reconstruction,
    capture: &train::inverse::capture::Capture,
    views: &[usize],
    args: &Args,
) -> Vec<vol::relight::Surfel> {
    let image_ids: collections::HashMap<&str, u32> = reconstruction
        .images
        .iter()
        .map(|image| (image.name.as_str(), image.id))
        .collect();
    let training_images: collections::HashSet<u32> = views
        .iter()
        .filter_map(|&index| image_ids.get(capture.views[index].name.as_str()).copied())
        .collect();
    let points: Vec<glam::Vec3> = reconstruction
        .points
        .iter()
        .filter(|point| has_training_track(&point.track_image_ids, &training_images))
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
            radius_factor: args.radius_factor * STATIC_SPARSE_RADIUS_SCALE,
            ..Default::default()
        },
    );
    println!(
        "geometry: retained {} of {} sparse points with a training track for the static Gaussian; {} retained points dropped as outliers",
        points.len(),
        reconstruction.points.len(),
        dropped,
    );
    surfels
}

fn has_training_track(
    track_image_ids: &[u32],
    training_images: &collections::HashSet<u32>,
) -> bool {
    track_image_ids
        .iter()
        .any(|image| training_images.contains(image))
}

fn neutral_static_surface(mut surface: vol::relight::RelightModel) -> vol::relight::RelightModel {
    for surfel in &mut surface.surfels {
        surfel.material = 0;
    }
    surface.materials = vec![vol::relight::Material::default()];
    surface
}

fn prepare_static_surface(
    mut surface: vol::relight::RelightModel,
    density_normal_source: Option<&DensityNormalSource>,
) -> vol::relight::RelightModel {
    if let Some(source) = density_normal_source {
        source.refine(&mut surface, 0.1);
    }
    neutral_static_surface(surface)
}

fn static_surface_with_sparse(
    surface: vol::relight::RelightModel,
    sparse: &[vol::relight::Surfel],
) -> vol::relight::RelightModel {
    let mut surface = neutral_static_surface(surface);
    surface
        .surfels
        .extend(sparse.iter().copied().map(|mut surfel| {
            surfel.material = 0;
            surfel
        }));
    surface
}

fn static_continuation_surface<'a>(
    static_surface: &'a mut Option<vol::relight::RelightModel>,
    pbr_surface: &vol::relight::RelightModel,
) -> &'a mut vol::relight::RelightModel {
    static_surface.get_or_insert_with(|| pbr_surface.clone())
}

/// Discs from where a trained density field absorbed each ray.
fn surfels_from_foam(
    foam: &path::Path,
    capture: &train::inverse::capture::Capture,
    reconstruction: &train::colmap::Reconstruction,
    views: &[usize],
    normal_captures: &[LitCapture],
    args: &Args,
    gpu: Option<sync::Arc<gpu::Context>>,
) -> (
    Vec<vol::relight::Surfel>,
    Vec<SparseSupport>,
    Vec<vol::relight::Surfel>,
    DensityNormalSource,
) {
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
        let gpu = gpu.expect("GPU depth tracing context was not initialized");
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
    for ((&view_index, (camera, _)), mut map) in views.iter().zip(requests).zip(depth_maps) {
        total += mask_depth(&mut map, &capture.views[view_index]);
        hit += map
            .alpha
            .iter()
            .zip(&map.peak)
            .filter(|&(&a, &p)| a >= options.min_alpha && p >= options.min_peak)
            .count();
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
    let foam_surface_count = surfels.len();
    let sparse_support = add_sparse_track_support(
        &mut surfels,
        reconstruction,
        capture,
        views,
        voxel,
        options.disc_radius,
        PBR_SPARSE_CELL_POINTS,
    );
    println!(
        "geometry: added {} particles from training-only sparse tracks",
        sparse_support.len()
    );
    let gaussian_sparse_support = if args.gaussian_output.is_some() {
        let core_count = surfels.len();
        let mut expanded = surfels.clone();
        add_sparse_track_support(
            &mut expanded,
            reconstruction,
            capture,
            views,
            voxel,
            options.disc_radius,
            STATIC_GAUSSIAN_SPARSE_CELL_POINTS,
        );
        expanded.split_off(core_count)
    } else {
        Vec::new()
    };
    if !gaussian_sparse_support.is_empty() {
        println!(
            "geometry: retained {} additional sparse-track particles for the static Gaussian",
            gaussian_sparse_support.len()
        );
    }
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
        if normal_captures.len() >= 3 {
            let response = train::inverse::capture::photometric_response([
                capture,
                &normal_captures[0].capture,
                &normal_captures[1].capture,
                &normal_captures[2].capture,
            ])
            .unwrap_or_else(|message| {
                eprintln!("cannot build aligned-light response: {message}");
                std::process::exit(1);
            });
            let response_started = std::time::Instant::now();
            let stats = train::inverse::refine::refine(
                &mut surfels,
                &response,
                &refine_views,
                train::inverse::refine::RefineOptions {
                    search_radius_factor: 0.25,
                    min_improvement: 0.1,
                    ..train::inverse::refine::RefineOptions::default()
                },
            );
            println!(
                "geometry: aligned-light responses scored {} and moved {} surfels by {:.3} cells on average ({:.1}% lower cost, {:.1} views) in {:.1} s",
                stats.scored,
                stats.moved,
                stats.mean_absolute_offset / voxel,
                100.0 * stats.mean_relative_improvement,
                stats.mean_views,
                response_started.elapsed().as_secs_f64(),
            );
        }
    }
    (
        surfels,
        sparse_support,
        gaussian_sparse_support,
        DensityNormalSource {
            model,
            surfels: foam_surface_count,
        },
    )
}

struct DensityNormalSource {
    model: vol::PointCloudModel,
    surfels: usize,
}

impl DensityNormalSource {
    fn refine(&self, surface: &mut vol::relight::RelightModel, blend: f32) -> usize {
        train::inverse::surface::refine_normals_from_density(
            &mut surface.surfels[..self.surfels],
            &self.model,
            blend,
        )
    }
}

struct SparseCell {
    position_sum: glam::Vec3,
    count: usize,
    views: collections::HashMap<usize, SparseViewCell>,
}

struct SparseViewCell {
    radiance_sum: [f32; 3],
    count: usize,
}

struct SparseSupport {
    observations: Vec<SparseTrackObservation>,
}

struct SparseTrackObservation {
    view: usize,
    radiance: [f32; 3],
}

fn add_sparse_track_support(
    surfels: &mut Vec<vol::relight::Surfel>,
    reconstruction: &train::colmap::Reconstruction,
    capture: &train::inverse::capture::Capture,
    views: &[usize],
    voxel: f32,
    radius_factor: f32,
    minimum_cell_points: usize,
) -> Vec<SparseSupport> {
    if surfels.is_empty() || !voxel.is_finite() || voxel <= 0.0 {
        return Vec::new();
    }
    let image_ids: collections::HashMap<&str, u32> = reconstruction
        .images
        .iter()
        .map(|image| (image.name.as_str(), image.id))
        .collect();
    let training_views: collections::HashMap<u32, usize> = views
        .iter()
        .filter_map(|&index| {
            image_ids
                .get(capture.views[index].name.as_str())
                .map(|&image_id| (image_id, index))
        })
        .collect();
    let mut cells = collections::HashMap::<[i32; 3], SparseCell>::new();
    for point in &reconstruction.points {
        let training_support = point
            .track_image_ids
            .iter()
            .filter_map(|image| training_views.get(image).copied())
            .count();
        if training_support < 2 || point.error > 1.0 {
            continue;
        }
        let point_views: collections::HashSet<usize> = point
            .track_image_ids
            .iter()
            .filter_map(|image| training_views.get(image).copied())
            .collect();
        let position = glam::Vec3::new(
            point.xyz[0] as f32,
            point.xyz[1] as f32,
            point.xyz[2] as f32,
        );
        let key = position
            .to_array()
            .map(|coordinate| (coordinate / voxel).floor() as i32);
        let cell = cells.entry(key).or_insert(SparseCell {
            position_sum: glam::Vec3::ZERO,
            count: 0,
            views: collections::HashMap::new(),
        });
        cell.position_sum += position;
        cell.count += 1;
        for view in point_views {
            let image = &capture.views[view];
            let Some((pixel, _)) = train::inverse::capture::project(
                &image.camera,
                capture.width,
                capture.height,
                position,
            ) else {
                continue;
            };
            if pixel[0] < 0.0
                || pixel[1] < 0.0
                || pixel[0] >= capture.width as f32
                || pixel[1] >= capture.height as f32
            {
                continue;
            }
            let pixel = pixel[1] as usize * capture.width + pixel[0] as usize;
            if !image.is_foreground(pixel) {
                continue;
            }
            let entry = cell.views.entry(view).or_insert(SparseViewCell {
                radiance_sum: [0.0; 3],
                count: 0,
            });
            for (sum, value) in entry.radiance_sum.iter_mut().zip(image.pixels[pixel]) {
                *sum += value;
            }
            entry.count += 1;
        }
    }
    let mut points: Vec<glam::Vec3> = cells
        .values()
        .filter(|cell| cell.count >= minimum_cell_points)
        .map(|cell| cell.position_sum / cell.count as f32)
        .collect();
    points.sort_by(|left, right| left.to_array().partial_cmp(&right.to_array()).unwrap());
    let cameras: Vec<glam::Vec3> = views
        .iter()
        .map(|&index| glam::Vec3::from(capture.views[index].camera.cam_position))
        .collect();
    let (candidates, _) = train::inverse::surface::surfels_from_points(
        &points,
        &cameras,
        train::inverse::surface::SurfaceOptions::default(),
    );

    let occupied: Vec<[f32; 3]> = surfels.iter().map(|surfel| surfel.center).collect();
    let occupied = kiddo::ImmutableKdTree::new_from_slice(&occupied);
    let minimum_distance_squared = voxel * voxel;
    let mut added = Vec::<glam::Vec3>::new();
    let mut support = Vec::<SparseSupport>::new();
    for mut candidate in candidates {
        let center = glam::Vec3::from(candidate.center);
        if occupied
            .nearest_one::<kiddo::SquaredEuclidean>(&candidate.center)
            .distance
            < minimum_distance_squared
            || added
                .iter()
                .any(|&other| center.distance_squared(other) < minimum_distance_squared)
        {
            continue;
        }
        candidate.radius = voxel * radius_factor.max(0.1);
        candidate.material = surfels.len() as u32;
        surfels.push(candidate);
        added.push(center);
        let key = center
            .to_array()
            .map(|coordinate| (coordinate / voxel).floor() as i32);
        let mut observations: Vec<SparseTrackObservation> = cells[&key]
            .views
            .iter()
            .filter(|entry| entry.1.count >= 2)
            .map(|entry| SparseTrackObservation {
                view: *entry.0,
                radiance: entry.1.radiance_sum.map(|sum| sum / entry.1.count as f32),
            })
            .collect();
        observations.sort_unstable_by_key(|entry| entry.view);
        support.push(SparseSupport { observations });
    }
    support
}

fn supplement_sparse_track_observations(
    observations: &mut train::inverse::decompose::Observations,
    model: &vol::relight::RelightModel,
    capture: &train::inverse::capture::Capture,
    support: &[SparseSupport],
    min_facing: f32,
) -> usize {
    if support.is_empty() {
        return 0;
    }
    let sparse_start = model.surfels.len() - support.len();
    let mut supplements = Vec::with_capacity(support.len());
    let mut supplemented = 0;
    for (offset, sparse) in support.iter().enumerate() {
        let surfel_index = sparse_start + offset;
        let surfel = &model.surfels[surfel_index];
        let center = glam::Vec3::from(surfel.center);
        let normal = glam::Vec3::from(surfel.normal);
        let mut samples = Vec::new();
        if observations.of(surfel_index).is_empty() && sparse.observations.len() >= 2 {
            for sparse_observation in &sparse.observations {
                let view = &capture.views[sparse_observation.view];
                let towards =
                    (glam::Vec3::from(view.camera.cam_position) - center).normalize_or_zero();
                let facing = normal.dot(towards);
                if facing < min_facing {
                    continue;
                }
                samples.push(train::inverse::decompose::Sample {
                    view: sparse_observation.view as u32,
                    radiance: sparse_observation.radiance,
                    towards,
                    facing,
                });
            }
            supplemented += usize::from(!samples.is_empty());
        }
        supplements.push(samples);
    }
    if supplemented == 0 {
        return 0;
    }

    let old_samples = std::mem::take(&mut observations.samples);
    let old_offsets = std::mem::take(&mut observations.offsets);
    let prefix_end = old_offsets[sparse_start] as usize;
    observations.samples =
        Vec::with_capacity(old_samples.len() + supplements.iter().map(Vec::len).sum::<usize>());
    observations
        .samples
        .extend_from_slice(&old_samples[..prefix_end]);
    observations
        .offsets
        .extend_from_slice(&old_offsets[..=sparse_start]);
    for (offset, samples) in supplements.iter().enumerate() {
        let index = sparse_start + offset;
        let begin = old_offsets[index] as usize;
        let end = old_offsets[index + 1] as usize;
        observations
            .samples
            .extend_from_slice(&old_samples[begin..end]);
        observations.samples.extend_from_slice(samples);
        observations.offsets.push(observations.samples.len() as u32);
    }
    supplemented
}

fn mask_depth(
    map: &mut train::inverse::depth::DepthMap,
    view: &train::inverse::capture::View,
) -> usize {
    let Some(ref mask) = view.mask else {
        return map.alpha.len();
    };
    assert_eq!(mask.len(), map.alpha.len());
    let mut foreground = 0;
    for (index, &coverage) in mask.iter().enumerate() {
        if coverage <= 0.5 {
            map.alpha[index] = 0.0;
            map.peak[index] = 0.0;
        } else {
            foreground += 1;
        }
    }
    foreground
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

fn score_gaussian_test(
    model: &vol::PointCloudModel,
    capture: &train::inverse::capture::Capture,
    views: &[usize],
) -> Result<Option<(f32, f32)>, String> {
    if views.is_empty() {
        return Ok(None);
    }
    let scores =
        train::gaussian_splat::evaluate_views(model, capture, views, 64, 1.0e-5, [0.0; 3])?;
    Ok(Some((
        scores.iter().sum::<f32>() / scores.len() as f32,
        scores.iter().copied().fold(f32::INFINITY, f32::min),
    )))
}

fn dump_static_gaussian(
    model: &vol::PointCloudModel,
    capture: &train::inverse::capture::Capture,
    views: &[usize],
    directory: &path::Path,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    let pixels = capture.width * capture.height;
    for &index in views {
        let view = &capture.views[index];
        let origin = glam::Vec3::from(view.camera.cam_position);
        let origins = vec![origin; pixels];
        let directions: Vec<_> = (0..capture.height)
            .flat_map(|y| {
                (0..capture.width).map(move |x| {
                    train::inverse::capture::pixel_direction(
                        &view.camera,
                        capture.width,
                        capture.height,
                        x,
                        y,
                    )
                })
            })
            .collect();
        let rendered =
            train::gaussian_splat::render_rays(model, &origins, &directions, 64, 1.0e-5, [0.0; 3]);
        let mut image = image::RgbImage::new(capture.width as u32, capture.height as u32);
        for (pixel, value) in rendered.iter().enumerate() {
            let color = value
                .truncate()
                .to_array()
                .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            image.put_pixel(
                (pixel % capture.width) as u32,
                (pixel / capture.width) as u32,
                image::Rgb(color),
            );
        }
        let stem = path::Path::new(&view.name).file_stem().map_or_else(
            || view.name.clone(),
            |stem| stem.to_string_lossy().into_owned(),
        );
        image
            .save(directory.join(format!("{stem}-render.png")))
            .map_err(|error| format!("cannot save static render {stem}: {error}"))?;
        train::inverse::score::save_rgb(
            &directory.join(format!("{stem}-photo.png")),
            &view.pixels,
            capture.width,
            capture.height,
        );
    }
    Ok(())
}

struct CaptureBaseline {
    mean: f32,
    worst: f32,
    foreground_mean: Option<f32>,
    foreground_worst: Option<f32>,
}

impl CaptureBaseline {
    fn foreground_text(&self) -> String {
        self.foreground_mean.zip(self.foreground_worst).map_or_else(
            || "—".to_string(),
            |(mean, worst)| format!("{mean:.2}/{worst:.2} dB"),
        )
    }
}

fn capture_baseline_psnr(
    prediction: Option<&train::inverse::capture::Capture>,
    reference: &train::inverse::capture::Capture,
    views: &[usize],
) -> CaptureBaseline {
    let mut scores = Vec::with_capacity(views.len());
    let mut foreground_scores = Vec::with_capacity(views.len());
    for &view in views {
        let reference_pixels = &reference.views[view].pixels;
        let prediction_pixels = prediction.map(|capture| &capture.views[view].pixels);
        let mut squared_error = 0.0f32;
        let mut foreground_error = 0.0f32;
        let mut foreground_weight = 0.0f32;
        for (pixel, truth) in reference_pixels.iter().enumerate() {
            let mask = reference.views[view]
                .mask
                .as_ref()
                .map_or(0.0, |mask| mask[pixel]);
            foreground_weight += mask;
            for channel in 0..3 {
                let predicted = prediction_pixels.map_or(0.0, |pixels| {
                    train::inverse::capture::linear_to_srgb(pixels[pixel][channel])
                });
                let truth = train::inverse::capture::linear_to_srgb(truth[channel]);
                let error = (predicted - truth).powi(2);
                squared_error += error;
                foreground_error += mask * error;
            }
        }
        squared_error /= (reference_pixels.len() * 3) as f32;
        scores.push(if squared_error == 0.0 {
            f32::INFINITY
        } else {
            -10.0 * squared_error.log10()
        });
        if foreground_weight > f32::EPSILON {
            foreground_error /= 3.0 * foreground_weight;
            foreground_scores.push(if foreground_error == 0.0 {
                f32::INFINITY
            } else {
                -10.0 * foreground_error.log10()
            });
        }
    }
    CaptureBaseline {
        mean: scores.iter().sum::<f32>() / scores.len() as f32,
        worst: scores.iter().copied().fold(f32::INFINITY, f32::min),
        foreground_mean: (!foreground_scores.is_empty())
            .then(|| foreground_scores.iter().sum::<f32>() / foreground_scores.len() as f32),
        foreground_worst: foreground_scores.into_iter().reduce(f32::min),
    }
}

fn print_reconstruction_summary(name: &str, summary: train::inverse::score::Summary) {
    let mask_recall = summary
        .mask_recall
        .map_or_else(|| "—".to_string(), |value| format!("{:.1}%", 100.0 * value));
    let mask_precision = summary
        .mask_precision
        .map_or_else(|| "—".to_string(), |value| format!("{:.1}%", 100.0 * value));
    let foreground = summary
        .foreground_srgb_psnr
        .map_or_else(|| "—".to_string(), |value| format!("{value:.2}"));
    let foreground_worst = summary
        .worst_foreground_srgb_psnr
        .map_or_else(|| "—".to_string(), |value| format!("{value:.2}"));
    println!(
        "{name:<12}{:>8}{:>12.2}{:>12.2}{:>12.2}{foreground:>12}{foreground_worst:>12}{:>11.1}%{mask_recall:>12}{mask_precision:>12}{:>12.2}",
        summary.views,
        summary.srgb_psnr,
        summary.worst_srgb_psnr,
        summary.linear_psnr,
        100.0 * summary.coverage,
        summary.covered_srgb_psnr
    );
}

fn guard_pbr_support(
    gaussian: &mut vol::PointCloudModel,
    established: &vol::PointCloudModel,
    stage: &str,
) -> bool {
    let guard =
        train::gaussian_splat::guard_pbr_support(gaussian, established).unwrap_or_else(|error| {
            eprintln!("cannot guard PBR Gaussian support after {stage}: {error}");
            std::process::exit(1);
        });
    if guard.restored {
        println!(
            "PBR support: rejected {stage} collapse ({} of {} particles persist); restored established opacity and scale",
            guard.retained, guard.particles,
        );
    }
    guard.restored
}

fn match_pbr_surface_extent(
    gaussian: &mut vol::PointCloudModel,
    surface: &vol::relight::RelightModel,
) {
    train::gaussian_splat::match_pbr_surface_extent(gaussian, surface).unwrap_or_else(|error| {
        eprintln!("cannot restore PBR Gaussian surface extent: {error}");
        std::process::exit(1);
    });
    println!("restored PBR Gaussian extent from the established surface");
}

fn update_surface_radii(surface: &mut vol::relight::RelightModel, gaussian: &vol::PointCloudModel) {
    train::gaussian_splat::update_surface_radii(surface, gaussian).unwrap_or_else(|error| {
        eprintln!("cannot update PBR support from direct Gaussian field: {error}");
        std::process::exit(1);
    });
    println!("updated PBR radii from learned Gaussian support");
}

fn validate_lit_captures(args: &Args) -> Result<(), String> {
    if args.normal_images.len() != args.normal_environment.len() {
        return Err(format!(
            "--normal-images was supplied {} times but --normal-environment was supplied {} times",
            args.normal_images.len(),
            args.normal_environment.len(),
        ));
    }
    if !args.normal_images.is_empty() && args.environment.is_none() {
        return Err(
            "photometric normal refinement also needs --environment for the primary capture"
                .to_string(),
        );
    }
    if args.render_refine_normals && args.environment.is_none() {
        return Err("rendered normal refinement needs --environment".to_string());
    }
    if args.held_out_images.len() != args.held_out_environment.len() {
        return Err(format!(
            "--held-out-images was supplied {} times but --held-out-environment was supplied {} times",
            args.held_out_images.len(),
            args.held_out_environment.len(),
        ));
    }
    if !args.held_out_images.is_empty() && args.test_every == 0 && args.test_list.is_none() {
        return Err(
            "held-out-light scoring needs --test-every or --test-list to reserve cameras"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_geometry_source(args: &Args) -> Result<(), String> {
    if args.foam.is_some() && args.dense_cloud.is_some() {
        return Err("--foam and --dense-cloud are alternative geometry sources".to_string());
    }
    if args.dense_cloud.is_some()
        && !(train::inverse::surface::SurfaceOptions::default().neighbours + 1..=u32::MAX as usize)
            .contains(&args.dense_max_points)
    {
        return Err(format!(
            "--dense-max-points must be between {} and {}",
            train::inverse::surface::SurfaceOptions::default().neighbours + 1,
            u32::MAX,
        ));
    }
    Ok(())
}

fn uses_sparse_static_gaussian_surface(args: &Args) -> bool {
    args.foam.is_none() && args.gaussian_output.is_some()
}

struct LitCapture {
    capture: train::inverse::capture::Capture,
    environment: vol::relight::Environment,
}

fn load_lit_capture(
    primary: &train::inverse::capture::Capture,
    sparse: &path::Path,
    height: usize,
    args: &Args,
    images: &str,
    environment: &str,
) -> Result<LitCapture, String> {
    let (capture, _) = train::inverse::capture::Capture::from_colmap_with_masks(
        sparse,
        path::Path::new(images),
        args.masks.as_deref().map(path::Path::new),
        args.width,
        height,
        args.stride,
    )?;
    if capture.views.len() != primary.views.len()
        || capture
            .views
            .iter()
            .zip(&primary.views)
            .any(|(left, right)| left.name != right.name)
    {
        return Err(format!(
            "{images} does not contain the same selected photographs as the primary capture"
        ));
    }
    let environment = vol::io::try_load_environment(path::Path::new(environment))
        .map_err(|error| format!("cannot read {environment}: {error}"))?;
    Ok(LitCapture {
        capture,
        environment,
    })
}

fn load_normal_captures(
    primary: &train::inverse::capture::Capture,
    sparse: &path::Path,
    height: usize,
    args: &Args,
) -> Result<Vec<LitCapture>, String> {
    args.normal_images
        .iter()
        .zip(&args.normal_environment)
        .map(|(images, environment)| {
            load_lit_capture(primary, sparse, height, args, images, environment)
        })
        .collect()
}

fn load_held_out_captures(
    primary: &train::inverse::capture::Capture,
    sparse: &path::Path,
    height: usize,
    args: &Args,
) -> Result<Vec<LitCapture>, String> {
    args.held_out_images
        .iter()
        .zip(&args.held_out_environment)
        .map(|(images, environment)| {
            load_lit_capture(primary, sparse, height, args, images, environment)
        })
        .collect()
}

fn refine_normals_from_captures(
    model: &mut vol::relight::RelightModel,
    primary: &train::inverse::capture::Capture,
    train_views: &[usize],
    primary_light: &vol::relight::Environment,
    secondary: &[LitCapture],
) -> train::inverse::decompose::NormalRefinement {
    let mut observations = vec![train::inverse::decompose::observe(
        model,
        primary,
        train_views,
        -1.0,
    )];
    let mut irradiance = vec![train::relight::Irradiance::project(
        &primary_light.texels,
        primary_light.width,
        primary_light.height,
    )];
    for entry in secondary {
        observations.push(train::inverse::decompose::observe(
            model,
            &entry.capture,
            train_views,
            -1.0,
        ));
        irradiance.push(train::relight::Irradiance::project(
            &entry.environment.texels,
            entry.environment.width,
            entry.environment.height,
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
    train::inverse::decompose::refine_normals_known_lights_per_view(model, &lights, 512)
}

fn refine_materials_from_captures(
    model: &mut vol::relight::RelightModel,
    primary_observations: &train::inverse::decompose::Observations,
    train_views: &[usize],
    primary_light: &vol::relight::Environment,
    secondary: &[LitCapture],
    ceiling: f32,
) -> train::inverse::decompose::MaterialRefinement {
    let observations: Vec<_> = secondary
        .iter()
        .map(|entry| {
            train::inverse::decompose::observe(
                model,
                &entry.capture,
                train_views,
                train::inverse::decompose::FitOptions::default().min_facing,
            )
        })
        .collect();
    let mut lights = Vec::with_capacity(secondary.len() + 1);
    lights.push(train::inverse::decompose::KnownLightObservations {
        irradiance: train::relight::Irradiance::project(
            &primary_light.texels,
            primary_light.width,
            primary_light.height,
        ),
        observations: primary_observations,
    });
    lights.extend(
        secondary
            .iter()
            .zip(&observations)
            .map(
                |(entry, observations)| train::inverse::decompose::KnownLightObservations {
                    irradiance: train::relight::Irradiance::project(
                        &entry.environment.texels,
                        entry.environment.width,
                        entry.environment.height,
                    ),
                    observations,
                },
            ),
    );
    train::inverse::decompose::refine_materials_known_lights(model, &lights, ceiling)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_source_refines_only_foam_surface_prefix() {
        let points: Vec<_> = (0..2)
            .flat_map(|z| {
                (0..2).flat_map(move |y| {
                    (0..2).map(move |x| {
                        glam::Vec3::new(x as f32, y as f32, z as f32).extend(y as f32)
                    })
                })
            })
            .collect();
        let source = DensityNormalSource {
            model: vol::PointCloudModel {
                sh_coefficients: vec![0.0; points.len() * 3],
                sh_degree: 0,
                points,
                transforms: None,
                adjacency: None,
                radii: None,
                surface_normals: None,
                surface_offsets: None,
                surface_detail: None,
                surface_color_coefficients: None,
                spherical_voronoi: None,
            },
            surfels: 1,
        };
        let normal = glam::Vec3::new(0.6, 0.8, 0.0).normalize().to_array();
        let surfel = vol::relight::Surfel {
            center: [0.5; 3],
            radius: 0.2,
            normal,
            material: 0,
        };
        let mut surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![surfel; 2],
            materials: vec![vol::relight::Material::default()],
        };
        assert_eq!(source.refine(&mut surface, 0.1), 1);
        assert_ne!(surface.surfels[0].normal, normal);
        assert_eq!(surface.surfels[1].normal, normal);
    }

    #[test]
    fn static_gaussian_surface_discards_source_material_references() {
        let mut surface = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![vol::relight::Surfel {
                center: [0.0; 3],
                radius: 1.0,
                normal: [0.0, 0.0, 1.0],
                material: 1,
            }],
            materials: vec![
                vol::relight::Material::default(),
                vol::relight::Material::default(),
            ],
        };
        surface = prepare_static_surface(surface, None);
        surface.validate().unwrap();
        train::gaussian_splat::from_surface(&surface).unwrap();
        assert_eq!(surface.surfels[0].material, 0);
        assert_eq!(surface.materials.len(), 1);
    }

    #[test]
    fn capture_baseline_compares_in_display_space() {
        let capture = train::inverse::capture::Capture {
            width: 1,
            height: 1,
            views: vec![train::inverse::capture::View {
                name: "view".to_string(),
                camera: vol::CameraParams::default(),
                pixels: vec![[0.25, 0.5, 0.75]],
                mask: None,
            }],
        };
        let copy = capture_baseline_psnr(Some(&capture), &capture, &[0]);
        let black = capture_baseline_psnr(None, &capture, &[0]);
        assert!(copy.mean.is_infinite());
        assert_eq!(copy.foreground_mean, None);
        assert!(black.mean.is_finite() && black.mean > 0.0);
    }

    #[test]
    fn static_continuation_does_not_mutate_the_pbr_surface() {
        let pbr = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![vol::relight::Surfel {
                center: [1.0, 2.0, 3.0],
                radius: 0.5,
                normal: [0.0, 1.0, 0.0],
                material: 0,
            }],
            materials: vec![vol::relight::Material::default()],
        };
        let mut seed = pbr.clone();
        seed.surfels[0].center = [4.0, 5.0, 6.0];
        let mut static_surface = Some(seed);
        let continued = static_continuation_surface(&mut static_surface, &pbr);
        continued.surfels[0].radius = 2.0;
        continued.surfels[0].normal = [1.0, 0.0, 0.0];

        assert_eq!(pbr.surfels[0].radius, 0.5);
        assert_eq!(pbr.surfels[0].normal, [0.0, 1.0, 0.0]);
        let continued = &static_surface.unwrap().surfels[0];
        assert_eq!(continued.center, [4.0, 5.0, 6.0]);
        assert_eq!(continued.radius, 2.0);
    }

    fn sparse_support_fixture(track_image_ids: Vec<u32>) -> train::colmap::Reconstruction {
        let mut points = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                for sample in 0..5 {
                    points.push(train::colmap::ColmapPoint3D {
                        id: points.len() as u64,
                        xyz: [x as f64 + 0.01 * sample as f64, y as f64, 0.0],
                        rgb: [128; 3],
                        error: 0.25,
                        track_len: track_image_ids.len() as u64,
                        track_image_ids: track_image_ids.clone(),
                    });
                }
            }
        }
        let images = [(10, "train-a"), (11, "train-b"), (12, "held-a")]
            .into_iter()
            .map(|(id, name)| train::colmap::ColmapImage {
                id,
                camera_id: 1,
                name: name.to_string(),
                quat_wxyz: [1.0, 0.0, 0.0, 0.0],
                translation: [0.0; 3],
                num_points2d: 0,
            })
            .collect();
        train::colmap::Reconstruction {
            cameras: collections::HashMap::new(),
            images,
            points,
        }
    }

    fn sparse_support_capture() -> train::inverse::capture::Capture {
        let camera = vol::CameraParams {
            cam_position: [1.5, 1.5, -5.0],
            depth: 100.0,
            cam_orientation: glam::Quat::IDENTITY.to_array(),
            fov: [1.0; 2],
            principal: [0.0; 2],
        };
        train::inverse::capture::Capture {
            width: 1,
            height: 1,
            views: ["train-a", "train-b", "held-a"]
                .into_iter()
                .map(|name| train::inverse::capture::View {
                    name: name.to_string(),
                    camera,
                    pixels: vec![[0.0; 3]],
                    mask: None,
                })
                .collect(),
        }
    }

    #[test]
    fn sparse_support_uses_only_tracks_from_selected_training_views() {
        let capture = sparse_support_capture();
        let mut supported = vec![vol::relight::Surfel {
            center: [100.0, 100.0, 100.0],
            radius: 1.0,
            normal: [0.0, 0.0, 1.0],
            material: 0,
        }];
        let added = add_sparse_track_support(
            &mut supported,
            &sparse_support_fixture(vec![10, 11]),
            &capture,
            &[0, 1],
            1.0,
            1.7,
            PBR_SPARSE_CELL_POINTS,
        );
        assert!(!added.is_empty());
        assert!(supported[1..].iter().all(|surfel| surfel.radius == 1.7));
        assert!(added.iter().all(|support| {
            support.observations.len() == 2
                && support
                    .observations
                    .iter()
                    .all(|observation| observation.view < 2)
        }));

        let mut held_only = supported[..1].to_vec();
        let added = add_sparse_track_support(
            &mut held_only,
            &sparse_support_fixture(vec![12, 12]),
            &capture,
            &[0, 1],
            1.0,
            1.7,
            PBR_SPARSE_CELL_POINTS,
        );
        assert!(added.is_empty());
    }

    #[test]
    fn static_sparse_support_requires_a_training_track() {
        let training_images = collections::HashSet::from([10, 11]);
        assert!(has_training_track(&[9, 10, 12], &training_images));
        assert!(!has_training_track(&[9, 12], &training_images));
        assert!(!has_training_track(&[], &training_images));
    }

    #[test]
    fn sparse_support_respects_the_output_confidence_floor() {
        let capture = sparse_support_capture();
        let mut reconstruction = sparse_support_fixture(vec![10, 11]);
        reconstruction.points.retain(|point| point.id % 5 < 3);
        let seed = vol::relight::Surfel {
            center: [100.0; 3],
            radius: 1.0,
            normal: [0.0, 0.0, 1.0],
            material: 0,
        };
        let mut pbr = vec![seed];
        let conservative = add_sparse_track_support(
            &mut pbr,
            &reconstruction,
            &capture,
            &[0, 1],
            1.0,
            1.7,
            PBR_SPARSE_CELL_POINTS,
        );
        assert!(conservative.is_empty());

        let mut static_gaussian = vec![seed];
        let expanded = add_sparse_track_support(
            &mut static_gaussian,
            &reconstruction,
            &capture,
            &[0, 1],
            1.0,
            1.7,
            STATIC_GAUSSIAN_SPARSE_CELL_POINTS,
        );
        assert!(!expanded.is_empty());
    }

    #[test]
    fn sparse_track_observations_supplement_only_unseen_particles() {
        let camera = vol::CameraParams {
            cam_position: [0.0, 0.0, -2.0],
            depth: 100.0,
            cam_orientation: glam::Quat::IDENTITY.to_array(),
            fov: [1.0; 2],
            principal: [0.0; 2],
        };
        let capture = train::inverse::capture::Capture {
            width: 1,
            height: 1,
            views: (0..2)
                .map(|index| train::inverse::capture::View {
                    name: format!("train-{index}"),
                    camera,
                    pixels: vec![[0.25, 0.5, 0.75]],
                    mask: None,
                })
                .collect(),
        };
        let model = vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Gaussian,
            surfels: vec![
                vol::relight::Surfel {
                    center: [0.0; 3],
                    radius: 1.0,
                    normal: [0.0, 0.0, -1.0],
                    material: 0,
                },
                vol::relight::Surfel {
                    center: [0.0; 3],
                    radius: 1.0,
                    normal: [0.0, 0.0, -1.0],
                    material: 1,
                },
            ],
            materials: vec![vol::relight::Material::default(); 2],
        };
        let original = train::inverse::decompose::Sample {
            view: 0,
            radiance: [1.0, 0.0, 0.0],
            towards: -glam::Vec3::Z,
            facing: 1.0,
        };
        let mut observations = train::inverse::decompose::Observations {
            samples: vec![original],
            offsets: vec![0, 1, 1],
        };
        let support = vec![SparseSupport {
            observations: vec![
                SparseTrackObservation {
                    view: 0,
                    radiance: [0.25, 0.5, 0.75],
                },
                SparseTrackObservation {
                    view: 1,
                    radiance: [0.5, 0.25, 0.125],
                },
            ],
        }];

        let supplemented = supplement_sparse_track_observations(
            &mut observations,
            &model,
            &capture,
            &support,
            0.15,
        );

        assert_eq!(supplemented, 1);
        assert_eq!(observations.of(0).len(), 1);
        assert_eq!(observations.of(0)[0].radiance, original.radiance);
        assert_eq!(observations.of(1).len(), 2);
        assert_eq!(observations.of(1)[0].radiance, [0.25, 0.5, 0.75]);
        assert_eq!(observations.of(1)[1].radiance, [0.5, 0.25, 0.125]);
    }

    #[test]
    fn known_environment_is_an_optional_cli_input() {
        let defaults = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &["--sparse", "sparse", "--images", "images"],
        )
        .unwrap();
        assert!(defaults.environment.is_none());
        assert!(defaults.masks.is_none());
        assert!(defaults.dense_cloud.is_none());
        assert_eq!(defaults.dense_max_points, 50_000);
        assert_eq!(defaults.surface_powerfoam_steps_per_view, 0);
        assert!(defaults.surface_powerfoam_output.is_none());
        assert!(defaults.gaussian_output.is_none());
        assert!(defaults.pbr_gaussian_output.is_none());
        assert!(defaults.held_out_images.is_empty());
        assert!(defaults.held_out_environment.is_empty());
        assert_eq!(defaults.gaussian_steps, 1_500);
        assert_eq!(defaults.diffuse_samples, 0);

        let known = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--environment",
                "capture.f32",
            ],
        )
        .unwrap();
        assert_eq!(known.environment.as_deref(), Some("capture.f32"));
    }

    #[test]
    fn dense_cloud_is_a_bounded_alternative_geometry_source() {
        let parse = |extra: &[&str]| {
            let mut arguments = vec!["--sparse", "sparse", "--images", "images"];
            arguments.extend_from_slice(extra);
            <Args as argh::FromArgs>::from_args(&["reconstruct"], &arguments).unwrap()
        };

        let dense = parse(&[
            "--dense-cloud",
            "dense/fused.ply",
            "--dense-max-points",
            "20000",
        ]);
        assert_eq!(dense.dense_cloud.as_deref(), Some("dense/fused.ply"));
        assert_eq!(dense.dense_max_points, 20_000);
        validate_geometry_source(&dense).unwrap();
        assert!(!uses_sparse_static_gaussian_surface(&dense));

        let dense_outputs = parse(&[
            "--dense-cloud",
            "dense/fused.ply",
            "--gaussian-output",
            "static.ply",
            "--pbr-gaussian-output",
            "pbr.ply",
        ]);
        assert!(uses_sparse_static_gaussian_surface(&dense_outputs));

        let ambiguous = parse(&["--dense-cloud", "dense/fused.ply", "--foam", "trained.ply"]);
        assert!(validate_geometry_source(&ambiguous)
            .unwrap_err()
            .contains("alternative geometry sources"));

        let too_small = parse(&[
            "--dense-cloud",
            "dense/fused.ply",
            "--dense-max-points",
            "12",
        ]);
        assert!(validate_geometry_source(&too_small).is_err());
    }

    #[test]
    fn held_lights_are_repeatable_and_paired() {
        let args = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--test-list",
                "test.txt",
                "--held-out-images",
                "held-a",
                "--held-out-environment",
                "held-a.f32",
                "--held-out-images",
                "held-b",
                "--held-out-environment",
                "held-b.f32",
            ],
        )
        .unwrap();
        assert_eq!(args.held_out_images, ["held-a", "held-b"]);
        assert_eq!(args.held_out_environment, ["held-a.f32", "held-b.f32"]);
        validate_lit_captures(&args).unwrap();

        let mismatched = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--test-list",
                "test.txt",
                "--held-out-images",
                "held-a",
            ],
        )
        .unwrap();
        assert!(validate_lit_captures(&mismatched)
            .unwrap_err()
            .contains("supplied 1 times"));
    }

    #[test]
    fn direct_gaussian_output_is_opt_in() {
        let args = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--gaussian-output",
                "light-field.ply",
                "--gaussian-steps",
                "800",
            ],
        )
        .unwrap();
        assert_eq!(args.gaussian_output.as_deref(), Some("light-field.ply"));
        assert!(args.pbr_gaussian_output.is_none());
        assert_eq!(args.gaussian_steps, 800);
    }

    #[test]
    fn relightable_gaussian_output_is_opt_in() {
        let args = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--pbr-gaussian-output",
                "relightable.ply",
            ],
        )
        .unwrap();
        assert!(args.gaussian_output.is_none());
        assert_eq!(args.pbr_gaussian_output.as_deref(), Some("relightable.ply"));
    }

    #[test]
    fn foreground_masks_are_an_optional_cli_input() {
        let masked = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse", "sparse", "--images", "images", "--masks", "masks",
            ],
        )
        .unwrap();
        assert_eq!(masked.masks.as_deref(), Some("masks"));
        assert_eq!(masked.surface_powerfoam_steps_per_view, 0);

        let continued = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--masks",
                "masks",
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
    fn foreground_masks_reject_foam_depth_before_fusion() {
        let mut map = train::inverse::depth::DepthMap {
            width: 2,
            height: 1,
            distance: vec![2.0, 3.0],
            alpha: vec![0.9, 0.8],
            peak: vec![0.4, 0.3],
        };
        let view = train::inverse::capture::View {
            name: "masked".to_string(),
            camera: vol::CameraParams::default(),
            pixels: vec![[0.0; 3]; 2],
            mask: Some(vec![0.5, 1.0]),
        };
        assert_eq!(mask_depth(&mut map, &view), 1);
        assert_eq!(map.distance, [2.0, 3.0]);
        assert_eq!(map.alpha, [0.0, 0.8]);
        assert_eq!(map.peak, [0.0, 0.3]);
    }

    #[test]
    fn production_defaults_decline_unidentified_specular_lobes() {
        let defaults = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &["--sparse", "sparse", "--images", "images"],
        )
        .unwrap();
        assert_eq!(defaults.specular_rounds, 0);

        let experimental = <Args as argh::FromArgs>::from_args(
            &["reconstruct"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--specular-rounds",
                "3",
            ],
        )
        .unwrap();
        assert_eq!(experimental.specular_rounds, 3);
    }

    #[test]
    fn normal_captures_require_paired_images_and_lights() {
        let parse = |extra: &[&str]| {
            let mut arguments = vec!["--sparse", "sparse", "--images", "images"];
            arguments.extend_from_slice(extra);
            <Args as argh::FromArgs>::from_args(&["reconstruct"], &arguments).unwrap()
        };

        let missing_primary = parse(&[
            "--normal-images",
            "images-west",
            "--normal-environment",
            "west.f32",
        ]);
        assert!(validate_lit_captures(&missing_primary).is_err());

        let mismatched = parse(&[
            "--environment",
            "east.f32",
            "--normal-images",
            "images-west",
        ]);
        assert!(validate_lit_captures(&mismatched).is_err());

        let rendered_without_secondary =
            parse(&["--environment", "east.f32", "--render-refine-normals"]);
        validate_lit_captures(&rendered_without_secondary).unwrap();

        let rendered_without_environment = parse(&["--render-refine-normals"]);
        assert!(validate_lit_captures(&rendered_without_environment).is_err());

        let paired = parse(&[
            "--environment",
            "east.f32",
            "--normal-images",
            "images-west",
            "--normal-environment",
            "west.f32",
            "--normal-images",
            "images-sky",
            "--normal-environment",
            "sky.f32",
            "--render-refine-normals",
        ]);
        validate_lit_captures(&paired).unwrap();
        assert_eq!(paired.normal_images.len(), 2);
    }

    #[test]
    fn held_out_light_requires_paired_inputs_and_a_test_split() {
        let parse = |extra: &[&str]| {
            let mut arguments = vec!["--sparse", "sparse", "--images", "images"];
            arguments.extend_from_slice(extra);
            <Args as argh::FromArgs>::from_args(&["reconstruct"], &arguments).unwrap()
        };

        let images_only = parse(&["--held-out-images", "images-west"]);
        assert!(validate_lit_captures(&images_only).is_err());

        let environment_only = parse(&["--held-out-environment", "west.f32"]);
        assert!(validate_lit_captures(&environment_only).is_err());

        let no_test = parse(&[
            "--held-out-images",
            "images-west",
            "--held-out-environment",
            "west.f32",
            "--test-every",
            "0",
        ]);
        assert!(validate_lit_captures(&no_test).is_err());

        let paired = parse(&[
            "--held-out-images",
            "images-west",
            "--held-out-environment",
            "west.f32",
        ]);
        validate_lit_captures(&paired).unwrap();
    }
}
