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
use std::{fs, path};

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

    /// optional output prefix for a strip of novel poses (writes `<prefix>_NN.png`)
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

    /// skip post-training PSNR evaluation
    #[argh(switch)]
    skip_eval: bool,

    /// training epochs (default 100)
    #[argh(option, default = "100")]
    epochs: usize,

    /// max traversal steps per ray (default 128). With 100K cells in a
    /// bonsai-scale scene, an average ray crosses ~50-80 cell boundaries
    /// at 64² resolution; higher resolution / more cells need more
    /// steps. Path truncation throws away the far end of the integral.
    #[argh(option, default = "128")]
    max_steps: usize,

    /// minimum PowerFoam sphere candidates per ray (default 0 = automatic:
    /// max of four times --max-steps and 1024). Increase independently when
    /// many supports overlap before radical-plane clipping.
    #[argh(option, default = "0")]
    powerfoam_candidate_capacity: u32,

    /// adam learning rate (default 0.1)
    #[argh(option, default = "0.1")]
    learning_rate: f32,

    /// cap on initial foam points (default 2000; 0 = policy default/all).
    /// Exact Delaunay construction scales poorly without Qhull.
    #[argh(option, default = "2000")]
    max_points: usize,

    /// initial site policy: "top-track" or "radfoam-v1" (default top-track).
    /// The reference policy samples sparse points with replacement, adds a
    /// broad background cloud, and starts appearance at gray.
    #[argh(option, default = "String::from(\"top-track\")")]
    initialization: String,

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

    /// initialize radii with PowerFoam's eight-sample mean and projected
    /// camera cap, then build Cech adjacency
    #[argh(switch)]
    powerfoam_reference_radii: bool,

    /// split every weighted PowerFoam cell at its site with a learned,
    /// camera-facing surface plane. Initial normals are estimated directly
    /// from point-cloud neighbourhoods.
    #[argh(switch)]
    oriented_powerfoam: bool,

    /// pixels per Adam step (default 1024). Random pixel sampling keeps the
    /// graph small regardless of image resolution. Set 0 to use every pixel.
    #[argh(option, default = "1024")]
    pixel_batch: usize,

    /// camera views per random-pixel Adam batch (default 0 = automatic: 16 for
    /// random pixels, 1 for patch and full-image batches). Rays are split evenly
    /// across a deterministic stratified view sample.
    #[argh(option, default = "0")]
    views_per_batch: usize,

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

    /// supervised RGB loss: "l1" or "smooth-l1" (default l1). The official
    /// RadFoam v1 trainer uses Smooth-L1 with beta 1.
    #[argh(option, default = "String::from(\"l1\")")]
    color_loss: String,

    /// adaptive densification: cells between splits (default 0 = off).
    /// Recommended 500–1000. RadFoam samples parents by accumulated position
    /// gradient times cell radius; PowerFoam uses per-site photometric-error
    /// EMA. Each parent gets an inherited sibling and adjacency is rebuilt
    /// with the configured exact backend. Requires `--pixel-batch`.
    #[argh(option, default = "0")]
    densify_every: usize,

    /// densification cadence: "fixed" or "radfoam-v1" (default fixed).
    /// The reference mode grows first at --densify-warmup, then derives each
    /// interval from the post-growth cell count; --densify-every only enables
    /// densification and is otherwise ignored.
    #[argh(option, default = "String::from(\"fixed\")")]
    densify_schedule: String,

    /// per-round growth: each densify round adds `fraction × current`
    /// cells, with parents drawn by weighted multinomial on the
    /// backend-specific statistic (RadFoam uses 0.15). Ignored when
    /// `--densify-every 0`.
    #[argh(option, default = "0.15")]
    densify_fraction: f32,

    /// final point budget (RadFoam bonsai ≈ 2,097,152). Fixed cadence stops
    /// at the budget; radfoam-v1 stops scheduling at 90% of it. Default
    /// 2_000_000.
    #[argh(option, default = "2000000")]
    densify_target: usize,

    /// fixed cadence: stop densifying after this step. Radfoam-v1: end of the
    /// linear-growth horizon used to derive intervals, not a hard stop.
    /// Default 0 maps to an unbounded horizon and is invalid for radfoam-v1.
    #[argh(option, default = "0")]
    densify_until: usize,

    /// prune small cells with no measured ray contribution, while protecting
    /// contributing cells and their neighbours. Off by default.
    #[argh(switch)]
    prune: bool,

    /// per-view ray-weight threshold that protects a cell and its neighbours
    /// from pruning (default 0.01). Requires `--prune`.
    #[argh(option, default = "0.01")]
    prune_contribution: f32,

    /// contribution threshold below which a cell's density parameter is
    /// suppressed before splitting (default 0.001). Requires `--prune`.
    #[argh(option, default = "0.001")]
    suppress_contribution: f32,

    /// also require farthest-neighbour radius below this (default 0.1) to
    /// prune. Uses the explicit support radius for weighted clouds and the
    /// farthest-neighbour radius otherwise, preserving large background cells.
    #[argh(option, default = "0.1")]
    prune_radius: f32,

    /// cap training views in each prune/densify contribution scan (default 0
    /// = exhaustive). Positive values select a deterministic rotating,
    /// stratified subset and are experimental until compared with exhaustive
    /// decisions on the same checkpoint.
    #[argh(option, default = "0")]
    contribution_views: usize,

    /// legacy no-op sibling-jitter flag, retained for CLI compatibility.
    /// Placement now follows fixed method-specific RadFoam/PowerFoam rules.
    #[argh(option, default = "0.5")]
    densify_jitter: f32,

    /// steps to wait before the first densify cycle (default 500). Lets
    /// the per-cell gradient signal settle from random init before
    /// committing to splits.
    #[argh(option, default = "500")]
    densify_warmup: usize,

    /// learning-rate schedule: "constant", "cosine", or "radfoam-v1"
    /// (default cosine). The reference schedule uses its official absolute
    /// per-parameter rates and requires geometry rebuilds.
    #[argh(option, default = "String::from(\"cosine\")")]
    lr_schedule: String,

    /// cosine-decay floor as a fraction of `--learning-rate` (default 0.01).
    /// At step `total_steps` the effective LR equals `learning_rate * lr_min_ratio`.
    #[argh(option, default = "0.01")]
    lr_min_ratio: f32,

    /// parameter-group multipliers for constant/cosine schedules: "legacy"
    /// or "radfoam-v1-relative" (default legacy). The relative v1 policy
    /// keeps the selected global curve but uses the reference initial
    /// position/DC/higher-SH ratios and requires geometry rebuilds.
    #[argh(option, default = "String::from(\"legacy\")")]
    lr_groups: String,

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

    /// weight on RadFoam's random transmittance-quantile depth separation
    /// loss (default 0 = off; reference value 0.0001).
    #[argh(option, default = "0.0")]
    quantile_weight: f32,

    /// initial weight on PowerFoam's radial-overlap interpenetration loss
    /// (default 0 = off; reference value 0.0001). Decays 1000x by the final
    /// step and requires trainable weighted geometry.
    #[argh(option, default = "0.0")]
    interpenetration_weight: f32,

    /// directed Cech edges sampled per interpenetration-loss update. The sampled
    /// sum is scaled to estimate the complete graph (default 4096).
    #[argh(option, default = "4096")]
    interpenetration_samples: usize,

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

    /// radius learning rate for PowerFoam as a fraction of the main learning rate
    /// (default 0 = fixed radii). Requires a weighted initializer or
    /// --init-ply and periodic geometry rebuilds.
    #[argh(option, default = "0.0")]
    radius_lr_ratio: f32,

    /// oriented PowerFoam normal learning rate as a fraction of the main
    /// learning rate (default 0 = fixed normals). Under the radfoam-v1
    /// schedule, 1 selects PowerFoam's absolute 0.1 to 0.01 schedule.
    #[argh(option, default = "0.0")]
    surface_normal_lr_ratio: f32,

    /// signed oriented-surface offset learning rate as a fraction of the main
    /// learning rate (default 0 = no offset table). Under the radfoam-v1
    /// schedule, 1 scales PowerFoam's absolute 0.005 to 0.0005 height rate.
    #[argh(option, default = "0.0")]
    surface_offset_lr_ratio: f32,

    /// spatial oriented-surface color learning rate as a fraction of the main
    /// learning rate (default 0 = no spatial residual table). Under the
    /// radfoam-v1 schedule, 1 selects the absolute 0.005 to 0.0005 rate.
    #[argh(option, default = "0.0")]
    surface_color_lr_ratio: f32,

    /// radius-normalized spatial-detail site learning rate as a fraction of
    /// the main rate (default 0 = no detail table). Under the radfoam-v1
    /// schedule, 1 selects the absolute 0.01 to 0.001 rate.
    #[argh(option, default = "0.0")]
    surface_detail_offset_lr_ratio: f32,

    /// radius-normalized spatial-detail height learning rate as a fraction
    /// of the main rate (default 0). Under the radfoam-v1 schedule, 1 selects
    /// the absolute 0.005 to 0.0005 rate.
    #[argh(option, default = "0.0")]
    surface_detail_height_lr_ratio: f32,

    /// spatial-detail RGB residual learning rate as a fraction of the main
    /// rate (default 0). Under the radfoam-v1 schedule, 1 selects the absolute
    /// 0.005 to 0.0005 rate.
    #[argh(option, default = "0.0")]
    surface_detail_color_lr_ratio: f32,

    /// spatial-detail density-logit learning rate as a fraction of the main
    /// rate (default 0 = no spatial density table). Equal logits preserve the
    /// base cell density.
    #[argh(option, default = "0.0")]
    surface_detail_density_lr_ratio: f32,

    /// per-spatial-site directional raw-axis learning rate as a fraction of
    /// the main rate (default 0 = no full directional detail table).
    #[argh(option, default = "0.0")]
    surface_detail_directional_axis_lr_ratio: f32,

    /// per-spatial-site directional RGB learning rate as a fraction of the
    /// main rate (default 0 = no full directional detail table).
    #[argh(option, default = "0.0")]
    surface_detail_directional_color_lr_ratio: f32,

    /// spherical Voronoi raw-axis learning rate as a fraction of the main
    /// rate (default 0 = no directional table). Under the radfoam-v1
    /// schedule, 1 selects the absolute 0.05 to 0.005 axis rate.
    #[argh(option, default = "0.0")]
    spherical_voronoi_axis_lr_ratio: f32,

    /// spherical Voronoi RGB-site learning rate as a fraction of the main
    /// rate (default 0 = no directional table). Under the radfoam-v1
    /// schedule, 1 selects the absolute 0.005 to 0.0005 color rate.
    #[argh(option, default = "0.0")]
    spherical_voronoi_color_lr_ratio: f32,

    /// initial weight on PowerFoam's view-facing normal loss (default 0 =
    /// off; reference value 0.1, decaying to 0.01).
    #[argh(option, default = "0.0")]
    surface_normal_weight: f32,

    /// steps between adjacency/path rebuilds during geometry optimisation
    /// (default 100). Ignored when all geometry rates are 0.
    #[argh(option, default = "100")]
    geometry_rebuild_every: usize,

    /// geometry/path rebuild cadence: "fixed" or "radfoam-v1" (default
    /// fixed). The reference cadence starts at one step, increases by two up
    /// to 101, and resets after densification.
    #[argh(option, default = "String::from(\"fixed\")")]
    geometry_rebuild_schedule: String,

    /// optional checkpoint PLY path written with exact safetensors optimizer
    /// and deterministic trainer-state sidecars at every densify cycle and
    /// bounded invocation endpoint. Defaults to `<output>.ckpt.ply` when either
    /// --densify-every or --stop-after-steps is nonzero; pass "none" to
    /// disable when running through the full budget.
    #[argh(option)]
    checkpoint: Option<String>,

    /// resume from a checkpoint PLY. Sibling `.safetensors` and `.trainstate`
    /// files are loaded automatically when present to restore parameters,
    /// Adam state, and deterministic RNG streams. Stored PowerFoam radii are
    /// preserved unless an explicit topology option replaces them.
    #[argh(option)]
    init_ply: Option<String>,

    /// resume: absolute step to continue the LR/densify schedule from.
    /// When omitted and --init-ply is set, read it from `<init-ply>.step`.
    #[argh(option)]
    resume_step: Option<usize>,

    /// stop after this many Adam updates in the current process and write a
    /// resumable checkpoint (default 0 = run to the global step budget).
    /// With active densification, choose a count that ends on its cadence.
    #[argh(option, default = "0")]
    stop_after_steps: usize,

    /// hold out every Nth image for testing (standard NVS "llffhold"
    /// protocol; Mip-NeRF-360 uses 8) and train on the rest. Default 0 =
    /// legacy contiguous split (first --views train, next --test-views
    /// test), which on filename-ordered captures tests on a tail arc of
    /// the trajectory (mostly extrapolation) and is not comparable to
    /// published numbers. --test-views caps the held-out count when > 0.
    #[argh(option, default = "0")]
    test_every: usize,
}

fn resolve_views_per_batch(
    requested: usize,
    pixel_batch: Option<usize>,
    patch_size: usize,
) -> Result<usize, &'static str> {
    let random_pixels = pixel_batch.is_some() && patch_size == 0;
    let selected = if requested == 0 {
        if random_pixels {
            16
        } else {
            1
        }
    } else {
        requested
    };
    if selected > 1 && !random_pixels {
        return Err("--views-per-batch > 1 requires random-pixel sampling");
    }
    Ok(selected)
}

fn select_adjacency(args: &Args) -> pipeline::AdjacencyKind {
    if args.powerfoam_reference_radii {
        pipeline::AdjacencyKind::PowerFoamReference
    } else if args.qhull {
        pipeline::AdjacencyKind::DelaunayQhull
    } else if args.cech_radius > 0.0 {
        pipeline::AdjacencyKind::Cech {
            radius_factor: args.cech_radius,
        }
    } else if args.knn > 0 {
        pipeline::AdjacencyKind::Knn(args.knn)
    } else if args.init_ply.is_some() {
        pipeline::AdjacencyKind::FromModel
    } else {
        pipeline::AdjacencyKind::Delaunay
    }
}

fn copy_endpoint_checkpoint(source: &path::Path, output: &path::Path) -> Result<(), String> {
    if source == output {
        return Ok(());
    }
    let tmp = output.with_extension("ply.tmp");
    if let Err(err) = fs::copy(source, &tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(format!(
            "failed to copy endpoint checkpoint {} to {}: {err}",
            source.display(),
            tmp.display(),
        ));
    }
    if let Err(err) = fs::rename(&tmp, output) {
        let _ = fs::remove_file(&tmp);
        return Err(format!(
            "failed to rename endpoint checkpoint copy to {}: {err}",
            output.display(),
        ));
    }
    Ok(())
}

fn main() {
    env_logger::init();
    let command_start = std::time::Instant::now();
    let args: Args = argh::from_env();
    let densify_schedule = match args.densify_schedule.as_str() {
        "fixed" => diff_render::DensifySchedule::Fixed,
        "radfoam-v1" => diff_render::DensifySchedule::RadFoamV1,
        other => {
            eprintln!("unknown --densify-schedule '{other}' (use 'fixed' or 'radfoam-v1')");
            std::process::exit(2);
        }
    };
    if densify_schedule == diff_render::DensifySchedule::RadFoamV1 {
        if args.densify_every == 0 {
            eprintln!("--densify-schedule radfoam-v1 requires --densify-every > 0 to enable it");
            std::process::exit(2);
        }
        if args.densify_until == 0 || args.densify_until <= args.densify_warmup {
            eprintln!(
                "--densify-schedule radfoam-v1 requires --densify-until greater than \
                 --densify-warmup"
            );
            std::process::exit(2);
        }
    }
    let geometry_rebuild_schedule = match args.geometry_rebuild_schedule.as_str() {
        "fixed" => diff_render::GeometryRebuildSchedule::Fixed,
        "radfoam-v1" => diff_render::GeometryRebuildSchedule::RadFoamV1,
        other => {
            eprintln!(
                "unknown --geometry-rebuild-schedule '{other}' (use 'fixed' or 'radfoam-v1')"
            );
            std::process::exit(2);
        }
    };

    if args.qhull && !cfg!(feature = "qhull") {
        eprintln!("--qhull requires building blade-volume-train with --features qhull");
        std::process::exit(2);
    }
    if !args.cech_radius.is_finite() || args.cech_radius < 0.0 {
        eprintln!("--cech-radius must be finite and non-negative");
        std::process::exit(2);
    }
    let topology_overrides = usize::from(args.qhull)
        + usize::from(args.knn > 0)
        + usize::from(args.cech_radius > 0.0)
        + usize::from(args.powerfoam_reference_radii);
    if topology_overrides > 1 {
        eprintln!(
            "--qhull, --knn, --cech-radius, and --powerfoam-reference-radii are mutually exclusive"
        );
        std::process::exit(2);
    }

    if args.stop_after_steps > 0 && args.checkpoint.as_deref() == Some("none") {
        eprintln!("--stop-after-steps requires checkpoints; remove --checkpoint none");
        std::process::exit(2);
    }
    if !args.position_lr_ratio.is_finite() || args.position_lr_ratio < 0.0 {
        eprintln!("--position-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.radius_lr_ratio.is_finite() || args.radius_lr_ratio < 0.0 {
        eprintln!("--radius-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_normal_lr_ratio.is_finite() || args.surface_normal_lr_ratio < 0.0 {
        eprintln!("--surface-normal-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_offset_lr_ratio.is_finite() || args.surface_offset_lr_ratio < 0.0 {
        eprintln!("--surface-offset-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_color_lr_ratio.is_finite() || args.surface_color_lr_ratio < 0.0 {
        eprintln!("--surface-color-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_detail_offset_lr_ratio.is_finite() || args.surface_detail_offset_lr_ratio < 0.0
    {
        eprintln!("--surface-detail-offset-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_detail_height_lr_ratio.is_finite() || args.surface_detail_height_lr_ratio < 0.0
    {
        eprintln!("--surface-detail-height-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_detail_color_lr_ratio.is_finite() || args.surface_detail_color_lr_ratio < 0.0 {
        eprintln!("--surface-detail-color-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_detail_density_lr_ratio.is_finite()
        || args.surface_detail_density_lr_ratio < 0.0
    {
        eprintln!("--surface-detail-density-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_detail_directional_axis_lr_ratio.is_finite()
        || args.surface_detail_directional_axis_lr_ratio < 0.0
    {
        eprintln!("--surface-detail-directional-axis-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_detail_directional_color_lr_ratio.is_finite()
        || args.surface_detail_directional_color_lr_ratio < 0.0
    {
        eprintln!("--surface-detail-directional-color-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.spherical_voronoi_axis_lr_ratio.is_finite()
        || args.spherical_voronoi_axis_lr_ratio < 0.0
    {
        eprintln!("--spherical-voronoi-axis-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.spherical_voronoi_color_lr_ratio.is_finite()
        || args.spherical_voronoi_color_lr_ratio < 0.0
    {
        eprintln!("--spherical-voronoi-color-lr-ratio must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.surface_normal_weight.is_finite() || args.surface_normal_weight < 0.0 {
        eprintln!("--surface-normal-weight must be finite and non-negative");
        std::process::exit(2);
    }
    if !args.interpenetration_weight.is_finite() || args.interpenetration_weight < 0.0 {
        eprintln!("--interpenetration-weight must be finite and non-negative");
        std::process::exit(2);
    }
    if args.interpenetration_weight > 0.0 && args.interpenetration_samples == 0 {
        eprintln!("--interpenetration-weight requires --interpenetration-samples > 0");
        std::process::exit(2);
    }
    if args.interpenetration_weight > 0.0
        && args.position_lr_ratio == 0.0
        && args.radius_lr_ratio == 0.0
    {
        eprintln!("--interpenetration-weight requires trainable positions or radii");
        std::process::exit(2);
    }
    if (args.position_lr_ratio > 0.0
        || args.radius_lr_ratio > 0.0
        || args.surface_normal_lr_ratio > 0.0
        || args.surface_offset_lr_ratio > 0.0
        || args.surface_detail_offset_lr_ratio > 0.0
        || args.surface_detail_height_lr_ratio > 0.0)
        && geometry_rebuild_schedule == diff_render::GeometryRebuildSchedule::Fixed
        && args.geometry_rebuild_every == 0
    {
        eprintln!("fixed geometry optimization requires --geometry-rebuild-every > 0");
        std::process::exit(2);
    }
    let fresh_weighted = args.cech_radius > 0.0 || args.powerfoam_reference_radii;
    if args.radius_lr_ratio > 0.0 && !fresh_weighted && args.init_ply.is_none() {
        eprintln!("--radius-lr-ratio requires a weighted initializer or --init-ply");
        std::process::exit(2);
    }
    if args.interpenetration_weight > 0.0 && !fresh_weighted && args.init_ply.is_none() {
        eprintln!("--interpenetration-weight requires a weighted initializer or --init-ply");
        std::process::exit(2);
    }
    if args.oriented_powerfoam && !fresh_weighted && args.init_ply.is_none() {
        eprintln!("--oriented-powerfoam requires a weighted initializer or --init-ply");
        std::process::exit(2);
    }
    if (args.surface_normal_lr_ratio > 0.0
        || args.surface_offset_lr_ratio > 0.0
        || args.surface_color_lr_ratio > 0.0
        || args.surface_detail_offset_lr_ratio > 0.0
        || args.surface_detail_height_lr_ratio > 0.0
        || args.surface_detail_color_lr_ratio > 0.0
        || args.surface_detail_density_lr_ratio > 0.0
        || args.surface_detail_directional_axis_lr_ratio > 0.0
        || args.surface_detail_directional_color_lr_ratio > 0.0
        || args.spherical_voronoi_axis_lr_ratio > 0.0
        || args.spherical_voronoi_color_lr_ratio > 0.0
        || args.surface_normal_weight > 0.0)
        && !args.oriented_powerfoam
        && args.init_ply.is_none()
    {
        eprintln!(
            "oriented-surface training requires --oriented-powerfoam or an oriented --init-ply"
        );
        std::process::exit(2);
    }
    let pixel_batch = if args.pixel_batch == 0 {
        None
    } else {
        Some(args.pixel_batch)
    };
    let views_per_batch =
        resolve_views_per_batch(args.views_per_batch, pixel_batch, args.patch_size).unwrap_or_else(
            |message| {
                eprintln!("{message}");
                std::process::exit(2);
            },
        );
    let adjacency = select_adjacency(&args);
    let initialization = match args.initialization.as_str() {
        "top-track" => pipeline::InitialPointPolicy::TopTrackLength,
        "radfoam-v1" => pipeline::InitialPointPolicy::RadFoamV1,
        other => {
            eprintln!("unknown --initialization '{other}' (use 'top-track' or 'radfoam-v1')");
            std::process::exit(2);
        }
    };
    let lr_schedule = match args.lr_schedule.as_str() {
        "constant" => diff_render::LrSchedule::Constant,
        "cosine" => diff_render::LrSchedule::Cosine,
        "radfoam-v1" => diff_render::LrSchedule::RadFoamV1,
        other => {
            eprintln!(
                "unknown --lr-schedule '{other}' (use 'constant', 'cosine', or 'radfoam-v1')"
            );
            std::process::exit(2);
        }
    };
    let lr_groups = match args.lr_groups.as_str() {
        "legacy" => diff_render::LrGroups::Legacy,
        "radfoam-v1-relative" => diff_render::LrGroups::RadFoamV1Relative,
        other => {
            eprintln!("unknown --lr-groups '{other}' (use 'legacy' or 'radfoam-v1-relative')");
            std::process::exit(2);
        }
    };
    if lr_schedule == diff_render::LrSchedule::RadFoamV1
        && geometry_rebuild_schedule == diff_render::GeometryRebuildSchedule::Fixed
        && args.geometry_rebuild_every == 0
    {
        eprintln!(
            "--lr-schedule radfoam-v1 with fixed geometry requires \
             --geometry-rebuild-every > 0"
        );
        std::process::exit(2);
    }
    if lr_groups == diff_render::LrGroups::RadFoamV1Relative
        && geometry_rebuild_schedule == diff_render::GeometryRebuildSchedule::Fixed
        && args.geometry_rebuild_every == 0
    {
        eprintln!(
            "--lr-groups radfoam-v1-relative with fixed geometry requires \
             --geometry-rebuild-every > 0"
        );
        std::process::exit(2);
    }
    if lr_schedule == diff_render::LrSchedule::RadFoamV1
        && lr_groups != diff_render::LrGroups::Legacy
    {
        eprintln!("--lr-schedule radfoam-v1 already supplies its parameter groups");
        std::process::exit(2);
    }
    let color_loss = match args.color_loss.as_str() {
        "l1" => diff_render::ColorLoss::L1,
        "smooth-l1" => diff_render::ColorLoss::SmoothL1,
        other => {
            eprintln!("unknown --color-loss '{other}' (use 'l1' or 'smooth-l1')");
            std::process::exit(2);
        }
    };
    // Resume bookkeeping: load deterministic trainer state first, then
    // determine the schedule step. Explicit --resume-step wins, followed by
    // the versioned trainer state and the legacy `<ply>.step` sidecar.
    let init_ply = args.init_ply.as_deref().map(path::PathBuf::from);
    let resume_training_state = init_ply.as_ref().and_then(|ply| {
        let state_path = ply.with_extension("trainstate");
        if !state_path.is_file() {
            return None;
        }
        match diff_render::load_training_state(ply) {
            Ok(state) => {
                eprintln!(
                    "resume: loaded trainer state at step {} from {}",
                    state.step,
                    state_path.display(),
                );
                Some(state)
            }
            Err(err) => {
                eprintln!("resume: invalid trainer state: {err}");
                std::process::exit(2);
            }
        }
    });
    let resume_step = match args.resume_step {
        Some(s) => s,
        None if resume_training_state.is_some() => resume_training_state.unwrap().step,
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
    if let Some(state) = resume_training_state {
        if state.step != resume_step {
            eprintln!(
                "resume: --resume-step {} disagrees with trainer-state step {}",
                resume_step, state.step,
            );
            std::process::exit(2);
        }
    }
    let resume_state_path = init_ply.as_ref().and_then(|ply| {
        let state = ply.with_extension("safetensors");
        state.is_file().then_some(state)
    });
    if resume_training_state.is_some() && resume_state_path.is_none() {
        eprintln!("resume: trainer state exists but the paired safetensors checkpoint is missing");
        std::process::exit(2);
    }

    let config = pipeline::PipelineConfig {
        resolution: (args.width, args.height),
        max_steps: args.max_steps,
        max_views: Some(args.views),
        max_initial_points: (args.max_points > 0).then_some(args.max_points),
        initialization,
        fit: diff_render::AppearanceFitConfig {
            learning_rate: args.learning_rate,
            epochs: args.epochs,
            pixel_batch,
            views_per_batch,
            steps_per_view: args.steps_per_view,
            sh_degree: args.sh_degree,
            color_loss,
            lr_schedule,
            lr_groups,
            lr_min_ratio: args.lr_min_ratio,
            patch_size: args.patch_size,
            grad_loss_weight: args.grad_loss_weight,
            opacity_weight: args.opacity_weight,
            distortion_weight: args.distortion_weight,
            quantile_weight: args.quantile_weight,
            interpenetration_weight: args.interpenetration_weight,
            interpenetration_samples: args.interpenetration_samples,
            softplus_beta: args.softplus_beta,
            background_rgb: if args.white_background {
                [1.0; 3]
            } else {
                [0.0; 3]
            },
            position_lr_ratio: args.position_lr_ratio,
            radius_lr_ratio: args.radius_lr_ratio,
            surface_normal_lr_ratio: args.surface_normal_lr_ratio,
            surface_offset_lr_ratio: args.surface_offset_lr_ratio,
            surface_color_lr_ratio: args.surface_color_lr_ratio,
            surface_detail_offset_lr_ratio: args.surface_detail_offset_lr_ratio,
            surface_detail_height_lr_ratio: args.surface_detail_height_lr_ratio,
            surface_detail_color_lr_ratio: args.surface_detail_color_lr_ratio,
            surface_detail_density_lr_ratio: args.surface_detail_density_lr_ratio,
            surface_detail_directional_axis_lr_ratio: args.surface_detail_directional_axis_lr_ratio,
            surface_detail_directional_color_lr_ratio: args
                .surface_detail_directional_color_lr_ratio,
            spherical_voronoi_axis_lr_ratio: args.spherical_voronoi_axis_lr_ratio,
            spherical_voronoi_color_lr_ratio: args.spherical_voronoi_color_lr_ratio,
            surface_normal_weight: args.surface_normal_weight,
            powerfoam_candidate_capacity: args.powerfoam_candidate_capacity,
            geometry_rebuild_every: args.geometry_rebuild_every,
            geometry_rebuild_schedule,
            rebuild_with_qhull: args.qhull,
            resume_step,
            stop_after_steps: (args.stop_after_steps > 0).then_some(args.stop_after_steps),
            resume_state_path,
            resume_training_state,
            checkpoint_path: match args.checkpoint.as_deref() {
                Some("none") => None,
                Some(p) => Some(path::PathBuf::from(p)),
                // Default: checkpoint next to the output when densifying or
                // ending this invocation before the global budget.
                None if args.densify_every > 0 || args.stop_after_steps > 0 => {
                    Some(path::Path::new(&args.output).with_extension("ckpt.ply"))
                }
                None => None,
            },
            densify: if args.densify_every > 0 {
                Some(diff_render::DensifyConfig {
                    schedule: densify_schedule,
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
                    prune_contribution: args.prune_contribution,
                    suppress_contribution: args.suppress_contribution,
                    prune_radius: args.prune_radius,
                    contribution_views: args.contribution_views,
                })
            } else {
                None
            },
            ..diff_render::AppearanceFitConfig::default()
        },
        far_plane: 100.0,
        initial_density: args.initial_density,
        adjacency,
        oriented_powerfoam: args.oriented_powerfoam,
        init_ply,
        test_every: args.test_every,
    };

    let sparse = path::Path::new(&args.sparse);
    let images = path::Path::new(&args.images);
    if !sparse.is_dir() {
        eprintln!(
            "sparse reconstruction directory does not exist: {}",
            sparse.display()
        );
        std::process::exit(2);
    }
    if !images.is_dir() {
        eprintln!("image directory does not exist: {}", images.display());
        std::process::exit(2);
    }
    let Some(gpu) = fit::try_init_gpu() else {
        eprintln!("no supported GPU device — cannot train");
        std::process::exit(2);
    };
    let pipeline_start = std::time::Instant::now();
    let outcome = pipeline::train_colmap_appearance_split(
        sparse,
        images,
        &config,
        args.test_views,
        gpu.clone(),
    );
    let pipeline_duration = pipeline_start.elapsed();

    let output = path::Path::new(&args.output);
    let serialization_start = std::time::Instant::now();
    let reused_checkpoint =
        outcome.endpoint_checkpoint.as_ref().is_some_and(
            |checkpoint| match copy_endpoint_checkpoint(checkpoint, output) {
                Ok(()) => true,
                Err(err) => {
                    eprintln!("{err}; serializing the in-memory model instead");
                    false
                }
            },
        );
    if !reused_checkpoint {
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
    }
    let serialization_duration = serialization_start.elapsed();
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
    let evaluation_start = std::time::Instant::now();
    if !args.skip_eval && (args.test_views > 0 || args.test_every > 0) {
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
            &config,
            test_images.iter().copied(),
        );
        if test_views.is_empty() {
            eprintln!("no usable test views (image files missing?)");
        } else {
            let mut evaluator =
                pipeline::GpuViewEvaluator::new(&outcome.model, &config, gpu.clone());
            let psnrs = evaluator
                .evaluate(&test_views, config.fit.background_rgb)
                .unwrap_or_else(|err| {
                    eprintln!("GPU test-view evaluation failed: {err}");
                    std::process::exit(3);
                });
            // Train-set PSNR too, for comparison.
            let train_views = pipeline::build_views_from(
                &outcome.reconstruction,
                images,
                &config,
                train_images.iter().copied(),
            );
            let train_psnrs = evaluator
                .evaluate(&train_views, config.fit.background_rgb)
                .unwrap_or_else(|err| {
                    eprintln!("GPU train-view evaluation failed: {err}");
                    std::process::exit(3);
                });
            evaluator.deinit();
            let avg_train: f32 =
                train_psnrs.iter().copied().sum::<f32>() / train_psnrs.len() as f32;
            let avg_test: f32 = psnrs.iter().copied().sum::<f32>() / psnrs.len() as f32;
            println!(
                "PSNR train (avg over {} views): {avg_train:.4} dB",
                train_psnrs.len()
            );
            println!(
                "PSNR test  (avg over {} views): {avg_test:.4} dB",
                psnrs.len()
            );
            for (img, p) in test_images.iter().zip(psnrs.iter()) {
                println!("  test {}: {:.2} dB", img.name, p);
            }
        }
    }
    let evaluation_duration = evaluation_start.elapsed();

    let novel_start = std::time::Instant::now();
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
    let novel_duration = novel_start.elapsed();
    println!(
        "phase timing: wall={:.3}s pipeline={:.3}s final-ply={:.3}s \
         evaluation={:.3}s novel-render={:.3}s",
        command_start.elapsed().as_secs_f64(),
        pipeline_duration.as_secs_f64(),
        serialization_duration.as_secs_f64(),
        evaluation_duration.as_secs_f64(),
        novel_duration.as_secs_f64(),
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_checkpoint_copy_atomically_replaces_final_ply() {
        let stem = format!("blade-volume-final-ply-copy-{}", std::process::id());
        let source = std::env::temp_dir().join(format!("{stem}.ckpt.ply"));
        let output = std::env::temp_dir().join(format!("{stem}.ply"));
        let tmp = output.with_extension("ply.tmp");
        fs::write(&source, b"current endpoint checkpoint").unwrap();
        fs::write(&output, b"stale final output").unwrap();

        copy_endpoint_checkpoint(&source, &output).unwrap();

        assert_eq!(fs::read(&output).unwrap(), b"current endpoint checkpoint");
        assert_eq!(fs::read(&source).unwrap(), b"current endpoint checkpoint");
        assert!(!tmp.exists());
        fs::remove_file(source).unwrap();
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn batching_uses_selected_defaults_and_accepts_overrides() {
        let default = <Args as argh::FromArgs>::from_args(
            &["train_colmap"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--output",
                "model.ply",
            ],
        )
        .unwrap();
        assert_eq!(default.pixel_batch, 1024);
        assert_eq!(default.views_per_batch, 0);
        assert!(!default.skip_eval);
        assert_eq!(
            resolve_views_per_batch(default.views_per_batch, Some(default.pixel_batch), 0),
            Ok(16)
        );

        let explicit = <Args as argh::FromArgs>::from_args(
            &["train_colmap"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--output",
                "model.ply",
                "--pixel-batch",
                "256",
                "--views-per-batch",
                "16",
            ],
        )
        .unwrap();
        assert_eq!(explicit.pixel_batch, 256);
        assert_eq!(explicit.views_per_batch, 16);
        assert!(!explicit.skip_eval);

        let skip_eval = <Args as argh::FromArgs>::from_args(
            &["train_colmap"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--output",
                "model.ply",
                "--skip-eval",
            ],
        )
        .unwrap();
        assert!(skip_eval.skip_eval);
    }

    #[test]
    fn batching_auto_selection_preserves_non_random_modes() {
        assert_eq!(resolve_views_per_batch(0, Some(1024), 0), Ok(16));
        assert_eq!(resolve_views_per_batch(0, None, 0), Ok(1));
        assert_eq!(resolve_views_per_batch(0, Some(1024), 16), Ok(1));
        assert_eq!(resolve_views_per_batch(1, None, 0), Ok(1));
        assert_eq!(
            resolve_views_per_batch(16, None, 0),
            Err("--views-per-batch > 1 requires random-pixel sampling")
        );
        assert_eq!(
            resolve_views_per_batch(16, Some(1024), 16),
            Err("--views-per-batch > 1 requires random-pixel sampling")
        );
    }

    #[test]
    fn checkpoint_adjacency_preserves_model_semantics_without_an_override() {
        let resumed = <Args as argh::FromArgs>::from_args(
            &["train_colmap"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--output",
                "model.ply",
                "--init-ply",
                "checkpoint.ply",
            ],
        )
        .unwrap();
        assert!(matches!(
            select_adjacency(&resumed),
            pipeline::AdjacencyKind::FromModel
        ));

        let reinitialized = <Args as argh::FromArgs>::from_args(
            &["train_colmap"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--output",
                "model.ply",
                "--init-ply",
                "checkpoint.ply",
                "--cech-radius",
                "1.7",
            ],
        )
        .unwrap();
        let pipeline::AdjacencyKind::Cech { radius_factor } = select_adjacency(&reinitialized)
        else {
            panic!("explicit Cech override was not selected");
        };
        assert_eq!(radius_factor, 1.7);

        let reference = <Args as argh::FromArgs>::from_args(
            &["train_colmap"],
            &[
                "--sparse",
                "sparse",
                "--images",
                "images",
                "--output",
                "model.ply",
                "--powerfoam-reference-radii",
            ],
        )
        .unwrap();
        assert!(matches!(
            select_adjacency(&reference),
            pipeline::AdjacencyKind::PowerFoamReference
        ));
    }
}
