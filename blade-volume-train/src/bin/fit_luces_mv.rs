//! Fit calibrated near-field surface properties on a reconstructed LUCES-MV cloud.
//!
//! Training uses the fixed 9-camera/12-light split. The three excluded LEDs
//! are not loaded until the fitted scalar and Gaussian point clouds have been
//! serialized, keeping the held-light cross-product out of model selection.

use blade_volume_train as train;
use std::path;

#[derive(argh::FromArgs)]
/// Fit LUCES-MV normals, diffuse materials, and optional Gaussian geometry.
struct Args {
    /// object directory containing the calibrated views
    #[argh(option)]
    input: String,

    /// first official camera parameter text file
    #[argh(option)]
    camera_one: String,

    /// second official camera parameter text file
    #[argh(option)]
    camera_two: String,

    /// reconstructed relightable surfel cloud
    #[argh(option)]
    surface: String,

    /// fitted relightable surfel output
    #[argh(option)]
    output: String,

    /// optional fitted relightable Gaussian output
    #[argh(option)]
    gaussian_output: Option<String>,

    /// optional directory for production held-light/held-camera renders
    #[argh(option)]
    dump: Option<String>,

    /// fitting image width (default 128; height preserves aspect)
    #[argh(option, default = "128")]
    width: usize,

    /// alternating normal/material rounds (default 3)
    #[argh(option, default = "3")]
    rounds: usize,

    /// hemisphere candidates tested per normal update (default 1024)
    #[argh(option, default = "1024")]
    normal_candidates: usize,

    /// maximum diffuse albedo (default 1)
    #[argh(option, default = "1.0")]
    albedo_ceiling: f32,
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let camera_one = path::Path::new(&args.camera_one);
    let camera_two = path::Path::new(&args.camera_two);
    train::calibrated::fit(
        || {
            train::inverse::luces::load(
                input,
                camera_one,
                camera_two,
                args.width,
                &train::inverse::luces::TRAIN_LIGHT_INDICES,
            )
        },
        || {
            train::inverse::luces::load(
                input,
                camera_one,
                camera_two,
                args.width,
                &train::inverse::luces::HELD_LIGHT_INDICES,
            )
        },
        train::calibrated::FitOptions {
            surface: path::Path::new(&args.surface),
            output: path::Path::new(&args.output),
            gaussian_output: args.gaussian_output.as_deref().map(path::Path::new),
            dump: args.dump.as_deref().map(path::Path::new),
            width: args.width,
            rounds: args.rounds,
            normal_candidates: args.normal_candidates,
            albedo_ceiling: args.albedo_ceiling,
            train_views: &train::inverse::luces::TRAIN_VIEW_INDICES,
            held_views: &train::inverse::luces::HELD_VIEW_INDICES,
            light_digits: 2,
        },
    )
}

fn main() {
    let args: Args = argh::from_env();
    if let Err(message) = run(&args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
