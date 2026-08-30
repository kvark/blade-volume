//! Fit surface properties on the fixed DiLiGenT-MV camera/light split.
//!
//! Training uses 16 cameras and 24 lights. The four excluded cameras and eight
//! excluded lights are not loaded until the models have been serialized.

use blade_volume_train as train;
use std::path;

#[path = "support/calibrated.rs"]
mod calibrated;

#[derive(argh::FromArgs)]
/// Fit DiLiGenT-MV normals, diffuse materials, and optional Gaussian geometry.
struct Args {
    /// object directory such as `mvpmsData/bearPNG`
    #[argh(option)]
    input: String,

    /// reconstructed relightable surfel cloud
    #[argh(option)]
    surface: String,

    /// fitted relightable surfel output
    #[argh(option)]
    output: String,

    /// optional fitted relightable Gaussian output
    #[argh(option)]
    gaussian_output: Option<String>,

    /// optional directory for held-light/held-camera renders
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

fn dataset(value: train::inverse::diligent_mv::Dataset) -> calibrated::Dataset {
    calibrated::Dataset {
        captures: value.captures,
        lights: value.lights,
        source_light_indices: value.source_light_indices,
    }
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    calibrated::fit(
        || {
            train::inverse::diligent_mv::load(
                input,
                args.width,
                &train::inverse::diligent_mv::TRAIN_LIGHT_INDICES,
            )
            .map(dataset)
        },
        || {
            train::inverse::diligent_mv::load(
                input,
                args.width,
                &train::inverse::diligent_mv::HELD_LIGHT_INDICES,
            )
            .map(dataset)
        },
        calibrated::FitOptions {
            surface: path::Path::new(&args.surface),
            output: path::Path::new(&args.output),
            gaussian_output: args.gaussian_output.as_deref().map(path::Path::new),
            dump: args.dump.as_deref().map(path::Path::new),
            width: args.width,
            rounds: args.rounds,
            normal_candidates: args.normal_candidates,
            albedo_ceiling: args.albedo_ceiling,
            train_views: &train::inverse::diligent_mv::TRAIN_VIEW_INDICES,
            held_views: &train::inverse::diligent_mv::HELD_VIEW_INDICES,
            light_digits: 3,
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
