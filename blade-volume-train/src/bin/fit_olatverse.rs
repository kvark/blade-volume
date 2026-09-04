//! Fit point-cloud surface properties on the published OLATverse split.
//!
//! The 104 construction lights and 24 construction cameras are loaded first.
//! The 103 held lights are not opened until fitted models are serialized.

use blade_volume_train as train;
use std::path;

#[derive(argh::FromArgs)]
/// Fit OLATverse normals, diffuse materials, and optional Gaussian geometry.
struct Args {
    /// extracted OLATverse object directory
    #[argh(option)]
    input: String,

    /// official shared/all_lights.json
    #[argh(option)]
    lights: String,

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

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let lights = path::Path::new(&args.lights);
    let train_lights = train::inverse::olatverse::train_light_indices();
    let held_lights = train::inverse::olatverse::held_light_indices();
    train::calibrated::fit(
        || train::inverse::olatverse::load(input, lights, args.width, &train_lights),
        || train::inverse::olatverse::load(input, lights, args.width, &held_lights),
        train::calibrated::FitOptions {
            surface: path::Path::new(&args.surface),
            output: path::Path::new(&args.output),
            gaussian_output: args.gaussian_output.as_deref().map(path::Path::new),
            dump: args.dump.as_deref().map(path::Path::new),
            width: args.width,
            rounds: args.rounds,
            normal_candidates: args.normal_candidates,
            albedo_ceiling: args.albedo_ceiling,
            train_views: &train::inverse::olatverse::TRAIN_VIEW_INDICES,
            held_views: &train::inverse::olatverse::HELD_VIEW_INDICES,
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
