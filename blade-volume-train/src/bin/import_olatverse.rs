//! Prepare one OLATverse object for the point-cloud reconstruction path.
//!
//! Only the full-bright capture is materialized for geometry. OLAT images stay
//! in the source dataset and are decoded directly by `fit_olatverse`, avoiding
//! thousands of redundant PNG copies. The released mesh, normals, and albedo
//! are never read by this path.

use blade_volume_train as train;
use std::{fs, path};

const PATCH_MATCH_SOURCES: usize = 12;

#[derive(argh::FromArgs)]
/// Convert one extracted OLATverse object to the training layout.
struct Args {
    /// object directory such as `OLATverse_Upload_Val/data-042325-C276`
    #[argh(option)]
    input: String,

    /// new output directory
    #[argh(option)]
    output: String,

    /// output image width (default 320; height preserves aspect)
    #[argh(option, default = "320")]
    width: usize,
}

fn view_name(index: usize) -> String {
    format!("{}.png", train::inverse::olatverse::VIEW_NAMES[index])
}

fn light_list(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|index| format!("{index:03}"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn view_list(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|&index| view_name(index))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn write_splits(output: &path::Path) -> Result<(), String> {
    fs::write(
        output.join("train-views.txt"),
        view_list(&train::inverse::olatverse::TRAIN_VIEW_INDICES),
    )
    .map_err(|error| format!("cannot write train views: {error}"))?;
    fs::write(
        output.join("test-views.txt"),
        view_list(&train::inverse::olatverse::HELD_VIEW_INDICES),
    )
    .map_err(|error| format!("cannot write test views: {error}"))?;
    fs::write(
        output.join("train-lights.txt"),
        light_list(&train::inverse::olatverse::train_light_indices()),
    )
    .map_err(|error| format!("cannot write train lights: {error}"))?;
    fs::write(
        output.join("test-lights.txt"),
        light_list(&train::inverse::olatverse::held_light_indices()),
    )
    .map_err(|error| format!("cannot write test lights: {error}"))
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let output = path::Path::new(&args.output);
    if output.exists() {
        return Err(format!("output {} already exists", output.display()));
    }
    let capture = train::inverse::olatverse::load_full_bright(input, args.width)?;
    fs::create_dir_all(output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    train::calibrated::write_colmap(&output.join("sparse/0"), &capture, view_name)
        .map_err(|error| format!("cannot write COLMAP poses: {error}"))?;
    train::calibrated::write_masks(&output.join("masks"), &capture, view_name)?;
    train::calibrated::write_capture_images(&output.join("images"), &capture, view_name)?;
    let patch_match = train::calibrated::patch_match_config(
        &capture,
        &train::inverse::olatverse::TRAIN_VIEW_INDICES,
        view_name,
        PATCH_MATCH_SOURCES,
    )?;
    fs::write(output.join("patch-match.cfg"), patch_match)
        .map_err(|error| format!("cannot write PatchMatch graph: {error}"))?;
    write_splits(output)
}

fn main() {
    let args: Args = argh::from_env();
    if let Err(message) = run(&args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_protocol_uses_source_camera_and_light_names() {
        assert_eq!(view_name(0), "Cam02.png");
        assert_eq!(view_name(23), "Cam40.png");
        assert_eq!(view_name(24), "Cam01.png");
        assert!(light_list(&[0, 9, 309]).starts_with("000\n009\n309\n"));
    }
}
