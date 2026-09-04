//! Prepare one LUCES-MV object for the pose-only point-cloud training path.
//!
//! The source is already geometrically and radiometrically calibrated. This
//! importer writes one shared COLMAP pose bundle, normalized sRGB images for
//! each LED, foreground masks, and fixed view/light lists. It does not read or
//! convert the released ground-truth mesh.

use blade_volume_train as train;
use std::{fs, path};

#[derive(argh::FromArgs)]
/// Convert a downloaded LUCES-MV object to the training layout.
struct Args {
    /// object directory containing `view_000` through `view_066`
    #[argh(option)]
    input: String,

    /// first official camera parameter text file
    #[argh(option)]
    camera_one: String,

    /// second official camera parameter text file
    #[argh(option)]
    camera_two: String,

    /// new output directory
    #[argh(option)]
    output: String,

    /// output image width (default 400; height preserves aspect)
    #[argh(option, default = "400")]
    width: usize,
}

fn view_name(index: usize) -> String {
    format!("view_{:03}.png", train::inverse::luces::VIEW_IDS[index])
}

fn write_splits(output: &path::Path, views: usize) -> std::io::Result<()> {
    let view_list = |held: bool| {
        (0..views)
            .filter(|index| train::inverse::luces::HELD_VIEW_INDICES.contains(index) == held)
            .map(view_name)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    let light_list = |held: bool| {
        (0..train::inverse::luces::LIGHT_COUNT)
            .filter(|index| train::inverse::luces::HELD_LIGHT_INDICES.contains(index) == held)
            .map(|index| format!("{:02}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    fs::write(output.join("train-views.txt"), view_list(false))?;
    fs::write(output.join("test-views.txt"), view_list(true))?;
    fs::write(output.join("train-lights.txt"), light_list(false))?;
    fs::write(output.join("test-lights.txt"), light_list(true))
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let camera_one = path::Path::new(&args.camera_one);
    let camera_two = path::Path::new(&args.camera_two);
    let output = path::Path::new(&args.output);
    if output.exists() {
        return Err(format!("output {} already exists", output.display()));
    }
    fs::create_dir_all(output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;

    for light in 0..train::inverse::luces::LIGHT_COUNT {
        let dataset =
            train::inverse::luces::load(input, camera_one, camera_two, args.width, &[light])?;
        let capture = &dataset.captures[0];
        if light == 0 {
            train::calibrated::write_colmap(&output.join("sparse/0"), capture, view_name)
                .map_err(|error| format!("cannot write COLMAP poses: {error}"))?;
            train::calibrated::write_masks(&output.join("masks"), capture, view_name)?;
            write_splits(output, capture.views.len())
                .map_err(|error| format!("cannot write fixed splits: {error}"))?;
        }
        train::calibrated::write_capture_images(
            &output.join(format!("light-{:02}/images", light + 1)),
            capture,
            view_name,
        )?;
        println!(
            "wrote LED {:02}/{}",
            light + 1,
            train::inverse::luces::LIGHT_COUNT
        );
    }
    Ok(())
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
    fn fixed_splits_are_disjoint_and_complete() {
        let train_views: Vec<_> = (0..12)
            .filter(|index| !train::inverse::luces::HELD_VIEW_INDICES.contains(index))
            .collect();
        let test_views: Vec<_> = (0..12)
            .filter(|index| train::inverse::luces::HELD_VIEW_INDICES.contains(index))
            .collect();
        let train_lights: Vec<_> = (0..train::inverse::luces::LIGHT_COUNT)
            .filter(|index| !train::inverse::luces::HELD_LIGHT_INDICES.contains(index))
            .collect();
        assert_eq!(train_views.len(), 9);
        assert_eq!(test_views.len(), 3);
        assert_eq!(train_lights.len(), 12);
        assert_eq!(train::inverse::luces::HELD_LIGHT_INDICES.len(), 3);
        assert_eq!(
            train_views.as_slice(),
            train::inverse::luces::TRAIN_VIEW_INDICES
        );
        assert_eq!(
            train_lights.as_slice(),
            train::inverse::luces::TRAIN_LIGHT_INDICES
        );
        assert!(train_views.iter().all(|index| !test_views.contains(index)));
        assert!(train_lights
            .iter()
            .all(|index| !train::inverse::luces::HELD_LIGHT_INDICES.contains(index)));
    }
}
