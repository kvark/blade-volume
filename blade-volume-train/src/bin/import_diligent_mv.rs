//! Prepare one DiLiGenT-MV object for pose-only point-cloud training.
//!
//! Only the predeclared 32-light gate is materialized. Source linear RGB is
//! intensity-normalized and encoded as sRGB for the ordinary capture loader;
//! the ground-truth mesh and normals are never read by this path.

use blade_volume_train as train;
use std::{fs, path};

#[path = "support/import_calibrated.rs"]
mod import_calibrated;

#[derive(argh::FromArgs)]
/// Convert a downloaded DiLiGenT-MV object to the training layout.
struct Args {
    /// object directory such as `mvpmsData/bearPNG`
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
    format!("view_{:02}.png", index + 1)
}

fn write_splits(output: &path::Path) -> std::io::Result<()> {
    let view_list = |held: bool| {
        (0..train::inverse::diligent_mv::VIEW_COUNT)
            .filter(|index| train::inverse::diligent_mv::HELD_VIEW_INDICES.contains(index) == held)
            .map(view_name)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    let light_list = |indices: &[usize]| {
        indices
            .iter()
            .map(|index| format!("{:03}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    fs::write(output.join("train-views.txt"), view_list(false))?;
    fs::write(output.join("test-views.txt"), view_list(true))?;
    fs::write(
        output.join("train-lights.txt"),
        light_list(&train::inverse::diligent_mv::TRAIN_LIGHT_INDICES),
    )?;
    fs::write(
        output.join("test-lights.txt"),
        light_list(&train::inverse::diligent_mv::HELD_LIGHT_INDICES),
    )
}

fn run(args: &Args) -> Result<(), String> {
    let input = path::Path::new(&args.input);
    let output = path::Path::new(&args.output);
    if output.exists() {
        return Err(format!("output {} already exists", output.display()));
    }
    let mut selected = train::inverse::diligent_mv::TRAIN_LIGHT_INDICES.to_vec();
    selected.extend(train::inverse::diligent_mv::HELD_LIGHT_INDICES);
    selected.sort_unstable();
    let dataset = train::inverse::diligent_mv::load(input, args.width, &selected)?;
    fs::create_dir_all(output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    import_calibrated::write_colmap(&output.join("sparse/0"), &dataset.captures[0], view_name)
        .map_err(|error| format!("cannot write COLMAP poses: {error}"))?;
    import_calibrated::write_masks(&output.join("masks"), &dataset.captures[0], view_name)?;
    write_splits(output).map_err(|error| format!("cannot write fixed splits: {error}"))?;
    for (capture, &light) in dataset.captures.iter().zip(&selected) {
        import_calibrated::write_capture_images(
            &output.join(format!("light-{:03}/images", light + 1)),
            capture,
            view_name,
        )?;
        println!("wrote light {:03}/{}", light + 1, selected.len());
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
    fn view_names_and_light_lists_are_source_indexed() {
        assert_eq!(view_name(0), "view_01.png");
        assert_eq!(view_name(19), "view_20.png");
        assert_eq!(train::inverse::diligent_mv::TRAIN_LIGHT_INDICES[0], 3);
        assert_eq!(train::inverse::diligent_mv::HELD_LIGHT_INDICES[0], 0);
    }
}
