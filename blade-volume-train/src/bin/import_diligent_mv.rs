//! Prepare one DiLiGenT-MV object for pose-only point-cloud training.
//!
//! Only the predeclared 32-light gate is materialized. Source linear RGB is
//! intensity-normalized and encoded as sRGB for the ordinary capture loader;
//! the ground-truth mesh and normals are never read by this path. The 24
//! construction lights also produce one diffuse-albedo image per construction
//! camera for a leakage-free dense-stereo pass.

use blade_volume_train as train;
use std::{fs, path};

const PATCH_MATCH_SOURCES: usize = 12;

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

fn construction_view_name(index: usize) -> String {
    view_name(train::inverse::diligent_mv::TRAIN_VIEW_INDICES[index])
}

fn construction_capture(
    capture: &train::inverse::capture::Capture,
) -> train::inverse::capture::Capture {
    let views = train::inverse::diligent_mv::TRAIN_VIEW_INDICES
        .iter()
        .map(|&index| {
            let source = &capture.views[index];
            train::inverse::capture::View {
                name: view_name(index),
                camera: source.camera,
                pixels: source.pixels.clone(),
                mask: source.mask.clone(),
            }
        })
        .collect();
    train::inverse::capture::Capture {
        width: capture.width,
        height: capture.height,
        views,
    }
}

fn patch_match_config(capture: &train::inverse::capture::Capture) -> Result<String, String> {
    train::calibrated::patch_match_config(
        capture,
        &(0..capture.views.len()).collect::<Vec<_>>(),
        construction_view_name,
        PATCH_MATCH_SOURCES,
    )
}

fn write_albedo_mvs(
    output: &path::Path,
    capture: &train::inverse::capture::Capture,
) -> Result<(), String> {
    let output = output.join("albedo");
    let capture = construction_capture(capture);
    train::calibrated::write_colmap(&output.join("sparse"), &capture, construction_view_name)
        .map_err(|error| format!("cannot write albedo COLMAP poses: {error}"))?;
    train::calibrated::write_capture_images(
        &output.join("images"),
        &capture,
        construction_view_name,
    )?;
    fs::write(
        output.join("patch-match.cfg"),
        patch_match_config(&capture)?,
    )
    .map_err(|error| format!("cannot write albedo PatchMatch graph: {error}"))
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
    let albedo = train::inverse::diligent_mv::construction_photometric_albedo(&dataset)?;
    fs::create_dir_all(output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    write_albedo_mvs(output, &albedo)?;
    drop(albedo);
    train::calibrated::write_colmap(&output.join("sparse/0"), &dataset.captures[0], view_name)
        .map_err(|error| format!("cannot write COLMAP poses: {error}"))?;
    train::calibrated::write_masks(&output.join("masks"), &dataset.captures[0], view_name)?;
    write_splits(output).map_err(|error| format!("cannot write fixed splits: {error}"))?;
    for (capture, &light) in dataset.captures.iter().zip(&selected) {
        train::calibrated::write_capture_images(
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

    #[test]
    fn albedo_patch_match_graph_is_nearest_camera_and_construction_only() {
        let views = (0..train::inverse::diligent_mv::TRAIN_VIEW_INDICES.len())
            .map(|index| train::inverse::capture::View {
                name: String::new(),
                camera: blade_volume::CameraParams {
                    cam_position: [index as f32, 0.0, 0.0],
                    ..Default::default()
                },
                pixels: Vec::new(),
                mask: None,
            })
            .collect();
        let capture = train::inverse::capture::Capture {
            width: 1,
            height: 1,
            views,
        };
        let graph = patch_match_config(&capture).unwrap();
        let lines = graph.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 2 * capture.views.len());
        assert_eq!(lines[0], "view_02.png");
        assert_eq!(lines[1].split(", ").count(), PATCH_MATCH_SOURCES);
        assert!(lines[1].starts_with("view_03.png, view_04.png"));
        for &held in &train::inverse::diligent_mv::HELD_VIEW_INDICES {
            assert!(!graph.contains(&view_name(held)));
        }
        assert!(!graph.contains("__auto__"));
    }

    #[test]
    fn albedo_capture_physically_excludes_held_cameras() {
        let source = train::inverse::capture::Capture {
            width: 1,
            height: 1,
            views: (0..train::inverse::diligent_mv::VIEW_COUNT)
                .map(|index| train::inverse::capture::View {
                    name: view_name(index),
                    camera: blade_volume::CameraParams::default(),
                    pixels: vec![[index as f32; 3]],
                    mask: Some(vec![1.0].into()),
                })
                .collect(),
        };
        let construction = construction_capture(&source);

        assert_eq!(construction.views.len(), 16);
        assert_eq!(construction.views[0].name, "view_02.png");
        assert_eq!(construction.views[15].name, "view_20.png");
        for &held in &train::inverse::diligent_mv::HELD_VIEW_INDICES {
            assert!(construction
                .views
                .iter()
                .all(|view| view.name != view_name(held)));
        }
    }
}
