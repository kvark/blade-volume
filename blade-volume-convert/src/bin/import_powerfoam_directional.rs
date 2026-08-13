use blade_volume as vol;
use blade_volume_convert as convert;
use std::{env, path, process};

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    process::exit(1);
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        fail(
            "usage: import_powerfoam_directional <mean-direction.ply> \
             <directional.bin> <output.ply>",
        );
    }
    let mut model = vol::io::try_load_radfoam(&args[0])
        .unwrap_or_else(|error| fail(&format!("failed to load base PLY: {error}")));
    let (point_count, directional) = convert::load_powerfoam_directional(&args[1])
        .unwrap_or_else(|error| fail(&format!("failed to load directional table: {error}")));
    if point_count != model.len() {
        fail(&format!(
            "directional table has {point_count} points, base PLY has {}",
            model.len(),
        ));
    }
    let Some(ref mut detail) = model.surface_detail else {
        fail("base PLY has no spatial surface-detail table");
    };
    detail.directional = Some(directional);
    model
        .validate()
        .unwrap_or_else(|error| fail(&format!("combined model is invalid: {error}")));
    convert::save_ply_with_options(
        path::Path::new(&args[2]),
        &model,
        &convert::SaveOptions {
            format: convert::PlyFormat::Binary,
        },
    )
    .unwrap_or_else(|error| fail(&format!("failed to save combined PLY: {error:?}")));
    println!("wrote {} ({} points)", args[2], model.len());
}
