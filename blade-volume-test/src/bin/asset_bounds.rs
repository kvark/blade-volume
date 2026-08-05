//! Print the world-space bounds of a converted asset.
//!
//! The reference generator on blade's side cannot know how big an asset is, so
//! it takes the framing as an argument. This is where that argument comes from,
//! and it uses the same conversion the renderer will, so both frame the same
//! thing.
//!
//! Usage:
//! ```text
//!   asset_bounds <file.gltf>
//! ```

fn main() {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: asset_bounds <file.gltf>");
            std::process::exit(2);
        }
    };
    let options = blade_volume_convert::ConvertOptions {
        // Coarse: the bounds do not depend on how densely it is sampled, and a
        // low resolution keeps this quick.
        resolution: Some(64.0),
        ..Default::default()
    };
    let model = match blade_volume_convert::relight_model_from_gltf(
        std::path::Path::new(&path),
        &options,
    ) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("cannot convert {path}: {e:?}");
            std::process::exit(1);
        }
    };
    let (min, max) = model.bounds().expect("the model has no surfels");
    let center = 0.5 * (min + max);
    println!("RELIGHT_TARGET={},{},{}", center.x, center.y, center.z);
    println!("RELIGHT_RADIUS={}", 0.5 * (max - min).length());
}
