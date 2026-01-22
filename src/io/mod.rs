mod ply;
mod radfoam_ply;
mod spz;

pub fn load(file_name: &str) -> crate::Model {
    if file_name.ends_with(".ply") {
        ply::load(file_name)
    } else if file_name.ends_with(".spz") {
        spz::load(file_name)
    } else {
        panic!("Unsupported file name: {}", file_name);
    }
}

/// Load an upstream Radiant Foam scene exported as a PLY file.
///
/// This expects the PLY format emitted by upstream `RadFoamScene.save_ply()`:
/// - a `vertex` element with `x,y,z`, `density`, `adjacency_offset`, and `color_sh_*`
/// - an `adjacency` element containing the flattened `uint32` neighbor indices.
///
/// Returns a `crate::RadFoamModel` containing points, packed attributes, and CSR adjacency.
pub fn load_radfoam_ply(file_name: &str) -> crate::RadFoamModel {
    if file_name.ends_with(".ply") {
        radfoam_ply::load(file_name)
    } else {
        panic!(
            "Unsupported file name for RadFoam PLY loader: {}",
            file_name
        );
    }
}
