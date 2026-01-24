mod ply;
mod radfoam_ply;
mod spz;

use std::{fs, io};

/// The kind of volumetric data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeKind {
    /// Gaussian splatting format (3DGS-style PLY or SPZ).
    Gaussian,
    /// RadFoam format (Voronoi cell-based).
    RadFoam,
}

/// Detected file format information.
#[derive(Debug, Clone)]
pub struct FormatInfo {
    pub kind: VolumeKind,
    pub path: String,
}

/// Detect the format of a PLY file by examining its header.
///
/// Returns `VolumeKind::RadFoam` if the file contains RadFoam-specific properties
/// like `density` and `adjacency_offset`, otherwise returns `VolumeKind::Gaussian`.
fn detect_ply_format(file_path: &str) -> VolumeKind {
    use io::BufRead as _;

    let file = match fs::File::open(file_path) {
        Ok(f) => f,
        Err(_) => return VolumeKind::Gaussian, // Default on error
    };
    let reader = io::BufReader::new(file);

    let mut has_density = false;
    let mut has_adjacency_offset = false;
    let mut has_adjacency_element = false;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();

        // Stop at end of header
        if trimmed == "end_header" {
            break;
        }

        let mut words = trimmed.split_whitespace();
        match words.next() {
            Some("property") => {
                // Skip type, get name
                let _ty = words.next();
                if let Some(name) = words.next() {
                    match name {
                        "density" => has_density = true,
                        "adjacency_offset" => has_adjacency_offset = true,
                        _ => {}
                    }
                }
            }
            Some("element") => {
                if let Some(name) = words.next() {
                    if name == "adjacency" {
                        has_adjacency_element = true;
                    }
                }
            }
            _ => {}
        }
    }

    // RadFoam PLY files have density, adjacency_offset, and an adjacency element
    if has_density && has_adjacency_offset && has_adjacency_element {
        VolumeKind::RadFoam
    } else {
        VolumeKind::Gaussian
    }
}

/// Detect the format of a file based on its extension and contents.
pub fn detect_format(file_path: &str) -> FormatInfo {
    let kind = if file_path.ends_with(".ply") {
        detect_ply_format(file_path)
    } else if file_path.ends_with(".spz") {
        VolumeKind::Gaussian
    } else {
        // Default to Gaussian for unknown formats
        VolumeKind::Gaussian
    };

    FormatInfo {
        kind,
        path: file_path.to_string(),
    }
}

/// Load volumetric data, automatically detecting the format.
///
/// For PLY files, examines the header to determine if it's a RadFoam or Gaussian format.
/// SPZ files are always loaded as Gaussian format.
pub fn load(file_path: &str) -> crate::PointCloudModel {
    let info = detect_format(file_path);
    match info.kind {
        VolumeKind::Gaussian => load_gaussian(file_path),
        VolumeKind::RadFoam => load_radfoam(file_path),
    }
}

/// Load a Gaussian splatting model from a PLY or SPZ file.
///
/// Returns a `PointCloudModel` with `transforms` set (rotation + scale).
pub fn load_gaussian(file_name: &str) -> crate::PointCloudModel {
    if file_name.ends_with(".ply") {
        ply::load(file_name)
    } else if file_name.ends_with(".spz") {
        spz::load(file_name)
    } else {
        panic!("Unsupported file name for Gaussian loader: {}", file_name);
    }
}

/// Load a RadFoam model from a PLY file.
///
/// Returns a `PointCloudModel` with `adjacency` set.
///
/// This expects the PLY format emitted by upstream `RadFoamScene.save_ply()`:
/// - a `vertex` element with `x,y,z`, `density`, `adjacency_offset`, and `color_sh_*`
/// - an `adjacency` element containing the flattened `uint32` neighbor indices.
pub fn load_radfoam(file_name: &str) -> crate::PointCloudModel {
    if file_name.ends_with(".ply") {
        radfoam_ply::load(file_name)
    } else {
        panic!("Unsupported file name for RadFoam loader: {}", file_name);
    }
}
