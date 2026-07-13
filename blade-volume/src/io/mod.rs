mod ply;
mod radfoam_ply;
mod spz;

use std::{error, fmt, fs, io};

/// Asset-loading failure with preserved IO causes and explicit format errors.
#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    InvalidData(String),
    UnsupportedFormat(String),
}

impl LoadError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Io(ref source) => write!(formatter, "{source}"),
            Self::InvalidData(ref message) => write!(formatter, "invalid asset: {message}"),
            Self::UnsupportedFormat(ref message) => {
                write!(formatter, "unsupported format: {message}")
            }
        }
    }
}

impl error::Error for LoadError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Io(ref source) => Some(source),
            Self::InvalidData(_) | Self::UnsupportedFormat(_) => None,
        }
    }
}

impl From<io::Error> for LoadError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

/// The kind of volumetric data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeKind {
    /// Gaussian splatting format (3DGS-style PLY or SPZ).
    Gaussian,
    /// RadFoam format (Voronoi cell-based).
    RadFoam,
}

/// Coordinate conventions used by common Gaussian interchange files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianCoordinateSystem {
    /// Right, up, back: default SPZ storage / Three.js convention.
    Rub,
    /// Right, down, front: standard Gaussian PLY convention.
    Rdf,
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

/// Load a Gaussian and convert it to an explicit coordinate convention.
///
/// Legacy/default SPZ storage is RUB; Gaussian PLY is RDF. SPZ files carrying
/// a coordinate-system extension are currently decoded as stored and emit a
/// warning, so callers should not request conversion for such files until the
/// extension is exposed by this loader.
pub fn load_gaussian_with_coordinates(
    file_name: &str,
    target: GaussianCoordinateSystem,
) -> crate::PointCloudModel {
    let (mut model, source) = if file_name.ends_with(".ply") {
        (ply::load(file_name), GaussianCoordinateSystem::Rdf)
    } else if file_name.ends_with(".spz") {
        (spz::load(file_name), GaussianCoordinateSystem::Rub)
    } else {
        panic!("Unsupported file name for Gaussian loader: {file_name}");
    };
    if source != target {
        convert_rub_rdf(&mut model);
    }
    model
}

fn convert_rub_rdf(model: &mut crate::PointCloudModel) {
    // RUB ↔ RDF is a 180-degree rotation around X and is its own inverse.
    for point in model.points.iter_mut() {
        point.y = -point.y;
        point.z = -point.z;
    }
    if let Some(ref mut transforms) = model.transforms {
        for rotation in transforms.rotations.iter_mut() {
            *rotation = glam::Quat::from_xyzw(rotation.x, -rotation.y, -rotation.z, rotation.w);
        }
    }
    // Real-SH sign changes for the standard 3DGS basis, excluding DC.
    const SIGNS: [f32; 15] = [
        -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
    ];
    let components = model.sh_component_count();
    let stride = components * 3;
    for point_index in 0..model.points.len() {
        for component in 1..components {
            let sign = SIGNS[component - 1];
            let base = point_index * stride + component * 3;
            for channel in 0..3 {
                model.sh_coefficients[base + channel] *= sign;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rub_rdf_conversion_preserves_sh_function() {
        let mut model = crate::PointCloudModel {
            points: vec![glam::Vec4::new(1.0, 2.0, 3.0, 0.5)],
            sh_coefficients: (0..48).map(|index| index as f32 * 0.01 - 0.2).collect(),
            sh_degree: 3,
            transforms: Some(crate::Transforms {
                rotations: vec![glam::Quat::from_xyzw(0.1, 0.2, -0.3, 0.9).normalize()],
                scales: vec![glam::Vec3::new(1.0, 2.0, 3.0)],
            }),
            adjacency: None,
            radii: None,
        };
        let original = model.clone();
        let rub_direction = glam::Vec3::new(0.3, -0.4, 0.8).normalize();
        let rub_color = crate::trace::eval_rgb_sh(&model, 0, rub_direction);

        convert_rub_rdf(&mut model);
        assert_eq!(model.points[0].truncate(), glam::Vec3::new(1.0, -2.0, -3.0));
        let rdf_direction = glam::Vec3::new(rub_direction.x, -rub_direction.y, -rub_direction.z);
        let rdf_color = crate::trace::eval_rgb_sh(&model, 0, rdf_direction);
        assert!((rub_color - rdf_color).abs().max_element() < 1e-5);

        convert_rub_rdf(&mut model);
        assert_eq!(model.points, original.points);
        assert_eq!(model.sh_coefficients, original.sh_coefficients);
        let rotation = model.transforms.as_ref().unwrap().rotations[0];
        let original_rotation = original.transforms.as_ref().unwrap().rotations[0];
        assert!(rotation.dot(original_rotation).abs() > 1.0 - 1e-6);
    }
}
