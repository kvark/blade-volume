mod ply;
mod radfoam_ply;
mod spz;
mod surfel;

pub use surfel::{
    load as load_relight, load_environment, save as save_relight, try_load as try_load_relight,
    try_load_environment, try_save as try_save_relight, try_save_environment,
};

/// The extension a relightable surfel asset is written with.
pub const SURFEL_EXTENSION: &str = "surfel";

use std::{error, fmt, fs, io, path};

const MAX_DETECTION_HEADER_BYTES: usize = 1024 * 1024;

/// Asset-loading failure with preserved IO causes and explicit format errors.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be opened or read.
    Io(io::Error),
    /// The container or decoded model violates its declared format.
    InvalidData(String),
    /// The path does not select a supported asset format.
    UnsupportedFormat(String),
}

impl LoadError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self::UnsupportedFormat(message.into())
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
    /// Relightable surface-particle clouds, which carry materials rather than
    /// baked radiance. Loaded through [`try_load_relight`].
    ///
    /// [`try_load_relight`]: fn.try_load_relight.html
    Surfel,
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
fn detect_ply_format(file_path: &str) -> Result<VolumeKind, LoadError> {
    let file = fs::File::open(file_path)?;
    let mut reader = io::BufReader::new(file);

    let mut has_density = false;
    let mut has_adjacency_offset = false;
    let mut has_adjacency_element = false;
    let mut header_bytes = 0usize;
    let mut saw_magic = false;
    let mut saw_end = false;
    let mut line = String::new();

    loop {
        line.clear();
        let remaining = MAX_DETECTION_HEADER_BYTES.saturating_sub(header_bytes);
        let mut limited = io::Read::take(&mut reader, (remaining + 1) as u64);
        let bytes = io::BufRead::read_line(&mut limited, &mut line)?;
        if bytes == 0 {
            break;
        }
        header_bytes += bytes;
        if header_bytes > MAX_DETECTION_HEADER_BYTES {
            return Err(LoadError::invalid(format!(
                "PLY header exceeds {MAX_DETECTION_HEADER_BYTES} bytes"
            )));
        }
        let mut words = line.split_whitespace();
        match words.next() {
            Some("ply") if !saw_magic && header_bytes == bytes => saw_magic = true,
            Some("end_header") => {
                saw_end = true;
                break;
            }
            Some("property") => {
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
    if !saw_magic {
        return Err(LoadError::invalid("file is missing the PLY magic"));
    }
    if !saw_end {
        return Err(LoadError::invalid("PLY header has no end_header"));
    }
    if has_density && has_adjacency_offset && has_adjacency_element {
        Ok(VolumeKind::RadFoam)
    } else {
        Ok(VolumeKind::Gaussian)
    }
}

fn extension(file_path: &str) -> Option<&str> {
    path::Path::new(file_path)
        .extension()
        .and_then(|extension| extension.to_str())
}

fn has_extension(file_path: &str, expected: &str) -> bool {
    extension(file_path).is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

/// Detect the format of a file based on its extension and contents.
pub fn try_detect_format(file_path: &str) -> Result<FormatInfo, LoadError> {
    let kind = if has_extension(file_path, "ply") {
        detect_ply_format(file_path)?
    } else if has_extension(file_path, "spz") {
        VolumeKind::Gaussian
    } else if has_extension(file_path, SURFEL_EXTENSION) {
        VolumeKind::Surfel
    } else {
        return Err(LoadError::unsupported(format!(
            "'{file_path}' has no supported .ply, .spz or .{SURFEL_EXTENSION} extension"
        )));
    };
    Ok(FormatInfo {
        kind,
        path: file_path.to_string(),
    })
}

/// Detect a format, panicking on IO, malformed headers, or unsupported paths.
pub fn detect_format(file_path: &str) -> FormatInfo {
    try_detect_format(file_path)
        .unwrap_or_else(|error| panic!("failed to detect format for '{file_path}': {error}"))
}

/// Load volumetric data, automatically detecting the format.
///
/// For PLY files, examines the header to determine if it's a RadFoam or Gaussian format.
/// SPZ files are always loaded as Gaussian format.
pub fn try_load(file_path: &str) -> Result<crate::PointCloudModel, LoadError> {
    let info = try_detect_format(file_path)?;
    match info.kind {
        VolumeKind::Gaussian => try_load_gaussian(file_path),
        VolumeKind::RadFoam => try_load_radfoam(file_path),
        // Deliberately not a silent conversion: a surfel carries a material
        // and no radiance, so there is nothing to put in the SH slots that
        // would not be an invention.
        VolumeKind::Surfel => Err(LoadError::unsupported(format!(
            "'{file_path}' holds relightable surfels, which load through try_load_relight"
        ))),
    }
}

/// Load volumetric data, panicking if detection, IO, or validation fails.
pub fn load(file_path: &str) -> crate::PointCloudModel {
    try_load(file_path).unwrap_or_else(|error| panic!("failed to load '{file_path}': {error}"))
}

/// Load a Gaussian splatting model from a PLY or SPZ file.
///
/// Returns a `PointCloudModel` with `transforms` set (rotation + scale).
pub fn try_load_gaussian(file_name: &str) -> Result<crate::PointCloudModel, LoadError> {
    if has_extension(file_name, "ply") {
        ply::try_load(file_name)
    } else if has_extension(file_name, "spz") {
        spz::try_load(file_name)
    } else {
        Err(LoadError::unsupported(format!(
            "Gaussian loader does not support '{file_name}'"
        )))
    }
}

/// Load a Gaussian, panicking if IO or validation fails.
pub fn load_gaussian(file_name: &str) -> crate::PointCloudModel {
    try_load_gaussian(file_name)
        .unwrap_or_else(|error| panic!("failed to load Gaussian '{file_name}': {error}"))
}

/// Load a Gaussian and convert it to an explicit coordinate convention.
///
/// Legacy/default SPZ storage is RUB; Gaussian PLY is RDF. SPZ files carrying
/// a coordinate-system extension are currently decoded as stored and emit a
/// warning, so callers should not request conversion for such files until the
/// extension is exposed by this loader.
pub fn try_load_gaussian_with_coordinates(
    file_name: &str,
    target: GaussianCoordinateSystem,
) -> Result<crate::PointCloudModel, LoadError> {
    let (mut model, source) = if has_extension(file_name, "ply") {
        (ply::try_load(file_name)?, GaussianCoordinateSystem::Rdf)
    } else if has_extension(file_name, "spz") {
        (spz::try_load(file_name)?, GaussianCoordinateSystem::Rub)
    } else {
        return Err(LoadError::unsupported(format!(
            "Gaussian loader does not support '{file_name}'"
        )));
    };
    if source != target {
        convert_rub_rdf(&mut model);
    }
    Ok(model)
}

/// Load and convert a Gaussian, panicking if IO or validation fails.
pub fn load_gaussian_with_coordinates(
    file_name: &str,
    target: GaussianCoordinateSystem,
) -> crate::PointCloudModel {
    try_load_gaussian_with_coordinates(file_name, target).unwrap_or_else(|error| {
        panic!("failed to load Gaussian '{file_name}' with coordinates: {error}")
    })
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
pub fn try_load_radfoam(file_name: &str) -> Result<crate::PointCloudModel, LoadError> {
    if has_extension(file_name, "ply") {
        radfoam_ply::try_load(file_name)
    } else {
        Err(LoadError::unsupported(format!(
            "RadFoam loader does not support '{file_name}'"
        )))
    }
}

/// Load a RadFoam or PowerFoam cloud, panicking if IO or validation fails.
pub fn load_radfoam(file_name: &str) -> crate::PointCloudModel {
    try_load_radfoam(file_name)
        .unwrap_or_else(|error| panic!("failed to load RadFoam '{file_name}': {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str, extension: &str) -> path::PathBuf {
        std::env::temp_dir().join(format!(
            "blade-volume-format-{name}-{}-{:?}.{extension}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn result_api_reports_formats_and_failures() {
        let radfoam = format!(
            "{}/tests/data/radfoam_tiny_ascii.ply",
            env!("CARGO_MANIFEST_DIR")
        );
        assert_eq!(
            try_detect_format(&radfoam).unwrap().kind,
            VolumeKind::RadFoam
        );
        assert!(try_load(&radfoam).is_ok());

        let gaussian = path("gaussian", "PLY");
        fs::write(
            &gaussian,
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n",
        )
        .unwrap();
        assert_eq!(
            try_detect_format(gaussian.to_str().unwrap()).unwrap().kind,
            VolumeKind::Gaussian
        );
        fs::remove_file(gaussian).unwrap();

        assert!(matches!(
            try_detect_format("unsupported.cloud"),
            Err(LoadError::UnsupportedFormat(_))
        ));
        assert!(matches!(
            try_detect_format("missing.ply"),
            Err(LoadError::Io(_))
        ));
    }

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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
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
