//! A file format for relightable surfels.
//!
//! Conversion from glTF takes seconds and produces the same answer every time,
//! so anything that renders a converted asset more than once should not be
//! converting it. This is the format in between: a header and two flat tables,
//! laid out the way the GPU wants them, so loading is a read and a length
//! check rather than a decode.
//!
//! There is deliberately no light in it. What separates this representation
//! from the spherical-harmonic ones is that the environment is an argument
//! rather than part of the asset, and a format that stored one would give that
//! away. Environments load separately, through [`try_load_environment`].
//!
//! ```text
//! magic          8 bytes  "BVSURFEL"
//! version        u32      1
//! flags          u32      bit 0: Gaussian particle kernel
//! surfel_count   u32
//! material_count u32
//! surfels        32 bytes each: centre, radius, normal, material index
//! materials      32 bytes each: albedo, roughness, F0, padding
//! ```
//!
//! Everything is little endian, which is what every target this runs on is,
//! and is checked on load rather than assumed.
//!
//! [`try_load_environment`]: fn.try_load_environment.html

use super::LoadError;
use crate::relight;
use bytemuck::Zeroable as _;
use std::{fs, io, io::Read as _, io::Write as _, mem, path};

const MAGIC: &[u8; 8] = b"BVSURFEL";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 24;
const RECORD_BYTES: usize = 32;
const FLAG_GAUSSIAN: u32 = 1 << 0;
const KNOWN_FLAGS: u32 = FLAG_GAUSSIAN;

/// The record layout is the in-memory layout, which is what makes reading one
/// of these a copy rather than a decode. If a struct ever gains a field, the
/// version above has to move with it.
const _: () = assert!(mem::size_of::<relight::Surfel>() == RECORD_BYTES);
const _: () = assert!(mem::size_of::<relight::Material>() == RECORD_BYTES);

/// Put a slice of records into the byte order the file uses.
///
/// Both records are whole four-byte fields, floats and one index, so a byte
/// swap of each word converts between native and little endian in either
/// direction. On the targets this actually runs on it compiles to nothing.
fn to_file_order<T: bytemuck::Pod>(records: &mut [T]) {
    if cfg!(target_endian = "little") {
        return;
    }
    for word in bytemuck::cast_slice_mut::<T, u32>(records) {
        *word = word.swap_bytes();
    }
}

/// The largest table this will allocate for on trust.
///
/// A truncated or corrupt file can name any count it likes, and the count is
/// read before the data that would contradict it. Sixty-four million surfels
/// is two gigabytes, well past any asset here and well short of a count that
/// would take the process down before the length check catches it.
const MAX_RECORDS: u32 = 64 * 1024 * 1024;

fn read_u32(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes([
        bytes[index],
        bytes[index + 1],
        bytes[index + 2],
        bytes[index + 3],
    ])
}

fn read_f32(bytes: &[u8], index: usize) -> f32 {
    f32::from_bits(read_u32(bytes, index))
}

/// Serialise a relightable model.
///
/// The model is validated first: a file that names a material it does not
/// contain is worth rejecting here, where the asset it came from is still in
/// front of you, rather than at the point some renderer indexes past the end
/// of the table.
pub fn try_save(path: &path::Path, model: &relight::RelightModel) -> Result<(), LoadError> {
    model.validate().map_err(LoadError::invalid)?;
    if model.surfels.len() as u64 > MAX_RECORDS as u64
        || model.materials.len() as u64 > MAX_RECORDS as u64
    {
        return Err(LoadError::invalid(format!(
            "a model of {} surfels and {} materials exceeds the {MAX_RECORDS} the format holds",
            model.surfels.len(),
            model.materials.len()
        )));
    }

    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    let flags = match model.kernel {
        relight::ParticleKernel::Compact => 0,
        relight::ParticleKernel::Gaussian => FLAG_GAUSSIAN,
    };
    header.extend_from_slice(&flags.to_le_bytes());
    header.extend_from_slice(&(model.surfels.len() as u32).to_le_bytes());
    header.extend_from_slice(&(model.materials.len() as u32).to_le_bytes());

    // The tables go out as they sit in memory. Padding is zeroed first so two
    // saves of the same model are the same file rather than differing in the
    // slot nothing reads.
    let mut surfels = model.surfels.clone();
    let mut materials = model.materials.clone();
    for material in materials.iter_mut() {
        material._padding = 0.0;
    }
    to_file_order(&mut surfels);
    to_file_order(&mut materials);

    let mut file = io::BufWriter::new(fs::File::create(path)?);
    file.write_all(&header)?;
    file.write_all(bytemuck::cast_slice(&surfels))?;
    file.write_all(bytemuck::cast_slice(&materials))?;
    file.flush()?;
    Ok(())
}

/// Save a relightable model, panicking if it is invalid or cannot be written.
pub fn save(path: &path::Path, model: &relight::RelightModel) {
    try_save(path, model)
        .unwrap_or_else(|error| panic!("failed to save '{}': {error}", path.display()));
}

/// Read a relightable model back.
pub fn try_load(path: &path::Path) -> Result<relight::RelightModel, LoadError> {
    let mut file = fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length < HEADER_BYTES as u64 {
        return Err(LoadError::invalid(format!(
            "{} is {length} bytes, shorter than the {HEADER_BYTES} byte header",
            path.display()
        )));
    }
    let mut header = [0u8; HEADER_BYTES];
    file.read_exact(&mut header)?;
    if &header[..8] != MAGIC {
        return Err(LoadError::UnsupportedFormat(format!(
            "{} does not start with the surfel magic",
            path.display()
        )));
    }
    let version = read_u32(&header, 8);
    if version != VERSION {
        return Err(LoadError::invalid(format!(
            "{} is version {version}, and this reads version {VERSION}",
            path.display()
        )));
    }
    let flags = read_u32(&header, 12);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(LoadError::invalid(format!(
            "{} has unknown surfel flags {:#x}",
            path.display(),
            flags & !KNOWN_FLAGS
        )));
    }
    let kernel = if flags & FLAG_GAUSSIAN != 0 {
        relight::ParticleKernel::Gaussian
    } else {
        relight::ParticleKernel::Compact
    };
    let surfel_count = read_u32(&header, 16);
    let material_count = read_u32(&header, 20);
    if surfel_count > MAX_RECORDS || material_count > MAX_RECORDS {
        return Err(LoadError::invalid(format!(
            "{} claims {surfel_count} surfels and {material_count} materials",
            path.display()
        )));
    }
    // Checked against the file's real size before anything is allocated, so a
    // corrupt count cannot ask for memory the data could never fill.
    let expected =
        HEADER_BYTES as u64 + (surfel_count as u64 + material_count as u64) * RECORD_BYTES as u64;
    if length != expected {
        return Err(LoadError::invalid(format!(
            "{} is {length} bytes, and {surfel_count} surfels with {material_count} materials would be {expected}",
            path.display()
        )));
    }

    // Read straight into the tables. They are `Pod` and the record layout is
    // their layout, so this is one copy out of the file and no decoding at
    // all — which is the reason to have a format rather than reconverting.
    let mut surfels = vec![relight::Surfel::zeroed(); surfel_count as usize];
    let mut materials = vec![relight::Material::zeroed(); material_count as usize];
    file.read_exact(bytemuck::cast_slice_mut(&mut surfels))?;
    file.read_exact(bytemuck::cast_slice_mut(&mut materials))?;
    to_file_order(&mut surfels);
    to_file_order(&mut materials);

    let model = relight::RelightModel {
        kernel,
        surfels,
        materials,
    };
    model.validate().map_err(LoadError::invalid)?;
    Ok(model)
}

/// Load a relightable model, panicking if the file is missing or malformed.
pub fn load(path: &path::Path) -> relight::RelightModel {
    try_load(path).unwrap_or_else(|error| panic!("failed to load '{}': {error}", path.display()))
}

// ---------------------------------------------------------------- environments

/// Bytes per texel in a float environment plane: four channels of `f32`.
const PLANE_TEXEL_BYTES: usize = 16;

/// Read an equirectangular environment as linear radiance.
///
/// The plane format the reference data generator writes: four floats per texel
/// with the fourth ignored, no header, and dimensions recovered from the two to
/// one aspect every equirectangular map has. It carries no transfer curve,
/// which is the point — a sun a hundred times brighter than the sky around it
/// survives here and does not survive an eight-bit image, and that contrast is
/// most of what a specular highlight is made of.
pub fn try_load_environment(path: &path::Path) -> Result<relight::Environment, LoadError> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() || bytes.len() % PLANE_TEXEL_BYTES != 0 {
        return Err(LoadError::invalid(format!(
            "{} is {} bytes, which is not a whole number of RGBA float texels",
            path.display(),
            bytes.len()
        )));
    }
    let texel_count = bytes.len() / PLANE_TEXEL_BYTES;
    let height = ((texel_count / 2) as f64).sqrt().round() as usize;
    let width = height * 2;
    if width * height != texel_count {
        return Err(LoadError::invalid(format!(
            "{} holds {texel_count} texels, which is not a two to one map",
            path.display()
        )));
    }
    let mut texels = Vec::with_capacity(texel_count);
    for index in 0..texel_count {
        let base = index * PLANE_TEXEL_BYTES;
        texels.push([
            read_f32(&bytes, base),
            read_f32(&bytes, base + 4),
            read_f32(&bytes, base + 8),
        ]);
    }
    if texels
        .iter()
        .any(|texel| texel.iter().any(|value| !value.is_finite() || *value < 0.0))
    {
        return Err(LoadError::invalid(format!(
            "{} holds radiance that is negative or not finite",
            path.display()
        )));
    }
    Ok(relight::Environment {
        width,
        height,
        texels,
    })
}

/// Load an environment, panicking if it is missing or malformed.
pub fn load_environment(path: &path::Path) -> relight::Environment {
    try_load_environment(path)
        .unwrap_or_else(|error| panic!("failed to load environment '{}': {error}", path.display()))
}

/// Write an environment back out in the same plane format.
pub fn try_save_environment(
    path: &path::Path,
    environment: &relight::Environment,
) -> Result<(), LoadError> {
    if environment.texels.len() != environment.width * environment.height {
        return Err(LoadError::invalid(format!(
            "an environment of {}x{} cannot hold {} texels",
            environment.width,
            environment.height,
            environment.texels.len()
        )));
    }
    if environment.width != environment.height * 2 {
        return Err(LoadError::invalid(format!(
            "an equirectangular map is two to one, and this is {}x{}",
            environment.width, environment.height
        )));
    }
    let mut bytes = Vec::with_capacity(environment.texels.len() * PLANE_TEXEL_BYTES);
    for texel in &environment.texels {
        for channel in texel {
            bytes.extend_from_slice(&channel.to_le_bytes());
        }
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
    }
    let mut file = io::BufWriter::new(fs::File::create(path)?);
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary(name: &str) -> path::PathBuf {
        std::env::temp_dir().join(format!(
            "blade-volume-surfel-{name}-{}-{:?}.surfel",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn model() -> relight::RelightModel {
        relight::RelightModel {
            kernel: relight::ParticleKernel::Gaussian,
            surfels: vec![
                relight::Surfel {
                    center: [1.0, -2.0, 3.5],
                    radius: 0.125,
                    normal: [0.0, 1.0, 0.0],
                    material: 0,
                },
                relight::Surfel {
                    center: [-4.0, 0.5, 0.0],
                    radius: 2.0,
                    normal: [0.6, 0.0, 0.8],
                    material: 1,
                },
            ],
            materials: vec![
                relight::Material {
                    albedo: [0.1, 0.2, 0.3],
                    roughness: 0.25,
                    specular_f0: [0.04; 3],
                    _padding: 0.0,
                },
                relight::Material {
                    albedo: [0.0; 3],
                    roughness: 0.9,
                    specular_f0: [0.95, 0.8, 0.4],
                    _padding: 0.0,
                },
            ],
        }
    }

    #[test]
    fn a_round_trip_changes_nothing() {
        let path = temporary("round-trip");
        let original = model();
        try_save(&path, &original).unwrap();
        let loaded = try_load(&path).unwrap();
        fs::remove_file(&path).unwrap();

        // Exact rather than approximate: the format stores the bits it was
        // given, so anything less would be hiding a conversion.
        assert_eq!(loaded.surfels.len(), original.surfels.len());
        assert_eq!(loaded.kernel, original.kernel);
        for (a, b) in loaded.surfels.iter().zip(&original.surfels) {
            assert_eq!(a.center, b.center);
            assert_eq!(a.radius, b.radius);
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.material, b.material);
        }
        for (a, b) in loaded.materials.iter().zip(&original.materials) {
            assert_eq!(a.albedo, b.albedo);
            assert_eq!(a.roughness, b.roughness);
            assert_eq!(a.specular_f0, b.specular_f0);
        }
    }

    #[test]
    fn flag_zero_remains_the_legacy_compact_kernel() {
        let path = temporary("compact");
        let mut original = model();
        original.kernel = relight::ParticleKernel::Compact;
        try_save(&path, &original).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(read_u32(&bytes, 12), 0);
        let loaded = try_load(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(loaded.kernel, relight::ParticleKernel::Compact);
    }

    #[test]
    fn the_file_is_the_size_the_header_says() {
        let path = temporary("size");
        try_save(&path, &model()).unwrap();
        let length = fs::metadata(&path).unwrap().len() as usize;
        fs::remove_file(&path).unwrap();
        assert_eq!(length, HEADER_BYTES + 4 * RECORD_BYTES);
    }

    #[test]
    fn a_truncated_file_is_rejected_rather_than_read_short() {
        let path = temporary("truncated");
        try_save(&path, &model()).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - RECORD_BYTES);
        fs::write(&path, &bytes).unwrap();
        let error = try_load(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(
            matches!(error, LoadError::InvalidData(ref message) if message.contains("would be")),
            "{error}"
        );
    }

    #[test]
    fn a_foreign_file_is_reported_as_unsupported() {
        let path = temporary("foreign");
        fs::write(&path, b"ply\nformat binary_little_endian 1.0\n").unwrap();
        let error = try_load(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(matches!(error, LoadError::UnsupportedFormat(_)), "{error}");
    }

    #[test]
    fn a_future_version_is_refused_instead_of_misread() {
        let path = temporary("version");
        try_save(&path, &model()).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let error = try_load(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(
            matches!(error, LoadError::InvalidData(ref message) if message.contains("version")),
            "{error}"
        );
    }

    #[test]
    fn unknown_flags_are_refused_instead_of_ignored() {
        let path = temporary("flags");
        try_save(&path, &model()).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[12..16].copy_from_slice(&(1u32 << 31).to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let error = try_load(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(
            matches!(error, LoadError::InvalidData(ref message) if message.contains("flags")),
            "{error}"
        );
    }

    #[test]
    fn a_dangling_material_index_does_not_survive_a_round_trip() {
        // Validation runs on both sides, so a file that would index past the
        // material table cannot be written and is refused if it appears.
        let path = temporary("dangling");
        let mut broken = model();
        broken.surfels[0].material = 7;
        assert!(try_save(&path, &broken).is_err());

        try_save(&path, &model()).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let material_field = HEADER_BYTES + 28;
        bytes[material_field..material_field + 4].copy_from_slice(&7u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let error = try_load(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(
            matches!(error, LoadError::InvalidData(ref message) if message.contains("material")),
            "{error}"
        );
    }

    #[test]
    fn an_environment_round_trips_through_the_plane_format() {
        let path = std::env::temp_dir().join(format!(
            "blade-volume-env-{}-{:?}.f32",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut environment = relight::Environment::uniform([0.25, 0.5, 0.75], 8, 4);
        // A sun, which is the part an eight-bit map would lose.
        environment.texels[5] = [120.0, 118.0, 100.0];
        try_save_environment(&path, &environment).unwrap();
        let loaded = try_load_environment(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded.width, 8);
        assert_eq!(loaded.height, 4);
        assert_eq!(loaded.texels, environment.texels);
    }

    #[test]
    fn an_environment_that_is_not_two_to_one_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "blade-volume-env-odd-{}-{:?}.f32",
            std::process::id(),
            std::thread::current().id()
        ));
        // Three texels: no whole 2:1 map has that many.
        fs::write(&path, vec![0u8; 3 * PLANE_TEXEL_BYTES]).unwrap();
        let error = try_load_environment(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(
            matches!(error, LoadError::InvalidData(ref message) if message.contains("two to one")),
            "{error}"
        );
    }
}
