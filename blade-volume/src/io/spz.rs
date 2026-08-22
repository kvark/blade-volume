use super::LoadError;

use std::{fs, io};

const MAGIC: u32 = 0x5053_474e;
const HEADER_SIZE: usize = 16;
const V4_HEADER_SIZE: usize = 32;
const FLAG_HAS_EXTENSIONS: u8 = 0x2;
const MAX_COMPRESSION_RATIO: u64 = 1024;
const MIN_BYTES_PER_POINT: u64 = 9;
const SCALE_LOG_SCALE: f32 = 1.0 / 16.0;
const SCALE_LOG_OFFSET: f32 = -10.0;
const COLOR_SCALE: f32 = 1.0 / 0.15;

#[derive(Debug, PartialEq, Eq)]
struct Header {
    magic: u32,
    version: u32,
    num_points: u32,
    sh_degree: u8,
    fractional_bits: u8,
    flags: u8,
    reserved: u8,
}

impl Header {
    fn read(reader: &mut impl io::Read) -> Result<Self, LoadError> {
        let mut bytes = [0u8; HEADER_SIZE];
        read_exact(reader, &mut bytes, "header")?;
        Ok(Self {
            magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            num_points: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            sh_degree: bytes[12],
            fractional_bits: bytes[13],
            flags: bytes[14],
            reserved: bytes[15],
        })
    }
}

struct StreamSizes {
    positions: usize,
    alphas: usize,
    colors: usize,
    scales: usize,
    rotations: usize,
    sh: usize,
    total: usize,
}

impl StreamSizes {
    fn new(header: &Header) -> Result<Self, LoadError> {
        let count = header.num_points as usize;
        let sh_components = crate::get_sh_component_count(header.sh_degree as usize);
        let sh_rest = sh_components
            .checked_sub(1)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| LoadError::invalid("SPZ SH stream size overflow"))?;
        let rotation_bytes = if header.version >= 3 { 4 } else { 3 };
        let positions = checked_stream_size(count, 9, "position")?;
        let alphas = count;
        let colors = checked_stream_size(count, 3, "color")?;
        let scales = checked_stream_size(count, 3, "scale")?;
        let rotations = checked_stream_size(count, rotation_bytes, "rotation")?;
        let sh = checked_stream_size(count, sh_rest, "SH")?;
        let total = [positions, alphas, colors, scales, rotations, sh]
            .into_iter()
            .try_fold(0usize, |sum, size| sum.checked_add(size))
            .ok_or_else(|| LoadError::invalid("SPZ decoded body size overflow"))?;
        Ok(Self {
            positions,
            alphas,
            colors,
            scales,
            rotations,
            sh,
            total,
        })
    }

    fn streams(&self) -> [usize; 6] {
        [
            self.positions,
            self.alphas,
            self.colors,
            self.scales,
            self.rotations,
            self.sh,
        ]
    }
}

fn checked_stream_size(count: usize, stride: usize, name: &str) -> Result<usize, LoadError> {
    count
        .checked_mul(stride)
        .ok_or_else(|| LoadError::invalid(format!("SPZ {name} stream size overflow")))
}

fn read_exact(
    reader: &mut impl io::Read,
    destination: &mut [u8],
    name: &str,
) -> Result<(), LoadError> {
    io::Read::read_exact(reader, destination)
        .map_err(|error| LoadError::invalid(format!("SPZ {name} is truncated or corrupt: {error}")))
}

fn validate_header(
    header: &Header,
    versions: &[u32],
    file_len: Option<u64>,
) -> Result<(), LoadError> {
    if header.magic != MAGIC {
        return Err(LoadError::invalid("invalid SPZ magic"));
    }
    if !versions.contains(&header.version) {
        return Err(LoadError::invalid(format!(
            "unsupported SPZ version {}",
            header.version
        )));
    }
    if header.num_points == 0 || header.num_points > i32::MAX as u32 {
        return Err(LoadError::invalid(format!(
            "invalid SPZ point count {}",
            header.num_points
        )));
    }
    if let Some(file_len) = file_len {
        let plausible_points = file_len.saturating_mul(MAX_COMPRESSION_RATIO) / MIN_BYTES_PER_POINT;
        if u64::from(header.num_points) > plausible_points {
            return Err(LoadError::invalid(format!(
                "SPZ point count {} is implausible for a {file_len}-byte file",
                header.num_points
            )));
        }
    }
    if header.sh_degree as usize > crate::MAX_SH_DEGREE {
        return Err(LoadError::invalid(format!(
            "SPZ SH degree {} exceeds supported degree {}",
            header.sh_degree,
            crate::MAX_SH_DEGREE
        )));
    }
    if header.fractional_bits >= 31 {
        return Err(LoadError::invalid(format!(
            "invalid SPZ fractional bit count {}",
            header.fractional_bits
        )));
    }
    Ok(())
}

fn unpack_fixed_24(bytes: &[u8], fractional_bits: u8) -> f32 {
    let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
    let signed = ((raw << 8) as i32) >> 8;
    signed as f32 / (1u32 << fractional_bits) as f32
}

fn unpack_color(raw: u8) -> f32 {
    COLOR_SCALE * (raw as f32 / 255.0 - 0.5)
}

fn unpack_sh(raw: u8) -> f32 {
    (raw as f32 - 128.0) / 128.0
}

fn unpack_quaternion_smallest_three(bytes: &[u8]) -> glam::Quat {
    let mut packed = u32::from_le_bytes(bytes.try_into().unwrap());
    let largest = (packed >> 30) as usize;
    let mut components = [0.0_f32; 4];
    let mut sum_squares = 0.0;
    for index in (0..4).rev() {
        if index == largest {
            continue;
        }
        let magnitude = packed & 0x1ff;
        let negative = (packed >> 9) & 1 != 0;
        packed >>= 10;
        let value = std::f32::consts::FRAC_1_SQRT_2 * magnitude as f32 / 511.0;
        components[index] = if negative { -value } else { value };
        sum_squares += value * value;
    }
    components[largest] = (1.0 - sum_squares).max(0.0).sqrt();
    glam::Quat::from_xyzw(components[0], components[1], components[2], components[3])
}

fn allocate_vec<T: Clone>(count: usize, value: T, name: &str) -> Result<Vec<T>, LoadError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|error| LoadError::invalid(format!("SPZ {name} allocation failed: {error}")))?;
    output.resize(count, value);
    Ok(output)
}

fn allocate_model(header: &Header) -> Result<crate::PointCloudModel, LoadError> {
    let count = header.num_points as usize;
    let sh_degree = header.sh_degree as usize;
    let sh_component_count = crate::get_sh_component_count(sh_degree);
    let sh_value_count = count
        .checked_mul(sh_component_count)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| LoadError::invalid("SPZ SH model allocation size overflow"))?;
    Ok(crate::PointCloudModel {
        points: allocate_vec(count, glam::Vec4::ZERO, "point")?,
        sh_coefficients: allocate_vec(sh_value_count, 0.0, "SH")?,
        sh_degree,
        transforms: Some(crate::Transforms {
            rotations: allocate_vec(count, glam::Quat::IDENTITY, "rotation")?,
            scales: allocate_vec(count, glam::Vec3::ONE, "scale")?,
            pbr: None,
        }),
        adjacency: None,
        radii: None,
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    })
}

fn decode_positions(
    reader: &mut impl io::Read,
    header: &Header,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    let mut packed = [0u8; 9];
    for point in model.points.iter_mut() {
        read_exact(reader, &mut packed, "position stream")?;
        let x = unpack_fixed_24(&packed[0..3], header.fractional_bits);
        let y = unpack_fixed_24(&packed[3..6], header.fractional_bits);
        let z = unpack_fixed_24(&packed[6..9], header.fractional_bits);
        *point = glam::Vec4::new(x, y, z, 0.0);
    }
    Ok(())
}

fn decode_alphas(
    reader: &mut impl io::Read,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    let mut packed = [0u8; 1];
    for point in model.points.iter_mut() {
        read_exact(reader, &mut packed, "alpha stream")?;
        point.w = packed[0] as f32 / 255.0;
    }
    Ok(())
}

fn decode_colors(
    reader: &mut impl io::Read,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    let stride = model.sh_component_count() * 3;
    let mut packed = [0u8; 3];
    for coefficients in model.sh_coefficients.chunks_exact_mut(stride) {
        read_exact(reader, &mut packed, "color stream")?;
        coefficients[0] = unpack_color(packed[0]);
        coefficients[1] = unpack_color(packed[1]);
        coefficients[2] = unpack_color(packed[2]);
    }
    Ok(())
}

fn decode_scales(
    reader: &mut impl io::Read,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    let mut packed = [0u8; 3];
    let transforms = model.transforms.as_mut().unwrap();
    for scale in transforms.scales.iter_mut() {
        read_exact(reader, &mut packed, "scale stream")?;
        let log_scale = glam::Vec3::new(packed[0] as f32, packed[1] as f32, packed[2] as f32)
            * SCALE_LOG_SCALE
            + SCALE_LOG_OFFSET;
        *scale = log_scale.exp();
    }
    Ok(())
}

fn decode_rotations(
    reader: &mut impl io::Read,
    header: &Header,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    let transforms = model.transforms.as_mut().unwrap();
    let mut packed = [0u8; 4];
    for output in transforms.rotations.iter_mut() {
        let packed = if header.version >= 3 {
            read_exact(reader, &mut packed, "rotation stream")?;
            &packed[..4]
        } else {
            read_exact(reader, &mut packed[..3], "rotation stream")?;
            &packed[..3]
        };
        let rotation = if header.version >= 3 {
            unpack_quaternion_smallest_three(packed)
        } else {
            let xyz = glam::Vec3::new(
                packed[0] as f32 / 127.5 - 1.0,
                packed[1] as f32 / 127.5 - 1.0,
                packed[2] as f32 / 127.5 - 1.0,
            );
            glam::Quat::from_xyzw(
                xyz.x,
                xyz.y,
                xyz.z,
                (1.0 - xyz.length_squared()).max(0.0).sqrt(),
            )
        };
        *output = rotation;
    }
    Ok(())
}

fn decode_sh(
    reader: &mut impl io::Read,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    let stride = model.sh_component_count() * 3;
    let rest = stride - 3;
    let mut packed = vec![0u8; rest];
    for coefficients in model.sh_coefficients.chunks_exact_mut(stride) {
        read_exact(reader, &mut packed, "SH stream")?;
        for (output, &raw) in coefficients[3..].iter_mut().zip(packed.iter()) {
            *output = unpack_sh(raw);
        }
    }
    Ok(())
}

fn decode_stream(
    index: usize,
    reader: &mut impl io::Read,
    header: &Header,
    model: &mut crate::PointCloudModel,
) -> Result<(), LoadError> {
    match index {
        0 => decode_positions(reader, header, model),
        1 => decode_alphas(reader, model),
        2 => decode_colors(reader, model),
        3 => decode_scales(reader, model),
        4 => decode_rotations(reader, header, model),
        5 => decode_sh(reader, model),
        _ => Err(LoadError::invalid(format!(
            "unexpected SPZ stream index {index}"
        ))),
    }
}

fn finish_model(model: crate::PointCloudModel) -> Result<crate::PointCloudModel, LoadError> {
    model
        .validate()
        .map_err(|error| LoadError::invalid(format!("SPZ model: {error}")))?;
    Ok(model)
}

/// Load a legacy gzip-compressed SPZ v2 or v3 cloud.
///
/// Values remain in SPZ's stored coordinate system (RUB unless an extension
/// says otherwise). Coordinate-system conversion is deliberately separate
/// from byte decoding so callers do not receive a silent, format-dependent
/// rotation.
pub fn try_load(file_path: &str) -> Result<crate::PointCloudModel, LoadError> {
    let mut file = fs::File::open(file_path)?;
    let file_len = file.metadata()?.len();
    let mut prefix = [0u8; 4];
    let prefix_len = io::Read::read(&mut file, &mut prefix)?;
    io::Seek::seek(&mut file, io::SeekFrom::Start(0))?;
    if prefix_len >= 2 && prefix[..2] == [0x1f, 0x8b] {
        load_legacy(file)
    } else if prefix_len == prefix.len() && u32::from_le_bytes(prefix) == MAGIC {
        load_v4(file, file_len)
    } else {
        Err(LoadError::invalid("unrecognized SPZ container"))
    }
}

fn load_legacy(file: fs::File) -> Result<crate::PointCloudModel, LoadError> {
    let source = io::BufReader::new(file);
    let mut decoder = flate2::read::GzDecoder::new(source);
    let header = Header::read(&mut decoder)?;
    log::info!("SPZ header: {:?}", header);
    validate_header(&header, &[2, 3], None)?;
    if header.reserved != 0 {
        return Err(LoadError::invalid(
            "legacy SPZ reserved header byte is non-zero",
        ));
    }
    let sizes = StreamSizes::new(&header)?;

    // Validate the complete decompressed body without allocating from its
    // header-controlled point count. Rewind the same file for the real decode.
    let expected_body = u64::try_from(sizes.total)
        .map_err(|_| LoadError::invalid("SPZ decoded body exceeds stream limits"))?;
    let decoded_body = {
        let mut body = io::Read::take(&mut decoder, expected_body);
        io::copy(&mut body, &mut io::sink())
            .map_err(|error| LoadError::invalid(format!("SPZ gzip body is corrupt: {error}")))?
    };
    if decoded_body != expected_body {
        return Err(LoadError::invalid(format!(
            "SPZ gzip body is {decoded_body} bytes, expected {expected_body}"
        )));
    }
    if header.flags & FLAG_HAS_EXTENSIONS == 0 {
        let mut trailing = [0u8; 1];
        if io::Read::read(&mut decoder, &mut trailing)
            .map_err(|error| LoadError::invalid(format!("SPZ gzip trailer is corrupt: {error}")))?
            != 0
        {
            return Err(LoadError::invalid("unexpected trailing legacy SPZ data"));
        }
    } else {
        let extension_bytes = io::copy(&mut decoder, &mut io::sink()).map_err(|error| {
            LoadError::invalid(format!("SPZ gzip extensions are corrupt: {error}"))
        })?;
        log::warn!(
            "ignoring {} bytes of legacy SPZ extensions; coordinate metadata may be lost",
            extension_bytes
        );
    }

    let mut source = decoder.into_inner();
    io::Seek::seek(&mut source, io::SeekFrom::Start(0))?;
    let mut decoder = flate2::read::GzDecoder::new(source);
    let repeated_header = Header::read(&mut decoder)?;
    if repeated_header != header {
        return Err(LoadError::invalid("SPZ header changed while loading"));
    }
    let mut model = allocate_model(&header)?;
    for (index, size) in sizes.streams().into_iter().enumerate() {
        if size != 0 {
            decode_stream(index, &mut decoder, &header, &mut model)?;
        }
    }
    finish_model(model)
}

fn read_u64(reader: &mut impl io::Read, name: &str) -> Result<u64, LoadError> {
    let mut bytes = [0u8; 8];
    read_exact(reader, &mut bytes, name)?;
    Ok(u64::from_le_bytes(bytes))
}

struct StreamInfo {
    compressed_size: u64,
    uncompressed_size: usize,
}

fn load_v4(file: fs::File, file_len: u64) -> Result<crate::PointCloudModel, LoadError> {
    let mut file = io::BufReader::new(file);
    let mut header_bytes = [0u8; V4_HEADER_SIZE];
    read_exact(&mut file, &mut header_bytes, "v4 header")?;
    let mut header_reader = &header_bytes[..HEADER_SIZE];
    let header = Header::read(&mut header_reader)?;
    validate_header(&header, &[4], Some(file_len))?;
    if header_bytes[20..].iter().any(|&byte| byte != 0) {
        return Err(LoadError::invalid(
            "SPZ v4 reserved header bytes are non-zero",
        ));
    }
    let sizes = StreamSizes::new(&header)?;
    let expected_sizes = sizes.streams();
    let expected_streams: Vec<usize> = expected_sizes
        .into_iter()
        .filter(|&size| size > 0)
        .collect();
    let num_streams = header.reserved as usize;
    if num_streams != expected_streams.len() {
        return Err(LoadError::invalid(format!(
            "SPZ v4 has {num_streams} streams, expected {}",
            expected_streams.len()
        )));
    }
    let toc_offset = u64::from(u32::from_le_bytes(header_bytes[16..20].try_into().unwrap()));
    let toc_size = u64::try_from(num_streams)
        .ok()
        .and_then(|value| value.checked_mul(16))
        .ok_or_else(|| LoadError::invalid("SPZ v4 TOC size overflow"))?;
    let toc_end = toc_offset
        .checked_add(toc_size)
        .ok_or_else(|| LoadError::invalid("SPZ v4 TOC end overflow"))?;
    if toc_offset < V4_HEADER_SIZE as u64 || toc_end > file_len {
        return Err(LoadError::invalid("invalid SPZ v4 TOC range"));
    }
    if header.flags & FLAG_HAS_EXTENSIONS == 0 && toc_offset != V4_HEADER_SIZE as u64 {
        return Err(LoadError::invalid(
            "SPZ v4 TOC is displaced without an extension flag",
        ));
    }
    if header.flags & FLAG_HAS_EXTENSIONS != 0 {
        log::warn!(
            "ignoring {} bytes of SPZ v4 extensions; coordinate metadata may be lost",
            toc_offset - V4_HEADER_SIZE as u64
        );
    }

    io::Seek::seek(&mut file, io::SeekFrom::Start(toc_offset))?;
    let mut compressed_end = toc_end;
    let mut streams = Vec::new();
    streams
        .try_reserve_exact(num_streams)
        .map_err(|error| LoadError::invalid(format!("SPZ TOC allocation failed: {error}")))?;
    for (stream_index, &expected_size) in expected_streams.iter().enumerate() {
        let compressed_size = read_u64(&mut file, "v4 TOC compressed size")?;
        let uncompressed_size_u64 = read_u64(&mut file, "v4 TOC decoded size")?;
        let uncompressed_size = usize::try_from(uncompressed_size_u64).map_err(|_| {
            LoadError::invalid(format!(
                "SPZ v4 stream {stream_index} decoded size exceeds platform limits"
            ))
        })?;
        if uncompressed_size != expected_size {
            return Err(LoadError::invalid(format!(
                "SPZ v4 stream {stream_index} decodes to {uncompressed_size} bytes, expected {expected_size}"
            )));
        }
        compressed_end = compressed_end
            .checked_add(compressed_size)
            .ok_or_else(|| LoadError::invalid("SPZ v4 compressed stream range overflow"))?;
        if compressed_end > file_len {
            return Err(LoadError::invalid(format!(
                "SPZ v4 stream {stream_index} extends past the file"
            )));
        }
        streams.push(StreamInfo {
            compressed_size,
            uncompressed_size,
        });
    }
    if compressed_end != file_len {
        return Err(LoadError::invalid("unexpected trailing SPZ v4 data"));
    }

    io::Seek::seek(&mut file, io::SeekFrom::Start(toc_end))?;
    let mut model = allocate_model(&header)?;
    for (stream_index, stream) in streams.into_iter().enumerate() {
        let source = io::Read::take(&mut file, stream.compressed_size);
        let mut decoder = ruzstd::decoding::StreamingDecoder::new(source).map_err(|error| {
            LoadError::invalid(format!(
                "SPZ v4 stream {stream_index} has an invalid zstd header: {error}"
            ))
        })?;
        decode_stream(stream_index, &mut decoder, &header, &mut model)?;
        let mut extra = [0u8; 1];
        if io::Read::read(&mut decoder, &mut extra).map_err(|error| {
            LoadError::invalid(format!("SPZ v4 stream {stream_index} is corrupt: {error}"))
        })? != 0
        {
            return Err(LoadError::invalid(format!(
                "SPZ v4 stream {stream_index} exceeds its decoded size {}",
                stream.uncompressed_size
            )));
        }
        let source = decoder.into_inner();
        if source.limit() != 0 {
            return Err(LoadError::invalid(format!(
                "SPZ v4 stream {stream_index} has {} unused compressed bytes",
                source.limit()
            )));
        }
    }
    finish_model(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn pack_fixed_24(value: i32) -> [u8; 3] {
        let bytes = value.to_le_bytes();
        [bytes[0], bytes[1], bytes[2]]
    }

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "blade-volume-spz-{name}-{}-{:?}.spz",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn loads_v2_stream_order_signed_positions_opacity_and_sh() {
        let path =
            std::env::temp_dir().join(format!("blade-volume-spz-v2-{}.spz", std::process::id()));
        let file = fs::File::create(&path).unwrap();
        let mut writer = flate2::write::GzEncoder::new(file, flate2::Compression::default());

        writer.write_all(&MAGIC.to_le_bytes()).unwrap();
        writer.write_all(&2u32.to_le_bytes()).unwrap();
        writer.write_all(&1u32.to_le_bytes()).unwrap();
        writer.write_all(&[1, 8, 0, 0]).unwrap();
        for fixed in [384, -512, 64] {
            writer.write_all(&pack_fixed_24(fixed)).unwrap();
        }
        writer.write_all(&[204]).unwrap();
        writer.write_all(&[128, 64, 255]).unwrap();
        writer.write_all(&[160, 144, 176]).unwrap();
        writer.write_all(&[128, 128, 128]).unwrap();
        writer
            .write_all(&[128, 192, 64, 255, 0, 160, 96, 144, 112])
            .unwrap();
        writer.finish().unwrap();

        let model = try_load(path.to_str().unwrap()).unwrap();
        assert_eq!(model.sh_degree, 1);
        assert_eq!(model.sh_coefficients.len(), 12);
        assert_eq!(model.points[0].truncate(), glam::Vec3::new(1.5, -2.0, 0.25));
        assert!((model.points[0].w - 0.8).abs() < 1e-6);
        assert_eq!(&model.sh_coefficients[3..6], &[0.0, 0.5, -0.5]);
        let transforms = model.transforms.unwrap();
        assert!((transforms.scales[0].x - 1.0).abs() < 1e-6);
        assert!((transforms.scales[0].y - (-1.0_f32).exp()).abs() < 1e-6);
        assert!((transforms.scales[0].z - 1.0_f32.exp()).abs() < 1e-6);
        assert!(transforms.rotations[0].is_finite());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_v3_smallest_three_quaternion_stream() {
        let path =
            std::env::temp_dir().join(format!("blade-volume-spz-v3-{}.spz", std::process::id()));
        let file = fs::File::create(&path).unwrap();
        let mut writer = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        writer.write_all(&MAGIC.to_le_bytes()).unwrap();
        writer.write_all(&3u32.to_le_bytes()).unwrap();
        writer.write_all(&1u32.to_le_bytes()).unwrap();
        writer.write_all(&[0, 8, 0, 0]).unwrap();
        for fixed in [256, 512, -256] {
            writer.write_all(&pack_fixed_24(fixed)).unwrap();
        }
        writer.write_all(&[255]).unwrap();
        writer.write_all(&[128, 128, 128]).unwrap();
        writer.write_all(&[160, 160, 160]).unwrap();
        // Largest component is w (index 3); all stored components are zero.
        writer.write_all(&0xC000_0000_u32.to_le_bytes()).unwrap();
        writer.finish().unwrap();

        let model = try_load(path.to_str().unwrap()).unwrap();
        assert_eq!(model.points[0].truncate(), glam::Vec3::new(1.0, 2.0, -1.0));
        let rotation = model.transforms.unwrap().rotations[0];
        assert!(rotation.dot(glam::Quat::IDENTITY).abs() > 1.0 - 1e-6);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn smallest_three_decoder_reconstructs_largest_component() {
        let rotation = unpack_quaternion_smallest_three(&0_u32.to_le_bytes());
        assert!(
            rotation
                .dot(glam::Quat::from_xyzw(1.0, 0.0, 0.0, 0.0))
                .abs()
                > 1.0 - 1e-6
        );
    }

    #[test]
    fn loads_v4_independent_zstd_streams() {
        let mut positions = Vec::new();
        for fixed in [-256, 128, 768] {
            positions.extend_from_slice(&pack_fixed_24(fixed));
        }
        let streams = [
            positions,
            vec![128],
            vec![128, 64, 255],
            vec![160, 160, 160],
            0xC000_0000_u32.to_le_bytes().to_vec(),
        ];
        let chunks: Vec<Vec<u8>> = streams
            .iter()
            .map(|stream| {
                ruzstd::encoding::compress_to_vec(
                    stream.as_slice(),
                    ruzstd::encoding::CompressionLevel::Fastest,
                )
            })
            .collect();
        let mut file = vec![0_u8; 32 + streams.len() * 16];
        file[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        file[4..8].copy_from_slice(&4_u32.to_le_bytes());
        file[8..12].copy_from_slice(&1_u32.to_le_bytes());
        file[13] = 8;
        file[15] = streams.len() as u8;
        file[16..20].copy_from_slice(&32_u32.to_le_bytes());
        for (index, pair) in streams.iter().zip(chunks.iter()).enumerate() {
            let entry = 32 + index * 16;
            file[entry..entry + 8].copy_from_slice(&(pair.1.len() as u64).to_le_bytes());
            file[entry + 8..entry + 16].copy_from_slice(&(pair.0.len() as u64).to_le_bytes());
        }
        for chunk in chunks {
            file.extend_from_slice(&chunk);
        }
        let path =
            std::env::temp_dir().join(format!("blade-volume-spz-v4-{}.spz", std::process::id()));
        fs::write(&path, file).unwrap();

        let model = try_load(path.to_str().unwrap()).unwrap();
        assert_eq!(model.points[0].truncate(), glam::Vec3::new(-1.0, 0.5, 3.0));
        assert!((model.points[0].w - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(model.sh_coefficients.len(), 3);
        assert!(
            model.transforms.unwrap().rotations[0]
                .dot(glam::Quat::IDENTITY)
                .abs()
                > 1.0 - 1e-6
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn truncated_legacy_body_and_absurd_count_return_errors() {
        let truncated = path("truncated-v2");
        let file = fs::File::create(&truncated).unwrap();
        let mut writer = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        writer.write_all(&MAGIC.to_le_bytes()).unwrap();
        writer.write_all(&2u32.to_le_bytes()).unwrap();
        writer.write_all(&1u32.to_le_bytes()).unwrap();
        writer.write_all(&[0, 8, 0, 0]).unwrap();
        writer.finish().unwrap();
        assert!(matches!(
            try_load(truncated.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(truncated).unwrap();

        let absurd = path("absurd-count");
        let file = fs::File::create(&absurd).unwrap();
        let mut writer = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        writer.write_all(&MAGIC.to_le_bytes()).unwrap();
        writer.write_all(&2u32.to_le_bytes()).unwrap();
        writer.write_all(&u32::MAX.to_le_bytes()).unwrap();
        writer.write_all(&[0, 8, 0, 0]).unwrap();
        writer.finish().unwrap();
        assert!(matches!(
            try_load(absurd.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(absurd).unwrap();
    }

    #[test]
    fn malformed_v4_toc_and_unknown_container_return_errors() {
        let malformed = path("malformed-v4-toc");
        let mut bytes = [0u8; V4_HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&4u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[13] = 8;
        bytes[15] = 5;
        bytes[16..20].copy_from_slice(&(V4_HEADER_SIZE as u32).to_le_bytes());
        fs::write(&malformed, bytes).unwrap();
        assert!(matches!(
            try_load(malformed.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(malformed).unwrap();

        let unknown = path("unknown");
        fs::write(&unknown, b"not an SPZ file").unwrap();
        assert!(matches!(
            try_load(unknown.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(unknown).unwrap();
    }
}
