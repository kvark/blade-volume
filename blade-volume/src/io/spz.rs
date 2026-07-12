use std::{fs, io};

const MAGIC: u32 = 0x5053_474e;
const HEADER_SIZE: usize = 16;
const FLAG_HAS_EXTENSIONS: u8 = 0x2;
const SCALE_LOG_SCALE: f32 = 1.0 / 16.0;
const SCALE_LOG_OFFSET: f32 = -10.0;
const COLOR_SCALE: f32 = 1.0 / 0.15;

#[derive(Debug)]
struct Header {
    magic: u32,
    version: u32,
    num_points: u32,
    sh_degree: u8,
    fractional_bits: u8,
    flags: u8,
}

impl Header {
    fn read(reader: &mut impl io::Read) -> Self {
        let mut bytes = [0u8; HEADER_SIZE];
        reader.read_exact(&mut bytes).unwrap();
        Self {
            magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            num_points: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            sh_degree: bytes[12],
            fractional_bits: bytes[13],
            flags: bytes[14],
        }
    }
}

fn read_bytes(reader: &mut impl io::Read, count: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; count];
    reader.read_exact(&mut bytes).unwrap();
    bytes
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

struct PackedStreams {
    positions: Vec<u8>,
    alphas: Vec<u8>,
    colors: Vec<u8>,
    scales: Vec<u8>,
    rotations: Vec<u8>,
    sh: Vec<u8>,
}

fn decode_packed(header: &Header, packed: PackedStreams) -> crate::PointCloudModel {
    let count = header.num_points as usize;
    let sh_degree = header.sh_degree as usize;
    let sh_component_count = crate::get_sh_component_count(sh_degree);
    let sh_rest_count = sh_component_count.saturating_sub(1) * 3;
    let rotation_bytes = if header.version >= 3 { 4 } else { 3 };
    let mut points = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut scales = Vec::with_capacity(count);
    let mut sh_coefficients = Vec::with_capacity(count * sh_component_count * 3);

    for (i, packed_alpha) in packed.alphas.iter().enumerate() {
        let position_base = i * 9;
        let x = unpack_fixed_24(
            &packed.positions[position_base..position_base + 3],
            header.fractional_bits,
        );
        let y = unpack_fixed_24(
            &packed.positions[position_base + 3..position_base + 6],
            header.fractional_bits,
        );
        let z = unpack_fixed_24(
            &packed.positions[position_base + 6..position_base + 9],
            header.fractional_bits,
        );
        points.push(glam::Vec4::new(x, y, z, *packed_alpha as f32 / 255.0));

        let scale_base = i * 3;
        let log_scale = glam::Vec3::new(
            packed.scales[scale_base] as f32,
            packed.scales[scale_base + 1] as f32,
            packed.scales[scale_base + 2] as f32,
        ) * SCALE_LOG_SCALE
            + SCALE_LOG_OFFSET;
        scales.push(log_scale.exp());

        let rotation_base = i * rotation_bytes;
        let rotation = if header.version >= 3 {
            unpack_quaternion_smallest_three(
                &packed.rotations[rotation_base..rotation_base + rotation_bytes],
            )
        } else {
            let xyz = glam::Vec3::new(
                packed.rotations[rotation_base] as f32 / 127.5 - 1.0,
                packed.rotations[rotation_base + 1] as f32 / 127.5 - 1.0,
                packed.rotations[rotation_base + 2] as f32 / 127.5 - 1.0,
            );
            glam::Quat::from_xyzw(
                xyz.x,
                xyz.y,
                xyz.z,
                (1.0 - xyz.length_squared()).max(0.0).sqrt(),
            )
        };
        rotations.push(rotation);

        let color_base = i * 3;
        sh_coefficients.push(unpack_color(packed.colors[color_base]));
        sh_coefficients.push(unpack_color(packed.colors[color_base + 1]));
        sh_coefficients.push(unpack_color(packed.colors[color_base + 2]));
        let sh_base = i * sh_rest_count;
        sh_coefficients.extend(
            packed.sh[sh_base..sh_base + sh_rest_count]
                .iter()
                .map(|&raw| unpack_sh(raw)),
        );
    }

    crate::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree,
        transforms: Some(crate::Transforms { rotations, scales }),
        adjacency: None,
        radii: None,
    }
}

/// Load a legacy gzip-compressed SPZ v2 or v3 cloud.
///
/// Values remain in SPZ's stored coordinate system (RUB unless an extension
/// says otherwise). Coordinate-system conversion is deliberately separate
/// from byte decoding so callers do not receive a silent, format-dependent
/// rotation.
pub fn load(file_path: &str) -> crate::PointCloudModel {
    assert!(file_path.ends_with(".spz"));
    let bytes = fs::read(file_path).unwrap();
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
        let mut decoded = Vec::new();
        io::Read::read_to_end(&mut decoder, &mut decoded).unwrap();
        load_legacy(&decoded)
    } else {
        load_v4(&bytes)
    }
}

fn load_legacy(bytes: &[u8]) -> crate::PointCloudModel {
    let mut reader = bytes;
    let header = Header::read(&mut reader);
    log::info!("SPZ header: {:?}", header);
    assert_eq!(header.magic, MAGIC, "invalid SPZ magic");
    assert!(
        matches!(header.version, 2 | 3),
        "only gzip SPZ versions 2 and 3 are supported"
    );
    assert!(
        header.sh_degree as usize <= crate::MAX_SH_DEGREE,
        "SPZ SH degree {} exceeds supported degree {}",
        header.sh_degree,
        crate::MAX_SH_DEGREE
    );
    assert!(header.fractional_bits < 31, "invalid fractional bit count");

    let count = header.num_points as usize;
    let sh_degree = header.sh_degree as usize;
    let sh_component_count = crate::get_sh_component_count(sh_degree);
    let sh_rest_count = sh_component_count.saturating_sub(1) * 3;
    let rotation_bytes = if header.version >= 3 { 4 } else { 3 };
    let expected_body = count
        .checked_mul(9 + 1 + 3 + 3 + rotation_bytes + sh_rest_count)
        .expect("SPZ decoded size overflow");
    assert!(expected_body <= reader.len(), "SPZ stream is truncated");

    // Legacy v2 stream order is positions, alphas, colors, scales,
    // rotations, then higher-order SH.
    let packed_positions = read_bytes(&mut reader, count * 9);
    let packed_alphas = read_bytes(&mut reader, count);
    let packed_colors = read_bytes(&mut reader, count * 3);
    let packed_scales = read_bytes(&mut reader, count * 3);
    let packed_rotations = read_bytes(&mut reader, count * rotation_bytes);
    let packed_sh = read_bytes(&mut reader, count * sh_rest_count);

    let trailing = reader;
    if header.flags & FLAG_HAS_EXTENSIONS == 0 {
        assert!(trailing.is_empty(), "unexpected trailing SPZ data");
    } else {
        log::warn!(
            "ignoring {} bytes of legacy SPZ extensions; coordinate metadata may be lost",
            trailing.len()
        );
    }

    decode_packed(
        &header,
        PackedStreams {
            positions: packed_positions,
            alphas: packed_alphas,
            colors: packed_colors,
            scales: packed_scales,
            rotations: packed_rotations,
            sh: packed_sh,
        },
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn load_v4(bytes: &[u8]) -> crate::PointCloudModel {
    use ruzstd::io::Read as _;

    assert!(bytes.len() >= 32, "SPZ v4 header is truncated");
    let mut header_reader = &bytes[..HEADER_SIZE];
    let header = Header::read(&mut header_reader);
    assert_eq!(header.magic, MAGIC, "invalid SPZ magic");
    assert_eq!(header.version, 4, "only stream SPZ version 4 is supported");
    assert!(
        header.sh_degree as usize <= crate::MAX_SH_DEGREE,
        "SPZ SH degree {} exceeds supported degree {}",
        header.sh_degree,
        crate::MAX_SH_DEGREE
    );
    assert!(header.fractional_bits < 31, "invalid fractional bit count");
    let count = header.num_points as usize;
    assert!(count > 0, "SPZ point count must be non-zero");
    assert!(
        header.num_points <= i32::MAX as u32
            && header.num_points as u64 <= (bytes.len() as u64).saturating_mul(1024) / 9,
        "SPZ point count is implausible for the file size"
    );
    let sh_rest_count =
        crate::get_sh_component_count(header.sh_degree as usize).saturating_sub(1) * 3;
    let expected_sizes = [
        count * 9,
        count,
        count * 3,
        count * 3,
        count * 4,
        count * sh_rest_count,
    ];
    let expected_streams: Vec<usize> = expected_sizes
        .into_iter()
        .filter(|&size| size > 0)
        .collect();
    let num_streams = bytes[15] as usize;
    assert_eq!(
        num_streams,
        expected_streams.len(),
        "unexpected SPZ v4 stream count"
    );
    let toc_offset = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let toc_end = toc_offset
        .checked_add(num_streams * 16)
        .expect("SPZ TOC size overflow");
    assert!(
        toc_offset >= 32 && toc_end <= bytes.len(),
        "invalid SPZ v4 TOC"
    );
    if header.flags & FLAG_HAS_EXTENSIONS != 0 {
        log::warn!(
            "ignoring {} bytes of SPZ v4 extensions; coordinate metadata may be lost",
            toc_offset - 32
        );
    }

    let mut compressed_offset = toc_end;
    let mut streams = Vec::with_capacity(num_streams);
    for (stream_index, &expected_size) in expected_streams.iter().enumerate() {
        let entry = toc_offset + stream_index * 16;
        let compressed_size = usize::try_from(read_u64(bytes, entry)).unwrap();
        let uncompressed_size = usize::try_from(read_u64(bytes, entry + 8)).unwrap();
        assert_eq!(
            uncompressed_size, expected_size,
            "SPZ v4 stream {stream_index} has an unexpected decoded size"
        );
        let compressed_end = compressed_offset
            .checked_add(compressed_size)
            .expect("SPZ stream size overflow");
        assert!(compressed_end <= bytes.len(), "SPZ v4 stream is truncated");
        let source = &bytes[compressed_offset..compressed_end];
        let mut decoder = ruzstd::decoding::StreamingDecoder::new(source).unwrap();
        let mut decoded = Vec::with_capacity(uncompressed_size);
        decoder.read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded.len(), uncompressed_size);
        streams.push(decoded);
        compressed_offset = compressed_end;
    }
    assert_eq!(
        compressed_offset,
        bytes.len(),
        "unexpected trailing SPZ v4 data"
    );

    let mut streams = streams.into_iter();
    decode_packed(
        &header,
        PackedStreams {
            positions: streams.next().unwrap(),
            alphas: streams.next().unwrap(),
            colors: streams.next().unwrap(),
            scales: streams.next().unwrap(),
            rotations: streams.next().unwrap(),
            sh: streams.next().unwrap_or_default(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn pack_fixed_24(value: i32) -> [u8; 3] {
        let bytes = value.to_le_bytes();
        [bytes[0], bytes[1], bytes[2]]
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

        let model = load(path.to_str().unwrap());
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

        let model = load(path.to_str().unwrap());
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

        let model = load(path.to_str().unwrap());
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
}
