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

/// Load a legacy gzip-compressed SPZ v2 or v3 cloud.
///
/// Values remain in SPZ's stored coordinate system (RUB unless an extension
/// says otherwise). Coordinate-system conversion is deliberately separate
/// from byte decoding so callers do not receive a silent, format-dependent
/// rotation.
pub fn load(file_path: &str) -> crate::PointCloudModel {
    use std::io::Read as _;

    assert!(file_path.ends_with(".spz"));
    let spz_file = fs::File::open(file_path).unwrap();
    let mut reader = flate2::read::GzDecoder::new(spz_file);
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

    // Legacy v2 stream order is positions, alphas, colors, scales,
    // rotations, then higher-order SH.
    let packed_positions = read_bytes(&mut reader, count * 9);
    let packed_alphas = read_bytes(&mut reader, count);
    let packed_colors = read_bytes(&mut reader, count * 3);
    let packed_scales = read_bytes(&mut reader, count * 3);
    let rotation_bytes = if header.version >= 3 { 4 } else { 3 };
    let packed_rotations = read_bytes(&mut reader, count * rotation_bytes);
    let packed_sh = read_bytes(&mut reader, count * sh_rest_count);

    let mut points = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut scales = Vec::with_capacity(count);
    let mut sh_coefficients = Vec::with_capacity(count * sh_component_count * 3);

    for (i, packed_alpha) in packed_alphas.iter().enumerate() {
        let position_base = i * 9;
        let x = unpack_fixed_24(
            &packed_positions[position_base..position_base + 3],
            header.fractional_bits,
        );
        let y = unpack_fixed_24(
            &packed_positions[position_base + 3..position_base + 6],
            header.fractional_bits,
        );
        let z = unpack_fixed_24(
            &packed_positions[position_base + 6..position_base + 9],
            header.fractional_bits,
        );
        let opacity = *packed_alpha as f32 / 255.0;
        points.push(glam::Vec4::new(x, y, z, opacity));

        let scale_base = i * 3;
        let log_scale = glam::Vec3::new(
            packed_scales[scale_base] as f32,
            packed_scales[scale_base + 1] as f32,
            packed_scales[scale_base + 2] as f32,
        ) * SCALE_LOG_SCALE
            + SCALE_LOG_OFFSET;
        scales.push(log_scale.exp());

        let rotation_base = i * rotation_bytes;
        let rotation = if header.version >= 3 {
            unpack_quaternion_smallest_three(
                &packed_rotations[rotation_base..rotation_base + rotation_bytes],
            )
        } else {
            let xyz = glam::Vec3::new(
                packed_rotations[rotation_base] as f32 / 127.5 - 1.0,
                packed_rotations[rotation_base + 1] as f32 / 127.5 - 1.0,
                packed_rotations[rotation_base + 2] as f32 / 127.5 - 1.0,
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
        sh_coefficients.push(unpack_color(packed_colors[color_base]));
        sh_coefficients.push(unpack_color(packed_colors[color_base + 1]));
        sh_coefficients.push(unpack_color(packed_colors[color_base + 2]));
        let sh_base = i * sh_rest_count;
        sh_coefficients.extend(
            packed_sh[sh_base..sh_base + sh_rest_count]
                .iter()
                .map(|&raw| unpack_sh(raw)),
        );
    }

    let mut trailing = Vec::new();
    reader.read_to_end(&mut trailing).unwrap();
    if header.flags & FLAG_HAS_EXTENSIONS == 0 {
        assert!(trailing.is_empty(), "unexpected trailing SPZ data");
    } else {
        log::warn!(
            "ignoring {} bytes of legacy SPZ extensions; coordinate metadata may be lost",
            trailing.len()
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
}
