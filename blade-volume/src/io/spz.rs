use std::{fs, mem, slice};

const MAGIC: u32 = 0x5053474e;
const SCALE_LOG_SCALE: f32 = 1.0 / 16.0;
const SCALE_LOG_OFFSET: f32 = -10.0;
const ROT_SCALE: f32 = 1.0 / 127.5;
const COLOR_SCALE: f32 = 1.0 / 0.15;

#[repr(C)]
#[derive(Default, Debug)]
struct Header {
    pub magic: u32,
    pub version: u32,
    pub num_points: u32,
    pub sh_degree: u8,
    pub fractional_bits: u8,
    pub flags: u8,
    reserverd: u8,
}

fn inv_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

fn unpack_color(raw: u8) -> f32 {
    COLOR_SCALE * (raw as f32 / 255.0 - 0.5)
}

pub fn load(file_path: &str) -> crate::PointCloudModel {
    use std::io::Read as _;

    assert!(file_path.ends_with(".spz"));
    let spz_file = fs::File::open(file_path).unwrap();
    let mut gsz = flate2::read::GzDecoder::new(spz_file);
    let mut header = Header::default();
    gsz.read_exact(unsafe {
        slice::from_raw_parts_mut(&mut header as *mut _ as *mut u8, mem::size_of::<Header>())
    })
    .unwrap();
    log::info!("SPZ header: {:?}", header);
    assert_eq!(header.version, 2);
    assert_eq!(header.magic, MAGIC);

    let count = header.num_points as usize;
    let mut scratch = Vec::<u8>::new();

    let mut points = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut scales = Vec::with_capacity(count);

    // positions
    scratch.resize(count * 3 * 24 / 8, 0);
    gsz.read_exact(scratch.as_mut_slice()).unwrap();
    let pos_divisor = 1.0 / (1 << header.fractional_bits) as f32;
    for p3 in scratch.chunks(3 * 24 / 8) {
        let mut pos = [0.0; 3];
        for (p_c1, p) in pos.iter_mut().zip(p3.chunks(24 / 8)) {
            let pos_u = u32::from_le_bytes([p[0], p[1], p[2], 0]);
            let sign = if p[2] & 0x80 != 0 { -1.0 } else { 1.0 };
            *p_c1 = sign * pos_u as f32 * pos_divisor;
        }
        // Temporarily store position, opacity will be filled later
        points.push(glam::Vec4::new(pos[0], pos[1], pos[2], 0.0));
    }

    // scales
    scratch.resize(count * 3, 0);
    gsz.read_exact(scratch.as_mut_slice()).unwrap();
    for s3 in scratch.chunks(3) {
        let scale = glam::Vec3::new(s3[0] as f32, s3[1] as f32, s3[2] as f32) * SCALE_LOG_SCALE
            + SCALE_LOG_OFFSET;
        // Note: scale is still in log-space, exponentiate for use
        scales.push(scale.exp());
    }

    // rotations
    scratch.resize(count * 3, 0);
    gsz.read_exact(scratch.as_mut_slice()).unwrap();
    for r3 in scratch.chunks(3) {
        let r =
            glam::Vec3::new(r3[0] as i8 as f32, r3[1] as i8 as f32, r3[2] as i8 as f32) * ROT_SCALE;
        rotations.push(glam::Quat::from_xyzw(
            r.x,
            r.y,
            r.z,
            (1.0 - r.dot(r)).max(0.0).sqrt(),
        ));
    }

    // alphas (opacity)
    scratch.resize(count, 0);
    gsz.read_exact(scratch.as_mut_slice()).unwrap();
    for (point, a) in points.iter_mut().zip(scratch.iter()) {
        point.w = inv_sigmoid(*a as f32 / 255.0);
    }

    // colors -> DC SH coefficients
    scratch.resize(count * 3, 0);
    gsz.read_exact(scratch.as_mut_slice()).unwrap();
    let mut sh_coefficients = Vec::with_capacity(count * 3);
    for c3 in scratch.chunks(3) {
        // Convert packed color to SH DC coefficients
        sh_coefficients.push(unpack_color(c3[0]));
        sh_coefficients.push(unpack_color(c3[1]));
        sh_coefficients.push(unpack_color(c3[2]));
    }

    //TODO: read higher-order SH coefficients from SPZ if present

    crate::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree: 0, // Only DC for now
        transforms: Some(crate::Transforms { rotations, scales }),
        adjacency: None,
        radii: None,
    }
}
