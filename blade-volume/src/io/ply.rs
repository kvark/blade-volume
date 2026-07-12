use std::{f32, fs, io, mem};

#[derive(Default)]
struct Offsets {
    mean: usize,
    rot: usize,
    scale: usize,
    opacity: usize,
    f_dc: usize,
    f_rest: usize,
}

fn read_f32(data: &[u8], offset: usize) -> f32 {
    let bytes: [u8; mem::size_of::<f32>()] = data[offset..offset + mem::size_of::<f32>()]
        .try_into()
        .unwrap();
    f32::from_le_bytes(bytes)
}

fn read_array<const N: usize>(data: &[u8], offset: usize) -> [f32; N] {
    std::array::from_fn(|i| read_f32(data, offset + i * mem::size_of::<f32>()))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn load(file_path: &str) -> crate::PointCloudModel {
    use std::io::{BufRead as _, Read as _};

    let mut count = 0;
    let mut stride = 0;
    let mut offsets = Offsets::default();
    let mut sh_rest_count = 0usize;

    assert!(file_path.ends_with(".ply"));
    let mut file = io::BufReader::new(fs::File::open(file_path).unwrap());
    let mut line = String::new();
    while let Ok(_) = file.read_line(&mut line) {
        let mut words = line.split_whitespace();
        match words.next().unwrap() {
            "ply" => {}
            "format" => {
                assert_eq!(words.next().unwrap(), "binary_little_endian");
                assert_eq!(words.next().unwrap(), "1.0");
            }
            "element" => {
                assert_eq!(words.next().unwrap(), "vertex");
                count = words.next().unwrap().parse().unwrap();
            }
            "property" => {
                let ty = words.next().unwrap();
                match words.next().unwrap() {
                    "x" => offsets.mean = stride,
                    "y" | "z" => (),
                    "nx" | "ny" | "nz" => (),
                    "f_dc_0" => offsets.f_dc = stride,
                    "f_dc_1" | "f_dc_2" => (),
                    "opacity" => offsets.opacity = stride,
                    "scale_0" => offsets.scale = stride,
                    "scale_1" | "scale_2" => (),
                    "rot_0" => offsets.rot = stride,
                    "rot_1" | "rot_2" | "rot_3" => (),
                    other => {
                        if let Some(index) = other.strip_prefix("f_rest_") {
                            if index == "0" {
                                offsets.f_rest = stride;
                            }
                            sh_rest_count += 1;
                        } else {
                            log::info!("Skipping property: {}", other);
                        }
                    }
                }
                match ty {
                    "float" => stride += mem::size_of::<f32>(),
                    other => panic!("Unsupported type: {}", other),
                }
            }
            "end_header" => break,
            other => panic!("Unepxected section: {}", other),
        }
        line.clear();
    }

    assert_ne!(offsets.rot, 0);
    assert_ne!(offsets.scale, 0);
    assert_ne!(offsets.opacity, 0);
    assert_ne!(offsets.f_dc, 0);

    assert!(
        sh_rest_count.is_multiple_of(3),
        "f_rest property count must be divisible by three"
    );
    let sh_component_count = sh_rest_count / 3 + 1;
    let root = (sh_component_count as f32).sqrt() as usize;
    assert_eq!(
        root * root,
        sh_component_count,
        "f_rest property count does not describe a complete SH degree"
    );
    let max_sh_degree = root.saturating_sub(1);
    assert!(
        max_sh_degree <= crate::MAX_SH_DEGREE,
        "Gaussian PLY SH degree {max_sh_degree} exceeds supported degree {}",
        crate::MAX_SH_DEGREE
    );
    let sh_component_count = crate::get_sh_component_count(max_sh_degree);
    let sh_rest_per_channel = sh_component_count - 1;
    let ply_rotation =
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), -f32::consts::FRAC_PI_2);

    log::info!("Reading {} vertices with stride {} from PLY", count, stride);

    let mut points = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut scales = Vec::with_capacity(count);
    let mut sh_coefficients = Vec::with_capacity(count * sh_component_count * 3);

    let mut scratch = vec![0u8; stride];
    for _ in 0..count {
        file.read_exact(&mut scratch).unwrap();
        let mean = read_array::<3>(&scratch, offsets.mean);
        let rot = read_array::<4>(&scratch, offsets.rot);
        let scale = read_array::<3>(&scratch, offsets.scale);
        let opacity = read_f32(&scratch, offsets.opacity);

        // Position + opacity
        let position = ply_rotation * glam::Vec3::from(mean);
        points.push(glam::Vec4::new(
            position.x,
            position.y,
            position.z,
            sigmoid(opacity),
        ));

        // Rotation
        let rotation =
            ply_rotation * glam::Quat::from_xyzw(rot[1], rot[2], rot[3], rot[0]).normalize();
        rotations.push(rotation);

        // Scale (exponentiated from log-space)
        scales.push(glam::Vec3::from(scale).exp());

        // SH coefficients - pack as RGB per component
        // DC term
        let f_dc = read_array::<3>(&scratch, offsets.f_dc);
        sh_coefficients.push(f_dc[0]);
        sh_coefficients.push(f_dc[1]);
        sh_coefficients.push(f_dc[2]);

        // Higher order terms
        for component in 1..sh_component_count {
            for channel in 0..3 {
                let property = channel * sh_rest_per_channel + component - 1;
                let offset = offsets.f_rest + property * mem::size_of::<f32>();
                sh_coefficients.push(read_f32(&scratch, offset));
            }
        }
    }

    // Ensure we are at the end of the file
    assert_eq!(file.read(&mut scratch).unwrap(), 0);

    crate::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree: max_sh_degree,
        transforms: Some(crate::Transforms { rotations, scales }),
        adjacency: None,
        radii: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn loads_standard_channel_major_sh3_properties() {
        let path = std::env::temp_dir().join(format!(
            "blade-volume-gaussian-sh3-{}.ply",
            std::process::id()
        ));
        let mut file = fs::File::create(&path).unwrap();
        writeln!(file, "ply").unwrap();
        writeln!(file, "format binary_little_endian 1.0").unwrap();
        writeln!(file, "element vertex 1").unwrap();
        for name in ["x", "y", "z", "nx", "ny", "nz"] {
            writeln!(file, "property float {name}").unwrap();
        }
        for channel in 0..3 {
            writeln!(file, "property float f_dc_{channel}").unwrap();
        }
        writeln!(file, "property float opacity").unwrap();
        for axis in 0..3 {
            writeln!(file, "property float scale_{axis}").unwrap();
        }
        for axis in 0..4 {
            writeln!(file, "property float rot_{axis}").unwrap();
        }
        for property in 0..45 {
            writeln!(file, "property float f_rest_{property}").unwrap();
        }
        writeln!(file, "end_header").unwrap();

        let fixed = [
            0.0_f32, 0.0, 0.0, // position
            0.0, 0.0, 0.0, // normal
            100.0, 200.0, 300.0, // DC
            0.0,   // opacity logit
            0.0, 0.0, 0.0, // log scale
            1.0, 0.0, 0.0, 0.0, // wxyz rotation
        ];
        for value in fixed {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        for property in 0..45 {
            file.write_all(&(property as f32).to_le_bytes()).unwrap();
        }
        drop(file);

        let model = load(path.to_str().unwrap());
        assert_eq!(model.sh_degree, 3);
        assert_eq!(model.sh_coefficients.len(), 48);
        assert_eq!(&model.sh_coefficients[..3], &[100.0, 200.0, 300.0]);
        assert_eq!(&model.sh_coefficients[3..6], &[0.0, 15.0, 30.0]);
        assert_eq!(&model.sh_coefficients[45..48], &[14.0, 29.0, 44.0]);
        std::fs::remove_file(path).unwrap();
    }
}
