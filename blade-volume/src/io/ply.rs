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

fn read_slice<const N: usize>(data: &[u8], offset: usize) -> [f32; N] {
    unsafe { *(data.as_ptr().add(offset) as *const [f32; N]) }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn load(file_path: &str) -> crate::Model {
    use std::io::{BufRead as _, Read as _};

    let mut count = 0;
    let mut stride = 0;
    let mut offsets = Offsets::default();
    let mut sh_rest_count = 1;

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
                    "f_rest_0" => offsets.f_rest = stride,
                    other => {
                        if let Some(_) = other.strip_prefix("f_rest_") {
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

    let max_sh_degree = crate::get_sh_degree(sh_rest_count / 3 + 1).min(crate::MAX_SH_DEGREE);
    let ply_rotation =
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), -f32::consts::FRAC_PI_2);

    log::info!("Reading {} vertices with stride {} from PLY", count, stride);
    let mut scratch = vec![0u8; stride];
    let gaussians = (0..count)
        .map(|_| {
            file.read_exact(&mut scratch).unwrap();
            let mean = read_slice::<3>(&scratch, offsets.mean);
            let rot = read_slice::<4>(&scratch, offsets.rot);
            let scale = read_slice::<3>(&scratch, offsets.scale);
            let opacity = read_slice::<1>(&scratch, offsets.opacity);
            let mut shc = [glam::Vec3::default(); crate::MAX_SH_COMPONENTS];
            shc[0] = glam::Vec3::from(read_slice::<3>(&scratch, offsets.f_dc));
            for i in 1..crate::get_sh_component_count(max_sh_degree) {
                for k in 0..2 {
                    let offset = offsets.f_rest + (k * sh_rest_count + i) * mem::size_of::<f32>();
                    shc[i][k] = read_slice::<1>(&scratch, offset)[0];
                }
            }
            crate::Gaussian {
                mean: ply_rotation * glam::Vec3::from(mean),
                rotation: ply_rotation
                    * glam::Quat::from_xyzw(rot[1], rot[2], rot[3], rot[0]).normalize(),
                scale: glam::Vec3::from(scale).exp(),
                opacity: sigmoid(opacity[0]),
                shc,
            }
        })
        .collect();
    // Ensure we are at the end of the file
    assert_eq!(file.read(&mut scratch).unwrap(), 0);

    crate::Model {
        gaussians,
        max_sh_degree,
    }
}
