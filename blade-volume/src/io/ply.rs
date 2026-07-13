use super::LoadError;

use std::{f32, fs, io, mem};

const MAX_HEADER_BYTES: usize = 1024 * 1024;

struct Schema {
    count: usize,
    stride: usize,
    mean: [usize; 3],
    rotation: [usize; 4],
    scale: [usize; 3],
    opacity: usize,
    dc: [usize; 3],
    sh_rest: Vec<usize>,
}

#[derive(Default)]
struct PendingSchema {
    count: Option<usize>,
    stride: usize,
    mean: [Option<usize>; 3],
    rotation: [Option<usize>; 4],
    scale: [Option<usize>; 3],
    opacity: Option<usize>,
    dc: [Option<usize>; 3],
    sh_rest: Vec<(usize, usize)>,
}

fn set_once(slot: &mut Option<usize>, offset: usize, name: &str) -> Result<(), LoadError> {
    if slot.replace(offset).is_some() {
        return Err(LoadError::invalid(format!(
            "duplicate Gaussian PLY property '{name}'"
        )));
    }
    Ok(())
}

fn require_offsets<const N: usize>(
    values: [Option<usize>; N],
    names: [&str; N],
) -> Result<[usize; N], LoadError> {
    let mut output = [0; N];
    for index in 0..N {
        output[index] = values[index].ok_or_else(|| {
            LoadError::invalid(format!(
                "Gaussian PLY vertex is missing property '{}'",
                names[index]
            ))
        })?;
    }
    Ok(output)
}

fn finish_schema(mut pending: PendingSchema) -> Result<Schema, LoadError> {
    let count = pending
        .count
        .ok_or_else(|| LoadError::invalid("Gaussian PLY is missing a vertex element"))?;
    let mean = require_offsets(pending.mean, ["x", "y", "z"])?;
    let rotation = require_offsets(pending.rotation, ["rot_0", "rot_1", "rot_2", "rot_3"])?;
    let scale = require_offsets(pending.scale, ["scale_0", "scale_1", "scale_2"])?;
    let dc = require_offsets(pending.dc, ["f_dc_0", "f_dc_1", "f_dc_2"])?;
    let opacity = pending
        .opacity
        .ok_or_else(|| LoadError::invalid("Gaussian PLY vertex is missing property 'opacity'"))?;

    pending.sh_rest.sort_unstable_by_key(|entry| entry.0);
    for (expected, &(actual, _)) in pending.sh_rest.iter().enumerate() {
        if actual != expected {
            return Err(LoadError::invalid(format!(
                "Gaussian PLY f_rest indices must be contiguous from zero; expected {expected}, got {actual}"
            )));
        }
    }
    if !pending.sh_rest.len().is_multiple_of(3) {
        return Err(LoadError::invalid(format!(
            "Gaussian PLY f_rest property count {} is not divisible by three",
            pending.sh_rest.len()
        )));
    }
    let component_count = pending.sh_rest.len() / 3 + 1;
    let degree = (0..=crate::MAX_SH_DEGREE)
        .find(|&degree| crate::get_sh_component_count(degree) == component_count)
        .ok_or_else(|| {
            LoadError::invalid(format!(
                "Gaussian PLY has {} SH components, expected a complete degree up to {}",
                component_count,
                crate::MAX_SH_DEGREE
            ))
        })?;
    debug_assert_eq!(component_count, crate::get_sh_component_count(degree));

    Ok(Schema {
        count,
        stride: pending.stride,
        mean,
        rotation,
        scale,
        opacity,
        dc,
        sh_rest: pending.sh_rest.into_iter().map(|entry| entry.1).collect(),
    })
}

fn parse_header(file: &mut io::BufReader<fs::File>) -> Result<Schema, LoadError> {
    use io::BufRead as _;

    let mut pending = PendingSchema::default();
    let mut line = String::new();
    let mut header_bytes = 0;
    let mut saw_magic = false;
    let mut saw_format = false;
    let mut saw_end = false;
    let mut in_vertex = false;

    loop {
        line.clear();
        let remaining = MAX_HEADER_BYTES.saturating_sub(header_bytes);
        let bytes = io::Read::take(&mut *file, (remaining + 1) as u64).read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        header_bytes += bytes;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(LoadError::invalid(format!(
                "Gaussian PLY header exceeds {MAX_HEADER_BYTES} bytes"
            )));
        }
        let mut words = line.split_whitespace();
        let Some(head) = words.next() else {
            continue;
        };
        match head {
            "ply" if !saw_magic && header_bytes == bytes => saw_magic = true,
            "ply" => return Err(LoadError::invalid("unexpected duplicate PLY magic")),
            "format" => {
                let format = words.next().unwrap_or("");
                let version = words.next().unwrap_or("");
                if format != "binary_little_endian" || version != "1.0" {
                    return Err(LoadError::InvalidData(format!(
                        "Gaussian PLY requires 'format binary_little_endian 1.0', got '{format} {version}'"
                    )));
                }
                if saw_format {
                    return Err(LoadError::invalid("duplicate Gaussian PLY format line"));
                }
                saw_format = true;
            }
            "comment" | "obj_info" => {}
            "element" => {
                let name = words.next().unwrap_or("");
                let count = words
                    .next()
                    .ok_or_else(|| LoadError::invalid(format!("element '{name}' has no count")))?
                    .parse::<usize>()
                    .map_err(|_| {
                        LoadError::invalid(format!("invalid element count for '{name}'"))
                    })?;
                if name != "vertex" {
                    return Err(LoadError::invalid(format!(
                        "Gaussian PLY does not support element '{name}'"
                    )));
                }
                if pending.count.replace(count).is_some() {
                    return Err(LoadError::invalid("duplicate Gaussian PLY vertex element"));
                }
                in_vertex = true;
            }
            "property" => {
                if !in_vertex {
                    return Err(LoadError::invalid(
                        "Gaussian PLY property appears before the vertex element",
                    ));
                }
                let ty = words.next().unwrap_or("");
                let name = words.next().unwrap_or("");
                if ty != "float" || name.is_empty() {
                    return Err(LoadError::invalid(format!(
                        "Gaussian PLY supports named float properties only, got '{ty} {name}'"
                    )));
                }
                let offset = pending.stride;
                match name {
                    "x" => set_once(&mut pending.mean[0], offset, name)?,
                    "y" => set_once(&mut pending.mean[1], offset, name)?,
                    "z" => set_once(&mut pending.mean[2], offset, name)?,
                    "f_dc_0" => set_once(&mut pending.dc[0], offset, name)?,
                    "f_dc_1" => set_once(&mut pending.dc[1], offset, name)?,
                    "f_dc_2" => set_once(&mut pending.dc[2], offset, name)?,
                    "opacity" => set_once(&mut pending.opacity, offset, name)?,
                    "scale_0" => set_once(&mut pending.scale[0], offset, name)?,
                    "scale_1" => set_once(&mut pending.scale[1], offset, name)?,
                    "scale_2" => set_once(&mut pending.scale[2], offset, name)?,
                    "rot_0" => set_once(&mut pending.rotation[0], offset, name)?,
                    "rot_1" => set_once(&mut pending.rotation[1], offset, name)?,
                    "rot_2" => set_once(&mut pending.rotation[2], offset, name)?,
                    "rot_3" => set_once(&mut pending.rotation[3], offset, name)?,
                    other => {
                        if let Some(suffix) = other.strip_prefix("f_rest_") {
                            let index = suffix.parse::<usize>().map_err(|_| {
                                LoadError::invalid(format!(
                                    "invalid Gaussian PLY property '{other}'"
                                ))
                            })?;
                            if pending.sh_rest.iter().any(|entry| entry.0 == index) {
                                return Err(LoadError::invalid(format!(
                                    "duplicate Gaussian PLY property '{other}'"
                                )));
                            }
                            pending.sh_rest.push((index, offset));
                        } else {
                            log::info!("Skipping Gaussian PLY property: {other}");
                        }
                    }
                }
                pending.stride = pending
                    .stride
                    .checked_add(mem::size_of::<f32>())
                    .ok_or_else(|| LoadError::invalid("Gaussian PLY vertex stride overflow"))?;
            }
            "end_header" => {
                saw_end = true;
                break;
            }
            other => {
                return Err(LoadError::invalid(format!(
                    "unexpected Gaussian PLY header token '{other}'"
                )));
            }
        }
    }
    if !saw_magic {
        return Err(LoadError::invalid(
            "Gaussian PLY is missing the 'ply' magic",
        ));
    }
    if !saw_format {
        return Err(LoadError::invalid(
            "Gaussian PLY is missing its format line",
        ));
    }
    if !saw_end {
        return Err(LoadError::invalid("Gaussian PLY header has no end_header"));
    }
    finish_schema(pending)
}

fn read_f32(data: &[u8], offset: usize) -> f32 {
    let bytes: [u8; mem::size_of::<f32>()] = data[offset..offset + mem::size_of::<f32>()]
        .try_into()
        .unwrap();
    f32::from_le_bytes(bytes)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn try_load(file_path: &str) -> Result<crate::PointCloudModel, LoadError> {
    use io::{Read as _, Seek as _};

    let mut file = io::BufReader::new(fs::File::open(file_path)?);
    let schema = parse_header(&mut file)?;
    let body_start = file.stream_position()?;
    let file_len = file.get_ref().metadata()?.len();
    let body_len = schema
        .count
        .checked_mul(schema.stride)
        .ok_or_else(|| LoadError::invalid("Gaussian PLY body size overflow"))?;
    let body_len_u64 = u64::try_from(body_len)
        .map_err(|_| LoadError::invalid("Gaussian PLY body size exceeds file limits"))?;
    let expected_end = body_start
        .checked_add(body_len_u64)
        .ok_or_else(|| LoadError::invalid("Gaussian PLY file size overflow"))?;
    if expected_end != file_len {
        return Err(LoadError::invalid(format!(
            "Gaussian PLY body is {} bytes, expected {body_len}",
            file_len.saturating_sub(body_start)
        )));
    }

    let component_count = schema.sh_rest.len() / 3 + 1;
    let sh_degree = crate::get_sh_degree(component_count);
    let sh_rest_per_channel = component_count - 1;
    let sh_value_count = schema
        .count
        .checked_mul(component_count)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| LoadError::invalid("Gaussian PLY SH allocation size overflow"))?;
    let mut points = Vec::new();
    let mut rotations = Vec::new();
    let mut scales = Vec::new();
    let mut sh_coefficients = Vec::new();
    points.try_reserve_exact(schema.count).map_err(|error| {
        LoadError::invalid(format!("Gaussian point allocation failed: {error}"))
    })?;
    rotations.try_reserve_exact(schema.count).map_err(|error| {
        LoadError::invalid(format!("Gaussian rotation allocation failed: {error}"))
    })?;
    scales.try_reserve_exact(schema.count).map_err(|error| {
        LoadError::invalid(format!("Gaussian scale allocation failed: {error}"))
    })?;
    sh_coefficients
        .try_reserve_exact(sh_value_count)
        .map_err(|error| LoadError::invalid(format!("Gaussian SH allocation failed: {error}")))?;

    let ply_rotation = glam::Quat::from_axis_angle(glam::Vec3::Y, -f32::consts::FRAC_PI_2);
    let mut row = vec![0u8; schema.stride];
    for _ in 0..schema.count {
        file.read_exact(&mut row)?;
        let mean = glam::Vec3::new(
            read_f32(&row, schema.mean[0]),
            read_f32(&row, schema.mean[1]),
            read_f32(&row, schema.mean[2]),
        );
        let position = ply_rotation * mean;
        points.push(position.extend(sigmoid(read_f32(&row, schema.opacity))));

        let rotation = glam::Quat::from_xyzw(
            read_f32(&row, schema.rotation[1]),
            read_f32(&row, schema.rotation[2]),
            read_f32(&row, schema.rotation[3]),
            read_f32(&row, schema.rotation[0]),
        )
        .normalize();
        rotations.push(ply_rotation * rotation);

        scales.push(
            glam::Vec3::new(
                read_f32(&row, schema.scale[0]),
                read_f32(&row, schema.scale[1]),
                read_f32(&row, schema.scale[2]),
            )
            .exp(),
        );

        for &offset in &schema.dc {
            sh_coefficients.push(read_f32(&row, offset));
        }
        for component in 1..component_count {
            for channel in 0..3 {
                let property = channel * sh_rest_per_channel + component - 1;
                sh_coefficients.push(read_f32(&row, schema.sh_rest[property]));
            }
        }
    }

    let model = crate::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree,
        transforms: Some(crate::Transforms { rotations, scales }),
        adjacency: None,
        radii: None,
    };
    model
        .validate()
        .map_err(|error| LoadError::invalid(format!("Gaussian PLY model: {error}")))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "blade-volume-gaussian-{name}-{}-{:?}.ply",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn loads_standard_channel_major_sh3_properties() {
        let path = path("sh3");
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

        let model = try_load(path.to_str().unwrap()).unwrap();
        assert_eq!(model.sh_degree, 3);
        assert_eq!(model.sh_coefficients.len(), 48);
        assert_eq!(&model.sh_coefficients[..3], &[100.0, 200.0, 300.0]);
        assert_eq!(&model.sh_coefficients[3..6], &[0.0, 15.0, 30.0]);
        assert_eq!(&model.sh_coefficients[45..48], &[14.0, 29.0, 44.0]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_header_and_truncated_body_return_errors() {
        let malformed = path("malformed");
        fs::write(&malformed, b"ply\nformat ascii 1.0\nend_header\n").unwrap();
        assert!(matches!(
            try_load(malformed.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(malformed).unwrap();

        let truncated = path("truncated");
        let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n";
        fs::write(&truncated, header).unwrap();
        assert!(matches!(
            try_load(truncated.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(truncated).unwrap();
    }

    #[test]
    fn absurd_vertex_count_is_rejected_before_allocation() {
        let path = path("absurd-count");
        let body = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n",
            usize::MAX
        );
        fs::write(&path, body).unwrap();
        assert!(matches!(
            try_load(path.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn oversized_unterminated_header_line_is_bounded() {
        let path = path("oversized-header");
        let mut data = b"ply\nformat binary_little_endian 1.0\ncomment ".to_vec();
        data.resize(MAX_HEADER_BYTES + 2, b'x');
        fs::write(&path, data).unwrap();
        assert!(matches!(
            try_load(path.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(path).unwrap();
    }
}
