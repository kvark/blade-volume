//! COLMAP dense-fusion point clouds.
//!
//! `stereo_fusion` writes a vertex-only binary PLY containing position,
//! normal, and colour. This reader intentionally accepts that exact format:
//! meshes and general-purpose PLY files belong outside the reconstruction
//! boundary.

use std::{collections, fs, io, path};

const HEADER_LIMIT: usize = 1024 * 1024;
const RECORD_BYTES: usize = 6 * size_of::<f32>() + 3;
const PROPERTIES: [(&str, &str); 9] = [
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("float", "nx"),
    ("float", "ny"),
    ("float", "nz"),
    ("uchar", "red"),
    ("uchar", "green"),
    ("uchar", "blue"),
];

/// One oriented point emitted by COLMAP's dense stereo fusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensePoint {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub color: [u8; 3],
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_header(reader: &mut io::BufReader<fs::File>) -> io::Result<(usize, usize)> {
    let mut line = String::new();
    let mut bytes = 0usize;
    let mut magic = false;
    let mut format = false;
    let mut count = None;
    let mut properties = Vec::new();
    let mut in_vertex = false;

    loop {
        line.clear();
        let remaining = HEADER_LIMIT.saturating_sub(bytes);
        let mut limited = io::Read::take(&mut *reader, (remaining + 1) as u64);
        let read = io::BufRead::read_line(&mut limited, &mut line)?;
        if read == 0 {
            return Err(invalid("COLMAP fused PLY header has no end_header"));
        }
        bytes += read;
        if bytes > HEADER_LIMIT {
            return Err(invalid(format!(
                "COLMAP fused PLY header exceeds {HEADER_LIMIT} bytes"
            )));
        }

        let words: Vec<&str> = line.split_whitespace().collect();
        match *words.as_slice() {
            ["ply"] if bytes == read => magic = true,
            ["format", "binary_little_endian", "1.0"] => format = true,
            ["comment", ..] | ["obj_info", ..] | [] => {}
            ["element", "vertex", value] if count.is_none() => {
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| invalid("invalid COLMAP fused PLY vertex count"))?;
                count = Some(parsed);
                in_vertex = true;
            }
            ["property", ty, name] if in_vertex => {
                properties.push(((*ty).to_string(), (*name).to_string()));
            }
            ["end_header"] => break,
            ["element", name, ..] => {
                return Err(invalid(format!(
                    "COLMAP fused PLY must contain only vertices, found element '{name}'"
                )));
            }
            _ => {
                return Err(invalid(format!(
                    "unexpected COLMAP fused PLY header line '{}'",
                    line.trim_end()
                )));
            }
        }
    }

    if !magic {
        return Err(invalid("COLMAP fused PLY is missing the ply magic"));
    }
    if !format {
        return Err(invalid(
            "COLMAP fused PLY must use binary_little_endian format 1.0",
        ));
    }
    let count = count.ok_or_else(|| invalid("COLMAP fused PLY has no vertex element"))?;
    if properties.len() != PROPERTIES.len()
        || properties
            .iter()
            .zip(PROPERTIES)
            .any(|(actual, expected)| actual.0 != expected.0 || actual.1 != expected.1)
    {
        return Err(invalid(
            "COLMAP fused PLY must have float x/y/z, float nx/ny/nz, and uchar red/green/blue",
        ));
    }
    Ok((count, bytes))
}

fn float(record: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap())
}

/// Read the canonical `fused.ply` written by COLMAP `stereo_fusion`.
pub fn try_load_colmap_fused(path: &path::Path) -> io::Result<Vec<DensePoint>> {
    let file = fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut reader = io::BufReader::new(file);
    let (count, header_bytes) = read_header(&mut reader)?;
    let body_bytes = count
        .checked_mul(RECORD_BYTES)
        .ok_or_else(|| invalid("COLMAP fused PLY body size overflows usize"))?;
    let expected_bytes = header_bytes
        .checked_add(body_bytes)
        .ok_or_else(|| invalid("COLMAP fused PLY file size overflows usize"))?;
    if file_bytes != expected_bytes as u64 {
        return Err(invalid(format!(
            "COLMAP fused PLY has {file_bytes} bytes, expected {expected_bytes}"
        )));
    }

    let mut points = Vec::new();
    points
        .try_reserve_exact(count)
        .map_err(|error| invalid(format!("cannot allocate COLMAP fused cloud: {error}")))?;
    let mut record = [0u8; RECORD_BYTES];
    for index in 0..count {
        io::Read::read_exact(&mut reader, &mut record)?;
        let position = glam::Vec3::new(float(&record, 0), float(&record, 4), float(&record, 8));
        let normal = glam::Vec3::new(float(&record, 12), float(&record, 16), float(&record, 20));
        let Some(normal) = normal.try_normalize() else {
            return Err(invalid(format!(
                "COLMAP fused PLY vertex {index} has an invalid normal"
            )));
        };
        if !position.is_finite() {
            return Err(invalid(format!(
                "COLMAP fused PLY vertex {index} has a non-finite position"
            )));
        }
        points.push(DensePoint {
            position,
            normal,
            color: [record[24], record[25], record[26]],
        });
    }
    Ok(points)
}

fn cell(position: glam::Vec3, origin: glam::Vec3, voxel: f32) -> [i64; 3] {
    ((position - origin) / voxel)
        .floor()
        .to_array()
        .map(|value| value as i64)
}

fn cell_count(points: &[DensePoint], origin: glam::Vec3, voxel: f32, limit: usize) -> usize {
    let mut occupied =
        collections::HashSet::with_capacity(limit.min(points.len()).saturating_add(1));
    for point in points {
        occupied.insert(cell(point.position, origin, voxel));
        if occupied.len() > limit {
            break;
        }
    }
    occupied.len()
}

#[derive(Clone, Copy)]
struct Accumulator {
    position: glam::DVec3,
    normal: glam::DVec3,
    first_normal: glam::Vec3,
    color: [u64; 3],
    count: u64,
}

/// Spatially average an oriented cloud to at most `max_points` vertices.
///
/// The selected voxel size depends only on the point positions. Output cells
/// are sorted, so identical input produces identical output even though the
/// accumulator uses a hash map.
pub fn downsample(points: &[DensePoint], max_points: usize) -> Vec<DensePoint> {
    if points.len() <= max_points {
        return points.to_vec();
    }
    if max_points == 0 {
        return Vec::new();
    }

    let mut minimum = glam::Vec3::splat(f32::INFINITY);
    let mut maximum = glam::Vec3::splat(f32::NEG_INFINITY);
    for point in points {
        minimum = minimum.min(point.position);
        maximum = maximum.max(point.position);
    }
    let extent = (maximum - minimum).max_element();
    let mut low = 0.0f32;
    let mut high = (extent / (max_points as f32).cbrt()).max(f32::EPSILON);
    while cell_count(points, minimum, high, max_points) > max_points {
        low = high;
        high *= 2.0;
    }
    for _ in 0..20 {
        let middle = 0.5 * (low + high);
        if cell_count(points, minimum, middle, max_points) > max_points {
            low = middle;
        } else {
            high = middle;
        }
    }

    let mut cells = collections::HashMap::<[i64; 3], Accumulator>::new();
    for point in points {
        let entry = cells
            .entry(cell(point.position, minimum, high))
            .or_insert(Accumulator {
                position: glam::DVec3::ZERO,
                normal: glam::DVec3::ZERO,
                first_normal: point.normal,
                color: [0; 3],
                count: 0,
            });
        entry.position += point.position.as_dvec3();
        entry.normal += point.normal.as_dvec3();
        for (sum, value) in entry.color.iter_mut().zip(point.color) {
            *sum += u64::from(value);
        }
        entry.count += 1;
    }
    let mut cells: Vec<_> = cells.into_iter().collect();
    cells.sort_unstable_by_key(|entry| entry.0);
    cells
        .into_iter()
        .map(|(_, entry)| {
            let inverse = 1.0 / entry.count as f64;
            let normal = (entry.normal * inverse)
                .as_vec3()
                .try_normalize()
                .unwrap_or(entry.first_normal);
            DensePoint {
                position: (entry.position * inverse).as_vec3(),
                normal,
                color: entry.color.map(|sum| (sum as f64 * inverse).round() as u8),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn temp(name: &str) -> path::PathBuf {
        std::env::temp_dir().join(format!("blade-volume-dense-{name}-{}", std::process::id()))
    }

    fn write_fused(path: &path::Path, points: &[DensePoint]) {
        let mut file = fs::File::create(path).unwrap();
        writeln!(file, "ply").unwrap();
        writeln!(file, "format binary_little_endian 1.0").unwrap();
        writeln!(file, "element vertex {}", points.len()).unwrap();
        for (ty, name) in PROPERTIES {
            writeln!(file, "property {ty} {name}").unwrap();
        }
        writeln!(file, "end_header").unwrap();
        for point in points {
            for value in point.position.to_array() {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            for value in point.normal.to_array() {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            file.write_all(&point.color).unwrap();
        }
    }

    #[test]
    fn reads_canonical_colmap_fused_cloud() {
        let path = temp("canonical");
        let points = [
            DensePoint {
                position: glam::Vec3::new(1.0, 2.0, 3.0),
                normal: glam::Vec3::Z,
                color: [10, 20, 30],
            },
            DensePoint {
                position: glam::Vec3::new(-4.0, 5.0, 6.0),
                normal: glam::Vec3::Y,
                color: [40, 50, 60],
            },
        ];
        write_fused(&path, &points);
        assert_eq!(try_load_colmap_fused(&path).unwrap(), points);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_meshes_and_truncated_records() {
        let mesh = temp("mesh");
        fs::write(
            &mesh,
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nelement face 1\nend_header\n",
        )
        .unwrap();
        assert!(try_load_colmap_fused(&mesh)
            .unwrap_err()
            .to_string()
            .contains("only vertices"));
        fs::remove_file(mesh).unwrap();

        let truncated = temp("truncated");
        write_fused(
            &truncated,
            &[DensePoint {
                position: glam::Vec3::ZERO,
                normal: glam::Vec3::Z,
                color: [0; 3],
            }],
        );
        let file = fs::OpenOptions::new().write(true).open(&truncated).unwrap();
        file.set_len(file.metadata().unwrap().len() - 1).unwrap();
        assert!(try_load_colmap_fused(&truncated)
            .unwrap_err()
            .to_string()
            .contains("expected"));
        fs::remove_file(truncated).unwrap();
    }

    #[test]
    fn spatial_downsample_is_bounded_and_deterministic() {
        let points: Vec<_> = (0..10)
            .flat_map(|y| {
                (0..10).map(move |x| DensePoint {
                    position: glam::Vec3::new(x as f32, y as f32, 0.0),
                    normal: glam::Vec3::new(x as f32 * 0.01, 0.0, 1.0).normalize(),
                    color: [x * 10, y * 10, 100],
                })
            })
            .collect();
        let first = downsample(&points, 16);
        let second = downsample(&points, 16);
        assert!(!first.is_empty() && first.len() <= 16);
        assert_eq!(first, second);
        assert!(first
            .iter()
            .all(|point| (point.normal.length() - 1.0).abs() < 1.0e-6));
    }
}
