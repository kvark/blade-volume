//! COLMAP dense-fusion point clouds.
//!
//! `stereo_fusion` writes a vertex-only binary PLY containing position,
//! normal, and colour. This reader intentionally accepts that exact format:
//! meshes and general-purpose PLY files belong outside the reconstruction
//! boundary.

use crate::inverse::capture;
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

/// Source-image indices recorded beside a COLMAP fused point cloud.
///
/// COLMAP stores this as a compact `fused.ply.vis` stream in the same point
/// order as `fused.ply`. Keeping the flattened representation avoids one heap
/// allocation per dense point on million-point captures.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseVisibility {
    offsets: Vec<usize>,
    image_indices: Vec<u32>,
}

impl DenseVisibility {
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, point: usize) -> &[u32] {
        &self.image_indices[self.offsets[point]..self.offsets[point + 1]]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u32]> {
        (0..self.len()).map(|point| self.get(point))
    }
}

/// Remove dense samples outside the soft visual hull of the training masks.
///
/// Pose-only stereo fusion can retain a depth from a single source image. A
/// fused point is still object geometry only if its projection lies in the
/// object silhouette from almost every training camera. The one-in-five
/// tolerance covers mask resampling and small calibration errors at contours.
/// Unmasked captures are left unchanged.
pub fn retain_soft_visual_hull(
    points: &mut Vec<DensePoint>,
    capture: &capture::Capture,
    views: &[usize],
) -> Option<usize> {
    let masked_views: Vec<_> = views
        .iter()
        .filter_map(|&index| {
            let view = &capture.views[index];
            view.mask.as_deref().map(|mask| {
                (
                    capture::PixelProjection::new(&view.camera, capture.width, capture.height),
                    mask,
                )
            })
        })
        .collect();
    if masked_views.is_empty() {
        return None;
    }

    let before = points.len();
    points.retain(|point| {
        let mut support = 0;
        for &(ref projection, mask) in &masked_views {
            let Some((pixel, _)) = projection.project(point.position) else {
                continue;
            };
            let x = pixel[0].floor() as isize;
            let y = pixel[1].floor() as isize;
            if x < 0 || y < 0 || x >= capture.width as isize || y >= capture.height as isize {
                continue;
            }
            support += (mask[y as usize * capture.width + x as usize] > 0.5) as usize;
        }
        support * 5 >= masked_views.len() * 4
    });
    Some(before - points.len())
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

fn read_u32(reader: &mut impl io::Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl io::Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
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

/// Read the canonical `fused.ply.vis` written beside a COLMAP fused cloud.
pub fn try_load_colmap_fused_visibility(
    path: &path::Path,
    point_count: usize,
) -> io::Result<DenseVisibility> {
    let file = fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let minimum_bytes = 8u64
        .checked_add(
            4u64.checked_mul(point_count as u64)
                .ok_or_else(|| invalid("COLMAP visibility size overflows u64"))?,
        )
        .ok_or_else(|| invalid("COLMAP visibility size overflows u64"))?;
    if file_bytes < minimum_bytes || (file_bytes - minimum_bytes) % 4 != 0 {
        return Err(invalid(format!(
            "COLMAP visibility has {file_bytes} bytes, smaller than or unaligned to its {point_count} records"
        )));
    }
    let observation_count = usize::try_from((file_bytes - minimum_bytes) / 4)
        .map_err(|_| invalid("COLMAP visibility observations do not fit usize"))?;
    let mut reader = io::BufReader::new(file);
    let stored_count = usize::try_from(read_u64(&mut reader)?)
        .map_err(|_| invalid("COLMAP visibility point count does not fit usize"))?;
    if stored_count != point_count {
        return Err(invalid(format!(
            "COLMAP visibility has {stored_count} points, expected {point_count}"
        )));
    }

    let offset_count = point_count
        .checked_add(1)
        .ok_or_else(|| invalid("COLMAP visibility point count overflows usize"))?;
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(offset_count).map_err(|error| {
        invalid(format!(
            "cannot allocate COLMAP visibility offsets: {error}"
        ))
    })?;
    let mut image_indices = Vec::new();
    image_indices
        .try_reserve_exact(observation_count)
        .map_err(|error| {
            invalid(format!(
                "cannot allocate COLMAP visibility indices: {error}"
            ))
        })?;
    offsets.push(0);
    for point in 0..point_count {
        let count = read_u32(&mut reader)? as usize;
        if count > observation_count - image_indices.len() {
            return Err(invalid(format!(
                "COLMAP visibility point {point} exceeds the remaining observation count"
            )));
        }
        for _ in 0..count {
            image_indices.push(read_u32(&mut reader)?);
        }
        offsets.push(image_indices.len());
    }
    if image_indices.len() != observation_count {
        return Err(invalid(format!(
            "COLMAP visibility contains {} unclaimed observations",
            observation_count - image_indices.len()
        )));
    }
    Ok(DenseVisibility {
        offsets,
        image_indices,
    })
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

    fn write_visibility(path: &path::Path, visibility: &[&[u32]]) {
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(visibility.len() as u64).to_le_bytes())
            .unwrap();
        for views in visibility {
            file.write_all(&(views.len() as u32).to_le_bytes()).unwrap();
            for view in *views {
                file.write_all(&view.to_le_bytes()).unwrap();
            }
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
    fn reads_canonical_colmap_fused_visibility() {
        let path = temp("visibility");
        write_visibility(&path, &[&[2, 14, 0], &[], &[7]]);
        let visibility = try_load_colmap_fused_visibility(&path, 3).unwrap();
        assert_eq!(visibility.len(), 3);
        assert!(!visibility.is_empty());
        assert_eq!(visibility.get(0), [2, 14, 0]);
        assert!(visibility.get(1).is_empty());
        assert_eq!(visibility.get(2), [7]);
        assert_eq!(
            visibility.iter().collect::<Vec<_>>(),
            vec![&[2, 14, 0][..], &[][..], &[7][..]]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_misaligned_colmap_fused_visibility() {
        let path = temp("visibility-count");
        write_visibility(&path, &[&[0], &[1]]);
        assert!(try_load_colmap_fused_visibility(&path, 3)
            .unwrap_err()
            .to_string()
            .contains("expected 3"));

        write_visibility(&path, &[&[0], &[1]]);
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0]).unwrap();
        assert!(try_load_colmap_fused_visibility(&path, 2)
            .unwrap_err()
            .to_string()
            .contains("unaligned"));
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

    fn masked_capture(masks: impl IntoIterator<Item = Option<Vec<f32>>>) -> capture::Capture {
        let camera = blade_volume::CameraParams {
            cam_position: [0.0, 0.0, -2.0],
            depth: 100.0,
            cam_orientation: glam::Quat::IDENTITY.to_array(),
            fov: [1.0; 2],
            principal: [0.0; 2],
        };
        capture::Capture {
            width: 4,
            height: 4,
            views: masks
                .into_iter()
                .enumerate()
                .map(|(index, mask)| capture::View {
                    name: format!("view-{index}"),
                    camera,
                    pixels: vec![[0.0; 3]; 16],
                    mask,
                })
                .collect(),
        }
    }

    fn centre_mask(foreground: bool) -> Vec<f32> {
        let mut mask = vec![0.0; 16];
        mask[10] = foreground as u8 as f32;
        mask
    }

    fn dense_point(position: glam::Vec3) -> DensePoint {
        DensePoint {
            position,
            normal: glam::Vec3::Z,
            color: [128; 3],
        }
    }

    #[test]
    fn soft_visual_hull_tolerates_one_mask_disagreement() {
        let capture = masked_capture((0..5).map(|index| Some(centre_mask(index != 4))));
        let mut points = vec![
            dense_point(glam::Vec3::ZERO),
            dense_point(glam::Vec3::new(10.0, 0.0, 0.0)),
        ];

        assert_eq!(
            retain_soft_visual_hull(&mut points, &capture, &[0, 1, 2, 3, 4]),
            Some(1)
        );
        assert_eq!(points, [dense_point(glam::Vec3::ZERO)]);
    }

    #[test]
    fn soft_visual_hull_uses_only_selected_masked_views() {
        let capture = masked_capture([
            Some(centre_mask(true)),
            Some(centre_mask(true)),
            Some(centre_mask(false)),
            None,
        ]);
        let mut selected = vec![dense_point(glam::Vec3::ZERO)];
        let mut all = selected.clone();
        let mut unmasked = selected.clone();

        assert_eq!(
            retain_soft_visual_hull(&mut selected, &capture, &[0, 1, 3]),
            Some(0)
        );
        assert_eq!(
            retain_soft_visual_hull(&mut all, &capture, &[0, 1, 2, 3]),
            Some(1)
        );
        assert_eq!(retain_soft_visual_hull(&mut unmasked, &capture, &[3]), None);
        assert_eq!(selected.len(), 1);
        assert!(all.is_empty());
        assert_eq!(unmasked.len(), 1);
    }
}
