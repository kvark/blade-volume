//! COLMAP dense-fusion point clouds and depth workspaces.
//!
//! `stereo_fusion` writes a vertex-only binary PLY containing position,
//! normal, and colour. This reader intentionally accepts that exact format:
//! meshes and general-purpose PLY files belong outside the reconstruction
//! boundary. Pose-only captures can instead be fused here from COLMAP depth
//! and normal maps. That path follows the explicit camera graph in
//! `stereo/patch-match.cfg`, so it does not need fake sparse geometry.

use crate::{colmap, inverse::capture};
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
const ARRAY_HEADER_LIMIT: usize = 128;
const FUSION_CACHE_MAGIC: &[u8; 8] = b"BVFUSN\x02\0";
const FUSION_POINT_BYTES: usize = RECORD_BYTES + size_of::<u64>();
const FUSION_OBSERVATION_BYTES: usize = 3 * size_of::<u32>() + 5 * size_of::<f32>();

/// One oriented point emitted by COLMAP's dense stereo fusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensePoint {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub color: [u8; 3],
}

/// One calibrated depth observation retained in a fused point group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseObservation {
    /// Registered-image index in the dense workspace's `sparse/images.bin`.
    pub image_index: u32,
    /// Integer `(column, row)` in the COLMAP depth map.
    pub pixel: [u32; 2],
    /// Camera-space Z depth read from the depth map.
    pub depth: f32,
    /// Depth-map normal rotated into world space.
    pub normal: glam::Vec3,
    /// Minimum normalized depth, reprojection, and normal agreement in `[0, 1]`.
    pub confidence: f32,
}

/// Fused oriented points with their complete calibrated observation groups.
///
/// Observations are flattened to avoid one allocation per dense point. The
/// group survives visual-hull rejection and point-budget selection; only then
/// is its representative passed to Gaussian support initialization.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseFusion {
    points: Vec<DensePoint>,
    offsets: Vec<usize>,
    observations: Vec<DenseObservation>,
}

impl DenseFusion {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn points(&self) -> &[DensePoint] {
        &self.points
    }

    pub fn observations(&self, point: usize) -> &[DenseObservation] {
        &self.observations[self.offsets[point]..self.offsets[point + 1]]
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn iter_observations(&self) -> impl Iterator<Item = &DenseObservation> {
        self.observations.iter()
    }

    pub fn view_count(&self) -> usize {
        (0..self.len())
            .map(|point| distinct_view_count(self.observations(point)))
            .sum()
    }

    /// Retain whole point groups for which `keep` returns true.
    pub fn retain(&mut self, mut keep: impl FnMut(&DensePoint) -> bool) -> usize {
        let before = self.points.len();
        let mut points = Vec::with_capacity(before);
        let mut offsets = Vec::with_capacity(before + 1);
        let mut observations = Vec::with_capacity(self.observations.len());
        offsets.push(0);
        for point in 0..before {
            if !keep(&self.points[point]) {
                continue;
            }
            points.push(self.points[point]);
            observations.extend_from_slice(self.observations(point));
            offsets.push(observations.len());
        }
        self.points = points;
        self.offsets = offsets;
        self.observations = observations;
        before - self.points.len()
    }

    /// Keep at most one intact evidence group per spatial cell.
    ///
    /// The most-observed group wins, followed by mean geometric confidence and
    /// original input order. Unlike [`downsample`], this never averages two
    /// independently reconstructed depth layers into a new unsupported point.
    /// Every observation belonging to the selected representative is retained.
    pub fn select_groups(&self, max_points: usize) -> Self {
        let Some((minimum, voxel)) = voxel_grid(&self.points, max_points) else {
            return self.subset(0..self.len().min(max_points));
        };
        let mut cells = collections::HashMap::<[i64; 3], usize>::new();
        for point in 0..self.points.len() {
            let key = cell(self.points[point].position, minimum, voxel);
            match cells.get_mut(&key) {
                Some(selected) if self.group_is_better(point, *selected) => *selected = point,
                Some(_) => {}
                None => {
                    cells.insert(key, point);
                }
            }
        }
        let mut cells: Vec<_> = cells.into_iter().collect();
        cells.sort_unstable_by_key(|entry| entry.0);
        let selected = cells
            .into_iter()
            .map(|(_, point)| point)
            .collect::<Vec<_>>();
        self.subset(selected)
    }

    fn subset(&self, points: impl IntoIterator<Item = usize>) -> Self {
        let mut selected_points = Vec::new();
        let mut offsets = vec![0];
        let mut observations = Vec::new();
        for point in points {
            selected_points.push(self.points[point]);
            observations.extend_from_slice(self.observations(point));
            offsets.push(observations.len());
        }
        Self {
            points: selected_points,
            offsets,
            observations,
        }
    }

    fn group_is_better(&self, candidate: usize, selected: usize) -> bool {
        let candidate_observations = self.observations(candidate);
        let selected_observations = self.observations(selected);
        match distinct_view_count(candidate_observations)
            .cmp(&distinct_view_count(selected_observations))
        {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => {
                let candidate_confidence = candidate_observations
                    .iter()
                    .map(|observation| observation.confidence)
                    .sum::<f32>()
                    / candidate_observations.len() as f32;
                let selected_confidence = selected_observations
                    .iter()
                    .map(|observation| observation.confidence)
                    .sum::<f32>()
                    / selected_observations.len() as f32;
                candidate_confidence > selected_confidence
            }
        }
    }
}

fn distinct_view_count(observations: &[DenseObservation]) -> usize {
    observations
        .iter()
        .enumerate()
        .filter(|&(index, observation)| {
            observations[..index]
                .iter()
                .all(|previous| previous.image_index != observation.image_index)
        })
        .count()
}

/// Geometric consistency thresholds for native point-only depth fusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkspaceFusionOptions {
    pub min_views: usize,
    pub max_observations: usize,
    pub max_traversal_depth: usize,
    pub max_reprojection_error: f32,
    pub max_relative_depth_error: f32,
    pub max_normal_error_degrees: f32,
}

impl Default for WorkspaceFusionOptions {
    fn default() -> Self {
        Self {
            min_views: 2,
            max_observations: 10_000,
            max_traversal_depth: 100,
            max_reprojection_error: 2.0,
            max_relative_depth_error: 0.01,
            max_normal_error_degrees: 10.0,
        }
    }
}

/// Source-image indices recorded beside a COLMAP fused point cloud.
///
/// COLMAP stores this as a compact `fused.ply.vis` stream in the same point
/// order as `fused.ply`. An index addresses the registered-image order in the
/// dense workspace's `sparse/images.bin`. Keeping the flattened representation
/// avoids one heap allocation per dense point on million-point captures.
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
    let masked_views = masked_projections(capture, views);
    if masked_views.is_empty() {
        return None;
    }

    let before = points.len();
    points.retain(|point| point_inside_soft_visual_hull(point, capture, &masked_views));
    Some(before - points.len())
}

/// Remove complete fused groups outside the soft visual hull.
pub fn retain_fusion_soft_visual_hull(
    fusion: &mut DenseFusion,
    capture: &capture::Capture,
    views: &[usize],
) -> Option<usize> {
    let masked_views = masked_projections(capture, views);
    if masked_views.is_empty() {
        return None;
    }
    Some(fusion.retain(|point| point_inside_soft_visual_hull(point, capture, &masked_views)))
}

fn masked_projections<'a>(
    capture: &'a capture::Capture,
    views: &[usize],
) -> Vec<(capture::PixelProjection, &'a [f32])> {
    views
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
        .collect()
}

fn point_inside_soft_visual_hull(
    point: &DensePoint,
    capture: &capture::Capture,
    masked_views: &[(capture::PixelProjection, &[f32])],
) -> bool {
    let mut support = 0;
    for &(ref projection, mask) in masked_views {
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

fn read_f32(reader: &mut impl io::Read) -> io::Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn write_u32(writer: &mut impl io::Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl io::Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_f32(writer: &mut impl io::Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
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

/// Persist fused point groups for deterministic point-budget comparisons.
///
/// The selected source-image names and geometric thresholds are part of the
/// file and must match on reload. This makes accidentally replaying a
/// held-camera or differently fused cache fail loudly.
pub fn try_save_fusion_cache(
    path: &path::Path,
    fusion: &DenseFusion,
    source_image_names: &[&str],
    options: WorkspaceFusionOptions,
) -> io::Result<()> {
    let mut writer = io::BufWriter::new(fs::File::create(path)?);
    io::Write::write_all(&mut writer, FUSION_CACHE_MAGIC)?;
    write_u64(&mut writer, source_image_names.len() as u64)?;
    for name in source_image_names {
        write_u64(&mut writer, name.len() as u64)?;
        io::Write::write_all(&mut writer, name.as_bytes())?;
    }
    write_u64(&mut writer, options.min_views as u64)?;
    write_u64(&mut writer, options.max_observations as u64)?;
    write_u64(&mut writer, options.max_traversal_depth as u64)?;
    write_f32(&mut writer, options.max_reprojection_error)?;
    write_f32(&mut writer, options.max_relative_depth_error)?;
    write_f32(&mut writer, options.max_normal_error_degrees)?;
    write_u64(&mut writer, fusion.len() as u64)?;
    write_u64(&mut writer, fusion.observation_count() as u64)?;
    for point in 0..fusion.len() {
        let representative = fusion.points[point];
        for value in representative.position.to_array() {
            write_f32(&mut writer, value)?;
        }
        for value in representative.normal.to_array() {
            write_f32(&mut writer, value)?;
        }
        io::Write::write_all(&mut writer, &representative.color)?;
        write_u64(&mut writer, fusion.observations(point).len() as u64)?;
        for observation in fusion.observations(point) {
            write_u32(&mut writer, observation.image_index)?;
            write_u32(&mut writer, observation.pixel[0])?;
            write_u32(&mut writer, observation.pixel[1])?;
            write_f32(&mut writer, observation.depth)?;
            for value in observation.normal.to_array() {
                write_f32(&mut writer, value)?;
            }
            write_f32(&mut writer, observation.confidence)?;
        }
    }
    io::Write::flush(&mut writer)
}

/// Reload a grouped fusion cache written by [`try_save_fusion_cache`].
pub fn try_load_fusion_cache(
    path: &path::Path,
    expected_source_image_names: &[&str],
    expected_options: WorkspaceFusionOptions,
) -> io::Result<DenseFusion> {
    let file = fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut reader = io::BufReader::new(file);
    let mut magic = [0u8; FUSION_CACHE_MAGIC.len()];
    io::Read::read_exact(&mut reader, &mut magic)?;
    if &magic != FUSION_CACHE_MAGIC {
        return Err(invalid("dense fusion cache has an unsupported format"));
    }
    let source_count = usize::try_from(read_u64(&mut reader)?)
        .map_err(|_| invalid("dense fusion cache source count does not fit usize"))?;
    if source_count != expected_source_image_names.len() {
        return Err(invalid(format!(
            "dense fusion cache has {source_count} source images, expected {}",
            expected_source_image_names.len()
        )));
    }
    for (index, expected) in expected_source_image_names.iter().enumerate() {
        let length = usize::try_from(read_u64(&mut reader)?)
            .map_err(|_| invalid("dense fusion cache source-name length does not fit usize"))?;
        if length != expected.len() {
            return Err(invalid(format!(
                "dense fusion cache source image {index} does not match '{expected}'"
            )));
        }
        let mut name = vec![0u8; length];
        io::Read::read_exact(&mut reader, &mut name)?;
        if name != expected.as_bytes() {
            return Err(invalid(format!(
                "dense fusion cache source image {index} does not match '{expected}'"
            )));
        }
    }
    let options = WorkspaceFusionOptions {
        min_views: usize::try_from(read_u64(&mut reader)?)
            .map_err(|_| invalid("dense fusion cache minimum views do not fit usize"))?,
        max_observations: usize::try_from(read_u64(&mut reader)?)
            .map_err(|_| invalid("dense fusion cache maximum observations do not fit usize"))?,
        max_traversal_depth: usize::try_from(read_u64(&mut reader)?)
            .map_err(|_| invalid("dense fusion cache traversal depth does not fit usize"))?,
        max_reprojection_error: read_f32(&mut reader)?,
        max_relative_depth_error: read_f32(&mut reader)?,
        max_normal_error_degrees: read_f32(&mut reader)?,
    };
    if options != expected_options {
        return Err(invalid(format!(
            "dense fusion cache options {options:?} do not match {expected_options:?}"
        )));
    }
    let point_count = usize::try_from(read_u64(&mut reader)?)
        .map_err(|_| invalid("dense fusion cache point count does not fit usize"))?;
    let observation_count = usize::try_from(read_u64(&mut reader)?)
        .map_err(|_| invalid("dense fusion cache observation count does not fit usize"))?;
    let header_bytes = io::Seek::stream_position(&mut reader)? as usize;
    let expected_bytes = header_bytes
        .checked_add(
            point_count
                .checked_mul(FUSION_POINT_BYTES)
                .ok_or_else(|| invalid("dense fusion cache point size overflows usize"))?,
        )
        .and_then(|bytes| {
            observation_count
                .checked_mul(FUSION_OBSERVATION_BYTES)
                .and_then(|observations| bytes.checked_add(observations))
        })
        .ok_or_else(|| invalid("dense fusion cache size overflows usize"))?;
    if file_bytes != expected_bytes as u64 {
        return Err(invalid(format!(
            "dense fusion cache has {file_bytes} bytes, expected {expected_bytes}"
        )));
    }

    let mut points = Vec::new();
    points.try_reserve_exact(point_count).map_err(|error| {
        invalid(format!(
            "cannot allocate dense fusion cache points: {error}"
        ))
    })?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(point_count.saturating_add(1))
        .map_err(|error| {
            invalid(format!(
                "cannot allocate dense fusion cache offsets: {error}"
            ))
        })?;
    let mut observations = Vec::new();
    observations
        .try_reserve_exact(observation_count)
        .map_err(|error| {
            invalid(format!(
                "cannot allocate dense fusion cache observations: {error}"
            ))
        })?;
    offsets.push(0);
    for point in 0..point_count {
        let position = glam::Vec3::new(
            read_f32(&mut reader)?,
            read_f32(&mut reader)?,
            read_f32(&mut reader)?,
        );
        let normal = glam::Vec3::new(
            read_f32(&mut reader)?,
            read_f32(&mut reader)?,
            read_f32(&mut reader)?,
        );
        let Some(normal) = normal.try_normalize() else {
            return Err(invalid(format!(
                "dense fusion cache point {point} has an invalid normal"
            )));
        };
        if !position.is_finite() {
            return Err(invalid(format!(
                "dense fusion cache point {point} has a non-finite position"
            )));
        }
        let mut color = [0u8; 3];
        io::Read::read_exact(&mut reader, &mut color)?;
        points.push(DensePoint {
            position,
            normal,
            color,
        });
        let count = usize::try_from(read_u64(&mut reader)?)
            .map_err(|_| invalid("dense fusion cache group size does not fit usize"))?;
        if count == 0 || count > observation_count - observations.len() {
            return Err(invalid(format!(
                "dense fusion cache point {point} has an invalid observation count"
            )));
        }
        for _ in 0..count {
            let image_index = read_u32(&mut reader)?;
            let pixel = [read_u32(&mut reader)?, read_u32(&mut reader)?];
            let depth = read_f32(&mut reader)?;
            let normal = glam::Vec3::new(
                read_f32(&mut reader)?,
                read_f32(&mut reader)?,
                read_f32(&mut reader)?,
            );
            let confidence = read_f32(&mut reader)?;
            let Some(normal) = normal.try_normalize() else {
                return Err(invalid(format!(
                    "dense fusion cache point {point} has an invalid observation normal"
                )));
            };
            if !depth.is_finite()
                || depth <= 0.0
                || !confidence.is_finite()
                || !(0.0..=1.0).contains(&confidence)
            {
                return Err(invalid(format!(
                    "dense fusion cache point {point} has invalid observation data"
                )));
            }
            observations.push(DenseObservation {
                image_index,
                pixel,
                depth,
                normal,
                confidence,
            });
        }
        offsets.push(observations.len());
    }
    if observations.len() != observation_count {
        return Err(invalid(format!(
            "dense fusion cache contains {} unclaimed observations",
            observation_count - observations.len()
        )));
    }
    Ok(DenseFusion {
        points,
        offsets,
        observations,
    })
}

#[derive(Clone, Debug)]
struct ColmapArray {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

impl ColmapArray {
    fn get(&self, x: usize, y: usize, channel: usize) -> f32 {
        self.values[channel * self.width * self.height + y * self.width + x]
    }
}

fn try_load_colmap_array(path: &path::Path, channels: usize) -> io::Result<ColmapArray> {
    let bytes = fs::read(path)?;
    let mut separators = Vec::with_capacity(3);
    for (index, &byte) in bytes.iter().take(ARRAY_HEADER_LIMIT).enumerate() {
        if byte == b'&' {
            separators.push(index);
            if separators.len() == 3 {
                break;
            }
        }
    }
    if separators.len() != 3 {
        return Err(invalid(format!(
            "COLMAP array {} has no width&height&channels& header",
            path.display()
        )));
    }
    let parse = |range: std::ops::Range<usize>| -> io::Result<usize> {
        std::str::from_utf8(&bytes[range])
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                invalid(format!(
                    "COLMAP array {} has an invalid header",
                    path.display()
                ))
            })
    };
    let width = parse(0..separators[0])?;
    let height = parse(separators[0] + 1..separators[1])?;
    let stored_channels = parse(separators[1] + 1..separators[2])?;
    if stored_channels != channels {
        return Err(invalid(format!(
            "COLMAP array {} has {stored_channels} channels, expected {channels}",
            path.display()
        )));
    }
    let count = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| invalid("COLMAP array element count overflows usize"))?;
    let header_bytes = separators[2] + 1;
    let expected_bytes = count
        .checked_mul(size_of::<f32>())
        .and_then(|value| value.checked_add(header_bytes))
        .ok_or_else(|| invalid("COLMAP array byte count overflows usize"))?;
    if bytes.len() != expected_bytes {
        return Err(invalid(format!(
            "COLMAP array {} has {} bytes, expected {expected_bytes}",
            path.display(),
            bytes.len()
        )));
    }
    let values = bytes[header_bytes..]
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect();
    Ok(ColmapArray {
        width,
        height,
        values,
    })
}

struct WorkspaceView {
    image_index: u32,
    camera: blade_volume::CameraParams,
    depth: ColmapArray,
    normal: ColmapArray,
}

fn parse_patch_match_graph(
    text: &str,
    local_images: &collections::HashMap<&str, usize>,
) -> io::Result<Vec<Vec<usize>>> {
    let lines: Vec<_> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() % 2 != 0 {
        return Err(invalid(
            "COLMAP patch-match configuration has an unpaired image",
        ));
    }
    let mut graph = vec![Vec::new(); local_images.len()];
    let mut configured = vec![false; local_images.len()];
    for pair in lines.as_chunks::<2>().0 {
        let Some(&reference) = local_images.get(pair[0]) else {
            continue;
        };
        configured[reference] = true;
        let sources = pair[1];
        if sources == "__all__" {
            graph[reference].extend(0..local_images.len());
        } else if sources.starts_with("__auto__") {
            return Err(invalid(format!(
                "pose-only fusion requires explicit sources for {}, found {sources}",
                pair[0]
            )));
        } else {
            for source in sources.split(',').map(str::trim) {
                let Some(&source) = local_images.get(source) else {
                    continue;
                };
                graph[reference].push(source);
            }
        }
    }
    if let Some(missing) = configured.iter().position(|&configured| !configured) {
        let name = local_images
            .iter()
            .find_map(|(&name, &index)| (index == missing).then_some(name))
            .unwrap();
        return Err(invalid(format!(
            "COLMAP patch-match configuration has no explicit sources for {name}"
        )));
    }

    // PatchMatch source selection is directed. Fusion needs an overlap
    // relationship independent of which camera happens to be processed first.
    let directed = graph.clone();
    for (reference, sources) in directed.iter().enumerate() {
        for &source in sources {
            if source != reference {
                graph[source].push(reference);
            }
        }
    }
    for (reference, sources) in graph.iter_mut().enumerate() {
        sources.retain(|&source| source != reference);
        let mut unique = collections::HashSet::with_capacity(sources.len());
        sources.retain(|&source| unique.insert(source));
    }
    Ok(graph)
}

fn try_load_workspace_views(
    workspace: &path::Path,
    allowed_image_names: &[&str],
) -> io::Result<(Vec<WorkspaceView>, Vec<Vec<usize>>)> {
    let reconstruction = colmap::try_load_reconstruction(&workspace.join("sparse"))?;
    let allowed: collections::HashSet<_> = allowed_image_names.iter().copied().collect();
    let mut registered = collections::HashMap::with_capacity(reconstruction.images.len());
    for (index, image) in reconstruction.images.iter().enumerate() {
        if registered
            .insert(image.name.as_str(), (index, image))
            .is_some()
        {
            return Err(invalid(format!(
                "dense workspace contains duplicate image name {}",
                image.name
            )));
        }
    }

    let fusion_path = workspace.join("stereo/fusion.cfg");
    let fusion_text = fs::read_to_string(&fusion_path)?;
    let mut selected = Vec::new();
    for name in fusion_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if !allowed.contains(name) {
            continue;
        }
        let Some(&(registered_index, image)) = registered.get(name) else {
            return Err(invalid(format!(
                "{} references unknown image {name}",
                fusion_path.display()
            )));
        };
        selected.push((name, registered_index, image));
    }
    if selected.len() < 2 {
        return Err(invalid(format!(
            "dense workspace has only {} selected images; fusion needs at least two",
            selected.len()
        )));
    }
    let local_images: collections::HashMap<_, _> = selected
        .iter()
        .enumerate()
        .map(|(local, &(name, _, _))| (name, local))
        .collect();
    let patch_match_path = workspace.join("stereo/patch-match.cfg");
    let patch_match_text = fs::read_to_string(&patch_match_path)?;
    let graph = parse_patch_match_graph(&patch_match_text, &local_images)?;

    let mut views = Vec::with_capacity(selected.len());
    for (name, registered_index, image) in selected {
        let depth_path = workspace.join(format!("stereo/depth_maps/{name}.geometric.bin"));
        let normal_path = workspace.join(format!("stereo/normal_maps/{name}.geometric.bin"));
        let depth = try_load_colmap_array(&depth_path, 1)?;
        let normal = try_load_colmap_array(&normal_path, 3)?;
        if depth.width != normal.width || depth.height != normal.height {
            return Err(invalid(format!(
                "dense workspace image {name} has mismatched depth and normal dimensions"
            )));
        }
        views.push(WorkspaceView {
            image_index: u32::try_from(registered_index)
                .map_err(|_| invalid("dense workspace image index exceeds u32"))?,
            camera: reconstruction.camera_params_for(image, f32::MAX),
            depth,
            normal,
        });
    }
    Ok((views, graph))
}

fn backproject(view: &WorkspaceView, x: usize, y: usize, depth: f32) -> glam::Vec3 {
    let extent = glam::Vec2::new(view.depth.width as f32, view.depth.height as f32);
    let pixel = glam::Vec2::new(x as f32, y as f32);
    let ndc = 2.0 * pixel / extent - glam::Vec2::ONE;
    let principal = glam::Vec2::from(view.camera.principal);
    let tan_half = glam::Vec2::new(
        (0.5 * view.camera.fov[0]).tan(),
        (0.5 * view.camera.fov[1]).tan(),
    );
    let local_xy = (ndc - principal) * tan_half * depth;
    let local = glam::Vec3::new(local_xy.x, local_xy.y, depth);
    glam::Vec3::from(view.camera.cam_position)
        + glam::Quat::from_array(view.camera.cam_orientation) * local
}

fn world_normal(view: &WorkspaceView, x: usize, y: usize) -> Option<glam::Vec3> {
    let local = glam::Vec3::new(
        view.normal.get(x, y, 0),
        view.normal.get(x, y, 1),
        view.normal.get(x, y, 2),
    );
    (glam::Quat::from_array(view.camera.cam_orientation) * local).try_normalize()
}

#[derive(Clone, Copy)]
struct FusionSample {
    position: glam::Vec3,
    observation: DenseObservation,
}

fn median(samples: &mut [f32]) -> f32 {
    let middle = samples.len() / 2;
    samples.select_nth_unstable_by(middle, f32::total_cmp);
    samples[middle]
}

fn fuse_workspace_views(
    views: &[WorkspaceView],
    graph: &[Vec<usize>],
    options: WorkspaceFusionOptions,
) -> io::Result<DenseFusion> {
    if options.min_views < 2
        || options.min_views > options.max_observations
        || options.max_traversal_depth == 0
        || !options.max_reprojection_error.is_finite()
        || options.max_reprojection_error <= 0.0
        || !options.max_relative_depth_error.is_finite()
        || options.max_relative_depth_error <= 0.0
        || !options.max_normal_error_degrees.is_finite()
        || !(0.0..90.0).contains(&options.max_normal_error_degrees)
    {
        return Err(invalid("invalid dense workspace fusion options"));
    }
    if views.len() != graph.len() {
        return Err(invalid("dense workspace graph does not match its images"));
    }

    let projections: Vec<_> = views
        .iter()
        .map(|view| {
            capture::PixelProjection::new(&view.camera, view.depth.width, view.depth.height)
        })
        .collect();
    let mut visited: Vec<_> = views
        .iter()
        .map(|view| vec![false; view.depth.width * view.depth.height])
        .collect();
    let mut completed = vec![false; views.len()];
    let mut points = Vec::new();
    let mut offsets = vec![0];
    let mut observations = Vec::new();
    let minimum_normal_cosine = options.max_normal_error_degrees.to_radians().cos();
    let maximum_reprojection_squared = options.max_reprojection_error.powi(2);

    let mut reference = 0;
    for _ in 0..views.len() {
        let reference_view = &views[reference];
        for y in 0..reference_view.depth.height {
            for x in 0..reference_view.depth.width {
                let pixel = y * reference_view.depth.width + x;
                if visited[reference][pixel] || reference_view.depth.get(x, y, 0) <= 0.0 {
                    continue;
                }
                let mut queue = vec![(reference, x, y, 0usize)];
                let mut samples = Vec::new();
                let mut reference_point = glam::Vec3::ZERO;
                let mut reference_normal = glam::Vec3::ZERO;
                while let Some((view_index, x, y, traversal_depth)) = queue.pop() {
                    let view = &views[view_index];
                    let pixel = y * view.depth.width + x;
                    if visited[view_index][pixel] {
                        continue;
                    }
                    let depth = view.depth.get(x, y, 0);
                    if !depth.is_finite() || depth <= 0.0 {
                        continue;
                    }
                    let Some(normal) = world_normal(view, x, y) else {
                        continue;
                    };
                    let mut confidence = 1.0;
                    if traversal_depth > 0 {
                        let Some((projected, projected_depth)) =
                            projections[view_index].project(reference_point)
                        else {
                            continue;
                        };
                        let depth_error = (projected_depth - depth).abs() / depth;
                        let reprojection_squared = (glam::Vec2::from(projected)
                            - glam::Vec2::new(x as f32, y as f32))
                        .length_squared();
                        let normal_cosine = reference_normal.dot(normal);
                        if depth_error > options.max_relative_depth_error
                            || reprojection_squared > maximum_reprojection_squared
                            || normal_cosine < minimum_normal_cosine
                        {
                            continue;
                        }
                        let depth_confidence = 1.0 - depth_error / options.max_relative_depth_error;
                        let reprojection_confidence =
                            1.0 - reprojection_squared.sqrt() / options.max_reprojection_error;
                        let normal_confidence =
                            (normal_cosine - minimum_normal_cosine) / (1.0 - minimum_normal_cosine);
                        confidence = depth_confidence
                            .min(reprojection_confidence)
                            .min(normal_confidence)
                            .clamp(0.0, 1.0);
                    }
                    let position = backproject(view, x, y, depth);
                    if !position.is_finite() {
                        continue;
                    }
                    visited[view_index][pixel] = true;
                    samples.push(FusionSample {
                        position,
                        observation: DenseObservation {
                            image_index: view.image_index,
                            pixel: [x as u32, y as u32],
                            depth,
                            normal,
                            confidence,
                        },
                    });
                    if traversal_depth == 0 {
                        reference_point = position;
                        reference_normal = normal;
                    }
                    if samples.len() >= options.max_observations
                        || traversal_depth + 1 >= options.max_traversal_depth
                    {
                        continue;
                    }
                    for &next in &graph[view_index] {
                        if completed[next] {
                            continue;
                        }
                        let Some((projected, _)) = projections[next].project(position) else {
                            continue;
                        };
                        let next_x = projected[0].round() as isize;
                        let next_y = projected[1].round() as isize;
                        if next_x < 0
                            || next_y < 0
                            || next_x >= views[next].depth.width as isize
                            || next_y >= views[next].depth.height as isize
                        {
                            continue;
                        }
                        queue.push((next, next_x as usize, next_y as usize, traversal_depth + 1));
                    }
                }
                let observed_views = samples
                    .iter()
                    .enumerate()
                    .filter(|&(index, sample)| {
                        samples[..index].iter().all(|previous| {
                            previous.observation.image_index != sample.observation.image_index
                        })
                    })
                    .count();
                if observed_views < options.min_views {
                    continue;
                }
                let mut coordinates = [Vec::new(), Vec::new(), Vec::new()];
                let mut normals = [Vec::new(), Vec::new(), Vec::new()];
                for sample in &samples {
                    for (values, value) in coordinates.iter_mut().zip(sample.position.to_array()) {
                        values.push(value);
                    }
                    for (values, value) in
                        normals.iter_mut().zip(sample.observation.normal.to_array())
                    {
                        values.push(value);
                    }
                }
                let position = glam::Vec3::new(
                    median(&mut coordinates[0]),
                    median(&mut coordinates[1]),
                    median(&mut coordinates[2]),
                );
                let Some(normal) = glam::Vec3::new(
                    median(&mut normals[0]),
                    median(&mut normals[1]),
                    median(&mut normals[2]),
                )
                .try_normalize() else {
                    continue;
                };
                points.push(DensePoint {
                    position,
                    normal,
                    color: [0; 3],
                });
                observations.extend(samples.into_iter().map(|sample| sample.observation));
                offsets.push(observations.len());
            }
        }
        completed[reference] = true;
        reference = graph[reference]
            .iter()
            .copied()
            .find(|&image| !completed[image])
            .or_else(|| completed.iter().position(|&done| !done))
            .unwrap_or(reference);
    }
    Ok(DenseFusion {
        points,
        offsets,
        observations,
    })
}

/// Fuse a COLMAP depth workspace through its explicit PatchMatch camera graph.
///
/// `allowed_image_names` is the leakage boundary: maps belonging to held
/// cameras are neither loaded nor traversed even if they exist in the
/// workspace. Automatic sparse-track source selection is rejected because a
/// pose-only reconstruction has no tracks from which to derive that graph.
pub fn try_fuse_colmap_workspace(
    workspace: &path::Path,
    allowed_image_names: &[&str],
    options: WorkspaceFusionOptions,
) -> io::Result<DenseFusion> {
    let (views, graph) = try_load_workspace_views(workspace, allowed_image_names)?;
    fuse_workspace_views(&views, &graph, options)
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

fn voxel_grid(points: &[DensePoint], max_points: usize) -> Option<(glam::Vec3, f32)> {
    if points.len() <= max_points || max_points == 0 {
        return None;
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
    Some((minimum, high))
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

    let (minimum, voxel) = voxel_grid(points, max_points).unwrap();

    let mut cells = collections::HashMap::<[i64; 3], Accumulator>::new();
    for point in points {
        let entry = cells
            .entry(cell(point.position, minimum, voxel))
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

    fn write_array(
        path: &path::Path,
        width: usize,
        height: usize,
        channels: usize,
        values: &[f32],
    ) {
        let mut file = fs::File::create(path).unwrap();
        write!(file, "{width}&{height}&{channels}&").unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn workspace_view(image_index: u32, x: f32, source_depth: f32) -> WorkspaceView {
        let width = 3;
        let height = 3;
        let count = width * height;
        let mut normals = vec![0.0; count * 3];
        normals[count * 2..].fill(-1.0);
        WorkspaceView {
            image_index,
            camera: blade_volume::CameraParams {
                cam_position: [x, 0.0, 0.0],
                depth: 100.0,
                cam_orientation: glam::Quat::IDENTITY.to_array(),
                fov: [1.0; 2],
                principal: [0.0; 2],
            },
            depth: ColmapArray {
                width,
                height,
                values: vec![source_depth; count],
            },
            normal: ColmapArray {
                width,
                height,
                values: normals,
            },
        }
    }

    fn observation(image_index: u32, confidence: f32) -> DenseObservation {
        DenseObservation {
            image_index,
            pixel: [0; 2],
            depth: 1.0,
            normal: glam::Vec3::Z,
            confidence,
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
    fn grouped_fusion_cache_round_trips_and_binds_the_training_split() {
        let path = temp("fusion-cache");
        let fusion = DenseFusion {
            points: vec![dense_point(glam::Vec3::new(1.0, 2.0, 3.0))],
            offsets: vec![0, 2],
            observations: vec![observation(4, 0.75), observation(9, 0.5)],
        };
        let options = WorkspaceFusionOptions::default();
        try_save_fusion_cache(&path, &fusion, &["a.jpg", "b.jpg"], options).unwrap();
        assert_eq!(
            try_load_fusion_cache(&path, &["a.jpg", "b.jpg"], options).unwrap(),
            fusion
        );
        assert!(try_load_fusion_cache(&path, &["b.jpg", "a.jpg"], options)
            .unwrap_err()
            .to_string()
            .contains("does not match"));
        let different = WorkspaceFusionOptions {
            min_views: 3,
            ..options
        };
        assert!(try_load_fusion_cache(&path, &["a.jpg", "b.jpg"], different)
            .unwrap_err()
            .to_string()
            .contains("options"));

        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(file.metadata().unwrap().len() - 1).unwrap();
        assert!(try_load_fusion_cache(&path, &["a.jpg", "b.jpg"], options)
            .unwrap_err()
            .to_string()
            .contains("expected"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_planar_colmap_float_arrays_and_rejects_wrong_shape() {
        let path = temp("array");
        write_array(
            &path,
            2,
            2,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
        );
        let array = try_load_colmap_array(&path, 3).unwrap();
        assert_eq!((array.width, array.height), (2, 2));
        assert_eq!(array.get(1, 0, 0), 2.0);
        assert_eq!(array.get(0, 1, 1), 30.0);
        assert_eq!(array.get(1, 1, 2), 400.0);
        assert!(try_load_colmap_array(&path, 1)
            .unwrap_err()
            .to_string()
            .contains("expected 1"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn patch_match_graph_is_explicit_symmetric_and_selected() {
        let images = collections::HashMap::from([("a.jpg", 0), ("b.jpg", 1), ("c.jpg", 2)]);
        let graph =
            parse_patch_match_graph("a.jpg\nb.jpg\nb.jpg\na.jpg\nc.jpg\n__all__\n", &images)
                .unwrap();
        assert_eq!(graph, [vec![1, 2], vec![0, 2], vec![0, 1]]);
        assert!(parse_patch_match_graph(
            "a.jpg\n__auto__, 20\nb.jpg\na.jpg\nc.jpg\na.jpg\n",
            &images,
        )
        .unwrap_err()
        .to_string()
        .contains("requires explicit sources"));
    }

    #[test]
    fn native_depth_fusion_retains_two_view_observation_groups() {
        let views = [workspace_view(4, 0.0, 2.0), workspace_view(9, 0.03, 2.0)];
        let fusion = fuse_workspace_views(
            &views,
            &[vec![1], vec![0]],
            WorkspaceFusionOptions::default(),
        )
        .unwrap();
        assert!(!fusion.is_empty());
        assert!(fusion
            .points()
            .iter()
            .all(|point| (point.position.z - 2.0).abs() < 1.0e-6));
        for point in 0..fusion.len() {
            let observations = fusion.observations(point);
            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].image_index, 4);
            assert_eq!(observations[1].image_index, 9);
            assert!(observations.iter().all(|observation| {
                observation.confidence.is_finite() && (0.0..=1.0).contains(&observation.confidence)
            }));
        }
    }

    #[test]
    fn native_depth_fusion_rejects_inconsistent_depth() {
        let views = [workspace_view(0, 0.0, 2.0), workspace_view(1, 0.0, 4.0)];
        let fusion = fuse_workspace_views(
            &views,
            &[vec![1], vec![0]],
            WorkspaceFusionOptions::default(),
        )
        .unwrap();
        assert!(fusion.is_empty());
        assert_eq!(fusion.observation_count(), 0);
    }

    #[test]
    fn group_selection_keeps_one_supported_observation_group() {
        let mut fusion = DenseFusion {
            points: vec![
                dense_point(glam::Vec3::ZERO),
                dense_point(glam::Vec3::new(0.01, 0.0, 0.0)),
            ],
            offsets: vec![0, 2, 5],
            observations: vec![
                observation(0, 1.0),
                observation(1, 1.0),
                observation(0, 0.8),
                observation(1, 0.8),
                observation(2, 0.8),
            ],
        };
        let selected = fusion.select_groups(1);
        assert_eq!(
            selected.points(),
            [dense_point(glam::Vec3::new(0.01, 0.0, 0.0))]
        );
        assert_eq!(selected.observations(0).len(), 3);

        assert_eq!(fusion.retain(|point| point.position.x > 0.0), 1);
        assert_eq!(fusion.len(), 1);
        assert_eq!(fusion.observations(0).len(), 3);
        assert_eq!(fusion.observations(0)[2].image_index, 2);
    }

    #[test]
    fn group_selection_uses_mean_confidence_after_distinct_view_count() {
        let fusion = DenseFusion {
            points: vec![
                dense_point(glam::Vec3::ZERO),
                dense_point(glam::Vec3::new(0.01, 0.0, 0.0)),
            ],
            offsets: vec![0, 2, 5],
            observations: vec![
                observation(0, 0.9),
                observation(1, 0.9),
                observation(0, 0.7),
                observation(1, 0.7),
                observation(0, 0.7),
            ],
        };

        assert_eq!(
            fusion.select_groups(1).points(),
            [dense_point(glam::Vec3::ZERO)]
        );
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
                    mask: mask.map(Into::into),
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
