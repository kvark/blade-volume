//! Automatic adjacency computation via 3D Delaunay tetrahedralization.
//!
//! Computes Voronoi neighbors from point positions using 3D Delaunay tetrahedralization.
//! Delaunay edges correspond to Voronoi neighbor pairs.

use crate::{Adjacency, CameraParams};
use simple_delaunay_lib::delaunay_3d::{
    delaunay_struct_3d::DelaunayStructure3D, simplicial_struct_3d::Node,
};

/// Configuration for adjacency computation.
#[derive(Clone, Debug)]
pub struct AdjacencyConfig {
    /// Maximum number of neighbors per point. The default is unbounded and
    /// preserves exact topology. A finite cap is approximate; shortest edges
    /// are selected globally while keeping the graph symmetric.
    pub max_neighbors: usize,
    /// Whether to validate the resulting CSR structure (default: true).
    pub validate: bool,
}

impl Default for AdjacencyConfig {
    fn default() -> Self {
        Self {
            max_neighbors: usize::MAX,
            validate: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdjacencyError {
    TooFewPoints { count: usize },
    TriangulationFailed(String),
}

impl std::fmt::Display for AdjacencyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            AdjacencyError::TooFewPoints { count } => {
                write!(
                    formatter,
                    "need at least 4 points for 3D Delaunay adjacency, got {count}"
                )
            }
            AdjacencyError::TriangulationFailed(ref message) => {
                write!(formatter, "3D Delaunay triangulation failed: {message}")
            }
        }
    }
}

impl std::error::Error for AdjacencyError {}

fn build_symmetric_csr(
    points: &[glam::Vec4],
    neighbor_sets: &mut [Vec<u32>],
    config: &AdjacencyConfig,
) -> Adjacency {
    if config.max_neighbors == usize::MAX {
        // Every private caller records both directions before reaching this
        // function. For exact, unbounded topology, sorting and deduplicating
        // each row is therefore sufficient. The capped path below needs a
        // global distance order to choose a symmetric subset; doing that work
        // here would sort every Delaunay edge and allocate a second graph only
        // to retain all of it.
        let capacity = neighbor_sets.iter().map(Vec::len).sum();
        let mut offsets = Vec::with_capacity(points.len() + 1);
        let mut neighbors = Vec::with_capacity(capacity);
        offsets.push(0);
        for (i, list) in neighbor_sets.iter_mut().enumerate() {
            list.sort_unstable();
            list.dedup();
            neighbors.extend(
                list.iter()
                    .copied()
                    .filter(|&neighbor| neighbor as usize != i),
            );
            offsets.push(neighbors.len() as u32);
        }
        if config.validate {
            validate_csr(&offsets, &neighbors, points.len());
        }
        return Adjacency { neighbors, offsets };
    }

    let mut edges = Vec::new();
    for (i, neighbors) in neighbor_sets.iter_mut().enumerate() {
        neighbors.sort_unstable();
        neighbors.dedup();
        for &neighbor in neighbors.iter() {
            let j = neighbor as usize;
            if i == j {
                continue;
            }
            edges.push((i.min(j), i.max(j)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges.sort_by(|&(a0, b0), &(a1, b1)| {
        let p0 = points[a0].truncate();
        let q0 = points[b0].truncate();
        let p1 = points[a1].truncate();
        let q1 = points[b1].truncate();
        p0.distance_squared(q0)
            .total_cmp(&p1.distance_squared(q1))
            .then_with(|| (a0, b0).cmp(&(a1, b1)))
    });

    let mut selected = vec![Vec::new(); points.len()];
    for (a, b) in edges {
        if selected[a].len() < config.max_neighbors && selected[b].len() < config.max_neighbors {
            selected[a].push(b as u32);
            selected[b].push(a as u32);
        }
    }

    let mut offsets = Vec::with_capacity(points.len() + 1);
    let mut neighbors = Vec::new();
    offsets.push(0);
    for list in selected.iter_mut() {
        list.sort_unstable();
        neighbors.extend_from_slice(list);
        offsets.push(neighbors.len() as u32);
    }
    if config.validate {
        validate_csr(&offsets, &neighbors, points.len());
    }
    Adjacency { neighbors, offsets }
}

/// Computes adjacency from point positions using Delaunay tetrahedralization.
///
/// # Algorithm
/// 1. Convert `Vec4` positions (xyz) to `[f64; 3]` format
/// 2. Compute 3D Delaunay tetrahedralization
/// 3. Extract edges from tetrahedra (Delaunay edges = Voronoi neighbor pairs)
/// 4. Build symmetric CSR adjacency, optionally capped by shortest edge
/// 5. Optionally validate using CSR invariants
///
pub fn try_compute_adjacency(
    points: &[glam::Vec4],
    config: &AdjacencyConfig,
) -> Result<Adjacency, AdjacencyError> {
    let num_points = points.len();

    if num_points < 4 {
        return Err(AdjacencyError::TooFewPoints { count: num_points });
    }

    // Convert to f64 for Delaunay computation
    let points_f64: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p.x as f64, p.y as f64, p.z as f64])
        .collect();

    // Compute Delaunay tetrahedralization
    let mut delaunay = DelaunayStructure3D::new();
    delaunay
        .insert_vertices(&points_f64, true)
        .map_err(|error| AdjacencyError::TriangulationFailed(format!("{error:?}")))?;

    // Build neighbor sets from tetrahedra edges
    let mut neighbor_sets: Vec<Vec<u32>> = vec![Vec::new(); num_points];

    let simplicial = delaunay.get_simplicial();
    let num_tetra = simplicial.get_nb_tetrahedra();

    for i in 0..num_tetra {
        let tetra = match simplicial.get_tetrahedron(i) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Skip tetrahedra containing the point at infinity
        if tetra.contains_infinity() {
            continue;
        }

        let nodes = tetra.nodes();

        // Extract all 6 edges from the tetrahedron (4 choose 2 = 6)
        // Each edge represents a Voronoi neighbor pair
        let edges: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        for (a, b) in edges {
            if let (Node::Value(idx_a), Node::Value(idx_b)) = (nodes[a], nodes[b]) {
                // Add bidirectional edges
                neighbor_sets[idx_a].push(idx_b as u32);
                neighbor_sets[idx_b].push(idx_a as u32);
            }
        }
    }

    Ok(build_symmetric_csr(points, &mut neighbor_sets, config))
}

/// Computes adjacency from point positions using Delaunay tetrahedralization.
///
/// # Panics
/// Panics when [`try_compute_adjacency`] cannot construct exact 3D topology.
pub fn compute_adjacency(points: &[glam::Vec4], config: &AdjacencyConfig) -> Adjacency {
    try_compute_adjacency(points, config).unwrap_or_else(|error| panic!("{error}"))
}

/// Computes adjacency with default configuration.
pub fn compute_adjacency_default(points: &[glam::Vec4]) -> Adjacency {
    compute_adjacency(points, &AdjacencyConfig::default())
}

pub fn try_compute_adjacency_default(points: &[glam::Vec4]) -> Result<Adjacency, AdjacencyError> {
    try_compute_adjacency(points, &AdjacencyConfig::default())
}

/// Spring-relaxation: each point pushes its Delaunay neighbours apart along
/// the connecting edge to a target spacing equal to the mean current edge
/// length. After each iteration the adjacency is rebuilt to reflect the new
/// positions.
///
/// This is a cheap cluster-spreading heuristic, not a Lloyd/CVT pass. True
/// Lloyd relaxation moves sites to Voronoi cell centroids and needs the dual
/// cell geometry; this routine only uses graph edges.
///
/// `step` in `(0, 1]` is the per-iteration relaxation rate. `0.3` is a
/// reasonable default. Density (`w`) and other model state are preserved.
pub fn spring_relax(model: &mut crate::PointCloudModel, iterations: usize, step: f32) {
    assert!(
        (0.0..=1.0).contains(&step),
        "spring_relax: step {step} must be in [0, 1]"
    );
    for _ in 0..iterations {
        if model.adjacency.is_none() {
            model.compute_adjacency_default();
        }
        let adj = model.adjacency.as_ref().unwrap();

        // Target spacing = mean edge length across the whole graph.
        let mut total_len = 0.0f64;
        let mut edge_count = 0usize;
        for i in 0..model.points.len() {
            let begin = adj.offsets[i] as usize;
            let end = adj.offsets[i + 1] as usize;
            for &n in &adj.neighbors[begin..end] {
                if (n as usize) < i {
                    continue; // count each edge once
                }
                let a = model.points[i];
                let b = model.points[n as usize];
                total_len += (glam::Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z).length()) as f64;
                edge_count += 1;
            }
        }
        let target = if edge_count > 0 {
            (total_len / edge_count as f64) as f32
        } else {
            0.0
        };
        if target <= 0.0 {
            break;
        }

        // Accumulate per-point displacement = sum over neighbours j of
        // (target - |edge|) * unit_vec_from_j_to_i. Long edges contract,
        // short edges expand.
        let mut disp: Vec<glam::Vec3> = vec![glam::Vec3::ZERO; model.points.len()];
        for (i, pi_slot) in disp.iter_mut().enumerate() {
            let begin = adj.offsets[i] as usize;
            let end = adj.offsets[i + 1] as usize;
            let pi = glam::Vec3::new(model.points[i].x, model.points[i].y, model.points[i].z);
            for &n in &adj.neighbors[begin..end] {
                let pj_pt = model.points[n as usize];
                let pj = glam::Vec3::new(pj_pt.x, pj_pt.y, pj_pt.z);
                let v = pi - pj;
                let len = v.length();
                if len < 1e-9 {
                    continue;
                }
                let unit = v / len;
                let force = (target - len) * step;
                *pi_slot += unit * force;
            }
        }

        let mut new_points: Vec<glam::Vec4> = Vec::with_capacity(model.points.len());
        for (i, p) in model.points.iter().enumerate() {
            let pos = glam::Vec3::new(p.x, p.y, p.z) + disp[i];
            new_points.push(glam::Vec4::new(pos.x, pos.y, pos.z, p.w));
        }
        model.points = new_points;
        model.adjacency = None;
    }
    model.compute_adjacency_default();
}

/// Estimate per-point radii from the nearest-neighbour distance, scaled by
/// `factor`. This is the simplest local-feature-size estimator for a raw point
/// cloud or sampled mesh. It is intentionally not the PowerFoam reference
/// initializer, which averages several nearest-neighbour distances and also
/// applies a camera-projected support cap.
///
/// Given a point cloud `points`, the radius assigned to site `i` is
/// `factor * min_{j != i} |p_i - p_j|`. A `factor` near `0.5` keeps the
/// balls just barely touching their nearest neighbours, which makes the
/// resulting Čech complex match the Delaunay graph in the common case but
/// shrinks cells in dense regions.
pub fn radii_from_nearest_neighbour(points: &[glam::Vec4], factor: f32) -> Vec<f32> {
    let n = points.len();
    if n < 2 {
        return vec![0.0; n];
    }

    // Insert each exact position once. Besides reducing work, this avoids a
    // fixed-bucket k-d tree degenerating when reconstruction/conversion emits
    // many coincident sites. Every duplicate receives the distance to the
    // nearest *distinct* position, matching the old quadratic implementation.
    let coordinate_key = |value: f32| if value == 0.0 { 0 } else { value.to_bits() };
    let mut unique_indices = std::collections::HashMap::new();
    let mut unique_positions = Vec::new();
    for point in points {
        let key = (
            coordinate_key(point.x),
            coordinate_key(point.y),
            coordinate_key(point.z),
        );
        if let std::collections::hash_map::Entry::Vacant(entry) = unique_indices.entry(key) {
            entry.insert(unique_positions.len());
            unique_positions.push([point.x, point.y, point.z]);
        }
    }
    if unique_positions.len() < 2 {
        return vec![0.0; n];
    }

    let tree = kiddo::ImmutableKdTree::new_from_slice(&unique_positions);
    points
        .iter()
        .map(|point| {
            let query = [point.x, point.y, point.z];
            let nearest_distinct = tree
                .nearest_n::<kiddo::SquaredEuclidean>(&query, std::num::NonZero::new(2).unwrap())
                .into_iter()
                .find(|hit| hit.distance > 0.0)
                .map_or(0.0, |hit| hit.distance.sqrt());
            (nearest_distinct * factor).max(0.0)
        })
        .collect()
}

/// Initialize support radii with the PowerFoam reference policy.
///
/// The raw radius is the mean of the eight closest site distances, including
/// the zero self-distance. For sites visible inside a training camera, it is
/// capped to 10% of the projected half-image height at that depth. The cap
/// prevents sparse outliers from creating very large balls and dense Čech
/// neighborhoods while preserving the reference's local spacing estimate.
pub fn radii_from_powerfoam_reference(points: &[glam::Vec4], cameras: &[CameraParams]) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }

    let positions = points
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    let tree = kiddo::ImmutableKdTree::new_from_slice(&positions);
    let sample_count = points.len().min(8);
    let sample_count_nonzero = std::num::NonZero::new(sample_count).unwrap();
    let mut radii = points
        .iter()
        .map(|point| {
            let query = [point.x, point.y, point.z];
            tree.nearest_n::<kiddo::SquaredEuclidean>(&query, sample_count_nonzero)
                .iter()
                .map(|hit| hit.distance.sqrt())
                .sum::<f32>()
                / sample_count as f32
        })
        .collect::<Vec<_>>();

    for camera in cameras {
        let eye = glam::Vec3::from_array(camera.cam_position);
        let orientation = glam::Quat::from_array(camera.cam_orientation);
        let tan_half = glam::Vec2::new((0.5 * camera.fov[0]).tan(), (0.5 * camera.fov[1]).tan());
        let principal = glam::Vec2::from_array(camera.principal);
        let lower_extent = (glam::Vec2::NEG_ONE - principal) * tan_half;
        let upper_extent = (glam::Vec2::ONE - principal) * tan_half;
        for (point, radius) in points.iter().zip(radii.iter_mut()) {
            let camera_point = orientation.inverse() * (point.truncate() - eye);
            if camera_point.z <= 0.0 {
                continue;
            }
            let projected = camera_point.truncate() / camera_point.z;
            if projected.cmpgt(lower_extent).all() && projected.cmplt(upper_extent).all() {
                *radius = radius.min(0.1 * camera_point.z * tan_half.y);
            }
        }
    }
    radii
}

const CECH_POINTS_PER_WORKER: usize = 4_096;
const CECH_MAX_WORKERS: usize = 16;
const CECH_RADIUS_BINS: usize = 8;

/// Computes the Čech-complex adjacency for a set of weighted points (Power Foam).
///
/// An edge `{i, j}` is emitted when the balls `B(p_i, r_i)` and `B(p_j, r_j)` overlap,
/// i.e. `|p_i - p_j| <= r_i + r_j`. The result is the same CSR [`Adjacency`] used by
/// the unweighted Voronoi path so downstream code (shaders, validation) is unchanged.
///
/// `radii.len()` must equal `points.len()`. Negative radii are clamped to `0`.
///
/// Implementation: partitions sites into logarithmic radius bands and builds
/// one shared immutable k-d tree per band. Each point queries a band within
/// `r_i + r_band_max`, then filters by the exact overlap predicate. Tighter
/// band bounds avoid searching to the largest outlier radius for every site.
pub fn compute_cech(points: &[glam::Vec4], radii: &[f32], config: &AdjacencyConfig) -> Adjacency {
    let available_workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let useful_workers = points.len().div_ceil(CECH_POINTS_PER_WORKER).max(1);
    let worker_count = available_workers.min(useful_workers).min(CECH_MAX_WORKERS);
    compute_cech_with_workers(points, radii, config, worker_count)
}

fn compute_cech_with_workers(
    points: &[glam::Vec4],
    radii: &[f32],
    config: &AdjacencyConfig,
    worker_count: usize,
) -> Adjacency {
    let num_points = points.len();
    assert_eq!(
        radii.len(),
        num_points,
        "compute_cech: radii.len() ({}) must equal points.len() ({})",
        radii.len(),
        num_points
    );
    assert!(worker_count > 0, "compute_cech needs at least one worker");
    if num_points == 0 {
        return build_symmetric_csr(points, &mut [], config);
    }

    let radii: Vec<f32> = radii.iter().map(|&r| r.max(0.0)).collect();
    // Immutable construction handles quantized/coincident coordinates without
    // overflowing the mutable tree's fixed leaf buckets.
    let positions = points
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    let positive_min = radii
        .iter()
        .copied()
        .filter(|&radius| radius > 0.0)
        .fold(f32::INFINITY, f32::min);
    let r_max = radii.iter().copied().fold(0.0_f32, f32::max);
    let log_span = if positive_min.is_finite() && r_max > positive_min {
        (r_max / positive_min).ln()
    } else {
        0.0
    };
    let mut bin_indices = vec![Vec::new(); CECH_RADIUS_BINS];
    for (index, &radius) in radii.iter().enumerate() {
        let bin = if radius == 0.0 || log_span == 0.0 {
            0
        } else {
            (((radius / positive_min).ln() / log_span) * CECH_RADIUS_BINS as f32).floor() as usize
        }
        .min(CECH_RADIUS_BINS - 1);
        bin_indices[bin].push(index);
    }
    let radius_bins = bin_indices
        .into_iter()
        .filter(|indices| !indices.is_empty())
        .map(|indices| {
            let bin_positions = indices
                .iter()
                .map(|&index| positions[index])
                .collect::<Vec<_>>();
            let bin_max = indices
                .iter()
                .map(|&index| radii[index])
                .fold(0.0_f32, f32::max);
            (
                indices,
                bin_max,
                kiddo::ImmutableKdTree::new_from_slice(&bin_positions),
            )
        })
        .collect::<Vec<_>>();

    let build_rows = |range: std::ops::Range<usize>| {
        range
            .map(|i| {
                let p = points[i];
                let r_i = radii[i];
                let q = [p.x, p.y, p.z];
                let mut neighbors = Vec::new();
                for &(ref indices, bin_max, ref kd_tree) in radius_bins.iter() {
                    let bound = r_i + bin_max;
                    let bound_sq = bound * bound;
                    for hit in kd_tree.within_unsorted::<kiddo::SquaredEuclidean>(&q, bound_sq) {
                        let j = indices[hit.item as usize];
                        if j == i {
                            continue;
                        }
                        let sum = r_i + radii[j];
                        if hit.distance <= sum * sum {
                            neighbors.push(j as u32);
                        }
                    }
                }
                neighbors
            })
            .collect::<Vec<_>>()
    };
    let worker_count = worker_count.min(num_points);
    let mut neighbor_sets = if worker_count == 1 {
        build_rows(0..num_points)
    } else {
        let chunk_size = num_points.div_ceil(worker_count);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for start in (0..num_points).step_by(chunk_size) {
                let end = (start + chunk_size).min(num_points);
                let build_rows = &build_rows;
                handles.push(scope.spawn(move || build_rows(start..end)));
            }
            let mut rows = Vec::with_capacity(num_points);
            for handle in handles {
                rows.extend(handle.join().expect("compute_cech worker panicked"));
            }
            rows
        })
    };

    build_symmetric_csr(points, &mut neighbor_sets, config)
}

/// [`compute_cech`] with the default [`AdjacencyConfig`].
pub fn compute_cech_default(points: &[glam::Vec4], radii: &[f32]) -> Adjacency {
    compute_cech(points, radii, &AdjacencyConfig::default())
}

/// Qhull-backed Delaunay tetrahedralisation.
///
/// Returns the exact Voronoi adjacency (each point's neighbour set is the
/// set of points sharing a Delaunay edge with it). Typical well-behaved 3D
/// inputs are near O(N log N + output), but Delaunay output and worst-case
/// time/space are quadratic. In practice this scales materially beyond the
/// `simple_delaunay_lib` backend; callers still need a point budget.
///
/// Adjacency output matches [`compute_adjacency`] modulo Qhull's joggle:
/// degenerate (cocircular / coplanar) configurations are perturbed by
/// ~1e-13 to break ties. For all but pathological inputs the resulting
/// Voronoi neighbour sets agree.
///
/// # Panics
/// Panics if fewer than 5 points are provided (Qhull needs at least 5
/// for a 3D Delaunay) or if the C library reports an error.
#[cfg(feature = "qhull")]
pub fn compute_adjacency_qhull(points: &[glam::Vec4], config: &AdjacencyConfig) -> Adjacency {
    let num_points = points.len();
    assert!(
        num_points >= 5,
        "compute_adjacency_qhull: need >= 5 points (got {num_points})",
    );

    // Qhull wants f64 coordinates. Build the input as a single Vec to
    // pass via `build_managed`; the borrowed-iterator overload runs out
    // of lifetime when the iterator outlives the input slice.
    let coords: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p.x as f64, p.y as f64, p.z as f64])
        .collect();
    let qh = qhull::Qh::new_delaunay(coords).expect("Qhull Delaunay construction failed");

    let mut neighbour_sets: Vec<Vec<u32>> = vec![Vec::new(); num_points];

    for facet in qh
        .simplices()
        .filter(|f| !f.is_sentinel() && !f.upper_delaunay())
    {
        let vs = match facet.vertices() {
            Some(v) => v,
            None => continue,
        };
        let ids: Vec<usize> = vs.iter().filter_map(|v| v.index(&qh)).collect();
        if ids.len() != 4 {
            continue;
        }
        // The six edges of the tetrahedron — each is a Voronoi
        // neighbour pair.
        const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (a, b) in EDGES {
            neighbour_sets[ids[a]].push(ids[b] as u32);
            neighbour_sets[ids[b]].push(ids[a] as u32);
        }
    }

    let adjacency = build_symmetric_csr(points, &mut neighbour_sets, config);

    // `qhull` 0.4's `Drop` calls `qh_freeqhull` but omits the required
    // `qh_memfreeshort`, retaining Qhull's short-allocation arena after every
    // topology rebuild. Position training rebuilds hundreds of times, so that
    // leak grows to gigabytes. Free both arenas explicitly. `qh_freeqhull`
    // zeros every non-allocator field; the crate's later Drop call is
    // therefore a harmless no-op over the cleared structure and still drops
    // its Rust-owned coordinate and IO buffers normally.
    unsafe {
        let raw = qhull::Qh::raw_ptr(&qh);
        qhull::sys::qh_freeqhull(raw, !qhull::sys::qh_ALL);
        let mut current_long = 0;
        let mut total_long = 0;
        qhull::sys::qh_memfreeshort(raw, &mut current_long, &mut total_long);
        assert_eq!(
            current_long, 0,
            "Qhull retained {current_long} long allocations ({total_long} bytes)",
        );
    }

    adjacency
}

/// [`compute_adjacency_qhull`] with the default [`AdjacencyConfig`].
#[cfg(feature = "qhull")]
pub fn compute_adjacency_qhull_default(points: &[glam::Vec4]) -> Adjacency {
    compute_adjacency_qhull(points, &AdjacencyConfig::default())
}

/// Symmetric k-nearest-neighbour adjacency.
///
/// Connects each point to its `k` nearest neighbours (Euclidean) and
/// symmetrises: if `A` lists `B` as a neighbour, `B` also lists `A`. The
/// resulting graph is a strict superset of the standard k-NN graph and
/// is typically slightly denser than `2k` neighbours per point in the
/// interior, less near the boundary.
///
/// Cost: O(N log N) time (kd-tree build + per-point k-NN queries) and
/// O(N · k) memory. This scales to tens of thousands of points where
/// `compute_adjacency` (Delaunay tetrahedralisation) becomes
/// O(N^1.5)-memory and runs out of RAM around 8 K.
///
/// The resulting adjacency is not the Voronoi/Delaunay one, so the
/// radfoam traversal it feeds will skip across "true" cell boundaries
/// occasionally. Empirically this is fine for appearance training:
/// the differentiable forward sums over a path, so getting the wrong
/// next cell once or twice just averages a slightly different bag of
/// nearby densities.
///
/// # Panics
/// Panics if `k >= points.len()` or `points.is_empty()`.
pub fn compute_knn(points: &[glam::Vec4], k: usize) -> Adjacency {
    let n = points.len();
    assert!(n > 0, "compute_knn: empty point cloud");
    assert!(
        k < n,
        "compute_knn: k ({k}) must be smaller than the point count ({n})",
    );

    let mut kd: kiddo::KdTree<f32, 3> = kiddo::KdTree::new();
    for (i, p) in points.iter().enumerate() {
        kd.add(&[p.x, p.y, p.z], i as u64);
    }

    // Pull k+1 (the closest is the point itself) for each query.
    let want = (k + 1).min(n);
    let mut neighbour_sets: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, p) in points.iter().enumerate() {
        let q = [p.x, p.y, p.z];
        let hits = kd.nearest_n::<kiddo::SquaredEuclidean>(&q, want);
        for hit in hits {
            let j = hit.item as usize;
            if j == i {
                continue;
            }
            neighbour_sets[i].push(j as u32);
            // Symmetrise: the kd-tree gives us i → j; record j → i too.
            // We'll dedup below.
            neighbour_sets[j].push(i as u32);
        }
    }

    let mut offsets = Vec::with_capacity(n + 1);
    let mut neighbours = Vec::new();
    offsets.push(0u32);
    for list in neighbour_sets.iter_mut() {
        list.sort_unstable();
        list.dedup();
        neighbours.extend_from_slice(list);
        offsets.push(neighbours.len() as u32);
    }

    let adj = Adjacency {
        neighbors: neighbours,
        offsets,
    };
    validate_csr(&adj.offsets, &adj.neighbors, n);
    adj
}

/// Validates a CSR adjacency structure.
///
/// # Panics
/// Panics if the CSR structure is invalid.
pub fn validate_csr(offsets: &[u32], neighbors: &[u32], num_points: usize) {
    validate_csr_result(offsets, neighbors, num_points).unwrap_or_else(|error| panic!("{error}"));
}

pub(crate) fn validate_csr_result(
    offsets: &[u32],
    neighbors: &[u32],
    num_points: usize,
) -> Result<(), String> {
    if offsets.is_empty() || offsets.len() != num_points + 1 {
        return Err(format!(
            "Invalid adjacency_offsets length: expected {}, got {}",
            num_points + 1,
            offsets.len()
        ));
    }
    if offsets[0] != 0 {
        return Err(format!(
            "Invalid adjacency_offsets[0]: expected 0, got {}",
            offsets[0]
        ));
    }
    let last = offsets[num_points] as usize;
    if last != neighbors.len() {
        return Err(format!(
            "Invalid adjacency_offsets[N]: expected {}, got {}",
            neighbors.len(),
            last
        ));
    }
    for i in 0..num_points {
        let a = offsets[i] as usize;
        let b = offsets[i + 1] as usize;
        if b < a || b > neighbors.len() {
            return Err(format!(
                "Invalid adjacency offset range for point {i}: [{a}, {b})"
            ));
        }
    }

    // Validate every list before using binary search for reverse edges.
    for i in 0..num_points {
        let begin = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let list = &neighbors[begin..end];
        if list.windows(2).any(|window| window[0] >= window[1]) {
            return Err(format!(
                "Adjacency list for point {i} must be sorted and unique"
            ));
        }
        for (local_index, &neighbor) in list.iter().enumerate() {
            if neighbor as usize >= num_points {
                let entry = begin + local_index;
                return Err(format!(
                    "Adjacency index out of bounds at entry {entry}: {neighbor} (num_points={num_points})"
                ));
            }
            if neighbor as usize == i {
                return Err(format!("Adjacency list for point {i} contains a self-edge"));
            }
        }
    }

    for i in 0..num_points {
        let begin = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        for &neighbor in &neighbors[begin..end] {
            let reverse_begin = offsets[neighbor as usize] as usize;
            let reverse_end = offsets[neighbor as usize + 1] as usize;
            if neighbors[reverse_begin..reverse_end]
                .binary_search(&(i as u32))
                .is_err()
            {
                return Err(format!(
                    "Adjacency edge {i}->{neighbor} has no reverse edge"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_csr_matches_capped_builder_when_every_edge_fits() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
            glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        ];
        let neighbor_sets = vec![
            vec![0, 1, 2, 2, 3],
            vec![0, 2, 4],
            vec![0, 0, 1, 3, 4],
            vec![0, 2, 4],
            vec![1, 2, 3],
        ];
        let mut unbounded_sets = neighbor_sets.clone();
        let unbounded =
            build_symmetric_csr(&points, &mut unbounded_sets, &AdjacencyConfig::default());
        let mut capped_sets = neighbor_sets;
        let capped = build_symmetric_csr(
            &points,
            &mut capped_sets,
            &AdjacencyConfig {
                max_neighbors: points.len(),
                validate: true,
            },
        );

        assert_eq!(unbounded.offsets, capped.offsets);
        assert_eq!(unbounded.neighbors, capped.neighbors);
    }

    #[test]
    fn test_tetrahedron_adjacency() {
        // 4 points forming a tetrahedron - each point should have 3 neighbors
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.5, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.5, 0.5, 1.0, 1.0),
        ];

        let adjacency = compute_adjacency_default(&points);

        // Each point should have exactly 3 neighbors
        assert_eq!(adjacency.offsets.len(), 5);
        for i in 0..4 {
            let start = adjacency.offsets[i] as usize;
            let end = adjacency.offsets[i + 1] as usize;
            assert_eq!(
                end - start,
                3,
                "Point {} should have 3 neighbors, got {}",
                i,
                end - start
            );
        }

        // Total edges: 4 points * 3 neighbors / 2 = 6 edges, stored bidirectionally = 12
        assert_eq!(adjacency.neighbors.len(), 12);
    }

    #[test]
    fn test_cube_adjacency() {
        // 8 points forming a cube
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 1.0, 1.0),
            glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 1.0, 1.0),
        ];

        let adjacency = compute_adjacency_default(&points);

        // CSR should be valid
        assert_eq!(adjacency.offsets.len(), 9);

        // Each point should have at least 3 neighbors (cube edges + diagonals from tetrahedralization)
        for i in 0..8 {
            let start = adjacency.offsets[i] as usize;
            let end = adjacency.offsets[i + 1] as usize;
            let neighbor_count = end - start;
            assert!(
                neighbor_count >= 3,
                "Point {} should have at least 3 neighbors, got {}",
                i,
                neighbor_count
            );
        }
    }

    #[test]
    fn test_random_points_csr_valid() {
        // Generate random points
        let num_points = 100;
        let mut points = Vec::with_capacity(num_points);

        // Simple pseudo-random sequence for reproducibility
        let mut seed = 12345u64;
        for _ in 0..num_points {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((seed >> 32) as f32) / (u32::MAX as f32);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = ((seed >> 32) as f32) / (u32::MAX as f32);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = ((seed >> 32) as f32) / (u32::MAX as f32);
            points.push(glam::Vec4::new(x, y, z, 1.0));
        }

        let adjacency = compute_adjacency_default(&points);

        // CSR structure should be valid (validation is enabled by default)
        assert_eq!(adjacency.offsets.len(), num_points + 1);
        assert_eq!(adjacency.offsets[0], 0);
        assert_eq!(
            adjacency.offsets[num_points] as usize,
            adjacency.neighbors.len()
        );

        // All neighbor indices should be in bounds
        for &idx in &adjacency.neighbors {
            assert!((idx as usize) < num_points);
        }
    }

    #[test]
    #[should_panic(expected = "need at least 4 points for 3D Delaunay adjacency, got 3")]
    fn test_too_few_points() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.5, 1.0, 0.0, 1.0),
        ];

        compute_adjacency_default(&points);
    }

    #[test]
    fn try_adjacency_reports_too_few_points() {
        let points = vec![glam::Vec4::ZERO; 3];
        assert!(matches!(
            try_compute_adjacency_default(&points),
            Err(AdjacencyError::TooFewPoints { count: 3 })
        ));
    }

    fn neighbors_of(adj: &Adjacency, i: usize) -> Vec<u32> {
        let start = adj.offsets[i] as usize;
        let end = adj.offsets[i + 1] as usize;
        adj.neighbors[start..end].to_vec()
    }

    #[test]
    fn cech_isolated_balls_have_no_edges() {
        // Balls separated by more than the sum of their radii.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(10.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 10.0, 0.0, 1.0),
        ];
        let radii = vec![1.0, 1.0, 1.0];
        let adj = compute_cech_default(&points, &radii);

        assert_eq!(adj.offsets, vec![0, 0, 0, 0]);
        assert!(adj.neighbors.is_empty());
    }

    #[test]
    fn cech_overlapping_balls_emit_bidirectional_edge() {
        // |p_0 - p_1| = 1.0, r_0 + r_1 = 1.5 → overlap.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(10.0, 0.0, 0.0, 1.0),
        ];
        let radii = vec![1.0, 0.5, 0.5];
        let adj = compute_cech_default(&points, &radii);

        assert_eq!(neighbors_of(&adj, 0), vec![1]);
        assert_eq!(neighbors_of(&adj, 1), vec![0]);
        assert!(neighbors_of(&adj, 2).is_empty());
    }

    #[test]
    fn cech_touching_balls_emit_edge() {
        // Exactly touching at the boundary — the predicate is `<=` so we include it.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        ];
        let radii = vec![0.4, 0.6];
        let adj = compute_cech_default(&points, &radii);

        assert_eq!(neighbors_of(&adj, 0), vec![1]);
        assert_eq!(neighbors_of(&adj, 1), vec![0]);
    }

    #[test]
    fn cech_fully_overlapping_cluster_is_complete_graph() {
        // 4 points within a sphere; each ball has radius 2 → all pairs overlap.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        ];
        let radii = vec![2.0; 4];
        let adj = compute_cech_default(&points, &radii);

        for i in 0..4 {
            let mut got = neighbors_of(&adj, i);
            got.sort();
            let expected: Vec<u32> = (0..4u32).filter(|&j| j as usize != i).collect();
            assert_eq!(got, expected, "point {i}");
        }
    }

    #[test]
    fn cech_zero_radii_emit_no_edges() {
        // Degenerate but valid: zero-radius balls touch nothing (except coincident points).
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.1, 0.0, 0.0, 1.0),
        ];
        let radii = vec![0.0, 0.0];
        let adj = compute_cech_default(&points, &radii);
        assert!(adj.neighbors.is_empty());
    }

    #[test]
    fn cech_handles_many_coincident_sites() {
        let points = vec![glam::Vec4::new(1.0, 2.0, 3.0, 1.0); 128];
        let radii = vec![0.0; points.len()];
        let adjacency = compute_cech(
            &points,
            &radii,
            &AdjacencyConfig {
                max_neighbors: usize::MAX,
                validate: true,
            },
        );
        validate_csr(&adjacency.offsets, &adjacency.neighbors, points.len());
        assert_eq!(adjacency.neighbors.len(), points.len() * (points.len() - 1));
        assert!(adjacency
            .offsets
            .windows(2)
            .all(|range| range[1] - range[0] == points.len() as u32 - 1));
    }

    #[test]
    fn cech_worker_partition_preserves_exact_csr() {
        let points = (0..1_152)
            .map(|i| {
                let x = i % 16;
                let y = i / 16 % 12;
                let z = i / (16 * 12);
                let jitter = (i * 17 % 7) as f32 * 0.001;
                glam::Vec4::new(
                    x as f32 * 0.1 + jitter,
                    y as f32 * 0.1 - jitter,
                    z as f32 * 0.1 + jitter,
                    1.0,
                )
            })
            .collect::<Vec<_>>();
        let radii = (0..points.len())
            .map(|i| 0.06 + (i % 5) as f32 * 0.005)
            .collect::<Vec<_>>();
        let config = AdjacencyConfig::default();

        let serial = compute_cech_with_workers(&points, &radii, &config, 1);
        let parallel = compute_cech_with_workers(&points, &radii, &config, 4);

        assert_eq!(parallel.offsets, serial.offsets);
        assert_eq!(parallel.neighbors, serial.neighbors);
    }

    #[test]
    fn cech_radius_bins_match_exhaustive_overlap_graph() {
        let points = (0..257)
            .map(|i| {
                glam::Vec4::new(
                    ((i * 37) % 257) as f32 / 256.0,
                    ((i * 73) % 251) as f32 / 250.0,
                    ((i * 109) % 241) as f32 / 240.0,
                    1.0,
                )
            })
            .collect::<Vec<_>>();
        let radii = (0..points.len())
            .map(|i| {
                if i % 34 == 0 {
                    -0.25
                } else if i % 17 == 0 {
                    0.0
                } else {
                    0.001 * 1.8_f32.powi((i % CECH_RADIUS_BINS) as i32)
                }
            })
            .collect::<Vec<_>>();
        let adjacency = compute_cech_with_workers(&points, &radii, &AdjacencyConfig::default(), 4);

        for (i, point) in points.iter().enumerate() {
            let mut expected = (0..points.len())
                .filter(|&j| {
                    j != i
                        && point.truncate().distance_squared(points[j].truncate())
                            <= (radii[i].max(0.0) + radii[j].max(0.0)).powi(2)
                })
                .map(|j| j as u32)
                .collect::<Vec<_>>();
            expected.sort_unstable();
            assert_eq!(neighbors_of(&adjacency, i), expected, "row {i}");
        }
    }

    #[test]
    fn cech_dispatch_via_model() {
        // PointCloudModel::compute_adjacency_default chooses Čech when radii is Some.
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(10.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 10.0, 0.0, 1.0),
        ];
        let mut model = crate::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: Some(vec![1.0, 0.5, 0.5, 0.5]),
            surface_normals: None,
            points,
        };
        model.compute_adjacency_default();
        let adj = model.adjacency.as_ref().unwrap();
        // Only the first two balls overlap (|0-1| = 1 ≤ 1.0 + 0.5).
        assert_eq!(neighbors_of(adj, 0), vec![1]);
        assert_eq!(neighbors_of(adj, 1), vec![0]);
        assert!(neighbors_of(adj, 2).is_empty());
        assert!(neighbors_of(adj, 3).is_empty());
    }

    #[test]
    fn spring_relax_increases_min_pairwise_distance() {
        // Eight points: four at the corners of a unit cube + four
        // tightly-clustered near the origin. Spring relaxation pushes the cluster
        // apart, raising the smallest pairwise distance in the cloud.
        let pts = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
            glam::Vec4::new(0.05, 0.05, 0.05, 1.0),
            glam::Vec4::new(0.06, 0.04, 0.05, 1.0),
            glam::Vec4::new(0.04, 0.06, 0.05, 1.0),
            glam::Vec4::new(0.05, 0.05, 0.06, 1.0),
        ];
        let min_pairwise = |pts: &[glam::Vec4]| -> f32 {
            let mut best = f32::INFINITY;
            for i in 0..pts.len() {
                for j in (i + 1)..pts.len() {
                    let d = (glam::Vec3::new(pts[i].x, pts[i].y, pts[i].z)
                        - glam::Vec3::new(pts[j].x, pts[j].y, pts[j].z))
                    .length();
                    if d < best {
                        best = d;
                    }
                }
            }
            best
        };
        let before = min_pairwise(&pts);

        let mut model = crate::PointCloudModel {
            points: pts,
            sh_coefficients: vec![0.0; 8 * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            surface_normals: None,
        };
        spring_relax(&mut model, 8, 0.5);
        let after = min_pairwise(&model.points);
        assert!(
            after > before * 2.0,
            "spring_relax should spread the cluster; min pairwise {before} → {after}"
        );
    }

    #[test]
    fn radii_from_nearest_neighbour_matches_pair_distance() {
        // Three colinear points 1, 2, 4 units apart. Nearest neighbours:
        // p0 → p1 (dist 1), p1 → p0 (dist 1), p2 → p1 (dist 2).
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(3.0, 0.0, 0.0, 1.0),
        ];
        let radii = radii_from_nearest_neighbour(&points, 0.5);
        assert!((radii[0] - 0.5).abs() < 1e-5);
        assert!((radii[1] - 0.5).abs() < 1e-5);
        assert!((radii[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn radii_from_nearest_neighbour_handles_many_coincident_sites() {
        let mut points = vec![glam::Vec4::new(0.0, -0.0, 0.0, 1.0); 64];
        points.push(glam::Vec4::new(2.0, 0.0, 0.0, 1.0));
        let radii = radii_from_nearest_neighbour(&points, 0.5);
        assert_eq!(radii.len(), points.len());
        assert!(radii.iter().all(|&radius| (radius - 1.0).abs() < 1e-6));
    }

    #[test]
    fn radii_from_nearest_neighbour_handles_planar_grids() {
        let points = (0..32)
            .flat_map(|x| (0..32).map(move |y| glam::Vec4::new(x as f32, y as f32, 0.0, 1.0)))
            .collect::<Vec<_>>();
        let radii = radii_from_nearest_neighbour(&points, 0.5);
        assert_eq!(radii.len(), points.len());
        assert!(radii.iter().all(|&radius| (radius - 0.5).abs() < 1.0e-6));
    }

    #[test]
    fn powerfoam_reference_radii_average_eight_samples_including_self() {
        let points = (0..10)
            .map(|index| glam::Vec4::new(index as f32, 0.0, 0.0, 1.0))
            .collect::<Vec<_>>();
        let radii = radii_from_powerfoam_reference(&points, &[]);
        assert!((radii[0] - 3.5).abs() < 1.0e-6);
        assert!((radii[4] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn powerfoam_reference_radii_apply_visible_camera_cap() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 10.0, 1.0),
            glam::Vec4::new(10.0, 0.0, 10.0, 1.0),
            glam::Vec4::new(-10.0, 0.0, 10.0, 1.0),
            glam::Vec4::new(0.0, 10.0, 10.0, 1.0),
            glam::Vec4::new(0.0, -10.0, 10.0, 1.0),
            glam::Vec4::new(10.0, 10.0, 10.0, 1.0),
            glam::Vec4::new(-10.0, 10.0, 10.0, 1.0),
            glam::Vec4::new(100.0, 0.0, 10.0, 1.0),
        ];
        let camera = CameraParams::looking_at(
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            std::f32::consts::FRAC_PI_2,
            1.0,
            100.0,
        );
        let radii = radii_from_powerfoam_reference(&points, &[camera]);
        assert!((radii[0] - 1.0).abs() < 1.0e-6);
        assert!(radii[7] > 1.0, "off-camera support should not be capped");
    }

    #[test]
    #[should_panic(expected = "radii.len()")]
    fn cech_panics_on_radii_length_mismatch() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        ];
        let radii = vec![1.0];
        compute_cech_default(&points, &radii);
    }

    #[test]
    fn test_max_neighbors_clamping() {
        // Create a dense cluster where some points may have many neighbors
        let mut points = Vec::new();

        // Central point
        points.push(glam::Vec4::new(0.0, 0.0, 0.0, 1.0));

        // Surrounding points in a sphere
        for i in 0..100 {
            let theta = (i as f32) * 0.2;
            let phi = (i as f32) * 0.1;
            let r = 1.0;
            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            points.push(glam::Vec4::new(x, y, z, 1.0));
        }

        let config = AdjacencyConfig {
            max_neighbors: 10,
            validate: true,
        };
        let adjacency = compute_adjacency(&points, &config);

        // No point should have more than max_neighbors
        for i in 0..points.len() {
            let start = adjacency.offsets[i] as usize;
            let end = adjacency.offsets[i + 1] as usize;
            let neighbor_count = end - start;
            assert!(
                neighbor_count <= config.max_neighbors,
                "Point {} has {} neighbors, exceeds max {}",
                i,
                neighbor_count,
                config.max_neighbors
            );
            for &neighbor in &adjacency.neighbors[start..end] {
                let reverse_start = adjacency.offsets[neighbor as usize] as usize;
                let reverse_end = adjacency.offsets[neighbor as usize + 1] as usize;
                assert!(adjacency.neighbors[reverse_start..reverse_end].contains(&(i as u32)));
            }
        }
    }

    #[test]
    fn compute_knn_returns_symmetric_graph_with_at_least_k_neighbours() {
        // Jittered grid — kiddo's default bucket size panics on exactly
        // co-axial points (which a pure integer grid would produce), so
        // perturb each coordinate slightly.
        let mut points = Vec::new();
        for ix in 0..6 {
            for iy in 0..6 {
                for iz in 0..6 {
                    let jitter = ((ix * 31 + iy * 7 + iz) as f32) * 1e-3;
                    points.push(glam::Vec4::new(
                        ix as f32 + jitter,
                        iy as f32 + jitter * 0.7,
                        iz as f32 + jitter * 1.3,
                        1.0,
                    ));
                }
            }
        }
        let k = 8usize;
        let adj = compute_knn(&points, k);

        // Each point has at least k neighbours (more, after symmetrising).
        for i in 0..points.len() {
            let a = adj.offsets[i] as usize;
            let b = adj.offsets[i + 1] as usize;
            assert!(
                b - a >= k,
                "point {i} has only {} neighbours, want >= {k}",
                b - a
            );
        }

        // Symmetric: if j ∈ N(i) then i ∈ N(j).
        for i in 0..points.len() {
            let a = adj.offsets[i] as usize;
            let b = adj.offsets[i + 1] as usize;
            for &j in &adj.neighbors[a..b] {
                let ja = adj.offsets[j as usize] as usize;
                let jb = adj.offsets[j as usize + 1] as usize;
                assert!(
                    adj.neighbors[ja..jb].contains(&(i as u32)),
                    "edge {i} → {j} present but reverse missing",
                );
            }
        }
    }

    #[cfg(feature = "qhull")]
    #[test]
    fn compute_adjacency_qhull_largely_agrees_with_simple_delaunay_on_random_input() {
        // The two backends agree on >95% of edges for typical
        // (non-degenerate) inputs; the residual differences come from
        // tie-breaking among ~co-spherical quadruples. Final
        // correctness for our use is checked end-to-end via the
        // appearance-fit smoke tests in blade-volume-train.
        let mut state: u64 = 0xCAFE_BABE_F00D_D00D;
        let mut next_f = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32 as f32) / (u32::MAX >> 1) as f32
        };
        let mut points = Vec::with_capacity(80);
        for _ in 0..80 {
            points.push(glam::Vec4::new(next_f(), next_f(), next_f(), 1.0));
        }
        let a = compute_adjacency_default(&points);
        let b = compute_adjacency_qhull_default(&points);
        // Count agreement at the (i, neighbour) pair level.
        let mut shared = 0usize;
        let mut only_a = 0usize;
        let mut only_b = 0usize;
        for i in 0..points.len() {
            let sa: std::collections::HashSet<u32> = a.neighbors
                [a.offsets[i] as usize..a.offsets[i + 1] as usize]
                .iter()
                .copied()
                .collect();
            let sb: std::collections::HashSet<u32> = b.neighbors
                [b.offsets[i] as usize..b.offsets[i + 1] as usize]
                .iter()
                .copied()
                .collect();
            shared += sa.intersection(&sb).count();
            only_a += sa.difference(&sb).count();
            only_b += sb.difference(&sa).count();
        }
        let total = shared + only_a + only_b;
        let agree = shared as f64 / total as f64;
        eprintln!(
            "qhull vs simple_delaunay: shared={shared} only_simple={only_a} only_qhull={only_b} agree={:.3}",
            agree
        );
        assert!(
            agree > 0.9,
            "expected >90% edge agreement, got {:.3}",
            agree
        );
    }

    #[cfg(feature = "qhull")]
    #[test]
    fn compute_adjacency_qhull_handles_a_jittered_grid_without_panicking() {
        // Just check it doesn't blow up — the exact adjacency on a
        // near-degenerate input depends on tie-breaking, see the
        // random-input test for the strict comparison.
        let mut points = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    let j = ((ix * 31 + iy * 7 + iz) as f32) * 1e-3;
                    points.push(glam::Vec4::new(
                        ix as f32 + j,
                        iy as f32 + j * 0.7,
                        iz as f32 + j * 1.3,
                        1.0,
                    ));
                }
            }
        }
        let adj = compute_adjacency_qhull_default(&points);
        // Every interior point should have at least 4 neighbours
        // (the four tetrahedral vertices of one of its incident tets).
        for i in 0..points.len() {
            let a = adj.offsets[i] as usize;
            let b = adj.offsets[i + 1] as usize;
            assert!(
                b - a >= 4,
                "point {i} has only {} neighbours after qhull build",
                b - a,
            );
        }
    }

    #[cfg(feature = "qhull")]
    #[test]
    fn compute_adjacency_qhull_can_rebuild_repeatedly() {
        let mut points = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    let j = ((ix * 31 + iy * 7 + iz) as f32) * 1e-3;
                    points.push(glam::Vec4::new(
                        ix as f32 + j,
                        iy as f32 + j * 0.7,
                        iz as f32 + j * 1.3,
                        1.0,
                    ));
                }
            }
        }
        for _ in 0..16 {
            let adjacency = compute_adjacency_qhull_default(&points);
            assert_eq!(adjacency.offsets.len(), points.len() + 1);
        }
    }

    #[test]
    fn compute_knn_picks_geometrically_nearest_neighbours_for_a_grid() {
        // Same jittering as the symmetry test — kiddo's default bucket
        // would otherwise panic on the co-axial integer grid.
        let mut points = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    let jitter = ((ix * 31 + iy * 7 + iz) as f32) * 1e-3;
                    points.push(glam::Vec4::new(
                        ix as f32 + jitter,
                        iy as f32 + jitter * 0.7,
                        iz as f32 + jitter * 1.3,
                        1.0,
                    ));
                }
            }
        }
        let adj = compute_knn(&points, 6);
        let center_idx = (2 * 5 + 2) * 5 + 2; // (2, 2, 2)
        let a = adj.offsets[center_idx] as usize;
        let b = adj.offsets[center_idx + 1] as usize;
        let neighbours = &adj.neighbors[a..b];
        // The 6 face neighbours must all be present (possibly with extras
        // from corners coming in via symmetrisation).
        let index = |x, y, z| ((x * 5 + y) * 5 + z) as u32;
        let want = [
            index(1, 2, 2), // x-1
            index(3, 2, 2), // x+1
            index(2, 1, 2), // y-1
            index(2, 3, 2), // y+1
            index(2, 2, 1), // z-1
            index(2, 2, 3), // z+1
        ];
        for w in want {
            assert!(
                neighbours.contains(&w),
                "center missing face neighbour {w}: have {neighbours:?}",
            );
        }
    }
}
