//! Automatic adjacency computation via 3D Delaunay tetrahedralization.
//!
//! Computes Voronoi neighbors from point positions using 3D Delaunay tetrahedralization.
//! Delaunay edges correspond to Voronoi neighbor pairs.

use crate::Adjacency;
use simple_delaunay_lib::delaunay_3d::{
    delaunay_struct_3d::DelaunayStructure3D, simplicial_struct_3d::Node,
};

/// Configuration for adjacency computation.
#[derive(Clone, Debug)]
pub struct AdjacencyConfig {
    /// Maximum number of neighbors per point (default: 64).
    pub max_neighbors: usize,
    /// Whether to validate the resulting CSR structure (default: true).
    pub validate: bool,
}

impl Default for AdjacencyConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 64,
            validate: true,
        }
    }
}

/// Computes adjacency from point positions using Delaunay tetrahedralization.
///
/// # Algorithm
/// 1. Convert `Vec4` positions (xyz) to `[f64; 3]` format
/// 2. Compute 3D Delaunay tetrahedralization
/// 3. Extract edges from tetrahedra (Delaunay edges = Voronoi neighbor pairs)
/// 4. Build CSR adjacency: sort, dedup, clamp to max_neighbors
/// 5. Optionally validate using CSR invariants
///
/// # Panics
/// Panics if fewer than 4 points are provided (minimum for a tetrahedron).
pub fn compute_adjacency(points: &[glam::Vec4], config: &AdjacencyConfig) -> Adjacency {
    let num_points = points.len();

    if num_points < 4 {
        panic!(
            "Cannot compute adjacency with fewer than 4 points (got {})",
            num_points
        );
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
        .expect("Delaunay tetrahedralization failed");

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

    // Build CSR: sort, dedup, clamp to max_neighbors
    let mut offsets = Vec::with_capacity(num_points + 1);
    let mut neighbors = Vec::new();
    offsets.push(0u32);

    for neighbor_list in neighbor_sets.iter_mut() {
        // Sort and remove duplicates
        neighbor_list.sort_unstable();
        neighbor_list.dedup();

        // Clamp to max_neighbors
        let count = neighbor_list.len().min(config.max_neighbors);
        neighbors.extend_from_slice(&neighbor_list[..count]);
        offsets.push(neighbors.len() as u32);
    }

    // Validate CSR structure if requested
    if config.validate {
        validate_csr(&offsets, &neighbors, num_points);
    }

    Adjacency { neighbors, offsets }
}

/// Computes adjacency with default configuration.
pub fn compute_adjacency_default(points: &[glam::Vec4]) -> Adjacency {
    compute_adjacency(points, &AdjacencyConfig::default())
}

/// Validates a CSR adjacency structure.
///
/// # Panics
/// Panics if the CSR structure is invalid.
pub fn validate_csr(offsets: &[u32], neighbors: &[u32], num_points: usize) {
    // Validate CSR offsets
    if offsets.is_empty() || offsets.len() != num_points + 1 {
        panic!(
            "Invalid adjacency_offsets length: expected {}, got {}",
            num_points + 1,
            offsets.len()
        );
    }
    if offsets[0] != 0 {
        panic!(
            "Invalid adjacency_offsets[0]: expected 0, got {}",
            offsets[0]
        );
    }
    let last = offsets[num_points] as usize;
    if last != neighbors.len() {
        panic!(
            "Invalid adjacency_offsets[N]: expected {}, got {}",
            neighbors.len(),
            last
        );
    }
    for i in 0..num_points {
        let a = offsets[i] as usize;
        let b = offsets[i + 1] as usize;
        if b < a || b > neighbors.len() {
            panic!("Invalid adjacency offset range for point {i}: [{a}, {b})");
        }
    }

    // Validate adjacency indices are in-bounds
    for (k, &idx) in neighbors.iter().enumerate() {
        if (idx as usize) >= num_points {
            panic!("Adjacency index out of bounds at entry {k}: {idx} (num_points={num_points})");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[should_panic(expected = "Cannot compute adjacency with fewer than 4 points")]
    fn test_too_few_points() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.5, 1.0, 0.0, 1.0),
        ];

        compute_adjacency_default(&points);
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
        }
    }
}
