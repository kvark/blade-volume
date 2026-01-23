//! Pure-Rust synthetic RadFoam fixture generator (chain topology).
//!
//! This module builds a small `RadFoamModel` entirely in-memory with an adjacency
//! that is guaranteed to be traversable for rays pointing along +Z.
//!
//! Why:
//! - A true RadFoam adjacency comes from a 3D Delaunay triangulation (Voronoi neighbors).
//! - Generating that robustly without native deps is non-trivial.
//! - For correctness iteration we primarily need a fixture where:
//!   - GPU and CPU see the *same* buffers
//!   - traversal is well-posed (no immediate dead-ends for the test rays)
//!   - integration produces non-zero, deterministic outputs
//!
//! The chain fixture provides that:
//! - Points are laid out along Z: p_i = (0,0,i*dz)
//! - Each point i connects to i-1 and i+1 (CSR adjacency)
//! - For a +Z ray, each cell has at least one neighbor with dp > 0,
//!   so traversal progresses forward and produces positive segment lengths.

use blade_volume as vol;

/// Build a chain fixture.
///
/// - `n`: number of points (must be >= 2)
/// - `dz`: spacing along +Z (must be > 0)
/// - `density`: per-cell density (last scalar in packed attribute row)
/// - `sh_degree`: spherical harmonics degree for packed attribute layout
/// - `dc`: RGB DC coefficients for component 0 (TEMPORARY; helps non-black output if SH is used)
///
/// Packed attribute row layout matches `RadFoamModel`:
/// - `attr_dim = 1 + 3 * (1 + sh_degree)^2`
/// - coefficients are interleaved RGB per SH component:
///     [R_c0, G_c0, B_c0, R_c1, G_c1, B_c1, ...]
/// - last scalar is `density`
pub fn make_chain_model(
    n: usize,
    dz: f32,
    density: f32,
    sh_degree: usize,
    dc: glam::Vec3,
) -> vol::RadFoamModel {
    assert!(n >= 2, "chain requires at least 2 points");
    assert!(dz.is_finite() && dz > 0.0, "dz must be finite and > 0");
    assert!(
        density.is_finite() && density >= 0.0,
        "density must be finite and >= 0"
    );

    let attr_dim = vol::RadFoamModel::attribute_dim(sh_degree);
    let comps = vol::get_sh_component_count(sh_degree);

    // Points along +Z.
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        points.push(glam::Vec3::new(0.0, 0.0, (i as f32) * dz));
    }

    // CSR adjacency for chain:
    // - endpoints have 1 neighbor
    // - interior points have 2 neighbors
    //
    // adjacency_offsets[i+1] = cumulative neighbor count up to i
    let mut point_adjacency_offsets = vec![0u32; n + 1];
    let mut point_adjacency: Vec<u32> = Vec::with_capacity((2 * n).saturating_sub(2));

    let mut running: u32 = 0;
    for i in 0..n {
        if i > 0 {
            point_adjacency.push((i - 1) as u32);
            running += 1;
        }
        if i + 1 < n {
            point_adjacency.push((i + 1) as u32);
            running += 1;
        }
        point_adjacency_offsets[i + 1] = running;
    }

    // Packed attributes.
    //
    // - Set DC (component 0) to `dc` (3 scalars).
    // - All other SH coefficients to 0.
    // - Density as last scalar of each row.
    let mut attributes = vec![0.0f32; n * attr_dim];
    for i in 0..n {
        let base = i * attr_dim;

        // component 0 (DC)
        attributes[base + 0] = dc.x;
        attributes[base + 1] = dc.y;
        attributes[base + 2] = dc.z;

        // components 1..(comps-1) remain 0
        // density last
        attributes[base + (3 * comps)] = density;
    }

    vol::RadFoamModel {
        points,
        attributes,
        sh_degree,
        point_adjacency,
        point_adjacency_offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_fixture_basic_invariants() {
        let m = make_chain_model(5, 1.0, 0.1, 1, glam::Vec3::splat(0.0));
        assert_eq!(m.points.len(), 5);
        assert_eq!(m.point_adjacency_offsets.len(), 6);

        // Expected neighbor counts: [1,2,2,2,1] => offsets [0,1,3,5,7,8]
        assert_eq!(m.point_adjacency_offsets, vec![0, 1, 3, 5, 7, 8]);
        assert_eq!(m.point_adjacency.len(), 8);

        // In-bounds adjacency
        for &idx in &m.point_adjacency {
            assert!((idx as usize) < m.points.len());
        }

        // Attribute length
        let attr_dim = vol::RadFoamModel::attribute_dim(m.sh_degree);
        assert_eq!(m.attributes.len(), m.points.len() * attr_dim);
    }
}
