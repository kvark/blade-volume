/*!
Pure-Rust synthetic RadFoam fixture generator (branching topology).

This fixture is designed to exercise **multi-neighbor face selection** in the RadFoam
Voronoi-cell traversal kernel without requiring a true 3D Delaunay triangulation
(and without native dependencies).

Why a branching fixture?
- The simple chain fixture only gives each interior node 2 neighbors, which is great
  for validating basic stepping and integration but doesn't stress the "pick the next
  face with the smallest t among many candidates" logic.
- This branching fixture creates nodes with higher degree (multiple outgoing directions),
  which forces both CPU and GPU tracers to:
  - iterate over multiple faces,
  - compute candidate intersection parameters `t`,
  - choose the minimum `t` for those with `dp > 0`.

Topology
--------
We build a "spine" along +Z with a branching cluster at each spine node:

- Spine nodes: S_i = (0, 0, i*dz), i=0..spine_len-1
- For each spine node i, we create `branch_degree` branch nodes:
    B_{i,j} = S_i + (R*cos(theta_j), R*sin(theta_j), branch_z_offset)
  where theta_j are evenly spaced angles around the circle.

Adjacency:
- Each spine node S_i connects to:
  - S_{i-1}, S_{i+1} (where applicable)
  - all its branch nodes B_{i,*}
- Each branch node connects back to its parent spine node S_i only.

This ensures:
- Some nodes have high degree (spine nodes).
- For rays pointing generally along +Z, `dp > 0` will exist for multiple neighbors
  (especially among branches with +Z offset and next spine node), producing a meaningful
  selection problem.

Attributes
----------
We construct a `RadFoamModel` compatible with the engine:
- packed attribute row length: `attr_dim = 1 + 3*(1+sh_degree)^2`
- coefficient layout: interleaved RGB per SH component:
    [R_c0, G_c0, B_c0, R_c1, G_c1, B_c1, ...]
- density is the last scalar.

For most tests:
- set `sh_degree = 0` to validate DC-only SH evaluation or constant color.
- use non-zero DC to keep renders non-black if SH is enabled.

Usage
-----
This module is intended to be used by GPU-vs-CPU tests to increase coverage.
Example:
- Generate with `make_branching_model(...)`
- Use a camera at z < 0 looking towards +Z
- Use start point = 0 (S_0) or nearest-neighbor start point.

Note:
This is not a true Voronoi/Delaunay adjacency, but it is deterministic and stresses
the selection logic. A "real" adjacency fixture can be swapped in later.
*/

use blade_volume as vol;

/// Parameters controlling the branching fixture shape.
#[derive(Clone, Copy, Debug)]
pub struct BranchingParams {
    /// Number of spine nodes along +Z. Must be >= 2.
    pub spine_len: usize,
    /// Spacing between spine nodes along +Z. Must be > 0.
    pub dz: f32,

    /// Number of branch nodes per spine node. Must be >= 1.
    pub branch_degree: usize,
    /// Radial distance of branch nodes in XY plane. Must be > 0.
    pub branch_radius: f32,
    /// Z offset of branch nodes relative to parent spine node.
    /// Recommend > 0 to ensure dp>0 candidates exist for +Z rays.
    pub branch_z_offset: f32,

    /// Per-cell density to pack into attributes. Must be finite and >= 0.
    pub density: f32,
    /// SH degree used for packed attribute layout.
    pub sh_degree: usize,
    /// DC (component 0) RGB coefficients (used if SH is enabled).
    pub dc: glam::Vec3,
}

impl Default for BranchingParams {
    fn default() -> Self {
        Self {
            spine_len: 64,
            dz: 0.05,
            branch_degree: 6,
            branch_radius: 0.10,
            branch_z_offset: 0.02,
            density: 0.1,
            sh_degree: 0,
            dc: glam::Vec3::splat(0.1),
        }
    }
}

/// Build a branching synthetic RadFoam model.
///
/// Returns a `RadFoamModel` with points, packed attributes, and CSR adjacency.
pub fn make_branching_model(params: BranchingParams) -> vol::RadFoamModel {
    assert!(params.spine_len >= 2, "spine_len must be >= 2");
    assert!(
        params.dz.is_finite() && params.dz > 0.0,
        "dz must be finite and > 0"
    );
    assert!(params.branch_degree >= 1, "branch_degree must be >= 1");
    assert!(
        params.branch_radius.is_finite() && params.branch_radius > 0.0,
        "branch_radius must be finite and > 0"
    );
    assert!(
        params.branch_z_offset.is_finite(),
        "branch_z_offset must be finite"
    );
    assert!(
        params.density.is_finite() && params.density >= 0.0,
        "density must be finite and >= 0"
    );

    let comps = vol::get_sh_component_count(params.sh_degree);
    let attr_dim = vol::RadFoamModel::attribute_dim(params.sh_degree);

    // Layout:
    // - First `spine_len` points are spine nodes S_i
    // - Then for each spine node i, we append `branch_degree` branch nodes B_{i,*}
    let n_spine = params.spine_len;
    let n_branch = params.spine_len * params.branch_degree;
    let n_total = n_spine + n_branch;

    // Precompute angles for branch nodes.
    let mut angles = Vec::with_capacity(params.branch_degree);
    for j in 0..params.branch_degree {
        let t = (j as f32) / (params.branch_degree as f32);
        angles.push(t * std::f32::consts::TAU);
    }

    // Points
    let mut points: Vec<glam::Vec3> = Vec::with_capacity(n_total);

    // Spine points
    for i in 0..n_spine {
        points.push(glam::Vec3::new(0.0, 0.0, (i as f32) * params.dz));
    }

    // Branch points
    // Branch index mapping:
    //   branch_linear_index(i,j) = n_spine + i*branch_degree + j
    for i in 0..n_spine {
        let spine = points[i];
        for &a in angles.iter() {
            let x = params.branch_radius * a.cos();
            let y = params.branch_radius * a.sin();
            let z = params.branch_z_offset;
            points.push(spine + glam::Vec3::new(x, y, z));
        }
    }
    debug_assert_eq!(points.len(), n_total);

    // Adjacency (build as Vec<Vec<u32>> then CSR)
    let mut neigh: Vec<Vec<u32>> = vec![Vec::new(); n_total];

    // Spine adjacency
    for i in 0..n_spine {
        let idx = i as u32;

        // Connect to previous/next spine node
        if i > 0 {
            neigh[i].push((i - 1) as u32);
        }
        if i + 1 < n_spine {
            neigh[i].push((i + 1) as u32);
        }

        // Connect to its branch nodes
        for j in 0..params.branch_degree {
            let b = (n_spine + i * params.branch_degree + j) as u32;
            neigh[i].push(b);
            // Branch node connects back to spine
            neigh[b as usize].push(idx);
        }
    }

    // Optional: sort & dedup neighbors for determinism
    for n in neigh.iter_mut() {
        n.sort_unstable();
        n.dedup();
    }

    // CSR
    let mut point_adjacency_offsets: Vec<u32> = Vec::with_capacity(n_total + 1);
    let mut point_adjacency: Vec<u32> = Vec::new();

    point_adjacency_offsets.push(0);
    let mut running: u32 = 0;
    for i in 0..n_total {
        let list = &neigh[i];
        point_adjacency.extend_from_slice(list);
        running += list.len() as u32;
        point_adjacency_offsets.push(running);
    }

    // Packed attributes: set DC, optional degree-1 term(s), set density as last scalar
    let mut attributes = vec![0.0f32; n_total * attr_dim];
    for i in 0..n_total {
        let base = i * attr_dim;

        // DC component (component 0)
        attributes[base + 0] = params.dc.x;
        attributes[base + 1] = params.dc.y;
        attributes[base + 2] = params.dc.z;

        // If sh_degree >= 1, populate component 1 with a small non-zero coefficient so that
        // SH evaluation is observable in GPU-vs-CPU tests.
        //
        // Layout is interleaved RGB per SH component:
        //   comp0:  base+0..2
        //   comp1:  base+3..5
        //   comp2:  base+6..8
        //   comp3:  base+9..11
        //   comp4:  base+12..14  (degree 2 starts here)
        //   comp9:  base+27..29  (degree 3 starts here)
        if params.sh_degree >= 1 {
            // A small, deterministic coefficient; keep it modest to avoid saturating.
            // Only the X channel is non-zero to make direction dependence easier to spot.
            attributes[base + 3] = 0.05;
            attributes[base + 4] = 0.0;
            attributes[base + 5] = 0.0;
        }

        // If sh_degree >= 2, also populate one degree-2 coefficient so the degree-2
        // code path is exercised meaningfully. We use component 4 (x*y term in our WGSL)
        // at base+12..14.
        if params.sh_degree >= 2 {
            attributes[base + 12] = 0.03;
            attributes[base + 13] = 0.0;
            attributes[base + 14] = 0.0;
        }

        // If sh_degree >= 3, populate one degree-3 coefficient so the degree-3
        // code path is exercised meaningfully. We use component 9 (the first degree-3 term
        // in our WGSL) at base+27..29.
        if params.sh_degree >= 3 {
            attributes[base + 27] = 0.02;
            attributes[base + 28] = 0.0;
            attributes[base + 29] = 0.0;
        }

        // Density last scalar (index 3*comps)
        attributes[base + (3 * comps)] = params.density;
    }

    vol::RadFoamModel {
        points,
        attributes,
        sh_degree: params.sh_degree,
        point_adjacency,
        point_adjacency_offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branching_fixture_invariants_hold() {
        let m = make_branching_model(BranchingParams {
            spine_len: 5,
            dz: 1.0,
            branch_degree: 4,
            branch_radius: 0.5,
            branch_z_offset: 0.25,
            density: 0.1,
            sh_degree: 0,
            dc: glam::Vec3::splat(0.1),
        });

        // Count: spine 5 + branches 5*4 = 20 => total 25
        assert_eq!(m.points.len(), 25);
        assert_eq!(m.point_adjacency_offsets.len(), 26);

        // Offsets should be monotonic and end at adjacency length
        for w in m.point_adjacency_offsets.windows(2) {
            assert!(w[1] >= w[0]);
        }
        assert_eq!(
            *m.point_adjacency_offsets.last().unwrap() as usize,
            m.point_adjacency.len()
        );

        // Adjacency in-bounds
        let n = m.points.len();
        for &idx in &m.point_adjacency {
            assert!((idx as usize) < n);
        }

        // Attribute length matches N * attr_dim
        let attr_dim = vol::RadFoamModel::attribute_dim(m.sh_degree);
        assert_eq!(m.attributes.len(), n * attr_dim);

        // Spot-check: a branch node should have exactly 1 neighbor (its parent spine)
        // Branch nodes start at index spine_len=5.
        let branch0 = 5usize;
        let a0 = m.point_adjacency_offsets[branch0] as usize;
        let a1 = m.point_adjacency_offsets[branch0 + 1] as usize;
        assert_eq!(a1 - a0, 1);

        // A spine node should have >= 1 + branch_degree neighbors (plus prev/next)
        let spine2 = 2usize;
        let s0 = m.point_adjacency_offsets[spine2] as usize;
        let s1 = m.point_adjacency_offsets[spine2 + 1] as usize;
        assert!((s1 - s0) >= 1 + 4);
    }
}
