#![allow(irrefutable_let_patterns)]

mod adjacency;
mod camera;
pub mod gpu;
mod scene;
pub mod shaders;
mod shape;
pub mod trace;

pub mod io;

pub use adjacency::{
    compute_adjacency, compute_adjacency_default, compute_adjacency_qhull,
    compute_adjacency_qhull_default, compute_cech, compute_cech_default, compute_knn, lloyd_relax,
    radii_from_nearest_neighbour, AdjacencyConfig,
};
pub use camera::CameraParams;
pub use gpu::{GaussianGpuCloud, InitParameters, RadFoamGpuCloud};
pub use scene::{
    GpuTransform, ObjectBounds, ObjectHandle, ObjectType, Scene, SceneRenderData, Transform,
};
pub use shape::Icosahedron;

pub const fn get_sh_component_count(degree: usize) -> usize {
    (1 + degree) * (1 + degree)
}
pub const fn get_sh_degree(count: usize) -> usize {
    if count <= 1 {
        0
    } else if count <= 4 {
        1
    } else if count <= 9 {
        2
    } else {
        3
    }
}

pub const MAX_SH_DEGREE: usize = 3;
pub const MAX_SH_COMPONENTS: usize = get_sh_component_count(MAX_SH_DEGREE);

// ============================================================================
// Unified Point Cloud Model
// ============================================================================

/// Gaussian ellipsoid transforms: rotation (quat) + scale (vec3).
#[derive(Clone)]
pub struct Transforms {
    /// Rotation quaternion per point.
    pub rotations: Vec<glam::Quat>,
    /// Scale per point (in log-space from PLY, exponentiated for use).
    pub scales: Vec<glam::Vec3>,
}

/// CSR adjacency structure for Voronoi cell traversal.
#[derive(Clone)]
pub struct Adjacency {
    /// Flattened neighbor indices.
    pub neighbors: Vec<u32>,
    /// CSR offsets (N+1 entries where N = number of points).
    pub offsets: Vec<u32>,
}

/// Unified point cloud model.
///
/// Core data common to all formats:
/// - `points`: position (xyz) + density/opacity (w) packed as Vec4
/// - `sh_coefficients`: spherical harmonics coefficients
/// - `sh_degree`: SH basis degree (0-3)
///
/// Optional extensions:
/// - `transforms`: rotation + scale for Gaussian ellipsoids
/// - `adjacency`: CSR adjacency for RadFoam Voronoi traversal
/// - `radii`: per-point weight/radius for Power-Foam-style power diagrams
#[derive(Clone)]
pub struct PointCloudModel {
    /// Position (xyz) + density/opacity (w) for each point.
    pub points: Vec<glam::Vec4>,

    /// Packed SH coefficients: 3 floats (RGB) per SH component, per point.
    /// Layout: `[p0_c0_r, p0_c0_g, p0_c0_b, p0_c1_r, ..., p1_c0_r, ...]`
    /// Length: `N * 3 * sh_component_count(sh_degree)`
    pub sh_coefficients: Vec<f32>,

    /// Spherical harmonics degree (0-3).
    pub sh_degree: usize,

    /// Optional: Gaussian ellipsoid transforms (rotation + scale).
    pub transforms: Option<Transforms>,

    /// Optional: RadFoam CSR adjacency.
    pub adjacency: Option<Adjacency>,

    /// Optional: per-point radius/weight. When `Some`, downstream code is free to treat
    /// the cloud as a power diagram (Power Foam) rather than a plain Voronoi diagram.
    /// Length must equal `points.len()`.
    pub radii: Option<Vec<f32>>,
}

impl PointCloudModel {
    /// Validate all cross-field lengths and CSR indices.
    ///
    /// Public fields intentionally keep model construction lightweight, so IO
    /// and GPU boundaries call this before indexing parallel arrays.
    pub fn validate(&self) -> Result<(), String> {
        let count = self.points.len();
        if self.sh_degree > MAX_SH_DEGREE {
            return Err(format!(
                "SH degree {} exceeds supported degree {MAX_SH_DEGREE}",
                self.sh_degree
            ));
        }
        let expected_sh = count * 3 * self.sh_component_count();
        if self.sh_coefficients.len() != expected_sh {
            return Err(format!(
                "SH coefficient length is {}, expected {expected_sh}",
                self.sh_coefficients.len()
            ));
        }
        if let Some(ref transforms) = self.transforms {
            if transforms.rotations.len() != count || transforms.scales.len() != count {
                return Err(format!(
                    "transform lengths are rotations={} scales={}, expected {count}",
                    transforms.rotations.len(),
                    transforms.scales.len()
                ));
            }
        }
        if let Some(ref radii) = self.radii {
            if radii.len() != count {
                return Err(format!(
                    "radius length is {}, expected {count}",
                    radii.len()
                ));
            }
        }
        if let Some(ref adjacency) = self.adjacency {
            if adjacency.offsets.len() != count + 1 {
                return Err(format!(
                    "adjacency offset length is {}, expected {}",
                    adjacency.offsets.len(),
                    count + 1
                ));
            }
            if adjacency.offsets.first().copied() != Some(0) {
                return Err("adjacency offsets must start at zero".to_string());
            }
            if adjacency.offsets.last().copied() != Some(adjacency.neighbors.len() as u32) {
                return Err("last adjacency offset must equal neighbor count".to_string());
            }
            if adjacency
                .offsets
                .windows(2)
                .any(|window| window[0] > window[1])
            {
                return Err("adjacency offsets must be monotonic".to_string());
            }
            if let Some(neighbor) = adjacency
                .neighbors
                .iter()
                .copied()
                .find(|&neighbor| neighbor as usize >= count)
            {
                return Err(format!("adjacency neighbor {neighbor} is out of range"));
            }
        }
        Ok(())
    }

    /// Returns the packed per-point attribute row length for RadFoam: `sh_dim + 1`.
    /// This is used when packing attributes for the GPU shader.
    pub fn attribute_dim(&self) -> usize {
        1 + 3 * self.sh_component_count()
    }

    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the number of SH components based on degree.
    pub fn sh_component_count(&self) -> usize {
        get_sh_component_count(self.sh_degree)
    }

    /// Returns the SH coefficients for a single point as RGB Vec3 array.
    pub fn get_sh_coefficients(&self, point_idx: usize) -> [glam::Vec3; MAX_SH_COMPONENTS] {
        let mut result = [glam::Vec3::ZERO; MAX_SH_COMPONENTS];
        let comp_count = self.sh_component_count();
        let base = point_idx * comp_count * 3;
        for (i, slot) in result.iter_mut().take(comp_count).enumerate() {
            *slot = glam::Vec3::new(
                self.sh_coefficients[base + i * 3],
                self.sh_coefficients[base + i * 3 + 1],
                self.sh_coefficients[base + i * 3 + 2],
            );
        }
        result
    }

    /// Computes adjacency from point positions.
    ///
    /// When `radii` is `Some`, builds the Čech complex on the weighted balls
    /// (Power Foam). Otherwise falls back to Delaunay tetrahedralisation.
    ///
    /// This replaces any existing adjacency with a newly computed one.
    pub fn compute_adjacency(&mut self, config: &adjacency::AdjacencyConfig) {
        let new = match self.radii {
            Some(ref radii) => adjacency::compute_cech(&self.points, radii, config),
            None => adjacency::compute_adjacency(&self.points, config),
        };
        self.adjacency = Some(new);
    }

    /// Computes adjacency with default configuration. See [`Self::compute_adjacency`].
    pub fn compute_adjacency_default(&mut self) {
        self.compute_adjacency(&adjacency::AdjacencyConfig::default());
    }

    /// Ensures adjacency is available, computing it if necessary.
    ///
    /// Returns a reference to the adjacency data.
    pub fn ensure_adjacency(&mut self) -> &Adjacency {
        if self.adjacency.is_none() {
            self.compute_adjacency_default();
        }
        self.adjacency.as_ref().unwrap()
    }
}

// ============================================================================
// GPU Representation
// ============================================================================

#[repr(C)]
pub struct GaussianGpu {
    pub mean: [f32; 3],
    pub pad: f32,
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub opacity: f32,
    pub harmonics: [(glam::Vec3, u32); MAX_SH_COMPONENTS],
}

#[cfg(test)]
mod model_tests {
    use super::*;

    fn valid_model() -> PointCloudModel {
        PointCloudModel {
            points: vec![glam::Vec4::ZERO],
            sh_coefficients: vec![0.0; 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(Adjacency {
                neighbors: Vec::new(),
                offsets: vec![0, 0],
            }),
            radii: Some(vec![1.0]),
        }
    }

    #[test]
    fn model_validation_accepts_consistent_parallel_arrays() {
        assert!(valid_model().validate().is_ok());
    }

    #[test]
    fn model_validation_rejects_bad_sh_and_adjacency_indices() {
        let mut model = valid_model();
        model.sh_coefficients.pop();
        assert!(model.validate().unwrap_err().contains("SH coefficient"));

        let mut model = valid_model();
        model.adjacency = Some(Adjacency {
            neighbors: vec![1],
            offsets: vec![0, 1],
        });
        assert!(model.validate().unwrap_err().contains("out of range"));
    }
}
