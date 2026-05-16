#![allow(irrefutable_let_patterns)]

mod adjacency;
mod camera;
mod gpu;
mod scene;
pub mod shaders;
mod shape;

pub mod io;

pub use adjacency::{compute_adjacency, compute_adjacency_default, AdjacencyConfig};
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

    /// Computes adjacency from point positions using Delaunay tetrahedralization.
    ///
    /// This replaces any existing adjacency with a newly computed one.
    pub fn compute_adjacency(&mut self, config: &adjacency::AdjacencyConfig) {
        self.adjacency = Some(adjacency::compute_adjacency(&self.points, config));
    }

    /// Computes adjacency with default configuration.
    ///
    /// This replaces any existing adjacency with a newly computed one.
    pub fn compute_adjacency_default(&mut self) {
        self.adjacency = Some(adjacency::compute_adjacency_default(&self.points));
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
