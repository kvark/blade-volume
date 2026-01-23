#![allow(irrefutable_let_patterns)]

mod camera;
pub mod io;
mod point_cloud;
mod radfoam_point_cloud;
mod shape;

pub use camera::CameraParams;
pub use point_cloud::{InitParameters, PointCloud};
pub use radfoam_point_cloud::RadFoamPointCloud;
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

#[derive(Clone, Default)]
pub struct Gaussian {
    pub mean: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub opacity: f32,
    pub shc: [glam::Vec3; MAX_SH_COMPONENTS],
}

pub struct Model {
    pub gaussians: Vec<Gaussian>,
    pub max_sh_degree: usize,
}

/// CPU-side storage for an upstream Radiant Foam (RadFoam) scene exported as PLY.
///
/// This matches the data consumed by the upstream RadFoam tracing kernel:
/// - `points` are primal points
/// - `attributes` contain packed SH coefficients followed by density as the last scalar
/// - `point_adjacency` + `point_adjacency_offsets` form a CSR adjacency structure
///
/// Notes:
/// - Upstream uses `attr_memory_size = sh_dim + 1`, where `sh_dim = 3 * (1 + sh_degree)^2`
/// - The last scalar in each attribute row is density `s`
///
/// This type intentionally does not store mesh/triangles; RadFoam traverses Voronoi
/// cells induced by the point set and its (Delaunay-derived) adjacency.
pub struct RadFoamModel {
    /// Primal points (`N` entries).
    pub points: Vec<glam::Vec3>,
    /// Packed per-point attributes (`N * (sh_dim + 1)` scalars):
    /// `[sh_coeffs..., density]` for each point.
    pub attributes: Vec<f32>,
    /// Spherical harmonics degree used to interpret `attributes`.
    pub sh_degree: usize,
    /// Flattened adjacency list (`K` entries).
    pub point_adjacency: Vec<u32>,
    /// CSR offsets into `point_adjacency` (`N + 1` entries).
    ///
    /// By convention:
    /// - `point_adjacency_offsets[0] == 0`
    /// - `point_adjacency_offsets[i+1]` equals the adjacency end offset for point `i`
    /// - `point_adjacency_offsets[N] == point_adjacency.len() as u32`
    pub point_adjacency_offsets: Vec<u32>,
}

impl RadFoamModel {
    /// Returns the packed per-point attribute row length: `sh_dim + 1`.
    pub const fn attribute_dim(sh_degree: usize) -> usize {
        1 + 3 * get_sh_component_count(sh_degree)
    }

    /// Returns the number of points (`N`).
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[repr(C)]
pub struct GaussianGpu {
    pub mean: [f32; 3],
    pub pad: f32,
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub opacity: f32,
    pub harmonics: [(glam::Vec3, u32); MAX_SH_COMPONENTS],
}
