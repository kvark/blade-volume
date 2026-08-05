#![allow(irrefutable_let_patterns)]

mod adjacency;
mod camera;
pub mod gpu;
pub mod relight;
mod scene;
pub mod shaders;
mod shape;
pub mod trace;

pub mod io;

pub use adjacency::{
    compute_adjacency, compute_adjacency_default, compute_cech, compute_cech_default, compute_knn,
    radii_from_nearest_neighbour, radii_from_powerfoam_reference, spring_relax,
    try_compute_adjacency, try_compute_adjacency_default, AdjacencyConfig, AdjacencyError,
};
#[cfg(feature = "qhull")]
pub use adjacency::{compute_adjacency_qhull, compute_adjacency_qhull_default};
pub use camera::{orientation_looking, CameraParams};
pub use gpu::{
    GaussianGpuCloud, InitParameters, MeshReferenceSettings, MeshReferenceTracer,
    PowerFoamGpuSplatTracer, RadFoamDepthSettings, RadFoamGpuCloud, RadFoamGpuDepthTracer,
    RadFoamGpuTracer, RadFoamTraceSettings, ReferenceMesh,
};
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
/// Spatial RGB residual basis terms stored for each oriented PowerFoam site.
///
/// The basis is `(q.x, q.y, q.z, dot(q, q))`, where `q` is the normalized
/// surface-plane query coordinate. Three world-coordinate terms span the
/// two-dimensional tangent plane without storing an arbitrary tangent frame;
/// the last term captures a radial center-to-edge change.
pub const SURFACE_COLOR_COMPONENTS: usize = 4;
/// Spatial detail sites stored for each oriented PowerFoam cell.
///
/// Eight matches the released PowerFoam implementation. Geometry stores one
/// radius-normalized tangent offset and height per site; appearance stores one
/// view-independent RGB residual per site.
pub const SURFACE_DETAIL_SITES: usize = 8;
/// Directional sites in the compact Spherical Voronoi appearance model.
///
/// Eight matches the parameter-cardinality comparison in the Spherical
/// Voronoi paper and the per-texel directional model released with PowerFoam.
pub const SPHERICAL_VORONOI_SITES: usize = 8;

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

/// Optional directional residual represented by a soft partition of the sphere.
///
/// Each raw axis vector encodes both a direction and temperature: the site
/// logit is `dot(axis, view_direction)`. Softmax over the eight logits blends
/// the matching RGB residuals. This is the published dot-product Spherical
/// Voronoi contract; it is deliberately distinct from PowerFoam's released
/// chord-distance kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalVoronoi {
    /// Point-major raw axes, eight per point.
    pub axes: Vec<glam::Vec3>,
    /// Point-major RGB residual values, eight per point.
    pub colors: Vec<glam::Vec3>,
}

/// Radius-normalized spatial detail for oriented PowerFoam cells.
///
/// `offsets` are object-space vectors projected onto the current surface
/// tangent plane before use. Keeping them in three components avoids a
/// discontinuous, implicit tangent-frame convention while retaining two
/// effective degrees of freedom. `heights` displace the surface along its
/// normal, and `colors` are RGB residuals added to the SH appearance. Optional
/// `density_logits` redistribute the cell's density across the same sites;
/// eight equal logits are an exact identity.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDetail {
    /// Point-major radius-normalized site offsets, eight per point.
    pub offsets: Vec<glam::Vec3>,
    /// Point-major radius-normalized signed heights, eight per point.
    pub heights: Vec<f32>,
    /// Point-major RGB residuals, eight per point.
    pub colors: Vec<glam::Vec3>,
    /// Optional point-major spatial density logits, eight per point.
    pub density_logits: Option<Vec<f32>>,
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
/// - `surface_normals`: oriented PowerFoam faces; the normal-facing half is empty
/// - `surface_offsets`: optional signed displacement of each oriented face
/// - `surface_detail`: optional eight-site spatial height and RGB detail
/// - `surface_color_coefficients`: optional within-cell surface appearance
/// - `spherical_voronoi`: optional eight-site directional appearance residual
#[derive(Clone)]
pub struct PointCloudModel {
    /// Position (xyz) + density/opacity (w) for each point.
    pub points: Vec<glam::Vec4>,

    /// Packed SH coefficients: 3 floats (RGB) per SH component, per point.
    /// Evaluated RGB follows the reference RadFoam and 3DGS convention:
    /// display-referred sRGB code values. Rendering and training do not apply
    /// an implicit transfer function or tone map; linear-light consumers must
    /// decode these values explicitly.
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

    /// Optional: per-point oriented surface normal for PowerFoam dipoles.
    /// The half of the bounded power cell in front of this normal has zero
    /// density; the opposite half retains the point's learned density.
    /// Length must equal `points.len()`, and `radii` must also be present.
    pub surface_normals: Option<Vec<glam::Vec3>>,

    /// Optional: signed per-point displacement of the oriented surface plane
    /// along its unit normal. The retained half-space is
    /// `dot(position - point, normal) <= surface_offset`. Missing offsets are
    /// exactly zero for backward-compatible oriented clouds. Length must equal
    /// `points.len()`, and `surface_normals` must also be present.
    pub surface_offsets: Option<Vec<f32>>,

    /// Optional eight-site spatial detail for oriented PowerFoam surfaces.
    ///
    /// A soft partition with weight `exp(-10 * distance_squared / radius^2)`
    /// first blends a radius-scaled height at the base-plane query. The ray is
    /// then intersected with that displaced plane and the weights are
    /// re-evaluated to blend RGB residuals. Missing detail is exactly
    /// equivalent to zero height and color. All three tables contain
    /// `points.len() * SURFACE_DETAIL_SITES` entries.
    pub surface_detail: Option<SurfaceDetail>,

    /// Optional spatial RGB residuals for oriented PowerFoam surfaces.
    ///
    /// Layout matches SH coefficients: `[point][basis component][RGB]`, with
    /// [`SURFACE_COLOR_COMPONENTS`] components per point. The residual is
    /// added to the view-dependent SH color before the non-negative clamp.
    /// This field requires both `radii` and `surface_normals`.
    pub surface_color_coefficients: Option<Vec<f32>>,

    /// Optional eight-site dot-product Spherical Voronoi RGB residual.
    ///
    /// The residual is added to SH before the non-negative clamp. This field
    /// currently requires an oriented PowerFoam model so every renderer which
    /// accepts it shares the same semantics. Both tables have
    /// `points.len() * SPHERICAL_VORONOI_SITES` entries in point-major order.
    pub spherical_voronoi: Option<SphericalVoronoi>,
}

impl PointCloudModel {
    /// Validate all cross-field lengths and CSR indices.
    ///
    /// Public fields intentionally keep model construction lightweight, so IO
    /// and GPU boundaries call this before indexing parallel arrays.
    pub fn validate(&self) -> Result<(), String> {
        let count = self.points.len();
        if let Some((index, _)) = self
            .points
            .iter()
            .enumerate()
            .find(|item| !item.1.is_finite() || item.1.w < 0.0)
        {
            return Err(format!(
                "point {index} must have finite coordinates and non-negative density/opacity"
            ));
        }
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
        if self.sh_coefficients.iter().any(|value| !value.is_finite()) {
            return Err("SH coefficients must be finite".to_string());
        }
        if let Some(ref transforms) = self.transforms {
            if transforms.rotations.len() != count || transforms.scales.len() != count {
                return Err(format!(
                    "transform lengths are rotations={} scales={}, expected {count}",
                    transforms.rotations.len(),
                    transforms.scales.len()
                ));
            }
            if transforms
                .rotations
                .iter()
                .any(|rotation| !rotation.is_finite())
                || transforms
                    .scales
                    .iter()
                    .any(|scale| !scale.is_finite() || scale.min_element() <= 0.0)
            {
                return Err("transforms must have finite rotations and positive scales".to_string());
            }
        }
        if let Some(ref radii) = self.radii {
            if radii.len() != count {
                return Err(format!(
                    "radius length is {}, expected {count}",
                    radii.len()
                ));
            }
            if radii
                .iter()
                .any(|radius| !radius.is_finite() || *radius < 0.0)
            {
                return Err("radii must be finite and non-negative".to_string());
            }
        }
        if let Some(ref normals) = self.surface_normals {
            if self.radii.is_none() {
                return Err("surface normals require PowerFoam radii".to_string());
            }
            if normals.len() != count {
                return Err(format!(
                    "surface normal length is {}, expected {count}",
                    normals.len()
                ));
            }
            if normals
                .iter()
                .any(|normal| !normal.is_finite() || normal.length_squared() <= 1.0e-20)
            {
                return Err("surface normals must be finite and non-zero".to_string());
            }
        }
        if let Some(ref offsets) = self.surface_offsets {
            if self.surface_normals.is_none() {
                return Err("surface offsets require oriented surface normals".to_string());
            }
            if offsets.len() != count {
                return Err(format!(
                    "surface offset length is {}, expected {count}",
                    offsets.len()
                ));
            }
            if offsets.iter().any(|offset| !offset.is_finite()) {
                return Err("surface offsets must be finite".to_string());
            }
        }
        if let Some(ref detail) = self.surface_detail {
            if self.surface_normals.is_none() || self.radii.is_none() {
                return Err(
                    "surface detail requires PowerFoam radii and surface normals".to_string(),
                );
            }
            let expected = count * SURFACE_DETAIL_SITES;
            if detail.offsets.len() != expected
                || detail.heights.len() != expected
                || detail.colors.len() != expected
                || detail
                    .density_logits
                    .as_ref()
                    .is_some_and(|logits| logits.len() != expected)
            {
                return Err(format!(
                    "surface detail lengths are offsets={} heights={} colors={} density_logits={}, expected {expected} and either 0 or {expected} density logits",
                    detail.offsets.len(),
                    detail.heights.len(),
                    detail.colors.len(),
                    detail.density_logits.as_ref().map_or(0, Vec::len),
                ));
            }
            if detail.offsets.iter().any(|offset| !offset.is_finite())
                || detail.heights.iter().any(|height| !height.is_finite())
                || detail.colors.iter().any(|color| !color.is_finite())
                || detail
                    .density_logits
                    .as_ref()
                    .is_some_and(|logits| logits.iter().any(|value| !value.is_finite()))
            {
                return Err("surface detail parameters must be finite".to_string());
            }
        }
        if let Some(ref coefficients) = self.surface_color_coefficients {
            if self.surface_normals.is_none() || self.radii.is_none() {
                return Err(
                    "surface color coefficients require PowerFoam radii and surface normals"
                        .to_string(),
                );
            }
            let expected = count * SURFACE_COLOR_COMPONENTS * 3;
            if coefficients.len() != expected {
                return Err(format!(
                    "surface color coefficient length is {}, expected {expected}",
                    coefficients.len()
                ));
            }
            if coefficients.iter().any(|value| !value.is_finite()) {
                return Err("surface color coefficients must be finite".to_string());
            }
        }
        if let Some(ref spherical_voronoi) = self.spherical_voronoi {
            if self.surface_normals.is_none() || self.radii.is_none() {
                return Err(
                    "Spherical Voronoi appearance requires PowerFoam radii and surface normals"
                        .to_string(),
                );
            }
            let expected = count * SPHERICAL_VORONOI_SITES;
            if spherical_voronoi.axes.len() != expected
                || spherical_voronoi.colors.len() != expected
            {
                return Err(format!(
                    "Spherical Voronoi lengths are axes={} colors={}, expected {expected}",
                    spherical_voronoi.axes.len(),
                    spherical_voronoi.colors.len(),
                ));
            }
            if spherical_voronoi.axes.iter().any(|axis| !axis.is_finite())
                || spherical_voronoi
                    .colors
                    .iter()
                    .any(|color| !color.is_finite())
            {
                return Err("Spherical Voronoi parameters must be finite".to_string());
            }
        }
        if let Some(ref adjacency) = self.adjacency {
            adjacency::validate_csr_result(&adjacency.offsets, &adjacency.neighbors, count)?;
        }
        Ok(())
    }

    /// Returns the packed per-point attribute row length for RadFoam.
    /// This is used when packing attributes for the GPU shader.
    pub fn attribute_dim(&self) -> usize {
        1 + 3 * self.sh_component_count()
            + self
                .surface_color_coefficients
                .as_ref()
                .map_or(0, |_| 3 * SURFACE_COLOR_COMPONENTS)
            + self
                .surface_detail
                .as_ref()
                .map_or(0, |_| 3 * SURFACE_DETAIL_SITES)
            + self
                .surface_detail
                .as_ref()
                .and_then(|detail| detail.density_logits.as_ref())
                .map_or(0, |_| SURFACE_DETAIL_SITES)
            + self
                .spherical_voronoi
                .as_ref()
                .map_or(0, |_| 6 * SPHERICAL_VORONOI_SITES)
    }

    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the site whose Voronoi or power cell contains `position`.
    ///
    /// Weighted clouds minimize `|position - point|² - radius²`; unweighted
    /// clouds reduce to ordinary Euclidean nearest-site selection.
    pub fn containing_cell(&self, position: glam::Vec3) -> u32 {
        assert!(!self.points.is_empty(), "point cloud is empty");
        assert!(position.is_finite(), "query position must be finite");
        let radii = self.radii.as_deref();
        let mut best_index = 0u32;
        let mut best_distance = f32::INFINITY;
        for (index, point) in self.points.iter().enumerate() {
            let radius = radii.map_or(0.0, |values| values[index]);
            let distance = position.distance_squared(point.truncate()) - radius * radius;
            if distance.total_cmp(&best_distance).is_lt() {
                best_distance = distance;
                best_index = index as u32;
            }
        }
        best_index
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
        }
    }

    fn valid_two_point_model() -> PointCloudModel {
        PointCloudModel {
            points: vec![glam::Vec4::ZERO; 2],
            sh_coefficients: vec![0.0; 6],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(Adjacency {
                neighbors: vec![1, 0],
                offsets: vec![0, 1, 2],
            }),
            radii: Some(vec![1.0; 2]),
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
        }
    }

    #[test]
    fn model_validation_accepts_consistent_parallel_arrays() {
        assert!(valid_model().validate().is_ok());
    }

    #[test]
    fn model_validation_checks_oriented_powerfoam_normals() {
        let mut model = valid_model();
        model.surface_normals = Some(vec![glam::Vec3::Z]);
        assert!(model.validate().is_ok());

        model.radii = None;
        assert!(model.validate().unwrap_err().contains("require PowerFoam"));

        model.radii = Some(vec![1.0]);
        model.surface_normals = Some(vec![glam::Vec3::ZERO]);
        assert!(model
            .validate()
            .unwrap_err()
            .contains("finite and non-zero"));
    }

    #[test]
    fn model_validation_checks_oriented_surface_offsets() {
        let mut model = valid_model();
        model.surface_offsets = Some(vec![0.0]);
        assert!(model.validate().unwrap_err().contains("require oriented"));

        model.surface_normals = Some(vec![glam::Vec3::Z]);
        assert!(model.validate().is_ok());

        model.surface_offsets = Some(Vec::new());
        assert!(model.validate().unwrap_err().contains("length"));

        model.surface_offsets = Some(vec![f32::NAN]);
        assert!(model.validate().unwrap_err().contains("must be finite"));
    }

    #[test]
    fn model_validation_checks_surface_color_coefficients() {
        let mut model = valid_model();
        model.surface_color_coefficients = Some(vec![0.0; SURFACE_COLOR_COMPONENTS * 3]);
        assert!(model.validate().unwrap_err().contains("require PowerFoam"));

        model.surface_normals = Some(vec![glam::Vec3::Z]);
        assert!(model.validate().is_ok());
        assert_eq!(model.attribute_dim(), 3 + 1 + SURFACE_COLOR_COMPONENTS * 3);

        model.surface_color_coefficients = Some(Vec::new());
        assert!(model.validate().unwrap_err().contains("length"));
        model.surface_color_coefficients = Some(vec![f32::NAN; SURFACE_COLOR_COMPONENTS * 3]);
        assert!(model.validate().unwrap_err().contains("must be finite"));
    }

    #[test]
    fn model_validation_checks_surface_detail() {
        let mut model = valid_model();
        model.surface_detail = Some(SurfaceDetail {
            offsets: vec![glam::Vec3::ZERO; SURFACE_DETAIL_SITES],
            heights: vec![0.0; SURFACE_DETAIL_SITES],
            colors: vec![glam::Vec3::ZERO; SURFACE_DETAIL_SITES],
            density_logits: None,
        });
        assert!(model.validate().unwrap_err().contains("requires PowerFoam"));

        model.surface_normals = Some(vec![glam::Vec3::Z]);
        assert!(model.validate().is_ok());
        assert_eq!(model.attribute_dim(), 3 + 1 + SURFACE_DETAIL_SITES * 3);

        model.surface_detail.as_mut().unwrap().density_logits =
            Some(vec![0.0; SURFACE_DETAIL_SITES]);
        assert!(model.validate().is_ok());
        assert_eq!(model.attribute_dim(), 3 + 1 + SURFACE_DETAIL_SITES * 4);
        model.surface_detail.as_mut().unwrap().density_logits = Some(Vec::new());
        assert!(model.validate().unwrap_err().contains("lengths"));
        model.surface_detail.as_mut().unwrap().density_logits =
            Some(vec![f32::NAN; SURFACE_DETAIL_SITES]);
        assert!(model.validate().unwrap_err().contains("must be finite"));
        model.surface_detail.as_mut().unwrap().density_logits = None;

        model.surface_detail.as_mut().unwrap().heights.pop();
        assert!(model.validate().unwrap_err().contains("lengths"));
        model.surface_detail.as_mut().unwrap().heights = vec![f32::NAN; SURFACE_DETAIL_SITES];
        assert!(model.validate().unwrap_err().contains("must be finite"));
    }

    #[test]
    fn model_validation_checks_spherical_voronoi() {
        let mut model = valid_model();
        model.spherical_voronoi = Some(SphericalVoronoi {
            axes: vec![glam::Vec3::ZERO; SPHERICAL_VORONOI_SITES],
            colors: vec![glam::Vec3::ZERO; SPHERICAL_VORONOI_SITES],
        });
        assert!(model.validate().unwrap_err().contains("requires PowerFoam"));

        model.surface_normals = Some(vec![glam::Vec3::Z]);
        assert!(model.validate().is_ok());
        assert_eq!(model.attribute_dim(), 3 + 1 + SPHERICAL_VORONOI_SITES * 6);

        model.spherical_voronoi.as_mut().unwrap().axes.pop();
        assert!(model.validate().unwrap_err().contains("lengths"));
        model.spherical_voronoi.as_mut().unwrap().axes =
            vec![glam::Vec3::splat(f32::NAN); SPHERICAL_VORONOI_SITES];
        assert!(model.validate().unwrap_err().contains("must be finite"));
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
        assert!(model.validate().unwrap_err().contains("out of bounds"));
    }

    #[test]
    fn model_validation_rejects_non_simple_or_asymmetric_adjacency() {
        let mut model = valid_two_point_model();
        model.adjacency = Some(Adjacency {
            neighbors: vec![0, 0],
            offsets: vec![0, 1, 2],
        });
        assert!(model.validate().unwrap_err().contains("self-edge"));

        let mut model = valid_two_point_model();
        model.adjacency = Some(Adjacency {
            neighbors: vec![1, 1, 0],
            offsets: vec![0, 2, 3],
        });
        assert!(model.validate().unwrap_err().contains("sorted and unique"));

        let mut model = valid_two_point_model();
        model.adjacency = Some(Adjacency {
            neighbors: vec![1],
            offsets: vec![0, 1, 1],
        });
        assert!(model.validate().unwrap_err().contains("no reverse edge"));
    }

    #[test]
    fn containing_cell_uses_power_distance_for_weighted_clouds() {
        let mut model = valid_two_point_model();
        model.points[0] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        model.points[1] = glam::Vec4::new(10.0, 0.0, 0.0, 1.0);
        model.radii = None;
        assert_eq!(model.containing_cell(glam::Vec3::ZERO), 0);

        model.radii = Some(vec![0.0, 20.0]);
        assert_eq!(model.containing_cell(glam::Vec3::ZERO), 1);
    }

    #[test]
    fn model_validation_rejects_non_finite_and_negative_values() {
        let mut model = valid_model();
        model.points[0].w = -1.0;
        assert!(model.validate().unwrap_err().contains("non-negative"));

        let mut model = valid_model();
        model.radii = Some(vec![f32::NAN]);
        assert!(model.validate().unwrap_err().contains("radii"));

        let mut model = valid_model();
        model.sh_coefficients[0] = f32::INFINITY;
        assert!(model.validate().unwrap_err().contains("finite"));
    }
}
