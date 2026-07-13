//! Software TLAS Scene API for unified multi-object rendering.
//!
//! This module provides a scene abstraction where cloud backends (Gaussian,
//! RadFoam, PowerFoam, and future sampled fields) coexist in one hierarchy.
//!
//! # Architecture
//!
//! Unlike hardware RT TLAS/BLAS, this uses a **software TLAS** traversed in compute:
//!
//! 1. **Scene bounds buffer**: Array of `ObjectBounds` (bounding spheres) on GPU
//! 2. **Unified traversal**: Compute shader traces rays through all object bounds
//! 3. **Backend dispatch**: When ray hits bounds, dispatch to object-specific renderer:
//!    - Gaussian: hardware RT query into per-object BLAS
//!    - RadFoam: Voronoi traversal via compute
//!    - sampled fields: backend-specific compute traversal (future)
//!
//! # Example
//! ```ignore
//! let mut scene = Scene::new();
//!
//! // Add objects of different types
//! let gaussian_obj = scene.add_gaussian(&model, min_opacity, &ctx, &mut enc);
//! let radfoam_obj = scene.add_radfoam(&model, &ctx, &mut enc);
//!
//! // Set transforms
//! scene.set_transform(gaussian_obj, Transform::from_position(pos1));
//! scene.set_transform(radfoam_obj, Transform::from_position(pos2));
//!
//! // Prepare uploads bounds buffer
//! scene.prepare(&ctx, &mut enc);
//!
//! // Render - unified compute traversal handles all object types
//! scene.render(&mut enc, output_view, camera);
//! ```

use blade_graphics as gpu;
use std::mem;

// ============================================================================
// Object Types
// ============================================================================

/// Object type identifiers for GPU-side dispatch.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectType {
    /// Gaussian splatting with hardware RT.
    Gaussian = 0,
    /// RadFoam volume with Voronoi traversal.
    RadFoam = 1,
}

// ============================================================================
// GPU Structures (must match shader)
// ============================================================================

/// Bounding sphere for software TLAS traversal.
///
/// Each object in the scene has one of these in the bounds buffer.
/// The compute shader tests ray-sphere intersection, then dispatches
/// to the appropriate backend based on `object_type`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectBounds {
    /// World-space center of bounding sphere.
    pub center: [f32; 3],
    /// Bounding sphere radius.
    pub radius: f32,
    /// Object type (see ObjectType enum).
    pub object_type: u32,
    /// Index into the type-specific data array.
    pub data_index: u32,
    /// Per-object SH degree.
    pub sh_degree: u32,
    /// Per-point attribute stride, or zero for backends that do not use it.
    pub attribute_stride: u32,
    /// Backend flags. Bit zero marks bounded PowerFoam support.
    pub flags: u32,
    /// Number of points in the object.
    pub point_count: u32,
    /// Camera-containing start cell for adjacency traversal.
    pub start_point: u32,
    /// Padding for the 16-byte storage-array stride.
    pub pad: u32,
}

/// Per-object transform stored separately for efficient updates.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransform {
    /// Object-to-world matrix (column-major 4x4).
    pub object_to_world: [[f32; 4]; 4],
    /// World-to-object matrix (column-major 4x4).
    pub world_to_object: [[f32; 4]; 4],
}

// ============================================================================
// Object Handle and Transform
// ============================================================================

/// Opaque handle to a scene object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectHandle(pub(crate) u32);

/// Transform for a scene object.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn from_position(position: glam::Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    pub fn from_position_rotation(position: glam::Vec3, rotation: glam::Quat) -> Self {
        Self {
            position,
            rotation,
            ..Default::default()
        }
    }

    pub fn to_matrix(&self) -> glam::Mat4 {
        assert!(
            self.position.is_finite(),
            "transform position must be finite"
        );
        assert!(self.scale.is_finite(), "transform scale must be finite");
        assert!(
            self.scale.abs().min_element() > 1e-8,
            "transform scale must be invertible"
        );
        let rotation_length = self.rotation.length_squared();
        assert!(
            rotation_length.is_finite() && rotation_length > 1e-12,
            "transform rotation must be finite and non-zero"
        );
        glam::Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation.normalize(),
            self.position,
        )
    }

    /// Converts to GPU transform struct with both forward and inverse matrices.
    pub fn to_gpu_transform(&self) -> GpuTransform {
        let m = self.to_matrix();
        let inv = m.inverse();
        GpuTransform {
            object_to_world: m.to_cols_array_2d(),
            world_to_object: inv.to_cols_array_2d(),
        }
    }
}

// ============================================================================
// Scene Object
// ============================================================================

/// Internal scene object representation.
struct SceneObject {
    object_type: ObjectType,
    /// Index into the type-specific data vector.
    data_index: u32,
    /// Current transform.
    transform: Transform,
    /// Local-space bounding sphere center (usually origin for point clouds).
    local_center: glam::Vec3,
    /// Local-space bounding sphere radius.
    local_radius: f32,
    sh_degree: u32,
    attribute_stride: u32,
    flags: u32,
    point_count: u32,
    start_point: u32,
}

// ============================================================================
// Scene
// ============================================================================

/// Scene manager with software TLAS for unified multi-object rendering.
///
/// The scene maintains:
/// - A list of objects with bounding spheres
/// - Per-type data arrays (Gaussian clouds, RadFoam clouds, etc.)
/// - GPU buffers for bounds and transforms
///
/// During rendering, a compute shader traverses all object bounds and
/// dispatches to the appropriate backend for each hit.
///
/// RadFoam objects use binding arrays, allowing multiple objects in the scene.
/// Gaussian objects currently use a single hardware TLAS per object.
pub struct Scene {
    /// All objects in the scene.
    objects: Vec<SceneObject>,

    /// Gaussian GPU clouds (indexed by SceneObject::data_index for Gaussian type).
    gaussian_clouds: Vec<crate::GaussianGpuCloud>,
    /// RadFoam GPU clouds (indexed by SceneObject::data_index for RadFoam type).
    radfoam_clouds: Vec<crate::RadFoamGpuCloud>,

    /// GPU buffer containing ObjectBounds for all objects.
    bounds_buffer: Option<gpu::Buffer>,
    /// GPU buffer containing GpuTransform for all objects.
    transforms_buffer: Option<gpu::Buffer>,

    /// Whether the scene needs to rebuild GPU buffers.
    dirty: bool,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            gaussian_clouds: Vec::new(),
            radfoam_clouds: Vec::new(),
            bounds_buffer: None,
            transforms_buffer: None,
            dirty: false,
        }
    }

    /// Returns the number of objects in the scene.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Computes bounding sphere for a point cloud model.
    fn compute_bounding_sphere(points: &[glam::Vec4], support_radii: &[f32]) -> (glam::Vec3, f32) {
        assert_eq!(points.len(), support_radii.len());
        if points.is_empty() {
            return (glam::Vec3::ZERO, 0.0);
        }

        // Compute centroid
        let mut center = glam::Vec3::ZERO;
        for p in points {
            center += glam::Vec3::new(p.x, p.y, p.z);
        }
        center /= points.len() as f32;

        // Compute max distance from centroid
        let mut radius = 0.0f32;
        for (point, &support_radius) in points.iter().zip(support_radii) {
            let distance = (point.truncate() - center).length();
            radius = radius.max(distance + support_radius);
        }

        (center, radius)
    }

    fn compute_gaussian_bounding_sphere(
        model: &crate::PointCloudModel,
        min_opacity: f32,
    ) -> (glam::Vec3, f32) {
        let transforms = model.transforms.as_ref().unwrap();
        let mut points = Vec::new();
        let mut support_radii = Vec::new();
        for (index, &point) in model.points.iter().enumerate() {
            if point.w <= min_opacity {
                continue;
            }
            let support_scale = (2.0 * (point.w / min_opacity).ln().max(0.0)).sqrt();
            points.push(point);
            support_radii.push(support_scale * transforms.scales[index].max_element());
        }
        Self::compute_bounding_sphere(&points, &support_radii)
    }

    fn compute_radfoam_bounding_sphere(model: &crate::PointCloudModel) -> (glam::Vec3, f32) {
        let zero_radii;
        let support_radii = match model.radii.as_deref() {
            Some(radii) => radii,
            None => {
                zero_radii = vec![0.0; model.len()];
                &zero_radii
            }
        };
        Self::compute_bounding_sphere(&model.points, support_radii)
    }

    /// Adds a Gaussian splatting object to the scene.
    pub fn add_gaussian(
        &mut self,
        model: &crate::PointCloudModel,
        min_opacity: f32,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> ObjectHandle {
        let params = crate::InitParameters { min_opacity };
        let cloud = crate::GaussianGpuCloud::new(model, &params, context, encoder);

        let (local_center, local_radius) =
            Self::compute_gaussian_bounding_sphere(model, min_opacity);
        let point_count = u32::try_from(cloud.num_points).expect("too many Gaussian points");

        let data_index = self.gaussian_clouds.len() as u32;
        self.gaussian_clouds.push(cloud);

        let handle = ObjectHandle(self.objects.len() as u32);
        self.objects.push(SceneObject {
            object_type: ObjectType::Gaussian,
            data_index,
            transform: Transform::default(),
            local_center,
            local_radius,
            sh_degree: model.sh_degree as u32,
            attribute_stride: 0,
            flags: 0,
            point_count,
            start_point: 0,
        });

        self.dirty = true;
        handle
    }

    /// Adds a RadFoam volume object to the scene.
    pub fn add_radfoam(
        &mut self,
        model: &crate::PointCloudModel,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> ObjectHandle {
        let cloud = crate::RadFoamGpuCloud::new(model, context, encoder);
        let (local_center, local_radius) = Self::compute_radfoam_bounding_sphere(model);
        let attribute_stride = u32::try_from(cloud.attr_dim).expect("attribute stride overflow");
        let flags = cloud.is_power_foam as u32;
        let point_count = u32::try_from(cloud.num_points).expect("too many RadFoam points");

        let data_index = self.radfoam_clouds.len() as u32;
        self.radfoam_clouds.push(cloud);

        let handle = ObjectHandle(self.objects.len() as u32);
        self.objects.push(SceneObject {
            object_type: ObjectType::RadFoam,
            data_index,
            transform: Transform::default(),
            local_center,
            local_radius,
            sh_degree: model.sh_degree as u32,
            attribute_stride,
            flags,
            point_count,
            start_point: 0,
        });

        self.dirty = true;
        handle
    }

    /// Sets the transform for an object.
    pub fn set_transform(&mut self, handle: ObjectHandle, transform: Transform) {
        let _ = transform.to_matrix();
        let obj = self
            .objects
            .get_mut(handle.0 as usize)
            .expect("invalid scene object handle");
        obj.transform = transform;
        self.dirty = true;
    }

    /// Updates each RadFoam object's traversal seed to the cell containing the
    /// camera in that object's local space.
    pub fn update_camera(&mut self, world_position: glam::Vec3) {
        assert!(world_position.is_finite(), "camera position must be finite");
        for object in self.objects.iter_mut() {
            if object.object_type != ObjectType::RadFoam {
                continue;
            }
            let local_position = object
                .transform
                .to_matrix()
                .inverse()
                .transform_point3(world_position);
            let start_point =
                self.radfoam_clouds[object.data_index as usize].containing_point(local_position);
            if object.start_point != start_point {
                object.start_point = start_point;
                self.dirty = true;
            }
        }
    }

    /// Gets the transform for an object.
    pub fn get_transform(&self, handle: ObjectHandle) -> Option<Transform> {
        self.objects.get(handle.0 as usize).map(|obj| obj.transform)
    }

    /// Returns the object type for a handle.
    pub fn get_object_type(&self, handle: ObjectHandle) -> Option<ObjectType> {
        self.objects
            .get(handle.0 as usize)
            .map(|obj| obj.object_type)
    }

    /// Prepares the scene for rendering.
    ///
    /// This uploads the bounds, transforms, and packed RadFoam buffers to GPU.
    pub fn prepare(&mut self, context: &gpu::Context, _encoder: &mut gpu::CommandEncoder) {
        if !self.dirty || self.objects.is_empty() {
            return;
        }

        let num_objects = self.objects.len();

        // Build bounds and transforms data
        let mut bounds_data = Vec::with_capacity(num_objects);
        let mut transforms_data = Vec::with_capacity(num_objects);

        for obj in &self.objects {
            // Transform local bounds to world space
            let world_center = obj.transform.to_matrix().transform_point3(obj.local_center);
            // Scale radius by max scale component
            let max_scale = obj.transform.scale.abs().max_element();
            let world_radius = obj.local_radius * max_scale;

            bounds_data.push(ObjectBounds {
                center: world_center.into(),
                radius: world_radius,
                object_type: obj.object_type as u32,
                data_index: obj.data_index,
                sh_degree: obj.sh_degree,
                attribute_stride: obj.attribute_stride,
                flags: obj.flags,
                point_count: obj.point_count,
                start_point: obj.start_point,
                pad: 0,
            });

            transforms_data.push(obj.transform.to_gpu_transform());
        }

        // Destroy old buffers
        if let Some(buf) = self.bounds_buffer.take() {
            context.destroy_buffer(buf);
        }
        if let Some(buf) = self.transforms_buffer.take() {
            context.destroy_buffer(buf);
        }

        // Create and upload bounds buffer
        let bounds_size = (num_objects * mem::size_of::<ObjectBounds>()) as u64;
        let bounds_buffer = context.create_buffer(gpu::BufferDesc {
            name: "scene-bounds",
            size: bounds_size,
            memory: gpu::Memory::Shared, // CPU-writable for easy updates
        });
        unsafe {
            std::ptr::copy_nonoverlapping(
                bounds_data.as_ptr(),
                bounds_buffer.data() as *mut ObjectBounds,
                num_objects,
            );
        }
        self.bounds_buffer = Some(bounds_buffer);

        // Create and upload transforms buffer
        let transforms_size = (num_objects * mem::size_of::<GpuTransform>()) as u64;
        let transforms_buffer = context.create_buffer(gpu::BufferDesc {
            name: "scene-transforms",
            size: transforms_size,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            std::ptr::copy_nonoverlapping(
                transforms_data.as_ptr(),
                transforms_buffer.data() as *mut GpuTransform,
                num_objects,
            );
        }
        self.transforms_buffer = Some(transforms_buffer);

        self.dirty = false;
    }

    /// Returns the scene render data for binding to shaders.
    pub fn render_data(&self) -> Option<SceneRenderData<'_>> {
        Some(SceneRenderData {
            bounds: self.bounds_buffer?.into(),
            transforms: self.transforms_buffer?.into(),
            object_count: self.objects.len() as u32,
            gaussian_clouds: &self.gaussian_clouds,
            radfoam_clouds: &self.radfoam_clouds,
        })
    }

    /// Returns the Gaussian cloud for a specific object.
    pub fn get_gaussian_cloud(&self, handle: ObjectHandle) -> Option<&crate::GaussianGpuCloud> {
        let obj = self.objects.get(handle.0 as usize)?;
        if obj.object_type != ObjectType::Gaussian {
            return None;
        }
        self.gaussian_clouds.get(obj.data_index as usize)
    }

    /// Returns the RadFoam cloud for a specific object.
    pub fn get_radfoam_cloud(&self, handle: ObjectHandle) -> Option<&crate::RadFoamGpuCloud> {
        let obj = self.objects.get(handle.0 as usize)?;
        if obj.object_type != ObjectType::RadFoam {
            return None;
        }
        self.radfoam_clouds.get(obj.data_index as usize)
    }

    /// Cleans up all GPU resources.
    pub fn destroy(&mut self, context: &gpu::Context) {
        // Destroy per-object resources
        for cloud in &mut self.gaussian_clouds {
            cloud.deinit(context);
        }
        self.gaussian_clouds.clear();
        for cloud in &mut self.radfoam_clouds {
            cloud.deinit(context);
        }
        self.radfoam_clouds.clear();

        // Destroy scene buffers
        if let Some(buf) = self.bounds_buffer.take() {
            context.destroy_buffer(buf);
        }
        if let Some(buf) = self.transforms_buffer.take() {
            context.destroy_buffer(buf);
        }

        self.objects.clear();
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Render Data
// ============================================================================

/// Data needed to render the scene.
///
/// Bind these to the unified scene traversal shader.
pub struct SceneRenderData<'a> {
    /// GPU buffer of ObjectBounds (one per object).
    pub bounds: gpu::BufferPiece,
    /// GPU buffer of GpuTransform (one per object).
    pub transforms: gpu::BufferPiece,
    /// Number of objects in the scene.
    pub object_count: u32,
    /// Gaussian GPU clouds for backend dispatch.
    pub gaussian_clouds: &'a [crate::GaussianGpuCloud],
    /// RadFoam GPU clouds for backend dispatch.
    pub radfoam_clouds: &'a [crate::RadFoamGpuCloud],
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_identity() {
        let t = Transform::identity();
        assert_eq!(t.position, glam::Vec3::ZERO);
        assert_eq!(t.rotation, glam::Quat::IDENTITY);
        assert_eq!(t.scale, glam::Vec3::ONE);
    }

    #[test]
    fn transform_to_gpu() {
        let t = Transform {
            position: glam::Vec3::new(1.0, 2.0, 3.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        };
        let gpu = t.to_gpu_transform();

        // Check translation in column 3 (w column)
        assert!((gpu.object_to_world[3][0] - 1.0).abs() < 1e-5);
        assert!((gpu.object_to_world[3][1] - 2.0).abs() < 1e-5);
        assert!((gpu.object_to_world[3][2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn object_handle_equality() {
        let h1 = ObjectHandle(0);
        let h2 = ObjectHandle(0);
        let h3 = ObjectHandle(1);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn gpu_scene_structures_have_wgsl_array_strides() {
        assert_eq!(mem::size_of::<ObjectBounds>(), 48);
        assert_eq!(mem::size_of::<GpuTransform>(), 128);
    }

    #[test]
    fn bounding_sphere_empty() {
        let (center, radius) = Scene::compute_bounding_sphere(&[], &[]);
        assert_eq!(center, glam::Vec3::ZERO);
        assert_eq!(radius, 0.0);
    }

    #[test]
    fn bounding_sphere_single_point() {
        let points = vec![glam::Vec4::new(1.0, 2.0, 3.0, 1.0)];
        let (center, radius) = Scene::compute_bounding_sphere(&points, &[0.0]);
        assert!((center - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        assert!(radius < 1e-5);
    }

    #[test]
    fn bounding_sphere_symmetric() {
        let points = vec![
            glam::Vec4::new(-1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        ];
        let (center, radius) = Scene::compute_bounding_sphere(&points, &[0.0, 0.0]);
        assert!(center.length() < 1e-5); // Center should be at origin
        assert!((radius - 1.0).abs() < 1e-5); // Radius should be 1
    }

    #[test]
    fn bounding_sphere_includes_per_point_support() {
        let points = [
            glam::Vec4::new(-1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        ];
        let (center, radius) = Scene::compute_bounding_sphere(&points, &[0.5, 2.0]);
        assert_eq!(center, glam::Vec3::ZERO);
        assert_eq!(radius, 3.0);
    }

    #[test]
    fn gaussian_bounds_include_finite_proxy_extent() {
        let model = crate::PointCloudModel {
            points: vec![glam::Vec4::new(2.0, 3.0, 4.0, 1.0)],
            sh_coefficients: vec![0.0; 3],
            sh_degree: 0,
            transforms: Some(crate::Transforms {
                rotations: vec![glam::Quat::IDENTITY],
                scales: vec![glam::Vec3::new(2.0, 1.0, 0.5)],
            }),
            adjacency: None,
            radii: None,
        };
        let (center, radius) = Scene::compute_gaussian_bounding_sphere(&model, 0.01);
        let expected = 2.0 * (2.0_f32 * 100.0_f32.ln()).sqrt();
        assert_eq!(center, glam::Vec3::new(2.0, 3.0, 4.0));
        assert!((radius - expected).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "transform scale must be invertible")]
    fn transform_rejects_singular_scale() {
        let transform = Transform {
            scale: glam::Vec3::new(1.0, 0.0, 1.0),
            ..Transform::identity()
        };
        let _ = transform.to_matrix();
    }
}
