//! Scene API for managing multiple objects with per-frame transforms.
//!
//! This module provides a unified scene abstraction that supports:
//! - Multiple objects (Gaussian splats, RadFoam volumes, meshes)
//! - Per-object transforms (position, rotation, scale)
//! - Automatic TLAS rebuild when transforms change
//!
//! # Example
//! ```ignore
//! let mut scene = Scene::new(&context);
//! let object = scene.create_gaussian_object(&model, &params, &context, &mut encoder);
//!
//! // Each frame:
//! scene.set_transform(object, Transform {
//!     position: glam::Vec3::new(1.0, 0.0, 0.0),
//!     rotation: glam::Quat::IDENTITY,
//!     scale: glam::Vec3::ONE,
//! });
//! scene.update(&context, &mut encoder);
//! ```

use blade_graphics as gpu;
use std::{mem, slice};

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
    /// Creates an identity transform.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Creates a transform with only position.
    pub fn from_position(position: glam::Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Creates a transform with position and rotation.
    pub fn from_position_rotation(position: glam::Vec3, rotation: glam::Quat) -> Self {
        Self {
            position,
            rotation,
            ..Default::default()
        }
    }

    /// Converts to a 4x4 transformation matrix.
    pub fn to_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Converts to a 3x4 affine transformation matrix (column-major).
    pub fn to_affine(&self) -> mint::ColumnMatrix3x4<f32> {
        let m = glam::Mat3::from_quat(self.rotation) * glam::Mat3::from_diagonal(self.scale);
        mint::ColumnMatrix3x4 {
            x: m.x_axis.into(),
            y: m.y_axis.into(),
            z: m.z_axis.into(),
            w: self.position.into(),
        }
    }
}

/// Backend-specific data for different object types.
pub enum ObjectData {
    /// Gaussian splatting object with hardware ray tracing.
    Gaussian(GaussianObjectData),
    /// RadFoam volume with compute-based Voronoi traversal.
    RadFoam(RadFoamObjectData),
    /// Triangle mesh with hardware ray tracing (future).
    Mesh(MeshObjectData),
}

/// Data for a Gaussian splatting object.
pub struct GaussianObjectData {
    /// Icosahedron mesh buffer (shared BLAS geometry).
    pub mesh_buf: gpu::Buffer,
    /// Per-instance data buffer for TLAS instances.
    pub instance_buf: gpu::Buffer,
    /// Gaussian parameters buffer.
    pub gauss_buf: gpu::Buffer,
    /// Bottom-level acceleration structure.
    pub blas: gpu::AccelerationStructure,
    /// Number of Gaussian splats.
    pub instance_count: u32,
    /// Minimum opacity for instance scaling.
    pub min_opacity: f32,
    /// Original model data for transform updates.
    pub points: Vec<glam::Vec4>,
    pub rotations: Vec<glam::Quat>,
    pub scales: Vec<glam::Vec3>,
}

/// Data for a RadFoam volume object.
pub struct RadFoamObjectData {
    /// Point positions buffer.
    pub points_buf: gpu::Buffer,
    /// Packed attributes buffer.
    pub attributes_buf: gpu::Buffer,
    /// Adjacency indices buffer.
    pub adjacency_buf: gpu::Buffer,
    /// Adjacency offsets buffer.
    pub adjacency_offsets_buf: gpu::Buffer,
    /// SH degree.
    pub sh_degree: usize,
    /// Attribute dimension.
    pub attr_dim: usize,
    /// Number of points.
    pub num_points: usize,
}

/// Data for a triangle mesh object (future).
pub struct MeshObjectData {
    /// Vertex buffer.
    pub vertex_buf: gpu::Buffer,
    /// Index buffer.
    pub index_buf: gpu::Buffer,
    /// Bottom-level acceleration structure.
    pub blas: gpu::AccelerationStructure,
    /// Number of triangles.
    pub triangle_count: u32,
}

/// A scene object with transform.
struct SceneObject {
    data: ObjectData,
    transform: Transform,
    transform_dirty: bool,
}

/// Scene manager for multiple objects with transforms.
///
/// Supports both hardware ray-traced objects (Gaussian, Mesh) and
/// compute-based objects (RadFoam).
pub struct Scene {
    objects: Vec<SceneObject>,
    /// Top-level acceleration structure for RT objects.
    tlas: Option<gpu::AccelerationStructure>,
    /// Whether TLAS needs rebuild.
    tlas_dirty: bool,
    /// Scratch buffer for AS builds.
    scratch_buf: Option<gpu::Buffer>,
    /// Instance buffer for TLAS.
    tlas_instance_buf: Option<gpu::Buffer>,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            tlas: None,
            tlas_dirty: false,
            scratch_buf: None,
            tlas_instance_buf: None,
        }
    }

    /// Returns the number of objects in the scene.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Creates a Gaussian splatting object from a point cloud model.
    ///
    /// Requires the model to have `transforms` (rotation + scale).
    pub fn create_gaussian_object(
        &mut self,
        model: &crate::PointCloudModel,
        min_opacity: f32,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> ObjectHandle {
        let transforms = model
            .transforms
            .as_ref()
            .expect("Gaussian object requires transforms");

        let count = model.len();

        // Create Gaussian data buffer
        let gauss_total_size = (count * mem::size_of::<crate::GaussianGpu>()) as u64;
        let gauss_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-gauss-blobs",
            size: gauss_total_size,
            memory: gpu::Memory::Device,
        });
        let gauss_scratch = context.create_buffer(gpu::BufferDesc {
            name: "scene-gauss-upload",
            size: gauss_total_size,
            memory: gpu::Memory::Upload,
        });

        {
            let gaussians_gpu = unsafe {
                slice::from_raw_parts_mut(gauss_scratch.data() as *mut crate::GaussianGpu, count)
            };
            for (i, gg) in gaussians_gpu.iter_mut().enumerate() {
                let point = model.points[i];
                gg.mean = [point.x, point.y, point.z];
                gg.rotation = transforms.rotations[i].into();
                gg.scale = transforms.scales[i].into();
                gg.opacity = point.w;
                let shc = model.get_sh_coefficients(i);
                for (h, c) in gg.harmonics.iter_mut().zip(shc.iter()) {
                    *h = (*c, 0)
                }
            }
        }

        // Create icosahedron mesh for BLAS
        let inner_radius = 1.0;
        let geometry = crate::Icosahedron::new(inner_radius);
        let vertex_data_size = (geometry.vertices.len() * mem::size_of::<[f32; 3]>()) as u64;
        let index_data_size = (geometry.triangles.len() * mem::size_of::<[u16; 3]>()) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-gauss-mesh",
            size: vertex_data_size + index_data_size,
            memory: gpu::Memory::Device,
        });

        let meshes = [gpu::AccelerationStructureMesh {
            vertex_data: mesh_buf.at(0),
            vertex_format: gpu::VertexFormat::F32Vec3,
            vertex_stride: mem::size_of::<[f32; 3]>() as u32,
            vertex_count: geometry.vertices.len() as u32,
            index_data: mesh_buf.at(vertex_data_size),
            index_type: Some(gpu::IndexType::U16),
            triangle_count: geometry.triangles.len() as u32,
            transform_data: gpu::Buffer::default().at(0),
            is_opaque: false,
        }];

        let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
        let blas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "scene-gauss-blas",
            ty: gpu::AccelerationStructureType::BottomLevel,
            size: blas_sizes.data,
        });

        // Create instance buffer (will be rebuilt when TLAS is updated)
        let instance_buf = self.create_gaussian_instance_buffer(
            context,
            &model.points,
            &transforms.rotations,
            &transforms.scales,
            min_opacity,
            blas,
        );

        // Create scratch buffer for BLAS build
        let scratch_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-gauss-scratch",
            size: blas_sizes.scratch,
            memory: gpu::Memory::Device,
        });

        // Upload mesh data
        let mesh_stage = context.create_buffer(gpu::BufferDesc {
            name: "scene-gauss-mesh-stage",
            size: vertex_data_size + index_data_size,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            std::ptr::copy_nonoverlapping(
                geometry.vertices.as_ptr(),
                mesh_stage.data() as *mut [f32; 3],
                geometry.vertices.len(),
            );
            std::ptr::copy_nonoverlapping(
                geometry.triangles.as_ptr(),
                mesh_stage.data().add(vertex_data_size as usize) as *mut [u16; 3],
                geometry.triangles.len(),
            );
        }

        // Encode uploads and BLAS build
        encoder.start();
        if let mut pass = encoder.transfer("scene-gauss-init") {
            pass.copy_buffer_to_buffer(
                mesh_stage.at(0),
                mesh_buf.at(0),
                vertex_data_size + index_data_size,
            );
            pass.copy_buffer_to_buffer(gauss_scratch.at(0), gauss_buf.at(0), gauss_total_size);
        }
        if let mut pass = encoder.acceleration_structure("scene-gauss-blas") {
            pass.build_bottom_level(blas, &meshes, scratch_buf.at(0));
        }

        let sync_point = context.submit(encoder);
        context.wait_for(&sync_point, !0);

        context.destroy_buffer(gauss_scratch);
        context.destroy_buffer(scratch_buf);
        context.destroy_buffer(mesh_stage);

        let handle = ObjectHandle(self.objects.len() as u32);
        self.objects.push(SceneObject {
            data: ObjectData::Gaussian(GaussianObjectData {
                mesh_buf,
                instance_buf,
                gauss_buf,
                blas,
                instance_count: count as u32,
                min_opacity,
                points: model.points.clone(),
                rotations: transforms.rotations.clone(),
                scales: transforms.scales.clone(),
            }),
            transform: Transform::default(),
            transform_dirty: true,
        });

        self.tlas_dirty = true;
        handle
    }

    /// Creates a RadFoam volume object from a point cloud model.
    ///
    /// Requires the model to have `adjacency` data.
    pub fn create_radfoam_object(
        &mut self,
        model: &crate::PointCloudModel,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> ObjectHandle {
        let adjacency = model
            .adjacency
            .as_ref()
            .expect("RadFoam object requires adjacency");

        let num_points = model.len();
        let num_adjacency = adjacency.neighbors.len();
        let sh_component_count = model.sh_component_count();
        let attr_dim = 1 + 3 * sh_component_count;

        // Sizes
        let points_size = (num_points * mem::size_of::<[f32; 4]>()) as u64;
        let attrs_size = (num_points * attr_dim * mem::size_of::<f32>()) as u64;
        let adj_size = (num_adjacency * mem::size_of::<u32>()) as u64;
        let adj_off_size = (adjacency.offsets.len() * mem::size_of::<u32>()) as u64;

        // Device buffers
        let points_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-points",
            size: points_size,
            memory: gpu::Memory::Device,
        });
        let attributes_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-attributes",
            size: attrs_size,
            memory: gpu::Memory::Device,
        });
        let adjacency_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-adjacency",
            size: adj_size,
            memory: gpu::Memory::Device,
        });
        let adjacency_offsets_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-adjacency-offsets",
            size: adj_off_size,
            memory: gpu::Memory::Device,
        });

        // Upload buffers
        let points_stage = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-points-upload",
            size: points_size,
            memory: gpu::Memory::Upload,
        });
        let attributes_stage = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-attributes-upload",
            size: attrs_size,
            memory: gpu::Memory::Upload,
        });
        let adjacency_stage = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-adjacency-upload",
            size: adj_size,
            memory: gpu::Memory::Upload,
        });
        let adjacency_offsets_stage = context.create_buffer(gpu::BufferDesc {
            name: "scene-radfoam-adjacency-offsets-upload",
            size: adj_off_size,
            memory: gpu::Memory::Upload,
        });

        // Fill staging buffers
        unsafe {
            let dst_points =
                slice::from_raw_parts_mut(points_stage.data() as *mut [f32; 4], num_points);
            for (i, dst) in dst_points.iter_mut().enumerate() {
                let p = model.points[i];
                dst[0] = p.x;
                dst[1] = p.y;
                dst[2] = p.z;
                dst[3] = 0.0;
            }

            let dst_attrs = slice::from_raw_parts_mut(
                attributes_stage.data() as *mut f32,
                num_points * attr_dim,
            );
            for i in 0..num_points {
                let base = i * attr_dim;
                let sh_base = i * sh_component_count * 3;
                for j in 0..(sh_component_count * 3) {
                    dst_attrs[base + j] = model.sh_coefficients[sh_base + j];
                }
                dst_attrs[base + sh_component_count * 3] = model.points[i].w;
            }

            if num_adjacency > 0 {
                std::ptr::copy_nonoverlapping(
                    adjacency.neighbors.as_ptr(),
                    adjacency_stage.data() as *mut u32,
                    num_adjacency,
                );
            }

            std::ptr::copy_nonoverlapping(
                adjacency.offsets.as_ptr(),
                adjacency_offsets_stage.data() as *mut u32,
                adjacency.offsets.len(),
            );
        }

        // Encode transfers
        encoder.start();
        if let mut pass = encoder.transfer("scene-radfoam-init") {
            if points_size > 0 {
                pass.copy_buffer_to_buffer(points_stage.at(0), points_buf.at(0), points_size);
            }
            if attrs_size > 0 {
                pass.copy_buffer_to_buffer(
                    attributes_stage.at(0),
                    attributes_buf.at(0),
                    attrs_size,
                );
            }
            if adj_size > 0 {
                pass.copy_buffer_to_buffer(adjacency_stage.at(0), adjacency_buf.at(0), adj_size);
            }
            if adj_off_size > 0 {
                pass.copy_buffer_to_buffer(
                    adjacency_offsets_stage.at(0),
                    adjacency_offsets_buf.at(0),
                    adj_off_size,
                );
            }
        }

        let sync_point = context.submit(encoder);
        context.wait_for(&sync_point, !0);

        context.destroy_buffer(points_stage);
        context.destroy_buffer(attributes_stage);
        context.destroy_buffer(adjacency_stage);
        context.destroy_buffer(adjacency_offsets_stage);

        let handle = ObjectHandle(self.objects.len() as u32);
        self.objects.push(SceneObject {
            data: ObjectData::RadFoam(RadFoamObjectData {
                points_buf,
                attributes_buf,
                adjacency_buf,
                adjacency_offsets_buf,
                sh_degree: model.sh_degree,
                attr_dim,
                num_points,
            }),
            transform: Transform::default(),
            transform_dirty: false,
        });

        handle
    }

    /// Sets the transform for an object.
    pub fn set_transform(&mut self, handle: ObjectHandle, transform: Transform) {
        let obj = &mut self.objects[handle.0 as usize];
        obj.transform = transform;
        obj.transform_dirty = true;

        // Only Gaussian objects affect TLAS
        if matches!(obj.data, ObjectData::Gaussian(_)) {
            self.tlas_dirty = true;
        }
    }

    /// Gets the transform for an object.
    pub fn get_transform(&self, handle: ObjectHandle) -> Transform {
        self.objects[handle.0 as usize].transform
    }

    /// Returns the object data for the given handle.
    pub fn get_object_data(&self, handle: ObjectHandle) -> &ObjectData {
        &self.objects[handle.0 as usize].data
    }

    /// Returns the TLAS for hardware ray tracing, if available.
    pub fn tlas(&self) -> Option<gpu::AccelerationStructure> {
        self.tlas
    }

    /// Updates the scene, rebuilding TLAS if needed.
    ///
    /// Call this after changing transforms and before rendering.
    pub fn update(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        if !self.tlas_dirty {
            return;
        }

        // Collect all Gaussian objects with their transforms
        let mut rt_objects: Vec<(usize, &GaussianObjectData, Transform)> = Vec::new();
        for (i, obj) in self.objects.iter().enumerate() {
            if let ObjectData::Gaussian(ref data) = obj.data {
                rt_objects.push((i, data, obj.transform));
            }
        }

        if rt_objects.is_empty() {
            // No RT objects, no TLAS needed
            if let Some(tlas) = self.tlas.take() {
                context.destroy_acceleration_structure(tlas);
            }
            if let Some(buf) = self.tlas_instance_buf.take() {
                context.destroy_buffer(buf);
            }
            if let Some(buf) = self.scratch_buf.take() {
                context.destroy_buffer(buf);
            }
            self.tlas_dirty = false;
            return;
        }

        // Calculate total instance count
        let total_instances: u32 = rt_objects.iter().map(|(_, d, _)| d.instance_count).sum();

        // Build instance data for TLAS
        let mut instances = Vec::with_capacity(total_instances as usize);
        let mut blas_list: Vec<gpu::AccelerationStructure> = Vec::new();

        for (_, data, obj_transform) in &rt_objects {
            let blas_idx = blas_list.len();
            blas_list.push(data.blas);

            // Build instances for this object
            for i in 0..data.instance_count as usize {
                let point = data.points[i];
                let rotation = data.rotations[i];
                let scale = data.scales[i];
                let opacity = point.w;
                let local_mean = glam::Vec3::new(point.x, point.y, point.z);

                // Apply object transform to local position
                let world_mean = obj_transform.rotation * (obj_transform.scale * local_mean)
                    + obj_transform.position;
                let world_rotation = obj_transform.rotation * rotation;
                let world_scale = obj_transform.scale * scale;

                let extra_scale = (2.0 * (opacity / data.min_opacity).ln().max(0.0)).sqrt();
                let m = glam::Mat3::from_quat(world_rotation)
                    * glam::Mat3::from_diagonal(extra_scale * world_scale);

                instances.push(gpu::AccelerationStructureInstance {
                    acceleration_structure_index: blas_idx as u32,
                    transform: mint::ColumnMatrix3x4 {
                        x: m.x_axis.into(),
                        y: m.y_axis.into(),
                        z: m.z_axis.into(),
                        w: world_mean.into(),
                    }
                    .into(),
                    mask: 0xFF,
                    custom_index: 0,
                });
            }
        }

        // Create/update instance buffer
        if let Some(old_buf) = self.tlas_instance_buf.take() {
            context.destroy_buffer(old_buf);
        }
        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&instances, &blas_list);
        self.tlas_instance_buf = Some(instance_buf);

        // Create/update TLAS
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(total_instances);

        if let Some(old_tlas) = self.tlas.take() {
            context.destroy_acceleration_structure(old_tlas);
        }
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "scene-tlas",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
        });
        self.tlas = Some(tlas);

        // Create/update scratch buffer
        if let Some(old_scratch) = self.scratch_buf.take() {
            context.destroy_buffer(old_scratch);
        }
        let scratch_buf = context.create_buffer(gpu::BufferDesc {
            name: "scene-tlas-scratch",
            size: tlas_sizes.scratch,
            memory: gpu::Memory::Device,
        });
        self.scratch_buf = Some(scratch_buf);

        // Build TLAS
        encoder.start();
        if let mut pass = encoder.acceleration_structure("scene-tlas-build") {
            pass.build_top_level(
                tlas,
                &blas_list,
                total_instances,
                instance_buf.at(0),
                scratch_buf.at(0),
            );
        }
        let sync_point = context.submit(encoder);
        context.wait_for(&sync_point, !0);

        self.tlas_dirty = false;
    }

    /// Cleans up GPU resources.
    pub fn destroy(&mut self, context: &gpu::Context) {
        for obj in self.objects.drain(..) {
            match obj.data {
                ObjectData::Gaussian(data) => {
                    context.destroy_buffer(data.mesh_buf);
                    context.destroy_buffer(data.instance_buf);
                    context.destroy_buffer(data.gauss_buf);
                    context.destroy_acceleration_structure(data.blas);
                }
                ObjectData::RadFoam(data) => {
                    context.destroy_buffer(data.points_buf);
                    context.destroy_buffer(data.attributes_buf);
                    context.destroy_buffer(data.adjacency_buf);
                    context.destroy_buffer(data.adjacency_offsets_buf);
                }
                ObjectData::Mesh(data) => {
                    context.destroy_buffer(data.vertex_buf);
                    context.destroy_buffer(data.index_buf);
                    context.destroy_acceleration_structure(data.blas);
                }
            }
        }

        if let Some(tlas) = self.tlas.take() {
            context.destroy_acceleration_structure(tlas);
        }
        if let Some(buf) = self.tlas_instance_buf.take() {
            context.destroy_buffer(buf);
        }
        if let Some(buf) = self.scratch_buf.take() {
            context.destroy_buffer(buf);
        }
    }

    /// Helper to create Gaussian instance buffer.
    fn create_gaussian_instance_buffer(
        &self,
        context: &gpu::Context,
        points: &[glam::Vec4],
        rotations: &[glam::Quat],
        scales: &[glam::Vec3],
        min_opacity: f32,
        blas: gpu::AccelerationStructure,
    ) -> gpu::Buffer {
        let instances: Vec<gpu::AccelerationStructureInstance> = (0..points.len())
            .map(|i| {
                let point = points[i];
                let rotation = rotations[i];
                let scale = scales[i];
                let opacity = point.w;
                let mean = glam::Vec3::new(point.x, point.y, point.z);

                let extra_scale = (2.0 * (opacity / min_opacity).ln().max(0.0)).sqrt();
                let m = glam::Mat3::from_quat(rotation)
                    * glam::Mat3::from_diagonal(extra_scale * scale);

                gpu::AccelerationStructureInstance {
                    acceleration_structure_index: 0,
                    transform: mint::ColumnMatrix3x4 {
                        x: m.x_axis.into(),
                        y: m.y_axis.into(),
                        z: m.z_axis.into(),
                        w: mean.into(),
                    }
                    .into(),
                    mask: 0xFF,
                    custom_index: 0,
                }
            })
            .collect();

        context.create_acceleration_structure_instance_buffer(&instances, &[blas])
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn transform_to_matrix() {
        let t = Transform {
            position: glam::Vec3::new(1.0, 2.0, 3.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::new(2.0, 2.0, 2.0),
        };
        let m = t.to_matrix();

        // Check translation
        let translated = m.transform_point3(glam::Vec3::ZERO);
        assert!((translated - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);

        // Check scale
        let scaled = m.transform_vector3(glam::Vec3::ONE);
        assert!((scaled - glam::Vec3::new(2.0, 2.0, 2.0)).length() < 1e-5);
    }

    #[test]
    fn object_handle_equality() {
        let h1 = ObjectHandle(0);
        let h2 = ObjectHandle(0);
        let h3 = ObjectHandle(1);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
