//! Hardware ray-traced renderer for a source triangle mesh.
//!
//! Offline conversion (`blade-volume-convert`) turns a mesh into a point
//! cloud. Judging that conversion needs a ground truth to compare renders
//! against, and the mesh itself is that ground truth — Blade already traces
//! triangles, so nothing about this is analytic or approximate.
//!
//! The shading here is intentionally minimal and mirrors what the converter
//! bakes into cloud samples, so a render difference measures the
//! *representation*, not two different lighting models.

use crate::{shaders, CameraParams};
use blade_graphics as gpu;
use std::{mem, ptr, slice};

/// A triangle mesh prepared for reference rendering.
///
/// Positions are world space; `indices` are triples into `positions`; and
/// `triangle_colors` has one linear-light RGB per triangle, in the same order
/// as `indices`. Flat per-triangle colour is exact for untextured materials,
/// which is what glTF assets with plain base-colour factors use.
#[derive(Clone, Debug, Default)]
pub struct ReferenceMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
    pub triangle_colors: Vec<[f32; 3]>,
}

impl ReferenceMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Check the invariants the GPU path relies on, so a malformed mesh fails
    /// here instead of as device-side corruption.
    pub fn validate(&self) -> Result<(), String> {
        if self.indices.is_empty() {
            return Err("reference mesh has no triangles".to_string());
        }
        if self.triangle_colors.len() != self.indices.len() {
            return Err(format!(
                "triangle_colors has {} entries for {} triangles",
                self.triangle_colors.len(),
                self.indices.len()
            ));
        }
        let vertices = self.positions.len() as u32;
        for (i, tri) in self.indices.iter().enumerate() {
            for &index in tri.iter() {
                if index >= vertices {
                    return Err(format!(
                        "triangle {i} references vertex {index} of {vertices}"
                    ));
                }
            }
        }
        for (i, position) in self.positions.iter().enumerate() {
            if position.iter().any(|c| !c.is_finite()) {
                return Err(format!("vertex {i} has a non-finite coordinate"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshReferenceSettings {
    /// Linear-light gain matching `ConvertOptions::ambient`.
    pub ambient: [f32; 3],
    /// Background for rays that miss, matching the volume backends.
    pub background_rgb: [f32; 3],
}

impl Default for MeshReferenceSettings {
    fn default() -> Self {
        Self {
            ambient: [1.0; 3],
            background_rgb: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct MeshRefParams {
    ambient: [f32; 3],
    pad0: f32,
    background: [f32; 3],
    pad1: f32,
}

#[derive(blade_macros::ShaderData)]
struct MeshRefData {
    g_camera: CameraParams,
    g_params: MeshRefParams,
    g_mesh_tlas: gpu::AccelerationStructure,
    g_triangle_colors: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

/// Renders a triangle mesh with hardware ray tracing, using the same camera
/// convention and output format as the cloud backends.
pub struct MeshReferenceTracer {
    pipeline: gpu::ComputePipeline,
    params: MeshRefParams,
    mesh_buf: gpu::Buffer,
    color_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    blas: gpu::AccelerationStructure,
    tlas: gpu::AccelerationStructure,
}

impl MeshReferenceTracer {
    pub fn new(
        mesh: &ReferenceMesh,
        settings: MeshReferenceSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        mesh.validate().expect("invalid reference mesh");

        let source = shaders::compose(shaders::MESH_REFERENCE);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <MeshRefData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "mesh-reference",
            data_layouts: &[&layout],
            compute: shader.at("trace_main"),
        });

        let vertex_size = (mesh.positions.len() * mem::size_of::<[f32; 3]>()) as u64;
        let index_size = (mesh.indices.len() * mem::size_of::<[u32; 3]>()) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "mesh-ref-geometry",
            size: vertex_size + index_size,
            memory: gpu::Memory::Device,
        });
        // Colour is looked up by primitive index, so this parallels `indices`.
        let color_size = (mesh.indices.len() * mem::size_of::<[f32; 4]>()) as u64;
        let color_buf = context.create_buffer(gpu::BufferDesc {
            name: "mesh-ref-colors",
            size: color_size,
            memory: gpu::Memory::Device,
        });

        let meshes = [gpu::AccelerationStructureMesh {
            vertex_data: mesh_buf.at(0),
            vertex_format: gpu::VertexFormat::F32Vec3,
            vertex_stride: mem::size_of::<[f32; 3]>() as u32,
            vertex_count: mesh.positions.len() as u32,
            index_data: mesh_buf.at(vertex_size),
            index_type: Some(gpu::IndexType::U32),
            triangle_count: mesh.indices.len() as u32,
            transform_data: gpu::Buffer::default().at(0),
            is_opaque: true,
        }];
        let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
        let blas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "mesh-ref-blas",
            ty: gpu::AccelerationStructureType::BottomLevel,
            size: blas_sizes.data,
            updatable: false,
        });

        // The mesh is already in world space, so one identity instance.
        let instances = [gpu::AccelerationStructureInstance {
            acceleration_structure_index: 0,
            transform: mint::ColumnMatrix3x4 {
                x: [1.0, 0.0, 0.0].into(),
                y: [0.0, 1.0, 0.0].into(),
                z: [0.0, 0.0, 1.0].into(),
                w: [0.0, 0.0, 0.0].into(),
            }
            .into(),
            mask: 0xFF,
            custom_index: 0,
        }];
        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&instances, &[blas]);
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(1);
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "mesh-ref-tlas",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
            updatable: false,
        });

        let tlas_scratch_offset =
            (blas_sizes.scratch | (gpu::limits::ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT - 1)) + 1;
        let scratch_buf = context.create_buffer(gpu::BufferDesc {
            name: "mesh-ref-scratch",
            size: tlas_scratch_offset + tlas_sizes.scratch,
            memory: gpu::Memory::Device,
        });

        let stage = context.create_buffer(gpu::BufferDesc {
            name: "mesh-ref-stage",
            size: vertex_size + index_size + color_size,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                mesh.positions.as_ptr(),
                stage.data() as *mut [f32; 3],
                mesh.positions.len(),
            );
            ptr::copy_nonoverlapping(
                mesh.indices.as_ptr(),
                stage.data().add(vertex_size as usize) as *mut [u32; 3],
                mesh.indices.len(),
            );
            let colors = slice::from_raw_parts_mut(
                stage.data().add((vertex_size + index_size) as usize) as *mut [f32; 4],
                mesh.indices.len(),
            );
            for (dst, src) in colors.iter_mut().zip(mesh.triangle_colors.iter()) {
                *dst = [src[0], src[1], src[2], 1.0];
            }
        }

        encoder.start();
        if let mut pass = encoder.transfer("mesh-ref-init") {
            pass.copy_buffer_to_buffer(stage.at(0), mesh_buf.at(0), vertex_size + index_size);
            pass.copy_buffer_to_buffer(
                stage.at(vertex_size + index_size),
                color_buf.at(0),
                color_size,
            );
        }
        if let mut pass = encoder.acceleration_structure("mesh-ref-bottom") {
            pass.build_bottom_level(blas, &meshes, scratch_buf.at(0));
        }
        if let mut pass = encoder.acceleration_structure("mesh-ref-top") {
            pass.build_top_level(
                tlas,
                &[blas],
                1,
                instance_buf.at(0),
                scratch_buf.at(tlas_scratch_offset),
            );
        }
        let sync_point = context.submit(encoder);
        let _ = context.wait_for(&sync_point, !0);

        context.destroy_buffer(scratch_buf);
        context.destroy_buffer(stage);

        Self {
            pipeline,
            params: MeshRefParams {
                ambient: settings.ambient,
                pad0: 0.0,
                background: settings.background_rgb,
                pad1: 0.0,
            },
            mesh_buf,
            color_buf,
            instance_buf,
            blas,
            tlas,
        }
    }

    pub fn dispatch(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        output: gpu::TextureView,
        camera: CameraParams,
        size: [u32; 2],
    ) {
        assert!(size[0] > 0 && size[1] > 0, "render size must be non-zero");
        let mut pass = encoder.compute("mesh-reference");
        let mut pen = pass.with(&self.pipeline);
        pen.bind(
            0,
            &MeshRefData {
                g_camera: camera,
                g_params: self.params,
                g_mesh_tlas: self.tlas,
                g_triangle_colors: self.color_buf.at(0),
                g_out: output,
            },
        );
        pen.dispatch([size[0].div_ceil(8), size[1].div_ceil(8), 1]);
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
        context.destroy_acceleration_structure(self.tlas);
        context.destroy_acceleration_structure(self.blas);
        context.destroy_buffer(self.instance_buf);
        context.destroy_buffer(self.color_buf);
        context.destroy_buffer(self.mesh_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::{MeshRefParams, ReferenceMesh};

    fn unit_mesh() -> ReferenceMesh {
        ReferenceMesh {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![[0, 1, 2]],
            triangle_colors: vec![[1.0, 0.5, 0.25]],
        }
    }

    #[test]
    fn uniform_layout_matches_the_wgsl_struct() {
        assert_eq!(std::mem::size_of::<MeshRefParams>(), 32);
        assert_eq!(std::mem::offset_of!(MeshRefParams, background), 16);
    }

    #[test]
    fn a_well_formed_mesh_validates() {
        assert!(unit_mesh().validate().is_ok());
        assert_eq!(unit_mesh().triangle_count(), 1);
    }

    #[test]
    fn malformed_meshes_are_rejected() {
        let empty = ReferenceMesh::default();
        assert!(empty.validate().is_err());

        let mut short_colors = unit_mesh();
        short_colors.triangle_colors.clear();
        assert!(short_colors.validate().is_err());

        let mut bad_index = unit_mesh();
        bad_index.indices = vec![[0, 1, 9]];
        assert!(bad_index.validate().is_err());

        let mut bad_vertex = unit_mesh();
        bad_vertex.positions[0][1] = f32::NAN;
        assert!(bad_vertex.validate().is_err());
    }
}
