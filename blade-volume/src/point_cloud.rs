use blade_graphics as gpu;

use std::{mem, ptr, slice};

pub struct InitParameters {
    pub min_opacity: f32,
}

pub struct PointCloud {
    mesh_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    pub gauss_buf: gpu::Buffer,
    blas: gpu::AccelerationStructure,
    pub tlas: gpu::AccelerationStructure,
}

impl PointCloud {
    pub fn new(
        model: &super::Model,
        params: &InitParameters,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        let count = model.gaussians.len();
        let gauss_total_size = (count * mem::size_of::<super::GaussianGpu>()) as u64;
        let gauss_buf = context.create_buffer(gpu::BufferDesc {
            name: "gauss-blobs",
            size: gauss_total_size,
            memory: gpu::Memory::Device,
        });
        let gauss_scratch = context.create_buffer(gpu::BufferDesc {
            name: "gauss-upload",
            size: gauss_total_size,
            memory: gpu::Memory::Upload,
        });
        {
            let gaussians_gpu = unsafe {
                slice::from_raw_parts_mut(gauss_scratch.data() as *mut super::GaussianGpu, count)
            };
            for (gg, g) in gaussians_gpu.iter_mut().zip(&model.gaussians) {
                gg.mean = g.mean.into();
                gg.rotation = g.rotation.into();
                gg.scale = g.scale.into();
                gg.opacity = g.opacity;
                for (h, shc) in gg.harmonics.iter_mut().zip(g.shc.iter()) {
                    *h = (*shc, 0)
                }
            }
        }

        let inner_radius = 1.0;
        let geometry = super::Icosahedron::new(inner_radius);
        let vertex_data_size = (geometry.vertices.len() * mem::size_of::<[f32; 3]>()) as u64;
        let index_data_size = (geometry.triangles.len() * mem::size_of::<[u16; 3]>()) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "gauss-mesh",
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
            name: "blas",
            ty: gpu::AccelerationStructureType::BottomLevel,
            size: blas_sizes.data,
        });

        // Build instances
        let instances = model
            .gaussians
            .iter()
            .map(|g| gpu::AccelerationStructureInstance {
                acceleration_structure_index: 0,
                transform: {
                    let extra_scale = (2.0 * (g.opacity / params.min_opacity).ln().max(0.0)).sqrt();
                    let m = glam::Mat3::from_quat(g.rotation)
                        * glam::Mat3::from_diagonal(extra_scale * g.scale);
                    mint::ColumnMatrix3x4 {
                        x: m.x_axis.into(),
                        y: m.y_axis.into(),
                        z: m.z_axis.into(),
                        w: g.mean.into(),
                    }
                    .into()
                },
                mask: 0xFF,
                custom_index: 0,
            })
            .collect::<Vec<_>>();
        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&instances, &[blas]);

        // Build TLAS
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(count as u32);
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "TLAS",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
        });

        let tlas_scratch_offset =
            (blas_sizes.scratch | (gpu::limits::ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT - 1)) + 1;
        let scratch_buf = context.create_buffer(gpu::BufferDesc {
            name: "scratch",
            size: tlas_scratch_offset + tlas_sizes.scratch,
            memory: gpu::Memory::Device,
        });

        let mesh_stage = context.create_buffer(gpu::BufferDesc {
            name: "gauss-mesh-stage",
            size: vertex_data_size + index_data_size,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                geometry.vertices.as_ptr(),
                mesh_stage.data() as *mut [f32; 3],
                geometry.vertices.len(),
            );
            ptr::copy_nonoverlapping(
                geometry.triangles.as_ptr(),
                mesh_stage.data().add(vertex_data_size as usize) as *mut [u16; 3],
                geometry.triangles.len(),
            );
        }

        // Encode init operations
        encoder.start();
        if let mut pass = encoder.transfer("init") {
            pass.copy_buffer_to_buffer(
                mesh_stage.at(0),
                mesh_buf.at(0),
                vertex_data_size + index_data_size,
            );
            pass.copy_buffer_to_buffer(gauss_scratch.at(0), gauss_buf.at(0), gauss_total_size);
        }
        if let mut pass = encoder.acceleration_structure("bottom") {
            pass.build_bottom_level(blas, &meshes, scratch_buf.at(0));
        }
        if let mut pass = encoder.acceleration_structure("top") {
            pass.build_top_level(
                tlas,
                &[blas],
                count as u32,
                instance_buf.at(0),
                scratch_buf.at(tlas_scratch_offset),
            );
        }
        let sync_point = context.submit(encoder);
        context.wait_for(&sync_point, !0);

        context.destroy_buffer(gauss_scratch);
        context.destroy_buffer(scratch_buf);
        context.destroy_buffer(mesh_stage);

        Self {
            mesh_buf,
            instance_buf,
            gauss_buf,
            blas,
            tlas,
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.mesh_buf);
        context.destroy_buffer(self.gauss_buf);
        context.destroy_buffer(self.instance_buf);
        context.destroy_acceleration_structure(self.blas);
        context.destroy_acceleration_structure(self.tlas);
    }
}
