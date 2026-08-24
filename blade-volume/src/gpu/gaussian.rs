use blade_graphics as gpu;

use std::{collections, mem, ptr, slice};

pub struct InitParameters {
    pub min_opacity: f32,
}

/// GPU representation of a Gaussian point cloud for ray-traced rendering.
pub struct GaussianGpuCloud {
    mesh_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    pub gauss_buf: gpu::Buffer,
    blas: gpu::AccelerationStructure,
    pub tlas: gpu::AccelerationStructure,
    /// Number of opacity-visible Gaussians packed into the TLAS/data buffer.
    pub num_points: usize,
}

fn visible_point_indices(model: &crate::PointCloudModel, min_opacity: f32) -> Vec<usize> {
    model
        .points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| (point.w > min_opacity).then_some(index))
        .collect()
}

fn gaussian_proxy() -> (Vec<[f32; 3]>, Vec<[u16; 3]>) {
    let shape = crate::Icosahedron::new(1.0);
    let mut vertices: Vec<_> = shape
        .vertices
        .into_iter()
        .map(|vertex| glam::Vec3::from(vertex).normalize())
        .collect();
    let mut midpoints = collections::HashMap::new();
    let mut triangles = Vec::with_capacity(shape.triangles.len() * 4);
    for triangle in shape.triangles {
        let mut middle = [0u16; 3];
        for (slot, edge) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ]
        .into_iter()
        .enumerate()
        {
            let key = if edge.0 < edge.1 {
                edge
            } else {
                (edge.1, edge.0)
            };
            middle[slot] = *midpoints.entry(key).or_insert_with(|| {
                let index = vertices.len() as u16;
                vertices.push((vertices[key.0 as usize] + vertices[key.1 as usize]).normalize());
                index
            });
        }
        triangles.extend_from_slice(&[
            [triangle[0], middle[0], middle[2]],
            [triangle[1], middle[1], middle[0]],
            [triangle[2], middle[2], middle[1]],
            [middle[0], middle[1], middle[2]],
        ]);
    }
    // Project the new vertices onto one sphere, then expand by the smallest
    // face distance. Every face consequently lies on or outside the unit
    // support sphere, so the triangle proxy stays conservative.
    let inradius = triangles
        .iter()
        .map(|triangle| {
            let a = vertices[triangle[0] as usize];
            let b = vertices[triangle[1] as usize];
            let c = vertices[triangle[2] as usize];
            (b - a).cross(c - a).normalize().dot(a).abs()
        })
        .fold(f32::INFINITY, f32::min);
    for vertex in &mut vertices {
        *vertex /= inradius;
    }
    (
        vertices
            .into_iter()
            .map(|vertex| vertex.to_array())
            .collect(),
        triangles,
    )
}

impl GaussianGpuCloud {
    /// Creates a GPU point cloud from a unified model.
    ///
    /// Requires the model to have `transforms` (rotation + scale).
    pub fn new(
        model: &crate::PointCloudModel,
        params: &InitParameters,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        model
            .validate()
            .unwrap_or_else(|err| panic!("invalid Gaussian model: {err}"));
        let transforms = model
            .transforms
            .as_ref()
            .expect("GaussianGpuCloud requires transforms");
        assert!(
            params.min_opacity.is_finite() && params.min_opacity > 0.0,
            "Gaussian min_opacity must be finite and positive"
        );

        let visible_indices = visible_point_indices(model, params.min_opacity);
        let count = visible_indices.len();
        assert!(
            count > 0,
            "Gaussian model has no points above min_opacity {}",
            params.min_opacity
        );
        let gauss_total_size = (count * mem::size_of::<crate::GaussianGpu>()) as u64;
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
                slice::from_raw_parts_mut(gauss_scratch.data() as *mut crate::GaussianGpu, count)
            };
            for (packed_index, gg) in gaussians_gpu.iter_mut().enumerate() {
                let model_index = visible_indices[packed_index];
                let point = model.points[model_index];
                gg.mean = [point.x, point.y, point.z];
                gg.rotation = transforms.rotations[model_index].into();
                gg.scale = transforms.scales[model_index].into();
                gg.opacity = point.w; // density/opacity stored in w
                let shc = model.get_sh_coefficients(model_index);
                for (h, c) in gg.harmonics.iter_mut().zip(shc.iter()) {
                    *h = (*c, 0)
                }
            }
        }

        let (vertices, triangles) = gaussian_proxy();
        let vertex_data_size = (vertices.len() * mem::size_of::<[f32; 3]>()) as u64;
        let index_data_size = (triangles.len() * mem::size_of::<[u16; 3]>()) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "gauss-mesh",
            size: vertex_data_size + index_data_size,
            memory: gpu::Memory::Device,
        });
        let meshes = [gpu::AccelerationStructureMesh {
            vertex_data: mesh_buf.at(0),
            vertex_format: gpu::VertexFormat::F32Vec3,
            vertex_stride: mem::size_of::<[f32; 3]>() as u32,
            vertex_count: vertices.len() as u32,
            index_data: mesh_buf.at(vertex_data_size),
            index_type: Some(gpu::IndexType::U16),
            triangle_count: triangles.len() as u32,
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
        let instances = visible_indices
            .iter()
            .map(|&model_index| {
                let point = model.points[model_index];
                let rotation = transforms.rotations[model_index];
                let scale = transforms.scales[model_index];
                let opacity = point.w;
                let mean = glam::Vec3::new(point.x, point.y, point.z);

                gpu::AccelerationStructureInstance {
                    acceleration_structure_index: 0,
                    transform: {
                        let extra_scale =
                            (2.0 * (opacity / params.min_opacity).ln().max(0.0)).sqrt();
                        let m = glam::Mat3::from_quat(rotation)
                            * glam::Mat3::from_diagonal(extra_scale * scale);
                        mint::ColumnMatrix3x4 {
                            x: m.x_axis.into(),
                            y: m.y_axis.into(),
                            z: m.z_axis.into(),
                            w: mean.into(),
                        }
                        .into()
                    },
                    mask: 0xFF,
                    custom_index: 0,
                }
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
                vertices.as_ptr(),
                mesh_stage.data() as *mut [f32; 3],
                vertices.len(),
            );
            ptr::copy_nonoverlapping(
                triangles.as_ptr(),
                mesh_stage.data().add(vertex_data_size as usize) as *mut [u16; 3],
                triangles.len(),
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
        let _ = context.wait_for(&sync_point, !0);

        context.destroy_buffer(gauss_scratch);
        context.destroy_buffer(scratch_buf);
        context.destroy_buffer(mesh_stage);

        Self {
            mesh_buf,
            instance_buf,
            gauss_buf,
            blas,
            tlas,
            num_points: count,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn filter_model() -> crate::PointCloudModel {
        let points = vec![
            glam::Vec4::new(10.0, 0.0, 0.0, 0.001),
            glam::Vec4::new(20.0, 0.0, 0.0, 0.5),
            glam::Vec4::new(30.0, 0.0, 0.0, 0.01),
            glam::Vec4::new(40.0, 0.0, 0.0, 0.8),
        ];
        crate::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: Some(crate::Transforms {
                rotations: vec![glam::Quat::IDENTITY; points.len()],
                scales: vec![glam::Vec3::ONE; points.len()],
                pbr: None,
            }),
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
            points,
        }
    }

    #[test]
    fn opacity_filter_preserves_model_order_for_instance_indices() {
        let model = filter_model();
        assert_eq!(visible_point_indices(&model, 0.01), vec![1, 3]);
    }

    #[test]
    fn subdivided_proxy_conservatively_encloses_unit_support() {
        let (vertices, triangles) = gaussian_proxy();
        assert_eq!(vertices.len(), 42);
        assert_eq!(triangles.len(), 80);
        let mut minimum_face_distance = f32::INFINITY;
        for triangle in triangles {
            let a = glam::Vec3::from(vertices[triangle[0] as usize]);
            let b = glam::Vec3::from(vertices[triangle[1] as usize]);
            let c = glam::Vec3::from(vertices[triangle[2] as usize]);
            let distance = (b - a).cross(c - a).normalize().dot(a).abs();
            assert!(distance >= 1.0 - 1.0e-6);
            minimum_face_distance = minimum_face_distance.min(distance);
        }
        assert!((minimum_face_distance - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn gpu_data_uses_the_same_filtered_order_as_instances() {
        if crate::gpu::access_disabled() {
            eprintln!("skipping Gaussian filtering GPU test: GPU access disabled");
            return;
        }
        let Some(context) = (unsafe {
            gpu::Context::init(gpu::ContextDesc {
                ray_tracing: true,
                ..gpu::ContextDesc::default()
            })
            .ok()
        }) else {
            eprintln!("skipping Gaussian filtering GPU test: no ray-query GPU");
            return;
        };
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "gaussian-filter-test",
            buffer_count: 1,
            manual_barriers: false,
        });
        let mut cloud = GaussianGpuCloud::new(
            &filter_model(),
            &InitParameters { min_opacity: 0.01 },
            &context,
            &mut encoder,
        );
        assert_eq!(cloud.num_points, 2);
        let size = (cloud.num_points * mem::size_of::<crate::GaussianGpu>()) as u64;
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "gaussian-filter-readback",
            size,
            memory: gpu::Memory::Shared,
        });
        encoder.start();
        if let mut transfer = encoder.transfer("gaussian-filter-readback") {
            transfer.copy_buffer_to_buffer(cloud.gauss_buf.at(0), readback.at(0), size);
        }
        let sync = context.submit(&mut encoder);
        let _ = context.wait_for(&sync, !0);
        let packed = unsafe {
            slice::from_raw_parts(
                readback.data() as *const crate::GaussianGpu,
                cloud.num_points,
            )
        };
        assert_eq!(packed[0].mean, [20.0, 0.0, 0.0]);
        assert_eq!(packed[1].mean, [40.0, 0.0, 0.0]);
        assert_eq!(packed[0].opacity, 0.5);
        assert_eq!(packed[1].opacity, 0.8);

        context.destroy_buffer(readback);
        cloud.deinit(&context);
        context.destroy_command_encoder(&mut encoder);
    }
}
