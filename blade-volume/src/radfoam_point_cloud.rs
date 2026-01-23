use blade_graphics as gpu;

use std::{mem, ptr, slice};

/// GPU-side storage for an upstream Radiant Foam (RadFoam) scene.
///
/// This uploads the buffers required by the upstream tracing kernel:
/// - `points`: `float3[N]`
/// - `attributes`: packed `f32[N * attr_dim]`, where `attr_dim = 1 + 3 * (1 + sh_degree)^2`
///   and the last scalar in each row is density `s`.
/// - `point_adjacency`: flattened neighbor list `u32[K]`
/// - `point_adjacency_offsets`: CSR offsets `u32[N+1]`
///
/// Notes:
/// - This does not build any hardware ray tracing acceleration structures.
/// - This is intended for compute-only traversal.
pub struct RadFoamPointCloud {
    points_buf: gpu::Buffer,
    attributes_buf: gpu::Buffer,
    point_adjacency_buf: gpu::Buffer,
    point_adjacency_offsets_buf: gpu::Buffer,

    pub sh_degree: usize,
    pub attr_dim: usize,
    pub num_points: usize,
    pub num_adjacency: usize,
}

impl RadFoamPointCloud {
    pub fn new(
        model: &crate::RadFoamModel,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        assert_eq!(
            model.points.len() + 1,
            model.point_adjacency_offsets.len(),
            "RadFoamModel.point_adjacency_offsets must have length N+1"
        );
        let num_points = model.points.len();
        let num_adjacency = model.point_adjacency.len();
        let attr_dim = crate::RadFoamModel::attribute_dim(model.sh_degree);

        assert_eq!(
            model.attributes.len(),
            num_points * attr_dim,
            "RadFoamModel.attributes must have length N*attr_dim"
        );
        assert!(
            !model.points.is_empty(),
            "RadFoamModel has zero points; nothing to upload"
        );

        // Sizes
        //
        // IMPORTANT:
        // Our WGSL shaders declare points as `array<vec4<f32>>`, so we must upload points
        // with 16-byte stride (vec4) to match the GPU layout.
        let points_size = (num_points * mem::size_of::<[f32; 4]>()) as u64;
        let attrs_size = (model.attributes.len() * mem::size_of::<f32>()) as u64;
        let adj_size = (num_adjacency * mem::size_of::<u32>()) as u64;
        let adj_off_size = (model.point_adjacency_offsets.len() * mem::size_of::<u32>()) as u64;

        // Device buffers
        let points_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-points",
            size: points_size,
            memory: gpu::Memory::Device,
        });
        let attributes_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-attributes",
            size: attrs_size,
            memory: gpu::Memory::Device,
        });
        let point_adjacency_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency",
            size: adj_size,
            memory: gpu::Memory::Device,
        });
        let point_adjacency_offsets_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency-offsets",
            size: adj_off_size,
            memory: gpu::Memory::Device,
        });

        // Upload buffers
        let points_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-points-upload",
            size: points_size,
            memory: gpu::Memory::Upload,
        });
        let attributes_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-attributes-upload",
            size: attrs_size,
            memory: gpu::Memory::Upload,
        });
        let adjacency_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency-upload",
            size: adj_size,
            memory: gpu::Memory::Upload,
        });
        let adjacency_offsets_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency-offsets-upload",
            size: adj_off_size,
            memory: gpu::Memory::Upload,
        });

        // Fill staging buffers
        unsafe {
            // points: write as `[f32; 4]` to match WGSL `array<vec4<f32>>` layout.
            // We keep `w = 0.0` unused.
            let dst_points =
                slice::from_raw_parts_mut(points_stage.data() as *mut [f32; 4], num_points);
            for (dst, p) in dst_points.iter_mut().zip(model.points.iter()) {
                dst[0] = p.x;
                dst[1] = p.y;
                dst[2] = p.z;
                dst[3] = 0.0;
            }

            // attributes: contiguous f32 array
            ptr::copy_nonoverlapping(
                model.attributes.as_ptr(),
                attributes_stage.data() as *mut f32,
                model.attributes.len(),
            );

            // adjacency: contiguous u32 array
            ptr::copy_nonoverlapping(
                model.point_adjacency.as_ptr(),
                adjacency_stage.data() as *mut u32,
                model.point_adjacency.len(),
            );

            // adjacency offsets: contiguous u32 array
            ptr::copy_nonoverlapping(
                model.point_adjacency_offsets.as_ptr(),
                adjacency_offsets_stage.data() as *mut u32,
                model.point_adjacency_offsets.len(),
            );
        }

        // Encode transfers
        encoder.start();
        if let mut pass = encoder.transfer("radfoam-init") {
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
                pass.copy_buffer_to_buffer(
                    adjacency_stage.at(0),
                    point_adjacency_buf.at(0),
                    adj_size,
                );
            }
            if adj_off_size > 0 {
                pass.copy_buffer_to_buffer(
                    adjacency_offsets_stage.at(0),
                    point_adjacency_offsets_buf.at(0),
                    adj_off_size,
                );
            }
        }

        let sync_point = context.submit(encoder);
        context.wait_for(&sync_point, !0);

        // Free staging buffers
        context.destroy_buffer(points_stage);
        context.destroy_buffer(attributes_stage);
        context.destroy_buffer(adjacency_stage);
        context.destroy_buffer(adjacency_offsets_stage);

        Self {
            points_buf,
            attributes_buf,
            point_adjacency_buf,
            point_adjacency_offsets_buf,
            sh_degree: model.sh_degree,
            attr_dim,
            num_points,
            num_adjacency,
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.points_buf);
        context.destroy_buffer(self.attributes_buf);
        context.destroy_buffer(self.point_adjacency_buf);
        context.destroy_buffer(self.point_adjacency_offsets_buf);
    }

    /// Storage buffer view for point positions.
    pub fn points(&self) -> gpu::BufferPiece {
        self.points_buf.into()
    }

    /// Storage buffer view for packed attributes.
    pub fn attributes(&self) -> gpu::BufferPiece {
        self.attributes_buf.into()
    }

    /// Storage buffer view for flattened adjacency indices.
    pub fn point_adjacency(&self) -> gpu::BufferPiece {
        self.point_adjacency_buf.into()
    }

    /// Storage buffer view for CSR offsets.
    pub fn point_adjacency_offsets(&self) -> gpu::BufferPiece {
        self.point_adjacency_offsets_buf.into()
    }
}
