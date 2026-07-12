use blade_graphics as gpu;

use std::{mem, ptr, slice};

/// GPU-side storage for RadFoam point cloud rendering.
///
/// This uploads the buffers required by the RadFoam tracing kernel:
/// - `points`: `vec4<f32>[N]` where xyz is position and w is per-point radius
///   (Power Foam weight, or 0 for plain Voronoi)
/// - `attributes`: packed `f32[N * attr_dim]`, where `attr_dim = 1 + 3 * (1 + sh_degree)^2`
///   and the last scalar in each row is density
/// - `point_adjacency`: flattened neighbor list `u32[K]`
/// - `point_adjacency_offsets`: CSR offsets `u32[N+1]`
///
/// Notes:
/// - This does not build any hardware ray tracing acceleration structures.
/// - This is intended for compute-only Voronoi traversal.
pub struct RadFoamGpuCloud {
    points_buf: gpu::Buffer,
    attributes_buf: gpu::Buffer,
    point_adjacency_buf: gpu::Buffer,
    point_adjacency_offsets_buf: gpu::Buffer,

    pub sh_degree: usize,
    pub attr_dim: usize,
    pub num_points: usize,
    pub num_adjacency: usize,
}

impl RadFoamGpuCloud {
    /// Creates a GPU point cloud from a unified model.
    ///
    /// Requires the model to have `adjacency` data.
    pub fn new(
        model: &crate::PointCloudModel,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        model
            .validate()
            .unwrap_or_else(|err| panic!("invalid RadFoam model: {err}"));
        let adjacency = model
            .adjacency
            .as_ref()
            .expect("RadFoamGpuCloud requires adjacency");

        let num_points = model.len();
        assert_eq!(
            num_points + 1,
            adjacency.offsets.len(),
            "adjacency.offsets must have length N+1"
        );

        let num_adjacency = adjacency.neighbors.len();
        let sh_component_count = model.sh_component_count();
        let attr_dim = 1 + 3 * sh_component_count; // SH coefficients + density

        assert!(num_points > 0, "Model has zero points; nothing to upload");

        // Sizes
        let points_size = (num_points * mem::size_of::<[f32; 4]>()) as u64;
        let attrs_size = (num_points * attr_dim * mem::size_of::<f32>()) as u64;
        let adj_size = (num_adjacency * mem::size_of::<u32>()) as u64;
        let adj_off_size = (adjacency.offsets.len() * mem::size_of::<u32>()) as u64;

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
            // Points: write as `[f32; 4]` to match WGSL `array<vec4<f32>>` layout.
            // xyz = position; w = radius (Power Foam weight, 0 for plain Voronoi).
            // Density is read by the shader from `attributes`, not from here.
            let dst_points =
                slice::from_raw_parts_mut(points_stage.data() as *mut [f32; 4], num_points);
            let radii = model.radii.as_deref();
            for (i, dst) in dst_points.iter_mut().enumerate() {
                let p = model.points[i];
                dst[0] = p.x;
                dst[1] = p.y;
                dst[2] = p.z;
                dst[3] = radii.map_or(0.0, |r| r[i]);
            }

            // Attributes: pack as [sh_coeffs..., density] per point
            // This matches the shader's expected layout
            let dst_attrs = slice::from_raw_parts_mut(
                attributes_stage.data() as *mut f32,
                num_points * attr_dim,
            );
            for i in 0..num_points {
                let base = i * attr_dim;
                let sh_len = sh_component_count * 3;
                let sh_base = i * sh_len;
                dst_attrs[base..base + sh_len]
                    .copy_from_slice(&model.sh_coefficients[sh_base..sh_base + sh_len]);
                dst_attrs[base + sh_len] = model.points[i].w;
            }

            // Adjacency: contiguous u32 array
            if num_adjacency > 0 {
                ptr::copy_nonoverlapping(
                    adjacency.neighbors.as_ptr(),
                    adjacency_stage.data() as *mut u32,
                    num_adjacency,
                );
            }

            // Adjacency offsets: contiguous u32 array
            ptr::copy_nonoverlapping(
                adjacency.offsets.as_ptr(),
                adjacency_offsets_stage.data() as *mut u32,
                adjacency.offsets.len(),
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
        let _ = context.wait_for(&sync_point, !0);

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
