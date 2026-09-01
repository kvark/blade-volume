//! GPU compaction of active path records for sparse downstream work.

use std::{mem, ptr, slice};

use crate::shaders;
use blade_graphics as gpu;

const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

fn compact_dispatch(num_pixels: u32) -> [u32; 3] {
    assert!(num_pixels > 0);
    let groups_y = num_pixels.div_ceil(MAX_WORKGROUPS_PER_DIMENSION);
    assert!(groups_y <= MAX_WORKGROUPS_PER_DIMENSION);
    let groups_x = num_pixels.div_ceil(groups_y);
    [groups_x, groups_y, 1]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompactParams {
    num_pixels: u32,
    max_steps: u32,
    _padding: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct PathCompactData {
    g_path_status: gpu::BufferPiece,
    g_cells: gpu::BufferPiece,
    g_count: gpu::BufferPiece,
    g_dense_slots: gpu::BufferPiece,
    g_compact_cells: gpu::BufferPiece,
    g_pixel_indices: gpu::BufferPiece,
    g_active: gpu::BufferPiece,
    g_params: CompactParams,
}

/// Reusable compute pipeline that compacts active path-row prefixes.
pub struct PathCompactor {
    pipeline: gpu::ComputePipeline,
}

impl PathCompactor {
    pub fn new(context: &gpu::Context) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: shaders::PATH_COMPACT,
            naga_module: None,
        });
        let layout = <PathCompactData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "path-compact",
            data_layouts: &[&layout],
            compute: shader.at("compact_paths"),
        });
        Self { pipeline }
    }

    /// Append every active path segment to `output`.
    ///
    /// The caller must clear `output.count` and `output.active` before the
    /// dispatch. Index outputs need initialization only before their first use;
    /// inactive aligned tail rows are made neutral by the cleared active mask.
    pub fn dispatch(
        &self,
        encoder: &mut gpu::CommandEncoder,
        paths: &super::PathRecordBuffers,
        output: &PathCompactBuffers,
    ) {
        assert_eq!(paths.num_pixels, output.num_pixels);
        assert_eq!(paths.max_steps, output.max_steps);
        let data = PathCompactData {
            g_path_status: paths.path_status.into(),
            g_cells: paths.cells.into(),
            g_count: output.count.into(),
            g_dense_slots: output.dense_slots.into(),
            g_compact_cells: output.cells.into(),
            g_pixel_indices: output.pixel_indices.into(),
            g_active: output.active.into(),
            g_params: CompactParams {
                num_pixels: paths.num_pixels,
                max_steps: paths.max_steps,
                _padding: [0; 2],
            },
        };
        let mut pass = encoder.compute("path-compact");
        let mut compute = pass.with(&self.pipeline);
        compute.bind(0, &data);
        compute.dispatch(compact_dispatch(paths.num_pixels));
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
    }
}

/// Full-capacity compact path stream and its active count.
///
/// Index and mask buffers may be exported directly into a Meganeura session.
/// Storage remains sized for `num_pixels * max_steps`; only the live prefix is
/// dispatched by the consumer.
pub struct PathCompactBuffers {
    pub dense_slots: gpu::Buffer,
    pub cells: gpu::Buffer,
    pub pixel_indices: gpu::Buffer,
    pub active: gpu::Buffer,
    count: gpu::Buffer,
    host_visible_count: bool,
    pub num_pixels: u32,
    pub max_steps: u32,
}

impl PathCompactBuffers {
    /// Allocate host-visible outputs, primarily for validation and tooling.
    pub fn new(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(context, num_pixels, max_steps, false)
    }

    /// Allocate exportable outputs for direct Meganeura input binding.
    pub fn new_external(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(context, num_pixels, max_steps, true)
    }

    fn new_with(context: &gpu::Context, num_pixels: u32, max_steps: u32, external: bool) -> Self {
        assert!(num_pixels > 0);
        assert!(max_steps > 0);
        let records = u64::from(num_pixels) * u64::from(max_steps);
        let bytes = records * mem::size_of::<u32>() as u64;
        let memory = || {
            if external {
                #[cfg(target_os = "linux")]
                {
                    gpu::Memory::External(gpu::ExternalMemorySource::Fd(None))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    gpu::Memory::Device
                }
            } else {
                gpu::Memory::Shared
            }
        };
        let dense_slots = context.create_buffer(gpu::BufferDesc {
            name: "path-compact-dense-slots",
            size: bytes,
            memory: memory(),
        });
        let cells = context.create_buffer(gpu::BufferDesc {
            name: "path-compact-cells",
            size: bytes,
            memory: memory(),
        });
        let pixel_indices = context.create_buffer(gpu::BufferDesc {
            name: "path-compact-pixel-indices",
            size: bytes,
            memory: memory(),
        });
        let active = context.create_buffer(gpu::BufferDesc {
            name: "path-compact-active",
            size: bytes,
            memory: memory(),
        });
        let count = context.create_buffer(gpu::BufferDesc {
            name: "path-compact-count",
            size: mem::size_of::<u32>() as u64,
            memory: if external {
                memory()
            } else {
                gpu::Memory::Shared
            },
        });
        Self {
            dense_slots,
            cells,
            pixel_indices,
            active,
            count,
            host_visible_count: !external,
            num_pixels,
            max_steps,
        }
    }

    /// Clear per-dispatch state. Clear index storage on the first dispatch so
    /// aligned inactive tail gathers remain in bounds.
    pub fn clear(&self, transfer: &mut gpu::TransferCommandEncoder<'_>, initialize_indices: bool) {
        let bytes = u64::from(self.num_pixels) * u64::from(self.max_steps) * 4;
        transfer.fill_buffer(self.count.at(0), 4, 0);
        transfer.fill_buffer(self.active.at(0), bytes, 0);
        if initialize_indices {
            transfer.fill_buffer(self.dense_slots.at(0), bytes, 0);
            transfer.fill_buffer(self.cells.at(0), bytes, 0);
            transfer.fill_buffer(self.pixel_indices.at(0), bytes, 0);
        }
    }

    /// Read the synchronized number of active output records.
    pub fn active_count(&self) -> u32 {
        assert!(
            self.host_visible_count,
            "an external compact count must be consumed on the GPU"
        );
        unsafe {
            let source = slice::from_raw_parts(self.count.data() as *const u32, 1);
            let mut value = 0_u32;
            ptr::copy_nonoverlapping(source.as_ptr(), &mut value, 1);
            value
        }
    }

    pub fn count(&self) -> gpu::Buffer {
        self.count
    }

    pub fn capacity(&self) -> u32 {
        self.num_pixels
            .checked_mul(self.max_steps)
            .expect("compact path capacity overflows u32")
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.dense_slots);
        context.destroy_buffer(self.cells);
        context.destroy_buffer(self.pixel_indices);
        context.destroy_buffer(self.active);
        context.destroy_buffer(self.count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_params_match_wgsl_uniform_layout() {
        assert_eq!(mem::size_of::<CompactParams>(), 16);
    }

    #[test]
    fn compact_dispatch_covers_large_ray_batches() {
        assert_eq!(compact_dispatch(1), [1, 1, 1]);
        assert_eq!(compact_dispatch(4_096), [4_096, 1, 1]);
        assert_eq!(compact_dispatch(65_535), [65_535, 1, 1]);
        assert_eq!(compact_dispatch(65_536), [32_768, 2, 1]);
        assert_eq!(compact_dispatch(100_001), [50_001, 2, 1]);
    }
}
