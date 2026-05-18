//! GPU path recorder for differentiable training.
//!
//! Mirrors the CPU `vol::trace::record_path` walk on the GPU, writing
//! `(cell, dt, mask)` per `(pixel, step)` into flat buffers that feed
//! straight into the meganeura training inputs. Removes the
//! single-threaded CPU ray-marching bottleneck that dominated each Adam
//! step at full image resolution.
//!
//! Layout (row-major `[num_pixels, max_steps]`):
//!   - `cells_out[u32]`
//!   - `dts_out[f32]`
//!   - `mask_out[f32]`
//!
//! The shader writes only the steps it actually takes; trailing slots
//! keep their pre-dispatch value, which the caller zeroes before each
//! dispatch.

use std::{mem, ptr, slice};

use crate::{shaders, CameraParams};
use blade_graphics as gpu;

/// Inputs to one path-record dispatch.
///
/// Buffer fields must already live on the device (typically from a
/// [`RadFoamGpuCloud`](crate::gpu::RadFoamGpuCloud) and three caller-
/// owned output buffers). The recorder writes into `cells_out`,
/// `dts_out`, and `mask_out` once per pixel, up to `max_steps` entries.
#[derive(Clone, Copy)]
pub struct RecordPathsArgs {
    pub camera: CameraParams,
    pub start_point: u32,
    pub max_steps: u32,
    pub image_width: u32,
    pub image_height: u32,
    /// Saturating cap for `dt` (matches CPU's `MAX_PATH_DT`).
    pub max_path_dt: f32,
    pub depth: f32,
    /// Number of rays to trace (= P in the `[P, L]` output).
    pub num_pixels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RecordParams {
    start_point: u32,
    max_steps: u32,
    num_pixels: u32,
    image_width: u32,
    image_height: u32,
    max_path_dt: f32,
    depth: f32,
    _pad: u32,
}

#[derive(blade_macros::ShaderData)]
struct PathRecordData {
    g_points: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_pixel_indices: gpu::BufferPiece,
    g_cells_out: gpu::BufferPiece,
    g_dts_out: gpu::BufferPiece,
    g_mask_out: gpu::BufferPiece,
    g_camera: CameraParams,
    g_params: RecordParams,
}

/// Reusable compute pipeline for the path-recording shader.
///
/// Build once per session (the WGSL parse / SPIR-V compile is the
/// expensive part); dispatch many times per training step.
pub struct PathRecorder {
    pipeline: gpu::ComputePipeline,
}

impl PathRecorder {
    pub fn new(context: &gpu::Context) -> Self {
        let raw = shaders::RADFOAM_RECORD_PATHS;
        let source = shaders::compose(raw);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <PathRecordData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-record-paths",
            data_layouts: &[&layout],
            compute: shader.at("record_paths"),
        });
        Self { pipeline }
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
    }

    /// Record `args.num_pixels` paths into the three caller-owned
    /// output buffers.
    ///
    /// The caller is responsible for:
    ///   - zeroing `cells_out`, `dts_out`, `mask_out` before this call
    ///     (the shader only writes the steps that were actually taken);
    ///   - making sure all four "in" buffers are valid for at least
    ///     `args.num_pixels * args.max_steps * sizeof(slot)` bytes and
    ///     `args.num_pixels * 4` bytes for `pixel_indices`;
    ///   - submitting the encoder afterwards.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        encoder: &mut gpu::CommandEncoder,
        cloud: &crate::gpu::RadFoamGpuCloud,
        pixel_indices: gpu::BufferPiece,
        cells_out: gpu::BufferPiece,
        dts_out: gpu::BufferPiece,
        mask_out: gpu::BufferPiece,
        args: RecordPathsArgs,
    ) {
        let params = RecordParams {
            start_point: args.start_point,
            max_steps: args.max_steps,
            num_pixels: args.num_pixels,
            image_width: args.image_width,
            image_height: args.image_height,
            max_path_dt: args.max_path_dt,
            depth: args.depth,
            _pad: 0,
        };

        let mut pass = encoder.compute("radfoam-record-paths");
        let mut pc = pass.with(&self.pipeline);
        pc.bind(
            0,
            &PathRecordData {
                g_points: cloud.points(),
                g_adjacency: cloud.point_adjacency(),
                g_adjacency_offsets: cloud.point_adjacency_offsets(),
                g_pixel_indices: pixel_indices,
                g_cells_out: cells_out,
                g_dts_out: dts_out,
                g_mask_out: mask_out,
                g_camera: args.camera,
                g_params: params,
            },
        );
        let groups = args.num_pixels.div_ceil(64);
        pc.dispatch([groups, 1, 1]);
    }
}

/// Allocate the three flat output buffers + the pixel-index input
/// buffer with the right sizes for `(num_pixels, max_steps)`.
///
/// Caller owns lifetime and is responsible for destroying via
/// [`gpu::Context::destroy_buffer`].
pub struct PathRecordBuffers {
    pub pixel_indices: gpu::Buffer,
    pub cells: gpu::Buffer,
    pub dts: gpu::Buffer,
    pub mask: gpu::Buffer,
    /// Upload-side staging for `pixel_indices` (write from CPU, copy to
    /// device via a `transfer` pass before dispatching). Persistent
    /// `Memory::Upload` is much cheaper than allocating staging every
    /// step.
    pub pixel_indices_stage: gpu::Buffer,
    pub num_pixels: u32,
    pub max_steps: u32,
}

impl PathRecordBuffers {
    pub fn new(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        let pl = (num_pixels as u64) * (max_steps as u64);
        let cells_bytes = pl * mem::size_of::<u32>() as u64;
        let dts_bytes = pl * mem::size_of::<f32>() as u64;
        let mask_bytes = pl * mem::size_of::<f32>() as u64;
        let pix_bytes = (num_pixels as u64) * mem::size_of::<u32>() as u64;

        let pixel_indices = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-pixels",
            size: pix_bytes,
            memory: gpu::Memory::Device,
        });
        let pixel_indices_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-pixels-upload",
            size: pix_bytes,
            memory: gpu::Memory::Upload,
        });
        let cells = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-cells",
            size: cells_bytes,
            memory: gpu::Memory::Device,
        });
        let dts = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dts",
            size: dts_bytes,
            memory: gpu::Memory::Device,
        });
        let mask = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-mask",
            size: mask_bytes,
            memory: gpu::Memory::Device,
        });

        Self {
            pixel_indices,
            cells,
            dts,
            mask,
            pixel_indices_stage,
            num_pixels,
            max_steps,
        }
    }

    /// Write `indices` to the upload-staging buffer; caller must issue
    /// a `transfer` pass to copy onto the device buffer before
    /// dispatching. Panics if `indices.len() != num_pixels`.
    pub fn write_pixel_indices(&self, indices: &[u32]) {
        assert_eq!(
            indices.len(),
            self.num_pixels as usize,
            "pixel index slice length must equal num_pixels",
        );
        unsafe {
            let dst = slice::from_raw_parts_mut(
                self.pixel_indices_stage.data() as *mut u32,
                indices.len(),
            );
            ptr::copy_nonoverlapping(indices.as_ptr(), dst.as_mut_ptr(), indices.len());
        }
    }

    /// Total bytes of all three output streams (for sanity checks).
    pub fn out_bytes(&self) -> u64 {
        let pl = (self.num_pixels as u64) * (self.max_steps as u64);
        pl * (mem::size_of::<u32>() + 2 * mem::size_of::<f32>()) as u64
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.pixel_indices);
        context.destroy_buffer(self.pixel_indices_stage);
        context.destroy_buffer(self.cells);
        context.destroy_buffer(self.dts);
        context.destroy_buffer(self.mask);
    }
}
