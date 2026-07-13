//! GPU path recorder for differentiable training.
//!
//! Mirrors the CPU `vol::trace::record_path` walk on the GPU, writing
//! `(cell, dt, mask)` per `(pixel, step)` into flat buffers that feed straight
//! into the meganeura training inputs. PowerFoam paths also record the real
//! entry neighbor and exact local interval Jacobians for position/radius
//! training. Removes the
//! single-threaded CPU ray-marching bottleneck that dominated each Adam
//! step at full image resolution.
//!
//! Layout (row-major `[num_pixels, max_steps]`):
//!   - `previous_cells_out[u32]` (PowerFoam only)
//!   - `cells_out[u32]`
//!   - `dts_out[f32]`
//!   - `mask_out[f32]`
//!   - three `dt_grad_*_out[vec4<f32>]` streams (PowerFoam only)
//!
//! The shader writes only the steps it actually takes; trailing slots
//! keep their pre-dispatch value, which the caller zeroes before each
//! dispatch.

use std::{mem, ptr, slice};

use crate::{shaders, CameraParams};
use blade_graphics as gpu;

fn output_bytes(num_pixels: u32, max_steps: u32, with_jacobians: bool) -> u64 {
    let pl = num_pixels as u64 * max_steps as u64;
    let base = pl * (2 * mem::size_of::<u32>() + 2 * mem::size_of::<f32>()) as u64;
    if with_jacobians {
        base + pl * (mem::size_of::<u32>() + 3 * mem::size_of::<[f32; 4]>()) as u64
    } else {
        base + (mem::size_of::<u32>() + 3 * mem::size_of::<[f32; 4]>()) as u64
    }
}

/// Inputs to one path-record dispatch.
///
/// Buffer fields must already live on the device (typically from a
/// [`RadFoamGpuCloud`](crate::gpu::RadFoamGpuCloud) and caller-owned output
/// buffers). The recorder writes the base streams once per pixel, up to
/// `max_steps` entries, plus the differential streams for weighted clouds.
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
    power_foam: u32,
}

#[derive(blade_macros::ShaderData)]
struct PathRecordData {
    g_points: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_pixel_indices: gpu::BufferPiece,
    g_previous_cells_out: gpu::BufferPiece,
    g_cells_out: gpu::BufferPiece,
    g_next_cells_out: gpu::BufferPiece,
    g_dts_out: gpu::BufferPiece,
    g_mask_out: gpu::BufferPiece,
    g_dt_grad_previous_out: gpu::BufferPiece,
    g_dt_grad_current_out: gpu::BufferPiece,
    g_dt_grad_next_out: gpu::BufferPiece,
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

    /// Record `args.num_pixels` paths into the caller-owned output buffers.
    ///
    /// The caller is responsible for:
    ///   - zeroing every active output buffer before this call (the shader
    ///     only writes the steps that were actually taken);
    ///   - making sure every binding is valid for the right number of bytes;
    ///   - submitting the encoder afterwards.
    pub fn dispatch(
        &self,
        encoder: &mut gpu::CommandEncoder,
        cloud: &crate::gpu::RadFoamGpuCloud,
        buffers: &PathRecordBuffers,
        args: RecordPathsArgs,
    ) {
        assert!(
            !cloud.is_power_foam || buffers.has_jacobians,
            "PowerFoam path recording requires full Jacobian buffers"
        );
        assert!(
            args.num_pixels <= buffers.num_pixels,
            "path dispatch exceeds pixel buffer capacity"
        );
        assert_eq!(
            args.max_steps, buffers.max_steps,
            "path dispatch max_steps must match buffer layout"
        );
        let params = RecordParams {
            start_point: args.start_point,
            max_steps: args.max_steps,
            num_pixels: args.num_pixels,
            image_width: args.image_width,
            image_height: args.image_height,
            max_path_dt: args.max_path_dt,
            depth: args.depth,
            power_foam: cloud.is_power_foam as u32,
        };

        let mut pass = encoder.compute("radfoam-record-paths");
        let mut pc = pass.with(&self.pipeline);
        pc.bind(
            0,
            &PathRecordData {
                g_points: cloud.points(),
                g_adjacency: cloud.point_adjacency(),
                g_adjacency_offsets: cloud.point_adjacency_offsets(),
                g_pixel_indices: buffers.pixel_indices.into(),
                g_previous_cells_out: buffers.previous_cells.into(),
                g_cells_out: buffers.cells.into(),
                g_next_cells_out: buffers.next_cells.into(),
                g_dts_out: buffers.dts.into(),
                g_mask_out: buffers.mask.into(),
                g_dt_grad_previous_out: buffers.dt_grad_previous.into(),
                g_dt_grad_current_out: buffers.dt_grad_current.into(),
                g_dt_grad_next_out: buffers.dt_grad_next.into(),
                g_camera: args.camera,
                g_params: params,
            },
        );
        let groups = args.num_pixels.div_ceil(64);
        pc.dispatch([groups, 1, 1]);
    }
}

/// Allocate flat path outputs plus the pixel-index input buffer with the right
/// sizes for `(num_pixels, max_steps)`.
///
/// Caller owns lifetime and is responsible for destroying via
/// [`gpu::Context::destroy_buffer`].
pub struct PathRecordBuffers {
    pub pixel_indices: gpu::Buffer,
    pub previous_cells: gpu::Buffer,
    pub cells: gpu::Buffer,
    pub next_cells: gpu::Buffer,
    pub dts: gpu::Buffer,
    pub mask: gpu::Buffer,
    pub dt_grad_previous: gpu::Buffer,
    pub dt_grad_current: gpu::Buffer,
    pub dt_grad_next: gpu::Buffer,
    /// Upload-side staging for `pixel_indices` (write from CPU, copy to
    /// device via a `transfer` pass before dispatching). Persistent
    /// `Memory::Upload` is much cheaper than allocating staging every
    /// step.
    pub pixel_indices_stage: gpu::Buffer,
    pub num_pixels: u32,
    pub max_steps: u32,
    has_jacobians: bool,
}

impl PathRecordBuffers {
    pub fn new(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(context, num_pixels, max_steps, false, true)
    }

    /// Allocate only the four always-written path streams. The derivative
    /// bindings are valid one-element dummies, so this is only safe to dispatch
    /// with an unweighted cloud (`RadFoamGpuCloud::is_power_foam == false`).
    pub fn new_recorded_only(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(context, num_pixels, max_steps, false, false)
    }

    /// Allocate every output stream as `Memory::External(Fd(None))` so the
    /// caller can pass each one as an
    /// `ExternalMemorySource` into the consumer's `bind_external_buffer`
    /// (meganeura's slot import). Same-context external memory works on
    /// Vulkan; Metal/GLES backends `unimplemented!()` on the buffer
    /// allocation.
    pub fn new_external(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(context, num_pixels, max_steps, true, true)
    }

    /// Allocate exportable base streams and optionally full PowerFoam
    /// Jacobians. Set `with_jacobians` from `model.radii.is_some()`.
    pub fn new_external_with_jacobians(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        with_jacobians: bool,
    ) -> Self {
        Self::new_with(context, num_pixels, max_steps, true, with_jacobians)
    }

    fn new_with(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        external: bool,
        with_jacobians: bool,
    ) -> Self {
        let pl = (num_pixels as u64) * (max_steps as u64);
        let cells_bytes = pl * mem::size_of::<u32>() as u64;
        let dts_bytes = pl * mem::size_of::<f32>() as u64;
        let mask_bytes = pl * mem::size_of::<f32>() as u64;
        let role_bytes = if with_jacobians {
            cells_bytes
        } else {
            mem::size_of::<u32>() as u64
        };
        let jacobian_bytes = if with_jacobians {
            pl * mem::size_of::<[f32; 4]>() as u64
        } else {
            mem::size_of::<[f32; 4]>() as u64
        };
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
        let mem = |external_flag: bool| {
            if external_flag {
                // None = export side; the consumer will import via the
                // FD this side returns from get_external_buffer_source.
                #[cfg(target_os = "linux")]
                {
                    gpu::Memory::External(gpu::ExternalMemorySource::Fd(None))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let _ = external_flag;
                    gpu::Memory::Device
                }
            } else {
                gpu::Memory::Device
            }
        };
        let cells = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-cells",
            size: cells_bytes,
            memory: mem(external),
        });
        let previous_cells = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-previous-cells",
            size: role_bytes,
            memory: mem(external && with_jacobians),
        });
        let next_cells = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-next-cells",
            size: cells_bytes,
            memory: mem(external),
        });
        let dts = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dts",
            size: dts_bytes,
            memory: mem(external),
        });
        let mask = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-mask",
            size: mask_bytes,
            memory: mem(external),
        });
        let dt_grad_previous = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-previous",
            size: jacobian_bytes,
            memory: mem(external && with_jacobians),
        });
        let dt_grad_current = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-current",
            size: jacobian_bytes,
            memory: mem(external && with_jacobians),
        });
        let dt_grad_next = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-next",
            size: jacobian_bytes,
            memory: mem(external && with_jacobians),
        });

        Self {
            pixel_indices,
            previous_cells,
            cells,
            next_cells,
            dts,
            mask,
            dt_grad_previous,
            dt_grad_current,
            dt_grad_next,
            pixel_indices_stage,
            num_pixels,
            max_steps,
            has_jacobians: with_jacobians,
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
        self.write_pixel_indices_prefix(indices);
    }

    /// Write an active prefix of pixel indices for a partial dispatch.
    /// The dispatch's `num_pixels` must equal `indices.len()`; unused
    /// capacity is left untouched and must not be dispatched.
    pub fn write_pixel_indices_prefix(&self, indices: &[u32]) {
        assert!(
            indices.len() <= self.num_pixels as usize,
            "pixel index prefix exceeds buffer capacity",
        );
        unsafe {
            let dst = slice::from_raw_parts_mut(
                self.pixel_indices_stage.data() as *mut u32,
                indices.len(),
            );
            ptr::copy_nonoverlapping(indices.as_ptr(), dst.as_mut_ptr(), indices.len());
        }
    }

    pub fn has_jacobians(&self) -> bool {
        self.has_jacobians
    }

    /// Total allocated bytes of all output streams (for sanity checks).
    pub fn out_bytes(&self) -> u64 {
        output_bytes(self.num_pixels, self.max_steps, self.has_jacobians)
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.pixel_indices);
        context.destroy_buffer(self.pixel_indices_stage);
        context.destroy_buffer(self.previous_cells);
        context.destroy_buffer(self.cells);
        context.destroy_buffer(self.next_cells);
        context.destroy_buffer(self.dts);
        context.destroy_buffer(self.mask);
        context.destroy_buffer(self.dt_grad_previous);
        context.destroy_buffer(self.dt_grad_current);
        context.destroy_buffer(self.dt_grad_next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_path_buffers_do_not_scale_jacobian_storage() {
        let slots = 4_096_u64 * 256;
        assert_eq!(output_bytes(4_096, 256, true), slots * 68);
        assert_eq!(output_bytes(4_096, 256, false), slots * 16 + 52);
    }
}
