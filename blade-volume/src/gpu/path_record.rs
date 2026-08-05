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
//!   - `next_cells_out[u32]`
//!   - `dts_out[f32]`
//!   - `mask_out[f32]`
//!   - `path_status_out[u32]` (recorded count + truncation bit)
//!   - `dt_reference_tangents_out[f32]` (selected PowerFoam differential)
//!   - up to four `dt_grad_*_out[vec4<f32>]` streams (PowerFoam only)
//!   - `surface_queries_out[vec2<f32>]` (detail query near + plane-branch mask)
//!   - two `surface_query_grad_*_out[vec4<f32>]` streams (full-detail only)
//!
//! PowerFoam gather workgroups initialize the index and mask rows before
//! recording. The unweighted walk still leaves trailing slots at their
//! pre-dispatch value. Payload streams are only written for active steps.

use std::{mem, ptr, slice};

use crate::{shaders, CameraParams};
use blade_graphics as gpu;

const SPLAT_TILE_SIZE: u32 = 16;
const SPLAT_TILE_INDEX_BUDGET: u64 = 4 * 1024 * 1024;
const PARALLEL_SPLAT_MIN_AVERAGE_NEIGHBORS: usize = 32;
const MAX_PARALLEL_SPLAT_WORKGROUPS_PER_DIMENSION: u32 = 65_535;
// Sphere-hit count and surviving path depth are different budgets. Real
// learned-radius clouds can intersect many supports before radical-plane
// clipping leaves a much shorter disjoint path, so keep a useful candidate
// floor without inflating every path/Jacobian row.
const MIN_SPLAT_CANDIDATE_CAPACITY: u32 = 1024;
const PATH_TRUNCATED_BIT: u32 = 1 << 31;

/// Differential streams emitted beside a weighted path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum PathJacobianMode {
    /// Record exact intervals without local derivatives.
    #[default]
    None = 0,
    /// Record position/radius derivatives and, for oriented clouds, surface
    /// plane derivatives.
    Full = 1,
    /// Record only oriented surface-plane derivatives. The reference tangent
    /// contains only the normal and offset contribution.
    Surface = 2,
}

fn splat_candidate_capacity(max_steps: u32, minimum: u32) -> u32 {
    max_steps
        .saturating_mul(4)
        .max(MIN_SPLAT_CANDIDATE_CAPACITY)
        .max(minimum)
}

fn use_parallel_splat_recording(num_points: usize, num_adjacency: usize) -> bool {
    num_points != 0 && num_adjacency / num_points >= PARALLEL_SPLAT_MIN_AVERAGE_NEIGHBORS
}

fn parallel_splat_dispatch(num_pixels: u32) -> [u32; 3] {
    assert!(
        num_pixels > 0,
        "PowerFoam path recording needs at least one ray"
    );
    let groups_y = num_pixels.div_ceil(MAX_PARALLEL_SPLAT_WORKGROUPS_PER_DIMENSION);
    assert!(
        groups_y <= MAX_PARALLEL_SPLAT_WORKGROUPS_PER_DIMENSION,
        "PowerFoam path batch exceeds the two-dimensional dispatch limit"
    );
    let groups_x = num_pixels.div_ceil(groups_y);
    [groups_x, groups_y, 1]
}

fn output_bytes(
    num_pixels: u32,
    max_steps: u32,
    jacobian_mode: PathJacobianMode,
    with_surface_queries: bool,
) -> u64 {
    let pl = num_pixels as u64 * max_steps as u64;
    let base = pl * (2 * mem::size_of::<u32>() + 2 * mem::size_of::<f32>()) as u64;
    let path_status = num_pixels as u64 * mem::size_of::<u32>() as u64;
    let previous_cells = match jacobian_mode {
        PathJacobianMode::Full => pl * mem::size_of::<u32>() as u64,
        PathJacobianMode::None | PathJacobianMode::Surface => mem::size_of::<u32>() as u64,
    };
    let reference_tangents = match jacobian_mode {
        PathJacobianMode::None => mem::size_of::<f32>() as u64,
        PathJacobianMode::Full | PathJacobianMode::Surface => pl * mem::size_of::<f32>() as u64,
    };
    let geometry_jacobians = match jacobian_mode {
        PathJacobianMode::Full => 3 * pl * mem::size_of::<[f32; 4]>() as u64,
        PathJacobianMode::None | PathJacobianMode::Surface => 3 * mem::size_of::<[f32; 4]>() as u64,
    };
    let surface_jacobians = match jacobian_mode {
        PathJacobianMode::None => mem::size_of::<[f32; 4]>() as u64,
        PathJacobianMode::Full | PathJacobianMode::Surface => {
            pl * mem::size_of::<[f32; 4]>() as u64
        }
    };
    let surface_queries = if with_surface_queries {
        pl * mem::size_of::<[f32; 2]>() as u64
    } else {
        mem::size_of::<[f32; 2]>() as u64
    };
    let surface_query_jacobians = if with_surface_queries && jacobian_mode == PathJacobianMode::Full
    {
        2 * pl * mem::size_of::<[f32; 4]>() as u64
    } else {
        2 * mem::size_of::<[f32; 4]>() as u64
    };
    base + path_status
        + previous_cells
        + reference_tangents
        + geometry_jacobians
        + surface_jacobians
        + surface_queries
        + surface_query_jacobians
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
    /// First row in the caller's `[buffer_capacity, max_steps]` outputs.
    /// Pixel indices use the same offset. This allows multiple camera
    /// dispatches to fill one optimizer batch without copying path data.
    pub pixel_offset: u32,
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
    pixel_offset: u32,
    image_width: u32,
    image_height: u32,
    max_path_dt: f32,
    depth: f32,
    power_foam: u32,
    num_points: u32,
    candidate_capacity: u32,
    jacobian_mode: u32,
    tile_width: u32,
    tile_height: u32,
    tile_capacity: u32,
    oriented: u32,
}

#[derive(blade_macros::ShaderData)]
struct PathRecordData {
    g_points: gpu::BufferPiece,
    g_surface_normals: gpu::BufferPiece,
    g_surface_details: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_support_bvh: gpu::BufferPiece,
    g_pixel_indices: gpu::BufferPiece,
    g_previous_cells_out: gpu::BufferPiece,
    g_cells_out: gpu::BufferPiece,
    g_next_cells_out: gpu::BufferPiece,
    g_dts_out: gpu::BufferPiece,
    g_mask_out: gpu::BufferPiece,
    g_path_status_out: gpu::BufferPiece,
    g_dt_reference_tangents_out: gpu::BufferPiece,
    g_dt_grad_previous_out: gpu::BufferPiece,
    g_dt_grad_current_out: gpu::BufferPiece,
    g_dt_grad_next_out: gpu::BufferPiece,
    g_dt_grad_surface_normal_out: gpu::BufferPiece,
    g_surface_queries_out: gpu::BufferPiece,
    g_surface_query_grad_previous_out: gpu::BufferPiece,
    g_surface_query_grad_current_out: gpu::BufferPiece,
    g_candidate_counts: gpu::BufferPiece,
    g_candidates: gpu::BufferPiece,
    g_candidate_depths: gpu::BufferPiece,
    g_candidate_faces: gpu::BufferPiece,
    g_candidate_neighbors: gpu::BufferPiece,
    g_projected_bounds: gpu::BufferPiece,
    g_tile_counts: gpu::BufferPiece,
    g_tile_candidates: gpu::BufferPiece,
    g_camera: CameraParams,
    g_params: RecordParams,
}

/// Reusable compute pipeline for the path-recording shader.
///
/// Build once per session (the WGSL parse / SPIR-V compile is the
/// expensive part); dispatch many times per training step.
pub struct PathRecorder {
    walk_pipeline: gpu::ComputePipeline,
    splat_project_pipeline: gpu::ComputePipeline,
    splat_bin_pipeline: gpu::ComputePipeline,
    splat_gather_pipeline: gpu::ComputePipeline,
    splat_bvh_gather_pipeline: gpu::ComputePipeline,
    splat_record_pipeline: gpu::ComputePipeline,
    splat_parallel_record_pipeline: gpu::ComputePipeline,
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
        let walk_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-record-paths",
            data_layouts: &[&layout],
            compute: shader.at("record_paths"),
        });
        let splat_project_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-project-path-candidates",
            data_layouts: &[&layout],
            compute: shader.at("project_powerfoam_candidates"),
        });
        let splat_bin_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-bin-path-candidates",
            data_layouts: &[&layout],
            compute: shader.at("bin_powerfoam_candidates"),
        });
        let splat_gather_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-gather-path-candidates",
            data_layouts: &[&layout],
            compute: shader.at("gather_powerfoam_candidates"),
        });
        let splat_bvh_gather_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-gather-path-candidates-bvh",
            data_layouts: &[&layout],
            compute: shader.at("gather_powerfoam_bvh_candidates"),
        });
        let splat_record_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-record-splat-paths",
            data_layouts: &[&layout],
            compute: shader.at("record_powerfoam_splats"),
        });
        let splat_parallel_record_pipeline =
            context.create_compute_pipeline(gpu::ComputePipelineDesc {
                name: "powerfoam-record-splat-paths-parallel",
                data_layouts: &[&layout],
                compute: shader.at("record_powerfoam_splats_parallel"),
            });
        Self {
            walk_pipeline,
            splat_project_pipeline,
            splat_bin_pipeline,
            splat_gather_pipeline,
            splat_bvh_gather_pipeline,
            splat_record_pipeline,
            splat_parallel_record_pipeline,
        }
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.walk_pipeline);
        context.destroy_compute_pipeline(&mut self.splat_project_pipeline);
        context.destroy_compute_pipeline(&mut self.splat_bin_pipeline);
        context.destroy_compute_pipeline(&mut self.splat_gather_pipeline);
        context.destroy_compute_pipeline(&mut self.splat_bvh_gather_pipeline);
        context.destroy_compute_pipeline(&mut self.splat_record_pipeline);
        context.destroy_compute_pipeline(&mut self.splat_parallel_record_pipeline);
    }

    /// Whether this cloud has enough clipping constraints per site to benefit
    /// from assigning a complete workgroup to every ray.
    pub fn uses_parallel_powerfoam_recording(cloud: &crate::gpu::RadFoamGpuCloud) -> bool {
        use_parallel_splat_recording(cloud.num_points, cloud.num_adjacency)
    }

    /// Whether this cloud uses its shared support-sphere hierarchy instead of
    /// the exhaustive candidate gather.
    pub fn uses_support_bvh(cloud: &crate::gpu::RadFoamGpuCloud) -> bool {
        cloud.has_support_bvh
    }

    fn prepare_dispatch(
        &self,
        cloud: &crate::gpu::RadFoamGpuCloud,
        buffers: &PathRecordBuffers,
        args: RecordPathsArgs,
    ) -> (PathRecordData, u32) {
        assert!(
            !cloud.is_power_foam || buffers.has_splat_scratch,
            "PowerFoam path recording requires splat scratch buffers"
        );
        assert!(
            args.num_pixels <= buffers.num_pixels
                && args.pixel_offset <= buffers.num_pixels - args.num_pixels,
            "path dispatch exceeds pixel buffer capacity"
        );
        assert_eq!(
            args.max_steps, buffers.max_steps,
            "path dispatch max_steps must match buffer layout"
        );
        assert!(
            buffers.jacobian_mode != PathJacobianMode::Surface || cloud.is_oriented,
            "surface-only path Jacobians require an oriented cloud"
        );
        assert!(
            !cloud.has_surface_detail || buffers.has_surface_queries,
            "surface-detail path recording requires surface-query outputs"
        );
        let tile_width = args.image_width.div_ceil(SPLAT_TILE_SIZE);
        let tile_height = args.image_height.div_ceil(SPLAT_TILE_SIZE);
        let tile_count = tile_width
            .checked_mul(tile_height)
            .expect("path image has too many projected tiles");
        if buffers.has_projected_splat_tiles() {
            assert!(
                tile_count <= buffers.splat_tile_count,
                "path image needs {tile_count} projected tiles, but buffers hold {}",
                buffers.splat_tile_count,
            );
            assert!(
                cloud.num_points as u32 <= buffers.splat_projected_point_capacity,
                "cloud has {} points, but projected buffers hold {}",
                cloud.num_points,
                buffers.splat_projected_point_capacity,
            );
        }
        let params = RecordParams {
            start_point: args.start_point,
            max_steps: args.max_steps,
            num_pixels: args.num_pixels,
            pixel_offset: args.pixel_offset,
            image_width: args.image_width,
            image_height: args.image_height,
            max_path_dt: args.max_path_dt,
            depth: args.depth,
            power_foam: cloud.is_power_foam as u32,
            num_points: cloud.num_points as u32,
            candidate_capacity: buffers.splat_candidate_capacity,
            jacobian_mode: buffers.jacobian_mode as u32,
            tile_width,
            tile_height,
            tile_capacity: buffers.splat_tile_capacity,
            oriented: cloud.is_oriented as u32 | (cloud.has_surface_detail as u32) << 1,
        };

        let data = PathRecordData {
            g_points: cloud.points(),
            g_surface_normals: cloud.surface_normals(),
            g_surface_details: cloud.surface_details(),
            g_adjacency: cloud.point_adjacency(),
            g_adjacency_offsets: cloud.point_adjacency_offsets(),
            g_support_bvh: cloud.support_bvh(),
            g_pixel_indices: buffers.pixel_indices.into(),
            g_previous_cells_out: buffers.previous_cells.into(),
            g_cells_out: buffers.cells.into(),
            g_next_cells_out: buffers.next_cells.into(),
            g_dts_out: buffers.dts.into(),
            g_mask_out: buffers.mask.into(),
            g_path_status_out: buffers.path_status.into(),
            g_dt_reference_tangents_out: buffers.dt_reference_tangents.into(),
            g_dt_grad_previous_out: buffers.dt_grad_previous.into(),
            g_dt_grad_current_out: buffers.dt_grad_current.into(),
            g_dt_grad_next_out: buffers.dt_grad_next.into(),
            g_dt_grad_surface_normal_out: buffers.dt_grad_surface_normal.into(),
            g_surface_queries_out: buffers.surface_queries.into(),
            g_surface_query_grad_previous_out: buffers.surface_query_grad_previous.into(),
            g_surface_query_grad_current_out: buffers.surface_query_grad_current.into(),
            g_candidate_counts: buffers.splat_candidate_counts.into(),
            g_candidates: buffers.splat_candidates.into(),
            g_candidate_depths: buffers.splat_candidate_depths.into(),
            g_candidate_faces: buffers.splat_candidate_faces.into(),
            g_candidate_neighbors: buffers.splat_candidate_neighbors.into(),
            g_projected_bounds: buffers.splat_projected_bounds.into(),
            g_tile_counts: buffers.splat_tile_counts.into(),
            g_tile_candidates: buffers.splat_tile_candidates.into(),
            g_camera: args.camera,
            g_params: params,
        };
        (data, tile_count)
    }

    /// Record `args.num_pixels` paths into the caller-owned output buffers.
    ///
    /// The caller is responsible for:
    ///   - zeroing payload buffers before their first use;
    ///   - zeroing index and mask rows before unweighted dispatches (weighted
    ///     gather workgroups initialize those rows themselves);
    ///   - making sure every binding is valid for the right number of bytes;
    ///   - submitting the encoder afterwards.
    pub fn dispatch(
        &self,
        encoder: &mut gpu::CommandEncoder,
        cloud: &crate::gpu::RadFoamGpuCloud,
        buffers: &PathRecordBuffers,
        args: RecordPathsArgs,
    ) {
        let (data, tile_count) = self.prepare_dispatch(cloud, buffers, args);
        if cloud.is_power_foam {
            if buffers.has_projected_splat_tiles() {
                {
                    let mut pass = encoder.compute("powerfoam-project-path-candidates");
                    let mut pc = pass.with(&self.splat_project_pipeline);
                    pc.bind(0, &data);
                    pc.dispatch([(cloud.num_points as u32).div_ceil(64), 1, 1]);
                }
                let mut pass = encoder.compute("powerfoam-bin-path-candidates");
                let mut pc = pass.with(&self.splat_bin_pipeline);
                pc.bind(0, &data);
                pc.dispatch([tile_count, 1, 1]);
            }
            if buffers.has_projected_splat_tiles() || !Self::uses_support_bvh(cloud) {
                let mut pass = encoder.compute("powerfoam-gather-path-candidates");
                let mut pc = pass.with(&self.splat_gather_pipeline);
                pc.bind(0, &data);
                pc.dispatch([args.num_pixels, 1, 1]);
            } else {
                let mut pass = encoder.compute("powerfoam-gather-path-candidates-bvh");
                let mut pc = pass.with(&self.splat_bvh_gather_pipeline);
                pc.bind(0, &data);
                pc.dispatch([args.num_pixels, 1, 1]);
            }
            if Self::uses_parallel_powerfoam_recording(cloud) {
                let mut pass = encoder.compute("powerfoam-record-splat-paths-parallel");
                let mut pc = pass.with(&self.splat_parallel_record_pipeline);
                pc.bind(0, &data);
                pc.dispatch(parallel_splat_dispatch(args.num_pixels));
            } else {
                let mut pass = encoder.compute("powerfoam-record-splat-paths");
                let mut pc = pass.with(&self.splat_record_pipeline);
                pc.bind(0, &data);
                let groups = args.num_pixels.div_ceil(64);
                pc.dispatch([groups, 1, 1]);
            }
            return;
        }

        let mut pass = encoder.compute("radfoam-record-paths");
        let mut pc = pass.with(&self.walk_pipeline);
        pc.bind(0, &data);
        let groups = args.num_pixels.div_ceil(64);
        pc.dispatch([groups, 1, 1]);
    }

    /// Record multiple disjoint camera slices into one output batch.
    ///
    /// PowerFoam slices share one gather pass and one record pass,
    /// while retaining a separate camera and pixel range per dispatch. Other
    /// modes use the ordinary dispatch sequence. Arguments must be ordered by
    /// non-overlapping `pixel_offset` ranges.
    pub fn dispatch_batch(
        &self,
        encoder: &mut gpu::CommandEncoder,
        cloud: &crate::gpu::RadFoamGpuCloud,
        buffers: &PathRecordBuffers,
        args: &[RecordPathsArgs],
    ) {
        if args.is_empty() {
            return;
        }
        for pair in args.windows(2) {
            let first_end = pair[0]
                .pixel_offset
                .checked_add(pair[0].num_pixels)
                .expect("path dispatch pixel range overflow");
            assert!(
                first_end <= pair[1].pixel_offset,
                "batched path dispatch ranges must be ordered and disjoint"
            );
        }
        if !cloud.is_power_foam || buffers.has_projected_splat_tiles() {
            for &arg in args {
                self.dispatch(encoder, cloud, buffers, arg);
            }
            return;
        }

        let data = args
            .iter()
            .map(|&arg| self.prepare_dispatch(cloud, buffers, arg).0)
            .collect::<Vec<_>>();
        {
            let (name, pipeline) = if Self::uses_support_bvh(cloud) {
                (
                    "powerfoam-gather-path-candidate-batch-bvh",
                    &self.splat_bvh_gather_pipeline,
                )
            } else {
                (
                    "powerfoam-gather-path-candidate-batch",
                    &self.splat_gather_pipeline,
                )
            };
            let mut pass = encoder.compute(name);
            let mut pc = pass.with(pipeline);
            for (datum, arg) in data.iter().zip(args) {
                pc.bind(0, datum);
                pc.dispatch([arg.num_pixels, 1, 1]);
            }
        }
        if Self::uses_parallel_powerfoam_recording(cloud) {
            let mut pass = encoder.compute("powerfoam-record-splat-path-batch-parallel");
            let mut pc = pass.with(&self.splat_parallel_record_pipeline);
            for (datum, arg) in data.iter().zip(args) {
                pc.bind(0, datum);
                pc.dispatch(parallel_splat_dispatch(arg.num_pixels));
            }
        } else {
            let mut pass = encoder.compute("powerfoam-record-splat-path-batch");
            let mut pc = pass.with(&self.splat_record_pipeline);
            for (datum, arg) in data.iter().zip(args) {
                pc.bind(0, datum);
                pc.dispatch([arg.num_pixels.div_ceil(64), 1, 1]);
            }
        }
    }
}

/// Summary of synchronized GPU path-record output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PathRecordStats {
    /// Largest number of active entries written for one ray.
    pub max_steps_used: u32,
    /// Rays which exhausted the path budget while another valid segment
    /// remained.
    pub truncated_rays: usize,
}

/// Flat path outputs plus the pixel-index input buffer with the right sizes
/// for `(num_pixels, max_steps)`.
///
/// Caller owns lifetime and is responsible for destroying via
/// [`Self::destroy`].
pub struct PathRecordBuffers {
    pub pixel_indices: gpu::Buffer,
    pub previous_cells: gpu::Buffer,
    pub cells: gpu::Buffer,
    pub next_cells: gpu::Buffer,
    pub dts: gpu::Buffer,
    pub mask: gpu::Buffer,
    /// Host-visible recorded-entry count and truncation flag per ray.
    path_status: gpu::Buffer,
    /// Reference tangent for the streams selected by [`PathJacobianMode`].
    /// Raw recorded intervals remain available in [`Self::dts`].
    pub dt_reference_tangents: gpu::Buffer,
    pub dt_grad_previous: gpu::Buffer,
    pub dt_grad_current: gpu::Buffer,
    pub dt_grad_next: gpu::Buffer,
    pub dt_grad_surface_normal: gpu::Buffer,
    /// Pre-surface near distance and base-plane-query mask for spatial detail.
    pub surface_queries: gpu::Buffer,
    /// Fixed-topology derivatives of the detail query depth with respect to
    /// the entry neighbor and current site's `(x, y, z, radius)`.
    pub surface_query_grad_previous: gpu::Buffer,
    pub surface_query_grad_current: gpu::Buffer,
    /// Host-visible number of intersected supports for every sampled ray.
    /// Values larger than [`Self::splat_candidate_capacity`] signal that the
    /// bounded scratch row overflowed and the dispatch must be rejected.
    splat_candidate_counts: gpu::Buffer,
    /// Device-only fixed-size candidate rows used by weighted compute splats.
    splat_candidates: gpu::Buffer,
    /// Gathered sphere roots, then compacted clipped entry depths.
    splat_candidate_depths: gpu::Buffer,
    /// Cached radical-plane entry/exit depths.
    splat_candidate_faces: gpu::Buffer,
    /// Cached radical-plane entry/exit neighbors.
    splat_candidate_neighbors: gpu::Buffer,
    /// Camera-specific conservative tile bounds, one `vec4<u32>` per site.
    splat_projected_bounds: gpu::Buffer,
    /// Per-camera projected tile occupancy. Counts may exceed the bounded row
    /// capacity; rays in such a tile fall back to the exhaustive point scan.
    splat_tile_counts: gpu::Buffer,
    /// Device-only fixed-size projected candidate rows, one per screen tile.
    splat_tile_candidates: gpu::Buffer,
    /// Upload-side staging for `pixel_indices` (write from CPU, copy to
    /// device via a `transfer` pass before dispatching). Persistent
    /// `Memory::Upload` is much cheaper than allocating staging every
    /// step.
    pub pixel_indices_stage: gpu::Buffer,
    pub num_pixels: u32,
    pub max_steps: u32,
    jacobian_mode: PathJacobianMode,
    has_surface_queries: bool,
    has_splat_scratch: bool,
    splat_candidate_capacity: u32,
    splat_tile_count: u32,
    splat_tile_capacity: u32,
    splat_projected_point_capacity: u32,
}

impl PathRecordBuffers {
    pub fn new(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            false,
            PathJacobianMode::Full,
            true,
            0,
            None,
            false,
        )
    }

    /// Allocate full path/Jacobian streams plus a conservative projected-tile
    /// candidate index for images up to `image_resolution`.
    pub fn new_projected(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        max_points: u32,
        image_resolution: [u32; 2],
        min_candidate_capacity: u32,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            false,
            PathJacobianMode::Full,
            true,
            min_candidate_capacity,
            Some((image_resolution, max_points)),
            false,
        )
    }

    /// Allocate only the four always-written path streams. The derivative
    /// bindings are valid one-element dummies, so this is only safe to dispatch
    /// with an unweighted cloud (`RadFoamGpuCloud::is_power_foam == false`).
    pub fn new_recorded_only(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            false,
            PathJacobianMode::None,
            false,
            0,
            None,
            false,
        )
    }

    /// Allocate compact base path streams plus PowerFoam candidate scratch,
    /// without the geometry-derivative streams used only by training.
    pub fn new_powerfoam_recorded_only(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            false,
            PathJacobianMode::None,
            true,
            0,
            None,
            false,
        )
    }

    /// Compact PowerFoam rendering streams with projected candidate tiles.
    pub fn new_powerfoam_recorded_only_projected(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        max_points: u32,
        image_resolution: [u32; 2],
        min_candidate_capacity: u32,
        with_surface_queries: bool,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            false,
            PathJacobianMode::None,
            true,
            min_candidate_capacity,
            Some((image_resolution, max_points)),
            with_surface_queries,
        )
    }

    /// Allocate every output stream as `Memory::External(Fd(None))` so the
    /// caller can pass each one as an
    /// `ExternalMemorySource` into the consumer's `bind_external_buffer`
    /// (meganeura's slot import). Same-context external memory works on
    /// Vulkan; Metal/GLES backends `unimplemented!()` on the buffer
    /// allocation.
    pub fn new_external(context: &gpu::Context, num_pixels: u32, max_steps: u32) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            true,
            PathJacobianMode::Full,
            true,
            0,
            None,
            false,
        )
    }

    /// Allocate exportable base streams and optionally full PowerFoam
    /// Jacobians. Set `with_jacobians` from `model.radii.is_some()`.
    pub fn new_external_with_jacobians(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        with_jacobians: bool,
        min_candidate_capacity: u32,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            true,
            if with_jacobians {
                PathJacobianMode::Full
            } else {
                PathJacobianMode::None
            },
            with_jacobians,
            min_candidate_capacity,
            None,
            false,
        )
    }

    /// Exportable training streams with conservative projected candidates for
    /// images up to `image_resolution`.
    pub fn new_external_with_jacobians_projected(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        with_jacobians: bool,
        max_points: u32,
        image_resolution: [u32; 2],
        min_candidate_capacity: u32,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            true,
            if with_jacobians {
                PathJacobianMode::Full
            } else {
                PathJacobianMode::None
            },
            with_jacobians,
            min_candidate_capacity,
            Some((image_resolution, max_points)),
            false,
        )
    }

    /// Exportable PowerFoam streams with the selected differential payload.
    pub fn new_external_powerfoam(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        jacobian_mode: PathJacobianMode,
        min_candidate_capacity: u32,
        with_surface_queries: bool,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            true,
            jacobian_mode,
            true,
            min_candidate_capacity,
            None,
            with_surface_queries,
        )
    }

    /// Exportable PowerFoam streams with projected candidates and the
    /// selected differential payload.
    #[allow(clippy::too_many_arguments)]
    pub fn new_external_powerfoam_projected(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        jacobian_mode: PathJacobianMode,
        max_points: u32,
        image_resolution: [u32; 2],
        min_candidate_capacity: u32,
        with_surface_queries: bool,
    ) -> Self {
        Self::new_with(
            context,
            num_pixels,
            max_steps,
            true,
            jacobian_mode,
            true,
            min_candidate_capacity,
            Some((image_resolution, max_points)),
            with_surface_queries,
        )
    }

    fn new_with(
        context: &gpu::Context,
        num_pixels: u32,
        max_steps: u32,
        external: bool,
        jacobian_mode: PathJacobianMode,
        with_splat_scratch: bool,
        min_splat_candidate_capacity: u32,
        projected: Option<([u32; 2], u32)>,
        with_surface_queries: bool,
    ) -> Self {
        assert!(max_steps > 0, "path-record max_steps must be non-zero");
        assert!(
            max_steps < PATH_TRUNCATED_BIT,
            "path-record max_steps exceeds status encoding"
        );
        let pl = (num_pixels as u64) * (max_steps as u64);
        let cells_bytes = pl * mem::size_of::<u32>() as u64;
        let dts_bytes = pl * mem::size_of::<f32>() as u64;
        let mask_bytes = pl * mem::size_of::<f32>() as u64;
        let role_bytes = if jacobian_mode == PathJacobianMode::Full {
            cells_bytes
        } else {
            mem::size_of::<u32>() as u64
        };
        let geometry_jacobian_bytes = if jacobian_mode == PathJacobianMode::Full {
            pl * mem::size_of::<[f32; 4]>() as u64
        } else {
            mem::size_of::<[f32; 4]>() as u64
        };
        let surface_jacobian_bytes = if jacobian_mode != PathJacobianMode::None {
            pl * mem::size_of::<[f32; 4]>() as u64
        } else {
            mem::size_of::<[f32; 4]>() as u64
        };
        let reference_tangent_bytes = if jacobian_mode != PathJacobianMode::None {
            dts_bytes
        } else {
            mem::size_of::<f32>() as u64
        };
        let surface_query_bytes = if with_surface_queries {
            pl * mem::size_of::<[f32; 2]>() as u64
        } else {
            mem::size_of::<[f32; 2]>() as u64
        };
        let has_surface_query_jacobians =
            with_surface_queries && jacobian_mode == PathJacobianMode::Full;
        let surface_query_jacobian_bytes = if has_surface_query_jacobians {
            pl * mem::size_of::<[f32; 4]>() as u64
        } else {
            mem::size_of::<[f32; 4]>() as u64
        };
        let pix_bytes = (num_pixels as u64) * mem::size_of::<u32>() as u64;
        let splat_candidate_capacity = if with_splat_scratch {
            splat_candidate_capacity(max_steps, min_splat_candidate_capacity)
        } else {
            1
        };
        let candidate_count_bytes = if with_splat_scratch {
            pix_bytes
        } else {
            mem::size_of::<u32>() as u64
        };
        let candidate_bytes = if with_splat_scratch {
            (num_pixels as u64) * (splat_candidate_capacity as u64) * mem::size_of::<u32>() as u64
        } else {
            mem::size_of::<u32>() as u64
        };
        let (splat_tile_count, splat_tile_capacity, splat_projected_point_capacity) =
            match projected {
                Some(([width, height], max_points)) => {
                    assert!(
                        with_splat_scratch,
                        "projected candidates require PowerFoam scratch buffers"
                    );
                    assert!(
                        width > 0 && height > 0,
                        "projected candidate resolution must be non-zero"
                    );
                    assert!(max_points > 0, "projected point capacity must be non-zero");
                    let count = width
                        .div_ceil(SPLAT_TILE_SIZE)
                        .checked_mul(height.div_ceil(SPLAT_TILE_SIZE))
                        .expect("projected candidate image has too many tiles");
                    let capacity =
                        (SPLAT_TILE_INDEX_BUDGET / u64::from(count)).clamp(1, 16_384) as u32;
                    (count, capacity, max_points)
                }
                None => (1, 0, 1),
            };
        let tile_count_bytes = u64::from(splat_tile_count) * mem::size_of::<u32>() as u64;
        let tile_candidate_bytes = u64::from(splat_tile_count)
            * u64::from(splat_tile_capacity.max(1))
            * mem::size_of::<u32>() as u64;
        let projected_bounds_bytes =
            u64::from(splat_projected_point_capacity) * mem::size_of::<[u32; 4]>() as u64;

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
            memory: mem(external && jacobian_mode == PathJacobianMode::Full),
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
        let path_status = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-status",
            size: pix_bytes,
            memory: gpu::Memory::Shared,
        });
        let dt_reference_tangents = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-reference-tangents",
            size: reference_tangent_bytes,
            memory: mem(external && jacobian_mode != PathJacobianMode::None),
        });
        let dt_grad_previous = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-previous",
            size: geometry_jacobian_bytes,
            memory: mem(external && jacobian_mode == PathJacobianMode::Full),
        });
        let dt_grad_current = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-current",
            size: geometry_jacobian_bytes,
            memory: mem(external && jacobian_mode == PathJacobianMode::Full),
        });
        let dt_grad_next = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-next",
            size: geometry_jacobian_bytes,
            memory: mem(external && jacobian_mode == PathJacobianMode::Full),
        });
        let dt_grad_surface_normal = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-dt-grad-surface-normal",
            size: surface_jacobian_bytes,
            memory: mem(external && jacobian_mode != PathJacobianMode::None),
        });
        let surface_queries = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-surface-queries",
            size: surface_query_bytes,
            memory: mem(external && with_surface_queries),
        });
        let surface_query_grad_previous = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-surface-query-grad-previous",
            size: surface_query_jacobian_bytes,
            memory: mem(external && has_surface_query_jacobians),
        });
        let surface_query_grad_current = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-path-record-surface-query-grad-current",
            size: surface_query_jacobian_bytes,
            memory: mem(external && has_surface_query_jacobians),
        });
        let splat_candidate_counts = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-candidate-counts",
            size: candidate_count_bytes,
            memory: gpu::Memory::Shared,
        });
        let splat_candidates = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-candidates",
            size: candidate_bytes,
            memory: gpu::Memory::Device,
        });
        let splat_candidate_depths = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-candidate-depths",
            size: candidate_bytes,
            memory: gpu::Memory::Device,
        });
        let splat_candidate_faces = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-candidate-faces",
            size: candidate_bytes * 2,
            memory: gpu::Memory::Device,
        });
        let splat_candidate_neighbors = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-candidate-neighbors",
            size: candidate_bytes * 2,
            memory: gpu::Memory::Device,
        });
        let splat_projected_bounds = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-projected-bounds",
            size: projected_bounds_bytes,
            memory: gpu::Memory::Device,
        });
        let splat_tile_counts = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-tile-counts",
            size: tile_count_bytes,
            memory: gpu::Memory::Shared,
        });
        let splat_tile_candidates = context.create_buffer(gpu::BufferDesc {
            name: "powerfoam-path-tile-candidates",
            size: tile_candidate_bytes,
            memory: gpu::Memory::Device,
        });

        Self {
            pixel_indices,
            previous_cells,
            cells,
            next_cells,
            dts,
            mask,
            path_status,
            dt_reference_tangents,
            dt_grad_previous,
            dt_grad_current,
            dt_grad_next,
            dt_grad_surface_normal,
            surface_queries,
            surface_query_grad_previous,
            surface_query_grad_current,
            splat_candidate_counts,
            splat_candidates,
            splat_candidate_depths,
            splat_candidate_faces,
            splat_candidate_neighbors,
            splat_projected_bounds,
            splat_tile_counts,
            splat_tile_candidates,
            pixel_indices_stage,
            num_pixels,
            max_steps,
            jacobian_mode,
            has_surface_queries: with_surface_queries,
            has_splat_scratch: with_splat_scratch,
            splat_candidate_capacity,
            splat_tile_count,
            splat_tile_capacity,
            splat_projected_point_capacity,
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
        self.jacobian_mode != PathJacobianMode::None
    }

    pub fn has_geometry_jacobians(&self) -> bool {
        self.jacobian_mode == PathJacobianMode::Full
    }

    pub fn has_surface_jacobians(&self) -> bool {
        self.jacobian_mode != PathJacobianMode::None
    }

    pub fn has_surface_queries(&self) -> bool {
        self.has_surface_queries
    }

    pub fn has_surface_query_jacobians(&self) -> bool {
        self.has_surface_queries && self.jacobian_mode == PathJacobianMode::Full
    }

    pub fn jacobian_mode(&self) -> PathJacobianMode {
        self.jacobian_mode
    }

    /// Summarize a synchronized path-record output range.
    ///
    /// The caller must wait for the recording submission before reading this
    /// host-visible status buffer.
    pub fn path_stats(&self, range: std::ops::Range<usize>) -> PathRecordStats {
        assert!(
            range.start <= range.end && range.end <= self.num_pixels as usize,
            "path status range exceeds buffer capacity",
        );
        let status = unsafe {
            slice::from_raw_parts(
                self.path_status.data() as *const u32,
                self.num_pixels as usize,
            )
        };
        let mut stats = PathRecordStats::default();
        for &value in &status[range] {
            stats.max_steps_used = stats.max_steps_used.max(value & !PATH_TRUNCATED_BIT);
            stats.truncated_rays += usize::from(value & PATH_TRUNCATED_BIT != 0);
        }
        stats
    }

    /// Maximum number of sphere hits observed in a synchronized output range.
    ///
    /// The caller must wait for the recording submission before reading this
    /// host-visible counter buffer. Panics for compact unweighted buffers.
    pub fn max_splat_candidate_count(&self, range: std::ops::Range<usize>) -> u32 {
        assert!(
            self.has_splat_scratch,
            "splat counters require PowerFoam scratch buffers"
        );
        assert!(
            range.start <= range.end && range.end <= self.num_pixels as usize,
            "splat counter range exceeds buffer capacity",
        );
        let counts = unsafe {
            slice::from_raw_parts(
                self.splat_candidate_counts.data() as *const u32,
                self.num_pixels as usize,
            )
        };
        counts[range].iter().copied().max().unwrap_or(0)
    }

    pub fn splat_candidate_capacity(&self) -> u32 {
        self.splat_candidate_capacity
    }

    pub fn has_projected_splat_tiles(&self) -> bool {
        self.splat_tile_capacity > 0
    }

    /// Largest projected tile row after a synchronized single-camera
    /// dispatch. Values above [`Self::splat_tile_capacity`] used the exact
    /// exhaustive fallback for rays in that tile.
    pub fn max_splat_tile_candidate_count(&self, image_resolution: [u32; 2]) -> u32 {
        assert!(
            self.has_projected_splat_tiles(),
            "projected tile counters require projected buffers"
        );
        let count = image_resolution[0]
            .div_ceil(SPLAT_TILE_SIZE)
            .checked_mul(image_resolution[1].div_ceil(SPLAT_TILE_SIZE))
            .expect("projected candidate image has too many tiles");
        assert!(
            count <= self.splat_tile_count,
            "projected tile range exceeds buffer capacity"
        );
        let counts = unsafe {
            slice::from_raw_parts(
                self.splat_tile_counts.data() as *const u32,
                self.splat_tile_count as usize,
            )
        };
        counts[..count as usize].iter().copied().max().unwrap_or(0)
    }

    pub fn splat_tile_capacity(&self) -> u32 {
        self.splat_tile_capacity
    }

    /// Total allocated bytes of all output streams (for sanity checks).
    pub fn out_bytes(&self) -> u64 {
        output_bytes(
            self.num_pixels,
            self.max_steps,
            self.jacobian_mode,
            self.has_surface_queries,
        )
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.pixel_indices);
        context.destroy_buffer(self.pixel_indices_stage);
        context.destroy_buffer(self.previous_cells);
        context.destroy_buffer(self.cells);
        context.destroy_buffer(self.next_cells);
        context.destroy_buffer(self.dts);
        context.destroy_buffer(self.mask);
        context.destroy_buffer(self.path_status);
        context.destroy_buffer(self.dt_reference_tangents);
        context.destroy_buffer(self.dt_grad_previous);
        context.destroy_buffer(self.dt_grad_current);
        context.destroy_buffer(self.dt_grad_next);
        context.destroy_buffer(self.dt_grad_surface_normal);
        context.destroy_buffer(self.surface_queries);
        context.destroy_buffer(self.surface_query_grad_previous);
        context.destroy_buffer(self.surface_query_grad_current);
        context.destroy_buffer(self.splat_candidate_counts);
        context.destroy_buffer(self.splat_candidates);
        context.destroy_buffer(self.splat_candidate_depths);
        context.destroy_buffer(self.splat_candidate_faces);
        context.destroy_buffer(self.splat_candidate_neighbors);
        context.destroy_buffer(self.splat_projected_bounds);
        context.destroy_buffer(self.splat_tile_counts);
        context.destroy_buffer(self.splat_tile_candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_params_match_wgsl_uniform_layout() {
        assert_eq!(mem::size_of::<RecordParams>(), 64);
    }

    #[test]
    fn compact_path_buffers_do_not_scale_jacobian_storage() {
        let slots = 4_096_u64 * 256;
        let path_status = 4_096_u64 * 4;
        assert_eq!(
            output_bytes(4_096, 256, PathJacobianMode::Full, false),
            slots * 88 + path_status + 40
        );
        assert_eq!(
            output_bytes(4_096, 256, PathJacobianMode::Surface, false),
            slots * 36 + 92 + path_status
        );
        assert_eq!(
            output_bytes(4_096, 256, PathJacobianMode::None, false),
            slots * 16 + 112 + path_status
        );
        assert_eq!(
            output_bytes(4_096, 256, PathJacobianMode::Surface, true),
            slots * 44 + 84 + path_status
        );
        assert_eq!(
            output_bytes(4_096, 256, PathJacobianMode::Full, true),
            slots * 128 + path_status
        );
    }

    #[test]
    fn powerfoam_candidate_floor_is_independent_of_short_path_rows() {
        assert_eq!(splat_candidate_capacity(64, 0), 1024);
        assert_eq!(splat_candidate_capacity(128, 0), 1024);
        assert_eq!(splat_candidate_capacity(256, 0), 1024);
        assert_eq!(splat_candidate_capacity(512, 0), 2048);
        assert_eq!(splat_candidate_capacity(128, 2048), 2048);
        assert_eq!(splat_candidate_capacity(128, 512), 1024);
    }

    #[test]
    fn parallel_powerfoam_recording_requires_dense_adjacency() {
        assert!(!use_parallel_splat_recording(0, 0));
        assert!(!use_parallel_splat_recording(200_000, 6_399_999));
        assert!(use_parallel_splat_recording(200_000, 6_400_000));
        assert!(use_parallel_splat_recording(200_000, 8_340_572));
    }

    #[test]
    fn parallel_powerfoam_dispatch_assigns_one_workgroup_per_ray() {
        assert_eq!(parallel_splat_dispatch(1), [1, 1, 1]);
        assert_eq!(parallel_splat_dispatch(4_096), [4_096, 1, 1]);
        assert_eq!(parallel_splat_dispatch(65_535), [65_535, 1, 1]);
        assert_eq!(parallel_splat_dispatch(65_536), [32_768, 2, 1]);
        assert_eq!(parallel_splat_dispatch(100_001), [50_001, 2, 1]);
    }
}
