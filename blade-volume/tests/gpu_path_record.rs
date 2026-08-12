//! Verify the GPU path-record shader matches CPU `record_path`.
//!
//! Builds a tiny Voronoi cloud (regular grid in a corner) with known
//! adjacency, traces a few rays both on CPU and GPU, and compares the
//! `(cell, dt, mask)` outputs entry-by-entry. The GPU path-record is
//! the replacement for the single-threaded CPU path tracer inside the
//! diff-render training loop — this test gates correctness.

use blade_graphics as gpu;
use blade_volume as vol;
use vol::gpu::{
    PathJacobianMode, PathRecordBuffers, PathRecordStats, PathRecorder, RadFoamGpuCloud,
    RecordPathsArgs,
};

// Some physical GPU drivers can busy-wait when two contexts are initialized
// concurrently in one test process. Keep these hardware tests independent
// without forcing the rest of the workspace test suite to run serially.
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn try_init_gpu() -> Option<gpu::Context> {
    if vol::gpu::access_disabled() {
        return None;
    }
    let desc = gpu::ContextDesc::default();
    unsafe { gpu::Context::init(desc) }.ok()
}

fn make_camera_looking_along_x(depth: f32) -> vol::CameraParams {
    // Camera at origin looking down +X axis.
    vol::CameraParams {
        cam_position: [-2.0, 0.0, 0.0],
        depth,
        cam_orientation: [
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
        ], // 90 deg about Y → forward = +X
        fov: [0.8, 0.8],
        principal: [0.05, -0.025],
    }
}

fn build_grid_model(n: usize) -> vol::PointCloudModel {
    // Regular grid of Voronoi sites along the X axis: each site at
    // (i, 0, 0) for i in 0..n. Adjacency: each site links to its
    // immediate +X / -X neighbours.
    let mut points = Vec::with_capacity(n);
    let mut sh = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n + 1);
    let mut neighbours = Vec::new();
    offsets.push(0u32);
    for i in 0..n {
        points.push(glam::Vec4::new(i as f32, 0.0, 0.0, 0.0));
        // sh degree 0: 3 floats per point (RGB), unused for record_paths
        sh.extend_from_slice(&[0.0, 0.0, 0.0]);
        if i > 0 {
            neighbours.push((i - 1) as u32);
        }
        if i + 1 < n {
            neighbours.push((i + 1) as u32);
        }
        offsets.push(neighbours.len() as u32);
    }
    vol::PointCloudModel {
        points,
        sh_coefficients: sh,
        sh_degree: 0,
        transforms: None,
        adjacency: Some(vol::Adjacency {
            neighbors: neighbours,
            offsets,
        }),
        radii: None,
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    }
}

struct CpuRecord {
    previous_cells: Vec<u32>,
    cells: Vec<u32>,
    next_cells: Vec<u32>,
    dts: Vec<f32>,
    dt_reference_tangents: Vec<f32>,
    mask: Vec<f32>,
    dt_grad_previous: Vec<[f32; 4]>,
    dt_grad_current: Vec<[f32; 4]>,
    dt_grad_next: Vec<[f32; 4]>,
    dt_grad_surface_normal: Vec<[f32; 4]>,
    surface_queries: Vec<[f32; 2]>,
    surface_query_grad_previous: Vec<[f32; 4]>,
    surface_query_grad_current: Vec<[f32; 4]>,
    surface_offsets: Vec<f32>,
}

fn point_geometry_relative(
    model: &vol::PointCloudModel,
    index: u32,
    ray_origin: glam::Vec3,
) -> [f32; 4] {
    let point = model.points[index as usize];
    let relative = point.truncate() - ray_origin;
    let radius = model
        .radii
        .as_ref()
        .map_or(0.0, |radii| radii[index as usize]);
    [relative.x, relative.y, relative.z, radius]
}

fn dot4(left: &[f32; 4], right: &[f32; 4]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn surface_normal_geometry(
    model: &vol::PointCloudModel,
    index: u32,
    effective_offset: f32,
) -> [f32; 4] {
    model.surface_normals.as_ref().map_or([0.0; 4], |normals| {
        normals[index as usize]
            .normalize()
            .extend(effective_offset)
            .to_array()
    })
}

fn cpu_record(
    model: &vol::PointCloudModel,
    rays: &[vol::trace::Ray],
    start_cell: u32,
    max_steps: usize,
    depth: f32,
) -> CpuRecord {
    let settings = vol::trace::TraceSettings {
        start_point: start_cell,
        max_steps: max_steps as u32,
        weight_threshold: 0.0,
        depth,
        eval_mode: vol::trace::EvalMode::Sh,
    };
    let p = rays.len();
    let mut previous_cells = vec![0u32; p * max_steps];
    let mut cells = vec![0u32; p * max_steps];
    let mut next_cells = vec![0u32; p * max_steps];
    let mut dts = vec![0.0f32; p * max_steps];
    let mut dt_reference_tangents = vec![0.0f32; p * max_steps];
    let mut mask = vec![0.0f32; p * max_steps];
    let mut dt_grad_previous = vec![[0.0; 4]; p * max_steps];
    let mut dt_grad_current = vec![[0.0; 4]; p * max_steps];
    let mut dt_grad_next = vec![[0.0; 4]; p * max_steps];
    let mut dt_grad_surface_normal = vec![[0.0; 4]; p * max_steps];
    let mut surface_queries = vec![[0.0; 2]; p * max_steps];
    let mut surface_query_grad_previous = vec![[0.0; 4]; p * max_steps];
    let mut surface_query_grad_current = vec![[0.0; 4]; p * max_steps];
    let mut surface_offsets = vec![0.0; p * max_steps];
    for (k, ray) in rays.iter().enumerate() {
        let path = if model.radii.is_some() {
            vol::trace::record_powerfoam_splats_jacobians(model, *ray, settings)
        } else {
            vol::trace::record_path_jacobians(model, *ray, settings)
        };
        for (idx, e) in path.entries.iter().take(max_steps).enumerate() {
            let slot = k * max_steps + idx;
            previous_cells[slot] = e.previous_cell;
            cells[slot] = e.cell;
            next_cells[slot] = e.next_cell;
            if e.dt.is_finite() && e.dt > 0.0 {
                dts[slot] = e.dt.min(50.0);
                mask[slot] = 1.0;
                if e.dt <= 50.0 {
                    dt_grad_previous[slot] = e.dt_d_previous.to_array();
                    dt_grad_current[slot] = e.dt_d_current.to_array();
                    dt_grad_next[slot] = e.dt_d_next.to_array();
                    dt_grad_surface_normal[slot] = e.dt_d_surface_normal.to_array();
                }
                surface_offsets[slot] = e.surface_offset;
                if let Some(ref normals) = model.surface_normals {
                    let normal = normals[e.cell as usize].normalize();
                    surface_queries[slot] = [
                        e.surface_query_near,
                        (path.ray_dir.dot(normal) < -1.0e-20) as u32 as f32,
                    ];
                    surface_query_grad_previous[slot] = e.surface_query_d_previous.to_array();
                    surface_query_grad_current[slot] = e.surface_query_d_current.to_array();
                }
                if model.radii.is_some() {
                    dt_reference_tangents[slot] = dot4(
                        &dt_grad_previous[slot],
                        &point_geometry_relative(model, e.previous_cell, ray.origin),
                    ) + dot4(
                        &dt_grad_current[slot],
                        &point_geometry_relative(model, e.cell, ray.origin),
                    ) + dot4(
                        &dt_grad_next[slot],
                        &point_geometry_relative(model, e.next_cell, ray.origin),
                    ) + dot4(
                        &dt_grad_surface_normal[slot],
                        &surface_normal_geometry(model, e.cell, e.surface_offset),
                    );
                }
            }
        }
    }
    CpuRecord {
        previous_cells,
        cells,
        next_cells,
        dts,
        dt_reference_tangents,
        mask,
        dt_grad_previous,
        dt_grad_current,
        dt_grad_next,
        dt_grad_surface_normal,
        surface_queries,
        surface_query_grad_previous,
        surface_query_grad_current,
        surface_offsets,
    }
}

fn pixel_indices_for_rays(width: u32, height: u32, count: u32) -> Vec<u32> {
    // Spread the rays across the image so they exercise different
    // angles. With width >> count this picks every (width/count)'th
    // pixel along the centre row.
    let row = height / 2;
    let stride = (width / count).max(1);
    (0..count).map(|i| row * width + i * stride).collect()
}

fn rays_for_pixels(
    cam: &vol::CameraParams,
    pixels: &[u32],
    width: u32,
    height: u32,
) -> Vec<vol::trace::Ray> {
    let tan_half = glam::Vec2::new((0.5 * cam.fov[0]).tan(), (0.5 * cam.fov[1]).tan());
    let origin = glam::Vec3::from_array(cam.cam_position);
    let q = glam::Quat::from_xyzw(
        cam.cam_orientation[0],
        cam.cam_orientation[1],
        cam.cam_orientation[2],
        cam.cam_orientation[3],
    );
    let wf = width as f32;
    let hf = height as f32;
    pixels
        .iter()
        .map(|&pidx| {
            let ix = pidx % width;
            let iy = pidx / width;
            let px = (ix as f32 + 0.5) / wf;
            let py = (iy as f32 + 0.5) / hf;
            let ndc_x = px * 2.0 - 1.0;
            let ndc_y = py * 2.0 - 1.0;
            let local = glam::Vec3::new(
                (ndc_x - cam.principal[0]) * tan_half.x,
                (ndc_y - cam.principal[1]) * tan_half.y,
                1.0,
            );
            vol::trace::Ray {
                origin,
                direction: (q * local).normalize(),
            }
        })
        .collect()
}

fn assert_gpu_path_record_matches_cpu(
    model: vol::PointCloudModel,
    world_translation: glam::Vec3,
    expect_tile_overflow: bool,
    expect_path_truncation: bool,
) {
    assert_gpu_path_record_matches_cpu_with_mode(
        model,
        world_translation,
        expect_tile_overflow,
        expect_path_truncation,
        false,
        PathJacobianMode::Full,
    );
}

fn assert_gpu_batched_path_record_matches_cpu(
    model: vol::PointCloudModel,
    world_translation: glam::Vec3,
    expect_path_truncation: bool,
) {
    assert_gpu_path_record_matches_cpu_with_mode(
        model,
        world_translation,
        false,
        expect_path_truncation,
        true,
        PathJacobianMode::Full,
    );
}

fn assert_gpu_path_record_matches_cpu_with_mode(
    mut model: vol::PointCloudModel,
    world_translation: glam::Vec3,
    expect_tile_overflow: bool,
    expect_path_truncation: bool,
    batched_exhaustive: bool,
    jacobian_mode: PathJacobianMode,
) {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .expect("GPU path-record test lock poisoned");
    let Some(ctx) = try_init_gpu() else {
        eprintln!("skipping: no GPU");
        return;
    };

    let max_steps = 16usize;
    let num_pixels = 8u32;
    let width = 64u32;
    let height = 64u32;
    let depth = 100.0f32;

    assert!(!model.points.is_empty());
    for point in &mut model.points {
        *point += world_translation.extend(0.0);
    }
    let mut camera = make_camera_looking_along_x(depth);
    camera.cam_position =
        (glam::Vec3::from_array(camera.cam_position) + world_translation).to_array();
    let mut cameras = [camera; 2];
    if batched_exhaustive {
        cameras[1].principal[0] -= 0.025;
        cameras[1].principal[1] += 0.015;
    }
    let pixels = pixel_indices_for_rays(width, height, num_pixels);
    let pixel_split = pixels.len() / 2;
    let mut rays = rays_for_pixels(&cameras[0], &pixels[..pixel_split], width, height);
    rays.extend(rays_for_pixels(
        &cameras[1],
        &pixels[pixel_split..],
        width,
        height,
    ));

    let weighted = model.radii.is_some();
    let with_surface_queries = model.surface_detail.is_some();
    let cpu = cpu_record(&model, &rays, 0, max_steps, depth);
    assert!(
        cpu.mask.iter().any(|&m| m != 0.0),
        "fixture records no segments"
    );

    let mut encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gpu-path-record-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut cloud = RadFoamGpuCloud::new_path_recording(&model, &ctx, &mut encoder);
    if weighted && batched_exhaustive && model.points.len() >= 32 * 1024 {
        assert!(PathRecorder::uses_support_bvh(&cloud));
    }
    let mut recorder = PathRecorder::new(&ctx);
    let mut bufs = if weighted && !batched_exhaustive {
        if jacobian_mode == PathJacobianMode::Full && !with_surface_queries {
            PathRecordBuffers::new_projected(
                &ctx,
                num_pixels,
                max_steps as u32,
                model.points.len() as u32,
                [width, height],
                0,
            )
        } else {
            PathRecordBuffers::new_external_powerfoam_projected(
                &ctx,
                num_pixels,
                max_steps as u32,
                jacobian_mode,
                model.points.len() as u32,
                [width, height],
                0,
                with_surface_queries,
            )
        }
    } else if weighted {
        assert_eq!(jacobian_mode, PathJacobianMode::Full);
        if with_surface_queries {
            PathRecordBuffers::new_external_powerfoam(
                &ctx,
                num_pixels,
                max_steps as u32,
                jacobian_mode,
                0,
                true,
            )
        } else {
            PathRecordBuffers::new(&ctx, num_pixels, max_steps as u32)
        }
    } else {
        PathRecordBuffers::new(&ctx, num_pixels, max_steps as u32)
    };
    bufs.write_pixel_indices(&pixels);

    encoder.start();
    {
        let mut tx = encoder.transfer("upload-pixels");
        let size = (pixels.len() * std::mem::size_of::<u32>()) as u64;
        tx.copy_buffer_to_buffer(
            bufs.pixel_indices_stage.at(0),
            bufs.pixel_indices.at(0),
            size,
        );
        // Weighted gather owns index/mask initialization. Poison those rows
        // so this oracle verifies that trailing entries do not depend on
        // allocator-zeroed memory or a caller-side fill.
        let pl = (num_pixels as u64) * (max_steps as u64);
        let index_fill = if weighted { 0xff } else { 0 };
        let mask_fill = if weighted { 0x3f } else { 0 };
        if bufs.has_geometry_jacobians() {
            tx.fill_buffer(bufs.previous_cells.at(0), pl * 4, index_fill);
        }
        tx.fill_buffer(bufs.cells.at(0), pl * 4, index_fill);
        tx.fill_buffer(bufs.next_cells.at(0), pl * 4, index_fill);
        tx.fill_buffer(bufs.dts.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.mask.at(0), pl * 4, mask_fill);
        if bufs.has_jacobians() {
            tx.fill_buffer(bufs.dt_reference_tangents.at(0), pl * 4, 0);
        }
        if bufs.has_geometry_jacobians() {
            tx.fill_buffer(bufs.dt_grad_previous.at(0), pl * 16, 0);
            tx.fill_buffer(bufs.dt_grad_current.at(0), pl * 16, 0);
            tx.fill_buffer(bufs.dt_grad_next.at(0), pl * 16, 0);
        }
        if bufs.has_surface_jacobians() {
            tx.fill_buffer(bufs.dt_grad_surface_normal.at(0), pl * 16, 0);
        }
        if bufs.has_surface_queries() {
            tx.fill_buffer(bufs.surface_queries.at(0), pl * 8, 0);
        }
        if bufs.has_surface_query_jacobians() {
            tx.fill_buffer(bufs.surface_query_grad_previous.at(0), pl * 16, 0);
            tx.fill_buffer(bufs.surface_query_grad_current.at(0), pl * 16, 0);
        }
    }
    // Fill the batch in two slices. Batched cases bind two distinct cameras
    // within each shared compute pass, matching mixed-view training.
    let dispatch_args = [
        RecordPathsArgs {
            camera: cameras[0],
            start_point: 0,
            pixel_offset: 0,
            max_steps: max_steps as u32,
            image_width: width,
            image_height: height,
            max_path_dt: 50.0,
            depth,
            num_pixels: num_pixels / 2,
        },
        RecordPathsArgs {
            camera: cameras[1],
            start_point: 0,
            pixel_offset: num_pixels / 2,
            max_steps: max_steps as u32,
            image_width: width,
            image_height: height,
            max_path_dt: 50.0,
            depth,
            num_pixels: num_pixels / 2,
        },
    ];
    if batched_exhaustive {
        recorder.dispatch_batch(&mut encoder, &cloud, &bufs, &dispatch_args);
    } else {
        for arg in dispatch_args {
            recorder.dispatch(&mut encoder, &cloud, &bufs, arg);
        }
    }
    // Read back via download staging buffers.
    let pl = (num_pixels as u64) * (max_steps as u64);
    let previous_cells_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "previous-cells-download",
        size: pl * 4,
        memory: gpu::Memory::Shared,
    });
    let cells_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "cells-download",
        size: pl * 4,
        memory: gpu::Memory::Shared,
    });
    let next_cells_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "next-cells-download",
        size: pl * 4,
        memory: gpu::Memory::Shared,
    });
    let dts_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "dts-download",
        size: pl * 4,
        memory: gpu::Memory::Shared,
    });
    let mask_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "mask-download",
        size: pl * 4,
        memory: gpu::Memory::Shared,
    });
    let dt_reference_tangents_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "dt-reference-tangents-download",
        size: pl * 4,
        memory: gpu::Memory::Shared,
    });
    let dt_grad_previous_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "dt-grad-previous-download",
        size: pl * 16,
        memory: gpu::Memory::Shared,
    });
    let dt_grad_current_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "dt-grad-current-download",
        size: pl * 16,
        memory: gpu::Memory::Shared,
    });
    let dt_grad_next_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "dt-grad-next-download",
        size: pl * 16,
        memory: gpu::Memory::Shared,
    });
    let dt_grad_surface_normal_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "dt-grad-surface-normal-download",
        size: pl * 16,
        memory: gpu::Memory::Shared,
    });
    let surface_queries_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "surface-queries-download",
        size: pl * 8,
        memory: gpu::Memory::Shared,
    });
    let surface_query_grad_previous_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "surface-query-grad-previous-download",
        size: pl * 16,
        memory: gpu::Memory::Shared,
    });
    let surface_query_grad_current_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "surface-query-grad-current-download",
        size: pl * 16,
        memory: gpu::Memory::Shared,
    });
    {
        let mut tx = encoder.transfer("download-outputs");
        if bufs.has_geometry_jacobians() {
            tx.copy_buffer_to_buffer(bufs.previous_cells.at(0), previous_cells_dl.at(0), pl * 4);
        }
        tx.copy_buffer_to_buffer(bufs.cells.at(0), cells_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.next_cells.at(0), next_cells_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.dts.at(0), dts_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.mask.at(0), mask_dl.at(0), pl * 4);
        if bufs.has_jacobians() {
            tx.copy_buffer_to_buffer(
                bufs.dt_reference_tangents.at(0),
                dt_reference_tangents_dl.at(0),
                pl * 4,
            );
        }
        if bufs.has_geometry_jacobians() {
            tx.copy_buffer_to_buffer(
                bufs.dt_grad_previous.at(0),
                dt_grad_previous_dl.at(0),
                pl * 16,
            );
            tx.copy_buffer_to_buffer(
                bufs.dt_grad_current.at(0),
                dt_grad_current_dl.at(0),
                pl * 16,
            );
            tx.copy_buffer_to_buffer(bufs.dt_grad_next.at(0), dt_grad_next_dl.at(0), pl * 16);
        }
        if bufs.has_surface_jacobians() {
            tx.copy_buffer_to_buffer(
                bufs.dt_grad_surface_normal.at(0),
                dt_grad_surface_normal_dl.at(0),
                pl * 16,
            );
        }
        if bufs.has_surface_queries() {
            tx.copy_buffer_to_buffer(bufs.surface_queries.at(0), surface_queries_dl.at(0), pl * 8);
        }
        if bufs.has_surface_query_jacobians() {
            tx.copy_buffer_to_buffer(
                bufs.surface_query_grad_previous.at(0),
                surface_query_grad_previous_dl.at(0),
                pl * 16,
            );
            tx.copy_buffer_to_buffer(
                bufs.surface_query_grad_current.at(0),
                surface_query_grad_current_dl.at(0),
                pl * 16,
            );
        }
    }
    let sync = ctx.submit(&mut encoder);
    let _ = ctx.wait_for(&sync, !0);
    let path_stats = bufs.path_stats(0..num_pixels as usize);
    let expected_max_steps = cpu
        .mask
        .chunks_exact(max_steps)
        .map(|row| row.iter().filter(|&&value| value > 0.0).count() as u32)
        .max()
        .unwrap_or(0);
    assert_eq!(path_stats.max_steps_used, expected_max_steps);
    assert_eq!(
        path_stats.truncated_rays > 0,
        expect_path_truncation,
        "unexpected path truncation stats: {path_stats:?}",
    );
    if weighted {
        let max_candidates = bufs.max_splat_candidate_count(0..num_pixels as usize);
        assert!(
            max_candidates <= bufs.splat_candidate_capacity(),
            "PowerFoam candidate scratch overflow: {max_candidates} > {}",
            bufs.splat_candidate_capacity(),
        );
        if bufs.has_projected_splat_tiles() {
            let max_tile_candidates = bufs.max_splat_tile_candidate_count([width, height]);
            assert!(
                max_tile_candidates > 0,
                "weighted fixture did not populate projected candidate tiles",
            );
            assert_eq!(
                max_tile_candidates > bufs.splat_tile_capacity(),
                expect_tile_overflow,
                "unexpected projected-tile overflow state: {max_tile_candidates} candidates, {} capacity",
                bufs.splat_tile_capacity(),
            );
        }
    }

    let gpu_previous_cells: Vec<u32> = if bufs.has_geometry_jacobians() {
        unsafe {
            std::slice::from_raw_parts(previous_cells_dl.data() as *const u32, pl as usize).to_vec()
        }
    } else {
        vec![0; pl as usize]
    };
    let gpu_cells: Vec<u32> =
        unsafe { std::slice::from_raw_parts(cells_dl.data() as *const u32, pl as usize).to_vec() };
    let gpu_next_cells: Vec<u32> = unsafe {
        std::slice::from_raw_parts(next_cells_dl.data() as *const u32, pl as usize).to_vec()
    };
    let gpu_dts: Vec<f32> =
        unsafe { std::slice::from_raw_parts(dts_dl.data() as *const f32, pl as usize).to_vec() };
    let gpu_mask: Vec<f32> =
        unsafe { std::slice::from_raw_parts(mask_dl.data() as *const f32, pl as usize).to_vec() };
    let gpu_dt_reference_tangents: Vec<f32> = if bufs.has_jacobians() {
        unsafe {
            std::slice::from_raw_parts(dt_reference_tangents_dl.data() as *const f32, pl as usize)
                .to_vec()
        }
    } else {
        vec![0.0; pl as usize]
    };
    let gpu_dt_grad_previous: Vec<f32> = if bufs.has_geometry_jacobians() {
        unsafe {
            std::slice::from_raw_parts(dt_grad_previous_dl.data() as *const f32, pl as usize * 4)
                .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 4]
    };
    let gpu_dt_grad_current: Vec<f32> = if bufs.has_geometry_jacobians() {
        unsafe {
            std::slice::from_raw_parts(dt_grad_current_dl.data() as *const f32, pl as usize * 4)
                .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 4]
    };
    let gpu_dt_grad_next: Vec<f32> = if bufs.has_geometry_jacobians() {
        unsafe {
            std::slice::from_raw_parts(dt_grad_next_dl.data() as *const f32, pl as usize * 4)
                .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 4]
    };
    let gpu_dt_grad_surface_normal: Vec<f32> = if bufs.has_surface_jacobians() {
        unsafe {
            std::slice::from_raw_parts(
                dt_grad_surface_normal_dl.data() as *const f32,
                pl as usize * 4,
            )
            .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 4]
    };
    let gpu_surface_queries: Vec<f32> = if bufs.has_surface_queries() {
        unsafe {
            std::slice::from_raw_parts(surface_queries_dl.data() as *const f32, pl as usize * 2)
                .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 2]
    };
    let gpu_surface_query_grad_previous: Vec<f32> = if bufs.has_surface_query_jacobians() {
        unsafe {
            std::slice::from_raw_parts(
                surface_query_grad_previous_dl.data() as *const f32,
                pl as usize * 4,
            )
            .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 4]
    };
    let gpu_surface_query_grad_current: Vec<f32> = if bufs.has_surface_query_jacobians() {
        unsafe {
            std::slice::from_raw_parts(
                surface_query_grad_current_dl.data() as *const f32,
                pl as usize * 4,
            )
            .to_vec()
        }
    } else {
        vec![0.0; pl as usize * 4]
    };

    let mut mismatches = 0usize;
    for i in 0..pl as usize {
        if cpu.mask[i] != gpu_mask[i] {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!(
                    "slot {i}: mask cpu={} gpu={} cells cpu={} gpu={} dt cpu={} gpu={}",
                    cpu.mask[i], gpu_mask[i], cpu.cells[i], gpu_cells[i], cpu.dts[i], gpu_dts[i]
                );
            }
            continue;
        }
        if cpu.mask[i] == 0.0 {
            continue;
        }
        if cpu.cells[i] != gpu_cells[i] || cpu.next_cells[i] != gpu_next_cells[i] {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!(
                    "slot {i}: cells cpu={}→{} gpu={}→{}",
                    cpu.cells[i], cpu.next_cells[i], gpu_cells[i], gpu_next_cells[i],
                );
            }
            continue;
        }
        let ddiff = (cpu.dts[i] - gpu_dts[i]).abs();
        // CPU and GPU normalize the ray independently, then evaluate a
        // square root at the support-sphere boundary. Near-tangent segments
        // amplify the backend rounding difference beyond a few ULPs.
        if ddiff > 5e-4 {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!(
                    "slot {i}: dt cpu={} gpu={} diff={}",
                    cpu.dts[i], gpu_dts[i], ddiff
                );
            }
        }
        if weighted {
            if jacobian_mode == PathJacobianMode::Full {
                if cpu.previous_cells[i] != gpu_previous_cells[i] {
                    mismatches += 1;
                    if mismatches <= 8 {
                        eprintln!(
                            "slot {i}: previous cell cpu={} gpu={}",
                            cpu.previous_cells[i], gpu_previous_cells[i],
                        );
                    }
                    continue;
                }
                for (name, cpu_gradient, gpu_gradient) in [
                    (
                        "previous",
                        cpu.dt_grad_previous[i],
                        &gpu_dt_grad_previous[i * 4..i * 4 + 4],
                    ),
                    (
                        "current",
                        cpu.dt_grad_current[i],
                        &gpu_dt_grad_current[i * 4..i * 4 + 4],
                    ),
                    (
                        "next",
                        cpu.dt_grad_next[i],
                        &gpu_dt_grad_next[i * 4..i * 4 + 4],
                    ),
                ] {
                    for component in 0..4 {
                        let expected = cpu_gradient[component];
                        let actual = gpu_gradient[component];
                        let absolute = (expected - actual).abs();
                        let scale = expected.abs().max(actual.abs()).max(1.0e-4);
                        if absolute > 5.0e-4 && absolute / scale > 5.0e-3 {
                            mismatches += 1;
                            if mismatches <= 8 {
                                eprintln!(
                                    "slot {i}: {name} gradient[{component}] cpu={expected} \
                                     gpu={actual} diff={absolute}",
                                );
                            }
                        }
                    }
                }
            }
            if model.surface_normals.is_some() && jacobian_mode != PathJacobianMode::None {
                for component in 0..4 {
                    let expected = cpu.dt_grad_surface_normal[i][component];
                    let actual = gpu_dt_grad_surface_normal[i * 4 + component];
                    let absolute = (expected - actual).abs();
                    let scale = expected.abs().max(actual.abs()).max(1.0e-4);
                    if absolute > 5.0e-4 && absolute / scale > 5.0e-3 {
                        mismatches += 1;
                        if mismatches <= 8 {
                            eprintln!(
                                "slot {i}: surface-normal gradient[{component}] cpu={expected} \
                                 gpu={actual} diff={absolute}",
                            );
                        }
                    }
                }
            }
            if with_surface_queries {
                for component in 0..2 {
                    let expected = cpu.surface_queries[i][component];
                    let actual = gpu_surface_queries[i * 2 + component];
                    if (expected - actual).abs() > 5.0e-4 {
                        mismatches += 1;
                        if mismatches <= 8 {
                            eprintln!(
                                "slot {i}: surface query[{component}] cpu={expected} \
                                 gpu={actual}",
                            );
                        }
                    }
                }
                if jacobian_mode == PathJacobianMode::Full {
                    for (name, cpu_gradient, gpu_gradient) in [
                        (
                            "surface-query previous",
                            cpu.surface_query_grad_previous[i],
                            &gpu_surface_query_grad_previous[i * 4..i * 4 + 4],
                        ),
                        (
                            "surface-query current",
                            cpu.surface_query_grad_current[i],
                            &gpu_surface_query_grad_current[i * 4..i * 4 + 4],
                        ),
                    ] {
                        for component in 0..4 {
                            let expected = cpu_gradient[component];
                            let actual = gpu_gradient[component];
                            let absolute = (expected - actual).abs();
                            let scale = expected.abs().max(actual.abs()).max(1.0e-4);
                            if absolute > 5.0e-4 && absolute / scale > 5.0e-3 {
                                mismatches += 1;
                                if mismatches <= 8 {
                                    eprintln!(
                                        "slot {i}: {name} gradient[{component}] cpu={expected} \
                                         gpu={actual} diff={absolute}",
                                    );
                                }
                            }
                        }
                    }

                    let ray_origin = rays[i / max_steps].origin;
                    let reconstructed_query = [
                        (
                            gpu_previous_cells[i],
                            &gpu_surface_query_grad_previous[i * 4..i * 4 + 4],
                        ),
                        (
                            gpu_cells[i],
                            &gpu_surface_query_grad_current[i * 4..i * 4 + 4],
                        ),
                    ]
                    .into_iter()
                    .map(|(cell, gradient)| {
                        gradient
                            .iter()
                            .zip(point_geometry_relative(&model, cell, ray_origin))
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                    })
                    .sum::<f32>();
                    if (reconstructed_query - gpu_surface_queries[i * 2]).abs() > 5.0e-4 {
                        mismatches += 1;
                        if mismatches <= 8 {
                            eprintln!(
                                "slot {i}: surface query={} reconstructed={reconstructed_query}",
                                gpu_surface_queries[i * 2],
                            );
                        }
                    }
                }
            }
            let expected_reference_tangent = match jacobian_mode {
                PathJacobianMode::Full => cpu.dt_reference_tangents[i],
                PathJacobianMode::Surface => dot4(
                    &cpu.dt_grad_surface_normal[i],
                    &surface_normal_geometry(&model, cpu.cells[i], cpu.surface_offsets[i]),
                ),
                PathJacobianMode::None => 0.0,
            };
            let tangent_absolute =
                (expected_reference_tangent - gpu_dt_reference_tangents[i]).abs();
            let tangent_scale = expected_reference_tangent
                .abs()
                .max(gpu_dt_reference_tangents[i].abs())
                .max(1.0e-4);
            if tangent_absolute > 2.0e-3 && tangent_absolute / tangent_scale > 5.0e-3 {
                mismatches += 1;
                if mismatches <= 8 {
                    eprintln!(
                        "slot {i}: reference tangent cpu={} gpu={} diff={tangent_absolute}",
                        expected_reference_tangent, gpu_dt_reference_tangents[i],
                    );
                }
            }
            let mut reconstructed_dt = gpu_dts[i] - gpu_dt_reference_tangents[i];
            if jacobian_mode == PathJacobianMode::Full {
                let ray_origin = rays[i / max_steps].origin;
                for (cell, gradient) in [
                    (
                        gpu_previous_cells[i],
                        &gpu_dt_grad_previous[i * 4..i * 4 + 4],
                    ),
                    (gpu_cells[i], &gpu_dt_grad_current[i * 4..i * 4 + 4]),
                    (gpu_next_cells[i], &gpu_dt_grad_next[i * 4..i * 4 + 4]),
                ] {
                    reconstructed_dt += gradient
                        .iter()
                        .zip(point_geometry_relative(&model, cell, ray_origin))
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                }
            }
            if model.surface_normals.is_some() && jacobian_mode != PathJacobianMode::None {
                reconstructed_dt += gpu_dt_grad_surface_normal[i * 4..i * 4 + 4]
                    .iter()
                    .zip(surface_normal_geometry(
                        &model,
                        gpu_cells[i],
                        cpu.surface_offsets[i],
                    ))
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
            let reconstruction_error = (gpu_dts[i] - reconstructed_dt).abs();
            if reconstruction_error > 5.0e-4 {
                mismatches += 1;
                if mismatches <= 8 {
                    eprintln!(
                        "slot {i}: raw dt={} reconstructed={} diff={reconstruction_error}",
                        gpu_dts[i], reconstructed_dt,
                    );
                }
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "GPU path-record differs from CPU on {} of {} slots",
        mismatches, pl
    );

    ctx.destroy_buffer(previous_cells_dl);
    ctx.destroy_buffer(cells_dl);
    ctx.destroy_buffer(next_cells_dl);
    ctx.destroy_buffer(dts_dl);
    ctx.destroy_buffer(mask_dl);
    ctx.destroy_buffer(dt_reference_tangents_dl);
    ctx.destroy_buffer(dt_grad_previous_dl);
    ctx.destroy_buffer(dt_grad_current_dl);
    ctx.destroy_buffer(dt_grad_next_dl);
    ctx.destroy_buffer(dt_grad_surface_normal_dl);
    ctx.destroy_buffer(surface_queries_dl);
    ctx.destroy_buffer(surface_query_grad_previous_dl);
    ctx.destroy_buffer(surface_query_grad_current_dl);
    bufs.destroy(&ctx);
    recorder.destroy(&ctx);
    cloud.deinit(&ctx);
    ctx.destroy_command_encoder(&mut encoder);
}

fn record_unweighted_gpu_bytes(
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    cloud: &RadFoamGpuCloud,
    recorder: &PathRecorder,
    pixels: &[u32],
    args: &[RecordPathsArgs],
    batched: bool,
) -> (Vec<u8>, PathRecordStats) {
    let num_pixels = pixels.len() as u32;
    let max_steps = args[0].max_steps;
    let mut buffers = PathRecordBuffers::new_recorded_only(context, num_pixels, max_steps);
    buffers.write_pixel_indices(pixels);

    let row_bytes = u64::from(num_pixels) * u64::from(max_steps) * 4;
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "radfoam-batch-parity-download",
        size: row_bytes * 4,
        memory: gpu::Memory::Shared,
    });
    encoder.start();
    {
        let mut tx = encoder.transfer("radfoam-batch-parity-prepare");
        tx.copy_buffer_to_buffer(
            buffers.pixel_indices_stage.at(0),
            buffers.pixel_indices.at(0),
            num_pixels as u64 * 4,
        );
        tx.fill_buffer(buffers.cells.at(0), row_bytes, 0);
        tx.fill_buffer(buffers.next_cells.at(0), row_bytes, 0);
        tx.fill_buffer(buffers.dts.at(0), row_bytes, 0);
        tx.fill_buffer(buffers.mask.at(0), row_bytes, 0);
    }
    if batched {
        recorder.dispatch_batch(encoder, cloud, &buffers, args);
    } else {
        for &arg in args {
            recorder.dispatch(encoder, cloud, &buffers, arg);
        }
    }
    {
        let mut tx = encoder.transfer("radfoam-batch-parity-download");
        for (index, source) in [
            &buffers.cells,
            &buffers.next_cells,
            &buffers.dts,
            &buffers.mask,
        ]
        .into_iter()
        .enumerate()
        {
            tx.copy_buffer_to_buffer(
                source.at(0),
                readback.at(index as u64 * row_bytes),
                row_bytes,
            );
        }
    }
    let sync = context.submit(encoder);
    let _ = context.wait_for(&sync, !0);
    let stats = buffers.path_stats(0..num_pixels as usize);
    let bytes = unsafe {
        std::slice::from_raw_parts(readback.data() as *const u8, (row_bytes * 4) as usize).to_vec()
    };

    context.destroy_buffer(readback);
    buffers.destroy(context);
    (bytes, stats)
}

#[test]
fn gpu_path_record_matches_cpu_on_grid() {
    assert_gpu_path_record_matches_cpu(build_grid_model(12), glam::Vec3::ZERO, false, false);
}

#[test]
fn gpu_batched_radfoam_paths_match_multi_camera_cpu() {
    assert_gpu_batched_path_record_matches_cpu(build_grid_model(12), glam::Vec3::ZERO, false);
}

#[test]
fn gpu_batched_radfoam_paths_are_bit_exact_with_separate_passes() {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .expect("GPU path-record test lock poisoned");
    let Some(context) = try_init_gpu() else {
        eprintln!("skipping: no GPU");
        return;
    };

    let model = build_grid_model(12);
    let depth = 100.0;
    let width = 64;
    let height = 64;
    let num_pixels = 8;
    let max_steps = 16;
    let pixels = pixel_indices_for_rays(width, height, num_pixels);
    let mut cameras = [make_camera_looking_along_x(depth); 2];
    cameras[1].principal[0] -= 0.025;
    cameras[1].principal[1] += 0.015;
    let args = [
        RecordPathsArgs {
            camera: cameras[0],
            start_point: 0,
            pixel_offset: 0,
            max_steps,
            image_width: width,
            image_height: height,
            max_path_dt: 50.0,
            depth,
            num_pixels: num_pixels / 2,
        },
        RecordPathsArgs {
            camera: cameras[1],
            start_point: 0,
            pixel_offset: num_pixels / 2,
            max_steps,
            image_width: width,
            image_height: height,
            max_path_dt: 50.0,
            depth,
            num_pixels: num_pixels / 2,
        },
    ];

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gpu-path-record-batch-parity-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut cloud = RadFoamGpuCloud::new_path_recording(&model, &context, &mut encoder);
    let mut recorder = PathRecorder::new(&context);
    let separate = record_unweighted_gpu_bytes(
        &context,
        &mut encoder,
        &cloud,
        &recorder,
        &pixels,
        &args,
        false,
    );
    let batched = record_unweighted_gpu_bytes(
        &context,
        &mut encoder,
        &cloud,
        &recorder,
        &pixels,
        &args,
        true,
    );
    assert_eq!(batched, separate);

    recorder.destroy(&context);
    cloud.deinit(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn gpu_path_record_full_walk_row_is_not_truncated() {
    assert_gpu_path_record_matches_cpu(build_grid_model(16), glam::Vec3::ZERO, false, false);
}

#[test]
fn gpu_path_record_reports_walk_truncation() {
    assert_gpu_path_record_matches_cpu(build_grid_model(24), glam::Vec3::ZERO, false, true);
}

#[test]
fn gpu_path_record_matches_bounded_powerfoam() {
    let mut model = build_grid_model(12);
    model.radii = Some(
        (0..model.points.len())
            .map(|i| 0.2 + 0.03 * (i % 3) as f32)
            .collect(),
    );
    assert_gpu_path_record_matches_cpu(model, glam::Vec3::ZERO, false, false);
}

#[test]
fn gpu_batched_powerfoam_paths_match_multi_camera_cpu() {
    let mut model = build_grid_model(12);
    model.radii = Some(
        (0..model.points.len())
            .map(|i| 0.2 + 0.03 * (i % 3) as f32)
            .collect(),
    );
    assert_gpu_batched_path_record_matches_cpu(model, glam::Vec3::ZERO, false);
}

fn build_bvh_frontier_model() -> vol::PointCloudModel {
    const POINT_COUNT: usize = 32 * 1024;
    let mut model = build_disconnected_ray_model(129);
    model
        .points
        .extend((model.points.len()..POINT_COUNT).map(|index| {
            glam::Vec4::new(
                (index % 127) as f32,
                1000.0 + (index % 113) as f32,
                (index % 109) as f32,
                1.0,
            )
        }));
    model.sh_coefficients.resize(POINT_COUNT * 3, 0.0);
    model.radii.as_mut().unwrap().resize(POINT_COUNT, 0.2);
    model
        .adjacency
        .as_mut()
        .unwrap()
        .offsets
        .resize(POINT_COUNT + 1, 0);
    model
}

#[test]
fn gpu_parallel_bvh_frontier_matches_translated_cpu() {
    assert_gpu_batched_path_record_matches_cpu(
        build_bvh_frontier_model(),
        glam::Vec3::new(8192.0, -4096.0, 2048.0),
        true,
    );
}

#[test]
fn gpu_full_cloud_omits_the_path_recording_bvh() {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .expect("GPU path-record test lock poisoned");
    let Some(ctx) = try_init_gpu() else {
        eprintln!("skipping: no GPU");
        return;
    };
    let model = build_bvh_frontier_model();
    let mut encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gpu-full-cloud-bvh-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut cloud = RadFoamGpuCloud::new(&model, &ctx, &mut encoder);
    assert!(!PathRecorder::uses_support_bvh(&cloud));
    cloud.deinit(&ctx);
}

#[test]
fn gpu_oriented_powerfoam_paths_and_normal_jacobians_match_cpu() {
    let mut model = build_disconnected_ray_model(12);
    let camera = make_camera_looking_along_x(100.0);
    let target_ray = rays_for_pixels(&camera, &[32 * 64 + 32], 64, 64)[0];
    model.surface_normals = Some(vec![-target_ray.direction; model.points.len()]);
    model.surface_offsets = Some(
        (0..model.points.len())
            .map(|index| 0.002 * (index % 5) as f32 - 0.004)
            .collect(),
    );
    assert_gpu_path_record_matches_cpu(model, glam::Vec3::ZERO, false, false);
}

#[test]
fn gpu_surface_detail_paths_and_queries_match_cpu() {
    let mut model = build_disconnected_ray_model(12);
    let camera = make_camera_looking_along_x(100.0);
    let target_ray = rays_for_pixels(&camera, &[32 * 64 + 32], 64, 64)[0];
    model.surface_normals = Some(vec![-target_ray.direction; model.points.len()]);
    model.surface_offsets = Some(
        (0..model.points.len())
            .map(|index| 0.002 * (index % 5) as f32 - 0.004)
            .collect(),
    );
    model.surface_detail = Some(vol::SurfaceDetail {
        offsets: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
            .map(|index| {
                let angle = (index % vol::SURFACE_DETAIL_SITES) as f32 * std::f32::consts::TAU
                    / vol::SURFACE_DETAIL_SITES as f32;
                glam::Vec3::new(0.25 * angle.cos(), 0.25 * angle.sin(), 0.07)
            })
            .collect(),
        heights: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
            .map(|index| 0.025 * ((index % 7) as f32 - 3.0))
            .collect(),
        colors: vec![glam::Vec3::ZERO; model.points.len() * vol::SURFACE_DETAIL_SITES],
        density_logits: None,
        directional: None,
    });
    assert_gpu_path_record_matches_cpu(model, glam::Vec3::ZERO, false, false);
}

#[test]
fn gpu_dense_powerfoam_parallel_paths_match_cpu() {
    let mut model = build_disconnected_ray_model(70);
    let point_count = model.points.len();
    let mut neighbors = Vec::with_capacity(point_count * (point_count - 1));
    let mut offsets = Vec::with_capacity(point_count + 1);
    offsets.push(0);
    for point in 0..point_count {
        neighbors.extend(
            (0..point_count)
                .filter(|&other| other != point)
                .map(|i| i as u32),
        );
        offsets.push(neighbors.len() as u32);
    }
    model.adjacency = Some(vol::Adjacency { neighbors, offsets });

    let camera = make_camera_looking_along_x(100.0);
    let target_ray = rays_for_pixels(&camera, &[32 * 64 + 32], 64, 64)[0];
    model.surface_normals = Some(vec![-target_ray.direction; point_count]);
    model.surface_offsets = Some(vec![0.0; point_count]);
    model.surface_detail = Some(vol::SurfaceDetail {
        offsets: (0..point_count * vol::SURFACE_DETAIL_SITES)
            .map(|index| {
                let angle = (index % vol::SURFACE_DETAIL_SITES) as f32 * std::f32::consts::TAU
                    / vol::SURFACE_DETAIL_SITES as f32;
                glam::Vec3::new(0.2 * angle.cos(), 0.2 * angle.sin(), 0.03)
            })
            .collect(),
        heights: vec![0.0; point_count * vol::SURFACE_DETAIL_SITES],
        colors: vec![glam::Vec3::ZERO; point_count * vol::SURFACE_DETAIL_SITES],
        density_logits: None,
        directional: None,
    });
    assert_gpu_batched_path_record_matches_cpu(model, glam::Vec3::ZERO, true);
}

#[test]
fn gpu_surface_only_tangent_matches_oriented_cpu_reference() {
    let mut model = build_disconnected_ray_model(12);
    let camera = make_camera_looking_along_x(100.0);
    let target_ray = rays_for_pixels(&camera, &[32 * 64 + 32], 64, 64)[0];
    model.surface_normals = Some(vec![-target_ray.direction; model.points.len()]);
    model.surface_offsets = Some(
        (0..model.points.len())
            .map(|index| 0.002 * (index % 5) as f32 - 0.004)
            .collect(),
    );
    assert_gpu_path_record_matches_cpu_with_mode(
        model,
        glam::Vec3::ZERO,
        false,
        false,
        false,
        PathJacobianMode::Surface,
    );
}

#[test]
fn gpu_weighted_linearization_is_translation_invariant() {
    let mut model = build_grid_model(12);
    model.radii = Some(
        (0..model.points.len())
            .map(|i| 0.2 + 0.03 * (i % 3) as f32)
            .collect(),
    );
    assert_gpu_path_record_matches_cpu(
        model,
        glam::Vec3::new(8192.0, -4096.0, 2048.0),
        false,
        false,
    );
}

#[test]
fn gpu_powerfoam_splats_cross_disconnected_cech_components() {
    let points = (0..12)
        .map(|index| glam::Vec4::new(index as f32, 0.0, 0.0, 1.0))
        .collect::<Vec<_>>();
    let model = vol::PointCloudModel {
        sh_coefficients: vec![0.0; points.len() * 3],
        sh_degree: 0,
        transforms: None,
        adjacency: Some(vol::Adjacency {
            neighbors: Vec::new(),
            offsets: vec![0; points.len() + 1],
        }),
        radii: Some(vec![0.3; points.len()]),
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
        points,
    };

    let camera = make_camera_looking_along_x(100.0);
    let pixels = pixel_indices_for_rays(64, 64, 8);
    let rays = rays_for_pixels(&camera, &pixels, 64, 64);
    let cpu = cpu_record(&model, &rays, 0, 16, 100.0);
    assert!(
        cpu.mask
            .chunks_exact(16)
            .any(|row| row.iter().filter(|&&mask| mask > 0.0).count() > 1),
        "fixture must require crossing disconnected supports",
    );

    assert_gpu_path_record_matches_cpu(model, glam::Vec3::ZERO, false, false);
}

fn build_disconnected_ray_model(point_count: usize) -> vol::PointCloudModel {
    let camera = make_camera_looking_along_x(100.0);
    let target_ray = rays_for_pixels(&camera, &[32 * 64 + 32], 64, 64)[0];
    let points = (0..point_count)
        .map(|index| {
            let center = target_ray.origin + (5.0 + 0.5 * index as f32) * target_ray.direction;
            center.extend(1.0)
        })
        .collect::<Vec<_>>();
    vol::PointCloudModel {
        points,
        sh_coefficients: vec![0.0; point_count * 3],
        sh_degree: 0,
        transforms: None,
        adjacency: Some(vol::Adjacency {
            neighbors: Vec::new(),
            offsets: vec![0; point_count + 1],
        }),
        radii: Some(vec![0.2; point_count]),
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    }
}

#[test]
fn gpu_powerfoam_full_row_without_remainder_is_not_truncated() {
    assert_gpu_path_record_matches_cpu(
        build_disconnected_ray_model(16),
        glam::Vec3::ZERO,
        false,
        false,
    );
}

#[test]
fn gpu_powerfoam_reports_path_truncation() {
    assert_gpu_path_record_matches_cpu(
        build_disconnected_ray_model(24),
        glam::Vec3::ZERO,
        false,
        true,
    );
}

#[test]
fn gpu_projected_tile_overflow_falls_back_to_exhaustive_scan() {
    const POINT_COUNT: usize = 16_385;
    let camera = make_camera_looking_along_x(100.0);
    let target_rays = rays_for_pixels(&camera, &[32 * 64 + 32, 38 * 64 + 38], 64, 64);
    let visible = target_rays[0].origin + 10.0 * target_rays[0].direction;
    let crowded = target_rays[1].origin + 10.0 * target_rays[1].direction;
    let mut points = vec![visible.extend(1.0)];
    points.resize(POINT_COUNT, crowded.extend(1.0));
    let mut radii = vec![1.0e-4; POINT_COUNT];
    radii[0] = 0.1;
    let model = vol::PointCloudModel {
        points,
        sh_coefficients: vec![0.0; POINT_COUNT * 3],
        sh_degree: 0,
        transforms: None,
        adjacency: Some(vol::Adjacency {
            neighbors: Vec::new(),
            offsets: vec![0; POINT_COUNT + 1],
        }),
        radii: Some(radii),
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    };

    assert_gpu_path_record_matches_cpu(model, glam::Vec3::ZERO, true, false);
}
