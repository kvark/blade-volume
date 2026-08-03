//! Verify the GPU path-record shader matches CPU `record_path`.
//!
//! Builds a tiny Voronoi cloud (regular grid in a corner) with known
//! adjacency, traces a few rays both on CPU and GPU, and compares the
//! `(cell, dt, mask)` outputs entry-by-entry. The GPU path-record is
//! the replacement for the single-threaded CPU path tracer inside the
//! diff-render training loop — this test gates correctness.

use blade_graphics as gpu;
use blade_volume as vol;
use vol::gpu::{PathRecordBuffers, PathRecorder, RadFoamGpuCloud, RecordPathsArgs};

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

fn surface_normal_geometry(model: &vol::PointCloudModel, index: u32) -> [f32; 4] {
    model.surface_normals.as_ref().map_or([0.0; 4], |normals| {
        let offset = model
            .surface_offsets
            .as_ref()
            .map_or(0.0, |offsets| offsets[index as usize]);
        normals[index as usize]
            .normalize()
            .extend(offset)
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
                        &surface_normal_geometry(model, e.cell),
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
    );
}

fn assert_gpu_path_record_matches_cpu_with_mode(
    mut model: vol::PointCloudModel,
    world_translation: glam::Vec3,
    expect_tile_overflow: bool,
    expect_path_truncation: bool,
    batched_exhaustive: bool,
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
    let mut recorder = PathRecorder::new(&ctx);
    let mut bufs = if weighted && !batched_exhaustive {
        PathRecordBuffers::new_projected(
            &ctx,
            num_pixels,
            max_steps as u32,
            model.points.len() as u32,
            [width, height],
            0,
        )
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
        // Zero the outputs (shader only writes the steps it actually
        // takes; leftover slots must be zero).
        let pl = (num_pixels as u64) * (max_steps as u64);
        tx.fill_buffer(bufs.previous_cells.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.cells.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.next_cells.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.dts.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.mask.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.dt_reference_tangents.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.dt_grad_previous.at(0), pl * 16, 0);
        tx.fill_buffer(bufs.dt_grad_current.at(0), pl * 16, 0);
        tx.fill_buffer(bufs.dt_grad_next.at(0), pl * 16, 0);
        if model.surface_normals.is_some() {
            tx.fill_buffer(bufs.dt_grad_surface_normal.at(0), pl * 16, 0);
        }
    }
    // Fill the batch in two slices. The exhaustive PowerFoam case binds two
    // distinct cameras within each shared compute pass, matching mixed-view
    // training. Other cases retain the ordinary per-slice dispatch path.
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
    {
        let mut tx = encoder.transfer("download-outputs");
        tx.copy_buffer_to_buffer(bufs.previous_cells.at(0), previous_cells_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.cells.at(0), cells_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.next_cells.at(0), next_cells_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.dts.at(0), dts_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.mask.at(0), mask_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(
            bufs.dt_reference_tangents.at(0),
            dt_reference_tangents_dl.at(0),
            pl * 4,
        );
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
        tx.copy_buffer_to_buffer(
            bufs.dt_grad_surface_normal.at(0),
            dt_grad_surface_normal_dl.at(0),
            pl * 16,
        );
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

    let gpu_previous_cells: Vec<u32> = unsafe {
        std::slice::from_raw_parts(previous_cells_dl.data() as *const u32, pl as usize).to_vec()
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
    let gpu_dt_reference_tangents: Vec<f32> = unsafe {
        std::slice::from_raw_parts(dt_reference_tangents_dl.data() as *const f32, pl as usize)
            .to_vec()
    };
    let gpu_dt_grad_previous: Vec<f32> = unsafe {
        std::slice::from_raw_parts(dt_grad_previous_dl.data() as *const f32, pl as usize * 4)
            .to_vec()
    };
    let gpu_dt_grad_current: Vec<f32> = unsafe {
        std::slice::from_raw_parts(dt_grad_current_dl.data() as *const f32, pl as usize * 4)
            .to_vec()
    };
    let gpu_dt_grad_next: Vec<f32> = unsafe {
        std::slice::from_raw_parts(dt_grad_next_dl.data() as *const f32, pl as usize * 4).to_vec()
    };
    let gpu_dt_grad_surface_normal: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            dt_grad_surface_normal_dl.data() as *const f32,
            pl as usize * 4,
        )
        .to_vec()
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
            if model.surface_normals.is_some() {
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
            let tangent_absolute =
                (cpu.dt_reference_tangents[i] - gpu_dt_reference_tangents[i]).abs();
            let tangent_scale = cpu.dt_reference_tangents[i]
                .abs()
                .max(gpu_dt_reference_tangents[i].abs())
                .max(1.0e-4);
            if tangent_absolute > 2.0e-3 && tangent_absolute / tangent_scale > 5.0e-3 {
                mismatches += 1;
                if mismatches <= 8 {
                    eprintln!(
                        "slot {i}: reference tangent cpu={} gpu={} diff={tangent_absolute}",
                        cpu.dt_reference_tangents[i], gpu_dt_reference_tangents[i],
                    );
                }
            }
            let mut reconstructed_dt = gpu_dts[i] - gpu_dt_reference_tangents[i];
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
            if model.surface_normals.is_some() {
                reconstructed_dt += gpu_dt_grad_surface_normal[i * 4..i * 4 + 4]
                    .iter()
                    .zip(surface_normal_geometry(&model, gpu_cells[i]))
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
    bufs.destroy(&ctx);
    recorder.destroy(&ctx);
    cloud.deinit(&ctx);
    ctx.destroy_command_encoder(&mut encoder);
}

#[test]
fn gpu_path_record_matches_cpu_on_grid() {
    assert_gpu_path_record_matches_cpu(build_grid_model(12), glam::Vec3::ZERO, false, false);
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
        surface_color_coefficients: None,
        spherical_voronoi: None,
    };

    assert_gpu_path_record_matches_cpu(model, glam::Vec3::ZERO, true, false);
}
