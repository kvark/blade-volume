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

fn try_init_gpu() -> Option<gpu::Context> {
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
        if i + 1 < n {
            neighbours.push((i + 1) as u32);
        }
        if i > 0 {
            neighbours.push((i - 1) as u32);
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
    }
}

fn cpu_record(
    model: &vol::PointCloudModel,
    rays: &[vol::trace::Ray],
    start_cell: u32,
    max_steps: usize,
    depth: f32,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let settings = vol::trace::TraceSettings {
        start_point: start_cell,
        max_steps: max_steps as u32,
        weight_threshold: 0.0,
        depth,
        eval_mode: vol::trace::EvalMode::Sh,
    };
    let p = rays.len();
    let mut cells = vec![0u32; p * max_steps];
    let mut dts = vec![0.0f32; p * max_steps];
    let mut mask = vec![0.0f32; p * max_steps];
    for (k, ray) in rays.iter().enumerate() {
        let path = vol::trace::record_path(model, *ray, settings);
        for (idx, e) in path.entries.iter().take(max_steps).enumerate() {
            let slot = k * max_steps + idx;
            cells[slot] = e.cell;
            if e.dt.is_finite() && e.dt > 0.0 {
                dts[slot] = e.dt.min(50.0);
                mask[slot] = 1.0;
            }
        }
    }
    (cells, dts, mask)
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

#[test]
fn gpu_path_record_matches_cpu_on_grid() {
    let Some(ctx) = try_init_gpu() else {
        eprintln!("skipping: no GPU");
        return;
    };

    let n = 12usize;
    let max_steps = 16usize;
    let num_pixels = 8u32;
    let width = 64u32;
    let height = 64u32;
    let depth = 100.0f32;

    let model = build_grid_model(n);
    let camera = make_camera_looking_along_x(depth);
    let pixels = pixel_indices_for_rays(width, height, num_pixels);
    let rays = rays_for_pixels(&camera, &pixels, width, height);

    let (cpu_cells, cpu_dts, cpu_mask) = cpu_record(&model, &rays, 0, max_steps, depth);

    let mut encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gpu-path-record-test",
        buffer_count: 1,
    });
    let cloud = RadFoamGpuCloud::new(&model, &ctx, &mut encoder);
    let recorder = PathRecorder::new(&ctx);
    let bufs = PathRecordBuffers::new(&ctx, num_pixels, max_steps as u32);
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
        tx.fill_buffer(bufs.cells.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.next_cells.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.dts.at(0), pl * 4, 0);
        tx.fill_buffer(bufs.mask.at(0), pl * 4, 0);
    }
    recorder.dispatch(
        &mut encoder,
        &cloud,
        bufs.pixel_indices.into(),
        bufs.cells.into(),
        bufs.next_cells.into(),
        bufs.dts.into(),
        bufs.mask.into(),
        RecordPathsArgs {
            camera,
            start_point: 0,
            max_steps: max_steps as u32,
            image_width: width,
            image_height: height,
            max_path_dt: 50.0,
            depth,
            num_pixels,
        },
    );
    // Read back via download staging buffers.
    let pl = (num_pixels as u64) * (max_steps as u64);
    let cells_dl = ctx.create_buffer(gpu::BufferDesc {
        name: "cells-download",
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
    {
        let mut tx = encoder.transfer("download-outputs");
        tx.copy_buffer_to_buffer(bufs.cells.at(0), cells_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.dts.at(0), dts_dl.at(0), pl * 4);
        tx.copy_buffer_to_buffer(bufs.mask.at(0), mask_dl.at(0), pl * 4);
    }
    let sync = ctx.submit(&mut encoder);
    let _ = ctx.wait_for(&sync, !0);

    let gpu_cells: Vec<u32> =
        unsafe { std::slice::from_raw_parts(cells_dl.data() as *const u32, pl as usize).to_vec() };
    let gpu_dts: Vec<f32> =
        unsafe { std::slice::from_raw_parts(dts_dl.data() as *const f32, pl as usize).to_vec() };
    let gpu_mask: Vec<f32> =
        unsafe { std::slice::from_raw_parts(mask_dl.data() as *const f32, pl as usize).to_vec() };

    let mut mismatches = 0usize;
    for i in 0..pl as usize {
        if cpu_mask[i] != gpu_mask[i] {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!(
                    "slot {i}: mask cpu={} gpu={} cells cpu={} gpu={} dt cpu={} gpu={}",
                    cpu_mask[i], gpu_mask[i], cpu_cells[i], gpu_cells[i], cpu_dts[i], gpu_dts[i]
                );
            }
            continue;
        }
        if cpu_mask[i] == 0.0 {
            continue;
        }
        if cpu_cells[i] != gpu_cells[i] {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!("slot {i}: cell cpu={} gpu={}", cpu_cells[i], gpu_cells[i]);
            }
            continue;
        }
        let ddiff = (cpu_dts[i] - gpu_dts[i]).abs();
        // f32 + radical-plane math: allow a few ULPs.
        if ddiff > 1e-4 {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!(
                    "slot {i}: dt cpu={} gpu={} diff={}",
                    cpu_dts[i], gpu_dts[i], ddiff
                );
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "GPU path-record differs from CPU on {} of {} slots",
        mismatches, pl
    );
}
