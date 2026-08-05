//! GPU vs CPU RadFoam forward tracing test.
//!
//! Goal:
//! - Execute the production RadFoam compute shader on a synthetic, deterministic fixture
//! - Read back pixels from an RGBA16F output texture
//! - Compare against the CPU reference tracer for matching rays
//!
//! Notes / assumptions:
//! - This test uses the production shader from blade-volume/shaders/radfoam.wgsl
//! - For this correctness check, we enable SH **degree 3** evaluation in both CPU and GPU
//!   to validate packed attribute reading + SH evaluation alongside traversal/integration.
//! - The synthetic fixture is a branching topology that creates higher-degree nodes to
//!   exercise multi-neighbor face selection. This avoids requiring a full 3D Delaunay
//!   triangulation (which would introduce native dependencies).
//! - This test compares a small set of pixels (not just one) to increase coverage while
//!   keeping runtime small and deterministic.
//!
//! Running on CI:
//! - This should be compatible with lavapipe as it only requires compute + storage textures
//!   + copy-to-buffer.
//! - If no supported GPU device is found (e.g. on some CI runners), this test should
//!   skip gracefully.
//!
//! Validation:
//! - When Vulkan validation is enabled, we must explicitly destroy GPU resources created by
//!   this test (command encoder, pipelines, textures, buffers) before dropping the Context.

#![allow(clippy::float_cmp)]

use blade_graphics as gpu;
use blade_volume as vol;
use half::f16;

// Import the CPU reference tracer as a sibling module (same directory).
mod radfoam_cpu_ref;

// Synthetic fixture generators.
mod radfoam_synth_branch;
mod radfoam_synth_chain;

static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

use radfoam_cpu_ref as cpu;

/// A minimal camera uniform matching the production shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraParams {
    position: [f32; 3],
    depth: f32,
    orientation: [f32; 4],
    fov: [f32; 2],
    principal: [f32; 2],
}

/// Trace parameters matching the production shader.
///
/// WGSL layout (std140/uniform alignment):
/// - vec3<u32> requires 16-byte alignment
/// - struct size rounds up to multiple of 16
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct TraceParams {
    sh_degree: u32,        // offset 0
    weight_threshold: f32, // offset 4
    max_steps: u32,        // offset 8
    start_point: u32,      // offset 12
    debug_mode: u32,       // offset 16
    _align_pad: [u32; 3],  // offset 20 (padding to align vec3 to 32)
    pad: [u32; 3],         // offset 32 (the vec3<u32> pad field)
    _size_pad: u32,        // offset 44 (padding to make struct 48 bytes)
}

#[derive(blade_macros::ShaderData)]
struct TraceData {
    g_camera: CameraParams,
    g_params: TraceParams,
    g_points: gpu::BufferPiece,
    g_surface_normals: gpu::BufferPiece,
    g_surface_details: gpu::BufferPiece,
    g_attributes: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

/// Create a headless GPU context for tests.
///
/// Note: we keep validation enabled in debug to catch issues early.
fn make_test_context() -> Option<gpu::Context> {
    if vol::gpu::access_disabled() {
        return None;
    }
    unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: false,
            capture: false,
            overlay: false,
            ray_tracing: true,
            xr: None,
            device_id: None,
        }) {
            Ok(ctx) => Some(ctx),
            Err(gpu::NotSupportedError::NoSupportedDeviceFound) => None,
            Err(other) => panic!("failed to init GPU context: {:?}", other),
        }
    }
}

/// Helper to destroy all resources created by this test in the correct order.
fn cleanup_test_resources(
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    pipeline: &mut gpu::ComputePipeline,
    radfoam_gpu: &mut vol::RadFoamGpuCloud,
    out_tex: gpu::Texture,
    out_view: gpu::TextureView,
    readback: gpu::Buffer,
) {
    // Ensure the GPU is done with the submitted work before destroying resources.
    // (Caller already waits, but keep this robust.)
    //
    // Destroy resources in reverse-ish creation order.
    context.destroy_buffer(readback);
    context.destroy_texture_view(out_view);
    context.destroy_texture(out_tex);

    radfoam_gpu.deinit(context);

    context.destroy_compute_pipeline(pipeline);

    // CommandEncoder owns command buffers / pools; destroy it explicitly to satisfy validation.
    context.destroy_command_encoder(encoder);
}

/// Create an RGBA16F storage texture to receive test output.
fn create_output_rgba16(
    context: &gpu::Context,
    w: u32,
    h: u32,
) -> (gpu::Texture, gpu::TextureView) {
    let tex = context.create_texture(gpu::TextureDesc {
        name: "radfoam-test-out",
        format: gpu::TextureFormat::Rgba16Float,
        size: gpu::Extent {
            width: w.max(1),
            height: h.max(1),
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: gpu::TextureDimension::D2,
        usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
        external: None,
    });
    let view = context.create_texture_view(
        tex,
        gpu::TextureViewDesc {
            name: "radfoam-test-out-view",
            format: gpu::TextureFormat::Rgba16Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );
    (tex, view)
}

fn create_readback_buffer(context: &gpu::Context, byte_size: u64) -> gpu::Buffer {
    context.create_buffer(gpu::BufferDesc {
        name: "radfoam-test-readback",
        size: byte_size.max(4),
        memory: gpu::Memory::Shared,
    })
}

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

/// Decode RGBA16F pixel data to Vec4.
fn decode_rgba16f(data: &[u16], pixel_idx: usize) -> glam::Vec4 {
    let base = pixel_idx * 4;
    glam::Vec4::new(
        f16::from_bits(data[base]).to_f32(),
        f16::from_bits(data[base + 1]).to_f32(),
        f16::from_bits(data[base + 2]).to_f32(),
        f16::from_bits(data[base + 3]).to_f32(),
    )
}

/// Render `model` on the GPU with the production RadFoam shader and assert each
/// pixel in `test_pixels` matches the CPU reference tracer.
///
/// Used by both the plain RadFoam regression test and the Power Foam radii test
/// below — they only differ in the model that's passed in.
fn assert_gpu_matches_cpu(
    context: gpu::Context,
    mut model: vol::PointCloudModel,
    test_pixels: &[(u32, u32)],
) {
    // Create command encoder early so we can explicitly destroy it for validation cleanliness.
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "radfoam-test",
        buffer_count: 1,
        manual_barriers: false,
    });

    // Upload buffers using the existing helper.
    let mut radfoam_gpu = vol::RadFoamGpuCloud::new(&model, &context, &mut encoder);
    if let Some(ref mut normals) = model.surface_normals {
        normals.fill(glam::Vec3::new(0.15, -0.1, -1.0).normalize());
        radfoam_gpu.update_surface_normals(
            normals,
            model.surface_offsets.as_deref(),
            &context,
            &mut encoder,
        );
    }

    // Compile production shader and pipeline.
    let source = vol::shaders::compose(vol::shaders::RADFOAM);
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let trace_layout = <TraceData as gpu::ShaderData>::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "radfoam-test-trace",
        data_layouts: &[&trace_layout],
        compute: shader.at("trace_main"),
    });

    // Output texture (RGBA16F to match production shader).
    let (out_tex, out_view) = create_output_rgba16(&context, 8, 8);

    // Setup camera: identity orientation looking +Z.
    // Place camera slightly before the start of the spine along -Z looking towards +Z.
    let cam_pos = glam::Vec3::new(0.0, 0.0, -1.0);
    let cam = CameraParams {
        position: cam_pos.into(),
        depth: 100.0,
        orientation: glam::Quat::IDENTITY.into(),
        fov: [1.0, 1.0],
        principal: [0.08, -0.04],
    };

    // Start point:
    // Use the first point (0). For a +Z chain this is always a valid start for a +Z ray.
    let params = TraceParams {
        sh_degree: model.sh_degree as u32,
        weight_threshold: 1e-4,
        max_steps: 256,
        start_point: 0,
        debug_mode: 0, // Normal rendering, not debug visualization
        _align_pad: [0, 0, 0],
        pad: [
            model.radii.is_some() as u32,
            model.surface_normals.is_some() as u32,
            model.surface_color_coefficients.is_some() as u32
                | (model.spherical_voronoi.is_some() as u32) << 1
                | (model.surface_detail.is_some() as u32) << 2
                | (model
                    .surface_detail
                    .as_ref()
                    .is_some_and(|detail| detail.density_logits.is_some())
                    as u32)
                    << 3,
        ],
        _size_pad: 0,
    };

    // Dispatch compute.
    encoder.start();
    encoder.init_texture(out_tex);

    let mut pass = encoder.compute("radfoam-test-trace");
    {
        let mut pen = pass.with(&pipeline);
        pen.bind(
            0,
            &TraceData {
                g_camera: cam,
                g_params: params,
                g_points: radfoam_gpu.points(),
                g_surface_normals: radfoam_gpu.surface_normals(),
                g_surface_details: radfoam_gpu.surface_details(),
                g_attributes: radfoam_gpu.attributes(),
                g_adjacency: radfoam_gpu.point_adjacency(),
                g_adjacency_offsets: radfoam_gpu.point_adjacency_offsets(),
                g_out: out_view,
            },
        );
        // 8x8 with @workgroup_size(8,8,1) => 1 group.
        pen.dispatch([1, 1, 1]);
    }
    drop(pass);

    // Read back pixels via copy_texture_to_buffer.
    // Texture is RGBA16F => 8 bytes per pixel.
    let bytes_per_row = 8 * 8; // 8 pixels * 8 bytes
    let readback = create_readback_buffer(&context, (bytes_per_row * 8) as u64);

    let mut tpass = encoder.transfer("radfoam-test-readback");
    tpass.copy_texture_to_buffer(
        gpu::TexturePiece {
            texture: out_tex,
            mip_level: 0,
            array_layer: 0,
            origin: [0, 0, 0],
        },
        readback.at(0),
        bytes_per_row as u32,
        gpu::Extent {
            width: 8,
            height: 8,
            depth: 1,
        },
    );
    drop(tpass);

    let sp = context.submit(&mut encoder);
    let _ = context.wait_for(&sp, !0);

    // Interpret output (RGBA16F): 8*8 pixels, 4 u16s each.
    let data =
        unsafe { std::slice::from_raw_parts(readback.data() as *const u16, (8 * 8 * 4) as usize) };

    let w = 8.0f32;
    let h = 8.0f32;
    let tan_half = glam::Vec2::new((0.5 * cam.fov[0]).tan(), (0.5 * cam.fov[1]).tan());

    // Use slightly larger epsilon for f16 precision loss
    let eps = 5e-3;

    for &(ix, iy) in test_pixels {
        // GPU pixel readback (decode f16 -> f32)
        let pixel_idx = (iy as usize) * 8 + (ix as usize);
        let gpu_px = decode_rgba16f(data, pixel_idx);

        // CPU ray for this pixel matches shader mapping:
        //   px = (x+0.5)/W; py=(y+0.5)/H; ndc = (px*2-1, py*2-1)
        //   local_dir = ((ndc - principal) * tan(0.5*fov), 1)
        //   ray_dir = normalize(qrot(orientation, local_dir))
        let x = ix as f32;
        let y = iy as f32;
        let px = (x + 0.5) / w;
        let py = (y + 0.5) / h;
        let ndc = glam::Vec2::new(px * 2.0 - 1.0, py * 2.0 - 1.0);
        let local_xy = (ndc - glam::Vec2::from_array(cam.principal)) * tan_half;
        let local_dir = glam::Vec3::new(local_xy.x, local_xy.y, 1.0);
        let ray_dir = local_dir.normalize(); // identity orientation

        let cpu_ray = cpu::Ray {
            origin: cam_pos,
            direction: ray_dir,
        };

        let cpu_settings = cpu::TraceSettings {
            weight_threshold: params.weight_threshold,
            max_steps: params.max_steps,
            start_point: params.start_point,
            depth: cam.depth,
            eval_mode: cpu::EvalMode::Sh,
        };

        let cpu_out = cpu::trace_one_ray(&model, cpu_ray, cpu_settings);
        let cpu_rgba = cpu_out.rgba;

        assert!(
            approx_eq(gpu_px.x, cpu_rgba.x, eps)
                && approx_eq(gpu_px.y, cpu_rgba.y, eps)
                && approx_eq(gpu_px.z, cpu_rgba.z, eps)
                && approx_eq(gpu_px.w, cpu_rgba.w, eps),
            "GPU vs CPU mismatch at pixel ({},{}).\nGPU: {:?}\nCPU: {:?}\n(steps cpu: {}, last_point: {}, t_end: {})",
            ix,
            iy,
            gpu_px,
            cpu_rgba,
            cpu_out.steps,
            cpu_out.last_point,
            cpu_out.t_end
        );
    }

    // Explicit cleanup to satisfy Vulkan validation:
    // destroy readback/output, buffers, pipeline, and command encoder before Context drops.
    cleanup_test_resources(
        &context,
        &mut encoder,
        &mut pipeline,
        &mut radfoam_gpu,
        out_tex,
        out_view,
        readback,
    );
}

// Pixels kept away from edges to reduce sensitivity to mapping conventions.
// The off-diagonal one ensures non-zero NDC x/y to exercise degree-1 SH terms.
const TEST_PIXELS: &[(u32, u32)] = &[(0, 0), (1, 1), (1, 6), (3, 3), (4, 4), (6, 6)];

fn make_branching_test_model() -> vol::PointCloudModel {
    radfoam_synth_branch::make_branching_model(radfoam_synth_branch::BranchingParams {
        spine_len: 64,
        dz: 0.05,
        branch_degree: 6,
        branch_radius: 0.10,
        branch_z_offset: 0.02,
        density: 0.1,
        sh_degree: 3,
        dc: glam::Vec3::splat(0.1),
    })
}

/// Original regression: plain RadFoam (no radii) — GPU must match CPU.
#[test]
fn radfoam_gpu_matches_cpu_on_tiny_fixture_for_some_pixels() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping RadFoam GPU-vs-CPU test: no supported GPU device found");
        return;
    };
    assert_gpu_matches_cpu(context, make_branching_test_model(), TEST_PIXELS);
}

/// Power Foam self-consistency: the same fixture with non-zero per-point radii
/// must produce matching GPU and CPU outputs.
///
/// The radii are deliberately asymmetric (spine vs. branches) so the radical
/// plane shifts noticeably away from the bisector — if either the WGSL or the
/// CPU reference got the formula wrong, the two would diverge here.
#[test]
fn powerfoam_gpu_matches_cpu_with_radii() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping Power Foam GPU-vs-CPU test: no supported GPU device found");
        return;
    };
    let mut model = make_branching_test_model();

    // Matches the BranchingParams above: first `spine_len` points are spine,
    // the rest are branch nodes.
    let spine_len = 64;
    let radii: Vec<f32> = (0..model.points.len())
        .map(|i| if i < spine_len { 0.015 } else { 0.030 })
        .collect();
    model.radii = Some(radii);

    assert_gpu_matches_cpu(context, model, TEST_PIXELS);
}

#[test]
fn oriented_powerfoam_gpu_matches_cpu() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping oriented PowerFoam GPU-vs-CPU test: no supported GPU device found");
        return;
    };
    let mut model = make_branching_test_model();
    model.radii = Some(vec![0.08; model.points.len()]);
    model.surface_normals = Some(vec![-glam::Vec3::Z; model.points.len()]);
    model.surface_offsets = Some(vec![0.01; model.points.len()]);
    model.surface_color_coefficients = Some(
        (0..model.points.len() * vol::SURFACE_COLOR_COMPONENTS * 3)
            .map(|index| (index % 17) as f32 * 0.01 - 0.08)
            .collect(),
    );
    model.surface_detail = Some(vol::SurfaceDetail {
        offsets: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
            .map(|index| {
                let site = (index % vol::SURFACE_DETAIL_SITES) as f32;
                let angle = site * std::f32::consts::TAU / vol::SURFACE_DETAIL_SITES as f32;
                glam::Vec3::new(0.35 * angle.cos(), 0.35 * angle.sin(), 0.1)
            })
            .collect(),
        heights: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
            .map(|index| 0.015 * ((index % 5) as f32 - 2.0))
            .collect(),
        colors: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
            .map(|index| {
                let value = 0.01 * ((index % 7) as f32 - 3.0);
                glam::Vec3::new(value, -0.25 * value, 0.5 * value)
            })
            .collect(),
        density_logits: Some(
            (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
                .map(|index| 0.2 * ((index % vol::SURFACE_DETAIL_SITES) as f32 - 3.5))
                .collect(),
        ),
    });
    model.spherical_voronoi = Some(vol::SphericalVoronoi {
        axes: (0..model.points.len() * vol::SPHERICAL_VORONOI_SITES)
            .map(|index| {
                let x = (index % vol::SPHERICAL_VORONOI_SITES) as f32 - 3.5;
                glam::Vec3::new(x * 0.4, 1.0 - x.abs() * 0.1, -0.5)
            })
            .collect(),
        colors: (0..model.points.len() * vol::SPHERICAL_VORONOI_SITES)
            .map(|index| {
                let value = (index % 13) as f32 * 0.01 - 0.06;
                glam::Vec3::new(value, -0.5 * value, 0.25 * value)
            })
            .collect(),
    });

    // The shared oracle helper replaces these normals after creating the GPU
    // cloud, exercising the lightweight training-cadence upload path.
    assert_gpu_matches_cpu(context, model, TEST_PIXELS);
}

#[test]
fn surface_detail_powerfoam_splat_gpu_matches_cpu() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping PowerFoam splat GPU-vs-CPU test: no supported GPU device found");
        return;
    };
    let mut offsets = vec![0.5 * glam::Vec3::X; vol::SURFACE_DETAIL_SITES];
    offsets[0] = glam::Vec3::ZERO;
    let mut heights = vec![0.0; vol::SURFACE_DETAIL_SITES];
    heights[0] = 0.1;
    let mut colors = vec![glam::Vec3::ZERO; vol::SURFACE_DETAIL_SITES];
    colors[0] = glam::Vec3::new(0.1, -0.03, 0.05);
    let model = vol::PointCloudModel {
        points: vec![glam::Vec4::new(0.0, 0.0, 5.0, 1.2)],
        sh_coefficients: vec![0.0; 3],
        sh_degree: 0,
        transforms: None,
        adjacency: Some(vol::Adjacency {
            neighbors: Vec::new(),
            offsets: vec![0, 0],
        }),
        radii: Some(vec![0.5]),
        surface_normals: Some(vec![-glam::Vec3::Z]),
        surface_offsets: Some(vec![0.02]),
        surface_detail: Some(vol::SurfaceDetail {
            offsets,
            heights,
            colors,
            density_logits: Some(
                (0..vol::SURFACE_DETAIL_SITES)
                    .map(|index| if index == 0 { 2.0 } else { -0.25 })
                    .collect(),
            ),
        }),
        surface_color_coefficients: None,
        spherical_voronoi: None,
    };
    let settings = vol::RadFoamTraceSettings {
        max_steps: 8,
        weight_threshold: 1.0e-4,
        ..vol::RadFoamTraceSettings::default()
    };
    let camera = vol::CameraParams {
        cam_position: [0.0; 3],
        depth: 10.0,
        cam_orientation: glam::Quat::IDENTITY.into(),
        fov: [0.5; 2],
        principal: [0.0; 2],
    };
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "powerfoam-splat-detail-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut tracer =
        vol::PowerFoamGpuSplatTracer::new(&model, settings, [1, 1], &context, &mut encoder);
    let (out_tex, out_view) = create_output_rgba16(&context, 1, 1);
    let readback = create_readback_buffer(&context, 8);
    encoder.start();
    encoder.init_texture(out_tex);
    tracer.dispatch(&mut encoder, out_view, camera);
    let mut transfer = encoder.transfer("powerfoam-splat-detail-readback");
    transfer.copy_texture_to_buffer(
        gpu::TexturePiece {
            texture: out_tex,
            mip_level: 0,
            array_layer: 0,
            origin: [0; 3],
        },
        readback.at(0),
        8,
        gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(transfer);
    let sync = context.submit(&mut encoder);
    assert!(context.wait_for(&sync, !0).unwrap_or(false));
    tracer.validate_candidate_counts().unwrap();

    let raw = unsafe { std::slice::from_raw_parts(readback.data() as *const u16, 4) };
    let actual = glam::Vec4::new(
        f16::from_bits(raw[0]).to_f32(),
        f16::from_bits(raw[1]).to_f32(),
        f16::from_bits(raw[2]).to_f32(),
        f16::from_bits(raw[3]).to_f32(),
    );
    let expected = vol::trace::trace_powerfoam_splats(
        &model,
        vol::trace::Ray {
            origin: glam::Vec3::ZERO,
            direction: glam::Vec3::Z,
        },
        vol::trace::TraceSettings {
            start_point: 0,
            max_steps: settings.max_steps,
            weight_threshold: settings.weight_threshold,
            depth: camera.depth,
            eval_mode: vol::trace::EvalMode::Sh,
        },
    )
    .rgba;
    assert!(
        (actual - expected).abs().max_element() < 5.0e-3,
        "GPU: {actual:?}, CPU: {expected:?}",
    );

    context.destroy_buffer(readback);
    context.destroy_texture_view(out_view);
    context.destroy_texture(out_tex);
    tracer.deinit(&context);
    context.destroy_command_encoder(&mut encoder);
}
