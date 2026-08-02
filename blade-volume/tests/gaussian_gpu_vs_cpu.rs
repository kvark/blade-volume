//! Production Gaussian WGSL versus the exhaustive CPU maximum-response oracle.

use blade_graphics as gpu;
use blade_volume as vol;
use half::f16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GaussianParams {
    min_opacity: f32,
    min_transmittance: f32,
    sh_degree: u32,
    debug_mode: u32,
    pad: [u32; 4],
}

#[derive(blade_macros::ShaderData)]
struct GaussianDrawData {
    g_camera: vol::CameraParams,
    g_params: GaussianParams,
    g_gaussian_tlas: gpu::AccelerationStructure,
    g_data: gpu::BufferPiece,
}

fn dc(color: glam::Vec3) -> [f32; 3] {
    ((color - 0.5) / 0.282_094_8).to_array()
}

fn adversarial_model() -> vol::PointCloudModel {
    // The broad far particle's support proxy starts before the narrow near
    // particle's proxy. Correct 3DGRT ordering is nevertheless near red at
    // maximum-response t=2, followed by far blue at t=4.
    let points = vec![
        glam::Vec4::new(0.0, 0.0, 4.0, 0.5),
        glam::Vec4::new(0.0, 0.0, 2.0, 0.5),
    ];
    let mut sh_coefficients = Vec::new();
    sh_coefficients.extend_from_slice(&dc(glam::Vec3::new(0.0, 0.0, 1.0)));
    sh_coefficients.extend_from_slice(&dc(glam::Vec3::new(1.0, 0.0, 0.0)));
    vol::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree: 0,
        transforms: Some(vol::Transforms {
            rotations: vec![glam::Quat::IDENTITY; 2],
            scales: vec![
                glam::Vec3::new(1.0, 1.0, 1.0),
                glam::Vec3::new(1.0, 1.0, 0.1),
            ],
        }),
        adjacency: None,
        radii: None,
    }
}

#[test]
fn gaussian_gpu_orders_overlapping_scales_like_cpu_oracle() {
    if vol::gpu::access_disabled() {
        eprintln!("skipping Gaussian GPU/CPU parity: GPU access disabled");
        return;
    }
    let Some(context) = (unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            ray_tracing: true,
            ..gpu::ContextDesc::default()
        })
        .ok()
    }) else {
        eprintln!("skipping Gaussian GPU/CPU parity: no ray-query GPU");
        return;
    };

    let model = adversarial_model();
    let params = GaussianParams {
        min_opacity: 0.01,
        min_transmittance: 0.0,
        sh_degree: 0,
        debug_mode: 0,
        pad: [0; 4],
    };
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gaussian-oracle-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut cloud = vol::GaussianGpuCloud::new(
        &model,
        &vol::InitParameters {
            min_opacity: params.min_opacity,
        },
        &context,
        &mut encoder,
    );

    let source = vol::shaders::compose(vol::shaders::GAUSSIAN);
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let draw_layout = <GaussianDrawData as gpu::ShaderData>::layout();
    let mut pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
        name: "gaussian-oracle-test",
        data_layouts: &[&draw_layout],
        primitive: gpu::PrimitiveState {
            topology: gpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        vertex: shader.at("draw_vs"),
        vertex_fetches: &[],
        fragment: Some(shader.at("draw_fs")),
        color_targets: &[gpu::TextureFormat::Rgba16Float.into()],
        depth_stencil: None,
        multisample_state: Default::default(),
    });
    let output = context.create_texture(gpu::TextureDesc {
        name: "gaussian-oracle-output",
        format: gpu::TextureFormat::Rgba16Float,
        size: gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: gpu::TextureDimension::D2,
        usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
        external: None,
    });
    let output_view = context.create_texture_view(
        output,
        gpu::TextureViewDesc {
            name: "gaussian-oracle-output",
            format: gpu::TextureFormat::Rgba16Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "gaussian-oracle-readback",
        size: 8,
        memory: gpu::Memory::Shared,
    });
    let camera = vol::CameraParams {
        cam_position: [0.0; 3],
        depth: 10.0,
        cam_orientation: glam::Quat::IDENTITY.into(),
        fov: [1.0; 2],
        principal: [0.0; 2],
    };

    encoder.start();
    encoder.init_texture(output);
    let mut pass = encoder.render(
        "gaussian-oracle-test",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: output_view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    );
    {
        let mut pen = pass.with(&pipeline);
        pen.bind(
            0,
            &GaussianDrawData {
                g_camera: camera,
                g_params: params,
                g_gaussian_tlas: cloud.tlas,
                g_data: cloud.gauss_buf.into(),
            },
        );
        pen.draw(0, 3, 0, 1);
    }
    drop(pass);
    let mut transfer = encoder.transfer("gaussian-oracle-readback");
    transfer.copy_texture_to_buffer(
        gpu::TexturePiece {
            texture: output,
            mip_level: 0,
            array_layer: 0,
            origin: [0, 0, 0],
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
    assert!(
        context.wait_for(&sync, !0).unwrap_or(false),
        "Gaussian parity readback timed out"
    );

    let values = unsafe { std::slice::from_raw_parts(readback.data() as *const u16, 4) };
    let gpu_rgb = glam::Vec3::new(
        f16::from_bits(values[0]).to_f32(),
        f16::from_bits(values[1]).to_f32(),
        f16::from_bits(values[2]).to_f32(),
    );
    let cpu = vol::trace::trace_gaussians(
        &model,
        vol::trace::Ray {
            origin: glam::Vec3::ZERO,
            direction: glam::Vec3::Z,
        },
        vol::trace::GaussianTraceSettings {
            min_opacity: params.min_opacity,
            min_transmittance: params.min_transmittance,
            t_start: 0.0,
            t_end: camera.depth,
        },
    );
    assert!(
        (gpu_rgb - cpu.rgba.truncate()).abs().max_element() < 2.0e-3,
        "GPU {gpu_rgb:?} disagrees with CPU {:?}",
        cpu.rgba.truncate()
    );

    context.destroy_buffer(readback);
    context.destroy_texture_view(output_view);
    context.destroy_texture(output);
    context.destroy_render_pipeline(&mut pipeline);
    cloud.deinit(&context);
    context.destroy_command_encoder(&mut encoder);
}
