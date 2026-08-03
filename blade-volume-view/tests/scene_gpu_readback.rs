//! Physical-GPU pixel validation for the unified point-cloud scene renderer.

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_view as view;
use half::f16;

const SIZE: view::RenderSize = view::RenderSize {
    width: 1,
    height: 1,
};
const BACKGROUND: glam::Vec3 = glam::Vec3::new(0.05, 0.10, 0.15);
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Target {
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
    initialized: bool,
}

impl Target {
    fn new(context: &gpu::Context) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name: "scene-readback-target",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent {
                width: SIZE.width,
                height: SIZE.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "scene-readback-target",
                format: gpu::TextureFormat::Rgba16Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "scene-readback-buffer",
            size: 8,
            memory: gpu::Memory::Shared,
        });
        Self {
            texture,
            view,
            readback,
            initialized: false,
        }
    }

    fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.readback);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

fn test_context(ray_tracing: bool) -> Option<gpu::Context> {
    if vol::gpu::access_disabled() {
        return None;
    }
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            ray_tracing,
            ..gpu::ContextDesc::default()
        })
        .ok()?
    };
    let capabilities = context.capabilities();
    if !capabilities.binding_array
        || (ray_tracing
            && !capabilities
                .ray_query
                .contains(gpu::ShaderVisibility::COMPUTE))
    {
        return None;
    }
    Some(context)
}

fn camera() -> vol::CameraParams {
    vol::CameraParams {
        cam_position: [0.0; 3],
        depth: 10.0,
        cam_orientation: glam::Quat::IDENTITY.into(),
        fov: [0.5; 2],
        principal: [0.0; 2],
    }
}

fn dc(color: glam::Vec3) -> [f32; 3] {
    ((color - 0.5) / 0.282_094_8).to_array()
}

fn powerfoam_model(color: glam::Vec3) -> vol::PointCloudModel {
    vol::PointCloudModel {
        points: vec![glam::Vec4::new(0.0, 0.0, 0.0, 1.0)],
        sh_coefficients: dc(color).to_vec(),
        sh_degree: 0,
        transforms: None,
        adjacency: Some(vol::Adjacency {
            neighbors: Vec::new(),
            offsets: vec![0, 0],
        }),
        radii: Some(vec![0.5]),
        surface_normals: None,
        surface_offsets: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    }
}

fn gaussian_model(color: glam::Vec3, scale: glam::Vec3) -> vol::PointCloudModel {
    vol::PointCloudModel {
        points: vec![glam::Vec4::new(0.0, 0.0, 0.0, 0.8)],
        sh_coefficients: dc(color).to_vec(),
        sh_degree: 0,
        transforms: Some(vol::Transforms {
            rotations: vec![glam::Quat::IDENTITY],
            scales: vec![scale],
        }),
        adjacency: None,
        radii: None,
        surface_normals: None,
        surface_offsets: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    }
}

fn read_pixel(
    renderer: &mut view::SceneRenderer,
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    target: &mut Target,
) -> glam::Vec4 {
    encoder.start();
    if !target.initialized {
        encoder.init_texture(target.texture);
        target.initialized = true;
    }
    renderer.render(
        encoder,
        target.view,
        camera(),
        glam::Vec3::ZERO,
        SIZE,
        context,
    );
    let mut transfer = encoder.transfer("scene-readback");
    transfer.copy_texture_to_buffer(
        gpu::TexturePiece {
            texture: target.texture,
            mip_level: 0,
            array_layer: 0,
            origin: [0, 0, 0],
        },
        target.readback.at(0),
        8,
        gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(transfer);
    let sync = context.submit(encoder);
    assert!(
        context.wait_for(&sync, !0).unwrap_or(false),
        "scene pixel readback timed out"
    );

    let values = unsafe { std::slice::from_raw_parts(target.readback.data() as *const u16, 4) };
    glam::Vec4::new(
        f16::from_bits(values[0]).to_f32(),
        f16::from_bits(values[1]).to_f32(),
        f16::from_bits(values[2]).to_f32(),
        f16::from_bits(values[3]).to_f32(),
    )
}

fn assert_close(actual: glam::Vec3, expected: glam::Vec3, tolerance: f32) {
    assert!(
        (actual - expected).abs().max_element() <= tolerance,
        "actual {actual:?}, expected {expected:?}, tolerance {tolerance}"
    );
}

fn composite(color: glam::Vec3, alpha: f32) -> glam::Vec3 {
    alpha * color + (1.0 - alpha) * BACKGROUND
}

#[test]
fn transformed_powerfoam_scene_matches_analytic_pixels() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = test_context(false) else {
        eprintln!("skipping PowerFoam scene readback: no binding-array GPU");
        return;
    };
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "powerfoam-scene-readback",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut renderer = view::SceneRenderer::new(&context, gpu::TextureFormat::Rgba16Float, SIZE);
    renderer.background_rgb = BACKGROUND.to_array();
    let color = glam::Vec3::new(0.8, 0.25, 0.1);
    let object = renderer.add_radfoam(&powerfoam_model(color), &context, &mut encoder);
    let mut target = Target::new(&context);

    renderer.scene.set_transform(
        object,
        vol::Transform {
            position: glam::Vec3::new(0.0, 0.0, 3.0),
            ..vol::Transform::identity()
        },
    );
    let identity = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    assert_close(
        identity.truncate(),
        composite(color, 1.0 - (-1.0_f32).exp()),
        3.0e-3,
    );
    assert!((identity.w - 1.0).abs() <= 1.0e-3);

    renderer.scene.set_transform(
        object,
        vol::Transform {
            position: glam::Vec3::new(0.0, 0.0, 3.0),
            scale: glam::Vec3::splat(2.0),
            ..vol::Transform::identity()
        },
    );
    let uniform_scale = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    assert_close(
        uniform_scale.truncate(),
        composite(color, 1.0 - (-2.0_f32).exp()),
        3.0e-3,
    );

    let elongated = vol::Transform {
        position: glam::Vec3::new(0.35, 0.0, 3.0),
        scale: glam::Vec3::new(2.0, 0.5, 1.0),
        ..vol::Transform::identity()
    };
    renderer.scene.set_transform(object, elongated);
    let nonuniform_scale = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    let interval = (1.0_f32 - (0.35_f32 / 1.0).powi(2)).sqrt();
    assert_close(
        nonuniform_scale.truncate(),
        composite(color, 1.0 - (-interval).exp()),
        3.0e-3,
    );

    renderer.scene.set_transform(
        object,
        vol::Transform {
            rotation: glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            ..elongated
        },
    );
    let rotated_miss = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    assert_close(rotated_miss.truncate(), BACKGROUND, 2.0e-3);

    renderer.scene.set_transform(
        object,
        vol::Transform {
            position: glam::Vec3::new(2.0, 0.0, 3.0),
            ..vol::Transform::identity()
        },
    );
    let translated_miss = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    assert_close(translated_miss.truncate(), BACKGROUND, 2.0e-3);

    target.destroy(&context);
    renderer.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn oriented_powerfoam_scene_applies_the_surface_offset() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = test_context(false) else {
        eprintln!("skipping oriented PowerFoam scene readback: no binding-array GPU");
        return;
    };
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "oriented-powerfoam-scene-readback",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut renderer = view::SceneRenderer::new(&context, gpu::TextureFormat::Rgba16Float, SIZE);
    renderer.background_rgb = BACKGROUND.to_array();
    let color = glam::Vec3::new(0.8, 0.25, 0.1);
    let mut model = powerfoam_model(color);
    model.surface_normals = Some(vec![-glam::Vec3::Z]);
    model.surface_offsets = Some(vec![0.125]);
    let mut surface_color = vec![0.0; vol::SURFACE_COLOR_COMPONENTS * 3];
    surface_color[0..3].copy_from_slice(&[0.4, 0.0, 0.0]);
    surface_color[9..12].copy_from_slice(&[0.0, 0.3, 0.0]);
    model.surface_color_coefficients = Some(surface_color);
    let mut axes = vec![glam::Vec3::ZERO; vol::SPHERICAL_VORONOI_SITES];
    let mut colors = vec![glam::Vec3::ZERO; vol::SPHERICAL_VORONOI_SITES];
    axes[0] = 20.0 * glam::Vec3::Z;
    colors[0] = glam::Vec3::new(0.0, 0.0, 0.2);
    model.spherical_voronoi = Some(vol::SphericalVoronoi { axes, colors });
    let object = renderer.add_radfoam(&model, &context, &mut encoder);
    let position = glam::Vec3::new(0.2, 0.0, 3.0);
    renderer.scene.set_transform(
        object,
        vol::Transform {
            position,
            ..vol::Transform::identity()
        },
    );
    let mut target = Target::new(&context);

    let pixel = read_pixel(&mut renderer, &context, &mut encoder, &mut target);

    let local_ray = vol::trace::Ray {
        origin: -position,
        direction: glam::Vec3::Z,
    };
    let expected = vol::trace::trace_one_ray(
        &model,
        local_ray,
        vol::trace::TraceSettings {
            start_point: 0,
            max_steps: 128,
            depth: 10.0,
            weight_threshold: 0.001,
            eval_mode: vol::trace::EvalMode::Sh,
        },
    )
    .rgba;
    assert_close(
        pixel.truncate(),
        expected.truncate() + (1.0 - expected.w) * BACKGROUND,
        3.0e-3,
    );
    assert!((pixel.w - 1.0).abs() <= 1.0e-3);
    target.destroy(&context);
    renderer.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn gaussian_scene_uses_independent_clouds_and_transforms() {
    let _gpu_test_guard = gpu_test_guard();
    let Some(context) = test_context(true) else {
        eprintln!("skipping Gaussian scene readback: no compute ray-query GPU");
        return;
    };
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gaussian-scene-readback",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut renderer = view::SceneRenderer::new(&context, gpu::TextureFormat::Rgba16Float, SIZE);
    renderer.background_rgb = BACKGROUND.to_array();
    renderer.params.weight_threshold = 0.0;
    let red = glam::Vec3::new(0.85, 0.1, 0.1);
    let blue = glam::Vec3::new(0.1, 0.2, 0.9);
    let red_model = gaussian_model(red, glam::Vec3::splat(0.2));
    let blue_model = gaussian_model(blue, glam::Vec3::splat(0.2));
    let red_object = renderer.add_gaussian(&red_model, 0.01, &context, &mut encoder);
    let blue_object = renderer.add_gaussian(&blue_model, 0.01, &context, &mut encoder);
    renderer.scene.set_transform(
        red_object,
        vol::Transform::from_position(glam::Vec3::new(0.0, 0.0, 2.0)),
    );
    renderer.scene.set_transform(
        blue_object,
        vol::Transform::from_position(glam::Vec3::new(0.0, 0.0, 4.0)),
    );
    let mut target = Target::new(&context);
    let layered = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    renderer.set_debug_mode(view::SceneDebugMode::Bounds);
    let bounds_debug = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    renderer.set_debug_mode(view::SceneDebugMode::ObjectType);
    let type_debug = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    renderer.set_debug_mode(view::SceneDebugMode::Off);
    assert_close(bounds_debug.truncate(), glam::Vec3::X, 2.0e-3);
    assert_close(
        type_debug.truncate(),
        glam::Vec3::new(1.0, 0.3, 0.3),
        2.0e-3,
    );

    let ray = vol::trace::Ray {
        origin: glam::Vec3::ZERO,
        direction: glam::Vec3::Z,
    };
    let settings = vol::trace::GaussianTraceSettings {
        min_opacity: 0.01,
        min_transmittance: 0.0,
        t_start: 0.0,
        t_end: 10.0,
    };
    let mut red_world = red_model.clone();
    red_world.points[0].z = 2.0;
    let mut blue_world = blue_model.clone();
    blue_world.points[0].z = 4.0;
    let red_result = vol::trace::trace_gaussians(&red_world, ray, settings).rgba;
    let blue_result = vol::trace::trace_gaussians(&blue_world, ray, settings).rgba;
    let expected_rgb = red_result.truncate()
        + (1.0 - red_result.w) * blue_result.truncate()
        + (1.0 - red_result.w) * (1.0 - blue_result.w) * BACKGROUND;
    assert_close(layered.truncate(), expected_rgb, 3.0e-3);

    let anisotropic = gaussian_model(red, glam::Vec3::new(0.4, 0.08, 0.08));
    let anisotropic_object = renderer.add_gaussian(&anisotropic, 0.01, &context, &mut encoder);
    renderer.scene.set_transform(
        red_object,
        vol::Transform::from_position(glam::Vec3::new(10.0, 0.0, 2.0)),
    );
    renderer.scene.set_transform(
        blue_object,
        vol::Transform::from_position(glam::Vec3::new(10.0, 0.0, 4.0)),
    );
    let elongated = vol::Transform::from_position(glam::Vec3::new(0.25, 0.0, 3.0));
    renderer.scene.set_transform(anisotropic_object, elongated);
    let unrotated = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    assert!(
        (unrotated.truncate() - BACKGROUND).abs().max_element() > 0.05,
        "unrotated anisotropic Gaussian should cover the center ray: {unrotated:?}"
    );

    renderer.scene.set_transform(
        anisotropic_object,
        vol::Transform {
            rotation: glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            ..elongated
        },
    );
    let rotated = read_pixel(&mut renderer, &context, &mut encoder, &mut target);
    assert_close(rotated.truncate(), BACKGROUND, 2.0e-3);

    target.destroy(&context);
    renderer.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}
