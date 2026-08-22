#![allow(irrefutable_let_patterns)]

//! The viewer's relightable backend, without a window.
//!
//! What the viewer adds on top of the tracer is a presentation step and the
//! ability to change the light while running. Neither is covered by the
//! tracer's own tests, and neither is visible in a still image of the kind the
//! offline tools produce, so both are checked here against an offscreen target
//! in exactly the format a swapchain would hand over.

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_view as view;
use std::sync;

const SIZE: view::RenderSize = view::RenderSize {
    width: 128,
    height: 96,
};
/// The format blade asks a surface for when it can get it: linear storage
/// presented in an sRGB colour space, which means nothing downstream applies
/// the display curve and the blit has to.
const FORMAT: gpu::TextureFormat = gpu::TextureFormat::Bgra8Unorm;

// These tests each own a complete Vulkan device. Creating all three devices
// concurrently intermittently faults inside the NVIDIA loader during memory
// allocation; the same tests and resources are stable when their device
// lifetimes do not overlap.
static GPU_TEST_LOCK: sync::Mutex<()> = sync::Mutex::new(());

struct Harness {
    context: gpu::Context,
    encoder: gpu::CommandEncoder,
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
}

impl Harness {
    fn new() -> Option<Self> {
        if vol::gpu::access_disabled() {
            println!("Skipping: GPU access disabled");
            return None;
        }
        let context = match unsafe {
            gpu::Context::init(gpu::ContextDesc {
                ray_tracing: true,
                ..Default::default()
            })
        } {
            Ok(context) => context,
            Err(error) => {
                println!("Skipping: no ray tracing context: {error:?}");
                return None;
            }
        };
        if !context
            .capabilities()
            .ray_query
            .contains(gpu::ShaderVisibility::COMPUTE)
        {
            println!("Skipping: ray_query in compute is not supported");
            return None;
        }

        let extent = gpu::Extent {
            width: SIZE.width,
            height: SIZE.height,
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "relight-backend-target",
            format: FORMAT,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "relight-backend-target",
                format: FORMAT,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "relight-backend-readback",
            size: (SIZE.width * SIZE.height) as u64 * 4,
            memory: gpu::Memory::Shared,
        });
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "relight-backend",
            buffer_count: 1,
            manual_barriers: false,
        });
        Some(Self {
            context,
            encoder,
            texture,
            view,
            readback,
        })
    }

    /// One frame, as bytes in the order the surface stores them.
    fn render(&mut self, backend: &mut view::RelightBackend) -> Vec<[u8; 4]> {
        self.encoder.start();
        self.encoder.init_texture(self.texture);
        backend.render(&mut self.encoder, self.view, camera(), SIZE);
        if let mut pass = self.encoder.transfer("relight-backend-readback") {
            pass.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                SIZE.width * 4,
                gpu::Extent {
                    width: SIZE.width,
                    height: SIZE.height,
                    depth: 1,
                },
            );
        }
        let sync_point = self.context.submit(&mut self.encoder);
        assert!(self.context.wait_for(&sync_point, 30_000).unwrap());

        let count = (SIZE.width * SIZE.height) as usize;
        unsafe { std::slice::from_raw_parts(self.readback.data() as *const [u8; 4], count) }
            .to_vec()
    }

    fn destroy(mut self) {
        self.context.destroy_buffer(self.readback);
        self.context.destroy_texture_view(self.view);
        self.context.destroy_texture(self.texture);
        self.context.destroy_command_encoder(&mut self.encoder);
    }
}

fn camera() -> vol::CameraParams {
    vol::CameraParams::looking_at(
        glam::Vec3::new(0.0, 0.0, -4.0),
        glam::Vec3::ZERO,
        0.9,
        SIZE.width as f32 / SIZE.height as f32,
        100.0,
    )
}

/// One disc facing the camera, filling much of the frame.
fn model() -> vol::relight::RelightModel {
    vol::relight::RelightModel {
        kernel: vol::relight::ParticleKernel::Compact,
        surfels: vec![vol::relight::Surfel {
            center: [0.0; 3],
            radius: 1.5,
            normal: [0.0, 0.0, -1.0],
            material: 0,
        }],
        materials: vec![vol::relight::Material {
            // Neutral, so the colour of the frame is the colour of the light
            // and not a mixture of the two that has to be reasoned about.
            albedo: [0.6; 3],
            roughness: 0.6,
            specular_f0: [0.04; 3],
            _padding: 0.0,
        }],
    }
}

/// Mean of the three colour channels over the frame.
fn mean(pixels: &[[u8; 4]]) -> f64 {
    let total: f64 = pixels
        .iter()
        .map(|texel| (texel[0] as f64 + texel[1] as f64 + texel[2] as f64) / 3.0)
        .sum();
    total / pixels.len() as f64
}

#[test]
fn the_backend_presents_a_frame_and_changes_it_with_the_light() {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(sync::PoisonError::into_inner);
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let environments = vec![
        view::NamedEnvironment::new(
            "warm",
            vol::relight::Environment::uniform([0.8, 0.25, 0.1], 32, 16),
        ),
        view::NamedEnvironment::new(
            "cool",
            vol::relight::Environment::uniform([0.1, 0.25, 0.8], 32, 16),
        ),
    ];
    let mut backend = view::RelightBackend::new(
        &model(),
        environments,
        view::RelightSettings {
            // Small, because prefiltering is the slow part and this test is
            // about presentation rather than about the ladder's resolution.
            specular_width: 32,
            ..Default::default()
        },
        &harness.context,
        &mut harness.encoder,
        FORMAT,
        SIZE,
    );

    let warm = harness.render(&mut backend);
    assert!(mean(&warm) > 1.0, "the frame came out black");
    assert_eq!(backend.current_environment(), 0);

    backend.next_environment(&harness.context, &mut harness.encoder);
    assert_eq!(backend.current_environment(), 1);
    let cool = harness.render(&mut backend);

    // Bgra: the blue channel is first. Under a red light the frame is red and
    // under a blue one blue, which no amount of caching a stale ladder or a
    // stale set of irradiance coefficients would produce. The ratio is well
    // under the eight to one of the lights themselves, because the display
    // curve compresses it — which is the point of having one.
    let sum =
        |pixels: &[[u8; 4]], channel: usize| pixels.iter().map(|t| t[channel] as u64).sum::<u64>();
    assert!(
        sum(&warm, 2) > 3 * sum(&warm, 0) / 2,
        "the warm light gave blue {} against red {}",
        sum(&warm, 0),
        sum(&warm, 2)
    );
    assert!(
        sum(&cool, 0) > 3 * sum(&cool, 2) / 2,
        "the cool light gave blue {} against red {}",
        sum(&cool, 0),
        sum(&cool, 2)
    );

    // And back again, which is the case the prefilter cache serves.
    backend.next_environment(&harness.context, &mut harness.encoder);
    assert_eq!(backend.current_environment(), 0);
    let again = harness.render(&mut backend);
    assert_eq!(again, warm, "returning to a light gave a different image");

    backend.destroy(&harness.context);
    harness.destroy();
}

/// What is above the camera has to appear at the top of the frame.
///
/// The tracer writes its first row at texture `y = 0` and blade renders with a
/// flipped viewport, so the presentation step has two conventions to reconcile
/// and exactly one way to get it right. Getting it wrong renders every scene
/// upside down, which is invisible in a symmetric test image and reads as an
/// odd camera angle in a real one — it survived a full round of renders that
/// way before this test existed.
#[test]
fn the_frame_is_the_right_way_up() {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(sync::PoisonError::into_inner);
    let Some(mut harness) = Harness::new() else {
        return;
    };
    // One small disc above the axis the camera looks along, and nothing else.
    let model = vol::relight::RelightModel {
        kernel: vol::relight::ParticleKernel::Compact,
        surfels: vec![vol::relight::Surfel {
            center: [0.0, 0.7, 0.0],
            radius: 0.35,
            normal: [0.0, 0.0, -1.0],
            material: 0,
        }],
        materials: vec![vol::relight::Material {
            albedo: [0.9; 3],
            roughness: 0.8,
            specular_f0: [0.04; 3],
            _padding: 0.0,
        }],
    };
    let mut backend = view::RelightBackend::new(
        &model,
        vec![view::NamedEnvironment::new(
            "uniform",
            vol::relight::Environment::uniform([0.6; 3], 32, 16),
        )],
        view::RelightSettings {
            specular_width: 32,
            // A black background, so the only lit pixels are the disc.
            show_environment: false,
            ..Default::default()
        },
        &harness.context,
        &mut harness.encoder,
        FORMAT,
        SIZE,
    );

    let frame = harness.render(&mut backend);
    let half = (SIZE.height / 2 * SIZE.width) as usize;
    let top = mean(&frame[..half]);
    let bottom = mean(&frame[half..]);
    assert!(
        top > 1.0,
        "nothing was drawn at all: top {top:.1}, bottom {bottom:.1}"
    );
    assert!(
        top > 4.0 * bottom,
        "the disc is above the camera axis and rendered at the bottom: \
         top {top:.1}, bottom {bottom:.1}"
    );

    backend.destroy(&harness.context);
    harness.destroy();
}

/// A sun is hundreds of times brighter than white and still has to be
/// distinguishable from a bright surface next to it.
///
/// Without a curve both clip to the same value, which is the failure this
/// backend exists to avoid and which a linear blit would show as a white
/// silhouette on a white background.
#[test]
fn tone_mapping_keeps_bright_things_apart() {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(sync::PoisonError::into_inner);
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let mut backend = view::RelightBackend::new(
        &model(),
        vec![view::NamedEnvironment::new(
            "bright",
            vol::relight::Environment::uniform([8.0; 3], 32, 16),
        )],
        view::RelightSettings {
            specular_width: 32,
            exposure: 1.0,
            ..Default::default()
        },
        &harness.context,
        &mut harness.encoder,
        FORMAT,
        SIZE,
    );

    let normal = harness.render(&mut backend);
    *backend.exposure_mut() = 8.0;
    let brighter = harness.render(&mut backend);

    let dim = mean(&normal);
    let bright = mean(&brighter);
    assert!(
        bright > dim,
        "eight times the exposure did not brighten the frame: {dim:.1} against {bright:.1}"
    );
    assert!(
        bright < 254.0,
        "the frame is saturated at {bright:.1}, so nothing in it is distinguishable"
    );

    backend.destroy(&harness.context);
    harness.destroy();
}

#[test]
fn relightable_gaussian_model_reaches_the_existing_backend() {
    let _gpu_test_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(sync::PoisonError::into_inner);
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let cloud = vol::PointCloudModel {
        points: vec![glam::Vec4::new(0.0, 0.0, 0.0, 0.9)],
        sh_coefficients: vec![0.0; 3],
        sh_degree: 0,
        transforms: Some(vol::Transforms {
            rotations: vec![glam::Quat::IDENTITY],
            scales: vec![glam::Vec3::splat(0.5)],
            pbr: Some(vol::PbrAttributes {
                normals: vec![-glam::Vec3::Z],
                material_indices: vec![0],
                materials: model().materials,
            }),
        }),
        adjacency: None,
        radii: None,
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    };
    let mut backend = view::RelightBackend::new_gaussian(
        &cloud,
        vec![view::NamedEnvironment::new(
            "uniform",
            vol::relight::Environment::uniform([0.6; 3], 32, 16),
        )],
        view::RelightSettings {
            specular_width: 32,
            show_environment: false,
            ..Default::default()
        },
        &harness.context,
        &mut harness.encoder,
        FORMAT,
        SIZE,
    );

    assert!(!backend.supports_shadow_rays());
    assert!(mean(&harness.render(&mut backend)) > 1.0);
    backend.destroy(&harness.context);
    harness.destroy();
}
