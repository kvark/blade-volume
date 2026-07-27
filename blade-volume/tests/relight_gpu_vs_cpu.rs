#![allow(irrefutable_let_patterns)]

//! The relightable renderer against the same arithmetic on the CPU.
//!
//! `blade_volume::relight::shade` is the readable version of what the shader
//! does, and it is unit tested on its own. Checking the two against each other
//! pins the GPU path to it, so a mistake in the WGSL shows up as a disagreement
//! here rather than as a picture that looks plausible.
//!
//! It also exercises the parts that have nothing to do with shading: the disc
//! rejection that makes a surfel round rather than triangular, and the
//! instance transform that puts it where it belongs.

use blade_graphics as gpu;
use blade_volume as vol;

const SIZE: [u32; 2] = [96, 72];
/// Both sides run the same formulas in the same order, so what separates them
/// is float precision and the sampler's interpolation against the CPU's
/// nearest-texel fetch.
const TOLERANCE: f32 = 0.02;

struct Harness {
    context: gpu::Context,
    encoder: gpu::CommandEncoder,
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
}

impl Harness {
    fn new() -> Option<Self> {
        if std::env::var("BLADE_VOLUME_DISABLE_GPU").is_ok() {
            println!("Skipping: GPU disabled by environment");
            return None;
        }
        let context = match unsafe {
            gpu::Context::init(gpu::ContextDesc {
                ray_tracing: true,
                ..Default::default()
            })
        } {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: no ray tracing context: {e:?}");
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

        let size = gpu::Extent {
            width: SIZE[0],
            height: SIZE[1],
            depth: 1,
        };
        let format = gpu::TextureFormat::Rgba16Float;
        let texture = context.create_texture(gpu::TextureDesc {
            name: "relight-target",
            format,
            size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            // A compute pass cannot write it without STORAGE, and the failure
            // mode is a silently black image rather than an error.
            usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "relight-target",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "relight-readback",
            size: (SIZE[0] * SIZE[1]) as u64 * 8,
            memory: gpu::Memory::Shared,
        });
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "relight-test",
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

    fn render(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        camera: vol::CameraParams,
    ) -> Vec<[f32; 4]> {
        self.encoder.start();
        self.encoder.init_texture(self.texture);
        tracer.dispatch(&mut self.encoder, self.view, camera, SIZE);
        if let mut pass = self.encoder.transfer("relight-readback") {
            pass.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                SIZE[0] * 8,
                gpu::Extent {
                    width: SIZE[0],
                    height: SIZE[1],
                    depth: 1,
                },
            );
        }
        let sync_point = self.context.submit(&mut self.encoder);
        assert!(self.context.wait_for(&sync_point, 20_000).unwrap());

        let count = (SIZE[0] * SIZE[1]) as usize;
        let halves =
            unsafe { std::slice::from_raw_parts(self.readback.data() as *const u16, count * 4) };
        halves
            .chunks_exact(4)
            .map(|texel| {
                let mut out = [0.0f32; 4];
                for (value, half) in out.iter_mut().zip(texel) {
                    *value = half::f16::from_bits(*half).to_f32();
                }
                out
            })
            .collect()
    }

    fn destroy(mut self) {
        self.context.destroy_buffer(self.readback);
        self.context.destroy_texture_view(self.view);
        self.context.destroy_texture(self.texture);
        self.context.destroy_command_encoder(&mut self.encoder);
    }
}

/// A camera looking down `+Z` at the origin from `distance` away.
fn camera(distance: f32) -> vol::CameraParams {
    let fov_y = 0.9f32;
    let aspect = SIZE[0] as f32 / SIZE[1] as f32;
    vol::CameraParams {
        cam_position: [0.0, 0.0, -distance],
        depth: 100.0,
        cam_orientation: [0.0, 0.0, 0.0, 1.0],
        fov: [2.0 * ((0.5 * fov_y).tan() * aspect).atan(), fov_y],
        principal: [0.0, 0.0],
    }
}

/// One surfel per material, spread across the view.
fn scene() -> vol::relight::RelightModel {
    let materials = vec![
        // A rough dielectric: almost all of what it sends back is diffuse.
        vol::relight::Material {
            albedo: [0.8, 0.2, 0.15],
            roughness: 0.9,
            specular_f0: [0.04; 3],
            _padding: 0.0,
        },
        // A smooth metal: no diffuse response at all, and a tinted reflection.
        vol::relight::Material {
            albedo: [0.0; 3],
            roughness: 0.15,
            specular_f0: [0.95, 0.72, 0.35],
            _padding: 0.0,
        },
        // Something in between, where both terms matter.
        vol::relight::Material {
            albedo: [0.3, 0.45, 0.6],
            roughness: 0.45,
            specular_f0: [0.08; 3],
            _padding: 0.0,
        },
    ];
    let surfels = vec![
        vol::relight::Surfel {
            center: [-1.2, 0.0, 0.0],
            radius: 0.5,
            normal: [0.0, 0.0, -1.0],
            material: 0,
        },
        vol::relight::Surfel {
            center: [0.0, 0.0, 0.0],
            radius: 0.5,
            // Tilted, so the reflection direction is not the view direction
            // and the specular lookup is actually exercised.
            normal: glam::Vec3::new(0.35, 0.25, -1.0).normalize().into(),
            material: 1,
        },
        vol::relight::Surfel {
            center: [1.2, 0.0, 0.0],
            radius: 0.5,
            normal: glam::Vec3::new(-0.3, 0.4, -1.0).normalize().into(),
            material: 2,
        },
    ];
    vol::relight::RelightModel { surfels, materials }
}

/// An environment with structure in it, so a mistake in the direction mapping
/// cannot hide behind a constant.
fn environment() -> vol::relight::Environment {
    let (width, height) = (128usize, 64usize);
    let mut texels = Vec::with_capacity(width * height);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let dir = vol::relight::equirect_direction(u, v);
            // A bright patch on one side, a dim sky, and a gradient with
            // height, which between them break every symmetry.
            let sun = (dir
                .dot(glam::Vec3::new(0.6, 0.5, -0.6).normalize())
                .max(0.0))
            .powf(64.0);
            let sky = 0.15 + 0.25 * (0.5 * (dir.y + 1.0));
            texels.push([
                sky + 12.0 * sun,
                sky * 0.9 + 9.0 * sun,
                sky * 1.1 + 6.0 * sun,
            ]);
        }
    }
    vol::relight::Environment {
        width,
        height,
        texels,
    }
}

/// The same ray generation the shader uses, so the two agree on what a pixel
/// looks at before they are asked to agree on what it sees.
fn ray_direction(camera: &vol::CameraParams, x: u32, y: u32) -> glam::Vec3 {
    let px = (x as f32 + 0.5) / SIZE[0] as f32;
    let py = (y as f32 + 0.5) / SIZE[1] as f32;
    let ndc = glam::Vec2::new(px * 2.0 - 1.0, py * 2.0 - 1.0);
    let tan_half = glam::Vec2::new((0.5 * camera.fov[0]).tan(), (0.5 * camera.fov[1]).tan());
    let principal = glam::Vec2::from(camera.principal);
    let local = glam::Vec3::new(
        (ndc.x - principal.x) * tan_half.x,
        (ndc.y - principal.y) * tan_half.y,
        1.0,
    );
    let rotation = glam::Quat::from_xyzw(
        camera.cam_orientation[0],
        camera.cam_orientation[1],
        camera.cam_orientation[2],
        camera.cam_orientation[3],
    );
    (rotation * local).normalize()
}

#[test]
fn gpu_shading_matches_the_cpu_reference() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let model = scene();
    let environment = environment();
    let specular = vol::relight::SpecularEnvironment::prefilter(&environment, 128, 64);
    let background = [0.02, 0.03, 0.05];

    let mut tracer = vol::gpu::RelightTracer::new(
        &model,
        &environment,
        &specular,
        vol::gpu::RelightSettings {
            background_rgb: background,
        },
        &harness.context,
        &mut harness.encoder,
    );
    let camera = camera(4.0);
    let rendered = harness.render(&mut tracer, camera);

    let mut covered = 0usize;
    let mut worst = 0.0f32;
    for y in 0..SIZE[1] {
        for x in 0..SIZE[0] {
            let direction = ray_direction(&camera, x, y);
            let origin = glam::Vec3::from(camera.cam_position);

            // Nearest surfel the ray passes through, the same test the shader
            // makes, so the comparison is of shading rather than of geometry.
            let mut nearest: Option<(f32, &vol::relight::Surfel)> = None;
            for surfel in &model.surfels {
                let normal = glam::Vec3::from(surfel.normal);
                let denominator = direction.dot(normal);
                if denominator.abs() < 1.0e-8 {
                    continue;
                }
                let t = (glam::Vec3::from(surfel.center) - origin).dot(normal) / denominator;
                if t <= 0.0 {
                    continue;
                }
                let offset = origin + t * direction - glam::Vec3::from(surfel.center);
                if offset.length_squared() > surfel.radius * surfel.radius {
                    continue;
                }
                if nearest.is_none_or(|(best, _)| t < best) {
                    nearest = Some((t, surfel));
                }
            }

            let expected = match nearest {
                Some((_, surfel)) => {
                    covered += 1;
                    let mut normal = glam::Vec3::from(surfel.normal);
                    if normal.dot(direction) > 0.0 {
                        normal = -normal;
                    }
                    vol::relight::shade(
                        normal,
                        -direction,
                        &model.materials[surfel.material as usize],
                        &environment.diffuse_irradiance(),
                        &specular,
                    )
                }
                None => background,
            };

            let actual = rendered[(y * SIZE[0] + x) as usize];
            for channel in 0..3 {
                worst = worst.max((actual[channel] - expected[channel]).abs());
            }
        }
    }

    println!("covered {covered} pixels, worst channel difference {worst:.4}");
    assert!(
        covered > 500,
        "the surfels barely covered the frame ({covered} pixels), so this proved little"
    );
    assert!(
        worst < TOLERANCE,
        "GPU and CPU shading differ by {worst:.4}, over the {TOLERANCE} allowed"
    );

    tracer.deinit(&harness.context);
    harness.destroy();
}

#[test]
fn relighting_changes_the_image_and_nothing_else_has_to() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let model = scene();
    let camera = camera(4.0);

    // The same model under two environments. Nothing about the geometry or the
    // materials is rebuilt between them, which is the entire point of storing
    // a material rather than a colour.
    let mut images = Vec::new();
    for tint in [[1.0f32, 0.3, 0.2], [0.2, 0.4, 1.0]] {
        let environment = vol::relight::Environment::uniform(tint, 64, 32);
        let specular = vol::relight::SpecularEnvironment::prefilter(&environment, 64, 32);
        let mut tracer = vol::gpu::RelightTracer::new(
            &model,
            &environment,
            &specular,
            vol::gpu::RelightSettings::default(),
            &harness.context,
            &mut harness.encoder,
        );
        images.push(harness.render(&mut tracer, camera));
        tracer.deinit(&harness.context);
    }

    // Under a red light the surfaces come back red, and under a blue one blue.
    let mut warm = [0.0f64; 3];
    let mut cool = [0.0f64; 3];
    let mut lit = 0usize;
    for (a, b) in images[0].iter().zip(&images[1]) {
        if a[0] + a[1] + a[2] + b[0] + b[1] + b[2] <= 0.0 {
            continue;
        }
        lit += 1;
        for channel in 0..3 {
            warm[channel] += a[channel] as f64;
            cool[channel] += b[channel] as f64;
        }
    }
    assert!(lit > 500, "too little of the frame was lit: {lit}");
    assert!(
        warm[0] > warm[2] * 1.5,
        "a red environment did not produce a red image: {warm:?}"
    );
    assert!(
        cool[2] > cool[0] * 1.5,
        "a blue environment did not produce a blue image: {cool:?}"
    );

    harness.destroy();
}
