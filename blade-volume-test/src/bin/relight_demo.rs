#![allow(irrefutable_let_patterns)]

//! Render one relightable model under several environments.
//!
//! The point of the representation is that nothing about the model changes
//! between these images: the same surfels and the same materials go in, and
//! only the light differs. A cloud that stored spherical harmonics could not
//! produce the second image at all without being rebuilt.
//!
//! Usage:
//!   relight_demo [--out <dir>] [--size WxH]

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;

const DEFAULT_SIZE: [u32; 2] = [320, 240];

/// A sphere approximated by outward facing discs.
///
/// Discs rather than a mesh because that is what a converter or a
/// reconstruction produces, and because the normal of a surfel is exact where
/// a normal inferred from a covariance is not.
fn sphere(
    center: glam::Vec3,
    radius: f32,
    material: u32,
    rings: usize,
    segments: usize,
) -> Vec<vol::relight::Surfel> {
    let mut surfels = Vec::with_capacity(rings * segments);
    // Overlap the discs a little, so the sphere has no gaps between them.
    let disc = 1.6 * radius * std::f32::consts::PI / segments as f32;
    for ring in 0..rings {
        let theta = std::f32::consts::PI * (ring as f32 + 0.5) / rings as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        // Fewer discs near the poles, where the rings are shorter.
        let count = ((segments as f32 * sin_theta).round() as usize).max(4);
        for segment in 0..count {
            let phi = 2.0 * std::f32::consts::PI * segment as f32 / count as f32;
            let normal = glam::Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
            surfels.push(vol::relight::Surfel {
                center: (center + normal * radius).into(),
                radius: disc,
                normal: normal.into(),
                material,
            });
        }
    }
    surfels
}

/// A grid of spheres sweeping roughness across and metalness down.
///
/// The same arrangement the relighting study used, so what a material does
/// under a new light is legible rather than having to be inferred.
fn scene(tessellation: usize) -> vol::relight::RelightModel {
    const COLUMNS: usize = 5;
    const ROWS: usize = 3;
    const SPACING: f32 = 1.5;
    const BASE: [f32; 3] = [0.95, 0.72, 0.35];

    let mut materials = Vec::new();
    let mut surfels = Vec::new();
    for row in 0..ROWS {
        let metalness = row as f32 / (ROWS - 1) as f32;
        for column in 0..COLUMNS {
            let roughness = 0.08 + 0.92 * column as f32 / (COLUMNS - 1) as f32;
            // The metallic-roughness convention: a metal keeps its colour in
            // the reflectance and loses it from the diffuse albedo.
            materials.push(vol::relight::Material {
                albedo: BASE.map(|c| c * (1.0 - metalness)),
                roughness,
                specular_f0: BASE.map(|c| 0.04 + (c - 0.04) * metalness),
                _padding: 0.0,
            });
            let center = glam::Vec3::new(
                (column as f32 - 0.5 * (COLUMNS - 1) as f32) * SPACING,
                (0.5 * (ROWS - 1) as f32 - row as f32) * SPACING,
                0.0,
            );
            surfels.extend(sphere(
                center,
                0.55,
                (materials.len() - 1) as u32,
                tessellation,
                2 * tessellation,
            ));
        }
    }
    vol::relight::RelightModel { surfels, materials }
}

fn sun(azimuth: f32, elevation: f32, color: [f32; 3], sky: [f32; 3]) -> vol::relight::Environment {
    let (width, height) = (256usize, 128usize);
    let az = azimuth.to_radians();
    let el = elevation.to_radians();
    let dir = glam::Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos());
    let mut texels = Vec::with_capacity(width * height);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let d = vol::relight::equirect_direction(u, v);
            let t = ((d.dot(dir) - 0.995) / 0.004).clamp(0.0, 1.0);
            texels.push([
                sky[0] + (color[0] - sky[0]) * t,
                sky[1] + (color[1] - sky[1]) * t,
                sky[2] + (color[2] - sky[2]) * t,
            ]);
        }
    }
    vol::relight::Environment {
        width,
        height,
        texels,
    }
}

fn studio() -> vol::relight::Environment {
    let (width, height) = (256usize, 128usize);
    let key = glam::Vec3::new(0.6, 0.4, -0.7).normalize();
    let fill = glam::Vec3::new(-0.8, 0.2, 0.5).normalize();
    let mut texels = Vec::with_capacity(width * height);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let d = vol::relight::equirect_direction(u, v);
            let k = d.dot(key).max(0.0).powf(120.0);
            let f = d.dot(fill).max(0.0).powf(40.0);
            texels.push([
                0.05 + 30.0 * k + 3.0 * f,
                0.05 + 25.0 * k + 5.0 * f,
                0.06 + 17.0 * k + 11.0 * f,
            ]);
        }
    }
    vol::relight::Environment {
        width,
        height,
        texels,
    }
}

fn environments() -> Vec<(&'static str, vol::relight::Environment)> {
    vec![
        (
            "overcast",
            vol::relight::Environment::uniform([0.55, 0.57, 0.62], 64, 32),
        ),
        (
            "sunset",
            sun(70.0, 12.0, [90.0, 45.0, 18.0], [0.10, 0.09, 0.13]),
        ),
        (
            "noon",
            sun(-25.0, 62.0, [110.0, 108.0, 100.0], [0.16, 0.20, 0.30]),
        ),
        ("studio", studio()),
    ]
}

fn encode_srgb(value: f32) -> u8 {
    // Reinhard first, because a sun a hundred times brighter than the sky has
    // nowhere to go in a display image otherwise.
    let mapped = value.max(0.0) / (1.0 + value.max(0.0));
    let encoded = if mapped <= 0.003_130_8 {
        12.92 * mapped
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn main() {
    let mut out_dir = std::path::PathBuf::from("relight-demo");
    let mut size = DEFAULT_SIZE;
    // Rings around each sphere. Adjacent surfels differ in normal by about
    // 180/tessellation degrees, and normal error is the approximation this
    // representation is most sensitive to.
    let mut tessellation = 24usize;
    let mut asset: Option<String> = None;
    let mut resolution = 96.0f32;
    // Rays per shading point for shadowing and the bounce that comes with it.
    let mut samples = 0u32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_dir = args.next().expect("--out needs a directory").into(),
            "--size" => {
                let spec = args.next().expect("--size needs WxH");
                let (w, h) = spec.split_once('x').expect("--size looks like WxH");
                size = [
                    w.parse().expect("bad width"),
                    h.parse().expect("bad height"),
                ];
            }
            "--tessellation" => {
                tessellation = args
                    .next()
                    .expect("--tessellation needs a count")
                    .parse()
                    .expect("bad tessellation");
            }
            "--asset" => asset = Some(args.next().expect("--asset needs a path")),
            "--samples" => {
                samples = args
                    .next()
                    .expect("--samples needs a count")
                    .parse()
                    .expect("bad sample count");
            }
            "--resolution" => {
                resolution = args
                    .next()
                    .expect("--resolution needs a number")
                    .parse()
                    .expect("bad resolution");
            }
            other => panic!("unexpected argument {other}"),
        }
    }

    let context = match unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        })
    } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no ray tracing context: {e:?}");
            std::process::exit(1);
        }
    };
    let info = context.device_information();
    println!("GPU: {}", info.device_name);
    assert!(
        !info.is_software_emulated,
        "refusing to render on a software rasterizer"
    );
    std::fs::create_dir_all(&out_dir).unwrap();

    // A converted asset when one is given, and the procedural grid otherwise.
    // The asset path is the one that matters: it is the whole point of the
    // representation that a glTF file can arrive carrying its own materials
    // and come out relightable without anything being fitted.
    let model = match asset {
        Some(ref path) => {
            let options = convert::ConvertOptions {
                resolution: Some(resolution),
                ..Default::default()
            };
            match convert::relight_model_from_gltf(std::path::Path::new(path), &options) {
                Ok(model) => model,
                Err(e) => {
                    eprintln!("cannot convert {path}: {e:?}");
                    std::process::exit(1);
                }
            }
        }
        None => scene(tessellation),
    };
    println!(
        "{} surfels over {} materials",
        model.surfels.len(),
        model.materials.len()
    );

    let extent = gpu::Extent {
        width: size[0],
        height: size[1],
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba16Float;
    let texture = context.create_texture(gpu::TextureDesc {
        name: "relight-demo",
        format,
        size: extent,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "relight-demo",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "relight-demo-readback",
        size: (size[0] * size[1]) as u64 * 8,
        memory: gpu::Memory::Shared,
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "relight-demo",
        buffer_count: 1,
        manual_barriers: false,
    });

    let fov_y = 0.85f32;
    let aspect = size[0] as f32 / size[1] as f32;
    // Frame whatever came in, rather than assuming the grid's extent.
    let (min, max) = model.bounds().expect("the model has no surfels");
    let center = 0.5 * (min + max);
    let radius = 0.5 * (max - min).length();
    let distance = radius / (0.5 * fov_y).tan() * 1.15;
    // A three-quarter view, so an asset is seen the way anyone would look at
    // it rather than straight down an axis.
    let (azimuth, elevation) = (0.6f32, 0.35f32);
    let rotation = glam::Quat::from_rotation_y(azimuth) * glam::Quat::from_rotation_x(elevation);
    let camera = vol::CameraParams {
        cam_position: (center - rotation * glam::Vec3::Z * distance).into(),
        depth: 10.0 * distance.max(1.0),
        cam_orientation: [rotation.x, rotation.y, rotation.z, rotation.w],
        fov: [2.0 * ((0.5 * fov_y).tan() * aspect).atan(), fov_y],
        principal: [0.0, 0.0],
    };

    for (name, environment) in environments() {
        let specular = vol::relight::SpecularEnvironment::prefilter(&environment, 256, 128);
        let mut tracer = vol::gpu::RelightTracer::new(
            &model,
            &environment,
            &specular,
            vol::gpu::RelightSettings {
                background_rgb: [0.02, 0.025, 0.035],
                diffuse_samples: samples,
            },
            &context,
            &mut encoder,
        );

        encoder.start();
        encoder.init_texture(texture);
        tracer.dispatch(&mut encoder, view, camera, size);
        if let mut pass = encoder.transfer("relight-demo-readback") {
            pass.copy_texture_to_buffer(texture.into(), readback.into(), size[0] * 8, extent);
        }
        let sync_point = context.submit(&mut encoder);
        assert!(context.wait_for(&sync_point, 30_000).unwrap());

        let count = (size[0] * size[1]) as usize;
        let halves =
            unsafe { std::slice::from_raw_parts(readback.data() as *const u16, count * 4) };
        let mut image = image::RgbImage::new(size[0], size[1]);
        for (index, texel) in halves.chunks_exact(4).enumerate() {
            image.put_pixel(
                index as u32 % size[0],
                index as u32 / size[0],
                image::Rgb([
                    encode_srgb(half::f16::from_bits(texel[0]).to_f32()),
                    encode_srgb(half::f16::from_bits(texel[1]).to_f32()),
                    encode_srgb(half::f16::from_bits(texel[2]).to_f32()),
                ]),
            );
        }
        let path = out_dir.join(format!("{name}.png"));
        image.save(&path).unwrap();
        println!("wrote {}", path.display());

        tracer.deinit(&context);
    }

    context.destroy_buffer(readback);
    context.destroy_texture_view(view);
    context.destroy_texture(texture);
    context.destroy_command_encoder(&mut encoder);
}
