#![allow(irrefutable_let_patterns)]

//! Render one relightable model under several environments.
//!
//! The point of the representation is that nothing about the model changes
//! between these images: the same surfels and the same materials go in, and
//! only the light differs. A cloud that stored spherical harmonics could not
//! produce the second image at all without being rebuilt.
//!
//! This goes through the viewer's backend rather than driving the tracer
//! directly, into a target in the format a swapchain would hand over. What
//! comes out is what the window shows — the same tone mapping, the same
//! environment switching — so these images are evidence about the interactive
//! path and not about a second one that resembles it.
//!
//! Usage:
//!   relight_demo [--out <dir>] [--size WxH] [--asset <file>] [--exposure F]

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_view as view;

/// What blade asks a surface for when it can get it.
const FORMAT: gpu::TextureFormat = gpu::TextureFormat::Bgra8Unorm;

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
    vol::relight::RelightModel {
        kernel: vol::relight::ParticleKernel::Compact,
        surfels,
        materials,
    }
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
    let mut exposure = 1.0f32;
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
            "--exposure" => {
                exposure = args
                    .next()
                    .expect("--exposure needs a number")
                    .parse()
                    .expect("bad exposure");
            }
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
            let path = std::path::Path::new(path);
            // A converted file loads as itself. Converting is seconds and
            // deterministic, so anything that renders an asset more than once
            // should be reading one of these instead.
            if path.extension().and_then(|e| e.to_str()) == Some(vol::io::SURFEL_EXTENSION) {
                match vol::io::try_load_relight(path) {
                    Ok(model) => model,
                    Err(e) => {
                        eprintln!("cannot load {}: {e}", path.display());
                        std::process::exit(1);
                    }
                }
            } else {
                let options = convert::ConvertOptions {
                    resolution: Some(resolution),
                    ..Default::default()
                };
                match convert::relight_model_from_gltf(path, &options) {
                    Ok(model) => model,
                    Err(e) => {
                        eprintln!("cannot convert {}: {e:?}", path.display());
                        std::process::exit(1);
                    }
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
    let texture = context.create_texture(gpu::TextureDesc {
        name: "relight-demo",
        format: FORMAT,
        size: extent,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let target = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "relight-demo",
            format: FORMAT,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "relight-demo-readback",
        size: (size[0] * size[1]) as u64 * 4,
        memory: gpu::Memory::Shared,
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "relight-demo",
        buffer_count: 1,
        manual_barriers: false,
    });

    let fov_y = 0.85f32;
    let aspect = size[0] as f32 / size[1] as f32;
    // Frame whatever came in, rather than assuming the grid's extent. The
    // same three-quarter view the viewer opens on, through the same helper:
    // composing the rotation by hand here is what produced a camera whose
    // local `+Y` was world up rather than image down, and so a series of
    // upside-down renders that read as an odd angle rather than as a bug.
    let (min, max) = model.bounds().expect("the model has no surfels");
    let center = 0.5 * (min + max);
    let radius = 0.5 * (max - min).length();
    let distance = 1.15 * radius / (0.5 * fov_y).sin();
    let position = center + glam::Vec3::new(0.7, 0.45, 1.0).normalize() * distance;
    let camera = vol::CameraParams::looking_at(
        position,
        center,
        fov_y,
        aspect,
        distance + 8.0 * radius.max(0.1),
    );

    let named = environments()
        .into_iter()
        .map(|(name, environment)| view::NamedEnvironment::new(name, environment))
        .collect::<Vec<_>>();
    let names = named
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();
    let render_size = view::RenderSize {
        width: size[0],
        height: size[1],
    };
    let mut backend = view::RelightBackend::new(
        &model,
        named,
        view::RelightSettings {
            diffuse_samples: samples,
            // The model against a flat background, so what differs between
            // these images is what the light did to it rather than the sky
            // behind it.
            show_environment: false,
            exposure,
            specular_width: 256,
            initial_environment: 0,
        },
        &context,
        &mut encoder,
        FORMAT,
        render_size,
    );

    for (index, name) in names.iter().enumerate() {
        // Timed, because this is the cost the viewer pays when the light
        // changes: prefiltering the environment, once, on first use.
        let started = std::time::Instant::now();
        backend.set_environment(index, &context, &mut encoder);
        let switched = started.elapsed();
        encoder.start();
        encoder.init_texture(texture);
        backend.render(&mut encoder, target, camera, render_size);
        if let mut pass = encoder.transfer("relight-demo-readback") {
            pass.copy_texture_to_buffer(texture.into(), readback.into(), size[0] * 4, extent);
        }
        let sync_point = context.submit(&mut encoder);
        assert!(context.wait_for(&sync_point, 30_000).unwrap());

        let count = (size[0] * size[1]) as usize;
        let texels =
            unsafe { std::slice::from_raw_parts(readback.data() as *const [u8; 4], count) };
        let mut image = image::RgbImage::new(size[0], size[1]);
        for (index, texel) in texels.iter().enumerate() {
            // Bgra as the surface stores it, already through the display
            // curve: this is the frame, not a second rendering of it.
            image.put_pixel(
                index as u32 % size[0],
                index as u32 / size[0],
                image::Rgb([texel[2], texel[1], texel[0]]),
            );
        }
        let path = out_dir.join(format!("{name}.png"));
        image.save(&path).unwrap();
        println!(
            "wrote {} (switching to this light took {:.2} s)",
            path.display(),
            switched.as_secs_f64()
        );
    }

    backend.destroy(&context);
    context.destroy_buffer(readback);
    context.destroy_texture_view(target);
    context.destroy_texture(texture);
    context.destroy_command_encoder(&mut encoder);
}
