//! Score a mesh -> cloud conversion on rendered images.
//!
//! The reference is the source triangle mesh itself, ray traced by
//! `MeshReferenceTracer`. The candidate is the converted cloud, rendered by
//! the ordinary viewer backend. Both use the same camera convention, the same
//! output format, and the same shading model, so the difference between them
//! is the cost of the representation.
//!
//! Usage:
//!   convert_quality [--asset PATH] [--resolutions 32,64,96] [--views 8]
//!                   [--kind radfoam|gaussian] [--size 256] [--dump DIR]

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_view as view;
use std::{env, path, process};

struct Args {
    asset: String,
    resolutions: Vec<f32>,
    views: u32,
    kind: convert::OutputKind,
    size: u32,
    dump: Option<String>,
    exterior_scale: Option<f32>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            asset: "blade-volume-test/data/police.glb".to_string(),
            resolutions: vec![24.0, 48.0, 96.0],
            views: 8,
            kind: convert::OutputKind::RadFoam,
            size: 256,
            dump: None,
            exterior_scale: None,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv = env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < argv.len() {
        let value = |i: usize| -> String {
            argv.get(i + 1)
                .unwrap_or_else(|| fail(&format!("missing value for {}", argv[i])))
                .clone()
        };
        match argv[i].as_str() {
            "--asset" => args.asset = value(i),
            "--resolutions" => {
                args.resolutions = value(i)
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<f32>()
                            .unwrap_or_else(|_| fail("bad resolution"))
                    })
                    .collect()
            }
            "--views" => args.views = value(i).parse().unwrap_or_else(|_| fail("bad views")),
            "--size" => args.size = value(i).parse().unwrap_or_else(|_| fail("bad size")),
            "--kind" => {
                args.kind = match value(i).as_str() {
                    "radfoam" => convert::OutputKind::RadFoam,
                    "gaussian" => convert::OutputKind::Gaussian,
                    other => fail(&format!("unknown kind: {other}")),
                }
            }
            "--dump" => args.dump = Some(value(i)),
            "--exterior-scale" => {
                args.exterior_scale = Some(
                    value(i)
                        .parse()
                        .unwrap_or_else(|_| fail("bad exterior scale")),
                )
            }
            "--help" | "-h" => {
                println!("{}", USAGE);
                process::exit(0);
            }
            other => fail(&format!("unknown option: {other}")),
        }
        i += 2;
    }
    args
}

const USAGE: &str = "usage: convert_quality [options]
  --asset PATH          glTF/glb to convert (default: blade-volume-test/data/police.glb)
  --resolutions A,B,C   conversion resolutions to score (default: 24,48,96)
  --views N             orbit viewpoints (default: 8)
  --kind KIND           radfoam | gaussian (default: radfoam)
  --size N              render resolution, square (default: 256)
  --dump DIR            write reference/candidate PNGs\n  --exterior-scale F    override the transparent exterior fill rate";

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    process::exit(1);
}

/// PSNR over RGB, MAX=1.0, on 8-bit display-referred values.
fn compute_psnr(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let pixels = a.len() / 4;
    let mut sse = 0.0f64;
    for px in 0..pixels {
        for c in 0..3 {
            let d = a[px * 4 + c] as f64 / 255.0 - b[px * 4 + c] as f64 / 255.0;
            sse += d * d;
        }
    }
    let mse = sse / (pixels as f64 * 3.0);
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    (-10.0 * mse.log10()) as f32
}

/// Drop alpha, compositing over black. PSNR is computed on RGB, so this is
/// exactly what is being compared -- an RGBA dump would hide misses behind
/// transparency in most viewers.
fn rgb_over_black(rgba: &[u8]) -> Vec<u8> {
    let pixels = rgba.len() / 4;
    let mut out = vec![0u8; pixels * 3];
    for px in 0..pixels {
        for c in 0..3 {
            out[px * 3 + c] = rgba[px * 4 + c];
        }
    }
    out
}

fn rgba16f_to_rgba8(data: &[u8], pixels: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels * 4];
    for px in 0..pixels {
        for c in 0..4 {
            let bits = u16::from_le_bytes([data[px * 8 + c * 2], data[px * 8 + c * 2 + 1]]);
            let v = half::f16::from_bits(bits).to_f32().clamp(0.0, 1.0);
            out[px * 4 + c] = (v * 255.0).round() as u8;
        }
    }
    out
}

/// Orbit cameras around the mesh bounds. Deterministic, and shared by the
/// reference and every candidate so poses are identical.
fn orbit_cameras(min: glam::Vec3, max: glam::Vec3, count: u32) -> Vec<vol::CameraParams> {
    let center = (min + max) * 0.5;
    let radius = ((max - min).length() * 0.5).max(0.1);
    let distance = radius * 2.5 + 0.1;
    (0..count)
        .map(|i| {
            let azimuth = std::f32::consts::TAU * i as f32 / count as f32;
            // A fixed modest elevation keeps every view looking at the object
            // from a plausible angle rather than edge-on.
            let elevation = 0.35f32;
            let dir = glam::Vec3::new(
                azimuth.cos() * elevation.cos(),
                elevation.sin(),
                azimuth.sin() * elevation.cos(),
            );
            let position = center + dir * distance;
            let forward = (center - position).normalize();
            let mut right = glam::Vec3::Y.cross(forward).normalize_or_zero();
            if right.length_squared() < 1e-6 {
                right = glam::Vec3::X.cross(forward).normalize_or_zero();
            }
            let up = forward.cross(right);
            let orientation = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, forward));
            vol::CameraParams {
                cam_position: position.into(),
                depth: distance + radius * 2.0,
                cam_orientation: orientation.into(),
                fov: [1.0, 1.0],
                principal: [0.0, 0.0],
            }
        })
        .collect()
}

struct Target {
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
    size: u32,
}

impl Target {
    fn new(context: &gpu::Context, size: u32) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name: "quality-target",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent {
                width: size,
                height: size,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            // STORAGE is required for the mesh reference compute pass to write
            // this as a storage texture; TARGET is what the cloud backends use.
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY | gpu::TextureUsage::STORAGE,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "quality-target-view",
                format: gpu::TextureFormat::Rgba16Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "quality-readback",
            size: (size * size * 8) as u64,
            memory: gpu::Memory::Shared,
        });
        Self {
            texture,
            view,
            readback,
            size,
        }
    }

    fn download(&self, encoder: &mut gpu::CommandEncoder) {
        let mut pass = encoder.transfer("quality-readback");
        pass.copy_texture_to_buffer(
            gpu::TexturePiece {
                texture: self.texture,
                mip_level: 0,
                array_layer: 0,
                origin: [0, 0, 0],
            },
            self.readback.at(0),
            self.size * 8,
            gpu::Extent {
                width: self.size,
                height: self.size,
                depth: 1,
            },
        );
    }

    fn pixels(&self) -> Vec<u8> {
        let data = unsafe {
            std::slice::from_raw_parts(
                self.readback.data() as *const u8,
                (self.size * self.size * 8) as usize,
            )
        };
        rgba16f_to_rgba8(data, (self.size * self.size) as usize)
    }

    fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.readback);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

fn main() {
    let args = parse_args();
    if vol::gpu::access_disabled() {
        eprintln!("GPU access disabled by BLADE_VOLUME_DISABLE_GPU; nothing to measure");
        process::exit(0);
    }

    let asset = path::Path::new(&args.asset);
    let mesh = convert::reference_mesh_from_gltf(asset)
        .unwrap_or_else(|err| fail(&format!("reference mesh extraction failed: {err:?}")));
    println!(
        "asset: {} ({} triangles)",
        args.asset,
        mesh.triangle_count()
    );

    // The converter's ambient default must be mirrored by the reference, or
    // the comparison measures exposure rather than geometry.
    let convert_defaults = convert::ConvertOptions::default();
    let ambient: [f32; 3] = convert_defaults.ambient.into();

    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for p in mesh.positions.iter() {
        let v = glam::Vec3::from(*p);
        min = min.min(v);
        max = max.max(v);
    }
    let cameras = orbit_cameras(min, max, args.views);
    println!(
        "bounds: min={min:?} max={max:?}; {} orbit views",
        cameras.len()
    );

    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: false,
            timing: false,
            capture: false,
            overlay: false,
            ray_tracing: true,
            xr: None,
            device_id: None,
        })
        .unwrap_or_else(|err| fail(&format!("GPU context init failed: {err:?}")))
    };
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "convert-quality",
        buffer_count: 1,
        manual_barriers: false,
    });
    let target = Target::new(&context, args.size);

    // ---- Reference: the mesh itself ----
    let mut reference_tracer = vol::MeshReferenceTracer::new(
        &mesh,
        vol::MeshReferenceSettings {
            ambient,
            background_rgb: [0.0; 3],
        },
        &context,
        &mut encoder,
    );
    let mut reference_images = Vec::new();
    for camera in cameras.iter() {
        encoder.start();
        encoder.init_texture(target.texture);
        reference_tracer.dispatch(&mut encoder, target.view, *camera, [args.size; 2]);
        target.download(&mut encoder);
        let sp = context.submit(&mut encoder);
        if !context.wait_for(&sp, 20000).unwrap_or(false) {
            fail("GPU timed out rendering the reference");
        }
        reference_images.push(target.pixels());
    }
    reference_tracer.deinit(&context);
    println!(
        "rendered {} reference views at {}x{}",
        args.views, args.size, args.size
    );

    if let Some(ref dir) = args.dump {
        std::fs::create_dir_all(dir).ok();
        for (i, image) in reference_images.iter().enumerate() {
            let path = path::Path::new(dir).join(format!("ref_{i:02}.png"));
            let rgb = rgb_over_black(image);
            image::save_buffer(&path, &rgb, args.size, args.size, image::ColorType::Rgb8).ok();
        }
    }

    // ---- Candidates: converted clouds ----
    println!();
    println!(
        "{:>10}  {:>10}  {:>9}  {:>9}",
        "resolution", "points", "PSNR dB", "worst dB"
    );
    let mut rows = Vec::new();
    for &resolution in args.resolutions.iter() {
        let options = convert::ConvertOptions {
            output: args.kind,
            resolution: Some(resolution),
            exterior_density_scale: args
                .exterior_scale
                .or(convert::ConvertOptions::default().exterior_density_scale),
            ..convert::ConvertOptions::default()
        };
        let model = convert::convert_gltf(asset, &options)
            .unwrap_or_else(|err| fail(&format!("conversion failed: {err:?}")));

        encoder.start();
        let mut backend = view::RenderBackend::new_for_model(
            &model,
            view::GaussianSettings {
                min_opacity: 0.0004,
                min_transmittance: 0.001,
                debug_mode: view::DebugMode::Off,
            },
            view::RadFoamSettings {
                max_steps: 1024,
                weight_threshold: 0.001,
                debug_mode: view::DebugMode::Off,
                background_rgb: [0.0; 3],
            },
            &context,
            &mut encoder,
            gpu::TextureFormat::Rgba16Float,
            view::RenderSize {
                width: args.size,
                height: args.size,
            },
        );

        let mut psnrs = Vec::new();
        for (i, camera) in cameras.iter().enumerate() {
            encoder.start();
            encoder.init_texture(target.texture);
            backend.render(
                &mut encoder,
                target.view,
                *camera,
                glam::Vec3::from(camera.cam_position),
                view::RenderSize {
                    width: args.size,
                    height: args.size,
                },
            );
            target.download(&mut encoder);
            let sp = context.submit(&mut encoder);
            if !context.wait_for(&sp, 20000).unwrap_or(false) {
                fail("GPU timed out rendering a candidate");
            }
            let candidate = target.pixels();
            psnrs.push(compute_psnr(&reference_images[i], &candidate));
            if let Some(ref dir) = args.dump {
                let path =
                    path::Path::new(dir).join(format!("cand_r{}_{i:02}.png", resolution as u32));
                let rgb = rgb_over_black(&candidate);
                image::save_buffer(&path, &rgb, args.size, args.size, image::ColorType::Rgb8).ok();
            }
        }
        backend.destroy(&context);

        let mean = psnrs.iter().sum::<f32>() / psnrs.len() as f32;
        let worst = psnrs.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "{:>10}  {:>10}  {:>9.2}  {:>9.2}",
            resolution as u32,
            model.len(),
            mean,
            worst
        );
        rows.push((resolution, model.len(), mean, worst));
    }

    target.destroy(&context);
    context.destroy_command_encoder(&mut encoder);

    println!();
    if rows.len() > 1 {
        let first = rows[0].2;
        let last = rows[rows.len() - 1].2;
        println!(
            "refinement: {:.2} -> {:.2} dB across the resolution ladder ({:+.2})",
            first,
            last,
            last - first
        );
    }
}
