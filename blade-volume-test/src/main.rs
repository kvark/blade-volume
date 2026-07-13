use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_view as view;
use std::{env, path};

const REF_WIDTH: u32 = 128;
const REF_HEIGHT: u32 = 128;
const REF_NAME: &str = "police_ref.png";
const INPUT_NAME: &str = "police.glb";
const REF_TOLERANCE: u8 = 3;

fn main() {
    let root = path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data");
    let input_path = data_dir.join(INPUT_NAME);
    let ref_path = data_dir.join(REF_NAME);

    let args: Vec<String> = env::args().collect();
    let update = args.iter().any(|arg| arg == "--update");
    let debug = args.iter().any(|arg| arg == "--debug");

    let options = convert::ConvertOptions {
        output: convert::OutputKind::Gaussian,
        density: 80.0,
        surface_density_scale: 80.0,
        interior_density_scale: 6.0,
        ambient: glam::Vec3::splat(0.9),
        surface_scale: 0.028,
        interior_scale: 0.3,
        surface_opacity: 0.8,
        interior_opacity: 0.95,
        surface_normal_scale: 0.015,
        ..Default::default()
    };

    let model = convert::convert_gltf(&input_path, &options)
        .unwrap_or_else(|err| panic!("conversion failed: {err:?}"));
    println!("converted points: {}", model.len());

    let image = render_offscreen(&model, REF_WIDTH, REF_HEIGHT, debug);

    if update || !ref_path.exists() {
        image::save_buffer(
            &ref_path,
            &image,
            REF_WIDTH,
            REF_HEIGHT,
            image::ColorType::Rgba8,
        )
        .unwrap();
        println!("wrote reference image: {}", ref_path.display());
        return;
    }

    let reference = image::open(&ref_path)
        .unwrap_or_else(|err| panic!("failed to load reference image: {err}"))
        .into_rgba8();
    if reference.width() != REF_WIDTH || reference.height() != REF_HEIGHT {
        panic!(
            "reference image size mismatch: expected {}x{}, got {}x{}",
            REF_WIDTH,
            REF_HEIGHT,
            reference.width(),
            reference.height()
        );
    }

    let psnr = compute_psnr(reference.as_raw(), &image);
    let mismatch = compare_images(reference.as_raw(), &image, REF_TOLERANCE);
    println!("PSNR vs reference: {psnr:.2} dB");
    if mismatch {
        let out_path = data_dir.join("police_out.png");
        image::save_buffer(
            &out_path,
            &image,
            REF_WIDTH,
            REF_HEIGHT,
            image::ColorType::Rgba8,
        )
        .unwrap();
        if psnr < 30.0 {
            panic!(
                "image mismatch; PSNR {psnr:.2} dB is below 30 dB. \
                 Wrote output to {}",
                out_path.display()
            );
        } else {
            println!(
                "byte-level mismatch but PSNR {psnr:.2} dB ≥ 30 dB; wrote {}",
                out_path.display()
            );
        }
    } else {
        println!("image matches reference (within {REF_TOLERANCE}/channel)");
    }
}

/// PSNR over the RGB channels (alpha ignored), MAX=1.0. Assumes both buffers
/// are RGBA8 of identical dimensions.
fn compute_psnr(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n_px = a.len() / 4;
    if n_px == 0 {
        return f32::NAN;
    }
    let mut sse = 0.0f64;
    for px in 0..n_px {
        for c in 0..3 {
            let da = a[px * 4 + c] as f64 / 255.0;
            let db = b[px * 4 + c] as f64 / 255.0;
            let d = da - db;
            sse += d * d;
        }
    }
    let mse = sse / (n_px as f64 * 3.0);
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    (-10.0 * mse.log10()) as f32
}

fn render_offscreen(model: &vol::PointCloudModel, width: u32, height: u32, debug: bool) -> Vec<u8> {
    assert!(
        !vol::gpu::access_disabled(),
        "GPU access disabled by BLADE_VOLUME_DISABLE_GPU"
    );
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: false,
            capture: false,
            overlay: false,
            ray_tracing: true,
            xr: None,
            device_id: None,
        })
        .expect("create context")
    };

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "volume-test",
        buffer_count: 1,
    });

    let size = view::RenderSize { width, height };
    let gaussian_debug_mode = if debug {
        view::DebugMode::ParticleDensity
    } else {
        view::DebugMode::Off
    };
    let gaussian_settings = view::GaussianSettings {
        min_opacity: 0.0004,
        min_transmittance: 0.001,
        debug_mode: gaussian_debug_mode,
    };
    let radfoam_debug_mode = if debug {
        view::DebugMode::ParticleDensity
    } else {
        view::DebugMode::Off
    };
    let radfoam_settings = view::RadFoamSettings {
        max_steps: 256,
        weight_threshold: 0.05,
        debug_mode: radfoam_debug_mode,
        background_rgb: [0.0; 3],
    };

    let mut backend = view::RenderBackend::new_for_model(
        model,
        gaussian_settings,
        radfoam_settings,
        &context,
        &mut encoder,
        gpu::TextureFormat::Rgba16Float,
        size,
    );

    let output = context.create_texture(gpu::TextureDesc {
        name: "ref-output",
        format: gpu::TextureFormat::Rgba16Float,
        size: gpu::Extent {
            width: width.max(1),
            height: height.max(1),
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
            name: "ref-output-view",
            format: gpu::TextureFormat::Rgba16Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );

    let camera = camera_from_bounds(model);

    encoder.start();
    encoder.init_texture(output);

    backend.render(
        &mut encoder,
        output_view,
        camera,
        glam::Vec3::from(camera.cam_position),
        size,
    );

    let bytes_per_row = width * 8;
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "ref-readback",
        size: (bytes_per_row * height) as u64,
        memory: gpu::Memory::Shared,
    });

    let mut tpass = encoder.transfer("ref-readback");
    tpass.copy_texture_to_buffer(
        gpu::TexturePiece {
            texture: output,
            mip_level: 0,
            array_layer: 0,
            origin: [0, 0, 0],
        },
        readback.at(0),
        bytes_per_row,
        gpu::Extent {
            width,
            height,
            depth: 1,
        },
    );
    drop(tpass);

    let sp = context.submit(&mut encoder);
    match context.wait_for(&sp, 20000) {
        Ok(true) => {}
        Ok(false) => panic!("GPU timed out while rendering reference image"),
        Err(err) => panic!("GPU wait failed: {err:?}"),
    }

    let data = unsafe {
        std::slice::from_raw_parts(
            readback.data() as *const u8,
            (bytes_per_row * height) as usize,
        )
    };

    let rgba = rgba16f_to_rgba8(data, width, height);

    context.destroy_buffer(readback);
    context.destroy_texture_view(output_view);
    context.destroy_texture(output);
    backend.destroy(&context);
    context.destroy_command_encoder(&mut encoder);

    rgba
}

fn rgba16f_to_rgba8(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    let mut idx = 0usize;
    let mut out_idx = 0usize;
    while out_idx < out.len() {
        let r = half::f16::from_bits(u16::from_le_bytes([data[idx], data[idx + 1]])).to_f32();
        let g = half::f16::from_bits(u16::from_le_bytes([data[idx + 2], data[idx + 3]])).to_f32();
        let b = half::f16::from_bits(u16::from_le_bytes([data[idx + 4], data[idx + 5]])).to_f32();
        let a = half::f16::from_bits(u16::from_le_bytes([data[idx + 6], data[idx + 7]])).to_f32();
        idx += 8;

        out[out_idx] = linear_to_srgb(r);
        out[out_idx + 1] = linear_to_srgb(g);
        out[out_idx + 2] = linear_to_srgb(b);
        out[out_idx + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        out_idx += 4;
    }
    out
}

fn linear_to_srgb(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 1.0);
    let srgb = if clamped <= 0.0031308 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

fn compare_images(a: &[u8], b: &[u8], tolerance: u8) -> bool {
    if a.len() != b.len() {
        return true;
    }
    for (aa, bb) in a.iter().zip(b.iter()) {
        if aa.abs_diff(*bb) > tolerance {
            return true;
        }
    }
    false
}

fn camera_from_bounds(model: &vol::PointCloudModel) -> vol::CameraParams {
    let (center, radius) = compute_bounds(model);
    let view_dir = glam::Vec3::new(0.8, -0.4, -0.6).normalize();
    let distance = radius * 2.5 + 0.1;
    let position = center - view_dir * distance;
    let orientation = look_at_orientation(position, center, -glam::Vec3::Y);

    vol::CameraParams {
        cam_position: position.into(),
        depth: distance + radius * 2.0,
        cam_orientation: orientation.into(),
        fov: [1.0, 1.0],
        principal: [0.0, 0.0],
    }
}

fn compute_bounds(model: &vol::PointCloudModel) -> (glam::Vec3, f32) {
    if model.points.is_empty() {
        return (glam::Vec3::ZERO, 1.0);
    }

    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for p in &model.points {
        let v = glam::Vec3::new(p.x, p.y, p.z);
        min = min.min(v);
        max = max.max(v);
    }
    let center = (min + max) * 0.5;
    let mut radius = 0.0f32;
    for p in &model.points {
        let v = glam::Vec3::new(p.x, p.y, p.z);
        radius = radius.max((v - center).length());
    }
    (center, radius.max(0.1))
}

fn look_at_orientation(position: glam::Vec3, target: glam::Vec3, up: glam::Vec3) -> glam::Quat {
    let forward = (target - position).normalize();
    let mut right = up.cross(forward).normalize_or_zero();
    if right.length_squared() < 1e-6 {
        right = glam::Vec3::X.cross(forward).normalize_or_zero();
    }
    let up = forward.cross(right);
    let basis = glam::Mat3::from_cols(right, up, forward);
    glam::Quat::from_mat3(&basis)
}
