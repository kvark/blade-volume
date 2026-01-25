use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_convert as convert;
use blade_volume_view as view;
use std::{env, path};

const REF_WIDTH: u32 = 512;
const REF_HEIGHT: u32 = 512;
const REF_NAME: &str = "police_ref.png";
const INPUT_NAME: &str = "police.glb";
const REF_TOLERANCE: u8 = 3;

fn main() {
    let root = path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data");
    let input_path = data_dir.join(INPUT_NAME);
    let ref_path = data_dir.join(REF_NAME);

    let update = env::args().any(|arg| arg == "--update");

    let mut options = convert::ConvertOptions::default();
    options.output = convert::OutputKind::Gaussian;

    let model = convert::convert_gltf(&input_path, &options)
        .unwrap_or_else(|err| panic!("conversion failed: {err:?}"));

    let image = render_offscreen(&model, REF_WIDTH, REF_HEIGHT);

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

    let mismatch = compare_images(reference.as_raw(), &image, REF_TOLERANCE);
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
        panic!("image mismatch; wrote output to {}", out_path.display());
    }

    println!("image matches reference");
}

fn render_offscreen(model: &vol::PointCloudModel, width: u32, height: u32) -> Vec<u8> {
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: false,
            capture: false,
            overlay: false,
            device_id: 0,
        })
        .expect("create context")
    };

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "volume-test",
        buffer_count: 1,
    });

    let size = view::RenderSize { width, height };
    let debug_mode = view::DebugMode::Off;
    let gaussian_settings = view::GaussianSettings {
        min_opacity: 0.01,
        min_transmittance: 0.01,
        debug_mode,
    };
    let radfoam_settings = view::RadFoamSettings {
        max_steps: 1024,
        weight_threshold: 0.001,
        debug_mode,
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

    let camera = vol::CameraParams {
        cam_position: [3.0, 2.0, -3.0],
        depth: 20.0,
        cam_orientation: glam::Quat::from_rotation_y(0.8)
            .mul_quat(glam::Quat::from_rotation_x(-0.2))
            .into(),
        fov: [1.0, 1.0],
        pad: [0, 0],
    };

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
    context.wait_for(&sp, !0);

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
