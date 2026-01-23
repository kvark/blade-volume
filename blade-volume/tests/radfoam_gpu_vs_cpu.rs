//! GPU vs CPU RadFoam forward tracing test (initial harness).
//!
//! Goal:
//! - Execute a WGSL compute shader on a synthetic, deterministic fixture
//! - Read back pixels from an RGBA32F output texture
//! - Compare against the CPU reference tracer for matching rays
//!
//! Notes / assumptions:
//! - This test uses a *test-only* compute shader wrapper that writes to `rgba32float`
//!   because readback/decoding is much simpler than `rgba16float`.
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

// Import the CPU reference tracer as a sibling module (same directory).
mod radfoam_cpu_ref;

// Synthetic fixture generators.
mod radfoam_synth_branch;
mod radfoam_synth_chain;

use radfoam_cpu_ref as cpu;

/// A minimal camera uniform matching `examples/radfoam.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraParams {
    position: [f32; 3],
    depth: f32,
    orientation: [f32; 4],
    fov: [f32; 2],
    pad: [u32; 2],
}

/// Trace parameters matching `examples/radfoam.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct TraceParams {
    sh_degree: u32,
    weight_threshold: f32,
    max_steps: u32,
    start_point: u32,
}

#[derive(blade_macros::ShaderData)]
struct TraceData {
    g_camera: CameraParams,
    g_params: TraceParams,
    g_points: gpu::BufferPiece,
    g_attributes: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

/// Test-only WGSL: identical in traversal/integration to runtime, except the output storage
/// texture uses `rgba32float` so we can read it back easily.
///
/// IMPORTANT:
/// - We still rely on name-based binding resolution, so there are no `@group/@binding` annotations.
/// - This shader expects the same packed buffers as the runtime shader.
///
/// For the first correctness check, we use constant RGB (ignore SH) in both CPU and GPU.
const RADFOAM_WGSL_RGBA32: &str = r#"
// Must match `MAX_SH_COMPONENTS` behavior in runtime shader.
const MAX_SH_COMPONENTS: u32 = 16u;

struct Camera {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    pad: vec2<u32>,
}

struct Params {
    sh_degree: u32,
    weight_threshold: f32,
    max_steps: u32,
    start_point: u32,
}

var<uniform> g_camera: Camera;
var<uniform> g_params: Params;

var<storage, read> g_points: array<vec4<f32>>;
var<storage, read> g_attributes: array<f32>;
var<storage, read> g_adjacency: array<u32>;
var<storage, read> g_adjacency_offsets: array<u32>;

// Test-only: rgba32float output for easy readback
var g_out: texture_storage_2d<rgba32float, write>;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn sh_basis_constants() -> array<f32, MAX_SH_COMPONENTS> {
    return array<f32, MAX_SH_COMPONENTS>(
        0.28209479177387814,
        -0.4886025119029199,
        0.4886025119029199,
        -0.4886025119029199,
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    );
}

fn sh_component_count(deg: u32) -> u32 {
    let d = deg + 1u;
    return d * d;
}

fn eval_sh_rgb(point_idx: u32, dir: vec3<f32>) -> vec3<f32> {
    // Test-only: SH degree 3.
    //
    // Packed layout per point row:
    //   coeffs: 3 * sh_components scalars (interleaved RGB per component)
    //   density: last scalar
    //
    // For degree 3: sh_components = (1+3)^2 = 16, sh_dim = 48, attr_dim = 49.
    // Coeff layout (interleaved RGB per component i):
    //   comp0 (DC):   base +  0.. 2
    //   comp1:        base +  3.. 5   (multiplied by dir.y)
    //   comp2:        base +  6.. 8   (multiplied by dir.z)
    //   comp3:        base +  9..11   (multiplied by dir.x)
    //   comp4:        base + 12..14   (multiplied by dir.x*dir.y)
    //   comp5:        base + 15..17   (multiplied by dir.y*dir.z)
    //   comp6:        base + 18..20   (multiplied by (3*dir.z^2 - 1))
    //   comp7:        base + 21..23   (multiplied by dir.x*dir.z)
    //   comp8:        base + 24..26   (multiplied by (dir.x^2 - dir.y^2))
    //   comp9:        base + 27..29   (multiplied by dir.y*(3*dir.x^2 - dir.y^2))
    //   comp10:       base + 30..32   (multiplied by dir.x*dir.y*dir.z)
    //   comp11:       base + 33..35   (multiplied by dir.y*(5*dir.z^2 - 1))
    //   comp12:       base + 36..38   (multiplied by dir.z*(5*dir.z^2 - 3))
    //   comp13:       base + 39..41   (multiplied by dir.x*(5*dir.z^2 - 1))
    //   comp14:       base + 42..44   (multiplied by dir.z*(dir.x^2 - dir.y^2))
    //   comp15:       base + 45..47   (multiplied by dir.x*(dir.x^2 - 3*dir.y^2))
    //
    // Basis constants match our runtime WGSL.
    // Match the runtime shader convention: add a 0.5 bias for visibility.
    let SH0 = 0.28209479177387814;
    let SH1 = -0.4886025119029199;
    let SH2 = 0.4886025119029199;
    let SH3 = -0.4886025119029199;
    let SH4 = 1.0925484305920792;
    let SH5 = -1.0925484305920792;
    let SH6 = 0.31539156525252005;
    let SH7 = -1.0925484305920792;
    let SH8 = 0.5462742152960396;
    let SH9 = -0.5900435899266435;
    let SH10 = 2.890611442640554;
    let SH11 = -0.4570457994644658;
    let SH12 = 0.3731763325901154;
    let SH13 = -0.4570457994644658;
    let SH14 = 1.445305721320277;
    let SH15 = -0.5900435899266435;

    let sh_dim = 48u;        // 3 * 16
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;

    let c0  = vec3<f32>(g_attributes[base + 0u],  g_attributes[base + 1u],  g_attributes[base + 2u]);
    let c1  = vec3<f32>(g_attributes[base + 3u],  g_attributes[base + 4u],  g_attributes[base + 5u]);
    let c2  = vec3<f32>(g_attributes[base + 6u],  g_attributes[base + 7u],  g_attributes[base + 8u]);
    let c3  = vec3<f32>(g_attributes[base + 9u],  g_attributes[base + 10u], g_attributes[base + 11u]);
    let c4  = vec3<f32>(g_attributes[base + 12u], g_attributes[base + 13u], g_attributes[base + 14u]);
    let c5  = vec3<f32>(g_attributes[base + 15u], g_attributes[base + 16u], g_attributes[base + 17u]);
    let c6  = vec3<f32>(g_attributes[base + 18u], g_attributes[base + 19u], g_attributes[base + 20u]);
    let c7  = vec3<f32>(g_attributes[base + 21u], g_attributes[base + 22u], g_attributes[base + 23u]);
    let c8  = vec3<f32>(g_attributes[base + 24u], g_attributes[base + 25u], g_attributes[base + 26u]);
    let c9  = vec3<f32>(g_attributes[base + 27u], g_attributes[base + 28u], g_attributes[base + 29u]);
    let c10 = vec3<f32>(g_attributes[base + 30u], g_attributes[base + 31u], g_attributes[base + 32u]);
    let c11 = vec3<f32>(g_attributes[base + 33u], g_attributes[base + 34u], g_attributes[base + 35u]);
    let c12 = vec3<f32>(g_attributes[base + 36u], g_attributes[base + 37u], g_attributes[base + 38u]);
    let c13 = vec3<f32>(g_attributes[base + 39u], g_attributes[base + 40u], g_attributes[base + 41u]);
    let c14 = vec3<f32>(g_attributes[base + 42u], g_attributes[base + 43u], g_attributes[base + 44u]);
    let c15 = vec3<f32>(g_attributes[base + 45u], g_attributes[base + 46u], g_attributes[base + 47u]);

    let xx = dir.x * dir.x;
    let yy = dir.y * dir.y;
    let zz = dir.z * dir.z;

    var color = vec3<f32>(0.0, 0.0, 0.0);
    color += SH0 * c0;
    color += SH1 * c1 * dir.y;
    color += SH2 * c2 * dir.z;
    color += SH3 * c3 * dir.x;

    color += SH4 * c4 * (dir.x * dir.y);
    color += SH5 * c5 * (dir.y * dir.z);
    color += SH6 * c6 * (3.0 * zz - 1.0);
    color += SH7 * c7 * (dir.x * dir.z);
    color += SH8 * c8 * (xx - yy);

    color += SH9 * c9 * (dir.y * (3.0 * xx - yy));
    color += SH10 * c10 * (dir.x * dir.y * dir.z);
    color += SH11 * c11 * (dir.y * (5.0 * zz - 1.0));
    color += SH12 * c12 * (dir.z * (5.0 * zz - 3.0));
    color += SH13 * c13 * (dir.x * (5.0 * zz - 1.0));
    color += SH14 * c14 * (dir.z * (xx - yy));
    color += SH15 * c15 * (dir.x * (xx - 3.0 * yy));

    return 0.5 + color;
}

fn load_density(point_idx: u32) -> f32 {
    let deg = g_params.sh_degree;
    let comps = min(sh_component_count(deg), MAX_SH_COMPONENTS);
    let sh_dim = 3u * comps;
    let attr_dim = sh_dim + 1u;
    let base = point_idx * attr_dim;
    return g_attributes[base + sh_dim];
}

fn trace_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var t0 = 0.0;
    var transmittance = 1.0;
    var accum_rgb = vec3<f32>(0.0);

    var current = g_params.start_point;
    var current_pos = g_points[current].xyz;

    var steps: u32 = 0u;
    loop {
        steps += 1u;
        if (steps > g_params.max_steps) { break; }
        if (transmittance <= g_params.weight_threshold) { break; }

        let begin = g_adjacency_offsets[current];
        let end = g_adjacency_offsets[current + 1u];
        let num_faces = end - begin;

        var t1: f32 = 3.402823466e+38;
        var next_face: u32 = 0xffffffffu;

        var j: u32 = 0u;
        loop {
            if (j >= num_faces) { break; }
            let next_idx = g_adjacency[begin + j];
            let next_pos = g_points[next_idx].xyz;
            let offset = next_pos - current_pos;

            let face_origin = current_pos + 0.5 * offset;
            let face_normal = offset;

            let dp = dot(face_normal, ray_dir);
            if (dp > 0.0) {
                let t = dot(face_origin - ray_origin, face_normal) / dp;
                if (t < t1) {
                    t1 = t;
                    next_face = j;
                }
            }
            j += 1u;
        }

        if (next_face == 0xffffffffu) { break; }

        let next_idx = g_adjacency[begin + next_face];
        let next_pos = g_points[next_idx].xyz;

        if (t1 > t0) {
            let s = load_density(current);
            if (s > 1e-6) {
                let dt = max(t1 - t0, 0.0);
                let alpha = 1.0 - exp(-s * dt);
                let w = transmittance * alpha;
                let rgb = eval_sh_rgb(current, ray_dir);
                accum_rgb += w * rgb;
                transmittance *= (1.0 - alpha);
            }
        }

        t0 = max(t0, t1);
        current = next_idx;
        current_pos = next_pos;

        if (t0 > g_camera.depth) { break; }
    }

    return vec4<f32>(accum_rgb, 1.0 - transmittance);
}

@compute @workgroup_size(8, 8, 1)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_out);
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }

    let px = (f32(gid.x) + 0.5) / f32(dims.x);
    let py = (f32(gid.y) + 0.5) / f32(dims.y);
    let ndc = vec2<f32>(px * 2.0 - 1.0, py * 2.0 - 1.0);

    let tan_half = tan(0.5 * g_camera.fov);
    let local_dir = vec3<f32>(ndc * tan_half, 1.0);
    let ray_dir = normalize(qrot(g_camera.orientation, local_dir));
    let ray_origin = g_camera.position;

    let rgba = trace_ray(ray_origin, ray_dir);
    textureStore(g_out, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
"#;

/// Create a headless GPU context for tests.
///
/// Note: we keep validation enabled in debug to catch issues early.
fn make_test_context() -> Option<gpu::Context> {
    unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: false,
            capture: false,
            overlay: false,
            device_id: 0,
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
    radfoam_gpu: &mut vol::RadFoamPointCloud,
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

/// Create an RGBA32F storage texture to receive test output.
fn create_output_rgba32(
    context: &gpu::Context,
    w: u32,
    h: u32,
) -> (gpu::Texture, gpu::TextureView) {
    let tex = context.create_texture(gpu::TextureDesc {
        name: "radfoam-test-out",
        format: gpu::TextureFormat::Rgba32Float,
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
            format: gpu::TextureFormat::Rgba32Float,
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

/// GPU-vs-CPU correctness check:
/// - render a tiny 8x8 image
/// - read back a small set of pixels
/// - compare each to CPU reference for the matching ray
///
/// This uses the same camera model as the shader (fullscreen NDC -> local_dir).
#[test]
fn radfoam_gpu_matches_cpu_on_tiny_fixture_for_some_pixels() {
    let Some(context) = make_test_context() else {
        eprintln!("Skipping RadFoam GPU-vs-CPU test: no supported GPU device found");
        return;
    };

    // Build a synthetic, deterministic fixture in-memory.
    //
    // Branching topology: creates higher-degree nodes (multiple neighbors) to
    // exercise multi-neighbor face selection.
    let model = radfoam_synth_branch::make_branching_model(radfoam_synth_branch::BranchingParams {
        spine_len: 64,
        dz: 0.05,
        branch_degree: 6,
        branch_radius: 0.10,
        branch_z_offset: 0.02,
        density: 0.1,
        sh_degree: 3, // SH degree 3
        dc: glam::Vec3::splat(0.1),
    });

    // Create command encoder early so we can explicitly destroy it for validation cleanliness.
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "radfoam-test",
        buffer_count: 1,
    });

    // Upload buffers using the existing helper.
    let mut radfoam_gpu = vol::RadFoamPointCloud::new(&model, &context, &mut encoder);

    // Compile test shader and pipeline.
    let shader = context.create_shader(gpu::ShaderDesc {
        source: RADFOAM_WGSL_RGBA32,
    });
    let trace_layout = <TraceData as gpu::ShaderData>::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "radfoam-test-trace",
        data_layouts: &[&trace_layout],
        compute: shader.at("trace_main"),
    });

    // Output texture.
    let (out_tex, out_view) = create_output_rgba32(&context, 8, 8);

    // Setup camera: identity orientation looking +Z.
    // Place camera slightly before the start of the spine along -Z looking towards +Z.
    let cam_pos = glam::Vec3::new(0.0, 0.0, -1.0);
    let cam = CameraParams {
        position: cam_pos.into(),
        depth: 100.0,
        orientation: glam::Quat::IDENTITY.into(),
        fov: [1.0, 1.0],
        pad: [0, 0],
    };

    // Start point:
    // Use the first point (0). For a +Z chain this is always a valid start for a +Z ray.
    let params = TraceParams {
        sh_degree: model.sh_degree as u32,
        weight_threshold: 1e-4,
        max_steps: 256,
        start_point: 0,
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

    // Read back the first pixel via copy_texture_to_buffer.
    // Texture is RGBA32F => 16 bytes per pixel.
    let bytes_per_row = 8 * 16;
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
    context.wait_for(&sp, !0);

    // Interpret output (RGBA32F): 8*8 pixels, 4 floats each.
    let data =
        unsafe { std::slice::from_raw_parts(readback.data() as *const f32, (8 * 8 * 4) as usize) };

    // Compare a small set of pixels for coverage.
    // Keep these away from edges to reduce sensitivity to mapping conventions.
    let test_pixels: &[(u32, u32)] = &[
        (0, 0),
        (1, 1),
        // Include a pixel that is off the diagonal to ensure non-zero x/y NDC components
        // and therefore exercise degree-1 directional SH terms more strongly.
        (1, 6),
        (3, 3),
        (4, 4),
        (6, 6),
    ];

    let w = 8.0f32;
    let h = 8.0f32;
    let tan_half = glam::Vec2::new((0.5 * cam.fov[0]).tan(), (0.5 * cam.fov[1]).tan());

    let eps = 2e-3;

    for &(ix, iy) in test_pixels {
        // GPU pixel readback
        let idx = ((iy as usize) * 8 + (ix as usize)) * 4;
        let gpu_px = glam::Vec4::new(data[idx + 0], data[idx + 1], data[idx + 2], data[idx + 3]);

        // CPU ray for this pixel matches shader mapping:
        //   px = (x+0.5)/W; py=(y+0.5)/H; ndc = (px*2-1, py*2-1)
        //   local_dir = (ndc * tan(0.5*fov), 1)
        //   ray_dir = normalize(qrot(orientation, local_dir))
        let x = ix as f32;
        let y = iy as f32;
        let px = (x + 0.5) / w;
        let py = (y + 0.5) / h;
        let ndc = glam::Vec2::new(px * 2.0 - 1.0, py * 2.0 - 1.0);
        let local_dir = glam::Vec3::new(ndc.x * tan_half.x, ndc.y * tan_half.y, 1.0);
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
            // Enable limited CPU traversal logging when debugging this test:
            //   RADFOAM_CPU_TRACE=1 cargo test --test radfoam_gpu_vs_cpu -- --nocapture
            debug_max_print_steps: Some(16),
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
