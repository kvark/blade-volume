//! Unified viewer for volumetric data with multiple rendering backends.
//!
//! NOTE: If you change any uniform structs in Rust, make sure the matching WGSL
//! structs (see `blade-volume/shaders/*.wgsl`) match in size/alignment.
//! A mismatch can cause validation asserts or GPU crashes.
//!
//! Usage:
//!   cargo run -p blade-volume-view -- <input_file> [options]
//!
//! The rendering method is automatically detected based on file contents:
//!   - PLY files are examined to detect RadFoam vs Gaussian format
//!   - SPZ files are always Gaussian format
//!   - Use --kind to override auto-detection
//!
//! Controls:
//!   WASD/ZX - Move camera
//!   Q/E - Roll camera
//!   Mouse drag - Look around
//!   Mouse wheel - Adjust fly speed
//!   I - Print info (camera pose, timings)
//!   Tab - Toggle debug mode (particle density visualization)
//!   F1 - Toggle UI overlay
//!   Escape - Exit
//!
//! Optimization work:
//! - RadFoam is being refactored toward a wavefront pipeline. The legacy tracer WGSL is preserved
//!   as `shaders/radfoam_trace_legacy.wgsl`.
//!
//! Benchmark diff mode:
//! - In headless benchmark mode you can optionally render RadFoam legacy and wavefront outputs,
//!   read back the HDR buffers, and report simple error metrics (mean/max absolute RGB error).

#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_view as view;
use std::{collections::VecDeque, fmt, str};

use blade_egui as begui;
use egui as ui;
use egui_winit as ui_winit;

const D2R: f32 = std::f32::consts::PI / 180.0;
const EULER: glam::EulerRot = glam::EulerRot::ZYX;
/// Arguments
#[derive(argh::FromArgs, Clone)]
struct Arguments {
    /// input file path
    #[argh(positional)]
    input_file: String,
    /// target resolution (e.g. 1920,1080)
    #[argh(option)]
    resolution: Option<String>,
    /// camera position and orientation (as Euler degrees): x,y,z,roll,pitch,yaw
    #[argh(option)]
    cam_pose: Option<String>,
    /// override format detection: "gaussian" or "radfoam"
    #[argh(option)]
    kind: Option<String>,
    /// max traversal steps (RadFoam only)
    #[argh(option, default = "1024")]
    max_steps: u32,
    /// stop when transmittance <= threshold (RadFoam only)
    #[argh(option, default = "0.001")]
    weight_threshold: f32,
    /// number of Voronoi traversal steps in the RadFoam init-screen pass (RadFoam only)
    #[argh(option, default = "8")]
    init_steps: u32,
    /// force legacy RadFoam path (single-pass tracer)
    #[argh(switch)]
    legacy: bool,
    /// minimum opacity for Gaussian rendering
    #[argh(option, default = "0.01")]
    min_opacity: f32,
    /// minimum transmittance for Gaussian rendering
    #[argh(option, default = "0.01")]
    min_transmittance: f32,
    /// start in debug mode (particle density visualization)
    #[argh(switch)]
    debug: bool,

    /// run a headless benchmark (no window); renders into an offscreen texture and prints timing stats
    #[argh(switch)]
    benchmark: bool,
    /// number of warmup frames to render before measuring (benchmark mode only)
    #[argh(option, default = "30")]
    benchmark_warmup: u32,
    /// number of measured frames to render (benchmark mode only)
    #[argh(option, default = "120")]
    benchmark_frames: u32,
    /// print per-pass timings for each measured frame (benchmark mode only)
    #[argh(switch)]
    benchmark_verbose: bool,

    /// in benchmark mode, run both RadFoam paths back-to-back (legacy first, then wavefront) and print two CSV sections
    #[argh(switch)]
    benchmark_compare_radfoam: bool,
    /// in benchmark mode, render one legacy frame and one wavefront frame, read back HDR and report error metrics (RadFoam only)
    #[argh(switch)]
    benchmark_diff_radfoam: bool,
}

fn parse_vec<const N: usize, T: Copy + Default + str::FromStr>(string: &str) -> [T; N]
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut vec = [T::default(); N];
    for (elem, sub) in vec.iter_mut().zip(string.split(',')) {
        *elem = sub.parse().unwrap();
    }
    vec
}

// ============================================================================
// Main Application
// ============================================================================

const FRAME_TIME_HISTORY_SIZE: usize = 120;

struct Example {
    camera: view::ControlledCamera,
    backend: view::RenderBackend,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,

    window_size: winit::dpi::PhysicalSize<u32>,
    surface: Option<gpu::Surface>,
    surface_format: gpu::TextureFormat,

    context: gpu::Context,
    debug_mode: view::DebugMode,

    // For command line reproduction
    input_file: String,

    // egui overlay (windowed only)
    ui_ctx: Option<ui::Context>,
    ui_state: Option<ui_winit::State>,
    ui_painter: Option<begui::GuiPainter>,
    ui_show: bool,

    // Frame timing history (milliseconds)
    frame_times: VecDeque<f64>,
}

struct BenchmarkStats {
    measured_frames: u32,
    total_ms_sum: f64,
    total_ms_min: f64,
    total_ms_max: f64,
}

impl BenchmarkStats {
    fn new() -> Self {
        Self {
            measured_frames: 0,
            total_ms_sum: 0.0,
            total_ms_min: f64::INFINITY,
            total_ms_max: 0.0,
        }
    }

    fn record_total_ms(&mut self, total_ms: f64) {
        self.measured_frames += 1;
        self.total_ms_sum += total_ms;
        self.total_ms_min = self.total_ms_min.min(total_ms);
        self.total_ms_max = self.total_ms_max.max(total_ms);
    }

    fn mean_total_ms(&self) -> f64 {
        if self.measured_frames == 0 {
            return 0.0;
        }
        self.total_ms_sum / self.measured_frames as f64
    }
}

impl Example {
    fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
        log::info!("Window size: {:?}", size);
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            color_space: gpu::ColorSpace::Srgb,
            ..Default::default()
        }
    }

    fn init_windowed(window: &winit::window::Window, args: Arguments) -> Self {
        let mut camera = view::ControlledCamera::default();
        if let Some(ref arg) = args.cam_pose {
            let v = parse_vec::<6, f32>(arg);
            camera.position = glam::Vec3::new(v[0], v[1], v[2]);
            camera.orientation = glam::Quat::from_euler(EULER, v[3] * D2R, v[4] * D2R, v[5] * D2R);
        }

        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: true,
                capture: false,
                overlay: true,
                device_id: 0,
            })
            .unwrap()
        };

        let window_size = window.inner_size();
        let surface = context
            .create_surface_configured(window, Self::make_surface_config(window_size))
            .unwrap();
        let surface_info = surface.info();
        let surface_format = surface_info.format;

        // egui init
        let ui_ctx = ui::Context::default();
        let ui_state = ui_winit::State::new(
            ui_ctx.clone(),
            ui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let ui_painter = begui::GuiPainter::new(surface_info, &context);
        let ui_show = true;

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        // Load volume data - use --kind override or auto-detect
        let model = match args.kind.as_deref() {
            Some("gaussian") => {
                log::info!("Loading Gaussian data (forced)");
                vol::io::load_gaussian(&args.input_file)
            }
            Some("radfoam") => {
                log::info!("Loading RadFoam data (forced)");
                vol::io::load_radfoam(&args.input_file)
            }
            Some(other) => panic!(
                "Unknown --kind '{}', expected 'gaussian' or 'radfoam'",
                other
            ),
            None => {
                log::info!("Auto-detecting format...");
                vol::io::load(&args.input_file)
            }
        };

        log::info!(
            "Loaded {} points ({})",
            model.len(),
            if model.transforms.is_some() {
                "Gaussian"
            } else {
                "RadFoam"
            }
        );

        let debug_mode = if args.debug {
            view::DebugMode::ParticleDensity
        } else {
            view::DebugMode::Off
        };

        let size = view::RenderSize {
            width: window_size.width,
            height: window_size.height,
        };
        let gaussian_settings = view::GaussianSettings {
            min_opacity: args.min_opacity,
            min_transmittance: args.min_transmittance,
            debug_mode,
        };
        let radfoam_settings = view::RadFoamSettings {
            max_steps: args.max_steps,
            weight_threshold: args.weight_threshold,
            debug_mode,
        };

        let backend = view::RenderBackend::new_for_model(
            &model,
            gaussian_settings,
            radfoam_settings,
            &context,
            &mut command_encoder,
            surface_info.format,
            size,
        );

        Self {
            camera,
            backend,
            command_encoder,
            prev_sync_point: None,

            window_size,
            surface: Some(surface),
            surface_format,

            context,
            debug_mode,
            input_file: args.input_file,

            ui_ctx: Some(ui_ctx),
            ui_state: Some(ui_state),
            ui_painter: Some(ui_painter),
            ui_show,

            frame_times: VecDeque::with_capacity(FRAME_TIME_HISTORY_SIZE),
        }
    }

    fn init_headless(args: Arguments) -> Self {
        let mut camera = view::ControlledCamera::default();
        if let Some(ref arg) = args.cam_pose {
            let v = parse_vec::<6, f32>(arg);
            camera.position = glam::Vec3::new(v[0], v[1], v[2]);
            camera.orientation = glam::Quat::from_euler(EULER, v[3] * D2R, v[4] * D2R, v[5] * D2R);
        }

        // No presentation/surface in headless mode.
        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: false,
                // Benchmark mode must not enable validation layers; they skew performance.
                validation: false,
                timing: true,
                capture: false,
                // Also keep overlays disabled for benchmark consistency.
                overlay: false,
                device_id: 0,
            })
            .unwrap()
        };

        let window_size = if let Some(ref arg) = args.resolution {
            let res = parse_vec::<2, u32>(arg);
            winit::dpi::PhysicalSize::new(res[0].max(1), res[1].max(1))
        } else {
            winit::dpi::PhysicalSize::new(1280, 720)
        };

        // Pick a stable offscreen format for pipelines in headless mode.
        // This is independent of any swapchain.
        let surface_format = gpu::TextureFormat::Rgba8Unorm;

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "benchmark",
            buffer_count: 2,
        });

        // Load volume data - use --kind override or auto-detect
        let model = match args.kind.as_deref() {
            Some("gaussian") => {
                log::info!("Loading Gaussian data (forced)");
                vol::io::load_gaussian(&args.input_file)
            }
            Some("radfoam") => {
                log::info!("Loading RadFoam data (forced)");
                vol::io::load_radfoam(&args.input_file)
            }
            Some(other) => panic!(
                "Unknown --kind '{}', expected 'gaussian' or 'radfoam'",
                other
            ),
            None => {
                log::info!("Auto-detecting format...");
                vol::io::load(&args.input_file)
            }
        };

        log::info!(
            "Loaded {} points ({})",
            model.len(),
            if model.transforms.is_some() {
                "Gaussian"
            } else {
                "RadFoam"
            }
        );

        let debug_mode = if args.debug {
            view::DebugMode::ParticleDensity
        } else {
            view::DebugMode::Off
        };
        let size = view::RenderSize {
            width: window_size.width,
            height: window_size.height,
        };
        let gaussian_settings = view::GaussianSettings {
            min_opacity: args.min_opacity,
            min_transmittance: args.min_transmittance,
            debug_mode,
        };
        let radfoam_settings = view::RadFoamSettings {
            max_steps: args.max_steps,
            weight_threshold: args.weight_threshold,
            debug_mode,
        };
        let backend = view::RenderBackend::new_for_model(
            &model,
            gaussian_settings,
            radfoam_settings,
            &context,
            &mut command_encoder,
            surface_format,
            size,
        );

        Self {
            camera,

            backend,
            command_encoder,
            prev_sync_point: None,

            window_size,
            surface: None,
            surface_format,

            context,
            debug_mode,
            input_file: args.input_file,

            ui_ctx: None,
            ui_state: None,
            ui_painter: None,
            ui_show: false,

            frame_times: VecDeque::with_capacity(FRAME_TIME_HISTORY_SIZE),
        }
    }

    fn render_headless_benchmark(mut self, args: &Arguments) {
        // CSV output:
        // - one header row
        // - one row per measured frame
        // - final summary row
        //
        // Columns:
        // frame,total_ms,<pass_0_ms>,<pass_1_ms>,...
        //
        // Benchmark notes:
        // - Benchmark mode is intended for Vulkan (notably AMD) and must run with GPU validation disabled,
        //   because validation layers skew timings substantially.
        // - This benchmark is GPU-timing based (`CommandEncoder::timings()`), not CPU frame time.
        //
        // Timing stability notes:
        // - We treat `CommandEncoder::timings()` as the ground truth after submit+wait.
        // - Some backends/drivers may not populate timings for every frame (or may change the pass set).
        //   To keep CSV parseable and debuggable:
        //   - We always print a header once, based on the first *non-empty* timing set seen in measured frames.
        //   - If later frames have extra passes, we ignore them but (in verbose mode) report the mismatch.
        //   - If later frames are missing passes, we print empty cells.
        //
        // Compare mode:
        // - If `--benchmark-compare-radfoam` is set and the backend is RadFoam, run two benchmark sections:
        //   1) legacy (`--wavefront` off)
        //   2) wavefront (`--wavefront` on)
        //   Each section prints its own CSV header and summary, prefixed with a `# section:` comment.
        //
        // Diff mode:
        // - If `--benchmark-diff-radfoam` is set and the backend is RadFoam, render one legacy frame and one
        //   wavefront frame (both offscreen) and report simple error metrics.
        // - This is a lightweight correctness check; it is not a full image diff tool.
        let extent = gpu::Extent {
            width: self.window_size.width.max(1),
            height: self.window_size.height.max(1),
            depth: 1,
        };

        let offscreen_tex = self.context.create_texture(gpu::TextureDesc {
            name: "benchmark-offscreen",
            format: self.surface_format,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            external: None,
        });
        let offscreen_view = self.context.create_texture_view(
            offscreen_tex,
            gpu::TextureViewDesc {
                name: "benchmark-offscreen-view",
                format: self.surface_format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        let diff_tex = self.context.create_texture(gpu::TextureDesc {
            name: "benchmark-diff-readback-src",
            format: self.surface_format,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            external: None,
        });
        let _diff_view = self.context.create_texture_view(
            diff_tex,
            gpu::TextureViewDesc {
                name: "benchmark-diff-readback-view",
                format: self.surface_format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        // IMPORTANT:
        // `CommandEncoder::timings()` may include passes from initialization (e.g. uploads like "radfoam-init")
        // if the first frame doesn't emit any timings. Clear any init-time timings by forcing an empty submit,
        // then begin warmup/measurement after that.
        self.command_encoder.start();
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.context.wait_for(&sync_point, !0);

        // Stats
        let mut stats = BenchmarkStats::new();
        let mut pass_names: Vec<String> = Vec::new();

        // CSV header will be printed once we have a non-empty pass set.
        let mut printed_header = false;

        let total_frames = args.benchmark_warmup + args.benchmark_frames;
        for frame_index in 0..total_frames {
            self.command_encoder.start();
            self.command_encoder.init_texture(offscreen_tex);

            let aspect = self.window_size.width as f32 / self.window_size.height as f32;
            let camera_params = self.camera.to_params(aspect);

            self.backend.render(
                &mut self.command_encoder,
                offscreen_view,
                camera_params,
                self.camera.position,
                view::RenderSize {
                    width: self.window_size.width,
                    height: self.window_size.height,
                },
            );

            let sync_point = self.context.submit(&mut self.command_encoder);
            self.context.wait_for(&sync_point, !0);

            if frame_index < args.benchmark_warmup {
                if args.benchmark_verbose {
                    println!(
                        "warmup frame {}: timings.len = {}",
                        frame_index,
                        self.command_encoder.timings().len()
                    );
                }
                continue;
            }

            let out_frame = frame_index - args.benchmark_warmup;

            // The first measured frame is often skewed (pipeline warmup, caches, driver behavior).
            // Ignore it entirely to keep results more representative.
            if out_frame == 0 {
                if args.benchmark_verbose {
                    println!("measured frame 0: ignored");
                }
                continue;
            }

            // Snapshot timings for this frame (avoid multiple calls while we debug stability).
            let timings = self.command_encoder.timings();

            if args.benchmark_verbose {
                println!(
                    "measured frame {}: timings.len = {}",
                    out_frame,
                    timings.len()
                );
                for &(ref name, value) in timings {
                    println!("\t{}: {:.3} ms", name, value.as_secs_f64() * 1000.0);
                }
            }

            // Build stable pass list from first measured frame that has any timings (after skipping frame 0).
            if !printed_header && !timings.is_empty() {
                for &(ref name, _) in timings {
                    pass_names.push(name.clone());
                }

                // Header + notes (as CSV comment lines for easy copy/paste).
                // Many CSV readers ignore lines starting with '#'.
                println!("# benchmark: validation=disabled");
                println!("# benchmark: timings=gpu (CommandEncoder::timings)");
                println!("# benchmark: backend=vulkan (assumed)");
                println!("# benchmark: note=first measured frame (frame 0) ignored");
                print!("frame,total_ms");
                for name in &pass_names {
                    print!(",{}", name);
                }
                println!();
                printed_header = true;
            }

            // If we still don't have a header (no timings reported yet), emit an empty row for visibility.
            if !printed_header {
                print!("{},", out_frame);
                println!();
                continue;
            }

            // Create a lookup for this frame's timings for the known pass list.
            let mut total_ms: f64 = 0.0;
            let mut values: Vec<Option<f64>> = vec![None; pass_names.len()];
            for &(ref name, value) in timings {
                for (i, pass_name) in pass_names.iter().enumerate() {
                    if name == pass_name {
                        let ms = value.as_secs_f64() * 1000.0;
                        values[i] = Some(ms);
                        total_ms += ms;
                        break;
                    }
                }
            }

            // In verbose mode, report if this frame's pass set differs from the header.
            if args.benchmark_verbose {
                for &(ref name, _) in timings {
                    if !pass_names.iter().any(|n| n == name) {
                        println!(
                            "benchmark: frame {} has extra timing pass '{}' not present in header",
                            out_frame, name
                        );
                    }
                }
                for pass_name in &pass_names {
                    if !timings.iter().any(|(n, _)| n == pass_name) {
                        println!(
                            "benchmark: frame {} is missing timing pass '{}' from header",
                            out_frame, pass_name
                        );
                    }
                }
            }

            stats.record_total_ms(total_ms);

            // Row
            print!("{},{}", out_frame, format!("{:.3}", total_ms));
            for v in values {
                match v {
                    Some(ms) => print!(",{}", format!("{:.3}", ms)),
                    None => print!(","),
                }
            }
            println!();
        }

        fn create_readback_buffer(context: &gpu::Context, byte_size: u64) -> gpu::Buffer {
            context.create_buffer(gpu::BufferDesc {
                name: "benchmark-diff-readback",
                size: byte_size.max(4),
                memory: gpu::Memory::Shared,
            })
        }

        fn readback_u32_buffer(
            context: &gpu::Context,
            encoder: &mut gpu::CommandEncoder,
            src: gpu::BufferPiece,
        ) -> u32 {
            let readback = create_readback_buffer(context, 4);
            encoder.start();
            let mut tpass = encoder.transfer("benchmark-diff-readback-u32");
            tpass.copy_buffer_to_buffer(src, readback.at(0), 4);
            drop(tpass);
            let sp = context.submit(encoder);
            context.wait_for(&sp, !0);

            let value = unsafe { *(readback.data() as *const u32) };
            context.destroy_buffer(readback);
            value
        }

        fn readback_bytes_buffer(
            context: &gpu::Context,
            encoder: &mut gpu::CommandEncoder,
            src: gpu::BufferPiece,
            size: u64,
        ) -> Vec<u8> {
            let readback = create_readback_buffer(context, size);
            encoder.start();
            let mut tpass = encoder.transfer("benchmark-diff-readback-bytes");
            tpass.copy_buffer_to_buffer(src, readback.at(0), size);
            drop(tpass);
            let sp = context.submit(encoder);
            context.wait_for(&sp, !0);

            let bytes =
                unsafe { std::slice::from_raw_parts(readback.data() as *const u8, size as usize) }
                    .to_vec();
            context.destroy_buffer(readback);
            bytes
        }

        // Helper: decode f16->f32 (IEEE 754 half) without external deps.
        fn f16_to_f32(bits: u16) -> f32 {
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1f) as u32;
            let mant = (bits & 0x03ff) as u32;

            let f_bits: u32 = if exp == 0 {
                if mant == 0 {
                    // zero
                    sign << 31
                } else {
                    // subnormal: normalize
                    let mut e: i32 = -14;
                    let mut m = mant;
                    while (m & 0x0400) == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    m &= 0x03ff;
                    let exp_f = (e + 127) as u32;
                    (sign << 31) | (exp_f << 23) | (m << 13)
                }
            } else if exp == 31 {
                // inf/nan
                (sign << 31) | (0xff << 23) | (mant << 13)
            } else {
                // normal
                let exp_f = (exp + (127 - 15)) as u32;
                (sign << 31) | (exp_f << 23) | (mant << 13)
            };

            f32::from_bits(f_bits)
        }

        // Helper: read back RGBA16F texture into Vec<u16> (4 components per pixel).
        fn readback_rgba16f(
            context: &gpu::Context,
            encoder: &mut gpu::CommandEncoder,
            tex: gpu::Texture,
            width: u32,
            height: u32,
        ) -> Vec<u16> {
            // Texture is RGBA16F => 8 bytes per pixel.
            let bytes_per_pixel = 8u32;
            let bytes_per_row = bytes_per_pixel * width;
            let byte_size = (bytes_per_row as u64) * (height as u64);

            let readback = create_readback_buffer(context, byte_size);
            encoder.start();

            // Copy texture -> buffer
            let mut tpass = encoder.transfer("benchmark-diff-readback");
            tpass.copy_texture_to_buffer(
                gpu::TexturePiece {
                    texture: tex,
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

            let sp = context.submit(encoder);
            context.wait_for(&sp, !0);

            let count_u16 = (width as usize) * (height as usize) * 4;
            let pixels =
                unsafe { std::slice::from_raw_parts(readback.data() as *const u16, count_u16) }
                    .to_vec();
            context.destroy_buffer(readback);
            pixels
        }

        if args.benchmark_diff_radfoam {
            println!("benchmark: diff mode is not supported with the shared render backend");
            return;
        }

        if args.benchmark_compare_radfoam {
            println!("benchmark: compare mode is not supported with the shared render backend");
        }

        // Cleanup: no surface in headless mode; in windowed mode surface is dropped with `Example`.
        self.backend.destroy(&self.context);
        self.context.destroy_texture_view(offscreen_view);
        self.context.destroy_texture(offscreen_tex);

        // CommandEncoder owns command buffers / pools; destroy it explicitly to satisfy validation.
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
    }

    fn toggle_debug_mode(&mut self) {
        self.debug_mode = self.debug_mode.toggle();
        self.backend.set_debug_mode(self.debug_mode);
        println!("Debug mode: {:?}", self.debug_mode);
    }

    fn deinit(mut self) {
        self.wait_for_gpu();
        self.backend.destroy(&self.context);

        if let Some(ref mut ui_painter) = self.ui_painter {
            ui_painter.destroy(&self.context);
        }

        // CommandEncoder owns command buffers / pools; destroy it explicitly to satisfy validation.
        // Note: destroy takes `&mut CommandEncoder` (not a reference to it).
        self.context
            .destroy_command_encoder(&mut self.command_encoder);

        if let Some(ref mut surface) = self.surface {
            self.context.destroy_surface(surface);
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = size;
        if let Some(ref mut surface) = self.surface {
            let config = Self::make_surface_config(size);
            self.context.reconfigure_surface(surface, config);
        }

        self.backend.resize(
            &self.context,
            view::RenderSize {
                width: size.width,
                height: size.height,
            },
        );
    }

    fn wait_for_gpu(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
    }

    fn render(&mut self, window: &winit::window::Window) {
        if self.window_size == Default::default() {
            return;
        }

        // Pre-compute command line for the UI button.
        // This must happen before we take mutable borrows of egui state/painter.
        let command_line = self.generate_command_line();

        let (ui_ctx, ui_state, ui_painter) = match (
            self.ui_ctx.as_ref(),
            self.ui_state.as_mut(),
            self.ui_painter.as_mut(),
        ) {
            (Some(ui_ctx), Some(ui_state), Some(ui_painter)) => (ui_ctx, ui_state, ui_painter),
            _ => return,
        };

        let surface = match self.surface.as_mut() {
            Some(surface) => surface,
            None => return,
        };

        // egui begin frame
        let raw_input = ui_state.take_egui_input(window);
        ui_ctx.begin_pass(raw_input);

        if self.ui_show {
            ui::Window::new("blade-volume-view")
                .default_open(true)
                .show(ui_ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Backend:");
                        match self.backend {
                            view::RenderBackend::Gaussian(_) => {
                                ui.label("Gaussian RT");
                            }
                            view::RenderBackend::RadFoam(_) => {
                                ui.label("RadFoam compute");
                            }
                        }
                    });

                    ui.separator();

                    // Quality controls (backend-specific)
                    ui.collapsing("Quality", |ui| match self.backend {
                        view::RenderBackend::Gaussian(ref mut backend) => {
                            ui.add(
                                ui::Slider::new(backend.min_opacity_mut(), 0.001..=0.5)
                                    .logarithmic(true)
                                    .text("Min opacity"),
                            );
                            ui.add(
                                ui::Slider::new(backend.min_transmittance_mut(), 0.001..=0.5)
                                    .logarithmic(true)
                                    .text("Min transmittance"),
                            );
                        }
                        view::RenderBackend::RadFoam(ref mut backend) => {
                            ui.add(
                                ui::Slider::new(backend.max_steps_mut(), 64..=4096)
                                    .logarithmic(true)
                                    .text("Max steps"),
                            );
                            ui.add(
                                ui::Slider::new(backend.weight_threshold_mut(), 0.0001..=0.1)
                                    .logarithmic(true)
                                    .text("Weight threshold"),
                            );

                        }
                    });

                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Debug mode:");
                        let mut enabled = self.debug_mode == view::DebugMode::ParticleDensity;
                        if ui.checkbox(&mut enabled, "Particle density").changed() {
                            self.debug_mode = if enabled {
                                view::DebugMode::ParticleDensity
                            } else {
                                view::DebugMode::Off
                            };
                            self.backend.set_debug_mode(self.debug_mode);
                        }
                    });

                    ui.separator();
                    ui.collapsing("GPU Timings", |ui| {
                        // Calculate total frame time from all timing passes
                        let total_ms: f64 = self
                            .command_encoder
                            .timings()
                            .iter()
                            .map(|(_, d)| d.as_secs_f64() * 1000.0)
                            .sum();

                        // Update history
                        if self.frame_times.len() >= FRAME_TIME_HISTORY_SIZE {
                            self.frame_times.pop_front();
                        }
                        self.frame_times.push_back(total_ms);

                        // Display individual pass timings
                        for &(ref name, value) in self.command_encoder.timings() {
                            ui.label(format!("{}: {:.3} ms", name, value.as_secs_f64() * 1000.0));
                        }

                        // Display frame time histogram
                        if !self.frame_times.is_empty() {
                            ui.separator();
                            let avg_ms: f64 = self.frame_times.iter().sum::<f64>()
                                / self.frame_times.len() as f64;
                            let max_ms = self.frame_times.iter().copied().fold(0.0, f64::max);
                            ui.label(format!("Frame: {:.2} ms avg, {:.2} ms max", avg_ms, max_ms));

                            // Simple bar visualization using egui's built-in plot
                            let height = 40.0;
                            let (response, painter) = ui.allocate_painter(
                                ui::Vec2::new(ui.available_width(), height),
                                ui::Sense::hover(),
                            );
                            let rect = response.rect;

                            // Scale to fit, max at 33ms (30fps baseline)
                            let max_display_ms = 33.0_f64.max(max_ms * 1.1);
                            let bar_width = rect.width() / self.frame_times.len() as f32;

                            for (i, &ms) in self.frame_times.iter().enumerate() {
                                let bar_height = (ms / max_display_ms * height as f64) as f32;
                                let x = rect.left() + i as f32 * bar_width;
                                let bar_rect = ui::Rect::from_min_size(
                                    ui::Pos2::new(x, rect.bottom() - bar_height),
                                    ui::Vec2::new(bar_width.max(1.0), bar_height),
                                );
                                // Color based on frame time (green < 16ms, yellow < 33ms, red > 33ms)
                                let color = if ms < 16.0 {
                                    ui::Color32::from_rgb(100, 200, 100)
                                } else if ms < 33.0 {
                                    ui::Color32::from_rgb(200, 200, 100)
                                } else {
                                    ui::Color32::from_rgb(200, 100, 100)
                                };
                                painter.rect_filled(bar_rect, 0.0, color);
                            }
                        }
                    });

                    ui.separator();

                    // Command line reproduction button
                    if ui.button("Copy command line").clicked() {
                        ui.ctx().copy_text(command_line.clone());
                        println!("{}", command_line);
                    }

                    ui.separator();
                    ui.small("Hotkeys: F1 toggle UI, Tab toggle debug, I print info");
                });
        }

        let full_output = ui_ctx.end_pass();
        let paint_jobs = ui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        let frame = surface.acquire_frame();

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        let aspect = self.window_size.width as f32 / self.window_size.height as f32;
        let camera_params = self.camera.to_params(aspect);
        self.backend.render(
            &mut self.command_encoder,
            frame.texture_view(),
            camera_params,
            self.camera.position,
            view::RenderSize {
                width: self.window_size.width,
                height: self.window_size.height,
            },
        );

        // egui textures update
        ui_painter.update_textures(
            &mut self.command_encoder,
            &full_output.textures_delta,
            &self.context,
        );

        // egui paint over the swapchain (load existing color)
        if let mut pass = self.command_encoder.render(
            "ui",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Load,
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let sd = begui::ScreenDescriptor {
                physical_size: (self.window_size.width, self.window_size.height),
                scale_factor: window.scale_factor() as f32,
            };
            ui_painter.paint(&mut pass, &paint_jobs, &sd, &self.context);
        }

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        ui_painter.after_submit(&sync_point);

        // Wait immediately after presenting to avoid swapchain semaphore reuse validation errors
        // when the swapchain rotates images faster than our timeline semaphore tracking.
        self.context.wait_for(&sync_point, !0);
        self.prev_sync_point = Some(sync_point);
    }

    fn print_info(&self) {
        println!("Camera:");
        let (roll, pitch, yaw) = self.camera.orientation.to_euler(EULER);
        println!("\tposition: {:?}", self.camera.position);
        println!(
            "\torientation: ({},{},{})",
            roll / D2R,
            pitch / D2R,
            yaw / D2R
        );
        println!("Debug Mode: {:?}", self.debug_mode);
        self.backend.print_info();
        println!("Timings:");
        for &(ref name, value) in self.command_encoder.timings() {
            println!("\t{}: {:.3} ms", name, value.as_secs_f64() * 1000.0);
        }
    }

    /// Generate command line arguments to reproduce the current view.
    fn generate_command_line(&self) -> String {
        let pos = self.camera.position;
        let (roll, pitch, yaw) = self.camera.orientation.to_euler(EULER);

        let mut args = format!(
            "cargo run -p blade-volume-view -- \"{}\" --resolution {},{} --cam-pose {:.3},{:.3},{:.3},{:.1},{:.1},{:.1}",
            self.input_file,
            self.window_size.width,
            self.window_size.height,
            pos.x, pos.y, pos.z,
            roll / D2R, pitch / D2R, yaw / D2R
        );

        // Add backend-specific parameters
        match &self.backend {
            view::RenderBackend::Gaussian(backend) => {
                args.push_str(&format!(" --min-opacity {}", backend.min_opacity()));
                args.push_str(&format!(
                    " --min-transmittance {}",
                    backend.min_transmittance()
                ));
            }
            view::RenderBackend::RadFoam(backend) => {
                args.push_str(&format!(" --max-steps {}", backend.max_steps()));
                args.push_str(&format!(
                    " --weight-threshold {}",
                    backend.weight_threshold()
                ));
            }
        }

        // Add debug flag if active
        if self.debug_mode == view::DebugMode::ParticleDensity {
            args.push_str(" --debug");
        }

        args
    }
}

fn main() {
    let args = argh::from_env::<Arguments>();
    env_logger::init();

    // Headless benchmark mode: truly no window/surface.
    if args.benchmark {
        let example = Example::init_headless(args.clone());
        example.render_headless_benchmark(&args);
        return;
    }

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut window_attributes = winit::window::Window::default_attributes();
    window_attributes.title = "blade-volume-viewer".to_string();
    if let Some(ref arg) = args.resolution {
        let res = parse_vec::<2, u32>(arg);
        window_attributes.inner_size = Some(winit::dpi::Size::Physical(res.into()));
    }
    let window = event_loop.create_window(window_attributes).unwrap();

    let mut example = Example::init_windowed(&window, args);
    let mut last_mouse_pos = [0i32; 2];
    let mut in_drag = false;
    let drag_speed = 0.01f32;

    event_loop
        .run(|event, target| {
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                winit::event::Event::AboutToWait => {
                    window.request_redraw();
                }
                winit::event::Event::WindowEvent {
                    event: ref win_event,
                    ..
                } => {
                    if let Some(ref mut ui_state) = example.ui_state {
                        let _ = ui_state.on_window_event(&window, win_event);
                    }
                    match win_event {
                        winit::event::WindowEvent::Resized(size) => {
                            example.resize(*size);
                        }
                        winit::event::WindowEvent::KeyboardInput {
                            event:
                                winit::event::KeyEvent {
                                    physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                                    state: winit::event::ElementState::Pressed,
                                    ..
                                },
                            ..
                        } => {
                            if *key_code == winit::keyboard::KeyCode::Escape {
                                target.exit();
                            }
                            if *key_code == winit::keyboard::KeyCode::F1 {
                                example.ui_show = !example.ui_show;
                            }
                            if *key_code == winit::keyboard::KeyCode::KeyI {
                                example.print_info();
                            }
                            if *key_code == winit::keyboard::KeyCode::Tab {
                                example.toggle_debug_mode();
                            }
                            let wants_keyboard = example
                                .ui_ctx
                                .as_ref()
                                .map(|c| c.wants_keyboard_input())
                                .unwrap_or(false);
                            if !wants_keyboard {
                                example.camera.on_key(*key_code, 1.0);
                            }
                        }
                        winit::event::WindowEvent::MouseInput {
                            state,
                            button: winit::event::MouseButton::Left,
                            ..
                        } => {
                            let wants_pointer = example
                                .ui_ctx
                                .as_ref()
                                .map(|c| c.wants_pointer_input())
                                .unwrap_or(false);
                            if !wants_pointer {
                                in_drag = *state == winit::event::ElementState::Pressed;
                            }
                        }
                        winit::event::WindowEvent::CursorMoved { position, .. } => {
                            let wants_pointer = example
                                .ui_ctx
                                .as_ref()
                                .map(|c| c.wants_pointer_input())
                                .unwrap_or(false);
                            if in_drag && !wants_pointer {
                                let dx = position.x as f32 - last_mouse_pos[0] as f32;
                                let dy = position.y as f32 - last_mouse_pos[1] as f32;
                                example.camera.on_mouse_drag(dx, dy, drag_speed);
                            }
                            last_mouse_pos = [position.x as i32, position.y as i32];
                        }
                        winit::event::WindowEvent::MouseWheel { delta, .. } => {
                            let wants_pointer = example
                                .ui_ctx
                                .as_ref()
                                .map(|c| c.wants_pointer_input())
                                .unwrap_or(false);
                            if !wants_pointer {
                                example.camera.on_wheel(*delta);
                            }
                        }
                        winit::event::WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        winit::event::WindowEvent::RedrawRequested => {
                            example.render(&window);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .unwrap();

    example.deinit();
}
