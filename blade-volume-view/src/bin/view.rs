//! Unified viewer for volumetric data with multiple rendering backends.
//!
//! NOTE: If you change any uniform structs in Rust, make sure the matching WGSL
//! structs (see `blade-volume-view/shaders/*.wgsl`) match in size/alignment.
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

#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use blade_volume as vol;
use blade_volume_view as view;
use std::{fmt, mem, str};

use blade_egui as begui;
use egui as ui;
use egui_winit as ui_winit;

const D2R: f32 = std::f32::consts::PI / 180.0;
const EULER: glam::EulerRot = glam::EulerRot::ZYX;

/// Arguments
#[derive(argh::FromArgs)]
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
    /// minimum opacity for Gaussian rendering
    #[argh(option, default = "0.01")]
    min_opacity: f32,
    /// minimum transmittance for Gaussian rendering
    #[argh(option, default = "0.01")]
    min_transmittance: f32,
    /// start in debug mode (particle density visualization)
    #[argh(switch)]
    debug: bool,
    /// start with UI visible
    #[argh(switch)]
    ui: bool,
}

/// Debug/visualization mode for rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    /// Normal rendering
    Off,
    /// Show particle density per pixel (heatmap)
    ParticleDensity,
}

impl DebugMode {
    fn toggle(self) -> Self {
        match self {
            DebugMode::Off => DebugMode::ParticleDensity,
            DebugMode::ParticleDensity => DebugMode::Off,
        }
    }
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
// Gaussian Ray-Tracing Backend
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GaussianParams {
    min_opacity: f32,
    min_transmittance: f32,
    sh_degree: u32,
    debug_mode: u32,
    pad: [u32; 4],
}

#[derive(blade_macros::ShaderData)]
struct GaussianDrawData {
    g_camera: vol::CameraParams,
    g_params: GaussianParams,
    g_acc_struct: gpu::AccelerationStructure,
    g_data: gpu::BufferPiece,
}

struct GaussianBackend {
    point_cloud: vol::PointCloud,
    draw_pipeline: gpu::RenderPipeline,
    params: GaussianParams,
}

impl GaussianBackend {
    fn new(
        model: &vol::Model,
        args: &Arguments,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_format: gpu::TextureFormat,
    ) -> Self {
        let shader = {
            let source = include_str!("../../shaders/gaussian.wgsl");
            context.create_shader(gpu::ShaderDesc { source })
        };
        assert_eq!(
            shader.get_struct_size("Gaussian"),
            mem::size_of::<vol::GaussianGpu>() as u32
        );

        let draw_layout = <GaussianDrawData as gpu::ShaderData>::layout();
        let draw_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "gaussian-main",
            data_layouts: &[&draw_layout],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            vertex_fetches: &[],
            fragment: Some(shader.at("draw_fs")),
            color_targets: &[surface_format.into()],
            depth_stencil: None,
            multisample_state: Default::default(),
        });

        let init_params = vol::InitParameters {
            min_opacity: args.min_opacity,
        };
        let point_cloud = vol::PointCloud::new(model, &init_params, context, encoder);

        let debug_mode = if args.debug {
            DebugMode::ParticleDensity
        } else {
            DebugMode::Off
        };
        let params = GaussianParams {
            min_opacity: args.min_opacity,
            min_transmittance: args.min_transmittance,
            sh_degree: model.max_sh_degree as u32,
            debug_mode: debug_mode as u32,
            pad: [0; 4],
        };

        Self {
            point_cloud,
            draw_pipeline,
            params,
        }
    }

    fn set_debug_mode(&mut self, mode: DebugMode) {
        self.params.debug_mode = mode as u32;
    }

    fn render(
        &self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera_params: vol::CameraParams,
    ) {
        if let mut pass = encoder.render(
            "gaussian-render",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut pen = pass.with(&self.draw_pipeline);
            pen.bind(
                0,
                &GaussianDrawData {
                    g_camera: camera_params,
                    g_params: self.params,
                    g_acc_struct: self.point_cloud.tlas,
                    g_data: self.point_cloud.gauss_buf.into(),
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }

    fn print_info(&self) {
        println!("Gaussian Params:");
        println!("\tmin_opacity: {}", self.params.min_opacity);
        println!("\tmin_transmittance: {}", self.params.min_transmittance);
        println!("\tsh_degree: {}", self.params.sh_degree);
        println!(
            "\tdebug_mode: {:?}",
            if self.params.debug_mode == 0 {
                DebugMode::Off
            } else {
                DebugMode::ParticleDensity
            }
        );
    }

    fn deinit(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.draw_pipeline);
        self.point_cloud.deinit(context);
    }
}

// ============================================================================
// RadFoam Compute Backend
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RadFoamTraceParams {
    sh_degree: u32,
    weight_threshold: f32,
    max_steps: u32,
    start_point: u32,
    debug_mode: u32,
    pad: [u32; 7],
}

#[derive(blade_macros::ShaderData)]
struct RadFoamTraceData {
    g_camera: vol::CameraParams,
    g_params: RadFoamTraceParams,
    g_points: gpu::BufferPiece,
    g_attributes: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct RadFoamBlitData {
    g_src: gpu::TextureView,
    g_sampler: gpu::Sampler,
}

struct RadFoamBackend {
    point_cloud: vol::RadFoamPointCloud,
    /// CPU-side KD-tree for start-point selection
    kd_tree: kiddo::KdTree<f32, 3>,
    trace_pipeline: gpu::ComputePipeline,
    blit_pipeline: gpu::RenderPipeline,
    hdr_tex: gpu::Texture,
    hdr_view: gpu::TextureView,
    sampler: gpu::Sampler,
    trace_params: RadFoamTraceParams,
}

impl RadFoamBackend {
    fn new(
        model: &vol::RadFoamModel,
        args: &Arguments,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_format: gpu::TextureFormat,
        window_size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        // Build KD-tree for start-point selection
        let mut kd_tree: kiddo::KdTree<f32, 3> = kiddo::KdTree::new();
        for (i, p) in model.points.iter().enumerate() {
            let _ = kd_tree.add(&[p.x, p.y, p.z], i as u64);
        }

        // Create HDR target
        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, window_size);

        // Sampler for blit
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "radfoam-sampler",
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });

        // Trace compute pipeline
        let shader = {
            let source = include_str!("../../shaders/radfoam.wgsl");
            context.create_shader(gpu::ShaderDesc { source })
        };
        let trace_layout = <RadFoamTraceData as gpu::ShaderData>::layout();
        let trace_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-trace",
            data_layouts: &[&trace_layout],
            compute: shader.at("trace_main"),
        });

        // Blit pipeline
        let blit_shader = {
            let source = include_str!("../../shaders/radfoam_blit.wgsl");
            context.create_shader(gpu::ShaderDesc { source })
        };
        let blit_layout = <RadFoamBlitData as gpu::ShaderData>::layout();
        let blit_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "radfoam-blit",
            data_layouts: &[&blit_layout],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            vertex: blit_shader.at("vs"),
            vertex_fetches: &[],
            fragment: Some(blit_shader.at("fs")),
            color_targets: &[surface_format.into()],
            depth_stencil: None,
            multisample_state: Default::default(),
        });

        // Upload scene data
        let point_cloud = vol::RadFoamPointCloud::new(model, context, encoder);

        let debug_mode = if args.debug {
            DebugMode::ParticleDensity
        } else {
            DebugMode::Off
        };
        let trace_params = RadFoamTraceParams {
            sh_degree: point_cloud.sh_degree as u32,
            weight_threshold: args.weight_threshold,
            max_steps: args.max_steps,
            start_point: 0,
            debug_mode: debug_mode as u32,
            pad: [0; 7],
        };

        Self {
            point_cloud,
            kd_tree,
            trace_pipeline,
            blit_pipeline,
            hdr_tex,
            hdr_view,
            sampler,
            trace_params,
        }
    }

    fn create_hdr_target(
        context: &gpu::Context,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> (gpu::Texture, gpu::TextureView) {
        let tex = context.create_texture(gpu::TextureDesc {
            name: "radfoam-hdr",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent {
                width: size.width.max(1),
                height: size.height.max(1),
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::STORAGE
                | gpu::TextureUsage::RESOURCE
                | gpu::TextureUsage::COPY,
            external: None,
        });
        let view = context.create_texture_view(
            tex,
            gpu::TextureViewDesc {
                name: "radfoam-hdr-view",
                format: gpu::TextureFormat::Rgba16Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        (tex, view)
    }

    fn resize(&mut self, context: &gpu::Context, size: winit::dpi::PhysicalSize<u32>) {
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, size);
        self.hdr_tex = hdr_tex;
        self.hdr_view = hdr_view;
    }

    fn set_debug_mode(&mut self, mode: DebugMode) {
        self.trace_params.debug_mode = mode as u32;
    }

    fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera_params: vol::CameraParams,
        camera_position: glam::Vec3,
        window_size: winit::dpi::PhysicalSize<u32>,
    ) {
        // Auto-start: pick nearest point to camera origin via KD-tree
        {
            let q = [camera_position.x, camera_position.y, camera_position.z];
            let nearest = self.kd_tree.nearest_one::<kiddo::SquaredEuclidean>(&q);
            self.trace_params.start_point = nearest.item as u32;
        }

        encoder.init_texture(self.hdr_tex);

        // Compute trace into HDR texture
        if let mut pass = encoder.compute("radfoam-trace") {
            let mut pen = pass.with(&self.trace_pipeline);
            pen.bind(
                0,
                &RadFoamTraceData {
                    g_camera: camera_params,
                    g_params: self.trace_params,
                    g_points: self.point_cloud.points(),
                    g_attributes: self.point_cloud.attributes(),
                    g_adjacency: self.point_cloud.point_adjacency(),
                    g_adjacency_offsets: self.point_cloud.point_adjacency_offsets(),
                    g_out: self.hdr_view,
                },
            );

            // Workgroup sizing matches radfoam.wgsl: @workgroup_size(8,8,1)
            let gx = (window_size.width + 7) / 8;
            let gy = (window_size.height + 7) / 8;
            pen.dispatch([gx, gy, 1]);
        }

        // Blit HDR -> swapchain with tonemap
        if let mut pass = encoder.render(
            "radfoam-present",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut pen = pass.with(&self.blit_pipeline);
            pen.bind(
                0,
                &RadFoamBlitData {
                    g_src: self.hdr_view,
                    g_sampler: self.sampler,
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }

    fn print_info(&self) {
        println!("RadFoam Trace Params:");
        println!("\tsh_degree: {}", self.trace_params.sh_degree);
        println!("\tstart_point: {}", self.trace_params.start_point);
        println!("\tmax_steps: {}", self.trace_params.max_steps);
        println!("\tweight_threshold: {}", self.trace_params.weight_threshold);
        println!(
            "\tdebug_mode: {:?}",
            if self.trace_params.debug_mode == 0 {
                DebugMode::Off
            } else {
                DebugMode::ParticleDensity
            }
        );
    }

    fn deinit(mut self, context: &gpu::Context) {
        context.destroy_sampler(self.sampler);
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        self.point_cloud.deinit(context);
        context.destroy_compute_pipeline(&mut self.trace_pipeline);
        context.destroy_render_pipeline(&mut self.blit_pipeline);
    }
}

// ============================================================================
// Unified Rendering Backend
// ============================================================================

enum RenderBackend {
    Gaussian(GaussianBackend),
    RadFoam(RadFoamBackend),
}

impl RenderBackend {
    fn resize(&mut self, context: &gpu::Context, size: winit::dpi::PhysicalSize<u32>) {
        match self {
            RenderBackend::Gaussian(_) => {
                // Gaussian backend doesn't need resize handling
            }
            RenderBackend::RadFoam(ref mut backend) => {
                backend.resize(context, size);
            }
        }
    }

    fn set_debug_mode(&mut self, mode: DebugMode) {
        match self {
            RenderBackend::Gaussian(ref mut backend) => {
                backend.set_debug_mode(mode);
            }
            RenderBackend::RadFoam(ref mut backend) => {
                backend.set_debug_mode(mode);
            }
        }
    }

    fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera: &view::ControlledCamera,
        window_size: winit::dpi::PhysicalSize<u32>,
    ) {
        let aspect = window_size.width as f32 / window_size.height as f32;
        let camera_params = camera.to_params(aspect);

        match self {
            RenderBackend::Gaussian(ref backend) => {
                backend.render(encoder, frame_view, camera_params);
            }
            RenderBackend::RadFoam(ref mut backend) => {
                backend.render(
                    encoder,
                    frame_view,
                    camera_params,
                    camera.position,
                    window_size,
                );
            }
        }
    }

    fn print_info(&self) {
        match self {
            RenderBackend::Gaussian(ref backend) => {
                println!("Backend: Gaussian Ray-Tracing");
                backend.print_info();
            }
            RenderBackend::RadFoam(ref backend) => {
                println!("Backend: RadFoam Compute");
                backend.print_info();
            }
        }
    }

    fn deinit(self, context: &gpu::Context) {
        match self {
            RenderBackend::Gaussian(backend) => backend.deinit(context),
            RenderBackend::RadFoam(backend) => backend.deinit(context),
        }
    }
}

// ============================================================================
// Main Application
// ============================================================================

struct Example {
    camera: view::ControlledCamera,
    backend: RenderBackend,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    window_size: winit::dpi::PhysicalSize<u32>,
    surface: gpu::Surface,
    context: gpu::Context,
    debug_mode: DebugMode,

    // egui overlay
    ui_ctx: ui::Context,
    ui_state: ui_winit::State,
    ui_painter: begui::GuiPainter,
    ui_show: bool,
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

    fn init(window: &winit::window::Window, args: Arguments) -> Self {
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

        // Determine volume kind: use --kind override or auto-detect
        let volume_kind = match args.kind.as_deref() {
            Some("gaussian") => vol::io::VolumeKind::Gaussian,
            Some("radfoam") => vol::io::VolumeKind::RadFoam,
            Some(other) => panic!(
                "Unknown --kind '{}', expected 'gaussian' or 'radfoam'",
                other
            ),
            None => {
                let info = vol::io::detect_format(&args.input_file);
                log::info!("Auto-detected format: {:?}", info.kind);
                info.kind
            }
        };

        // Create backend based on detected/specified kind
        let backend = match volume_kind {
            vol::io::VolumeKind::RadFoam => {
                log::info!("Loading RadFoam PLY");
                let model = vol::io::load_radfoam(&args.input_file);
                RenderBackend::RadFoam(RadFoamBackend::new(
                    &model,
                    &args,
                    &context,
                    &mut command_encoder,
                    surface_info.format,
                    window_size,
                ))
            }
            vol::io::VolumeKind::Gaussian => {
                log::info!("Loading Gaussian data");
                let model = vol::io::load_gaussian(&args.input_file);
                RenderBackend::Gaussian(GaussianBackend::new(
                    &model,
                    &args,
                    &context,
                    &mut command_encoder,
                    surface_info.format,
                ))
            }
        };

        let debug_mode = if args.debug {
            DebugMode::ParticleDensity
        } else {
            DebugMode::Off
        };

        Self {
            camera,
            backend,
            command_encoder,
            prev_sync_point: None,
            window_size,
            surface,
            context,
            debug_mode,

            ui_ctx,
            ui_state,
            ui_painter,
            ui_show,
        }
    }

    fn toggle_debug_mode(&mut self) {
        self.debug_mode = self.debug_mode.toggle();
        self.backend.set_debug_mode(self.debug_mode);
        println!("Debug mode: {:?}", self.debug_mode);
    }

    fn deinit(mut self) {
        self.wait_for_gpu();
        self.backend.deinit(&self.context);
        self.ui_painter.destroy(&self.context);
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.context.destroy_surface(&mut self.surface);
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = size;
        let config = Self::make_surface_config(size);
        self.context.reconfigure_surface(&mut self.surface, config);
        self.backend.resize(&self.context, size);
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

        // egui begin frame
        let raw_input = self.ui_state.take_egui_input(window);
        self.ui_ctx.begin_pass(raw_input);

        if self.ui_show {
            ui::Window::new("blade-volume-view")
                .default_open(true)
                .show(&self.ui_ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Backend:");
                        match self.backend {
                            RenderBackend::Gaussian(_) => {
                                ui.label("Gaussian RT");
                            }
                            RenderBackend::RadFoam(_) => {
                                ui.label("RadFoam compute");
                            }
                        }
                    });

                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Debug mode:");
                        let mut enabled = self.debug_mode == DebugMode::ParticleDensity;
                        if ui.checkbox(&mut enabled, "Particle density").changed() {
                            self.debug_mode = if enabled {
                                DebugMode::ParticleDensity
                            } else {
                                DebugMode::Off
                            };
                            self.backend.set_debug_mode(self.debug_mode);
                        }
                    });

                    ui.separator();
                    ui.label("Timings (GPU):");
                    for &(ref name, value) in self.command_encoder.timings() {
                        ui.label(format!("{}: {:.3} ms", name, value.as_secs_f64() * 1000.0));
                    }

                    ui.separator();
                    ui.label("Hotkeys: F1 toggle UI, Tab toggle debug, I print info");
                });
        }

        let full_output = self.ui_ctx.end_pass();
        let paint_jobs = self
            .ui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let frame = self.surface.acquire_frame();

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        self.backend.render(
            &mut self.command_encoder,
            frame.texture_view(),
            &self.camera,
            self.window_size,
        );

        // egui textures update
        self.ui_painter.update_textures(
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
            self.ui_painter
                .paint(&mut pass, &paint_jobs, &sd, &self.context);
        }

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.ui_painter.after_submit(&sync_point);

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
}

fn main() {
    let args = argh::from_env::<Arguments>();
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut window_attributes = winit::window::Window::default_attributes();
    window_attributes.title = "blade-volume-viewer".to_string();
    if let Some(ref arg) = args.resolution {
        let res = parse_vec::<2, u32>(arg);
        window_attributes.inner_size = Some(winit::dpi::Size::Physical(res.into()));
    }
    let window = event_loop.create_window(window_attributes).unwrap();

    let mut example = Example::init(&window, args);
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
                    let _ = example.ui_state.on_window_event(&window, win_event);
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
                            let wants_keyboard = example.ui_ctx.wants_keyboard_input();
                            if !wants_keyboard {
                                example.camera.on_key(*key_code, 1.0);
                            }
                        }
                        winit::event::WindowEvent::MouseInput {
                            state,
                            button: winit::event::MouseButton::Left,
                            ..
                        } => {
                            let wants_pointer = example.ui_ctx.wants_pointer_input();
                            if !wants_pointer {
                                in_drag = *state == winit::event::ElementState::Pressed;
                            }
                        }
                        winit::event::WindowEvent::CursorMoved { position, .. } => {
                            let wants_pointer = example.ui_ctx.wants_pointer_input();
                            if in_drag && !wants_pointer {
                                let dx = position.x as f32 - last_mouse_pos[0] as f32;
                                let dy = position.y as f32 - last_mouse_pos[1] as f32;
                                example.camera.on_mouse_drag(dx, dy, drag_speed);
                            }
                            last_mouse_pos = [position.x as i32, position.y as i32];
                        }
                        winit::event::WindowEvent::MouseWheel { delta, .. } => {
                            let wants_pointer = example.ui_ctx.wants_pointer_input();
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
