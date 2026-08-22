//! Unified viewer for volumetric data with multiple rendering backends.
//!
//! NOTE: If you change any uniform structs in Rust, make sure the matching WGSL
//! structs (see `blade-volume/shaders/*.wgsl`) match in size/alignment.
//! A mismatch can cause validation asserts or GPU crashes.
//!
//! Usage:
//! `cargo run -p blade-volume-view -- <input_file> [options]`
//!
//! The rendering method is automatically detected based on file contents:
//!   - PLY files are examined to detect RadFoam vs Gaussian format
//!   - SPZ files are always Gaussian format
//!   - `.surfel` files hold relightable surfels and are lit at render time
//!   - Use --kind to override auto-detection
//!
//! A `.surfel` asset carries materials and no light, so one has to be supplied:
//! `--environment` takes a comma-separated list of float environment planes,
//! and with none given the viewer makes a sky and moves the sun around it. `L`
//! cycles through them, which is the thing this representation exists to do.
//!
//! Controls:
//!   WASD/ZX - Move camera
//!   Q/E - Roll camera
//!   Mouse drag - Look around
//!   Mouse wheel - Adjust fly speed
//!   I - Print info (camera pose, timings)
//!   Tab - Toggle debug mode (particle density visualization)
//!   L - Next environment (relightable surfels)
//!   F1 - Toggle UI overlay
//!   Escape - Exit

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
    /// override format detection: "gaussian", "radfoam" or "surfel"
    #[argh(option)]
    kind: Option<String>,
    /// environment maps to light relightable point clouds with, comma separated.
    /// Float planes as written by blade's relight_data. Without this the
    /// viewer builds a sky and moves the sun around it.
    #[argh(option)]
    environment: Option<String>,
    /// multiply relightable radiance by this before the display curve.
    /// Without it the viewer picks one from the environment's average
    /// radiance, which is in whatever units the capture used.
    #[argh(option)]
    exposure: Option<f32>,
    /// rays per surfel for shadowing and one bounce (surfel only).
    /// Zero keeps the analytic path, which is noise free and seven times
    /// faster, and measures closer to a path traced reference than shadows
    /// without a bounce do.
    #[argh(option, default = "0")]
    diffuse_samples: u32,
    /// equirectangular width the specular ladder is prefiltered at, per
    /// environment. Seconds of CPU work, once per light.
    #[argh(option, default = "256")]
    specular_size: usize,
    /// which environment to open under, by name or index (relightable only).
    /// `L` cycles through the rest.
    #[argh(option)]
    light: Option<String>,
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
    /// composite RadFoam on white instead of the default black background
    #[argh(switch)]
    white_background: bool,
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

/// The environments a relightable asset can be put under.
///
/// A list of float planes if the caller has any, and otherwise a sky with the
/// sun moved around it. The synthetic set is not a substitute for measured
/// light — it exists so that a converted asset can be looked at, and relit,
/// without also having to produce a dataset first.
fn load_environments(argument: Option<&str>) -> Vec<view::NamedEnvironment> {
    if let Some(list) = argument {
        let loaded = list
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                let path = std::path::Path::new(entry);
                let name = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or(entry);
                view::NamedEnvironment::new(name, vol::io::load_environment(path))
            })
            .collect::<Vec<_>>();
        if !loaded.is_empty() {
            return loaded;
        }
    }
    default_skies()
}

/// An exposure that puts an ordinary surface somewhere near the middle.
///
/// Radiance arrives in whatever units the environment was captured in — the
/// reference datasets here run from a tenth to a hundred and forty within one
/// map — so a fixed multiplier renders one of them and blows out or blackens
/// the rest. Aim the photographic key at 18 %, which is what a light meter
/// does and is wrong in exactly the cases a light meter is wrong.
fn exposure_for(entry: &view::NamedEnvironment) -> f32 {
    let key = entry.environment.key_luminance();
    if key <= 0.0 {
        return 1.0;
    }
    let exposure = 0.18 / key;
    log::info!("exposure {exposure:.3} from a key luminance of {key:.4}");
    exposure.clamp(1.0e-3, 1.0e3)
}

/// Find an environment by name, or by position if the name is a number.
///
/// A name that matches nothing is a mistake worth stopping for: silently
/// opening under the wrong light would look like the renderer ignoring the
/// request rather than like the request being wrong.
fn resolve_light(wanted: &str, environments: &[view::NamedEnvironment]) -> usize {
    if let Some(index) = environments.iter().position(|entry| entry.name == wanted) {
        return index;
    }
    if let Ok(index) = wanted.parse::<usize>() {
        if index < environments.len() {
            return index;
        }
    }
    let names = environments
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    panic!("no environment called '{wanted}'; loaded: {names}");
}

fn default_skies() -> Vec<view::NamedEnvironment> {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 256;
    // Carrying about half the energy in the sky, which is what makes moving
    // it visible. A tenth of that is still a sun and still physical, and the
    // difference between one side of the car and the other is then small
    // enough to argue about — the nine-coefficient irradiance basis smooths a
    // small source out, so a timid one arrives as a gentle gradient.
    const SUN: [f32; 3] = [420.0, 390.0, 330.0];
    const ELEVATION: f32 = 0.6;

    let sun_at = |azimuth: f32| {
        glam::Vec3::new(
            ELEVATION.cos() * azimuth.sin(),
            ELEVATION.sin(),
            ELEVATION.cos() * azimuth.cos(),
        )
    };
    let quarter = std::f32::consts::FRAC_PI_2;
    vec![
        view::NamedEnvironment::new(
            "sun-east",
            vol::relight::Environment::sky(sun_at(quarter), SUN, 0.05, WIDTH, HEIGHT),
        ),
        view::NamedEnvironment::new(
            "sun-front",
            vol::relight::Environment::sky(sun_at(0.0), SUN, 0.05, WIDTH, HEIGHT),
        ),
        view::NamedEnvironment::new(
            "sun-west",
            vol::relight::Environment::sky(sun_at(-quarter), SUN, 0.05, WIDTH, HEIGHT),
        ),
        // No sun at all: everything is lit by the sky, which is what removes
        // every shadow and every highlight at once.
        view::NamedEnvironment::new(
            "overcast",
            vol::relight::Environment::sky(glam::Vec3::ZERO, SUN, 0.0, WIDTH, HEIGHT),
        ),
    ]
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
    surface: gpu::Surface,
    context: gpu::Context,
    debug_mode: view::DebugMode,

    // For command line reproduction
    input_file: String,

    // egui overlay
    ui_ctx: ui::Context,
    ui_state: ui_winit::State,
    ui_painter: begui::GuiPainter,
    ui_show: bool,

    // Frame timing history (milliseconds)
    frame_times: VecDeque<f64>,
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
        assert!(
            !vol::gpu::access_disabled(),
            "GPU access disabled by BLADE_VOLUME_DISABLE_GPU"
        );
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
                ray_tracing: true,
                xr: None,
                device_id: None,
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
            manual_barriers: false,
        });

        let size = view::RenderSize {
            width: window_size.width,
            height: window_size.height,
        };

        // Relightable surfels are not a point cloud and do not go through the
        // cloud loader at all: they carry a material and no radiance, and need
        // a light supplied before they can be drawn.
        let relightable = match args.kind.as_deref() {
            Some("surfel") => true,
            Some(_) => false,
            None => vol::io::try_detect_format(&args.input_file)
                .is_ok_and(|info| info.kind == vol::io::VolumeKind::Surfel),
        };
        if relightable {
            let model = vol::io::load_relight(std::path::Path::new(&args.input_file));
            log::info!(
                "Loaded {} surfels over {} materials",
                model.surfels.len(),
                model.materials.len()
            );
            // Assets arrive at whatever scale their author used, so the
            // default camera is inside some of them and nowhere near others.
            if args.cam_pose.is_none() {
                if let Some((min, max)) = model.bounds() {
                    camera.frame_bounds(min, max);
                }
            }
            let environments = load_environments(args.environment.as_deref());
            let initial = match args.light {
                Some(ref wanted) => resolve_light(wanted, &environments),
                None => 0,
            };
            // Chosen once, from the light the viewer opens under, and left
            // alone when the light changes: an exposure that followed the
            // environment would cancel out the very difference that switching
            // between them is meant to show.
            let exposure = args
                .exposure
                .unwrap_or_else(|| exposure_for(&environments[initial]));
            let backend = view::RenderBackend::Relight(view::RelightBackend::new(
                &model,
                environments,
                view::RelightSettings {
                    diffuse_samples: args.diffuse_samples,
                    show_environment: true,
                    exposure,
                    specular_width: args.specular_size,
                    initial_environment: initial,
                },
                &context,
                &mut command_encoder,
                surface_info.format,
                size,
            ));
            return Self {
                camera,
                backend,
                command_encoder,
                prev_sync_point: None,
                window_size,
                surface,
                context,
                debug_mode: view::DebugMode::Off,
                input_file: args.input_file,

                ui_ctx,
                ui_state,
                ui_painter,
                ui_show,

                frame_times: VecDeque::with_capacity(FRAME_TIME_HISTORY_SIZE),
            };
        }

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

        if model
            .transforms
            .as_ref()
            .and_then(|transforms| transforms.pbr.as_ref())
            .is_some()
        {
            let environments = load_environments(args.environment.as_deref());
            let initial = match args.light {
                Some(ref wanted) => resolve_light(wanted, &environments),
                None => 0,
            };
            let exposure = args
                .exposure
                .unwrap_or_else(|| exposure_for(&environments[initial]));
            let backend = view::RenderBackend::Relight(view::RelightBackend::new_gaussian(
                &model,
                environments,
                view::RelightSettings {
                    diffuse_samples: args.diffuse_samples,
                    show_environment: true,
                    exposure,
                    specular_width: args.specular_size,
                    initial_environment: initial,
                },
                &context,
                &mut command_encoder,
                surface_info.format,
                size,
            ));
            return Self {
                camera,
                backend,
                command_encoder,
                prev_sync_point: None,
                window_size,
                surface,
                context,
                debug_mode: view::DebugMode::Off,
                input_file: args.input_file,

                ui_ctx,
                ui_state,
                ui_painter,
                ui_show,

                frame_times: VecDeque::with_capacity(FRAME_TIME_HISTORY_SIZE),
            };
        }

        let debug_mode = if args.debug {
            view::DebugMode::ParticleDensity
        } else {
            view::DebugMode::Off
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
            background_rgb: if args.white_background {
                [1.0; 3]
            } else {
                [0.0; 3]
            },
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
            surface,
            context,
            debug_mode,
            input_file: args.input_file,

            ui_ctx,
            ui_state,
            ui_painter,
            ui_show,

            frame_times: VecDeque::with_capacity(FRAME_TIME_HISTORY_SIZE),
        }
    }

    /// Put the model under one of the loaded environments.
    ///
    /// Nothing but the light is rebuilt, so this is the cheap operation the
    /// representation is for — except the first time a given environment is
    /// used, when prefiltering it costs a second or two of CPU.
    fn set_environment(&mut self, index: usize) {
        if let view::RenderBackend::Relight(ref mut backend) = self.backend {
            backend.set_environment(index, &self.context, &mut self.command_encoder);
        }
    }

    fn next_environment(&mut self) {
        if let view::RenderBackend::Relight(ref mut backend) = self.backend {
            backend.next_environment(&self.context, &mut self.command_encoder);
            println!(
                "Light: {}",
                backend
                    .environment_names()
                    .nth(backend.current_environment())
                    .unwrap_or("?")
            );
        }
    }

    fn toggle_debug_mode(&mut self) {
        self.debug_mode = self.debug_mode.toggle();
        self.backend.set_debug_mode(self.debug_mode);
        println!("Debug mode: {:?}", self.debug_mode);
    }

    fn deinit(mut self) {
        self.wait_for_gpu();
        self.backend.destroy(&self.context);
        self.ui_painter.destroy(&self.context);
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.context.destroy_surface(&mut self.surface);
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = size;
        let config = Self::make_surface_config(size);
        self.context.reconfigure_surface(&mut self.surface, config);
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
            let _ = self.context.wait_for(&sp, !0);
        }
    }

    fn render(&mut self, window: &winit::window::Window) {
        if self.window_size == Default::default() {
            return;
        }

        // egui begin frame
        let raw_input = self.ui_state.take_egui_input(window);
        self.ui_ctx.begin_pass(raw_input);

        // Pre-compute command line for the UI button (to avoid borrow conflicts)
        let command_line = self.generate_command_line();
        // Applied after the pass is closed: switching a light submits its own
        // uploads and waits on them, which cannot happen mid-frame.
        let mut chosen_environment: Option<usize> = None;

        if self.ui_show {
            ui::Window::new("blade-volume-view")
                .default_open(true)
                .show(&self.ui_ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Backend:");
                        match self.backend {
                            view::RenderBackend::Gaussian(_) => {
                                ui.label("Gaussian RT");
                            }
                            view::RenderBackend::RadFoam(_) => {
                                ui.label("RadFoam compute");
                            }
                            view::RenderBackend::Relight(_) => {
                                ui.label("Relightable point cloud");
                            }
                        }
                    });

                    // The light, which for this backend is a control rather
                    // than a property of the asset. Switching to one that has
                    // not been prefiltered yet blocks for a second or two.
                    if let view::RenderBackend::Relight(ref backend) = self.backend {
                        let names = backend.environment_names().collect::<Vec<_>>();
                        let current = backend.current_environment();
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Light:");
                            for (index, name) in names.iter().enumerate() {
                                if ui.selectable_label(index == current, *name).clicked() {
                                    chosen_environment = Some(index);
                                }
                            }
                        });
                    }

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
                        view::RenderBackend::Relight(ref mut backend) => {
                            ui.add(
                                ui::Slider::new(backend.exposure_mut(), 0.01..=100.0)
                                    .logarithmic(true)
                                    .text("Exposure"),
                            );
                            if backend.supports_shadow_rays() {
                                ui.add(
                                    ui::Slider::new(backend.diffuse_samples_mut(), 0..=64)
                                        .text("Shadow rays"),
                                );
                                // Said plainly, because the slider above looks
                                // like a quality control and is not one: it buys
                                // shadows and one bounce, at seven times the cost,
                                // and against a path traced reference it scores
                                // worse than leaving both out.
                                ui.small(
                                    "0 is analytic: no shadows, no noise, and closer to a path trace",
                                );
                            } else {
                                ui.small("Volumetric Gaussians currently use analytic lighting");
                            }
                            let mut show = backend.show_environment();
                            if ui.checkbox(&mut show, "Show the environment").changed() {
                                backend.set_show_environment(show);
                            }
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
                            .map(|&(_, d)| d.as_secs_f64() * 1000.0)
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

        let full_output = self.ui_ctx.end_pass();
        let paint_jobs = self
            .ui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        if let Some(index) = chosen_environment {
            self.set_environment(index);
        }

        let frame = self.surface.acquire_frame();

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
        let _ = self.context.wait_for(&sync_point, !0);
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
        match self.backend {
            view::RenderBackend::Gaussian(ref backend) => {
                args.push_str(&format!(" --min-opacity {}", backend.min_opacity()));
                args.push_str(&format!(
                    " --min-transmittance {}",
                    backend.min_transmittance()
                ));
            }
            view::RenderBackend::RadFoam(ref backend) => {
                args.push_str(&format!(" --max-steps {}", backend.max_steps()));
                args.push_str(&format!(
                    " --weight-threshold {}",
                    backend.weight_threshold()
                ));
            }
            view::RenderBackend::Relight(ref backend) => {
                // Including the light: a reproduction that came back under a
                // different one would not be a reproduction.
                args.push_str(&format!(" --light {}", backend.current_environment_name()));
                args.push_str(&format!(" --exposure {}", backend.exposure()));
                args.push_str(&format!(" --diffuse-samples {}", backend.diffuse_samples()));
            }
        }

        // Add debug flag if active
        if self.debug_mode == view::DebugMode::ParticleDensity {
            args.push_str(" --debug");
        }

        args
    }
}

struct App {
    args: Option<Arguments>,
    window: Option<winit::window::Window>,
    example: Option<Example>,
    last_mouse_pos: [i32; 2],
    in_drag: bool,
}

const DRAG_SPEED: f32 = 0.01;

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let args = self.args.take().expect("resumed called twice without args");
        let mut window_attributes = winit::window::Window::default_attributes();
        window_attributes.title = "blade-volume-viewer".to_string();
        if let Some(ref arg) = args.resolution {
            let res = parse_vec::<2, u32>(arg);
            window_attributes.inner_size = Some(winit::dpi::Size::Physical(res.into()));
        }
        let window = event_loop.create_window(window_attributes).unwrap();
        let example = Example::init(&window, args);
        self.window = Some(window);
        self.example = Some(example);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(ref window) = self.window else {
            return;
        };
        let Some(ref mut example) = self.example else {
            return;
        };
        let _ = example.ui_state.on_window_event(window, &event);
        match event {
            winit::event::WindowEvent::Resized(size) => {
                example.resize(size);
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
                if key_code == winit::keyboard::KeyCode::Escape {
                    event_loop.exit();
                }
                if key_code == winit::keyboard::KeyCode::F1 {
                    example.ui_show = !example.ui_show;
                }
                if key_code == winit::keyboard::KeyCode::KeyI {
                    example.print_info();
                }
                if key_code == winit::keyboard::KeyCode::Tab {
                    example.toggle_debug_mode();
                }
                if key_code == winit::keyboard::KeyCode::KeyL {
                    example.next_environment();
                }
                if !example.ui_ctx.egui_wants_keyboard_input() {
                    example.camera.on_key(key_code, 1.0);
                }
            }
            winit::event::WindowEvent::MouseInput {
                state,
                button: winit::event::MouseButton::Left,
                ..
            } if !example.ui_ctx.egui_wants_pointer_input() => {
                self.in_drag = state == winit::event::ElementState::Pressed;
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                if self.in_drag && !example.ui_ctx.egui_wants_pointer_input() {
                    let dx = position.x as f32 - self.last_mouse_pos[0] as f32;
                    let dy = position.y as f32 - self.last_mouse_pos[1] as f32;
                    example.camera.on_mouse_drag(dx, dy, DRAG_SPEED);
                }
                self.last_mouse_pos = [position.x as i32, position.y as i32];
            }
            winit::event::WindowEvent::MouseWheel { delta, .. }
                if !example.ui_ctx.egui_wants_pointer_input() =>
            {
                example.camera.on_wheel(delta);
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                example.render(window);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let args = argh::from_env::<Arguments>();
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App {
        args: Some(args),
        window: None,
        example: None,
        last_mouse_pos: [0i32; 2],
        in_drag: false,
    };
    event_loop.run_app(&mut app).unwrap();

    if let Some(example) = app.example {
        example.deinit();
    }
}
