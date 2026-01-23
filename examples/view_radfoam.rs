#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use blade_volume as vol;
use std::{f32, fmt, str};

use kiddo::KdTree;
use kiddo::SquaredEuclidean;

const D2R: f32 = f32::consts::PI / 180.0;
const EULER: glam::EulerRot = glam::EulerRot::ZYX;
const MAX_FLY_SPEED: f32 = 1_000_000.0;

/// arguments
#[derive(argh::FromArgs)]
struct Arguments {
    /// input PLY file path exported by upstream RadFoam `save_ply()`
    #[argh(positional)]
    input_file: String,
    /// target resolution (e.g. 1920,1080)
    #[argh(option)]
    resolution: Option<String>,
    /// camera postion and orientation (as Euler degrees): x,y,z,roll,pitch,yaw
    #[argh(option)]
    cam_pose: Option<String>,
    /// radfoam start point index (single cell for all rays) - MVP (ignored when auto-start is enabled)
    #[argh(option)]
    start_point: Option<u32>,
    /// max traversal steps
    #[argh(option, default = "1024")]
    max_steps: u32,
    /// stop when transmittance <= threshold
    #[argh(option, default = "0.001")]
    weight_threshold: f32,
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

#[derive(Default)]
pub struct ControlledCamera {
    pub position: glam::Vec3,
    pub orientation: glam::Quat,
    pub fov_y: f32,
    pub depth: f32,
    pub fly_speed: f32,
}

impl ControlledCamera {
    pub fn move_by(&mut self, offset: glam::Vec3) {
        self.position += self.orientation * offset;
    }

    pub fn rotate_z_by(&mut self, angle: f32) {
        self.orientation *= glam::Quat::from_rotation_z(angle);
    }

    pub fn on_key(&mut self, code: winit::keyboard::KeyCode, delta: f32) -> bool {
        use winit::keyboard::KeyCode as Kc;

        let move_offset = self.fly_speed * delta;
        let rotate_offset_z = 1000.0 * delta;
        match code {
            Kc::KeyW => self.move_by(glam::Vec3::new(0.0, 0.0, move_offset)),
            Kc::KeyS => self.move_by(glam::Vec3::new(0.0, 0.0, -move_offset)),
            Kc::KeyA => self.move_by(glam::Vec3::new(-move_offset, 0.0, 0.0)),
            Kc::KeyD => self.move_by(glam::Vec3::new(move_offset, 0.0, 0.0)),
            Kc::KeyZ => self.move_by(glam::Vec3::new(0.0, -move_offset, 0.0)),
            Kc::KeyX => self.move_by(glam::Vec3::new(0.0, move_offset, 0.0)),
            Kc::KeyQ => self.rotate_z_by(rotate_offset_z),
            Kc::KeyE => self.rotate_z_by(-rotate_offset_z),
            _ => return false,
        }
        true
    }

    pub fn on_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
        let shift = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, lines) => lines,
            winit::event::MouseScrollDelta::PixelDelta(position) => position.y as f32,
        };
        self.fly_speed = (self.fly_speed * shift.exp()).clamp(1.0, MAX_FLY_SPEED);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CameraParams {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 2],
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct TraceParams {
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

#[derive(blade_macros::ShaderData)]
struct BlitData {
    g_src: gpu::TextureView,
    g_sampler: gpu::Sampler,
}

struct Example {
    camera: ControlledCamera,

    trace_pipeline: gpu::ComputePipeline,
    blit_pipeline: gpu::RenderPipeline,

    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,

    window_size: winit::dpi::PhysicalSize<u32>,
    surface: gpu::Surface,
    context: gpu::Context,

    radfoam: vol::RadFoamPointCloud,

    // CPU-side acceleration for start-point selection (MVP: KD-tree via kiddo)
    radfoam_kd: KdTree<f32, 3>,

    // HDR intermediate
    hdr_tex: gpu::Texture,
    hdr_view: gpu::TextureView,
    sampler: gpu::Sampler,

    // runtime params
    trace_params: TraceParams,
}

impl Example {
    fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
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

    fn init(window: &winit::window::Window, args: Arguments) -> Self {
        let mut camera = ControlledCamera {
            depth: 10_000.0,
            fov_y: 1.0,
            fly_speed: 1.0,
            ..Default::default()
        };
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

        // Create HDR target
        let (hdr_tex, hdr_view) = Self::create_hdr_target(&context, window_size);

        // Sampler for blit
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "radfoam-sampler",
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });

        // Shader module (create pipelines before loading assets to surface shader issues early)
        let shader = {
            let source = std::fs::read_to_string("examples/radfoam.wgsl").unwrap();
            context.create_shader(gpu::ShaderDesc { source: &source })
        };

        // Trace compute pipeline
        let trace_layout = <TraceData as gpu::ShaderData>::layout();
        let trace_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-trace",
            data_layouts: &[&trace_layout],
            compute: shader.at("trace_main"),
        });

        // Blit shader (inline WGSL)
        let blit_shader = {
            let source = r#"
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle
    var out: VSOut;
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0)
    );
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = uv[vi];
    return out;
}

var g_src: texture_2d<f32>;
var g_sampler: sampler;

fn tonemap_reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (1.0 + x);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    // The compute pass writes RGBA into the HDR texture:
    //   rgb = accumulated radiance
    //   a   = opacity = 1 - transmittance
    //
    // Composite an explicit sky/background using alpha to avoid black patches
    // when rays miss / terminate early.
    let sample = textureSample(g_src, g_sampler, in.uv);
    let hdr_rgb = sample.xyz;
    let alpha = clamp(sample.w, 0.0, 1.0);

    // Simple sky-like background (linear space)
    let sky = vec3<f32>(0.65, 0.75, 0.90);

    // "Over" compositing with premultiplied assumption:
    // output = rgb + (1 - alpha) * bg
    let hdr_composited = hdr_rgb + (1.0 - alpha) * sky;

    let ldr = tonemap_reinhard(max(hdr_composited, vec3<f32>(0.0)));
    return vec4<f32>(ldr, 1.0);
}
"#;
            context.create_shader(gpu::ShaderDesc { source })
        };

        let blit_layout = <BlitData as gpu::ShaderData>::layout();
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
            color_targets: &[surface_info.format.into()],
            depth_stencil: None,
            multisample_state: Default::default(),
        });

        // Load scene (after pipelines are created)
        log::info!("Loading RadFoam PLY");
        let model = vol::io::load_radfoam_ply(&args.input_file);

        // Build KD-tree for start-point selection
        // NOTE: this is a CPU-only helper, independent of the GPU traversal.
        let mut radfoam_kd: KdTree<f32, 3> = KdTree::new();
        for (i, p) in model.points.iter().enumerate() {
            // kiddo's `KdTree` type alias stores items as `u64`.
            let _ = radfoam_kd.add(&[p.x, p.y, p.z], i as u64);
        }

        // Create command encoder
        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "radfoam",
            buffer_count: 2,
        });

        // Upload scene buffers
        let radfoam = vol::RadFoamPointCloud::new(&model, &context, &mut command_encoder);

        // Runtime trace params
        // NOTE: start_point will be updated every frame from camera origin (auto-start).
        let start_point = args.start_point.unwrap_or(0);
        let trace_params = TraceParams {
            sh_degree: radfoam.sh_degree as u32,
            weight_threshold: args.weight_threshold,
            max_steps: args.max_steps,
            start_point,
        };

        Self {
            camera,
            trace_pipeline,
            blit_pipeline,
            command_encoder,
            prev_sync_point: None,
            window_size,
            surface,
            context,
            radfoam,
            radfoam_kd,
            hdr_tex,
            hdr_view,
            sampler,
            trace_params,
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = size;
        let config = Self::make_surface_config(size);
        self.context.reconfigure_surface(&mut self.surface, config);

        // Recreate HDR target
        self.context.destroy_texture_view(self.hdr_view);
        self.context.destroy_texture(self.hdr_tex);
        let (hdr_tex, hdr_view) = Self::create_hdr_target(&self.context, size);
        self.hdr_tex = hdr_tex;
        self.hdr_view = hdr_view;
    }

    fn wait_for_gpu(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
    }

    fn render(&mut self) {
        if self.window_size == Default::default() {
            return;
        }

        let frame = self.surface.acquire_frame();
        let aspect = self.window_size.width as f32 / self.window_size.height as f32;

        // Auto-start: pick nearest point to camera origin via KD-tree (kiddo).
        // This matches upstream behavior conceptually (they do NN via an AABB tree),
        // but is still a CPU helper.
        {
            let q = [
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
            ];
            let nearest = self.radfoam_kd.nearest_one::<SquaredEuclidean>(&q);
            // kiddo returns NearestNeighbour { item: u64, distance: f32 } for this KdTree alias
            self.trace_params.start_point = nearest.item as u32;
        }

        self.command_encoder.start();

        // Ensure textures are in GENERAL layout
        self.command_encoder.init_texture(frame.texture());
        self.command_encoder.init_texture(self.hdr_tex);

        // Compute trace into HDR texture
        if let mut pass = self.command_encoder.compute("radfoam-trace") {
            let mut pen = pass.with(&self.trace_pipeline);
            pen.bind(
                0,
                &TraceData {
                    g_camera: CameraParams {
                        cam_position: self.camera.position.into(),
                        depth: self.camera.depth,
                        cam_orientation: self.camera.orientation.into(),
                        fov: [aspect * self.camera.fov_y, self.camera.fov_y],
                        pad: [0; 2],
                    },
                    g_params: self.trace_params,
                    g_points: self.radfoam.points(),
                    g_attributes: self.radfoam.attributes(),
                    g_adjacency: self.radfoam.point_adjacency(),
                    g_adjacency_offsets: self.radfoam.point_adjacency_offsets(),
                    g_out: self.hdr_view,
                },
            );

            // Workgroup sizing matches radfoam.wgsl: @workgroup_size(8,8,1)
            let gx = (self.window_size.width + 7) / 8;
            let gy = (self.window_size.height + 7) / 8;
            pen.dispatch([gx, gy, 1]);
        }

        // Blit HDR -> swapchain with tonemap
        if let mut pass = self.command_encoder.render(
            "radfoam-present",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut pen = pass.with(&self.blit_pipeline);
            pen.bind(
                0,
                &BlitData {
                    g_src: self.hdr_view,
                    g_sampler: self.sampler,
                },
            );
            pen.draw(0, 3, 0, 1);
        }

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);

        // Wait immediately after presenting to avoid swapchain semaphore reuse validation errors
        // when the swapchain rotates images faster than our timeline semaphore tracking.
        self.context.wait_for(&sync_point, !0);
        self.prev_sync_point = Some(sync_point);
    }

    fn deinit(&mut self) {
        self.wait_for_gpu();

        self.context.destroy_sampler(self.sampler);
        self.context.destroy_texture_view(self.hdr_view);
        self.context.destroy_texture(self.hdr_tex);

        self.radfoam.deinit(&self.context);

        self.context
            .destroy_compute_pipeline(&mut self.trace_pipeline);
        self.context
            .destroy_render_pipeline(&mut self.blit_pipeline);
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.context.destroy_surface(&mut self.surface);
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
        println!("Trace Params:");
        println!("\tsh_degree: {}", self.trace_params.sh_degree);
        println!("\tstart_point: {}", self.trace_params.start_point);
        println!("\tmax_steps: {}", self.trace_params.max_steps);
        println!("\tweight_threshold: {}", self.trace_params.weight_threshold);
        println!("Timings:");
        for &(ref name, value) in self.command_encoder.timings() {
            println!("\t{}: {} ms", name, value.as_millis());
        }
    }
}

fn main() {
    let args = argh::from_env::<Arguments>();
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut window_attributes = winit::window::Window::default_attributes();
    window_attributes.title = "blade-radfoam-viewer".to_string();
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
                winit::event::Event::WindowEvent { event, .. } => match event {
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
                            target.exit();
                        }
                        if key_code == winit::keyboard::KeyCode::KeyI {
                            example.print_info();
                        }
                        example.camera.on_key(key_code, 1.0);
                    }
                    winit::event::WindowEvent::MouseInput {
                        state,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        in_drag = state == winit::event::ElementState::Pressed;
                    }
                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        if in_drag {
                            let prev = example.camera.orientation;

                            // Mouse deltas (pixels)
                            let dx = position.x as f32 - last_mouse_pos[0] as f32;
                            let dy = position.y as f32 - last_mouse_pos[1] as f32;

                            // Yaw around global up (assume +Y is world up)
                            let world_up = glam::Vec3::Y;
                            let yaw = glam::Quat::from_axis_angle(world_up, dx * drag_speed);

                            // Pitch around camera's local right axis, with sign chosen to avoid inverted look
                            let right = prev * glam::Vec3::X;
                            let pitch = glam::Quat::from_axis_angle(right, dy * drag_speed);

                            example.camera.orientation = yaw * pitch * prev;
                        }
                        last_mouse_pos = [position.x as i32, position.y as i32];
                    }
                    winit::event::WindowEvent::MouseWheel { delta, .. } => {
                        example.camera.on_wheel(delta);
                    }
                    winit::event::WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        example.render();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();

    example.deinit();
}
