use blade_graphics as gpu;
use blade_volume as vol;
use std::mem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugMode {
    Off,
    ParticleDensity,
}

impl DebugMode {
    pub fn toggle(self) -> Self {
        match self {
            DebugMode::Off => DebugMode::ParticleDensity,
            DebugMode::ParticleDensity => DebugMode::Off,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct GaussianSettings {
    pub min_opacity: f32,
    pub min_transmittance: f32,
    pub debug_mode: DebugMode,
}

#[derive(Clone, Copy, Debug)]
pub struct RadFoamSettings {
    pub max_steps: u32,
    pub weight_threshold: f32,
    pub debug_mode: DebugMode,
    /// Display-referred sRGB-code-value presentation background.
    pub background_rgb: [f32; 3],
}

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
    g_gaussian_tlas: gpu::AccelerationStructure,
    g_data: gpu::BufferPiece,
}

pub struct GaussianBackend {
    point_cloud: vol::GaussianGpuCloud,
    draw_pipeline: gpu::RenderPipeline,
    params: GaussianParams,
}

impl GaussianBackend {
    pub fn min_opacity_mut(&mut self) -> &mut f32 {
        &mut self.params.min_opacity
    }

    pub fn min_transmittance_mut(&mut self) -> &mut f32 {
        &mut self.params.min_transmittance
    }

    pub fn min_opacity(&self) -> f32 {
        self.params.min_opacity
    }

    pub fn min_transmittance(&self) -> f32 {
        self.params.min_transmittance
    }

    pub fn new(
        model: &vol::PointCloudModel,
        settings: GaussianSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_format: gpu::TextureFormat,
    ) -> Self {
        let shader = {
            let raw_source = vol::shaders::GAUSSIAN;
            let source = vol::shaders::compose(raw_source);
            context.create_shader(gpu::ShaderDesc {
                source: &source,
                naga_module: None,
            })
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
            min_opacity: settings.min_opacity,
        };
        let point_cloud = vol::GaussianGpuCloud::new(model, &init_params, context, encoder);

        let params = GaussianParams {
            min_opacity: settings.min_opacity,
            min_transmittance: settings.min_transmittance,
            sh_degree: model.sh_degree as u32,
            debug_mode: settings.debug_mode as u32,
            pad: [0; 4],
        };

        Self {
            point_cloud,
            draw_pipeline,
            params,
        }
    }

    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.params.debug_mode = mode as u32;
    }

    pub fn render(
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
                    g_gaussian_tlas: self.point_cloud.tlas,
                    g_data: self.point_cloud.gauss_buf.into(),
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }

    pub fn print_info(&self) {
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

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.draw_pipeline);
        self.point_cloud.deinit(context);
    }
}

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
    g_background: Background,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Background {
    color: [f32; 3],
    pad: f32,
}

pub struct RadFoamBackend {
    point_cloud: vol::RadFoamGpuCloud,
    trace_pipeline: gpu::ComputePipeline,
    blit_pipeline: gpu::RenderPipeline,
    hdr_tex: gpu::Texture,
    hdr_view: gpu::TextureView,
    sampler: gpu::Sampler,
    trace_params: RadFoamTraceParams,
    background: Background,
}

impl RadFoamBackend {
    pub fn max_steps_mut(&mut self) -> &mut u32 {
        &mut self.trace_params.max_steps
    }

    pub fn weight_threshold_mut(&mut self) -> &mut f32 {
        &mut self.trace_params.weight_threshold
    }

    pub fn max_steps(&self) -> u32 {
        self.trace_params.max_steps
    }

    pub fn weight_threshold(&self) -> f32 {
        self.trace_params.weight_threshold
    }

    pub fn new(
        model: &vol::PointCloudModel,
        settings: RadFoamSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_format: gpu::TextureFormat,
        size: RenderSize,
    ) -> Self {
        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, size);

        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "radfoam-sampler",
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = {
            let raw_source = vol::shaders::RADFOAM;
            let source = vol::shaders::compose(raw_source);
            context.create_shader(gpu::ShaderDesc {
                source: &source,
                naga_module: None,
            })
        };
        let trace_layout = <RadFoamTraceData as gpu::ShaderData>::layout();
        let trace_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-trace",
            data_layouts: &[&trace_layout],
            compute: shader.at("trace_main"),
        });

        let blit_shader = {
            let source = vol::shaders::RADFOAM_BLIT;
            context.create_shader(gpu::ShaderDesc {
                source,
                naga_module: None,
            })
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

        let point_cloud = vol::RadFoamGpuCloud::new(model, context, encoder);

        let trace_params = RadFoamTraceParams {
            sh_degree: point_cloud.sh_degree as u32,
            weight_threshold: settings.weight_threshold,
            max_steps: settings.max_steps,
            start_point: 0,
            debug_mode: settings.debug_mode as u32,
            pad: [point_cloud.is_power_foam as u32, 0, 0, 0, 0, 0, 0],
        };
        let background = Background {
            color: settings.background_rgb,
            pad: 0.0,
        };

        Self {
            point_cloud,
            trace_pipeline,
            blit_pipeline,
            hdr_tex,
            hdr_view,
            sampler,
            trace_params,
            background,
        }
    }

    fn create_hdr_target(
        context: &gpu::Context,
        size: RenderSize,
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

    pub fn resize(&mut self, context: &gpu::Context, size: RenderSize) {
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, size);
        self.hdr_tex = hdr_tex;
        self.hdr_view = hdr_view;
    }

    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.trace_params.debug_mode = mode as u32;
    }

    pub fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera_params: vol::CameraParams,
        camera_position: glam::Vec3,
        size: RenderSize,
    ) {
        self.trace_params.start_point = self.point_cloud.containing_point(camera_position);

        encoder.init_texture(self.hdr_tex);

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

            let gx = size.width.div_ceil(8);
            let gy = size.height.div_ceil(8);
            pen.dispatch([gx, gy, 1]);
        }

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
                    g_background: self.background,
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }

    pub fn print_info(&self) {
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

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_sampler(self.sampler);
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        self.point_cloud.deinit(context);
        context.destroy_compute_pipeline(&mut self.trace_pipeline);
        context.destroy_render_pipeline(&mut self.blit_pipeline);
    }
}

// Only one backend lives at a time; boxing would just add indirection.
#[allow(clippy::large_enum_variant)]
pub enum RenderBackend {
    Gaussian(GaussianBackend),
    RadFoam(RadFoamBackend),
}

impl RenderBackend {
    pub fn new_for_model(
        model: &vol::PointCloudModel,
        gaussian: GaussianSettings,
        radfoam: RadFoamSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_format: gpu::TextureFormat,
        size: RenderSize,
    ) -> Self {
        if model.transforms.is_some() {
            RenderBackend::Gaussian(GaussianBackend::new(
                model,
                gaussian,
                context,
                encoder,
                surface_format,
            ))
        } else {
            RenderBackend::RadFoam(RadFoamBackend::new(
                model,
                radfoam,
                context,
                encoder,
                surface_format,
                size,
            ))
        }
    }

    pub fn resize(&mut self, context: &gpu::Context, size: RenderSize) {
        match *self {
            RenderBackend::Gaussian(_) => {}
            RenderBackend::RadFoam(ref mut backend) => backend.resize(context, size),
        }
    }

    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        match *self {
            RenderBackend::Gaussian(ref mut backend) => backend.set_debug_mode(mode),
            RenderBackend::RadFoam(ref mut backend) => backend.set_debug_mode(mode),
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera_params: vol::CameraParams,
        camera_position: glam::Vec3,
        size: RenderSize,
    ) {
        match *self {
            RenderBackend::Gaussian(ref backend) => {
                backend.render(encoder, frame_view, camera_params);
            }
            RenderBackend::RadFoam(ref mut backend) => {
                backend.render(encoder, frame_view, camera_params, camera_position, size);
            }
        }
    }

    pub fn print_info(&self) {
        match *self {
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

    pub fn destroy(self, context: &gpu::Context) {
        match self {
            RenderBackend::Gaussian(backend) => backend.destroy(context),
            RenderBackend::RadFoam(backend) => backend.destroy(context),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radfoam_backend_compiles_explicit_background_shader() {
        if vol::gpu::access_disabled() {
            eprintln!("skipping RadFoam shader compilation: GPU access disabled");
            return;
        }
        let Some(context) = (unsafe { gpu::Context::init(gpu::ContextDesc::default()).ok() })
        else {
            eprintln!("skipping RadFoam shader compilation: no GPU");
            return;
        };
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        ];
        let mut model = vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            points,
        };
        model.compute_adjacency_default();
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "radfoam-background-test",
            buffer_count: 1,
        });
        let backend = RadFoamBackend::new(
            &model,
            RadFoamSettings {
                max_steps: 16,
                weight_threshold: 0.001,
                debug_mode: DebugMode::Off,
                background_rgb: [1.0; 3],
            },
            &context,
            &mut encoder,
            gpu::TextureFormat::Rgba16Float,
            RenderSize {
                width: 4,
                height: 4,
            },
        );
        backend.destroy(&context);
        context.destroy_command_encoder(&mut encoder);
    }
}
