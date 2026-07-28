//! Interactive rendering of relightable surfels.
//!
//! The other backends draw what a point looked like when it was captured, and
//! there is nothing to interact with beyond the camera. This one draws what a
//! surface is made of under a light that is supplied at render time, so the
//! light is a control rather than a property of the asset — which is the whole
//! reason the representation exists, and is not visible in a still image.
//!
//! Two things cost real time and are handled here rather than hidden:
//!
//! - **Prefiltering an environment is seconds of CPU work.** It happens once
//!   per environment, on first use, and is cached, so the first switch to a
//!   light stutters and every later one does not.
//! - **Tone mapping is not optional.** The tracer produces linear radiance
//!   with a sun in it, which can be hundreds of times white. What reaches the
//!   surface goes through the same curve `relight_quality` scores through.

use blade_graphics as gpu;
use blade_volume as vol;

use crate::RenderSize;

/// An environment under a name, so the viewer can say which light it is under.
pub struct NamedEnvironment {
    pub name: String,
    pub environment: vol::relight::Environment,
    /// Prefiltered on first use. Seconds of work, and most sessions look at
    /// two or three of the lights they were given.
    specular: Option<vol::relight::SpecularEnvironment>,
}

impl NamedEnvironment {
    pub fn new(name: impl Into<String>, environment: vol::relight::Environment) -> Self {
        Self {
            name: name.into(),
            environment,
            specular: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelightSettings {
    /// Rays per shading point for shadowing and the bounce that comes with it.
    ///
    /// Zero is the analytic path: no shadows, no indirect light, no noise, and
    /// measurably closer to a path traced reference than shadows alone. It is
    /// also seven times faster, which is why it is the default here.
    pub diffuse_samples: u32,
    /// Draw the environment behind the model rather than a flat background.
    pub show_environment: bool,
    /// Multiplies radiance before the display curve.
    pub exposure: f32,
    /// Equirectangular width the specular ladder is prefiltered at; the height
    /// is half of it. Cost is roughly quadratic in this, and the mirror end of
    /// the ladder cannot resolve a source smaller than one of its texels.
    pub specular_width: usize,
}

impl Default for RelightSettings {
    fn default() -> Self {
        Self {
            diffuse_samples: 0,
            show_environment: true,
            exposure: 1.0,
            specular_width: 256,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Present {
    exposure: f32,
    encode_srgb: u32,
    pad: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct RelightBlitData {
    g_src: gpu::TextureView,
    g_sampler: gpu::Sampler,
    g_present: Present,
}

pub struct RelightBackend {
    tracer: vol::gpu::RelightTracer,
    blit_pipeline: gpu::RenderPipeline,
    hdr_tex: gpu::Texture,
    hdr_view: gpu::TextureView,
    sampler: gpu::Sampler,
    present: Present,
    environments: Vec<NamedEnvironment>,
    current: usize,
    specular_width: usize,
    surfels: usize,
    materials: usize,
}

/// Whether the surface encodes for us.
///
/// Blade asks for a linear format presented in an sRGB colour space where it
/// can, in which case nothing downstream applies the transfer curve and the
/// shader has to. If it fell back to an sRGB format, the hardware does it and
/// doing it twice would wash the image out.
fn surface_encodes_srgb(format: gpu::TextureFormat) -> bool {
    match format {
        gpu::TextureFormat::Rgba8UnormSrgb | gpu::TextureFormat::Bgra8UnormSrgb => true,
        _ => false,
    }
}

/// Prefilter an environment if it has not been already.
///
/// Seconds of CPU work the first time and nothing after it, which is what
/// makes a light already looked at free to go back to.
fn ensure_prefiltered(entry: &mut NamedEnvironment, width: usize) {
    if entry.specular.is_some() {
        return;
    }
    let started = std::time::Instant::now();
    entry.specular = Some(vol::relight::SpecularEnvironment::prefilter(
        &entry.environment,
        width,
        width / 2,
    ));
    log::info!(
        "prefiltered '{}' at {width}x{} in {:.2} s",
        entry.name,
        width / 2,
        started.elapsed().as_secs_f64()
    );
}

impl RelightBackend {
    pub fn new(
        model: &vol::relight::RelightModel,
        mut environments: Vec<NamedEnvironment>,
        settings: RelightSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_format: gpu::TextureFormat,
        size: RenderSize,
    ) -> Self {
        assert!(
            !environments.is_empty(),
            "a relightable model needs at least one environment to be lit by"
        );
        let specular_width = settings.specular_width.max(8);
        ensure_prefiltered(&mut environments[0], specular_width);

        let tracer = vol::gpu::RelightTracer::new(
            model,
            &environments[0].environment,
            environments[0].specular.as_ref().unwrap(),
            vol::gpu::RelightSettings {
                background_rgb: [0.0; 3],
                diffuse_samples: settings.diffuse_samples,
                show_environment: settings.show_environment,
            },
            context,
            encoder,
        );

        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, size);
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "relight-present",
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = context.create_shader(gpu::ShaderDesc {
            source: vol::shaders::RELIGHT_BLIT,
            naga_module: None,
        });
        let layout = <RelightBlitData as gpu::ShaderData>::layout();
        let blit_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "relight-present",
            data_layouts: &[&layout],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            vertex: shader.at("vs"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs")),
            color_targets: &[surface_format.into()],
            depth_stencil: None,
            multisample_state: Default::default(),
        });

        Self {
            tracer,
            blit_pipeline,
            hdr_tex,
            hdr_view,
            sampler,
            present: Present {
                exposure: settings.exposure,
                encode_srgb: !surface_encodes_srgb(surface_format) as u32,
                pad: [0; 2],
            },
            environments,
            current: 0,
            specular_width,
            surfels: model.surfels.len(),
            materials: model.materials.len(),
        }
    }

    fn create_hdr_target(
        context: &gpu::Context,
        size: RenderSize,
    ) -> (gpu::Texture, gpu::TextureView) {
        // Sixteen bits of float per channel: the tracer's output is linear
        // radiance and a sun is far outside what a unorm target could hold.
        let format = gpu::TextureFormat::Rgba16Float;
        let tex = context.create_texture(gpu::TextureDesc {
            name: "relight-hdr",
            format,
            size: gpu::Extent {
                width: size.width.max(1),
                height: size.height.max(1),
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
            external: None,
        });
        let view = context.create_texture_view(
            tex,
            gpu::TextureViewDesc {
                name: "relight-hdr",
                format,
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

    pub fn environment_names(&self) -> impl Iterator<Item = &str> {
        self.environments.iter().map(|entry| entry.name.as_str())
    }

    pub fn current_environment(&self) -> usize {
        self.current
    }

    pub fn environment_count(&self) -> usize {
        self.environments.len()
    }

    /// Put the model under a different one of the loaded environments.
    ///
    /// The surfels, the materials and both acceleration structures are
    /// untouched; only the light is replaced. The first time a given
    /// environment is chosen this blocks for as long as prefiltering it takes.
    pub fn set_environment(
        &mut self,
        index: usize,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) {
        if index >= self.environments.len() || index == self.current {
            return;
        }
        ensure_prefiltered(&mut self.environments[index], self.specular_width);
        let entry = &self.environments[index];
        self.tracer.set_environment(
            &entry.environment,
            entry.specular.as_ref().unwrap(),
            context,
            encoder,
        );
        self.current = index;
    }

    pub fn next_environment(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        let next = (self.current + 1) % self.environments.len();
        self.set_environment(next, context, encoder);
    }

    pub fn exposure_mut(&mut self) -> &mut f32 {
        &mut self.present.exposure
    }

    pub fn exposure(&self) -> f32 {
        self.present.exposure
    }

    pub fn diffuse_samples_mut(&mut self) -> &mut u32 {
        self.tracer.diffuse_samples_mut()
    }

    pub fn diffuse_samples(&self) -> u32 {
        self.tracer.diffuse_samples()
    }

    pub fn show_environment(&self) -> bool {
        self.tracer.show_environment()
    }

    pub fn set_show_environment(&mut self, show: bool) {
        self.tracer.set_show_environment(show);
    }

    pub fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera_params: vol::CameraParams,
        size: RenderSize,
    ) {
        encoder.init_texture(self.hdr_tex);
        self.tracer.dispatch(
            encoder,
            self.hdr_view,
            camera_params,
            [size.width.max(1), size.height.max(1)],
        );

        if let mut pass = encoder.render(
            "relight-present",
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
                &RelightBlitData {
                    g_src: self.hdr_view,
                    g_sampler: self.sampler,
                    g_present: self.present,
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }

    pub fn print_info(&self) {
        println!("Relight Params:");
        println!("\tsurfels: {}", self.surfels);
        println!("\tmaterials: {}", self.materials);
        println!(
            "\tenvironment: {} ({} of {})",
            self.environments[self.current].name,
            self.current + 1,
            self.environments.len()
        );
        println!("\tdiffuse_samples: {}", self.diffuse_samples());
        println!("\texposure: {}", self.present.exposure);
        println!("\tshow_environment: {}", self.show_environment());
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.blit_pipeline);
        context.destroy_sampler(self.sampler);
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        self.tracer.deinit(context);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_encodes_exactly_when_the_surface_does_not() {
        assert!(!surface_encodes_srgb(gpu::TextureFormat::Bgra8Unorm));
        assert!(surface_encodes_srgb(gpu::TextureFormat::Bgra8UnormSrgb));
        assert!(surface_encodes_srgb(gpu::TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn the_present_uniform_is_one_vector_wide() {
        assert_eq!(std::mem::size_of::<Present>(), 16);
    }

    #[test]
    fn an_environment_is_prefiltered_once_and_then_reused() {
        let mut entry = NamedEnvironment::new(
            "uniform",
            vol::relight::Environment::uniform([0.5; 3], 32, 16),
        );
        ensure_prefiltered(&mut entry, 32);
        let first = entry.specular.as_ref().unwrap().levels[0][0];
        // The second call must not recompute: the point of the cache is that
        // switching back to a light already seen is free.
        let started = std::time::Instant::now();
        ensure_prefiltered(&mut entry, 32);
        let second = entry.specular.as_ref().unwrap().levels[0][0];
        assert_eq!(first, second);
        assert!(
            started.elapsed().as_millis() < 5,
            "the second call took {:?}, so it prefiltered again",
            started.elapsed()
        );
    }
}
