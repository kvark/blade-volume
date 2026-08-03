//! Unified scene renderer using software TLAS traversal.
//!
//! This module provides `SceneRenderer` which renders scenes containing
//! multiple object types (Gaussian, RadFoam, etc.) using a unified
//! compute-based traversal of bounding volumes.
//!
//! # Architecture
//!
//! Unlike the single-backend viewers, the scene renderer:
//! 1. Traverses object bounding spheres in a compute shader
//! 2. Dispatches to object-specific backends when rays hit bounds
//! 3. Composites results from all objects
//!
//! This allows mixing Gaussian and RadFoam objects in the same scene
//! with independent transforms.

use crate::RenderSize;
use blade_graphics as gpu;
use blade_volume as vol;

/// Maximum number of objects supported per point-cloud backend.
const MAX_CLOUD_OBJECTS: gpu::ResourceIndex = 64;

/// Parameters passed to the scene traversal shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SceneParams {
    /// Number of objects in the scene.
    pub object_count: u32,
    /// Stop when transmittance <= threshold.
    pub weight_threshold: f32,
    /// Maximum cell transitions for RadFoam.
    pub max_steps: u32,
    /// Debug visualization mode.
    pub debug_mode: u32,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            object_count: 0,
            weight_threshold: 0.001,
            max_steps: 1024,
            debug_mode: 0,
        }
    }
}

/// Debug modes for scene visualization.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneDebugMode {
    /// Normal rendering.
    Off = 0,
    /// Show bounding sphere intersections (heatmap by count).
    Bounds = 1,
    /// Color by object type.
    ObjectType = 2,
}

/// Shader data layout for scene traversal.
///
/// This struct defines the bindings for the unified scene traversal shader.
/// RadFoam uses binding arrays to support multiple objects.
#[derive(blade_macros::ShaderData)]
pub struct SceneTraverseData<'a> {
    /// Camera parameters.
    pub g_camera: vol::CameraParams,
    /// Scene traversal parameters.
    pub g_scene_params: SceneParams,
    /// Object bounding spheres buffer.
    pub g_bounds: gpu::BufferPiece,
    /// Object transforms buffer.
    pub g_transforms: gpu::BufferPiece,
    /// RadFoam points buffers (binding array).
    pub g_radfoam_points: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    /// RadFoam oriented-surface normal buffers (binding array).
    pub g_radfoam_surface_normals: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    /// RadFoam attributes buffers (binding array).
    pub g_radfoam_attributes: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    /// RadFoam adjacency buffers (binding array).
    pub g_radfoam_adjacency: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    /// RadFoam adjacency offsets buffers (binding array).
    pub g_radfoam_adjacency_offsets: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    /// Gaussian TLASes (binding array).
    pub g_gaussian_tlas: &'a gpu::AccelerationStructureArray<MAX_CLOUD_OBJECTS>,
    /// Gaussian data buffers (binding array).
    pub g_gaussian_data: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    /// Output HDR texture.
    pub g_out: gpu::TextureView,
}

/// Shader data for the RadFoam/PowerFoam-only scene pipeline.
#[derive(blade_macros::ShaderData)]
pub struct RadFoamSceneTraverseData<'a> {
    pub g_camera: vol::CameraParams,
    pub g_scene_params: SceneParams,
    pub g_bounds: gpu::BufferPiece,
    pub g_transforms: gpu::BufferPiece,
    pub g_radfoam_points: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    pub g_radfoam_surface_normals: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    pub g_radfoam_attributes: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    pub g_radfoam_adjacency: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    pub g_radfoam_adjacency_offsets: &'a gpu::BufferArray<MAX_CLOUD_OBJECTS>,
    pub g_out: gpu::TextureView,
}

/// Shader data layout for HDR to swapchain blit.
#[derive(blade_macros::ShaderData)]
pub struct SceneBlitData {
    /// HDR source texture.
    pub g_src: gpu::TextureView,
    /// Texture sampler.
    pub g_sampler: gpu::Sampler,
    /// Explicit display-referred sRGB-code-value background.
    pub g_background: SceneBackground,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SceneBackground {
    pub color: [f32; 3],
    pub pad: f32,
}

/// Unified scene renderer.
///
/// Renders scenes with mixed object types using software TLAS traversal.
pub struct SceneRenderer {
    /// The scene being rendered.
    pub scene: vol::Scene,
    /// Mixed Gaussian/RadFoam traversal compute pipeline.
    mixed_traverse_pipeline: Option<gpu::ComputePipeline>,
    /// RadFoam-only traversal compute pipeline without ray-query bindings.
    radfoam_traverse_pipeline: gpu::ComputePipeline,
    /// HDR to swapchain blit pipeline.
    blit_pipeline: gpu::RenderPipeline,
    /// HDR render target.
    hdr_tex: gpu::Texture,
    /// HDR render target view.
    hdr_view: gpu::TextureView,
    /// Texture sampler for blit.
    sampler: gpu::Sampler,
    /// Display-referred sRGB-code-value presentation background.
    pub background_rgb: [f32; 3],
    /// Rendering parameters.
    pub params: SceneParams,
}

impl SceneRenderer {
    /// Creates a new scene renderer.
    pub fn new(
        context: &gpu::Context,
        surface_format: gpu::TextureFormat,
        window_size: RenderSize,
    ) -> Self {
        let capabilities = context.capabilities();
        assert!(
            capabilities.binding_array,
            "SceneRenderer requires resource binding arrays"
        );
        // Create HDR target
        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, window_size);

        // Sampler for blit
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "scene-sampler",
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });

        // Acceleration-structure arrays are currently implemented by Blade's
        // Vulkan backend only. Keep RadFoam-only scenes usable on binding-array
        // devices without ray queries or an unsupported mixed pipeline.
        let mixed_traverse_pipeline = if !cfg!(target_vendor = "apple")
            && capabilities
                .ray_query
                .contains(gpu::ShaderVisibility::COMPUTE)
        {
            let source = vol::shaders::compose(vol::shaders::SCENE_TRAVERSE);
            let shader = context.create_shader(gpu::ShaderDesc {
                source: &source,
                naga_module: None,
            });
            let layout = <SceneTraverseData as gpu::ShaderData>::layout();
            Some(context.create_compute_pipeline(gpu::ComputePipelineDesc {
                name: "scene-traverse",
                data_layouts: &[&layout],
                compute: shader.at("main"),
            }))
        } else {
            None
        };

        let radfoam_shader = {
            let source = vol::shaders::compose(vol::shaders::SCENE_RADFOAM);
            context.create_shader(gpu::ShaderDesc {
                source: &source,
                naga_module: None,
            })
        };
        let radfoam_traverse_layout = <RadFoamSceneTraverseData as gpu::ShaderData>::layout();
        let radfoam_traverse_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "scene-traverse-radfoam",
            data_layouts: &[&radfoam_traverse_layout],
            compute: radfoam_shader.at("main"),
        });

        // Blit pipeline (reuse radfoam_blit shader)
        let blit_shader = {
            let source = vol::shaders::RADFOAM_BLIT;
            context.create_shader(gpu::ShaderDesc {
                source,
                naga_module: None,
            })
        };
        let blit_layout = <SceneBlitData as gpu::ShaderData>::layout();
        let blit_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "scene-blit",
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

        Self {
            scene: vol::Scene::new(),
            mixed_traverse_pipeline,
            radfoam_traverse_pipeline,
            blit_pipeline,
            hdr_tex,
            hdr_view,
            sampler,
            background_rgb: [0.0; 3],
            params: SceneParams::default(),
        }
    }

    fn create_hdr_target(
        context: &gpu::Context,
        size: RenderSize,
    ) -> (gpu::Texture, gpu::TextureView) {
        let tex = context.create_texture(gpu::TextureDesc {
            name: "scene-hdr",
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
                name: "scene-hdr-view",
                format: gpu::TextureFormat::Rgba16Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        (tex, view)
    }

    /// Handles window resize.
    pub fn resize(&mut self, context: &gpu::Context, size: RenderSize) {
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        let (hdr_tex, hdr_view) = Self::create_hdr_target(context, size);
        self.hdr_tex = hdr_tex;
        self.hdr_view = hdr_view;
    }

    /// Adds a RadFoam model to the scene.
    ///
    /// Returns the object handle for transform manipulation.
    pub fn add_radfoam(
        &mut self,
        model: &vol::PointCloudModel,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> vol::ObjectHandle {
        self.scene.add_radfoam(model, context, encoder)
    }

    /// Adds a Gaussian model to the scene.
    ///
    /// Returns the object handle for transform manipulation.
    pub fn add_gaussian(
        &mut self,
        model: &vol::PointCloudModel,
        min_opacity: f32,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> vol::ObjectHandle {
        assert!(
            self.mixed_traverse_pipeline.is_some(),
            "Gaussian scenes require Vulkan compute ray queries and acceleration-structure arrays"
        );
        self.scene
            .add_gaussian(model, min_opacity, context, encoder)
    }

    /// Sets the debug mode.
    pub fn set_debug_mode(&mut self, mode: SceneDebugMode) {
        self.params.debug_mode = mode as u32;
    }

    /// Renders the scene.
    pub fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        frame_view: gpu::TextureView,
        camera_params: vol::CameraParams,
        camera_position: glam::Vec3,
        window_size: RenderSize,
        context: &gpu::Context,
    ) {
        // Seed each adjacency traversal from the camera-containing local cell,
        // then upload bounds/transforms if either camera or objects changed.
        self.scene.update_camera(camera_position);
        self.scene.prepare(context, encoder);

        // Get render data
        let Some(render_data) = self.scene.render_data() else {
            return; // Empty scene
        };

        // Update params
        self.params.object_count = render_data.object_count;

        encoder.init_texture(self.hdr_tex);

        // Build RadFoam binding arrays from all objects
        assert!(
            render_data.radfoam_clouds.len() <= MAX_CLOUD_OBJECTS as usize,
            "SceneRenderer supports at most {MAX_CLOUD_OBJECTS} RadFoam objects"
        );
        assert!(
            render_data.gaussian_clouds.len() <= MAX_CLOUD_OBJECTS as usize,
            "SceneRenderer supports at most {MAX_CLOUD_OBJECTS} Gaussian objects"
        );
        let mut radfoam_points: gpu::BufferArray<MAX_CLOUD_OBJECTS> = gpu::BufferArray::new();
        let mut radfoam_surface_normals: gpu::BufferArray<MAX_CLOUD_OBJECTS> =
            gpu::BufferArray::new();
        let mut radfoam_attributes: gpu::BufferArray<MAX_CLOUD_OBJECTS> = gpu::BufferArray::new();
        let mut radfoam_adjacency: gpu::BufferArray<MAX_CLOUD_OBJECTS> = gpu::BufferArray::new();
        let mut radfoam_adjacency_offsets: gpu::BufferArray<MAX_CLOUD_OBJECTS> =
            gpu::BufferArray::new();

        for cloud in render_data.radfoam_clouds {
            radfoam_points.alloc(cloud.points());
            radfoam_surface_normals.alloc(cloud.surface_normals());
            radfoam_attributes.alloc(cloud.attributes());
            radfoam_adjacency.alloc(cloud.point_adjacency());
            radfoam_adjacency_offsets.alloc(cloud.point_adjacency_offsets());
        }

        let dispatch = [
            window_size.width.div_ceil(8),
            window_size.height.div_ceil(8),
            1,
        ];
        if let Some(first_gaussian) = render_data.gaussian_clouds.first() {
            let mixed_traverse_pipeline = self
                .mixed_traverse_pipeline
                .as_ref()
                .expect("Gaussian object added without a mixed scene pipeline");
            let mut gaussian_tlas: gpu::AccelerationStructureArray<MAX_CLOUD_OBJECTS> =
                gpu::AccelerationStructureArray::new();
            let mut gaussian_data: gpu::BufferArray<MAX_CLOUD_OBJECTS> = gpu::BufferArray::new();
            for cloud in render_data.gaussian_clouds {
                gaussian_tlas.alloc(cloud.tlas);
                gaussian_data.alloc(cloud.gauss_buf.into());
            }

            // Vulkan descriptor arrays must not be empty. These entries are
            // never indexed when the scene has no RadFoam objects.
            if render_data.radfoam_clouds.is_empty() {
                let fallback = first_gaussian.gauss_buf.into();
                radfoam_points.alloc(fallback);
                radfoam_surface_normals.alloc(fallback);
                radfoam_attributes.alloc(fallback);
                radfoam_adjacency.alloc(fallback);
                radfoam_adjacency_offsets.alloc(fallback);
            }

            if let mut pass = encoder.compute("scene-traverse-mixed") {
                let mut pen = pass.with(mixed_traverse_pipeline);
                pen.bind(
                    0,
                    &SceneTraverseData {
                        g_camera: camera_params,
                        g_scene_params: self.params,
                        g_bounds: render_data.bounds,
                        g_transforms: render_data.transforms,
                        g_radfoam_points: &radfoam_points,
                        g_radfoam_surface_normals: &radfoam_surface_normals,
                        g_radfoam_attributes: &radfoam_attributes,
                        g_radfoam_adjacency: &radfoam_adjacency,
                        g_radfoam_adjacency_offsets: &radfoam_adjacency_offsets,
                        g_gaussian_tlas: &gaussian_tlas,
                        g_gaussian_data: &gaussian_data,
                        g_out: self.hdr_view,
                    },
                );
                pen.dispatch(dispatch);
            }
        } else if let mut pass = encoder.compute("scene-traverse-radfoam") {
            let mut pen = pass.with(&self.radfoam_traverse_pipeline);
            pen.bind(
                0,
                &RadFoamSceneTraverseData {
                    g_camera: camera_params,
                    g_scene_params: self.params,
                    g_bounds: render_data.bounds,
                    g_transforms: render_data.transforms,
                    g_radfoam_points: &radfoam_points,
                    g_radfoam_surface_normals: &radfoam_surface_normals,
                    g_radfoam_attributes: &radfoam_attributes,
                    g_radfoam_adjacency: &radfoam_adjacency,
                    g_radfoam_adjacency_offsets: &radfoam_adjacency_offsets,
                    g_out: self.hdr_view,
                },
            );
            pen.dispatch(dispatch);
        }

        // Blit HDR -> swapchain
        if let mut pass = encoder.render(
            "scene-present",
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
                &SceneBlitData {
                    g_src: self.hdr_view,
                    g_sampler: self.sampler,
                    g_background: SceneBackground {
                        color: self.background_rgb,
                        pad: 0.0,
                    },
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }

    /// Cleans up GPU resources.
    pub fn destroy(mut self, context: &gpu::Context) {
        self.scene.destroy(context);
        context.destroy_sampler(self.sampler);
        context.destroy_texture_view(self.hdr_view);
        context.destroy_texture(self.hdr_tex);
        if let Some(ref mut pipeline) = self.mixed_traverse_pipeline {
            context.destroy_compute_pipeline(pipeline);
        }
        context.destroy_compute_pipeline(&mut self.radfoam_traverse_pipeline);
        context.destroy_render_pipeline(&mut self.blit_pipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_params_match_one_wgsl_uniform_block() {
        assert_eq!(std::mem::size_of::<SceneParams>(), 16);
    }

    #[test]
    fn scene_pipelines_compile_with_explicit_background_binding() {
        let _gpu_test_guard = crate::gpu_test_guard();
        if vol::gpu::access_disabled() {
            eprintln!("skipping scene shader compilation: GPU access disabled");
            return;
        }
        let Some(context) = (unsafe {
            gpu::Context::init(gpu::ContextDesc {
                ray_tracing: true,
                ..gpu::ContextDesc::default()
            })
            .ok()
        }) else {
            eprintln!("skipping scene shader compilation: no ray-query GPU");
            return;
        };
        let renderer = SceneRenderer::new(
            &context,
            gpu::TextureFormat::Rgba16Float,
            RenderSize {
                width: 4,
                height: 4,
            },
        );
        renderer.destroy(&context);
    }
}
