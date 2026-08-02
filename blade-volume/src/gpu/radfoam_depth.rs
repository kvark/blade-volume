use crate::{shaders, CameraParams, PointCloudModel};
use blade_graphics as gpu;

use super::radfoam_trace::TraceParams;

#[derive(Clone, Copy, Debug)]
pub struct RadFoamDepthSettings {
    pub max_steps: u32,
    pub weight_threshold: f32,
}

impl Default for RadFoamDepthSettings {
    fn default() -> Self {
        Self {
            max_steps: 1024,
            weight_threshold: 0.001,
        }
    }
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

/// Full-precision depth-mode output from the production RadFoam/PowerFoam
/// compute walk.
///
/// The caller owns the `rgba32float` output texture. Its channels are mode
/// depth, accumulated alpha, peak segment weight, and one respectively.
pub struct RadFoamGpuDepthTracer {
    cloud: super::RadFoamGpuCloud,
    pipeline: gpu::ComputePipeline,
    params: TraceParams,
}

impl RadFoamGpuDepthTracer {
    pub fn new(
        model: &PointCloudModel,
        settings: RadFoamDepthSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        let source = shaders::compose(shaders::RADFOAM_DEPTH);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <TraceData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-depth",
            data_layouts: &[&layout],
            compute: shader.at("trace_depth_main"),
        });
        let cloud = super::RadFoamGpuCloud::new(model, context, encoder);
        let params = TraceParams {
            sh_degree: cloud.sh_degree as u32,
            weight_threshold: settings.weight_threshold,
            max_steps: settings.max_steps,
            start_point: 0,
            debug_mode: 0,
            align_pad: [0; 3],
            power_foam: cloud.is_power_foam as u32,
            size_pad: [0; 3],
        };
        Self {
            cloud,
            pipeline,
            params,
        }
    }

    pub fn dispatch(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        output: gpu::TextureView,
        camera: CameraParams,
        size: [u32; 2],
    ) {
        assert!(size[0] > 0 && size[1] > 0, "depth map must be non-empty");
        self.params.start_point = self
            .cloud
            .containing_point(glam::Vec3::from(camera.cam_position));

        let mut pass = encoder.compute("radfoam-depth");
        let mut pen = pass.with(&self.pipeline);
        pen.bind(
            0,
            &TraceData {
                g_camera: camera,
                g_params: self.params,
                g_points: self.cloud.points(),
                g_attributes: self.cloud.attributes(),
                g_adjacency: self.cloud.point_adjacency(),
                g_adjacency_offsets: self.cloud.point_adjacency_offsets(),
                g_out: output,
            },
        );
        pen.dispatch([size[0].div_ceil(8), size[1].div_ceil(8), 1]);
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
        self.cloud.deinit(context);
    }
}
