use crate::{shaders, CameraParams, PointCloudModel};
use blade_graphics as gpu;

#[derive(Clone, Copy, Debug)]
pub struct RadFoamTraceSettings {
    pub max_steps: u32,
    /// Minimum PowerFoam sphere-candidate row capacity. Zero selects the
    /// automatic `max(4 * max_steps, 1024)` budget.
    pub powerfoam_candidate_capacity: u32,
    pub weight_threshold: f32,
    pub debug_cell_density: bool,
}

impl Default for RadFoamTraceSettings {
    fn default() -> Self {
        Self {
            max_steps: 1024,
            powerfoam_candidate_capacity: 0,
            weight_threshold: 0.001,
            debug_cell_density: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(super) struct TraceParams {
    pub(super) sh_degree: u32,
    pub(super) weight_threshold: f32,
    pub(super) max_steps: u32,
    pub(super) start_point: u32,
    pub(super) debug_mode: u32,
    pub(super) align_pad: [u32; 3],
    pub(super) power_foam: u32,
    pub(super) size_pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct TraceData {
    g_camera: CameraParams,
    g_params: TraceParams,
    g_points: gpu::BufferPiece,
    g_surface_normals: gpu::BufferPiece,
    g_attributes: gpu::BufferPiece,
    g_adjacency: gpu::BufferPiece,
    g_adjacency_offsets: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

/// Reusable production RadFoam/PowerFoam compute tracer.
///
/// The caller owns the output storage texture and command encoder. Keeping
/// those outside lets windowed rendering and headless readback share the same
/// cloud upload, uniform layout, and WGSL entry point.
pub struct RadFoamGpuTracer {
    cloud: super::RadFoamGpuCloud,
    pipeline: gpu::ComputePipeline,
    params: TraceParams,
}

impl RadFoamGpuTracer {
    pub fn new(
        model: &PointCloudModel,
        settings: RadFoamTraceSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        let source = shaders::compose(shaders::RADFOAM);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <TraceData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "radfoam-trace",
            data_layouts: &[&layout],
            compute: shader.at("trace_main"),
        });
        let cloud = super::RadFoamGpuCloud::new(model, context, encoder);
        let params = TraceParams {
            sh_degree: cloud.sh_degree as u32,
            weight_threshold: settings.weight_threshold,
            max_steps: settings.max_steps,
            start_point: 0,
            debug_mode: settings.debug_cell_density as u32,
            align_pad: [0; 3],
            power_foam: cloud.is_power_foam as u32,
            size_pad: [cloud.is_oriented as u32, 0, 0],
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
        assert!(size[0] > 0 && size[1] > 0, "render size must be non-zero");
        self.params.start_point = self
            .cloud
            .containing_point(glam::Vec3::from(camera.cam_position));

        let mut pass = encoder.compute("radfoam-trace");
        let mut pen = pass.with(&self.pipeline);
        pen.bind(
            0,
            &TraceData {
                g_camera: camera,
                g_params: self.params,
                g_points: self.cloud.points(),
                g_surface_normals: self.cloud.surface_normals(),
                g_attributes: self.cloud.attributes(),
                g_adjacency: self.cloud.point_adjacency(),
                g_adjacency_offsets: self.cloud.point_adjacency_offsets(),
                g_out: output,
            },
        );
        pen.dispatch([size[0].div_ceil(8), size[1].div_ceil(8), 1]);
    }

    pub fn max_steps(&self) -> u32 {
        self.params.max_steps
    }

    pub fn max_steps_mut(&mut self) -> &mut u32 {
        &mut self.params.max_steps
    }

    pub fn set_max_steps(&mut self, max_steps: u32) {
        self.params.max_steps = max_steps;
    }

    pub fn weight_threshold(&self) -> f32 {
        self.params.weight_threshold
    }

    pub fn weight_threshold_mut(&mut self) -> &mut f32 {
        &mut self.params.weight_threshold
    }

    pub fn set_weight_threshold(&mut self, weight_threshold: f32) {
        self.params.weight_threshold = weight_threshold;
    }

    pub fn set_debug_cell_density(&mut self, enabled: bool) {
        self.params.debug_mode = enabled as u32;
    }

    pub fn sh_degree(&self) -> u32 {
        self.params.sh_degree
    }

    pub fn start_point(&self) -> u32 {
        self.params.start_point
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
        self.cloud.deinit(context);
    }
}

#[cfg(test)]
mod tests {
    use super::TraceParams;

    #[test]
    fn power_flag_matches_wgsl_uniform_layout() {
        assert_eq!(std::mem::size_of::<TraceParams>(), 48);
        assert_eq!(std::mem::offset_of!(TraceParams, power_foam), 32);
    }
}
