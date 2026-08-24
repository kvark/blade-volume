use crate::{shaders, CameraParams, PointCloudModel};
use blade_graphics as gpu;

use super::radfoam_depth::RadFoamDepthSettings;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatDepthParams {
    sh_degree: u32,
    max_steps: u32,
    width: u32,
    height: u32,
    weight_threshold: f32,
    appearance_flags: u32,
    _padding: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct SplatDepthData {
    g_camera: CameraParams,
    g_params: SplatDepthParams,
    g_points: gpu::BufferPiece,
    g_surface_normals: gpu::BufferPiece,
    g_surface_details: gpu::BufferPiece,
    g_attributes: gpu::BufferPiece,
    g_cells: gpu::BufferPiece,
    g_entry_depths: gpu::BufferPiece,
    g_dts: gpu::BufferPiece,
    g_mask: gpu::BufferPiece,
    g_surface_queries: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

/// Full-precision PowerFoam depth statistics from independently clipped
/// supports.
///
/// Unlike the camera-seeded adjacency walk, this discovers valid supports in
/// every disconnected component of a Čech graph.
pub struct PowerFoamGpuDepthTracer {
    cloud: super::RadFoamGpuCloud,
    recorder: super::PathRecorder,
    buffers: super::PathRecordBuffers,
    integrate_pipeline: gpu::ComputePipeline,
    params: SplatDepthParams,
    resolution: [u32; 2],
}

impl PowerFoamGpuDepthTracer {
    pub fn new(
        model: &PointCloudModel,
        settings: RadFoamDepthSettings,
        resolution: [u32; 2],
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        assert!(
            model.radii.is_some(),
            "PowerFoam depth tracing requires radii"
        );
        assert!(
            resolution[0] > 0 && resolution[1] > 0,
            "PowerFoam depth resolution must be non-zero"
        );
        let num_pixels = resolution[0]
            .checked_mul(resolution[1])
            .expect("PowerFoam depth resolution is too large");
        let cloud = super::RadFoamGpuCloud::new(model, context, encoder);
        let recorder = super::PathRecorder::new(context);
        let buffers = super::PathRecordBuffers::new_powerfoam_depth(
            context,
            num_pixels,
            settings.max_steps,
            model.points.len() as u32,
            resolution,
            0,
            model.surface_detail.is_some(),
        );
        let pixel_indices = (0..num_pixels).collect::<Vec<_>>();
        buffers.write_pixel_indices(&pixel_indices);
        encoder.start();
        {
            let mut transfer = encoder.transfer("powerfoam-depth-pixel-indices");
            transfer.copy_buffer_to_buffer(
                buffers.pixel_indices_stage.at(0),
                buffers.pixel_indices.at(0),
                u64::from(num_pixels) * std::mem::size_of::<u32>() as u64,
            );
        }
        let sync = context.submit(encoder);
        let completed = context
            .wait_for(&sync, !0)
            .expect("PowerFoam depth pixel-index upload failed");
        assert!(completed, "PowerFoam depth pixel-index upload timed out");

        let source = shaders::compose(shaders::POWERFOAM_DEPTH);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <SplatDepthData as gpu::ShaderData>::layout();
        let integrate_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-integrate-depth",
            data_layouts: &[&layout],
            compute: shader.at("integrate_powerfoam_depth"),
        });
        let params = SplatDepthParams {
            sh_degree: cloud.sh_degree as u32,
            max_steps: settings.max_steps,
            width: resolution[0],
            height: resolution[1],
            weight_threshold: settings.weight_threshold,
            appearance_flags: cloud.has_surface_color as u32
                | (cloud.has_spherical_voronoi as u32) << 1
                | (cloud.has_surface_detail as u32) << 2
                | (cloud.has_surface_detail_density as u32) << 3
                | (cloud.has_surface_detail_directional as u32) << 4,
            _padding: [0; 2],
        };
        Self {
            cloud,
            recorder,
            buffers,
            integrate_pipeline,
            params,
            resolution,
        }
    }

    pub fn dispatch(
        &self,
        encoder: &mut gpu::CommandEncoder,
        output: gpu::TextureView,
        camera: CameraParams,
    ) {
        let num_pixels = self.resolution[0] * self.resolution[1];
        let path_bytes = u64::from(num_pixels)
            * u64::from(self.params.max_steps)
            * std::mem::size_of::<f32>() as u64;
        {
            let mut transfer = encoder.transfer("powerfoam-depth-path-prepare");
            transfer.fill_buffer(self.buffers.mask.at(0), path_bytes, 0);
        }
        self.recorder.dispatch(
            encoder,
            &self.cloud,
            &self.buffers,
            super::RecordPathsArgs {
                camera,
                start_point: 0,
                pixel_offset: 0,
                max_steps: self.params.max_steps,
                image_width: self.resolution[0],
                image_height: self.resolution[1],
                max_path_dt: camera.depth,
                depth: camera.depth,
                num_pixels,
            },
        );

        let mut pass = encoder.compute("powerfoam-integrate-depth");
        let mut compute = pass.with(&self.integrate_pipeline);
        compute.bind(
            0,
            &SplatDepthData {
                g_camera: camera,
                g_params: self.params,
                g_points: self.cloud.points(),
                g_surface_normals: self.cloud.surface_normals(),
                g_surface_details: self.cloud.surface_details(),
                g_attributes: self.cloud.attributes(),
                g_cells: self.buffers.cells.into(),
                g_entry_depths: self.buffers.next_cells.into(),
                g_dts: self.buffers.dts.into(),
                g_mask: self.buffers.mask.into(),
                g_surface_queries: self.buffers.surface_queries.into(),
                g_out: output,
            },
        );
        compute.dispatch([
            self.resolution[0].div_ceil(8),
            self.resolution[1].div_ceil(8),
            1,
        ]);
    }

    /// Reject a completed depth map if candidate or path scratch overflowed.
    pub fn validate(&self) -> Result<(), String> {
        let num_pixels = (self.resolution[0] * self.resolution[1]) as usize;
        let observed = self.buffers.max_splat_candidate_count(0..num_pixels);
        let capacity = self.buffers.splat_candidate_capacity();
        if observed > capacity {
            return Err(format!(
                "PowerFoam depth needs {observed} candidates for one ray, but scratch capacity is {capacity}"
            ));
        }
        let stats = self.buffers.path_stats(0..num_pixels);
        if stats.truncated_rays != 0 {
            return Err(format!(
                "PowerFoam depth truncated {} rays at {} segments",
                stats.truncated_rays, self.params.max_steps,
            ));
        }
        Ok(())
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.integrate_pipeline);
        self.buffers.destroy(context);
        self.recorder.destroy(context);
        self.cloud.deinit(context);
    }
}

#[cfg(test)]
mod tests {
    use super::SplatDepthParams;

    #[test]
    fn depth_params_match_wgsl_scalar_layout() {
        assert_eq!(std::mem::size_of::<SplatDepthParams>(), 32);
        assert_eq!(std::mem::offset_of!(SplatDepthParams, weight_threshold), 16,);
    }
}
