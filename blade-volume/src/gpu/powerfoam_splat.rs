use crate::{shaders, CameraParams, PointCloudModel};
use blade_graphics as gpu;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatIntegrateParams {
    sh_degree: u32,
    max_steps: u32,
    width: u32,
    height: u32,
    weight_threshold: f32,
    _padding: [f32; 3],
}

#[derive(blade_macros::ShaderData)]
struct SplatIntegrateData {
    g_camera: CameraParams,
    g_params: SplatIntegrateParams,
    g_attributes: gpu::BufferPiece,
    g_cells: gpu::BufferPiece,
    g_dts: gpu::BufferPiece,
    g_mask: gpu::BufferPiece,
    g_out: gpu::TextureView,
}

/// Compute-splat PowerFoam renderer used for faithful headless evaluation.
///
/// The first pass discovers every support sphere hit by each pixel and clips
/// it against its Cech-neighbor radical planes. The second pass integrates the
/// resulting disjoint intervals front-to-back. Unlike the camera-seeded walk,
/// this remains correct when the overlapping-ball graph is disconnected.
pub struct PowerFoamGpuSplatTracer {
    cloud: super::RadFoamGpuCloud,
    recorder: super::PathRecorder,
    buffers: super::PathRecordBuffers,
    integrate_pipeline: gpu::ComputePipeline,
    params: SplatIntegrateParams,
    resolution: [u32; 2],
}

impl PowerFoamGpuSplatTracer {
    pub fn new(
        model: &PointCloudModel,
        settings: super::RadFoamTraceSettings,
        resolution: [u32; 2],
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        assert!(
            model.radii.is_some(),
            "PowerFoam splat tracer requires radii"
        );
        assert!(
            resolution[0] > 0 && resolution[1] > 0,
            "PowerFoam splat resolution must be non-zero"
        );
        assert!(
            settings.max_steps > 0,
            "PowerFoam splat max_steps must be non-zero"
        );
        assert!(
            !settings.debug_cell_density,
            "PowerFoam splat tracer has no cell-density debug mode"
        );
        let num_pixels = resolution[0]
            .checked_mul(resolution[1])
            .expect("PowerFoam splat resolution is too large");
        let cloud = super::RadFoamGpuCloud::new(model, context, encoder);
        let recorder = super::PathRecorder::new(context);
        let buffers = super::PathRecordBuffers::new_powerfoam_recorded_only_projected(
            context,
            num_pixels,
            settings.max_steps,
            model.points.len() as u32,
            resolution,
            settings.powerfoam_candidate_capacity,
        );
        let pixel_indices = (0..num_pixels).collect::<Vec<_>>();
        buffers.write_pixel_indices(&pixel_indices);
        encoder.start();
        {
            let mut transfer = encoder.transfer("powerfoam-splat-pixel-indices");
            transfer.copy_buffer_to_buffer(
                buffers.pixel_indices_stage.at(0),
                buffers.pixel_indices.at(0),
                u64::from(num_pixels) * std::mem::size_of::<u32>() as u64,
            );
        }
        let sync = context.submit(encoder);
        let completed = context
            .wait_for(&sync, !0)
            .expect("PowerFoam splat pixel-index upload failed");
        assert!(completed, "PowerFoam splat pixel-index upload timed out");

        let source = shaders::compose(shaders::POWERFOAM_SPLAT);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <SplatIntegrateData as gpu::ShaderData>::layout();
        let integrate_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "powerfoam-integrate-splats",
            data_layouts: &[&layout],
            compute: shader.at("integrate_powerfoam_splats"),
        });
        let params = SplatIntegrateParams {
            sh_degree: cloud.sh_degree as u32,
            max_steps: settings.max_steps,
            width: resolution[0],
            height: resolution[1],
            weight_threshold: settings.weight_threshold,
            _padding: [0.0; 3],
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
            let mut transfer = encoder.transfer("powerfoam-splat-path-prepare");
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
                max_path_dt: 50.0,
                depth: camera.depth,
                num_pixels,
            },
        );

        let mut pass = encoder.compute("powerfoam-integrate-splats");
        let mut compute = pass.with(&self.integrate_pipeline);
        compute.bind(
            0,
            &SplatIntegrateData {
                g_camera: camera,
                g_params: self.params,
                g_attributes: self.cloud.attributes(),
                g_cells: self.buffers.cells.into(),
                g_dts: self.buffers.dts.into(),
                g_mask: self.buffers.mask.into(),
                g_out: output,
            },
        );
        compute.dispatch([
            self.resolution[0].div_ceil(8),
            self.resolution[1].div_ceil(8),
            1,
        ]);
    }

    /// Reject a completed render if its bounded candidate scratch overflowed.
    pub fn validate_candidate_counts(&self) -> Result<u32, String> {
        let num_pixels = (self.resolution[0] * self.resolution[1]) as usize;
        let observed = self.buffers.max_splat_candidate_count(0..num_pixels);
        let capacity = self.buffers.splat_candidate_capacity();
        let tile_observed = self.buffers.max_splat_tile_candidate_count(self.resolution);
        log::debug!(
            "PowerFoam candidates: ray max={observed}/{capacity}, tile max={tile_observed}/{}",
            self.buffers.splat_tile_capacity(),
        );
        if observed > capacity {
            Err(format!(
                "PowerFoam render needs {observed} candidates for one ray, but scratch capacity is {capacity}"
            ))
        } else {
            Ok(observed)
        }
    }

    /// Fixed sphere-candidate row capacity used by this tracer.
    pub fn candidate_capacity(&self) -> u32 {
        self.buffers.splat_candidate_capacity()
    }

    /// Summarize recorded paths after the render submission has completed.
    pub fn path_stats(&self) -> super::PathRecordStats {
        let num_pixels = (self.resolution[0] * self.resolution[1]) as usize;
        self.buffers.path_stats(0..num_pixels)
    }

    /// Fixed path-row capacity used by this tracer.
    pub fn max_steps(&self) -> u32 {
        self.params.max_steps
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
    use super::SplatIntegrateParams;

    #[test]
    fn integrate_params_match_wgsl_scalar_layout() {
        assert_eq!(std::mem::size_of::<SplatIntegrateParams>(), 32);
        assert_eq!(
            std::mem::offset_of!(SplatIntegrateParams, weight_threshold),
            16
        );
    }
}
