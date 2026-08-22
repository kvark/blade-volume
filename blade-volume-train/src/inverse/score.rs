//! Rendering a reconstruction back into the capture it was recovered from.
//!
//! This is the scoreboard, and it is deliberately the least clever part of the
//! program: the scene goes through the same [`RelightTracer`] a viewer uses,
//! at the capture's own poses and intrinsics, and the result is compared to
//! the photograph. Nothing here fits an exposure or aligns a colour balance.
//! A number that needed one of those would be measuring the alignment.
//!
//! [`RelightTracer`]: vol::gpu::RelightTracer

use crate::inverse::capture;
use blade_graphics as gpu;
use blade_volume as vol;
use std::{path, sync};

/// Everything a reconstruction consists of.
#[derive(Clone)]
pub struct Scene {
    pub model: vol::relight::RelightModel,
    pub environment: vol::relight::Environment,
}

/// How well one image was reproduced.
#[derive(Clone, Copy, Debug)]
pub struct Score {
    /// PSNR in linear radiance against a peak of one. Dominated by whatever
    /// is brightest in the frame, which in a room is usually a lamp.
    pub linear_psnr: f64,
    /// PSNR after the display transfer function, which is the number a
    /// novel-view-synthesis paper reports.
    pub srgb_psnr: f64,
    /// Fraction of the frame that hit reconstructed geometry. A high PSNR
    /// over a tenth of the pixels is not a reconstruction.
    pub coverage: f64,
    /// PSNR over the covered part of the frame alone.
    ///
    /// The whole-frame number mixes two failures that need separate fixes —
    /// geometry that is not there, and material and light that are wrong
    /// where it is. This one holds the first constant so the second can be
    /// read. It is not a result on its own: a reconstruction covering a tenth
    /// of the frame can score well here and be worthless.
    pub covered_srgb_psnr: f64,
}

/// The mean and the worst of a set of scores.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub linear_psnr: f64,
    pub srgb_psnr: f64,
    pub worst_srgb_psnr: f64,
    pub coverage: f64,
    pub covered_srgb_psnr: f64,
    pub views: usize,
    pub render_ms: f64,
}

impl Summary {
    fn of(scores: &[Score], render_ms: f64) -> Self {
        if scores.is_empty() {
            return Self::default();
        }
        let count = scores.len() as f64;
        Self {
            linear_psnr: scores.iter().map(|s| s.linear_psnr).sum::<f64>() / count,
            srgb_psnr: scores.iter().map(|s| s.srgb_psnr).sum::<f64>() / count,
            worst_srgb_psnr: scores
                .iter()
                .map(|s| s.srgb_psnr)
                .fold(f64::INFINITY, f64::min),
            coverage: scores.iter().map(|s| s.coverage).sum::<f64>() / count,
            covered_srgb_psnr: scores.iter().map(|s| s.covered_srgb_psnr).sum::<f64>() / count,
            views: scores.len(),
            render_ms,
        }
    }
}

fn psnr(mean_square_error: f64) -> f64 {
    if mean_square_error <= 1.0e-12 {
        99.0
    } else {
        -10.0 * mean_square_error.log10()
    }
}

/// Compare one rendered frame against its photograph.
pub fn compare(rendered: &[[f32; 4]], reference: &[[f32; 3]], covered: &[f32]) -> Score {
    compare_coverage(rendered, reference, Some(covered))
}

fn compare_coverage(
    rendered: &[[f32; 4]],
    reference: &[[f32; 3]],
    covered: Option<&[f32]>,
) -> Score {
    /// A pixel counts as covered when geometry accounts for most of it.
    const MOSTLY: f32 = 0.5;

    let mut linear = 0.0f64;
    let mut encoded = 0.0f64;
    let mut inside = 0.0f64;
    let mut inside_count = 0usize;
    let mut coverage_sum = 0.0f64;
    for (index, texel) in rendered.iter().enumerate() {
        let truth = reference[index];
        let coverage = covered.map_or(texel[3], |values| values[index]);
        let is_covered = coverage >= MOSTLY;
        coverage_sum += coverage as f64;
        for channel in 0..3 {
            let difference = (texel[channel] - truth[channel]) as f64;
            linear += difference * difference;
            let difference = (capture::linear_to_srgb(texel[channel])
                - capture::linear_to_srgb(truth[channel])) as f64;
            encoded += difference * difference;
            if is_covered {
                inside += difference * difference;
            }
        }
        inside_count += is_covered as usize;
    }
    let samples = (rendered.len() * 3) as f64;
    Score {
        linear_psnr: psnr(linear / samples),
        srgb_psnr: psnr(encoded / samples),
        coverage: coverage_sum / rendered.len() as f64,
        covered_srgb_psnr: if inside_count > 0 {
            psnr(inside / (inside_count * 3) as f64)
        } else {
            0.0
        },
    }
}

/// A GPU that renders one scene at one resolution, repeatedly.
pub struct Renderer {
    context: sync::Arc<gpu::Context>,
    texture: gpu::Texture,
    target: gpu::TextureView,
    readback: gpu::Buffer,
    encoder: gpu::CommandEncoder,
    staged_surfels: Vec<vol::relight::Surfel>,
    geometry_update_pending: bool,
    extent: gpu::Extent,
    width: usize,
    height: usize,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Result<Self, String> {
        if vol::gpu::access_disabled() {
            return Err("GPU access disabled by BLADE_VOLUME_DISABLE_GPU".to_string());
        }
        let context = scoring_context()?;
        let extent = gpu::Extent {
            width: width as u32,
            height: height as u32,
            depth: 1,
        };
        let format = gpu::TextureFormat::Rgba32Float;
        let texture = context.create_texture(gpu::TextureDesc {
            name: "inverse-score",
            format,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let target = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "inverse-score",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "inverse-score-readback",
            size: (width * height) as u64 * 16,
            memory: gpu::Memory::Download,
        });
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "inverse-score",
            buffer_count: 1,
            manual_barriers: false,
        });
        Ok(Self {
            context,
            texture,
            target,
            readback,
            encoder,
            staged_surfels: Vec::new(),
            geometry_update_pending: false,
            extent,
            width,
            height,
        })
    }

    pub fn device_name(&self) -> String {
        self.context.device_information().device_name.clone()
    }

    fn draw(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        camera: vol::CameraParams,
    ) -> Vec<[f32; 4]> {
        assert!(!self.geometry_update_pending);
        self.encoder.start();
        self.encoder.init_texture(self.texture);
        tracer.dispatch(
            &mut self.encoder,
            self.target,
            camera,
            [self.width as u32, self.height as u32],
        );
        if let mut pass = self.encoder.transfer("inverse-score-readback") {
            pass.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                self.width as u32 * 16,
                self.extent,
            );
        }
        let sync_point = self.context.submit(&mut self.encoder);
        assert!(self.context.wait_for(&sync_point, 60_000).unwrap());
        let count = self.width * self.height;
        unsafe { std::slice::from_raw_parts(self.readback.data() as *const [f32; 4], count) }
            .to_vec()
    }

    fn ensure_readback_frames(&mut self, frames: usize) {
        let needed = (self.width * self.height * frames) as u64 * 16;
        if self.readback.size() >= needed {
            return;
        }
        self.context.destroy_buffer(self.readback);
        self.readback = self.context.create_buffer(gpu::BufferDesc {
            name: "inverse-score-readback",
            size: needed,
            memory: gpu::Memory::Download,
        });
    }

    /// A tracer for one scene, ready to draw.
    ///
    /// The tracer starts and submits the encoder itself; wrapping the call in
    /// another start/submit pair submits it twice and the driver takes the
    /// process down with it.
    fn tracer(
        &mut self,
        scene: &Scene,
        diffuse_samples: u32,
        show_environment: bool,
    ) -> vol::gpu::RelightTracer {
        assert!(!self.geometry_update_pending);
        let specular = vol::relight::SpecularEnvironment::prefilter(
            &scene.environment,
            scene.environment.width,
            scene.environment.height,
        );
        vol::gpu::RelightTracer::new(
            &scene.model,
            &scene.environment,
            &specular,
            vol::gpu::RelightSettings {
                background_rgb: [0.0; 3],
                diffuse_samples,
                show_environment,
            },
            &self.context,
            &mut self.encoder,
        )
    }

    /// Draw a scene from a set of poses, with no reference to compare against.
    ///
    /// This is how a capture with known truth is made: the scene is the truth,
    /// and what comes out is what a camera would have recorded of it.
    pub fn render_views(
        &mut self,
        scene: &Scene,
        cameras: &[vol::CameraParams],
        diffuse_samples: u32,
        show_environment: bool,
    ) -> Vec<Vec<[f32; 4]>> {
        let mut tracer = self.tracer(scene, diffuse_samples, show_environment);
        let frames = self.render_prepared_views(&mut tracer, cameras);
        tracer.deinit(&self.context);
        frames
    }

    /// Build one scene tracer for repeated renders while its particle geometry
    /// is updated between calls.
    pub(crate) fn prepare_scene(
        &mut self,
        scene: &Scene,
        diffuse_samples: u32,
        show_environment: bool,
    ) -> vol::gpu::RelightTracer {
        self.tracer(scene, diffuse_samples, show_environment)
    }

    pub(crate) fn update_prepared_surfels(&mut self, surfels: &[vol::relight::Surfel]) {
        self.staged_surfels.clear();
        self.staged_surfels.extend_from_slice(surfels);
        self.geometry_update_pending = true;
    }

    pub(crate) fn update_prepared_surfel_geometry(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        surfels: &[vol::relight::Surfel],
    ) {
        assert!(!self.geometry_update_pending);
        tracer.update_surfels(surfels, &self.context, &mut self.encoder);
    }

    pub(crate) fn update_prepared_materials(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        materials: &[vol::relight::Material],
    ) {
        assert!(!self.geometry_update_pending);
        tracer.update_materials(materials, &self.context, &mut self.encoder);
    }

    fn render_prepared_flat(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        cameras: &[vol::CameraParams],
    ) -> &[[f32; 4]] {
        if cameras.is_empty() {
            assert!(!self.geometry_update_pending);
            return &[];
        }
        self.ensure_readback_frames(cameras.len());
        let frame_bytes = (self.width * self.height) as u64 * 16;
        self.encoder.start();
        if self.geometry_update_pending {
            tracer.record_surfels_update(&self.staged_surfels, &self.context, &mut self.encoder);
            self.geometry_update_pending = false;
        }
        self.encoder.init_texture(self.texture);
        for (index, &camera) in cameras.iter().enumerate() {
            tracer.dispatch(
                &mut self.encoder,
                self.target,
                camera,
                [self.width as u32, self.height as u32],
            );
            if let mut pass = self.encoder.transfer("inverse-score-batch-readback") {
                pass.copy_texture_to_buffer(
                    self.texture.into(),
                    self.readback.at(index as u64 * frame_bytes),
                    self.width as u32 * 16,
                    self.extent,
                );
            }
        }
        let sync_point = self.context.submit(&mut self.encoder);
        assert!(self.context.wait_for(&sync_point, 60_000).unwrap());
        unsafe {
            std::slice::from_raw_parts(
                self.readback.data() as *const [f32; 4],
                self.width * self.height * cameras.len(),
            )
        }
    }

    pub(crate) fn render_prepared_views(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        cameras: &[vol::CameraParams],
    ) -> Vec<Vec<[f32; 4]>> {
        let frame_pixels = self.width * self.height;
        self.render_prepared_flat(tracer, cameras)
            .chunks(frame_pixels)
            .map(<[_]>::to_vec)
            .collect()
    }

    pub(crate) fn prepared_srgb_loss(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        capture: &capture::Capture,
        indices: &[usize],
        cameras: &[vol::CameraParams],
    ) -> f64 {
        let frame_pixels = capture.width * capture.height;
        let pixels = self.render_prepared_flat(tracer, cameras);
        let mut error = 0.0f64;
        for (frame, &index) in pixels.chunks(frame_pixels).zip(indices) {
            for (rendered, reference) in frame.iter().zip(&capture.views[index].pixels) {
                for channel in 0..3 {
                    let difference = capture::linear_to_srgb(rendered[channel])
                        - capture::linear_to_srgb(reference[channel]);
                    error += (difference * difference) as f64;
                }
            }
        }
        error / (indices.len() * frame_pixels * 3) as f64
    }

    pub(crate) fn prepared_srgb_errors(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        capture: &capture::Capture,
        indices: &[usize],
        cameras: &[vol::CameraParams],
        coverage_weight: f32,
    ) -> Vec<f32> {
        let frame_pixels = capture.width * capture.height;
        let pixels = self.render_prepared_flat(tracer, cameras);
        let mut errors = Vec::with_capacity(pixels.len());
        for (frame, &index) in pixels.chunks(frame_pixels).zip(indices) {
            for (pixel, (rendered, reference)) in
                frame.iter().zip(&capture.views[index].pixels).enumerate()
            {
                let reference_coverage = capture.views[index].mask.as_ref().map(|mask| mask[pixel]);
                errors.push(srgb_error(
                    rendered,
                    reference,
                    reference_coverage,
                    coverage_weight,
                ));
            }
        }
        errors
    }

    pub(crate) fn destroy_prepared_scene(&mut self, mut tracer: vol::gpu::RelightTracer) {
        self.geometry_update_pending = false;
        tracer.deinit(&self.context);
    }

    /// Render the given views and score them against their photographs.
    ///
    /// Coverage comes directly from the production renderer's alpha channel;
    /// RGB is scored against a black background.
    pub fn score(
        &mut self,
        scene: &Scene,
        capture: &capture::Capture,
        indices: &[usize],
        diffuse_samples: u32,
        dump: Option<&path::Path>,
    ) -> Summary {
        self.score_splits(scene, capture, &[(indices, dump)], diffuse_samples)[0]
    }

    /// Score several view sets while reusing one scene tracer.
    ///
    /// Building the acceleration structure and prefiltering the environment
    /// do not depend on the camera split. Keeping them alive also guarantees
    /// every split is evaluated against exactly the same GPU scene state.
    pub fn score_splits(
        &mut self,
        scene: &Scene,
        capture: &capture::Capture,
        splits: &[(&[usize], Option<&path::Path>)],
        diffuse_samples: u32,
    ) -> Vec<Summary> {
        assert_eq!(capture.width, self.width);
        assert_eq!(capture.height, self.height);
        if splits.is_empty() {
            return Vec::new();
        }
        let mut tracer = self.tracer(scene, diffuse_samples, false);

        let summaries = splits
            .iter()
            .map(|&(indices, dump)| self.score_with_tracer(&mut tracer, capture, indices, dump))
            .collect();
        tracer.deinit(&self.context);
        summaries
    }

    fn score_with_tracer(
        &mut self,
        tracer: &mut vol::gpu::RelightTracer,
        capture: &capture::Capture,
        indices: &[usize],
        dump: Option<&path::Path>,
    ) -> Summary {
        let mut scores = Vec::new();
        let mut elapsed = std::time::Duration::ZERO;
        tracer.set_background([0.0; 3]);
        for &index in indices {
            let view = &capture.views[index];
            let started = std::time::Instant::now();
            let rendered = self.draw(tracer, view.camera);
            elapsed += started.elapsed();

            scores.push(compare_coverage(&rendered, &view.pixels, None));

            if let Some(directory) = dump {
                let stem = path::Path::new(&view.name)
                    .file_stem()
                    .map_or_else(|| view.name.clone(), |s| s.to_string_lossy().into_owned());
                save_rgba(
                    &directory.join(format!("{stem}-render.png")),
                    &rendered,
                    self.width,
                    self.height,
                );
                save_rgb(
                    &directory.join(format!("{stem}-photo.png")),
                    &view.pixels,
                    self.width,
                    self.height,
                );
            }
        }
        let per_frame_ms = elapsed.as_secs_f64() * 1000.0 / indices.len().max(1) as f64;
        Summary::of(&scores, per_frame_ms)
    }

    pub fn destroy(mut self) {
        assert!(!self.geometry_update_pending);
        self.context.destroy_buffer(self.readback);
        self.context.destroy_texture_view(self.target);
        self.context.destroy_texture(self.texture);
        self.context.destroy_command_encoder(&mut self.encoder);
    }
}

fn init_scoring_context() -> Result<sync::Arc<gpu::Context>, String> {
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        })
    }
    .map_err(|error| format!("no ray tracing context: {error:?}"))?;
    let information = context.device_information();
    if information.is_software_emulated {
        return Err(format!(
            "refusing to score on the software rasterizer '{}'",
            information.device_name
        ));
    }
    Ok(sync::Arc::new(context))
}

#[cfg(not(test))]
fn scoring_context() -> Result<sync::Arc<gpu::Context>, String> {
    init_scoring_context()
}

#[cfg(test)]
fn scoring_context() -> Result<sync::Arc<gpu::Context>, String> {
    static CONTEXT: sync::OnceLock<Result<sync::Arc<gpu::Context>, String>> = sync::OnceLock::new();
    CONTEXT.get_or_init(init_scoring_context).clone()
}

fn srgb_error(
    rendered: &[f32; 4],
    reference: &[f32; 3],
    reference_coverage: Option<f32>,
    coverage_weight: f32,
) -> f32 {
    let color = (0..3)
        .map(|channel| {
            let difference = capture::linear_to_srgb(rendered[channel])
                - capture::linear_to_srgb(reference[channel]);
            difference * difference
        })
        .sum::<f32>()
        / 3.0;
    let coverage = reference_coverage.map_or(0.0, |reference| {
        let difference = rendered[3] - reference;
        coverage_weight * difference * difference
    });
    color + coverage
}

pub fn save_rgba(path: &path::Path, pixels: &[[f32; 4]], width: usize, height: usize) {
    let mut image = image::RgbImage::new(width as u32, height as u32);
    for (index, texel) in pixels.iter().enumerate() {
        image.put_pixel(
            (index % width) as u32,
            (index / width) as u32,
            image::Rgb([
                (capture::linear_to_srgb(texel[0]) * 255.0).round() as u8,
                (capture::linear_to_srgb(texel[1]) * 255.0).round() as u8,
                (capture::linear_to_srgb(texel[2]) * 255.0).round() as u8,
            ]),
        );
    }
    let _ = image.save(path);
}

pub fn save_rgb(path: &path::Path, pixels: &[[f32; 3]], width: usize, height: usize) {
    let mut image = image::RgbImage::new(width as u32, height as u32);
    for (index, texel) in pixels.iter().enumerate() {
        image.put_pixel(
            (index % width) as u32,
            (index / width) as u32,
            image::Rgb([
                (capture::linear_to_srgb(texel[0]) * 255.0).round() as u8,
                (capture::linear_to_srgb(texel[1]) * 255.0).round() as u8,
                (capture::linear_to_srgb(texel[2]) * 255.0).round() as u8,
            ]),
        );
    }
    let _ = image.save(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_reproduction_scores_at_the_ceiling() {
        let reference = vec![[0.2, 0.5, 0.8]; 16];
        let rendered: Vec<[f32; 4]> = reference.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
        let score = compare(&rendered, &reference, &[1.0; 16]);
        assert!(score.linear_psnr > 90.0 && score.srgb_psnr > 90.0);
        assert!((score.coverage - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn coverage_comes_from_rendered_alpha() {
        let reference = vec![[0.2, 0.3, 0.4]; 2];
        let rendered = vec![[0.2, 0.3, 0.4, 0.25], [0.2, 0.3, 0.4, 0.75]];
        let score = compare_coverage(&rendered, &reference, None);
        assert!((score.coverage - 0.5).abs() < 1.0e-6);
        assert!(score.covered_srgb_psnr > 90.0);
    }

    #[test]
    fn masked_render_error_penalizes_wrong_coverage() {
        let rendered = [0.2, 0.3, 0.4, 0.25];
        let reference = [0.2, 0.3, 0.4];
        assert_eq!(srgb_error(&rendered, &reference, None, 0.1), 0.0);
        assert!((srgb_error(&rendered, &reference, Some(0.75), 0.1) - 0.025).abs() < 1.0e-6);
    }

    #[test]
    fn the_encoded_score_is_the_harsher_one_in_the_shadows() {
        // A tenth of a stop of error on a dark surface is invisible in linear
        // radiance and obvious on a display. Reporting only the linear number
        // would call a black-crushed reconstruction excellent.
        let reference = vec![[0.01, 0.01, 0.01]; 64];
        let rendered = vec![[0.02, 0.02, 0.02, 1.0]; 64];
        let score = compare(&rendered, &reference, &[1.0; 64]);
        assert!(
            score.srgb_psnr < score.linear_psnr - 10.0,
            "linear {:.1}, encoded {:.1}",
            score.linear_psnr,
            score.srgb_psnr
        );
    }
}
