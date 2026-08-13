//! A posed capture: the images an inverse renderer has to explain.
//!
//! Everything here works in **linear** radiance. The files on disk are
//! display-encoded, and fitting a physical material against display-encoded
//! values recovers a material that is wrong by a power of 2.2 — an error that
//! looks like a plausible albedo and is not one. The decode happens once, on
//! load, and nothing downstream has to remember.

use crate::{colmap, pipeline, relight};
use blade_volume as vol;
use std::{path, thread};

/// One image and the camera that took it.
pub struct View {
    /// The file it came from, so a dumped render can be matched to its source.
    pub name: String,
    pub camera: vol::CameraParams,
    /// Linear radiance, row major, `width * height` texels.
    pub pixels: Vec<[f32; 3]>,
    /// Optional foreground coverage, row major. Values are in `[0, 1]`.
    pub mask: Option<Vec<f32>>,
}

impl View {
    pub fn is_foreground(&self, pixel: usize) -> bool {
        self.mask.as_ref().is_none_or(|mask| mask[pixel] > 0.5)
    }
}

/// Every posed image of one scene, at one resolution.
pub struct Capture {
    pub width: usize,
    pub height: usize,
    pub views: Vec<View>,
}

/// The display transfer function, and its inverse.
///
/// sRGB, not a 2.2 power law: the linear toe near black is where a dark
/// surface's albedo lives, and the two differ by 8 % there.
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn decode_pixels(encoded: &[f32], mask: Option<&[f32]>) -> Vec<[f32; 3]> {
    encoded
        .chunks_exact(3)
        .enumerate()
        .map(|(index, rgb)| {
            let coverage = mask.map_or(1.0, |mask| mask[index]);
            [
                coverage * srgb_to_linear(rgb[0]),
                coverage * srgb_to_linear(rgb[1]),
                coverage * srgb_to_linear(rgb[2]),
            ]
        })
        .collect()
}

/// Where a world point lands in a view, in pixels, and how far away it is.
///
/// The inverse of the ray generation every backend shares: a local direction
/// of `((ndc - principal) * tan(fov/2), 1)` rotated by the camera orientation.
/// Points behind the camera return `None` rather than a mirrored position.
pub fn project(
    camera: &vol::CameraParams,
    width: usize,
    height: usize,
    point: glam::Vec3,
) -> Option<([f32; 2], f32)> {
    let orientation = glam::Quat::from_array(camera.cam_orientation);
    let local = orientation.inverse() * (point - glam::Vec3::from(camera.cam_position));
    if local.z <= 1.0e-6 {
        return None;
    }
    let tan_half = glam::Vec2::new((0.5 * camera.fov[0]).tan(), (0.5 * camera.fov[1]).tan());
    let ndc = glam::Vec2::new(local.x, local.y) / (local.z * tan_half)
        + glam::Vec2::from(camera.principal);
    let pixel = [
        0.5 * (ndc.x + 1.0) * width as f32,
        0.5 * (ndc.y + 1.0) * height as f32,
    ];
    Some((pixel, local.z))
}

/// The direction the ray through a pixel centre travels, in world space.
pub fn pixel_direction(
    camera: &vol::CameraParams,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> glam::Vec3 {
    PixelRays::new(camera, width, height).direction(x, y)
}

/// Camera terms shared by every ray in one image.
///
/// Kept crate-private because callers generally need only one direction. The
/// depth extractor needs every direction and must not rebuild the quaternion
/// or evaluate the field-of-view tangents for every pixel.
#[derive(Clone, Copy)]
pub(crate) struct PixelRays {
    orientation: glam::Quat,
    tan_half: glam::Vec2,
    principal: glam::Vec2,
    width: f32,
    height: f32,
}

impl PixelRays {
    pub(crate) fn new(camera: &vol::CameraParams, width: usize, height: usize) -> Self {
        Self {
            orientation: glam::Quat::from_array(camera.cam_orientation),
            tan_half: glam::Vec2::new((0.5 * camera.fov[0]).tan(), (0.5 * camera.fov[1]).tan()),
            principal: glam::Vec2::from(camera.principal),
            width: width as f32,
            height: height as f32,
        }
    }

    pub(crate) fn direction(&self, x: usize, y: usize) -> glam::Vec3 {
        let ndc = glam::Vec2::new(
            (x as f32 + 0.5) / self.width * 2.0 - 1.0,
            (y as f32 + 0.5) / self.height * 2.0 - 1.0,
        );
        let local = (ndc - self.principal) * self.tan_half;
        (self.orientation * glam::Vec3::new(local.x, local.y, 1.0)).normalize()
    }
}

impl Capture {
    /// Load one environment of Blade's synthetic relighting dataset.
    ///
    /// When `foreground_only` is set, pixels whose geometry plane records a
    /// miss become black. That lets reconstruction scoring distinguish missing
    /// point-cloud geometry from the unrelated task of reproducing a visible
    /// environment background.
    pub fn from_relight_dataset(
        dataset: &relight::Dataset,
        environment: usize,
        foreground_only: bool,
    ) -> Result<Self, String> {
        let Some(environment_name) = dataset.environments.get(environment) else {
            return Err(format!(
                "environment {environment} is outside a dataset with {} environments",
                dataset.environments.len()
            ));
        };
        let mut views = Vec::with_capacity(dataset.views.len());
        for view in &dataset.views {
            let radiance = dataset.read_plane(&view.radiance[environment])?;
            let geometry = if foreground_only {
                Some(dataset.read_plane(&view.geometry)?)
            } else {
                None
            };
            let mask: Option<Vec<f32>> = geometry.as_ref().map(|plane| {
                plane
                    .iter()
                    .map(|texel| if texel[3] > 0.0 { 1.0 } else { 0.0 })
                    .collect()
            });
            let pixels = radiance
                .iter()
                .enumerate()
                .map(|(index, texel)| {
                    if mask.as_ref().is_some_and(|mask| mask[index] == 0.0) {
                        [0.0; 3]
                    } else {
                        [texel[0], texel[1], texel[2]]
                    }
                })
                .collect();
            views.push(View {
                name: format!("view_{:03}-{environment_name}", view.index),
                camera: relight::camera_params(view, dataset.width, dataset.height),
                pixels,
                mask,
            });
        }
        Ok(Self {
            width: dataset.width,
            height: dataset.height,
            views,
        })
    }

    /// Read a COLMAP reconstruction and the images it poses.
    ///
    /// `stride` keeps every n-th image; the full mip-NeRF scenes are 290
    /// views of one room and a solver does not need all of them to see every
    /// surface. Images are rectified onto the calibrated pinhole plane, so a
    /// distorted camera still supervises the right pixel.
    pub fn from_colmap(
        sparse: &path::Path,
        images: &path::Path,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(Self, colmap::Reconstruction), String> {
        Self::from_colmap_with_masks(sparse, images, None, width, height, stride)
    }

    /// Read a COLMAP reconstruction, its images, and optional foreground masks.
    ///
    /// The mask directory mirrors the image directory: a view named
    /// `room/frame.png` reads `masks/room/frame.png`. Masks are rectified onto
    /// the same pinhole plane as their photographs. Missing masks are errors;
    /// silently mixing masked and unmasked views would change the objective.
    pub fn from_colmap_with_masks(
        sparse: &path::Path,
        images: &path::Path,
        masks: Option<&path::Path>,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(Self, colmap::Reconstruction), String> {
        let reconstruction = colmap::try_load_reconstruction(sparse)
            .map_err(|e| format!("cannot read {}: {e}", sparse.display()))?;
        if reconstruction.images.is_empty() {
            return Err(format!("{} poses no images", sparse.display()));
        }
        // Sorted by name so a strided subset and an every-n-th test split mean
        // the same thing between runs and between scenes.
        let mut ordered = reconstruction.images.clone();
        ordered.sort_by(|a, b| a.name.cmp(&b.name));

        let far = far_plane(&reconstruction);
        let mut inputs = Vec::new();
        for image in ordered.iter().step_by(stride.max(1)) {
            let camera = reconstruction
                .cameras
                .get(&image.camera_id)
                .ok_or_else(|| format!("image {} has no camera", image.name))?;
            let path = images.join(&image.name);
            // A pose without its photograph is not an error: the small test
            // fixtures ship a full reconstruction next to a subset of the
            // frames, and refusing to open them would be refusing to run on
            // the only data that is checked out.
            if !path.exists() {
                continue;
            }
            let mask = masks.map(|directory| directory.join(&image.name));
            if let Some(ref mask) = mask {
                if !mask.exists() {
                    return Err(format!(
                        "cannot read mask {}: file does not exist",
                        mask.display()
                    ));
                }
            }
            inputs.push((image, camera, path, mask));
        }
        let mut views = Vec::with_capacity(inputs.len());
        if !inputs.is_empty() {
            // Decoding and rectification dominate capture loading. Keep the
            // pool bounded because every worker temporarily holds a full-size
            // source image as well as the resized result.
            let workers = thread::available_parallelism()
                .map_or(1, |count| count.get())
                .min(8)
                .min(inputs.len());
            let chunk_size = inputs.len().div_ceil(workers);
            thread::scope(|scope| -> Result<(), String> {
                let mut tasks = Vec::with_capacity(workers);
                for chunk in inputs.chunks(chunk_size) {
                    let reconstruction = &reconstruction;
                    tasks.push(scope.spawn(move || {
                        let mut loaded = Vec::with_capacity(chunk.len());
                        for input in chunk {
                            let image = input.0;
                            let camera = input.1;
                            let path = &input.2;
                            let encoded = pipeline::load_and_rectify_image(
                                path,
                                width as u32,
                                height as u32,
                                camera,
                            )
                            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
                            let mask = input
                                .3
                                .as_ref()
                                .map(|mask_path| {
                                    pipeline::load_and_rectify_image(
                                        mask_path,
                                        width as u32,
                                        height as u32,
                                        camera,
                                    )
                                    .map_err(|error| {
                                        format!("cannot read mask {}: {error}", mask_path.display())
                                    })
                                    .map(|encoded| {
                                        encoded
                                            .chunks_exact(3)
                                            .map(|rgb| rgb[0].clamp(0.0, 1.0))
                                            .collect::<Vec<_>>()
                                    })
                                })
                                .transpose()?;
                            let pixels = decode_pixels(&encoded, mask.as_deref());
                            loaded.push(View {
                                name: image.name.clone(),
                                camera: reconstruction.camera_params_for(image, far),
                                pixels,
                                mask,
                            });
                        }
                        Ok::<Vec<View>, String>(loaded)
                    }));
                }
                for task in tasks {
                    views.extend(task.join().expect("capture worker panicked")?);
                }
                Ok(())
            })?;
        }
        Ok((
            Self {
                width,
                height,
                views,
            },
            reconstruction,
        ))
    }

    /// Indices of the views held out of the fit, and the ones kept.
    ///
    /// Every eighth image, which is what the mip-NeRF 360 protocol uses and
    /// what published numbers on these scenes are measured against.
    pub fn split(&self, every: usize) -> (Vec<usize>, Vec<usize>) {
        let mut train = Vec::new();
        let mut test = Vec::new();
        for index in 0..self.views.len() {
            if every > 0 && index % every == every - 1 {
                test.push(index);
            } else {
                train.push(index);
            }
        }
        (train, test)
    }
}

/// A far plane that contains the reconstruction with room to spare.
///
/// The sparse cloud has outliers far outside the room, so this uses a
/// percentile of the distance from the centroid rather than the maximum: one
/// stray point kilometres away would otherwise push the far plane out until
/// depth resolution near the scene is gone.
fn far_plane(reconstruction: &colmap::Reconstruction) -> f32 {
    let points = &reconstruction.points;
    if points.is_empty() {
        return 1000.0;
    }
    let mut centroid = glam::DVec3::ZERO;
    for point in points {
        centroid += glam::DVec3::from_array(point.xyz);
    }
    centroid /= points.len() as f64;
    let mut distances: Vec<f64> = points
        .iter()
        .map(|p| (glam::DVec3::from_array(p.xyz) - centroid).length())
        .collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = distances[distances.len() * 99 / 100];
    (8.0 * percentile).max(1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> vol::CameraParams {
        vol::CameraParams {
            cam_position: [1.0, 2.0, -3.0],
            depth: 100.0,
            cam_orientation: glam::Quat::from_rotation_y(0.7).to_array(),
            fov: [1.2, 0.9],
            principal: [0.05, -0.02],
        }
    }

    #[test]
    fn projecting_a_pixels_own_ray_returns_that_pixel() {
        // The two directions have to be exact inverses, or a surfel is fitted
        // against the colour of a different part of the image — which reads as
        // a bad material rather than as a broken projection.
        let camera = camera();
        let (width, height) = (64usize, 48usize);
        for &(x, y) in &[(0usize, 0usize), (31, 17), (63, 47), (7, 40)] {
            let direction = pixel_direction(&camera, width, height, x, y);
            let point = glam::Vec3::from(camera.cam_position) + 5.0 * direction;
            let (pixel, distance) = project(&camera, width, height, point).unwrap();
            assert!(
                (pixel[0] - (x as f32 + 0.5)).abs() < 1.0e-3
                    && (pixel[1] - (y as f32 + 0.5)).abs() < 1.0e-3,
                "pixel ({x}, {y}) came back as {pixel:?}"
            );
            assert!(distance > 0.0);
        }
    }

    #[test]
    fn a_point_behind_the_camera_does_not_project() {
        let camera = camera();
        let behind = glam::Vec3::from(camera.cam_position)
            - 5.0 * (glam::Quat::from_array(camera.cam_orientation) * glam::Vec3::Z);
        assert!(project(&camera, 32, 32, behind).is_none());
    }

    #[test]
    fn the_transfer_function_round_trips() {
        for step in 0..=64 {
            let value = step as f32 / 64.0;
            assert!((linear_to_srgb(srgb_to_linear(value)) - value).abs() < 1.0e-4);
        }
        // And it is the sRGB one rather than a 2.2 power law: they disagree
        // most in the toe, which is exactly where dark albedo is decided.
        assert!((srgb_to_linear(0.05) - 0.05f32.powf(2.2)).abs() > 1.0e-3);
    }

    #[test]
    fn mask_coverage_composites_linear_radiance_over_black() {
        let encoded = [1.0, 0.5, 0.0, 0.25, 0.75, 1.0];
        let unmasked = decode_pixels(&encoded, None);
        let masked = decode_pixels(&encoded, Some(&[1.0, 0.25]));
        assert_eq!(masked[0], unmasked[0]);
        for channel in 0..3 {
            assert!((masked[1][channel] - 0.25 * unmasked[1][channel]).abs() < 1.0e-7);
        }

        let view = View {
            name: "mask".to_string(),
            camera: camera(),
            pixels: masked,
            mask: Some(vec![0.5, 0.500_1]),
        };
        assert!(!view.is_foreground(0));
        assert!(view.is_foreground(1));
    }
}
