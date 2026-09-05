//! Diffuse material fitting through the surface-particle blend.
//!
//! A rendered pixel usually averages many overlapping surfels. Assigning that
//! complete pixel to every surfel independently therefore fits the average
//! colour back into all of them. With fixed geometry, normals, and calibrated
//! point lights, the runtime compact-disc response is linear in diffuse
//! albedo. This module builds those sparse pixel equations on the host and
//! solves all materials together.

use blade_volume as vol;

use crate::inverse::capture;

const RIDGE: f64 = 0.1;
const SOLVER_ITERATIONS: usize = 20;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Refinement {
    pub equations: usize,
    pub terms: usize,
    pub supported: usize,
    pub changed: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
}

#[derive(Clone, Copy)]
struct Hit {
    surfel: usize,
    depth: f32,
    coverage: f32,
    point: glam::Vec3,
    normal: glam::Vec3,
}

#[derive(Clone, Copy)]
struct BlendTerm {
    material: u32,
    weight: f32,
    point: glam::Vec3,
    normal: glam::Vec3,
}

struct PixelBlend {
    view: usize,
    pixel: usize,
    terms: Vec<BlendTerm>,
}

#[derive(Clone, Copy)]
struct EquationTerm {
    material: u32,
    response: [f32; 3],
}

struct Equation {
    target: [f32; 3],
    terms: Vec<EquationTerm>,
}

/// Re-solve diffuse albedos against complete foreground pixels and calibrated
/// point lights while preserving geometry, normals, and material assignment.
pub fn refine_materials_point_lights(
    model: &mut vol::relight::RelightModel,
    captures: &[capture::Capture],
    lights: &[Vec<vol::relight::PointLight>],
    views: &[usize],
    ceiling: f32,
) -> Result<Refinement, String> {
    model.validate()?;
    if captures.len() != lights.len() {
        return Err("blended material captures and lights must have equal lengths".to_string());
    }
    if !ceiling.is_finite() || ceiling <= 0.0 {
        return Err("blended material ceiling must be finite and positive".to_string());
    }
    if captures.is_empty() || views.is_empty() || model.materials.is_empty() {
        return Ok(Refinement::default());
    }
    let reference = &captures[0];
    for (capture, lights) in captures.iter().zip(lights) {
        if capture.width != reference.width
            || capture.height != reference.height
            || capture.views.len() != reference.views.len()
        {
            return Err("blended material captures must be camera-aligned".to_string());
        }
        if lights.len() != capture.views.len() {
            return Err("blended material lights must match capture views".to_string());
        }
        if capture
            .views
            .iter()
            .zip(&reference.views)
            .any(|(view, reference)| {
                !same_camera(&view.camera, &reference.camera) || view.mask != reference.mask
            })
        {
            return Err(
                "blended material captures must have aligned cameras and masks".to_string(),
            );
        }
        if let Some(&view) = views.iter().find(|&&view| view >= capture.views.len()) {
            return Err(format!("blended material view {view} is out of bounds"));
        }
    }

    let blends = pixel_blends(model, reference, views);
    let equations = build_equations(captures, lights, &blends);
    let initial_loss = loss(model, &equations);
    let original = model.materials.clone();
    let mut supported = vec![false; model.materials.len()];
    for channel in 0..3 {
        let initial: Vec<_> = model
            .materials
            .iter()
            .map(|material| material.albedo[channel] as f64)
            .collect();
        let (solved, channel_supported) =
            solve_channel(&equations, channel, &initial, ceiling as f64);
        for (index, material) in model.materials.iter_mut().enumerate() {
            material.albedo[channel] = solved[index] as f32;
            supported[index] |= channel_supported[index];
        }
    }
    let candidate_loss = loss(model, &equations);
    let (changed, final_loss) = if candidate_loss < initial_loss {
        (
            original
                .iter()
                .zip(&model.materials)
                .filter(|&(before, after)| before.albedo != after.albedo)
                .count(),
            candidate_loss,
        )
    } else {
        model.materials = original;
        (0, initial_loss)
    };
    Ok(Refinement {
        equations: equations.len(),
        terms: equations.iter().map(|equation| equation.terms.len()).sum(),
        supported: supported.into_iter().filter(|&value| value).count(),
        changed,
        initial_loss,
        final_loss,
    })
}

fn same_camera(left: &vol::CameraParams, right: &vol::CameraParams) -> bool {
    left.cam_position == right.cam_position
        && left.depth == right.depth
        && left.cam_orientation == right.cam_orientation
        && left.fov == right.fov
        && left.principal == right.principal
}

fn pixel_blends(
    model: &vol::relight::RelightModel,
    capture: &capture::Capture,
    views: &[usize],
) -> Vec<PixelBlend> {
    let (width, height) = (capture.width, capture.height);
    let mut out = Vec::new();
    for &view_index in views {
        let view = &capture.views[view_index];
        let origin = glam::Vec3::from(view.camera.cam_position);
        let focal = (0.5 * width as f32 / (0.5 * view.camera.fov[0]).tan())
            .max(0.5 * height as f32 / (0.5 * view.camera.fov[1]).tan());
        let directions: Vec<_> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| capture::pixel_direction(&view.camera, width, height, x, y))
            })
            .collect();
        let mut hits = vec![Vec::<Hit>::new(); width * height];
        for (index, surfel) in model.surfels.iter().enumerate() {
            let center = glam::Vec3::from(surfel.center);
            let normal = glam::Vec3::from(surfel.normal);
            let Some((pixel, distance)) = capture::project(&view.camera, width, height, center)
            else {
                continue;
            };
            // A projected bounding sphere conservatively contains the disc,
            // including an oblique one whose ellipse does not follow its
            // centre normal in screen space.
            let radius = surfel.radius * focal / (distance - surfel.radius).max(1.0e-3);
            let min_x = ((pixel[0] - radius - 1.0).floor() as isize).max(0) as usize;
            let min_y = ((pixel[1] - radius - 1.0).floor() as isize).max(0) as usize;
            let max_x = ((pixel[0] + radius + 1.0).ceil() as isize).min(width as isize - 1);
            let max_y = ((pixel[1] + radius + 1.0).ceil() as isize).min(height as isize - 1);
            if max_x < 0 || max_y < 0 {
                continue;
            }
            for y in min_y..=max_y as usize {
                for x in min_x..=max_x as usize {
                    let pixel_index = y * width + x;
                    if !view.is_foreground(pixel_index) {
                        continue;
                    }
                    let direction = directions[pixel_index];
                    let denominator = direction.dot(normal);
                    if denominator.abs() <= 1.0e-6 {
                        continue;
                    }
                    let depth = (center - origin).dot(normal) / denominator;
                    if depth <= 0.0 || depth >= view.camera.depth {
                        continue;
                    }
                    let point = origin + depth * direction;
                    let normalized = point.distance_squared(center) / surfel.radius.powi(2);
                    let coverage = vol::relight::particle_coverage(model.kernel, normalized);
                    if coverage <= 0.0 {
                        continue;
                    }
                    hits[pixel_index].push(Hit {
                        surfel: index,
                        depth,
                        coverage,
                        point,
                        normal: if normal.dot(direction) > 0.0 {
                            -normal
                        } else {
                            normal
                        },
                    });
                }
            }
        }

        for (pixel, mut pixel_hits) in hits.into_iter().enumerate() {
            if pixel_hits.is_empty() {
                continue;
            }
            pixel_hits.sort_unstable_by(|left, right| {
                left.depth
                    .total_cmp(&right.depth)
                    .then_with(|| left.surfel.cmp(&right.surfel))
            });
            let mut terms = Vec::with_capacity(pixel_hits.len());
            let mut transmittance = 1.0f32;
            let mut begin = 0;
            while begin < pixel_hits.len() && transmittance > 0.003 {
                let limit = pixel_hits[begin].depth
                    + vol::relight::SURFACE_BAND * model.surfels[pixel_hits[begin].surfel].radius;
                let mut end = begin + 1;
                while end < pixel_hits.len() && pixel_hits[end].depth <= limit {
                    end += 1;
                }
                let sum = pixel_hits[begin..end]
                    .iter()
                    .map(|hit| hit.coverage)
                    .sum::<f32>();
                let alpha = sum.min(1.0);
                for hit in &pixel_hits[begin..end] {
                    terms.push(BlendTerm {
                        material: model.surfels[hit.surfel].material,
                        weight: transmittance * alpha * hit.coverage / sum,
                        point: hit.point,
                        normal: hit.normal,
                    });
                }
                transmittance *= 1.0 - alpha;
                begin = end;
            }
            out.push(PixelBlend {
                view: view_index,
                pixel,
                terms,
            });
        }
    }
    out
}

fn build_equations(
    captures: &[capture::Capture],
    lights: &[Vec<vol::relight::PointLight>],
    blends: &[PixelBlend],
) -> Vec<Equation> {
    let mut out = Vec::with_capacity(blends.len() * captures.len());
    for (capture, lights) in captures.iter().zip(lights) {
        for blend in blends {
            let light = lights[blend.view];
            out.push(Equation {
                target: capture.views[blend.view].pixels[blend.pixel],
                terms: blend
                    .terms
                    .iter()
                    .filter_map(|term| {
                        let response = light
                            .diffuse(term.point, term.normal)
                            .map(|value| term.weight * value);
                        (response != [0.0; 3]).then_some(EquationTerm {
                            material: term.material,
                            response,
                        })
                    })
                    .collect(),
            });
        }
    }
    out
}

fn loss(model: &vol::relight::RelightModel, equations: &[Equation]) -> f64 {
    let squared = equations
        .iter()
        .map(|equation| {
            let mut predicted = [0.0f64; 3];
            for term in &equation.terms {
                let albedo = model.materials[term.material as usize].albedo;
                for channel in 0..3 {
                    predicted[channel] += term.response[channel] as f64 * albedo[channel] as f64;
                }
            }
            predicted
                .into_iter()
                .zip(equation.target)
                .map(|(predicted, target)| (predicted - target as f64).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    squared / (3 * equations.len()).max(1) as f64
}

fn solve_channel(
    equations: &[Equation],
    channel: usize,
    initial: &[f64],
    ceiling: f64,
) -> (Vec<f64>, Vec<bool>) {
    let count = initial.len();
    let mut right = vec![0.0f64; count];
    let mut diagonal = vec![RIDGE; count];
    for equation in equations {
        let current = equation
            .terms
            .iter()
            .map(|term| term.response[channel] as f64 * initial[term.material as usize])
            .sum::<f64>();
        let residual = equation.target[channel] as f64 - current;
        for term in &equation.terms {
            let index = term.material as usize;
            let response = term.response[channel] as f64;
            right[index] += response * residual;
            diagonal[index] += response * response;
        }
    }
    let supported: Vec<_> = diagonal.iter().map(|&value| value > RIDGE).collect();
    let mut delta = vec![0.0f64; count];
    let mut residual = right;
    let mut direction: Vec<_> = residual
        .iter()
        .zip(&diagonal)
        .map(|(&value, &diagonal)| value / diagonal)
        .collect();
    let mut rz = dot(&residual, &direction);
    let mut product = vec![0.0f64; count];
    for _ in 0..SOLVER_ITERATIONS {
        matrix_vector(equations, channel, &direction, &mut product);
        let denominator = dot(&direction, &product);
        if denominator <= 1.0e-20 || rz <= 1.0e-20 {
            break;
        }
        let alpha = rz / denominator;
        for index in 0..count {
            delta[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        let next: Vec<_> = residual
            .iter()
            .zip(&diagonal)
            .map(|(&value, &diagonal)| value / diagonal)
            .collect();
        let next_rz = dot(&residual, &next);
        let beta = next_rz / rz;
        for index in 0..count {
            direction[index] = next[index] + beta * direction[index];
        }
        rz = next_rz;
    }
    let solved = initial
        .iter()
        .zip(delta)
        .map(|(&initial, delta)| (initial + delta).clamp(0.005, ceiling))
        .collect();
    (solved, supported)
}

fn matrix_vector(equations: &[Equation], channel: usize, input: &[f64], output: &mut [f64]) {
    for (value, &input) in output.iter_mut().zip(input) {
        *value = RIDGE * input;
    }
    for equation in equations {
        let prediction = equation
            .terms
            .iter()
            .map(|term| term.response[channel] as f64 * input[term.material as usize])
            .sum::<f64>();
        for term in &equation.terms {
            output[term.material as usize] += term.response[channel] as f64 * prediction;
        }
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> vol::CameraParams {
        vol::CameraParams {
            cam_position: [0.0; 3],
            depth: 10.0,
            cam_orientation: glam::Quat::IDENTITY.to_array(),
            fov: [1.0; 2],
            principal: [0.0; 2],
        }
    }

    fn model(albedos: [[f32; 3]; 2]) -> vol::relight::RelightModel {
        vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Compact,
            surfels: vec![
                vol::relight::Surfel {
                    center: [-0.05, 0.0, 2.0],
                    radius: 0.7,
                    normal: glam::Vec3::new(-0.35, 0.0, -1.0).normalize().to_array(),
                    material: 0,
                },
                vol::relight::Surfel {
                    center: [0.05, 0.0, 2.0],
                    radius: 0.7,
                    normal: glam::Vec3::new(0.35, 0.0, -1.0).normalize().to_array(),
                    material: 1,
                },
            ],
            materials: albedos
                .into_iter()
                .map(|albedo| vol::relight::Material {
                    albedo,
                    specular_f0: [0.0; 3],
                    ..vol::relight::Material::default()
                })
                .collect(),
        }
    }

    #[test]
    fn coupled_pixels_recover_two_materials() {
        let truth = model([[0.8, 0.2, 0.4], [0.2, 0.7, 0.5]]);
        let blank = capture::Capture {
            width: 8,
            height: 8,
            views: vec![capture::View {
                name: "view".to_string(),
                camera: camera(),
                pixels: vec![[0.0; 3]; 64],
                mask: Some(vec![1.0; 64].into()),
            }],
        };
        let blends = pixel_blends(&truth, &blank, &[0]);
        let positions = [
            glam::Vec3::new(-5.0, 0.0, 0.0),
            glam::Vec3::new(5.0, 0.0, 0.0),
            glam::Vec3::new(0.0, 5.0, 0.0),
        ];
        let mut captures = Vec::new();
        let mut lights = Vec::new();
        for position in positions {
            let light = vol::relight::PointLight {
                position: position.to_array(),
                direction: [0.0, 0.0, 1.0],
                intensity: [29.0; 3],
                exponent: 0.0,
            };
            let mut capture = capture::Capture {
                width: blank.width,
                height: blank.height,
                views: vec![capture::View {
                    name: "view".to_string(),
                    camera: camera(),
                    pixels: vec![[0.0; 3]; 64],
                    mask: Some(vec![1.0; 64].into()),
                }],
            };
            for blend in &blends {
                for term in &blend.terms {
                    let diffuse = light.diffuse(term.point, term.normal);
                    let albedo = truth.materials[term.material as usize].albedo;
                    for channel in 0..3 {
                        capture.views[0].pixels[blend.pixel][channel] +=
                            term.weight * diffuse[channel] * albedo[channel];
                    }
                }
            }
            captures.push(capture);
            lights.push(vec![light]);
        }

        let mut candidate = model([[0.5; 3]; 2]);
        let stats =
            refine_materials_point_lights(&mut candidate, &captures, &lights, &[0], 1.0).unwrap();
        assert_eq!(stats.supported, 2);
        assert_eq!(stats.changed, 2);
        assert!(stats.final_loss < 0.02 * stats.initial_loss, "{stats:?}");
        for (actual, expected) in candidate.materials.iter().zip(&truth.materials) {
            for (&actual, &expected) in actual.albedo.iter().zip(&expected.albedo) {
                assert!((actual - expected).abs() < 0.03, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn equations_discard_exactly_unlit_terms() {
        let capture = capture::Capture {
            width: 1,
            height: 1,
            views: vec![capture::View {
                name: "view".to_string(),
                camera: camera(),
                pixels: vec![[0.0; 3]],
                mask: Some(vec![1.0].into()),
            }],
        };
        let light = vol::relight::PointLight {
            position: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            intensity: [1.0; 3],
            exponent: 0.0,
        };
        let blends = [PixelBlend {
            view: 0,
            pixel: 0,
            terms: vec![
                BlendTerm {
                    material: 0,
                    weight: 1.0,
                    point: glam::Vec3::new(0.0, 0.0, 2.0),
                    normal: glam::Vec3::NEG_Z,
                },
                BlendTerm {
                    material: 1,
                    weight: 1.0,
                    point: glam::Vec3::new(0.0, 0.0, 2.0),
                    normal: glam::Vec3::Z,
                },
            ],
        }];

        let equations = build_equations(&[capture], &[vec![light]], &blends);

        assert_eq!(equations.len(), 1);
        assert_eq!(equations[0].terms.len(), 1);
        assert_eq!(equations[0].terms[0].material, 0);
    }

    #[test]
    fn sparse_equations_match_the_production_surface_blend() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Ok(mut renderer) = crate::inverse::score::Renderer::new(8, 8) else {
            eprintln!("skipping production-blend parity test: no ray-tracing GPU");
            return;
        };
        let truth = model([[0.8, 0.2, 0.4], [0.2, 0.7, 0.5]]);
        let light = vol::relight::PointLight {
            position: [-5.0, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            intensity: [29.0; 3],
            exponent: 0.0,
        };
        let mut capture = capture::Capture {
            width: 8,
            height: 8,
            views: vec![capture::View {
                name: "view".to_string(),
                camera: camera(),
                pixels: vec![[0.0; 3]; 64],
                mask: Some(vec![1.0; 64].into()),
            }],
        };
        for blend in pixel_blends(&truth, &capture, &[0]) {
            for term in blend.terms {
                let diffuse = light.diffuse(term.point, term.normal);
                let albedo = truth.materials[term.material as usize].albedo;
                for channel in 0..3 {
                    capture.views[0].pixels[blend.pixel][channel] +=
                        term.weight * diffuse[channel] * albedo[channel];
                }
            }
        }
        let scene = crate::inverse::score::Scene::new(
            truth,
            vol::relight::Environment::uniform([0.0; 3], 8, 4),
        );
        let summary = renderer.score_point_lights(&scene, &capture, &[light], &[0], false, None);
        assert!(summary.linear_psnr > 90.0, "{summary:?}");
        renderer.destroy();
    }
}
