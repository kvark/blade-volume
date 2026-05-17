//! Differentiable per-pixel volumetric integration in meganeura.
//!
//! Given a stack of per-pixel paths recorded by
//! [`vol::trace::record_path`], this module builds a meganeura `Graph` whose
//! parameters are per-cell `density` and per-cell `sh_r`/`sh_g`/`sh_b` (SH
//! degree 0). The forward pass:
//!
//! 1. Gathers per-step density and per-channel SH DC for each (pixel, step)
//!    via `embedding(cell_index, table)`.
//! 2. Computes per-step `raw = relu(density) * dt`. `relu` is the simplest
//!    way to keep density non-negative without an `exp` op we don't have.
//! 3. Per-pixel cumulative sum of `raw` via matmul with a fixed lower-
//!    triangular ones matrix — that's the part that gives transmittance.
//! 4. Expresses `exp(-x)` via the identity
//!    `exp(-x) = recip(sigmoid(x)) - 1` (valid for x >= 0), since meganeura
//!    has `sigmoid` and `recip` but no raw `exp`.
//! 5. Per-pixel-per-step `weight = T * alpha * mask`, then per-channel
//!    pixel colour `pixel_c = (weight * color_c) @ ones_L`.
//! 6. L1 loss against ground-truth pixels.
//!
//! The cell-walk traversal is the non-differentiable part and stays in
//! `vol::trace::record_path`. Positions, radii, and adjacency are *frozen*
//! during a training step — only per-cell appearance (density + SH DC)
//! is optimised. Updating geometry needs a different scheme (PowerFoam's
//! raytrace-mode trick: detach the traversal, keep the integration smooth).

use blade_volume as vol;
use meganeura as mn;

/// Handle to the parameter and input nodes of a built graph, so callers can
/// `set_parameter` / `set_input` / read parameters back without re-deriving
/// the names.
#[derive(Clone, Copy, Debug)]
pub struct VolumetricGraph {
    pub n_cells: usize,
    pub n_pixels: usize,
    pub max_steps: usize,

    pub log_density: mn::NodeId,
    pub sh_r: mn::NodeId,
    pub sh_g: mn::NodeId,
    pub sh_b: mn::NodeId,

    pub cell_indices: mn::NodeId,
    pub dt: mn::NodeId,
    pub mask: mn::NodeId,
    pub target: mn::NodeId,

    pub loss: mn::NodeId,
}

/// Build the volumetric forward + L1 loss subgraph and return handles.
///
/// `n_cells` is the number of cells in the model (the embedding-table size).
/// `n_pixels` is `P = width * height`. `max_steps` is the longest path the
/// recorder will produce; shorter paths get zero `mask`/`dt` padding.
pub fn build_volumetric_graph(
    g: &mut mn::Graph,
    n_cells: usize,
    n_pixels: usize,
    max_steps: usize,
) -> VolumetricGraph {
    let p = n_pixels;
    let l = max_steps;
    let pl = p * l;

    let cell_indices = g.input_u32("cell_indices", &[pl]);
    let dt = g.input("dt", &[pl]);
    let mask = g.input("mask", &[pl]);
    // Target is fed as [1, P*3] to match meganeura's "batch × dim" convention
    // for L1 loss; we reshape rather than introduce a batch dimension upstream.
    let target = g.input("labels", &[1, p * 3]);

    // Parameters live as [N, 1] tables so `embedding` returns [P*L, 1] which
    // reshapes cheaply to [P, L].
    let log_density = g.parameter("log_density", &[n_cells, 1]);
    let sh_r = g.parameter("sh_r", &[n_cells, 1]);
    let sh_g = g.parameter("sh_g", &[n_cells, 1]);
    let sh_b = g.parameter("sh_b", &[n_cells, 1]);

    // Gather and reshape to [P, L] per channel.
    let density_flat = g.embedding(cell_indices, log_density);
    let density_flat = g.relu(density_flat); // non-negative density
    let density = g.reshape(density_flat, &[p, l]);
    let dt_2d = g.reshape(dt, &[p, l]);
    let mask_2d = g.reshape(mask, &[p, l]);

    let raw = g.mul(density, dt_2d); // [P, L], non-negative
    let raw_masked = g.mul(raw, mask_2d); // zero-out padded steps

    // Cumulative sum over the L axis via matmul with a fixed strictly-lower-
    // triangular ones matrix: cum[p, k] = sum_{i<k} raw_masked[p, i].
    let cum_data = strict_lower_triangular_ones(l);
    let cum_matrix = g.constant(cum_data, &[l, l]);
    let cumsum = g.matmul(raw_masked, cum_matrix); // [P, L]

    // Transmittance: T = exp(-cumsum). Use the identity
    //   exp(-x) = recip(sigmoid(x)) - 1   (valid for x >= 0)
    let ones_pl = g.constant(vec![1.0; p * l], &[p, l]);
    let twos_pl = g.constant(vec![2.0; p * l], &[p, l]);
    let sig_cum = g.sigmoid(cumsum);
    let rec_sig_cum = g.recip(sig_cum);
    let neg_ones_pl = g.neg(ones_pl);
    let t = g.add(rec_sig_cum, neg_ones_pl); // [P, L]

    // alpha = 1 - exp(-raw_masked) = 2 - recip(sigmoid(raw_masked))
    let sig_raw = g.sigmoid(raw_masked);
    let rec_sig_raw = g.recip(sig_raw);
    let neg_rec_sig_raw = g.neg(rec_sig_raw);
    let alpha = g.add(twos_pl, neg_rec_sig_raw); // [P, L]

    let weight = g.mul(t, alpha);
    let weight = g.mul(weight, mask_2d); // zero on padded steps

    // Per-channel pixel: pixel_c = (weight * color_c) @ ones_L
    let ones_l1 = g.constant(vec![1.0; l], &[l, 1]);
    let pixel_r = channel_pixel(g, cell_indices, sh_r, weight, ones_l1, p, l);
    let pixel_g = channel_pixel(g, cell_indices, sh_g, weight, ones_l1, p, l);
    let pixel_b = channel_pixel(g, cell_indices, sh_b, weight, ones_l1, p, l);

    // Three per-channel L1 losses summed into one scalar. We could concat
    // and call l1_loss once, but concat in meganeura works on flat NCHW
    // shapes; summing the three losses is shorter.
    let target_2d = g.reshape(target, &[p, 3]);
    let split_r = g.split_a(target_2d, p as u32, 1, 2, 1);
    let target_r = g.reshape(split_r, &[p, 1]);
    let target_rest = g.split_b(target_2d, p as u32, 1, 2, 1);
    let target_rest_2d = g.reshape(target_rest, &[p, 2]);
    let split_g = g.split_a(target_rest_2d, p as u32, 1, 1, 1);
    let target_g = g.reshape(split_g, &[p, 1]);
    let split_b = g.split_b(target_rest_2d, p as u32, 1, 1, 1);
    let target_b = g.reshape(split_b, &[p, 1]);

    let loss_r = g.l1_loss(pixel_r, target_r);
    let loss_g = g.l1_loss(pixel_g, target_g);
    let loss_b = g.l1_loss(pixel_b, target_b);
    let loss_rg = g.add(loss_r, loss_g);
    let loss = g.add(loss_rg, loss_b);
    g.set_outputs(vec![loss]);

    VolumetricGraph {
        n_cells,
        n_pixels,
        max_steps,
        log_density,
        sh_r,
        sh_g,
        sh_b,
        cell_indices,
        dt,
        mask,
        target,
        loss,
    }
}

/// `[L, L]` row-major matrix with `M[i, k] = 1` iff `i < k`, else `0`.
fn strict_lower_triangular_ones(l: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; l * l];
    for i in 0..l {
        for k in (i + 1)..l {
            data[i * l + k] = 1.0;
        }
    }
    data
}

/// SH degree-0 basis constant — matches `vol::trace::eval_rgb_sh` so the
/// graph's per-cell colour expression mirrors the production renderer.
const SH_C0: f32 = 0.282_094_8;

fn channel_pixel(
    g: &mut mn::Graph,
    cell_indices: mn::NodeId,
    sh_chan: mn::NodeId,
    weight: mn::NodeId,
    ones_l1: mn::NodeId,
    p: usize,
    l: usize,
) -> mn::NodeId {
    let color_flat = g.embedding(cell_indices, sh_chan); // [P*L, 1]
    let color = g.reshape(color_flat, &[p, l]);
    // Match `eval_rgb_sh`: per-cell colour = SH_C0 * sh_chan + 0.5.
    // We multiply weight by this whole expression; equivalently emit the
    // scaled-plus-biased colour, then mul by weight, then reduce.
    let scale = g.constant(vec![SH_C0; p * l], &[p, l]);
    let bias = g.constant(vec![0.5; p * l], &[p, l]);
    let scaled = g.mul(color, scale);
    let biased = g.add(scaled, bias);
    let weighted = g.mul(weight, biased); // [P, L]
    g.matmul(weighted, ones_l1) // [P, L] @ [L, 1] = [P, 1]
}

/// Flatten per-pixel paths into the three tensors meganeura consumes.
///
/// Each path is padded to `max_steps` with `cell=0`, `dt=0`, `mask=0`. The
/// pad-cell index is harmless because `mask=0` forces all per-step
/// contributions to zero downstream.
pub fn flatten_paths(paths: &[vol::trace::PathResult], max_steps: usize) -> FlatPaths {
    let p = paths.len();
    let pl = p * max_steps;
    let mut cell = vec![0u32; pl];
    let mut dt = vec![0.0f32; pl];
    let mut mask = vec![0.0f32; pl];
    for (pi, path) in paths.iter().enumerate() {
        let n = path.entries.len().min(max_steps);
        for (k, e) in path.entries[..n].iter().enumerate() {
            let idx = pi * max_steps + k;
            cell[idx] = e.cell;
            dt[idx] = e.dt;
            mask[idx] = 1.0;
        }
    }
    FlatPaths { cell, dt, mask }
}

#[derive(Clone, Debug)]
pub struct FlatPaths {
    pub cell: Vec<u32>,
    pub dt: Vec<f32>,
    pub mask: Vec<f32>,
}

/// Knobs for [`fit_appearance_to_pixels`].
#[derive(Clone, Copy, Debug)]
pub struct AppearanceFitConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
}

impl Default for AppearanceFitConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epochs: 200,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
        }
    }
}

/// Fit per-cell density + SH degree-0 DC of `model` to match a target image,
/// given precomputed per-pixel paths. Geometry (positions, radii, adjacency)
/// is left untouched.
///
/// `target_pixels.len()` must equal `paths.len() * 3` (RGB per pixel,
/// row-major). All paths are flattened to `max_steps` — shorter paths are
/// padded with zero `mask` so they contribute nothing.
///
/// Returns the loss after each Adam step.
pub fn fit_appearance_to_pixels(
    model: &mut vol::PointCloudModel,
    target_pixels: &[f32],
    paths: &[vol::trace::PathResult],
    max_steps: usize,
    config: AppearanceFitConfig,
    gpu: std::sync::Arc<blade_graphics::Context>,
) -> Vec<f32> {
    let p = paths.len();
    let n_cells = model.points.len();
    assert_eq!(
        target_pixels.len(),
        p * 3,
        "target_pixels.len() must be P*3"
    );
    assert!(
        model.sh_degree == 0,
        "fit_appearance_to_pixels needs SH degree 0"
    );

    let flat = flatten_paths(paths, max_steps);

    let mut g = mn::Graph::new();
    let _vg = build_volumetric_graph(&mut g, n_cells, p, max_steps);

    let (mut session, _report) = mn::build(
        &g,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );

    // Lift the model's current values into the four parameter tables.
    let mut init_density = Vec::with_capacity(n_cells);
    let mut init_r = Vec::with_capacity(n_cells);
    let mut init_g = Vec::with_capacity(n_cells);
    let mut init_b = Vec::with_capacity(n_cells);
    for (i, p) in model.points.iter().enumerate() {
        init_density.push(p.w);
        init_r.push(model.sh_coefficients[i * 3]);
        init_g.push(model.sh_coefficients[i * 3 + 1]);
        init_b.push(model.sh_coefficients[i * 3 + 2]);
    }
    session.set_parameter("log_density", &init_density);
    session.set_parameter("sh_r", &init_r);
    session.set_parameter("sh_g", &init_g);
    session.set_parameter("sh_b", &init_b);

    session.set_input_u32("cell_indices", &flat.cell);
    session.set_input("dt", &flat.dt);
    session.set_input("mask", &flat.mask);
    session.set_input("labels", target_pixels);

    let mut losses = Vec::with_capacity(config.epochs);
    for _ in 0..config.epochs {
        // `set_adam` queues a one-shot — `step()` consumes it. Re-arm before
        // every step or the optimiser stops updating after the first.
        session.set_adam(
            config.learning_rate,
            config.adam_beta1,
            config.adam_beta2,
            config.adam_eps,
        );
        session.step();
        session.wait();
        let read = session.read_output(1);
        losses.push(read.first().copied().unwrap_or(f32::NAN));
    }

    // Write trained parameters back into the model. `relu` in the graph
    // means densities are clamped to >= 0 in the forward; mirror that here.
    let density_buf = session.param_buffer("log_density").unwrap();
    let mut out_density = vec![0.0f32; n_cells];
    session.read_buffer(density_buf, &mut out_density);

    let mut out_chan = vec![0.0f32; n_cells];
    let r_buf = session.param_buffer("sh_r").unwrap();
    let g_buf = session.param_buffer("sh_g").unwrap();
    let b_buf = session.param_buffer("sh_b").unwrap();
    for (i, d) in out_density.iter().enumerate() {
        model.points[i].w = d.max(0.0);
    }
    session.read_buffer(r_buf, &mut out_chan);
    for (i, &c) in out_chan.iter().enumerate() {
        model.sh_coefficients[i * 3] = c;
    }
    session.read_buffer(g_buf, &mut out_chan);
    for (i, &c) in out_chan.iter().enumerate() {
        model.sh_coefficients[i * 3 + 1] = c;
    }
    session.read_buffer(b_buf, &mut out_chan);
    for (i, &c) in out_chan.iter().enumerate() {
        model.sh_coefficients[i * 3 + 2] = c;
    }

    losses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::try_init_gpu;

    fn tiny_model() -> vol::PointCloudModel {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 4.0),
            glam::Vec4::new(0.5, 0.0, 0.0, 4.0),
            glam::Vec4::new(0.0, 0.5, 0.0, 4.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 4.0),
        ];
        let mut m = vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
            points,
        };
        m.compute_adjacency_default();
        m
    }

    #[test]
    fn strict_lower_triangular_ones_shape() {
        let m = strict_lower_triangular_ones(3);
        // row 0: 0 1 1
        // row 1: 0 0 1
        // row 2: 0 0 0
        assert_eq!(m, vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn flatten_paths_zero_pads_missing_steps() {
        let paths = vec![
            vol::trace::PathResult {
                entries: vec![
                    vol::trace::PathEntry { cell: 1, dt: 0.5 },
                    vol::trace::PathEntry { cell: 2, dt: 0.3 },
                ],
                ray_dir: glam::Vec3::Z,
            },
            vol::trace::PathResult {
                entries: vec![vol::trace::PathEntry { cell: 0, dt: 0.1 }],
                ray_dir: glam::Vec3::Z,
            },
        ];
        let f = flatten_paths(&paths, 3);
        assert_eq!(f.cell, vec![1, 2, 0, 0, 0, 0]);
        assert_eq!(f.dt, vec![0.5, 0.3, 0.0, 0.1, 0.0, 0.0]);
        assert_eq!(f.mask, vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    /// Build the graph, feed inputs, run one Adam step via the lower-level
    /// `session.step()` (rather than `Trainer`/`DataLoader`). No convergence
    /// assertion — that's M3c-4c. This just proves the forward/backward
    /// dispatch succeeds end-to-end on the GPU.
    ///
    /// We bypass `Trainer` because its `(data, labels)` shape doesn't fit our
    /// graph: we need a `u32` input (`cell_indices`) plus three `f32` inputs
    /// (`dt`, `mask`, `labels`) per step.
    #[test]
    fn volumetric_graph_runs_one_step() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping volumetric_graph_runs_one_step: no GPU");
            return;
        };

        let model = tiny_model();
        let n_cells = model.points.len();

        let ray = vol::trace::Ray {
            origin: glam::Vec3::new(0.1, 0.1, -1.0),
            direction: glam::Vec3::new(0.0, 0.0, 1.0),
        };
        let settings = vol::trace::TraceSettings {
            start_point: 0,
            max_steps: 8,
            weight_threshold: 0.0,
            depth: 100.0,
            eval_mode: vol::trace::EvalMode::ConstantRgb(glam::Vec3::ONE),
        };
        let path = vol::trace::record_path(&model, ray, settings);
        let max_steps = 8usize;
        let p = 1usize;
        let flat = flatten_paths(&[path], max_steps);

        let mut g = mn::Graph::new();
        let _vg = build_volumetric_graph(&mut g, n_cells, p, max_steps);

        let (mut session, _report) = mn::build(
            &g,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );

        session.set_parameter("log_density", &vec![1.0; n_cells]);
        session.set_parameter("sh_r", &vec![0.0; n_cells]);
        session.set_parameter("sh_g", &vec![0.0; n_cells]);
        session.set_parameter("sh_b", &vec![0.0; n_cells]);

        session.set_input_u32("cell_indices", &flat.cell);
        session.set_input("dt", &flat.dt);
        session.set_input("mask", &flat.mask);
        let target = [0.4_f32, 0.6, 0.8];
        session.set_input("labels", &target);

        session.set_adam(0.1, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();
    }

    /// End-to-end: a small target image is the ground-truth render of a
    /// known model. Re-initialise the same geometry with bad appearance
    /// values, fit, and check that the loss decreases by an order of
    /// magnitude over 200 steps. This is the smoke test that proves
    /// gradients actually flow into per-cell density/SH and reduce error.
    #[test]
    fn fit_appearance_reduces_loss() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping fit_appearance_reduces_loss: no GPU");
            return;
        };

        // Ground-truth model: tetrahedron, density 4, cell 0 is red-ish.
        let gt = {
            let mut m = tiny_model();
            // Use SH_C0 to encode a target RGB through the DC channel.
            const SH_C0: f32 = 0.282_094_8;
            let target_rgb = [0.7f32, 0.2, 0.1]; // cell 0 hue
                                                 // (rgb - 0.5) / SH_C0 maps the renderer's 0.5+bias output back
                                                 // to the desired colour.
            for (i, &c) in target_rgb.iter().enumerate() {
                m.sh_coefficients[i] = (c - 0.5) / SH_C0;
            }
            m
        };

        // Single pixel, ray nearly through cell 0.
        let cam = vol::CameraParams {
            cam_position: [0.05, 0.05, -1.0],
            depth: 100.0,
            cam_orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [0.5, 0.5],
            pad: [0, 0],
        };
        let target_pixels = crate::render::render_cpu(
            &gt,
            &cam,
            crate::render::RenderSettings {
                width: 1,
                height: 1,
                start_point: 0,
                max_steps: 16,
                weight_threshold: 1e-4,
            },
        );
        let target_rgb = [target_pixels[0], target_pixels[1], target_pixels[2]];

        // Init model: same geometry, density a couple of orders of magnitude
        // away from the ground truth (4.0). Starting too low (e.g. 0.01)
        // shrinks alpha to ~0.004 and gradients can't move SH at all in 200
        // Adam steps. Zero SH DC gives a fully-grey starting prediction.
        let mut init = tiny_model();
        for i in 0..init.points.len() {
            init.points[i].w = 1.0;
        }
        for v in init.sh_coefficients.iter_mut() {
            *v = 0.0;
        }

        // Record one path through the init geometry for that single pixel.
        // The ray follows the same camera→pixel mapping render_cpu uses.
        let ndc = glam::Vec2::new(0.0, 0.0); // pixel (0,0) of a 1x1 image
        let tan_half = glam::Vec2::new((0.5 * cam.fov[0]).tan(), (0.5 * cam.fov[1]).tan());
        let local_dir = glam::Vec3::new(ndc.x * tan_half.x, ndc.y * tan_half.y, 1.0);
        let orientation = glam::Quat::from_xyzw(
            cam.cam_orientation[0],
            cam.cam_orientation[1],
            cam.cam_orientation[2],
            cam.cam_orientation[3],
        );
        let ray = vol::trace::Ray {
            origin: glam::Vec3::from_array(cam.cam_position),
            direction: (orientation * local_dir).normalize(),
        };
        let path = vol::trace::record_path(
            &init,
            ray,
            vol::trace::TraceSettings {
                start_point: 0,
                max_steps: 16,
                weight_threshold: 0.0,
                depth: cam.depth,
                eval_mode: vol::trace::EvalMode::Sh,
            },
        );

        let losses = fit_appearance_to_pixels(
            &mut init,
            &target_rgb,
            &[path],
            16,
            AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 500,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        let first = losses.first().copied().unwrap();
        let last = losses.last().copied().unwrap();
        assert!(first.is_finite() && last.is_finite(), "non-finite loss");
        assert!(
            last < first * 0.1,
            "loss did not drop enough: first {first}, last {last}"
        );
    }

    /// Per-pixel ray for an 8×8 image at the given camera. Mirrors
    /// `render::render_cpu`'s mapping.
    fn rays_for(cam: &vol::CameraParams, w: u32, h: u32) -> Vec<vol::trace::Ray> {
        let tan_half = glam::Vec2::new((0.5 * cam.fov[0]).tan(), (0.5 * cam.fov[1]).tan());
        let orientation = glam::Quat::from_xyzw(
            cam.cam_orientation[0],
            cam.cam_orientation[1],
            cam.cam_orientation[2],
            cam.cam_orientation[3],
        );
        let origin = glam::Vec3::from_array(cam.cam_position);
        let mut rays = Vec::with_capacity((w * h) as usize);
        let wf = w as f32;
        let hf = h as f32;
        for iy in 0..h {
            let py = (iy as f32 + 0.5) / hf;
            let ndc_y = py * 2.0 - 1.0;
            for ix in 0..w {
                let px = (ix as f32 + 0.5) / wf;
                let ndc_x = px * 2.0 - 1.0;
                let local = glam::Vec3::new(ndc_x * tan_half.x, ndc_y * tan_half.y, 1.0);
                rays.push(vol::trace::Ray {
                    origin,
                    direction: (orientation * local).normalize(),
                });
            }
        }
        rays
    }

    fn mean_l1(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / a.len() as f32
    }

    /// Train on view A, render trained model from view B, compare to the
    /// ground-truth render at view B. This is the novel-pose generalisation
    /// check: with appearance-only training (frozen geometry), the trained
    /// model should reproduce cell colours that *any* ray sees consistently,
    /// so a different camera should also see them.
    #[test]
    fn novel_pose_render_matches_ground_truth() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping novel_pose_render_matches_ground_truth: no GPU");
            return;
        };

        // Ground truth: 4 cells, each with a distinct strong colour.
        const SH_C0: f32 = 0.282_094_8;
        let cell_colours = [
            [0.9_f32, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.9, 0.9, 0.1],
        ];
        let gt = {
            let mut m = tiny_model();
            for (i, c) in cell_colours.iter().enumerate() {
                for (k, v) in c.iter().enumerate() {
                    m.sh_coefficients[i * 3 + k] = (v - 0.5) / SH_C0;
                }
            }
            m
        };

        let cam_a = vol::CameraParams {
            cam_position: [0.10, 0.10, -1.0],
            depth: 100.0,
            cam_orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [0.5, 0.5],
            pad: [0, 0],
        };
        // View B: yaw camera 90° so it looks along +X instead of +Z, from a
        // different starting position. Same fov / depth.
        let half_pi = std::f32::consts::FRAC_PI_2;
        let rot = glam::Quat::from_axis_angle(glam::Vec3::Y, half_pi);
        let cam_b = vol::CameraParams {
            cam_position: [-1.0, 0.10, 0.10],
            depth: 100.0,
            cam_orientation: [rot.x, rot.y, rot.z, rot.w],
            fov: [0.5, 0.5],
            pad: [0, 0],
        };

        let w = 8u32;
        let h = 8u32;
        let render_settings = crate::render::RenderSettings {
            width: w,
            height: h,
            start_point: 0,
            max_steps: 32,
            weight_threshold: 1e-4,
        };
        let target_a = crate::render::render_cpu(&gt, &cam_a, render_settings);
        let gt_render_b = crate::render::render_cpu(&gt, &cam_b, render_settings);

        // Strip the alpha channel from render_cpu's RGBA output for the
        // L1-target since the graph only predicts RGB.
        let target_a_rgb = strip_alpha(&target_a);

        let mut init = tiny_model();
        for p in init.points.iter_mut() {
            p.w = 1.0;
        }
        for v in init.sh_coefficients.iter_mut() {
            *v = 0.0;
        }

        // Record paths from the init model (geometry = gt's geometry; only
        // appearance is being optimised, so traversal is identical).
        let trace_settings = vol::trace::TraceSettings {
            start_point: 0,
            max_steps: 32,
            weight_threshold: 0.0,
            depth: cam_a.depth,
            eval_mode: vol::trace::EvalMode::Sh,
        };
        let paths_a: Vec<_> = rays_for(&cam_a, w, h)
            .into_iter()
            .map(|ray| vol::trace::record_path(&init, ray, trace_settings))
            .collect();

        fit_appearance_to_pixels(
            &mut init,
            &target_a_rgb,
            &paths_a,
            32,
            AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 500,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        // Render the trained model from the novel pose.
        let trained_b = crate::render::render_cpu(&init, &cam_b, render_settings);

        // Compare just RGB. The trained model should produce *something* not
        // wildly different from the GT render at view B.
        let trained_b_rgb = strip_alpha(&trained_b);
        let gt_b_rgb = strip_alpha(&gt_render_b);
        let novel_err = mean_l1(&trained_b_rgb, &gt_b_rgb);

        let baseline = strip_alpha(&vec![0.0f32; target_a.len()]); // pure black
        let baseline_err = mean_l1(&baseline, &gt_b_rgb);

        eprintln!("novel-pose mean L1: {novel_err:.4}; black baseline: {baseline_err:.4}");
        // Trained novel-view should beat the trivial black baseline by a
        // healthy margin.
        assert!(
            novel_err < baseline_err * 0.8,
            "novel-pose render didn't beat the black baseline: \
             trained={novel_err} baseline={baseline_err}"
        );
    }

    fn strip_alpha(rgba: &[f32]) -> Vec<f32> {
        let n_px = rgba.len() / 4;
        let mut rgb = Vec::with_capacity(n_px * 3);
        for px in 0..n_px {
            rgb.push(rgba[px * 4]);
            rgb.push(rgba[px * 4 + 1]);
            rgb.push(rgba[px * 4 + 2]);
        }
        rgb
    }
}
