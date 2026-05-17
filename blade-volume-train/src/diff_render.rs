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
    let weighted = g.mul(weight, color); // [P, L]
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
}
