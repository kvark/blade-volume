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
#[derive(Clone, Debug)]
pub struct VolumetricGraph {
    pub n_cells: usize,
    pub n_pixels: usize,
    pub max_steps: usize,
    pub sh_degree: usize,
    pub num_views: usize,

    pub log_density: mn::NodeId,
    pub positions: mn::NodeId,
    /// `sh_coefficients[c][k]` is the `[n_cells, 1]` parameter table for
    /// channel `c` ∈ {0=R, 1=G, 2=B} and SH component `k` ∈ `0..(1+sh_degree)²`.
    pub sh_coefficients: Vec<Vec<mn::NodeId>>,
    /// Per-view, per-channel RGB gain: one `[num_views, 1]` table per
    /// channel. Multiplies each rendered pixel before the L1 loss.
    /// Initialised to 1.0; frozen at 1.0 when the `exposure_*` LR
    /// multipliers are 0 (default OFF for an apples-to-apples
    /// baseline). With LR > 0 Adam absorbs per-image brightness
    /// variation into these tables instead of the SH chain. Three
    /// separate tables (rather than one `[num_views, 3]`) so the
    /// gradient flows through `embedding` only — `SplitA`/`SplitB`
    /// have an empty backward in meganeura and would silently zero
    /// the gradient.
    pub exposure_r: mn::NodeId,
    pub exposure_g: mn::NodeId,
    pub exposure_b: mn::NodeId,

    pub cell_indices: mn::NodeId,
    pub next_cell_indices: mn::NodeId,
    pub mask: mn::NodeId,
    pub ray_origin: mn::NodeId,
    pub ray_dir_per_pixel: mn::NodeId,
    pub pixel_idx_per_step: mn::NodeId,
    /// Single u32 scalar — index into `exposure` for the current view.
    /// Set once per Adam step via `set_input_u32("view_idx", &[vi])`.
    pub view_idx: mn::NodeId,
    /// `basis_inputs[k-1]` is the `[P, 1]` per-pixel basis input for SH
    /// component `k` (k ≥ 1). Component 0 is the constant `SH_C0`, no
    /// input needed. Empty when `sh_degree == 0`.
    pub basis_inputs: Vec<mn::NodeId>,
    pub target: mn::NodeId,

    pub loss: mn::NodeId,
    /// Graph-computed `dt` per (pixel, step) shape `[P, L]`. Exposed as
    /// the second output so callers can compare against the shader's
    /// `dt` for the position-optimisation sanity check.
    pub dt_from_positions: mn::NodeId,
}

/// SH-component name suffix, e.g. `parameter_name("sh_r", 0)` → `"sh_r"`,
/// `parameter_name("sh_r", 3)` → `"sh_r_3"`. Component 0 keeps the
/// historical bare name for backward-compat with SH-0 checkpoints.
fn parameter_name(channel: &str, k: usize) -> String {
    if k == 0 {
        channel.to_string()
    } else {
        format!("{channel}_{k}")
    }
}

/// Build the volumetric forward + L1 loss subgraph and return handles.
///
/// `n_cells` is the number of cells in the model (the embedding-table size).
/// `n_pixels` is `P = width * height`. `max_steps` is the longest path the
/// recorder will produce; shorter paths get zero `mask`/`dt` padding.
///
/// `sh_degree` ∈ {0, 1, 2, 3} controls the per-cell colour expressiveness.
/// SH-0 (default) is a flat scalar — same as the original SH-degree-0
/// pipeline. Higher degrees take per-pixel basis-value inputs named
/// `basis_1`, `basis_2`, … through `basis_{(1+sh_degree)²-1}` and learn
/// `(1+sh_degree)²` coefficients per cell per RGB channel.
#[allow(clippy::too_many_arguments)]
pub fn build_volumetric_graph(
    g: &mut mn::Graph,
    n_cells: usize,
    n_pixels: usize,
    max_steps: usize,
    sh_degree: usize,
    num_views: usize,
    patch_size: usize,
    grad_loss_weight: f32,
    opacity_weight: f32,
    softplus_beta: f32,
) -> VolumetricGraph {
    assert!(
        sh_degree <= 3,
        "build_volumetric_graph: sh_degree {sh_degree} not supported (max 3)",
    );
    assert!(
        num_views >= 1,
        "build_volumetric_graph: num_views must be at least 1",
    );
    if patch_size > 0 {
        assert_eq!(
            n_pixels,
            patch_size * patch_size,
            "patch mode: n_pixels must equal patch_size²"
        );
    }
    let p = n_pixels;
    let l = max_steps;
    let pl = p * l;
    let num_components = (1 + sh_degree) * (1 + sh_degree);

    let cell_indices = g.input_u32("cell_indices", &[pl]);
    let next_cell_indices = g.input_u32("next_cell_indices", &[pl]);
    let mask = g.input("mask", &[pl]);
    // Target is fed as [1, P*3] to match meganeura's "batch × dim" convention
    // for L1 loss; we reshape rather than introduce a batch dimension upstream.
    let target = g.input("labels", &[1, p * 3]);

    // Ray geometry inputs — needed to compute dt from positions
    // differentiably. Set once per Adam step; the host code computes the
    // ray-origin (single 3-vector per view) and ray-direction (one per
    // sampled pixel) from the view camera.
    let ray_origin = g.input("ray_origin", &[1, 3]);
    let ray_dir_per_pixel = g.input("ray_dir_per_pixel", &[p, 3]);
    // Constant gather: pixel_idx_per_step[k] = k / L = which pixel this
    // (pixel, step) entry belongs to. Used to broadcast per-pixel ray
    // direction to per-step via `embedding(..., ray_dir_per_pixel)`.
    let pixel_idx_per_step = g.input_u32("pixel_idx_per_step", &[pl]);

    // Per-pixel SH basis values. Component 0 is the constant SH_C0
    // (handled inside `channel_pixel_sh`), so we only declare inputs for
    // components 1..K.
    let basis_inputs: Vec<mn::NodeId> = (1..num_components)
        .map(|k| g.input(&format!("basis_{k}"), &[p, 1]))
        .collect();

    // Parameters live as [N, 1] tables (or [N, 3] for positions) so
    // `embedding` returns [P*L, 1] (or [P*L, 3]). For SH-degree > 0 we
    // have K coefficient tables per channel.
    let log_density = g.parameter("log_density", &[n_cells, 1]);
    let positions = g.parameter("positions", &[n_cells, 3]);
    // Per-view RGB gain: separate `[num_views, 1]` table per channel
    // so each channel's gradient flows back through `embedding`
    // (which has a `scatter_add` backward). Folding channels into a
    // single `[num_views, 3]` table + `split_a`/`split_b` would also
    // work post-meganeura `ca11915` (which implemented split's
    // backward), but the per-channel layout is what landed during
    // the original investigation and is easier to read; keeping it.
    let exposure_r = g.parameter("exposure_r", &[num_views, 1]);
    let exposure_g = g.parameter("exposure_g", &[num_views, 1]);
    let exposure_b = g.parameter("exposure_b", &[num_views, 1]);
    let view_idx = g.input_u32("view_idx", &[1]);
    let mut sh_coefficients: Vec<Vec<mn::NodeId>> = Vec::with_capacity(3);
    for channel in ["sh_r", "sh_g", "sh_b"] {
        let mut per_channel = Vec::with_capacity(num_components);
        for k in 0..num_components {
            per_channel.push(g.parameter(&parameter_name(channel, k), &[n_cells, 1]));
        }
        sh_coefficients.push(per_channel);
    }

    // Density activation. ReLU (legacy) zeroes negative log-density AND
    // its gradient, so a cell that dips negative dies permanently (dead
    // ReLU) — these accumulate and destabilise densification. softplus
    // (RadFoam's choice) keeps a small gradient for negatives so cells
    // recover. Stable form with meganeura ops (no exp/softplus builtin):
    //   softplus_β(x) = (1/β)[relu(βx) − log(sigmoid(|βx|))]
    // (sigmoid(|βx|) ∈ [0.5,1] so the log can't overflow). At β=10,
    // log_density = 1.0 → density ≈ 1.0, preserving the ReLU init.
    let density_pre = g.embedding(cell_indices, log_density);
    let density_flat = if softplus_beta > 0.0 {
        let beta_c = g.constant(vec![softplus_beta; pl], &[pl, 1]);
        let bx = g.mul(density_pre, beta_c);
        let relu_bx = g.relu(bx);
        let abs_bx = g.abs(bx);
        let sig = g.sigmoid(abs_bx);
        let log_sig = g.log(sig);
        let neg_log_sig = g.neg(log_sig);
        let sp = g.add(relu_bx, neg_log_sig);
        let inv_beta = g.constant(vec![1.0 / softplus_beta; pl], &[pl, 1]);
        g.mul(sp, inv_beta)
    } else {
        g.relu(density_pre) // legacy non-negative density
    };
    let density = g.reshape(density_flat, &[p, l]);
    let mask_2d = g.reshape(mask, &[p, l]);

    // --- Differentiable dt from positions ---
    //
    // Gather positions for the current and next cell of every (pixel,
    // step) entry; reconstruct the bisector plane between them; compute
    // the ray's intersection `t` with that plane; and difference
    // adjacent t values to get `dt`.
    //
    // Bisector plane (Voronoi face) between cell i and cell j:
    //   midpoint m = (p_i + p_j) / 2
    //   normal   n = p_j - p_i
    // Ray-plane intersection for ray `o + s * d`:
    //   t = dot(m - o, n) / dot(n, d)
    //
    // Sequential `dt_k = t_k - t_{k-1}` is implemented as a matmul
    // against a shift-right matrix (super-diagonal of 1s).
    //
    // Numerical safety: invalid path steps have `mask = 0`, and may
    // also have `cell == next_cell == 0` (buffers zeroed before
    // dispatch). To prevent NaN/Inf when `dot(n, d) = 0` for those
    // entries, we add a small ε to the divisor; the final `mask`
    // multiplication zeros the result anyway.
    let pos_cell = g.embedding(cell_indices, positions); // [P*L, 3]
    let pos_next = g.embedding(next_cell_indices, positions); // [P*L, 3]
    let half_pl3 = g.constant(vec![0.5_f32; pl * 3], &[pl, 3]);
    let pos_sum = g.add(pos_cell, pos_next);
    let midpoint = g.mul(pos_sum, half_pl3);
    let neg_pos_cell = g.neg(pos_cell);
    let normal = g.add(pos_next, neg_pos_cell);

    // Broadcast ray_origin [1,3] → [P*L, 3] via matmul with [P*L, 1] ones.
    let ones_pl1 = g.constant(vec![1.0_f32; pl], &[pl, 1]);
    let ray_origin_pl = g.matmul(ones_pl1, ray_origin); // [P*L, 3]
    // Gather ray_dir [P, 3] → [P*L, 3] via embedding with pixel_idx_per_step.
    let ray_dir_pl = g.embedding(pixel_idx_per_step, ray_dir_per_pixel);

    let neg_ray_origin_pl = g.neg(ray_origin_pl);
    let mo_diff = g.add(midpoint, neg_ray_origin_pl); // [P*L, 3]

    // dot products via element-wise mul then matmul against [3, 1] ones.
    let ones_3_1 = g.constant(vec![1.0_f32; 3], &[3, 1]);
    let mn_prod = g.mul(mo_diff, normal);
    let dot_num = g.matmul(mn_prod, ones_3_1); // [P*L, 1]
    let nd_prod = g.mul(normal, ray_dir_pl);
    let dot_den_raw = g.matmul(nd_prod, ones_3_1); // [P*L, 1]
    // ε regularisation: for invalid steps `dot_den_raw = 0`; downstream
    // mask zeros the result, but the division itself must stay finite.
    let eps_pl1 = g.constant(vec![1.0e-6_f32; pl], &[pl, 1]);
    let dot_den = g.add(dot_den_raw, eps_pl1);
    let t_pl1 = g.div(dot_num, dot_den);
    let t_2d = g.reshape(t_pl1, &[p, l]); // [P, L]

    // Build the L×L shift-right matrix: M[i, k] = 1 iff k == i+1 (and k≥1).
    // (t @ M)[p, k] = t[p, k-1] for k ≥ 1, else 0.
    let mut shift_data = vec![0.0_f32; l * l];
    for i in 0..l.saturating_sub(1) {
        shift_data[i * l + (i + 1)] = 1.0;
    }
    let shift_mat = g.constant(shift_data, &[l, l]);
    let t_shifted = g.matmul(t_2d, shift_mat); // [P, L]
    let neg_t_shifted = g.neg(t_shifted);
    let dt_raw_2d = g.add(t_2d, neg_t_shifted); // [P, L]

    // Clamp dt to [0, MAX_PATH_DT] to keep the sigmoid-surrogate-for-exp
    // numerically stable. `min(relu(x), c) = c - relu(c - relu(x))`.
    let dt_pos = g.relu(dt_raw_2d);
    let max_dt_pl = g.constant(vec![MAX_PATH_DT; p * l], &[p, l]);
    let neg_dt_pos = g.neg(dt_pos);
    let cap_minus_dt = g.add(max_dt_pl, neg_dt_pos);
    let over_cap = g.relu(cap_minus_dt);
    let neg_over_cap = g.neg(over_cap);
    let dt_clamped = g.add(max_dt_pl, neg_over_cap); // = min(dt_pos, MAX_PATH_DT)
    let dt_2d = g.mul(dt_clamped, mask_2d); // zero out invalid steps
    let dt_from_positions = dt_2d;

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
    let ones_1l = g.constant(vec![1.0; l], &[1, l]);

    // Accumulated opacity per pixel = Σ_L weight = 1 − T_final. Drives the
    // RadFoam opacity loss + white-background compositing.
    let opacity = g.matmul(weight, ones_l1); // [P, 1]
    let pixel_r = channel_pixel_sh(
        g,
        cell_indices,
        &sh_coefficients[0],
        &basis_inputs,
        weight,
        ones_l1,
        ones_1l,
        p,
        l,
    );
    let pixel_g = channel_pixel_sh(
        g,
        cell_indices,
        &sh_coefficients[1],
        &basis_inputs,
        weight,
        ones_l1,
        ones_1l,
        p,
        l,
    );
    let pixel_b = channel_pixel_sh(
        g,
        cell_indices,
        &sh_coefficients[2],
        &basis_inputs,
        weight,
        ones_l1,
        ones_1l,
        p,
        l,
    );

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

    // --- Per-view exposure ---
    //
    // Each channel's exposure is its own `[num_views, 1]` table. Gather
    // the row for the current view via embedding (which is fully
    // differentiable — backward is scatter_add into the table). The
    // result is `[1, 1]`; broadcast to `[p, 1]` with a matmul against
    // the `ones[p, 1]` constant, then elementwise-multiply the
    // rendered channel.
    let ones_p1 = g.constant(vec![1.0_f32; p], &[p, 1]);
    let exp_r_11 = g.embedding(view_idx, exposure_r); // [1, 1]
    let exp_r_p = g.matmul(ones_p1, exp_r_11); // [p, 1]
    let pixel_r = g.mul(pixel_r, exp_r_p);
    let exp_g_11 = g.embedding(view_idx, exposure_g);
    let exp_g_p = g.matmul(ones_p1, exp_g_11);
    let pixel_g = g.mul(pixel_g, exp_g_p);
    let exp_b_11 = g.embedding(view_idx, exposure_b);
    let exp_b_p = g.matmul(ones_p1, exp_b_11);
    let pixel_b = g.mul(pixel_b, exp_b_p);

    // White-background compositing (RadFoam): add (1 − opacity) per channel
    // so rays that accumulate little opacity read as white. Active only
    // alongside the opacity loss; otherwise the legacy un-composited path
    // is preserved exactly.
    let (pixel_r, pixel_g, pixel_b) = if opacity_weight > 0.0 {
        let neg_op = g.neg(opacity);
        let bg = g.add(ones_p1, neg_op); // [P,1] = 1 − opacity
        (
            g.add(pixel_r, bg),
            g.add(pixel_g, bg),
            g.add(pixel_b, bg),
        )
    } else {
        (pixel_r, pixel_g, pixel_b)
    };

    let loss_r = g.l1_loss(pixel_r, target_r);
    let loss_g = g.l1_loss(pixel_g, target_g);
    let loss_b = g.l1_loss(pixel_b, target_b);
    let loss_rg = g.add(loss_r, loss_g);
    let l1 = g.add(loss_rg, loss_b);

    // Gradient (structural) L1 loss, computed in patch mode only. The
    // finite-difference ∂/∂x and ∂/∂y of the rendered patch is matched
    // against the corresponding difference of the target patch. Acts as
    // a poor-man's SSIM — captures local edge structure that random-
    // pixel L1 alone cannot see.
    let color_loss = if patch_size > 0 && grad_loss_weight > 0.0 {
        let q = patch_size;
        let grad_r = patch_grad_l1(g, pixel_r, target_r, q);
        let grad_g = patch_grad_l1(g, pixel_g, target_g, q);
        let grad_b = patch_grad_l1(g, pixel_b, target_b, q);
        let grad_rg = g.add(grad_r, grad_g);
        let grad_loss = g.add(grad_rg, grad_b);
        // Combine: (1 - α) * L1 + α * L1_grad
        let l1_weight = g.constant(vec![1.0 - grad_loss_weight], &[1, 1]);
        let grad_weight = g.constant(vec![grad_loss_weight], &[1, 1]);
        let l1_2d = g.reshape(l1, &[1, 1]);
        let grad_2d = g.reshape(grad_loss, &[1, 1]);
        let l1_scaled = g.mul(l1_2d, l1_weight);
        let grad_scaled = g.mul(grad_2d, grad_weight);
        let combined = g.add(l1_scaled, grad_scaled);
        g.reshape(combined, &[1])
    } else {
        l1
    };

    // RadFoam opacity loss: push accumulated opacity → 1 (the all-ones
    // target alpha for opaque COLMAP scenes). Penalises semi-transparent
    // floaters that random-pixel L1 alone tolerates. `loss = color +
    // opacity_weight · mean((opacity − 1)²)`.
    let loss = if opacity_weight > 0.0 {
        let op_loss = g.mse_loss(opacity, ones_p1); // scalar [1]
        let opw = g.constant(vec![opacity_weight], &[1, 1]);
        let cl_2d = g.reshape(color_loss, &[1, 1]);
        let op_2d = g.reshape(op_loss, &[1, 1]);
        let op_scaled = g.mul(op_2d, opw);
        let total = g.add(cl_2d, op_scaled);
        g.reshape(total, &[1])
    } else {
        color_loss
    };

    // `dt_from_positions` as a second output so callers can read it
    // back and compare against the shader-computed dt during the
    // position-optimisation sanity check.
    g.set_outputs(vec![loss, dt_from_positions]);

    VolumetricGraph {
        n_cells,
        n_pixels,
        max_steps,
        sh_degree,
        num_views,
        log_density,
        positions,
        sh_coefficients,
        exposure_r,
        exposure_g,
        exposure_b,
        cell_indices,
        next_cell_indices,
        mask,
        ray_origin,
        ray_dir_per_pixel,
        pixel_idx_per_step,
        view_idx,
        basis_inputs,
        target,
        loss,
        dt_from_positions,
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

/// Build the L1 distance between finite-difference gradients of a single
/// channel for the rendered patch and the target patch. Both inputs are
/// `[q*q, 1]` (row-major flat patches); returns a scalar.
///
/// `diff_x[i, j] = X[i, j+1] - X[i, j]` is computed via matmul with a
/// constant `[q, q-1]` band matrix on the right. `diff_y[i, j] = X[i+1, j]
/// - X[i, j]` uses a `[q-1, q]` band matrix on the left. The matmuls are
/// differentiable end-to-end — `SplitA`/`SplitB` cannot be used here
/// because their autodiff backward is empty (gradients would silently
/// drop to zero before reaching the rendered pixels).
fn patch_grad_l1(
    g: &mut mn::Graph,
    pred: mn::NodeId,
    target: mn::NodeId,
    q: usize,
) -> mn::NodeId {
    // Reshape flat [q*q, 1] → [q, q] (rows × cols).
    let pred_2d = g.reshape(pred, &[q, q]);
    let target_2d = g.reshape(target, &[q, q]);

    // X-direction: D_x has shape [q, q-1], D_x[k, j] = +1 iff k == j+1,
    // -1 iff k == j, else 0. Then `X @ D_x` is shape [q, q-1] with
    // `(X @ D_x)[i, j] = X[i, j+1] - X[i, j]`.
    let mut d_x_data = vec![0.0_f32; q * (q - 1)];
    for j in 0..(q - 1) {
        d_x_data[j * (q - 1) + j] = -1.0; // row k=j, col j
        d_x_data[(j + 1) * (q - 1) + j] = 1.0; // row k=j+1, col j
    }
    let d_x = g.constant(d_x_data, &[q, q - 1]);
    let pred_dx = g.matmul(pred_2d, d_x);
    let tgt_dx = g.matmul(target_2d, d_x);
    let dx_flat_pred = g.reshape(pred_dx, &[1, q * (q - 1)]);
    let dx_flat_tgt = g.reshape(tgt_dx, &[1, q * (q - 1)]);
    let loss_dx = g.l1_loss(dx_flat_pred, dx_flat_tgt);

    // Y-direction: D_y has shape [q-1, q], D_y[i, k] = +1 iff k == i+1,
    // -1 iff k == i, else 0. Then `D_y @ X` is shape [q-1, q] with
    // `(D_y @ X)[i, j] = X[i+1, j] - X[i, j]`.
    let mut d_y_data = vec![0.0_f32; (q - 1) * q];
    for i in 0..(q - 1) {
        d_y_data[i * q + i] = -1.0; // row i, col k=i
        d_y_data[i * q + (i + 1)] = 1.0; // row i, col k=i+1
    }
    let d_y = g.constant(d_y_data, &[q - 1, q]);
    let pred_dy = g.matmul(d_y, pred_2d);
    let tgt_dy = g.matmul(d_y, target_2d);
    let dy_flat_pred = g.reshape(pred_dy, &[1, (q - 1) * q]);
    let dy_flat_tgt = g.reshape(tgt_dy, &[1, (q - 1) * q]);
    let loss_dy = g.l1_loss(dy_flat_pred, dy_flat_tgt);

    g.add(loss_dx, loss_dy)
}

/// SH basis constants — degree-0 first (matches `vol::trace::eval_rgb_sh`).
const SH_C0: f32 = 0.282_094_8;
/// SH degree-1 coefficient: `|Y_1m| = SH_C1 * <axis>` for the three m's.
pub const SH_C1: f32 = 0.488_602_5;
/// SH degree-2 coefficients (m = -2..2). Sign / axis pairings are applied
/// at basis-evaluation time, see [`sh_basis`].
pub const SH_C2: [f32; 5] = [
    1.092_548_4,
    -1.092_548_4,
    0.315_391_57,
    -1.092_548_4,
    0.546_274_2,
];
/// SH degree-3 coefficients (m = -3..3).
pub const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_33,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Camera quantities that don't depend on the pixel; computed once per
/// view and reused for every per-pixel ray during a step.
struct RayConstants {
    origin: glam::Vec3,
    orientation: glam::Quat,
    tan_half: glam::Vec2,
}

fn ray_constants(cam: &vol::CameraParams) -> RayConstants {
    RayConstants {
        origin: glam::Vec3::from_array(cam.cam_position),
        orientation: glam::Quat::from_xyzw(
            cam.cam_orientation[0],
            cam.cam_orientation[1],
            cam.cam_orientation[2],
            cam.cam_orientation[3],
        ),
        tan_half: glam::Vec2::new((0.5 * cam.fov[0]).tan(), (0.5 * cam.fov[1]).tan()),
    }
}

fn ray_dir_for_pixel(c: &RayConstants, ix: u32, iy: u32, w: u32, h: u32) -> glam::Vec3 {
    let px = (ix as f32 + 0.5) / w as f32;
    let py = (iy as f32 + 0.5) / h as f32;
    let ndc = glam::Vec2::new(px * 2.0 - 1.0, py * 2.0 - 1.0);
    let local = glam::Vec3::new(ndc.x * c.tan_half.x, ndc.y * c.tan_half.y, 1.0);
    let _ = c.origin;
    (c.orientation * local).normalize()
}

/// Evaluate the spherical-harmonics basis at unit direction `dir`, for
/// the first `num_components` components. Matches
/// `blade-volume/shaders/sh_eval.wgsl::sh_eval_color` term-by-term, so
/// CPU-precomputed bases used by the training graph stay numerically
/// consistent with the production renderer's GPU evaluation.
pub fn sh_basis(dir: glam::Vec3, num_components: usize) -> Vec<f32> {
    let mut b = Vec::with_capacity(num_components);
    let (x, y, z) = (dir.x, dir.y, dir.z);
    b.push(SH_C0);
    if num_components > 1 {
        b.push(-SH_C1 * y);
        b.push(SH_C1 * z);
        b.push(-SH_C1 * x);
    }
    if num_components > 4 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b.push(SH_C2[0] * x * y);
        b.push(SH_C2[1] * y * z);
        b.push(SH_C2[2] * (3.0 * zz - 1.0));
        b.push(SH_C2[3] * x * z);
        b.push(SH_C2[4] * (xx - yy));
    }
    if num_components > 9 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b.push(SH_C3[0] * y * (3.0 * xx - yy));
        b.push(SH_C3[1] * x * y * z);
        b.push(SH_C3[2] * y * (5.0 * zz - 1.0));
        b.push(SH_C3[3] * z * (5.0 * zz - 3.0));
        b.push(SH_C3[4] * x * (5.0 * zz - 1.0));
        b.push(SH_C3[5] * z * (xx - yy));
        b.push(SH_C3[6] * x * (xx - 3.0 * yy));
    }
    b.truncate(num_components);
    b
}

#[allow(clippy::too_many_arguments)]
fn channel_pixel_sh(
    g: &mut mn::Graph,
    cell_indices: mn::NodeId,
    sh_chans: &[mn::NodeId], // K parameter tables [N, 1]
    basis_inputs: &[mn::NodeId], // K-1 per-pixel basis [P, 1]; basis_0 is SH_C0 constant
    weight: mn::NodeId,
    ones_l1: mn::NodeId, // [L, 1] for the final reduce
    ones_1l: mn::NodeId, // [1, L] for broadcasting basis along the L axis
    p: usize,
    l: usize,
) -> mn::NodeId {
    let k = sh_chans.len();
    assert_eq!(
        basis_inputs.len(),
        k.saturating_sub(1),
        "channel_pixel_sh: basis_inputs.len() ({}) must equal sh_chans.len() - 1 ({})",
        basis_inputs.len(),
        k.saturating_sub(1),
    );

    // Component 0: per-cell colour scaled by the constant SH_C0.
    let color_flat = g.embedding(cell_indices, sh_chans[0]);
    let color_2d = g.reshape(color_flat, &[p, l]);
    let scale = g.constant(vec![SH_C0; p * l], &[p, l]);
    let mut color_total = g.mul(color_2d, scale);

    // Components 1..K: per-cell colour times per-pixel basis broadcast
    // across the L axis via matmul with `ones_1l`.
    for (idx, &sh_chan_k) in sh_chans.iter().enumerate().skip(1) {
        let color_flat_k = g.embedding(cell_indices, sh_chan_k);
        let color_k = g.reshape(color_flat_k, &[p, l]);
        let basis_k_2d = g.matmul(basis_inputs[idx - 1], ones_1l); // [P, L]
        let contrib = g.mul(color_k, basis_k_2d);
        color_total = g.add(color_total, contrib);
    }

    let bias = g.constant(vec![0.5; p * l], &[p, l]);
    let biased = g.add(color_total, bias);
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
    let mut total = 0usize;
    let mut max_dt = 0.0f32;
    let mut bad_dt = 0usize;
    for (pi, path) in paths.iter().enumerate() {
        let n = path.entries.len().min(max_steps);
        total += n;
        for (k, e) in path.entries[..n].iter().enumerate() {
            let idx = pi * max_steps + k;
            cell[idx] = e.cell;
            // Clamp pathological dt: rays starting outside any cell can give
            // huge segments that overflow sigmoid/recip in the graph; cap so
            // the cumsum stays in a numerically benign range.
            let dt_clamped = if !e.dt.is_finite() || e.dt < 0.0 {
                bad_dt += 1;
                0.0
            } else {
                e.dt.min(MAX_PATH_DT)
            };
            if dt_clamped > max_dt {
                max_dt = dt_clamped;
            }
            dt[idx] = dt_clamped;
            mask[idx] = if dt_clamped > 0.0 { 1.0 } else { 0.0 };
        }
    }
    if bad_dt > 0 {
        log::warn!("flatten_paths: clamped {bad_dt} non-finite dt values to zero");
    }
    log::debug!(
        "flatten_paths: {} paths, {} entries, max_dt {:.3}, total steps avg {:.1}",
        p,
        total,
        max_dt,
        total as f32 / p as f32,
    );
    FlatPaths { cell, dt, mask }
}

/// Path segments longer than this saturate the sigmoid surrogate for
/// `exp(-density * dt)` in the differentiable forward. Capping here keeps
/// gradients well-behaved without changing the renderer's CPU output (which
/// uses real `exp` and tolerates arbitrary dt).
pub const MAX_PATH_DT: f32 = 50.0;

#[derive(Clone, Debug)]
pub struct FlatPaths {
    pub cell: Vec<u32>,
    pub dt: Vec<f32>,
    pub mask: Vec<f32>,
}

/// Learning-rate schedule applied per Adam step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrSchedule {
    /// Constant: every step uses `learning_rate`.
    Constant,
    /// Cosine decay from `learning_rate` to `learning_rate * lr_min_ratio`
    /// over `total_steps` Adam updates: `lr(t) = lr_min + (lr_max - lr_min)
    /// * (1 + cos(π t / T)) / 2`. Default in most NeRF / 3DGS recipes;
    /// lets early training move fast and late training fine-tune.
    Cosine,
}

/// Knobs for [`fit_appearance_to_pixels`].
#[derive(Clone, Debug)]
pub struct AppearanceFitConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    /// Learning-rate schedule. Default `Cosine`.
    pub lr_schedule: LrSchedule,
    /// Floor of the cosine schedule, as a fraction of `learning_rate`.
    /// `0.01` decays to 1 % of base by the final step. Unused when
    /// `lr_schedule == Constant`.
    pub lr_min_ratio: f32,
    /// When `Some(K)`, each Adam step samples `K` random pixels from a
    /// randomly chosen training view rather than feeding every pixel of
    /// every view. Cheap stochastic mini-batching that scales to full
    /// image resolutions. `None` falls back to feeding every pixel at
    /// the view's resolution — fine for tiny synthetic tests.
    pub pixel_batch: Option<usize>,
    /// Number of Adam steps per view in batched mode. Used only when
    /// `pixel_batch.is_some()`. Default 200.
    pub steps_per_view: usize,
    /// SH degree for view-dependent colour. 0 = flat colour (default,
    /// matches the original radfoam pipeline). 1–3 enable view
    /// dependence: ~3-5 dB PSNR improvement on Mip-NeRF 360 scenes at
    /// the cost of `(1+sh_degree)²` per-cell parameters per RGB
    /// channel. Only the `pixel_batch`-mode pipeline supports SH > 0.
    pub sh_degree: usize,
    /// Adaptive densification: split the highest-gradient cells
    /// periodically during training. `None` disables densification
    /// entirely (geometry stays frozen at initialisation count).
    pub densify: Option<DensifyConfig>,
    /// Patch-based sampling + structural-gradient L1 loss. When
    /// `patch_size > 0`, each Adam step samples a single contiguous
    /// `patch_size × patch_size` patch (so `pixel_batch == patch_size²`),
    /// and the loss becomes `(1 - grad_loss_weight) * L1 + grad_loss_weight * L1_grad`
    /// where `L1_grad` is the L1 distance between rendered and target
    /// finite-difference gradients (∂/∂x, ∂/∂y). Acts as a poor-man's
    /// SSIM and captures edge structure that random-pixel L1 misses.
    /// `0` (default) keeps the legacy random-pixel L1.
    pub patch_size: usize,
    /// Weight on the gradient L1 term when `patch_size > 0`. Common
    /// choices in the literature are 0.1–0.2.
    pub grad_loss_weight: f32,
    /// Weight on the RadFoam opacity loss `mean((opacity − 1)²)`, which
    /// pushes every ray to full opacity (white-background composite) and
    /// suppresses semi-transparent floaters. `0.0` (default) disables it
    /// and keeps the legacy un-composited L1 path.
    pub opacity_weight: f32,
    /// Softplus β for the density activation. `0.0` (default) uses legacy
    /// ReLU; `> 0` (RadFoam uses 10) uses `(1/β)·softplus(βx)` so cells
    /// that dip to negative log-density keep a gradient and recover
    /// instead of dying — important for stable densification.
    pub softplus_beta: f32,
    /// When `Some(path)`, write a PLY checkpoint at the end of every
    /// densify cycle (≈ every `densify.every` steps) so a crash/reboot
    /// during a long run loses at most one cycle instead of the whole
    /// run. Exposure is baked into a clone, never the live model. `None`
    /// (default) disables intermediate checkpoints.
    pub checkpoint_path: Option<std::path::PathBuf>,
}

/// Adaptive densification: every `every` training steps after `warmup`
/// steps, split the top-`fraction` cells by accumulated `|log_density|`
/// gradient by inserting a sibling cell at `pos + jitter`. The sibling
/// inherits density and SH coefficients, so the colour stays continuous
/// across the split.
///
/// The accumulator runs in parallel with training and is reset every
/// cycle. Each split rebuilds the meganeura session, GPU cloud, and
/// (via Qhull) Voronoi adjacency — about 3 s overhead per cycle at 100K
/// cells, amortised over the `every` steps in between.
#[derive(Clone, Copy, Debug)]
pub struct DensifyConfig {
    /// Steps between densify rounds.
    pub every: usize,
    /// Per-round growth factor: each round adds `fraction × current_cells`
    /// new cells (RadFoam uses 0.15 = +15%/round), selected by weighted
    /// multinomial on `accumulated|grad(log_density)| × cell_radius`.
    pub fraction: f32,
    /// Unused legacy knob (sibling jitter); the RadFoam-style child
    /// placement (0.25× toward the farthest neighbour + a 0.1× random
    /// kick) is now hardcoded. Kept for CLI compatibility.
    pub jitter_scale: f32,
    /// Skip the first `warmup` steps before the first densify (lets
    /// the per-cell gradient signal settle). RadFoam: 2000.
    pub warmup: usize,
    /// Stop growing once the cell count reaches this budget (RadFoam
    /// bonsai: ~2,097,152).
    pub target_points: usize,
    /// Stop densifying after this step (refinement-only phase follows).
    /// RadFoam densifies iters 2000–11000 of 20000 (~55 %).
    pub densify_until: usize,
    /// Prune near-transparent small cells each densify round (the floater
    /// remover RadFoam has and we previously lacked).
    pub prune: bool,
    /// Prune a cell only if its post-activation density is below this.
    pub prune_density: f32,
    /// …and its farthest-neighbour radius is below this (only cull small
    /// dim cells; large empty background cells are kept).
    pub prune_radius: f32,
}

impl Default for DensifyConfig {
    fn default() -> Self {
        Self {
            every: 500,
            fraction: 0.15,
            jitter_scale: 0.5,
            warmup: 2000,
            target_points: 2_097_152,
            densify_until: usize::MAX,
            prune: true,
            prune_density: 0.01,
            prune_radius: 0.1,
        }
    }
}

impl Default for AppearanceFitConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epochs: 200,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            pixel_batch: None,
            steps_per_view: 200,
            sh_degree: 0,
            densify: None,
            lr_schedule: LrSchedule::Cosine,
            lr_min_ratio: 0.01,
            patch_size: 0,
            grad_loss_weight: 0.0,
            opacity_weight: 0.0,
            softplus_beta: 0.0,
            checkpoint_path: None,
        }
    }
}

/// Effective learning rate at Adam step `t` (1-indexed) given `total`
/// steps and the config's schedule. `t` may exceed `total` (e.g. when
/// resuming) — in that case the floor is returned.
fn lr_at_step(config: &AppearanceFitConfig, t: usize, total: usize) -> f32 {
    match config.lr_schedule {
        LrSchedule::Constant => config.learning_rate,
        LrSchedule::Cosine => {
            let lr_max = config.learning_rate;
            let lr_min = lr_max * config.lr_min_ratio;
            if total == 0 || t >= total {
                lr_min
            } else {
                let phase = (t as f32) / (total as f32);
                let cos_term = (std::f32::consts::PI * phase).cos();
                lr_min + (lr_max - lr_min) * 0.5 * (1.0 + cos_term)
            }
        }
    }
}

/// One supervised view: a camera + the pixel image we want the trained model
/// to reproduce at that camera. `target_rgb` is `width * height * 3` floats
/// in row-major RGB order. `start_cell` is the index of the Voronoi cell the
/// per-pixel rays of this view should start in — caller picks (usually via
/// a kd-tree from the camera origin).
#[derive(Clone)]
pub struct ViewSupervision {
    pub camera: vol::CameraParams,
    pub target_rgb: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub start_cell: u32,
}

/// Per-pixel ray for a view, matching `render::render_cpu`'s mapping.
fn rays_for_view(cam: &vol::CameraParams, w: u32, h: u32) -> Vec<vol::trace::Ray> {
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

/// Fit per-cell density + SH degree-0 DC of `model` so it reproduces every
/// view in `views`. Geometry is frozen — the cell-walk paths through each
/// view are recorded once at the start (positions and adjacency don't
/// change during appearance training) and reused for every Adam step.
///
/// All views must share the same `width` × `height` and `max_steps`. The
/// graph is built once for that fixed shape; per-epoch we cycle through
/// views and run one Adam step per view.
///
/// Returns one loss per Adam step (`epochs * views.len()` total).
pub fn fit_appearance_multi_view(
    model: &mut vol::PointCloudModel,
    views: &[ViewSupervision],
    width: u32,
    height: u32,
    max_steps: usize,
    config: AppearanceFitConfig,
    gpu: std::sync::Arc<blade_graphics::Context>,
) -> Vec<f32> {
    assert!(
        !views.is_empty(),
        "fit_appearance_multi_view needs >=1 view"
    );
    let n_cells = model.points.len();
    for v in views {
        assert!(
            (v.start_cell as usize) < n_cells,
            "view start_cell out of range"
        );
        assert_eq!(
            v.target_rgb.len() as u32,
            v.width * v.height * 3,
            "view target_rgb length mismatches its width*height*3"
        );
    }

    if let Some(k) = config.pixel_batch {
        return fit_appearance_pixel_batched(model, views, max_steps, k, config, gpu);
    }

    let p = (width as usize) * (height as usize);
    for v in views {
        assert_eq!(
            v.target_rgb.len(),
            p * 3,
            "whole-image mode: target_rgb length must equal width*height*3"
        );
    }

    // Record paths per view once; geometry is fixed during appearance training.
    let trace_settings = vol::trace::TraceSettings {
        start_point: 0, // overwritten per view below
        max_steps: max_steps as u32,
        weight_threshold: 0.0,
        depth: views[0].camera.depth,
        eval_mode: vol::trace::EvalMode::Sh,
    };
    let mut flats: Vec<FlatPaths> = Vec::with_capacity(views.len());
    for v in views {
        let mut s = trace_settings;
        s.start_point = v.start_cell;
        s.depth = v.camera.depth;
        let rays = rays_for_view(&v.camera, width, height);
        let paths: Vec<_> = rays
            .into_iter()
            .map(|ray| vol::trace::record_path(model, ray, s))
            .collect();
        flats.push(flatten_paths(&paths, max_steps));
    }

    let mut g = mn::Graph::new();
    let _vg =
        build_volumetric_graph(&mut g, n_cells, p, max_steps, 0, views.len().max(1), 0, 0.0, 0.0, 0.0);
    let (mut session, _report) = mn::build(
        &g,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu),
            ..Default::default()
        },
    );

    upload_model_parameters(&mut session, model, 0.0);

    let mut losses = Vec::with_capacity(config.epochs * views.len());
    let log_every = config.epochs.div_ceil(10).max(1);
    session.set_adam(
        config.learning_rate,
        config.adam_beta1,
        config.adam_beta2,
        config.adam_eps,
    );
    for epoch in 0..config.epochs {
        for (vi, v) in views.iter().enumerate() {
            session.set_input_u32("cell_indices", &flats[vi].cell);
            session.set_input("dt", &flats[vi].dt);
            session.set_input("mask", &flats[vi].mask);
            session.set_input("labels", &v.target_rgb);
            session.step();
            session.wait();
            let loss = session.read_output(1).first().copied().unwrap_or(f32::NAN);
            log::trace!("step {} (view {vi}): loss {loss}", losses.len());
            losses.push(loss);
        }
        if epoch == 0 || (epoch + 1).is_multiple_of(log_every) {
            let recent_avg: f32 =
                losses.iter().rev().take(views.len()).copied().sum::<f32>() / views.len() as f32;
            log::info!(
                "epoch {}/{}: avg loss {:.4}",
                epoch + 1,
                config.epochs,
                recent_avg
            );
        }
    }

    download_model_parameters(&session, model, 0.0);

    losses
}

fn upload_model_parameters(
    session: &mut mn::Session,
    model: &vol::PointCloudModel,
    softplus_beta: f32,
) {
    let n_cells = model.points.len();
    let mut init_density = Vec::with_capacity(n_cells);
    let mut init_positions = Vec::with_capacity(n_cells * 3);
    for point in &model.points {
        // `.w` stores the density; seed the raw `log_density` parameter by
        // inverting the activation so softplus(seed) == .w (exact round-trip).
        init_density.push(inv_density_activation(point.w, softplus_beta));
        init_positions.push(point.x);
        init_positions.push(point.y);
        init_positions.push(point.z);
    }
    session.set_parameter("log_density", &init_density);
    session.set_parameter("positions", &init_positions);

    // `model.sh_coefficients` layout (lib.rs spec):
    //   `[p0_c0_r, p0_c0_g, p0_c0_b, p0_c1_r, p0_c1_g, p0_c1_b, ..., p1_c0_r, ...]`
    // i.e. per point, RGB interleaved within each SH component, then
    // components contiguous. We unpack into `[N]` slices, one per
    // `sh_<chan>_<k>` parameter the graph declared.
    let num_components = model.sh_component_count();
    let row_stride = num_components * 3;
    let mut scratch = vec![0.0_f32; n_cells];
    for (chan_idx, chan) in ["sh_r", "sh_g", "sh_b"].iter().enumerate() {
        for k in 0..num_components {
            for (i, slot) in scratch.iter_mut().enumerate() {
                *slot = model.sh_coefficients[i * row_stride + k * 3 + chan_idx];
            }
            session.set_parameter(&parameter_name(chan, k), &scratch);
        }
    }
}

/// Bake the mean per-view exposure into the SH-DC coefficients so the
/// model evaluates correctly through a renderer that does not know
/// about the `exposure_*` parameters. During training,
/// `pixel * exposure[view_idx] ≈ target`; if Adam drives `mean(exposure)`
/// away from 1.0 the SH chain absorbs the inverse, leaving an un-
/// corrected eval (no exposure multiplier) systematically too bright
/// or too dark. Multiplying the SH-DC term per channel by the channel-
/// wise mean exposure removes the bias: a fresh tracer with no
/// exposure knowledge then sees the "average-exposure" calibration,
/// which is the closest a test-view render can get without learning a
/// new exposure for that view.
/// Atomically write a model to `path` as a binary PLY: write to a
/// sibling `.tmp` then rename, so a crash mid-write can't leave a
/// truncated checkpoint clobbering the previous good one.
fn save_checkpoint(
    path: &std::path::Path,
    model: &vol::PointCloudModel,
) -> Result<(), String> {
    let tmp = path.with_extension("ply.tmp");
    blade_volume_convert::save_ply_with_options(
        &tmp,
        model,
        &blade_volume_convert::SaveOptions {
            format: blade_volume_convert::PlyFormat::Binary,
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{e:?}"))?;
    Ok(())
}

fn bake_mean_exposure_into_sh(
    session: &mn::Session,
    model: &mut vol::PointCloudModel,
    num_views: usize,
) {
    if num_views == 0 {
        return;
    }
    let mut r = vec![0.0_f32; num_views];
    let mut g = vec![0.0_f32; num_views];
    let mut b = vec![0.0_f32; num_views];
    session.read_param("exposure_r", &mut r);
    session.read_param("exposure_g", &mut g);
    session.read_param("exposure_b", &mut b);
    let mean = |v: &[f32]| v.iter().copied().sum::<f32>() / v.len() as f32;
    let mean_r = mean(&r);
    let mean_g = mean(&g);
    let mean_b = mean(&b);
    // If exposure stayed at 1.0 (the OFF mode) this is a no-op.
    if (mean_r - 1.0).abs() < 1e-6 && (mean_g - 1.0).abs() < 1e-6 && (mean_b - 1.0).abs() < 1e-6 {
        return;
    }
    eprintln!(
        "baking mean exposure into SH DC: r={mean_r:.4} g={mean_g:.4} b={mean_b:.4}"
    );
    let num_components = model.sh_component_count();
    let row_stride = num_components * 3;
    let mean_per_channel = [mean_r, mean_g, mean_b];
    for i in 0..model.points.len() {
        for (c, scale) in mean_per_channel.iter().enumerate() {
            // SH layout: [p_c0_r, p_c0_g, p_c0_b, p_c1_r, ...]. Only
            // the DC term (k=0) needs scaling; higher-order SH
            // components are view-dependent details that exposure
            // should not bake into.
            let dc_idx = i * row_stride + c;
            model.sh_coefficients[dc_idx] *= scale;
        }
    }
}

fn debug_dump_exposure(session: &mn::Session, num_views: usize) {
    let mut r = vec![0.0_f32; num_views];
    let mut g = vec![0.0_f32; num_views];
    let mut b = vec![0.0_f32; num_views];
    session.read_param("exposure_r", &mut r);
    session.read_param("exposure_g", &mut g);
    session.read_param("exposure_b", &mut b);
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for buf in [&r, &g, &b] {
        for &v in buf {
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
            n += 1;
        }
    }
    let mean = sum / n as f64;
    eprintln!(
        "exposure[{num_views}]: min={min:.4} max={max:.4} mean={mean:.4} \
         first_view=[r={:.3}, g={:.3}, b={:.3}]",
        r[0], g[0], b[0],
    );
}

/// Density activation applied on the host when baking the trained
/// `log_density` parameter into `model.points[i].w` (the density the eval
/// tracer reads). MUST match `build_volumetric_graph`'s in-graph
/// activation or training and eval disagree (an activation mismatch
/// silently collapses eval PSNR while training loss looks fine).
/// `beta <= 0` = legacy ReLU; `beta > 0` = `(1/β)·softplus(βx)`.
fn density_activation(x: f32, beta: f32) -> f32 {
    if beta <= 0.0 {
        return x.max(0.0);
    }
    let bx = beta * x;
    let sp = if bx > 20.0 { bx } else { (1.0 + bx.exp()).ln() };
    sp / beta
}

/// Inverse of [`density_activation`] — recovers a `log_density` seed from a
/// stored density so the `param → .w → param` round-trip across densify
/// rebuilds is exact (softplus preserves negatives, unlike ReLU's clamp).
fn inv_density_activation(y: f32, beta: f32) -> f32 {
    if beta <= 0.0 {
        return y; // ReLU stores density (≥0) directly as the log-density seed
    }
    let by = beta * y;
    let inv = if by > 20.0 {
        by
    } else {
        (by.exp() - 1.0).max(1e-20).ln()
    };
    inv / beta
}

fn download_model_parameters(
    session: &mn::Session,
    model: &mut vol::PointCloudModel,
    softplus_beta: f32,
) {
    let n_cells = model.points.len();
    let mut out_density = vec![0.0f32; n_cells];
    session.read_param("log_density", &mut out_density);
    for (i, d) in out_density.iter().enumerate() {
        model.points[i].w = density_activation(*d, softplus_beta);
    }

    let mut out_positions = vec![0.0_f32; n_cells * 3];
    session.read_param("positions", &mut out_positions);
    for i in 0..n_cells {
        model.points[i].x = out_positions[i * 3];
        model.points[i].y = out_positions[i * 3 + 1];
        model.points[i].z = out_positions[i * 3 + 2];
    }

    let num_components = model.sh_component_count();
    let row_stride = num_components * 3;
    let mut out_chan = vec![0.0f32; n_cells];
    for (chan_idx, chan) in ["sh_r", "sh_g", "sh_b"].iter().enumerate() {
        for k in 0..num_components {
            session.read_param(&parameter_name(chan, k), &mut out_chan);
            for (i, &c) in out_chan.iter().enumerate() {
                model.sh_coefficients[i * row_stride + k * 3 + chan_idx] = c;
            }
        }
    }
}

/// Per-cell farthest-neighbour distance (`cell_radius`) and the index of
/// that farthest neighbour, from the current adjacency CSR. `cell_radius`
/// weights densification (big high-error cells get subdivided) and the
/// farthest-neighbour direction is where RadFoam places the split child.
fn per_cell_farthest(model: &vol::PointCloudModel) -> (Vec<f32>, Vec<usize>) {
    let n = model.points.len();
    let mut radius = vec![0.0f32; n];
    let mut farthest: Vec<usize> = (0..n).collect();
    let Some(adj) = model.adjacency.as_ref() else {
        return (vec![1.0; n], farthest);
    };
    for i in 0..n {
        let pi = glam::Vec3::new(model.points[i].x, model.points[i].y, model.points[i].z);
        let start = adj.offsets[i] as usize;
        let end = adj.offsets[i + 1] as usize;
        let mut best = 0.0f32;
        let mut bj = i;
        for &j in &adj.neighbors[start..end] {
            let pj_v = model.points[j as usize];
            let pj = glam::Vec3::new(pj_v.x, pj_v.y, pj_v.z);
            let d = (pi - pj).length();
            if d > best {
                best = d;
                bj = j as usize;
            }
        }
        radius[i] = best;
        farthest[i] = bj;
    }
    (radius, farthest)
}

/// One RadFoam-style prune+densify round on the post-download `model`
/// (positions in `points.xyz`, density in `points.w`, current adjacency).
///
/// 1. **Prune** small near-transparent cells (`density < prune_density &&
///    cell_radius < prune_radius`) — the floater remover.
/// 2. **Densify** by appending `fraction × survivors` children, parents
///    drawn by weighted multinomial (without replacement) on
///    `grad_accum × cell_radius`. Each child sits 0.25× toward the
///    parent's farthest neighbour plus a small random kick, inheriting the
///    parent's density + SH.
///
/// Returns `(new_to_old, pruned, added)`: `new_to_old[j]` is the OLD cell
/// index whose Adam (m,v) the rebuilt cell `j` should inherit (survivor →
/// itself, child → parent), used to carry optimiser momentum across the
/// session rebuild.
fn prune_and_densify(
    model: &mut vol::PointCloudModel,
    grad_accum: &[f32],
    cfg: &DensifyConfig,
    rng_state: &mut u64,
) -> (Vec<usize>, usize, usize) {
    let n_old = model.points.len();
    let sh_block = model.sh_component_count() * 3;
    let (radius, farthest) = per_cell_farthest(model);

    let mut next_unit = || {
        *rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*rng_state >> 32) as i32 as f32) / (i32::MAX as f32)
    };

    // --- Prune ---
    let survivors: Vec<usize> = (0..n_old)
        .filter(|&i| {
            if !cfg.prune {
                return true;
            }
            let dim = model.points[i].w < cfg.prune_density;
            let small = radius[i] < cfg.prune_radius;
            !(dim && small)
        })
        .collect();
    let n_surv = survivors.len();
    let pruned = n_old - n_surv;

    // --- Densify: weighted sampling without replacement (Efraimidis–
    // Spirakis: key = ln(u)/w, take the top-`want` keys). ---
    let headroom = cfg.target_points.saturating_sub(n_surv);
    let want = (((n_surv as f32) * cfg.fraction).round() as usize).min(headroom);
    let parents_local: Vec<usize> = if want == 0 {
        Vec::new()
    } else {
        let mut keyed: Vec<(f32, usize)> = (0..n_surv)
            .map(|local| {
                let oi = survivors[local];
                let w = (grad_accum[oi] * radius[oi]).max(1e-12);
                let u = (next_unit() * 0.5 + 0.5).clamp(1e-6, 1.0); // (0,1]
                (u.ln() / w, local)
            })
            .collect();
        let k = want.min(keyed.len());
        keyed.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        keyed.truncate(k);
        keyed.into_iter().map(|(_, local)| local).collect()
    };
    let added = parents_local.len();

    // --- Rebuild model arrays: survivors compacted, then children ---
    let n_new = n_surv + added;
    let mut new_points = Vec::with_capacity(n_new);
    let mut new_sh = Vec::with_capacity(n_new * sh_block);
    let mut new_to_old = Vec::with_capacity(n_new);
    for &oi in &survivors {
        new_points.push(model.points[oi]);
        new_sh.extend_from_slice(&model.sh_coefficients[oi * sh_block..(oi + 1) * sh_block]);
        new_to_old.push(oi);
    }
    for &local in &parents_local {
        let oi = survivors[local];
        let p = model.points[oi];
        let pf = model.points[farthest[oi]];
        let toward = glam::Vec3::new(pf.x - p.x, pf.y - p.y, pf.z - p.z) * 0.25;
        let kick_scale = (toward.length() * 0.1).max(1e-5);
        let kick = glam::Vec3::new(next_unit(), next_unit(), next_unit()) * kick_scale;
        let off = toward + kick;
        new_points.push(glam::Vec4::new(p.x + off.x, p.y + off.y, p.z + off.z, p.w));
        new_sh.extend_from_slice(&model.sh_coefficients[oi * sh_block..(oi + 1) * sh_block]);
        new_to_old.push(oi);
    }
    model.points = new_points;
    model.sh_coefficients = new_sh;
    (new_to_old, pruned, added)
}

/// Enumerate every per-cell parameter name with its per-cell element
/// stride. Stride is 1 for scalar tables (`log_density`, `sh_<chan>_<k>`)
/// and 3 for `positions`. Matches `build_volumetric_graph`'s declaration
/// order.
fn per_cell_param_names_with_stride(sh_degree: usize) -> Vec<(String, usize)> {
    let num_components = (1 + sh_degree) * (1 + sh_degree);
    let mut names = vec![("log_density".to_string(), 1), ("positions".to_string(), 3)];
    for chan in ["sh_r", "sh_g", "sh_b"] {
        for k in 0..num_components {
            names.push((parameter_name(chan, k), 1));
        }
    }
    names
}

/// Snapshot of Adam optimizer state for every per-cell parameter at
/// the current cell count. Used to carry momentum across a session
/// rebuild when densification grows the cell table.
struct AdamSnapshot {
    /// Per-parameter entries in compile order. Each holds the param
    /// name, the per-cell stride (1 or 3), and flat (m, v) buffers of
    /// length `n_cells * stride`.
    entries: Vec<AdamEntry>,
    /// Adam step counter (bias-correction `t`).
    t: u32,
    /// Cell count at snapshot time.
    n_cells: usize,
}

struct AdamEntry {
    name: String,
    stride: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

fn save_adam_state(session: &mn::Session, sh_degree: usize, n_cells: usize) -> AdamSnapshot {
    let names = per_cell_param_names_with_stride(sh_degree);
    let mut entries = Vec::with_capacity(names.len());
    for (name, stride) in names {
        let size = n_cells * stride;
        let mut m = vec![0.0_f32; size];
        let mut v = vec![0.0_f32; size];
        session.read_adam_m(&name, &mut m);
        session.read_adam_v(&name, &mut v);
        entries.push(AdamEntry { name, stride, m, v });
    }
    AdamSnapshot {
        entries,
        t: session.adam_step_count(),
        n_cells,
    }
}

/// Restore Adam state into a freshly built session after a prune+densify
/// round remapped the cell table. `new_to_old[j]` is the OLD cell index
/// whose `(m, v)` the rebuilt cell `j` inherits — survivors map to their
/// former slot (momentum preserved through pruning's compaction), split
/// children map to their parent (a twin should share the parent's
/// momentum; starting children at `(m=0, v=0)` against a converged
/// neighbourhood causes destabilising first-step updates).
///
/// The Adam step counter is preserved exactly so bias correction stays
/// continuous.
fn restore_adam_state_remap(
    session: &mn::Session,
    snap: &AdamSnapshot,
    new_to_old: &[usize],
    sh_degree: usize,
) {
    let n_new = new_to_old.len();
    let names = per_cell_param_names_with_stride(sh_degree);
    debug_assert_eq!(names.len(), snap.entries.len());
    for (i, (_name, stride)) in names.iter().enumerate() {
        let entry = &snap.entries[i];
        debug_assert_eq!(entry.stride, *stride);
        let s = *stride;
        let mut m = vec![0.0_f32; n_new * s];
        let mut v = vec![0.0_f32; n_new * s];
        for (j, &oi) in new_to_old.iter().enumerate() {
            m[j * s..j * s + s].copy_from_slice(&entry.m[oi * s..oi * s + s]);
            v[j * s..j * s + s].copy_from_slice(&entry.v[oi * s..oi * s + s]);
        }
        session.write_adam_m(&entry.name, &m);
        session.write_adam_v(&entry.name, &v);
    }
}

/// Build (or rebuild) the meganeura session + GPU cloud for the current
/// model. Sized for `pixel_batch` and `max_steps`; both are constant
/// across densify cycles so only the parameter-count-dependent pieces
/// (graph parameters, embedding tables, GPU adjacency buffer) get
/// rebuilt. Path-record buffers and the recorder pipeline survive the
/// rebuild.
#[allow(clippy::too_many_arguments)]
fn build_train_session(
    model: &vol::PointCloudModel,
    pixel_batch: usize,
    max_steps: usize,
    sh_degree: usize,
    num_views: usize,
    patch_size: usize,
    grad_loss_weight: f32,
    opacity_weight: f32,
    softplus_beta: f32,
    gpu: &std::sync::Arc<blade_graphics::Context>,
    path_bufs: &vol::gpu::PathRecordBuffers,
    lr: f32,
    betas: (f32, f32, f32),
) -> (mn::Session, vol::RadFoamGpuCloud) {
    let n_cells = model.points.len();
    let mut g = mn::Graph::new();
    let _vg = build_volumetric_graph(
        &mut g,
        n_cells,
        pixel_batch,
        max_steps,
        sh_degree,
        num_views,
        patch_size,
        grad_loss_weight,
        opacity_weight,
        softplus_beta,
    );
    let (mut session, _report) = mn::build(
        &g,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu.clone()),
            ..Default::default()
        },
    );
    upload_model_parameters(&mut session, model, softplus_beta);

    let gpu_cloud = {
        let mut init_encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "path-record-init",
            buffer_count: 1,
        });
        let cloud = vol::RadFoamGpuCloud::new(model, gpu, &mut init_encoder);
        gpu.destroy_command_encoder(&mut init_encoder);
        cloud
    };

    // The graph computes `dt` from `positions` (differentiable), so the
    // shader's `dt` output isn't bound as a meganeura input anymore. We
    // still let the shader write it (cheap; useful for sanity checks).
    let pl_bytes = (pixel_batch as u64) * (max_steps as u64) * 4;
    for (slot, buf) in [
        ("cell_indices", path_bufs.cells),
        ("next_cell_indices", path_bufs.next_cells),
        ("mask", path_bufs.mask),
    ] {
        let source = gpu
            .get_external_buffer_source(buf)
            .expect("PathRecordBuffers::new_external must produce an exportable buffer");
        session
            .bind_external_buffer(meganeura::ExternalSlot::Input(slot), source, pl_bytes)
            .unwrap_or_else(|err| panic!("bind_external_buffer({slot}) failed: {err:?}"));
    }

    session.set_adam(lr, betas.0, betas.1, betas.2);
    if sh_degree >= 1 {
        for chan in ["sh_r_", "sh_g_", "sh_b_"] {
            session.set_lr_multiplier(chan, 0.1);
        }
    }
    // Positions are differentiable parameters but frozen by default.
    // Unfreezing requires periodically rebuilding adjacency (the GPU
    // path-record walks a CSR captured at training start). The May
    // 2026 50K-cell × 128² A/B at ratio=0.01 dropped train PSNR from
    // 20.88 → 9.69 dB and test PSNR from 17.77 → 7.36 dB versus the
    // ratio=0 baseline: the trained gradient targets a geometry that
    // doesn't survive a fresh trace. Opt in via
    // `BLADE_VOLUME_POSITION_LR_RATIO=0.01` once in-loop rebuilds land.
    let position_lr_ratio = std::env::var("BLADE_VOLUME_POSITION_LR_RATIO")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    session.set_lr_multiplier("positions", position_lr_ratio);

    // Per-view exposure: init to 1.0 (identity). Defaults to enabled
    // (LR multiplier 1.0) when more than one view is being trained;
    // set `BLADE_VOLUME_PER_VIEW_EXPOSURE=0` to freeze at 1.0 for an
    // apples-to-apples comparison. With `num_views == 1` there's only
    // one row, equivalent to a global gain — still safe at LR 1.0,
    // but the multiplier is forced to 0 to keep the single-view case
    // identical to pre-feature behaviour.
    let exposure_init = vec![1.0_f32; num_views];
    session.set_parameter("exposure_r", &exposure_init);
    session.set_parameter("exposure_g", &exposure_init);
    session.set_parameter("exposure_b", &exposure_init);
    // Default OFF: the May 2026 production A/B at 75K × 256² × SH-3 ×
    // 6400-steps with LR ratio 0.05 still degraded test PSNR by
    // 3.3 dB (18.00 → 14.67). Exposure absorbed per-view brightness
    // variance during training, but eval (a fresh tracer with no
    // exposure knowledge) sees a model calibrated against the
    // average-exposure brightness — close for views near the mean,
    // far for the rest. The `bake_mean_exposure_into_sh` step
    // post-training renormalises the SH-DC term, but cannot recover
    // the per-view residual. Set `BLADE_VOLUME_PER_VIEW_EXPOSURE=$r`
    // to opt in (r ~ 0.01–0.05; anything higher overshoots).
    let exposure_lr_ratio = std::env::var("BLADE_VOLUME_PER_VIEW_EXPOSURE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    // "exposure_" prefix matches exposure_r/g/b at once.
    session.set_lr_multiplier("exposure_", exposure_lr_ratio);

    (session, gpu_cloud)
}

/// Pixel-batched training mode. Each Adam step picks a random training view
/// and `pixel_batch` random pixels from its image, records paths through the
/// current model, and runs one optimiser step. Allows training at full image
/// resolution without exceeding the meganeura matmul shape limits.
fn fit_appearance_pixel_batched(
    model: &mut vol::PointCloudModel,
    views: &[ViewSupervision],
    max_steps: usize,
    pixel_batch: usize,
    config: AppearanceFitConfig,
    gpu: std::sync::Arc<blade_graphics::Context>,
) -> Vec<f32> {
    let n_cells = model.points.len();
    let total_steps = config.steps_per_view.max(1) * views.len();

    // The model's existing SH degree drives the graph: callers wanting to
    // expand a SH-0 model to a higher degree must reshape
    // `model.sh_coefficients` and update `model.sh_degree` before calling
    // this function.
    let sh_degree = config.sh_degree.max(model.sh_degree);
    let num_components = (1 + sh_degree) * (1 + sh_degree);
    if model.sh_degree < sh_degree {
        // Expand `sh_coefficients` to the wider SH layout, zero-padding
        // the new components. Existing colour (component 0) is
        // preserved exactly; the higher-order coefficients start at
        // zero, so the initial colour matches a SH-0 model.
        let n_cells_local = model.points.len();
        let old_stride = model.sh_component_count() * 3;
        let new_stride = num_components * 3;
        let mut expanded = vec![0.0_f32; n_cells_local * new_stride];
        for i in 0..n_cells_local {
            // Component 0 = first 3 entries per cell.
            for chan in 0..3 {
                expanded[i * new_stride + chan] = model.sh_coefficients[i * old_stride + chan];
            }
        }
        model.sh_coefficients = expanded;
        model.sh_degree = sh_degree;
    }

    // Path-recorder pipeline and the record-side encoder are
    // cycle-independent. `PathRecordBuffers`, however, holds three
    // `Memory::External(Fd(None))` buffers; meganeura's
    // `bind_external_buffer` consumes those FDs on import (Vulkan
    // takes ownership), so once a session has imported them the
    // producer's `buffer.external` field is stale. We therefore
    // recreate `path_bufs` alongside the session+cloud each densify
    // cycle. The buffers are small (≤1.5 MB total at our shape).
    let recorder = vol::gpu::PathRecorder::new(&gpu);
    let pl_bytes = (pixel_batch as u64) * (max_steps as u64) * 4;
    let mut record_encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "path-record-step",
        buffer_count: 2,
    });

    // Deterministic LCG so reruns produce identical results.
    let mut state: u64 = 0xDEAD_BEEF_F00D_CAFE;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };

    let mut target_buf = vec![0.0f32; pixel_batch * 3];
    let mut pixel_indices = vec![0u32; pixel_batch];
    let mut basis_inputs: Vec<Vec<f32>> = (1..num_components)
        .map(|_| vec![0.0_f32; pixel_batch])
        .collect();
    // Position-opt graph inputs:
    //   - ray_origin (per view, [1,3]): camera position
    //   - ray_dir_per_pixel ([P, 3]): per-pixel ray direction
    //   - pixel_idx_per_step ([P*L]): constant gather index for
    //     broadcasting per-pixel data across the L step dimension
    let mut ray_origin_buf = vec![0.0_f32; 3];
    let mut ray_dir_per_pixel_buf = vec![0.0_f32; pixel_batch * 3];
    let pixel_idx_per_step: Vec<u32> = (0..(pixel_batch * max_steps))
        .map(|i| (i / max_steps) as u32)
        .collect();

    let mut losses = Vec::with_capacity(total_steps);
    // Frequent loss readouts (every ~2000 steps) so long multi-hour runs
    // surface their trajectory instead of only ~10 lines total.
    let log_every = 2000.min(total_steps).max(1);

    // Densification splits cells between training cycles. The
    // session + GPU cloud + path-record buffers are kept alive across
    // cycle boundaries when no split happens, so Adam momentum is
    // preserved through the warmup → first-densify transition. They're
    // only torn down and rebuilt when a split actually changes the cell
    // count.
    let densify = config.densify;
    let mut steps_done = 0usize;
    let mut grad_accum = vec![0.0f32; model.points.len()];
    let mut grad_scratch = vec![0.0f32; model.points.len()];
    let mut rng_split: u64 = 0xCAFE_F00D_DEAD_BEEF;
    let mut cycle = 0usize;

    let _ = n_cells;
    let mut path_bufs =
        vol::gpu::PathRecordBuffers::new_external(&gpu, pixel_batch as u32, max_steps as u32);
    let patch_size = config.patch_size;
    let grad_loss_weight = config.grad_loss_weight;
    let (mut session, mut gpu_cloud) = build_train_session(
        model,
        pixel_batch,
        max_steps,
        sh_degree,
        views.len(),
        patch_size,
        grad_loss_weight,
        config.opacity_weight,
        config.softplus_beta,
        &gpu,
        &path_bufs,
        config.learning_rate,
        (config.adam_beta1, config.adam_beta2, config.adam_eps),
    );

    // `pixel_idx_per_step` is constant across all training steps —
    // upload once. The new session built for each densify cycle gets
    // its own upload below.
    session.set_input_u32("pixel_idx_per_step", &pixel_idx_per_step);

    while steps_done < total_steps {
        let cycle_budget = match densify {
            Some(d) => {
                if steps_done < d.warmup {
                    (d.warmup - steps_done).min(total_steps - steps_done)
                } else {
                    d.every.min(total_steps - steps_done)
                }
            }
            None => total_steps - steps_done,
        };

        for cycle_step in 0..cycle_budget {
            let step = steps_done + cycle_step;
            let vi = (next_u32() as usize) % views.len();
            let v = &views[vi];
            let img_size = v.width * v.height;

            let cam_ray_constants = ray_constants(&v.camera);
            ray_origin_buf[0] = cam_ray_constants.origin.x;
            ray_origin_buf[1] = cam_ray_constants.origin.y;
            ray_origin_buf[2] = cam_ray_constants.origin.z;

            // Two sampling modes:
            //   - random pixels: pick `pixel_batch` independent random
            //     pixels (legacy behaviour, used with patch_size == 0).
            //   - patch: pick a random `q × q` patch corner and emit
            //     pixel indices in row-major order across the patch, so
            //     the graph can treat them as a 2D image for gradient
            //     L1.
            if patch_size > 0 {
                let q = patch_size as u32;
                let max_x = v.width.saturating_sub(q);
                let max_y = v.height.saturating_sub(q);
                let x0 = next_u32() % (max_x + 1);
                let y0 = next_u32() % (max_y + 1);
                for ky in 0..q {
                    for kx in 0..q {
                        let k = (ky * q + kx) as usize;
                        let ix = x0 + kx;
                        let iy = y0 + ky;
                        let pidx = iy * v.width + ix;
                        pixel_indices[k] = pidx;
                        let base = (pidx as usize) * 3;
                        target_buf[k * 3] = v.target_rgb[base];
                        target_buf[k * 3 + 1] = v.target_rgb[base + 1];
                        target_buf[k * 3 + 2] = v.target_rgb[base + 2];
                        let dir = ray_dir_for_pixel(&cam_ray_constants, ix, iy, v.width, v.height);
                        ray_dir_per_pixel_buf[k * 3] = dir.x;
                        ray_dir_per_pixel_buf[k * 3 + 1] = dir.y;
                        ray_dir_per_pixel_buf[k * 3 + 2] = dir.z;
                        if num_components > 1 {
                            let basis = sh_basis(dir, num_components);
                            for (j, &bv) in basis.iter().enumerate().skip(1) {
                                basis_inputs[j - 1][k] = bv;
                            }
                        }
                    }
                }
            } else {
                for k in 0..pixel_batch {
                    let pidx = next_u32() % img_size;
                    pixel_indices[k] = pidx;
                    let base = (pidx as usize) * 3;
                    target_buf[k * 3] = v.target_rgb[base];
                    target_buf[k * 3 + 1] = v.target_rgb[base + 1];
                    target_buf[k * 3 + 2] = v.target_rgb[base + 2];

                    let ix = pidx % v.width;
                    let iy = pidx / v.width;
                    let dir = ray_dir_for_pixel(&cam_ray_constants, ix, iy, v.width, v.height);
                    ray_dir_per_pixel_buf[k * 3] = dir.x;
                    ray_dir_per_pixel_buf[k * 3 + 1] = dir.y;
                    ray_dir_per_pixel_buf[k * 3 + 2] = dir.z;

                    if num_components > 1 {
                        let basis = sh_basis(dir, num_components);
                        for (j, &bv) in basis.iter().enumerate().skip(1) {
                            basis_inputs[j - 1][k] = bv;
                        }
                    }
                }
            }
            path_bufs.write_pixel_indices(&pixel_indices);
            session.set_input("ray_origin", &ray_origin_buf);
            session.set_input("ray_dir_per_pixel", &ray_dir_per_pixel_buf);
            for (j, vec) in basis_inputs.iter().enumerate() {
                session.set_input(&format!("basis_{}", j + 1), vec);
            }

            record_encoder.start();
            {
                let mut tx = record_encoder.transfer("path-record-prepare");
                let pix_size = (pixel_batch * std::mem::size_of::<u32>()) as u64;
                tx.copy_buffer_to_buffer(
                    path_bufs.pixel_indices_stage.at(0),
                    path_bufs.pixel_indices.at(0),
                    pix_size,
                );
                tx.fill_buffer(path_bufs.cells.at(0), pl_bytes, 0);
                tx.fill_buffer(path_bufs.next_cells.at(0), pl_bytes, 0);
                tx.fill_buffer(path_bufs.dts.at(0), pl_bytes, 0);
                tx.fill_buffer(path_bufs.mask.at(0), pl_bytes, 0);
            }
            recorder.dispatch(
                &mut record_encoder,
                &gpu_cloud,
                path_bufs.pixel_indices.into(),
                path_bufs.cells.into(),
                path_bufs.next_cells.into(),
                path_bufs.dts.into(),
                path_bufs.mask.into(),
                vol::gpu::RecordPathsArgs {
                    camera: v.camera,
                    start_point: v.start_cell,
                    max_steps: max_steps as u32,
                    image_width: v.width,
                    image_height: v.height,
                    max_path_dt: MAX_PATH_DT,
                    depth: v.camera.depth,
                    num_pixels: pixel_batch as u32,
                },
            );
            let _ = gpu.submit(&mut record_encoder);

            session.set_input("labels", &target_buf);
            session.set_input_u32("view_idx", &[vi as u32]);
            // Apply LR schedule: re-set Adam every step with the
            // current effective LR. `set_adam` is cheap (just updates
            // a session field), and per-parameter LR multipliers (set
            // once at build_train_session) survive across set_adam.
            let lr_now = lr_at_step(&config, step, total_steps);
            session.set_adam(lr_now, config.adam_beta1, config.adam_beta2, config.adam_eps);
            session.step();
            session.wait();
            let loss = session.read_output(1).first().copied().unwrap_or(f32::NAN);
            losses.push(loss);

            if densify.is_some() && session.has_param_grad("log_density") {
                session.read_param_grad("log_density", &mut grad_scratch);
                for (a, g) in grad_accum.iter_mut().zip(grad_scratch.iter()) {
                    *a += g.abs();
                }
            }

            if step == 0 || (step + 1).is_multiple_of(log_every) {
                let window: usize = log_every.min(losses.len());
                let recent_avg: f32 =
                    losses.iter().rev().take(window).copied().sum::<f32>() / window as f32;
                log::info!(
                    "step {}/{}: avg loss {:.4} (window {}) cells={}",
                    step + 1,
                    total_steps,
                    recent_avg,
                    window,
                    model.points.len(),
                );
            }
        }

        steps_done += cycle_budget;
        cycle += 1;

        // Crash-safety checkpoint: write the current model to disk at the
        // end of every cycle so a reboot/suspend during a multi-hour run
        // loses at most one cycle's progress, not the whole run. The PLY
        // only gets written at the very end otherwise. Downloading into
        // the live `model` mid-cycle is harmless (host state is only
        // pushed back to the GPU at densify rebuilds); exposure is baked
        // into a throwaway clone so the live model is never mutated.
        if let Some(ckpt) = &config.checkpoint_path {
            download_model_parameters(&session, model, config.softplus_beta);
            let mut snapshot = model.clone();
            bake_mean_exposure_into_sh(&session, &mut snapshot, views.len());
            match save_checkpoint(ckpt, &snapshot) {
                Ok(()) => log::info!(
                    "checkpoint: wrote {} ({} cells) at step {}",
                    ckpt.display(),
                    snapshot.points.len(),
                    steps_done,
                ),
                Err(err) => log::warn!("checkpoint save failed: {err:?}"),
            }
        }

        // Prune + densify, gated by the RadFoam schedule: only between
        // `warmup` and `densify_until`, and only while under the point
        // budget. Past that it's a refinement-only phase (no rebuilds).
        if let Some(d) = densify {
            let gate = steps_done >= d.warmup
                && steps_done < total_steps
                && steps_done < d.densify_until
                && model.points.len() < d.target_points;
            if gate {
                // Snapshot params + Adam state at the OLD size, before
                // prune+densify remaps `model.points`.
                let n_old = model.points.len();
                download_model_parameters(&session, model, config.softplus_beta);
                let adam_snap = save_adam_state(&session, sh_degree, n_old);

                let (new_to_old, pruned, added) =
                    prune_and_densify(model, &grad_accum, &d, &mut rng_split);
                log::info!(
                    "densify cycle {}: {} cells (-{} pruned, +{} split) → {} total",
                    cycle,
                    n_old,
                    pruned,
                    added,
                    model.points.len(),
                );
                // Rebuild Voronoi adjacency for the new cell set so the
                // GPU path-record sees real neighbours of the new cells.
                let adj = vol::compute_adjacency_qhull_default(&model.points);
                model.adjacency = Some(adj);
                grad_accum = vec![0.0f32; model.points.len()];
                grad_scratch = vec![0.0f32; model.points.len()];

                // Topology changed: tear down and rebuild the
                // cell-count-dependent resources. Adam state is reset
                // here (meganeura's optimizer state lives in the
                // session); the first few-hundred steps of the next
                // cycle pay that cost re-warming momentum.
                drop(session);
                gpu_cloud.deinit(&gpu);
                path_bufs.destroy(&gpu);
                path_bufs = vol::gpu::PathRecordBuffers::new_external(
                    &gpu,
                    pixel_batch as u32,
                    max_steps as u32,
                );
                let rebuilt = build_train_session(
                    model,
                    pixel_batch,
                    max_steps,
                    sh_degree,
                    views.len(),
                    patch_size,
                    grad_loss_weight,
                    config.opacity_weight,
                    config.softplus_beta,
                    &gpu,
                    &path_bufs,
                    config.learning_rate,
                    (config.adam_beta1, config.adam_beta2, config.adam_eps),
                );
                session = rebuilt.0;
                gpu_cloud = rebuilt.1;
                // pixel_idx_per_step is constant across cycles — re-upload
                // to the fresh session.
                session.set_input_u32("pixel_idx_per_step", &pixel_idx_per_step);

                // Restore Adam momentum: survivors keep their (m, v),
                // children inherit their parent's, via the new_to_old
                // remap. Bias-correction `t` is preserved so the
                // optimiser doesn't think it's step 1 again.
                restore_adam_state_remap(&session, &adam_snap, &new_to_old, sh_degree);
                session.set_adam_step_count(adam_snap.t);
            }
        }
    }

    debug_dump_exposure(&session, views.len());
    download_model_parameters(&session, model, config.softplus_beta);
    bake_mean_exposure_into_sh(&session, model, views.len());
    drop(session);
    gpu_cloud.deinit(&gpu);
    path_bufs.destroy(&gpu);
    let mut recorder = recorder;
    recorder.destroy(&gpu);
    gpu.destroy_command_encoder(&mut record_encoder);

    losses
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
    let _vg = build_volumetric_graph(&mut g, n_cells, p, max_steps, 0, 1, 0, 0.0, 0.0, 0.0);

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
    session.set_adam(
        config.learning_rate,
        config.adam_beta1,
        config.adam_beta2,
        config.adam_eps,
    );
    for _ in 0..config.epochs {
        session.step();
        session.wait();
        let read = session.read_output(1);
        losses.push(read.first().copied().unwrap_or(f32::NAN));
    }

    download_model_parameters(&session, model, 0.0);

    losses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::try_init_gpu;

    #[test]
    fn adam_state_roundtrip() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping adam_state_roundtrip: no GPU");
            return;
        };
        // Mirror the known-good `fit_constant_rgb` shape: a parameter
        // [1, dim], plus a dead `x` input multiplied by a zero constant
        // so x flows in without contributing. mse_loss against labels.
        let mut g = mn::Graph::new();
        let n = 8usize;
        let x = g.input("x", &[1, n]);
        let labels = g.input("labels", &[1, n]);
        let p = g.parameter("log_density", &[1, n]);
        let zero = g.constant(vec![0.0_f32; n], &[1, n]);
        let dead = g.mul(x, zero);
        let pred = g.add(p, dead);
        let loss = g.mse_loss(pred, labels);
        g.set_outputs(vec![loss]);

        let (mut sess, _) = mn::build(
            &g,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        sess.set_adam(0.01, 0.9, 0.999, 1e-8);
        sess.set_parameter("log_density", &vec![1.0_f32; n]);
        sess.set_input("x", &vec![0.0_f32; n]);
        sess.set_input("labels", &vec![0.0_f32; n]);
        sess.step();
        sess.wait();

        eprintln!("param_names: {:?}", sess.param_names());
        eprintln!("has_param_grad(log_density): {}", sess.has_param_grad("log_density"));
        eprintln!("adam_step_count: {}", sess.adam_step_count());

        let mut param_after = vec![0.0_f32; n];
        sess.read_param("log_density", &mut param_after);
        eprintln!("param[0..5] after step: {:?}", &param_after[..5]);

        let mut grad_after = vec![0.0_f32; n];
        sess.read_param_grad("log_density", &mut grad_after);
        eprintln!("grad[0..5] after step: {:?}", &grad_after[..5]);

        let mut m_after_step = vec![0.0_f32; n];
        sess.read_adam_m("log_density", &mut m_after_step);
        eprintln!("m[0..{n}] after step: {:?}", m_after_step);

        assert!(
            m_after_step.iter().any(|&v| v.abs() > 1e-9),
            "m should be non-zero after one Adam step"
        );

        let pattern: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        sess.write_adam_m("log_density", &pattern);
        sess.set_adam_step_count(42);

        let mut m_back = vec![0.0_f32; n];
        sess.read_adam_m("log_density", &mut m_back);
        eprintln!("m[0..{n}] after write: {:?}", m_back);
        eprintln!("expected: {:?}", pattern);
        for (i, (&got, &want)) in m_back.iter().zip(pattern.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "m[{i}] roundtrip: got {got}, want {want}"
            );
        }
        assert_eq!(sess.adam_step_count(), 42);

        // Verify the step counter survives across step(): bias correction
        // uses `t`, so if my set_adam_step_count isn't being respected,
        // the next Adam update will use t=1 instead of t=43.
        sess.set_input("labels", &vec![0.6_f32; n]);
        sess.step();
        sess.wait();
        assert_eq!(sess.adam_step_count(), 43, "step counter must increment from 42");
    }

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
    fn parent_inheritance_padding_layout() {
        // Mirror what `restore_adam_state` does for one parameter
        // entry: take the old (m, v) for `n_cells_old` cells with a
        // given per-cell stride, append entries for new cells that
        // inherit from their parents, verify the layout.
        let n_cells_old = 4usize;
        let stride = 3usize; // position-like
        // Per-cell values: cell i gets [10*i + 0, 10*i + 1, 10*i + 2]
        let mut old = Vec::with_capacity(n_cells_old * stride);
        for i in 0..n_cells_old {
            for c in 0..stride {
                old.push((i * 10 + c) as f32);
            }
        }
        // Append 2 new cells; both inherit from parent index 2.
        let parents = vec![2usize, 2usize];
        let n_new = n_cells_old + parents.len();
        let mut padded = vec![0.0_f32; n_new * stride];
        padded[..n_cells_old * stride].copy_from_slice(&old);
        for (k, &parent) in parents.iter().enumerate() {
            let dst = (n_cells_old + k) * stride;
            let src = parent * stride;
            padded[dst..dst + stride].copy_from_slice(&old[src..src + stride]);
        }
        // Verify:
        // - cells 0..3 unchanged
        for i in 0..n_cells_old {
            for c in 0..stride {
                let got = padded[i * stride + c];
                let want = (i * 10 + c) as f32;
                assert_eq!(got, want, "cell {i} c{c}");
            }
        }
        // - new cells 4 and 5 = parent (cell 2) = [20, 21, 22]
        for k in 0..parents.len() {
            for c in 0..stride {
                let got = padded[(n_cells_old + k) * stride + c];
                let want = (2 * 10 + c) as f32;
                assert_eq!(got, want, "new cell {k} c{c}");
            }
        }
    }

    #[test]
    fn per_cell_param_names_includes_positions_with_stride_3() {
        for sh_degree in 0..=3 {
            let names = per_cell_param_names_with_stride(sh_degree);
            // Required entries:
            assert!(names.contains(&("log_density".to_string(), 1)));
            assert!(names.contains(&("positions".to_string(), 3)));
            // Total count: 1 (density) + 1 (positions) + 3 * (1+deg)² SH params.
            let num_components = (1 + sh_degree) * (1 + sh_degree);
            assert_eq!(names.len(), 2 + 3 * num_components);
        }
    }

    #[test]
    fn build_volumetric_graph_constructs_for_sh_degrees_0_to_3() {
        // No GPU dispatch: just exercise `build_volumetric_graph` to
        // catch shape mismatches in the position-opt subgraph (g.add
        // asserts equal shapes). Runs in a few ms on CI.
        for sh_degree in 0..=3 {
            let mut g = mn::Graph::new();
            let n_cells = 32usize;
            let n_pixels = 8usize;
            let max_steps = 4usize;
            let num_views = 5usize;
            let vg = build_volumetric_graph(
                &mut g, n_cells, n_pixels, max_steps, sh_degree, num_views, 0, 0.0, 0.0, 0.0,
            );
            assert_eq!(vg.sh_degree, sh_degree);
            assert_eq!(vg.n_cells, n_cells);
            assert_eq!(vg.n_pixels, n_pixels);
            assert_eq!(vg.max_steps, max_steps);
            assert_eq!(vg.num_views, num_views);
            // SH coefficient tables: 3 channels × (1+deg)² components.
            let num_components = (1 + sh_degree) * (1 + sh_degree);
            assert_eq!(vg.sh_coefficients.len(), 3);
            for chan in &vg.sh_coefficients {
                assert_eq!(chan.len(), num_components);
            }
            // basis_inputs: K-1 entries (component 0 is folded in).
            assert_eq!(vg.basis_inputs.len(), num_components - 1);
        }
    }

    #[test]
    fn build_volumetric_graph_constructs_in_patch_mode() {
        // Patch mode: `n_pixels == patch_size²`, gradient L1 added to
        // the loss. Catches shape mismatches inside `patch_grad_l1`.
        let mut g = mn::Graph::new();
        let patch_size = 4usize;
        let n_pixels = patch_size * patch_size;
        let vg = build_volumetric_graph(&mut g, 16, n_pixels, 4, 0, 2, patch_size, 0.2, 0.0, 0.0);
        assert_eq!(vg.n_pixels, n_pixels);
    }

    #[test]
    fn lr_schedule_constant_is_constant() {
        let mut cfg = AppearanceFitConfig {
            learning_rate: 0.1,
            lr_schedule: LrSchedule::Constant,
            ..AppearanceFitConfig::default()
        };
        cfg.lr_min_ratio = 0.01;
        for t in [0, 1, 100, 9999] {
            assert_eq!(lr_at_step(&cfg, t, 10000), 0.1);
        }
    }

    #[test]
    fn lr_schedule_cosine_endpoints_and_midpoint() {
        let cfg = AppearanceFitConfig {
            learning_rate: 0.1,
            lr_schedule: LrSchedule::Cosine,
            lr_min_ratio: 0.01,
            ..AppearanceFitConfig::default()
        };
        let total = 10_000usize;
        // At t=0: full LR.
        let at_start = lr_at_step(&cfg, 0, total);
        assert!((at_start - 0.1).abs() < 1e-6, "got {at_start}");
        // At t=total: floor LR (lr_min_ratio * base).
        let at_end = lr_at_step(&cfg, total, total);
        assert!((at_end - 0.001).abs() < 1e-6, "got {at_end}");
        // At midpoint: halfway between base and floor (cos(π/2)=0).
        let at_mid = lr_at_step(&cfg, total / 2, total);
        let expected_mid = 0.001 + (0.1 - 0.001) * 0.5;
        assert!(
            (at_mid - expected_mid).abs() < 1e-5,
            "midpoint got {at_mid}, want {expected_mid}",
        );
        // Monotonic decay.
        let qtr = lr_at_step(&cfg, total / 4, total);
        let three_qtr = lr_at_step(&cfg, 3 * total / 4, total);
        assert!(qtr > at_mid, "qtr {qtr} should exceed mid {at_mid}");
        assert!(at_mid > three_qtr, "mid {at_mid} should exceed 3qtr {three_qtr}");
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
        let _vg = build_volumetric_graph(&mut g, n_cells, p, max_steps, 0, 1, 0, 0.0, 0.0, 0.0);

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
    ///
    /// TODO: position-optimisation removed the `dt` input from
    /// `build_volumetric_graph` (dt is now computed from `positions` +
    /// ray geometry). The legacy `fit_appearance_to_pixels` path doesn't
    /// have a camera so it can't supply the ray inputs; restoring this
    /// test would mean either re-adding a dt-input mode or refactoring
    /// the test to use the pixel-batched code path. The same coverage
    /// is provided by `multi_view_training_beats_single_view_on_novel_pose`
    /// which uses the camera-aware path.
    #[test]
    #[ignore = "legacy precomputed-path API: dt is now computed from positions"]
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
    ///
    /// TODO: same as `fit_appearance_reduces_loss` — uses the legacy
    /// precomputed-path API. `multi_view_training_beats_single_view_on_novel_pose`
    /// covers the same generalisation check via the pixel-batched path.
    #[test]
    #[ignore = "legacy precomputed-path API: dt is now computed from positions"]
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
        let paths_a: Vec<_> = rays_for_view(&cam_a, w, h)
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

    /// Approximate COLMAP-style training: 4 cameras around a 4-cell scene,
    /// each contributing one ground-truth render. Train appearance against
    /// all four views, then evaluate on a held-out novel view. The novel
    /// view's L1 against ground truth should beat what a single-view
    /// trainer would get and certainly beat the black baseline.
    #[test]
    fn multi_view_training_beats_single_view_on_novel_pose() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping multi_view_training: no GPU");
            return;
        };

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

        // Camera ring around the cluster.
        fn ring_camera(theta: f32) -> vol::CameraParams {
            let r = 1.5_f32;
            let pos = glam::Vec3::new(r * theta.sin(), 0.10, -(r * theta.cos()));
            // Orientation: look toward +Z origin from this point; we'll yaw
            // around Y by `theta`.
            let q = glam::Quat::from_axis_angle(glam::Vec3::Y, theta);
            vol::CameraParams {
                cam_position: pos.into(),
                depth: 100.0,
                cam_orientation: [q.x, q.y, q.z, q.w],
                fov: [0.6, 0.6],
                pad: [0, 0],
            }
        }

        let train_thetas = [0.0_f32, 0.5, -0.5, 1.0];
        let novel_theta = 0.25_f32;
        let w = 8u32;
        let h = 8u32;
        let render_settings = crate::render::RenderSettings {
            width: w,
            height: h,
            start_point: 0,
            max_steps: 32,
            weight_threshold: 1e-4,
        };

        let views: Vec<ViewSupervision> = train_thetas
            .iter()
            .map(|&t| {
                let cam = ring_camera(t);
                let rgba = crate::render::render_cpu(&gt, &cam, render_settings);
                ViewSupervision {
                    camera: cam,
                    target_rgb: strip_alpha(&rgba),
                    width: w,
                    height: h,
                    start_cell: 0,
                }
            })
            .collect();

        let mut init = tiny_model();
        for p in init.points.iter_mut() {
            p.w = 1.0;
        }
        for v in init.sh_coefficients.iter_mut() {
            *v = 0.0;
        }

        let losses = fit_appearance_multi_view(
            &mut init,
            &views,
            w,
            h,
            32,
            AppearanceFitConfig {
                learning_rate: 0.1,
                epochs: 200,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );
        assert!(losses.first().unwrap().is_finite());
        assert!(losses.last().unwrap().is_finite());

        let novel_cam = ring_camera(novel_theta);
        let gt_novel = strip_alpha(&crate::render::render_cpu(&gt, &novel_cam, render_settings));
        let trained_novel = strip_alpha(&crate::render::render_cpu(
            &init,
            &novel_cam,
            render_settings,
        ));
        let baseline = vec![0.0f32; gt_novel.len()];

        let err_trained = mean_l1(&trained_novel, &gt_novel);
        let err_black = mean_l1(&baseline, &gt_novel);
        eprintln!(
            "multi-view novel-pose: trained L1 {err_trained:.4}, black baseline {err_black:.4}, \
             final loss {:.4}",
            losses.last().unwrap()
        );
        assert!(
            err_trained < err_black * 0.5,
            "trained novel view did not beat baseline: trained={err_trained} baseline={err_black}"
        );
    }
}
