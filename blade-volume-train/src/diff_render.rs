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
//! `vol::trace::record_path`. Adjacency and the discrete cell walk are frozen
//! during one training cycle. Positions may be optimised through the smooth
//! face-intersection calculation, provided the caller periodically rebuilds
//! adjacency and recorded paths between cycles.

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
    /// Weighted-cloud-only radius parameter and differential path inputs.
    pub weighted_path: Option<WeightedPathGraph>,
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
    /// Selected `dt` per (pixel, step) shape `[P, L]`. Unweighted models
    /// compute it differentiably from positions; weighted models use the
    /// recorder's radical-plane/sphere-clipped interval.
    pub dt_from_positions: mn::NodeId,
}

#[derive(Clone, Debug)]
pub struct WeightedPathGraph {
    /// Softplus pre-image of each rendered support radius.
    pub log_radii: mn::NodeId,
    pub previous_cell_indices: mn::NodeId,
    pub dt_grad_previous: mn::NodeId,
    pub dt_grad_current: mn::NodeId,
    pub dt_grad_next: mn::NodeId,
    /// Geometry snapshot against which the recorder evaluated its Jacobians.
    pub reference_positions: mn::NodeId,
    pub reference_radii: mn::NodeId,
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

const RADIUS_SOFTPLUS_BETA: f32 = 100.0;

/// Positive activation for a flat `[count, 1]` tensor. This is the stable
/// identity `(relu(βx) - log(sigmoid(|βx|))) / β`, which avoids needing an
/// explicit exponential graph op.
fn positive_activation(
    g: &mut mn::Graph,
    input: mn::NodeId,
    count: usize,
    beta: f32,
) -> mn::NodeId {
    if beta <= 0.0 {
        return g.relu(input);
    }
    let beta_c = g.constant(vec![beta; count], &[count, 1]);
    let bx = g.mul(input, beta_c);
    let relu_bx = g.relu(bx);
    let abs_bx = g.abs(bx);
    let sig = g.sigmoid(abs_bx);
    let log_sig = g.log(sig);
    let neg_log_sig = g.neg(log_sig);
    let sp = g.add(relu_bx, neg_log_sig);
    let inv_beta = g.constant(vec![1.0 / beta; count], &[count, 1]);
    g.mul(sp, inv_beta)
}

#[allow(clippy::too_many_arguments)]
fn weighted_role_tangent(
    g: &mut mn::Graph,
    indices: mn::NodeId,
    actual_positions: mn::NodeId,
    log_radii: mn::NodeId,
    reference_positions: mn::NodeId,
    reference_radii: mn::NodeId,
    recorded_jacobian: mn::NodeId,
    ones_3_1: mn::NodeId,
    count: usize,
) -> mn::NodeId {
    let reference_position = g.embedding(indices, reference_positions);
    let neg_reference_position = g.neg(reference_position);
    let position_delta = g.add(actual_positions, neg_reference_position);

    let raw_radius = g.embedding(indices, log_radii);
    let actual_radius = positive_activation(g, raw_radius, count, RADIUS_SOFTPLUS_BETA);
    let reference_radius = g.embedding(indices, reference_radii);
    let neg_reference_radius = g.neg(reference_radius);
    let radius_delta = g.add(actual_radius, neg_reference_radius);

    let position_jacobian_flat = g.split_a(recorded_jacobian, count as u32, 3, 1, 1);
    let position_jacobian = g.reshape(position_jacobian_flat, &[count, 3]);
    let radius_jacobian_flat = g.split_b(recorded_jacobian, count as u32, 3, 1, 1);
    let radius_jacobian = g.reshape(radius_jacobian_flat, &[count, 1]);
    let position_product = g.mul(position_delta, position_jacobian);
    let position_tangent = g.matmul(position_product, ones_3_1);
    let radius_tangent = g.mul(radius_delta, radius_jacobian);
    g.add(position_tangent, radius_tangent)
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
    distortion_weight: f32,
    quantile_weight: f32,
    softplus_beta: f32,
    background_rgb: [f32; 3],
    use_recorded_dt: bool,
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
    let recorded_dt = g.input("recorded_dt", &[pl]);
    let mask = g.input("mask", &[pl]);
    let weighted_inputs = use_recorded_dt.then(|| {
        (
            g.input_u32("previous_cell_indices", &[pl]),
            g.input("dt_grad_previous", &[pl, 4]),
            g.input("dt_grad_current", &[pl, 4]),
            g.input("dt_grad_next", &[pl, 4]),
            g.input("reference_positions", &[n_cells, 3]),
            g.input("reference_radii", &[n_cells, 1]),
        )
    });
    // Target is fed as [1, P*3] to match meganeura's "batch × dim" convention
    // for L1 loss; we reshape rather than introduce a batch dimension upstream.
    let target = g.input("labels", &[1, p * 3]);
    let quantile_inputs = if quantile_weight > 0.0 {
        Some((
            g.input("quantile_near", &[p, 1]),
            g.input("quantile_far", &[p, 1]),
            g.input("quantile_scale", &[1, 1]),
        ))
    } else {
        None
    };

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
    let log_radii = use_recorded_dt.then(|| g.parameter("log_radii", &[n_cells, 1]));
    let weighted_path = weighted_inputs.map(
        |(
            previous_cell_indices,
            dt_grad_previous,
            dt_grad_current,
            dt_grad_next,
            reference_positions,
            reference_radii,
        )| WeightedPathGraph {
            log_radii: log_radii.unwrap(),
            previous_cell_indices,
            dt_grad_previous,
            dt_grad_current,
            dt_grad_next,
            reference_positions,
            reference_radii,
        },
    );
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
    let density_flat = positive_activation(g, density_pre, pl, softplus_beta);
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
    let normal_squared = g.mul(normal, normal);

    // Broadcast ray_origin [1,3] → [P*L, 3] via matmul with [P*L, 1] ones.
    let ones_pl1 = g.constant(vec![1.0_f32; pl], &[pl, 1]);
    let ray_origin_pl = g.matmul(ones_pl1, ray_origin); // [P*L, 3]
                                                        // Gather ray_dir [P, 3] → [P*L, 3] via embedding with pixel_idx_per_step.
    let ray_dir_pl = g.embedding(pixel_idx_per_step, ray_dir_per_pixel);

    let neg_ray_origin_pl = g.neg(ray_origin_pl);
    let mo_diff = g.add(midpoint, neg_ray_origin_pl); // [P*L, 3]

    // dot products via element-wise mul then matmul against [3, 1] ones.
    let ones_3_1 = g.constant(vec![1.0_f32; 3], &[3, 1]);
    let normal_length_squared = g.matmul(normal_squared, ones_3_1);
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

    // A terminal segment has `next_cell == cell`, so its face normal is zero
    // and its end is the fixed far plane rather than a differentiable face.
    // Blend to the recorder's exact dt in that case. For real faces the gate
    // is effectively one; the epsilon keeps the expression finite.
    let normal_gate_denominator = g.add(normal_length_squared, eps_pl1);
    let normal_gate_flat = g.div(normal_length_squared, normal_gate_denominator);
    let normal_gate = g.reshape(normal_gate_flat, &[p, l]);
    let neg_normal_gate = g.neg(normal_gate);
    let ones_for_gate = g.constant(vec![1.0_f32; pl], &[p, l]);
    let terminal_gate = g.add(ones_for_gate, neg_normal_gate);
    let recorded_dt_2d = g.reshape(recorded_dt, &[p, l]);
    let selected_dt = if use_recorded_dt {
        // The recorder evaluates the exact weighted, sphere-clipped interval
        // and its active-branch Jacobian at the geometry snapshot used to
        // build the GPU cloud. Reconstruct a first-order tangent around that
        // snapshot so both positions and positive radii affect the forward
        // value between discrete topology rebuilds.
        let weighted = weighted_path.as_ref().unwrap();
        let pos_previous = g.embedding(weighted.previous_cell_indices, positions);
        let previous_tangent = weighted_role_tangent(
            g,
            weighted.previous_cell_indices,
            pos_previous,
            weighted.log_radii,
            weighted.reference_positions,
            weighted.reference_radii,
            weighted.dt_grad_previous,
            ones_3_1,
            pl,
        );
        let current_tangent = weighted_role_tangent(
            g,
            cell_indices,
            pos_cell,
            weighted.log_radii,
            weighted.reference_positions,
            weighted.reference_radii,
            weighted.dt_grad_current,
            ones_3_1,
            pl,
        );
        let next_tangent = weighted_role_tangent(
            g,
            next_cell_indices,
            pos_next,
            weighted.log_radii,
            weighted.reference_positions,
            weighted.reference_radii,
            weighted.dt_grad_next,
            ones_3_1,
            pl,
        );
        let entry_and_current = g.add(previous_tangent, current_tangent);
        let tangent = g.add(entry_and_current, next_tangent);
        let recorded_dt_flat = g.reshape(recorded_dt, &[pl, 1]);
        let linear_dt_flat = g.add(recorded_dt_flat, tangent);
        let linear_dt = g.reshape(linear_dt_flat, &[p, l]);
        let positive_dt = g.relu(linear_dt);
        let neg_positive_dt = g.neg(positive_dt);
        let remaining = g.add(max_dt_pl, neg_positive_dt);
        let within_cap = g.relu(remaining);
        let neg_within_cap = g.neg(within_cap);
        g.add(max_dt_pl, neg_within_cap)
    } else {
        let face_dt = g.mul(dt_clamped, normal_gate);
        let terminal_dt = g.mul(recorded_dt_2d, terminal_gate);
        g.add(face_dt, terminal_dt)
    };
    let dt_2d = g.mul(selected_dt, mask_2d); // zero out invalid steps
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

    // Explicit premultiplied-alpha background compositing. Opacity
    // regularization is independent: changing its weight must not silently
    // change the supervised image convention.
    let neg_op = g.neg(opacity);
    let remaining = g.add(ones_p1, neg_op); // [P,1] = 1 − opacity
    let composite = |g: &mut mn::Graph, pixel, channel: f32| {
        if channel == 0.0 {
            pixel
        } else {
            let background = g.constant(vec![channel; p], &[p, 1]);
            let uncovered = g.mul(remaining, background);
            g.add(pixel, uncovered)
        }
    };
    let pixel_r = composite(g, pixel_r, background_rgb[0]);
    let pixel_g = composite(g, pixel_g, background_rgb[1]);
    let pixel_b = composite(g, pixel_b, background_rgb[2]);

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

    // Smooth ray-thickness penalty. Segment midpoints come from the exact dt
    // stream selected above. `opacity * E[t²] - E[t]²` is the unnormalised
    // weighted depth variance; it is zero for a single concentrated surface
    // and grows when a ray spreads contribution across floaters/layers.
    let loss = if distortion_weight > 0.0 {
        let depth_prefix_matrix = g.constant(strict_lower_triangular_ones(l), &[l, l]);
        let depth_prefix = g.matmul(dt_2d, depth_prefix_matrix);
        let half = g.constant(vec![0.5_f32; p * l], &[p, l]);
        let half_dt = g.mul(dt_2d, half);
        let midpoint = g.add(depth_prefix, half_dt);
        let weighted_midpoint = g.mul(weight, midpoint);
        let first_moment = g.matmul(weighted_midpoint, ones_l1);
        let midpoint_squared = g.mul(midpoint, midpoint);
        let weighted_midpoint_squared = g.mul(weight, midpoint_squared);
        let second_moment = g.matmul(weighted_midpoint_squared, ones_l1);
        let opacity_second = g.mul(opacity, second_moment);
        let first_squared = g.mul(first_moment, first_moment);
        let neg_first_squared = g.neg(first_squared);
        let variance_raw = g.add(opacity_second, neg_first_squared);
        let variance = g.relu(variance_raw);
        let variance_loss = g.mean_all(variance);
        let base_2d = g.reshape(loss, &[1, 1]);
        let variance_2d = g.reshape(variance_loss, &[1, 1]);
        let scale = g.constant(vec![distortion_weight], &[1, 1]);
        let scaled = g.mul(variance_2d, scale);
        let total = g.add(base_2d, scaled);
        g.reshape(total, &[1])
    } else {
        loss
    };

    // Reference RadFoam thickness loss. Two random transmittance quantiles
    // are sampled per ray on the host. For optical-depth target τ = -ln(q),
    // each segment contributes `clamp((τ - prefix) / density, 0, dt)` to the
    // distance travelled before that quantile. Summing those clamped segment
    // distances exactly reproduces the piecewise-constant-density crossing
    // depth without a custom traversal backward. Rays that do not reach the
    // farther quantile are masked, matching the reference's invalid-depth
    // handling. `quantile_scale` carries its half-training warmup schedule.
    let loss = if let Some((quantile_near, quantile_far, quantile_scale)) = quantile_inputs {
        let near_depth = optical_depth_quantile(
            g,
            quantile_near,
            cumsum,
            density,
            dt_2d,
            mask_2d,
            ones_1l,
            ones_l1,
            p,
            l,
        );
        let far_depth = optical_depth_quantile(
            g,
            quantile_far,
            cumsum,
            density,
            dt_2d,
            mask_2d,
            ones_1l,
            ones_l1,
            p,
            l,
        );
        let total_optical_depth = g.matmul(raw_masked, ones_l1);
        let valid = g.greater(total_optical_depth, quantile_far);
        let neg_near = g.neg(near_depth);
        let spread_raw = g.add(far_depth, neg_near);
        let spread = g.abs(spread_raw);
        let valid_spread = g.mul(spread, valid);
        let quantile_loss = g.mean_all(valid_spread);
        let base_2d = g.reshape(loss, &[1, 1]);
        let quantile_2d = g.reshape(quantile_loss, &[1, 1]);
        let scaled = g.mul(quantile_2d, quantile_scale);
        let total = g.add(base_2d, scaled);
        g.reshape(total, &[1])
    } else {
        loss
    };

    // `dt_from_positions` as a second output so callers can compare the graph
    // interval against the recorder during geometry-optimisation checks.
    g.set_outputs(vec![loss, dt_from_positions]);

    VolumetricGraph {
        n_cells,
        n_pixels,
        max_steps,
        sh_degree,
        num_views,
        log_density,
        positions,
        weighted_path,
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

#[allow(clippy::too_many_arguments)]
fn optical_depth_quantile(
    g: &mut mn::Graph,
    optical_depth: mn::NodeId,
    prefix_optical_depth: mn::NodeId,
    density: mn::NodeId,
    dt: mn::NodeId,
    mask: mn::NodeId,
    ones_1l: mn::NodeId,
    ones_l1: mn::NodeId,
    p: usize,
    l: usize,
) -> mn::NodeId {
    let target = g.matmul(optical_depth, ones_1l);
    let neg_prefix = g.neg(prefix_optical_depth);
    let remaining = g.add(target, neg_prefix);
    let density_epsilon = g.constant(vec![1.0e-8_f32; p * l], &[p, l]);
    let safe_density = g.add(density, density_epsilon);
    let distance_raw = g.div(remaining, safe_density);
    let distance_positive = g.relu(distance_raw);
    let neg_distance = g.neg(distance_positive);
    let dt_minus_distance = g.add(dt, neg_distance);
    let unused_distance = g.relu(dt_minus_distance);
    let neg_unused = g.neg(unused_distance);
    let distance_clamped = g.add(dt, neg_unused);
    let distance_masked = g.mul(distance_clamped, mask);
    g.matmul(distance_masked, ones_l1)
}

/// Build the L1 distance between finite-difference gradients of a single
/// channel for the rendered patch and the target patch. Both inputs are
/// `[q*q, 1]` (row-major flat patches); returns a scalar.
///
/// `diff_x[i, j] = X[i, j+1] - X[i, j]` is computed via matmul with a
/// constant `[q, q-1]` band matrix on the right. `diff_y[i, j] = X[i+1, j]
/// - X[i, j]` uses a `[q-1, q]` band matrix on the left. The matmuls are
/// differentiable end-to-end — `SplitA`/`SplitB` cannot be used here
///
/// because their autodiff backward is empty (gradients would silently
/// drop to zero before reaching the rendered pixels).
fn patch_grad_l1(g: &mut mn::Graph, pred: mn::NodeId, target: mn::NodeId, q: usize) -> mn::NodeId {
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
    principal: glam::Vec2,
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
        principal: glam::Vec2::from_array(cam.principal),
    }
}

fn ray_dir_for_pixel(c: &RayConstants, ix: u32, iy: u32, w: u32, h: u32) -> glam::Vec3 {
    let px = (ix as f32 + 0.5) / w as f32;
    let py = (iy as f32 + 0.5) / h as f32;
    let ndc = glam::Vec2::new(px * 2.0 - 1.0, py * 2.0 - 1.0);
    let local_xy = (ndc - c.principal) * c.tan_half;
    let local = glam::Vec3::new(local_xy.x, local_xy.y, 1.0);
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
    sh_chans: &[mn::NodeId],     // K parameter tables [N, 1]
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
    ///
    /// lets early training move fast and late training fine-tune.
    Cosine,
}

const SAMPLING_RNG_SEED: u64 = 0xDEAD_BEEF_F00D_CAFE;
const QUANTILE_RNG_SEED: u64 = 0x51A7_E5D0_9B3C_2468;
const DENSIFY_RNG_SEED: u64 = 0xCAFE_F00D_DEAD_BEEF;
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const TRAINING_STATE_HEADER: &str = "blade-volume-training-state-v1";

/// Deterministic trainer state which is not owned by meganeura.
///
/// Parameter values, Adam moments, and the Adam counter live in the paired
/// safetensors checkpoint. These RNG states preserve view/pixel sampling,
/// quantile regularization, and densification decisions across a process
/// restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingState {
    pub step: usize,
    pub cycle: usize,
    pub sampling_rng: u64,
    pub quantile_rng: u64,
    pub densify_rng: u64,
}

/// Knobs for [`fit_appearance_multi_view`].
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
    /// randomly chosen training view. `None` feeds every pixel exactly once
    /// per step and uses `epochs` steps per view. Both modes use the same GPU
    /// path recorder and differentiable graph.
    pub pixel_batch: Option<usize>,
    /// Number of Adam steps per view in randomly batched mode. Default 200.
    pub steps_per_view: usize,
    /// SH degree for view-dependent colour. 0 = flat colour (default,
    /// matches the original radfoam pipeline). 1–3 enable view
    /// dependence: ~3-5 dB PSNR improvement on Mip-NeRF 360 scenes at
    /// the cost of `(1+sh_degree)²` per-cell parameters per RGB
    /// channel.
    pub sh_degree: usize,
    /// Adaptive densification: split cells with the largest accumulated
    /// position-gradient magnitude times cell radius. `None` disables
    /// densification entirely (geometry stays at its initial cell count).
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
    /// Weight on a smooth per-ray weighted depth-variance loss. `0.0`
    /// disables it. Small values such as `1e-4` discourage contribution
    /// spread across floaters without requiring a non-differentiable sampled
    /// depth-quantile lookup.
    pub distortion_weight: f32,
    /// Weight on RadFoam's random transmittance-quantile separation loss.
    /// Two uniform quantiles are sampled per ray; the weight ramps from zero
    /// to this value over the first half of training. The reference uses
    /// `1e-4`. `0.0` (default) disables it.
    pub quantile_weight: f32,
    /// Softplus β for the density activation. `0.0` (default) uses legacy
    /// ReLU; `> 0` (RadFoam uses 10) uses `(1/β)·softplus(βx)` so cells
    /// that dip to negative log-density keep a gradient and recover
    /// instead of dying — important for stable densification.
    pub softplus_beta: f32,
    /// Display-referred sRGB-code-value background composited behind
    /// premultiplied cloud color. Default black. Use `[1.0; 3]` for datasets
    /// prepared on white.
    pub background_rgb: [f32; 3],
    /// Position learning rate as a fraction of `learning_rate`. `0.0`
    /// (default) freezes geometry. A positive value requires
    /// `geometry_rebuild_every > 0`, because the discrete adjacency and
    /// recorded cell walks must be refreshed as points move.
    pub position_lr_ratio: f32,
    /// PowerFoam support-radius learning rate as a fraction of
    /// `learning_rate`. Radii are optimized through a β=100 softplus and this
    /// must remain zero for unweighted clouds. Like position optimization, a
    /// positive value requires periodic geometry rebuilds.
    pub radius_lr_ratio: f32,
    /// Number of Adam steps between adjacency, GPU-cloud, and path-buffer
    /// rebuilds while positions or radii are trainable. Ignored when both
    /// geometry learning-rate ratios are zero.
    pub geometry_rebuild_every: usize,
    /// Use Qhull for geometry/densification rebuilds. This is the practical
    /// exact backend for large unweighted clouds; the Rust backend remains
    /// available for small dependency-minimal jobs.
    pub rebuild_with_qhull: bool,
    /// When `Some(path)`, write an interchange PLY plus lossless parameter,
    /// Adam, and trainer-state sidecars at every densify boundary and at this
    /// invocation's endpoint. Exposure is baked only into the PLY clone.
    /// `None` disables checkpoints and is invalid for a bounded invocation.
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// Absolute step to resume training from (default 0). When resuming a
    /// model loaded from a checkpoint PLY, this offsets the loop counter
    /// so the cosine LR schedule and densify gates (warmup/until)
    /// continue from where the interrupted run left off instead of
    /// restarting at step 0 with peak LR. The total step count is
    /// unchanged, so a resume runs `total_steps − resume_step` more
    /// steps.
    pub resume_step: usize,
    /// Maximum Adam updates to execute in this process. `None` runs through
    /// the full global budget. A bounded invocation keeps LR and topology
    /// schedules based on the unchanged global step count and writes a
    /// resumable checkpoint at its endpoint.
    pub stop_after_steps: Option<usize>,
    /// Optional meganeura safetensors checkpoint paired with `init_ply`.
    /// Restores exact parameters, Adam moments, and the Adam step counter
    /// after the graph has been rebuilt for the checkpoint's cell count.
    pub resume_state_path: Option<std::path::PathBuf>,
    /// Optional deterministic trainer state paired with `init_ply`.
    /// Restores the sampling, quantile, and densification RNG streams.
    pub resume_training_state: Option<TrainingState>,
}

/// Adaptive densification: every `every` training steps after `warmup`
/// steps, split cells sampled by accumulated `|grad(position)| × cell_radius`.
/// RadFoam inserts a sibling near the farthest face. PowerFoam copies the
/// support radius and perturbs both sites by 5% of that radius, following the
/// reference resampler without introducing its deferred normal semantics. The
/// sibling inherits density and SH coefficients, so colour stays continuous.
///
/// The accumulator runs in parallel with training and is reset every
/// cycle. Each split rebuilds the meganeura session, GPU cloud, and exact
/// Voronoi adjacency, amortised over the `every` steps in between.
#[derive(Clone, Copy, Debug)]
pub struct DensifyConfig {
    /// Steps between densify rounds.
    pub every: usize,
    /// Per-round growth factor: each round adds `fraction × current_cells`
    /// new cells (RadFoam uses 0.15 = +15%/round), selected by weighted
    /// multinomial on `accumulated|grad(position)| × cell_radius`.
    pub fraction: f32,
    /// Unused legacy knob (sibling jitter). RadFoam placement and PowerFoam's
    /// 5%-of-support-radius resampling are method-specific and fixed. Kept for
    /// CLI compatibility.
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
    /// Prune low-contribution small cells each densify round, while
    /// protecting visible cells and their direct neighbours.
    pub prune: bool,
    /// A cell with at least this much maximum per-view ray weight protects
    /// itself and its direct adjacency neighbours from pruning.
    pub prune_contribution: f32,
    /// Cells below this contribution have their density parameter suppressed
    /// before splitting, matching RadFoam's dead-cell handling.
    pub suppress_contribution: f32,
    /// Only cull an unprotected cell when its farthest-neighbour radius
    /// (RadFoam) or explicit support radius (PowerFoam) is below this value;
    /// large empty background cells remain as traversal support.
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
            prune_contribution: 0.01,
            suppress_contribution: 0.001,
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
            distortion_weight: 0.0,
            quantile_weight: 0.0,
            softplus_beta: 0.0,
            background_rgb: [0.0; 3],
            position_lr_ratio: 0.0,
            radius_lr_ratio: 0.0,
            geometry_rebuild_every: 0,
            rebuild_with_qhull: false,
            checkpoint_path: None,
            resume_step: 0,
            stop_after_steps: None,
            resume_state_path: None,
            resume_training_state: None,
        }
    }
}

fn next_lcg_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add(LCG_INCREMENT);
    (*state >> 32) as u32
}

fn next_quantile(state: &mut u64) -> f32 {
    let bits = next_lcg_u32(state);
    (((bits as f64 + 0.5) / (u32::MAX as f64 + 1.0)) as f32)
        .clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON)
}

/// Advance the affine LCG by `delta` draws in O(log delta), preserving the
/// exact wrapping-u64 sequence. This reconstructs sampling state for legacy
/// checkpoints which predate the training-state sidecar.
fn advance_lcg(state: u64, mut delta: u64) -> u64 {
    let mut current_multiplier = LCG_MULTIPLIER;
    let mut current_increment = LCG_INCREMENT;
    let mut accumulated_multiplier = 1_u64;
    let mut accumulated_increment = 0_u64;
    while delta > 0 {
        if delta & 1 != 0 {
            accumulated_multiplier = accumulated_multiplier.wrapping_mul(current_multiplier);
            accumulated_increment = accumulated_increment
                .wrapping_mul(current_multiplier)
                .wrapping_add(current_increment);
        }
        current_increment = current_multiplier
            .wrapping_add(1)
            .wrapping_mul(current_increment);
        current_multiplier = current_multiplier.wrapping_mul(current_multiplier);
        delta >>= 1;
    }
    accumulated_multiplier
        .wrapping_mul(state)
        .wrapping_add(accumulated_increment)
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

fn densify_due(config: DensifyConfig, steps_done: usize) -> bool {
    steps_done >= config.warmup && (steps_done - config.warmup).is_multiple_of(config.every)
}

fn invocation_end_step(
    total_steps: usize,
    resume_step: usize,
    stop_after_steps: Option<usize>,
) -> usize {
    stop_after_steps
        .map(|count| resume_step.saturating_add(count).min(total_steps))
        .unwrap_or(total_steps)
}

fn segment_preserves_densify_accumulator(
    config: DensifyConfig,
    current_points: usize,
    end_step: usize,
    total_steps: usize,
) -> bool {
    end_step >= total_steps
        || current_points >= config.target_points
        || end_step >= config.densify_until
        || densify_due(config, end_step)
}

fn steps_until_densify(config: DensifyConfig, steps_done: usize) -> usize {
    if steps_done < config.warmup {
        config.warmup - steps_done
    } else {
        let elapsed = steps_done - config.warmup;
        let remainder = elapsed % config.every;
        if remainder == 0 {
            config.every
        } else {
            config.every - remainder
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

/// Fit per-cell density and SH coefficients of `model` so it reproduces every
/// view in `views`. Paths are recorded from the current model on the GPU for
/// every Adam step.
///
/// `pixel_batch = None` uses the complete `width × height` image as the batch
/// and preserves the legacy `epochs * views.len()` step count without using
/// the obsolete precomputed-path graph. `Some(K)` uses random mini-batches and
/// `steps_per_view * views.len()` steps.
pub fn fit_appearance_multi_view(
    model: &mut vol::PointCloudModel,
    views: &[ViewSupervision],
    width: u32,
    height: u32,
    max_steps: usize,
    mut config: AppearanceFitConfig,
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

    let p = (width as usize) * (height as usize);
    for v in views {
        assert_eq!(v.width, width, "view width must match graph width");
        assert_eq!(v.height, height, "view height must match graph height");
        assert_eq!(
            v.target_rgb.len(),
            p * 3,
            "target_rgb length must equal width*height*3"
        );
    }
    let pixel_batch = config.pixel_batch.unwrap_or(p);
    assert!(pixel_batch > 0, "pixel_batch must be greater than zero");
    assert!(max_steps > 0, "max_steps must be greater than zero");
    assert!(
        config.learning_rate.is_finite() && config.learning_rate >= 0.0,
        "learning_rate must be finite and non-negative"
    );
    assert!(
        config.opacity_weight.is_finite() && config.opacity_weight >= 0.0,
        "opacity_weight must be finite and non-negative"
    );
    assert!(
        config.distortion_weight.is_finite() && config.distortion_weight >= 0.0,
        "distortion_weight must be finite and non-negative"
    );
    assert!(
        config.quantile_weight.is_finite() && config.quantile_weight >= 0.0,
        "quantile_weight must be finite and non-negative"
    );
    assert!(
        config
            .background_rgb
            .iter()
            .all(|&channel| channel.is_finite() && channel >= 0.0),
        "background_rgb must be finite and non-negative"
    );
    assert!(
        config.position_lr_ratio.is_finite() && config.position_lr_ratio >= 0.0,
        "position_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.radius_lr_ratio.is_finite() && config.radius_lr_ratio >= 0.0,
        "radius_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.radius_lr_ratio == 0.0 || model.radii.is_some(),
        "radius optimisation requires a weighted cloud"
    );
    assert!(
        (config.position_lr_ratio == 0.0 && config.radius_lr_ratio == 0.0)
            || config.geometry_rebuild_every > 0,
        "geometry optimisation requires geometry_rebuild_every > 0"
    );
    if let Some(ref densify) = config.densify {
        assert!(densify.every > 0, "densify.every must be greater than zero");
        assert!(
            densify.fraction.is_finite() && densify.fraction >= 0.0,
            "densify.fraction must be finite and non-negative"
        );
        assert!(
            densify.prune_contribution.is_finite() && densify.prune_contribution >= 0.0,
            "densify.prune_contribution must be finite and non-negative"
        );
        assert!(
            densify.suppress_contribution.is_finite() && densify.suppress_contribution >= 0.0,
            "densify.suppress_contribution must be finite and non-negative"
        );
        assert!(
            densify.prune_radius.is_finite() && densify.prune_radius >= 0.0,
            "densify.prune_radius must be finite and non-negative"
        );
    }
    if config.pixel_batch.is_none() {
        config.steps_per_view = config.epochs;
    }
    fit_appearance_pixel_batched(model, views, max_steps, pixel_batch, config, gpu)
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
    if let Some(ref radii) = model.radii {
        let init_radii: Vec<f32> = radii
            .iter()
            .map(|&radius| inv_radius_activation(radius))
            .collect();
        session.set_parameter("log_radii", &init_radii);
        session.set_input("reference_positions", &init_positions);
        session.set_input("reference_radii", radii);
    }

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
fn save_checkpoint(path: &std::path::Path, model: &vol::PointCloudModel) -> Result<(), String> {
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

fn save_optimizer_checkpoint(
    session: &mut mn::Session,
    model_path: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    let path = model_path.with_extension("safetensors");
    let tmp = model_path.with_extension("safetensors.tmp");
    session
        .save_checkpoint(&tmp)
        .map_err(|err| format!("{err:?}"))?;
    std::fs::rename(&tmp, &path).map_err(|err| format!("{err:?}"))?;
    Ok(path)
}

fn save_checkpoint_step(model_path: &std::path::Path, step: usize) -> Result<(), String> {
    let path = model_path.with_extension("ply.step");
    let tmp = model_path.with_extension("ply.step.tmp");
    std::fs::write(&tmp, step.to_string())
        .map_err(|err| format!("failed to write {}: {err}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .map_err(|err| format!("failed to rename {}: {err}", path.display()))
}

fn training_state_path(model_path: &std::path::Path) -> std::path::PathBuf {
    model_path.with_extension("trainstate")
}

fn encode_training_state(state: TrainingState) -> String {
    format!(
        "{TRAINING_STATE_HEADER}\nstep {}\ncycle {}\nsampling_rng {}\nquantile_rng {}\ndensify_rng {}\n",
        state.step, state.cycle, state.sampling_rng, state.quantile_rng, state.densify_rng,
    )
}

fn decode_training_state(text: &str) -> Result<TrainingState, String> {
    let mut lines = text.lines();
    if lines.next() != Some(TRAINING_STATE_HEADER) {
        return Err("unsupported or missing training-state header".to_string());
    }
    let mut read_value = |expected: &str| -> Result<u64, String> {
        let line = lines
            .next()
            .ok_or_else(|| format!("missing training-state field {expected}"))?;
        let mut words = line.split_whitespace();
        let name = words.next().unwrap_or_default();
        let value = words.next().unwrap_or_default();
        if name != expected || words.next().is_some() {
            return Err(format!("expected training-state field {expected}"));
        }
        value
            .parse::<u64>()
            .map_err(|err| format!("invalid training-state field {expected}: {err}"))
    };
    let step_u64 = read_value("step")?;
    let state = TrainingState {
        step: usize::try_from(step_u64)
            .map_err(|_| format!("training-state step {step_u64} does not fit usize"))?,
        cycle: usize::try_from(read_value("cycle")?)
            .map_err(|_| "training-state cycle does not fit usize".to_string())?,
        sampling_rng: read_value("sampling_rng")?,
        quantile_rng: read_value("quantile_rng")?,
        densify_rng: read_value("densify_rng")?,
    };
    if lines.any(|line| !line.trim().is_empty()) {
        return Err("unexpected trailing training-state data".to_string());
    }
    Ok(state)
}

/// Load the deterministic trainer-state sidecar paired with `model_path`.
pub fn load_training_state(model_path: &std::path::Path) -> Result<TrainingState, String> {
    let path = training_state_path(model_path);
    let text = std::fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    decode_training_state(&text).map_err(|err| format!("{}: {err}", path.display()))
}

fn save_training_state(
    model_path: &std::path::Path,
    state: TrainingState,
) -> Result<std::path::PathBuf, String> {
    let path = training_state_path(model_path);
    let tmp = model_path.with_extension("trainstate.tmp");
    std::fs::write(&tmp, encode_training_state(state))
        .map_err(|err| format!("failed to write {}: {err}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .map_err(|err| format!("failed to rename {}: {err}", path.display()))?;
    Ok(path)
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
    eprintln!("baking mean exposure into SH DC: r={mean_r:.4} g={mean_g:.4} b={mean_b:.4}");
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

fn radius_activation(x: f32) -> f32 {
    density_activation(x, RADIUS_SOFTPLUS_BETA)
}

fn inv_radius_activation(radius: f32) -> f32 {
    inv_density_activation(radius, RADIUS_SOFTPLUS_BETA)
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

    if let Some(ref mut radii) = model.radii {
        let mut out_radii = vec![0.0_f32; n_cells];
        session.read_param("log_radii", &mut out_radii);
        for (radius, &raw) in radii.iter_mut().zip(out_radii.iter()) {
            *radius = radius_activation(raw);
        }
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
/// that farthest neighbour, from the current adjacency CSR. Unweighted
/// RadFoam uses `cell_radius` to weight densification and place the split
/// child. PowerFoam uses its explicit support radius instead.
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

#[derive(Debug)]
struct PathContributionStats {
    per_cell: Vec<f32>,
    rays: usize,
    segments: usize,
    truncated_rays: usize,
    max_steps_used: usize,
}

fn accumulate_path_contributions(
    cells: &[u32],
    next_cells: &[u32],
    dts: &[f32],
    mask: &[f32],
    points: &[glam::Vec4],
    max_steps: usize,
    view_contribution: &mut [f32],
    stats: &mut PathContributionStats,
) {
    assert_eq!(cells.len(), next_cells.len());
    assert_eq!(cells.len(), dts.len());
    assert_eq!(cells.len(), mask.len());
    assert_eq!(cells.len() % max_steps, 0);
    for ray in 0..cells.len() / max_steps {
        let base = ray * max_steps;
        let mut transmittance = 1.0_f32;
        let mut steps_used = 0usize;
        for step in 0..max_steps {
            let slot = base + step;
            if mask[slot] <= 0.0 {
                break;
            }
            let cell = cells[slot] as usize;
            assert!(cell < points.len(), "path recorder returned invalid cell");
            let optical_depth = points[cell].w.max(0.0) * dts[slot].max(0.0);
            let attenuation = (-optical_depth).exp();
            let weight = transmittance * (1.0 - attenuation);
            view_contribution[cell] += weight;
            transmittance *= attenuation;
            steps_used += 1;
        }
        stats.rays += 1;
        stats.segments += steps_used;
        stats.max_steps_used = stats.max_steps_used.max(steps_used);
        if steps_used == max_steps {
            let last = base + max_steps - 1;
            if cells[last] != next_cells[last] {
                stats.truncated_rays += 1;
            }
        }
    }
}

/// Measure the same per-cell contribution signal used by RadFoam pruning:
/// sum volumetric ray weights within each view, then retain the maximum over
/// views. Each view uses one deterministic 2× downsample phase, matching the
/// reference collector without burdening every Adam mini-batch with a
/// readback. The readback is bounded to 4096 rays (16 MiB per device/shared
/// path set at 256 steps) and reused across views.
fn collect_path_contributions(
    context: &blade_graphics::Context,
    recorder: &vol::gpu::PathRecorder,
    gpu_cloud: &vol::gpu::RadFoamGpuCloud,
    model: &vol::PointCloudModel,
    views: &[ViewSupervision],
    max_steps: usize,
    cycle: usize,
) -> PathContributionStats {
    const MAX_RAYS_PER_BATCH: usize = 4096;

    let max_sampled_pixels = views
        .iter()
        .map(|view| view.width.div_ceil(2) as usize * view.height.div_ceil(2) as usize)
        .max()
        .unwrap_or(1);
    let capacity = max_sampled_pixels.clamp(1, MAX_RAYS_PER_BATCH);
    let mut buffers = if model.radii.is_some() {
        vol::gpu::PathRecordBuffers::new(context, capacity as u32, max_steps as u32)
    } else {
        vol::gpu::PathRecordBuffers::new_recorded_only(context, capacity as u32, max_steps as u32)
    };
    let pl_capacity = capacity as u64 * max_steps as u64;
    let readback_size = pl_capacity * std::mem::size_of::<u32>() as u64;
    let cells_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-cells-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Shared,
    });
    let next_cells_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-next-cells-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Shared,
    });
    let dts_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-dts-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Shared,
    });
    let mask_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-mask-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Shared,
    });
    let mut encoder = context.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "collect-path-contributions",
        buffer_count: 1,
    });
    let mut stats = PathContributionStats {
        per_cell: vec![0.0; model.points.len()],
        rays: 0,
        segments: 0,
        truncated_rays: 0,
        max_steps_used: 0,
    };

    for (view_index, view) in views.iter().enumerate() {
        let phase = cycle
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(view_index.wrapping_mul(0x85EB_CA6B));
        let x_offset = if view.width > 1 { phase as u32 & 1 } else { 0 };
        let y_offset = if view.height > 1 {
            (phase as u32 >> 1) & 1
        } else {
            0
        };
        let mut sampled_pixels =
            Vec::with_capacity(view.width.div_ceil(2) as usize * view.height.div_ceil(2) as usize);
        for y in (y_offset..view.height).step_by(2) {
            for x in (x_offset..view.width).step_by(2) {
                sampled_pixels.push(y * view.width + x);
            }
        }
        let mut view_contribution = vec![0.0_f32; model.points.len()];
        let start_point =
            gpu_cloud.containing_point(glam::Vec3::from_array(view.camera.cam_position));

        for pixel_batch in sampled_pixels.chunks(capacity) {
            buffers.write_pixel_indices_prefix(pixel_batch);
            let num_pixels = pixel_batch.len();
            let pl = num_pixels * max_steps;
            let path_bytes = (pl * std::mem::size_of::<u32>()) as u64;
            encoder.start();
            {
                let mut transfer = encoder.transfer("contribution-path-prepare");
                transfer.copy_buffer_to_buffer(
                    buffers.pixel_indices_stage.at(0),
                    buffers.pixel_indices.at(0),
                    std::mem::size_of_val(pixel_batch) as u64,
                );
                transfer.fill_buffer(buffers.cells.at(0), path_bytes, 0);
                transfer.fill_buffer(buffers.next_cells.at(0), path_bytes, 0);
                transfer.fill_buffer(buffers.dts.at(0), path_bytes, 0);
                transfer.fill_buffer(buffers.mask.at(0), path_bytes, 0);
                if buffers.has_jacobians() {
                    transfer.fill_buffer(buffers.previous_cells.at(0), path_bytes, 0);
                    transfer.fill_buffer(buffers.dt_grad_previous.at(0), path_bytes * 4, 0);
                    transfer.fill_buffer(buffers.dt_grad_current.at(0), path_bytes * 4, 0);
                    transfer.fill_buffer(buffers.dt_grad_next.at(0), path_bytes * 4, 0);
                }
            }
            recorder.dispatch(
                &mut encoder,
                gpu_cloud,
                &buffers,
                vol::gpu::RecordPathsArgs {
                    camera: view.camera,
                    start_point,
                    max_steps: max_steps as u32,
                    image_width: view.width,
                    image_height: view.height,
                    max_path_dt: MAX_PATH_DT,
                    depth: view.camera.depth,
                    num_pixels: num_pixels as u32,
                },
            );
            {
                let mut transfer = encoder.transfer("contribution-path-readback");
                transfer.copy_buffer_to_buffer(
                    buffers.cells.at(0),
                    cells_readback.at(0),
                    path_bytes,
                );
                transfer.copy_buffer_to_buffer(
                    buffers.next_cells.at(0),
                    next_cells_readback.at(0),
                    path_bytes,
                );
                transfer.copy_buffer_to_buffer(buffers.dts.at(0), dts_readback.at(0), path_bytes);
                transfer.copy_buffer_to_buffer(buffers.mask.at(0), mask_readback.at(0), path_bytes);
            }
            let sync = context.submit(&mut encoder);
            let completed = context
                .wait_for(&sync, !0)
                .expect("contribution path readback failed");
            assert!(completed, "contribution path readback timed out");
            let cells =
                unsafe { std::slice::from_raw_parts(cells_readback.data() as *const u32, pl) };
            let next_cells =
                unsafe { std::slice::from_raw_parts(next_cells_readback.data() as *const u32, pl) };
            let dts = unsafe { std::slice::from_raw_parts(dts_readback.data() as *const f32, pl) };
            let mask =
                unsafe { std::slice::from_raw_parts(mask_readback.data() as *const f32, pl) };
            accumulate_path_contributions(
                cells,
                next_cells,
                dts,
                mask,
                &model.points,
                max_steps,
                &mut view_contribution,
                &mut stats,
            );
        }
        for (maximum, contribution) in stats.per_cell.iter_mut().zip(view_contribution) {
            *maximum = maximum.max(contribution);
        }
    }

    context.destroy_command_encoder(&mut encoder);
    context.destroy_buffer(cells_readback);
    context.destroy_buffer(next_cells_readback);
    context.destroy_buffer(dts_readback);
    context.destroy_buffer(mask_readback);
    buffers.destroy(context);
    stats
}

/// One RadFoam-style prune+densify round on the post-download `model`
/// (positions in `points.xyz`, density in `points.w`, current adjacency).
///
/// 1. **Prune** small cells with neither direct nor adjacent measured ray
///    contribution (`contribution <= prune_contribution && cell_radius <
///    prune_radius`) — the RadFoam floater remover.
/// 2. **Densify** by appending `fraction × survivors` children, parents
///    drawn by weighted multinomial (without replacement) on
///    `grad_accum × cell_radius`. Unweighted children sit 0.25× toward the
///    parent's farthest neighbour plus a small random kick. Weighted parent
///    and child sites each move by 0.05× the copied support radius, adapting
///    PowerFoam's reference resampling to the current normal-free SH model.
///    Density, appearance, radius, and optimizer ancestry are inherited.
///
/// Returns `(new_to_old, pruned, added)`: `new_to_old[j]` is the OLD cell
/// index whose Adam (m,v) the rebuilt cell `j` should inherit (survivor →
/// itself, child → parent), used to carry optimiser momentum across the
/// session rebuild.
fn prune_and_densify(
    model: &mut vol::PointCloudModel,
    grad_accum: &[f32],
    contribution: &[f32],
    cfg: &DensifyConfig,
    rng_state: &mut u64,
    softplus_beta: f32,
) -> (Vec<usize>, usize, usize) {
    let n_old = model.points.len();
    assert_eq!(grad_accum.len(), n_old);
    assert_eq!(contribution.len(), n_old);
    let sh_block = model.sh_component_count() * 3;
    let (neighbor_radius, farthest) = per_cell_farthest(model);
    let weighted = model.radii.is_some();
    let cell_radius = model.radii.clone().unwrap_or(neighbor_radius);

    let mut next_unit = || {
        *rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*rng_state >> 32) as i32 as f32) / (i32::MAX as f32)
    };

    // --- Prune ---
    // RadFoam retains cells which contribute materially in any view and
    // their one-ring neighbours. Neighbour protection preserves traversal
    // support around visible surfaces instead of punching holes merely
    // because a support cell's own density/weight is low.
    let protected: Vec<bool> = contribution
        .iter()
        .map(|&value| value > cfg.prune_contribution)
        .collect();
    let adjacency = model.adjacency.as_ref();
    let mut survivors: Vec<usize> = (0..n_old)
        .filter(|&i| {
            if !cfg.prune {
                return true;
            }
            let protected_by_neighbor = adjacency.is_some_and(|adj| {
                let start = adj.offsets[i] as usize;
                let end = adj.offsets[i + 1] as usize;
                adj.neighbors[start..end]
                    .iter()
                    .any(|&neighbor| protected[neighbor as usize])
            });
            let unprotected = !protected[i] && !protected_by_neighbor;
            let small = cell_radius[i] < cfg.prune_radius;
            !(unprotected && small)
        })
        .collect();
    if survivors.is_empty() {
        let keep = (0..n_old)
            .max_by(|&a, &b| contribution[a].total_cmp(&contribution[b]))
            .expect("densification requires at least one point");
        survivors.push(keep);
    }
    let n_surv = survivors.len();
    let pruned = n_old - n_surv;

    if cfg.prune {
        let suppressed_density = density_activation(-1.0, softplus_beta);
        for (point, &value) in model.points.iter_mut().zip(contribution) {
            if value < cfg.suppress_contribution {
                point.w = suppressed_density;
            }
        }
    }

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
                let w = (grad_accum[oi] * cell_radius[oi]).max(1e-12);
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
    let mut is_split_parent = vec![false; n_old];
    for &local in &parents_local {
        is_split_parent[survivors[local]] = true;
    }

    let random_unit = |next_unit: &mut dyn FnMut() -> f32| loop {
        let value = glam::Vec3::new(next_unit(), next_unit(), next_unit());
        let length_squared = value.length_squared();
        if length_squared > 1.0e-12 && length_squared <= 1.0 {
            break value / length_squared.sqrt();
        }
    };

    // --- Rebuild model arrays: survivors compacted, then children ---
    let n_new = n_surv + added;
    let mut new_points = Vec::with_capacity(n_new);
    let mut new_sh = Vec::with_capacity(n_new * sh_block);
    let mut new_radii = model.radii.as_ref().map(|_| Vec::with_capacity(n_new));
    let mut new_transforms = model.transforms.as_ref().map(|_| vol::Transforms {
        rotations: Vec::with_capacity(n_new),
        scales: Vec::with_capacity(n_new),
    });
    let mut new_to_old = Vec::with_capacity(n_new);
    for &oi in &survivors {
        let mut point = model.points[oi];
        if weighted && is_split_parent[oi] {
            let offset = random_unit(&mut next_unit) * (0.05 * cell_radius[oi].max(1.0e-5));
            point.x += offset.x;
            point.y += offset.y;
            point.z += offset.z;
        }
        new_points.push(point);
        new_sh.extend_from_slice(&model.sh_coefficients[oi * sh_block..(oi + 1) * sh_block]);
        if let Some(ref mut radii) = new_radii {
            radii.push(model.radii.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut transforms) = new_transforms {
            let old = model.transforms.as_ref().unwrap();
            transforms.rotations.push(old.rotations[oi]);
            transforms.scales.push(old.scales[oi]);
        }
        new_to_old.push(oi);
    }
    for &local in &parents_local {
        let oi = survivors[local];
        let p = model.points[oi];
        let off = if weighted {
            random_unit(&mut next_unit) * (0.05 * cell_radius[oi].max(1.0e-5))
        } else {
            let pf = model.points[farthest[oi]];
            let toward = glam::Vec3::new(pf.x - p.x, pf.y - p.y, pf.z - p.z) * 0.25;
            let kick_scale = (toward.length() * 0.1).max(1e-5);
            let kick = glam::Vec3::new(next_unit(), next_unit(), next_unit()) * kick_scale;
            toward + kick
        };
        new_points.push(glam::Vec4::new(p.x + off.x, p.y + off.y, p.z + off.z, p.w));
        new_sh.extend_from_slice(&model.sh_coefficients[oi * sh_block..(oi + 1) * sh_block]);
        if let Some(ref mut radii) = new_radii {
            radii.push(model.radii.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut transforms) = new_transforms {
            let old = model.transforms.as_ref().unwrap();
            transforms.rotations.push(old.rotations[oi]);
            transforms.scales.push(old.scales[oi]);
        }
        new_to_old.push(oi);
    }
    model.points = new_points;
    model.sh_coefficients = new_sh;
    model.radii = new_radii;
    model.transforms = new_transforms;
    model.adjacency = None;
    (new_to_old, pruned, added)
}

/// Enumerate every per-cell parameter name with its per-cell element
/// stride. Stride is 1 for scalar tables (`log_density`, `log_radii`,
/// `sh_<chan>_<k>`) and 3 for `positions`.
fn per_cell_param_names_with_stride(sh_degree: usize, has_radii: bool) -> Vec<(String, usize)> {
    let num_components = (1 + sh_degree) * (1 + sh_degree);
    let mut names = vec![("log_density".to_string(), 1), ("positions".to_string(), 3)];
    if has_radii {
        names.push(("log_radii".to_string(), 1));
    }
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
    /// View-sized parameters are not represented in `PointCloudModel`, so
    /// their values and moments must be carried explicitly across a graph
    /// rebuild as well.
    exposure_entries: Vec<FixedAdamEntry>,
    /// Adam step counter (bias-correction `t`).
    t: u32,
}

struct AdamEntry {
    name: String,
    stride: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

struct FixedAdamEntry {
    name: &'static str,
    parameter: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

fn save_adam_state(
    session: &mn::Session,
    sh_degree: usize,
    n_cells: usize,
    num_views: usize,
    has_radii: bool,
) -> AdamSnapshot {
    let names = per_cell_param_names_with_stride(sh_degree, has_radii);
    let mut entries = Vec::with_capacity(names.len());
    for (name, stride) in names {
        let size = n_cells * stride;
        let mut m = vec![0.0_f32; size];
        let mut v = vec![0.0_f32; size];
        session.read_adam_m(&name, &mut m);
        session.read_adam_v(&name, &mut v);
        entries.push(AdamEntry { name, stride, m, v });
    }
    let exposure_entries = ["exposure_r", "exposure_g", "exposure_b"]
        .into_iter()
        .map(|name| {
            let mut parameter = vec![0.0_f32; num_views];
            let mut m = vec![0.0_f32; num_views];
            let mut v = vec![0.0_f32; num_views];
            session.read_param(name, &mut parameter);
            session.read_adam_m(name, &mut m);
            session.read_adam_v(name, &mut v);
            FixedAdamEntry {
                name,
                parameter,
                m,
                v,
            }
        })
        .collect();
    AdamSnapshot {
        entries,
        exposure_entries,
        t: session.adam_step_count(),
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
    session: &mut mn::Session,
    snap: &AdamSnapshot,
    new_to_old: &[usize],
    sh_degree: usize,
    has_radii: bool,
) {
    let n_new = new_to_old.len();
    let names = per_cell_param_names_with_stride(sh_degree, has_radii);
    debug_assert_eq!(names.len(), snap.entries.len());
    for (i, name_and_stride) in names.iter().enumerate() {
        let entry = &snap.entries[i];
        debug_assert_eq!(entry.stride, name_and_stride.1);
        let s = name_and_stride.1;
        let mut m = vec![0.0_f32; n_new * s];
        let mut v = vec![0.0_f32; n_new * s];
        for (j, &oi) in new_to_old.iter().enumerate() {
            m[j * s..j * s + s].copy_from_slice(&entry.m[oi * s..oi * s + s]);
            v[j * s..j * s + s].copy_from_slice(&entry.v[oi * s..oi * s + s]);
        }
        session.write_adam_m(&entry.name, &m);
        session.write_adam_v(&entry.name, &v);
    }
    for entry in &snap.exposure_entries {
        session.set_parameter(entry.name, &entry.parameter);
        session.write_adam_m(entry.name, &entry.m);
        session.write_adam_v(entry.name, &entry.v);
    }
}

fn rebuild_training_adjacency(model: &mut vol::PointCloudModel, use_qhull: bool) {
    if model.radii.is_some() || !use_qhull {
        model.compute_adjacency_default();
    } else {
        model.adjacency = Some(vol::compute_adjacency_qhull_default(&model.points));
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
    distortion_weight: f32,
    quantile_weight: f32,
    softplus_beta: f32,
    background_rgb: [f32; 3],
    gpu: &std::sync::Arc<blade_graphics::Context>,
    path_bufs: &vol::gpu::PathRecordBuffers,
    lr: f32,
    position_lr_ratio: f32,
    radius_lr_ratio: f32,
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
        distortion_weight,
        quantile_weight,
        softplus_beta,
        background_rgb,
        model.radii.is_some(),
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

    // Unweighted interior dt is reconstructed from positions, with recorded
    // terminal intervals. Weighted dt is the exact recorder value plus its
    // local position/radius tangent around this model snapshot.
    let pl_bytes = (pixel_batch as u64) * (max_steps as u64) * 4;
    for (slot, buf) in [
        ("cell_indices", path_bufs.cells),
        ("next_cell_indices", path_bufs.next_cells),
        ("recorded_dt", path_bufs.dts),
        ("mask", path_bufs.mask),
    ] {
        let source = gpu
            .get_external_buffer_source(buf)
            .expect("PathRecordBuffers::new_external must produce an exportable buffer");
        session
            .bind_external_buffer(meganeura::ExternalSlot::Input(slot), source, pl_bytes)
            .unwrap_or_else(|err| panic!("bind_external_buffer({slot}) failed: {err:?}"));
    }
    if model.radii.is_some() {
        assert!(
            path_bufs.has_jacobians(),
            "weighted training requires Jacobian path buffers"
        );
        for (slot, buf, size) in [
            ("previous_cell_indices", path_bufs.previous_cells, pl_bytes),
            ("dt_grad_previous", path_bufs.dt_grad_previous, pl_bytes * 4),
            ("dt_grad_current", path_bufs.dt_grad_current, pl_bytes * 4),
            ("dt_grad_next", path_bufs.dt_grad_next, pl_bytes * 4),
        ] {
            let source = gpu
                .get_external_buffer_source(buf)
                .expect("weighted path buffer must be exportable");
            session
                .bind_external_buffer(meganeura::ExternalSlot::Input(slot), source, size)
                .unwrap_or_else(|err| panic!("bind_external_buffer({slot}) failed: {err:?}"));
        }
    }

    session.set_adam(lr, betas.0, betas.1, betas.2);
    if sh_degree >= 1 {
        for chan in ["sh_r_", "sh_g_", "sh_b_"] {
            session.set_lr_multiplier(chan, 0.1);
        }
    }
    session.set_lr_multiplier("positions", position_lr_ratio);
    if model.radii.is_some() {
        session.set_lr_multiplier("log_radii", radius_lr_ratio);
    }

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
    let full_image_batch = config.pixel_batch.is_none();
    assert!(
        config.resume_step <= total_steps,
        "resume_step {} exceeds the configured total step budget {}",
        config.resume_step,
        total_steps,
    );
    if let Some(stop_after_steps) = config.stop_after_steps {
        assert!(
            stop_after_steps > 0,
            "stop_after_steps must be greater than zero"
        );
    }
    // A checkpoint at the completed step budget is already final; do not
    // repeat its last update.
    let mut steps_done = config.resume_step;
    let invocation_end =
        invocation_end_step(total_steps, config.resume_step, config.stop_after_steps);
    if invocation_end < total_steps {
        assert!(
            config.checkpoint_path.is_some(),
            "a bounded training invocation requires checkpoint_path"
        );
        if let Some(densify) = config.densify {
            assert!(
                segment_preserves_densify_accumulator(
                    densify,
                    model.points.len(),
                    invocation_end,
                    total_steps,
                ),
                "bounded invocation must end on a densification boundary while future \
                 densification remains enabled"
            );
        }
        log::info!(
            "bounded training invocation: steps {}..{} of {}",
            steps_done,
            invocation_end,
            total_steps,
        );
    }

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

    // Restore all trainer-owned stochastic state. Legacy checkpoints have no
    // sidecar, but view/pixel and quantile streams have a fixed number of
    // draws per completed step and can therefore be reconstructed exactly.
    // Densification depends on prior prune counts, so only the new sidecar can
    // preserve that stream across a restart.
    let sampling_draws_per_step = if config.patch_size > 0 {
        3_u64 // view + x/y patch origin
    } else if full_image_batch {
        1_u64 // view only
    } else {
        1_u64.saturating_add(pixel_batch as u64) // view + each sampled pixel
    };
    let quantile_draws_per_step = if config.quantile_weight > 0.0 {
        (pixel_batch as u64).saturating_mul(2)
    } else {
        0
    };
    let legacy_sampling_draws = (steps_done as u64).saturating_mul(sampling_draws_per_step);
    let legacy_quantile_draws = (steps_done as u64).saturating_mul(quantile_draws_per_step);
    let (mut sampling_rng, mut quantile_rng, mut rng_split, mut cycle) =
        match config.resume_training_state {
            Some(state) => {
                assert_eq!(
                    state.step, steps_done,
                    "training-state step must match resume_step"
                );
                (
                    state.sampling_rng,
                    state.quantile_rng,
                    state.densify_rng,
                    state.cycle,
                )
            }
            None => {
                if steps_done > 0 && config.densify.is_some() {
                    log::warn!(
                        "resume has no trainer-state sidecar: reconstructing sampling streams, \
                         but future densification decisions may differ"
                    );
                }
                (
                    advance_lcg(SAMPLING_RNG_SEED, legacy_sampling_draws),
                    advance_lcg(QUANTILE_RNG_SEED, legacy_quantile_draws),
                    DENSIFY_RNG_SEED,
                    0,
                )
            }
        };

    let mut target_buf = vec![0.0f32; pixel_batch * 3];
    let mut pixel_indices = vec![0u32; pixel_batch];
    let mut quantile_near = vec![0.0_f32; pixel_batch];
    let mut quantile_far = vec![0.0_f32; pixel_batch];
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

    let mut losses = Vec::with_capacity(invocation_end.saturating_sub(steps_done));
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
    let geometry_trainable = config.position_lr_ratio > 0.0 || config.radius_lr_ratio > 0.0;
    // Resume at the checkpoint's absolute step so the cosine LR and densify
    // schedule pick up where the interrupted run stopped.
    if steps_done > 0 {
        log::info!(
            "resuming at step {}/{} with {} cells",
            steps_done,
            total_steps,
            model.points.len(),
        );
    }
    let mut position_grad_accum = vec![0.0f32; model.points.len()];
    let mut position_grad_scratch = vec![0.0f32; model.points.len() * 3];
    let _ = n_cells;
    let mut path_bufs = vol::gpu::PathRecordBuffers::new_external_with_jacobians(
        &gpu,
        pixel_batch as u32,
        max_steps as u32,
        model.radii.is_some(),
    );
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
        config.distortion_weight,
        config.quantile_weight,
        config.softplus_beta,
        config.background_rgb,
        &gpu,
        &path_bufs,
        config.learning_rate,
        config.position_lr_ratio,
        config.radius_lr_ratio,
        (config.adam_beta1, config.adam_beta2, config.adam_eps),
    );
    if let Some(ref state_path) = config.resume_state_path {
        session
            .load_checkpoint(state_path)
            .unwrap_or_else(|err| panic!("failed to load {}: {err:?}", state_path.display()));
        if let Some(state) = config.resume_training_state {
            assert_eq!(
                session.adam_step_count() as usize,
                state.step,
                "optimizer and trainer-state checkpoints disagree on the completed step"
            );
        }
        log::info!(
            "resume: restored parameters and Adam state from {} at Adam step {}",
            state_path.display(),
            session.adam_step_count()
        );
    }

    // `pixel_idx_per_step` is constant across all training steps —
    // upload once. The new session built for each densify cycle gets
    // its own upload below.
    session.set_input_u32("pixel_idx_per_step", &pixel_idx_per_step);

    while steps_done < invocation_end {
        let densify_budget = match densify {
            Some(d) => steps_until_densify(d, steps_done).min(invocation_end - steps_done),
            None => invocation_end - steps_done,
        };
        let geometry_budget = if geometry_trainable {
            let every = config.geometry_rebuild_every;
            (every - steps_done % every).min(invocation_end - steps_done)
        } else {
            invocation_end - steps_done
        };
        let cycle_budget = densify_budget.min(geometry_budget);

        for cycle_step in 0..cycle_budget {
            let step = steps_done + cycle_step;
            let vi = (next_lcg_u32(&mut sampling_rng) as usize) % views.len();
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
                let x0 = next_lcg_u32(&mut sampling_rng) % (max_x + 1);
                let y0 = next_lcg_u32(&mut sampling_rng) % (max_y + 1);
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
            } else if full_image_batch {
                assert_eq!(pixel_batch, img_size as usize);
                for (k, pidx) in (0..img_size).enumerate() {
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
            } else {
                for k in 0..pixel_batch {
                    let pidx = next_lcg_u32(&mut sampling_rng) % img_size;
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
                if path_bufs.has_jacobians() {
                    tx.fill_buffer(path_bufs.previous_cells.at(0), pl_bytes, 0);
                    tx.fill_buffer(path_bufs.dt_grad_previous.at(0), pl_bytes * 4, 0);
                    tx.fill_buffer(path_bufs.dt_grad_current.at(0), pl_bytes * 4, 0);
                    tx.fill_buffer(path_bufs.dt_grad_next.at(0), pl_bytes * 4, 0);
                }
            }
            recorder.dispatch(
                &mut record_encoder,
                &gpu_cloud,
                &path_bufs,
                vol::gpu::RecordPathsArgs {
                    camera: v.camera,
                    start_point: gpu_cloud
                        .containing_point(glam::Vec3::from_array(v.camera.cam_position)),
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
            if config.quantile_weight > 0.0 {
                for (near, far) in quantile_near.iter_mut().zip(quantile_far.iter_mut()) {
                    let qa = next_quantile(&mut quantile_rng);
                    let qb = next_quantile(&mut quantile_rng);
                    *near = -qa.max(qb).ln();
                    *far = -qa.min(qb).ln();
                }
                let ramp = (2.0 * step as f32 / total_steps as f32).min(1.0);
                session.set_input("quantile_near", &quantile_near);
                session.set_input("quantile_far", &quantile_far);
                session.set_input("quantile_scale", &[config.quantile_weight * ramp]);
            }
            // Apply LR schedule: re-set Adam every step with the
            // current effective LR. `set_adam` is cheap (just updates
            // a session field), and per-parameter LR multipliers (set
            // once at build_train_session) survive across set_adam.
            let lr_now = lr_at_step(&config, step, total_steps);
            session.set_adam(
                lr_now,
                config.adam_beta1,
                config.adam_beta2,
                config.adam_eps,
            );
            session.step();
            session.wait();
            let loss = session.read_output(1).first().copied().unwrap_or(f32::NAN);
            losses.push(loss);

            if densify.is_some() && session.has_param_grad("positions") {
                session.read_param_grad("positions", &mut position_grad_scratch);
                for (i, accumulated) in position_grad_accum.iter_mut().enumerate() {
                    let base = i * 3;
                    let x = position_grad_scratch[base];
                    let y = position_grad_scratch[base + 1];
                    let z = position_grad_scratch[base + 2];
                    *accumulated += (x * x + y * y + z * z).sqrt();
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

        // Prune + densify, gated by the RadFoam schedule: only between
        // `warmup` and `densify_until`, and only while under the point
        // budget. Past that it's a refinement-only phase (no rebuilds).
        let mut topology_rebuilt = false;
        if let Some(d) = densify {
            let gate = densify_due(d, steps_done)
                && steps_done < total_steps
                && steps_done < d.densify_until
                && model.points.len() < d.target_points;
            if gate {
                // Snapshot params + Adam state at the OLD size, before
                // prune+densify remaps `model.points`.
                let n_old = model.points.len();
                download_model_parameters(&session, model, config.softplus_beta);
                let adam_snap = save_adam_state(
                    &session,
                    sh_degree,
                    n_old,
                    views.len(),
                    model.radii.is_some(),
                );

                let contribution = if d.prune {
                    let stats = collect_path_contributions(
                        &gpu, &recorder, &gpu_cloud, model, views, max_steps, cycle,
                    );
                    let truncated_percent = if stats.rays == 0 {
                        0.0
                    } else {
                        100.0 * stats.truncated_rays as f32 / stats.rays as f32
                    };
                    let mean_segments = if stats.rays == 0 {
                        0.0
                    } else {
                        stats.segments as f32 / stats.rays as f32
                    };
                    log::info!(
                        "contribution cycle {}: {} rays, {:.1} mean segments, max {}, \
                         {} truncated ({:.3}%)",
                        cycle,
                        stats.rays,
                        mean_segments,
                        stats.max_steps_used,
                        stats.truncated_rays,
                        truncated_percent,
                    );
                    if stats.truncated_rays > 0 {
                        log::warn!(
                            "{} contribution rays hit max_steps={}; pruning may miss cells \
                             beyond the recorded path",
                            stats.truncated_rays,
                            max_steps,
                        );
                    }
                    stats.per_cell
                } else {
                    vec![f32::INFINITY; n_old]
                };
                let (new_to_old, pruned, added) = prune_and_densify(
                    model,
                    &position_grad_accum,
                    &contribution,
                    &d,
                    &mut rng_split,
                    config.softplus_beta,
                );
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
                rebuild_training_adjacency(model, config.rebuild_with_qhull);
                position_grad_accum = vec![0.0f32; model.points.len()];
                position_grad_scratch = vec![0.0f32; model.points.len() * 3];

                // Topology changed: tear down and rebuild the
                // cell-count-dependent resources, then remap Adam moments
                // from survivors and split parents into the new session.
                drop(session);
                gpu_cloud.deinit(&gpu);
                path_bufs.destroy(&gpu);
                path_bufs = vol::gpu::PathRecordBuffers::new_external_with_jacobians(
                    &gpu,
                    pixel_batch as u32,
                    max_steps as u32,
                    model.radii.is_some(),
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
                    config.distortion_weight,
                    config.quantile_weight,
                    config.softplus_beta,
                    config.background_rgb,
                    &gpu,
                    &path_bufs,
                    config.learning_rate,
                    config.position_lr_ratio,
                    config.radius_lr_ratio,
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
                restore_adam_state_remap(
                    &mut session,
                    &adam_snap,
                    &new_to_old,
                    sh_degree,
                    model.radii.is_some(),
                );
                session.set_adam_step_count(adam_snap.t);
                topology_rebuilt = true;
            }
        }

        // Weighted interval Jacobians and the unweighted discrete cell walk
        // are local to the current geometry snapshot. At the configured
        // cadence, download moved positions/radii, rebuild exact topology, and
        // recreate traversal resources. The final cycle only needs the
        // host-side refresh because no further optimizer step will run.
        let geometry_due = geometry_trainable
            && (steps_done.is_multiple_of(config.geometry_rebuild_every)
                || steps_done == invocation_end);
        if geometry_due && !topology_rebuilt {
            let n_cells = model.points.len();
            download_model_parameters(&session, model, config.softplus_beta);
            let adam_snap = (steps_done < invocation_end).then(|| {
                save_adam_state(
                    &session,
                    sh_degree,
                    n_cells,
                    views.len(),
                    model.radii.is_some(),
                )
            });
            rebuild_training_adjacency(model, config.rebuild_with_qhull);
            log::info!(
                "geometry cycle {}: rebuilt adjacency for {} moved points at step {}",
                cycle,
                n_cells,
                steps_done,
            );

            if let Some(adam_snap) = adam_snap {
                drop(session);
                gpu_cloud.deinit(&gpu);
                path_bufs.destroy(&gpu);
                path_bufs = vol::gpu::PathRecordBuffers::new_external_with_jacobians(
                    &gpu,
                    pixel_batch as u32,
                    max_steps as u32,
                    model.radii.is_some(),
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
                    config.distortion_weight,
                    config.quantile_weight,
                    config.softplus_beta,
                    config.background_rgb,
                    &gpu,
                    &path_bufs,
                    config.learning_rate,
                    config.position_lr_ratio,
                    config.radius_lr_ratio,
                    (config.adam_beta1, config.adam_beta2, config.adam_eps),
                );
                session = rebuilt.0;
                gpu_cloud = rebuilt.1;
                session.set_input_u32("pixel_idx_per_step", &pixel_idx_per_step);
                let identity: Vec<usize> = (0..n_cells).collect();
                restore_adam_state_remap(
                    &mut session,
                    &adam_snap,
                    &identity,
                    sh_degree,
                    model.radii.is_some(),
                );
                session.set_adam_step_count(adam_snap.t);
            }
        }

        // Write checkpoints after topology maintenance so every PLY pairs
        // its point positions with matching adjacency. Exposure is baked
        // into a throwaway clone; the exact sidecar retains the live values.
        let checkpoint_due = steps_done == invocation_end
            || densify.is_some_and(|config| densify_due(config, steps_done));
        if checkpoint_due {
            if let Some(ref ckpt) = config.checkpoint_path {
                download_model_parameters(&session, model, config.softplus_beta);
                let mut snapshot = model.clone();
                bake_mean_exposure_into_sh(&session, &mut snapshot, views.len());
                match save_checkpoint(ckpt, &snapshot)
                    .and_then(|()| save_optimizer_checkpoint(&mut session, ckpt))
                {
                    Ok(optimizer_path) => {
                        if let Err(err) = save_checkpoint_step(ckpt, steps_done) {
                            log::warn!("checkpoint step-sidecar write failed: {err}");
                        }
                        let trainer_state = TrainingState {
                            step: steps_done,
                            cycle,
                            sampling_rng,
                            quantile_rng,
                            densify_rng: rng_split,
                        };
                        match save_training_state(ckpt, trainer_state) {
                            Ok(trainer_path) => log::info!(
                                "checkpoint: wrote {}, {}, and {} ({} cells) at step {}",
                                ckpt.display(),
                                optimizer_path.display(),
                                trainer_path.display(),
                                snapshot.points.len(),
                                steps_done,
                            ),
                            Err(err) => log::warn!(
                                "checkpoint trainer-state save failed after model/optimizer: {err}"
                            ),
                        }
                    }
                    Err(err) => log::warn!("checkpoint save failed: {err:?}"),
                }
            }
        }
    }

    if invocation_end < total_steps {
        log::info!(
            "training segment complete at step {}/{}; resume from the checkpoint to continue",
            invocation_end,
            total_steps,
        );
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
        eprintln!(
            "has_param_grad(log_density): {}",
            sess.has_param_grad("log_density")
        );
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

        let checkpoint = std::env::temp_dir().join(format!(
            "blade-volume-adam-roundtrip-{}.safetensors",
            std::process::id()
        ));
        sess.save_checkpoint(&checkpoint).unwrap();
        sess.set_parameter("log_density", &vec![9.0_f32; n]);
        sess.write_adam_m("log_density", &vec![0.0_f32; n]);
        sess.set_adam_step_count(0);
        sess.load_checkpoint(&checkpoint).unwrap();
        let mut restored_param = vec![0.0_f32; n];
        sess.read_param("log_density", &mut restored_param);
        let mut restored_m = vec![0.0_f32; n];
        sess.read_adam_m("log_density", &mut restored_m);
        assert_eq!(restored_param, param_after);
        assert_eq!(restored_m, pattern);
        assert_eq!(sess.adam_step_count(), 42);
        std::fs::remove_file(checkpoint).unwrap();

        // Verify the step counter survives across step(): bias correction
        // uses `t`, so if my set_adam_step_count isn't being respected,
        // the next Adam update will use t=1 instead of t=43.
        sess.set_input("labels", &vec![0.6_f32; n]);
        sess.step();
        sess.wait();
        assert_eq!(
            sess.adam_step_count(),
            43,
            "step counter must increment from 42"
        );
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
        let parents = [2usize, 2usize];
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
    fn densification_selects_position_gradient_times_radius_signal() {
        let mut model = tiny_model();
        let mut rng = 0x1234_5678_9ABC_DEF0;
        let config = DensifyConfig {
            fraction: 0.25,
            target_points: 5,
            prune: false,
            ..DensifyConfig::default()
        };
        let gradients = [0.0, 0.0, 1.0e20, 0.0];
        let contribution = [1.0; 4];
        let (new_to_old, pruned, added) = prune_and_densify(
            &mut model,
            &gradients,
            &contribution,
            &config,
            &mut rng,
            0.0,
        );
        assert_eq!(pruned, 0);
        assert_eq!(added, 1);
        assert_eq!(new_to_old.last().copied(), Some(2));
        assert_eq!(model.points.len(), 5);
        assert_eq!(model.sh_coefficients.len(), 15);
    }

    #[test]
    fn weighted_densification_copies_radius_and_uses_small_support_scale_offsets() {
        let mut model = tiny_model();
        model.radii = Some(vec![0.4, 0.3, 0.2, 0.1]);
        model.compute_adjacency_default();
        let parent = 2;
        let parent_point = model.points[parent];
        let parent_sh = model.sh_coefficients[parent * 3..parent * 3 + 3].to_vec();
        let mut rng = 0x1234_5678_9ABC_DEF0;
        let config = DensifyConfig {
            fraction: 0.25,
            target_points: 5,
            prune: false,
            ..DensifyConfig::default()
        };
        let gradients = [0.0, 0.0, 1.0e20, 0.0];
        let contribution = [1.0; 4];

        let (new_to_old, pruned, added) = prune_and_densify(
            &mut model,
            &gradients,
            &contribution,
            &config,
            &mut rng,
            0.0,
        );

        assert_eq!(pruned, 0);
        assert_eq!(added, 1);
        assert_eq!(new_to_old, [0, 1, 2, 3, 2]);
        let radii = model.radii.as_ref().unwrap();
        assert_eq!(radii, &[0.4, 0.3, 0.2, 0.1, 0.2]);
        for index in [parent, 4] {
            let point = model.points[index];
            let offset = glam::Vec3::new(
                point.x - parent_point.x,
                point.y - parent_point.y,
                point.z - parent_point.z,
            );
            assert!((offset.length() - 0.01).abs() < 1.0e-6);
            assert_eq!(point.w, parent_point.w);
            assert_eq!(
                &model.sh_coefficients[index * 3..index * 3 + 3],
                parent_sh.as_slice()
            );
        }
        assert_ne!(model.points[parent], model.points[4]);
        assert!(model.adjacency.is_none());
    }

    #[test]
    fn contribution_integration_matches_volumetric_weights_and_flags_truncation() {
        let points = [
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 2.0),
        ];
        let cells = [0, 1, 0, 0];
        let next_cells = [1, 2, 0, 0];
        let dts = [
            std::f32::consts::LN_2,
            std::f32::consts::LN_2,
            2.0 * std::f32::consts::LN_2,
            0.0,
        ];
        let mask = [1.0, 1.0, 1.0, 0.0];
        let mut contribution = [0.0; 2];
        let mut stats = PathContributionStats {
            per_cell: Vec::new(),
            rays: 0,
            segments: 0,
            truncated_rays: 0,
            max_steps_used: 0,
        };
        accumulate_path_contributions(
            &cells,
            &next_cells,
            &dts,
            &mask,
            &points,
            2,
            &mut contribution,
            &mut stats,
        );

        assert!((contribution[0] - 1.25).abs() < 1.0e-6);
        assert!((contribution[1] - 0.375).abs() < 1.0e-6);
        assert_eq!(stats.rays, 2);
        assert_eq!(stats.segments, 3);
        assert_eq!(stats.max_steps_used, 2);
        assert_eq!(stats.truncated_rays, 1);
    }

    #[test]
    fn contribution_pruning_protects_direct_neighbors_and_parallel_data() {
        let points: Vec<glam::Vec4> = (0..4)
            .map(|i| glam::Vec4::new(i as f32 * 0.01, 0.0, 0.0, 4.0))
            .collect();
        let mut model = vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: Some(vol::Transforms {
                rotations: vec![glam::Quat::IDENTITY; points.len()],
                scales: vec![glam::Vec3::ONE; points.len()],
            }),
            adjacency: Some(vol::Adjacency {
                neighbors: vec![1, 0, 2, 1, 3, 2],
                offsets: vec![0, 1, 3, 5, 6],
            }),
            radii: Some(vec![0.02; points.len()]),
            points,
        };
        let config = DensifyConfig {
            fraction: 0.0,
            target_points: 4,
            prune: true,
            prune_contribution: 0.01,
            suppress_contribution: 0.001,
            prune_radius: 0.1,
            ..DensifyConfig::default()
        };
        let mut rng = 7;
        let (new_to_old, pruned, added) = prune_and_densify(
            &mut model,
            &[0.0; 4],
            &[0.02, 0.0, 0.0, 0.0],
            &config,
            &mut rng,
            0.0,
        );

        assert_eq!(new_to_old, [0, 1]);
        assert_eq!(pruned, 2);
        assert_eq!(added, 0);
        assert_eq!(model.points.len(), 2);
        assert_eq!(model.radii.as_ref().unwrap().len(), 2);
        assert_eq!(model.transforms.as_ref().unwrap().rotations.len(), 2);
        assert_eq!(model.transforms.as_ref().unwrap().scales.len(), 2);
        assert_eq!(model.points[0].w, 4.0);
        assert_eq!(model.points[1].w, 0.0);
        assert!(model.adjacency.is_none());
    }

    #[test]
    fn weighted_pruning_uses_explicit_support_radius() {
        let points: Vec<glam::Vec4> = (0..4)
            .map(|i| glam::Vec4::new(i as f32 * 0.01, 0.0, 0.0, 1.0))
            .collect();
        let mut model = vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(vol::Adjacency {
                neighbors: vec![1, 0, 2, 1, 3, 2],
                offsets: vec![0, 1, 3, 5, 6],
            }),
            radii: Some(vec![1.0; points.len()]),
            points,
        };
        let config = DensifyConfig {
            fraction: 0.0,
            target_points: 4,
            prune: true,
            prune_contribution: 0.01,
            prune_radius: 0.1,
            ..DensifyConfig::default()
        };
        let mut rng = 7;

        let (new_to_old, pruned, added) =
            prune_and_densify(&mut model, &[0.0; 4], &[0.0; 4], &config, &mut rng, 0.0);

        assert_eq!(new_to_old, [0, 1, 2, 3]);
        assert_eq!(pruned, 0);
        assert_eq!(added, 0);
    }

    #[test]
    fn per_cell_param_names_includes_positions_with_stride_3() {
        for sh_degree in 0..=3 {
            let names = per_cell_param_names_with_stride(sh_degree, false);
            // Required entries:
            assert!(names.contains(&("log_density".to_string(), 1)));
            assert!(names.contains(&("positions".to_string(), 3)));
            // Total count: 1 (density) + 1 (positions) + 3 * (1+deg)² SH params.
            let num_components = (1 + sh_degree) * (1 + sh_degree);
            assert_eq!(names.len(), 2 + 3 * num_components);
            let weighted_names = per_cell_param_names_with_stride(sh_degree, true);
            assert!(weighted_names.contains(&("log_radii".to_string(), 1)));
            assert_eq!(weighted_names.len(), names.len() + 1);
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
                &mut g, n_cells, n_pixels, max_steps, sh_degree, num_views, 0, 0.0, 0.0, 0.0, 0.0,
                0.0, [0.0; 3], false,
            );
            assert_eq!(vg.sh_degree, sh_degree);
            assert_eq!(vg.n_cells, n_cells);
            assert_eq!(vg.n_pixels, n_pixels);
            assert_eq!(vg.max_steps, max_steps);
            assert_eq!(vg.num_views, num_views);
            assert!(vg.weighted_path.is_none());
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
    fn build_volumetric_graph_constructs_weighted_tangent_path() {
        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph(
            &mut graph, 16, 4, 3, 0, 2, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0; 3], true,
        );
        assert!(vg.weighted_path.is_some());
    }

    #[test]
    fn radius_activation_roundtrips_positive_values() {
        for radius in [1.0e-4_f32, 0.01, 0.1, 1.0, 10.0] {
            let roundtrip = radius_activation(inv_radius_activation(radius));
            let tolerance = 1.0e-5_f32.max(radius * 1.0e-5);
            assert!((roundtrip - radius).abs() <= tolerance);
        }
    }

    #[test]
    fn build_volumetric_graph_constructs_in_patch_mode() {
        // Patch mode: `n_pixels == patch_size²`, gradient L1 added to
        // the loss. Catches shape mismatches inside `patch_grad_l1`.
        let mut g = mn::Graph::new();
        let patch_size = 4usize;
        let n_pixels = patch_size * patch_size;
        let vg = build_volumetric_graph(
            &mut g, 16, n_pixels, 4, 0, 2, patch_size, 0.2, 0.0, 0.0, 0.0, 0.0, [0.0; 3], false,
        );
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
        assert!(
            at_mid > three_qtr,
            "mid {at_mid} should exceed 3qtr {three_qtr}"
        );
    }

    #[test]
    fn lcg_jump_matches_iterated_draws() {
        for count in [0_u64, 1, 2, 3, 17, 1_000, 1_000_001] {
            let mut iterated = SAMPLING_RNG_SEED;
            for _ in 0..count {
                let _ = next_lcg_u32(&mut iterated);
            }
            assert_eq!(advance_lcg(SAMPLING_RNG_SEED, count), iterated);
        }
    }

    #[test]
    fn training_state_text_roundtrips_and_rejects_trailing_data() {
        let state = TrainingState {
            step: 9_000,
            cycle: 94,
            sampling_rng: 11,
            quantile_rng: 22,
            densify_rng: 33,
        };
        let encoded = encode_training_state(state);
        assert_eq!(decode_training_state(&encoded).unwrap(), state);
        assert!(decode_training_state(&(encoded + "unexpected 1\n")).is_err());
    }

    #[test]
    fn training_state_sidecar_roundtrips_atomically() {
        let model_path = std::env::temp_dir().join(format!(
            "blade-volume-training-state-{}.ply",
            std::process::id(),
        ));
        let state = TrainingState {
            step: 321,
            cycle: 7,
            sampling_rng: u64::MAX,
            quantile_rng: 123,
            densify_rng: 456,
        };
        let path = save_training_state(&model_path, state).unwrap();
        assert_eq!(load_training_state(&model_path).unwrap(), state);
        assert!(!model_path.with_extension("trainstate.tmp").exists());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn densify_schedule_is_independent_of_geometry_boundaries() {
        let config = DensifyConfig {
            warmup: 2000,
            every: 500,
            ..DensifyConfig::default()
        };
        assert_eq!(steps_until_densify(config, 0), 2000);
        assert_eq!(steps_until_densify(config, 1900), 100);
        assert!(densify_due(config, 2000));
        for geometry_boundary in [2100, 2200, 2300, 2400] {
            assert!(!densify_due(config, geometry_boundary));
            assert_eq!(
                steps_until_densify(config, geometry_boundary),
                2500 - geometry_boundary,
            );
        }
        assert!(densify_due(config, 2500));
        assert_eq!(steps_until_densify(config, 2500), 500);
    }

    #[test]
    fn bounded_invocation_keeps_global_step_budget() {
        assert_eq!(invocation_end_step(16_000, 0, None), 16_000);
        assert_eq!(invocation_end_step(16_000, 9_000, Some(1_000)), 10_000);
        assert_eq!(invocation_end_step(16_000, 15_500, Some(1_000)), 16_000);
        assert_eq!(invocation_end_step(16_000, 16_000, Some(1_000)), 16_000);
    }

    #[test]
    fn bounded_invocation_requires_complete_densify_window() {
        let config = DensifyConfig {
            every: 500,
            warmup: 2_000,
            target_points: 200_000,
            densify_until: 11_000,
            ..DensifyConfig::default()
        };
        assert!(segment_preserves_densify_accumulator(
            config, 100_000, 10_000, 16_000,
        ));
        assert!(!segment_preserves_densify_accumulator(
            config, 100_000, 10_100, 16_000,
        ));
        assert!(segment_preserves_densify_accumulator(
            config, 200_000, 10_100, 16_000,
        ));
        assert!(segment_preserves_densify_accumulator(
            config, 100_000, 11_000, 16_000,
        ));
        assert!(segment_preserves_densify_accumulator(
            config, 100_000, 16_000, 16_000,
        ));
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

    /// Build the graph, feed its differentiable path inputs, and run one Adam
    /// step via the lower-level Session API. This catches shader-generation,
    /// dtype, and shape errors that graph-construction tests cannot see.
    #[test]
    fn volumetric_graph_runs_one_step() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping volumetric_graph_runs_one_step: no GPU");
            return;
        };

        let model = tiny_model();
        let n_cells = model.points.len();

        let max_steps = 1usize;
        let p = 1usize;

        let mut g = mn::Graph::new();
        let _vg = build_volumetric_graph(
            &mut g, n_cells, p, max_steps, 0, 1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0; 3], false,
        );

        let (mut session, _report) = mn::build(
            &g,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );

        session.set_parameter("log_density", &vec![1.0; n_cells]);
        let positions: Vec<f32> = model
            .points
            .iter()
            .flat_map(|point| [point.x, point.y, point.z])
            .collect();
        session.set_parameter("positions", &positions);
        session.set_parameter("sh_r", &vec![0.0; n_cells]);
        session.set_parameter("sh_g", &vec![0.0; n_cells]);
        session.set_parameter("sh_b", &vec![0.0; n_cells]);
        session.set_parameter("exposure_r", &[1.0]);
        session.set_parameter("exposure_g", &[1.0]);
        session.set_parameter("exposure_b", &[1.0]);

        session.set_input_u32("cell_indices", &[0]);
        session.set_input_u32("next_cell_indices", &[3]);
        session.set_input("recorded_dt", &[1.25]);
        session.set_input("mask", &[1.0]);
        session.set_input("ray_origin", &[0.1, 0.1, -1.0]);
        session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
        session.set_input_u32("pixel_idx_per_step", &[0]);
        session.set_input_u32("view_idx", &[0]);
        let target = [0.4_f32, 0.6, 0.8];
        session.set_input("labels", &target);

        session.set_adam(0.1, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();

        let mut position_grad = vec![0.0_f32; n_cells * 3];
        session.read_param_grad("positions", &mut position_grad);
        assert!(position_grad.iter().all(|v| v.is_finite()));
        assert!(
            position_grad.iter().any(|v| v.abs() > 1e-7),
            "interior face integration must produce a position gradient"
        );
    }

    #[test]
    fn position_gradient_matches_central_finite_difference_for_fixed_path() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping position gradient check: no GPU");
            return;
        };
        let model = tiny_model();
        let n_cells = model.points.len();
        let mut graph = mn::Graph::new();
        build_volumetric_graph(
            &mut graph, n_cells, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0; 3], false,
        );
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        upload_model_parameters(&mut session, &model, 0.0);
        for channel in ["exposure_r", "exposure_g", "exposure_b"] {
            session.set_parameter(channel, &[1.0]);
        }

        let set_inputs = |session: &mut mn::Session| {
            // Cell 0 -> cell 3 crosses the horizontal bisector at z=0.25.
            // The ray is well away from a parallel face, clamp, topology, or
            // L1 kink, so a small perturbation stays in one smooth branch.
            session.set_input_u32("cell_indices", &[0]);
            session.set_input_u32("next_cell_indices", &[3]);
            session.set_input("recorded_dt", &[1.25]);
            session.set_input("mask", &[1.0]);
            session.set_input("ray_origin", &[0.1, 0.1, -1.0]);
            session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
            session.set_input_u32("pixel_idx_per_step", &[0]);
            session.set_input_u32("view_idx", &[0]);
            session.set_input("labels", &[0.1, 0.2, 0.3]);
        };
        session.set_adam(0.0, 0.9, 0.999, 1e-8);

        let mut baseline = vec![0.0_f32; n_cells * 3];
        session.read_param("positions", &mut baseline);
        set_inputs(&mut session);
        session.step();
        session.wait();
        let mut analytical = vec![0.0_f32; baseline.len()];
        session.read_param_grad("positions", &mut analytical);

        const EPSILON: f32 = 5.0e-3;
        let mut numerical = vec![0.0_f32; baseline.len()];
        let mut perturbed = baseline.clone();
        for index in 0..baseline.len() {
            perturbed[index] = baseline[index] + EPSILON;
            session.set_parameter("positions", &perturbed);
            set_inputs(&mut session);
            session.step();
            session.wait();
            let plus = session.read_loss();

            perturbed[index] = baseline[index] - EPSILON;
            session.set_parameter("positions", &perturbed);
            set_inputs(&mut session);
            session.step();
            session.wait();
            let minus = session.read_loss();

            numerical[index] = (plus - minus) / (2.0 * EPSILON);
            perturbed[index] = baseline[index];
        }
        session.set_parameter("positions", &baseline);

        assert!(analytical.iter().all(|value| value.is_finite()));
        assert!(numerical.iter().all(|value| value.is_finite()));
        assert!(
            numerical.iter().any(|value| value.abs() > 1.0e-3),
            "fixture must exercise a material position derivative"
        );
        for index in 0..baseline.len() {
            let absolute = (analytical[index] - numerical[index]).abs();
            let scale = analytical[index]
                .abs()
                .max(numerical[index].abs())
                .max(1.0e-4);
            let relative = absolute / scale;
            assert!(
                absolute < 5.0e-3 || relative < 2.0e-2,
                "position[{index}] gradient mismatch: analytical={} numerical={} absolute={absolute} relative={relative}",
                analytical[index],
                numerical[index],
            );
        }
    }

    #[test]
    fn weighted_graph_uses_recorded_intervals_and_jacobians() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping weighted graph Jacobian test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![1.0; model.points.len()]);
        model.compute_adjacency_default();
        let n_cells = model.points.len();
        let mut g = mn::Graph::new();
        let vg = build_volumetric_graph(
            &mut g, n_cells, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0; 3], true,
        );
        let dt_output = g.reshape(vg.dt_from_positions, &[1]);
        g.set_outputs(vec![dt_output]);
        let (mut session, _) = mn::build(
            &g,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        upload_model_parameters(&mut session, &model, 0.0);
        for channel in ["exposure_r", "exposure_g", "exposure_b"] {
            session.set_parameter(channel, &[1.0]);
        }
        session.set_input_u32("cell_indices", &[0]);
        session.set_input_u32("previous_cell_indices", &[0]);
        session.set_input_u32("next_cell_indices", &[3]);
        session.set_input("recorded_dt", &[7.5]);
        session.set_input("dt_grad_previous", &[0.0; 4]);
        session.set_input("dt_grad_current", &[0.25, -0.5, 0.75, 1.25]);
        session.set_input("dt_grad_next", &[0.0; 4]);
        session.set_input("mask", &[1.0]);
        session.set_input("ray_origin", &[0.1, 0.1, -1.0]);
        session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
        session.set_input_u32("pixel_idx_per_step", &[0]);
        session.set_input_u32("view_idx", &[0]);
        session.set_input("labels", &[0.4, 0.6, 0.8]);
        session.set_adam(0.0, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();

        let outputs = session.read_output(1);
        assert!((outputs[0] - 7.5).abs() < 1e-5, "outputs={outputs:?}");
        let mut position_grad = vec![0.0_f32; n_cells * 3];
        session.read_param_grad("positions", &mut position_grad);
        assert!((position_grad[0] - 0.25).abs() < 1.0e-5);
        assert!((position_grad[1] + 0.5).abs() < 1.0e-5);
        assert!((position_grad[2] - 0.75).abs() < 1.0e-5);
        assert!(position_grad[3..].iter().all(|&value| value == 0.0));
        let mut radius_grad = vec![0.0_f32; n_cells];
        session.read_param_grad("log_radii", &mut radius_grad);
        assert!((radius_grad[0] - 1.25).abs() < 1.0e-4);
        assert!(radius_grad[1..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn graph_composites_the_configured_background() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping graph_composites_the_configured_background: no GPU");
            return;
        };
        let evaluate = |background_rgb| {
            let mut model = tiny_model();
            for point in model.points.iter_mut() {
                point.w = 0.0;
            }
            let n_cells = model.points.len();
            let mut graph = mn::Graph::new();
            build_volumetric_graph(
                &mut graph,
                n_cells,
                1,
                1,
                0,
                1,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                background_rgb,
                false,
            );
            let (mut session, _) = mn::build(
                &graph,
                mn::SessionConfig {
                    mode: mn::Mode::Training,
                    gpu: Some(gpu.clone()),
                    ..Default::default()
                },
            );
            upload_model_parameters(&mut session, &model, 0.0);
            for channel in ["exposure_r", "exposure_g", "exposure_b"] {
                session.set_parameter(channel, &[1.0]);
            }
            session.set_input_u32("cell_indices", &[0]);
            session.set_input_u32("next_cell_indices", &[0]);
            session.set_input("recorded_dt", &[1.0]);
            session.set_input("mask", &[1.0]);
            session.set_input("ray_origin", &[0.0, 0.0, -1.0]);
            session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
            session.set_input_u32("pixel_idx_per_step", &[0]);
            session.set_input_u32("view_idx", &[0]);
            session.set_input("labels", &[1.0, 1.0, 1.0]);
            session.set_adam(0.0, 0.9, 0.999, 1e-8);
            session.step();
            session.wait();
            session.read_output(1)[0]
        };

        let black_loss = evaluate([0.0; 3]);
        let white_loss = evaluate([1.0; 3]);
        assert!(black_loss > 1.0, "black loss={black_loss}");
        assert!(white_loss < 1e-6, "white loss={white_loss}");
    }

    #[test]
    fn distortion_loss_penalizes_depth_spread() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping distortion_loss_penalizes_depth_spread: no GPU");
            return;
        };
        let evaluate = |distortion_weight| {
            let mut model = tiny_model();
            for point in model.points.iter_mut() {
                point.w = 1.0;
            }
            let mut graph = mn::Graph::new();
            build_volumetric_graph(
                &mut graph,
                model.points.len(),
                1,
                2,
                0,
                1,
                0,
                0.0,
                0.0,
                distortion_weight,
                0.0,
                0.0,
                [0.0; 3],
                false,
            );
            let (mut session, _) = mn::build(
                &graph,
                mn::SessionConfig {
                    mode: mn::Mode::Training,
                    gpu: Some(gpu.clone()),
                    ..Default::default()
                },
            );
            upload_model_parameters(&mut session, &model, 0.0);
            for channel in ["exposure_r", "exposure_g", "exposure_b"] {
                session.set_parameter(channel, &[1.0]);
            }
            session.set_input_u32("cell_indices", &[0, 1]);
            session.set_input_u32("next_cell_indices", &[0, 1]);
            session.set_input("recorded_dt", &[1.0, 4.0]);
            session.set_input("mask", &[1.0, 1.0]);
            session.set_input("ray_origin", &[0.0, 0.0, -1.0]);
            session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
            session.set_input_u32("pixel_idx_per_step", &[0, 0]);
            session.set_input_u32("view_idx", &[0]);
            session.set_input("labels", &[0.0, 0.0, 0.0]);
            session.set_adam(0.0, 0.9, 0.999, 1e-8);
            session.step();
            session.wait();
            session.read_output(1)[0]
        };

        let base = evaluate(0.0);
        let regularized = evaluate(1.0);
        assert!(base.is_finite() && regularized.is_finite());
        assert!(
            regularized > base + 1e-3,
            "base={base} regularized={regularized}"
        );
    }

    #[test]
    fn quantile_loss_matches_piecewise_constant_crossing_depths() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping quantile_loss_matches_piecewise_constant_crossing_depths: no GPU");
            return;
        };
        let evaluate = |quantiles: Option<[f32; 2]>| {
            let mut model = tiny_model();
            for point in model.points.iter_mut() {
                point.w = 1.0;
            }
            let mut graph = mn::Graph::new();
            build_volumetric_graph(
                &mut graph,
                model.points.len(),
                1,
                2,
                0,
                1,
                0,
                0.0,
                0.0,
                0.0,
                if quantiles.is_some() { 1.0 } else { 0.0 },
                0.0,
                [0.0; 3],
                false,
            );
            let (mut session, _) = mn::build(
                &graph,
                mn::SessionConfig {
                    mode: mn::Mode::Training,
                    gpu: Some(gpu.clone()),
                    ..Default::default()
                },
            );
            upload_model_parameters(&mut session, &model, 0.0);
            for channel in ["exposure_r", "exposure_g", "exposure_b"] {
                session.set_parameter(channel, &[1.0]);
            }
            session.set_input_u32("cell_indices", &[0, 1]);
            session.set_input_u32("next_cell_indices", &[0, 1]);
            session.set_input("recorded_dt", &[1.0, 4.0]);
            session.set_input("mask", &[1.0, 1.0]);
            session.set_input("ray_origin", &[0.0, 0.0, -1.0]);
            session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
            session.set_input_u32("pixel_idx_per_step", &[0, 0]);
            session.set_input_u32("view_idx", &[0]);
            session.set_input("labels", &[0.0, 0.0, 0.0]);
            if let Some([near, far]) = quantiles {
                session.set_input("quantile_near", &[near]);
                session.set_input("quantile_far", &[far]);
                session.set_input("quantile_scale", &[1.0]);
            }
            session.set_adam(0.0, 0.9, 0.999, 1e-8);
            session.step();
            session.wait();
            session.read_output(1)[0]
        };

        let base = evaluate(None);
        // Density is one: τ=0.5 crosses at depth 0.5, while τ=2 crosses
        // one unit into the second segment at depth 2.
        let regularized = evaluate(Some([0.5, 2.0]));
        // The full path has optical depth 5, so τ=6 is invalid and must not
        // contribute to the mean, matching the reference's depth > 0 mask.
        let invalid = evaluate(Some([0.5, 6.0]));
        assert!(base.is_finite() && regularized.is_finite());
        assert!(
            (regularized - base - 1.5).abs() < 1.0e-4,
            "base={base} regularized={regularized}"
        );
        assert!(
            (invalid - base).abs() < 1.0e-5,
            "base={base} invalid={invalid}"
        );
    }

    /// End-to-end: a small target image is the ground-truth render of a
    /// known model. Re-initialise the same geometry with bad appearance
    /// values, fit, and check that the loss decreases by an order of
    /// magnitude over 200 steps. This is the smoke test that proves
    /// gradients actually flow into per-cell density/SH and reduce error.
    ///
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
            principal: [0.0, 0.0],
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

        let view = ViewSupervision {
            camera: cam,
            target_rgb: target_rgb.to_vec(),
            width: 1,
            height: 1,
            start_cell: 0,
        };
        let losses = fit_appearance_multi_view(
            &mut init,
            &[view],
            1,
            1,
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

    #[test]
    fn position_training_rebuilds_valid_topology() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping position_training_rebuilds_valid_topology: no GPU");
            return;
        };
        let mut model = tiny_model();
        let positions_before: Vec<[f32; 3]> =
            model.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: [0.0, 0.0, 0.0, 1.0],
                fov: [0.5, 0.5],
                principal: [0.0, 0.0],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            width: 1,
            height: 1,
            start_cell: 0,
        };
        fit_appearance_multi_view(
            &mut model,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                learning_rate: 0.01,
                epochs: 4,
                position_lr_ratio: 1.0,
                geometry_rebuild_every: 2,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        model.validate().unwrap();
        assert!(model
            .points
            .iter()
            .zip(positions_before)
            .any(|(after, before)| {
                (after.x - before[0]).abs() > 1e-7
                    || (after.y - before[1]).abs() > 1e-7
                    || (after.z - before[2]).abs() > 1e-7
            }));
    }

    #[test]
    fn densification_rebuilds_and_continues_training() {
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping densification_rebuilds_and_continues_training: no GPU");
            return;
        };
        let mut model = tiny_model();
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: [0.0, 0.0, 0.0, 1.0],
                fov: [0.5, 0.5],
                principal: [0.0, 0.0],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            width: 1,
            height: 1,
            start_cell: 0,
        };
        let losses = fit_appearance_multi_view(
            &mut model,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                learning_rate: 0.01,
                epochs: 3,
                densify: Some(DensifyConfig {
                    every: 1,
                    fraction: 0.25,
                    warmup: 1,
                    target_points: 5,
                    prune: true,
                    ..DensifyConfig::default()
                }),
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        assert_eq!(losses.len(), 3);
        assert_eq!(model.points.len(), 5);
        model.validate().unwrap();
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
            principal: [0.0, 0.0],
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
            principal: [0.0, 0.0],
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

        let view = ViewSupervision {
            camera: cam_a,
            target_rgb: target_a_rgb,
            width: w,
            height: h,
            start_cell: 0,
        };
        fit_appearance_multi_view(
            &mut init,
            &[view],
            w,
            h,
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
                principal: [0.0, 0.0],
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
