//! Differentiable per-pixel volumetric integration in meganeura.
//!
//! Given a stack of per-pixel paths recorded by
//! [`vol::trace::record_path`], this module builds a meganeura `Graph` whose
//! parameters are per-cell density and RGB spherical-harmonic coefficients.
//! The forward pass:
//!
//! 1. Packs each RGB channel's SH DC and higher-order parameters into a
//!    row-major table, then gathers and reduces each channel without an
//!    intermediate RGB copy.
//! 2. Applies a positive density activation and computes `raw = density * dt`.
//! 3. Computes the exclusive per-pixel cumulative sum that gives transmittance.
//! 4. Expresses `exp(-x)` via the identity
//!    `exp(-x) = recip(sigmoid(x)) - 1` (valid for x >= 0), since meganeura
//!    has `sigmoid` and `recip` but no raw `exp`.
//! 5. Per-pixel-per-step `weight = T * alpha`, then per-channel
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
    /// Weighted-densification-only zero-valued parameter. Its gradient is the
    /// per-site `T * alpha * L1(cell_color, target)` score; its frozen Adam
    /// first moment retains that statistic on the GPU between resamples.
    pub point_error_probe: Option<mn::NodeId>,
    /// Weighted-cloud-only radius parameter and differential path inputs.
    pub weighted_path: Option<WeightedPathGraph>,
    /// Per-channel DC and packed higher-order SH parameter tables.
    pub sh_coefficients: Vec<ShChannelGraph>,
    /// Optional `[N, 12]` channel-major spatial surface residual table.
    pub surface_color_coefficients: Option<mn::NodeId>,
    /// Optional eight-site spatial height and RGB detail tables.
    pub surface_detail: Option<SurfaceDetailGraph>,
    /// Optional eight-site directional residual parameter tables.
    pub spherical_voronoi: Option<SphericalVoronoiGraph>,
    /// Per-view, per-channel RGB gain: one `[num_views, 1]` table per
    /// channel. Multiplies each rendered pixel before the L1 loss.
    /// Initialised to 1.0; frozen at 1.0 when the `exposure_*` LR
    /// multipliers are 0 (default OFF for an apples-to-apples
    /// baseline). With LR > 0 Adam absorbs per-image brightness
    /// variation into these tables instead of the SH chain. Three
    /// separate tables retain the established checkpoint layout.
    pub exposure_r: mn::NodeId,
    pub exposure_g: mn::NodeId,
    pub exposure_b: mn::NodeId,

    pub cell_indices: mn::NodeId,
    pub next_cell_indices: mn::NodeId,
    pub mask: mn::NodeId,
    pub ray_origin: mn::NodeId,
    pub ray_dir_per_pixel: mn::NodeId,
    pub pixel_idx_per_step: mn::NodeId,
    /// `[P]` u32 indices into the exposure tables, one per sampled ray.
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
pub struct SphericalVoronoiGraph {
    /// `[N, 24]` site-major raw axes. Vector magnitude is temperature.
    pub axes: mn::NodeId,
    /// `[N, 24]` channel-major RGB site values.
    pub colors: mn::NodeId,
}

#[derive(Clone, Debug)]
pub struct SurfaceDetailGraph {
    /// `[N, 24]` site-major radius-normalized object-space offsets.
    pub offsets: mn::NodeId,
    /// `[N, 8]` radius-normalized signed heights.
    pub heights: mn::NodeId,
    /// `[N, 24]` channel-major RGB residuals.
    pub colors: mn::NodeId,
    /// `[P * L, 2]` recorded query-near distance and plane-branch mask.
    pub surface_queries: mn::NodeId,
}

#[derive(Clone, Debug)]
pub struct ShChannelGraph {
    /// `[N, 1]` DC parameter. The historical bare `sh_r/g/b` names are kept.
    pub dc: mn::NodeId,
    /// `[N, K-1]` packed higher-order terms, absent for degree zero.
    pub rest: Option<mn::NodeId>,
}

#[derive(Clone, Debug)]
pub struct WeightedPathGraph {
    /// Softplus pre-image of each rendered support radius.
    pub log_radii: mn::NodeId,
    /// Reference tangent for the differential streams retained by this graph.
    /// Geometry-frozen oriented training contains only the surface-plane term.
    pub dt_reference_tangent: Option<mn::NodeId>,
    pub previous_cell_indices: Option<mn::NodeId>,
    pub dt_grad_previous: Option<mn::NodeId>,
    pub dt_grad_current: Option<mn::NodeId>,
    pub dt_grad_next: Option<mn::NodeId>,
    /// Raw per-site oriented surface normals, normalized in the graph.
    pub surface_normals: Option<mn::NodeId>,
    /// Signed displacement of each oriented surface plane.
    pub surface_offsets: Option<mn::NodeId>,
    /// Recorder derivative with respect to the selected site's unit normal
    /// in xyz and signed surface offset in w.
    pub dt_grad_surface_normal: Option<mn::NodeId>,
    /// Per-step scale for PowerFoam's view-facing normal regularizer.
    pub surface_normal_loss_scale: Option<mn::NodeId>,
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

fn sh_rest_parameter_name(channel: &str) -> String {
    format!("{channel}_rest")
}

fn declare_sh_parameters(
    g: &mut mn::Graph,
    n_cells: usize,
    num_components: usize,
) -> Vec<ShChannelGraph> {
    ["sh_r", "sh_g", "sh_b"]
        .into_iter()
        .map(|channel| ShChannelGraph {
            dc: g.parameter(channel, &[n_cells, 1]),
            rest: (num_components > 1).then(|| {
                g.parameter(
                    &sh_rest_parameter_name(channel),
                    &[n_cells, num_components - 1],
                )
            }),
        })
        .collect()
}

fn set_sh_lr_multipliers(
    session: &mut mn::Session,
    sh_degree: usize,
    dc_multiplier: f32,
    rest_multiplier: f32,
) {
    let num_components = (1 + sh_degree) * (1 + sh_degree);
    for channel in ["sh_r", "sh_g", "sh_b"] {
        session.set_lr_multiplier(channel, dc_multiplier);
        if num_components > 1 {
            session.set_lr_multiplier(&sh_rest_parameter_name(channel), rest_multiplier);
        }
    }
}

const RADIUS_SOFTPLUS_BETA: f32 = 100.0;
const POINT_ERROR_PROBE: &str = "point_error_probe";

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
fn weighted_role_linear_term(
    g: &mut mn::Graph,
    indices: mn::NodeId,
    geometry: mn::NodeId,
    neg_ray_origin_geometry: mn::NodeId,
    recorded_jacobian: mn::NodeId,
) -> mn::NodeId {
    let role_geometry = g.embedding(indices, geometry);
    let relative_geometry = g.add(role_geometry, neg_ray_origin_geometry);
    let product = g.mul(relative_geometry, recorded_jacobian);
    g.sum_inner(product)
}

/// Repeat a `[rows, 1]` column into `[rows, 3]` without launching a tiled
/// matrix multiplication for a scalar broadcast.
fn repeat_xyz(g: &mut mn::Graph, column: mn::NodeId, rows: usize) -> mn::NodeId {
    let column_flat = g.reshape(column, &[rows]);
    let xy_flat = g.concat(column_flat, column_flat, rows as u32, 1, 1, 1);
    let xyz_flat = g.concat(xy_flat, column_flat, rows as u32, 2, 1, 1);
    g.reshape(xyz_flat, &[rows, 3])
}

#[allow(clippy::too_many_arguments)]
fn surface_color_basis_graph(
    g: &mut mn::Graph,
    cell_indices: mn::NodeId,
    positions: mn::NodeId,
    step_normals: mn::NodeId,
    step_offsets: Option<mn::NodeId>,
    actual_radii: mn::NodeId,
    ray_origin_pl: mn::NodeId,
    ray_dir_pl: mn::NodeId,
    pl: usize,
) -> mn::NodeId {
    let centers = g.embedding(cell_indices, positions);
    let neg_ray_origin = g.neg(ray_origin_pl);
    let center_relative = g.add(centers, neg_ray_origin);
    let numerator_terms = g.mul(center_relative, step_normals);
    let mut numerator = g.sum_inner(numerator_terms);
    if let Some(offsets) = step_offsets {
        numerator = g.add(numerator, offsets);
    }
    let denominator_terms = g.mul(ray_dir_pl, step_normals);
    let denominator = g.sum_inner(denominator_terms);
    let denominator_squared = g.mul(denominator, denominator);
    let epsilon_squared = g.constant(vec![1.0e-12_f32; pl], &[pl, 1]);
    let regularized_denominator = g.add(denominator_squared, epsilon_squared);
    let regularized_numerator = g.mul(numerator, denominator);
    let t = g.div(regularized_numerator, regularized_denominator);

    let t_xyz = repeat_xyz(g, t, pl);
    let ray_offset = g.mul(ray_dir_pl, t_xyz);
    let hit = g.add(ray_origin_pl, ray_offset);
    let plane_center = match step_offsets {
        Some(offsets) => {
            let offset_xyz = repeat_xyz(g, offsets, pl);
            let normal_offset = g.mul(step_normals, offset_xyz);
            g.add(centers, normal_offset)
        }
        None => centers,
    };
    let neg_plane_center = g.neg(plane_center);
    let relative = g.add(hit, neg_plane_center);
    let normal_distance_terms = g.mul(relative, step_normals);
    let normal_distance = g.sum_inner(normal_distance_terms);
    let normal_distance_xyz = repeat_xyz(g, normal_distance, pl);
    let normal_component = g.mul(step_normals, normal_distance_xyz);
    let neg_normal_component = g.neg(normal_component);
    let tangent = g.add(relative, neg_normal_component);

    let radii = g.embedding(cell_indices, actual_radii);
    let negative_radius_floor = g.constant(vec![-1.0e-6_f32; pl], &[pl, 1]);
    let radius_above_floor = g.add(radii, negative_radius_floor);
    let radius_above_floor = g.relu(radius_above_floor);
    let radius_floor = g.constant(vec![1.0e-6_f32; pl], &[pl, 1]);
    let radii = g.add(radius_above_floor, radius_floor);
    let radius_xyz = repeat_xyz(g, radii, pl);
    let q = g.div(tangent, radius_xyz);

    // clamp(q, -1, 1) = relu(q + 1) - relu(q - 1) - 1.
    let ones_pl3 = g.constant(vec![1.0_f32; pl * 3], &[pl, 3]);
    let neg_ones_pl3 = g.neg(ones_pl3);
    let q_plus_one = g.add(q, ones_pl3);
    let lower_clamped = g.relu(q_plus_one);
    let q_minus_one = g.add(q, neg_ones_pl3);
    let upper_excess = g.relu(q_minus_one);
    let neg_upper_excess = g.neg(upper_excess);
    let lower_minus_upper = g.add(lower_clamped, neg_upper_excess);
    let q = g.add(lower_minus_upper, neg_ones_pl3);
    let q_squared = g.mul(q, q);
    let q_squared = g.sum_inner(q_squared);
    let ones_pl1 = g.constant(vec![1.0_f32; pl], &[pl, 1]);
    let neg_q_squared = g.neg(q_squared);
    let remaining = g.add(ones_pl1, neg_q_squared);
    let remaining = g.relu(remaining);
    let neg_remaining = g.neg(remaining);
    let radial = g.add(ones_pl1, neg_remaining);

    let q_flat = g.reshape(q, &[pl * 3]);
    let radial_flat = g.reshape(radial, &[pl]);
    let basis_flat = g.concat(q_flat, radial_flat, pl as u32, 3, 1, 1);
    let basis = g.reshape(basis_flat, &[pl, vol::SURFACE_COLOR_COMPONENTS]);
    g.stop_gradient(basis)
}

fn repeat_surface_detail_vec3(g: &mut mn::Graph, values: mn::NodeId, rows: usize) -> mn::NodeId {
    debug_assert_eq!(vol::SURFACE_DETAIL_SITES, 8);
    let flat = g.reshape(values, &[rows * 3]);
    let twice = g.concat(flat, flat, rows as u32, 3, 3, 1);
    let four = g.concat(twice, twice, rows as u32, 6, 6, 1);
    let eight = g.concat(four, four, rows as u32, 12, 12, 1);
    g.reshape(eight, &[rows * vol::SURFACE_DETAIL_SITES, 3])
}

fn negative_exponential(g: &mut mn::Graph, input: mn::NodeId, count: usize) -> mn::NodeId {
    let sigmoid = g.sigmoid(input);
    let reciprocal = g.recip(sigmoid);
    let negative_ones = g.constant(vec![-1.0_f32; count], &[count, 1]);
    g.add(reciprocal, negative_ones)
}

#[allow(clippy::too_many_arguments)]
fn surface_detail_query(
    g: &mut mn::Graph,
    centers: mn::NodeId,
    normals: mn::NodeId,
    offsets: mn::NodeId,
    radii: mn::NodeId,
    ray_origin: mn::NodeId,
    ray_direction: mn::NodeId,
    query_near: mn::NodeId,
    plane_branch: mn::NodeId,
    rows: usize,
) -> mn::NodeId {
    let neg_origin = g.neg(ray_origin);
    let center_relative = g.add(centers, neg_origin);
    let numerator_terms = g.mul(center_relative, normals);
    let numerator = g.sum_inner(numerator_terms);
    let numerator = g.add(numerator, offsets);
    let denominator_terms = g.mul(ray_direction, normals);
    let denominator = g.sum_inner(denominator_terms);
    let denominator_squared = g.mul(denominator, denominator);
    let epsilon_squared = g.constant(vec![1.0e-12_f32; rows], &[rows, 1]);
    let regularized_denominator = g.add(denominator_squared, epsilon_squared);
    let regularized_numerator = g.mul(numerator, denominator);
    let plane_t = g.div(regularized_numerator, regularized_denominator);
    let neg_query_near = g.neg(query_near);
    let plane_after_near = g.add(plane_t, neg_query_near);
    let plane_advance = g.relu(plane_after_near);
    let selected_advance = g.mul(plane_advance, plane_branch);
    let query_t = g.add(query_near, selected_advance);

    let query_t_xyz = repeat_xyz(g, query_t, rows);
    let ray_offset = g.mul(ray_direction, query_t_xyz);
    let hit = g.add(ray_origin, ray_offset);
    let neg_centers = g.neg(centers);
    let relative = g.add(hit, neg_centers);
    let radius_xyz = repeat_xyz(g, radii, rows);
    g.div(relative, radius_xyz)
}

fn surface_detail_weights(
    g: &mut mn::Graph,
    query: mn::NodeId,
    tangent_sites: mn::NodeId,
    rows: usize,
) -> mn::NodeId {
    let query_sites = repeat_surface_detail_vec3(g, query, rows);
    let neg_sites = g.neg(tangent_sites);
    let delta = g.add(query_sites, neg_sites);
    let delta_squared = g.mul(delta, delta);
    let distance_squared = g.sum_inner(delta_squared);
    let count = rows * vol::SURFACE_DETAIL_SITES;
    let temperature = g.constant(vec![10.0_f32; count], &[count, 1]);
    let exponent = g.mul(distance_squared, temperature);
    let weights = negative_exponential(g, exponent, count);
    g.reshape(weights, &[rows, vol::SURFACE_DETAIL_SITES])
}

fn normalize_surface_detail_weights(
    g: &mut mn::Graph,
    weights: mn::NodeId,
    rows: usize,
) -> mn::NodeId {
    let sum = g.sum_inner(weights);
    // A shader-style 1e-20 floor is forward-safe but its squared denominator
    // underflows during division backward on GPUs. Padded rows then produce
    // 0/0 before their loss mask is applied. Below 1e-12 every site weight is
    // already numerically irrelevant, so keep both passes finite here.
    let negative_floor = g.constant(vec![-1.0e-12_f32; rows], &[rows, 1]);
    let above_floor = g.add(sum, negative_floor);
    let above_floor = g.relu(above_floor);
    let floor = g.constant(vec![1.0e-12_f32; rows], &[rows, 1]);
    let denominator = g.add(above_floor, floor);
    let denominator_sites = {
        let flat = g.reshape(denominator, &[rows]);
        let twice = g.concat(flat, flat, rows as u32, 1, 1, 1);
        let four = g.concat(twice, twice, rows as u32, 2, 2, 1);
        let eight = g.concat(four, four, rows as u32, 4, 4, 1);
        g.reshape(eight, &[rows, vol::SURFACE_DETAIL_SITES])
    };
    g.div(weights, denominator_sites)
}

struct SurfaceDetailEvaluation {
    effective_offsets: mn::NodeId,
    color_weights: mn::NodeId,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_surface_detail_graph(
    g: &mut mn::Graph,
    cell_indices: mn::NodeId,
    parameters: &SurfaceDetailGraph,
    centers: mn::NodeId,
    normals: mn::NodeId,
    base_offsets: mn::NodeId,
    radii: mn::NodeId,
    ray_origin: mn::NodeId,
    ray_direction: mn::NodeId,
    rows: usize,
) -> SurfaceDetailEvaluation {
    let queries = g.materialize(parameters.surface_queries);
    let query_near = g.split_a(queries, rows as u32, 1, 1, 1);
    let query_near = g.reshape(query_near, &[rows, 1]);
    let plane_branch = g.split_b(queries, rows as u32, 1, 1, 1);
    let plane_branch = g.reshape(plane_branch, &[rows, 1]);

    let raw_sites = g.embedding(cell_indices, parameters.offsets);
    let raw_sites = g.reshape(raw_sites, &[rows * vol::SURFACE_DETAIL_SITES, 3]);
    let site_normals = repeat_surface_detail_vec3(g, normals, rows);
    let normal_components = g.mul(raw_sites, site_normals);
    let normal_components = g.sum_inner(normal_components);
    let normal_components = repeat_xyz(g, normal_components, rows * vol::SURFACE_DETAIL_SITES);
    let projected_normals = g.mul(site_normals, normal_components);
    let neg_projected_normals = g.neg(projected_normals);
    let tangent_sites = g.add(raw_sites, neg_projected_normals);

    let base_query = surface_detail_query(
        g,
        centers,
        normals,
        base_offsets,
        radii,
        ray_origin,
        ray_direction,
        query_near,
        plane_branch,
        rows,
    );
    let height_weights = surface_detail_weights(g, base_query, tangent_sites, rows);
    let normalized_height_weights = normalize_surface_detail_weights(g, height_weights, rows);
    let heights = g.embedding(cell_indices, parameters.heights);
    let weighted_heights = g.mul(normalized_height_weights, heights);
    let normalized_height = g.sum_inner(weighted_heights);
    let height = g.mul(radii, normalized_height);
    let effective_offsets = g.add(base_offsets, height);

    let displaced_query = surface_detail_query(
        g,
        centers,
        normals,
        effective_offsets,
        radii,
        ray_origin,
        ray_direction,
        query_near,
        plane_branch,
        rows,
    );
    let color_weights = surface_detail_weights(g, displaced_query, tangent_sites, rows);
    let color_weights = normalize_surface_detail_weights(g, color_weights, rows);
    SurfaceDetailEvaluation {
        effective_offsets,
        color_weights,
    }
}

/// Add a sampled PowerFoam interpenetration penalty to a scalar loss.
///
/// `edge_direction` is the unit vector from endpoint B to endpoint A at the
/// most recent topology rebuild. Its dot product with the live endpoint
/// delta is the first-order distance used by the reference loss
/// `max(r_a + r_b - distance, 0)^2`. The host supplies per-sample scales that
/// include both the scheduled loss weight and each sampled edge stratum's
/// size. Zero scales safely pad graphs with fewer edges than samples.
fn add_interpenetration_loss(
    g: &mut mn::Graph,
    base_loss: mn::NodeId,
    positions: mn::NodeId,
    log_radii: mn::NodeId,
    sample_count: usize,
) -> mn::NodeId {
    let edge_a = g.input_u32("interpenetration_edge_a", &[sample_count]);
    let edge_b = g.input_u32("interpenetration_edge_b", &[sample_count]);
    let edge_direction = g.input("interpenetration_edge_direction", &[sample_count, 3]);
    let edge_scale = g.input("interpenetration_edge_scale", &[sample_count, 1]);

    let position_a = g.embedding(edge_a, positions);
    let position_b = g.embedding(edge_b, positions);
    let neg_position_b = g.neg(position_b);
    let edge_delta = g.add(position_a, neg_position_b);
    let projected_delta = g.mul(edge_delta, edge_direction);
    let distance = g.sum_inner(projected_delta);

    let raw_radius_a = g.embedding(edge_a, log_radii);
    let raw_radius_b = g.embedding(edge_b, log_radii);
    let radius_a = positive_activation(g, raw_radius_a, sample_count, RADIUS_SOFTPLUS_BETA);
    let radius_b = positive_activation(g, raw_radius_b, sample_count, RADIUS_SOFTPLUS_BETA);
    let radius_sum = g.add(radius_a, radius_b);
    let neg_distance = g.neg(distance);
    let overlap_raw = g.add(radius_sum, neg_distance);
    let overlap = g.relu(overlap_raw);
    let overlap_squared = g.mul(overlap, overlap);
    let scaled_overlap = g.mul(overlap_squared, edge_scale);
    let interpenetration_loss = g.sum_all(scaled_overlap);
    g.add(base_loss, interpenetration_loss)
}

/// Supervised RGB loss used by the volumetric training graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorLoss {
    /// Mean absolute error per channel. This preserves the original
    /// blade-volume training behavior.
    L1,
    /// PyTorch-compatible Smooth-L1 with beta 1, averaged across pixels and
    /// RGB channels. This is the loss used by the official RadFoam v1 trainer.
    SmoothL1,
}

fn smooth_l1_loss(
    g: &mut mn::Graph,
    prediction: mn::NodeId,
    target: mn::NodeId,
    n: usize,
) -> mn::NodeId {
    let neg_target = g.neg(target);
    let difference = g.add(prediction, neg_target);
    let absolute = g.abs(difference);
    let ones = g.constant(vec![1.0_f32; n], &[n, 1]);
    let neg_ones = g.neg(ones);
    let offset = g.add(absolute, neg_ones);
    let above_one = g.relu(offset);
    let neg_above_one = g.neg(above_one);
    let clamped = g.add(absolute, neg_above_one);
    let half = g.constant(vec![0.5_f32; n], &[n, 1]);
    let clamped_squared = g.mul(clamped, clamped);
    let quadratic = g.mul(clamped_squared, half);
    let elementwise = g.add(quadratic, above_one);
    g.mean_all(elementwise)
}

/// Build the volumetric forward + supervised loss subgraph and return handles.
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
    use_surface_normals: bool,
    use_surface_offsets: bool,
    use_surface_color: bool,
    collect_point_error: bool,
    use_spherical_voronoi: bool,
    color_loss_kind: ColorLoss,
) -> VolumetricGraph {
    build_volumetric_graph_with_options(
        g,
        n_cells,
        n_pixels,
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
        use_recorded_dt,
        use_surface_normals,
        VolumetricGraphOptions {
            use_surface_normal_loss: use_surface_normals,
            train_positions: true,
            train_radii: true,
            use_surface_detail: false,
        },
        use_surface_offsets,
        use_surface_color,
        collect_point_error,
        use_spherical_voronoi,
        color_loss_kind,
    )
}

#[derive(Clone, Copy)]
struct VolumetricGraphOptions {
    use_surface_normal_loss: bool,
    train_positions: bool,
    train_radii: bool,
    use_surface_detail: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_volumetric_graph_with_options(
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
    use_surface_normals: bool,
    options: VolumetricGraphOptions,
    use_surface_offsets: bool,
    use_surface_color: bool,
    collect_point_error: bool,
    use_spherical_voronoi: bool,
    color_loss_kind: ColorLoss,
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
    assert!(
        !collect_point_error || use_recorded_dt,
        "point-error collection is only defined for weighted splats",
    );
    assert!(
        !use_surface_normals || use_recorded_dt,
        "oriented surfaces require weighted recorded paths",
    );
    assert!(
        !options.use_surface_normal_loss || use_surface_normals,
        "surface-normal loss requires oriented surfaces",
    );
    assert!(
        !use_surface_offsets || use_surface_normals,
        "surface offsets require oriented surfaces",
    );
    assert!(
        !use_surface_color || use_surface_normals,
        "surface color requires oriented surfaces",
    );
    assert!(
        !options.use_surface_detail || use_surface_normals,
        "surface detail requires oriented surfaces",
    );
    assert!(
        !use_spherical_voronoi || use_surface_normals,
        "Spherical Voronoi appearance requires oriented surfaces",
    );
    let p = n_pixels;
    let l = max_steps;
    let pl = p * l;
    let num_components = (1 + sh_degree) * (1 + sh_degree);

    let cell_indices = g.input_u32("cell_indices", &[pl]);
    let next_cell_indices = g.input_u32("next_cell_indices", &[pl]);
    let recorded_dt = g.input("recorded_dt", &[pl]);
    let mask = g.input("mask", &[pl]);
    let use_geometry_jacobians =
        use_recorded_dt && (options.train_positions || options.train_radii);
    let use_surface_jacobians = use_recorded_dt && use_surface_normals;
    let use_any_jacobians = use_geometry_jacobians || use_surface_jacobians;
    let dt_reference_tangent = use_any_jacobians.then(|| g.input("dt_reference_tangent", &[pl]));
    let previous_cell_indices =
        use_geometry_jacobians.then(|| g.input_u32("previous_cell_indices", &[pl]));
    let dt_grad_previous = use_geometry_jacobians.then(|| g.input("dt_grad_previous", &[pl, 4]));
    let dt_grad_current = use_geometry_jacobians.then(|| g.input("dt_grad_current", &[pl, 4]));
    let dt_grad_next = use_geometry_jacobians.then(|| g.input("dt_grad_next", &[pl, 4]));
    let dt_grad_surface_normal =
        use_surface_jacobians.then(|| g.input("dt_grad_surface_normal", &[pl, 4]));
    let surface_queries = options
        .use_surface_detail
        .then(|| g.input("surface_queries", &[pl, 2]));
    let surface_normal_loss_scale = options
        .use_surface_normal_loss
        .then(|| g.input("surface_normal_loss_scale", &[1, 1]));
    // Target is fed as [1, P*3] to match meganeura's "batch × dim" convention
    // for L1 loss; we reshape rather than introduce a batch dimension upstream.
    let target = g.input("labels", &[1, p * 3]);
    let target_alpha = (opacity_weight > 0.0).then(|| g.input("target_alpha", &[p, 1]));
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
    // differentiably. Both are per pixel because one Adam batch may contain
    // rays from multiple camera views.
    let ray_origin = g.input("ray_origin", &[p, 3]);
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
    let surface_normals =
        use_surface_normals.then(|| g.parameter("surface_normals", &[n_cells, 3]));
    let surface_offsets =
        use_surface_offsets.then(|| g.parameter("surface_offsets", &[n_cells, 1]));
    let surface_color_coefficients = use_surface_color.then(|| {
        g.parameter(
            "surface_color_coefficients",
            &[n_cells, vol::SURFACE_COLOR_COMPONENTS * 3],
        )
    });
    let surface_detail = options.use_surface_detail.then(|| SurfaceDetailGraph {
        offsets: g.parameter(
            "surface_detail_offsets",
            &[n_cells, vol::SURFACE_DETAIL_SITES * 3],
        ),
        heights: g.parameter(
            "surface_detail_heights",
            &[n_cells, vol::SURFACE_DETAIL_SITES],
        ),
        colors: g.parameter(
            "surface_detail_colors",
            &[n_cells, vol::SURFACE_DETAIL_SITES * 3],
        ),
        surface_queries: surface_queries.unwrap(),
    });
    let spherical_voronoi = use_spherical_voronoi.then(|| SphericalVoronoiGraph {
        axes: g.parameter(
            "spherical_voronoi_axes",
            &[n_cells, vol::SPHERICAL_VORONOI_SITES * 3],
        ),
        colors: g.parameter(
            "spherical_voronoi_colors",
            &[n_cells, vol::SPHERICAL_VORONOI_SITES * 3],
        ),
    });
    let normalized_surface_normals = surface_normals.map(|normals| {
        let unit_scale = 1.0_f32 / 3.0_f32.sqrt();
        let weight = g.constant(vec![unit_scale; 3], &[3]);
        g.rms_norm(normals, weight, 1.0e-12)
    });
    let weighted_path = use_recorded_dt.then(|| WeightedPathGraph {
        log_radii: log_radii.unwrap(),
        dt_reference_tangent,
        previous_cell_indices,
        dt_grad_previous,
        dt_grad_current,
        dt_grad_next,
        surface_normals,
        surface_offsets,
        dt_grad_surface_normal,
        surface_normal_loss_scale,
    });
    let actual_radii = weighted_path
        .as_ref()
        .map(|weighted| positive_activation(g, weighted.log_radii, n_cells, RADIUS_SOFTPLUS_BETA));
    let differentiable_positions = if options.train_positions {
        positions
    } else {
        g.stop_gradient(positions)
    };
    let differentiable_radii = actual_radii.map(|radii| {
        if options.train_radii {
            radii
        } else {
            g.stop_gradient(radii)
        }
    });
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
    let view_idx = g.input_u32("view_idx", &[p]);
    let sh_coefficients = declare_sh_parameters(g, n_cells, num_components);

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
    let pos_cell = g.embedding(cell_indices, differentiable_positions); // [P*L, 3]
    let pos_next = g.embedding(next_cell_indices, differentiable_positions); // [P*L, 3]
    let half_pl3 = g.constant(vec![0.5_f32; pl * 3], &[pl, 3]);
    let pos_sum = g.add(pos_cell, pos_next);
    let midpoint = g.mul(pos_sum, half_pl3);
    let neg_pos_cell = g.neg(pos_cell);
    let normal = g.add(pos_next, neg_pos_cell);
    let normal_squared = g.mul(normal, normal);

    // Gather per-pixel origins and directions into the `[P*L, 3]` path layout.
    let ray_origin_pl = g.embedding(pixel_idx_per_step, ray_origin);
    let ray_dir_pl = g.embedding(pixel_idx_per_step, ray_dir_per_pixel);
    // The same oriented plane drives the recorded-path linearization, spatial
    // appearance, and optional view-facing loss. Gather it once so those
    // branches share both the forward payload and its backward accumulation.
    let step_surface_normals =
        normalized_surface_normals.map(|normals| g.embedding(cell_indices, normals));
    let step_surface_offsets = surface_offsets.map(|offsets| g.embedding(cell_indices, offsets));
    let surface_detail_evaluation = surface_detail.as_ref().map(|parameters| {
        let base_offsets =
            step_surface_offsets.unwrap_or_else(|| g.constant(vec![0.0_f32; pl], &[pl, 1]));
        let radii = g.embedding(cell_indices, differentiable_radii.unwrap());
        let negative_radius_floor = g.constant(vec![-1.0e-6_f32; pl], &[pl, 1]);
        let radius_above_floor = g.add(radii, negative_radius_floor);
        let radius_above_floor = g.relu(radius_above_floor);
        let radius_floor = g.constant(vec![1.0e-6_f32; pl], &[pl, 1]);
        let radii = g.add(radius_above_floor, radius_floor);
        evaluate_surface_detail_graph(
            g,
            cell_indices,
            parameters,
            pos_cell,
            step_surface_normals.unwrap(),
            base_offsets,
            radii,
            ray_origin_pl,
            ray_dir_pl,
            pl,
        )
    });
    let effective_surface_offsets = surface_detail_evaluation
        .as_ref()
        .map(|evaluation| evaluation.effective_offsets)
        .or(step_surface_offsets);
    let surface_basis = surface_color_coefficients.map(|_| {
        surface_color_basis_graph(
            g,
            cell_indices,
            positions,
            step_surface_normals.unwrap(),
            effective_surface_offsets,
            actual_radii.unwrap(),
            ray_origin_pl,
            ray_dir_pl,
            pl,
        )
    });

    let neg_ray_origin_pl = g.neg(ray_origin_pl);
    let mo_diff = g.add(midpoint, neg_ray_origin_pl); // [P*L, 3]

    // dot products via element-wise mul then matmul against [3, 1] ones.
    let ones_3_1 = g.constant(vec![1.0_f32; 3], &[3, 1]);
    let normal_length_squared = g.matmul(normal_squared, ones_3_1);
    let mn_prod = g.mul(mo_diff, normal);
    let dot_num = g.matmul(mn_prod, ones_3_1); // [P*L, 1]
    let nd_prod = g.mul(normal, ray_dir_pl);
    let dot_den_raw = g.matmul(nd_prod, ones_3_1);
    // A plain `dot_den_raw + ε` can cancel when a nearly parallel face has
    // `dot_den_raw ≈ -ε`, producing infinite position gradients before the
    // path mask or dt clamp can suppress the entry. Use the bounded
    // Tikhonov reciprocal `d / (d² + ε²)` instead. It preserves `1/d` away
    // from parallel faces, keeps the sign, maps an invalid zero normal to
    // zero, and bounds both the forward value and its derivative.
    let dot_den_squared = g.mul(dot_den_raw, dot_den_raw);
    let eps_squared_pl1 = g.constant(vec![1.0e-12_f32; pl], &[pl, 1]);
    let regularized_denominator = g.add(dot_den_squared, eps_squared_pl1);
    let regularized_numerator = g.mul(dot_num, dot_den_raw);
    let t_pl1 = g.div(regularized_numerator, regularized_denominator);
    let t_2d = g.reshape(t_pl1, &[p, l]); // [P, L]

    let t_shifted = g.shift_inner(t_2d, 1); // [P, L], t[p, k-1] or zero
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
    let normal_gate_epsilon = g.constant(vec![1.0e-6_f32; pl], &[pl, 1]);
    let normal_gate_denominator = g.add(normal_length_squared, normal_gate_epsilon);
    let normal_gate_flat = g.div(normal_length_squared, normal_gate_denominator);
    let normal_gate = g.reshape(normal_gate_flat, &[p, l]);
    let neg_normal_gate = g.neg(normal_gate);
    let ones_for_gate = g.constant(vec![1.0_f32; pl], &[p, l]);
    let terminal_gate = g.add(ones_for_gate, neg_normal_gate);
    let selected_dt = if use_recorded_dt {
        // The recorder evaluates the exact weighted, sphere-clipped interval
        // and its active-branch Jacobian. Evaluate its local linearization as
        // `dt_ref + tangent_actual - tangent_ref`; keeping `dt_ref` separate
        // avoids losing its low bits when the tangent is much larger. Omit the
        // position/radius tangent entirely when both tables are frozen; an
        // oriented graph then carries only its trainable surface-plane term.
        let weighted = weighted_path.as_ref().unwrap();
        let geometry_terms = match (
            weighted.previous_cell_indices,
            weighted.dt_grad_previous,
            weighted.dt_grad_current,
            weighted.dt_grad_next,
        ) {
            (
                Some(previous_indices),
                Some(previous_gradient),
                Some(current_gradient),
                Some(next_gradient),
            ) => {
                // Activate radii once at the parameter table, then pack
                // `(x,y,z,r)`. Each path role gathers the same vec4 table.
                let positions_flat = g.reshape(differentiable_positions, &[n_cells * 3]);
                let actual_radii_flat = g.reshape(differentiable_radii.unwrap(), &[n_cells]);
                let geometry_flat =
                    g.concat(positions_flat, actual_radii_flat, n_cells as u32, 3, 1, 1);
                let geometry = g.reshape(geometry_flat, &[n_cells, 4]);
                let zero_radius = g.constant(vec![0.0_f32; p], &[p, 1]);
                let ray_origin_flat = g.reshape(ray_origin, &[p * 3]);
                let zero_radius_flat = g.reshape(zero_radius, &[p]);
                let ray_origin_geometry_flat =
                    g.concat(ray_origin_flat, zero_radius_flat, p as u32, 3, 1, 1);
                let ray_origin_geometry = g.reshape(ray_origin_geometry_flat, &[p, 4]);
                let ray_origin_geometry_pl = g.embedding(pixel_idx_per_step, ray_origin_geometry);
                let neg_ray_origin_geometry = g.neg(ray_origin_geometry_pl);
                let previous_term = weighted_role_linear_term(
                    g,
                    previous_indices,
                    geometry,
                    neg_ray_origin_geometry,
                    previous_gradient,
                );
                let current_term = weighted_role_linear_term(
                    g,
                    cell_indices,
                    geometry,
                    neg_ray_origin_geometry,
                    current_gradient,
                );
                let next_term = weighted_role_linear_term(
                    g,
                    next_cell_indices,
                    geometry,
                    neg_ray_origin_geometry,
                    next_gradient,
                );
                let entry_and_current = g.add(previous_term, current_term);
                Some(g.add(entry_and_current, next_term))
            }
            (None, None, None, None) => None,
            _ => unreachable!("geometry path inputs must be declared together"),
        };
        let linear_terms = match (
            step_surface_normals,
            effective_surface_offsets,
            weighted.dt_grad_surface_normal,
            geometry_terms,
        ) {
            (Some(step_normals), offsets, Some(recorded_gradient), geometry_terms) => {
                let recorded_gradient = g.materialize(recorded_gradient);
                let gradient_xyz_flat = g.split_a(recorded_gradient, pl as u32, 3, 1, 1);
                let gradient_xyz = g.reshape(gradient_xyz_flat, &[pl, 3]);
                let product = g.mul(step_normals, gradient_xyz);
                let mut surface_term = g.sum_inner(product);
                if let Some(offsets) = offsets {
                    let gradient_w_flat = g.split_b(recorded_gradient, pl as u32, 3, 1, 1);
                    let gradient_w = g.reshape(gradient_w_flat, &[pl, 1]);
                    let offset_term = g.mul(offsets, gradient_w);
                    surface_term = g.add(surface_term, offset_term);
                }
                Some(match geometry_terms {
                    Some(geometry_terms) => g.add(geometry_terms, surface_term),
                    None => surface_term,
                })
            }
            (None, None, None, geometry_terms) => geometry_terms,
            _ => unreachable!("oriented path graph inputs must be declared together"),
        };
        match (linear_terms, weighted.dt_reference_tangent) {
            (Some(linear_terms), Some(reference_tangent)) => {
                let reference_tangent = g.reshape(reference_tangent, &[pl, 1]);
                let neg_reference_tangent = g.neg(reference_tangent);
                let tangent_delta = g.add(linear_terms, neg_reference_tangent);
                let recorded_dt_flat = g.reshape(recorded_dt, &[pl, 1]);
                let linear_dt_flat = g.add(recorded_dt_flat, tangent_delta);
                let linear_dt = g.reshape(linear_dt_flat, &[p, l]);
                let positive_dt = g.relu(linear_dt);
                let neg_positive_dt = g.neg(positive_dt);
                let remaining = g.add(max_dt_pl, neg_positive_dt);
                let within_cap = g.relu(remaining);
                let neg_within_cap = g.neg(within_cap);
                g.add(max_dt_pl, neg_within_cap)
            }
            (None, None) => g.reshape(recorded_dt, &[p, l]),
            _ => unreachable!("path tangent and reference must be declared together"),
        }
    } else {
        let face_dt = g.mul(dt_clamped, normal_gate);
        let recorded_dt_2d = g.reshape(recorded_dt, &[p, l]);
        let terminal_dt = g.mul(recorded_dt_2d, terminal_gate);
        g.add(face_dt, terminal_dt)
    };
    let dt_2d = g.mul(selected_dt, mask_2d); // zero out invalid steps
    let dt_from_positions = dt_2d;

    let raw = g.mul(density, dt_2d); // [P, L], non-negative

    // cum[p, k] = sum_{i<k} raw[p, i]. `dt_2d` is already masked, so a
    // second multiplication by the same binary path mask would be redundant.
    let cumsum = g.exclusive_cumsum(raw, false); // [P, L]

    // Transmittance: T = exp(-cumsum). Use the identity
    //   exp(-x) = recip(sigmoid(x)) - 1   (valid for x >= 0)
    let ones_pl = g.constant(vec![1.0; p * l], &[p, l]);
    let twos_pl = g.constant(vec![2.0; p * l], &[p, l]);
    let sig_cum = g.sigmoid(cumsum);
    let rec_sig_cum = g.recip(sig_cum);
    let neg_ones_pl = g.neg(ones_pl);
    let t = g.add(rec_sig_cum, neg_ones_pl); // [P, L]

    // alpha = 1 - exp(-raw) = 2 - recip(sigmoid(raw))
    let sig_raw = g.sigmoid(raw);
    let rec_sig_raw = g.recip(sig_raw);
    let neg_rec_sig_raw = g.neg(rec_sig_raw);
    let alpha = g.add(twos_pl, neg_rec_sig_raw); // [P, L]

    // A padded step has masked `dt = 0`, hence `raw = 0`, `alpha = 0`, and
    // `weight = 0`; another path-mask multiplication here would be redundant.
    let weight = g.mul(t, alpha);

    // PowerFoam dipoles should face the camera. Penalize the normal-facing
    // half-cell when it points along the camera ray: mean over rays of
    // `sum_segments(T * alpha * max(dot(normal, ray_dir), 0)^2)`.
    let surface_normal_loss = match (
        step_surface_normals,
        weighted_path
            .as_ref()
            .and_then(|weighted| weighted.surface_normal_loss_scale),
    ) {
        (Some(step_normals), Some(scale)) => {
            let normal_ray_product = g.mul(step_normals, ray_dir_pl);
            let normal_ray_dot_flat = g.sum_inner(normal_ray_product);
            let normal_ray_dot = g.reshape(normal_ray_dot_flat, &[p, l]);
            let facing = g.relu(normal_ray_dot);
            let facing_squared = g.mul(facing, facing);
            let weighted_facing = g.mul(weight, facing_squared);
            let sum = g.sum_all(weighted_facing);
            let sum_2d = g.reshape(sum, &[1, 1]);
            let mean_scale = g.constant(vec![1.0_f32 / p as f32], &[1, 1]);
            let mean = g.mul(sum_2d, mean_scale);
            Some(g.mul(mean, scale))
        }
        (_, None) => None,
        (None, Some(_)) => unreachable!("surface-normal loss requires oriented normals"),
    };

    // Per-channel pixel: pixel_c = (weight * color_c) @ ones_L
    let ones_1l = g.constant(vec![1.0; l], &[1, l]);

    // Accumulated opacity per pixel = Σ_L weight = 1 − T_final. Drives the
    // RadFoam opacity loss + white-background compositing.
    let opacity = g.sum_inner(weight); // [P, 1]
    let sh = pixel_sh(
        g,
        cell_indices,
        &sh_coefficients,
        surface_color_coefficients,
        surface_basis,
        surface_detail.as_ref().zip(
            surface_detail_evaluation
                .as_ref()
                .map(|evaluation| evaluation.color_weights),
        ),
        spherical_voronoi
            .as_ref()
            .map(|parameters| (parameters, ray_dir_pl)),
        &basis_inputs,
        pixel_idx_per_step,
        weight,
        n_cells,
        p,
        l,
    );
    let [pixel_r, pixel_g, pixel_b] = sh.pixels;

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

    // PowerFoam resamples sites according to an EMA of each site's
    // photometric responsibility, not position-gradient magnitude. Encode the
    // exact per-segment score into the gradient of a frozen, zero-forward
    // probe parameter. Build this branch as soon as its inputs exist so the
    // per-step colors need not stay live through the remaining loss graph.
    let point_error = collect_point_error.then(|| {
        let target_channels = [target_r, target_g, target_b];
        let mut errors = Vec::with_capacity(3);
        for (color, target_channel) in sh.step_colors.into_iter().zip(target_channels) {
            let target_per_step = g.matmul(target_channel, ones_1l);
            let neg_target = g.neg(target_per_step);
            let difference = g.add(color, neg_target);
            errors.push(g.abs(difference));
        }
        let error_rg = g.add(errors[0], errors[1]);
        let color_error = g.add(error_rg, errors[2]);
        let weighted_error = g.mul(weight, color_error);
        let weighted_error_flat = g.reshape(weighted_error, &[pl, 1]);
        let weighted_error_flat = g.stop_gradient(weighted_error_flat);

        let probe = g.parameter(POINT_ERROR_PROBE, &[n_cells, 1]);
        let gathered = g.embedding(cell_indices, probe);
        let frozen = g.stop_gradient(gathered);
        let neg_frozen = g.neg(frozen);
        let zero_forward = g.add(gathered, neg_frozen);
        let encoded_error = g.mul(zero_forward, weighted_error_flat);
        (probe, g.sum_all(encoded_error))
    });

    // --- Per-view exposure ---
    //
    // Each channel's exposure is its own `[num_views, 1]` table. Gather
    // the row for each pixel's view via embedding (which is fully
    // differentiable — backward is scatter_add into the table), then
    // elementwise-multiply the rendered channel.
    let ones_p1 = g.constant(vec![1.0_f32; p], &[p, 1]);
    let exp_r_p = g.embedding(view_idx, exposure_r); // [p, 1]
    let pixel_r = g.mul(pixel_r, exp_r_p);
    let exp_g_p = g.embedding(view_idx, exposure_g);
    let pixel_g = g.mul(pixel_g, exp_g_p);
    let exp_b_p = g.embedding(view_idx, exposure_b);
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

    let loss_r = match color_loss_kind {
        ColorLoss::L1 => g.l1_loss(pixel_r, target_r),
        ColorLoss::SmoothL1 => smooth_l1_loss(g, pixel_r, target_r, p),
    };
    let loss_g = match color_loss_kind {
        ColorLoss::L1 => g.l1_loss(pixel_g, target_g),
        ColorLoss::SmoothL1 => smooth_l1_loss(g, pixel_g, target_g, p),
    };
    let loss_b = match color_loss_kind {
        ColorLoss::L1 => g.l1_loss(pixel_b, target_b),
        ColorLoss::SmoothL1 => smooth_l1_loss(g, pixel_b, target_b, p),
    };
    let loss_rg = g.add(loss_r, loss_g);
    let color_loss_sum = g.add(loss_rg, loss_b);
    let l1 = match color_loss_kind {
        ColorLoss::L1 => color_loss_sum,
        ColorLoss::SmoothL1 => {
            let one_third = g.constant(vec![1.0_f32 / 3.0], &[1]);
            g.mul(color_loss_sum, one_third)
        }
    };

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

    // RadFoam opacity loss. Ordinary RGB captures supply an all-ones target,
    // preserving the opaque-scene reference behavior. A masked capture can
    // instead supervise foreground opacity and empty background explicitly.
    // `loss = color + opacity_weight · mean((opacity − target_alpha)²)`.
    let loss = if opacity_weight > 0.0 {
        let op_loss = g.mse_loss(opacity, target_alpha.unwrap()); // scalar [1]
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
        let depth_prefix = g.exclusive_cumsum(dt_2d, false);
        let half = g.constant(vec![0.5_f32; p * l], &[p, l]);
        let half_dt = g.mul(dt_2d, half);
        let midpoint = g.add(depth_prefix, half_dt);
        let weighted_midpoint = g.mul(weight, midpoint);
        let first_moment = g.sum_inner(weighted_midpoint);
        let midpoint_squared = g.mul(midpoint, midpoint);
        let weighted_midpoint_squared = g.mul(weight, midpoint_squared);
        let second_moment = g.sum_inner(weighted_midpoint_squared);
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
            p,
            l,
        );
        let total_optical_depth = g.sum_inner(raw);
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

    let loss = match surface_normal_loss {
        Some(normal_loss) => {
            let base = g.reshape(loss, &[1, 1]);
            let total = g.add(base, normal_loss);
            g.reshape(total, &[1])
        }
        None => loss,
    };

    let (loss, point_error_probe) = match point_error {
        Some((probe, probe_loss)) => (g.add(loss, probe_loss), Some(probe)),
        None => (loss, None),
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
        point_error_probe,
        weighted_path,
        sh_coefficients,
        surface_color_coefficients,
        surface_detail,
        spherical_voronoi,
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

#[allow(clippy::too_many_arguments)]
fn optical_depth_quantile(
    g: &mut mn::Graph,
    optical_depth: mn::NodeId,
    prefix_optical_depth: mn::NodeId,
    density: mn::NodeId,
    dt: mn::NodeId,
    mask: mn::NodeId,
    ones_1l: mn::NodeId,
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
    g.sum_inner(distance_masked)
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

/// Concatenate scalar columns into a row-major matrix. A balanced tree keeps
/// each value moving O(log K) times when many SH components are present.
fn concat_columns(g: &mut mn::Graph, columns: &[mn::NodeId], rows: usize) -> mn::NodeId {
    assert!(
        !columns.is_empty(),
        "concat_columns needs at least one column"
    );
    let mut parts: Vec<(mn::NodeId, u32)> =
        columns.iter().copied().map(|column| (column, 1)).collect();
    while parts.len() > 1 {
        let mut next = Vec::with_capacity(parts.len().div_ceil(2));
        let mut iter = parts.into_iter();
        while let Some((a, channels_a)) = iter.next() {
            if let Some((b, channels_b)) = iter.next() {
                let merged = g.concat(a, b, rows as u32, channels_a, channels_b, 1);
                next.push((merged, channels_a + channels_b));
            } else {
                next.push((a, channels_a));
            }
        }
        parts = next;
    }
    let (flat, channels) = parts[0];
    g.reshape(flat, &[rows, channels as usize])
}

fn split_rgb_table(
    g: &mut mn::Graph,
    table: mn::NodeId,
    rows: usize,
    channel_width: usize,
) -> [mn::NodeId; 3] {
    let rg = g.split_a(
        table,
        rows as u32,
        (2 * channel_width) as u32,
        channel_width as u32,
        1,
    );
    let blue = g.split_b(
        table,
        rows as u32,
        (2 * channel_width) as u32,
        channel_width as u32,
        1,
    );
    let red = g.split_a(
        rg,
        rows as u32,
        channel_width as u32,
        channel_width as u32,
        1,
    );
    let green = g.split_b(
        rg,
        rows as u32,
        channel_width as u32,
        channel_width as u32,
        1,
    );
    [red, green, blue].map(|channel| g.reshape(channel, &[rows, channel_width]))
}

struct PixelSh {
    pixels: [mn::NodeId; 3],
    step_colors: [mn::NodeId; 3],
}

#[allow(clippy::too_many_arguments)]
fn pixel_sh(
    g: &mut mn::Graph,
    cell_indices: mn::NodeId,
    sh_coefficients: &[ShChannelGraph],
    surface_color_coefficients: Option<mn::NodeId>,
    surface_basis: Option<mn::NodeId>,
    surface_detail: Option<(&SurfaceDetailGraph, mn::NodeId)>,
    spherical_voronoi: Option<(&SphericalVoronoiGraph, mn::NodeId)>,
    basis_inputs: &[mn::NodeId], // K-1 per-pixel basis [P, 1]; basis_0 is SH_C0 constant
    pixel_idx_per_step: mn::NodeId,
    weight: mn::NodeId,
    n_cells: usize,
    p: usize,
    l: usize,
) -> PixelSh {
    assert_eq!(sh_coefficients.len(), 3, "pixel_sh needs RGB tables");
    let k = basis_inputs.len() + 1;
    assert!(k > 0);
    assert!(sh_coefficients.iter().all(|channel| {
        if k == 1 {
            channel.rest.is_none()
        } else {
            channel.rest.is_some()
        }
    }));
    assert_eq!(
        basis_inputs.len(),
        k.saturating_sub(1),
        "pixel_sh: basis_inputs.len() ({}) must equal coefficient count - 1 ({})",
        basis_inputs.len(),
        k.saturating_sub(1),
    );

    let pl = p * l;
    let coefficient_tables: Vec<mn::NodeId> = sh_coefficients
        .iter()
        .map(|channel| match channel.rest {
            Some(rest) => {
                let table = g.concat(channel.dc, rest, n_cells as u32, 1, (k - 1) as u32, 1);
                g.reshape(table, &[n_cells, k])
            }
            None => channel.dc,
        })
        .collect();

    let mut basis_columns = Vec::with_capacity(k);
    basis_columns.push(g.constant(vec![SH_C0; p], &[p, 1]));
    basis_columns.extend_from_slice(basis_inputs);
    let basis_per_pixel = concat_columns(g, &basis_columns, p); // [P, K]
    let basis = g.embedding(pixel_idx_per_step, basis_per_pixel);
    // Camera directions are inputs, not learned values. Keeping RGB as three
    // reductions lets meganeura fold each coefficient embedding and multiply
    // into the reduction without copying a repeated `[PL, 3*K]` basis.
    let basis = g.stop_gradient(basis);
    let mut colors: Vec<mn::NodeId> = coefficient_tables
        .into_iter()
        .map(|table| {
            let coefficients = g.embedding(cell_indices, table);
            let terms = g.mul(coefficients, basis);
            g.sum_inner(terms)
        })
        .collect();

    match (surface_color_coefficients, surface_basis) {
        (Some(coefficient_table), Some(basis)) => {
            let tables =
                split_rgb_table(g, coefficient_table, n_cells, vol::SURFACE_COLOR_COMPONENTS);
            for (color, table) in colors.iter_mut().zip(tables) {
                let coefficients = g.embedding(cell_indices, table);
                let terms = g.mul(coefficients, basis);
                let residual = g.sum_inner(terms);
                *color = g.add(*color, residual);
            }
        }
        (None, None) => {}
        _ => unreachable!("surface color parameter and basis must be declared together"),
    }
    if let Some((parameters, weights)) = surface_detail {
        let tables = split_rgb_table(g, parameters.colors, n_cells, vol::SURFACE_DETAIL_SITES);
        for (color, table) in colors.iter_mut().zip(tables) {
            let coefficients = g.embedding(cell_indices, table);
            let terms = g.mul(coefficients, weights);
            let residual = g.sum_inner(terms);
            *color = g.add(*color, residual);
        }
    }
    match spherical_voronoi {
        Some((parameters, ray_dir_pl)) => {
            let axes = g.embedding(cell_indices, parameters.axes);
            let axes = g.reshape(axes, &[pl * vol::SPHERICAL_VORONOI_SITES, 3]);

            let direction = g.stop_gradient(ray_dir_pl);
            let direction_flat = g.reshape(direction, &[pl * 3]);
            let direction_2 = g.concat(direction_flat, direction_flat, pl as u32, 3, 3, 1);
            let direction_4 = g.concat(direction_2, direction_2, pl as u32, 6, 6, 1);
            let direction_8 = g.concat(direction_4, direction_4, pl as u32, 12, 12, 1);
            let direction_sites = g.reshape(direction_8, &[pl * vol::SPHERICAL_VORONOI_SITES, 3]);
            let logit_terms = g.mul(axes, direction_sites);
            let logits_flat = g.sum_inner(logit_terms);
            let logits = g.reshape(logits_flat, &[pl, vol::SPHERICAL_VORONOI_SITES]);
            let weights = g.softmax(logits);

            let tables =
                split_rgb_table(g, parameters.colors, n_cells, vol::SPHERICAL_VORONOI_SITES);
            for (color, table) in colors.iter_mut().zip(tables) {
                let coefficients = g.embedding(cell_indices, table);
                let terms = g.mul(coefficients, weights);
                let residual = g.sum_inner(terms);
                *color = g.add(*color, residual);
            }
        }
        None => {}
    }

    let bias = g.constant(vec![0.5; pl], &[pl, 1]);
    let step_colors: [mn::NodeId; 3] = colors
        .into_iter()
        .map(|color| {
            let biased = g.add(color, bias);
            // Match RadFoam's per-cell `max(rgb, 0)` before volumetric
            // compositing. Clamping only the accumulated pixel is not equivalent.
            let non_negative = g.relu(biased);
            g.reshape(non_negative, &[p, l])
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("pixel_sh needs exactly three channels");
    let pixels = step_colors.map(|color_2d| {
        let weighted = g.mul(weight, color_2d);
        g.sum_inner(weighted)
    });
    PixelSh {
        pixels,
        step_colors,
    }
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
const MIN_PROJECTED_RAYS_PER_CAMERA: usize = 1024;

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
    /// Parameter-specific schedule from official RadFoam v1. Density warms
    /// up for the first 10%, higher-order SH for 20%, and positions freeze
    /// after 90% of the global budget. Absolute rates and Adam epsilon match
    /// the reference; `learning_rate`, `lr_min_ratio`, and the historical
    /// position/radius multipliers are ignored.
    RadFoamV1,
}

/// Static parameter-group multipliers used with constant or cosine schedules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrGroups {
    /// Historical blade-volume behavior: position uses
    /// [`AppearanceFitConfig::position_lr_ratio`], while every SH coefficient
    /// uses `0.1×` when degree is nonzero (or `1×` at degree zero).
    Legacy,
    /// Ratios between the official RadFoam v1 initial rates, normalized by
    /// its density rate: position `0.002×`, SH DC `0.05×`, higher SH
    /// `0.005×`. The global time curve, Adam epsilon, and final-to-initial
    /// ratio still come from the selected constant/cosine configuration.
    RadFoamV1Relative,
}

/// Cadence for rebuilding adjacency and recorded paths while geometry moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryRebuildSchedule {
    /// Rebuild every [`AppearanceFitConfig::geometry_rebuild_every`] global
    /// optimizer steps.
    Fixed,
    /// Official RadFoam v1 cadence. Rebuild after 1 step, then after 3, 5,
    /// 7, ... 99, and 101 steps. The period remains 101 thereafter and
    /// resets to 1 after a successful densification round.
    RadFoamV1,
}

/// Cadence and stopping policy for adaptive cloud densification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensifySchedule {
    /// Grow every [`DensifyConfig::every`] steps from `warmup` until
    /// `densify_until`, stopping at `target_points`.
    Fixed,
    /// Official RadFoam v1 policy. Grow first at `warmup`, then derive the
    /// next interval from the post-growth cell count and clamp it to at least
    /// 100 steps. Growth stops once the current count reaches 90% of
    /// `target_points`; `densify_until` defines the linear-growth horizon in
    /// the interval formula rather than acting as a hard stop.
    RadFoamV1,
}

const SAMPLING_RNG_SEED: u64 = 0xDEAD_BEEF_F00D_CAFE;
const QUANTILE_RNG_SEED: u64 = 0x51A7_E5D0_9B3C_2468;
const DENSIFY_RNG_SEED: u64 = 0xCAFE_F00D_DEAD_BEEF;
const INTERPENETRATION_RNG_SEED: u64 = 0xC0DE_C7ED_6E0F_AA5A;
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const TRAINING_STATE_HEADER: &str = "blade-volume-training-state-v3";
const TOPOLOGY_TRAINING_STATE_HEADER: &str = "blade-volume-training-state-v2";
const LEGACY_TRAINING_STATE_HEADER: &str = "blade-volume-training-state-v1";

/// Deterministic trainer state which is not owned by meganeura.
///
/// Parameter values, Adam moments, and the Adam counter live in the paired
/// safetensors checkpoint. This sidecar preserves view/pixel sampling,
/// quantile regularization, densification decisions, and dynamic topology
/// cadence across a process restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingState {
    pub step: usize,
    pub cycle: usize,
    pub sampling_rng: u64,
    pub quantile_rng: u64,
    pub densify_rng: u64,
    /// Current RadFoam topology-update period. Zero means the fixed schedule
    /// was active when the checkpoint was written.
    pub topology_period: usize,
    /// RadFoam topology steps accumulated toward `topology_period`.
    pub topology_steps_since_update: usize,
    /// Original cloud size used by the RadFoam densification interval
    /// formula. Zero means the fixed densification schedule was active.
    pub densify_initial_points: usize,
    /// Active steps accumulated toward `densify_next_after`.
    pub densify_steps_since_update: usize,
    /// Current RadFoam densification interval.
    pub densify_next_after: usize,
    /// Number of successful densification rounds already completed.
    pub densification_round: usize,
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
    /// Static parameter groups for constant/cosine schedules. The exact
    /// [`LrSchedule::RadFoamV1`] policy supplies its own time-varying absolute
    /// groups and ignores this field.
    pub lr_groups: LrGroups,
    /// Floor of the cosine schedule, as a fraction of `learning_rate`.
    /// `0.01` decays to 1 % of base by the final step. Unused when
    /// `lr_schedule == Constant`.
    pub lr_min_ratio: f32,
    /// When `Some(K)`, each Adam step samples `K` random pixels from a
    /// randomly chosen training view. `None` feeds every pixel exactly once
    /// per step and uses `epochs` steps per view. Both modes use the same GPU
    /// path recorder and differentiable graph.
    pub pixel_batch: Option<usize>,
    /// Camera views represented in each random-pixel Adam batch. Rays are
    /// split evenly across a deterministic stratified sample of this many
    /// views. The value is capped to the available views and rays. Default 1
    /// preserves historical one-view batches. Patch and full-image modes
    /// require 1.
    pub views_per_batch: usize,
    /// Number of Adam steps per view in randomly batched mode. Default 200.
    pub steps_per_view: usize,
    /// SH degree for view-dependent colour. 0 = flat colour (default,
    /// matches the original radfoam pipeline). 1–3 enable view
    /// dependence: ~3-5 dB PSNR improvement on Mip-NeRF 360 scenes at
    /// the cost of `(1+sh_degree)²` per-cell parameters per RGB
    /// channel.
    pub sh_degree: usize,
    /// Supervised RGB loss. The official RadFoam v1 trainer uses
    /// [`ColorLoss::SmoothL1`]; [`ColorLoss::L1`] preserves the historical
    /// blade-volume behavior.
    pub color_loss: ColorLoss,
    /// Adaptive densification. Unweighted RadFoam samples by accumulated
    /// position-gradient magnitude times cell radius; weighted PowerFoam
    /// samples by per-site photometric-error EMA. `None` disables
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
    /// Weight on the RadFoam opacity loss. Views without an alpha target use
    /// one, matching the opaque-scene reference behavior. Views with a mask
    /// train opacity to that mask, including transparent background rays.
    /// `0.0` (default) disables the term.
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
    /// Initial weight on PowerFoam's interpenetration loss. It penalizes
    /// squared radial overlap across the current directed Čech graph and
    /// decays exponentially to one-thousandth of this value by the final
    /// step. `0.0` (default) disables it; the reference starts at `1e-4`.
    pub interpenetration_weight: f32,
    /// Directed Čech edges sampled per optimizer step for the interpenetration
    /// loss. The sampled sum is scaled to estimate the complete graph. This
    /// bounds graph size independently of point count.
    pub interpenetration_samples: usize,
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
    /// (default) freezes geometry. A positive value requires a geometry
    /// rebuild schedule; the fixed schedule additionally requires
    /// `geometry_rebuild_every > 0` because discrete adjacency and recorded
    /// cell walks must be refreshed as points move.
    pub position_lr_ratio: f32,
    /// PowerFoam support-radius learning rate as a fraction of
    /// `learning_rate`. Radii are optimized through a β=100 softplus and this
    /// must remain zero for unweighted clouds. Like position optimization, a
    /// positive value requires periodic geometry rebuilds.
    pub radius_lr_ratio: f32,
    /// Oriented PowerFoam surface-normal learning rate as a fraction of the
    /// global rate. Under [`LrSchedule::RadFoamV1`] it instead scales the
    /// official `0.1 → 0.01` orientation schedule. `0.0` freezes orientation.
    pub surface_normal_lr_ratio: f32,
    /// Signed surface-plane offset learning rate as a fraction of the global
    /// rate. Under [`LrSchedule::RadFoamV1`] it scales PowerFoam's
    /// `5e-3 → 5e-4` height schedule. `0.0` freezes offsets.
    pub surface_offset_lr_ratio: f32,
    /// Spatial surface-color learning rate as a fraction of the global rate.
    /// Under [`LrSchedule::RadFoamV1`] it scales the same `5e-3 → 5e-4`
    /// schedule as the surface-plane offset. `0.0` freezes the residual.
    pub surface_color_lr_ratio: f32,
    /// Radius-normalized spatial-detail site learning rate as a fraction of
    /// the global rate. Under [`LrSchedule::RadFoamV1`] it scales a
    /// `1e-2 → 1e-3` schedule. `0.0` freezes the sites.
    pub surface_detail_offset_lr_ratio: f32,
    /// Radius-normalized spatial-detail height learning rate as a fraction of
    /// the global rate. Under [`LrSchedule::RadFoamV1`] it scales the
    /// `5e-3 → 5e-4` surface schedule. `0.0` freezes the heights.
    pub surface_detail_height_lr_ratio: f32,
    /// Spatial-detail RGB residual learning rate as a fraction of the global
    /// rate. Under [`LrSchedule::RadFoamV1`] it scales the `5e-3 → 5e-4`
    /// surface schedule. `0.0` freezes the colors.
    pub surface_detail_color_lr_ratio: f32,
    /// Raw Spherical Voronoi axis learning rate as a fraction of the global
    /// rate. Axis magnitude is directional temperature. Under
    /// [`LrSchedule::RadFoamV1`] this scales PowerFoam's `5e-2 → 5e-3`
    /// directional-axis schedule. `0.0` freezes the sites.
    pub spherical_voronoi_axis_lr_ratio: f32,
    /// Spherical Voronoi RGB-site learning rate as a fraction of the global
    /// rate. Under [`LrSchedule::RadFoamV1`] this scales PowerFoam's
    /// `5e-3 → 5e-4` directional-color schedule. `0.0` freezes the values.
    pub spherical_voronoi_color_lr_ratio: f32,
    /// Initial weight of PowerFoam's view-facing normal regularizer. It
    /// decays exponentially to one tenth of this value over training. `0.0`
    /// disables the term; the reference uses `0.1`.
    pub surface_normal_weight: f32,
    /// Minimum PowerFoam sphere-candidate row capacity. Zero selects the
    /// automatic `max(4 * max_steps, 1024)` budget. This remains independent
    /// of the shorter surviving path/Jacobian row.
    pub powerfoam_candidate_capacity: u32,
    /// Number of Adam steps between adjacency, GPU-cloud, and path-buffer
    /// rebuilds while positions or radii are trainable under the fixed
    /// schedule. Ignored by [`GeometryRebuildSchedule::RadFoamV1`] and when
    /// geometry is frozen.
    pub geometry_rebuild_every: usize,
    /// Fixed historical cadence or the increasing official RadFoam v1
    /// topology cadence. Default [`GeometryRebuildSchedule::Fixed`].
    pub geometry_rebuild_schedule: GeometryRebuildSchedule,
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
    /// Restores sampling, quantile, densification, and topology cadence state.
    pub resume_training_state: Option<TrainingState>,
}

/// Adaptive densification on the selected [`DensifySchedule`] after `warmup`.
/// RadFoam splits cells sampled by accumulated
/// `|grad(position)| × cell_radius`; PowerFoam uses the 99th-percentile-capped
/// EMA of `T × alpha × L1(cell_color, target)`.
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
    /// Fixed or official RadFoam v1 cadence. Default fixed.
    pub schedule: DensifySchedule,
    /// Steps between densify rounds under [`DensifySchedule::Fixed`].
    pub every: usize,
    /// Per-round growth factor: each round adds `fraction × current_cells`
    /// new cells (RadFoam uses 0.15 = +15%/round), selected by the
    /// method-specific statistic documented on [`DensifyConfig`].
    pub fraction: f32,
    /// Unused legacy knob (sibling jitter). RadFoam placement and PowerFoam's
    /// 5%-of-support-radius resampling are method-specific and fixed. Kept for
    /// CLI compatibility.
    pub jitter_scale: f32,
    /// Skip the first `warmup` steps before the first densify (lets
    /// the per-cell gradient signal settle). RadFoam: 2000.
    pub warmup: usize,
    /// Final cell-count budget (RadFoam Bonsai: 2,097,152). Fixed cadence
    /// stops at this count; RadFoam v1 stops scheduling at 90% of it, as in
    /// the reference trainer.
    pub target_points: usize,
    /// Fixed cadence stops after this step. RadFoam v1 instead uses the
    /// `warmup..densify_until` span in its cell-count-dependent interval
    /// formula; it is a growth horizon, not a hard cutoff in the reference.
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
    /// Maximum training views used by the contribution collector at each
    /// prune/densify boundary. `0` (default) evaluates every view. A positive
    /// value deterministically rotates a stratified subset between absolute
    /// densification rounds; this is an experimental scaling control and must
    /// be checked against exhaustive decisions before use in a quality run.
    pub contribution_views: usize,
}

impl Default for DensifyConfig {
    fn default() -> Self {
        Self {
            schedule: DensifySchedule::Fixed,
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
            contribution_views: 0,
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
            views_per_batch: 1,
            steps_per_view: 200,
            sh_degree: 0,
            color_loss: ColorLoss::L1,
            densify: None,
            lr_schedule: LrSchedule::Cosine,
            lr_groups: LrGroups::Legacy,
            lr_min_ratio: 0.01,
            patch_size: 0,
            grad_loss_weight: 0.0,
            opacity_weight: 0.0,
            distortion_weight: 0.0,
            quantile_weight: 0.0,
            interpenetration_weight: 0.0,
            interpenetration_samples: 4096,
            softplus_beta: 0.0,
            background_rgb: [0.0; 3],
            position_lr_ratio: 0.0,
            radius_lr_ratio: 0.0,
            surface_normal_lr_ratio: 0.0,
            surface_offset_lr_ratio: 0.0,
            surface_color_lr_ratio: 0.0,
            surface_detail_offset_lr_ratio: 0.0,
            surface_detail_height_lr_ratio: 0.0,
            surface_detail_color_lr_ratio: 0.0,
            spherical_voronoi_axis_lr_ratio: 0.0,
            spherical_voronoi_color_lr_ratio: 0.0,
            surface_normal_weight: 0.0,
            powerfoam_candidate_capacity: 0,
            geometry_rebuild_every: 0,
            geometry_rebuild_schedule: GeometryRebuildSchedule::Fixed,
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

fn sample_training_views(state: &mut u64, num_views: usize, count: usize) -> Vec<usize> {
    assert!(count > 0 && count <= num_views);
    (0..count)
        .map(|slot| {
            let begin = slot * num_views / count;
            let end = (slot + 1) * num_views / count;
            begin + (next_lcg_u32(state) as usize) % (end - begin)
        })
        .collect()
}

fn batch_view_range(slot: usize, count: usize, pixel_batch: usize) -> std::ops::Range<usize> {
    slot * pixel_batch / count..(slot + 1) * pixel_batch / count
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

fn interpenetration_weight_at_step(initial: f32, step: usize, total_steps: usize) -> f32 {
    if initial == 0.0 {
        return 0.0;
    }
    let phase = if total_steps <= 1 {
        0.0
    } else {
        step.min(total_steps - 1) as f32 / (total_steps - 1) as f32
    };
    initial * 1.0e-3_f32.powf(phase)
}

fn surface_normal_weight_at_step(initial: f32, step: usize, total_steps: usize) -> f32 {
    if initial == 0.0 {
        return 0.0;
    }
    let phase = if total_steps <= 1 {
        0.0
    } else {
        step.min(total_steps - 1) as f32 / (total_steps - 1) as f32
    };
    initial * 0.1_f32.powf(phase)
}

struct InterpenetrationBatch {
    edge_sources: Vec<u32>,
    edge_a: Vec<u32>,
    edge_b: Vec<u32>,
    edge_direction: Vec<f32>,
    edge_scale: Vec<f32>,
}

impl InterpenetrationBatch {
    fn new(model: &vol::PointCloudModel, sample_count: usize) -> Self {
        let mut batch = Self {
            edge_sources: Vec::new(),
            edge_a: vec![0; sample_count],
            edge_b: vec![0; sample_count],
            edge_direction: vec![0.0; sample_count * 3],
            edge_scale: vec![0.0; sample_count],
        };
        batch.rebuild(model);
        batch
    }

    fn rebuild(&mut self, model: &vol::PointCloudModel) {
        let adjacency = model
            .adjacency
            .as_ref()
            .expect("interpenetration loss requires adjacency");
        self.edge_sources.clear();
        self.edge_sources.reserve(adjacency.neighbors.len());
        for (cell, offsets) in adjacency.offsets.windows(2).enumerate() {
            self.edge_sources.resize(offsets[1] as usize, cell as u32);
        }
        assert_eq!(self.edge_sources.len(), adjacency.neighbors.len());
    }

    fn upload(
        &mut self,
        session: &mut mn::Session,
        model: &vol::PointCloudModel,
        step: usize,
        total_steps: usize,
        initial_weight: f32,
    ) {
        self.prepare(model, step, total_steps, initial_weight);
        session.set_input_u32("interpenetration_edge_a", &self.edge_a);
        session.set_input_u32("interpenetration_edge_b", &self.edge_b);
        session.set_input("interpenetration_edge_direction", &self.edge_direction);
        session.set_input("interpenetration_edge_scale", &self.edge_scale);
    }

    fn prepare(
        &mut self,
        model: &vol::PointCloudModel,
        step: usize,
        total_steps: usize,
        initial_weight: f32,
    ) {
        let adjacency = model
            .adjacency
            .as_ref()
            .expect("interpenetration loss requires adjacency");
        let edge_count = adjacency.neighbors.len();
        let sample_count = self.edge_a.len();
        let scheduled_weight = interpenetration_weight_at_step(initial_weight, step, total_steps);
        self.edge_scale.fill(0.0);

        if edge_count <= sample_count {
            self.edge_a[edge_count..].fill(0);
            self.edge_b[edge_count..].fill(0);
            self.edge_direction[edge_count * 3..].fill(0.0);
            for edge in 0..edge_count {
                self.fill_sample(model, adjacency, edge, edge, scheduled_weight);
            }
        } else {
            let mut rng = advance_lcg(
                INTERPENETRATION_RNG_SEED,
                (step as u64).wrapping_mul(sample_count as u64),
            );
            for sample in 0..sample_count {
                let begin = sample * edge_count / sample_count;
                let end = (sample + 1) * edge_count / sample_count;
                let edge = begin + next_lcg_u32(&mut rng) as usize % (end - begin);
                let sample_scale = scheduled_weight * (end - begin) as f32;
                self.fill_sample(model, adjacency, sample, edge, sample_scale);
            }
        }
    }

    fn fill_sample(
        &mut self,
        model: &vol::PointCloudModel,
        adjacency: &vol::Adjacency,
        sample: usize,
        edge: usize,
        scale: f32,
    ) {
        let a = self.edge_sources[edge];
        let b = adjacency.neighbors[edge];
        self.edge_a[sample] = a;
        self.edge_b[sample] = b;
        let delta = model.points[a as usize].truncate() - model.points[b as usize].truncate();
        let direction = delta.try_normalize().unwrap_or(glam::Vec3::ZERO);
        self.edge_direction[sample * 3] = direction.x;
        self.edge_direction[sample * 3 + 1] = direction.y;
        self.edge_direction[sample * 3 + 2] = direction.z;
        self.edge_scale[sample] = scale;
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
        LrSchedule::RadFoamV1 => radfoam_v1_lrs(t, total).density,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RadFoamV1Lrs {
    density: f32,
    position: f32,
    sh_dc: f32,
    sh_rest: f32,
}

fn radfoam_cosine_lr(
    step: usize,
    total: usize,
    initial: f32,
    final_lr: f32,
    warmup: usize,
    max_step: usize,
) -> f32 {
    if step == 0 {
        return initial;
    }
    // Upstream updates the scheduler after each optimizer step. Consequently
    // the declared initial rate is used at step zero and scheduler index zero
    // is used at step one. Preserve that one-step lag for exact comparisons.
    let scheduler_step = step - 1;
    if warmup > 0 && scheduler_step < warmup {
        return initial * scheduler_step as f32 / warmup as f32;
    }
    if scheduler_step > max_step || total == 0 || max_step <= warmup {
        return 0.0;
    }
    let phase = ((scheduler_step - warmup) as f32 / (max_step - warmup) as f32).clamp(0.0, 1.0);
    final_lr + 0.5 * (initial - final_lr) * (1.0 + (std::f32::consts::PI * phase).cos())
}

fn radfoam_v1_lrs(step: usize, total: usize) -> RadFoamV1Lrs {
    let density_warmup = total / 10;
    let sh_warmup = total / 5;
    let position_end = total.saturating_mul(9) / 10;
    RadFoamV1Lrs {
        density: radfoam_cosine_lr(step, total, 0.1, 0.01, density_warmup, total),
        position: radfoam_cosine_lr(step, total, 2.0e-4, 5.0e-6, 0, position_end),
        sh_dc: radfoam_cosine_lr(step, total, 5.0e-3, 5.0e-4, 0, total),
        // The optimizer declares the DC rate for every SH parameter before
        // the first update, then switches higher-order terms to their 0.1×
        // schedule. Match that reference quirk at step zero.
        sh_rest: if step == 0 {
            5.0e-3
        } else {
            radfoam_cosine_lr(step, total, 5.0e-4, 5.0e-5, sh_warmup, total)
        },
    }
}

fn configure_optimizer(
    session: &mut mn::Session,
    config: &AppearanceFitConfig,
    step: usize,
    total_steps: usize,
) {
    match config.lr_schedule {
        LrSchedule::Constant | LrSchedule::Cosine => {
            session.set_adam(
                lr_at_step(config, step, total_steps),
                config.adam_beta1,
                config.adam_beta2,
                config.adam_eps,
            );
            session.set_lr_multiplier("surface_normals", config.surface_normal_lr_ratio);
            session.set_lr_multiplier("surface_offsets", config.surface_offset_lr_ratio);
            session.set_lr_multiplier("surface_color_coefficients", config.surface_color_lr_ratio);
            session.set_lr_multiplier(
                "surface_detail_offsets",
                config.surface_detail_offset_lr_ratio,
            );
            session.set_lr_multiplier(
                "surface_detail_heights",
                config.surface_detail_height_lr_ratio,
            );
            session.set_lr_multiplier(
                "surface_detail_colors",
                config.surface_detail_color_lr_ratio,
            );
            session.set_lr_multiplier(
                "spherical_voronoi_axes",
                config.spherical_voronoi_axis_lr_ratio,
            );
            session.set_lr_multiplier(
                "spherical_voronoi_colors",
                config.spherical_voronoi_color_lr_ratio,
            );
        }
        LrSchedule::RadFoamV1 => {
            let rates = radfoam_v1_lrs(step, total_steps);
            session.set_adam(1.0, config.adam_beta1, config.adam_beta2, 1.0e-15);
            session.set_lr_multiplier("log_density", rates.density);
            session.set_lr_multiplier("positions", rates.position);
            set_sh_lr_multipliers(session, config.sh_degree, rates.sh_dc, rates.sh_rest);
            session.set_lr_multiplier("exposure_", 0.0);
            session.set_lr_multiplier("log_radii", 0.0);
            let surface_rate = radfoam_cosine_lr(step, total_steps, 0.1, 0.01, 0, total_steps);
            session.set_lr_multiplier(
                "surface_normals",
                surface_rate * config.surface_normal_lr_ratio,
            );
            let surface_offset_rate =
                radfoam_cosine_lr(step, total_steps, 5.0e-3, 5.0e-4, 0, total_steps);
            session.set_lr_multiplier(
                "surface_offsets",
                surface_offset_rate * config.surface_offset_lr_ratio,
            );
            session.set_lr_multiplier(
                "surface_color_coefficients",
                surface_offset_rate * config.surface_color_lr_ratio,
            );
            let surface_detail_offset_rate =
                radfoam_cosine_lr(step, total_steps, 1.0e-2, 1.0e-3, 0, total_steps);
            session.set_lr_multiplier(
                "surface_detail_offsets",
                surface_detail_offset_rate * config.surface_detail_offset_lr_ratio,
            );
            session.set_lr_multiplier(
                "surface_detail_heights",
                surface_offset_rate * config.surface_detail_height_lr_ratio,
            );
            session.set_lr_multiplier(
                "surface_detail_colors",
                surface_offset_rate * config.surface_detail_color_lr_ratio,
            );
            let spherical_axis_rate =
                radfoam_cosine_lr(step, total_steps, 5.0e-2, 5.0e-3, 0, total_steps);
            session.set_lr_multiplier(
                "spherical_voronoi_axes",
                spherical_axis_rate * config.spherical_voronoi_axis_lr_ratio,
            );
            session.set_lr_multiplier(
                "spherical_voronoi_colors",
                surface_offset_rate * config.spherical_voronoi_color_lr_ratio,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RelativeLrMultipliers {
    position: f32,
    sh_dc: f32,
    sh_rest: f32,
}

fn relative_lr_multipliers(
    groups: LrGroups,
    sh_degree: usize,
    position_lr_ratio: f32,
) -> RelativeLrMultipliers {
    match groups {
        LrGroups::Legacy => {
            let sh = if sh_degree == 0 { 1.0 } else { 0.1 };
            RelativeLrMultipliers {
                position: position_lr_ratio,
                sh_dc: sh,
                sh_rest: sh,
            }
        }
        LrGroups::RadFoamV1Relative => RelativeLrMultipliers {
            position: 2.0e-3,
            sh_dc: 5.0e-2,
            sh_rest: 5.0e-3,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TopologyCadenceState {
    period: usize,
    steps_since_update: usize,
}

impl TopologyCadenceState {
    fn radfoam_v1_initial() -> Self {
        Self {
            period: 1,
            steps_since_update: 1,
        }
    }

    fn disabled() -> Self {
        Self {
            period: 0,
            steps_since_update: 0,
        }
    }

    fn steps_until_update(self) -> usize {
        debug_assert!(self.period > 0);
        if self.steps_since_update >= self.period {
            1
        } else {
            self.period - self.steps_since_update + 1
        }
    }

    /// Advance through a cycle that cannot contain more than one scheduled
    /// update. Returns whether the final step performs that update.
    fn advance(&mut self, steps: usize) -> bool {
        let until_update = self.steps_until_update();
        assert!(steps <= until_update);
        if steps < until_update {
            self.steps_since_update += steps;
            return false;
        }

        if self.period < 100 {
            self.period += 2;
        }
        // The reference resets before incrementing the counter at the end of
        // the same optimizer iteration.
        self.steps_since_update = 1;
        true
    }

    fn reset_period_after_densification(&mut self) {
        debug_assert!(self.period > 0);
        // Official v1 preserves `iters_since_update`; only the period resets.
        self.period = 1;
    }
}

fn fixed_densify_due(config: DensifyConfig, steps_done: usize) -> bool {
    steps_done >= config.warmup && (steps_done - config.warmup).is_multiple_of(config.every)
}

fn fixed_densification_round(config: DensifyConfig, steps_done: usize) -> usize {
    debug_assert!(fixed_densify_due(config, steps_done));
    (steps_done - config.warmup) / config.every
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DensifyCadenceState {
    initial_points: usize,
    steps_since_update: usize,
    next_after: usize,
    round: usize,
}

impl DensifyCadenceState {
    fn disabled() -> Self {
        Self {
            initial_points: 0,
            steps_since_update: 0,
            next_after: 0,
            round: 0,
        }
    }

    fn radfoam_v1_initial(initial_points: usize) -> Self {
        assert!(initial_points > 0);
        Self {
            initial_points,
            steps_since_update: 0,
            next_after: 1,
            round: 0,
        }
    }

    fn steps_until_update(&self, steps_done: usize, warmup: usize) -> usize {
        debug_assert!(self.next_after > 0);
        if steps_done < warmup {
            warmup - steps_done
        } else {
            assert!(
                self.steps_since_update < self.next_after,
                "cannot resume RadFoam v1 densification from an unprocessed boundary"
            );
            self.next_after - self.steps_since_update
        }
    }

    /// Advance completed optimizer steps. RadFoam increments its counter at
    /// the end of each iteration for which `i + 1 >= densify_from`, so the
    /// warmup boundary contributes exactly one count.
    fn advance(&mut self, steps_done: usize, steps: usize, warmup: usize) -> bool {
        debug_assert!(self.next_after > 0);
        let end = steps_done + steps;
        let active_start = steps_done.max(warmup.saturating_sub(1));
        self.steps_since_update += end.saturating_sub(active_start);
        assert!(
            self.steps_since_update <= self.next_after,
            "densification cycle crossed its scheduled boundary"
        );
        self.steps_since_update == self.next_after
    }

    fn finish_round(&mut self, config: DensifyConfig, current_points: usize) {
        debug_assert_eq!(config.schedule, DensifySchedule::RadFoamV1);
        debug_assert_eq!(self.steps_since_update, self.next_after);
        self.round += 1;
        self.steps_since_update = 0;
        self.next_after =
            radfoam_v1_next_densify_after(config, self.initial_points, current_points);
    }
}

fn radfoam_v1_next_densify_after(
    config: DensifyConfig,
    initial_points: usize,
    current_points: usize,
) -> usize {
    assert!(config.densify_until > config.warmup);
    assert!(config.target_points > initial_points);
    let growth_window = config.densify_until - config.warmup;
    let point_growth = config.target_points - initial_points;
    let interval = (f64::from(config.fraction) * current_points as f64 * growth_window as f64
        / point_growth as f64) as usize;
    interval.max(100)
}

fn radfoam_v1_densify_active(config: DensifyConfig, current_points: usize) -> bool {
    (current_points as u128) * 10 < (config.target_points as u128) * 9
}

fn restore_densify_cadence(
    config: Option<DensifyConfig>,
    current_points: usize,
    resume_step: usize,
    training_state: Option<TrainingState>,
) -> DensifyCadenceState {
    match config.map(|value| value.schedule) {
        None | Some(DensifySchedule::Fixed) => DensifyCadenceState::disabled(),
        Some(DensifySchedule::RadFoamV1) => match training_state {
            Some(state) => {
                assert!(
                    state.densify_initial_points > 0
                        && state.densify_next_after > 0
                        && state.densify_steps_since_update <= state.densify_next_after,
                    "RadFoam v1 densification cadence requires a v3 trainer-state checkpoint"
                );
                DensifyCadenceState {
                    initial_points: state.densify_initial_points,
                    steps_since_update: state.densify_steps_since_update,
                    next_after: state.densify_next_after,
                    round: state.densification_round,
                }
            }
            None => {
                assert_eq!(
                    resume_step, 0,
                    "RadFoam v1 densification cadence cannot resume without trainer state"
                );
                DensifyCadenceState::radfoam_v1_initial(current_points)
            }
        },
    }
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
    resume_step: usize,
    end_step: usize,
    total_steps: usize,
    cadence: DensifyCadenceState,
) -> bool {
    end_step >= total_steps
        || match config.schedule {
            DensifySchedule::Fixed => {
                current_points >= config.target_points
                    || end_step >= config.densify_until
                    || fixed_densify_due(config, end_step)
            }
            DensifySchedule::RadFoamV1 => {
                !radfoam_v1_densify_active(config, current_points)
                    || end_step - resume_step
                        == cadence.steps_until_update(resume_step, config.warmup)
            }
        }
}

fn steps_until_fixed_densify(config: DensifyConfig, steps_done: usize) -> usize {
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

/// One supervised view: a camera plus the pixel image the trained model should
/// reproduce there. `target_rgb` is `width * height * 3` floats in row-major
/// RGB order. `target_alpha`, when present, is one float per pixel. Path
/// recording derives the containing cell from the current model, so topology
/// and radius updates cannot leave a stale seed here.
#[derive(Clone)]
pub struct ViewSupervision {
    pub camera: vol::CameraParams,
    pub target_rgb: Vec<f32>,
    pub target_alpha: Option<Vec<f32>>,
    pub width: u32,
    pub height: u32,
}

/// Fit per-cell density and SH coefficients of `model` so it reproduces every
/// view in `views`. Paths are recorded from the current model on the GPU for
/// every Adam step.
///
/// `pixel_batch = None` uses the complete `width × height` image as the batch
/// and preserves the legacy `epochs * views.len()` step count without using
/// the obsolete precomputed-path graph. `Some(K)` uses random mini-batches and
/// `steps_per_view * views.len()` steps. `views_per_batch > 1` distributes
/// each random-pixel batch across a deterministic stratified camera sample.
pub fn fit_appearance_multi_view(
    model: &mut vol::PointCloudModel,
    views: &[ViewSupervision],
    width: u32,
    height: u32,
    max_steps: usize,
    config: AppearanceFitConfig,
    gpu: std::sync::Arc<blade_graphics::Context>,
) -> Vec<f32> {
    fit_appearance_multi_view_outcome(model, views, width, height, max_steps, config, gpu).losses
}

pub(crate) struct AppearanceFitOutcome {
    pub losses: Vec<f32>,
    pub endpoint_checkpoint: Option<std::path::PathBuf>,
}

pub(crate) fn fit_appearance_multi_view_outcome(
    model: &mut vol::PointCloudModel,
    views: &[ViewSupervision],
    width: u32,
    height: u32,
    max_steps: usize,
    mut config: AppearanceFitConfig,
    gpu: std::sync::Arc<blade_graphics::Context>,
) -> AppearanceFitOutcome {
    assert!(
        !views.is_empty(),
        "fit_appearance_multi_view needs >=1 view"
    );
    for v in views {
        assert_eq!(
            v.target_rgb.len() as u32,
            v.width * v.height * 3,
            "view target_rgb length mismatches its width*height*3"
        );
        if let Some(ref target_alpha) = v.target_alpha {
            assert_eq!(
                target_alpha.len() as u32,
                v.width * v.height,
                "view target_alpha length mismatches its width*height"
            );
            assert!(
                target_alpha
                    .iter()
                    .all(|&alpha| alpha.is_finite() && (0.0..=1.0).contains(&alpha)),
                "view target_alpha values must be finite and in [0, 1]"
            );
        }
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
        config.interpenetration_weight.is_finite() && config.interpenetration_weight >= 0.0,
        "interpenetration_weight must be finite and non-negative"
    );
    assert!(
        config.interpenetration_weight == 0.0 || config.interpenetration_samples > 0,
        "interpenetration loss requires at least one sampled edge"
    );
    assert!(
        config.interpenetration_weight == 0.0 || model.radii.is_some(),
        "interpenetration loss requires a weighted cloud"
    );
    assert!(
        config.interpenetration_weight == 0.0
            || config.position_lr_ratio > 0.0
            || config.radius_lr_ratio > 0.0,
        "interpenetration loss requires trainable positions or radii"
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
        config.surface_normal_lr_ratio.is_finite() && config.surface_normal_lr_ratio >= 0.0,
        "surface_normal_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.surface_offset_lr_ratio.is_finite() && config.surface_offset_lr_ratio >= 0.0,
        "surface_offset_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.surface_color_lr_ratio.is_finite() && config.surface_color_lr_ratio >= 0.0,
        "surface_color_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.surface_detail_offset_lr_ratio.is_finite()
            && config.surface_detail_offset_lr_ratio >= 0.0,
        "surface_detail_offset_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.surface_detail_height_lr_ratio.is_finite()
            && config.surface_detail_height_lr_ratio >= 0.0,
        "surface_detail_height_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.surface_detail_color_lr_ratio.is_finite()
            && config.surface_detail_color_lr_ratio >= 0.0,
        "surface_detail_color_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.spherical_voronoi_axis_lr_ratio.is_finite()
            && config.spherical_voronoi_axis_lr_ratio >= 0.0,
        "spherical_voronoi_axis_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.spherical_voronoi_color_lr_ratio.is_finite()
            && config.spherical_voronoi_color_lr_ratio >= 0.0,
        "spherical_voronoi_color_lr_ratio must be finite and non-negative"
    );
    assert!(
        config.surface_normal_weight.is_finite() && config.surface_normal_weight >= 0.0,
        "surface_normal_weight must be finite and non-negative"
    );
    assert!(
        config.radius_lr_ratio == 0.0 || model.radii.is_some(),
        "radius optimisation requires a weighted cloud"
    );
    assert!(
        config.surface_normal_lr_ratio == 0.0 || model.surface_normals.is_some(),
        "surface-normal optimisation requires an oriented PowerFoam cloud"
    );
    assert!(
        config.surface_offset_lr_ratio == 0.0 || model.surface_offsets.is_some(),
        "surface-offset optimisation requires initialized surface offsets"
    );
    assert!(
        config.surface_color_lr_ratio == 0.0 || model.surface_color_coefficients.is_some(),
        "surface-color optimisation requires initialized surface coefficients"
    );
    assert!(
        (config.surface_detail_offset_lr_ratio == 0.0
            && config.surface_detail_height_lr_ratio == 0.0
            && config.surface_detail_color_lr_ratio == 0.0)
            || model.surface_detail.is_some(),
        "surface-detail optimisation requires initialized sites, heights, and colors"
    );
    assert!(
        (config.spherical_voronoi_axis_lr_ratio == 0.0
            && config.spherical_voronoi_color_lr_ratio == 0.0)
            || model.spherical_voronoi.is_some(),
        "Spherical Voronoi optimisation requires initialized axes and colors"
    );
    assert!(
        config.surface_normal_weight == 0.0 || model.surface_normals.is_some(),
        "surface-normal loss requires an oriented PowerFoam cloud"
    );
    let topology_requested = config.position_lr_ratio > 0.0
        || config.radius_lr_ratio > 0.0
        || config.lr_groups == LrGroups::RadFoamV1Relative
        || config.lr_schedule == LrSchedule::RadFoamV1;
    assert!(
        model.surface_detail.is_none() || !topology_requested,
        "surface-detail training currently requires frozen positions and radii because the \
         recorded support-entry query has no topology Jacobian"
    );
    let geometry_requested = topology_requested
        || config.surface_normal_lr_ratio > 0.0
        || config.surface_offset_lr_ratio > 0.0
        || config.surface_detail_offset_lr_ratio > 0.0
        || config.surface_detail_height_lr_ratio > 0.0;
    assert!(
        !geometry_requested
            || config.geometry_rebuild_schedule == GeometryRebuildSchedule::RadFoamV1
            || config.geometry_rebuild_every > 0,
        "fixed geometry optimisation requires geometry_rebuild_every > 0"
    );
    if let Some(ref densify) = config.densify {
        assert!(
            densify.schedule != DensifySchedule::Fixed || densify.every > 0,
            "fixed densify.every must be greater than zero"
        );
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
        if densify.schedule == DensifySchedule::RadFoamV1 {
            assert!(
                densify.densify_until > densify.warmup,
                "RadFoam v1 densify_until must be greater than warmup"
            );
            let initial_points = config
                .resume_training_state
                .map(|state| state.densify_initial_points)
                .unwrap_or(model.points.len());
            assert!(
                initial_points > 0 && densify.target_points > initial_points,
                "RadFoam v1 target_points must exceed the initial cloud size"
            );
        }
    }
    if config.pixel_batch.is_none() {
        config.steps_per_view = config.epochs;
    }
    fit_appearance_pixel_batched(model, views, max_steps, pixel_batch, config, gpu)
}

#[derive(Default)]
struct TrainingPhaseTimings {
    setup: std::time::Duration,
    input_prepare: std::time::Duration,
    path_submit: std::time::Duration,
    gpu_step_wait: std::time::Duration,
    gradient_readback: std::time::Duration,
    state_readback: std::time::Duration,
    contribution: std::time::Duration,
    topology: std::time::Duration,
    densify: std::time::Duration,
    resource_rebuild: std::time::Duration,
    checkpoint: std::time::Duration,
    checkpoint_download: std::time::Duration,
    checkpoint_snapshot: std::time::Duration,
    checkpoint_model: std::time::Duration,
    checkpoint_optimizer: std::time::Duration,
    checkpoint_metadata: std::time::Duration,
    finalize: std::time::Duration,
}

impl TrainingPhaseTimings {
    fn log(&self, wall: std::time::Duration, steps: usize) {
        log::info!(
            "training phase timing: wall={:.3}s steps={} setup={:.3}s \
             input={:.3}s path-submit={:.3}s gpu-step-wait={:.3}s \
             gradient-readback={:.3}s state-readback={:.3}s \
             contribution={:.3}s topology={:.3}s densify={:.3}s \
             resource-rebuild={:.3}s checkpoint={:.3}s \
             checkpoint-download={:.3}s checkpoint-snapshot={:.3}s \
             checkpoint-model={:.3}s checkpoint-optimizer={:.3}s \
             checkpoint-metadata={:.3}s finalize={:.3}s",
            wall.as_secs_f64(),
            steps,
            self.setup.as_secs_f64(),
            self.input_prepare.as_secs_f64(),
            self.path_submit.as_secs_f64(),
            self.gpu_step_wait.as_secs_f64(),
            self.gradient_readback.as_secs_f64(),
            self.state_readback.as_secs_f64(),
            self.contribution.as_secs_f64(),
            self.topology.as_secs_f64(),
            self.densify.as_secs_f64(),
            self.resource_rebuild.as_secs_f64(),
            self.checkpoint.as_secs_f64(),
            self.checkpoint_download.as_secs_f64(),
            self.checkpoint_snapshot.as_secs_f64(),
            self.checkpoint_model.as_secs_f64(),
            self.checkpoint_optimizer.as_secs_f64(),
            self.checkpoint_metadata.as_secs_f64(),
            self.finalize.as_secs_f64(),
        );
    }
}

fn dump_path_record_gpu_timings(timings: &[(String, std::time::Duration)]) {
    if timings.is_empty() {
        return;
    }
    let total: std::time::Duration = timings.iter().map(|&(_, duration)| duration).sum();
    eprintln!(
        "--- path recorder GPU timings ({} passes, {:.2}ms total) ---",
        timings.len(),
        total.as_secs_f64() * 1000.0,
    );
    for &(ref name, duration) in timings {
        eprintln!("  {name:>44}: {:>8.2}ms", duration.as_secs_f64() * 1000.0,);
    }
    eprintln!("---");
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
    }
    if let Some(ref normals) = model.surface_normals {
        let mut init_normals = Vec::with_capacity(n_cells * 3);
        for &normal in normals {
            init_normals.extend_from_slice(&normal.normalize().to_array());
        }
        session.set_parameter("surface_normals", &init_normals);
    }
    if let Some(ref offsets) = model.surface_offsets {
        session.set_parameter("surface_offsets", offsets);
    }
    if let Some(ref coefficients) = model.surface_color_coefficients {
        let components = vol::SURFACE_COLOR_COMPONENTS;
        let mut packed = vec![0.0_f32; n_cells * components * 3];
        for point in 0..n_cells {
            for basis in 0..components {
                for channel in 0..3 {
                    packed[point * components * 3 + channel * components + basis] =
                        coefficients[point * components * 3 + basis * 3 + channel];
                }
            }
        }
        session.set_parameter("surface_color_coefficients", &packed);
    }
    if let Some(ref detail) = model.surface_detail {
        let sites = vol::SURFACE_DETAIL_SITES;
        let mut offsets = Vec::with_capacity(n_cells * sites * 3);
        for &offset in &detail.offsets {
            offsets.extend_from_slice(&offset.to_array());
        }
        let mut colors = vec![0.0_f32; n_cells * sites * 3];
        for point in 0..n_cells {
            let base = point * sites * 3;
            for site in 0..sites {
                let color = detail.colors[point * sites + site];
                for channel in 0..3 {
                    colors[base + channel * sites + site] = color[channel];
                }
            }
        }
        session.set_parameter("surface_detail_offsets", &offsets);
        session.set_parameter("surface_detail_heights", &detail.heights);
        session.set_parameter("surface_detail_colors", &colors);
    }
    if let Some(ref spherical_voronoi) = model.spherical_voronoi {
        let sites = vol::SPHERICAL_VORONOI_SITES;
        let mut axes = Vec::with_capacity(n_cells * sites * 3);
        for &axis in &spherical_voronoi.axes {
            axes.extend_from_slice(&axis.to_array());
        }
        let mut colors = vec![0.0_f32; n_cells * sites * 3];
        for point in 0..n_cells {
            for site in 0..sites {
                let color = spherical_voronoi.colors[point * sites + site];
                for channel in 0..3 {
                    colors[point * sites * 3 + channel * sites + site] = color[channel];
                }
            }
        }
        session.set_parameter("spherical_voronoi_axes", &axes);
        session.set_parameter("spherical_voronoi_colors", &colors);
    }

    // `model.sh_coefficients` layout (lib.rs spec):
    //   `[p0_c0_r, p0_c0_g, p0_c0_b, p0_c1_r, p0_c1_g, p0_c1_b, ..., p1_c0_r, ...]`
    // i.e. per point, RGB interleaved within each SH component, then
    // components contiguous. The graph retains one historical `[N, 1]` DC
    // table per channel and packs all higher-order terms into `[N, K-1]`.
    let num_components = model.sh_component_count();
    let row_stride = num_components * 3;
    for (chan_idx, chan) in ["sh_r", "sh_g", "sh_b"].iter().enumerate() {
        let mut dc = vec![0.0_f32; n_cells];
        for (i, slot) in dc.iter_mut().enumerate() {
            *slot = model.sh_coefficients[i * row_stride + chan_idx];
        }
        session.set_parameter(chan, &dc);
        if num_components > 1 {
            let mut rest = vec![0.0_f32; n_cells * (num_components - 1)];
            for i in 0..n_cells {
                for k in 1..num_components {
                    rest[i * (num_components - 1) + k - 1] =
                        model.sh_coefficients[i * row_stride + k * 3 + chan_idx];
                }
            }
            session.set_parameter(&sh_rest_parameter_name(chan), &rest);
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

fn legacy_sh_tensor(
    checkpoint: &mn::data::safetensors::SafeTensorsModel,
    prefix: Option<&str>,
    channel: &str,
    num_components: usize,
    n_cells: usize,
) -> Result<Option<Vec<f32>>, String> {
    let names: Vec<String> = (1..num_components)
        .map(|component| {
            let parameter = parameter_name(channel, component);
            match prefix {
                Some(prefix) => format!("{prefix}.{parameter}"),
                None => parameter,
            }
        })
        .collect();
    let available = names
        .iter()
        .filter(|name| checkpoint.tensor_info().contains_key(*name))
        .count();
    if available == 0 {
        return Ok(None);
    }
    if available != names.len() {
        return Err(format!(
            "legacy checkpoint has {available}/{} {channel} SH tensors for prefix {prefix:?}",
            names.len()
        ));
    }

    let stride = num_components - 1;
    let mut packed = vec![0.0_f32; n_cells * stride];
    for (component, name) in names.iter().enumerate() {
        let values = checkpoint
            .tensor_f32(name)
            .map_err(|err| format!("failed to read legacy checkpoint tensor {name}: {err}"))?;
        if values.len() != n_cells {
            return Err(format!(
                "legacy checkpoint tensor {name} has {} values, expected {n_cells}",
                values.len()
            ));
        }
        for (cell, value) in values.into_iter().enumerate() {
            packed[cell * stride + component] = value;
        }
    }
    Ok(Some(packed))
}

fn checkpoint_has_tensor(path: &std::path::Path, name: &str) -> Result<bool, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut header_size_bytes = [0_u8; 8];
    std::io::Read::read_exact(&mut file, &mut header_size_bytes)
        .map_err(|err| format!("failed to read {} header size: {err}", path.display()))?;
    let header_size = usize::try_from(u64::from_le_bytes(header_size_bytes))
        .map_err(|_| format!("{} header size does not fit usize", path.display()))?;
    const MAX_HEADER_SIZE: usize = 64 * 1024 * 1024;
    if header_size > MAX_HEADER_SIZE {
        return Err(format!(
            "{} checkpoint header is {header_size} bytes (limit {MAX_HEADER_SIZE})",
            path.display()
        ));
    }
    let mut header = vec![0_u8; header_size];
    std::io::Read::read_exact(&mut file, &mut header)
        .map_err(|err| format!("failed to read {} header: {err}", path.display()))?;
    let quoted_name = format!("\"{name}\"");
    Ok(header
        .windows(quoted_name.len())
        .any(|window| window == quoted_name.as_bytes()))
}

/// Load a native checkpoint, migrating the pre-packed degree>0 SH layout.
///
/// The paired PLY has already initialized every current parameter. New
/// checkpoints load directly. A legacy checkpoint instead stores one tensor
/// per higher-order component; pack its parameter values and Adam moments into
/// the current row-major tables after Meganeura restores all matching fields.
fn load_optimizer_checkpoint(
    session: &mut mn::Session,
    path: &std::path::Path,
    sh_degree: usize,
    n_cells: usize,
) -> Result<bool, String> {
    let num_components = (1 + sh_degree) * (1 + sh_degree);
    let has_packed_sh = num_components > 1 && checkpoint_has_tensor(path, "sh_r_rest")?;
    session
        .load_checkpoint(path)
        .map_err(|err| format!("failed to load {}: {err:?}", path.display()))?;
    if num_components == 1 || has_packed_sh {
        return Ok(false);
    }

    let checkpoint = mn::data::safetensors::SafeTensorsModel::load(path.to_path_buf())
        .map_err(|err| format!("failed to inspect {}: {err}", path.display()))?;
    let mut migrated = false;
    for channel in ["sh_r", "sh_g", "sh_b"] {
        let Some(parameter) =
            legacy_sh_tensor(&checkpoint, None, channel, num_components, n_cells)?
        else {
            continue;
        };
        let rest_name = sh_rest_parameter_name(channel);
        session.set_parameter(&rest_name, &parameter);
        if let Some(moment) = legacy_sh_tensor(
            &checkpoint,
            Some("adam_m"),
            channel,
            num_components,
            n_cells,
        )? {
            session.write_adam_m(&rest_name, &moment);
        }
        if let Some(moment) = legacy_sh_tensor(
            &checkpoint,
            Some("adam_v"),
            channel,
            num_components,
            n_cells,
        )? {
            session.write_adam_v(&rest_name, &moment);
        }
        migrated = true;
    }
    Ok(migrated)
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
        "{TRAINING_STATE_HEADER}\nstep {}\ncycle {}\nsampling_rng {}\nquantile_rng {}\n\
         densify_rng {}\ntopology_period {}\ntopology_steps_since_update {}\n\
         densify_initial_points {}\ndensify_steps_since_update {}\n\
         densify_next_after {}\ndensification_round {}\n",
        state.step,
        state.cycle,
        state.sampling_rng,
        state.quantile_rng,
        state.densify_rng,
        state.topology_period,
        state.topology_steps_since_update,
        state.densify_initial_points,
        state.densify_steps_since_update,
        state.densify_next_after,
        state.densification_round,
    )
}

fn decode_training_state(text: &str) -> Result<TrainingState, String> {
    let mut lines = text.lines();
    let header = lines.next();
    let version = match header {
        Some(TRAINING_STATE_HEADER) => 3,
        Some(TOPOLOGY_TRAINING_STATE_HEADER) => 2,
        Some(LEGACY_TRAINING_STATE_HEADER) => 1,
        _ => return Err("unsupported or missing training-state header".to_string()),
    };
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
    let mut state = TrainingState {
        step: usize::try_from(step_u64)
            .map_err(|_| format!("training-state step {step_u64} does not fit usize"))?,
        cycle: usize::try_from(read_value("cycle")?)
            .map_err(|_| "training-state cycle does not fit usize".to_string())?,
        sampling_rng: read_value("sampling_rng")?,
        quantile_rng: read_value("quantile_rng")?,
        densify_rng: read_value("densify_rng")?,
        topology_period: 0,
        topology_steps_since_update: 0,
        densify_initial_points: 0,
        densify_steps_since_update: 0,
        densify_next_after: 0,
        densification_round: 0,
    };
    if version >= 2 {
        state.topology_period = usize::try_from(read_value("topology_period")?)
            .map_err(|_| "training-state topology_period does not fit usize".to_string())?;
        state.topology_steps_since_update =
            usize::try_from(read_value("topology_steps_since_update")?).map_err(|_| {
                "training-state topology_steps_since_update does not fit usize".to_string()
            })?;
        let fixed = state.topology_period == 0 && state.topology_steps_since_update == 0;
        let radfoam = state.topology_period > 0
            && state.topology_period <= 101
            && state.topology_period % 2 == 1
            && state.topology_steps_since_update > 0;
        if !fixed && !radfoam {
            return Err("invalid training-state topology cadence".to_string());
        }
    }
    if version >= 3 {
        state.densify_initial_points = usize::try_from(read_value("densify_initial_points")?)
            .map_err(|_| "training-state densify_initial_points does not fit usize".to_string())?;
        state.densify_steps_since_update =
            usize::try_from(read_value("densify_steps_since_update")?).map_err(|_| {
                "training-state densify_steps_since_update does not fit usize".to_string()
            })?;
        state.densify_next_after = usize::try_from(read_value("densify_next_after")?)
            .map_err(|_| "training-state densify_next_after does not fit usize".to_string())?;
        state.densification_round = usize::try_from(read_value("densification_round")?)
            .map_err(|_| "training-state densification_round does not fit usize".to_string())?;
        let fixed = state.densify_initial_points == 0
            && state.densify_steps_since_update == 0
            && state.densify_next_after == 0
            && state.densification_round == 0;
        let radfoam = state.densify_initial_points > 0
            && state.densify_next_after > 0
            && state.densify_steps_since_update <= state.densify_next_after;
        if !fixed && !radfoam {
            return Err("invalid training-state densification cadence".to_string());
        }
    }
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

type ParameterReadback = std::iter::Zip<std::vec::IntoIter<String>, std::vec::IntoIter<Vec<f32>>>;

fn append_surface_plane_parameter_names(model: &vol::PointCloudModel, names: &mut Vec<String>) {
    if model.surface_normals.is_some() {
        names.push("surface_normals".to_string());
    }
    if model.surface_offsets.is_some() {
        names.push("surface_offsets".to_string());
    }
    if model.surface_detail.is_some() {
        names.push("surface_detail_offsets".to_string());
        names.push("surface_detail_heights".to_string());
    }
}

fn append_geometry_parameter_names(model: &vol::PointCloudModel, names: &mut Vec<String>) {
    names.push("positions".to_string());
    if model.radii.is_some() {
        names.push("log_radii".to_string());
    }
    append_surface_plane_parameter_names(model, names);
}

fn read_model_parameters(session: &mn::Session, names: Vec<String>) -> ParameterReadback {
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let values = session.read_params(&name_refs);
    names.into_iter().zip(values)
}

fn next_parameter(readback: &mut ParameterReadback, expected: &str) -> Vec<f32> {
    let (name, values) = readback
        .next()
        .unwrap_or_else(|| panic!("missing parameter readback for {expected}"));
    assert_eq!(name, expected);
    values
}

fn apply_model_surface_planes(model: &mut vol::PointCloudModel, readback: &mut ParameterReadback) {
    if let Some(ref mut normals) = model.surface_normals {
        let out_normals = next_parameter(readback, "surface_normals");
        for (normal, values) in normals.iter_mut().zip(out_normals.chunks_exact(3)) {
            *normal = glam::Vec3::from_slice(values).normalize_or_zero();
            if *normal == glam::Vec3::ZERO {
                *normal = glam::Vec3::Z;
            }
        }
    }
    if let Some(ref mut offsets) = model.surface_offsets {
        *offsets = next_parameter(readback, "surface_offsets");
    }
    if let Some(ref mut detail) = model.surface_detail {
        let offsets = next_parameter(readback, "surface_detail_offsets");
        detail.heights = next_parameter(readback, "surface_detail_heights");
        for (offset, values) in detail.offsets.iter_mut().zip(offsets.chunks_exact(3)) {
            *offset = glam::Vec3::from_slice(values);
        }
    }
}

fn apply_model_geometry(model: &mut vol::PointCloudModel, readback: &mut ParameterReadback) {
    let n_cells = model.points.len();
    let out_positions = next_parameter(readback, "positions");
    for i in 0..n_cells {
        model.points[i].x = out_positions[i * 3];
        model.points[i].y = out_positions[i * 3 + 1];
        model.points[i].z = out_positions[i * 3 + 2];
    }

    if let Some(ref mut radii) = model.radii {
        let out_radii = next_parameter(readback, "log_radii");
        for (radius, &raw) in radii.iter_mut().zip(out_radii.iter()) {
            *radius = radius_activation(raw);
        }
    }
    apply_model_surface_planes(model, readback);
}

fn download_model_surface_planes(session: &mn::Session, model: &mut vol::PointCloudModel) {
    let mut names = Vec::new();
    append_surface_plane_parameter_names(model, &mut names);
    let mut readback = read_model_parameters(session, names);
    apply_model_surface_planes(model, &mut readback);
    assert!(readback.next().is_none());
}

fn download_model_geometry(session: &mn::Session, model: &mut vol::PointCloudModel) {
    let mut names = Vec::new();
    append_geometry_parameter_names(model, &mut names);
    let mut readback = read_model_parameters(session, names);
    apply_model_geometry(model, &mut readback);
    assert!(readback.next().is_none());
}

fn download_model_parameters(
    session: &mn::Session,
    model: &mut vol::PointCloudModel,
    softplus_beta: f32,
) {
    let n_cells = model.points.len();
    let num_components = model.sh_component_count();
    let mut names = Vec::new();
    append_geometry_parameter_names(model, &mut names);
    names.push("log_density".to_string());
    for chan in ["sh_r", "sh_g", "sh_b"] {
        names.push(chan.to_string());
        if num_components > 1 {
            names.push(sh_rest_parameter_name(chan));
        }
    }
    if model.surface_color_coefficients.is_some() {
        names.push("surface_color_coefficients".to_string());
    }
    if model.surface_detail.is_some() {
        names.push("surface_detail_colors".to_string());
    }
    if model.spherical_voronoi.is_some() {
        names.push("spherical_voronoi_axes".to_string());
        names.push("spherical_voronoi_colors".to_string());
    }

    let mut readback = read_model_parameters(session, names);
    apply_model_geometry(model, &mut readback);

    let out_density = next_parameter(&mut readback, "log_density");
    for (i, d) in out_density.iter().enumerate() {
        model.points[i].w = density_activation(*d, softplus_beta);
    }

    let row_stride = num_components * 3;
    for (chan_idx, chan) in ["sh_r", "sh_g", "sh_b"].iter().enumerate() {
        let dc = next_parameter(&mut readback, chan);
        for (i, &value) in dc.iter().enumerate() {
            model.sh_coefficients[i * row_stride + chan_idx] = value;
        }
        if num_components > 1 {
            let rest_name = sh_rest_parameter_name(chan);
            let rest = next_parameter(&mut readback, &rest_name);
            for i in 0..n_cells {
                for k in 1..num_components {
                    model.sh_coefficients[i * row_stride + k * 3 + chan_idx] =
                        rest[i * (num_components - 1) + k - 1];
                }
            }
        }
    }
    if let Some(ref mut coefficients) = model.surface_color_coefficients {
        let components = vol::SURFACE_COLOR_COMPONENTS;
        let packed = next_parameter(&mut readback, "surface_color_coefficients");
        for point in 0..n_cells {
            for basis in 0..components {
                for channel in 0..3 {
                    coefficients[point * components * 3 + basis * 3 + channel] =
                        packed[point * components * 3 + channel * components + basis];
                }
            }
        }
    }
    if let Some(ref mut detail) = model.surface_detail {
        let sites = vol::SURFACE_DETAIL_SITES;
        let colors = next_parameter(&mut readback, "surface_detail_colors");
        for point in 0..n_cells {
            let base = point * sites * 3;
            for site in 0..sites {
                detail.colors[point * sites + site] = glam::Vec3::new(
                    colors[base + site],
                    colors[base + sites + site],
                    colors[base + 2 * sites + site],
                );
            }
        }
    }
    if let Some(ref mut spherical_voronoi) = model.spherical_voronoi {
        let sites = vol::SPHERICAL_VORONOI_SITES;
        let axes = next_parameter(&mut readback, "spherical_voronoi_axes");
        let colors = next_parameter(&mut readback, "spherical_voronoi_colors");
        for point in 0..n_cells {
            for site in 0..sites {
                let base = point * sites * 3;
                spherical_voronoi.axes[point * sites + site] =
                    glam::Vec3::from_slice(&axes[base + site * 3..base + site * 3 + 3]);
                spherical_voronoi.colors[point * sites + site] = glam::Vec3::new(
                    colors[base + site],
                    colors[base + sites + site],
                    colors[base + 2 * sites + site],
                );
            }
        }
    }
    assert!(readback.next().is_none());
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

fn finite_quantile(values: impl Iterator<Item = f32>, quantile: f32) -> f32 {
    debug_assert!((0.0..=1.0).contains(&quantile));
    let mut sorted = values
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(f32::total_cmp);
    let position = quantile * (sorted.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f32;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
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

fn contribution_view_indices(view_count: usize, limit: usize, round: usize) -> Vec<usize> {
    if limit == 0 || limit >= view_count {
        return (0..view_count).collect();
    }
    let offset = round.wrapping_mul(0x9E37_79B9) % view_count;
    (0..limit)
        .map(|slot| (slot * view_count / limit + offset) % view_count)
        .collect()
}

/// Measure the same per-cell contribution signal used by RadFoam pruning:
/// sum volumetric ray weights within each view, then retain the maximum over
/// views. Each view uses one deterministic 2× downsample phase keyed by the
/// absolute densification round, matching the reference collector without
/// burdening every Adam mini-batch with a readback. The readback is bounded to
/// 4096 rays (16 MiB per device/shared path set at 256 steps) and reused across
/// views.
fn collect_path_contributions(
    context: &blade_graphics::Context,
    recorder: &vol::gpu::PathRecorder,
    gpu_cloud: &vol::gpu::RadFoamGpuCloud,
    model: &vol::PointCloudModel,
    views: &[ViewSupervision],
    max_steps: usize,
    powerfoam_candidate_capacity: u32,
    round: usize,
    view_limit: usize,
) -> PathContributionStats {
    const MAX_RAYS_PER_BATCH: usize = 4096;

    let view_indices = contribution_view_indices(views.len(), view_limit, round);
    let max_sampled_pixels = view_indices
        .iter()
        .map(|&index| {
            let view = &views[index];
            view.width.div_ceil(2) as usize * view.height.div_ceil(2) as usize
        })
        .max()
        .unwrap_or(1);
    let max_image_resolution = [
        view_indices
            .iter()
            .map(|&index| views[index].width)
            .max()
            .unwrap_or(1),
        view_indices
            .iter()
            .map(|&index| views[index].height)
            .max()
            .unwrap_or(1),
    ];
    let capacity = max_sampled_pixels.clamp(1, MAX_RAYS_PER_BATCH);
    let mut buffers = if model.radii.is_some() {
        vol::gpu::PathRecordBuffers::new_projected(
            context,
            capacity as u32,
            max_steps as u32,
            model.points.len() as u32,
            max_image_resolution,
            powerfoam_candidate_capacity,
        )
    } else {
        vol::gpu::PathRecordBuffers::new_recorded_only(context, capacity as u32, max_steps as u32)
    };
    let pl_capacity = capacity as u64 * max_steps as u64;
    let readback_size = pl_capacity * std::mem::size_of::<u32>() as u64;
    // Contribution scoring scans every value on the CPU, so use cached
    // download memory rather than the write-combined shared mapping.
    let cells_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-cells-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Download,
    });
    let next_cells_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-next-cells-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Download,
    });
    let dts_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-dts-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Download,
    });
    let mask_readback = context.create_buffer(blade_graphics::BufferDesc {
        name: "contribution-mask-readback",
        size: readback_size,
        memory: blade_graphics::Memory::Download,
    });
    let mut encoder = context.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "collect-path-contributions",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut stats = PathContributionStats {
        per_cell: vec![0.0; model.points.len()],
        rays: 0,
        segments: 0,
        truncated_rays: 0,
        max_steps_used: 0,
    };

    for &view_index in &view_indices {
        let view = &views[view_index];
        let phase = round
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
                if model.radii.is_none() {
                    transfer.fill_buffer(buffers.cells.at(0), path_bytes, 0);
                    transfer.fill_buffer(buffers.next_cells.at(0), path_bytes, 0);
                    transfer.fill_buffer(buffers.mask.at(0), path_bytes, 0);
                }
                transfer.fill_buffer(buffers.dts.at(0), path_bytes, 0);
                if buffers.has_jacobians() && model.radii.is_none() {
                    transfer.fill_buffer(buffers.previous_cells.at(0), path_bytes, 0);
                }
                if buffers.has_jacobians() {
                    transfer.fill_buffer(buffers.dt_reference_tangents.at(0), path_bytes, 0);
                    transfer.fill_buffer(buffers.dt_grad_previous.at(0), path_bytes * 4, 0);
                    transfer.fill_buffer(buffers.dt_grad_current.at(0), path_bytes * 4, 0);
                    transfer.fill_buffer(buffers.dt_grad_next.at(0), path_bytes * 4, 0);
                    if model.surface_normals.is_some() {
                        transfer.fill_buffer(
                            buffers.dt_grad_surface_normal.at(0),
                            path_bytes * 4,
                            0,
                        );
                    }
                }
            }
            recorder.dispatch(
                &mut encoder,
                gpu_cloud,
                &buffers,
                vol::gpu::RecordPathsArgs {
                    camera: view.camera,
                    start_point,
                    pixel_offset: 0,
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
            let recorded_path = buffers.path_stats(0..num_pixels);
            if model.radii.is_some() {
                let observed = buffers.max_splat_candidate_count(0..num_pixels);
                assert!(
                    observed <= buffers.splat_candidate_capacity(),
                    "PowerFoam contribution path needs {observed} candidates for one ray, but \
                     scratch capacity is {}",
                    buffers.splat_candidate_capacity(),
                );
            }
            let cells =
                unsafe { std::slice::from_raw_parts(cells_readback.data() as *const u32, pl) };
            let next_cells =
                unsafe { std::slice::from_raw_parts(next_cells_readback.data() as *const u32, pl) };
            let dts = unsafe { std::slice::from_raw_parts(dts_readback.data() as *const f32, pl) };
            let mask =
                unsafe { std::slice::from_raw_parts(mask_readback.data() as *const f32, pl) };
            let previous_truncated_rays = stats.truncated_rays;
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
            stats.truncated_rays = previous_truncated_rays + recorded_path.truncated_rays;
            stats.max_steps_used = stats
                .max_steps_used
                .max(recorded_path.max_steps_used as usize);
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
///    drawn by weighted multinomial (without replacement). Unweighted
///    RadFoam uses `position_gradient × cell_radius`; weighted PowerFoam uses
///    its per-site photometric-error EMA directly. Unweighted children sit
///    0.25× toward the parent's farthest neighbour plus a small random kick.
///    Both weighted duplicate siblings move by 0.05× the copied support
///    radius; oriented siblings move tangent to the inherited normal,
///    matching PowerFoam.
///    Density, appearance, radius, normal, and optimizer ancestry are inherited.
///
/// Returns `(new_to_old, pruned, added)`: `new_to_old[j]` is the OLD cell
/// index whose Adam (m,v) the rebuilt cell `j` should inherit (survivor →
/// itself, child → parent), used to carry optimiser momentum across the
/// session rebuild.
fn prune_and_densify(
    model: &mut vol::PointCloudModel,
    densify_score: &[f32],
    contribution: &[f32],
    cfg: &DensifyConfig,
    rng_state: &mut u64,
    softplus_beta: f32,
) -> (Vec<usize>, usize, usize) {
    let n_old = model.points.len();
    assert_eq!(densify_score.len(), n_old);
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
    let point_error_cap = weighted
        .then(|| finite_quantile(survivors.iter().map(|&index| densify_score[index]), 0.99));

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
                let w = if weighted {
                    densify_score[oi].min(point_error_cap.unwrap())
                } else {
                    densify_score[oi] * cell_radius[oi]
                }
                .max(1e-12);
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
    let split_offset = |oi: usize, next_unit: &mut dyn FnMut() -> f32| {
        let random = random_unit(next_unit);
        let direction = match model.surface_normals.as_ref() {
            Some(normals) => {
                let normal = normals[oi].normalize();
                let tangent = random - random.dot(normal) * normal;
                tangent
                    .try_normalize()
                    .unwrap_or_else(|| normal.any_orthonormal_vector())
            }
            None => random,
        };
        direction * (0.05 * cell_radius[oi].max(1.0e-5))
    };

    // --- Rebuild model arrays: survivors compacted, then children ---
    let n_new = n_surv + added;
    let mut new_points = Vec::with_capacity(n_new);
    let mut new_sh = Vec::with_capacity(n_new * sh_block);
    let mut new_radii = model.radii.as_ref().map(|_| Vec::with_capacity(n_new));
    let mut new_surface_normals = model
        .surface_normals
        .as_ref()
        .map(|_| Vec::with_capacity(n_new));
    let mut new_surface_offsets = model
        .surface_offsets
        .as_ref()
        .map(|_| Vec::with_capacity(n_new));
    let surface_color_block = vol::SURFACE_COLOR_COMPONENTS * 3;
    let mut new_surface_color = model
        .surface_color_coefficients
        .as_ref()
        .map(|_| Vec::with_capacity(n_new * surface_color_block));
    let mut new_surface_detail = model.surface_detail.as_ref().map(|_| vol::SurfaceDetail {
        offsets: Vec::with_capacity(n_new * vol::SURFACE_DETAIL_SITES),
        heights: Vec::with_capacity(n_new * vol::SURFACE_DETAIL_SITES),
        colors: Vec::with_capacity(n_new * vol::SURFACE_DETAIL_SITES),
    });
    let mut new_spherical_voronoi =
        model
            .spherical_voronoi
            .as_ref()
            .map(|_| vol::SphericalVoronoi {
                axes: Vec::with_capacity(n_new * vol::SPHERICAL_VORONOI_SITES),
                colors: Vec::with_capacity(n_new * vol::SPHERICAL_VORONOI_SITES),
            });
    let mut new_transforms = model.transforms.as_ref().map(|_| vol::Transforms {
        rotations: Vec::with_capacity(n_new),
        scales: Vec::with_capacity(n_new),
    });
    let mut new_to_old = Vec::with_capacity(n_new);
    for &oi in &survivors {
        let mut point = model.points[oi];
        if weighted && is_split_parent[oi] {
            let offset = split_offset(oi, &mut next_unit);
            point.x += offset.x;
            point.y += offset.y;
            point.z += offset.z;
        }
        new_points.push(point);
        new_sh.extend_from_slice(&model.sh_coefficients[oi * sh_block..(oi + 1) * sh_block]);
        if let Some(ref mut radii) = new_radii {
            radii.push(model.radii.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut normals) = new_surface_normals {
            normals.push(model.surface_normals.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut offsets) = new_surface_offsets {
            offsets.push(model.surface_offsets.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut coefficients) = new_surface_color {
            let old = model.surface_color_coefficients.as_ref().unwrap();
            coefficients
                .extend_from_slice(&old[oi * surface_color_block..(oi + 1) * surface_color_block]);
        }
        if let Some(ref mut detail) = new_surface_detail {
            let old = model.surface_detail.as_ref().unwrap();
            let begin = oi * vol::SURFACE_DETAIL_SITES;
            let end = begin + vol::SURFACE_DETAIL_SITES;
            detail.offsets.extend_from_slice(&old.offsets[begin..end]);
            detail.heights.extend_from_slice(&old.heights[begin..end]);
            detail.colors.extend_from_slice(&old.colors[begin..end]);
        }
        if let Some(ref mut spherical_voronoi) = new_spherical_voronoi {
            let old = model.spherical_voronoi.as_ref().unwrap();
            let begin = oi * vol::SPHERICAL_VORONOI_SITES;
            let end = begin + vol::SPHERICAL_VORONOI_SITES;
            spherical_voronoi
                .axes
                .extend_from_slice(&old.axes[begin..end]);
            spherical_voronoi
                .colors
                .extend_from_slice(&old.colors[begin..end]);
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
            split_offset(oi, &mut next_unit)
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
        if let Some(ref mut normals) = new_surface_normals {
            normals.push(model.surface_normals.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut offsets) = new_surface_offsets {
            offsets.push(model.surface_offsets.as_ref().unwrap()[oi]);
        }
        if let Some(ref mut coefficients) = new_surface_color {
            let old = model.surface_color_coefficients.as_ref().unwrap();
            coefficients
                .extend_from_slice(&old[oi * surface_color_block..(oi + 1) * surface_color_block]);
        }
        if let Some(ref mut detail) = new_surface_detail {
            let old = model.surface_detail.as_ref().unwrap();
            let begin = oi * vol::SURFACE_DETAIL_SITES;
            let end = begin + vol::SURFACE_DETAIL_SITES;
            detail.offsets.extend_from_slice(&old.offsets[begin..end]);
            detail.heights.extend_from_slice(&old.heights[begin..end]);
            detail.colors.extend_from_slice(&old.colors[begin..end]);
        }
        if let Some(ref mut spherical_voronoi) = new_spherical_voronoi {
            let old = model.spherical_voronoi.as_ref().unwrap();
            let begin = oi * vol::SPHERICAL_VORONOI_SITES;
            let end = begin + vol::SPHERICAL_VORONOI_SITES;
            spherical_voronoi
                .axes
                .extend_from_slice(&old.axes[begin..end]);
            spherical_voronoi
                .colors
                .extend_from_slice(&old.colors[begin..end]);
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
    model.surface_normals = new_surface_normals;
    model.surface_offsets = new_surface_offsets;
    model.surface_color_coefficients = new_surface_color;
    model.surface_detail = new_surface_detail;
    model.spherical_voronoi = new_spherical_voronoi;
    model.transforms = new_transforms;
    model.adjacency = None;
    (new_to_old, pruned, added)
}

/// Enumerate every per-cell parameter name with its per-cell element
/// stride. Stride is 1 for scalar tables (`log_density`, `log_radii`,
/// `surface_offsets`, `sh_<chan>_<k>`) and 3 for vector tables (`positions`,
/// `surface_normals`). Spatial surface color uses stride 12; spatial-detail
/// offsets/colors use stride 24 and heights use stride 8; both Spherical
/// Voronoi tables use stride 24.
fn per_cell_param_names_with_stride(
    sh_degree: usize,
    has_radii: bool,
    has_surface_normals: bool,
    has_surface_offsets: bool,
    has_surface_color: bool,
    has_surface_detail: bool,
    has_spherical_voronoi: bool,
    has_point_error: bool,
) -> Vec<(String, usize)> {
    let num_components = (1 + sh_degree) * (1 + sh_degree);
    let mut names = vec![("log_density".to_string(), 1), ("positions".to_string(), 3)];
    if has_radii {
        names.push(("log_radii".to_string(), 1));
    }
    if has_surface_normals {
        names.push(("surface_normals".to_string(), 3));
    }
    if has_surface_offsets {
        names.push(("surface_offsets".to_string(), 1));
    }
    if has_surface_color {
        names.push((
            "surface_color_coefficients".to_string(),
            vol::SURFACE_COLOR_COMPONENTS * 3,
        ));
    }
    if has_surface_detail {
        names.push((
            "surface_detail_offsets".to_string(),
            vol::SURFACE_DETAIL_SITES * 3,
        ));
        names.push((
            "surface_detail_heights".to_string(),
            vol::SURFACE_DETAIL_SITES,
        ));
        names.push((
            "surface_detail_colors".to_string(),
            vol::SURFACE_DETAIL_SITES * 3,
        ));
    }
    if has_spherical_voronoi {
        names.push((
            "spherical_voronoi_axes".to_string(),
            vol::SPHERICAL_VORONOI_SITES * 3,
        ));
        names.push((
            "spherical_voronoi_colors".to_string(),
            vol::SPHERICAL_VORONOI_SITES * 3,
        ));
    }
    if has_point_error {
        names.push((POINT_ERROR_PROBE.to_string(), 1));
    }
    for chan in ["sh_r", "sh_g", "sh_b"] {
        names.push((chan.to_string(), 1));
        if num_components > 1 {
            names.push((sh_rest_parameter_name(chan), num_components - 1));
        }
    }
    names
}

/// Snapshot of Adam optimizer state for every per-cell parameter at
/// the current cell count. Used to carry momentum across a session
/// rebuild when densification grows the cell table.
struct AdamSnapshot {
    /// Per-parameter entries in compile order. Each holds the param
    /// name, its per-cell stride, and flat (m, v) buffers of length
    /// `n_cells * stride`.
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
    /// Number of contiguous values belonging to one cloud site.
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
    has_surface_normals: bool,
    has_surface_offsets: bool,
    has_surface_color: bool,
    has_surface_detail: bool,
    has_spherical_voronoi: bool,
    has_point_error: bool,
) -> AdamSnapshot {
    let names = per_cell_param_names_with_stride(
        sh_degree,
        has_radii,
        has_surface_normals,
        has_surface_offsets,
        has_surface_color,
        has_surface_detail,
        has_spherical_voronoi,
        has_point_error,
    );
    let name_refs = names
        .iter()
        .map(|entry| entry.0.as_str())
        .collect::<Vec<_>>();
    let states = session.read_adam_states(&name_refs);
    let mut entries = Vec::with_capacity(names.len());
    for ((name, stride), (m, v)) in names.into_iter().zip(states) {
        assert_eq!(m.len(), n_cells * stride);
        assert_eq!(v.len(), n_cells * stride);
        entries.push(AdamEntry { name, stride, m, v });
    }
    let exposure_names = ["exposure_r", "exposure_g", "exposure_b"];
    let exposure_parameters = session.read_params(&exposure_names);
    let exposure_states = session.read_adam_states(&exposure_names);
    let exposure_entries = exposure_names
        .into_iter()
        .zip(exposure_parameters)
        .zip(exposure_states)
        .map(|((name, parameter), (m, v))| {
            assert_eq!(parameter.len(), num_views);
            assert_eq!(m.len(), num_views);
            assert_eq!(v.len(), num_views);
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
    has_surface_normals: bool,
    has_surface_offsets: bool,
    has_surface_color: bool,
    has_surface_detail: bool,
    has_spherical_voronoi: bool,
    has_point_error: bool,
) {
    let n_new = new_to_old.len();
    let names = per_cell_param_names_with_stride(
        sh_degree,
        has_radii,
        has_surface_normals,
        has_surface_offsets,
        has_surface_color,
        has_surface_detail,
        has_spherical_voronoi,
        has_point_error,
    );
    debug_assert_eq!(names.len(), snap.entries.len());
    let n_old = snap
        .entries
        .first()
        .map(|entry| entry.m.len() / entry.stride)
        .unwrap_or(0);
    let mut inheritance_count = vec![0usize; n_old];
    for &old_index in new_to_old {
        inheritance_count[old_index] += 1;
    }
    for (i, name_and_stride) in names.iter().enumerate() {
        let entry = &snap.entries[i];
        debug_assert_eq!(entry.stride, name_and_stride.1);
        let s = name_and_stride.1;
        let mut m = vec![0.0_f32; n_new * s];
        let mut v = vec![0.0_f32; n_new * s];
        for (j, &oi) in new_to_old.iter().enumerate() {
            m[j * s..j * s + s].copy_from_slice(&entry.m[oi * s..oi * s + s]);
            v[j * s..j * s + s].copy_from_slice(&entry.v[oi * s..oi * s + s]);
            if entry.name == POINT_ERROR_PROBE {
                debug_assert_eq!(s, 1);
                let count = inheritance_count[oi] as f32;
                m[j] /= count;
                v[j] /= count * count;
            }
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
        #[cfg(feature = "qhull")]
        {
            model.adjacency = Some(vol::compute_adjacency_qhull_default(&model.points));
        }
        #[cfg(not(feature = "qhull"))]
        panic!("Qhull adjacency requires blade-volume-train feature `qhull`");
    }
}

fn build_training_gpu_cloud(
    model: &vol::PointCloudModel,
    gpu: &std::sync::Arc<blade_graphics::Context>,
) -> vol::RadFoamGpuCloud {
    let mut init_encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "path-record-init",
        buffer_count: 1,
        manual_barriers: false,
    });
    let cloud = vol::RadFoamGpuCloud::new_path_recording(model, gpu, &mut init_encoder);
    gpu.destroy_command_encoder(&mut init_encoder);
    cloud
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
    collect_point_error: bool,
    sh_degree: usize,
    color_loss: ColorLoss,
    num_views: usize,
    patch_size: usize,
    grad_loss_weight: f32,
    opacity_weight: f32,
    distortion_weight: f32,
    quantile_weight: f32,
    interpenetration_samples: usize,
    softplus_beta: f32,
    background_rgb: [f32; 3],
    gpu: &std::sync::Arc<blade_graphics::Context>,
    path_bufs: &vol::gpu::PathRecordBuffers,
    lr: f32,
    lr_groups: LrGroups,
    position_lr_ratio: f32,
    radius_lr_ratio: f32,
    surface_normal_lr_ratio: f32,
    use_surface_normal_loss: bool,
    train_positions: bool,
    train_radii: bool,
    surface_offset_lr_ratio: f32,
    surface_color_lr_ratio: f32,
    surface_detail_offset_lr_ratio: f32,
    surface_detail_height_lr_ratio: f32,
    surface_detail_color_lr_ratio: f32,
    spherical_voronoi_axis_lr_ratio: f32,
    spherical_voronoi_color_lr_ratio: f32,
    betas: (f32, f32, f32),
) -> (mn::Session, vol::RadFoamGpuCloud) {
    let n_cells = model.points.len();
    let mut g = mn::Graph::new();
    let mut vg = build_volumetric_graph_with_options(
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
        model.surface_normals.is_some(),
        VolumetricGraphOptions {
            use_surface_normal_loss,
            train_positions,
            train_radii,
            use_surface_detail: model.surface_detail.is_some(),
        },
        model.surface_offsets.is_some(),
        model.surface_color_coefficients.is_some(),
        collect_point_error,
        model.spherical_voronoi.is_some(),
        color_loss,
    );
    if interpenetration_samples > 0 {
        let log_radii = vg
            .weighted_path
            .as_ref()
            .expect("interpenetration loss requires a weighted cloud")
            .log_radii;
        let positions = if train_positions {
            vg.positions
        } else {
            g.stop_gradient(vg.positions)
        };
        let log_radii = if train_radii {
            log_radii
        } else {
            g.stop_gradient(log_radii)
        };
        vg.loss = add_interpenetration_loss(
            &mut g,
            vg.loss,
            positions,
            log_radii,
            interpenetration_samples,
        );
        g.set_outputs(vec![vg.loss, vg.dt_from_positions]);
    }
    let (mut session, _report) = mn::build(
        &g,
        mn::SessionConfig {
            mode: mn::Mode::Training,
            gpu: Some(gpu.clone()),
            ..Default::default()
        },
    );
    if std::env::var_os("BLADE_VOLUME_PROFILE_GPU").is_some() {
        session.set_profiling(true);
    }
    if model.surface_detail.is_some() {
        for name in [
            "surface_detail_offsets",
            "surface_detail_heights",
            "surface_detail_colors",
        ] {
            assert!(
                session.has_param_grad(name),
                "surface-detail parameter {name} has no training gradient"
            );
        }
    }
    upload_model_parameters(&mut session, model, softplus_beta);

    let gpu_cloud = build_training_gpu_cloud(model, gpu);

    // Unweighted interior dt is reconstructed from positions, with recorded
    // terminal intervals. Weighted dt uses the recorder's raw interval plus
    // the change in its local position/radius tangent.
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
        let expected_jacobians = if train_positions || train_radii {
            vol::gpu::PathJacobianMode::Full
        } else if model.surface_normals.is_some() {
            vol::gpu::PathJacobianMode::Surface
        } else {
            vol::gpu::PathJacobianMode::None
        };
        assert_eq!(
            path_bufs.jacobian_mode(),
            expected_jacobians,
            "weighted graph and path-buffer Jacobian modes disagree"
        );
        if path_bufs.has_jacobians() {
            let source = gpu
                .get_external_buffer_source(path_bufs.dt_reference_tangents)
                .expect("weighted path buffer must be exportable");
            session
                .bind_external_buffer(
                    meganeura::ExternalSlot::Input("dt_reference_tangent"),
                    source,
                    pl_bytes,
                )
                .unwrap_or_else(|err| {
                    panic!("bind_external_buffer(dt_reference_tangent) failed: {err:?}")
                });
        }
        if path_bufs.has_geometry_jacobians() {
            for (slot, buf, size) in [
                ("previous_cell_indices", path_bufs.previous_cells, pl_bytes),
                ("dt_grad_previous", path_bufs.dt_grad_previous, pl_bytes * 4),
                ("dt_grad_current", path_bufs.dt_grad_current, pl_bytes * 4),
                ("dt_grad_next", path_bufs.dt_grad_next, pl_bytes * 4),
            ] {
                let source = gpu
                    .get_external_buffer_source(buf)
                    .expect("weighted geometry path buffer must be exportable");
                session
                    .bind_external_buffer(meganeura::ExternalSlot::Input(slot), source, size)
                    .unwrap_or_else(|err| panic!("bind_external_buffer({slot}) failed: {err:?}"));
            }
        }
        if path_bufs.has_surface_jacobians() {
            let source = gpu
                .get_external_buffer_source(path_bufs.dt_grad_surface_normal)
                .expect("surface-normal path buffer must be exportable");
            session
                .bind_external_buffer(
                    meganeura::ExternalSlot::Input("dt_grad_surface_normal"),
                    source,
                    pl_bytes * 4,
                )
                .unwrap_or_else(|err| {
                    panic!("bind_external_buffer(dt_grad_surface_normal) failed: {err:?}")
                });
        }
        if model.surface_detail.is_some() {
            assert!(
                path_bufs.has_surface_queries(),
                "surface-detail graph requires recorded query inputs"
            );
            let source = gpu
                .get_external_buffer_source(path_bufs.surface_queries)
                .expect("surface-query path buffer must be exportable");
            session
                .bind_external_buffer(
                    meganeura::ExternalSlot::Input("surface_queries"),
                    source,
                    pl_bytes * 2,
                )
                .unwrap_or_else(|err| {
                    panic!("bind_external_buffer(surface_queries) failed: {err:?}")
                });
        }
    }

    session.set_adam(lr, betas.0, betas.1, betas.2);
    let multipliers = relative_lr_multipliers(lr_groups, sh_degree, position_lr_ratio);
    set_sh_lr_multipliers(
        &mut session,
        sh_degree,
        multipliers.sh_dc,
        multipliers.sh_rest,
    );
    session.set_lr_multiplier("positions", multipliers.position);
    if model.radii.is_some() {
        session.set_lr_multiplier("log_radii", radius_lr_ratio);
    }
    if model.surface_normals.is_some() {
        session.set_lr_multiplier("surface_normals", surface_normal_lr_ratio);
    }
    if model.surface_offsets.is_some() {
        session.set_lr_multiplier("surface_offsets", surface_offset_lr_ratio);
    }
    if model.surface_color_coefficients.is_some() {
        session.set_lr_multiplier("surface_color_coefficients", surface_color_lr_ratio);
    }
    if model.surface_detail.is_some() {
        session.set_lr_multiplier("surface_detail_offsets", surface_detail_offset_lr_ratio);
        session.set_lr_multiplier("surface_detail_heights", surface_detail_height_lr_ratio);
        session.set_lr_multiplier("surface_detail_colors", surface_detail_color_lr_ratio);
    }
    if model.spherical_voronoi.is_some() {
        session.set_lr_multiplier("spherical_voronoi_axes", spherical_voronoi_axis_lr_ratio);
        session.set_lr_multiplier("spherical_voronoi_colors", spherical_voronoi_color_lr_ratio);
    }
    if collect_point_error {
        session.set_parameter(POINT_ERROR_PROBE, &vec![0.0; n_cells]);
        session.set_lr_multiplier(POINT_ERROR_PROBE, 0.0);
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

/// Pixel-batched training mode. Each Adam step picks a deterministic
/// stratified set of training views, distributes `pixel_batch` random pixels
/// across them, records paths through the current model, and runs one
/// optimiser step. Allows training at full image resolution without exceeding
/// the meganeura matmul shape limits.
fn fit_appearance_pixel_batched(
    model: &mut vol::PointCloudModel,
    views: &[ViewSupervision],
    max_steps: usize,
    pixel_batch: usize,
    config: AppearanceFitConfig,
    gpu: std::sync::Arc<blade_graphics::Context>,
) -> AppearanceFitOutcome {
    let wall_start = std::time::Instant::now();
    let mut phase_timings = TrainingPhaseTimings::default();
    let n_cells = model.points.len();
    let profile_gpu = std::env::var_os("BLADE_VOLUME_PROFILE_GPU").is_some();
    if profile_gpu && std::env::var_os("MEGANEURA_GPU_TIMING").is_none() {
        log::warn!("BLADE_VOLUME_PROFILE_GPU requires MEGANEURA_GPU_TIMING=1 for GPU timestamps");
    }
    let total_steps = config.steps_per_view.max(1) * views.len();
    let full_image_batch = config.pixel_batch.is_none();
    assert!(
        config.views_per_batch > 0,
        "views_per_batch must be non-zero"
    );
    if full_image_batch || config.patch_size > 0 {
        assert_eq!(
            config.views_per_batch, 1,
            "mixed-view batches require random-pixel sampling"
        );
    }
    let views_per_batch = config.views_per_batch.min(views.len()).min(pixel_batch);
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
    let densify = config.densify;
    let mut densify_cadence = restore_densify_cadence(
        densify,
        model.points.len(),
        config.resume_step,
        config.resume_training_state,
    );
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
                    config.resume_step,
                    invocation_end,
                    total_steps,
                    densify_cadence,
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
    // cycle-independent. `PathRecordBuffers`, however, holds exportable
    // `Memory::External(Fd(None))` buffers; meganeura's
    // `bind_external_buffer` consumes those FDs on import (Vulkan
    // takes ownership), so once a session has imported them the
    // producer's `buffer.external` field is stale. We therefore
    // recreate `path_bufs` alongside the session+cloud each densify
    // cycle. They occupy at most 36 MiB at the current 2,048 × 256 weighted
    // training shape.
    let recorder = vol::gpu::PathRecorder::new(&gpu);
    let pl_bytes = (pixel_batch as u64) * (max_steps as u64) * 4;
    let mut record_encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "path-record-step",
        buffer_count: 2,
        manual_barriers: false,
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
        (views_per_batch as u64).saturating_add(pixel_batch as u64)
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
    let mut target_alpha_buf = vec![1.0f32; pixel_batch];
    let mut pixel_indices = vec![0u32; pixel_batch];
    let mut quantile_near = vec![0.0_f32; pixel_batch];
    let mut quantile_far = vec![0.0_f32; pixel_batch];
    let mut basis_inputs: Vec<Vec<f32>> = (1..num_components)
        .map(|_| vec![0.0_f32; pixel_batch])
        .collect();
    // Position-opt graph inputs:
    //   - ray_origin ([P,3]): per-pixel camera position
    //   - ray_dir_per_pixel ([P, 3]): per-pixel ray direction
    //   - pixel_idx_per_step ([P*L]): constant gather index for
    //     broadcasting per-pixel data across the L step dimension
    let mut ray_origin_buf = vec![0.0_f32; pixel_batch * 3];
    let mut ray_dir_per_pixel_buf = vec![0.0_f32; pixel_batch * 3];
    let mut view_idx_buf = vec![0_u32; pixel_batch];
    let mut record_args = Vec::with_capacity(views_per_batch);
    let pixel_idx_per_step: Vec<u32> = (0..(pixel_batch * max_steps))
        .map(|i| (i / max_steps) as u32)
        .collect();

    let mut losses = Vec::with_capacity(invocation_end.saturating_sub(steps_done));
    let mut endpoint_checkpoint = None;
    let mut training_path_rays = 0usize;
    let mut training_path_truncated_rays = 0usize;
    let mut training_path_max_steps_used = 0u32;
    let mut training_candidate_max_used = 0u32;
    // Frequent loss readouts (every ~2000 steps) so long multi-hour runs
    // surface their trajectory instead of only ~10 lines total.
    let log_every = 2000.min(total_steps).max(1);

    // Densification splits cells between training cycles. The
    // session + GPU cloud + path-record buffers are kept alive across
    // cycle boundaries when no split happens, so Adam momentum is
    // preserved through the warmup → first-densify transition. They're
    // only torn down and rebuilt when a split actually changes the cell
    // count.
    let topology_trainable = config.position_lr_ratio > 0.0
        || config.radius_lr_ratio > 0.0
        || config.lr_groups == LrGroups::RadFoamV1Relative
        || config.lr_schedule == LrSchedule::RadFoamV1;
    let geometry_trainable = topology_trainable
        || config.surface_normal_lr_ratio > 0.0
        || config.surface_offset_lr_ratio > 0.0
        || config.surface_detail_offset_lr_ratio > 0.0
        || config.surface_detail_height_lr_ratio > 0.0;
    let mut topology_cadence = match config.geometry_rebuild_schedule {
        GeometryRebuildSchedule::Fixed => TopologyCadenceState::disabled(),
        GeometryRebuildSchedule::RadFoamV1 => match config.resume_training_state {
            Some(state) => {
                assert!(
                    state.topology_period > 0 && state.topology_steps_since_update > 0,
                    "RadFoam v1 geometry cadence requires a v2 trainer-state checkpoint"
                );
                TopologyCadenceState {
                    period: state.topology_period,
                    steps_since_update: state.topology_steps_since_update,
                }
            }
            None => {
                assert_eq!(
                    steps_done, 0,
                    "RadFoam v1 geometry cadence cannot resume without trainer state"
                );
                TopologyCadenceState::radfoam_v1_initial()
            }
        },
    };
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
    let setup_start = std::time::Instant::now();
    let mut position_grad_accum = vec![0.0f32; model.points.len()];
    let mut position_grad_scratch = vec![0.0f32; model.points.len() * 3];
    let collect_powerfoam_point_error = densify.is_some() && model.radii.is_some();
    // Densification rebuilds the session and remaps every geometry moment, so
    // retain the established full-gradient graph while it is enabled. At a
    // fixed topology, a zero-rate parameter is truly frozen: no downstream
    // decision consumes its gradient or Adam moments. Frozen checkpoints omit
    // those moments; enabling geometry later intentionally starts them at zero.
    let train_positions = densify.is_some()
        || config.position_lr_ratio > 0.0
        || config.lr_groups == LrGroups::RadFoamV1Relative
        || config.lr_schedule == LrSchedule::RadFoamV1;
    let train_radii = densify.is_some()
        || (config.radius_lr_ratio > 0.0 && config.lr_schedule != LrSchedule::RadFoamV1);
    let path_jacobian_mode = if train_positions || train_radii {
        vol::gpu::PathJacobianMode::Full
    } else if model.surface_normals.is_some() {
        vol::gpu::PathJacobianMode::Surface
    } else {
        vol::gpu::PathJacobianMode::None
    };
    let _ = n_cells;
    // Building a complete per-camera screen index does not pay for sparse
    // mixed-view batches. Keep their already parallel exhaustive gather, but
    // use projected candidates for large single-camera batches.
    let use_projected_candidates = model.radii.is_some()
        && pixel_batch.div_ceil(views_per_batch) >= MIN_PROJECTED_RAYS_PER_CAMERA;
    if model.radii.is_some() {
        log::info!(
            "PowerFoam path candidates: {} (up to {} rays per camera)",
            if use_projected_candidates {
                "projected tiles"
            } else {
                "exhaustive gather"
            },
            pixel_batch.div_ceil(views_per_batch),
        );
    }
    let max_image_resolution = [
        views.iter().map(|view| view.width).max().unwrap_or(1),
        views.iter().map(|view| view.height).max().unwrap_or(1),
    ];
    let mut path_bufs = if use_projected_candidates {
        vol::gpu::PathRecordBuffers::new_external_powerfoam_projected(
            &gpu,
            pixel_batch as u32,
            max_steps as u32,
            path_jacobian_mode,
            model.points.len() as u32,
            max_image_resolution,
            config.powerfoam_candidate_capacity,
            model.surface_detail.is_some(),
        )
    } else if model.radii.is_some() {
        vol::gpu::PathRecordBuffers::new_external_powerfoam(
            &gpu,
            pixel_batch as u32,
            max_steps as u32,
            path_jacobian_mode,
            config.powerfoam_candidate_capacity,
            model.surface_detail.is_some(),
        )
    } else {
        vol::gpu::PathRecordBuffers::new_external_with_jacobians(
            &gpu,
            pixel_batch as u32,
            max_steps as u32,
            false,
            config.powerfoam_candidate_capacity,
        )
    };
    let patch_size = config.patch_size;
    let grad_loss_weight = config.grad_loss_weight;
    let interpenetration_sample_count = if config.interpenetration_weight > 0.0 {
        config.interpenetration_samples
    } else {
        0
    };
    let mut interpenetration_batch = (interpenetration_sample_count > 0)
        .then(|| InterpenetrationBatch::new(model, interpenetration_sample_count));
    let (mut session, mut gpu_cloud) = build_train_session(
        model,
        pixel_batch,
        max_steps,
        collect_powerfoam_point_error,
        sh_degree,
        config.color_loss,
        views.len(),
        patch_size,
        grad_loss_weight,
        config.opacity_weight,
        config.distortion_weight,
        config.quantile_weight,
        interpenetration_sample_count,
        config.softplus_beta,
        config.background_rgb,
        &gpu,
        &path_bufs,
        config.learning_rate,
        config.lr_groups,
        config.position_lr_ratio,
        config.radius_lr_ratio,
        config.surface_normal_lr_ratio,
        config.surface_normal_weight > 0.0,
        train_positions,
        train_radii,
        config.surface_offset_lr_ratio,
        config.surface_color_lr_ratio,
        config.surface_detail_offset_lr_ratio,
        config.surface_detail_height_lr_ratio,
        config.surface_detail_color_lr_ratio,
        config.spherical_voronoi_axis_lr_ratio,
        config.spherical_voronoi_color_lr_ratio,
        (config.adam_beta1, config.adam_beta2, config.adam_eps),
    );
    if let Some(ref state_path) = config.resume_state_path {
        let migrated_legacy_sh = load_optimizer_checkpoint(
            &mut session,
            state_path,
            config.sh_degree,
            model.points.len(),
        )
        .unwrap_or_else(|err| panic!("{err}"));
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
        if migrated_legacy_sh {
            log::info!("resume: migrated legacy per-component SH parameters and Adam moments");
        }
    }

    // `pixel_idx_per_step` is constant across all training steps —
    // upload once. The new session built for each densify cycle gets
    // its own upload below.
    session.set_input_u32("pixel_idx_per_step", &pixel_idx_per_step);
    // The checkpoint PLY may have exposure baked into it while the restored
    // safetensors sidecar carries the exact live parameters. Track whether
    // `model` reflects the current session so endpoint checkpoint/finalization
    // does not repeat the same full parameter readback.
    let mut model_parameters_current = false;
    // Inactive rows are masked and use zeroed gather indices. Initialize their
    // remaining payload once, then retain the previous finite payload instead
    // of clearing tens of MiB of dt/Jacobian storage before every dispatch.
    // `padded_weighted_steps_have_zero_loss_and_parameter_gradient` covers
    // non-zero masked payload for every trainable weighted-path table.
    let mut path_payload_initialized = false;
    phase_timings.setup += setup_start.elapsed();

    while steps_done < invocation_end {
        let densify_schedule_active = densify.is_some_and(|d| match d.schedule {
            DensifySchedule::Fixed => {
                steps_done < d.densify_until && model.points.len() < d.target_points
            }
            DensifySchedule::RadFoamV1 => radfoam_v1_densify_active(d, model.points.len()),
        });
        let densify_budget = match densify.filter(|_| densify_schedule_active) {
            Some(d) => match d.schedule {
                DensifySchedule::Fixed => {
                    steps_until_fixed_densify(d, steps_done).min(invocation_end - steps_done)
                }
                DensifySchedule::RadFoamV1 => densify_cadence
                    .steps_until_update(steps_done, d.warmup)
                    .min(invocation_end - steps_done),
            },
            None => invocation_end - steps_done,
        };
        let geometry_budget = if geometry_trainable {
            match config.geometry_rebuild_schedule {
                GeometryRebuildSchedule::Fixed => {
                    let every = config.geometry_rebuild_every;
                    (every - steps_done % every).min(invocation_end - steps_done)
                }
                GeometryRebuildSchedule::RadFoamV1 => topology_cadence
                    .steps_until_update()
                    .min(invocation_end - steps_done),
            }
        } else {
            invocation_end - steps_done
        };
        let cycle_budget = densify_budget.min(geometry_budget);

        let cycle_start = steps_done;
        for cycle_step in 0..cycle_budget {
            let input_start = std::time::Instant::now();
            let step = steps_done + cycle_step;
            let selected_views =
                sample_training_views(&mut sampling_rng, views.len(), views_per_batch);
            for (slot, &vi) in selected_views.iter().enumerate() {
                let range = batch_view_range(slot, views_per_batch, pixel_batch);
                let origin = ray_constants(&views[vi].camera).origin;
                for k in range {
                    ray_origin_buf[k * 3] = origin.x;
                    ray_origin_buf[k * 3 + 1] = origin.y;
                    ray_origin_buf[k * 3 + 2] = origin.z;
                    view_idx_buf[k] = vi as u32;
                }
            }

            // Two sampling modes:
            //   - random pixels: split the batch across the selected views,
            //     then pick independent random pixels within each view.
            //   - patch: pick a random `q × q` patch corner and emit
            //     pixel indices in row-major order across the patch, so
            //     the graph can treat them as a 2D image for gradient
            //     L1.
            if patch_size > 0 {
                let v = &views[selected_views[0]];
                let cam_ray_constants = ray_constants(&v.camera);
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
                        target_alpha_buf[k] = v
                            .target_alpha
                            .as_ref()
                            .map_or(1.0, |target| target[pidx as usize]);
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
                let v = &views[selected_views[0]];
                let cam_ray_constants = ray_constants(&v.camera);
                let img_size = v.width * v.height;
                assert_eq!(pixel_batch, img_size as usize);
                for (k, pidx) in (0..img_size).enumerate() {
                    pixel_indices[k] = pidx;
                    let base = (pidx as usize) * 3;
                    target_buf[k * 3] = v.target_rgb[base];
                    target_buf[k * 3 + 1] = v.target_rgb[base + 1];
                    target_buf[k * 3 + 2] = v.target_rgb[base + 2];
                    target_alpha_buf[k] = v
                        .target_alpha
                        .as_ref()
                        .map_or(1.0, |target| target[pidx as usize]);

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
                for (slot, &vi) in selected_views.iter().enumerate() {
                    let v = &views[vi];
                    let cam_ray_constants = ray_constants(&v.camera);
                    let img_size = v.width * v.height;
                    for k in batch_view_range(slot, views_per_batch, pixel_batch) {
                        let pidx = next_lcg_u32(&mut sampling_rng) % img_size;
                        pixel_indices[k] = pidx;
                        let base = (pidx as usize) * 3;
                        target_buf[k * 3] = v.target_rgb[base];
                        target_buf[k * 3 + 1] = v.target_rgb[base + 1];
                        target_buf[k * 3 + 2] = v.target_rgb[base + 2];
                        target_alpha_buf[k] = v
                            .target_alpha
                            .as_ref()
                            .map_or(1.0, |target| target[pidx as usize]);

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
            }
            path_bufs.write_pixel_indices(&pixel_indices);
            session.set_input("ray_origin", &ray_origin_buf);
            session.set_input("ray_dir_per_pixel", &ray_dir_per_pixel_buf);
            for (j, vec) in basis_inputs.iter().enumerate() {
                session.set_input(&format!("basis_{}", j + 1), vec);
            }
            phase_timings.input_prepare += input_start.elapsed();

            let path_submit_start = std::time::Instant::now();
            record_encoder.start();
            if profile_gpu {
                dump_path_record_gpu_timings(record_encoder.timings());
            }
            {
                let mut tx = record_encoder.transfer("path-record-prepare");
                let pix_size = (pixel_batch * std::mem::size_of::<u32>()) as u64;
                tx.copy_buffer_to_buffer(
                    path_bufs.pixel_indices_stage.at(0),
                    path_bufs.pixel_indices.at(0),
                    pix_size,
                );
                if model.radii.is_none() {
                    tx.fill_buffer(path_bufs.cells.at(0), pl_bytes, 0);
                    tx.fill_buffer(path_bufs.next_cells.at(0), pl_bytes, 0);
                    tx.fill_buffer(path_bufs.mask.at(0), pl_bytes, 0);
                    if path_bufs.has_geometry_jacobians() {
                        tx.fill_buffer(path_bufs.previous_cells.at(0), pl_bytes, 0);
                    }
                }
                if !path_payload_initialized {
                    tx.fill_buffer(path_bufs.dts.at(0), pl_bytes, 0);
                }
                if path_bufs.has_jacobians() && !path_payload_initialized {
                    tx.fill_buffer(path_bufs.dt_reference_tangents.at(0), pl_bytes, 0);
                }
                if path_bufs.has_geometry_jacobians() && !path_payload_initialized {
                    tx.fill_buffer(path_bufs.dt_grad_previous.at(0), pl_bytes * 4, 0);
                    tx.fill_buffer(path_bufs.dt_grad_current.at(0), pl_bytes * 4, 0);
                    tx.fill_buffer(path_bufs.dt_grad_next.at(0), pl_bytes * 4, 0);
                }
                if path_bufs.has_surface_jacobians() && !path_payload_initialized {
                    tx.fill_buffer(path_bufs.dt_grad_surface_normal.at(0), pl_bytes * 4, 0);
                }
                if path_bufs.has_surface_queries() && !path_payload_initialized {
                    tx.fill_buffer(path_bufs.surface_queries.at(0), pl_bytes * 2, 0);
                }
            }
            record_args.clear();
            for (slot, &vi) in selected_views.iter().enumerate() {
                let v = &views[vi];
                let range = batch_view_range(slot, views_per_batch, pixel_batch);
                record_args.push(vol::gpu::RecordPathsArgs {
                    camera: v.camera,
                    start_point: gpu_cloud
                        .containing_point(glam::Vec3::from_array(v.camera.cam_position)),
                    pixel_offset: range.start as u32,
                    max_steps: max_steps as u32,
                    image_width: v.width,
                    image_height: v.height,
                    max_path_dt: MAX_PATH_DT,
                    depth: v.camera.depth,
                    num_pixels: range.len() as u32,
                });
            }
            recorder.dispatch_batch(&mut record_encoder, &gpu_cloud, &path_bufs, &record_args);
            let _ = gpu.submit(&mut record_encoder);
            path_payload_initialized = true;
            phase_timings.path_submit += path_submit_start.elapsed();

            let gpu_step_start = std::time::Instant::now();
            session.set_input("labels", &target_buf);
            if config.opacity_weight > 0.0 {
                session.set_input("target_alpha", &target_alpha_buf);
            }
            session.set_input_u32("view_idx", &view_idx_buf);
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
            if let Some(ref mut batch) = interpenetration_batch {
                batch.upload(
                    &mut session,
                    model,
                    step,
                    total_steps,
                    config.interpenetration_weight,
                );
            }
            if config.surface_normal_weight > 0.0 {
                let normal_weight =
                    surface_normal_weight_at_step(config.surface_normal_weight, step, total_steps);
                session.set_input("surface_normal_loss_scale", &[normal_weight]);
            }
            // Reconfigure Adam and any parameter-specific multipliers for the
            // current global step. This is a cheap session-field update.
            configure_optimizer(&mut session, &config, step, total_steps);
            session.step();
            session.wait();
            let recorded_path = path_bufs.path_stats(0..pixel_batch);
            let first_path_truncation =
                training_path_truncated_rays == 0 && recorded_path.truncated_rays > 0;
            training_path_rays += pixel_batch;
            training_path_truncated_rays += recorded_path.truncated_rays;
            training_path_max_steps_used =
                training_path_max_steps_used.max(recorded_path.max_steps_used);
            if first_path_truncation {
                log::warn!(
                    "training paths first truncated at step {}: {} / {} rays reached \
                     max_steps={}",
                    step + 1,
                    recorded_path.truncated_rays,
                    pixel_batch,
                    max_steps,
                );
            }
            if model.radii.is_some() {
                let observed = path_bufs.max_splat_candidate_count(0..pixel_batch);
                training_candidate_max_used = training_candidate_max_used.max(observed);
                assert!(
                    observed <= path_bufs.splat_candidate_capacity(),
                    "PowerFoam training path needs {observed} candidates for one ray at step {}, \
                     but scratch capacity is {}",
                    step + 1,
                    path_bufs.splat_candidate_capacity(),
                );
            }
            model_parameters_current = false;
            if profile_gpu {
                session.dump_gpu_timings();
            }
            let loss = session.read_output(1).first().copied().unwrap_or(f32::NAN);
            losses.push(loss);
            phase_timings.gpu_step_wait += gpu_step_start.elapsed();

            let gradient_readback_start = std::time::Instant::now();
            if densify_schedule_active
                && !collect_powerfoam_point_error
                && session.has_param_grad("positions")
            {
                session.read_param_grad("positions", &mut position_grad_scratch);
                for (i, accumulated) in position_grad_accum.iter_mut().enumerate() {
                    let base = i * 3;
                    let x = position_grad_scratch[base];
                    let y = position_grad_scratch[base + 1];
                    let z = position_grad_scratch[base + 2];
                    *accumulated += (x * x + y * y + z * z).sqrt();
                }
            }
            phase_timings.gradient_readback += gradient_readback_start.elapsed();

            if step == 0 || (step + 1).is_multiple_of(log_every) {
                let window: usize = log_every.min(losses.len());
                let recent_avg: f32 =
                    losses.iter().rev().take(window).copied().sum::<f32>() / window as f32;
                log::info!(
                    "step {}/{}: avg loss {:.4} (window {}) cells={} paths=max {}/{}, \
                     truncated={}",
                    step + 1,
                    total_steps,
                    recent_avg,
                    window,
                    model.points.len(),
                    recorded_path.max_steps_used,
                    max_steps,
                    recorded_path.truncated_rays,
                );
            }
        }

        steps_done += cycle_budget;
        cycle += 1;
        let densify_schedule_due = densify_schedule_active
            && densify.is_some_and(|d| match d.schedule {
                DensifySchedule::Fixed => fixed_densify_due(d, steps_done),
                DensifySchedule::RadFoamV1 => {
                    densify_cadence.advance(cycle_start, cycle_budget, d.warmup)
                }
            });
        let topology_schedule_due = geometry_trainable
            && match config.geometry_rebuild_schedule {
                GeometryRebuildSchedule::Fixed => {
                    steps_done.is_multiple_of(config.geometry_rebuild_every)
                }
                GeometryRebuildSchedule::RadFoamV1 => topology_cadence.advance(cycle_budget),
            };

        // Prune + densify at the selected schedule boundary. The fixed mode
        // uses `densify_until` and the exact point budget. Reference mode
        // uses its cell-count-derived interval and 90%-of-target stop gate.
        let mut topology_rebuilt = false;
        if let Some(d) = densify {
            let gate = densify_schedule_due && steps_done < total_steps;
            if gate {
                let densification_round = match d.schedule {
                    DensifySchedule::Fixed => fixed_densification_round(d, steps_done),
                    DensifySchedule::RadFoamV1 => densify_cadence.round,
                };
                // Snapshot params + Adam state at the OLD size, before
                // prune+densify remaps `model.points`.
                let n_old = model.points.len();
                let state_readback_start = std::time::Instant::now();
                download_model_parameters(&session, model, config.softplus_beta);
                model_parameters_current = true;
                let adam_snap = save_adam_state(
                    &session,
                    sh_degree,
                    n_old,
                    views.len(),
                    model.radii.is_some(),
                    model.surface_normals.is_some(),
                    model.surface_offsets.is_some(),
                    model.surface_color_coefficients.is_some(),
                    model.surface_detail.is_some(),
                    model.spherical_voronoi.is_some(),
                    collect_powerfoam_point_error,
                );
                phase_timings.state_readback += state_readback_start.elapsed();

                // Refresh trainable geometry before collecting densification
                // statistics. Moving positions/radii also require exact
                // adjacency; changing only orientation does not.
                if topology_schedule_due {
                    if topology_trainable {
                        let topology_start = std::time::Instant::now();
                        rebuild_training_adjacency(model, config.rebuild_with_qhull);
                        phase_timings.topology += topology_start.elapsed();
                    }
                    if d.prune {
                        let resource_rebuild_start = std::time::Instant::now();
                        if topology_trainable {
                            gpu_cloud.deinit(&gpu);
                            gpu_cloud = build_training_gpu_cloud(model, &gpu);
                        } else {
                            gpu_cloud.update_surface_geometry(
                                model.surface_normals.as_deref().unwrap(),
                                model.surface_offsets.as_deref(),
                                model.surface_detail.as_ref(),
                                &gpu,
                                &mut record_encoder,
                            );
                        }
                        phase_timings.resource_rebuild += resource_rebuild_start.elapsed();
                    }
                    log::info!(
                        "densify round {}: refreshed current geometry before resampling",
                        densification_round,
                    );
                }

                let contribution = if d.prune {
                    let contribution_start = std::time::Instant::now();
                    let stats = collect_path_contributions(
                        &gpu,
                        &recorder,
                        &gpu_cloud,
                        model,
                        views,
                        max_steps,
                        config.powerfoam_candidate_capacity,
                        densification_round,
                        d.contribution_views,
                    );
                    phase_timings.contribution += contribution_start.elapsed();
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
                        "contribution round {}: {} of {} views, {} rays, \
                         {:.1} mean segments, max {}, \
                         {} truncated ({:.3}%)",
                        densification_round,
                        contribution_view_indices(
                            views.len(),
                            d.contribution_views,
                            densification_round,
                        )
                        .len(),
                        views.len(),
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
                let densify_score = if collect_powerfoam_point_error {
                    adam_snap
                        .entries
                        .iter()
                        .find(|entry| entry.name == POINT_ERROR_PROBE)
                        .map(|entry| entry.m.as_slice())
                        .expect("weighted densification requires point-error state")
                } else {
                    position_grad_accum.as_slice()
                };
                let densify_start = std::time::Instant::now();
                let (new_to_old, pruned, added) = prune_and_densify(
                    model,
                    densify_score,
                    &contribution,
                    &d,
                    &mut rng_split,
                    config.softplus_beta,
                );
                phase_timings.densify += densify_start.elapsed();
                log::info!(
                    "densify round {}: {} cells (-{} pruned, +{} split) → {} total",
                    densification_round,
                    n_old,
                    pruned,
                    added,
                    model.points.len(),
                );
                if d.schedule == DensifySchedule::RadFoamV1 {
                    densify_cadence.finish_round(d, model.points.len());
                    log::info!(
                        "densify round {}: next RadFoam v1 growth after {} steps",
                        densification_round,
                        densify_cadence.next_after,
                    );
                }
                // Rebuild Voronoi adjacency for the new cell set so the
                // GPU path-record sees real neighbours of the new cells.
                let topology_start = std::time::Instant::now();
                rebuild_training_adjacency(model, config.rebuild_with_qhull);
                if let Some(ref mut batch) = interpenetration_batch {
                    batch.rebuild(model);
                }
                phase_timings.topology += topology_start.elapsed();
                position_grad_accum = vec![0.0f32; model.points.len()];
                position_grad_scratch = vec![0.0f32; model.points.len() * 3];

                // Topology changed: tear down and rebuild the
                // cell-count-dependent resources, then remap Adam moments
                // from survivors and split parents into the new session.
                let resource_rebuild_start = std::time::Instant::now();
                drop(session);
                gpu_cloud.deinit(&gpu);
                path_bufs.destroy(&gpu);
                path_bufs = if use_projected_candidates {
                    vol::gpu::PathRecordBuffers::new_external_powerfoam_projected(
                        &gpu,
                        pixel_batch as u32,
                        max_steps as u32,
                        path_jacobian_mode,
                        model.points.len() as u32,
                        max_image_resolution,
                        config.powerfoam_candidate_capacity,
                        model.surface_detail.is_some(),
                    )
                } else if model.radii.is_some() {
                    vol::gpu::PathRecordBuffers::new_external_powerfoam(
                        &gpu,
                        pixel_batch as u32,
                        max_steps as u32,
                        path_jacobian_mode,
                        config.powerfoam_candidate_capacity,
                        model.surface_detail.is_some(),
                    )
                } else {
                    vol::gpu::PathRecordBuffers::new_external_with_jacobians(
                        &gpu,
                        pixel_batch as u32,
                        max_steps as u32,
                        false,
                        config.powerfoam_candidate_capacity,
                    )
                };
                let rebuilt = build_train_session(
                    model,
                    pixel_batch,
                    max_steps,
                    collect_powerfoam_point_error,
                    sh_degree,
                    config.color_loss,
                    views.len(),
                    patch_size,
                    grad_loss_weight,
                    config.opacity_weight,
                    config.distortion_weight,
                    config.quantile_weight,
                    interpenetration_sample_count,
                    config.softplus_beta,
                    config.background_rgb,
                    &gpu,
                    &path_bufs,
                    config.learning_rate,
                    config.lr_groups,
                    config.position_lr_ratio,
                    config.radius_lr_ratio,
                    config.surface_normal_lr_ratio,
                    config.surface_normal_weight > 0.0,
                    train_positions,
                    train_radii,
                    config.surface_offset_lr_ratio,
                    config.surface_color_lr_ratio,
                    config.surface_detail_offset_lr_ratio,
                    config.surface_detail_height_lr_ratio,
                    config.surface_detail_color_lr_ratio,
                    config.spherical_voronoi_axis_lr_ratio,
                    config.spherical_voronoi_color_lr_ratio,
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
                    model.surface_normals.is_some(),
                    model.surface_offsets.is_some(),
                    model.surface_color_coefficients.is_some(),
                    model.surface_detail.is_some(),
                    model.spherical_voronoi.is_some(),
                    collect_powerfoam_point_error,
                );
                session.set_adam_step_count(adam_snap.t);
                phase_timings.resource_rebuild += resource_rebuild_start.elapsed();
                topology_rebuilt = true;
                if config.geometry_rebuild_schedule == GeometryRebuildSchedule::RadFoamV1 {
                    topology_cadence.reset_period_after_densification();
                }
            }
        }

        // Interval Jacobians and the discrete cell walk are local to the
        // current geometry snapshot. At the configured cadence, download the
        // trainable geometry and recreate traversal resources. Moving
        // positions/radii additionally requires exact adjacency; changing
        // only orientation does not. The final cycle only needs the host-side
        // refresh because no further optimizer step will run.
        let geometry_due =
            geometry_trainable && (topology_schedule_due || steps_done == invocation_end);
        if geometry_due && !topology_rebuilt {
            let n_cells = model.points.len();
            let continuing = steps_done < invocation_end;
            let state_readback_start = std::time::Instant::now();
            if continuing {
                // The Meganeura graph and its external path buffers do not
                // depend on adjacency. Only the path recorder needs current
                // traversal geometry for the next discrete walk, so keep the
                // live session and Adam state in place instead of copying all
                // appearance parameters and moments through uncached shared
                // memory just to recreate an identical graph. When topology
                // is frozen, positions and radii are already current on the
                // host, so read back only the moving surface-plane tables.
                if topology_trainable {
                    download_model_geometry(&session, model);
                } else {
                    download_model_surface_planes(&session, model);
                }
            } else {
                download_model_parameters(&session, model, config.softplus_beta);
                model_parameters_current = true;
            }
            phase_timings.state_readback += state_readback_start.elapsed();
            if topology_trainable {
                let topology_start = std::time::Instant::now();
                rebuild_training_adjacency(model, config.rebuild_with_qhull);
                if let Some(ref mut batch) = interpenetration_batch {
                    batch.rebuild(model);
                }
                phase_timings.topology += topology_start.elapsed();
                log::info!(
                    "geometry cycle {}: rebuilt adjacency for {} moved points at step {}",
                    cycle,
                    n_cells,
                    steps_done,
                );
            } else {
                log::info!(
                    "geometry cycle {}: refreshed surface planes for {} points at step {}",
                    cycle,
                    n_cells,
                    steps_done,
                );
            }

            if continuing {
                let resource_rebuild_start = std::time::Instant::now();
                if topology_trainable {
                    gpu_cloud.deinit(&gpu);
                    gpu_cloud = build_training_gpu_cloud(model, &gpu);
                } else {
                    gpu_cloud.update_surface_geometry(
                        model.surface_normals.as_deref().unwrap(),
                        model.surface_offsets.as_deref(),
                        model.surface_detail.as_ref(),
                        &gpu,
                        &mut record_encoder,
                    );
                }
                phase_timings.resource_rebuild += resource_rebuild_start.elapsed();
            }
        }

        // Write checkpoints after topology maintenance so every PLY pairs
        // its point positions with matching adjacency. Exposure is baked
        // into a throwaway clone; the exact sidecar retains the live values.
        let checkpoint_due = steps_done == invocation_end || densify_schedule_due;
        if checkpoint_due {
            if let Some(ref ckpt) = config.checkpoint_path {
                let checkpoint_start = std::time::Instant::now();
                let download_start = std::time::Instant::now();
                if !model_parameters_current {
                    download_model_parameters(&session, model, config.softplus_beta);
                    model_parameters_current = true;
                }
                phase_timings.checkpoint_download += download_start.elapsed();
                let snapshot_start = std::time::Instant::now();
                let mut snapshot = model.clone();
                bake_mean_exposure_into_sh(&session, &mut snapshot, views.len());
                phase_timings.checkpoint_snapshot += snapshot_start.elapsed();
                let model_start = std::time::Instant::now();
                let model_checkpoint = save_checkpoint(ckpt, &snapshot);
                phase_timings.checkpoint_model += model_start.elapsed();
                if model_checkpoint.is_ok() && steps_done == invocation_end {
                    endpoint_checkpoint = Some(ckpt.clone());
                }
                let optimizer_start = std::time::Instant::now();
                match model_checkpoint.and_then(|()| save_optimizer_checkpoint(&mut session, ckpt))
                {
                    Ok(optimizer_path) => {
                        phase_timings.checkpoint_optimizer += optimizer_start.elapsed();
                        let metadata_start = std::time::Instant::now();
                        if let Err(err) = save_checkpoint_step(ckpt, steps_done) {
                            log::warn!("checkpoint step-sidecar write failed: {err}");
                        }
                        let trainer_state = TrainingState {
                            step: steps_done,
                            cycle,
                            sampling_rng,
                            quantile_rng,
                            densify_rng: rng_split,
                            topology_period: topology_cadence.period,
                            topology_steps_since_update: topology_cadence.steps_since_update,
                            densify_initial_points: densify_cadence.initial_points,
                            densify_steps_since_update: densify_cadence.steps_since_update,
                            densify_next_after: densify_cadence.next_after,
                            densification_round: densify_cadence.round,
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
                        phase_timings.checkpoint_metadata += metadata_start.elapsed();
                    }
                    Err(err) => {
                        phase_timings.checkpoint_optimizer += optimizer_start.elapsed();
                        log::warn!("checkpoint save failed: {err:?}");
                    }
                }
                phase_timings.checkpoint += checkpoint_start.elapsed();
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
    let training_path_truncated_percent = if training_path_rays == 0 {
        0.0
    } else {
        100.0 * training_path_truncated_rays as f32 / training_path_rays as f32
    };
    if model.radii.is_some() {
        log::info!(
            "training path telemetry: {} rays, max {}/{}, {} truncated ({:.6}%); \
             candidates max {}/{}",
            training_path_rays,
            training_path_max_steps_used,
            max_steps,
            training_path_truncated_rays,
            training_path_truncated_percent,
            training_candidate_max_used,
            path_bufs.splat_candidate_capacity(),
        );
    } else {
        log::info!(
            "training path telemetry: {} rays, max {}/{}, {} truncated ({:.6}%)",
            training_path_rays,
            training_path_max_steps_used,
            max_steps,
            training_path_truncated_rays,
            training_path_truncated_percent,
        );
    }

    let finalize_start = std::time::Instant::now();
    debug_dump_exposure(&session, views.len());
    if !model_parameters_current {
        download_model_parameters(&session, model, config.softplus_beta);
    }
    bake_mean_exposure_into_sh(&session, model, views.len());
    drop(session);
    gpu_cloud.deinit(&gpu);
    path_bufs.destroy(&gpu);
    let mut recorder = recorder;
    recorder.destroy(&gpu);
    gpu.destroy_command_encoder(&mut record_encoder);
    phase_timings.finalize += finalize_start.elapsed();
    phase_timings.log(
        wall_start.elapsed(),
        invocation_end.saturating_sub(config.resume_step),
    );

    AppearanceFitOutcome {
        losses,
        endpoint_checkpoint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::try_init_gpu;

    #[test]
    fn smooth_l1_matches_pytorch_beta_one_definition() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping smooth_l1_matches_pytorch_beta_one_definition: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let prediction = graph.input("prediction", &[4, 1]);
        let target = graph.input("target", &[4, 1]);
        let loss = smooth_l1_loss(&mut graph, prediction, target, 4);
        graph.set_outputs(vec![loss]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Inference,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_input("prediction", &[0.0, 0.5, 2.0, -2.0]);
        session.set_input("target", &[0.0; 4]);
        session.step();
        session.wait();
        // Element losses: 0, 0.5*0.5², 2-0.5, 2-0.5.
        let actual = session.read_output(1)[0];
        let expected = (0.0 + 0.125 + 1.5 + 1.5) / 4.0;
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    #[test]
    fn repeat_xyz_preserves_row_order() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping repeat_xyz row-order test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let column = graph.input("column", &[4, 1]);
        let repeated = repeat_xyz(&mut graph, column, 4);
        graph.set_outputs(vec![repeated]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Inference,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_input("column", &[0.25, -2.0, 7.5, 1.0]);
        session.step();
        session.wait();
        assert_eq!(
            session.read_output(12),
            vec![0.25, 0.25, 0.25, -2.0, -2.0, -2.0, 7.5, 7.5, 7.5, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn surface_plane_readback_leaves_frozen_topology_unchanged() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-normal readback test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![0.25; model.points.len()]);
        model.surface_normals = Some(vec![glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.0; model.points.len()]);
        let detail_count = model.points.len() * vol::SURFACE_DETAIL_SITES;
        model.surface_detail = Some(vol::SurfaceDetail {
            offsets: vec![glam::Vec3::ZERO; detail_count],
            heights: vec![0.0; detail_count],
            colors: vec![glam::Vec3::ZERO; detail_count],
        });
        let points_before = model.points.clone();
        let radii_before = model.radii.clone();

        let mut graph = mn::Graph::new();
        let normals = graph.parameter("surface_normals", &[model.points.len(), 3]);
        let offsets = graph.parameter("surface_offsets", &[model.points.len(), 1]);
        let detail_offsets = graph.parameter(
            "surface_detail_offsets",
            &[model.points.len(), vol::SURFACE_DETAIL_SITES * 3],
        );
        let detail_heights = graph.parameter(
            "surface_detail_heights",
            &[model.points.len(), vol::SURFACE_DETAIL_SITES],
        );
        let output = graph.reshape(normals, &[model.points.len() * 3]);
        graph.set_outputs(vec![output, offsets, detail_offsets, detail_heights]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Inference,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        let raw_normal = glam::Vec3::new(3.0, -4.0, 0.0);
        let raw_normals = (0..model.points.len())
            .flat_map(|_| raw_normal.to_array())
            .collect::<Vec<_>>();
        session.set_parameter("surface_normals", &raw_normals);
        let raw_offsets = (0..model.points.len())
            .map(|index| 0.01 * index as f32 - 0.02)
            .collect::<Vec<_>>();
        session.set_parameter("surface_offsets", &raw_offsets);
        let raw_detail_offsets = (0..detail_count)
            .flat_map(|index| glam::Vec3::new(index as f32, -1.0, 2.0).to_array())
            .collect::<Vec<_>>();
        let raw_detail_heights = (0..detail_count)
            .map(|index| 0.001 * index as f32)
            .collect::<Vec<_>>();
        session.set_parameter("surface_detail_offsets", &raw_detail_offsets);
        session.set_parameter("surface_detail_heights", &raw_detail_heights);

        download_model_surface_planes(&session, &mut model);

        assert_eq!(model.points, points_before);
        assert_eq!(model.radii, radii_before);
        assert_eq!(
            model.surface_normals.as_ref().unwrap(),
            &vec![raw_normal.normalize(); model.points.len()]
        );
        assert_eq!(model.surface_offsets.as_ref().unwrap(), &raw_offsets);
        let detail = model.surface_detail.as_ref().unwrap();
        for (offset, raw) in detail
            .offsets
            .iter()
            .zip(raw_detail_offsets.chunks_exact(3))
        {
            assert_eq!(*offset, glam::Vec3::from_slice(raw));
        }
        assert_eq!(detail.heights, raw_detail_heights);
    }

    #[test]
    fn surface_offset_only_training_keeps_topology_and_other_geometry_frozen() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-offset-only training test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![0.2; model.points.len()]);
        model.surface_normals = Some(vec![-glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.0; model.points.len()]);
        model.compute_adjacency_default();
        let positions_before = model
            .points
            .iter()
            .map(|point| point.truncate())
            .collect::<Vec<_>>();
        let radii_before = model.radii.clone();
        let normals_before = model.surface_normals.clone();
        let adjacency_before = model.adjacency.clone().unwrap();
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.0, 0.0, -1.0],
                depth: 10.0,
                cam_orientation: glam::Quat::IDENTITY.to_array(),
                fov: [0.5; 2],
                principal: [0.0; 2],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            target_alpha: None,
            width: 1,
            height: 1,
        };

        let losses = fit_appearance_multi_view(
            &mut model,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                learning_rate: 0.01,
                epochs: 2,
                surface_offset_lr_ratio: 1.0,
                geometry_rebuild_every: 1,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        assert_eq!(losses.len(), 2);
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert!(model
            .surface_offsets
            .as_ref()
            .unwrap()
            .iter()
            .any(|offset| offset.abs() > 1.0e-6));
        assert_eq!(
            model
                .points
                .iter()
                .map(|point| point.truncate())
                .collect::<Vec<_>>(),
            positions_before
        );
        assert_eq!(model.radii, radii_before);
        assert_eq!(model.surface_normals, normals_before);
        assert_eq!(
            model.adjacency.as_ref().unwrap().offsets,
            adjacency_before.offsets
        );
        assert_eq!(
            model.adjacency.as_ref().unwrap().neighbors,
            adjacency_before.neighbors
        );
        model.validate().unwrap();
    }

    #[test]
    fn surface_detail_training_updates_tables_with_frozen_topology() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-detail training test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![0.2; model.points.len()]);
        model.surface_normals = Some(vec![-glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.0; model.points.len()]);
        let detail_count = model.points.len() * vol::SURFACE_DETAIL_SITES;
        model.surface_detail = Some(vol::SurfaceDetail {
            offsets: (0..detail_count)
                .map(|index| {
                    let angle = std::f32::consts::TAU * (index % vol::SURFACE_DETAIL_SITES) as f32
                        / vol::SURFACE_DETAIL_SITES as f32;
                    0.1 * glam::Vec3::new(angle.cos(), angle.sin(), 0.0)
                })
                .collect(),
            heights: (0..detail_count)
                .map(|index| 0.01 * ((index % vol::SURFACE_DETAIL_SITES) as f32 - 3.5))
                .collect(),
            colors: (0..detail_count)
                .map(|index| {
                    let value = 0.01 * (index % vol::SURFACE_DETAIL_SITES) as f32;
                    glam::Vec3::new(value, -0.5 * value, 0.25 * value)
                })
                .collect(),
        });
        model.compute_adjacency_default();
        let positions_before = model
            .points
            .iter()
            .map(|point| point.truncate())
            .collect::<Vec<_>>();
        let radii_before = model.radii.clone();
        let adjacency_before = model.adjacency.clone().unwrap();
        let detail_before = model.surface_detail.clone().unwrap();
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.0, 0.0, -1.0],
                depth: 10.0,
                cam_orientation: glam::Quat::IDENTITY.to_array(),
                fov: [0.5; 2],
                principal: [0.0; 2],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            target_alpha: None,
            width: 1,
            height: 1,
        };

        let losses = fit_appearance_multi_view(
            &mut model,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                learning_rate: 0.01,
                epochs: 4,
                surface_detail_offset_lr_ratio: 0.25,
                surface_detail_height_lr_ratio: 0.25,
                surface_detail_color_lr_ratio: 1.0,
                geometry_rebuild_every: 1,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        assert_eq!(losses.len(), 4);
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert_ne!(model.surface_detail.as_ref().unwrap(), &detail_before);
        assert_eq!(
            model
                .points
                .iter()
                .map(|point| point.truncate())
                .collect::<Vec<_>>(),
            positions_before,
        );
        assert_eq!(model.radii, radii_before);
        assert_eq!(
            model.adjacency.as_ref().unwrap().offsets,
            adjacency_before.offsets,
        );
        assert_eq!(
            model.adjacency.as_ref().unwrap().neighbors,
            adjacency_before.neighbors,
        );
        model.validate().unwrap();
    }

    #[test]
    fn sh_color_graph_clamps_negative_values_before_weighting() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping sh_color_graph_clamps_negative_values_before_weighting: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let cell_indices = graph.input_u32("cell_indices", &[1]);
        let pixel_idx_per_step = graph.input_u32("pixel_idx_per_step", &[1]);
        let weight = graph.input("weight", &[1, 1]);
        let sh_r = graph.parameter("sh_r", &[1, 1]);
        let sh_g = graph.parameter("sh_g", &[1, 1]);
        let sh_b = graph.parameter("sh_b", &[1, 1]);
        let [pixel, _, _] = pixel_sh(
            &mut graph,
            cell_indices,
            &[
                ShChannelGraph {
                    dc: sh_r,
                    rest: None,
                },
                ShChannelGraph {
                    dc: sh_g,
                    rest: None,
                },
                ShChannelGraph {
                    dc: sh_b,
                    rest: None,
                },
            ],
            None,
            None,
            None,
            None,
            &[],
            pixel_idx_per_step,
            weight,
            1,
            1,
            1,
        )
        .pixels;
        graph.set_outputs(vec![pixel]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Inference,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_parameter("sh_r", &[-4.0]);
        session.set_parameter("sh_g", &[-4.0]);
        session.set_parameter("sh_b", &[-4.0]);
        session.set_input_u32("cell_indices", &[0]);
        session.set_input_u32("pixel_idx_per_step", &[0]);
        session.set_input("weight", &[1.0]);
        session.step();
        session.wait();

        let actual = session.read_output(1)[0];
        assert!(actual.abs() < 1.0e-6, "clamped color was {actual}");
    }

    #[test]
    fn packed_sh_graph_matches_scalar_reference_and_backpropagates() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!(
                "skipping packed_sh_graph_matches_scalar_reference_and_backpropagates: no GPU"
            );
            return;
        };
        let (n_cells, p, l, k) = (3usize, 2usize, 2usize, 4usize);
        let pl = p * l;
        let mut graph = mn::Graph::new();
        let cell_indices = graph.input_u32("cell_indices", &[pl]);
        let pixel_idx_per_step = graph.input_u32("pixel_idx_per_step", &[pl]);
        let weight = graph.input("weight", &[p, l]);
        let basis_inputs: Vec<mn::NodeId> = (1..k)
            .map(|component| graph.input(&format!("basis_{component}"), &[p, 1]))
            .collect();
        let sh_coefficients = declare_sh_parameters(&mut graph, n_cells, k);
        let pixels = pixel_sh(
            &mut graph,
            cell_indices,
            &sh_coefficients,
            None,
            None,
            None,
            None,
            &basis_inputs,
            pixel_idx_per_step,
            weight,
            n_cells,
            p,
            l,
        )
        .pixels;
        let mean_r = graph.mean_all(pixels[0]);
        let mean_g = graph.mean_all(pixels[1]);
        let mean_b = graph.mean_all(pixels[2]);
        let mean_rg = graph.add(mean_r, mean_g);
        let loss = graph.add(mean_rg, mean_b);
        graph.set_outputs(vec![loss, pixels[0], pixels[1], pixels[2]]);

        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        let coefficient = |channel: usize, component: usize, cell: usize| {
            0.01 * (1 + channel * k * n_cells + component * n_cells + cell) as f32
        };
        for (channel_index, channel) in ["sh_r", "sh_g", "sh_b"].iter().enumerate() {
            let dc: Vec<f32> = (0..n_cells)
                .map(|cell| coefficient(channel_index, 0, cell))
                .collect();
            session.set_parameter(channel, &dc);
            let mut rest = vec![0.0_f32; n_cells * (k - 1)];
            for cell in 0..n_cells {
                for component in 1..k {
                    rest[cell * (k - 1) + component - 1] =
                        coefficient(channel_index, component, cell);
                }
            }
            session.set_parameter(&sh_rest_parameter_name(channel), &rest);
        }
        let cells = [0_u32, 1, 2, 0];
        let pixel_indices = [0_u32, 0, 1, 1];
        let weights = [0.2_f32, 0.3, 0.4, 0.1];
        let basis = [[0.11_f32, 0.17], [0.23, 0.29], [0.31, 0.37]];
        session.set_input_u32("cell_indices", &cells);
        session.set_input_u32("pixel_idx_per_step", &pixel_indices);
        session.set_input("weight", &weights);
        for (component, values) in basis.iter().enumerate() {
            session.set_input(&format!("basis_{}", component + 1), values);
        }
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);
        session.step();
        session.wait();

        for channel in 0..3 {
            let mut actual = vec![0.0_f32; p];
            session.read_output_by_index(channel + 1, &mut actual);
            let mut expected = vec![0.0_f32; p];
            for pixel in 0..p {
                for step in 0..l {
                    let path_index = pixel * l + step;
                    let cell = cells[path_index] as usize;
                    let mut color = coefficient(channel, 0, cell) * SH_C0;
                    for component in 1..k {
                        color +=
                            coefficient(channel, component, cell) * basis[component - 1][pixel];
                    }
                    expected[pixel] += weights[path_index] * (color + 0.5).max(0.0);
                }
            }
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
            }
        }

        for channel in ["sh_r", "sh_g", "sh_b"] {
            let mut dc_gradient = vec![0.0_f32; n_cells];
            session.read_param_grad(channel, &mut dc_gradient);
            assert!(
                dc_gradient
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
                "missing DC gradient for {channel}: {dc_gradient:?}"
            );
            let rest_name = sh_rest_parameter_name(channel);
            let mut rest_gradient = vec![0.0_f32; n_cells * (k - 1)];
            session.read_param_grad(&rest_name, &mut rest_gradient);
            assert!(
                rest_gradient
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
                "missing packed gradient for {rest_name}: {rest_gradient:?}"
            );
        }
    }

    #[test]
    fn packed_surface_color_matches_scalar_reference_and_backpropagates() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping packed surface-color graph test: no GPU");
            return;
        };
        let (n_cells, p, l) = (2usize, 1usize, 2usize);
        let components = vol::SURFACE_COLOR_COMPONENTS;
        let mut graph = mn::Graph::new();
        let cell_indices = graph.input_u32("cell_indices", &[p * l]);
        let pixel_idx_per_step = graph.input_u32("pixel_idx_per_step", &[p * l]);
        let weight = graph.input("weight", &[p, l]);
        let surface_basis = graph.input("surface_basis", &[p * l, components]);
        let surface_color =
            graph.parameter("surface_color_coefficients", &[n_cells, components * 3]);
        let sh_coefficients = declare_sh_parameters(&mut graph, n_cells, 1);
        let pixels = pixel_sh(
            &mut graph,
            cell_indices,
            &sh_coefficients,
            Some(surface_color),
            Some(surface_basis),
            None,
            None,
            &[],
            pixel_idx_per_step,
            weight,
            n_cells,
            p,
            l,
        )
        .pixels;
        let sum_r = graph.sum_all(pixels[0]);
        let sum_g = graph.sum_all(pixels[1]);
        let sum_b = graph.sum_all(pixels[2]);
        let sum_rg = graph.add(sum_r, sum_g);
        let loss = graph.add(sum_rg, sum_b);
        graph.set_outputs(vec![loss, pixels[0], pixels[1], pixels[2]]);

        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        for channel in ["sh_r", "sh_g", "sh_b"] {
            session.set_parameter(channel, &[0.0; 2]);
        }
        let coefficients: Vec<f32> = (0..n_cells * components * 3)
            .map(|index| 0.01 * (index + 1) as f32)
            .collect();
        let basis = [0.25_f32, -0.5, 0.75, 0.4, -0.2, 0.1, 0.3, 0.14];
        let weights = [0.4_f32, 0.6];
        session.set_parameter("surface_color_coefficients", &coefficients);
        session.set_input_u32("cell_indices", &[0, 1]);
        session.set_input_u32("pixel_idx_per_step", &[0, 0]);
        session.set_input("weight", &weights);
        session.set_input("surface_basis", &basis);
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);
        session.step();
        session.wait();

        for channel in 0..3 {
            let mut actual = [0.0_f32];
            session.read_output_by_index(channel + 1, &mut actual);
            let expected = (0..l)
                .map(|step| {
                    let base = step * components * 3 + channel * components;
                    let residual = (0..components)
                        .map(|component| {
                            coefficients[base + component] * basis[step * components + component]
                        })
                        .sum::<f32>();
                    weights[step] * (0.5 + residual)
                })
                .sum::<f32>();
            assert!((actual[0] - expected).abs() < 2.0e-6);
        }

        let mut gradient = vec![0.0_f32; coefficients.len()];
        session.read_param_grad("surface_color_coefficients", &mut gradient);
        for step in 0..l {
            for channel in 0..3 {
                for component in 0..components {
                    let index = step * components * 3 + channel * components + component;
                    let expected = weights[step] * basis[step * components + component];
                    assert!((gradient[index] - expected).abs() < 2.0e-6);
                }
            }
        }
    }

    #[test]
    fn surface_detail_graph_matches_cpu_and_backpropagates() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-detail graph test: no GPU");
            return;
        };
        let sites = vol::SURFACE_DETAIL_SITES;
        let mut graph = mn::Graph::new();
        let cell_indices = graph.input_u32("cell_indices", &[1]);
        let parameters = SurfaceDetailGraph {
            offsets: graph.parameter("surface_detail_offsets", &[1, sites * 3]),
            heights: graph.parameter("surface_detail_heights", &[1, sites]),
            colors: graph.parameter("surface_detail_colors", &[1, sites * 3]),
            surface_queries: graph.input("surface_queries", &[1, 2]),
        };
        let centers = graph.parameter("positions", &[1, 3]);
        let normals = graph.parameter("surface_normals", &[1, 3]);
        let base_offsets = graph.parameter("surface_offsets", &[1, 1]);
        let radii = graph.parameter("radii", &[1, 1]);
        let ray_origin = graph.input("ray_origin", &[1, 3]);
        let ray_direction = graph.input("ray_direction", &[1, 3]);
        let evaluation = evaluate_surface_detail_graph(
            &mut graph,
            cell_indices,
            &parameters,
            centers,
            normals,
            base_offsets,
            radii,
            ray_origin,
            ray_direction,
            1,
        );
        let red_table = split_rgb_table(&mut graph, parameters.colors, 1, sites)[0];
        let red = graph.embedding(cell_indices, red_table);
        let red = graph.mul(red, evaluation.color_weights);
        let red = graph.sum_inner(red);
        let loss_terms = graph.add(evaluation.effective_offsets, red);
        let loss = graph.sum_all(loss_terms);
        graph.set_outputs(vec![loss, evaluation.effective_offsets, red]);

        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        let mut offsets = vec![-0.5 * glam::Vec3::X; sites];
        offsets[0] = 0.5 * glam::Vec3::X;
        let packed_offsets = offsets
            .iter()
            .flat_map(|offset| offset.to_array())
            .collect::<Vec<_>>();
        let mut heights = vec![0.0_f32; sites];
        heights[0] = 0.25;
        let selected_color = glam::Vec3::new(0.2, -0.1, 0.05);
        let mut colors = vec![glam::Vec3::ZERO; sites];
        colors[0] = selected_color;
        let mut packed_colors = vec![0.0_f32; sites * 3];
        for site in 0..sites {
            for channel in 0..3 {
                packed_colors[channel * sites + site] = colors[site][channel];
            }
        }
        let query_near = 10.0 - 3.0_f32.sqrt();
        session.set_parameter("surface_detail_offsets", &packed_offsets);
        session.set_parameter("surface_detail_heights", &heights);
        session.set_parameter("surface_detail_colors", &packed_colors);
        session.set_parameter("positions", &[0.0, 0.0, 10.0]);
        session.set_parameter("surface_normals", &[0.0, 0.0, -1.0]);
        session.set_parameter("surface_offsets", &[0.0]);
        session.set_parameter("radii", &[2.0]);
        session.set_input_u32("cell_indices", &[0]);
        session.set_input("surface_queries", &[query_near, 1.0]);
        session.set_input("ray_origin", &[1.0, 0.0, 0.0]);
        session.set_input("ray_direction", &[0.0, 0.0, 1.0]);
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);
        session.step();
        session.wait();

        let model = vol::PointCloudModel {
            points: vec![glam::Vec4::new(0.0, 0.0, 10.0, 1.0)],
            sh_coefficients: vec![0.0; 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(vol::Adjacency {
                neighbors: Vec::new(),
                offsets: vec![0, 0],
            }),
            radii: Some(vec![2.0]),
            surface_normals: Some(vec![-glam::Vec3::Z]),
            surface_offsets: Some(vec![0.0]),
            surface_detail: Some(vol::SurfaceDetail {
                offsets,
                heights,
                colors,
            }),
            surface_color_coefficients: None,
            spherical_voronoi: None,
        };
        let (expected_offset, expected_color) = vol::trace::eval_surface_detail(
            &model,
            0,
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::Z,
            query_near,
        );
        let mut actual_offset = [0.0_f32];
        let mut actual_red = [0.0_f32];
        session.read_output_by_index(1, &mut actual_offset);
        session.read_output_by_index(2, &mut actual_red);
        assert!((actual_offset[0] - expected_offset).abs() < 2.0e-5);
        assert!((actual_red[0] - expected_color.x).abs() < 2.0e-5);

        for (name, size) in [
            ("surface_detail_offsets", sites * 3),
            ("surface_detail_heights", sites),
            ("surface_detail_colors", sites * 3),
        ] {
            let mut gradient = vec![0.0_f32; size];
            session.read_param_grad(name, &mut gradient);
            assert!(gradient.iter().all(|value| value.is_finite()));
            assert!(gradient.iter().any(|value| value.abs() > 1.0e-8));
        }
    }

    #[test]
    fn packed_spherical_voronoi_matches_cpu_reference_and_backpropagates() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping packed Spherical Voronoi graph test: no GPU");
            return;
        };
        let (n_cells, p, l) = (2usize, 1usize, 2usize);
        let sites = vol::SPHERICAL_VORONOI_SITES;
        let mut graph = mn::Graph::new();
        let cell_indices = graph.input_u32("cell_indices", &[p * l]);
        let pixel_idx_per_step = graph.input_u32("pixel_idx_per_step", &[p * l]);
        let weight = graph.input("weight", &[p, l]);
        let ray_directions = graph.input("ray_directions", &[p * l, 3]);
        let parameters = SphericalVoronoiGraph {
            axes: graph.parameter("spherical_voronoi_axes", &[n_cells, sites * 3]),
            colors: graph.parameter("spherical_voronoi_colors", &[n_cells, sites * 3]),
        };
        let sh_coefficients = declare_sh_parameters(&mut graph, n_cells, 1);
        let pixels = pixel_sh(
            &mut graph,
            cell_indices,
            &sh_coefficients,
            None,
            None,
            None,
            Some((&parameters, ray_directions)),
            &[],
            pixel_idx_per_step,
            weight,
            n_cells,
            p,
            l,
        )
        .pixels;
        let sum_r = graph.sum_all(pixels[0]);
        let sum_g = graph.sum_all(pixels[1]);
        let sum_b = graph.sum_all(pixels[2]);
        let sum_rg = graph.add(sum_r, sum_g);
        let loss = graph.add(sum_rg, sum_b);
        graph.set_outputs(vec![loss, pixels[0], pixels[1], pixels[2]]);

        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        for channel in ["sh_r", "sh_g", "sh_b"] {
            session.set_parameter(channel, &[0.0; 2]);
        }
        let axes = (0..n_cells * sites)
            .map(|index| {
                let site = (index % sites) as f32;
                glam::Vec3::new(0.35 * site - 1.2, 0.2 - 0.15 * site, 0.1 * site - 0.3)
            })
            .collect::<Vec<_>>();
        let colors = (0..n_cells * sites)
            .map(|index| {
                let value = (index % sites) as f32 * 0.02 - 0.06;
                glam::Vec3::new(value, -0.5 * value, 0.25 * value)
            })
            .collect::<Vec<_>>();
        let mut packed_axes = Vec::with_capacity(n_cells * sites * 3);
        for &axis in &axes {
            packed_axes.extend_from_slice(&axis.to_array());
        }
        let mut packed_colors = vec![0.0_f32; n_cells * sites * 3];
        for cell in 0..n_cells {
            for site in 0..sites {
                for channel in 0..3 {
                    packed_colors[cell * sites * 3 + channel * sites + site] =
                        colors[cell * sites + site][channel];
                }
            }
        }
        let directions = [glam::Vec3::X, glam::Vec3::Y];
        let direction_values = directions
            .iter()
            .flat_map(|direction| direction.to_array())
            .collect::<Vec<_>>();
        let weights = [0.4_f32, 0.6];
        session.set_parameter("spherical_voronoi_axes", &packed_axes);
        session.set_parameter("spherical_voronoi_colors", &packed_colors);
        session.set_input_u32("cell_indices", &[0, 1]);
        session.set_input_u32("pixel_idx_per_step", &[0, 0]);
        session.set_input("weight", &weights);
        session.set_input("ray_directions", &direction_values);
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);
        session.step();
        session.wait();

        let spherical_voronoi = vol::SphericalVoronoi { axes, colors };
        let mut expected = glam::Vec3::ZERO;
        for step in 0..l {
            let residual = vol::trace::eval_spherical_voronoi(
                &spherical_voronoi,
                step as u32,
                directions[step],
            );
            expected += weights[step] * (glam::Vec3::splat(0.5) + residual).max(glam::Vec3::ZERO);
        }
        for channel in 0..3 {
            let mut actual = [0.0_f32];
            session.read_output_by_index(channel + 1, &mut actual);
            assert!((actual[0] - expected[channel]).abs() < 2.0e-6);
        }

        let mut axis_gradient = vec![0.0_f32; packed_axes.len()];
        let mut color_gradient = vec![0.0_f32; packed_colors.len()];
        session.read_param_grad("spherical_voronoi_axes", &mut axis_gradient);
        session.read_param_grad("spherical_voronoi_colors", &mut color_gradient);
        assert!(axis_gradient.iter().all(|value| value.is_finite()));
        assert!(color_gradient.iter().all(|value| value.is_finite()));
        assert!(axis_gradient.iter().any(|value| value.abs() > 1.0e-7));
        assert!(color_gradient.iter().any(|value| value.abs() > 1.0e-7));
    }

    #[test]
    fn differentiable_surface_basis_matches_cpu_oracle() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping differentiable surface-basis test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let cells = graph.input_u32("cells", &[1]);
        let positions = graph.parameter("positions", &[1, 3]);
        let normals = graph.parameter("normals", &[1, 3]);
        let offsets = graph.parameter("offsets", &[1, 1]);
        let radii = graph.parameter("radii", &[1, 1]);
        let ray_origin = graph.input("ray_origin", &[1, 3]);
        let ray_direction = graph.input("ray_direction", &[1, 3]);
        let basis = surface_color_basis_graph(
            &mut graph,
            cells,
            positions,
            normals,
            Some(offsets),
            radii,
            ray_origin,
            ray_direction,
            1,
        );
        graph.set_outputs(vec![basis]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Inference,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        let mut model = tiny_model();
        model.points.truncate(1);
        model.sh_coefficients.truncate(3);
        model.radii = Some(vec![2.0]);
        model.surface_normals = Some(vec![glam::Vec3::Z]);
        model.surface_offsets = Some(vec![0.2]);
        model.adjacency = Some(vol::Adjacency {
            neighbors: Vec::new(),
            offsets: vec![0, 0],
        });
        let origin = glam::Vec3::new(1.0, 0.0, -1.0);
        let direction = glam::Vec3::Z;
        session.set_parameter("positions", &model.points[0].truncate().to_array());
        session.set_parameter("normals", &glam::Vec3::Z.to_array());
        session.set_parameter("offsets", &[0.2]);
        session.set_parameter("radii", &[2.0]);
        session.set_input_u32("cells", &[0]);
        session.set_input("ray_origin", &origin.to_array());
        session.set_input("ray_direction", &direction.to_array());
        session.step();
        session.wait();
        let mut actual = [0.0_f32; vol::SURFACE_COLOR_COMPONENTS];
        session.read_output_by_index(0, &mut actual);
        let expected = vol::trace::surface_color_basis(&model, 0, origin, direction);
        for (actual, expected) in actual.iter().zip(expected.to_array()) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn adam_state_roundtrip() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
            points,
        };
        m.compute_adjacency_default();
        m
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
    fn weighted_densification_perturbs_both_duplicate_siblings() {
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
            let offset = (point - parent_point).truncate();
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
    fn oriented_densification_offsets_both_siblings_in_the_surface_plane() {
        let mut model = tiny_model();
        model.radii = Some(vec![0.4, 0.3, 0.2, 0.1]);
        let normal = glam::Vec3::new(0.2, -0.3, 1.0).normalize();
        model.surface_normals = Some(vec![normal; model.points.len()]);
        model.surface_offsets = Some(vec![-0.03, -0.01, 0.02, 0.04]);
        model.surface_color_coefficients = Some(
            (0..model.points.len() * vol::SURFACE_COLOR_COMPONENTS * 3)
                .map(|index| index as f32)
                .collect(),
        );
        let detail_count = model.points.len() * vol::SURFACE_DETAIL_SITES;
        model.surface_detail = Some(vol::SurfaceDetail {
            offsets: (0..detail_count)
                .map(|index| glam::Vec3::new(index as f32, 1.0, -1.0))
                .collect(),
            heights: (0..detail_count).map(|index| 0.01 * index as f32).collect(),
            colors: (0..detail_count)
                .map(|index| glam::Vec3::splat(0.02 * index as f32))
                .collect(),
        });
        model.compute_adjacency_default();
        let parent = 2;
        let parent_point = model.points[parent];
        let mut rng = 0x1234_5678_9ABC_DEF0;
        let config = DensifyConfig {
            fraction: 0.25,
            target_points: 5,
            prune: false,
            ..DensifyConfig::default()
        };

        let (new_to_old, pruned, added) = prune_and_densify(
            &mut model,
            &[0.0, 0.0, 1.0e20, 0.0],
            &[1.0; 4],
            &config,
            &mut rng,
            0.0,
        );

        assert_eq!(new_to_old, [0, 1, 2, 3, 2]);
        assert_eq!((pruned, added), (0, 1));
        for index in [parent, 4] {
            let offset = (model.points[index] - parent_point).truncate();
            assert!((offset.length() - 0.01).abs() < 1.0e-6);
            assert!(offset.dot(normal).abs() < 1.0e-6);
        }
        assert_eq!(model.surface_normals.as_ref().unwrap(), &vec![normal; 5]);
        assert_eq!(
            model.surface_offsets.as_ref().unwrap(),
            &vec![-0.03, -0.01, 0.02, 0.04, 0.02]
        );
        let coefficients = model.surface_color_coefficients.as_ref().unwrap();
        let block = vol::SURFACE_COLOR_COMPONENTS * 3;
        assert_eq!(
            &coefficients[4 * block..5 * block],
            &coefficients[2 * block..3 * block]
        );
        let detail = model.surface_detail.as_ref().unwrap();
        let detail_block = vol::SURFACE_DETAIL_SITES;
        assert_eq!(
            &detail.offsets[4 * detail_block..5 * detail_block],
            &detail.offsets[2 * detail_block..3 * detail_block],
        );
        assert_eq!(
            &detail.heights[4 * detail_block..5 * detail_block],
            &detail.heights[2 * detail_block..3 * detail_block],
        );
        assert_eq!(
            &detail.colors[4 * detail_block..5 * detail_block],
            &detail.colors[2 * detail_block..3 * detail_block],
        );
        model.validate().unwrap();
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
    fn contribution_view_selection_is_exhaustive_or_rotating_and_stratified() {
        assert_eq!(contribution_view_indices(5, 0, 7), [0, 1, 2, 3, 4]);
        assert_eq!(contribution_view_indices(5, 5, 7), [0, 1, 2, 3, 4]);
        assert!(contribution_view_indices(0, 3, 7).is_empty());

        let first = contribution_view_indices(17, 5, 0);
        let second = contribution_view_indices(17, 5, 1);
        assert_eq!(first.len(), 5);
        assert_eq!(second.len(), 5);
        assert_ne!(first, second);
        for selected in [first, second] {
            let mut sorted = selected.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), selected.len());
            assert!(selected.iter().all(|&index| index < 17));
        }
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
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
            let names = per_cell_param_names_with_stride(
                sh_degree, false, false, false, false, false, false, false,
            );
            // Required entries:
            assert!(names.contains(&("log_density".to_string(), 1)));
            assert!(names.contains(&("positions".to_string(), 3)));
            // One DC table per channel, plus one packed rest table when K>1.
            let num_components = (1 + sh_degree) * (1 + sh_degree);
            assert_eq!(names.len(), 2 + if num_components > 1 { 6 } else { 3 });
            for channel in ["sh_r", "sh_g", "sh_b"] {
                assert!(names.contains(&(channel.to_string(), 1)));
                if num_components > 1 {
                    assert!(names.contains(&(sh_rest_parameter_name(channel), num_components - 1,)));
                }
            }
            let weighted_names = per_cell_param_names_with_stride(
                sh_degree, true, false, false, false, false, false, true,
            );
            assert!(weighted_names.contains(&("log_radii".to_string(), 1)));
            assert!(weighted_names.contains(&(POINT_ERROR_PROBE.to_string(), 1)));
            assert_eq!(weighted_names.len(), names.len() + 2);
            let oriented_names = per_cell_param_names_with_stride(
                sh_degree, true, true, true, true, true, true, true,
            );
            assert!(oriented_names.contains(&("surface_normals".to_string(), 3)));
            assert!(oriented_names.contains(&("surface_offsets".to_string(), 1)));
            assert!(oriented_names.contains(&(
                "surface_color_coefficients".to_string(),
                vol::SURFACE_COLOR_COMPONENTS * 3,
            )));
            assert!(oriented_names.contains(&(
                "spherical_voronoi_axes".to_string(),
                vol::SPHERICAL_VORONOI_SITES * 3,
            )));
            assert!(oriented_names.contains(&(
                "spherical_voronoi_colors".to_string(),
                vol::SPHERICAL_VORONOI_SITES * 3,
            )));
            assert!(oriented_names.contains(&(
                "surface_detail_offsets".to_string(),
                vol::SURFACE_DETAIL_SITES * 3,
            )));
            assert!(oriented_names.contains(&(
                "surface_detail_heights".to_string(),
                vol::SURFACE_DETAIL_SITES,
            )));
            assert!(oriented_names.contains(&(
                "surface_detail_colors".to_string(),
                vol::SURFACE_DETAIL_SITES * 3,
            )));
            assert_eq!(oriented_names.len(), names.len() + 10);
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
                &mut g,
                n_cells,
                n_pixels,
                max_steps,
                sh_degree,
                num_views,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                [0.0; 3],
                false,
                false,
                false,
                false,
                false,
                false,
                ColorLoss::L1,
            );
            assert_eq!(vg.sh_degree, sh_degree);
            assert_eq!(vg.n_cells, n_cells);
            assert_eq!(vg.n_pixels, n_pixels);
            assert_eq!(vg.max_steps, max_steps);
            assert_eq!(vg.num_views, num_views);
            assert!(vg.weighted_path.is_none());
            // SH coefficient tables: one DC and one packed rest per channel.
            let num_components = (1 + sh_degree) * (1 + sh_degree);
            assert_eq!(vg.sh_coefficients.len(), 3);
            for chan in &vg.sh_coefficients {
                assert_eq!(chan.rest.is_some(), num_components > 1);
            }
            // basis_inputs: K-1 entries (component 0 is folded in).
            assert_eq!(vg.basis_inputs.len(), num_components - 1);
        }
    }

    #[test]
    fn build_volumetric_graph_constructs_weighted_tangent_path() {
        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph(
            &mut graph,
            16,
            4,
            3,
            0,
            2,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            true,
            false,
            false,
            false,
            false,
            false,
            ColorLoss::L1,
        );
        assert!(vg.weighted_path.is_some());

        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph(
            &mut graph,
            16,
            4,
            3,
            0,
            2,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            true,
            true,
            true,
            false,
            false,
            false,
            ColorLoss::L1,
        );
        let weighted = vg.weighted_path.unwrap();
        assert!(weighted.surface_normals.is_some());
        assert!(weighted.surface_offsets.is_some());
        assert!(weighted.dt_grad_surface_normal.is_some());
        assert!(weighted.surface_normal_loss_scale.is_some());

        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph_with_options(
            &mut graph,
            16,
            4,
            3,
            0,
            2,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            true,
            true,
            VolumetricGraphOptions {
                use_surface_normal_loss: false,
                train_positions: true,
                train_radii: true,
                use_surface_detail: false,
            },
            true,
            false,
            false,
            false,
            ColorLoss::L1,
        );
        let weighted = vg.weighted_path.unwrap();
        assert!(weighted.dt_grad_surface_normal.is_some());
        assert!(weighted.surface_normal_loss_scale.is_none());
    }

    #[test]
    fn oriented_surface_data_is_gathered_once_per_path_row() {
        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph_with_options(
            &mut graph,
            16,
            4,
            3,
            0,
            2,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            true,
            true,
            VolumetricGraphOptions {
                use_surface_normal_loss: true,
                train_positions: false,
                train_radii: false,
                use_surface_detail: false,
            },
            true,
            true,
            false,
            false,
            ColorLoss::L1,
        );
        let weighted = vg.weighted_path.unwrap();
        let surface_jacobian = weighted.dt_grad_surface_normal.unwrap();
        assert!(
            matches!(graph.node(surface_jacobian).op, mn::graph::Op::Input { ref name } if name == "dt_grad_surface_normal")
        );
        let surface_normals = weighted.surface_normals.unwrap();
        let normalized_normals = graph
            .nodes()
            .iter()
            .find(|node| {
                matches!(&node.op, mn::graph::Op::RmsNorm { .. })
                    && node.inputs.first() == Some(&surface_normals)
            })
            .unwrap()
            .id;
        let surface_offsets = weighted.surface_offsets.unwrap();
        let embedding_count = |table| {
            graph
                .nodes()
                .iter()
                .filter(|node| {
                    matches!(&node.op, mn::graph::Op::Embedding)
                        && node.inputs.get(1) == Some(&table)
                })
                .count()
        };
        assert_eq!(embedding_count(normalized_normals), 1);
        assert_eq!(embedding_count(surface_offsets), 1);
        let materializes: Vec<&mn::graph::Node> = graph
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, mn::graph::Op::Materialize)
                    && node.inputs.first() == Some(&surface_jacobian)
            })
            .collect();
        assert_eq!(materializes.len(), 1);
        let staged = materializes[0].id;

        let split_count = graph
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.op,
                    mn::graph::Op::SplitA { .. } | mn::graph::Op::SplitB { .. }
                ) && node.inputs.first() == Some(&staged)
            })
            .count();
        assert_eq!(split_count, 2);
    }

    #[test]
    fn frozen_weighted_geometry_is_absent_from_the_backward_graph() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping frozen weighted-geometry graph test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph_with_options(
            &mut graph,
            4,
            1,
            2,
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,
            [0.0; 3],
            true,
            true,
            VolumetricGraphOptions {
                use_surface_normal_loss: false,
                train_positions: false,
                train_radii: false,
                use_surface_detail: false,
            },
            true,
            true,
            false,
            false,
            ColorLoss::SmoothL1,
        );
        let weighted = vg.weighted_path.unwrap();
        assert!(weighted.dt_reference_tangent.is_some());
        assert!(weighted.previous_cell_indices.is_none());
        assert!(weighted.dt_grad_previous.is_none());
        assert!(weighted.dt_grad_current.is_none());
        assert!(weighted.dt_grad_next.is_none());
        assert!(weighted.dt_grad_surface_normal.is_some());
        let (session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );

        assert!(!session.has_param_grad("positions"));
        assert!(!session.has_param_grad("log_radii"));
        assert!(session.has_param_grad("log_density"));
        assert!(session.has_param_grad("surface_normals"));
        assert!(session.has_param_grad("surface_offsets"));
        assert!(session.has_param_grad("surface_color_coefficients"));
    }

    #[test]
    fn powerfoam_point_error_probe_tracks_local_photometric_responsibility() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping PowerFoam point-error probe test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph(
            &mut graph,
            2,
            1,
            2,
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            true,
            false,
            false,
            false,
            true,
            false,
            ColorLoss::L1,
        );
        assert!(vg.point_error_probe.is_some());
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );

        session.set_parameter("log_density", &[std::f32::consts::LN_2; 2]);
        session.set_parameter("positions", &[0.0, 0.0, 1.0, 0.0, 0.0, 2.0]);
        session.set_parameter("log_radii", &[inv_radius_activation(1.0); 2]);
        let bright_coefficient = 0.5 / SH_C0;
        for channel in ["sh_r", "sh_g", "sh_b"] {
            session.set_parameter(channel, &[0.0, bright_coefficient]);
        }
        for channel in ["exposure_r", "exposure_g", "exposure_b"] {
            session.set_parameter(channel, &[1.0]);
        }
        session.set_parameter(POINT_ERROR_PROBE, &[0.0; 2]);

        session.set_input_u32("cell_indices", &[0, 1]);
        session.set_input_u32("previous_cell_indices", &[0, 1]);
        session.set_input_u32("next_cell_indices", &[0, 1]);
        session.set_input("recorded_dt", &[1.0; 2]);
        session.set_input("dt_reference_tangent", &[0.0; 2]);
        session.set_input("dt_grad_previous", &[0.0; 8]);
        session.set_input("dt_grad_current", &[0.0; 8]);
        session.set_input("dt_grad_next", &[0.0; 8]);
        session.set_input("mask", &[1.0; 2]);
        session.set_input("ray_origin", &[0.0; 3]);
        session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
        session.set_input_u32("pixel_idx_per_step", &[0, 0]);
        session.set_input_u32("view_idx", &[0]);
        session.set_input("labels", &[0.5; 3]);
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);
        session.step();
        session.wait();

        let loss = session.read_output(1)[0];
        assert!(loss.abs() < 1.0e-6, "probe changed forward loss: {loss}");
        let mut gradient = [0.0_f32; 2];
        session.read_param_grad(POINT_ERROR_PROBE, &mut gradient);
        assert!(gradient[0].abs() < 1.0e-6);
        assert!((gradient[1] - 0.375).abs() < 1.0e-5, "{gradient:?}");
        let mut first_moment = [0.0_f32; 2];
        session.read_adam_m(POINT_ERROR_PROBE, &mut first_moment);
        assert!(first_moment[0].abs() < 1.0e-6);
        assert!(
            (first_moment[1] - 0.0375).abs() < 1.0e-5,
            "{first_moment:?}",
        );
        let mut probe = [f32::NAN; 2];
        session.read_param(POINT_ERROR_PROBE, &mut probe);
        assert_eq!(probe, [0.0; 2]);
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
    fn interpenetration_loss_matches_overlap_value_and_gradients() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping interpenetration loss test: no GPU");
            return;
        };
        let mut graph = mn::Graph::new();
        let positions = graph.parameter("positions", &[2, 3]);
        let log_radii = graph.parameter("log_radii", &[2, 1]);
        let base_loss = graph.constant(vec![0.0], &[1]);
        let loss = add_interpenetration_loss(&mut graph, base_loss, positions, log_radii, 1);
        graph.set_outputs(vec![loss]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_parameter("positions", &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let raw_radius = inv_radius_activation(1.5);
        session.set_parameter("log_radii", &[raw_radius, raw_radius]);
        session.set_input_u32("interpenetration_edge_a", &[0]);
        session.set_input_u32("interpenetration_edge_b", &[1]);
        session.set_input("interpenetration_edge_direction", &[-1.0, 0.0, 0.0]);
        session.set_input("interpenetration_edge_scale", &[0.25]);
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);
        session.step();
        session.wait();

        let value = session.read_output(1)[0];
        assert!((value - 0.25).abs() < 1.0e-6, "value={value}");
        let mut position_grad = [0.0_f32; 6];
        session.read_param_grad("positions", &mut position_grad);
        assert!((position_grad[0] - 0.5).abs() < 1.0e-5);
        assert!((position_grad[3] + 0.5).abs() < 1.0e-5);
        assert!(position_grad[1..3].iter().all(|&value| value == 0.0));
        assert!(position_grad[4..].iter().all(|&value| value == 0.0));
        let mut radius_grad = [0.0_f32; 2];
        session.read_param_grad("log_radii", &mut radius_grad);
        assert!((radius_grad[0] - 0.5).abs() < 1.0e-5);
        assert!((radius_grad[1] - 0.5).abs() < 1.0e-5);
    }

    #[test]
    fn build_volumetric_graph_constructs_in_patch_mode() {
        // Patch mode: `n_pixels == patch_size²`, gradient L1 added to
        // the loss. Catches shape mismatches inside `patch_grad_l1`.
        let mut g = mn::Graph::new();
        let patch_size = 4usize;
        let n_pixels = patch_size * patch_size;
        let vg = build_volumetric_graph(
            &mut g,
            16,
            n_pixels,
            4,
            0,
            2,
            patch_size,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            false,
            false,
            false,
            false,
            false,
            false,
            ColorLoss::L1,
        );
        assert_eq!(vg.n_pixels, n_pixels);
    }

    #[test]
    fn mixed_view_sampling_is_stratified_and_preserves_single_view_draws() {
        let seed = 0x1234_5678_9ABC_DEF0;
        let mut single_state = seed;
        let single = sample_training_views(&mut single_state, 11, 1);
        let mut legacy_state = seed;
        let legacy = (next_lcg_u32(&mut legacy_state) as usize) % 11;
        assert_eq!(single, [legacy]);
        assert_eq!(single_state, legacy_state);

        let mut mixed_state = seed;
        let mixed = sample_training_views(&mut mixed_state, 11, 4);
        assert_eq!(mixed.len(), 4);
        for (slot, &view) in mixed.iter().enumerate() {
            assert!(view >= slot * 11 / 4);
            assert!(view < (slot + 1) * 11 / 4);
        }
        assert_eq!(mixed_state, advance_lcg(seed, 4));

        let ranges: Vec<_> = (0..4).map(|slot| batch_view_range(slot, 4, 10)).collect();
        assert_eq!(ranges, [0..2, 2..5, 5..7, 7..10]);
    }

    #[test]
    fn interpenetration_schedule_matches_reference_endpoints() {
        let initial = 1.0e-4;
        assert_eq!(interpenetration_weight_at_step(0.0, 42, 100), 0.0);
        assert_eq!(interpenetration_weight_at_step(initial, 0, 100), initial);
        let final_weight = interpenetration_weight_at_step(initial, 99, 100);
        assert!((final_weight - 1.0e-7).abs() < 1.0e-13);
        let midpoint = interpenetration_weight_at_step(initial, 50, 101);
        assert!((midpoint - initial * 1.0e-3_f32.sqrt()).abs() < 1.0e-10);
        assert_eq!(
            interpenetration_weight_at_step(initial, 1000, 100),
            final_weight
        );
    }

    #[test]
    fn surface_normal_schedule_matches_reference_endpoints() {
        let initial = 0.1;
        assert_eq!(surface_normal_weight_at_step(0.0, 42, 100), 0.0);
        assert_eq!(surface_normal_weight_at_step(initial, 0, 100), initial);
        let final_weight = surface_normal_weight_at_step(initial, 99, 100);
        assert!((final_weight - 0.01).abs() < 1.0e-7);
        let midpoint = surface_normal_weight_at_step(initial, 50, 101);
        assert!((midpoint - initial * 0.1_f32.sqrt()).abs() < 1.0e-7);
        assert_eq!(
            surface_normal_weight_at_step(initial, 1000, 100),
            final_weight
        );
    }

    #[test]
    fn interpenetration_sampling_is_stratified_and_resume_stable() {
        let points = (0..4)
            .map(|index| glam::Vec4::new(index as f32, 0.0, 0.0, 1.0))
            .collect::<Vec<_>>();
        let model = vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: None,
            adjacency: Some(vol::Adjacency {
                neighbors: vec![1, 0, 2, 1, 3, 2],
                offsets: vec![0, 1, 3, 5, 6],
            }),
            radii: Some(vec![1.0; points.len()]),
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
            points,
        };
        let initial_weight = 1.0e-4;
        let step = 7;
        let total_steps = 100;
        let mut first = InterpenetrationBatch::new(&model, 4);
        first.prepare(&model, step, total_steps, initial_weight);
        let mut resumed = InterpenetrationBatch::new(&model, 4);
        resumed.prepare(&model, step, total_steps, initial_weight);

        assert_eq!(first.edge_a, resumed.edge_a);
        assert_eq!(first.edge_b, resumed.edge_b);
        assert_eq!(first.edge_direction, resumed.edge_direction);
        assert_eq!(first.edge_scale, resumed.edge_scale);

        let weight = interpenetration_weight_at_step(initial_weight, step, total_steps);
        let relative_scales = first
            .edge_scale
            .iter()
            .map(|&scale| (scale / weight).round() as usize)
            .collect::<Vec<_>>();
        assert_eq!(relative_scales, [1, 2, 1, 2]);
        assert!((first.edge_scale.iter().sum::<f32>() - 6.0 * weight).abs() < 1.0e-10);
        for ((&a, &b), direction) in first
            .edge_a
            .iter()
            .zip(first.edge_b.iter())
            .zip(first.edge_direction.chunks_exact(3))
        {
            let expected = (model.points[a as usize] - model.points[b as usize])
                .truncate()
                .normalize();
            assert_eq!(direction, expected.to_array());
        }
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
    fn relative_lr_groups_separate_legacy_and_radfoam_ratios() {
        assert_eq!(
            relative_lr_multipliers(LrGroups::Legacy, 0, 0.03),
            RelativeLrMultipliers {
                position: 0.03,
                sh_dc: 1.0,
                sh_rest: 1.0,
            }
        );
        assert_eq!(
            relative_lr_multipliers(LrGroups::Legacy, 3, 0.03),
            RelativeLrMultipliers {
                position: 0.03,
                sh_dc: 0.1,
                sh_rest: 0.1,
            }
        );
        assert_eq!(
            relative_lr_multipliers(LrGroups::RadFoamV1Relative, 3, 0.03),
            RelativeLrMultipliers {
                position: 0.002,
                sh_dc: 0.05,
                sh_rest: 0.005,
            }
        );
    }

    #[test]
    fn sh_learning_rate_multipliers_reach_bare_dc_parameter_names() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!(
                "skipping sh_learning_rate_multipliers_reach_bare_dc_parameter_names: no GPU"
            );
            return;
        };
        let mut graph = mn::Graph::new();
        let dc = graph.parameter("sh_r", &[1, 1]);
        let rest = graph.parameter("sh_r_rest", &[1, 3]);
        let dc_loss = graph.mean_all(dc);
        let rest_loss = graph.mean_all(rest);
        let loss = graph.add(dc_loss, rest_loss);
        graph.set_outputs(vec![loss]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        session.set_parameter("sh_r", &[0.0]);
        session.set_parameter("sh_r_rest", &[0.0; 3]);
        session.set_adam(0.1, 0.0, 0.0, 1.0e-8);
        set_sh_lr_multipliers(&mut session, 1, 0.25, 0.5);
        session.step();
        session.wait();

        let mut actual_dc = [0.0];
        let mut actual_rest = [0.0; 3];
        session.read_param("sh_r", &mut actual_dc);
        session.read_param("sh_r_rest", &mut actual_rest);
        assert!((actual_dc[0] + 0.025).abs() < 1.0e-5, "{actual_dc:?}");
        assert!(
            actual_rest
                .iter()
                .all(|value| (*value + 0.05).abs() < 1.0e-5),
            "{actual_rest:?}"
        );
    }

    #[test]
    fn legacy_component_sh_checkpoint_migrates_values_and_moments() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping legacy SH checkpoint migration test: no GPU");
            return;
        };
        let (n_cells, sh_degree) = (2usize, 1usize);
        let num_components = (1 + sh_degree) * (1 + sh_degree);
        let value = |channel: usize, component: usize, cell: usize| {
            (100 * channel + 10 * component + cell) as f32 * 0.01
        };

        let mut legacy_graph = mn::Graph::new();
        let mut legacy_parameters = Vec::new();
        let mut legacy_loss = None;
        for channel in ["sh_r", "sh_g", "sh_b"] {
            for component in 0..num_components {
                let parameter =
                    legacy_graph.parameter(&parameter_name(channel, component), &[n_cells, 1]);
                let term = legacy_graph.mean_all(parameter);
                legacy_loss = Some(match legacy_loss {
                    Some(loss) => legacy_graph.add(loss, term),
                    None => term,
                });
                legacy_parameters.push((channel, component));
            }
        }
        legacy_graph.set_outputs(vec![legacy_loss.unwrap()]);
        let (mut legacy_session, _) = mn::build(
            &legacy_graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu.clone()),
                ..Default::default()
            },
        );
        for &(channel, component) in &legacy_parameters {
            let channel_index = match channel {
                "sh_r" => 0,
                "sh_g" => 1,
                "sh_b" => 2,
                _ => unreachable!(),
            };
            let values: Vec<f32> = (0..n_cells)
                .map(|cell| value(channel_index, component, cell))
                .collect();
            let name = parameter_name(channel, component);
            legacy_session.set_parameter(&name, &values);
            legacy_session.write_adam_m(
                &name,
                &values.iter().map(|entry| entry + 1.0).collect::<Vec<_>>(),
            );
            legacy_session.write_adam_v(
                &name,
                &values.iter().map(|entry| entry + 2.0).collect::<Vec<_>>(),
            );
        }
        legacy_session.set_adam_step_count(17);
        let checkpoint = std::env::temp_dir().join(format!(
            "blade-volume-legacy-sh-{}.safetensors",
            std::process::id()
        ));
        legacy_session.save_checkpoint(&checkpoint).unwrap();
        drop(legacy_session);

        let mut packed_graph = mn::Graph::new();
        let packed_parameters = declare_sh_parameters(&mut packed_graph, n_cells, num_components);
        let mut packed_loss = None;
        for channel in &packed_parameters {
            for parameter in [Some(channel.dc), channel.rest].into_iter().flatten() {
                let term = packed_graph.mean_all(parameter);
                packed_loss = Some(match packed_loss {
                    Some(loss) => packed_graph.add(loss, term),
                    None => term,
                });
            }
        }
        packed_graph.set_outputs(vec![packed_loss.unwrap()]);
        let (mut packed_session, _) = mn::build(
            &packed_graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        assert!(
            load_optimizer_checkpoint(&mut packed_session, &checkpoint, sh_degree, n_cells,)
                .unwrap()
        );
        assert_eq!(packed_session.adam_step_count(), 17);

        for (channel_index, channel) in ["sh_r", "sh_g", "sh_b"].iter().enumerate() {
            let expected_dc: Vec<f32> = (0..n_cells)
                .map(|cell| value(channel_index, 0, cell))
                .collect();
            let mut actual_dc = vec![0.0_f32; n_cells];
            packed_session.read_param(channel, &mut actual_dc);
            assert_eq!(actual_dc, expected_dc);

            let mut expected_rest = vec![0.0_f32; n_cells * (num_components - 1)];
            for cell in 0..n_cells {
                for component in 1..num_components {
                    expected_rest[cell * (num_components - 1) + component - 1] =
                        value(channel_index, component, cell);
                }
            }
            let rest_name = sh_rest_parameter_name(channel);
            let mut actual_rest = vec![0.0_f32; expected_rest.len()];
            let mut actual_m = vec![0.0_f32; expected_rest.len()];
            let mut actual_v = vec![0.0_f32; expected_rest.len()];
            packed_session.read_param(&rest_name, &mut actual_rest);
            packed_session.read_adam_m(&rest_name, &mut actual_m);
            packed_session.read_adam_v(&rest_name, &mut actual_v);
            assert_eq!(actual_rest, expected_rest);
            assert_eq!(
                actual_m,
                expected_rest
                    .iter()
                    .map(|entry| entry + 1.0)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                actual_v,
                expected_rest
                    .iter()
                    .map(|entry| entry + 2.0)
                    .collect::<Vec<_>>()
            );
        }
        std::fs::remove_file(checkpoint).unwrap();
    }

    #[test]
    fn radfoam_v1_parameter_schedule_matches_reference_boundaries() {
        let at_zero = radfoam_v1_lrs(0, 20_000);
        assert_eq!(at_zero.density, 0.1);
        assert_eq!(at_zero.position, 2.0e-4);
        assert_eq!(at_zero.sh_dc, 5.0e-3);
        assert_eq!(at_zero.sh_rest, 5.0e-3);

        let after_first_update = radfoam_v1_lrs(1, 20_000);
        assert_eq!(after_first_update.density, 0.0);
        assert_eq!(after_first_update.sh_rest, 0.0);

        let density_warm = radfoam_v1_lrs(2_001, 20_000);
        assert!((density_warm.density - 0.1).abs() < 1.0e-7);
        let sh_warm = radfoam_v1_lrs(4_001, 20_000);
        assert!((sh_warm.sh_rest - 5.0e-4).abs() < 1.0e-8);

        let position_last = radfoam_v1_lrs(18_001, 20_000);
        assert!((position_last.position - 5.0e-6).abs() < 1.0e-9);
        assert_eq!(radfoam_v1_lrs(18_002, 20_000).position, 0.0);

        let final_rates = radfoam_v1_lrs(20_001, 20_000);
        assert!((final_rates.density - 0.01).abs() < 1.0e-7);
        assert!((final_rates.sh_dc - 5.0e-4).abs() < 1.0e-8);
        assert!((final_rates.sh_rest - 5.0e-5).abs() < 1.0e-9);
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
            topology_period: 51,
            topology_steps_since_update: 17,
            densify_initial_points: 50_000,
            densify_steps_since_update: 23,
            densify_next_after: 517,
            densification_round: 2,
        };
        let encoded = encode_training_state(state);
        assert_eq!(decode_training_state(&encoded).unwrap(), state);
        assert!(decode_training_state(&(encoded + "unexpected 1\n")).is_err());
    }

    #[test]
    fn legacy_training_state_decodes_without_topology_phase() {
        let encoded = "blade-volume-training-state-v1\n\
                       step 9000\n\
                       cycle 94\n\
                       sampling_rng 11\n\
                       quantile_rng 22\n\
                       densify_rng 33\n";
        let state = decode_training_state(encoded).unwrap();
        assert_eq!(state.step, 9_000);
        assert_eq!(state.topology_period, 0);
        assert_eq!(state.topology_steps_since_update, 0);
        assert_eq!(state.densify_initial_points, 0);
    }

    #[test]
    fn v2_training_state_decodes_without_densification_phase() {
        let encoded = "blade-volume-training-state-v2\n\
                       step 9000\n\
                       cycle 94\n\
                       sampling_rng 11\n\
                       quantile_rng 22\n\
                       densify_rng 33\n\
                       topology_period 51\n\
                       topology_steps_since_update 17\n";
        let state = decode_training_state(encoded).unwrap();
        assert_eq!(state.step, 9_000);
        assert_eq!(state.topology_period, 51);
        assert_eq!(state.topology_steps_since_update, 17);
        assert_eq!(state.densify_initial_points, 0);
        assert_eq!(state.densify_next_after, 0);
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
            topology_period: 101,
            topology_steps_since_update: 42,
            densify_initial_points: 0,
            densify_steps_since_update: 0,
            densify_next_after: 0,
            densification_round: 0,
        };
        let path = save_training_state(&model_path, state).unwrap();
        assert_eq!(load_training_state(&model_path).unwrap(), state);
        assert!(!model_path.with_extension("trainstate.tmp").exists());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn radfoam_v1_topology_cadence_matches_reference_counter_order() {
        let mut state = TopologyCadenceState::radfoam_v1_initial();
        for gap in [1, 3, 5, 7, 9, 11] {
            assert_eq!(state.steps_until_update(), gap);
            assert!(state.advance(gap));
            assert_eq!(state.steps_since_update, 1);
        }

        let mut state = TopologyCadenceState::radfoam_v1_initial();
        for period in (1..=99).step_by(2) {
            assert_eq!(state.period, period);
            let gap = state.steps_until_update();
            assert_eq!(gap, period);
            assert!(state.advance(gap));
        }
        assert_eq!(state.period, 101);
        assert_eq!(state.steps_until_update(), 101);
        assert!(state.advance(101));
        assert_eq!(state.period, 101);

        state.steps_since_update = 42;
        state.reset_period_after_densification();
        assert_eq!(state.period, 1);
        assert_eq!(state.steps_since_update, 42);
        assert_eq!(state.steps_until_update(), 1);
    }

    #[test]
    fn densify_schedule_is_independent_of_geometry_boundaries() {
        let config = DensifyConfig {
            warmup: 2000,
            every: 500,
            ..DensifyConfig::default()
        };
        assert_eq!(steps_until_fixed_densify(config, 0), 2000);
        assert_eq!(steps_until_fixed_densify(config, 1900), 100);
        assert!(fixed_densify_due(config, 2000));
        assert_eq!(fixed_densification_round(config, 2000), 0);
        for geometry_boundary in [2100, 2200, 2300, 2400] {
            assert!(!fixed_densify_due(config, geometry_boundary));
            assert_eq!(
                steps_until_fixed_densify(config, geometry_boundary),
                2500 - geometry_boundary,
            );
        }
        assert!(fixed_densify_due(config, 2500));
        assert_eq!(fixed_densification_round(config, 2500), 1);
        assert_eq!(steps_until_fixed_densify(config, 2500), 500);
    }

    #[test]
    fn radfoam_v1_densify_cadence_matches_reference_counter_order() {
        let config = DensifyConfig {
            schedule: DensifySchedule::RadFoamV1,
            fraction: 0.15,
            warmup: 2_000,
            target_points: 200_000,
            densify_until: 11_000,
            ..DensifyConfig::default()
        };
        let mut state = DensifyCadenceState::radfoam_v1_initial(50_000);
        assert_eq!(state.steps_until_update(0, config.warmup), 2_000);
        assert!(state.advance(0, 2_000, config.warmup));
        assert_eq!(state.steps_since_update, 1);
        assert_eq!(state.round, 0);

        state.finish_round(config, 57_500);
        assert_eq!(state.next_after, 517);
        assert_eq!(state.steps_since_update, 0);
        assert_eq!(state.round, 1);
        assert!(!state.advance(2_000, 516, config.warmup));
        assert!(state.advance(2_516, 1, config.warmup));
    }

    #[test]
    fn radfoam_v1_densify_interval_has_reference_floor_and_stop_gate() {
        let config = DensifyConfig {
            schedule: DensifySchedule::RadFoamV1,
            fraction: 0.15,
            warmup: 2_000,
            target_points: 2_097_152,
            densify_until: 11_000,
            ..DensifyConfig::default()
        };
        assert_eq!(radfoam_v1_next_densify_after(config, 131_072, 150_733), 103,);
        assert_eq!(radfoam_v1_next_densify_after(config, 131_072, 140_000), 100,);
        assert!(radfoam_v1_densify_active(config, 1_887_436));
        assert!(!radfoam_v1_densify_active(config, 1_887_437));
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
            config,
            100_000,
            0,
            10_000,
            16_000,
            DensifyCadenceState::disabled(),
        ));
        assert!(!segment_preserves_densify_accumulator(
            config,
            100_000,
            0,
            10_100,
            16_000,
            DensifyCadenceState::disabled(),
        ));
        assert!(segment_preserves_densify_accumulator(
            config,
            200_000,
            0,
            10_100,
            16_000,
            DensifyCadenceState::disabled(),
        ));
        assert!(segment_preserves_densify_accumulator(
            config,
            100_000,
            0,
            11_000,
            16_000,
            DensifyCadenceState::disabled(),
        ));
        assert!(segment_preserves_densify_accumulator(
            config,
            100_000,
            0,
            16_000,
            16_000,
            DensifyCadenceState::disabled(),
        ));
    }

    #[test]
    fn bounded_radfoam_v1_invocation_requires_the_next_dynamic_boundary() {
        let config = DensifyConfig {
            schedule: DensifySchedule::RadFoamV1,
            fraction: 0.15,
            warmup: 2_000,
            target_points: 200_000,
            densify_until: 11_000,
            ..DensifyConfig::default()
        };
        let initial = DensifyCadenceState::radfoam_v1_initial(50_000);
        assert!(segment_preserves_densify_accumulator(
            config, 50_000, 0, 2_000, 16_000, initial,
        ));
        assert!(!segment_preserves_densify_accumulator(
            config, 50_000, 0, 2_100, 16_000, initial,
        ));

        let mut after_first = initial;
        assert!(after_first.advance(0, 2_000, config.warmup));
        after_first.finish_round(config, 57_500);
        assert!(segment_preserves_densify_accumulator(
            config,
            57_500,
            2_000,
            2_517,
            16_000,
            after_first,
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
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
            &mut g,
            n_cells,
            p,
            max_steps,
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            false,
            false,
            false,
            false,
            false,
            false,
            ColorLoss::L1,
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

        // A nearly parallel face can have dot(normal, ray) = -ε exactly.
        // The old `denominator + ε` regularization cancelled to zero here
        // and injected NaNs into the position table before masking/clamping.
        session.set_parameter("positions", &positions);
        session.set_input_u32("next_cell_indices", &[1]);
        session.set_input("ray_dir_per_pixel", &[-2.0e-6, 0.0, 1.0]);
        session.step();
        session.wait();
        session.read_param_grad("positions", &mut position_grad);
        assert!(position_grad.iter().all(|value| value.is_finite()));
        let mut updated_positions = vec![0.0_f32; positions.len()];
        session.read_param("positions", &mut updated_positions);
        assert!(updated_positions.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn padded_weighted_steps_have_zero_loss_and_parameter_gradient() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping padded weighted path test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![1.0; model.points.len()]);
        model.surface_normals = Some(vec![glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.0; model.points.len()]);
        model.surface_color_coefficients =
            Some(vec![
                3.0;
                model.points.len() * vol::SURFACE_COLOR_COMPONENTS * 3
            ]);
        let n_cells = model.points.len();
        let mut graph = mn::Graph::new();
        build_volumetric_graph(
            &mut graph,
            n_cells,
            1,
            2,
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            true,
            true,
            true,
            true,
            false,
            false,
            ColorLoss::SmoothL1,
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
        session.set_input_u32("cell_indices", &[1, 2]);
        session.set_input_u32("previous_cell_indices", &[0, 1]);
        session.set_input_u32("next_cell_indices", &[2, 3]);
        session.set_input("recorded_dt", &[10.0, 20.0]);
        session.set_input("dt_reference_tangent", &[0.0, 0.0]);
        session.set_input("dt_grad_previous", &[1.0; 8]);
        session.set_input("dt_grad_current", &[1.0; 8]);
        session.set_input("dt_grad_next", &[1.0; 8]);
        session.set_input("dt_grad_surface_normal", &[1.0; 8]);
        session.set_input("surface_normal_loss_scale", &[0.0]);
        session.set_input("mask", &[0.0; 2]);
        session.set_input("ray_origin", &[0.1, 0.1, -1.0]);
        session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
        session.set_input_u32("pixel_idx_per_step", &[0, 0]);
        session.set_input_u32("view_idx", &[0]);
        session.set_input("labels", &[0.0; 3]);
        session.set_adam(0.0, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();

        assert_eq!(session.read_loss(), 0.0);
        for (name, count) in [
            ("log_density", n_cells),
            ("positions", 3 * n_cells),
            ("log_radii", n_cells),
            ("surface_normals", 3 * n_cells),
            ("surface_offsets", n_cells),
            (
                "surface_color_coefficients",
                vol::SURFACE_COLOR_COMPONENTS * 3 * n_cells,
            ),
            ("sh_r", n_cells),
            ("sh_g", n_cells),
            ("sh_b", n_cells),
        ] {
            let mut gradient = vec![f32::NAN; count];
            session.read_param_grad(name, &mut gradient);
            assert!(
                gradient.iter().all(|&value| value == 0.0),
                "{name} gradient is {gradient:?}"
            );
        }
    }

    #[test]
    fn position_gradient_matches_central_finite_difference_for_fixed_path() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping position gradient check: no GPU");
            return;
        };
        let model = tiny_model();
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
            [0.0; 3],
            false,
            false,
            false,
            false,
            false,
            false,
            ColorLoss::L1,
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
    fn weighted_graph_uses_recorded_stable_linearization() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping weighted graph Jacobian test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![1.0; model.points.len()]);
        model.surface_normals = Some(vec![glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.15; model.points.len()]);
        model.compute_adjacency_default();
        let n_cells = model.points.len();
        let mut g = mn::Graph::new();
        let vg = build_volumetric_graph(
            &mut g,
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
            [0.0; 3],
            true,
            true,
            true,
            false,
            false,
            false,
            ColorLoss::L1,
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
        session.set_input("dt_grad_previous", &[0.0; 4]);
        let current_jacobian = [0.25, -0.5, 0.75, 1.25];
        let surface_normal_jacobian = [0.4, -0.2, 0.75, -0.6];
        session.set_input("dt_grad_current", &current_jacobian);
        session.set_input("dt_grad_next", &[0.0; 4]);
        session.set_input("dt_grad_surface_normal", &surface_normal_jacobian);
        session.set_input("surface_normal_loss_scale", &[0.0]);
        session.set_input("mask", &[1.0]);
        let ray_origin = [0.1, 0.1, -1.0];
        session.set_input("ray_origin", &ray_origin);
        session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
        session.set_input_u32("pixel_idx_per_step", &[0]);
        session.set_input_u32("view_idx", &[0]);
        session.set_input("labels", &[0.4, 0.6, 0.8]);
        let initial_positions: Vec<f32> = model
            .points
            .iter()
            .flat_map(|point| [point.x, point.y, point.z])
            .collect();
        let initial_radii = model.radii.as_deref().expect("fixture has radii");
        let linear_term =
            |positions: &[f32], radii: &[f32], surface_normal: glam::Vec3, surface_offset: f32| {
                current_jacobian[0] * (positions[0] - ray_origin[0])
                    + current_jacobian[1] * (positions[1] - ray_origin[1])
                    + current_jacobian[2] * (positions[2] - ray_origin[2])
                    + current_jacobian[3] * radii[0]
                    + surface_normal_jacobian[0] * surface_normal.x
                    + surface_normal_jacobian[1] * surface_normal.y
                    + surface_normal_jacobian[2] * surface_normal.z
                    + surface_normal_jacobian[3] * surface_offset
            };
        session.set_input("recorded_dt", &[7.5]);
        session.set_input(
            "dt_reference_tangent",
            &[linear_term(
                &initial_positions,
                initial_radii,
                glam::Vec3::Z,
                model.surface_offsets.as_ref().unwrap()[0],
            )],
        );
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
        let mut normal_grad = vec![0.0_f32; n_cells * 3];
        session.read_param_grad("surface_normals", &mut normal_grad);
        assert!((normal_grad[0] - 0.4).abs() < 1.0e-5);
        assert!((normal_grad[1] + 0.2).abs() < 1.0e-5);
        assert!(normal_grad[2].abs() < 1.0e-5);
        assert!(normal_grad[3..].iter().all(|&value| value == 0.0));
        let mut offset_grad = vec![0.0_f32; n_cells];
        session.read_param_grad("surface_offsets", &mut offset_grad);
        assert!((offset_grad[0] + 0.6).abs() < 1.0e-5);
        assert!(offset_grad[1..].iter().all(|&value| value == 0.0));

        let mut moved_positions = initial_positions;
        moved_positions[0] += 2.0;
        let mut moved_radii = model.radii.clone().expect("fixture has radii");
        moved_radii[0] += 0.4;
        let moved_log_radii: Vec<f32> = moved_radii
            .iter()
            .map(|&radius| inv_radius_activation(radius))
            .collect();
        let moved_normal = glam::Vec3::new(0.3, 0.0, 1.0).normalize();
        let mut moved_offsets = model.surface_offsets.clone().unwrap();
        moved_offsets[0] += 0.2;
        let mut moved_normals = vec![0.0_f32; n_cells * 3];
        for values in moved_normals.chunks_exact_mut(3) {
            values.copy_from_slice(glam::Vec3::Z.as_ref());
        }
        moved_normals[..3].copy_from_slice(moved_normal.as_ref());
        session.set_parameter("positions", &moved_positions);
        session.set_parameter("log_radii", &moved_log_radii);
        session.set_parameter("surface_normals", &moved_normals);
        session.set_parameter("surface_offsets", &moved_offsets);
        session.step();
        session.wait();
        let outputs = session.read_output(1);
        let expected = 8.5
            + glam::Vec3::from_slice(&surface_normal_jacobian[..3])
                .dot(moved_normal - glam::Vec3::Z)
            + surface_normal_jacobian[3] * 0.2;
        assert!(
            (outputs[0] - expected).abs() < 1.0e-5,
            "outputs={outputs:?}, expected={expected}"
        );

        session.set_input("recorded_dt", &[9.25]);
        session.set_input(
            "dt_reference_tangent",
            &[linear_term(
                &moved_positions,
                &moved_radii,
                moved_normal,
                moved_offsets[0],
            )],
        );
        session.step();
        session.wait();
        let outputs = session.read_output(1);
        assert!((outputs[0] - 9.25).abs() < 1.0e-5, "outputs={outputs:?}");
    }

    #[test]
    fn surface_only_graph_uses_surface_reference_tangent() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-only graph Jacobian test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![1.0; model.points.len()]);
        model.surface_normals = Some(vec![glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.15; model.points.len()]);
        let mut graph = mn::Graph::new();
        let vg = build_volumetric_graph_with_options(
            &mut graph,
            model.points.len(),
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
            [0.0; 3],
            true,
            true,
            VolumetricGraphOptions {
                use_surface_normal_loss: false,
                train_positions: false,
                train_radii: false,
                use_surface_detail: false,
            },
            true,
            false,
            false,
            false,
            ColorLoss::L1,
        );
        let dt_output = graph.reshape(vg.dt_from_positions, &[1]);
        graph.set_outputs(vec![dt_output]);
        let (mut session, _) = mn::build(
            &graph,
            mn::SessionConfig {
                mode: mn::Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        );
        upload_model_parameters(&mut session, &model, 0.0);
        let gradient = [0.4_f32, -0.2, 0.75, -0.6];
        let reference = gradient[2] + gradient[3] * 0.15;
        session.set_input_u32("cell_indices", &[0]);
        session.set_input("recorded_dt", &[7.5]);
        session.set_input("dt_reference_tangent", &[reference]);
        session.set_input("dt_grad_surface_normal", &gradient);
        session.set_input("mask", &[1.0]);
        session.set_adam(0.0, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();

        let output = session.read_output(1)[0];
        assert!((output - 7.5).abs() < 1.0e-5, "output={output}");
        assert!(!session.has_param_grad("positions"));
        assert!(!session.has_param_grad("log_radii"));
        let mut normal_gradient = vec![0.0_f32; model.points.len() * 3];
        session.read_param_grad("surface_normals", &mut normal_gradient);
        assert!((normal_gradient[0] - gradient[0]).abs() < 1.0e-5);
        assert!((normal_gradient[1] - gradient[1]).abs() < 1.0e-5);
        assert!(normal_gradient[2].abs() < 1.0e-5);
        let mut offset_gradient = vec![0.0_f32; model.points.len()];
        session.read_param_grad("surface_offsets", &mut offset_gradient);
        assert!((offset_gradient[0] - gradient[3]).abs() < 1.0e-5);

        let moved_normal = glam::Vec3::new(0.3, 0.0, 1.0).normalize();
        let mut moved_normals = vec![0.0_f32; model.points.len() * 3];
        for normal in moved_normals.chunks_exact_mut(3) {
            normal.copy_from_slice(glam::Vec3::Z.as_ref());
        }
        moved_normals[..3].copy_from_slice(moved_normal.as_ref());
        let mut moved_offsets = model.surface_offsets.clone().unwrap();
        moved_offsets[0] += 0.2;
        session.set_parameter("surface_normals", &moved_normals);
        session.set_parameter("surface_offsets", &moved_offsets);
        session.step();
        session.wait();
        let expected = 7.5
            + glam::Vec3::from_slice(&gradient[..3]).dot(moved_normal - glam::Vec3::Z)
            + gradient[3] * 0.2;
        let output = session.read_output(1)[0];
        assert!(
            (output - expected).abs() < 1.0e-5,
            "output={output}, expected={expected}"
        );
    }

    #[test]
    fn surface_normal_loss_penalizes_normals_pointing_along_the_camera_ray() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-normal loss test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![1.0; model.points.len()]);
        model.surface_normals = Some(vec![-glam::Vec3::Z; model.points.len()]);
        model.compute_adjacency_default();
        let mut graph = mn::Graph::new();
        build_volumetric_graph(
            &mut graph,
            model.points.len(),
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
            [0.0; 3],
            true,
            true,
            false,
            false,
            false,
            false,
            ColorLoss::L1,
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
        session.set_input_u32("cell_indices", &[0]);
        session.set_input_u32("previous_cell_indices", &[0]);
        session.set_input_u32("next_cell_indices", &[0]);
        session.set_input("recorded_dt", &[1.0]);
        session.set_input("dt_reference_tangent", &[0.0]);
        session.set_input("dt_grad_previous", &[0.0; 4]);
        session.set_input("dt_grad_current", &[0.0; 4]);
        session.set_input("dt_grad_next", &[0.0; 4]);
        session.set_input("dt_grad_surface_normal", &[0.0; 4]);
        session.set_input("mask", &[1.0]);
        session.set_input("ray_origin", &[0.0, 0.0, -1.0]);
        session.set_input("ray_dir_per_pixel", &[0.0, 0.0, 1.0]);
        session.set_input_u32("pixel_idx_per_step", &[0]);
        session.set_input_u32("view_idx", &[0]);
        session.set_input("labels", &[0.0; 3]);
        let scale = 0.2;
        session.set_input("surface_normal_loss_scale", &[scale]);
        session.set_adam(0.0, 0.9, 0.999, 1.0e-8);

        session.step();
        session.wait();
        let facing_camera = session.read_loss();

        let mut normals = vec![0.0_f32; model.points.len() * 3];
        for values in normals.chunks_exact_mut(3) {
            values.copy_from_slice(&glam::Vec3::Z.to_array());
        }
        session.set_parameter("surface_normals", &normals);
        session.step();
        session.wait();
        let pointing_along_ray = session.read_loss();

        let expected_penalty = scale * (1.0 - (-model.points[0].w).exp());
        assert!(
            ((pointing_along_ray - facing_camera) - expected_penalty).abs() < 1.0e-5,
            "facing={facing_camera}, along={pointing_along_ray}, expected={expected_penalty}"
        );
    }

    #[test]
    fn graph_composites_the_configured_background() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
                false,
                false,
                false,
                false,
                false,
                ColorLoss::L1,
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
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
                false,
                false,
                false,
                false,
                false,
                ColorLoss::L1,
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
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
                false,
                false,
                false,
                false,
                false,
                ColorLoss::L1,
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
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
            target_alpha: None,
            width: 1,
            height: 1,
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
    fn masked_opacity_separates_empty_and_opaque_rays() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping masked_opacity_separates_empty_and_opaque_rays: no GPU");
            return;
        };
        const SH_C0: f32 = 0.282_094_8;
        let mut initial = tiny_model();
        for point in initial.points.iter_mut() {
            point.w = 1.0;
        }
        for coefficients in initial.sh_coefficients.chunks_exact_mut(3) {
            coefficients.fill(-0.5 / SH_C0);
        }
        let camera = vol::CameraParams {
            cam_position: [0.05, 0.05, -1.0],
            depth: 10.0,
            cam_orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [0.5, 0.5],
            principal: [0.0, 0.0],
        };
        let train = |target_alpha, mut model: vol::PointCloudModel| {
            fit_appearance_multi_view(
                &mut model,
                &[ViewSupervision {
                    camera,
                    target_rgb: vec![0.0; 3],
                    target_alpha: Some(vec![target_alpha]),
                    width: 1,
                    height: 1,
                }],
                1,
                1,
                16,
                AppearanceFitConfig {
                    learning_rate: 0.05,
                    epochs: 40,
                    opacity_weight: 1.0,
                    softplus_beta: 10.0,
                    ..AppearanceFitConfig::default()
                },
                gpu.clone(),
            );
            model.points.iter().map(|point| point.w).sum::<f32>() / model.points.len() as f32
        };
        let empty_density = train(0.0, initial.clone());
        let opaque_density = train(1.0, initial);
        assert!(
            empty_density < opaque_density,
            "empty density {empty_density} must fall below opaque density {opaque_density}"
        );
    }

    #[test]
    fn position_training_rebuilds_valid_topology() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
            target_alpha: None,
            width: 1,
            height: 1,
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
    fn position_training_supports_radfoam_v1_topology_cadence() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping position_training_supports_radfoam_v1_topology_cadence: no GPU");
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
            target_alpha: None,
            width: 1,
            height: 1,
        };
        fit_appearance_multi_view(
            &mut model,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                learning_rate: 0.01,
                epochs: 5,
                position_lr_ratio: 1.0,
                geometry_rebuild_every: 0,
                geometry_rebuild_schedule: GeometryRebuildSchedule::RadFoamV1,
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        model.validate().unwrap();
    }

    #[test]
    fn densification_rebuilds_and_continues_training() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
            target_alpha: None,
            width: 1,
            height: 1,
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
                position_lr_ratio: 1.0,
                geometry_rebuild_every: 1,
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

    #[test]
    fn powerfoam_point_error_survives_densification_rebuild() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping PowerFoam densification rebuild test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![0.2; model.points.len()]);
        model.surface_normals = Some(vec![
            glam::Vec3::new(0.1, 0.0, -1.0).normalize();
            model.points.len()
        ]);
        model.surface_offsets = Some(vec![0.01; model.points.len()]);
        model.surface_color_coefficients =
            Some(vec![
                0.0;
                model.points.len() * vol::SURFACE_COLOR_COMPONENTS * 3
            ]);
        model.spherical_voronoi = Some(vol::SphericalVoronoi {
            axes: vec![4.0 * glam::Vec3::Z; model.points.len() * vol::SPHERICAL_VORONOI_SITES],
            colors: vec![glam::Vec3::ZERO; model.points.len() * vol::SPHERICAL_VORONOI_SITES],
        });
        model.compute_adjacency_default();
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: [0.0, 0.0, 0.0, 1.0],
                fov: [0.5, 0.5],
                principal: [0.0, 0.0],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            target_alpha: None,
            width: 1,
            height: 1,
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
                position_lr_ratio: 1.0,
                radius_lr_ratio: 0.01,
                surface_normal_lr_ratio: 0.1,
                surface_offset_lr_ratio: 0.1,
                surface_color_lr_ratio: 0.1,
                spherical_voronoi_axis_lr_ratio: 0.1,
                spherical_voronoi_color_lr_ratio: 0.1,
                surface_normal_weight: 0.1,
                geometry_rebuild_every: 1,
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
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert_eq!(model.points.len(), 5);
        assert_eq!(model.radii.as_ref().unwrap().len(), 5);
        assert_eq!(model.surface_normals.as_ref().unwrap().len(), 5);
        assert_eq!(model.surface_offsets.as_ref().unwrap().len(), 5);
        assert_eq!(
            model.surface_color_coefficients.as_ref().unwrap().len(),
            5 * vol::SURFACE_COLOR_COMPONENTS * 3
        );
        assert_eq!(
            model.spherical_voronoi.as_ref().unwrap().axes.len(),
            5 * vol::SPHERICAL_VORONOI_SITES
        );
        assert_eq!(
            model.spherical_voronoi.as_ref().unwrap().colors.len(),
            5 * vol::SPHERICAL_VORONOI_SITES
        );
        assert!(model
            .surface_color_coefficients
            .as_ref()
            .unwrap()
            .iter()
            .any(|value| value.abs() > 1.0e-7));
        assert!(model
            .spherical_voronoi
            .as_ref()
            .unwrap()
            .colors
            .iter()
            .any(|color| color.abs().max_element() > 1.0e-7));
        assert!(model
            .surface_offsets
            .as_ref()
            .unwrap()
            .iter()
            .all(|offset| offset.is_finite()));
        assert!(model
            .surface_normals
            .as_ref()
            .unwrap()
            .iter()
            .all(|normal| (normal.length() - 1.0).abs() < 1.0e-5));
        model.validate().unwrap();
    }

    #[test]
    fn surface_detail_survives_densification_and_adam_remap() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping surface-detail densification test: no GPU");
            return;
        };
        let mut model = tiny_model();
        model.radii = Some(vec![0.2; model.points.len()]);
        model.surface_normals = Some(vec![-glam::Vec3::Z; model.points.len()]);
        model.surface_offsets = Some(vec![0.0; model.points.len()]);
        let detail_count = model.points.len() * vol::SURFACE_DETAIL_SITES;
        model.surface_detail = Some(vol::SurfaceDetail {
            offsets: (0..detail_count)
                .map(|index| {
                    let angle = std::f32::consts::TAU * (index % vol::SURFACE_DETAIL_SITES) as f32
                        / vol::SURFACE_DETAIL_SITES as f32;
                    0.1 * glam::Vec3::new(angle.cos(), angle.sin(), 0.0)
                })
                .collect(),
            heights: vec![0.0; detail_count],
            colors: vec![glam::Vec3::ZERO; detail_count],
        });
        model.compute_adjacency_default();
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: glam::Quat::IDENTITY.to_array(),
                fov: [0.5; 2],
                principal: [0.0; 2],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            target_alpha: None,
            width: 1,
            height: 1,
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
                surface_detail_offset_lr_ratio: 0.1,
                surface_detail_height_lr_ratio: 0.1,
                surface_detail_color_lr_ratio: 0.1,
                geometry_rebuild_every: 1,
                densify: Some(DensifyConfig {
                    every: 1,
                    fraction: 0.25,
                    warmup: 1,
                    target_points: 5,
                    prune: false,
                    ..DensifyConfig::default()
                }),
                ..AppearanceFitConfig::default()
            },
            gpu,
        );

        assert_eq!(losses.len(), 3);
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert_eq!(model.points.len(), 5);
        let detail = model.surface_detail.as_ref().unwrap();
        assert_eq!(detail.offsets.len(), 5 * vol::SURFACE_DETAIL_SITES);
        assert_eq!(detail.heights.len(), 5 * vol::SURFACE_DETAIL_SITES);
        assert_eq!(detail.colors.len(), 5 * vol::SURFACE_DETAIL_SITES);
        assert!(detail
            .colors
            .iter()
            .any(|color| color.abs().max_element() > 1.0e-7));
        model.validate().unwrap();
    }

    #[test]
    fn mixed_view_resume_matches_uninterrupted_training() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping mixed_view_resume_matches_uninterrupted_training: no GPU");
            return;
        };
        let cameras = [
            vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: [0.0, 0.0, 0.0, 1.0],
                fov: [0.5, 0.5],
                principal: [0.0, 0.0],
            },
            vol::CameraParams {
                cam_position: [-1.0, 0.05, 0.05],
                depth: 10.0,
                cam_orientation: [
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
                fov: [0.5, 0.5],
                principal: [0.0, 0.0],
            },
        ];
        let targets = [[0.9_f32, 0.2, 0.1], [0.1, 0.2, 0.9]];
        let views: Vec<_> = cameras
            .into_iter()
            .zip(targets)
            .map(|(camera, target)| ViewSupervision {
                camera,
                target_rgb: target.repeat(4),
                target_alpha: None,
                width: 2,
                height: 2,
            })
            .collect();
        let base = AppearanceFitConfig {
            learning_rate: 0.01,
            pixel_batch: Some(4),
            views_per_batch: 2,
            steps_per_view: 4,
            ..AppearanceFitConfig::default()
        };
        let checkpoint = std::env::temp_dir().join(format!(
            "blade-volume-mixed-view-resume-{}.ply",
            std::process::id()
        ));

        let mut uninterrupted = tiny_model();
        fit_appearance_multi_view(
            &mut uninterrupted,
            &views,
            2,
            2,
            16,
            base.clone(),
            gpu.clone(),
        );

        let mut first_segment = tiny_model();
        let first_outcome = fit_appearance_multi_view_outcome(
            &mut first_segment,
            &views,
            2,
            2,
            16,
            AppearanceFitConfig {
                checkpoint_path: Some(checkpoint.clone()),
                stop_after_steps: Some(4),
                ..base.clone()
            },
            gpu.clone(),
        );
        assert_eq!(
            first_outcome.endpoint_checkpoint.as_deref(),
            Some(checkpoint.as_path())
        );
        let training_state = load_training_state(&checkpoint).unwrap();
        assert_eq!(training_state.step, 4);

        let mut resumed = vol::io::try_load(checkpoint.to_str().unwrap()).unwrap();
        fit_appearance_multi_view(
            &mut resumed,
            &views,
            2,
            2,
            16,
            AppearanceFitConfig {
                checkpoint_path: Some(checkpoint.clone()),
                resume_step: 4,
                resume_state_path: Some(checkpoint.with_extension("safetensors")),
                resume_training_state: Some(training_state),
                ..base
            },
            gpu,
        );

        assert_eq!(uninterrupted.points, resumed.points);
        assert_eq!(uninterrupted.sh_coefficients, resumed.sh_coefficients);
        let uninterrupted_adjacency = uninterrupted.adjacency.as_ref().unwrap();
        let resumed_adjacency = resumed.adjacency.as_ref().unwrap();
        assert_eq!(uninterrupted_adjacency.offsets, resumed_adjacency.offsets);
        assert_eq!(
            uninterrupted_adjacency.neighbors,
            resumed_adjacency.neighbors
        );

        for path in [
            checkpoint.clone(),
            checkpoint.with_extension("safetensors"),
            checkpoint.with_extension("trainstate"),
            checkpoint.with_extension("ply.step"),
        ] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn oriented_powerfoam_resume_matches_uninterrupted_training() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!("skipping oriented PowerFoam resume test: no GPU");
            return;
        };
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: glam::Quat::IDENTITY.to_array(),
                fov: [0.5; 2],
                principal: [0.0; 2],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            target_alpha: None,
            width: 1,
            height: 1,
        };
        let make_model = || {
            let mut model = tiny_model();
            model.radii = Some(vec![0.2; model.points.len()]);
            model.surface_normals = Some(vec![
                glam::Vec3::new(0.2, 0.0, -1.0).normalize();
                model.points.len()
            ]);
            model.surface_offsets = Some(vec![0.01; model.points.len()]);
            model.surface_color_coefficients =
                Some(vec![
                    0.0;
                    model.points.len() * vol::SURFACE_COLOR_COMPONENTS * 3
                ]);
            let detail_count = model.points.len() * vol::SURFACE_DETAIL_SITES;
            model.surface_detail = Some(vol::SurfaceDetail {
                offsets: (0..detail_count)
                    .map(|index| {
                        let angle = std::f32::consts::TAU
                            * (index % vol::SURFACE_DETAIL_SITES) as f32
                            / vol::SURFACE_DETAIL_SITES as f32;
                        0.1 * glam::Vec3::new(angle.cos(), angle.sin(), 0.02)
                    })
                    .collect(),
                heights: vec![0.0; detail_count],
                colors: vec![glam::Vec3::ZERO; detail_count],
            });
            model.spherical_voronoi = Some(vol::SphericalVoronoi {
                axes: (0..model.points.len() * vol::SPHERICAL_VORONOI_SITES)
                    .map(|index| {
                        let site = index % vol::SPHERICAL_VORONOI_SITES;
                        4.0 * glam::Vec3::new(
                            if site & 1 == 0 { -1.0 } else { 1.0 },
                            if site & 2 == 0 { -1.0 } else { 1.0 },
                            if site & 4 == 0 { -1.0 } else { 1.0 },
                        )
                        .normalize()
                    })
                    .collect(),
                colors: vec![glam::Vec3::ZERO; model.points.len() * vol::SPHERICAL_VORONOI_SITES],
            });
            model.compute_adjacency_default();
            model
        };
        let base = AppearanceFitConfig {
            learning_rate: 0.01,
            pixel_batch: Some(1),
            steps_per_view: 4,
            surface_normal_lr_ratio: 0.1,
            surface_offset_lr_ratio: 0.1,
            surface_color_lr_ratio: 0.1,
            surface_detail_offset_lr_ratio: 0.1,
            surface_detail_height_lr_ratio: 0.1,
            surface_detail_color_lr_ratio: 0.1,
            spherical_voronoi_axis_lr_ratio: 0.1,
            spherical_voronoi_color_lr_ratio: 0.1,
            surface_normal_weight: 0.1,
            geometry_rebuild_every: 1,
            ..AppearanceFitConfig::default()
        };
        let checkpoint = std::env::temp_dir().join(format!(
            "blade-volume-oriented-resume-{}.ply",
            std::process::id()
        ));

        let mut uninterrupted = make_model();
        fit_appearance_multi_view(
            &mut uninterrupted,
            std::slice::from_ref(&view),
            1,
            1,
            16,
            base.clone(),
            gpu.clone(),
        );

        let mut first_segment = make_model();
        fit_appearance_multi_view(
            &mut first_segment,
            std::slice::from_ref(&view),
            1,
            1,
            16,
            AppearanceFitConfig {
                checkpoint_path: Some(checkpoint.clone()),
                stop_after_steps: Some(2),
                ..base.clone()
            },
            gpu.clone(),
        );
        let training_state = load_training_state(&checkpoint).unwrap();
        assert_eq!(training_state.step, 2);
        let mut resumed = vol::io::try_load(checkpoint.to_str().unwrap()).unwrap();
        fit_appearance_multi_view(
            &mut resumed,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                checkpoint_path: Some(checkpoint.clone()),
                resume_step: 2,
                resume_state_path: Some(checkpoint.with_extension("safetensors")),
                resume_training_state: Some(training_state),
                ..base
            },
            gpu,
        );

        assert_eq!(uninterrupted.points, resumed.points);
        assert_eq!(uninterrupted.sh_coefficients, resumed.sh_coefficients);
        assert_eq!(uninterrupted.radii, resumed.radii);
        assert_eq!(uninterrupted.surface_normals, resumed.surface_normals);
        assert_eq!(uninterrupted.surface_offsets, resumed.surface_offsets);
        assert_eq!(uninterrupted.surface_detail, resumed.surface_detail);
        assert_eq!(
            uninterrupted.surface_color_coefficients,
            resumed.surface_color_coefficients
        );
        assert_eq!(uninterrupted.spherical_voronoi, resumed.spherical_voronoi);
        let uninterrupted_adjacency = uninterrupted.adjacency.as_ref().unwrap();
        let resumed_adjacency = resumed.adjacency.as_ref().unwrap();
        assert_eq!(uninterrupted_adjacency.offsets, resumed_adjacency.offsets);
        assert_eq!(
            uninterrupted_adjacency.neighbors,
            resumed_adjacency.neighbors
        );
        resumed.validate().unwrap();

        for path in [
            checkpoint.clone(),
            checkpoint.with_extension("safetensors"),
            checkpoint.with_extension("trainstate"),
            checkpoint.with_extension("ply.step"),
        ] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn radfoam_v1_densification_resume_matches_uninterrupted_training() {
        let _gpu_test_guard = crate::fit::gpu_test_guard();
        let Some(gpu) = try_init_gpu() else {
            eprintln!(
                "skipping radfoam_v1_densification_resume_matches_uninterrupted_training: no GPU"
            );
            return;
        };
        let view = ViewSupervision {
            camera: vol::CameraParams {
                cam_position: [0.05, 0.05, -1.0],
                depth: 10.0,
                cam_orientation: [0.0, 0.0, 0.0, 1.0],
                fov: [0.5, 0.5],
                principal: [0.0, 0.0],
            },
            target_rgb: vec![0.9, 0.2, 0.1],
            target_alpha: None,
            width: 1,
            height: 1,
        };
        let base = AppearanceFitConfig {
            learning_rate: 0.01,
            epochs: 102,
            position_lr_ratio: 1.0,
            geometry_rebuild_every: 100,
            densify: Some(DensifyConfig {
                schedule: DensifySchedule::RadFoamV1,
                fraction: 0.25,
                warmup: 1,
                target_points: 6,
                densify_until: 101,
                prune: true,
                ..DensifyConfig::default()
            }),
            ..AppearanceFitConfig::default()
        };
        let stem = format!("blade-volume-dynamic-resume-{}", std::process::id());
        let segmented_path = std::env::temp_dir().join(format!("{stem}-segmented.ply"));

        let mut uninterrupted = tiny_model();
        fit_appearance_multi_view(
            &mut uninterrupted,
            std::slice::from_ref(&view),
            1,
            1,
            16,
            base.clone(),
            gpu.clone(),
        );

        let mut first_segment = tiny_model();
        fit_appearance_multi_view(
            &mut first_segment,
            std::slice::from_ref(&view),
            1,
            1,
            16,
            AppearanceFitConfig {
                checkpoint_path: Some(segmented_path.clone()),
                stop_after_steps: Some(1),
                ..base.clone()
            },
            gpu.clone(),
        );
        assert_eq!(first_segment.points.len(), 5);

        let mut resumed = vol::io::try_load(
            segmented_path
                .to_str()
                .expect("temporary checkpoint path is UTF-8"),
        )
        .unwrap();
        let training_state = load_training_state(&segmented_path).unwrap();
        assert_eq!(training_state.step, 1);
        assert_eq!(training_state.densification_round, 1);
        assert_eq!(training_state.densify_next_after, 100);
        fit_appearance_multi_view(
            &mut resumed,
            &[view],
            1,
            1,
            16,
            AppearanceFitConfig {
                checkpoint_path: Some(segmented_path.clone()),
                resume_step: 1,
                resume_state_path: Some(segmented_path.with_extension("safetensors")),
                resume_training_state: Some(training_state),
                ..base
            },
            gpu,
        );

        assert_eq!(uninterrupted.points, resumed.points);
        assert_eq!(uninterrupted.sh_coefficients, resumed.sh_coefficients);
        assert_eq!(uninterrupted.radii, resumed.radii);
        let uninterrupted_adjacency = uninterrupted.adjacency.as_ref().unwrap();
        let resumed_adjacency = resumed.adjacency.as_ref().unwrap();
        assert_eq!(uninterrupted_adjacency.offsets, resumed_adjacency.offsets);
        assert_eq!(
            uninterrupted_adjacency.neighbors,
            resumed_adjacency.neighbors,
        );
        assert_eq!(uninterrupted.points.len(), 6);

        for path in [
            segmented_path.clone(),
            segmented_path.with_extension("safetensors"),
            segmented_path.with_extension("trainstate"),
            segmented_path.with_extension("ply.step"),
        ] {
            let _ = std::fs::remove_file(path);
        }
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
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
            target_alpha: None,
            width: w,
            height: h,
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
        let _gpu_test_guard = crate::fit::gpu_test_guard();
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
                    target_alpha: None,
                    width: w,
                    height: h,
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
