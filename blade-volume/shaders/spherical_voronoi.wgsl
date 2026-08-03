// Dot-product Spherical Voronoi directional appearance.
//
// A raw axis carries both direction and temperature. Its logit for a unit
// view direction is dot(axis, direction); a stable softmax blends RGB values.
// This follows the published Spherical Voronoi definition and is intentionally
// distinct from PowerFoam's released chord-distance evaluator.

const SPHERICAL_VORONOI_SITES: u32 = 8u;

fn spherical_voronoi_evaluate(
    axes: array<vec3<f32>, SPHERICAL_VORONOI_SITES>,
    colors: array<vec3<f32>, SPHERICAL_VORONOI_SITES>,
    direction: vec3<f32>,
) -> vec3<f32> {
    var logits: array<f32, SPHERICAL_VORONOI_SITES>;
    var max_logit = -3.402823466e+38;
    for (var site = 0u; site < SPHERICAL_VORONOI_SITES; site += 1u) {
        let logit = dot(axes[site], direction);
        logits[site] = logit;
        max_logit = max(max_logit, logit);
    }

    var weight_sum = 0.0;
    var value_sum = vec3<f32>(0.0);
    for (var site = 0u; site < SPHERICAL_VORONOI_SITES; site += 1u) {
        let weight = exp(logits[site] - max_logit);
        weight_sum += weight;
        value_sum += weight * colors[site];
    }
    return value_sum / max(weight_sum, 1e-20);
}
