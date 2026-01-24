//! RadFoam tests (initial)
//!
//! These tests are intended to give us *some* correctness signals early so we can
//! iterate on the loader and shader pipeline without relying on visual inspection.
//!
//! Notes:
//! - We start with CPU-only loader invariants using an ASCII PLY fixture.
//! - GPU tests will be added once we decide on a readback strategy (rgba16f vs rgba32f).

use blade_volume as vol;

const TINY_ASCII_PLY: &str = "tests/data/radfoam_tiny_ascii.ply";

#[test]
fn radfoam_ascii_ply_loads_and_has_expected_shapes() {
    let model = vol::io::load_radfoam(TINY_ASCII_PLY);

    // Basic counts from the fixture
    assert_eq!(model.points.len(), 4, "fixture should contain 4 points");

    let adjacency = model
        .adjacency
        .as_ref()
        .expect("RadFoam model should have adjacency");
    assert_eq!(adjacency.offsets.len(), 5, "CSR offsets should be N+1");
    assert_eq!(
        adjacency.neighbors.len(),
        12,
        "fixture adjacency should contain 12 entries"
    );

    // CSR offsets in the fixture are: [0,3,6,9,12]
    assert_eq!(
        adjacency.offsets,
        vec![0, 3, 6, 9, 12],
        "CSR offsets mismatch"
    );

    // SH degree inferred from color_sh_* count
    assert_eq!(model.sh_degree, 1, "fixture should infer SH degree 1");

    // SH coefficients length = N * sh_component_count * 3
    let sh_dim = model.sh_component_count() * 3;
    assert_eq!(
        model.sh_coefficients.len(),
        model.points.len() * sh_dim,
        "sh_coefficients should be packed as N * sh_dim"
    );

    // Ensure adjacency indices are in-bounds
    let n = model.points.len();
    for (k, &idx) in adjacency.neighbors.iter().enumerate() {
        assert!(
            (idx as usize) < n,
            "adjacency index out of bounds at entry {}: {} (n={})",
            k,
            idx,
            n
        );
    }
}

#[test]
fn radfoam_ascii_ply_dc_approx_is_reasonable_for_mid_gray_preview() {
    let model = vol::io::load_radfoam(TINY_ASCII_PLY);

    // The fixture uses red=green=blue=128 (mid gray-ish).
    // The loader approximates DC by inverting:
    //   rgb8 = clamp(255 * (0.5 + C0 * dc), 0, 255)
    // -> dc ≈ (rgb8/255 - 0.5) / C0
    //
    // For rgb8=128: (128/255 - 0.5) ~= 0.0019607843
    // so dc ~= 0.0019607843 / 0.28209479 ~= 0.00695
    //
    // We don't need exact equality here; we just want to catch obvious regressions
    // (e.g. forgetting to apply inversion or writing DC to wrong slots).
    const C0: f32 = 0.282_094_791_773_878_14;
    let expected_dc = ((128.0 / 255.0) - 0.5) / C0;

    let sh_dim = model.sh_component_count() * 3;
    assert_eq!(
        sh_dim, 12,
        "degree 1 should produce sh_dim=12 (4 components * 3 channels)"
    );

    for i in 0..model.points.len() {
        let base = i * sh_dim;

        let dc_r = model.sh_coefficients[base + 0];
        let dc_g = model.sh_coefficients[base + 1];
        let dc_b = model.sh_coefficients[base + 2];

        // Allow a little slack (parsing + float error); this should be quite tight.
        let eps = 1e-3;
        assert!(
            (dc_r - expected_dc).abs() <= eps,
            "dc_r mismatch for point {}: got {}, expected {}",
            i,
            dc_r,
            expected_dc
        );
        assert!(
            (dc_g - expected_dc).abs() <= eps,
            "dc_g mismatch for point {}: got {}, expected {}",
            i,
            dc_g,
            expected_dc
        );
        assert!(
            (dc_b - expected_dc).abs() <= eps,
            "dc_b mismatch for point {}: got {}, expected {}",
            i,
            dc_b,
            expected_dc
        );

        // Density is now in points[i].w
        let density = model.points[i].w;
        assert!(
            density.is_finite() && density > 0.0,
            "density should be finite and >0 for point {} (got {})",
            i,
            density
        );
    }
}
