//! End-to-end conversion of a real glTF asset.
//!
//! The unit tests in the library cover individual glTF semantics against
//! synthetic inputs. This suite converts the checked-in `police.glb` the way a
//! user would and asserts the properties an asset pipeline depends on:
//! non-empty output, a structurally valid model, a PLY that round-trips, and
//! reproducibility. It runs on CPU only, so it is CI-safe.

use blade_volume as vol;
use blade_volume_convert as convert;
use std::path;

/// Resolution kept deliberately low: this suite checks correctness, not
/// quality, and CI should not spend minutes on it.
const TEST_RESOLUTION: f32 = 12.0;

fn asset() -> path::PathBuf {
    path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../blade-volume-test/data/police.glb")
        .canonicalize()
        .expect("checked-in police.glb should exist")
}

fn options(output: convert::OutputKind) -> convert::ConvertOptions {
    convert::ConvertOptions {
        output,
        resolution: Some(TEST_RESOLUTION),
        ..convert::ConvertOptions::default()
    }
}

fn out_path(name: &str) -> path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("blade-volume-convert-{name}.ply"));
    path
}

#[test]
fn radfoam_conversion_produces_a_valid_model() {
    let model = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam))
        .expect("conversion should succeed");

    assert!(!model.is_empty(), "expected a non-empty cloud");
    assert_eq!(model.sh_degree, 0);
    assert!(model.transforms.is_none(), "RadFoam output is not Gaussian");
    let adjacency = model.adjacency.as_ref().expect("RadFoam needs adjacency");
    assert!(!adjacency.neighbors.is_empty());
    // Catches asymmetric, out-of-range, self-referencing, or unsorted CSR.
    model
        .validate()
        .expect("model should be structurally valid");
}

#[test]
fn gaussian_conversion_produces_a_valid_model() {
    let model = convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian))
        .expect("conversion should succeed");

    assert!(!model.is_empty());
    assert!(
        model.adjacency.is_none(),
        "Gaussian output has no adjacency"
    );
    let transforms = model
        .transforms
        .as_ref()
        .expect("Gaussian output needs transforms");
    assert_eq!(transforms.rotations.len(), model.len());
    assert_eq!(transforms.scales.len(), model.len());
    model
        .validate()
        .expect("model should be structurally valid");
}

#[test]
fn conversion_is_reproducible_for_a_fixed_seed() {
    // An asset pipeline that silently produces a different cloud each run
    // cannot be cached or diffed, so this is a hard requirement.
    let first = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let second = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();

    assert_eq!(first.len(), second.len());
    assert_eq!(first.points, second.points);
    assert_eq!(first.sh_coefficients, second.sh_coefficients);
    let (a, b) = (first.adjacency.unwrap(), second.adjacency.unwrap());
    assert_eq!(a.offsets, b.offsets);
    assert_eq!(a.neighbors, b.neighbors);
}

#[test]
fn a_different_seed_moves_the_samples() {
    let base = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let shifted = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            seed: 12345,
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .unwrap();
    assert_ne!(
        base.points, shifted.points,
        "the seed should actually drive sampling"
    );
}

#[test]
fn resolution_monotonically_refines_the_cloud() {
    // Unit-scale invariance itself is covered by the library test that scales
    // a synthetic mesh; this asset only exists at one scale. What it can check
    // is that the knob is monotone and actually connected.
    let base = convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian)).unwrap();
    let finer = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            resolution: Some(TEST_RESOLUTION * 2.0),
            ..options(convert::OutputKind::Gaussian)
        },
    )
    .unwrap();
    assert!(
        finer.len() > base.len(),
        "doubling resolution should add samples: {} vs {}",
        finer.len(),
        base.len()
    );
}

#[test]
fn interior_jitter_defaults_per_output_kind() {
    // Jitter exists to break the lattice degeneracy that makes Delaunay slow
    // and its result ambiguous. Gaussian output is never triangulated, so the
    // automatic default must leave it on the lattice; scattering those splats
    // measurably degrades the reference render.
    let gaussian_auto =
        convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian)).unwrap();
    let gaussian_explicit_zero = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            interior_jitter: Some(0.0),
            ..options(convert::OutputKind::Gaussian)
        },
    )
    .unwrap();
    assert_eq!(
        gaussian_auto.points, gaussian_explicit_zero.points,
        "Gaussian output should default to an unjittered lattice"
    );

    let radfoam_auto =
        convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let radfoam_explicit_zero = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            interior_jitter: Some(0.0),
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .unwrap();
    assert_ne!(
        radfoam_auto.points, radfoam_explicit_zero.points,
        "RadFoam output should default to a jittered interior"
    );

    // An explicit request must win over the per-kind default in both
    // directions.
    let radfoam_matches_default = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            interior_jitter: Some(convert::DEFAULT_INTERIOR_JITTER),
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .unwrap();
    assert_eq!(radfoam_auto.points, radfoam_matches_default.points);
}

#[test]
fn radfoam_gets_a_transparent_exterior_and_gaussian_does_not() {
    // Without transparent exterior cells, empty space belongs to unbounded
    // cells owned by opaque surface sites and the object cannot be viewed from
    // outside. Gaussian splatting has no cells, so it must not pay the cost.
    let radfoam = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let radfoam_bare = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            exterior_density_scale: Some(0.0),
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .unwrap();
    assert!(
        radfoam.len() > radfoam_bare.len(),
        "RadFoam should add exterior cells by default: {} vs {}",
        radfoam.len(),
        radfoam_bare.len()
    );
    // Those extra cells must be fully transparent, or they would fog the view.
    let transparent = radfoam.points.iter().filter(|p| p.w == 0.0).count();
    assert!(
        transparent >= radfoam.len() - radfoam_bare.len(),
        "every added exterior cell should have zero density"
    );

    let gaussian = convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian)).unwrap();
    let gaussian_bare = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            exterior_density_scale: Some(0.0),
            ..options(convert::OutputKind::Gaussian)
        },
    )
    .unwrap();
    assert_eq!(
        gaussian.len(),
        gaussian_bare.len(),
        "Gaussian output should not carry exterior fill"
    );
}

#[test]
fn radfoam_opacity_is_a_density_independent_of_sampling_rate() {
    // RadFoam integrates `alpha = 1 - exp(-w * dt)`, so `w` is a density per
    // unit length. Storing an alpha there would make the object fade as cells
    // shrink; the stored value must instead rise as resolution rises.
    let coarse = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let fine = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            resolution: Some(TEST_RESOLUTION * 4.0),
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .unwrap();

    let peak =
        |model: &vol::PointCloudModel| model.points.iter().map(|p| p.w).fold(0.0f32, f32::max);
    assert!(
        peak(&fine) > peak(&coarse) * 2.0,
        "finer cells need proportionally higher density: {} vs {}",
        peak(&fine),
        peak(&coarse)
    );

    // Gaussian stores alpha directly and must stay bounded by 1.
    let gaussian = convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian)).unwrap();
    assert!(
        gaussian.points.iter().all(|p| p.w <= 1.0),
        "Gaussian opacity is an alpha and must not exceed 1"
    );
}

#[test]
fn radfoam_ply_round_trips_through_disk() {
    let model = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let path = out_path("radfoam-roundtrip");
    convert::save_ply(&path, &model).expect("save should succeed");

    let loaded = vol::io::load_radfoam(path.to_str().unwrap());
    assert_eq!(loaded.points.len(), model.points.len());
    assert_eq!(loaded.sh_degree, model.sh_degree);
    loaded.validate().expect("reloaded model should be valid");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn power_foam_output_carries_radii() {
    let model = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            assign_radii: true,
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .expect("conversion should succeed");

    let radii = model.radii.as_ref().expect("radii requested");
    assert_eq!(radii.len(), model.len());
    assert!(
        radii.iter().all(|r| r.is_finite() && *r > 0.0),
        "every radius should be positive and finite"
    );
    model.validate().expect("Cech model should be valid");
}

#[test]
fn curvature_boost_redistributes_without_inflating_the_cloud() {
    let flat = convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian)).unwrap();
    let boosted = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            curvature_boost: 2.0,
            ..options(convert::OutputKind::Gaussian)
        },
    )
    .unwrap();

    assert_ne!(flat.points, boosted.points, "boost should move samples");
    // Area-weight normalization keeps the budget roughly constant; integer
    // rounding per triangle means it is not exact.
    let ratio = boosted.len() as f32 / flat.len() as f32;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "curvature boost should redistribute, not inflate: ratio {ratio}"
    );
}

#[test]
fn interior_fill_can_be_disabled() {
    let surface_only = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            interior_density_scale: 0.0,
            ..options(convert::OutputKind::Gaussian)
        },
    )
    .unwrap();
    let filled = convert::convert_gltf(asset(), &options(convert::OutputKind::Gaussian)).unwrap();

    assert!(!surface_only.is_empty(), "surface samples should remain");
    assert!(
        surface_only.len() < filled.len(),
        "interior fill should add points: {} vs {}",
        surface_only.len(),
        filled.len()
    );
}

#[test]
fn a_missing_asset_is_an_error_not_a_panic() {
    let result =
        convert::convert_gltf("does-not-exist.glb", &options(convert::OutputKind::RadFoam));
    assert!(matches!(result, Err(convert::ConvertError::Gltf(_))));
}

#[test]
fn a_non_positive_rate_is_rejected() {
    for bad in [0.0f32, -1.0] {
        let result = convert::convert_gltf(
            asset(),
            &convert::ConvertOptions {
                resolution: None,
                density: bad,
                ..options(convert::OutputKind::Gaussian)
            },
        );
        assert!(
            matches!(result, Err(convert::ConvertError::InvalidDensity)),
            "density {bad} should be rejected"
        );
    }
}

#[cfg(not(feature = "qhull"))]
#[test]
fn qhull_topology_reports_a_missing_feature() {
    let result = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            topology: convert::Topology::Qhull,
            ..options(convert::OutputKind::RadFoam)
        },
    );
    assert!(matches!(
        result,
        Err(convert::ConvertError::QhullUnavailable)
    ));
}

#[cfg(feature = "qhull")]
#[test]
fn qhull_and_exact_topologies_agree_on_the_cloud() {
    // The two builders may legitimately differ on which Delaunay of a
    // near-degenerate set they return, but they must agree on the sites and
    // both must produce a structurally valid model.
    let exact = convert::convert_gltf(asset(), &options(convert::OutputKind::RadFoam)).unwrap();
    let qhull = convert::convert_gltf(
        asset(),
        &convert::ConvertOptions {
            topology: convert::Topology::Qhull,
            ..options(convert::OutputKind::RadFoam)
        },
    )
    .unwrap();

    assert_eq!(
        exact.points, qhull.points,
        "sites must not depend on backend"
    );
    exact.validate().expect("exact model should be valid");
    qhull.validate().expect("qhull model should be valid");
}
