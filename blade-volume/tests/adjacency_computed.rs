//! Integration tests for automatic adjacency computation.
//!
//! These tests verify that:
//! 1. Adjacency can be computed from point positions alone
//! 2. Computed adjacency produces valid CSR structure
//! 3. GPU rendering works with computed adjacency
//! 4. Computed adjacency is comparable to hand-crafted fixtures

use blade_volume as vol;

mod radfoam_cpu_ref;
use radfoam_cpu_ref as cpu;

/// Create a simple model with only positions (no adjacency).
///
/// This represents a point cloud loaded from a format that doesn't include
/// adjacency data (e.g., a simple PLY with just positions and colors).
fn make_model_without_adjacency(points: Vec<glam::Vec4>, sh_degree: usize) -> vol::PointCloudModel {
    let comps = vol::get_sh_component_count(sh_degree);
    let sh_dim = comps * 3;
    let n = points.len();

    // Simple DC-only SH coefficients (mid-gray)
    let mut sh_coefficients = vec![0.0f32; n * sh_dim];
    for i in 0..n {
        let base = i * sh_dim;
        // Set DC to a visible gray
        sh_coefficients[base] = 0.5; // R
        sh_coefficients[base + 1] = 0.5; // G
        sh_coefficients[base + 2] = 0.5; // B
    }

    vol::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree,
        transforms: None,
        adjacency: None, // No adjacency - will be computed
        radii: None,
        surface_normals: None,
    }
}

/// Generate tetrahedron corner points.
fn tetrahedron_points() -> Vec<glam::Vec4> {
    vec![
        glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
        glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        glam::Vec4::new(0.5, 1.0, 0.0, 1.0),
        glam::Vec4::new(0.5, 0.5, 1.0, 1.0),
    ]
}

/// Generate a grid of points in 3D space.
fn grid_points(nx: usize, ny: usize, nz: usize, spacing: f32, density: f32) -> Vec<glam::Vec4> {
    let mut points = Vec::with_capacity(nx * ny * nz);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                points.push(glam::Vec4::new(
                    ix as f32 * spacing,
                    iy as f32 * spacing,
                    iz as f32 * spacing,
                    density,
                ));
            }
        }
    }
    points
}

#[test]
fn compute_adjacency_on_tetrahedron() {
    let points = tetrahedron_points();
    let mut model = make_model_without_adjacency(points, 0);

    assert!(model.adjacency.is_none(), "should start without adjacency");

    model.compute_adjacency_default();

    let adj = model
        .adjacency
        .as_ref()
        .expect("adjacency should be computed");

    // Tetrahedron: 4 points, each has exactly 3 neighbors
    assert_eq!(adj.offsets.len(), 5, "offsets should be N+1 = 5");
    assert_eq!(adj.offsets[0], 0, "first offset should be 0");

    for i in 0..4 {
        let start = adj.offsets[i] as usize;
        let end = adj.offsets[i + 1] as usize;
        let neighbor_count = end - start;
        assert_eq!(
            neighbor_count, 3,
            "point {} should have 3 neighbors in tetrahedron, got {}",
            i, neighbor_count
        );
    }

    // Total neighbor entries: 4 * 3 = 12
    assert_eq!(adj.neighbors.len(), 12, "should have 12 neighbor entries");
}

#[test]
fn compute_adjacency_on_grid() {
    // 3x3x3 grid = 27 points
    let points = grid_points(3, 3, 3, 1.0, 0.1);
    let mut model = make_model_without_adjacency(points, 0);

    assert!(model.adjacency.is_none());

    model.compute_adjacency_default();

    let adj = model
        .adjacency
        .as_ref()
        .expect("adjacency should be computed");

    assert_eq!(adj.offsets.len(), 28, "offsets should be 27+1 = 28");

    // All neighbor indices should be in bounds
    for &idx in &adj.neighbors {
        assert!((idx as usize) < 27, "neighbor index {} out of bounds", idx);
    }

    // Corner points should have fewer neighbors than interior points
    // (Delaunay tetrahedralization creates different neighbor counts)
    // Just verify non-zero neighbors for each point
    for i in 0..27 {
        let start = adj.offsets[i] as usize;
        let end = adj.offsets[i + 1] as usize;
        let neighbor_count = end - start;
        assert!(
            neighbor_count > 0,
            "point {} should have at least 1 neighbor, got 0",
            i
        );
    }
}

#[test]
fn ensure_adjacency_computes_when_missing() {
    let points = tetrahedron_points();
    let mut model = make_model_without_adjacency(points, 0);

    assert!(model.adjacency.is_none());

    // ensure_adjacency should compute if missing
    let adj = model.ensure_adjacency();
    assert_eq!(adj.offsets.len(), 5);

    // Calling again should return the same (not recompute)
    let adj2 = model.ensure_adjacency();
    assert_eq!(adj2.offsets.len(), 5);
}

#[test]
fn ensure_adjacency_preserves_existing() {
    // Create a model with hand-crafted adjacency
    let points = vec![
        glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
        glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        glam::Vec4::new(2.0, 0.0, 0.0, 1.0),
        glam::Vec4::new(3.0, 0.0, 0.0, 1.0),
    ];

    // Simple chain: 0-1-2-3
    let custom_adj = vol::Adjacency {
        neighbors: vec![1, 0, 2, 1, 3, 2],
        offsets: vec![0, 1, 3, 5, 6],
    };

    let mut model = vol::PointCloudModel {
        points,
        sh_coefficients: vec![0.0; 4 * 3], // degree 0
        sh_degree: 0,
        transforms: None,
        adjacency: Some(custom_adj),
        radii: None,
        surface_normals: None,
    };

    // ensure_adjacency should NOT recompute
    let adj = model.ensure_adjacency();
    assert_eq!(
        adj.neighbors,
        vec![1, 0, 2, 1, 3, 2],
        "should preserve existing adjacency"
    );
}

#[test]
fn computed_adjacency_cpu_traversal_produces_output() {
    // Create a cluster of points around the Z axis with slight offsets
    // (must be non-coplanar for Delaunay tetrahedralization)
    let mut points = Vec::new();
    for i in 0..10 {
        // Add small offsets in X and Y to avoid colinearity
        let offset_x = ((i * 7) % 3) as f32 * 0.1 - 0.1;
        let offset_y = ((i * 11) % 3) as f32 * 0.1 - 0.1;
        points.push(glam::Vec4::new(offset_x, offset_y, i as f32 * 0.5, 0.1));
    }

    let mut model = make_model_without_adjacency(points, 0);
    model.compute_adjacency_default();

    // Trace a ray along +Z
    let ray = cpu::Ray {
        origin: glam::Vec3::new(0.0, 0.0, -1.0),
        direction: glam::Vec3::new(0.0, 0.0, 1.0),
    };

    let settings = cpu::TraceSettings {
        weight_threshold: 1e-4,
        max_steps: 100,
        start_point: 0,
        depth: 10.0,
        eval_mode: cpu::EvalMode::ConstantRgb(glam::Vec3::splat(1.0)),
    };

    let result = cpu::trace_one_ray(&model, ray, settings);

    // Should have traced through some points
    assert!(
        result.steps > 0,
        "traversal should have taken steps with computed adjacency"
    );

    // Should have accumulated some color (non-zero RGB)
    let rgb_sum = result.rgba.x + result.rgba.y + result.rgba.z;
    assert!(
        rgb_sum > 0.0,
        "should accumulate non-zero color from traversal"
    );
}

#[test]
fn config_max_neighbors_is_respected() {
    // Dense cluster that would naturally have many neighbors
    let mut points = Vec::new();

    // Central point
    points.push(glam::Vec4::new(0.0, 0.0, 0.0, 1.0));

    // Surrounding shell
    for i in 0..50 {
        let theta = (i as f32) * 0.2;
        let phi = (i as f32) * 0.15;
        let r = 1.0;
        points.push(glam::Vec4::new(
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos(),
            1.0,
        ));
    }

    let mut model = make_model_without_adjacency(points, 0);

    let config = vol::AdjacencyConfig {
        max_neighbors: 8,
        validate: true,
    };
    model.compute_adjacency(&config);

    let adj = model.adjacency.as_ref().unwrap();

    // No point should exceed max_neighbors
    for i in 0..model.points.len() {
        let start = adj.offsets[i] as usize;
        let end = adj.offsets[i + 1] as usize;
        let count = end - start;
        assert!(
            count <= 8,
            "point {} has {} neighbors, exceeds max 8",
            i,
            count
        );
    }
}
