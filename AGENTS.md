This is a Rust+WGSL library that directly works with volumetric data.

# Principles

- low dependencies, only Rust
- simple code, don't overcomplicate and assume future cases, assume users know what they are doing
- strict style:
  - single `use` per crate, prefer to import modules instead of individual items
  - no implicit references in `match`, prefer explicit `ref` instead

# Development Tracks

## Productionization

Modularize the shaders:
  - move the common shader code into a separate WGSL in the examples
  - move backend-specific code to the new `shaders` folder, target embedding into other apps
  - make shared shaders between backends, e.g. for spherical harmonics evaluation

Add an API that allows the user to create backend-specific data (BLAS) from triangular meshes.

Implement the distinction between BLAS and TLAS, allow the user to control the transformation of objects every frame. Basically, allowing them to:
```rust
let object = scene.create_object("point_cloud.ply");
for frame in frames {
  scene.set_transform(object, position, rotation, scale);
  scene.render(target_view);
}
```

## Optimization

Gather more ideas?

Dynamic wave regrouping:
- do the first few steps (applicable to both radfoam and ray-traced gaussians) from the screen
- re-pack the survived rays into new ways, continue for another number of steps
- straightforward way would be to re-pack using a different compute invocation, followed by indirect dispatch for the main traversal
- perhaps this can be implemented to work entirely within a single dispatch by using atomics?

## Extension

New rendering methods:
- SDF based on compute
- Gaussians splatted with compute (instead of the current ray tracing), in 3DGUT formulation
- **Power Foam** (successor to RadFoam, https://github.com/theialab/powerfoam): power-diagram cells with per-point radius (weight). See *Power Foam Adoption* below.

Implement a way to build BLAS by the means of reconstruction from a sequence of images (and masks).

## End-to-End Pipeline

Goal: phone video capture → training → interactive viewer.

Stages:
1. **Capture**: short video (or image burst) from a phone.
2. **Pose & sparse points**: COLMAP (external tool) emits `images/` + `sparse/0/{cameras,images,points3D}.bin`.
3. **Training**: Rust-side reconstruction using [`meganeura`](https://github.com/kvark/meganeura) as the autograd / training backend. No PyTorch, no Burn. Inputs: COLMAP output. Outputs: a `PointCloudModel` (positions + density + SH, optional rotation/scale, optional adjacency).
4. **Viewer**: existing `blade-volume-view`, picking the backend (Gaussian / RadFoam / future PowerFoam) from the model contents.

The library boundary stays at `PointCloudModel` — the training crate produces it, the viewer consumes it. Keep the dependency graph one-way: a new `blade-volume-train` crate may depend on `blade-volume`, never the reverse.

## Power Foam Adoption

PowerFoam (theialab/powerfoam, arXiv 2604.24994) generalises RadFoam: Voronoi → **power diagram** (weighted Voronoi), each site carries an extra **radius/weight** parameter, and adjacency is a **Čech complex** built from overlapping balls rather than a Delaunay tetrahedralisation. The same primitives serve both a rasterizer and a ray-tracer with adjacency-walk traversal.

Concrete deltas vs current RadFoam path:
- **Per-point data**: `PointCloudModel.radii: Option<Vec<f32>>` carries the weight (done, M2a). `SurfaceDetail::directional` carries the released eight-axis colour function at each of eight detail sites; the older compact cell-level Spherical Voronoi residual remains a separate rejected experiment.
- **Adjacency builder**: `adjacency::compute_cech` emits an edge `{i,j}` when `|p_i - p_j| ≤ r_i + r_j` (done, M2b). `PointCloudModel::compute_adjacency*` dispatches Čech vs Delaunay based on `radii.is_some()`. CSR storage is unchanged.
- **Traversal**: WGSL `radfoam_trace.wgsl` uses the radical plane `shift = 0.5 + 0.5·(r_i² - r_j²)/|p_j - p_i|²` (done, M2c). The radius lives in the `.w` channel of `g_points`; unweighted clouds upload 0 and the formula degenerates to the bisector — no new bind-group entry, no fork.
- **Oriented dipoles**: `PointCloudModel.surface_normals` optionally clips each bounded power cell to its retained surface half. PLY IO, CPU/WGSL traversal, analytical training Jacobians, PCA initialization, normal loss, densification, resume, eight spatial detail sites, and their released per-site directional appearance are implemented. The Bonsai gate selects learned normals as opt-in but not as the default; staged directional appearance passes held-out Room/Bonsai quality and production-cost gates.
- **PLY format**: per-vertex `property float radius` added to the RadFoam PLY reader/writer (done, M2a). Round-trips both binary and ASCII.
- **Don't pull Python in**: keep PowerFoam adoption to a clean re-implementation in Rust/WGSL. Use the paper + their Warp kernels as a reference, not a dependency.

Remaining work to fully cover the PowerFoam paper:
1. Make the released directional table part of a robust joint reconstruction objective; its staged frozen-base fit is practical, but a fresh joint fit remains quality-negative.
2. Validate the full appearance model beyond Room and Bonsai before enabling it by default.

## Style

- `cargo clippy --workspace --all-targets -- -D warnings` must pass; lints live in the workspace `[workspace.lints]` block in the root `Cargo.toml`.
- Both `cargo fmt --all -- --check` and clippy run in CI.
