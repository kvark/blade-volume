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
- **Per-point data**: add `radius: f32` (or `weight: f32`) to the point representation. PowerFoam also carries a rotation quat and a texel/spherical-Voronoi colour model — start by ignoring those and reusing our SH path.
- **Adjacency builder**: replace the `simple_delaunay_lib` call in `adjacency.rs` with a Čech-complex builder (AABB tree over balls of radius `radii[i]`, edges where balls overlap). The CSR storage format we already have is fine.
- **Traversal**: the face between sites `i` and `j` is no longer the perpendicular bisector — it is the **radical plane** `|x - p_i|² - r_i² = |x - p_j|² - r_j²`. In the WGSL ray-walk this means changing how `face_origin` / `face_normal` and the `t` parameter are derived (positions only → positions + weights).
- **PLY format**: PowerFoam's checkpointed `points.ply` is a lossy visualisation only — the real state lives in `model.pt`. Define a small extension to our RadFoam PLY adding per-vertex `radius` (and quat, when we adopt the full model).
- **Don't pull Python in**: keep PowerFoam adoption to a clean re-implementation in Rust/WGSL. Use the paper + their Warp kernels as a reference, not a dependency.

Suggested order:
1. Extend `PointCloudModel` with `Option<Vec<f32>>` radii.
2. Add a Čech-complex adjacency builder behind a feature flag, share the CSR storage with existing RadFoam.
3. Fork the RadFoam WGSL into `powerfoam.wgsl`, swap the face-plane derivation, gate on the radii buffer presence.
4. Once stable, extend to the full PowerFoam appearance model (quaternion + texel sites).

## Style

- `cargo clippy --workspace --all-targets -- -D warnings` must pass; lints live in the workspace `[workspace.lints]` block in the root `Cargo.toml`.
- Both `cargo fmt --all -- --check` and clippy run in CI.
