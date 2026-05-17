# Mesh → Foam Without 2D Snapshots

The question: given a triangle mesh (glTF, OBJ, USD, …), can we convert it
directly to a RadFoam / Power Foam point cloud — without re-rendering it
from many viewpoints and then re-training appearance from those snapshots?

**Short answer: yes, and we already do it.** `blade-volume-convert` takes a
glTF and emits a renderable foam from pure geometric + material data; no
image renders, no training, no GPU loop. The result trades some appearance
fidelity (no view-dependent BRDF, simple density profile) for being
deterministic, fast, and gradient-free.

This doc explains what's there, what its limits are, and which improvements
are worth chasing.

## What `convert_gltf` does today

1. **Geometry collection.** Walk the glTF scene graph, accumulate every
   triangle (with world-space positions, UVs, normals) and its material
   index. The transform stack is applied as we walk.
2. **Bounds + grid.** AABB of the gathered triangles → a uniform 3D grid at
   `spacing = bound_diag / density^(1/3)` cells per side.
3. **Interior sampling.** For each grid cell whose centre is *inside* the
   mesh (ray-cast intersection-count parity test against the gathered
   triangles), drop `interior_subdiv^3` jittered points carrying the
   mesh's average colour. The grid + subdivision keeps the interior cloud
   uniform in volume.
4. **Surface sampling.** Per triangle, pick `ceil(area * surface_density)`
   uniformly-random barycentric points; look up each point's UV in the
   material's base-colour texture (sRGB → linear); push it with that colour
   and a small scale (so the surface "skin" is dense and thin).
5. **Adjacency.** `vol::compute_adjacency_default` runs 3D Delaunay over the
   union of interior + surface points. Each Delaunay edge is a Voronoi
   neighbour pair → the CSR adjacency the renderer wants.

The whole pipeline is `O(N log N)` for Delaunay, `O(T * area)` for
sampling. On a small glTF (a few thousand triangles), it finishes in
seconds. The output goes straight into `blade-volume-view` and renders
without any training.

## Why this works without snapshots

- **Diffuse appearance is in the material.** A textured mesh tells us the
  surface RGB per UV. Sampling the texture once at sample time captures
  the same information rendered images would, modulo view-dependence
  (which SH degree 0 can't represent anyway).
- **Geometry doesn't need to be inferred.** We have the triangles. The
  hard part of RadFoam/PowerFoam from photographs — recovering scene
  geometry from images — is already done.
- **Voronoi adjacency is intrinsic.** Once we have points in space, the
  cell structure is a deterministic function of their positions. No
  optimisation, no per-pixel loss, no learning rate.

## Where it falls short

- **View-dependent surfaces** (metals, glass, anisotropic) lose their
  highlights. SH degree 0 stores one RGB per cell. There is no concept
  of "this surface looks different from over there."
- **Density profile is binary-ish.** Surface points use `surface_opacity`,
  interior points use `interior_opacity`. A real translucent material —
  skin, wax, jade — has a smooth density falloff we don't capture.
- **Sample distribution is naive.** Uniform random on each triangle
  over-samples large flat areas and under-samples sharp edges.
  Delaunay-based Voronoi cells around badly-distributed sites have ugly
  long faces that produce visible artefacts in traversal.
- **No power-diagram radii.** Every point has implicit radius zero, so
  the foam is plain Voronoi rather than the Power Foam variant the
  shader can already render.

## Improvement paths, in increasing complexity

### 1. Curvature-aware surface sampling (easy)

Bias sample density on a triangle by how curved its neighbourhood is —
estimate curvature from the dihedral angle to its neighbour or from
normal variation across a small geodesic disc. The current implementation
already has triangle normals; weighting `count` by
`area * (1 + k * curvature)` is a one-line change. Result: more samples
on edges/creases, fewer on flat panels, same total budget.

### 2. Per-cell radii from local feature size (modest)

For each sample point, set `radius = α · distance_to_nearest_geometric_feature`
where the feature is the nearest sharp edge or change in material. With
radii populated, the converter emits a Power Foam: `model.radii.is_some()`
in `PointCloudModel`, traversal uses the radical plane, cell shapes shrink
near edges and grow in flat regions. The Čech-complex adjacency builder
(`vol::compute_cech`) is already in the tree.

### 3. Density from signed distance to surface (modest)

Instead of binary surface/interior opacity, compute a signed distance
field at each sample point and set density via a smoothstep around the
surface. For opaque solids this gives the "thick skin" silhouette anti-
aliasing RadFoam papers describe. SDFs of meshes are well-trodden
territory — even brute-force nearest-triangle works on small meshes.

### 4. Centroidal Voronoi Tessellation relaxation (moderate)

After initial sampling, run a few Lloyd's iterations: compute the Voronoi
cell centroid of each point, move the point partway toward it, repeat.
Two or three passes give visibly more uniform cell sizes and remove the
spike artefacts you can see in `etc/rf-bike.jpg`-style renders. Needs an
efficient Voronoi-centroid routine — we have the Delaunay already, the
centroid of cell `i` is just the average of the dual vertices.

### 5. View-dependent appearance from a few captured renders (large)

Even without "training" in the gradient-descent sense, we could fit SH
degrees 1–3 per cell from a small number (~8) of analytic mesh renders
at canonical viewpoints. This is a *closed-form* SH fit per cell —
solve `A @ x = b` where rows are basis evaluations at each view direction.
No GPU autodiff needed; just a small per-cell linear system. Result: the
foam shows specular hints and stays a one-shot conversion.

### 6. Full differentiable refinement with the training loop (largest)

The training crate is already plumbed. Convert via the methods above to
get a strong initial foam, then run `fit_appearance_multi_view` on a
handful of analytic mesh renders to polish density + SH against the
ground-truth pixels. This *does* introduce 2D images, but only as a
post-processing step on top of an already-good foam. Cheaper than
training from scratch.

## Recommendation

For most use cases (asset preview, mesh → fast-rendering point-cloud
proxy), **(1) + (3) + (4) are the sweet spot**: cheap to implement, no
new dependencies, no images required, visibly better than what we have
today. The output is a Power Foam (radii present), traversal uses the
radical plane, and CPU-only training stays optional.

(5) and (6) are interesting research-y extensions but cross a line —
either we accept analytic mesh rendering as an intermediate (which the
user explicitly wanted to avoid), or we keep things gradient-free and
limit ourselves to SH degree 0.

## Stretch idea: tetrahedralised mesh interiors

A different framing entirely: tetrahedralise the mesh interior (NOT the
ambient space) using a constrained Delaunay implementation. Each tet's
*site* becomes a foam cell with density derived from interior
homogeneity. This gives strictly-mesh-conforming cells, no leakage outside
the surface, no need for the parity-test interior detection. The cost is
finding a Rust constrained-Delaunay library (we don't currently have one;
`simple_delaunay_lib` doesn't constrain).
