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

1. **Geometry collection.** Walk the glTF default scene (or first scene when
   no default is declared), accumulate every triangle with world-space
   positions, UVs, normals, vertex colours, and its material index. The
   transform stack is applied as we walk.
2. **Bounds + grid.** AABB of the gathered triangles → a uniform 3D grid at
   `spacing = bound_diag / density^(1/3)` cells per side.
3. **Interior sampling.** For each grid cell whose centre is *inside* the
   mesh (ray-cast intersection-count parity test against the gathered
   triangles), drop `interior_subdiv^3` jittered points carrying the
   mesh's average colour. The grid + subdivision keeps the interior cloud
   uniform in volume.
4. **Surface sampling.** Per triangle, pick seeded random barycentric points
   from an area budget, optionally redistributed toward high-curvature
   triangles. Base colour combines glTF texture transforms/wrap modes,
   factors, vertex colour, alpha mode, and ambient gain in linear light, then
   encodes once into the model's display-sRGB contract. Gaussian footprint is
   derived from the local sampling density.
5. **Adjacency.** `vol::compute_adjacency_default` runs 3D Delaunay over the
   union of interior + surface points. Each Delaunay edge is a Voronoi
   neighbour pair → the CSR adjacency the renderer wants. RadFoam conversion
   can then run the explicitly named spring heuristic. Optional nearest-
   neighbour radii switch the output to PowerFoam and rebuild a Čech graph.

Sampling is roughly proportional to the configured surface/volume budgets.
Exact 3D Delaunay can have quadratic output and the pure-Rust implementation is
not production-scalable at large site counts. On a small glTF the conversion
finishes in seconds. The converter does not yet expose the training crate's
isolated Qhull selection, so it remains a small-asset path until that explicit
feature-controlled backend is wired. The output goes straight into
`blade-volume-view` and renders without any training.

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
- **Sample distribution remains heuristic.** Curvature redistribution improves
  the area sampler, but it is not blue-noise sampling or a surface-aware CVT.
  Delaunay-based Voronoi cells around badly distributed sites can still have
  long faces that produce visible traversal artefacts.
- **Power-diagram radii are only an initial heuristic.** The converter can use
  scaled nearest-neighbour distance and rebuild Čech adjacency, but it does not
  infer radii from features or optimise them against rendered evidence.

## Improvement paths, in increasing complexity

### 1. Curvature-aware surface sampling (implemented)

Bias sample density on a triangle by how curved its neighbourhood is —
The converter estimates a vertex-sharing normal-variation proxy and applies an
area-normalized boost. Result: more samples on edges/creases, fewer on flat
panels, and roughly the same pre-rounding total budget.

### 2. Feature-aware radii beyond nearest-neighbour scale (modest)

For each sample point, set `radius = α · distance_to_nearest_geometric_feature`
where the feature is the nearest sharp edge or change in material. With
radii populated, the converter emits a Power Foam: `model.radii.is_some()` in
`PointCloudModel`, traversal uses radical planes, and adjacency is rebuilt as a
Čech complex. The existing nearest-neighbour initializer is a baseline for this
feature-aware rule, not the finished rule itself.

### 3. Density from signed distance to surface (modest)

Instead of binary surface/interior opacity, compute a signed distance
field at each sample point and set density via a smoothstep around the
surface. For opaque solids this gives the "thick skin" silhouette anti-
aliasing RadFoam papers describe. SDFs of meshes are well-trodden
territory — even brute-force nearest-triangle works on small meshes.

### 4. Centroidal Voronoi Tessellation relaxation (moderate)

After initial sampling, run bounded Lloyd iterations: compute each clipped
Voronoi-cell centroid, move the site partway toward it, and repeat. The current
`spring_relax` function is deliberately not called Lloyd/CVT; averaging dual
vertices is not generally the volume centroid of a polyhedron. A true
implementation needs a defined clipping domain and volume integration.

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

For asset preview and offline mesh → point-cloud conversion, first benchmark
the implemented curvature redistribution and optional nearest-neighbour
PowerFoam initializer against the plain RadFoam baseline. Add signed-distance
density or a true bounded CVT only when held-out render metrics identify
density leakage or site distribution as the limiting error. None requires a
polygonal runtime path, and training remains optional.

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
