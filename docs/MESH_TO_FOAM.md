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
2. **Bounds + grid.** AABB of the gathered triangles → a uniform 3D grid.
   `--density` is *absolute*: spacing is `density^(-1/3)` world units, with no
   bounding-box term, so the same value gives wildly different clouds for
   assets authored in metres versus centimetres. `--resolution N` asks for `N`
   cells across the bounding-box diagonal instead and is therefore scale
   invariant; prefer it whenever the asset's units are not under your control.
   `--spacing` states the world-unit spacing directly. All three resolve to the
   single length scale that drives both the interior and surface budgets.
3. **Interior sampling.** For each grid cell whose centre is *inside* the
   mesh (ray-cast intersection-count parity test against the gathered
   triangles), drop `interior_subdiv^3` jittered points carrying the
   mesh's average colour. The grid + subdivision keeps the interior cloud
   uniform in volume. The parity test is accelerated by one triangle index per
   parity direction, bucketed by footprint on the plane perpendicular to that
   direction; the narrowing is exact, so results match an exhaustive scan.
   `--interior-jitter` displaces each sample within its sub-cell: an exact
   lattice is a cospherical, degenerate configuration for Delaunay, which both
   slows the builder down and makes the resulting adjacency arbitrary.
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

Sampling is roughly proportional to the configured surface/volume budgets. The
output goes straight into `blade-volume-view` and renders without any training.

`--topology qhull` (needs `--features qhull`) selects the same Qhull builder the
training crate uses. Note the measured result below: once interior jitter breaks
the lattice degeneracy, the pure-Rust builder is the *faster* of the two at
800k sites, so Qhull is an alternative rather than the scaling escape hatch this
document previously assumed it would be.

### Measured cost

`police.glb` (2,076 triangles), single core, release build:

| Stage | Before | After |
| --- | ---: | ---: |
| Sampling at resolution 128 (804,464 points) | 42.6 s | 0.087 s |
| Exact Delaunay on that cloud, lattice interior | 30.9 s | — |
| Exact Delaunay on that cloud, jittered interior | — | 8.4 s |
| End-to-end RadFoam at resolution 128 | 42.9 s | 8.5 s |

The interior parity test used to scan every triangle for every grid point,
making it `O(grid^3 * triangles)` and 99.3% of runtime; the per-direction index
removed that. What remains is dominated by Delaunay, so *that* is where the next
scaling work belongs — not in the sampler.

Interior jitter matters for more than speed. On the lattice, the exact and Qhull
builders disagreed on 15.6% of edges, because a cospherical point set has no
unique Delaunay triangulation. Half a sub-cell of jitter cuts the disagreement
to 3.9% and makes the exact builder 3.7x faster.

Jitter defaults *per output kind*, because that is where the trade-off differs.
RadFoam is triangulated and gets it; Gaussian output builds no adjacency, so
jitter buys nothing there and measurably degrades the render by scattering
splats off their lattice (the reference image drops ~2.3 dB). `--interior-jitter`
overrides either way.

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
- **RadFoam saturates far below Gaussian.** With the exterior fill and correct
  density semantics in place, RadFoam reaches ~13 dB against the mesh and stops
  improving (+0.66 dB from resolution 24 to 96), while Gaussian climbs past
  20 dB and is still rising. The residual is surface speckle: random
  barycentric samples give an incoherent set of opaque cells, not a coherent
  surface. This is now the top open item, and it is measurable.
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

## Measuring conversion quality

The ground truth for a converted asset is the source mesh, and Blade traces
triangles, so there is nothing analytic about it: `MeshReferenceTracer`
(`blade-volume/src/gpu/mesh_reference.rs`) builds a BLAS/TLAS from the glTF
triangles and ray traces them with the same camera convention, output format,
and shading model the cloud backends use. Any difference in the rendered image
is therefore the cost of the representation.

```text
cargo run --release -p blade-volume-test --bin convert_quality -- \
    --kind gaussian --resolutions 16,24,32,48,64 --views 8 --size 256
```

| Resolution | Points | Gaussian PSNR |
| ---: | ---: | ---: |
| 16 | 3,376 | 14.04 dB |
| 24 | 7,049 | 16.38 dB |
| 32 | 14,784 | 17.87 dB |
| 48 | 42,728 | 19.83 dB |
| 64 | 104,942 | 20.77 dB |

The metric immediately exposed two representation defects that no structural
check could have caught, which is precisely why cost-only benchmarking missed
them:

1. **Opaque exterior fog.** An object-centric RadFoam cloud rendered as a white
   smear from *any* external camera, ~5 dB and not improving with resolution.
   A Voronoi diagram built only from surface and interior sites hands every
   point of empty space to an unbounded cell owned by an opaque surface site.
   Trained scenes never show this because optimisation drives background cells
   to near-zero density; a converted asset has no such stage. Fixed by
   `--exterior-density-scale`, which seeds zero-density cells around the mesh.
   5.01 → 13.90 dB.
2. **Alpha stored where a density belongs.** `point.w` is alpha for Gaussian
   splatting, but RadFoam integrates `alpha = 1 - exp(-w·dt)`, so there it is a
   density per unit length. The converter stored the same number for both,
   making coverage depend on cell size — objects grew *more* transparent as
   sampling got finer. Fixed by solving for the density that achieves the
   requested opacity across one local cell. Note this *lowered* measured PSNR
   at first, because translucency had been masking silhouette bloat; the fix is
   dimensionally correct and the bloat is what the exterior fill addresses.

Full ladders, the exterior-fill rate gate, and the defect records are in
`benchmarks/mesh_conversion.toml`.

## Backend choice for the first interactive prototype

**Gaussian is the target.** The measurements above are the reason rather than a
preference: at comparable point counts it reaches 20.77 dB against the source
mesh and is still improving with sampling rate, while RadFoam plateaus near
13 dB. It also sidesteps the two failure modes that only exist for a
cell-based representation — empty space needing explicit transparent sites, and
opacity being a density that has to track cell size. Neither has an analogue in
splatting, where a splat carries its own finite support and alpha.

That is a statement about the *conversion* path today, not about RadFoam as a
representation: the trained-from-photographs track continues on RadFoam, where
optimisation supplies exactly what conversion cannot (learned background
density, learned per-cell opacity, sites placed by error rather than by a
sampling heuristic). Revisit this once surface sample distribution improves,
since that is what the RadFoam ceiling is made of.

## Recommendation

Every knob described here is now reachable from the `convert` binary — they
were previously library-only and defaulted off, so in practice every
command-line conversion was the plain baseline. `etc/convert_smoke.sh` runs the
binary end to end and is wired into CI alongside the crate's tests.

The next step is **surface sample distribution**, and it is now decidable by
measurement rather than argument: RadFoam's residual error is speckle from
random barycentric sampling, so blue-noise or a true bounded CVT (improvement 4)
can be scored directly with `convert_quality`. Signed-distance density
(improvement 3) is the natural follow-up if silhouettes remain the limit. None
of this requires a polygonal runtime path, and training remains optional.

(5) and (6) remain optional rather than blocked. Rendering the mesh is not the
line this document once assumed it was: Blade traces triangles, and the
reference renderer above already does it. What (5) and (6) add is *fitting*
against those renders, which is a separate decision from having them — the
quality metric needs only the renders, and stays gradient-free.

## Stretch idea: tetrahedralised mesh interiors

A different framing entirely: tetrahedralise the mesh interior (NOT the
ambient space) using a constrained Delaunay implementation. Each tet's
*site* becomes a foam cell with density derived from interior
homogeneity. This gives strictly-mesh-conforming cells, no leakage outside
the surface, no need for the parity-test interior detection. The cost is
finding a Rust constrained-Delaunay library (we don't currently have one;
`simple_delaunay_lib` doesn't constrain).
