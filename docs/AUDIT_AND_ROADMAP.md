# Audit and implementation roadmap

Date: 2026-07-12

This document records the correctness audit of `blade-volume` and the staged
plan for turning it into a dependable, Rust-native point-cloud graphics engine.

## Product direction

The engine is cloud-only at runtime. RadFoam, PowerFoam, Gaussian, SDF, and
future representations are different semantics over point-cloud primitives,
not escape hatches back to polygonal rendering. Polygonal data may be accepted
as an offline reconstruction or conversion input, but the renderer, scene, and
asset boundary remain point-cloud based.

The short-term objective is narrower than a general-purpose engine: first make
one complete reconstruction and rendering path reproduce its reference method.
The reusable scene and engine layers should then be built on top of validated
cloud backends.

## Audit baseline

The audit covered the Rust model and IO code, adjacency builders, CPU and WGSL
traversal, Gaussian hardware ray tracing, the COLMAP pipeline, differentiable
training, checkpointing, the converter, and the experimental scene renderer.
It also ran workspace formatting, linting, tests, render regressions, and two
Bonsai training experiments.

The working pixel-batched trainer demonstrably learns appearance. A diagnostic
run with 20,000 cells, 64 training views, SH degree 3, and 6,400 Adam steps
reached 18.71 dB on eight every-eighth held-out Bonsai views. The reconstruction
was recognizable but showed strong blur and cell-boundary streaking. This
validates the plumbing, not competitive reconstruction quality.

At the audited revision:

- `cargo fmt --all -- --check` failed.
- `cargo clippy --workspace --all-targets -- -D warnings` failed.
- Most tests passed, including CPU/GPU traversal parity and weighted traversal,
  but three training tests failed and two convergence tests were ignored.
- The default whole-image training path failed with `unknown input: dt`.
- Training shutdown reported a leaked `grad_clip_acc` GPU allocation.

## Correct foundations

- The ordinary Voronoi and weighted radical-plane equations are correct.
- The stable softplus and volumetric transmittance construction are sound.
- COLMAP pose inversion is correct for the centered pinhole case.
- Native SH evaluation agrees between the CPU and WGSL layouts.
- Native RadFoam PLY adjacency and radius round-trips are covered.
- The standalone RadFoam renderer and path recorder form a useful base.

## Blocking findings

### Training

- The default whole-image path uses the obsolete precomputed-`dt` graph API.
- The working pixel-batched path is not the default and lacks equivalent
  end-to-end convergence coverage.
- Positions are differentiable but frozen. Moving them without synchronizing
  paths and adjacency produces stale-topology gradients.
- Densification is driven by density gradients rather than the reference
  position-gradient and cell-size signal.
- Reference RadFoam geometry optimization, quantile regularization, and
  incremental topology updates are missing.
- Optional white-background training disagrees with black/uncomposited
  evaluation and viewer output.
- PLY is a lossy training checkpoint for the DC coefficient and carries no Adam
  state.

### PowerFoam

- Traversal shifts radical planes but does not clip a cell by its radius sphere;
  it therefore does not implement the bounded power diagram.
- The Čech pipeline computes radii but does not store them in the model.
- Differentiable path lengths use unweighted midplanes and have no radius
  parameter.
- Radii are neither optimized nor validated against a reference checkpoint.
- The full appearance model remains intentionally deferred until geometry is
  correct.

### Adjacency and traversal

- Exact adjacency builders truncate sorted point indices to 64 neighbors. This
  can remove required faces and independently break graph symmetry.
- CSR validation is structural only; it does not enforce symmetry, uniqueness,
  absence of self-edges, or geometric completeness.
- Traversal does not integrate the last segment up to the requested end depth
  when no later face is found.
- `lloyd_relax` is a global spring relaxation rather than a Lloyd/CVT step.
- Nearest-neighbor radius estimation is quadratic.

### Cameras and color

- Principal point and COLMAP distortion parameters are discarded.
- Source pixels and SH colors are optimized in an implicit nonlinear RGB space.
- Camera, background, and color conventions are not encoded in a shared model
  used by training, evaluation, and viewing.

### Gaussian backend and formats

- Standard 3DGS higher-order SH PLY import/export uses the wrong coefficient
  layout.
- SPZ signed positions and opacity are decoded incorrectly, and higher SH is
  ignored.
- The five-hit hardware-RT sorting window has not been measured against a
  trusted renderer.
- No Gaussian training implementation exists.

### Scene layer

- RadFoam-only scenes do not render through `SceneRenderer`.
- Only the first Gaussian TLAS is bound.
- Gaussian object transforms are not applied dynamically.
- RadFoam objects start traversal at cell zero.
- Ray intervals are incorrect under object scaling.
- Mixed SH layouts, more than sixteen objects, and overlapping cloud volumes
  are not handled robustly.
- Scene tests check state but not rendered pixels.

### Project constraints

- `qhull-sys` introduces C into a project whose stated dependency policy is
  Rust-only. It must be isolated behind a non-default feature, moved out of the
  core library, or replaced.

## Implementation stages

Each stage lands as one or more focused commits. Every commit must pass its
targeted tests and formatting; stage boundaries require workspace formatting,
clippy with warnings denied, and the full practical test suite.

### Stage 0: trustworthy baseline

1. Make the maintained pixel-batched trainer the default public workflow.
2. Remove or update the obsolete whole-image implementation.
3. Restore convergence tests on the maintained path and add an end-to-end
   COLMAP training test.
4. Fix formatting, clippy, and GPU resource lifetime failures.
5. Add deterministic benchmark manifests recording dataset, split, cell count,
   resolution, optimizer steps, seed, hardware, and metrics.

Acceptance gate: documented training commands run; workspace fmt and clippy
pass; all non-hardware-optional tests pass; GPU tests have explicit skip
behavior rather than silent gaps.

### Stage 1: cameras, formats, and model invariants

1. Represent pixel-to-ray projection explicitly, including principal point and
   the supported COLMAP distortion models.
2. Share camera and background conventions between training, CPU evaluation,
   and WGSL rendering.
3. Validate model vector lengths and supported SH degrees at IO boundaries.
4. Correct standard Gaussian PLY SH layout with external fixtures.
5. Correct and complete SPZ v2 decoding with known-good fixtures.
6. Introduce a lossless native training checkpoint with optimizer state.
7. Isolate the C-backed Qhull option from the default Rust-only build.

Acceptance gate: external Gaussian PLY and SPZ fixtures reproduce known values;
off-center and distorted camera rays match a CPU oracle; checkpoint resume is
numerically continuous.

### Stage 2: reference-faithful RadFoam

1. Make position optimization safe by rebuilding or incrementally updating
   adjacency and paths after geometry changes.
2. Validate analytical position gradients against finite differences.
3. Implement position-gradient times cell-radius densification and the
   reference pruning schedule.
4. Implement quantile/distortion regularization.
5. Make background and color-space behavior explicit and consistent.
6. Add truncation and topology diagnostics to training and evaluation.
7. Run matched-protocol comparisons against the reference implementation.

Acceptance gate: a canonical scene matches the reference implementation within
0.5-1.0 dB at the same cell budget, split, image scale, and training budget,
with no systematic streaking from stale topology.

### Stage 3: bounded PowerFoam

1. Persist and upload radii whenever Čech adjacency is selected.
2. Intersect each cell interval with the current site's radius sphere.
3. Compute differentiable weighted-face and sphere intersections.
4. Optimize radii together with positions and validate their gradients.
5. Compare traversal against a brute-force bounded-power oracle.
6. Cross-render a reference PowerFoam checkpoint.
7. After geometry passes, add quaternion, dipole, detail-site, and
   spherical-Voronoi appearance semantics.

Acceptance gate: CPU, GPU, and brute-force bounded traversal agree; a reference
checkpoint renders within a defined image tolerance; trained radii improve a
fixed ablation rather than merely changing topology.

### Stage 4: Gaussian backend

1. Establish a trusted raster or reference 3DGRT image oracle.
2. Sweep hit-window and proxy bounds for accuracy and performance.
3. Add cloud transforms without rebuilding point data.
4. Decide whether native Gaussian reconstruction is justified after RadFoam and
   PowerFoam quality is established.

Acceptance gate: imported standard checkpoints match the oracle at documented
quality and performance; transformations pass rendered-pixel tests.

### Stage 5: multi-cloud engine

1. Support RadFoam-only and multiple-Gaussian scenes.
2. Bind per-object layouts and locate correct RadFoam entry cells.
3. Preserve ray parameterization and optical depth under transforms.
4. Define exact ordering for intersecting cloud volumes.
5. Replace the first-sixteen-object scan with bounded actual-hit collection.
6. Add rendered-pixel tests for translation, rotation, uniform scale,
   nonuniform scale, mixed backends, and overlapping volumes.

Acceptance gate: a scene made exclusively from independently transformable
cloud objects renders deterministically and agrees with equivalent standalone
backend renders.

## Benchmark and go/no-go policy

Good-looking screenshots and self-regression images are not quality evidence.
Every backend needs a small deterministic correctness scene and at least one
recognized reconstruction dataset. Metrics must be evaluated from a freshly
serialized model on held-out views.

If reference-faithful RadFoam cannot approach its reference quality after
topology-safe geometry optimization, development should pause before expanding
the scene API. If bounded PowerFoam succeeds, it becomes the preferred common
cloud representation. Gaussian ray tracing remains valuable where its
ellipsoidal semantics or hardware acceleration are a better fit; backends do
not need to be forced into one geometry model to share the cloud-only engine.
