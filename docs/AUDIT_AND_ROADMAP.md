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

## Current status and remaining gaps

### Training

- Whole-image and pixel-batched fitting now use the maintained current-path
  recorder; deterministic convergence and end-to-end COLMAP tests cover it.
- Optional position/radius optimization downloads geometry, rebuilds topology
  and paths on an explicit cadence, and validates active-path Jacobians against
  finite differences.
- Densification uses position-gradient × cell-radius signal, contribution-aware
  pruning, optimizer ancestry, and the weighted copied-radius split policy.
- Quantile regularization, explicit background compositing, lossless DC SH, and
  versioned parameter/Adam/RNG checkpoints are implemented.
- The blocker is quality evidence: the corrected trainer still needs a matched
  reference RadFoam protocol and a completed full-scene run on a stable GPU.

### Remaining PowerFoam gaps

- Bounded traversal, persistent radii, exact active-path Jacobians, and
  trainable positive radii are implemented, but the new WGSL-Jacobian-to-
  meganeura integration test still needs to run on a recovered physical GPU.
- Weighted-cloud densification follows the reference resampler's copied-radius,
  5%-support-scale split; a real-scene ablation remains outstanding.
- No official pretrained checkpoint is published by the reference project, so
  cross-rendering and a matched training ablation remain outstanding.
- The reference quaternion, texel-site, and spherical-Voronoi appearance model
  remains intentionally deferred until weighted geometry is validated on a
  real scene.

### Adjacency and traversal

- Exact adjacency builders now preserve unbounded topology by default. An
  explicit finite cap greedily selects shortest undirected edges without
  breaking graph symmetry, but remains an approximate topology option.
- Model-boundary CSR validation now requires monotonic ranges, in-range sorted
  unique lists, no self-edges, and a reverse edge for every neighbor. It cannot
  prove geometric completeness without rebuilding topology.
- Čech construction uses an immutable k-d tree that tolerates coincident and
  quantized sites. Mesh conversion now rebuilds Čech adjacency after assigning
  radii instead of retaining the preceding unweighted Delaunay graph.
- Traversal now integrates the terminal cell up to the requested end depth
  when no later face is found.
- The former `lloyd_relax` API was renamed to `spring_relax` so it no longer
  claims to implement centroidal Voronoi tessellation. A true bounded Lloyd/CVT
  operation remains unimplemented.
- Nearest-neighbor radius estimation now uses an exact duplicate-safe k-d tree
  query instead of the original quadratic scan.

### Cameras and color

- COLMAP intrinsics preserve principal point and all current camera-model
  parameters; supervised images are rectified onto the explicit pinhole camera
  used by CPU and WGSL ray generation. Model-specific projection tests cover
  off-center, radial, and fisheye cases.
- Training, SH evaluation, image output, PSNR, backgrounds, and viewers now
  explicitly use display-referred sRGB code values without a hidden transfer
  function or tone map. Linear-light clients must decode explicitly.
- Distorted projection is handled in the reconstruction/training boundary; the
  runtime `PointCloudModel` intentionally carries cloud data rather than source
  capture-camera calibration.

### Remaining Gaussian backend gaps

- The shared icosahedron BLAS proxy now has the requested insphere radius; the
  original formula inflated every proxy axis by sqrt(3), increasing false RT
  candidates and pressure on the five-hit window.
- Standard channel-major 3DGS SH PLY and SPZ v2-v4 signed positions, opacity,
  higher SH, and quaternion streams now have known-value regression fixtures.
- Gaussian PLY, SPZ, and RadFoam/PowerFoam PLY now validate bounded headers,
  checked body sizes, complete schemas, model arrays, and fallible allocations.
  Public `try_detect_format` and `try_load*` entry points preserve IO and data
  errors; the original panic wrappers remain for compatibility. SPZ decoding
  streams attributes directly into the final model instead of retaining the
  compressed file and a second packed copy. The official Niantic
  `racoonfamily.spz` sample (932,560 points, SH3) validated in a CPU-only
  cgroup at a 212 MiB warm-run memory peak with zero swap or OOM events.
- Gaussian compositing now has an exhaustive CPU oracle ordered by each
  particle's maximum-response depth, matching the
  [official 3DGRUT implementation](https://github.com/nv-tlabs/3dgrut). The
  triangle ray-query path uses a lexicographic `(depth, point index)` cursor and
  complete-interval rescans, so its five-hit window changes work batching
  rather than omitting or proxy-face-ordering particles. Static WGSL validation
  passes; physical-GPU pixel parity and the window-size performance sweep
  remain blocked on driver recovery.
- No Gaussian training implementation exists.

### Scene layer

- RadFoam/PowerFoam-only scenes now select a dedicated compute pipeline with no
  ray-query extension, Gaussian buffers, acceleration-structure descriptors,
  or dummy geometry. The mixed and RadFoam-only shaders share binding and
  software-TLAS traversal modules and both pass static WGSL validation.
  Physical-GPU readback remains pending after driver recovery.
- All Gaussian clouds now bind independent TLAS and data-buffer array entries
  on Vulkan. Gaussian rays are transformed into each cloud's local space, so
  per-frame scene transforms do not rebuild point data or per-cloud TLASes.
  Static WGSL validation passes; rendered-pixel validation remains blocked on
  driver recovery. Blade does not yet implement resource binding arrays on
  Metal, so the multi-cloud scene layer remains Vulkan-only and needs a scalar
  or native bindless Metal path.
- RadFoam scene objects now seed traversal from the camera-containing local
  cell, using Euclidean distance for Voronoi clouds and exact power distance
  `|x-p|²-r²` for weighted clouds. The same seed rule is shared by standalone
  viewing, training path recording, CPU evaluation, and transformed scenes.
  They traverse from the camera while clipping integration to their
  software-TLAS interval. Physical-GPU validation remains pending.
- Affine-transformed rays now preserve the world-distance parameter under
  uniform and nonuniform scale; bounded support intersections accept the
  resulting non-unit object-space direction. Bounds include PowerFoam support
  radii and finite Gaussian proxy extents.
- The fixed first-sixteen-object scan is removed. Traversal deterministically
  visits every intersected object in lexicographic `(bounds entry, object
  index)` order without a fixed hit array. This is correctness-first O(N²)
  selection and treats whole overlapping clouds as ordered layers; physically
  interleaved volume integration and a scalable software TLAS remain open.
- Scene tests check state but not rendered pixels.
- A 2026-07-12 RadFoam-only dispatch probe reached a driver fault even after
  reducing the compute entry point to a constant texture write. The scene has
  since split out a no-ray-query RadFoam pipeline and removed dummy resources,
  but this new path has only static validation until the driver is recovered.
  A physical readback test remains the enabling gate.

### Project constraints

- The C-backed Qhull path is isolated behind the non-default `qhull` feature;
  the default dependency graph is Rust-only. Production-size exact Delaunay
  training currently opts into that feature because the available pure-Rust
  implementation exceeded the measured memory budget. A scalable Rust
  replacement remains preferable.

## Implementation stages

Each stage lands as one or more focused commits. Every commit must pass its
targeted tests and formatting; stage boundaries require workspace formatting,
clippy with warnings denied, and the full practical test suite.

Progress through 2026-07-13: Stages 0 and 1 are substantially complete. The
first versioned Bonsai smoke result now evaluates a freshly serialized PLY at
16.58 dB train / 17.00 dB held out, identical to live evaluation; exact DC SH
extension properties removed the prior 0.90/0.94 dB serialization loss. The
run used a 2 GiB/no-swap cgroup and peaked at 68.6 MiB. A controlled pure-Rust
Delaunay attempt reached its 8 GiB limit before training, while exact Qhull
completed adjacency in 0.02 seconds, so production benchmark protocols select
the isolated Qhull feature explicitly.

Stage 2 now has topology-safe opt-in position optimization, exact symmetric
adjacency caps, terminal-segment integration, reference position-gradient ×
cell-radius densification, explicit background compositing, and opt-in smooth
depth-variance regularization. Analytical unweighted position gradients now
match central finite differences on a fixed, smooth cell path. Densification
now collects maximum per-view ray contribution on the GPU at a deterministic
2× downsample, protects contributing cells and their adjacency neighbours,
suppresses dead-cell density, and reports max-step truncation. Held-out CPU
evaluation also distinguishes hard step-cap truncation from opacity, far-plane,
and terminal-cell exits and warns with an exact ray count. A matched reference
benchmark remains. The trainer now also implements RadFoam's exact
random transmittance-quantile depth-separation loss and half-training weight
ramp; the earlier smooth depth-variance term remains available as a separate
ablation. The color contract now explicitly follows reference RadFoam/3DGS:
training, SH appearance, backgrounds, PNG output, and PSNR use display-referred
sRGB code values. The viewer no longer applies an extra Reinhard curve to the
RadFoam backend. Lossless checkpoints now include a versioned trainer-state
sidecar for the sampling, quantile, and densification RNG streams as well as
the existing parameter/Adam safetensors; legacy resumes reconstruct fixed-draw
sampling streams by jumping the LCG to the absolute step. Stage 3 now persists
Čech radii and clips PowerFoam intervals to support spheres consistently on
CPU, production WGSL, and the GPU training recorder. An exact CPU oracle covers
the three active path roles (previous/current/next), radical-plane exits,
support-sphere entry/exit, skipped cells, and central finite differences. The
WGSL recorder stores those same position/radius Jacobians, and the meganeura
graph optimizes radii through a beta=100 softplus while periodically rebuilding
the discrete Čech graph and recorded paths. Static WGSL validation and the full
CPU-isolated workspace suite pass. Physical-GPU execution of the new integrated
gradient test is deferred until the wedged NVIDIA driver is recovered. Weighted
densification copies the parent's radius and optimizer ancestry while applying
the reference 5%-of-radius perturbation; reference-checkpoint and real-scene
training validation still remain.

Long-running topology optimization is now memory-bounded. The upstream
`qhull` 0.4 destructor omits Qhull's required short-arena cleanup, which leaked
about 70 MiB per 50,000-site rebuild in the quality run. The wrapper performs
the complete Qhull teardown explicitly, and a 16-rebuild Bonsai stress test
held cgroup memory at roughly 197–200 MiB after initialization (204 MiB peak,
zero swap and OOM events). Densification boundaries are also scheduled from
their own warmup/cadence instead of firing at every position-topology boundary;
checkpoints follow that independent cadence. The failed Radeon quality probes
were GPU ring timeouts at low host-memory use, not cgroup OOMs. Production-size
training is therefore isolated and pinned to the NVIDIA device while the AMD
driver path remains excluded from long runs.

A subsequent NVIDIA quality run confirmed the memory fix but did not produce a
valid benchmark result. It grew from 50,000 to the 200,000-cell target by step
7,500, reported no hard traversal truncation, and reduced the rolling training
loss from 0.7446 initially to 0.1868 at step 8,000. Near step 9,400 the NVIDIA
management interface and trainer both stopped responding in kernel waits,
without an Xid or other kernel fault record. Cgroup memory remained below
1.20 GiB of its 4 GiB limit with zero swap and OOM events; sampled VRAM was
approximately 525 MiB. This rules out host-memory exhaustion but does not by
itself distinguish an application-triggered driver defect from an independent
driver failure. The exact PLY and Adam-state checkpoint at step 9,000 was
verified intact, so the run is resumable after GPU recovery.

That incident exposed a fault in the benchmark harness: a synchronous
`nvidia-smi` sampler can hang in the same driver wait as the workload. GPU and
Vulkan probes now have deadlines, a telemetry timeout terminates the isolated
scope without waiting for the stuck probe, and `--cpu-only` retains cgroup
memory telemetry while skipping every GPU probe and denying GPU character
devices at the cgroup boundary. Synthetic stalled-probe tests cover the
preflight and in-run failure paths.

Long runs can also be divided into bounded process lifetimes with
`--stop-after-steps`. Segment endpoints force an exact checkpoint while the LR,
densification, and regularization schedules retain their original global step
budget. A segment is rejected if it would discard a partially accumulated
densification window; once the target count or densification cutoff is reached,
arbitrary endpoints are safe.

Benchmark provenance is now split explicitly. The original local Bonsai
fixture has 80 image files for a 292-image COLMAP reconstruction and remains a
subset smoke/quality protocol. A separate pinned `nvs-bench/mipnerf360`
fetch and manifest cover the complete scene: 292 image files, 292 registered
names with zero mismatches, one camera, and 206,613 sparse points. The 373 MB
download completed in a 1 GiB/no-swap scope at a 546 MiB memory peak with zero
OOM events. Its initial blade-volume budget is still internal, not a claim of
paper-matched hyperparameters.

### Stage 0: trustworthy baseline

1. Make the maintained pixel-batched trainer the default public workflow.
   (Done.)
2. Remove or update the obsolete whole-image implementation. (Done.)
3. Restore convergence tests on the maintained path and add an end-to-end
   COLMAP training test. (Done.)
4. Fix formatting, clippy, and GPU resource lifetime failures. (Done for all
   reproducible software issues; the host driver incident remains external.)
5. Add deterministic benchmark manifests recording dataset, split, cell count,
   resolution, optimizer steps, seed, hardware, and metrics.

Acceptance gate: documented training commands run; workspace fmt and clippy
pass; all non-hardware-optional tests pass; GPU tests have explicit skip
behavior rather than silent gaps.

### Stage 1: cameras, formats, and model invariants

1. Represent pixel-to-ray projection explicitly, including principal point and
   the supported COLMAP distortion models. (Done.)
2. Share camera and background conventions between training, CPU evaluation,
   and WGSL rendering. (Done for the rectified pinhole runtime contract.)
3. Validate model vector lengths and supported SH degrees at IO boundaries.
   (Done for Gaussian PLY, SPZ v2-v4, and RadFoam/PowerFoam PLY.)
4. Correct standard Gaussian PLY SH layout with external fixtures. (Done.)
5. Correct and complete SPZ v2 decoding with known-good fixtures. (Done;
   v3/v4 and an official production-size v4 sample are covered as well.)
6. Introduce a lossless native training checkpoint with optimizer state.
   (Done, including RNG state.)
7. Isolate the C-backed Qhull option from the default Rust-only build. (Done.)

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

Items 1-5 are implemented at the CPU-oracle, production-WGSL, recorder, and
training-graph levels. The physical-GPU integration test for items 3-4 is
compiled but intentionally not run while the host driver is wedged. Weighted
densification now uses the reference copied-radius split policy. Items 6-7 and a
real-scene radius-learning/densification ablation remain.

Acceptance gate: CPU, GPU, and brute-force bounded traversal agree; a reference
checkpoint renders within a defined image tolerance; trained radii improve a
fixed ablation rather than merely changing topology.

### Stage 4: Gaussian backend

1. Establish a trusted raster or reference 3DGRT image oracle.
2. Sweep hit-window and proxy bounds for accuracy and performance.
3. Add cloud transforms without rebuilding point data.
4. Decide whether native Gaussian reconstruction is justified after RadFoam and
   PowerFoam quality is established.

The CPU maximum-response oracle and exact batched ordering are implemented.
Cross-rendering a recognized checkpoint against official 3DGRUT, physical-GPU
pixel parity, and the hit-window performance sweep remain. Scene traversal now
applies Gaussian cloud transforms without rebuilding point data or the local
TLAS, but its rendered-pixel transform tests still require GPU recovery.

Acceptance gate: imported standard checkpoints match the oracle at documented
quality and performance; transformations pass rendered-pixel tests.

### Stage 5: multi-cloud engine

1. Support RadFoam-only and multiple-Gaussian scenes. (Implemented for the
   statically validated RadFoam-only and Vulkan mixed paths; physical-GPU
   readback and a Metal resource-binding path remain.)
2. Bind per-object layouts and locate correct RadFoam entry cells.
3. Preserve ray parameterization and optical depth under transforms. (Done in
   CPU logic and statically validated WGSL; rendered-pixel validation remains.)
4. Define exact ordering for intersecting cloud volumes. (Deterministic
   whole-cloud layering is defined; physical interleaving remains.)
5. Replace the first-sixteen-object scan with bounded actual-hit collection.
   (Done with exhaustive cursor selection; acceleration remains future work.)
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
