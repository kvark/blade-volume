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

### Remaining PowerFoam gaps

- Bounded traversal, persistent radii, exact active-path Jacobians, and
  trainable positive radii are implemented, but the new WGSL-Jacobian-to-
  meganeura integration test still needs to run on a recovered physical GPU.
- Weighted-cloud densification remains disabled until a radius-preserving split
  policy is defined and validated.
- No official pretrained checkpoint is published by the reference project, so
  cross-rendering and a matched training ablation remain outstanding.
- The reference quaternion, texel-site, and spherical-Voronoi appearance model
  remains intentionally deferred until weighted geometry is validated on a
  real scene.

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
- A 2026-07-12 RadFoam-only dispatch probe reached a driver fault even after
  reducing the compute entry point to a constant texture write. Pipeline
  creation succeeds, so Stage 5 must replace/audit the prototype descriptor
  arrays and add a readback test before enabling scene rendering; dummy ray
  tracing resources are not an acceptable workaround.

### Project constraints

- `qhull-sys` introduces C into a project whose stated dependency policy is
  Rust-only. It must be isolated behind a non-default feature, moved out of the
  core library, or replaced.

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
gradient test is deferred until the wedged NVIDIA driver is recovered; weighted
densification and reference-checkpoint validation still remain.

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

Items 1-5 are implemented at the CPU-oracle, production-WGSL, recorder, and
training-graph levels. The physical-GPU integration test for items 3-4 is
compiled but intentionally not run while the host driver is wedged. Items 6-7,
a real-scene radius-learning ablation, and a weighted densification split policy
remain.

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
