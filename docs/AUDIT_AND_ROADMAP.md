# Audit and implementation roadmap

Initial audit: 2026-07-12

Last updated: 2026-07-16

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

## Verdict

The idea is worth pursuing as a stage-gated, point-cloud-native research and
runtime project. It is not yet justified as a production-ready general graphics
engine. The shared `PointCloudModel` boundary, cloud-only scene taxonomy,
RadFoam/PowerFoam traversal, imported Gaussian path, and Rust-native training
pipeline form a coherent technical direction; the maintained trainer also
demonstrably learns a recognizable held-out reconstruction.

The unresolved risk is empirical rather than architectural. A complete-dataset
Bonsai run now reaches 17.50 dB train / 16.13 dB held-out PSNR after a fresh
serialized-model reload, but plateaus well below a compelling reference target.
The corrected paths pass the targeted NVIDIA/Vulkan physical-GPU gates,
including weighted differentiable traversal, Gaussian CPU/GPU parity, and
transformed-scene pixel readback. Whole-cloud layering is still not exact
interleaved volume compositing, the Metal multi-cloud binding design is open,
and no matched reference-trainer comparison exists. New rendering methods and
broad scene features should therefore remain behind these gates:

1. Match reference RadFoam within the Stage 2 quality tolerance on a complete,
   reproducible scene.
2. Demonstrate that learned PowerFoam radii improve a fixed held-out ablation.
3. Keep physical-GPU parity and transformed-scene pixel tests passing across
   supported vendors without driver faults or unbounded memory growth. The
   current NVIDIA/Vulkan gate passes; AMD long runs and Metal remain uncovered.
4. Define correct overlapping-cloud compositing before presenting the scene
   layer as general volumetric composition.

Failure at the first two gates should pause algorithm expansion and trigger a
focused comparison with the reference trainers. Success would justify
productionizing the point-cloud engine API without weakening the cloud-only
constraint.

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
- The blocker is quality evidence. A fresh, segmented run over all 292
  registered Bonsai images reached step 10,000 and the 200,000-cell target on a
  stable NVIDIA/Vulkan path. Its held-out curve flattened near 16.1 dB by step
  8,000, so the run was deliberately stopped before the nominal 20,400-step
  budget rather than spending more compute on an unchanged protocol. The
  corrected trainer still needs a reference-matched RadFoam run and controlled
  ablations.

### Remaining PowerFoam gaps

- Bounded traversal, persistent radii, exact active-path Jacobians, and
  trainable positive radii are implemented. The WGSL-Jacobian-to-meganeura
  integration suite now passes on a physical NVIDIA GPU, including finite
  differences, weighted intervals, topology rebuilds, densification, and
  multi-view/novel-pose cases.
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
- Pure-Rust Delaunay construction has fallible `try_compute_adjacency*` entry
  points beneath the compatibility wrappers. Offline RadFoam conversion now
  reports undersized or failed exact topology through `ConvertError` instead of
  panicking or silently substituting an approximate graph.
- Model-boundary CSR validation now requires monotonic ranges, in-range sorted
  unique lists, no self-edges, and a reverse edge for every neighbor. It cannot
  prove geometric completeness without rebuilding topology.
- Čech construction uses an immutable k-d tree that tolerates coincident and
  quantized sites. Mesh conversion now rebuilds Čech adjacency after assigning
  radii instead of retaining the preceding unweighted Delaunay graph.
- Traversal now integrates the terminal cell up to the requested end depth
  when no later face is found.
- The production path recorder applies the same maximum interval clamp to
  unweighted terminal segments as to weighted segments; previously the
  unweighted early-return path bypassed the configured bound.
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
- Camera model IDs and projection equations were rechecked against current
  [COLMAP `models.h`](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h).
  The single-focal `RADIAL_FISHEYE` layout is corrected, EUCM is supported,
  and equirectangular records are parsed/projectable but skipped by training
  because a 360-degree panorama cannot be represented by the pinhole runtime.
- COLMAP binary parsing now has fallible entry points beneath the compatibility
  wrappers. File-size-bounded record counts, fallible reservations, bounded
  image names, checked variable-array sizes, model IDs, duplicate cameras, and
  image-to-camera references are validated before data reaches training;
  invalid dimensions, focal lengths, poses, coordinates, and errors are
  rejected before they can seed NaNs.
- Training, SH evaluation, image output, PSNR, backgrounds, and viewers now
  explicitly use display-referred sRGB code values without a hidden transfer
  function or tone map. Linear-light clients must decode explicitly.
- The glTF converter decodes texture samples, combines base-color factors and
  ambient gain in linear light, then encodes once at the `PointCloudModel`
  boundary. It no longer stores linear midtones in an sRGB-coded model.
- glTF conversion now respects each base-color texture's coordinate set and
  wrap modes, supports non-indexed triangle primitives, uses the specified
  white default material, and rejects incomplete/out-of-range attribute data
  instead of indexing it unchecked.
- All decoded glTF image channel formats are converted from their raw 8-bit,
  16-bit, or float representation; luma-alpha channels retain alpha. The old
  path mistakenly reparsed raw high-bit-depth pixels as an encoded image and
  silently substituted black on failure.
- Curvature-aware surface sampling now normalizes its area-weighted boost, so
  it redistributes a fixed pre-rounding point budget toward features instead
  of silently growing the cloud as the boost increases. Gaussian footprints
  track half the local area-sampling spacing, including explicit surface-density
  scaling and curvature redistribution.
- glTF conversion follows the declared default scene (or the first scene when
  no default is declared), rather than merging alternative scenes into one
  cloud and duplicating or spatially mixing their geometry.
- Base-color sampling follows glTF's upper-left UV origin and the ratified
  [`KHR_texture_transform`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_transform/README.md)
  offset, rotation, scale, and texture-coordinate-set override semantics.
  `OPAQUE`, `MASK`, and `BLEND` alpha modes respectively ignore alpha, apply
  the declared cutoff, and modulate generated point opacity. `COLOR_0` is
  interpolated and multiplied into sampled base color; interior fallback color
  uses area-weighted triangle-centroid samples rather than tessellation count.
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
  rather than omitting or proxy-face-ordering particles. Physical-GPU pixel
  parity against the CPU oracle now passes. The official-checkpoint
  cross-render and window-size performance sweep remain outstanding.
- Scene Gaussian tracing now keeps its hardware query interval separate from
  the Gaussian's semantic support interval. Reusing the finite semantic bounds
  for triangle queries excluded conservative icosahedron proxy faces lying
  outside the ellipsoid support and could produce zero radiance. The local TLAS
  is now queried over the full forward interval, while maximum-response depth
  is filtered against semantic support before compositing.
- No Gaussian training implementation exists.

### Scene layer

- RadFoam/PowerFoam-only scenes now select a dedicated compute pipeline with no
  ray-query extension, Gaussian buffers, acceleration-structure descriptors,
  or dummy geometry. The mixed and RadFoam-only shaders share binding and
  software-TLAS traversal modules and both pass static WGSL validation and
  physical-GPU pixel readback.
- All Gaussian clouds now bind independent TLAS and data-buffer array entries
  on Vulkan. Gaussian rays are transformed into each cloud's local space, so
  per-frame scene transforms do not rebuild point data or per-cloud TLASes.
  Static WGSL validation and rendered-pixel tests with two independently bound
  Gaussian clouds pass. Blade does not yet implement resource binding arrays
  on Metal, so the multi-cloud scene layer remains Vulkan-only and needs a
  scalar or native bindless Metal path.
- RadFoam scene objects now seed traversal from the camera-containing local
  cell, using Euclidean distance for Voronoi clouds and exact power distance
  `|x-p|²-r²` for weighted clouds. The same seed rule is shared by standalone
  viewing, training path recording, CPU evaluation, and transformed scenes.
  They traverse from the camera while clipping integration to their
  software-TLAS interval. Physical-GPU validation passes for translated,
  rotated, uniformly scaled, and nonuniformly scaled bounded PowerFoam clouds.
- Affine-transformed rays now preserve the world-distance parameter under
  uniform and nonuniform scale; bounded support intersections accept the
  resulting non-unit object-space direction. Bounds include PowerFoam support
  radii and finite Gaussian proxy extents.
- The fixed first-sixteen-object scan is removed. Traversal deterministically
  visits every intersected object in lexicographic `(bounds entry, object
  index)` order without a fixed hit array. This is correctness-first O(N²)
  selection and treats whole overlapping clouds as ordered layers; physically
  interleaved volume integration and a scalable software TLAS remain open.
- Scene tests now read back rendered pixels for transformed bounded PowerFoam,
  two independent Gaussian bindings, anisotropic Gaussian rotation without a
  TLAS rebuild, and object-bounds/backend debug views. Exact interleaved
  overlapping-volume composition and cross-backend image equivalence remain
  untested.
- The public scene object taxonomy contains only implemented cloud backends:
  Gaussian and RadFoam/PowerFoam. Polygon meshes remain offline conversion
  input rather than a runtime scene-object escape hatch; a future SDF backend
  should add a concrete point-sampled representation instead of a placeholder.
- Gaussian hardware tracing does use one shared icosahedron triangle BLAS as a
  conservative candidate-generation envelope. It is an acceleration proxy for
  point indices, not user-visible polygonal geometry: there is no mesh scene
  object, triangle material, or polygon surface in the model/API contract.
- Scene traversal uses per-object SH/attribute metadata. Obsolete global
  metadata and the unimplemented backend-density debug mode were removed so
  public controls correspond to shader behavior.
- A 2026-07-12 RadFoam-only dispatch probe reached a driver fault even after
  reducing the compute entry point to a constant texture write. After rebooting
  into NVIDIA driver 595.71.05, the split no-ray-query path and the mixed scene
  path pass repeated physical readbacks without a fault. The fresh segmented
  training run also crosses the old failure region. This makes a deterministic
  application failure at that step less likely, while retaining cgroup and
  telemetry isolation as a required long-run safeguard.

### Project constraints

- A source-wide stub inventory found no repository-owned `TODO`/`FIXME` markers
  or executable `todo!`/`unimplemented!` placeholders in the implemented
  paths. The remaining deliberate panics are compatibility loaders/builders,
  documented representation preconditions, and internal shader or training
  invariants; new integration code should prefer the available fallible IO and
  Delaunay entry points. Exportable training buffers remain Vulkan-only because
  the upstream Metal/GLES allocation path is not implemented.
- The C-backed Qhull path is isolated behind the non-default `qhull` feature;
  both the core library and training crate keep it out of their default
  dependency graphs. Production-size exact Delaunay training opts into that
  feature explicitly because the available pure-Rust implementation exceeded
  the measured memory budget. A scalable Rust replacement remains preferable.
- The default `blade-volume` normal graph has ten direct dependencies and no
  repository-owned build script. Repository production `unsafe` is confined to
  Blade GPU context/resource mapping and the feature-gated Qhull teardown;
  converter, format, camera, and CPU traversal code remain safe Rust.
- Whole-workspace duplication is concentrated outside the core: the current
  viewer/autograd graph carries crates.io and git-source copies of
  `blade-macros`, two Wayland/calloop generations, and several normal transitive
  version splits. Align the Blade/egui pins before packaging, but do not churn
  working upstream revisions during algorithm validation.
- A current RustSec scan found four patchable lockfile issues; `anyhow`,
  `crossbeam-epoch`, `memmap2`, and the otherwise-unused target-specific
  `quinn-proto` entry are updated to their fixed releases. Two high-severity
  denial-of-service advisories remain in `quick-xml` 0.39.4
  ([RUSTSEC-2026-0194](https://rustsec.org/advisories/RUSTSEC-2026-0194),
  [RUSTSEC-2026-0195](https://rustsec.org/advisories/RUSTSEC-2026-0195)). It is
  present only through `wayland-scanner`'s build-time protocol generator, not a
  runtime untrusted-XML input, and the latest `wayland-scanner` 0.31.10 still
  pins the affected minor series. Track its upstream migration to
  `quick-xml >=0.41` instead of carrying a local parser fork. The scan also
  reports unmaintained `number_prefix`, `paste`, and `ttf-parser` transitively
  through meganeura/tokenizers/image and the viewer stack; replacing them is an
  upstream dependency-alignment task, not a core renderer change.
- CI no longer relies on the archived `actions-rs` toolchain and cargo actions.
  It exercises both default and all-feature workspace tests, all-feature clippy,
  workspace formatting, and a RustSec gate. The gate explicitly exempts only
  the two documented build-time `quick-xml` findings so any new advisory still
  fails the workflow.
- Workspace crates now expose the repository and MIT package metadata, and the
  regression harness is explicitly non-publishable. A crates.io package is
  still intentionally blocked: the core depends on the pinned, unreleased
  `blade-graphics` revision without a registry version requirement. Embedding by
  git/path is supported; publishing requires a compatible Blade release and
  versioned internal path dependencies rather than silently stripping the git
  pin during `cargo package`.

## Implementation stages

Each stage lands as one or more focused commits. Every commit must pass its
targeted tests and formatting; stage boundaries require workspace formatting,
clippy with warnings denied, and the full practical test suite.

Progress through 2026-07-16: Stages 0 and 1 are substantially complete. The
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
the discrete Čech graph and recorded paths. Static WGSL validation, the full
CPU-isolated workspace suite, and all 33 physical-GPU differentiable-renderer
tests pass. Weighted densification copies the parent's radius and optimizer
ancestry while applying the reference 5%-of-radius perturbation;
reference-checkpoint and real-scene radius ablations still remain.

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

The final combined CPU-only workspace gate (all-target/all-feature clippy,
default tests, and all-feature tests) completed under a 3 GiB hard limit with
zero swap and zero OOM events, but reached the limit and recorded 406
`memory.max` pressure events. CI or local combined gates should therefore use a
4 GiB scope, or split clippy/default/all-feature tests into separate 3 GiB
scopes; a 3 GiB combined scope is functional but needlessly reclaim-bound.

The final 2026-07-16 delivery gate split each command into its own 4 GiB/no-swap
scope. Formatting peaked at 7.4 MiB, all-target/all-feature clippy at 686.4 MiB,
default workspace tests at 3,304.6 MiB, and all-feature workspace tests at
2,491.5 MiB. Every command passed with zero swap, OOM events, or `memory.max`
pressure. This independently confirms that 4 GiB is an appropriate test-scope
limit; 3 GiB is not reliably roomy enough for the default test link phase.

A previous NVIDIA quality run confirmed the memory fix but ended in a driver
incident rather than a valid benchmark result. It grew from 50,000 to the
200,000-cell target by step 7,500, reported no hard traversal truncation, and
reduced rolling training loss from 0.7446 initially to 0.1868 at step 8,000.
Near step 9,400 the NVIDIA management interface and trainer both stopped
responding in kernel waits, without an Xid or other kernel fault record. Cgroup
memory remained below 1.20 GiB of its 4 GiB limit with zero swap and OOM events;
sampled VRAM was approximately 525 MiB. This ruled out host-memory exhaustion
but could not distinguish an application-triggered driver defect from an
independent driver failure. The exact step-9,000 PLY and Adam checkpoint was
verified before the reboot, but its `/tmp` location was intentionally
non-durable and the file did not survive.

That trainer task remained in kernel/driver execution even after its scope
received `SIGKILL`: one thread, roughly 172 MiB current host memory, zero swap,
and no cgroup OOM. A 1% runtime CPU quota did not throttle the stuck kernel
execution. The 2026-07-16 host reboot cleared the task and brought up NVIDIA
driver 595.71.05. The replacement segmented run passed step 9,400 and reached
step 10,000 without a fault, so the old boundary is not a deterministic failure
in the current environment.

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

The first complete-dataset run uses that pinned 292-image reconstruction, 255
training views, eight held-out every-eighth views, 128×128 pixels, SH degree 3,
50,000 initial cells, a 200,000-cell target, and the schedule recorded in
`benchmarks/bonsai_full_quality.toml`. The manifest's first segment now reaches
the step-2,000 warmup boundary; the trainer correctly rejects an earlier
endpoint because it would discard a partially accumulated densification
window. All later 1,000-step segments completed normally. The following
segment-end diagnostics evaluate the in-memory model after saving:

| Global step | Cells | Train PSNR | Held-out PSNR |
| ---: | ---: | ---: | ---: |
| 2,000 | 57,313 | 13.08 dB | 13.08 dB |
| 3,000 | 75,636 | 14.98 dB | 14.14 dB |
| 4,000 | 99,798 | 16.15 dB | 15.23 dB |
| 5,000 | 131,682 | 16.61 dB | 15.47 dB |
| 6,000 | 173,695 | 16.43 dB | 15.79 dB |
| 7,000 | 200,000 | 17.32 dB | 15.95 dB |
| 8,000 | 200,000 | 17.46 dB | 16.15 dB |
| 9,000 | 200,000 | 17.47 dB | 16.13 dB |
| 10,000 | 200,000 | 17.50 dB | 16.13 dB |

An independent CPU evaluation reloaded the final PLY and reproduced 17.50 dB
train / 16.13 dB held out exactly. The model has 3,035,704 directed adjacency
entries. Its durable ignored checkpoint lives under
`target/audit-runs/bonsai-full-93c996f/` and includes PLY, safetensors,
trainer-state, step, cycle, and RNG sidecars. The largest training segment used
1,200,984,064 bytes (1.118 GiB) of its 4 GiB cgroup, with zero swap, OOM events,
or GPU recovery markers. The independent evaluation peaked at 600,907,776
bytes.

The curve is effectively flat from step 8,000 through 10,000; continuing the
same nominal 20,400-step protocol is therefore low-value until an ablation
changes the result. Profiling also shows that exhaustive all-view contribution
scans dominate densification as the cloud grows, while exact Qhull rebuilds
every 100 position steps remain the main cost after reaching 200,000 cells.
The exhaustive scan must remain the correctness oracle while a deterministic
sampled or incremental alternative is evaluated; performance alone is not a
reason to silently change pruning decisions.

### Stage 0: trustworthy baseline

1. Make the maintained pixel-batched trainer the default public workflow.
   (Done.)
2. Remove or update the obsolete whole-image implementation. (Done.)
3. Restore convergence tests on the maintained path and add an end-to-end
   COLMAP training test. (Done.)
4. Fix formatting, clippy, and GPU resource lifetime failures. (Done for all
   reproducible software issues; the historical driver incident is isolated
   and the current NVIDIA physical gate passes.)
5. Add deterministic benchmark manifests recording dataset, split, cell count,
   resolution, optimizer steps, seed, hardware, and metrics. (Done; Qhull
   benchmark commands also opt into the non-default feature explicitly.)

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

Items 1-6 are implemented and covered by CPU and physical-GPU tests. Item 7 is
the gating work: the complete-dataset curve above is an internal protocol, not
a reference-matched result.

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
training-graph levels and pass physical-GPU integration. Weighted densification
uses the reference copied-radius split policy. Items 6-7 and a real-scene
radius-learning/densification ablation remain.

Acceptance gate: CPU, GPU, and brute-force bounded traversal agree; a reference
checkpoint renders within a defined image tolerance; trained radii improve a
fixed ablation rather than merely changing topology.

### Stage 4: Gaussian backend

1. Establish a trusted raster or reference 3DGRT image oracle.
2. Sweep hit-window and proxy bounds for accuracy and performance.
3. Add cloud transforms without rebuilding point data.
4. Decide whether native Gaussian reconstruction is justified after RadFoam and
   PowerFoam quality is established.

The CPU maximum-response oracle, exact batched ordering, and physical-GPU pixel
parity are implemented. Scene traversal applies Gaussian cloud transforms
without rebuilding point data or the local TLAS, and transformed-pixel tests
pass. Cross-rendering a recognized checkpoint against official 3DGRUT and the
hit-window performance sweep remain.

Acceptance gate: imported standard checkpoints match the oracle at documented
quality and performance; transformations pass rendered-pixel tests.

### Stage 5: multi-cloud engine

1. Support RadFoam-only and multiple-Gaussian scenes. (Implemented and
   physically read back on Vulkan; a Metal resource-binding path remains.)
2. Bind per-object layouts and locate correct RadFoam entry cells.
3. Preserve ray parameterization and optical depth under transforms. (Done in
   CPU logic, WGSL, and rendered-pixel validation.)
4. Define exact ordering for intersecting cloud volumes. (Deterministic
   whole-cloud layering is defined; physical interleaving remains.)
5. Replace the first-sixteen-object scan with bounded actual-hit collection.
   (Done with exhaustive cursor selection; acceleration remains future work.)
6. Add rendered-pixel tests for translation, rotation, uniform scale,
   nonuniform scale, mixed backends, and overlapping volumes.

Translation, rotation, uniform scale, nonuniform scale, bounded PowerFoam, two
independent Gaussian bindings, and backend/bounds debug output pass physical
readback. Exact mixed-backend equivalence and overlapping-volume compositing
tests remain with the interleaving design.

Acceptance gate: a scene made exclusively from independently transformable
cloud objects renders deterministically and agrees with equivalent standalone
backend renders.

## Prioritized improvement plan

The next cycle should improve evidence and the existing cloud paths before
adding another representation. Polygonal geometry remains offline conversion
input only; none of these steps adds a triangle scene object or polygonal
material path.

### P0: explain and close the RadFoam quality gap

1. Pin a reference RadFoam revision and reproduce one run on the same 292 image
   files, train/held-out indices, scale, initial/target cell counts, SH degree,
   backgrounds, and effective ray budget. Record reference-side serialized
   renders and PSNR rather than quoting a paper number with a different setup.
2. Compare initialization, optimizer parameter groups, learning-rate curves,
   opacity parameterization, geometry update cadence, pruning decisions,
   densification samples, and topology/path refresh timing step by step. Add a
   small deterministic trace fixture for every discovered semantic difference.
3. Run a controlled matrix from identical initialization: appearance-only;
   position optimization with fixed topology; position optimization with exact
   rebuilds; densification/pruning disabled and enabled; quantile loss disabled
   and enabled; and topology cadences 100/250/500. Evaluate both train and
   held-out views from a reloaded PLY and retain wall time, cell count,
   adjacency size, truncation counts, cgroup peak, and GPU fault telemetry.
4. Do not declare Stage 2 complete until the same-budget result is within
   0.5–1.0 dB of the reference or the remaining difference is isolated to a
   documented unsupported feature.

### P1: remove scaling bottlenecks without changing decisions silently

1. Prototype deterministic stratified-view contribution sampling and an
   incremental per-cell accumulator. Keep the exhaustive all-view scan as the
   oracle; require pruning/densification decision agreement and a fixed PSNR
   tolerance before changing the default.
2. Measure exact topology cadences 100/250/500 after the cloud reaches its
   target size. Retain feature-gated Qhull as the production-size oracle while
   investigating a memory-bounded Rust implementation; runtime geometry stays
   point-cloud-only in either case.
3. Add per-phase timing around recording, optimization, contribution scans,
   downloads, topology construction, and evaluation so a long run explains its
   cost without profiler-only evidence.

### P1: validate PowerFoam and Gaussian semantics on real assets

1. On the winning RadFoam configuration, compare fixed equal radii against
   trainable positive radii from identical seeds. Require a held-out improvement
   and stable cell/topology statistics before implementing the full quaternion
   and spherical-Voronoi appearance model.
2. Obtain or train a reference PowerFoam asset and cross-render it against the
   bounded-power CPU oracle and production WGSL.
3. Cross-render a recognized Gaussian checkpoint against official 3DGRUT and
   sweep the ray-query batch window for invariant pixels, query count, and frame
   time. The conservative triangle BLAS remains an invisible point-candidate
   accelerator, not polygonal scene geometry.

### P2: finish engine-level composition

1. Define exact interleaved optical-depth composition for overlapping clouds,
   then test mixed RadFoam/PowerFoam/Gaussian scenes against standalone
   segment oracles. Whole-cloud layer ordering is deterministic but is not the
   final physical model.
2. Implement a Metal-compatible per-cloud binding strategy and run the same
   transformed-pixel suite on Metal. Accelerate the exhaustive software-TLAS
   cursor only after pixel equivalence is locked down.
3. Defer new SDF and compute-splat backends until the Stage 2 quality gate is
   resolved; broadening the API before the core training result is understood
   would increase surface area without reducing the primary project risk.

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
