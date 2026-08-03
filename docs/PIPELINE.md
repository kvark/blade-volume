# Pipeline Plan: phone → train → view

Status: in-progress. Living document — keep it short, edit as we learn.

## North star

```
phone video ──▶ COLMAP ──▶ blade-volume-train ──▶ PointCloudModel ──▶ blade-volume-view
                              (meganeura)
```

The library boundary is `PointCloudModel`, including its optional PowerFoam radii. The
viewer doesn't know how the model was produced; the trainer doesn't know how it gets
rendered. Both link `blade-volume` for shared types and shaders.

## Non-goals

- No Python, no PyTorch, no Burn.
- No new representations beyond what we render. The maintained radiance-field
  trainer optimises RadFoam or bounded PowerFoam. Reconstruction now emits
  finite surface-Gaussian particles for the relightable path; direct training
  of volumetric 3DGS transforms is still not implemented. See
  [GAUSSIAN_RECONSTRUCTION_PLAN.md](GAUSSIAN_RECONSTRUCTION_PLAN.md).
- No on-device capture code initially — phone capture stays manual (record + transfer).
  We'll automate that later if and only if the rest of the pipeline is solid.

## Milestones

Each milestone is sized to fit in a handful of focused sessions and lands behind passing
`cargo clippy --workspace --all-targets --all-features -- -D warnings`, default
and all-feature workspace tests, and workspace formatting.

### M1 — Shader modularisation (foundation)

Today the test harness re-implements `// #include` for WGSL (`radfoam_gpu_vs_cpu.rs`).
Promote it to the library so M2 and M3 don't have to duplicate it.

- Move the `preprocess_shader` helper into `blade-volume/src/shaders.rs`.
- Expose a `compose(entry)` returning a fully-expanded WGSL string.
- Have `gpu/{gaussian,radfoam}.rs` and `blade-volume-view` use it.
- Tests: keep the existing GPU-vs-CPU test working unchanged.

Risk: low. Pure refactor.

### M2 — PowerFoam adoption

Power Foam = weighted Voronoi (power diagram) instead of plain Voronoi. Per-site weight,
radical-plane faces, Čech-complex adjacency. The PowerFoam paper bundles a much richer
appearance model (per-cell quaternion + texel sites + spherical-Voronoi colours); we
defer all of that and reuse our SH path. Adoption order:

#### M2a — `radius` in `PointCloudModel`

- Add `pub radii: Option<Vec<f32>>` to `PointCloudModel`.
- Round-trip through the RadFoam PLY reader/writer: optional `radius` float property.
- No shader changes yet — render code ignores the radii.
- Tests: extend the PLY round-trip test with a radii-present case.

#### M2b — Čech-complex adjacency builder (done)

- `adjacency::compute_cech(points, radii, config)` emits an edge `{i,j}` when
  `|p_i - p_j| ≤ r_i + r_j`; returns the existing CSR `Adjacency`.
- Candidates are pruned with a `kiddo` k-d tree range query (radius
  `r_i + r_max`), then the exact overlap predicate filters them.
- `PointCloudModel::compute_adjacency_default` dispatches: Čech when
  `radii.is_some()`, Delaunay otherwise.
- `kiddo` promoted from dev-dep to runtime dep on `blade-volume`.
- `train_colmap --powerfoam-reference-radii` initializes each support with
  PowerFoam's mean of the eight nearest-site distances (including self), caps
  it to 10% of projected half-image height in every visible training camera,
  and builds the resulting Čech graph. This is the faithful initializer;
  `--cech-radius` remains the simpler nearest-neighbour-factor experiment.

#### M2c — Power Foam WGSL (done)

Implementation deviates from the original plan in one way: rather than fork a new
shader / backend, we extend the existing `radfoam_trace.wgsl` to take a
`rf_get_radius` accessor and use the radical-plane formula

```
shift       = 0.5 + 0.5 * (r_i² - r_j²) / |p_j - p_i|²
face_origin = p_i + shift * (p_j - p_i)
face_normal = p_j - p_i
```

When radii are zero the shift collapses to `0.5` and the plane is the standard
Voronoi bisector — so unweighted clouds traverse identically to before.

To avoid adding a bind-group entry, the per-point radius lives in the otherwise-
unused `.w` channel of `g_points`. `gpu/radfoam.rs` writes
`model.radii.as_deref().map_or(0.0, |r| r[i])` into that slot at upload.
Both shader includers (`radfoam.wgsl` single-object, `scene_traverse.wgsl`
multi-object) gained a one-line `rf_get_radius` accessor.

Viewer auto-selection: no change needed — `RadFoamBackend` now handles both
weighted and unweighted models. `radii.is_some()` produces Power Foam traversal
automatically.

#### M2d — Cross-check (done, self-consistency)

External cross-check against a PowerFoam checkpoint is parked: it needs a CUDA-
trained scene we don't have. Instead we proved internal consistency:

- `radfoam_cpu_ref.rs` (CPU reference tracer) now uses the same radical-plane
  formula as `radfoam_trace.wgsl`. A `read_radius` helper returns 0 when
  `model.radii` is `None`, so unweighted clouds still trace as before.
- New `powerfoam_gpu_matches_cpu_with_radii` test runs the same fixture as the
  plain RadFoam regression but with asymmetric per-point radii (spine 0.015,
  branches 0.030) — the radical plane shifts noticeably away from the bisector,
  so any mismatch between WGSL and CPU formulas would surface here. Both
  tracers match within f16 precision.
- The original `radfoam_gpu_matches_cpu_on_tiny_fixture_for_some_pixels`
  regression and the new radii test now share a `assert_gpu_matches_cpu(...)`
  helper.

When a real PowerFoam scene shows up, conversion must remain Rust-only: either
consume an upstream interchange export or add the smallest required checkpoint
reader to a tool crate. No Python runtime becomes part of this project.

#### M2e — Differentiable weighted geometry (implemented and device-validated)

- The CPU path oracle records the exact active-branch derivative of every
  independently splatted, sphere-clipped interval with respect to its entry,
  current, and exit site's position and radius. Central finite differences
  cover radical planes and support spheres.
- The GPU recorder writes the same three `vec4(position, radius)` Jacobians,
  raw interval, and ray-relative reference tangent. The graph evaluates the stable local form
  `dt_ref + tangent_actual - tangent_ref`; unweighted training keeps the
  compact path layout.
- Weighted training optimizes a beta=100 softplus radius parameter and uses the
  recorder value plus its local geometry tangent between periodic Čech/path
  rebuilds. Radius and position learning rates are independent. Weighted
  densification copies the parent radius and optimizer ancestry and perturbs
  both sites at 5% of the support radius, matching the reference resampler's
  geometry policy without introducing its deferred normal semantics.
- Static WGSL validation, CPU-isolated workspace tests, and physical-GPU
  Jacobian/meganeura integration pass. A translated-world oracle also rejects
  numerically unstable absolute affine intercepts.

#### M2f — PowerFoam interpenetration loss (implemented, opt-in)

- `--interpenetration-weight` adds the reference squared-overlap objective
  `sum(max(r_i + r_j - |p_i - p_j|, 0)^2)` over directed Čech edges. Its
  weight decays exponentially to one-thousandth of the initial value over the
  global training horizon, including across checkpoint resumes.
- The graph uses an exact local first-order distance at each topology snapshot
  and differentiates both live positions and positive radii. A deterministic,
  stratified edge sample estimates the complete directed sum; the default
  `--interpenetration-samples 4096` bounds graph size independently of cloud
  size and needs no extra dependency or persistent RNG state.
- A physical two-site test checks the scalar loss and both position/radius
  gradients. Complete Mip-NeRF-360 Bonsai gates show that the term controls
  graph growth, but its useful scale depends strongly on scene units, radius
  learning rate, and Adam epsilon. It therefore remains off by default.
- At the exploratory high radius rate, `1e-10` improves all-37 held-out PSNR
  from 11.62 to 11.73 dB while reducing directed edges from 1,089,252 to
  880,212. Training is 6.1% slower because the sampled backward pass costs
  more than the smaller topology saves. The paper's `1e-4` weight is only
  stable with its much smaller radius rate in this protocol; pairing it with
  the exploratory rate collapses support.

#### M2g — Correct compute-splat path discovery (implemented for training/evaluation)

The original weighted trainer reused RadFoam's camera-seeded adjacency walk.
That is not a valid way to discover bounded PowerFoam supports: a Čech graph
contains only overlapping balls and may have many disconnected components, so
the walk can terminate before reaching any photographed surface. On Bonsai the
failure appeared as zero mean segments and the black-background 9.33 dB score
after 2,000 steps.

- One GPU workgroup per sampled ray scans support spheres in parallel. A
  second pass clips every hit against all radical planes in its Čech row and
  deterministically selects the surviving intervals front-to-back. A ball
  outside that row cannot win power distance inside the current ball, because
  non-overlap makes its power distance positive wherever the current ball's is
  non-positive.
- A CPU oracle and physical GPU tests cover disconnected components, exact
  interval/Jacobian parity, translated geometry, and compact evaluation paths.
  Candidate rows are bounded to `max(4 * max_steps, 256)` entries; synchronized
  training and evaluation reject overflow instead of silently truncating.
- Headless weighted evaluation now uses the same compute-splat semantics. The
  existing RadFoam walk remains unchanged for unweighted clouds and the
  interactive PowerFoam viewer remains on the approximate walk until projected
  tile binning makes splats practical at window resolutions.
- Complete Bonsai gates at 128², 50,000 initial sites, 4,096 rays/update and 16
  views/update move from 9.33 dB after 2,000 broken-walk steps to 11.67 dB after
  10 splat steps, 12.55 dB after 100, and 13.51 dB after 2,000. The 2,000-step
  run reaches loss 0.1507, grows to 57,500 sites without hard pruning, takes
  150.9 seconds, peaks near 1.1 GB host memory under a 6 GB cgroup, and records
  no memory or candidate-overflow event.

The exhaustive gather remains as the correctness fallback and for sparse
multi-view mini-batches. M2i adds a projected index for dense camera batches.

#### M2h — Device-resident PowerFoam resampling statistic (implemented and gated)

Weighted densification previously downloaded the complete position-gradient
table after every optimizer step, accumulated its norm on the CPU, and then
multiplied it by support radius. Besides costing 26.6 seconds in the matched
2,000-step run, this was not PowerFoam's resampling signal.

- The differentiable graph now computes each segment's
  `T × alpha × L1(cell_color, target)` responsibility. A frozen zero-forward
  probe receives that value as its gradient, so its Adam first moment keeps the
  per-site EMA on the GPU. The table is read only at a resampling boundary.
- Parent sampling caps the statistic at its survivor-set 99th percentile and
  samples without replacement. When a site is split, its inherited statistic
  is divided among the parent and children so their total probability is
  conserved. Unweighted RadFoam retains position-gradient × cell-radius
  sampling unchanged.
- On the same Bonsai 128²/50,000-site/4,096-ray/16-view gate, 2,000-step
  training falls from 150.9 to 128.9 seconds (14.6%). Gradient readback falls
  from 26.6 seconds to zero; added graph work raises GPU wait from 101.7 to
  105.1 seconds. Loss changes from 0.1507 to 0.1505, and the resulting
  57,500-site model scores 13.72/13.52 dB train/all-37 versus 13.70/13.51 dB.
  The isolated run peaks at 1.11 GB under a 6 GB cgroup, with no swap, pressure,
  OOM, candidate overflow, or GPU fault.

The probe currently inherits the configured optimizer's first-moment horizon
(`beta1 = 0.9` in this gate), while the reference implementation uses 0.99 for
this statistic. The signal, cap, and split conservation match; a dedicated
moment override is a lower-priority ablation if resampling noise becomes the
next measured quality limit.

#### M2i — Conservative projected PowerFoam candidates (implemented and gated)

- A parallel projection pass writes conservative screen bounds for every
  support sphere. One workgroup per 16×16 tile compacts overlapping sites into
  a bounded row, and each ray still performs the exact sphere and radical-plane
  tests. Candidate ordering cannot affect output because interval selection
  retains the depth/index tie-break.
- Tile occupancy retains its unbounded count. If a row exceeds its storage
  budget, rays in that tile automatically execute the original exhaustive
  scan. The implementation needs only workgroup atomics; an early device-scope
  atomic design was rejected after Vulkan validation caught unsupported memory
  scope. A physical GPU test forces 16,385 sites into one tile and proves the
  exhaustive fallback against the CPU oracle.
- Dense headless evaluation and large per-camera training/contribution batches
  use projected rows. Sparse mixed-view training keeps the exhaustive gather:
  on the selected 4,096-ray/16-view shape, indexing 256 rays per camera was
  neutral (46.28 versus 46.19 seconds for the matched 500-step segment) while
  allocating extra scratch.
- On the 100,569-site Bonsai step-4,000 checkpoint, maximum occupancy is
  8,945/16,384 sites per tile and 130/1,024 exact sphere hits per ray, with no
  fallback. Three alternating 72-view runs average 8.79 seconds projected
  versus 9.09 seconds exhaustive (3.4% faster); the complete 292-view pass is
  33.55 versus 35.08 seconds (4.4%). Train/test PSNR remains exactly
  14.65/14.35 dB. The full pass peaks at 194 MB under the 6 GB cgroup with no
  swap, pressure, OOM, candidate overflow, or GPU fault.

The interactive viewer still uses the adjacency walk for weighted clouds.
Promoting the now-bounded compute-splat tracer into that backend, including
resize and live settings handling, is the remaining presentation step.

#### M2j — Packed weighted path linearization (implemented and gated)

- The differentiable PowerFoam path now activates the radius table once and
  packs each live site as `(x,y,z,r)`. Previous/current/next roles gather that
  table and dot directly with the recorder's existing `vec4` Jacobians. This is
  algebraically identical to separately gathering positions and radii, but it
  removes repeated million-slot radius activations and six large Jacobian
  split/copy passes. Parameter names, optimizer state, and checkpoints do not
  change.
- On the 57,500-site, 4,096-ray, 256-step profile, the optimizer graph falls
  from 432 to 396 dispatches, 30.30 to 27.45 ms GPU time, 3.06 to 2.63 GB of
  logical scratch, and 1.99 to 1.50 GB of physical allocations. The matched
  500-step segment falls from 45.59 to 43.93 seconds.
- A complete 2,000-step checkpoint resume crosses four densification/resource
  rebuilds and reaches the same 100,569-site target in 321.50 versus 326.71
  seconds. Its cgroup peak falls from 1.287 GB to 0.857 GB with no swap,
  pressure, OOM, or GPU fault. Train/test PSNR is 14.64/14.33 dB versus
  14.65/14.35 dB; the 0.01–0.02 dB variation accompanies a 0.1% change in the
  atomically sampled topology and is below the precision of this stochastic
  gate.

#### M2k — Exact path-budget telemetry (implemented and gated)

- Every GPU-recorded ray now writes one host-visible status word containing
  its active segment count and a hard-truncation bit. Both the adjacency walk
  and the disconnected-support PowerFoam splat path distinguish a row that
  ended exactly at `max_steps` from one that still had a valid segment. Tests
  force both cases on a physical GPU. Training reports one aggregate and the
  evaluator reports one aggregate per view set instead of silently clipping.
- On the 100,569-site Bonsai checkpoint, a 64-step all-view render truncates
  3,646 of 4,784,128 rays (0.0762%, across 187 of 292 views), although the
  aggregate PSNR happens to round to the same two decimals. At 128 steps the
  observed maxima are 100 for training views and 96 for held-out views, with
  zero truncation. The standalone evaluator default is therefore 128 rather
  than 96; 64 remains an explicitly approximate option for this scene scale.
- A matched 2,000-step resume at 128 steps records 8,192,000 optimizer rays,
  reaches a maximum of 83, and truncates none while crossing four
  densification boundaries. Training falls from 321.50 to 297.85 seconds
  (7.4%) and the cgroup peak from 0.857 to 0.575 GB (32.9%) relative to the
  exact 256-step run. Train/test PSNR is 14.66/14.35 dB versus 14.64/14.33 dB;
  both runs reach 100,569 sites and the small positive delta is within their
  atomically sampled topology variation. No run used swap or recorded
  memory-pressure, OOM, or GPU-fault events.

#### M2l — Cached PowerFoam path intervals (implemented and gated)

- The compute-splat recorder now clips every sphere candidate against its
  radical planes once, during the unchanged first selection scan. It caches
  the effective entry depth, two limiting face depths, and their neighboring
  site indices for the later front-to-back scans. Candidate discovery,
  ordering, tie-breaking, interval differentials, and bounded-row rejection
  remain unchanged; only repeated adjacency clipping is removed.
- The cache costs five device-only scalar words per candidate. At the selected
  4,096-ray/128-step training shape this adds 40 MiB of scratch; a dense 128²
  evaluation adds 160 MiB. Physical CPU/GPU path tests cover disconnected
  supports, exact-cap and truncated rows, projected overflow, and weighted
  Jacobians. Rendering the same 100,569-site checkpoint with the parent and
  cached recorders produces byte-identical PNGs for all eight selected
  held-out views.
- On a matched 100-step segment, GPU step wait falls from 36.338 to 7.222
  seconds (80.1%), training from 40.011 to 11.346 seconds (71.6%), and the
  complete command from 45.91 to 16.75 seconds (63.5%). Printed loss, final
  adjacency, and all 409,600 path statuses are identical.
- A complete 2,000-step resume crosses four densification/resource rebuilds,
  reaches the same 100,569-site target, and records zero truncation over
  8,192,000 rays. Training falls from 297.847 to 123.026 seconds (58.7%) and
  GPU wait from 252.257 to 77.617 seconds (69.2%). The cgroup peak rises from
  575 to 604 MB and sampled GPU memory from 1,263 to 1,295 MiB. Fresh-Ply
  train/test PSNR is 14.64/14.34 dB versus 14.66/14.35 dB; the 0.01–0.02 dB
  change accompanies a 0.3% atomically sampled topology difference. The
  complete all-view evaluation falls from 34.73 to 19.47 seconds. Every scope
  remains below 6 GB with zero swap, pressure, OOM, candidate overflow, or GPU
  fault.

#### M2m — Parallel Čech topology rebuilds (implemented and gated)

- Čech rows are independent k-d-tree queries, so the builder now partitions
  them across the standard library's available workers and assembles the
  chunks back in point-index order. Each CSR row still takes the exact overlap
  predicate and the existing sort/dedup path; there is no approximate
  topology, new dependency, or output-order change. Work is enabled above
  4,096 sites per worker and capped at 16 workers to bound stacks and allocator
  pressure. A forced one-versus-four-worker test produces identical CSR arrays.
- On the 100,569-site matched 100-step gate, the topology phase falls from
  2.168 to 0.466 seconds (78.5%), training from 11.346 to 8.800 seconds (22.4%),
  and the complete command from 16.75 to 12.89 seconds (23.0%). Initial and
  final directed-edge counts, printed loss, and path telemetry remain exact.
  Twelve logical workers take 0.47 seconds versus 0.59 seconds for six
  physical-core workers on the gate machine; their one-step cgroup peaks are
  580 and 512 MB respectively.
- A complete 2,000-step resume reduces accumulated topology time from 33.347
  to 7.427 seconds (77.7%), training from 123.026 to 96.246 seconds (21.8%),
  and whole-command time from 127.76 to 100.52 seconds (21.3%). It crosses all
  four growth/resource boundaries, reaches the same 100,569 sites, and records
  zero truncation over 8,192,000 rays. Fresh-Ply quality is 14.66/14.36 dB
  train/test versus 14.64/14.34 dB; the small positive delta accompanies a
  0.2% atomically sampled topology difference. Host peak rises from 604 to
  894 MB while sampled GPU memory remains 1,295 MiB. No scope records swap,
  pressure, OOM, candidate overflow, or a GPU fault.

#### M2n — Batched multi-view path recording (implemented and gated)

- Sparse mixed-view PowerFoam training now records all disjoint camera slices
  through one exhaustive-candidate gather pass and one front-to-back record
  pass. Each dispatch retains its own camera constants, pixel offset, and ray
  count; the only removed work is the compute barrier between independent
  camera slices. Projected-tile batches and unweighted adjacency walks keep
  the existing sequential path because their scratch or traversal semantics
  differ. The public batch entry point validates ordered, disjoint output
  ranges and falls back automatically.
- A physical-GPU oracle binds two distinct cameras inside both shared passes
  and matches CPU cells, previous/next roles, intervals, path status, and all
  weighted geometry Jacobians. The complete ten-case path suite also covers
  disconnected supports, exact-cap and truncated rows, projected overflow,
  and translated geometry. In a matched 100-step gate, GPU wait falls from
  7.203 to 2.709 seconds (62.4%), training from 8.800 to 4.418 seconds (49.8%),
  and whole-command time from 12.89 to 8.64 seconds (33.0%). A repeat records
  2.718 seconds of GPU wait and 8.39 seconds whole-command time.
- The 100-step baseline-to-batched numerical delta is smaller than ordinary
  batched repeat variation: maximum position/SH differences are
  3.8e-6/7.9e-6 versus 1.4e-5/1.4e-5 between two batched runs. CSR topology is
  byte-identical, as is the sampling/trainer-state sidecar. This is consistent
  with the existing floating-point atomic accumulation order rather than a
  path-semantic change.
- A complete 2,000-step resume crosses the same four growth boundaries and
  reaches exactly 57,500 → 66,125 → 76,044 → 87,451 → 100,569 sites. Training
  falls from 96.246 to 67.071 seconds (30.3%), GPU wait from 76.927 to 47.745
  seconds (37.9%), and whole-command time from 100.52 to 71.13 seconds (29.2%).
  All 8,192,000 rays remain untruncated. Fresh-Ply train/test quality is
  14.64/14.34 dB versus 14.66/14.36 dB, within the established stochastic
  gate. Host peak is 652 MB, with zero swap, pressure, OOM, candidate overflow,
  or GPU fault.

### M3 — Training crate scaffolding

New crate: `blade-volume-train`. Depends on `blade-volume` + `meganeura`. Never the
reverse.

#### M3a — Crate skeleton (done)

- `blade-volume-train` crate wired into the workspace, depends on
  `blade-volume` + `meganeura 0.2`.
- `TrainerState` mirrors the optimisable parts of `PointCloudModel` but stores
  density and radius in their `softplus` pre-image, matching what PowerFoam
  does. Free parameters: positions, SH coefficients, sh_degree.
- `TrainerState::from_model` / `::to_model` round-trip a `PointCloudModel`
  through trainer space; tests verify both with and without radii.
- Gradient-bearing meganeura graph nodes deferred to M3c — the dep is already
  pulled and re-exported (`pub use meganeura`) so the M3c entry point is
  unblocked.

#### M3b — COLMAP loader (done)

- `blade_volume_train::colmap` parses `sparse/0/{cameras,images,points3D}.bin`
  in pure Rust (little-endian, mirrors `reconstruction_io_binary.cc`). No
  pycolmap, no native deps.
- Camera intrinsics + per-image extrinsics → `vol::CameraParams`. Inverts
  `cam_from_world` (COLMAP) to `world_from_cam` (renderer convention) and
  converts focal length + width/height to full-angle fov.
- Every camera ID in COLMAP's current model registry is recognised. Distorted
  source images are rectified onto the explicit pinhole training camera;
  equirectangular records are parsed but skipped because they cannot be
  represented by that runtime camera.
- `Reconstruction::to_initial_model` builds a starting `PointCloudModel`
  with positions + DC-only SH from RGB + uniform initial density.
- The per-image `points2D` array and per-point `track` are skipped on read —
  they're the bulk of `images.bin` (~68 MB on bonsai) and aren't used for
  initialisation.
- Tests synthesise a tiny COLMAP dump (3 points, 2 images, 1 PINHOLE camera)
  in a temp dir and round-trip parse + conversion. End-to-end smoke tests
  against a real scene pull via `etc/fetch_test_dataset.sh`.

#### M3c — Differentiable forward + loss

Broken into sub-steps:

- **M3c-1 — Public CPU tracer + screen renderer (done).** The CPU tracer
  promoted from `tests/radfoam_cpu_ref.rs` into `blade_volume::trace`; existing
  tests now call into the library. `blade_volume_train::render::render_cpu`
  wraps it with the camera→ray mapping so we can produce a rendered image
  from a `PointCloudModel` + `CameraParams`. Non-differentiable so far.
- **M3c-2 — Meganeura plumbing proof (done).** `blade_volume_train::fit`
  exercises the full Graph → autodiff → Adam loop on a toy: optimise a
  `[1, 3]` parameter to match a target RGB via MSE. Test converges to within
  0.02 on hosts that have a GPU; skips gracefully otherwise via
  `try_init_gpu()`. Meganeura is pinned at `378f0e5`, the newest verified
  compatible revision: the following `3d24850` e-graph extraction change and
  current upstream head erase the custom RadFoam position gradient in both the
  one-step and finite-difference integration tests. Published 0.2.0 does not
  expose `build`/`SessionConfig`/`Mode`.
  blade-graphics is unified at `ba0fb5a` (the rev meganeura pins) so we
  share one GPU context across our renderer and the meganeura session.
  Notes on meganeura's op set captured in the module doc — biggest gap for
  the real renderer is no `where`/scan/while primitive, so M3c-4 will
  likely need a custom op (recorded-path-then-integrate, à la PowerFoam's
  raytrace mode) rather than expressing the cell-walk as a composition of
  existing ops.
- **M3c-3 — Image-shaped L1 loss + Adam (done, SSIM deferred).**
  `fit::fit_constant_image(target, w, h, ...)` trains a `[1, w*h*3]`
  parameter to match a target image via L1 loss. Test converges an 8×8
  RGB gradient image to within 0.02 max per-pixel error in 300 epochs.
  Forward is still identity (parameter → prediction) — M3c-4 swaps in the
  real renderer. SSIM is parked: meganeura doesn't expose a shortcut for
  the per-window stats and composing it from `conv2d` adds shape-juggling
  overhead we don't need until L1 alone stops being enough.
- **M3c-4 — Real differentiable renderer with frozen geometry (done).**
  Four sub-pieces:
  - `vol::trace::record_path` records the `(cell, dt)` sequence each ray
    covers without integrating. Non-differentiable but the geometry
    decisions live entirely on the CPU side.
  - `blade_volume_train::diff_render::build_volumetric_graph` constructs
    the meganeura subgraph: gather per-step density/SH via `embedding`,
    cumulative-sum-by-pixel via `[P, L] @ [L, L]` matmul with a
    strict-lower-triangular ones matrix, `exp(-x) = recip(sigmoid(x)) - 1`
    surrogate (meganeura lacks raw `exp`), three parallel per-channel
    pipelines summed into one scalar L1 loss.
  - `fit_appearance_to_pixels` (single view) and `fit_appearance_multi_view`
    (camera ring) drive Adam manually via `set_adam` + `step` + `wait`
    rather than the Trainer/DataLoader path (whose `(data, labels)` shape
    can't express our four-input graph). Tests show 10× loss reduction
    on a one-pixel scene and trained-view-B L1 of 0.01 vs 0.37 black
    baseline on a four-camera ring.
  - `pipeline::train_colmap_appearance` orchestrates the COLMAP→foam
    pipeline; `train_colmap` is the CLI. Bonsai (MipNeRF-360) trains in
    ~2 s on an RTX 5070: loss 0.36 → 0.09 over 1600 Adam steps, 8 views
    at 24×24, 2000 subsampled cells. Outputs a binary PLY + an
    interpolated-camera novel-view strip.

    Training checkpoints pair the interchange PLY with a meganeura
    `.safetensors` parameter/Adam sidecar, a versioned `.trainstate` sidecar
    for all deterministic RNG streams plus dynamic topology and densification
    phases, and a legacy `.ply.step` marker. The v3 reader remains compatible
    with v1/v2 fixed-densification sidecars. A resume validates that the
    optimizer, trainer state, and absolute schedule step agree before taking
    another update. With no explicit topology override, the CLI rebuilds from
    the serialized model semantics: stored PowerFoam radii select Čech
    adjacency and remain learned parameters instead of being reinitialized.

  A historical whole-image graph exposed a meganeura matmul shape bug for P×L
  with P≥784 and L≥16. The maintained trainer is pixel-batched and uses the
  current path recorder, so production training no longer relies on that
  obsolete whole-image workaround.

#### M3d — Online viewer attach

- During training, periodically convert `TrainerState → PointCloudModel` and hand it
  to a running `blade-volume-view` instance via a shared `Arc<Mutex<...>>`.
- This mirrors PowerFoam's `--viewer` flag.
- Not yet wired; the CLI dumps a final PLY and the existing viewer can
  load it after the run.

### M4 — Capture stage

Only after M3 produces something worth looking at.

- A short doc page: which phone apps work, what frame rate / exposure settings.
- `etc/colmap.sh` wrapper: video → frames → COLMAP sparse reconstruction.

### M-mesh — Direct mesh → foam conversion (investigated)

`docs/MESH_TO_FOAM.md` covers the "without 2D snapshots" question in
detail. Summary: yes, `blade-volume-convert` already does it (gathered
triangles + grid-interior + barycentric-surface sampling + material
textures → Delaunay → renderable foam). Improvements ordered by cost
documented there. Initial Power-Foam radii helper
(`adjacency::radii_from_nearest_neighbour`) shipped — one-line upgrade
from plain Voronoi to Power Foam for any mesh-derived cloud.

Productionized since: every sampling, appearance, and topology option is
exposed on the `convert` CLI (they were library-only and default-off before,
so all command-line output was the plain baseline); `--resolution` gives a
scale-invariant sampling rate for assets whose units you do not control; the
interior parity test is indexed per parity direction, which cut sampling from
42.6 s to 0.087 s at 804k points; `--topology qhull` selects the Qhull builder;
and interior jitter breaks the lattice degeneracy that was making Delaunay
both slow and ambiguous. `etc/convert_smoke.sh` runs the binary end to end in
CI.

Conversion quality is now measured, not asserted: `MeshReferenceTracer` ray
traces the source triangles through Blade's acceleration structures, and
`convert_quality` scores converted clouds against that reference at matched
poses. Gaussian output climbs 14.04 → 20.77 dB across the resolution ladder.
The metric immediately found two representation defects — opaque exterior fog
that made object-centric RadFoam unviewable from outside, and an alpha stored
where RadFoam expects a density — both fixed. The open item is RadFoam's ~13 dB
ceiling, whose residual is surface-sampling speckle.

On that evidence the **Gaussian backend is the target for the first interactive
prototype**; see the backend-choice section in `docs/MESH_TO_FOAM.md`. This
scopes the offline conversion path only — the trained-from-photographs track
stays on RadFoam.

## Sequencing recommendation

```
M1 ──▶ M2a ──▶ M2b ──▶ M2c ──▶ M2d
         │
         └─▶ M3a ──▶ M3b ──▶ M3c ──▶ M3d ──▶ M4
```

`M3a..M3d` can run in parallel with `M2b..M2d` because M3 only needs the
`PointCloudModel` shape from M2a, not the new renderer.

## Decisions parking lot

- Čech-complex builder: roll our own AABB tree, or pull `rstar`? Decide at M2b.
- meganeura API: tensors-of-vec3 or flat tensor-of-f32? Decide at M3a after reading
  meganeura's actual surface.
- Color contract (resolved): match reference RadFoam and standard 3DGS by
  optimizing and evaluating display-referred sRGB code values. Cloud SH,
  backgrounds, PNG output, and PSNR all use that domain. Presentation clamps
  directly to the sRGB-advertised unorm swapchain; there is no implicit
  transfer function or tone map. Linear-light consumers decode explicitly.

## Test data

Survey of Hugging Face datasets we can use to test the training pipeline, picked
because they ship with the canonical COLMAP `sparse/0/{cameras,images,points3D}.bin`
layout (not just raw images):

| Repo | Scene | Images | Size | Notes |
|---|---|---:|---:|---|
| [`yuangjia/mipnerf-bonsai`](https://hf.co/datasets/yuangjia/mipnerf-bonsai) | bonsai subset | 80 | — | **default smoke test**; COLMAP reconstruction references 292 frames, but only 80 images are present |
| [`pablovela5620/example-colmap-glomap`](https://hf.co/datasets/pablovela5620/example-colmap-glomap) | mixed | 104 | 420 MB | ships both `colmap/` and `glomap/` outputs for cross-checking |
| [`nvs-bench/mipnerf360`](https://hf.co/datasets/nvs-bench/mipnerf360) | all 9 Mip-NeRF-360 scenes | 1.9K | 1.9 GB | full benchmark; MIT licence |
| [`DL3DV/DL3DV-ALL-ColmapCache`](https://hf.co/datasets/DL3DV/DL3DV-ALL-ColmapCache) | 10K scenes | — | >1 TB | gated; use the `DL3DV-10K-Sample` companion if/when needed |

`etc/fetch_test_dataset.sh <scene>` pulls one of these into `etc/data/<scene>/`.
Nothing in this directory is checked in.

Use `etc/fetch_test_dataset.sh bonsai-full` for the pinned complete Bonsai
scene. It is intentionally separate from the small `bonsai` fixture so the
80-image subset and 292-image full-scene benchmark cannot share a result
manifest accidentally.

## Out-of-scope (for now)

- Texel/spherical-Voronoi appearance model from PowerFoam (M2 keeps SH).
- Mobile capture app.
- Multi-GPU / distributed training.
- LOD or streaming for huge scenes.
