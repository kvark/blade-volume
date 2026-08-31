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
  trainer optimises RadFoam, bounded PowerFoam, or anisotropic Gaussian
  particles. Reconstruction emits finite surface-Gaussian particles for the
  relightable path, and both `reconstruct` and the synthetic pipeline can fit a
  static Gaussian light field from the same cloud with `--gaussian-output`.
  See
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
- Candidates are pruned with eight logarithmic radius bands, each backed by an
  immutable `kiddo` k-d tree. A site queries each band to `r_i + r_band_max`,
  then the exact overlap predicate filters the results.
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

#### M2d — Cross-check (geometry complete; full appearance pending)

Internal consistency is covered continuously:

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

An external checkpoint gate now covers real trained geometry as well. The
official implementation at commit `9639225` was trained outside the project
dependency graph on full Mip-NeRF-360 Bonsai. The 30,000-step, 350,000-site
attempt was stopped safely when the live CUDA allocator reached 11,487 of the
5070's 12,227 MiB at step 10,965; its last periodic checkpoint is step 10,000,
162,373 sites, and 674,224 serialized directed edges. The official quarter-
resolution rasterizer scores it over all 37 held-out views at 28.3432 dB PSNR,
0.8681 SSIM, and 0.2138 LPIPS. Rebuilding its adjacency before evaluation
leaves all three metrics identical to four decimals.

Blade's Rust Čech builder processes the activated checkpoint radii in 69 ms and
emits 673,866 directed edges. A fresh official Warp BVH rebuild emits 673,870:
the undirected sets have 336,933 edges in common, Blade has no extra edge, and
the reference has two. Both official-only pairs fail the exact predicate on
the original centers/radii by 0.09--0.22 micrometres. They pass only after the
reference reconstructs radius from f32 AABB min/max, so the 2/336,935 delta is
a reference quantization artifact rather than a Blade topology defect. The
checkpoint's serialized adjacency differs from its own fresh rebuild by 1,443
old-only and 1,266 fresh-only undirected edges because the reference saves
post-optimizer parameters with pre-optimizer adjacency.

The checkpoint isolated the final appearance gap. Repeating each
detail site's mean RGB across its eight directional entries changes nothing
except directional variation and drops the official rasterizer from 28.3432
to 16.4463 dB (-11.8969 dB). Projecting that same ablation into Blade preserves
points, activated density/radii, quaternion-derived normals, spatial sites,
heights, and mean site colours. On identical quantized targets the official
and Blade mean-direction renders score 16.5035 and 16.5067 dB; the renders agree
with each other at 59.18 dB averaged over all 37 views. Thus geometry,
orientation, spatial detail, camera mapping, and compute traversal are already
cross-rendered. The then-unsupported per-detail directional colour function,
not a traversal defect, explained essentially the complete quality gap. That
released function is now represented, trained, serialized, and evaluated by
the shared CPU/WGSL path. A Rust interchange importer now attaches the released
table to the mean-direction PLY. The full Blade render reaches 28.3078 dB and
agrees with the official render at 59.37 dB over all 37 views, closing the
checkpoint pixel gate to output-quantization precision.

The reference repository and project page publish no checkpoint asset. The
native checkpoint, renders, environment, and comparison binaries therefore
remain ignored under `target/reference/powerfoam/`; they are not vendored into
the library or benchmark corpus. The checked-in Rust importer consumes a small
documented directional interchange alongside a regular Blade PLY; producing
that interchange remains an upstream/offline concern. Python does not become a
project dependency.

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
  Candidate rows are bounded to `max(4 * max_steps, 1024)` entries; synchronized
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

#### M2o — Packed differentiable reductions (implemented and gated)

- The weighted path linearization and SH evaluation now express their
  four- and sixteen-component dot products as `meganeura::Graph::sum_inner`
  instead of multiplying by constant columns of ones. Meganeura commit
  `7717181` packs narrow rows into workgroups, folds single-use pointwise and
  gather producers into the reduction, and uses a runtime-one rounding step
  to prevent driver FMA contraction. Adversarial tests remain bit-identical to
  the materialized scalar column order, including a partial final workgroup.
- At 100,569 sites, the profiled differentiable graph falls from 414 passes
  and 16.44 ms to 408 passes and 14.99 ms (8.8%). Two matched 100-step runs
  record 2.487--2.495 seconds of GPU wait versus 2.709--2.718 seconds before
  the change. Loss, site and edge counts, path telemetry, and the trainer-state
  sidecar agree; parameter deltas stay below ordinary repeated-run atomic
  accumulation variation.
- A complete 2,000-step resume reduces GPU wait from 47.745 to 42.939 seconds
  (10.1%), training from 67.071 to 62.082 seconds (7.4%), and whole-command
  phase time from 71.054 to 66.389 seconds (6.6%). It reproduces the exact
  57,500 → 66,125 → 76,044 → 87,451 → 100,569 growth schedule with zero
  truncation over 8,192,000 rays. Fresh-Ply train/test quality is
  14.65/14.35 dB versus 14.64/14.34 dB. Host peak rises from 652 MB to
  1.01 GB because the fused schedule changes buffer lifetimes, but the 6 GB
  cgroup records no swap, limit, or OOM event.

#### M2p — Real-scene PowerFoam radius-learning gate (selected through step 6,000)

- The first fixed-versus-trainable radius experiments used the invalid
  camera-seeded walk and cannot judge bounded supports. The corrected gate
  initializes both arms from the same PowerFoam eight-sample projected radii,
  keeps the same position updates, 4,096 rays across 16 views, fixed growth
  schedule, and site counts, and changes only the radius learning-rate ratio
  from zero to `0.0005`.
- At step 2,000, the 57,500-site fixed arm reaches 13.41/13.22 dB train/all-37
  held out, while learned radii reach 13.70/13.51 dB. At step 4,000, after four
  topology-changing growth rounds, the identical 100,569-site arms reach
  14.32/14.05 and 14.66/14.36 dB respectively. Radius learning therefore adds
  +0.34/+0.31 dB at the stronger boundary instead of merely changing capacity.
- Learned radii retain 1,927,808 directed Čech edges at step 4,000 versus
  1,493,974 when frozen. The maximum observed training path is 85/128 steps;
  both arms record zero truncation, swap, memory-pressure, OOM, or GPU-fault
  events under a 6 GB cgroup. The full protocol and ignored artifacts are
  recorded by `benchmarks/bonsai_powerfoam_radius_learning.toml`.
- Continuing the selected arm through four more growth rounds reaches 175,895
  sites and 8,120,582 directed edges at step 6,000. Fresh-Ply train/all-37
  quality rises to 15.22/14.81 dB; 8,192,000 optimizer rays use at most
  109/128 intervals and none truncate. The sharp edge-count growth is now an
  explicit stability signal to inspect at the 200,000-site boundary.

The fixed-versus-learned causal gate ends at step 4,000. Its selected arm now
also has an adaptive 200,000-site/20,400-step continuation, documented in M2r;
that endpoint is not misrepresented as a single-hyperparameter ablation.

#### M2q — Independent PowerFoam candidate and path budgets (implemented and gated)

- A learned-radius support can intersect a ray without surviving radical-plane
  clipping. Candidate-hit count therefore cannot be inferred from the shorter
  differentiable path depth. The step-6,000 checkpoint demonstrates the
  distinction: full evaluation needs as many as 647 sphere candidates but only
  127 surviving intervals. The previous 512-candidate row rejected this valid
  128-step render.
- Commit `7d5bf5a` gives splat candidate rows a 1,024-entry floor while leaving
  path/Jacobian rows controlled by `max_steps`. Evaluation reports both observed
  budgets and still fails loudly on either overflow. Physical GPU path tests,
  the default and all-feature workspace suites, and strict clippy pass; the two
  workspace suites peak at 5.47 and 5.06 GiB in separate 6 GiB cgroups with
  zero swap, pressure, OOM, or GPU faults.
- Re-rendering all 37 held-out views with 128- and 256-step paths produces
  byte-identical PNGs and the same 15.22/14.81 dB aggregate. On matched
  100-step resumes, the 128-step graph lowers physical allocation from
  1,895.8 to 1,034.7 MB and device-local allocation from 872.9 to 475.3 MB.
  Training falls from 9.733 to 8.141 seconds (16.4%) and GPU wait from 6.103
  to 4.699 seconds (23.0%); host peak falls from 936 to 762 MB. This retains
  the smaller path budget without silently constraining support discovery.

#### M2r — Stable post-cap PowerFoam continuation (completed, quality gate still open)

- Commit `2a0fd73` fixed the bare SH-DC learning-rate selector, so a fresh
  intended-schedule trajectory supersedes the old absolute endpoint numbers.
  At the 200,000-site step-6,500 boundary, starting a `1e-6` overlap loss at
  step 4,000 changes exact train/all-37 quality from 15.8363/15.3752 to
  15.8389/15.3845 dB and reduces the graph 11,145,224→10,083,298 edges. Both
  arms use 128-step rows through step 6,000 and 160 thereafter; neither
  truncates a ray.
- The selected growth-overlap arm reaches 16.1697/15.6274 dB and 9,398,758
  edges at step 8,000. Exact topology cadence 200 is retained after the cap.
  At step 16,000 it reaches 17.2475/16.4321 dB and 8,359,188 edges versus the
  post-cap-only arm's 17.2406/16.4301 dB and 9,196,930 edges.
- Step 20,000 is the corrected practical endpoint at 17.4151/16.5247 dB and
  8,328,306 directed edges. The post-cap-only control reaches
  17.4077/16.5243 dB and 9,150,448 edges, so growth-time overlap is
  quality-neutral/slightly positive while removing 9.0% of the graph. The
  256² all-37 evaluation is exact at 16.5190 dB, using at most 132/160 path
  entries and 641/1,024 candidates. Raw checkpoints and comparisons live under
  `target/audit-runs/powerfoam-dc-fixed/`.
- The completed curve does not clear the visual gate. At 256² the scene is
  recognizable, but large circular supports obscure foreground detail, thin
  geometry is blurred, and black holes/background floaters remain severe. The
  next gate must change cloud support/appearance semantics rather than extend
  this saturated tail.

#### M2s — Radius-banded exact Čech queries (implemented and gated)

- PowerFoam radii span roughly three orders of magnitude, so querying every
  site to `r_i + r_max` makes a single large support inflate nearly every
  k-d-tree search. The exact builder now partitions sites into eight
  logarithmic radius bands and queries each immutable tree only to
  `r_i + r_band_max`. The original overlap predicate, sorted CSR rows, and
  public model remain unchanged; no approximation or dependency is added.
- A brute-force oracle covers varied positive radii, zeros, and clamped
  negatives. Saved 100,569- and 200,000-site clouds produce identical CSR, and
  a 200-step physical-GPU replay ends with the same 9,158,040 directed edges.
  Its only parameter differences are ordinary separate-run sub-ULP GPU drift.
- On the 200K replay, the topology phase falls 1.877→0.481 seconds (74.4%),
  training 15.714→13.953 seconds (11.2%), and the whole command
  21.134→18.510 seconds (12.4%). A forced 100K rebuild falls 0.530→0.085
  seconds (84.0%). A 4/8/12/16-band sweep selects eight; all cgroups stay below
  1.05 GB with zero swap, pressure, OOM, or GPU faults.

#### M2t — Oriented PowerFoam dipoles (implemented, opt-in)

- `PointCloudModel.surface_normals` stores an optional per-site dipole normal
  and is valid only alongside PowerFoam radii. Binary and ASCII RadFoam PLY
  preserve `nx/ny/nz`; partial, non-finite, zero, or unweighted normal records
  are rejected. Unoriented clouds retain their previous format and allocate
  only one 16-byte dummy GPU normal.
- CPU, standalone WGSL, depth, scene, and differentiable path traversal clip
  each bounded power cell to `dot(x - center, normal) <= 0`. The recorder emits
  the exact active-plane center and unit-normal Jacobians. Meganeura normalizes
  the raw parameter in-graph and applies PowerFoam's view-facing
  `relu(dot(normal, ray_direction))²` loss with a 10× exponential decay.
- `train_colmap --oriented-powerfoam` initializes normals with the existing
  Rust 12-neighbour local-PCA estimator, flips each toward its nearest training
  camera, and uses a nearest-camera fallback for underconstrained sites. The
  normal rate and loss are independently configurable. Densification shifts
  both oriented duplicate siblings in the tangent plane and copies the normal;
  checkpoint, Adam remapping, and exact segmented resume include normals.
  Loss weighting, schedules, plane derivatives, and two-sibling resampling
  were checked directly against PowerFoam commit `9639225`. RadFoam-v1 mode
  exposes the reference 0.1→0.01 normal schedule; the Bonsai refit below keeps
  the selected legacy cosine protocol, so its ratio-one normal rate follows
  the global 0.1→0.001 curve instead.
- A normal-only cadence neither changes nor rebuilds Čech adjacency. It reads
  back and uploads only the learned-normal parameter, leaving frozen positions,
  radii, attributes, adjacency, and host index untouched. On the matched
  2,040-step 200K-site gate, specializing both directions cuts state readback
  from 2.740 to 1.959 seconds (28.5%), training from 123.416 to 122.699 seconds,
  command time from 128.653 to 127.628 seconds, and peak host memory from 969
  to 762 MB. The run records 8.36 million rays with zero truncation, swap,
  pressure, OOM, or GPU faults.
- Starting from the selected step-20,400 endpoint and freezing positions and
  radii, the reference normal loss is slightly counterproductive on Bonsai:
  disabling it improves the 2,040-step refit from 17.0258/16.1830 to
  17.0939/16.2340 dB train/all-37 held out. Convergence, rather than dipole
  geometry, explains the remaining short-run gap. Loss-free refits reach
  17.2791/16.3264 at 4,080 steps, 17.4388/16.3905 at 8,160, and
  17.5798/16.4097 at 16,320. The original unoriented cloud is
  17.2334/16.4105, so 8,160 steps is the practical point (within 0.020 dB held
  out) and 16,320 establishes parity within 0.001 dB. Oriented dipoles remain
  explicit rather than default; M2v is the first spatial-appearance gate beyond
  this geometry-parity result.

#### M2u — Signed oriented-surface offsets (implemented, opt-in, selected)

- `PointCloudModel.surface_offsets` stores one optional world-space signed
  plane displacement per oriented site. It requires `surface_normals`, is
  preserved as `surface_offset` by binary and ASCII RadFoam PLY, and shares
  the normal buffer's `.w` lane on the GPU. Missing offsets are exactly zero,
  so every pre-M2u oriented model retains byte-for-byte traversal semantics
  after loading. Validation rejects partial and non-finite tables.
- CPU tracing, standalone rendering, depth, scene rendering, and
  differentiable recording use
  `dot(position - point, normal) <= surface_offset`. The recorder emits the
  exact scalar derivative `1 / dot(ray_direction, normal)` beside the existing
  normal derivative. CPU finite differences, physical-GPU CPU/WGSL parity,
  exact recorder/Jacobian parity, transformed-scene readback, PLY round trips,
  checkpoint/resume, and densification ancestry cover the new field.
- `--surface-offset-lr-ratio` is zero by default and requires an oriented
  model. Enabling it initializes zero offsets and trains them through
  Meganeura. Offset-only cycles read back and upload just normals/offsets;
  positions, radii, CSR, and the host index remain frozen, and no Čech rebuild
  occurs. The stored value is a single per-site world displacement. It is the
  minimal geometric subset of PowerFoam's radius-scaled per-texel height, not
  a claim to implement detail sites or spherical-Voronoi appearance.
- A matched 2,040-step, 200K-site Bonsai sweep uses the same source model,
  255/37 split, 128² images, 4,096 rays over 16 views, loss-free learned
  normals, and frozen positions/radii. The zero-offset control is
  17.0939/16.2340 dB train/all-37 held out. Ratios 0.002, 0.005, 0.01, 0.02,
  and 0.04 reach 16.2560, 16.2740, 16.2808, 16.2856, and 16.2285 dB held out.
  The apparent 0.02 edge is not selected: 6.63% of its planes move beyond
  their support radius, versus 1.93% at 0.01, and 0.04 both regresses and
  raises that fraction to 17.00%. Ratio 0.01 improves 29/37 held-out views and
  adds only 2.3 seconds to the 122.7-second control.
- Duration-adjusted rates keep `ratio × total_steps = 40.8` after the short
  sweep. At 4,080/8,160/16,320 steps, ratios 0.01/0.005/0.0025 reach
  17.2908/16.3896, 17.4614/16.4583, and 17.6039/16.4773 dB. Their matched
  normal-only controls are 17.2791/16.3264, 17.4388/16.3905, and
  17.5798/16.4097, so offsets add +0.0632, +0.0678, and +0.0676 dB held out.
  The selected 8,160-step arm beats the original unoriented endpoint by
  0.0478 dB; the 16,320-step arm adds only another 0.0190 dB for twice the
  time. The practical recipe is therefore 8,160 steps at ratio 0.005, while
  16,320/0.0025 is the measured quality ceiling.
- The selected and ceiling runs process 33.42M and 66.85M rays with maxima
  106/128 and 109/128 path entries, 661/1,024 candidates, and zero truncation.
  Training takes 497.7 and 993.4 seconds, peaks at 627 and 562 MB host memory,
  and records zero swap, pressure, OOM, or GPU faults.
- An independent Room gate converts the selected 200K-site RadFoam endpoint to
  reference PowerFoam radii/Čech adjacency without changing its points,
  density, or SH. The 3.01M-edge source has median radius 0.068 versus 0.073
  on Bonsai and starts at 18.6818/18.1646 dB. After the same 2,040-step
  loss-free oriented refit, normal-only reaches 25.0824/23.0112 dB. Offset
  ratios 0.005 and 0.01 reach 25.3393/23.2402 and 25.3000/23.2607: held-out
  gains of +0.2290 and +0.2495 dB, with 37/39 views improving in either arm.
  Ratio 0.005 is the safer cross-scene selection because only 0.45% of planes
  leave their support, versus 2.92% at 0.01, for just 0.0205 dB less held-out
  quality. The two-scene causal gate therefore passes. Offsets remain explicit
  because they are not the full texel appearance model, not because their
  value is unconfirmed.
- Weighted training no longer clears the inactive dt/Jacobian payload before
  every path dispatch. It initializes that payload once and continues clearing
  all gather indices and masks, so padded rows can reuse only finite masked
  values. At 4,096 rays × 128 entries this reduces per-step buffer fills from
  44 to 8 MiB. The matched 2,040-step ratio-0.01 replay cuts training
  125.036→120.709 seconds (3.46%) and command time 129.763→125.521 seconds
  (3.27%); fresh-Ply quality changes only 17.1232/16.2808→17.1236/16.2806 dB,
  within separate-run GPU drift. Peak memory is flat at 563 MB and the full
  serial training suite, strict workspace clippy, and zero-loss/zero-gradient
  padded oriented-path test pass with no cgroup or GPU faults.
- The selected 200K-site Room model needs at most 111 recorded entries over all
  294 train/held-out views at 128². Re-evaluating it with 160 rather than 256
  entries is pixel- and PSNR-identical. A matched 255-step, zero-learning-rate
  replay reduces training time from 20.510 to 17.510 seconds (14.6%) and GPU
  wait from 12.475 to 10.165 seconds (18.5%), with a 152/160 observed maximum
  and no truncation. This is a measured Room recipe, not a lower global
  default: the 400K-site stress scene still needs more than 1,024 sphere
  candidates, and other scenes must retain their own telemetry gate. Lowering
  the projected-index crossover from 1,024 to 256 rays is also rejected: on
  the same sparse mixed-view replay it increases training time by 36% because
  index construction costs more than the exhaustive gather saves.

#### M2v — Compact spatial surface colour (implemented, opt-in, selected experimentally)

- `PointCloudModel.surface_color_coefficients` stores twelve floats per
  oriented site: four basis coefficients for each RGB channel. The basis is
  `(q.x, q.y, q.z, min(dot(q, q), 1))`, where `q` is the displaced-plane ray
  intersection projected into the tangent plane, divided by support radius,
  and clamped. Keeping `q` in object space removes an arbitrary tangent-frame
  gauge and makes the compact representation rotate with the cloud.
- Binary and ASCII PLY, the CPU oracle, standalone/depth/scene/splat WGSL,
  Meganeura training, densification ancestry, Adam remapping, and exact
  segmented resume share the same basis-major interchange layout. The
  differentiable graph uses a channel-major working layout and stops geometry
  gradients at the basis, isolating the appearance experiment. Zero
  coefficients preserve every M2u pixel exactly.
- A matched 8,160-step Bonsai replay at colour-rate ratio 0.02 reaches
  17.5094/16.4759 dB train/all-37 held out: +0.0480/+0.0176 dB over the
  oriented-offset control, with 24/37 held-out frames improving. Training
  rises 497.689→521.475 seconds (+4.8%), the 200K-site PLY grows from about
  77 to 86 MiB, and complete GPU evaluation changes only
  28.844→28.972 seconds.
- An independent 2,040-step Room replay reaches 25.4107/23.2551 dB
  train/all-39 held out: +0.0714/+0.0149 dB, with 22/39 frames improving.
  Its deliberately conservative 256-entry graph costs 184.053 seconds and
  peaks at 1,241,747,456 bytes although training uses at most 110 entries.
  Replaying the identical arm at the already validated 160-entry Room budget
  reaches 25.4133/23.2645 dB, trains in 148.774 seconds (-19.2%), and peaks at
  1,003,597,824 bytes (-19.2%). GPU-step wait falls
  129.448→96.320 seconds (-25.6%); all 8.36M training rays remain untruncated
  and the cgroup records zero swap, pressure, OOM, or GPU fault.
- Ratio 0.02 is selected only for the experimental spatial arm. The small
  cross-scene mean gain is causal, but several tail views regress and this
  four-term residual is not PowerFoam's full detail-site model.

#### M2w — PowerFoam appearance reference audit (complete; implementation complete)

- The released implementation was audited at official commit `9639225`.
  It uses eight detail sites per power cell and eight spherical axes per
  detail site. A normalized quaternion supplies normal, tangent, and
  bitangent; each two-dimensional detail-site coordinate is scaled by the
  support radius. Each site also has a radius-scaled height and a directional
  RGB function. That is 412 additional floats per point before optimizer
  moments, versus 12 for M2v: about 314 MiB of parameter storage alone at
  200K points. The full representation must therefore land as independently
  gated pieces, not as one unprofiled model-format expansion.
- The source first blends height at the base-plane intersection, shifts the
  plane along its normal, then re-evaluates spatial weights at the displaced
  intersection for colour. Its spatial Warp kernel uses
  `exp(-10 * squared_distance / radius²)`. Equations 3 and 4 of the paper
  instead print an unsquared Euclidean norm. Checkpoint-compatible code must
  follow the released kernel or explicitly version the alternative.
- The standalone Spherical Voronoi definition uses
  `softmax(temperature * dot(axis, direction))`. The PowerFoam repository
  derives each axis and temperature from one raw vector but evaluates
  `exp(-temperature * length(direction - axis))`. These are not identical
  when learned temperatures differ. The first directional experiment must
  state which contract it implements and test its CPU/WGSL/training parity;
  it must not claim official-checkpoint parity until this discrepancy is
  resolved against a published checkpoint.

#### M2x — Compact Spherical Voronoi colour (implemented, opt-in, rejected as a baseline)

- `PointCloudModel.spherical_voronoi` stores eight raw directional axes and
  eight RGB values per point: 48 floats, point-major on disk. Axis length is
  the softmax temperature and evaluation follows the published dot-product
  contract. This is a compact additive residual, not the released
  PowerFoam checkpoint layout with eight detail sites and eight directional
  functions per detail site.
- Binary/ASCII PLY, validation, the CPU oracle, standalone/depth/scene/splat
  WGSL, Meganeura training, densification ancestry, Adam remapping, and exact
  segmented resume share the same representation. Fresh training starts from
  deterministic cube-corner axes and zero RGB values, preserving the prior
  model exactly before the first update. Both axis and colour rates default
  to zero.
- A matched 2,040-step Room gate with learned axes/colours reaches
  25.4019/23.2477 dB train/all-39 held out versus the selected spatial-only
  control's 25.4133/23.2645 dB. Training rises 148.774→181.549 seconds
  (+22.0%), and the 200K-point PLY grows 66→103 MiB. Removing only the learned
  residual from that endpoint gives 25.2077/23.1679 dB, proving that the new
  term carries signal but also displaces the shared SH/spatial solution.
- Freezing the eight axes recovers 25.4116 dB training quality but still gives
  only 23.2508 dB held out, 0.0137 dB below control, with effectively the same
  181.928-second cost. The first scene therefore fails before a costly Bonsai
  replay. Keep the end-to-end implementation as explicit experimental
  infrastructure; do not enable it by default or describe it as a quality
  improvement.

#### M2y — Remove no-op training work (implemented and gated)

- A zero `surface_normal_weight` now omits the view-facing normal-loss branch
  when the production training graph is built. The surface normal/offset
  traversal Jacobian remains present, so oriented geometry continues to
  train. A positive weight retains the previous graph and public low-level
  graph construction retains its established input contract.
- The GPU cloud used only by `PathRecorder` now uploads geometry, oriented
  planes, and adjacency without an unread appearance buffer. On the selected
  200K-point SH-3 + spatial-colour model this avoids 48,800,000 bytes of
  persistent device attributes and the same-size transient staging upload.
  Full render/depth/scene clouds retain their complete attributes.
- The representative 4,096-ray × 160-row graph falls from 460 to 443 GPU
  passes and 30.62→29.54 ms. A matched 2,040-step Room replay reduces training
  148.774→146.390 seconds (-1.6%), GPU wait 96.320→94.796 seconds, and complete
  command time 153.394→150.587 seconds (-1.8%). Quality is preserved at
  25.4129/23.2680 dB, a -0.0004/+0.0035 dB train/held-out change; 19/39 held-out
  frames improve, 15 regress, and five tie at two-decimal precision. The 6 GiB
  scope peaks at 640,094,208 bytes with zero swap, pressure, OOM, kill, or GPU
  fault events.
- A more aggressive 160→128 training-row reduction is rejected even though
  every one of 8.36M training rays fits within 110 rows and all 4.82M
  evaluation rays fit within 109. It cuts training to 136.130 seconds, but the
  changed padded reduction shape reaches only 25.4098/23.2487 dB. Path budgets
  therefore remain telemetry-selected without silently treating exact
  traversal as an identical optimizer trajectory.

#### M2z — Freeze zero-rate geometry in the graph (implemented and gated)

- Fixed-topology training now detaches positions and radii when their effective
  learning rates are zero. Densification deliberately retains the established
  full-gradient graph because its statistics and topology rebuilds remap all
  geometry moments. The RadFoam-v1 schedule and relative parameter groups also
  retain position gradients when their policy supplies a non-zero rate. Public
  low-level graph construction keeps its full-gradient contract.
- The original gate used Meganeura `ad08f97`, which excluded detached
  gradient sentinels from optimizer state. Without that compiler fix, dead
  parameters still appeared trainable and Adam could read beyond the
  one-element sentinel. A structural GPU test requires frozen weighted
  positions/radii to have no gradient while density, normals, offsets, and
  spatial colour remain trainable. The workspace has since moved to the merged
  dependency head; see the current reconstruction plan for its scalar-sentinel
  follow-up.
- Frozen checkpoints contain their exact parameter values but omit unused Adam
  moments. Resuming the same configuration is exact; deliberately enabling
  geometry later starts the previously absent moments at zero. Mixed-view,
  oriented-PowerFoam, and topology-changing RadFoam segmented resumes all match
  their uninterrupted controls.
- The representative 4,096-ray × 160-row graph falls from 443 to 419 GPU passes
  and 29.73→26.25 ms. This removes 800,000 gradient elements and 1.6 million
  Adam-moment elements (9.6 MB total) for the selected 200K-point position and
  radius tables. The matched 2,040-step Room replay reduces training
  146.390→139.111 seconds (-5.0%), GPU wait 94.796→88.849 seconds (-6.3%), and
  complete command time 150.587→143.588 seconds (-4.6%). Fresh-Ply quality is
  preserved at 25.4113/23.2625 dB versus 25.4129/23.2680 dB. All 8.36M training
  and 4.82M evaluation rays remain untruncated; the 6 GiB scopes record no swap,
  pressure, OOM, kill, or GPU fault.
- The remaining profile is dominated by live path-table embeddings, nonlinear
  activation, and appearance projection. Further optimization must preserve
  the 160-row training shape or pass another matched quality gate; exact path
  telemetry alone did not justify the faster 128-row trajectory.

#### M2aa — Pack higher-order SH training parameters (implemented and gated)

- Degree-three training no longer declares 48 one-column SH parameters and
  reconstructs them with a balanced 47-concatenation tree every step. Each RGB
  channel retains its historical `[N, 1]` DC table and stores the remaining 15
  terms in one row-major `[N, 15]` table. The graph still gathers the identical
  channel-major `[N, 48]` table once, so the PLY/model layout and renderer are
  unchanged. DC/rest learning-rate schedules remain separate.
- Upload, readback, densification moment remapping, and native checkpoints all
  understand the packed per-site stride. A compatibility loader migrates the
  old `sh_<channel>_<component>` parameter and Adam-moment tensors without
  resetting optimizer time or state. Physical-GPU tests cover the packed
  forward and gradients, legacy migration, DC/rest rates, and exact segmented
  resumes for mixed-view, oriented-PowerFoam, and topology-changing RadFoam-v1
  training.
- Four matched 200K-site profiles put the old 419-pass graph at 26.52--27.13 ms
  and the packed 293-pass graph at 25.89--26.21 ms (median 26.88→25.99 ms,
  -3.3%). On the matched 2,040-step Room replay, training falls
  139.111→133.847 seconds (-3.8%), GPU wait 88.849→86.933 seconds (-2.2%),
  command submission 45.349→41.649 seconds (-8.2%), and complete command time
  143.588→138.415 seconds (-3.6%). Fresh-Ply quality is unchanged at
  25.4112/23.2624 dB versus 25.4113/23.2625 dB.
- All 8.36M training and 4.82M evaluation rays remain untruncated. Training
  peaks at 725 MiB under the 6 GiB scope, with no swap, pressure, OOM, kill, or
  GPU fault. The serialized parameter count changes, not the number of SH
  values, so this is a dispatch/command-overhead win rather than a memory-size
  claim.

#### M2ab — Use direct reductions for per-ray sums (implemented and gated)

- Opacity, distortion moments, optical-depth quantiles, and final RGB now use
  Meganeura's inner-dimension reduction directly. The previous formulation
  multiplied each `[P, L]` value by a synthetic `[L, 1]` all-ones tensor. This
  is the same sum but removes the constant, its plumbing, and four matrix
  multiplication dispatches.
- Three alternating profiles against the packed-SH control reduce the graph
  from 293 to 289 passes and median GPU time from 25.94 to 25.66 ms (-1.1%).
  The affected loss/gradient oracles and an exact segmented mixed-view resume
  pass on the physical GPU.
- The matched 2,040-step Room replay does not reproduce an end-to-end speedup:
  GPU wait is effectively unchanged at 86.933→86.908 seconds, while noisy CPU
  submission makes training 133.847→137.618 seconds and the complete command
  138.415→142.114 seconds. This milestone is retained as a smaller exact graph
  and isolated dispatch win, not as a full-run performance claim. Quality is
  preserved at 25.4134/23.2697 dB, all training and evaluation rays remain
  untruncated, and the 6 GiB scopes report no swap, pressure, OOM, kill, or GPU
  fault.

#### M2ac — Trim frozen weighted-path differentials (implemented and gated)

- Weighted path recording now selects one of three explicit differential
  payloads: none, complete position/radius plus surface geometry, or oriented
  surface planes only. Fixed-topology oriented training uses the compact
  surface mode when position and radius rates are zero; densification and any
  live position/radius schedule retain the complete mode. The recorder's
  reference tangent and the Meganeura graph therefore contain the same active
  terms without evaluating a large frozen geometry tangent merely to cancel it.
- CPU↔WGSL coverage checks the surface-only tangent and four-component normal
  and offset derivative directly. Separate graph coverage perturbs the live
  plane, verifies its forward linearization and gradients, and requires absent
  position/radius gradients. Mixed-view, oriented-surface, and densifying
  segmented resumes all remain exact.
- Three alternating 200K-site profiles reduce the fixed-geometry graph from
  289 to 280 passes and median GPU time from 25.84 to 20.74 ms (-19.7%). The
  `[4096, 160]` path-output allocation falls from 55.0 to 22.5 MiB because the
  three geometry-Jacobian streams and previous-cell row are absent. Measured
  process peak falls from 730 to 719 MiB.
- On the matched 2,040-step Room replay, training falls 137.618→114.757 seconds
  (-16.6%), GPU wait 86.908→68.631 seconds (-21.0%), command submission
  45.581→40.890 seconds (-10.3%), and complete command time 142.114→119.244
  seconds (-16.1%). Quality remains in the established run-to-run band at
  25.4170/23.2647 dB versus 25.4134/23.2697 dB. All 8.36M training and 4.82M
  evaluation rays remain untruncated, and the 6 GiB scopes report no swap,
  pressure, OOM, kill, or GPU fault.

#### M2ad — Avoid matrix multiplication for XYZ scalar repeats (implemented and gated)

- The oriented spatial-colour basis needs four per-path scalars repeated over
  XYZ. It now performs that exact row-wise copy with two concatenations instead
  of treating each copy as a `[P·L, 1] × [1, 3]` tiled matrix multiplication.
  A distinct-value row-order test and the existing CPU surface-basis oracle
  cover the replacement directly; exact segmented resumes and the complete
  physical-GPU workspace suite pass.
- Three alternating 200K-site profiles replace four of five
  `MatMul[655360x3x1]` passes. The graph grows from 280 to 284 cheaper passes,
  while median GPU time falls 20.81→19.94 ms (-4.2%). Logged loss trajectories
  are unchanged.
- On a back-to-back matched 2,040-step Room pair, training falls
  115.862→111.844 seconds (-3.5%), GPU wait 69.291→65.990 seconds (-4.8%),
  command submission 41.254→40.184 seconds (-2.6%), and complete command time
  120.389→116.054 seconds (-3.6%). Fresh-Ply quality is preserved at
  25.4159/23.2685 dB versus 25.4115/23.2642 dB. All 8.36M training and 4.82M
  evaluation rays remain untruncated. The paired scope peaks at 881 MiB and
  records no swap, pressure, OOM, kill, or GPU fault.

#### M2ae — Pack narrow RMSNorm rows (implemented and gated)

- Meganeura revision `fc20c16` packs independent 2–32-column RMSNorm rows
  into one 256-lane workgroup for both the forward reduction and the input
  gradient. Three-column reconstruction rows use four lanes each, so 64 rows
  share a workgroup instead of leaving 253 lanes idle. Wider rows and the
  weight-gradient path are unchanged, and the power-of-two lane group keeps
  the previous reduction order.
- A 513×3 physical-GPU test compares the forward result and every input
  gradient against a CPU oracle while exercising a partial final workgroup.
  Compiler coverage checks the packed dispatch and the 32/33-column boundary.
  Exact oriented resume, strict lint, the complete Blade workspace suite, and
  the Meganeura all-target suite pass. Four pre-existing Meganeura tests that
  interfere only when sharing one long-lived GPU test process were reproduced
  identically on the untouched dependency and pass individually on both
  revisions.
- Three matched 200K-site profiles keep the graph at 284 passes while reducing
  median GPU time from 19.89 to 17.54 ms (-11.8%). RMSNorm forward falls from
  0.68 to 0.02–0.03 ms and its input gradient from 1.62 to 0.02 ms.
- On the matched 2,040-step Room replay, training falls 111.590→104.632 seconds
  (-6.2%), GPU wait 66.601→61.230 seconds (-8.1%), command submission
  39.388→38.224 seconds (-3.0%), and complete command time 116.142→108.892
  seconds (-6.2%). Fresh-Ply quality is preserved at 25.4103/23.2724 dB versus
  25.4110/23.2709 dB. All 8.36M training and 4.82M evaluation rays remain
  untruncated. The paired scope peaks at 885 MiB and records no swap, pressure,
  OOM, kill, or GPU fault.

#### M2af — Reuse oriented surface gathers (implemented and gated)

- The normalized surface normal and offset for each path row are now gathered
  once and shared by recorded-path linearization, spatial appearance, and the
  optional view-facing loss. This keeps the plane definition explicit and
  prevents independently materializing the same offset payload. Meganeura
  already fused the path-linearization normal lookup into its reduction; the
  production graph therefore realizes one fewer offset embedding rather than
  two fewer standalone gathers.
- Structural coverage requires exactly one embedding of each oriented table.
  The CPU surface-basis oracle, surface tangent and normal-loss tests, exact
  oriented resume, strict lint, and complete physical-GPU workspace suite pass.
- Three alternating 200K-site profiles reduce the graph from 284 to 283 passes,
  the large embedding count from 11 to 10, and median GPU time from 17.60 to
  17.01 ms (-3.4%). The logged three-step loss trajectory is unchanged.
- On a back-to-back matched 2,040-step Room pair, training falls
  104.582→103.209 seconds (-1.3%), GPU wait 61.444→60.366 seconds (-1.8%),
  command submission 37.818→37.707 seconds (-0.3%), and complete command time
  109.103→107.490 seconds (-1.5%). Fresh-Ply quality is preserved at
  25.4125/23.2703 dB versus 25.4096/23.2625 dB. All 8.36M training and 4.82M
  evaluation rays remain untruncated. The paired scope peaks at 897 MiB and
  records no swap, pressure, OOM, kill, or GPU fault.

#### M2ag — Tighten embedding dispatch (implemented and gated)

- Meganeura revision `09d0873` dispatches embedding work over the flattened
  output size, `ceil(seq × hidden / 256)`, for both f32 and f16 tables. The
  previous `seq × ceil(hidden / 256)` expression over-dispatched narrow tables
  by as much as 256×; reconstruction's one- to sixteen-component tables sent
  almost all invocations through the shader's bounds check.
- Compiler coverage exercises a narrow partial workgroup and the one-element
  f16 boundary beyond 515 complete workgroups. Direct f32/f16 GPU checks,
  exact oriented resume, strict lint, the complete Blade workspace suite, and
  the Meganeura all-target suite under its established four order-sensitive
  exclusions pass.
- Three matched 200K-site profiles keep the graph at 283 passes while reducing
  the ten embedding passes from 4.98 to 1.55 ms and median GPU time from 16.67
  to 13.55 ms (-18.7%). The three-step loss trajectories are unchanged.
- On a back-to-back matched 2,040-step Room pair, training falls
  102.918→90.827 seconds (-11.7%), GPU wait 59.915→52.443 seconds (-12.5%),
  command submission 38.443→33.288 seconds (-13.4%), and complete command time
  107.495→95.070 seconds (-11.6%). Fresh-Ply quality is preserved at
  25.4099/23.2624 dB versus 25.4095/23.2647 dB. All training and evaluation
  rays remain untruncated. The paired 6 GiB scope peaks at 897 MiB and records
  no swap, pressure, OOM, kill, or GPU fault.

#### M2ah — Broadcast row-reduction gradients directly (implemented and gated)

- Meganeura revision `41951b1` lowers the transpose of `sum_inner` to an
  explicit `BroadcastInner` copy. The previous autodiff multiplied every
  `[M, 1]` row gradient by an all-ones `[1, N]` matrix, sending simple copies
  through the general matrix-multiplication pipeline. The new internal op has
  its own entry point in the existing row-broadcast shader module and leaves
  global-average-pool gradients unchanged.
- A 513×16 physical-GPU test requires every repeated gradient to be bit exact
  across a partial final workgroup. Compiler coverage requires the 33-group
  direct dispatch, and shader validation, exact oriented resume, strict lint,
  the complete Blade workspace suite, and the Meganeura all-target suite under
  its established four order-sensitive exclusions pass.
- Three alternating 200K-site profiles change the graph from 283 to 284 passes
  but reduce median GPU time from 13.59 to 11.52 ms (-15.2%). The largest
  31.5M-element backward broadcast falls from a 1.08 ms matrix multiply to a
  0.29 ms copy; all three-step loss trajectories are unchanged.
- On a back-to-back matched 2,040-step Room pair, training falls
  90.989→86.245 seconds (-5.2%), GPU wait 52.586→48.788 seconds (-7.2%),
  command submission 33.856→32.680 seconds (-3.5%), and complete command time
  95.520→90.481 seconds (-5.3%). Fresh-Ply quality is preserved at
  25.4134/23.2717 dB versus 25.4072/23.2625 dB. All 8.36M training and 4.82M
  evaluation rays remain untruncated. The paired 6 GiB scope peaks at 887 MiB
  and records no swap, pressure, OOM, kill, or GPU fault.

#### M2ai — Reduce cloud appearance by channel (implemented and gated)

- SH keeps the same per-channel `[N, 1]` DC and `[N, K-1]` rest parameters,
  but the training graph no longer concatenates RGB into `[N, 3K]` or repeats
  every basis row to `[PL, 3K]`. Three channel reductions let Meganeura fold
  each coefficient embedding and forward multiply into `sum_inner`. The
  channel-major spatial and spherical-Voronoi colour tables are split at their
  compact per-site representation instead of after per-path expansion. PLY,
  checkpoint, learning-rate, and viewer layouts are unchanged.
- Physical-GPU forward/backward oracles pass for SH, spatial surface colour,
  and spherical-Voronoi colour. Exact oriented-PowerFoam segmented resume,
  formatting, strict all-target lint, and the complete locked workspace suite
  pass.
- Three alternating 200K-site profiles increase the graph from 284 to 298
  passes but reduce median GPU time from 11.51 to 9.43 ms (-18.1%). The
  655,360-row embedding aggregate falls from 10 passes/1.65 ms to 8
  passes/0.97 ms, while large per-path concatenations fall from 1.32 to
  0.20 ms; all loss and traversal decisions in the three-step gate are
  unchanged.
- On a back-to-back matched 2,040-step Room pair, training falls
  86.002→80.580 seconds (-6.3%), GPU wait 48.733→44.994 seconds (-7.7%),
  command submission 32.610→31.428 seconds (-3.6%), and complete command time
  90.564→84.783 seconds (-6.4%). Fresh-Ply quality is preserved within
  0.011 dB at 25.4121/23.2632 dB versus 25.4136/23.2739 dB. All 8.36M training
  and 4.82M evaluation rays remain untruncated. The paired 6 GiB scope peaks at
  870 MiB and records no swap, pressure, OOM, kill, or GPU fault.

#### M2aj — Fuse row-scaled embedding scatters (implemented and gated)

- Meganeura revision `d01e58f` recognizes the exact table-gradient chain
  `BroadcastInner(row gradient) -> Mul(factors) -> ScatterAddAtomic` after
  scheduling and lowers it to one row-scaled atomic scatter. The existing
  zeroing pass, index bounds check, and compare-exchange accumulation stay in
  place; the expanded row gradient and product buffers disappear.
- A physical-GPU oracle covers 2,049 rows by 16 columns, a permuted 2,053-row
  table, positive and negative products, and a partial final workgroup. Every
  table-gradient bit matches the scalar reference. Shader validation,
  Meganeura's practical all-target suite under its established four
  order-sensitive exclusions, the three cloud-appearance forward/backward
  oracles, exact oriented-PowerFoam segmented resume, strict lint, and the
  complete locked Blade workspace suite pass.
- Three order-balanced 200K-site profiles reduce the graph from 298 to 284
  passes and median GPU time from 9.88 to 8.41 ms (-14.9%). Seven large
  broadcast/multiply pairs become seven direct row-scaled scatters; every
  three-step loss and traversal result is unchanged.
- On a back-to-back matched 2,040-step Room pair, training falls
  79.988→76.629 seconds (-4.2%), GPU wait 45.158→42.308 seconds (-6.3%),
  command submission 30.710→30.061 seconds (-2.1%), and complete command
  time 84.497→80.833 seconds (-4.3%). Fresh-Ply quality is preserved at
  25.4111/23.2654 dB versus 25.4112/23.2594 dB. All 8.36M training and 4.82M
  evaluation rays remain untruncated. The paired 6 GiB scope peaks at 871 MiB
  and records no swap, pressure, OOM, kill, or GPU fault.

#### M2ak — Square PowerFoam camera exclusions (implemented and gated)

- CPU and WGSL splat traversal compare squared camera distance against the
  squared `4r` exclusion radius. This removes a square root from the
  exhaustive ray-site gather while preserving the same bounded-cloud rule in
  direct clipping and projected candidate construction.
- The complete physical path-record suite passes against the independent CPU
  oracle, including mixed-camera batches, projected overflow fallback,
  oriented Jacobians, and truncation boundaries. Exact oriented resume,
  formatting, strict all-target lint, and the complete locked workspace suite
  pass.
- Two order-balanced 510-step Room pairs reduce the recorder/submission phase
  from 7.604 to 7.320 seconds (-3.7%). A longer 2,040-step pair confirms
  30.557 to 29.664 seconds (-2.9%), training 76.859 to 76.542 seconds (-0.4%),
  and whole-command time 81.016 to 80.664 seconds (-0.4%).
- Fresh-Ply quality is preserved at 25.4131/23.2677 dB versus
  25.4126/23.2667 dB. All 8.36M training and 4.82M evaluation rays remain
  untruncated. The long pair peaks at 646 MiB; the serialized full workspace
  gate peaks at 5.3 GiB, with no swap, OOM, kill, or GPU fault.
- The same profiling round rejects separate planar Jacobian buffers (training
  +9.5%) and projected indexing for 256-ray camera slices (+102.7%). Keep the
  packed Jacobian staging path and the 1,024-ray projected-index crossover.

#### M2al — Order PowerFoam intervals once (implemented and gated)

- `BLADE_VOLUME_PROFILE_GPU=1` now reports path-recorder timestamps alongside
  Meganeura's graph timings. On the 200K-site, 4,096-ray, 16-view Room shape,
  transfer/clear costs 1.15 ms, exhaustive gathering 3.05 ms, and the original
  cached interval recorder 7.37 ms per step.
- The recorder still clips each candidate once, but compacts valid intervals
  and heap-sorts them by the established `(effective depth, cell index)` key.
  It then emits the ordered row directly instead of scanning the entire
  candidate row again for every segment. Cached faces, neighbours, exact
  tie-breaking, and truncation semantics are unchanged.
- The timestamped recorder pass falls from 7.37 to 6.43 ms (-12.8%), and total
  recorder GPU time from 11.56 to 10.46 ms (-9.5%). Two order-balanced
  510-step pairs reduce combined path submission/GPU wait from 18.409 to
  17.772 seconds (-3.5%) and training from 21.220 to 20.681 seconds (-2.5%).
- A 2,040-step pair confirms combined path/GPU work at 71.693 to 69.304
  seconds (-3.3%), training at 75.477 to 73.556 seconds (-2.5%), and complete
  command time at 79.649 to 77.762 seconds (-2.4%). Fresh-Ply quality is
  preserved at 25.4139/23.2701 dB versus 25.4122/23.2639 dB, with zero
  truncation across all 8.36M training and 4.82M evaluation rays.
- The complete physical path-record suite, exact oriented resume, formatting,
  strict all-target lint, and locked workspace suite pass. The long pair peaks
  at 648 MiB and the serialized workspace gate at 4.5 GiB, with no swap, OOM,
  kill, or GPU fault.

#### M2am — Initialize weighted path rows in gather (implemented and gated)

- Each PowerFoam gather workgroup already assigns 64 lanes to one sampled ray.
  Those lanes now initialize that ray's cell, next-cell, mask, and optional
  previous-cell row before gathering supports. Weighted training and
  contribution collection no longer encode four redundant transfer fills per
  dispatch; payload/Jacobian initialization is unchanged.
- On the 200K-site, 4,096-ray, 16-view Room shape, the recurring transfer pass
  falls from 0.32 to 0.01 ms. The added stores raise gather from 3.31 to
  3.54 ms, leaving recorder GPU time at 10.24 to 10.18 ms. The one-time
  payload-initialization step regresses by 0.29 ms and is amortized over the
  session.
- Two order-balanced 510-step pairs reduce combined path submission/GPU wait
  from 17.914 to 17.456 seconds (-2.6%) and training from 20.518 to 20.165
  seconds (-1.7%). Two 2,040-step pairs confirm 69.460 to 68.648 seconds
  (-1.2%) and 74.004 to 73.166 seconds (-1.1%), respectively.
- Average fresh-Ply quality is preserved at 25.4137/23.2738 dB versus
  25.4105/23.2644 dB. All 33.42M long-gate training rays and 19.27M evaluation
  rays complete without truncation.
- The physical path oracle deliberately poisons weighted index/mask rows before
  dispatch and all 12 cases pass. Exact oriented resume, formatting, strict
  all-target lint, and the locked workspace suite also pass. The long gate
  peaks at 778 MiB and the serialized workspace gate at 5.28 GiB under the
  6 GiB cgroup limit, with no swap, OOM, kill, or GPU fault.

#### M2an — Stage external surface Jacobians once (implemented and gated)

- Meganeura revision `17e794b` adds an explicit differentiable `materialize`
  operation: an f32 identity copy into a distinct intermediate buffer. It is a
  memory-placement barrier, so graph optimization and pointwise fusion cannot
  replace it with an alias. Identity autodiff, compiler/runtime mappings,
  WGSL/SPIR-V validation, and a physical-GPU value/gradient test cover the new
  primitive.
- Oriented PowerFoam training materializes the recorder's packed normal/offset
  Jacobian immediately before its two channel splits. The public graph field
  remains the external input, and structural coverage requires exactly one
  materialization feeding both splits. This replaces two cold reads from the
  external allocation with one read into device-local memory.
- On the 200K-site, 4,096-ray, 16-view Room shape, the external splits average
  1.11 ms. The final graph instead spends 0.67 ms on materialization and about
  0.04 ms on both device-local splits; total graph time falls from 8.73 to
  8.44 ms (-3.3%) despite growing from 287 to 288 passes.
- Two order-balanced 510-step pairs reduce combined path submission/GPU wait
  from 17.820 to 17.427 seconds (-2.2%) and training from 20.541 to 20.202
  seconds (-1.7%). Two 2,040-step pairs confirm 69.682 to 68.351 seconds
  (-1.9%) and 74.480 to 72.767 seconds (-2.3%), respectively.
- Average fresh-Ply quality is preserved at 25.4154/23.2714 dB versus
  25.4116/23.2706 dB. All 33.42M long-gate training rays and 19.27M evaluation
  rays complete without truncation. Exact oriented resume, all 12 physical
  path-oracle cases, formatting, strict workspace lint, and the complete
  locked workspace suite pass. The long gate peaks at 783 MiB and the
  serialized workspace gate at 5.49 GiB under the 6 GiB cgroup limit, with no
  swap, OOM, kill, or GPU fault.

#### M2ao — Eight-site spatial surface detail (implemented and two-scene-gated)

- Oriented PowerFoam models can now carry eight radius-normalized tangent
  sites, eight signed heights, and eight RGB residuals per point. The compact
  56-float increment keeps the cloud-only representation and avoids the full
  reference model's nested directional table.
- The Meganeura graph matches the renderer's two-stage evaluation: height is
  blended at the base-plane query, then RGB is blended again at the displaced
  plane. The recorder exports the pre-surface support query and discrete plane
  branch as one `vec2` stream; the graph materializes it once before splitting.
- `train_colmap` exposes independent site, height, and color rate ratios. A
  fresh table uses a deterministic eight-site tangent ring with zero height
  and color, so enabling the representation without training preserves the
  preceding model exactly.
- Detail geometry is refreshed with surface normals in one GPU submission.
  PLY checkpoints, optimizer moments, pruning/densification ancestry, and
  segmented resume all retain the three tables exactly.
- The differentiable normalizer uses a `1e-12` floor instead of the renderer's
  forward-only `1e-20`: squaring the latter in division backward underflows on
  padded rows and can inject `0/0` before masking. Active-row forward parity is
  checked against the CPU oracle, and a physical end-to-end test covers finite
  updates plus frozen topology.
- Position and radius optimization are supported while detail is present. The
  recorder carries the active support-entry derivative for both the preceding
  and current site's `(position, radius)`, and the graph evaluates that fixed-
  topology tangent against live geometry. Frozen geometry retains the compact
  query-only payload; full detail adds two `vec4` streams only when either
  spatial table is trainable.
- Physical gates cover CPU parity and nonzero gradients for every table,
  external query-buffer binding, repeated geometry refresh, table/Adam remap
  through densification, and exact interrupted/resumed training.
- The matched 200K-site Room gate uses 255 training and 39 every-eighth
  held-out views at 128², 4,096 rays, 16 views per batch, and 2,040 updates.
  Two control replicas average 25.4147/23.2643 dB train/held out. Site,
  height, and colour ratios `0.1/0.05/0.05` average 25.6484/23.4155 dB,
  gains of +0.2337/+0.1513 dB. Detail improves 37/39 held-out frames; the two
  regressions are -0.115 and -0.070 dB.
- The matched 8,160-step Bonsai gate uses the same 200K-site, 128²,
  4,096-ray, 16-view configuration across 255 training and 37 held-out views.
  Two detail replicas average 17.6167/16.5467 dB versus
  17.5108/16.4767 dB for two controls, gains of +0.1059/+0.0701 dB. Detail
  improves 33/37 held-out frames, ties two, and regresses two by only 0.010
  and 0.005 dB. The representation and `0.1/0.05/0.05` rates therefore pass
  the cross-scene quality gate.
- The selected long models remain geometrically bounded: tangent-site radius
  p99 is about 0.507 support radii on Room and 0.849 on Bonsai. Absolute
  normalized-height p99 is about 0.267 and 0.354 respectively; only 25 of
  1.6M Bonsai sites exceed one support radius in height. The Bonsai PLY grows
  from 89,964,416 to 134,766,940 bytes (+49.8%), and training rises from
  344.325 to 463.063 seconds (+34.5%). Full GPU Room evaluation is effectively
  neutral (22.43 versus 23.03 seconds across order-balanced runs). Detail
  remains opt-in because its quality win is real but its storage and training
  costs are substantial.
- Projecting all detail sites once before path gathering saves 6.0% training
  wall time, but is rejected: its four-replica mean is within aggregate noise
  while 17/39 averaged held-out frames regress and the worst loses 0.09 dB.
  The production graph keeps the path-local projection until a quality-neutral
  formulation is demonstrated.
- Meganeura revision `7967ca2` instead batches final and geometry-refresh
  parameter reads through cached download memory. This does not alter a graph
  operation or optimizer value. Against four pre-change detail replicas,
  state readback falls from 15.051 to 0.864 seconds (-94.3%) and two new
  training replicas average 126.293 versus 141.018 seconds (-10.4%). Their
  25.6383/23.4041 and 25.6443/23.4022 dB results stay inside the prior replica
  spread. The 6 GiB scope peaks at 1,658,667,008 bytes with zero swap,
  pressure, OOM, kill, or GPU fault.
- Meganeura revision `226041c` batches Adam first/second moments in the same
  way, and Blade uses it for densification snapshots. On a real 200K→200.1K
  Bonsai detail boundary, order-balanced state readback falls from 11.585 to
  0.185 seconds (-98.4%, 62.8×) and whole-command time from 16.890 to 5.471
  seconds (-67.6%, 3.09×). Isolated runs peak at 1,863,389,184 and
  1,861,894,144 bytes respectively, with no swap, pressure, OOM, kill, or GPU
  fault. Exact multi-parameter moment tests and detail densification/remapping
  coverage guard the value and ordering contract.
- Meganeura revision `760bb29` exposes the existing row broadcast as a
  differentiable graph operation. Scalar-to-XYZ expansion and eight-site
  normalization now use one broadcast instead of concat/split trees. On the
  200K-site Bonsai profile this cuts the detail graph from 537 to 491 passes
  and steady GPU time from about 21.0 to 19.5 ms; the control remains about
  7.1 ms. Four matched Room replicas reduce training from 126.674 to 122.261
  seconds (-3.5%) while held-out PSNR changes 23.3988→23.4061 dB; 22/39
  averaged views improve, one ties, and the worst regression is 0.0575 dB.
  Two Bonsai replicas reduce training 117.223→114.856 seconds (-2.0%) while
  held-out PSNR changes 16.3152→16.3140 dB; per-view extrema are symmetric at
  ±0.04 dB. Physical forward/gradient tests, cross-scene gates, and exact
  resume therefore select the simpler broadcast path.
- Meganeura revision `048c8be` adds a differentiable inner tiling operation,
  replacing the remaining three-level concat tree used to repeat each detail
  site's RGB vector. The Bonsai detail profile falls from 491 to 461 passes
  and from 19.63 to 17.98 ms of steady GPU time; the 284-pass, approximately
  7.1 ms control graph is unchanged. Two full Room replicas reduce training
  from 122.261 to 115.639 seconds (-5.4%) and GPU wait by 6.9%, while changing
  train/held-out PSNR from 25.6427/23.4061 to 25.6517/23.4138 dB. Averaged
  held-out views split 20 improvements, five ties, and 14 regressions, with a
  +0.0075 dB mean. Two Bonsai replicas reduce training from 114.856 to 111.409
  seconds (-3.0%) and GPU wait by 3.3%; train/held-out PSNR changes only
  17.1983/16.3140→17.1987/16.3137 dB. Exact 513×3×8 forward/gradient
  coverage, exact reconstruction resume, strict lint, shader validation, and
  the practical full Meganeura suite pass. The full gates peak at 1.60 GB on
  Room and 1.52 GB on Bonsai under a 6 GiB scope, with zero swap, pressure,
  OOM, kill, or GPU fault.

#### M2ap — Parallel dense PowerFoam interval clipping (implemented and gated)

- The weighted path recorder now has a deterministic dense-graph entry point.
  One 64-lane workgroup clips one ray's candidate cells in parallel, then lane
  zero compacts the valid intervals, performs the existing depth/index heap
  sort, and emits the unchanged path, surface-query, and Jacobian streams.
  There are no atomics or device-scope storage barriers, and the sparse serial
  entry point retains its original inline shader body.
- The recorder selects the parallel path at an average of 32 adjacency entries
  per site and logs the decision during training. Audited real clouds separate
  cleanly around this boundary: sparse graphs have at most 19.2 entries/site,
  while the dense Bonsai checkpoint has 41.7. A universal parallel path is
  rejected because Room regresses from the 115.639-second baseline to 117.378
  seconds; a two-pass clip/record variant is also rejected at 117.158 seconds.
  Refactoring the sparse emit tail into a helper caused a repeatable 1.1%
  regression, so the selected shader deliberately duplicates that small tail.
  An otherwise faster cross-lane storage prototype is also rejected because
  its device-scope barrier requires an optional Vulkan memory-model feature.
- On the 200K-site Bonsai profile, the record pass falls from about 26.4 to
  4.4 ms and the complete recorder from about 30 to 8 ms. The final portable
  2,040-step replay reduces training from 111.409 to 67.035 seconds (-39.8%)
  and GPU wait from 101.729 to 57.483 seconds (-43.5%). Train/held-out PSNR
  remains neutral at 17.2001/16.3171 dB versus 17.1987/16.3137 dB.
- The final Room replay selects the serial path at 15.1 entries/site and takes
  116.229 seconds, within 0.51% of the preceding mean; GPU wait is slightly
  lower at 69.486 versus 69.759 seconds. Train/held-out PSNR is
  25.6390/23.4010 dB versus 25.6517/23.4138 dB. All 14 physical path tests,
  including a forced dense batched oriented-detail oracle, pass. The Room and
  Bonsai scopes both peak at 1.4 GB under 6 GiB with no swap, pressure, OOM,
  kill, or GPU fault. The complete locked workspace suite also passes without
  Vulkan validation errors; its scope reaches the 6 GiB cap with 36 reclaim
  events but records no swap, OOM, or kill.

#### M2aq — Direct pairwise detail distances (implemented and two-scene-gated)

- Meganeura revision `508c0ab` adds a differentiable row-wise pairwise
  squared-distance operation. A compact `[M, D]` query is compared with its
  consecutive `[M * N, D]` site rows directly; dedicated backward kernels
  reduce the `N` contributions into the query gradient and write site
  gradients without atomics.
- Both eight-site surface-detail partitions use the primitive instead of
  materializing a `[rows, 8, 3]` query tile, subtraction, square, and inner
  reduction. At the 4,096-ray, 128-step shape, each removed full-size
  intermediate is 12,582,912 floats (48 MiB). The graph falls from 461 to 454
  passes and steady measured time from about 17.9 to 16.5–16.8 ms; the new two
  forward and four backward dispatches together take about 1.0 ms.
- Two 2,040-step Room replicas take 113.149 and 110.928 seconds, averaging
  112.039 seconds versus the current build's 116.229-second replay (-3.6%).
  Mean GPU wait falls 69.486→67.314 seconds. Their train/held-out mean is
  25.6435/23.4102 dB versus the preceding two-run 25.6517/23.4138 dB. Against
  the exact current-build replay, averaged held-out views improve 20, tie
  three, and regress 16; the mean is +0.0088 dB.
- Two Bonsai replicas take 65.114 and 64.019 seconds, averaging 64.567 seconds
  versus 67.035 seconds (-3.7%); mean GPU wait falls 57.483→54.521 seconds
  (-5.2%). Train/held-out quality is 17.1979/16.3147 dB versus
  17.1987/16.3137 dB. Averaged held-out views split 16 improvements, five
  ties, and 16 regressions, with a +0.0011 dB mean.
- A physical 513×3×8 Meganeura oracle covers forward values and both
  gradients. All 177 library and 11 reduction/runtime tests, shader-module and
  SPIR-V validation, strict lint, Blade's detail/densification tests, and exact
  oriented-detail resume pass. The complete locked Blade workspace suite also
  passes and peaks at 2.4 GB. Room/Bonsai gates peak at 1.5/1.6 GB under the
  6 GiB scope, all with zero swap, pressure, OOM, kill, or GPU fault.

#### M2ar — Direct tangent-site rejection (implemented and two-scene-gated)

- Meganeura revision `f31ef08` adds differentiable pairwise vector rejection:
  each of `N` consecutive vectors is projected off one shared unit direction.
  Dedicated kernels evaluate the forward value and gradients for both the
  vectors and the shared direction without atomics or device-scope barriers.
- The surface-detail graph uses this primitive to project eight learned site
  offsets into each oriented cell's tangent plane. It replaces the explicit
  normal tile, dot-product reduction, scalar broadcast, projection, negation,
  and addition chains. At the 4,096-ray, 128-step shape, the graph falls
  454→446 passes and warmed GPU time falls 16.53–16.86→15.05–15.12 ms
  (about -9.8%). The new forward and two backward dispatches total about
  0.54 ms, while the removed normal tile alone held 12,582,912 floats
  (48 MiB).
- Two 2,040-step Room replicas take 105.929 and 103.883 seconds, averaging
  104.906 seconds versus the pairwise-distance build's 112.039 seconds
  (-6.4%). Mean GPU wait falls 67.314→62.521 seconds (-7.1%). Train/held-out
  quality is neutral at 25.6442/23.4108 dB versus 25.6435/23.4102 dB;
  averaged held-out views improve 24, tie one, and regress 14, with a
  +0.0010 dB mean.
- Two Bonsai replicas take 61.809 and 61.149 seconds, averaging 61.479 seconds
  versus 64.567 seconds (-4.8%); mean GPU wait falls 54.521→51.540 seconds
  (-5.5%). Train/held-out quality is 17.1986/16.3120 dB versus
  17.1979/16.3147 dB. Averaged held-out views split 14 improvements, six
  ties, and 17 regressions, with a -0.0027 dB mean.
- A physical 257×3×8 Meganeura oracle covers the forward value and both
  gradients. All 177 library tests, shader-module/SPIR-V/runtime-binding
  validation, and strict lint pass. The unconstrained all-target Meganeura
  run exhausts the 6 GiB scope while parallel GPU tests run and crashes in an
  unrelated cooperative-matmul binary; serialized cooperative-matmul tests
  all pass. The remaining four attention/model numerical failures reproduce
  exactly at parent revision `508c0ab`, isolating them from this operation.
  Blade's focused detail/resume tests, strict lint, and complete locked
  workspace suite pass with the pinned dependency. The full suite, profile,
  and two-scene scopes peak at 2.5/2.1/1.7 GB, with zero swap, pressure, OOM,
  kill, or GPU fault.

#### M2as — Direct detail-weight exponential (implemented and two-scene-gated)

- Meganeura revision `22bb539` exposes the pointwise IR's existing `exp`
  primitive through the public graph, autodiff, fallback WGSL, compiler, and
  runtime. The gradient is `grad * exp(x)`, and the existing pointwise pass
  fuses single-use chains on both sides of the derivative.
- Surface-detail weights now evaluate `exp(-x)` directly instead of the
  algebraic identity `recip(sigmoid(x)) - 1`. The two are mathematically
  equal, but the direct form avoids the identity's expanded autodiff graph.
  The 4,096-ray, 128-step detail graph falls 446→438 passes; four large
  multiplies, two adds, two reciprocals, and both sigmoids disappear. Warmed
  GPU time falls from 15.05–15.12 to 14.56–14.68 ms (about -3.2%), while the
  two direct exponentials together take about 0.12 ms.
- Two 2,040-step Room replicas take 102.885 and 102.927 seconds, averaging
  102.906 seconds versus 104.906 seconds (-1.9%). Mean GPU wait falls
  62.521→61.006 seconds (-2.4%). Train/held-out quality changes
  25.6442/23.4108→25.6480/23.4202 dB; averaged held-out views improve 19,
  tie one, and regress 19, with a +0.0099 dB mean.
- Two Bonsai replicas take 59.300 and 58.979 seconds, averaging 59.140 seconds
  versus 61.479 seconds (-3.8%); mean GPU wait falls 51.540→49.840 seconds
  (-3.3%). Train/held-out quality changes 17.1986/16.3120→17.2006/16.3139
  dB. Averaged held-out views improve 17, tie six, and regress 14, with a
  +0.0014 dB mean.
- A physical 513-element Meganeura oracle covers forward values and gradients,
  and handwritten/scheduled shader parity is explicit. All 177 library tests,
  all 17 pointwise schedule tests, shader-module/SPIR-V/runtime-binding
  validation, and strict lint pass. The profile and both long gates peak below
  1.9 GB under 6 GiB. Blade's focused detail/resume tests, strict lint, and
  complete locked workspace suite also pass; the latter peaks at 2.4 GB. All
  scopes record zero swap, pressure, OOM, kill, or GPU fault.

#### M2at — Fused detail-weight normalization (implemented and two-scene-gated)

- Meganeura revision `85e919f` adds differentiable `normalize_inner_sum` for
  two-dimensional f32 tensors. It computes each row denominator as
  `relu(sum - floor) + floor`, preserving the existing behavior when all
  non-negative detail weights are negligible instead of silently replacing
  it with ordinary softmax semantics. One forward and one closed-form backward
  dispatch replace each row-sum/floor/broadcast/reciprocal graph.
- Floating-point order is part of the contract. Prototype `36b9670` factored
  the shared reciprocal-square term out of the gradient reduction. It passed
  ordinary tolerances and was faster, but four Room replicas averaged
  -0.0187 dB held out with 13–14 improvements versus 24–25 regressions. That
  prototype is rejected. Revision `85e919f` applies the factor per site before
  reduction and mirrors Meganeura's runtime multiply-by-one contraction
  barrier. A physical 257×8 oracle covering zero, below-floor, boundary, and
  ordinary rows is bit-exact against the expanded graph for every forward
  value and parameter gradient.
- The 4,096-ray, 128-step detail graph falls 438→418 passes. Short profiles
  show a slow GPU clock ramp after process startup, but their settled endpoints
  are 13.66 and 13.97 ms versus the prior 14.56–14.68 ms. The two fused
  forward and two backward normalization dispatches total about 0.25 ms once
  warm.
- Two 2,040-step Room replicas take 100.380 and 100.487 seconds, averaging
  100.434 seconds versus 102.906 seconds (-2.4%). Mean GPU wait falls
  61.006→59.251 seconds (-2.9%). Train/held-out quality changes
  25.6480/23.4202→25.6428/23.4208 dB; averaged held-out views improve 16,
  tie two, and regress 21, with a +0.0006 dB mean.
- Two Bonsai replicas take 57.634 and 57.824 seconds, averaging 57.729 seconds
  versus 59.140 seconds (-2.4%); mean GPU wait falls 49.840→48.580 seconds
  (-2.5%). Train/held-out quality changes 17.2006/16.3139→17.2008/16.3172
  dB. Averaged held-out views improve 22, tie two, and regress 13, with a
  +0.0039 dB mean.
- Meganeura's 177 library tests, all 13 serial reduction GPU tests,
  shader-module/SPIR-V/runtime-binding validation, formatting, and strict lint
  pass. Blade's focused surface-detail and exact oriented-resume tests, strict
  workspace lint, and complete locked all-target suite also pass. The Room and
  Bonsai gates peak at 1.45 and 1.15 GB; the full suite peaks at 2.54 GB. All
  selected scopes record zero swap, pressure, OOM, kill, truncation, or GPU
  fault. A concurrent Meganeura integration run is excluded: it reproduced
  the known NVIDIA multi-context semaphore crash at only 0.87 GB, while the
  same complete integration file passes serially at 0.89 GB.

#### M2au — Reuse gathered PowerFoam sphere roots (implemented and two-scene-gated)

- The PowerFoam candidate gather already computes each accepted support
  sphere's discriminant square root. It now retains that one f32 in the
  existing candidate-depth scratch. Serial and workgroup recorders reconstruct
  the original `-b ± root` bounds with the same arithmetic order, then
  overwrite the scratch with the clipped entry depth as before. This removes
  the record pass's duplicate `dot(oc, oc)`, discriminant, and square root
  without adding a buffer, binding, allocation, or traversal approximation.
- Across two order-balanced 32-step profiles (the last 20 timestamps from each
  arm), the 4,096-ray Bonsai parallel record pass falls 4.504→4.208 ms
  (-6.6%). Gather remains flat at 3.469→3.452 ms, and total recorder time falls
  7.972→7.660 ms (-3.9%).
- Two 2,040-step Room replicas reduce mean training 100.359→99.821 seconds
  (-0.54%), path submission 36.137→35.915 seconds (-0.61%), and GPU wait
  59.551→59.251 seconds (-0.50%). Train/held-out quality changes
  25.6432/23.4102→25.6461/23.4184 dB; averaged held-out views improve 21 and
  regress 18, with a +0.0074 dB mean.
- Two Bonsai replicas reduce mean training 57.902→57.712 seconds (-0.33%),
  path submission 4.070→4.052 seconds (-0.44%), and GPU wait 48.792→48.593
  seconds (-0.41%). Train/held-out quality changes
  17.2004/16.3162→17.1999/16.3148 dB; averaged held-out views improve 16, tie
  one, and regress 20, with a -0.0014 dB mean.
- All 14 physical path-record tests match the independent CPU oracle across
  serial and workgroup clipping, multiple cameras, disconnected Čech
  components, projected overflow, oriented/detail Jacobians, and truncation
  boundaries. The long gates cover 16.71 million rays per scene with zero
  truncation or candidate overflow. Room/Bonsai scopes peak at 1.68/1.38 GB
  under 6 GiB. Strict all-target lint and the complete locked, serialized
  workspace suite pass; the latter peaks at 5.75 GB. No selected scope records
  swap, pressure, OOM, kill, or GPU fault.
- Three nearby prototypes remain rejected: replacing regular per-pixel gathers
  with row tiling is neutral in the graph profile and slows Bonsai 0.22%; a
  workgroup-shared camera ray slows exhaustive gathering 2.8%; caching both
  sphere faces rather than only the root is neutral at 7.809→7.813 ms total
  recorder time. These results favor the narrow scalar reuse.

#### M2av — Reuse gathered surface centers (implemented and two-scene-gated)

- Oriented spatial detail already gathers the selected site's center for every
  path row. The spatial surface-colour basis now consumes that same node
  instead of issuing an identical position embedding. This keeps frozen
  appearance geometry behind the existing `train_positions` stop-gradient and
  combines the shared backward path when positions are trainable.
- The 4,096-ray, 128-step detail graph falls 418→416 passes and 15→14 large
  embeddings. Across two order-balanced 32-step profiles, averaging the last
  20 timestamps per run, graph time falls 13.952→13.766 ms (-1.33%); recorder
  time remains flat within noise at 7.848→7.817 ms.
- Two 2,040-step Room replicas reduce mean training 100.296→99.663 seconds
  (-0.63%) and GPU wait 58.912→58.359 seconds (-0.94%); path submission is
  flat at 36.765→36.738 seconds. Train/held-out quality changes
  25.6418/23.4075→25.6453/23.4109 dB.
- Two Bonsai replicas are end-to-end neutral: mean training changes
  57.682→57.630 seconds (-0.09%), GPU wait 48.606→48.492 seconds (-0.23%),
  and path submission 4.050→4.104 seconds (+1.33%). Train/held-out quality
  changes 17.1989/16.3125→17.1983/16.3176 dB. Both scene gates record zero
  path truncation, candidate overflow, memory pressure, or GPU fault and peak
  below 1.73 GB under 6 GiB.
- A nearby structure-of-arrays prototype is rejected. Emitting query-near and
  plane-branch values into separate recorder buffers removes one graph
  dispatch and improves that graph 0.58%, but two scalar storage writes replace
  each `vec2` write and slow the recorder 8.4%; combined measured GPU time
  regresses 2.65%. Feeding those external scalar streams directly is slower
  still because fused consumers repeatedly read host-visible memory. Retain the
  interleaved payload and its measured device-local staging.
- Focused surface/detail tests, formatting, strict all-target lint, and the
  complete locked serialized workspace suite pass. The full suite peaks at
  6.10 GB (5.68 GiB) under the 6 GiB cgroup. No selected scope records swap,
  pressure, OOM, kill, or GPU fault.

#### M2aw — Reuse gathered surface radii (implemented and two-scene-gated)

- Spatial detail and the spatial surface-colour basis now consume one shared
  gathered, positively floored surface-radius node. This preserves the exact
  floor arithmetic while removing two graph passes and one large embedding:
  the detail graph falls 416→414 passes and 14→13 embeddings.
- Balanced last-20 timestamps improve 13.756→13.599 ms (-1.14%). Two 2,040-step
  Room replicas reduce mean training 99.839→98.754 seconds (-1.09%); two
  Bonsai replicas reduce it 57.375→56.633 seconds (-1.29%). Mean held-out
  quality changes by -0.0009/+0.0010 dB, respectively.
- A train-mode physical test gathers and floors the radius through the
  production path, then matches its gradient to an independent CPU central
  finite difference. Strict lint and the serialized workspace suite pass with
  zero memory or GPU fault events.

#### M2ax — Widen exhaustive PowerFoam gathering (implemented and two-scene-gated)

- One workgroup still owns one sampled ray and scans the same exact support
  set, but 256 lanes now divide the scan instead of 64. Candidate storage,
  intersection arithmetic, overflow handling, and the deterministic
  `(effective depth, cell index)` ordering are unchanged. 256 is the maximum
  workgroup size guaranteed across WebGPU adapters.
- In order-balanced profiles, 64/128/256 lanes take 3.455/3.055/2.843 ms for
  the 200K-site, 4,096-ray exhaustive gather. The selected size is 17.7%
  faster than 64 lanes while the following record and 414-pass training graph
  remain neutral. Removing interval sorting entirely provided only a noisy
  0.15 ms lower bound and immediately changed loss, so a parallel sort was not
  pursued.
- Two 2,040-step Room replicas reduce mean training 99.699→96.906 seconds
  (-2.80%), path submission 36.851→35.485 seconds (-3.71%), and GPU wait
  58.207→56.907 seconds (-2.23%). Train/held-out PSNR changes
  25.6486/23.4196→25.6457/23.4122 dB.
- Two Bonsai replicas reduce mean training 57.288→56.034 seconds (-2.19%) and
  GPU wait 48.106→46.876 seconds (-2.56%). Train/held-out PSNR changes
  17.1994/16.3180→17.1982/16.3182 dB. Across all ABBA arms, 66.85 million
  training rays and 38.40 million evaluation rays remain untruncated;
  candidate maxima are 231/1,024 on Room and 661/1,024 on Bonsai. The scene
  scopes peak at 1.69/1.35 GB under 6 GiB with zero swap, pressure, OOM, kill,
  or GPU fault.
  All 14 physical path tests, strict all-target lint, and the complete
  serialized workspace suite pass; the latter peaks at 5.69 GB (5.30 GiB)
  under the same limit with no event.

#### M2ay — Differentiate spatial-detail support entry (implemented and two-scene-gated)

- Eight-site detail now composes with topology-safe position and radius
  training. The recorder exports the active derivative of the pre-surface
  support entry with respect to the preceding and current site's
  `(x, y, z, radius)`. Support-sphere and radical-face branches share the CPU
  and WGSL formulas. Meganeura reconstructs the stable local query by dotting
  those recorded derivatives with live ray-relative geometry; Euler
  homogeneity makes the snapshot value exact without another reference stream.
- The two extra `vec4` outputs exist only for full geometry plus detail: they
  add 32 bytes per path slot (20 MiB at 4,096 rays × 160 entries). Frozen
  detail continues to bind two 16-byte dummies. A two-replica profile measures
  414→471 graph passes and 26.90→44.47 ms combined steady recorder/graph GPU
  time (+65.3%). The full Room gate measures 95.53→150.11 seconds (+57.1%);
  Bonsai measures 55.33→92.82 seconds (+67.8%). This is an explicit opt-in
  quality/cost tradeoff, not a new default schedule.
- On the 200K-site, 2,040-step Room gate, two frozen replicas average
  25.6449/23.4125 dB train/held out. Joint geometry at position/radius ratios
  `0.01/0.01` averages 26.4852/24.2963 dB, gains of +0.8403/+0.8838 dB, and
  improves 38/39 averaged held-out views. Mean Čech edges grow 3.011M→3.250M
  (+7.9%). A reference-radius-rate `0.01/0.0005` probe reaches
  26.5906/24.1554 dB while shrinking edges to 2.908M, so the radius rate stays
  a scene-level policy rather than being silently selected globally.
- On Bonsai at the same horizon, frozen detail reaches 17.1996/16.3188 dB.
  Ratios `0.01/0.0005` reach 17.2352/16.4668 dB with 1.7% fewer edges; ratios
  `0.01/0.01` reach 17.3405/16.7241 dB with 5.3% more edges. The high rate
  improves 22/37 held-out views, regresses 14, and ties one, so it demonstrates
  cross-scene mean benefit but does not qualify as a universal default.
- CPU central differences cover sphere and radical-face query branches. All 14
  physical recorder/oracle cases compare the new GPU streams and reconstruct
  their query values. A train-mode Meganeura finite difference covers the
  complete spatial-detail query, and end-to-end tests cover finite joint
  updates, densification/Adam remap, and bit-exact interrupted resume. All
  scene scopes record zero truncation, swap, pressure, OOM, kill, or GPU fault.
  Formatting, strict workspace lint, and the complete serialized workspace
  suite pass; the last peaks at 5.37 GB (5.00 GiB) under the 6 GiB limit with
  no event.

#### M2az — Reuse differentiable path-role geometry (implemented and gated)

- Interval length and spatial-detail entry previously gathered the same
  preceding/current point geometry independently. The graph now creates one
  ray-relative `(x, y, z, radius)` node for each of the three unique path
  roles and shares the preceding/current nodes with both linearizations. This
  removes four large row-scaled embedding-gradient scatters without changing
  the fixed-topology derivative contract.
- A two-replica 32-step profile reduces mean graph time 27.43→25.85 ms
  (-5.7%) and combined recorder/graph time 45.71→43.70 ms (-4.4%). An
  order-balanced 510-step Room gate reduces mean training 39.998→38.577
  seconds (-3.6%) and GPU wait 24.439→23.125 seconds (-5.4%), while held-out
  PSNR changes 22.4460→22.4504 dB.
- At the full 2,040-step horizon, two candidate replicas average
  26.4991/24.3063 dB train/held out versus the preceding implementation's
  26.4852/24.2963 dB. Mean training time falls 150.107→147.417 seconds
  (-1.8%), GPU wait falls 93.397→91.012 seconds (-2.6%), and final Čech edges
  change by -0.04%. All training and evaluation rays remain untruncated; the
  cgroup peaks at 1.13 GB with zero swap, pressure, OOM, kill, or GPU fault.
- The complete support-query graph finite difference, joint update,
  densification/Adam-remap, and exact-resume tests exercise the shared nodes.
  The arithmetic regrouping is quality-gated rather than claimed bit-identical
  to the previous graph. Formatting, strict all-target lint, and the complete
  serialized workspace suite pass; the suite peaks at 3.16 GB (2.94 GiB) under
  6 GiB with no event.

#### M2ba — Fold shared producers into reductions (implemented and two-scene-gated)

- Meganeura `2acdfeb` extends reduction-prologue fusion to shared pointwise
  and embedding producers when every consumer is a compatible reduction. It
  clones the cheap expression into each reduction, leaves the producer alive
  until its last use is folded, and rejects mixed or protected consumers.
  Blade's preceding/current ray-relative geometry can therefore be loaded
  directly by both interval and detail-query dot reductions without
  materializing the shared four-wide rows.
- The 200K-site profile falls 478→470 graph passes, 16→13 large embeddings,
  and four→two materialized four-wide adds. Two-replica mean graph time falls
  25.82→24.46 ms (-5.3%) and combined recorder/graph time falls 43.47→42.23
  ms (-2.8%). An order-balanced 510-step Room gate reduces training
  38.710→38.040 seconds (-1.7%) and GPU wait 23.324→22.551 seconds (-3.3%);
  held-out PSNR changes by +0.0005 dB and mean topology by 37 edges out of
  3.07 million.
- On two full 2,040-step Room replicas, training falls 147.417→144.434
  seconds (-2.0%) and GPU wait falls 91.012→88.466 seconds (-2.8%). Mean
  train/held-out PSNR changes 26.4991/24.3063→26.4940/24.2937 dB; averaged
  held-out views split 18 improved / 21 regressed and topology changes by
  -0.08%. On an order-balanced Bonsai gate, training falls 92.085→89.907
  seconds (-2.4%) and GPU wait falls 67.794→65.596 seconds (-3.2%). PSNR
  changes 17.3804/16.7620→17.3770/16.7515 dB, views split 19/18, and topology
  changes by +0.04%. The small metric shifts are inside replica spread and
  balanced per view, so the repeatable performance win is selected.
- White-box dispatch coverage proves both consumers fold and the standalone
  embedding disappears. Fused/unfused device parity and an analytical shared
  table-gradient test cover values and backward accumulation. Meganeura
  formatting and strict all-target lint pass; every unaffected all-target
  test passes. Four unrelated attention/model tests fail identically at both
  `2acdfeb` and the pinned `85e919f` baseline on this stack and are recorded as
  pre-existing rather than attributed to this change. All scene rays remain
  untruncated and all scoped memory/GPU counters remain zero. Blade formatting,
  strict workspace lint, and the complete serialized workspace suite pass
  against the remote pin; the suite peaks at 3.55 GB (3.31 GiB) under 6 GiB.

#### M2bb — Share invariant spatial-detail query geometry (implemented and two-scene-gated)

- The base-plane and displaced-plane spatial-detail queries differ only by
  their signed plane offset. They now share the center/ray and ray/normal dot
  products, regularized denominator, negated center/query, and broadcast
  radius. The height-dependent displaced query remains downstream of the base
  query, so this changes graph construction without changing the model or
  training schedule.
- A two-replica 32-step 200K-site profile falls 470→462 graph passes. Mean
  graph time falls 24.583→24.405 ms (-0.7%) and combined recorder/graph time
  falls 42.829→42.495 ms (-0.8%). An order-balanced 510-step Room gate is
  wall-time neutral (37.872→37.886 seconds) while GPU wait falls
  22.762→22.655 seconds (-0.5%), held-out PSNR changes by +0.0042 dB, and
  topology changes by -0.07%.
- At the full 2,040-step horizon, two Room replicas reduce mean training
  144.434→143.652 seconds (-0.5%) and GPU wait 88.466→87.697 seconds (-0.9%).
  Mean train/held-out PSNR changes 26.4940/24.2937→26.4988/24.3179 dB, the
  averaged held-out views split 23 improved / 16 regressed, and topology
  changes by +0.19%. Two Bonsai replicas reduce training 89.907→89.603
  seconds (-0.3%) and GPU wait 65.596→65.321 seconds (-0.4%). Mean
  train/held-out PSNR changes 17.3770/16.7515→17.3955/16.7673 dB, views split
  17 improved / 19 regressed / 1 tied, and topology changes by +0.01%. The
  repeatable two-scene performance gain is selected; arithmetic regrouping is
  quality-gated rather than claimed bit-identical.
- The complete query finite difference, joint position/radius update, and
  bit-exact interrupted-resume tests pass. Formatting, strict all-target lint,
  and the complete serialized workspace suite pass. Its warm-cache rerun with
  two build jobs peaks at 228,335,616 bytes (217.8 MiB) under the 6 GiB cgroup,
  with zero swap, pressure, OOM, kill, or GPU fault.

#### M2bc — Select radius rates at the long horizon (scene-level policy gated)

- A 2,040-step Bonsai screen holds every other setting fixed and sweeps radius
  ratios `0.0005`, `0.001`, `0.0025`, `0.005`, and `0.01`. Held-out PSNR rises
  monotonically through 16.4758, 16.4910, 16.5657, 16.6304, and 16.7634 dB.
  The `0.005` arm limits the final graph to 8.433M edges and degree p99/max
  203/1,968 versus 8.762M and 217/3,666 at `0.01`, but captures only 54% of
  the high rate's gain over `0.0005`. It therefore advances as a topology-aware
  intermediate, not as a winner from the short horizon.
- The 8,160-step gate resolves the choice differently by scene:

  | Scene | radius ratio | train/test PSNR (dB) | directed edges | training/GPU wait (s) |
  | --- | ---: | ---: | ---: | ---: |
  | Room | `0.0005` | 29.1776 / 25.5704 | 2,872,340 | 556.718 / 344.110 |
  | Room | **`0.005`** | **28.9541 / 25.6867** | 3,188,820 | 554.830 / 338.675 |
  | Room | `0.01` | 28.4694 / 25.6062 | 3,498,108 | 562.564 / 343.182 |
  | Bonsai | `0.0005` | 18.2475 / 17.2852 | 8,107,072 | 357.470 / 270.196 |
  | Bonsai | `0.005` | 18.9665 / 17.9072 | 8,517,388 | 357.026 / 268.167 |
  | Bonsai | **`0.01`** | **19.6609 / 18.6322** | 8,930,116 | 359.710 / 267.021 |

- Room `0.005` improves 29/39 held-out views over `0.0005` and 25/39 over
  `0.01`. Its degree p99/max is 55/238, between 32/73 and 80/457. Bonsai
  `0.01` improves 31/37 views over `0.005`; giving up 0.7250 dB to reduce edges
  by 4.6% is not justified under the quality-first gate. Bonsai's degree
  p99/max progresses 200/1,926 → 212/4,211 → 231/5,648 across the three rates,
  so the selected high rate still carries an explicit hub-tail warning.
- The policy is consequently scene-level: use `0.005` for the measured Room
  trajectory and `0.01` for the measured Bonsai trajectory. Keep the public
  default at zero (frozen geometry), and bracket at least these two rates on a
  held-out split before training a new scene. No arm truncates a training or
  evaluation ray; candidate maxima remain at or below 679/1,024. The combined long
  intermediate scope peaks at 1,141,944,320 bytes under 6 GiB with zero swap,
  pressure, OOM, kill, throttling, or GPU fault.
- Comparison renders agree with the metric selection, but also show that rate
  tuning is not the remaining representation fix. Room is recognizable and
  the intermediate rate removes some errors visible in the high-rate arm. Bonsai's
  high-rate gain is real, yet both rates retain large pale support blobs,
  holes, and background floaters. The next quality experiment should constrain
  unsupported support/opacity or improve spatial appearance responsibility,
  not merely extend this fixed-cap schedule or choose one universal rate.

#### M2bd — Fuse stable positive activations (implemented and two-scene-gated)

- Meganeura `3c2d81e` adds a native stable softplus pointwise operation, so
  Blade no longer expands each density/radius activation into constants,
  multiply, ReLU, absolute value, sigmoid, log, negate, add, and final
  multiply nodes. Its forward expression remains
  `(relu(βx) - log(sigmoid(abs(βx)))) / β`, including finite behavior at
  large magnitudes. The first backward implementation used the conventional
  sigmoid derivative. It was mathematically correct and reduced the profile
  from 462 to 448 passes, but it was rejected: at 2,040 steps Room and Bonsai
  held-out means changed by -0.0236 and -0.0187 dB, with only 14/39 and 18/37
  views improving, while timing was mixed. Equivalent real arithmetic is not
  enough when floating-point regrouping changes the optimizer trajectory.
- Meganeura `f82d0b6` instead lowers a fused backward helper that preserves the
  expanded graph's derivative operation and accumulation order. A 513-element
  physical process test compares the two complete graphs: parameter-gradient
  values are bit-exact and forward values agree within strict f32 tolerance.
  The selected Blade graph falls 462→446 passes. Across an order-balanced,
  two-replica 32-step profile, warmed graph time falls 24.331→24.199 ms
  (-0.54%), combined recorder/graph time falls 41.679→41.509 ms (-0.41%),
  and short-run wall time falls 4.323→4.270 seconds (-1.23%).
- At 2,040 steps, two Room replicas reduce mean training
  143.706→143.184 seconds (-0.36%) and GPU wait 88.718→87.946 seconds
  (-0.87%). Mean train/held-out PSNR changes
  26.6700/24.3137→26.6601/24.3312 dB, with 21/39 held-out views improving,
  two tying, and 16 regressing. Two Bonsai replicas are wall-time neutral at
  89.672→89.690 seconds; path submission falls 4.028→3.991 seconds while
  GPU wait changes 65.432→65.568 seconds. Mean train/held-out PSNR improves
  17.3647/16.7411→17.3843/16.7657 dB, with a 19/1/17 view split. The
  repeatable Room gain, neutral Bonsai cost, and positive held-out means select
  the exact-order implementation.
- All long-gate rays remain untruncated and the combined scope peaks at
  1,907,179,520 bytes with zero swap, pressure, limit, OOM, kill, throttling,
  or GPU fault. Meganeura's 20 serial pointwise tests, formatting, strict lint,
  and all 177 library tests pass. Its broad all-target run reproduces the same
  four unrelated attention/model failures already recorded for `2acdfeb`; it
  reaches the 6 GiB cap and incurs reclaim events without swap or OOM. Blade
  formatting, strict all-target workspace lint, and the complete locked,
  serialized workspace suite pass against the remote pin. The Blade suite
  peaks at 3,492,728,832 bytes with no memory or GPU event.

#### M2be — Deduplicate exact training dispatches (profile-positive, rejected)

- An instrumented plan dump showed that the profile's dominant
  `Relu[655360]` label mostly represents packed three-, four-, eight-, and
  sixteen-term reductions using a legacy routing sentinel, not plain ReLU
  kernels. It also exposed exact duplicate reciprocal, narrow-dot, broadcast,
  and negate dispatches produced by independent autodiff branches. A
  conservative Meganeura prototype reused only byte-identical scheduled
  operations, protected every public/state/gradient buffer, and excluded
  materialization and atomic work. A 513-element physical test matched
  analytic forward values and gradients.
- The prototype removes 15 passes, taking the representative graph 446→431.
  Two order-balanced replicas reduce warmed graph time
  24.331→24.183 ms (-0.61%) and short-run wall time
  4.297→4.258 seconds (-0.91%). The 510-step screen is mixed: Room held-out
  PSNR changes 22.4034→22.4070 dB while wall time falls 0.64%; Bonsai changes
  15.9777→15.9725 dB and is wall-time neutral. Per-view splits are 22/1/16
  and 16/2/19 improved/tied/regressed.
- The full 2,040-step gate rejects the change. Two Room replicas reduce mean
  wall time 144.037→143.407 seconds (-0.44%) but change train/held-out PSNR
  26.6722/24.3080→26.6586/24.3037 dB, with a 17/1/21 view split. Two Bonsai
  replicas reduce wall time only 89.930→89.811 seconds (-0.13%) while changing
  17.3947/16.7943→17.3764/16.7611 dB; only 13/37 views improve, one ties,
  and 23 regress. The directional -0.0332 dB Bonsai loss fails the
  quality-first gate, so neither the CSE nor its diagnostic instrumentation is
  retained. The long scope peaks at 1,914,609,664 bytes with zero swap,
  pressure, limit, OOM, kill, throttling, or GPU fault.

#### M2bf — Attribute held-out support failures (implemented; prune shortcut rejected)

- `eval_psnr --diagnose-worst N` now separates the global held-out tail into
  false-positive radiance and missing-radiance pixels. The GPU renders remain
  the metric source; only the selected `2N` rays use the CPU oracle. Each row
  reports terminal opacity, peak weight/depth/cell, support radius, density,
  Čech degree, and the pruning collector's maximum weight and sampled-view
  count. `TraceResult::peak_point` retains the cell already identified while
  computing mode depth, and `render::camera_ray` keeps diagnostic and renderer
  camera math on one implementation.
- The read-only contribution scan exposed a latent fixed-cap/detail-model
  combination: the pruning collector allocated full PowerFoam Jacobians but
  omitted the spatial-detail query contract, so it panicked before dispatch.
  It now uses the existing compact recorded-path layout, requests detail
  queries only when the cloud carries them, and retains per-cell sampled-view
  counts without changing the established maximum-contribution pruning
  decision. A physical one-cell detail path covers the corrected combination.
- On the selected 8,160-step models, Room's 16 strongest false positives are
  one cell with radius/density/degree `0.476/0.580/7`; it is nevertheless
  supported in 32 sampled training views. Bonsai's false-positive peak cells
  span 20--249 views, and its strongest missing-radiance cluster comes from
  two degree-227/415 cells supported in 223/235 views. A minimum-view rule is
  therefore not the missing visual mechanism: heavily observed cells can
  still assign the wrong geometry/appearance across a broad support.
- The scan also found 118,280/200,000 Room cells and 141,999/200,000 Bonsai
  cells with no contribution in the collector's deterministic 2× phase. An
  aggressive weighted-cloud prune removed only cells lacking direct or
  one-ring contribution and reduced Room to 188,970 sites and 3.132M edges.
  It is rejected: held-out PSNR changes 25.6867→25.6700 dB, one view loses
  0.60 dB, and the worst tail loses another 0.05 dB. One downsample phase is
  diagnostic evidence, not sufficient proof that a support can be deleted.
- The successful two-scene diagnostic scopes peak at 506 and 475 MB with
  zero swap, pressure, limit, OOM, kill, throttling, or GPU fault. The failed
  pre-fix detail scan stopped inside its 346 MB scope with the same zero event
  counters; leaked-buffer messages were panic cleanup, not memory pressure.
  The next quality experiment must resolve position/direction responsibility
  within an observed support rather than disguise the problem as opacity or
  low view count.

#### M2bg — Bound directional detail capacity (implemented experimentally; rejected)

- Two zero-preserving directional additions were implemented across the model,
  ASCII/binary PLY, CPU and all WGSL renderers, Meganeura training,
  densification/Adam remapping, and exact resume, then removed after their
  quality gates. The first attached three RGB direction coefficients to each
  of the eight spatial sites (72 floats per cell). The second compressed that
  to one RGB coefficient per site, weighted by signed camera incidence
  `-dot(normal, normalize(ray_direction))` (24 floats per cell). Neither mixed
  polygonal geometry into the runtime representation.
- The 72-float form produced only noisy two-replica 2,040-step mean changes:
  Room train/held-out moved by +0.0009/+0.0121 dB and Bonsai by
  +0.0330/+0.0177 dB. Per-view splits were 21/0/18 and 20/1/16, with worst
  regressions of 0.145 and 0.260 dB. Training rose about 12%/15.5%, and each
  200K-cell PLY grew by about 55 MiB. That cost and tail risk reject the form.
- The 24-float incidence form screened best at learning-rate ratio `0.01`, but
  failed the fresh 2,040-step control. Room changed
  26.6757/24.3516→26.6570/24.2995 dB, with 10/2/27 held-out views
  improving/tying/regressing; training rose 142.018→149.216 seconds (+5.1%).
  Bonsai changed 17.3911/16.7571→17.3703/16.7309 dB, with a 13/5/19 split;
  training rose 89.343→95.180 seconds (+6.5%). Each PLY grew by about
  18.3 MiB. Ratio `0.005` was also negative on Room and held-out-neutral but
  train-negative on Bonsai.
- The long incidence scope peaked at 1,859,657,728 bytes under 6 GiB with zero
  swap, pressure, limit, OOM, kill, throttling, or GPU fault. The rejected
  source remains absent; local evidence lives under
  `target/audit-runs/directional-detail/`. Directional capacity is therefore
  not the next bottleneck. A smaller spatial-density/responsibility probe over
  the existing eight detail weights is the next quality experiment.

#### M2bh — Redistribute density over spatial detail sites (implemented and two-scene-gated; opt-in)

- `SurfaceDetail::density_logits` optionally stores eight scalar logits per
  cell. Their softmax is multiplied by eight, so the site scales have arithmetic
  mean one. The displaced-plane detail weights blend those scales for each ray;
  equal logits return exactly one and preserve every legacy model. Binary and
  ASCII PLY use `blade_surface_detail_density_logit_0..7`. The table adds eight
  floats per cell (6.10 MiB at 200K sites) and remains absent by default.
- CPU traversal, standalone WGSL, scene WGSL, compute splats, Meganeura
  training, Adam checkpoint/remapping, densification, and exact resume share
  the same contract. `--surface-detail-density-lr-ratio` creates an identity
  table only for a fresh oriented-detail run. Colour rendering evaluates the
  detail RGB residual and density scale in one spatial kernel; depth-only
  rendering keeps the scalar path. On identical trained models this fusion
  preserves every reported PSNR/per-view value and reduces three-run all-view
  evaluator means by 0.50% on Room and 0.27% on Bonsai.
- A 510-step screen over ratios `0.005`, `0.01`, `0.02`, and `0.05` improves
  held-out Room by +0.0230, +0.0090, +0.0241, and +0.0366 dB and Bonsai by
  +0.0108, +0.0094, +0.0140, and +0.0269 dB. Boundary probes at `0.1` and
  `0.2` raise Room further but weaken Bonsai and introduce -0.52/-0.61 dB
  Room tail regressions, so the cross-scene gate selects `0.05`.
- Two order-balanced 2,040-step replicas confirm a mean improvement. Room
  train/held-out changes 26.6764/24.3367→26.7580/24.3529 dB
  (+0.0816/+0.0162); averaging each view across replicas gives a 21/1/17
  improved/tied/regressed split and a -0.52 dB worst delta. Bonsai changes
  17.3616/16.7627→17.4389/16.8070 dB (+0.0774/+0.0443), with a 25/0/12
  averaged split and -0.21 dB worst delta. The positive two-scene means retain
  the representation as an opt-in tool; the Room tail prevents making it the
  default recipe.
- Mean training time rises 142.876→158.567 seconds on Room (+11.0%) and
  89.591→100.682 seconds on Bonsai (+12.4%). After the shared runtime kernel,
  three-run all-view evaluator means are effectively neutral versus matched
  controls: 19.353 versus 19.300 seconds on Room and 20.777 versus 20.863 on
  Bonsai (the latter also has a slightly different learned topology). Both long
  scopes peak at or below 1,501,564,928 bytes under 6 GiB with zero swap,
  pressure, limit, OOM, kill, throttling, or GPU fault. Artifacts and telemetry
  remain ignored under `target/audit-runs/spatial-detail-density/`. Formatting,
  strict all-target lint, and the complete locked serialized workspace suite
  pass; the suite peaks at 5,500,235,776 bytes with zero cgroup events.

#### M2bi — Pack narrow softmax rows (performance-positive, rejected)

- The eight-site density table makes softmax the next measurable appearance
  cost. Meganeura's existing reduction template can assign eight lanes to each
  row and pack 32 rows in one 256-lane workgroup. The prototype added no graph
  op, shader entry/group, WGSL file, binding, or pipeline-map variant. Compiler
  coverage checked the 32-row dispatch and the 33-column fallback; a 513x8
  physical-GPU oracle matched an independent CPU forward sum and every table
  gradient.
- A graph-only simplification was rejected first. Replacing the zero-preserving
  residual form with the algebraic `8 * dot(weights, softmax(logits))` removes
  surrounding nodes but leaves GPU-step time unchanged and changes the 32-step
  loss from 0.0117 to 0.0132. Equal logits must return exactly one even when
  normalized spatial weights do not sum to one bit-for-bit.
- Two 2,040-step Room replicas reduced mean training 158.204 to 144.815
  seconds (-8.5%) and GPU wait 99.216 to 88.565 seconds (-10.7%). Mean held-out
  PSNR changed 24.3480 to 24.3395 dB, with 15/4/20 averaged views
  improving/tying/regressing.
- Two Bonsai replicas reduced mean training 98.136 to 89.213 seconds (-9.1%)
  and GPU wait 73.203 to 64.275 seconds (-12.2%), but mean held-out PSNR changed
  16.8319 to 16.8083 dB and the worst averaged view fell 0.045 dB. The 16/0/21
  view split makes the quality loss directional rather than replica noise.
- The implementation is removed. It is mathematically correct, but replacing
  the 256-lane identity-padded reduction with an eight-lane tree changes f32
  accumulation and the optimizer trajectory. Future reduction work on this
  path must preserve the original tree or pass a new quality gate. The long
  serialized scope peaked at 2,075,435,008 bytes under 6 GiB with zero swap,
  pressure, OOM, kill, throttling, truncation, or GPU fault.

#### M2bj — Recycle fixed cloud capacity (implemented experimentally; rejected)

- A minimal prototype kept the fixed densification schedule active at the
  200,000-site budget. At one boundary it reused the existing exhaustive
  maximum-contribution collector to prune small unobserved cells, then filled
  the exact headroom by splitting sites selected from the existing PowerFoam
  photometric-error EMA. The model stayed cloud-only and exactly fixed-size;
  the existing topology rebuild, field inheritance, and Adam remap handled the
  change. The prototype added no model field, graph op, shader entry/group,
  WGSL, binding, or pipeline variant.
- With the established `0.01` contribution threshold, the step-510 boundary
  recycles 5,326 Room sites and 87 Bonsai sites. After another 510 matched
  updates, Room train/held-out PSNR changes
  25.3359/23.4194→25.2459/23.3695 dB and Bonsai changes
  16.8923/16.2997→16.8728/16.2993 dB. One exhaustive scan and rebuild raise
  training time 74.184→77.824 seconds on Room and 46.055→49.979 seconds on
  Bonsai.
- Lowering the threshold tenfold does not isolate safe redundancy. It still
  recycles 4,798 Room sites and changes train/held-out PSNR to
  25.2567/23.3536 dB; the 38-site Bonsai change reaches only
  16.8705/16.2889 dB. Thus even cells with very small sampled maximum weight
  can be necessary traversal or held-view support, while copying a high-error
  site's complete appearance/support state does not create missing spatial
  responsibility.
- All four candidate scans cover 1,044,480 rays over all 255 training views
  with zero truncation. The two serialized scopes peak at 1,415,856,128 and
  1,218,797,568 bytes under 6 GiB with no swap, pressure, OOM, kill,
  throttling, or GPU fault. The implementation and CLI are removed. Future
  capacity work needs a held-view-safe spatial responsibility signal rather
  than another policy over the current contribution and point-error scores.

#### M2bk — Drop inactive densification score graph (done)

- Weighted training used to build the temporary per-point photometric-error
  probe whenever a densification configuration existed, including fixed clouds
  already at their target and RadFoam-v1 schedules past their final growth
  boundary. The probe has exactly zero forward value and only exists to feed a
  later split decision, so keeping its parameter and gradient path after the
  schedule closes cannot affect any model gradient.
- Graph construction now follows the same schedule predicate as the growth
  loop. The probe remains live while a future round can consume it and is
  omitted immediately after the final topology rebuild. Adam remapping matches
  rebuilt parameters by name, allowing this temporary state to disappear while
  preserving every surviving parameter, exposure, and bias-correction step.
  A physical-GPU regression grows an oriented PowerFoam cloud to its target and
  completes the following fixed-topology optimizer step.
- Two order-balanced 1,020-step replicas at a fixed 200,000-site capacity cut
  mean Room training time from 74.455 to 71.854 seconds (-3.5%) and GPU wait
  from 45.217 to 43.083 seconds (-4.7%). Bonsai falls from 45.946 to 44.351
  seconds (-3.5%) and from 32.873 to 31.081 GPU-wait seconds (-5.5%). Mean
  train/held-out PSNR changes only 25.3357/23.4034→25.3378/23.4067 dB on Room
  and 16.8860/16.2978→16.9054/16.3098 dB on Bonsai. The implementation adds no
  model field, graph op, shader entry/group, WGSL, binding, or pipeline variant.

#### M2bl — Released per-detail directional appearance (checkpoint gate passed; training rejected)

- `SurfaceDetail::directional` stores eight raw axes and eight RGB residuals
  for each of the existing eight spatial sites: the released 384-float nested
  table, optional and absent from legacy models. Evaluation matches the
  released chord-distance kernel rather than the older compact dot-product
  experiment. PLY checkpoints preserve every indexed axis and colour.
- CPU traversal and all WGSL consumers evaluate the table through the existing
  shared surface-detail function. It adds one packed-attribute capability bit
  but no shader entry, shader group, pipeline, bind group, binding, or
  Meganeura operation. The standalone RadFoam, compute-splat, and scene paths
  agree with the CPU oracle on the RTX 5070, including models that combine
  density logits and both older compact colour residuals.
- The training graph composes existing reductions, normalization, exponent,
  and softmax operations. Independent axis/colour rates default to zero;
  enabling them initializes deterministic cube axes and zero colours. Zero
  colours add no directional residual; the released per-texel nonnegative
  clamp remains part of the table's semantics. Densification, Adam ancestry,
  PLY/safetensors resume, and uninterrupted-versus-segmented training are
  covered.
- The first full-checkpoint render exposed one material semantic missed by the
  source audit: PowerFoam clamps each directional texel before spatial
  interpolation. Moving that clamp into the shared CPU/WGSL/training contract
  raises Blade's all-37 Bonsai score from 27.8508 to 28.3078 dB versus the
  official 28.3432 dB. Quantized official and Blade renders agree at 59.37 dB.
  The Rust-only `import_powerfoam_directional` tool attaches the 249,404,948-
  byte released table to the already cross-rendered PLY. Assets stay ignored.
- Zero-initialized 255-step Room screens reject training this table at the
  measured schedule. Learned axes/colours reach 24.4147 dB held out and fixed
  axes reach 24.4089, versus 24.4668 control. Training takes 88.4/88.6 seconds
  versus 14.6 seconds, and the 200K-cell PLY grows 115→408 MiB. A 4,096-ray,
  160-path graph needs 6.19 GiB host and 11.64 GiB VRAM; 2,048 rays reduce that
  to 3.78 GiB and 4.96 GiB. Keep the table opt-in and do not run a long
  two-scene gate until backward cost is materially lower.
- The complete 164-test training suite now reuses one serialized physical GPU
  context. Previously, repeated Vulkan context teardown eventually selected
  llvmpipe and made later tests report success by skipping. Every GPU test now
  executes on the 5070; the 6 GiB cgroup reports no OOM, pressure, swap, Xid,
  or device fault. `train_colmap` also reuses the existing GPU view evaluator;
  scalar CPU evaluation became disproportionately slow for the nested table.

#### M2bm — Omit fixed directional-axis backward (done)

- A zero directional-axis learning-rate now places the existing axis parameter
  behind Meganeura's zero-cost `stop_gradient` alias. The axes remain ordinary
  serialized parameters and use the same forward graph, but no axis gradient
  or Adam state is built. Positive axis rates retain the original learned-axis
  path. Adam snapshots used across densification now enumerate only parameters
  that actually have gradients, so fixed axes are reloaded from the model while
  every trainable table still inherits its optimizer state.
- Physical-GPU tests cover fixed-axis colour learning, fixed-axis
  densification, learned-axis training, and checkpoint resume. A checkpoint
  produced by the earlier learned-axis graph also resumes into the fixed-axis
  graph; its extra axis moments are ignored and its PLY axis values are kept.
  This adds no model field, Meganeura operation, shader, shader entry/group,
  binding, pipeline, or public option.
- On the matched 200K-site Room screen at a 2,048-ray batch, training falls
  from 59.704 to 44.106 seconds (-26.1%) and GPU wait from 49.411 to 34.478
  seconds (-30.2%). The selected 4,096-ray run completes in 65.774 seconds at
  6.37 GB peak host memory with zero swap, OOM, throttling, or GPU fault and
  scores 26.6822/24.4125 dB train/held out. That is within 0.0022 dB of the
  earlier learned-axis result while retaining fixed axes.
- The first matched Bonsai screen is quality-neutral: the no-directional
  control scores 18.1176/17.4160 dB and fixed-axis directional colour scores
  18.1287/17.4162 dB. Directional training still takes 53.900 versus 8.383
  seconds (6.4x), identifying the 384-float colour-table backward as the
  remaining dominant cost. Keep full directional training opt-in and do not
  run a long gate until that cost falls without adding backend/shader variants.

#### M2bn — Omit frozen directional-weight backward (done)

- Hardware timestamps now follow `MEGANEURA_GPU_TIMING` when Blade creates the
  GPU context shared by rendering and training. The matched 4,096×160 Room
  profile measures 21.5 ms for the no-directional graph and 202.2 ms for
  fixed-axis directional colour. Adam grows from 1.2 to 65.3 ms, while the
  remaining large directional pointwise and reduction gradients account for
  most of the rest; the roughly 14 ms path recorder is not the bottleneck.
- Directional weights depend on axes, positions, radii, surface normals, and
  spatial detail offsets. When all five sources are already frozen, the same
  forward weights now sit behind a zero-cost `stop_gradient` alias. Any
  positive source rate, or a densification graph that retains geometry
  gradients, preserves the original backward path. Directional RGB gradients
  are unchanged. This adds no public option, model field, Meganeura operation,
  shader, binding, pipeline, or backend variant.
- The profiled directional graph falls from 202.2 to 176.2 ms (-12.9%). The
  255-step Room run falls from 65.774 to 58.503 seconds (-11.1%) and scores
  26.6813/24.4126 dB versus 26.6822/24.4125 before. Bonsai falls from 53.900
  to 47.399 seconds (-12.1%) with exactly the same reported 18.1287/17.4162
  dB and per-view scores. Both 10 GB cgroups report zero swap, OOM, throttle,
  or GPU fault. The remaining 5.7× Bonsai gap is chiefly the full colour
  parameter's Adam update and its required colour backward, not dead geometry.

#### M2bo — Keep private Adam state device-local (done)

- The large-table optimizer profile exposed a memory-placement bug rather than
  an Adam arithmetic problem. Meganeura allocated every first- and second-moment
  buffer as host-visible shared memory so that infrequent checkpoint and
  densification transfers could use a mapped pointer. The 200K-site directional
  colour table therefore updated two 153.6 MB moments across the host-visible
  heap on every step.
- Adam moments now use device-local memory. Initialization joins the existing
  build-time GPU clear, while explicit state reads, writes, and checkpoint
  restores use upload/download staging. Parameters, public APIs, update order,
  and the generic Adam shader are unchanged. This adds no graph operation,
  shader edit, shader entry/group, binding, pipeline, or backend variant.
- On the exact 200K-site 4,096-ray Room profile, Adam falls from
  65.27--66.16 ms to 16.16--16.93 ms (74.5% at the midpoint), and the complete
  directional graph falls from 202.2 ms before the frozen-weight change to
  126.6--127.5 ms with both changes. Four-step GPU wait falls from 0.899 to
  0.584 seconds. The matched control and candidate both score 25.8181 dB on
  the four training views and 25.2813 dB on eight held-out views, including
  identical reported per-view scores. The 10 GB cgroup peak falls from 6.797
  to 6.530 GB with zero swap, OOM, or GPU fault.
- Meganeura's complete all-target test and clippy gates pass on the RTX 5070;
  the state suite now explicitly verifies the GPU-cleared zero moments before
  exercising staged read/write and checkpoint paths. The remaining bounded
  performance target is the required directional-colour backward itself, not
  another Adam or shader-taxonomy specialization.

#### M2bp — Keep dedicated parameter gradients device-local (done)

- The next profile found the same placement mistake on parameter gradients.
  They must keep dedicated allocations because runtime optimizer passes consume
  them after the static graph, but optional gradient inspection does not require
  a permanently mapped buffer. Large directional-colour accumulation was
  therefore paying host-visible bandwidth on both the backward write and the
  following optimizer read.
- Meganeura's memory plan now separates allocation lifetime from host
  visibility: parameter gradients stay dedicated but use device-local memory.
  Existing inspection, CPU clipping, and legacy CPU-SGD paths stage transfers;
  the device-local kill switch still restores the all-shared diagnostic layout.
  A new CPU-clipping test verifies the complete staged read/scale/write path,
  and the full suite caught and closed the initially missed CPU-SGD direct
  mapping before the change was selected.
- The downstream all-target gate found one more concrete host-visibility
  contract: a parameter gradient can itself be a compile-time constant, and
  session construction initializes constants from the host. The memory plan
  now leaves only those constant gradients shared. A direct scalar-loss
  regression and the previously aborting oriented-surface Jacobian test both
  pass; this exception is tiny and does not affect the measured colour table.
- On the same 200K-site Room profile, the 38.4M-element colour-gradient add
  falls from 9.51 to 0.76 ms, Adam from 16.93 to 3.01 ms, and the steady full
  graph from 126.65 to 87.72 ms (-30.7%). Four-step GPU wait falls from 0.584
  to 0.444 seconds. The four training views remain 25.8181 dB and all eight
  held views remain 25.2813 dB with identical reported per-view scores. The
  10 GB profile scope peaks at 6.934 GB with zero limit, swap, OOM, or GPU
  fault event.
- Meganeura fmt, warning-denied clippy, and every all-target test pass in a
  12 GB cgroup; the final scope peaks at 10.219 GB with no limit event or swap.
  This generic allocator change adds no graph op, shader edit, shader
  entry/group, binding, pipeline, renderer, or backend variant. The remaining
  directional cost is now the actual per-ray colour function rather than
  storage placement.
- The final locked Blade workspace fmt, warning-denied clippy, and all-target
  tests also pass under a 12 GB cgroup. That scope peaks at 4.791 GB with zero
  limit, swap, OOM, kill, or GPU-fault events on the RTX 5070.

#### M2bq — Omit every frozen surface-field gradient (done)

- The directional-only graph still differentiated all optional surface
  parameters except its axes, even when their learning-rate ratios were zero.
  Normals, plane offsets, spatial colour/detail tables, density logits, and
  both Spherical Voronoi tables now use the same zero-cost `stop_gradient`
  contract as fixed directional axes. They remain ordinary named parameters
  for PLY/safetensors upload and readback. Checkpoint state already enumerates
  only parameters that actually have gradients, so old extra moments remain
  safely ignored and enabling a formerly fixed field starts its moments at
  zero.
- A wholly frozen oriented geometry now consumes the recorder's exact interval
  directly and requests no surface Jacobian buffers. When positions or radii
  move, the full tangent still retains the fixed surface-plane term: the
  recorder includes it in its reference tangent, so it must be evaluated to
  cancel exactly even though no gradient reaches the fixed normal or offset.
  Surface queries remain available independently for spatial detail.
- On a matched 98,831-site Room directional-colour continuation at
  4,096 rays × 160 path entries, the fixed-geometry training graph falls from
  354 to 262 passes and its steady GPU time falls from 75.03 to 68.88 ms
  (-8.2%). Four-step GPU wait falls from 0.349 to 0.328 seconds. The control
  and candidate both score 21.2093 dB on four training views and 20.4618 dB on
  the next view. With the `radfoam-v1` moving-position schedule, the graph
  still falls from 424 to 377 passes, setup from 19.630 to 5.838 seconds, and
  peak host memory from 6.876 to 5.774 GB; the required full tangent limits
  steady improvement to 111.87→108.67 ms (-2.9%). Its 20.1748/19.7137 dB
  scores are unchanged.
- The complete 520-test all-feature workspace gate, warning-denied all-target
  clippy, and formatting pass under 12 GB cgroups on the RTX 5070. Tests peak
  at 3.776 GB and clippy at 0.976 GB, with zero swap, OOM, throttling, Xid, or
  GPU fault. This adds no public option, model field, graph operation, shader,
  shader entry/group, binding, pipeline, backend variant, or dependency. The
  remaining cost is the required released directional colour function itself.

#### M2br — Fuse grouped directional-table scatters (upstreamed)

- Meganeura commit `c409bb2` extends its existing row-scaled embedding-scatter
  fusion to equal-width groups within a gathered row. The released directional
  table reduces each 64-float colour row in eight-float groups; its backward
  previously materialized three 41,943,040-element broadcasts and three
  equally large multiplies before the existing atomic scatters. The compiler
  now encodes the group width in the existing scatter mode and performs the
  multiply at accumulation time. This adds no graph operation, shader entry or
  group, binding, pipeline, or backend variant.
- On a matched 98,831-site Room continuation at 4,096 rays × 160 path entries,
  the graph falls from 262 to 256 passes. Across 32 warmed updates, GPU graph
  time stays within 34.48--34.85 ms for the pinned control and 30.83--31.33 ms
  for the candidate, about a 10.4% reduction. The path recorder remains about
  6.1 ms in both arms. After all 32 updates, both freshly serialized PLYs score
  24.2420 dB over the four training views and 21.1172 dB on the next held view.
- Structural coverage proves the grouped broadcast is removed. A physical-GPU
  repeated-index test compares the complete table gradient against CPU
  accumulation, and Meganeura's formatting, warning-denied all-target clippy,
  and complete all-target suite pass. The test scope peaks at 9.97 GB; both
  profile scopes stay below 5.41 GB, with zero swap, OOM, throttle, or GPU
  fault. A longer two-scene quality gate remains deferred until this upstream
  commit is merged and pinned.

#### M2bs — Compact directional-distance expansion (done)

- The directional kernel no longer copies every spatial view direction eight
  times or broadcasts every learned temperature over XYZ. It evaluates the
  equivalent identity `|T d - a|² = T² |d - a/T|²` through Meganeura's
  existing eight-wide pairwise-distance operation, then uses the same
  regularized square root and softmax as before. A shared square-root helper
  keeps that graph construction in one place. This adds no graph
  operation, shader, shader entry/group, binding, pipeline, backend variant,
  public option, model field, or dependency.
- Stacked on grouped-scatter commit `c409bb2`, the matched 98,831-site Room
  graph falls from 256 to 253 passes. Across 32 warmed updates at 4,096 rays ×
  160 path entries, GPU graph time falls from 30.83--31.33 to 26.34--26.82 ms,
  about 14.5%. A matched 256-update continuation falls from 17.873 to 15.943
  seconds (-10.8%). Both arms score exactly 29.1014 dB over the four training
  views and 18.9553 dB over eight held-out views at reported precision; every
  printed held-out view is identical.
- The physical-GPU graph test agrees with the CPU renderer for the complete
  directional colour and checks both radius and learned-axis gradients against
  CPU finite differences. The 256-update control/candidate scopes peak at
  5.50/5.54 GB with zero swap, OOM, pressure, throttle, Xid, or GPU fault. No
  scene-matched weighted Bonsai checkpoint is present locally, so the longer
  second-scene gate remains deferred rather than substituting a cross-scene
  initialization.

#### M2bt — Prepack frozen directional axes (done)

- When the directional-axis learning rate is zero, session upload now derives
  a unit-axis table and a scalar-temperature table once. The raw axes remain
  the checkpoint authority, while each training step gathers the two immutable
  tables instead of normalizing 41,943,040 repeated axes on the GPU. Learned
  axes retain the original differentiable path. The prepack is internal to the
  training graph: no model/PLY field, public option, graph operation, shader,
  shader entry/group, binding, pipeline, backend variant, or dependency is
  added.
- Stacked on `c409bb2` and M2bs, the matched 98,831-site Room graph falls from
  253 to 251 passes and from 26.40--26.78 to 22.55--22.94 ms, about -14.5%.
  The 256-update pipeline falls from 15.943 to 13.393 seconds (-16.0%), while
  cgroup memory falls from 5.54 to 3.56 GB (-35.7%). Both full-table arms score
  exactly 29.1014/18.9553 dB over four training/eight held views at reported
  precision, including identical printed results for every held view.
- A varying-magnitude physical-GPU test compares all 64 prepacked directional
  weights with the direct CPU identity. Fixed-axis graph construction and
  training tests verify that both derived tables are present, frozen, and
  usable, while the learned-axis CPU finite-difference gate continues through
  the unmodified path. The complete 523-test all-feature workspace gate,
  warning-denied all-target clippy, and formatting pass on the RTX 5070. Tests
  peak at 3.19 GB and clippy at 1.00 GB, with zero swap, pressure, OOM,
  throttling, Xid, or GPU fault.
- A current no-table control, made by removing only the directional table from
  the same initializer, takes 6.014 seconds and 0.919 GB for the same 256
  updates versus 13.393 seconds and 3.565 GB with the table. Directional colour
  raises training PSNR from 27.1422 to 29.1014 dB but lowers the eight-view
  held mean from 19.8841 to 18.9553 dB (-0.9288 dB). The former 5.7--6× time
  gap is materially smaller at 2.23× on this Room gate, but the table remains
  quality-negative and memory-heavy; keep it opt-in.

#### M2bu — Multi-view directional-table quality gate (rejected)

- The four-view loss is not just an underconstrained screen. At the same 256
  updates, increasing Room training coverage to 32 views gives the no-table
  control 24.7981/20.2976 dB over 32 training/eight held views. Directional
  colour at learning-rate ratio 1.0 reaches 24.8909/18.4958 dB: only +0.0928
  dB on the training mean and -1.8018 dB held out.
- Ratios 0.1 and 0.01 score 25.0152/20.2332 and 24.8291/20.2862 dB. Their held
  deltas are still -0.0644 and -0.0114 dB; reducing the rate simply converges
  back toward the no-table result. No selected rate improves held quality, so
  another scalar-rate sweep is not justified.
- No implementation from this experiment is retained. The released table
  remains valuable for exact checkpoint interchange/rendering, and M2bt makes
  opt-in training materially cheaper, but zero-initialized training remains
  rejected until a different spatial/directional responsibility or
  regularization hypothesis is defined and passes a second scene.

#### M2bv — Select path candidates at the measured crossover (done)

- The newer camera-independent sphere BVH is now faster than building a
  projected screen index for the sparse per-camera slices used by mixed-view
  training. On the 98,831-site Room cloud, matched 4,096-ray profiles put the
  BVH path recorder at 3.7--4.4 ms for four cameras, versus 5.8--6.2 ms for
  projected candidates. It remains faster at two cameras (3.8--4.3 ms versus
  4.4--4.8 ms) and eight cameras (3.9--4.4 ms); the two paths converge at one
  4,096-ray camera, where both take about 3.7--4.2 ms.
- Training therefore shares the BVH until a camera contributes at least 4,096
  rays, up from the obsolete 1,024-ray crossover. Projected candidates remain
  available for dense batches and complete images. This changes only the
  internal selection threshold: no path algorithm, shader, entry/group,
  binding, public option, model field, format, or dependency is added.
- A matched 256-update, four-view Room continuation falls from 6.014 to 5.035
  seconds (-16.3%) and from 0.919 to 0.523 GB peak cgroup memory (-43.1%). Its
  freshly serialized result preserves 27.1422 dB over the four training views
  and 19.8841 dB over eight held views. Both runs cover the same exact path
  oracle; the held result is unchanged at reported precision.

#### M2bw — Prepack frozen spatial-detail sites (done)

- The production continuation freezes its surface normals and spatial-detail
  offsets. Session upload now projects those offsets into their tangent planes
  once, while the raw values remain the checkpoint authority. Training either
  field keeps the original differentiable projection path. This is an internal
  derived table, with no public API, model/PLY field, graph operation, shader,
  entry/group, binding, pipeline, backend variant, or dependency.
- On the same 98,831-site Room continuation, the graph falls from 197 to 195
  passes and its warmed time from 5.48--5.53 to 5.31--5.40 ms. Three matched
  256-update runs average 5.167 seconds before and 5.081 seconds after (-1.7%)
  at the same approximately 0.523 GB peak memory. The freshly serialized result
  preserves 27.1422/19.8841 dB over four training/eight held views at reported
  precision.

#### M2bx — Omit frozen identity surface detail (done)

- A model may retain the eight spatial-detail sites for exact PLY and
  checkpoint interchange while its frozen residual is the identity: zero
  height and colour, uniform density logits, and zero directional colour. The
  training graph now detects that case and keeps every raw parameter as the
  serialization authority without evaluating the residual graph fed by the
  recorded spatial queries. Any effective or trainable detail field restores
  the full graph.
  Nonzero site offsets and directional axes alone do not affect the rendered
  value, so they remain valid identity metadata.
- The current 98,831-site Room initializer has exactly this form. Its graph
  falls from 195 to 153 passes and warmed graph time from 5.42--5.81 to
  2.78--2.92 ms. Three matched 256-update training phases fall from a
  5.081-second mean to a 3.520-second mean (-30.7%), while peak cgroup memory
  falls from approximately 0.523 to 0.293 GB. A freshly serialized result
  preserves 27.1422/19.8841 dB over four training/eight held views at reported
  precision.
  This adds no public API, model/PLY field, graph operation, shader,
  entry/group, binding, pipeline, backend variant, or dependency.

#### M2by — Omit identity-detail path queries (done)

- The path recorder already exposes surface-query payloads as optional, but it
  previously required them whenever the uploaded cloud retained spatial-detail
  metadata. Training now requests those existing outputs only when it evaluates
  effective detail. The cloud and serialized fields stay intact; zero heights
  preserve the base oriented interval, while any effective or trainable detail
  restores the query path. A physical-GPU oracle covers the same identity cloud
  with and without query outputs. Nonzero-height detail continues to require
  and test the full path.
- On the 98,831-site Room continuation, the warmed path recorder falls from
  3.77--4.44 to 2.83--3.28 ms; its interval-clipping pass falls from
  3.03--3.70 to 2.09--2.54 ms. Three matched 256-update training phases average
  3.111 seconds, another 11.6% below M2bx and 38.8% below M2bw, at approximately
  0.294 GB peak cgroup memory. The freshly serialized result again preserves
  27.1422/19.8841 dB over four training/eight held views. No operation, shader,
  entry/group, binding, pipeline, backend variant, public option, model field,
  format, or dependency is added.

#### M2bz — Omit frozen per-view exposure gradients (done)

- Per-view exposure is opt-in through `BLADE_VOLUME_PER_VIEW_EXPOSURE`; its
  production learning-rate ratio defaults to zero. The graph nevertheless
  used to differentiate all three exposure channels and allocate optimizer
  state before applying that zero multiplier. Training now reads the existing
  rate before graph construction and keeps the exposure parameters as frozen
  forward inputs unless the rate is nonzero. Opt-in exposure training and its
  parameter names are unchanged.
- On the same 98,831-site, four-view Room continuation, the graph falls from
  153 to 147 passes and its warmed time from 2.76--2.84 to 2.43--2.48 ms.
  Three matched 256-update training phases average 3.049 seconds, 2.0% below
  M2by's 3.111-second mean; traversal is now the larger cost. A freshly
  serialized result again preserves 27.1422/19.8841 dB over four
  training/eight held views at reported precision. No graph operation,
  shader, entry/group, binding, pipeline, public option, model field, format,
  or dependency is added.

#### M2ca — Reuse frozen forward intervals (done)

- PowerFoam clipping already computes each candidate's final support-,
  radical-plane-, and oriented-surface-bounded interval before sorting. The
  recorder nevertheless repeated the sphere and surface clip while emitting
  `dt`, even when training requested neither geometry Jacobians nor spatial
  detail queries. That forward-only path now carries the effective far
  endpoint through the existing candidate-face scratch and subtracts the
  already stored near endpoint. Full/surface Jacobian and detail-query modes
  retain the original face endpoints and differential path.
- On the same production Room continuation, the warmed serial record pass
  falls from 2.23--2.43 to 1.96--2.05 ms (about 13%), and total recorder time
  falls from 2.97--3.16 to 2.70--2.79 ms (about 10%). Three matched 256-update
  training phases average 2.970 seconds, 2.6% below M2bz's 3.049-second mean.
  A freshly serialized result again preserves 27.1422/19.8841 dB over four
  training/eight held views at reported precision.
- A dedicated physical-GPU oracle covers the weighted oriented forward-only
  path against the independent CPU implementation; the complete path suite
  continues to cover differential/detail modes, serial/workgroup clipping,
  batching, overflow, and truncation. No buffer, binding, shader entry/group,
  pipeline, public option, model field, format, graph operation, or dependency
  is added.

#### M2cb — Use direct volumetric exponentials (done)

- The core integrator retained its original
  `exp(-x) = recip(sigmoid(x)) - 1` workaround from before Meganeura exposed a
  differentiable exponential. It now evaluates transmittance as
  `exp(-cumsum)` and alpha as `1 - exp(-density * dt)`, matching the existing
  direct-exponential surface-detail path and deleting the obsolete surrogate.
- The production Room graph falls from 147 to 138 passes and its warmed time
  from 2.42--2.48 to 2.33--2.37 ms. Three matched 256-update training phases
  average 2.946 seconds, 0.8% below M2ca's 2.970-second mean; mean GPU wait
  improves by about 0.6%. The direct form changes f32 rounding, but the freshly
  serialized result preserves 27.1422/19.8841 dB over four training/eight held
  views at reported precision. A longer two-replica, 2,040-update gate confirms
  the result: mean training falls 18.923→18.772 seconds (-0.8%) and mean GPU
  wait 14.160→13.956 seconds (-1.4%), while both arms and both replicas score
  exactly 29.3164/17.8472 dB at reported precision. No new graph operation,
  shader, entry/group, binding, pipeline, option, format, or dependency is
  added.

#### M2cc — Select parallel PowerFoam clipping at its measured crossover (done)

- PowerFoam already had serial-per-ray and workgroup-per-ray interval clippers,
  but the workgroup path was reserved for at least 32 adjacency entries/site.
  A sweep over the same 98,831-site Room cloud places the crossover much lower:
  at 2.6 entries/site the paths are effectively tied, at 6.6 the workgroup path
  is 6--8% faster, and at 15.9 it cuts the settled record pass from
  2.00--2.12 to 1.75--1.80 ms. The selection floor is therefore four. The
  serial path remains useful: at 0.9 entries/site it is still slightly faster,
  so removing that existing entry point would regress genuinely sparse clouds.
- Three matched 256-update production Room runs average 2.789 seconds, 5.3%
  below M2cb's 2.946-second mean; mean GPU wait improves by the same 5.3%.
  The freshly serialized result remains 27.1422/19.8841 dB over four
  training/eight held views. This only changes the selection constant and its
  unit test; no shader, entry/group, pipeline, buffer, binding, operation,
  public option, format, or dependency is added.

#### M2cd — Stop clipping an already-empty PowerFoam interval (done)

- Radical-plane clipping can only raise `face_near` and lower `face_far`.
  Once they cross, no remaining neighbor can make the candidate valid again.
  The GPU recorder now returns at that point instead of evaluating the rest of
  the site's adjacency row. Final interval math and every valid path are
  unchanged.
- Three short profiles put the settled parallel-record median at
  1.85--1.87 ms versus 1.90 ms for the exact-source control. The complete
  256-update mean is effectively neutral at 2.784 versus 2.789 seconds. Two
  longer 2,040-update runs average 17.358 seconds versus 17.568 seconds
  (-1.2%), with the same reported loss. All 19 physical-GPU path-oracle tests
  pass. The change adds no shader entry/group, pipeline, buffer, binding,
  operation, public option, format, or dependency.

#### M2ce — Size the shared BVH stack to its exact bound (done)

- The support hierarchy is balanced by construction and capped at `2^30`
  leaves. After the gather workgroup expands six levels, each lane owns at most
  a `2^24`-leaf subtree. Its depth-first traversal needs 25 stack words
  including the root, not the previous conservative 32. One WGSL constant now
  drives both the 64-lane allocation and lane stride.
- Reducing shared scratch from 2,048 to 1,600 words lowers three settled Room
  BVH-gather profiles from 0.73--0.75 to 0.62--0.64 ms (about 14%). Three
  matched 256-update runs improve 2.784→2.753 seconds (-1.1%), and two
  2,040-update runs improve 17.358→17.067 seconds (-1.7%); mean GPU wait falls
  1.2% in both gates and reported loss is unchanged. All 19 physical-GPU path
  oracles pass. No traversal, shader entry/group, pipeline, buffer, binding,
  operation, option, format, or dependency is added.

#### M2cf — Extract depth from every PowerFoam support component (done)

- Reconstruction used the camera-seeded adjacency walk for both RadFoam and
  PowerFoam depth. That is exact for a connected Voronoi walk, but a weighted
  cloud's Čech graph may be disconnected: the walk can stop after one
  component even when a later support sphere is valid and more opaque. CPU
  depth extraction now uses the existing independently clipped PowerFoam
  splat oracle whenever radii are present; unweighted RadFoam keeps the walk.
- GPU depth uses the same projected candidates, exact interval clipping,
  `(effective depth, cell index)` sort, fixed path buffers, overflow fallback,
  and truncation checks as the existing headless PowerFoam renderer. Its one
  additional integration entry writes full-precision mode depth, opacity, and
  peak weight. The recorder stores the otherwise-unused forward next-cell
  word as the interval-entry f32 bits in depth mode. No model field, file
  format, dependency, training graph operation, or candidate implementation is
  added.
- A disconnected two-support regression makes the near component weak and the
  far component dominant. Both CPU and physical-GPU extraction select the far
  depth, while the broader GPU/CPU oracle covers ordinary, oriented, and
  spatial-detail weighted clouds. All 19 existing path-recording tests and the
  seven-test standalone rendering suite remain green.
- The selected dense and nested synthetic fixtures retain their exact printed
  depth RMSE and fused counts (2,880 and 3,156); depth time is 0.029 seconds in
  both. A 98,831-cell Room checkpoint traces five 64x42 maps in 0.6 seconds,
  with no candidate overflow or path truncation. Its 12 GiB scope peaks at
  242,622,464 host bytes with zero swap, pressure, OOM, Xid, or GPU fault.

#### M2cg — Fuse grouped directional-table gathers (upstream branch)

- Meganeura commits `ba4720b` and `db14f62`, stacked on the grouped-scatter
  branch at `c409bb2`, remove the remaining three materialized directional
  colour gathers. The compiler recognizes an embedding row reshaped into
  equal contiguous reduction groups and carries that factor through the
  existing generated reduction kernel's row-repeat metadata. The scheduler
  loads the selected source subrow directly. This is a generic code-generation
  mutation: it adds no graph operation, shader entry/group, pre-made shader,
  binding, pipeline, backend variant, or public API.
- Fusion also exposed a generic allocator bug: buffers whose producers and
  consumers had both been compiled away were treated as dedicated live-in
  allocations. Persistent and user-visible buffers remain pinned, while all
  untouched internal artifacts now share one four-byte dummy allocation. On
  the exact grouped-scatter control, this reduces the physical plan from
  6.603 to 2.898 GB. Adding grouped-gather fusion reduces it again to 2.463 GB,
  including a 435 MB reduction in device-local storage.
- On a matched 98,831-site Room continuation at 4,096 rays × 160 path entries,
  the warm graph falls from 234 to 231 passes and from 22.25--22.46 to
  19.96--19.99 ms, about -10.7%. Across 256 updates with the allocator fix in
  both arms, GPU wait falls 7.173→6.701 seconds (-6.6%), training wall time
  falls 11.258→10.947 seconds (-2.8%), active VRAM falls 3,758→3,310 MiB, and
  cgroup peak memory falls 3,474,280,448→3,369,148,416 bytes.
- Both arms score exactly 29.1014 dB over four training views and 18.9553 dB
  over eight held views at reported precision; every printed held-view score
  is identical. Structural compiler tests, generated-source tests, a physical
  GPU fused/unfused parity test, formatting, warning-denied all-target clippy,
  and the complete Meganeura all-target suite pass. The suite peaks at
  8,366,551,040 bytes under a 12 GiB cgroup with zero swap, pressure, OOM,
  throttle, Xid, or GPU fault.
- This makes the released table substantially more practical but does not
  change its quality decision: the matched no-table held score remains
  19.8841 dB, so zero-initialized directional-table training stays opt-in.
  Blade-volume remains pinned to Meganeura `0f87a8d` until the stacked upstream
  branches are merged.

#### M2ch — Store directional-colour training tables by channel (done)

- The released directional appearance remains RGB in `PointCloudModel` and in
  PLY files, but the training graph now keeps three `[N, 64]` parameter tables
  instead of one channel-major `[N, 192]` table. This directly matches the
  three colour consumers and removes the large split/concatenate gradient
  path. It adds no operation, shader, binding, pipeline, backend, dependency,
  public API, or file-format variant.
- Resume remains compatible with packed checkpoints. Loading detects the old
  `surface_detail_directional_colors` tensor and splits its parameter values
  and both Adam moments by cell and channel; the optimizer step is preserved.
  A physical-GPU regression verifies the exact migrated values and moments.
  Upload, readback, optimizer-rate, densification, and Adam-state remapping
  paths all enumerate the same three tables.
- With the same locally optimized Meganeura compiler in both arms, the matched
  98,831-site, 4,096-ray graph falls from 231 to 221 passes and from
  5.14--5.29 to 3.17--3.34 ms after warm-up, about -37%. Across 256 updates,
  pipeline time falls 11.125→10.323 seconds (-7.2%); a repeat takes 10.254
  seconds. Active VRAM falls 3,310→3,164 MiB and cgroup peak memory falls
  3,528,081,408→3,037,446,144 bytes.
- Both arms and the candidate repeat score exactly 29.1014 dB over four
  training views and 18.9553 dB over eight held views at reported precision;
  all eight printed held-view scores are identical. The changed scatter
  destination layout does change a sparse set of atomic accumulation results:
  directional-colour RMS delta is 3.53e-6 with 281 of 18,975,552 values above
  1e-5 and maximum delta 0.0092. Candidate repeats are closer than this, and
  all reported quality scores remain exact. This is the existing
  separate-process atomic-order tolerance, not a semantic change.
- Formatting, warning-denied workspace all-target clippy, the focused
  physical-GPU forward/backward and checkpoint-migration tests, and the full
  default and all-feature workspace all-target suites pass. The cold default
  suite peaks at 4,251,430,912 bytes in a 12 GiB cgroup; the all-feature repeat
  peaks at 3,491,340,288 bytes. Both have zero swap, pressure, OOM, Xid, or GPU
  fault. The no-directional-table held score remains 19.8841 dB, so the table
  stays opt-in pending a positive two-scene quality gate.

#### M2ci — Shared-site directional capacity screen (rejected)

- A bounded regularization prototype replaced each cell's 64 directional
  residuals per colour channel with eight residuals shared across its eight
  spatial detail sites. Upload averaged an existing full table and download
  expanded the shared values back into the unchanged model/PLY layout. This
  tested the smallest representation that could cut optimizer state by eight
  while preventing every spatial site from independently memorizing the four
  training views.
- The extra row replication makes the graph larger despite the smaller table:
  266 passes and 2.70--2.75 ms versus 221 passes and 3.17--3.34 ms for the full
  table. More importantly, the matched 256-update run takes 13.139 seconds and
  scores only 26.9244/18.1785 dB over four training/eight held views. The full
  table takes 10.323 seconds and scores 29.1014/18.9553 dB; the production
  no-table control takes 2.583 seconds and scores 27.1422/19.8841 dB.
- Sharing spatial sites therefore removes useful capacity without fixing the
  table's generalization or cost. The prototype and its alternate parameter
  layout are removed completely. A future retry needs a different
  spatial/directional responsibility model; neither another scalar rate nor a
  smaller copy of the same table is supported by the current gates.

#### M2cj — Temporal gradient accumulation (implemented experimentally; rejected)

- A bounded-memory prototype used Meganeura's native persistent gradient
  buffers to average multiple independently recorded ray batches before one
  Adam update. A second increment jointly stratified cameras across the update
  while preserving the exact one-batch RNG path and update-boundary resume.
  Repeated-input and segmented physical-GPU tests passed.
- Fixed-topology screens looked positive when update count was held constant.
  On the 98,831-site Room cloud, 2x accumulation with 32 training views raised
  train/held PSNR from 24.7981/20.2976 to 25.3196/20.5602 dB. With only four
  training views it instead overfit: 27.1422/19.8841 became
  27.5404/19.7973, and 4x fell to 19.5675 dB held out.
- The production-schedule gate resumed the 171,396-site Room step-6,000
  checkpoint through a complete trainable-geometry and densification window.
  At 500 Adam updates, 2x accumulation improved all-39 held PSNR
  23.9801→24.0771 dB and improved 30/39 frames, but training time grew
  31.571→41.033 seconds. Spending a slightly larger 43.230-second budget on
  700 ordinary updates instead reached 24.3756 dB held out and beat the
  accumulated arm on all 39 frames by 0.02--0.54 dB.
- Optimizer updates, not rays per update, are the better use of compute in the
  current production regime. The configuration field, CLI option, sampling
  helper, training-loop nesting, and tests are removed; no runtime or
  checkpoint format ever changed. Revisit only if a future model cannot fit a
  statistically adequate per-update batch in memory.

#### M2ck — Sparser trainable-geometry topology cadence (rejected)

- Exact Qhull rebuilds account for 16.6--17.5 of 30.4--31.6 training seconds
  in a 500-update Room window, so the existing `--geometry-rebuild-every`
  control was screened without adding code. Candidates divide the 500-update
  densification interval, ensuring contribution sampling and densification see
  freshly rebuilt topology.
- Rebuilding every 250 rather than 100 updates cut repeated Room training time
  from a mean 31.001 to 23.298 seconds (-24.8%). Mean held-out PSNR was
  statistically neutral, 23.9613→23.9819 dB. The independent 200,000-site
  Bonsai gate rejected it: 28.018→18.216 seconds came with
  23.4068→23.2195 dB held out (-0.1873 dB).
- A less aggressive 125-update cadence was repeated in order-balanced Bonsai
  runs. Mean training time fell 27.957→24.691 seconds (-11.7%), but mean
  train/held PSNR fell 24.1401/23.4216→24.0611/23.3829 dB. Its two paired
  held-view comparisons regressed 25/37 and 27/37 frames, with mean deltas of
  -0.0343 and -0.0454 dB.
- The production default remains 100 updates. No new schedule, heuristic, or
  runtime variant is retained. A future performance attempt should make exact
  rebuilds cheaper rather than exposing the model to staler adjacency.

#### M2cl — Stack-allocate Qhull simplex indices (done)

- Profiling a 200,000-site exact rebuild attributes 2.51--2.57 seconds to
  Qhull and about 0.41 seconds to extracting and deduplicating its 1.32 million
  tetrahedra. Extraction previously allocated a four-element `Vec` for every
  tetrahedron. Filling the same four indices on the stack preserves facet,
  edge, CSR-row, and neighbor order without changing the topology algorithm.
- On fixed Bonsai and Room clouds, the new builder reproduces the stored
  3,041,020- and 2,975,854-entry CSR arrays byte for byte. Their extraction
  phases fall 410.3→386.4 ms (-5.8%) and 410.6→382.5 ms (-6.8%). A global
  edge-vector alternative is rejected because sorting it takes 446--461 ms.
- A complete five-rebuild Bonsai replay lowers topology time from the repeated
  control mean of 15.551 to 15.284 seconds (-1.7%) and training from 27.957 to
  27.609 seconds (-1.2%). Its 24.1694/23.4580 dB train/held-out score is within
  the adjacent controls' separate-process GPU variation. The strict all-target
  clippy gate and complete all-feature physical-GPU workspace suite pass; the
  latter peaks at 7,099,281,408 bytes with zero swap, OOM, or GPU fault.

#### M2cm — Ray-normalized half-batch continuation (scene-specific)

- The accumulation gate showed that optimizer updates were more valuable than
  extra rays per update, so a no-code schedule gate held total rays and exact
  topology work constant. The candidate replaces each 4,096-ray update with
  two 2,048-ray updates, keeps 16 views per batch, halves the base learning
  rate `0.1→0.05`, and doubles fixed topology and densification intervals.
  Keeping the original learning rate is rejected on Bonsai at 22.7129 dB held
  out, 0.7087 dB below the repeated control mean.
- From the 200,000-site Bonsai step-8,000 checkpoint, two candidate replicas
  process the same 2.048 million rays and five rebuilds as the controls. Mean
  train/held PSNR rises 24.1401/23.4216→24.4689/23.6844 dB. Both replicas
  improve 35/37 held views. Mean training time is 28.295 versus 27.957 seconds
  (+1.2%).
- A second equal-ray window confirms that the Bonsai gain persists rather than
  merely arriving earlier. Across 4.096 million cumulative rays and ten exact
  rebuilds from the common step-8,000 checkpoint, 4,096 rays reaches
  24.2704/23.5192 dB while 2,048 rays reaches 24.7751/23.8901 dB
  (+0.5047/+0.3709). All 37 held views improve. Cumulative training is
  55.558→56.254 seconds (+1.3%).
- An independent Room continuation includes one complete growth boundary and
  ends at 196,943/196,955 sites versus 196,946/196,943 for the controls. Mean
  train/held PSNR changes 24.8786/23.9613→24.8982/23.9886 dB, while training
  falls 31.001→30.233 seconds (-2.5%). The two paired held-view splits are
  19/2/18 and 24/0/15 improved/tied/regressed.
- A second matched window takes both Room trajectories through their final
  growth to exactly 200,000 sites. Across two replicas, the 4,096-ray control
  averages 25.5531/24.4810 dB train/held in 33.190 seconds; 2,048 rays averages
  25.5460/24.4800 dB in 33.910 seconds. The effectively zero -0.0010 dB held
  delta costs 2.2% more time, and paired view splits disagree at 10/2/27 and
  20/2/17 improved/tied/regressed. Room therefore does not preserve the
  first-window gain at the capacity boundary.
- Continuing the ray-normalized ladder to 1,024 rays is rejected by the
  cross-scene gate. Two Bonsai replicas improve the 2,048-ray mean by
  +0.3344/+0.2357 dB train/held at 29.864 seconds (+5.5%). Two Room replicas
  instead change it by -0.0153/-0.0129 dB at 31.796 seconds (+5.2%), with
  paired held-view splits of 16/2/21 and 13/1/25. The extra updates help
  Bonsai, but do not transfer well enough to justify their cost.
- A final 512-ray rung shows diminishing returns even on Bonsai: its single
  24.9883/24.0491 dB run adds only 0.1291 dB held over the repeated 1,024-ray
  mean while taking 32.846 seconds (+10.0%). The corrected Room run reaches
  24.7929/23.9046 dB in 36.404 seconds, -0.1053/-0.0840 dB below the repeated
  2,048-ray mean while taking 20.4% longer; it regresses 33/39 held views
  against the first 2,048-ray replica. The ladder stops here.
- The half-batch schedule advances only as a scene-gated Bonsai post-cap
  quality option using existing CLI controls. It is not a general Room or
  growth-stage recipe. The global 4,096-ray defaults remain selected; no
  automatic phase, configuration field, checkpoint rule, shader, graph
  operation, or dependency is added.

#### M2cn — Decouple densification correctness from topology cadence (done)

- The 512-ray Room screen exposed an operation-order bug when its 800-update
  topology cadence did not divide the 4,000-update densification phase from
  the resumed global step. The trainer downloaded current parameters at the
  growth boundary but refreshed adjacency and GPU traversal geometry only
  when the independent topology schedule was also due. Contribution scores,
  one-ring pruning, and unweighted sibling placement could therefore combine
  the current host model with the preceding traversal snapshot.
- Every active trainable-geometry densification boundary now performs the
  required pre-resampling refresh. Coincident schedules still execute exactly
  one pre-resampling refresh; frozen geometry remains unchanged. A physical
  GPU regression makes the first growth and topology boundaries deliberately
  non-coincident and asserts that the refresh path runs before training
  continues on a valid grown model.
- The completed pre-fix 512-ray Room run is excluded from the quality ladder.
  Earlier 4,096/2,048/1,024-ray screens aligned their schedules and remain
  valid. The affected arm was rerun from its common checkpoint after this fix;
  it refreshed at the non-coincident step-10,000 boundary, grew to 196,988
  sites, recorded zero path truncation, and supplied the rejected 512-ray
  quality result above.

#### M2co — Stage released directional residuals (two-scene quality gate passed; performance gate open)

- Jointly fitting a fresh directional table with base density and SH remains
  rejected. A clean 32-view Room rerun reaches 24.8823/18.4798 dB
  train/held versus the table-free base's 24.7981/20.2976 dB. The learned
  table puts 42.01% of its energy in its per-row mean and 57.99% in directional
  variation, but removing 25--100% of that mean raises held quality only to
  18.7060 dB. Scaling the entire table toward zero peaks at 19.3585 dB. This
  is responsibility transfer during joint optimization, not merely a DC bias.
- Training the table as a residual after the base has converged resolves that
  conflict. `--freeze-base-appearance` omits density and SH gradients and Adam
  state while leaving their values in the ordinary graph. On Room, 256
  directional-only updates improve 24.7981/20.2976 to 25.1415/20.3937 dB and
  improve all eight held views. On an independently prepared 200,000-site
  weighted, oriented Bonsai cloud, the matched tiny-rate control scores
  21.3762/20.7550 and the exact frozen-base candidate scores
  21.5280/20.8709 dB; all 37 held views improve. A graph regression asserts
  that only the three released directional-colour tables receive gradients;
  an end-to-end PLY audit finds zero changed density or SH bit patterns.
- The quality result does not yet make the table a default. With the rebased
  grouped scatter/gather Meganeura compiler work and a 76-entry Room path row,
  directional-only training takes 7.009 seconds versus 2.352 seconds for the
  matched base graph (2.98×). It records a 75/76 maximum and zero truncated
  rays while preserving 25.1415/20.3937 dB. The new synchronized telemetry
  reports only 23.7 active entries per ray on average: 68.8% of the fixed row
  is padding. Profiling assigns 9.21 ms to
  the directional graph versus 1.73 ms for the base graph; the required dense
  per-path table evaluation and reductions dominate, not Adam or traversal.
  Keep training opt-in until active path rows can be compacted/repacked, or an
  equivalent generic sparse reduction is compiled, below the 2× gate. Add no
  directional-specific operation, shader group, or shader-entry variant.
- Shorter fixed rows do not close the gate. At 48 entries, 0.985% of Room
  training rays truncate, but final quality is effectively unchanged at
  25.1413/20.3938 dB. Directional/base GPU wait is still 2.968/1.262 seconds
  (2.35×). At 32 entries, the wait ratio reaches 2.01× only by truncating
  11.31% of rays; held quality falls 0.0176 dB to 20.3761 dB. Neither cap is
  selected.
- Two exact-shape sparse scheduling prototypes are also rejected and removed.
  Folding the binary mask into generated reductions cuts ten passes but raises
  the warm graph from about 9.21 to 9.44 ms. Short-circuiting zero-absorbing
  pointwise DAGs reaches 8.77--8.86 ms in the isolated profile but does not
  improve the complete 48-entry run (2.994 seconds GPU wait). Predicating a
  fully dispatched tensor is not compaction. The next performance attempt must
  reduce dispatched rows, most likely by repacking active paths and driving
  the continuation through an indirect dispatch.

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
  `try_init_gpu()`. Meganeura is pinned at `84a8a45`, including the merged
  shader-modifier cleanup and the scalable training kernels required by the
  production graph. blade-graphics is unified at `200188b` (the revision
  Meganeura pins) so the renderer and Meganeura session share one GPU context
  without Vulkan validation errors.
  The discrete cell walk remains outside autograd: the GPU path recorder emits
  stable cell roles, intervals, and geometry Jacobians, then meganeura
  differentiates the continuous integration graph.
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
    strict-lower-triangular ones matrix, direct exponential transmittance,
    and three parallel per-channel pipelines summed into one scalar L1 loss.
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

    `train_colmap --masks masks/` reads a directory mirroring the relative
    image paths, rectifies continuous foreground coverage with the calibrated
    camera, composites target RGB over the selected training background, and
    supervises opacity. Masked CLI runs default the opacity weight to 1;
    unmasked behavior remains unchanged. A missing selected mask is an error.

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

- **M3c-5 — Complete-render Gaussian normal refinement (implemented and
  four-cloud gated).** For calibrated captures under at least two measured
  lights, the relightable surface can now refine every observed Gaussian normal
  against the complete production PBR renders rather than treating overlapping
  center samples independently. Each of eight rounds renders one deterministic
  antithetic normal pair, uses the per-pixel error difference inside each
  projected footprint to choose local directions, and accepts only a lower
  full multi-light objective. Every proposal refreshes all affected TLASes:
  surface normals rotate the finite Gaussian proxy, so a buffer-only update is
  incorrect even when centers and radii are fixed. An analytical GPU test
  verifies recovery toward a known normal and exact agreement with a freshly
  rebuilt tracer.
- On four independently trained synthetic clouds, with six training poses,
  four known lights, two held-out poses, and `studio` entirely excluded from
  fitting, held-light mean/worst PSNR improves 19.08/18.84→19.43/19.20,
  20.19/19.78→20.44/20.02, 19.80/19.21→20.05/19.42, and
  19.61/19.36→19.91/19.59 dB. The pass takes 0.72--0.75 seconds and changes no
  shader, graph operation, shader entry/group, binding, pipeline, or model
  format. Coverage falls 0.1--0.4 points and nearest-truth normal RMSE rises
  0.17--0.35 degrees, so this is explicitly a rendered shading-normal polish,
  not evidence that image-only geometry is solved. `synthetic_foam` and the
  production `reconstruct` CLI expose it as opt-in; the latter requires a
  paired secondary known-light capture.

- **M3c-6 — Preserve repeat-light view correspondence (implemented and
  four-cloud gated).** The calibrated multi-light normal solve no longer asks
  an average of radiance from different projected image locations to provide
  its only correction. It retains the existing shared normal as an anchor,
  solves each repeated camera observation across the measured lights, averages
  those view-local directions in normal space, and applies a halfway update.
  A deterministic outlier regression covers the distinction. Before the
  complete-render pass, four truth-normal RMSEs improve by 0.13--0.37 degrees;
  after it they improve by 0.01--0.42 degrees. Held-light mean/worst PSNR gains
  0.06--0.17/0.05--0.11 dB on all four clouds, with 0.0--0.3 coverage-point
  movement and about 0.05 seconds of added solve time. A robust 45-degree
  inlier arm and larger 1.62-cell support are rejected by the joint geometry
  and held-light gate. Production and synthetic calibrated-repeat paths share
  the selected implementation; single-light capture is unchanged.

- **M3c-7 — Gate Gaussian surfaces through masked PowerFoam (implemented).**
  `synthetic_foam --surface-powerfoam-steps-per-view N` now runs the same
  fixed-center continuation exposed by production `reconstruct`, before the
  known-light normal and PBR material stages. This keeps the complete
  synthetic truth and held-light gate available as the continuation evolves,
  without adding a renderer or serialized representation. A four-cloud gate
  rejects trainable surface offsets and a sparser topology rebuild cadence;
  the retained continuation still learns only density, SH, radii, and normals
  at the selected 100-update refresh cadence. The benchmark can persist the
  trained static light field through `--surface-powerfoam-output`, mirroring
  production. A profiled, order-balanced gate selects a 192-entry training
  path row: it cuts continuation time by about 19% with zero truncation,
  unchanged position/coverage, and neutral-to-positive median static-LF and
  held-light PBR scores. A faster 160-entry row is rejected by a 0.09 dB
  static-LF tail regression.
  A follow-up single-light audit keeps the normal prior and trainable support
  radii: removing either is faster but consistently worsens truth geometry and
  held-light output. Freezing the converged appearance for a separate offset
  pass and deriving materials from that static field are also rejected. The
  continuation therefore retains one representation and one joint schedule;
  no staged optimizer mode, shader variant, or material fallback is added.
  A dispatch-level follow-up also rejects materializing the shared expressions
  feeding the surface-colour reductions: it adds two passes and slows a GPU
  step from about 6.0 to 6.21--6.28 ms. The retained Meganeura path discovers
  the fused multi-gather reductions through code generation; the remaining
  performance targets are their generic implementation and the radius-gradient
  atomic accumulation, not new operation or shader-entry variants.
  Profiling the three dominant four-column reductions rejects two further
  generic-code mutations. Vector evaluation changes low floating-point bits;
  preserving the scalar loop while omitting its single-lane workgroup scratch
  is bit-exact, but leaves the 258-pass step unchanged at 5.06--5.32 ms versus
  5.06--5.35 ms. Meganeura therefore keeps the existing reduction lowerer.
  Blade now forwards runtime environment options into the shared training
  session, so plan dumps, dispatch serialization, alias controls, and GPU
  timing can diagnose this production graph instead of silently using the
  defaults.
  The static Gaussian consumer now keeps the full continuation only when a
  Gaussian fitted without the last training camera beats the corresponding
  baseline by at least 0.05 dB on that camera. The full five-cloud gate selects
  clouds 1--4 and rejects cloud 5, matching the direction of every true
  held-out result. Selected static mean/worst PSNR is 25.17/24.35 dB versus
  24.97/24.22 without continuation. This is orchestration around the existing
  PowerFoam and Gaussian fits; it adds no shader, operation, entry point,
  binding, model field, format, or dependency.
  Packing a second independent row into each invocation is retained: it keeps
  the portable 256-thread workgroup and exact scalar order while amortizing
  scheduling. On the same 5070 it lowers the warm graph from 5.14--5.33 ms to
  4.91--5.21 ms and the dominant reduction family from 1.76--1.79 ms to
  1.58--1.74 ms. A 120-update gate produces byte-identical PLY and raw-scene
  outputs; four rows per invocation loses the gain late in the run.

#### M3d — Online viewer attach

- During training, periodically convert `TrainerState → PointCloudModel` and hand it
  to a running `blade-volume-view` instance via a shared `Arc<Mutex<...>>`.
- This mirrors PowerFoam's `--viewer` flag.
- Not yet wired; the CLI dumps a final PLY and the existing viewer can
  load it after the run.

### M4 — Capture stage (done)

Only after M3 produces something worth looking at.

- [`CAPTURE.md`](CAPTURE.md) covers phone-app controls, recording geometry,
  frame-rate selection, cgroup containment, and sparse-model inspection.
- `etc/colmap.sh` now extracts frames, runs shared-camera feature extraction,
  sequential matching, and sparse mapping, then verifies the exact binary
  layout consumed by both training paths. It publishes atomically and refuses
  to replace an existing capture.

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

- Default-on training for the full PowerFoam directional table. Staged
  frozen-base fitting now passes Room and weighted/oriented Bonsai quality,
  but its locally optimized training time remains 2.98× the matched no-table
  graph and therefore misses the 2× production gate.
- Mobile capture app.
- Multi-GPU / distributed training.
- LOD or streaming for huge scenes.
