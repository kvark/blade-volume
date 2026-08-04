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

#### M2w — PowerFoam appearance reference audit (complete; implementation staged)

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
- Meganeura is pinned to `ad08f97`, which excludes detached scalar-gradient
  sentinels from optimizer state. Without that compiler fix, dead parameters
  still appeared trainable and Adam could read beyond the one-element sentinel.
  A structural GPU test requires frozen weighted positions/radii to have no
  gradient while density, normals, offsets, and spatial colour remain trainable.
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
  `try_init_gpu()`. Meganeura is pinned at `7717181`, including the scalable
  training, zero-sparse scatter, and packed-reduction work required by the
  production graph. blade-graphics is unified at `bd74bdc` (the revision
  meganeura pins) so the renderer and meganeura session share one GPU context.
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

- Full eight-detail-site displacement and per-detail-site directional
  appearance from PowerFoam. M2v and M2x are compact spatial and directional
  residuals; M2w records the exact remaining semantics and their storage cost.
- Mobile capture app.
- Multi-GPU / distributed training.
- LOD or streaming for huge scenes.
