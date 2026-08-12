# Gaussian reconstruction plan

Status: active. This plan narrows the reconstruction work to the user's current
goal: posed images in; a cloud of Gaussian surface particles, PBR materials,
and lighting out; relit novel views measured against known truth.

## Constraints

- Runtime geometry stays a point cloud. Polygonal assets may be used by Blade
  to generate synthetic truth, but no polygonal representation enters the
  reconstructed asset or renderer.
- Rust and WGSL only. Blade supplies synthetic PBR radiance and G-buffer truth;
  Meganeura remains the differentiable training backend.
- Keep durable geometry at the `PointCloudModel` boundary. The current
  relightable surface model is an experimental staging format; once its
  Gaussian contract is measured, move the surviving geometry and material
  fields into the shared model rather than creating another parallel engine.
- Every quality claim includes held-out poses, coverage, and worst-view PSNR.
  Training-view error alone is not evidence.
- Heavy runs use `etc/cgroup_run.sh` with memory and GPU telemetry.

## A. Establish Gaussian surface semantics

1. Persist the particle kernel in `.surfel`/`.rply` without invalidating old
   files. Old flag-zero files remain compact-kernel controls.
2. Implement a finite isotropic surface Gaussian in CPU and WGSL. A stored
   radius is three standard deviations and is also the acceleration support.
3. Make foam reconstruction emit Gaussian particles by default, with an
   explicit compact-kernel ablation.
4. Cross-check CPU and physical-GPU pixels, then sweep support on Bonsai and
   Room at identical geometry/material budgets.

Current result (2026-08-02): implemented. At the selected 1.7-cell support and
unchanged particle counts, Gaussian versus compact improves held-out sRGB PSNR
from 14.07 to 14.50 dB on Bonsai and 14.17 to 14.37 dB on Room. Covered-region
PSNR improves by 0.92 and 0.50 dB respectively. The support sweep from 1.4 to
2.6 selects 1.7 across both scenes; broader kernels regain coverage but blur
appearance and reduce PSNR.

## B. Make Blade truth reproducible

1. Port the `relight_data` harness from Blade's research branch to the current
   Blade API and keep a small ignored smoke configuration.
2. Generate deterministic direct-lighting data first: several poses, known
   intrinsics, one known training environment, one held-out relighting
   environment, and linear radiance plus depth/normal/material truth.
3. Store only the generator and a manifest schema in Git. Generated planes,
   images, and benchmark outputs stay under `target/audit-runs`.

Current result (2026-08-02): the harness is ported onto Blade's exact revision
pinned by this workspace and pushed as Blade branch
`reconstruction-relight-data`. The port exposed and includes the missing
environment-map lifetime fix: assigning a real map used to destroy its prepare
pipeline and jump through an invalid driver handle. Its GPU regression passes,
and the 4-view smoke fixture is bit-for-bit identical to the research-branch
output. The full direct-light fixture is 8 poses, 5 environments, 200x150 and
128 paths; generated data remains under `target/audit-runs`.

## C. Close the synthetic upper bound

1. Fuse depth/normal truth from training poses into Gaussian surface particles.
   This deliberately hands geometry to the first material/light experiment so
   representation error is measured apart from image-only geometry error.
2. With the training environment known, fit diffuse albedo first; then enable
   roughness and F0 only where multiple view directions provide lobe evidence.
3. Render held-out poses under both the training light and an unseen light.
   Report geometry coverage, normal/material error, whole-frame and covered
   PSNR, and the exact truth-renderer ceiling.
4. Replace truth materials, then truth normals, one at a time. Do not proceed
   to harder geometry while a simpler stage is below its measured ceiling for
   unexplained reasons.

Current result (2026-08-02): the first upper bound is implemented by
`synthetic_reconstruct`. Six selected training cameras contribute 99,881
foreground G-buffer samples; held-out cameras 1 and 5 are not loaded by fusion.
At a 0.08 voxel and 1.7-cell support they produce 28,835 Gaussian particles,
whose 56.4% held-out coverage is within 0.3 percentage points of truth. A
single known `sun-east` light fits the material, and unseen `studio` lighting
at the two unseen poses scores 26.57 dB (worst 25.79). Truth materials on the
identical geometry score 27.33 dB, leaving a measured 0.76 dB material gap.
The compact-kernel control reaches 26.06 dB, while 16 sampled visibility rays
fall to 25.06 dB and cost roughly 35x more per frame.

The fitted parameters are deterministically clustered into 64 shared PBR
materials. This reduces the asset from 1.85 MB to 0.92 MB while changing the
held-out score by only 0.01 dB. Parallel roughness-level prefiltering reduces
the two-light setup from 3.70 s to 0.64 s with identical scores. The latest
artifact is
`target/audit-runs/synthetic-reconstruct-final-v3/scene.rply`, SHA-256
`a4ec939a45d6ab5bdef3f774dcb3d81a375ac7130e04a72b722dacf16c54e886`.

The depth-only upper bound estimates each particle normal by
centroided local covariance over the fused positions and resolves its sign from
the cameras that actually contributed that particle. Seven neighbours give
1.97-degree normal RMSE. The fitted held-out relight score is 26.45 dB versus
26.57 dB with truth normals; the matched material oracle is 27.18 versus 27.33
dB. This closes the normal-truth ablation to 0.12--0.15 dB without adding any
polygonal geometry. `--truth-normals` retains the old control. Section D
performs the next honest ablation by replacing truth depth with the
image-trained foam surface.

## D. Remove truth geometry

1. Initialize from the existing image-trained foam's multi-view depth surface;
   convert each oriented particle to Gaussian center, normal/covariance,
   support, and opacity without a polygonal intermediate.
2. Optimize center offsets, tangent scale, normal/covariance, and opacity
   against all training photographs. Re-record visibility after geometry moves;
   stale hit assignments are not a valid gradient.
3. Densify only from measured residual and geometric support. Prune particles
   that remain unsupported or transparent. Keep held-out poses out of fusion,
   visibility, fitting, and topology decisions.
4. Move the validated Gaussian PBR fields into `PointCloudModel`, add a native
   checkpoint with optimizer state, and make the viewer consume that model
   directly.

Current result (2026-08-02): `synthetic_foam` is the first end-to-end image-side
gate. It initializes 2,048 RadFoam sites from cameras alone and trains them from
six views of RGB plus foreground alpha; G-buffer positions and normals are read
only after training for diagnostics. Cameras 1 and 5 are excluded from
training, surface fusion, refinement, observations, and material fitting.

Aligning the deliberately sparse outer lattice layer with the mean camera side
raises held-pose radiance from the 20.13 dB world-axis baseline to 24.17 dB
(23.72 worst) without increasing the site budget. The trained foam yields 2,152
Gaussian surface particles, 53.3% held-pose coverage, and 18.31 dB under the
unseen light (17.94 worst). A second independent training run reaches 24.27 dB
radiance and 18.38 dB relighting, so the result is not a selected lucky
checkpoint.

The 2026-08-03 rerun keeps the conservative 256-cell path cap and removes
contention from its exact-zero padded gradients in Meganeura. Full training
falls from 65.60 to 29.51 seconds, while held-pose radiance remains 24.18 dB
and unseen-light relighting reaches 18.56 dB (18.30 worst). It extracts 2,167
Gaussian particles at 53.9% coverage with 0.6201 position RMSE. The latest
baseline artifacts are `target/audit-runs/synthetic-foam-zero-scatter-v1/model.ply`
(SHA-256 `90e32fe1ed1bf6021204f8d9a3e8aa3bd7e2dd381adaf618b48e0e4a007f8374`)
and `scene.rply`
(SHA-256 `38221090e2f45f94bf8122015a8e2d195a5224f9d062b10b10e1ef66ffc198da`).

The selected graph-packing rerun keeps every checkpoint parameter and the
256-step path cap, but gathers the degree-2 RGB SH coefficients as one wide
table. Three full repeats train in 16.00--16.05 seconds and reach 24.09--24.20 dB
held-pose radiance, 0.624--0.627 position RMSE, and 18.39--18.65 dB under the
unseen light. The latest selected unweighted synthetic control, produced after
the follow-up path-mask cleanup and weighted-reference correction, is
`target/audit-runs/weighted-reference-rebase-v3/model.ply` (SHA-256
`cbdf65d844d7749806fa8227a49990d978bb9ee2bc6226013404cfcc3955fbe8`); its
Gaussian PBR surface is `scene.rply` beside it (SHA-256
`8bf8e6b3b9dd86040906ab65250b0d82329d169c24e0ab4b4e630d4d869758ae`).
That command constructs `radii = None`, so the artifact and repeats do not
exercise the weighted path; they remain useful only as a shared-default
regression control.

The correction closes a weighted-training error at geometry rebuilds. Fresh
recorder intervals and Jacobians were based on the newly downloaded cloud, but
the differentiable graph still measured position/radius deltas from the
session-initial cloud. The graph now rebases those inputs whenever the recorder
cloud is rebuilt, so `dt_ref + J * (x - x_ref)` uses one consistent snapshot.
A physical-GPU regression demonstrates both the stale offset and exact rebased
identity. This physical test, rather than the unweighted synthetic repeats, is
the correctness gate for the weighted fix.

Checkpoint resume now preserves that weighted representation as well. With no
explicit topology option, `train_colmap --init-ply` rebuilds Čech adjacency
from the radii stored in a PowerFoam PLY rather than dropping or reinitializing
them. A physical 8+8-step split matches an uninterrupted 16-step run exactly
for positions, radii, and CSR adjacency; density and SH differ by at most one
ULP across the separate GPU processes.

Higher camera-lattice position rates were not selected. A 0.08 ratio with a
300-step warmup improves the first three repeats to 25.52--25.85 dB on held
poses and 18.60--18.74 dB under the unseen light, but a fourth falls to 22.53
dB on the worst held pose. One repeat also has 12.58 world-unit training-depth
RMSE. Tiny atomic-order differences are being amplified by discrete topology;
neither more frequent rebuilds, a longer warmup, nor stronger quantile loss
stabilizes it. A matched Bonsai continuation also loses 0.40 dB over all 37
held views. The temporary warmup implementation is removed and both training
paths retain their previous position-rate defaults.

This is progress, not completion. The selected surface still has 0.620
world-unit position RMSE and 66.42-degree normal RMSE. Specular fitting is therefore
disabled for this gate: with two shared materials it can mistake a noisy
cluster for a perfect mirror and move held-light quality from 18.50 to 14.70
dB across otherwise similar runs. The fixed rough-dielectric fit is stable
within 0.07 dB and is about 5.8x faster. The next geometry milestone is a
differentiable shared Gaussian surface objective with refreshed visibility,
followed by normal/covariance optimization. Specular parameters stay gated until the
normal error and multi-light control show that they are identifiable.

## E. Performance and production gates

1. Profile separately: depth extraction, fusion/refinement, observation,
   material/light fitting, acceleration build, and per-view rendering.
2. Reuse scene acceleration and prefiltered lighting across views and splits.
3. Evaluate dynamic ray regrouping only after profiling shows traversal
   divergence dominates; retain a simpler control implementation.
4. Before each pushed milestone: format, clippy with warnings denied, full
   workspace tests, physical-GPU parity, cgroup peak memory/events, and a
   committed benchmark manifest with artifact hashes.

COLMAP capture loading now decodes and rectifies at most eight photographs in
parallel while preserving the name-sorted output and deterministic error order.
Ten alternating warm-cache runs reduce the Bonsai median from 0.76 to 0.32
seconds and Room from 0.68 to 0.23. Five complete two-scene A/B runs improve
median reconstruction throughput by 1.22x and 1.16x respectively; scene and
raw-float files are byte-identical, all reported scores are unchanged, and
median process RSS rises by only 3.6--4.0 MiB. The combined scope peaks at 450
MB with no pressure, swap, OOM, or GPU event.

Batching GPU depth submission/readback is not selected. Eight-camera groups
keep memory bounded and produce byte-identical scenes, but improve complete
default commands by only 0.8--1.0%. On the selected 256-pixel protocol Room
improves 2.79 to 2.74 seconds while Bonsai regresses 1.97 to 1.98. An unbounded
one-submission variant is similarly marginal and makes mapped memory grow with
the capture. Both are removed: traversal, not the host wait, is the remaining
depth bottleneck.

Specializing the GPU walk for unweighted RadFoam is also rejected. A uniform
branch around the power-radius math slows both scenes by 1--2%. A compile-time
variant removes those operations entirely, but leaves Room at 2.54 seconds and
slows the Bonsai median from 1.79 to 1.82. All weighted, oriented, relight, and
depth oracles pass and serialized outputs remain exact; the extra shader path
is nevertheless removed.

Changing the depth workgroup from 8x8 is rejected as well. Wider 16x4 and 32x2
groups regress Bonsai by 0.03 and 0.02 seconds; 32x2 also regresses Room by
0.07 seconds. A single-warp 8x4 group regresses both scene medians by 0.02
seconds. All variants pass the physical-GPU depth oracle and emit byte-exact
outputs, so the evidence isolates launch efficiency rather than correctness.
The simpler 8x8 shader and dispatch remain selected.

Rewriting radical-plane intersections as differences of ray-relative powers
is rejected. It removes the per-neighbour squared-distance division and passes
all plain, weighted, oriented, relight, and depth CPU/GPU oracles, but produces
no end-to-end speedup. More importantly, the altered f32 order changes a few
near-tie cell exits: the reconstructed clouds lose one Bonsai and six Room
particles, and both held-out means fall by 0.02 dB. Traversal topology needs a
decision-level quality gate in addition to approximate pixel parity.

Packing alpha and peak weight as UNORM16 while retaining f32 world depth is not
selected either. The 8-byte target improves Room from 2.56 to 2.52 seconds but
regresses Bonsai from 1.77 to 1.78, and the tiny confidence error still changes
one Bonsai refinement choice and three Room fusion cells. Rounded held-out
scores are neutral, but a non-joint bandwidth result does not justify changing
the reconstruction topology; the 16-byte rgba32float target remains canonical.

Observation gathering now processes contiguous camera chunks with at most
eight scoped workers and merges them in original view order. The visibility
and sample decisions are independent per photograph, while the ordered merge
keeps downstream floating-point reductions exact. The pass falls from 0.3 to
0.1 seconds on both real scenes; complete reconstruction improves 1.98 to 1.79
seconds on Bonsai and 2.74 to 2.54 on Room. Scene and environment files are
byte-identical, and the fixed synthetic observation pass falls from 9 to 2 ms.

Current result (2026-08-03): GPU timestamps isolate the training step, not path
recording or CPU work, as the bottleneck. At 2,048 pixels and 256 path slots, 28
scalar embedding-gradient scatters (density plus degree-2 RGB SH) spent about
80 ms per update atomically adding mostly zero padding. Meganeura `59f15ac`
skips exact-zero atomic sources, reducing that group to 5.1--6.5 ms and the
120-update GPU wait from 8.291 to 2.804 seconds. The full 1,200-update gate is
2.22x faster with stable quality. Ordinary gathers now lead the profile at
about 13 ms per update.

The follow-up coalesces the 27 individually named RGB/SH parameter tables only
inside the graph, preserving checkpoint and learning-rate semantics. One wide
gather/scatter replaces the scalar pairs: the graph falls from 391 to 348 GPU
passes and 24.5--25.5 to 11.6--11.7 ms per update. Full training is 1.84x
faster (29.51 to 16.00--16.05 seconds), with three quality repeats and a Bonsai
SH-2 fresh-PLY smoke inside the acceptance gate. Peak synthetic memory is
363.2 MB and the 3.37 GB full-workspace test scope records no memory events.
At the current production shape (200K cells, SH-3, 4,096 rays), the fair
current-Meganeura control falls from 506 to 421 timed passes and 62.34 to 26.62
ms per step (2.34x); cgroup peak falls 1.510 to 1.436 GB. The packed graph does
raise planned device-local memory from 495 to 962 MB. A per-channel layout
reduces that to 849 MB, but one of two full quality repeats develops the same
bad material/depth tail as a rejected traversal-cap arm, so RGB remains one
atomic dispatch. A future native packed-parameter representation should remove
the compatibility concat buffers before scaling far beyond 200K cells.

The selected path-mask cleanup then removes two redundant full-path
multiplications: the recorded interval is already masked, so padding has zero
raw optical depth, alpha, and weight. A physical-GPU regression proves exact
zero loss and parameter gradients even when padded weighted-path intervals and
Jacobians contain arbitrary nonzero values. The graph falls from 348 to 346
passes and ten profiled steps improve from a median 11.60 to 11.10 ms. Three
matched full runs improve median training time from 15.907 to 15.434 seconds
(3.1%), with overlapping held-pose, geometry, and held-light ranges. The full
workspace GPU test scope peaks at 3.08 GB and records no memory event.

Re-tuning after the weighted-reference correction does not justify a default
change. A 0.02 position-rate ratio improves all four synthetic held-pose means
and reduces held-depth RMSE, but its material/depth tail still varies; 0.03 and
0.04 lose the worst held pose immediately. On full Bonsai, 0.02 leads 0.01 at
step 2,000 by 0.90 dB on the selected eight and 0.24 dB over all 37 views. At
the losslessly continued step-4,000 checkpoint, it is only +0.06 dB selected
and is -0.23/-0.14 dB train/all-37. Both sweeps are unweighted and therefore
gate only the shared position-rate default, which remains 0.01.

Larger image batches do not stabilize that geometry either. Three matched
1,200-update synthetic repeats at 8,192 rather than 2,048 rays make training
4.66x slower and reduce ray throughput by 14.1%. Mean held-pose radiance moves
only +0.03 dB while its worst-view mean loses 0.32 dB. Position RMSE rises from
0.619 to 0.631 world units and normal RMSE from 66.10 to 67.71 degrees. The
+0.19 dB mean relighting change follows a 0.73-point coverage increase, but its
worst-view mean still loses 0.09 dB. Training-depth outliers remain in both
arms. The production default therefore stays at 2,048 rays; batch size is not
a substitute for the shared surface objective.

The first control artifact also localizes that objective's input failure.
Freshly rescored training depths have a 2.91-world-unit median per-view RMSE in
every camera and a rare 278-unit tail, rather than one defective pose. Normals
formed from adjacent depth pixels are already 60.08 degrees RMSE before
fusion. Requiring four freshly projected supporting views improves the final
normal RMSE only from 67.47 to 64.54 degrees and position RMSE from 0.681 to
0.569. Shared-view refinement must therefore improve particle centres without
treating the current depth-gradient normals as truth; normals should be derived
after the surface signal is coherent.

Two normal-independent 3D center searches fail that gate and are removed. A
camera-basis search using robust cross-view center colour passes an analytic
plane whose supplied normals are 70 degrees wrong, but changes synthetic
position RMSE from 0.6280 to 0.6281 and loses 0.07 dB under the held light. A
50% acceptance floor suppresses moves without improving geometry. Scoring the
same candidates by median refreshed source-depth residual moves 573 of 642
supported particles, but worsens position RMSE to 0.6297, normal RMSE from
66.38 to 66.83 degrees, coverage from 53.8% to 53.3%, and held-light PSNR from
18.53 to 18.49 dB. The synthetic truth gate therefore rejects both before a
real-scene sweep: inconsistent per-view modes are not a shared surface target
merely because the optimizer is allowed to move in three dimensions.

Retaining multiple pre-fusion density modes isolates the missing information
more sharply, but still does not supply the missing responsibility. On the
fixed synthetic foam, a truth oracle choosing among the strongest one, two,
three, or four ray segments reduces training-depth RMSE from 2.7468 to 1.3424,
1.0550, and 0.9092 world units. A top-three GPU extractor agrees with its CPU
oracle, and robust cross-view depth plus colour improves position RMSE in the
fixed model and three independent replicas (for example 0.6280 to 0.5977 in
the fixed gate). Real captures reject it: the unrestricted rule moves Bonsai
14.50/11.68 to 14.30/11.32 dB held-out mean/worst while Room improves
14.07/13.15 to 14.34/13.50. Seven stricter colour/view-support settings have no
joint winner, and changing only unsupported primaries regresses both. All
runtime and CLI code is removed. The modes prove the geometry has not vanished;
the next objective must learn which modes share one surface instead of asking
the same per-view density field and RGB observations to vote heuristically.

A differentiable shared-ray prior reaches the same conclusion from inside the
trainer. Penalizing a contributing site's angular distance from mixed-view
camera rays reduces the synthetic position error by about 10% and adds roughly
0.5 dB under the unseen light, but it aligns every opacity layer that carries
responsibility. On Bonsai the early 2K gain reverses by 6K, and the extracted
surface falls from 13.73 to 13.17 dB held out and from 14.54 to 13.43 dB where
hit. Decaying the loss before densification and ignoring contributions below
the extractor's 0.05 confidence floor do not fix it; all experimental API and
graph code is removed. A high-temperature soft argmax over each ray's weights
then tests the narrower dominant-mode hypothesis. It preserves small Bonsai
radiance gains through 4K and nearly recovers the control surface, but still
loses every held-surface metric at 6K: 13.65 versus 13.73 dB mean, 11.10 versus
11.19 worst, and 93.1% versus 93.7% coverage. Per-ray responsibility selection
is closed as a family. The next geometry objective needs explicit cell-level
cross-view consensus rather than another path-local weighting rule.

Cell identity alone is not enough for that consensus. A post-training screen
grouped confident dominant-mode rays by owning cell and retained only the 129
cells seen from at least two of the six synthetic training cameras. Moving a
cell toward either the mean of its per-camera mode points or the least-squares
intersection of its per-camera ray bundles fails even at 1% of a one-cell
clamped step. The centroid changes surface position RMSE from 0.6183 to 0.6210
and held-light PSNR from 18.34 to 18.32 dB; ray intersection changes them to
0.6233 and 18.22 dB. Both immediately enlarge the depth tail, and practical
steps collapse the reconstruction. A Voronoi cell owns a volume of rays and
depths, not one corresponding surface point. The implementation is removed.
The next objective must first define a local surface coordinate inside each
cell—such as the existing oriented plane/detail sites—and aggregate
cross-camera responsibility there, rather than triangulating the cell center.

The weighted linearization no longer gathers three reference positions and
three reference radii for every path slot. The recorder emits the raw interval,
its Jacobians, and a ray-relative reference tangent; the graph evaluates the
numerically stable `dt_ref + tangent_actual - tangent_ref`. A translated-world
physical-GPU oracle rejected the simpler absolute affine intercept after an
8.76e-4 interval error and now passes below 5e-4. On the matched 2,000-point
weighted Bonsai subset, the selected graph falls from 366 to 362 passes and
from 15 to 10 large embeddings; median profiled GPU time falls from 16.82 to
14.73 ms (12.4%). A 1,200-step run with 12 geometry rebuilds takes 21.60 rather
than 23.97 seconds and exactly retains 11.91/11.64 dB train/held PSNR, including
both 11.63/11.65 dB held frames. Its cgroup peaks at 489 MB with no pressure,
swap, or OOM event. The default-parallel full workspace test passed but peaked
at 5.90 GB, too close to the 6 GiB safety cap; the identical
`--workspace --all-targets -- --test-threads=1` gate passed at 301 MB once
warmed. A later rebuild showed that serial test threads do not serialize
Cargo's compile/link jobs: that gate peaked at 5.52 GB during its first eight
seconds, then passed without pressure or swap. Use `CARGO_BUILD_JOBS=1` as well
as `--test-threads=1` for cold local gates; the corresponding all-target clippy
gate peaked at 870 MB.

Sparse multi-view PowerFoam batches now use a camera-independent software BVH
over support spheres. This is a point-cloud BLAS, not a polygonal or hardware
ray-tracing proxy: one 64-lane workgroup expands six balanced tree levels,
traverses disjoint subtrees, and retains the exact sphere, overflow, clipping,
and depth/index ordering rules. On the 200K-site Room checkpoint, median gather
time falls from 2.85 to 1.07 ms and the complete recorder from 17.46 to 15.63
ms (10.5%). Bonsai falls from 2.86 to 1.14 ms in gather and from 10.90 to 9.31
ms overall (14.6%). The 12.8 MB hierarchy adds about 43 ms to a Room training
resource build and pays back within roughly 25 updates. Only the geometry-only
`new_path_recording` cloud builds it; ordinary renderer, depth, splat, and scene
clouds bind one unread 32-byte node. Eight order-balanced 200K-point startup
probes consequently fall from a 0.69 to 0.65 second median and shed 12,000 KiB
of median process RSS. At 2K and 10K sites its fixed workgroup cost is
0.05--0.06 ms, so clouds below the measured 32K crossover retain the exhaustive
kernel and do not allocate the tree. A translated 32K frontier oracle matches
the complete CPU path and all sixteen physical-GPU path tests pass; Room and
Bonsai candidate/path maxima are unchanged. Profile scopes peak at 1.78 GB with
no pressure, swap, or OOM event.

Global path caps of 128 and 192 are rejected: 128 gives no meaningful speed
advantage over packing at 256 and loses the held-view tail, while 192 produces
6.80 world-unit training-depth RMSE and an unstable material fit. An earlier
generic packed `SumInner` engine kernel is bit-exact at its output, but a Blade
graph rewrite to use it changes the atomic schedule. Repeating that direct
rewrite on the current graph reduces median graph time by 29.0%, yet doubles
median training-depth RMSE from 3.3444 to 6.4169 and loses 0.09 dB on the mean
worst held pose. The source rewrite is removed.

The accepted follow-up keeps MatMul and MatMulBT in the logical graph.
Meganeura `0fdc696` recognizes a narrow f32 all-ones column only while lowering
the physical plan: forward becomes the scalar-order packed reduction and the
input gradient becomes an exact row broadcast. A 513x3 physical-GPU test
matches generic MatMul/MatMulBT forward values and parameter gradients
bit-for-bit. Three production-shape profiles reduce median graph time from
11.6825 to 9.2265 ms (21.0%). Eighteen order-balanced synthetic runs reduce
median training time from 11.1205 to 9.3645 seconds (15.8%); median held-depth
RMSE is 2.1246 versus 2.1255, surface position is 0.6214 versus 0.6225, and
held-light PSNR is 18.43 versus 18.47 dB. One candidate depth map contains a
single extreme ray, but its p99, extracted geometry, radiance, and relighting
remain stable, while large surface-tail events occur at nearly the same rate
in both arms (7/18 versus 8/18). A two-by-two late Bonsai continuation cuts GPU
wait by 11.4% with identical selected-view PSNR to four decimals and all-view
means within 0.0001 dB. The engine work is merged through Meganeura `f0d9d50`.
Its merge changed the engine's Blade/Naga pins back past the reconstruction
validation fixes, which would give Blade Volume two incompatible
`blade-graphics` types and restore invalid workgroup SPIR-V. Meganeura branch
`fix/blade-volume-dependencies` (`9f8fa54`) restores the exact
`bd74bdc`/`cefd48f` pair used here; Meganeura's full all-target suite passes
under the 6 GiB cgroup. A local override of that commit also passes Blade
Volume's complete all-target suite and warnings-denied clippy. A fresh
synthetic run reaches 24.11 dB at the held pose and 18.48 dB under the held
light with no truncated rays; a 20.4k-to-20.8k Bonsai continuation reaches
24.4900 dB on the selected eight and 24.9420 dB on all 37 held views. Peak host
memory is 5.2 GiB for the exhaustive suite and 1.4 GiB for Bonsai, with no
cgroup or GPU failure. Blade Volume deliberately stays at `f82d0b6` until that
small prerequisite lands; then the local override can become the checked
revision without changing results. Other performance targets remain native
checkpoint-compatible packed parameters, dead input-gradient work, and compact
active path storage, still at the unchanged safe traversal extent.

Unweighted multi-view path recording now binds all camera slices in one
compute pass. Each slice already owns a disjoint output range, so the old
per-camera pass boundary added fifteen barriers at the production 16-view
batch size without ordering useful data. Projected PowerFoam remains on its
per-camera sequence because it reuses projection and binning scratch. A new
physical-GPU regression compares cells, next cells, intervals, masks, and path
status byte-for-byte with the old separate-pass sequence. Three production
profiles reduce the median recorder from 17 passes and 11.48 ms to 2 passes
and 10.31 ms (10.2%); the differentiable graph remains neutral. A two-by-two
400-step Bonsai continuation reduces median training time from 25.261 to
24.006 seconds (5.0%) and GPU wait from 10.268 to 9.582 seconds (6.7%). Mean
selected-view PSNR changes by +0.0008 dB and all-view PSNR by +0.0017 dB. The
benchmark scope peaks at 2.08 GB; the cold full-workspace physical-GPU gate
peaks at 5.58 GB. Neither records pressure, swap, OOM, or GPU events.

A 32-neighbour fused-cloud covariance prior was also tested against the 66°
normal failure. A sign-aligned 50% blend reduces synthetic normal RMSE to
62.08° and raises unseen-light PSNR by 0.09 dB without losing coverage. It does
not transfer: the worst held-out view falls by 0.24 dB on Bonsai and 0.31 dB on
Room as Euclidean neighbourhoods cross thin surfaces and depth layers. The
implementation is removed. Normal updates need shared-view correspondence or
surface-aware neighbourhoods, not another local point-cloud smoother.

Balancing fused normals by observing view was also rejected. Equal view votes
over-correct for dense close views; square-root confidence weights preserve the
particle set and improve synthetic normal RMSE from 66.42° to 66.04°, but lose
0.01 dB mean and 0.07 dB on the worst Bonsai held-out view. Room is neutral to
slightly positive. Pixel-count heuristics therefore are not a reliable normal
objective either; the next experiment should improve the shared positions and
derive orientation from geometry that is consistent across views.

A direct photometric tangent-plane search was rejected as well. Two rings of
normal candidates repair a 25° error on an analytic textured plane, but rotate
4,723 of 5,445 scorable Bonsai particles by 24° on average and reduce held-out
PSNR from 14.50 to 14.42 dB (11.68 to 11.51 worst). Room falls from 14.37 to
14.34 dB. Normalized training-patch agreement therefore confounds orientation
with local depth layers and occlusion; normal/covariance work remains gated on
a shared surface objective with refreshed visibility.

Changing the tangent-patch scale does not expand that objective reliably.
Global radius factors from 0.25 to 2.0 have no joint synthetic/Bonsai/Room
winner. A conservative quarter-radius retry only for previously unscored
surfels adds 369 scored candidates on Bonsai and 203 on Room, but three
independent synthetic controls remain byte-identical. Exact PNG scoring changes
the held-out means by only +0.0021 and +0.0076 dB respectively. The retry and
its CLI knobs are removed: this is measurement noise, not enough evidence for
another reconstruction mode.

Dominant-cell ray intersections are not the missing correspondence signal.
An experimental depth channel grouped training rays by dominant foam cell,
nearest oriented-detail site, and camera, then least-squares intersected the
camera bundles and moved only that site's tangent offset and height. At 128²,
three-view support covered 12,103 Bonsai sites but only 28 Room sites. Bounded
Bonsai steps of 5% and 20% changed held-out PSNR from 16.8278 to only 16.8285
and 16.8293 dB; a full step fell to 16.7765 dB. Allowing two-view bundles did
not improve the 20% result. This construction can only recover the model's own
projection consistency: it assigns rays using the model render and never
consults image appearance. The prototype is removed. Future geometry work must
introduce an observation-derived correspondence or residual, rather than
triangulating identifiers emitted by the current model.

A calibrated image-derived plane sweep tests that requirement directly. For
each confident source pixel it unprojects a fronto-parallel patch at 17 depth
hypotheses, compares normalized radiance patches in the other five training
photographs, and optionally requires the independently refined target depths
to reproject consistently. It repairs an analytically displaced plane without
using a reconstructed normal. On the fixed synthetic foam, however, the most
geometrically positive arm changes position/normal RMSE from 0.6280/66.38° to
0.6188/64.99° while reducing held-light PSNR from 18.53 to 18.41 dB. A wider
search falls to 18.34 dB. Tight one-percent depth consistency recovers only
18.45 dB, and requiring three supporting photographs returns geometry nearly
to the control at 0.6274/66.38° and 18.50 dB. An analogous particle-level
camera-ray search has the same trade-off. Both implementations are removed.
Normalized local patch agreement supplies real observation evidence, but its
minimum is still not the common PBR surface in occluded and view-dependent
regions. The next objective needs learned correspondence descriptors or a
joint rendered-surface objective, not a larger hand-designed plane sweep.

Mask-only initialization also needs to preserve a support volume. Filling the
strict intersection of all six training silhouettes with all 2,048 sites makes
almost every site opaque and drops held-pose quality from 24.11 to 22.45 dB;
tenfold lower initial density still reaches only 22.57 dB and worsens surface
position RMSE from 0.6209 to 0.7495. That prototype is removed. The masks are
valuable supervision, but a filled visual hull is not a surface representation.

Adaptive point allocation does improve the same fixed-size cloud. Four paired
1,800-update runs compare a fixed 2,048-site lattice with a 1,024-site support
lattice grown to 2,048 by four existing gradient-directed densification rounds.
The staged median improves held-pose radiance from 24.61 to 25.06 dB, surface
position RMSE from 0.6302 to 0.5945, normal RMSE from 66.78° to 66.65°, coverage
from 53.8% to 55.6%, and held-light PBR from 18.10 to 19.10 dB. Training time
rises 15.7%, from 16.83 to 19.47 seconds. The synthetic benchmark now exposes
the initial and final budgets separately and defaults to the staged schedule;
the production trainer already uses the same tested densification machinery.
This is the first current gate to improve static radiance, truth geometry, and
relighting together, but the remaining ~66° normal error keeps the next major
milestone unchanged: learn a shared local surface coordinate and optimize it
against a rendered-surface objective.
