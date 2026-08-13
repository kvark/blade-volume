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

The 2026-08-12 gate also removes the last synthetic-only shortcut from the
default: `sun-east` is now recovered from those RGB observations rather than
handed to the material fit. The first unknown-light sweep selected twelve
shared diffuse materials and rejected 64--128 because one repeat collapsed
below 14.7 dB even as the other four improved. The later five-cloud staged
gate selected six; it improves every held-light mean and tail over two and is
now the default. A fresh default run reaches 25.02 dB at unseen poses and
19.23 dB (18.92 worst) under the unseen `studio` light at 55.8% coverage. Its
recovered capture-light shape is still 62.7% RMS from truth after the
unidentifiable colour/scale gauge, so this is an honest end-to-end result, not
a claim that illumination is solved.
`--true-light` retains the former ceiling, and `--brightest-albedo` exposes the
otherwise unknowable scale prior. The complete latest asset is
`target/audit-runs/unknown-light-default-v1/{model.ply,scene.rply,scene.f32}`;
the `.f32` sidecar is the recovered environment.

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

That surface coordinate cannot be introduced on the initial camera-filled
volume lattice. A staged 1,024→2,048-site synthetic control initialized
PowerFoam reference radii and camera-facing half-cell planes before training;
held-view radiance falls from roughly 25 dB on the selected unweighted foam to
20.52 dB, held-depth recall is zero, and no Gaussian survives multi-view
fusion. Removing the oriented clipping still reaches only 20.96 dB, with less
than one percent depth recall and again no fused surface. Both benchmark
switches are removed. PowerFoam surface/detail sites remain appropriate after
a surface correspondence exists, but weighted splats over a volume lattice do
not create that correspondence.

The unoriented isolation exposed and fixed an independent weighted-training
bug. Radius optimization requests full position/radius path Jacobians, whose
buffer layout also contains an unused surface-normal stream. Session setup was
binding that stream even when the graph and model had no oriented normals, so
Meganeura correctly returned `UnknownSlot`. Binding now follows the model/graph
semantic instead of buffer capacity, and a physical-GPU regression trains an
unoriented weighted cloud for two updates with a per-step rebuild cadence. No
dummy normal parameter, shader, operation, or entry variant is added.

Starting weighted surface semantics only after a Gaussian surface exists is a
different result. A fixed-center continuation converts the selected 2,179-site
Gaussian reconstruction to oriented PowerFoam, learns radii, normals, density,
and appearance for 1,800 updates, then converts the learned sites directly
back to surface Gaussians. With the independently rendered foreground masks it
improves held-view light-field PSNR from 24.11 to 24.19 dB, holds position RMSE
exactly at 0.6204, changes pre/final normal RMSE from 53.66/53.95° to
53.45/53.85°, and improves held-light PBR mean/worst from 19.49/19.25 to
19.80/19.52 dB while coverage rises from 54.0% to 54.2%. This is the first
joint winner for a surface-stage PowerFoam continuation.

The evidence source is decisive. RGB-only continuation reaches 23.66 dB
held-view light-field PSNR and self-distilled Gaussian alpha reaches only
21.81 dB; neither is selected. The latter also regresses held-light PBR to
19.59/19.32 dB and 53.9% coverage. Synthetic truth masks therefore remain a
gate, not an image-only default. Production capture and initial light-field
training now accept an optional mask directory mirroring the COLMAP image
paths and rectify masks with the same camera. Capture applies them to foam-
depth fusion, radiance observations, and patch refinement; light-field
training supervises opacity and composites RGB over its configured background.
The production fixed-surface continuation is now explicit
and enabled only when those independent masks are supplied. It holds centers
and offsets fixed, reuses the existing differentiable PowerFoam path to learn
density, degree-two SH, radii, and normals, updates the Gaussian surface, and
optionally writes the trained static light field. It adds no shader, graph
operation, binding, pipeline, `ShaderEntry`, or runtime representation variant.

The production implementation repeats the six-view/1,800-update gate at a
24.17 dB held-view light-field mean (23.40/24.94 dB), within 0.02 dB of the
prototype. After the unchanged photometric-normal, material, and rendered-
normal stages it reaches 19.84/19.57 dB held-light PBR at 54.2% coverage,
slightly above the prototype's 19.80/19.52 dB. Centers remain bit-exact. The
22.3-second isolated training run peaks at 222,801,920 bytes with zero swap,
pressure, OOM, throttle, Xid, or GPU fault.
The complete all-target workspace gate passes on the physical RTX 5070 in
93.5 seconds at a 2,632,908,800-byte cgroup peak, with zero swap, pressure,
OOM, throttle, Xid, or GPU-fault event. Workspace clippy also passes at a
654,110,720-byte peak. This is a clean post-750 W PSU stability sample.

Learning those established centers more gently is still not selected. A
`0.001` position/base learning-rate ratio was paired with the fixed-center
continuation on all five synthetic clouds under the simplified Meganeura
runtime. Held-light mean/worst deltas were +0.04/+0.09, +0.09/+0.12,
-0.15/-0.10, +0.05/+0.01, and +0.06/+0.00 dB. Position RMSE regressed by
0.0001--0.0002 world units on every cloud and normal error was mixed. The
lower final training loss therefore reflects appearance/topology overfit, not
a better shared surface. The experimental rate is removed; production keeps
the centers fixed.

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
Meganeura recognizes a narrow f32 all-ones column only while lowering
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
means within 0.0001 dB. The shader-modifier cleanup is now merged through
Meganeura `19a1f00`.

The final uprev also closes the validation issue that had looked machine
specific. Latest Blade had lost the reconstruction branch's workgroup SPIR-V,
device-address allocation, and descriptor-array pool fixes; Meganeura's wider
atomic suite additionally showed that enabling the Vulkan memory model for
cooperative matrices must also enable device scope. Blade branch
`fix/reconstruction-vulkan-validation-2026` (`200188b`) rebases the three old
fixes and adds the device-scope fix. Meganeura branch
`fix/blade-volume-dependencies-v2` (`84a8a45`) contains merged main plus
matching Blade, macro, and Naga pins. Against those exact remote commits, Blade
Volume's complete all-target suite has zero Vulkan validation messages, peaks
at 5.37 GiB with no swap/OOM/GPU fault, and warnings-denied clippy passes. A
fresh staged reconstruction trains 1,800 updates in 16.44 seconds, reaches
25.00 dB on held
poses, 0.5853 position RMSE, and 19.07 dB under the unseen light at 55.6%
coverage. The validation-uprev control artifacts are
`target/audit-runs/dependency-uprev-v2/{model.ply,scene.rply}`. Other
performance targets remain native checkpoint-compatible packed parameters,
dead input-gradient work, and compact active path storage, still at the
unchanged safe traversal extent.

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

Applying those masks after surface extraction confirms both halves of that
conclusion. Strictly removing particles that contradict any training
silhouette improves normal error and every training/held-light score on five
clouds. Even after a small support correction, however, position RMSE regresses
on four clouds and coverage falls on four. Moving the contradictory particles
toward the nearest silhouette instead of deleting them improves the newest
position RMSE from 0.5853 to 0.5837, but slightly regresses held-light mean,
worst view, and coverage. Both prototypes are removed. Masks can say that a
point is impossible; away from a silhouette boundary they provide no depth or
surface coordinate for the replacement.

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

The same staged schedule does not improve further by changing its initial
budget. Starting from 768 sites gives excellent geometry in two repeats
(0.5595--0.5793 position and 63.4--65.0° normal RMSE), but its unseen-light
tail is lower and variable. Starts of 1,280 and 1,536 are worse. The 1,024-site
default remains the joint winner. On its fixed output clouds, increasing the
shared diffuse palette from two to six materials is independently positive:
five clouds gain 0.06--0.14 dB under the unseen light and 0.04--0.09 dB on
every worst view, with identical geometry and coverage. Six is now the
synthetic default; larger palettes do not improve the difficult tail.

When repeat captures under controlled known lights are available, the normal
part is now directly solvable. `refine_normals_known_lights` searches 512
directions per Gaussian and eliminates diffuse albedo analytically for each
candidate, so illumination cannot win by being baked into a free material.
The synthetic `--photometric-normals` gate uses sun-east, sun-west, sky-dome,
and uniform captures while keeping studio entirely held out. Correct normals
make discs more edge-on; raising only this opt-in path's minimum support from
1.4 to 1.6 fused-cell radii closes the resulting gaps. Across the fixed
reference and three staged clouds, normal RMSE moves from 65.98–69.43° to
53.96–57.56°, held-light PSNR gains 0.37–0.68 dB, and coverage gains 0.6–1.0
points. The solve takes 0.04–0.05 seconds after observations are gathered.
This is a real reusable multi-light result, but remains opt-in: ordinary phone
video supplies one illumination, so its next milestone is still a joint
rendered-surface objective or an observation-trained correspondence descriptor.

Sharing a small diffuse material table does not by itself make that one-light
case identifiable. An alternating experiment fitted two shared materials from
sun-east and searched 512 normal directions, with studio kept entirely unseen.
At matched 1.6-cell support it improves normal RMSE by 0.65--2.00 degrees and
mean held-light PSNR by 0.21--0.46 dB, but loses 1.5--1.9 coverage points and
regresses the worst unseen-light view on two of four clouds. Rescaling every
Gaussian to preserve its projected area restores coverage, but then every
worst-view score and three of four mean unseen-light scores regress. The
prototype is removed. The next single-light step needs an observation-derived
cross-view surface constraint; another per-particle shading prior will merely
choose a point on the same normal/albedo ambiguity.

That cross-view objective now exists without introducing a second geometry
representation. An opt-in final pass moves one observed Gaussian at a time by
one quarter-radius along its current normal, renders every training camera
through the production PBR tracer, and keeps a direction only when it reduces
the joint sRGB error. Materials and the recovered capture light stay fixed;
held cameras and the held studio light never enter the solve. On five fixed
synthetic clouds, a bounded 300-particle pass improves every held-light mean
and worst view by 0.03--0.08 dB. A full 1,309-particle pass improves position
RMSE from 0.5842 to 0.5824 and held-light PSNR from 19.23/18.92 to
19.59/19.28 dB, with normal RMSE essentially unchanged at 67.3 degrees and
coverage down only 0.2 points. This is an output-correct surface objective,
not yet a normal solver.

The objective also transfers to real captures. Moving 3,000 of roughly 30,000
particles raises held-view Bonsai by 0.02 dB mean / 0.03 dB worst and Room by
0.07/0.05 dB, with at most 0.1 coverage point of change. A complete Bonsai
pass raises the held mean by 0.18 dB but loses 0.01 dB on its single worst view
and 0.1 coverage point. It therefore remains explicit through
`--render-refine`, not the default. A complete Room pass is unambiguously
positive at +0.16 dB mean, +0.13 dB worst, and +0.1 coverage point. The next
shape coordinate is also identifiable: a fixed 20% radius search adds another
0.04/0.03 dB on Bonsai and 0.21/0.22 dB on Room at 3,000 particles, restoring
or increasing coverage. It is exposed by `--render-refine-radii`. Normal
coordinates are not: four 10-degree candidates buy only 0.03--0.05 dB and
about 0.1 degree of truth normal error on synthetic clouds, so that prototype
is removed. The next algorithmic step is a tail/coverage-aware batched or
differentiable version that makes position and support cheaper; normal work
still needs stronger evidence than single-light PBR error.

Three cheap shortcuts were measured and removed. Adding the worst
training-camera MSE to the objective does not protect unseen poses. Scoring a
particle on only the cameras that observed it is 43% faster, but loses up to
0.03 dB on the synthetic held-view tail; verifying its winner against all
cameras restores quality and all of the cost. Reusing Blade's host-side TLAS
instance buffer is byte-exact but measures 5.406 seconds in both arms over ten
alternating pairs. The cost is the synchronized TLAS rebuild itself, not the
allocation. Further throughput work should therefore batch or differentiate
geometry proposals so one rebuild evaluates more than one scalar coordinate.

Screen-impact ordering does not make a bounded pass safer. Ranking particles
by radius squared times summed view-facing improves four of five position-only
synthetic means by 0.08--0.10 dB, but loses 0.1--0.3 coverage points everywhere
and drops the fixed cloud's worst view by 0.01 dB. Adding radius updates raises
all five held-light means by 0.05--0.39 dB while losing 0.6--1.2 coverage
points. The earlier real-scene priority gate confirms the trade: Bonsai gains
0.07 dB mean but loses 0.05 dB worst and 0.2 coverage points, while Room gains
only 0.02/0.02 dB. The prototype is removed. Any future bounded selection must
preserve spatial/support diversity rather than concentrating exclusively on
the largest visible discs.

Dividing the cloud order into equal strata and choosing one high-impact
particle per stratum reduces the synthetic support loss, but still fails the
real tail. Position-only Bonsai is mean-neutral and loses 0.04 dB worst;
adding radius search gains 0.04 dB mean but loses 0.12 dB worst and 0.2
coverage points. Room gains only 0.01/0.01 dB and 0.03/0.01 dB respectively.
The stratified selector is removed too.

A self-supervised alpha anchor also fails to make impact ordering safe. Weight
0.05 leaves 0.6--1.1 points of radius-path coverage loss. At 0.5 most support
drift closes, but the fixed position gate loses 0.05 dB mean/worst, the fifth
loses 0.02/0.02 dB, and the fixed radius gate loses 0.10/0.04 dB while still
dropping 0.3 coverage points. The anchor needs no masks or extra render, but it
only preserves the initializer's silhouette; it adds no evidence about missing
geometry. The implementation is removed.

That batching direction now has a selected localized form. Each paired
perturbation still renders every training view through the production tracer,
but every Gaussian correlates its deterministic sign with the error difference
inside its own projected footprint instead of inheriting one whole-frame
direction. The resulting full-cloud proposal and both original global
directions are scored by complete renders with a 0.001 anchor prior. This adds
only a CPU-side error field and selection pass: no shader, op, binding,
pipeline, or acceleration-structure variant.

Eight localized rounds beat the earlier 64 global rounds on all five synthetic
held-light means and tails: 18.40/18.37, 19.31/19.09, 19.33/18.80,
18.79/18.75, and 18.73/18.56 dB versus 18.30/18.29, 19.27/19.02,
19.28/18.74, 18.70/18.65, and 18.62/18.44. Where-hit PSNR improves on all
five. Position RMSE improves on four and changes by only 0.0002 on the fifth.
The pass takes 0.227--0.234 seconds instead of 0.460--0.475. Matched current
Bonsai and Room controls preserve or improve held coverage while raising both
mean and tail: Bonsai 14.23/11.85 to 14.30/11.92 dB and Room 14.18/13.41 to
14.24/13.47 dB, in 2.6--2.7 seconds. Longer schedules continue improving RGB:
64 rounds reach 14.42/11.98 on Bonsai and 14.54/13.79 dB on Room. They are not
the conservative choice because Bonsai loses 0.1 coverage point from 16 rounds
onward and several synthetic clouds lose up to 0.4 points. The recommended
coverage-preserving opt-in schedule is therefore `--render-refine-rounds 8`;
larger values remain available as an explicit PSNR-first choice, and none
becomes a default.

The localized signal does not make surfel-normal optimization strong enough to
retain. A prototype applies deterministic one-degree tangent perturbations and
uses the same per-footprint antithetic selection plus complete-render
acceptance. Eight normal rounds improve all five synthetic held-light
mean/tails by 0.05--0.07 dB and truth-normal RMSE by only 0.03--0.10 degrees,
while four clouds lose 0.1 coverage point. Room gains 0.05/0.06 dB, but Bonsai
is unchanged at 14.30/11.92 dB, loses 0.1 coverage point, and total refinement
time grows from 2.7 to 5.2 seconds. Two to four rounds preserve real coverage
but buy only 0.00--0.03 dB and still move synthetic support. The prototype and
CLI are removed. A useful normal/covariance update needs multi-light or
geometric evidence, not more single-light rendered coordinates.

Localized radius updates do not solve the support trade either. Allowing both
directions for eight rounds raises synthetic held means by 0.42--0.49 dB and
Room by 0.18/0.23 dB, but contracts support by 0.2--0.5 points and drops the
Bonsai tail by 0.14 dB. Expansion-only updates restore support. Eight rounds
then lose covered-region quality on every synthetic cloud and regress the
fourth mean by 0.02 dB; Bonsai is exactly neutral while its refinement time
doubles. Capping expansion at two rounds leaves Bonsai neutral, Room only
+0.03/+0.02 dB, and still regresses the fourth synthetic mean by 0.01 dB.
Both variants are removed. Radius fitting needs observed foreground evidence,
not RGB error alone.

Making the localized screen region more geometrically literal is negative too.
An affine projected ellipse weighted by the production particle kernel
improves synthetic scores, but over-focuses the centre and loses 0.05 dB on
Bonsai and 0.03 dB on Room while making CPU selection 2.2--2.6 times slower.
Using only the ellipse's mathematically correct axis-aligned bounds avoids that
cost but dilutes the useful signal: four synthetic means regress, Bonsai loses
0.01 dB mean, and Room loses 0.01 dB tail. Both are removed. The selected
smaller rectangular patch is an empirical contextual estimator, not a claim
that it exactly rasterizes the Gaussian footprint.

Robust per-light aggregation does not improve the controlled-light normal
solver physically. Replacing each light's mean across camera views with a
channel median raises held-light mean PSNR by 0.05--0.28 dB, but loses 0.2--0.3
coverage points and worsens truth-normal RMSE on two of four clouds. A 1.62
support correction restores most coverage, then worsens truth normals on three
clouds and loses one tail. Dropping only the per-channel minimum and maximum is
less aggressive, but still loses one tail by 0.03 dB, reduces coverage on all
four clouds, and worsens two truth-normal scores. Both aggregators are removed;
they suppress real cross-view surface disagreement as though it were an
outlier, improving appearance without recovering orientation.

The retained rectangular localizer is now cheaper without changing that
selection rule. Each antithetic render pair builds one f64 summed-area table
of its per-pixel error difference; every projected rectangle then reads four
values instead of scanning its pixels again for every Gaussian. Three
alternating same-source pairs reduce Bonsai refinement from 2.7 to 2.2 seconds
(18.5%) and Room from 2.6 to a 2.3-second median (11.5%). All five synthetic
scenes and all twelve real pair outputs are byte-identical to the direct-scan
control. This is CPU-only bookkeeping and adds no shader, op, binding,
pipeline, or acceleration-structure variant.

The same batching does not generalize to support radii. A separate 64-round
radius phase improves all five synthetic held-light means and tails over exact
300-particle position-plus-radius descent in 1.60--1.64 rather than 7.07--7.28
seconds. It also improves Room by 0.09/0.06 dB mean/worst over paired positions,
but loses 0.11 dB on the Bonsai tail and 0.2 coverage points. Joint proposals
have the same failure; expansion-only proposals are neutral, and reducing the
radius phase to 8--16 rounds makes Room neutral or worse. The prototype is
removed. Batched radius updates need an objective that observes support loss,
not another schedule or radius prior.

That missing observation was tested directly after the runtime renderer began
returning accumulated cloud coverage in alpha. A binary synthetic foreground
MSE with weight 0.05 improved the batched-radius held-light mean/worst score on
three of five fixed clouds, was mixed on one, and regressed the fifth. The
fixed-cloud deltas versus RGB-only were -0.01/+0.03, +0.02/+0.03,
+0.03/+0.07, +0.02/+0.03, and -0.04/-0.01 dB (mean/worst). The available real
COLMAP benchmark captures do not supply that mask, so the term cannot repair
the earlier Bonsai failure. The foreground-loss API and batched-radius
prototype are therefore removed rather than adding another synthetic-only
control.

Coverage alpha does select a smaller performance change. Reconstruction
scoring now reads coverage from the same black-background render used for
PSNR instead of rendering every camera again against white. This halves score
dispatches and readbacks, removes a temporary coverage image, and adds no
shader or bind-group variant. The physical-GPU compositor oracle checks RGB
and alpha together; the fixed synthetic cloud retains exactly 53.6% reported
coverage after the change.

Refreshing observations and re-running the material/light decomposition after
an exact position-and-radius pass is not the missing joint update. Re-solving
both appearance and illumination improves two fixed clouds but drops a third
held-light score by 0.54 dB. Holding the recovered illumination fixed removes
that collapse, but the five-cloud deltas are only +0.04, +0.01, +0.03, -0.01,
and -0.02 dB. The prototype is removed. Geometry and appearance must share the
rendered objective while a proposal is selected; a post-hoc decomposition only
relabels the already chosen surface.

Two apparent correctness cleanups are also rejected by the full output gate.
Discarding the depth mode of an unbounded terminal RadFoam cell reduces extreme
training-depth RMSE (12.29 to 2.05 world units on one cloud), but those tail
samples almost never survive multi-view fusion and changing three retained
surfels loses 0.93 dB under the unseen light. Replacing index-strided palette
initialization with order-independent chromaticity quantiles is similarly
well-motivated but loses 0.06--1.16 dB on all five clouds. The latter confirms
that the small shared-material solve is poorly conditioned: a lower recovered
light RMS does not imply better unseen-light rendering. Both code changes are
removed; future decomposition work needs an explicit held-independent prior or
joint objective, not a cosmetically more stable initializer.

Reducing the recovered environment from 32x16 to 16x8 is effectively
byte-equivalent at the output gate; 8x4 is mixed by up to -0.05 dB and 4x2
regresses all five clouds. None changes the measured 0.02-second fit time.
The temporary resolution control is removed. The material/light ambiguity is
not caused by using too many environment texels: diffuse transport already
projects those texels onto a much smaller observable subspace.

Profiling appearance inside exact geometry descent is not yet a selected joint
optimizer. Re-running the six-way chromaticity clustering and one diffuse solve
for every position/radius proposal improves four five-cloud held-light gates,
but the fifth loses 0.03 dB mean and 0.05 dB worst while cost rises from
7.1--7.2 to 12.4--17.5 seconds. The failure is not just support coupling: its
position-only tail still loses 0.05 dB. Keeping every material assignment fixed
removes the relabelling discontinuity. A 300-position gate then changes the
five held-light mean/worst scores by +0.06/+0.05, +0.22/+0.21, +0.02/+0.02,
+0.02/-0.01, and +0.03/+0.01 dB, but position RMSE is neutral or worse on all
five and runtime nearly doubles. Adding radii again loses 0.03 dB on the fifth
tail. Both prototypes and their GPU material-update API are removed. A future
joint solve needs a batched or differentiable geometry gradient with a stable
appearance parameterization; thousands of full appearance solves per scalar
coordinate are too costly for the small, tail-unsafe gain.

The exact rendered-surface loop now records its particle upload, TLAS rebuild,
all training-camera renders, and readback into one command buffer. An
unrendered restoration is coalesced into the next scored state. Three
alternating 300-position/radius pairs reduce median refinement time from 7.194
to 6.989 seconds (2.9%); every pair improves, and final losses, scores, and both
asset hashes are identical. Simply omitting the old wait is invalid: Blade's
one-buffer encoder then resets an in-flight command buffer and produced a
reproducible NVIDIA Xid 31. The retained path submits once and waits once, and
its physical update/rebuild oracle plus the analytical recovery fixture pass.
It adds no shader, operation, bind-group, or pipeline variant.

Putting both signs of one exact coordinate into that submission is a smaller
but rejected extension. Three byte-identical position/radius runs have a 6.876
second median, only 1.6% below the selected 6.989 seconds. Supporting it needs
a second full-cloud upload slot, delayed instance-buffer retirement, and a
separate pair-render/readback path. That complexity is not justified by the
gain, so the prototype is removed. Further acceleration has to reduce TLAS
rebuild or render work itself rather than only one more host submission.

The retained loop now avoids that second rebuild when its first coordinate
direction already improves the complete training-render objective. Three
alternating same-source binary pairs reduce a 300-position synthetic median
from 3.487 to 2.594 seconds (25.6%), and position-plus-radius from 6.983 to
5.371 seconds (23.1%). Five synthetic clouds keep every reported held-light
mean, tail, and coverage score; truth position/normal changes stay within
0.0004 world units and 0.04 degrees. Position-only Bonsai falls 45.7→35.2
seconds and Room 50.7→36.6; with radii they fall 91.3→72.7 and 101.0→79.5.
Trying radius expansion before shrinkage matches the more common accepted
direction and lowers those radius passes again to 68.1 and 71.5 seconds, or
25--29% below the original loop. All reported real train/test quality and
coverage values are unchanged. This is CPU control flow only: it adds no
shader, operation, buffer, bind group, or pipeline variant.

Removing the shrink proposal entirely saves another 13--17% and gains 0.2--0.4
synthetic coverage points, but gives back most of the radius path's held-light
gain. The fixed score falls from 18.43/18.37 to 18.33/18.29 dB and the fifth
from 18.80/18.62 to 18.66/18.49. Expansion-only is removed; the explicit radius
option remains a final-quality rather than minimum-cost path.

Re-fitting normals from only the plane-sweep particles that move, their
32 nearest plane-consistent neighbours, and at least three shared visible
source views does not resolve the single-light ambiguity either. A 25% blend
repairs an analytical plane and leaves exact normals unchanged, but changes
five synthetic truth-normal RMSEs by only -0.03 to +0.02 degrees. Against a
same-source, same-command control, unseen-light mean/worst deltas are
+0.02/+0.02, +0.04/+0.02, -0.01/-0.01, -0.08/-0.06, and -0.01/-0.01 dB;
coverage is unchanged or +0.1 point. Even explicit shared-view and
tangent-plane filters therefore produce a mixed, tail-unsafe smoother without
meaningfully improving orientation. The implementation is removed before
running real scenes. Future normal updates need a held-independent physical
prior or repeat lighting, not a tighter Euclidean covariance neighbourhood.

A second half-sized rendered position search after each accepted quarter-radius
move is also removed. It improves all five synthetic unseen-light means and
tails by 0.01--0.03 dB, but makes truth-position error mixed and loses 0.1
coverage point on one cloud while raising a 300-particle pass from about 2.6
to 5.6 seconds. More decisively, Bonsai and Room held means, tails, and
coverage remain unchanged at report precision even though runtime rises
35.2→70.8 and 36.6→82.5 seconds. The lower training loss does not transfer;
the quarter-radius coordinate is already fine enough for this objective.

Reassigning shared materials after illumination correction does not stabilize
the unknown-light decomposition. An unconstrained halfway EM step gains as
much as +0.44/+0.37 dB on one synthetic mean/tail but loses -0.25/-0.20 on
another. Requiring a new material to be two or four times closer in corrected
chromaticity reduces label churn, yet the same cloud still loses -0.08/-0.06
dB and the fixed-cloud tail loses 0.02 dB. Recovered-light RMS is anti-correlated
with output quality in the decisive arms: it worsens on the large relighting
gain and improves on the loss. The implementation is removed. With noisy
orientation, neither the fitted environment nor corrected per-surfel albedo is
an independent signal for choosing the other.

The controlled-light path is now available to ordinary COLMAP captures, not
only synthetic diagnostics. `reconstruct --environment capture.f32` loads a
measured linear-radiance environment, uses its resolution for visibility, and
holds it fixed while fitting materials. Unknown-light recovery remains the
default. A reduced Bonsai end-to-end smoke reconstructs and scores normally,
and its written environment sidecar is byte-identical to the supplied file;
the command adds no alternate material or rendering implementation.

That path now also accepts repeated aligned captures under additional measured
lights through paired repeatable `--normal-images` and
`--normal-environment` arguments. It exposes the already selected synthetic
photometric-normal solve to COLMAP input before the primary material fit. A
new identifiability guard deduplicates irradiance maps with the same shape up
to per-channel scale: the initial identity smoke incorrectly moved 26 of 63
observed normals under two copies of one light, while the corrected path
reports no supported normals and emits a byte-identical scene. Distinct known
lights retain the analytical normal-recovery result and the earlier synthetic
0.37--0.68 dB unseen-light gain.

Two attempts to extend that result beyond directly constrained particles are
rejected by a four-cloud held-light gate. Replacing the squared photometric
residual with a 0.05 Huber loss changes truth-normal RMSE by only -0.16 to
+0.04 degrees and loses 0.03 dB on the fixed mean plus 0.07/0.09 dB on the
fourth mean/tail. Local propagation has a real geometric signal: fully copying
four-light normals from coherent tangent-plane neighbours improves normal RMSE
by 5.75--7.81 degrees. It also loses 0.6--0.8 coverage points and regresses
three held-light tails; increasing support to 1.7 cells restores coverage but
still loses 0.27--0.34 dB on difficult tails. A strict four-neighbour, 20%
blend preserves coverage and improves all four normal RMSEs by 0.67--0.94
degrees, but exact paired scoring still changes the fourth mean/tail by
+0.0020/-0.0068 dB. All implementations are removed. Unobserved normals need
an objective that also constrains their rendered support and appearance, not
spatial diffusion of otherwise accurate measurements.

Using those same measured lights to choose shared materials is not selected
either. A prototype first divides each observed RGB by the diffuse irradiance
predicted from all four lights, clusters six materials from the resulting
albedo chromaticities, then fits the actual materials against the primary
capture exactly as before. It improves two clouds by up to +0.02/+0.06 dB
mean/tail, but loses 0.04/0.02 dB on the fixed cloud and 0.06/0.05 dB on the
fourth. Its primary residual is likewise mixed. Even known illumination does
not make that corrected colour independent evidence while orientation,
visibility, and per-pixel Gaussian overlap remain wrong. The alternate fit API
and albedo-observation path are removed.

Correcting duplicated material evidence by projected-centre claims is also
too weak. Keeping only the nearest Gaussian centre per photograph pixel gains
0.01 dB on two held-light tails, loses 0.02 dB on the other two, and makes
3--6 additional particles unseen. Retaining all observations but dividing
their confidence by the claim count improves three clouds or leaves them
neutral, yet still loses 0.01 dB on the fourth tail; square-root weighting has
the same last regression and smaller gains. All variants are removed. A centre
claim is not the production renderer's depth-band Gaussian blend, so a valid
correction must use the actual per-pixel compositing weights rather than an
overlap count surrogate.

Using the actual production compositing fractions as confidence weights gets
closer but still does not define the right solve. The observation pass sums the
same truncated Gaussian coverage over the same two-radius depth band as WGSL,
then weights each centre sample by its share. Full fractions improve three of
four exact held-light gates, including +0.0242/+0.0240 dB on the fourth
mean/tail, but regress the second by 0.0036/0.0016 dB and 0.0114 dB where hit.
Square-root fractions clear the second and improve the third and fourth, but
move the fixed worst view by -0.0054 dB. Both implementations are removed.
The photograph contains the weighted *sum* of several unknown surfel colours;
changing independent-sample confidence cannot invert that equation. A future
joint material objective should render or solve those mixtures explicitly.

The first direct rendered-material objective is selected as an opt-in final
pass. `--render-refine-materials` coordinate descends only the RGB albedo of
the small shared palette, first with a 0.025 step and then 0.0125, against
complete production renders of every training camera. Geometry, light,
assignments, roughness, and specular response stay fixed. This differs from
the rejected appearance-inside-geometry prototype: one six-material pass
evaluates at most 72 proposals, rather than re-solving appearance for thousands
of geometry proposals.

Four exact known-light synthetic pairs improve held-light mean/worst PSNR by
0.120/0.107, 0.126/0.139, 0.152/0.134, and 0.154/0.131 dB; covered-pixel PSNR
improves by 0.143--0.195 dB with identical coverage. The transfer gate at
128² and every-eighth held views is also positive: Bonsai improves held
mean/worst/covered PSNR by 0.111/0.102/0.138 dB and Room by
0.190/0.137/0.212 dB, again with identical coverage. The six-material pass
takes 1.9--2.1 seconds on the real scenes. It remains explicit because it
optimizes training images and the absolute real-scene score is still low, but
the consistent unseen-view result is enough to retain it.

The implementation adds one in-place material-buffer update to the existing
relight tracer and an oracle proves its output is byte-identical to rebuilding
the tracer. It reuses the current upload storage and changes no WGSL, shader
entry, operation, binding, acceleration structure, or pipeline variant.

Extending the existing simultaneous rendered-position search from observed
particles to the entire cloud does not supply the missing geometry evidence.
Across five fixed synthetic clouds it lowers held-light mean PSNR by
0.14--0.54 dB, lowers every worst-view score by 0.10--0.43 dB, and usually
reduces coverage. Truth position error is neutral or worse. Particles that do
not affect a training render receive the accepted random direction of the
joint perturbation anyway, then enter held cameras unconstrained. The one-line
candidate-set prototype is removed without a real-scene run. A whole-render
objective still needs training-view support or an independent prior for every
coordinate it moves.

Repeated known-light captures still do not make roughness or metalness safe to
recover on the current geometry. Re-enabling the existing constrained lobe
solver after four-light photometric normals improves three unseen-light means
by only 0.03--0.06 dB, while one false-metal assignment collapses the fourth
from 19.80 to 16.43 dB despite a lower training residual. Evaluating the same
dielectric/metal hypotheses directly through complete production renders of
all measured lights avoids that collapse, but calls two to four of six shared
clusters metallic and regresses one held-light mean/tail by 0.26/0.12 dB.

Forbidding metalness makes every held-light mean and tail improve by
0.07--0.12 dB, but a nearest-truth material diagnostic shows the mechanism is
wrong: roughness MAE rises from 0.08--0.09 to 0.20--0.34 on all four clouds.
The optimizer is using gloss to absorb geometry and missing-bounce error. A
rough-dielectric prior strong enough to preserve truth accepts zero roughness
changes. Even reusing all four measured lights for diffuse-only rendered
refinement is consistently 0.02--0.03 dB below the selected primary-light
pass. Every prototype and CLI/API addition is removed. PBR recovery remains
blocked on a more accurate shared surface and forward light transport; image
score alone is explicitly not evidence of physical material recovery.

The production `reconstruct` CLI and the library `FitOptions` therefore default
specular fitting off. A nonzero `--specular-rounds` remains available for
calibrated experiments, but rough dielectrics are the honest default until the
missing evidence exists.

Feeding the successful measured-light normals back into the existing
multi-view position sweep does not produce that surface. A second sweep moves
97--148 particles and improves three held-light means by 0.02--0.12 dB, but
the fixed mean/tail lose 0.025/0.02 dB and one cloud loses 0.1 coverage point.
A 5% acceptance threshold gives away the strongest gain without fixing the
loss. Retaining moves only for particles whose normal actually changed makes
the fixed regression larger at 0.04/0.02 dB. Truth position changes by at most
0.0006, while truth normal error becomes 0.1--0.2 degrees worse in every arm.
The implementation is removed. A photometrically better shading normal is not
automatically the tangent plane that matches patches around a noisy layered
center; those variables need one joint surface correspondence objective.

The rendered diffuse pass should also remain in display space. Replacing its
sRGB objective with linear-radiance MSE is 34% faster, but direct nearest-truth
albedo RMSE improves on only two of four clouds and worsens on the other two.
Every exact held-light mean/tail is below the selected pass, including
0.07/0.09 dB losses on the second cloud. The decomposition itself continues
to fit physical quantities in linear radiance; this final coordinate pass is
explicitly selected for rendered output quality, where sRGB is the measured
objective.

Widening the depth-map derivative does not recover the missing surface either.
A symmetric left/right and up/down derivative improves truth position by
0.006--0.014 world units and normal RMSE by 0.6--1.6 degrees on four of five
fixed clouds, but requiring all four neighbours removes 13--16% of the fused
particles, costs about two coverage points, and regresses four held-light
means. Falling back to the original one-sided derivative wherever the extra
samples are unavailable preserves support exactly, yet produces held-light
mean/tail deltas of -0.28/-0.35, +0.33/+0.24, -1.19/-0.92, -1.00/-1.05, and
+0.04/+0.02 dB. Both prototypes are removed. Filtering derivatives across
inconsistent density modes changes orientation but cannot decide which mode is
the shared surface.

Unweighted densification no longer downloads the complete position gradient
after every optimizer step. Meganeura `419e928` optionally folds an exact
grouped gradient-norm accumulator into its existing Adam shader: for the
`[N,3]` position table, one lane per point adds
`sqrt(gx² + gy² + gz²)` to a persistent `N`-element buffer. Blade reads that
buffer once at the densification boundary. This adds no graph operation,
shader group, entry point, pipeline variant, or extra dispatch.

The matched three-round Bonsai segment falls from 84.741 to 41.164 seconds;
gradient readback falls from 44.134 to 0.064 seconds and selected held-out PSNR
changes from 16.0160 to 16.0187 dB. Room's final 698,940→735,103 growth
boundary falls from 123.183 to 97.391 seconds; readback falls from 25.007 to
0.007 seconds and held-out PSNR changes from 22.7424 to 22.7397 dB. Sampling
the old CPU readback every two or four steps was faster but lost about 0.12 dB
on Bonsai after three growth rounds, so temporal subsampling is rejected.

The same Meganeura change fixes the previously suspicious local test crashes.
The default builder now shares one process GPU context instead of creating a
validated Vulkan instance/device in every parallel test. The formerly
crashing 43-test `gpu_smoke` target passes at normal parallelism in 2.24
seconds with an 888 MB scope peak; its serial workaround took 13.77 seconds.
The complete all-target gate passes in its 12 GiB cgroup with zero swap, OOM,
throttling, or recorded GPU faults.

The next profile closes the easy exact-topology optimizations. An isolated
698,940-cell Room rebuild spends 10.169 of 12.749 seconds inside Qhull; edge
extraction and CSR construction account for only 1.150 and 0.717 seconds. A
stack-only tetrahedron-index micro-optimization moves the total by just 0.040
seconds and is removed. Less frequent exact rebuilds are fast but not stable:
Bonsai cadence 125/250 changes individual held views by as much as -5.25/-1.40
dB, and Room cadence 50 spans -0.14 to +0.14 dB despite a neutral mean. The
existing growth cadences remain selected. A meaningful further topology win
needs an algorithmic replacement for repeated exact Delaunay construction,
not more shader variants, skipped correctness checks, or stale-topology
tuning.

The first fresh Room loss gate was not a valid all-view tail comparison. Its
255-view cap omitted the last 17 non-held-out cameras, and the two apparent
Smooth-L1 failures were held-out views beyond that training arc. Repeating
from initialization with all 272 non-held-out cameras raises those endpoint
views by roughly 6--10 dB and removes the false tail failure. This also
motivates the minimal CLI convention `--views 0`: it now forwards the
library's existing unlimited-view mode instead of requesting an empty set.

The corrected loss conclusion is budget-dependent. Smooth-L1 gains 0.364 dB
all-39 at step 1,000 and improves 32 views, but at the shared 735,103-point
cap it is -0.005 dB. After 500 fixed-cap steps it is +0.008 dB, with per-view
changes spanning -0.84 to +0.80 dB. Smooth-L1 is useful for early Room
convergence and remains selected for Bonsai, but it is not a robust Room
endpoint default. The experimental beta control remains removed. The next
quality work should target spatial appearance or sampling efficiency rather
than add another scene-sensitive loss knob.

Resume now avoids one redundant exact topology build. Training checkpoints
are finalized with adjacency matching their stored positions, so an unchanged
representation can upload that CSR directly; Qhull remains configured for the
first later position update. On the 735,103-cell Room checkpoint this removes
13.73--13.78 seconds from startup, while the forced post-update rebuild still
takes 13.77 seconds. Weighted-to-unweighted and other explicit topology
overrides continue to rebuild. This is a representation-preserving lifecycle
fix, not a new topology mode or shader path.

A surface-only visual hull does not close the synthetic geometry gap. The
missing control sampled Gaussian particles only on the boundary of the strict
six-training-mask intersection, excluding faces caused by the finite search
box and never constructing polygons. At a 64-cell resolution it produces
3,174 particles, but only 273 have support from two source-depth views. The
rest are silhouette-cone curtains at depths the masks cannot constrain.
Against a matched known-light, no-refinement control, position RMSE changes
0.6287→1.8447, normal RMSE 66.88°→102.37°, coverage 53.7%→44.3%, and unseen-
light mean/worst PSNR 18.51/18.29→16.31/15.90 dB. The prototype and CLI switch
are removed. Together with the rejected filled-hull initializer and
post-extraction silhouette filter, this closes silhouette-only geometry as the
next path: the independent cue must select depth, most directly through dense
multi-view correspondence, while masks remain occupancy supervision.

The existing joint production-render objective also cannot safely manufacture
that correspondence by adding free axes. Cycling its unchanged antithetic
probe over each particle's normal, tangent and bitangent improves all five
synthetic unseen-light means by 0.09--0.13 dB and tails by 0.06--0.16 dB. It
also lowers every training objective, yet Bonsai held-out mean/worst fall by
0.02/0.15 dB and coverage by 0.1 point; Room gains 0.01/0.03 dB. Retaining the
selected eight normal rounds and appending just one tangent and bitangent round
still loses 0.05 dB on the Bonsai tail and 0.1 coverage point for a 0.01 dB
Room-tail gain. The implementation and diagnostic switch are removed. Normal-
only motion remains selected: a photometric loss can exploit tangent freedom,
but without correspondence it does not know which neighbouring surface point
the moved particle is meant to represent.

The selected known-light path now closes one mismatch between normal fitting
and the final renderer. After the sample-wise photometric solve and material
fit, an opt-in complete-render pass perturbs every observed Gaussian normal in
deterministic antithetic pairs across all measured lights. A summed-area table
of the plus/minus pixel-error difference chooses a direction inside each
particle's projected footprint; a full multi-light render accepts or rejects
the combined proposal. Centers, radii, materials, assignments, and lights stay
fixed. Normal changes rebuild every affected TLAS because they rotate the
finite surface proxy; a regression verifies that the retained state has the
same loss as a fresh tracer.

On the fixed cloud and three independent training replicas, with `studio`
entirely held out, held-light mean/worst PSNR changes from 19.08/18.84 to
19.43/19.20, 20.19/19.78 to 20.44/20.02, 19.80/19.21 to 20.05/19.42, and
19.61/19.36 to 19.91/19.59 dB. All eight rounds are accepted and reduce their
complete four-light training objective by 10.1--12.2% in 0.72--0.75 seconds.
The fixed-cloud normal-only isolation gains 0.36/0.37 dB before the existing
material polish. Coverage falls 0.1--0.4 points and nearest-truth normal RMSE
rises 0.17--0.35 degrees: these are effective PBR shading normals compensating
for remaining surface/transport error, not more accurate geometric normals.
The pass is therefore opt-in through `--render-refine-normals` in both the
synthetic gate and calibrated multi-capture `reconstruct`; ordinary one-light
phone capture remains unchanged. No shader, Meganeura operation, entry/group,
binding, pipeline, or serialized field is added.

The calibrated repeat-light solve now preserves its exact per-view
correspondence before that rendered pass. The original solver first averages
one Gaussian's radiance across camera views for each light; an occlusion or
overlap error at one projected image location can therefore contaminate a
different location before orientation is solved. The selected correction
keeps that shared multi-view solution as an anchor, independently solves the
same projected observation across the measured lights, averages the resulting
view-local directions, and applies half of that normal-space correction.

Against the exact four-cloud gate above, pre-render truth-normal RMSE improves
53.96→53.66, 54.41→54.28, 56.31→56.09, and 57.56→57.19 degrees. After the
complete-render normal and material passes it improves 54.31→53.95,
54.68→54.67, 56.48→56.23, and 57.91→57.49 degrees. Held-light mean/worst
PSNR improves 19.43/19.20→19.49/19.25, 20.44/20.02→20.53/20.10,
20.05/19.42→20.22/19.53, and 19.91/19.59→20.02/19.69 dB. Coverage changes
by 0.0, -0.3, -0.1, and -0.1 points. The correction adds about 0.05 seconds
at this scale and is used only when aligned captures under distinct measured
lights were already requested.

A 45-degree medoid/inlier consensus is rejected: it worsens truth normals on
three clouds and gives back output quality on two. Raising support from 1.60
to 1.62 cells restores coverage, but weakens geometry and loses up to 0.09 dB
of the selected gain on the third cloud. The simpler anchored mean and existing
1.60-cell support remain. This adds no shader, graph operation, runtime field,
or alternate point-cloud representation.

Repeated lights do not improve the earlier center plane sweep by being treated
as extra texture channels. On the fixed cloud, concatenating independently
normalized patches from all known lights raises held-light mean/worst PSNR by
0.03/0.03 dB, but position RMSE rises from 0.6204 to 0.6215 and pre/final
normal RMSE rises from 53.66/53.95 to 54.09/54.48 degrees. Requiring the
primary light to be textured, taking the median light cost, and encoding a
centered per-pixel lighting signature also fail to improve geometry and output
together. Analytically inverting the four known lights per pixel is no better:
plane-sweeping the recovered world-normal field reaches 0.6192 position RMSE
but regresses pre/final normal RMSE to 53.74/54.07 degrees and held-light
mean/worst PSNR to 19.46/19.23 dB; the recovered diffuse-albedo field reaches
0.6210, 54.00/54.37, and 19.43/19.17 respectively. The temporary capture APIs
and benchmark switches are removed. Lighting variation is useful for fitting
orientation at an established correspondence, as in the selected normal
solve, but is not a reliable substitute for view-invariant spatial texture
when establishing that correspondence.

Applying the selected known-light normal solve before the original plane sweep
does not fix this ordering problem. It reduces the fixed cloud from 259 to 207
scorable center particles and changes position/pre-render normal RMSE only to
0.6203/53.93 degrees, while held-light mean/worst PSNR falls to 19.37/19.19
dB. Correcting orientation first also changes the tangent plane in which the
radiance patch is sampled; it cannot supply the missing spatial
correspondence. The production order and benchmark CLI remain unchanged.

Extending the exact rendered-material coordinate descent across all four
measured lights is not selected. On the fixed cloud it changes 35 of 36 albedo
coordinates but lowers unseen-light mean/worst PSNR from the selected
19.49/19.25 to 19.46/19.24 dB. Giving the primary capture half of the total
objective weight reaches only 19.47/19.25 dB while increasing material-polish
time from about 0.52 to 3.15 seconds. The complete renderer resolves the
overlapping-Gaussian colour mixture, but remaining geometry and visibility
error still makes the extra lights an inconsistent material constraint. The
multi-tracer material API and synthetic branch are removed; the selected
primary-light polish remains the smaller and better output path.

Screen-disjoint batching is not selected for the exact rendered center/radius
polish. Grouping at most eight particles whose 1.25-radius projected bounds do
not overlap cuts the fixed 1,170-particle pass from 20.49 to 7.16 seconds. A
failed whole-frame proposal recursively falls back to smaller batches, so all
retained batches lower the training objective. The four-cloud held-light
mean/worst deltas versus the sequential pass are 0.00/+0.01, +0.02/+0.02,
-0.02/-0.03, and -0.01/-0.02 dB; coverage also loses 0.1 point on two clouds,
while position and normal truth errors are mixed. A cap of four is slower and
does not recover the tail. Finite Gaussian support and visibility remain
coupled outside nominally disjoint image rectangles, so the prototype and its
host-only batching types are removed. A safe speedup needs a cheaper exact
TLAS update or differentiable/batched geometry objective, not approximate
independence. No shader, graph operation, entry variant, or output format was
added.

An exact in-place TLAS refit is also not worth the extra Blade API. A local
Vulkan `UPDATE`/Metal refit implementation reserves the correct update scratch,
passes the existing updated-versus-rebuilt pixel regression, and produces
byte-identical `.ply`, `.rply`, and `.f32` outputs. The same fixed pass changes
only from 20.49 to 20.07 seconds. Rebuilding the small 2,179-instance TLAS is
therefore not the bottleneck; repeated render/readback synchronization is. The
Blade changes, local dependency override, and one changed relight call are
removed. Future exact work should reduce synchronization boundaries while
retaining the sequential acceptance decisions.

Submitting both exact center/radius directions together through two prepared
tracers does remove one wait, but defeats the selected first-improvement exit.
The fixed 1,170-particle center-plus-radius pass slows from 20.49 to 26.49
seconds. Scoring both directions instead of retaining the first improvement is
also neutral to slightly negative across all five known and held environments;
for example, tone-mapped `studio` mean/worst changes 23.84/21.00 to
23.83/20.99 dB. The dual-tracer host API and hidden benchmark gate are removed.
An exact speedup must preserve early acceptance while reducing work within one
proposal, rather than batching work the current algorithm often skips.

Learning the signed oriented-surface offset during the new masked PowerFoam
continuation is not selected. `synthetic_foam` now exposes the production
continuation stage so this can be gated through the same four-cloud geometry,
known-light-normal, material, and unseen-light chain. At an offset-rate ratio
of 0.01, final truth-normal RMSE improves on all four clouds by 0.24--0.93
degrees, but position RMSE worsens by 0.0007--0.0011 on every cloud and one
held-light result loses 0.11/0.09 dB mean/worst. A 0.0025 ratio limits the
position regression to 0.0002--0.0003, but worsens one normal result, one tail,
and one coverage result. Both lower the masked training objective. The offset
API and CLI control are removed: the differentiable renderer can use this
coordinate to improve its own oriented-cell appearance without recovering the
common Gaussian surface.

Halving the continuation's topology-refresh count is not a meaningful speedup
either. Raising the fixed rebuild cadence from 100 to 200 updates changes the
four 1,800-update times from 21.49--21.62 seconds to 21.17--21.50 seconds,
with three improvements below 0.12 seconds and one 0.32-second outlier. It also
loses up to 0.2 coverage point and shifts held-light means negatively on two
clouds. Cadence 100 remains. The measured bottleneck is the differentiable
PowerFoam work between rebuilds, not adjacency refresh. All runs were isolated
under the 12 GiB scope; peak host memory was below 0.37 GB with zero swap, OOM,
Xid, or reset events.

The differentiable continuation itself now uses a selected 192-entry path row
instead of 256. Profiling the 2,179-particle surface shows that the graph, not
the recorder, dominates each update: 8.1 ms across 258 graph passes versus
1.3--1.5 ms for path recording, with 524,288-element padded activations at the
old budget. In three order-balanced repeats on each of two discriminating
clouds, median continuation time falls 21.17→17.21 and 21.38→17.29 seconds
(18.7% and 19.1%). The fixed cloud's median static-light-field score remains
23.78/23.21 dB mean/worst and held-light PBR changes by -0.01/-0.01 dB; the
second cloud gains +0.04/+0.04 dB static LF and +0.03/+0.01 dB PBR. Position
RMSE and coverage are unchanged, median normal RMSE changes by at most 0.01
degree, and two further independent clouds stay within 0.03 dB downstream
while improving their static-LF aggregate. All four record zero truncation at
28--36/192 paths. A 160-entry arm is faster but loses 0.07/0.09 dB static LF
on one cloud, so it remains rejected. The synthetic gate can now persist the
continued light field with `--surface-powerfoam-output`, matching production
`reconstruct` and preventing a PBR-only performance decision.

The first honest single-light continuation audit makes the remaining ambiguity
explicit. Without photometric normals or the later multi-light rendered-normal
oracle, the four selected clouds finish at 0.5842--0.6213 position RMSE but
63.77--67.15 degrees normal RMSE; held-light PBR spans 19.18--20.33 dB. The
ordinary phone-capture path therefore remains much less geometrically accurate
than the calibrated-repeat gate. Removing the selected view-facing normal loss
speeds continuation by about 11%, but worsens truth-normal RMSE by 1.19--1.59
degrees on every cloud and regresses held-light PBR on three. Doubling its
normal rate also worsens every normal gate by 1.5--2.3 degrees. The selected
0.1 loss weight and 0.1 normal-rate ratio remain.

Appearance-derived material shortcuts do not repair that single-light gap.
Assigning unseen Gaussian particles from the nearest recovered-light palette
colour loses 0.02--0.06 dB held-light PBR on all four clouds. Treating the
trained PowerFoam SH field as dense material observations nearly eliminates
unseen particles, but loses 0.9--1.2 dB because the static field correctly
bakes the training illumination into appearance. Both prototypes are removed;
the light field and relightable material surface remain distinct outputs.

A staged geometry-only continuation with the converged density and SH frozen
is also rejected. A 600-update, 0.0025-rate surface-offset pass worsens position
RMSE 0.6213→0.6220 and held-light mean/worst 19.18/18.99→19.09/18.90 dB. A
conservative 150-update, 0.0005-rate pass leaves the mean effectively neutral
but loses the tail 18.99→18.97 dB. In both cases the sampled frozen-field loss
fails to show a stable improvement. Finally, freezing support radii removes the
full geometry Jacobians and cuts the same 1,800-update pass from 17.5 to 12.3
seconds, but worsens normal RMSE 65.13→66.36 degrees, coverage by 0.5 point,
and held-light mean/worst to 18.93/18.68 dB. The prototypes are removed. Radius
learning is expensive but useful; the next performance work must optimize its
existing Jacobian path, while the next fidelity work still needs independent
dense multi-view correspondence rather than another rendered-coordinate knob.

Per-dispatch profiling at the selected 192-entry row narrows that performance
work further. The graph step is about 6.0 ms on the RTX 5070. Three existing
four-term, two-gather surface-colour reductions account for roughly 1.15 ms;
the three radius-gradient `ScatterAddAtomicRowMul` passes account for another
1.06--1.17 ms. A Meganeura scheduling prototype that stopped cloning shared
pointwise producers into compatible reductions materialized two extra passes,
raised each surface-colour reduction from about 0.38 to 0.44--0.47 ms, and
raised the complete graph step to 6.21--6.28 ms. The prototype and diagnostic
hooks are removed. Existing code-generated reduction fusion is beneficial;
future work should improve the generic multi-gather reduction or atomic
accumulation itself, not add a Blade-specific op or pre-made shader variant.

Splitting the facing-ranked photographs into two disjoint correspondence sets
does not make the image-derived center sweep more reliable. Requiring both
halves to prefer the same depth cuts accepted moves from 63/92/86/90 to
31/54/46/52 on four clouds, but changes neither position nor normal error
meaningfully. Held-light mean/worst PSNR changes by 0.00/-0.05,
-0.04/-0.04, 0.00/-0.06, and -0.02/+0.02 dB while scoring work roughly
triples. The prototype is removed. Agreement between smaller subsets of the
same normalized patches cannot resolve their shared occlusion and material
ambiguity; the fidelity target remains a learned descriptor or joint rendered
surface objective.

The narrow row-scaled atomic accumulation is now selected without adding an
operation or shader variant. Meganeura mutates the existing fused scatter's
work mapping when its row has at most 16 channels: one invocation loads the
source row, index, and scale once, then accumulates its channels. Wider rows
retain the scalar mapping. On the surface continuation's four- and nine-channel
gradients, the three dominant scatters fall from 1.06--1.23 ms to about
0.08 ms total and the 258-pass graph from about 6.1 to 5.0 ms. Four complete
1,800-update continuations fall from 17.69--17.95 to 15.57--15.68 seconds
(12--13%) with unchanged position error and coverage within 0.1 point.
Held-light mean PSNR changes by +0.01, +0.03, +0.13, and -0.07 dB in the
first paired runs; repeated fourth-cloud medians narrow the last delta to
-0.04 dB, within both kernels' observed atomic variance. A repeated-index
physical-GPU gradient oracle is bit-exact, the complete Meganeura all-target
suite passes. After the shader-entry cleanup merged, the seven-commit
performance stack rebased conflict-free onto Meganeura `db4fe42`; Blade pins
the resulting `2e508fb`. Meganeura's complete all-target physical-GPU suite
and Blade's workspace all-target suite pass. The fixed 1,800-update
continuation remains at 15.54 seconds with unchanged position error and
coverage, so the rebase preserves the selected scheduling result.

The next two generic scheduling shortcuts are not selected. Packing four
pointwise elements into each invocation initially exposed that the runtime
uniform path deliberately zeroes the unused padding fields; reducing the
dispatch without plumbing the new count consequently left three quarters of
each output unwritten. The probe was stopped inside the 12 GiB scope and
removed. More importantly, even that invalid quarter-work dispatch did not
improve the dominant ReLU family, so adding runtime protocol and vector/scalar
codegen branches has no performance case. Separately, one-lane-per-row
reductions can provably bypass their workgroup scratch and barrier. A generated
WGSL test and the complete physical-GPU reduction oracle passed, but the
surface-colour family remained 1.58--1.84 ms versus 1.58--1.77 ms before, and
the graph remained about 5.0 ms. Explicitly hoisting each gather's indirect
row index out of the nine-column loop is likewise bit-exact and flat at
1.63--1.82 ms; the driver already handles that invariant effectively. Both
reduction branches are removed: table traffic dominates, and none of these
experiments justifies another schedule mode, operation, shader entry, or
retained special case.

Amortizing that portable scalar path across two independent rows is selected.
It keeps the existing 256-thread workgroup, bindings, operation set, and
`ShaderEntry` sentinel, but dispatches half as many workgroups and lets each
invocation stride to its second row. The 258-pass continuation graph falls
from 5.14--5.33 to 4.91--5.21 ms and its dominant reductions from 1.76--1.79
to 1.58--1.74 ms. A 120-update gate produces byte-identical PLY and raw-scene
outputs. Three rows loses the tail gain and four returns to the control late
in the run, so two is retained in Meganeura `1b16215` and Blade `b92a1b9`.

The final simplified-main integration deliberately keeps less of that stack.
Merged Meganeura main unifies generated shaders and entry resolution, while a
four-commit integration branch adds only Blade/Naga dependency alignment, the
required grouped Adam diagnostic, and the already selected narrow row-scaled
scatter mapping. Clean simplified main regressed the fixed 1,800-update
continuation to 17.566 seconds; restoring that existing-shader work mapping
reduces it to 15.772 seconds (10.2%) without a graph operation or
`ShaderEntry`. The smaller two-row reduction optimization is omitted rather
than restoring its schedule branch. Meganeura's full all-target suite and
Blade's workspace/all-target suite pass at 10.94 GB and 3.84 GB cgroup peaks,
respectively, with zero swap, pressure, OOM, Xid, reset, or GPU fault.

Estimating a normal from the fused positions inside each voxel does not repair
the geometric initializer. Replacing its depth normal with the least-variance
axis of a weighted intra-voxel covariance raises the fixed held-light mean from
19.16 to 19.26 dB, but lowers the worst view to 18.88 dB, reduces coverage from
52.2% to 50.8%, and worsens normal RMSE from 65.13 to 72.05 degrees. Requiring
the covariance residual to be below 5% of its trace, aligning its axis within
60 degrees, and blending it halfway still gives only 19.07/18.91 dB held-light
mean/worst with 65.88-degree normals. Both prototypes are removed. Samples
inside a spatial voxel retain camera-ray depth uncertainty; their covariance
is not reliable surface-tangent evidence.

The selected rendered-material pass now begins with a joint linear solve over
the small shared albedo table. With geometry, assignments, light, roughness,
and specular response fixed, the production renderer is affine in diffuse
albedo: an all-zero render supplies the intercept and one unit render per
albedo coordinate supplies the basis. Normal equations solve the overlapping
Gaussian pixel mixtures directly, with a `1e-4` ridge toward the observation-
based initializer; the existing exact sRGB coordinate descent then polishes
and accepts the result. The host-only implementation caps the system at 96
coordinates (32 materials) and adds no WGSL, operation, binding, shader entry,
or pipeline variant.

Against the prior selected outputs, four independent full continuations improve
held-light mean/worst PSNR by 0.11/0.02, 0.06/0.03, 0.02/0.04, and 0.06/0.07
dB with unchanged coverage. Two in-process controls clone the identical fitted
scene immediately before refinement and compare against coordinate descent
alone, removing atomic continuation variance: fixed improves held-light
mean/worst/covered PSNR by 0.07/0.02/0.13 dB, and the formerly regressing second
cloud improves by 0.05/0.04/0.08 dB. The linear stage adds about 0.14--0.15
seconds to the six-material synthetic pass. A deliberately sparse replay with
only 19% coverage loses 0.58 dB where hit despite a lower training objective;
the pass therefore remains explicit through `--render-refine-materials` rather
than becoming an unconditional material fit.

Image-edge agreement does not make localized rendered-surface updates safer.
A host-side prototype computes forward-difference RGB edge residuals from the
same antithetic production renders and vetoes a particle move when its edge and
colour directions disagree. On the five independent synthetic clouds it
changes held-light mean PSNR by -0.03, -0.03, 0.00, -0.01, and -0.06 dB; every
worst-view score is flat or lower. It also raises the eight-round refinement
time from 0.224--0.232 to 0.415--0.439 seconds because retaining rendered
frames and building a second summed-area field costs more than the selected
colour-only path. The prototype is removed. Single-scale image gradients are
not independent geometry evidence and should not be added to this localizer.

Requiring the localized rendered direction to agree between alternating
three-view subsets is more conservative but not more accurate. It reuses the
same antithetic renders and summed-area data, yet lowers all five held-light
means by 0.01--0.03 dB versus the all-view direction, loses as much as 0.05 dB
on a worst view, and leaves truth geometry neutral to slightly worse. Like the
earlier split-patch test, weaker subsets discard useful evidence without
making their shared visibility and material ambiguity independent. The
prototype is removed without adding a control or representation variant.

A correspondence-trained diagonal patch metric also fails the five-cloud
gate. The diagnostic learns 27 reliability weights for the existing 3x3 RGB
descriptor from 1,114 exact correspondences visible in at least four training
cameras; it uses neither held-out poses nor the held-out light. Against paired
same-binary controls, held-light mean/worst PSNR changes by +0.08/+0.08,
+0.08/+0.09, -0.87/-0.67, -0.27/-0.20, and +0.04/+0.05 dB. Position RMSE
changes by at most 0.0001 and normal RMSE by at most 0.03 degrees, so the large
appearance regressions are not buying a better surface. Three repeats of the
first pair are identical at printed precision, ruling out GPU atomic variance.
The estimator, synthetic truth-track adapter, tests, and option are removed;
the production COLMAP loader remains minimal and does not retain observations
for a metric that has no transfer case. A useful learned descriptor needs
spatial context or nonlinear view invariance, not just rescaling the same
normalized patch coordinates.

Training the same positive metric on the desired depth ordering fails an even
earlier gate. A second diagnostic uses 1,206 training-only truth tracks and
nearby normal offsets to increase descriptor coordinates whose disagreement
ranks the exact surface ahead of a displaced tangent patch. On the fixed cloud
it improves position RMSE from 0.6213 to 0.6208 and normal RMSE from 66.56 to
66.55 degrees, but held-light mean/worst PSNR falls from 18.07/18.06 to
18.04/18.01 dB. It is rejected before the five-cloud gate and removed. This
control is more fundamental than the choice of descriptor: with noisy normals,
finite overlapping Gaussian support, and an imperfect material/visibility
model, moving a proxy toward the nearest truth surface can make the final
renderer less correct. The next joint objective must optimize surface and
support together; correspondence depth alone is not a sufficient target for
the current proxy.

Halving the localized rendered-center step for the last four of eight rounds
does not provide a safer trust region. Against fresh same-binary controls, the
five held-light mean/worst deltas are -0.03/-0.03, -0.03/-0.02,
-0.04/-0.05, -0.04/-0.04, and -0.03/-0.03 dB. All eight proposals are still
accepted by every training objective, while position and normal truth errors
are mixed at the fourth decimal and hundredth of a degree. The prototype and
its unit test are removed. The retained constant 2.5%-radius step is not merely
overshooting late in the selected schedule; its full-size later moves contain
novel-view signal that a generic coarse-to-fine rule discards.

The first direct-Gaussian image-formation path is now implemented without a
Meganeura operation, shader entry, or pre-made shader. A CPU oracle implements
the 3DGUT unscented projection (seven sigma points at alpha=1, beta=2,
kappa=0), exact 3D maximum response along a ray, per-ray depth sorting, and
front-to-back alpha compositing. The training graph consumes host-recorded
candidate indices and expresses isotropic scale, opacity, SH-0 colour, exact
ray response, transmittance, RGB loss, and optional foreground-opacity loss
entirely with existing Meganeura graph operations. Analytical projection and
rotation tests, a CPU/GPU pixel check, and a two-view optimizer recovery test
pass on the physical RTX 5070.

On five independently reconstructed 2,179--2,340-particle surfaces, 500 masked
appearance-only updates improve the direct Gaussian renderer's two held-pose
sRGB mean PSNR from 12.28 to 16.47, 12.77 to 17.65, 12.80 to 17.62, 12.70 to
17.45, and 12.65 to 17.48 dB. Both held views improve on every cloud. These
values are not directly comparable to the PBR renderer's scores: this gate
starts every extracted proxy at a deliberately flat grey and measures the new
static light-field image formation before material decomposition. Capture
radiance is converted from linear light at the boundary so the optimized SH
obeys `PointCloudModel`'s display-referred sRGB contract. Geometry stays
byte-for-byte fixed in this selected appearance gate.

Freeing centres after appearance fitting is rejected for now. On the fixed
cloud another 500 masked updates lower the training objective from 0.4013 to
0.3411 but worsen nearest-truth position RMSE from 0.62094 to 0.62487. Direct
rendering therefore fixes an important representation mismatch but does not by
itself resolve multi-view depth ambiguity. The next geometry
gate should add a surface-concentration/densification prior and screen-tile
candidate recording while keeping the same continuous graph, then extend
scale to anisotropic covariance. The temporary benchmark driver is removed;
generated PLYs remain outside version control under
`target/audit-runs/direct-gaussian-*`.

The first candidate accelerator keeps that graph and operation set unchanged.
Each selected camera gets a private 16x16-pixel tile index. The 3DGUT conic and
opacity threshold give its screen bound, with a full-screen fallback when the
finite support crosses the near plane; exact 3D response and per-ray depth
sorting still decide every retained hit. On the fixed
2,179-particle cloud and a 2,048-ray six-view batch, indexed candidates are
bit-exact with the exhaustive recorder. Median recording time falls from 25.68
to 9.49 ms; rebuilding all six view indices costs 2.99 ms and is amortized over
the configured ten-step geometry interval, for a 2.62x candidate-stage speedup.
The complete deterministic 500-update gate retains the exact 0.6073759 to
0.40126857 audit-loss trajectory and completes in 1.775 seconds. The index adds
no public option, graph operation, shader, shader entry, binding, or GPU
pipeline variant.

A radius-normalized centre-anchor prior does not turn that image gradient into
geometry evidence. On a fresh fixed-cloud gate, a 500-update free-centre stage
at rate 0.001 improves an 8,192-ray held-pose static-field sample from 14.28 to
15.50 dB, but worsens nearest-training-truth position RMSE from 0.62094 to
0.63382. Anchor weights from 0.0001 through 10 only trace a tradeoff: at weight
10 the RMSE still worsens to 0.62149 while the held gain shrinks to 0.13 dB.
The experimental graph inputs and public option are removed. This confirms
that a centre can become a better light-field proxy without becoming better
surface geometry; the next accepted geometry step needs new support on the
existing cloud surface or independent multi-view depth evidence, not another
penalty on the same ambiguous image gradient.

Naively splitting broad Gaussians is also rejected. Offsetting two smaller
children damages both truth distance and held images; keeping the children
coincident avoids moving geometry, but only adds opacity overdraw. Duplicating
the highest-opacity 5% improves the five held samples by 0.118, 0.016, 0.061,
0.073, and 0.101 dB, while a small update to the existing opacity parameter is
both substantially stronger and free of new particles. No densification API is
kept until a residual or gradient can place genuinely new surface support.

The accelerated recorder does make a larger exact candidate budget practical.
With matched 500-update SH/opacity continuations, 64 candidates improve the
same five held samples over 32 by 0.020, 0.071, 0.145, 0.080, and 0.150 dB.
On the fixed 2,179-particle cloud, fit time rises only 1.606 to 1.650 seconds
(2.8%); the isolated scope peaks at 1.06 GB with zero swap or OOM events.
`FitOptions` therefore defaults to 64 candidates. This is a one-line policy
change on the same graph and index, not another shader or operation variant.

The direct graph is generalized in place from one isotropic scale to three
fixed-rotation anisotropic scales. The existing extracted clouds already carry
a useful frame: local Y has median/mean absolute alignment 1.000/0.887 with the
nearest training-truth normal. At scale rate 0.005, matched five-cloud
continuations improve the held samples over fixed scales by 2.469, 1.009,
1.263, 1.643, and 1.443 dB. Median learned axis ratio is 1.22--1.24 and p90 is
1.81--1.89. Replacing every learned ellipsoid with a volume-equivalent sphere
changes held PSNR by -0.239, -0.115, -0.064, -0.159, and +0.050 dB: most of
the gain comes from correcting support size, while directional covariance adds
0.105 dB on average. Larger scale rates over-expand support and are rejected.

The former `fit_isotropic`/`build_isotropic_graph` surface is renamed to
`fit`/`build_graph` instead of adding parallel APIs. Rotations stay fixed host
inputs and isotropic clouds remain the equal-scale special case. No graph
operation, shader, shader entry, binding, pipeline variant, or dependency is
added.

The accepted direct path is now available from the real synthetic pipeline as
an opt-in persisted artifact. `--gaussian-output <path>` converts the final
refined Gaussian relight surface to a neutral `PointCloudModel`, runs the
selected two-stage fit, scores the complete held-out views, and writes the
result through the ordinary Gaussian PLY serializer. The conversion preserves
centres, maps local Y onto the extracted normal, and divides the relight
surfel's three-sigma finite-support radius by three to obtain the backend's
one-sigma scale. SH-0 starts at neutral grey, so PBR materials and their
training illumination cannot leak into the static light field.

The initial 1,000-update schedule spends 500 updates on SH appearance with
fixed geometry, then 500 on SH, opacity, and three anisotropic scales. Centres
remain fixed in both stages; the rejected free-centre experiment is also
reflected in `FitOptions::default`. This reuses one graph and the existing
operations throughout.

The end-to-end five-cloud gate, starting from independently reconstructed
RadFoam fields and rerunning surface extraction plus final rendered-surface
refinement, gives:

| cloud | source RadFoam held mean | direct held mean | direct held worst |
| --- | ---: | ---: | ---: |
| fixed | 14.81 dB | 20.32 dB | 19.49 dB |
| v2 | 14.77 dB | 20.18 dB | 19.95 dB |
| v3 | 14.90 dB | 20.39 dB | 19.82 dB |
| v4 | 14.83 dB | 20.39 dB | 20.21 dB |
| v5 | 14.89 dB | 20.32 dB | 19.54 dB |

Every cloud improves over its source RadFoam field on the same environment and
pose split, by 5.48 dB on average. The fixed-scene stages reduce the audit loss
from 0.511946 to 0.373538 and then 0.128335 in 4.482 seconds. A warm release
process takes 6.6 seconds for the complete surface and direct-field path, and
the isolated scopes peak at 135--141 MiB of host memory. All five runs record
zero swap, memory-pressure, OOM, or GPU-fault events on the RTX 5070. The
generated `light-field.ply`, relight scene, source foam, and telemetry remain
deliberately untracked under
`target/audit-runs/direct-gaussian-production-v1/{fixed,v2,v3,v4,v5}/`.

The identical schedule is also exposed by the production COLMAP
`reconstruct` command. Both callers share `fit_staged`; the CLI does not copy
optimizer constants or introduce another graph. Complete held-out views are
scored before and after fitting through the same exact response oracle, now
using the private tile index for candidate enumeration.

A real-input smoke gate uses five selected Bonsai cameras at 64x42, four for
training and one held out. Depth from an existing 57,484-cell field fuses to a
deliberately coarse 162-particle surface. The then-default 1,000 updates reduce
loss from 0.822654 to 0.538042 and then 0.351435, while held-view static-field
PSNR improves from 9.01 to 11.38 dB. The same coarse PBR proxy scores 10.16 dB,
so the direct field is 1.22 dB better even in this low-capacity smoke test. The
complete warm run takes 4.76 seconds and peaks at 125 MiB with no swap,
memory-pressure, OOM, kernel, or GPU fault. Artifacts remain ignored under
`target/audit-runs/reconstruct-direct-gaussian-bonsai-default/`; this validates
the production path, not a full-resolution Bonsai quality claim.

A larger real-input gate raises the working width to 128, uses 18 training and
two held cameras, and fuses the same source field into 3,060 particles. The
source RadFoam scores 12.49 dB on the exact held split; the neutral Gaussian
surface starts at 9.80 dB. The 1,000-update baseline reaches 17.81 dB mean and
17.57 dB worst, a 5.32 dB gain over the source field. This still is not a
full-resolution Bonsai result, but it shows that the production gain survives
more views and twenty times the smoke-gate particles.

Profiling that gate attributes only 1.02 seconds to Meganeura steps and 0.05
seconds to ray sampling. Candidate collection/upload costs 6.21 seconds with
16x16 screen tiles, while model readback and index rebuild cost 1.35 seconds.
Changing the private tile size to 8x8 lowers those terms to 5.06 and 1.53
seconds and total fit time from 9.6 to 8.5 seconds, with unchanged held scores
and loss within the run-to-run noise. At 4x4, rebuild rises to 2.33 seconds and
total time regresses to 8.8 seconds. Eight-pixel tiles are therefore selected.
This is a one-constant index policy change: no operation, graph, shader, entry,
binding, or public option is added.

The staged budget is then separated rather than simply doubled. On the same
real gate, 1,000 appearance plus 500 support updates reaches 17.99 dB mean and
17.53 dB worst in 13.6 seconds: extra appearance barely helps and slightly
hurts the tail. Keeping appearance at 500 and extending support to 1,000
reaches 18.26/18.22 dB in 11.6 seconds. A balanced 1,000+1,000 run reaches a
higher 18.44 dB mean but a lower 18.17 dB worst and takes 17.1 seconds.

The 500+1,000 schedule also improves every synthetic cloud over 500+500. Held
mean gains are +0.51, +0.33, +0.22, +0.43, and +0.34 dB; worst-view gains are
+0.38, +0.16, +0.28, +0.27, and +0.28 dB. The fixed-cloud fit rises only from
about 4.1 to 5.3 seconds after the tile optimization. Staged fitting therefore
defaults to 1,500 updates, spending one third on appearance and the rest on
support. This changes only schedule policy; the graph and parameter set stay
unchanged.

Uniformly increasing extracted surface density is not selected. On the larger
real gate, reducing `--voxel-factor` from 5 to 3 grows the cloud from 3,060 to
5,783 particles and improves direct held quality from 18.26/18.22 to
18.83/18.80 dB for 12.8 seconds of fitting. Going to factor 2 and 9,406
particles crosses over at 18.78/18.68 dB and 16.1 seconds. The apparent real
win does not generalize: factor 3 changes the five synthetic means by -0.27,
-0.13, +0.22, -0.22, and +0.02 dB, with three worst views also regressing.
Held-light PBR quality falls on four clouds and coverage drops from roughly
53--56% to 46--49%. The existing factor-5 default remains; useful new capacity
needs residual-guided placement or a representation improvement rather than a
uniformly finer voxel grid.

Degree-one spherical harmonics are selected only when the capture provides at
least eight training views. The existing graph is generalized in place: each
colour channel becomes a one- or four-column table and the already-supported
`sum_inner(embedding(indices, coefficients) * basis)` expression evaluates it.
There is no new Meganeura operation, shader, shader entry, binding, pipeline,
or alternate renderer graph. Training still begins with 500 SH-0 appearance
updates; sufficiently observed captures are promoted by copying DC and zeroing
the three directional terms before the 1,000 support updates.

The policy comes from paired held-view gates. With 18 training views, Bonsai
improves from 18.26/18.22 dB mean/worst at SH-0 to 18.62/18.55 dB at SH-1;
Room improves from 17.15/16.42 to 17.45/16.74 dB. Fit time rises from 11.6 to
12.1 seconds and 17.8 to 18.4 seconds respectively. Conversely, unconditional
SH-1 overfits all five six-view uniform-light synthetic clouds: mean changes
are -0.04, -0.08, -0.35, -0.02, and -0.04 dB, and worst-view changes are
-0.29, -0.18, -0.41, -0.16, and -0.08 dB. Keeping those captures at SH-0
reproduces the selected fixed-cloud 20.83/19.87 dB gate and writes no
directional PLY properties.

The exact CPU evaluator, differentiable graph, coefficient upload/readback,
PLY persistence, and production Gaussian WGSL path are covered by degree-one
tests. The Bonsai PLY grows from 204 to 312 KiB; runtime GPU allocation does
not grow because `GaussianGpu` already reserves the renderer's maximum 16 SH
components. All paired runs completed in 12 GiB scopes with zero swap,
pressure, OOM, Xid, reset, or GPU-fault events on the RTX 5070.

A residual-guided split after the appearance stage is also rejected. The
prototype attributed full training-view RGB residual to contributing
Gaussians, moved a low-opacity child toward the weighted tangent-plane query,
and selected only the highest-residual 5% of parents. It adds 106 particles to
the 2,179-point fixed synthetic cloud, but changes held quality from
20.83/19.87 to 20.82/19.79 dB and raises fit time from 5.4 to 6.3 seconds. On
Bonsai it adds 151 particles and lowers the training objective from 0.212007
to 0.205646, yet held quality stays 18.62/18.55 dB while fit time rises from
12.1 to 15.6 seconds. Like the earlier opacity duplication, it creates
training capacity rather than new multi-view evidence. The prototype is
removed; future support must come from independent depth evidence or a
different surface representation, not another split of the same image
residual.
