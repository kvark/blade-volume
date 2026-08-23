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

The current five-cloud gate repeats that calibrated solve after the selected
density and complete-render refinements. Relative to the one-light pipeline,
four known training lights raise unseen-light PBR quality from 21.84/21.49 dB
to 22.44/21.89 dB on average and coverage from about 56.7% to 57.1%. Sharing
the calibrated tangent frames with the static Gaussian output is wrong,
however: it lowers held-view capture-light quality to 23.62/22.95 dB. The
selected path snapshots the extracted Gaussian geometry before photometric
normal calibration and trains the static field independently. It recovers
24.97/24.22 dB while the paired PBR result changes by +0.006/-0.010 dB and
-0.08 coverage points, all below the observed run-to-run fitting noise. The
extra independent fit takes 4.93--5.68 seconds versus 4.66--5.73 seconds for
the shared-output control and peaks at 242,933,760 bytes with zero swap, OOM,
pressure, or GPU fault. `reconstruct` follows the same output boundary when
repeat calibrated images are supplied; a reduced Bonsai CLI smoke completes
at a 266,240,000-byte peak.

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

Candidate geometry is now synchronized only when its parameters can change.
The appearance stage has zero position, scale, and opacity learning rates, so
its former 50 full model readbacks and all-view tile-index rebuilds were exact
no-ops. Skipping them preserves the fixed-cloud 20.83/19.87 dB gate and the
Bonsai 18.62/18.55 dB gate while reducing fit time from 5.4 to 5.1 seconds and
12.1 to 11.3 seconds respectively. This is one condition around the existing
sync path; it adds no cache, operation, shader, or public setting. Extending
the support-stage sync interval from 10 to 20 is rejected despite reaching
4.9 and 10.4 seconds: stale growing supports reduce Bonsai mean/worst quality
by 0.05/0.08 dB. Support therefore retains the ten-update cadence.

Indexed per-ray candidate recording is now divided into at most eight
contiguous host ranges. Each worker owns disjoint output slices and retains the
same exact response test, depth sort, and point-index tie break, so the
existing indexed-versus-exhaustive test remains bit-exact. Two warm fixed-cloud
runs take 2.72 and 2.77 seconds instead of 5.10 seconds after the static-index
change (5.4 seconds before either optimization), with unchanged 20.83/19.87 dB
held quality. Bonsai falls from 11.3 to 7.3 seconds while retaining
18.62/18.55 dB. The cap avoids unbounded host contention; batches below 64
rays per worker use fewer threads. This changes only private CPU work
partitioning and adds no dependency, operation, graph, shader, or public API.

Hierarchical depth fusion does not rescue the geometry either. A prototype
kept the factor-5 cloud everywhere but replaced a coarse cell with factor-3
children only when multiple fine subcells had independent camera support.
Requiring at least four of six training views grows the fixed cloud from 2,179
to 2,538 particles, changes direct held quality from 20.83/19.87 to
20.81/19.87 dB, and lowers held-light PBR from 17.88/17.76 to 17.86/17.73 dB
while reducing coverage from 53.5% to 53.2%. Requiring all six views adds only
11 particles but still reaches just 20.79/19.85 dB and worsens nearest-truth
position RMSE from 0.6212 to 0.6239. The prototype is removed. Agreement among
depth modes is not accuracy when the source foam places those modes at the
same biased depth; the next geometry step needs a correction signal rather
than finer aggregation.

COLMAP sparse tracks do provide an independent triangulated coordinate, but
they do not improve the already extracted surface by acting as another centre
anchor. After excluding every point without support from at least two of the
18 selected training cameras, 26,524 Bonsai tracks remain. A conservative
prototype considered only tracks below one-pixel reprojection error, took a
robust local normal-direction median, moved each supported surfel halfway, and
kept the cloud fixed at 3,060 particles. It supports 424 surfels and moves 399
by only 0.110 voxel on average, yet direct held quality falls from 18.62/18.55
to 18.52/18.49 dB. The anchor is removed. Sparse correspondences overlap the
textured region already handled by the image patch sweep; their useful role is
to add otherwise missing support, not compete with that surface coordinate.

That support-only form is selected. Tracks are retained only to identify
points triangulated by at least two selected training cameras; held-camera-only
tracks cannot contribute. Low-error points are averaged into the same coarse
cells as depth fusion, cells with fewer than five samples are rejected, local
point-cloud covariance supplies orientation, and candidates within one cell
of existing support are suppressed. Existing particles never move. This adds
241 particles to Bonsai and changes its direct held mean/worst from
18.62/18.55 to 18.63/18.56 dB. On the paired Room gate it adds 180 particles,
improves direct held quality from 17.46/16.76 to 17.70/17.16 dB, and changes
the relightable proxy from 11.28/10.25 to 11.31/10.26 dB while coverage rises
73.3% to 75.6%. The implementation adds no dependency, shader, graph
operation, rendering path, or point representation; it reuses the existing
surfel estimator and stores only the COLMAP track image IDs that the loader
previously skipped.

Reducing indexed candidate recording from eight scoped workers to four is
rejected. On the selected Room gate it preserves the exact loss and held-view
quality but raises direct-fit time from 6.2 to 7.3 seconds. The ray work still
dominates scoped thread creation at this batch and cloud size; eight workers
remain simpler than introducing a persistent pool.

Sparse additions now carry material evidence from the same training tracks
that justified their geometry. The ordinary centre/depth observation pass sees
only 3 of 180 new Room particles and 4 of 241 new Bonsai particles. For an
otherwise-unseen addition, each training image contributes only when at least
two low-error sparse points in that cell were tracked there. Radiance is
averaged at the original point projections rather than sampled at the cell
centre, avoiding the aggregate centre's error at depth discontinuities, and at
least two distinct training views are required. Held images remain excluded.

On the exact Room geometry, this supplements 177 particles and improves held
PBR mean/worst from 11.31/10.26 to 11.52/10.38 dB without changing 75.6%
coverage; direct Gaussian quality remains 17.70/17.15 dB versus the prior
17.70/17.16 dB. On Bonsai it supplements 235 particles and improves held PBR
from 12.78/12.40 to 12.96/12.67 dB while direct quality changes from
18.63/18.56 to 18.63/18.57 dB. Final artifacts and telemetry are ignored under
`target/audit-runs/sparse-track-observation-v6/{room,bonsai}/`.

Filtering sparse geometry by agreement with the source foam depth is rejected:
it retains only 52 Room additions and erases the direct gain, reaching
17.40/16.65 dB. Raising per-cell/image evidence from two tracked points to
three is also rejected; it lowers Room PBR to 11.50/10.09 dB. Sparse tracks are
useful precisely where the source field's visible surface is missing or biased,
so the independent correspondence is the appropriate observation gate.

Partially selecting the nearest 64 Gaussian hits before sorting is rejected.
It preserves the exact `(depth, particle)` ordering and Room quality, but the
selected tile index already leaves short hit lists: fit time is 6.3 seconds
instead of 6.2 seconds with the full small-vector sort. The prototype is
removed rather than adding another candidate-list policy.

Reducing the fixed candidate table is also rejected. On Room, 48 candidates
keeps the 6.2-second fit time but lowers held direct quality from 17.70/17.16
to 17.47/16.95 dB. At 32 candidates, quality collapses to 14.98/14.55 dB and
time rises to 6.8 seconds. The selected 64 rows are a measured capacity floor
for overlapping Gaussian layers at this resolution, not removable padding.

The existing private candidate index now retains each particle's normalized
inverse rotation and scale alongside its tile membership. Exact ray response
previously normalized and inverted the same quaternion for every candidate on
every sampled ray; those values can change only when the index is already
rebuilt after a geometry synchronization. The indexed path remains bit-exact
with the exhaustive oracle. Room fit time falls from 6.2 to 5.7 seconds with
unchanged 17.70/17.15 dB held direct quality, and Bonsai falls from 5.5 to 5.2
seconds while reaching 18.64/18.58 dB. Artifacts remain ignored under
`target/audit-runs/gaussian-candidate-transform-v1/{room,bonsai}/`. This adds
no public cache, dependency, graph operation, shader, or synchronization rule;
it extends only the lifetime of arithmetic already owned by the tile index.

Reprojecting fused depth particles into their source depth maps is not selected
as material supervision. Requiring two agreeing training cameras supplements
1,329 otherwise-unseen Room particles and raises held PBR mean from 11.52 to
11.64 dB, but lowers the worst view from 10.38 to 10.22 dB. Three cameras
reach 11.75/10.27 dB; four cameras reach 11.67/10.36 dB. None improves both
held views. The prototype is removed. Unlike sparse tracks, a fused cell centre
is not an independently observed point: sampling its projected pixel after
averaging source depths can cross an occlusion boundary. A future attempt must
retain the original per-cell sample locations or model visibility explicitly.

Retaining those exact source pixels is tested next and still rejected. Taking
the sharpest contributing depth sample per camera supplements 2,114 of 2,118
otherwise-unseen Room depth particles, but reaches only 11.73/10.26 dB. A
four-camera consensus supplements 496 and reaches 11.70/10.35 dB. Thus pixel
reprojection is not the remaining cause: these back-layer particles are valid
volumetric support but do not identify the first visible PBR surface. The
temporary provenance API is removed. Further work should improve joint
geometry/visibility or use known-light supervision rather than attach fallback
materials to occluded foam layers.

An analytical multi-light diffuse solve is also rejected without per-light
visibility. It solves one least-squares albedo per shared material from every
calibrated environment except the held-out one. Before rendered refinement it
is neutral-to-negative by 0.01 dB. Applied after normals and materials settle,
it lowers fixed-cloud held-light PBR from 19.54/19.26 to 19.25/19.04 dB. The
extra captures contain cast-shadow and interreflection energy absent from the
Lambertian irradiance basis, so the solver paints that energy into albedo. The
prototype and CLI switch are removed. Multi-light material fitting must include
visibility for each calibrated light rather than merely add observations.

Adding direct-light visibility does not rescue that analytical solve. A
temporary implementation reused the existing per-environment shadow maps and
GGX lobe subtraction, without adding a shader or operation. Applied before the
render-based material optimizer, all six materials move but the optimizer
returns exactly to its control loss and 19.54/19.26 dB held-light PBR. Applied
after it, held-light PBR falls to 18.88/18.73 dB. The full 256x128 visibility
passes take 49-50 seconds and peak near 1 GiB under the 12 GiB cgroup, with no
memory or GPU fault. Direct visibility is therefore necessary in isolation but
not sufficient on reconstructed geometry: its error and unmodelled indirect
transport still enter the albedo. The API, unit test, benchmark switch, and
helper are removed. A future multi-light material update must be judged through
the complete renderer and must model indirect transport rather than adding
another closed-form material stage.

A multi-light version of the complete-render material optimizer is rejected as
well. The temporary path prepares one existing production tracer for each of
uniform, east-sun, west-sun, and sky-dome supervision, concatenates their
rendered affine albedo responses, and accepts coordinate steps only against the
combined render loss; studio remains held out. It improves that training-light
objective from 0.0142439 to 0.0136802 in 2.46 seconds, but fixed-cloud studio PBR
falls from 19.54/19.26 to 19.48/19.25 dB. The generalized evidence API, helper,
and benchmark switch are removed. More known lights do not compensate for the
renderer missing the capture's transport; optimizing their aggregate instead
slightly moves away from the primary-light solution that generalizes best.

Production reconstruction now defaults to the selected analytic transport.
The old `reconstruct` default computed 512 CPU shadow directions during the
material fit and used 32 visibility-plus-bounce rays at every final-render
shading point, even though the selected synthetic pipeline and viewer default
to zero. On the fixed 2,179-particle reconstruction, changing only final
scoring from zero to 32 samples lowers held-studio PBR from 19.54/19.26 to
19.17/18.84 dB and raises 0.44 to 32.04 ms/frame. The depth upper bound agrees:
16 sampled rays lower 26.57 to about 25.06 dB. Removing the approximate blocker
bounce is worse still at 24.76/24.15 dB; even 64 direct-visibility samples reach
only 25.90/25.05 dB at 38--44 ms/frame. The direct-only shader prototype and
synthetic scoring switch are removed. `reconstruct --diffuse-samples N` keeps
the existing point-cloud visibility and bounce as an explicit option; at the
zero default it now also skips the corresponding CPU visibility fit so training
and rendering use one model. This changes no representation, shader, operation,
binding, or pipeline.

Two real COLMAP smokes confirm the production default rather than merely
agreeing with the synthetic gate. On five selected 64-pixel-wide Bonsai views,
analytic and explicit 32-sample transport both score 9.23 dB on the held view
(13.47 dB linear), while analytic training score is 11.66 versus 11.65 dB and
removes 1.2 seconds of visibility construction plus 3.1 seconds of
decomposition time. On the paired Room smoke, analytic transport improves the
held view from 12.64 to 13.11 dB and training mean from 10.09 to 10.40 dB;
visibility construction plus decomposition falls from 3.5 to 0.1 seconds.
Coverage is identical in each pair. The opt-in sampled path remains useful for
captures whose transport has been independently shown to need it, but it is
neither a quality nor a performance default for current reconstructed clouds.

The selected direct Gaussian continuation now returns its learned support to
the PBR surface. The static field and relightable model already have a one-to-
one particle correspondence: centers and fixed rotations came from that PBR
surface, while the direct support stage learns three one-sigma scales. The
volume-equivalent scalar `3*cbrt(sx*sy*sz)` becomes the corresponding surfel's
finite radius after direct fitting. No center, normal, material, assignment,
opacity, particle count, representation, shader, graph operation, or pipeline
changes. A checked shared function rejects missing transforms and count
mismatches; both `synthetic_foam` and `reconstruct` call it only when
`--gaussian-output` already requested the fit.

On the fixed cloud, held-studio PBR improves from 19.54/19.26 to 21.67/21.01
dB and coverage from 53.9% to 55.0%. A tangential-area radius reaches
21.62/20.56 dB; maximum tangential extent over-expands to 61.7% coverage and
falls to 19.62/18.60 dB, so both alternatives are removed. Four independently
reconstructed variants finish at 21.49--21.76 dB means and 20.70--21.18 dB
tails, improving every current mean and tail. The learned static fields remain
at 22.79--24.11 dB on those held poses.

The exact selected real-scene commands agree. Room retains its 17.70/17.15 dB
static field while PBR test mean/worst improves 11.52/10.38 to 12.49/12.20 dB
and coverage 75.6% to 79.3%. Bonsai retains 18.64/18.59 dB static quality and
improves PBR 12.96/12.67 to 13.04/12.79 dB; its already saturated coverage
changes 99.8% to 99.5%. Fit time is unchanged because the scale learning was
already part of the selected direct output.

Two geometry controls explain why this works where earlier cleanup did not.
Replacing every center with its nearest training-truth surface sample collapses
overlap support and lowers fixed PBR to 17.78/17.30 dB; replacing only normals
reaches merely 19.67/19.24 dB. Removing the 64 particles that are both unseen
and below refreshed two-view depth support is also neutral-negative at
19.51/19.25 dB. Those temporary diagnostics are removed. The reconstruction
needs a better learned footprint around its existing centers, not pointwise
surface snapping or unsupported-layer pruning.

Learned opacity does not improve the scalar PBR footprint. Multiplying the
selected volume-equivalent radius by `sqrt(opacity)` preserves projected
Gaussian energy and raises the fixed held-light gate from 21.67/21.01 to
21.90/21.15 dB, but it fails every independent-cloud mean: 21.49 to 21.40,
21.51 to 21.35, 21.76 to 21.66, and 21.74 to 21.63 dB. Three of the four tails
are neutral or worse as well. A cube-root opacity factor is tied with the
square-root factor on the fixed gate. Both are removed. The PBR compositor
normalizes overlapping hits into one opaque surface band, so Gaussian alpha
cannot be converted into that representation by shrinking each radius in
isolation.

Repeating complete-render material refinement after radius feedback is also
rejected. It improves all five synthetic means and tails by 0.01--0.09 dB and
raises Bonsai from 13.51/13.37 to 13.70/13.56 dB, but Room changes
12.58/12.23 to 12.64/12.22 dB. Halving the second step retains the Room tail
loss. Moving the one existing material pass after support learning instead of
repeating it is worse on the fixed tail, Room tail, and both Bonsai scores.
All reorder/repeat code is removed. A lower training render loss is not enough
to resolve the mismatch between single-light material evidence and overlapping
volumetric support.

A final private recorder micro-optimization is likewise not retained. Caching
the reciprocal Gaussian scale alongside the already cached inverse rotation
passes the bit-exact indexed/exhaustive oracle, but three complete Room fits
remain at 5.6 seconds, inside the established 5.6--5.7 second range. The
compiler already eliminates enough repeated division that an additional
cached field has no measurable production value.

Composing the cached inverse rotation and scale into one world-to-Gaussian
matrix is selected. Each candidate hit now performs two matrix-vector products
instead of two quaternion rotations and two vector divisions; the matrix is
rebuilt only when the existing tile index is rebuilt. Three fixed-cloud
repeats reduce median direct-fit time from 2.83 to 2.71 seconds (4.4%), while
Room falls from 5.6 to 5.2 seconds and Bonsai from 5.2 to 5.0 seconds. The
fixed static-field median changes from 23.40/22.59 to 23.41/22.58 dB and PBR
remains 21.68/21.01 dB; Room and Bonsai retain 12.49/12.20 and 13.04/12.79 dB
PBR respectively. A unit oracle compares the composed transform with the
original quaternion-space response, and indexed candidates remain exact
against exhaustive recording. This changes one private cached value and adds
no dependency, operation, shader, entry, setting, or synchronization point.

The existing per-view tile index now also caches each particle's camera origin
in Gaussian space. Every sampled ray from one view shares that origin, so
recomputing its matrix transform for every candidate was redundant; only the
ray direction still needs a per-hit transform. The indexed candidate table
remains byte-exact against exhaustive recording. Three-run medians reduce the
fixed fit from 2.71 to 2.66 seconds, Room from 5.2 to 5.1 seconds, and Bonsai
from 5.0 to 4.9 seconds, with unchanged 21.68/21.01, 12.49/12.20, and
13.04/12.79 dB PBR gates respectively. The cache adds about 0.72 MiB on the
18-view Room gate and is rebuilt with the existing index; it adds no public
state, synchronization, operation, shader, or scheduling branch.

The learned Gaussian's thinnest covariance axis is not a valid surface-normal
feedback signal. Selecting that axis from the fixed extracted frame raises the
fixed held-light PBR gate from 21.68/21.01 to 21.79/21.17 dB, but nearest-truth
normal RMSE collapses from 54.73 to 78.51 degrees and median error from 35.42
to 63.48 degrees. The apparent image gain is the relight renderer compensating
for missing transport with incorrect geometry. The prototype is removed; PBR
score alone cannot gate learned normal feedback.

Sparse-track cell confidence is now output-specific. Lowering the shared
five-point cell floor to three adds 138 Room particles and raises its direct
held field from 17.70/17.16 to 18.43/18.31 dB, but the opaque PBR proxy's
worst view falls from 12.20 to 12.06 dB. A floor of four still loses 0.10 dB
on the Room tail and lowers Bonsai PBR from 13.04/12.79 to 12.97/12.68 dB.
Both shared-threshold policies are removed.

The selected path keeps five-point cells for the relightable surface and
appends three-point cells only to the persisted static Gaussian. Individual
COLMAP points still require sub-pixel error and tracks in at least two selected
training cameras; held-camera-only tracks remain excluded. This contributes
126 static-only particles on Room and 60 on Bonsai. Room direct held quality
becomes 18.42/18.32 dB and Bonsai becomes 18.66/18.61 dB, while their PBR
models score 12.49/12.19 and 13.04/12.79 dB respectively. The committed Room
baseline scores 12.49/12.20 dB. An independent oracle run of the selected path
also rounds to 12.49/12.20 dB; three ordinary repeats round to 12.49/12.19 dB.
This one-hundredth tail band is expected from the unordered floating-point
accumulation in Meganeura's atomic embedding gradient, not a change in the
rendered support policy. A diagnostic CPU evaluation perturbed dispatch timing
and restored 12.20 dB, but is removed because a timing warm-up is not an
algorithmic fix.

The two confidence sets are fitted independently so the extra radiance support
cannot perturb the PBR radius solution. Attempts to shorten that conservative
fit to 1,000 or 1,250 staged updates lose the Bonsai tail; 1,000 support-only
updates lose 0.04 dB there as well. The complete core fit is retained as the
quality oracle, adding about five seconds only when static-only support exists
and direct Gaussian output was requested. It adds no particle type, shader,
graph operation, dependency, or output-format field.

Per-view Gaussian candidate grids are now rebuilt in parallel. A temporary
host profile of Room's 1,000-update support stage attributes 2.05--2.13 seconds
to rebuilding the 18 independent camera grids, 1.20--1.22 seconds to candidate
collection and upload, and only 0.46--0.51 seconds to GPU execution. Dividing
those camera grids across at most eight scoped workers lowers rebuild time to
0.80--0.94 seconds. Three Room runs reduce the conservative core fit from 4.7
to 3.4 seconds and the expanded fit from 5.0 to 3.7 seconds; three Bonsai runs
finish at 3.3--3.5 seconds instead of 4.5--4.7 seconds. Room retains
18.42--18.43/18.32 dB static and 12.49/12.20 dB PBR quality; Bonsai retains
18.65--18.66/18.59--18.61 dB static and 13.04/12.79 dB PBR quality. Existing
indexed-versus-exhaustive candidate tests cover the parallel construction.
The change is private host scheduling only: no model, graph operation, shader,
pipeline, dependency, or public setting is added.

Degree-two spherical harmonics are selected for direct Gaussian captures with
at least 18 training views. The existing coefficient-table graph now consumes
the five standard quadratic basis terms in addition to SH-1; the CPU oracle,
model, PLY format, and production renderer already supported them. There is no
new Meganeura operation, shader, entry, binding, pipeline, or public option.
Captures with 8--17 views remain SH-1 and smaller captures remain SH-0.

Three-run medians on the 18-view split improve Room's direct held field from
18.43/18.32 to 18.61/18.52 dB and its learned-support PBR result from
12.49/12.20 to 12.56/12.29 dB, with coverage rising from 79.3% to 80.9%.
Bonsai improves from 18.65/18.59 to 18.81/18.63 dB direct while retaining
13.04/12.79 dB PBR; coverage rises from 99.5% to 99.6%. Room's two fits take
about 3.8--4.0 seconds instead of 3.4--3.7, and Bonsai takes 3.5--3.9 seconds
instead of 3.3--3.5. The binary PLYs grow by about 58%, while runtime GPU
allocation is unchanged because it already reserves all 16 supported SH terms.

A matched four-held-view control rejects promotion at 16 training views.
SH-2 improves the Room and Bonsai direct fields by 0.30/0.40 and 0.27/0.36 dB
mean/worst respectively, but Bonsai PBR loses 0.02/0.04 dB. The selected
18-view floor is therefore measured rather than inferred from the coefficient
count. A physical-GPU graph test exercises non-zero degree-two gradients, and
promotion tests prove that DC is preserved while all new terms start at zero.

Complete-render material refinement remains explicit after the SH-2 promotion.
The combined 18-view gate raises Bonsai PBR from 13.04/12.79 to 13.51/13.36 dB,
but Room moves from 12.56/12.29 to 12.58/12.22 dB. The lower Room tail repeats
the existing cross-scene conflict even though the complete training-render
loss falls on both captures. Static Gaussian fitting does not consume the PBR
materials. There is therefore still no safe evidence for enabling
`--render-refine-materials` by default or adding a confidence-policy branch.

SH-3 at the same 18-view threshold is rejected. A temporary extension reused
the coefficient-table graph, existing operations, and the renderer's existing
SH-3 storage rather than adding a graph or shader variant. Room direct held
quality falls from 18.61/18.52 to 18.14/17.81 dB and learned-support PBR falls
from 12.56/12.29 to 12.17/11.97 dB. Bonsai direct changes from 18.81/18.63 to
18.76/18.65 dB while PBR falls from 13.04/12.79 to 12.89/12.74 dB. Coverage
rises, but the added support is wrong for both opaque PBR proxies. The complete
SH-3 prototype is removed; SH-2 remains the measured expressiveness limit.
Both gates completed under the 12 GiB cgroup without an OOM event or GPU Xid.

Splitting the same 500 appearance updates around the 1,000 support updates is
also rejected. The temporary schedule performs 250 SH-0 appearance updates,
the unchanged support stage, then 250 appearance-only updates against the
learned support. It improves Room direct held quality from 18.61/18.52 to
18.64/18.55 dB and Bonsai from 18.81/18.63 to 18.89/18.84 dB. Room PBR also
improves from 12.56/12.29 to 12.60/12.33 dB, but Bonsai PBR falls from
13.04/12.79 to 13.02/12.74 dB. The exact controls use the same one-sample
transport score; an earlier analytic-score diagnostic is not compared across
transport settings. The original uninterrupted 500-step appearance
initialization therefore remains the cross-scene choice. The third fit and all
scheduling changes are removed.

Partial sorting does not measurably accelerate candidate recording. A temporary
host-only change partitions each hit list at the retained 64 candidates and
sorts only that prefix; the total ordering makes its output identical to the
full sort. Three complete Room and Bonsai repeats nevertheless stay inside the
existing 3.5--4.1 second fit band. Tile culling has already reduced most lists
enough that selection is not the remaining cost. The helper is removed rather
than retaining an unmeasured second ordering path.

Increasing the unchanged two-stage budget from 1,500 to 1,750 updates is
rejected. Bonsai improves to 18.96/18.88 dB direct and 13.06/12.82 dB PBR, but
Room falls to 18.56/18.44 dB direct and 12.55/12.25 dB PBR despite a lower
training loss and 0.3 more coverage points. The fixed update budget remains
1,500; more optimization is not a scene-independent quality improvement.

The selected support-stage candidate-grid cadence is now 20 updates instead of
10. Candidate grids are a conservative discrete superset for the continuous
Gaussian graph, and the smaller support changes over twenty updates remain
inside that margin. Across three Room repeats, the core fit falls from about
3.8--4.0 to 3.2--3.3 seconds and the expanded fit to 3.4--3.5 seconds while
retaining 18.60--18.62/18.51--18.52 dB direct and exactly 12.56/12.29 dB PBR.
Three Bonsai runs retain 18.81--18.83/18.66--18.67 dB direct and 13.04/12.79
dB PBR while both fits finish in 3.2--3.4 seconds. The six-view synthetic gate
remains at 23.41/22.58 dB direct and 21.67/21.00 dB held-light PBR with 55.0%
coverage, inside its established one-hundredth atomic band. This is a one-line
private scheduling change: no graph operation, shader, entry, binding, model,
dependency, or public option is added.

Less frequent grid refreshes are not selected. At 40 updates, three Room runs
still retain 18.59--18.60/18.51 dB direct and 12.56/12.29 dB PBR, and three
Bonsai runs retain 18.80--18.81/18.64--18.65 dB direct and 13.04/12.79 dB PBR;
fits fall further to 2.9--3.2 seconds. The independent six-view synthetic gate
exposes stale support, however: direct quality rises to 23.50/22.67 dB while
held-light PBR falls from 21.68/21.01 at 55.0% coverage to 21.63/20.93 dB at
54.9%. At 80 updates, stale candidates become visible on the real gate too:
Bonsai direct falls to 18.76/18.58 dB and its PBR mean loses 0.01 dB. Both
cadences are removed. Twenty updates is the measured joint-quality endpoint,
not merely the first faster value tried.

A soft census descriptor does not improve the image-derived center sweep
reliably. Replacing normalized RGB patches with bounded center-relative colour
orders improves the fixed cloud's position/normal RMSE from 0.6213/54.41 to
0.6206/54.23 and its direct held field from 23.41/22.58 to 23.45/22.67 dB.
It also raises the second cloud's direct and held-light PBR scores, but the
third cloud loses 0.20/0.25 dB direct and the fifth loses 0.10/0.05 dB direct
plus 0.08/0.09 dB held-light PBR. Appending half-strength census values to the
selected normalized descriptor is worse on the fixed gate, lowering direct by
0.07/0.10 dB and PBR by 0.02/0.05 dB. Both private descriptor variants are
removed. Local order statistics still share the same occlusion ambiguity as
NCC; a future learned descriptor needs genuinely broader spatial context.

Static light fields now learn centers jointly with opacity, anisotropic scale,
and directional appearance during the existing support stage. The position
rate is a conservative `1e-4`; particle count, rotations, schedule, candidate
recording, and every graph operation remain unchanged. PBR support is fitted
independently with centers frozen and returns only its volume-equivalent radius
to the relightable surface. Thus the static field can move its proxy particles
to reproduce images without moving the geometric surface used for relighting.
The distinction is output-specific, like sparse support: it is not another
particle type or runtime representation.

Same-process fixed-center controls on five independently reconstructed
synthetic clouds show direct held mean/worst gains of +0.22/+0.12,
+0.17/+0.19, +0.32/+0.33, +0.22/+0.26, and +0.33/+0.24 dB. Their PBR geometry
and support field remain the fixed-center control by construction. Three Room
repeats improve direct quality from 18.61/18.52 to 18.73--18.74/18.69--18.71
dB while retaining exactly 12.56/12.29 dB PBR and 80.9% coverage. Three Bonsai
runs improve 18.81/18.63 to 18.97/18.83 dB while retaining exactly
13.04/12.79 dB PBR and 99.6% coverage. At 16 training views, Room improves
14.10/10.73 to 14.24/10.93 dB and Bonsai improves 18.67/17.84 to
18.80/17.94 dB, again with unchanged PBR scores. Existing Room/Bonsai captures
already required independent core and expanded fits, so the center gradient
adds no update or graph-build cost there; each field remains in the measured
2.7--3.5 second range. This adds no shader, operation, binding, model field,
dependency, CLI switch, or output format.

Feeding learned static-field centers back into the relightable surface is
rejected, including when limited to particles supported by at least four
refreshed depth maps. A 25% center blend raises held-light PBR by only
0.00--0.02 dB on five masked synthetic reconstructions, while position RMSE
worsens on all five; the second cloud also worsens in normal RMSE.
Unrestricted feedback had already slightly regressed the unmasked Room
capture. Learned centers therefore remain an output-specific light-field
degree of freedom; PBR retains the fixed reconstructed centers and receives
only the separately fitted fixed-center Gaussian radii. The feedback and its
diagnostic control are removed rather than adding a mask or confidence policy
for a marginal, geometry-negative effect. The five-case gate peaked at 197.4
MiB in the 12 GiB cgroup, with no memory-pressure event, OOM, or GPU Xid.

The two Gaussian outputs now share their appearance-only initialization. The
PBR cloud is an exact prefix of the static field before fitting; the first 500
updates cannot alter positions, scales, or opacity, so their learned DC
coefficients can initialize the common PBR particles without repeating the
same graph session. The fixed-center and learned-center support stages remain
independent, preserving the output-specific geometry decision above. Prefix
geometry, transforms, and initial appearance are validated explicitly.

Three paired 18-view runs retain Room at 18.73--18.74/18.70 dB static and
exactly 12.56/12.29 dB PBR, while Bonsai retains 18.96--18.98/18.82--18.84 dB
static and exactly 13.04/12.79 dB PBR. Median combined Gaussian fit time falls
from 6.6 to 5.8 seconds on Room and from 6.5 to 5.3 seconds on Bonsai. The
six-view masked synthetic gate retains 23.63/22.70 dB static and
21.67/21.00 dB PBR while falling from about 3.6--3.8 to 3.2 seconds. The
four-repeat real scope peaks at 427.7 MiB with no pressure, swap, OOM, or GPU
fault. This removes 500 optimizer updates and one graph build without adding a
graph operation, shader, entry, binding, model field, dependency, file format,
or CLI setting.

The fit loop now reuses its final synchronized model and candidate grids. A
support stage whose last update is a scheduled geometry refresh had already
downloaded every learned parameter and rebuilt all per-view grids; doing both
again immediately before the audit was redundant. Appearance-only stages
still download learned colour but retain their unchanged geometry index.
Non-divisible schedules keep the final download and rebuild. A unit gate covers
all three cases. Three real-scene repeats preserve the scores above while
moving the shared-output median from 5.8 to 5.7 seconds on Room and from 5.3 to
5.2 seconds on Bonsai. This is a private synchronization removal with no
schedule, update, graph, shader, model, format, or API change.

Learning a Gaussian covariance rotation around each fixed surface normal is
rejected. The temporary graph represented the angle as a normalized cosine/
sine pair and passed an isolated physical-GPU recovery test, but the added
image-space freedom overfits the training views. On matched sparse-only
128-pixel-wide, 18-training-view gates, Room held quality falls from
13.39/13.34 to 12.88/12.11 dB mean/worst and Bonsai falls from 13.27 to
13.20 dB. Training objectives improve in both cases. PBR rotations remain
fixed and its scores are unchanged within noise, confirming that the failure
is isolated to the static field. The parameter, optimizer setting, graph
nodes, and test are removed. All four runs completed in a 12 GiB cgroup; the
largest host-memory peak was 511.0 MB, with no swap, OOM, or GPU fault.

Raising private candidate-grid construction from eight to twelve host workers
is not selected. On the same 109,764-particle Room reconstruction, three
eight-worker fits take 57.8, 57.0, and 53.9 seconds; three twelve-worker fits
take 55.9, 56.8, and 53.9 seconds. The 1.9% median difference is smaller than
the overlapping run-to-run range, while held quality stays inside the normal
atomic-gradient band. The existing eight-worker cap is restored rather than
claiming a scheduling win from noise.

Sparse-only reconstruction now gives its static Gaussian an output-specific
training-track cloud. The former path included every COLMAP point, including
points tracked only by held or unselected cameras: Room trained 109,764
particles although only 39,295 input points had a selected-training-camera
track, and Bonsai trained 169,432 instead of 74,796 surviving tracked
particles. The static field now requires one such track. Its local spacing is
sparser, so a measured 15/14 radius correction restores overlap. The complete
sparse cloud remains the PBR geometry and receives an independent fixed-centre
support fit; no static centre, appearance, or radius feeds back into it.

On the 128-pixel, 18-training-view Room gate, static held quality improves from
13.39/13.34 to 15.07/15.02 dB mean/worst and total Gaussian fitting falls from
a 57.0-second control median to 43.4 seconds. PBR remains 13.42/12.94 dB with
78.0% coverage. On Bonsai, static held quality improves from 13.27 to 13.48 dB
and fitting falls from 84.1 to 63.4 seconds; PBR remains 13.00 dB with 86.2%
coverage. A stricter two-track filter is rejected: it cuts Room to 11,110
particles but loses the direct tail and collapses PBR when applied jointly.
Foam-derived reconstruction already builds support from training-view depth
and is unchanged. The selected sparse path adds no CLI setting, model field,
shader, graph operation, dependency, or runtime representation.

### Fresh current-Blade control (2026-08-22)

The canonical synthetic fixture was regenerated from the current Blade
relighting harness at eight views, five environments, 200x150 pixels and 128
paths per pixel. The selected six-view image-only foam reaches 25.59/24.68 dB
held-view radiance before extraction. Its two cloud-only outputs reach
25.06/24.34 dB for the static Gaussian field and 18.90/18.68 dB for PBR under
the unseen `studio` light, with 56.4% coverage. The recovered capture light is
64.2% relative RMS from truth after gauge. These reproducible numbers replace
the stale synthetic README row; generated data and logs remain under
`target/audit-runs/current-synthetic-v1/`.

The substitutions locate two similarly sized gaps. Holding the capture light
to truth raises PBR to 20.77/20.31 dB with the selected six materials, and 12
shared materials reach 20.90/20.41 dB. Replacing only the reconstructed
particle normals by nearest synthetic truth while retaining reconstructed
centers and radii reaches 21.87/21.42 dB. Ground-truth geometry with fitted
materials reaches 22.86 dB. Predefined lighting therefore removes roughly
half the current end-to-end loss, while normal/surface reconstruction explains
roughly half of the remainder.

Three tempting local changes are rejected. Correcting material chromaticity by
the known diffuse shade lowers held-light quality by 0.19--0.28 dB at 6--24
materials even though its training residual falls. A strict centered depth
derivative removes 13--15% of the fused particles and is mixed across five
independent foam checkpoints. Retaining every forward-difference sample and
using a centered derivative only where all four neighbors exist is still
mixed: with exact lighting its five held means change by +0.05, -0.06, -0.03,
+0.06 and +0.08 dB, and with recovered lighting it can lose more than a
decibel. Both stencil prototypes were removed. The next geometry step must
couple evidence across views; another per-image derivative or material prior
does not close the measured truth-normal gap.

A primary-light-only use of the existing rendered-normal search is promising
but not selected. With the capture environment fixed to truth and 12 shared
materials, eight antithetic five-degree rounds improve unseen-light means on
all five independent foams by +0.23, +0.10, +0.34, +0.24, and +0.23 dB.
Normal RMSE improves by 0.08--0.20 degrees before the subsequent surface
pass. One held-view tail nevertheless loses 0.07 dB and coverage falls by
0.1--0.4 point on every cloud. That five-degree substitution was removed and
motivated the smaller, coverage-aware gate below.

The dependency uprev also exposed a severe Meganeura regression before it
could be accepted as a production baseline. Its column-parallel scatter
kernel assigned one invocation to each narrow parameter column and made that
invocation walk every source row. The unchanged 1,500-update dual-Gaussian
fit rose from about 4.0 to 42.6 seconds. Meganeura `dece560` is based directly
on current `main`, aligns Blade at `88cdfc1`, keeps folded gradient constants
host-visible, and restores the existing float-CAS source-parallel mapping.
It adds no operation, shader group, entry, or variant. The same fit returns to
3.82 seconds and its physical-GPU scatter oracle passes. The host-visibility
regression has both a planner test and the blade-volume surface-only graph
test that originally exposed the null mapped pointer.

The primary-light normal search becomes tail-safe when its trust region is
halved to 2.5 degrees and its per-pixel objective includes `0.1 * (alpha -
mask)^2` whenever a capture mask exists. Against the same five independent
foams, exact capture lighting, 12 shared materials, and the same downstream
eight-round surface pass, held-light mean/worst PSNR changes from
20.90/20.41, 21.17/20.77, 21.06/20.76, 21.29/21.02, and 20.98/20.62 dB to
21.06/20.55, 21.30/20.93, 21.22/20.84, 21.38/21.06, and 21.08/20.73 dB.
Thus every cloud improves both scores; the five-cloud averages improve by
0.13/0.11 dB. Coverage changes by only -0.1, -0.1, 0.0, -0.1, and 0.0 point,
instead of falling by as much as 0.4 point at five degrees, and nearest-truth
normal RMSE improves by 0.14--0.29 degrees before the surface pass. This
schedule uses only the primary measured light for complete-render refinement;
paired secondary lights remain useful for the preceding per-surfel
photometric initializer but no longer dilute the final image-space objective.

The learned Gaussian support now receives one final check from the renderer
that actually consumes the PBR cloud. After the direct Gaussian fit transfers
its volume-equivalent scales back to the relightable surfels, an opt-in radius
pass uses the same masked complete-render objective and localized antithetic
search as the normal pass. Centers, normals, materials, assignments, and light
stay fixed. Both reconstruction commands reuse `--render-refine-radii`; no model
field, graph operation, shader, entry point, binding, or dependency is added.

On the same five exact-light, 12-material clouds, paired in-process controls
change from 21.05/20.55, 21.30/20.93, 21.22/20.84, 21.38/21.07, and
21.08/20.73 dB mean/worst under the unseen light to 21.57/21.11,
21.62/21.13, 21.51/21.20, 21.70/21.27, and 21.50/21.20 dB. Every mean and tail
improves; the five-cloud averages gain 0.37/0.36 dB. Coverage changes by
0.0, +0.1, -0.1, +0.3, and -0.1 point. Eight rounds at a 10% initial radius
step are selected. A 20% sweep improves the aggregate by another 0.06/0.07 dB
but gives quality back on one cloud, while 2.5% and 5% leave consistent gains
unused. The selected pass costs about 0.26 seconds at 1,274--1,307 observed
particles. Its five-run scope peaks at 214 MiB with zero swap, pressure, OOM,
or GPU fault; artifacts and logs remain under
`target/audit-runs/current-synthetic-v1/radius-refine-10pct-five/`.

Because support changes the same overlapping mixture that supplied the first
normal gradient, the selected joint path performs one more conservative normal
pass after radius refinement when `--render-refine-normals` is also present.
It reuses the existing eight-round, 2.5-degree search and adds no optimizer or
option. Against exact paired post-radius controls, the five held-light
mean/worst scores change from 21.58/21.11, 21.62/21.14, 21.52/21.19,
21.70/21.27, and 21.49/21.21 dB to 21.70/21.19, 21.72/21.15,
21.59/21.25, 21.78/21.32, and 21.56/21.25 dB. Every mean and tail is
non-negative; averages gain another 0.09/0.05 dB, and coverage changes only
-0.1, 0.0, 0.0, 0.0, and 0.0 point.

This remains a renderer-facing shading-normal polish, not a geometry claim:
nearest-truth normal RMSE changes by -0.11, +0.01, +0.02, -0.05, and +0.07
degrees across the five clouds. Five- and ten-degree repeats improve images
more aggressively, but on the fixed cloud worsen normal RMSE by 0.06 and 0.94
degrees respectively; both are rejected. The conservative pass costs about
0.26 seconds, and its complete diagnostic artifacts remain under
`target/audit-runs/current-synthetic-v1/post-support-normal-2p5-truth-five/`.

The opt-in rendered-material path now closes the same ordering gap. Its first
linear-plus-coordinate fit happens before learned Gaussian support is copied
back to the PBR cloud, so the final radii and normal polish change the pixel
mixtures that selected the material table. After those changes, a finer
coordinate-only pass reuses the existing production-render objective and
keeps the first fit as its initializer. Repeating the linear initializer is
rejected: at a 0.025 step it loses 0.0145/0.0049 dB mean/worst on the second
cloud, while simply reducing its step does not remove the conflict.

Against in-process post-support controls, the selected 0.0125 coordinate
polish changes five exact-light held means by +0.0359, -0.0001, +0.0121,
+0.0358, and +0.0547 dB; worst-view changes are +0.0442, +0.0053, +0.0018,
+0.0481, and +0.0553 dB. The effectively unchanged second mean varies by about
0.002 dB across repeated GPU continuations, while its tail remains positive.
Average mean/worst gains are 0.028/0.031 dB and coverage is unchanged. Reduced
production smokes also move Room test PSNR from 10.17 to 10.23 dB and Bonsai
from 13.18 to 13.33 dB; the final training-render losses fall in both cases.
The extra pass takes about 1.05 seconds for twelve synthetic materials and
0.4 seconds for six real-scene materials. It adds no shader, operation,
binding, pipeline, dependency, or acceleration-structure rebuild. Artifacts
remain under
`target/audit-runs/current-synthetic-v1/post-support-material-coordinate-five/`
and `target/audit-runs/post-support-material-{room,bonsai}-{control,smoke}/`.

Unknown-light iteration count is not the next answer. Stopping the existing
alternation at 12 rounds improves recovered-light RMS on all five clouds but
regresses unseen-light rendering on three. Extending it from 24 to 36 rounds
improves four means by at most 0.05 dB but loses 0.11/0.08 dB mean/worst on the
third. Both controls are removed. The remaining light/material gap needs new
held-independent evidence, not a different stopping point for the same
ambiguous factorization.

Raising the adaptive foam capacity is also not a safe default. Five complete
2,560-site runs improve aggregate held-light PBR by 0.04/0.13 dB mean/worst
over the corresponding 2,048-site clouds, but two means regress and one tail
loses 0.13 dB. At 3,072 sites the aggregate gain is 0.05/0.11 dB, yet three
means regress and one tail loses 0.37 dB. The larger cloud does contain a real
orientation signal: its nearest-truth normal RMSE improves on all five runs by
0.39--3.22 degrees, while position RMSE improves on three. Static Gaussian
held quality remains mixed, however, so more cells change which surface the
single-light objective selects rather than monotonically resolving it.

Training rises from the current roughly 16.5 seconds to 17.2--18.0 seconds;
both five-run scopes stay below 415 MiB with no swap, pressure, OOM, or GPU
fault. `--target-points` already exposes capacity for a scene-specific sweep,
but choosing it from the reserved poses would contaminate the held-view gate.
The 2,048-site default therefore remains. Artifacts are under
`target/audit-runs/current-synthetic-v1/capacity-{2560,3072}-five/`.

The density field itself supplies the first useful cross-view normal prior.
At each extracted Gaussian surfel, a weighted linear regression over the 32
nearest foam sites estimates the local density gradient. A full replacement
is not justified: density is volumetric, its slope sign is ambiguous, and the
best single-cloud full-gradient normal errors remain 67--74 degrees. The
selected path aligns the slope to the existing camera-facing normal and
normalizes a 10% blend. Constant density leaves the input unchanged, and a
synthetic lattice test checks both the correction direction and the small
trust region. Sparse-track additions are deliberately excluded because they
were not generated by the learned density field.

On five independently trained 2,048-site clouds, extracted normal RMSE changes
from 66.98, 66.51, 66.38, 65.93, and 67.31 degrees to 65.34, 65.03, 64.78,
64.53, and 65.66 degrees. After the unchanged support, normal, and material
passes, unseen-light PBR changes from 21.6822/21.3072, 21.7820/21.2872,
21.7077/21.4656, 21.8633/21.5129, and 21.6082/21.3489 dB mean/worst to
21.81/21.47, 21.80/21.33, 21.82/21.55, 22.00/21.67, and 21.75/21.43 dB.
Every mean and tail improves; average gains are 0.11/0.11 dB. Average coverage
changes by -0.06 percentage point, within the measured atomic-continuation
variation. A 5% blend is safer but leaves quality unused; 25% and 50% improve
PSNR further while losing up to 0.5 and 1.4 coverage points, respectively.

The final current-Meganeura replay reproduces 65.34-degree normal RMSE,
21.81/21.48 dB PBR, 57.0% coverage, and a 3.82-second dual-Gaussian fit on the
canonical cloud. A fresh 512-cell Bonsai foam also passes through the actual
`reconstruct --foam` path: 218 fused surface particles survive and all 218
receive the density correction before material/light decomposition. The
implementation adds no parameter, model field, shader, graph operation,
dependency, or runtime representation. Complete artifacts remain under
`target/audit-runs/current-synthetic-v1/density-gradient-32-{twentieth,tenth,quarter,half}-five/`,
`target/audit-runs/current-synthetic-v1/density-gradient-final-check/`, and
`target/audit-runs/density-gradient-production-smoke/`.

Two follow-up geometry schedules do not improve that calibrated result. First,
splitting the existing eight complete-render center rounds into six before and
two after Gaussian support fitting lowers the post-support training loss from
0.0038207 to 0.0037877 and position RMSE from 0.5817 to 0.5809. The immediate
same-machine control nevertheless reaches 22.48/21.76 dB held-light PBR at
57.0% coverage, while the split reaches 22.50/21.70 dB at 56.7%. The lower
training loss and slightly truer centers do not compensate for the worse
novel-view tail and support; the original eight-before schedule is restored.

Second, multiplying only the PBR Gaussian support stage's foreground-opacity
loss leaves the independently fitted static field unchanged but does not make
better relightable support. A 2x multiplier reaches 22.48/21.69 dB at 56.8%
coverage; 4x reaches 22.37/21.75 dB at 57.3%, versus the same 22.48/21.76 dB,
57.0% control. More silhouette pressure trades radiance-supported interior
geometry for boundary coverage rather than improving both. Both multipliers
and the extra fitting policy are removed.

Changing the calibrated normal search is not the next improvement either. On
an immediate fixed-cloud control, 512 equal-weight directions plus the current
half consensus correction reaches 22.48/21.76 dB held-light mean/worst at
57.0% coverage in 0.088 seconds. Reducing the grid to 256 directions is faster
but reaches only 22.44/21.65 dB; expanding it to 2,048 takes 0.275 seconds and
reaches 22.37/21.66 dB. Quarter or three-quarter consensus corrections and
inverse-residual view weights all regress the tail as well. These variants are
removed and the simple 512-direction equal-weight solve remains.

That audit exposed one real production mismatch. Synthetic gates apply the
photometric calibrated-normal solve before the trusted 10% density-gradient
correction, while production previously did the reverse. Reversing the
synthetic order loses 0.10/0.13 dB, so production now follows the selected
schedule. The independently fitted static Gaussian snapshots the original
surface and receives its density correction directly; the PBR surface first
uses all calibrated photographs and then receives the correction. Only the
foam-derived prefix is eligible, so sparse-track support is unchanged. A
distinct-light production smoke refines all 139 supported photometric normals
and 218 foam normals below 255 MiB, while a one-light replay remains byte-for-
byte identical to the previous output.

The density trust region can be stronger once calibrated photographs have
removed most of the albedo/normal ambiguity. On the same five fixed clouds,
raising only the PBR post-photometric blend from 10% to 20% improves every
held-light mean by 0.04--0.11 dB and every worst view by 0.03--0.11 dB. Mean
final normal RMSE falls from 54.24° to 53.35°, while average coverage changes
from 57.10% to 57.04%. A 25% blend has a slightly better aggregate but loses
0.03 dB on one cloud's strict tail, so it is rejected. Removing the prior or
halving it is clearly worse on the fixed control. Ordinary single-light
reconstruction and the independent static light field retain their already-
gated 10% correction; only calibrated PBR geometry uses 20%.

The split policy reproduces the static control within 0.01 dB and a physical
production smoke refines all 139 supported photometric normals followed by
218 foam normals. It completes at a 135.7 MiB cgroup peak without OOM or GPU
faults. Artifacts are under
`target/audit-runs/current-synthetic-v1/density-calibrated-*` and
`target/audit-runs/density-calibrated-production-smoke/`.

Three immediate follow-ups do not improve that selected schedule. Giving only
photometrically changed particles the 20% density correction and falling back
to 10% elsewhere restores 0.3 coverage point but loses 0.06 dB held-light
mean, with no tail gain and 0.70° worse final normals. Retuning the later
normal polish from 2.5° to 1.25° similarly trades 0.07 dB mean for 0.03 dB
tail and 0.3 coverage point; raising it to 5° loses 0.03/0.06 dB and worsens
normal RMSE by 0.39°. Finally, another 0.00625 material coordinate pass lowers
training loss from 0.0036352 to 0.0036313 but leaves the held mean unchanged,
loses 0.01 dB tail, and costs 1.27 seconds. All three prototypes are removed;
the uniform calibrated prior and existing downstream schedules remain.

Using all calibrated lights for the shared material table is geometry-limited
as well. A temporary extension reused the existing complete-render coordinate
objective with one prepared tracer per measured environment; its one-light
path was bit-identical and added no shader or pipeline variant. Equal weighting
on the fixed cloud trades 0.05 dB mean for 0.03 dB tail and 0.3 coverage point.
Counting the primary capture twice wins 0.02/0.02 dB on that cloud, but the
five-cloud gate regresses both mean and tail on two clouds and lowers aggregate
coverage from 57.04% to 56.92%. Material-polish time rises from 1.29 to 6.33
seconds on average. The added API and implementation are removed: until the
shared surface is consistent under every light, extra illumination makes the
material table absorb geometry disagreement rather than identify a better
BRDF. Artifacts are under
`target/audit-runs/current-synthetic-v1/calibrated-multilight-material-*`.

Support radii do not become a joint winner from the extra lights either. On
the fixed cloud, refining radii against all four measured captures moves
held-light mean/worst from 22.58/21.85 to 22.56/21.91 dB and coverage from
56.6% to 56.8%. Counting the primary light twice reaches 22.56/21.89 dB at the
same coverage. Both trade 0.02 dB mean for tail/support rather than improving
the full gate, and increase the radius pass from 0.30 seconds to 0.95--1.15
seconds. They are rejected before a five-cloud run and the synthetic path is
restored exactly. Extra lighting currently exposes the same inconsistent
surface under more appearances; it does not add new silhouette information.

Moving centers with the extra lights confirms the deeper limitation. A
physical-GPU oracle first verifies that the temporary normal-axis center
primitive repairs a known displaced Gaussian and retains its loss after a
fresh tracer build. Appending two such calibrated rounds to the selected
primary-light schedule lowers the four-light training loss from 0.0137116 to
0.0134029 in 0.27 seconds. On the fixed reconstructed cloud, however, position
RMSE worsens from 0.5819 to 0.5837 and normal RMSE from 55.11° to 55.17°;
held-light mean/worst moves from 22.58/21.85 to 22.52/21.90 dB while coverage
rises from 56.6% to 57.0%. The optimizer is again broadening a photometrically
convenient surface rather than recovering the shared one. The wrapper, test,
and integration are removed before a five-cloud run.

Broader normalized patches do not resolve that correspondence either. On the
fixed calibrated cloud, expanding the retained 3x3 half-radius descriptor to
a 5x5 patch over 0.75 radius improves extracted position/normal RMSE from
0.5810/56.73 degrees to 0.5803/56.45 degrees, but lowers direct held-view
quality from 24.98/24.26 to 24.93/24.20 dB and held-light PBR from
22.58/21.85 to 22.49/21.86 dB. A full-radius patch reaches slightly better
geometry at 0.5799/56.54 degrees and improves the PBR tail and coverage, but
still loses 0.05 dB mean PBR and 0.06/0.05 dB direct mean/tail. Concatenating
the original descriptor with a lightly weighted full-radius 5x5 descriptor
fails the analytical gate first: it moves 22 of 49 particles on an exact
textured sphere by 0.0056 world units on average. All prototypes are removed.
More samples of the same normalized radiance remain coupled to occlusion and
proxy support; a future correspondence model needs a learned invariant or a
joint surface/support objective rather than another patch-size sweep.

The descriptor test also reproduced the earlier intermittent llvmpipe test
fallback. Complete-render scoring created and destroyed several independent
ray-tracing contexts in one test process; after the first physical context,
Blade could reject a later 5070 initialization and continue to the software
adapter. Scoring tests now share one ray-tracing context while production
renderers retain their existing per-instance lifetime. Five repeated
four-thread focused runs execute all 35 GPU tests on the 5070 with no skip or
fallback, peaking at 164.1 MiB with zero swap, OOM, or GPU fault.

The full current-main replay also corrects the dependency history above:
Blade `88cdfc1` contains the device-scope memory-model fix, but not the three
remaining validation fixes. The minimal successor branch
`fix/reconstruction-vulkan-validation-final` contains only upstream wgpu/Naga
`cefd48f`, the external device-address allocation flag, and descriptor-array
pool sizing. Meganeura must use the same Naga revision because it passes
`naga::Module` values directly into Blade; branch `deps/blade-validation-final`
adds only that one-line alignment. There is no wgpu fork. With temporary local
path overrides to those two branches, the complete workspace all-target suite
passes with zero Vulkan validation errors or warnings. It peaks at 4.6 GiB
inside the 12 GiB cgroup with no swap, OOM, GPU fault, or software-adapter
fallback. The path overrides and lockfile changes were removed after the run;
blade-volume should move directly to the merged main commits, not retain either
temporary branch.

Repeating the normalized-patch surface search after calibrated normal recovery
does not resolve its objective mismatch. The original post-density repeat
improves fixed-cloud position/normal RMSE from 0.5810/54.56 degrees to
0.5805/54.30 degrees and moves the held-light tail from 21.85 to 21.88 dB, but
the mean falls from 22.58 to 22.57 dB. Halving the search region loses more
mean, while running before the density correction reaches a 21.93 dB tail but
only 22.53 dB mean. All three schedules are removed. Better normals make the
same photometric correspondence safer; they do not make normalized capture
radiance a shared-surface target.

More direct-Gaussian optimization is similarly synthetic-specific. Raising
the independent PBR budget from 1,500 to 3,000 updates improves all five
synthetic worst views by 0.13--0.26 dB and raises average held-light PBR from
22.52/21.95 to 22.61/22.13 dB, at 56.8% rather than 57.0% coverage. A minimal
implementation doubled only PBR while leaving the static field at 1,500
updates. The reduced production gate rejects it: Bonsai improves from 13.33
to 13.40 dB, but Room falls from 10.23 to 10.08 dB and fitting rises from
34.6/34.7 seconds to 66.9/64.2 seconds. The implementation is removed. PBR
support needs a better held-independent geometry/coverage objective rather
than longer optimization of the existing image loss.

Simple foreground balancing is not that objective. Giving foreground mask
errors twice the background weight uses only existing graph arithmetic and
raises fixed-cloud coverage from 56.6% to 57.1%, but static held quality falls
from 25.00/24.33 to 24.71/23.95 dB and PBR from 22.58/21.85 to 22.44/21.76
dB. The prototype is removed before a five-cloud run. Treating every
foreground pixel as missing support broadens overlapping Gaussians without
identifying which surface should own the ray; future coverage evidence must be
localized to geometry or visibility rather than reweighted globally.

## Full-covariance relightable Gaussian selection (2026-08-22)

The scalar-surface approximation is no longer the best durable PBR geometry.
An initial tangent-ellipse transfer was neutral and removed, but that test
discarded the learned normal-axis thickness and opacity. A CPU oracle retaining
the complete direct-Gaussian covariance raised the fixed-cloud held-light score
from 22.68/22.13 to 23.14/22.52 dB. The production GPU implementation confirms
the result across five independently trained density clouds: mean/worst PBR
averages improve from 22.658/22.208 to 23.210/22.666 dB, and covered-pixel
quality from 21.640 to 22.374 dB. Every cloud improves all three image-quality
measures. Mean coverage falls from 56.80% to 53.83%, so support remains a
separate objective rather than being hidden inside PSNR.

The implementation keeps one relight shader and one backend enum variant. A
runtime kernel value selects exact maximum-response evaluation and
front-to-back alpha composition of the learned ellipsoids; the established
surface paths remain unchanged. The canonical `PointCloudModel` carries
optional point-major normals, material assignments, and a shared material
table inside its Gaussian transforms. Binary Gaussian PLY stores those normals
plus seven namespaced material properties, reconstructing the shared table on
load. Both `synthetic_foam` and the COLMAP `reconstruct` command reload a
requested `--pbr-gaussian-output` before the final score, and the viewer routes
such a PLY through the existing relight backend automatically.

The learned response is now bounded at alpha 0.03 instead of retaining
4--5-sigma proxy tails down to 1e-5. On the same five serialized clouds this
raises mean/worst quality from 23.113/22.557 to 23.210/22.666 dB and reduces
the average per-cloud median from 8.86 to 1.49 ms per 100x75 frame. The weaker
tails consumed ray-query candidates and could hide meaningful intersections
behind the fixed hit window, so removing them improves quality as well as
speed. A sweep through 0.05 confirms 0.03 as the peak rather than merely the
most aggressive tested cutoff.

The retained core now receives a mild opacity remap
`1 - (1 - response)^1.1`. Truncation removed response mass as well as proxy
overlap; this restores part of that opacity without widening a single proxy or
adding traversal work. On the same five files, mean/worst averages rise again
to 23.234/22.676 dB and coverage from 53.83% to 54.24%. Every individual mean
and tail improves. A 1.25 exponent overcorrects two tails, while an exact
integrated-response compensation overweights low-opacity particles and
regresses one cloud. The conditional score over pixels above 50% alpha falls
from 22.374 to 22.339 dB because additional boundary pixels enter that changing
set; full-frame PSNR, worst-view PSNR, and coverage all move together and are
the selection gate.

Two newly persisted reduced production reconstructions provide the first
full-covariance real-scene smoke. Against the identical PLYs, the core remap
raises Room from 10.742 to 10.778 dB with 47.63%→48.90% coverage and Bonsai
from 12.989 to 13.084 dB with 54.32%→56.13% coverage. The final volumetric
model beats the scalar point-surface control on Room (10.78 vs 10.22 dB) but
not Bonsai (13.08 vs 13.32 dB). Thus the remap is robust within the volumetric
path, while full-covariance superiority on real captures remains unproven; the
all-point scalar backend remains the production fallback rather than claiming
the synthetic result generalizes universally.

The remap deliberately remains a final-render calibration. Expressing the
exact same exponent in PBR training needs no Meganeura extension—the existing
log and exp nodes implement `1-exp(1.1*log(1-alpha))`—but the matched first-cloud
run regresses 23.291/22.712→23.238/22.671 dB and
53.93%→53.59% coverage. The optimizer lowers learned opacity and cancels the
benefit. A cheaper endpoint-preserving quadratic is also rejected: it improves
almost every fixed-file measure but is consistently weaker than the exponent
and misses one cloud's tail by 0.0007 dB. Both prototypes are removed; no new
Meganeura operation or permanent training branch is introduced.

Fixed-file traversal probes reject the other obvious global changes: a 48-hit
window helps the former long-tail path but slows the surface path by about 37%;
96 hits loses occupancy; conservative octahedron and cube proxies create more
false candidates; early duplicate rejection and host-precomputed inverse
transforms are noise-level neutral. The shared 12-hit icosahedron traversal
remains selected without a new shader entry or backend variant. Raw models,
logs, and telemetry are under
`target/audit-runs/current-synthetic-v1/{volumetric-pbr-five,volumetric-pbr-persistence}`.

Using the same 0.03 threshold for differentiable PBR support training is not
equivalent and is rejected. On the first fixed cloud it lowers held-light
quality from 23.259/22.703 to 23.030/22.409 dB, while covered-pixel quality
falls from 22.498 to 22.193 dB and coverage is unchanged. Runtime clipping
bounds a completed continuous response; training candidate selection is
discrete, so clipping there removes the gradients that would let a particle
grow across its current boundary. Training therefore retains conservative
1e-5 candidates even though the final relight renderer uses bounded support.

Repeating material coordinate descent through the final volumetric response is
also too small to keep. It improves mean and covered-pixel PSNR on all five
clouds, but the full step regresses three worst views. Blending only a quarter
of the update changes the five-cloud average by +0.005 dB mean and +0.0004 dB
worst while adding roughly 2.2 seconds and a Gaussian-specific refinement API.
The prototype is removed. Material/geometry compensation exists, but another
local albedo polish is not large or consistent enough to resolve it.

Repeating normal refinement through the final volumetric response is rejected
for the same reason. On the first fixed cloud, four accepted rounds lower the
training objective from 0.0049890 to 0.0049636, but held-light mean/worst PSNR
changes only 23.2593/22.6995→23.2664/22.6961 dB. Quarter and half blends show
the same monotonic trade: mean and covered-pixel quality rise by thousandths
while the worst view falls. The temporary attribute-update API is removed.
The selected surface-proxy normal pass already captures nearly all available
normal signal; the next fidelity work should address geometry/visibility or
the reconstruction objective instead of adding another final-render polish.

## Gaussian surface-sheet compositing (2026-08-22)

The reduced real-scene smoke was not representative enough to decide whether
full covariance transfers. Fresh production gates use 18 training and two
held cameras at 128 pixels wide, persist and reload the PBR Gaussian, and score
the exact same file through both renderers. Before the compositing change,
Room scores 12.400/11.994 dB mean/worst at 74.57% coverage through the
volumetric path versus 12.489/12.011 dB at 87.37% through the scalar control.
Bonsai scores 14.776/14.716 dB at 79.42% versus 14.694/14.399 dB at 87.55%.
The Gaussian image signal transfers, but overlapping particles from a single
surface remain too transparent when composed as independent volumes.

The selected renderer groups maximum-response hits no farther than half the
first particle's scalar support in depth. It averages their PBR shading by
coverage, computes both their ordinary union opacity and their capped opacity
sum, and moves 75% from the union toward the capped sum. Distinct depth sheets
still composite front to back. This stays inside the existing Gaussian runtime
branch: there is no new backend, pipeline, bind group, shader entry, or asset
field.

Across five fixed reconstructed clouds, mean/worst/coverage averages improve
from 23.234/22.676 dB and 54.24% to 23.282/22.704 dB and 55.03%. Every
individual mean and tail improves. Conditional covered-pixel PSNR changes from
22.339 to 22.317 dB because the selected set expands. On the complete real
gates, Room reaches 12.492/12.009 dB at 77.67% coverage and Bonsai reaches
14.855/14.820 dB at 85.13%. That makes Room image quality effectively equal
to the scalar control and makes Bonsai clearly better, while leaving the
remaining coverage difference visible rather than claiming it solved.

Full saturation raises coverage farther but regresses one synthetic tail;
25%, 50%, 100%, and 200% support bands do not improve the joint gate. A 50%
saturation setting also clears every tail but gives up useful coverage. Global,
low-opacity-only, and isotropic-only 5% runtime scale expansion are removed:
they can help the real gates but lose synthetic means or tails, demonstrating
that no clean particle-local scale heuristic transfers. The next support work
belongs in the reconstruction objective or visibility model, not another
renderer-wide radius multiplier.

## Gaussian support-refinement transfer (2026-08-22)

The selected complete-render radius pass happened after direct Gaussian
support fitting. It therefore improved the scalar PBR surface but did not
affect the persisted full-covariance Gaussian. Blindly replacing the
ellipsoid's volume-equivalent radius is not valid: on the five exact saved
clouds it raises coverage by 3.6--4.8 points but loses 1.55--2.01 dB mean and
1.36--1.83 dB worst-view quality. Scalar disc support and bounded Gaussian
sheet support are related evidence, not interchangeable representations.

The retained transfer preserves each fitted rotation and axis ratio, and moves
only 2.5% of the log-radius distance toward the final renderer-refined scalar
radius. On the five fixed clouds, mean/worst/coverage averages change from
23.2822/22.7045 dB and 55.0328% to 23.2893/22.7108 dB and 55.0936%. Every
individual mean and tail improves. Conditional covered PSNR changes from
22.3169 to 22.3030 dB as the evaluated set expands. A freshly reconstructed,
persisted, and reloaded first cloud reproduces 23.362/22.751 dB at 54.85%
coverage, confirming that the production command applies the same correction.

The complete real gates also remain positive. Room changes from
12.492/12.009 dB at 77.67% coverage to 12.516/12.039 dB at 77.98%; Bonsai
changes from 14.855/14.820 dB at 85.13% to 14.859/14.827 dB at 85.56%.
Five-percent transfer already regresses one synthetic tail, and 10%, 25%, 50%,
and 100% increasingly trade image quality for coverage, so 2.5% is a bounded
residual rather than a claim that the scalar radius is authoritative. The
implementation is one host-side scale update in the two reconstruction
commands, adds no model field, graph operation, shader, pipeline, binding, or
runtime branch, and runs only when the existing radius-refinement option is
selected.

The immediate covariance and objective follow-ups do not improve that result.
Minimally rotating the Gaussian frame so local Y follows the final PBR normal
helps only two of five synthetic clouds; even a quarter rotation regresses the
other three. Uniformly scaling only the normal axis by 0.9 or 1.1 also trades
coverage against image quality, while applying the radius residual to only the
tangent axes, only the normal axis, only expansions, or only contractions is
weaker than the selected small all-axis update. The frame is already aligned on
the complete Room and Bonsai gates (below 0.04 degrees at p99), so these
post-fit rules are removed. Freezing SH during PBR support fitting lowers the
first fixed-cloud held-light result from 23.362/22.751 dB at 54.85% coverage to
23.144/22.444 dB at 53.39%; replacing its RGB L1 objective with MSE reaches
only 23.160/22.628 dB at 53.27%. Appearance/support co-adaptation remains
necessary; neither prototype survives to the five-cloud gate.

## Batched direct-Gaussian candidate preparation (2026-08-22)

Direct Gaussian fitting now records the next 20 deterministic ray batches in
one host-parallel candidate pass. Twenty is not a new training schedule: it is
the existing support-stage geometry synchronization interval. Between those
boundaries the CPU model, tile index, transforms, opacity, and support are
deliberately unchanged, so recording each batch in a separate scoped thread
group repeated setup without observing new geometry. Appearance-only fitting
uses the same bounded group size. Optimizer steps, random ray sequence, input
shapes, candidate sort, uploads, waits, and geometry refreshes are unchanged.
A unit test compares every index, mask, and depth from grouped preparation with
the same batches recorded separately across different worker partitions.

On the explicitly rebuilt 109,764-particle Room gate, preparation width one
takes 48.4 seconds for the paired PBR/static fit and scores the persisted PBR
Gaussian at 12.51/12.03 dB with 78.0% coverage. Width twenty takes 42.8 seconds
(11.6% faster) and scores 12.51/12.04 dB with 78.0% coverage; direct static
quality remains 15.04/15.01 dB. The final implementation keeps one combined
candidate table, regenerates only the cheap deterministic ray metadata per
optimizer step, and peaks at 347.9 MiB of scoped host memory.
On the 169,432-particle Bonsai gate, fit time falls from 64.1 to 61.0 seconds
(4.8%) while preserving 16.84/16.53 dB static quality and the selected
14.86/14.83 dB, 85.6%-coverage PBR result. All runs use the RTX 5070 inside the
12 GiB cgroup with zero swap, OOM, or GPU fault. This adds no dependency,
operation, graph, shader, model field, public option, or renderer path.

## Rejected direct-Gaussian depth concentration (2026-08-22)

A smooth layer-concentration objective is mathematically well behaved but does
not improve the final reconstruction. The prototype used the already sorted
Gaussian compositing weights and exact differentiable response depths to form
`opacity * E[t²] - E[t]²` per ray. Dividing by detached squared mean depth made
the term scene-scale stable without rewarding particles for moving away from
the cameras. An isolated GPU oracle confirmed zero loss for a single depth
layer, agreement within 2e-5 after scaling all depths by two, and gradients
that pull shallow and deep contributors together. It required no new shader,
Meganeura operation, or public option.

Fresh first-cloud production reconstructions at weights 0, 0.001, 0.003, and
0.01 score 23.3619/22.7520, 23.3652/22.7433, 23.3586/22.7401, and
23.3616/22.7459 dB mean/worst respectively after persist and reload. Coverage
stays within 0.006 percentage point. The small mean changes are within the
known atomic-gradient variation, while every nonzero setting lowers the worst
held-light view. The prototype is therefore removed before five-cloud and
real-scene gates. Concentrating contributors along each ray does not supply the
missing cross-view evidence needed to decide which image-consistent layer is
the true surface; future geometry work should improve correspondence or
visibility evidence rather than add another undirected compactness prior.

## Rejected static-field center feedback (2026-08-22)

The independently optimized static light field does contain a repeatable
synthetic center signal, but it does not transfer to real PBR geometry. The
prototype recorded exact shared-center correspondences before either fit,
allowing reordered and subset clouds without a nearest-neighbor guess. After
fitting, it moved each matched PBR Gaussian by 10% of the static displacement
tangent to the Gaussian's fixed local Y axis. Normal displacement is harmful
on all five clouds; discarding it makes every fixed synthetic mean and worst
view improve. Average gains are about 0.005 dB mean and 0.004 dB worst, with a
small coverage increase. A fresh paired coefficient-zero/0.1 reconstruction
confirms 23.3558/22.7490→23.3624/22.7535 dB after persist and reload, so this is
not merely a fixed-file scoring artifact.

The complete real gates nevertheless reject it. Room changes from
12.5158/12.0387 dB at 77.976% coverage to 12.5137/12.0384 dB at 77.950%;
Bonsai changes from 14.8592/14.8267 dB at 85.560% to 14.8569/14.8235 dB at
85.555%. Covered-pixel quality also falls on both. The matching table, host
update, constant, and analytical test are removed. Static radiance fitting can
learn a view-consistent tangent correction on the controlled fixture while
still absorbing texture or visibility error on photographs. It is therefore
not independent geometry evidence; a future shared-center update needs a
feature or correspondence objective that is invariant to appearance rather
than a residual copied from the light-field optimizer.

## Batched direct-Gaussian parameter readback (2026-08-22)

Profiling the selected paired Gaussian fit separates a CPU synchronization
bottleneck from training itself. On the 109,764-particle Room gate, the four
fit stages spend only 1.6 seconds in optimizer dispatch and wait, but the two
support stages spend 18.0 seconds reading parameters through six independent
host-visible mappings and 7.3 seconds rebuilding their candidate grids. The
foam trainer already uses Meganeura's cached bulk readback; the newer direct
Gaussian path had retained its original per-parameter implementation.

Each full synchronization now stages positions, log-scales, opacity logits,
and the three SH channels in one transfer. Intermediate candidate refreshes
stage only the first three because neither the candidate transform nor tile
membership reads SH; the exact final boundary still downloads all six. A pure
unit gate verifies that applying candidate geometry preserves appearance and
fixed covariance rotations. The optimizer schedule, batches, graph, candidate
order, model format, and public API are unchanged.

On a fresh same-machine Room control, paired Gaussian fitting falls from 43.6
to 26.1 seconds (40.1%). Bonsai falls from the selected 61.0-second baseline
to 31.9 seconds (47.7%) at 169,432 particles. Persisted held-light Gaussian
PBR remains 12.51/12.05 dB at 77.9% coverage on Room and 14.86/14.82 dB at
85.6% on Bonsai. A fresh five-cloud gate averages 23.2859/22.7023 dB and
55.071% coverage, inside the established atomic-continuation range of the
selected 23.2893/22.7108 dB and 55.094% baseline. The scopes peak at 364.5
MiB for Room, 497.3 MiB for Bonsai, and 258 MiB across the five-cloud run,
with zero swap, memory pressure, OOM, or GPU fault.

Two adjacent CPU changes are rejected and removed. Partially selecting the
nearest 64 hits before sorting takes 44.9 seconds on Room because most tile
lists already fit the cap; four-pixel tiles take 44.6 seconds because their
extra grid construction outweighs fewer per-ray candidates. The established
complete sort and eight-pixel tile remain.

## Cached direct-Gaussian camera terms (2026-08-22)

Candidate-grid construction projected every Gaussian support point through
the same view while repeatedly reconstructing and inverting its camera
quaternion and evaluating its field-of-view tangents. Ray sampling repeated
the corresponding forward quaternion and tangent setup for every sampled
pixel. The camera and capture are immutable throughout a fit, so projection
now caches the inverse transform and pinhole constants once per grid, while
ray generation caches the forward transform and constants once per view.
The equations, pixel centres, support bounds, candidate order, optimizer
schedule, graph, and persistence path are unchanged. Exact tests continue to
match tiled candidates against exhaustive traversal and grouped candidate
preparation against individual recording.

On a fresh 109,764-particle Room reconstruction, paired Gaussian fitting falls
from the bulk-readback baseline of 26.1 to 24.1 seconds, a further 7.7%, while
the persisted held-light PBR Gaussian scores 12.52/12.05 dB at 78.0% coverage.
The 169,432-particle Bonsai fit falls from 31.9 to 28.7 seconds, a further
10.0%, and scores 14.86/14.83 dB at 85.5% coverage. Relative to the original
pre-readback implementation, the combined reduction is 44.7% on Room and
53.0% on Bonsai.

A fresh five-cloud reconstruction and persistence gate averages
23.2882/22.7016 dB at 55.081% coverage and 22.2954 dB where hit, equivalent to
the selected 23.2893/22.7108 dB, 55.094% baseline. The Room, Bonsai, and
five-cloud reconstruction scopes peak at 376.8, 487.9, and 271.7 MiB with zero
swap, memory pressure, OOM, or GPU fault. This optimization adds no public API,
option, shader, graph operation, model field, file-format change, dependency,
or training schedule.

## Locality-ordered Gaussian candidate preparation (2026-08-22)

A temporary host profile of the current Room paired fit attributes 13.71 of
24.0 seconds to candidate preparation and another 5.64 seconds to candidate
grid construction. Optimizer execution takes 1.63 seconds, batch construction
and upload 0.39 seconds, and bulk parameter readback 1.24 seconds. Regenerating
the small ray metadata is therefore not the next bottleneck. Candidate
preparation instead consumed its twenty deterministic batches in their random
sample order, repeatedly moving between cameras and screen tiles even though
nearby rays read the same tile candidate list, camera-space origins, and
Gaussian transforms.

Each existing worker now evaluates its independent rays in `(view, tile)`
order and scatters the results back to their original rows. Worker partitions,
candidate membership, exact maximum-response evaluation, depth sorting, and
all optimizer inputs remain unchanged. The indexed-versus-exhaustive test
continues to compare every candidate index, mask, and depth in original ray
order. This is a private host locality change, not a training reorder.

The 109,764-particle Room paired fit falls from 24.1 to 21.5 seconds, a further
10.8%, while its persisted held-light PBR Gaussian remains at 12.52/12.05 dB
and 78.0% coverage. The 169,432-particle Bonsai fit falls from 28.7 to 25.9
seconds, a further 9.8%, while retaining 14.86/14.83 dB and 85.6% coverage.
Together with batched readback and cached camera terms, this halves the former
Room time (43.6 to 21.5 seconds) and reduces Bonsai by 57.5% (61.0 to 25.9
seconds).

A fresh five-cloud reconstruction and persistence gate averages
23.2826/22.7010 dB at 55.076% coverage and 22.2868 dB where hit, inside the
established unordered-atomic continuation range. Its small 2.2K-particle fits
average 4.57 seconds. Room, Bonsai, and the five-cloud scope peak at 363.4,
476.1, and 260.0 MiB respectively, with zero swap, memory pressure, OOM, or GPU
fault. The selected change adds no public API, setting, graph operation,
shader, model field, file format, dependency, or training schedule.

## Shared Gaussian projection support (2026-08-22)

After locality ordering, candidate-grid construction remains the largest
separable cost. Every one of the eighteen view workers independently validated
each Gaussian, normalized its covariance rotation, rotated the three 3DGUT
sigma axes, and derived the same opacity-dependent support radius. Only the
projection of those model-space terms is camera-dependent.

Each geometry refresh now constructs the Gaussian mean, three rotated sigma
axes, Gaussian cutoff radius, and conservative world radius once, then shares
that immutable table across the existing camera workers. The public conic
helper and candidate grid also use one sigma-axis implementation, removing a
duplicated construction path. Each camera still projects the same seven points
in the same order, accumulates the same covariance, applies the same near-plane
fallback, and emits the same tile lists. Exact indexed-versus-exhaustive and
grouped-versus-individual candidate tests continue to match every row.

The final compact cache stores only the mean and three axes. A prototype that
stored all seven world points is removed: it is at most 0.5 seconds faster in
one noisy Bonsai run but raises peak memory from 511 to 605 MiB. Reconstructing
the six cheap `mean ± axis` positions per view retains the expensive shared
quaternion work without that temporary allocation.

The 109,764-particle Room paired fit falls from 21.5 to 19.0 seconds, another
11.6%, while persisted held-light PBR remains 12.51/12.04 dB at 77.9%
coverage. The 169,432-particle Bonsai fit falls from 25.9 to 21.5 seconds,
another 17.0%, while retaining 14.86/14.83 dB at 85.6% coverage. Relative to
the pre-readback implementation, Room is now 56.4% faster and Bonsai 64.8%
faster.

A fresh five-cloud reconstruction and persistence gate averages
23.2831/22.6967 dB at 55.078% coverage and 22.2880 dB where hit, within the
established atomic-gradient continuation band. Its Gaussian fits average 4.53
seconds. Room, Bonsai, and five-cloud scopes peak at 377.5, 511.3, and 248.3
MiB with zero swap, memory pressure, OOM, or GPU fault. This introduces no
public API, option, graph operation, shader, model field, format, dependency,
or training schedule.

## Minimal Gaussian candidate rows (2026-08-22)

The private tiled recorder used by fitting retained four arrays even though
training consumed only two. Candidate depths are needed transiently to sort a
row, but no downstream graph input reads them; the candidate-to-pixel map is
fixed by the graph shape and was nevertheless rebuilt and uploaded for every
optimizer update. The tiled hot path now keeps only its ordered indices and
masks, and installs the invariant pixel map once when the session is created.
The public exhaustive recorder retains its depth and pixel diagnostics, so the
library API and its small correctness oracle do not change.

Exact tiled-versus-exhaustive and grouped-versus-individual tests still match
every candidate index and mask. The physical direct-fit oracle also converges
with the pixel map held across steps. Persisted Room remains at 12.51/12.04 dB
with 78.0% coverage; Bonsai remains at 14.86/14.82 dB with 85.6% coverage. A
fresh five-cloud gate averages 23.284/22.698 dB mean/worst, 55.06% coverage,
and 22.294 dB where hit, within the established continuation band.

This is selected as allocation cleanup rather than a broad speed claim. Room
is unchanged at 19.0 seconds and the sampled Bonsai run moves from 21.5 to
22.7 seconds, while the smaller five-cloud fits average 4.36 seconds versus
4.53 previously. The real-scene cgroup peaks move from 395.9 to 390.6 MB on
Room and from 536.2 to 431.7 MB on Bonsai. The complete five-cloud process
peaks in another reconstruction phase and rises from 260.4 to 296.2 MB, so it
does not corroborate a global peak reduction. All scopes report zero swap,
memory pressure, OOM, or GPU fault. The surviving code adds no option, shader,
graph operation, model field, format, dependency, or alternate training path.

## Current Blade and Meganeura revisions (2026-08-22)

The workspace now uses Blade `dd93e63`, the current upstream main revision,
for `blade-graphics`, `blade-macros`, and `blade-egui`. Meganeura moves from
`dece560` to `f20a464`, a three-commit descendant of its current main that
aligns those Blade dependencies, keeps compile-time gradient seeds host
visible, and retains the source-parallel scatter implementation already used
by this reconstruction branch. All Blade packages resolve to one revision.
These are branches in the original repositories, not additional forks.

The quality gate caught an important dependency-integration regression before
selection. The first bridge omitted the final scatter commit: two identical
small-cloud fits took 44.29 and 44.47 seconds instead of the established 4.4
seconds. Restoring source-parallel scatter reduces the same first-cloud fit to
4.29 seconds. The final five fits average 4.365 seconds and held-light
volumetric Gaussian PBR averages 23.284/22.698 dB mean/worst, 55.06% coverage,
and 22.292 dB where hit. This is inside the established quality and throughput
band rather than merely compiling against newer APIs.

Meganeura's complete 206-test library suite and blade-volume's complete
workspace/all-target suite pass. The latter peaks at 2.61 GB; the five-cloud
scope peaks at 253.5 MB. Both report zero swap, memory pressure, OOM, or GPU
fault. Vulkan validation still prints the three known Blade-main diagnostics:
workgroup `ArrayStride`, descriptor-pool sizing for binding arrays, and missing
device-address allocation flags. Their already-isolated Blade fixes are not
silently introduced through another dependency branch; the tests complete on
the physical RTX 5070 without device loss or numerical failure.

## Camera-space Gaussian projection (2026-08-22)

Candidate-grid construction projected the same 3DGUT sigma-point mean through
the camera once for the near-plane test and another seven times while building
the screen-space conic. A rigid camera transform is affine: the mean can be
transformed once and each of the three sigma axes once, after which the same
seven points are formed in camera space before the unchanged nonlinear pinhole
projection. This reduces eight quaternion-vector transforms to four per
Gaussian and view without replacing the unscented projection by a Jacobian or
changing its conservative support margin. A direct oracle checks that the
camera-space endpoint construction matches world-space projection.

On the 109,764-particle, 18-training-view Room gate, two candidate runs take
18.7 and 19.1 seconds for the paired PBR/static Gaussian fits. A same-source
control built from the parent commit takes 19.4 seconds, while the previously
accepted run was 19.0 seconds. This is retained as a small exact compute
cleanup rather than a broad throughput claim. The candidate persists at
12.51--12.52/12.04--12.05 dB with 78.0% coverage, versus 12.51/12.04 dB and
78.0% for the paired control. All three scopes peak below 492 MiB with zero
swap, memory pressure, OOM, or GPU fault. The change adds no public API,
setting, graph operation, shader, model field, format, dependency, or training
schedule.

The independent per-view grids now use the CPU parallelism reported by the
process instead of stopping at eight requested workers. With 18 views, the
old `div_ceil` partition produced six three-view workers on this 12-thread
machine; the uncapped partition produces nine two-view workers. Two fresh Room
runs fall further to 17.8 and 17.7 seconds, retain 12.52/12.04--12.05 dB and
78.0% coverage, and peak below 358 MiB with zero swap, pressure, OOM, or GPU
fault. The worker count remains bounded by the number of views, and this is a
private scheduling change rather than a new pool or policy setting.

## Rejected direct volumetric PBR scale refinement (2026-08-22)

The final anisotropic PBR Gaussian was tested against its own exact runtime
renderer instead of the scalar support proxy. Eight deterministic antithetic
whole-cloud log-scale proposals perturbed the 1,202 observed particles, rebuilt
the runtime Gaussian, and accepted only a lower masked training-view sRGB loss.
At a 2.5% log step, two proposals are accepted and loss falls from 0.0045896 to
0.0045774 in 7.043 seconds. The held-light result is 23.36/22.75 dB at 54.9%
coverage and 22.49 dB where hit, versus the 23.36/22.75 dB, 54.8%, and 22.50 dB
control. At a 5% step, one proposal reaches the same final loss in 6.842 seconds,
but held-light quality moves to 23.34/22.75 dB at 55.0% coverage and 22.47 dB
where hit.

Lowering the final renderer's aggregate training error therefore still does not
localize particle responsibility: the proposals broaden support without
improving covered relighting quality. The temporary scoring API, optimizer, and
synthetic integration are removed. A viable exact-runtime refinement needs
localized per-particle scale responsibility and an efficient in-place Gaussian
geometry update, rather than whole-cloud perturbations followed by renderer
reconstruction.

That localized path was then implemented and gated before selection. An
in-place Gaussian-buffer and TLAS refresh reduced eight exact-runtime rounds
from about seven seconds to 0.33 seconds for one light. Antithetic per-particle
directions were inferred only from the error difference inside each projected
ellipsoid. Nevertheless, a 2.5% step lowers training loss from 0.0045929 to
0.0044537 while held-light quality falls to 23.33/22.70 dB, 55.1% coverage,
and 22.45 dB where hit. A 1% step is less destructive but still reaches only
23.36/22.74 dB, 55.0%, and 22.49 dB.

Using all four non-held lighting environments does not make the signal
geometric: at 1%, the held result is 23.35/22.73 dB, 54.9%, and 22.48 dB.
Removing explicit mask coverage from that multi-light objective produces the
closest trade, 23.36/22.74 dB at the control's 54.8% coverage and 22.51 dB
where hit, but exchanges 0.01 dB of worst-view quality for 0.01 dB where-hit
quality. This is below the measured continuation noise and cannot justify a
new optimizer or runtime update API. The localizer, updater, public statistics,
and synthetic integration are all removed; only this negative gate remains.

## Rejected direct-Gaussian scheduling shortcuts (2026-08-22)

A phase profile of the current 169,432-particle Bonsai fit confirms that host
candidate work, not GPU optimization, is still dominant. Across its four fit
stages, candidate-row preparation takes 11.4 seconds and the two support
stages spend 7.4 seconds downloading geometry and rebuilding view grids. GPU
dispatch and wait take 2.3 seconds total; deterministic ray sampling and input
upload take less than 0.3 seconds. Removing the second ray-metadata generation
would therefore optimize well under one percent of this run.

Refreshing learned support every 40 updates halves the expensive grid rebuilds
and reduces the established 18-training-view Bonsai fit to 16.5 seconds, but
persisted Gaussian PBR coverage falls from 85.6% to 82.4%. A 25-update refresh
is less stale and takes 18.0 seconds, yet coverage falls further to 81.8%.
Static light-field PSNR remains within 0.03 dB and Gaussian PBR PSNR does not
fall, showing that both variants exchange missing support for easier covered
pixels rather than preserving the reconstruction. The selected 20-update
cadence is restored.

An adjacent attempt to remove the indexed candidate recorder's explicit
eight-worker cap was initially removed as a no-op for an individual 512-ray
batch: the existing minimum of 64 rays per worker independently limits that
call to eight workers. That conclusion applies to the individual audit and
final-loss batches, but not to the later grouped candidate preparation; the
grouped case is revisited below.

## Rejected localized foreground ownership (2026-08-22)

Foreground masks were tested as a localized support signal instead of the
already rejected whole-ray opacity target. At each candidate refresh, a masked
training ray below the final renderer's 50% coverage threshold assigned its
missing-opacity loss only to the particle with the largest current
front-to-back compositing weight. Already covered foreground rays, background
supervision, colour loss, candidate order, and the static light field remained
unchanged. CPU and physical-GPU oracles verified the owner choice and exact
loss contribution without adding a Meganeura operation or shader path.

The paired first-cloud result raises volumetric coverage by 0.1 point with
identical 23.37/22.75 dB held-light quality. Across five clouds, however, mean
and covered-pixel PSNR each fall by 0.004 dB while coverage rises only 0.08
point. The third cloud loses 0.03/0.03 dB mean/worst and the fourth loses its
tail. Assigning one existing contributor is more precise than forcing every
foreground mixture, but the mask still says nothing about which depth sheet
should own the ray. The owner table, graph input, objective, and tests are
removed; PBR mask supervision remains negative-only.

## Rejected depth-ordered Gaussian tiles (2026-08-22)

The private candidate grid was tested with each view tile sorted once by
camera-space Gaussian-centre depth. Exact maximum-response depth still sorted
every retained ray row with the existing `(depth, particle)` tie break, so the
23-test Gaussian suite proved candidate indices and masks identical to the
exhaustive oracle. The intent was only to give Rust's adaptive unstable sort a
nearly ordered input.

On the exact 169,432-particle, 18-training-view Bonsai A/B, however, the paired
fit rises from 20.8 to 25.3 seconds while static quality remains 16.84/16.54 dB.
The per-ray hit lists are already short; repeatedly sorting duplicated tile
membership at every geometry refresh costs more than it saves. The tile order
change is removed before Room and five-cloud gates. Future candidate work
should reduce hit evaluation itself rather than add another persistent order.

## Selected squared-distance candidate cutoff (2026-08-22)

The indexed candidate recorder no longer evaluates a Gaussian exponential only
to discard its value after thresholding. For each particle it precomputes the
exact equivalent of `opacity * exp(-distance_squared / 2) >= min_alpha` as
`distance_squared <= -2 * ln(min_alpha / opacity)`. The exhaustive oracle and
the differentiable renderer still evaluate the original response; this changes
only private host-side indexed culling and adds no operation, shader, API, or
training schedule.

The exact indexed-versus-exhaustive and grouped-versus-individual row tests
remain green, with an additional boundary test for the algebraic cutoff. On the
169,432-particle Bonsai gate the fit falls from 19.5--20.8 to 16.1 seconds
(17--23%); Room falls from 17.9 to 14.3 seconds (20%). Across five fresh small
synthetic clouds, average fit time falls from 4.341 to 4.114 seconds (5.2%).
Their held-light average remains in the established continuation band at
23.284/22.700 dB, 55.08% coverage, and 22.286 dB where hit, compared with
23.284/22.698 dB, 55.06%, and 22.292 dB for the selected dependency-uprev
baseline. The full physical-GPU workspace suite passes in the 12 GiB cgroup,
peaking at 6.30 GB with no swap, OOM, or GPU fault.

## Rejected compact Gaussian transform cache (2026-08-22)

After per-view Gaussian origins are built, the indexed ray loop does not read
the mean retained in each 48-byte candidate transform. A prototype kept only
the 36-byte inverse matrix in that hot cache while leaving the public response
oracle and every floating-point operation unchanged. All 24 Gaussian tests,
including exact indexed-versus-exhaustive rows, passed.

Order-balanced Room fits averaged 14.55 seconds for the control and 13.95 for
the compact cache, while Bonsai averaged 16.15 and 15.80 seconds. The effect
does not transfer meaningfully to the production-sized synthetic gate: five
fits average 4.094 seconds versus the selected 4.114-second control, only 0.5%.
Real-scene cgroup peaks are inconsistent and the five-cloud peak is unchanged
at about 281 MB. Held-light quality remains within atomic continuation noise.
Carrying a second internal transform representation is therefore not worth a
sub-percent production gain; the prototype is removed.

## Rejected larger controlled-light material palettes (2026-08-22)

The selected predefined-light pipeline was screened at 12, 16, and 24 shared
diffuse materials on the same reconstructed cloud. Geometry, calibrated normal
passes, Gaussian support fitting, final material/radius/normal refinement, and
the unseen `studio` evaluation light were otherwise identical. Increasing the
palette lowers the observation-space clustering residual slightly and improves
the scalar surface fallback from 22.69/22.13 to 22.72/22.23 dB.

The durable full-covariance Gaussian result moves the other way. Twelve
materials reach 23.36/22.75 dB at 54.8% coverage and 22.50 dB where hit;
sixteen reach 23.32/22.68 dB at 54.9% and 22.46 dB where hit; twenty-four reach
23.31/22.64 dB at 54.9% and 22.42 dB where hit. Peak memory also rises from
241 to 316 MB across the screen. More clusters let the scalar initializer
explain local mixture errors, but those assignments do not remain physical
when the learned ellipsoids overlap. The existing 12-material controlled-light
gate remains selected and no new option or code path is added.

## Selected calibrated-light Gaussian geometry continuation (2026-08-22)

The relightable Gaussian now uses repeated measured-light captures for one
final position-only continuation. For each light, the current PBR normal and
diffuse material predict one fixed SH-0 color per corresponding particle. The
ordinary image-formation fit then moves centers for 50 updates while keeping
covariance, opacity, and appearance fixed. Lights are visited forward and
backward so the final capture does not become an order prior. The scratch
colors are discarded and the learned Gaussian coefficients restored before
the asset is attached to PBR data and serialized. An error rolls both scratch
appearance and partial center changes back.

This reuses the existing Gaussian graph and optimizer: it adds no Meganeura
operation, shader, bind group, model field, CLI setting, or runtime path. The
production binary enables it only when a relightable Gaussian was requested
and at least one aligned `--normal-images` / `--normal-environment` pair joins
the primary measured light. A two-light Bonsai production smoke completed 200
updates, persisted and reloaded the PBR cloud, and scored it through the normal
viewer renderer.

On the exact five fixed synthetic clouds, the prior selected result was
23.284/22.700 dB mean/worst, 55.08% coverage, and 22.286 dB where hit. Four
calibrated lights and 400 symmetric updates at a `2e-4` position rate reach
23.344/22.770 dB, 55.26%, and 22.320 dB. Every cloud improves or ties its mean
and worst view; one newly covered subset trades 0.02 dB where hit while the
other four and the aggregate improve. The continuation averages 3.067 seconds
per cloud. Its 12 GiB cgroup peaks at 248 MB with no swap, OOM, or GPU fault.
The result is intentionally modest: nearest-truth material and normal
substitutions previously showed that poor cross-view geometry correspondence,
not a missing specular material lobe, is the current volumetric PBR limiter.
Independent calibrated lights provide a genuine geometry signal without
baking one light into the persisted asset.

A production dependency check also matters for this path. Meganeura `main`'s
column-parallel scatter takes 3.6--3.7 seconds for the same 200-update Bonsai
smoke; its source-parallel CAS path takes 1.4--1.5 seconds and preserves the
same output scores. The retained Meganeura branch is therefore current `main`
plus only Blade revision alignment, constant-gradient host visibility, and
source-parallel scatter accumulation.

Allowing the same continuation to update anisotropic log-scales at `2e-4` is
rejected. Across the five fixed clouds it reaches 23.346/22.766 dB, 55.38%
coverage, and 22.308 dB where hit: the mean changes by only +0.002 dB while
worst-view and hit-region quality regress. The second cloud alone loses
0.02/0.01/0.02 dB in mean/worst/hit-region quality. The temporary scale update
is removed; calibrated lights continue to refine centers only.

Stronger center rates are screened and rejected too. Rates of `3e-4` and
`4e-4` both reach 23.354 dB mean over five clouds, but trade quality between
clouds. The `3e-4` schedule averages 22.788 dB worst, 55.30% coverage, and
22.330 dB where hit, yet cloud 2 repeatedly loses mean and hit-region quality.
The `4e-4` schedule averages 22.790 dB worst, 55.38%, and 22.330 dB, while
clouds 2 and 5 lose mean quality and cloud 5 loses worst-view quality. These
noise-sized aggregate gains do not justify a less robust default; `2e-4`
remains selected.

A balanced consensus alternative is also rejected before a five-cloud run.
It fits every calibrated light from the same round baseline and averages the
independent center deltas, eliminating sequential ordering without changing
the graph. On the first fixed cloud it falls to 23.40/22.79 dB at 54.9%
coverage, versus the selected 23.43/22.87 dB at 55.0%. Averaging removes too
much of the useful cumulative geometry update; the prototype is removed.

Doubling the light-switch frequency at fixed work is rejected as well. Sixteen
25-update fits over four forward/reverse passes score 23.41/22.85 dB at 54.9%
coverage on the first cloud, below the selected 23.43/22.87 dB at 55.0%.
Repeated graph/session construction also raises continuation time from about
3.1 to 5.5 seconds. The selected eight 50-update fits remain both faster and
more accurate.

Projecting each calibrated-light center update onto its particle normal is
rejected. Although this preserves a surfel's tangential identity, the first
fixed cloud falls to 23.38/22.78 dB at 54.9% coverage from 23.43/22.87 dB at
55.0%. The fused initializer contains meaningful lateral correspondence error,
so unrestricted three-dimensional center motion remains necessary. The
temporary host-side projection is removed.

## Selected joint calibrated-light Gaussian fit (2026-08-23)

The calibrated-light continuation now compiles one image-formation graph and
keeps one Adam state for all lights. Every update swaps only the fixed SH-0
table predicted by that light's environment, then feeds the corresponding RGB
capture and camera rays. A forward/reverse light schedule removes an endpoint
prior. Each sequence number is used exactly once by every light, so aligned
captures supervise matching random rays without repeating the old two-pass
batches. Candidate grids remain private per capture and refresh together every
20 center updates. Learned appearance is still restored on exit and the public
asset/API is unchanged.

This is simpler at the system boundary than eight independent fits: one graph,
one optimizer, no new Meganeura operation or shader, no bind group, no model
field, and no CLI option. The explicit schedule helper is unit-tested. The
first prototype accidentally repeated each ray sequence twice; it tied mean
quality but lost 0.01 dB where hit and is superseded by unique paired batches.

On the same five fixed clouds, the selected sequential control is
23.344/22.770 dB, 55.26% coverage, and 22.320 dB where hit. The joint fit reaches
23.352/22.778 dB, 55.30%, and 22.322 dB. Every cloud's mean and worst view
improves or ties. Continuation time falls from 3.067 to 0.920 seconds on
average, a 3.33x speedup, while peak memory remains 248 MB with no swap, OOM,
or GPU fault.

The real two-light Bonsai smoke also completes 200 updates, persists and
reloads the Gaussian PBR asset, and improves its training score from
11.30/11.09 to 11.35/11.13 dB. Where-hit quality rises from 11.69 to 12.89 dB;
coverage changes from 23.0% to 22.9%. The continuation itself falls from
1.4--1.5 seconds to 0.5 seconds. This replaces the sequential implementation.

Native gradient accumulation across all lights is tested and rejected. Four
corresponding-light micro-batches are averaged before each Adam update, with a
fourfold global rate preserving approximately the same total displacement.
This is a true simultaneous light objective and needs no new graph operation,
but the five-cloud result is mixed: 23.356/22.778 dB, 55.30% coverage, and
22.318 dB where hit. Mean rises by 0.004 dB while hit-region quality loses
0.004 dB; cloud 2 regresses in mean, worst view, and where-hit quality. Runtime
only falls from 0.920 to 0.889 seconds. The temporary accumulation schedule is
removed, retaining per-light Adam updates in the shared session.

Per-light RGB gain fitting for the fixed scratch appearance is also rejected.
A deterministic 4,096-ray least-squares probe first fit the premultiplied
Gaussian render to each measured capture. Its gains of roughly 1.05 improved
the first cloud's worst view by 0.01 dB but reduced where-hit quality by
0.02 dB, showing that the scalar was compensating opacity as well as exposure.
Fitting unpremultiplied color reduced the gains to 1.01--1.03, but the same
cloud still reached only 23.45/22.87 dB, 55.0% coverage, and 22.55 dB where
hit, versus 23.45/22.87 dB, 55.0%, and 22.56 dB for the selected uncalibrated
scratch table. Both prototypes are removed. Real exposure calibration needs
capture metadata or a calibrated reference, rather than a render-space scalar
that is not identifiable separately from volumetric opacity and transport.

Updating opacity in the joint calibrated-light continuation is rejected too.
The corrected probe enables the existing opacity logits at a conservative
`1e-3` rate, leaving the preceding support fit and every other setting equal.
On the first fixed cloud it reaches 23.45/22.86 dB, 55.2% coverage, and
22.54 dB where hit, versus 23.45/22.87 dB, 55.0%, and 22.56 dB with frozen
opacity. Extra coverage comes at the expense of tail and covered-pixel
fidelity: calibrated RGB residuals can still change transmittance to compensate
imperfect material, normal, and transport predictions. The temporary rate is
removed before a five-cloud gate; the dedicated single-light support stage
remains authoritative for opacity.

Scalar projected-center observation counts are not a safe confidence gate for
the volumetric continuation. Restoring centers with fewer than two primary-
capture observations keeps only 320 of 2,323 learned displacements on the
first cloud and lowers held-light output to 23.41/22.84 dB, 54.9% coverage,
and 22.54 dB where hit. The control reaches 23.45/22.87 dB, 55.0%, and
22.56 dB. Finite overlapping Gaussians receive useful mixture gradients even
when the older scalar observer does not assign their projected center to two
views. The rollback prototype is removed; future confidence must be measured
from the Gaussian responsibilities themselves, not borrowed from a different
renderer.

The selected continuation now weakly conditions foreground color supervision
on the frozen accumulated opacity. Ordinary premultiplied RGB can otherwise
move a center to repair an opacity-magnitude error when the scratch PBR color
is imperfect, even though opacity itself deliberately remains under the
dedicated support fit. For a masked foreground ray, the conditioned share is
`0.5 * stop_gradient(opacity)` and compares the render against
`target * stop_gradient(opacity)`; the rest is the ordinary residual. Poorly
covered pixels therefore keep most of the coverage-driving objective. Known
background rays keep their ordinary residual exactly. With no masks,
`target_alpha` is zero and the two terms algebraically collapse to the original
loss, so real capture behavior is unchanged.

A fixed conditioned share was first screened at full, half, quarter, and
eighth strength, reaching 23.47/22.94, 23.48/22.93, 23.46/22.89, and
23.47/22.89 dB on the first cloud. The eighth blend was the first balanced
selection and reached 23.368/22.800 dB, 55.28% coverage, and 22.334 dB where
hit over five clouds. Scaling the share by detached opacity improves that
balance. Ceilings of 0.25, 0.5, and 1.0 reach 23.46/22.89 at 55.0% and 22.56
where hit, 23.47/22.91 at 55.0% and 22.57, and 23.48/22.92 at 54.9% and 22.57.
Squaring opacity reaches only 23.47/22.90 at 54.9% and adds another graph node,
so both stronger shapes are rejected.

The selected 0.5 ceiling reaches 23.386/22.812 dB, 55.22% coverage, and
22.346 dB where hit over the definitive five clouds. A fresh fixed-eighth
control reaches 23.364/22.792 dB, 55.28%, and 22.330 dB. All five means
improve; four tails improve and one changes by -0.01 dB. Against the original
fixed five-cloud selection, every mean, tail, and covered-pixel score improves
or ties. Continuation averages 0.942 seconds and peaks at 247 MB with no swap,
OOM, or GPU fault. The earlier two-light maskless Bonsai production smoke
remains an exact loss identity at 11.35/11.13 dB, 22.9%, and 12.89 dB where
hit. This changes no model field, public option, Meganeura operation, shader,
or runtime representation.

The cleaner objective also changes the center-rate gate. At `3e-4`, every
five-cloud mean and tail improves over `2e-4`, reaching 23.408/22.848 dB,
55.30% coverage, and the same 22.346 dB where hit. The selected `4e-4` rate
then improves every mean, every tail, every coverage value, and every
covered-pixel value relative to `3e-4`; it reaches 23.428/22.874 dB, 55.34%,
and 22.366 dB. First-cloud probes continue upward through `6e-4` and `1e-3`,
but the `1e-3` five-cloud gate regresses cloud 3 by 0.04/0.05 dB mean/tail
against `4e-4`; a repeat reaches the same 23.22/22.55 dB boundary. At `2e-3`,
the first-cloud mean and covered score also turn downward. The robust `4e-4`
knee is retained rather than the slightly higher aggregate but scene-mixed
`1e-3` result.

The repeated-image Bonsai integration smoke remains intentionally unsuitable
as a fidelity gate because its second environment is paired with the same RGB
capture. At `4e-4` it nevertheless completes 200 maskless updates, persists
and reloads the asset, and changes training-view output from 11.35/11.13 dB,
22.9%, and 12.89 dB where hit to 12.04/11.73 dB, 22.9%, and 12.44 dB. This is
reported as a stability check and explicit covered-region trade, not evidence
of real-scene relighting quality. Runtime remains 0.6 seconds and peaks at
243 MB with no swap, OOM, or GPU fault.

Longer calibrated-light continuation is rejected after the rate retune. On the
first cloud, 75 steps per light reaches 23.55/23.05 dB, 55.2% coverage, and
22.60 dB where hit, while 100 steps turns downward to 23.54/23.01 dB and
22.57 dB where hit despite another 0.1 coverage point. Across five clouds, 75
steps reaches 23.432/22.886 dB, 55.46%, and 22.354 dB where hit, versus
23.428/22.874 dB, 55.34%, and 22.366 dB for 50 steps. The noise-sized
mean/tail gains trade covered fidelity, cloud 3 regresses by 0.01/0.04 dB, and
cloud 5 loses 0.05 dB where hit. Continuation time also rises from 0.934 to
1.238 seconds. The temporary 75/100-step constants are removed and the robust
50-step schedule remains selected.

Replacing the diffuse scratch table with directional PBR appearance is also
rejected. The first prototype evaluates the exact CPU PBR model over 64 sphere
directions and projects it into degree-2 SH. On the first fixed cloud it
reaches the same rounded 23.52/23.00 dB, 55.1% coverage, and 22.59 dB where-hit
result as the selected diffuse continuation, but position fitting grows from
about 0.9 to 2.490 seconds. A smaller prototype fits degree-1 SH only at the
actual training-view directions; it reaches 23.52/22.99 dB, 55.1%, and
22.59 dB in 2.448 seconds. Both pay for specular environment prefiltering and
extra SH fitting without improving the held-light result. They are removed
before the five-cloud gate. This closes missing directional scratch appearance
as the immediate error source: the next geometry investigation should measure
responsibility in the Gaussian compositor itself, where overlapping particles
share each residual.

That compositor-responsibility follow-up is measured and rejected as a center
rollback gate. Over 10,240 deterministic training rays, a particle counts as
responsible in a view when its exact front-to-back compositing weight reaches
1%. The first cloud has median/p90/p99 support of 3/6/6 training views and
responsibility-mass quantiles of 0.32/5.53/27.88. Restoring the pre-continuation
centers of the 719 particles with zero responsible views changes the held-light
result from 23.53/23.00 dB, 55.1%, and 22.59 dB where hit to 23.52/22.99 dB,
55.1%, and 22.60 dB. Requiring two views restores 864 centers and reaches
23.52/22.99 dB, 55.1%, and 22.59 dB. The one-view boundary merely exchanges a
hundredth of a decibel between global and hit-region scores, while the stronger
gate is non-positive. The temporary collector and rollback are removed before
a five-cloud gate; responsibility is descriptive here, not reliable enough to
justify another production heuristic.

Foreground-mask supervision during calibrated-light center continuation is
also rejected. Unlike the preceding RGB objective, the silhouette is
lighting-independent, and keeping opacity frozen initially appeared to make
it clean geometry evidence. It does not make ownership unique among
overlapping particles. On the first fixed cloud, adding mask MSE at weight
`0.01` changes the selected 23.53/23.00 dB, 55.1% coverage, and 22.60 dB
where-hit result to 23.52/22.99 dB, 55.1%, and 22.59 dB. Weight `0.1` falls
again to 23.51/22.98 dB and 22.58 dB where hit, without changing coverage.
Both constants and the graph connection are removed before the five-cloud
gate. The dedicated support fit remains responsible for masks; the later
center pass remains color-only and opacity-conditioned.

Deterministically oversampling the same silhouette evidence is rejected as
well. A private sampler devoted one eighth of each aligned calibrated-light
batch to the two-sided four-neighbour mask boundary while leaving the other
seven eighths on the selected uniform sequence. It reused the existing RGB
objective, and maskless captures were exactly unchanged. On the first fixed
cloud it falls from 23.53/23.00 dB, 55.1% coverage, and 22.60 dB where hit to
23.48/22.94 dB, 55.1%, and 22.58 dB. Concentrating updates on a narrow image
boundary weakens the interior multi-view correspondence without recovering
support. The sampler and its test are removed before a five-cloud gate.

An aligned-light contrast objective is also rejected. The private scratch
renderer stayed in the same graph but subtracted 25% of the next calibrated
light's predicted diffuse appearance and matching RGB capture from every
ordinary target. This cancels part of the light-independent material and
opacity residual while retaining 75% of the original signal. Across five
clouds it reaches exactly the selected 23.428 dB mean, but changes the worst
view from 22.874 to 22.866 dB, coverage from 55.34% to 55.30%, and where-hit
quality from 22.366 to 22.382 dB. Clouds 3 and 4 regress in both mean and tail;
the aggregate covered-region gain is therefore another mixed trade rather
than a robust geometry improvement. Raising the reference share to 50% on the
first cloud lowers its tail from 23.00 to 22.97 dB while reaching only 22.62
dB where hit. The signed scratch path, paired labels, validation branch, and
tests are removed; the selected continuation keeps ordinary per-light RGB.

Refreshing shading normals from the continued Gaussian centers is rejected.
After the selected position updates, a point-cloud-only PCA fit used twelve
nearest centers, aligned each fitted plane to the established photometric
normal, and applied only a 10% normalized blend before PBR attachment. It did
not rotate covariance or change positions, support, materials, or the scalar
surface. On the first fixed cloud the held-light result falls from
23.53/23.00 dB and 22.60 dB where hit to 23.51/22.97 dB and 22.57 dB, with
coverage unchanged at 55.1%. The moved volumetric centers still contain
overlapping depth sheets, so Euclidean neighbours do not define a cleaner
surface manifold than the existing multi-view normal. The helper is removed
without tuning a smaller blend toward an identity.

Aligned calibrated-light captures now share one pixel-ray table and one
Gaussian candidate index. Candidate membership depends on particle geometry
and camera calibration, not captured radiance or illumination, so rebuilding
the same structure four times was redundant. The entry point now checks the
documented alignment invariant exactly before sharing either cache. On the
first fixed cloud the continuation takes 0.866 seconds and retains
23.52/22.99 dB, 55.1% coverage, and 22.59 dB where hit, within the established
atomic support-fit variation. This removes no training signal and adds no
shader, Meganeura operation, model field, or public option.

The selected calibrated-light continuation now compares radiance in linear
space when every selected view has a foreground mask. Those masks provide an
independent coverage constraint, so the residual no longer needs sRGB's extra
weight near black. Across the five fixed clouds this reaches approximately
23.438/22.904 dB, 55.28% coverage, and 22.386 dB where hit, versus
23.428/22.874 dB, 55.34%, and 22.366 dB for the display-referred control.
Every cloud's tail and covered-pixel score improves; every mean improves or
ties, while aggregate coverage changes by -0.06 percentage point. Maskless or
mixed captures deliberately retain the established sRGB residual because
they have no independent alpha evidence. The repeated-image Bonsai stability
smoke therefore preserves its exact 12.04/11.73 dB, 22.9% coverage, and 12.44
dB where-hit result over 200 updates. This changes only the private scratch
appearance and labels used by the position continuation; the durable
appearance, model, renderer, public API, shaders, and Meganeura graph
vocabulary are unchanged. The five-cloud logs are under
`target/audit-runs/current-synthetic-v1/multilight-linear-five/`, and the
maskless replay is under
`target/audit-runs/multilight-masked-linear-production/`.

The existing rendered-material opt-in now closes the final ordering gap with
one Gaussian-specific polish after calibrated-light center continuation and
scalar-radius feedback. It attaches the final PBR attributes, builds the exact
volumetric Gaussian tracer, then coordinate-descends only the shared diffuse
albedo table at steps 0.025 and 0.0125. The earlier observation/linear solve
and scalar-render polish remain its initializer; repeating either would
discard an already selected solution. The helper updates the tracer's existing
material buffer in place and copies the accepted table only into the Gaussian
model, leaving the independently fitted surface output unchanged.

Across the five fixed clouds this raises held-light quality from approximately
23.438/22.904 dB and 22.386 dB where hit to 23.496/22.926 dB and 22.486 dB
where hit. Every cloud improves its mean, worst view, and covered-pixel score;
coverage remains exactly 55.28% because geometry, opacity, and assignments are
fixed. The pass lowers its exact Gaussian training objective on every cloud
and takes 1.54--1.84 seconds for twelve materials. Steps 0.00625 and 0.0125 are
positive but leave gains unused; 0.05 lowers the first-cloud tail from the
selected 23.04 to 23.02 dB. The repeated-image maskless production smoke also
improves from 12.04/11.73 dB, 22.9% coverage, and 12.44 dB where hit to
12.06/11.74 dB, 23.2%, and 12.53 dB. The pass runs only when both the existing
material-refinement opt-in and calibrated-light Gaussian continuation are
active. It adds no shader, graph operation, binding, model field, file format,
or CLI option. The definitive logs are under
`target/audit-runs/current-synthetic-v1/post-continuation-gaussian-material-025-five/`
and `target/audit-runs/post-continuation-gaussian-material-production/`.

Extending that final polish to all four calibrated training lights is
rejected. A prototype kept one exact Gaussian tracer per aligned capture and
minimized their pixel-count-weighted sRGB loss while changing the same shared
diffuse table. Across the five fixed clouds it reaches approximately
23.436/22.880 dB, 55.28% coverage, and 22.410 dB where hit, below the selected
single-light polish at 23.496/22.926 dB, 55.28%, and 22.486 dB where hit.
Every mean falls, and four of five tails and covered-pixel scores fall. The
joint objective also takes 5.99--7.28 seconds instead of 1.54--1.84 seconds.
The auxiliary lights are valuable for geometry because their changing
shading disambiguates position, but averaging them into the final small
diffuse palette pulls that deliberately low-capacity appearance model away
from the unseen-light solution. The extra evidence type, tracers, and joint
loss are removed. Logs remain under
`target/audit-runs/current-synthetic-v1/joint-final-gaussian-material-five/`.

Linear radiance is also rejected for the final single-light Gaussian material
polish, even when every selected view has a mask. Geometry and support are
fixed here, so it does not reproduce the earlier coverage collapse, but the
first fixed cloud still falls from the selected 23.59/23.03 dB and 22.71 dB
where hit to 23.51/22.97 dB and 22.61 dB where hit at the same 55.0% coverage.
Its own linear training loss falls from 0.0023666 to 0.0023314, confirming that
the regression is objective choice rather than a failed optimizer. The sRGB
objective's extra emphasis on low radiance remains useful for the final
appearance table, unlike the independently masked center continuation. The
temporary linear scorer and dispatch are removed. The diagnostic log is under
`target/audit-runs/current-synthetic-v1/final-gaussian-material-linear-probe/`.

Reversing the coordinate order only for the half-step Gaussian polish is an
identity at the first-cloud gate: 23.58/23.03 dB, 55.0% coverage, and 22.71 dB
where hit, equal to the selected forward schedule at reported precision. It
changes 32 rather than 30 coordinates without improving held-light output or
runtime. The symmetric-order prototype is removed before a five-cloud gate;
the simpler one-order loop remains shared with scalar material refinement.
Its log is under
`target/audit-runs/current-synthetic-v1/final-gaussian-material-reverse-half-isolated-probe/`.

Restricting final material error to independently masked foreground pixels is
rejected as well. On the first fixed cloud it reaches 23.54/23.00 dB and
22.65 dB where hit, below the selected 23.59/23.03 dB and 22.71 dB, with the
same 55.0% coverage. Although geometry is frozen, background pixels still
constrain the colour carried by low-opacity Gaussian tails at the silhouette;
discarding them weakens rather than cleans the material signal. The masked
scorer is removed before a five-cloud gate. Its log is under
`target/audit-runs/current-synthetic-v1/final-gaussian-material-foreground-probe/`.

An exact-render normal continuation after the final Gaussian center move is
also rejected. The prototype added a shading-record-only Gaussian upload and
reused the existing antithetic, locally projected normal search over all four
calibrated lights. At 2.5 degrees the first cloud lowers its training
objective in all eight rounds but regresses to 23.57/23.02 dB and 22.68 dB
where hit. Halving the step to 1.25 degrees reaches approximately
23.504/22.934 dB, 55.28% coverage, and 22.496 dB where hit over five clouds,
versus 23.496/22.926 dB, 55.28%, and 22.486 dB for the selected model. Those
roughly 0.01 dB aggregate gains are mixed: cloud 4 regresses in mean, tail,
and covered-pixel quality, and cloud 1 loses mean. The pass also takes
1.14--1.36 seconds. This does not justify a new renderer update API and about
two hundred lines of optimizer plumbing, so the API, continuation, and call
sites are removed. The five-cloud logs remain under
`target/audit-runs/current-synthetic-v1/post-center-gaussian-normal-125-five/`;
the larger-step diagnostic is under
`target/audit-runs/current-synthetic-v1/post-center-gaussian-normal-probe/`.

Re-enabling anisotropic scale learning under that new objective remains a
mixed trade. At scale rates `1e-4` and `2e-4`, the five-cloud aggregates reach
approximately 23.442/22.908 dB, 55.28% coverage, and 22.382 dB where hit, and
23.444/22.908 dB, 55.26%, and 22.366 dB respectively. The center-only control
reaches 23.438/22.904 dB, 55.28%, and 22.386 dB. Thus the tiny mean/tail gains
come with weaker covered-pixel fidelity, and the stronger rate also loses
coverage. Projecting every particle's learned log-scale delta to preserve its
initial volume is much worse on the first cloud, falling to 23.22/22.67 dB,
54.5%, and 22.49 dB where hit. Shape and support are not separable in the
overlapping compositor. All scale updates and the projection helper are
removed; logs remain under
`target/audit-runs/current-synthetic-v1/multilight-linear-scale-lr1e4-five/`,
`target/audit-runs/current-synthetic-v1/multilight-linear-scale-lr2e4-five/`,
and `target/audit-runs/current-synthetic-v1/multilight-linear-scale-shape-lr2e4-probe/`.

Linear radiance is deliberately not extended backward into the Gaussian PBR
support fit. That stage still has to discover opacity from foreground RGB;
the masks contribute only negative background evidence so that overlapping
particles are not all forced opaque. A first-cloud prototype that trained its
appearance, opacity, and scales entirely in linear space collapsed volumetric
coverage from about 55% to 32.4% and held-light quality to 19.42/19.28 dB.
Covered pixels rise to 23.50 dB only because the difficult low-opacity region
disappears. The prototype is removed, and the established sRGB support
objective remains under the later masked-linear, frozen-support geometry
continuation. Its diagnostic log is under
`target/audit-runs/current-synthetic-v1/pbr-support-linear-probe/`.
Even a 5% interpolation from sRGB toward linear is negative on the first
cloud: final volumetric quality falls from the selected approximately
23.53/23.03 dB, 55.0% coverage, and 22.62 dB where hit to 23.50/22.99 dB,
54.7%, and 22.60 dB. The temporary blended sampler is removed too; smaller
weights would converge toward the selected identity, while this arm provides
no evidence that the trade reverses. Its log is under
`target/audit-runs/current-synthetic-v1/pbr-support-linear-blend-5pct-probe/`.

Foreground importance sampling does not make the final center continuation
more efficient. A private deterministic sampler reserved one quarter of each
512-ray batch for pixels that are foreground in every calibrated capture and
left the other three quarters on the exact selected uniform sequence. On the
first cloud it falls to 23.47/22.96 dB, 55.3% coverage, and 22.55 dB where hit,
from approximately 23.53/23.03 dB, 55.0%, and 22.62 dB. The extra silhouette
interior updates weaken background/spill control and change the objective's
measure rather than adding independent correspondence. The sampler, common-
foreground table, and constants are removed. Its log is under
`target/audit-runs/current-synthetic-v1/multilight-linear-foreground-quarter-probe/`.

Anchoring that continuation to its initial centers is rejected. A private
mean-squared displacement term at weight one reaches approximately
23.422/22.868 dB, 55.32% coverage, and 22.356 dB where hit over the five fixed
clouds, slightly below the selected 23.428/22.874 dB, 55.34%, and 22.366 dB
on every aggregate. Raising the weight to ten lowers the first cloud to
23.51/22.97 dB and 55.0% coverage. The initial centers still contain genuine
cross-view error, so a global trust region suppresses useful corrections
before it distinguishes photometric drift. The graph input and loss are
removed; a future geometric constraint needs particle-specific confidence or
independent correspondence evidence.

Privately pre-fitting one disposable SH-0 appearance table per calibrated
light is rejected too. Fifty frozen-geometry updates per light leave the
first-cloud held-light mean and tail at the selected 23.53/23.00 dB, but lower
covered-pixel quality from 22.60 to 22.58 dB and raise continuation time from
roughly 0.87 to 1.05 seconds. The nuisance fit absorbs part of the same
cross-view residual needed to move centers; the PBR-predicted diffuse tables
remain the stronger geometry supervision. The prefit loop and temporary
tables are removed before a five-cloud gate.

## Validation-clean dependency stack (2026-08-23)

The workspace now pins Blade `3d01645`, current `main` plus three isolated
Vulkan fixes: valid Naga workgroup layouts, the device-address allocation
flag for external memory, and descriptor-pool counts that include binding
arrays. Meganeura `52f5cc6` is current `main` plus Blade/Naga alignment,
constant-gradient host visibility, and source-parallel scatter accumulation.
All Blade crates resolve to the same revision and Naga resolves once.

Blade's focused graphics suite, Meganeura's 206 library tests plus optimizer
and training integration tests, and blade-volume's full workspace/all-target
suite pass on the RTX 5070. The blade-volume suite peaks at 5.5 GB inside its
12 GB cgroup and reports no swap, OOM, GPU fault, Vulkan validation warning,
or VUID. Three superseded Blade validation branches were deleted. The one
final Blade branch and one Meganeura dependency branch remain necessary until
these commits are merged into their respective `main` branches.

The earlier integration pull requests are merged, but they predate these
August 23 fixes. A direct retest of Blade `20fcb84` (the merged integration
revision) with Meganeura `3dc9621` compiles, then reports invalid workgroup
`ArrayStride`, undersized descriptor pools, and missing device-address
allocation flags before Meganeura aborts a surface-only graph test on an
invalid `copy_nonoverlapping`. This is deterministic inside the 12 GB cgroup
on the stable RTX 5070 and is not a power-supply failure. The two remaining
heads therefore cannot be removed merely by repinning to the already-merged
integration revisions.

## Rejected weak aggregate foreground undercoverage (2026-08-23)

A weak one-sided foreground term does not close the remaining Gaussian PBR
coverage gap cleanly. The private prototype added 5% of squared missing
aggregate opacity on masked foreground rays to the selected background-only
support loss. It did not penalize excess foreground opacity, choose an owning
particle, add a graph operation, or change the static light field.

Across the five calibrated synthetic clouds, average volumetric coverage rises
from 55.28% to 55.42%, but held-light mean and worst-view quality fall by
0.006 and 0.008 dB, and covered-pixel quality falls by 0.036 dB. The third
cloud loses 0.05/0.06 dB mean/worst and the fourth loses 0.02/0.02 dB. The
extra support therefore admits marginal pixels without reconstructing them
better, consistent with the earlier rejected localized-ownership experiment.
The foreground term is removed and PBR masks remain negative-only. Artifacts
are under
`target/audit-runs/current-synthetic-v1/pbr-foreground-undercoverage-5pct-{first,five}/`.

## Refreshed truth-surface bound and roughness gate (2026-08-23)

The current Blade fixture's exact-surface relighting bound invalidates the old
component-substitution interpretation without making nearest-truth particle
matches admissible again. Fitting diffuse albedo and the specular lobe on
133,924 actual multi-view surface samples reaches 22.86 dB under held-out
`studio`; exact albedo plus the true lobe reaches 23.01 dB, and fitting only
albedo with true roughness and F0 reaches 22.92 dB. A uniform one-degree normal
tilt already lowers the fitted result to 22.29 dB, and two degrees reaches
21.32 dB. Normal accuracy is therefore still high leverage, but the final
Gaussian's 23.53 dB whole-image mean at only 55.28% volumetric coverage is not
directly comparable with the fully covered truth surface; its 22.52 dB
covered-pixel score is the honest local reference. The complete bound is under
`target/audit-runs/current-synthetic-v1/relight-bound-refresh/`.

A forward-identical attempt to keep unmodelled highlights from steering the
joint normal fit is rejected. It scaled only the normal gradient by each
particle's fitted roughness squared, leaving fully rough gradients, center
gradients, rendered colors, operations, and optimizer budget unchanged. Across
five clouds, coverage is identical and mean/covered-pixel PSNR are neutral to
0.002 dB, but average worst-view PSNR falls by 0.006 dB. The fourth cloud loses
0.02/0.01 dB mean/worst and 0.04 dB where hit. Fitted roughness is itself
coupled to approximate normals and shared material clusters, so it is not a
reliable confidence signal. The graph input and gradient filter are removed;
artifacts remain under
`target/audit-runs/current-synthetic-v1/roughness-weighted-normal-{first,five}/`.

## Rejected Gaussian candidate micro-optimizations (2026-08-23)

Two further private candidate-recorder changes are deliberately not retained.
Preallocating every screen tile to the active-particle count divided by the
tile count removes some small-vector growth, but fresh 109,764-particle Room
and 169,432-particle Bonsai fits remain exactly 14.6 and 16.0 seconds, the same
as their paired controls. It also reserves storage for off-screen supports and
for tiles that receive fewer than the view average. The extra allocation policy
has no measured return.

Precomputing each camera-space Gaussian origin's squared length and evaluating
the minimum ray distance as `|o|² + dot(o,d) * depth` also fails the gate. The
exact indexed-versus-exhaustive and grouped-versus-individual candidate-row
tests pass, but Room changes only from 14.6 to 14.5 seconds and Bonsai from
16.0 to 15.9 seconds. More importantly, the algebraically equivalent form has
different floating-point rounding at the support boundary and changes the
Bonsai continuation enough to move held-view quality and coverage. This is not
a lossless hot-loop cleanup. Both prototypes are removed; their cgrouped logs
remain under `target/audit-runs/candidate-grid-reserve/` and
`target/audit-runs/gaussian-quadratic-distance/`.

## Full-CPU grouped Gaussian candidate preparation (2026-08-23)

Grouped direct-Gaussian preparation now uses the process's available CPU
parallelism instead of retaining the recorder's older eight-worker ceiling.
The minimum of 64 rays per worker still bounds small calls: the individual
512-ray audit and final-loss batches remain at eight workers. The production
path, however, prepares twenty unchanged 512-ray training batches together,
so its 10,240 independent rays can use all twelve hardware threads on the test
machine. Candidate membership, per-ray locality order, exact hit arithmetic,
row scatter, optimizer inputs, and training order do not change.

Reverse-order controls confirm that this is not a warm-run effect. The
109,764-particle Room paired fit falls from 14.6--15.4 seconds to 12.4 seconds
(15--19%), and the 169,432-particle Bonsai fit falls from 16.0--16.5 seconds to
14.1 seconds (12--15%). The exact grouped-versus-individual candidate-row test
continues to match every index and mask. Persisted Room held-view Gaussian PBR
remains approximately 14.04/12.91 dB with 78.8% coverage, and Bonsai remains
12.38/9.98 dB with 85.2% coverage, within the controls' existing atomic-update
variation. The cgroup scopes peak at 336.5 MiB and 487.5 MiB with no swap, OOM,
GPU fault, validation warning, or Xid. The selected implementation removes one
private scheduling cap and adds no API, option, dependency, graph operation,
shader, binding, model field, or format.

The five fixed synthetic clouds average 23.496/22.924 dB, 55.28% coverage,
and 22.484 dB where hit after the complete calibrated-light continuation and
persist/reload gate, matching the selected 23.496/22.926 dB, 55.28%, and
22.486 dB baseline. Their smaller paired fits average 4.225 seconds; the extra
workers target high-particle production captures rather than changing the
small-scene schedule.

Balancing the separate 18-view grid rebuild across all twelve reported threads
is rejected. Replacing the existing nine two-view chunks with six two-view and
six one-view chunks raises the same Room fit from 12.4 to 14.4 seconds. These
workers stream large per-particle projection tables, so logical-thread
occupancy is not the right target on the six-core test CPU. The balanced
partition is removed before a Bonsai gate; its diagnostic remains under
`target/audit-runs/balanced-view-grid-workers/`.

Repeating the ridge-regularized affine albedo solve at the final exact-Gaussian
material pass is rejected. The existing solve is valuable before scalar
render refinement, but at this later point the table has already survived two
complete-render coordinate passes and the final Gaussian geometry differs only
slightly. Over the five fixed clouds, repeating the solve before the selected
two coordinate steps reaches approximately 23.498/22.934 dB, 55.30% coverage,
and 22.484 dB where hit, versus 23.496/22.926 dB, 55.28%, and 22.486 dB for
coordinate polish alone. The tiny aggregate tail change is mixed: cloud 4
loses mean and covered-pixel quality, and cloud 2 loses covered-pixel quality.
The pass also grows from roughly 1.5--1.8 to 2.1--2.5 seconds. Lower exact
training loss is not enough to select another initializer at the final
ordering boundary, so the one-boolean prototype is removed. Logs remain under
`target/audit-runs/current-synthetic-v1/final-gaussian-linear-init-probe/`.

Adding a worst-training-view term to the final Gaussian material objective is
rejected too. At 25% weight, the five-cloud aggregate is effectively tied at
approximately 23.500/22.930 dB, 55.28% coverage, and 22.486 dB where hit, but
cloud 2 regresses in mean, tail, and covered-pixel quality and cloud 3 loses
its tail. Reducing the term to 10% does not recover the discriminating second
cloud: it reaches only 23.61/23.00 dB and 22.56 dB where hit, below the
selected 23.64/23.02 dB and 22.62 dB. Balancing the known-light training
cameras is not a proxy for unseen-light novel-pose robustness. The alternate
loss, private parameter, and call branches are removed; logs remain under
`target/audit-runs/current-synthetic-v1/final-gaussian-view-balance-{quarter,tenth}-probe/`.

## Grouped RGB basis for rendered material fitting (2026-08-23)

The accepted affine material initializer now renders one white diffuse basis
per shared material instead of one basis per RGB coordinate. PBR diffuse
transport is channel-diagonal: the red albedo affects only red radiance, and
likewise for green and blue. Splitting the three channels of one `[1,1,1]`
render therefore produces the same three response columns while reducing the
12-material basis from 36 production renders to 12. The intercept, normal
equations, ridge, exact sRGB acceptance, and following coordinate descent are
unchanged.

The physical runtime recovery test still reaches the known albedo. On the
first fixed cloud, the grouped and scalar bases report identical material
losses (`0.0069934 -> 0.0063249`) and the complete held-light result remains
23.58/23.03 dB at 55.0% coverage and 22.71 dB where hit. In reverse order, the
accepted rendered-material stage takes 1.420 seconds versus 1.487 seconds for
the scalar control, a 4.5% reduction. The change adds no API, option, shader,
operation, binding, pipeline, model field, format, or dependency. Logs remain
under `target/audit-runs/current-synthetic-v1/rendered-material-rgb-basis-{probe,control-after}/`.

## Incremental affine material scoring (2026-08-23)

The rendered-material coordinate passes now reuse that same affine diffuse
basis instead of dispatching a complete GPU render for every lower and upper
albedo proposal. One black-diffuse render supplies the fixed specular/base
term, and one white-diffuse render per material supplies all three RGB
responses. The implementation stores that response once rather than as three
sparse coordinate vectors. A proposal changes only its matching RGB channel,
so its sRGB error is updated over one third of the pixels while the other two
channels and their error sum stay untouched. The accepted table is uploaded
once and must still lower a fresh production render; otherwise the entire
coordinate pass is rolled back. Palettes above the existing 96-coordinate
linear-solve boundary retain the old direct-render fallback.

The five fixed synthetic clouds retain the selected aggregate at
23.496/22.926 dB, about 55.3% coverage, and 22.48 dB where hit. Their initial,
post-support, and final material passes take roughly 0.79--0.86 seconds; the
first-cloud paired control takes 1.404, 1.382, and 1.545 seconds respectively.
The physical known-material recovery test still reaches the exact authored
albedo.

The larger 18-view gates show the intended scaling. Room's initial and
post-support passes fall from 2.9 and 2.1 seconds to 0.6 and 0.6 seconds;
Bonsai falls from 2.5 and 1.9 seconds to 0.6 and 0.6 seconds. Room's persisted
Gaussian test result is unchanged at 14.34/12.89 dB, 78.8% coverage, and
13.70 dB where hit. Bonsai remains 12.28/9.78 dB at 85.1% coverage; its
0.01 dB where-hit movement is within the paired direct-Gaussian atomic
variation. Peak scoped memory is 416 MB on Room and 603 MB on Bonsai, versus
355 and 514 MB for the controls, with no swap, OOM, validation warning, Xid,
or GPU fault. The change is private CPU scoring and adds no API, option,
shader, graph operation, binding, pipeline, model field, format, or
dependency. Logs remain under
`target/audit-runs/rendered-material-incremental-basis-{room,bonsai}/` and
their `-control-after` siblings; the fixed-cloud gate is under
`target/audit-runs/current-synthetic-v1/rendered-material-compact-basis-five/`.

## Rejected Gaussian support splitting (2026-08-23)

Three private point-cloud densification variants are rejected. Splitting the
largest observation-weighted tangent supports by half an extent, shrinking
their footprint, and conserving coincident opacity lowers the first fixed
cloud from the selected roughly 23.58/23.03 dB at 55.0% coverage to
23.53/22.97 dB at 54.7%. Replacing that heuristic with the conventional
multi-light `position-gradient magnitude * tangent extent` signal and
splitting the top 10% around the midpoint of the existing 400-update budget is
worse at 23.41/22.92 dB and 54.3% coverage. A zero-split midpoint graph-rebuild
control reaches 23.57/23.04 dB and 55.0%, isolating the loss to the split rather
than the Adam restart.

A conservative split preserves scale, combined opacity, and almost the same
rendered field by separating the two children only 0.1 tangent sigma. Across
the five fixed clouds it reaches approximately 23.512/22.928 dB and 55.88%
coverage, versus the selected 23.496/22.926 dB and 55.28%. That apparent
coverage gain is blur: quality where both images hit falls from 22.486 to
22.330 dB, and the persisted particle count and frame cost both rise by about
10%. The split primitive, gradient readback, graph rebuild, synthetic-only
call path, and tests are removed. Future densification needs a child
initialization that preserves both footprint and local detail, plus an
independent pruning or opacity continuation stage; adding overlapping copies
is not geometry recovery. Logs remain under
`target/audit-runs/current-synthetic-v1/densify-{25-first,gradient-10-first,rebuild-control-first,gradient-gentle-first,gradient-gentle-five}/`.

## Final low-opacity Gaussian compaction (2026-08-23)

Persisted relightable Gaussians now discard particles with learned peak
opacity below 0.05 after geometry and material fitting are complete. The pass
is a compact CPU remap of points, every SH row, rotation, scale, PBR normal,
and material index; the shared material table is retained. It rejects clouds
carrying adjacency or other surface-specific point semantics instead of
silently invalidating their indices. Training topology and every optimizer
remain unchanged.

A paired replay over the five previously persisted synthetic outputs removes
846 of 11,565 particles (7.3%). Mean and average worst-view quality remain
23.428/22.865 dB at reported precision, coverage changes by less than 0.01
percentage points, and where-hit quality changes by less than 0.001 dB. The
paired renderer sample falls from 1.362 to 1.184 ms/frame, a 13.1% reduction.
The full current pipeline independently removes 847 particles and remains
within ordinary atomic-fit variation of the selected aggregate at
approximately 23.492/22.924 dB, 55.26% coverage, and 22.478 dB where hit.

The 18-view real gates keep the same result. Room removes 15,580 of 109,764
particles (14.2%) and reports 14.34/12.90 dB, 78.7% coverage, and 13.70 dB
where hit, versus 14.34/12.89 dB, 78.8%, and 13.70 for the control. Bonsai
removes 10,391 of 169,432 particles (6.1%) and remains 12.28/9.78 dB at 85.2%
coverage and 12.51 dB where hit. The gates peak at 393 and 500 MB respectively
with no swap, OOM, validation error, Xid, or GPU fault. The change adds no
option, shader, graph operation, binding, format field, or dependency. Logs
remain under `target/audit-runs/opacity-prune-{five,005-room,005-bonsai}/` and
`target/audit-runs/current-synthetic-v1/opacity-prune-005-five/`.

The same threshold is deliberately not applied to the static light field.
It improves every fixed synthetic mean and tail while removing 3.5--4.2% of
those SH-0 clouds, but the exact paired SH-2 Room field loses 0.056 dB mean and
0.092 dB worst-view quality when 17.4% is removed. A 0.025 threshold still
loses 0.015/0.031 dB while removing 6.7%. At 0.01, Room and Bonsai are
quality-neutral to 0.001 dB, but only 1.2% and 0.1% of their particles are
removed. That negligible cross-scene saving does not justify a second output
policy. Static fields retain their low-opacity directional tails; only the
independently gated PBR cloud is compacted.

A refreshed nearest-truth component substitution is also rejected as a
quality diagnostic for the final joint model. On the first current cloud,
replacing only centers, normals, or materials by the Euclidean-nearest
training-truth sample lowers the 23.59 dB baseline to 20.40, 23.52, and
22.37 dB respectively. The fitted fields compensate one another around an
imperfect center correspondence, so changing one field through a nearest
spatial match breaks the represented surface rather than measuring its
independent error. Future component bounds need the particle's actual
multi-view surface correspondence; nearest truth is no longer admissible.

## Rejected alternating Gaussian normal refresh (2026-08-23)

Alternating calibrated-light center fitting with a moved-center normal refresh
is rejected. The prototype split the existing 400 position updates into four
equal rounds. After each round it copied the moved Gaussian centers into the
surface model, re-observed every calibrated-light capture there, and ran the
existing 512-candidate per-view normal solver. This kept the total center
optimization budget fixed and refreshed 1,398 normals on the first fixed
cloud.

The result falls to 23.30/22.73 dB, 54.9% coverage, and 22.41 dB where hit,
from the selected approximately 23.59/23.05 dB, 55.0%, and 22.71 dB where
hit. It also takes about 1.98 seconds instead of roughly 0.9 seconds for the
ordinary center pass. Moving centers changes which observed surface sample a
particle represents; refreshing only normals at the new coordinates breaks
the jointly fitted appearance/material correspondence. The prototype is
removed before a five-cloud gate. A future alternating method must update the
whole coupled surface state from explicit multi-view correspondences, not
re-sample one attribute independently. The rejected output remains under
`target/audit-runs/current-synthetic-v1/alternating-center-normal-first/`.

## Rejected post-continuation patch correspondence (2026-08-23)

Re-running the existing point-cloud patch matcher after calibrated-light
Gaussian center continuation is rejected. The prototype moved the complete
corresponding surface state, rather than refreshing normals independently: it
started each patch search at the final Gaussian center, retained source-depth
visibility, searched one quarter of the particle radius along its established
normal, then copied accepted centers back before PBR attachment and final
material polish. No truth geometry or held-out view entered the pass.

Only 154 of 2,323 particles carry four sufficiently textured,
depth-consistent training views. The ordinary 2% acceptance threshold moves
121 of them and lowers the first fixed cloud from the selected approximately
23.59/23.03 dB and 22.71 dB where hit to 23.57/22.97 dB and 22.69 dB where
hit. Requiring a 10% patch-cost reduction still moves 97 centers, with a 23.5%
average reduction among scored particles, but reaches only 23.57/22.97 dB and
22.70 dB where hit. Coverage stays at 55.0%. Thus even strong local patch
agreement does not identify the correct center of an overlapping volumetric
mixture: the matcher assumes one tangent surface sheet. The prototype is
removed rather than tuned toward an identity. Logs remain under
`target/audit-runs/current-synthetic-v1/post-center-correspondence-{first,strict-first}/`.

## Rejected coarse-to-fine calibrated-light continuation (2026-08-23)

A half-resolution warm start for the final calibrated-light Gaussian center
fit is rejected. The prototype area-averaged every aligned RGB capture and
mask, rebuilt the exact projected Gaussian candidate index at that resolution,
and spent the first quarter of the unchanged 400-update budget there before
returning to full resolution. It retained one graph, optimizer, light schedule,
support, opacity, and appearance, so only the spatial scale of the first 100
targets changed.

On the first fixed cloud it reaches 23.58/23.02 dB, 55.1% coverage, and
22.68 dB where hit, versus the selected approximately 23.59/23.03 dB, 55.0%,
and 22.71 dB where hit. Coarse supervision marginally broadens support but
weakens the relit covered surface; the following 300 full-resolution updates
do not recover that detail. The downsampler, second ray table, second candidate
index, and schedule branch are removed. The diagnostic remains under
`target/audit-runs/current-synthetic-v1/multilight-coarse-quarter-first/`.

## Independent static PowerFoam continuation (2026-08-23)

The optional masked surface-PowerFoam continuation now belongs only to the
static light-field branch. Both production `reconstruct` and the synthetic
gate clone the established Gaussian surface before the continuation; learned
radii and normals seed the direct static Gaussian, while learned density and
SH remain in the optional PowerFoam static output. The PBR surface keeps its
independently selected photometric normals, materials, and support path. The
old orchestration mutated the shared surface in production
and therefore changed PBR geometry while the direct static Gaussian could keep
an earlier clone—the opposite of the option's intended output ownership.

On the five current fixed clouds, 300 updates per view raise direct-Gaussian
held-view mean PSNR from 24.98/24.90/24.70/25.21/25.02 dB to
25.15/25.08/25.26/25.32/25.01 dB. Worst-view PSNR changes from
24.26/23.87/24.24/24.39/24.31 to 24.40/24.11/24.29/24.64/24.21 dB. The
aggregate gains are 0.202 dB mean and 0.168 dB worst-view. The separately
persisted/reloaded PBR outputs remain at approximately 23.50/22.93 dB,
55.3% coverage, and 22.48 dB where hit in aggregate; per-cloud movements are
at the established atomic-fit scale because their inputs and code path are
unchanged.

The continuation remains opt-in rather than becoming a default. Cloud 5 loses
0.01 dB mean and 0.10 dB worst-view static quality. Halving the schedule to
150 updates per view is worse at 24.96/24.13 dB on that cloud; doubling it to
600 reaches only 24.99/24.21 despite a substantially lower training loss.
Training convergence therefore cannot safely select the continued field for
every scene. Explicit users receive the strong static improvement without the
former PBR side effect; the ordinary default remains unchanged. A focused
orchestration test proves mutating the selected static continuation target
does not mutate the PBR surface. Five-cloud logs and 0.30 GB peak-memory
telemetry remain under
`target/audit-runs/current-synthetic-v1/surface-powerfoam-static-only-{first,five}/`;
the schedule probes are under the neighboring `-150-cloud5` and `-600-cloud5`
directories. All scopes report zero swap, OOM, validation error, Xid, or GPU
fault. The complete 511-test workspace/all-target physical-GPU gate passes at
a 3,408,805,888-byte cgroup peak; formatting and warnings-as-errors clippy also
pass. Validation logs remain under
`target/audit-runs/static-powerfoam-output-isolation-validation/`.

## Rejected calibrated-light Gaussian specular polish (2026-08-23)

Exact-render coordinate refinement of the shared Gaussian roughness/F0 table
is rejected. The prototype kept final centers, covariance, opacity, normals,
material assignments, diffuse albedo, and all four calibrated environments
fixed. It prepared one production Gaussian tracer per aligned light and
coordinate-descended twelve shared roughness values plus three F0 channels per
material at two step sizes. Thus it tested whether multi-light evidence makes
specular parameters identifiable without changing geometry or adding a
training approximation.

The four-light training loss falls from 0.0054151 to 0.0052844 and 32 of 96
visited coordinates change in 7.27 seconds. Held-light output nevertheless
falls from the selected roughly 23.58/23.03 dB and 22.70 dB where hit to
23.51/22.99 dB and 22.64 dB where hit, at the same 55.0% coverage. An isolated
roughness arm, with F0 effectively frozen, lowers its training loss from
0.0054127 to 0.0053660 but is an exact held-quality identity at reported
precision while still taking 7.30 seconds. F0 therefore compensates known-light
errors actively, and roughness adds cost without demonstrated generalization.
At the current roughly 55-degree normal error, calibrated repeats still do not
make the specular lobe durable. The optimizer, evidence reload, call path, and
about 170 lines of API/plumbing are removed; production retains the selected
rough-dielectric prior. Logs remain under
`target/audit-runs/current-synthetic-v1/multilight-gaussian-{specular,roughness}-first/`.

## Rejected final-normal Gaussian covariance alignment (2026-08-23)

The learned Gaussian covariance frame is not forcibly rotated onto the final
explicit PBR normal. Direct Gaussian training starts from the extracted
surface frame and then freezes covariance rotation while later support-aware
surface refinement changes the shading normal. A prototype preserved all
learned axis lengths and rotated each covariance as a whole so that its local
Y axis matched that final normal before attaching PBR data. This removes a
semantic frame mismatch without introducing another learned parameter.

On the first fixed cloud it changes 1,202 supported covariance frames. The
selected held-light volumetric Gaussian result is roughly 23.58/23.03 dB
mean/worst, 55.0% coverage, and 22.70 dB where hit. Alignment produces
23.58/23.02 dB, the same 55.0% coverage, and 22.70 dB where hit. The slight
worst-view regression and otherwise exact identity do not justify mutating
geometry after its image-space optimization, so the helper, call path, and
test are removed without expanding the run to five clouds. The complete run
and cgroup telemetry remain under
`target/audit-runs/current-synthetic-v1/covariance-normal-alignment-first/`.

## Joint calibrated-light Gaussian normal fit (2026-08-23)

The selected calibrated-light Gaussian continuation now optimizes explicit
diffuse shading normals together with particle centers when every aligned
capture supplies a foreground mask. The existing anisotropic compositor
evaluates the exact nine-term diffuse-irradiance basis from normalized learned
normals, fixed shared albedos, and each known environment. It compares linear
radiance, retains the established forward/reverse light schedule and Adam
state, and keeps support, opacity, covariance, materials, and durable Gaussian
appearance fixed. The learned SH field is still restored after continuation;
only centers and normalized PBR normals persist. Maskless captures continue
through the previous display-referred position-only path.

At the selected `5e-4` normal rate, the five fixed clouds change held-light
volumetric Gaussian mean PSNR from 23.58/23.63/23.33/23.58/23.36 dB to
23.59/23.66/23.37/23.59/23.38 dB. Worst-view PSNR changes from
23.03/23.02/22.67/23.03/22.88 to 23.05/23.05/22.74/23.05/22.90 dB, and
where-hit quality changes from 22.70/22.58/22.21/22.56/22.35 to
22.72/22.62/22.26/22.58/22.37 dB. Thus every cloud improves all three quality
measures; aggregate gains are approximately +0.022/+0.032/+0.030 dB while
coverage is effectively unchanged. On a fresh first-cloud replay,
nearest-truth normal RMSE falls from 55.18 to 54.03 degrees and the joint pass
takes 0.886 seconds, matching the old position-only runtime after caching
albedo and irradiance inputs.

A `1e-3` normal rate is rejected despite lowering first-cloud normal RMSE to
53.05 degrees: it overfits the calibrated lights and lowers held-light output
to 23.56/23.00 dB and 22.67 dB where hit. The selected conservative rate is
therefore an image-quality choice rather than a truth-normal oracle. The
implementation adds no Meganeura operation, shader, shader group, entry
variant, bind group, model field, file property, dependency, or public option;
it folds into the existing multi-light entry point. A physical-GPU graph test
checks the SH-2 diffuse value and its finite normal gradient, an integration
test exercises the masked path, and a Bonsai production smoke preserves the
maskless fallback exactly. Benchmark artifacts and cgroup telemetry remain
under `target/audit-runs/current-synthetic-v1/joint-center-normal-*` and
`target/audit-runs/joint-center-normal-production-fallback/`.

## Rejected calibrated-normal anchor prior (2026-08-23)

A normalized chord-distance prior does not make the stronger calibrated-light
normal update generalize. The prototype raised the joint normal rate from the
selected `5e-4` to `1e-3` and penalized mean squared distance from each
pre-continuation unit normal in the same graph. It kept the renderer, center
rate, ray schedule, calibrated environments, material polish, and 400-update
budget unchanged.

On the first fixed cloud, anchor weight 0.1 limits nearest-truth normal RMSE to
54.23 degrees rather than the unanchored high-rate result's 53.05 degrees, but
held-light output still falls to the same 23.56/23.00 dB and 22.67 dB where
hit. Weight 1.0 further limits motion and reaches 54.91 degrees, yet only
recovers 23.58/23.04 dB and 22.70 dB where hit. Both remain below the selected
low-rate result's 23.59/23.05 dB, 22.71 dB where hit, and 54.03-degree normal
RMSE. The harmful high-rate direction is therefore not explained by excessive
angular displacement from the previous normal. The extra graph input, loss,
constant, and higher rate are removed. Runs and capped telemetry remain under
`target/audit-runs/current-synthetic-v1/joint-center-normal-anchor-{01,1}-first/`.

## Selected aligned-light Gaussian normal contrast (2026-08-23)

The calibrated-light Gaussian continuation now ends with 25 normals-only
updates per light on full aligned-light differences. For light `i`, the same
graph receives diffuse irradiance `E_i - E_(i+1)` and corresponding linear
labels `I_i - I_(i+1)` at identical camera rays. Gaussian weights,
transmittance, albedo, and opacity are shared by the pair, so their common
magnitude error cancels more directly than in another absolute-light update.
Centers are frozen for this tail. It reuses the compiled graph, Adam state,
candidate index, schedule, and existing operations; there is no new shader,
entry, bind group, model field, file property, dependency, or public option.

Against the selected joint-normal five-cloud control, mean held-light PSNR
changes from 23.59/23.66/23.37/23.59/23.38 dB to
23.59/23.67/23.40/23.61/23.38 dB. Worst-view PSNR changes from
23.05/23.05/22.74/23.05/22.90 to 23.06/23.06/22.77/23.07/22.91 dB, and
where-hit quality changes from 22.72/22.62/22.26/22.58/22.37 to
22.71/22.62/22.28/22.61/22.38 dB. Coverage is identical for every cloud.
Aggregate mean/worst/where-hit gains are approximately
+0.012/+0.016/+0.010 dB; every mean and tail improves or ties. A repeat of the
first cloud resolves its one rounded where-hit trade at 23.59/23.05 dB and
22.72 dB where hit, exactly matching the selected image scores while lowering
nearest-truth normal RMSE from 54.03 to 53.69 degrees. The tail raises the
joint pass from roughly 0.89 to 1.01 seconds.

Twelve updates per light are rejected: the first-cloud normal RMSE reaches
53.86 degrees, but held-light tail quality falls to 23.04 dB rather than the
25-update repeat's 23.05 dB. The selected tail is therefore the smallest
screened duration that preserves the fixed-cloud image gate. Runs and capped
telemetry remain under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-*`.

A symmetric mean-of-other-lights reference is rejected. It replaces the
cyclic partner with the average diffuse irradiance and aligned labels from
every other calibrated capture, removing capture-order dependence at the same
100-update budget. On the first fixed cloud it reaches 23.59/23.04 dB and
22.71 dB where hit, below the retained cyclic repeat's 23.59/23.05 dB and
22.72 dB where hit. Nearest-truth normal RMSE is also slightly worse at 53.75
rather than 53.69 degrees. The additional reference samples and averaging are
removed without a five-cloud expansion. The capped diagnostic remains under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-mean-first/`.

Choosing each light's most distant diffuse-irradiance reference is rejected as
well. It retains one reference and the selected runtime, but replaces cyclic
pairing with the environment having the largest squared SH-2 irradiance
distance. The first fixed cloud ties mean/tail at 23.59/23.05 dB, while
where-hit quality falls to 22.71 dB and normal RMSE worsens to 53.86 degrees;
the cyclic repeat reaches 22.72 dB and 53.69 degrees. Maximizing global light
separation overemphasizes a less transferable differential direction. The
selector is removed without a five-cloud expansion. Its capped diagnostic is
under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-farthest-first/`.

Stronger opacity conditioning in the normals-only contrast tail is rejected.
The ordinary joint pass retains its selected 0.5 ceiling, while the tail can
raise it without moving centers, covariance, opacity, or coverage. A full 1.0
ceiling ties the selected five-cloud mean and covered-pixel aggregates exactly,
keeps every coverage value identical, and lowers average worst-view PSNR by
0.002 dB. Clouds 3 and 4 regress; cloud 4 loses 0.01 dB tail and 0.03 dB where
hit. The intermediate 0.75 ceiling fails the first-cloud screen at
23.59/23.05 dB and 22.71 dB where hit, versus the selected repeat's
23.59/23.05 dB and 22.72 dB, while normal RMSE worsens from 53.69 to 53.73
degrees. Both one-line batch scalings are removed. Runs remain under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-{full-opacity,three-quarter}-*`.

Resetting Adam at the contrast boundary is rejected too. Zeroing only the
normal first/second moments and restarting bias correction after centers are
frozen reaches 23.59/23.05 dB and 22.71 dB where hit on the first cloud, while
nearest-truth normal RMSE worsens to 53.76 degrees. The selected inherited
state reaches 23.59/23.05 dB, 22.72 dB where hit, and 53.69 degrees. Its second
moment therefore supplies useful damping despite the loss transition. The
three reset calls are removed; the diagnostic is under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-reset-first/`.

A differentiable SH-2 specular approximation is rejected before a five-cloud
expansion. The private graph evaluated each candidate's reflection direction,
looked up three particle-indexed prefiltered SH tables, and added a fixed
roughness/F0 split-sum approximation to the existing diffuse prediction. It
reused existing normalize, embedding, multiply, and reduction operations, but
still required candidate-level reflection tensors and three new graph inputs.
On the first cloud it reaches 23.59/23.06 dB and 22.71 dB where hit, versus the
selected repeat's 23.59/23.05 dB and 22.72 dB. Normal RMSE worsens from 53.69
to 53.96 degrees, and the release joint pass grows from about 1.01 to 1.11
seconds. A nine-coefficient lobe cannot represent the concentrated environment
features that identify glossy normals, while its approximate magnitude still
biases the diffuse solution. The reflection graph, SH tables, inputs, helper
refactor, and tests are removed. The capped run remains under
`target/audit-runs/current-synthetic-v1/specular-sh-normal-first/`.

## Rejected rank-augmented plane-sweep correspondence (2026-08-23)

An illumination-order descriptor does not improve the final reconstruction.
The prototype augmented the normalized plane-sweep patch with a half-weight
ternary rank descriptor, preserving the original descriptor and all geometry,
fusion, Gaussian, and PBR stages. Analytical plane and sphere correspondence
tests passed. On the first fixed cloud, extracted position RMSE changes only
from 0.5810 to 0.5807 world units, while held-light volumetric Gaussian output
falls from 23.59/23.05 dB and 22.72 dB where hit to 23.44/22.86 dB and 22.49
dB where hit. The descriptor and its tests are removed without a five-cloud
expansion. The capped diagnostic remains under
`target/audit-runs/current-synthetic-v1/rank-correspondence-first/`.

## Selected validation-gated static continuation (2026-08-23)

The optional masked PowerFoam surface continuation is now admitted to the
static Gaussian output only by an internal holdout. The last training camera
is excluded from both a baseline Gaussian fit and a continued-surface Gaussian
fit. Continuation must improve that camera by at least 0.05 dB; the chosen
surface is then fitted normally using every training camera. The production
and synthetic paths share the selector. The complete PowerFoam output still
uses all views and is always written when requested, so this gate changes only
its use as initialization for the separate static Gaussian asset.

Validation deltas on the five fixed clouds are +0.085, +0.178, +0.403,
+0.133, and -0.069 dB. The selector therefore retains the full continuation
for clouds 1--4 and the baseline for cloud 5, exactly matching the sign of the
true held-out-pose result in every case. Selected static mean/worst PSNR is
25.16/24.41, 25.09/24.12, 25.26/24.28, 25.32/24.64, and 25.02/24.31 dB,
raising the aggregate from 24.97/24.22 to 25.17/24.35 dB without using either
test pose for selection. The selected held-light PBR gate remains
23.59/23.06, 23.67/23.06, 23.40/22.77, 23.61/23.07, and 23.38/22.91 dB.

Each integrated run takes about 46.4 seconds because the opt-in path adds a
five-view continuation and two validation Gaussian fits before the full
six-view continuation. All five runs completed in 12 GiB cgroups with zero
OOM events and 271--317 MiB host-memory peaks. The implementation adds no
shader, graph operation, shader group or entry, binding, model field, file
format, dependency, or public CLI flag. Artifacts and telemetry remain under
`target/audit-runs/current-synthetic-v1/static-powerfoam-validation-integrated/`.

## Rejected calibrated-normal contrast variants (2026-08-23)

Using every unordered light pair does not improve the selected cyclic
normals-only contrast tail. It preserves the 100-update budget and reuses the
same graph, but the second fixed cloud falls from 23.67/23.06 to 23.65/23.05
dB mean/worst while covered-pixel quality moves from 22.62 to 22.64 dB. The
first cloud is effectively tied. Global light-pair completeness is therefore
not a substitute for a transferable differential direction, and the pair
table is removed. Runs remain under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-all-pairs-*`.

Foreground-focused sampling is mixed as well. Spending the entire contrast
tail on masked foreground rays improves clouds 1 and 2, including normal RMSE,
but regresses every tail on clouds 3--5 by 0.01--0.03 dB. Forcing only one
quarter of lanes repairs cloud 4 to 23.63/23.10 dB and 22.64 dB where hit, yet
clouds 3 and 5 still lose about 0.01 dB. At one lane in eight, cloud 3 retains
its mean and where-hit score but still loses 0.01 dB tail. Background
light-difference samples therefore provide useful silhouette-tail
regularization even with frozen centers. All sampling branches are removed;
artifacts remain under
`target/audit-runs/current-synthetic-v1/joint-center-normal-contrast-foreground-*`.

## Rejected robust and nuisance-variable normal fits (2026-08-23)

A `0.05` smooth-L1 residual in the discrete calibrated photometric-normal
initializer reduces the influence of unmodelled highlights, but lowers the
first fixed cloud from the selected 23.59/23.06 dB and 22.72 dB where hit to
23.54/22.93 dB and 22.66 dB where hit. Final normal RMSE improves from about
53.69 to 53.56 degrees, again showing that nearest-surface normal error does
not preserve an overlapping particle mixture. The L2 candidate score is
restored. The diagnostic remains under
`target/audit-runs/current-synthetic-v1/photometric-normal-smooth-l1-first/`.

Allowing a temporary per-particle diffuse albedo to absorb material error in
the differentiable center/normal pass is also rejected. Training it for the
whole center phase gives cloud 1 a 0.01 dB mean gain but costs cloud 2
0.02/0.01 dB mean/tail. Restricting it to the first quarter recovers cloud 2
and raises its where-hit quality by 0.03 dB, but cloud 1 falls below the
selected tail and where-hit reference. The scratch parameter, optimizer state,
and schedule branch are removed. Runs remain under
`target/audit-runs/current-synthetic-v1/joint-center-normal-scratch-albedo-*`.

Finally, treating learned Gaussian centers as refreshed surfel locations and
re-running observation/material assignment is strongly invalid. On the first
cloud, unseen particles rise from 1,121 to 1,846 and held-light quality falls
to 23.29/22.74 dB and 22.34 dB where hit. A learned volumetric center remains
a mixture parameter rather than a new one-to-one surface correspondence. The
late re-observation and material fit are removed; the capped run remains under
`target/audit-runs/current-synthetic-v1/post-center-coupled-material-first/`.

## Rejected foam-depth-owned Gaussian support (2026-08-23)

The independent foam reconstruction's modal depth does not provide a safe
owner for missing Gaussian opacity. The private prototype selected, for each
foreground training ray with a confident foam hit, the Gaussian candidate
nearest that strongest-absorption depth and routed a small one-sided opacity deficit only
to that candidate. It reused the existing graph operations and Gaussian
candidates; no backend operation, shader, entry point, dispatch, binding, or
model field was added.

At weight `0.1`, limiting the target to opacity below `0.5` produces five-cloud
held-light mean/worst PSNR of 23.60/23.05, 23.67/23.07, 23.40/22.77,
23.63/23.10, and 23.39/22.92 dB. Coverage remains 55.0%, 55.3%, 55.5%, 55.4%,
and 55.2%, and where-hit quality is 22.72, 22.63, 22.29, 22.63, and 22.39 dB.
Those aggregates look positive, but an immediate paired first-cloud control
reaches 23.60/23.06 dB and 22.73 dB where hit: the candidate loses both strict
tail measures under the same executable and machine state.

Weakening the full-deficit term to `0.05` only moves the failure. It ties the
paired first-cloud control at 23.60/23.06 dB and 22.73 dB where hit, but cloud
4 reaches 23.60/23.07 dB and 22.58 dB where hit instead of the selected
23.61/23.07 dB and 22.61 dB. The unthresholded `0.1` form likewise helps the
first two clouds and regresses clouds 3--5. A single foam depth mode collapses
a multi-surface ray distribution and cannot reliably identify the
Gaussian mixture component that owns a silhouette deficit.

The depth input, candidate-depth storage, owner mask, graph loss, API surface,
and synthetic wiring are removed. A future ownership signal must retain
multiple depth modes or establish cross-view particle correspondence rather
than selecting the candidate nearest one modal depth. All capped runs and
telemetry remain under
`target/audit-runs/current-synthetic-v1/depth-owned-pbr-support-*`.

## Rejected post-geometry single-light lobe recovery (2026-08-23)

Re-running the existing physical material solver after the selected calibrated
Gaussian center/normal continuation does not make specular semantics safe. The
private screen rebuilt surface observations with the final normals and centers,
then allowed four roughness/F0 hypothesis rounds against the known `sun-east`
environment before attaching the resulting 12-material table to the Gaussian.
It reused the existing CPU solver and exact renderer; no operation, shader,
entry point, binding, model field, dependency, or public option was added.

At the solver's established `0.15` lobe margin, 52.7% of observed particles
have multiple-view angular evidence and the one-light residual is `0.06753`.
Nevertheless, first-cloud volumetric Gaussian held-light quality collapses
from the paired diffuse-only control's 23.60/23.06 dB and 22.73 dB where hit
to 18.66/18.37 dB and 16.65 dB where hit. Training-light quality also falls to
23.62/22.16 dB, so the result is not merely a held-light trade.

Requiring a 50% residual improvement before accepting any non-default lobe is
not conservative enough: the same gate reaches only 18.52/18.01 dB and 16.54
dB where hit under the held light. The reconstructed surface observations and
shared chromaticity clusters still confound highlights, overlap, and geometry;
a stronger hypothesis threshold cannot turn them into correspondence. The
post-geometry solve is removed. Recovering roughness and F0 now requires a
joint multi-light rendered objective on the final Gaussian mixture, with an
explicit validation holdout, rather than transferring a single-light surface
decomposition. Artifacts remain under
`target/audit-runs/current-synthetic-v1/post-geometry-specular-{first,half-margin-first}/`.

## Rejected validation-gated Gaussian lobe polish (2026-08-23)

An exact multi-light rendered objective still does not identify transferable
roughness or F0 on the reconstructed Gaussian mixture. The private candidate
optimized the final Gaussian material buffer through the production PBR
renderer using three calibrated lights, reserved the fourth calibrated light
as an internal gate, and kept `studio` plus both novel poses entirely unseen.
It reused prepared tracers and material-buffer updates, with no new shader,
backend operation, binding, model field, file property, or dependency.

The coupled roughness/F0 fit changes 40 of 96 visited coordinates, reducing
the three-light sRGB objective from `0.0060431` to `0.0059213` and the reserved
light from `0.0094783` to `0.0091762`. The gate therefore retains it, but true
held-light mean/worst/where-hit quality falls from the paired control's
23.60/23.06/22.73 dB to 23.57/23.03/22.70 dB. Calibrated-light transfer alone
is not an adequate proxy for novel illumination.

Holding F0 at the dielectric default removes the large regression but not the
strict failure. A `0.2` roughness step changes seven coordinates and reaches
23.60/23.04/22.73 dB; a `0.1` step reaches 23.58/23.04/22.71 dB. Both improve
their training and reserved-light objectives and both lose the true tail.
Consequently the stats type, fitting API, prepared-tracer loop, synthetic
wiring, and incidental material-copy change are removed. The selected asset
continues to state only the PBR semantics supported by the evidence: fitted
diffuse albedo with conservative rough-dielectric specular defaults. Runs and
telemetry remain under
`target/audit-runs/current-synthetic-v1/multilight-gaussian-pbr-first/` and
`target/audit-runs/current-synthetic-v1/multilight-gaussian-roughness-{first,half-first}/`.

## Shared compute-context lifetime (2026-08-23)

The sequential compute stages now retain one Blade context instead of creating
a new device for foam-depth tracing, direct Gaussian fitting, and calibrated
multi-light geometry. The ray-traced scoring renderer remains separate because
it requires a ray-tracing-capable context while Meganeura deliberately creates
its compute context with ray tracing disabled. No shader, graph operation,
binding, model field, dependency, file format, or public option changes.

The ordinary paired five-cloud synthetic gate reduces compute-context creation
from three devices to one. Wall time falls from 17.890 to 16.486 seconds
(`-7.8%`) and cgroup memory peak from 260.3 to 254.5 MiB. The volumetric
Gaussian held-light result is unchanged at the reported precision:
23.59/23.05 dB mean/worst, 55.0% coverage, and 22.71 dB where hit. The control
and candidate remain under
`target/audit-runs/current-synthetic-v1/shared-compute-context-{paired-control,first}/`.

PowerFoam needs a more careful handoff. Retaining the initial density context
alongside a fresh continuation context changed the validation decision on the
first cloud. The selected lifecycle therefore releases the density context,
creates a fresh context for the validation/full continuation pair, and retains
that context for later Gaussian stages. This reduces four compute contexts to
two while preserving the fresh validation allocation state. Against an exact
old-lifecycle control, wall time falls from 46.331 to 44.854 seconds (`-3.2%`).
The continuation remains selected at 22.30 -> 22.35 dB rather than
22.29 -> 22.36 dB, and held-light Gaussian PBR is 23.60/23.06 dB with 55.0%
coverage and 22.73 dB where hit, matching or improving every reported rounded
quality measure. Peak memory rises from 267.2 to 326.3 MiB, still far below the
12 GiB limit with zero swap or OOM. Runs remain under
`target/audit-runs/current-synthetic-v1/shared-compute-context-powerfoam-{paired-control,handoff-first}/`.

Production smokes exercise the same lifetimes. Foam depth plus calibrated
Gaussian fitting creates one compute context and completes in 2.157 seconds at
233.8 MiB. A four-view masked PowerFoam run creates exactly two compute
contexts, completes continuation, Gaussian fitting, serialization, and scoring
in 5.356 seconds at 220.2 MiB, and reports no swap or OOM. That smoke also
exposed that the static pre-decomposition surface inherited per-surfel material
indices while owning only one default material. Static-surface preparation now
neutralizes those indices before PowerFoam validation; the PBR surface keeps
its independent indices for the later material solve. The successful artifacts
and telemetry are under `target/audit-runs/shared-compute-context-production-v2/`
and `target/audit-runs/shared-compute-context-production-powerfoam-v4/`.
