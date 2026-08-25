# Audit and implementation roadmap

Initial audit: 2026-07-12

Last updated: 2026-08-25

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

The unresolved risk is now specifically trainer scaling rather than the
representation or renderer. Official RadFoam v1 reaches 30.02 dB on Room after
its first 5,000 updates with 735,103 cloud cells, while Blade cross-renders the
same PLY and full-resolution held-out split at 29.59 dB. The 0.43 dB renderer
gap meets the Stage 2 tolerance. Blade's selected scaled Room trainer remains
far less optimized: its 16-view, 14,976,000-ray step-6,000 checkpoint matches
the 735,103-cell capacity and reaches 26.22 / 24.57 dB on the selected
train/held-out split and 24.14 dB across all 39 held-out views at 256². The
official prefix has processed roughly five billion rays and reaches 30.02 dB.
The Rust path therefore remains a useful scaling baseline, not a
production-ready reconstruction pipeline.

Temporal gradient accumulation now provides a bounded-memory way to raise
multi-view coverage per optimizer update. It is deliberately opt-in: on the
98,831-site Room gate, 2x accumulation improves the 32-training-view held score
from 20.2976 to 20.4819 dB, while the same setting over only four training
views lowers held quality from 19.8841 to 19.7973 dB. The implementation keeps
the static graph and path-buffer sizes unchanged, preserves exact
update-boundary resume, and uses 20% less host memory than the matched
monolithic 8,192-ray run, at a 3% training-time cost.

The corrected paths pass the targeted NVIDIA/Vulkan physical-GPU gates,
including weighted differentiable traversal, Gaussian CPU/GPU parity, and
transformed-scene pixel readback. Whole-cloud layering is still not exact
interleaved volume compositing and the Metal multi-cloud binding design is
open. New rendering methods and broad scene features should therefore remain
behind these gates:

1. Match reference RadFoam training within the Stage 2 quality tolerance on a
   complete, reproducible scene. Checkpoint renderer parity is done.
2. Keep the full directional table opt-in until both cost and quality change.
   Released-checkpoint pixel parity passes. Generic Meganeura grouped
   scatter/gather fusion, frozen-axis prepacking, and channel-wise Blade
   parameters reduce the current matched graph to 221 passes and
   3.17--3.34 ms. Even so, 256 updates take 10.323 seconds and peak at 3.04 GB,
   versus 2.583 seconds and 292 MB for the no-table production graph. The full
   table loses 0.9288 dB held out despite fitting the four training views
   1.9592 dB better. Sharing one eight-direction residual across all eight
   spatial sites is also rejected at 18.1785 dB held out and 13.139 seconds;
   smaller copies of the same table are not the missing regularizer. A future
   retry needs a different spatial/directional responsibility model and must
   pass a two-scene gate.
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
It also ran workspace formatting, linting, tests, render regressions, and three
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
- Densification uses position-gradient × cell-radius for unweighted RadFoam and
  a device-resident per-site photometric-error EMA for weighted PowerFoam, plus
  contribution-aware pruning and optimizer ancestry.
- Quantile regularization, explicit background compositing, lossless DC SH, and
  versioned parameter/Adam/RNG checkpoints are implemented.
- Trainer-scale quality remains the blocker, but the first Bonsai plateau was
  an update-composition failure rather than a hard capacity limit. Re-evaluated
  under current renderer semantics, its 200,000-cell step-10,000 checkpoint
  measures 16.76/15.83 dB train/selected-held-out and 16.16 dB over all 37
  held-out views. Lossless continuation with 4,096 rays distributed across 16
  views on the original 20,400-step cosine horizon reaches
  22.89/20.41/21.35 dB at the exact endpoint. The last 400 updates add only
  0.02 dB all-37. This validates mixed-view scaling on a second complete
  scene, but 256² comparisons still show translucent floaters, smeared
  backgrounds, and weak thin geometry.
- The official RadFoam v1 code is now pinned and compared line by line in
  [`RADFOAM_REFERENCE_COMPARISON.md`](RADFOAM_REFERENCE_COMPARISON.md). Its
  Room recipe processes one million mixed-view rays per update, targets roughly
  106K→2.1M sites, stages native-aspect resolution, uses Smooth-L1 on white,
  and applies independent parameter schedules. An exact serialized prefix at
  update 5,000 contains 735,103 sites and reaches 30.0239 dB over all 39
  held-out views. The unchanged upstream run cannot cross its downsample-2
  reload on this host: its cached ray data exceeds a 32 GiB cgroup and it is
  already using 7.3 GiB of 12 GiB VRAM at that boundary.
- Blade loads that official PLY without conversion. Correcting a missing
  per-cell nonnegative SH clamp improves the identical 39-view cross-render
  from 28.97 to 29.59 dB, only 0.43 dB below upstream. CPU, standalone WGSL,
  scene WGSL, Gaussian WGSL, and the differentiable Meganeura graph now share
  the clamp and pass CPU/GPU regression tests. Renderer compatibility is no
  longer the primary Stage 2 blocker.
- `radfoam-v1` initialization and learning-rate policies plus a beta-1
  Smooth-L1 color loss are implemented. Reference initialization is now the
  default after its quality gain and surface-only Delaunay failure gate; the
  other reference policies remain opt-in. A versioned scaled semantic-ablation
  manifest records both the controls and the remaining
  batching/topology/resolution differences.
- The first bundled semantic ablation was stopped at step 2,000 after its fresh
  serialized PLY reached 10.83 dB train / 12.37 dB held out, below the old
  scaled protocol's 13.08 / 13.08 dB checkpoint. It was memory-stable, but had
  processed only 512,000 rays versus 2 billion at the same official update
  count. The result therefore rejects continuing that low-batch bundle; it does
  not reject reference initialization, loss, background, or learning rates in
  isolation. The next comparison matrix changes one factor at a time.
- The first one-factor matrix identifies the main quality lever and the bundled
  failure. Scaled v1 initialization improves the reloaded held-out result by
  2.03 dB; white background improves it by 0.45 dB. Smooth-L1 improves the mean
  by 0.28 dB but collapses one view to 4.44 dB, so it is not yet accepted. The
  exact v1 step schedule loses 4.58 dB at batch 256, confirming that its warmup
  is coupled to the official one-million-ray updates. The next candidate keeps
  the stable L1/cosine optimizer and combines only v1 initialization with white.
- That interaction is not additive: initialization plus white reaches 14.83 dB
  train / 14.74 dB held out, versus 14.59 / 15.11 dB for initialization alone.
  The selected scaled trajectory is therefore v1 initialization with the
  existing black/L1/cosine path, resumed from its lossless step-2,000 state.
- The resumed initialization curve reaches 15.11, 15.39, and 15.58 dB held out
  at steps 2K, 3K, and 4K. Its advantage over the old curve narrows from +2.03
  to +1.25 to +0.35 dB while the last 1K segment grows to 1,021 seconds for a
  0.19 dB gain. It is stopped at 99,262 cells: initialization is a supported
  improvement, but not the explanation for the remaining quality plateau, and
  exhaustive contribution scanning is now the dominant scaling bottleneck.
- An opt-in rotating stratified contribution cap is implemented while the
  default stays exhaustive. On the same step-4K state, 32/255 views cut the
  next boundary from 635 to 299 seconds and cost only 0.03 dB held out, but
  retained 1,223 fewer cells and changed topology. It fails the plan's exact
  decision-agreement gate, so it remains experimental pending larger caps and
  multi-boundary drift measurements.
- Parameter-group scaling is now independent from the exact v1 time schedule.
  The official initial density/position/DC/higher-SH ratios can run under the
  stable cosine curve, but the isolated step-2,000 result is negative: 14.72
  dB train / 14.49 dB held out, compared with 14.59 / 15.11 dB for the legacy
  relative groups. Fresh-Ply evaluation reproduced the scores and cell counts
  were nearly equal. The option remains for larger-batch work; it does not
  replace the selected scaled optimizer.
- The official topology-refresh counter is available independently as
  `radfoam-v1`: exact Rust topology/path rebuilds follow periods 1, 3, 5, ...,
  99, then 101, and reset their period after densification. The current v3
  training sidecar persists the phase and still reads v2; a physical-GPU
  4+6-step resume produced a PLY
  byte-identical to an uninterrupted 10-step run. In the clean step-2,000
  comparison, dynamic cadence performs 44 rather than 20 scheduled updates,
  takes 676 rather than 652 seconds, and reaches 14.58 / 15.05 dB train/held
  out versus fixed-100's 14.60 / 15.13. It remains an opt-in reference control;
  fixed-100 remains the selected scaled cadence.
- The official densification counter is independently available as
  `--densify-schedule radfoam-v1`. It grows first at the warmup boundary,
  derives each later interval from the post-growth cell count with a 100-step
  floor, and stops at 90% of the requested final count. A v3 trainer sidecar
  persists its initial count, phase, next interval, and absolute round while
  retaining v1/v2 read compatibility. A physical-GPU two-boundary resume is
  bit-identical to uninterrupted cloud parameters and adjacency. All 81
  trainer tests passed on the pinned RTX 5070 at a 985,112,576-byte cgroup
  peak; the full all-feature workspace suite peaked at 1,182,797,824 bytes.
  Both recorded zero swap, OOM, or GPU faults. In the isolated two-round
  Bonsai comparison, the dynamic policy delayed the second growth from step
  2,500 to 2,517 and finished with the same 66,054 cells, but changed
  train/held-out PSNR from 16.34 / 16.25 to 16.11 / 16.18 dB. Both training
  scopes peaked near 481 MB without swap, OOM, or GPU faults. Fixed cadence
  remains the scaled default; the dynamic policy remains a reference-scale
  control.
- A batch-1,024 gate now distinguishes update-indexed controls from schedules
  normalized by sampled rays. At 512,000 rays, the exact v1 schedule remains
  decisively negative at 9.97 / 10.53 dB train/held out versus 15.11 / 14.89
  dB for cosine/legacy. Relative v1 parameter groups are neutral at 14.94 /
  14.95 dB (-0.17/+0.06 dB). Scaling the cosine horizon, exact-topology
  cadence, and densification timeline by four yields 15.26 / 15.33 dB, a
  +0.66/+0.20 dB same-ray improvement over the corrected batch-256
  checkpoint. Fresh-Ply evaluation reproduces every score. All four training
  scopes peak between 409 and 420 MB with zero swap, pressure, OOM,
  truncation, or GPU faults. Ray-normalized larger batching advances to a
  multi-boundary/native-aspect gate without changing the default yet. At the
  second boundary the square arm reaches 15.55 / 15.29 dB, a +0.29/-0.04 dB
  change from step 500. On a common 192×128 evaluation it beats native-aspect
  training by 0.14 dB train and 0.23 dB held out while using one-third fewer
  contribution rays. The final clouds differ by only eight cells; both have
  zero truncation. Native training peaks at 519 MB and square continuation at
  451 MB, again with zero swap, pressure, OOM, or GPU faults. Because both
  output grids cover the full calibrated camera domain, 128×128 remains the
  efficient scaled resolution rather than a cropped geometry path.
- The corrected same-ray batch gate now extends both trajectories through two
  growth rounds. At 640,000 optimizer rays and 25 topology refreshes, batch
  256 reaches 15.32 / 15.12 dB train/held out with 65,801 cells; batch 1,024
  reaches 15.55 / 15.29 dB with 66,078 cells (+0.23/+0.17 dB). Matched final
  continuations take 391 and 384 seconds and peak at 424 and 451 MB,
  respectively. Both record zero truncation, swap, pressure, OOM, or GPU
  faults. The larger batch prunes 112 fewer cells in round 1, showing that the
  batch changes learned contribution decisions as well as optimizer progress.
  It is the selected direction, but its mixed per-frame 0.17 dB gain requires
  a third boundary or another scene before becoming the default.
- The third same-ray boundary resolves the batch decision. At 768,000 optimizer
  rays, batch 256 reaches 15.41 / 14.76 dB train/held out with 75,452 cells;
  batch 1,024 reaches 16.08 / 15.66 dB with 75,969 cells (+0.67/+0.90 dB) and
  improves seven of eight held-out frames. Matched continuations take 445.0
  and 443.9 seconds and peak at 475 and 489 MB. Both have mean paths near 47,
  maxima 150–151, zero truncation, and zero swap, pressure, OOM, or GPU faults.
  The smaller batch prunes 191 cells versus 18, consistent with noisier
  geometry/contribution gradients. Batch 1,024 passes the Bonsai gate and is
  selected as the scaled pixel-batch default. `train_colmap` now uses it when
  the flag is omitted, with a parser regression test covering both the new
  default and an explicit 256 override.
- The second-scene Room gate confirms that selection and has now been retrained
  after the per-cell SH clamp correction. Under the identical selected
  batch-1,024 protocol, the corrected cloud reaches 75,809 cells and 19.02 /
  18.84 dB train/held out after a fresh PLY reload, versus the historical
  75,718-cell result at 18.50 / 18.23 dB. Rendering the historical cloud under
  corrected semantics gives only 17.79 / 17.37 dB, so the +0.52/+0.61 dB new
  result is not an evaluation-only artifact. The 24m 0.142s retrain peaks at
  543,166,464 host bytes and 551 MiB sampled GPU memory; all three exhaustive
  contribution rounds cover 1,044,480 rays with zero truncation, swap,
  pressure, OOM, or GPU faults. Its 18.44 dB all-39-view diagnostic is not
  compared with official v1 because this bounded run caps training at 255
  views and selected validation at the first eight held-out frames. Visual
  inspection still shows severe cell/color fragmentation and lost fine
  structure, so the positive metric delta does not make this viewer-ready.
- Deterministic mixed-view optimizer batches are implemented through the
  `--views-per-batch` control. Camera-specific recorder dispatches fill
  disjoint rows of one fixed-size path batch, while the Meganeura graph consumes
  per-ray origins and view indices. One-view mode retains its exact historical
  RNG draw order; a physical-GPU 4+4-step resume is bit-identical to an
  uninterrupted run, and sliced RadFoam/PowerFoam path recording agrees with
  the CPU oracle. On Room, changing only 1→16 views within the same 1,024-ray
  batch raises fresh-Ply quality from 19.02 / 18.84 to 20.21 / 19.94 dB and
  improves all eight selected held-out frames. The all-39 diagnostic rises
  from 18.44 to 19.66 dB. Cell count stays within 0.2%; wall time rises 1.6%,
  host peak 3.1%, GPU peak is unchanged, and no truncation, swap, pressure,
  OOM, or GPU fault occurs. Matched continuations to step 1,000 retain a
  +0.95 dB held-out gain (20.90 versus 19.95 dB), improve all eight frames,
  and keep capacity and continuation cost effectively equal. The mixed
  750-step checkpoint already matches the one-view 1,000-step score with 25%
  fewer optimizer rays. `train_colmap` therefore selects 16 views
  automatically for random-pixel batches; full-image, patch, and direct
  library defaults remain one view.
- Continuing the selected mixed-view trajectory to step 1,500 grows the cloud
  from 99,970 to 174,410 cells and improves fresh-Ply train/held-out quality
  from 21.31 / 20.90 to 22.39 / 21.83 dB. The all-39 diagnostic improves from
  20.70 to 21.60 dB. Four exhaustive contribution rounds process 1,044,480 rays
  each with mean paths rising from 69.4 to 88.2, maxima from 155 to 178, and
  zero truncation. The 48m 3.365s continuation peaks at 1,038,524,416 host
  bytes and 719 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU
  faults. Visuals remain fragmented, so scaling is productive but not yet
  sufficient.
- The next 125-step segment reaches the configured 200,000-cell cap exactly
  and improves fresh-Ply train/held-out quality to 22.63 / 22.00 dB; all 39
  diagnostic views average 21.80 dB. Its exhaustive scan averages 94.3
  segments, reaches 188, and truncates no rays. The 16m 4.421s continuation
  peaks at 1,060,331,520 host bytes and 688 MiB sampled GPU memory with zero
  swap, pressure, OOM, or GPU faults. This validates the first capacity
  boundary; visuals remain too fragmented for use.
- Raising only the cap to 400,000 and continuing to step 2,000 reaches 303,891
  cells and improves fresh-Ply train/held-out quality to 23.16 / 22.31 dB; all
  39 diagnostic views average 22.22 dB. Three exhaustive scans average
  100.6→113.5 segments, peak at 223, and truncate no rays. The 55m 54.647s
  continuation peaks at 1,730,691,072 host bytes and 936 MiB sampled GPU memory
  with zero swap, pressure, OOM, or GPU faults. Scaling remains productive, but
  the result is not viewer-ready.
- Continuing to step 2,250 reaches the 400,000-cell cap and improves fresh-Ply
  train/held-out quality to 23.42 / 22.43 dB; all 39 diagnostic views average
  22.38 dB. The two scans average 120.2/127.0 segments, peak at 246, and
  truncate no rays. A 512-step reload is metric-identical to the 256-step
  result, but only ten traversal steps remain, so the next growth segment must
  raise the budget. The 47m 43.453s continuation peaks at 2,309,640,192 host
  bytes and 954 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU
  faults.
- A guarded max-512 round to step 2,375 reaches 459,888 cells and improves
  fresh-Ply train/held-out quality to 23.55 / 22.53 dB; all 39 diagnostic views
  average 22.47 dB. Its exhaustive scan measures 133.6 mean / 259 maximum
  segments with zero truncation, proving that the old 256-step budget is no
  longer sufficient. The 40m 11.225s continuation peaks at 2,324,807,680 host
  bytes and 1,362 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU
  faults.
- The next max-512 round reaches 528,732 cells and improves fresh-Ply
  train/held-out quality to 23.69 / 22.57 dB; all 39 diagnostic views average
  22.56 dB. Its scan measures 140.9 mean / 267 maximum segments with zero
  truncation. The 45m 2.106s continuation peaks at 2,646,769,664 host bytes and
  1,690 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU faults.
  Held-out improvement has slowed to +0.04 dB at this boundary.
- The following max-512 round reaches 607,908 cells and improves fresh-Ply
  train/held-out quality to 23.80 / 22.64 dB; all 39 diagnostic views average
  22.62 dB. Its scan measures 148.5 mean / 275 maximum segments with zero
  truncation. The 50m 21.045s continuation peaks at 3,013,816,320 host bytes and
  1,786 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU faults.
- At step 2,750 the ladder reaches 698,940 cells. Train quality rises to 23.89
  dB, selected held-out slips 0.02 dB to 22.62 because one foreground smear
  regresses, and all 39 diagnostic views improve 0.03 dB to 22.65. The scan
  measures 156.2 mean / 299 maximum segments with zero truncation. The 56m
  51.154s continuation peaks at 3,572,768,768 host bytes and 1,773 MiB sampled
  GPU memory with zero swap, pressure, OOM, or GPU faults.
- At step 2,875 the ladder reaches the 735,103-cell reference capacity exactly
  and improves fresh-Ply train/held-out quality to 24.07 / 22.77 dB; all 39
  diagnostic views average 22.76 dB. Its scan measures 164.2 mean / 305 maximum
  segments with zero truncation. The 1h 3m 35.955s continuation peaks at
  3,634,937,856 host bytes and 1,822 MiB sampled GPU memory with zero swap,
  pressure, OOM, or GPU faults. Capacity matching is done; optimization rays,
  resolution, and protocol remain unmatched.
- The first fixed-cap continuation completed geometry cycles through step 2,975
  before NVIDIA Xid 79 reported that the GPU fell off the bus. The cgroup fault
  watcher terminated it before any step-3,000 checkpoint was written. Its
  2,692,427,776-byte host peak, zero swap/pressure/OOM counters, and final
  1,788 MiB / 72 °C / 100% GPU sample rule out cgroup memory exhaustion. Resume
  remains at the validated step-2,875 checkpoint after reboot.
- After reboot, the identical fixed-cap retry completes step 3,000 and improves
  fresh-Ply train/selected-held-out quality to 24.29 / 22.92 dB; all 39
  diagnostic views improve 0.14 dB to 22.90. The 49m 42.305s scope peaks at
  1,835 MiB sampled GPU memory and 74 °C with no GPU fault, swap, or OOM. Final
  serialization/evaluation reaches the exact 4 GiB memory limit and increments
  `memory.events:max` 221 times, so later 735K-cell timing scopes use 6 GiB.
  The general metric gain is real, but comparison images remain visibly
  fragmented.
- Per-dispatch profiling shows that 51 dense embedding-gradient scatter-adds
  consume 19.747 of 19.776 GPU seconds in a representative 735K-cell step.
  Meganeura `fba040a` switches large scatters to a zero pass plus atomic f32
  accumulation while retaining the deterministic dense path for small
  workloads. One-step Room parameter deltas stay below `1.49e-8`; loss, PSNR,
  capacity, and topology size agree. The measured optimizer portion is about
  98× faster, and the full 25-step optimizer/topology boundary falls from 505
  to 21 seconds.
- Blade `27242ad` continues the fixed-cap run to step 3,125 in a 218-second
  training/topology interval, 12.3× faster than the old comparable five
  cycles. Fresh-Ply quality reaches 24.44 / 23.06 dB and 23.00 dB across all
  39 held-out views. A 4,096-ray batch then processes four times as many rays
  in a 227-second interval, only 4% longer, and reaches 24.74 / 23.26 dB plus
  23.19 dB across all 39. Its 3,831,681,024-byte host peak, 3,763 MiB sampled
  VRAM peak, and zero pressure/swap/OOM/fault events select 4,096 rays for the
  next resolution boundary. Visual fragmentation remains.
- Raising the training and evaluation resolution to 256² from the same
  step-3,250 checkpoint gives a 24.55 / 23.23 dB train/held-out baseline and
  23.19 dB across all 39 views. Continuing 125 steps reaches 24.77 / 23.39 dB
  and 23.34 dB across all 39. A matched 64-view arm ties both held-out means
  while taking 6% more training/topology time and 3% more peak host memory, so
  the simpler 16-view policy remains selected.
- Fixed-cap topology cadence is now measured rather than assumed. At step
  3,500, rebuilding every 125 rather than 25 steps cuts the matched
  training/topology interval from 219 to 46 seconds with identical 23.51 /
  23.45 dB selected/all-39 quality. At step 3,750, cadence 250 ties cadence 125
  at 23.68 / 23.62 dB while cutting the interval from 111 to 63 seconds. A
  longer cadence-500 gate loses 0.01 dB selected and all-39 at step 4,500 for
  only a 25% interval saving. Cadence 250 is therefore selected after the
  cloud reaches fixed capacity; dense refreshes remain appropriate while
  topology-changing densification is active.
- Continuing the selected fixed-cap path to aligned step 5,000 processes
  10.88 million optimizer rays and reaches 25.69 / 24.03 dB train/held out and
  23.93 dB across all 39 views. The final 500 updates add only 0.06 dB on the
  selected set and 0.05 dB across all views. Large surfaces and room layout
  improve, but foreground furniture, thin structures, and occlusion edges
  remain visibly smeared. More identical low-batch continuation is now
  lower-value than closing the remaining reference-protocol differences.
- Commit `77c19b7` centralizes the production RadFoam/PowerFoam WGSL compute
  tracer so the viewer and opt-in headless evaluator cannot drift. Physical
  pixel tests match the CPU oracle for weighted and unweighted clouds. On the
  selected 735,103-cell Room PLY, GPU and CPU evaluation agree at
  25.69/24.03 dB train/held out and 23.93 dB all-39; per-view reporting differs
  by at most 0.01 dB from the RGBA16F target. The identical 255+8-view pass
  falls from 548.251 to 119.668 seconds (4.58×), and all 39 held-out views
  fall from 77.923 to 7.758 seconds (10.04×). Host peaks remain near 0.5 GB,
  sampled VRAM peaks at 279 MiB, and both isolated runs record zero swap,
  pressure, OOM, or GPU faults. The CPU path remains the default oracle.
- Commit `5e3f81d` makes long training runs report setup, input, GPU wait,
  readback, topology, densification, checkpoint, finalization, pipeline, and
  CLI-output timings. The first matched 735,103-cell profile exposed redundant
  full parameter downloads at the geometry/checkpoint/finalization endpoint.
  Reusing current host parameters cuts training from 167.369 to 120.675
  seconds (1.39×), whole-command time from 205.641 to 158.960 seconds (1.29×),
  and host peak from 4,421,963,776 to 4,200,992,768 bytes. Loss, cell and edge
  counts, selected 24.03 dB held-out quality, and checkpoint/final PLY identity
  are preserved. Serialization remains the largest phase.
- Commit `fe7517d` eliminates the remaining duplicate endpoint PLY encoding
  without inferring success from stale files: the trainer explicitly returns
  its successfully written endpoint snapshot, and the CLI atomically copies
  it with serialization fallback. At 735,103 cells, final PLY output falls
  from 13.991 to 0.045 seconds (311×), whole-command time reaches 142.098
  seconds (1.45× over the original profile), and exact terminal host peak is
  3,899,666,432 bytes. Checkpoint/final hashes and the 24.03 dB held-out result
  match. Commit `c8ca20f` records terminal cgroup counters from inside the
  live scope so short-lived peaks can no longer escape the sampler.
- Checkpoint subphases reveal the underlying problem: optimizer persistence
  spent 61.486 seconds reading write-combined GPU-backed mappings on the CPU,
  while the model PLY spent 13.970 seconds issuing millions of tiny writes.
  Blade `f724f6a` adds CPU-cached download memory, and Meganeura `a2ce41c`
  snapshots all parameters and Adam state in one GPU transfer before streaming
  safetensors to disk. Together with Blade-volume's buffered PLY writer, the
  optimizer save falls to 0.425 seconds, model PLY output to 0.094 seconds, and
  the complete checkpoint to 0.531 seconds. The matched step-5,000→5,100
  command falls from 144.032 to 68.663 seconds (2.10×) while retaining the
  exact 0.0919→0.0854 loss trace and 24.03 dB fresh-Ply result. Exact host peak
  changes only from 4,303,171,584 to 4,348,669,952 bytes (+1.1%), with zero
  swap, pressure, OOM, or GPU faults. Blade-volume now pins both exact
  dependency revisions so the tested path is reproducible.
- A larger-batch gate exposed a Meganeura dispatch-limit correctness bug.
  With 8,192 rays and 512 path steps, the scalar `[P*L,3]@[3,1]` matmul
  requested 65,536 workgroups in Y and produced non-finite position gradients
  before Qhull. Meganeura `2e20ec6` splits tall scalar matmuls across Y/Z and
  verifies a physical 4,194,304-row result end to end. The repaired Room run
  is stable, but 8,192 rays add only 0.01 dB selected and all-39 held-out
  quality while increasing training time 59%, graph allocation 90%, and
  sampled VRAM from 3,891 to 6,219 MiB. The correctness fix is retained;
  4,096 rays remains selected.
- Commit `0da6a85` keeps the live Meganeura session, Adam state, and external
  path buffers across ordinary fixed-cap topology refreshes. Adjacency belongs
  to the path recorder, so recreating an identical training graph and copying
  every appearance parameter/moment was unnecessary. In the deterministic
  step-4,500→5,000 Room gate, state readback falls from 33.010 to 3.574
  seconds (9.24×), training from 158.513 to 127.533 seconds (1.24×), and whole
  command time from 178.520 to 147.806 seconds (1.21×). Every selected view
  retains its reported PSNR and the all-39 mean remains 23.93 dB. Exact host
  peak falls 4.5%; swap, pressure, OOM, and GPU-fault counters remain zero.
- Fixed-cap quality controls rule out 512² training and distortion weights
  `1e-4`/`1e-3`: each is neutral at reported selected and all-39 held-out
  precision. A four-arm schedule gate instead finds a coupled improvement.
  Reopening the cosine horizon alone collapses all-39 quality from 23.93 to
  22.70 dB, and relative RadFoam-v1 parameter groups alone reach 23.91 dB;
  together they restore reference-scale rates and reach 24.00 dB at step
  5,000. Lossless continuation reaches 26.22/24.57 dB train/selected and
  24.14 dB all-39 at step 6,000. The final 500 steps add only 0.02 dB all-39
  and regress some late held-out views, so step 6,000 is selected as the
  stopping point. The cloud is still visibly blurred and is not viewer-ready.
- Commit `1c46fb2` specializes exact unbounded CSR construction. Exact
  Delaunay/Čech inputs are already symmetric, so they no longer globally
  distance-sort every edge or allocate a second graph merely to retain all
  neighbors; finite approximate caps preserve the old path. On the matched
  step-5,500→6,000 Room continuation, topology falls from 30.949 to 26.114
  seconds (1.19×), training from 127.416 to 122.912 seconds, and the command
  from 152.098 to 145.236 seconds. All-39 quality changes only 24.14→24.15 dB.
  An isolated exact rebuild also lowers peak memory 10.1%, though another
  phase dominates the full training peak.
- The selected mixed-view direction transfers to the complete Bonsai scene,
  but Room's parameter and topology choices do not transfer wholesale. From
  the current-semantics step-10,000 checkpoint, 4,096 rays across 16 views
  reach 22.89/20.41 dB train/selected and 21.35 dB all-37 at step 20,400,
  gains of 6.13/4.58/5.19 dB at the same 200,000-cell capacity. Relative
  RadFoam-v1 groups lose 0.11/0.17 dB selected/all-37 at the first gate. A
  229,566-cell densification loses 0.13/0.11 dB, and cadence 250 is 1.25×
  faster but loses 0.02/0.05 dB. Bonsai therefore selects legacy groups,
  fixed 200K capacity, cadence 100, and the completed horizon.
- Commit `86cfcab` uses Blade's CPU-cached download memory for the four
  exhaustive contribution buffers. The scoring arithmetic and serial order
  are unchanged. In the matched Bonsai 200K→230K gate, contribution time falls
  from 561.315 to 1.197 seconds (468.9×), training from 721.832 to 169.076
  seconds, and the command from 728.308 to 175.622 seconds (4.15×). All-37
  quality remains 19.17 dB, host peak falls 4.9%, and both scopes record zero
  swap, pressure, OOM, or GPU faults.
- Forming geometry under mixed-view batches from initialization is much
  stronger than continuing geometry formed by 256-ray updates. The fresh L1
  run reaches 26.05/24.00/24.69 dB train/selected/all-37 at step 20,400,
  improving the completed continuation by +3.16/+3.59/+3.34 dB at the same
  200,000-cell capacity. The final 400 updates are flat. Its segmented 4 GiB
  scopes peak at 1,198,428,160 bytes and record zero swap, pressure, OOM, or
  kill events.
- A final-400 Bonsai traversal gate reduces the graph cap from 256 to 224
  steps. Logical/device-local graph allocation falls 2,342.4→2,060.8 MB and
  541.1→473.4 MB; GPU-step time improves 1.13× and total training 1.08×.
  Quality remains 26.05/24.00/24.69 dB, and the candidate's 256² renders are
  byte-identical at 224 and 256 steps. This is selected for the capped scene,
  not generalized to a growing cloud.
- A fresh Smooth-L1 gate reverses the old tiny-batch result. After a matched
  400-step post-growth settle, it beats L1 by +0.54/+0.98/+0.45 dB
  train/selected/all-37. Under normal densification it retains
  +0.43/+0.84/+0.33 dB at step 6,000, and its completed continuation reaches
  26.78/24.48/24.94 dB at the same 20,400-step, 200,000-cell endpoint —
  +0.73/+0.48/+0.25 dB over the L1 control. It leads at every measured
  checkpoint, so the loss is selected rather than merely opt-in on this scene.
- The relightable depth-mode extractor now defaults to the two-scene selected
  factor-5 fusion and 1.7-cell disc support. Against the previous factor-4/2.5
  protocol, Bonsai train/held-out PSNR rises 14.34/13.55→14.89/14.03 dB and
  Room rises 14.40/14.08→14.59/14.14 dB, with every worst-view PSNR also
  improving and 22–24% fewer surfels. The scenes remain visibly coarse; this
  improves the initializer and does not close the shared-surface geometry gap.
- The extractor now runs the production RadFoam/PowerFoam walk on the GPU and
  writes full-precision mode depth, alpha, and peak weight. Five-run end-to-end
  medians fall 3.73→2.45 seconds on Bonsai and 6.93→3.19 on Room, with selected
  quality preserved. The complete GPU-inclusive workspace suite peaks at
  5,226,614,784 bytes inside its 6 GiB scope with zero swap or memory events.
- Weighted extraction no longer assumes that a camera-seeded Čech walk can
  discover every PowerFoam component. CPU and GPU depth now integrate the
  independently clipped, depth-sorted support rows used by the exact headless
  renderer; RadFoam retains the walk. A disconnected-support oracle selects
  the farther dominant segment on both implementations. Dense/nested fixture
  depth metrics remain exact, and a 98,831-cell Room stress gate traces five
  64x42 maps in 0.6 seconds at a 242,622,464-byte host peak with no overflow,
  truncation, swap, OOM, or GPU fault.
- Commit `3211300` adds a cloud-only normal-direction plane sweep after depth
  fusion. Normalized world-space tangent patches, candidate-specific source
  depth visibility, and local spatial coherence recover all 49 particles in a
  synthetic 0.05-unit displacement while leaving an exact surface fixed. On
  same-binary gates it changes Bonsai train/held-out PSNR from 14.89/14.03 to
  14.97/14.07 dB and Room from 14.59/14.14 to 14.59/14.17 dB. Coverage is
  unchanged and Bonsai's worst held-out view improves 0.19 dB; one training
  tail regresses 0.11 dB. The pass costs 0.11–0.20 seconds and moves only
  11–12% of the clouds, so it is selected as an initializer refinement rather
  than evidence that shared-surface reconstruction is solved. The latest full
  workspace suite peaks at 5,069,111,296 bytes inside 6 GiB with zero swap,
  pressure, OOM, or GPU faults.
- Follow-up gates reject a second identical sweep, texture floors 0.005/0.02,
  and texture-first view selection. Repeating improves Room held out by 0.03
  dB but loses Bonsai average quality and coverage; threshold changes and view
  reordering also regress at least one scene. The diagnostic finds four
  geometrically valid/source-visible views for only 7,786 Bonsai and 9,504
  Room surfels, before texture reduces those to 5,445/5,407. Better visibility
  support is the next constraint; loosening local photometric confidence is
  not selected.
- Commit `cfb7005` adds an analytic textured-sphere gate with curvature and
  self-occlusion. All 25 of 49 displaced particles that retain four valid
  views recover the exact 0.0375 radial correction; an exact sphere scores all
  49 and moves none. Three rounds of similarly oriented neighbor propagation
  fill most of the synthetic support gap, but lose 0.01 dB held-out Bonsai
  under both 75% and unanimous sign agreement; Room is neutral for the broader
  form. The propagation code is removed; unsupported particles require an
  independent cue.
- A robust source-depth fallback scores all 7,786/9,504 geometrically supported
  Bonsai/Room particles and improves Room average train/held-out PSNR by
  0.02/0.01 dB, but loses 0.01 dB on the worst held-out view. A stricter 5%
  acceptance threshold retains the same tail. Because it reuses the foam depths
  that initialized fusion rather than adding evidence, it is removed and those
  depths remain visibility checks only.
- Re-estimating normals only among photometrically moved neighbors also fails
  the cross-scene tail gate. A 25% local-plane blend raises Bonsai held out by
  0.04 dB but lowers Room's worst held-out view by 0.02 dB; a 10% blend removes
  the mean gain and lowers both Bonsai tails by 0.02 dB. The implementation is
  removed. Fused local neighborhoods remain too noisy to supply final normals.
- Commit `b593e3c` reuses one relight tracer and prefiltered environment across
  reconstruction's train/test score splits. Five complete Room runs fall from
  a 3.25-second median to 3.18 seconds (1.022×), with median user CPU down
  3.36→3.06 seconds, unchanged 293 MB peak memory, byte-identical scene output,
  and unchanged Bonsai/Room scores. The full GPU workspace suite passes at a
  1,825,452,032-byte peak with zero swap, pressure, OOM, or GPU faults.
- The physical path-record integration tests used to initialize two GPU
  contexts concurrently under Cargo's default test threading. On this NVIDIA
  driver that can leave one test busy-waiting indefinitely even though its
  4 GiB scope stayed below 620 MB and recorded zero swap, pressure, OOM, or GPU
  faults. A process-local mutex now serializes only those two hardware tests.
  They pass in 3.61 seconds with a 259 MB peak; the unmodified concurrent
  workspace command subsequently passes in 25.82 seconds with a 476 MB peak
  and zero swap, pressure, OOM, or GPU faults.
- When a topology and densification boundary coincide, the trainer now
  refreshes adjacency and the GPU cloud before collecting contribution and
  resampling statistics. Previously it downloaded current positions but paired
  them with the preceding GPU geometry for that scan. A moving-geometry,
  prune-and-densify physical-GPU test covers the corrected operation order.
- Contribution view/phase rotation is keyed by the absolute densification
  round, not the generic session-rebuild counter. Topology-cadence ablations
  therefore see identical contribution samples at the same global boundary;
  segmented resumes derive the same round directly from the absolute step.
- The historical 256-ray Smooth-L1 arm's 4.44 dB held-out failure is unchanged
  at twice the traversal cap and under either background. That result does not
  transfer to fresh 4,096-ray/16-view geometry formation: the matched
  step-6,000 gate selects Smooth-L1 by +0.84 dB selected and +0.33 dB all-37,
  and the completed 20,400-step continuation holds +0.48 dB selected and
  +0.25 dB all-37. The loss remains opt-in until the same result reproduces on
  another scene.
- Deterministic per-view spatial pixel stratification does not transfer across
  scenes or losses. Starting it before geometry warmup lowers Bonsai selected
  held-out PSNR by 0.31 dB at step 2,000. Enabling it after the shared warmup
  raises the 200,000-cell step-8,000 Bonsai result from 22.72 to 22.95 dB on
  the selected eight views and from 23.33 to 23.46 dB over all 37. The same
  continuation lowers Room's all-39 result from 23.76 to 23.68 dB under L1.
  Under Smooth-L1, a favorable selected-eight change of +0.06 dB reverses on
  all 39 views, from 23.10 to 23.06 dB. Runtime is neutral and all scopes stay
  below 1.3 GiB with zero swap, OOM, validation message, Xid, or GPU fault.
  The implementation is removed. Uniform screen coverage is not a robust
  proxy for the distributed far/background residual; the next sampler gate
  must use an explicit residual or visibility signal and preserve an unbiased
  loss estimate.
- Full residual attribution rules out literal empty-ray holes at the current
  growth checkpoints. For each scene, all of the worst 512 overpredicted and
  512 underpredicted held-out pixels hit a peak cell. Median opacity is 1.0000
  in Bonsai and at least 0.9996 in Room. The errors span 403/419 unique
  Bonsai cells and 276/273 unique Room cells; their median training-view
  support is 46/65 and 62/82 for over/underprediction. Missing support is not
  the dominant failure at this resolution.
- Replacing RadFoam's position-gradient densification score with the existing
  per-cell photometric-responsibility EMA confirms that distinction. From the
  exact step-6,000 state it improves Bonsai selected-eight/all-37 PSNR by
  0.16/0.06 dB at the 200,000-cell step-8,000 boundary. From Room's exact
  step-4,000 state it instead lowers selected-eight PSNR by 0.27 dB at step
  6,000. Applying PowerFoam's established 99th-percentile cap still loses
  0.21 dB. Both forms keep the existing cell-radius weighting and add no new
  operation or shader, but they replace a geometry signal with a scene-mixed
  appearance signal. A matched Room score dump also removes the rationale for
  a tuned blend: the existing gradient already puts 58.3%/79.1% of unique
  held-out over/underprediction cells in its top 15%, versus 57.4%/84.7% for
  the residual. Their rank correlation is 0.58, but their top-10% sets overlap
  only 32.7%; photometric scoring mostly reorders already-visible error sites
  instead of finding missing ones. The prototype and dump instrumentation are
  removed. Before changing support again, compare complete per-cell training
  and held-view errors to separate appearance underfit from geometry that
  fails to generalize.
- The complete per-cell comparison now closes that question at the sampled
  checkpoints. Every 128x128 pixel in all 272/39 Room and 255/37 Bonsai
  train/held views was rendered on the GPU and independently traced to its
  peak cell with the CPU oracle. Shared train-visible cells own 99.73% of
  Room's and 99.43% of Bonsai's held residual mass. Among cells hit in at
  least two views on each side, log-mean train/held squared-error correlation
  is 0.7999/0.7494, and 61.06%/51.44% of held top-decile cells are also in the
  training top decile. The exact passes cover 5,095,424 Room and 4,784,128
  Bonsai rays; all have a peak-cell attribution. They finish in 16/29 minutes
  with 192/172 MB host peaks and zero swap, OOM, or GPU fault events. This
  rules out view-exclusive cell ownership as the main residual source, but
  does not by itself distinguish appearance capacity from systematic geometry
  error on shared cells. The bounded next gate is residual-aware pixel
  sampling with inverse-probability weights, preserving the existing loss in
  expectation. The one-off attribution implementation is removed; complete
  generated logs remain under `target/audit-runs/cell-residual-attribution/`.
- A fixed initial-residual pixel distribution with exact inverse-probability
  loss weights does not pass the cross-scene gate. A 50% uniform mixture
  improves Bonsai train/all-37 PSNR from 24.0121/23.3282 to
  24.2762/23.5440 dB and improves 35/37 held views. On Room it changes
  23.7792/23.0983 to 23.7745/23.1035 dB, improves 25/39 views, but loses
  1.67 dB on one held view. A safer 75% uniform mixture instead lowers Room
  train/all-39 quality to 23.7548/23.0179 dB. The full-view residual prepass
  also raises matched pipeline time by 39.7% on Room and 22.8% on Bonsai.
  All three 12 GiB scopes stay at or below 1.5 GB host memory with zero swap,
  OOM, or GPU faults. The prototype weighted color, opacity, distortion,
  quantile, normal, and densification terms without adding an operation or
  shader, but the large topology-dependent Room variance means unbiasedness
  in expectation is not enough for a finite training trajectory. The CLI,
  graph inputs, and sampler are removed; generated artifacts remain under
  `target/audit-runs/residual-importance-sampling/`.
- A direct cap gate on the selected fresh Smooth-L1 Bonsai trajectory rejects
  capacity as the next quality lever. From the exact 200,000-cell step-8,000
  state, matched continuations to step 10,000 score 24.4968/23.2723/23.7049 dB
  train/selected-eight/all-37 at 200K, 24.5809/23.2673/23.7214 dB at 230K, and
  24.5742/23.1781/23.7189 dB at 300K. The tiny +0.0165/+0.0140 dB all-view
  gains cost 17.8%/35.6% more pipeline time; the 230K arm loses 1.01 dB on one
  held view and the 300K arm loses 0.54 dB on the same tail. Host peaks are
  1.27/1.32/2.05 GB with zero swap, OOM, throttling, or GPU faults. Fixed 200K
  remains selected; no cap-specific code or control is retained. Generated
  checkpoints remain under `target/audit-runs/robust-bonsai-cap-gate/`.
- That gate exposed a separate resume correctness bug: default-zero per-view
  exposures deliberately have parameters but no gradient or Adam state, while
  the densification rebuild unconditionally requested their moments. The
  snapshot now always preserves exposure values and carries moments only for
  channels which actually have gradients. A physical-GPU pixel-batched test
  crosses a prune/densify session rebuild with frozen exposure and continues
  training with finite loss.

### Remaining PowerFoam gaps

- Bounded traversal, persistent radii, exact active-path Jacobians, and
  trainable positive radii are implemented. The WGSL-Jacobian-to-meganeura
  integration suite now passes on a physical NVIDIA GPU, including finite
  differences, weighted intervals, topology rebuilds, densification, and
  multi-view/novel-pose cases.
- Weighted-cloud densification follows the reference resampler's photometric
  responsibility signal, 99th-percentile cap, copied radius, conserved sibling
  statistic, and 5%-support-scale split. The original learned-radius causal
  gate remains valid, but commit `2a0fd73` showed that its absolute trajectory
  used the wrong SH-DC multiplier. The fresh intended-schedule run reaches
  15.72/15.28 dB at 175,895 sites and finishes the selected 200,000-site
  step-20,000 trajectory at 17.4151/16.5247 dB with 8,328,306 directed edges.
- The real checkpoint/resume path reaches 100,569 sites at step 4,000 and
  improves Bonsai train/all-37 PSNR to 14.65/14.35 dB. Conservative projected
  candidates preserve those pixels exactly and make the complete weighted
  evaluation 4.4% faster; sparse 16-view training correctly retains the
  exhaustive gather because its matched indexed arm was neutral. Packing the
  differentiable `(position,radius)` roles then lowers the 2,000-step training
  memory peak by 33.4% and wall time by 1.6%, with quality within 0.02 dB.
  Exact per-ray GPU status then selects a 128-step budget: all 4.78 million
  evaluation rays and 8.19 million optimizer rays complete without
  truncation, reducing the matched training run by another 7.4% and its peak
  by 32.9%. A 64-step render silently clipped 0.0762% of full-view rays and is
  retained only as an approximate setting.
- Candidate-hit pressure is now measured separately from surviving path depth.
  The step-6,000 cloud needs 647 candidate supports but only 127 path intervals;
  a 1,024-candidate floor fixes the former 512-entry rejection without enlarging
  every path/Jacobian row. All 37 held-out renders are byte-identical between
  128- and 256-step paths. The smaller path graph cuts physical/device-local
  allocation by 45.4%/45.6% and matched training time by 16.4%.
- On the corrected trajectory, adding `1e-6` overlap control only after step
  8,000 reduces the matched step-10,000 graph 11,117,102→9,974,452 edges,
  changes held-out PSNR 15.88→15.89 dB, and cuts training time 4.2%. Exact
  160-step rows and cadence-200 topology remain selected after the 200K cap.
- Starting that loss at step 4,000 is the better complete trajectory. At the
  exact step-6,500 cap it changes train/all-37 quality
  15.8363/15.3752→15.8389/15.3845 dB and graph size
  11,145,224→10,083,298. At step 20,000 it reaches
  17.4151/16.5247 dB and 8,328,306 edges versus post-cap-only
  17.4077/16.5243 dB and 9,150,448 edges. Sampled fractional overlap falls to
  median/p90/p99 0.224/0.626/0.887, but 256² renders retain the same discs,
  holes, blurred thin structure, and background floaters.
- Raw density is not the missing quality mechanism. A bounded 400,000-site
  branch reaches 16.21/15.62 dB at step 10,000 versus 16.15/15.61 for the
  200,000-site control, while exact Čech adjacency grows 8.89M→49.90M directed
  edges. The 400K graph spends 100.6 of 365.2 seconds per 1,000 steps in
  topology/resource rebuilds and does not visibly repair the scene, so it is
  rejected.
- The 400K run found a correctness boundary before any memory boundary: a
  264,500-site ray needed 1,050 sphere candidates despite a much shorter path.
  Commit `6184551` makes the minimum candidate capacity explicit and independent
  of the path/Jacobian row. Resumed 304K training and full evaluation pass with
  2,048 candidates and zero truncation; the eventual 400K checkpoint uses at
  most 1,822/2,048 candidates and 171/192 path entries.
- A cloud-only surface-aware split prototype estimates a normal from each
  site's 12 nearest Čech neighbours and projects the reference 5%-radius jitter
  into the tangent plane when least-normal variance is at most 5% of total
  local variance. 88.6% of 20,000 sampled step-4K neighbourhoods pass that
  confidence gate, but the matched step-6,500 arm ties isotropic splitting at
  14.95 dB held out with effectively identical edges and runtime. It is not
  landed; learned orientation plus spatial texel appearance, not split direction
  alone, is the remaining reference-semantic hypothesis.
- Fixed adjacency-derived surface planes are negative, but learnable dipoles
  recover that loss with sufficient convergence. The exact implementation
  persists optional per-site normals, clips CPU and every WGSL traversal path
  to the retained half-cell, differentiates the active plane, and trains
  normalized normals with an optional decaying view-facing loss. That reference
  loss costs 0.0510 dB held out in the matched 2,040-step 200K-site Bonsai
  refit, so the selected arm disables it. Loss-free normals reach
  17.4388/16.3905 dB train/all-37 at
  8,160 steps and 17.5798/16.4097 at 16,320, versus 17.2334/16.4105 for the
  unoriented endpoint. The practical endpoint is within 0.020 dB and the long
  endpoint within 0.001 dB held out. The feature therefore lands opt-in; it
  establishes geometry parity but visible support discs and speckle still keep
  it from replacing the default or establishing the full appearance model.
- A learned signed offset for each oriented surface plane now moves that
  geometry beyond parity without adding polygonal state or rebuilding Čech
  topology. On the matched 200K-site Bonsai gate, duration-adjusted offset
  rates improve normal-only held-out PSNR by +0.0632 dB at 4,080 steps,
  +0.0678 at 8,160, and +0.0676 at 16,320. The selected 8,160-step model is
  17.4614/16.4583 dB train/all-37 held out and improves 32/37 test views; the
  16,320-step ceiling is 17.6039/16.4773 but adds only 0.0190 dB for 2× time.
  The selected run records 33.42M rays, max 106/128 paths, zero truncation,
  627 MB peak host memory, and no swap, pressure, OOM, or GPU fault. The
  independent 200K-site Room gate is stronger: ratio 0.005 raises
  25.0824/23.0112 to 25.3393/23.2402 dB and improves 37/39 held-out views,
  while only 0.45% of planes leave their support. Ratio 0.01 reaches 23.2607
  dB held out but raises that tail to 2.92%, so 0.005 is the safer cross-scene
  choice. Offsets remain opt-in because they are one value per site, not the
  reference detail-site/spherical-Voronoi appearance model.
- An opt-in four-component spatial surface-color residual now addresses part
  of the remaining within-cell appearance gap without introducing meshes or a
  per-site tangent frame. Each oriented site stores 12 floats and evaluates
  RGB against `(q.x, q.y, q.z, min(dot(q,q), 1))`, where `q` is the query on
  the displaced plane, projected tangent and normalized by support radius.
  The model, ASCII/binary PLY, CPU oracle, all WGSL render paths, Meganeura
  training graph, pruning/densification ancestry, and exact resume path share
  the same layout. Zero coefficients preserve the preceding pixels exactly.
  A matched 8,160-step Bonsai replay at ratio 0.02 reaches
  17.5094/16.4759 dB train/all-37 held out, +0.0480/+0.0176 dB over the
  oriented-offset control and improves 24/37 held-out frames. Training rises
  497.689→521.475 seconds (+4.8%), the 200K-site PLY grows 77→86 MiB, while
  complete GPU evaluation changes only 28.844→28.972 seconds. The independent
  2,040-step Room replay reaches 25.4107/23.2551 dB train/all-39 held out,
  +0.0714/+0.0149 dB, improving 22/39 frames. Its conservative 256-row graph
  raises training 160.571→184.053 seconds (+14.6%) and peak host memory
  761,446,400→1,241,747,456 bytes; all rays actually fit within 110 rows, so
  the already selected 160-row budget removes avoidable padding. Both 6 GiB
  scopes record zero swap, pressure, OOM, truncation, or GPU fault. Ratio 0.02
  is selected for the experimental spatial arm, but the option remains off by
  default: the cross-scene gain is consistent yet small, some tail views
  regress, and this linear basis is not the reference detail-site and
  spherical-Voronoi color model.
- The matched 160-row Room replay closes that performance gate: it reaches
  25.4133/23.2645 dB, slightly above the separate 256-row run, while training
  falls 184.053→148.774 seconds (-19.2%), GPU-step wait
  129.448→96.320 seconds (-25.6%), command time 188.697→153.394 seconds
  (-18.7%), and peak host memory 1,241,747,456→1,003,597,824 bytes (-19.2%).
  All 8,355,840 optimizer rays fit within 110/160 entries, candidate use peaks
  at 227/1,024, and the scope records no swap, pressure, OOM, or GPU fault.
- The official PowerFoam appearance source was audited at commit `9639225`.
  Its complete per-point increment is 412 floats: one quaternion, eight 2D
  detail sites, eight heights, and eight directional functions with eight
  axis/RGB pairs each. At 200K points that is about 314 MiB for parameters
  before Adam moments or graph intermediates, so it will be staged and gated
  component by component. The audit also found two paper/source differences:
  paper equations 3/4 print an unsquared spatial norm while the Warp texture
  kernel uses squared normalized distance at temperature 10; the standalone
  Spherical Voronoi definition is dot-product softmax while the PowerFoam
  kernel uses temperature-scaled chordal distance. Those alternatives need
  explicit versioned semantics rather than being silently conflated.
- The bounded directional slice is now complete end to end. It stores eight
  raw axes plus eight RGB values (48 floats per point), follows the published
  dot-product-softmax contract, initializes to an exact zero residual, and has
  CPU/WGSL/Meganeura, PLY, densification, and exact-resume coverage. It is an
  additive directional residual, not a claim of released-checkpoint parity.
  On the matched 2,040-step Room gate, learned axes/colours reach
  25.4019/23.2477 dB train/all-39 held out, below the 25.4133/23.2645 dB
  spatial-only control, while training rises 148.774→181.549 seconds and the
  PLY grows 66→103 MiB. Fixed cube-corner axes recover 25.4116 dB train but
  only 23.2508 dB held out at the same cost. The feature remains opt-in and a
  second-scene run is rejected until a spatial detail-site formulation gives
  it a stronger reason to exist.
- The compact spatial detail-site slice is now implemented end to end: eight
  tangent sites, eight radius-scaled heights, and eight view-independent RGB
  residuals add 56 floats per point. CPU and every WGSL renderer share the
  same two-stage height/color rule, while Meganeura consumes the recorder's
  support query and branch mask directly. Deterministic zero-preserving
  initialization, PLY, one-submit GPU geometry refresh, densification/Adam
  inheritance, and exact resume are covered. Position/radius learning now
  carries the support-entry query's fixed-topology sphere/radical-face
  Jacobian, while frozen detail retains its compact payload. The matched
  200K-site Room gate selects site/height/colour ratios `0.1/0.05/0.05`:
  two replicas improve the control mean by +0.2337/+0.1513 dB train/held out
  and improve 37/39 held-out views. Two tail views lose 0.115 and 0.070 dB,
  while a matched two-replica 8,160-step Bonsai gate improves the control by
  +0.1059/+0.0701 dB and improves 33/37 held-out views, ties two, and loses
  only 0.010/0.005 dB on the other two. The arm passes the cross-scene quality
  gate but remains opt-in: its Bonsai PLY grows 89,964,416→134,766,940 bytes
  (+49.8%) and training grows 344.325→463.063 seconds (+34.5%). A pre-gather
  site projection optimization is rejected despite a 6.0% wall-time win
  because 17/39 averaged Room views regress. Quality remains the selection
  boundary.
- Joint detail/geometry training passes a two-scene gate. Two 2,040-step Room
  replicas at position/radius ratios `0.01/0.01` improve frozen-detail means
  from 25.6449/23.4125 to 26.4852/24.2963 dB train/held out and improve 38/39
  averaged held-out views; mean edges grow 7.9%. The same Bonsai probe improves
  17.1996/16.3188→17.3405/16.7241 dB but improves only 22/37 views and grows
  edges 5.3%. The reference radius ratio `0.0005` is more conservative:
  24.1554 dB Room and 16.4668 dB Bonsai held out while shrinking topology on
  both scenes. Full geometry raises training time by 57.1% on Room and 67.8%
  on Bonsai, so both geometry and its rate remain opt-in. The implementation
  adds two active `vec4` query-gradient streams only in full-detail mode;
  sphere/face CPU finite differences, 14 physical GPU path oracles, a complete
  graph finite difference, joint updates, densification, and bit-exact resume
  pass. All measured scopes remain untruncated and free of memory/GPU events;
  the complete serialized workspace suite peaks at 5.37 GB (5.00 GiB) under
  the 6 GiB limit with no event.
- The detail-entry and interval linearizations now share the three unique
  ray-relative path-role geometry gathers. A two-replica 32-step profile cuts
  graph time by 5.7%; an order-balanced 510-step gate cuts training by 3.6%
  with a +0.0044 dB held-out change. At 2,040 steps, two fresh candidates cut
  mean training 150.107→147.417 seconds (-1.8%) and GPU wait by 2.6%, while
  changing train/held-out PSNR from 26.4852/24.2963 to 26.4991/24.3063 dB and
  mean topology by -0.04%. All paths remain exact and all isolated scopes are
  free of memory and GPU events. This selects graph sharing while leaving a
  fused indexed geometry-dot operation as a separately measurable follow-up.
  Formatting, strict all-target lint, and the complete serialized workspace
  suite pass; the suite peaks at 3.16 GB (2.94 GiB) under the 6 GiB limit with
  no event.
- Meganeura `2acdfeb` now folds shared pointwise and embedding producers into
  reduction prologues only when every consumer is compatible. Blade's graph
  falls 478→470 passes and mean graph time falls 5.3%. Full 2,040-step Room
  training falls 147.417→144.434 seconds (-2.0%) with a -0.0126 dB held-out
  change and an 18/21 improved/regressed view split. An order-balanced Bonsai
  gate falls 92.085→89.907 seconds (-2.4%) with -0.0105 dB held out and a
  19/18 split. Topology changes remain within 0.08%; all paths are exact and
  all isolated scopes are free of memory/GPU events. White-box fusion,
  fused/unfused value parity, and analytical shared-gradient tests pass. Four
  unrelated standalone Meganeura attention/model tests reproduce identical
  failures at the pinned baseline; strict lint and every unaffected all-target
  test pass. Blade's complete serialized workspace suite passes against the
  remote pin at a 3.55 GB (3.31 GiB) peak with no memory or GPU event.
- The two spatial-detail plane queries now share all geometry that is invariant
  to their signed offset. The 200K-site graph falls 470→462 passes. At 2,040
  steps, two Room replicas reduce mean training 144.434→143.652 seconds
  (-0.5%) and GPU wait by 0.9%, while held-out PSNR changes
  24.2937→24.3179 dB and averaged views split 23/16 improved/regressed. Two
  Bonsai replicas reduce training 89.907→89.603 seconds (-0.3%) and GPU wait
  by 0.4%, while held-out PSNR changes 16.7515→16.7673 dB and views split
  17/19/1. Topology changes stay within 0.19%. The complete query finite
  difference, joint-update, exact-resume, formatting, strict lint, and
  serialized workspace gates pass. The clean warm-cache suite rerun peaks at
  217.8 MiB under 6 GiB with no memory or GPU event.
- A five-rate Bonsai screen followed by a two-scene 8,160-step gate rejects a
  universal learned-radius rate. Room selects `0.005`: it reaches 25.6867 dB
  held out versus 25.5704 at `0.0005` and 25.6062 at `0.01`, improves 29/39
  and 25/39 views respectively, and ends at 3.189M edges. Bonsai selects
  `0.01`: it reaches 18.6322 dB versus 17.9072 at `0.005`, improves 31/37
  views, and ends at 8.930M rather than 8.517M edges. The associated degree
  maxima are 238 for selected Room and 5,648 for selected Bonsai, so the
  latter retains a hub-tail warning. All rays remain exact and the combined
  long scope peaks at 1.14 GB without memory or GPU events. Visual comparisons
  confirm the relative choices but retain Bonsai's large pale support blobs,
  holes, and background floaters; another identical fixed-cap continuation is
  not the next quality experiment. Geometry therefore remains opt-in with a
  held-out scene-level rate bracket rather than a changed global default.
- Meganeura `7967ca2` adds one-transfer F32 parameter readback through cached
  download memory, and Blade uses it for every model/geometry snapshot. It
  leaves the training graph and values unchanged while reducing four-replica
  mean detail state readback from 15.051 to 0.864 seconds (-94.3%). Two fresh
  2,040-step replicas average 126.293 seconds versus 141.018 seconds before
  (-10.4%) and land inside the previous PSNR spread at 23.4041 and 23.4022 dB
  held out. Peak scoped memory is 1,658,667,008 bytes with no swap, pressure,
  OOM, kill, or GPU fault.
- Meganeura `226041c` extends the one-transfer path to batched Adam moments,
  and Blade uses it at densification boundaries. On a real 200K→200.1K Bonsai
  detail model, order-balanced state readback falls 11.585→0.185 seconds
  (-98.4%, 62.8×) and whole-command time falls 16.890→5.471 seconds (-67.6%,
  3.09×). Isolated peak memory is effectively unchanged at 1,863,389,184
  versus 1,861,894,144 bytes, with no cgroup or GPU fault. Exact multi-name
  Meganeura tests and Blade's detail densification/Adam-remap test cover the
  ordering and value contract.
- Meganeura `760bb29` makes its direct inner broadcast public and
  differentiable, allowing Blade to replace scalar-to-XYZ and eight-site
  normalization concat/split trees. The Bonsai detail profile falls 537→491
  passes and about 21.0→19.5 ms of steady GPU time. Four Room replicas reduce
  training 126.674→122.261 seconds (-3.5%) and change held-out PSNR
  23.3988→23.4061 dB; two Bonsai replicas reduce training 117.223→114.856
  seconds (-2.0%) and change held-out PSNR 16.3152→16.3140 dB. Room improves
  22/39 averaged views with a 0.0575 dB worst regression; Bonsai's worst and
  best deltas are -0.04/+0.04 dB. Physical value/gradient coverage and exact
  resume pass, so the cross-scene gate selects the direct broadcast.
- Meganeura `048c8be` then replaces the last three-level detail-vector concat
  tree with one differentiable inner tile. The Bonsai profile drops 491→461
  passes and 19.63→17.98 ms. Full 2,040-step Room replicas improve training
  122.261→115.639 seconds (-5.4%) and held-out PSNR
  23.4061→23.4138 dB; Bonsai improves training 114.856→111.409 seconds
  (-3.0%) while held-out PSNR is neutral at 16.3140→16.3137 dB. Room's
  averaged per-view mean is +0.0075 dB; Bonsai's is -0.0004 dB, with balanced
  extrema inside replica noise. Exact physical forward/backward coverage,
  exact resume, shader validation, strict lint, and the practical full
  Meganeura suite pass. Scoped Room/Bonsai peaks are 1.60/1.52 GB with no
  swap, pressure, OOM, kill, or GPU fault.
- Dense PowerFoam interval clipping now assigns a 64-lane workgroup to each
  ray, while graphs below 32 adjacency entries per site retain the original
  serial shader. On the 41.7-entry/site Bonsai checkpoint, the record pass
  falls from about 26.4 to 4.4 ms and matched training falls
  111.409→67.035 seconds (-39.8%) at neutral held-out quality
  (16.3137→16.3171 dB). Applying the same kernel universally is rejected: the
  15.1-entry/site Room scene regresses to 117.378 seconds. The selected
  adaptive build preserves the original Room path at 116.229 versus 115.639
  seconds and 23.4010 versus 23.4138 dB held out. All 14 physical path oracles
  pass, and scoped Room/Bonsai peaks are 1.4/1.3 GB with no memory or GPU
  fault events.
- Meganeura `508c0ab` computes each detail query's eight squared site
  distances directly, including both gradients, instead of materializing
  48 MiB query/delta/square tensors. The production detail graph falls
  461→454 passes and about 17.9→16.5–16.8 ms. Two Room replicas average
  112.039 seconds versus the current 116.229-second replay (-3.6%) and two
  Bonsai replicas average 64.567 versus 67.035 seconds (-3.7%). Held-out means
  remain neutral at 23.4102 versus 23.4138 dB on Room and 16.3147 versus
  16.3137 dB on Bonsai; per-view gains/regressions are balanced. Physical
  forward/backward, detail training/remap, exact resume, shader/SPIR-V, and
  cross-scene gates pass without memory or GPU fault events.
- Meganeura `f31ef08` similarly projects the eight learned site vectors off
  each shared unit normal in one differentiable operation. Blade no longer
  materializes the 48 MiB normal tile or its dot/broadcast/projection chains:
  the detail graph falls 454→446 passes and warmed GPU time falls about 9.8%,
  from 16.53–16.86 to 15.05–15.12 ms. Two Room replicas reduce mean training
  112.039→104.906 seconds (-6.4%) and GPU wait 67.314→62.521 seconds (-7.1%);
  held-out PSNR is neutral at 23.4102→23.4108 dB. Two Bonsai replicas reduce
  mean training 64.567→61.479 seconds (-4.8%) and GPU wait
  54.521→51.540 seconds (-5.5%); held-out PSNR changes
  16.3147→16.3120 dB. Averaged per-view means are +0.0010 dB on Room and
  -0.0027 dB on Bonsai. The physical forward/two-gradient oracle, shader and
  SPIR-V validation, strict lint, pinned Blade detail/resume tests, and the
  complete locked Blade workspace suite pass; all measured scopes record zero
  OOM or GPU faults, and the full suite peaks at 2.5 GB.
- Meganeura `22bb539` exposes its existing pointwise exponential through the
  graph and autodiff, so detail weights use `exp(-x)` rather than
  `recip(sigmoid(x)) - 1`. The direct mathematical form removes eight passes:
  the detail graph falls 446→438 and warmed GPU time falls about 3.2%, from
  15.05–15.12 to 14.56–14.68 ms. Two Room replicas reduce mean training
  104.906→102.906 seconds (-1.9%) and GPU wait 62.521→61.006 seconds (-2.4%);
  held-out PSNR rises 23.4108→23.4202 dB. Two Bonsai replicas reduce mean
  training 61.479→59.140 seconds (-3.8%) and GPU wait 51.540→49.840 seconds
  (-3.3%); held-out PSNR changes 16.3120→16.3139 dB. Physical exponential
  forward/gradient and shader-path parity, the Meganeura library/pointwise
  suites, strict lint, Blade's focused and complete locked workspace suites,
  and cross-scene gates pass without memory or GPU faults. The full Blade
  suite peaks at 2.4 GB.
- Meganeura `85e919f` fuses each surface-detail row sum, positive floor,
  reciprocal broadcast, and expanded backward chain into differentiable
  `normalize_inner_sum` dispatches. The detail graph falls 438→418 passes;
  settled short-profile endpoints fall from 14.56–14.68 to 13.66–13.97 ms.
  Two Room replicas reduce mean training 102.906→100.434 seconds (-2.4%) and
  GPU wait 61.006→59.251 seconds (-2.9%), while held-out PSNR is neutral at
  23.4202→23.4208 dB. Two Bonsai replicas reduce mean training
  59.140→57.729 seconds (-2.4%) and GPU wait 49.840→48.580 seconds (-2.5%);
  held-out PSNR changes 16.3139→16.3172 dB. An initial factored-gradient
  prototype was rejected after four Room replicas exposed a -0.0187 dB
  held-out shift. The selected kernel instead preserves the expanded graph's
  per-site reduction order and runtime rounding barrier; a 257×8 physical
  oracle is bit-exact for every forward value and gradient. Meganeura's 177
  library and 13 serial reduction tests, shader/SPIR-V/runtime-binding checks,
  strict lint, Blade's focused detail/resume gates, and the complete locked
  workspace suite pass. Room/Bonsai scopes peak at 1.45/1.15 GB and the full
  suite at 2.54 GB, all without swap, pressure, OOM, kill, or GPU fault.
- PowerFoam path gathering now retains each accepted sphere's discriminant
  root in existing candidate-depth scratch, allowing both record kernels to
  reconstruct the same `-b ± root` bounds without a second discriminant and
  square root. A balanced profile reduces the dense parallel record pass
  4.504→4.208 ms and total recorder time 7.972→7.660 ms. Two-replica Room and
  Bonsai gates reduce mean training 100.359→99.821 and 57.902→57.712 seconds
  at neutral held-out quality (23.4102→23.4184 and 16.3162→16.3148 dB).
  All 14 physical CPU-parity path tests pass; 33.42 million long-gate rays
  remain untruncated with no candidate overflow, memory event, or GPU fault.
  Strict all-target lint and the complete locked serialized workspace suite
  pass; the latter peaks at 5.75 GB under its 6 GiB cgroup with no event.
  Broader row tiling, workgroup ray sharing, and vec2 sphere-face caching were
  measured and rejected as neutral or slower.
- Spatial detail and the spatial surface-colour basis now share the current
  site's already-gathered center. This removes two redundant graph dispatches
  (418→416) and one of 15 large embeddings without changing forward arithmetic.
  Balanced last-20 GPU timestamps improve 13.952→13.766 ms. Two-replica Room
  training improves 100.296→99.663 seconds with held-out PSNR
  23.4075→23.4109 dB; Bonsai is timing-neutral at 57.682→57.630 seconds while
  held-out changes 16.3125→16.3176 dB. Both gates remain untruncated and peak
  below 1.73 GB with no memory or GPU fault event. Splitting the recorder's
  interleaved detail query into scalar buffers was rejected: graph time falls
  0.58%, but recorder time rises 8.4% and combined GPU time regresses 2.65%.
  Strict all-target lint and the complete locked workspace suite pass; the
  latter peaks at 6.10 GB under 6 GiB with no memory or GPU fault event.
- Those same appearance branches now also share the gathered, positively
  floored surface radius. This removes another two dispatches (416→414) and
  one of 14 large embeddings. Balanced last-20 GPU timestamps improve
  13.756→13.599 ms (-1.14%). In two-replica ABBA gates, Room training improves
  99.839→98.754 seconds (-1.09%) and GPU wait improves 58.516→57.870 seconds;
  train/held-out PSNR changes by only -0.0026/-0.0009 dB. Bonsai training
  improves 57.375→56.633 seconds (-1.29%) and GPU wait improves
  48.294→47.538 seconds; train/held-out PSNR changes by -0.0001/+0.0010 dB.
  Both gates remain untruncated with unchanged candidate ceilings. Their
  6 GiB scopes peak at 1.72/1.37 GB with zero swap, pressure, OOM, kill, or GPU
  fault. A train-mode physical test covers the production gather/floor path
  and matches the independent CPU finite-difference radius gradient. Strict
  all-target lint and the complete serialized workspace suite pass; the latter
  peaks at 2.68 GB under its 6 GiB cgroup with no memory or GPU fault event.
- Exhaustive PowerFoam candidate gathering now assigns the WebGPU-guaranteed
  maximum of 256 lanes, rather than 64, to each exact 200K-site ray scan.
  Balanced 64/128/256-lane profiles measure 3.455/3.055/2.843 ms for the gather
  (-17.7% at 256), with the downstream record pass and 414-pass graph neutral.
  Two-replica Room training improves 99.699→96.906 seconds (-2.80%) and GPU
  wait improves 58.207→56.907 seconds; held-out PSNR changes by -0.0074 dB.
  Bonsai improves 57.288→56.034 seconds (-2.19%) and 48.106→46.876 seconds,
  with held-out PSNR changing by +0.0002 dB. Across all ABBA arms, 66.85
  million training and 38.40 million evaluation rays remain untruncated;
  candidate maxima stay below capacity, and the 6 GiB scopes peak at
  1.69/1.35 GB without a memory or GPU fault event. Removing interval sorting
  entirely saves at most a noisy 0.15 ms and immediately changes loss, so
  parallel-sort complexity is rejected. All 14 physical path tests, strict
  all-target lint, and the complete serialized workspace suite pass; the
  latter peaks at 5.69 GB under 6 GiB with no event.
- Profiling after that negative gate removes two pieces of no-op training
  work. Zero-weight view-facing normal regularization is absent from the
  production graph, reducing a representative step from 460 to 443 GPU passes
  and 30.62→29.54 ms. Path recording uses geometry-only GPU storage, avoiding
  48,800,000 bytes of persistent attributes plus the same transient upload on
  the selected 200K-point model. A full matched Room replay preserves
  25.4129/23.2680 dB while reducing training 148.774→146.390 seconds and total
  command time 153.394→150.587 seconds. All 8.36M rays remain untruncated and
  the 6 GiB scope reports no swap, pressure, OOM, kill, or GPU fault.
- A 128-row training graph is not selected despite exact traversal: training
  uses at most 110 rows and evaluation at most 109, but the different padded
  reduction shape changes the optimizer trajectory. It is 8.5% faster and
  reaches 25.4098/23.2487 dB, losing 0.0158 dB held out versus 160 rows.
  Exact paths are necessary but not sufficient evidence for a training
  configuration change.
- Zero-rate fixed-topology positions and radii are now detached from the
  production graph instead of accumulating unused gradients and Adam moments.
  Densification and reference schedules keep their required geometry paths,
  and the public low-level graph remains fully differentiable. This required a
  focused Meganeura compiler correction (`ad08f97`) so scalar placeholders for
  detached parameters are not registered as optimizer gradients. Structural
  gradient coverage and exact mixed-view/oriented/densifying resume tests pass.
  On the matched 2,040-step Room gate, the graph drops 443→419 passes and
  training falls 146.390→139.111 seconds (-5.0%) while held-out quality moves
  only 23.2680→23.2625 dB. GPU wait falls 6.3%; all training/evaluation rays are
  untruncated, and the 6 GiB scopes report no swap, pressure, OOM, kill, or GPU
  fault. Frozen checkpoints omit only unused moments; a later explicit
  geometry unfreeze starts those moments from zero.
- Degree-three SH training now uses six DC/rest parameter tables instead of 48
  scalar-column tables while preserving the exact model layout and separate
  learning-rate schedules. Legacy per-component parameter and Adam tensors are
  migrated on resume, and densification remaps each packed row as one site's
  state. The representative graph falls 419→293 passes; four-run median GPU
  time falls 26.88→25.99 ms. The matched 2,040-step Room gate cuts training
  139.111→133.847 seconds (-3.8%) and whole-command time 143.588→138.415
  seconds (-3.6%) while preserving 25.4112/23.2624 dB versus
  25.4113/23.2625 dB. All rays remain untruncated and the 6 GiB cgroup records
  no swap, pressure, OOM, kill, or GPU fault.
- Four per-ray sums now use the graph's direct inner-dimension reduction rather
  than multiplying by an all-ones column. This removes the synthetic tensor and
  four GPU passes; three alternating profiles improve median GPU time
  25.94→25.66 ms (-1.1%). The matched 2,040-step replay has effectively flat
  GPU wait (86.933→86.908 seconds) and a noisy wall-time regression, so no
  end-to-end speedup is claimed. Exact-resume and affected numerical oracles
  pass, Room quality is preserved at 25.4134/23.2697 dB, and all rays remain
  untruncated without cgroup or GPU faults.
- Fixed-topology weighted training no longer evaluates complete
  position/radius path differentials when those tables have zero learning
  rates. The recorder emits only the oriented surface-plane reference tangent
  and derivative; full geometry and densification retain the complete stream.
  CPU↔WGSL, local-linearization, zero-gradient, and all three exact-resume
  regimes pass. Three alternating profiles improve 25.84→20.74 ms (-19.7%)
  and cut the `[4096, 160]` differential allocation by 32.5 MiB. The matched
  Room replay reduces training 137.618→114.757 seconds (-16.6%) and GPU wait
  86.908→68.631 seconds (-21.0%), while preserving 25.4170/23.2647 dB and
  zero truncation. Peak memory is 719 MiB with no cgroup or GPU faults.
- Four scalar-to-XYZ copies in the oriented spatial-colour basis now use exact
  row-wise concatenation instead of tiled matrix multiplication. Direct
  row-order and CPU-oracle tests, exact resumes, and the full workspace GPU
  suite pass. Three alternating profiles improve median GPU time
  20.81→19.94 ms (-4.2%). A back-to-back Room pair reduces training
  115.862→111.844 seconds (-3.5%) and GPU wait 69.291→65.990 seconds (-4.8%)
  while preserving 25.4159/23.2685 dB versus 25.4115/23.2642 dB and zero
  truncation. The 6 GiB scope peaks at 881 MiB without memory or GPU faults.
- Meganeura `fc20c16` packs 2–32-column RMSNorm rows into otherwise idle
  workgroup lanes while retaining the previous power-of-two reduction order.
  A 513×3 CPU/GPU forward-and-gradient oracle covers the partial final group,
  and the wide path remains unchanged. Three 200K-site profiles improve median
  GPU time 19.89→17.54 ms (-11.8%) at the same 284 passes. The matched Room
  replay reduces training 111.590→104.632 seconds (-6.2%), GPU wait
  66.601→61.230 seconds (-8.1%), and complete time 116.142→108.892 seconds
  (-6.2%), while preserving 25.4103/23.2724 dB versus 25.4110/23.2709 dB and
  zero truncation. The paired 6 GiB scope peaks at 885 MiB without swap,
  pressure, OOM, kill, or GPU fault.
- Oriented training now gathers each path row's normalized normal and offset
  once for the recorded tangent, spatial appearance, and optional normal loss.
  Structural and numerical coverage require the shared values. Three profiles
  reduce the graph 284→283 passes and median GPU time 17.60→17.01 ms (-3.4%).
  The matched Room replay reduces training 104.582→103.209 seconds (-1.3%),
  GPU wait 61.444→60.366 seconds (-1.8%), and complete time
  109.103→107.490 seconds (-1.5%), while preserving 25.4125/23.2703 dB versus
  25.4096/23.2625 dB and zero truncation. The paired 6 GiB scope peaks at
  897 MiB without memory or GPU faults.
- Meganeura `09d0873` computes f32/f16 embedding dispatches from the flattened
  output size instead of rounding each narrow row independently. This removes
  up to 256× excess workgroups without changing the shader or graph. Compiler
  boundary coverage, direct GPU checks, exact resume, and both repositories'
  practical suites pass. Three profiles keep 283 passes while reducing the ten
  embeddings 4.98→1.55 ms and median GPU time 16.67→13.55 ms (-18.7%). The
  matched Room replay reduces training 102.918→90.827 seconds (-11.7%), GPU
  wait 59.915→52.443 seconds (-12.5%), and complete time 107.495→95.070
  seconds (-11.6%), while preserving 25.4099/23.2624 dB versus
  25.4095/23.2647 dB and zero truncation. The paired 6 GiB scope peaks at
  897 MiB without swap, pressure, OOM, kill, or GPU fault.
- Meganeura `41951b1` replaces `sum_inner` backward's multiplication by an
  all-ones matrix with a direct row-gradient broadcast. A 513×16 physical-GPU
  oracle is bit exact through the partial final workgroup, and compiler
  coverage requires the specialized dispatch. Three profiles change 283→284
  passes but reduce median GPU time 13.59→11.52 ms (-15.2%). The matched Room
  replay reduces training 90.989→86.245 seconds (-5.2%), GPU wait
  52.586→48.788 seconds (-7.2%), and complete time 95.520→90.481 seconds
  (-5.3%), while preserving 25.4134/23.2717 dB versus 25.4072/23.2625 dB and
  zero truncation. The paired 6 GiB scope peaks at 887 MiB without swap,
  pressure, OOM, kill, or GPU fault.
- Cloud appearance now stays channel-wise through SH, spatial surface colour,
  and spherical-Voronoi reductions. This preserves every serialized parameter
  and viewer layout while avoiding expanded RGB basis copies and allowing
  Meganeura's embedding/reduction fusion to apply. Three profiles change
  284→298 passes but reduce median GPU time 11.51→9.43 ms (-18.1%). The matched
  Room replay reduces training 86.002→80.580 seconds (-6.3%), GPU wait
  48.733→44.994 seconds (-7.7%), and complete time 90.564→84.783 seconds
  (-6.4%). Fresh-Ply quality remains within 0.011 dB at 25.4121/23.2632 dB
  versus 25.4136/23.2739 dB, with zero truncation. The paired 6 GiB scope peaks
  at 870 MiB without swap, pressure, OOM, kill, or GPU fault.
- Meganeura `d01e58f` fuses each single-use row-gradient broadcast and factor
  multiply directly into the following atomic embedding-table scatter. A
  2,049×16 physical-GPU oracle is bit exact through a partial final workgroup,
  and shader validation plus both repositories' practical suites pass. Three
  order-balanced 200K-site profiles reduce 298→284 passes and median GPU time
  9.88→8.41 ms (-14.9%). The matched Room replay reduces training
  79.988→76.629 seconds (-4.2%), GPU wait 45.158→42.308 seconds (-6.3%), and
  complete time 84.497→80.833 seconds (-4.3%), while preserving
  25.4111/23.2654 dB versus 25.4112/23.2594 dB and zero truncation. The paired
  6 GiB scope peaks at 871 MiB without swap, pressure, OOM, kill, or GPU fault.
- Reusing finite masked path payload after a one-time initialization removes
  36 MiB of redundant dt/Jacobian fills per 4,096-ray, 128-entry training
  step, while still clearing every gather index and mask. The matched
  2,040-step oriented replay is 3.46% faster in training and 3.27% end to end;
  its fresh-Ply train/held-out score moves only +0.0004/-0.0002 dB. A physical
  GPU test injects non-zero padding across positions, radii, normals, and
  offsets and observes exactly zero loss and gradients. The complete serial
  training suite and strict workspace lint pass; peak memory remains 563 MB
  with no swap, pressure, OOM, or GPU fault.
- The reference squared-overlap interpenetration loss is available as a
  deterministic sampled objective. On the 50,000-cell Bonsai gate it trims
  the selected trainable-radius graph by 19.2% and adds 0.11 dB all-37, but
  increases training time by 6.1%. Its scale is coupled to scene units,
  geometry rates, and Adam epsilon, so it remains opt-in rather than a default.
- The locally trained official checkpoint is not a vendored asset. Its geometry
  and full appearance cross-render are complete. Blade reaches 28.3078 dB
  versus the official 28.3432 dB, and quantized renders agree at 59.37 dB over
  all 37 views. The local interchange and generated PLY remain ignored.
- The released per-detail-site eight-axis colour model is now implemented as
  an optional 384-float table nested under the eight spatial sites. It shares
  the existing surface-detail CPU/WGSL path and uses existing Meganeura
  operations; no shader-entry or graph-op variants were added. Three smaller
  directional alternatives remain rejected by their held-out gates. The full
  table also fails its first 255-step Room screen by 0.0521 dB while taking
  6.1× longer; fixed axes lose 0.0579 dB. Its 4,096-ray graph nearly fills the
  5070 (11.64 GiB) and needs 6.19 GiB host memory. It remains an exact import/
  render feature, not a selected training default.

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
- Exact Čech queries are split across eight logarithmic radius bands, avoiding
  the global `r_max` bound for every site. A 200K topology phase falls
  1.877→0.481 seconds with identical CSR; a forced 100K rebuild falls
  0.530→0.085 seconds. The full replay stays below 1.05 GB with zero swap,
  pressure, OOM, or GPU faults.
- Traversal now integrates the terminal cell up to the requested end depth
  when no later face is found.
- Official v1 instead caches neighbor offsets in half precision and does not
  integrate a cell without an exit face. Blade keeps full-precision site data
  and terminal integration as its runtime contract. After matching the
  reference per-cell SH clamp, the official Room PLY cross-render is within
  0.43 dB mean PSNR; the remaining storage/traversal delta is explicit rather
  than silently emulated.
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
- RadFoam and Gaussian SH radiance is clamped to nonnegative per point before
  opacity weighting, matching the reference forward and zero-gradient branch.
  Clamping only the accumulated output had left a systematic 1.05 dB mean gap
  when cross-rendering the official Room checkpoint.
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
  parity against the CPU oracle now passes. The official 1,256,396-Gaussian
  3DGRUT Bonsai checkpoint cross-render is complete: Blade agrees with the
  upstream 3DGRT images at 35.38 dB. The invariant window sweep selects 48
  candidates: at 512² it cuts Bonsai/Room/small submit-plus-wait time from
  8.29/6.87/4.23 ms to 3.91/3.22/3.51 ms with identical output hashes. A
  regression keeps a broad support containing the camera, whose proxy exit is
  beyond camera depth, behind more than one complete candidate batch.
- Scene Gaussian tracing now keeps its hardware query interval separate from
  the Gaussian's semantic support interval. Reusing the finite semantic bounds
  for triangle queries excluded conservative icosahedron proxy faces lying
  outside the ellipsoid support and could produce zero radiance. The local TLAS
  is now queried over the full forward interval, while maximum-response depth
  is filtered against semantic support before compositing.
- Native Gaussian training now exists in `blade-volume-train`: the 3DGUT-style
  exact ray response, host-recorded candidate rows, staged SH/support fitting,
  calibrated multi-light center/normal continuation, and relightable PBR
  attachment all use the existing Meganeura graph operations. A bounded static
  split now exercises gradient accumulation and optimizer-state remapping, but
  a recurring native-Gaussian grow/prune/reset lifecycle remains unfinished.

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

Progress through 2026-07-17: Stages 0 and 1 are substantially complete. The
first versioned Bonsai smoke result now evaluates a freshly serialized PLY at
16.58 dB train / 17.00 dB held out, identical to live evaluation; exact DC SH
extension properties removed the prior 0.90/0.94 dB serialization loss. The
run used a 2 GiB/no-swap cgroup and peaked at 68.6 MiB. A controlled pure-Rust
Delaunay attempt reached its 8 GiB limit before training, while exact Qhull
completed adjacency in 0.02 seconds, so production benchmark protocols select
the isolated Qhull feature explicitly. A post-Meganeura-uprev rerun on commit
`9d224dd` reproduces the same live and fresh-PLY metrics exactly.

Stage 2 now has topology-safe opt-in position optimization, exact symmetric
adjacency caps, terminal-segment integration, reference position-gradient ×
cell-radius RadFoam densification, explicit background compositing, and opt-in
smooth depth-variance regularization. Analytical unweighted position gradients now
match central finite differences on a fixed, smooth cell path. Densification
now collects maximum per-view ray contribution on the GPU at a deterministic
2× downsample, protects contributing cells and their adjacency neighbours,
suppresses dead-cell density, and reports max-step truncation. Held-out CPU
evaluation also distinguishes hard step-cap truncation from opacity, far-plane,
and terminal-cell exits and warns with an exact ray count. Official-checkpoint
renderer parity is complete; a matched-scale Rust trainer benchmark remains.
The trainer now also implements RadFoam's exact
random transmittance-quantile depth-separation loss and half-training weight
ramp; the earlier smooth depth-variance term remains available as a separate
ablation. The color contract now explicitly follows reference RadFoam/3DGS:
training, SH appearance, backgrounds, PNG output, and PSNR use display-referred
sRGB code values, and SH radiance is clamped per cell before compositing. The
viewer no longer applies an extra Reinhard curve to the RadFoam backend.
Lossless checkpoints now include a versioned trainer-state
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
tests pass. Weighted densification samples a 99th-percentile-capped
`T × alpha × L1(cell_color, target)` EMA without per-step host readback, then
copies the parent's radius and optimizer ancestry while applying the reference
5%-of-radius perturbation;
reference-checkpoint and real-scene radius ablations still remain.

The first bundled RadFoam-v1 semantic direction check is also complete. At
step 2,000 its reloaded model trailed the old scaled protocol by 2.25 dB train
and 0.71 dB held out, so the run was stopped. This comparison bundled four
changes while retaining a 256-ray batch, and represented 512,000 rays rather
than the official schedule's 2 billion at the same step. Stage 2 now proceeds
with one-factor-at-a-time scaled ablations before any larger reference-budget
run.

That first one-factor matrix now shows that the reference initialization is a
strong positive change (+2.03 dB held out) and white compositing is a smaller
positive change (+0.45 dB). The v1 schedule is the dominant negative at this
small batch (-4.58 dB); Smooth-L1's +0.28 dB mean is compromised by a 4.44 dB
held-out-frame failure. The next run combines only the two robust positives on
the existing L1/cosine path.

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

The post-ablation gate on commit `549759f` split those jobs into separate 4 GiB
scopes. Default workspace tests peaked at 1,127,636,992 bytes, all-feature
workspace tests at 1,133,342,720 bytes, and the 72-test physical-NVIDIA trainer
gate at 414,027,776 bytes. Formatting and all-target/all-feature clippy also
pass. Every final scope recorded zero swap, `memory.max`, OOM, and group-kill
events; the lower peaks reflect warm build artifacts as well as job isolation.

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

That original 256-ray curve is effectively flat from step 8,000 through
10,000; continuing it unchanged was correctly rejected. Re-evaluation with the
current renderer gives 16.76/15.83 dB train/selected-held-out and 16.16 dB
all-37 at step 10,000. The controlled change is update composition: 4,096 rays
distributed across 16 views on the original 20,400-step cosine horizon. Legacy
parameter groups win the first cross-scene gate, and lossless continuation
produces:

| Global step | Cells | Train PSNR | Selected-8 PSNR | All-37 PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 200,000 | 16.76 dB | 15.83 dB | 16.16 dB |
| 10,500 | 200,000 | 18.38 dB | 16.77 dB | 17.47 dB |
| 12,000 | 200,000 | 19.78 dB | 17.97 dB | 18.61 dB |
| 14,000 | 200,000 | 21.17 dB | 19.07 dB | 19.89 dB |
| 16,000 | 200,000 | 22.13 dB | 19.88 dB | 20.73 dB |
| 18,000 | 200,000 | 22.67 dB | 20.26 dB | 21.19 dB |
| 20,000 | 200,000 | 22.87 dB | 20.39 dB | 21.33 dB |
| 20,400 | 200,000 | 22.89 dB | 20.41 dB | 21.35 dB |

The endpoint PLY and checkpoint PLY are byte-identical with SHA-256
`047aa82cf8e91fc74acaaf9bf94f8497dbf8913d4ef3d2fb0de36c6f40f6b4f3`.
The model has 3,042,928 directed adjacency edges and lives at
`target/audit-runs/bonsai-batch4096-long-cosine-legacy-step20400-local/`.
The final 400 updates add only 0.02 dB all-37, and 256² comparisons still show
large floaters and smeared thin structure, so the result closes this schedule
rather than the reconstruction-quality gate.

Repeating the same mixed-view protocol from initialization demonstrates that
the old trajectory's geometry, not just its unfinished appearance fit, was the
limitation:

| Global step | Cells | Train PSNR | Selected-8 PSNR | All-37 PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 57,365 | 16.42 dB | 15.47 dB | 16.35 dB |
| 4,000 | 98,838 | 19.90 dB | 18.95 dB | 19.63 dB |
| 6,000 | 170,203 | 21.68 dB | 20.14 dB | 21.41 dB |
| 7,000 | 200,000 | 22.44 dB | 20.78 dB | 22.03 dB |
| 8,000 | 200,000 | 22.93 dB | 21.39 dB | 22.43 dB |
| 10,000 | 200,000 | 23.37 dB | 22.01 dB | 22.80 dB |
| 12,000 | 200,000 | 24.01 dB | 22.63 dB | 23.30 dB |
| 14,000 | 200,000 | 24.76 dB | 23.18 dB | 23.88 dB |
| 16,000 | 200,000 | 25.40 dB | 23.65 dB | 24.30 dB |
| 18,000 | 200,000 | 25.85 dB | 23.92 dB | 24.61 dB |
| 20,000 | 200,000 | 26.03 dB | 24.00 dB | 24.69 dB |
| 20,400 | 200,000 | 26.05 dB | 24.00 dB | 24.69 dB |

The final/checkpoint PLY SHA-256 is
`4378223af64a491747637e5a86490e95f176fa11aa88218fbead2d4ea1cda97d`;
the model has 3,055,304 directed edges and lives at
`target/audit-runs/bonsai-fresh-batch4096-top-track-step20400-local/`.
At 256² it reaches 23.95 dB on the selected views. Visual inspection shows
solid recognizable scene structure instead of the old large translucent
floaters, but colour speckle and smeared thin/background detail remain.

The fresh residual diagnostic also changes the next hypothesis. Across the
worst 512 pixels in each selected view, error is distributed over 1,896
dominant cells, primarily dense far/background cells at path depths 13–20,
rather than one corrupt cell or the old near-camera floater population.
Topology changes by 2.34% over steps 18,000→20,000 and 0.46% in the final 400,
so freezing adjacency is not justified.

A scene-specific traversal gate shows that 224 path steps are enough at the
capped endpoint: the candidate matches all reported PSNRs, and its 256² PNGs
are byte-identical when rendered with caps 224 and 256. It reduces GPU-step
time 35.425→31.407 seconds and graph/device allocation by about 12%. The
setting is not generalized while densification can lengthen paths.

Finally, fresh Smooth-L1 geometry is now a positive direction. It is mixed at
step 2,000, but after a matched 400-step no-growth settle reaches
22.00/20.86/21.70 dB versus L1's 21.46/19.88/21.25. Under the ordinary growth
schedule at step 6,000 it reaches 22.11/20.98/21.74 dB, gains of
+0.43/+0.84/+0.33 dB. Its 174,331-cell PLY/checkpoint SHA-256 is
`23eaac7270cc0a6d7cee5272560f2c714113dfd4ede6f76cb877f4a86a0d7037`.

That continuation, previously blocked by a reset-required driver state, is now
complete on driver 595.71.05 and resumed from the intact step-6,000
checkpoint. It leads the L1 control at every measured step:

| Global step | Cells | Train PSNR | Selected-8 PSNR | All-37 PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 6,000 | 174,331 | 22.11 dB | 20.98 dB | 21.74 dB |
| 8,000 | 200,000 | 23.47 dB | 22.30 dB | 22.81 dB |
| 10,000 | 200,000 | 23.83 dB | 22.63 dB | 23.10 dB |
| 12,000 | 200,000 | 24.50 dB | 23.15 dB | 23.56 dB |
| 14,000 | 200,000 | 25.26 dB | 23.64 dB | 24.09 dB |
| 16,000 | 200,000 | 25.96 dB | 24.08 dB | 24.51 dB |
| 18,000 | 200,000 | 26.50 dB | 24.38 dB | 24.82 dB |
| 20,000 | 200,000 | 26.76 dB | 24.48 dB | 24.92 dB |
| 20,400 | 200,000 | 26.78 dB | 24.48 dB | 24.94 dB |

At the exact 20,400-step endpoint this is 26.78/24.48/24.94 dB against the L1
control's 26.05/24.00/24.69 dB: +0.73/+0.48/+0.25 dB at identical capacity and
schedule. The 200,000-cell PLY has 3,041,490 directed edges, SHA-256
`18481e5530c24d077f7612351ef8987dced52f6fe14867711d99980a17d515fa`, and lives
at `target/audit-runs/bonsai-fresh-batch4096-smooth-l1-step20400-local/`. It
reaches 24.46 dB on the selected views at 256². Its 4 GiB scopes peak at
1,121,284,096 bytes with zero swap, pressure, OOM, or kill events.

The advantage is largest in the mid-growth interval (+0.91 dB selected at step
8,000) and narrows as the capped cloud saturates (+0.48 dB at the endpoint),
and the final 400 updates add 0.02 dB all-37 — the same flat tail the L1 arm
showed. So Smooth-L1 improves the trajectory rather than the ceiling: this
schedule, not the loss, is what closes.

The 200,000-cell cap is not that ceiling. A fresh-trajectory capacity gate
branches at step 10,000, grows once to 229,273 cells at step 10,500, and settles
against the existing controls. At step 12,000 it changes train/selected/all-37
by +0.05/-0.02/+0.02 dB; at step 14,000 by +0.07/+0.04/+0.02 dB, while taking
14.4% and 6.7% longer over the two matched 2,000-step segments. More
importantly, depth-mode extraction at step 14,000 changes the relightable
scene's train/test/worst PSNR from 13.38/11.23/4.29 to 13.43/10.91/3.19. The
extra cells marginally fit radiance while making the shared-surface signal
worse. Reopening the cap only after step 20,400 is a stronger failure: even
after 500 settling updates the source field loses 1.28 dB selected and 1.01 dB
all-37. Both branches stay experimental artifacts; 200,000 remains selected.

Profiling originally showed exhaustive all-view contribution scans dominating
densification. The correctness oracle remains exhaustive, but its GPU results
now land in cached download memory before the unchanged CPU scoring pass.
Matched contribution time falls 468.9× and full command time 4.15×, removing
the bottleneck without sampling views or changing pruning semantics. Exact
Qhull rebuilds every 100 position steps are now the main fixed-cap Bonsai cost;
a cadence-250 replay is 1.25× faster but loses 0.05 dB all-37 and is rejected.

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

Items 1-6 are implemented and covered by CPU and physical-GPU tests. The
reference source/configuration audit for item 7 is complete, an official Room
prefix is serialized, and Blade cross-renders it within 0.43 dB. The trainer
now exposes the reference initialization, loss, and parameter schedule, but the
actual same-budget Rust training run remains the gate: the complete-dataset
curves above are internal scaled protocols, not reference-matched results.

Acceptance gate: a canonical scene matches the reference implementation within
0.5-1.0 dB at the same cell budget, split, image scale, and training budget,
with no systematic streaking from stale topology.

### Stage 3: bounded PowerFoam

1. Persist and upload radii whenever Čech adjacency is selected.
2. Intersect each cell interval with the current site's radius sphere.
3. Compute differentiable weighted-face and sphere intersections.
4. Optimize radii together with positions and validate their gradients.
5. Compare traversal against a brute-force bounded-power oracle.
6. Cross-render a reference PowerFoam checkpoint. (Done to 59.37 dB render
   agreement over all 37 held-out views.)
7. Gate the implemented dipole, detail-site, and released per-detail-site
   directional appearance semantics on held-out scenes.

Items 1-6 are implemented at the CPU-oracle, production-WGSL, recorder, and
training-graph levels and pass physical-GPU integration. Weighted densification
uses the reference copied-radius split policy, and the selected real-scene
radius trajectory reaches its 200,000-site endpoint. The dipole-normal subset
of item 7 is implemented across IO, traversal, training, densification, and
resume and passes a positive fixed-versus-learned Bonsai gate. Spatial detail
also passes its held-out gate. The complete released per-detail directional
table passes the official-checkpoint pixel gate, but its zero-initialized short
training screen loses held-out quality. The latest four-view Room continuation
cuts the historical time gap to 2.23×, but uses 3.88× the cgroup memory and
loses 0.9288 dB across eight held views. Compact directional alternatives also
fail their held-out gates. A 32-training-view ratio sweep likewise changes the
eight-view held mean by -1.8018/-0.0644/-0.0114 dB at ratios 1.0/0.1/0.01.
The remaining item 7 work is therefore a new quality/responsibility hypothesis
and memory improvement, not another runtime appearance variant or scalar-rate
sweep.

Acceptance gate: CPU, GPU, and brute-force bounded traversal agree; a reference
checkpoint renders within a defined image tolerance; trained radii improve a
fixed ablation rather than merely changing topology.

### Stage 4: Gaussian backend

1. Establish a trusted raster or reference 3DGRT image oracle.
2. Sweep hit-window and proxy bounds for accuracy and performance. (Done.)
3. Add cloud transforms without rebuilding point data.
4. Decide whether native Gaussian reconstruction is justified after RadFoam and
   PowerFoam quality is established.

The CPU maximum-response oracle, exact batched ordering, and physical-GPU pixel
parity are implemented. Scene traversal applies Gaussian cloud transforms
without rebuilding point data or the local TLAS, and transformed-pixel tests
pass. The relightable surface-Gaussian pipeline additionally has an opt-in
complete-render normal polish for calibrated multi-light captures. Four
synthetic clouds improve unseen-light novel-view mean PSNR by 0.25--0.35 dB
and every worst view by 0.21--0.36 dB, while the exact optimizer/rebuilt-tracer
oracle passes. Preserving exact repeat-light correspondence per camera before
that pass then improves all four truth-normal RMSEs by 0.13--0.37 degrees and
held-light mean/worst PSNR by another 0.06--0.17/0.05--0.11 dB. Coverage moves
by at most 0.3 points. The large remaining 53.7--57.5 degree error keeps dense
single-light correspondence as the geometry bottleneck. The recognized-
checkpoint gate is complete: Blade agrees with the official 3DGRUT Bonsai
3DGRT images at 35.38 dB on the full 37-view split.

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

1. Pin a reference RadFoam revision and reproduce one run on a complete,
   checksum-verified scene. Record reference-side serialized renders and PSNR
   rather than quoting a paper number with a different setup. (Done for the
   exact first 5,000 updates of official v1 on Room: 735,103 cells and 30.0239
   dB; Blade cross-renders the PLY at 29.59 dB.)
2. Compare initialization, optimizer parameter groups, learning-rate curves,
   opacity parameterization, geometry update cadence, pruning decisions,
   densification samples, and topology/path refresh timing step by step. Add a
   small deterministic trace fixture for every discovered semantic difference.
   (Initialization, background, loss, exact schedule, and separated initial
   parameter-group ratios, topology cadence, and cell-count-dependent
   densification are measured at the 256-ray boundary. At batch 1,024 the
   exact schedule remains negative, relative parameter groups become neutral,
   and ray-normalized schedules are a positive direction. A corrected
   two-round same-ray comparison retains a +0.17 dB held-out advantage for
   batch 1,024, which expands to +0.90 dB at round 3; native 3:2 training is
   negative. The matched Room result confirms +0.82 dB held out and improves
   six of eight test frames. Official-checkpoint comparison then exposed and
   fixed the missing per-cell SH clamp; renderer parity now passes. Retraining
   the selected Room protocol under the corrected graph reaches 19.02 / 18.84
   dB, +0.52/+0.61 dB over its historical published result. Distributing the
   same rays across 16 views then reaches 20.21 / 19.94 dB and improves all
   eight selected held-out frames. At the matched 1,024,000-ray boundary the
   gain persists: 21.31 / 20.90 dB versus 20.18 / 19.95 dB, again improving
   all eight frames at nearly identical capacity and cost.)
3. Continue the deterministic 16-view protocol through a sampled-ray and
   resolution ladder. (Done through step 6,000 at 256² on Room: 735,103 cells,
   26.22/24.57 dB train/selected-held-out, and 24.14 dB across all 39 held-out
   views. A 64-view batch, 512² training, and distortion regularization are
   neutral and rejected. A matched 4,096/8,192-ray gate confirms that doubling
   the batch adds only 0.01 dB for 59% more training time, so 4,096 remains
   selected. Factor isolation shows that neither reopening the cosine horizon
   nor relative RadFoam-v1 parameter groups works alone; together they restore
   reference-scale rates and add 0.21 dB all-39 by step 6,000. The final
   500-step segment adds only 0.02 dB all-39 and begins regressing late views,
   Repeating the mixed-view direction from Bonsai initialization, rather than
   attaching it to geometry formed by tiny batches, reaches
   26.05/24.00/24.69 dB train/selected/all-37 at the exact 20,400-step
   endpoint. This is +3.16/+3.59/+3.34 dB over the completed continuation at
   the same capacity. A fresh Smooth-L1 gate then adds +0.43/+0.84/+0.33 dB
   through step 6,000, and its completed continuation reaches
   26.78/24.48/24.94 dB at the same endpoint — +0.73/+0.48/+0.25 dB over the
   L1 control, leading at every measured step. Relative parameter groups,
   extra capacity, and cadence 250 still fail their Bonsai gates. Both scenes
   remain visibly below the quality bar, and the robust-loss trajectory is now
   finished, so the next experiment should target the distributed
   far/background residuals and the 200,000-cell cap rather than extend a
   saturated fixed-cap schedule.)
4. Do not declare Stage 2 complete until the same-budget result is within
   0.5–1.0 dB of the reference or the remaining difference is isolated to a
   documented unsupported feature.

### P1: remove scaling bottlenecks without changing decisions silently

1. Prototype deterministic stratified-view contribution sampling and an
   incremental per-cell accumulator. Keep the exhaustive all-view scan as the
   oracle; require pruning/densification decision agreement and a fixed PSNR
   tolerance before changing the default. (A 32/255-view rotating prototype is
   2.12× faster and within 0.03 dB, but differs by 1,223 retained cells; the
   default therefore remains exhaustive.)
2. Measure exact topology cadences after the cloud reaches its target size.
   (Done on fixed-cap Room across matched 25/125, 125/250, and 250/500 gates.
   Cadence 250 is selected: it ties denser refreshes through step 3,750, while
   cadence 500 loses 0.01 dB selected and all-39 at step 4,500. Retain
   feature-gated Qhull as the production-size oracle while investigating a
   memory-bounded Rust implementation; runtime geometry stays point-cloud-only
   in either case. Commit `1c46fb2` removes redundant global edge sorting from
   exact unbounded CSR construction, making the matched topology phase 1.19×
   faster without changing the graph-selection semantics. On Bonsai,
   cadence 250 makes the matched 1,000-step interval 1.25× faster but loses
   0.05 dB all-37, so cadence 100 remains selected there.)
3. Add per-phase timing around recording, optimization, contribution scans,
   downloads, topology construction, and evaluation so a long run explains its
   cost without profiler-only evidence. (Done in `5e3f81d`. The matched
   step-5,000→5,100 profile found redundant parameter downloads and removed
   them without changing quality or checkpoint contents: training is 1.39×
   faster and whole-command time is 1.29× faster. Checkpoint serialization is
   now the largest measured endpoint phase. Reusing the losslessly equivalent
   endpoint PLY then makes final output 311× faster and brings the combined
   whole-command speedup to 1.45×. Upstream safetensor streaming reduced peak
   memory in isolation but did not reproduce a benefit when combined, so it
   remains unselected pending an exact explanation. Commit `0da6a85` then
   removes unnecessary graph/optimizer reconstruction at fixed-cap topology
   boundaries: state readback is 9.24× faster, training 1.24× faster, and
   whole-command time 1.21× faster with identical held-out quality. Commit
   `86cfcab` then moves exhaustive contribution readback to cached download
   memory: the matched CPU scoring phase is 468.9× faster and the complete
   densification command 4.15× faster at unchanged all-view quality.
   Meganeura `7967ca2` subsequently batches model parameter downloads, and
   `226041c` batches Adam moments: the latter cuts a real 200K-site Bonsai
   detail boundary's state readback 62.8× and whole-command time 3.09× without
   increasing peak memory. Meganeura `760bb29` then replaces Blade's repeated
   scalar concat/split broadcasts, and `048c8be` removes the remaining
   detail-vector concat tree: the detail graph falls from 537 to 461 passes
   and matched Room/Bonsai training becomes a further 5.4%/3.0% faster at
   neutral cross-scene quality. Dense PowerFoam clipping subsequently assigns
   a workgroup per ray only above 32 adjacency entries/site: Bonsai training
   falls another 39.8% (111.409→67.035 seconds) while Room keeps the serial
   kernel within 0.51% of its prior timing, both at neutral held-out quality.
   Meganeura `508c0ab` then removes the two 48 MiB tiled detail-distance
   chains; Room/Bonsai improve another 3.6%/3.7% at neutral held-out quality.
   Meganeura `f31ef08` removes the remaining 48 MiB normal tile and explicit
   tangent-projection chains; Room/Bonsai improve another 6.4%/4.8% at neutral
   held-out quality. Meganeura `22bb539` then replaces the expanded
   sigmoid/reciprocal exponential identity; Room/Bonsai improve another
   1.9%/3.8% at neutral held-out quality. Meganeura `85e919f` then fuses the
   two detail-weight normalization graphs without changing their f32 reduction
   order; Room/Bonsai improve another 2.4%/2.4% at neutral held-out quality.)
4. Reuse the production GPU tracer for exhaustive checkpoint evaluation while
   preserving the CPU implementation as the default oracle. (Done for
   unweighted RadFoam in `77c19b7`: physical pixel parity passes, aggregate
   Room PSNR is identical, and the 255+8-view pass is 4.58× faster. Weighted
   PowerFoam now uses its separate compute-splat evaluator; treating the Čech
   graph as a global traversal graph was invalid. Device-resident photometric
   resampling then removes 26.6 seconds of per-step gradient readback: matched
   2,000-step training is 14.6% faster and all-37 quality changes from 13.51 to
   13.52 dB. Conservative 16×16 projected candidates subsequently make the
   100,569-site all-view pass 4.4% faster with identical pixels and an exact
   overflow fallback. Packing weighted geometry/Jacobians reduces the matched
   2,000-step training peak from 1.287 GB to 0.857 GB and time from 326.71 to
   321.50 seconds while held-out quality stays within 0.02 dB. Exact path-cap
   telemetry then proves 128 steps sufficient for the 100,569-site gate and
   cuts the same segment to 297.85 seconds and 0.575 GB, with 14.35 dB held-out
   quality and zero truncation. Caching each candidate's radical-plane
   interval removes repeated adjacency clipping while retaining byte-identical
   held-out renders. The same segment falls again to 123.03 seconds, crosses
   all four resource rebuilds, and retains 14.34 dB held-out quality with zero
   truncation or memory events. Parallel exact Čech-row queries then reduce
   topology time by 77.7% and the segment to 96.25 seconds; held-out quality is
   14.36 dB, all 8.2 million rays remain untruncated, and host peak stays below
   0.9 GB. Finally, sharing the gather and record compute passes across
   disjoint mixed-view slices removes redundant per-camera barriers without
   changing ray or Jacobian semantics. The segment falls to 67.07 seconds
   (71.13 seconds end to end), keeps all four densification counts exact,
   retains 14.34 dB held-out quality, and peaks at 652 MB with no memory or GPU
   fault events. At the 200K endpoint, replaying the final 400 updates with 128
   rather than 160 path entries cuts graph/device allocation 16.6%/16.4% and
   training time 4.6%. Training uses at most 123 entries and the full 256²
   evaluation at most 127, with zero truncation and identical rounded PSNR;
   32/37 comparison PNGs are byte-identical and the rest differ by one channel
   count in one pixel.)
5. Correct optimizer learning-rate selection for the bare SH DC parameter
   names. (Done in `2a0fd73`. The old code targeted nonexistent `_0` names,
   leaving DC at multiplier 1.0 while higher-order SH rates worked. A matched
   fresh 2,000-step Bonsai PowerFoam gate improves legacy train/all-37 PSNR
   from 13.70/13.52 to 14.24/14.04 dB and absolute RadFoam-v1 from
   10.18/10.15 to 13.54/13.45 dB. Earlier parameter-group artifacts retain
   their measured quality but do not validate their labelled DC schedules and
   must be rerun before changing the selected schedule.)

### P1: validate PowerFoam and Gaussian semantics on real assets

1. On the winning RadFoam configuration, compare fixed initialized radii against
   trainable positive radii from identical seeds. Require a held-out improvement
   and stable cell/topology statistics before implementing the full quaternion,
   detail-site, and per-detail-directional appearance model. (The earlier
   50,000-cell ablation is
   superseded: its camera-seeded weighted walk missed disconnected Čech
   components and sat near the black baseline. The corrected compute-splat
   trainer reaches 11.67 dB after 10 steps, 12.55 after 100, and 13.52 after
   2,000 on all 37 Bonsai held-out views, versus 9.33 after 2,000 broken-walk
   steps. A corrected same-seed ablation now freezes or trains the identical
   reference-initialized radii while retaining position optimization. At step
   2,000, radius learning improves train/held-out PSNR from 13.41/13.22 to
   13.70/13.51 dB. At step 4,000, after four topology-changing growth rounds,
   it improves 14.32/14.05 to 14.66/14.36 dB at the identical 100,569-site
   capacity. Both arms record zero truncation, swap, OOM, or GPU faults; the
   learned arm remains selected. Those causal numbers predate the SH-DC group
   fix; the intended schedule reaches 175,895 sites and 15.72/15.28 dB at step
   6,000. Its 647 candidate hits versus 127 surviving path intervals prompted
   independent candidate/path capacities; 128-step output stays byte-identical
   to the 256-step control and makes matched training 16.4% faster. Continue
   to the 200,000-cell boundary only after checking that topology and radius
   growth remain bounded. The corrected selected endpoint applies weak overlap
   during growth, uses 160-step paths and cadence-200 exact rebuilds, and reaches
   17.4151/16.5247 dB with 8,328,306 edges at step 20,000. The 256² comparisons
   still show large support discs, holes, blurred thin structure, and background
   floaters, so the production gate remains open. Revisit cloud support and
   spatial appearance semantics rather than adding more identical steps. The
   first oriented-dipole gate is now complete: learned normals add 0.5382 dB
   over fixed PCA planes at 2,040 loss-free steps; 8,160 steps come within
   0.020 dB of the unoriented held-out endpoint, and 16,320 come within 0.001
   dB. A per-site signed surface offset then raises the 8,160-step endpoint to
   16.4583 dB held out (+0.0678 over normal-only and +0.0478 over the original
   unoriented cloud); the 16,320-step ceiling is 16.4773 dB and adds only
   0.0190 dB for 2× time. The independent Room gate raises normal-only held-out
   quality from 23.0112 to 23.2402 dB at the safer 0.005 offset ratio and
   improves 37/39 frames. Keep both fields explicit because this is still one
   plane per site. The first minimal spatial-color arm now passes the two-scene
   mean gate at ratio 0.02: +0.0176 dB all-37 on Bonsai and +0.0149 dB all-39
   on Room, for +4.8% and +14.6% training time at their measured graph sizes.
   Keep it opt-in while checking the regressed tail views. The subsequent
   compact eight-site detail arm passes its Room gate at site/height/colour
   ratios `0.1/0.05/0.05`, adding 0.1513 dB held out and improving 37/39
   views. It also passes a matched Bonsai gate at +0.0701 dB held out and
   improves 33/37 views, but remains opt-in because training and PLY size grow
   34.5% and 49.8% respectively. Joint support-entry gradients subsequently
   unlock position/radius learning and add +0.884/+0.405 dB held out on
   Room/Bonsai at the exploratory high radius rate, but raise training time by
   57%/68% and produce mixed Bonsai tail views. Sharing the three unique
   ray-relative geometry gathers for the five linearization consumers then
   cuts Room training by 1.8% at unchanged quality. Meganeura then folds those
   shared gathers and offsets directly into every compatible dot reduction,
   cutting another 2.0% on Room and 2.4% on Bonsai at neutral quality. Sharing
   the invariant geometry between the base and displaced detail-plane queries
   then cuts another 0.5% on Room and 0.3% on Bonsai at neutral quality. The
   long radius-rate gate now selects `0.005` for Room and `0.01` for Bonsai;
   neither becomes a universal default, and the Bonsai selection retains a
   heavy degree tail. Contribution diagnostics then reject a minimum-view
   pruning shortcut, and two bounded directional-detail variants fail fresh
   long controls. The resulting eight-logit spatial-density probe is retained
   as opt-in: two order-balanced 2,040-step replicas improve held-out Room by
   0.0162 dB and Bonsai by 0.0443 dB, while adding 6.10 MiB per 200K sites and
   11.0%/12.4% training time. Averaged per-view splits are 21/1/17 and
   25/0/12, with a -0.52 dB Room outlier, so the default remains unchanged.
   Shared runtime evaluation keeps all-view inference within 0.5% of matched
   controls. A compact
   48-float additive Spherical Voronoi residual is now implemented but rejected:
   learned and fixed axes lose 0.0168 and 0.0137 dB held out on Room while
   adding about 22% training time. On Room, a 160-entry path budget is exact
   on all 294 views; the full
   learned spatial-colour replay cuts training by 19.2% and peak host memory by
   19.2% versus 256 while slightly improving held-out PSNR. Retain the larger
   candidate floor and select path budgets from telemetry. A 128-row replay is
   exact but loses 0.0158 dB through a changed optimizer reduction shape, so
   160 remains selected. Packing eight-wide softmax rows into the existing
   reduction template is 8.5–9.1% faster but loses 0.0236 dB held out on
   Bonsai because the changed f32 tree changes the optimizer trajectory; no
   Meganeura prototype is retained. Fixed-size point recycling also fails:
   replacing 5,326/87 low-contribution Room/Bonsai sites at one exact
   all-view boundary changes held-out PSNR by -0.0499/-0.0004 dB, and a
   tenfold stricter threshold loses -0.0658/-0.0108 dB. Both policies are
   removed. Current contribution and point-error scores are not a safe proxy
   for held-view spatial responsibility. Conversely, the error-probe graph is
   now omitted once no future densification round can consume it. Two
   order-balanced fixed-capacity replicas reduce mean training time by 3.5%
   on both Room and Bonsai and GPU wait by 4.7%/5.5%, with neutral-to-positive
   mean train and held-out PSNR. This is a scheduler-only graph-lifetime change:
   it adds no shader, graph op, or model/runtime variant. The official
   appearance audit at commit `9639225` counts 412 extra floats per point and
   identifies spatial-norm and directional-kernel differences between paper
   and source, so each staged increment must name its exact contract.)
2. Improve the released PowerFoam directional table's training efficiency and
   quality. (The checkpoint and
   geometry gate now exist: an official-implementation Bonsai checkpoint at
   step 10,000 has 162,373 full-appearance sites and scores 28.3432 dB over all
   37 quarter-resolution held-out views. Blade's exact Rust Čech builder agrees
   with 336,933/336,935 fresh official undirected edges, has no extra edge, and
   builds the graph in 69 ms. The two official-only edges come from its f32
   AABB radius round-trip and do not satisfy the predicate on the original
   values. Serialized-vs-fresh official topology differs by 2,709 edge
   decisions because checkpoint parameters and adjacency straddle an optimizer
   step, but native metrics are unchanged to four decimals after rebuild. Full
   replacing the 384-float per-detail directional colour table with each
   detail site's mean colour costs 11.8969 dB in the official rasterizer, while
   the matched mean-colour official and Blade renders agree at 59.18 dB. The
   complete table is now nested under spatial detail and shares its existing
   shader path: the full official and Blade renders agree at 59.37 dB. A
   zero-initialized short training screen is six times slower and slightly
   worse held out. Fixed axes now skip their gradient and Adam state through a
   zero-cost stop-gradient alias: the matched Room run is 26.1% faster at a
   2,048-ray batch, and a stable 4,096-ray run scores 24.4125 dB held out. A
   matched Bonsai screen is neutral at 17.4160→17.4162 dB but remains 6.4×
   slower than its no-directional control. Hardware profiling then isolates
   the cost: Adam alone grows from 1.2 to 65.3 ms, and the path recorder is not
   responsible. Omitting directional-weight gradients only when every geometry
   source is frozen cuts another 11.1%/12.1% on Room/Bonsai at unchanged
   held-out quality; the Bonsai gap is then 5.7×. The Adam profile subsequently
   identifies an allocator bug: both large private moments live in host-visible
   memory. Keeping those moments device-local, with staged state/checkpoint
   transfers, cuts the 200K-site Adam pass from 65.3--66.2 to 16.2--16.9 ms and
   the complete profiled graph to 126.6--127.5 ms. Eight held-out Room views are
   unchanged at 25.2813 dB, and this runtime-only change adds no graph op,
   shader edit, shader entry/group, or backend variant. Separating dedicated
   gradient lifetime from host visibility then keeps parameter gradients
   device-local while staging the diagnostic and CPU fallback APIs. The
   38.4M-element colour-gradient add falls from 9.51 to 0.76 ms, Adam falls
   again from 16.93 to 3.01 ms, and the steady graph reaches 87.72 ms (-30.7%)
   at the same reported per-view quality. Full shared/device, clipping,
   accumulation, CPU-SGD, checkpoint, and all-target gates pass. Constant
   parameter gradients remain shared because the session initializes constants
   from the host; regressions cover this rare dual-role buffer and the
   downstream oriented-surface graph that exposed it. Blade now also omits
   gradients and Adam state for every optional surface field whose rate is
   zero, and wholly frozen oriented geometry no longer records a surface
   tangent. A matched 98,831-site directional-colour continuation falls from
   354 to 262 graph passes and from 75.03 to 68.88 ms (-8.2%) at unchanged
   21.2093/20.4618 dB train/next-view quality. This remains a generic
   stop-gradient cleanup with no new operation or shader variant. Meganeura
   `c409bb2` then fuses each table gradient's eight-wide broadcast and multiply
   into the existing atomic scatter. The matched 32-update Room graph falls
   from 262 to 256 passes and from 34.48--34.85 to 30.83--31.33 ms (about
   -10.4%), while both arms score 24.2420/21.1172 dB on the four training/next
   held view. This adds no operation, shader entry/group, binding, pipeline, or
   backend variant. Blade then factors the eight repeated direction rows into
   Meganeura's existing pairwise-distance operation using the equivalent
   `|T d - a|² = T² |d - a/T|²` form. Stacked on `c409bb2`, the graph falls
   from 256 to 253 passes and from 30.83--31.33 to 26.34--26.82 ms (about
   -14.5%). A matched 256-update pair falls from 17.873 to 15.943 seconds and
   scores identical 29.1014/18.9553 dB over four training/eight held views.
   CPU/GPU forward and learned-axis finite-difference gates pass. This also
   adds no operation or backend/shader variant. Frozen axes are then packed
   once as unit vectors and temperatures, removing their repeated GPU
   normalization. The graph falls from 253 to 251 passes and from
   26.40--26.78 to 22.55--22.94 ms; the matched 256-update pipeline falls from
   15.943 to 13.393 seconds and from 5.54 to 3.56 GB. A no-table control still
   takes only 6.014 seconds and 0.919 GB, and scores 27.1422/19.8841 dB versus
   the table's 29.1014/18.9553: the table overfits these four views and loses
   0.9288 dB across eight held views. Pin `c409bb2` after merge; retain the
   full table as opt-in until a scene-matched weighted second-scene checkpoint
   is available for the longer gate. A same-budget 32-view screen does not
   rescue training: ratios 1.0/0.1/0.01 change held PSNR by
   -1.8018/-0.0644/-0.0114 dB relative to no table. Do not spend another gate
   on the scalar rate without a new regularization or responsibility model.
   Returning to the production no-table graph found its next generic win in
   candidate selection: the shared sphere BVH is faster below 4,096 rays per
   camera, while projected candidates catch up at that boundary. Raising the
   internal crossover from 1,024 to 4,096 cuts a matched 256-update Room run
   from 6.014 to 5.035 seconds and peak cgroup memory from 0.919 to 0.523 GB,
   while preserving 27.1422/19.8841 dB over four training/eight held views.
   Prepacking frozen spatial-detail tangent sites then removes another two
   graph passes and improves three matched 256-update runs by 1.7%, with the
   same reported quality and memory. Detecting that the production
   continuation's frozen spatial-detail residual is exactly the identity then
   keeps its PLY/checkpoint fields but omits their unused graph: 195→153 passes,
   a 30.7% reduction in three-run mean training time, and 0.523→0.293 GB peak
   memory, again at identical reported train/held quality. Making the path
   recorder's existing surface-query output genuinely optional then reduces
   its warmed time from 3.77--4.44 to 2.83--3.28 ms. The three-run training
   mean falls another 11.6% to 3.111 seconds, still at identical reported
   train/held quality. Finally, the default-zero per-view exposure rate now
   freezes those three parameters before differentiation instead of building
   gradients and Adam state that a zero multiplier later discards. The graph
   falls from 153 to 147 passes and from 2.76--2.84 to 2.43--2.48 ms; three
   matched 256-update runs average 3.049 seconds (-2.0%) at the same
   27.1422/19.8841 dB train/held quality. Traversal is now the larger remaining
   production cost. The forward-only weighted recorder then reuses each
   candidate's already-computed effective interval instead of repeating its
   sphere and oriented-plane clip merely to emit `dt`. The warmed record pass
   falls about 13%, total recorder time about 10%, and three matched 256-update
   runs average 2.970 seconds (-2.6%) at the same reported quality. The
   existing scratch carries the far endpoint; differential and detail-query
   paths are unchanged, and no entry/group, binding, buffer, pipeline,
   operation, or dependency is added. Finally, the core integrator drops its
   obsolete sigmoid/reciprocal exponential identity in favor of Meganeura's
   existing differentiable `exp`. The graph falls from 147 to 138 passes and
   from 2.42--2.48 to 2.33--2.37 ms; three matched 256-update runs average
   2.946 seconds (-0.8%) at the same reported train/held quality. A
   two-replica 2,040-update gate confirms -0.8% mean training time and -1.4%
   mean GPU wait, with both arms exactly 29.3164/17.8472 dB at reported
   precision. Finally, a measured crossover sweep lowers the existing
   workgroup-per-ray PowerFoam clipping gate from 32 to four adjacency
   entries/site. At the production Room cloud's 15.9 entries/site the settled
   record pass falls from 2.00--2.12 to 1.75--1.80 ms, and three matched
   256-update runs average 2.789 seconds (-5.3%) with the same reported
   train/held quality. The serial path remains faster at 0.9 entries/site, so
   both existing paths are retained. The clipper now also stops scanning a
   site's remaining neighbors as soon as its monotonically shrinking near/far
   interval becomes empty. Two 2,040-update runs average 17.358 seconds versus
   17.568 seconds (-1.2%) at the same reported loss; all 19 physical-GPU path
   oracles pass. The balanced hierarchy's exact post-frontier stack bound then
   reduces shared scratch from 32 to 25 words per lane. BVH gather falls about
   14%, and two matched 2,040-update runs average 17.067 seconds (-1.7%) with
   1.2% less GPU wait. The dense clipper then replaces lane zero's serial
   64-item compaction loop with the workgroup atomic already used by candidate
   gathering. Its existing depth/cell sort makes the temporary compact order
   unobservable. Six matched settled samples reduce the record pass from 1.91
   to 1.69 ms (-11.8%); three 256-update replicas reduce mean pipeline time
   from 2.879 to 2.796 seconds (-2.9%) and mean wall time from 3.136 to 3.040
   seconds (-3.1%). Every replica remains exactly 27.1422/19.8841 dB at
   reported train/held precision, all 19 path and eight image oracles pass,
   and host peak stays below 294 MB. These changes add no shader entry/group,
   pipeline, operation, or dependency.)
3. Cross-render a recognized Gaussian checkpoint against official 3DGRUT.
   (Done: the complete 1,256,396-Gaussian Bonsai checkpoint agrees with
   upstream 3DGRT at 35.38 dB over 37 held views. The 48-candidate ray-query
   window preserves output hashes while the conservative triangle BLAS remains
   an invisible point-candidate accelerator, not polygonal scene geometry.)

### P2: finish engine-level composition

1. Define exact interleaved optical-depth composition for overlapping clouds,
   then test mixed RadFoam/PowerFoam/Gaussian scenes against standalone
   segment oracles. Whole-cloud layer ordering is deterministic but is not the
   final physical model.
2. Implement a Metal-compatible per-cloud binding strategy and run the same
   transformed-pixel suite on Metal. Accelerate the exhaustive software-TLAS
   cursor only after pixel equivalence is locked down.
3. PowerFoam compute splats are now required for correctness in training and
   headless evaluation. Conservative projected-tile candidates have exact
   path/PSNR parity and a tested exhaustive overflow fallback. The interactive
   viewer now selects that same tracer whenever `radii.is_some()`, while plain
   RadFoam retains the global adjacency walk. Disconnected weighted components
   therefore no longer disappear only in the viewer. Full-image splats reuse
   4,096 compact rows instead of allocating path and candidate scratch in
   proportion to window area; the cloud projection and tile binning still run
   only once per frame. A physical-GPU fixture forces four image batches and
   is byte-identical to the original full-row renderer. Another gate covers
   semantic backend selection, max-step and threshold changes, resize, and a
   post-resize render. The existing full-row constructor and its complete
   candidate/path telemetry remain unchanged for training and headless
   evaluation.

   On the current 98,831-site Room checkpoint with spatial detail and a
   192-step exact row, the bounded viewer measures 2.00 ms at 320x180, 6.19 ms
   at 640x360, and 23.27 ms at 1280x720 over settled frames on the RTX 5070.
   The 8 GiB host cgroup peaks at 153 MB with zero swap, pressure, OOM, or GPU
   fault. Resize and live max-step changes rebuild only resolution-dependent
   scratch; no model clone, renderer/shader variant, graph operation,
   dependency, or alternate point-cloud representation is added. Defer the
   unrelated SDF and Gaussian-compute backends until the Stage 2 quality gate
   is resolved. Generated benchmark code and output were removed; telemetry
   remains outside version control under `/tmp/blade-volume-gpu-3776137.log`.
   The default and all-feature workspace test matrices both pass serially in a
   12 GiB cgroup; the larger run peaked at 6.13 GB without swap or OOM events.

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
