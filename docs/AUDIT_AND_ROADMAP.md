# Audit and implementation roadmap

Initial audit: 2026-07-12

Last updated: 2026-07-24

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
far less optimized: its 16-view, 10,880,000-ray step-5,000 checkpoint matches
the 735,103-cell capacity and reaches 25.69 / 24.03 dB on the selected
train/held-out split and 23.93 dB across all 39 held-out views at 256². The
official prefix has processed roughly five billion rays and reaches 30.02 dB.
The Rust path therefore remains a useful scaling baseline, not a
production-ready reconstruction pipeline.

The corrected paths pass the targeted NVIDIA/Vulkan physical-GPU gates,
including weighted differentiable traversal, Gaussian CPU/GPU parity, and
transformed-scene pixel readback. Whole-cloud layering is still not exact
interleaved volume compositing and the Metal multi-cloud binding design is
open. New rendering methods and broad scene features should therefore remain
behind these gates:

1. Match reference RadFoam training within the Stage 2 quality tolerance on a
   complete, reproducible scene. Checkpoint renderer parity is done.
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
- Densification uses position-gradient × cell-radius signal, contribution-aware
  pruning, optimizer ancestry, and the weighted copied-radius split policy.
- Quantile regularization, explicit background compositing, lossless DC SH, and
  versioned parameter/Adam/RNG checkpoints are implemented.
- The blocker is trainer-scale quality. A fresh, segmented run over all 292
  registered Bonsai images reached step 10,000 and the 200,000-cell target on a
  stable NVIDIA/Vulkan path. Its held-out curve flattened near 16.1 dB by step
  8,000, so the run was deliberately stopped before the nominal 20,400-step
  budget rather than spending more compute on an unchanged protocol.
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
- Opt-in `radfoam-v1` initialization and learning-rate policies plus a
  beta-1 Smooth-L1 color loss are implemented without changing historical
  defaults. A versioned scaled semantic-ablation manifest records both the
  controls and the remaining batching/topology/resolution differences.
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
- Meganeura branch `perf/stream-checkpoints-fba` (`cb64f67`) removes the second
  checkpoint-sized host allocation by streaming safetensors to disk. A matched
  run cuts peak host memory by 695,529,472 bytes (15.7%) with the same selected
  24.03 dB result and no memory or GPU faults. Checkpoint time is unchanged
  within run variance (100.060 versus 98.460 seconds), so this is a pending
  upstream memory improvement rather than a speed claim. Blade remains pinned
  to merged Meganeura `fba040a` until it lands.
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
- Smooth-L1's 4.44 dB held-out failure is unchanged at twice the traversal cap
  and under either background. Visual comparisons identify large near-camera
  Voronoi floaters; the loss remains opt-in until that geometry interaction is
  understood.

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
cell-radius densification, explicit background compositing, and opt-in smooth
depth-variance regularization. Analytical unweighted position gradients now
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
tests pass. Weighted densification copies the parent's radius and optimizer
ancestry while applying the reference 5%-of-radius perturbation;
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
   resolution ladder. (Done through step 5,000 at 256²: 735,103 cells,
   10.88 million rays, 25.69/24.03 dB train/held out, and 23.93 dB across all
   39 held-out views. A 64-view batch is neutral and rejected. The next
   quality experiment should change a remaining reference-protocol variable
   or repeat the selected policy on another scene, not merely extend the same
   nearly exhausted cosine schedule.)
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
   in either case.)
3. Add per-phase timing around recording, optimization, contribution scans,
   downloads, topology construction, and evaluation so a long run explains its
   cost without profiler-only evidence. (Done in `5e3f81d`. The matched
   step-5,000→5,100 profile found redundant parameter downloads and removed
   them without changing quality or checkpoint contents: training is 1.39×
   faster and whole-command time is 1.29× faster. Checkpoint serialization is
   now the largest measured endpoint phase; upstream streaming reduces peak
   memory by 15.7% but does not improve its wall time.)
4. Reuse the production GPU tracer for exhaustive checkpoint evaluation while
   preserving the CPU implementation as the default oracle. (Done in
   `77c19b7`: weighted/unweighted physical pixel parity passes, aggregate Room
   PSNR is identical, and the 255+8-view pass is 4.58× faster.)

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
