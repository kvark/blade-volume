# RadFoam v1 reference comparison

Date: 2026-07-18

This comparison pins the official [`theialab/radfoam` v1
tag](https://github.com/theialab/radfoam/tree/366e1a1b4349023b18e7867fabd6b734983f5c3c)
at commit `366e1a1b4349023b18e7867fabd6b734983f5c3c`. The current upstream head
inspected during the audit was
`3e7b52cf74e37ab2ab5e695f53570f515f537e3d`; its relevant COLMAP training
semantics are unchanged except that opacity supervision can use dataset alpha.
COLMAP scenes have an all-ones alpha target, so v1 is the appropriate published
baseline.

## Main conclusion

The representation is worth pursuing. An exact official-v1 Room run reaches
30.02 dB on all 39 held-out views after the first 5,000 of 20,000 optimizer
updates, with 735,103 cells. This is direct evidence that a cloud-only
Voronoi-volume representation can reconstruct a difficult real scene well;
the earlier Rust plateau does not reject the geometry premise.

The current Rust trainer is still a much smaller optimization experiment. Its
selected Room protocol processes 2,944,000 sampled optimizer rays and now
matches the official prefix's 735,103 cells, reaching 22.77 dB on its selected
eight held-out views. The serialized official prefix has processed five billion
mixed-view rays. Resolution, split coverage, loss, background, initialization,
and optimizer schedules also differ. These scores are therefore diagnostics,
not an apples-to-apples trainer ranking.

Renderer compatibility is now separately established. Loading the official
PLY directly in Blade and evaluating the same full-resolution split reaches
29.59 dB, 0.43 dB below the upstream renderer. Before correcting Blade's
missing per-cell nonnegative SH clamp, the same model reached 28.97 dB. The
remaining Stage 2 risk is consequently training scale and protocol efficiency,
not basic PLY, camera, SH, or volumetric-renderer compatibility.

## Protocol differences

| Area | Official RadFoam v1 Bonsai | Plateau run (`93c996f`) | Consequence |
| --- | --- | --- | --- |
| Train/test split | Filename-sorted, every eighth held out | COLMAP order, every eighth held out | The checked Bonsai `images.bin` order is already filename-sorted, so the actual split agrees. |
| Resolution | Downsample 4 through step 4,999, then downsample 2 | Fixed 128×128 | Reference trains substantially more samples and preserves the 3:2 aspect ratio. |
| Batch | 1,000,000 rays shuffled across all training views | 256 rays from one randomly selected view | Reference gradients have far lower variance and mix cameras within every update. |
| Steps | 20,000 | 20,400 nominal; stopped at 10,000 | Step counts look similar but effective ray budgets do not. |
| Sparse initialization | `floor(0.9 × 206,613)` samples with replacement plus 5,000 broad normal sites | 50,000 highest-track-length COLMAP sites | Reference starts at 190,951 sites and explicitly covers empty/background space. |
| Initial appearance | All SH coefficients zero (neutral gray) | COLMAP RGB in SH DC | Different starting problem; sparse color can help early loss but is not reference behavior. |
| Initial raw density | Uniform `[0,1)` for sparse samples, `-0.5` for broad sites | Uniform activated density 1 | Different early opacity and gradient distribution. |
| Target cells | 2,097,152 | 200,000 | The internal run has roughly one tenth the spatial capacity. |
| RGB loss | Smooth-L1, beta 1, mean over rays and channels | L1, per-channel means summed | Changes both robustness and the relative scale of opacity/quantile terms. |
| Background | White | Black | White is part of the official opaque COLMAP recipe. |
| Opacity loss | Weight 1 against opacity 1 | Weight 1 against opacity 1 | Formula agrees, but old RGB normalization made its relative strength differ. |
| Quantile loss | Weight `1e-4`, half-run linear ramp | Same | Implemented semantics agree. |
| Density activation | Softplus beta 10 | Softplus beta 10 | Implemented semantics agree. |
| Adam epsilon | `1e-15` | `1e-8` | Usually secondary, but material for an exact reproduction. |
| Learning rates | Independent density, position, DC, and higher-SH schedules | One cosine base with static multipliers | Position and SH rates differed by up to orders of magnitude during the run. |
| Topology refresh | Period 1, then 3, 5, …, 99, 101; reset after densification | Exact rebuild every 100 steps | The v1 source increments 99 to 101 before stabilizing; reference refreshes rapidly while early geometry moves fastest. |
| Densification cadence | Cell-count-dependent linear-growth interval, minimum 100 | Fixed 500 by default; exact policy available as an opt-in | The scaled baseline reaches its cap, but not at the same times or with the same contribution windows. |
| Contribution scan | Every view at a random 2× phase; max ray contribution | Same correctness-oriented exhaustive strategy | Semantics agree; this is the measured runtime bottleneck at large cell counts. |

Official v1 parameter rates are:

| Parameter | Initial | Final | Warmup/freeze |
| --- | ---: | ---: | --- |
| Density | `1e-1` | `1e-2` | Linear warmup through 10% of training, then cosine |
| Position | `2e-4` | `5e-6` | Cosine, zero after 90% |
| SH DC | `5e-3` | `5e-4` | Cosine |
| Higher SH | `5e-4` | `5e-5` | Linear warmup through 20%, then cosine |

The upstream loop updates its scheduler after each optimizer step. Therefore
step zero uses each parameter group's declared rate, step one uses scheduler
index zero, and the density/higher-SH rates are zero for that second update.
The Rust `radfoam-v1` schedule deliberately preserves this one-step quirk.

## Official Room execution and renderer parity

The reference run uses the official Mip-NeRF-360 Room images and sparse COLMAP
model. Checksums of the local `images_2` tier and all three sparse binary files
were compared with the official archive; the `images_4` training tier was
extracted from that archive. Official v1 holds out every eighth
filename-sorted image, trains at downsample 4 through update 4,999, switches to
downsample 2 at update 5,000, and evaluates all 39 held-out images at
1,557×1,038.

The first exact 20,000-update attempt reached update 4,999, approximately
735,351 cells, and a periodic 30.16 dB held-out result. At update 5,000 the
upstream loader replaced its cached downsample-4 rays with all downsample-2
rays. Host memory rose from about 9 GiB to the 32 GiB cgroup limit between
one-second samples and the kernel killed the isolated scope. The process had
also reached 7,297 MiB of the 12 GiB GPU. No model was serialized because
upstream saves only after training. This is a bounded resource failure, not a
quality result, and retrying the unchanged loader on this machine is not
responsible.

A second run retained the original 20,000-update schedules but used a stop-only
hook immediately after update 4,999, before the high-resolution reload. It
serialized the exact first-5,000-update prefix:

| Measurement | Official v1 prefix |
| --- | ---: |
| Optimizer rays | 5,000,000,000 |
| Cells | 735,103 |
| Directed adjacency entries | 11,185,482 |
| Held-out views / resolution | 39 / 1,557×1,038 |
| Upstream held-out PSNR | 30.0239 dB |
| Wall time | 14m 14.978s |
| Host-memory peak | 10,280,112,128 bytes |
| Sampled GPU-memory peak | 7,133 MiB |

The 12 GiB run completed with zero swap, memory-pressure, OOM, or GPU-fault
events. Repeating the first prefix was not bit deterministic: the failed full
attempt and serialized prefix differed by 248 cells and about 0.13 dB at the
boundary despite the upstream seeds. Reference comparisons should therefore
retain run-to-run variance rather than imply exact CUDA determinism.

The serialized PLY is directly compatible with `PointCloudModel`. A
full-resolution CPU cross-render over the identical 39 views gives:

| Blade rendering of official PLY | Mean PSNR | Delta vs upstream |
| --- | ---: | ---: |
| Before per-cell SH clamp | 28.97 dB | -1.05 dB |
| Corrected renderer (`00de721`) | 29.59 dB | -0.43 dB |

The corrected run takes 5m 6s, peaks below 1 GiB RSS, and has no truncated-ray,
swap, pressure, OOM, or fault report. The remaining per-view delta ranges from
-0.07 to -1.50 dB and is systematic across all views. Remaining implementation
differences include upstream's half-precision cached neighbor offsets and its
boundary/terminal-cell behavior; Blade evaluates faces from full-precision
sites. Their individual contributions have not been isolated, so the residual
is documented rather than attributed to one mechanism or hidden by changing
Blade's runtime contract.

The build ran from the pinned v1 commit in an isolated Ubuntu 24.04 root with
Python 3.12, PyTorch 2.9.1 CUDA 13.0 wheels, the CUDA 13.1 toolkit, GCC 13, and
the RTX 5070 on driver 595.71.05. Local reference-only compatibility edits
allowed the toolkit/PyTorch minor mismatch and replaced CUDA-removed CUB
iterators with their Thrust equivalents. They are not dependencies of the Rust
project. The machine-readable record is
[`room_radfoam_v1_reference.toml`](../benchmarks/room_radfoam_v1_reference.toml).

## Implemented comparison controls

The Rust trainer now exposes the following opt-in controls while preserving its
existing defaults:

- `--initialization radfoam-v1` reproduces the sparse-with-replacement,
  perturbed-site, broad-background, neutral-appearance, and raw-density
  distributions. A fixed Rust PRNG seed makes it reproducible, but its bit
  sequence is not expected to equal PyTorch's.
- `--color-loss smooth-l1` implements the official beta-1 piecewise loss and
  averages across RGB channels.
- `--lr-schedule radfoam-v1` applies the official absolute parameter schedules,
  position freeze, higher-SH warmup, and Adam epsilon. It enables position
  training even when the legacy position ratio is zero.
- `--lr-groups radfoam-v1-relative` applies the official initial ratios between
  density, position, DC, and higher-SH rates while retaining the selected
  constant/cosine time curve and Adam epsilon. This separates parameter-group
  scaling from the update-indexed schedule that fails at a 256-ray batch.
- `--geometry-rebuild-schedule radfoam-v1` reproduces the reference counter
  order: periods 1, 3, 5, ... 99, then 101; densification resets only the
  period, so the next optimizer step rebuilds again. Rust continues to use a
  full exact rebuild rather than the reference's incremental implementation,
  isolating cadence from triangulation-backend behavior. The topology phase is
  stored in the current v3 trainer-state sidecar for exact segmented resumes.
  The reader remains compatible with the original v2 topology state. When a
  scheduled refresh and densification coincide, current adjacency/GPU geometry
  is rebuilt before contribution collection, matching the reference operation
  order. Contribution phases are keyed by absolute densification round rather
  than topology rebuild count so cadence comparisons do not alter their
  pruning sample.
- `--densify-schedule radfoam-v1` reproduces the reference counter order,
  post-growth cell-count interval formula, 100-step floor, and 90%-of-target
  stop gate. `--densify-until` is the linear-growth horizon in this mode, not
  a hard cutoff. The original point count, active counter, next interval, and
  absolute round live in the v3 trainer sidecar; v1/v2 fixed-cadence
  checkpoints remain readable. A physical-GPU segmented test crosses two
  dynamic rounds and requires exact equality with uninterrupted parameters and
  adjacency.

These controls do not pretend to solve the remaining batch-size or update
cadence differences. The historical L1/top-track/unified-cosine behavior stays
available for regression and ablation.

## Scaled semantic-ablation result

The first bundled direction check retained the old 50,000→200,000 cell cap,
128×128 resolution, split, 256-ray batch, and topology cadence, while enabling
the v1 initialization, Smooth-L1 loss, white background, and parameter-specific
schedule together. It was stopped at the planned step-2,000 decision point:

| Metric | Old scaled protocol | Bundled v1 semantics | Delta |
| --- | ---: | ---: | ---: |
| Reloaded train PSNR | 13.08 dB | 10.83 dB | -2.25 dB |
| Reloaded held-out PSNR | 13.08 dB | 12.37 dB | -0.71 dB |
| Cells | 57,313 | 56,218 | -1,095 |

The serialized model reproduced the live metrics exactly. Training peaked at
624,214,016 bytes and evaluation at 593,547,264 bytes in separate 4 GiB,
zero-swap cgroups, with no OOM, memory-pressure, or GPU-fault marker. The
protocol is
[`bonsai_radfoam_v1_semantics.toml`](../benchmarks/bonsai_radfoam_v1_semantics.toml).

This is a negative bundled ablation, not a judgment on the v1 semantics. The
run had consumed only 512,000 rays at the decision point; 2,000 official
updates consume 2 billion. In particular, the neutral initialization and
schedule warmups were compared at step parity but not ray or gradient-estimate
parity. Continuing the unchanged bundle would not identify which factor caused
the early deficit.

The subsequent one-factor matrix isolates the main effects at the same scaled
step-2,000 boundary:

| Single change from old scaled protocol | Train PSNR | Held-out PSNR | Held-out delta |
| --- | ---: | ---: | ---: |
| None (baseline) | 13.08 dB | 13.08 dB | -- |
| White background | 13.64 dB | 13.53 dB | +0.45 dB |
| Smooth-L1 beta 1 | 13.40 dB | 13.36 dB | +0.28 dB |
| v1 initialization, scaled to 50K | 14.59 dB | 15.11 dB | +2.03 dB |
| v1 parameter schedule | 8.23 dB | 8.50 dB | -4.58 dB |

All metrics reproduce after a fresh PLY reload. Every run completed in a 4 GiB
zero-swap cgroup with a 382‑391 MB training peak, a 92 MB evaluation peak, zero
memory-pressure/OOM counters, and no GPU-fault marker. The initialization gain
is consistent across all eight held-out frames. Smooth-L1's mean gain is not
yet sufficient evidence to adopt it: `DSCF5613.JPG` fell to 4.44 dB while the
other held-out frames improved. The severe schedule loss explains the bundled
regression and demonstrates step-to-ray-budget coupling at batch 256.

Combining v1 initialization with white was not additive. The reloaded result
reached 14.83 dB train / 14.74 dB held out: 0.24 dB higher train but 0.37 dB
lower held-out PSNR than initialization alone. All held-out frames remained
healthy, so this is a generalization tradeoff rather than another collapsed
view. The selected scaled trajectory is therefore v1 initialization with the
existing black/L1/cosine path.

The selected trajectory was resumed losslessly through step 4,000:

| Step | Cells | Train PSNR | Held-out PSNR | Held-out delta vs old curve | Segment wall time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 57,348 | 14.59 dB | 15.11 dB | +2.03 dB | 646 s |
| 3,000 | 75,445 | 15.95 dB | 15.39 dB | +1.25 dB | 793 s |
| 4,000 | 99,262 | 16.35 dB | 15.58 dB | +0.35 dB | 1,021 s |

The initialization advantage is real but converges toward the old scaled curve
as training and densification proceed. The final segment adds only 0.19 dB
held out while exhaustive contribution/topology work raises its wall time to
17 minutes. It peaked at 622,166,016 bytes in a 4 GiB cgroup with zero swap,
pressure, OOM, or GPU faults. The diagnostic curve stops here; the next dollar
of compute is better spent on batch/schedule and oracle-scaling experiments.

The Smooth-L1 outlier was re-evaluated at 512 rather than 256 traversal steps
and on both black and white backgrounds; `DSCF5613.JPG` remained exactly 4.44
dB. Side-by-side renders show a few huge near-camera Voronoi floaters, ruling
out step truncation and background compositing. Smooth-L1 therefore remains an
opt-in reference control until its geometry/floater interaction is resolved.

An opt-in deterministic stratified contribution cap now addresses oracle cost
without weakening the default. From the identical lossless step-4K state, 32
of 255 rotating views cut the next 500-step segment from 635 to 299 seconds
(2.12×). Fresh-PLY train/held-out PSNR changed only -0.03/-0.03 dB, but the
sample retained 112,667 rather than 113,890 cells and its topology hash
differed. That is promising quality neutrality, not decision agreement; the
exhaustive collector remains the quality default.

The batch-appropriate parameter-group isolation is also negative. Starting
from the same v1 initialization with black/L1/cosine training, official initial
rate ratios reached 14.72 dB train / 14.49 dB held out at step 2,000, versus
14.59 / 15.11 dB for the legacy relative groups. Cell counts were nearly equal
(57,410 versus 57,348), and fresh-Ply evaluation exactly reproduced both new
metrics. The +0.13 dB training change paired with -0.62 dB held out is evidence
of worse generalization, not under-training. The option remains available for
larger-batch experiments, but the legacy groups remain the scaled winner. The
run took 645 seconds and peaked at 615,665,664 bytes in a 4 GiB cgroup; its
separate evaluation peaked at 599,007,232 bytes. Both recorded zero swap,
pressure, OOM, or GPU faults. The result is summarized in the
[benchmark ledger](../benchmarks/README.md).

The clean topology-cadence isolation is likewise neutral-to-negative at this
scale. Both runs use the selected initialization/black/L1/cosine configuration,
the corrected pre-densification operation order, and the identical exhaustive
densification-round-0 contribution sample. Dynamic cadence performs 44 rather
than 20 scheduled updates before step 2,000 and reaches 14.58 dB train / 15.05
dB held out, versus 14.60 / 15.13 dB for fixed-100. It adds 52 cells and 1,046
adjacency entries while increasing wall time from 652 to 676 seconds (+3.7%).
Fresh-Ply evaluation reproduces all four scores exactly. Both 4 GiB scopes
record zero swap, pressure, OOM, or GPU faults; their raw memory peaks are not
directly comparable because only the fixed run compiled the release binary.
This rejects dynamic cadence as the 256-ray default, not as part of a future
reference-scale protocol. The result is summarized in the
[benchmark ledger](../benchmarks/README.md).

The independently isolated densification cadence is also negative at the
256-ray scale. Fixed-500 and `radfoam-v1` use identical data, initialization,
718,080 sampled training rays, fixed-100 topology cadence, and two exhaustive
contribution rounds. Their first step-2,000 growth decision is identical. The
reference formula then schedules its second growth at step 2,517 rather than
2,500; both arms finish at step 2,805 with 66,054 cells. Dynamic cadence
reaches 16.11 dB train / 16.18 dB held out, versus 16.34 / 16.25 dB for fixed
cadence. It takes 1,087 rather than 1,101 seconds, a 1.2% difference that does
not compensate for the quality loss. Fresh-Ply evaluation reproduces the
scores exactly. Both 4 GiB scopes peak near 481 MB and record zero swap,
pressure, OOM, or GPU faults. The dedicated protocol is
[`bonsai_densification_cadence.toml`](../benchmarks/bonsai_densification_cadence.toml)
with metrics summarized in the [benchmark ledger](../benchmarks/README.md).
This retains the dynamic policy as a reference-scale control while selecting
fixed cadence for the current scaled protocol.

The first larger-batch gate separates update-indexed reference controls from
a same-ray schedule. At batch 1,024 and 512,000 sampled training rays, keeping
the original 20,400-update horizon and fixed-100 topology cadence gives these
fresh-Ply results:

| Optimizer control | Train PSNR | Held-out PSNR | Delta vs cosine/legacy |
| --- | ---: | ---: | ---: |
| Cosine, legacy groups | 15.11 dB | 14.89 dB | — |
| Exact v1 schedules | 9.97 dB | 10.53 dB | -5.14 / -4.36 dB |
| Cosine, v1-relative groups | 14.94 dB | 14.95 dB | -0.17 / +0.06 dB |

The exact schedule remains decisively rejected. Relative groups change from a
0.62 dB held-out loss at batch 256 to a neutral 0.06 dB gain, which is worth
retaining but too small and early to select. A fourth arm scales all relevant
update-indexed controls by four: 5,100 total cosine updates, exact topology
every 25 steps, a 125-step post-warmup densification cadence, and a 2,750-step
growth horizon. It performs the same 512,000 training rays and 20 topology
refreshes as the corrected batch-256 fixed-cadence checkpoint. The result is
15.26 dB train / 15.33 dB held out, +0.66/+0.20 dB, with 57,485 cells. All
four arms report zero truncated rays, use fresh serialized PLY evaluation,
peak between 409 and 420 MB in 4 GiB zero-swap scopes, and record no pressure,
OOM, or GPU fault. This selects ray-normalized larger batching for a longer
confirmation without changing the current default. The protocol is
[`bonsai_batch1024_optimizer.toml`](../benchmarks/bonsai_batch1024_optimizer.toml)
with metrics summarized in the [benchmark ledger](../benchmarks/README.md).

The two-round/native-aspect gate does not justify increasing image area. The
losslessly resumed 128×128 arm grows again at step 625 to 66,078 cells and
reaches 15.55 dB train / 15.29 dB held out at its training resolution. That is
+0.29/-0.04 dB from step 500: more capacity raises fit but has not yet improved
generalization. A from-scratch 192×128 arm uses identical optimizer rays,
update schedule, topology cadence, and growth steps. On a common fresh-Ply
192×128 evaluation the square-trained model reaches 15.56 / 15.28 dB, while
native-aspect training reaches 15.42 / 15.05 dB (-0.14/-0.23 dB). The latter
uses 1,566,720 rather than 1,044,480 contribution rays per round and takes
1,224 seconds versus 1,026 seconds summed across the square scopes, even though
the square total includes an extra intermediate evaluation. Final clouds are
nearly equal at 66,086 versus 66,078 cells; all four contribution rounds have
zero truncation. Training peaks are 519 and 451 MB, and both zero-swap scopes
record no pressure, OOM, or GPU fault. Because rectification maps the full
calibrated source domain to either output grid and ray directions use normalized
pixel coordinates with the original camera field of view, square training is
lower resolution rather than a crop. The protocol is
[`bonsai_batch1024_native_aspect.toml`](../benchmarks/bonsai_batch1024_native_aspect.toml)
with metrics summarized in the [benchmark ledger](../benchmarks/README.md).

The corrected same-ray batch comparison confirms that the larger-batch signal
survives the second boundary. Both arms use 640,000 optimizer rays, 25 exact
topology refreshes, two fixed-cadence growth rounds, and the same 128×128
contribution/evaluation grid. Batch 256 reaches 65,801 cells and 15.32 dB
train / 15.12 dB held out; ray-normalized batch 1,024 reaches 66,078 cells and
15.55 / 15.29 dB, a +0.23/+0.17 dB gain. Their matched final continuation
scopes take 391 and 384 seconds and peak at 424 and 451 MB. Both have zero
truncation, swap, pressure, OOM, or GPU faults. The larger batch prunes 26
rather than 138 cells in round 1, so its gradients also change the learned
contribution distribution. Per-frame deltas remain mixed, however, and the
0.17 dB mean is not enough to change defaults without a third boundary or a
second scene. The protocol is
[`bonsai_batch_same_ray.toml`](../benchmarks/bonsai_batch_same_ray.toml) and
the metrics are summarized in the [benchmark ledger](../benchmarks/README.md).

Round 3 turns the modest signal into a clear selection. At 768,000 optimizer
rays, 30 exact topology refreshes, and three exhaustive growth rounds, batch
256 reaches 75,452 cells and 15.41 dB train / 14.76 dB held out. Ray-normalized
batch 1,024 reaches 75,969 cells and 16.08 / 15.66 dB, a +0.67/+0.90 dB gain
that improves seven of eight held-out frames. The matched final continuations
take 445.0 and 443.9 seconds; peak memory is 475 and 489 MB. Mean/max paths are
46.8/150 and 46.7/151 with zero truncation, so traversal does not explain the
gap. Batch 256 prunes 191 cells while batch 1,024 prunes only 18, extending the
round-2 evidence that lower-variance gradients preserve useful contributors.
Both 4 GiB zero-swap scopes record no pressure, OOM, or GPU fault, and fresh
PLY evaluation reproduces every metric. This passes the Bonsai gate and
selects batch 1,024 for the scaled trainer. The CLI now uses 1,024 when
`--pixel-batch` is omitted while preserving every explicit override. The
protocol is
[`bonsai_batch_same_ray_round3.toml`](../benchmarks/bonsai_batch_same_ray_round3.toml)
with metrics summarized in the [benchmark ledger](../benchmarks/README.md).

Room confirms both the batch decision and the corrected radiance semantics.
The original second-scene gate and its controlled from-scratch retrain use the
same 768,000 optimizer rays, 30 topology refreshes, three exhaustive growth
rounds, 128×128 grid, black background, and L1/cosine path:

| Room batch-1,024 result | Cells | Train PSNR | Held-out PSNR | Wall time | Host peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical semantics (`9d224dd`) | 75,718 | 18.50 dB | 18.23 dB | 1,484.545 s | 556,830,720 B |
| Historical PLY, corrected renderer | 75,718 | 17.79 dB | 17.37 dB | — | — |
| Corrected retrain (`00de721`) | 75,809 | 19.02 dB | 18.84 dB | 1,440.142 s | 543,166,464 B |
| Corrected, 16 views/batch (`3d2ba74`) | 75,722 | 20.21 dB | 19.94 dB | 1,463.566 s | 560,238,592 B |
| One view/batch, step 1,000 | 100,158 | 20.18 dB | 19.95 dB | 2,484.232 s cumulative | 670,986,240 B |
| 16 views/batch, step 1,000 | 99,970 | 21.31 dB | 20.90 dB | 2,517.075 s cumulative | 650,641,408 B |
| 16 views/batch, step 1,500 | 174,410 | 22.39 dB | 21.83 dB | 5,400.440 s cumulative | 1,038,524,416 B |
| 16 views/batch, step 1,625 | 200,000 | 22.63 dB | 22.00 dB | 6,364.861 s cumulative | 1,060,331,520 B |
| 16 views/batch, step 2,000 | 303,891 | 23.16 dB | 22.31 dB | 9,719.508 s cumulative | 1,730,691,072 B |
| 16 views/batch, step 2,250 | 400,000 | 23.42 dB | 22.43 dB | 12,582.961 s cumulative | 2,309,640,192 B |
| 16 views/batch, step 2,375, max-512 | 459,888 | 23.55 dB | 22.53 dB | 14,994.186 s cumulative | 2,324,807,680 B |
| 16 views/batch, step 2,500, max-512 | 528,732 | 23.69 dB | 22.57 dB | 17,696.292 s cumulative | 2,646,769,664 B |
| 16 views/batch, step 2,625, max-512 | 607,908 | 23.80 dB | 22.64 dB | 20,717.337 s cumulative | 3,013,816,320 B |
| 16 views/batch, step 2,750, max-512 | 698,940 | 23.89 dB | 22.62 dB | 24,128.491 s cumulative | 3,572,768,768 B |
| 16 views/batch, step 2,875, max-512 | 735,103 | 24.07 dB | 22.77 dB | 27,944.446 s cumulative | 3,634,937,856 B |

Fresh-Ply evaluation exactly reproduces the corrected live result, which is
+0.52/+0.61 dB over the historical published train/held-out metric and
+1.23/+1.47 dB over rendering the old cloud with the corrected semantics. The
new cloud has 1,132,150 directed adjacency entries. All three contribution
rounds examine 1,044,480 rays across all 255 training views and report zero
truncation; the run also records zero swap, pressure, OOM, or GPU faults and a
551 MiB sampled GPU-memory peak.

The mixed-view arm changes only the camera distribution inside each 1,024-ray
Adam batch: 16 deterministic stratified views receive 64 rays each. It
improves the corrected baseline by +1.19/+1.10 dB train/held out, and every
selected test frame improves by +0.25 to +1.62 dB. Capacity is effectively
unchanged, wall time rises 1.6%, host memory rises 3.1%, and sampled GPU memory
remains 551 MiB. Exact 4+4-step checkpoint-resume coverage and sliced
CPU/GPU path-record oracles cover the new stochastic and buffer semantics.

The matched step-1,000 continuations add 256,000 rays and two growth rounds to
each arm. Mixed-view training retains +1.13/+0.95 dB train/held out, improves
all eight selected frames by +0.63 to +1.23 dB, and raises the all-39 coverage
diagnostic from 19.53 to 20.70 dB. Capacity differs by only 188 cells (0.19%);
continuation time differs by 0.9%, and the mixed run has the lower host-memory
peak. Both peak at 567 MiB sampled GPU memory with zero truncation, swap, OOM,
or GPU faults. Notably, the 750-step mixed checkpoint already matches the
one-view 1,000-step held-out score with 25% fewer optimizer rays and about 24%
fewer cells. This later boundary selects 16 views as `train_colmap`'s automatic
random-pixel policy. Full-image and patch modes, plus the direct library
default, remain one view.

Continuing the selected arm to step 1,500 adds four exhaustive growth rounds
and reaches 174,410 cells. Fresh-Ply quality improves by +1.08/+0.93 dB over
step 1,000, and the all-39 coverage diagnostic rises by +0.90 dB to 21.60.
Mean contribution paths grow from 69.4 to 88.2 segments and maxima from 155 to
178, still with zero truncation across every 1,044,480-ray scan. The 500-step
continuation takes 2,883.365 seconds, peaks at 1,038,524,416 host bytes and
719 MiB sampled GPU memory, and records zero swap, pressure, OOM, or GPU faults.
The monotonic metric gain supports further scaling, although comparison PNGs
remain visibly fragmented.

The next 125-step segment reaches the configured 200,000-cell cap. Its one
exhaustive round prunes 67 cells and adds 25,657 splits, with mean/max path
lengths of 94.3/188 and zero truncation. Fresh-Ply quality rises by another
+0.24/+0.17 dB to 22.63/22.00; all 39 held-out views average 21.80 dB, +0.20
dB. The continuation takes 964.421 seconds, peaks at 1,060,331,520 host bytes
and 688 MiB sampled GPU memory, and records zero swap, pressure, OOM, or GPU
faults. Comparison PNGs remain visibly fragmented, so the next ladder segment
must raise the cell cap rather than treating this boundary as convergence.

Raising only the cap to 400,000 and continuing for 375 steps reaches 303,891
cells. The three exhaustive rounds grow mean path length from 100.6 to 113.5
segments, peak at 223, and truncate no rays. Fresh-Ply quality rises by
+0.53/+0.31 dB to 23.16/22.31; all 39 held-out views average 22.22 dB, +0.42
dB. The continuation takes 3,354.647 seconds, peaks at 1,730,691,072 host bytes
and 936 MiB sampled GPU memory, and records zero swap, pressure, OOM, or GPU
faults. Scaling remains productive, but fine detail is still visibly
fragmented.

Continuing to step 2,250 reaches the 400,000-cell cap exactly. The two
exhaustive rounds average 120.2/127.0 path segments, peak at 235/246, and
truncate no rays. Fresh-Ply quality rises by +0.26/+0.12 dB to 23.42/22.43;
all 39 held-out views average 22.38 dB, +0.16 dB. A controlled reload with a
512-step rather than 256-step traversal budget is metric-identical, showing
that the current artifact is not clipped, but the ten-step remaining headroom
is insufficient for another capacity increase. The continuation takes
2,863.453 seconds, peaks at 2,309,640,192 host bytes and 954 MiB sampled GPU
memory, and records zero swap, pressure, OOM, or GPU faults.

The next guarded round raises the target to the 735,103-cell reference prefix,
extends growth through step 2,875, and doubles the traversal budget to 512. At
step 2,375 it reaches 459,888 cells. Its exhaustive scan measures 133.6 mean /
259 maximum segments with zero truncation, directly proving that 256 is no
longer a sufficient contribution budget. A diagnostic 256-step reload still
rounds to the same 23.55/22.53 dB as the fresh max-512 evaluation because the
over-budget paths are rare; all 39 held-out views average 22.47 dB. The
continuation takes 2,411.225 seconds, peaks at 2,324,807,680 host bytes and
1,362 MiB sampled GPU memory, and records zero swap, pressure, OOM, or GPU
faults.

At step 2,500 the max-512 ladder reaches 528,732 cells. The exhaustive scan
measures 140.9 mean / 267 maximum segments with zero truncation, prunes 121
cells, and adds 68,965 splits. Fresh-Ply quality rises by +0.14/+0.04 dB to
23.69/22.57; all 39 held-out views average 22.56 dB, +0.09 dB. The continuation
takes 2,702.106 seconds, peaks at 2,646,769,664 host bytes and 1,690 MiB sampled
GPU memory, and records zero swap, pressure, OOM, or GPU faults. Quality is
still improving, but held-out gains are now small.

At step 2,625 the max-512 ladder reaches 607,908 cells. The exhaustive scan
measures 148.5 mean / 275 maximum segments with zero truncation, prunes 116
cells, and adds 79,292 splits. Fresh-Ply quality rises by +0.11/+0.07 dB to
23.80/22.64; all 39 held-out views average 22.62 dB, +0.06 dB. The continuation
takes 3,021.045 seconds, peaks at 3,013,816,320 host bytes and 1,786 MiB sampled
GPU memory, and records zero swap, pressure, OOM, or GPU faults.

At step 2,750 the max-512 ladder reaches 698,940 cells, within 5% of the
reference prefix capacity. Its exhaustive scan measures 156.2 mean / 299
maximum segments with zero truncation. Fresh-Ply train quality rises by 0.09 dB
to 23.89, selected held-out slips by 0.02 dB to 22.62 because DSCF4707 develops
a larger foreground smear, and all 39 held-out views still rise by 0.03 dB to
22.65. The continuation takes 3,411.154 seconds, peaks at 3,572,768,768 host
bytes and 1,773 MiB sampled GPU memory, and records zero swap, pressure, OOM,
or GPU faults.

At step 2,875 the ladder reaches the official prefix's 735,103-cell capacity
exactly. The exhaustive scan measures 164.2 mean / 305 maximum segments with
zero truncation, prunes 192 cells, and adds the 36,355 splits needed to hit the
cap. Fresh-Ply quality rises by +0.18/+0.15 dB to 24.07/22.77; all 39 held-out
views average 22.76 dB, +0.11 dB. The continuation takes 3,815.955 seconds,
peaks at 3,634,937,856 host bytes and 1,822 MiB sampled GPU memory, and records
zero swap, pressure, OOM, or GPU faults. Cell capacity is therefore no longer
the primary unmatched variable: the Rust checkpoint has only 2.944 million
optimizer rays versus roughly five billion in the 30.02 dB official prefix,
and still differs in resolution and training protocol.

The first fixed-cap continuation toward step 3,000 did not produce a result.
It completed geometry cycles through step 2,975, then NVIDIA Xid 79 reported
that the GPU fell off the bus during the final 25-step interval. The cgroup
fault watcher terminated the scope with exit 143 before a checkpoint or PLY
was written. Host peak was 2,692,427,776 bytes with zero swap, pressure, or OOM
events; the final GPU sample was 1,788 MiB at 72 °C and 100% utilization. This
is a hardware/driver reset event rather than an experiment-quality result. The
validated step-2,875 checkpoint remains the resume boundary after reboot.

At the 750-step boundary, the all-39-view coverage diagnostic improves from
18.44 to 19.66 dB (+1.22), including large recovery near the previously weak
capture tail. It is still not an official comparison: this bounded protocol
caps training at 255 views and selected validation at the first eight held-out
views. Mixed-view PNGs have more coherent room structure but remain visibly
fragmented, while the official 30 dB prefix is visually close to its targets.
This is a stronger scaling baseline, not a usable reconstruction. The
machine-readable result remains in
[`room_batch_same_ray_round3.toml`](../benchmarks/room_batch_same_ray_round3.toml).

## Next experiments

1. Test larger stratified caps and cumulative multi-boundary drift against the
   exhaustive oracle before considering a value above the selected 16 views.
2. Continue the selected 16-view protocol through a bounded sampled-ray and
   resolution ladder from the 735,103-cell checkpoint on Room, retaining
   fresh-Ply metrics, per-phase timing, truncation, and cgroup telemetry at
   every boundary. After reboot, retry the failed step-2,875→3,000 fixed-cap
   segment. Keep capacity fixed until optimizer-budget returns are measured;
   do not jump to the 2.1M-cell final target.
3. Repeat the selected automatic random-pixel policy on another complete scene
   before generalizing the efficiency claim beyond Room. Keep the one-view
   library default and the full-image/patch compatibility behavior.
4. If a complete upstream baseline is still needed, make its image/ray loader
   streaming or run it on a machine with more than 32 GiB host memory and more
   than 12 GiB VRAM. Do not retry the unchanged caching path here.

The renderer-parity acceptance criterion is met at a 0.43 dB mean gap. Trainer
parity remains open until a ray-, cell-, split-, and resolution-matched run is
within 0.5–1.0 dB; the official prefix and scaled ablations bound the problem
but do not substitute for that gate.
