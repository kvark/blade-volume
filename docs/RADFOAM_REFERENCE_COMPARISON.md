# RadFoam v1 reference comparison

Date: 2026-07-16

This comparison pins the official [`theialab/radfoam` v1
tag](https://github.com/theialab/radfoam/tree/366e1a1b4349023b18e7867fabd6b734983f5c3c)
at commit `366e1a1b4349023b18e7867fabd6b734983f5c3c`. The current upstream head
inspected during the audit was
`3e7b52cf74e37ab2ab5e695f53570f515f537e3d`; its relevant COLMAP training
semantics are unchanged except that opacity supervision can use dataset alpha.
COLMAP scenes have an all-ones alpha target, so v1 is the appropriate published
baseline.

## Main conclusion

The 17.50 dB train / 16.13 dB held-out blade-volume result is not a failed
reproduction of RadFoam. It is a much smaller plumbing and scaling experiment.
At the step-10,000 stopping point it had processed 2.56 million sampled rays;
official v1 processes one million mixed-view rays per optimizer step, or about
20 billion rays over its run. The Rust run also used one tenth as many final
cells, a square low-resolution image, a different loss/background, different
initialization, and a shared optimizer schedule. Its plateau says that this
particular low-budget protocol is exhausted, not that point-cloud volumetric
geometry is exhausted.

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
versioned result is
[`bonsai-radfoam-v1-semantics-step2000-86239aa.toml`](../benchmarks/results/bonsai-radfoam-v1-semantics-step2000-86239aa.toml).

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
pressure, OOM, or GPU faults. The versioned result is
[`bonsai-radfoam-v1-relative-groups-step2000-a91766b.toml`](../benchmarks/results/bonsai-radfoam-v1-relative-groups-step2000-a91766b.toml).

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
reference-scale protocol. The versioned result is
[`bonsai-radfoam-v1-topology-cadence-step2000-e50e965.toml`](../benchmarks/results/bonsai-radfoam-v1-topology-cadence-step2000-e50e965.toml).

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
pressure, OOM, or GPU faults. The dedicated protocol and versioned result are
[`bonsai_densification_cadence.toml`](../benchmarks/bonsai_densification_cadence.toml)
and
[`bonsai-radfoam-v1-densification-cadence-step2805-11a7118.toml`](../benchmarks/results/bonsai-radfoam-v1-densification-cadence-step2805-11a7118.toml).
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
confirmation without changing the current default. The protocol and result
are
[`bonsai_batch1024_optimizer.toml`](../benchmarks/bonsai_batch1024_optimizer.toml)
and
[`bonsai-batch1024-optimizer-step500-6877bea.toml`](../benchmarks/results/bonsai-batch1024-optimizer-step500-6877bea.toml).

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
lower resolution rather than a crop. The protocol and result are
[`bonsai_batch1024_native_aspect.toml`](../benchmarks/bonsai_batch1024_native_aspect.toml)
and
[`bonsai-batch1024-native-aspect-step625-07ad939.toml`](../benchmarks/results/bonsai-batch1024-native-aspect-step625-07ad939.toml).

## Next experiments

1. Test larger stratified caps and cumulative multi-boundary drift against the
   exhaustive oracle; do not change the default without decision agreement or
   a deliberately revised acceptance gate.
2. Compare ray-normalized batches 256 and 1,024 at 640,000 or more sampled rays
   under the corrected operation order, then continue only if the larger batch
   retains a held-out advantage beyond the second growth boundary. Native 3:2
   training is rejected for the scaled gate.
3. Move toward 190,951→2,097,152 cells and the staged
   780×520→1559×1039 image schedule only after that direction survives a larger
   batch.

The acceptance criterion remains a same-budget reference run within 0.5–1.0 dB;
the scaled ablation is a direction check, not a substitute for that gate.
