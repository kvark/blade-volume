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
| Topology refresh | Period 1, then 3, 5, …, 99, 100; reset after densification | Exact rebuild every 100 steps | Reference refreshes rapidly while early geometry moves fastest. |
| Densification cadence | Cell-count-dependent linear-growth interval, minimum 100 | Fixed 500 steps | The scaled run reaches its cap, but not at the same times or with the same contribution windows. |
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

## Next experiments

1. Combine the two independently supported changes—v1 initialization and white
   background—while retaining L1 and the existing cosine optimizer.
2. If that combination remains positive, extend its serialized curve beyond
   step 2,000 before spending on a larger multi-view batch.
3. Diagnose the Smooth-L1 single-view collapse, then retry it on the winning
   initialization/background only if the failure is understood.
4. Compare a batch-aware schedule by cumulative rays as well as updates; do not
   use the exact v1 step schedule for batch 256.
5. Add reference dynamic topology and cell-count-dependent densification only
   after the scaled appearance/initialization direction is positive.
6. Move toward 190,951→2,097,152 cells and the staged
   780×520→1559×1039 image schedule only after that direction survives a larger
   batch.

The acceptance criterion remains a same-budget reference run within 0.5–1.0 dB;
the scaled ablation is a direction check, not a substitute for that gate.
