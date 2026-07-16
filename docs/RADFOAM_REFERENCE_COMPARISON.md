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

## Next experiment

Run a scaled semantic ablation before committing to a multi-day exact-budget
attempt:

1. Use the same 50,000→200,000 cell cap and held-out split as the plateau run.
2. Change only initialization, Smooth-L1, white compositing, and the v1
   parameter schedule; retain the exhaustive contribution oracle.
3. Increase the pixel batch as memory permits and record the exact cumulative
   ray count, rather than comparing optimizer-step counts alone.
4. Evaluate serialized checkpoints at steps 2,000, 5,000, 8,000, and 10,000.
5. If the held-out curve improves, add reference dynamic topology and
   cell-count-dependent densification cadence, then repeat from the same seed.
6. Only after the scaled direction is positive should the run move toward
   190,951→2,097,152 cells and the staged 780×520→1559×1039 image schedule.

The acceptance criterion remains a same-budget reference run within 0.5–1.0 dB;
the scaled ablation is a direction check, not a substitute for that gate.
