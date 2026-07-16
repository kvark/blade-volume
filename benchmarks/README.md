# Reproducible reconstruction benchmarks

These manifests define the protocol before a run starts. A result is valid only
when it records the manifest name, Git commit, GPU/driver, wall time, final cell
count, train PSNR, and held-out PSNR. Metrics must come from a freshly serialized
PLY loaded by `eval_psnr`, not the live training object.

The current trainer uses fixed internal RNG seeds (`0xDEAD_BEEF_F00D_CAFE` for
pixel/view sampling and `0xCAFE_F00D_DEAD_BEEF` for densification), so identical
manifests and commits have deterministic sampling decisions.

Run the smoke protocol with:

```text
etc/cgroup_run.sh --mem 2G -- \
  cargo run --release -p blade-volume-train --features qhull --bin train_colmap -- \
  --sparse etc/data/bonsai/sparse/0 \
  --images etc/data/bonsai/images \
  --output /tmp/blade-volume-bonsai-smoke.ply \
  --width 32 --height 32 --views 8 --test-views 2 --test-every 8 \
  --max-points 2000 --max-steps 128 --pixel-batch 256 \
  --steps-per-view 25 --learning-rate 0.1 --sh-degree 0 --qhull

etc/cgroup_run.sh --mem 1G -- \
  cargo run --release -p blade-volume-train --features qhull --bin eval_psnr -- \
  --ply /tmp/blade-volume-bonsai-smoke.ply \
  --sparse etc/data/bonsai/sparse/0 \
  --images etc/data/bonsai/images \
  --width 32 --height 32 --train 8 --test 2 --test-every 8 \
  --max-steps 128
```

The explicit Qhull backend is part of the protocol. On this Bonsai subset,
the pure-Rust `simple_delaunay_lib` path exceeded 8 GiB while constructing
2,000-site adjacency; Qhull produced the same exact Delaunay class in under
400 MiB for the complete smoke run. Run benchmarks in a dedicated cgroup and
record its memory peak and OOM events alongside the required result fields.

For long-lived GPU drivers, `train_colmap --stop-after-steps N` ends the
current process cleanly after `N` updates while retaining the original global
LR schedule. It writes `<output>.ckpt.{ply,safetensors,trainstate}` by default;
continue with the identical training arguments plus
`--init-ply <output>.ckpt.ply`. When densification can still run, each segment
must end on a densification boundary so its accumulated geometry signal is not
discarded. The CLI rejects unsafe endpoints. In particular,
`bonsai_full_quality.toml` uses a 2,000-step first segment to reach its warmup
boundary, followed by 1,000-step segments whose endpoints remain aligned to
the 500-step densification cadence. Dynamic `radfoam-v1` densification stores
its initial count, active counter, next interval, and round in the v3 trainer
sidecar. A bounded dynamic segment may end at its next known boundary; run to
the global endpoint when an invocation needs to span multiple count-dependent
intervals.

`bonsai_quality.toml` is the first meaningful go/no-go protocol. Do not compare
its numbers with the historical audit run: camera rectification, terminal
integration, adjacency, and training semantics have changed since that run.

The checked-out `etc/data/bonsai` fixture contains 80 source images while its
COLMAP reconstruction describes 292. The quality manifest is therefore a
controlled subset protocol, not a full Mip-NeRF-360 or paper-comparable
benchmark. Published-quality comparisons require all 292 corresponding images
and a separate manifest that records that provenance explicitly.

Fetch the pinned 292-image scene with
`etc/fetch_test_dataset.sh bonsai-full`. The script sparse-checks out only the
373 MB Bonsai directory and exposes the canonical
`etc/data/bonsai-full/{images,sparse}` layout. Use
`bonsai_full_quality.toml` for this data; it deliberately remains an internal
budget rather than claiming published-method comparability.

The first full-dataset attempt is recorded as
`results/bonsai-full-partial-93c996f.toml`. It deliberately sets
`benchmark_complete = false`: training stopped at step 10,000 of 20,400 after
held-out PSNR stayed flat through three evaluations. Its final metrics do come
from a freshly reloaded PLY and are useful diagnostic evidence, but they must
not be presented as a completed manifest result or a reference comparison.

`bonsai_radfoam_v1_semantics.toml` defines a bundled semantic direction check.
It keeps the plateau run's 50K→200K cells, 128×128 resolution, split, batch,
and global step budget, but opts into the official v1 initialization
distribution, Smooth-L1 RGB loss, white background, and parameter-specific
learning rates. Its step-2,000 decision point is recorded in
`results/bonsai-radfoam-v1-semantics-step2000-86239aa.toml`. The freshly
reloaded model reached 10.83 dB train / 12.37 dB held out, versus 13.08 / 13.08
dB for the old scaled protocol at the same step, so the bundled run was stopped
instead of spending the remaining budget.

This negative result does not isolate a semantic regression. At that point the
small-batch run had processed 512,000 rays, while 2,000 official one-million-ray
updates represent 2 billion rays; the reference initialization and warmup were
therefore evaluated at a radically different data budget. Follow-up runs must
change one factor at a time from the old scaled baseline and record cumulative
rays. Only after initialization, loss, background, and parameter scheduling
have independent curves should they be recombined or used to justify a larger
batch/reference-scale attempt.

The first step-2,000 one-factor matrix is recorded in four additional result
files. Against the 13.08 / 13.08 dB train/held-out baseline, v1 initialization
alone reached 14.59 / 15.11 dB, white alone reached 13.64 / 13.53 dB, and
Smooth-L1 alone reached 13.40 / 13.36 dB. Smooth-L1 nevertheless collapsed one
held-out frame to 4.44 dB. The v1 schedule alone reached only 8.23 / 8.50 dB,
confirming that its update-indexed warmup is not transferable to a 256-ray
batch. The next controlled combination is therefore v1 initialization plus
white on the existing L1/cosine optimizer; Smooth-L1 and the v1 schedule remain
excluded pending larger-batch evidence.

That two-factor interaction is recorded in
`results/bonsai-radfoam-v1-init-white-step2000-8bbc167.toml`. It reached 14.83
dB train / 14.74 dB held out after reload. Relative to initialization alone,
white added 0.24 dB train but lost 0.37 dB held out. The robust scaled winner is
therefore v1 initialization alone with the existing black/L1/cosine training
path; its lossless step-2,000 checkpoint is the curve selected for continuation.

That continuation is recorded in
`results/bonsai-radfoam-v1-initialization-curve-step4000-01e68e3.toml`. Reloaded
held-out PSNR progresses 15.11 → 15.39 → 15.58 dB at steps 2K/3K/4K, while
cells grow 57,348 → 75,445 → 99,262. The advantage over the historical
scaled curve narrows from +2.03 to +1.25 to +0.35 dB. The last 1K segment takes
1,021 seconds for only +0.19 dB held out, so this diagnostic curve stops at 4K;
it is not a completed manifest or reference result.

`train_colmap --contribution-views N` is an experimental scaling control for
the prune/densify oracle. `0` remains the exhaustive default. The first
same-checkpoint comparison is recorded in
`results/bonsai-contribution-views32-step4500-ee63217.toml`: 32 stratified,
rotating views cut the 4K→4.5K segment from 635 to 299 seconds (2.12×), while
fresh-PLY held-out PSNR changed from 15.69 to 15.66 dB. It retained 1,223 fewer
cells and produced a different artifact/topology, so it has not met the
decision-agreement gate and must not replace exhaustive scans by default.

`train_colmap --lr-groups radfoam-v1-relative` separates the official initial
parameter-rate ratios from the official update-indexed time curves. Its first
isolated result is
`results/bonsai-radfoam-v1-relative-groups-step2000-a91766b.toml`. Under the
winning v1-initialization/black/L1/cosine setup it reached 14.72 dB train /
14.49 dB held out, versus 14.59 / 15.11 dB with legacy groups. The +0.13 dB
train and -0.62 dB held-out change rejects it as the 256-ray default while
preserving it as a larger-batch reference control.

`train_colmap --geometry-rebuild-schedule radfoam-v1` reproduces the reference
1, 3, 5, ... 99, 101 topology-update periods and persists its phase in the v3
trainer-state sidecar. The clean isolated comparison is
`results/bonsai-radfoam-v1-topology-cadence-step2000-e50e965.toml`. Contribution
view and pixel phase are keyed by absolute densification round in both runs.
Dynamic cadence used 44 rather than 20 scheduled updates, cost 24 seconds
(+3.7%), and changed train/held-out PSNR from 14.60 / 15.13 to 14.58 / 15.05
dB. Fixed-100 therefore remains the scaled default.

`train_colmap --densify-schedule radfoam-v1` implements the reference
cell-count-dependent growth interval independently from topology cadence. The
two-round protocol and result are recorded in
`bonsai_densification_cadence.toml` and
`results/bonsai-radfoam-v1-densification-cadence-step2805-11a7118.toml`.
Fixed-500 grew at steps 2,000 and 2,500; the dynamic arm used the identical
first contribution round and delayed the second growth to step 2,517. Both
ended with 66,054 cells after the same 718,080 sampled training rays. Dynamic
cadence changed train/held-out PSNR from 16.34 / 16.25 to 16.11 / 16.18 dB.
Its 13.2-second wall-time reduction is only 1.2%, and both training peaks were
about 481 MB with zero swap, OOM, or GPU faults. Fixed cadence remains the
scaled default.

The first larger-batch gate is defined in `bonsai_batch1024_optimizer.toml`
and recorded in `results/bonsai-batch1024-optimizer-step500-6877bea.toml`.
All arms process 512,000 sampled training rays and use the same exhaustive
round-0 contribution sample. With the original update-indexed 20,400-step
horizon and fixed-100 topology cadence, the exact v1 schedule reaches only
9.97 / 10.53 dB train/held out, versus 15.11 / 14.89 dB for cosine/legacy.
Relative v1 parameter groups under cosine are neutral at 14.94 / 14.95 dB:
-0.17 dB train and +0.06 dB held out. Neither changes the default.

A fourth arm normalizes schedules by sampled rays: 5,100 total updates,
fixed-25 topology refreshes, and a 125-step post-warmup densification cadence.
At batch 1,024 it reaches 15.26 / 15.33 dB, improving the corrected batch-256
same-ray checkpoint by +0.66 / +0.20 dB. This is the selected direction for a
multi-boundary and native-aspect confirmation, not yet a new default. All four
training scopes peak between 409 and 420 MB under a 4 GiB, zero-swap cgroup and
record no pressure, OOM, truncation, or GPU-fault event.

The two-round/native-aspect follow-up is defined in
`bonsai_batch1024_native_aspect.toml` and recorded in
`results/bonsai-batch1024-native-aspect-step625-07ad939.toml`. The lossless
square continuation reaches 66,078 cells and 15.55 / 15.29 dB at 128×128;
held-out quality is 0.04 dB below its step-500 value. On a common fresh-Ply
192×128 evaluation, square training reaches 15.56 / 15.28 dB while native 3:2
training reaches 15.42 / 15.05 dB, a -0.14/-0.23 dB change. Native training
uses 1,566,720 rather than 1,044,480 contribution rays per round and raises
the training peak from 451 to 519 MB, with zero swap, pressure, OOM,
truncation, or GPU faults in either arm. The rectifier and ray generator cover
the full calibrated camera domain at either output shape, so 128×128 is kept
as the efficient scaled protocol; it is not a crop of the source field.

The corrected two-round same-ray comparison is defined in
`bonsai_batch_same_ray.toml` and recorded in
`results/bonsai-same-ray-batch-step640k-945b931.toml`. At 640,000 optimizer
rays, 25 topology refreshes, and two exhaustive growth rounds, batch 256
reaches 65,801 cells and 15.32 / 15.12 dB train/held out. Ray-normalized batch
1,024 reaches 66,078 cells and 15.55 / 15.29 dB: +0.23/+0.17 dB. The matched
continuation scopes take 391 and 384 seconds and peak at 424 and 451 MB,
respectively, with zero swap, pressure, OOM, truncation, or GPU faults. Batch
1,024 remains the selected next direction, but the modest, mixed per-frame
gain still requires a third boundary or another scene before changing defaults.
