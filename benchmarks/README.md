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

The post-Meganeura-uprev smoke run on commit `9d224dd` reproduced the accepted
baseline exactly: both live and fresh-PLY evaluation report 16.58 dB train /
17.00 dB held out for 2,000 cells. Its ignored artifact and telemetry live at
`target/audit-runs/bonsai-smoke-9d224dd/`.

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

Raw models, checkpoints, telemetry, and result snapshots belong under `target`
or the ignored `benchmarks/results` directory. Accepted conclusions are
summarized here with their source commit; the protocol manifests remain the
tracked machine-readable inputs.

The first full-dataset attempt, from commit `93c996f`, deliberately stopped at
step 10,000 of 20,400 after
held-out PSNR stayed flat through three evaluations. Its final metrics do come
from a freshly reloaded PLY and are useful diagnostic evidence, but they must
not be presented as a completed manifest result or a reference comparison.

`bonsai_radfoam_v1_semantics.toml` defines a bundled semantic direction check.
It keeps the plateau run's 50K→200K cells, 128×128 resolution, split, batch,
and global step budget, but opts into the official v1 initialization
distribution, Smooth-L1 RGB loss, white background, and parameter-specific
learning rates. At its step-2,000 decision point on commit `86239aa`, the freshly
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

The first step-2,000 one-factor matrix was run from commit `71c8946`. Against
the 13.08 / 13.08 dB train/held-out baseline, v1 initialization
alone reached 14.59 / 15.11 dB, white alone reached 13.64 / 13.53 dB, and
Smooth-L1 alone reached 13.40 / 13.36 dB. Smooth-L1 nevertheless collapsed one
held-out frame to 4.44 dB. The v1 schedule alone reached only 8.23 / 8.50 dB,
confirming that its update-indexed warmup is not transferable to a 256-ray
batch. The next controlled combination is therefore v1 initialization plus
white on the existing L1/cosine optimizer; Smooth-L1 and the v1 schedule remain
excluded pending larger-batch evidence.

That two-factor interaction on commit `8bbc167` reached 14.83
dB train / 14.74 dB held out after reload. Relative to initialization alone,
white added 0.24 dB train but lost 0.37 dB held out. The robust scaled winner is
therefore v1 initialization alone with the existing black/L1/cosine training
path; its lossless step-2,000 checkpoint is the curve selected for continuation.

Through commit `01e68e3`, reloaded held-out PSNR progressed
15.11 → 15.39 → 15.58 dB at steps 2K/3K/4K, while
cells grow 57,348 → 75,445 → 99,262. The advantage over the historical
scaled curve narrows from +2.03 to +1.25 to +0.35 dB. The last 1K segment takes
1,021 seconds for only +0.19 dB held out, so this diagnostic curve stops at 4K;
it is not a completed manifest or reference result.

`train_colmap --contribution-views N` is an experimental scaling control for
the prune/densify oracle. `0` remains the exhaustive default. In the first
same-checkpoint comparison on commit `ee63217`, 32 stratified,
rotating views cut the 4K→4.5K segment from 635 to 299 seconds (2.12×), while
fresh-PLY held-out PSNR changed from 15.69 to 15.66 dB. It retained 1,223 fewer
cells and produced a different artifact/topology, so it has not met the
decision-agreement gate and must not replace exhaustive scans by default.

`train_colmap --lr-groups radfoam-v1-relative` separates the official initial
parameter-rate ratios from the official update-indexed time curves. In its
first isolated result on commit `a91766b`, under the
winning v1-initialization/black/L1/cosine setup it reached 14.72 dB train /
14.49 dB held out, versus 14.59 / 15.11 dB with legacy groups. The +0.13 dB
train and -0.62 dB held-out change rejects it as the 256-ray default while
preserving it as a larger-batch reference control.

`train_colmap --geometry-rebuild-schedule radfoam-v1` reproduces the reference
1, 3, 5, ... 99, 101 topology-update periods and persists its phase in the v3
trainer-state sidecar. The clean isolated comparison on commit `e50e965` keys
contribution view and pixel phases by absolute densification round in both
runs.
Dynamic cadence used 44 rather than 20 scheduled updates, cost 24 seconds
(+3.7%), and changed train/held-out PSNR from 14.60 / 15.13 to 14.58 / 15.05
dB. Fixed-100 therefore remains the scaled default.

`train_colmap --densify-schedule radfoam-v1` implements the reference
cell-count-dependent growth interval independently from topology cadence. The
two-round protocol is recorded in `bonsai_densification_cadence.toml`; the
result comes from commit `11a7118`.
Fixed-500 grew at steps 2,000 and 2,500; the dynamic arm used the identical
first contribution round and delayed the second growth to step 2,517. Both
ended with 66,054 cells after the same 718,080 sampled training rays. Dynamic
cadence changed train/held-out PSNR from 16.34 / 16.25 to 16.11 / 16.18 dB.
Its 13.2-second wall-time reduction is only 1.2%, and both training peaks were
about 481 MB with zero swap, OOM, or GPU faults. Fixed cadence remains the
scaled default.

The first larger-batch gate is defined in `bonsai_batch1024_optimizer.toml`
and was run on commit `6877bea`.
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
`bonsai_batch1024_native_aspect.toml` and was run on commit `07ad939`. The
lossless square continuation reaches 66,078 cells and 15.55 / 15.29 dB at 128×128;
held-out quality is 0.04 dB below its step-500 value. On a common fresh-Ply
192×128 evaluation, square training reaches 15.56 / 15.28 dB while native 3:2
training reaches 15.42 / 15.05 dB, a -0.14/-0.23 dB change. Native training
uses 1,566,720 rather than 1,044,480 contribution rays per round and raises
the training peak from 451 to 519 MB, with zero swap, pressure, OOM,
truncation, or GPU faults in either arm. The rectifier and ray generator cover
the full calibrated camera domain at either output shape, so 128×128 is kept
as the efficient scaled protocol; it is not a crop of the source field.

The corrected two-round same-ray comparison is defined in
`bonsai_batch_same_ray.toml` and was run on commit `945b931`. At 640,000
optimizer rays, 25 topology refreshes, and two exhaustive growth rounds, batch 256
reaches 65,801 cells and 15.32 / 15.12 dB train/held out. Ray-normalized batch
1,024 reaches 66,078 cells and 15.55 / 15.29 dB: +0.23/+0.17 dB. The matched
continuation scopes take 391 and 384 seconds and peak at 424 and 451 MB,
respectively, with zero swap, pressure, OOM, truncation, or GPU faults. Batch
1,024 remains the selected next direction, but the modest, mixed per-frame
gain still requires a third boundary or another scene before changing defaults.

The third-round decision protocol is
`bonsai_batch_same_ray_round3.toml`; its result comes from commit `ad697dd`.
At 768,000 optimizer
rays and three growth rounds, batch 256 reaches 75,452 cells and 15.41 / 14.76
dB train/held out. Ray-normalized batch 1,024 reaches 75,969 cells and 16.08 /
15.66 dB: +0.67/+0.90 dB, improving seven of eight test frames. The matched
continuations both take about 444–445 seconds; batch 1,024 peaks at 489 MB
versus 475 MB. Both report zero swap, pressure, OOM, truncation, or GPU faults.
This passes the Bonsai gate and selects 1,024 as the scaled pixel-batch default.
`train_colmap` now uses that value when `--pixel-batch` is omitted; explicit
values, including `0` for full-image batches, retain their existing semantics.

The second-scene gate is
[`room_batch_same_ray_round3.toml`](room_batch_same_ray_round3.toml), run on
commit `9d224dd` against all 311
registered Room images from the same pinned Mip-NeRF-360 revision. At 768,000
optimizer rays, 30 topology refreshes, and three exhaustive growth rounds,
batch 256 reaches 74,642 cells and 17.54 / 17.41 dB train/held out. Batch 1,024
reaches 75,718 cells and 18.50 / 18.23 dB: +0.96/+0.82 dB, improving six of
eight test frames. Fresh-PLY reload reproduces every metric. The arms take
1495.5 and 1484.5 seconds and peak at 494 and 531 MiB respectively; both report
zero swap, pressure, OOM, truncation, or GPU faults. Room therefore confirms
the batch-1,024 direction rather than exposing a Bonsai-only result.

The executed official baseline is
[`room_radfoam_v1_reference.toml`](room_radfoam_v1_reference.toml). The exact
first 5,000 updates of RadFoam v1 on Room process five billion mixed-view rays,
serialize 735,103 cells with 11,185,482 directed edges, and reach 30.0239 dB on
all 39 held-out views at 1,557×1,038. The run takes 14m 14.978s, peaks at
10,280,112,128 bytes of host memory and 7,133 MiB sampled GPU memory, and has
zero swap, pressure, OOM, or GPU fault. An unchanged full run is not feasible
on this host: switching its cached training rays from downsample 4 to 2 at
update 5,000 exceeds a 32 GiB cgroup.

Blade directly loads the official PLY. Commit `00de721` fixes the missing
per-cell nonnegative SH clamp in the CPU renderer, standalone and scene WGSL,
Gaussian WGSL, and differentiable training graph. The identical 39-view
cross-render improves from 28.97 to 29.59 dB, leaving a 0.43 dB mean gap to
upstream and passing the renderer-parity gate. Models trained before that
commit retain historical metrics; they must be retrained before their quality
is compared under the corrected runtime semantics.

That controlled retrain is recorded without replacing the historical result in
[`room_batch_same_ray_round3.toml`](room_batch_same_ray_round3.toml). The
identical selected batch-1,024 protocol at commit `00de721` produces 75,809
cells and 1,132,150 directed edges; fresh-Ply evaluation reaches 19.02 dB train
/ 18.84 dB held out, +0.52/+0.61 dB over the originally published result. The
old cloud reaches only 17.79 / 17.37 dB when evaluated under the corrected
renderer, confirming that old and new models cannot be compared without
retraining. The new run takes 24m 0.142s, peaks at 543,166,464 host bytes and
551 MiB sampled GPU memory, and records zero truncation, swap, pressure, OOM,
or GPU faults. Its ignored raw artifact is
`target/audit-runs/room-batch1024-step750-00de721`; the `comparisons/` directory
contains the eight selected held-out render comparisons. An all-39-view
18.44 dB evaluation is retained only as a coverage diagnostic because the
bounded protocol trains on 255 views and selects the first eight held-out
views, leaving the end of the capture under-covered. The selected comparison
PNGs are recognizable but still have severe cell/color fragmentation and lose
fine room structure; this is a scaling baseline, not viewer-ready quality.

Commit `3d2ba74` adds deterministic mixed-view optimizer batches without
changing the default one-view policy. The controlled Room arm changes only
`views_per_batch` from 1 to 16, giving each selected camera 64 of the same
1,024 rays per Adam step. At the same 768,000-ray budget it reaches 75,722
cells and 20.21 / 19.94 dB after fresh-Ply reload: +1.19/+1.10 dB over the
corrected one-view train/held-out result, with all eight selected test frames
improving. The all-39-view coverage diagnostic rises from 18.44 to 19.66 dB
(+1.22). Wall time increases only 1.6%, from 1,440.142 to 1,463.566 seconds;
host peak rises 3.1% to 560,238,592 bytes and sampled GPU memory remains 551
MiB. All three exhaustive contribution rounds report zero truncation, and the
scope records zero swap, pressure, OOM, or GPU faults. Visual structure is
more coherent but still visibly fragmented. Raw results remain ignored under
`target/audit-runs/room-batch1024-mixed16-step750-3d2ba74`.

Matched continuations to step 1,000 retain the result after two more growth
rounds. One-view training reaches 100,158 cells and 20.18 / 19.95 dB, while
16-view training reaches 99,970 cells and 21.31 / 20.90 dB. All eight selected
held-out frames still improve, by +0.63 to +1.23 dB; the all-39 coverage
diagnostic rises from 19.53 to 20.70 dB (+1.17). Continuation wall time differs
by only 0.9% (1,044.090 versus 1,053.509 seconds), the mixed run has the lower
host peak (650,641,408 versus 670,986,240 bytes), and both peak at 567 MiB
sampled GPU memory with zero truncation, swap, OOM, or GPU faults. The 750-step
mixed run already matches the 1,000-step one-view held-out score (19.94 versus
19.95 dB) with 25% fewer optimizer rays and about 24% fewer cells.

This later boundary satisfies the selection gate. `train_colmap` now chooses
16 views automatically for random-pixel batches; full-image and patch modes
remain one view, and explicit overrides remain available. The library-level
`AppearanceFitConfig` default stays at one view so direct callers do not change
semantics implicitly. The ignored step-1,000 artifacts are
`target/audit-runs/room-batch1024-{one1,mixed16}-step1000-3d2ba74`.

The selected mixed-view trajectory continues from step 1,000 to 1,500 through
four additional exhaustive growth rounds. It reaches 174,410 cells and
2,658,968 directed edges; fresh-Ply evaluation exactly reproduces 22.39 dB
train / 21.83 dB held out, +1.08/+0.93 dB over step 1,000. The all-39 coverage
diagnostic rises from 20.70 to 21.60 dB (+0.90). Mean contribution-path length
grows smoothly from 69.4 to 88.2 segments, the maximum from 155 to 178, and all
four 1,044,480-ray scans retain zero truncation. The 500-step continuation takes
2,883.365 seconds, peaks at 1,038,524,416 host bytes and 719 MiB sampled GPU
memory, and records zero swap, pressure, OOM, or GPU faults. Visual structure
continues to improve but remains painterly and fragmented rather than
viewer-ready. The raw artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step1500-a86707a`.

One final 125-step segment under the initial 200,000-cell cap reaches that cap
exactly: contribution round 9 prunes 67 cells and splits 25,657 survivors from
174,410 to 200,000 cells. Fresh-Ply evaluation reproduces 22.63 dB train /
22.00 dB held out, +0.24/+0.17 dB over step 1,500; the all-39 diagnostic rises
by 0.20 dB to 21.80. Its 1,044,480-ray scan averages 94.3 segments, reaches a
maximum of 188, and truncates none. The 964.421-second continuation peaks at
1,060,331,520 host bytes and 688 MiB sampled GPU memory with zero swap,
pressure, OOM, or GPU faults. Visual structure continues to sharpen but remains
fragmented. The raw artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step1625-d454892`.

Raising only the capacity cap to 400,000 and continuing to step 2,000 adds
three exhaustive growth rounds and reaches 303,891 cells with 4,658,338
directed edges. Fresh-Ply evaluation reproduces 23.16 dB train / 22.31 dB held
out, +0.53/+0.31 dB over step 1,625; the all-39 diagnostic rises by 0.42 dB to
22.22. Mean contribution paths increase smoothly from 100.6 to 113.5 segments,
the maximum from 205 to 223, and all three 1,044,480-ray scans truncate none.
The 3,354.647-second continuation peaks at 1,730,691,072 host bytes and 936 MiB
sampled GPU memory with zero swap, pressure, OOM, or GPU faults. Large surfaces
and boundaries are cleaner, but fine detail remains fragmented. The raw
artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step2000-1ea4b28`.

Continuing to step 2,250 reaches the 400,000-cell cap exactly, with 6,140,550
directed edges. Fresh-Ply evaluation reproduces 23.42 dB train / 22.43 dB held
out, +0.26/+0.12 dB over step 2,000; the all-39 diagnostic rises by 0.16 dB to
22.38. The two exhaustive scans average 120.2 and 127.0 segments, reach maxima
of 235 and 246, and truncate no rays. A controlled reload at 512 rather than
256 maximum steps is metric-identical at 23.42/22.43 dB, so this artifact is
not clipped, but the 10-step headroom gates further growth on raising the
budget. The 2,863.453-second continuation peaks at 2,309,640,192 host bytes and
954 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU faults. The
raw artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step2250-f92281f`.

The next guarded round raises the target to the 735,103-cell reference prefix,
extends the growth horizon to step 2,875, and raises traversal capacity to 512.
At step 2,375 it reaches 459,888 cells and 7,065,808 directed edges. The
exhaustive scan measures 133.6 mean / 259 maximum segments with zero
truncation, proving that the old 256-step budget is no longer sufficient even
though a diagnostic 256-step reload rounds to the same PSNR. Fresh max-512 PLY
evaluation reproduces 23.55 dB train / 22.53 dB held out; all 39 held-out views
average 22.47 dB. The 2,411.225-second continuation peaks at 2,324,807,680 host
bytes and 1,362 MiB sampled GPU memory with zero swap, pressure, OOM, or GPU
faults. The raw artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step2375-1fd1783`.

At step 2,500 the same max-512 protocol reaches 528,732 cells and 8,128,410
directed edges. Fresh-Ply evaluation reproduces 23.69 dB train / 22.57 dB held
out, +0.14/+0.04 dB over step 2,375; the all-39 diagnostic rises by 0.09 dB to
22.56. The exhaustive scan measures 140.9 mean / 267 maximum segments with
zero truncation, prunes 121 cells, and adds 68,965 splits. The 2,702.106-second
continuation peaks at 2,646,769,664 host bytes and 1,690 MiB sampled GPU memory
with zero swap, pressure, OOM, or GPU faults. Quality is still improving, but
held-out gains are now small. The raw artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step2500-2f997cf`.

At step 2,625 the max-512 ladder reaches 607,908 cells and 9,350,692 directed
edges. Fresh-Ply evaluation reproduces 23.80 dB train / 22.64 dB held out,
+0.11/+0.07 dB over step 2,500; the all-39 diagnostic rises by 0.06 dB to
22.62. The exhaustive scan measures 148.5 mean / 275 maximum segments with
zero truncation, prunes 116 cells, and adds 79,292 splits. The 3,021.045-second
continuation peaks at 3,013,816,320 host bytes and 1,786 MiB sampled GPU memory
with zero swap, pressure, OOM, or GPU faults. The raw artifact remains ignored
at `target/audit-runs/room-batch1024-mixed16-step2625-1a013d7`.

At step 2,750 the max-512 ladder reaches 698,940 cells and 10,756,702 directed
edges, within 5% of the reference capacity. Fresh-Ply evaluation reproduces
23.89 dB train / 22.62 dB held out, +0.09/-0.02 dB versus step 2,625; the
all-39 diagnostic still rises by 0.03 dB to 22.65. The selected regression is
concentrated in a larger dark foreground smear on DSCF4707. The exhaustive scan
measures 156.2 mean / 299 maximum segments with zero truncation. The
3,411.154-second continuation peaks at 3,572,768,768 host bytes and 1,773 MiB
sampled GPU memory with zero swap, pressure, OOM, or GPU faults. The raw
artifact remains ignored at
`target/audit-runs/room-batch1024-mixed16-step2750-66002d9`.

The step-2,875 boundary reaches the 735,103-cell official-prefix capacity
exactly, with 11,310,690 directed edges. Fresh-Ply evaluation reproduces 24.07
dB train / 22.77 dB held out, +0.18/+0.15 dB over step 2,750; the all-39
diagnostic rises by 0.11 dB to 22.76. The exhaustive scan measures 164.2 mean /
305 maximum segments with zero truncation, prunes 192 cells, and adds the
36,355 splits needed to hit the cap. The 3,815.955-second continuation peaks at
3,634,937,856 host bytes and 1,822 MiB sampled GPU memory with zero swap,
pressure, OOM, or GPU faults. Matching capacity is beneficial but not
sufficient: this 128² run has processed only 2,944,000 optimizer rays, versus
roughly five billion for the 30.02 dB official prefix. The raw artifact remains
ignored at `target/audit-runs/room-batch1024-mixed16-step2875-67ef5c8`.

A fixed-cap step-2,875→3,000 continuation then attempted to measure optimizer
return without further densification. It completed geometry cycles through
step 2,975, but the NVIDIA device emitted Xid 79 (“GPU has fallen off the bus”)
during the final 25-step interval. The cgroup fault watcher killed the scope
with exit 143 before a checkpoint or model could be written. This was not a
cgroup OOM: host peak was 2,692,427,776 bytes with zero swap, pressure, or OOM
events; the final GPU sample was 1,788 MiB, 72 °C, and 100% utilization. The
validated step-2,875 checkpoint remains the only resume source. A host reboot
is required before retrying because both NVIDIA and this user's systemd manager
remain wedged. Failure logs remain ignored at
`target/audit-runs/room-batch1024-mixed16-step3000-3792145`.

After reboot, the identical fixed-cap retry completed step 3,000 with 735,103
cells and 11,311,556 directed edges. Fresh-Ply evaluation reproduces 24.29 dB
train / 22.92 dB on the selected held-out views, +0.22/+0.15 dB over step
2,875; the all-39 diagnostic rises by 0.14 dB to 22.90. The 2,982.305-second
continuation peaks at 1,835 MiB sampled GPU memory and 74 °C with no GPU fault,
swap, or OOM. Final serialization/evaluation does reach the 4 GiB cgroup cap
and increments `memory.events:max` 221 times, so later 735K-cell timing runs
use a 6 GiB scope to avoid reclaim-distorted measurements. Comparison images
remain visibly fragmented despite the general metric gain. The raw artifact
remains ignored at
`target/audit-runs/room-batch1024-mixed16-step3000-retry-bd28c53`.

GPU pass profiling then identified the real training bottleneck. A representative
735,103-cell step issued 414 Meganeura dispatches and spent 19.747 of 19.776 GPU
seconds (99.85%) in 51 dense embedding-gradient scatter-adds. The dense kernel
scanned every table row for every recorded path element. Meganeura `fba040a`
keeps that deterministic implementation for small workloads and switches large
ones to a zero pass plus atomic f32 scatter. A one-step Room replay changes model
parameters by at most `1.49e-8`, with identical reported loss, PSNR, cell count,
and adjacency size. The step-2,875→2,900 optimizer/topology boundary falls from
505 to 21 seconds; after subtracting the Qhull rebuild, the optimizer portion is
about 98× faster. The raw profile and parity artifacts remain ignored under
`target/audit-runs/room-step2878-gpu-profile-209d94a` and
`target/audit-runs/room-step2900-atomic-scatter-local`.

With Blade `27242ad`, the fixed-cap step-3,000→3,125 continuation completes its
five training/topology cycles in 218 seconds instead of the old comparable
2,677 seconds (12.3×). Fresh-Ply quality reaches 24.44/23.06 dB and 23.00 dB
across all 39 held-out views. Raising the batch from 1,024 to 4,096 rays then
adds four times as many samples in 227 seconds—only 4% longer—and reaches
24.74/23.26 dB plus 23.19 dB across all 39 views. It stays bounded at 3.57 GiB
host peak and 3,763 MiB sampled VRAM with zero pressure, swap, OOM, or GPU
faults. The 4,096-ray/16-view setting is selected for the next resolution
ladder. Images remain visibly fragmented, so the result improves the training
economics and evidence, not the production-readiness verdict. Artifacts remain
ignored at `target/audit-runs/room-batch1024-mixed16-step3125-27242ad` and
`target/audit-runs/room-batch4096-mixed16-step3250-27242ad`.

The selected path then moves to 256². Reloading step 3,250 at that resolution
gives 24.55/23.23 dB train/held out and 23.19 dB across all 39 views. Continuing
to step 3,375 reaches 24.77/23.39 dB and 23.34 dB all-39. A matched 64-view
batch ties the 16-view result while taking 6% more training/topology time and
3% more peak host memory, so 16 views remains selected.

Matched fixed-cap topology gates select an exact Qhull rebuild every 250 steps.
Cadence 125 ties cadence 25 at step 3,500 while cutting the training/topology
interval from 219 to 46 seconds. Cadence 250 then ties cadence 125 at step
3,750 while cutting 111 seconds to 63. Cadence 500 loses 0.01 dB selected and
all-39 at step 4,500 for only a 25% interval saving, so it is rejected.

The selected cadence-250 path reaches aligned step 5,000 with 735,103 cells,
11,308,074 directed edges, and 10.88 million optimizer rays. Fresh-Ply
evaluation gives 25.69/24.03 dB train/held out and 23.93 dB across all 39
views. The final 500 steps add only 0.06 dB selected and 0.05 dB all-39.
Foreground furniture and thin occlusion boundaries remain visibly smeared, so
this is the latest reconstruction checkpoint rather than a viewer-ready
result. Raw artifacts remain ignored under
`target/audit-runs/room-batch4096-mixed16-resolution256-step*-{e23a4d3,98d9ee9}`.

Commit `77c19b7` moves headless evaluation onto the same production
RadFoam/PowerFoam compute tracer as the viewer while keeping the CPU oracle as
the default. On the selected step-5,000 Room PLY, the opt-in GPU path exactly
matches the aggregate 25.69/24.03 dB train/held-out and 23.93 dB all-39
metrics; reported per-view PSNR differs by at most 0.01 dB because the
production target is RGBA16F. The apples-to-apples 255+8-view pass falls from
548.251 to 119.668 seconds (4.58×), and the 39-view pass falls from 77.923 to
7.758 seconds (10.04×). Host peaks are 510,726,144 and 499,122,176 bytes,
respectively; the GPU pass samples 279 MiB VRAM and records zero swap,
pressure, OOM, or GPU faults. Raw parity and timing artifacts remain ignored
at `target/audit-runs/room-step5000-gpu-eval-local`.

Commit `5e3f81d` adds built-in phase timing and reuses an already-current model
download across geometry, checkpoint, and finalization boundaries. In a
matched step-5,000→5,100 continuation, training falls from 167.369 to 120.675
seconds (1.39×) and whole-command time falls from 205.641 to 158.960 seconds
(1.29×). The eliminated duplicate finalization download accounts for 24.906
seconds; checkpoint time also falls from 98.460 to 76.406 seconds because it
no longer starts with another full parameter download. Both runs follow the
same 0.0919→0.0854 loss trace and produce 735,103 cells with 11,308,046
directed edges. The optimized checkpoint PLY is byte-identical to its final
PLY, selected held-out evaluation remains 24.03 dB, and the 6 GiB scope records
zero pressure, swap, OOM, or GPU faults. Peak host memory falls from
4,421,963,776 to 4,200,992,768 bytes. Raw evidence remains ignored at
`target/audit-runs/room-step5000-phase-profile-{local,readback-local}`.

The phase profile identifies serialization as the remaining dominant endpoint
cost. Meganeura branch `perf/stream-checkpoints-fba` at `cb64f67` writes
safetensors directly to the destination file rather than materializing a
second complete byte vector. A matched Blade run reduces peak host memory from
4,421,963,776 to 3,726,434,304 bytes (15.7%) while preserving 24.03 dB selected
held-out quality and byte-identical checkpoint/final PLYs within the run.
Checkpoint time is effectively unchanged at 100.060 versus 98.460 seconds, so
this is a pending upstream memory fix, not a runtime claim. Blade remains
pinned to merged Meganeura commit `fba040a` until that branch lands. The raw
artifact remains ignored at
`target/audit-runs/room-step5000-phase-profile-stream2-local`.
