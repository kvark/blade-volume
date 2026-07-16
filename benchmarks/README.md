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
the 500-step densification cadence.

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
