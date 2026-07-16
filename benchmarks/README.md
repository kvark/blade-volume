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
