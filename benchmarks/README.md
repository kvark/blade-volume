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
cargo run --release -p blade-volume-train --bin train_colmap -- \
  --sparse etc/data/bonsai/sparse/0 \
  --images etc/data/bonsai/images \
  --output /tmp/blade-volume-bonsai-smoke.ply \
  --width 32 --height 32 --views 8 --test-views 2 --test-every 8 \
  --max-points 2000 --max-steps 128 --pixel-batch 256 \
  --steps-per-view 25 --learning-rate 0.1 --sh-degree 0

cargo run --release -p blade-volume-train --bin eval_psnr -- \
  --ply /tmp/blade-volume-bonsai-smoke.ply \
  --sparse etc/data/bonsai/sparse/0 \
  --images etc/data/bonsai/images \
  --width 32 --height 32 --train 8 --test 2 --test-every 8 \
  --max-steps 128
```

`bonsai_quality.toml` is the first meaningful go/no-go protocol. Do not compare
its numbers with the historical audit run: camera rectification, terminal
integration, adjacency, and training semantics have changed since that run.

