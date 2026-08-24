# Phone capture

The capture stage produces posed photographs and sparse points. It does not
introduce a mesh: COLMAP is used only for camera calibration, poses, and the
initial point cloud consumed by `blade-volume-train`.

## Record

Use one physical lens for the entire clip. A stock camera app is sufficient if
it holds its settings; [Blackmagic Camera](https://www.blackmagicdesign.com/products/blackmagiccamera)
on iOS or Android and [Open Camera](https://opencamera.sourceforge.io/) on
Android expose the useful locks directly.

- Lock focus, exposure, ISO, white balance, lens selection, and resolution.
- Prefer the main lens without digital zoom, portrait mode, HDR, or lens
  switching. Avoid electronic stabilisation if the app permits it.
- Use diffuse, steady light. Keep highlights below clipping and use a shutter
  fast enough that individual frames are sharp.
- Walk one slow, continuous loop around the subject for 10--30 seconds. Keep
  the subject in view, include texture, and maintain strong overlap between
  neighbouring views. Do not pause in one place to make the clip longer.
- Keep the scene rigid. Moving people, foliage, screens, reflections, and
  automatic lighting changes make one static cloud an inconsistent target.

Transfer the original video without social-media recompression.

## Recover poses and sparse points

Install `ffmpeg` and [COLMAP](https://colmap.github.io/install.html) (a
CUDA-enabled build is recommended), then run:

```bash
etc/colmap.sh phone.mov etc/data/my-capture 3
```

The optional final argument is the number of frames extracted per second.
Start at 3; use 2 for a very slow recording or 4--5 when neighbouring frames
otherwise move too far. More nearly identical frames cost matching and mapping
time without adding useful baseline.

The command refuses to overwrite an existing path and publishes the result
only after COLMAP succeeds. Its output is already in the trainer's canonical
layout:

```text
etc/data/my-capture/
├── images/frame_000001.png
├── database.db
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

For a memory-limited workstation, contain the full extraction and mapping run:

```bash
etc/cgroup_run.sh --mem 12G --gpu-log capture-gpu.log -- \
    etc/colmap.sh phone.mov etc/data/my-capture 3
```

The wrapper intentionally assumes one unknown `SIMPLE_RADIAL` camera because
all extracted frames came from one locked phone lens. If the camera is already
calibrated, run the same three COLMAP stages manually and supply its known
model and parameters to `feature_extractor`; guessed high-order calibration is
worse than the simple shared model.

Inspect the sparse model in COLMAP before training. Most frames should be
registered in one model, the camera path should follow the recorded motion,
and points should outline the subject rather than form disconnected islands.
If `sparse/0` is absent, improve overlap or texture, reduce blur, and verify
that the phone did not switch lenses.

## Train clouds

For the direct Gaussian light field and relightable Gaussian surface:

```bash
cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse etc/data/my-capture/sparse/0 \
    --images etc/data/my-capture/images \
    --gaussian-output my-capture-static.ply \
    --pbr-gaussian-output my-capture-pbr.ply \
    --output my-capture.scene
```

A single uncontrolled-light clip can measure capture-light re-rendering, but
it cannot establish that recovered material and illumination are physically
separate. Relighting validation needs camera-aligned repeated captures under
measured lights, as described in [Inverse rendering](INVERSE_RENDERING.md).
