# Phone capture

The capture stage produces posed photographs and a point cloud. Sparse points
are the quick default; optional dense multi-view stereo supplies independent
surface correspondences and normals. It does not introduce a mesh: COLMAP is
used only for camera calibration, poses, and point-cloud initialization.

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

## Recover poses and point clouds

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

To also recover the independently matched surface cloud, add `--dense`:

```bash
etc/colmap.sh --dense phone.mov etc/data/my-capture 3
```

This runs COLMAP's geometrically consistent PatchMatch stereo and fuses only
the oriented points; it deliberately stops before Poisson, Delaunay, or any
other meshing stage. The additional output is `dense/fused.ply` (and COLMAP's
workspace/visibility side data). Dense images default to at most 2000 pixels
on their long side; set `COLMAP_DENSE_MAX_IMAGE_SIZE` when the GPU requires a
smaller bound or a carefully monitored run justifies a larger one.

For a controlled relighting capture, the measured-light photographs may be a
poor stereo source because shadows and highlights move between views. Capture
one broad, diffuse image without moving the camera at every pose and give that
directory to the dense stage:

```bash
etc/colmap.sh --dense-images etc/data/my-broad-light phone.mov \
    etc/data/my-capture 3
```

The alternate directory must contain the same `frame_*.png` basenames at the
same resolution, through the same locked lens, with the object and camera at
the exact corresponding poses. It is used only by PatchMatch and fusion;
`OUTPUT/images` still contains the video frames used to fit appearance. The
wrapper rejects a missing counterpart before starting COLMAP. A separately
registered orbit is not equivalent: a real soft-object gate produced a dense
cloud but failed held-light transfer after global alignment.

For a memory-limited workstation, contain the full extraction and mapping run:

```bash
etc/cgroup_run.sh --mem 12G --gpu-log capture-gpu.log -- \
    etc/colmap.sh --dense phone.mov etc/data/my-capture 3
```

Dense stereo is also VRAM-heavy. Watch `capture-gpu.log`; the memory cgroup
bounds system memory but cannot turn a GPU out-of-memory failure into a useful
COLMAP result.

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

Registration count and reprojection error are necessary but not sufficient.
A short, low-parallax arc can register every frame below one pixel while
triangulating the subject into a long depth spike. Check that the cloud has a
plausible shape and that the camera path covers the intended sides of it. If
`reconstruct` drops nearly every sparse point as an outlier, treat that as a
capture failure and repeat a wider, continuous orbit; do not weaken the
geometry filter to admit the bad cloud.

## Train clouds

For the direct Gaussian light field and relightable Gaussian surface:

```bash
cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse etc/data/my-capture/sparse/0 \
    --images etc/data/my-capture/images \
    --dense-cloud etc/data/my-capture/dense/fused.ply \
    --gaussian-output my-capture-static.ply \
    --pbr-gaussian-output my-capture-pbr.ply \
    --output my-capture.scene
```

`reconstruct` spatially averages a dense input to at most 50,000 particles by
default, retains COLMAP's fused normals, estimates local disc support, and then
fits appearance from the photographs. The relightable model uses this dense
surface, while the direct captured-light Gaussian keeps the sparse training
tracks: a leakage-free Bonsai gate selected that split because dense geometry
improved PBR support but softened the direct light field. Both outputs remain
clouds. `--dense-max-points` changes the bound; it is a memory/throughput limit,
not a request to duplicate points. Omit `--dense-cloud` to use sparse geometry
for both outputs.

Missing-surface experiments can replay one exact trained Gaussian with
`--missing-tracks-base my-capture-pbr.ply --missing-tracks-output tracks.ply`.
This dedicated diagnostic input is mutually exclusive with
`--pbr-gaussian-output`, so a repeat cannot accidentally retrain the base whose
coverage defines the missing pixels. It additionally needs masks, at least
three aligned `--normal-images`/`--normal-environment` pairs, and at least six
training cameras. It writes a separate cloud and does not modify the base.
The diagnostic keeps each triangulated oriented surfel and adds smaller point
samples between local neighbors in a coherent shared-view component; no mesh
is built or stored.
When test cameras or `--held-out-images` are supplied, that same immutable base
also goes through the ordinary Gaussian scoreboard after the diagnostic; this
allows an exact before/after control without retraining it.

When sampled visibility is enabled, `--diffuse-samples N` controls the forward
model used while fitting. Use `--score-diffuse-samples M` to spend a larger ray
budget only on final metrics and PNG dumps. It defaults to `N`, so existing
commands still evaluate exactly the renderer they trained against; setting
`N=8, M=64` is a useful final-quality split for the same-session light-stage
gate and does not retrain a different asset.

For a measured novel-view score, the dense cloud must be reconstructed from
the training photographs only. A `fused.ply` built from every frame has already
used the held cameras to select and refine geometry, even if `reconstruct`
later receives `--test-list`; that is useful for a final production asset but
is test leakage. Prepare a training-only COLMAP dense workspace before
reporting held-view or held-light numbers.

A single uncontrolled-light clip can measure capture-light re-rendering, but
it cannot establish that recovered material and illumination are physically
separate. Relighting validation needs camera-aligned repeated captures under
measured lights, as described in [Inverse rendering](INVERSE_RENDERING.md).

## Capture a relighting gate

For the first controlled experiment, stop the camera at repeatable pose marks
and photograph every pose under at least four independently controlled lights.
Do not move the camera while cycling lights. Lock the settings listed above,
record the position and RGB power of every emitter, and capture a gray card for
radiometric calibration. Include one broad diffuse frame for dense stereo and
use the same filenames in every light directory.

Reserve several camera poses and one complete light before fitting. Pass the
remaining aligned lights with paired `--normal-images` and
`--normal-environment` options, then score the excluded light with
`--held-out-images` and `--held-out-environment`. With `--dump`, its reference
and rendered images are written below `held-light/{scalar,gaussian}/`; the
`relight` and `g-relight` rows are the scalar-surface and volumetric-Gaussian
scores at the held camera/light cross-product.

When a capture or public dataset publishes its own camera split, put the exact
COLMAP image names in a text file and pass the same `--test-list` to
`train_colmap` and `reconstruct`. This takes precedence over the periodic
`--test-every` convention and prevents dataset test cameras from entering
geometry or appearance fitting.

See [Relighting capture and dataset ladder](RELIGHTING_DATASETS.md) for the
public-data starting point and the path from this aligned rig to a moving-phone
capture with independently posed `(camera, light)` observations.
