# blade-volume

Point-cloud-native volumetric rendering methods based on Blade graphics. The
runtime scene contains clouds only: Gaussian, RadFoam, PowerFoam, and future
point-sampled representations. Triangle meshes are accepted solely as offline
conversion input, never as a runtime geometry fallback.

The longer-term goal is a phone-video → reconstruction → interactive viewer pipeline.
Training lives in a separate crate built on [meganeura](https://github.com/kvark/meganeura);
no Python, no Burn. See `docs/AUDIT_AND_ROADMAP.md` for the audited status and
stage gates.

## Workspace Structure

This repository is organized as a Cargo workspace:

```
blade-volume/          # Core library (no windowing dependencies)
blade-volume-view/     # Viewer utilities with winit (camera, input)
blade-volume-convert/  # glTF → point-cloud sampling
blade-volume-test/     # Image-reference regression harness
blade-volume-train/    # Meganeura-backed appearance training (COLMAP → foam)
```

## Training a foam from a COLMAP scene

```bash
etc/fetch_test_dataset.sh bonsai                  # ~280 MB into etc/data/bonsai/
cargo run --release -p blade-volume-train --bin train_colmap -- \
    --sparse etc/data/bonsai/sparse/0 \
    --images etc/data/bonsai/images \
    --output  bonsai.ply \
    --novel-strip-prefix novel \
    --width 24 --height 24 --views 8 --epochs 200 \
    --max-steps 24 --max-points 2000 --learning-rate 0.05
```

Outputs a binary RadFoam PLY plus a 5-frame interpolated-camera strip
(`novel_00.png` … `novel_04.png`). The PLY can be opened with the viewer
above. Add `--masks masks/` for a foreground directory mirroring the image
paths; masked runs supervise opacity and default its loss weight to 1. See
`docs/PIPELINE.md` for the design and `docs/MESH_TO_FOAM.md`
for the parallel mesh-to-foam path.

## Current Reconstruction Results

The image-only pipeline now produces two cloud-only outputs from posed views:
a static anisotropic Gaussian light field, and a Gaussian surface with shared
PBR materials plus a recovered environment. No polygonal geometry enters
either reconstructed asset.

| Gate | Training / held views | Static held PSNR | PBR held PSNR | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Synthetic (PBR unseen light) | 6 / 2 | 25.06 / 24.34 dB | 18.90 / 18.68 dB | 56.4% |
| Synthetic (predefined light, refined) | 6 / 2 | 25.15 / 24.44 dB | 21.81 / 21.48 dB | 57.0% |
| Synthetic (four calibrated lights, five-cloud average) | 6 / 2 | 24.97 / 24.22 dB | 22.68 / 22.22 dB | 56.7% |
| Synthetic (full Gaussian PBR geometry, five-cloud average) | 6 / 2 | 24.97 / 24.22 dB | 23.44 / 22.90 dB | 55.3% |
| Room | 18 / 2 | 18.74 / 18.70 dB | 12.56 / 12.29 dB | 80.9% |
| Bonsai | 18 / 2 | 18.97 / 18.83 dB | 13.04 / 12.79 dB | 99.6% |

Each PSNR cell is mean / worst held view at the 128-pixel-wide research gate.
The static columns always use the capture light. The synthetic PBR score uses
a held-out environment; Room and Bonsai have no relighting truth, so their PBR
columns measure held poses under the recovered capture light. They must not be
read as real-scene relighting accuracy.

The predefined-light row is the fixed-cloud result after learned-density
normal initialization and the conservative complete-render normal, radius,
and material passes. Across five independently trained source clouds, the
final unseen-light PBR score averages 21.84/21.49 dB mean/worst at 56.7%
coverage. The density signal alone adds 0.11/0.11 dB over the otherwise
identical post-support pipeline and improves nearest-truth normal RMSE on all
five clouds. This is a controlled lighting milestone rather than an end-to-end
unknown-light result.

With four aligned captures under measured lights, the same five-cloud gate
improves unseen-light PBR quality to 22.52/21.95 dB. Photometric normals are
useful for the PBR surface but not for reproducing the original photographs,
so the static light field now keeps the pre-calibration Gaussian geometry and
fits its appearance and support independently. That restores static quality
from 23.62/22.95 to 24.97/24.22 dB while changing PBR by only
+0.01/-0.01 dB on average. A conservative 20% density-gradient correction
after the calibrated normal solve improves every one of the five held-light
means and tails; single-light and static outputs retain their gated 10%
correction. Both outputs remain point clouds.

The PBR support fit uses masks only as negative visibility evidence: predicted
opacity is penalized on known background rays, but a foreground mask does not
force one of several overlapping particles to own that ray. The static light
field still receives full foreground/background mask supervision. This moves
the calibrated five-cloud PBR average from 22.52/21.95 to 22.68/22.22 dB
mean/worst, with 56.7% rather than 57.0% coverage; reduced Room and Bonsai
smokes remain non-regressive.

Retaining the learned Gaussian covariance and opacity for PBR rendering
improves all five calibrated clouds. After the selected response and compositing
corrections, the mean held-light score gains 0.60 dB, the worst view gains
0.49 dB, and covered-pixel quality gains 0.68 dB over the scalar surface.
Coverage remains 1.7 percentage points lower. Bounding each particle where its alpha response
falls below 0.03 removes weak overlapping tails, improving the fixed-cloud
volumetric score by another 0.10/0.11 dB while reducing its median render time
from 8.86 ms to 1.49 ms per 100x75 frame. A mild remap of only the retained
core then improves every five-cloud mean and tail and recovers 0.4 coverage
point without widening the acceleration proxies. Grouping overlapping hits
from the same thin depth layer into a partially saturated surface sheet adds
another 0.8 coverage point while improving every synthetic mean and tail. A
final 2.5% log-space residual of the already selected scalar radius refinement
then improves all five means and tails without changing learned ellipsoid
orientation or aspect ratio. The scalar surface remains faster at about 0.7
ms, but it is no longer the only practical interactive path. Full 18/2-view
Room and Bonsai gates put Gaussian and scalar rendering at 12.52/12.04 versus
12.49/12.01 dB on Room, while Gaussian wins 14.86/14.83 versus 14.69/14.40 dB
on Bonsai. Real captures have no held-light truth, so these remain
capture-light novel-view gates rather than relighting accuracy claims.

When two or more aligned captures have measured environments, the final PBR
Gaussian geometry also receives a short position-only continuation under each
known light. One optimizer interleaves paired rays from every light while
swapping fixed diffuse appearance predicted from the recovered material and
normal, then restores the pre-continuation coefficients; covariance, opacity,
materials, and runtime representation are unchanged. A weak foreground
residual conditions at most 50% of the color loss in proportion to detached
predicted opacity. This prevents well-covered center motion from repairing
errors in frozen transmittance while poorly covered, background, and maskless
rays retain the ordinary coverage-driving residual. The four-light joint fit
uses linear-radiance residuals when every selected view has a mask, while
maskless or mixed captures retain the display-referred residual needed for
low-radiance coverage. It reaches 23.44/22.90 dB at 55.3% coverage across the
five fixed clouds. It is automatic only for a requested relightable Gaussian
output and adds no training option or shader.

`reconstruct --gaussian-output light-field.ply --pbr-gaussian-output relightable.ply`
writes the two durable cloud outputs. `relightable.f32` stores the recovered
environment beside the PBR Gaussian. The writer reloads that PLY before its
final score, so reported quality includes serialization. Current research artifacts and complete logs are generated under
`target/audit-runs/` and intentionally remain outside version control. The
exact protocols, negative results, and artifact locations are recorded in
[`docs/GAUSSIAN_RECONSTRUCTION_PLAN.md`](docs/GAUSSIAN_RECONSTRUCTION_PLAN.md).

CI enforces workspace formatting, all-feature clippy with warnings denied,
default and all-feature tests, and a RustSec dependency audit.
The clippy policy mirrors `blade-graphics/src/lib.rs` and lives in the root `Cargo.toml`'s
`[workspace.lints]` block — keep both in sync.

## Unified Viewer

The `view` binary in `blade-volume-view` supports multiple rendering backends with shared camera controls.
The format is **automatically detected** by examining the PLY file header:

```bash
# Auto-detection (works for both Gaussian and RadFoam PLY files)
cargo run -p blade-volume-view -- <path_to_file.ply>
cargo run -p blade-volume-view -- <path_to_file.spz>

# Override auto-detection with --kind
cargo run -p blade-volume-view -- <path_to_file.ply> --kind=radfoam
cargo run -p blade-volume-view -- <path_to_file.ply> --kind=gaussian
```

### Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move forward/left/back/right |
| Z/X | Move down/up |
| Q/E | Roll camera |
| Mouse drag | Look around |
| Mouse wheel | Adjust fly speed |
| I | Print info (camera pose, GPU timings) |
| Tab | Toggle debug mode (particle density visualization) |
| L | Next environment (relightable point clouds) |
| Escape | Exit |

### Options

```
  --resolution <W,H>       Target resolution (e.g. 1920,1080)
  --cam-pose <x,y,z,r,p,y> Camera position and orientation (Euler degrees)
  --kind <gaussian|radfoam|surfel> Override format auto-detection
  --max-steps <N>          Max traversal steps (RadFoam only, default: 1024)
  --weight-threshold <F>   Stop when transmittance <= threshold (RadFoam only, default: 0.001)
  --min-opacity <F>        Minimum opacity for Gaussian rendering (default: 0.01)
  --min-transmittance <F>  Minimum transmittance for Gaussian rendering (default: 0.01)
  --environment <a.f32,b.f32> Lights for a relightable asset; without it the viewer
                           builds a sky and moves the sun around it
  --light <name|index>     Environment to open under (relightable only); L cycles
  --exposure <F>           Multiply radiance before the display curve. Without
                           it, the value is chosen from the environment's
                           photographic key, so any capture's units render
  --diffuse-samples <N>    Shadow rays per shading point (surfel only, default: 0)
  --specular-size <N>      Prefiltered environment width (relightable, default: 256)
  --debug                  Start in debug mode (particle density visualization)
```

## Relightable Surface Particles

The other representations store what a point looked like: the light that was
there when it was captured is already inside the number, and cannot be taken
back out. This one stores what the surface is made of — albedo, specular
reflectance, roughness, and an exact normal — and works out the radiance at
render time from whatever environment it is handed.

Convert once, then light it as often as you like:

```bash
cargo run --release -p blade-volume-convert -- model.glb --kind surfel --resolution 400
cargo run --release -p blade-volume-view -- model.surfel
```

The viewer opens framed on the asset with a procedural sky; `L` moves the sun,
and the model is not rebuilt between lights. `--environment` takes measured
environments instead, as the float planes blade's `relight_data` writes, and
`--light` picks which to open under. Exposure comes from the environment's own
key luminance unless you set it, because radiance arrives in whatever units
the capture used.

To reconstruct Gaussian particles from selected cameras of Blade's synthetic
relighting fixture and score both unseen poses and unseen illumination:

```bash
cargo run --release -p blade-volume-train --bin synthetic_reconstruct -- \
  --dataset /path/to/relight-data --output target/reconstruction.rply
```

This command is deliberately a depth upper bound: it fuses depth truth from
training views only, estimates normals from the resulting cloud, and fits
materials from radiance. It prints a matched truth-material control and never
uses held-out camera geometry for fusion. `--truth-normals` selects the earlier
normal upper bound. The image-only path and its current gap to this ceiling are
summarized above.

Scored against blade's canonical path tracer over six views of `police.glb`
under five environments, the direct-lighting path reaches **27.95 dB linear /
23.78 dB tone mapped at 0.7 ms a frame** (320x240, 235k surfels). Shadow rays
are available and are *not* an improvement: they buy visibility and one bounce
at seven times the cost, and against a four-bounce reference they score worse
than leaving both out. See `benchmarks/mesh_conversion.toml` for the numbers
and what they do not cover.

## Gaussian Blobs

Implementing [3DGRT paper](https://gaussiantracer.github.io/) with hardware ray tracing.

![koala](/etc/gs-koala.jpg)

### Example

```bash
cargo run -p blade-volume-view -- /path/to/koala.ply --resolution 800,800 --cam-pose -2.6,-1.7,-0.8,0,73,-17
```

Some assets can be found in [GSOP](https://github.com/cgnomads/GSOPs/tree/91e1c34a92f2334a85a3545152d905c5403ee0e0/hip/splats/cleaned).

## Radiant Foam

Implementing the [Radiant Foam paper](https://radfoam.github.io/) with pure compute.

![bike](/etc/rf-bike.jpg)

### Example

```bash
cargo run -p blade-volume-view -- "/path/to/Bicycle.ply" --resolution 1200,900 --cam-pose -1.278,0.002,1.267,-0.0,-57.4,-146.3 --max-steps 1024 --weight-threshold 0.001
```

## Debug Mode

Press `Tab` to toggle debug visualization mode, which shows a heatmap of:
- **Gaussian backend**: Number of particles hit per pixel
- **RadFoam backend**: Number of Voronoi cells traversed per pixel

The color scale goes from blue (few) → cyan → green → yellow → red (many).
