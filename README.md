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
| Synthetic (PBR unseen light) | 6 / 2 | 23.63 / 22.70 dB | 21.67 / 21.00 dB | 55.0% |
| Room | 18 / 2 | 18.74 / 18.70 dB | 12.56 / 12.29 dB | 80.9% |
| Bonsai | 18 / 2 | 18.97 / 18.83 dB | 13.04 / 12.79 dB | 99.6% |

Each PSNR cell is mean / worst held view at the 128-pixel-wide research gate.
The static columns always use the capture light. The synthetic PBR score uses
a held-out environment; Room and Bonsai have no relighting truth, so their PBR
columns measure held poses under the recovered capture light. They must not be
read as real-scene relighting accuracy.

`reconstruct --gaussian-output light-field.ply --output scene.rply` writes the
two outputs; `scene.f32` stores the recovered environment beside the PBR
scene. Current research artifacts and complete logs are generated under
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
| L | Next environment (relightable surfels) |
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
  --environment <a.f32,b.f32> Lights for a .surfel asset; without it the viewer
                           builds a sky and moves the sun around it
  --light <name|index>     Environment to open under (surfel only); L cycles
  --exposure <F>           Multiply radiance before the display curve (surfel
                           only). Without it, chosen from the environment's
                           photographic key, so any capture's units render
  --diffuse-samples <N>    Shadow rays per shading point (surfel only, default: 0)
  --specular-size <N>      Prefiltered environment width (surfel only, default: 256)
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
