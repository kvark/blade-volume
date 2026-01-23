# blade-volume

Volumetric rendering methods based on Blade graphics.

## Workspace Structure

This repository is organized as a Cargo workspace:

```
blade-volume/          # Core library (no windowing dependencies)
blade-volume-view/     # Viewer utilities with winit (camera, input)
```

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
| Escape | Exit |

### Options

```
  --resolution <W,H>       Target resolution (e.g. 1920,1080)
  --cam-pose <x,y,z,r,p,y> Camera position and orientation (Euler degrees)
  --kind <gaussian|radfoam> Override format auto-detection
  --max-steps <N>          Max traversal steps (RadFoam only, default: 1024)
  --weight-threshold <F>   Stop when transmittance <= threshold (RadFoam only, default: 0.001)
  --min-opacity <F>        Minimum opacity for Gaussian rendering (default: 0.01)
  --min-transmittance <F>  Minimum transmittance for Gaussian rendering (default: 0.01)
  --debug                  Start in debug mode (particle density visualization)
```

## Gaussian Blobs

Implementing [3DGRT paper](https://gaussiantracer.github.io/) with hardware ray tracing.

![koala](/etc/gs-koala.jpg)

### Example

```bash
cargo run -p blade-volume-view -- /path/to/koala.ply --resolution 800,800 --cam-pose -2.6,-1.7,-0.8,0,73,-17
```

## Radiant Foam

Implementing [Radiant Foam paper](https://radiantfoam.github.io/) with pure compute.

![bike](/etc/rf-bike.jpg)

### Example

```bash
cargo run -p blade-volume-view -- /path/to/bicycle.ply --radfoam
```

## Debug Mode

Press `Tab` to toggle debug visualization mode, which shows a heatmap of:
- **Gaussian backend**: Number of particles hit per pixel
- **RadFoam backend**: Number of Voronoi cells traversed per pixel

The color scale goes from blue (few) → cyan → green → yellow → red (many).