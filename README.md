# blade-volume

Volumetric rendering methods based on Blade graphics.

## Gaussian Blobs

Implementing [3DGRT paper](https://gaussiantracer.github.io/) with hardware ray tracing.

![koala](/etc/gs-koala.jpg)

### Benchmark

Invocation:
```bash
cargo run --example view -- c:\Work\GaussianSplats\koala.ply --resolution 800,800 --cam-pose -2.6,-1.7,-0.8,0,73,-17
```

## Radiant Foam

Implementing [Radiant Foam paper](https://radiantfoam.github.io/) with pure compute.

![bike](/etc/rf-bike.jpg)

Invocation:
```bash
cargo run --example view_radfoam -- <path_to_bicycle.ply>
```
