# blade-volume-convert

Mesh-to-volume conversion for `blade-volume`.

This crate ingests triangle meshes (glTF) and produces `blade_volume::PointCloudModel`
for either the Gaussian or RadFoam backend. The initial model uses Lambert shading with
ambient baked into the constant SH term (degree 0).

## Example

```rust
use blade_volume_convert::{convert_gltf, save_ply, ConvertOptions, OutputKind};

let mut options = ConvertOptions::default();
options.output = OutputKind::Gaussian;
options.density = 12.0;

let model = convert_gltf("scene.glb", &options)?;
save_ply("scene.ply", &model)?;
println!("points: {}", model.len());
# Ok::<(), blade_volume_convert::ConvertError>(())
```

Or from the binary (saves PLY by default):

```bash
cargo run -p blade-volume-convert -- scene.glb -k gaussian -o scene.ply -f binary
```

## Notes

- Alpha is used as a discard threshold; samples below `alpha_threshold` are dropped.
- Surface samples come from mesh triangles, interior samples are a coarse voxel fill.
- Output uses SH degree 0; higher-degree SH will be added later.

## Models

For local testing, the Kenney Car Kit is useful and available at:
`/x/Assets/Kits/kenney_car-kit/Models/GLTF format/`

This repo does not bundle the assets. See https://kenney.nl/assets/car-kit
