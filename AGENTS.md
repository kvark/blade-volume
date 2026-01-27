This is a Rust+WGSL library that directly works with volumetric data.

# Principles

- low dependencies, only Rust
- simple code, don't overcomplicate and assume future cases, assume users know what they are doing
- strict style:
  - single `use` per crate, prefer to import modules instead of individual items
  - either publicly export a module, or some of its types, but not both
  - no implicit references in `match`, prefer explicit `ref` instead

# Development Tracks

## Productionization

Modularize the shaders:
  - move the common shader code into a separate WGSL in the examples
  - move backend-specific code to the new `shaders` folder, target embedding into other apps
  - make shared shaders between backends, e.g. for spherical harmonics evaluation

Add an API that allows the user to create backend-specific data (BLAS) from triangular meshes.

Implement the distinction between BLAS and TLAS, allow the user to control the transformation of objects every frame. Basically, allowing them to:
```rust
let object = scene.create_object("point_cloud.ply");
for frame in frames {
  scene.set_transform(object, position, rotation, scale);
  scene.render(target_view);
}
```

## Optimization

Gather more ideas?

Dynamic wave regrouping:
- do the first few steps (applicable to both radfoam and ray-traced gaussians) from the screen
- re-pack the survived rays into new ways, continue for another number of steps
- straightforward way would be to re-pack using a different compute invocation, followed by indirect dispatch for the main traversal
- perhaps this can be implemented to work entirely within a single dispatch by using atomics?

## Extension

New rendering methods:
- SDF based on compute
- Gaussians splatted with compute (instead of the current ray tracing), in 3DGUT formulation

Implement a way to build BLAS by the means of reconstruction from a sequence of images (and masks).

## Testing

Image reftests for all methods.
CI based on Vulkan lavapipe.
It would help to generate some of the models programmatically, so that we don't commit big assets into the repo. Could be some simple-ish geometric primitives, like a christmas tree.
