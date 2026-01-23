Tasks:
1. Make the `view` example able to run different rendering methods, while sharing the input and camera logic.
2. Try to integrate radfoam specific point clouds into the general point cloud format.
3. Move the common shader code into a separate WGSL in the examples, while backend-specific code would be moved to `shaders` folder and be made for embedding into other apps. There would also be shared shaders between backends, e.g. for spherical harmonics evaluation.
4. Add SDF rendering method based on compute.
5. Add an API that allows the user to create backend-specific data (BLAS) from triangular meshes.
6. Implement the distinction between BLAS and TLAS, allow the user to control the transformation of objects every frame.
7. Implement a way to build BLAS by the means of reconstruction from a sequence of images (and masks).
