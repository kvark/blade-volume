# Pipeline Plan: phone → train → view

Status: in-progress. Living document — keep it short, edit as we learn.

## North star

```
phone video ──▶ COLMAP ──▶ blade-volume-train ──▶ PointCloudModel ──▶ blade-volume-view
                              (meganeura)
```

The library boundary is `PointCloudModel` (or a PowerFoam-extended version of it). The
viewer doesn't know how the model was produced; the trainer doesn't know how it gets
rendered. Both link `blade-volume` for shared types and shaders.

## Non-goals

- No Python, no PyTorch, no Burn.
- No new representations beyond what we render. The trainer optimises one of:
  Gaussian (current), RadFoam (current), or PowerFoam (M2). We will not invent a new one.
- No on-device capture code initially — phone capture stays manual (record + transfer).
  We'll automate that later if and only if the rest of the pipeline is solid.

## Milestones

Each milestone is sized to fit in a handful of focused sessions and lands behind passing
`cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace`.

### M1 — Shader modularisation (foundation)

Today the test harness re-implements `// #include` for WGSL (`radfoam_gpu_vs_cpu.rs`).
Promote it to the library so M2 and M3 don't have to duplicate it.

- Move the `preprocess_shader` helper into `blade-volume/src/shaders.rs`.
- Expose a `compose(entry)` returning a fully-expanded WGSL string.
- Have `gpu/{gaussian,radfoam}.rs` and `blade-volume-view` use it.
- Tests: keep the existing GPU-vs-CPU test working unchanged.

Risk: low. Pure refactor.

### M2 — PowerFoam adoption

Power Foam = weighted Voronoi (power diagram) instead of plain Voronoi. Per-site weight,
radical-plane faces, Čech-complex adjacency. The PowerFoam paper bundles a much richer
appearance model (per-cell quaternion + texel sites + spherical-Voronoi colours); we
defer all of that and reuse our SH path. Adoption order:

#### M2a — `radius` in `PointCloudModel`

- Add `pub radii: Option<Vec<f32>>` to `PointCloudModel`.
- Round-trip through the RadFoam PLY reader/writer: optional `radius` float property.
- No shader changes yet — render code ignores the radii.
- Tests: extend the PLY round-trip test with a radii-present case.

#### M2b — Čech-complex adjacency builder (done)

- `adjacency::compute_cech(points, radii, config)` emits an edge `{i,j}` when
  `|p_i - p_j| ≤ r_i + r_j`; returns the existing CSR `Adjacency`.
- Candidates are pruned with a `kiddo` k-d tree range query (radius
  `r_i + r_max`), then the exact overlap predicate filters them.
- `PointCloudModel::compute_adjacency_default` dispatches: Čech when
  `radii.is_some()`, Delaunay otherwise.
- `kiddo` promoted from dev-dep to runtime dep on `blade-volume`.

#### M2c — PowerFoam WGSL

- Fork `blade-volume/shaders/radfoam.wgsl` → `powerfoam.wgsl`.
- Swap the bisector face derivation for the radical plane:
  - `face_normal = p_j - p_i`
  - `face_origin = midpoint + (r_i² − r_j²) / (2 |p_j − p_i|²) · (p_j − p_i)`
- Add a `radii` storage buffer in the bind group.
- New backend `gpu/powerfoam.rs` mirroring `gpu/radfoam.rs`.
- Viewer auto-selects PowerFoam when the loaded model has `radii.is_some()`.

#### M2d — Cross-check

- Port one PowerFoam checkpoint scene (via a one-off Python export script in `etc/`)
  to our radii-extended PLY.
- Image-diff against PowerFoam's own renderer at a fixed pose. Wide tolerance is fine;
  we just want the same scene to be recognisable.

### M3 — Training crate scaffolding

New crate: `blade-volume-train`. Depends on `blade-volume` + `meganeura`. Never the
reverse.

#### M3a — Crate skeleton

- `cargo new --lib blade-volume-train`, wire into workspace.
- Define the trainer state: parameters mirror `PointCloudModel` fields but live as
  meganeura tensors with gradients.
- Conversion: `TrainerState ⇆ PointCloudModel`.

#### M3b — COLMAP loader

- Read `sparse/0/{cameras,images,points3D}.bin`. Pure Rust, no native deps.
- Camera intrinsics + extrinsics → our `CameraParams` type.
- Initial sparse points → starting `PointCloudModel`.
- Tests: parse a small COLMAP dump we keep under `etc/` (git-LFS or generated in a
  doc-test).

#### M3c — Differentiable forward + loss

- The renderer is currently GPU compute. We need either:
  (a) a CPU reference forward that's differentiable through meganeura, or
  (b) a GPU forward + custom backward.
- Start with (a) at low resolution to debug the math, then move to (b) for speed.
- Loss: L1 + SSIM over rendered vs. ground-truth images.
- Optimiser: Adam from meganeura.

#### M3d — Online viewer attach

- During training, periodically convert `TrainerState → PointCloudModel` and hand it
  to a running `blade-volume-view` instance via a shared `Arc<Mutex<...>>`.
- This mirrors PowerFoam's `--viewer` flag.

### M4 — Capture stage

Only after M3 produces something worth looking at.

- A short doc page: which phone apps work, what frame rate / exposure settings.
- `etc/colmap.sh` wrapper: video → frames → COLMAP sparse reconstruction.

## Sequencing recommendation

```
M1 ──▶ M2a ──▶ M2b ──▶ M2c ──▶ M2d
         │
         └─▶ M3a ──▶ M3b ──▶ M3c ──▶ M3d ──▶ M4
```

`M3a..M3d` can run in parallel with `M2b..M2d` because M3 only needs the
`PointCloudModel` shape from M2a, not the new renderer.

## Decisions parking lot

- Čech-complex builder: roll our own AABB tree, or pull `rstar`? Decide at M2b.
- meganeura API: tensors-of-vec3 or flat tensor-of-f32? Decide at M3a after reading
  meganeura's actual surface.
- Loss-domain (linear vs. sRGB) and tonemap: punt to M3c.

## Out-of-scope (for now)

- Texel/spherical-Voronoi appearance model from PowerFoam (M2 keeps SH).
- Mobile capture app.
- Multi-GPU / distributed training.
- LOD or streaming for huge scenes.
