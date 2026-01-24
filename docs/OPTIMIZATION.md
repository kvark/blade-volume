# Optimization Plan

This document outlines a concrete plan of attack for performance optimization, focused on **dynamic ray regrouping** (hybrid screen-ordered + wavefront execution) and **profiling-driven iteration**.

It is written to align with the current workspace:
- `blade-volume` (core GPU data, e.g. `GaussianGpuCloud`, `RadFoamGpuCloud`)
- `blade-volume-view` (the `view` binary, shaders, GPU timings UI)

The goal is to keep the implementation understandable and modular, in line with the project principles (simple code, minimal dependencies).

## Goals

1. **Get trustworthy measurements** (already partially implemented)
   - Per-pass GPU timings via `CommandEncoder::timings()` (visible in the viewer UI: “GPU Timings”).
   - Stable repro scenes and camera paths (viewer already has “Copy command line”).
2. **Reduce wasted work from divergent rays** (main new work)
   - Keep an initial phase in screen order for coherence.
   - Compact surviving rays into queues.
   - Run the expensive traversal in a wavefront stage (ideally launched via indirect dispatch).
3. **Keep complexity bounded**
   - Prefer multi-pass, explicit buffers, and clear ownership of ray state.
   - Avoid “persistent threads / single dispatch with global queues” unless proven necessary.

## Non-goals (for now)

- Micro-optimizing math or shader code without evidence of being compute-bound.
- Complex dynamic scheduling and work-stealing in one dispatch.
- Forcing one shared ray-state for Gaussian RT + RadFoam compute. (The regrouping machinery should be reusable, but payloads can be backend-specific.)

---

## Current pipeline snapshot (as of today)

### Viewer structure (`blade-volume-view`)
- **Gaussian backend**: implemented as a render pipeline driven by `shaders/gaussian.wgsl`, using hardware ray tracing through a TLAS from `GaussianGpuCloud`.
  - The viewer issues a single render pass: `"gaussian-render"`.
- **RadFoam backend**: implemented as a compute pass + blit.
  - Compute pass: `"radfoam-trace"` runs `shaders/radfoam.wgsl` (`@compute @workgroup_size(8,8,1)`) and writes HDR into `radfoam-hdr` (`rgba16float` storage texture).
  - Render pass: `"radfoam-present"` runs `shaders/radfoam_blit.wgsl` to tonemap/blit HDR to the swapchain.

### What this means for optimization
- The regrouping plan in this doc primarily targets **RadFoam** first (it is currently a single screen-ordered megakernel in `radfoam.wgsl` with a potentially divergent `while` loop per pixel).
- Gaussian RT could use wavefront scheduling too, but the current implementation is a fullscreen raster pass that relies on hardware RT inside the fragment shader; changing it is a separate track.

---

## Baseline: What to measure first (and how it maps to the repo)

Before changing algorithms, establish a baseline with repeatable timings and counters.

### 1) Per-pass GPU timings (already present)
The viewer already collects GPU timings (requires context init with timing enabled) and displays them in the UI by iterating:
- `command_encoder.timings()` in `blade-volume-view/src/bin/view.rs`

Action items:
- Ensure key passes have clear, stable names (they do today: `"radfoam-trace"`, `"radfoam-present"`, `"gaussian-render"`, `"ui"`).
- When adding new passes for regrouping, name them consistently, e.g.:
  - `"radfoam-init-screen"`
  - `"radfoam-compact"`
  - `"radfoam-wavefront"`
  - `"radfoam-wavefront-2"` (if needed)

Record in benchmarks:
- device + backend
- resolution
- `max_steps`, `weight_threshold`, `sh_degree`
- scene ID + the exact “Copy command line” output

### 2) Add cheap counters and debug visualizations
You already have a debug heatmap mode in `radfoam.wgsl` (`DEBUG_MODE_CELL_DENSITY`) that visualizes `cells_visited`.

Extend this approach with small counters (either:
- per-frame readback of a few `u32` counters, or
- write to a small 2D debug texture / overlay).

Suggested counters for RadFoam:
- rays launched (pixels)
- rays terminated by:
  - `transmittance <= weight_threshold`
  - `steps >= max_steps`
  - “no next face” (the `next_face == 0xffffffffu` path)
- average / histogram proxy for `cells_visited` (optional)

### 3) Benchmark scenes (use viewer CLI)
Use the existing viewer CLI patterns (documented in `README.md`) and the “Copy command line” button for reproduction.
Pick 3–5 scenes:
- sparse (many early terminations)
- dense (many long rays)
- mixed
- worst-case divergence (if you have one)

---

## Dynamic ray regrouping for RadFoam: Design

### Why RadFoam is a good first target
`radfoam.wgsl` today is a single screen-ordered compute shader:
- one invocation per pixel
- a `while` loop with conditions:
  - `t0 < depth`
  - `steps < max_steps`
  - `transmittance > weight_threshold`
- and an inner neighbor scan over `num_faces` (from CSR adjacency offsets)

This creates warp/wave divergence both from:
- different rays terminating at different times
- different neighbor list lengths per cell

Wavefronting can reduce wasted work by compacting “still alive” rays and processing only those.

### Concept (hybrid)
1. **Screen-ordered initial steps** (`@compute (x,y)` like today):
   - do `N0` traversal steps per pixel (small fixed number)
   - terminate obvious early-exit rays
2. **Compact surviving rays into a queue** (scan + scatter)
3. **Wavefront traversal** over the queue:
   - continue until termination, or do `N1` steps then regroup once more (optional)

### Why multi-pass first
- It matches the current architecture (explicit named passes, measured via the encoder timings UI).
- It avoids the complexity and contention of global atomic queues.
- It’s easier to validate correctness (each pass has well-defined inputs/outputs).

---

## Implementation Plan (Step-by-step, aligned to current code)

### Step 0 — Inventory RadFoam data flow (today)
Today RadFoam’s shader bindings are:
- `g_camera` (uniform)
- `g_params` (uniform: `sh_degree`, `weight_threshold`, `max_steps`, `start_point`, `debug_mode`)
- `g_points` (`array<vec4<f32>>`)
- `g_attributes` (packed `array<f32>`)
- `g_adjacency`, `g_adjacency_offsets` (CSR)
- `g_out` (HDR storage texture)

And the viewer pass sequence is:
- `"radfoam-trace"` compute → HDR
- `"radfoam-present"` render → swapchain

Document this diagrammatically here as the source of truth before you refactor.

### Step 1 — Define a RadFoam ray state layout (WGSL + Rust mirror if needed)
For RadFoam wavefronting, define a `RayState` that can fully resume traversal:

Minimum fields (from `trace_ray` in `radfoam.wgsl`):
- `pixel_xy` (u32 packed, or `u32 pixel_index`)
- `ray_origin` (vec3<f32>) — for RadFoam it’s constant = camera position, but store or reconstruct
- `ray_dir` (vec3<f32>)
- `t0` (f32)
- `transmittance` (f32)
- `accum_rgb` (vec3<f32>)
- `cells_visited` (u32) (for debug / stats)
- `current` (u32) and `current_pos` (vec3<f32>) (or reconstruct `current_pos` from `g_points[current]`)

Guidelines:
- Prefer storing `current` and recomputing `current_pos` from `g_points` to reduce state size (unless profiling proves it’s too costly).
- Keep the struct aligned for storage buffers (avoid `vec3` in structs unless you’re careful with padding).

### Step 2 — Split `radfoam.wgsl` into “step” logic
Refactor the shader logic conceptually into:
- `init_ray(gid) -> RayState`
- `trace_steps(state, step_count) -> (state, alive)`
- `write_output(state)` (write to HDR or to an intermediate buffer)

You don’t need to literally make those functions immediately, but treat them as the design decomposition.

### Step 3 — Pass A: screen-ordered initial steps
Create a new compute shader entry point (or a new shader) that:
- computes `ray_dir` the same way as today (camera model in `trace_main`)
- initializes `RayState`
- runs `N0` traversal steps (not “until done”)
- writes:
  - `alive_flags[pixel] = 1/0`
  - `ray_state_dense[pixel] = RayState` (dense per-pixel buffer)

Notes:
- Keep `start_point` logic as-is initially (the viewer sets it from the KD-tree each frame).
- Make sure debug mode behavior is defined for partial traversal:
  - either disable debug in wavefront phases initially, or define it as “final cells visited”.

### Step 4 — Pass B: compaction (scan + scatter)
Implement compaction over `alive_flags` to produce:
- `ray_queue` (contiguous `RayState`)
- `ray_count` (u32)

Implementation options:
- If you already have a scan utility somewhere in `blade_graphics`, use it.
- Otherwise implement a simple hierarchical scan in WGSL (workgroup scan + block sums + add offsets).

Keep this code isolated and reusable (queue builder).

### Step 5 — Pass C: wavefront traversal over `ray_queue`
Compute pass where each invocation processes one queued ray:
- continue traversal until termination, or for `N1` steps
- write final color to the appropriate pixel in `g_out`

If you do a bounded `N1`:
- output `alive_flags_queue[i]` and `ray_state_queue_out[i]` for a second compaction stage (Step 6)

Dispatch strategy:
- Prefer indirect dispatch sized to `ray_count` (if the graphics abstraction supports it).
- If indirect dispatch is not available/portable in your current `blade_graphics` path, fall back to:
  - dispatch at max size and early-return when `idx >= ray_count` (still works, just less optimal).

### Step 6 — Optional second regroup point (only if profiling proves it)
Add at most one more regroup cycle:
- `"radfoam-wavefront-1"` (bounded steps)
- `"radfoam-compact-2"`
- `"radfoam-wavefront-2"` (finish)

---

## Heuristics to reduce divergence further (after baseline wavefront works)

### Tile binning (recommended first)
Before (or during) compaction, bin rays by:
- screen tile ID (e.g. 16×16 tiles) to improve locality, or
- coarse depth bin (based on `t0`)

This is often cheaper than full sorting and improves memory behavior.

### Work estimation bins (optional)
Bin by a proxy of “expected remaining work”, e.g.:
- `cells_visited` so far
- `transmittance` (rays near termination may finish quickly)
- `num_faces` of current cell (from CSR offsets)

---

## Correctness and validation checklist (RadFoam-focused)

1. **Determinism / reproducibility**
   - Use the viewer “Copy command line” output for exact reproduction.
   - Keep camera and parameters fixed during comparisons.
2. **Visual diff**
   - Compare legacy `"radfoam-trace"` output vs regrouped pipeline output.
3. **Invariants**
   - `ray_count <= width*height`
   - queue writes are bounds-checked
   - termination counters sum to rays launched (accounting for multi-stage regrouping)
4. **Performance sanity**
   - Ensure compaction doesn’t dominate in dense scenes where most rays survive.
   - Ensure p95 frame time improves, not just mean.

---

## Profiling-driven iteration loop

For each change:
1. Record GPU timings (viewer UI already shows them).
2. Compare:
   - total frame time
   - `"radfoam-init-screen"`, `"radfoam-compact"`, `"radfoam-wavefront"` timings
   - alive rate after Pass A
3. Decide:
   - if compaction dominates: regroup less often; optimize scan; reduce state size
   - if traversal dominates and alive rate is high: wavefronting may not help; focus on per-step cost (neighbor scan, memory layout)
   - if memory bandwidth dominates: pack state, reduce loads, improve locality (tile binning)

---

## When to consider “single dispatch via atomics”
Only after the multi-pass approach is working and measured.

Possible triggers:
- compaction + dispatch overhead dominates even after tuning
- you need many regroup points (more than 1–2) to keep waves coherent

If you go there, prefer hierarchical queues:
- per-workgroup queues in shared memory
- batched flush to global
- minimize global contention

Treat as an advanced track.

---

## References / prior work (external)

These ideas are well-trodden in GPU rendering literature; useful starting points:

- **Radiant Foam (paper / project site)** — the method you’re implementing for the compute tracer:
  - https://radiantfoam.github.io/

- **3DGRT (Gaussian ray tracing)** — the Gaussian backend reference already cited in the repo:
  - https://gaussiantracer.github.io/

- **Wavefront / queue-based GPU path tracing (“megakernel vs wavefront”)**
  - The general technique of compacting “active” work into queues between stages is widely known as *wavefront* path tracing.
  - NVIDIA has multiple public resources and SDK samples discussing wavefront scheduling (e.g. OptiX SDK sample renderers and blog posts on wavefront path tracing). A good entry point is NVIDIA’s developer blog index:
    - https://developer.nvidia.com/blog/

- **ReSTIR / reservoir-based methods (conceptual reference for heavy reuse of compaction + indirect scheduling)**
  - While not directly applicable to RadFoam traversal, ReSTIR implementations frequently use ray queues and stream compaction patterns:
    - https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic

- **GPU stream compaction / prefix-sum primitives**
  - Modern GPU programming references for scan/compaction patterns (useful for implementing Pass B):
    - https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

Keep this section updated with any particularly relevant implementations you end up borrowing ideas from (papers, SDK samples, blog posts).
