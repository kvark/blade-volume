// Wavefront compaction (alive -> queue) - simple prefix-sum based implementation
//
// This WGSL module provides two compute entry points:
//
// 1) `scan_blocks`
//    - Inputs:  g_alive[0..N) u32 flags (0 or 1)
//    - Outputs: g_prefix[0..N) u32 exclusive prefix within the whole array
//               g_block_sums[0..num_blocks) u32 number of alive flags in each block
//
// 2) `add_block_offsets_and_scatter`
//    - Inputs:  g_prefix, g_block_offsets (exclusive scan of g_block_sums), g_alive, g_state_dense
//    - Outputs: g_queue (compacted RayState), g_queue_count (total alive)
//
// Assumptions / scope:
// - This is intentionally simple and not the fastest possible scan.
// - Block size is fixed to 256 threads. One block handles 256 elements.
// - `g_block_offsets` must be provided by the caller. You can compute it on GPU
//   with another scan over `g_block_sums` (same technique, smaller N), or compute
//   on CPU for small images. For "right away WGSL", implement a second-level scan
//   using the same kernels on the `g_block_sums` buffer.
//
// Binding expectations (via blade_graphics ShaderData):
// - `g_params.count` == N (number of rays/pixels)
// - Buffers sized accordingly:
//   - g_alive: N * 4 bytes
//   - g_prefix: N * 4 bytes
//   - g_state_dense: N * sizeof(RayState)
//   - g_queue: N * sizeof(RayState)  (worst-case all alive)
//   - g_block_sums: num_blocks * 4 bytes
//   - g_block_offsets: num_blocks * 4 bytes
//   - g_queue_count: 4 bytes
//
// Notes on correctness:
// - `g_prefix[i]` is the exclusive prefix sum of alive flags.
// - If alive[i]==1, the destination index is:
//     dst = g_block_offsets[block] + g_prefix[i]
// - The total queue count is:
//     g_block_offsets[last] + g_block_sums[last]
//
// This file does not depend on RadFoam specifics beyond the RayState struct layout.
// Keep RayState in sync with the producer shader (`radfoam_init_screen.wgsl`).

const BLOCK_SIZE: u32 = 256u;

struct Params {
    count: u32,          // N elements
    num_blocks: u32,     // ceil(count / BLOCK_SIZE)
    pad0: vec2<u32>,
};

var<uniform> g_params: Params;

// Alive flags input: 0 or 1 per element
var<storage, read> g_alive: array<u32>;

// Exclusive prefix output per element
var<storage, read_write> g_prefix: array<u32>;

// Per-block sums (number of alive flags in each block)
var<storage, read_write> g_block_sums: array<u32>;

// Per-block offsets (exclusive scan of g_block_sums), provided by caller
var<storage, read> g_block_offsets: array<u32>;

// Queue count output (single u32). Written by add_block_offsets_and_scatter.
var<storage, read_write> g_queue_count: atomic<u32>;

// Dense state produced by init-screen
struct RayState {
    pixel_x: u32,
    pixel_y: u32,

    steps: u32,
    current: u32,

    t0: f32,
    transmittance: f32,

    ray_dir: vec4<f32>,
    accum_rgb: vec4<f32>,

    cells_visited: u32,
    terminated: u32,
    pad: vec2<u32>,
};

var<storage, read> g_state_dense: array<RayState>;

// Output queue (compacted)
var<storage, read_write> g_queue: array<RayState>;

// Workgroup shared memory for scan
var<workgroup> s_data: array<u32, BLOCK_SIZE>;

// Utility: inclusive scan in-place in s_data, returns inclusive result.
// We will convert to exclusive by subtracting the original value.
fn inclusive_scan_local(lid: u32) -> u32 {
    // Hillis-Steele scan: O(n log n), simple and fine for BLOCK_SIZE=256.
    // Note: This uses barriers; it is not the most optimal, but it’s robust.
    var offset: u32 = 1u;
    loop {
        if (offset >= BLOCK_SIZE) {
            break;
        }
        var t: u32 = 0u;
        if (lid >= offset) {
            t = s_data[lid - offset];
        }
        workgroupBarrier();
        s_data[lid] = s_data[lid] + t;
        workgroupBarrier();
        offset = offset * 2u;
    }
    return s_data[lid];
}

// Pass 1: scan each block, write g_prefix (exclusive) and g_block_sums
@compute @workgroup_size(256, 1, 1)
fn scan_blocks(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wid3: vec3<u32>,
) {
    let lid = lid3.x;
    let block = wid3.x;
    let i = block * BLOCK_SIZE + lid;

    // Load alive flag or 0 if out of bounds.
    var v: u32 = 0u;
    if (i < g_params.count) {
        v = g_alive[i];
    }
    s_data[lid] = v;
    workgroupBarrier();

    // Inclusive scan in shared memory.
    let incl = inclusive_scan_local(lid);

    // Exclusive prefix = inclusive - v.
    if (i < g_params.count) {
        g_prefix[i] = incl - v;
    }

    // Last lane writes block sum = inclusive sum at end of block.
    if (lid == BLOCK_SIZE - 1u) {
        g_block_sums[block] = incl;
    }
}

// Pass 2: add block offsets and scatter alive states into queue, write total count
@compute @workgroup_size(256, 1, 1)
fn add_block_offsets_and_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wid3: vec3<u32>,
) {
    let lid = lid3.x;
    let block = wid3.x;
    let i = block * BLOCK_SIZE + lid;

    if (i >= g_params.count) {
        // One invocation (0,0) still computes total count; do it below.
        // Everyone else can just exit.
        if (i != 0u) {
            return;
        }
    }

    if (i < g_params.count) {
        let alive = g_alive[i];
        if (alive == 1u) {
            let local_prefix = g_prefix[i];
            let block_off = g_block_offsets[block];
            let dst = block_off + local_prefix;
            g_queue[dst] = g_state_dense[i];
        }
    }

    // One thread computes total queue count and stores it.
    // We use an atomic for simplicity; only one writer is expected.
    if (gid.x == 0u) {
        let nb = g_params.num_blocks;
        if (nb == 0u) {
            atomicStore(&g_queue_count, 0u);
            return;
        }
        let last = nb - 1u;
        let total = g_block_offsets[last] + g_block_sums[last];
        atomicStore(&g_queue_count, total);
    }
}
