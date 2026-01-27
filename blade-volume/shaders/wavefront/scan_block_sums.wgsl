// Wavefront compaction helper: scan block sums -> block offsets (GPU-side)
//
// This is the "second level" scan intended to be used with `compact_alive.wgsl`.
// Given `g_block_sums[0..num_blocks)` (each entry is the number of alive rays
// in one 256-element block), compute:
//
// - `g_block_offsets[0..num_blocks)` = exclusive prefix sum of g_block_sums
//
// That is:
//   g_block_offsets[0] = 0
//   g_block_offsets[i] = sum_{k=0..i-1} g_block_sums[k]
//
// This enables fully GPU-driven compaction:
//   1) scan_blocks (per-pixel flags) -> g_prefix + g_block_sums
//   2) scan_block_sums (this file)   -> g_block_offsets
//   3) add_block_offsets_and_scatter -> g_queue + g_queue_count
//
// Constraints / assumptions:
// - This implementation is intentionally simple. It handles up to MAX_BLOCKS blocks
//   in a single workgroup. If num_blocks exceeds MAX_BLOCKS, it will still produce
//   correct offsets for the first MAX_BLOCKS entries, but the remaining entries will
//   be left untouched (caller must enforce bounds).
// - For typical render targets, num_blocks = ceil((W*H)/256) is usually well within 8192.
//   If you need larger, either increase MAX_BLOCKS (watch workgroup memory limits)
//   or implement a hierarchical multi-dispatch scan.
//
// WGSL / uniform layout:
// - Uses 16-byte aligned uniform Params.
// - All buffers are u32 arrays.
//
// Binding expectations (via blade_graphics ShaderData):
// - g_params.num_blocks == number of blocks in g_block_sums/offsets
// - g_block_sums length >= num_blocks
// - g_block_offsets length >= num_blocks

const MAX_BLOCKS: u32 = 8192u;

struct Params {
    num_blocks: u32,
    pad0: vec3<u32>,
};

var<uniform> g_params: Params;

var<storage, read> g_block_sums: array<u32>;
var<storage, read_write> g_block_offsets: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn scan_block_sums(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    let n = min(g_params.num_blocks, MAX_BLOCKS);

    // Single-threaded exclusive scan to avoid workgroup array layout issues.
    var sum: u32 = 0u;
    for (var i = 0u; i < n; i += 1u) {
        let v = g_block_sums[i];
        g_block_offsets[i] = sum;
        sum += v;
    }

    // If num_blocks > MAX_BLOCKS, we intentionally do not write the remaining offsets.
    // The caller must either:
    // - increase MAX_BLOCKS, or
    // - perform a multi-level scan, or
    // - clamp workload to supported sizes.
}
