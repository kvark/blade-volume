// Copy compacted queue count into a uniform-like buffer layout.
//
// Purpose
// -------
// After compaction, the queue count is written into a storage buffer (u32).
// Some passes prefer to read it as a uniform struct (e.g. `WavefrontParams`).
//
// This compute shader copies:
//   src[0] -> dst.queue_count
// and optionally clears remaining padding.
//
// Intended bindings (via ShaderData)
// ---------------------------------
// Uniform:
//   g_params: Params { clear_padding: u32, ... }
//
// Storage (read):
//   g_src_count: array<u32>        // at least 1 element
//
// Storage (read_write):
//   g_dst: WavefrontParams         // uniform-like layout stored in a buffer
//
// Notes
// -----
// - WGSL does not allow writing to true uniforms. This writes to a storage buffer
//   containing the same struct layout as the uniform you want to bind later.
// - Keep `WavefrontParams` in sync with `radfoam_wavefront.wgsl` and Rust.
//
// Dispatch
// --------
// Dispatch 1 workgroup of 1 thread.

struct Params {
    // If non-zero, pad fields in `WavefrontParams` are set to 0.
    clear_padding: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct WavefrontParams {
    queue_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

var<uniform> g_params: Params;

var<storage, read> g_src_count: array<u32>;
var<storage, read_write> g_dst: WavefrontParams;

@compute @workgroup_size(1, 1, 1)
fn copy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    g_dst.queue_count = g_src_count[0];

    if (g_params.clear_padding != 0u) {
        g_dst.pad0 = 0u;
        g_dst.pad1 = 0u;
        g_dst.pad2 = 0u;
    }
}
