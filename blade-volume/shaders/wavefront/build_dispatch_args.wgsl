// Build indirect dispatch arguments for wavefront phases.
//
// Reads the current queue_count and writes dispatch groups for @workgroup_size(256,1,1).

struct DispatchArgs {
    x: u32,
    y: u32,
    z: u32,
    pad0: u32,
};

var<storage, read> g_queue_count: atomic<u32>;
var<storage, read_write> g_dispatch: DispatchArgs;

@compute @workgroup_size(1, 1, 1)
fn build_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    let count = atomicLoad(&g_queue_count);
    let groups = (count + 255u) / 256u;
    g_dispatch.x = groups;
    g_dispatch.y = 1u;
    g_dispatch.z = 1u;
    g_dispatch.pad0 = 0u;
}
