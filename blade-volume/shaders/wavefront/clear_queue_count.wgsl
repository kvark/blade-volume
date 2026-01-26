// Clear a queue count atomic to zero.

var<storage, read_write> g_count: atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    atomicStore(&g_count, 0u);
}
