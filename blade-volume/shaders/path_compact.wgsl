// Compact the active prefix of every fixed-capacity path row into one dense
// record stream. One invocation owns a ray, so only one device-scope atomic is
// needed per ray rather than per segment.

struct CompactParams {
    num_pixels: u32,
    max_steps: u32,
    _padding0: u32,
    _padding1: u32,
};

var<storage, read> g_path_status: array<u32>;
var<storage, read> g_cells: array<u32>;
var<storage, read_write> g_count: atomic<u32>;
var<storage, read_write> g_dense_slots: array<u32>;
var<storage, read_write> g_compact_cells: array<u32>;
var<storage, read_write> g_pixel_indices: array<u32>;
var<storage, read_write> g_active: array<f32>;
var<uniform> g_params: CompactParams;

@compute @workgroup_size(64)
fn compact_paths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.x;
    if (pixel >= g_params.num_pixels) {
        return;
    }

    let steps = g_path_status[pixel] & 0x7fffffffu;
    let output_start = atomicAdd(&g_count, steps);
    let input_start = pixel * g_params.max_steps;
    for (var step = 0u; step < steps; step += 1u) {
        let input = input_start + step;
        let output = output_start + step;
        g_dense_slots[output] = input;
        g_compact_cells[output] = g_cells[input];
        g_pixel_indices[output] = pixel;
        g_active[output] = 1.0;
    }
}
