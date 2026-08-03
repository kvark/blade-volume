// Shared bindings and data structures for scene traversal backends.

// Object type constants must match Rust ObjectType.
const OBJECT_TYPE_GAUSSIAN: u32 = 0u;
const OBJECT_TYPE_RADFOAM: u32 = 1u;

struct ObjectBounds {
    center: vec3<f32>,
    radius: f32,
    object_type: u32,
    data_index: u32,
    sh_degree: u32,
    attribute_stride: u32,
    flags: u32,
    point_count: u32,
    start_point: u32,
    pad: u32,
}

struct GpuTransform {
    object_to_world: mat4x4<f32>,
    world_to_object: mat4x4<f32>,
}

struct SceneParams {
    object_count: u32,
    weight_threshold: f32,
    max_steps: u32,
    debug_mode: u32,
}

var<uniform> g_camera: Camera;
var<uniform> g_scene_params: SceneParams;
var<storage, read> g_bounds: array<ObjectBounds>;
var<storage, read> g_transforms: array<GpuTransform>;

struct RadFoamPointsBuffer {
    data: array<vec4<f32>>,
}
struct RadFoamAttributesBuffer {
    data: array<f32>,
}
struct RadFoamSurfaceNormalsBuffer {
    data: array<vec4<f32>>,
}
struct RadFoamAdjacencyBuffer {
    data: array<u32>,
}
struct RadFoamAdjacencyOffsetsBuffer {
    data: array<u32>,
}

var<storage, read> g_radfoam_points: binding_array<RadFoamPointsBuffer>;
var<storage, read> g_radfoam_surface_normals: binding_array<RadFoamSurfaceNormalsBuffer>;
var<storage, read> g_radfoam_attributes: binding_array<RadFoamAttributesBuffer>;
var<storage, read> g_radfoam_adjacency: binding_array<RadFoamAdjacencyBuffer>;
var<storage, read> g_radfoam_adjacency_offsets: binding_array<RadFoamAdjacencyOffsetsBuffer>;
