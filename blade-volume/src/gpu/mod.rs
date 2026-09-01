mod gaussian;
mod mesh_reference;
mod path_compact;
mod path_record;
mod powerfoam_depth;
mod powerfoam_splat;
mod radfoam;
mod radfoam_depth;
mod radfoam_trace;
mod relight;
mod sphere_bvh;

pub use gaussian::{GaussianGpuCloud, InitParameters};
pub use mesh_reference::{MeshReferenceSettings, MeshReferenceTracer, ReferenceMesh};
pub use path_compact::{PathCompactBuffers, PathCompactor};
pub use path_record::{
    PathJacobianMode, PathRecordBuffers, PathRecordStats, PathRecorder, RecordPathsArgs,
};
pub use powerfoam_depth::PowerFoamGpuDepthTracer;
pub use powerfoam_splat::PowerFoamGpuSplatTracer;
pub use radfoam::RadFoamGpuCloud;
pub use radfoam_depth::{RadFoamDepthSettings, RadFoamGpuDepthTracer};
pub use radfoam_trace::{RadFoamGpuTracer, RadFoamTraceSettings};
pub use relight::{RelightSettings, RelightTracer};

/// Whether this process is forbidden from initializing a GPU context.
///
/// The cgroup runner sets this in `--cpu-only` scopes in addition to hiding
/// Vulkan ICDs and applying the device cgroup policy. Checking before the
/// Vulkan loader runs is important on a host whose vendor driver is already
/// wedged: a supposedly skippable hardware test can otherwise become
/// unkillable while probing the driver.
pub fn access_disabled() -> bool {
    std::env::var_os("BLADE_VOLUME_DISABLE_GPU").is_some()
}
