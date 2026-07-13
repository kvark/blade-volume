mod gaussian;
mod path_record;
mod radfoam;

pub use gaussian::{GaussianGpuCloud, InitParameters};
pub use path_record::{PathRecordBuffers, PathRecorder, RecordPathsArgs};
pub use radfoam::RadFoamGpuCloud;

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
