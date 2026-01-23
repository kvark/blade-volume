//! GPU-compatible camera parameters for shaders.

/// GPU-compatible camera parameters for uniform binding.
///
/// This struct is designed to be used directly in shader uniforms.
/// For interactive camera control, see `blade-volume-view::ControlledCamera`.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CameraParams {
    pub cam_position: [f32; 3],
    pub depth: f32,
    pub cam_orientation: [f32; 4],
    pub fov: [f32; 2],
    pub pad: [u32; 2],
}
