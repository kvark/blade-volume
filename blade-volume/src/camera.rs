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
    /// Optical principal point in normalized device coordinates. `[0, 0]`
    /// is the image center; COLMAP `(cx, cy)` maps to
    /// `[2*cx/width - 1, 2*cy/height - 1]`.
    pub principal: [f32; 2],
}

/// The orientation of a camera looking along `forward`, the right way up.
///
/// The camera's local frame here is right, **down**, forward: the first image
/// row looks along local `-Y`, so the axis that has to oppose world up is `Y`.
/// Getting this backwards costs nothing at the time and produces an
/// upside-down image that is otherwise entirely convincing, which is why it
/// lives next to the parameters it belongs to rather than in each caller.
pub fn orientation_looking(forward: glam::Vec3, world_up: glam::Vec3) -> glam::Quat {
    let forward = forward.normalize_or_zero();
    if forward == glam::Vec3::ZERO {
        return glam::Quat::IDENTITY;
    }
    // A world up parallel to the view has no projection to use as a reference.
    let reference = if forward.dot(world_up).abs() > 0.999 {
        if world_up.z.abs() < 0.9 {
            glam::Vec3::Z
        } else {
            glam::Vec3::X
        }
    } else {
        world_up
    };
    let down = (forward * forward.dot(reference) - reference).normalize();
    let right = down.cross(forward).normalize();
    glam::Quat::from_mat3(&glam::Mat3::from_cols(right, down, forward)).normalize()
}

impl CameraParams {
    /// A camera at `position` looking at `target`, with world `+Y` upwards.
    ///
    /// `fov_y` is the vertical field of view in radians; the horizontal one
    /// follows from the aspect ratio, which is how every other camera here is
    /// built.
    pub fn looking_at(
        position: glam::Vec3,
        target: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        depth: f32,
    ) -> Self {
        let orientation = orientation_looking(target - position, glam::Vec3::Y);
        Self {
            cam_position: position.into(),
            depth,
            cam_orientation: orientation.into(),
            fov: [2.0 * ((0.5 * fov_y).tan() * aspect).atan(), fov_y],
            principal: [0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_looking_camera_faces_its_target_and_is_not_mirrored() {
        let position = glam::Vec3::new(2.0, 1.5, -3.0);
        let target = glam::Vec3::new(0.0, 0.5, 0.0);
        let params = CameraParams::looking_at(position, target, 0.9, 4.0 / 3.0, 100.0);
        let orientation = glam::Quat::from_array(params.cam_orientation);

        let forward = orientation * glam::Vec3::Z;
        assert!(forward.dot((target - position).normalize()) > 0.9999);
        // Local +Y is the bottom of the image, so it has to point downwards.
        assert!((orientation * glam::Vec3::Y).dot(glam::Vec3::Y) < 0.0);
        assert!((glam::Mat3::from_quat(orientation).determinant() - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn looking_straight_down_still_gives_a_rotation() {
        // World up is parallel to the view here, so the usual reference has to
        // be replaced rather than normalised to nothing.
        let orientation = orientation_looking(glam::Vec3::NEG_Y, glam::Vec3::Y);
        let matrix = glam::Mat3::from_quat(orientation);
        assert!(matrix.is_finite());
        assert!((matrix.determinant() - 1.0).abs() < 1.0e-4);
        assert!((orientation * glam::Vec3::Z).dot(glam::Vec3::NEG_Y) > 0.9999);
    }
}
