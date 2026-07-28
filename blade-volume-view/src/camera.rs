//! Interactive camera for viewer applications.
//!
//! This module provides `ControlledCamera` for keyboard/mouse-controlled
//! 3D navigation, with winit input handling.

use blade_volume::CameraParams;
use winit::event::MouseScrollDelta;
use winit::keyboard::KeyCode;

const MAX_FLY_SPEED: f32 = 1_000_000.0;
/// The slowest the wheel can make the camera.
///
/// This used to be one world unit per key press, which is a sensible floor for
/// a captured scene tens of units across and useless for a converted asset
/// that is three: every step crossed the whole car. The floor has to be below
/// the scale of the smallest thing anyone would want to fly around.
const MIN_FLY_SPEED: f32 = 1.0e-4;

/// A camera that can be controlled via keyboard and mouse input.
#[derive(Clone)]
pub struct ControlledCamera {
    pub position: glam::Vec3,
    pub orientation: glam::Quat,
    pub fov_y: f32,
    pub depth: f32,
    pub fly_speed: f32,
}

impl Default for ControlledCamera {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            orientation: glam::Quat::IDENTITY,
            fov_y: 1.0,
            depth: 10_000.0,
            fly_speed: 1.0,
        }
    }
}

impl ControlledCamera {
    /// Returns the view matrix (world-to-camera transform).
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.orientation, self.position).inverse()
    }

    /// Returns the projection matrix for the given aspect ratio.
    pub fn get_projection_matrix(&self, aspect: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov_y, aspect, 1.0, self.depth)
    }

    /// Moves the camera by an offset in local space.
    pub fn move_by(&mut self, offset: glam::Vec3) {
        self.position += self.orientation * offset;
    }

    /// Rotates the camera around its local Z axis.
    pub fn rotate_z_by(&mut self, angle: f32) {
        self.orientation *= glam::Quat::from_rotation_z(angle);
    }

    /// Handles keyboard input for camera movement.
    /// Returns `true` if the key was handled.
    pub fn on_key(&mut self, code: KeyCode, delta: f32) -> bool {
        let move_offset = self.fly_speed * delta;
        let rotate_offset_z = 1000.0 * delta;

        match code {
            KeyCode::KeyW => self.move_by(glam::Vec3::new(0.0, 0.0, move_offset)),
            KeyCode::KeyS => self.move_by(glam::Vec3::new(0.0, 0.0, -move_offset)),
            KeyCode::KeyA => self.move_by(glam::Vec3::new(-move_offset, 0.0, 0.0)),
            KeyCode::KeyD => self.move_by(glam::Vec3::new(move_offset, 0.0, 0.0)),
            KeyCode::KeyZ => self.move_by(glam::Vec3::new(0.0, -move_offset, 0.0)),
            KeyCode::KeyX => self.move_by(glam::Vec3::new(0.0, move_offset, 0.0)),
            KeyCode::KeyQ => self.rotate_z_by(rotate_offset_z),
            KeyCode::KeyE => self.rotate_z_by(-rotate_offset_z),
            _ => return false,
        }
        true
    }

    /// Handles mouse wheel input for fly speed adjustment.
    pub fn on_wheel(&mut self, delta: MouseScrollDelta) {
        let shift = match delta {
            MouseScrollDelta::LineDelta(_, lines) => lines,
            MouseScrollDelta::PixelDelta(position) => position.y as f32,
        };
        self.fly_speed = (self.fly_speed * shift.exp()).clamp(MIN_FLY_SPEED, MAX_FLY_SPEED);
    }

    /// Point the camera at `target`, keeping world `+Y` upwards in the image.
    ///
    /// The local frame is right, *down*, forward: this renderer's first image
    /// row looks along `-Y` in camera space, so the axis that has to oppose
    /// world up is `Y` and not the other way round. Getting it backwards
    /// produces an upside-down image that is otherwise entirely convincing.
    pub fn look_at(&mut self, target: glam::Vec3) {
        if target == self.position {
            return;
        }
        self.orientation = blade_volume::orientation_looking(target - self.position, glam::Vec3::Y);
    }

    /// Put the whole of a bounding box in view, from a three-quarter angle.
    ///
    /// Converted assets arrive at whatever scale their author used, and a
    /// default camera at the origin looking down `+Z` is inside some of them
    /// and light-years from others. This also sets the fly speed from the size
    /// of the thing, so the controls are usable without being told the scale.
    pub fn frame_bounds(&mut self, min: glam::Vec3, max: glam::Vec3) {
        let center = 0.5 * (min + max);
        let radius = (0.5 * (max - min)).length().max(1.0e-6);
        // Far enough that the bounding sphere fits the narrower field of view,
        // with a margin so the thing is not touching the frame edges. The
        // sphere is the diagonal of the box, so most assets already sit well
        // inside it and a large margin just renders a small picture.
        let distance = 1.15 * radius / (0.5 * self.fov_y).sin().max(1.0e-3);
        self.position = center + glam::Vec3::new(0.7, 0.45, 1.0).normalize() * distance;
        self.look_at(center);
        self.depth = distance + 8.0 * radius;
        // A key press should cross a tenth of the object, not all of it.
        self.fly_speed = (0.1 * radius).clamp(MIN_FLY_SPEED, MAX_FLY_SPEED);
    }

    /// Handles mouse drag for camera rotation.
    /// `dx` and `dy` are the mouse movement in pixels.
    /// `drag_speed` controls the sensitivity.
    pub fn on_mouse_drag(&mut self, dx: f32, dy: f32, drag_speed: f32) {
        let prev = self.orientation;

        // Yaw around global up (assume +Y is world up)
        let world_up = glam::Vec3::Y;
        let yaw = glam::Quat::from_axis_angle(world_up, dx * drag_speed);

        // Pitch around camera's local right axis
        let right = prev * glam::Vec3::X;
        let pitch = glam::Quat::from_axis_angle(right, dy * drag_speed);

        self.orientation = yaw * pitch * prev;
    }

    /// The direction the camera looks, in world space.
    pub fn forward(&self) -> glam::Vec3 {
        self.orientation * glam::Vec3::Z
    }

    /// Converts to GPU-compatible camera parameters.
    pub fn to_params(&self, aspect: f32) -> CameraParams {
        CameraParams {
            cam_position: self.position.into(),
            depth: self.depth,
            cam_orientation: self.orientation.into(),
            fov: [aspect * self.fov_y, self.fov_y],
            principal: [0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_puts_the_whole_box_in_front_of_the_camera() {
        let min = glam::Vec3::new(-0.76, -0.01, -1.51);
        let max = glam::Vec3::new(0.76, 1.31, 1.51);
        let center = 0.5 * (min + max);
        let mut camera = ControlledCamera::default();
        camera.frame_bounds(min, max);

        // Looking at the middle of it, from outside it.
        let to_center = center - camera.position;
        assert!(
            camera.forward().dot(to_center.normalize()) > 0.999,
            "the camera is not pointed at the centre"
        );
        let radius = (0.5 * (max - min)).length();
        assert!(to_center.length() > radius, "the camera is inside the box");

        // Every corner is in front of the camera and inside the field of view.
        let half_angle = 0.5 * camera.fov_y;
        for corner in 0..8 {
            let point = glam::Vec3::new(
                if corner & 1 == 0 { min.x } else { max.x },
                if corner & 2 == 0 { min.y } else { max.y },
                if corner & 4 == 0 { min.z } else { max.z },
            );
            let offset = point - camera.position;
            let along = offset.dot(camera.forward());
            assert!(along > 0.0, "corner {corner} is behind the camera");
            let across = (offset - along * camera.forward()).length();
            assert!(
                (across / along).atan() < half_angle,
                "corner {corner} is outside the vertical field of view"
            );
        }
        // And the controls are scaled to the thing rather than to nothing.
        assert!(camera.fly_speed < radius && camera.fly_speed > 0.0);
    }

    #[test]
    fn looking_at_something_keeps_world_up_at_the_top_of_the_image() {
        let mut camera = ControlledCamera {
            position: glam::Vec3::new(3.0, 2.0, -4.0),
            ..Default::default()
        };
        camera.look_at(glam::Vec3::ZERO);
        // The first image row looks along local -Y, so world up has to have a
        // negative component there for the picture to be the right way up.
        let local_down = camera.orientation * glam::Vec3::Y;
        assert!(
            local_down.dot(glam::Vec3::Y) < 0.0,
            "the image would be upside down: {local_down:?}"
        );
        // A rotation, not a reflection: a mirrored frame would render a
        // plausible image of a mirrored scene.
        let matrix = glam::Mat3::from_quat(camera.orientation);
        assert!((matrix.determinant() - 1.0).abs() < 1.0e-4);
    }
}
