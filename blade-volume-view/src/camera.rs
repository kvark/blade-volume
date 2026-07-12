//! Interactive camera for viewer applications.
//!
//! This module provides `ControlledCamera` for keyboard/mouse-controlled
//! 3D navigation, with winit input handling.

use blade_volume::CameraParams;
use winit::event::MouseScrollDelta;
use winit::keyboard::KeyCode;

const MAX_FLY_SPEED: f32 = 1_000_000.0;

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
        self.fly_speed = (self.fly_speed * shift.exp()).clamp(1.0, MAX_FLY_SPEED);
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
