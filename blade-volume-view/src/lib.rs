//! Viewer utilities for blade-volume.
//!
//! This crate provides interactive viewing capabilities for volumetric data:
//! - `ControlledCamera` for keyboard/mouse-controlled 3D navigation
//! - `SceneRenderer` for unified multi-object scene rendering
//! - Rendering backends for different volumetric representations
//!
//! This crate depends on `winit` for input handling, keeping the core
//! `blade-volume` library free of windowing dependencies.

#![allow(irrefutable_let_patterns)]

mod camera;
mod render;
mod scene_renderer;

pub use camera::ControlledCamera;
pub use render::{
    preprocess_shader, DebugMode, GaussianSettings, RadFoamSettings, RenderBackend, RenderSize,
};
pub use scene_renderer::{SceneDebugMode, SceneParams, SceneRenderer};

// Re-export commonly used types from dependencies for convenience
pub use blade_volume::CameraParams;
pub use winit;
