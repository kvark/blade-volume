//! Inverse rendering: posed photographs in, a scene a renderer can light.
//!
//! The pipeline is five stages, and each one is separately measurable:
//!
//! 1. [`capture`] reads the images and their poses, in linear radiance.
//! 2. [`surface`] turns a point cloud into oriented discs.
//! 3. [`powerfoam`] can continue a masked surface through weighted cells.
//! 4. [`refine`] moves those particles to one shared multi-view surface.
//! 5. [`decompose`] splits what was observed into a material and a light.
//! 6. [`score`] renders the result back into the capture and compares.
//!
//! The fifth stage is the one that decides whether any of the others worked,
//! so it does not share any code with them: it goes through the same GPU
//! tracer a viewer would use, and it has no opportunity to agree with the
//! solver by construction.

pub mod capture;
pub mod decompose;
pub mod depth;
pub mod powerfoam;
pub mod refine;
pub mod score;
pub mod surface;
pub mod tracks;
pub mod truth;
pub mod visibility;
