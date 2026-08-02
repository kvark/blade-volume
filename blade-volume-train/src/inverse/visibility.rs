//! Which parts of the sky each surfel can actually see.
//!
//! Without this, a patch of floor in a sphere's shadow has only one way to be
//! dark: dark paint. The fit duly supplies it, and the recovered material is a
//! photograph of the shadow rather than a description of the surface. Every
//! measurement of this pipeline so far has been limited by that, and it is the
//! one thing a smooth spherical-harmonic light can never explain, because a
//! cast shadow has an edge and nine coefficients do not.
//!
//! The visibility is computed the same way a renderer computes it — a shadow
//! map per direction — rather than by ray casting. One orthographic depth
//! buffer along a direction answers the question for every surfel at once, and
//! the whole environment costs one buffer per texel rather than one ray per
//! surfel per texel.

use blade_volume as vol;
use std::thread;

/// How the shadow maps are drawn.
#[derive(Clone, Copy, Debug)]
pub struct VisibilityOptions {
    /// Side of each orthographic depth buffer, in texels.
    pub resolution: usize,
    /// Depth slack, as a multiple of a surfel's radius. A surface shadowing
    /// itself is the classic failure here, and a disc has thickness only in
    /// the sense that its own samples disagree about where it is.
    pub bias: f32,
}

impl Default for VisibilityOptions {
    fn default() -> Self {
        Self {
            resolution: 512,
            bias: 3.0,
        }
    }
}

/// Whether each surfel can see each environment texel.
///
/// One bit each, because there are a lot of them: a million surfels against
/// five hundred texels is half a gigabyte stored as bytes and sixty-four
/// megabytes stored as bits, and the answer is genuinely binary.
pub struct Visibility {
    pub texels: usize,
    words: usize,
    bits: Vec<u64>,
}

impl Visibility {
    fn new(surfels: usize, texels: usize, fill: u64) -> Self {
        let words = texels.div_ceil(64);
        Self {
            texels,
            words,
            bits: vec![fill; surfels * words],
        }
    }

    /// Everything sees everything: the answer when nothing occludes.
    pub fn open(surfels: usize, texels: usize) -> Self {
        let mut result = Self::new(surfels, texels, !0);
        // The last word runs past the end of the texels, and its spare bits
        // would otherwise be counted as sky that is not there.
        let spare = result.words * 64 - texels;
        if spare > 0 && result.words > 0 {
            let mask = !0u64 >> spare;
            for surfel in 0..surfels {
                result.bits[surfel * result.words + result.words - 1] &= mask;
            }
        }
        result
    }

    pub fn visible(&self, surfel: usize, texel: usize) -> bool {
        self.bits[surfel * self.words + texel / 64] & (1 << (texel % 64)) != 0
    }

    /// Mean fraction of the sphere a surfel can see, counting texels rather
    /// than solid angle. Ambient occlusion, roughly, and worth printing: a
    /// value near one means nothing was found to occlude anything, which is
    /// either a very open scene or a broken shadow map.
    pub fn mean_openness(&self) -> f32 {
        if self.bits.is_empty() || self.texels == 0 {
            return 1.0;
        }
        let surfels = self.bits.len() / self.words;
        let set: u32 = self.bits.iter().map(|word| word.count_ones()).sum();
        set as f32 / (surfels * self.texels) as f32
    }
}

/// An orthonormal frame with `axis` as its third column.
fn frame(axis: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let up = if axis.y.abs() < 0.9 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let right = up.cross(axis).normalize();
    (right, axis.cross(right))
}

/// One direction's shadow map: what is nearest to the light in each texel,
/// and which surfel that is.
///
/// Keeping the owner and not just the depth is what makes a bounce free. A
/// surfel that is in shadow is in the shadow *of something*, and that
/// something is the only thing that can be lighting it from that direction.
struct Map {
    right: glam::Vec3,
    up: glam::Vec3,
    direction: glam::Vec3,
    centre: glam::Vec3,
    extent: f32,
    scale: f32,
    resolution: usize,
    depth: Vec<f32>,
    owner: Vec<u32>,
}

impl Map {
    fn build(
        surfels: &[vol::relight::Surfel],
        direction: glam::Vec3,
        min: glam::Vec3,
        max: glam::Vec3,
        options: VisibilityOptions,
    ) -> Self {
        let (right, up) = frame(direction);
        let centre = 0.5 * (min + max);
        // A cube that contains the scene from any angle, so one extent serves
        // every direction and the maps are comparable.
        let extent = (0.5 * (max - min)).length().max(1.0e-6);
        let resolution = options.resolution.max(4);
        let scale = resolution as f32 / (2.0 * extent);
        let mut map = Self {
            right,
            up,
            direction,
            centre,
            extent,
            scale,
            resolution,
            depth: vec![f32::NEG_INFINITY; resolution * resolution],
            owner: vec![u32::MAX; resolution * resolution],
        };

        for (index, surfel) in surfels.iter().enumerate() {
            let (x, y, along) = map.project(surfel);
            let normal = glam::Vec3::from(surfel.normal);
            let facing = normal.dot(direction).abs();
            let radius = (surfel.radius * scale).max(0.5);

            // A disc seen edge-on covers a sliver, not a circle. Splatting the
            // circumscribed circle instead is what turned a room into a cave:
            // at a grazing angle every element occluded its own neighbours,
            // and the scene came back with a sixth of its sky.
            //
            // The projection is an ellipse: full radius across the normal's
            // projection, and `radius * |n . d|` along it.
            let projected = glam::Vec2::new(normal.dot(right), normal.dot(up));
            let (major, minor) = match projected.try_normalize() {
                Some(axis) => (glam::Vec2::new(-axis.y, axis.x), axis),
                None => (glam::Vec2::X, glam::Vec2::Y),
            };
            let minor_radius = (radius * facing).max(0.5);
            let extent_x = (major.x.abs() * radius).max(minor.x.abs() * minor_radius);
            let extent_y = (major.y.abs() * radius).max(minor.y.abs() * minor_radius);
            let min_x = ((x - extent_x).floor() as isize).max(0) as usize;
            let min_y = ((y - extent_y).floor() as isize).max(0) as usize;
            let max_x = ((x + extent_x).ceil() as isize).min(resolution as isize - 1);
            let max_y = ((y + extent_y).ceil() as isize).min(resolution as isize - 1);
            if max_x < 0 || max_y < 0 {
                continue;
            }
            for py in min_y..=max_y as usize {
                for px in min_x..=max_x as usize {
                    let offset = glam::Vec2::new(px as f32 + 0.5 - x, py as f32 + 0.5 - y);
                    let across = offset.dot(major) / radius;
                    let along_normal = offset.dot(minor) / minor_radius;
                    if across * across + along_normal * along_normal > 1.0 {
                        continue;
                    }
                    let cell = py * resolution + px;
                    if along > map.depth[cell] {
                        map.depth[cell] = along;
                        map.owner[cell] = index as u32;
                    }
                }
            }
        }
        map
    }

    fn project(&self, surfel: &vol::relight::Surfel) -> (f32, f32, f32) {
        let offset = glam::Vec3::from(surfel.center) - self.centre;
        (
            (offset.dot(self.right) + self.extent) * self.scale,
            (offset.dot(self.up) + self.extent) * self.scale,
            offset.dot(self.direction),
        )
    }

    /// Whether this surfel reaches the light, and if not, what is in the way.
    ///
    /// `None` means the sky. A surfel facing away is neither: the cosine term
    /// already says it receives nothing, so it is reported as shadowed by
    /// nothing at all.
    fn occluder(&self, surfel: &vol::relight::Surfel, options: VisibilityOptions) -> Reach {
        let facing = glam::Vec3::from(surfel.normal).dot(self.direction);
        if facing <= 0.0 {
            return Reach::FacingAway;
        }
        let (x, y, along) = self.project(surfel);
        let px = x as usize;
        let py = y as usize;
        if x < 0.0 || y < 0.0 || px >= self.resolution || py >= self.resolution {
            return Reach::Sky;
        }
        // Slope-scaled: a surface nearly edge-on to the light spans a whole
        // texel in depth, and a fixed bias would have it shadow itself.
        let bias = options.bias * surfel.radius / facing.max(0.1);
        let cell = py * self.resolution + px;
        if along + bias >= self.depth[cell] {
            Reach::Sky
        } else {
            Reach::Blocked(self.owner[cell])
        }
    }
}

enum Reach {
    Sky,
    Blocked(u32),
    FacingAway,
}

/// Compute what every surfel can see of every environment texel.
pub fn compute(
    model: &vol::relight::RelightModel,
    directions: &[glam::Vec3],
    options: VisibilityOptions,
) -> Visibility {
    let texels = directions.len();
    let mut result = Visibility::new(model.surfels.len(), texels, 0);
    let Some((min, max)) = model.bounds() else {
        return Visibility::open(model.surfels.len(), texels);
    };

    // One direction per thread rather than one chunk of surfels each: a
    // direction owns a whole depth buffer, and splitting a buffer between
    // threads would need it shared.
    //
    // Directions are handed out in blocks of sixty-four so every thread owns
    // whole words of the bit table. Without that, two threads setting bits in
    // one word would race, and the loss would be a scattering of surfels
    // wrongly in shadow — which looks exactly like a scene with more occlusion
    // than it has.
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let words = result.words;
    let surfels = &model.surfels;
    let per_thread = (texels.div_ceil(threads).div_ceil(64) * 64).max(64);
    let slots: Vec<usize> = (0..texels).collect();
    let base = result.bits.as_mut_ptr() as usize;
    let length = result.bits.len();
    thread::scope(|scope| {
        for block in slots.chunks(per_thread) {
            scope.spawn(move || {
                let bits = unsafe { std::slice::from_raw_parts_mut(base as *mut u64, length) };
                for &slot in block {
                    let map = Map::build(surfels, directions[slot], min, max, options);
                    let word = slot / 64;
                    let mask = 1u64 << (slot % 64);
                    for (index, surfel) in surfels.iter().enumerate() {
                        if let Reach::Sky = map.occluder(surfel, options) {
                            bits[index * words + word] |= mask;
                        }
                    }
                }
            });
        }
    });
    result
}

/// One bounce: the light a surfel receives from whatever is shadowing it.
///
/// Modelling shadows without this is worse than modelling neither, and that is
/// a measured result rather than a caution. A fit that knows a patch of floor
/// is shadowed but not that the sphere above it is bouncing light back down
/// sees a patch far brighter than its model allows, and has nowhere to put the
/// difference except into the sky — which comes back wrong in shape, not just
/// in scale.
///
/// `outgoing` is what each surfel is currently believed to emit. The result is
/// irradiance in the same units the direct term is in, so the two add.
pub fn bounce(
    model: &vol::relight::RelightModel,
    kernel: &crate::inverse::decompose::Kernel,
    outgoing: &[[f32; 3]],
    options: VisibilityOptions,
) -> Vec<[f32; 3]> {
    let count = model.surfels.len();
    let mut total = vec![[0.0f32; 3]; count];
    let Some((min, max)) = model.bounds() else {
        return total;
    };
    let surfels = &model.surfels;
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let texels = kernel.texels();
    let per_thread = texels.div_ceil(threads).max(1);
    let slots: Vec<usize> = (0..texels).collect();

    // Each thread accumulates into its own copy and they are summed after.
    // Sharing one would race per surfel rather than per word, and a bounce is
    // a sum rather than a bit, so there is no alignment trick to lean on.
    let mut partials: Vec<Vec<[f32; 3]>> = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for block in slots.chunks(per_thread) {
            handles.push(scope.spawn(move || {
                let mut local = vec![[0.0f32; 3]; count];
                for &slot in block {
                    let direction = kernel.directions[slot];
                    let solid_angle = kernel.solid_angles[slot];
                    let map = Map::build(surfels, direction, min, max, options);
                    for (index, surfel) in surfels.iter().enumerate() {
                        let Reach::Blocked(owner) = map.occluder(surfel, options) else {
                            continue;
                        };
                        if owner as usize == index {
                            continue;
                        }
                        let cosine = glam::Vec3::from(surfel.normal).dot(direction);
                        if cosine <= 0.0 {
                            continue;
                        }
                        let weight = cosine * solid_angle * std::f32::consts::FRAC_1_PI;
                        let source = outgoing[owner as usize];
                        for (value, radiance) in local[index].iter_mut().zip(source) {
                            *value += weight * radiance;
                        }
                    }
                }
                local
            }));
        }
        for handle in handles {
            partials.push(handle.join().unwrap());
        }
    });
    for partial in &partials {
        for (sum, part) in total.iter_mut().zip(partial) {
            for (value, add) in sum.iter_mut().zip(part) {
                *value += add;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A floor, and one disc floating above the middle of it.
    fn floor_with_a_lid() -> vol::relight::RelightModel {
        let mut surfels = Vec::new();
        for i in 0..41 {
            for j in 0..41 {
                surfels.push(vol::relight::Surfel {
                    center: [(i as f32 - 20.0) * 0.1, 0.0, (j as f32 - 20.0) * 0.1],
                    radius: 0.075,
                    normal: [0.0, 1.0, 0.0],
                    material: 0,
                });
            }
        }
        // A wide blocker directly overhead, at the origin.
        for i in 0..9 {
            for j in 0..9 {
                surfels.push(vol::relight::Surfel {
                    center: [(i as f32 - 4.0) * 0.1, 1.0, (j as f32 - 4.0) * 0.1],
                    radius: 0.075,
                    normal: [0.0, 1.0, 0.0],
                    material: 0,
                });
            }
        }
        vol::relight::RelightModel {
            kernel: vol::relight::ParticleKernel::Compact,
            surfels,
            materials: vec![vol::relight::Material::default()],
        }
    }

    #[test]
    fn the_floor_under_the_lid_cannot_see_straight_up() {
        let model = floor_with_a_lid();
        let straight_up = [glam::Vec3::Y];
        let visibility = compute(&model, &straight_up, VisibilityOptions::default());

        let under = model
            .surfels
            .iter()
            .position(|s| {
                s.center[1] == 0.0 && s.center[0].abs() < 0.01 && s.center[2].abs() < 0.01
            })
            .unwrap();
        let clear = model
            .surfels
            .iter()
            .position(|s| s.center[1] == 0.0 && s.center[0] > 1.5)
            .unwrap();
        assert!(
            !visibility.visible(under, 0),
            "the shadowed patch was called lit"
        );
        assert!(
            visibility.visible(clear, 0),
            "a patch in the open was called shadowed"
        );
    }

    #[test]
    fn a_flat_floor_does_not_shadow_itself() {
        // Shadow acne, stated as a test. A plane lit from above must come back
        // entirely lit; a bias too small turns the whole thing black and every
        // albedo afterwards is wrong by the same factor.
        let mut model = floor_with_a_lid();
        model.surfels.truncate(41 * 41);
        let visibility = compute(&model, &[glam::Vec3::Y], VisibilityOptions::default());
        let lit = (0..model.surfels.len())
            .filter(|&index| visibility.visible(index, 0))
            .count();
        assert_eq!(
            lit,
            model.surfels.len(),
            "{} of {} patches shadowed themselves",
            model.surfels.len() - lit,
            model.surfels.len()
        );
    }

    #[test]
    fn a_surface_facing_away_from_the_light_gets_nothing() {
        let model = floor_with_a_lid();
        let visibility = compute(&model, &[glam::Vec3::NEG_Y], VisibilityOptions::default());
        assert!(
            (0..model.surfels.len()).all(|index| !visibility.visible(index, 0)),
            "an upward-facing floor was lit from below"
        );
    }
}
