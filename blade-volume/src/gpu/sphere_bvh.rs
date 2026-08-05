use std::mem;

pub(super) const LEAF_BIT: u32 = 1 << 31;
// The shared hierarchy loses 0.05--0.06 ms to its fixed dispatch overhead at
// 2K and 10K sites, but saves 1.11 ms of recorder time per batch at 50K. Keep
// small clouds on the already parallel exhaustive gather.
pub(super) const MIN_POINTS: usize = 32 * 1024;

/// Flat binary BVH node for bounded point-cloud supports.
///
/// The layout matches two WGSL `vec3 + u32` pairs. Internal nodes store child
/// node indices in `left` and `right`; leaves set [`LEAF_BIT`] in `left` and
/// store the point index in `right`.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SphereBvhNode {
    min: [f32; 3],
    left: u32,
    max: [f32; 3],
    right: u32,
}

const _: () = assert!(mem::size_of::<SphereBvhNode>() == 32);

pub(super) fn build(points: &[glam::Vec4], radii: &[f32]) -> Vec<SphereBvhNode> {
    assert_eq!(points.len(), radii.len());
    assert!(!points.is_empty(), "sphere BVH needs at least one point");
    assert!(
        points.len() <= LEAF_BIT as usize / 2,
        "sphere BVH exceeds the node-index encoding"
    );

    let mut point_indices = (0..points.len() as u32).collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(2 * points.len() - 1);
    let root = build_node(&mut point_indices, points, radii, &mut nodes);
    debug_assert_eq!(root, 0);
    debug_assert_eq!(nodes.len(), 2 * points.len() - 1);
    nodes
}

fn build_node(
    point_indices: &mut [u32],
    points: &[glam::Vec4],
    radii: &[f32],
    nodes: &mut Vec<SphereBvhNode>,
) -> u32 {
    let node_index = nodes.len() as u32;
    nodes.push(SphereBvhNode::default());

    if point_indices.len() == 1 {
        let point_index = point_indices[0];
        let center = points[point_index as usize].truncate();
        let radius = radii[point_index as usize];
        // Expand by several ULPs so the hierarchy cannot reject an exact
        // shader sphere hit at a grazing or large-coordinate boundary.
        let scale = center.abs().max_element().max(radius).max(1.0);
        let extent = glam::Vec3::splat(radius + 8.0 * f32::EPSILON * scale);
        nodes[node_index as usize] = SphereBvhNode {
            min: (center - extent).to_array(),
            left: LEAF_BIT,
            max: (center + extent).to_array(),
            right: point_index,
        };
        return node_index;
    }

    let first = points[point_indices[0] as usize].truncate();
    let (center_min, center_max) =
        point_indices
            .iter()
            .skip(1)
            .fold((first, first), |(minimum, maximum), &point_index| {
                let center = points[point_index as usize].truncate();
                (minimum.min(center), maximum.max(center))
            });
    let extent = center_max - center_min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let middle = point_indices.len() / 2;
    point_indices.select_nth_unstable_by(middle, |&left, &right| {
        points[left as usize][axis]
            .total_cmp(&points[right as usize][axis])
            .then_with(|| left.cmp(&right))
    });
    let (left_points, right_points) = point_indices.split_at_mut(middle);
    let left = build_node(left_points, points, radii, nodes);
    let right = build_node(right_points, points, radii, nodes);
    let left_node = nodes[left as usize];
    let right_node = nodes[right as usize];
    nodes[node_index as usize] = SphereBvhNode {
        min: glam::Vec3::from(left_node.min)
            .min(glam::Vec3::from(right_node.min))
            .to_array(),
        left,
        max: glam::Vec3::from(left_node.max)
            .max(glam::Vec3::from(right_node.max))
            .to_array(),
        right,
    };
    node_index
}

#[cfg(test)]
mod tests {
    use super::{SphereBvhNode, LEAF_BIT};

    fn intersects(
        node: SphereBvhNode,
        origin: glam::Vec3,
        direction: glam::Vec3,
        depth: f32,
    ) -> bool {
        let minimum = glam::Vec3::from(node.min);
        let maximum = glam::Vec3::from(node.max);
        let mut near = 0.0f32;
        let mut far = depth;
        for axis in 0..3 {
            if direction[axis].abs() <= 1.0e-20 {
                if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                    return false;
                }
                continue;
            }
            let first = (minimum[axis] - origin[axis]) / direction[axis];
            let second = (maximum[axis] - origin[axis]) / direction[axis];
            near = near.max(first.min(second));
            far = far.min(first.max(second));
            if far < near {
                return false;
            }
        }
        true
    }

    fn candidates(
        nodes: &[SphereBvhNode],
        origin: glam::Vec3,
        direction: glam::Vec3,
        depth: f32,
    ) -> Vec<u32> {
        let mut result = Vec::new();
        let mut stack = vec![0u32];
        while let Some(node_index) = stack.pop() {
            let node = nodes[node_index as usize];
            if !intersects(node, origin, direction, depth) {
                continue;
            }
            if node.left & LEAF_BIT != 0 {
                result.push(node.right);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        result.sort_unstable();
        result
    }

    fn sphere_hit(
        point: glam::Vec4,
        radius: f32,
        origin: glam::Vec3,
        direction: glam::Vec3,
        depth: f32,
    ) -> bool {
        let offset = origin - point.truncate();
        let along = offset.dot(direction);
        let discriminant = along * along - (offset.length_squared() - radius * radius);
        if discriminant <= 0.0 {
            return false;
        }
        let root = discriminant.sqrt();
        -along + root > 0.0 && -along - root < depth
    }

    #[test]
    fn hierarchy_contains_every_exact_sphere_hit() {
        let points = (0..97)
            .map(|index| {
                let translation = if index >= 80 { 100_000.0 } else { 0.0 };
                glam::Vec4::new(
                    translation + (index * 17 % 29) as f32 - 14.0,
                    (index * 11 % 23) as f32 - 11.0,
                    (index * 7 % 19) as f32 - 9.0,
                    0.0,
                )
            })
            .collect::<Vec<_>>();
        let radii = (0..points.len())
            .map(|index| 0.25 + (index * 5 % 13) as f32 * 0.15)
            .collect::<Vec<_>>();
        let nodes = super::build(&points, &radii);
        assert_eq!(nodes.len(), 2 * points.len() - 1);

        let origins = [
            glam::Vec3::new(-30.0, 2.0, -4.0),
            glam::Vec3::new(99_970.0, -3.0, 2.0),
        ];
        for (index, point) in points.iter().enumerate() {
            let origin = origins[usize::from(index >= 80)];
            let direction = (point.truncate() - origin).normalize();
            let depth = 100.0;
            let found = candidates(&nodes, origin, direction, depth);
            let expected = points
                .iter()
                .zip(&radii)
                .enumerate()
                .filter_map(|(candidate, (&point, &radius))| {
                    sphere_hit(point, radius, origin, direction, depth).then_some(candidate as u32)
                })
                .collect::<Vec<_>>();
            assert!(!expected.is_empty());
            for expected_index in expected {
                assert!(
                    found.binary_search(&expected_index).is_ok(),
                    "ray towards {index} omitted intersected sphere {expected_index}"
                );
            }
        }
    }

    #[test]
    fn coincident_centres_form_a_balanced_complete_tree() {
        let points = vec![glam::Vec4::ZERO; 33];
        let radii = vec![1.0; points.len()];
        let nodes = super::build(&points, &radii);
        let mut leaves = nodes
            .iter()
            .filter_map(|node| (node.left & LEAF_BIT != 0).then_some(node.right))
            .collect::<Vec<_>>();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..points.len() as u32).collect::<Vec<_>>());
    }
}
