use blade_graphics as gpu;

use std::{mem, ptr, slice};

/// GPU-side storage for RadFoam point cloud rendering.
///
/// This uploads the buffers required by the RadFoam tracing kernel:
/// - `points`: `vec4<f32>[N]` where xyz is position and w is per-point radius
///   (Power Foam weight, or 0 for plain Voronoi)
/// - `surface_normals`: `vec4<f32>[N]` containing unit dipole normals, or one
///   zero dummy element for an unoriented cloud
/// - `attributes`: packed `f32[N * attr_dim]`, where `attr_dim = 1 + 3 * (1 + sh_degree)^2`
///   and the last scalar in each row is density
/// - `point_adjacency`: flattened neighbor list `u32[K]`
/// - `point_adjacency_offsets`: CSR offsets `u32[N+1]`
///
/// Notes:
/// - This does not build any hardware ray tracing acceleration structures.
/// - This is intended for compute-only Voronoi traversal.
pub struct RadFoamGpuCloud {
    points_buf: gpu::Buffer,
    surface_normals_buf: gpu::Buffer,
    attributes_buf: gpu::Buffer,
    point_adjacency_buf: gpu::Buffer,
    point_adjacency_offsets_buf: gpu::Buffer,
    point_index: PointIndex,

    pub sh_degree: usize,
    pub attr_dim: usize,
    pub num_points: usize,
    pub num_adjacency: usize,
    pub is_power_foam: bool,
    pub is_oriented: bool,
}

struct PointIndex {
    tree: kiddo::ImmutableKdTree<f32, 3>,
    radii: Option<Vec<f32>>,
    max_radius_squared: f32,
}

impl PointIndex {
    fn new(points: &[glam::Vec4], radii: Option<&[f32]>) -> Self {
        let positions: Vec<[f32; 3]> = points
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect();
        let radii = radii.map(<[f32]>::to_vec);
        let max_radius_squared = radii
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|radius| radius * radius)
            .fold(0.0_f32, f32::max);
        Self {
            tree: kiddo::ImmutableKdTree::new_from_slice(&positions),
            radii,
            max_radius_squared,
        }
    }

    fn containing_point(&self, position: glam::Vec3) -> u32 {
        let query = position.to_array();
        let nearest = self.tree.nearest_one::<kiddo::SquaredEuclidean>(&query);
        let mut best_index = nearest.item as u32;
        let initial_radius = self
            .radii
            .as_deref()
            .map_or(0.0, |radii| radii[best_index as usize]);
        let mut best_power_distance = nearest.distance - initial_radius * initial_radius;
        let search_distance = if self.radii.is_some() {
            (best_power_distance + self.max_radius_squared).max(0.0)
        } else {
            nearest.distance
        };

        // Any omitted point has d² greater than search_distance. Even at the
        // global maximum radius its power distance cannot improve the current
        // best, so this bounded Euclidean query is an exact power-nearest query.
        for hit in self
            .tree
            .within_unsorted::<kiddo::SquaredEuclidean>(&query, search_distance)
        {
            let index = hit.item as u32;
            let radius = self
                .radii
                .as_deref()
                .map_or(0.0, |radii| radii[index as usize]);
            let power_distance = hit.distance - radius * radius;
            let ordering = power_distance.total_cmp(&best_power_distance);
            if ordering.is_lt() || (ordering.is_eq() && index < best_index) {
                best_power_distance = power_distance;
                best_index = index;
            }
        }
        best_index
    }
}

impl RadFoamGpuCloud {
    /// Creates a GPU point cloud from a unified model.
    ///
    /// Requires the model to have `adjacency` data.
    pub fn new(
        model: &crate::PointCloudModel,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        model
            .validate()
            .unwrap_or_else(|err| panic!("invalid RadFoam model: {err}"));
        let adjacency = model
            .adjacency
            .as_ref()
            .expect("RadFoamGpuCloud requires adjacency");

        let num_points = model.len();
        assert_eq!(
            num_points + 1,
            adjacency.offsets.len(),
            "adjacency.offsets must have length N+1"
        );

        let num_adjacency = adjacency.neighbors.len();
        let sh_component_count = model.sh_component_count();
        let attr_dim = 1 + 3 * sh_component_count; // SH coefficients + density

        assert!(num_points > 0, "Model has zero points; nothing to upload");

        // Sizes
        let points_size = (num_points * mem::size_of::<[f32; 4]>()) as u64;
        let surface_normal_count = if model.surface_normals.is_some() {
            num_points.max(1)
        } else {
            1
        };
        let surface_normals_size = (surface_normal_count * mem::size_of::<[f32; 4]>()) as u64;
        let attrs_size = (num_points * attr_dim * mem::size_of::<f32>()) as u64;
        let adj_size = (num_adjacency * mem::size_of::<u32>()) as u64;
        // An isolated PowerFoam site has a valid empty neighbor array, but
        // Vulkan does not have zero-sized buffers. Keep one unread dummy word
        // while retaining the logical edge count in `num_adjacency`.
        let adj_buffer_size = adj_size.max(mem::size_of::<u32>() as u64);
        let adj_off_size = (adjacency.offsets.len() * mem::size_of::<u32>()) as u64;
        // The mutable tree's fixed leaf buckets panic on point clouds with
        // many repeated coordinates along one axis (regular grids, planes,
        // and quantized scans). The immutable builder explicitly supports
        // that distribution and assigns each input row its original index.
        let point_index = PointIndex::new(&model.points, model.radii.as_deref());

        // Device buffers
        let points_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-points",
            size: points_size,
            memory: gpu::Memory::Device,
        });
        let surface_normals_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-surface-normals",
            size: surface_normals_size,
            memory: gpu::Memory::Device,
        });
        let attributes_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-attributes",
            size: attrs_size,
            memory: gpu::Memory::Device,
        });
        let point_adjacency_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency",
            size: adj_buffer_size,
            memory: gpu::Memory::Device,
        });
        let point_adjacency_offsets_buf = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency-offsets",
            size: adj_off_size,
            memory: gpu::Memory::Device,
        });

        // Upload buffers
        let points_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-points-upload",
            size: points_size,
            memory: gpu::Memory::Upload,
        });
        let surface_normals_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-surface-normals-upload",
            size: surface_normals_size,
            memory: gpu::Memory::Upload,
        });
        let attributes_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-attributes-upload",
            size: attrs_size,
            memory: gpu::Memory::Upload,
        });
        let adjacency_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency-upload",
            size: adj_buffer_size,
            memory: gpu::Memory::Upload,
        });
        let adjacency_offsets_stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-adjacency-offsets-upload",
            size: adj_off_size,
            memory: gpu::Memory::Upload,
        });

        // Fill staging buffers
        unsafe {
            // Points: write as `[f32; 4]` to match WGSL `array<vec4<f32>>` layout.
            // xyz = position; w = radius (Power Foam weight, 0 for plain Voronoi).
            // Density is read by the shader from `attributes`, not from here.
            let dst_points =
                slice::from_raw_parts_mut(points_stage.data() as *mut [f32; 4], num_points);
            let radii = model.radii.as_deref();
            for (i, dst) in dst_points.iter_mut().enumerate() {
                let p = model.points[i];
                dst[0] = p.x;
                dst[1] = p.y;
                dst[2] = p.z;
                dst[3] = radii.map_or(0.0, |r| r[i]);
            }

            match model.surface_normals.as_deref() {
                Some(normals) => {
                    let dst_normals = slice::from_raw_parts_mut(
                        surface_normals_stage.data() as *mut [f32; 4],
                        surface_normal_count,
                    );
                    dst_normals.fill([0.0; 4]);
                    for (dst, &normal) in dst_normals.iter_mut().zip(normals) {
                        *dst = normal.normalize().extend(0.0).to_array();
                    }
                }
                None => {
                    *(surface_normals_stage.data() as *mut [f32; 4]) = [0.0; 4];
                }
            }

            // Attributes: pack as [sh_coeffs..., density] per point
            // This matches the shader's expected layout
            let dst_attrs = slice::from_raw_parts_mut(
                attributes_stage.data() as *mut f32,
                num_points * attr_dim,
            );
            for i in 0..num_points {
                let base = i * attr_dim;
                let sh_len = sh_component_count * 3;
                let sh_base = i * sh_len;
                dst_attrs[base..base + sh_len]
                    .copy_from_slice(&model.sh_coefficients[sh_base..sh_base + sh_len]);
                dst_attrs[base + sh_len] = model.points[i].w;
            }

            // Adjacency: contiguous u32 array
            if num_adjacency > 0 {
                ptr::copy_nonoverlapping(
                    adjacency.neighbors.as_ptr(),
                    adjacency_stage.data() as *mut u32,
                    num_adjacency,
                );
            }

            // Adjacency offsets: contiguous u32 array
            ptr::copy_nonoverlapping(
                adjacency.offsets.as_ptr(),
                adjacency_offsets_stage.data() as *mut u32,
                adjacency.offsets.len(),
            );
        }

        // Encode transfers
        encoder.start();
        if let mut pass = encoder.transfer("radfoam-init") {
            if points_size > 0 {
                pass.copy_buffer_to_buffer(points_stage.at(0), points_buf.at(0), points_size);
            }
            if surface_normals_size > 0 {
                pass.copy_buffer_to_buffer(
                    surface_normals_stage.at(0),
                    surface_normals_buf.at(0),
                    surface_normals_size,
                );
            }
            if attrs_size > 0 {
                pass.copy_buffer_to_buffer(
                    attributes_stage.at(0),
                    attributes_buf.at(0),
                    attrs_size,
                );
            }
            if adj_size > 0 {
                pass.copy_buffer_to_buffer(
                    adjacency_stage.at(0),
                    point_adjacency_buf.at(0),
                    adj_size,
                );
            }
            if adj_off_size > 0 {
                pass.copy_buffer_to_buffer(
                    adjacency_offsets_stage.at(0),
                    point_adjacency_offsets_buf.at(0),
                    adj_off_size,
                );
            }
        }

        let sync_point = context.submit(encoder);
        let _ = context.wait_for(&sync_point, !0);

        // Free staging buffers
        context.destroy_buffer(points_stage);
        context.destroy_buffer(surface_normals_stage);
        context.destroy_buffer(attributes_stage);
        context.destroy_buffer(adjacency_stage);
        context.destroy_buffer(adjacency_offsets_stage);

        Self {
            points_buf,
            surface_normals_buf,
            attributes_buf,
            point_adjacency_buf,
            point_adjacency_offsets_buf,
            point_index,
            sh_degree: model.sh_degree,
            attr_dim,
            num_points,
            num_adjacency,
            is_power_foam: model.radii.is_some(),
            is_oriented: model.surface_normals.is_some(),
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.points_buf);
        context.destroy_buffer(self.surface_normals_buf);
        context.destroy_buffer(self.attributes_buf);
        context.destroy_buffer(self.point_adjacency_buf);
        context.destroy_buffer(self.point_adjacency_offsets_buf);
    }

    /// Replaces oriented-surface normals without recreating unchanged point,
    /// attribute, adjacency, or host-index data.
    pub fn update_surface_normals(
        &self,
        normals: &[glam::Vec3],
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) {
        assert!(self.is_oriented, "cloud has no surface-normal buffer");
        assert_eq!(normals.len(), self.num_points);
        let size = (normals.len() * mem::size_of::<[f32; 4]>()) as u64;
        let stage = context.create_buffer(gpu::BufferDesc {
            name: "radfoam-surface-normals-update",
            size,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            let dst = slice::from_raw_parts_mut(stage.data() as *mut [f32; 4], normals.len());
            for (dst_normal, &normal) in dst.iter_mut().zip(normals) {
                *dst_normal = normal.normalize().extend(0.0).to_array();
            }
        }

        encoder.start();
        if let mut pass = encoder.transfer("radfoam-surface-normals-update") {
            pass.copy_buffer_to_buffer(stage.at(0), self.surface_normals_buf.at(0), size);
        }
        let sync_point = context.submit(encoder);
        let _ = context.wait_for(&sync_point, !0);
        context.destroy_buffer(stage);
    }

    /// Storage buffer view for point positions.
    pub fn points(&self) -> gpu::BufferPiece {
        self.points_buf.into()
    }

    /// Storage buffer view for optional oriented-surface unit normals.
    pub fn surface_normals(&self) -> gpu::BufferPiece {
        self.surface_normals_buf.into()
    }

    /// Storage buffer view for packed attributes.
    pub fn attributes(&self) -> gpu::BufferPiece {
        self.attributes_buf.into()
    }

    /// Storage buffer view for flattened adjacency indices.
    pub fn point_adjacency(&self) -> gpu::BufferPiece {
        self.point_adjacency_buf.into()
    }

    /// Storage buffer view for CSR offsets.
    pub fn point_adjacency_offsets(&self) -> gpu::BufferPiece {
        self.point_adjacency_offsets_buf.into()
    }

    /// Site whose Voronoi or power cell contains a local-space position.
    pub fn containing_point(&self, position: glam::Vec3) -> u32 {
        self.point_index.containing_point(position)
    }
}

#[cfg(test)]
mod point_index_tests {
    use super::PointIndex;

    #[test]
    fn weighted_index_finds_non_euclidean_power_cell() {
        let points = [
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(10.0, 0.0, 0.0, 1.0),
        ];
        let unweighted = PointIndex::new(&points, None);
        assert_eq!(unweighted.containing_point(glam::Vec3::ZERO), 0);

        let weighted = PointIndex::new(&points, Some(&[0.0, 20.0]));
        assert_eq!(weighted.containing_point(glam::Vec3::ZERO), 1);
    }

    #[test]
    fn weighted_index_matches_exhaustive_power_distance() {
        let points = (0..32)
            .map(|index| {
                let x = (index * 17 % 23) as f32 - 11.0;
                let y = (index * 11 % 19) as f32 - 9.0;
                let z = (index * 7 % 13) as f32 - 6.0;
                glam::Vec4::new(x, y, z, 1.0)
            })
            .collect::<Vec<_>>();
        let radii = (0..points.len())
            .map(|index| (index * 5 % 9) as f32 * 0.75)
            .collect::<Vec<_>>();
        let index = PointIndex::new(&points, Some(&radii));

        for query_index in 0..40 {
            let query = glam::Vec3::new(
                query_index as f32 * 0.37 - 7.0,
                query_index as f32 * -0.19 + 3.0,
                query_index as f32 * 0.11 - 2.0,
            );
            let expected = points
                .iter()
                .enumerate()
                .min_by(|&(left_index, left), &(right_index, right)| {
                    let left_distance =
                        query.distance_squared(left.truncate()) - radii[left_index].powi(2);
                    let right_distance =
                        query.distance_squared(right.truncate()) - radii[right_index].powi(2);
                    left_distance
                        .total_cmp(&right_distance)
                        .then_with(|| left_index.cmp(&right_index))
                })
                .unwrap()
                .0 as u32;
            assert_eq!(index.containing_point(query), expected);
        }
    }
}
