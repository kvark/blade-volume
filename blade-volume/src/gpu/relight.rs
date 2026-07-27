//! Hardware ray-traced forward renderer for relightable surfels.
//!
//! The other backends store what a point looked like. This one stores what it
//! is made of, and works out what it looks like from the environment it is
//! handed, so the same model can be put under a different light without being
//! rebuilt.
//!
//! Geometry is one triangle per surfel, instanced from a shared unit
//! primitive the way the Gaussian backend instances its icosahedron. The
//! triangle circumscribes the disc rather than being it, so the shader still
//! has to reject the corners; that is what keeps a surfel round.

use crate::{relight, shaders, CameraParams};
use blade_graphics as gpu;
use std::{mem, ptr, slice};

/// Half-width of the triangle that stands in for a unit disc.
///
/// An equilateral triangle circumscribing a unit circle touches it at the
/// midpoint of each edge, which puts the vertices at radius two and the edge
/// midpoints at radius one.
const CIRCUMSCRIBED: f32 = 1.732_050_8; // sqrt(3)

#[derive(Clone, Copy, Debug)]
pub struct RelightSettings {
    /// Radiance for rays that hit nothing.
    pub background_rgb: [f32; 3],
    /// Rays cast per shading point for the shadowed diffuse term.
    ///
    /// Zero keeps the analytic unshadowed irradiance, which is cheaper and
    /// free of noise. Anything above it buys shadowing and one bounce of
    /// indirect light together — they cannot be had separately, because each
    /// ray either reaches the environment or meets something, and what it
    /// meets is what lights the point instead.
    pub diffuse_samples: u32,
    /// Show the environment where nothing was hit, rather than
    /// [`background_rgb`]. What a path traced reference does, so a comparison
    /// against one is of the whole frame rather than of a mask.
    ///
    /// [`background_rgb`]: Self::background_rgb
    pub show_environment: bool,
}

impl Default for RelightSettings {
    fn default() -> Self {
        Self {
            background_rgb: [0.0; 3],
            diffuse_samples: 0,
            show_environment: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RelightParams {
    irradiance: [[f32; 4]; 9],
    background: [f32; 3],
    max_specular_level: f32,
    diffuse_samples: u32,
    frame_index: u32,
    show_environment: u32,
    pad: u32,
}

#[derive(blade_macros::ShaderData)]
struct RelightData {
    g_camera: CameraParams,
    g_params: RelightParams,
    g_tlas: gpu::AccelerationStructure,
    g_surfels: gpu::BufferPiece,
    g_materials: gpu::BufferPiece,
    g_specular: gpu::TextureView,
    g_sampler: gpu::Sampler,
    g_out: gpu::TextureView,
}

pub struct RelightTracer {
    pipeline: gpu::ComputePipeline,
    params: RelightParams,
    mesh_buf: gpu::Buffer,
    surfel_buf: gpu::Buffer,
    material_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    specular_texture: gpu::Texture,
    specular_view: gpu::TextureView,
    sampler: gpu::Sampler,
    blas: gpu::AccelerationStructure,
    tlas: gpu::AccelerationStructure,
}

/// Rotation taking `+Z` onto `normal`, as the columns of a 3x3.
///
/// The instance transform has to put the shared triangle into the surfel's
/// plane, and any rotation that lands `+Z` on the normal will do since the
/// primitive is rotationally symmetric about it.
fn basis_from_normal(normal: glam::Vec3) -> glam::Mat3 {
    let up = if normal.z.abs() < 0.9 {
        glam::Vec3::Z
    } else {
        glam::Vec3::X
    };
    let tangent = up.cross(normal).normalize();
    let bitangent = normal.cross(tangent);
    glam::Mat3::from_cols(tangent, bitangent, normal)
}

impl RelightTracer {
    pub fn new(
        model: &relight::RelightModel,
        environment: &relight::Environment,
        specular: &relight::SpecularEnvironment,
        settings: RelightSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        model.validate().expect("invalid relightable model");
        assert!(
            specular.levels.len() == relight::SPECULAR_LEVELS as usize,
            "expected {} prefiltered levels, got {}",
            relight::SPECULAR_LEVELS,
            specular.levels.len()
        );

        let source = shaders::compose(shaders::RELIGHT);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let layout = <RelightData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "relight",
            data_layouts: &[&layout],
            compute: shader.at("trace_main"),
        });

        // One triangle, in the XY plane, circumscribing the unit disc.
        let vertices: [[f32; 3]; 3] = [
            [0.0, 2.0, 0.0],
            [-CIRCUMSCRIBED, -1.0, 0.0],
            [CIRCUMSCRIBED, -1.0, 0.0],
        ];
        let indices: [[u32; 3]; 1] = [[0, 1, 2]];
        let vertex_size = mem::size_of_val(&vertices) as u64;
        let index_size = mem::size_of_val(&indices) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-primitive",
            size: vertex_size + index_size,
            memory: gpu::Memory::Device,
        });

        let surfel_size = (model.surfels.len() * mem::size_of::<relight::Surfel>()) as u64;
        let surfel_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-surfels",
            size: surfel_size,
            memory: gpu::Memory::Device,
        });
        let material_size = (model.materials.len() * mem::size_of::<relight::Material>()) as u64;
        let material_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-materials",
            size: material_size,
            memory: gpu::Memory::Device,
        });

        let meshes = [gpu::AccelerationStructureMesh {
            vertex_data: mesh_buf.at(0),
            vertex_format: gpu::VertexFormat::F32Vec3,
            vertex_stride: mem::size_of::<[f32; 3]>() as u32,
            vertex_count: vertices.len() as u32,
            index_data: mesh_buf.at(vertex_size),
            index_type: Some(gpu::IndexType::U32),
            triangle_count: indices.len() as u32,
            transform_data: gpu::Buffer::default().at(0),
            // The disc test lives in the shader, so hits are candidates.
            is_opaque: false,
        }];
        let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
        let blas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "relight-blas",
            ty: gpu::AccelerationStructureType::BottomLevel,
            size: blas_sizes.data,
        });

        let instances = model
            .surfels
            .iter()
            .map(|surfel| {
                let normal = glam::Vec3::from(surfel.normal);
                let m = basis_from_normal(normal) * surfel.radius;
                gpu::AccelerationStructureInstance {
                    acceleration_structure_index: 0,
                    transform: mint::ColumnMatrix3x4 {
                        x: m.x_axis.into(),
                        y: m.y_axis.into(),
                        z: m.z_axis.into(),
                        w: glam::Vec3::from(surfel.center).into(),
                    }
                    .into(),
                    mask: 0xFF,
                    custom_index: 0,
                }
            })
            .collect::<Vec<_>>();
        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&instances, &[blas]);
        let count = model.surfels.len() as u32;
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(count);
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "relight-tlas",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
        });

        let format = gpu::TextureFormat::Rgba32Float;
        let specular_extent = gpu::Extent {
            width: specular.width as u32,
            height: specular.height as u32,
            depth: 1,
        };
        let specular_texture = context.create_texture(gpu::TextureDesc {
            name: "relight-specular",
            format,
            size: specular_extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: relight::SPECULAR_LEVELS,
            mip_level_count: 1,
            usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let specular_view = context.create_texture_view(
            specular_texture,
            gpu::TextureViewDesc {
                name: "relight-specular",
                format,
                dimension: gpu::ViewDimension::D2Array,
                subresources: &Default::default(),
            },
        );
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "relight-specular",
            address_modes: [
                // Longitude wraps around; latitude stops at the poles.
                gpu::AddressMode::Repeat,
                gpu::AddressMode::ClampToEdge,
                gpu::AddressMode::ClampToEdge,
            ],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let tlas_scratch_offset =
            (blas_sizes.scratch | (gpu::limits::ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT - 1)) + 1;
        let scratch_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-scratch",
            size: tlas_scratch_offset + tlas_sizes.scratch,
            memory: gpu::Memory::Device,
        });

        let level_size = specular.level_bytes() as u64;
        let stage_size = vertex_size + index_size + surfel_size + material_size;
        let stage = context.create_buffer(gpu::BufferDesc {
            name: "relight-stage",
            size: stage_size,
            memory: gpu::Memory::Upload,
        });
        let specular_stage = context.create_buffer(gpu::BufferDesc {
            name: "relight-specular-stage",
            size: level_size * relight::SPECULAR_LEVELS as u64,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(vertices.as_ptr(), stage.data() as *mut [f32; 3], 3);
            ptr::copy_nonoverlapping(
                indices.as_ptr(),
                stage.data().add(vertex_size as usize) as *mut [u32; 3],
                1,
            );
            let surfels = slice::from_raw_parts_mut(
                stage.data().add((vertex_size + index_size) as usize) as *mut relight::Surfel,
                model.surfels.len(),
            );
            surfels.copy_from_slice(&model.surfels);
            let materials = slice::from_raw_parts_mut(
                stage
                    .data()
                    .add((vertex_size + index_size + surfel_size) as usize)
                    as *mut relight::Material,
                model.materials.len(),
            );
            materials.copy_from_slice(&model.materials);

            for (level, plane) in specular.levels.iter().enumerate() {
                ptr::copy_nonoverlapping(
                    plane.as_ptr(),
                    specular_stage.data().add(level * level_size as usize) as *mut [f32; 4],
                    plane.len(),
                );
            }
        }

        encoder.start();
        encoder.init_texture(specular_texture);
        if let mut pass = encoder.transfer("relight-init") {
            pass.copy_buffer_to_buffer(stage.at(0), mesh_buf.at(0), vertex_size + index_size);
            pass.copy_buffer_to_buffer(
                stage.at(vertex_size + index_size),
                surfel_buf.at(0),
                surfel_size,
            );
            pass.copy_buffer_to_buffer(
                stage.at(vertex_size + index_size + surfel_size),
                material_buf.at(0),
                material_size,
            );
            for level in 0..relight::SPECULAR_LEVELS {
                pass.copy_buffer_to_texture(
                    specular_stage.at(level as u64 * level_size),
                    specular_extent.width * mem::size_of::<[f32; 4]>() as u32,
                    gpu::TexturePiece {
                        texture: specular_texture,
                        mip_level: 0,
                        array_layer: level,
                        origin: [0; 3],
                    },
                    specular_extent,
                );
            }
        }
        if let mut pass = encoder.acceleration_structure("relight-bottom") {
            pass.build_bottom_level(blas, &meshes, scratch_buf.at(0));
        }
        if let mut pass = encoder.acceleration_structure("relight-top") {
            pass.build_top_level(
                tlas,
                &[blas],
                count,
                instance_buf.at(0),
                scratch_buf.at(tlas_scratch_offset),
            );
        }
        let sync_point = context.submit(encoder);
        let _ = context.wait_for(&sync_point, !0);

        context.destroy_buffer(scratch_buf);
        context.destroy_buffer(stage);
        context.destroy_buffer(specular_stage);

        Self {
            pipeline,
            params: RelightParams {
                irradiance: environment.diffuse_irradiance(),
                background: settings.background_rgb,
                max_specular_level: (relight::SPECULAR_LEVELS - 1) as f32,
                diffuse_samples: settings.diffuse_samples,
                frame_index: 0,
                show_environment: settings.show_environment as u32,
                pad: 0,
            },
            mesh_buf,
            surfel_buf,
            material_buf,
            instance_buf,
            specular_texture,
            specular_view,
            sampler,
            blas,
            tlas,
        }
    }

    pub fn dispatch(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        output: gpu::TextureView,
        camera: CameraParams,
        size: [u32; 2],
    ) {
        assert!(size[0] > 0 && size[1] > 0, "render size must be non-zero");
        // Advanced per dispatch, so an accumulating viewer converges rather
        // than settling on one noisy estimate.
        self.params.frame_index = self.params.frame_index.wrapping_add(1);
        let mut pass = encoder.compute("relight");
        let mut pen = pass.with(&self.pipeline);
        pen.bind(
            0,
            &RelightData {
                g_camera: camera,
                g_params: self.params,
                g_tlas: self.tlas,
                g_surfels: self.surfel_buf.at(0),
                g_materials: self.material_buf.at(0),
                g_specular: self.specular_view,
                g_sampler: self.sampler,
                g_out: output,
            },
        );
        pen.dispatch([size[0].div_ceil(8), size[1].div_ceil(8), 1]);
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
        context.destroy_acceleration_structure(self.tlas);
        context.destroy_acceleration_structure(self.blas);
        context.destroy_sampler(self.sampler);
        context.destroy_texture_view(self.specular_view);
        context.destroy_texture(self.specular_texture);
        context.destroy_buffer(self.instance_buf);
        context.destroy_buffer(self.material_buf);
        context.destroy_buffer(self.surfel_buf);
        context.destroy_buffer(self.mesh_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_layout_matches_the_shader() {
        // `vec4` arrays pack tight; the trailing `vec3` plus `f32` fills one
        // more and the four `u32` another, so the struct is eleven of them.
        assert_eq!(mem::size_of::<RelightParams>(), 11 * 16);
    }

    #[test]
    fn a_surfel_is_eight_floats() {
        assert_eq!(mem::size_of::<relight::Surfel>(), 32);
        assert_eq!(mem::size_of::<relight::Material>(), 32);
    }

    #[test]
    fn the_basis_lands_z_on_the_normal() {
        for normal in [
            glam::Vec3::Y,
            glam::Vec3::Z,
            glam::Vec3::new(0.3, -0.5, 0.81).normalize(),
        ] {
            let basis = basis_from_normal(normal);
            assert!(
                (basis * glam::Vec3::Z - normal).length() < 1.0e-5,
                "basis for {normal:?} sent +Z to {:?}",
                basis * glam::Vec3::Z
            );
            assert!(
                (basis.determinant() - 1.0).abs() < 1.0e-4,
                "basis for {normal:?} is not a rotation"
            );
        }
    }

    #[test]
    fn the_stand_in_triangle_contains_the_unit_disc() {
        // Every edge midpoint sits exactly on the circle, so the triangle
        // covers the disc without being much larger than it needs to be.
        let vertices = [
            glam::Vec2::new(0.0, 2.0),
            glam::Vec2::new(-CIRCUMSCRIBED, -1.0),
            glam::Vec2::new(CIRCUMSCRIBED, -1.0),
        ];
        for index in 0..3 {
            let midpoint = 0.5 * (vertices[index] + vertices[(index + 1) % 3]);
            assert!(
                (midpoint.length() - 1.0).abs() < 1.0e-4,
                "edge {index} touches at radius {}",
                midpoint.length()
            );
        }
    }
}
