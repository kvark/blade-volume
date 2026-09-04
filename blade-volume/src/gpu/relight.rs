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

use crate::{relight, shaders, CameraParams, PointCloudModel};
use blade_graphics as gpu;
use std::{mem, ptr, slice};

/// Half-width of the triangle that stands in for a unit disc.
///
/// An equilateral triangle circumscribing a unit circle touches it at the
/// midpoint of each edge, which puts the vertices at radius two and the edge
/// midpoints at radius one.
const CIRCUMSCRIBED: f32 = 1.732_050_8; // sqrt(3)

// Responses below this are both visually negligible and expensive: retaining
// their 4-5 sigma proxy tails multiplies ray-query candidates. This cutoff is
// selected by the fixed five-cloud held-light gate.
const GAUSSIAN_MIN_ALPHA: f32 = 3.0e-2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PbrGaussian {
    rotation: [f32; 4],
    scale: [f32; 3],
    opacity: f32,
}

struct PreparedGeometry {
    surfels: Vec<relight::Surfel>,
    gaussians: Vec<PbrGaussian>,
    materials: Vec<relight::Material>,
    instances: Vec<gpu::AccelerationStructureInstance>,
    kernel: u32,
}

#[derive(Clone, Copy)]
enum GeometrySource<'a> {
    Surface(&'a relight::RelightModel),
    Gaussian(&'a PointCloudModel),
}

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
    ///
    /// A finite point light needs exactly one direction, so any non-zero value
    /// enables one deterministic finite-distance visibility ray.
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
    env_width: u32,
    env_height: u32,
    kernel: u32,
    pad: [u32; 2],
    point_position: [f32; 3],
    point_exponent: f32,
    point_direction: [f32; 3],
    point_enabled: u32,
    point_intensity: [f32; 3],
    point_pad: f32,
}

#[derive(blade_macros::ShaderData)]
struct RelightData {
    g_camera: CameraParams,
    g_params: RelightParams,
    g_tlas: gpu::AccelerationStructure,
    g_surfels: gpu::BufferPiece,
    g_gaussians: gpu::BufferPiece,
    g_materials: gpu::BufferPiece,
    g_alias: gpu::BufferPiece,
    g_specular: gpu::TextureView,
    g_sampler: gpu::Sampler,
    g_out: gpu::TextureView,
}

pub struct RelightTracer {
    pipeline: gpu::ComputePipeline,
    params: RelightParams,
    mesh_buf: gpu::Buffer,
    surfel_buf: gpu::Buffer,
    gaussian_buf: gpu::Buffer,
    material_buf: gpu::Buffer,
    alias_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    stage_buf: gpu::Buffer,
    scratch_buf: gpu::Buffer,
    specular_texture: gpu::Texture,
    specular_view: gpu::TextureView,
    sampler: gpu::Sampler,
    blas: gpu::AccelerationStructure,
    tlas: gpu::AccelerationStructure,
    surfel_stage_offset: u64,
    material_stage_offset: u64,
    tlas_scratch_offset: u64,
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

fn make_instances(surfels: &[relight::Surfel]) -> Vec<gpu::AccelerationStructureInstance> {
    surfels
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
        .collect()
}

fn prepare_geometry(source: GeometrySource<'_>) -> PreparedGeometry {
    let GeometrySource::Gaussian(gaussian) = source else {
        let GeometrySource::Surface(model) = source else {
            unreachable!()
        };
        model.validate().expect("invalid relightable model");
        return PreparedGeometry {
            surfels: model.surfels.clone(),
            gaussians: vec![PbrGaussian {
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
                opacity: 0.0,
            }],
            materials: model.materials.clone(),
            instances: make_instances(&model.surfels),
            kernel: model.kernel as u32,
        };
    };
    gaussian
        .validate()
        .unwrap_or_else(|error| panic!("invalid PBR Gaussian model: {error}"));
    let transforms = gaussian.transforms.as_ref().unwrap();
    let pbr = transforms
        .pbr
        .as_ref()
        .expect("PBR Gaussian model requires explicit PBR attributes");

    let mut surfels = Vec::new();
    let mut gaussians = Vec::new();
    let mut instances = Vec::new();
    for (index, &point) in gaussian.points.iter().enumerate() {
        if point.w <= GAUSSIAN_MIN_ALPHA {
            continue;
        }
        let rotation = transforms.rotations[index];
        let scale = transforms.scales[index];
        let surfel = relight::Surfel {
            center: point.truncate().to_array(),
            radius: 3.0 * (scale.x * scale.y * scale.z).cbrt(),
            normal: pbr.normals[index].to_array(),
            material: pbr.material_indices[index],
        };
        surfels.push(surfel);
        gaussians.push(PbrGaussian {
            rotation: rotation.into(),
            scale: scale.into(),
            opacity: point.w,
        });

        let support = (2.0 * (point.w / GAUSSIAN_MIN_ALPHA).ln()).sqrt();
        let matrix = glam::Mat3::from_quat(rotation) * glam::Mat3::from_diagonal(support * scale);
        instances.push(gpu::AccelerationStructureInstance {
            acceleration_structure_index: 0,
            transform: mint::ColumnMatrix3x4 {
                x: matrix.x_axis.into(),
                y: matrix.y_axis.into(),
                z: matrix.z_axis.into(),
                w: point.truncate().into(),
            }
            .into(),
            mask: 0xFF,
            custom_index: 0,
        });
    }
    assert!(
        !surfels.is_empty(),
        "PBR Gaussian geometry has no particles above alpha {GAUSSIAN_MIN_ALPHA}"
    );
    PreparedGeometry {
        surfels,
        gaussians,
        materials: pbr.materials.clone(),
        instances,
        kernel: 2,
    }
}

impl RelightTracer {
    /// The prefiltered ladder as one array texture, a layer per roughness.
    fn create_specular(
        context: &gpu::Context,
        specular: &relight::SpecularEnvironment,
    ) -> (gpu::Texture, gpu::TextureView, gpu::Extent) {
        let format = gpu::TextureFormat::Rgba32Float;
        let extent = gpu::Extent {
            width: specular.width as u32,
            height: specular.height as u32,
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "relight-specular",
            format,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: relight::SPECULAR_LEVELS,
            mip_level_count: 1,
            usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "relight-specular",
                format,
                dimension: gpu::ViewDimension::D2Array,
                subresources: &Default::default(),
            },
        );
        (texture, view, extent)
    }

    pub fn new(
        model: &relight::RelightModel,
        environment: &relight::Environment,
        specular: &relight::SpecularEnvironment,
        settings: RelightSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        Self::new_impl(
            GeometrySource::Surface(model),
            environment,
            specular,
            settings,
            context,
            encoder,
        )
    }

    /// Render learned anisotropic Gaussian geometry with explicit PBR surface
    /// normals and materials. Particle correspondence is by index; transparent
    /// Gaussians are omitted from the acceleration structure. Non-zero diffuse
    /// samples use the same coupled visibility and one-bounce path as compact
    /// surfels.
    pub fn new_gaussian(
        model: &PointCloudModel,
        environment: &relight::Environment,
        specular: &relight::SpecularEnvironment,
        settings: RelightSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        Self::new_impl(
            GeometrySource::Gaussian(model),
            environment,
            specular,
            settings,
            context,
            encoder,
        )
    }

    fn new_impl(
        source: GeometrySource<'_>,
        environment: &relight::Environment,
        specular: &relight::SpecularEnvironment,
        settings: RelightSettings,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
        assert!(
            specular.levels.len() == relight::SPECULAR_LEVELS as usize,
            "expected {} prefiltered levels, got {}",
            relight::SPECULAR_LEVELS,
            specular.levels.len()
        );

        let shader_source = shaders::compose(shaders::RELIGHT);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &shader_source,
            naga_module: None,
        });
        let layout = <RelightData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "relight",
            data_layouts: &[&layout],
            compute: shader.at("trace_main"),
        });

        let geometry = prepare_geometry(source);
        // Surface particles use the original circumscribed triangle. A full
        // Gaussian needs a closed conservative proxy, so use the same
        // circumscribed icosahedron as the appearance renderer.
        let (vertices, indices) = if geometry.kernel == 2 {
            let shape = crate::Icosahedron::new(1.0);
            (
                shape.vertices.to_vec(),
                shape
                    .triangles
                    .iter()
                    .map(|triangle| triangle.map(u32::from))
                    .collect(),
            )
        } else {
            (
                vec![
                    [0.0, 2.0, 0.0],
                    [-CIRCUMSCRIBED, -1.0, 0.0],
                    [CIRCUMSCRIBED, -1.0, 0.0],
                ],
                vec![[0, 1, 2]],
            )
        };
        let vertex_size = mem::size_of_val(vertices.as_slice()) as u64;
        let index_size = mem::size_of_val(indices.as_slice()) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-primitive",
            size: vertex_size + index_size,
            memory: gpu::Memory::Device,
        });

        let surfel_size = mem::size_of_val(geometry.surfels.as_slice()) as u64;
        let surfel_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-surfels",
            size: surfel_size,
            memory: gpu::Memory::Device,
        });
        let gaussian_size = mem::size_of_val(geometry.gaussians.as_slice()) as u64;
        let gaussian_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-gaussians",
            size: gaussian_size,
            memory: gpu::Memory::Device,
        });
        let material_size = mem::size_of_val(geometry.materials.as_slice()) as u64;
        let material_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-materials",
            size: material_size,
            memory: gpu::Memory::Device,
        });
        // Built here rather than taken as an argument: it depends on nothing
        // but the environment, and having one way to build it is what keeps
        // the density the shader divides by the same one it drew from.
        let sampler_table = relight::EnvironmentSampler::build(environment);
        let alias_size =
            (sampler_table.entries.len() * mem::size_of::<relight::AliasEntry>()) as u64;
        let alias_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-alias",
            size: alias_size,
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
            updatable: false,
        });

        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&geometry.instances, &[blas]);
        let count = geometry.surfels.len() as u32;
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(count);
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "relight-tlas",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
            updatable: false,
        });

        let (specular_texture, specular_view, specular_extent) =
            Self::create_specular(context, specular);
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
        let stage_size =
            vertex_size + index_size + surfel_size + gaussian_size + material_size + alias_size;
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
            ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                stage.data() as *mut [f32; 3],
                vertices.len(),
            );
            ptr::copy_nonoverlapping(
                indices.as_ptr(),
                stage.data().add(vertex_size as usize) as *mut [u32; 3],
                indices.len(),
            );
            let surfels = slice::from_raw_parts_mut(
                stage.data().add((vertex_size + index_size) as usize) as *mut relight::Surfel,
                geometry.surfels.len(),
            );
            surfels.copy_from_slice(&geometry.surfels);
            let gaussians = slice::from_raw_parts_mut(
                stage
                    .data()
                    .add((vertex_size + index_size + surfel_size) as usize)
                    as *mut PbrGaussian,
                geometry.gaussians.len(),
            );
            gaussians.copy_from_slice(&geometry.gaussians);
            let materials = slice::from_raw_parts_mut(
                stage
                    .data()
                    .add((vertex_size + index_size + surfel_size + gaussian_size) as usize)
                    as *mut relight::Material,
                geometry.materials.len(),
            );
            materials.copy_from_slice(&geometry.materials);
            let alias = slice::from_raw_parts_mut(
                stage.data().add(
                    (vertex_size + index_size + surfel_size + gaussian_size + material_size)
                        as usize,
                ) as *mut relight::AliasEntry,
                sampler_table.entries.len(),
            );
            alias.copy_from_slice(&sampler_table.entries);

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
                gaussian_buf.at(0),
                gaussian_size,
            );
            pass.copy_buffer_to_buffer(
                stage.at(vertex_size + index_size + surfel_size + gaussian_size),
                material_buf.at(0),
                material_size,
            );
            pass.copy_buffer_to_buffer(
                stage.at(vertex_size + index_size + surfel_size + gaussian_size + material_size),
                alias_buf.at(0),
                alias_size,
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
                env_width: environment.width as u32,
                env_height: environment.height as u32,
                kernel: geometry.kernel,
                pad: [0; 2],
                point_position: [0.0; 3],
                point_exponent: 0.0,
                point_direction: [0.0, 0.0, 1.0],
                point_enabled: 0,
                point_intensity: [0.0; 3],
                point_pad: 0.0,
            },
            mesh_buf,
            surfel_buf,
            gaussian_buf,
            material_buf,
            alias_buf,
            instance_buf,
            stage_buf: stage,
            scratch_buf,
            specular_texture,
            specular_view,
            sampler,
            blas,
            tlas,
            surfel_stage_offset: vertex_size + index_size,
            material_stage_offset: vertex_size + index_size + surfel_size + gaussian_size,
            tlas_scratch_offset,
        }
    }

    /// Replace the shared material table without rebuilding geometry or
    /// lighting. The material count must stay fixed.
    pub fn update_materials(
        &mut self,
        materials: &[relight::Material],
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) {
        let material_size = mem::size_of_val(materials) as u64;
        assert_eq!(
            material_size,
            self.material_buf.size(),
            "relight material update must preserve the material count"
        );
        unsafe {
            let target = slice::from_raw_parts_mut(
                self.stage_buf
                    .data()
                    .add(self.material_stage_offset as usize)
                    as *mut relight::Material,
                materials.len(),
            );
            target.copy_from_slice(materials);
        }
        encoder.start();
        if let mut pass = encoder.transfer("relight-materials-update") {
            pass.copy_buffer_to_buffer(
                self.stage_buf.at(self.material_stage_offset),
                self.material_buf.at(0),
                material_size,
            );
        }
        let sync_point = context.submit(encoder);
        let _ = context.wait_for(&sync_point, !0);
    }

    /// Replace particle geometry while retaining materials, lighting, and the
    /// shared triangle BLAS. The particle count must stay fixed.
    pub fn update_surfels(
        &mut self,
        surfels: &[relight::Surfel],
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) {
        encoder.start();
        self.record_surfels_update(surfels, context, encoder);
        let sync_point = context.submit(encoder);
        let _ = context.wait_for(&sync_point, !0);
    }

    /// Record a particle-geometry update into an encoder that is already
    /// active. Previously submitted work using this tracer must be finished;
    /// the caller must submit and finish this work before recording another
    /// update or changing the upload data.
    pub fn record_surfels_update(
        &mut self,
        surfels: &[relight::Surfel],
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) {
        assert_ne!(
            self.params.kernel, 2,
            "update the learned Gaussian geometry rather than its staging surfels"
        );
        let surfel_size = mem::size_of_val(surfels) as u64;
        assert_eq!(
            surfel_size,
            self.surfel_buf.size(),
            "relight surfel update must preserve the particle count"
        );
        unsafe {
            let target = slice::from_raw_parts_mut(
                self.stage_buf.data().add(self.surfel_stage_offset as usize)
                    as *mut relight::Surfel,
                surfels.len(),
            );
            target.copy_from_slice(surfels);
        }
        let instances = make_instances(surfels);
        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&instances, &[self.blas]);

        if let mut pass = encoder.transfer("relight-surfels-update") {
            pass.copy_buffer_to_buffer(
                self.stage_buf.at(self.surfel_stage_offset),
                self.surfel_buf.at(0),
                surfel_size,
            );
        }
        if let mut pass = encoder.acceleration_structure("relight-top-update") {
            pass.build_top_level(
                self.tlas,
                &[self.blas],
                surfels.len() as u32,
                instance_buf.at(0),
                self.scratch_buf.at(self.tlas_scratch_offset),
            );
        }
        context.destroy_buffer(self.instance_buf);
        self.instance_buf = instance_buf;
    }

    /// Put the model under a different light.
    ///
    /// The whole claim of this representation is that the environment is an
    /// argument rather than part of the asset, and this is where that gets
    /// paid off: the surfels, the materials and both acceleration structures
    /// are untouched, and only what describes the light is replaced. Nothing
    /// is rebuilt that does not depend on the environment, which is why this
    /// costs a couple of uploads rather than a conversion.
    ///
    /// The prefiltered ladder is passed in rather than computed here, because
    /// prefiltering is seconds of CPU work and the caller is the one that
    /// knows whether it can afford to do it now.
    pub fn set_environment(
        &mut self,
        environment: &relight::Environment,
        specular: &relight::SpecularEnvironment,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) {
        assert!(
            specular.levels.len() == relight::SPECULAR_LEVELS as usize,
            "expected {} prefiltered levels, got {}",
            relight::SPECULAR_LEVELS,
            specular.levels.len()
        );

        let sampler_table = relight::EnvironmentSampler::build(environment);
        let alias_size =
            (sampler_table.entries.len() * mem::size_of::<relight::AliasEntry>()) as u64;
        let alias_buf = context.create_buffer(gpu::BufferDesc {
            name: "relight-alias",
            size: alias_size,
            memory: gpu::Memory::Device,
        });
        let (specular_texture, specular_view, specular_extent) =
            Self::create_specular(context, specular);

        let level_size = specular.level_bytes() as u64;
        let stage = context.create_buffer(gpu::BufferDesc {
            name: "relight-environment-stage",
            size: alias_size + level_size * relight::SPECULAR_LEVELS as u64,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            let alias = slice::from_raw_parts_mut(
                stage.data() as *mut relight::AliasEntry,
                sampler_table.entries.len(),
            );
            alias.copy_from_slice(&sampler_table.entries);
            for (level, plane) in specular.levels.iter().enumerate() {
                ptr::copy_nonoverlapping(
                    plane.as_ptr(),
                    stage
                        .data()
                        .add(alias_size as usize + level * level_size as usize)
                        as *mut [f32; 4],
                    plane.len(),
                );
            }
        }

        encoder.start();
        encoder.init_texture(specular_texture);
        if let mut pass = encoder.transfer("relight-environment") {
            pass.copy_buffer_to_buffer(stage.at(0), alias_buf.at(0), alias_size);
            for level in 0..relight::SPECULAR_LEVELS {
                pass.copy_buffer_to_texture(
                    stage.at(alias_size + level as u64 * level_size),
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
        let sync_point = context.submit(encoder);
        // Everything submitted before this is done too, so what the old
        // bindings pointed at is no longer being read.
        let _ = context.wait_for(&sync_point, !0);

        context.destroy_buffer(stage);
        context.destroy_texture_view(self.specular_view);
        context.destroy_texture(self.specular_texture);
        context.destroy_buffer(self.alias_buf);

        self.alias_buf = alias_buf;
        self.specular_texture = specular_texture;
        self.specular_view = specular_view;
        self.params.irradiance = environment.diffuse_irradiance();
        self.params.env_width = environment.width as u32;
        self.params.env_height = environment.height as u32;
        self.params.point_enabled = 0;
        self.params.frame_index = 0;
    }

    /// Replace environment illumination with one finite-distance emitter.
    ///
    /// The emitter is evaluated analytically at every particle hit. This
    /// changes only the uniform light parameters; geometry, materials,
    /// acceleration structures, and shader pipelines remain untouched.
    /// Environment sampling and the prefiltered specular ladder are bypassed
    /// while the point light is active. Any non-zero
    /// [`RelightSettings::diffuse_samples`] enables its single visibility ray.
    pub fn set_point_light(&mut self, light: relight::PointLight) {
        let direction = glam::Vec3::from(light.direction);
        assert!(
            light.position.iter().all(|value| value.is_finite())
                && light
                    .intensity
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && light.exponent.is_finite()
                && light.exponent >= 0.0
                && (light.exponent == 0.0 || direction.length_squared() > f32::EPSILON),
            "invalid point light",
        );
        self.params.point_position = light.position;
        self.params.point_exponent = light.exponent;
        self.params.point_direction = if light.exponent == 0.0 {
            [0.0, 0.0, 1.0]
        } else {
            direction.normalize().to_array()
        };
        self.params.point_enabled = 1;
        self.params.point_intensity = light.intensity;
        self.params.frame_index = 0;
    }

    /// Rays cast per shading point for the shadowed environment diffuse term,
    /// or zero/one visibility for a finite point light.
    ///
    /// See [`RelightSettings::diffuse_samples`]. Zero is the analytic path.
    ///
    /// [`RelightSettings::diffuse_samples`]: struct.RelightSettings.html#structfield.diffuse_samples
    pub fn diffuse_samples_mut(&mut self) -> &mut u32 {
        &mut self.params.diffuse_samples
    }

    pub fn diffuse_samples(&self) -> u32 {
        self.params.diffuse_samples
    }

    /// Restart the deterministic sampling sequence used by shadowed renders.
    ///
    /// This lets two otherwise identical frame sequences use identical sample
    /// directions, which is required when their image errors are compared.
    pub fn reset_sampling(&mut self) {
        self.params.frame_index = 0;
    }

    /// Radiance for rays that hit nothing, when the environment is not shown.
    ///
    /// Rendering the same pose against two of these measures how much of the
    /// frame the geometry actually covers: the part that changes is the part
    /// that missed everything.
    pub fn set_background(&mut self, rgb: [f32; 3]) {
        self.params.background = rgb;
    }

    pub fn set_show_environment(&mut self, show: bool) {
        self.params.show_environment = show as u32;
    }

    pub fn show_environment(&self) -> bool {
        self.params.show_environment != 0
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
                g_gaussians: self.gaussian_buf.at(0),
                g_materials: self.material_buf.at(0),
                g_alias: self.alias_buf.at(0),
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
        context.destroy_buffer(self.scratch_buf);
        context.destroy_buffer(self.stage_buf);
        context.destroy_buffer(self.instance_buf);
        context.destroy_buffer(self.alias_buf);
        context.destroy_buffer(self.material_buf);
        context.destroy_buffer(self.gaussian_buf);
        context.destroy_buffer(self.surfel_buf);
        context.destroy_buffer(self.mesh_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pbr_gaussian_with_opacities(opacities: &[f32]) -> PointCloudModel {
        let points = opacities
            .iter()
            .enumerate()
            .map(|(index, &opacity)| glam::Vec4::new(index as f32, 0.0, 0.0, opacity))
            .collect::<Vec<_>>();
        PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            sh_degree: 0,
            transforms: Some(crate::Transforms {
                rotations: vec![glam::Quat::IDENTITY; points.len()],
                scales: vec![glam::Vec3::ONE; points.len()],
                pbr: Some(crate::PbrAttributes {
                    normals: vec![glam::Vec3::Z; points.len()],
                    material_indices: (0..points.len() as u32).collect(),
                    materials: (0..points.len())
                        .map(|index| relight::Material {
                            albedo: [index as f32; 3],
                            roughness: 0.5,
                            specular_f0: [0.04; 3],
                            _padding: 0.0,
                        })
                        .collect(),
                }),
            }),
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
            points,
        }
    }

    #[test]
    fn the_uniform_layout_matches_the_shader() {
        // `vec4` arrays pack tight, and everything after them is padded out to
        // land on a multiple of sixteen, which is what the binding requires.
        assert_eq!(mem::size_of::<RelightParams>(), 15 * 16);
    }

    #[test]
    fn a_surfel_is_eight_floats() {
        assert_eq!(mem::size_of::<relight::Surfel>(), 32);
        assert_eq!(mem::size_of::<relight::Material>(), 32);
        assert_eq!(mem::size_of::<PbrGaussian>(), 32);
    }

    #[test]
    fn bounded_gaussian_support_preserves_filtered_particle_order() {
        let model =
            pbr_gaussian_with_opacities(&[GAUSSIAN_MIN_ALPHA, GAUSSIAN_MIN_ALPHA.next_up(), 0.8]);
        let geometry = prepare_geometry(GeometrySource::Gaussian(&model));

        assert_eq!(geometry.surfels.len(), 2);
        assert_eq!(geometry.gaussians.len(), 2);
        assert_eq!(geometry.instances.len(), 2);
        assert_eq!(geometry.surfels[0].center, [1.0, 0.0, 0.0]);
        assert_eq!(geometry.surfels[1].center, [2.0, 0.0, 0.0]);
        assert_eq!(geometry.surfels[0].material, 1);
        assert_eq!(geometry.surfels[1].material, 2);
        assert_eq!(geometry.gaussians[0].opacity, GAUSSIAN_MIN_ALPHA.next_up());
        assert_eq!(geometry.gaussians[1].opacity, 0.8);
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
