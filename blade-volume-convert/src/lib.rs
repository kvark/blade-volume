use blade_volume as vol;
use std::path;

const SH_C0: f32 = 0.282_094_8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Gaussian,
    RadFoam,
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub output: OutputKind,
    pub density: f32,
    pub surface_density_scale: f32,
    pub interior_density_scale: f32,
    pub alpha_threshold: f32,
    pub ambient: glam::Vec3,
    pub seed: u64,
    pub surface_opacity: f32,
    pub interior_opacity: f32,
    pub surface_scale: f32,
    pub interior_scale: f32,
    pub surface_normal_scale: f32,
    /// Curvature exponent for per-triangle surface sampling. `0` keeps the
    /// uniform-by-area density. `>0` boosts the sample count on triangles
    /// that are at a high dihedral angle to their (vertex-shared) neighbours,
    /// at the cost of sparser sampling on flat regions. Total sample budget
    /// stays roughly the same; mass is just redistributed.
    pub curvature_boost: f32,
    /// Number of spring-relaxation iterations applied after Delaunay (RadFoam
    /// path only). `0` skips the relaxation entirely.
    pub lloyd_iterations: usize,
    /// Per-iteration step size for [`vol::lloyd_relax`].
    pub lloyd_step: f32,
    /// When `true`, populate `model.radii` from the post-Lloyd nearest-
    /// neighbour distance (Power Foam mode). When `false`, leave `radii`
    /// as `None` (plain Voronoi).
    pub assign_radii: bool,
    /// Multiplier for [`vol::radii_from_nearest_neighbour`].
    pub radius_factor: f32,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            output: OutputKind::Gaussian,
            density: 10.0,
            surface_density_scale: 1.0,
            interior_density_scale: 0.25,
            alpha_threshold: 0.01,
            ambient: glam::Vec3::splat(1.5),
            seed: 0,
            surface_opacity: 1.0,
            interior_opacity: 0.25,
            surface_scale: 1.0,
            interior_scale: 1.0,
            surface_normal_scale: 1.0,
            curvature_boost: 0.0,
            lloyd_iterations: 0,
            lloyd_step: 0.3,
            assign_radii: false,
            radius_factor: 0.5,
        }
    }
}

#[derive(Debug)]
pub enum ConvertError {
    Io(std::io::Error),
    Gltf(gltf::Error),
    InvalidDensity,
    MissingMeshData,
    UnsupportedPrimitiveMode,
    MissingOutputData,
}

impl From<std::io::Error> for ConvertError {
    fn from(err: std::io::Error) -> Self {
        ConvertError::Io(err)
    }
}

impl From<gltf::Error> for ConvertError {
    fn from(err: gltf::Error) -> Self {
        ConvertError::Gltf(err)
    }
}

struct MaterialInfo {
    base_color: glam::Vec4,
    base_color_texture: Option<Texture>,
}

struct Texture {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Texture {
    fn sample(&self, uv: glam::Vec2) -> glam::Vec4 {
        let u = uv.x.fract().clamp(0.0, 0.999_999);
        let v = uv.y.fract().clamp(0.0, 0.999_999);
        let x = u * (self.width as f32 - 1.0);
        let y = (1.0 - v) * (self.height as f32 - 1.0);
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let c00 = self.fetch(x0, y0);
        let c10 = self.fetch(x1, y0);
        let c01 = self.fetch(x0, y1);
        let c11 = self.fetch(x1, y1);

        let c0 = c00.lerp(c10, tx);
        let c1 = c01.lerp(c11, tx);
        c0.lerp(c1, ty)
    }

    fn fetch(&self, x: u32, y: u32) -> glam::Vec4 {
        let idx = ((y * self.width + x) * 4) as usize;
        let r = self.data[idx] as f32 / 255.0;
        let g = self.data[idx + 1] as f32 / 255.0;
        let b = self.data[idx + 2] as f32 / 255.0;
        let a = self.data[idx + 3] as f32 / 255.0;
        glam::Vec4::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)
    }
}

/// Per-triangle curvature proxy in `[0, 1]`. Higher means the triangle's
/// normal disagrees more with the normals of triangles sharing any of its
/// vertices. Computed by hashing each triangle into buckets keyed by its
/// vertex positions (snapped to a coarse grid to make exact-same-vertex
/// triangles co-locate), then per triangle averaging `1 - dot(n, n_j)`
/// across its neighbours.
fn compute_triangle_curvatures(triangles: &[Triangle]) -> Vec<f32> {
    use std::collections::HashMap;
    if triangles.is_empty() {
        return Vec::new();
    }

    // Bounding-box-derived snap epsilon so different mesh scales work.
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for t in triangles {
        for v in &[t.v0, t.v1, t.v2] {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    let diag = (max - min).length().max(1e-6);
    let snap = diag * 1e-5;

    let key = |v: glam::Vec3| -> (i64, i64, i64) {
        (
            (v.x / snap).round() as i64,
            (v.y / snap).round() as i64,
            (v.z / snap).round() as i64,
        )
    };

    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, t) in triangles.iter().enumerate() {
        for v in &[t.v0, t.v1, t.v2] {
            buckets.entry(key(*v)).or_default().push(i);
        }
    }

    let mut out = Vec::with_capacity(triangles.len());
    for (i, t) in triangles.iter().enumerate() {
        let mut sum_diff = 0.0_f32;
        let mut count = 0usize;
        for v in &[t.v0, t.v1, t.v2] {
            if let Some(neighbours) = buckets.get(&key(*v)) {
                for &j in neighbours {
                    if j == i {
                        continue;
                    }
                    let cos = t.normal.dot(triangles[j].normal).clamp(-1.0, 1.0);
                    sum_diff += 1.0 - cos.abs();
                    count += 1;
                }
            }
        }
        let val = if count == 0 {
            0.0
        } else {
            (sum_diff / count as f32).clamp(0.0, 1.0)
        };
        out.push(val);
    }
    out
}

struct Triangle {
    v0: glam::Vec3,
    v1: glam::Vec3,
    v2: glam::Vec3,
    uv0: glam::Vec2,
    uv1: glam::Vec2,
    uv2: glam::Vec2,
    normal: glam::Vec3,
    material: usize,
    area: f32,
}

pub fn convert_gltf(
    path: impl AsRef<path::Path>,
    options: &ConvertOptions,
) -> Result<vol::PointCloudModel, ConvertError> {
    let (document, buffers, images) = gltf::import(path.as_ref())?;
    let mut materials = Vec::new();

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let base_color_texture = pbr.base_color_texture().map(|tex| {
            let image = tex.texture().source();
            let data = &images[image.index()];
            Texture {
                width: data.width,
                height: data.height,
                data: rgba8_from_gltf_image(data),
            }
        });
        materials.push(MaterialInfo {
            base_color: glam::Vec4::from(base_color),
            base_color_texture,
        });
    }

    if materials.is_empty() {
        materials.push(MaterialInfo {
            base_color: glam::Vec4::ONE,
            base_color_texture: None,
        });
    }

    let mut triangles = Vec::new();

    for scene in document.scenes() {
        for node in scene.nodes() {
            gather_node_triangles(
                &node,
                glam::Mat4::IDENTITY,
                &buffers,
                &materials,
                &mut triangles,
            )?;
        }
    }

    if triangles.is_empty() {
        return Err(ConvertError::MissingMeshData);
    }

    let density = options.density;
    if density <= 0.0 {
        return Err(ConvertError::InvalidDensity);
    }

    let bbox = compute_bounds(&triangles);
    let avg_color = compute_average_color(&materials);

    let interior_density = density * options.interior_density_scale;
    let mut points = Vec::new();
    let mut sh_coefficients = Vec::new();
    let mut rotations = Vec::new();
    let mut scales = Vec::new();

    if interior_density > 0.0 {
        let spacing = interior_density.powf(-1.0 / 3.0);
        let sub_div = 3u32;
        let sub_spacing = spacing / sub_div as f32;
        let start = bbox.min + glam::Vec3::splat(0.5 * spacing);
        let mut z = start.z;
        while z <= bbox.max.z {
            let mut y = start.y;
            while y <= bbox.max.y {
                let mut x = start.x;
                while x <= bbox.max.x {
                    let p = glam::Vec3::new(x, y, z);
                    if is_point_inside_mesh(p, &triangles) {
                        let color =
                            (avg_color * options.ambient).clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
                        let base = p - glam::Vec3::splat(0.5 * spacing);
                        let scale = sub_spacing * 0.5 * options.interior_scale;
                        let mut iz = 0u32;
                        while iz < sub_div {
                            let mut iy = 0u32;
                            while iy < sub_div {
                                let mut ix = 0u32;
                                while ix < sub_div {
                                    let offset = glam::Vec3::new(
                                        (ix as f32 + 0.5) * sub_spacing,
                                        (iy as f32 + 0.5) * sub_spacing,
                                        (iz as f32 + 0.5) * sub_spacing,
                                    );
                                    let sp = base + offset;
                                    push_point(
                                        &mut points,
                                        &mut sh_coefficients,
                                        &mut rotations,
                                        &mut scales,
                                        sp,
                                        color,
                                        options.interior_opacity,
                                        scale,
                                        None,
                                        options,
                                    );
                                    ix += 1;
                                }
                                iy += 1;
                            }
                            iz += 1;
                        }
                    }
                    x += spacing;
                }
                y += spacing;
            }
            z += spacing;
        }
    }

    let surface_density = density.powf(2.0 / 3.0) * options.surface_density_scale;
    if surface_density > 0.0 {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(options.seed);
        let curvatures = if options.curvature_boost > 0.0 {
            compute_triangle_curvatures(&triangles)
        } else {
            vec![0.0_f32; triangles.len()]
        };
        for (ti, tri) in triangles.iter().enumerate() {
            let factor = if options.curvature_boost > 0.0 {
                1.0 + options.curvature_boost * curvatures[ti]
            } else {
                1.0
            };
            let count = (tri.area * surface_density * factor).ceil() as u32;
            for _ in 0..count {
                let (u, v) = sample_barycentric(&mut rng);
                let w = 1.0 - u - v;
                let p = tri.v0 * u + tri.v1 * v + tri.v2 * w;
                let uv = tri.uv0 * u + tri.uv1 * v + tri.uv2 * w;
                let mat = &materials[tri.material];
                let base = sample_material_color(mat, uv);
                let color =
                    (base.truncate() * options.ambient).clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
                let alpha = base.w;
                if alpha < options.alpha_threshold {
                    continue;
                }

                push_point(
                    &mut points,
                    &mut sh_coefficients,
                    &mut rotations,
                    &mut scales,
                    p,
                    color,
                    options.surface_opacity,
                    surface_point_scale(density) * options.surface_scale,
                    Some(tri.normal),
                    options,
                );
            }
        }
    }

    let mut model = vol::PointCloudModel {
        points,
        sh_coefficients,
        sh_degree: 0,
        transforms: None,
        adjacency: None,
        radii: None,
    };

    match options.output {
        OutputKind::Gaussian => {
            model.transforms = Some(vol::Transforms { rotations, scales });
        }
        OutputKind::RadFoam => {
            model.adjacency = Some(vol::compute_adjacency_default(&model.points));
            if options.lloyd_iterations > 0 {
                vol::lloyd_relax(&mut model, options.lloyd_iterations, options.lloyd_step);
            }
            if options.assign_radii {
                model.radii = Some(vol::radii_from_nearest_neighbour(
                    &model.points,
                    options.radius_factor,
                ));
                // Keep Delaunay adjacency: it's a superset of Čech for radii
                // ≤ nearest-neighbour, and kiddo panics on the coincident
                // grid-interior points the converter emits.
            }
        }
    }

    Ok(model)
}

pub fn save_ply(
    path: impl AsRef<path::Path>,
    model: &vol::PointCloudModel,
) -> Result<(), ConvertError> {
    save_ply_with_options(path, model, &SaveOptions::default())
}

#[derive(Clone, Copy, Debug)]
pub struct SaveOptions {
    pub format: PlyFormat,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            format: PlyFormat::Binary,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlyFormat {
    Ascii,
    Binary,
}

pub fn save_ply_with_options(
    path: impl AsRef<path::Path>,
    model: &vol::PointCloudModel,
    options: &SaveOptions,
) -> Result<(), ConvertError> {
    if model.transforms.is_some() {
        match options.format {
            PlyFormat::Ascii => write_gaussian_ply_ascii(path.as_ref(), model)?,
            PlyFormat::Binary => write_gaussian_ply_binary(path.as_ref(), model)?,
        }
        return Ok(());
    }

    if model.adjacency.is_some() {
        match options.format {
            PlyFormat::Ascii => write_radfoam_ply_ascii(path.as_ref(), model)?,
            PlyFormat::Binary => write_radfoam_ply_binary(path.as_ref(), model)?,
        }
        return Ok(());
    }

    Err(ConvertError::MissingOutputData)
}

fn gather_node_triangles(
    node: &gltf::Node,
    parent: glam::Mat4,
    buffers: &[gltf::buffer::Data],
    materials: &[MaterialInfo],
    triangles: &mut Vec<Triangle>,
) -> Result<(), ConvertError> {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let transform = parent * local;
    let normal_transform = transform.inverse().transpose();

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(ConvertError::UnsupportedPrimitiveMode);
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions = reader
                .read_positions()
                .ok_or(ConvertError::MissingMeshData)?
                .map(|p| transform.transform_point3(glam::Vec3::from(p)))
                .collect::<Vec<_>>();
            let uvs = reader
                .read_tex_coords(0)
                .map(|uv| uv.into_f32().map(glam::Vec2::from).collect::<Vec<_>>())
                .unwrap_or_else(|| vec![glam::Vec2::ZERO; positions.len()]);
            let normals = reader
                .read_normals()
                .map(|n| {
                    n.map(|nn| {
                        let v = normal_transform.transform_vector3(glam::Vec3::from(nn));
                        v.normalize_or_zero()
                    })
                    .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![glam::Vec3::ZERO; positions.len()]);
            let indices = reader
                .read_indices()
                .ok_or(ConvertError::MissingMeshData)?
                .into_u32()
                .collect::<Vec<_>>();

            let material_index = primitive.material().index().unwrap_or(0);
            let material_index = material_index.min(materials.len() - 1);

            for tri in indices.chunks_exact(3) {
                let i0 = tri[0] as usize;
                let i1 = tri[1] as usize;
                let i2 = tri[2] as usize;
                let v0 = positions[i0];
                let v1 = positions[i1];
                let v2 = positions[i2];
                let normal = if normals[i0] != glam::Vec3::ZERO {
                    let n = (normals[i0] + normals[i1] + normals[i2]) / 3.0;
                    n.normalize_or_zero()
                } else {
                    (v1 - v0).cross(v2 - v0).normalize_or_zero()
                };
                let area = 0.5 * (v1 - v0).cross(v2 - v0).length();
                triangles.push(Triangle {
                    v0,
                    v1,
                    v2,
                    uv0: uvs[i0],
                    uv1: uvs[i1],
                    uv2: uvs[i2],
                    normal,
                    material: material_index,
                    area,
                });
            }
        }
    }

    for child in node.children() {
        gather_node_triangles(&child, transform, buffers, materials, triangles)?;
    }

    Ok(())
}

fn compute_bounds(triangles: &[Triangle]) -> Bounds {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for tri in triangles {
        for v in [tri.v0, tri.v1, tri.v2] {
            min = min.min(v);
            max = max.max(v);
        }
    }
    Bounds { min, max }
}

struct Bounds {
    min: glam::Vec3,
    max: glam::Vec3,
}

fn compute_average_color(materials: &[MaterialInfo]) -> glam::Vec3 {
    let mut sum = glam::Vec3::ZERO;
    for material in materials {
        sum += material.base_color.truncate();
    }
    sum / materials.len() as f32
}

fn is_point_inside_mesh(point: glam::Vec3, triangles: &[Triangle]) -> bool {
    const RAYS: [glam::Vec3; 3] = [
        glam::Vec3::new(0.321, 0.571, 0.755),
        glam::Vec3::new(-0.742, 0.201, 0.639),
        glam::Vec3::new(0.189, -0.911, 0.367),
    ];

    let mut odd_hits = 0u32;
    for ray_dir in &RAYS {
        let dir = ray_dir.normalize();
        let origin = point + dir * 1e-4;
        let mut hits = 0u32;
        for tri in triangles {
            if ray_intersects_triangle(origin, dir, tri) {
                hits += 1;
            }
        }
        if hits % 2 == 1 {
            odd_hits += 1;
        }
    }

    odd_hits >= 2
}

fn ray_intersects_triangle(origin: glam::Vec3, dir: glam::Vec3, tri: &Triangle) -> bool {
    let eps = 1e-6;
    let v0v1 = tri.v1 - tri.v0;
    let v0v2 = tri.v2 - tri.v0;
    let pvec = dir.cross(v0v2);
    let det = v0v1.dot(pvec);
    if det.abs() < eps {
        return false;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - tri.v0;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qvec = tvec.cross(v0v1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = v0v2.dot(qvec) * inv_det;
    t > eps
}

fn sample_barycentric(rng: &mut rand::rngs::StdRng) -> (f32, f32) {
    use rand::RngExt as _;
    let r1: f32 = rng.random();
    let r2: f32 = rng.random();
    let sqrt_r1 = r1.sqrt();
    let u = 1.0 - sqrt_r1;
    let v = r2 * sqrt_r1;
    (u, v)
}

fn sample_material_color(material: &MaterialInfo, uv: glam::Vec2) -> glam::Vec4 {
    if let Some(ref tex) = material.base_color_texture {
        tex.sample(uv) * material.base_color
    } else {
        material.base_color
    }
}

fn surface_point_scale(density: f32) -> f32 {
    let spacing = density.powf(-1.0 / 3.0);
    spacing * 0.5
}

fn push_point(
    points: &mut Vec<glam::Vec4>,
    sh_coefficients: &mut Vec<f32>,
    rotations: &mut Vec<glam::Quat>,
    scales: &mut Vec<glam::Vec3>,
    position: glam::Vec3,
    color: glam::Vec3,
    opacity: f32,
    scale: f32,
    normal: Option<glam::Vec3>,
    options: &ConvertOptions,
) {
    points.push(glam::Vec4::new(position.x, position.y, position.z, opacity));

    match options.output {
        OutputKind::Gaussian => {
            let biased = (color - glam::Vec3::splat(0.5)) / SH_C0;
            sh_coefficients.extend_from_slice(&[biased.x, biased.y, biased.z]);
            let (rotation, scale_vec) = match normal {
                Some(n) if n.length_squared() > 0.0 => {
                    let rot = glam::Quat::from_rotation_arc(glam::Vec3::Z, n.normalize());
                    let s = glam::Vec3::new(scale, scale, scale * options.surface_normal_scale);
                    (rot, s)
                }
                _ => (glam::Quat::IDENTITY, glam::Vec3::splat(scale)),
            };
            rotations.push(rotation);
            scales.push(scale_vec);
        }
        OutputKind::RadFoam => {
            let biased = (color - glam::Vec3::splat(0.5)) / SH_C0;
            sh_coefficients.extend_from_slice(&[biased.x, biased.y, biased.z]);
        }
    }
}

fn rgba8_from_gltf_image(image: &gltf::image::Data) -> Vec<u8> {
    match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => image
            .pixels
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        gltf::image::Format::R8G8 => image
            .pixels
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[1], 0, 255])
            .collect(),
        gltf::image::Format::R8 => image
            .pixels
            .iter()
            .flat_map(|c| [*c, *c, *c, 255])
            .collect(),
        _ => {
            let img = image::load_from_memory(&image.pixels)
                .unwrap_or_else(|_| image::DynamicImage::new_rgba8(image.width, image.height));
            img.to_rgba8().into_raw()
        }
    }
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn write_gaussian_ply_binary(
    path: &path::Path,
    model: &vol::PointCloudModel,
) -> Result<(), ConvertError> {
    use std::io::Write as _;

    let Some(ref transforms) = model.transforms else {
        return Err(ConvertError::MissingOutputData);
    };

    let count = model.len();
    let sh_component_count = vol::get_sh_component_count(model.sh_degree);
    let sh_rest_count = sh_component_count.saturating_sub(1) * 3;
    let ply_rotation =
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), -std::f32::consts::FRAC_PI_2);
    let inv_rotation = ply_rotation.inverse();

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "ply")?;
    writeln!(file, "format binary_little_endian 1.0")?;
    writeln!(file, "element vertex {}", count)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property float nx")?;
    writeln!(file, "property float ny")?;
    writeln!(file, "property float nz")?;
    writeln!(file, "property float f_dc_0")?;
    writeln!(file, "property float f_dc_1")?;
    writeln!(file, "property float f_dc_2")?;
    writeln!(file, "property float opacity")?;
    writeln!(file, "property float scale_0")?;
    writeln!(file, "property float scale_1")?;
    writeln!(file, "property float scale_2")?;
    writeln!(file, "property float rot_0")?;
    writeln!(file, "property float rot_1")?;
    writeln!(file, "property float rot_2")?;
    writeln!(file, "property float rot_3")?;
    for i in 0..sh_rest_count {
        writeln!(file, "property float f_rest_{}", i)?;
    }
    writeln!(file, "end_header")?;

    for i in 0..count {
        let point = model.points[i];
        let position = inv_rotation * glam::Vec3::new(point.x, point.y, point.z);
        let rotation = inv_rotation * transforms.rotations[i];
        let scale = transforms.scales[i].max(glam::Vec3::splat(1e-6));
        let log_scale = glam::Vec3::new(scale.x.ln(), scale.y.ln(), scale.z.ln());
        let opacity = logit(point.w);

        let base = i * sh_component_count * 3;
        let f_dc = [
            model.sh_coefficients[base],
            model.sh_coefficients[base + 1],
            model.sh_coefficients[base + 2],
        ];

        file.write_all(&position.x.to_le_bytes())?;
        file.write_all(&position.y.to_le_bytes())?;
        file.write_all(&position.z.to_le_bytes())?;
        file.write_all(&0.0f32.to_le_bytes())?;
        file.write_all(&0.0f32.to_le_bytes())?;
        file.write_all(&0.0f32.to_le_bytes())?;
        file.write_all(&f_dc[0].to_le_bytes())?;
        file.write_all(&f_dc[1].to_le_bytes())?;
        file.write_all(&f_dc[2].to_le_bytes())?;
        file.write_all(&opacity.to_le_bytes())?;
        file.write_all(&log_scale.x.to_le_bytes())?;
        file.write_all(&log_scale.y.to_le_bytes())?;
        file.write_all(&log_scale.z.to_le_bytes())?;
        file.write_all(&rotation.w.to_le_bytes())?;
        file.write_all(&rotation.x.to_le_bytes())?;
        file.write_all(&rotation.y.to_le_bytes())?;
        file.write_all(&rotation.z.to_le_bytes())?;

        for j in 0..sh_rest_count {
            let coeff = model
                .sh_coefficients
                .get(base + 3 + j)
                .copied()
                .unwrap_or(0.0);
            file.write_all(&coeff.to_le_bytes())?;
        }
    }

    Ok(())
}

fn write_gaussian_ply_ascii(
    path: &path::Path,
    model: &vol::PointCloudModel,
) -> Result<(), ConvertError> {
    use std::io::Write as _;

    let Some(ref transforms) = model.transforms else {
        return Err(ConvertError::MissingOutputData);
    };

    let count = model.len();
    let sh_component_count = vol::get_sh_component_count(model.sh_degree);
    let sh_rest_count = sh_component_count.saturating_sub(1) * 3;
    let ply_rotation =
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), -std::f32::consts::FRAC_PI_2);
    let inv_rotation = ply_rotation.inverse();

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", count)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property float nx")?;
    writeln!(file, "property float ny")?;
    writeln!(file, "property float nz")?;
    writeln!(file, "property float f_dc_0")?;
    writeln!(file, "property float f_dc_1")?;
    writeln!(file, "property float f_dc_2")?;
    writeln!(file, "property float opacity")?;
    writeln!(file, "property float scale_0")?;
    writeln!(file, "property float scale_1")?;
    writeln!(file, "property float scale_2")?;
    writeln!(file, "property float rot_0")?;
    writeln!(file, "property float rot_1")?;
    writeln!(file, "property float rot_2")?;
    writeln!(file, "property float rot_3")?;
    for i in 0..sh_rest_count {
        writeln!(file, "property float f_rest_{}", i)?;
    }
    writeln!(file, "end_header")?;

    for i in 0..count {
        let point = model.points[i];
        let position = inv_rotation * glam::Vec3::new(point.x, point.y, point.z);
        let rotation = inv_rotation * transforms.rotations[i];
        let scale = transforms.scales[i].max(glam::Vec3::splat(1e-6));
        let log_scale = glam::Vec3::new(scale.x.ln(), scale.y.ln(), scale.z.ln());
        let opacity = logit(point.w);

        let base = i * sh_component_count * 3;
        let f_dc = [
            model.sh_coefficients[base],
            model.sh_coefficients[base + 1],
            model.sh_coefficients[base + 2],
        ];

        write!(
            file,
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            position.x,
            position.y,
            position.z,
            0.0f32,
            0.0f32,
            0.0f32,
            f_dc[0],
            f_dc[1],
            f_dc[2],
            opacity,
            log_scale.x,
            log_scale.y,
            log_scale.z,
            rotation.w,
            rotation.x,
            rotation.y,
            rotation.z
        )?;

        for j in 0..sh_rest_count {
            let coeff = model
                .sh_coefficients
                .get(base + 3 + j)
                .copied()
                .unwrap_or(0.0);
            write!(file, " {}", coeff)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

fn write_radfoam_ply_ascii(
    path: &path::Path,
    model: &vol::PointCloudModel,
) -> Result<(), ConvertError> {
    use std::io::Write as _;

    let Some(ref adjacency) = model.adjacency else {
        return Err(ConvertError::MissingOutputData);
    };

    let count = model.len();
    let num_adjacency = adjacency.neighbors.len();
    let sh_components = vol::get_sh_component_count(model.sh_degree);
    let sh_rest = (sh_components - 1) * 3;
    let sh_block = sh_components * 3;
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", count)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property float density")?;
    writeln!(file, "property uint adjacency_offset")?;
    writeln!(file, "property uchar red")?;
    writeln!(file, "property uchar green")?;
    writeln!(file, "property uchar blue")?;
    for k in 0..sh_rest {
        writeln!(file, "property float color_sh_{k}")?;
    }
    if model.radii.is_some() {
        writeln!(file, "property float radius")?;
    }
    writeln!(file, "element adjacency {}", num_adjacency)?;
    writeln!(file, "property uint adjacency")?;
    writeln!(file, "end_header")?;

    for i in 0..count {
        let point = model.points[i];
        let end_off = adjacency.offsets[i + 1];
        let color = sh0_to_color(model, i);
        let r = (color.x * 255.0).round().clamp(0.0, 255.0) as u8;
        let g = (color.y * 255.0).round().clamp(0.0, 255.0) as u8;
        let b = (color.z * 255.0).round().clamp(0.0, 255.0) as u8;
        write!(
            file,
            "{} {} {} {} {} {} {} {}",
            point.x, point.y, point.z, point.w, end_off, r, g, b
        )?;
        let base = i * sh_block;
        for j in 0..sh_rest {
            let comp = 1 + j / 3;
            let ch = j % 3;
            let value = model.sh_coefficients[base + 3 * comp + ch];
            write!(file, " {value}")?;
        }
        if let Some(ref radii) = model.radii {
            write!(file, " {}", radii[i])?;
        }
        writeln!(file)?;
    }

    for idx in &adjacency.neighbors {
        writeln!(file, "{}", idx)?;
    }

    Ok(())
}

fn write_radfoam_ply_binary(
    path: &path::Path,
    model: &vol::PointCloudModel,
) -> Result<(), ConvertError> {
    use std::io::Write as _;

    let Some(ref adjacency) = model.adjacency else {
        return Err(ConvertError::MissingOutputData);
    };

    let count = model.len();
    let num_adjacency = adjacency.neighbors.len();
    let sh_components = vol::get_sh_component_count(model.sh_degree);
    // Number of `color_sh_*` properties = (components excluding DC) × 3 channels.
    let sh_rest = (sh_components - 1) * 3;
    let sh_block = sh_components * 3;
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "ply")?;
    writeln!(file, "format binary_little_endian 1.0")?;
    writeln!(file, "element vertex {}", count)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property float density")?;
    writeln!(file, "property uint adjacency_offset")?;
    writeln!(file, "property uchar red")?;
    writeln!(file, "property uchar green")?;
    writeln!(file, "property uchar blue")?;
    for k in 0..sh_rest {
        writeln!(file, "property float color_sh_{k}")?;
    }
    if model.radii.is_some() {
        writeln!(file, "property float radius")?;
    }
    writeln!(file, "element adjacency {}", num_adjacency)?;
    writeln!(file, "property uint adjacency")?;
    writeln!(file, "end_header")?;

    for i in 0..count {
        let point = model.points[i];
        let end_off = adjacency.offsets[i + 1];
        let color = sh0_to_color(model, i);
        let r = (color.x * 255.0).round().clamp(0.0, 255.0) as u8;
        let g = (color.y * 255.0).round().clamp(0.0, 255.0) as u8;
        let b = (color.z * 255.0).round().clamp(0.0, 255.0) as u8;

        file.write_all(&point.x.to_le_bytes())?;
        file.write_all(&point.y.to_le_bytes())?;
        file.write_all(&point.z.to_le_bytes())?;
        file.write_all(&point.w.to_le_bytes())?;
        file.write_all(&end_off.to_le_bytes())?;
        file.write_all(&[r, g, b])?;
        // color_sh_* layout per RadFoam upstream loader: index j maps to
        // SH component `1 + j/3`, channel `j%3`. DC (component 0) lives
        // in red/green/blue above (as 8-bit-quantised preview).
        let base = i * sh_block;
        for j in 0..sh_rest {
            let comp = 1 + j / 3;
            let ch = j % 3;
            let value = model.sh_coefficients[base + 3 * comp + ch];
            file.write_all(&value.to_le_bytes())?;
        }
        if let Some(ref radii) = model.radii {
            file.write_all(&radii[i].to_le_bytes())?;
        }
    }

    for idx in &adjacency.neighbors {
        file.write_all(&idx.to_le_bytes())?;
    }

    Ok(())
}

fn sh0_to_color(model: &vol::PointCloudModel, index: usize) -> glam::Vec3 {
    // The SH layout is `[p0_c0_r, p0_c0_g, p0_c0_b, p0_c1_r, ..., p1_c0_r, ...]`:
    // 3 channels × `(1+sh_degree)²` components per point. The DC term lives at
    // the start of each per-point stride. The previous `index * 3` worked
    // accidentally for SH-0 (stride 3) but read garbage out of higher-order
    // coefficients for SH-1+ models — that round-tripped a "DC" that was
    // actually some other point's higher SH coefficient, hard-coding a
    // ~8 dB PSNR collapse into every saved SH-3 PLY. The 2026-05-22 exposure
    // A/B caught it: in-training eval 20.85 dB, fresh eval of the same PLY
    // 11.64 dB.
    let stride = vol::get_sh_component_count(model.sh_degree) * 3;
    let base = index * stride;
    let coeff = glam::Vec3::new(
        model.sh_coefficients[base],
        model.sh_coefficients[base + 1],
        model.sh_coefficients[base + 2],
    );
    (coeff * SH_C0 + glam::Vec3::splat(0.5)).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

fn logit(value: f32) -> f32 {
    let clamped = value.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_ply_roundtrip_keeps_count() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 0.8),
            glam::Vec4::new(1.0, 0.5, -0.5, 0.4),
        ];
        let sh_coefficients = vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.0];
        let transforms = vol::Transforms {
            rotations: vec![glam::Quat::IDENTITY; points.len()],
            scales: vec![glam::Vec3::splat(0.25); points.len()],
        };
        let model = vol::PointCloudModel {
            points,
            sh_coefficients,
            sh_degree: 0,
            transforms: Some(transforms),
            adjacency: None,
            radii: None,
        };

        let mut path = std::env::temp_dir();
        path.push("blade_volume_convert_roundtrip.ply");
        save_ply(&path, &model).expect("save ply");
        let loaded = vol::io::load_gaussian(path.to_str().unwrap());

        assert_eq!(loaded.len(), model.len());
        assert!(loaded.transforms.is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn radfoam_binary_roundtrip_preserves_sh3() {
        // Two points with SH degree 3 (16 components × 3 channels per cell).
        // The DC coefficients (component 0) round-trip through the 8-bit
        // RGB preview path and lose a little precision; the higher-order
        // coefficients (1..15) must survive exactly through the new
        // `color_sh_*` properties.
        let n = 2usize;
        let sh_degree = 3usize;
        let sh_components = vol::get_sh_component_count(sh_degree);
        let sh_block = sh_components * 3;
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 0.5),
        ];
        // Distinct values per cell × component × channel so any layout
        // swap shows up.
        let sh_coefficients: Vec<f32> = (0..n * sh_block)
            .map(|i| 0.001 * (i as f32) - 0.5)
            .collect();
        let model = vol::PointCloudModel {
            points,
            sh_coefficients: sh_coefficients.clone(),
            sh_degree,
            transforms: None,
            adjacency: Some(vol::Adjacency {
                neighbors: vec![1, 0],
                offsets: vec![0, 1, 2],
            }),
            radii: None,
        };

        let mut path = std::env::temp_dir();
        path.push("blade_volume_convert_roundtrip_radfoam_sh3.ply");
        save_ply_with_options(
            &path,
            &model,
            &SaveOptions {
                format: PlyFormat::Binary,
            },
        )
        .expect("save ply");
        let loaded = vol::io::load_radfoam(path.to_str().unwrap());

        assert_eq!(loaded.sh_degree, sh_degree, "sh_degree must round-trip");
        assert_eq!(loaded.sh_coefficients.len(), n * sh_block);
        // Higher-order coefficients (component 1..15) must match exactly.
        for i in 0..n {
            let base = i * sh_block;
            for c in 1..sh_components {
                for ch in 0..3 {
                    let off = base + 3 * c + ch;
                    assert!(
                        (loaded.sh_coefficients[off] - sh_coefficients[off]).abs() < 1e-6,
                        "sh[{i},{c},{ch}] roundtrip mismatch: got {}, want {}",
                        loaded.sh_coefficients[off],
                        sh_coefficients[off],
                    );
                }
            }
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn radfoam_binary_roundtrip_keeps_count() {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 0.5),
        ];
        let sh_coefficients = vec![0.0; points.len() * 3];
        let adjacency = vol::Adjacency {
            neighbors: vec![1, 0],
            offsets: vec![0, 1, 2],
        };
        let model = vol::PointCloudModel {
            points,
            sh_coefficients,
            sh_degree: 0,
            transforms: None,
            adjacency: Some(adjacency),
            radii: None,
        };

        let mut path = std::env::temp_dir();
        path.push("blade_volume_convert_roundtrip_radfoam.ply");
        let options = SaveOptions {
            format: PlyFormat::Binary,
        };
        save_ply_with_options(&path, &model, &options).expect("save ply");
        let loaded = vol::io::load_radfoam(path.to_str().unwrap());

        assert_eq!(loaded.len(), model.len());
        assert!(loaded.adjacency.is_some());
        let _ = std::fs::remove_file(path);
    }

    fn make_radfoam_radii_model() -> vol::PointCloudModel {
        let points = vec![
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 0.5),
            glam::Vec4::new(0.0, 1.0, 0.0, 0.25),
        ];
        let radii = vec![0.10, 0.25, 0.5];
        vol::PointCloudModel {
            sh_coefficients: vec![0.0; points.len() * 3],
            adjacency: Some(vol::Adjacency {
                neighbors: vec![1, 0, 2, 1],
                offsets: vec![0, 1, 3, 4],
            }),
            radii: Some(radii),
            transforms: None,
            sh_degree: 0,
            points,
        }
    }

    fn assert_radii_roundtrip(format: PlyFormat) {
        let model = make_radfoam_radii_model();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "blade_volume_convert_roundtrip_radii_{}.ply",
            match format {
                PlyFormat::Ascii => "ascii",
                PlyFormat::Binary => "binary",
            }
        ));
        save_ply_with_options(&path, &model, &SaveOptions { format }).expect("save ply");
        let loaded = vol::io::load_radfoam(path.to_str().unwrap());
        let expected = model.radii.as_ref().unwrap();
        let actual = loaded.radii.as_ref().expect("radii preserved");
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "radius {a} != {b}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn radfoam_binary_roundtrip_preserves_radii() {
        assert_radii_roundtrip(PlyFormat::Binary);
    }

    #[test]
    fn radfoam_ascii_roundtrip_preserves_radii() {
        assert_radii_roundtrip(PlyFormat::Ascii);
    }

    #[test]
    fn radfoam_without_radii_stays_none_after_roundtrip() {
        let mut model = make_radfoam_radii_model();
        model.radii = None;
        let mut path = std::env::temp_dir();
        path.push("blade_volume_convert_roundtrip_no_radii.ply");
        save_ply_with_options(
            &path,
            &model,
            &SaveOptions {
                format: PlyFormat::Binary,
            },
        )
        .expect("save ply");
        let loaded = vol::io::load_radfoam(path.to_str().unwrap());
        assert!(loaded.radii.is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ray_intersects_triangle_hits() {
        let tri = Triangle {
            v0: glam::Vec3::new(0.0, 0.0, 0.0),
            v1: glam::Vec3::new(0.0, 1.0, 0.0),
            v2: glam::Vec3::new(0.0, 0.0, 1.0),
            uv0: glam::Vec2::ZERO,
            uv1: glam::Vec2::ZERO,
            uv2: glam::Vec2::ZERO,
            normal: glam::Vec3::X,
            material: 0,
            area: 0.5,
        };

        let origin = glam::Vec3::new(-1.0, 0.25, 0.25);
        let dir = glam::Vec3::X;
        assert!(ray_intersects_triangle(origin, dir, &tri));
    }

    #[test]
    fn point_inside_mesh_tetrahedron() {
        let triangles = vec![
            Triangle {
                v0: glam::Vec3::new(0.0, 0.0, 0.0),
                v1: glam::Vec3::new(1.0, 0.0, 0.0),
                v2: glam::Vec3::new(0.0, 1.0, 0.0),
                uv0: glam::Vec2::ZERO,
                uv1: glam::Vec2::ZERO,
                uv2: glam::Vec2::ZERO,
                normal: glam::Vec3::Z,
                material: 0,
                area: 0.5,
            },
            Triangle {
                v0: glam::Vec3::new(0.0, 0.0, 0.0),
                v1: glam::Vec3::new(0.0, 1.0, 0.0),
                v2: glam::Vec3::new(0.0, 0.0, 1.0),
                uv0: glam::Vec2::ZERO,
                uv1: glam::Vec2::ZERO,
                uv2: glam::Vec2::ZERO,
                normal: glam::Vec3::X,
                material: 0,
                area: 0.5,
            },
            Triangle {
                v0: glam::Vec3::new(0.0, 0.0, 0.0),
                v1: glam::Vec3::new(0.0, 0.0, 1.0),
                v2: glam::Vec3::new(1.0, 0.0, 0.0),
                uv0: glam::Vec2::ZERO,
                uv1: glam::Vec2::ZERO,
                uv2: glam::Vec2::ZERO,
                normal: glam::Vec3::Y,
                material: 0,
                area: 0.5,
            },
            Triangle {
                v0: glam::Vec3::new(1.0, 0.0, 0.0),
                v1: glam::Vec3::new(0.0, 0.0, 1.0),
                v2: glam::Vec3::new(0.0, 1.0, 0.0),
                uv0: glam::Vec2::ZERO,
                uv1: glam::Vec2::ZERO,
                uv2: glam::Vec2::ZERO,
                normal: glam::Vec3::new(1.0, 1.0, 1.0).normalize(),
                material: 0,
                area: 0.5,
            },
        ];

        let inside = glam::Vec3::new(0.1, 0.1, 0.1);
        let outside = glam::Vec3::new(1.5, 1.5, 1.5);
        assert!(is_point_inside_mesh(inside, &triangles));
        assert!(!is_point_inside_mesh(outside, &triangles));
    }
}
