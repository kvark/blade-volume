use blade_volume as vol;
use std::path;

const SH_C0: f32 = 0.282_094_8;

/// Interior jitter applied to triangulated output when none is requested.
///
/// Measured on `police.glb` at resolution 128 (804,464 points): an exact
/// lattice makes the pure-Rust Delaunay take 30.9s and disagree with Qhull on
/// 15.6% of edges, because a cospherical site set has no unique triangulation.
/// Half a sub-cell brings that to 8.4s and 3.9%.
pub const DEFAULT_INTERIOR_JITTER: f32 = 0.5;

/// Exterior fill rate for triangulated output when none is requested, relative
/// to [`ConvertOptions::density`].
///
/// Exterior sites do not resolve detail, but they do pin the silhouette: a
/// surface cell extends outward until it meets another site, so a fill that is
/// too sparse inflates the object's outline. Measured against the mesh on
/// `police.glb` at resolution 48 (rendered PSNR / total points):
/// 0.002 -> 10.38 dB / 43k, 0.02 -> 12.06 / 47k, **0.1 -> 12.63 / 63k**,
/// 0.5 -> 13.00 / 139k, 1.0 -> 13.11 / 240k. 0.1 is the knee — beyond it the
/// point count grows far faster than the image improves.
///
/// See [`ConvertOptions::exterior_density_scale`] for why this is not optional
/// in practice for RadFoam.
pub const DEFAULT_EXTERIOR_DENSITY_SCALE: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Gaussian,
    RadFoam,
}

/// Which Delaunay builder produces the unweighted RadFoam adjacency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Topology {
    /// Pure-Rust exact construction. Dependency free and exact, but its cost
    /// grows steeply and it is not practical much past ~100k sites.
    #[default]
    Exact,
    /// Qhull. Requires building with the `qhull` feature; converting a
    /// asset-sized cloud without it is impractically slow.
    Qhull,
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub output: OutputKind,
    /// Samples per cubic world unit. The interior grid spacing is
    /// `density^(-1/3)` and the surface budget is `density^(2/3)` samples per
    /// square world unit, so a single length scale drives both. Because it is
    /// absolute, the same value gives wildly different point counts for assets
    /// authored in different units; prefer [`ConvertOptions::resolution`] when
    /// the asset's scale is not under your control.
    pub density: f32,
    /// Scale-invariant alternative to [`ConvertOptions::density`]: the number
    /// of grid cells across the bounding-box diagonal. When `Some`, `density`
    /// is derived per asset as `(resolution / diagonal)^3` and the field's own
    /// value is ignored.
    pub resolution: Option<f32>,
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
    /// Curvature multiplier for per-triangle surface sampling. `0` keeps the
    /// uniform-by-area density. `>0` boosts the sample count on triangles that
    /// are at a high dihedral angle to their (vertex-shared) neighbours, at
    /// the cost of sparser sampling on flat regions. Area-weight normalization
    /// keeps the total sample budget roughly constant before integer rounding.
    pub curvature_boost: f32,
    /// Number of spring-relaxation iterations applied after Delaunay (RadFoam
    /// path only). `0` skips the relaxation entirely.
    pub spring_iterations: usize,
    /// Per-iteration step size for [`vol::spring_relax`].
    pub spring_step: f32,
    /// When `true`, populate `model.radii` from the post-relaxation nearest-
    /// neighbour distance (Power Foam mode). When `false`, leave `radii`
    /// as `None` (plain Voronoi).
    pub assign_radii: bool,
    /// Multiplier for [`vol::radii_from_nearest_neighbour`].
    pub radius_factor: f32,
    /// Delaunay builder for the RadFoam adjacency. Ignored by the Gaussian
    /// output, and by the Čech rebuild that follows radius assignment.
    pub topology: Topology,
    /// Zero-density samples placed in the padded space *outside* the mesh, as
    /// a fraction of [`ConvertOptions::density`]. `None` picks per output kind.
    ///
    /// This exists because adjacency-walk traversal integrates whatever cell a
    /// ray is in, and a Voronoi diagram built only from surface and interior
    /// sites gives every point of empty space to an *unbounded* cell owned by
    /// an opaque surface site. A camera outside the object therefore looks
    /// through opaque fog: object-centric RadFoam renders as a white smear
    /// regardless of sampling rate. Trained scenes never show this because
    /// optimisation drives background cells to near-zero density; a converted
    /// asset has no such stage, so the transparent exterior must be built.
    ///
    /// Only affects [`OutputKind::RadFoam`]; Gaussian splatting has no cells
    /// and ignores empty space already.
    pub exterior_density_scale: Option<f32>,
    /// How far the exterior fill extends beyond the mesh bounds, as a fraction
    /// of the bounding-box diagonal.
    pub exterior_padding: f32,
    /// Random displacement applied to interior samples, as a fraction of the
    /// sub-cell spacing. `Some(0.0)` places them on an exact lattice, which is
    /// a degenerate (cospherical) configuration for Delaunay: the
    /// triangulation is then not unique, different builders legitimately
    /// return different adjacencies, and both take substantially longer. A
    /// small jitter breaks the ties.
    ///
    /// `None` picks per output kind, because that is exactly where the
    /// trade-off differs: [`OutputKind::RadFoam`] is triangulated and gets
    /// [`DEFAULT_INTERIOR_JITTER`], while [`OutputKind::Gaussian`] builds no
    /// adjacency, so jitter there buys nothing and measurably degrades the
    /// render by scattering splats off their lattice.
    pub interior_jitter: Option<f32>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            output: OutputKind::Gaussian,
            density: 10.0,
            resolution: None,
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
            spring_iterations: 0,
            spring_step: 0.3,
            assign_radii: false,
            radius_factor: 0.5,
            topology: Topology::Exact,
            exterior_density_scale: None,
            exterior_padding: 0.35,
            interior_jitter: None,
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
    Adjacency(vol::AdjacencyError),
    /// [`Topology::Qhull`] was requested from a build without the feature.
    QhullUnavailable,
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

impl From<vol::AdjacencyError> for ConvertError {
    fn from(err: vol::AdjacencyError) -> Self {
        ConvertError::Adjacency(err)
    }
}

struct MaterialInfo {
    base_color: glam::Vec4,
    /// glTF metallic-roughness, kept as authored. A relightable conversion
    /// needs both: a metal keeps its colour in the specular reflectance and
    /// has no diffuse response at all, and neither fact is recoverable from
    /// the base colour alone.
    metallic: f32,
    roughness: f32,
    base_color_texture: Option<Texture>,
    tex_coord: u32,
    uv_transform: UvTransform,
    alpha_mode: gltf::material::AlphaMode,
    alpha_cutoff: f32,
}

#[derive(Clone, Copy, Debug)]
struct UvTransform {
    offset: glam::Vec2,
    rotation: f32,
    scale: glam::Vec2,
}

impl UvTransform {
    fn identity() -> Self {
        Self {
            offset: glam::Vec2::ZERO,
            rotation: 0.0,
            scale: glam::Vec2::ONE,
        }
    }

    fn apply(self, uv: glam::Vec2) -> glam::Vec2 {
        let scaled = uv * self.scale;
        let (sin, cos) = self.rotation.sin_cos();
        glam::Vec2::new(
            cos * scaled.x - sin * scaled.y,
            sin * scaled.x + cos * scaled.y,
        ) + self.offset
    }
}

fn texture_info_uv_transform(info: &gltf::texture::Info<'_>) -> (u32, UvTransform) {
    let texture_transform = info.texture_transform();
    let tex_coord = texture_transform
        .as_ref()
        .and_then(gltf::texture::TextureTransform::tex_coord)
        .unwrap_or_else(|| info.tex_coord());
    let uv_transform = match texture_transform {
        Some(ref transform) => UvTransform {
            offset: glam::Vec2::from(transform.offset()),
            rotation: transform.rotation(),
            scale: glam::Vec2::from(transform.scale()),
        },
        None => UvTransform::identity(),
    };
    (tex_coord, uv_transform)
}

struct Texture {
    width: u32,
    height: u32,
    data: Vec<u8>,
    wrap_s: gltf::texture::WrappingMode,
    wrap_t: gltf::texture::WrappingMode,
}

impl Texture {
    fn sample(&self, uv: glam::Vec2) -> glam::Vec4 {
        let u = wrap_coordinate(uv.x, self.wrap_s);
        let v = wrap_coordinate(uv.y, self.wrap_t);
        let x = u * (self.width as f32 - 1.0);
        let y = v * (self.height as f32 - 1.0);
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

fn wrap_coordinate(value: f32, mode: gltf::texture::WrappingMode) -> f32 {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => value.clamp(0.0, 1.0),
        gltf::texture::WrappingMode::MirroredRepeat => {
            let phase = value.rem_euclid(2.0);
            if phase <= 1.0 {
                phase
            } else {
                2.0 - phase
            }
        }
        gltf::texture::WrappingMode::Repeat => value.rem_euclid(1.0),
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

fn normalized_curvature_factors(areas: &[f32], curvatures: &[f32], boost: f32) -> Vec<f32> {
    assert_eq!(areas.len(), curvatures.len());
    let total_area = areas.iter().sum::<f32>();
    let weighted_area = areas
        .iter()
        .zip(curvatures)
        .map(|(&area, &curvature)| area * (1.0 + boost * curvature))
        .sum::<f32>();
    let normalization = if weighted_area > 0.0 {
        total_area / weighted_area
    } else {
        1.0
    };
    curvatures
        .iter()
        .map(|&curvature| (1.0 + boost * curvature) * normalization)
        .collect()
}

struct Triangle {
    v0: glam::Vec3,
    v1: glam::Vec3,
    v2: glam::Vec3,
    uv0: glam::Vec2,
    uv1: glam::Vec2,
    uv2: glam::Vec2,
    color0: glam::Vec4,
    color1: glam::Vec4,
    color2: glam::Vec4,
    normal: glam::Vec3,
    material: usize,
    area: f32,
}

/// Walk the glTF default scene and collect world-space triangles plus the
/// material table they index.
///
/// Shared by cloud conversion and reference-mesh extraction so that both see
/// exactly the same geometry and materials — otherwise a render comparison
/// between them would measure the discrepancy, not the representation.
fn gather_scene(path: &path::Path) -> Result<(Vec<Triangle>, Vec<MaterialInfo>), ConvertError> {
    let (document, buffers, images) = gltf::import(path)?;
    let mut materials = Vec::new();

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let (base_color_texture, tex_coord, uv_transform) = match pbr.base_color_texture() {
            Some(info) => {
                let texture = info.texture();
                let image = texture.source();
                let sampler = texture.sampler();
                let data = &images[image.index()];
                let (tex_coord, uv_transform) = texture_info_uv_transform(&info);
                (
                    Some(Texture {
                        width: data.width,
                        height: data.height,
                        data: rgba8_from_gltf_image(data),
                        wrap_s: sampler.wrap_s(),
                        wrap_t: sampler.wrap_t(),
                    }),
                    tex_coord,
                    uv_transform,
                )
            }
            None => (None, 0, UvTransform::identity()),
        };
        materials.push(MaterialInfo {
            base_color: glam::Vec4::from(base_color),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            base_color_texture,
            tex_coord,
            uv_transform,
            alpha_mode: material.alpha_mode(),
            alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        });
    }

    // A primitive with no material uses glTF's default white PBR material,
    // regardless of whether unrelated named materials exist in the asset.
    let default_material = materials.len();
    materials.push(MaterialInfo {
        base_color: glam::Vec4::ONE,
        metallic: 1.0,
        roughness: 1.0,
        base_color_texture: None,
        tex_coord: 0,
        uv_transform: UvTransform::identity(),
        alpha_mode: gltf::material::AlphaMode::Opaque,
        alpha_cutoff: 0.5,
    });

    let mut triangles = Vec::new();
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or(ConvertError::MissingMeshData)?;
    for node in scene.nodes() {
        gather_node_triangles(
            &node,
            glam::Mat4::IDENTITY,
            &buffers,
            &materials,
            default_material,
            &mut triangles,
        )?;
    }

    if triangles.is_empty() {
        return Err(ConvertError::MissingMeshData);
    }

    Ok((triangles, materials))
}

/// Turn a glTF asset into surfels that can be lit by an environment.
///
/// The cloud conversion bakes a colour: it evaluates the base colour under a
/// fixed ambient gain and stores the result, which cannot be relit because the
/// light is already inside the number. This keeps the material instead.
///
/// Three things follow from what the relighting study measured rather than
/// from convenience:
///
/// - **Normals come from the triangle**, not from anything inferred later.
///   Five degrees of normal error costs 1.3 dB of relighting accuracy, and a
///   triangle knows its own normal exactly.
/// - **One material per glTF material**, shared by every surfel that came off
///   it, because a patch of surface does not determine a BRDF and there is no
///   reason to pretend otherwise when the asset already says so.
/// - **The metallic-roughness conversion happens once, here.** A metal keeps
///   its base colour as specular reflectance and loses its diffuse albedo,
///   which is why both factors had to be carried this far.
///
/// Sampling density follows the same `resolution` or `density` the cloud
/// conversion uses, so the two are comparable.
pub fn relight_model_from_gltf(
    path: &path::Path,
    options: &ConvertOptions,
) -> Result<vol::relight::RelightModel, ConvertError> {
    let (triangles, materials) = gather_scene(path)?;
    if triangles.is_empty() {
        return Err(ConvertError::MissingMeshData);
    }

    // Resolved exactly as the cloud conversion resolves it, so `--resolution`
    // means the same thing for both and the two stay comparable.
    let bbox = compute_bounds(&triangles);
    let density = match options.resolution {
        Some(resolution) => {
            let diagonal = (bbox.max - bbox.min).length();
            if resolution <= 0.0 || !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(ConvertError::InvalidDensity);
            }
            (resolution / diagonal).powi(3)
        }
        None => options.density,
    };
    if !density.is_finite() || density <= 0.0 {
        return Err(ConvertError::InvalidDensity);
    }
    let surface_density = density.powf(2.0 / 3.0) * options.surface_density_scale;
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(options.seed);

    let converted = materials
        .iter()
        .map(|material| {
            let base = material.base_color.truncate();
            let metallic = material.metallic.clamp(0.0, 1.0);
            vol::relight::Material {
                // The specular workflow: interpolating towards the base colour
                // for a metal and towards a dielectric's 4% for everything else.
                albedo: (base * (1.0 - metallic)).into(),
                roughness: material.roughness.clamp(0.02, 1.0),
                specular_f0: (glam::Vec3::splat(0.04)
                    + (base - glam::Vec3::splat(0.04)) * metallic)
                    .into(),
                _padding: 0.0,
            }
        })
        .collect::<Vec<_>>();

    let mut surfels = Vec::new();
    for tri in &triangles {
        let count = (tri.area * surface_density).ceil() as u32;
        if count == 0 {
            continue;
        }
        // Discs large enough that randomly placed ones actually cover the
        // triangle. Giving each its exact share of the area would only work if
        // they tiled; scattered at random, the fraction left uncovered is
        // `exp(-k)` for a total disc area of `k` times the surface, so three
        // times over leaves about five percent showing through. Sizing them
        // for their share instead is what perforates the result.
        const OVERLAP: f32 = 3.0;
        let radius = (OVERLAP * tri.area / count as f32 / std::f32::consts::PI)
            .max(1e-12)
            .sqrt()
            * options.surface_scale;
        let material = &materials[tri.material];
        for _ in 0..count {
            let (u, v) = sample_barycentric(&mut rng);
            let w = 1.0 - u - v;
            let base = sample_triangle_color(tri, material, u, v, w);
            let coverage = material_alpha_coverage(material, base.w);
            if coverage <= 0.0 || coverage < options.alpha_threshold {
                continue;
            }
            surfels.push(vol::relight::Surfel {
                center: (tri.v0 * u + tri.v1 * v + tri.v2 * w).into(),
                radius,
                normal: tri.normal.into(),
                material: tri.material as u32,
            });
        }
    }

    let model = vol::relight::RelightModel {
        kernel: vol::relight::ParticleKernel::Compact,
        surfels,
        materials: converted,
    };
    model
        .validate()
        .unwrap_or_else(|err| panic!("conversion produced an invalid model: {err}"));
    Ok(model)
}

/// Extract the source mesh in the form the GPU reference renderer wants.
///
/// Colour is the converter's own per-sample colour function evaluated at each
/// triangle's centroid, left in **linear light**: the renderer applies the
/// ambient gain and the sRGB encode, mirroring [`display_color`]. That is
/// exact for flat base-colour materials. For textured materials it is a
/// per-triangle average, so a texture varying within one triangle is the one
/// place the reference is an approximation rather than ground truth.
pub fn reference_mesh_from_gltf(
    path: impl AsRef<path::Path>,
) -> Result<vol::ReferenceMesh, ConvertError> {
    let (triangles, materials) = gather_scene(path.as_ref())?;

    // The acceleration structure wants indexed geometry; conversion works on
    // loose triangles, so emit three vertices each and keep the order aligned
    // with `triangle_colors` (the shader indexes it by primitive index).
    let mut positions = Vec::with_capacity(triangles.len() * 3);
    let mut indices = Vec::with_capacity(triangles.len());
    let mut triangle_colors = Vec::with_capacity(triangles.len());
    for triangle in triangles.iter() {
        let base = positions.len() as u32;
        positions.push(triangle.v0.to_array());
        positions.push(triangle.v1.to_array());
        positions.push(triangle.v2.to_array());
        indices.push([base, base + 1, base + 2]);
        let third = 1.0 / 3.0;
        let color =
            sample_triangle_color(triangle, &materials[triangle.material], third, third, third);
        triangle_colors.push(color.truncate().to_array());
    }

    Ok(vol::ReferenceMesh {
        positions,
        indices,
        triangle_colors,
    })
}

pub fn convert_gltf(
    path: impl AsRef<path::Path>,
    options: &ConvertOptions,
) -> Result<vol::PointCloudModel, ConvertError> {
    let (triangles, materials) = gather_scene(path.as_ref())?;

    let bbox = compute_bounds(&triangles);

    // A scale-invariant request is resolved against this asset's own extent,
    // so the same flags give comparable clouds for meshes authored in metres,
    // centimetres, or arbitrary engine units.
    let density = match options.resolution {
        Some(resolution) => {
            let diagonal = (bbox.max - bbox.min).length();
            if resolution <= 0.0 || !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(ConvertError::InvalidDensity);
            }
            (resolution / diagonal).powi(3)
        }
        None => options.density,
    };
    // NaN fails `is_finite`, so it is rejected here rather than reaching the
    // sampling loops as a silently empty grid.
    if !density.is_finite() || density <= 0.0 {
        return Err(ConvertError::InvalidDensity);
    }

    let avg_color = compute_average_color(&triangles, &materials);

    let interior_density = density * options.interior_density_scale;
    let mut points = Vec::new();
    let mut sh_coefficients = Vec::new();
    let mut rotations = Vec::new();
    let mut scales = Vec::new();

    if interior_density > 0.0 {
        let inside = InsideTester::new(&triangles);
        // Seeded separately from the surface sampler so that changing one
        // stage's rate does not reshuffle the other's samples.
        let mut interior_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            options.seed ^ 0x0117_e210_7ab0,
        );
        // Only the triangulated output benefits; see `DEFAULT_INTERIOR_JITTER`.
        let jitter = options
            .interior_jitter
            .unwrap_or(match options.output {
                OutputKind::RadFoam => DEFAULT_INTERIOR_JITTER,
                OutputKind::Gaussian => 0.0,
            })
            .max(0.0);
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
                    if inside.is_inside(p, &triangles) {
                        let color = display_color(avg_color, options.ambient);
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
                                    let mut sp = base + offset;
                                    if jitter > 0.0 {
                                        use rand::RngExt as _;
                                        let amount = jitter * sub_spacing;
                                        let mut axis = || {
                                            let r: f32 = interior_rng.random();
                                            (r - 0.5) * amount
                                        };
                                        sp += glam::Vec3::new(axis(), axis(), axis());
                                    }
                                    push_point(
                                        &mut points,
                                        &mut sh_coefficients,
                                        &mut rotations,
                                        &mut scales,
                                        sp,
                                        color,
                                        stored_density(
                                            options.interior_opacity,
                                            sub_spacing,
                                            options.output,
                                        ),
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

    // Transparent cells for the space around the object. Without these, every
    // exterior point belongs to an unbounded cell owned by an opaque surface
    // site and the object cannot be viewed from outside at all — see
    // `ConvertOptions::exterior_density_scale`.
    let exterior_density = density
        * options
            .exterior_density_scale
            .unwrap_or(match options.output {
                OutputKind::RadFoam => DEFAULT_EXTERIOR_DENSITY_SCALE,
                OutputKind::Gaussian => 0.0,
            });
    if exterior_density > 0.0 {
        let inside = InsideTester::new(&triangles);
        let spacing = exterior_density.powf(-1.0 / 3.0);
        let padding = (bbox.max - bbox.min).length() * options.exterior_padding.max(0.0);
        let lo = bbox.min - glam::Vec3::splat(padding);
        let hi = bbox.max + glam::Vec3::splat(padding);
        let mut z = lo.z + 0.5 * spacing;
        while z <= hi.z {
            let mut y = lo.y + 0.5 * spacing;
            while y <= hi.y {
                let mut x = lo.x + 0.5 * spacing;
                while x <= hi.x {
                    let p = glam::Vec3::new(x, y, z);
                    // Only fill genuinely empty space; interior sampling owns
                    // the inside, at its own much finer rate.
                    if !inside.is_inside(p, &triangles) {
                        push_point(
                            &mut points,
                            &mut sh_coefficients,
                            &mut rotations,
                            &mut scales,
                            p,
                            glam::Vec3::ZERO,
                            0.0,
                            spacing * 0.5,
                            None,
                            options,
                        );
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
        let curvature_factors = if options.curvature_boost > 0.0 {
            let curvatures = compute_triangle_curvatures(&triangles);
            let areas = triangles
                .iter()
                .map(|triangle| triangle.area)
                .collect::<Vec<_>>();
            normalized_curvature_factors(&areas, &curvatures, options.curvature_boost)
        } else {
            vec![1.0; triangles.len()]
        };
        for (ti, tri) in triangles.iter().enumerate() {
            let local_surface_density = surface_density * curvature_factors[ti];
            let count = (tri.area * local_surface_density).ceil() as u32;
            for _ in 0..count {
                let (u, v) = sample_barycentric(&mut rng);
                let w = 1.0 - u - v;
                let p = tri.v0 * u + tri.v1 * v + tri.v2 * w;
                let mat = &materials[tri.material];
                let base = sample_triangle_color(tri, mat, u, v, w);
                let color = display_color(base.truncate(), options.ambient);
                let coverage = material_alpha_coverage(mat, base.w);
                if coverage <= 0.0 || coverage < options.alpha_threshold {
                    continue;
                }

                push_point(
                    &mut points,
                    &mut sh_coefficients,
                    &mut rotations,
                    &mut scales,
                    p,
                    color,
                    stored_density(
                        options.surface_opacity * coverage,
                        // Surface samples sit on a sheet; their characteristic
                        // cell extent is the local sample spacing.
                        local_surface_density.max(1e-12).powf(-0.5),
                        options.output,
                    ),
                    surface_point_scale(local_surface_density) * options.surface_scale,
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
        surface_normals: None,
        surface_offsets: None,
        surface_detail: None,
        surface_color_coefficients: None,
        spherical_voronoi: None,
    };

    match options.output {
        OutputKind::Gaussian => {
            model.transforms = Some(vol::Transforms { rotations, scales });
        }
        OutputKind::RadFoam => {
            model.adjacency = Some(match options.topology {
                Topology::Exact => vol::try_compute_adjacency_default(&model.points)?,
                #[cfg(feature = "qhull")]
                Topology::Qhull => vol::compute_adjacency_qhull_default(&model.points),
                #[cfg(not(feature = "qhull"))]
                Topology::Qhull => return Err(ConvertError::QhullUnavailable),
            });
            if options.spring_iterations > 0 {
                vol::spring_relax(&mut model, options.spring_iterations, options.spring_step);
            }
            if options.assign_radii {
                model.radii = Some(vol::radii_from_nearest_neighbour(
                    &model.points,
                    options.radius_factor,
                ));
                // Radius assignment changes the representation to PowerFoam;
                // rebuild the required ball-overlap graph instead of retaining
                // an unrelated unweighted Delaunay graph.
                model.compute_adjacency_default();
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
    default_material: usize,
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
            let material_index = primitive.material().index().unwrap_or(default_material);
            let material = materials
                .get(material_index)
                .ok_or(ConvertError::MissingMeshData)?;
            let uvs = match reader.read_tex_coords(material.tex_coord) {
                Some(uv) => uv.into_f32().map(glam::Vec2::from).collect::<Vec<_>>(),
                None if material.base_color_texture.is_some() => {
                    return Err(ConvertError::MissingMeshData);
                }
                None => vec![glam::Vec2::ZERO; positions.len()],
            };
            let colors = reader
                .read_colors(0)
                .map(|colors| {
                    colors
                        .into_rgba_f32()
                        .map(glam::Vec4::from)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![glam::Vec4::ONE; positions.len()]);
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
            let indices = reader.read_indices().map_or_else(
                || (0..positions.len() as u32).collect::<Vec<_>>(),
                |indices| indices.into_u32().collect::<Vec<_>>(),
            );
            if indices.len() % 3 != 0 {
                return Err(ConvertError::MissingMeshData);
            }

            for tri in indices.chunks_exact(3) {
                let i0 = tri[0] as usize;
                let i1 = tri[1] as usize;
                let i2 = tri[2] as usize;
                let v0 = *positions.get(i0).ok_or(ConvertError::MissingMeshData)?;
                let v1 = *positions.get(i1).ok_or(ConvertError::MissingMeshData)?;
                let v2 = *positions.get(i2).ok_or(ConvertError::MissingMeshData)?;
                let n0 = *normals.get(i0).ok_or(ConvertError::MissingMeshData)?;
                let n1 = *normals.get(i1).ok_or(ConvertError::MissingMeshData)?;
                let n2 = *normals.get(i2).ok_or(ConvertError::MissingMeshData)?;
                let normal = if n0 != glam::Vec3::ZERO {
                    let n = (n0 + n1 + n2) / 3.0;
                    n.normalize_or_zero()
                } else {
                    (v1 - v0).cross(v2 - v0).normalize_or_zero()
                };
                let area = 0.5 * (v1 - v0).cross(v2 - v0).length();
                triangles.push(Triangle {
                    v0,
                    v1,
                    v2,
                    uv0: *uvs.get(i0).ok_or(ConvertError::MissingMeshData)?,
                    uv1: *uvs.get(i1).ok_or(ConvertError::MissingMeshData)?,
                    uv2: *uvs.get(i2).ok_or(ConvertError::MissingMeshData)?,
                    color0: *colors.get(i0).ok_or(ConvertError::MissingMeshData)?,
                    color1: *colors.get(i1).ok_or(ConvertError::MissingMeshData)?,
                    color2: *colors.get(i2).ok_or(ConvertError::MissingMeshData)?,
                    normal,
                    material: material_index,
                    area,
                });
            }
        }
    }

    for child in node.children() {
        gather_node_triangles(
            &child,
            transform,
            buffers,
            materials,
            default_material,
            triangles,
        )?;
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

fn compute_average_color(triangles: &[Triangle], materials: &[MaterialInfo]) -> glam::Vec3 {
    let mut sum = glam::Vec3::ZERO;
    let mut total_area = 0.0;
    for triangle in triangles {
        let color = sample_triangle_color(
            triangle,
            &materials[triangle.material],
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        );
        sum += color.truncate() * triangle.area;
        total_area += triangle.area;
    }
    if total_area > 0.0 {
        sum / total_area
    } else {
        glam::Vec3::ZERO
    }
}

/// Oblique parity directions. Three independent casts with a majority vote
/// tolerate the occasional degenerate hit; the directions are deliberately not
/// axis aligned, because axis-aligned geometry makes axis-aligned rays graze
/// shared edges and coplanar faces far more often.
const INSIDE_RAYS: [glam::Vec3; 3] = [
    glam::Vec3::new(0.321, 0.571, 0.755),
    glam::Vec3::new(-0.742, 0.201, 0.639),
    glam::Vec3::new(0.189, -0.911, 0.367),
];

/// Triangles bucketed by their footprint on the plane perpendicular to one
/// parity direction, in CSR form.
///
/// A ray along that direction keeps a fixed position in the plane, so it can
/// only hit triangles whose projected bounding box covers that position. The
/// narrowing is therefore exact, not conservative-with-error: the same
/// triangles pass the intersection test as in an exhaustive scan.
struct DirectionIndex {
    dir: glam::Vec3,
    u: glam::Vec3,
    v: glam::Vec3,
    min: glam::Vec2,
    inv_cell: glam::Vec2,
    dims: [usize; 2],
    offsets: Vec<u32>,
    indices: Vec<u32>,
}

impl DirectionIndex {
    fn new(dir: glam::Vec3, triangles: &[Triangle]) -> Self {
        let dir = dir.normalize();
        // Any orthonormal pair spanning the plane perpendicular to `dir`.
        let helper = if dir.z.abs() < 0.9 {
            glam::Vec3::Z
        } else {
            glam::Vec3::X
        };
        let u = dir.cross(helper).normalize();
        let v = dir.cross(u);

        // Roughly two triangles per cell keeps candidate lists short without
        // the bucket array dominating memory.
        let side = ((triangles.len() as f32 / 2.0).sqrt().ceil() as usize).clamp(1, 1024);
        let dims = [side, side];

        let project = |p: glam::Vec3| glam::Vec2::new(p.dot(u), p.dot(v));
        let mut min = glam::Vec2::splat(f32::INFINITY);
        let mut max = glam::Vec2::splat(f32::NEG_INFINITY);
        let bounds = triangles
            .iter()
            .map(|tri| {
                let (a, b, c) = (project(tri.v0), project(tri.v1), project(tri.v2));
                let lo = a.min(b).min(c);
                let hi = a.max(b).max(c);
                min = min.min(lo);
                max = max.max(hi);
                (lo, hi)
            })
            .collect::<Vec<_>>();
        if !min.is_finite() || !max.is_finite() {
            min = glam::Vec2::ZERO;
            max = glam::Vec2::ZERO;
        }

        // A degenerate axis would divide by zero; a single cell is correct
        // there because every triangle projects onto the same position.
        let extent = (max - min).max(glam::Vec2::splat(f32::MIN_POSITIVE));
        let inv_cell = glam::Vec2::new(dims[0] as f32, dims[1] as f32) / extent;

        let cell_range = |lo: glam::Vec2, hi: glam::Vec2| {
            let to_cell = |value: f32, axis: usize| {
                let scaled = (value - min[axis]) * inv_cell[axis];
                (scaled.floor().max(0.0) as usize).min(dims[axis] - 1)
            };
            [
                (to_cell(lo.x, 0), to_cell(hi.x, 0)),
                (to_cell(lo.y, 1), to_cell(hi.y, 1)),
            ]
        };

        // Count first, then fill: CSR without per-cell allocations.
        let mut offsets = vec![0u32; dims[0] * dims[1] + 1];
        for &(lo, hi) in bounds.iter() {
            let [(x0, x1), (y0, y1)] = cell_range(lo, hi);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    offsets[y * dims[0] + x + 1] += 1;
                }
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor = offsets.clone();
        let mut indices = vec![0u32; offsets[offsets.len() - 1] as usize];
        for (ti, &(lo, hi)) in bounds.iter().enumerate() {
            let [(x0, x1), (y0, y1)] = cell_range(lo, hi);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let cell = y * dims[0] + x;
                    indices[cursor[cell] as usize] = ti as u32;
                    cursor[cell] += 1;
                }
            }
        }

        Self {
            dir,
            u,
            v,
            min,
            inv_cell,
            dims,
            offsets,
            indices,
        }
    }

    /// Triangles a ray from `origin` along `self.dir` could possibly hit. A
    /// point projecting outside the built extent has none.
    fn candidates(&self, origin: glam::Vec3) -> &[u32] {
        let projected = glam::Vec2::new(origin.dot(self.u), origin.dot(self.v));
        let mut cell = [0usize; 2];
        for axis in 0..2 {
            let scaled = (projected[axis] - self.min[axis]) * self.inv_cell[axis];
            // Rejects NaN as well as negatives, so the cast below cannot
            // silently land in cell 0.
            if !scaled.is_finite() || scaled < 0.0 {
                return &[];
            }
            let slot = scaled.floor() as usize;
            if slot >= self.dims[axis] {
                // The upper boundary belongs to the last cell; anything beyond
                // it is outside every triangle's footprint.
                if slot > self.dims[axis] {
                    return &[];
                }
                cell[axis] = self.dims[axis] - 1;
            } else {
                cell[axis] = slot;
            }
        }
        let flat = cell[1] * self.dims[0] + cell[0];
        let start = self.offsets[flat] as usize;
        let end = self.offsets[flat + 1] as usize;
        &self.indices[start..end]
    }
}

/// Even-odd containment test against a closed mesh, accelerated by one
/// projected index per parity direction.
///
/// Building the indices costs `O(triangles)`; each query then touches only the
/// triangles sharing its cell, instead of the whole mesh. The exhaustive
/// version made interior sampling `O(grid^3 * triangles)`, which dominated
/// conversion so heavily that only toy assets finished.
struct InsideTester {
    indices: [DirectionIndex; 3],
}

impl InsideTester {
    fn new(triangles: &[Triangle]) -> Self {
        Self {
            indices: INSIDE_RAYS.map(|dir| DirectionIndex::new(dir, triangles)),
        }
    }

    fn is_inside(&self, point: glam::Vec3, triangles: &[Triangle]) -> bool {
        let mut odd_hits = 0u32;
        for index in self.indices.iter() {
            let dir = index.dir;
            let origin = point + dir * 1e-4;
            let mut hits = 0u32;
            for &ti in index.candidates(origin) {
                if ray_intersects_triangle(origin, dir, &triangles[ti as usize]) {
                    hits += 1;
                }
            }
            if hits % 2 == 1 {
                odd_hits += 1;
            }
        }

        odd_hits >= 2
    }
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
        tex.sample(material.uv_transform.apply(uv)) * material.base_color
    } else {
        material.base_color
    }
}

fn sample_triangle_color(
    triangle: &Triangle,
    material: &MaterialInfo,
    u: f32,
    v: f32,
    w: f32,
) -> glam::Vec4 {
    let uv = triangle.uv0 * u + triangle.uv1 * v + triangle.uv2 * w;
    let vertex_color = triangle.color0 * u + triangle.color1 * v + triangle.color2 * w;
    sample_material_color(material, uv) * vertex_color
}

fn material_alpha_coverage(material: &MaterialInfo, alpha: f32) -> f32 {
    match material.alpha_mode {
        gltf::material::AlphaMode::Opaque => 1.0,
        gltf::material::AlphaMode::Mask => {
            if alpha >= material.alpha_cutoff {
                1.0
            } else {
                0.0
            }
        }
        gltf::material::AlphaMode::Blend => alpha.clamp(0.0, 1.0),
    }
}

fn surface_point_scale(surface_density: f32) -> f32 {
    0.5 / surface_density.sqrt()
}

/// Translate an alpha-like opacity knob into the value the target backend
/// stores in `point.w`.
///
/// Gaussian splatting uses that field directly as alpha. RadFoam integrates
/// `alpha = 1 - exp(-w * dt)` along the path, so `w` is a density per unit
/// length: storing an alpha there makes coverage depend on cell size, and the
/// object grows *more* transparent as sampling gets finer. Solving for the
/// density that yields `opacity` across one cell removes that dependence.
fn stored_density(opacity: f32, cell_size: f32, output: OutputKind) -> f32 {
    match output {
        OutputKind::Gaussian => opacity,
        OutputKind::RadFoam => {
            // Fully opaque would need infinite density; cap so a saturated
            // request stays finite and well conditioned.
            let alpha = opacity.clamp(0.0, 0.999);
            if alpha <= 0.0 {
                0.0
            } else {
                -(1.0 - alpha).ln() / cell_size.max(1e-6)
            }
        }
    }
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
    let unorm16 = |bytes: &[u8]| {
        let value = u16::from_ne_bytes([bytes[0], bytes[1]]);
        ((u32::from(value) * 255 + 32_767) / 65_535) as u8
    };
    let float32 = |bytes: &[u8]| {
        let value = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    };
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
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        gltf::image::Format::R8 => image
            .pixels
            .iter()
            .flat_map(|c| [*c, *c, *c, 255])
            .collect(),
        gltf::image::Format::R16 => image
            .pixels
            .chunks_exact(2)
            .flat_map(|c| {
                let value = unorm16(c);
                [value, value, value, 255]
            })
            .collect(),
        gltf::image::Format::R16G16 => image
            .pixels
            .chunks_exact(4)
            .flat_map(|c| {
                let value = unorm16(&c[..2]);
                [value, value, value, unorm16(&c[2..])]
            })
            .collect(),
        gltf::image::Format::R16G16B16 => image
            .pixels
            .chunks_exact(6)
            .flat_map(|c| [unorm16(&c[..2]), unorm16(&c[2..4]), unorm16(&c[4..]), 255])
            .collect(),
        gltf::image::Format::R16G16B16A16 => image
            .pixels
            .chunks_exact(8)
            .flat_map(|c| {
                [
                    unorm16(&c[..2]),
                    unorm16(&c[2..4]),
                    unorm16(&c[4..6]),
                    unorm16(&c[6..]),
                ]
            })
            .collect(),
        gltf::image::Format::R32G32B32FLOAT => image
            .pixels
            .chunks_exact(12)
            .flat_map(|c| [float32(&c[..4]), float32(&c[4..8]), float32(&c[8..]), 255])
            .collect(),
        gltf::image::Format::R32G32B32A32FLOAT => image
            .pixels
            .chunks_exact(16)
            .flat_map(|c| {
                [
                    float32(&c[..4]),
                    float32(&c[4..8]),
                    float32(&c[8..12]),
                    float32(&c[12..]),
                ]
            })
            .collect(),
    }
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn display_color(linear: glam::Vec3, ambient: glam::Vec3) -> glam::Vec3 {
    // glTF base-color factors are linear and base-color textures are decoded
    // to linear by `Texture::fetch`. Apply lighting there, then cross the
    // PointCloudModel boundary as the display-referred sRGB code values used
    // by training, SH evaluation, metrics, and presentation.
    let color = (linear * ambient).clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
    glam::Vec3::new(
        linear_to_srgb(color.x),
        linear_to_srgb(color.y),
        linear_to_srgb(color.z),
    )
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
    let sh_rest_per_channel = sh_component_count.saturating_sub(1);
    let sh_rest_count = sh_rest_per_channel * 3;
    let ply_rotation =
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), -std::f32::consts::FRAC_PI_2);
    let inv_rotation = ply_rotation.inverse();

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
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

        for property in 0..sh_rest_count {
            let channel = property / sh_rest_per_channel;
            let component = property % sh_rest_per_channel + 1;
            let coeff = model.sh_coefficients[base + component * 3 + channel];
            file.write_all(&coeff.to_le_bytes())?;
        }
    }

    file.flush()?;
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
    let sh_rest_per_channel = sh_component_count.saturating_sub(1);
    let sh_rest_count = sh_rest_per_channel * 3;
    let ply_rotation =
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), -std::f32::consts::FRAC_PI_2);
    let inv_rotation = ply_rotation.inverse();

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
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

        for property in 0..sh_rest_count {
            let channel = property / sh_rest_per_channel;
            let component = property % sh_rest_per_channel + 1;
            let coeff = model.sh_coefficients[base + component * 3 + channel];
            write!(file, " {}", coeff)?;
        }
        writeln!(file)?;
    }

    file.flush()?;
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
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);

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
    writeln!(file, "property float blade_sh_dc_0")?;
    writeln!(file, "property float blade_sh_dc_1")?;
    writeln!(file, "property float blade_sh_dc_2")?;
    for k in 0..sh_rest {
        writeln!(file, "property float color_sh_{k}")?;
    }
    if model.radii.is_some() {
        writeln!(file, "property float radius")?;
    }
    if model.surface_normals.is_some() {
        writeln!(file, "property float nx")?;
        writeln!(file, "property float ny")?;
        writeln!(file, "property float nz")?;
    }
    if model.surface_offsets.is_some() {
        writeln!(file, "property float surface_offset")?;
    }
    if model.surface_detail.is_some() {
        for component in 0..vol::SURFACE_DETAIL_SITES * 3 {
            writeln!(
                file,
                "property float blade_surface_detail_offset_{component}"
            )?;
        }
        for site in 0..vol::SURFACE_DETAIL_SITES {
            writeln!(file, "property float blade_surface_detail_height_{site}")?;
        }
        for component in 0..vol::SURFACE_DETAIL_SITES * 3 {
            writeln!(
                file,
                "property float blade_surface_detail_color_{component}"
            )?;
        }
    }
    if model.surface_color_coefficients.is_some() {
        for component in 0..vol::SURFACE_COLOR_COMPONENTS * 3 {
            writeln!(file, "property float blade_surface_color_{component}")?;
        }
    }
    if model.spherical_voronoi.is_some() {
        for component in 0..vol::SPHERICAL_VORONOI_SITES * 3 {
            writeln!(
                file,
                "property float blade_spherical_voronoi_axis_{component}"
            )?;
        }
        for component in 0..vol::SPHERICAL_VORONOI_SITES * 3 {
            writeln!(
                file,
                "property float blade_spherical_voronoi_color_{component}"
            )?;
        }
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
        let base = i * sh_block;
        write!(
            file,
            "{} {} {} {} {} {} {} {} {} {} {}",
            point.x,
            point.y,
            point.z,
            point.w,
            end_off,
            r,
            g,
            b,
            model.sh_coefficients[base],
            model.sh_coefficients[base + 1],
            model.sh_coefficients[base + 2],
        )?;
        for j in 0..sh_rest {
            let comp = 1 + j / 3;
            let ch = j % 3;
            let value = model.sh_coefficients[base + 3 * comp + ch];
            write!(file, " {value}")?;
        }
        if let Some(ref radii) = model.radii {
            write!(file, " {}", radii[i])?;
        }
        if let Some(ref normals) = model.surface_normals {
            let normal = normals[i];
            write!(file, " {} {} {}", normal.x, normal.y, normal.z)?;
        }
        if let Some(ref offsets) = model.surface_offsets {
            write!(file, " {}", offsets[i])?;
        }
        if let Some(ref detail) = model.surface_detail {
            let base = i * vol::SURFACE_DETAIL_SITES;
            for value in &detail.offsets[base..base + vol::SURFACE_DETAIL_SITES] {
                write!(file, " {} {} {}", value.x, value.y, value.z)?;
            }
            for value in &detail.heights[base..base + vol::SURFACE_DETAIL_SITES] {
                write!(file, " {value}")?;
            }
            for value in &detail.colors[base..base + vol::SURFACE_DETAIL_SITES] {
                write!(file, " {} {} {}", value.x, value.y, value.z)?;
            }
        }
        if let Some(ref coefficients) = model.surface_color_coefficients {
            let stride = vol::SURFACE_COLOR_COMPONENTS * 3;
            for value in &coefficients[i * stride..(i + 1) * stride] {
                write!(file, " {value}")?;
            }
        }
        if let Some(ref spherical_voronoi) = model.spherical_voronoi {
            let base = i * vol::SPHERICAL_VORONOI_SITES;
            for value in &spherical_voronoi.axes[base..base + vol::SPHERICAL_VORONOI_SITES] {
                write!(file, " {} {} {}", value.x, value.y, value.z)?;
            }
            for value in &spherical_voronoi.colors[base..base + vol::SPHERICAL_VORONOI_SITES] {
                write!(file, " {} {} {}", value.x, value.y, value.z)?;
            }
        }
        writeln!(file)?;
    }

    for idx in &adjacency.neighbors {
        writeln!(file, "{}", idx)?;
    }

    file.flush()?;
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
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);

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
    writeln!(file, "property float blade_sh_dc_0")?;
    writeln!(file, "property float blade_sh_dc_1")?;
    writeln!(file, "property float blade_sh_dc_2")?;
    for k in 0..sh_rest {
        writeln!(file, "property float color_sh_{k}")?;
    }
    if model.radii.is_some() {
        writeln!(file, "property float radius")?;
    }
    if model.surface_normals.is_some() {
        writeln!(file, "property float nx")?;
        writeln!(file, "property float ny")?;
        writeln!(file, "property float nz")?;
    }
    if model.surface_offsets.is_some() {
        writeln!(file, "property float surface_offset")?;
    }
    if model.surface_detail.is_some() {
        for component in 0..vol::SURFACE_DETAIL_SITES * 3 {
            writeln!(
                file,
                "property float blade_surface_detail_offset_{component}"
            )?;
        }
        for site in 0..vol::SURFACE_DETAIL_SITES {
            writeln!(file, "property float blade_surface_detail_height_{site}")?;
        }
        for component in 0..vol::SURFACE_DETAIL_SITES * 3 {
            writeln!(
                file,
                "property float blade_surface_detail_color_{component}"
            )?;
        }
    }
    if model.surface_color_coefficients.is_some() {
        for component in 0..vol::SURFACE_COLOR_COMPONENTS * 3 {
            writeln!(file, "property float blade_surface_color_{component}")?;
        }
    }
    if model.spherical_voronoi.is_some() {
        for component in 0..vol::SPHERICAL_VORONOI_SITES * 3 {
            writeln!(
                file,
                "property float blade_spherical_voronoi_axis_{component}"
            )?;
        }
        for component in 0..vol::SPHERICAL_VORONOI_SITES * 3 {
            writeln!(
                file,
                "property float blade_spherical_voronoi_color_{component}"
            )?;
        }
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
        let base = i * sh_block;

        file.write_all(&point.x.to_le_bytes())?;
        file.write_all(&point.y.to_le_bytes())?;
        file.write_all(&point.z.to_le_bytes())?;
        file.write_all(&point.w.to_le_bytes())?;
        file.write_all(&end_off.to_le_bytes())?;
        file.write_all(&[r, g, b])?;
        file.write_all(&model.sh_coefficients[base].to_le_bytes())?;
        file.write_all(&model.sh_coefficients[base + 1].to_le_bytes())?;
        file.write_all(&model.sh_coefficients[base + 2].to_le_bytes())?;
        // color_sh_* layout per RadFoam upstream loader: index j maps to
        // SH component `1 + j/3`, channel `j%3`. Exact DC lives in the
        // blade_sh_dc_* extension above; RGB remains an upstream-compatible
        // 8-bit preview.
        for j in 0..sh_rest {
            let comp = 1 + j / 3;
            let ch = j % 3;
            let value = model.sh_coefficients[base + 3 * comp + ch];
            file.write_all(&value.to_le_bytes())?;
        }
        if let Some(ref radii) = model.radii {
            file.write_all(&radii[i].to_le_bytes())?;
        }
        if let Some(ref normals) = model.surface_normals {
            let normal = normals[i];
            file.write_all(&normal.x.to_le_bytes())?;
            file.write_all(&normal.y.to_le_bytes())?;
            file.write_all(&normal.z.to_le_bytes())?;
        }
        if let Some(ref offsets) = model.surface_offsets {
            file.write_all(&offsets[i].to_le_bytes())?;
        }
        if let Some(ref detail) = model.surface_detail {
            let base = i * vol::SURFACE_DETAIL_SITES;
            for value in &detail.offsets[base..base + vol::SURFACE_DETAIL_SITES] {
                file.write_all(&value.x.to_le_bytes())?;
                file.write_all(&value.y.to_le_bytes())?;
                file.write_all(&value.z.to_le_bytes())?;
            }
            for value in &detail.heights[base..base + vol::SURFACE_DETAIL_SITES] {
                file.write_all(&value.to_le_bytes())?;
            }
            for value in &detail.colors[base..base + vol::SURFACE_DETAIL_SITES] {
                file.write_all(&value.x.to_le_bytes())?;
                file.write_all(&value.y.to_le_bytes())?;
                file.write_all(&value.z.to_le_bytes())?;
            }
        }
        if let Some(ref coefficients) = model.surface_color_coefficients {
            let stride = vol::SURFACE_COLOR_COMPONENTS * 3;
            for value in &coefficients[i * stride..(i + 1) * stride] {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        if let Some(ref spherical_voronoi) = model.spherical_voronoi {
            let base = i * vol::SPHERICAL_VORONOI_SITES;
            for value in &spherical_voronoi.axes[base..base + vol::SPHERICAL_VORONOI_SITES] {
                file.write_all(&value.x.to_le_bytes())?;
                file.write_all(&value.y.to_le_bytes())?;
                file.write_all(&value.z.to_le_bytes())?;
            }
            for value in &spherical_voronoi.colors[base..base + vol::SPHERICAL_VORONOI_SITES] {
                file.write_all(&value.x.to_le_bytes())?;
                file.write_all(&value.y.to_le_bytes())?;
                file.write_all(&value.z.to_le_bytes())?;
            }
        }
    }

    for idx in &adjacency.neighbors {
        file.write_all(&idx.to_le_bytes())?;
    }

    file.flush()?;
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
    fn srgb_transfer_roundtrips_code_values() {
        for value in [0.0, 0.01, 0.25, 0.5, 0.9, 1.0] {
            let roundtrip = linear_to_srgb(srgb_to_linear(value));
            assert!((roundtrip - value).abs() < 1e-6, "{value} -> {roundtrip}");
        }
    }

    #[test]
    fn display_color_encodes_after_linear_ambient_gain() {
        let color = display_color(
            glam::Vec3::new(0.25, 0.5, 0.75),
            glam::Vec3::new(2.0, 1.0, 0.0),
        );
        assert!((color.x - 0.735_357).abs() < 1e-6);
        assert!((color.y - 0.735_357).abs() < 1e-6);
        assert_eq!(color.z, 0.0);
    }

    #[test]
    fn curvature_factors_redistribute_without_changing_area_budget() {
        let areas = [1.0, 3.0];
        let factors = normalized_curvature_factors(&areas, &[0.0, 1.0], 3.0);
        assert!(factors[1] > factors[0]);
        let weighted_area = areas
            .iter()
            .zip(&factors)
            .map(|(&area, &factor)| area * factor)
            .sum::<f32>();
        assert!((weighted_area - areas.iter().sum::<f32>()).abs() < 1e-6);
        assert_eq!(
            normalized_curvature_factors(&areas, &[0.25, 1.0], 0.0),
            [1.0, 1.0]
        );
    }

    #[test]
    fn surface_footprint_tracks_local_area_sampling_spacing() {
        assert_eq!(surface_point_scale(4.0), 0.25);
        assert_eq!(surface_point_scale(16.0), 0.125);
    }

    #[test]
    fn texture_coordinates_follow_gltf_wrap_modes() {
        assert_eq!(
            wrap_coordinate(-0.25, gltf::texture::WrappingMode::Repeat),
            0.75
        );
        assert_eq!(
            wrap_coordinate(1.25, gltf::texture::WrappingMode::Repeat),
            0.25
        );
        assert_eq!(
            wrap_coordinate(-0.25, gltf::texture::WrappingMode::MirroredRepeat),
            0.25
        );
        assert_eq!(
            wrap_coordinate(1.25, gltf::texture::WrappingMode::MirroredRepeat),
            0.75
        );
        assert_eq!(
            wrap_coordinate(-0.25, gltf::texture::WrappingMode::ClampToEdge),
            0.0
        );
        assert_eq!(
            wrap_coordinate(1.25, gltf::texture::WrappingMode::ClampToEdge),
            1.0
        );
    }

    #[test]
    fn texture_sampling_uses_the_gltf_upper_left_origin() {
        let texture = Texture {
            width: 1,
            height: 2,
            data: vec![255, 0, 0, 255, 0, 0, 255, 255],
            wrap_s: gltf::texture::WrappingMode::ClampToEdge,
            wrap_t: gltf::texture::WrappingMode::ClampToEdge,
        };
        assert_eq!(
            texture.sample(glam::Vec2::new(0.0, 0.0)),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0)
        );
        assert_eq!(
            texture.sample(glam::Vec2::new(0.0, 1.0)),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0)
        );
    }

    #[test]
    fn texture_transform_applies_scale_then_rotation_then_offset() {
        let transform = UvTransform {
            offset: glam::Vec2::new(0.0, 1.0),
            rotation: std::f32::consts::FRAC_PI_2,
            scale: glam::Vec2::splat(0.5),
        };
        let actual = transform.apply(glam::Vec2::X);
        assert!((actual - glam::Vec2::new(0.0, 1.5)).length() < 1e-6);
    }

    #[test]
    fn texture_transform_parser_honors_texcoord_override() {
        let source = br#"{
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_texture_transform"],
            "extensionsRequired": ["KHR_texture_transform"],
            "images": [{"uri": "unused.png"}],
            "textures": [{"source": 0}],
            "materials": [{
                "pbrMetallicRoughness": {
                    "baseColorTexture": {
                        "index": 0,
                        "texCoord": 0,
                        "extensions": {
                            "KHR_texture_transform": {
                                "offset": [0, 1],
                                "rotation": 1.57079632679,
                                "scale": [0.5, 0.5],
                                "texCoord": 1
                            }
                        }
                    }
                }
            }]
        }"#;
        let gltf = gltf::Gltf::from_slice(source).unwrap();
        let material = gltf.document.materials().next().unwrap();
        let info = material
            .pbr_metallic_roughness()
            .base_color_texture()
            .unwrap();
        let (tex_coord, transform) = texture_info_uv_transform(&info);
        assert_eq!(tex_coord, 1);
        assert!((transform.apply(glam::Vec2::X) - glam::Vec2::new(0.0, 1.5)).length() < 1e-6);
    }

    #[test]
    fn material_alpha_modes_produce_gltf_coverage() {
        let mut material = MaterialInfo {
            metallic: 0.0,
            roughness: 1.0,
            base_color: glam::Vec4::ONE,
            base_color_texture: None,
            tex_coord: 0,
            uv_transform: UvTransform::identity(),
            alpha_mode: gltf::material::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
        };
        assert_eq!(material_alpha_coverage(&material, 0.0), 1.0);

        material.alpha_mode = gltf::material::AlphaMode::Mask;
        assert_eq!(material_alpha_coverage(&material, 0.49), 0.0);
        assert_eq!(material_alpha_coverage(&material, 0.5), 1.0);

        material.alpha_mode = gltf::material::AlphaMode::Blend;
        assert_eq!(material_alpha_coverage(&material, 0.25), 0.25);
    }

    #[test]
    fn vertex_color_interpolates_and_multiplies_base_color() {
        let material = MaterialInfo {
            metallic: 0.0,
            roughness: 1.0,
            base_color: glam::Vec4::new(0.5, 1.0, 0.25, 0.8),
            base_color_texture: None,
            tex_coord: 0,
            uv_transform: UvTransform::identity(),
            alpha_mode: gltf::material::AlphaMode::Blend,
            alpha_cutoff: 0.5,
        };
        let triangle = Triangle {
            v0: glam::Vec3::ZERO,
            v1: glam::Vec3::X,
            v2: glam::Vec3::Y,
            uv0: glam::Vec2::ZERO,
            uv1: glam::Vec2::X,
            uv2: glam::Vec2::Y,
            color0: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            color1: glam::Vec4::new(0.0, 1.0, 0.0, 0.5),
            color2: glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
            normal: glam::Vec3::Z,
            material: 0,
            area: 0.5,
        };
        assert_eq!(
            sample_triangle_color(&triangle, &material, 0.5, 0.25, 0.25),
            glam::Vec4::new(0.25, 0.25, 0.0625, 0.5)
        );
    }

    #[test]
    fn decoded_gltf_image_formats_preserve_color_and_alpha() {
        let luma_alpha = gltf::image::Data {
            pixels: vec![64, 128],
            format: gltf::image::Format::R8G8,
            width: 1,
            height: 1,
        };
        assert_eq!(rgba8_from_gltf_image(&luma_alpha), [64, 64, 64, 128]);

        let mut unorm_pixels = Vec::new();
        for value in [0_u16, 32_768, 65_535, 16_384] {
            unorm_pixels.extend_from_slice(&value.to_ne_bytes());
        }
        let unorm = gltf::image::Data {
            pixels: unorm_pixels,
            format: gltf::image::Format::R16G16B16A16,
            width: 1,
            height: 1,
        };
        assert_eq!(rgba8_from_gltf_image(&unorm), [0, 128, 255, 64]);

        let mut float_pixels = Vec::new();
        for value in [-1.0_f32, 0.5, 2.0, 0.25] {
            float_pixels.extend_from_slice(&value.to_ne_bytes());
        }
        let float = gltf::image::Data {
            pixels: float_pixels,
            format: gltf::image::Format::R32G32B32A32FLOAT,
            width: 1,
            height: 1,
        };
        assert_eq!(rgba8_from_gltf_image(&float), [0, 128, 255, 64]);
    }

    #[test]
    fn unindexed_gltf_primitive_without_material_uses_white_default() {
        let stem = format!("blade_volume_unindexed_{}", std::process::id());
        let directory = std::env::temp_dir();
        let bin_path = directory.join(format!("{stem}.bin"));
        let gltf_path = directory.join(format!("{stem}.gltf"));
        let mut positions = Vec::new();
        for value in [
            0.0_f32, 0.0, 0.0, // v0
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
        ] {
            positions.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, positions).unwrap();
        let document = format!(
            r#"{{
                "asset": {{"version": "2.0"}},
                "buffers": [{{"uri": "{stem}.bin", "byteLength": 36}}],
                "bufferViews": [{{"buffer": 0, "byteOffset": 0, "byteLength": 36}}],
                "accessors": [{{
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "min": [0, 0, 0],
                    "max": [1, 1, 0]
                }}],
                "materials": [{{
                    "pbrMetallicRoughness": {{"baseColorFactor": [1, 0, 0, 1]}}
                }}],
                "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
                "nodes": [{{"mesh": 0}}],
                "scenes": [{{"nodes": [0]}}],
                "scene": 0
            }}"#,
        );
        std::fs::write(&gltf_path, document).unwrap();

        let model = convert_gltf(
            &gltf_path,
            &ConvertOptions {
                density: 1.0,
                interior_density_scale: 0.0,
                ambient: glam::Vec3::ONE,
                ..ConvertOptions::default()
            },
        )
        .unwrap();

        assert!(!model.is_empty());
        let white_dc = 0.5 / SH_C0;
        assert!(model
            .sh_coefficients
            .iter()
            .all(|&coefficient| (coefficient - white_dc).abs() < 1e-6));

        let error = match convert_gltf(
            &gltf_path,
            &ConvertOptions {
                output: OutputKind::RadFoam,
                density: 1.0,
                interior_density_scale: 0.0,
                ..ConvertOptions::default()
            },
        ) {
            Ok(_) => panic!("undersized RadFoam conversion unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ConvertError::Adjacency(vol::AdjacencyError::TooFewPoints { count: 1 })
        ));
        std::fs::remove_file(gltf_path).unwrap();
        std::fs::remove_file(bin_path).unwrap();
    }

    #[test]
    fn gltf_color_accessor_reaches_point_appearance() {
        let stem = format!("blade_volume_vertex_color_{}", std::process::id());
        let directory = std::env::temp_dir();
        let bin_path = directory.join(format!("{stem}.bin"));
        let gltf_path = directory.join(format!("{stem}.gltf"));
        let mut attributes = Vec::new();
        for value in [
            0.0_f32, 0.0, 0.0, // v0
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
        ] {
            attributes.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..3 {
            for value in [0.25_f32, 0.25, 0.25, 0.5] {
                attributes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(&bin_path, attributes).unwrap();
        let document = format!(
            r#"{{
                "asset": {{"version": "2.0"}},
                "buffers": [{{"uri": "{stem}.bin", "byteLength": 84}}],
                "bufferViews": [
                    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                    {{"buffer": 0, "byteOffset": 36, "byteLength": 48}}
                ],
                "accessors": [
                    {{
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 3,
                        "type": "VEC3",
                        "min": [0, 0, 0],
                        "max": [1, 1, 0]
                    }},
                    {{
                        "bufferView": 1,
                        "componentType": 5126,
                        "count": 3,
                        "type": "VEC4"
                    }}
                ],
                "materials": [{{
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {{
                        "baseColorFactor": [1, 1, 1, 0.5]
                    }}
                }}],
                "meshes": [{{"primitives": [{{
                    "attributes": {{"POSITION": 0, "COLOR_0": 1}},
                    "material": 0
                }}]}}],
                "nodes": [{{"mesh": 0}}],
                "scenes": [{{"nodes": [0]}}],
                "scene": 0
            }}"#,
        );
        std::fs::write(&gltf_path, document).unwrap();

        let model = convert_gltf(
            &gltf_path,
            &ConvertOptions {
                density: 1.0,
                interior_density_scale: 0.0,
                ambient: glam::Vec3::ONE,
                ..ConvertOptions::default()
            },
        )
        .unwrap();

        assert_eq!(model.len(), 1);
        let expected = (linear_to_srgb(0.25) - 0.5) / SH_C0;
        assert!(model
            .sh_coefficients
            .iter()
            .all(|&coefficient| (coefficient - expected).abs() < 1e-6));
        assert_eq!(model.points[0].w, 0.25);
        std::fs::remove_file(gltf_path).unwrap();
        std::fs::remove_file(bin_path).unwrap();
    }

    #[test]
    fn gltf_conversion_uses_only_the_default_scene() {
        let stem = format!("blade_volume_default_scene_{}", std::process::id());
        let directory = std::env::temp_dir();
        let bin_path = directory.join(format!("{stem}.bin"));
        let gltf_path = directory.join(format!("{stem}.gltf"));
        let mut positions = Vec::new();
        for value in [
            0.0_f32, 0.0, 0.0, // v0
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
        ] {
            positions.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, positions).unwrap();
        let document = format!(
            r#"{{
                "asset": {{"version": "2.0"}},
                "buffers": [{{"uri": "{stem}.bin", "byteLength": 36}}],
                "bufferViews": [{{"buffer": 0, "byteOffset": 0, "byteLength": 36}}],
                "accessors": [{{
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "min": [0, 0, 0],
                    "max": [1, 1, 0]
                }}],
                "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
                "nodes": [
                    {{"mesh": 0}},
                    {{"mesh": 0, "translation": [10, 0, 0]}}
                ],
                "scenes": [{{"nodes": [0]}}, {{"nodes": [1]}}],
                "scene": 1
            }}"#,
        );
        std::fs::write(&gltf_path, document).unwrap();

        let model = convert_gltf(
            &gltf_path,
            &ConvertOptions {
                density: 1.0,
                interior_density_scale: 0.0,
                ambient: glam::Vec3::ONE,
                ..ConvertOptions::default()
            },
        )
        .unwrap();

        assert_eq!(model.len(), 1);
        assert!(model.points[0].x >= 10.0);
        std::fs::remove_file(gltf_path).unwrap();
        std::fs::remove_file(bin_path).unwrap();
    }

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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
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
    fn gaussian_ply_roundtrip_preserves_sh3_layout() {
        let points = vec![glam::Vec4::new(0.25, -0.5, 1.0, 0.8)];
        let sh_coefficients: Vec<f32> = (0..48).map(|i| i as f32 * 0.125 - 2.0).collect();
        let transforms = vol::Transforms {
            rotations: vec![glam::Quat::from_rotation_x(0.3)],
            scales: vec![glam::Vec3::new(0.1, 0.2, 0.3)],
        };
        let model = vol::PointCloudModel {
            points,
            sh_coefficients: sh_coefficients.clone(),
            sh_degree: 3,
            transforms: Some(transforms),
            adjacency: None,
            radii: None,
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
        };

        let path = std::env::temp_dir().join("blade_volume_convert_roundtrip_sh3.ply");
        save_ply(&path, &model).expect("save ply");
        let loaded = vol::io::load_gaussian(path.to_str().unwrap());

        assert_eq!(loaded.sh_degree, 3);
        assert_eq!(loaded.sh_coefficients, sh_coefficients);
        assert!((loaded.points[0].w - model.points[0].w).abs() < 1e-6);
        std::fs::remove_file(path).unwrap();
    }

    fn assert_radfoam_sh3_roundtrip(format: PlyFormat) {
        // Two points with SH degree 3 (16 components × 3 channels per cell).
        // Exact DC uses the blade_sh_dc_* extension while the higher-order
        // coefficients use the upstream-compatible color_sh_* properties.
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
        };

        let suffix = match format {
            PlyFormat::Ascii => "ascii",
            PlyFormat::Binary => "binary",
        };
        let path = std::env::temp_dir().join(format!(
            "blade_volume_convert_roundtrip_radfoam_sh3_{suffix}.ply"
        ));
        save_ply_with_options(&path, &model, &SaveOptions { format }).expect("save ply");
        let loaded = vol::io::load_radfoam(path.to_str().unwrap());

        assert_eq!(loaded.sh_degree, sh_degree, "sh_degree must round-trip");
        assert_eq!(loaded.sh_coefficients, sh_coefficients);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn radfoam_binary_roundtrip_preserves_exact_sh3() {
        assert_radfoam_sh3_roundtrip(PlyFormat::Binary);
    }

    #[test]
    fn radfoam_ascii_roundtrip_preserves_exact_sh3() {
        assert_radfoam_sh3_roundtrip(PlyFormat::Ascii);
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
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
            surface_normals: None,
            surface_offsets: None,
            surface_detail: None,
            surface_color_coefficients: None,
            spherical_voronoi: None,
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

    fn assert_surface_planes_roundtrip(format: PlyFormat) {
        let mut model = make_radfoam_radii_model();
        model.surface_normals = Some(vec![
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(0.0, -1.0, 0.0),
            glam::Vec3::new(0.2, 0.3, -0.9),
        ]);
        model.surface_offsets = Some(vec![-0.01, 0.0, 0.025]);
        model.surface_detail = Some(vol::SurfaceDetail {
            offsets: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
                .map(|index| {
                    glam::Vec3::new(
                        index as f32 * 0.01 - 0.2,
                        index as f32 * -0.02 + 0.1,
                        index as f32 * 0.005,
                    )
                })
                .collect(),
            heights: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
                .map(|index| index as f32 * 0.002 - 0.01)
                .collect(),
            colors: (0..model.points.len() * vol::SURFACE_DETAIL_SITES)
                .map(|index| {
                    glam::Vec3::new(
                        index as f32 * 0.003,
                        index as f32 * -0.004,
                        index as f32 * 0.002 - 0.05,
                    )
                })
                .collect(),
        });
        model.surface_color_coefficients = Some(
            (0..model.points.len() * vol::SURFACE_COLOR_COMPONENTS * 3)
                .map(|index| index as f32 * 0.01 - 0.2)
                .collect(),
        );
        model.spherical_voronoi = Some(vol::SphericalVoronoi {
            axes: (0..model.points.len() * vol::SPHERICAL_VORONOI_SITES)
                .map(|index| {
                    glam::Vec3::new(
                        index as f32 * 0.01 - 0.4,
                        index as f32 * -0.02 + 0.3,
                        index as f32 * 0.03 - 0.2,
                    )
                })
                .collect(),
            colors: (0..model.points.len() * vol::SPHERICAL_VORONOI_SITES)
                .map(|index| {
                    glam::Vec3::new(
                        index as f32 * -0.01,
                        index as f32 * 0.005,
                        index as f32 * 0.002 - 0.1,
                    )
                })
                .collect(),
        });
        let suffix = match format {
            PlyFormat::Ascii => "ascii",
            PlyFormat::Binary => "binary",
        };
        let path = std::env::temp_dir().join(format!(
            "blade_volume_convert_roundtrip_surface_planes_{suffix}.ply"
        ));
        save_ply_with_options(&path, &model, &SaveOptions { format }).expect("save ply");
        let loaded = vol::io::load_radfoam(path.to_str().unwrap());
        assert_eq!(loaded.radii, model.radii);
        assert_eq!(loaded.surface_normals, model.surface_normals);
        assert_eq!(loaded.surface_offsets, model.surface_offsets);
        assert_eq!(loaded.surface_detail, model.surface_detail);
        assert_eq!(
            loaded.surface_color_coefficients,
            model.surface_color_coefficients
        );
        assert_eq!(loaded.spherical_voronoi, model.spherical_voronoi);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn radfoam_binary_roundtrip_preserves_surface_planes() {
        assert_surface_planes_roundtrip(PlyFormat::Binary);
    }

    #[test]
    fn radfoam_ascii_roundtrip_preserves_surface_planes() {
        assert_surface_planes_roundtrip(PlyFormat::Ascii);
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
            color0: glam::Vec4::ONE,
            color1: glam::Vec4::ONE,
            color2: glam::Vec4::ONE,
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
                color0: glam::Vec4::ONE,
                color1: glam::Vec4::ONE,
                color2: glam::Vec4::ONE,
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
                color0: glam::Vec4::ONE,
                color1: glam::Vec4::ONE,
                color2: glam::Vec4::ONE,
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
                color0: glam::Vec4::ONE,
                color1: glam::Vec4::ONE,
                color2: glam::Vec4::ONE,
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
                color0: glam::Vec4::ONE,
                color1: glam::Vec4::ONE,
                color2: glam::Vec4::ONE,
                normal: glam::Vec3::new(1.0, 1.0, 1.0).normalize(),
                material: 0,
                area: 0.5,
            },
        ];

        let tester = super::InsideTester::new(&triangles);
        let inside = glam::Vec3::new(0.1, 0.1, 0.1);
        let outside = glam::Vec3::new(1.5, 1.5, 1.5);
        assert!(tester.is_inside(inside, &triangles));
        assert!(!tester.is_inside(outside, &triangles));
    }

    /// Exhaustive reference: the pre-index behaviour, kept only as a test
    /// oracle. `InsideTester` must agree with it everywhere.
    fn is_point_inside_mesh_exhaustive(point: glam::Vec3, triangles: &[super::Triangle]) -> bool {
        let mut odd_hits = 0u32;
        for ray_dir in super::INSIDE_RAYS.iter() {
            let dir = ray_dir.normalize();
            let origin = point + dir * 1e-4;
            let mut hits = 0u32;
            for tri in triangles.iter() {
                if super::ray_intersects_triangle(origin, dir, tri) {
                    hits += 1;
                }
            }
            if hits % 2 == 1 {
                odd_hits += 1;
            }
        }
        odd_hits >= 2
    }

    fn unit_triangle(v0: glam::Vec3, v1: glam::Vec3, v2: glam::Vec3) -> super::Triangle {
        super::Triangle {
            v0,
            v1,
            v2,
            uv0: glam::Vec2::ZERO,
            uv1: glam::Vec2::ZERO,
            uv2: glam::Vec2::ZERO,
            color0: glam::Vec4::ONE,
            color1: glam::Vec4::ONE,
            color2: glam::Vec4::ONE,
            normal: (v1 - v0).cross(v2 - v0).normalize_or_zero(),
            material: 0,
            area: 0.5 * (v1 - v0).cross(v2 - v0).length(),
        }
    }

    /// Two disjoint axis-aligned boxes. The gap between them, the interiors,
    /// and the surrounding space all have to classify identically with and
    /// without the projected index — including points outside its extent.
    fn two_box_mesh() -> Vec<super::Triangle> {
        let mut triangles = Vec::new();
        for offset in [glam::Vec3::ZERO, glam::Vec3::new(3.0, 0.0, 0.0)] {
            let lo = offset;
            let hi = offset + glam::Vec3::ONE;
            let corner = |x: bool, y: bool, z: bool| {
                glam::Vec3::new(
                    if x { hi.x } else { lo.x },
                    if y { hi.y } else { lo.y },
                    if z { hi.z } else { lo.z },
                )
            };
            // Six quads, two triangles each, consistent winding not required
            // by an even-odd test.
            let quads = [
                [
                    corner(false, false, false),
                    corner(true, false, false),
                    corner(true, true, false),
                    corner(false, true, false),
                ],
                [
                    corner(false, false, true),
                    corner(true, false, true),
                    corner(true, true, true),
                    corner(false, true, true),
                ],
                [
                    corner(false, false, false),
                    corner(false, true, false),
                    corner(false, true, true),
                    corner(false, false, true),
                ],
                [
                    corner(true, false, false),
                    corner(true, true, false),
                    corner(true, true, true),
                    corner(true, false, true),
                ],
                [
                    corner(false, false, false),
                    corner(true, false, false),
                    corner(true, false, true),
                    corner(false, false, true),
                ],
                [
                    corner(false, true, false),
                    corner(true, true, false),
                    corner(true, true, true),
                    corner(false, true, true),
                ],
            ];
            for quad in quads.iter() {
                triangles.push(unit_triangle(quad[0], quad[1], quad[2]));
                triangles.push(unit_triangle(quad[0], quad[2], quad[3]));
            }
        }
        triangles
    }

    #[test]
    fn projected_index_matches_an_exhaustive_scan() {
        let triangles = two_box_mesh();
        let tester = super::InsideTester::new(&triangles);

        let mut inside_count = 0u32;
        let mut outside_count = 0u32;
        // Sweep well beyond the mesh so query points land outside the index
        // extent too, where the candidate list must be empty rather than wrong.
        let steps = 24;
        for iz in 0..steps {
            for iy in 0..steps {
                for ix in 0..steps {
                    let t = |i: i32| -1.0 + 6.0 * (i as f32 + 0.5) / steps as f32;
                    let p = glam::Vec3::new(t(ix), t(iy), t(iz));
                    let fast = tester.is_inside(p, &triangles);
                    let slow = is_point_inside_mesh_exhaustive(p, &triangles);
                    assert_eq!(fast, slow, "disagreement at {p:?}");
                    if fast {
                        inside_count += 1;
                    } else {
                        outside_count += 1;
                    }
                }
            }
        }
        // Guard against a vacuous pass where nothing was ever classified inside.
        assert!(inside_count > 0, "no interior samples were exercised");
        assert!(outside_count > 0, "no exterior samples were exercised");
    }

    #[test]
    fn interior_fill_is_scale_invariant_under_resolution() {
        // The same asset at two scales must produce the same cloud size when
        // the sampling rate is requested as a resolution rather than a density.
        let base = two_box_mesh();
        let scaled = base
            .iter()
            .map(|tri| unit_triangle(tri.v0 * 100.0, tri.v1 * 100.0, tri.v2 * 100.0))
            .collect::<Vec<_>>();

        let count_inside = |triangles: &[super::Triangle], resolution: f32| {
            let bounds = super::compute_bounds(triangles);
            let diagonal = (bounds.max - bounds.min).length();
            let density = (resolution / diagonal).powi(3) * 0.25;
            let spacing = density.powf(-1.0 / 3.0);
            let tester = super::InsideTester::new(triangles);
            let mut count = 0u32;
            let start = bounds.min + glam::Vec3::splat(0.5 * spacing);
            let mut z = start.z;
            while z <= bounds.max.z {
                let mut y = start.y;
                while y <= bounds.max.y {
                    let mut x = start.x;
                    while x <= bounds.max.x {
                        if tester.is_inside(glam::Vec3::new(x, y, z), triangles) {
                            count += 1;
                        }
                        x += spacing;
                    }
                    y += spacing;
                }
                z += spacing;
            }
            count
        };

        let small = count_inside(&base, 24.0);
        let large = count_inside(&scaled, 24.0);
        assert!(small > 0, "expected a non-empty interior");
        assert_eq!(small, large, "resolution sampling must not depend on units");
    }
}
