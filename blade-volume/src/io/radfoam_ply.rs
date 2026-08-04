//! Loader for RadFoam and weighted PowerFoam PLY point clouds.
//!
//! The vertex element carries position, density, CSR end offsets, SH data, and
//! optional radius, oriented-surface normal, signed surface offset, and
//! spatial surface-color coefficients. The adjacency element carries the flattened CSR neighbours. Binary
//! little-endian and ASCII PLY 1.0 are supported.

use super::LoadError;

use std::{fs, io};

const MAX_HEADER_BYTES: usize = 1024 * 1024;
const MAX_ASCII_ROW_BYTES: usize = 1024 * 1024;
const SH_DC_FACTOR: f32 = 0.282_094_8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    BinaryLittleEndian,
    Ascii,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyScalarType {
    Float32,
    Uint32,
    Uint8,
}

impl PlyScalarType {
    fn parse(word: &str) -> Result<Self, LoadError> {
        match word {
            "float" | "float32" => Ok(Self::Float32),
            "uint" | "uint32" => Ok(Self::Uint32),
            "uchar" | "uint8" => Ok(Self::Uint8),
            other => Err(LoadError::invalid(format!(
                "unsupported RadFoam PLY scalar type '{other}'"
            ))),
        }
    }

    fn size_bytes(self) -> usize {
        match self {
            Self::Float32 | Self::Uint32 => 4,
            Self::Uint8 => 1,
        }
    }
}

#[derive(Debug)]
struct Property {
    name: String,
    ty: PlyScalarType,
    offset: usize,
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    stride: usize,
    properties: Vec<Property>,
}

impl Element {
    fn property(&self, name: &str) -> Option<&Property> {
        self.properties
            .iter()
            .find(|property| property.name == name)
    }

    fn require(&self, name: &str, ty: PlyScalarType) -> Result<&Property, LoadError> {
        let property = self.property(name).ok_or_else(|| {
            LoadError::invalid(format!(
                "RadFoam PLY element '{}' is missing property '{name}'",
                self.name
            ))
        })?;
        if property.ty != ty {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY property '{}.{name}' has type {:?}, expected {:?}",
                self.name, property.ty, ty
            )));
        }
        Ok(property)
    }
}

#[derive(Debug)]
struct Header {
    format: PlyFormat,
    elements: Vec<Element>,
}

struct Schema {
    point_count: usize,
    adjacency_count: usize,
    x: usize,
    y: usize,
    z: usize,
    density: usize,
    adjacency_offset: usize,
    exact_dc: Option<[usize; 3]>,
    preview_dc: Option<[usize; 3]>,
    sh_rest: Vec<usize>,
    sh_degree: usize,
    sh_stride: usize,
    radius: Option<usize>,
    surface_normal: Option<[usize; 3]>,
    surface_offset: Option<usize>,
    surface_detail_offsets: Vec<usize>,
    surface_detail_heights: Vec<usize>,
    surface_detail_colors: Vec<usize>,
    surface_color: Vec<usize>,
    spherical_voronoi_axes: Vec<usize>,
    spherical_voronoi_colors: Vec<usize>,
    adjacency: usize,
}

impl Schema {
    fn new(header: &Header) -> Result<Self, LoadError> {
        if header.elements.len() != 2
            || header.elements[0].name != "vertex"
            || header.elements[1].name != "adjacency"
        {
            return Err(LoadError::invalid(
                "RadFoam PLY elements must be exactly 'vertex' followed by 'adjacency'",
            ));
        }
        let vertex = &header.elements[0];
        let adjacency = &header.elements[1];
        if vertex.count > u32::MAX as usize || adjacency.count > u32::MAX as usize {
            return Err(LoadError::invalid(
                "RadFoam PLY point and adjacency counts must fit in uint32",
            ));
        }

        let exact_dc = property_triplet(
            vertex,
            ["blade_sh_dc_0", "blade_sh_dc_1", "blade_sh_dc_2"],
            PlyScalarType::Float32,
        )?;
        let preview_dc = property_triplet(vertex, ["red", "green", "blue"], PlyScalarType::Uint8)?;
        if exact_dc.is_none() && preview_dc.is_none() {
            return Err(LoadError::invalid(
                "RadFoam PLY vertex needs exact blade_sh_dc_0..2 or red/green/blue preview DC",
            ));
        }

        let mut sh_rest = Vec::new();
        for property in &vertex.properties {
            if let Some(suffix) = property.name.strip_prefix("color_sh_") {
                if property.ty != PlyScalarType::Float32 {
                    return Err(LoadError::invalid(format!(
                        "RadFoam PLY property '{}' must be float32",
                        property.name
                    )));
                }
                let index = suffix.parse::<usize>().map_err(|_| {
                    LoadError::invalid(format!("invalid RadFoam PLY property '{}'", property.name))
                })?;
                sh_rest.push((index, property.offset));
            }
        }
        sh_rest.sort_unstable_by_key(|entry| entry.0);
        for (expected, &(actual, _)) in sh_rest.iter().enumerate() {
            if actual != expected {
                return Err(LoadError::invalid(format!(
                    "RadFoam PLY color_sh indices must be contiguous; expected {expected}, got {actual}"
                )));
            }
        }
        let sh_degree = infer_sh_degree(sh_rest.len())?;
        let sh_stride = crate::get_sh_component_count(sh_degree)
            .checked_mul(3)
            .ok_or_else(|| LoadError::invalid("RadFoam PLY SH stride overflow"))?;

        let radius = match vertex.property("radius") {
            Some(property) if property.ty == PlyScalarType::Float32 => Some(property.offset),
            Some(_) => {
                return Err(LoadError::invalid(
                    "RadFoam PLY vertex radius must be float32",
                ));
            }
            None => None,
        };
        let surface_normal = property_triplet(vertex, ["nx", "ny", "nz"], PlyScalarType::Float32)?;
        if surface_normal.is_some() && radius.is_none() {
            return Err(LoadError::invalid(
                "RadFoam PLY surface normals require a radius property",
            ));
        }
        let surface_offset = match vertex.property("surface_offset") {
            Some(property) if property.ty == PlyScalarType::Float32 => Some(property.offset),
            Some(_) => {
                return Err(LoadError::invalid(
                    "RadFoam PLY vertex surface_offset must be float32",
                ));
            }
            None => None,
        };
        if surface_offset.is_some() && surface_normal.is_none() {
            return Err(LoadError::invalid(
                "RadFoam PLY surface offsets require nx, ny, and nz properties",
            ));
        }

        let surface_detail_offsets = indexed_float_properties(
            vertex,
            "blade_surface_detail_offset_",
            "surface-detail offset",
        )?;
        let surface_detail_heights = indexed_float_properties(
            vertex,
            "blade_surface_detail_height_",
            "surface-detail height",
        )?;
        let surface_detail_colors = indexed_float_properties(
            vertex,
            "blade_surface_detail_color_",
            "surface-detail color",
        )?;
        let expected_detail_offsets = crate::SURFACE_DETAIL_SITES * 3;
        let expected_detail_heights = crate::SURFACE_DETAIL_SITES;
        let expected_detail_colors = crate::SURFACE_DETAIL_SITES * 3;
        let has_surface_detail = !surface_detail_offsets.is_empty()
            || !surface_detail_heights.is_empty()
            || !surface_detail_colors.is_empty();
        if has_surface_detail
            && (surface_detail_offsets.len() != expected_detail_offsets
                || surface_detail_heights.len() != expected_detail_heights
                || surface_detail_colors.len() != expected_detail_colors)
        {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY surface-detail property counts are offsets={} heights={} colors={}, expected {expected_detail_offsets}, {expected_detail_heights}, and {expected_detail_colors}",
                surface_detail_offsets.len(),
                surface_detail_heights.len(),
                surface_detail_colors.len(),
            )));
        }
        if has_surface_detail && surface_normal.is_none() {
            return Err(LoadError::invalid(
                "RadFoam PLY surface detail requires radius, nx, ny, and nz properties",
            ));
        }

        let mut surface_color = Vec::new();
        for property in &vertex.properties {
            if let Some(suffix) = property.name.strip_prefix("blade_surface_color_") {
                if property.ty != PlyScalarType::Float32 {
                    return Err(LoadError::invalid(format!(
                        "RadFoam PLY property '{}' must be float32",
                        property.name
                    )));
                }
                let index = suffix.parse::<usize>().map_err(|_| {
                    LoadError::invalid(format!("invalid RadFoam PLY property '{}'", property.name))
                })?;
                surface_color.push((index, property.offset));
            }
        }
        surface_color.sort_unstable_by_key(|entry| entry.0);
        for (expected, &(actual, _)) in surface_color.iter().enumerate() {
            if actual != expected {
                return Err(LoadError::invalid(format!(
                    "RadFoam PLY surface-color indices must be contiguous; expected {expected}, got {actual}"
                )));
            }
        }
        let expected_surface_color = crate::SURFACE_COLOR_COMPONENTS * 3;
        if !surface_color.is_empty() && surface_color.len() != expected_surface_color {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY surface-color property count is {}, expected {expected_surface_color}",
                surface_color.len()
            )));
        }
        if !surface_color.is_empty() && surface_normal.is_none() {
            return Err(LoadError::invalid(
                "RadFoam PLY surface color requires radius, nx, ny, and nz properties",
            ));
        }

        let spherical_voronoi_axes = indexed_float_properties(
            vertex,
            "blade_spherical_voronoi_axis_",
            "Spherical Voronoi axis",
        )?;
        let spherical_voronoi_colors = indexed_float_properties(
            vertex,
            "blade_spherical_voronoi_color_",
            "Spherical Voronoi color",
        )?;
        let expected_spherical = crate::SPHERICAL_VORONOI_SITES * 3;
        if spherical_voronoi_axes.len() != spherical_voronoi_colors.len()
            || (!spherical_voronoi_axes.is_empty()
                && spherical_voronoi_axes.len() != expected_spherical)
        {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY Spherical Voronoi property counts are axes={} colors={}, expected either zero or {expected_spherical} each",
                spherical_voronoi_axes.len(),
                spherical_voronoi_colors.len(),
            )));
        }
        if !spherical_voronoi_axes.is_empty() && surface_normal.is_none() {
            return Err(LoadError::invalid(
                "RadFoam PLY Spherical Voronoi appearance requires radius, nx, ny, and nz properties",
            ));
        }

        Ok(Self {
            point_count: vertex.count,
            adjacency_count: adjacency.count,
            x: vertex.require("x", PlyScalarType::Float32)?.offset,
            y: vertex.require("y", PlyScalarType::Float32)?.offset,
            z: vertex.require("z", PlyScalarType::Float32)?.offset,
            density: vertex.require("density", PlyScalarType::Float32)?.offset,
            adjacency_offset: vertex
                .require("adjacency_offset", PlyScalarType::Uint32)?
                .offset,
            exact_dc,
            preview_dc,
            sh_rest: sh_rest.into_iter().map(|entry| entry.1).collect(),
            sh_degree,
            sh_stride,
            radius,
            surface_normal,
            surface_offset,
            surface_detail_offsets,
            surface_detail_heights,
            surface_detail_colors,
            surface_color: surface_color.into_iter().map(|entry| entry.1).collect(),
            spherical_voronoi_axes,
            spherical_voronoi_colors,
            adjacency: adjacency
                .require("adjacency", PlyScalarType::Uint32)?
                .offset,
        })
    }
}

fn indexed_float_properties(
    element: &Element,
    prefix: &str,
    label: &str,
) -> Result<Vec<usize>, LoadError> {
    let mut properties = Vec::new();
    for property in &element.properties {
        if let Some(suffix) = property.name.strip_prefix(prefix) {
            if property.ty != PlyScalarType::Float32 {
                return Err(LoadError::invalid(format!(
                    "RadFoam PLY property '{}' must be float32",
                    property.name
                )));
            }
            let index = suffix.parse::<usize>().map_err(|_| {
                LoadError::invalid(format!("invalid RadFoam PLY property '{}'", property.name))
            })?;
            properties.push((index, property.offset));
        }
    }
    properties.sort_unstable_by_key(|entry| entry.0);
    for (expected, &(actual, _)) in properties.iter().enumerate() {
        if actual != expected {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY {label} indices must be contiguous; expected {expected}, got {actual}"
            )));
        }
    }
    Ok(properties.into_iter().map(|entry| entry.1).collect())
}

fn property_triplet(
    element: &Element,
    names: [&str; 3],
    ty: PlyScalarType,
) -> Result<Option<[usize; 3]>, LoadError> {
    let properties = [
        element.property(names[0]),
        element.property(names[1]),
        element.property(names[2]),
    ];
    match properties {
        [None, None, None] => Ok(None),
        [Some(first), Some(second), Some(third)] => {
            for property in [first, second, third] {
                if property.ty != ty {
                    return Err(LoadError::invalid(format!(
                        "RadFoam PLY property '{}' has type {:?}, expected {:?}",
                        property.name, property.ty, ty
                    )));
                }
            }
            Ok(Some([first.offset, second.offset, third.offset]))
        }
        _ => Err(LoadError::invalid(format!(
            "RadFoam PLY vertex must contain either all or none of {}, {}, and {}",
            names[0], names[1], names[2]
        ))),
    }
}

fn infer_sh_degree(sh_rest_count: usize) -> Result<usize, LoadError> {
    if !sh_rest_count.is_multiple_of(3) {
        return Err(LoadError::invalid(format!(
            "RadFoam PLY color_sh property count {sh_rest_count} is not divisible by three"
        )));
    }
    let component_count = sh_rest_count / 3 + 1;
    (0..=crate::MAX_SH_DEGREE)
        .find(|&degree| crate::get_sh_component_count(degree) == component_count)
        .ok_or_else(|| {
            LoadError::invalid(format!(
                "RadFoam PLY has {component_count} SH components, expected a complete degree up to {}",
                crate::MAX_SH_DEGREE
            ))
        })
}

fn read_header(file: &mut io::BufReader<fs::File>) -> Result<Header, LoadError> {
    let mut line = String::new();
    let mut header_bytes = 0usize;
    let mut format = None;
    let mut elements = Vec::new();
    let mut current_element: Option<Element> = None;
    let mut saw_magic = false;
    let mut saw_end = false;

    loop {
        line.clear();
        let remaining = MAX_HEADER_BYTES.saturating_sub(header_bytes);
        let mut limited = io::Read::take(&mut *file, (remaining + 1) as u64);
        let bytes = io::BufRead::read_line(&mut limited, &mut line)?;
        if bytes == 0 {
            break;
        }
        header_bytes += bytes;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY header exceeds {MAX_HEADER_BYTES} bytes"
            )));
        }
        let mut words = line.split_whitespace();
        let Some(head) = words.next() else {
            continue;
        };
        match head {
            "ply" if !saw_magic && header_bytes == bytes => saw_magic = true,
            "ply" => return Err(LoadError::invalid("unexpected duplicate PLY magic")),
            "format" => {
                if format.is_some() {
                    return Err(LoadError::invalid("duplicate RadFoam PLY format line"));
                }
                let name = words.next().unwrap_or("");
                let version = words.next().unwrap_or("");
                if version != "1.0" {
                    return Err(LoadError::invalid(format!(
                        "unsupported RadFoam PLY version '{version}'"
                    )));
                }
                format = Some(match name {
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "ascii" => PlyFormat::Ascii,
                    other => {
                        return Err(LoadError::invalid(format!(
                            "unsupported RadFoam PLY format '{other}'"
                        )));
                    }
                });
            }
            "comment" | "obj_info" => {}
            "element" => {
                if let Some(element) = current_element.take() {
                    elements.push(element);
                }
                let name = words.next().unwrap_or("");
                if name.is_empty() {
                    return Err(LoadError::invalid("RadFoam PLY element has no name"));
                }
                if elements
                    .iter()
                    .any(|element: &Element| element.name == name)
                {
                    return Err(LoadError::invalid(format!(
                        "duplicate RadFoam PLY element '{name}'"
                    )));
                }
                let count = words
                    .next()
                    .ok_or_else(|| LoadError::invalid(format!("element '{name}' has no count")))?
                    .parse::<usize>()
                    .map_err(|_| {
                        LoadError::invalid(format!("invalid count for element '{name}'"))
                    })?;
                current_element = Some(Element {
                    name: name.to_string(),
                    count,
                    stride: 0,
                    properties: Vec::new(),
                });
            }
            "property" => {
                let element = current_element.as_mut().ok_or_else(|| {
                    LoadError::invalid("RadFoam PLY property appears before an element")
                })?;
                let ty_name = words.next().unwrap_or("");
                if ty_name == "list" {
                    return Err(LoadError::invalid(format!(
                        "RadFoam PLY list properties are unsupported in element '{}'",
                        element.name
                    )));
                }
                let ty = PlyScalarType::parse(ty_name)?;
                let name = words.next().unwrap_or("");
                if name.is_empty() {
                    return Err(LoadError::invalid(format!(
                        "RadFoam PLY property has no name in element '{}'",
                        element.name
                    )));
                }
                if element
                    .properties
                    .iter()
                    .any(|property| property.name == name)
                {
                    return Err(LoadError::invalid(format!(
                        "duplicate RadFoam PLY property '{}.{name}'",
                        element.name
                    )));
                }
                let offset = element.stride;
                element.stride = element
                    .stride
                    .checked_add(ty.size_bytes())
                    .ok_or_else(|| LoadError::invalid("RadFoam PLY element stride overflow"))?;
                element.properties.push(Property {
                    name: name.to_string(),
                    ty,
                    offset,
                });
            }
            "end_header" => {
                saw_end = true;
                break;
            }
            other => {
                return Err(LoadError::invalid(format!(
                    "unexpected RadFoam PLY header token '{other}'"
                )));
            }
        }
    }
    if let Some(element) = current_element {
        elements.push(element);
    }
    if !saw_magic {
        return Err(LoadError::invalid("RadFoam PLY is missing the 'ply' magic"));
    }
    if !saw_end {
        return Err(LoadError::invalid("RadFoam PLY header has no end_header"));
    }
    let format = format.ok_or_else(|| LoadError::invalid("RadFoam PLY has no format line"))?;
    Ok(Header { format, elements })
}

fn allocate_vec<T: Clone>(count: usize, value: T, name: &str) -> Result<Vec<T>, LoadError> {
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|error| {
        LoadError::invalid(format!("RadFoam PLY {name} allocation failed: {error}"))
    })?;
    output.resize(count, value);
    Ok(output)
}

fn allocate_model(
    schema: &Schema,
) -> Result<(crate::PointCloudModel, Vec<u32>, Vec<u32>), LoadError> {
    let sh_count = schema
        .point_count
        .checked_mul(schema.sh_stride)
        .ok_or_else(|| LoadError::invalid("RadFoam PLY SH allocation size overflow"))?;
    let offset_count = schema
        .point_count
        .checked_add(1)
        .ok_or_else(|| LoadError::invalid("RadFoam PLY CSR offset count overflow"))?;
    let radii = match schema.radius {
        Some(_) => Some(allocate_vec(schema.point_count, 0.0, "radius")?),
        None => None,
    };
    let surface_normals = match schema.surface_normal {
        Some(_) => Some(allocate_vec(
            schema.point_count,
            glam::Vec3::ZERO,
            "surface normal",
        )?),
        None => None,
    };
    let surface_offsets = match schema.surface_offset {
        Some(_) => Some(allocate_vec(schema.point_count, 0.0, "surface offset")?),
        None => None,
    };
    let surface_detail = if schema.surface_detail_offsets.is_empty() {
        None
    } else {
        let count = schema
            .point_count
            .checked_mul(crate::SURFACE_DETAIL_SITES)
            .ok_or_else(|| LoadError::invalid("RadFoam PLY surface-detail allocation overflow"))?;
        Some(crate::SurfaceDetail {
            offsets: allocate_vec(count, glam::Vec3::ZERO, "surface-detail offset")?,
            heights: allocate_vec(count, 0.0, "surface-detail height")?,
            colors: allocate_vec(count, glam::Vec3::ZERO, "surface-detail color")?,
        })
    };
    let surface_color_coefficients = if schema.surface_color.is_empty() {
        None
    } else {
        let count = schema
            .point_count
            .checked_mul(crate::SURFACE_COLOR_COMPONENTS * 3)
            .ok_or_else(|| LoadError::invalid("RadFoam PLY surface-color allocation overflow"))?;
        Some(allocate_vec(count, 0.0, "surface color")?)
    };
    let spherical_voronoi = if schema.spherical_voronoi_axes.is_empty() {
        None
    } else {
        let count = schema
            .point_count
            .checked_mul(crate::SPHERICAL_VORONOI_SITES)
            .ok_or_else(|| {
                LoadError::invalid("RadFoam PLY Spherical Voronoi allocation overflow")
            })?;
        Some(crate::SphericalVoronoi {
            axes: allocate_vec(count, glam::Vec3::ZERO, "Spherical Voronoi axes")?,
            colors: allocate_vec(count, glam::Vec3::ZERO, "Spherical Voronoi colors")?,
        })
    };
    let model = crate::PointCloudModel {
        points: allocate_vec(schema.point_count, glam::Vec4::ZERO, "point")?,
        sh_coefficients: allocate_vec(sh_count, 0.0, "SH")?,
        sh_degree: schema.sh_degree,
        transforms: None,
        adjacency: None,
        radii,
        surface_normals,
        surface_offsets,
        surface_detail,
        surface_color_coefficients,
        spherical_voronoi,
    };
    Ok((
        model,
        allocate_vec(offset_count, 0, "CSR offset")?,
        allocate_vec(schema.adjacency_count, 0, "adjacency")?,
    ))
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn set_preview_dc(coefficients: &mut [f32], preview: [u8; 3]) {
    coefficients[0] = (preview[0] as f32 / 255.0 - 0.5) / SH_DC_FACTOR;
    coefficients[1] = (preview[1] as f32 / 255.0 - 0.5) / SH_DC_FACTOR;
    coefficients[2] = (preview[2] as f32 / 255.0 - 0.5) / SH_DC_FACTOR;
}

fn decode_binary(
    file: &mut io::BufReader<fs::File>,
    header: &Header,
    schema: &Schema,
    model: &mut crate::PointCloudModel,
    offsets: &mut [u32],
    neighbors: &mut [u32],
) -> Result<(), LoadError> {
    let vertex = &header.elements[0];
    let adjacency = &header.elements[1];
    let mut row = allocate_vec(vertex.stride, 0u8, "vertex row")?;
    for index in 0..schema.point_count {
        io::Read::read_exact(file, &mut row).map_err(|error| {
            LoadError::invalid(format!("RadFoam PLY vertex body is truncated: {error}"))
        })?;
        model.points[index] = glam::Vec4::new(
            read_f32(&row, schema.x),
            read_f32(&row, schema.y),
            read_f32(&row, schema.z),
            read_f32(&row, schema.density),
        );
        offsets[index + 1] = read_u32(&row, schema.adjacency_offset);
        let coefficients =
            &mut model.sh_coefficients[index * schema.sh_stride..(index + 1) * schema.sh_stride];
        if let Some(exact) = schema.exact_dc {
            for channel in 0..3 {
                coefficients[channel] = read_f32(&row, exact[channel]);
            }
        } else if let Some(preview) = schema.preview_dc {
            set_preview_dc(
                coefficients,
                [row[preview[0]], row[preview[1]], row[preview[2]]],
            );
        }
        for (rest, &offset) in schema.sh_rest.iter().enumerate() {
            coefficients[3 + rest] = read_f32(&row, offset);
        }
        if let (Some(ref mut radii), Some(radius)) = (model.radii.as_mut(), schema.radius) {
            radii[index] = read_f32(&row, radius);
        }
        if let (Some(ref mut normals), Some(normal)) =
            (model.surface_normals.as_mut(), schema.surface_normal)
        {
            normals[index] = glam::Vec3::new(
                read_f32(&row, normal[0]),
                read_f32(&row, normal[1]),
                read_f32(&row, normal[2]),
            );
        }
        if let (Some(ref mut offsets), Some(offset)) =
            (model.surface_offsets.as_mut(), schema.surface_offset)
        {
            offsets[index] = read_f32(&row, offset);
        }
        if let Some(ref mut detail) = model.surface_detail {
            let base = index * crate::SURFACE_DETAIL_SITES;
            for site in 0..crate::SURFACE_DETAIL_SITES {
                let component = site * 3;
                detail.offsets[base + site] = glam::Vec3::new(
                    read_f32(&row, schema.surface_detail_offsets[component]),
                    read_f32(&row, schema.surface_detail_offsets[component + 1]),
                    read_f32(&row, schema.surface_detail_offsets[component + 2]),
                );
                detail.heights[base + site] = read_f32(&row, schema.surface_detail_heights[site]);
                detail.colors[base + site] = glam::Vec3::new(
                    read_f32(&row, schema.surface_detail_colors[component]),
                    read_f32(&row, schema.surface_detail_colors[component + 1]),
                    read_f32(&row, schema.surface_detail_colors[component + 2]),
                );
            }
        }
        if let Some(ref mut coefficients) = model.surface_color_coefficients {
            let stride = crate::SURFACE_COLOR_COMPONENTS * 3;
            for (component, &offset) in schema.surface_color.iter().enumerate() {
                coefficients[index * stride + component] = read_f32(&row, offset);
            }
        }
        if let Some(ref mut spherical_voronoi) = model.spherical_voronoi {
            let base = index * crate::SPHERICAL_VORONOI_SITES;
            for site in 0..crate::SPHERICAL_VORONOI_SITES {
                let component = site * 3;
                spherical_voronoi.axes[base + site] = glam::Vec3::new(
                    read_f32(&row, schema.spherical_voronoi_axes[component]),
                    read_f32(&row, schema.spherical_voronoi_axes[component + 1]),
                    read_f32(&row, schema.spherical_voronoi_axes[component + 2]),
                );
                spherical_voronoi.colors[base + site] = glam::Vec3::new(
                    read_f32(&row, schema.spherical_voronoi_colors[component]),
                    read_f32(&row, schema.spherical_voronoi_colors[component + 1]),
                    read_f32(&row, schema.spherical_voronoi_colors[component + 2]),
                );
            }
        }
    }

    row = allocate_vec(adjacency.stride, 0u8, "adjacency row")?;
    for neighbor in neighbors.iter_mut() {
        io::Read::read_exact(file, &mut row).map_err(|error| {
            LoadError::invalid(format!("RadFoam PLY adjacency body is truncated: {error}"))
        })?;
        *neighbor = read_u32(&row, schema.adjacency);
    }
    Ok(())
}

fn read_ascii_data_line(
    file: &mut io::BufReader<fs::File>,
    line: &mut String,
) -> Result<bool, LoadError> {
    loop {
        line.clear();
        let mut limited = io::Read::take(&mut *file, (MAX_ASCII_ROW_BYTES + 1) as u64);
        let bytes = io::BufRead::read_line(&mut limited, line)?;
        if bytes == 0 {
            return Ok(false);
        }
        if bytes > MAX_ASCII_ROW_BYTES {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY ASCII row exceeds {MAX_ASCII_ROW_BYTES} bytes"
            )));
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with("comment") {
            return Ok(true);
        }
    }
}

fn parse_f32(token: &str, name: &str) -> Result<f32, LoadError> {
    token
        .parse::<f32>()
        .map_err(|_| LoadError::invalid(format!("invalid RadFoam PLY float for '{name}'")))
}

fn parse_u32(token: &str, name: &str) -> Result<u32, LoadError> {
    token
        .parse::<u32>()
        .map_err(|_| LoadError::invalid(format!("invalid RadFoam PLY uint for '{name}'")))
}

fn parse_u8(token: &str, name: &str) -> Result<u8, LoadError> {
    token
        .parse::<u8>()
        .map_err(|_| LoadError::invalid(format!("invalid RadFoam PLY uchar for '{name}'")))
}

fn decode_ascii(
    file: &mut io::BufReader<fs::File>,
    header: &Header,
    schema: &Schema,
    model: &mut crate::PointCloudModel,
    offsets: &mut [u32],
    neighbors: &mut [u32],
) -> Result<(), LoadError> {
    let vertex = &header.elements[0];
    let adjacency = &header.elements[1];
    let mut line = String::new();
    for index in 0..schema.point_count {
        if !read_ascii_data_line(file, &mut line)? {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY is missing ASCII vertex row {index}"
            )));
        }
        let mut tokens = line.split_whitespace();
        let mut position = glam::Vec4::ZERO;
        let mut adjacency_offset = 0;
        let mut exact_dc = [0.0; 3];
        let mut preview_dc = [0u8; 3];
        let mut radius = 0.0;
        let mut surface_normal = glam::Vec3::ZERO;
        let mut surface_offset = 0.0;
        let mut surface_detail_offsets = [0.0_f32; crate::SURFACE_DETAIL_SITES * 3];
        let mut surface_detail_heights = [0.0_f32; crate::SURFACE_DETAIL_SITES];
        let mut surface_detail_colors = [0.0_f32; crate::SURFACE_DETAIL_SITES * 3];
        let mut surface_color = [0.0_f32; crate::SURFACE_COLOR_COMPONENTS * 3];
        let mut spherical_voronoi_axes = [0.0_f32; crate::SPHERICAL_VORONOI_SITES * 3];
        let mut spherical_voronoi_colors = [0.0_f32; crate::SPHERICAL_VORONOI_SITES * 3];
        let coefficients =
            &mut model.sh_coefficients[index * schema.sh_stride..(index + 1) * schema.sh_stride];
        for property in &vertex.properties {
            let token = tokens.next().ok_or_else(|| {
                LoadError::invalid(format!(
                    "RadFoam PLY ASCII vertex row {index} has too few columns"
                ))
            })?;
            match property.name.as_str() {
                "x" => position.x = parse_f32(token, "x")?,
                "y" => position.y = parse_f32(token, "y")?,
                "z" => position.z = parse_f32(token, "z")?,
                "density" => position.w = parse_f32(token, "density")?,
                "adjacency_offset" => adjacency_offset = parse_u32(token, "adjacency_offset")?,
                "blade_sh_dc_0" => exact_dc[0] = parse_f32(token, "blade_sh_dc_0")?,
                "blade_sh_dc_1" => exact_dc[1] = parse_f32(token, "blade_sh_dc_1")?,
                "blade_sh_dc_2" => exact_dc[2] = parse_f32(token, "blade_sh_dc_2")?,
                "red" => preview_dc[0] = parse_u8(token, "red")?,
                "green" => preview_dc[1] = parse_u8(token, "green")?,
                "blue" => preview_dc[2] = parse_u8(token, "blue")?,
                "radius" => radius = parse_f32(token, "radius")?,
                "nx" => surface_normal.x = parse_f32(token, "nx")?,
                "ny" => surface_normal.y = parse_f32(token, "ny")?,
                "nz" => surface_normal.z = parse_f32(token, "nz")?,
                "surface_offset" => surface_offset = parse_f32(token, "surface_offset")?,
                name if name.starts_with("blade_surface_detail_offset_") => {
                    let component = name["blade_surface_detail_offset_".len()..]
                        .parse::<usize>()
                        .unwrap();
                    surface_detail_offsets[component] = parse_f32(token, name)?;
                }
                name if name.starts_with("blade_surface_detail_height_") => {
                    let component = name["blade_surface_detail_height_".len()..]
                        .parse::<usize>()
                        .unwrap();
                    surface_detail_heights[component] = parse_f32(token, name)?;
                }
                name if name.starts_with("blade_surface_detail_color_") => {
                    let component = name["blade_surface_detail_color_".len()..]
                        .parse::<usize>()
                        .unwrap();
                    surface_detail_colors[component] = parse_f32(token, name)?;
                }
                name if name.starts_with("blade_surface_color_") => {
                    let component = name["blade_surface_color_".len()..]
                        .parse::<usize>()
                        .unwrap();
                    surface_color[component] = parse_f32(token, name)?;
                }
                name if name.starts_with("blade_spherical_voronoi_axis_") => {
                    let component = name["blade_spherical_voronoi_axis_".len()..]
                        .parse::<usize>()
                        .unwrap();
                    spherical_voronoi_axes[component] = parse_f32(token, name)?;
                }
                name if name.starts_with("blade_spherical_voronoi_color_") => {
                    let component = name["blade_spherical_voronoi_color_".len()..]
                        .parse::<usize>()
                        .unwrap();
                    spherical_voronoi_colors[component] = parse_f32(token, name)?;
                }
                name if name.starts_with("color_sh_") => {
                    let rest = name["color_sh_".len()..].parse::<usize>().unwrap();
                    coefficients[3 + rest] = parse_f32(token, name)?;
                }
                _ => {}
            }
        }
        model.points[index] = position;
        offsets[index + 1] = adjacency_offset;
        if schema.exact_dc.is_some() {
            coefficients[..3].copy_from_slice(&exact_dc);
        } else {
            set_preview_dc(coefficients, preview_dc);
        }
        if let Some(ref mut radii) = model.radii {
            radii[index] = radius;
        }
        if let Some(ref mut normals) = model.surface_normals {
            normals[index] = surface_normal;
        }
        if let Some(ref mut offsets) = model.surface_offsets {
            offsets[index] = surface_offset;
        }
        if let Some(ref mut detail) = model.surface_detail {
            let base = index * crate::SURFACE_DETAIL_SITES;
            for (site, &height) in surface_detail_heights.iter().enumerate() {
                let component = site * 3;
                detail.offsets[base + site] =
                    glam::Vec3::from_slice(&surface_detail_offsets[component..component + 3]);
                detail.heights[base + site] = height;
                detail.colors[base + site] =
                    glam::Vec3::from_slice(&surface_detail_colors[component..component + 3]);
            }
        }
        if let Some(ref mut coefficients) = model.surface_color_coefficients {
            let stride = crate::SURFACE_COLOR_COMPONENTS * 3;
            coefficients[index * stride..(index + 1) * stride].copy_from_slice(&surface_color);
        }
        if let Some(ref mut spherical_voronoi) = model.spherical_voronoi {
            let base = index * crate::SPHERICAL_VORONOI_SITES;
            for site in 0..crate::SPHERICAL_VORONOI_SITES {
                let component = site * 3;
                spherical_voronoi.axes[base + site] =
                    glam::Vec3::from_slice(&spherical_voronoi_axes[component..component + 3]);
                spherical_voronoi.colors[base + site] =
                    glam::Vec3::from_slice(&spherical_voronoi_colors[component..component + 3]);
            }
        }
    }

    for (index, neighbor) in neighbors.iter_mut().enumerate() {
        if !read_ascii_data_line(file, &mut line)? {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY is missing ASCII adjacency row {index}"
            )));
        }
        let mut tokens = line.split_whitespace();
        let mut value = None;
        for property in &adjacency.properties {
            let token = tokens.next().ok_or_else(|| {
                LoadError::invalid(format!(
                    "RadFoam PLY ASCII adjacency row {index} has too few columns"
                ))
            })?;
            if property.name == "adjacency" {
                value = Some(parse_u32(token, "adjacency")?);
            }
        }
        *neighbor = value.unwrap();
    }
    if read_ascii_data_line(file, &mut line)? {
        return Err(LoadError::invalid(
            "unexpected trailing RadFoam PLY ASCII data",
        ));
    }
    Ok(())
}

fn validate_csr(offsets: &[u32], neighbors: &[u32], point_count: usize) -> Result<(), LoadError> {
    if offsets.len() != point_count + 1 || offsets[0] != 0 {
        return Err(LoadError::invalid("invalid RadFoam PLY CSR offset array"));
    }
    if offsets[point_count] as usize != neighbors.len() {
        return Err(LoadError::invalid(format!(
            "RadFoam PLY final CSR offset is {}, expected {}",
            offsets[point_count],
            neighbors.len()
        )));
    }
    for index in 0..point_count {
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        if start > end || end > neighbors.len() {
            return Err(LoadError::invalid(format!(
                "invalid RadFoam PLY CSR range [{start}, {end}) for point {index}"
            )));
        }
    }
    for (entry, &neighbor) in neighbors.iter().enumerate() {
        if neighbor as usize >= point_count {
            return Err(LoadError::invalid(format!(
                "RadFoam PLY adjacency entry {entry} references point {neighbor}, but there are {point_count} points"
            )));
        }
    }
    Ok(())
}

pub fn try_load(file_path: &str) -> Result<crate::PointCloudModel, LoadError> {
    let mut file = io::BufReader::new(fs::File::open(file_path)?);
    let header = read_header(&mut file)?;
    let schema = Schema::new(&header)?;
    let body_start = io::Seek::stream_position(&mut file)?;
    let body_bytes = file.get_ref().metadata()?.len().saturating_sub(body_start);
    match header.format {
        PlyFormat::BinaryLittleEndian => {
            let expected = schema
                .point_count
                .checked_mul(header.elements[0].stride)
                .and_then(|value| {
                    schema
                        .adjacency_count
                        .checked_mul(header.elements[1].stride)
                        .and_then(|adjacency| value.checked_add(adjacency))
                })
                .ok_or_else(|| LoadError::invalid("RadFoam PLY binary body size overflow"))?;
            let expected = u64::try_from(expected)
                .map_err(|_| LoadError::invalid("RadFoam PLY body exceeds file limits"))?;
            if body_bytes != expected {
                return Err(LoadError::invalid(format!(
                    "RadFoam PLY binary body is {body_bytes} bytes, expected {expected}"
                )));
            }
        }
        PlyFormat::Ascii => {
            let records = schema
                .point_count
                .checked_add(schema.adjacency_count)
                .ok_or_else(|| LoadError::invalid("RadFoam PLY ASCII record count overflow"))?;
            if u64::try_from(records).unwrap_or(u64::MAX) > body_bytes {
                return Err(LoadError::invalid(
                    "RadFoam PLY ASCII element counts are implausible for the file size",
                ));
            }
        }
    }

    log::info!(
        "RadFoam PLY: {} points, {} adjacency entries, SH degree {}, format {:?}",
        schema.point_count,
        schema.adjacency_count,
        schema.sh_degree,
        header.format
    );
    let (mut model, mut offsets, mut neighbors) = allocate_model(&schema)?;
    match header.format {
        PlyFormat::BinaryLittleEndian => decode_binary(
            &mut file,
            &header,
            &schema,
            &mut model,
            &mut offsets,
            &mut neighbors,
        )?,
        PlyFormat::Ascii => decode_ascii(
            &mut file,
            &header,
            &schema,
            &mut model,
            &mut offsets,
            &mut neighbors,
        )?,
    }
    validate_csr(&offsets, &neighbors, schema.point_count)?;
    model.adjacency = Some(crate::Adjacency { neighbors, offsets });
    model
        .validate()
        .map_err(|error| LoadError::invalid(format!("RadFoam PLY model: {error}")))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "blade-volume-radfoam-{name}-{}-{:?}.ply",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn malformed_and_implausible_files_return_errors_before_allocation() {
        let malformed = path("malformed");
        fs::write(&malformed, b"ply\nformat ascii 1.0\nend_header\n").unwrap();
        assert!(matches!(
            try_load(malformed.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(malformed).unwrap();

        let implausible = path("implausible");
        let data = b"ply\nformat binary_little_endian 1.0\nelement vertex 4294967295\nproperty float x\nproperty float y\nproperty float z\nproperty float density\nproperty uint adjacency_offset\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement adjacency 0\nproperty uint adjacency\nend_header\n";
        fs::write(&implausible, data).unwrap();
        assert!(matches!(
            try_load(implausible.to_str().unwrap()),
            Err(LoadError::InvalidData(_))
        ));
        fs::remove_file(implausible).unwrap();
    }
}
