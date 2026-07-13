//! Loader for upstream Radiant Foam (RadFoam) scenes exported as PLY.
//!
//! This expects the format written by upstream `RadFoamScene.save_ply()`:
//! - `element vertex <N>` with at least:
//!   - `x`, `y`, `z` (float32)
//!   - `density` (float32)
//!   - `adjacency_offset` (uint32)  // stored as offsets[i+1]
//!   - `color_sh_0..color_sh_M-1` (float32) // SH coefficients excluding DC in upstream exporter
//!   - `red`, `green`, `blue` (uchar) // DC-derived preview colors
//! - `element adjacency <K>` with:
//!   - `adjacency` (uint32) // flattened neighbor indices
//! - blade-volume files additionally carry `blade_sh_dc_0..2` (float32),
//!   preserving the exact DC coefficients while retaining the upstream RGB
//!   preview properties for compatibility.
//!
//! The upstream tracer expects a packed per-point attribute row of length:
//!     attr_dim = 1 + 3 * (1 + sh_degree)^2
//! with the last scalar being density `s` and the first `3 * (1+sh_degree)^2` scalars being
//! SH coefficients laid out as 3 channels interleaved per SH basis component.
//!
//! Upstream training stores DC separately (att_dc) and SH rest separately (att_sh). The PLY
//! written by `save_ply()` includes:
//! - RGB preview bytes (`red/green/blue`) derived from DC (for visualization only)
//! - `color_sh_*` fields containing *only* att_sh values (excluding DC)
//! - `density`
//!
//! Therefore, this loader packs SH coefficients as:
//! - DC loaded exactly from `blade_sh_dc_0..2` when present, otherwise
//!   approximated from upstream's `red/green/blue` preview fields.
//! - Remaining coefficients loaded from `color_sh_*` in-order.
//! - Density appended as the last scalar.
//!
//! This is sufficient to build buffers, validate adjacency, and run a tracer implementation.
//!
//! Supported PLY formats:
//! - `format binary_little_endian 1.0`
//! - `format ascii 1.0` (added for test fixtures and broader compatibility)
//!
//! Notes:
//! - Only the `vertex` and `adjacency` elements are used.
//! - Property types supported:
//!   - `float` (f32)
//!   - `uchar` (u8) for preview color
//!   - `uint`/`uint32` (u32) for adjacency data and offsets
//!
//! Any extra properties are skipped if their type size is known.

use std::{collections::HashMap, fs, io};

fn validate_csr(adjacency_offsets: &[u32], point_adjacency: &[u32], num_points: usize) {
    // Validate CSR offsets
    if adjacency_offsets.is_empty() || adjacency_offsets.len() != num_points + 1 {
        panic!(
            "Invalid adjacency_offsets length: expected {}, got {}",
            num_points + 1,
            adjacency_offsets.len()
        );
    }
    if adjacency_offsets[0] != 0 {
        panic!(
            "Invalid adjacency_offsets[0]: expected 0, got {}",
            adjacency_offsets[0]
        );
    }
    let last = adjacency_offsets[num_points] as usize;
    if last != point_adjacency.len() {
        panic!(
            "Invalid adjacency_offsets[N]: expected {}, got {}",
            point_adjacency.len(),
            last
        );
    }
    for i in 0..num_points {
        let a = adjacency_offsets[i] as usize;
        let b = adjacency_offsets[i + 1] as usize;
        if b < a || b > point_adjacency.len() {
            panic!("Invalid adjacency offset range for point {i}: [{a}, {b})");
        }
    }

    // Validate adjacency indices are in-bounds
    for (k, &idx) in point_adjacency.iter().enumerate() {
        if (idx as usize) >= num_points {
            panic!("Adjacency index out of bounds at entry {k}: {idx} (num_points={num_points})");
        }
    }
}

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
    fn size_bytes(self) -> usize {
        match self {
            PlyScalarType::Float32 => 4,
            PlyScalarType::Uint32 => 4,
            PlyScalarType::Uint8 => 1,
        }
    }
}

fn parse_scalar_type(word: &str) -> PlyScalarType {
    match word {
        "float" | "float32" => PlyScalarType::Float32,
        "uint" | "uint32" => PlyScalarType::Uint32,
        "uchar" | "uint8" => PlyScalarType::Uint8,
        other => panic!("Unsupported PLY scalar type: {other}"),
    }
}

#[derive(Debug, Clone)]
struct Property {
    name: String,
    ty: PlyScalarType,
    offset: usize,
}

#[derive(Debug, Clone)]
struct Element {
    name: String,
    count: usize,
    stride: usize,
    props: Vec<Property>,
}

impl Element {
    fn prop_offset(&self, name: &str) -> Option<(PlyScalarType, usize)> {
        self.props
            .iter()
            .find(|p| p.name == name)
            .map(|p| (p.ty, p.offset))
    }
}

#[derive(Debug)]
struct Header {
    format: PlyFormat,
    elements: Vec<Element>,
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    let b = &bytes[offset..offset + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    let b = &bytes[offset..offset + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_u8(bytes: &[u8], offset: usize) -> u8 {
    bytes[offset]
}

fn parse_header(mut file: io::BufReader<fs::File>) -> (Header, io::BufReader<fs::File>) {
    use std::io::BufRead as _;

    let mut line = String::new();

    // First line
    file.read_line(&mut line).unwrap();
    if line.trim() != "ply" {
        panic!("Not a PLY file (expected 'ply' first line)");
    }
    line.clear();

    let mut format: Option<PlyFormat> = None;
    let mut elements: Vec<Element> = Vec::new();
    let mut current_element: Option<Element> = None;

    while file.read_line(&mut line).unwrap() > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        let mut words = trimmed.split_whitespace();
        let head = words.next().unwrap();

        match head {
            "format" => {
                let fmt = words.next().unwrap_or("");
                let ver = words.next().unwrap_or("");
                if ver != "1.0" {
                    panic!("Unsupported PLY format version: {ver}");
                }
                match fmt {
                    "binary_little_endian" => format = Some(PlyFormat::BinaryLittleEndian),
                    "ascii" => format = Some(PlyFormat::Ascii),
                    other => panic!("Unsupported PLY format: {other}"),
                }
            }
            "comment" => {
                // ignore
            }
            "element" => {
                // finalize previous
                if let Some(el) = current_element.take() {
                    elements.push(el);
                }
                let name = words.next().unwrap().to_string();
                let count: usize = words
                    .next()
                    .unwrap_or_else(|| panic!("Missing element count for '{name}'"))
                    .parse()
                    .unwrap();
                current_element = Some(Element {
                    name,
                    count,
                    stride: 0,
                    props: Vec::new(),
                });
            }
            "property" => {
                let el = current_element
                    .as_mut()
                    .unwrap_or_else(|| panic!("'property' encountered before any 'element'"));

                // We only support scalar properties for MVP.
                // Upstream PLY uses only scalar properties in the exported file.
                let ty_word = words.next().unwrap_or("");
                if ty_word == "list" {
                    let _count_ty = words.next().unwrap_or("");
                    let _item_ty = words.next().unwrap_or("");
                    let name = words.next().unwrap_or("");
                    panic!(
                        "Unsupported PLY list property '{name}' in element '{}'",
                        el.name
                    );
                }
                let ty = parse_scalar_type(ty_word);
                let name = words.next().unwrap_or("").to_string();
                if name.is_empty() {
                    panic!("PLY property without a name in element '{}'", el.name);
                }

                let offset = el.stride;
                el.props.push(Property { name, ty, offset });
                el.stride += ty.size_bytes();
            }
            "end_header" => {
                break;
            }
            other => {
                panic!("Unexpected PLY header token: {other}");
            }
        }

        line.clear();
    }

    if let Some(el) = current_element.take() {
        elements.push(el);
    }
    let format = format.unwrap_or_else(|| panic!("PLY header missing 'format'"));

    (Header { format, elements }, file)
}

fn infer_sh_degree_from_sh_rest_count(sh_rest_count: usize) -> usize {
    // Upstream stores `att_sh` with dimension:
    //   3 * ((1 + deg)^2 - 1)
    // where `deg` is SH degree and component 0 (DC) is excluded.
    //
    // So:
    //   sh_rest_count / 3 = (1 + deg)^2 - 1
    //   (1 + deg)^2 = sh_rest_count / 3 + 1
    // We solve for deg by trying 0..=3 (consistent with this crate’s MAX_SH_DEGREE).
    if !sh_rest_count.is_multiple_of(3) {
        panic!("color_sh_* property count must be divisible by 3, got {sh_rest_count}");
    }
    let per_channel = sh_rest_count / 3;
    let target = per_channel + 1;
    for deg in 0..=crate::MAX_SH_DEGREE {
        let comps = crate::get_sh_component_count(deg);
        if comps == target {
            return deg;
        }
    }
    panic!(
        "Unable to infer SH degree from color_sh_* count: {sh_rest_count} (target components: {target})"
    );
}

pub fn load(file_path: &str) -> crate::PointCloudModel {
    use std::io::{BufRead as _, Read as _};

    assert!(
        file_path.ends_with(".ply"),
        "RadFoam loader expects a .ply file"
    );

    let file = fs::File::open(file_path).unwrap();
    let reader = io::BufReader::new(file);
    let (header, mut file) = parse_header(reader);

    // Index elements by name
    let mut element_map: HashMap<String, Element> = HashMap::new();
    for el in &header.elements {
        element_map.insert(el.name.clone(), el.clone());
    }

    let vertex_el = element_map
        .get("vertex")
        .unwrap_or_else(|| panic!("PLY missing 'vertex' element"))
        .clone();
    let adjacency_el = element_map
        .get("adjacency")
        .unwrap_or_else(|| panic!("PLY missing 'adjacency' element"))
        .clone();

    // Validate required vertex properties
    let (x_ty, x_off) = vertex_el
        .prop_offset("x")
        .unwrap_or_else(|| panic!("vertex element missing 'x'"));
    let (y_ty, y_off) = vertex_el
        .prop_offset("y")
        .unwrap_or_else(|| panic!("vertex element missing 'y'"));
    let (z_ty, z_off) = vertex_el
        .prop_offset("z")
        .unwrap_or_else(|| panic!("vertex element missing 'z'"));

    if x_ty != PlyScalarType::Float32
        || y_ty != PlyScalarType::Float32
        || z_ty != PlyScalarType::Float32
    {
        panic!("vertex x/y/z must be float32");
    }

    let (density_ty, density_off) = vertex_el
        .prop_offset("density")
        .unwrap_or_else(|| panic!("vertex element missing 'density'"));
    if density_ty != PlyScalarType::Float32 {
        panic!("vertex density must be float32");
    }

    let (adj_off_ty, adj_off_off) = vertex_el
        .prop_offset("adjacency_offset")
        .unwrap_or_else(|| panic!("vertex element missing 'adjacency_offset'"));
    if adj_off_ty != PlyScalarType::Uint32 {
        panic!("vertex adjacency_offset must be uint32");
    }

    // Optional preview color (uchar) fields, used to approximate missing DC
    // coefficients in upstream files.
    let red_prop = vertex_el.prop_offset("red");
    let green_prop = vertex_el.prop_offset("green");
    let blue_prop = vertex_el.prop_offset("blue");

    // blade-volume extension: exact DC coefficients. All three channels must
    // be present together so a partially-written file cannot silently mix
    // exact and quantized channels.
    let exact_dc_offsets = match [
        vertex_el.prop_offset("blade_sh_dc_0"),
        vertex_el.prop_offset("blade_sh_dc_1"),
        vertex_el.prop_offset("blade_sh_dc_2"),
    ] {
        [Some((r_ty, r_off)), Some((g_ty, g_off)), Some((b_ty, b_off))] => {
            if r_ty != PlyScalarType::Float32
                || g_ty != PlyScalarType::Float32
                || b_ty != PlyScalarType::Float32
            {
                panic!("vertex blade_sh_dc_0..2 must be float32");
            }
            Some([r_off, g_off, b_off])
        }
        [None, None, None] => None,
        _ => panic!("vertex must contain either all or none of blade_sh_dc_0..2"),
    };

    // Optional per-point radius/weight (Power Foam).
    let radius_prop = vertex_el.prop_offset("radius");
    if let Some((ty, _)) = radius_prop {
        if ty != PlyScalarType::Float32 {
            panic!("vertex radius must be float32");
        }
    }

    // Discover color_sh_* properties and their offsets (in order by index)
    let mut color_sh: Vec<(usize, usize)> = Vec::new(); // (index, offset)
    for p in &vertex_el.props {
        if let Some(rest) = p.name.strip_prefix("color_sh_") {
            let idx: usize = rest.parse().unwrap_or_else(|_| {
                panic!("Invalid color_sh_* property name: {}", p.name);
            });
            if p.ty != PlyScalarType::Float32 {
                panic!("{} must be float32", p.name);
            }
            color_sh.push((idx, p.offset));
        }
    }
    color_sh.sort_by_key(|&(idx, _)| idx);

    // Ensure indices are contiguous 0..M-1 if present.
    if !color_sh.is_empty() {
        for (expect, &(idx, _)) in color_sh.iter().enumerate() {
            if idx != expect {
                panic!(
                    "color_sh_* indices must be contiguous starting from 0, expected {expect} got {idx}"
                );
            }
        }
    }

    let sh_rest_count = color_sh.len();
    let sh_degree = infer_sh_degree_from_sh_rest_count(sh_rest_count);
    let sh_components = crate::get_sh_component_count(sh_degree);
    let sh_dim = 3 * sh_components;
    let attr_dim = 1 + sh_dim;

    // Upstream PLY stores only SH-rest (excluding DC): 3 * (sh_components - 1).
    let expected_rest = 3 * (sh_components.saturating_sub(1));
    if sh_rest_count != expected_rest {
        panic!(
            "color_sh_* count mismatch for inferred degree {sh_degree}: got {sh_rest_count}, expected {expected_rest}"
        );
    }

    // Adjacency element format
    let (adj_ty, adj_off) = adjacency_el
        .prop_offset("adjacency")
        .unwrap_or_else(|| panic!("adjacency element missing 'adjacency' property"));
    if adj_ty != PlyScalarType::Uint32 {
        panic!("adjacency property must be uint32");
    }

    let num_points = vertex_el.count;
    let num_adjacency = adjacency_el.count;

    log::info!(
        "RadFoam PLY: {} points, {} adjacency entries, SH degree {}, attr_dim {}, format {:?}",
        num_points,
        num_adjacency,
        sh_degree,
        attr_dim,
        header.format
    );

    // Read vertex records
    // Points: Vec4 with xyz=position, w=density
    let mut points = vec![glam::Vec4::ZERO; num_points];
    let mut adjacency_offsets = vec![0u32; num_points + 1];

    // SH coefficients only (no density - that goes in points.w). Prefer our
    // exact extension, and approximate DC from upstream preview bytes when it
    // is absent.
    let mut sh_coefficients = vec![0.0f32; num_points * sh_dim];

    let mut radii: Option<Vec<f32>> = radius_prop.map(|_| vec![0.0f32; num_points]);

    const C0: f32 = 0.282_094_8;

    match header.format {
        PlyFormat::BinaryLittleEndian => {
            let mut vertex_row = vec![0u8; vertex_el.stride];
            for i in 0..num_points {
                file.read_exact(&mut vertex_row).unwrap();

                let x = read_f32_le(&vertex_row, x_off);
                let y = read_f32_le(&vertex_row, y_off);
                let z = read_f32_le(&vertex_row, z_off);
                let density = read_f32_le(&vertex_row, density_off);
                points[i] = glam::Vec4::new(x, y, z, density);

                // adjacency_offset stores offsets[i+1] in upstream exporter
                let end_off = read_u32_le(&vertex_row, adj_off_off);
                adjacency_offsets[i + 1] = end_off;

                // SH layout: [R_comp0, G_comp0, B_comp0, R_comp1, G_comp1, B_comp1, ...]
                let base = i * sh_dim;

                if let Some(offsets) = exact_dc_offsets {
                    sh_coefficients[base] = read_f32_le(&vertex_row, offsets[0]);
                    sh_coefficients[base + 1] = read_f32_le(&vertex_row, offsets[1]);
                    sh_coefficients[base + 2] = read_f32_le(&vertex_row, offsets[2]);
                } else if let (Some((r_ty, r_off)), Some((g_ty, g_off)), Some((b_ty, b_off))) =
                    (red_prop, green_prop, blue_prop)
                {
                    if r_ty != PlyScalarType::Uint8
                        || g_ty != PlyScalarType::Uint8
                        || b_ty != PlyScalarType::Uint8
                    {
                        panic!("vertex red/green/blue must be uchar (uint8) if present");
                    }
                    let r8 = read_u8(&vertex_row, r_off) as f32 / 255.0;
                    let g8 = read_u8(&vertex_row, g_off) as f32 / 255.0;
                    let b8 = read_u8(&vertex_row, b_off) as f32 / 255.0;

                    sh_coefficients[base] = (r8 - 0.5) / C0;
                    sh_coefficients[base + 1] = (g8 - 0.5) / C0;
                    sh_coefficients[base + 2] = (b8 - 0.5) / C0;
                }

                // Fill components 1..sh_components-1 from color_sh_*
                if !color_sh.is_empty() {
                    for (j, &(_, off)) in color_sh.iter().enumerate() {
                        let v = read_f32_le(&vertex_row, off);
                        let comp = 1 + (j / 3);
                        let ch = j % 3;
                        let dst = base + 3 * comp + ch;
                        sh_coefficients[dst] = v;
                    }
                }

                if let (Some(ref mut radii), Some((_, off))) = (radii.as_mut(), radius_prop) {
                    radii[i] = read_f32_le(&vertex_row, off);
                }
            }

            // Read adjacency records
            let mut point_adjacency = vec![0u32; num_adjacency];
            let mut adj_row = vec![0u8; adjacency_el.stride];
            for slot in point_adjacency.iter_mut().take(num_adjacency) {
                file.read_exact(&mut adj_row).unwrap();
                *slot = read_u32_le(&adj_row, adj_off);
            }

            // Ensure there is no trailing data (optional strictness)
            {
                let mut extra = [0u8; 1];
                let n = file.read(&mut extra).unwrap();
                if n != 0 {
                    log::warn!("RadFoam PLY has trailing bytes after expected elements");
                }
            }

            // Validate CSR offsets + indices
            validate_csr(&adjacency_offsets, &point_adjacency, num_points);

            crate::PointCloudModel {
                points,
                sh_coefficients,
                sh_degree,
                transforms: None,
                adjacency: Some(crate::Adjacency {
                    neighbors: point_adjacency,
                    offsets: adjacency_offsets,
                }),
                radii,
            }
        }
        PlyFormat::Ascii => {
            // ASCII: read per-row tokens.
            // NOTE: parse_header left the reader positioned at the first data line, so we can continue using BufRead.
            let mut line = String::new();

            for i in 0..num_points {
                // Skip empty lines and comment lines (common in ASCII PLY fixtures).
                line.clear();
                file.read_line(&mut line).unwrap();
                while line.trim().is_empty() || line.trim_start().starts_with("comment") {
                    line.clear();
                    file.read_line(&mut line).unwrap();
                }
                let parts: Vec<&str> = line.split_whitespace().collect();

                // Expect at least as many tokens as there are properties.
                if parts.len() < vertex_el.props.len() {
                    panic!(
                        "ASCII vertex row {} has {} columns, expected at least {}",
                        i,
                        parts.len(),
                        vertex_el.props.len()
                    );
                }

                // We parse by property order, not by offsets (offsets are for binary).
                // This is robust as long as the header order matches row order (PLY spec).

                let mut x = 0.0f32;
                let mut y = 0.0f32;
                let mut z = 0.0f32;
                let mut density = 0.0f32;
                let mut end_off: u32 = 0;
                let mut radius: Option<f32> = None;

                // optional RGB
                let mut r8_opt: Option<u8> = None;
                let mut g8_opt: Option<u8> = None;
                let mut b8_opt: Option<u8> = None;
                let mut exact_dc = [0.0_f32; 3];

                // SH-rest floats in order
                let mut sh_rest_vals: Vec<f32> = Vec::with_capacity(color_sh.len());

                for (col, p) in vertex_el.props.iter().enumerate() {
                    let tok = parts[col];
                    match (p.name.as_str(), p.ty) {
                        ("x", PlyScalarType::Float32) => x = tok.parse().unwrap(),
                        ("y", PlyScalarType::Float32) => y = tok.parse().unwrap(),
                        ("z", PlyScalarType::Float32) => z = tok.parse().unwrap(),
                        ("density", PlyScalarType::Float32) => density = tok.parse().unwrap(),
                        ("adjacency_offset", PlyScalarType::Uint32) => {
                            end_off = tok.parse().unwrap()
                        }
                        ("red", PlyScalarType::Uint8) => r8_opt = Some(tok.parse::<u8>().unwrap()),
                        ("green", PlyScalarType::Uint8) => {
                            g8_opt = Some(tok.parse::<u8>().unwrap())
                        }
                        ("blue", PlyScalarType::Uint8) => b8_opt = Some(tok.parse::<u8>().unwrap()),
                        ("blade_sh_dc_0", PlyScalarType::Float32) => {
                            exact_dc[0] = tok.parse().unwrap()
                        }
                        ("blade_sh_dc_1", PlyScalarType::Float32) => {
                            exact_dc[1] = tok.parse().unwrap()
                        }
                        ("blade_sh_dc_2", PlyScalarType::Float32) => {
                            exact_dc[2] = tok.parse().unwrap()
                        }
                        ("radius", PlyScalarType::Float32) => radius = Some(tok.parse().unwrap()),
                        (name, PlyScalarType::Float32) if name.starts_with("color_sh_") => {
                            sh_rest_vals.push(tok.parse().unwrap())
                        }
                        // Skip unsupported-but-sized props
                        _ => {}
                    }
                }

                points[i] = glam::Vec4::new(x, y, z, density);
                adjacency_offsets[i + 1] = end_off;

                let base = i * sh_dim;

                if exact_dc_offsets.is_some() {
                    sh_coefficients[base..base + 3].copy_from_slice(&exact_dc);
                } else if let (Some(r8), Some(g8), Some(b8)) = (r8_opt, g8_opt, b8_opt) {
                    let r = (r8 as f32) / 255.0;
                    let g = (g8 as f32) / 255.0;
                    let b = (b8 as f32) / 255.0;
                    sh_coefficients[base] = (r - 0.5) / C0;
                    sh_coefficients[base + 1] = (g - 0.5) / C0;
                    sh_coefficients[base + 2] = (b - 0.5) / C0;
                }

                // Fill components 1.. from parsed SH-rest
                for (j, v) in sh_rest_vals.iter().enumerate() {
                    let comp = 1 + (j / 3);
                    let ch = j % 3;
                    let dst = base + 3 * comp + ch;
                    sh_coefficients[dst] = *v;
                }

                if let (Some(ref mut radii), Some(r)) = (radii.as_mut(), radius) {
                    radii[i] = r;
                }
            }

            // adjacency element lines
            let mut point_adjacency = vec![0u32; num_adjacency];
            for slot in point_adjacency.iter_mut() {
                // Skip empty lines and comment lines.
                line.clear();
                file.read_line(&mut line).unwrap();
                while line.trim().is_empty() || line.trim_start().starts_with("comment") {
                    line.clear();
                    file.read_line(&mut line).unwrap();
                }
                let tok = line.split_whitespace().next().unwrap();
                *slot = tok.parse().unwrap();
            }

            // Validate CSR offsets + indices
            validate_csr(&adjacency_offsets, &point_adjacency, num_points);

            crate::PointCloudModel {
                points,
                sh_coefficients,
                sh_degree,
                transforms: None,
                adjacency: Some(crate::Adjacency {
                    neighbors: point_adjacency,
                    offsets: adjacency_offsets,
                }),
                radii,
            }
        }
    }
}
