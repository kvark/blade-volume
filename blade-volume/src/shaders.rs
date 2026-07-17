pub const COMMON: &str = include_str!("../shaders/common.wgsl");
pub const SH_EVAL: &str = include_str!("../shaders/sh_eval.wgsl");
pub const RADFOAM_TRACE: &str = include_str!("../shaders/radfoam_trace.wgsl");
pub const GAUSSIAN_TRACE: &str = include_str!("../shaders/gaussian_trace.wgsl");
pub const RADFOAM: &str = include_str!("../shaders/radfoam.wgsl");
pub const GAUSSIAN: &str = include_str!("../shaders/gaussian.wgsl");
pub const SCENE_TRAVERSE: &str = include_str!("../shaders/scene_traverse.wgsl");
pub const SCENE_RADFOAM: &str = include_str!("../shaders/scene_radfoam.wgsl");
pub const RADFOAM_BLIT: &str = include_str!("../shaders/radfoam_blit.wgsl");
pub const RADFOAM_RECORD_PATHS: &str = include_str!("../shaders/radfoam_record_paths.wgsl");

/// Library of WGSL fragments referenced by `// #include "<name>"` directives.
const INCLUDES: &[(&str, &str)] = &[
    ("common.wgsl", COMMON),
    ("sh_eval.wgsl", SH_EVAL),
    ("radfoam_trace.wgsl", RADFOAM_TRACE),
    ("gaussian_trace.wgsl", GAUSSIAN_TRACE),
    (
        "scene_bindings.wgsl",
        include_str!("../shaders/scene_bindings.wgsl"),
    ),
    (
        "scene_trace_core.wgsl",
        include_str!("../shaders/scene_trace_core.wgsl"),
    ),
];

const MAX_INCLUDE_DEPTH: usize = 10;

/// Expand `// #include "<name>"` directives in a WGSL source by inlining fragments
/// from the internal include table. Nested includes have a fixed depth limit.
///
/// Panics if the include name is unknown or the recursion depth is exceeded.
pub fn compose(source: &str) -> String {
    let mut result = compose_recursive(source, 0);
    if let Some(query) = source
        .lines()
        .find_map(|line| line.trim().strip_prefix("// #gaussian_query "))
    {
        result = result.replace(
            "// #initialize_gaussian_query",
            &compose_gaussian_query(query),
        );
    }
    result
}

fn compose_gaussian_query(query: &str) -> String {
    let mut words = query.split_whitespace();
    match words.next() {
        Some("scalar") => {
            let tlas = words.next().expect("missing scalar Gaussian TLAS name");
            assert!(words.next().is_none(), "invalid scalar Gaussian query");
            format!("rayQueryInitialize(&rq, {tlas}, desc);")
        }
        Some("array") => {
            let tlas = words.next().expect("missing Gaussian TLAS array name");
            let index = words.next().expect("missing Gaussian TLAS array index");
            let count = words
                .next()
                .expect("missing Gaussian TLAS array size")
                .parse::<u32>()
                .expect("invalid Gaussian TLAS array size");
            assert!(count > 0, "Gaussian TLAS array must not be empty");
            assert!(words.next().is_none(), "invalid array Gaussian query");

            let mut output = format!("switch ({index}) {{\n");
            for i in 0..count {
                output.push_str(&format!(
                    "case {i}u: {{ rayQueryInitialize(&rq, {tlas}[{i}], desc); }}\n"
                ));
            }
            output.push_str(&format!(
                "default: {{ rayQueryInitialize(&rq, {tlas}[0], desc); }}\n}}"
            ));
            output
        }
        _ => panic!("invalid Gaussian query declaration: {query}"),
    }
}

fn compose_recursive(source: &str, depth: usize) -> String {
    if depth > MAX_INCLUDE_DEPTH {
        panic!("WGSL include depth exceeded {MAX_INCLUDE_DEPTH} — possible cycle");
    }

    let mut result = String::with_capacity(source.len() * 2);
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("// #include") {
            let rest = rest.trim();
            if let Some(name) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                let body = INCLUDES
                    .iter()
                    .find_map(|&(n, body)| if n == name { Some(body) } else { None })
                    .unwrap_or_else(|| panic!("Unknown WGSL include: {name}"));
                result.push_str("// === Begin included: ");
                result.push_str(name);
                result.push_str(" ===\n");
                result.push_str(&compose_recursive(body, depth + 1));
                result.push_str("\n// === End included: ");
                result.push_str(name);
                result.push_str(" ===\n");
                continue;
            }
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{compose, SCENE_RADFOAM, SCENE_TRAVERSE};

    #[test]
    fn expands_a_known_include() {
        let out = compose("// #include \"common.wgsl\"\nfoo");
        assert!(out.contains("Begin included: common.wgsl"));
        assert!(out.ends_with("foo\n"));
    }

    #[test]
    #[should_panic(expected = "Unknown WGSL include")]
    fn panics_on_unknown_include() {
        compose("// #include \"nope.wgsl\"");
    }

    #[test]
    fn leaves_unrelated_lines_alone() {
        let src = "fn main() {}\n";
        assert_eq!(compose(src), src);
    }

    #[test]
    fn expands_scalar_gaussian_query() {
        let src = "// #gaussian_query scalar tlas\n// #initialize_gaussian_query\n";
        let out = compose(src);
        assert!(out.contains("rayQueryInitialize(&rq, tlas, desc);"));
    }

    #[test]
    fn expands_array_gaussian_query_with_constant_indices() {
        let src = "// #gaussian_query array tlas index 2\n// #initialize_gaussian_query\n";
        let out = compose(src);
        assert!(out.contains("switch (index)"));
        assert!(out.contains("tlas[0]"));
        assert!(out.contains("tlas[1]"));
        assert!(!out.contains("tlas[2]"));
    }

    #[test]
    fn scene_variants_keep_ray_queries_out_of_radfoam_only_shader() {
        let radfoam = compose(SCENE_RADFOAM);
        assert!(radfoam.contains("fn trace_scene"));
        assert!(radfoam.contains("enable wgpu_binding_array"));
        assert!(!radfoam.contains("enable wgpu_ray_query"));
        assert!(!radfoam.contains("g_gaussian_tlas"));

        let mixed = compose(SCENE_TRAVERSE);
        assert!(mixed.contains("enable wgpu_binding_array"));
        assert!(mixed.contains("enable wgpu_ray_query"));
        assert!(mixed.contains("g_gaussian_tlas"));
        assert!(!mixed.contains("#initialize_gaussian_query"));
    }
}
