pub const COMMON: &str = include_str!("../shaders/common.wgsl");
pub const SH_EVAL: &str = include_str!("../shaders/sh_eval.wgsl");
pub const RADFOAM_TRACE: &str = include_str!("../shaders/radfoam_trace.wgsl");
pub const GAUSSIAN_TRACE: &str = include_str!("../shaders/gaussian_trace.wgsl");
pub const RADFOAM: &str = include_str!("../shaders/radfoam.wgsl");
pub const GAUSSIAN: &str = include_str!("../shaders/gaussian.wgsl");
pub const SCENE_TRAVERSE: &str = include_str!("../shaders/scene_traverse.wgsl");
pub const RADFOAM_BLIT: &str = include_str!("../shaders/radfoam_blit.wgsl");

/// Library of WGSL fragments referenced by `// #include "<name>"` directives.
const INCLUDES: &[(&str, &str)] = &[
    ("common.wgsl", COMMON),
    ("sh_eval.wgsl", SH_EVAL),
    ("radfoam_trace.wgsl", RADFOAM_TRACE),
    ("gaussian_trace.wgsl", GAUSSIAN_TRACE),
];

const MAX_INCLUDE_DEPTH: usize = 10;

/// Expand `// #include "<name>"` directives in a WGSL source by inlining fragments
/// from [`INCLUDES`]. Nested includes are supported up to [`MAX_INCLUDE_DEPTH`].
///
/// Panics if the include name is unknown or the recursion depth is exceeded.
pub fn compose(source: &str) -> String {
    compose_recursive(source, 0)
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
    use super::compose;

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
}
