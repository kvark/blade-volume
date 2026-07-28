use blade_volume as vol;
use blade_volume_convert as convert;
use std::{env, path, process, time};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let parsed = match parse_args(&args) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => {
            println!("{}", usage_text());
            return;
        }
        Err(ref err) => usage(err),
    };

    if parsed.surfels {
        convert_surfels(&parsed);
        return;
    }

    let started = time::Instant::now();
    let model = match convert::convert_gltf(&parsed.input, &parsed.options) {
        Ok(model) => model,
        Err(convert::ConvertError::QhullUnavailable) => fail(
            "--topology qhull needs a build with the feature enabled:\n  \
             cargo build --release -p blade-volume-convert --features qhull",
        ),
        Err(ref err) => fail(&format!("conversion failed: {err:?}")),
    };
    let converted = started.elapsed();

    let save_options = convert::SaveOptions {
        format: parsed.format,
    };
    if let Err(ref err) = convert::save_ply_with_options(&parsed.output, &model, &save_options) {
        fail(&format!("save failed: {err:?}"));
    }

    println!("output: {}", parsed.output);
    println!("points: {}", model.len());
    println!("sh_degree: {}", model.sh_degree);
    println!("has_transforms: {}", model.transforms.is_some());
    println!("has_adjacency: {}", model.adjacency.is_some());
    println!("has_radii: {}", model.radii.is_some());
    if let Some(ref adjacency) = model.adjacency {
        println!("adjacency_edges: {}", adjacency.neighbors.len());
    }
    println!("seconds: {:.3}", converted.as_secs_f64());
}

/// Convert to relightable surfels rather than to a point cloud.
///
/// A separate path rather than another output kind, because it is a different
/// conversion and not a different way of writing the same one: what comes out
/// carries materials and normals and no radiance at all, so none of the
/// appearance options above apply to it.
fn convert_surfels(parsed: &ParsedArgs) {
    let started = time::Instant::now();
    let model =
        match convert::relight_model_from_gltf(path::Path::new(&parsed.input), &parsed.options) {
            Ok(model) => model,
            Err(ref err) => fail(&format!("conversion failed: {err:?}")),
        };
    let converted = started.elapsed();

    let output = path::Path::new(&parsed.output);
    if let Err(ref err) = vol::io::try_save_relight(output, &model) {
        fail(&format!("save failed: {err}"));
    }

    let bytes = std::fs::metadata(output)
        .map(|meta| meta.len())
        .unwrap_or(0);
    println!("output: {}", parsed.output);
    println!("surfels: {}", model.surfels.len());
    println!("materials: {}", model.materials.len());
    println!("megabytes: {:.1}", bytes as f64 / (1024.0 * 1024.0));
    if let Some((min, max)) = model.bounds() {
        println!("bounds: {min:?} to {max:?}");
    }
    println!("seconds: {:.3}", converted.as_secs_f64());
}

fn usage_text() -> String {
    let defaults = convert::ConvertOptions::default();
    format!(
        "usage: convert <input.gltf|glb> [options]

Offline conversion of a glTF asset into a point-sampled cloud. No rendering,
no training: geometry and materials are sampled directly.

Output:
  -o, --output PATH            output path (default: input with the extension
                               the kind implies)
  -k, --kind KIND              gaussian | radfoam | surfel (default: gaussian).
                               surfel writes relightable surfels to .surfel:
                               materials and normals, no baked radiance, so the
                               light is supplied at render time. None of the
                               appearance options below apply to it.
  -f, --format FORMAT          ascii | binary (default: binary)

Sampling rate (pick one; later flags win):
  -d, --density F              samples per cubic world unit (default: {density})
      --spacing S              grid spacing in world units (density = S^-3)
  -r, --resolution N           grid cells across the bounding-box diagonal.
                               Scale invariant: prefer this for arbitrary assets
                               whose units you do not control.

Sampling detail:
      --surface-density-scale F   (default: {surface_density_scale})
      --interior-density-scale F  (default: {interior_density_scale}; 0 disables interior fill)
      --curvature-boost F         bias samples toward creases (default: {curvature_boost}, off)
      --exterior-density-scale F  transparent fill for the space *outside* the
                                  mesh, relative to density. Default {exterior_density_scale} for
                                  radfoam, 0 for gaussian. Without it an
                                  object-centric foam cannot be viewed from
                                  outside at all: empty space belongs to
                                  unbounded cells owned by opaque surface sites.
      --exterior-padding F        how far that fill extends past the bounds, as
                                  a fraction of the diagonal (default: {exterior_padding})
      --interior-jitter F         displace interior samples by this fraction of
                                  the sub-cell spacing. Default {interior_jitter} for
                                  radfoam, 0 for gaussian: an exact lattice is
                                  degenerate for Delaunay, but gaussian output
                                  is never triangulated.
      --alpha-threshold F         drop samples below this coverage (default: {alpha_threshold})
      --seed U64                  sampling seed (default: {seed})

Appearance:
      --ambient F | R,G,B      ambient gain in linear light (default: {ambient})
      --surface-opacity F      (default: {surface_opacity})
      --interior-opacity F     (default: {interior_opacity})
      --surface-scale F        (default: {surface_scale})
      --interior-scale F       (default: {interior_scale})
      --surface-normal-scale F (default: {surface_normal_scale})

RadFoam / PowerFoam (--kind radfoam only):
  -t, --topology BACKEND       exact | qhull (default: exact). The pure-Rust
                               exact builder dominates runtime past ~100k
                               points; qhull needs --features qhull at build.
      --spring-iterations N    spring relaxation passes (default: {spring_iterations}, off)
      --spring-step F          relaxation step size (default: {spring_step})
      --radii                  assign nearest-neighbour radii; emits PowerFoam
                               with a rebuilt Cech adjacency
      --radius-factor F        radius multiplier (default: {radius_factor})

  -h, --help                   show this message",
        density = defaults.density,
        surface_density_scale = defaults.surface_density_scale,
        interior_density_scale = defaults.interior_density_scale,
        curvature_boost = defaults.curvature_boost,
        interior_jitter = convert::DEFAULT_INTERIOR_JITTER,
        exterior_density_scale = convert::DEFAULT_EXTERIOR_DENSITY_SCALE,
        exterior_padding = defaults.exterior_padding,
        alpha_threshold = defaults.alpha_threshold,
        seed = defaults.seed,
        ambient = defaults.ambient.x,
        surface_opacity = defaults.surface_opacity,
        interior_opacity = defaults.interior_opacity,
        surface_scale = defaults.surface_scale,
        interior_scale = defaults.interior_scale,
        surface_normal_scale = defaults.surface_normal_scale,
        spring_iterations = defaults.spring_iterations,
        spring_step = defaults.spring_step,
        radius_factor = defaults.radius_factor,
    )
}

fn usage(message: &str) -> ! {
    eprintln!("{message}");
    eprintln!("{}", usage_text());
    process::exit(1);
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    process::exit(1);
}

fn default_output_path(input: &str, extension: &str) -> String {
    if let Some(pos) = input.rfind('.') {
        let mut output = input.to_string();
        output.replace_range(pos + 1.., extension);
        output
    } else {
        format!("{input}.{extension}")
    }
}

struct ParsedArgs {
    input: String,
    output: String,
    format: convert::PlyFormat,
    /// Relightable surfels rather than a point cloud.
    surfels: bool,
    options: convert::ConvertOptions,
}

/// Parse the command line. `Ok(None)` means help was requested.
fn parse_args(args: &[String]) -> Result<Option<ParsedArgs>, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut format = convert::PlyFormat::Binary;
    let mut surfels = false;
    let mut options = convert::ConvertOptions::default();

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        // Options that consume a value share this lookup; flags ignore it.
        let value = || -> Result<&String, String> {
            args.get(i + 1)
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        let mut consumed = 2usize;
        match arg {
            "-h" | "--help" => return Ok(None),
            "-k" | "--kind" => {
                surfels = false;
                options.output = match value()?.as_str() {
                    "gaussian" => convert::OutputKind::Gaussian,
                    "radfoam" => convert::OutputKind::RadFoam,
                    "surfel" => {
                        surfels = true;
                        // Unused by the surfel path, and left at the default
                        // rather than given a meaning it does not have.
                        convert::OutputKind::Gaussian
                    }
                    other => return Err(format!("unknown output kind: {other}")),
                };
            }
            "-o" | "--output" => output = Some(value()?.clone()),
            "-f" | "--format" => {
                format = match value()?.as_str() {
                    "ascii" => convert::PlyFormat::Ascii,
                    "binary" => convert::PlyFormat::Binary,
                    other => return Err(format!("unknown format: {other}")),
                };
            }
            "-d" | "--density" => {
                options.density = parse_f32(arg, value()?)?;
                options.resolution = None;
            }
            "--spacing" => {
                let spacing = parse_f32(arg, value()?)?;
                if spacing <= 0.0 {
                    return Err("--spacing must be positive".to_string());
                }
                options.density = spacing.powi(-3);
                options.resolution = None;
            }
            "-r" | "--resolution" => {
                let resolution = parse_f32(arg, value()?)?;
                if resolution <= 0.0 {
                    return Err("--resolution must be positive".to_string());
                }
                options.resolution = Some(resolution);
            }
            "--surface-density-scale" => options.surface_density_scale = parse_f32(arg, value()?)?,
            "--interior-density-scale" => {
                options.interior_density_scale = parse_f32(arg, value()?)?
            }
            "--curvature-boost" => options.curvature_boost = parse_f32(arg, value()?)?,
            "--interior-jitter" => options.interior_jitter = Some(parse_f32(arg, value()?)?),
            "--exterior-density-scale" => {
                options.exterior_density_scale = Some(parse_f32(arg, value()?)?)
            }
            "--exterior-padding" => options.exterior_padding = parse_f32(arg, value()?)?,
            "--alpha-threshold" => options.alpha_threshold = parse_f32(arg, value()?)?,
            "--seed" => {
                options.seed = value()?
                    .parse::<u64>()
                    .map_err(|err| format!("invalid value for --seed: {err}"))?
            }
            "--ambient" => options.ambient = parse_vec3(arg, value()?)?,
            "--surface-opacity" => options.surface_opacity = parse_f32(arg, value()?)?,
            "--interior-opacity" => options.interior_opacity = parse_f32(arg, value()?)?,
            "--surface-scale" => options.surface_scale = parse_f32(arg, value()?)?,
            "--interior-scale" => options.interior_scale = parse_f32(arg, value()?)?,
            "--surface-normal-scale" => options.surface_normal_scale = parse_f32(arg, value()?)?,
            "--spring-iterations" => {
                options.spring_iterations = value()?
                    .parse::<usize>()
                    .map_err(|err| format!("invalid value for --spring-iterations: {err}"))?
            }
            "--spring-step" => options.spring_step = parse_f32(arg, value()?)?,
            "--radii" => {
                options.assign_radii = true;
                consumed = 1;
            }
            "--radius-factor" => options.radius_factor = parse_f32(arg, value()?)?,
            "-t" | "--topology" => {
                options.topology = match value()?.as_str() {
                    "exact" => convert::Topology::Exact,
                    "qhull" => convert::Topology::Qhull,
                    other => return Err(format!("unknown topology backend: {other}")),
                };
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option: {arg}")),
            _ => {
                if input.is_none() {
                    input = Some(arg.to_string());
                } else if output.is_none() {
                    output = Some(arg.to_string());
                } else {
                    return Err(format!("unexpected argument: {arg}"));
                }
                consumed = 1;
            }
        }
        i += consumed;
    }

    let input = input.ok_or("missing input gltf path")?;
    let extension = if surfels {
        vol::io::SURFEL_EXTENSION
    } else {
        "ply"
    };
    let output = output.unwrap_or_else(|| default_output_path(&input, extension));

    Ok(Some(ParsedArgs {
        input,
        output,
        format,
        surfels,
        options,
    }))
}

fn parse_f32(flag: &str, text: &str) -> Result<f32, String> {
    text.parse::<f32>()
        .map_err(|err| format!("invalid value for {flag}: {err}"))
}

/// Accept either a single scalar (broadcast to all channels) or `r,g,b`.
fn parse_vec3(flag: &str, text: &str) -> Result<glam::Vec3, String> {
    let parts = text.split(',').collect::<Vec<_>>();
    match parts.len() {
        1 => Ok(glam::Vec3::splat(parse_f32(flag, parts[0])?)),
        3 => Ok(glam::Vec3::new(
            parse_f32(flag, parts[0])?,
            parse_f32(flag, parts[1])?,
            parse_f32(flag, parts[2])?,
        )),
        _ => Err(format!("{flag} expects one value or three comma-separated")),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_args, parse_vec3};
    use blade_volume_convert as convert;

    fn parse(args: &[&str]) -> super::ParsedArgs {
        let owned = args.iter().map(|a| a.to_string()).collect::<Vec<_>>();
        parse_args(&owned).unwrap().unwrap()
    }

    #[test]
    fn defaults_match_the_library() {
        let parsed = parse(&["model.glb"]);
        let defaults = convert::ConvertOptions::default();
        assert_eq!(parsed.input, "model.glb");
        assert_eq!(parsed.output, "model.ply");
        assert_eq!(parsed.options.density, defaults.density);
        assert_eq!(parsed.options.output, defaults.output);
        assert!(parsed.options.resolution.is_none());
        assert!(!parsed.options.assign_radii);
    }

    #[test]
    fn help_is_reported_without_an_input() {
        let owned = vec!["--help".to_string()];
        assert!(parse_args(&owned).unwrap().is_none());
    }

    #[test]
    fn every_sampling_option_reaches_the_library() {
        let parsed = parse(&[
            "model.glb",
            "--kind",
            "radfoam",
            "--density",
            "42",
            "--surface-density-scale",
            "2",
            "--interior-density-scale",
            "0.5",
            "--curvature-boost",
            "1.5",
            "--alpha-threshold",
            "0.25",
            "--seed",
            "7",
            "--surface-opacity",
            "0.9",
            "--interior-opacity",
            "0.1",
            "--surface-scale",
            "1.25",
            "--interior-scale",
            "1.75",
            "--surface-normal-scale",
            "0.5",
            "--spring-iterations",
            "3",
            "--spring-step",
            "0.4",
            "--radii",
            "--radius-factor",
            "0.75",
        ]);
        let options = &parsed.options;
        assert_eq!(options.output, convert::OutputKind::RadFoam);
        assert_eq!(options.density, 42.0);
        assert_eq!(options.surface_density_scale, 2.0);
        assert_eq!(options.interior_density_scale, 0.5);
        assert_eq!(options.curvature_boost, 1.5);
        assert_eq!(options.alpha_threshold, 0.25);
        assert_eq!(options.seed, 7);
        assert_eq!(options.surface_opacity, 0.9);
        assert_eq!(options.interior_opacity, 0.1);
        assert_eq!(options.surface_scale, 1.25);
        assert_eq!(options.interior_scale, 1.75);
        assert_eq!(options.surface_normal_scale, 0.5);
        assert_eq!(options.spring_iterations, 3);
        assert_eq!(options.spring_step, 0.4);
        assert!(options.assign_radii);
        assert_eq!(options.radius_factor, 0.75);
    }

    #[test]
    fn the_surfel_kind_picks_its_own_path_and_extension() {
        let parsed = parse(&["model.glb", "--kind", "surfel"]);
        assert!(parsed.surfels);
        assert_eq!(parsed.output, "model.surfel");

        // Asking for a cloud after asking for surfels gets a cloud, so the
        // last flag decides here the way it does for the sampling rate.
        let parsed = parse(&["model.glb", "--kind", "surfel", "--kind", "radfoam"]);
        assert!(!parsed.surfels);
        assert_eq!(parsed.output, "model.ply");

        // An explicit output is never overridden by the kind.
        let parsed = parse(&["model.glb", "--kind", "surfel", "-o", "elsewhere.bin"]);
        assert_eq!(parsed.output, "elsewhere.bin");
    }

    #[test]
    fn spacing_is_the_inverse_cube_of_density() {
        let parsed = parse(&["model.glb", "--spacing", "0.5"]);
        assert_eq!(parsed.options.density, 8.0);
        assert!(parsed.options.resolution.is_none());
    }

    #[test]
    fn resolution_and_absolute_rates_override_each_other() {
        // Whichever flag comes last decides, so a scale-invariant request is
        // never silently mixed with an absolute one.
        let by_resolution = parse(&["model.glb", "--density", "5", "--resolution", "64"]);
        assert_eq!(by_resolution.options.resolution, Some(64.0));

        let by_density = parse(&["model.glb", "--resolution", "64", "--density", "5"]);
        assert!(by_density.options.resolution.is_none());
        assert_eq!(by_density.options.density, 5.0);
    }

    #[test]
    fn ambient_accepts_a_scalar_or_a_triple() {
        assert_eq!(
            parse_vec3("--ambient", "1.5").unwrap(),
            glam::Vec3::splat(1.5)
        );
        assert_eq!(
            parse_vec3("--ambient", "0.1,0.2,0.3").unwrap(),
            glam::Vec3::new(0.1, 0.2, 0.3)
        );
        assert!(parse_vec3("--ambient", "0.1,0.2").is_err());
    }

    #[test]
    fn invalid_input_is_rejected() {
        let cases: &[&[&str]] = &[
            &["model.glb", "--kind", "voxels"],
            &["model.glb", "--format", "obj"],
            &["model.glb", "--density"],
            &["model.glb", "--density", "abc"],
            &["model.glb", "--spacing", "0"],
            &["model.glb", "--resolution", "-1"],
            &["model.glb", "--nope"],
            &["a.glb", "b.ply", "c.ply"],
            &[],
        ];
        for case in cases {
            let owned = case.iter().map(|a| a.to_string()).collect::<Vec<_>>();
            assert!(parse_args(&owned).is_err(), "expected rejection: {case:?}");
        }
    }
}
