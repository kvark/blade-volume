use blade_volume_convert as convert;
use std::{env, process};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let parsed = parse_args(&args).unwrap_or_else(|err| usage(&err));

    let kind = match parsed.kind.as_str() {
        "gaussian" => convert::OutputKind::Gaussian,
        "radfoam" => convert::OutputKind::RadFoam,
        other => usage(&format!("unknown output kind: {other}")),
    };

    let options = convert::ConvertOptions {
        output: kind,
        ..Default::default()
    };

    let model = match convert::convert_gltf(&parsed.input, &options) {
        Ok(model) => model,
        Err(err) => usage(&format!("conversion failed: {err:?}")),
    };

    let save_options = convert::SaveOptions {
        format: parsed.format,
    };
    if let Err(err) = convert::save_ply_with_options(&parsed.output, &model, &save_options) {
        usage(&format!("save failed: {err:?}"));
    }

    println!("output: {}", parsed.output);
    println!("points: {}", model.len());
    println!("sh_degree: {}", model.sh_degree);
    println!("has_transforms: {}", model.transforms.is_some());
    println!("has_adjacency: {}", model.adjacency.is_some());
}

fn usage(message: &str) -> ! {
    eprintln!("{message}");
    eprintln!(
        "usage: cargo run -p blade-volume-convert -- <input.gltf/glb> [-k|--kind gaussian|radfoam] [-o|--output output.ply] [-f|--format ascii|binary]"
    );
    process::exit(1);
}

fn default_output_path(input: &str) -> String {
    if let Some(pos) = input.rfind('.') {
        let mut output = input.to_string();
        output.replace_range(pos + 1.., "ply");
        output
    } else {
        format!("{input}.ply")
    }
}

struct ParsedArgs {
    input: String,
    output: String,
    kind: String,
    format: convert::PlyFormat,
}

fn parse_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut kind = "gaussian".to_string();
    let mut format = convert::PlyFormat::Binary;

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-k" | "--kind" => {
                let value = args.get(i + 1).ok_or("missing value for --kind")?;
                kind = value.clone();
                i += 2;
            }
            "-o" | "--output" => {
                let value = args.get(i + 1).ok_or("missing value for --output")?;
                output = Some(value.clone());
                i += 2;
            }
            "-f" | "--format" => {
                let value = args.get(i + 1).ok_or("missing value for --format")?;
                format = match value.as_str() {
                    "ascii" => convert::PlyFormat::Ascii,
                    "binary" => convert::PlyFormat::Binary,
                    other => return Err(format!("unknown format: {other}")),
                };
                i += 2;
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option: {arg}"));
            }
            _ => {
                if input.is_none() {
                    input = Some(arg.to_string());
                } else if output.is_none() {
                    output = Some(arg.to_string());
                } else {
                    return Err(format!("unexpected argument: {arg}"));
                }
                i += 1;
            }
        }
    }

    let input = input.ok_or("missing input gltf path")?;
    let output = output.unwrap_or_else(|| default_output_path(&input));

    Ok(ParsedArgs {
        input,
        output,
        kind,
        format,
    })
}
