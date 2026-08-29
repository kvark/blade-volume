#!/bin/sh
set -eu

if [ "${1:-}" != "--accept-license" ]; then
  echo "usage: $0 --accept-license [OUTPUT_DIR]" >&2
  echo "LUCES-MV is restricted to non-commercial research use." >&2
  echo "Read the official licence before passing --accept-license:" >&2
  echo "https://drive.google.com/drive/folders/1634yweYUpLvNPC1qEG8hRpmhtxVtFrLi" >&2
  exit 2
fi

output=${2:-target/datasets/luces-mv}
source_dir="$output/source"
data_dir="$output/data"
mkdir -p "$source_dir" "$data_dir"

fetch() {
  id=$1
  name=$2
  expected=$3
  path="$source_dir/$name"
  if [ -f "$path" ] && printf '%s  %s\n' "$expected" "$path" | sha256sum --check --status; then
    echo "using verified $path"
    return
  fi
  rm -f "$path.part"
  curl --fail --location --retry 5 --output "$path.part" \
    "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
  printf '%s  %s\n' "$expected" "$path.part" | sha256sum --check --status
  mv "$path.part" "$path"
}

fetch 1cl5MWZS8jy2cveRbvirdSxxiNigILCqb Owl.zip \
  ced6a0fb5a6e8ac4fa447ebfcd965ee4c6a74e20fe61dbef4722a3db1942bc2f
fetch 1LffPMctk1QUGC5-bsGEcXs7EOvTAzbmd cam1_params.txt \
  3f9343f9eb9bbca0b84baa13f602490d13faccbf630f3264f32cb1c20ec9737c
fetch 1GzbsH_wVSFdaTm2M0z1KWMMzpIwPflLP cam2_params.txt \
  7bdf1b06f591d1baf053e7655a687ac185794262aee326b0b39e096a0f2e0cca
fetch 10nn-Louz9MpWxOHWLwPdC-H2JdK_oC8J ReadMe.txt \
  ad4ca4f40ed16570f304662300fb598316cd4d822167a326c620624b02c01031
fetch 10olvxoLFMGR-t7EcphidyCNGOfR2yUx2 licence.txt \
  85121a4af7bb1c59a37d4df0b12422447137cae669798c7c6017217569573a2e

if [ ! -d "$data_dir/Owl" ]; then
  unzip -q "$source_dir/Owl.zip" -d "$data_dir"
fi

echo "LUCES-MV Owl is ready under $data_dir/Owl"
