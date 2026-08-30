#!/bin/sh
set -eu

if [ "${1:-}" != "--accept-license" ]; then
  echo "usage: $0 --accept-license [OUTPUT_DIR]" >&2
  echo "Read the official DiLiGenT-MV terms before passing --accept-license:" >&2
  echo "https://sites.google.com/site/photometricstereodata/mv" >&2
  exit 2
fi

output=${2:-target/datasets/diligent-mv}
source_dir="$output/source"
data_dir="$output/data"
archive="$source_dir/DiLiGenT-MV.zip"
expected=c5f010f7ec502d7deb072f3e37fca1efade4019106af8203193846faf4d285a5
mkdir -p "$source_dir" "$data_dir"

if [ ! -f "$archive" ] || ! printf '%s  %s\n' "$expected" "$archive" | sha256sum --check --status; then
  rm -f "$archive.part"
  curl --fail --location --retry 5 --output "$archive.part" \
    'https://drive.usercontent.google.com/download?id=18dheWmAxCNaBpYoH3usuFeH9vGlhODvx&export=download&confirm=t'
  printf '%s  %s\n' "$expected" "$archive.part" | sha256sum --check --status
  mv "$archive.part" "$archive"
else
  echo "using verified $archive"
fi

object="$data_dir/DiLiGenT-MV/mvpmsData/bearPNG"
if [ ! -d "$object" ]; then
  unzip -q "$archive" 'DiLiGenT-MV/mvpmsData/bearPNG/*' -d "$data_dir"
fi

echo "DiLiGenT-MV Bear is ready under $object"
