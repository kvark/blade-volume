#!/usr/bin/env bash
# Selectively fetch one OpenIllumination OLAT object. The complete dataset is
# about 900 GB; this pulls only calibrated poses, masks, and four 22-view lights.
#
# Usage:
#   etc/fetch_openillumination.sh
#   etc/fetch_openillumination.sh obj_16_friends_cup 000 062 082 092

set -euo pipefail

OBJECT="${1:-obj_16_friends_cup}"
if [ "$#" -gt 0 ]; then
  shift
fi
if [ "$#" -eq 0 ]; then
  LIGHTS=(000 062 082 092)
else
  LIGHTS=("$@")
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/data/openillumination"
OBJECT_PATH="OLAT/$OBJECT"
REPO="OpenIllumination/OpenIllumination"
REVISION="006beced878d3fb7187ded8e20ee63d9c5020fcb"
CODE_REVISION="868b831cd3477cce4c0e4d6a5833110f4fd3b7f3"

if ! command -v jq >/dev/null 2>&1; then
  echo "[fetch] jq is required to select the calibrated camera names" >&2
  exit 3
fi

mkdir -p "$DEST"
download_file() {
  local relative="$1"
  local destination="$DEST/$relative"
  local temporary
  if [ -f "$destination" ]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"
  temporary="$destination.partial.$$"
  curl -LfsS \
    "https://huggingface.co/datasets/$REPO/resolve/$REVISION/$relative?download=true" \
    -o "$temporary"
  mv "$temporary" "$destination"
}

echo "[fetch] camera calibration for $OBJECT"
download_file "$OBJECT_PATH/output/transforms_train.json"
download_file "$OBJECT_PATH/output/transforms_test.json"

mapfile -t CAMERAS < <(
  jq -r '.frames | keys[]' \
    "$DEST/$OBJECT_PATH/output/transforms_train.json" \
    "$DEST/$OBJECT_PATH/output/transforms_test.json" | sort -u
)

FILES=()
for camera in "${CAMERAS[@]}"; do
  FILES+=("$OBJECT_PATH/output/obj_masks/$camera.png")
  for light in "${LIGHTS[@]}"; do
    FILES+=("$OBJECT_PATH/Lights/$light/raw_undistorted/$camera.jpg")
  done
done

echo "[fetch] ${#CAMERAS[@]} cameras under ${#LIGHTS[@]} OLATs: ${LIGHTS[*]}"
for file in "${FILES[@]}"; do
  download_file "$file" &
  while [ "$(jobs -pr | wc -l)" -ge 8 ]; do
    wait -n
  done
done
wait

if [ ! -f "$DEST/light_pos.npy" ]; then
  TEMP="$(mktemp "$DEST/light_pos.npy.XXXXXX")"
  trap 'rm -f "$TEMP"' EXIT
  curl -LfsS \
    "https://raw.githubusercontent.com/oppo-us-research/OpenIlluminationCapture/$CODE_REVISION/tools/ps_recon/light_pos.npy" \
    -o "$TEMP"
  mv "$TEMP" "$DEST/light_pos.npy"
  trap - EXIT
fi

echo "[fetch] downloaded to $DEST/$OBJECT_PATH"
echo "[fetch] prepare with:"
printf 'cargo run -p blade-volume-train --bin import_openillumination -- --input %q --output %q --light-positions %q' \
  "$DEST/$OBJECT_PATH" "$DEST/prepared/$OBJECT" "$DEST/light_pos.npy"
for light in "${LIGHTS[@]}"; do
  printf ' --light %q' "$light"
done
printf '\n'
