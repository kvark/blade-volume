#!/usr/bin/env bash
# Selectively fetch one OpenIllumination object. The complete dataset is about
# 900 GB; this pulls only calibrated poses, masks, and the requested lights.
#
# Usage:
#   etc/fetch_openillumination.sh
#   etc/fetch_openillumination.sh obj_16_friends_cup 000 062 082 092
#   etc/fetch_openillumination.sh --capture lighting_patterns OBJECT LABEL=LED,...

set -euo pipefail

CAPTURE="OLAT"
if [ "${1:-}" = "--capture" ]; then
  if [ "$#" -lt 2 ]; then
    echo "--capture requires a dataset directory" >&2
    exit 2
  fi
  CAPTURE="$2"
  shift 2
fi
case "$CAPTURE" in
  OLAT) DEFAULT_OBJECT="obj_16_friends_cup"; IMAGE_EXTENSION="jpg" ;;
  lighting_patterns) DEFAULT_OBJECT="obj_63-fabric-friends-cup"; IMAGE_EXTENSION="JPG" ;;
  *) echo "unsupported capture directory: $CAPTURE" >&2; exit 2 ;;
esac

OBJECT="${1:-$DEFAULT_OBJECT}"
if [ "$#" -gt 0 ]; then
  shift
fi
if [ "$#" -eq 0 ]; then
  if [ "$CAPTURE" = "OLAT" ]; then
    LIGHTS=(000 062 082 092)
  else
    # The first six published patterns are disjoint 23-LED spatial groups.
    # Pattern 013 is the broad all-LED geometry light.
    LIGHTS=(
      "001=0,1,2,8,9,10,19,20,21,22,35,36,37,38,39,51,52,53,54,55,56,71,75"
      "002=3,4,7,11,12,13,23,24,25,26,40,41,42,43,57,58,59,60,72,73,74,76,77"
      "003=5,14,15,16,27,28,29,30,44,45,46,47,61,62,63,64,65,68,69,70,78,79,80"
      "004=6,17,18,31,32,33,34,48,49,50,66,67,81,82,83,88,106,107,108,123,124,125,134"
      "005=84,85,86,87,89,90,91,92,93,104,105,109,110,111,112,113,126,127,129,133,135,136,137"
      "006=94,95,96,97,98,101,102,103,114,115,116,117,118,119,120,121,128,130,131,132,138,139,141"
      "013=all"
    )
  fi
else
  LIGHTS=("$@")
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/data/openillumination"
OBJECT_PATH="$CAPTURE/$OBJECT"
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
    label="${light%%=*}"
    FILES+=("$OBJECT_PATH/Lights/$label/raw_undistorted/$camera.$IMAGE_EXTENSION")
  done
done

echo "[fetch] ${#CAMERAS[@]} cameras under ${#LIGHTS[@]} lights: ${LIGHTS[*]}"
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
