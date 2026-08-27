#!/usr/bin/env bash
# Extract a phone video and reconstruct the canonical COLMAP layout.
#
# Usage:
#   etc/colmap.sh [--dense | --dense-images PATH] VIDEO OUTPUT [FRAMES_PER_SECOND]

set -euo pipefail

usage() {
  echo "usage: $0 [--dense | --dense-images PATH] VIDEO OUTPUT [FRAMES_PER_SECOND]"
  echo "writes OUTPUT/images and OUTPUT/sparse/0; --dense also writes OUTPUT/dense/fused.ply"
  echo "--dense-images uses same-pose broad-light images for dense stereo"
  echo "default frame rate is 3"
}

DENSE=0
DENSE_IMAGES=""
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
elif [ "${1:-}" = "--dense" ]; then
  DENSE=1
  shift
elif [ "${1:-}" = "--dense-images" ]; then
  if [ "$#" -lt 2 ]; then
    echo "[colmap] --dense-images needs a directory" >&2
    exit 2
  fi
  DENSE=1
  DENSE_IMAGES="${2%/}"
  shift 2
fi
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  usage >&2
  exit 2
fi

VIDEO="$1"
OUTPUT="${2%/}"
FPS="${3:-3}"
FFMPEG="${FFMPEG:-ffmpeg}"
COLMAP="${COLMAP:-colmap}"
DENSE_MAX_IMAGE_SIZE="${COLMAP_DENSE_MAX_IMAGE_SIZE:-2000}"

if [ ! -f "$VIDEO" ]; then
  echo "[colmap] video does not exist: $VIDEO" >&2
  exit 2
fi
if [ -n "$DENSE_IMAGES" ] && [ ! -d "$DENSE_IMAGES" ]; then
  echo "[colmap] dense image directory does not exist: $DENSE_IMAGES" >&2
  exit 2
fi
if [ -z "$OUTPUT" ]; then
  echo "[colmap] output must name a new directory" >&2
  exit 2
fi
if ! [[ "$FPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[colmap] frames per second must be a positive integer: $FPS" >&2
  exit 2
fi
if [ "$DENSE" -eq 1 ] && ! [[ "$DENSE_MAX_IMAGE_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[colmap] dense maximum image size must be a positive integer: $DENSE_MAX_IMAGE_SIZE" >&2
  exit 2
fi
if [ -e "$OUTPUT" ]; then
  echo "[colmap] refusing to replace existing path: $OUTPUT" >&2
  exit 2
fi
if ! command -v "$FFMPEG" >/dev/null 2>&1; then
  echo "[colmap] executable not found: $FFMPEG" >&2
  exit 3
fi
if ! command -v "$COLMAP" >/dev/null 2>&1; then
  echo "[colmap] executable not found: $COLMAP" >&2
  exit 3
fi

PARENT="$(dirname -- "$OUTPUT")"
NAME="$(basename -- "$OUTPUT")"
mkdir -p -- "$PARENT"
WORK="$(mktemp -d -- "$PARENT/.${NAME}.incomplete.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
mkdir -p -- "$WORK/images" "$WORK/sparse"

echo "[colmap] extracting ${FPS} frame(s)/s from $VIDEO"
"$FFMPEG" -nostdin -hide_banner -loglevel warning \
  -i "$VIDEO" -map 0:v:0 -vf "fps=$FPS" \
  "$WORK/images/frame_%06d.png"

shopt -s nullglob
FRAMES=("$WORK"/images/*.png)
if [ "${#FRAMES[@]}" -lt 3 ]; then
  echo "[colmap] need at least three extracted frames, found ${#FRAMES[@]}" >&2
  exit 4
fi
if [ -n "$DENSE_IMAGES" ]; then
  for FRAME in "${FRAMES[@]}"; do
    FRAME_NAME="$(basename -- "$FRAME")"
    if [ ! -f "$DENSE_IMAGES/$FRAME_NAME" ]; then
      echo "[colmap] dense image set is missing $FRAME_NAME" >&2
      exit 4
    fi
  done
fi

echo "[colmap] extracting features from ${#FRAMES[@]} frames"
"$COLMAP" feature_extractor \
  --database_path "$WORK/database.db" \
  --image_path "$WORK/images" \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.single_camera 1

echo "[colmap] matching neighbouring video frames"
"$COLMAP" sequential_matcher \
  --database_path "$WORK/database.db"

echo "[colmap] mapping sparse points and camera poses"
"$COLMAP" mapper \
  --database_path "$WORK/database.db" \
  --image_path "$WORK/images" \
  --output_path "$WORK/sparse"

for FILE in cameras.bin images.bin points3D.bin; do
  if [ ! -f "$WORK/sparse/0/$FILE" ]; then
    echo "[colmap] reconstruction did not produce sparse/0/$FILE" >&2
    exit 4
  fi
done

if [ "$DENSE" -eq 1 ]; then
  DENSE_IMAGE_PATH="${DENSE_IMAGES:-$WORK/images}"
  echo "[colmap] preparing images for dense stereo at up to ${DENSE_MAX_IMAGE_SIZE}px"
  "$COLMAP" image_undistorter \
    --image_path "$DENSE_IMAGE_PATH" \
    --input_path "$WORK/sparse/0" \
    --output_path "$WORK/dense" \
    --output_type COLMAP \
    --max_image_size "$DENSE_MAX_IMAGE_SIZE"

  echo "[colmap] matching dense multi-view correspondences"
  "$COLMAP" patch_match_stereo \
    --workspace_path "$WORK/dense" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

  echo "[colmap] fusing the oriented point cloud"
  "$COLMAP" stereo_fusion \
    --workspace_path "$WORK/dense" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$WORK/dense/fused.ply"
  if [ ! -s "$WORK/dense/fused.ply" ]; then
    echo "[colmap] dense reconstruction did not produce dense/fused.ply" >&2
    exit 4
  fi
fi

mv -- "$WORK" "$OUTPUT"
trap - EXIT
if [ "$DENSE" -eq 1 ]; then
  echo "[colmap] ready: $OUTPUT/images, $OUTPUT/sparse/0, and $OUTPUT/dense/fused.ply"
else
  echo "[colmap] ready: $OUTPUT/images and $OUTPUT/sparse/0"
fi
