#!/usr/bin/env bash
# Pull a small COLMAP-format scene from Hugging Face into etc/data/<scene>/.
# Nothing under etc/data/ is checked in — see .gitignore.
#
# Usage:
#   etc/fetch_test_dataset.sh                    # defaults to "bonsai"
#   etc/fetch_test_dataset.sh bonsai             # yuangjia/mipnerf-bonsai
#   etc/fetch_test_dataset.sh glomap-example     # pablovela5620/example-colmap-glomap
#
# Each scene lands in etc/data/<scene>/ with the canonical COLMAP layout:
#   images/
#   sparse/0/{cameras,images,points3D}.bin
#
# Requires `git` and Hugging Face's `git lfs`. Pulls only the directories we
# actually use (no model checkpoints, no language-feature side-cars).

set -euo pipefail

SCENE="${1:-bonsai}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/data/$SCENE"

case "$SCENE" in
  bonsai)
    REPO="https://huggingface.co/datasets/yuangjia/mipnerf-bonsai"
    PATHS=("images" "sparse")
    ;;
  glomap-example)
    REPO="https://huggingface.co/datasets/pablovela5620/example-colmap-glomap"
    PATHS=("images" "colmap")
    ;;
  *)
    echo "unknown scene: $SCENE" >&2
    echo "known: bonsai | glomap-example" >&2
    exit 2
    ;;
esac

if [ -d "$DEST/.git" ]; then
  echo "[fetch] $SCENE already cloned at $DEST — pulling latest"
  git -C "$DEST" pull --ff-only
  exit 0
fi

mkdir -p "$DEST"
echo "[fetch] cloning $REPO into $DEST (sparse checkout of: ${PATHS[*]})"

# Sparse checkout so we don't pull stuff we don't use.
git clone --filter=blob:none --no-checkout "$REPO" "$DEST"
git -C "$DEST" sparse-checkout init --no-cone
git -C "$DEST" sparse-checkout set "${PATHS[@]}"
git -C "$DEST" checkout main

echo "[fetch] done. Layout under $DEST:"
ls "$DEST"
