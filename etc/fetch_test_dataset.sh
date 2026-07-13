#!/usr/bin/env bash
# Pull a small COLMAP-format scene from Hugging Face into etc/data/<scene>/.
# Nothing under etc/data/ is checked in — see .gitignore.
#
# Usage:
#   etc/fetch_test_dataset.sh                    # defaults to "bonsai"
#   etc/fetch_test_dataset.sh bonsai             # yuangjia/mipnerf-bonsai
#   etc/fetch_test_dataset.sh bonsai-full        # nvs-bench/mipnerf360
#   etc/fetch_test_dataset.sh glomap-example     # pablovela5620/example-colmap-glomap
#
# Each scene lands in etc/data/<scene>/ with the canonical COLMAP layout:
#   images/
#   sparse/0/{cameras,images,points3D}.bin
#
# The small fixtures require `git` and Hugging Face's `git lfs` support. The
# pinned full Bonsai scene uses the `hf` CLI directly. Both paths pull only the
# directories we actually use.

set -euo pipefail

SCENE="${1:-bonsai}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/data/$SCENE"
SOURCE_SUBDIR=""
REVISION="main"
DOWNLOAD_MODE="git"
REPO_ID=""

case "$SCENE" in
  bonsai)
    REPO="https://huggingface.co/datasets/yuangjia/mipnerf-bonsai"
    PATHS=("images" "sparse")
    ;;
  bonsai-full)
    REPO="https://huggingface.co/datasets/nvs-bench/mipnerf360"
    REPO_ID="nvs-bench/mipnerf360"
    PATHS=("bonsai")
    SOURCE_SUBDIR="bonsai"
    REVISION="2e0758f7f5d2bf82d5a29c795d3cbfb64af35474"
    DOWNLOAD_MODE="hf"
    ;;
  glomap-example)
    REPO="https://huggingface.co/datasets/pablovela5620/example-colmap-glomap"
    PATHS=("images" "colmap")
    ;;
  *)
    echo "unknown scene: $SCENE" >&2
    echo "known: bonsai | bonsai-full | glomap-example" >&2
    exit 2
    ;;
esac

normalize_layout() {
  local name source_path link_path
  if [ -z "$SOURCE_SUBDIR" ]; then
    return
  fi
  for name in images sparse; do
    source_path="$DEST/$SOURCE_SUBDIR/$name"
    link_path="$DEST/$name"
    if [ ! -d "$source_path" ]; then
      echo "[fetch] expected directory is missing: $source_path" >&2
      exit 3
    fi
    if [ -e "$link_path" ] && [ ! -L "$link_path" ]; then
      echo "[fetch] refusing to replace non-symlink path: $link_path" >&2
      exit 3
    fi
    rm -f "$link_path"
    ln -s "$SOURCE_SUBDIR/$name" "$link_path"
  done
}

if [ "$DOWNLOAD_MODE" = "hf" ]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "[fetch] the Hugging Face 'hf' CLI is required for $SCENE" >&2
    exit 3
  fi
  mkdir -p "$DEST"
  echo "[fetch] downloading pinned $REPO_ID@$REVISION into $DEST"
  hf download "$REPO_ID" \
    --repo-type dataset \
    --revision "$REVISION" \
    --include "$SOURCE_SUBDIR/**" \
    --local-dir "$DEST"
  normalize_layout
  echo "[fetch] done. Layout under $DEST:"
  ls "$DEST"
  exit 0
fi

if [ -d "$DEST/.git" ]; then
  if [ "$REVISION" = "main" ]; then
    echo "[fetch] $SCENE already cloned at $DEST — pulling latest"
    git -C "$DEST" pull --ff-only
  else
    echo "[fetch] $SCENE already cloned at $DEST — restoring pinned revision $REVISION"
    git -C "$DEST" fetch origin "$REVISION"
    git -C "$DEST" checkout --detach "$REVISION"
  fi
  normalize_layout
  exit 0
fi

mkdir -p "$DEST"
echo "[fetch] cloning $REPO into $DEST (sparse checkout of: ${PATHS[*]})"

# Sparse checkout so we don't pull stuff we don't use.
git clone --filter=blob:none --no-checkout "$REPO" "$DEST"
git -C "$DEST" sparse-checkout init --no-cone
git -C "$DEST" sparse-checkout set "${PATHS[@]}"
git -C "$DEST" checkout "$REVISION"
normalize_layout

echo "[fetch] done. Layout under $DEST:"
ls "$DEST"
