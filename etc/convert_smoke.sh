#!/usr/bin/env bash
# End-to-end smoke test for the offline asset converter.
#
# The cargo tests exercise the library. This exercises the thing a user
# actually runs: the `convert` binary, its flags, and the PLY it leaves on
# disk. It is CPU-only and takes a few seconds, so it is safe in CI.
#
# Usage: etc/convert_smoke.sh [path/to/convert]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${1:-$ROOT/target/release/convert}
ASSET=$ROOT/blade-volume-test/data/police.glb
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$BIN" ]; then
    echo "convert binary not found at $BIN" >&2
    echo "build it with: cargo build --release -p blade-volume-convert" >&2
    exit 2
fi
if [ ! -f "$ASSET" ]; then
    echo "test asset not found at $ASSET" >&2
    exit 2
fi

fail() { echo "FAIL: $*" >&2; exit 1; }

# `convert` prints `key: value` lines; pull one out.
field() { grep -E "^$2: " "$1" | head -1 | cut -d' ' -f2-; }

echo "== help =="
"$BIN" --help > "$WORK/help.txt" || fail "--help exited non-zero"
for flag in --resolution --density --topology --radii --curvature-boost --interior-jitter; do
    grep -q -- "$flag" "$WORK/help.txt" || fail "help does not document $flag"
done
echo "ok"

echo "== radfoam conversion =="
"$BIN" "$ASSET" -k radfoam -r 24 -o "$WORK/foam.ply" > "$WORK/foam.txt" || fail "radfoam conversion failed"
cat "$WORK/foam.txt"
[ -s "$WORK/foam.ply" ] || fail "no PLY written"
points=$(field "$WORK/foam.txt" points)
edges=$(field "$WORK/foam.txt" adjacency_edges)
[ "$points" -gt 1000 ] || fail "suspiciously few points: $points"
[ "$edges" -gt "$points" ] || fail "fewer adjacency edges ($edges) than points ($points)"
[ "$(field "$WORK/foam.txt" has_adjacency)" = "true" ] || fail "radfoam output lacks adjacency"
[ "$(field "$WORK/foam.txt" has_transforms)" = "false" ] || fail "radfoam output should not be Gaussian"

echo "== gaussian conversion =="
"$BIN" "$ASSET" -k gaussian -r 24 -o "$WORK/gauss.ply" > "$WORK/gauss.txt" || fail "gaussian conversion failed"
[ -s "$WORK/gauss.ply" ] || fail "no Gaussian PLY written"
[ "$(field "$WORK/gauss.txt" has_transforms)" = "true" ] || fail "gaussian output lacks transforms"
[ "$(field "$WORK/gauss.txt" has_adjacency)" = "false" ] || fail "gaussian output should not carry adjacency"

echo "== power foam (radii) =="
"$BIN" "$ASSET" -k radfoam -r 24 --radii -o "$WORK/power.ply" > "$WORK/power.txt" || fail "power foam conversion failed"
[ "$(field "$WORK/power.txt" has_radii)" = "true" ] || fail "--radii did not emit radii"

echo "== determinism =="
# A pipeline that is not byte-reproducible cannot be cached or diffed.
"$BIN" "$ASSET" -k radfoam -r 24 -o "$WORK/again.ply" > /dev/null || fail "second conversion failed"
cmp -s "$WORK/foam.ply" "$WORK/again.ply" || fail "same inputs produced different bytes"
echo "ok"

echo "== seed changes the sampling =="
"$BIN" "$ASSET" -k radfoam -r 24 --seed 99 -o "$WORK/seeded.ply" > /dev/null || fail "seeded conversion failed"
cmp -s "$WORK/foam.ply" "$WORK/seeded.ply" && fail "--seed did not change the output"
echo "ok"

echo "== resolution is monotone =="
"$BIN" "$ASSET" -k gaussian -r 32 -o "$WORK/fine.ply" > "$WORK/fine.txt" || fail "fine conversion failed"
coarse_points=$(field "$WORK/gauss.txt" points)
fine_points=$(field "$WORK/fine.txt" points)
[ "$fine_points" -gt "$coarse_points" ] || fail "raising resolution did not add points ($coarse_points -> $fine_points)"
echo "ok ($coarse_points -> $fine_points)"

echo "== bad input is rejected =="
"$BIN" "$ASSET" --kind voxels -o "$WORK/x.ply" > /dev/null 2>&1 && fail "unknown kind was accepted"
"$BIN" missing.glb -o "$WORK/x.ply" > /dev/null 2>&1 && fail "missing asset was accepted"
"$BIN" "$ASSET" --density 0 -o "$WORK/x.ply" > /dev/null 2>&1 && fail "zero density was accepted"
echo "ok"

# Only meaningful when built with the feature; without it the binary must say
# so clearly instead of failing obscurely.
echo "== qhull topology =="
if "$BIN" "$ASSET" -k radfoam -r 24 -t qhull -o "$WORK/qh.ply" > "$WORK/qh.txt" 2>&1; then
    qh_points=$(field "$WORK/qh.txt" points)
    [ "$qh_points" = "$points" ] || fail "qhull changed the site count ($points -> $qh_points)"
    echo "ok (built with qhull, $qh_points points)"
else
    grep -q "features qhull" "$WORK/qh.txt" || fail "unhelpful error for a build without qhull"
    echo "ok (no qhull feature; error message points at the fix)"
fi

echo
echo "convert smoke: all checks passed"
