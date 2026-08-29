#!/bin/bash
# Export a curated bundle for one training run: weights + training curves +
# DUO eval plots + detection images, collected out of runs/ into a single
# self-contained directory.
#
# runs/ is the source of truth for trained models; this produces a portable
# snapshot of one run for presentation or archival. Safe to re-run.
#
# Usage:
#   tools/export_run_bundle.sh <run_name> [dest_dir] [repo_root]
#
#   run_name   name under runs/train/, e.g. v9_flux2dev_duo_dr_v3
#   dest_dir   output dir (default: results/yolo26x/<run_name>)
#   repo_root  repository root (default: this script's parent directory)
#
# Examples:
#   tools/export_run_bundle.sh v9_flux2dev_duo_dr_v3
#   tools/export_run_bundle.sh v9_flux2dev results/yolo26x/v1
#   tools/export_run_bundle.sh v9_flux2dev "" /workspace/ImgGen_AI

set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    exit 1
fi

RUN_NAME="$1"
DEST="${2:-}"
REPO_ROOT="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$REPO_ROOT"
DEST="${DEST:-results/yolo26x/$RUN_NAME}"

if [ ! -d "runs/train/$RUN_NAME" ]; then
    echo "error: runs/train/$RUN_NAME not found under $REPO_ROOT" >&2
    echo "available runs:" >&2
    ls -1 runs/train 2>/dev/null | grep -v '\.json$' | sed 's/^/  /' >&2
    exit 1
fi

mkdir -p "$DEST/weights" "$DEST/training" "$DEST/eval_duo_test" "$DEST/detections_duo_test"

# weights are the only required artefacts; everything else is best-effort
cp "runs/train/$RUN_NAME/weights/best.pt" "$DEST/weights/best.pt"
cp "runs/train/$RUN_NAME/weights/last.pt" "$DEST/weights/last.pt"

for f in results.png results.csv confusion_matrix.png confusion_matrix_normalized.png labels.jpg args.yaml; do
    cp "runs/train/$RUN_NAME/$f" "$DEST/training/" 2>/dev/null || true
done
cp runs/train/"$RUN_NAME"/Box*.png "$DEST/training/" 2>/dev/null || true

for f in confusion_matrix.png confusion_matrix_normalized.png; do
    cp "runs/eval_duo/$RUN_NAME/$f" "$DEST/eval_duo_test/" 2>/dev/null || true
done
cp runs/eval_duo/"$RUN_NAME"/Box*.png "$DEST/eval_duo_test/" 2>/dev/null || true

cp -r runs/predict_duo/"$RUN_NAME"_detections/* "$DEST/detections_duo_test/" 2>/dev/null || true

echo "$RUN_NAME -> $DEST  ($(du -sh "$DEST" | cut -f1))"
