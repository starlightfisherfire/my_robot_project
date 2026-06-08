#!/bin/bash
# Batch render all npz episodes to 224×224 frames
# Usage: bash scripts/batch_render_npz.sh

set -e

NPZ_DIR="data/sim/mppi_stage2c/episodes"
TEMPLATES="data/sim/metadata/reset_templates_v0.json"
OUT_DIR="runs/lewn_dataset/frames"
SCRIPT="scripts/render_npz_to_frames.py"
PYTHON="/home/brucewu/miniconda3/envs/lerobot/bin/python"

mkdir -p "$OUT_DIR"
mkdir -p runs/lewn_dataset/logs

TOTAL=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
echo "=== Batch rendering $TOTAL episodes ==="
echo "Output: $OUT_DIR"
echo "Logs: runs/lewn_dataset/logs/"
echo ""

COUNT=0
FAILED=0
START_TIME=$(date +%s)

for npz in "$NPZ_DIR"/*.npz; do
    eid=$(basename "$npz" .npz)
    log_file="runs/lewn_dataset/logs/${eid}.log"
    
    if MUJOCO_GL=egl PYTHONPATH=. "$PYTHON" "$SCRIPT" \
        --npz "$npz" \
        --templates "$TEMPLATES" \
        --out "$OUT_DIR/$eid" \
        --width 224 --height 224 \
        > "$log_file" 2>&1; then
        COUNT=$((COUNT + 1))
        # Print progress every 50
        if [ $((COUNT % 50)) -eq 0 ]; then
            ELAPSED=$(($(date +%s) - START_TIME))
            echo "  [$COUNT/$TOTAL] ${ELAPSED}s elapsed"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  FAILED: $eid (see $log_file)"
    fi
done

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "=== Done ==="
echo "  Rendered: $COUNT / $TOTAL"
echo "  Failed: $FAILED"
echo "  Time: ${ELAPSED}s ($(echo "scale=1; $ELAPSED / 60" | bc)min)"
echo "  Output: $OUT_DIR"
