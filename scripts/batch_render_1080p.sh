#!/bin/bash
# Batch render 1080p + 224p videos for completed episodes.
# Uses EGL (minimal GPU memory). Low CPU priority.
# Usage: bash scripts/batch_render_1080p.sh [episodes_dir] [max_workers]

set -euo pipefail

EP_DIR="${1:-runs/oracle_staged_full_sweep_20260601/episodes}"
MAX_WORKERS="${2:-4}"
LOGFILE="${EP_DIR}/../render_1080p_log.txt"
RENDER_SCRIPT="scripts/render_1080p.py"

echo "=== Batch render started at $(date), workers=$MAX_WORKERS ===" >> "$LOGFILE"

# Build list of episodes needing rendering
TO_RENDER=()
for ep in "$EP_DIR"/*/; do
    # Skip if both already exist
    [ -f "$ep/top_rgb_1080p.mp4" ] && [ -f "$ep/top_rgb_224.mp4" ] && continue
    # Skip if no replay
    [ ! -f "$ep/replay.npz" ] && continue
    TO_RENDER+=("$ep")
done

TOTAL=${#TO_RENDER[@]}
echo "  Episodes to render: $TOTAL" >> "$LOGFILE"
echo "  Episodes to render: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
    echo "  Nothing to render." >> "$LOGFILE"
    exit 0
fi

# Render function
render_one() {
    local ep="$1"
    local ep_name
    ep_name=$(basename "$ep")
    MUJOCO_GL=egl PYTHONPATH=. nice -n 10 \
        /home/brucewu/miniconda3/envs/lerobot/bin/python \
        "$RENDER_SCRIPT" --episode_dir "$ep" 2>&1
}
export -f render_one
export RENDER_SCRIPT

# Run in parallel
done_count=0
fail_count=0

for ep in "${TO_RENDER[@]}"; do
    ep_name=$(basename "$ep")
    
    # Run in background (respect MAX_WORKERS)
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]; do
        sleep 2
    done
    
    (
        if render_one "$ep" >> "$LOGFILE" 2>&1; then
            echo "[$(date)] OK $ep_name" >> "$LOGFILE"
        else
            echo "[$(date)] FAIL $ep_name" >> "$LOGFILE"
        fi
    ) &
done

# Wait for all background jobs
wait

echo "=== Batch render done at $(date) ===" >> "$LOGFILE"
echo "Done. Check $LOGFILE for details."
