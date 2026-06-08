#!/bin/bash
# Background render: render 224x224 MP4 for completed episodes.
# Runs with low CPU priority to not interfere with sweep.
# Usage: bash scripts/render_completed_episodes.sh [episodes_dir]

EP_DIR="${1:-runs/oracle_staged_full_sweep_20260601/episodes}"
LOGFILE="${EP_DIR}/../render_log.txt"

echo "Render started at $(date)" >> "$LOGFILE"

rendered=0
skipped=0

for ep in "$EP_DIR"/*/; do
    ep_name=$(basename "$ep")

    # Skip if already rendered
    if [ -f "$ep/top_rgb_224.mp4" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Skip if no replay
    if [ ! -f "$ep/replay.npz" ]; then
        echo "SKIP $ep_name: no replay.npz" >> "$LOGFILE"
        continue
    fi

    # Render with low priority
    echo "Rendering $ep_name..." >> "$LOGFILE"
    nice -n 10 MUJOCO_GL=egl PYTHONPATH=. /home/brucewu/miniconda3/envs/lerobot/bin/python \
        scripts/render_episode_from_replay.py \
        --episode_dir "$ep" \
        --height 224 --width 224 \
        >> "$LOGFILE" 2>&1

    if [ $? -eq 0 ]; then
        rendered=$((rendered + 1))
        echo "OK $ep_name" >> "$LOGFILE"
    else
        echo "FAIL $ep_name" >> "$LOGFILE"
    fi
done

echo "Render done at $(date): rendered=$rendered skipped=$skipped" >> "$LOGFILE"
