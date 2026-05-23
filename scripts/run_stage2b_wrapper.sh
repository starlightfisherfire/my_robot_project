#!/bin/bash
# Self-contained MPPI Stage 2B runner
# Runs sweep, summarizer, copies output — all in one shot
set -e

cd /home/brucewu/my_robot_project
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Run sweep
echo "=== Starting Stage 2B Speed Sweep at $(date) ==="
bash scripts/run_mppi_stage2b_speed_sweep.sh --run

# Wait for sweep to finish (manifest should be built)
SWEEP_EXIT=$?
echo "=== Sweep exited with code $SWEEP_EXIT at $(date) ==="

# Find the latest run dir
RUN_DIR=$(ls -td runs/mppi_stage2b_speed_20* 2>/dev/null | grep -v smoke | head -1)
echo "Run dir: $RUN_DIR"

# Run summarizer
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/manifest.csv" ]; then
  echo "=== Running summarizer ==="
  /home/brucewu/miniconda3/envs/lerobot/bin/python scripts/summarize_mppi_stage2b_speed_sweep.py --run-dir "$RUN_DIR"
  
  # Copy summary to docs
  if [ -f "$RUN_DIR/stage2b_speed_summary.md" ]; then
    cp "$RUN_DIR/stage2b_speed_summary.md" docs/mppi_stage2b_speed_sweep_summary.md
    echo "=== Summary copied to docs/mppi_stage2b_speed_sweep_summary.md ==="
  fi
fi

echo "=== DONE at $(date) ==="
echo "STAGE2B_COMPLETE"
