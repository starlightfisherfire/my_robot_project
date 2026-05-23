#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Blocking Hard Best Config Sweep
# Config: s0.50_h120_b600, 700g pusher, CEM, 30 workers
# Templates: ALL 10 blocking_hard (indices 0-9)
# Waits for 700g/250g/planner_trio sweeps to finish first
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=. MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
SPLIT="test_sim_layout_ood_blocking_hard"
CEM_WORKERS=30
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5
WIDTH=1280
HEIGHT=720
FPS=10
PUSHER_MASS=0.700
SPEED=0.50
HORIZON=120
EXEC=20
MAX_MPC=30

RUN_ROOT="runs/best_bh_sweep_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

TOTAL_TEMPLATES=10  # indices 0-9

echo "============================================================"
echo "Blocking Hard Best Config Sweep"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Config: speed=${SPEED}m/s | horizon=${HORIZON} | budget=600"
echo "Pusher: 700g | Workers: ${CEM_WORKERS} | Templates: ${TOTAL_TEMPLATES}"
echo "============================================================"

# ── Wait for existing sweeps to finish ──
echo ""
echo "[$(date '+%H:%M:%S')] Waiting for existing sweeps to finish..."
WAIT_START=$(date +%s)
CHECK_INTERVAL=60  # seconds between checks

while true; do
  RUNNING=""
  pgrep -f "run_heavy_pusher_700g_sweep" > /dev/null 2>&1 && RUNNING="${RUNNING}700g "
  pgrep -f "run_heavy_pusher_250g_sweep" > /dev/null 2>&1 && RUNNING="${RUNNING}250g "
  pgrep -f "run_planner_trio_300g_sweep" > /dev/null 2>&1 && RUNNING="${RUNNING}trio "
  
  if [ -z "$RUNNING" ]; then
    echo "[$(date '+%H:%M:%S')] All sweeps finished! Waited $(( ($(date +%s) - WAIT_START) / 60 )) min."
    break
  fi
  
  ELAPSED=$(( ($(date +%s) - WAIT_START) / 60 ))
  echo "[$(date '+%H:%M:%S')] Still running: ${RUNNING} (waited ${ELAPSED} min)"
  sleep ${CHECK_INTERVAL}
done

# ── Start the sweep ──
echo ""
echo "[$(date '+%H:%M:%S')] Starting blocking_hard sweep..."
echo "config,template_index,split,status,success,best_dist,avg_cost,min_cost,collision,mpc_steps,runtime_sec" > "$MANIFEST"

START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0

for IDX in $(seq 0 $((TOTAL_TEMPLATES - 1))); do
  CFG_NAME="bh_s050_h120_b600_idx${IDX}"
  
  echo ""
  echo "============================================================"
  echo "[$(date '+%H:%M:%S')] Template ${IDX}/$((TOTAL_TEMPLATES - 1)): idx=${IDX}"
  echo "============================================================"
  
  RUN_START=$(date +%s)
  
  python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates "$TEMPLATES" \
    --split "$SPLIT" \
    --template-index "$IDX" \
    --planner-mode cem \
    --horizon "$HORIZON" \
    --execute-steps "$EXEC" \
    --max-mpc-steps "$MAX_MPC" \
    --num-samples "$NUM_SAMPLES" \
    --num-elites "$NUM_ELITES" \
    --num-iterations "$NUM_ITER" \
    --max-speed-mps "$SPEED" \
    --pusher-mass "$PUSHER_MASS" \
    --strict-pose-stop \
    --camera topdown \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" \
    --parallel-cem \
    --cem-workers "$CEM_WORKERS" \
    --mp-start-method spawn \
    --out-video "${VIDEO_DIR}/${CFG_NAME}.mp4" \
    2>&1 | tee "${LOG_DIR}/${CFG_NAME}.log"
  
  RUNTIME=$(( $(date +%s) - RUN_START ))
  
  # Extract results
  if grep -q "Success: True" "${LOG_DIR}/${CFG_NAME}.log" 2>/dev/null; then
    SUCCESS="True"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    STATUS="✅"
  else
    SUCCESS="False"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    STATUS="❌"
  fi
  
  BEST_DIST=$(grep "Best dist:" "${LOG_DIR}/${CFG_NAME}.log" 2>/dev/null | awk '{print $3}' | sed 's/m//' || echo "N/A")
  MPC_STEPS=$(grep "MPC Step" "${LOG_DIR}/${CFG_NAME}.log" 2>/dev/null | tail -1 | grep -oP 'Step \K\d+' || echo "N/A")
  
  echo "${CFG_NAME},${IDX},${SPLIT},completed,${SUCCESS},${BEST_DIST},,,,${MPC_STEPS},${RUNTIME}" >> "$MANIFEST"
  
  echo "  ${STATUS} idx=${IDX} | dist=${BEST_DIST}m | mpc=${MPC_STEPS} | runtime=${RUNTIME}s"
  
  TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))
  echo "  Progress: $((IDX + 1))/${TOTAL_TEMPLATES} | Success: ${SUCCESS_COUNT} | Elapsed: ${TOTAL_ELAPSED} min"
done

TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "ALL DONE"
echo "Total: ${TOTAL_TEMPLATES} | Success: ${SUCCESS_COUNT} | Fail: ${FAIL_COUNT}"
echo "Total time: ${TOTAL_ELAPSED} min"
echo "Videos: ${VIDEO_DIR}/"
echo "Manifest: ${MANIFEST}"
echo "============================================================"
