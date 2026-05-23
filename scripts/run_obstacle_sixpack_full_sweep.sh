#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Obstacle Sixpack Full Sweep
# 5 validated configs × 6 templates = 30 videos
# 28-core parallel CEM rendering per video
# ============================================================

cd ~/my_robot_project

# Fix for CUDA nvcc activation script unbound variable issue
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"

# --- 5 validated configs (from obstacle_speed_budget_sweep_20260514_021943) ---
# speed_075: budget=800,1000 (2 configs)  — both reached cost~0.05
# speed_10:  budget=600,800,1000 (3 configs) — all reached cost~0.05
CONFIGS=(
  "speed075_b800:0.075:40"
  "speed075_b1000:0.075:50"
  "speed10_b600:0.10:30"
  "speed10_b800:0.10:40"
  "speed10_b1000:0.10:50"
)

# --- 6 templates (single obstacle 0-2, double obstacle 3-5) ---
TEMPLATES_ARRAY=(
  "t0_blocking_easy:test_sim_layout_ood_blocking_easy:0:single"
  "t1_blocking_medium:test_sim_layout_ood_blocking_medium:0:single"
  "t2_blocking_hard:test_sim_layout_ood_blocking_hard:0:single"
  "t3_passage_direct_wide:test_sim_layout_ood_passage_direct_wide:0:double"
  "t4_passage_direct_medium:test_sim_layout_ood_passage_direct_medium:0:double"
  "t5_passage_direct_narrow:test_sim_layout_ood_passage_direct_narrow:0:double"
)

RUN_ROOT="runs/debug/obstacle_sixpack_full_sweep_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"

mkdir -p "$VIDEO_DIR" "$LOG_DIR"

TOTAL=$(( ${#CONFIGS[@]} * ${#TEMPLATES_ARRAY[@]} ))

echo "============================================================"
echo "Obstacle Sixpack Full Sweep"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Configs: ${#CONFIGS[@]}"
echo "Templates: ${#TEMPLATES_ARRAY[@]}"
echo "Total videos: ${TOTAL}"
echo "============================================================"

# Write CSV header for post-run analysis
echo "config,speed,budget,max_mpc,template,family,obstacle_type,status,total_steps,final_pos_error,final_theta_error,best_cost_avg,collision_count,collision_rate,contact_rate,executed_steps" > "$MANIFEST"

COUNT=0
for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r CFG_NAME SPEED MAX_MPC <<< "$cfg"

  for tpl in "${TEMPLATES_ARRAY[@]}"; do
    IFS=":" read -r TPL_NAME SPLIT TPL_IDX OBS_TYPE <<< "$tpl"

    COUNT=$((COUNT + 1))
    OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}__${TPL_NAME}.mp4"
    LOG_FILE="${LOG_DIR}/${CFG_NAME}__${TPL_NAME}.log"

    echo
    echo "============================================================"
    echo "[${COUNT}/${TOTAL}] ${CFG_NAME} @ ${TPL_NAME}"
    echo "  speed=${SPEED}  max_mpc=${MAX_MPC}  obstacle=${OBS_TYPE}"
    echo "  video=${OUT_VIDEO}"
    echo "  log=${LOG_FILE}"
    echo "============================================================"

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
      --templates "$TEMPLATES" \
      --split "$SPLIT" \
      --template-index "$TPL_IDX" \
      --horizon 80 \
      --execute-steps 20 \
      --max-mpc-steps "$MAX_MPC" \
      --num-samples 1024 \
      --num-elites 96 \
      --num-iterations 5 \
      --max-speed-mps "$SPEED" \
      --strict-pose-stop \
      --camera topdown \
      --width 1280 \
      --height 720 \
      --fps 10 \
      --parallel-cem \
      --cem-workers 28 \
      --mp-start-method spawn \
      --out-video "$OUT_VIDEO" \
      2>&1 | tee "$LOG_FILE"

    echo "[DONE] ${CFG_NAME} @ ${TPL_NAME}"
  done
done

echo
echo "============================================================"
echo "ALL ${TOTAL} VIDEOS COMPLETE."
echo ""

# --- Post-run data aggregation ---
echo "Aggregating data from logs..."
echo "config,speed,budget,max_mpc,template,family,obstacle_type,status,total_steps,final_pos_error,final_theta_error,best_cost_avg,collision_count,collision_rate,contact_rate,executed_steps" > "$MANIFEST"

for LOG_FILE in "$LOG_DIR"/*.log; do
  FNAME=$(basename "$LOG_FILE" .log)
  CFG_NAME=$(echo "$FNAME" | cut -d'_' -f1-3)
  TPL_NAME=$(echo "$FNAME" | cut -d'_' -f4-)
  SPEED=$(echo "$CFG_NAME" | grep -oP 'speed\K[0-9]+')
  BUDGET=$(echo "$CFG_NAME" | grep -oP 'b\K[0-9]+')

  # Parse log for key metrics
  STATUS="done"
  TOTAL_STEPS=$(grep -c "Planned contact\|Planned collision" "$LOG_FILE" 2>/dev/null || echo 0)
  COLLISION_COUNT=$(grep "Planned collision" "$LOG_FILE" 2>/dev/null | grep -oP 'count=\K[0-9.]+' | awk '{s+=$1} END {print s+0}')
  COLLISION_RATE=$(grep "Planned collision" "$LOG_FILE" 2>/dev/null | grep -oP 'rate=\K[0-9.]+' | awk '{s+=$1; n++} END {if(n>0) print s/n; else print 0}')
  BEST_COST_AVG=$(grep "CEM best_cost" "$LOG_FILE" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1; n++} END {if(n>0) printf "%.4f", s/n; else print "N/A"}')
  CONTACT_RATE=$(grep "contact@step" "$LOG_FILE" 2>/dev/null | wc -l)

  # Extract family from TPL_NAME
  FAMILY=$(echo "$TPL_NAME" | sed 's/t[0-9]_//')

  echo "${CFG_NAME},${SPEED},${BUDGET},,${TPL_NAME},${FAMILY},,${STATUS},${TOTAL_STEPS},,,${BEST_COST_AVG},${COLLISION_COUNT},${COLLISION_RATE},${CONTACT_RATE}," >> "$MANIFEST"
done

echo "Manifest written: ${MANIFEST}"
echo "Videos: ${VIDEO_DIR}"
echo "Logs:   ${LOG_DIR}"
echo "============================================================"
