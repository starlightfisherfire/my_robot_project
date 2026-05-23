#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Speed Ablation: 0.15 & 0.20 m/s — budgets 800 + 1000
# 2 speeds × 2 budgets × 6 templates = 24 videos
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"

CONFIGS=(
  "speed015_b800:0.15:40"
  "speed015_b1000:0.15:50"
  "speed020_b800:0.20:40"
  "speed020_b1000:0.20:50"
)

TEMPLATES_ARRAY=(
  "t0_blocking_easy:test_sim_layout_ood_blocking_easy:0:single"
  "t1_blocking_medium:test_sim_layout_ood_blocking_medium:0:single"
  "t2_blocking_hard:test_sim_layout_ood_blocking_hard:0:single"
  "t3_passage_direct_wide:test_sim_layout_ood_passage_direct_wide:0:double"
  "t4_passage_direct_medium:test_sim_layout_ood_passage_direct_medium:0:double"
  "t5_passage_direct_narrow:test_sim_layout_ood_passage_direct_narrow:0:double"
)

RUN_ROOT="runs/debug/obstacle_speed_ablation_high_b800b1000_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

TOTAL=$(( ${#CONFIGS[@]} * ${#TEMPLATES_ARRAY[@]} ))
echo "=== Speed Ablation: 0.15 & 0.20 — budgets 800+1000 ==="
echo "RUN_ROOT=${RUN_ROOT}"
echo "Total videos: ${TOTAL}"
echo

echo "config,speed,budget,template,family,obstacle_type,status,best_dist,avg_cost,min_cost,collision_count,mpc_steps,runtime_sec" > "$MANIFEST"

COUNT=0
for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r CFG_NAME SPEED MAX_MPC <<< "$cfg"
  for tpl in "${TEMPLATES_ARRAY[@]}"; do
    IFS=":" read -r TPL_NAME SPLIT TPL_IDX OBS_TYPE <<< "$tpl"
    COUNT=$((COUNT + 1))
    OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}__${TPL_NAME}.mp4"
    LOG_FILE="${LOG_DIR}/${CFG_NAME}__${TPL_NAME}.log"

    echo "========================================"
    echo "[${COUNT}/${TOTAL}] ${CFG_NAME} @ ${TPL_NAME}"
    echo "  speed=${SPEED} max_mpc=${MAX_MPC}"

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
      --templates "$TEMPLATES" --split "$SPLIT" --template-index "$TPL_IDX" \
      --horizon 80 --execute-steps 20 --max-mpc-steps "$MAX_MPC" \
      --num-samples 1024 --num-elites 96 --num-iterations 5 \
      --max-speed-mps "$SPEED" --strict-pose-stop \
      --camera topdown --width 1280 --height 720 --fps 10 \
      --parallel-cem --cem-workers 30 --mp-start-method spawn \
      --out-video "$OUT_VIDEO" 2>&1 | tee "$LOG_FILE"

    # Extract metrics
    SUCC="False"; grep -q "Success: True" "$LOG_FILE" 2>/dev/null && SUCC="True"
    BEST_DIST=$(grep "Best dist:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
    COSTS=$(grep "CEM best_cost:" "$LOG_FILE" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++;if($1<m||!m)m=$1} END{printf "%.4f,%.4f",s/n,m}')
    COLL=$(grep "Planned collision:" "$LOG_FILE" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1} END{print s+0}')
    MPC_STEPS=$(grep "MPC Step" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
    RT=$(grep "Total runtime:" "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
    FAMILY=$(echo "$TPL_NAME" | sed 's/t[0-9]_//')
    echo "${CFG_NAME},${SPEED},,${TPL_NAME},${FAMILY},${OBS_TYPE},${SUCC},${BEST_DIST},${COSTS},${COLL},${MPC_STEPS},${RT}" >> "$MANIFEST"
    echo "[DONE] ${CFG_NAME} @ ${TPL_NAME}"
  done
done

echo
echo "=== ALL ${TOTAL} DONE ==="
echo "Videos: ${VIDEO_DIR}"
echo "Logs:   ${LOG_DIR}"
echo "Manifest: ${MANIFEST}"
