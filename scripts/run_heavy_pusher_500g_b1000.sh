#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heavy Pusher 500g: Budget 1000, 3 模板, 全分辨率
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ── Parameters ──
CEM_WORKERS=30
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5
WIDTH=1280
HEIGHT=720
FPS=10
HORIZON=100
EXEC_STEPS=10
MAX_MPC=100
TOTAL_BUDGET=1000

# ── Output ──
RUN_ROOT="runs/heavy_pusher_500g_b1000_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

# ── Speeds ──
SPEEDS=(0.01 0.02 0.05 0.10)

# ── 3 个模板 ──
TEMPLATES_CONFIG=(
  "open:data/sim/metadata/reset_templates_v0.json:test_sim_id:0"
  "be00:data/sim/metadata/reset_templates_obstacle_difficulty_v0.json:test_sim_layout_ood_blocking_easy:0"
  "pw00:data/sim/metadata/reset_templates_obstacle_difficulty_v0.json:test_sim_layout_ood_passage_direct_wide:0"
)

TOTAL=$(( ${#SPEEDS[@]} * ${#TEMPLATES_CONFIG[@]} ))

echo "============================================================"
echo "Heavy Pusher 500g Verification"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Speeds: ${SPEEDS[*]} m/s"
echo "Horizon: ${HORIZON}, Budget: ${TOTAL_BUDGET}"
echo "Pusher: 500g, Resolution: ${WIDTH}x${HEIGHT}"
echo "Templates: ${#TEMPLATES_CONFIG[@]} | Total: ${TOTAL} runs"
echo "============================================================"

echo "config,label,speed,horizon,budget,pusher_mass,split,template_index,status,success,best_dist,avg_cost,min_cost,collision,mpc_steps,runtime_sec" > "$MANIFEST"

COUNT=0
SUCCESS=0

for SPEED in "${SPEEDS[@]}"; do
  for entry in "${TEMPLATES_CONFIG[@]}"; do
    IFS=":" read -r LABEL TPL_FILE SPLIT TPL_IDX <<< "$entry"
    COUNT=$((COUNT + 1))
    CFG_NAME="h500g_s${SPEED}_${LABEL}"
    OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}.mp4"
    LOG_FILE="${LOG_DIR}/${CFG_NAME}.log"

    echo ""
    echo "=== [$COUNT/$TOTAL] $CFG_NAME speed=$SPEED m/s ==="

    if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
      --templates "$TPL_FILE" --split "$SPLIT" --template-index "$TPL_IDX" \
      --horizon "$HORIZON" --execute-steps "$EXEC_STEPS" --max-mpc-steps "$MAX_MPC" \
      --num-samples "$NUM_SAMPLES" --num-elites "$NUM_ELITES" --num-iterations "$NUM_ITER" \
      --max-speed-mps "$SPEED" --pusher-mass 0.500 \
      --strict-pose-stop --camera topdown --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
      --parallel-cem --cem-workers "$CEM_WORKERS" --mp-start-method spawn \
      --out-video "$OUT_VIDEO" 2>&1 | tee "$LOG_FILE"; then
      
      SUCC=$(grep -c "Success: True" "$LOG_FILE" 2>/dev/null || echo 0)
      [ "$SUCC" -gt 0 ] && S="True" || S="False"
      D=$(grep "Best dist:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
      C=$(grep "CEM best_cost:" "$LOG_FILE" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++;if($1<m||!m)m=$1}END{if(n>0)printf "%.4f,%.4f",s/n,m;else print "N/A,N/A"}')
      COL=$(grep "Planned collision:" "$LOG_FILE" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
      M=$(grep "MPC Step" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
      R=$(grep "Total runtime:" "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
      
      echo "${CFG_NAME},${LABEL},${SPEED},${HORIZON},${TOTAL_BUDGET},0.500,${SPLIT},${TPL_IDX},completed,${S},${D},${C},${COL},${M},${R}" >> "$MANIFEST"
      [ "$S" = "True" ] && SUCCESS=$((SUCCESS+1))
      echo "[DONE] $CFG_NAME status=$S"
    else
      echo "${CFG_NAME},${LABEL},${SPEED},${HORIZON},${TOTAL_BUDGET},0.500,${SPLIT},${TPL_IDX},failed,,,,,,,,," >> "$MANIFEST"
      echo "[FAIL] $CFG_NAME"
    fi
  done
done

echo ""
echo "=== ALL DONE ==="
echo "Success: ${SUCCESS}/${TOTAL}"
echo "Videos:  ${VIDEO_DIR}"
echo "Manifest: ${MANIFEST}"
cat "$MANIFEST" | column -t -s','