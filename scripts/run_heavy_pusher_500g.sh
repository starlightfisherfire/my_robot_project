#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heavy Pusher Validation: 500g pusher, low speeds
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ── Parameters ──
TEMPLATES_DIFFICULTY="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
TEMPLATES_V0="data/sim/metadata/reset_templates_v0.json"
CEM_WORKERS=30
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5
WIDTH=640
HEIGHT=360
FPS=5
HORIZON=100
EXEC_STEPS=10
MAX_MPC=50
TOTAL_BUDGET=$((EXEC_STEPS * MAX_MPC))  # 500 steps

# ── Output ──
RUN_ROOT="runs/heavy_pusher_500g_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

# ── Speeds ──
SPEEDS=(0.01 0.02 0.05 0.10)

# ── Templates ──
# Format: "label:split:template_index"
TEMPLATES=(
  # open_space (from v0, index 0)
  "open0:test_sim_id:0"
  # 6 obstacle difficulty templates
  "be00:test_sim_layout_ood_blocking_easy:0"
  "bm00:test_sim_layout_ood_blocking_medium:0"
  "bh00:test_sim_layout_ood_blocking_hard:0"
  "pw00:test_sim_layout_ood_passage_direct_wide:0"
  "pm00:test_sim_layout_ood_passage_direct_medium:0"
  "ph00:test_sim_layout_ood_passage_direct_narrow:0"
)

# ── Manifest header ──
echo "config,label,speed,horizon,execute_steps,max_mpc,total_budget,pusher_mass,split,template_index,status,success,best_dist,avg_cost,min_cost,collision_count,mpc_steps,runtime_sec" > "$MANIFEST"

TOTAL=$((${#SPEEDS[@]} * ${#TEMPLATES[@]}))

echo "============================================================"
echo "Heavy Pusher Validation: 500g pusher"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Speeds: ${SPEEDS[*]} m/s"
echo "Horizon: ${HORIZON}, Execute steps: ${EXEC_STEPS}, Max MPC: ${MAX_MPC}"
echo "Total budget: ${TOTAL_BUDGET} steps"
echo "Pusher mass: 500g (10x default)"
echo "Templates: ${#TEMPLATES[@]}"
echo "Total runs: ${TOTAL}"
echo "============================================================"

COUNT=0
SUCCESS=0
FAIL=0

for SPEED in "${SPEEDS[@]}"; do
  for entry in "${TEMPLATES[@]}"; do
    IFS=":" read -r LABEL SPLIT TPL_IDX <<< "$entry"
    COUNT=$((COUNT + 1))

    # Select template file
    if [ "$SPLIT" = "test_sim_id" ]; then
      TPL_FILE="$TEMPLATES_V0"
    else
      TPL_FILE="$TEMPLATES_DIFFICULTY"
    fi

    SPEED_TAG=$(echo "$SPEED" | sed 's/\.//g' | sed 's/^0*//')
    CFG_NAME="hp500_s${SPEED_TAG}_${LABEL}"
    OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}.mp4"
    LOG_FILE="${LOG_DIR}/${CFG_NAME}.log"

    echo ""
    echo "============================================================"
    echo "[${COUNT}/${TOTAL}] ${CFG_NAME}"
    echo "  speed=${SPEED}m/s horizon=${HORIZON} pusher=500g split=${SPLIT} tpl=${TPL_IDX}"
    echo "============================================================"

    if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
      --templates "$TPL_FILE" \
      --split "$SPLIT" \
      --template-index "$TPL_IDX" \
      --horizon "$HORIZON" \
      --execute-steps "$EXEC_STEPS" \
      --max-mpc-steps "$MAX_MPC" \
      --num-samples "$NUM_SAMPLES" \
      --num-elites "$NUM_ELITES" \
      --num-iterations "$NUM_ITER" \
      --max-speed-mps "$SPEED" \
      --pusher-mass 0.500 \
      --strict-pose-stop \
      --camera topdown \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --fps "$FPS" \
      --parallel-cem \
      --cem-workers "$CEM_WORKERS" \
      --mp-start-method spawn \
      --out-video "$OUT_VIDEO" \
      2>&1 | tee "$LOG_FILE"; then

      SUCC=$(grep -c "Success: True" "$LOG_FILE" 2>/dev/null || echo 0)
      if [ "$SUCC" -gt 0 ]; then SUCC="True"; else SUCC="False"; fi
      BEST_DIST=$(grep "Best dist:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
      COSTS=$(grep "CEM best_cost:" "$LOG_FILE" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++; if($1<m||!m)m=$1} END{if(n>0)printf "%.4f,%.4f",s/n,m; else print "N/A,N/A"}')
      COLL=$(grep "Planned collision:" "$LOG_FILE" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1} END{print s+0}')
      MPC_STEPS=$(grep "MPC Step" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
      RT=$(grep "Total runtime:" "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")

      echo "${CFG_NAME},${LABEL},${SPEED},${HORIZON},${EXEC_STEPS},${MAX_MPC},${TOTAL_BUDGET},0.500,${SPLIT},${TPL_IDX},completed,${SUCC},${BEST_DIST},${COSTS},${COLL},${MPC_STEPS},${RT}" >> "$MANIFEST"

      if [ "$SUCC" = "True" ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "[DONE ✅] ${CFG_NAME}"
      else
        FAIL=$((FAIL + 1))
        echo "[DONE ❌] ${CFG_NAME}"
      fi
    else
      FAIL=$((FAIL + 1))
      echo "${CFG_NAME},${LABEL},${SPEED},${HORIZON},${EXEC_STEPS},${MAX_MPC},${TOTAL_BUDGET},0.500,${SPLIT},${TPL_IDX},failed,,,,,,,,,," >> "$MANIFEST"
      echo "[FAIL] ${CFG_NAME}"
    fi
  done
done

echo ""
echo "============================================================"
echo "ALL ${TOTAL} DONE."
echo "Success: ${SUCCESS}/${TOTAL} ($(( SUCCESS * 100 / TOTAL ))%)"
echo "Failure: ${FAIL}/${TOTAL}"
echo "Videos:   ${VIDEO_DIR}"
echo "Logs:     ${LOG_DIR}"
echo "Manifest: ${MANIFEST}"
echo "============================================================"

echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
column -t -s',' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
