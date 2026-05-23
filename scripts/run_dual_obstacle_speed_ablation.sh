#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dual-Obstacle Speed Ablation
# Speeds: 0.75, 1.0, 1.5, 2.0 m/s
# Horizon: 80, execute_steps=20, max_mpc=50 → 1000 total steps
# Modes: passage_direct_wide, passage_direct_medium, passage_direct_narrow (2 tpl each)
# 30-core parallel CEM
# 4 × 3 × 2 = 24 videos
# ============================================================

cd ~/my_robot_project

set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
HORIZON=80
EXEC_STEPS=20       # 50 × 20 = 1000 total steps
MAX_MPC=50
CEM_WORKERS=30
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5

SPEEDS=("0.75" "1.0" "1.5" "2.0")

MODES=(
  "passage_direct_wide:test_sim_layout_ood_passage_direct_wide"
  "passage_direct_medium:test_sim_layout_ood_passage_direct_medium"
  "passage_direct_narrow:test_sim_layout_ood_passage_direct_narrow"
)
TEMPLATE_INDICES=("0" "1")

RUN_ROOT="runs/dual_obstacle_speed_ablation_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

TOTAL=$(( ${#SPEEDS[@]} * ${#MODES[@]} * ${#TEMPLATE_INDICES[@]} ))

echo "============================================================"
echo "Dual-Obstacle Speed Ablation"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Speeds: ${SPEEDS[*]}"
echo "Horizon: ${HORIZON}, execute_steps=${EXEC_STEPS}, max_mpc=${MAX_MPC}"
echo "Modes: passage_direct_wide, passage_direct_medium, passage_direct_narrow"
echo "Templates per mode: ${#TEMPLATE_INDICES[@]}"
echo "Total videos: ${TOTAL}"
echo "Config: CEM_workers=${CEM_WORKERS}, samples=${NUM_SAMPLES}, elites=${NUM_ELITES}, iter=${NUM_ITER}"
echo "============================================================"

echo "config,speed,horizon,mode,template_idx,obstacle_type,status,best_dist,avg_cost,min_cost,collision_count,mpc_steps,runtime_sec" > "$MANIFEST"

COUNT=0
FAIL=0
for SPEED in "${SPEEDS[@]}"; do
  SPEED_TAG=$(echo "$SPEED" | sed 's/\.//g')
  for mode_entry in "${MODES[@]}"; do
    IFS=":" read -r MODE SPLIT <<< "$mode_entry"
    for TPL_IDX in "${TEMPLATE_INDICES[@]}"; do
      COUNT=$((COUNT + 1))
      CFG_NAME="sp${SPEED_TAG}_h${HORIZON}_${MODE}_t${TPL_IDX}"

      OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}.mp4"
      LOG_FILE="${LOG_DIR}/${CFG_NAME}.log"

      echo
      echo "============================================================"
      echo "[${COUNT}/${TOTAL}] ${CFG_NAME}"
      echo "  speed=${SPEED} horizon=${HORIZON} mode=${MODE} tpl=${TPL_IDX}"
      echo "============================================================"

      if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
        --templates "$TEMPLATES" \
        --split "$SPLIT" \
        --template-index "$TPL_IDX" \
        --horizon "$HORIZON" \
        --execute-steps "$EXEC_STEPS" \
        --max-mpc-steps "$MAX_MPC" \
        --num-samples "$NUM_SAMPLES" \
        --num-elites "$NUM_ELITES" \
        --num-iterations "$NUM_ITER" \
        --max-speed-mps "$SPEED" \
        --strict-pose-stop \
        --camera topdown \
        --width 1280 \
        --height 720 \
        --fps 10 \
        --parallel-cem \
        --cem-workers "$CEM_WORKERS" \
        --mp-start-method spawn \
        --out-video "$OUT_VIDEO" \
        2>&1 | tee "$LOG_FILE"; then

        # Extract metrics
        SUCC=$(grep -c "Success: True" "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$SUCC" -gt 0 ]; then SUCC="True"; else SUCC="False"; fi
        BEST_DIST=$(grep "Best dist:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        COSTS=$(grep "CEM best_cost:" "$LOG_FILE" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++; if($1<m||!m)m=$1} END{if(n>0)printf "%.4f,%.4f",s/n,m; else print "N/A,N/A"}')
        COLL=$(grep "Planned collision:" "$LOG_FILE" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1} END{print s+0}')
        MPC_STEPS=$(grep "MPC Step" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
        RT=$(grep "Total runtime:" "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")

        echo "${CFG_NAME},${SPEED},${HORIZON},${MODE},${TPL_IDX},double,${SUCC},${BEST_DIST},${COSTS},${COLL},${MPC_STEPS},${RT}" >> "$MANIFEST"
        echo "[DONE] ${CFG_NAME}"
      else
        FAIL=$((FAIL + 1))
        echo "${CFG_NAME},${SPEED},${HORIZON},${MODE},${TPL_IDX},double,FAILED,N/A,N/A,N/A,N/A,N/A,N/A" >> "$MANIFEST"
        echo "[FAIL] ${CFG_NAME}"
      fi
    done
  done
done

echo
echo "============================================================"
echo "ALL ${TOTAL} DONE. (Failures: ${FAIL})"
echo "Videos:   ${VIDEO_DIR}"
echo "Logs:     ${LOG_DIR}"
echo "Manifest: ${MANIFEST}"
echo "============================================================"

# ── Summary Table ──
echo
echo "=== RESULTS SUMMARY ==="
echo
column -t -s',' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
