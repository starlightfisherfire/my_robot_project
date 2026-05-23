#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Planner Trio 300g Sweep
# CEM / MultimodalCEM / MPPI × 6 obstacle templates
# 32-core parallel rendering
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=. MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"
CEM_WORKERS=32
WIDTH=1280
HEIGHT=720
FPS=10
PUSHER_MASS=0.300

# ── Planner configs: label:mode:extra_args ──
PLANNERS=(
  "cem:cem:"
  "mmcem:multimodal_cem:--lateral-offset 0.5"
  "mppi:mppi:--mppi-temperature 0.1"
)

# ── Speed/budget configs ──
# Based on what worked for heavy pusher experiments
CONFIGS=(
  "s005_h80_b500:0.05:80:10:50"
  "s005_h100_b500:0.05:100:10:50"
  "s010_h80_b500:0.10:80:10:50"
  "s010_h100_b500:0.10:100:10:50"
  "s015_h80_b500:0.15:80:10:50"
  "s015_h100_b800:0.15:100:16:50"
)

# ── 6 obstacle templates ──
TEMPLATES_ARRAY=(
  "be00:test_sim_layout_ood_blocking_easy:0"
  "bm00:test_sim_layout_ood_blocking_medium:0"
  "bh00:test_sim_layout_ood_blocking_hard:0"
  "pw00:test_sim_layout_ood_passage_direct_wide:0"
  "pm00:test_sim_layout_ood_passage_direct_medium:0"
  "ph00:test_sim_layout_ood_passage_direct_narrow:0"
)

RUN_ROOT="runs/planner_trio_300g_sweep_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

TOTAL=$((${#PLANNERS[@]} * ${#CONFIGS[@]} * ${#TEMPLATES_ARRAY[@]}))
echo "============================================================"
echo "Planner Trio 300g Sweep"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Planners: cem, multimodal_cem, mppi"
echo "Configs: ${#CONFIGS[@]} | Templates: ${#TEMPLATES_ARRAY[@]}"
echo "Total: ${TOTAL} runs | Workers: ${CEM_WORKERS} | Pusher: ${PUSHER_MASS}kg"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo "config,planner,label,speed,horizon,exec_steps,max_mpc,pusher_mass,split,idx,status,success,best_dist,avg_cost,min_cost,collision,mpc_steps,runtime_sec" > "$MANIFEST"

COUNT=0; SUCCESS=0

for PLANNER_CFG in "${PLANNERS[@]}"; do
  IFS=":" read -r PLANNER_LABEL PLANNER_MODE PLANNER_EXTRA <<< "$PLANNER_CFG"

  for TPL in "${TEMPLATES_ARRAY[@]}"; do
    IFS=":" read -r LABEL SPLIT IDX <<< "$TPL"

    for CFG in "${CONFIGS[@]}"; do
      IFS=":" read -r CFG_NAME SPEED HORIZON EXEC MPC <<< "$CFG"
      COUNT=$((COUNT+1))

      RUN_NAME="${PLANNER_LABEL}_${CFG_NAME}_${LABEL}"
      OUT="${VIDEO_DIR}/${RUN_NAME}.mp4"
      LOG="${LOG_DIR}/${RUN_NAME}.log"

      echo "=== [${COUNT}/${TOTAL}] ${RUN_NAME} ==="
      if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
        --templates "$TEMPLATES" --split "$SPLIT" --template-index "$IDX" \
        --planner-mode "$PLANNER_MODE" \
        --horizon "$HORIZON" --execute-steps "$EXEC" --max-mpc-steps "$MPC" \
        --num-samples 1024 --num-elites 96 --num-iterations 5 \
        --max-speed-mps "$SPEED" --pusher-mass "$PUSHER_MASS" \
        --strict-pose-stop --camera topdown --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
        --parallel-cem --cem-workers "$CEM_WORKERS" --mp-start-method spawn \
        $PLANNER_EXTRA \
        --out-video "$OUT" 2>&1 | tee "$LOG"; then

        SC=$(grep -c "Success: True" "$LOG" 2>/dev/null || echo 0)
        [ "$SC" -gt 0 ] && ST="True" || ST="False"
        D=$(grep "Best dist:" "$LOG" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        CO=$(grep "CEM best_cost:" "$LOG" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++;if($1<m||!m)m=$1}END{if(n>0)printf "%.4f,%.4f",s/n,m;else print "N/A,N/A"}')
        CL=$(grep "Planned collision:" "$LOG" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
        MS=$(grep "MPC Step" "$LOG" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
        RT=$(grep "Total runtime:" "$LOG" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")

        echo "${RUN_NAME},${PLANNER_LABEL},${LABEL},${SPEED},${HORIZON},${EXEC},${MPC},${PUSHER_MASS},${SPLIT},${IDX},completed,${ST},${D},${CO},${CL},${MS},${RT}" >> "$MANIFEST"
        [ "$ST" = "True" ] && SUCCESS=$((SUCCESS+1))
        echo "[DONE] ${RUN_NAME} status=${ST}"
      else
        echo "${RUN_NAME},${PLANNER_LABEL},${LABEL},${SPEED},${HORIZON},${EXEC},${MPC},${PUSHER_MASS},${SPLIT},${IDX},failed,,,,,,,,," >> "$MANIFEST"
        echo "[FAIL] ${RUN_NAME}"
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "ALL ${TOTAL} DONE. Success: ${SUCCESS}/${TOTAL} ($(( SUCCESS * 100 / TOTAL ))%)"
echo "Videos: ${VIDEO_DIR}"
echo "Manifest: ${MANIFEST}"
echo "============================================================"
cat "$MANIFEST" | column -t -s',' 2>/dev/null || cat "$MANIFEST"
