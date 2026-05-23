#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 700g Heavy Pusher Sweep
# Budgets: 600/1200 | Horizons: 100/120/140 | Exec: 20/30
# Speeds: 0.1-1.0 m/s | 32 cores
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=. MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
CEM_WORKERS=32
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5
WIDTH=1280
HEIGHT=720
FPS=10
PUSHER_MASS=0.700

# ── Variables ──
SPEEDS=(0.10 0.15 0.20 0.30 0.50 0.75 1.00)
HORIZONS=(100 120 140)

# Budget, execute_steps, max_mpc
BUDGETS=(
  "600:20:30"
  "600:30:20"
  "1200:20:60"
  "1200:30:40"
)

# Templates: label:split:index
TEMPLATES_CFG=(
  "bh00:test_sim_layout_ood_blocking_hard:0"
  "pm02:test_sim_layout_ood_passage_direct_medium:2"
)

RUN_ROOT="runs/heavy_pusher_700g_sweep_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

TOTAL=$((${#SPEEDS[@]} * ${#HORIZONS[@]} * ${#BUDGETS[@]} * ${#TEMPLATES_CFG[@]}))
echo "============================================================"
echo "700g Pusher Sweep"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Speeds: ${SPEEDS[*]} m/s | Horizons: ${HORIZONS[*]} | Budgets: 600/1200"
echo "Pusher: 700g | Workers: ${CEM_WORKERS} | Templates: ${#TEMPLATES_CFG[@]}"
echo "Total: ${TOTAL} runs | Start: $(date '+%H:%M')"
echo "============================================================"

echo "config,label,speed,horizon,budget,exec_steps,max_mpc,pusher_mass,split,idx,status,success,best_dist,avg_cost,min_cost,collision,mpc_steps,runtime_sec" > "$MANIFEST"

COUNT=0; SUCCESS=0

for TPL in "${TEMPLATES_CFG[@]}"; do
  IFS=":" read -r LABEL SPLIT IDX <<< "$TPL"
  for BUDGET_CFG in "${BUDGETS[@]}"; do
    IFS=":" read -r BUDGET EXEC MPC <<< "$BUDGET_CFG"
    for H in "${HORIZONS[@]}"; do
      for S in "${SPEEDS[@]}"; do
        COUNT=$((COUNT+1))
        CFG="h700g_s${S}_h${H}_b${BUDGET}_e${EXEC}_${LABEL}"
        OUT="${VIDEO_DIR}/${CFG}.mp4"
        LOG="${LOG_DIR}/${CFG}.log"

        echo "=== [${COUNT}/${TOTAL}] ${CFG} ==="
        if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
          --templates "$TEMPLATES" --split "$SPLIT" --template-index "$IDX" \
          --horizon "$H" --execute-steps "$EXEC" --max-mpc-steps "$MPC" \
          --num-samples "$NUM_SAMPLES" --num-elites "$NUM_ELITES" --num-iterations "$NUM_ITER" \
          --max-speed-mps "$S" --pusher-mass "$PUSHER_MASS" \
          --strict-pose-stop --camera topdown --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
          --parallel-cem --cem-workers "$CEM_WORKERS" --mp-start-method spawn \
          --out-video "$OUT" 2>&1 | tee "$LOG"; then
          
          SC=$(grep -c "Success: True" "$LOG" 2>/dev/null || echo 0)
          [ "$SC" -gt 0 ] && ST="True" || ST="False"
          D=$(grep "Best dist:" "$LOG" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
          CO=$(grep "CEM best_cost:" "$LOG" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++;if($1<m||!m)m=$1}END{if(n>0)printf "%.4f,%.4f",s/n,m;else print "N/A,N/A"}')
          CL=$(grep "Planned collision:" "$LOG" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
          MS=$(grep "MPC Step" "$LOG" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
          RT=$(grep "Total runtime:" "$LOG" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
          
          echo "${CFG},${LABEL},${S},${H},${BUDGET},${EXEC},${MPC},${PUSHER_MASS},${SPLIT},${IDX},completed,${ST},${D},${CO},${CL},${MS},${RT}" >> "$MANIFEST"
          [ "$ST" = "True" ] && SUCCESS=$((SUCCESS+1))
          echo "[DONE] ${CFG} status=${ST}"
        else
          echo "${CFG},${LABEL},${S},${H},${BUDGET},${EXEC},${MPC},${PUSHER_MASS},${SPLIT},${IDX},failed,,,,,,,,," >> "$MANIFEST"
          echo "[FAIL] ${CFG}"
        fi
      done
    done
  done
done

echo ""
echo "============================================================"
echo "ALL ${TOTAL} DONE. Success: ${SUCCESS}/${TOTAL} ($((SUCCESS*100/TOTAL))%)"
echo "Videos: ${VIDEO_DIR}"
echo "Manifest: ${MANIFEST}"
echo "============================================================"
cat "$MANIFEST" | column -t -s',' 2>/dev/null || cat "$MANIFEST"
