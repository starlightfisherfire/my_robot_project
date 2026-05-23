#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Best Config Ablation: Speed 0.05 to 1.0 m/s
# Stage 1: Core diagnostic scout
# ============================================================

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ── Parameters ──
TEMPLATES="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
CEM_WORKERS=30
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5
WIDTH=640
HEIGHT=360
FPS=5

# Time guard
START_TIME=$(date +%s)
MAX_WALL_SECONDS=$((8 * 3600))
STOP_LAUNCH_AFTER_SECONDS=$((7 * 3600 + 30 * 60))

# ── Output ──
RUN_ROOT="runs/best_config_ablation_speed005_to_100_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
SELECTED_TEMPLATES="${RUN_ROOT}/selected_templates.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

# ── Core templates (8) ──
# Format: "label:split:template_index"
CORE_TEMPLATES=(
  "bm08:test_sim_layout_ood_blocking_medium:8"
  "bh00:test_sim_layout_ood_blocking_hard:0"
  "bh01:test_sim_layout_ood_blocking_hard:1"
  "bh09:test_sim_layout_ood_blocking_hard:9"
  "pm02:test_sim_layout_ood_passage_direct_medium:2"
  "pm06:test_sim_layout_ood_passage_direct_medium:6"
  "ph04:test_sim_layout_ood_passage_direct_narrow:4"
  "ph07:test_sim_layout_ood_passage_direct_narrow:7"
)

# ── Stage 1 variables ──
SPEEDS=(0.05 0.10 0.20 0.35 0.50 0.75 1.00)
HORIZONS=(80 120 140 160)

# ── Priority groups ──
# Priority 1: Main candidate matrix
P1_SPEEDS=(0.20 0.50 0.75 1.00)
P1_HORIZONS=(80 120 140 160)

# Priority 2: Low-speed boundary
P2_SPEEDS=(0.05 0.10 0.35)
P2_HORIZONS=(80 140)

# ── Generate selected_templates.csv ──
echo "stage,label,split,template_index,selected_template_id,layout_family,num_obstacles,object_goal_distance,passage_gap,effective_passage_gap,blocking_difficulty,feasibility,obstacle_size_x,obstacle_size_y" > "$SELECTED_TEMPLATES"

for entry in "${CORE_TEMPLATES[@]}"; do
  IFS=":" read -r LABEL SPLIT TPL_IDX <<< "$entry"
  
  # Extract template info using python
  python3 -c "
import json
with open('${TEMPLATES}') as f:
    data = json.load(f)
templates = [t for t in data if t['split'] == '${SPLIT}']
if ${TPL_IDX} >= len(templates):
    print(f'ERROR: template-index ${TPL_IDX} out of range for split ${SPLIT}')
    exit(1)
t = templates[${TPL_IDX}]
tid = t.get('reset_template_id', 'N/A')
layout = t.get('layout_family', 'N/A')
nobs = len(t.get('obstacles', []))
diff = t.get('blocking_difficulty', 'N/A')
sx = t.get('obstacle_size_x', 0)
sy = t.get('obstacle_size_y', 0)
obj = t.get('object_initial_pose', {})
goal = t.get('goal_pose', {})
dist = ((goal.get('x',0)-obj.get('x',0))**2 + (goal.get('y',0)-obj.get('y',0))**2)**0.5
pgap = t.get('passage_gap', '')
egap = t.get('effective_passage_gap', '')
print(f'1,${LABEL},${SPLIT},${TPL_IDX},{tid},{layout},{nobs},{dist:.3f},{pgap},{egap},{diff},unknown,{sx},{sy}')
" >> "$SELECTED_TEMPLATES"
done

echo "=== Selected Templates ==="
cat "$SELECTED_TEMPLATES"
echo ""

# ── Manifest header ──
echo "stage,priority,group,config,label,speed,horizon,execute_steps,max_mpc_steps,total_budget,split,template_index,selected_template_id,layout_family,num_obstacles,status,success,best_dist,avg_cost,min_cost,collision_count,collision_rate,mpc_steps,runtime_sec,video_path,log_path,error" > "$MANIFEST"

# ── Run function ──
run_case() {
  local STAGE=$1
  local PRIORITY=$2
  local GROUP=$3
  local SPEED=$4
  local HORIZON=$5
  local LABEL=$6
  local SPLIT=$7
  local TPL_IDX=$8
  
  local SPEED_TAG=$(echo "$SPEED" | sed 's/\.//g' | sed 's/^0*//')
  local CFG_NAME="s${SPEED_TAG}_h${HORIZON}_${LABEL}"
  local OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}.mp4"
  local LOG_FILE="${LOG_DIR}/${CFG_NAME}.log"
  
  # Check time guard
  local NOW=$(date +%s)
  local ELAPSED=$((NOW - START_TIME))
  if [ $ELAPSED -gt $STOP_LAUNCH_AFTER_SECONDS ]; then
    echo "[TIME GUARD] stop launching new cases (elapsed=${ELAPSED}s)"
    echo "${STAGE},${PRIORITY},${GROUP},${CFG_NAME},${LABEL},${SPEED},${HORIZON},10,100,1000,${SPLIT},${TPL_IDX},,,0,timeout_guard_skipped,,,,,,,,,,,," >> "$MANIFEST"
    return 1
  fi
  
  echo ""
  echo "============================================================"
  echo "[${STAGE}] P${PRIORITY} ${CFG_NAME} (speed=${SPEED}m/s horizon=${HORIZON})"
  echo "  split=${SPLIT} tpl_idx=${TPL_IDX}"
  echo "  elapsed=$((ELAPSED/60))min"
  echo "============================================================"
  
  if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates "$TEMPLATES" \
    --split "$SPLIT" \
    --template-index "$TPL_IDX" \
    --horizon "$HORIZON" \
    --execute-steps 10 \
    --max-mpc-steps 100 \
    --num-samples "$NUM_SAMPLES" \
    --num-elites "$NUM_ELITES" \
    --num-iterations "$NUM_ITER" \
    --max-speed-mps "$SPEED" \
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
    
    echo "${STAGE},${PRIORITY},${GROUP},${CFG_NAME},${LABEL},${SPEED},${HORIZON},10,100,1000,${SPLIT},${TPL_IDX},,,$(python3 -c "import json; d=json.load(open('${TEMPLATES}')); ts=[t for t in d if t['split']=='${SPLIT}']; print(len(ts[${TPL_IDX}].get('obstacles',[]))) if ${TPL_IDX}<len(ts) else print(0)"),completed,${SUCC},${BEST_DIST},${COSTS},${COLL},0,${MPC_STEPS},${RT},${OUT_VIDEO},${LOG_FILE}," >> "$MANIFEST"
    
    echo "[DONE] ${CFG_NAME} status=${SUCC}"
  else
    echo "${STAGE},${PRIORITY},${GROUP},${CFG_NAME},${LABEL},${SPEED},${HORIZON},10,100,1000,${SPLIT},${TPL_IDX},,,0,failed,,,,,,,,,,${OUT_VIDEO},${LOG_FILE},script_error" >> "$MANIFEST"
    echo "[FAIL] ${CFG_NAME}"
  fi
}

# ── Stage 1 execution ──
echo ""
echo "============================================================"
echo "Stage 1: Core diagnostic scout"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Speeds: ${SPEEDS[*]} m/s"
echo "Horizons: ${HORIZONS[*]}"
echo "Templates: ${#CORE_TEMPLATES[@]} core"
echo "Priority 1: ${#P1_SPEEDS[@]} speeds × ${#P1_HORIZONS[@]} horizons × 8 templates = $((${#P1_SPEEDS[@]} * ${#P1_HORIZONS[@]} * 8)) runs"
echo "Priority 2: ${#P2_SPEEDS[@]} speeds × ${#P2_HORIZONS[@]} horizons × 8 templates = $((${#P2_SPEEDS[@]} * ${#P2_HORIZONS[@]} * 8)) runs"
echo "Time guard: ${MAX_WALL_SECONDS}s (stop launch at ${STOP_LAUNCH_AFTER_SECONDS}s)"
echo "============================================================"

# Priority 1: Main candidate matrix
echo ""
echo "=== Priority 1: Main candidate matrix ==="
P1_COUNT=0
for SPEED in "${P1_SPEEDS[@]}"; do
  for HORIZON in "${P1_HORIZONS[@]}"; do
    for entry in "${CORE_TEMPLATES[@]}"; do
      IFS=":" read -r LABEL SPLIT TPL_IDX <<< "$entry"
      P1_COUNT=$((P1_COUNT + 1))
      echo "[P1 ${P1_COUNT}/128] speed=${SPEED} horizon=${HORIZON} ${LABEL}"
      run_case "stage1" "1" "main_candidate" "$SPEED" "$HORIZON" "$LABEL" "$SPLIT" "$TPL_IDX" || break 3
    done
  done
done

# Priority 2: Low-speed boundary
echo ""
echo "=== Priority 2: Low-speed boundary ==="
P2_COUNT=0
for SPEED in "${P2_SPEEDS[@]}"; do
  for HORIZON in "${P2_HORIZONS[@]}"; do
    for entry in "${CORE_TEMPLATES[@]}"; do
      IFS=":" read -r LABEL SPLIT TPL_IDX <<< "$entry"
      P2_COUNT=$((P2_COUNT + 1))
      echo "[P2 ${P2_COUNT}/48] speed=${SPEED} horizon=${HORIZON} ${LABEL}"
      run_case "stage1" "2" "low_speed_boundary" "$SPEED" "$HORIZON" "$LABEL" "$SPLIT" "$TPL_IDX" || break 3
    done
  done
done

# Priority 3: Complete grid (if time permits)
echo ""
echo "=== Priority 3: Complete grid (if time permits) ==="
P3_COUNT=0
for SPEED in "${SPEEDS[@]}"; do
  for HORIZON in "${HORIZONS[@]}"; do
    for entry in "${CORE_TEMPLATES[@]}"; do
      IFS=":" read -r LABEL SPLIT TPL_IDX <<< "$entry"
      # Skip if already run in P1 or P2
      CFG_NAME="s$(echo "$SPEED" | sed 's/\.//g' | sed 's/^0*//')_h${HORIZON}_${LABEL}"
      if grep -q "$CFG_NAME" "$MANIFEST" 2>/dev/null; then
        continue
      fi
      P3_COUNT=$((P3_COUNT + 1))
      echo "[P3 ${P3_COUNT}] speed=${SPEED} horizon=${HORIZON} ${LABEL}"
      run_case "stage1" "3" "complete_grid" "$SPEED" "$HORIZON" "$LABEL" "$SPLIT" "$TPL_IDX" || break 3
    done
  done
done

# ── Summary ──
echo ""
echo "============================================================"
echo "Stage 1 Complete!"
echo "============================================================"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "Total time: $((TOTAL_TIME/60)) minutes"
echo "Manifest: ${MANIFEST}"
echo "Videos: ${VIDEO_DIR}"
echo "Logs: ${LOG_DIR}"

# Count results
TOTAL=$(tail -1 "$MANIFEST" | wc -l)
SUCCESS=$(grep "completed,True" "$MANIFEST" | wc -l)
FAIL=$(grep "completed,False" "$MANIFEST" | wc -l)
echo ""
echo "Results: ${SUCCESS} success / ${FAIL} fail / ${TOTAL} total"

# Run summarizer
echo ""
echo "Running summarizer..."
python3 scripts/summarize_best_config_ablation.py --run-root "$RUN_ROOT" 2>/dev/null || echo "Summarizer not yet created"

echo ""
echo "=== DONE ==="
