#!/usr/bin/env bash
set -eo pipefail
# MPPI Parameter Sweep — Fixed version
# Fixes: logging, no time limit, all phases sequential, parallel=8

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u
export PYTHONPATH=. MUJOCO_GL=egl
# Don't force single-thread — let PyTorch/MuJoCo use multiple cores per process

# ── Constants ──
RENDER_SCRIPT="scripts/render_closed_loop_rollout_mppi_with_data.py"
TEMPLATE_FILE="data/sim/metadata/reset_templates_obstacle_10family_v0.json"
DATA_DIR="data/sim/mppi_sweep_fixed"

FIXED_H=140
FIXED_EXEC=10
FIXED_MPC=100
FIXED_BUDGET=1000
FIXED_SPEED=0.75
FIXED_SAMPLES=1024
FIXED_ITER=5
FIXED_INIT_STD=0.7
FIXED_SMOOTHING=0.2
MAX_PARALLEL=32

TEMPS=(0.1 0.3 1.0 3.0 10.0)

declare -A FAMILY_SPLIT
FAMILY_SPLIT[open]="test_sim_layout_ood_open"
FAMILY_SPLIT[blocking_easy]="test_sim_layout_ood_blocking_easy"
FAMILY_SPLIT[blocking_medium]="test_sim_layout_ood_blocking_medium"
FAMILY_SPLIT[blocking_hard]="test_sim_layout_ood_blocking_hard"
FAMILY_SPLIT[passage_direct_wide]="test_sim_layout_ood_passage_direct_wide"
FAMILY_SPLIT[passage_direct_medium]="test_sim_layout_ood_passage_direct_medium"
FAMILY_SPLIT[passage_direct_narrow]="test_sim_layout_ood_passage_direct_narrow"
FAMILY_SPLIT[passage_bypass_wide]="test_sim_layout_ood_passage_bypass_wide"
FAMILY_SPLIT[passage_bypass_medium]="test_sim_layout_ood_passage_bypass_medium"
FAMILY_SPLIT[passage_bypass_narrow]="test_sim_layout_ood_passage_bypass_narrow"

# Priority A: 6 hard/bypass templates
PRIORITY_A=(
  "passage_bypass_wide:0"
  "passage_bypass_medium:0"
  "passage_bypass_narrow:0"
  "blocking_hard:0"
  "blocking_hard:1"
  "passage_direct_narrow:0"
)
# Priority B: 6 supplementary
PRIORITY_B=(
  "blocking_hard:5"
  "blocking_hard:9"
  "passage_direct_narrow:4"
  "passage_direct_narrow:7"
  "passage_direct_medium:2"
  "passage_direct_medium:6"
)
# Priority C: 8 easy/sanity
PRIORITY_C=(
  "open:0"
  "open:5"
  "blocking_easy:0"
  "blocking_easy:5"
  "blocking_medium:2"
  "blocking_medium:8"
  "passage_direct_wide:0"
  "passage_direct_wide:5"
)

# ── Init ──
RUN_ROOT="runs/mppi_sweep_fixed_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"; LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR" "$DATA_DIR"

echo "stage,priority,config,family,temperature,status,success,runtime_sec,video_path,log_path,error" > "$MANIFEST"

START_TIME=$(date +%s)

# ── Run a single case ──
run_case() {
  local family="$1" idx="$2" T="$3" stage="$4" priority="$5"
  local split="${FAMILY_SPLIT[$family]}"
  local cfg="mppi_T${T}_${family}_idx${idx}"
  local out="${VIDEO_DIR}/${cfg}.mp4"
  local log="${LOG_DIR}/${cfg}.log"
  local json_tmp="/tmp/mppi_json_${stage}_$$.json"

  local t_start=$(date +%s)
  echo "[Stage $stage] START $cfg T=$T" >&2

  # Fixed: direct stderr redirect instead of process substitution
  if python "$RENDER_SCRIPT" \
    --templates "$TEMPLATE_FILE" --split "$split" --template-index "$idx" \
    --planner-mode mppi --mppi-temperature "$T" \
    --horizon "$FIXED_H" --execute-steps "$FIXED_EXEC" --max-mpc-steps "$FIXED_MPC" \
    --num-samples "$FIXED_SAMPLES" --num-iterations "$FIXED_ITER" \
    --mppi-init-std "$FIXED_INIT_STD" --mppi-smoothing "$FIXED_SMOOTHING" \
    --max-speed-mps "$FIXED_SPEED" --pusher-mass 0.300 \
    --strict-pose-stop --camera topdown --width 1280 --height 720 --fps 10 \
    --out-video "$out" --data-output-dir "$DATA_DIR" \
    > "$json_tmp" 2> "$log"; then

    local t_end=$(date +%s)
    local runtime=$((t_end - t_start))

    if [ -s "$json_tmp" ]; then
      local success
      success=$(python3 -c "import json; print(json.load(open('$json_tmp')).get('success',False))" 2>/dev/null || echo "unknown")
      echo "${stage},${priority},${cfg},${family},${T},completed,${success},${runtime},${out},${log}," >> "$MANIFEST"
      echo "[Stage $stage] DONE $cfg success=$success (${runtime}s)" >&2
    else
      echo "${stage},${priority},${cfg},${family},${T},failed,,${runtime},${out},${log},no_json_output" >> "$MANIFEST"
      echo "[Stage $stage] FAIL $cfg (no JSON, ${runtime}s)" >&2
    fi
  else
    local t_end=$(date +%s)
    local runtime=$((t_end - t_start))
    echo "${stage},${priority},${cfg},${family},${T},error,,${runtime},${out},${log},render_crash" >> "$MANIFEST"
    echo "[Stage $stage] FAIL $cfg (crash, ${runtime}s)" >&2
  fi
  rm -f "$json_tmp"
}

# ── Build full run list (all 3 phases = Priority A + B + C) ──
declare -a RUN_LIST=()
for prio_label in A B C; do
  eval "prio_arr=(\"\${PRIORITY_${prio_label}[@]}\")"
  for entry in "${prio_arr[@]}"; do
    IFS=':' read -r fam idx <<< "$entry"
    for T in "${TEMPS[@]}"; do
      RUN_LIST+=("${prio_label}|${fam}|${idx}|${T}")
    done
  done
done

TOTAL=${#RUN_LIST[@]}
echo "=========================================="
echo "MPPI Sweep Fixed — All Phases"
echo "=========================================="
echo "Run root:  $RUN_ROOT"
echo "Templates: 20 × 5 temps = $TOTAL runs"
echo "Parallel:  $MAX_PARALLEL"
echo "Start:     $(date)"
echo "=========================================="
echo ""

# ── Parallel execution ──
STAGE=0
running=0
for entry in "${RUN_LIST[@]}"; do
  IFS='|' read -r prio_label fam idx T <<< "$entry"

  # Wait if at parallel limit
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n 2>/dev/null || true
    running=$((running - 1))
  fi

  STAGE=$((STAGE + 1))
  elapsed=$(($(date +%s) - START_TIME))
  elapsed_m=$((elapsed / 60))
  echo "[$(date +%H:%M:%S) ${elapsed_m}m] Launch ${STAGE}/${TOTAL}: ${fam} T=${T} (${prio_label})"

  run_case "$fam" "$idx" "$T" "$STAGE" "$prio_label" >> "${RUN_ROOT}/parallel_log.txt" 2>&1 &
  running=$((running + 1))
done

# Wait for all remaining
echo "Waiting for $running remaining jobs..."
wait

# ── Summary ──
ELAPSED=$(($(date +%s) - START_TIME))
COMPLETED=$(grep -c "completed" "$MANIFEST" 2>/dev/null || echo 0)
FAILED=$(grep -c "failed\|error" "$MANIFEST" 2>/dev/null || echo 0)
SUCCESS_TRUE=$(grep "success=True" "$MANIFEST" 2>/dev/null | wc -l || echo 0)

echo ""
echo "============================================================"
echo "Sweep Complete"
echo "============================================================"
echo "Run root:   $RUN_ROOT"
echo "Elapsed:    $((ELAPSED / 60))m"
echo "Total:      $TOTAL"
echo "Completed:  $COMPLETED"
echo "Failed:     $FAILED"
echo "Success:    $SUCCESS_TRUE"
echo "Manifest:   $MANIFEST"
echo ""
echo "Next: python scripts/summarize_mppi_param_sweep_v1.py --run-root $RUN_ROOT"
echo "============================================================"
