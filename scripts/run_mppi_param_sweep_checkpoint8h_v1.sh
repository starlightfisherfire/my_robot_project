#!/usr/bin/env bash
set -eo pipefail
# MPPI Parameter Sweep Checkpoint 8h v1
# Phase 1: 8-hour balanced temperature sweep with transition-level data saving.
#
# Usage:
#   --dry-run        Print planned runs, no execution
#   --smoke          1 quick run to validate pipeline
#   --phase1         Full 8-hour temperature sweep
#   --resume DIR     Resume from previous RUN_ROOT

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u
export PYTHONPATH=. MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ── Parse CLI ──
MODE=""
RESUME_RUN=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dryrun" ;;
    --smoke)   MODE="smoke" ;;
    --phase1)  MODE="phase1" ;;
    --resume)  MODE="resume"; RESUME_RUN="$2"; shift ;;
  esac
  shift 2>/dev/null || true
done

if [ -z "$MODE" ]; then
  echo "Usage: $0 --dry-run | --smoke | --phase1 | --resume RUN_ROOT"
  exit 1
fi

# ── Constants ──
RENDER_SCRIPT="scripts/render_closed_loop_rollout_mppi_with_data.py"
TEMPLATE_FILE="data/sim/metadata/reset_templates_obstacle_10family_v0.json"
DATA_DIR="data/sim/mppi_sweep_v1"

# Phase 1 fixed params (Section H)
FIXED_H=140
FIXED_EXEC=10
FIXED_MPC=100
FIXED_BUDGET=1000
FIXED_SPEED=0.75
FIXED_SAMPLES=1024
FIXED_ITER=5
FIXED_INIT_STD=0.7
FIXED_SMOOTHING=0.2
CEM_WORKERS=32
TEMPS=(0.1 0.3 1.0 3.0 10.0)

# 8h checkpoint (Section J)
MAX_CHECKPOINT_SECONDS=$((8 * 3600))
STOP_LAUNCH_AFTER_SECONDS=$((7 * 3600 + 45 * 60))   # 7h45m

# ── 10-family template mapping (family → split) ──
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

# CORE20 templates (Section H): family:template_index
# Priority A: 6 most critical (hard+bypass) — 5 temps × 6 = 30 runs
PRIORITY_A=(
  "passage_bypass_wide:0"
  "passage_bypass_medium:0"
  "passage_bypass_narrow:0"
  "blocking_hard:0"
  "blocking_hard:1"
  "passage_direct_narrow:0"
)

# Priority B: 6 supplementary (hard/medium) — 5 temps × 6 = 30 runs
PRIORITY_B=(
  "blocking_hard:5"
  "blocking_hard:9"
  "passage_direct_narrow:4"
  "passage_direct_narrow:7"
  "passage_direct_medium:2"
  "passage_direct_medium:6"
)

# Priority C: 8 sanity/easy — 5 temps × 8 = 40 runs
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

# ── Init / Resume ──
if [ "$MODE" = "resume" ]; then
  RUN_ROOT="${RESUME_RUN%/}"
  if [ ! -d "$RUN_ROOT" ]; then
    echo "ERROR: Resume run root not found: $RUN_ROOT"
    exit 1
  fi
  VIDEO_DIR="${RUN_ROOT}/videos"; LOG_DIR="${RUN_ROOT}/logs"
  MANIFEST="${RUN_ROOT}/manifest.csv"
  echo "Resuming from $RUN_ROOT (existing $(wc -l < "$MANIFEST" 2>/dev/null || echo 0) manifest rows)"
else
  RUN_ROOT="runs/mppi_param_sweep_checkpoint8h_v1_$(date +%Y%m%d_%H%M%S)"
  VIDEO_DIR="${RUN_ROOT}/videos"; LOG_DIR="${RUN_ROOT}/logs"
  MANIFEST="${RUN_ROOT}/manifest.csv"
  mkdir -p "$VIDEO_DIR" "$LOG_DIR" "$DATA_DIR"

  # Manifest header (Section K)
  echo "stage,priority,config,family,type,label,planner_mode,temperature,num_samples,num_iterations,init_std,smoothing,speed_mps,speed_cm_s,horizon,execute_steps,max_mpc_steps,total_budget,template_file,split,template_index,template_id,obstacle_count,passage_gap,is_direct,is_bypass,status,success,best_dist,avg_cost,min_cost,collision_count,collision_rate,contact_count,mpc_steps,runtime_sec,effective_sample_size_mean,effective_sample_size_min,weight_entropy_mean,collapse_rate,temperature_collapse_flag,nan_check,episode_id,video_path,log_path,error" > "$MANIFEST"
fi

# ── Helper: build render command ──
build_cmd() {
  local family="$1" idx="$2" T="$3" cfg="$4"
  split="${FAMILY_SPLIT[$family]}"
  out="${VIDEO_DIR}/${cfg}.mp4"
  log="${LOG_DIR}/${cfg}.log"
  echo "$out" "$log" "$split"
}

# ── Helper: parse JSON summary into CSV row ──
parse_summary() {
  local json_file="$1" stage="$2" priority="$3" cfg="$4" family="$5" T="$6" video="$7" log="$8"
  python3 scripts/_parse_mppi_summary.py "$json_file" "$stage" "$priority" "$cfg" "$family" "$T" "$video" "$log"
}

# ── Helper: run a single case ──
run_case() {
  local family="$1" idx="$2" T="$3" stage="$4" priority="$5" phase_label="$6"
  local cfg="mppi_T${T}_${family}_idx${idx}"
  read -r out log split < <(build_cmd "$family" "$idx" "$T" "$cfg")
  local json_tmp="/tmp/mppi_json_$$.json"

  echo "  [${phase_label} ${stage}] $cfg T=${T} split=${split} idx=${idx}"

  if python "$RENDER_SCRIPT" \
    --templates "$TEMPLATE_FILE" --split "$split" --template-index "$idx" \
    --planner-mode mppi --mppi-temperature "$T" \
    --horizon "$FIXED_H" --execute-steps "$FIXED_EXEC" --max-mpc-steps "$FIXED_MPC" \
    --num-samples "$FIXED_SAMPLES" --num-iterations "$FIXED_ITER" \
    --mppi-init-std "$FIXED_INIT_STD" --mppi-smoothing "$FIXED_SMOOTHING" \
    --max-speed-mps "$FIXED_SPEED" --pusher-mass 0.300 \
    --strict-pose-stop --camera topdown --width 1280 --height 720 --fps 10 \
    --out-video "$out" --data-output-dir "$DATA_DIR" \
    > "$json_tmp" 2> >(tee "$log" >&2); then

    if [ -s "$json_tmp" ]; then
      local row; row=$(parse_summary "$json_tmp" "$stage" "$priority" "$cfg" "$family" "$T" "$out" "$log")
      echo "${row}" >> "$MANIFEST"
      local success; success=$(python3 -c "import json; print(json.load(open('$json_tmp')).get('success',False))")
      echo "    ✓ success=$success"
      rm -f "$json_tmp"
      return 0
    else
      echo "${stage},${priority},${cfg},${family},,${family},mppi,${T},${FIXED_SAMPLES},${FIXED_ITER},${FIXED_INIT_STD},${FIXED_SMOOTHING},${FIXED_SPEED},$(echo "${FIXED_SPEED}*100" | bc),${FIXED_H},${FIXED_EXEC},${FIXED_MPC},${FIXED_BUDGET},${TEMPLATE_FILE},${split},${idx},,,,,,,failed,,,,,,,,,,,,,,,${out},${log},no_json_output" >> "$MANIFEST"
      echo "    ✗ no JSON output (failed?)"
      rm -f "$json_tmp"
      return 1
    fi
  else
    echo "${stage},${priority},${cfg},${family},,${family},mppi,${T},${FIXED_SAMPLES},${FIXED_ITER},${FIXED_INIT_STD},${FIXED_SMOOTHING},${FIXED_SPEED},$(echo "${FIXED_SPEED}*100" | bc),${FIXED_H},${FIXED_EXEC},${FIXED_MPC},${FIXED_BUDGET},${TEMPLATE_FILE},${split},${idx},,,,,,,failed,,,,,,,,,,,,,,,${out},${log},render_error" >> "$MANIFEST"
    echo "    ✗ FAILED"
    rm -f "$json_tmp"
    return 1
  fi
}

# ═══════════════════════════════════════════════════════
# MODE: dry-run
# ═══════════════════════════════════════════════════════
if [ "$MODE" = "dryrun" ]; then
  echo "=== MPPI Sweep v1 Dry-Run ==="
  echo ""
  echo "Template file: $TEMPLATE_FILE"
  echo "Data output:   $DATA_DIR"
  echo "Run root:      $RUN_ROOT"
  echo ""
  echo "10-family inventory:"
  for family in "${!FAMILY_SPLIT[@]}"; do
    echo "  $family → ${FAMILY_SPLIT[$family]}"
  done
  echo ""
  echo "Core templates (CORE20):"

  total_runs=0
  print_priority() {
    local label="$1"; shift
    local -n arr=$1
    echo "  Priority $label (${#arr[@]} templates × ${#TEMPS[@]} temps = $((${#arr[@]} * ${#TEMPS[@]})) runs):"
    for entry in "${arr[@]}"; do
      IFS=':' read -r fam idx <<< "$entry"
      echo "    $fam idx=$idx"
    done
    total_runs=$((total_runs + ${#arr[@]} * ${#TEMPS[@]}))
  }

  print_priority "A" PRIORITY_A
  print_priority "B" PRIORITY_B
  print_priority "C" PRIORITY_C
  echo "  Total planned: $total_runs runs"

  echo ""
  echo "Scheduling: interleaved temperature within each priority group"
  echo "  For each priority group: for each template: for T in ${TEMPS[*]}: run"
  echo ""
  echo "First 20 cases:"
  stage=0
  for prio_label in A B C; do
    prio_ref="PRIORITY_${prio_label}[@]"
    prio_arr=("${!prio_ref}")
    for entry in "${prio_arr[@]}"; do
      IFS=':' read -r fam idx <<< "$entry"
      for T in "${TEMPS[@]}"; do
        stage=$((stage + 1))
        if [ $stage -le 20 ]; then
          cfg="mppi_T${T}_${fam}_idx${idx}"
          echo "  ${stage}. $cfg --split ${FAMILY_SPLIT[$fam]} --template-index $idx --mppi-temperature $T"
        fi
      done
    done
  done

  echo ""
  echo "8h checkpoint: STOP_LAUNCH_AFTER=$((STOP_LAUNCH_AFTER_SECONDS / 3600))h$(((STOP_LAUNCH_AFTER_SECONDS % 3600) / 60))m"
  echo "✅ Dry-run complete. No files written."
  exit 0
fi

# ═══════════════════════════════════════════════════════
# MODE: smoke
# ═══════════════════════════════════════════════════════
if [ "$MODE" = "smoke" ]; then
  echo "=== MPPI Sweep v1 Smoke Test ==="
  echo ""
  echo "Params: T=1.0, samples=64, iter=1, init_std=0.7, horizon=40, exec=5, mpc=2"
  echo "Template: blocking_hard idx=0"
  echo ""

  cfg="smoke_test"
  out="${VIDEO_DIR}/smoke_test.mp4"
  log="${LOG_DIR}/smoke_test.log"
  split="${FAMILY_SPLIT[blocking_hard]}"

  if python "$RENDER_SCRIPT" \
    --templates "$TEMPLATE_FILE" --split "$split" --template-index 0 \
    --planner-mode mppi --mppi-temperature 1.0 \
    --horizon 40 --execute-steps 5 --max-mpc-steps 2 \
    --num-samples 64 --num-iterations 1 \
    --mppi-init-std 0.7 --mppi-smoothing 0.2 \
    --max-speed-mps "$FIXED_SPEED" --pusher-mass 0.300 \
    --strict-pose-stop --camera topdown --width 1280 --height 720 --fps 10 \
    --out-video "$out" --data-output-dir "$DATA_DIR" \
    > /tmp/mppi_smoke.json 2> >(tee "$log" >&2); then

    json_output=$(cat /tmp/mppi_smoke.json 2>/dev/null)
    echo ""
    echo "--- Smoke Test Results ---"
    checks_passed=0; checks_total=0

    check() { checks_total=$((checks_total+1)); if eval "$1"; then echo "  ✅ $2"; checks_passed=$((checks_passed+1)); else echo "  ❌ $2"; fi; }

    check "[ -f \"$out\" ]" "1. Video exists"
    check "[ -f \"$log\" ]" "2. Log exists"
    check "[ -s \"$MANIFEST\" ]" "3. Manifest file exists"

    ep_id=$(python3 -c "import json; print(json.load(open('/tmp/mppi_smoke.json')).get('episode_id',''))" 2>/dev/null || echo "")
    ep_npz="${DATA_DIR}/episodes/${ep_id}.npz"
    check "[ -n \"$ep_id\" ] && [ -f \"$ep_npz\" ]" "4. Episode npz exists (${ep_id})"

    meta_file="${DATA_DIR}/metadata/episodes.jsonl"
    check "[ -f \"$meta_file\" ] && [ \$(wc -l < \"$meta_file\") -ge 1 ]" "5. episodes.jsonl has row"

    nan_ok=$(python3 -c "import json; print(json.load(open('/tmp/mppi_smoke.json')).get('nan_check','FAIL'))" 2>/dev/null || echo "FAIL")
    check "[ \"$nan_ok\" = \"PASS\" ]" "6. No NaN in states/actions/next_states"

    shape_ok=$(python3 -c "
import numpy as np; d=np.load('$ep_npz');
s=d['states']; a=d['actions_norm']; ns=d['next_states'];
print('OK' if s.ndim==2 and a.ndim==2 and ns.ndim==2 and s.shape[1]==5 and a.shape[1]==2 and ns.shape[1]==5 else 'FAIL')
" 2>/dev/null || echo "FAIL")
    check "[ \"$shape_ok\" = \"OK\" ]" "7. States/actions/next_states shapes correct"

    no_tb=$(grep -c "Traceback (most recent call last):" "$log" 2>/dev/null | head -1 || true); no_tb=${no_tb:-0}
    check "[ \"$no_tb\" = \"0\" ]" "8. No Traceback in log (EGL cleanup allowed)"

    echo ""
    echo "Smoke result: ${checks_passed}/${checks_total} checks passed"

    if [ "$checks_passed" -eq "$checks_total" ]; then
      echo "✅ SMOKE TEST PASSED — Pipeline ready for Phase 1"
    else
      echo "❌ SMOKE TEST FAILED — Fix issues before --phase1"
      exit 1
    fi
  else
    echo "❌ Smoke test render FAILED"
    exit 1
  fi
  exit 0
fi

# ═══════════════════════════════════════════════════════
# MODE: phase1 (Section H+I+J)
# ═══════════════════════════════════════════════════════
if [ "$MODE" = "phase1" ]; then
  echo "=== MPPI Sweep Phase 1: 8-hour temperature sweep ==="
  echo "Run root: $RUN_ROOT"
  echo "Start time: $(date)"
  echo ""

  START_TIME=$(date +%s)
  STAGE=0
  COMPLETED=0
  FAILED=0
  SKIPPED=0
  STOPPED_EARLY=0

  # Build ordered run list: Priority A → B → C, interleave temperature within
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

  TOTAL_PLANNED=${#RUN_LIST[@]}
  MAX_PARALLEL=24
  echo "Planned: ${TOTAL_PLANNED} runs across Priority A/B/C (parallel=$MAX_PARALLEL)"
  echo "Checkpoint: stop launching after ${STOP_LAUNCH_AFTER_SECONDS}s (~7h45m)"
  echo ""

  # Parallel execution with wait -n
  STAGE=0
  running=0
  COMPLETED=0
  FAILED=0
  for entry in "${RUN_LIST[@]}"; do
    IFS='|' read -r prio_label fam idx T <<< "$entry"

    # Checkpoint guard
    elapsed=$(($(date +%s) - START_TIME))
    if [ "$elapsed" -gt "$STOP_LAUNCH_AFTER_SECONDS" ]; then
      echo "[$(date +%H:%M:%S)] CHECKPOINT: stopping launches"
      STOPPED_EARLY=1
      break
    fi

    # Wait if at parallel limit
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
      wait -n
      running=$((running - 1))
    fi

    STAGE=$((STAGE + 1))
    elapsed_m=$((elapsed / 60))
    echo "[$(date +%H:%M:%S) ${elapsed_m}m ${STAGE}/${TOTAL_PLANNED}] Launch ${fam} T=${T}"

    # Launch in background
    run_case "$fam" "$idx" "$T" "$STAGE" "$prio_label" "P1" >> "${RUN_ROOT}/parallel_log.txt" 2>&1 &
    running=$((running + 1))
  done

  # Wait for all remaining
  echo "Waiting for $running remaining jobs..."
  wait

  # Count results
  COMPLETED=$(grep -c '✓ success=' "${RUN_ROOT}/parallel_log.txt" 2>/dev/null || echo 0)
  FAILED=$(grep -c '✗' "${RUN_ROOT}/parallel_log.txt" 2>/dev/null || echo 0)

  if [ "$STOPPED_EARLY" -eq 1 ]; then
    SKIPPED=$((TOTAL_PLANNED - STAGE))
  fi

  # ── Summary ──
  ELAPSED_TOTAL=$(($(date +%s) - START_TIME))
  echo ""
  echo "============================================================"
  echo "Phase 1 Checkpoint Summary"
  echo "============================================================"
  echo "Run root:      $RUN_ROOT"
  echo "Total elapsed: $((ELAPSED_TOTAL / 60))m $((ELAPSED_TOTAL % 60))s"
  echo "Parallel:      $MAX_PARALLEL"
  echo "Launched:      $STAGE"
  echo "Completed:     $COMPLETED"
  echo "Failed:        $FAILED"
  echo "Skipped:       $SKIPPED"
  echo "Episodes:      $(ls ${DATA_DIR}/episodes/*.npz 2>/dev/null | wc -l)"
  echo ""
  echo "Next: run summarizer:"
  echo "  python scripts/summarize_mppi_param_sweep_v1.py --run-root $RUN_ROOT"
  echo "============================================================"
  exit 0
fi

echo "Unknown mode: $MODE"
exit 1
