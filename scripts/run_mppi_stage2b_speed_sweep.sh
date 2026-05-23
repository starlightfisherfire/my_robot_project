#!/bin/bash
# MPPI Stage 2B: Speed sweep — 3 temperatures × 5 speeds × 8 core templates = 120 runs
# Balanced scheduling: template → speed → temperature
# Config name: mppi_T0.2_sp030_ex10_b100_passage_direct_narrow_idx4
set -eo pipefail

RUN_DIR="runs/mppi_stage2b_speed_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="data/sim/mppi_stage2b_speed"
PYTHON="/home/brucewu/miniconda3/envs/lerobot/bin/python"
TEMPLATES="data/sim/metadata/reset_templates_obstacle_10family_v0.json"
MAX_PARALLEL=31
JOBS_FILE="/tmp/mppi_stage2b_speed_jobs.txt"

# ── Sweep parameters ─────────────────────────────────────────────────────
TEMPERATURES="0.1 0.2 0.3"
SPEEDS_MPS="0.1 0.2 0.3 0.5 0.75"

# core8 templates: split:index
CORE8=(
  "test_sim_layout_ood_passage_direct_narrow:0"
  "test_sim_layout_ood_passage_direct_narrow:4"
  "test_sim_layout_ood_passage_direct_narrow:7"
  "test_sim_layout_ood_passage_bypass_wide:0"
  "test_sim_layout_ood_passage_bypass_medium:0"
  "test_sim_layout_ood_passage_bypass_narrow:0"
  "test_sim_layout_ood_blocking_hard:0"
  "test_sim_layout_ood_blocking_hard:5"
)

# ── Fixed parameters ─────────────────────────────────────────────────────
HORIZON=140
EXECUTE_STEPS=10
MAX_MPC_STEPS=100
NUM_SAMPLES=1024
NUM_ITERATIONS=5
INIT_STD=0.7
SMOOTHING=0.2
PUSHER_MASS=0.300

# ── Modes ─────────────────────────────────────────────────────────────────
MODE="run"
if [ "${1:-}" = "--dry-run" ]; then MODE="dry"; fi
if [ "${1:-}" = "--smoke" ]; then MODE="smoke"; fi
if [ "${1:-}" = "--run" ]; then MODE="run"; fi

speed_to_cms() {
  # Convert mps to cm/s string: 0.1 → 010, 0.75 → 075
  local val
  val=$(printf "%.0f" "$(echo "$1 * 100" | bc -l 2>/dev/null || python3 -c "print(int($1*100))")")
  printf "%03d" "$val"
}

# ── Dry-run ───────────────────────────────────────────────────────────────
if [ "$MODE" = "dry" ]; then
  echo "============================================"
  echo "MPPI Stage 2B Speed Sweep — DRY RUN"
  echo "============================================"
  echo ""
  echo "Grid:"
  echo "  temperatures: $TEMPERATURES"
  echo "  speeds_mps:   $SPEEDS_MPS"
  echo "  templates (core8):"
  for SI in "${CORE8[@]}"; do
    SPLIT="${SI%:*}"
    IDX="${SI#*:}"
    FAM="${SPLIT##*ood_}"
    echo "    $FAM idx$IDX"
  done
  echo ""
  TOTAL=$((3 * 5 * 8))
  echo "Total runs: $TOTAL (3 T × 5 speed × 8 templates)"
  echo ""
  echo "Scheduling order: template → speed → temperature"
  echo ""


  echo "============================================"
  echo "First 20 cases:"
  echo "============================================"
  echo ""
  COUNT=0
  printf "%-5s %-6s %-4s %-7s %-35s %-4s\n" "#" "T" "Sp" "Sp_cm" "Family" "Idx"
  for SI in "${CORE8[@]}"; do
    SPLIT="${SI%:*}"
    IDX="${SI#*:}"
    FAM="${SPLIT##*ood_}"
    for SPEED in $SPEEDS_MPS; do
      SCMS=$(speed_to_cms "$SPEED")
      for T in $TEMPERATURES; do
        COUNT=$((COUNT+1))
        CFG="mppi_T${T}_sp${SCMS}_ex${EXECUTE_STEPS}_b${MAX_MPC_STEPS}_${FAM}_idx${IDX}"
        printf "%-5s T=%-4s sp=%-5s %-7s %-35s idx=%-2s  → %s\n" \
          "#$COUNT" "$T" "$SPEED" "$SCMS" "$FAM" "$IDX" "$CFG"
        if [ $COUNT -ge 20 ]; then break 4; fi
      done
    done
  done
  echo ""
  echo "Dry-run complete. No data generated."
  exit 0
fi

# ── Smoke ──────────────────────────────────────────────────────────────────
if [ "$MODE" = "smoke" ]; then
  echo "============================================"
  echo "MPPI Stage 2B Speed Sweep — SMOKE TEST"
  echo "============================================"
  echo "Running 2 cases: speed=0.2, 0.75 with T=0.2 on blocking_hard idx 0"
  echo "Reduced: horizon=40, num_samples=64, num_iterations=1, max_mpc_steps=2"
  echo ""

  SMOKE_DIR="runs/mppi_stage2b_speed_smoke_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SMOKE_DIR"/{logs,videos}
  mkdir -p "$DATA_DIR"/{episodes,metadata}
  export MUJOCO_GL=egl OMP_NUM_THREADS=1

  SMOKE_SPEEDS="0.2 0.75"
  SMOKE_T="0.2"
  SMOKE_SPLIT="test_sim_layout_ood_blocking_hard"
  SMOKE_IDX="0"

  for SPEED in $SMOKE_SPEEDS; do
    SCMS=$(speed_to_cms "$SPEED")
    CFG="mppi_T${SMOKE_T}_sp${SCMS}_ex10_b2_blocking_hard_idx0"
    LOG="$SMOKE_DIR/logs/${CFG}.log"
    VID="$SMOKE_DIR/videos/${CFG}.mp4"

    echo "--- Smoke: $CFG (speed=$SPEED) ---"
    $PYTHON scripts/render_closed_loop_rollout_mppi_with_data.py \
      --planner-mode mppi --horizon 40 --max-speed-mps $SPEED --pusher-mass $PUSHER_MASS \
      --num-samples 64 --num-iterations 1 --mppi-init-std $INIT_STD --mppi-smoothing $SMOOTHING \
      --mppi-temperature $SMOKE_T --execute-steps 10 --max-mpc-steps 2 \
      --templates $TEMPLATES --split $SMOKE_SPLIT --template-index $SMOKE_IDX \
      --data-output-dir $DATA_DIR --out-video $VID \
      > "$LOG" 2>&1

    # Check for path efficiency metrics in last line
    LAST=$(tail -1 "$LOG" 2>/dev/null || echo "")
    echo "  Checking path efficiency metrics..."
    if echo "$LAST" | $PYTHON -c "
import sys, json
d = json.load(sys.stdin)
required = ['ee_path_length_m', 'object_path_length_m', 'net_progress_m',
            'progress_efficiency_ee', 'wasted_motion_ratio', 'wasted_motion_ratio_capped',
            'random_walk_flag',
            'inefficient_success_flag', 'action_smoothness_mean',
            'action_direction_change_count', 'mean_action_norm',
            'contact_efficiency', 'time_to_success_env_steps']
ok = all(k in d for k in required)
print('  ALL METRICS PRESENT' if ok else '  MISSING: ' + ', '.join(k for k in required if k not in d))
for k in required[:6]:
    print(f'    {k}: {d.get(k, \"N/A\")}')
sys.exit(0 if ok else 1)
" 2>/dev/null; then
      echo "  ✅ PASS"
    else
      echo "  ❌ FAIL — missing path efficiency metrics"
    fi
    echo ""
  done

  echo "=== Smoke complete ==="
  echo "Smoke dir: $SMOKE_DIR"
  exit 0
fi

# ── Full run ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "MPPI Stage 2B — Speed Sweep"
echo "Run dir: $RUN_DIR"
echo "Grid: 3 T × 5 speed × 8 templates = 120 jobs"
echo "Parallel: $MAX_PARALLEL"
echo "=========================================="

mkdir -p "$RUN_DIR"/{logs,videos}
mkdir -p "$DATA_DIR"/{episodes,metadata}
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

> "$JOBS_FILE"

add_job() { echo "$1|$2" >> "$JOBS_FILE"; }

# Balanced scheduling: template → speed → temperature
for SI in "${CORE8[@]}"; do
  SPLIT="${SI%:*}"
  IDX="${SI#*:}"
  FAM="${SPLIT##*ood_}"
  for SPEED in $SPEEDS_MPS; do
    SCMS=$(speed_to_cms "$SPEED")
    for T in $TEMPERATURES; do
      CFG="mppi_T${T}_sp${SCMS}_ex${EXECUTE_STEPS}_b${MAX_MPC_STEPS}_${FAM}_idx${IDX}"
      LOG="$RUN_DIR/logs/${CFG}.log"
      VID="$RUN_DIR/videos/${CFG}.mp4"
      add_job "$CFG" "$PYTHON scripts/render_closed_loop_rollout_mppi_with_data.py \
        --planner-mode mppi \
        --horizon $HORIZON \
        --max-speed-mps $SPEED \
        --pusher-mass $PUSHER_MASS \
        --num-samples $NUM_SAMPLES \
        --num-iterations $NUM_ITERATIONS \
        --mppi-init-std $INIT_STD \
        --mppi-smoothing $SMOOTHING \
        --mppi-temperature $T \
        --execute-steps $EXECUTE_STEPS \
        --max-mpc-steps $MAX_MPC_STEPS \
        --templates $TEMPLATES \
        --split $SPLIT \
        --template-index $IDX \
        --data-output-dir $DATA_DIR \
        --out-video $VID \
        > $LOG 2>&1"
    done
  done
done

TOTAL=$(wc -l < "$JOBS_FILE")
echo "Jobs queued: $TOTAL"
echo "Launching with xargs -P $MAX_PARALLEL..."

START_TS=$(date +%s)

# Run jobs in parallel
cut -d'|' -f2- "$JOBS_FILE" > "$JOBS_FILE.cmds"
xargs -0 -P "$MAX_PARALLEL" -I {} bash -c "{}" < <(tr '\n' '\0' < "$JOBS_FILE.cmds") || true

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "=== All jobs finished in ${ELAPSED}s ==="

# ── Build manifest ─────────────────────────────────────────────────────────
echo "=== Building manifest ==="

MANIFEST="$RUN_DIR/manifest.csv"
cat > "$MANIFEST" << HEREDOC
stage,priority,config,family,type,template_index,template_id,planner_mode,temperature,execute_steps,max_mpc_steps,total_budget,num_samples,num_iterations,init_std,smoothing,speed_mps,speed_cm_s,horizon,pusher_mass,status,success,early_stop_reason,failure_type,initial_pos_dist_m,final_pos_dist_m,best_pos_dist_m,best_pos_dist_step,final_theta_error_deg,best_theta_error_deg,best_theta_error_step,best_pose_score,best_pose_score_step,best_pose_pos_dist_m,best_pose_theta_error_deg,total_progress_m,last_progress_m,success_pos_1mm,success_pos_2mm,success_pos_5mm,success_pos_10mm,success_pos_50mm,success_pose_2mm_10deg,success_pose_5mm_10deg,success_pose_10mm_10deg,success_pose_5mm_5deg,success_pose_10mm_5deg,reached_pos_5mm_once,reached_pos_10mm_once,reached_pose_5mm_10deg_once,reached_pose_10mm_10deg_once,regressed_after_near_success,collision_count,collision_rate,contact_count,mpc_steps,total_env_steps,runtime_sec,ee_path_length_m,object_path_length_m,net_progress_m,progress_efficiency_ee,progress_efficiency_object,wasted_motion_ratio,wasted_motion_ratio_capped,action_smoothness_mean,action_direction_change_count,mean_action_norm,contact_efficiency,time_to_success_env_steps,time_to_near_success_10mm_env_steps,random_walk_flag,inefficient_success_flag,ee_positions_count,object_positions_count,path_includes_initial_position,episode_id,video_path,log_path,error
HEREDOC

SUCCESS_CNT=0; FAIL_CNT=0

while IFS='|' read -r CFG _; do
  [ -z "$CFG" ] && continue
  LOGF=$(find "$RUN_DIR/logs" -name "${CFG}.log" -print -quit 2>/dev/null)
  if [ -z "$LOGF" ] || [ ! -s "$LOGF" ]; then
    echo ",,$CFG,,,,,,,,,,,,,,,,,,failed,False,startup_no_log,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"$LOGF",startup_failed" >> "$MANIFEST"
    FAIL_CNT=$((FAIL_CNT+1))
    continue
  fi
  LAST=$(tail -1 "$LOGF" 2>/dev/null || echo "")
  if echo "$LAST" | $PYTHON -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    echo "$LAST" | $PYTHON -c "
import sys,json,csv,io
d = json.loads(sys.stdin.read())
d['status'] = 'completed'
d['log_path'] = '${LOGF#$RUN_DIR/}'
if 'video_path' not in d: d['video_path'] = ''
if 'planner_mode' not in d: d['planner_mode'] = 'mppi'
if 'horizon' not in d: d['horizon'] = $HORIZON
if 'pusher_mass' not in d: d['pusher_mass'] = $PUSHER_MASS
if 'total_budget' not in d: d['total_budget'] = $EXECUTE_STEPS * $MAX_MPC_STEPS
if 'type' not in d: d['type'] = ''
if 'template_id' not in d: d['template_id'] = ''
if 'smoothing' not in d: d['smoothing'] = $SMOOTHING
if 'init_std' not in d: d['init_std'] = $INIT_STD
if 'stage' not in d: d['stage'] = ''
if 'priority' not in d: d['priority'] = ''
if 'collision_rate' not in d: d['collision_rate'] = ''
if 'error' not in d: d['error'] = ''
# Stage 2B metrics defaults
for k in ['ee_path_length_m','object_path_length_m','net_progress_m',
          'progress_efficiency_ee','progress_efficiency_object','wasted_motion_ratio',
          'wasted_motion_ratio_capped',
          'action_smoothness_mean','action_direction_change_count','mean_action_norm',
          'contact_efficiency','time_to_success_env_steps','time_to_near_success_10mm_env_steps',
          'random_walk_flag','inefficient_success_flag',
          'ee_positions_count','object_positions_count','path_includes_initial_position',
          'speed_mps','speed_cm_s']:
    if k not in d: d[k] = ''
hdr = '$(head -1 "$MANIFEST")'.split(',')
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=hdr, extrasaction='ignore')
w.writerow(d)
print(out.getvalue().strip())
" >> "$MANIFEST" 2>/dev/null && SUCCESS_CNT=$((SUCCESS_CNT+1)) || {
      echo ",,$CFG,,,,,,,,,,,,,,,,,,failed,False,json_parse_error,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"$LOGF",json_parse_error" >> "$MANIFEST"
      FAIL_CNT=$((FAIL_CNT+1))
    }
  else
    ERR_TAIL=$(tail -3 "$LOGF" 2>/dev/null | tr '\n' ' ' | head -c 200)
    echo ",,$CFG,,,,,,,,,,,,,,,,,,failed,False,crash,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"$LOGF",$ERR_TAIL" >> "$MANIFEST"
    FAIL_CNT=$((FAIL_CNT+1))
  fi
done < "$JOBS_FILE"

ROWS=$(tail -n +2 "$MANIFEST" | wc -l)
echo "Manifest: $ROWS rows ($SUCCESS_CNT ok, $FAIL_CNT failed)"

# ── Run summarizer ─────────────────────────────────────────────────────────
if [ $SUCCESS_CNT -gt 0 ]; then
  echo "Running summarizer..."
  $PYTHON scripts/summarize_mppi_stage2b_speed_sweep.py --run-dir "$RUN_DIR" 2>&1
  MD_FILE="$RUN_DIR/stage2b_speed_summary.md"
  if [ -f "$MD_FILE" ]; then
    cp "$MD_FILE" docs/mppi_stage2b_speed_sweep_summary.md
    echo "  Summary copied to docs/mppi_stage2b_speed_sweep_summary.md"
  fi
fi

echo ""
echo "DONE: $RUN_DIR"
echo "SWEEP_COMPLETE $RUN_DIR $MANIFEST"
