#!/bin/bash
# MPPI Stage 2C: Horizon × Num_Samples × Init_Std sweep
# 3 top_configs × 3 horizons × 2 num_samples × 3 init_std × 8 core templates = 432 runs
# Balanced scheduling: template → top_config → horizon → num_samples → init_std
# Config name: mppi_sp030_T0.1_h120_n2048_std0.7_ex10_b100_passage_direct_narrow_idx4
set -eo pipefail

RUN_DIR="runs/mppi_stage2c_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="data/sim/mppi_stage2c"
PYTHON="/home/brucewu/miniconda3/envs/lerobot/bin/python"
TEMPLATES="data/sim/metadata/reset_templates_obstacle_10family_v0.json"
MAX_PARALLEL=8
JOBS_FILE="/tmp/mppi_stage2c_jobs.txt"

# ── Sweep parameters ─────────────────────────────────────────────────────
HORIZONS="100 120 140"
NUM_SAMPLES_LIST="1024 2048"
INIT_STDS="0.5 0.7 1.0"

# Top configs from Stage 2B
declare -A TOP_CONFIGS
TOP_CONFIGS[A]="sp030_T0.1:0.3:0.1"
TOP_CONFIGS[B]="sp050_T0.2:0.5:0.2"
TOP_CONFIGS[C]="sp020_T0.3:0.2:0.3"

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
EXECUTE_STEPS=10
MAX_MPC_STEPS=100
NUM_ITERATIONS=5
SMOOTHING=0.2
PUSHER_MASS=0.300

# ── Modes ─────────────────────────────────────────────────────────────────
MODE="run"
RESUME_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE="dry"; shift ;;
    --smoke) MODE="smoke"; shift ;;
    --run) MODE="run"; shift ;;
    --resume) RESUME_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

speed_to_cms() {
  local val
  val=$(printf "%.0f" "$(echo "$1 * 100" | bc -l 2>/dev/null || python3 -c "print(int($1*100))")")
  printf "%03d" "$val"
}

# ── Dry-run ───────────────────────────────────────────────────────────────
if [ "$MODE" = "dry" ]; then
  echo "============================================"
  echo "MPPI Stage 2C Horizon × Samples × Std — DRY RUN"
  echo "============================================"
  echo ""
  echo "Grid:"
  echo "  horizons:     $HORIZONS"
  echo "  num_samples:  $NUM_SAMPLES_LIST"
  echo "  init_std:     $INIT_STDS"
  echo "  top_configs:  A (sp030_T0.1), B (sp050_T0.2), C (sp020_T0.3)"
  echo "  templates (core8):"
  for SI in "${CORE8[@]}"; do
    SPLIT="${SI%:*}"
    IDX="${SI#*:}"
    FAM="${SPLIT##*ood_}"
    echo "    $FAM idx$IDX"
  done
  echo ""
  TOTAL=$((3 * 3 * 2 * 3 * 8))
  echo "Total runs: $TOTAL (3 configs × 3 horizons × 2 num_samples × 3 init_std × 8 templates)"
  echo ""
  echo "Scheduling order: template → top_config → horizon → num_samples → init_std"
  echo ""

  echo "============================================"
  echo "First 30 cases:"
  echo "============================================"
  echo ""
  COUNT=0
  printf "%-5s %-8s %-4s %-5s %-4s %-7s %-35s %-4s\n" "#" "Cfg" "H" "N" "Std" "Sp_cm" "Family" "Idx"
  for SI in "${CORE8[@]}"; do
    SPLIT="${SI%:*}"
    IDX="${SI#*:}"
    FAM="${SPLIT##*ood_}"
    for TC_KEY in A B C; do
      IFS=':' read -r TC_LABEL TC_SPEED TC_TEMP <<< "${TOP_CONFIGS[$TC_KEY]}"
      SCMS=$(speed_to_cms "$TC_SPEED")
      for H in $HORIZONS; do
        for N in $NUM_SAMPLES_LIST; do
          for STD in $INIT_STDS; do
            COUNT=$((COUNT+1))
            CFG="mppi_${TC_LABEL}_h${H}_n${N}_std${STD}_ex${EXECUTE_STEPS}_b${MAX_MPC_STEPS}_${FAM}_idx${IDX}"
            printf "%-5s %-8s H=%-3s N=%-5s std=%-4s %-7s %-35s idx=%-2s  → %s\n" \
              "#$COUNT" "$TC_KEY" "$H" "$N" "$STD" "$SCMS" "$FAM" "$IDX" "$CFG"
            if [ $COUNT -ge 30 ]; then break 6; fi
          done
        done
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
  echo "MPPI Stage 2C — SMOKE TEST"
  echo "============================================"
  echo "Running 4 smoke cases with reduced params"
  echo "  num_iterations=1, max_mpc_steps=2, num_samples=64"
  echo ""

  SMOKE_DIR="runs/mppi_stage2c_smoke_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SMOKE_DIR"/{logs,videos}
  mkdir -p "$DATA_DIR"/{episodes,metadata}
  export MUJOCO_GL=egl OMP_NUM_THREADS=1

  SMOKE_TEMPLATE="test_sim_layout_ood_blocking_hard"
  SMOKE_IDX="0"
  SMOKE_SAMPLES=64
  SMOKE_ITER=1
  SMOKE_MPC=2

  # Case 1: A, H=100, std=0.5
  # Case 2: A, H=140, std=1.0
  # Case 3: B, H=120, std=0.7
  # Case 4: C, H=120, std=0.7
  declare -a SMOKE_CASES=(
    "A:100:0.5"
    "A:140:1.0"
    "B:120:0.7"
    "C:120:0.7"
  )

  MANIFEST="$SMOKE_DIR/manifest.csv"
  cat > "$MANIFEST" << 'HEREDOC'
stage,top_config_label,config,family,type,template_index,template_id,planner_mode,temperature,speed_mps,speed_cm_s,horizon,num_samples,num_iterations,init_std,smoothing,execute_steps,max_mpc_steps,total_budget,pusher_mass,status,success,early_stop_reason,failure_type,initial_pos_dist_m,final_pos_dist_m,best_pos_dist_m,best_pos_dist_step,final_theta_error_deg,best_theta_error_deg,best_theta_error_step,best_pose_score,best_pose_score_step,best_pose_pos_dist_m,best_pose_theta_error_deg,total_progress_m,last_progress_m,success_pos_1mm,success_pos_2mm,success_pos_5mm,success_pos_10mm,success_pos_50mm,success_pose_2mm_10deg,success_pose_5mm_10deg,success_pose_10mm_10deg,success_pose_5mm_5deg,success_pose_10mm_5deg,reached_pos_5mm_once,reached_pos_10mm_once,reached_pose_5mm_10deg_once,reached_pose_10mm_10deg_once,regressed_after_near_success,collision_count,collision_rate,contact_count,mpc_steps,total_env_steps,runtime_sec,ee_path_length_m,object_path_length_m,net_progress_m,progress_efficiency_ee,progress_efficiency_object,object_to_ee_motion_ratio,wasted_motion_ratio,wasted_motion_ratio_capped,action_smoothness_mean,action_direction_change_count,mean_action_norm,contact_efficiency,time_to_success_env_steps,time_to_near_success_10mm_env_steps,random_walk_flag,inefficient_success_flag,excessive_wander_flag,clean_success_flag,early_ee_path_length_m,middle_ee_path_length_m,late_ee_path_length_m,early_object_path_length_m,middle_object_path_length_m,late_object_path_length_m,early_progress_m,middle_progress_m,late_progress_m,early_progress_efficiency_ee,middle_progress_efficiency_ee,late_progress_efficiency_ee,early_action_direction_change_count,middle_action_direction_change_count,late_action_direction_change_count,early_contact_count,middle_contact_count,late_contact_count,early_best_dist_improvement_m,middle_best_dist_improvement_m,late_best_dist_improvement_m,late_breakthrough_flag,front_loaded_wander_flag,meaningless_exploration_flag,path_includes_initial_position,ee_positions_count,object_positions_count,effective_sample_size_mean,weight_entropy_mean,collapse_rate,episode_id,video_path,log_path,error
HEREDOC

  for CASE in "${SMOKE_CASES[@]}"; do
    IFS=':' read -r TC_KEY H STD <<< "$CASE"
    IFS=':' read -r TC_LABEL TC_SPEED TC_TEMP <<< "${TOP_CONFIGS[$TC_KEY]}"
    SCMS=$(speed_to_cms "$TC_SPEED")
    CFG="mppi_${TC_LABEL}_h${H}_n${SMOKE_SAMPLES}_std${STD}_ex${EXECUTE_STEPS}_b${SMOKE_MPC}_blocking_hard_idx${SMOKE_IDX}"
    LOG="$SMOKE_DIR/logs/${CFG}.log"
    VID="$SMOKE_DIR/videos/${CFG}.mp4"

    echo "--- Smoke: $CFG (config=$TC_KEY, H=$H, N=$SMOKE_SAMPLES, std=$STD) ---"
    $PYTHON scripts/render_closed_loop_rollout_mppi_with_data.py \
      --planner-mode mppi \
      --horizon $H \
      --max-speed-mps $TC_SPEED \
      --pusher-mass $PUSHER_MASS \
      --num-samples $SMOKE_SAMPLES \
      --num-iterations $SMOKE_ITER \
      --mppi-init-std $STD \
      --mppi-smoothing $SMOOTHING \
      --mppi-temperature $TC_TEMP \
      --execute-steps $EXECUTE_STEPS \
      --max-mpc-steps $SMOKE_MPC \
      --templates $TEMPLATES \
      --split $SMOKE_TEMPLATE \
      --template-index $SMOKE_IDX \
      --data-output-dir $DATA_DIR \
      --out-video $VID \
      > "$LOG" 2>&1 || true

    # Parse JSON and append to manifest
    LAST=$(tail -1 "$LOG" 2>/dev/null || echo "")
    if echo "$LAST" | $PYTHON -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
      echo "$LAST" | $PYTHON -c "
import sys,json,csv,io
d = json.loads(sys.stdin.read())
d['stage'] = 'stage2c'
d['status'] = 'completed'
d['top_config_label'] = '$TC_KEY'
d['log_path'] = '${LOG#$SMOKE_DIR/}'
for k in ['video_path','planner_mode','horizon','pusher_mass','total_budget',
          'type','template_id','smoothing','init_std','speed_mps','speed_cm_s',
          'ee_path_length_m','object_path_length_m','net_progress_m',
          'progress_efficiency_ee','progress_efficiency_object','object_to_ee_motion_ratio',
          'wasted_motion_ratio','wasted_motion_ratio_capped',
          'action_smoothness_mean','action_direction_change_count','mean_action_norm',
          'contact_efficiency','time_to_success_env_steps','time_to_near_success_10mm_env_steps',
          'random_walk_flag','inefficient_success_flag','excessive_wander_flag','clean_success_flag',
          'early_ee_path_length_m','middle_ee_path_length_m','late_ee_path_length_m',
          'early_object_path_length_m','middle_object_path_length_m','late_object_path_length_m',
          'early_progress_m','middle_progress_m','late_progress_m',
          'early_progress_efficiency_ee','middle_progress_efficiency_ee','late_progress_efficiency_ee',
          'early_action_direction_change_count','middle_action_direction_change_count','late_action_direction_change_count',
          'early_contact_count','middle_contact_count','late_contact_count',
          'early_best_dist_improvement_m','middle_best_dist_improvement_m','late_best_dist_improvement_m',
          'late_breakthrough_flag','front_loaded_wander_flag','meaningless_exploration_flag',
          'path_includes_initial_position','ee_positions_count','object_positions_count',
          'effective_sample_size_mean','weight_entropy_mean','collapse_rate']:
    if k not in d: d[k] = ''
hdr = '$(head -1 "$MANIFEST")'.split(',')
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=hdr, extrasaction='ignore')
w.writerow(d)
print(out.getvalue().strip())
" >> "$MANIFEST" 2>/dev/null
      echo "  ✅ PASS"
    else
      echo "  ❌ FAIL — no valid JSON output"
    fi
    echo ""
  done

  # Run summarizer on smoke
  echo "Running summarizer on smoke..."
  $PYTHON scripts/summarize_mppi_stage2c_horizon_samples_std.py --run-dir "$SMOKE_DIR" 2>&1 || true

  echo ""
  echo "=== Smoke complete ==="
  echo "Smoke dir: $SMOKE_DIR"
  echo "Manifest: $MANIFEST"
  exit 0
fi

# ── Full run ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "MPPI Stage 2C — Horizon × Samples × Std"
echo "Run dir: $RUN_DIR"
echo "Grid: 3 configs × 3 horizons × 2 num_samples × 3 init_std × 8 templates = 432 jobs"
echo "Parallel: $MAX_PARALLEL"
echo "=========================================="

mkdir -p "$RUN_DIR"/{logs,videos}
mkdir -p "$DATA_DIR"/{episodes,metadata}
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

> "$JOBS_FILE"

add_job() { echo "$1|$2" >> "$JOBS_FILE"; }

# Balanced scheduling: template → top_config → horizon → num_samples → init_std
for SI in "${CORE8[@]}"; do
  SPLIT="${SI%:*}"
  IDX="${SI#*:}"
  FAM="${SPLIT##*ood_}"
  for TC_KEY in A B C; do
    IFS=':' read -r TC_LABEL TC_SPEED TC_TEMP <<< "${TOP_CONFIGS[$TC_KEY]}"
    SCMS=$(speed_to_cms "$TC_SPEED")
    for H in $HORIZONS; do
      for N in $NUM_SAMPLES_LIST; do
        for STD in $INIT_STDS; do
          CFG="mppi_${TC_LABEL}_h${H}_n${N}_std${STD}_ex${EXECUTE_STEPS}_b${MAX_MPC_STEPS}_${FAM}_idx${IDX}"
          LOG="$RUN_DIR/logs/${CFG}.log"
          VID="$RUN_DIR/videos/${CFG}.mp4"
          add_job "$CFG" "$PYTHON scripts/render_closed_loop_rollout_mppi_with_data.py \
            --planner-mode mppi \
            --horizon $H \
            --max-speed-mps $TC_SPEED \
            --pusher-mass $PUSHER_MASS \
            --num-samples $N \
            --num-iterations $NUM_ITERATIONS \
            --mppi-init-std $STD \
            --mppi-smoothing $SMOOTHING \
            --mppi-temperature $TC_TEMP \
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
cat > "$MANIFEST" << 'HEREDOC'
stage,top_config_label,config,family,type,template_index,template_id,planner_mode,temperature,speed_mps,speed_cm_s,horizon,num_samples,num_iterations,init_std,smoothing,execute_steps,max_mpc_steps,total_budget,pusher_mass,status,success,early_stop_reason,failure_type,initial_pos_dist_m,final_pos_dist_m,best_pos_dist_m,best_pos_dist_step,final_theta_error_deg,best_theta_error_deg,best_theta_error_step,best_pose_score,best_pose_score_step,best_pose_pos_dist_m,best_pose_theta_error_deg,total_progress_m,last_progress_m,success_pos_1mm,success_pos_2mm,success_pos_5mm,success_pos_10mm,success_pos_50mm,success_pose_2mm_10deg,success_pose_5mm_10deg,success_pose_10mm_10deg,success_pose_5mm_5deg,success_pose_10mm_5deg,reached_pos_5mm_once,reached_pos_10mm_once,reached_pose_5mm_10deg_once,reached_pose_10mm_10deg_once,regressed_after_near_success,collision_count,collision_rate,contact_count,mpc_steps,total_env_steps,runtime_sec,ee_path_length_m,object_path_length_m,net_progress_m,progress_efficiency_ee,progress_efficiency_object,object_to_ee_motion_ratio,wasted_motion_ratio,wasted_motion_ratio_capped,action_smoothness_mean,action_direction_change_count,mean_action_norm,contact_efficiency,time_to_success_env_steps,time_to_near_success_10mm_env_steps,random_walk_flag,inefficient_success_flag,excessive_wander_flag,clean_success_flag,early_ee_path_length_m,middle_ee_path_length_m,late_ee_path_length_m,early_object_path_length_m,middle_object_path_length_m,late_object_path_length_m,early_progress_m,middle_progress_m,late_progress_m,early_progress_efficiency_ee,middle_progress_efficiency_ee,late_progress_efficiency_ee,early_action_direction_change_count,middle_action_direction_change_count,late_action_direction_change_count,early_contact_count,middle_contact_count,late_contact_count,early_best_dist_improvement_m,middle_best_dist_improvement_m,late_best_dist_improvement_m,late_breakthrough_flag,front_loaded_wander_flag,meaningless_exploration_flag,path_includes_initial_position,ee_positions_count,object_positions_count,effective_sample_size_mean,weight_entropy_mean,collapse_rate,episode_id,video_path,log_path,error
HEREDOC

SUCCESS_CNT=0; FAIL_CNT=0

while IFS='|' read -r CFG _; do
  [ -z "$CFG" ] && continue
  LOGF=$(find "$RUN_DIR/logs" -name "${CFG}.log" -print -quit 2>/dev/null)
  if [ -z "$LOGF" ] || [ ! -s "$LOGF" ]; then
    echo "stage2c,,,,,,,,,,,,,,,,,,,,,,failed,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"$LOGF",startup_failed" >> "$MANIFEST"
    FAIL_CNT=$((FAIL_CNT+1))
    continue
  fi
  LAST=$(tail -1 "$LOGF" 2>/dev/null || echo "")
  if echo "$LAST" | $PYTHON -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    echo "$LAST" | $PYTHON -c "
import sys,json,csv,io
d = json.loads(sys.stdin.read())
d['stage'] = 'stage2c'
d['status'] = 'completed'
d['log_path'] = '${LOGF#$RUN_DIR/}'
# Extract top_config_label from config name
cfg = d.get('config','')
for tc in ['sp030_T0.1','sp050_T0.2','sp020_T0.3']:
    if tc in cfg:
        d['top_config_label'] = tc.split('_')[0] + '_' + tc.split('_')[1]
        break
for k in ['video_path','planner_mode','horizon','pusher_mass','total_budget',
          'type','template_id','smoothing','init_std','speed_mps','speed_cm_s',
          'ee_path_length_m','object_path_length_m','net_progress_m',
          'progress_efficiency_ee','progress_efficiency_object','object_to_ee_motion_ratio',
          'wasted_motion_ratio','wasted_motion_ratio_capped',
          'action_smoothness_mean','action_direction_change_count','mean_action_norm',
          'contact_efficiency','time_to_success_env_steps','time_to_near_success_10mm_env_steps',
          'random_walk_flag','inefficient_success_flag','excessive_wander_flag','clean_success_flag',
          'early_ee_path_length_m','middle_ee_path_length_m','late_ee_path_length_m',
          'early_object_path_length_m','middle_object_path_length_m','late_object_path_length_m',
          'early_progress_m','middle_progress_m','late_progress_m',
          'early_progress_efficiency_ee','middle_progress_efficiency_ee','late_progress_efficiency_ee',
          'early_action_direction_change_count','middle_action_direction_change_count','late_action_direction_change_count',
          'early_contact_count','middle_contact_count','late_contact_count',
          'early_best_dist_improvement_m','middle_best_dist_improvement_m','late_best_dist_improvement_m',
          'late_breakthrough_flag','front_loaded_wander_flag','meaningless_exploration_flag',
          'path_includes_initial_position','ee_positions_count','object_positions_count',
          'effective_sample_size_mean','weight_entropy_mean','collapse_rate']:
    if k not in d: d[k] = ''
hdr = '$(head -1 "$MANIFEST")'.split(',')
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=hdr, extrasaction='ignore')
w.writerow(d)
print(out.getvalue().strip())
" >> "$MANIFEST" 2>/dev/null && SUCCESS_CNT=$((SUCCESS_CNT+1)) || {
      echo "stage2c,,,,,,,,,,,,,,,,,,,,,,failed,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"$LOGF",json_parse_error" >> "$MANIFEST"
      FAIL_CNT=$((FAIL_CNT+1))
    }
  else
    ERR_TAIL=$(tail -3 "$LOGF" 2>/dev/null | tr '\n' ' ' | head -c 200)
    echo "stage2c,,,,,,,,,,,,,,,,,,,,,,failed,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"$LOGF",$ERR_TAIL" >> "$MANIFEST"
    FAIL_CNT=$((FAIL_CNT+1))
  fi
done < "$JOBS_FILE"

ROWS=$(tail -n +2 "$MANIFEST" | wc -l)
echo "Manifest: $ROWS rows ($SUCCESS_CNT ok, $FAIL_CNT failed)"

# ── Run summarizer ─────────────────────────────────────────────────────────
if [ $SUCCESS_CNT -gt 0 ]; then
  echo "Running summarizer..."
  $PYTHON scripts/summarize_mppi_stage2c_horizon_samples_std.py --run-dir "$RUN_DIR" 2>&1
  MD_FILE="$RUN_DIR/stage2c_horizon_samples_std_summary.md"
  if [ -f "$MD_FILE" ]; then
    cp "$MD_FILE" docs/mppi_stage2c_horizon_samples_std_summary.md
    echo "  Summary copied to docs/mppi_stage2c_horizon_samples_std_summary.md"
  fi
fi

echo ""
echo "DONE: $RUN_DIR"
echo "SWEEP_COMPLETE $RUN_DIR $MANIFEST"
