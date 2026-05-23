#!/bin/bash
# Resume Stage 2C sweep — skip already completed runs
set -o pipefail

RUN_DIR="${1:-runs/mppi_stage2c_20260520_194856}"
DATA_DIR="data/sim/mppi_stage2c"
PYTHON="/home/brucewu/miniconda3/envs/lerobot/bin/python"
TEMPLATES="data/sim/metadata/reset_templates_obstacle_10family_v0.json"
MAX_PARALLEL=30
JOBS_FILE="/tmp/mppi_stage2c_resume_jobs.txt"

# Sweep parameters
HORIZONS="100 120 140"
NUM_SAMPLES_LIST="1024 2048"
INIT_STDS="0.5 0.7 1.0"
EXECUTE_STEPS=10
MAX_MPC_STEPS=100
NUM_ITERATIONS=5
SMOOTHING=0.2
PUSHER_MASS=0.300

declare -A TOP_CONFIGS
TOP_CONFIGS[A]="sp030_T0.1:0.3:0.1"
TOP_CONFIGS[B]="sp050_T0.2:0.5:0.2"
TOP_CONFIGS[C]="sp020_T0.3:0.2:0.3"

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

speed_to_cms() {
  local val
  val=$(printf "%.0f" "$(echo "$1 * 100" | bc -l 2>/dev/null || python3 -c "print(int($1*100))")")
  printf "%03d" "$val"
}

is_completed() {
  local cfg="$1"
  local logf="$RUN_DIR/logs/${cfg}.log"
  [ -f "$logf" ] && [ -s "$logf" ] && \
    tail -1 "$logf" 2>/dev/null | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('ee_path_length_m','') != ''" 2>/dev/null
}

# Count existing
EXISTING=$(ls "$RUN_DIR/logs/"*.log 2>/dev/null | wc -l)
COMPLETED=0
for f in "$RUN_DIR/logs/"*.log; do
  [ -s "$f" ] && tail -1 "$f" 2>/dev/null | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('ee_path_length_m','') != ''" 2>/dev/null && COMPLETED=$((COMPLETED+1))
done

echo "=========================================="
echo "MPPI Stage 2C — Resume Sweep"
echo "Run dir: $RUN_DIR"
echo "Already completed: $COMPLETED / 432"
echo "=========================================="

# Build job list (skip completed)
> "$JOBS_FILE"
TOTAL=0; SKIPPED=0

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
          if is_completed "$CFG"; then
            SKIPPED=$((SKIPPED+1))
          else
            LOG="$RUN_DIR/logs/${CFG}.log"
            echo "$CFG|$PYTHON scripts/render_closed_loop_rollout_mppi_with_data.py \
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
              > $LOG 2>&1" >> "$JOBS_FILE"
            TOTAL=$((TOTAL+1))
          fi
        done
      done
    done
  done
done

echo "To run: $TOTAL"
echo "Skipped (already done): $SKIPPED"
echo ""

if [ "$TOTAL" -eq 0 ]; then
  echo "All 432 jobs completed! Building manifest..."
  bash scripts/run_mppi_stage2c_horizon_samples_std_sweep.sh --run 2>&1 | tail -5
  exit 0
fi

# Run remaining jobs
mkdir -p "$RUN_DIR"/{logs,videos}
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

START_TS=$(date +%s)
echo "Launching $TOTAL jobs with $MAX_PARALLEL parallel..."

cut -d'|' -f2- "$JOBS_FILE" > "$JOBS_FILE.cmds"
xargs -0 -P "$MAX_PARALLEL" -I {} bash -c "{}" < <(tr '\n' '\0' < "$JOBS_FILE.cmds") || true

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "=== Resume sweep finished in ${ELAPSED}s ==="

# Count results
NEW_OK=0; NEW_FAIL=0
while IFS='|' read -r CFG _; do
  [ -z "$CFG" ] && continue
  LOGF="$RUN_DIR/logs/${CFG}.log"
  if [ -s "$LOGF" ] && tail -1 "$LOGF" 2>/dev/null | $PYTHON -c "import sys,json; d=json.load(sys.stdin); assert d.get('ee_path_length_m','') != ''" 2>/dev/null; then
    NEW_OK=$((NEW_OK+1))
  else
    NEW_FAIL=$((NEW_FAIL+1))
  fi
done < "$JOBS_FILE"

echo "New completed: $NEW_OK"
echo "New failed: $NEW_FAIL"
TOTAL_OK=$((COMPLETED + NEW_OK))
echo "Total completed: $TOTAL_OK / 432"

# Build manifest if we have enough data
if [ "$TOTAL_OK" -gt 0 ]; then
  echo ""
  echo "=== Building manifest ==="
  MANIFEST="$RUN_DIR/manifest.csv"
  cat > "$MANIFEST" << 'HEREDOC'
stage,top_config_label,config,family,type,template_index,template_id,planner_mode,temperature,speed_mps,speed_cm_s,horizon,num_samples,num_iterations,init_std,smoothing,execute_steps,max_mpc_steps,total_budget,pusher_mass,status,success,early_stop_reason,failure_type,initial_pos_dist_m,final_pos_dist_m,best_pos_dist_m,best_pos_dist_step,final_theta_error_deg,best_theta_error_deg,best_theta_error_step,best_pose_score,best_pose_score_step,best_pose_pos_dist_m,best_pose_theta_error_deg,total_progress_m,last_progress_m,success_pos_1mm,success_pos_2mm,success_pos_5mm,success_pos_10mm,success_pos_50mm,success_pose_2mm_10deg,success_pose_5mm_10deg,success_pose_10mm_10deg,success_pose_5mm_5deg,success_pose_10mm_5deg,reached_pos_5mm_once,reached_pos_10mm_once,reached_pose_5mm_10deg_once,reached_pose_10mm_10deg_once,regressed_after_near_success,collision_count,collision_rate,contact_count,mpc_steps,total_env_steps,runtime_sec,ee_path_length_m,object_path_length_m,net_progress_m,progress_efficiency_ee,progress_efficiency_object,object_to_ee_motion_ratio,wasted_motion_ratio,wasted_motion_ratio_capped,action_smoothness_mean,action_direction_change_count,mean_action_norm,contact_efficiency,time_to_success_env_steps,time_to_near_success_10mm_env_steps,random_walk_flag,inefficient_success_flag,excessive_wander_flag,clean_success_flag,early_ee_path_length_m,middle_ee_path_length_m,late_ee_path_length_m,early_object_path_length_m,middle_object_path_length_m,late_object_path_length_m,early_progress_m,middle_progress_m,late_progress_m,early_progress_efficiency_ee,middle_progress_efficiency_ee,late_progress_efficiency_ee,early_action_direction_change_count,middle_action_direction_change_count,late_action_direction_change_count,early_contact_count,middle_contact_count,late_contact_count,early_best_dist_improvement_m,middle_best_dist_improvement_m,late_best_dist_improvement_m,late_breakthrough_flag,front_loaded_wander_flag,meaningless_exploration_flag,path_includes_initial_position,ee_positions_count,object_positions_count,effective_sample_size_mean,weight_entropy_mean,collapse_rate,episode_id,video_path,log_path,error
HEREDOC

  for f in "$RUN_DIR/logs/"*.log; do
    [ -s "$f" ] || continue
    LAST=$(tail -1 "$f" 2>/dev/null || echo "")
    if echo "$LAST" | $PYTHON -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
      CFG_NAME=$(basename "$f" .log)
      echo "$LAST" | $PYTHON -c "
import sys,json,csv,io
d = json.loads(sys.stdin.read())
d['stage'] = 'stage2c'
d['status'] = 'completed'
d['log_path'] = '${f#$RUN_DIR/}'
# Extract top_config_label
for tc in ['sp030_T0.1','sp050_T0.2','sp020_T0.3']:
    if tc in '$CFG_NAME':
        d['top_config_label'] = tc
        break
for k in ['video_path','planner_mode','horizon','pusher_mass','total_budget','type','template_id',
          'smoothing','init_std','speed_mps','speed_cm_s','num_samples','num_iterations',
          'ee_path_length_m','object_path_length_m','net_progress_m','progress_efficiency_ee',
          'progress_efficiency_object','object_to_ee_motion_ratio','wasted_motion_ratio',
          'wasted_motion_ratio_capped','action_smoothness_mean','action_direction_change_count',
          'mean_action_norm','contact_efficiency','time_to_success_env_steps',
          'time_to_near_success_10mm_env_steps','random_walk_flag','inefficient_success_flag',
          'excessive_wander_flag','clean_success_flag','early_ee_path_length_m','middle_ee_path_length_m',
          'late_ee_path_length_m','early_object_path_length_m','middle_object_path_length_m',
          'late_object_path_length_m','early_progress_m','middle_progress_m','late_progress_m',
          'early_progress_efficiency_ee','middle_progress_efficiency_ee','late_progress_efficiency_ee',
          'early_action_direction_change_count','middle_action_direction_change_count',
          'late_action_direction_change_count','early_contact_count','middle_contact_count',
          'late_contact_count','early_best_dist_improvement_m','middle_best_dist_improvement_m',
          'late_best_dist_improvement_m','late_breakthrough_flag','front_loaded_wander_flag',
          'meaningless_exploration_flag','path_includes_initial_position','ee_positions_count',
          'object_positions_count','effective_sample_size_mean','weight_entropy_mean','collapse_rate']:
    if k not in d: d[k] = ''
hdr = '$(head -1 "$MANIFEST")'.split(',')
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=hdr, extrasaction='ignore')
w.writerow(d)
print(out.getvalue().strip())
" >> "$MANIFEST" 2>/dev/null
    fi
  done

  MANIFEST_ROWS=$(tail -n +2 "$MANIFEST" | wc -l)
  echo "Manifest: $MANIFEST_ROWS rows"

  # Run summarizer
  echo "Running summarizer..."
  $PYTHON scripts/summarize_mppi_stage2c_horizon_samples_std.py --run-dir "$RUN_DIR" 2>&1 || true
fi

echo ""
echo "DONE: $RUN_DIR"
