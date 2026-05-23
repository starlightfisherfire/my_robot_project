#!/bin/bash
# MPPI Stage 2A: 8 temps × 2 execute settings × 12 core templates = 192 runs
# Balanced scheduling: template → execute_steps → temperature
# Config name: mppi_T0.75_ex20_b50_passage_bypass_wide_idx0
set -eo pipefail

RUN_DIR="runs/mppi_stage2a_temperature_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="data/sim/mppi_stage2a"
PYTHON="/home/brucewu/miniconda3/envs/lerobot/bin/python"
TEMPLATES="data/sim/metadata/reset_templates_obstacle_10family_v0.json"
MAX_PARALLEL=32
JOBS_FILE="/tmp/mppi_stage2a_jobs.txt"

mkdir -p "$RUN_DIR"/{logs,videos}
mkdir -p "$DATA_DIR"/{episodes,metadata}
export MUJOCO_GL=egl OMP_NUM_THREADS=1

> "$JOBS_FILE"

TEMPS="0.1 0.2 0.3 0.5 0.75 1.0 1.25 1.5"
# es10: exec=10, mpc=100; es20: exec=20, mpc=50 — both total_budget=1000
ES_LIST=("10:100" "20:50")

# core12 templates (split:index)
CORE12=(
  "test_sim_layout_ood_passage_bypass_wide:0"
  "test_sim_layout_ood_passage_bypass_medium:0"
  "test_sim_layout_ood_passage_bypass_narrow:0"
  "test_sim_layout_ood_blocking_hard:0"
  "test_sim_layout_ood_blocking_hard:1"
  "test_sim_layout_ood_blocking_hard:5"
  "test_sim_layout_ood_blocking_hard:9"
  "test_sim_layout_ood_passage_direct_narrow:0"
  "test_sim_layout_ood_passage_direct_narrow:4"
  "test_sim_layout_ood_passage_direct_narrow:7"
  "test_sim_layout_ood_passage_direct_medium:2"
  "test_sim_layout_ood_passage_direct_wide:5"
)

add_job() { echo "$1|$2" >> "$JOBS_FILE"; }

# Balanced: for each template, for each execute_setting, for each temperature
for SI in "${CORE12[@]}"; do
  SPLIT="${SI%:*}"
  IDX="${SI#*:}"
  FAM="${SPLIT##*ood_}"
  for ES_PAIR in "${ES_LIST[@]}"; do
    ES="${ES_PAIR%:*}"
    MPC="${ES_PAIR#*:}"
    for T in $TEMPS; do
      CFG="mppi_T${T}_ex${ES}_b${MPC}_${FAM}_idx${IDX}"
      LOG="$RUN_DIR/logs/${CFG}.log"
      VID="$RUN_DIR/videos/${CFG}.mp4"
      add_job "$CFG" "$PYTHON scripts/render_closed_loop_rollout_mppi_with_data.py \
        --planner-mode mppi --horizon 140 --max-speed-mps 0.75 --pusher-mass 0.300 \
        --num-samples 1024 --num-iterations 5 --mppi-init-std 0.7 --mppi-smoothing 0.2 \
        --mppi-temperature $T --execute-steps $ES --max-mpc-steps $MPC \
        --templates $TEMPLATES --split $SPLIT --template-index $IDX \
        --data-output-dir $DATA_DIR --out-video $VID > $LOG 2>&1"
    done
  done
done

TOTAL=$(wc -l < "$JOBS_FILE")
echo "=========================================="
echo "MPPI Stage 2A — Temperature × Execute-Steps"
echo "Run: $RUN_DIR  Jobs: $TOTAL  Parallel: $MAX_PARALLEL"
echo "Grid: 8 T × 2 ES × 12 templates = 192"
echo "=========================================="

# Run jobs in parallel
cut -d'|' -f2- "$JOBS_FILE" > "$JOBS_FILE.cmds"
xargs -0 -P "$MAX_PARALLEL" -I {} bash -c "{}" < <(tr '\n' '\0' < "$JOBS_FILE.cmds") || true

echo ""
echo "=== Building manifest ==="

# Full manifest header
MANIFEST="$RUN_DIR/manifest.csv"
cat > "$MANIFEST" << 'HEREDOC'
stage,priority,config,family,type,template_index,template_id,planner_mode,temperature,execute_steps,max_mpc_steps,total_budget,num_samples,num_iterations,init_std,smoothing,speed_mps,horizon,pusher_mass,status,success,early_stop_reason,failure_type,initial_pos_dist_m,final_pos_dist_m,best_pos_dist_m,best_pos_dist_step,final_theta_error_deg,best_theta_error_deg,best_theta_error_step,best_pose_score,best_pose_score_step,best_pose_pos_dist_m,best_pose_theta_error_deg,total_progress_m,last_progress_m,success_pos_1mm,success_pos_2mm,success_pos_5mm,success_pos_10mm,success_pos_50mm,success_pose_2mm_10deg,success_pose_5mm_10deg,success_pose_10mm_10deg,success_pose_5mm_5deg,success_pose_10mm_5deg,reached_pos_5mm_once,reached_pos_10mm_once,reached_pose_5mm_10deg_once,reached_pose_10mm_10deg_once,regressed_after_near_success,collision_count,collision_rate,contact_count,mpc_steps,total_env_steps,runtime_sec,effective_sample_size_mean,weight_entropy_mean,collapse_rate,episode_id,video_path,log_path,error
HEREDOC

SUCCESS_CNT=0; FAIL_CNT=0

while IFS='|' read -r CFG _; do
  [ -z "$CFG" ] && continue
  LOGF=$(find "$RUN_DIR/logs" -name "${CFG}.log" -print -quit 2>/dev/null)
  if [ -z "$LOGF" ] || [ ! -s "$LOGF" ]; then
    echo ",,$CFG,,,,,,,,,,,,,,,,,,failed,False,startup_no_log,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,$LOGF,startup_failed" >> "$MANIFEST"
    FAIL_CNT=$((FAIL_CNT+1))
    continue
  fi
  LAST=$(tail -1 "$LOGF" 2>/dev/null || echo "")
  if echo "$LAST" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    echo "$LAST" | python3 -c "
import sys,json,csv,io
d = json.loads(sys.stdin.read())
d['status'] = 'completed'
d['log_path'] = '${LOGF#$RUN_DIR/}'
if 'video_path' not in d: d['video_path'] = ''
# Fill missing manifest fields
d['planner_mode'] = d.get('planner_mode','mppi')
d['speed_mps'] = 0.75; d['horizon'] = 140; d['pusher_mass'] = 0.300
d['total_budget'] = int(d.get('execute_steps',10)) * int(d.get('max_mpc_steps',100))
if 'type' not in d: d['type'] = ''
if 'template_id' not in d: d['template_id'] = ''
d['smoothing'] = float(d.get('smoothing',0.2))
d['priority'] = ''; d['stage'] = ''
d['collision_rate'] = d.get('collision_count',0) / max(d.get('total_env_steps',1),1)
for k in ['effective_sample_size_mean','weight_entropy_mean','collapse_rate','error']:
    if k not in d: d[k] = ''
hdr = '$(head -1 "$MANIFEST")'.split(',')
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=hdr, extrasaction='ignore')
w.writerow(d)
print(out.getvalue().strip())
" >> "$MANIFEST" 2>/dev/null && SUCCESS_CNT=$((SUCCESS_CNT+1)) || {
      echo ",,$CFG,,,,,,,,,,,,,,,,,,failed,False,json_parse_error,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,$LOGF,json_parse_error" >> "$MANIFEST"
      FAIL_CNT=$((FAIL_CNT+1))
    }
  else
    ERR_TAIL=$(tail -3 "$LOGF" 2>/dev/null | tr '\n' ' ' | head -c 200)
    echo ",,$CFG,,,,,,,,,,,,,,,,,,failed,False,crash,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,$LOGF,$ERR_TAIL" >> "$MANIFEST"
    FAIL_CNT=$((FAIL_CNT+1))
  fi
done < "$JOBS_FILE"

ROWS=$(tail -n +2 "$MANIFEST" | wc -l)
echo "Manifest: $ROWS rows ($SUCCESS_CNT ok, $FAIL_CNT failed)"

# Run summarizer
if [ $SUCCESS_CNT -gt 0 ]; then
  echo "Running summarizer..."
  python3 scripts/summarize_stage2a.py "$MANIFEST" 2>&1
  # Copy to docs
  MD_FILE=$(dirname "$MANIFEST")/stage2a_summary.md
  [ -f "$MD_FILE" ] && cp "$MD_FILE" docs/mppi_stage2a_temperature_execute_summary.md
fi

echo "DONE: $RUN_DIR"
echo "SWEEP_COMPLETE $RUN_DIR $MANIFEST" > /tmp/mppi_stage2a_done.txt
