#!/usr/bin/env bash
set -eo pipefail
# MPPI Parameter Sweep v1 — Full Pipeline Runner
# Stage 1: temperature sweep | Stage 2: sample/std refinement | Stage 3: full confirmation

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u
export PYTHONPATH=. MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ── Config ──
RUN_ROOT="runs/mppi_param_sweep_v1_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="data/sim/mppi_sweep_v1"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
TEMPLATE_FILE="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"
CEM_WORKERS=32

mkdir -p "$VIDEO_DIR" "$LOG_DIR" "$DATA_DIR"

# ── Manifest header (proposal Section K) ──
echo "stage,config,family,type,label,planner_mode,temperature,num_samples,num_iterations,init_std,smoothing,speed_mps,speed_cm_s,horizon,execute_steps,max_mpc_steps,total_budget,template_file,split,template_index,template_id,obstacle_count,passage_gap,is_direct,is_bypass,status,success,best_dist,avg_cost,min_cost,collision_count,collision_rate,contact_count,mpc_steps,runtime_sec,effective_sample_size_mean,effective_sample_size_min,weight_entropy_mean,collapse_rate,temperature_collapse_flag,episode_id,video_path,log_path,error" > "$MANIFEST"

# ── Template selection ──
declare -A TEMPLATE_SPLITS
TEMPLATE_SPLITS[be00]="test_sim_layout_ood_blocking_easy:0"
TEMPLATE_SPLITS[bm00]="test_sim_layout_ood_blocking_medium:0"
TEMPLATE_SPLITS[bh00]="test_sim_layout_ood_blocking_hard:0"
TEMPLATE_SPLITS[pw00]="test_sim_layout_ood_passage_direct_wide:0"
TEMPLATE_SPLITS[pm00]="test_sim_layout_ood_passage_direct_medium:0"
TEMPLATE_SPLITS[ph00]="test_sim_layout_ood_passage_direct_narrow:0"

# ── Stage 1: Temperature sweep ──
run_stage1() {
  local temps=(0.1 0.3 1.0 3.0 10.0)
  local fixed_h=140 fixed_exec=10 fixed_mpc=100
  local fixed_samples=1024 fixed_iter=5 fixed_std=0.7 fixed_smoothing=0.2 fixed_speed=0.75
  
  # Core20 templates: family:indices
  local core20=(
    "be00:0" "be00:5" "bm00:2" "bm00:8"
    "bh00:0" "bh00:1" "bh00:5" "bh00:9"
    "pw00:0" "pw00:5" "pm00:2" "pm00:6" "ph00:0" "ph00:4" "ph00:7"
  )
  
  local total=$((${#temps[@]} * ${#core20[@]}))
  local count=0
  echo "=== Stage 1: Temperature Sweep (${total} runs) ==="
  
  for T in "${temps[@]}"; do
    for entry in "${core20[@]}"; do
      IFS=':' read -r label idx <<< "$entry"
      local split_idx="${TEMPLATE_SPLITS[$label]}"
      IFS=':' read -r split tidx <<< "$split_idx"
      count=$((count+1))
      local cfg="mppi_T${T}_${label}_idx${idx}"
      
      echo "[S1 ${count}/${total}] $cfg T=${T}"
      
      local out="${VIDEO_DIR}/${cfg}.mp4"
      local log="${LOG_DIR}/${cfg}.log"
      
      if python scripts/render_closed_loop_rollout_parallel_cem.py \
        --templates "$TEMPLATE_FILE" --split "$split" --template-index "$idx" \
        --planner-mode mppi --mppi-temperature "$T" \
        --horizon "$fixed_h" --execute-steps "$fixed_exec" --max-mpc-steps "$fixed_mpc" \
        --num-samples "$fixed_samples" --num-elites 96 --num-iterations "$fixed_iter" \
        --max-speed-mps "$fixed_speed" --pusher-mass 0.300 \
        --strict-pose-stop --camera topdown --width 1280 --height 720 --fps 10 \
        --parallel-cem --cem-workers "$CEM_WORKERS" --mp-start-method spawn \
        --out-video "$out" 2>&1 | tee "$log"; then
        
        # Extract metrics from log
        local st; st=$(grep -c "Success: True" "$log" 2>/dev/null && echo "True" || echo "False")
        local d; d=$(grep "Best dist:" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        local co; co=$(grep "best_cost:" "$log" | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++;if($1<m||!m)m=$1}END{if(n>0)printf "%.4f,%.4f",s/n,m;else print "N/A,N/A"}')
        local cl; cl=$(grep "Planned collision:" "$log" | grep -oP 'count=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
        local ms; ms=$(grep "MPC Step" "$log" | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
        local rt; rt=$(grep "Total runtime:" "$log" | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
        
        echo "1,${cfg},${label},blocking,${label},mppi,${T},${fixed_samples},${fixed_iter},${fixed_std},${fixed_smoothing},${fixed_speed},$(echo "${fixed_speed}*100" | bc),${fixed_h},${fixed_exec},${fixed_mpc},1000,${TEMPLATE_FILE},${split},${idx},${label},1,,false,false,completed,${st},${d},${co},${cl},,0,${ms},${rt},,,,,,,${out},${log}," >> "$MANIFEST"
      else
        echo "1,${cfg},${label},blocking,${label},mppi,${T},${fixed_samples},${fixed_iter},${fixed_std},${fixed_smoothing},${fixed_speed},$(echo "${fixed_speed}*100" | bc),${fixed_h},${fixed_exec},${fixed_mpc},1000,${TEMPLATE_FILE},${split},${idx},${label},1,,false,false,failed,,,,,,,,,,,,,${out},${log},render_error" >> "$MANIFEST"
      fi
    done
  done
  
  echo "Stage 1 complete: $(grep -c "completed" "$MANIFEST")/${total} runs"
}

run_stage1 "$@"
echo "Pipeline runner v1 — Stage 1 done. Run summarizer next."
