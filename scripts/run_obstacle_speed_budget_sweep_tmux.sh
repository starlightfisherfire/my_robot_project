#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Obstacle speed × budget sweep
# Paper 1 / MuJoCo Oracle-MPC obstacle capacity gate
#
# Total:
#   speeds: 0.05, 0.075, 0.10
#   budgets: 600, 800, 1000 env steps
#   templates: single obstacle x2, double obstacle x2
#   videos: 3 * 3 * 4 = 36
# ============================================================

cd ~/my_robot_project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TEMPLATES="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"

# 如果你的 single / double obstacle split 名字不同，只改这里。
# 当前先按 sixpack blocking easy 统一 split 处理。
SPLIT_SINGLE="test_sim_layout_ood_blocking_easy"
SPLIT_DOUBLE="test_sim_layout_ood_blocking_easy"

# 这里默认：
#   template 0,1 = 单障碍物两个模板
#   template 2,3 = 双障碍物两个模板
#
# 如果 sixpack 里的索引不是这样，只改这四行即可。
CASES=(
  "single_obs_A:${SPLIT_SINGLE}:0"
  "single_obs_B:${SPLIT_SINGLE}:1"
  "double_obs_A:${SPLIT_DOUBLE}:2"
  "double_obs_B:${SPLIT_DOUBLE}:3"
)

SPEEDS=("0.05" "0.075" "0.10")

# budget label -> max_mpc_steps
# execute_steps=20, so:
#   30 * 20 = 600
#   40 * 20 = 800
#   50 * 20 = 1000
BUDGET_LABELS=("600" "800" "1000")

get_max_mpc_steps() {
  case "$1" in
    600) echo 30 ;;
    800) echo 40 ;;
    1000) echo 50 ;;
    *)
      echo "Unknown budget: $1" >&2
      exit 1
      ;;
  esac
}

speed_tag() {
  local s="$1"
  echo "$s" | sed 's/0\.//g' | sed 's/\.//g'
}

RUN_ROOT="runs/debug/obstacle_speed_budget_sweep_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "$VIDEO_DIR" "$LOG_DIR"

echo "============================================================"
echo "Obstacle speed × budget sweep"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TEMPLATES=${TEMPLATES}"
echo "Total expected videos: $(( ${#SPEEDS[@]} * ${#BUDGET_LABELS[@]} * ${#CASES[@]} ))"
echo "============================================================"

for case_item in "${CASES[@]}"; do
  IFS=":" read -r CASE_NAME SPLIT TEMPLATE_INDEX <<< "$case_item"

  for SPEED in "${SPEEDS[@]}"; do
    SPEED_TAG="$(speed_tag "$SPEED")"

    for BUDGET in "${BUDGET_LABELS[@]}"; do
      MAX_MPC_STEPS="$(get_max_mpc_steps "$BUDGET")"

      OUT_VIDEO="${VIDEO_DIR}/${CASE_NAME}_speed${SPEED_TAG}_budget${BUDGET}_t${TEMPLATE_INDEX}.mp4"
      LOG_FILE="${LOG_DIR}/${CASE_NAME}_speed${SPEED_TAG}_budget${BUDGET}_t${TEMPLATE_INDEX}.log"

      echo
      echo "------------------------------------------------------------"
      echo "[RUN]"
      echo "case=${CASE_NAME}"
      echo "split=${SPLIT}"
      echo "template_index=${TEMPLATE_INDEX}"
      echo "speed=${SPEED}"
      echo "budget=${BUDGET}"
      echo "max_mpc_steps=${MAX_MPC_STEPS}"
      echo "out=${OUT_VIDEO}"
      echo "log=${LOG_FILE}"
      echo "------------------------------------------------------------"

      MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
        --templates "$TEMPLATES" \
        --split "$SPLIT" \
        --template-index "$TEMPLATE_INDEX" \
        --horizon 80 \
        --execute-steps 20 \
        --max-mpc-steps "$MAX_MPC_STEPS" \
        --num-samples 1024 \
        --num-elites 96 \
        --num-iterations 5 \
        --max-speed-mps "$SPEED" \
        --strict-pose-stop \
        --camera topdown \
        --width 1280 \
        --height 720 \
        --fps 10 \
        --parallel-cem \
        --cem-workers 28 \
        --mp-start-method spawn \
        --out-video "$OUT_VIDEO" \
        2>&1 | tee "$LOG_FILE"

      echo "[DONE] ${OUT_VIDEO}"
    done
  done
done

echo
echo "============================================================"
echo "Sweep finished."
echo "Videos: ${VIDEO_DIR}"
echo "Logs:   ${LOG_DIR}"
echo "============================================================"
