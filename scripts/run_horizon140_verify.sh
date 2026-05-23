#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Horizon140 验证实验
# 目标: 验证 Horizon=140 + Speed=0.75 是否能达到 100% 成功率
# 模板: blocking_hard (10个) + blocking_medium (10个) + passage_direct_medium (1个)
# 条件: E_per_replan=10 和 E_per_replan=20
# ============================================================

cd ~/my_robot_project

set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u

export PYTHONPATH=.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TEMPLATES_DIFFICULTY="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
TEMPLATES_SIXPACK="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"
HORIZON=140
EXEC_STEPS=10
MAX_MPC=100
CEM_WORKERS=30
NUM_SAMPLES=1024
NUM_ELITES=96
NUM_ITER=5
SPEED=0.75

RUN_ROOT="runs/horizon140_verify_$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${RUN_ROOT}/videos"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

# 定义实验配置
# 格式: "模板文件:split:模板索引:模式名"
# 注意: template-index 是相对于 split 过滤后的列表，从 0 开始
CONFIGS=(
  # blocking_hard (10个模板, 过滤后索引 0-9)
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:0:blocking_hard_t0"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:1:blocking_hard_t1"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:2:blocking_hard_t2"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:3:blocking_hard_t3"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:4:blocking_hard_t4"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:5:blocking_hard_t5"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:6:blocking_hard_t6"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:7:blocking_hard_t7"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:8:blocking_hard_t8"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_hard:9:blocking_hard_t9"
  # blocking_medium (10个模板, 过滤后索引 0-9)
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:0:blocking_medium_t0"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:1:blocking_medium_t1"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:2:blocking_medium_t2"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:3:blocking_medium_t3"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:4:blocking_medium_t4"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:5:blocking_medium_t5"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:6:blocking_medium_t6"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:7:blocking_medium_t7"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:8:blocking_medium_t8"
  "${TEMPLATES_DIFFICULTY}:test_sim_layout_ood_blocking_medium:9:blocking_medium_t9"
  # passage_direct_medium (1个模板, 过滤后索引 0)
  "${TEMPLATES_SIXPACK}:test_sim_layout_ood_passage_direct_medium:0:passage_direct_medium_t0"
)

TOTAL=${#CONFIGS[@]}

echo "============================================================"
echo "Horizon140 验证实验"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Horizon: ${HORIZON}, Speed: ${SPEED}"
echo "Execute steps: ${EXEC_STEPS}, Max MPC: ${MAX_MPC}"
echo "Total budget: $((EXEC_STEPS * MAX_MPC))"
echo "CEM: workers=${CEM_WORKERS}, samples=${NUM_SAMPLES}, elites=${NUM_ELITES}, iter=${NUM_ITER}"
echo "Total configs: ${TOTAL}"
echo "  - blocking_hard: 10 templates"
echo "  - blocking_medium: 10 templates"
echo "  - passage_direct_medium: 1 template"
echo "============================================================"

echo "config,tpl_file,split,template_idx,mode,status,best_dist,avg_cost,min_cost,collision_count,mpc_steps,runtime_sec" > "$MANIFEST"

COUNT=0
FAIL=0
SUCCESS=0

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r TPL_FILE SPLIT TPL_IDX MODE <<< "$cfg"
  COUNT=$((COUNT + 1))
  CFG_NAME="h140_sp075_${MODE}"

  OUT_VIDEO="${VIDEO_DIR}/${CFG_NAME}.mp4"
  LOG_FILE="${LOG_DIR}/${CFG_NAME}.log"

  echo
  echo "============================================================"
  echo "[${COUNT}/${TOTAL}] ${CFG_NAME}"
  echo "  split=${SPLIT} tpl_idx=${TPL_IDX} mode=${MODE}"
  echo "============================================================"

  if MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates "$TPL_FILE" \
    --split "$SPLIT" \
    --template-index "$TPL_IDX" \
    --horizon "$HORIZON" \
    --execute-steps "$EXEC_STEPS" \
    --max-mpc-steps "$MAX_MPC" \
    --num-samples "$NUM_SAMPLES" \
    --num-elites "$NUM_ELITES" \
    --num-iterations "$NUM_ITER" \
    --max-speed-mps "$SPEED" \
    --strict-pose-stop \
    --camera topdown \
    --width 1280 \
    --height 720 \
    --fps 10 \
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

    echo "${CFG_NAME},${TPL_FILE},${SPLIT},${TPL_IDX},${MODE},${SUCC},${BEST_DIST},${COSTS},${COLL},${MPC_STEPS},${RT}" >> "$MANIFEST"
    
    if [ "$SUCC" = "True" ]; then
      SUCCESS=$((SUCCESS + 1))
      echo "[DONE ✅] ${CFG_NAME}"
    else
      FAIL=$((FAIL + 1))
      echo "[DONE ❌] ${CFG_NAME}"
    fi
  else
    FAIL=$((FAIL + 1))
    echo "${CFG_NAME},${TPL_FILE},${SPLIT},${TPL_IDX},${MODE},FAILED,N/A,N/A,N/A,N/A,N/A,N/A" >> "$MANIFEST"
    echo "[FAIL] ${CFG_NAME}"
  fi
done

echo
echo "============================================================"
echo "ALL ${TOTAL} DONE."
echo "Success: ${SUCCESS}/${TOTAL} ($(( SUCCESS * 100 / TOTAL ))%)"
echo "Failure: ${FAIL}/${TOTAL}"
echo "Videos:   ${VIDEO_DIR}"
echo "Logs:     ${LOG_DIR}"
echo "Manifest: ${MANIFEST}"
echo "============================================================"

echo
echo "=== RESULTS SUMMARY ==="
echo
column -t -s',' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
