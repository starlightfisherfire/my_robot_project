#!/usr/bin/env bash
set -eo pipefail
# 重跑 Planner Trio 失败的 27 个任务（修复版：更新 manifest）

cd ~/my_robot_project
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; set -u
export PYTHONPATH=. MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_DIR="runs/planner_trio_300g_sweep_20260516_090909"
VIDEO_DIR="${RUN_DIR}/videos"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST="${RUN_DIR}/manifest.csv"
TEMPLATES="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"

declare -A PLANNER_MODE
PLANNER_MODE[cem]="cem"
PLANNER_MODE[mmcem]="multimodal_cem:--lateral-offset 0.5"
PLANNER_MODE[mppi]="mppi:--mppi-temperature 0.1"

declare -A LABEL_MAP
LABEL_MAP[be00]="test_sim_layout_ood_blocking_easy:0"
LABEL_MAP[bm00]="test_sim_layout_ood_blocking_medium:0"
LABEL_MAP[bh00]="test_sim_layout_ood_blocking_hard:0"
LABEL_MAP[pw00]="test_sim_layout_ood_passage_direct_wide:0"
LABEL_MAP[pm00]="test_sim_layout_ood_passage_direct_medium:0"
LABEL_MAP[ph00]="test_sim_layout_ood_passage_direct_narrow:0"

declare -A CONFIG_MAP
CONFIG_MAP[s005_h80_b500]="0.05:80:10:50"
CONFIG_MAP[s005_h100_b500]="0.05:100:10:50"
CONFIG_MAP[s010_h80_b500]="0.10:80:10:50"
CONFIG_MAP[s010_h100_b500]="0.10:100:10:50"
CONFIG_MAP[s015_h80_b500]="0.15:80:10:50"
CONFIG_MAP[s015_h100_b800]="0.15:100:16:50"

FAILED=$(grep ",failed," "$MANIFEST" | cut -d',' -f1)
TOTAL=$(echo "$FAILED" | wc -l)
echo "=== 重跑 $TOTAL 个失败任务（manifest更新版） ==="
echo

COUNT=0
while IFS= read -r cfg; do
  COUNT=$((COUNT+1))
  
  PLANNER=$(echo "$cfg" | cut -d'_' -f1)
  SPEED_TAG=$(echo "$cfg" | cut -d'_' -f2)
  HORIZON_TAG=$(echo "$cfg" | cut -d'_' -f3)
  BUDGET_TAG=$(echo "$cfg" | cut -d'_' -f4)
  LABEL=$(echo "$cfg" | cut -d'_' -f5)
  
  CONFIG_KEY="${SPEED_TAG}_${HORIZON_TAG}_${BUDGET_TAG}"
  CONFIG_VAL="${CONFIG_MAP[$CONFIG_KEY]}"
  IFS=':' read -r SPEED HORIZON EXEC MPC <<< "$CONFIG_VAL"
  
  LBL_VAL="${LABEL_MAP[$LABEL]}"
  IFS=':' read -r SPLIT IDX <<< "$LBL_VAL"
  
  PM_VAL="${PLANNER_MODE[$PLANNER]}"
  PM_MODE="${PM_VAL%%:*}"
  PM_EXTRA="${PM_VAL#*:}"
  [[ "$PM_EXTRA" == "$PM_VAL" ]] && PM_EXTRA=""
  
  OUT="${VIDEO_DIR}/${cfg}.mp4"
  LOG="${LOG_DIR}/${cfg}.log"
  
  echo "[${COUNT}/${TOTAL}] ${cfg} planner=${PM_MODE}"
  rm -f "$LOG"
  
  # 跑渲染，捕获输出
  if python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates "$TEMPLATES" --split "$SPLIT" --template-index "$IDX" \
    --planner-mode "$PM_MODE" \
    ${PM_EXTRA:+$PM_EXTRA} \
    --horizon "$HORIZON" --execute-steps "$EXEC" --max-mpc-steps "$MPC" \
    --num-samples 1024 --num-elites 96 --num-iterations 5 \
    --max-speed-mps "$SPEED" --pusher-mass 0.300 \
    --strict-pose-stop --camera topdown --width 1280 --height 720 --fps 10 \
    --parallel-cem --cem-workers 32 --mp-start-method spawn \
    --out-video "$OUT" \
    2>&1 | tee "$LOG"; then

    # 成功：从 log 中提取数据
    SC=$(grep -c "Success: True" "$LOG" 2>/dev/null || echo 0)
    [ "$SC" -gt 0 ] && ST="True" || ST="False"
    D=$(grep "Best dist:" "$LOG" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
    CO=$(grep "CEM best_cost:" "$LOG" 2>/dev/null | grep -oP 'best_cost: \K[0-9.]+' | awk '{s+=$1;n++;if($1<m||!m)m=$1}END{if(n>0)printf "%.4f,%.4f",s/n,m;else print "N/A,N/A"}')
    CL=$(grep "Planned collision:" "$LOG" 2>/dev/null | grep -oP 'count=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
    MS=$(grep "MPC Step" "$LOG" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+(?=/)' || echo "N/A")
    RT=$(grep "Total runtime:" "$LOG" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
    
    NEWLINE="${cfg},${PLANNER},${LABEL},${SPEED},${HORIZON},${EXEC},${MPC},0.300,${SPLIT},${IDX},completed,${ST},${D},${CO},${CL},${MS},${RT}"
    
    # 替换 manifest 中对应的 failed 行
    sed -i "/^${cfg},/c\\${NEWLINE}" "$MANIFEST"
    echo "  ✓ DONE success=${ST} dist=${D}"
  else
    echo "  ✗ FAILED"
  fi
  echo
done <<< "$FAILED"

COMPLETED=$(grep -c "completed" "$MANIFEST" || echo 0)
echo "=== 全部完成: ${COMPLETED}/108 ==="
