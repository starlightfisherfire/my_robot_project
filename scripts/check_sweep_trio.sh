#!/usr/bin/env bash
# Planner Trio 监控：检查是否108个任务全部完成
# 完成时跑分析并输出结果
set -euo pipefail

DIR="/home/brucewu/my_robot_project/runs/planner_trio_300g_sweep_20260516_090909"
MF="$DIR/manifest.csv"
TOTAL_NEEDED=108

if [ ! -f "$MF" ]; then
  echo "⚠️ manifest not found"
  exit 1
fi

COMPLETED=$(grep -c "completed" "$MF" || echo 0)
echo "Planner Trio: $COMPLETED/$TOTAL_NEEDED tasks ($(date +%H:%M))"

if [ "$COMPLETED" -ge "$TOTAL_NEEDED" ]; then
  echo ""
  echo "=== ALL DONE ==="
  cd /home/brucewu/my_robot_project
  python3 scripts/analyze_sweeps_trio.py 2>&1
  echo ""
  echo "=== 分析完成，请通知用户 ==="
fi
