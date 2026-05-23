#!/bin/bash
# layout_ood_state16_v1: Full canonical state16 data collection
# Run in tmux: tmux new -s collect_state16_v1
# Then: bash scripts/collect_state16_v1_full.sh
set -eo pipefail

source /home/brucewu/miniconda3/bin/activate lerobot
export PYTHONPATH=/home/brucewu/my_robot_project
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/brucewu/my_robot_project

echo "=== Full state16 v1 collection ==="
echo "Start: $(date)"
echo "Output: data/sim/layout_ood_state16_v1/"
echo ""

python3 scripts/collect_layout_ood_state16_v1.py \
  --config configs/experiments/collect_layout_ood_state16_v1.yaml \
  --full 2>&1 | tee runs/collect_state16_v1_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Done: $(date)"
echo "Next: validate dataset"
echo "  python scripts/validate_layout_ood_state16_dataset.py --data-dir data/sim/layout_ood_state16_v1"
