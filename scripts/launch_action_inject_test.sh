#!/bin/bash
# Wait for training to finish, then run control tests on all 3 models.
set -e

TRAIN_LOG="/tmp/action_inject_train.log"
CKPT_DIR="/home/brucewu/my_robot_project/runs/retrain_action_inject_50ep"
NORM="/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json"
OUT_BASE="/home/brucewu/my_robot_project/runs/test_action_inject_$(date +%Y%m%d_%H%M%S)"

echo "Waiting for training to finish..."
while true; do
    if grep -q "Training summary:" "$TRAIN_LOG" 2>/dev/null; then
        echo "Training complete!"
        break
    fi
    sleep 30
done

echo ""
echo "=== Running control tests ==="

for ENC in flat object_centric causality_aware; do
    CKPT="${CKPT_DIR}/${ENC}/checkpoints/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP $ENC: checkpoint not found"
        continue
    fi

    echo ""
    echo "--- Testing $ENC ---"
    OUT="${OUT_BASE}/${ENC}"

    cd /home/brucewu/my_robot_project
    PYTHONPATH=. python3 scripts/test_action_inject_control.py \
        --checkpoint "$CKPT" \
        --encoder-type "$ENC" \
        --normalizer "$NORM" \
        --planner MPPI \
        --max-templates 5 \
        --families "open,blocking_easy,passage_direct_medium" \
        --max-speed 0.5 \
        --out-dir "$OUT" \
        --compare-old \
        --old-checkpoint runs/retrain_nomass_50ep/${ENC}/checkpoints/best.pt \
        2>&1 | tee "${OUT}.log"
done

echo ""
echo "=== All tests complete ==="
echo "Results in: $OUT_BASE"
