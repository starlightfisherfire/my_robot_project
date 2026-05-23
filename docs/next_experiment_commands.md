# Next Experiment Commands

**Last updated:** 2026-05-23

---

## Phase 0: Self-Check (每次改代码后必跑)

```bash
cd ~/my_robot_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

PYTHONPATH=. python scripts/self_check_learned_rollout_stack.py
```

---

## Phase 1: Data Re-Collection (优先)

**当前数据 89.4% zero movement，必须重采。**

```bash
# 使用 mppi_stage2c 最优配置重采
PYTHONPATH=. python scripts/collect_layout_ood_state16_v1.py --full

# 验证新数据质量
PYTHONPATH=. python scripts/audit_state16_transitions.py
```

**目标：**
- 每 family 至少 50 episodes
- success_rate > 30%
- zero_movement_rate < 30%

---

## Phase 2: Three-Model Train Smoke (数据准备好后)

```bash
PYTHONPATH=. python scripts/train_high_level.py \
    --config configs/experiments/train_state16_poc.yaml \
    --model flat --smoke \
    --out runs/train_smoke/flat

PYTHONPATH=. python scripts/train_high_level.py \
    --config configs/experiments/train_state16_poc.yaml \
    --model object_centric --smoke \
    --out runs/train_smoke/object_centric

PYTHONPATH=. python scripts/train_high_level.py \
    --config configs/experiments/train_state16_poc.yaml \
    --model causality_aware --smoke \
    --out runs/train_smoke/causality_aware
```

---

## Phase 3: Learned MPC Smoke (三模型)

```bash
for model in flat object_centric causality_aware; do
    PYTHONPATH=. python scripts/run_learned_mpc_eval.py \
        --checkpoint runs/train_smoke/${model}/checkpoints/best.pt \
        --model-type ${model} \
        --max-templates 10 \
        --horizon 10 \
        --num-samples 128 \
        --num-elites 16 \
        --num-iterations 3 \
        --max-mpc-steps 5 \
        --init-std 0.01 \
        --out runs/learned_mpc_eval/${model}_smoke_10eps.json
done
```

---

## Phase 4: Full Training + ID Eval (Phase 3 通过后)

```bash
# Full training
for model in flat object_centric causality_aware; do
    PYTHONPATH=. python scripts/train_high_level.py \
        --config configs/train/${model}.yaml --model $model
done

# ID evaluation
for model in flat object_centric causality_aware; do
    PYTHONPATH=. python scripts/run_learned_mpc_eval.py \
        --checkpoint runs/train_full/${model}/checkpoints/best.pt \
        --model-type ${model} \
        --split test_sim_id \
        --out runs/id_eval/${model}.json
done
```

---

## Phase 5: OOD Eval (Phase 4 通过后)

```bash
for model in flat object_centric causality_aware; do
    for split in test_sim_layout_ood_blocking test_sim_layout_ood_narrow_passage test_sim_layout_ood_edge_goal; do
        PYTHONPATH=. python scripts/run_learned_mpc_eval.py \
            --checkpoint runs/train_full/${model}/checkpoints/best.pt \
            --model-type ${model} --split ${split} \
            --out runs/ood_eval/${model}_${split}.json
    done
done
```
