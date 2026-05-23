# Learned MPC MuJoCo Closed-Loop Smoke Plan

**Status:** PLAN ONLY — not executed  

---

## Purpose

Verify that learned rollout planning advantage (from internal eval) translates to real MuJoCo physics.

## Current Readiness

- ✅ MuJoCo env: reset/step/clone/restore
- ✅ State extractor: collect_layout_ood_state16_v1.py
- ✅ Action scale: oracle rollout uses env.step, same as learned MPC
- ✅ Replan loop: learned rollout + CEM replan
- ✅ Success metrics: mujoco_oracle_capacity.py

## Minimum Smoke Command (NOT executed)

```bash
# Build learned MPC closed-loop eval script (NEW, needed)
PYTHONPATH=. python scripts/run_learned_mujoco_closed_loop.py \
  --checkpoint runs/pilot_state16_mppi_stage2c/train_causality_aware/causality_aware/checkpoints/best.pt \
  --model-type causality_aware \
  --normalizer runs/pilot_state16_mppi_stage2c/train_causality_aware/normalizer_state16.json \
  --split-file configs/splits/mppi_stage2c_state16_pilot.yaml \
  --split-name random_episode_split/test \
  --max-templates 5 \
  --horizon 10 \
  --num-samples 128 \
  --max-mpc-steps 30 \
  --out runs/pilot_state16_mppi_stage2c/closed_loop_smoke/causality_aware_smoke.json
```

## Required New Script

`scripts/run_learned_mujoco_closed_loop.py` — minimal changes from run_learned_mpc_eval.py:
1. Replace learned rollout state update with real MuJoCo step
2. Extract canonical_state16 from MuJoCo observation after each step
3. Rebuild history window from recent observations
4. Call CEM plan with learned rollout cost function
5. Execute first action on real MuJoCo
6. Check success criteria (pos < 2cm, theta < 10°)

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Action scale mismatch | MEDIUM | Verify action range matches MPPI data |
| State extraction mismatch | MEDIUM | Use same _extract_structured_state as data collection |
| Normalizer mismatch | LOW | Use same normalizer as training |
| History window construction | LOW | Same logic as State16Dataset |

## Decision

- Ready to implement: YES (1 new script needed)
- Risk level: LOW-MEDIUM (well-understood interfaces)
- Recommend: Implement and run 5-template smoke before scaling
