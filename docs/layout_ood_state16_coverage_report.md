# Coverage Report — layout_ood_state16_v1

## Audit Summary

- **Total npz**: 507
- **canonical_state16 [T,6,16]**: 18 (3.6%)
- **compact_planner_rollout [T,5]**: 489 (96.4%)
- **Training-ready state16**: 18

## Canonical state16 coverage

| Split | Family | Episodes available | Target | Status |
|-------|--------|--------------------|--------|--------|
| train_sim_id | open | 8 | 50 | ❌ insufficient |
| train_sim_id | blocking_easy | 0 | 50 | ❌ missing |
| train_sim_id | blocking_medium | 0 | 50 | ❌ missing |
| train_sim_id | blocking_hard | 0 | 50 | ❌ missing |
| val_sim_id | heldout | 0 | 10 | ❌ missing |
| test_sim_id | heldout | 0 | 10 | ❌ missing |
| test_layout_ood_direct | passage_direct_wide | 0 | 15 | ❌ missing |
| test_layout_ood_direct | passage_direct_medium | 0 | 15 | ❌ missing |
| test_layout_ood_direct | passage_direct_narrow | 0 | 15 | ❌ missing |
| test_layout_ood_bypass | passage_bypass_wide | 0 | 3 | ❌ missing |
| test_layout_ood_bypass | passage_bypass_medium | 0 | 3 | ❌ missing |
| test_layout_ood_bypass | passage_bypass_narrow | 0 | 3 | ❌ missing |

## Compact planner rollout metadata (489 files, NOT usable for state16 training)

These files have rich metadata (family, split, template_id) but states are [T,5] compact format, NOT [T,6,16] canonical state16.

Per user instruction: "不要把 compact rollout npz 伪装成 state16"

## Gate 2 Result: FAIL

Coverage insufficient. Only 18 state16 episodes, all open/T-shape/train.
No blocking, no passage, no heldout data.

**Action**: Generate missing_data_plan.md
