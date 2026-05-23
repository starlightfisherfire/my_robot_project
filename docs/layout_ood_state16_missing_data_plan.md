# Missing Data Plan — layout_ood_state16_v1

## Current Status

Only **18 canonical_state16** episodes exist, all from `open/T-shape/train`.
489 compact planner rollout files exist but are NOT state16 format (states are [T,5] not [T,6,16]).

## Missing Data Summary

| Split | Family | Episodes needed | Source |
|-------|--------|----------------|--------|
| train_sim_id | open | 50 | collect new |
| train_sim_id | blocking_easy | 50 | collect new |
| train_sim_id | blocking_medium | 50 | collect new |
| train_sim_id | blocking_hard | 50 | collect new |
| val_sim_id | open (heldout templates) | 10 | collect new |
| val_sim_id | blocking_easy (heldout) | 10 | collect new |
| test_sim_id | blocking_medium (heldout) | 10 | collect new |
| test_sim_id | blocking_hard (heldout) | 10 | collect new |
| test_layout_ood_direct | passage_direct_wide | 15 | collect new |
| test_layout_ood_direct | passage_direct_medium | 15 | collect new |
| test_layout_ood_direct | passage_direct_narrow | 15 | collect new |
| test_layout_ood_bypass | passage_bypass_wide | 3 | collect new |
| test_layout_ood_bypass | passage_bypass_medium | 3 | collect new |
| test_layout_ood_bypass | passage_bypass_narrow | 3 | collect new |
| **Total** | | **~304** | |

## Recommended Collector

Script: `scripts/collect_layout_ood_state16.py`

### Recommended MPPI Config (from Stage 2C best)

```yaml
mppi:
  horizon: 140
  num_samples: 1024
  num_iterations: 5
  init_std: 0.7
  temperature: 0.1
  smoothing: 0.2
  execute_steps: 10
  max_mpc_steps: 100
  speed_mps: 0.3
```

### Template Source

File: `data/sim/metadata/reset_templates_obstacle_10family_v0.json`

Available families:
- open: 10 templates
- blocking_easy: 10 templates
- blocking_medium: 10 templates
- blocking_hard: 10 templates
- passage_direct_wide: 10 templates
- passage_direct_medium: 10 templates
- passage_direct_narrow: 10 templates
- passage_bypass_wide: 1 template
- passage_bypass_medium: 1 template
- passage_bypass_narrow: 1 template

### Collection Commands

```bash
# Train: open (5 templates × 10 eps = 50)
PYTHONPATH=. python scripts/collect_layout_ood_state16.py \
  --config configs/experiments/layout_ood_state16_v0.yaml \
  --family open --templates 0-4 --episodes-per-template 10

# Train: blocking_easy (5 templates × 10 eps = 50)
PYTHONPATH=. python scripts/collect_layout_ood_state16.py \
  --family blocking_easy --templates 0-4 --episodes-per-template 10

# Train: blocking_medium (5 templates × 10 eps = 50)
PYTHONPATH=. python scripts/collect_layout_ood_state16.py \
  --family blocking_medium --templates 0-4 --episodes-per-template 10

# Train: blocking_hard (5 templates × 10 eps = 50)
PYTHONPATH=. python scripts/collect_layout_ood_state16.py \
  --family blocking_hard --templates 0-4 --episodes-per-template 10

# Val: heldout (5-6 templates × 2 eps = 10-12)
# Use templates 5-9 from open + blocking families

# Test: heldout (7-9 templates × 2 eps = 10-12)
# Use templates 7-9 from blocking families

# OOD: passage_direct (3 families × 5 templates × 3 eps = 45)
# Use templates 0,2,4,6,8 from passage_direct families

# OOD: passage_bypass (3 families × 1 template × 3 eps = 9)
# Use template 0 from passage_bypass families
```

## Priority Order

1. **Blocking train data** (blocking_easy/medium/hard × 150 eps) — most important for train
2. **Passage OOD data** (direct × 45 + bypass × 9) — needed for OOD evaluation
3. **Open + heldout** (open 50 + val/test heldout ~20) — supplementary

## Estimated Collection Time

- ~304 episodes × ~250 steps × ~0.1s per step = ~2.1 hours
- With MPPI overhead (5 iterations × 1024 samples): ~5-10x → ~10-20 hours total

## Alternative: Use Existing Compact Data as Guide

The 489 compact planner rollout files already have correct family/split/template metadata.
We can use them as a guide for which templates/parameters work, then collect canonical state16 versions.

## Recommendation

**DO NOT_BUILD_DATASET_DUE_TO_MISSING_COVERAGE**

Need to run collector first. Estimated time: 10-20 hours of MuJoCo simulation.
