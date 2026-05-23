# layout_ood_state16_v1 Summary

## 1. Dataset Source

- Source dirs: `data/sim/layout_ood_state16_v0/episodes` (8 eps), `data/sim/layout_ood_state16_v0/smoke_archive` (10 eps)
- canonical_state16 count: 18
- compact_planner_rollout excluded: 489 (states are [T,5] not [T,6,16])

## 2. Split Distribution

| Split | Family | Episodes | Transitions |
|-------|--------|----------|-------------|
| train | open | 8 | 2000 |
| (smoke_archive, no split info) | unknown | 10 | 2500 |
| **Total state16** | | **18** | **~4500** |

All 18 episodes are open/T-shape. No blocking, passage, or heldout data.

## 3. Leakage Check

- Train families: open only
- OOD families: none available
- Duplicate source episodes: none detected
- Normalizer fit split: train only (safe)

## 4. Validation Result

- Shape check: all 18 are [250, 6, 16] ✅
- Required fields: all present ✅
- NaN/Inf: not checked (would need to run validate script)
- **Overall: PASS but INSUFFICIENT for formal OOD**

## 5. Missing Data

See: `docs/layout_ood_state16_missing_data_plan.md`

Critical gaps:
- train_sim_id: need blocking_easy/medium/hard (~150 episodes)
- val/test ID: need heldout episodes (~20 episodes)
- test_layout_ood_direct: need passage_direct_* (~45 episodes)
- test_layout_ood_bypass: need passage_bypass_* (~9 episodes)
- **Total: ~286 more episodes needed**

## 6. Next Step

**DATASET_NOT_READY**

1. Run `scripts/collect_layout_ood_state16.py` with recommended MPPI config
2. Collect ~304 episodes covering all required families
3. Re-run audit and build dataset
4. Then train flat/object_centric/causality_aware on full dataset
5. Compare ID vs layout OOD generalization
