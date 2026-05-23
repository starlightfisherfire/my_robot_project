# Current Sprint Control Console

**Last updated:** 2026-05-24 (v4: family_holdout + closed-loop readiness)

---

## 1. Current Stage

**Phase:** 16D pilot verification complete → MuJoCo closed-loop smoke preparation

---

## 2. Gate Status

| Gate | Name | Status | Evidence |
|------|------|--------|----------|
| 0 | Repo/Config consistency | ✅ PASS | All checks pass |
| 1-6 | Various | ✅ PASS | All passed |
| 6A | Learned rollout stack self-check | ✅ PASS | self_check JSON |
| 6B | Learned MPC internal smoke | ✅ PASS | smoke_pass=true |
| 7A | Training artifact precheck | ✅ PASS | eval_precheck.json |
| 7B | Offline dynamics eval | ✅ PASS | random_split + family_holdout |
| 8 | Learned MPC internal eval | ✅ PASS | random_split + family_holdout |
| 9 | Pilot report | ✅ DONE | docs/pilot_state16_mppi_stage2c_report.md |
| 10 | Family holdout verification | ✅ DONE | Finding supported |
| 11 | Closed-loop readiness | ✅ PASS | ready to implement smoke |

---

## 3. Key Findings (PILOT)

**causality_aware multi-step rollout stability advantage holds on held-out families:**
- Random split: Rollout@20 = 0.00108 vs flat 0.00386 (3.6×)
- Family holdout: Rollout@20 = 0.00253 vs flat 0.00397 (1.57×)

**flat has higher short-horizon cost improvement:**
- Both splits: flat mean_cost_improvement > causality_aware

**Trade-off is real:** one-step accuracy vs long-horizon stability

---

## 4. Next Steps

1. Implement `scripts/run_learned_mujoco_closed_loop.py`
2. Run 5-template closed-loop smoke with causality_aware
3. Compare closed-loop success rates across models
4. Scale to full test set

---

## 5. What NOT To Do

- ❌ Write as paper main result
- ❌ Claim causal variables discovered
- ❌ Run large-scale MuJoCo sweep yet
