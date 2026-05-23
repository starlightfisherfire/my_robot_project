# Results Snapshot: Paper 1

**Last updated:** 2026-05-23

---

## ⚠️ No Paper Main Results Yet

All available results are internal smoke / capacity exploration only.

---

## 1. Self-Check Status

**Overall: ✅ PASS**

Report: `runs/self_check/learned_rollout_stack_self_check.json`

| Check | Status |
|-------|--------|
| repo_import_check | ✅ PASS |
| py_compile_check | ✅ PASS |
| dummy_interface_check | ✅ PASS |
| cost_fn_check | ✅ PASS |
| dataset_check | ✅ PASS |
| checkpoint_check | ✅ PASS |
| learned_mpc_smoke_check | ✅ PASS |

---

## 2. Learned MPC Internal Smoke

**⚠️ These are internal smoke tests, NOT paper results.**

### 2.1 3-Episode Smoke
Report: `runs/learned_mpc_eval/flat_internal_smoke_3eps.json`

| Metric | Value |
|--------|-------|
| smoke_pass | true |
| episodes | 3 |
| planned_cost < zero_cost | 3/3 (100%) |
| mean cost improvement | 1.93 |

### 2.2 10-Episode Smoke
Report: `runs/learned_mpc_eval/flat_internal_smoke_10eps.json`

| Metric | Value |
|--------|-------|
| smoke_pass | true |
| episodes | 10 |
| planned_cost < zero_cost | 10/10 (100%) |
| mean cost improvement | 2.75 |

**Interpretation:**
- CEM consistently finds better plans than zero action
- Model predictions are consistent across episodes
- But success_rate = 0% (expected: internal smoke only, no real physics)

---

## 3. Data Quality Audit

Report: `runs/self_check/state16_transition_audit.json`

| Metric | Value |
|--------|-------|
| Total episodes | 54 |
| Total transitions | 9900 |
| Valid episodes | 54 (100%) |
| Success episodes | 0 (0%) |
| Contact transitions | 3239 (32.7%) |
| Object moved transitions | 1045 (10.6%) |
| **Zero movement transitions** | **8855 (89.4%)** |
| Recommendation | **RE_COLLECT** |

**By family:**
| Family | Episodes |
|--------|----------|
| open | 18 |
| blocking_easy | 10 |
| blocking_medium | 8 |
| passage_direct_wide | 8 |
| passage_direct_narrow | 10 |

**Key finding:** 89.4% of transitions have zero object movement. This means the Oracle-MPC during data collection was mostly pushing into empty space or failing to contact the object. The data is too low quality for meaningful dynamics learning.

---

## 4. Oracle-MPC Capacity (Debug Only)

| Run | Success Rate | Best Dist |
|-----|-------------|-----------|
| horizon140_verify | 95% | 0.0001m |
| mppi_stage2c | 75% (passage) | — |
| heavy_pusher_700g | 51% | 0.0001m |

**Conclusion:** Oracle-MPC CAN solve these tasks with the right config. Data collection used suboptimal configs.

---

## 5. Training POC

- 8 episodes, open_space only
- All three models train successfully
- dynamics_rmse: 0.0064-0.0073
- **Cannot serve as paper result**
