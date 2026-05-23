# Current Sprint Control Console

**Last updated:** 2026-05-23 (v3: self-check PASS + smoke expansion)

---

## 1. Current Stage

**Phase:** Learned MPC internal smoke expansion (3-10 episodes)

---

## 2. Gate Status

| Gate | Name | Status | Evidence |
|------|------|--------|----------|
| 0 | Repo/Config consistency | ✅ PASS | train configs filled, self_check passes |
| 1 | MuJoCo env/reset/step | ✅ PASS | Gate 4 passed |
| 2 | Oracle-MPC ID/open | ✅ PASS | boundary_video_night2 mm-level |
| 3 | Oracle-MPC layout OOD | ⚠️ PARTIAL | mppi_stage2c 94.3% passage |
| 4 | canonical_state16 data | ⚠️ PARTIAL | 54 eps, success=False, smoke-usable |
| 5 | high-level model train smoke | ✅ PASS | train_state16_poc passed |
| 6 | learned rollout interface | ✅ PASS | check_model_interfaces.py all pass |
| 6A | learned rollout stack self-check | ✅ PASS | runs/self_check JSON 2026-05-23 |
| 6B | learned MPC internal smoke | ✅ PASS | smoke_pass=true, planned<zero |
| 7 | learned MPC 3-10 eps smoke | 🔧 RUNNING | expanding to verify stability |
| 8 | flat/object/causal OOD compare | ❌ FAIL | No OOD data |
| 9 | Real robot validation | ❌ UNKNOWN | No real data |

---

## 3. Self-Check PASS Record

**报告路径：** `runs/self_check/learned_rollout_stack_self_check.json`

| Check | Status |
|-------|--------|
| repo_import_check | ✅ PASS |
| py_compile_check | ✅ PASS |
| dummy_interface_check | ✅ PASS |
| cost_fn_check | ✅ PASS |
| dataset_check | ✅ PASS |
| checkpoint_check | ✅ PASS |
| learned_mpc_smoke_check | ✅ PASS |

**overall_status: PASS**

---

## 4. Current Task

1. ✅ Record PASS state in docs
2. 🔧 Run 3-episode learned MPC internal smoke
3. 🔧 Run 10-episode learned MPC internal smoke
4. ⬜ Data quality audit (state16_transition_audit)
5. ⬜ Decision: train three models or re-collect data

---

## 5. What NOT To Do

- ❌ 大规模训练
- ❌ OOD eval
- ❌ 真机实验
- ❌ 把 run_learned_mpc_eval.py 的 success_rate 当论文结果
