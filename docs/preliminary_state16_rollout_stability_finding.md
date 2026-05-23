# Preliminary Finding: Rollout Stability of Causality-Aware Representation

**Date:** 2026-05-24  
**Status:** PRELIMINARY — not paper main result  

---

## 1. Executive Summary

**Question:** Does causality_aware encoder maintain better long-horizon rollout stability on held-out families?

**Answer:** Yes, preliminary evidence supports this finding on family_holdout_split/test_ood (passage_bypass families), though the advantage is smaller than on random_split/test.

---

## 2. Evidence Table

### Random split/test (random_episode_split/test)

| Model | One-step RMSE | Rollout@5 | Rollout@10 | Rollout@20 |
|-------|---------------|-----------|------------|------------|
| flat | **0.01578** | 0.00271 | 0.00311 | 0.00386 |
| object_centric | 0.01787 | 0.00083 | 0.00204 | 0.00284 |
| causality_aware | 0.01818 | **0.00062** | **0.00059** | **0.00108** |

### Family holdout test_ood (passage_bypass families)

| Model | One-step RMSE | Rollout@5 | Rollout@10 | Rollout@20 |
|-------|---------------|-----------|------------|------------|
| flat | **0.02385** | 0.00072 | 0.00191 | 0.00397 |
| object_centric | 0.02630 | 0.00052 | 0.00107 | 0.00319 |
| causality_aware | 0.02698 | **0.00031** | **0.00067** | **0.00253** |

### Learned MPC internal eval — random_split/test

| Model | Cost Improve Ratio | Mean Improvement | Mean Best Dist |
|-------|--------------------|------------------|----------------|
| flat | 30/30 (100%) | **2.5547** | 0.2745m |
| object_centric | 30/30 (100%) | 2.4018 | 0.2736m |
| causality_aware | 30/30 (100%) | 1.8727 | 0.2737m |

### Learned MPC internal eval — family_holdout/test_ood

| Model | Cost Improve Ratio | Mean Improvement | Mean Best Dist |
|-------|--------------------|------------------|----------------|
| flat | 30/30 (100%) | **2.4581** | 0.2984m |
| object_centric | 30/30 (100%) | 2.0442 | 0.2972m |
| causality_aware | 30/30 (100%) | 2.0852 | 0.2972m |

---

## 3. Key Finding

**Preliminary evidence supports:**

"One-step prediction accuracy does not fully predict long-horizon rollout stability."

- flat has the lowest one-step RMSE in both random_split and family_holdout
- causality_aware has the lowest rollout RMSE at horizons 5, 10, and 20 in both splits
- On family_holdout, causality_aware Rollout@20 is **1.57× smaller** than flat (0.00253 vs 0.00397)
- On random_split, the ratio was even larger: **3.6× smaller** (0.00108 vs 0.00386)

**The causality_aware advantage in multi-step rollout stability holds on held-out families.**

**Caveats:**
- Family_holdout test_ood = passage_bypass (54+54+54 episodes), not a broad OOD test
- Learned MPC internal eval shows flat has higher cost improvement — planner-facing utility depends on more than rollout stability
- Internal eval ≠ real MuJoCo closed-loop

---

## 4. Interpretation

1. **flat: better single-step, worse long-horizon** — flat encoder optimizes well for one-step accuracy but accumulates errors faster over many steps

2. **causality_aware: worse single-step, better long-horizon** — causal slot decomposition (z_stable/z_dynamics/z_affordance/z_nuisance) provides structural regularization that reduces error accumulation

3. **object_centric: middle ground** — object-centric structure helps somewhat, but less than full causal decomposition

4. **Planner-facing utility is nuanced** — flat's higher cost improvement suggests that for short-horizon CEM planning (5 steps), one-step accuracy matters more. But for longer planning horizons, rollout stability may dominate.

---

## 5. Limitations

1. mppi_stage2c comes from MPPI parameter sweep, not a formal data protocol
2. Learned MPC internal eval uses learned rollout for both planning and state update — not real MuJoCo closed-loop
3. Family_holdout test_ood = passage_bypass only — not a broad OOD test
4. Current causality_aware is a factorized representation structure — cannot claim discovery of true causal variables
5. Need formal data protocol and MuJoCo closed-loop to write paper conclusions

---

## 6. Next Actions

1. **MuJoCo closed-loop learned MPC smoke** — verify that internal planning advantage translates to real physics
2. **Slot diagnostics** — examine what z_stable, z_dynamics, z_affordance, z_nuisance actually encode
3. **Scale up** — more episodes, more families, shape OOD
4. **Visual state v2 profile ablation** — test if richer structured state improves all models
