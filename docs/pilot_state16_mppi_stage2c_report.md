# Pilot Report: mppi_stage2c_state16 16D Representation Pilot

**Date:** 2026-05-24  
**Status:** PILOT — NOT paper main result  

---

## Executive Summary

All three models (flat, object_centric, causality_aware) trained successfully on canonical_state16 data converted from mppi_stage2c MPPI sweep. Offline dynamics eval shows **causality_aware has significantly better multi-step rollout stability** despite worse one-step accuracy, suggesting causal decomposition helps with error accumulation. Learned MPC internal eval shows all three models successfully guide CEM search, with flat having the highest cost improvement.

**Result level:** PILOT — not paper main result. Requires formal data protocol and MuJoCo closed-loop validation.

---

## 1. Preflight Results

**Path:** `runs/pilot_state16_mppi_stage2c/preflight_ready.json`  
**Status:** PASS (9/9 checks)

---

## 2. Data / Split

| Item | Value |
|------|-------|
| Dataset | `data/sim/mppi_stage2c_state16/` |
| Split | `configs/splits/mppi_stage2c_state16_pilot.yaml` |
| Train | 313 episodes |
| Val | 67 episodes |
| Test | 72 episodes |
| Total transitions | ~200K |
| Success rate | 79% |
| Families | blocking_hard(108), passage_direct_narrow(182), passage_bypass(162) |

---

## 3. Training Results

| Model | Params | Final Train Loss | Best Val Loss | Checkpoint |
|-------|--------|-----------------|---------------|------------|
| flat | 590,598 | 0.000417 | **0.000408** | `train_flat/flat/checkpoints/best.pt` |
| object_centric | 845,830 | 0.000519 | 0.000514 | `train_object_centric/object_centric/checkpoints/best.pt` |
| causality_aware | 1,240,710 | 0.000571 | 0.000570 | `train_causality_aware/causality_aware/checkpoints/best.pt` |

**Observation:** flat has lowest val_loss despite fewest parameters. Object_centric and causality_aware have higher loss, possibly due to Transformer training difficulty on this data scale.

---

## 4. Offline Dynamics Eval

| Model | One-step RMSE | dtheta RMSE | Rollout@5 | Rollout@10 | Rollout@20 |
|-------|---------------|-------------|-----------|------------|------------|
| flat | **0.01578** | 0.02729 | 0.00271 | 0.00311 | 0.00386 |
| object_centric | 0.01787 | 0.03090 | 0.00083 | 0.00204 | 0.00284 |
| causality_aware | 0.01818 | 0.03143 | **0.00062** | **0.00059** | **0.00108** |

**Key finding:** causality_aware has worst one-step RMSE but **dramatically better multi-step rollout stability**. At horizon 20, causality_aware rollout error is **3.6× smaller than flat** (0.00108 vs 0.00386). This suggests causal decomposition helps control error accumulation over long horizons.

**Interpretation:** The causal slot decomposition (z_stable, z_dynamics, z_affordance, z_nuisance) may provide a more structured latent space that is more robust to compounding prediction errors.

---

## 5. Learned MPC Internal Eval

| Model | Cost Improve Ratio | Mean Improvement | Mean Best Dist | Internal Success Ref |
|-------|--------------------|------------------|----------------|----------------------|
| flat | 30/30 (100%) | **2.5547** | 0.2745m | 0% |
| object_centric | 30/30 (100%) | 2.4018 | 0.2736m | 0% |
| causality_aware | 30/30 (100%) | 1.8727 | **0.2737m** | 0% |

**Key findings:**
- All three models achieve 100% planned_cost < zero_cost — CEM always finds better plans than zero action
- flat has highest cost improvement (2.55) — consistent with its lower one-step loss
- Mean best distance is similar across models (~0.27m) — all models reach similar physical proximity
- Internal success rate is 0% — learned rollout is too inaccurate to achieve <2cm precision, as expected for a pilot

---

## 6. Interpretation

### flat vs object/causal tradeoff
- **flat:** Lower one-step error, higher cost improvement, but worse multi-step rollout stability
- **causality_aware:** Higher one-step error but dramatically better multi-step rollout stability
- This suggests that causal decomposition provides structural regularization that reduces error accumulation

### Why causality_aware rollout is better
The causal slot decomposition forces the model to separate:
- **z_stable:** Features that don't change across actions
- **z_dynamics:** Features that change predictably with actions
- **z_affordance:** Task-relevant features
- **z_nuisance:** Irrelevant features

This separation may prevent the model from "confusing" stable and dynamic features during multi-step rollout, reducing compounding errors.

### Why flat has higher cost improvement
Flat's lower one-step error means it provides more accurate single-step predictions to CEM, leading to better immediate cost optimization. However, this advantage diminishes over longer horizons where error accumulation dominates.

---

## 7. Limitations

1. **Data source:** mppi_stage2c comes from MPPI parameter sweep, not a formal data collection protocol designed for Paper 1
2. **Learned MPC eval is internal:** Uses learned rollout for both planning and state update — not real MuJoCo closed-loop
3. **Not OOD evaluation:** All results are on random_episode_split/test from the same distribution
4. **Small scale:** 30 episodes for learned MPC eval, ~5000 samples for offline eval
5. **No ablation yet:** Haven't tested family_holdout_split or shape OOD

---

## 8. Next Actions

### Immediate (recommended)
1. **Run family_holdout_split eval** — test if causality_aware generalizes better to unseen families
2. **Investigate causality_aware slots** — examine z_stable, z_dynamics, z_affordance, z_nuisance for interpretability
3. **Scale up learned MPC eval** — increase to 100+ episodes per model

### Medium-term
4. **MuJoCo closed-loop small eval** — verify that internal planning advantage translates to real physics
5. **Formal data collection** — design proper train/test splits for Paper 1 main claim
6. **Visual state v2 pilot** — test if richer structured state improves all models

### Decision tree
- If causality_aware OOD advantage holds → proceed to formal OOD evaluation
- If flat OOD better → investigate if causal decomposition needs more data or different architecture
- If all models OOD poor → check target alignment, action scale, rollout update logic

---

## 9. Family Holdout Verification (2026-05-24)

### Offline Eval — family_holdout_split/test_ood (passage_bypass, 162 episodes)

| Model | One-step RMSE | Rollout@5 | Rollout@10 | Rollout@20 |
|-------|---------------|-----------|------------|------------|
| flat | **0.02385** | 0.00072 | 0.00191 | 0.00397 |
| object_centric | 0.02630 | 0.00052 | 0.00107 | 0.00319 |
| causality_aware | 0.02698 | **0.00031** | **0.00067** | **0.00253** |

**Finding holds:** causality_aware Rollout@20 is 1.57× smaller than flat on held-out families.

### Learned MPC Internal Eval — family_holdout/test_ood

| Model | Cost Improve Ratio | Mean Improvement | Mean Best Dist |
|-------|--------------------|------------------|----------------|
| flat | 30/30 (100%) | **2.4581** | 0.2984m |
| object_centric | 30/30 (100%) | 2.0442 | 0.2972m |
| causality_aware | 30/30 (100%) | 2.0852 | 0.2972m |

**Observation:** flat still has highest cost improvement; causality_aware/object_centric have slightly better mean best distance.

### Conclusion

**Preliminary finding SUPPORTED on family_holdout:** causality_aware maintains better multi-step rollout stability on held-out passage_bypass families. Flat retains higher short-horizon cost improvement. The trade-off between one-step accuracy and long-horizon stability is real and holds across splits.

---

## 10. MuJoCo Closed-Loop Learned MPC Smoke (2026-05-24)

### Setup
- Planner: CEM_FALLBACK (not MPPI, as learned rollout not compatible with MPPI interface)
- CEM: horizon=10, samples=128, elites=16, iterations=3
- MuJoCo: max_speed_mps=0.3, control_dt=0.1
- Templates: 3 open_space templates
- Max MPC steps: 30

### Results

| Model | Success | Mean Final Dist | Mean Best Dist | Mean Runtime |
|-------|---------|-----------------|----------------|-------------|
| flat | 0/3 | 0.2535m | 0.2535m | 27.0s |
| object_centric | 0/3 | 0.2579m | 0.2579m | 88.2s |
| causality_aware | 0/3 | 0.2540m | 0.2540m | 89.5s |

### Interpretation

1. **All three models achieve ~0.25m final distance** — consistent with internal eval (~0.27m)
2. **No model achieves success** (<2cm) — learned rollout is too inaccurate for precise control
3. **flat is 3× faster** — simpler model, faster inference
4. **All models move the object somewhat** — from ~0.27m initial to ~0.25m, a small but real improvement
5. **The learned rollout provides a useful but imprecise planning signal**

### Why no success?

The learned rollout models were trained on only ~131K samples from MPPI sweep data. The one-step prediction RMSE (~0.016m) compounds over 30 MPC steps, preventing precise convergence. This is expected for a pilot — the models need:
- More diverse training data
- Better action coverage
- Possibly MPPI (not CEM) for better exploration
- Possibly longer training or larger models

### Key Insight

**The closed-loop results are consistent with the offline/internal eval findings:**
- flat: fastest inference, similar accuracy
- causality_aware: similar accuracy, slower inference
- All models: useful but imprecise planning signal

The causality_aware multi-step rollout stability advantage from offline eval does NOT translate to better closed-loop performance in this smoke test. This suggests that for short-horizon CEM planning (10 steps), one-step accuracy matters more than long-horizon stability.
