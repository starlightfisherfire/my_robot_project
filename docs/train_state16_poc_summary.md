# train_state16_poc Summary

## 1. What was tested

Proof-of-concept training pipeline for Paper 1's three high-level representation models (flat, object-centric, causality-aware) on structured state data. **This is NOT an OOD result.** This is a pipeline validation.

## 2. Data status

- **8 episodes**, all from MPPI sweep data
- All open-space, T-shape only, train split
- ~1944 training samples, ~243 validation samples
- **Missing:** blocking, narrow_passage, edge_goal, L-shape OOD data
- Data format: structured state [steps, 6, 16] + actions + poses + goal

## 3. Model status

All three models can train and produce finite predictions:

| Model | Trainable | Params | Loss decrease |
|-------|-----------|--------|-------------|
| flat | ✅ | 590,598 | ✅ |
| object_centric | ✅ | 845,830 | ✅ |
| causality_aware | ✅ | 1,240,710 | ✅ |

## 4. Loss summary

| Model | train_loss_start | train_loss_end | val_loss_start | val_loss_end | dynamics_rmse | subgoal_rmse |
|-------|----------------|---------------|---------------|-------------|-------------|-------------|
| flat | 0.0056 | 0.0021 | 0.0025 | 0.0017 | 0.0064 | 0.0887 |
| object_centric | 0.0049 | 0.0034 | 0.0026 | 0.0018 | 0.0073 | 0.0911 |
| causality_aware | 0.0061 | 0.0035 | 0.0023 | 0.0018 | 0.0067 | 0.0931 |

## 5. Known limitations

- ⚠️ Data is too small: 8 episodes, ~2K samples
- ⚠️ No OOD test data: no blocking, narrow_passage, L-shape
- ⚠️ Only one-step eval, no multi-step rollout
- ⚠️ All data is from MPPI sweep (open-space, T-shape), not diverse
- ⚠️ Cannot compare representation OOD quality yet
- ⚠️ This is an engineering pipeline validation only

## 6. Next step

**If pipeline is successful (✅):**
- Collect formal layout_ood_state16_v0 data:
  - train: open + blocking layouts
  - test: narrow_passage, edge_goal layouts, L-shape objects
- Train all three models on the full dataset
- Compare OOD generalization across encoder variants
- Run multi-step rollout evaluation

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Gate report | runs/train_state16_poc/gate_report.md |
| Flat checkpoint | runs/train_state16_poc/flat/checkpoints/best.pt |
| Object-centric checkpoint | runs/train_state16_poc/object_centric/checkpoints/best.pt |
| Causal-aware checkpoint | runs/train_state16_poc/causality_aware/checkpoints/best.pt |
| Eval metrics | runs/train_state16_poc/eval/*_metrics.csv |
| Train logs | runs/train_state16_poc/*/train_log.jsonl |
| Config | configs/experiments/train_state16_poc.yaml |
