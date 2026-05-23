# Experiment Matrix: Paper 1 Main Comparison

**Last updated:** 2026-05-23

---

## Core Comparison

**Research question:** Does causality-aware object-level representation reduce structural OOD degradation compared with flat and object-centric non-causal representations?

**Fixed executor:** CEM-MPC (same planner for all methods)

---

## Models

| Model | Encoder | z_dim | Parameters |
|-------|---------|-------|------------|
| flat | FlatEncoder | 256 | ~590K |
| object_centric | ObjectCentricEncoder | 256 | ~845K |
| causality_aware | CausalityAwareEncoder | 256 | ~1.2M |

**Interface contract:**
- Input: `[B, 6, 6, 16]` canonical_state16
- Output: `z [B, 256]`, `pred_delta [B, 3]`, `pred_subgoal [B, 3]`

---

## Evaluation Conditions

| Condition | Split | Families | Object |
|-----------|-------|----------|--------|
| ID | train_sim_id / test_sim_id | open_space, mild_offset | T |
| Layout OOD - blocking | test_sim_layout_ood_blocking | blocking | T |
| Layout OOD - passage | test_sim_layout_ood_narrow_passage | narrow_passage | T |
| Layout OOD - edge | test_sim_layout_ood_edge_goal | edge_goal | T |
| Shape OOD | test_sim_shape_ood_L | open_space, mild_offset, blocking | L |

---

## Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| success_rate | % episodes reaching goal (pos<Xcm, θ<Y°) | CEM-MPC eval |
| final_pos_dist_m | Final position error (m) | CEM-MPC eval |
| best_pos_dist_m | Best position error during episode | CEM-MPC eval |
| final_theta_error_deg | Final rotation error (deg) | CEM-MPC eval |
| dynamics_mse | One-step prediction MSE | Training eval |
| dynamics_rmse | One-step prediction RMSE | Training eval |
| subgoal_mse | Subgoal prediction MSE | Training eval |
| rollout_error | Multi-step rollout accumulation error | Rollout eval |
| collision_rate | % steps with collision | CEM-MPC eval |
| contact_rate | % steps with EE-object contact | CEM-MPC eval |
| OOD_gap | ID_performance - OOD_performance | Computed |

---

## Planner Config (Fixed)

```yaml
planner: cem_mpc
horizon: 10
num_samples: 1024
num_elites: 64
num_iterations: 5
action_dim: 2
action_range: [-0.02, 0.02]
```

**Cost weights (shared):**
- w_pos: 10.0
- w_theta: 2.0
- w_reach: 5.0
- w_collision: 20.0
- w_smooth: 0.1

---

## Training Config (Shared)

```yaml
batch_size: 128
lr: 0.0003
weight_decay: 0.000001
grad_clip_norm: 1.0
epochs: 50 (full) / 5 (POC) / 2 (smoke)
normalizer: fit on train only
```

---

## Experiment Flow

```
1. Data Collection
   └─ canonical_state16 episodes per family

2. Training (per model)
   └─ train_high_level.py --model {flat|object_centric|causality_aware}

3. Evaluation
   ├─ One-step: dynamics_rmse, subgoal_rmse
   ├─ Multi-step: rollout_error at H={1,5,10}
   └─ CEM-MPC: success_rate, final_dist, collision_rate

4. Comparison
   ├─ ID table
   ├─ Layout OOD table
   ├─ Shape OOD table
   └─ OOD gap analysis
```

---

## Paper Tables (Planned)

### Table 1: ID Performance
| Model | success_rate | final_dist | dynamics_rmse |
|-------|-------------|------------|---------------|
| flat | | | |
| object_centric | | | |
| causality_aware | | | |

### Table 2: Layout OOD Performance
| Model | blocking | narrow_passage | edge_goal |
|-------|----------|----------------|-----------|
| flat | | | |
| object_centric | | | |
| causality_aware | | | |

### Table 3: OOD Gap
| Model | ID→blocking | ID→passage | ID→edge | Mean OOD gap |
|-------|-------------|------------|---------|--------------|
| flat | | | | |
| object_centric | | | | |
| causality_aware | | | | |
