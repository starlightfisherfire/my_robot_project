# Planner Config Policy

**Last updated:** 2026-05-12

This document clarifies the CEM-MPC parameter policy to avoid config confusion and ensure fair comparison.

---

## Problem Statement

CEM-MPC has many parameters (horizon, num_samples, num_elites, num_iterations, etc.). Different use cases require different configs:

1. **Smoke test:** Fast interface validation
2. **Oracle capacity:** Upper bound with true dynamics
3. **Learned eval:** Fair comparison across encoder variants

**Risk:** Using different planner configs for different encoder variants invalidates the comparison.

---

## Core CEM-MPC Parameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `horizon` | Planning horizon (number of steps) | Longer horizon = more search space, but learned model accumulates error |
| `action_dim` | Action dimensionality | Fixed at 2 for planar push |
| `num_samples` | Number of candidate sequences per iteration | More samples = better exploration, slower |
| `num_elites` | Number of top candidates to refit distribution | More elites = more robust, but less selective |
| `num_iterations` | Number of CEM refinement iterations | More iterations = better convergence, slower |
| `action_low` / `action_high` | Action bounds | Fixed at [-1, 1] for normalized actions |
| `init_std` | Initial sampling std | Higher std = more exploration initially |
| `smoothing` | Temporal smoothing factor | Higher smoothing = smoother mean/std updates |

---

## Three Config Types

### A. smoke_test Config

**Purpose:** Interface validation, fast debugging

**Characteristics:**
- Lightweight and fast
- Just needs to run without crashing
- Does not need to solve task

**Example:**
```yaml
horizon: 80
num_samples: 1536
num_elites: 128
num_iterations: 7
```

**Current usage:**
- `scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc` (default args)

**Status:** ✅ Used in current MuJoCo Oracle-MPC interface smoke test

---

### B. oracle_capacity_strong Config

**Purpose:** Establish upper bound with true MuJoCo dynamics

**Characteristics:**
- Can be stronger than learned eval config
- Goal: success_rate > 80% on train_sim_id
- Establishes what is possible with perfect dynamics
- Used to validate that task is solvable

**Update (2026-05-12):** Concrete configs are now defined in `docs/planner_capacity_protocol.md`:

| Config | horizon | execute_steps | max_mpc_steps | num_samples | num_elites | num_iterations | Total budget |
|--------|---------|---------------|---------------|-------------|------------|----------------|--------------|
| **c23_precise** (main baseline) | 80 | 20 | 25 | 1024 | 96 | 5 | 500 steps |
| **c23_strict600** (confirmatory) | 80 | 20 | 30 | 1024 | 96 | 5 | 600 steps |
| **c25_fast** (conservative backup) | 80 | 30 | 20 | 512 | 48 | 5 | 600 steps |

**Results (boundary_video_night2, 2026-05-12):**
- c23_precise: mean_final_pos_error ≈ 2.70mm, success_pos_1cm_rate = 1.0
- c25_fast: mean_final_pos_error ≈ 2.38mm, success_pos_0p5cm_rate = 1.0

**Important:**
- This config does NOT need to match learned eval config
- Oracle can use longer horizon because it has perfect dynamics
- Learned model cannot use very long horizon due to error accumulation

---

### C. learned_eval_default Config

**Purpose:** Fair comparison across flat/object/causal encoder variants

**Characteristics:**
- **MUST be identical for all three encoder variants**
- Shorter horizon than oracle_capacity_strong (to avoid error accumulation)
- Tuned for learned model characteristics
- Used for final Paper 1 comparison

**Example (future):**
```yaml
horizon: 40-60  # shorter than oracle strong
num_samples: 1024  # same for all variants
num_elites: 128  # same for all variants
num_iterations: 5-7  # same for all variants
```

**Critical rules:**
1. Flat / object / causal MUST use identical config
2. Do NOT tune separately for each encoder variant
3. Can do horizon ablation later, but default config must be shared

**Status:** ⬜ NOT DEFINED YET (waiting for learned model training)

---

## Why Oracle Strong ≠ Learned Eval

### Reason 1: Error Accumulation
- **Oracle:** Uses true MuJoCo dynamics, no prediction error
- **Learned:** Accumulates prediction error over horizon
- **Impact:** Learned model with horizon=100 may diverge from reality

### Reason 2: Computational Cost
- **Oracle:** Each rollout is expensive (MuJoCo simulation)
- **Learned:** Each rollout is cheap (neural network forward pass)
- **Impact:** Learned eval can afford more samples, but not longer horizon

### Reason 3: Experimental Purpose
- **Oracle:** Establishes upper bound ("what is possible")
- **Learned:** Compares representation quality ("which is better")
- **Impact:** Oracle can be stronger to prove task is solvable

---

## Current Config Status

### configs/planner/cem_mpc.yaml

**Current content:**
```yaml
cem:
  horizon: 10
  num_candidates: 512
  num_elites: 64
  num_iterations: 5
```

**Interpretation:**
- This is a **placeholder config** from early development
- horizon=10 is very short (1 second at 0.1s control_dt)
- This is NOT the oracle_capacity_strong config
- This is NOT the learned_eval_default config
- This is NOT the smoke_test config (which uses horizon=80)

**Action needed:**
- Do NOT assume this is the final config
- Do NOT use this for Oracle-MPC capacity tuning
- Do NOT use this for learned eval
- Update this file after configs are finalized

---

## Current Oracle Capacity Config (2026-05-12)

config23 / c23_precise 已作为 oracle capacity config，见 docs/planner_capacity_protocol.md。

---

## Recommended Workflow

### Step 1: Tune Oracle-MPC for Task Success
1. Start with smoke_test config (horizon=80, num_samples=1536, etc.)
2. If success_rate < 80%, increase horizon / num_samples / num_iterations
3. Tune cost weights if needed
4. Save final config as `configs/planner/oracle_capacity_strong.yaml`
5. Document: "This is the upper bound config with true dynamics"

### Step 2: Define Learned Eval Config
1. After learned models are trained, test different horizons
2. Find horizon where learned model does not diverge too much
3. Likely horizon=40-60 (shorter than oracle strong)
4. Keep num_samples / num_elites / num_iterations reasonable
5. Save as `configs/planner/learned_eval_default.yaml`
6. Document: "This is the shared config for flat/object/causal comparison"

### Step 3: Horizon Ablation (Optional)
1. After main comparison, test different horizons
2. Plot success_rate vs horizon for each encoder variant
3. Analyze: does causality-aware degrade slower with longer horizon?
4. This is a secondary analysis, not the main comparison

---

## Config Snapshot Protocol

**Rule:** Every experiment run must snapshot its exact config.

**Implementation:**
```python
# In train/eval scripts
import shutil
from pathlib import Path

run_dir = Path(f"runs/{run_id}")
config_snapshot_dir = run_dir / "config_snapshot"
config_snapshot_dir.mkdir(parents=True, exist_ok=True)

# Copy all configs used
shutil.copy("configs/planner/cem_mpc.yaml", config_snapshot_dir / "cem_mpc.yaml")
shutil.copy("configs/planner/cost_weights.yaml", config_snapshot_dir / "cost_weights.yaml")
shutil.copy("configs/train/flat.yaml", config_snapshot_dir / "train_flat.yaml")
```

**Purpose:**
- Reproducibility
- Detect accidental config changes
- Audit trail for paper

---

## Common Pitfalls

### Pitfall 1: Using smoke_test config for Oracle capacity
**Problem:** horizon=80 may not be enough for task success
**Fix:** Tune separately for oracle_capacity_strong

### Pitfall 2: Using oracle_capacity_strong config for learned eval
**Problem:** Learned model accumulates error over long horizon
**Fix:** Use shorter horizon for learned eval

### Pitfall 3: Tuning planner separately for each encoder variant
**Problem:** Unfair comparison, cannot isolate representation quality
**Fix:** Use identical learned_eval_default config for all variants

### Pitfall 4: Assuming configs/planner/cem_mpc.yaml is final
**Problem:** This is a placeholder from early development
**Fix:** Define oracle_capacity_strong and learned_eval_default separately

### Pitfall 5: Not snapshotting configs
**Problem:** Cannot reproduce results, cannot detect config drift
**Fix:** Always snapshot configs into runs/{run_id}/config_snapshot/

---

## Decision Tree

```
Is this for interface validation?
    YES → Use smoke_test config (fast, lightweight)
    NO ↓

Is this for Oracle-MPC capacity with true dynamics?
    YES → Use oracle_capacity_strong config (can be stronger)
           See docs/planner_capacity_protocol.md for concrete configs
    NO ↓

Is this for learned model evaluation?
    YES → Use learned_eval_default config (MUST be identical for all variants)
    NO ↓

Is this for horizon ablation?
    YES → Vary horizon, but keep other params fixed
    NO ↓

Unknown use case → Ask user for clarification
```

---

## Summary

1. **Three config types:** smoke_test, oracle_capacity_strong, learned_eval_default
2. **Oracle strong ≠ learned eval:** Oracle can use longer horizon
3. **Learned eval MUST be identical:** Flat/object/causal use same config
4. **Current configs/planner/cem_mpc.yaml is placeholder:** Do NOT assume it's final
5. **Oracle strong configs now defined:** See docs/planner_capacity_protocol.md (c23_precise, c23_strict600, c25_fast)
6. **Current oracle capacity config:** c23_precise, see docs/planner_capacity_protocol.md
7. **Always snapshot configs:** Reproducibility and audit trail

---

## Next Actions

1. ✅ Oracle-MPC task capacity partially established (millimeter precision on open_space/mild_offset)
2. ⬜ Strict pose stop smoke test pending
3. ⬜ Implement obstacles (Gate 8) for true oracle_capacity_strong validation
4. ⬜ After learned model training, define learned_eval_default config
5. ⬜ Implement config snapshot protocol in train/eval scripts
6. ⬜ Update configs/planner/cem_mpc.yaml with clear documentation
