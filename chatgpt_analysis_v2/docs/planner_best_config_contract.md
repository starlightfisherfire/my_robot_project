# Planner Best Config Contract

**Date:** 2026-05-24  
**Status:** DEFINITIVE — Use these configs for closed-loop experiments

---

## 1. MPPI Best Config

**Source:** `runs/mppi_stage2c_20260520_194856/summary_by_top_config.csv`  
**Config Label:** `sp050_T0.2`  
**Success Rate:** 92.4% (pose_2mm_10deg)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | 0.2 | Lower = more exploitation |
| `speed_mps` | 0.5 | max_speed_mps |
| `horizon` | 100 | Planning horizon |
| `num_samples` | 1024 | Samples per iteration |
| `init_std` | 0.5 | Initial noise std |
| `execute_steps` | 10 | Steps before re-plan |
| `max_mpc_steps` | 100 | Max MPC iterations |

**Derived values:**
- `control_dt` = 0.1 s
- `max_speed_mps` = 0.5 m/s
- `action_range` = [-1, 1] (normalized)

---

## 2. MPPI Alternative Config

**Config Label:** `sp030_T0.1`  
**Success Rate:** 90.3%

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.1 |
| `speed_mps` | 0.3 |
| `horizon` | 100 |
| `num_samples` | 1024 |
| `init_std` | 0.5 |
| `execute_steps` | 10 |
| `max_mpc_steps` | 100 |

---

## 3. CEM Config (for Learned Rollout)

**Source:** Derived from MPPI best + CEM defaults  
**Purpose:** Learned rollout planning

| Parameter | Value | Notes |
|-----------|-------|-------|
| `horizon` | 20 | Increased from 10 |
| `num_samples` | 512 | Increased from 128 |
| `num_elites` | 64 | Increased from 16 |
| `num_iterations` | 5 | Increased from 3 |
| `action_low` | -1.0 | Normalized space |
| `action_high` | 1.0 | Normalized space |
| `init_std` | 0.3 | Reduced from 0.5 |
| `smoothing` | 0.2 | Keep default |

**Action conversion:**
```python
# CEM samples in [-1, 1] (normalized)
# Convert to physical for model:
a_phys = a_norm * max_speed_mps  # * 0.5
# Convert to env:
a_env = a_norm  # env.step expects normalized
```

---

## 4. Previous Failed Config (DO NOT USE)

**Date:** 2026-05-24 first attempt  
**Failure:** Action scale mismatch

| Parameter | Old Value | Problem |
|-----------|-----------|---------|
| `horizon` | 10 | Too short |
| `num_samples` | 128 | Too few |
| `num_elites` | 16 | Too few |
| `num_iterations` | 3 | Too few |
| `action_low/high` | -1.0/1.0 | No conversion to physical |

**Root cause:** Model received `a_norm` instead of `a_phys`

---

## 5. Implementation Notes

### 5.1 Learned Rollout Adapter

```python
class LearnedRolloutCostFn:
    def __call__(self, action_sequence_norm):
        for a_norm in action_sequence_norm:
            a_phys = a_norm * max_speed_mps  # Convert to physical
            delta = model(state, a_phys)      # Model expects physical
            # Update state...
```

### 5.2 EE Update

```python
# Correct:
ee_xy += a_phys * control_dt

# Wrong (old):
ee_xy += a_norm
```

---

**Last Updated:** 2026-05-24
