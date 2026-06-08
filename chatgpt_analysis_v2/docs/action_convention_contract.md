# Action Convention Contract

**Date:** 2026-05-24  
**Status:** DEFINITIVE — All code must follow this contract

---

## 1. Overview

This document defines the action convention for the Paper1 push manipulation system.

---

## 2. Action Types

| Quantity | Symbol | Unit | Range | Used By |
|----------|--------|------|-------|---------|
| Normalized action | `a_norm` | dimensionless | [-1, 1] | CEM/MPPI planner, env.step() |
| Physical velocity | `a_phys` | m/s | [-max_speed, max_speed] | Dataset `actions_physical`, model input |
| State displacement | `disp` | meters | [-max_speed*dt, max_speed*dt] | EE/object position update |

---

## 3. Conversion Formulas

```
a_phys = a_norm * max_speed_mps
disp = a_phys * control_dt
```

**Current values:**
- `max_speed_mps` = 0.5 (from MPPI best config sp050_T0.2)
- `control_dt` = 0.1 s

---

## 4. Component Contracts

### 4.1 Dataset (actions_physical)

**Source:** MPPI oracle rollout data  
**Unit:** Physical velocity (m/s)  
**Range:** [-max_speed_mps, max_speed_mps]  
**Evidence:** `actions_physical = actions_norm * max_speed_mps` (verified ratio = const)

### 4.2 Model Input

**Field:** `actions_physical` from `State16Dataset.__getitem__()`  
**Unit:** Physical velocity (m/s)  
**Range:** [-0.5, 0.5] for sp050 data  
**Training:** Model learns `state + a_phys → delta_object_pose`

### 4.3 env.step(action)

**Input:** Normalized action `a_norm` in [-1, 1]  
**Internal:** `velocity_cmd = a_norm * max_speed_mps`  
**Evidence:** Source code `mujoco_push_env.py` L626

### 4.4 Learned Rollout EE Update

**Correct:** `ee_xy += a_phys * control_dt`  
**Wrong:** `ee_xy += a_norm` (old bug, now fixed)

---

## 5. Planner → Model → Env Flow

```
Planner samples: a_norm ∈ [-1, 1]
         ↓
Convert to model action: a_phys = a_norm * max_speed_mps
         ↓
Model predicts: delta_object = model(state, a_phys)
         ↓
Convert to env action: a_env = a_norm
         ↓
Env executes: velocity_cmd = a_env * max_speed_mps
         ↓
EE update in rollout: ee_xy += a_phys * control_dt
```

---

## 6. Verification Checklist

- [ ] Dataset `actions_physical` range matches `max_speed_mps`
- [ ] Model trained on `actions_physical` (not `actions_norm`)
- [ ] `env.step()` receives normalized action
- [ ] Learned rollout uses `a_phys * control_dt` for EE update
- [ ] CEM/MPPI search in normalized space [-1, 1]
- [ ] All conversions are finite and clipped

---

## 7. Historical Bugs (Fixed)

### Bug 1: Action Scale Mismatch (2026-05-24)
- **Problem:** CEM searched in [-1, 1] but model expected physical velocity
- **Fix:** Convert `a_norm → a_phys` before feeding to model
- **Impact:** All 9 closed-loop runs failed (best_dist = final_dist)

### Bug 2: EE Update Error (2026-05-24)
- **Problem:** `ee_xy += action_sequence[t]` used raw planner action
- **Fix:** `ee_xy += a_phys * control_dt`
- **Impact:** EE position incorrectly updated in learned rollout

---

**Last Updated:** 2026-05-24
