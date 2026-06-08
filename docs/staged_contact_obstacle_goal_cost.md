# Staged Contact-Obstacle-Goal Cost

## Overview

`staged_contact_obstacle_goal_cost` is the primary cost function for MuJoCo Oracle CEM/MPPI sweep. It replaces the original `rollout_cost` (now `cost_mode="current"`, kept only as baseline).

## Key Design Decisions

### Object Geometry: Circle Approximation

**Current limitation:** The object is modeled as a circle with `object_radius=0.034`. This is an approximation — the real PushT object is a T-shape. Collision/proximity checks use this circle, which means:
- Corners of the T-shape may penetrate obstacles undetected
- Clearance margins must be set conservatively

**Risk for narrow passages:** If `margin` is too large, the object cannot pass through gaps between obstacles even when geometrically feasible. If too small, collisions go undetected. The current `margin=0.03` is a compromise.

### Progress Cost: Dense Progress Shaping (NOT Straight-Line Alignment)

`obstacle_aware_progress_cost` is **not** a replacement for `push_alignment_cost` in the naive sense. It is fundamentally different:

```
potential(x) = ||x - goal||^2 + obstacle_repulsion(x)
progress_decrease = potential_t - potential_{t+1}
cost = mean(max(0, progress_margin - progress_decrease))
```

**What it does:**
- Defines a potential field that combines goal distance and obstacle repulsion
- Penalizes *lack of progress* (insufficient potential decrease per step)
- A **static** trajectory has cost > 0 (unlike the old version)
- Allows any feasible path — the object can go left, right, around, as long as potential decreases

**What it does NOT do:**
- Does NOT assume straight-line motion
- Does NOT penalize lateral movement
- Does NOT require alignment between motion direction and goal direction

### Staged Cost is the Oracle Sweep Default

- `cost_mode="staged_contact_obstacle_goal"` → default for next MuJoCo Oracle CEM/MPPI sweep
- `cost_mode="current"` → kept only as backward-compatible baseline, not used in new sweeps

## Why Not Default to push_alignment?

The original `push_alignment_cost` assumes the object should move in a straight line from its initial position to the goal. This assumption fails when obstacles are present because:

1. **Straight-line paths may collide with obstacles** — forcing the object through obstacles is not feasible
2. **Side-ward movement is necessary** — to navigate around obstacles, the object must move laterally, which push_alignment penalizes
3. **Obstacle-aware planning requires detours** — the optimal path may not be aligned with the initial-goal direction

## Why Disable no_contact and reach?

In the staged cost:

- **`reach_cost` is disabled** (`w_reach = 0.0`): The early_contact cost already encourages the EE to approach the object. Using both would create redundant gradients.

- **`no_contact_cost` is disabled** (`w_no_contact = 0.0`): The contact_duration cost encourages maintaining contact once established. The no_contact cost is too coarse (binary) and doesn't provide useful gradient for the staged approach.

## Contact Cost Distinctions

Three contact-related costs work together:

| Cost | What It Measures | When It Helps |
|------|------------------|---------------|
| `first_contact_cost` (early_contact) | How quickly the EE first touches the object | Phase 1: Getting to the object |
| `persistent_contact_cost` | Penalizes losing contact after first touch | Phase 2: Maintaining contact |
| `contact_duration_cost` | Overall contact rate (1 - contact_rate) | Phase 2: Longer contact is better |

**Key difference:** `first_contact_cost` cares about *when* contact starts. `persistent_contact_cost` cares about *not losing* contact after it starts. `contact_duration_cost` cares about *total time* in contact.

## object_obstacle vs collision

Two levels of obstacle interaction for the object:

| Cost | Threshold | Gradient | Purpose |
|------|-----------|----------|---------|
| `object_obstacle_proximity_cost` | Soft (margin zone) | Smooth gradient before collision | Guides CEM/MPPI samples away from obstacles |
| `object_collision_cost` | Hard (geometric overlap) | Binary collision detection | Strong penalty when collision actually occurs |

**Why both?** The proximity cost provides smooth gradient signal even when no collision occurs, helping CEM/MPPI find obstacle-avoiding paths. The collision cost provides a strong penalty when the object actually collides, ensuring collision-free final plans.

**Horizon normalization:** Both `object_obstacle_proximity_cost` and `ee_obstacle_proximity` are normalized by T (mean over timesteps), so cost magnitude is invariant to horizon length.

## obstacle_aware_progress

The `obstacle_aware_progress_cost` replaces `push_alignment_cost` for obstacle-aware navigation:

```
potential(x) = ||x - goal||^2 + obstacle_repulsion(x)
progress_decrease = potential_t - potential_{t+1}
cost = mean(max(0, progress_margin - progress_decrease))
```

**Key properties:**
- `progress_margin` (default `1e-4`) sets the minimum expected potential decrease per step
- Static trajectory → cost > 0 (penalizes stall)
- Moving toward goal → cost ≈ 0
- Moving away from goal or into obstacle → cost > 0

**Why it's better than push_alignment:**
- Doesn't assume straight-line motion
- Accounts for obstacles in the potential field
- Encourages progress along any feasible path
- Dense signal even when object is far from goal

## terminal_hold

The `terminal_hold_cost` solves the "best_dist small but final_dist large" problem:

**Problem:** The object passes through the goal midway during the rollout, but drifts away by the end.

**Solution:** Compute average pose error over the last K timesteps.

```
terminal_hold = mean(pose_error over last K steps)
```

## make_staged_cost_weights()

Factory function that returns `CostWeights` tuned for staged cost:

```python
from src.planners.cost_functions import make_staged_cost_weights

w = make_staged_cost_weights()
# w_reach = 0.0, w_no_contact = 0.0, w_push_alignment = 0.0
# w_early_contact = 4.0, w_persistent_contact = 4.0
# w_collision = 30.0, w_collision_step = 2.0
# w_action = 0.02, w_smooth = 0.05
```

When `weights=None` in `staged_contact_obstacle_goal_cost`, these defaults are used automatically.

## Cost Terms Summary

| Term | Weight | Normalized | Stage | Purpose |
|------|--------|------------|-------|---------|
| `final_pos` | w_pos = 10.0 | — | Final | Position error at rollout end |
| `final_theta` | w_theta = 2.0 | — | Final | Orientation error at rollout end |
| `terminal_hold` | w_terminal_hold = 10.0 | — | Final | Stay at goal after reaching it |
| `early_contact` | w_early_contact = 4.0 | — | Phase 1 | Contact object quickly |
| `persistent_contact` | w_persistent_contact = 4.0 | — | Phase 2 | Don't lose contact |
| `contact_duration` | w_contact_duration = 2.0 | — | Phase 2 | Maintain long contact |
| `obstacle_aware_progress` | w_progress = 5.0 | — | Phase 3 | Dense progress shaping |
| `object_obstacle_proximity` | w_object_obstacle = 25.0 | mean/T | Phase 3 | Soft repulsion from obstacles |
| `object_collision_any` | w_object_collision = 50.0 | — | Phase 3 | Hard penalty for any collision |
| `object_collision_step` | w_object_collision_step = 3.0 | count/T | Phase 3 | Collision rate |
| `ee_obstacle_proximity` | w_ee_obstacle = 8.0 | mean/T | All | EE avoids obstacles |
| `collision_any` | w_collision = 30.0 | — | All | EE collision penalty |
| `collision_step` | w_collision_step = 2.0 | count/T | All | EE collision rate |
| `action` | w_action = 0.02 | — | All | Regularize action magnitude |
| `smooth` | w_smooth = 0.05 | — | All | Penalize action changes |

**Disabled in staged cost:**
- `w_reach = 0.0` (replaced by early_contact)
- `w_no_contact = 0.0` (replaced by contact_duration)
- `w_push_alignment = 0.0` (replaced by obstacle_aware_progress)

## Usage

### cost_mode Selection

```python
from src.planners.cost_functions import rollout_cost, make_staged_cost_weights

# Staged cost (default for Oracle sweep)
cost = rollout_cost(..., cost_mode="staged_contact_obstacle_goal")

# Original cost (baseline only)
cost = rollout_cost(..., cost_mode="current")
```

### Direct Call

```python
from src.planners.cost_functions import staged_contact_obstacle_goal_cost

result = staged_contact_obstacle_goal_cost(
    predicted_object_poses=poses,
    ee_positions=ee,
    action_sequence=actions,
    goal_pose=goal,
    weights=None,  # Uses make_staged_cost_weights() defaults
    contact_flags=contact,
    obstacle_positions=obs_pos,
    obstacle_radii=obs_rad,
    return_breakdown=True,
)
```

## Self-Check

15 checks verify all cost terms:

```bash
PYTHONPATH=. python scripts/self_check_staged_cost.py
```

| Check | What It Verifies |
|-------|-----------------|
| terminal_hold_check | Last K steps near goal < mid-pass-through |
| early_contact_check | Early < late < no contact |
| persistent_contact_check | Maintained < contact then lose |
| contact_duration_check | High rate < low rate |
| obstacle_aware_progress_check | Around obstacle < through obstacle < away |
| object_obstacle_check | Near obstacle > far from obstacle |
| collision_check | Collision > no collision |
| push_alignment_disabled_check | No push_alignment in breakdown |
| current_backward_compatibility_check | cost_mode="current" unchanged |
| finite_check | All values finite |
| static_object_progress_check | Static trajectory cost > moving cost |
| around_vs_through_obstacle_check | Around total < through total |
| narrow_passage_clearance_check | Clearance path < hit obstacle |
| horizon_invariance_check | Cost doesn't scale linearly with T |
| staged_weights_check | Staged defaults correct |

Results: `runs/self_check/staged_cost_self_check.json`
