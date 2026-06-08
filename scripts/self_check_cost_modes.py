#!/usr/bin/env python3
"""
Self-check for cost modes — synthetic test cases.

Tests each cost mode against carefully constructed scenarios that
should distinguish good cost design from bad.

Run: python scripts/self_check_cost_modes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.planners.cost_modes import (
    rollout_cost_with_mode,
    list_cost_modes,
    push_efficiency_cost,
    object_obstacle_proximity_cost,
    terminal_hold_cost,
    drift_after_best_cost,
)

ALL_MODES = list_cost_modes()

# ──────────────────────────────────────────────────────────
# Helper: build synthetic rollout
# ──────────────────────────────────────────────────────────

def make_rollout(T: int = 20):
    """Create zero-initialized arrays for T timesteps."""
    obj = np.zeros((T, 3))
    ee = np.zeros((T, 2))
    actions = np.zeros((T - 1, 2))
    return obj, ee, actions


# ──────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────

def test_early_vs_late_contact():
    """early_contact: contact at t=2 should cost less than contact at t=15."""
    T = 20
    goal = np.array([1.0, 0.0, 0.0])
    results = {}

    # Early contact: EE reaches object at t=2, pushes to goal
    obj_early, ee_early, actions = make_rollout(T)
    obj_early[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_early[t] = [t * 0.05, 0.0, 0.0]  # moving toward goal
    ee_early[:3] = [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]]  # near object early
    ee_early[3:] = obj_early[3:, :2]  # stay with object
    contact_early = np.zeros(T)
    contact_early[2:] = 1.0

    # Late contact: EE wanders, finds object at t=15
    obj_late, ee_late, _ = make_rollout(T)
    obj_late[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_late[t] = [0.0, 0.0, 0.0]  # object doesn't move until contact
    ee_late[:16] = [[0.5, 0.5]] * 16  # EE far away
    ee_late[15:] = obj_late[15:, :2]  # contact at t=15
    obj_late[15:] = [[(t - 15) * 0.067, 0.0, 0.0] for t in range(15, T)]
    contact_late = np.zeros(T)
    contact_late[15:] = 1.0

    for mode in ALL_MODES:
        cost_early = rollout_cost_with_mode(
            obj_early, ee_early, actions, goal,
            cost_mode=mode, contact_flags=contact_early,
        )
        cost_late = rollout_cost_with_mode(
            obj_late, ee_late, actions, goal,
            cost_mode=mode, contact_flags=contact_late,
        )
        results[mode] = {
            "early": round(cost_early, 4),
            "late": round(cost_late, 4),
            "early_wins": cost_early < cost_late,
        }

    return results


def test_persistent_vs_transient_contact():
    """persistent_contact: continuous contact should cost less than touch-and-go."""
    T = 20
    goal = np.array([1.0, 0.0, 0.0])
    results = {}

    # Persistent: contact from t=5 to end
    obj_p, ee_p, actions = make_rollout(T)
    obj_p[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_p[t] = [t * 0.05, 0.0, 0.0]
    ee_p[5:] = obj_p[5:, :2]
    contact_p = np.zeros(T)
    contact_p[5:] = 1.0

    # Transient: contact t=5-8, then lost, re-contact t=15-18
    obj_t, ee_t, _ = make_rollout(T)
    obj_t[0] = [0.0, 0.0, 0.0]
    for t in range(1, 9):
        obj_t[t] = [t * 0.05, 0.0, 0.0]
    for t in range(9, 16):
        obj_t[t] = obj_t[8]  # object stops
    for t in range(16, T):
        obj_t[t] = [obj_t[15, 0] + (t - 15) * 0.05, 0.0, 0.0]
    ee_t[5:9] = obj_t[5:9, :2]
    ee_t[15:19] = obj_t[15:19, :2]
    ee_t[:5] = [[0.5, 0.5]] * 5
    ee_t[9:15] = [[0.5, 0.5]] * 6
    ee_t[19:] = [[0.5, 0.5]] * max(0, T - 19)
    contact_t = np.zeros(T)
    contact_t[5:9] = 1.0
    contact_t[15:19] = 1.0

    for mode in ALL_MODES:
        cost_p = rollout_cost_with_mode(
            obj_p, ee_p, actions, goal,
            cost_mode=mode, contact_flags=contact_p,
        )
        cost_t = rollout_cost_with_mode(
            obj_t, ee_t, actions, goal,
            cost_mode=mode, contact_flags=contact_t,
        )
        results[mode] = {
            "persistent": round(cost_p, 4),
            "transient": round(cost_t, 4),
            "persistent_wins": cost_p < cost_t,
        }

    return results


def test_push_toward_vs_away():
    """push_efficiency: pushing toward goal should cost less than pushing away."""
    T = 20
    goal = np.array([1.0, 0.0, 0.0])
    results = {}

    # Toward goal
    obj_toward, ee_toward, actions = make_rollout(T)
    obj_toward[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_toward[t] = [t * 0.05, 0.0, 0.0]
    ee_toward[:] = obj_toward[:, :2]
    contact_toward = np.ones(T)

    # Away from goal
    obj_away, ee_away, _ = make_rollout(T)
    obj_away[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_away[t] = [-t * 0.05, 0.0, 0.0]
    ee_away[:] = obj_away[:, :2]
    contact_away = np.ones(T)

    for mode in ALL_MODES:
        cost_toward = rollout_cost_with_mode(
            obj_toward, ee_toward, actions, goal,
            cost_mode=mode, contact_flags=contact_toward,
        )
        cost_away = rollout_cost_with_mode(
            obj_away, ee_away, actions, goal,
            cost_mode=mode, contact_flags=contact_away,
        )
        results[mode] = {
            "toward": round(cost_toward, 4),
            "away": round(cost_away, 4),
            "toward_wins": cost_toward < cost_away,
        }

    return results


def test_object_obstacle_collision():
    """object_obstacle: object near obstacle should cost more than object far from obstacle."""
    T = 10
    # Goal is above obstacle — going around is also toward goal
    goal = np.array([1.0, 0.3, 0.0])
    obstacle_pos = np.array([[0.5, 0.0]])
    obstacle_rad = np.array([0.1])
    results = {}

    # Object passes through obstacle (hits obstacle, ends at wrong y)
    obj_hit, ee_hit, actions = make_rollout(T)
    obj_hit[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_hit[t] = [t * 0.1, 0.0, 0.0]  # goes through obstacle at (0.5, 0)
    ee_hit[:] = obj_hit[:, :2]
    contact_hit = np.ones(T)

    # Object goes around obstacle
    obj_safe, ee_safe, _ = make_rollout(T)
    obj_safe[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj_safe[t] = [t * 0.1, 0.3, 0.0]  # goes above obstacle
    ee_safe[:] = obj_safe[:, :2]
    contact_safe = np.ones(T)

    for mode in ALL_MODES:
        cost_hit = rollout_cost_with_mode(
            obj_hit, ee_hit, actions, goal,
            cost_mode=mode, contact_flags=contact_hit,
            obstacle_positions=obstacle_pos, obstacle_radii=obstacle_rad,
        )
        cost_safe = rollout_cost_with_mode(
            obj_safe, ee_safe, actions, goal,
            cost_mode=mode, contact_flags=contact_safe,
            obstacle_positions=obstacle_pos, obstacle_radii=obstacle_rad,
        )
        results[mode] = {
            "through_obstacle": round(cost_hit, 4),
            "around_obstacle": round(cost_safe, 4),
            "around_wins": cost_safe < cost_hit,
        }

    return results


def test_drift_after_best():
    """drift: object reaches goal then drifts should cost more than staying."""
    T = 20
    goal = np.array([1.0, 0.0, 0.0])
    results = {}

    # Object reaches goal at t=10, stays
    obj_stay, ee_stay, actions = make_rollout(T)
    obj_stay[0] = [0.0, 0.0, 0.0]
    for t in range(1, 11):
        obj_stay[t] = [t * 0.1, 0.0, 0.0]
    for t in range(11, T):
        obj_stay[t] = [1.0, 0.0, 0.0]  # stays at goal
    ee_stay[:] = obj_stay[:, :2]
    contact_stay = np.ones(T)

    # Object reaches goal at t=10, drifts away
    obj_drift, ee_drift, _ = make_rollout(T)
    obj_drift[0] = [0.0, 0.0, 0.0]
    for t in range(1, 11):
        obj_drift[t] = [t * 0.1, 0.0, 0.0]
    for t in range(11, T):
        obj_drift[t] = [1.0 + (t - 10) * 0.05, 0.0, 0.0]  # drifts past goal
    ee_drift[:] = obj_drift[:, :2]
    contact_drift = np.ones(T)

    for mode in ALL_MODES:
        cost_stay = rollout_cost_with_mode(
            obj_stay, ee_stay, actions, goal,
            cost_mode=mode, contact_flags=contact_stay,
        )
        cost_drift = rollout_cost_with_mode(
            obj_drift, ee_drift, actions, goal,
            cost_mode=mode, contact_flags=contact_drift,
        )
        results[mode] = {
            "stay_at_goal": round(cost_stay, 4),
            "drift_past": round(cost_drift, 4),
            "stay_wins": cost_stay < cost_drift,
        }

    return results


def test_smooth_vs_jerky():
    """smooth: smooth actions should cost less than jerky."""
    T = 10
    goal = np.array([0.5, 0.0, 0.0])
    obj = np.zeros((T, 3))
    obj[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj[t] = [t * 0.05, 0.0, 0.0]
    ee = obj[:, :2].copy()
    contact = np.ones(T)
    results = {}

    # Smooth actions
    actions_smooth = np.ones((T - 1, 2)) * 0.1

    # Jerky actions
    actions_jerky = np.random.RandomState(42).randn(T - 1, 2) * 0.5

    for mode in ALL_MODES:
        cost_smooth = rollout_cost_with_mode(
            obj, ee, actions_smooth, goal,
            cost_mode=mode, contact_flags=contact,
        )
        cost_jerky = rollout_cost_with_mode(
            obj, ee, actions_jerky, goal,
            cost_mode=mode, contact_flags=contact,
        )
        results[mode] = {
            "smooth": round(cost_smooth, 4),
            "jerky": round(cost_jerky, 4),
            "smooth_wins": cost_smooth < cost_jerky,
        }

    return results


def test_terminal_hold():
    """terminal_hold: EE near object at end should cost less than EE far away."""
    T = 20
    goal = np.array([1.0, 0.0, 0.0])
    results = {}

    obj, _, actions = make_rollout(T)
    obj[0] = [0.0, 0.0, 0.0]
    for t in range(1, T):
        obj[t] = [t * 0.05, 0.0, 0.0]

    # EE stays near object at end
    ee_near = obj[:, :2].copy()
    contact_near = np.ones(T)

    # EE leaves object at end
    ee_far = obj[:, :2].copy()
    ee_far[T - 5:] = [[2.0, 2.0]] * 5  # EE wanders off
    contact_far = np.ones(T)
    contact_far[T - 5:] = 0.0

    for mode in ALL_MODES:
        cost_near = rollout_cost_with_mode(
            obj, ee_near, actions, goal,
            cost_mode=mode, contact_flags=contact_near,
        )
        cost_far = rollout_cost_with_mode(
            obj, ee_far, actions, goal,
            cost_mode=mode, contact_flags=contact_far,
        )
        results[mode] = {
            "ee_near_end": round(cost_near, 4),
            "ee_far_end": round(cost_far, 4),
            "near_wins": cost_near < cost_far,
        }

    return results


# ──────────────────────────────────────────────────────────
# New term unit tests
# ──────────────────────────────────────────────────────────

def test_new_terms_unit():
    """Direct unit tests for new cost terms."""
    results = {}

    # push_efficiency_cost
    contact = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    obj_toward = np.array([[i * 0.1, 0, 0] for i in range(10)], dtype=float)
    obj_away = np.array([[-i * 0.1, 0, 0] for i in range(10)], dtype=float)
    goal = np.array([1.0, 0.0, 0.0])

    eff_toward = push_efficiency_cost(contact, obj_toward, goal)
    eff_away = push_efficiency_cost(contact, obj_away, goal)
    results["push_efficiency"] = {
        "toward": round(eff_toward, 4),
        "away": round(eff_away, 4),
        "toward_better": eff_toward < eff_away,
    }

    # object_obstacle_proximity_cost
    obj_pos = np.array([[0.5, 0.0], [0.6, 0.0], [0.7, 0.0]])
    obs_pos = np.array([[0.5, 0.0]])
    obs_rad = np.array([0.1])
    obj_safe_pos = np.array([[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]])

    cost_hit = object_obstacle_proximity_cost(obj_pos, obs_pos, obs_rad)
    cost_safe = object_obstacle_proximity_cost(obj_safe_pos, obs_pos, obs_rad)
    results["object_obstacle"] = {
        "hit": round(cost_hit, 4),
        "safe": round(cost_safe, 4),
        "safe_better": cost_safe < cost_hit,
    }

    # terminal_hold_cost
    ee_near = np.array([[0.0, 0.0]] * 8 + [[0.1, 0.0]] * 2)
    obj = np.array([[0.0, 0.0]] * 8 + [[0.1, 0.0]] * 2)
    ee_far = np.array([[0.0, 0.0]] * 8 + [[5.0, 5.0]] * 2)

    cost_near = terminal_hold_cost(ee_near, obj)
    cost_far = terminal_hold_cost(ee_far, obj)
    results["terminal_hold"] = {
        "near": round(cost_near, 4),
        "far": round(cost_far, 4),
        "near_better": cost_near < cost_far,
    }

    # drift_after_best_cost
    obj_stay = np.array([[i * 0.1, 0, 0] for i in range(10)] + [[1.0, 0, 0]] * 5)
    obj_drift = np.array([[i * 0.1, 0, 0] for i in range(10)] + [[1.0 + (i + 1) * 0.1, 0, 0] for i in range(5)])

    cost_stay = drift_after_best_cost(obj_stay, goal)
    cost_drift = drift_after_best_cost(obj_drift, goal)
    results["drift_after_best"] = {
        "stay": round(cost_stay, 4),
        "drift": round(cost_drift, 4),
        "stay_better": cost_stay < cost_drift,
    }

    return results


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Cost Mode Self-Check")
    print("=" * 70)

    all_results = {}

    tests = [
        ("early_vs_late_contact", test_early_vs_late_contact),
        ("persistent_vs_transient", test_persistent_vs_transient_contact),
        ("push_toward_vs_away", test_push_toward_vs_away),
        ("object_obstacle_collision", test_object_obstacle_collision),
        ("drift_after_best", test_drift_after_best),
        ("smooth_vs_jerky", test_smooth_vs_jerky),
        ("terminal_hold", test_terminal_hold),
        ("new_terms_unit", test_new_terms_unit),
    ]

    all_pass = True
    for name, test_fn in tests:
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")
        try:
            result = test_fn()
            all_results[name] = result

            # Check if expected winner wins in staged_full
            if "staged_full" in result:
                sf = result["staged_full"]
                wins = [v for k, v in sf.items() if k.endswith("_wins") or k.endswith("_better")]
                if wins and not all(wins):
                    print(f"  ⚠️  staged_full did NOT win all sub-tests")
                    all_pass = False
                else:
                    print(f"  ✅ staged_full passes all expectations")

            # Print summary
            for mode, vals in result.items():
                if isinstance(vals, dict):
                    print(f"  {mode}: {vals}")
                else:
                    print(f"  {mode} = {vals}")

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            all_results[name] = {"error": str(e)}
            all_pass = False

    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "runs" / "cost_structure_ablation_20260528_162959"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "self_check_cost_modes.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Results saved to {out_path}")

    if all_pass:
        print("\n🎉 ALL SELF-CHECKS PASSED")
    else:
        print("\n⚠️  SOME CHECKS NEED ATTENTION — see details above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
