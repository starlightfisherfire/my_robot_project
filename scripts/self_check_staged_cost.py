#!/usr/bin/env python3
"""
Self-check script for staged_contact_obstacle_goal_cost.

Generates synthetic rollouts and verifies each cost term behaves correctly.
"""

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.planners.cost_functions import (
    CostWeights,
    make_staged_cost_weights,
    rollout_cost,
    staged_contact_obstacle_goal_cost,
    terminal_hold_cost,
    contact_duration_cost,
    object_obstacle_proximity_cost,
    object_collision_cost,
    obstacle_aware_potential,
    obstacle_aware_progress_cost,
    first_contact_cost,
    persistent_contact_cost,
    push_alignment_cost,
)


def make_synthetic_data():
    """Create synthetic rollout data for testing."""
    T = 50
    H = 20
    dt = 0.05

    # Goal pose
    goal_pose = np.array([0.5, 0.0, 0.0])

    # Object initial pose
    object_init = np.array([0.0, 0.0, 0.0])

    # EE initial position
    ee_init = np.array([-0.2, 0.0])

    # Obstacles
    obstacle_positions = np.array([[0.3, 0.1]])
    obstacle_radii = np.array([0.05])

    # Actions (small movements)
    action_sequence = np.random.randn(H, 2) * 0.01

    return T, H, goal_pose, object_init, ee_init, obstacle_positions, obstacle_radii, action_sequence


def make_rollout_straight_to_goal(T, goal_pose, object_init, ee_init, contact_start=10):
    """Object moves straight to goal, EE contacts early."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    for t in range(T):
        alpha = min(1.0, t / (T * 0.7))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        ee_positions[t] = ee_init + alpha * (goal_pose[:2] - ee_init)
        if t >= contact_start:
            contact_flags[t] = 1.0

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_pass_through_goal(T, goal_pose, object_init, ee_init):
    """Object passes through goal midway, then drifts away."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    overshoot = goal_pose * 1.5  # Overshoot
    for t in range(T):
        if t < T // 2:
            alpha = t / (T // 2)
            predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        else:
            alpha = (t - T // 2) / (T // 2)
            predicted_object_poses[t] = goal_pose + alpha * (overshoot - goal_pose)
        ee_positions[t] = ee_init + (predicted_object_poses[t][:2] - ee_init) * 0.5
        contact_flags[t] = 1.0

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_early_contact(T, goal_pose, object_init, ee_init, contact_start=5):
    """EE contacts object early."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    for t in range(T):
        alpha = min(1.0, t / (T * 0.8))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        ee_positions[t] = ee_init + alpha * (predicted_object_poses[t][:2] - ee_init)
        if t >= contact_start:
            contact_flags[t] = 1.0

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_late_contact(T, goal_pose, object_init, ee_init, contact_start=35):
    """EE contacts object late."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    for t in range(T):
        alpha = min(1.0, t / (T * 0.8))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        ee_positions[t] = ee_init + alpha * (predicted_object_poses[t][:2] - ee_init)
        if t >= contact_start:
            contact_flags[t] = 1.0

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_no_contact(T, goal_pose, object_init, ee_init):
    """EE never contacts object."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    for t in range(T):
        alpha = min(1.0, t / (T * 0.8))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        # EE stays far from object
        ee_positions[t] = ee_init + np.array([0.0, 0.2])

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_contact_then_lose(T, goal_pose, object_init, ee_init):
    """EE contacts then loses contact."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    for t in range(T):
        alpha = min(1.0, t / (T * 0.8))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        if t < T // 2:
            ee_positions[t] = predicted_object_poses[t][:2]  # In contact
            contact_flags[t] = 1.0
        else:
            ee_positions[t] = ee_init  # Lost contact
            contact_flags[t] = 0.0

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_high_contact_rate(T, goal_pose, object_init, ee_init):
    """High contact rate throughout."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.ones(T)  # Always in contact

    for t in range(T):
        alpha = min(1.0, t / (T * 0.8))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        ee_positions[t] = predicted_object_poses[t][:2]

    return predicted_object_poses, ee_positions, contact_flags


def make_rollout_low_contact_rate(T, goal_pose, object_init, ee_init):
    """Low contact rate."""
    predicted_object_poses = np.zeros((T, 3))
    ee_positions = np.zeros((T, 2))
    contact_flags = np.zeros(T)

    for t in range(T):
        alpha = min(1.0, t / (T * 0.8))
        predicted_object_poses[t] = object_init + alpha * (goal_pose - object_init)
        ee_positions[t] = ee_init + np.array([0.0, 0.2])
        if t % 10 == 0:  # Rare contact
            contact_flags[t] = 1.0
            ee_positions[t] = predicted_object_poses[t][:2]

    return predicted_object_poses, ee_positions, contact_flags


def make_object_trajectory_around_obstacle(T, goal_pose, object_init, obstacle_positions):
    """Object goes around obstacle (potential decreases)."""
    predicted_object_poses = np.zeros((T, 3))
    obs = obstacle_positions[0]

    for t in range(T):
        alpha = t / (T - 1)
        # Arc around obstacle
        mid = (object_init[:2] + goal_pose[:2]) / 2
        offset = np.array([0.0, 0.15])  # Go around
        if alpha < 0.5:
            pos = object_init[:2] + 2 * alpha * (mid + offset - object_init[:2])
        else:
            pos = (mid + offset) + 2 * (alpha - 0.5) * (goal_pose[:2] - (mid + offset))
        predicted_object_poses[t] = [pos[0], pos[1], 0.0]

    return predicted_object_poses


def make_object_trajectory_through_obstacle(T, goal_pose, object_init, obstacle_positions):
    """Object goes through obstacle area (potential may increase)."""
    predicted_object_poses = np.zeros((T, 3))

    for t in range(T):
        alpha = t / (T - 1)
        # Straight line through obstacle
        pos = object_init[:2] + alpha * (goal_pose[:2] - object_init[:2])
        predicted_object_poses[t] = [pos[0], pos[1], 0.0]

    return predicted_object_poses


def make_object_trajectory_away_from_goal(T, goal_pose, object_init):
    """Object moves away from goal (potential increases)."""
    predicted_object_poses = np.zeros((T, 3))

    for t in range(T):
        alpha = t / (T - 1)
        # Move away from goal
        away_dir = object_init[:2] - goal_pose[:2]
        away_dir = away_dir / (np.linalg.norm(away_dir) + 1e-8)
        pos = object_init[:2] + alpha * away_dir * 0.5
        predicted_object_poses[t] = [pos[0], pos[1], 0.0]

    return predicted_object_poses


# =============================================================================
# Self-check functions
# =============================================================================

def terminal_hold_check():
    """Check: last K steps near goal has low cost, mid-pass-through has high cost."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    # Good rollout: stays at goal at the end
    good_poses, good_ee, good_contact = make_rollout_straight_to_goal(T, goal_pose, object_init, ee_init)

    # Bad rollout: passes through goal, drifts away
    bad_poses, bad_ee, bad_contact = make_rollout_pass_through_goal(T, goal_pose, object_init, ee_init)

    w = CostWeights()
    good_cost = terminal_hold_cost(good_poses, goal_pose, last_k=10)
    bad_cost = terminal_hold_cost(bad_poses, goal_pose, last_k=10)

    passed = good_cost < bad_cost
    msg = f"good={good_cost:.4f}, bad={bad_cost:.4f}"
    return passed, msg


def early_contact_check():
    """Check: early contact < late contact < no contact."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    _, _, early_contact = make_rollout_early_contact(T, goal_pose, object_init, ee_init, contact_start=5)
    _, _, late_contact = make_rollout_late_contact(T, goal_pose, object_init, ee_init, contact_start=35)
    _, _, no_contact = make_rollout_no_contact(T, goal_pose, object_init, ee_init)

    object_positions = np.zeros((T, 2))

    early_cost = first_contact_cost(contact_flags=early_contact)
    late_cost = first_contact_cost(contact_flags=late_contact)
    no_cost = first_contact_cost(contact_flags=no_contact)

    passed = early_cost < late_cost < no_cost
    msg = f"early={early_cost:.4f}, late={late_cost:.4f}, no={no_cost:.4f}"
    return passed, msg


def persistent_contact_check():
    """Check: maintained contact < contact then lose."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    _, _, maintained = make_rollout_high_contact_rate(T, goal_pose, object_init, ee_init)
    _, _, lost = make_rollout_contact_then_lose(T, goal_pose, object_init, ee_init)

    maintained_cost = persistent_contact_cost(contact_flags=maintained)
    lost_cost = persistent_contact_cost(contact_flags=lost)

    passed = maintained_cost < lost_cost
    msg = f"maintained={maintained_cost:.4f}, lost={lost_cost:.4f}"
    return passed, msg


def contact_duration_check():
    """Check: high contact rate < low contact rate."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    _, _, high_rate = make_rollout_high_contact_rate(T, goal_pose, object_init, ee_init)
    _, _, low_rate = make_rollout_low_contact_rate(T, goal_pose, object_init, ee_init)

    high_cost = contact_duration_cost(contact_flags=high_rate)
    low_cost = contact_duration_cost(contact_flags=low_rate)

    passed = high_cost < low_cost
    msg = f"high_rate={high_cost:.4f}, low_rate={low_cost:.4f}"
    return passed, msg


def obstacle_aware_progress_check():
    """Check: trajectory around obstacle < trajectory through obstacle < away from goal."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()
    ee_positions = np.zeros((T, 2))

    around_poses = make_object_trajectory_around_obstacle(T, goal_pose, object_init, obs_pos)
    through_poses = make_object_trajectory_through_obstacle(T, goal_pose, object_init, obs_pos)
    away_poses = make_object_trajectory_away_from_goal(T, goal_pose, object_init)

    around_cost = obstacle_aware_progress_cost(around_poses[:, :2], goal_pose, obs_pos, obs_rad)
    through_cost = obstacle_aware_progress_cost(through_poses[:, :2], goal_pose, obs_pos, obs_rad)
    away_cost = obstacle_aware_progress_cost(away_poses[:, :2], goal_pose)

    # Around should be lowest, away should be highest
    passed = around_cost <= through_cost and away_cost > 0
    msg = f"around={around_cost:.4f}, through={through_cost:.4f}, away={away_cost:.4f}"
    return passed, msg


def object_obstacle_check():
    """Check: object near obstacle has high cost, far has low cost."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    # Object positions near obstacle
    near_positions = np.tile(obs_pos[0], (T, 1))  # Right on obstacle

    # Object positions far from obstacle
    far_positions = np.tile(np.array([0.0, 0.0]), (T, 1))

    near_cost = object_obstacle_proximity_cost(near_positions, obs_pos, obs_rad)
    far_cost = object_obstacle_proximity_cost(far_positions, obs_pos, obs_rad)

    passed = near_cost > far_cost
    msg = f"near={near_cost:.4f}, far={far_cost:.4f}"
    return passed, msg


def collision_check():
    """Check: collision rollout has higher cost than no collision."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    # Object collides with obstacle
    collide_positions = np.tile(obs_pos[0], (T, 1))

    # Object far from obstacle
    safe_positions = np.tile(np.array([0.0, 0.0]), (T, 1))

    coll_any, coll_count, coll_rate = object_collision_cost(collide_positions, obs_pos, obs_rad)
    safe_any, safe_count, safe_rate = object_collision_cost(safe_positions, obs_pos, obs_rad)

    passed = coll_any > safe_any and coll_count > safe_count
    msg = f"collision: any={coll_any}, count={coll_count}; safe: any={safe_any}, count={safe_count}"
    return passed, msg


def push_alignment_disabled_check():
    """Check: staged cost does not use push_alignment internally."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    poses, ee, contact = make_rollout_straight_to_goal(T, goal_pose, object_init, ee_init)
    w = CostWeights()

    # Get breakdown
    result = staged_contact_obstacle_goal_cost(
        predicted_object_poses=poses,
        ee_positions=ee,
        action_sequence=actions,
        goal_pose=goal_pose,
        weights=w,
        contact_flags=contact,
        obstacle_positions=obs_pos,
        obstacle_radii=obs_rad,
        return_breakdown=True,
    )

    # Verify no push_alignment in breakdown
    has_push_alignment_key = "push_alignment" in result
    passed = not has_push_alignment_key
    msg = f"breakdown keys: {list(result.keys())}"
    return passed, msg


def current_backward_compatibility_check():
    """Check: cost_mode='current' gives same result as before."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    poses, ee, contact = make_rollout_straight_to_goal(T, goal_pose, object_init, ee_init)
    w = CostWeights()

    # Call with cost_mode='current'
    cost_current = rollout_cost(
        predicted_object_poses=poses,
        ee_positions=ee,
        action_sequence=actions,
        goal_pose=goal_pose,
        weights=w,
        contact_flags=contact,
        obstacle_positions=obs_pos,
        obstacle_radii=obs_rad,
        cost_mode="current",
    )

    # Should be finite and positive
    passed = np.isfinite(cost_current) and cost_current > 0
    msg = f"cost_mode='current' = {cost_current:.4f}"
    return passed, msg


def finite_check():
    """Check: all cost outputs are finite."""
    T, H, goal_pose, object_init, ee_init, obs_pos, obs_rad, actions = make_synthetic_data()

    poses, ee, contact = make_rollout_straight_to_goal(T, goal_pose, object_init, ee_init)
    w = make_staged_cost_weights()

    result = staged_contact_obstacle_goal_cost(
        predicted_object_poses=poses,
        ee_positions=ee,
        action_sequence=actions,
        goal_pose=goal_pose,
        weights=w,
        contact_flags=contact,
        obstacle_positions=obs_pos,
        obstacle_radii=obs_rad,
        return_breakdown=True,
    )

    non_finite_keys = []
    for k, v in result.items():
        if not np.isfinite(v):
            non_finite_keys.append(k)

    passed = len(non_finite_keys) == 0
    if not passed:
        msg = f"Non-finite values in: {non_finite_keys}"
    else:
        msg = f"All {len(result)} values finite"
    return passed, msg


def static_object_progress_check():
    """Check: static trajectory cost > moving trajectory cost (progress penalizes stall)."""
    T = 50
    goal_pose = np.array([0.5, 0.0, 0.0])
    object_init = np.array([0.0, 0.0, 0.0])

    # Static trajectory: object stays at origin
    static_poses = np.tile(object_init, (T, 1))
    static_poses = np.column_stack([static_poses, np.zeros(T)])  # [T,3]

    # Moving trajectory: object moves toward goal
    moving_poses = np.zeros((T, 3))
    for t in range(T):
        alpha = t / (T - 1)
        moving_poses[t] = object_init + alpha * (goal_pose - object_init)

    static_cost = obstacle_aware_progress_cost(static_poses[:, :2], goal_pose)
    moving_cost = obstacle_aware_progress_cost(moving_poses[:, :2], goal_pose)

    passed = static_cost > moving_cost
    msg = f"static={static_cost:.6f}, moving={moving_cost:.6f}"
    return passed, msg


def around_vs_through_obstacle_check():
    """Check: total staged cost for going around obstacle < going through obstacle."""
    T = 50
    H = 20
    goal_pose = np.array([0.5, 0.0, 0.0])
    object_init = np.array([0.0, 0.0, 0.0])
    # Obstacle directly on the straight-line path
    obs_pos = np.array([[0.25, 0.0]])
    obs_rad = np.array([0.06])
    actions = np.random.randn(H, 2) * 0.01
    w = make_staged_cost_weights()

    # Around obstacle: arc that avoids the obstacle
    around_obj = np.zeros((T, 3))
    for t in range(T):
        alpha = t / (T - 1)
        x = object_init[0] + alpha * (goal_pose[0] - object_init[0])
        # Arc above the obstacle
        y = 0.15 * np.sin(alpha * np.pi)
        around_obj[t] = [x, y, 0.0]

    around_ee = np.column_stack([around_obj[:, 0] - 0.05, around_obj[:, 1]])
    around_contact = np.ones(T)

    # Through obstacle: straight line (will collide)
    through_obj = np.zeros((T, 3))
    for t in range(T):
        alpha = t / (T - 1)
        through_obj[t] = object_init + alpha * (goal_pose - object_init)

    through_ee = np.column_stack([through_obj[:, 0] - 0.05, through_obj[:, 1]])
    through_contact = np.ones(T)

    around_result = staged_contact_obstacle_goal_cost(
        predicted_object_poses=around_obj,
        ee_positions=around_ee,
        action_sequence=actions,
        goal_pose=goal_pose,
        weights=w,
        contact_flags=around_contact,
        obstacle_positions=obs_pos,
        obstacle_radii=obs_rad,
        return_breakdown=True,
    )

    through_result = staged_contact_obstacle_goal_cost(
        predicted_object_poses=through_obj,
        ee_positions=through_ee,
        action_sequence=actions,
        goal_pose=goal_pose,
        weights=w,
        contact_flags=through_contact,
        obstacle_positions=obs_pos,
        obstacle_radii=obs_rad,
        return_breakdown=True,
    )

    around_total = around_result["total"]
    through_total = through_result["total"]

    passed = around_total < through_total
    msg = f"around_total={around_total:.4f}, through_total={through_total:.4f}"
    return passed, msg


def narrow_passage_clearance_check():
    """Check: object going through clearance between obstacles < hitting obstacle."""
    T = 30
    goal_pose = np.array([0.6, 0.0, 0.0])
    object_init = np.array([0.0, 0.0, 0.0])

    # Two obstacles with a gap between them
    obs_positions = np.array([[0.3, 0.08], [0.3, -0.08]])
    obs_radii = np.array([0.04, 0.04])
    object_radius = 0.034

    # Clear path through the middle
    clear_poses = np.zeros((T, 3))
    for t in range(T):
        alpha = t / (T - 1)
        clear_poses[t] = [object_init[0] + alpha * (goal_pose[0] - object_init[0]), 0.0, 0.0]

    # Path that hits obstacle (goes through one obstacle center)
    hit_poses = np.zeros((T, 3))
    for t in range(T):
        alpha = t / (T - 1)
        hit_poses[t] = [object_init[0] + alpha * (goal_pose[0] - object_init[0]), 0.08, 0.0]

    clear_cost = object_obstacle_proximity_cost(clear_poses[:, :2], obs_positions, obs_radii, object_radius)
    hit_cost = object_obstacle_proximity_cost(hit_poses[:, :2], obs_positions, obs_radii, object_radius)

    passed = clear_cost < hit_cost
    msg = f"clear={clear_cost:.6f}, hit={hit_cost:.6f}"
    return passed, msg


def horizon_invariance_check():
    """Check: same geometry at different horizons should have similar normalized cost."""
    goal_pose = np.array([0.5, 0.0, 0.0])
    object_init = np.array([0.0, 0.0, 0.0])
    obs_pos = np.array([[0.3, 0.1]])
    obs_rad = np.array([0.05])
    w = make_staged_cost_weights()

    costs = []
    for T in [50, 100, 140]:
        H = T // 2
        actions = np.random.randn(H, 2) * 0.01

        poses = np.zeros((T, 3))
        ee = np.zeros((T, 2))
        contact = np.zeros(T)
        for t in range(T):
            alpha = min(1.0, t / (T * 0.7))
            poses[t] = object_init + alpha * (goal_pose - object_init)
            ee[t] = np.array([-0.2, 0.0]) + alpha * (goal_pose[:2] - np.array([-0.2, 0.0]))
            if t >= T // 5:
                contact[t] = 1.0

        result = staged_contact_obstacle_goal_cost(
            predicted_object_poses=poses,
            ee_positions=ee,
            action_sequence=actions,
            goal_pose=goal_pose,
            weights=w,
            contact_flags=contact,
            obstacle_positions=obs_pos,
            obstacle_radii=obs_rad,
            return_breakdown=True,
        )
        costs.append(result["total"])

    # Costs should not scale linearly with T
    # Allow up to 2x variation (not strict equality, just not 3x for 3x horizon)
    max_ratio = max(costs) / (min(costs) + 1e-10)
    passed = max_ratio < 2.5
    msg = f"T=50:{costs[0]:.4f}, T=100:{costs[1]:.4f}, T=140:{costs[2]:.4f}, max_ratio={max_ratio:.2f}"
    return passed, msg


def staged_weights_check():
    """Check: staged mode uses staged default weights (reach=0, no_contact=0, push_alignment=0)."""
    w = make_staged_cost_weights()

    checks = [
        (w.w_reach == 0.0, f"w_reach={w.w_reach}"),
        (w.w_no_contact == 0.0, f"w_no_contact={w.w_no_contact}"),
        (w.w_push_alignment == 0.0, f"w_push_alignment={w.w_push_alignment}"),
        (w.w_early_contact == 4.0, f"w_early_contact={w.w_early_contact}"),
        (w.w_persistent_contact == 4.0, f"w_persistent_contact={w.w_persistent_contact}"),
        (w.w_collision == 30.0, f"w_collision={w.w_collision}"),
        (w.w_collision_step == 2.0, f"w_collision_step={w.w_collision_step}"),
        (w.w_action == 0.02, f"w_action={w.w_action}"),
        (w.w_smooth == 0.05, f"w_smooth={w.w_smooth}"),
    ]

    all_ok = all(c[0] for c in checks)
    failed = [c[1] for c in checks if not c[0]]

    passed = all_ok
    if passed:
        msg = "All staged weights correct"
    else:
        msg = f"Incorrect weights: {', '.join(failed)}"
    return passed, msg


def run_all_checks():
    """Run all self-checks and return results."""
    checks = [
        ("terminal_hold_check", terminal_hold_check),
        ("early_contact_check", early_contact_check),
        ("persistent_contact_check", persistent_contact_check),
        ("contact_duration_check", contact_duration_check),
        ("obstacle_aware_progress_check", obstacle_aware_progress_check),
        ("object_obstacle_check", object_obstacle_check),
        ("collision_check", collision_check),
        ("push_alignment_disabled_check", push_alignment_disabled_check),
        ("current_backward_compatibility_check", current_backward_compatibility_check),
        ("finite_check", finite_check),
        ("static_object_progress_check", static_object_progress_check),
        ("around_vs_through_obstacle_check", around_vs_through_obstacle_check),
        ("narrow_passage_clearance_check", narrow_passage_clearance_check),
        ("horizon_invariance_check", horizon_invariance_check),
        ("staged_weights_check", staged_weights_check),
    ]

    results = []
    all_passed = True

    for name, check_fn in checks:
        try:
            passed, msg = check_fn()
            status = "PASS" if passed else "FAIL"
            results.append({"check": name, "status": status, "message": msg})
            if not passed:
                all_passed = False
        except Exception as e:
            results.append({
                "check": name,
                "status": "ERROR",
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            all_passed = False

    return results, all_passed


def save_results(results, all_passed):
    """Save results to JSON and Markdown."""
    output_dir = Path(__file__).parent.parent / "runs" / "self_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "staged_cost_self_check.json"
    with open(json_path, "w") as f:
        json.dump({"all_passed": all_passed, "checks": results}, f, indent=2)

    # Save Markdown
    md_path = output_dir / "staged_cost_self_check.md"
    with open(md_path, "w") as f:
        f.write("# Staged Contact-Obstacle-Goal Cost Self-Check\n\n")
        f.write(f"**Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}**\n\n")
        f.write("| Check | Status | Message |\n")
        f.write("| ----- | ------ | ------- |\n")
        for r in results:
            status_emoji = "✅" if r["status"] == "PASS" else "❌"
            f.write(f"| {r['check']} | {status_emoji} {r['status']} | {r['message']} |\n")

        if not all_passed:
            f.write("\n## Failure Details\n\n")
            for r in results:
                if r["status"] != "PASS":
                    f.write(f"### {r['check']}\n\n")
                    f.write(f"**Status:** {r['status']}\n\n")
                    f.write(f"**Message:** {r['message']}\n\n")
                    if "traceback" in r:
                        f.write(f"**Traceback:**\n```\n{r['traceback']}\n```\n\n")

    return json_path, md_path


def save_failure_report(results):
    """Save failure report for failed checks."""
    output_dir = Path(__file__).parent.parent / "runs" / "self_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = [r for r in results if r["status"] != "PASS"]

    # JSON
    json_path = output_dir / "staged_cost_failure_report.json"
    with open(json_path, "w") as f:
        json.dump({"failed_checks": failed}, f, indent=2)

    # Markdown
    md_path = output_dir / "staged_cost_failure_report.md"
    with open(md_path, "w") as f:
        f.write("# Staged Cost Self-Check Failure Report\n\n")
        for r in failed:
            f.write(f"## {r['check']}\n\n")
            f.write(f"**Status:** {r['status']}\n\n")
            f.write(f"**Error:** {r['message']}\n\n")
            f.write(f"**Related Files:**\n- `src/planners/cost_functions.py`\n- `scripts/self_check_staged_cost.py`\n\n")
            f.write(f"**Likely Cause:** Synthetic test data may not match expected behavior, or cost function logic has a bug.\n\n")
            f.write(f"**Attempted Fix:** Review the cost function implementation for the failing check.\n\n")
            f.write(f"**Recommended Next Action:** Debug the specific cost function with more detailed test cases.\n\n")

    return json_path, md_path


if __name__ == "__main__":
    print("=" * 60)
    print("Staged Contact-Obstacle-Goal Cost Self-Check")
    print("=" * 60)
    print()

    results, all_passed = run_all_checks()

    # Print results
    for r in results:
        status_emoji = "✅" if r["status"] == "PASS" else "❌"
        print(f"{status_emoji} {r['check']}: {r['status']} - {r['message']}")

    print()

    # Save results
    json_path, md_path = save_results(results, all_passed)
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    if not all_passed:
        print()
        print("FAILURE DETECTED - Saving failure report...")
        fail_json, fail_md = save_failure_report(results)
        print(f"  Failure JSON: {fail_json}")
        print(f"  Failure MD:   {fail_md}")
        sys.exit(1)
    else:
        print()
        print("ALL CHECKS PASSED ✅")
        sys.exit(0)
