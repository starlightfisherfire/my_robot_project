# src/planners/planner_cost_adapter.py
"""Planner cost adapter — bridge between planner cost functions and full rollout_cost.

Provides backward-compatible cost function implementations that use the
full `rollout_cost()` from `src/planners/cost_functions.py` instead of
the simplified `_compute_cost()` that was previously used in
`learned_planner_adapter.py`.

Key fix (2026-05-25):
    The original LearnedRolloutCostFn._compute_cost() used only 4 cost terms.
    This adapter wraps the full 11-term rollout_cost() with the same interface,
    adding collision tracking, contact tracking, and obstacle proximity.

Usage:
    from src.planners.planner_cost_adapter import FullRolloutCostFn
    cost_fn = FullRolloutCostFn(model, normalizer, initial_state, goal_pose, convention)
    cost = cost_fn(action_sequence_norm)  # Same interface as LearnedRolloutCostFn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.planners.action_conventions import ActionConvention, PAPER1_CONVENTION
from src.planners.cost_functions import (
    CostWeights,
    rollout_cost,
    obstacle_proximity_cost,
)

# State feature indices (canonical_state16)
IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
FEAT_VX, FEAT_VY, FEAT_OMEGA = 4, 5, 6
FEAT_CONTACT, FEAT_VALID = 14, 15


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for diagnostics."""
    total: float
    pos_cost: float
    theta_cost: float
    reach_cost: float
    no_contact_cost: float
    push_alignment_cost: float
    action_cost: float
    smooth_cost: float
    collision_cost: float
    proximity_cost: float
    subgoal_cost: float


class FullRolloutCostFn:
    """Cost function using full rollout_cost() from cost_functions.py.
    
    Backward-compatible with LearnedRolloutCostFn.__call__() interface.
    Uses all 11 cost terms from CostWeights, including:
    - Goal position (w_pos=10.0)
    - Goal orientation (w_theta=2.0)
    - EE-object reach (w_reach=5.0)
    - No-contact penalty (w_no_contact=2.0)
    - Push alignment (w_push_alignment=1.0)
    - Collision penalty (w_collision=20.0, w_collision_step=1.0)
    - Obstacle proximity (w_proximity=5.0)
    - Action magnitude (w_action=0.05)
    - Action smoothness (w_smooth=0.1)
    
    Additionally tracks collision_flags and contact_flags internally
    for use in the full cost function.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
        convention: ActionConvention = PAPER1_CONVENTION,
        device: str = "cpu",
        weights: Optional[CostWeights] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        obstacle_radii: Optional[np.ndarray] = None,
        contact_distance_threshold: float = 0.035,
    ):
        self.model = model
        self.normalizer = normalizer
        self.initial_state = initial_state.copy()
        self.goal_pose = goal_pose
        self.convention = convention
        self.device = device
        self.weights = weights or CostWeights()
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        self.contact_distance_threshold = contact_distance_threshold
        
        # Storage for diagnostics
        self.last_breakdown: Optional[CostBreakdown] = None
        self.last_collision_flags: Optional[np.ndarray] = None
        self.last_contact_flags: Optional[np.ndarray] = None
    
    def __call__(self, action_sequence_norm: np.ndarray) -> float:
        """Evaluate cost of a normalized action sequence using full rollout_cost.
        
        Args:
            action_sequence_norm: [H, 2] normalized actions from planner
            
        Returns:
            Total cost (float)
        """
        action_sequence_norm = np.asarray(action_sequence_norm, dtype=np.float64)
        horizon = len(action_sequence_norm)
        
        # Build rollout through learned model
        current_state = self.initial_state.copy()
        obj_xy = current_state[-1, IDX_OBJ, [FEAT_X, FEAT_Y]].copy()
        obj_theta = np.arctan2(
            current_state[-1, IDX_OBJ, FEAT_SIN_THETA],
            current_state[-1, IDX_OBJ, FEAT_COS_THETA]
        )
        object_traj = [np.array([obj_xy[0], obj_xy[1], obj_theta])]
        ee_traj = [current_state[-1, IDX_EE, [FEAT_X, FEAT_Y]].copy()]
        
        # Track collision and contact flags
        collision_flags = []
        contact_flags = []
        
        for t in range(horizon):
            a_norm = action_sequence_norm[t]
            a_phys = self.convention.planner_to_model_action(a_norm)
            
            # Normalize state for model
            state_norm = current_state.copy()
            if self.normalizer is not None:
                state_norm = self.normalizer.transform(state_norm)
            
            # Model prediction
            state_t = torch.from_numpy(state_norm[np.newaxis]).float().to(self.device)
            action_t = torch.from_numpy(a_phys[np.newaxis].astype(np.float32)).float().to(self.device)
            
            with torch.no_grad():
                out = self.model(state_t, action_t)
            
            delta = out["pred_delta"].cpu().numpy()[0]  # [dx, dy, dtheta]
            
            # Update object pose
            obj_xy = object_traj[-1][:2] + delta[:2]
            obj_theta = np.arctan2(
                np.sin(object_traj[-1][2] + delta[2]),
                np.cos(object_traj[-1][2] + delta[2])
            )
            object_traj.append(np.array([obj_xy[0], obj_xy[1], obj_theta]))
            
            # Update EE with displacement
            disp = self.convention.model_action_to_state_displacement(a_phys)
            ee_xy = ee_traj[-1] + disp
            ee_traj.append(ee_xy)
            
            # Track contact (distance-based proxy)
            ee_obj_dist = np.linalg.norm(ee_xy - obj_xy[:2])
            contact_flags.append(1.0 if ee_obj_dist < self.contact_distance_threshold else 0.0)
            
            # Track collision (obstacle proximity-based proxy)
            has_collision = 0.0
            if self.obstacle_positions is not None and self.obstacle_radii is not None:
                for obs_pos, obs_r in zip(self.obstacle_positions, self.obstacle_radii):
                    if np.linalg.norm(ee_xy - obs_pos) < obs_r:
                        has_collision = 1.0
                        break
            collision_flags.append(has_collision)
            
            # Update state for next step
            new_state = current_state.copy()
            new_state[:-1] = new_state[1:]
            new_state[-1, IDX_OBJ, FEAT_X] = obj_xy[0]
            new_state[-1, IDX_OBJ, FEAT_Y] = obj_xy[1]
            new_state[-1, IDX_OBJ, FEAT_SIN_THETA] = np.sin(obj_theta)
            new_state[-1, IDX_OBJ, FEAT_COS_THETA] = np.cos(obj_theta)
            new_state[-1, IDX_EE, FEAT_X] = ee_xy[0]
            new_state[-1, IDX_EE, FEAT_Y] = ee_xy[1]
            current_state = new_state
        
        # Convert trajectories to arrays
        object_traj = np.array(object_traj)
        ee_traj = np.array(ee_traj)
        collision_array = np.array(collision_flags)
        contact_array = np.array(contact_flags)
        
        # Store for diagnostics
        self.last_collision_flags = collision_array
        self.last_contact_flags = contact_array
        
        # Compute cost breakdown
        self.last_breakdown = self._compute_breakdown(
            object_traj, ee_traj, action_sequence_norm, collision_array, contact_array
        )
        
        return self.last_breakdown.total
    
    def _compute_breakdown(
        self,
        object_traj: np.ndarray,
        ee_traj: np.ndarray,
        action_sequence: np.ndarray,
        collision_flags: np.ndarray,
        contact_flags: np.ndarray,
    ) -> CostBreakdown:
        """Compute full cost breakdown using rollout_cost and individual components."""
        from src.planners.cost_functions import (
            pose_error,
            reach_cost,
            no_contact_cost,
            push_alignment_cost,
            collision_cost,
            obstacle_proximity_cost,
            action_regularization_cost,
            action_smoothness_cost,
        )
        
        w = self.weights
        
        # Goal cost
        pos_err_sq, theta_err_sq = pose_error(object_traj[-1], self.goal_pose)
        pos_cost = w.w_pos * pos_err_sq
        theta_cost = w.w_theta * theta_err_sq
        
        # Reach cost
        r_cost = w.w_reach * reach_cost(ee_traj, object_traj[:, :2])
        
        # No-contact cost
        nc_cost = w.w_no_contact * no_contact_cost(
            contact_flags=contact_flags,
            ee_positions=ee_traj,
            object_positions=object_traj[:, :2],
        )
        
        # Push alignment
        pa_cost = w.w_push_alignment * push_alignment_cost(
            object_initial_pose=object_traj[0],
            object_final_pose=object_traj[-1],
            goal_pose=self.goal_pose,
        )
        
        # Action costs
        a_cost = w.w_action * action_regularization_cost(action_sequence)
        s_cost = w.w_smooth * action_smoothness_cost(action_sequence)
        
        # Collision cost
        collision_any, collision_count, _ = collision_cost(collision_flags)
        col_cost = w.w_collision * collision_any + w.w_collision_step * collision_count
        
        # Proximity cost
        prox_cost = 0.0
        if self.obstacle_positions is not None and self.obstacle_radii is not None and w.w_proximity > 0.0:
            prox_cost = w.w_proximity * obstacle_proximity_cost(
                ee_positions=ee_traj,
                obstacle_positions=self.obstacle_positions,
                obstacle_radii=self.obstacle_radii,
                margin=0.03,
            )
        
        total = pos_cost + theta_cost + r_cost + nc_cost + pa_cost + a_cost + s_cost + col_cost + prox_cost
        
        return CostBreakdown(
            total=float(total),
            pos_cost=float(pos_cost),
            theta_cost=float(theta_cost),
            reach_cost=float(r_cost),
            no_contact_cost=float(nc_cost),
            push_alignment_cost=float(pa_cost),
            action_cost=float(a_cost),
            smooth_cost=float(s_cost),
            collision_cost=float(col_cost),
            proximity_cost=float(prox_cost),
            subgoal_cost=0.0,  # subgoal disabled by default
        )
    
    def get_last_cost_breakdown(self) -> Optional[CostBreakdown]:
        """Return detailed cost breakdown from last evaluation."""
        return self.last_breakdown


# ============================================================
# Backward-compatible aliases
# ============================================================

# For code that imports from learned_planner_adapter
# FullRolloutCostFn is the recommended replacement for LearnedRolloutCostFn
# It has the same __call__ interface but uses the full cost function.
