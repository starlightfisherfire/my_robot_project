# src/planners/learned_planner_adapter.py
"""Learned rollout planner adapter for Paper1.

Provides CEM and MPPI adapters that use learned rollout models
for action planning in MuJoCo push manipulation.

Key fix (2026-05-24): Correct action convention handling.
- Planner samples in normalized space [-1, 1]
- Model receives physical velocity [m/s]
- Env receives normalized action [-1, 1]
- EE update uses displacement = a_phys * control_dt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from src.planners.action_conventions import ActionConvention, PAPER1_CONVENTION
from src.planners.cem_mpc import CEMMPC, CEMResult


# State feature indices (canonical_state16)
IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
FEAT_VX, FEAT_VY, FEAT_OMEGA = 4, 5, 6


@dataclass
class PlannerResult:
    """Unified planner output."""
    first_action_norm: np.ndarray      # Normalized action for env.step()
    first_action_phys: np.ndarray      # Physical velocity for model
    action_sequence_norm: np.ndarray   # Full sequence in normalized space
    planned_cost: float
    zero_cost: float
    cost_improvement: float
    planner_backend: str
    planner_config: dict


class LearnedRolloutCostFn:
    """Cost function using learned rollout for CEM/MPPI planning.
    
    This correctly handles action convention:
    - Receives normalized actions from planner
    - Converts to physical velocity for model
    - Updates EE with displacement = a_phys * control_dt
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
        convention: ActionConvention = PAPER1_CONVENTION,
        device: str = "cpu",
        cost_weights: Optional[dict] = None,
    ):
        self.model = model
        self.normalizer = normalizer
        self.initial_state = initial_state.copy()
        self.goal_pose = goal_pose
        self.convention = convention
        self.device = device
        
        # Default cost weights
        self.cost_weights = cost_weights or {
            "pos_weight": 1.0,
            "theta_weight": 0.1,
            "action_weight": 0.01,
            "contact_bonus": 0.1,
        }
    
    def __call__(self, action_sequence_norm: np.ndarray) -> float:
        """Evaluate cost of a normalized action sequence."""
        action_sequence_norm = np.asarray(action_sequence_norm, dtype=np.float64)
        horizon = len(action_sequence_norm)
        
        # Build rollout
        current_state = self.initial_state.copy()
        obj_xy = current_state[-1, IDX_OBJ, [FEAT_X, FEAT_Y]].copy()
        obj_theta = np.arctan2(
            current_state[-1, IDX_OBJ, FEAT_SIN_THETA],
            current_state[-1, IDX_OBJ, FEAT_COS_THETA]
        )
        object_traj = [np.array([obj_xy[0], obj_xy[1], obj_theta])]
        ee_traj = [current_state[-1, IDX_EE, [FEAT_X, FEAT_Y]].copy()]
        
        for t in range(horizon):
            # Get normalized action
            a_norm = action_sequence_norm[t]
            
            # Convert to physical velocity for model
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
            
            # Update EE with displacement = a_phys * control_dt
            disp = self.convention.model_action_to_state_displacement(a_phys)
            ee_xy = ee_traj[-1] + disp
            ee_traj.append(ee_xy)
            
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
        
        object_traj = np.array(object_traj)
        ee_traj = np.array(ee_traj)
        
        return self._compute_cost(object_traj, ee_traj, action_sequence_norm)
    
    def _compute_cost(
        self,
        object_traj: np.ndarray,
        ee_traj: np.ndarray,
        action_sequence: np.ndarray,
    ) -> float:
        """Compute cost from trajectories."""
        # Object-goal distance cost
        obj_pos = object_traj[1:, :2]  # Skip initial
        goal_xy = self.goal_pose[:2]
        pos_dist = np.linalg.norm(obj_pos - goal_xy, axis=1)
        pos_cost = pos_dist.mean() * self.cost_weights["pos_weight"]
        
        # Theta cost
        obj_theta = object_traj[1:, 2]
        goal_theta = self.goal_pose[2]
        theta_err = np.abs(np.arctan2(
            np.sin(obj_theta - goal_theta),
            np.cos(obj_theta - goal_theta)
        ))
        theta_cost = theta_err.mean() * self.cost_weights["theta_weight"]
        
        # Action cost
        action_cost = np.linalg.norm(action_sequence, axis=1).mean() * self.cost_weights["action_weight"]
        
        # Contact bonus (negative cost)
        contact_bonus = 0.0
        if len(ee_traj) > 1:
            ee_pos = ee_traj[1:]
            for i in range(len(ee_pos)):
                if pos_dist[i] < 0.05:  # Close to goal
                    contact_bonus += self.cost_weights["contact_bonus"]
        
        total_cost = pos_cost + theta_cost + action_cost - contact_bonus
        return float(total_cost)


class CEMLearnedPlanner:
    """CEM planner using learned rollout."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,
        convention: ActionConvention = PAPER1_CONVENTION,
        horizon: int = 20,
        num_samples: int = 512,
        num_elites: int = 64,
        num_iterations: int = 5,
        init_std: float = 0.3,
        device: str = "cpu",
    ):
        self.model = model
        self.normalizer = normalizer
        self.convention = convention
        self.device = device
        
        self.cem = CEMMPC(
            horizon=horizon,
            action_dim=convention.action_dim,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations,
            action_low=-1.0,
            action_high=1.0,
            init_std=init_std,
            smoothing=0.2,
            seed=42,
        )
        
        self.config = {
            "backend": "CEM_BEST_LEARNED_ROLLOUT",
            "horizon": horizon,
            "num_samples": num_samples,
            "num_elites": num_elites,
            "num_iterations": num_iterations,
            "init_std": init_std,
            "convention": convention.describe(),
        }
    
    def plan(
        self,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
    ) -> PlannerResult:
        """Plan actions using CEM with learned rollout."""
        # Create cost function
        cost_fn = LearnedRolloutCostFn(
            model=self.model,
            normalizer=self.normalizer,
            initial_state=initial_state,
            goal_pose=goal_pose,
            convention=self.convention,
            device=self.device,
        )
        
        # Compute zero-action cost
        zero_actions = np.zeros((self.cem.horizon, self.cem.action_dim))
        zero_cost = cost_fn(zero_actions)
        
        # Run CEM
        result: CEMResult = self.cem.optimize(cost_fn)
        
        # Get first action
        first_action_norm = result.action_sequence[0].copy()
        first_action_phys = self.convention.planner_to_model_action(first_action_norm)
        
        return PlannerResult(
            first_action_norm=first_action_norm,
            first_action_phys=first_action_phys,
            action_sequence_norm=result.action_sequence,
            planned_cost=result.best_cost,
            zero_cost=zero_cost,
            cost_improvement=zero_cost - result.best_cost,
            planner_backend="CEM_BEST_LEARNED_ROLLOUT",
            planner_config=self.config,
        )
