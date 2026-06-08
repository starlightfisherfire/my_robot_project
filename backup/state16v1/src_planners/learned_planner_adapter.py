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
from src.planners.mppi import MPPI, MPPIResult
from src.planners.cost_functions import (
    CostWeights, rollout_cost, pose_error, reach_cost, no_contact_cost,
    push_alignment_cost, action_regularization_cost, action_smoothness_cost,
    collision_cost, obstacle_proximity_cost,
)


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
    """Simplified cost function (legacy). Use FullLearnedRolloutCostFn for fair eval."""
    
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


class FullLearnedRolloutCostFn:
    """Full cost function matching cost_functions.py rollout_cost.
    
    Uses the same 11-term cost as Oracle-MPPI Stage2C:
    - pos_error (w=10)
    - theta_error (w=2)
    - reach (w=5)
    - no_contact (w=2)
    - push_alignment (w=1)
    - action_reg (w=0.05)
    - smoothness (w=0.1)
    - collision (w=20)
    - collision_step (w=1)
    - proximity (w=5)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
        convention: ActionConvention = PAPER1_CONVENTION,
        device: str = "cpu",
        cost_weights: Optional[CostWeights] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        obstacle_radii: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.normalizer = normalizer
        self.initial_state = initial_state.copy()
        self.goal_pose = goal_pose
        self.convention = convention
        self.device = device
        self.weights = cost_weights or CostWeights()
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        self.cost_breakdown = {}
    
    def __call__(self, action_sequence_norm: np.ndarray) -> float:
        """Evaluate cost using full rollout_cost."""
        return float(self.evaluate_batch(action_sequence_norm[np.newaxis])[0])
    
    def evaluate_batch(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate cost for a batch of action sequences (GPU-optimized).
        
        This is the key optimization: instead of calling the model once per
        (sample, timestep), we batch all N samples together at each timestep.
        
        On GPU, this gives ~N× speedup for the model forward pass.
        
        Args:
            samples: [N, horizon, action_dim] normalized action sequences
        
        Returns:
            costs: [N] cost values
        """
        samples = np.asarray(samples, dtype=np.float64)
        N, horizon, action_dim = samples.shape
        
        # ── Pre-process: convert all actions once (vectorized) ──
        actions_norm_flat = samples.reshape(-1, action_dim)
        actions_phys_flat = self.convention.planner_to_model_action(actions_norm_flat)
        actions_phys = actions_phys_flat.reshape(N, horizon, action_dim)  # [N, H, 2]
        # Pre-compute EE displacements (same physical→displacement conversion)
        displacements = self.convention.model_action_to_state_displacement(
            actions_phys_flat
        ).reshape(N, horizon, action_dim)  # [N, H, 2]
        
        # ── Initialize per-sample states ──
        init = self.initial_state.copy()  # [H, N_tokens, D_raw]
        H_len, N_tok, D_raw = init.shape
        
        # Clone initial state for all N samples
        states = np.tile(init[np.newaxis], (N, 1, 1, 1)).astype(np.float32)  # [N, H, N_tok, D]
        
        # Initial object (x, y, theta) and EE (x, y) for all samples
        obj_xy = np.tile(init[-1, IDX_OBJ, [FEAT_X, FEAT_Y]], (N, 1)).copy()  # [N, 2]
        obj_sin = np.full(N, init[-1, IDX_OBJ, FEAT_SIN_THETA])
        obj_cos = np.full(N, init[-1, IDX_OBJ, FEAT_COS_THETA])
        obj_theta = np.arctan2(obj_sin, obj_cos)  # [N]
        ee_xy = np.tile(init[-1, IDX_EE, [FEAT_X, FEAT_Y]], (N, 1)).copy()  # [N, 2]
        
        # Trajectory accumulators: [N, horizon+1, 3] for object, [N, horizon+1, 2] for EE
        obj_traj = np.zeros((N, horizon + 1, 3), dtype=np.float64)
        ee_traj = np.zeros((N, horizon + 1, 2), dtype=np.float64)
        obj_traj[:, 0, 0] = obj_xy[:, 0]
        obj_traj[:, 0, 1] = obj_xy[:, 1]
        obj_traj[:, 0, 2] = obj_theta
        ee_traj[:, 0] = ee_xy
        
        # ── Autoregressive rollout (batched across samples, sequential across time) ──
        for t in range(horizon):
            # Normalize states (batched - reshape handles any leading dims)
            if self.normalizer is not None:
                states_norm = self.normalizer.transform(states)  # [N, H, N_tok, D]
            else:
                states_norm = states
            
            # Batch model forward: [N, H, N_tok, D] + [N, 2] → dict
            state_batch = torch.from_numpy(states_norm).float().to(self.device)
            action_batch = torch.from_numpy(actions_phys[:, t].astype(np.float32)).float().to(self.device)
            
            with torch.no_grad():
                out = self.model(state_batch, action_batch)
            
            deltas = out["pred_delta"].cpu().numpy()  # [N, 3]
            
            # Update object pose for all samples (vectorized)
            obj_xy_new = obj_xy + deltas[:, :2]  # [N, 2]
            obj_theta_new = np.arctan2(
                np.sin(obj_theta + deltas[:, 2]),
                np.cos(obj_theta + deltas[:, 2])
            )
            
            # Update EE for all samples (vectorized)
            ee_xy_new = ee_xy + displacements[:, t]  # [N, 2]
            
            # Store in trajectory
            obj_traj[:, t+1, 0] = obj_xy_new[:, 0]
            obj_traj[:, t+1, 1] = obj_xy_new[:, 1]
            obj_traj[:, t+1, 2] = obj_theta_new
            ee_traj[:, t+1] = ee_xy_new
            
            # Shift history window and insert new frame
            states[:, :-1] = states[:, 1:]  # Shift left
            new_frame = states[:, -1].copy()  # Last frame as template
            new_frame[:, IDX_OBJ, FEAT_X] = obj_xy_new[:, 0]
            new_frame[:, IDX_OBJ, FEAT_Y] = obj_xy_new[:, 1]
            new_frame[:, IDX_OBJ, FEAT_SIN_THETA] = np.sin(obj_theta_new)
            new_frame[:, IDX_OBJ, FEAT_COS_THETA] = np.cos(obj_theta_new)
            new_frame[:, IDX_EE, FEAT_X] = ee_xy_new[:, 0]
            new_frame[:, IDX_EE, FEAT_Y] = ee_xy_new[:, 1]
            states[:, -1] = new_frame
            
            # Advance for next iteration
            obj_xy = obj_xy_new
            obj_theta = obj_theta_new
            ee_xy = ee_xy_new
        
        # ── Compute costs for all N rollouts ──
        costs = np.zeros(N, dtype=np.float64)
        for i in range(N):
            costs[i] = rollout_cost(
                predicted_object_poses=obj_traj[i],
                ee_positions=ee_traj[i],
                action_sequence=samples[i],
                goal_pose=self.goal_pose,
                weights=self.weights,
                obstacle_positions=self.obstacle_positions,
                obstacle_radii=self.obstacle_radii,
            )
        
        # Save breakdown for last sample (for diagnostics)
        obj_final = obj_traj[-1, -1]
        pos_err_sq, theta_err_sq = pose_error(obj_final, self.goal_pose)
        self.cost_breakdown = {
            'pos_cost': self.weights.w_pos * pos_err_sq,
            'theta_cost': self.weights.w_theta * theta_err_sq,
            'reach_cost': self.weights.w_reach * reach_cost(ee_traj[-1], obj_traj[-1, :, :2]),
            'no_contact_cost': self.weights.w_no_contact * no_contact_cost(ee_positions=ee_traj[-1], object_positions=obj_traj[-1, :, :2]),
            'push_alignment': self.weights.w_push_alignment * push_alignment_cost(obj_traj[-1, 0], obj_traj[-1, -1], self.goal_pose),
            'action_cost': self.weights.w_action * action_regularization_cost(samples[-1]),
            'smooth_cost': self.weights.w_smooth * action_smoothness_cost(samples[-1]),
            'total': float(costs[-1]),
        }
        
        return costs


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
        cost_mode: str = "simplified",
        cost_weights = None,
        obstacle_positions: Optional[np.ndarray] = None,
        obstacle_radii: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.normalizer = normalizer
        self.convention = convention
        self.device = device
        self.cost_mode = cost_mode
        self.cost_weights = cost_weights
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        
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
            "cost_mode": cost_mode,
            "convention": convention.describe(),
        }
    
    def plan(
        self,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
    ) -> PlannerResult:
        """Plan actions using CEM with learned rollout."""
        # Create cost function based on mode
        if self.cost_mode == "full":
            cost_fn = FullLearnedRolloutCostFn(
                model=self.model,
                normalizer=self.normalizer,
                initial_state=initial_state,
                goal_pose=goal_pose,
                convention=self.convention,
                device=self.device,
                cost_weights=self.cost_weights,
                obstacle_positions=self.obstacle_positions,
                obstacle_radii=self.obstacle_radii,
            )
        else:
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


class MPPILearnedPlanner:
    """MPPI planner using learned rollout."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,
        convention: ActionConvention = PAPER1_CONVENTION,
        horizon: int = 30,
        num_samples: int = 256,
        num_iterations: int = 5,
        temperature: float = 0.1,
        init_std: float = 0.5,
        speed: float = 0.3,
        execute_steps: int = 5,
        max_mpc_steps: int = 30,
        device: str = "cpu",
        cost_mode: str = "simplified",
        cost_weights = None,
        obstacle_positions: Optional[np.ndarray] = None,
        obstacle_radii: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.normalizer = normalizer
        self.convention = convention
        self.device = device
        self.speed = speed
        self.execute_steps = execute_steps
        self.max_mpc_steps = max_mpc_steps
        self.cost_mode = cost_mode
        self.cost_weights = cost_weights
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        
        self.mppi = MPPI(
            horizon=horizon,
            action_dim=convention.action_dim,
            num_samples=num_samples,
            num_iterations=num_iterations,
            action_low=-1.0,
            action_high=1.0,
            init_std=init_std,
            temperature=temperature,
            smoothing=0.2,
            seed=42,
        )
        
        self.config = {
            "backend": "MPPI_LEARNED_ROLLOUT",
            "horizon": horizon,
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "temperature": temperature,
            "init_std": init_std,
            "speed": speed,
            "execute_steps": execute_steps,
            "max_mpc_steps": max_mpc_steps,
            "cost_mode": cost_mode,
            "convention": convention.describe(),
        }
    
    def plan(
        self,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
    ) -> PlannerResult:
        """Plan actions using MPPI with learned rollout."""
        # Create cost function based on mode
        if self.cost_mode == "full":
            cost_fn = FullLearnedRolloutCostFn(
                model=self.model,
                normalizer=self.normalizer,
                initial_state=initial_state,
                goal_pose=goal_pose,
                convention=self.convention,
                device=self.device,
                cost_weights=self.cost_weights,
                obstacle_positions=self.obstacle_positions,
                obstacle_radii=self.obstacle_radii,
            )
        else:
            cost_fn = LearnedRolloutCostFn(
                model=self.model,
                normalizer=self.normalizer,
                initial_state=initial_state,
                goal_pose=goal_pose,
                convention=self.convention,
                device=self.device,
            )
        
        # Compute zero-action cost
        zero_actions = np.zeros((self.mppi.horizon, self.mppi.action_dim))
        zero_cost = cost_fn(zero_actions)
        
        # Run MPPI
        result: MPPIResult = self.mppi.optimize(cost_fn)
        
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
            planner_backend="MPPI_LEARNED_ROLLOUT",
            planner_config=self.config,
        )
