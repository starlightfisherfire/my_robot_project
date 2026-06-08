#!/usr/bin/env python3
"""Batched learned rollout cost function — evaluates all samples in one model forward pass.

Replaces per-sample sequential model forward with batched inference.
For CEM with 512 samples and horizon=100, this reduces 256,000 model calls
to just 500 (5 iterations × 100 steps), a ~500x speedup on CPU.
"""

import numpy as np
import torch

# State feature indices
IDX_EE, IDX_OBJ = 0, 1
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
FEAT_VX, FEAT_VY, FEAT_OMEGA = 4, 5, 6

from src.planners.cost_functions import CostWeights, rollout_cost
from src.planners.action_conventions import PAPER1_CONVENTION


class BatchedLearnedRolloutCostFn:
    """Cost function that evaluates ALL samples in one batched model forward pass.
    
    Interface compatible with CEMMPC.optimize() — __call__(action_sequence) still 
    accepts single [H, A] array, but internally accumulates into a batch for 
    efficient multi-sample evaluation.
    
    For batch-mode use: call evaluate_batch([B, H, A]) → [B] costs.
    """
    
    def __init__(self, model, normalizer, initial_state, goal_pose, 
                 convention=PAPER1_CONVENTION, device="cpu",
                 weights=None, obstacle_positions=None, obstacle_radii=None):
        self.model = model
        self.normalizer = normalizer
        self.initial_state = initial_state.copy()
        self.goal_pose = goal_pose
        self.convention = convention
        self.device = device
        self.weights = weights or CostWeights()
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        self.cost_breakdown = {}
    
    def __call__(self, action_sequence_norm: np.ndarray) -> float:
        """Single-sequence cost (for backward compat). Delegates to evaluate_batch."""
        seq = np.asarray(action_sequence_norm, dtype=np.float64)
        if seq.ndim == 2:
            costs = self.evaluate_batch(seq[np.newaxis])
            return float(costs[0])
        return float(self.evaluate_batch(seq[None])[0])
    
    def evaluate_batch(self, action_sequences_norm: np.ndarray) -> np.ndarray:
        """Evaluate costs for a batch of action sequences.
        
        Args:
            action_sequences_norm: [B, H, 2] — B independent action sequences
            
        Returns:
            costs: [B] float array
        """
        action_sequences_norm = np.asarray(action_sequences_norm, dtype=np.float64)
        B, H, A = action_sequences_norm.shape
        
        # ── Pre-process: convert ALL actions once (vectorized) ──
        actions_flat = action_sequences_norm.reshape(-1, A)
        actions_phys_flat = self.convention.planner_to_model_action(actions_flat)
        actions_phys = actions_phys_flat.reshape(H, B, A)  # [H, B, 2]
        # Pre-compute EE displacements
        displacements_flat = self.convention.model_action_to_state_displacement(actions_phys_flat)
        displacements = displacements_flat.reshape(H, B, A)  # [H, B, 2]
        
        # ── Initialize per-sample states (batch-all-the-way) ──
        init = self.initial_state.copy()  # [H_hist, N, D]
        states = np.tile(init[np.newaxis], (B, 1, 1, 1)).astype(np.float32)  # [B, H_hist, N, D]
        
        # Initial object and EE for all B samples
        obj_xy0 = init[-1, IDX_OBJ, [FEAT_X, FEAT_Y]]  # [2]
        obj_theta0 = np.arctan2(
            init[-1, IDX_OBJ, FEAT_SIN_THETA],
            init[-1, IDX_OBJ, FEAT_COS_THETA]
        )
        obj_xy = np.tile(obj_xy0, (B, 1))  # [B, 2]
        obj_theta = np.full(B, obj_theta0)  # [B]
        ee_xy = np.tile(init[-1, IDX_EE, [FEAT_X, FEAT_Y]], (B, 1))  # [B, 2]
        
        # Trajectory accumulators: step-major list for compatibility with cost functions
        obj_traj = [np.column_stack([obj_xy, obj_theta])]  # step 0: [B, 3]
        ee_traj = [ee_xy.copy()]  # step 0: [B, 2]
        
        # ── Autoregressive rollout (batched across samples, sequential across time) ──
        for t in range(H):
            # Normalize states (batched — reshape handles any leading dims)
            if self.normalizer is not None:
                states_norm = self.normalizer.transform(states)  # [B, H_hist, N, D]
            else:
                states_norm = states
            
            # Batched model forward: [B, H_hist, N, D] + [B, 2] → dict
            state_batch = torch.from_numpy(states_norm).float().to(self.device)
            action_batch = torch.from_numpy(actions_phys[t].astype(np.float32)).float().to(self.device)
            
            with torch.no_grad():
                out = self.model(state_batch, action_batch)
            
            deltas = out["pred_delta"].cpu().numpy()  # [B, 3]
            
            # Update object pose (vectorized)
            obj_xy_new = obj_xy + deltas[:, :2]  # [B, 2]
            obj_theta_new = np.arctan2(
                np.sin(obj_theta + deltas[:, 2]),
                np.cos(obj_theta + deltas[:, 2])
            )  # [B]
            
            # Update EE (vectorized)
            ee_xy_new = ee_xy + displacements[t]  # [B, 2]
            
            # Store trajectories
            obj_traj.append(np.column_stack([obj_xy_new, obj_theta_new]))  # [B, 3]
            ee_traj.append(ee_xy_new)  # [B, 2]
            
            # Shift history window and insert new frame (vectorized)
            states[:, :-1] = states[:, 1:]  # Shift left for all B
            new_frame = states[:, -1].copy()  # [B, N, D]
            new_frame[:, IDX_OBJ, FEAT_X] = obj_xy_new[:, 0]
            new_frame[:, IDX_OBJ, FEAT_Y] = obj_xy_new[:, 1]
            new_frame[:, IDX_OBJ, FEAT_SIN_THETA] = np.sin(obj_theta_new)
            new_frame[:, IDX_OBJ, FEAT_COS_THETA] = np.cos(obj_theta_new)
            new_frame[:, IDX_EE, FEAT_X] = ee_xy_new[:, 0]
            new_frame[:, IDX_EE, FEAT_Y] = ee_xy_new[:, 1]
            states[:, -1] = new_frame
            
            # Advance
            obj_xy = obj_xy_new
            obj_theta = obj_theta_new
            ee_xy = ee_xy_new
        
        # Transpose trajectories from step-major to sample-major
        obj_traj = np.stack(obj_traj, axis=1)  # [B, H+1, 3]
        ee_traj = np.stack(ee_traj, axis=1)    # [B, H+1, 2]
        
        # Compute costs for each sample
        costs = np.zeros(B, dtype=np.float64)
        for b in range(B):
            costs[b] = rollout_cost(
                predicted_object_poses=obj_traj[b],
                ee_positions=ee_traj[b],
                action_sequence=action_sequences_norm[b],
                goal_pose=self.goal_pose,
                weights=self.weights,
                obstacle_positions=self.obstacle_positions,
                obstacle_radii=self.obstacle_radii,
            )
        
        return costs
