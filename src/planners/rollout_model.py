# src/planners/rollout_model.py
"""
Learned rollout model for Paper 1.

Wraps a trained RIGWorldModel (encoder + dynamics_head) to provide
multi-step rollout predictions for CEM-MPC planning.

Key design:
    - LearnedRolloutModel wraps a trained model checkpoint
    - forward_step: single-step prediction (state + action → delta_pose)
    - rollout_sequence: multi-step rollout (initial_state + action_seq → trajectory)
    - State update is explicit object-pose based (not full token update)
    - Compatible with CEMMPC cost_fn interface

Limitations (v0.1):
    - State update only modifies manipulated_object token (index 1)
    - EE position is updated by action accumulation
    - History window is updated by shifting and appending
    - Obstacle tokens are not updated (static obstacles)
    - No learned dynamics for EE or obstacle movement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# Token indices in canonical_state16 [H, N, D]
IDX_EE = 0
IDX_OBJ = 1
IDX_GOAL = 2

# Feature indices in D=16
FEAT_X = 0
FEAT_Y = 1
FEAT_SIN_THETA = 2
FEAT_COS_THETA = 3
FEAT_VX = 4
FEAT_VY = 5
FEAT_OMEGA = 6
FEAT_VALID = 15


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass
class LearnedRolloutStepResult:
    """Result of a single learned rollout step."""
    delta_pose: np.ndarray       # [3] = [dx, dy, dtheta]
    pred_object_pose: np.ndarray # [3] = [x, y, theta]


@dataclass
class LearnedRolloutResult:
    """Result of a multi-step learned rollout."""
    object_traj: np.ndarray  # [horizon+1, 3] object poses (includes initial)
    delta_traj: np.ndarray   # [horizon, 3] predicted deltas
    ee_traj: np.ndarray      # [horizon+1, 2] EE positions (includes initial)


class LearnedRolloutModel:
    """
    Learned rollout model wrapping a trained RIGWorldModel.

    Uses the encoder + dynamics_head to predict object pose changes,
    then explicitly updates the state for multi-step rollouts.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        normalizer: object | None = None,
    ):
        """
        Args:
            model: Trained RIGWorldModel (or compatible with encoder + dynamics_head).
            device: Torch device for inference.
            normalizer: Optional StateNormalizer for input normalization.
        """
        self.model = model
        self.device = torch.device(device)
        self.normalizer = normalizer

        self.model.to(self.device)
        self.model.eval()

        # Cache model dimensions
        self.z_dim = model.z_dim
        self.action_dim = model.action_dim
        self.history_len = model.history_len
        self.num_tokens = model.num_tokens
        self.raw_token_dim = model.raw_token_dim

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state if normalizer is provided."""
        if self.normalizer is not None:
            return self.normalizer.transform(state)
        return state

    @torch.no_grad()
    def forward_step(
        self,
        state_tokens: np.ndarray,
        action: np.ndarray,
    ) -> LearnedRolloutStepResult:
        """
        Single-step learned prediction.

        Args:
            state_tokens: [H, N, D] or [B, H, N, D] current state history.
            action: [2] or [B, 2] action.

        Returns:
            LearnedRolloutStepResult with delta_pose and pred_object_pose.
        """
        squeeze = False
        if state_tokens.ndim == 3:
            state_tokens = state_tokens[np.newaxis]  # [1, H, N, D]
            action = action[np.newaxis]  # [1, 2]
            squeeze = True

        # Normalize
        state_norm = self._normalize_state(state_tokens)

        # To tensors
        state_t = torch.from_numpy(state_norm).float().to(self.device)
        action_t = torch.from_numpy(action).float().to(self.device)

        # Forward
        out = self.model(state_t, action_t)

        delta_pose = out["pred_delta"].cpu().numpy()  # [B, 3]

        # Current object pose from last frame, token index 1
        current_obj = state_tokens[:, -1, IDX_OBJ, :]  # [B, D]
        current_xy = current_obj[:, [FEAT_X, FEAT_Y]]  # [B, 2]
        current_theta = np.arctan2(
            current_obj[:, FEAT_SIN_THETA],
            current_obj[:, FEAT_COS_THETA],
        )  # [B]

        # Predicted next pose
        pred_xy = current_xy + delta_pose[:, :2]
        pred_theta = _wrap_angle(current_theta + delta_pose[:, 2])
        pred_object_pose = np.stack([
            pred_xy[:, 0], pred_xy[:, 1], pred_theta
        ], axis=-1)  # [B, 3]

        if squeeze:
            delta_pose = delta_pose[0]
            pred_object_pose = pred_object_pose[0]

        return LearnedRolloutStepResult(
            delta_pose=delta_pose,
            pred_object_pose=pred_object_pose,
        )

    @torch.no_grad()
    def rollout_sequence(
        self,
        initial_state: np.ndarray,
        action_seq: np.ndarray,
    ) -> LearnedRolloutResult:
        """
        Multi-step rollout using learned dynamics.

        Args:
            initial_state: [H, N, D] or [B, H, N, D] initial state history.
            action_seq: [horizon, action_dim] or [B, horizon, action_dim]

        Returns:
            LearnedRolloutResult with object_traj, delta_traj, ee_traj.
        """
        squeeze = False
        if initial_state.ndim == 3:
            initial_state = initial_state[np.newaxis]  # [1, H, N, D]
            action_seq = action_seq[np.newaxis]  # [1, horizon, action_dim]
            squeeze = True

        B = initial_state.shape[0]
        horizon = action_seq.shape[1]

        # Initialize trajectories
        # Initial object pose from last frame
        init_obj = initial_state[:, -1, IDX_OBJ, :]  # [B, D]
        init_xy = init_obj[:, [FEAT_X, FEAT_Y]]  # [B, 2]
        init_theta = np.arctan2(
            init_obj[:, FEAT_SIN_THETA],
            init_obj[:, FEAT_COS_THETA],
        )  # [B]

        # Initial EE position
        init_ee = initial_state[:, -1, IDX_EE, :]  # [B, D]
        init_ee_xy = init_ee[:, [FEAT_X, FEAT_Y]]  # [B, 2]

        object_traj = np.zeros((B, horizon + 1, 3), dtype=np.float64)
        delta_traj = np.zeros((B, horizon, 3), dtype=np.float64)
        ee_traj = np.zeros((B, horizon + 1, 2), dtype=np.float64)

        object_traj[:, 0] = np.stack([init_xy[:, 0], init_xy[:, 1], init_theta], axis=-1)
        ee_traj[:, 0] = init_ee_xy

        # Current state (will be updated each step)
        current_state = initial_state.copy()  # [B, H, N, D]

        for t in range(horizon):
            action_t = action_seq[:, t]  # [B, 2]

            # Predict delta
            step_result = self.forward_step(current_state, action_t)
            delta_traj[:, t] = step_result.delta_pose
            object_traj[:, t + 1] = step_result.pred_object_pose

            # Update EE position: accumulate action
            ee_traj[:, t + 1] = ee_traj[:, t] + action_t

            # Update state for next step
            current_state = self._update_state(
                current_state,
                step_result.pred_object_pose,
                ee_traj[:, t + 1],
            )

        if squeeze:
            object_traj = object_traj[0]
            delta_traj = delta_traj[0]
            ee_traj = ee_traj[0]

        return LearnedRolloutResult(
            object_traj=object_traj,
            delta_traj=delta_traj,
            ee_traj=ee_traj,
        )

    def _update_state(
        self,
        current_state: np.ndarray,
        new_object_pose: np.ndarray,
        new_ee_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Update state tokens after a rollout step.

        Updates:
            - manipulated_object token (index 1): x, y, sin(theta), cos(theta)
            - end_effector token (index 0): x, y
            - Shifts history window: drop oldest frame, append updated frame

        Args:
            current_state: [B, H, N, D]
            new_object_pose: [B, 3] = [x, y, theta]
            new_ee_pos: [B, 2] = [x, y]

        Returns:
            updated_state: [B, H, N, D]
        """
        B, H, N, D = current_state.shape
        updated = current_state.copy()

        # Shift history: drop oldest, duplicate newest
        updated[:, :-1] = updated[:, 1:]
        # Now the newest frame is at index H-1 (duplicate of what was H-2)
        # We'll overwrite it with updated values

        newest = updated[:, -1]  # [B, N, D]

        # Update EE token
        newest[:, IDX_EE, FEAT_X] = new_ee_pos[:, 0]
        newest[:, IDX_EE, FEAT_Y] = new_ee_pos[:, 1]

        # Update object token
        newest[:, IDX_OBJ, FEAT_X] = new_object_pose[:, 0]
        newest[:, IDX_OBJ, FEAT_Y] = new_object_pose[:, 1]
        newest[:, IDX_OBJ, FEAT_SIN_THETA] = np.sin(new_object_pose[:, 2])
        newest[:, IDX_OBJ, FEAT_COS_THETA] = np.cos(new_object_pose[:, 2])

        # Clear velocities (we don't predict them)
        newest[:, IDX_OBJ, FEAT_VX] = 0.0
        newest[:, IDX_OBJ, FEAT_VY] = 0.0
        newest[:, IDX_OBJ, FEAT_OMEGA] = 0.0

        updated[:, -1] = newest

        return updated

    def make_cost_fn_for_cem(
        self,
        initial_state: np.ndarray,
        goal_pose: np.ndarray,
        weights: object | None = None,
    ) -> callable:
        """
        Create a cost function compatible with CEMMPC.optimize().

        Args:
            initial_state: [H, N, D] current state history.
            goal_pose: [3] target object pose.
            weights: CostWeights or None for defaults.

        Returns:
            cost_fn(action_sequence) → float
        """
        from src.planners.cost_functions import CostWeights, rollout_cost

        if weights is None:
            weights = CostWeights()

        initial_state = np.asarray(initial_state, dtype=np.float64)
        goal_pose = np.asarray(goal_pose, dtype=np.float64)

        def cost_fn(action_sequence: np.ndarray) -> float:
            action_sequence = np.asarray(action_sequence, dtype=np.float64)

            result = self.rollout_sequence(initial_state, action_sequence)

            return rollout_cost(
                predicted_object_poses=result.object_traj,
                ee_positions=result.ee_traj,
                action_sequence=action_sequence,
                goal_pose=goal_pose,
                weights=weights,
            )

        return cost_fn


def load_learned_rollout_model(
    checkpoint_path: str,
    model_type: str = "flat",
    device: str = "cpu",
    normalizer_path: str | None = None,
) -> LearnedRolloutModel:
    """
    Load a trained RIGWorldModel from checkpoint and wrap as LearnedRolloutModel.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        model_type: One of "flat", "object_centric", "causality_aware".
        device: Torch device.
        normalizer_path: Optional path to normalizer JSON.

    Returns:
        LearnedRolloutModel ready for rollout.
    """
    import torch
    from src.models.rig_world import RIGWorldModel
    from src.data.state_normalizer import StateNormalizer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer model config from checkpoint or use defaults
    model = RIGWorldModel(
        model_type=model_type,
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        action_dim=2,
        gru_hidden=256,
        head_hidden_dim=256,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalizer = None
    if normalizer_path is not None:
        normalizer = StateNormalizer.load(normalizer_path)

    return LearnedRolloutModel(
        model=model,
        device=device,
        normalizer=normalizer,
    )
