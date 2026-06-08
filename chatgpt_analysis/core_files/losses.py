# src/models/losses.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _check_same_shape(pred: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"{name}: pred and target must have the same shape, "
            f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )


def _wrap_angle_error(angle_error: torch.Tensor) -> torch.Tensor:
    """
    Wrap angle error to [-pi, pi].
    Useful when the last dimension contains raw theta / delta_theta.
    """
    return torch.atan2(torch.sin(angle_error), torch.cos(angle_error))


def pose_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    angle_index: int = 2,
) -> torch.Tensor:
    """
    MSE loss for pose-like vectors [dx, dy, dtheta].
    The angle dimension is wrapped without in-place modification.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    diff = pred - target

    before = diff[..., :angle_index]
    angle = _wrap_angle_error(diff[..., angle_index : angle_index + 1])
    after = diff[..., angle_index + 1 :]

    diff = torch.cat([before, angle, after], dim=-1)

    sq = diff.pow(2)

    if weights is not None:
        weights = weights.to(device=sq.device, dtype=sq.dtype)
        sq = sq * weights

    return sq.mean()


def dynamics_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return pose_mse_loss(
        pred=pred_delta,
        target=target_delta,
        weights=weights,
        angle_index=2,
    )


def subgoal_loss(
    pred_subgoal: torch.Tensor,
    target_subgoal: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return pose_mse_loss(
        pred=pred_subgoal,
        target=target_subgoal,
        weights=weights,
        angle_index=2,
    )


def total_high_level_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    pred_subgoal: torch.Tensor,
    target_subgoal: torch.Tensor,
    lambda_dyn: float = 1.0,
    lambda_subgoal: float = 1.0,
    pose_weights: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    Total supervised high-level loss for Paper 1 v0.1.

    v0.1:
        loss = lambda_dyn * dynamics_loss + lambda_subgoal * subgoal_loss

    Later v0.2 can add:
        affordance loss
        invariance / causal regularization
    """
    l_dyn = dynamics_loss(pred_delta, target_delta, weights=pose_weights)
    l_sub = subgoal_loss(pred_subgoal, target_subgoal, weights=pose_weights)

    total = lambda_dyn * l_dyn + lambda_subgoal * l_sub

    return {
        "loss": total,
        "loss_dyn": l_dyn.detach(),
        "loss_sub": l_sub.detach(),
    }