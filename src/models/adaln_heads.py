"""
AdaLN-based dynamics heads for Paper 1 world model.

Inspired by LeWM's action conditioning via Adaptive Layer Normalization.
Action is injected as scale/shift modulation on the latent state,
rather than simple concatenation.

Variants:
    AdaLNDynamicsHead:     Single-layer AdaLN (drop-in replacement for DynamicsHead)
    MultiLayerAdaLNDHead:  Multi-layer MLP with per-layer AdaLN modulation
    AdaLNWithGoal:         AdaLN conditioned on both action and goal

Usage:
    from src.models.adaln_heads import AdaLNDynamicsHead

    head = AdaLNDynamicsHead(z_dim=256, action_dim=2, hidden_dim=256)
    pred_delta = head(z, action)  # same interface as DynamicsHead
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNModulator(nn.Module):
    """
    Generates scale (gamma) and shift (beta) from a conditioning signal.

    Output: (1 + gamma) * x + beta   (residual-style modulation)
    """

    def __init__(self, cond_dim: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        # Zero-init so modulation starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta


class AdaLNDynamicsHead(nn.Module):
    """
    Single-layer AdaLN dynamics head.

    Drop-in replacement for DynamicsHead. Same interface:
        forward(z, action) → pred_delta [B, 3]

    Architecture:
        action → AdaLNModulator → (gamma, beta)
        z' = (1 + gamma) * LayerNorm(z) + beta
        pred_delta = Linear(z')
    """

    def __init__(
        self,
        z_dim: int = 256,
        action_dim: int = 2,
        hidden_dim: int = 256,
        action_embed_dim: int = 64,  # unused, kept for interface compat
        use_action_embed: bool = True,  # unused, kept for interface compat
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim

        self.norm = nn.LayerNorm(z_dim)
        self.adaln = AdaLNModulator(action_dim, z_dim, hidden_dim=hidden_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaln(action)
        z_normed = self.norm(z)
        z_modulated = (1 + gamma) * z_normed + beta
        return self.pred_head(z_modulated)


class MultiLayerAdaLNDHead(nn.Module):
    """
    Multi-layer MLP with per-layer AdaLN modulation from action.

    Each hidden layer is preceded by AdaLN modulation:
        z_i' = (1 + gamma_i) * LayerNorm_i(z_i) + beta_i
        z_{i+1} = SiLU(Linear_i(z_i'))

    The action generates a separate (gamma, beta) pair for each layer.
    """

    def __init__(
        self,
        z_dim: int = 256,
        action_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.n_layers = n_layers

        # Per-layer norms and AdaLN modulators
        self.norms = nn.ModuleList([
            nn.LayerNorm(z_dim if i == 0 else hidden_dim)
            for i in range(n_layers)
        ])
        self.adalns = nn.ModuleList([
            AdaLNModulator(action_dim, z_dim if i == 0 else hidden_dim, hidden_dim=hidden_dim)
            for i in range(n_layers)
        ])

        # Per-layer linear transforms
        self.linears = nn.ModuleList([
            nn.Linear(z_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])

        # Final prediction head
        self.pred_head = nn.Linear(hidden_dim, 3)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = z
        for i in range(self.n_layers):
            gamma, beta = self.adalns[i](action)
            x_normed = self.norms[i](x)
            x_modulated = (1 + gamma) * x_normed + beta
            x = F.silu(self.linears[i](x_modulated))

        return self.pred_head(x)


class AdaLNWithGoalDHead(nn.Module):
    """
    AdaLN dynamics head conditioned on both action AND goal.

    Two separate modulators:
        - action → modulates state features (what to do)
        - goal   → modulates state features (where to go)

    Combined: z' = (1 + γ_a + γ_g) * LayerNorm(z) + β_a + β_g
    """

    def __init__(
        self,
        z_dim: int = 256,
        action_dim: int = 2,
        goal_dim: int = 3,  # [x, y, theta]
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self.norm = nn.LayerNorm(z_dim)
        self.action_adaln = AdaLNModulator(action_dim, z_dim, hidden_dim=hidden_dim)
        self.goal_adaln = AdaLNModulator(goal_dim, z_dim, hidden_dim=hidden_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma_a, beta_a = self.action_adaln(action)

        gamma_g = torch.zeros_like(gamma_a)
        beta_g = torch.zeros_like(beta_a)
        if goal is not None:
            gamma_g, beta_g = self.goal_adaln(goal)

        z_normed = self.norm(z)
        z_modulated = (1 + gamma_a + gamma_g) * z_normed + beta_a + beta_g
        return self.pred_head(z_modulated)


# ──────────────────────────────────────────────────────────
# Registry for easy switching
# ──────────────────────────────────────────────────────────

ADALN_HEAD_REGISTRY = {
    "single": AdaLNDynamicsHead,
    "multi": MultiLayerAdaLNDHead,
    "goal": AdaLNWithGoalDHead,
}


def build_adaln_head(
    variant: str = "single",
    z_dim: int = 256,
    action_dim: int = 2,
    hidden_dim: int = 256,
    n_layers: int = 3,
    goal_dim: int = 3,
) -> nn.Module:
    """Build an AdaLN dynamics head by variant name."""
    if variant == "single":
        return AdaLNDynamicsHead(z_dim=z_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    elif variant == "multi":
        return MultiLayerAdaLNDHead(z_dim=z_dim, action_dim=action_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    elif variant == "goal":
        return AdaLNWithGoalDHead(z_dim=z_dim, action_dim=action_dim, goal_dim=goal_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown AdaLN variant: {variant}. Available: {list(ADALN_HEAD_REGISTRY.keys())}")
