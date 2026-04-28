"""
Smoke test for Paper 1 encoder variants.

This script verifies:

dummy structured state
→ FlatEncoder / ObjectCentricEncoder
→ DynamicsHead + SubgoalHead
→ loss
→ backward

It does not use real MuJoCo or real robot data.
"""

from __future__ import annotations

import torch

from src.models.encoders import FlatEncoder, ObjectCentricEncoder
from src.models.heads import DynamicsHead, SubgoalHead
from src.models.losses import total_high_level_loss


def make_dummy_states(
    batch_size: int = 4,
    history_len: int = 6,
    num_tokens: int = 6,
    raw_token_dim: int = 16,
) -> torch.Tensor:
    """
    Create dummy structured state tensor.

    Shape:
        [B, H, N, D_raw]

    Token layout:
        0 = ee
        1 = object
        2 = goal
        3 = obstacle_1
        4 = obstacle_2
        5 = obstacle_3

    In this dummy test:
        obstacle_2 and obstacle_3 are invalid padding tokens.
    """
    x = torch.randn(batch_size, history_len, num_tokens, raw_token_dim)

    # Make sin/cos fields physically plausible.
    theta = torch.empty(batch_size, history_len, num_tokens).uniform_(
        -torch.pi, torch.pi
    )
    x[..., 2] = torch.sin(theta)
    x[..., 3] = torch.cos(theta)

    # shape one-hot: default shape_T.
    x[..., 9] = 1.0
    x[..., 10] = 0.0
    x[..., 11] = 0.0

    # contact_flag: binary.
    x[..., 14] = torch.randint(
        low=0,
        high=2,
        size=(batch_size, history_len, num_tokens),
    ).float()

    # valid_flag = 1 for all tokens first.
    x[..., 15] = 1.0

    # Simulate padded obstacle tokens:
    # token 4 and token 5 are invalid obstacles.
    x[:, :, 4:, :] = 0.0
    x[:, :, 4:, 15] = 0.0

    return x


def run_one_encoder(name: str, encoder: torch.nn.Module) -> None:
    torch.manual_seed(42)

    batch_size = 4
    history_len = 6
    num_tokens = 6
    raw_token_dim = 16
    action_dim = 2
    z_dim = 256

    x = make_dummy_states(
        batch_size=batch_size,
        history_len=history_len,
        num_tokens=num_tokens,
        raw_token_dim=raw_token_dim,
    )

    action = torch.randn(batch_size, action_dim)
    target_delta = torch.randn(batch_size, 3)
    target_subgoal = torch.randn(batch_size, 3)

    dyn_head = DynamicsHead(z_dim=z_dim, action_dim=action_dim)
    sub_head = SubgoalHead(z_dim=z_dim)

    params = (
        list(encoder.parameters())
        + list(dyn_head.parameters())
        + list(sub_head.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=3e-4)

    z = encoder(x)
    pred_delta = dyn_head(z, action)
    pred_subgoal = sub_head(z)

    assert z.shape == (batch_size, z_dim), z.shape
    assert pred_delta.shape == (batch_size, 3), pred_delta.shape
    assert pred_subgoal.shape == (batch_size, 3), pred_subgoal.shape

    losses = total_high_level_loss(
        pred_delta=pred_delta,
        target_delta=target_delta,
        pred_subgoal=pred_subgoal,
        target_subgoal=target_subgoal,
        lambda_dyn=1.0,
        lambda_subgoal=1.0,
    )

    assert torch.isfinite(losses["loss"]).item(), "Loss is not finite"

    optimizer.zero_grad()
    losses["loss"].backward()

    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += p.grad.detach().norm().item()

    assert grad_norm > 0.0, "No gradients found"

    optimizer.step()

    invalid_mask = x[..., 15] < 0.5
    assert invalid_mask.any(), "Dummy input should contain invalid padding tokens."

    print(f"[{name}] input:", x.shape)
    print(f"[{name}] invalid tokens:", int(invalid_mask.sum().item()))
    print(f"[{name}] z:", z.shape)
    print(f"[{name}] pred_delta:", pred_delta.shape)
    print(f"[{name}] pred_subgoal:", pred_subgoal.shape)
    print(f"[{name}] loss:", float(losses["loss"].detach()))
    print(f"[{name}] grad_norm:", grad_norm)
    print(f"[{name}] backward ok")
    print("-" * 80)


def main() -> None:
    torch.manual_seed(42)

    flat = FlatEncoder(
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        frame_embed_dim=128,
        gru_hidden=256,
    )

    obj = ObjectCentricEncoder(
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        d_model=128,
        transformer_layers=2,
        nhead=4,
        ffn_dim=256,
        gru_hidden=256,
    )

    run_one_encoder("flat", flat)
    run_one_encoder("object_centric", obj)


if __name__ == "__main__":
    main()