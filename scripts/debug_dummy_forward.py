"""
Smoke test for Paper 1 high-level model.

This script does not train a meaningful model.
It verifies that a dummy structured-state batch can pass through:

dummy state → FlatEncoder → DynamicsHead/SubgoalHead → loss → backward
"""

import torch

from src.models.encoders import FlatEncoder
from src.models.heads import DynamicsHead, SubgoalHead
from src.models.losses import total_high_level_loss


def main():
    torch.manual_seed(42)

    batch_size = 4
    history_len = 6
    num_tokens = 6
    raw_token_dim = 16
    action_dim = 2
    z_dim = 256

    x = torch.randn(batch_size, history_len, num_tokens, raw_token_dim)
    action = torch.randn(batch_size, action_dim)

    target_delta = torch.randn(batch_size, 3)
    target_subgoal = torch.randn(batch_size, 3)

    encoder = FlatEncoder(
        history_len=history_len,
        num_tokens=num_tokens,
        raw_token_dim=raw_token_dim,
        frame_embed_dim=128,
        gru_hidden=z_dim,
    )

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

    print("input:", x.shape)
    print("z:", z.shape)
    print("pred_delta:", pred_delta.shape)
    print("pred_subgoal:", pred_subgoal.shape)
    print("loss:", float(losses["loss"].detach()))
    print("grad_norm:", grad_norm)
    print("backward ok")


if __name__ == "__main__":
    main()