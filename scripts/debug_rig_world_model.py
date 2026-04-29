"""
Smoke test for RIGWorldModel.

This script verifies:

dummy structured state + action
→ RIGWorldModel(model_type=...)
→ pred_delta / pred_subgoal
→ loss
→ backward

It tests three model types:

    flat
    object_centric
    causality_aware

It does not train a meaningful model.
It only checks model interfaces, tensor shapes, loss, and gradients.
"""

from __future__ import annotations

import torch

from src.models.losses import total_high_level_loss
from src.models.rig_world import RIGWorldModel


def make_dummy_batch(
    batch_size: int = 4,
    history_len: int = 6,
    num_tokens: int = 6,
    raw_token_dim: int = 16,
    action_dim: int = 2,
):
    """
    Create dummy structured-state batch.

    x shape:
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

    # Simulate padded obstacle tokens.
    # token 4 and token 5 are invalid obstacles.
    x[:, :, 4:, :] = 0.0
    x[:, :, 4:, 15] = 0.0

    action = torch.randn(batch_size, action_dim)

    target_delta = torch.randn(batch_size, 3)
    target_subgoal = torch.randn(batch_size, 3)

    return x, action, target_delta, target_subgoal


def run_one_model(model_type: str) -> None:
    torch.manual_seed(42)

    batch_size = 4
    z_dim = 256
    slot_dim = 32

    x, action, target_delta, target_subgoal = make_dummy_batch(
        batch_size=batch_size,
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        action_dim=2,
    )

    invalid_mask = x[..., 15] < 0.5
    assert invalid_mask.any(), "Dummy input should contain invalid padding tokens."

    model = RIGWorldModel(
        model_type=model_type,
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        action_dim=2,
        frame_embed_dim=128,
        d_model=128,
        transformer_layers=2,
        nhead=4,
        ffn_dim=256,
        gru_hidden=z_dim,
        head_hidden_dim=256,
        slot_dim=slot_dim,
        dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    return_slots = model_type == "causality_aware"

    out = model(
        x=x,
        action=action,
        return_slots=return_slots,
    )

    assert out["z"].shape == (batch_size, z_dim), out["z"].shape
    assert out["pred_delta"].shape == (batch_size, 3), out["pred_delta"].shape
    assert out["pred_subgoal"].shape == (batch_size, 3), out["pred_subgoal"].shape

    assert torch.isfinite(out["z"]).all().item(), f"{model_type}: z is not finite"
    assert torch.isfinite(out["pred_delta"]).all().item(), (
        f"{model_type}: pred_delta is not finite"
    )
    assert torch.isfinite(out["pred_subgoal"]).all().item(), (
        f"{model_type}: pred_subgoal is not finite"
    )

    if model_type == "causality_aware":
        assert out["z_stable"].shape == (batch_size, slot_dim), out["z_stable"].shape
        assert out["z_dynamics"].shape == (batch_size, slot_dim), out["z_dynamics"].shape
        assert out["z_affordance"].shape == (
            batch_size,
            slot_dim,
        ), out["z_affordance"].shape
        assert out["z_nuisance"].shape == (batch_size, slot_dim), out["z_nuisance"].shape

        for key in ["z_stable", "z_dynamics", "z_affordance", "z_nuisance"]:
            assert torch.isfinite(out[key]).all().item(), (
                f"{model_type}: {key} is not finite"
            )

    losses = total_high_level_loss(
        pred_delta=out["pred_delta"],
        target_delta=target_delta,
        pred_subgoal=out["pred_subgoal"],
        target_subgoal=target_subgoal,
        lambda_dyn=1.0,
        lambda_subgoal=1.0,
    )

    assert torch.isfinite(losses["loss"]).item(), f"{model_type}: loss is not finite"

    optimizer.zero_grad()
    losses["loss"].backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().norm().item()

    assert grad_norm > 0.0, f"{model_type}: no gradients found"

    optimizer.step()

    print(f"[{model_type}] input:", x.shape)
    print(f"[{model_type}] invalid tokens:", int(invalid_mask.sum().item()))
    print(f"[{model_type}] action:", action.shape)
    print(f"[{model_type}] z:", out["z"].shape)
    print(f"[{model_type}] pred_delta:", out["pred_delta"].shape)
    print(f"[{model_type}] pred_subgoal:", out["pred_subgoal"].shape)
    print(f"[{model_type}] loss:", float(losses["loss"].detach()))
    print(f"[{model_type}] grad_norm:", grad_norm)

    if model_type == "causality_aware":
        print(f"[{model_type}] z_stable:", out["z_stable"].shape)
        print(f"[{model_type}] z_dynamics:", out["z_dynamics"].shape)
        print(f"[{model_type}] z_affordance:", out["z_affordance"].shape)
        print(f"[{model_type}] z_nuisance:", out["z_nuisance"].shape)
        print(f"[{model_type}] slot outputs ok")

    print(f"[{model_type}] backward ok")
    print("-" * 80)


def main() -> None:
    for model_type in ["flat", "object_centric", "causality_aware"]:
        run_one_model(model_type)


if __name__ == "__main__":
    main()