"""
Smoke test for Paper 1 encoder variants.

This script verifies:

dummy structured state
→ FlatEncoder / ObjectCentricEncoder / CausalityAwareEncoder
→ DynamicsHead + SubgoalHead
→ loss
→ backward

It does not train a meaningful model.
It only checks that the model interfaces, tensor shapes, loss, and gradients work.
"""

import torch

from src.models.encoders import (
    FlatEncoder,
    ObjectCentricEncoder,
    CausalityAwareEncoder,
)
from src.models.heads import DynamicsHead, SubgoalHead
from src.models.losses import total_high_level_loss


def make_dummy_batch(
    batch_size: int = 4,
    history_len: int = 6,
    num_tokens: int = 6,
    raw_token_dim: int = 16,
    action_dim: int = 2,
):
    """
    Create a dummy structured-state batch.

    x shape:
        [B, H, N, D_raw] = [4, 6, 6, 16]

    Token feature index 15 is valid_flag.
    For this basic smoke test, all tokens are marked valid.
    """
    x = torch.randn(batch_size, history_len, num_tokens, raw_token_dim)

    # valid_flag = 1 for all tokens in this smoke test
    x[..., 15] = 1.0

    action = torch.randn(batch_size, action_dim)

    target_delta = torch.randn(batch_size, 3)
    target_subgoal = torch.randn(batch_size, 3)

    return x, action, target_delta, target_subgoal


def run_one_encoder(name: str, encoder: torch.nn.Module):
    torch.manual_seed(42)

    batch_size = 4
    action_dim = 2
    z_dim = 256

    x, action, target_delta, target_subgoal = make_dummy_batch(
        batch_size=batch_size,
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        action_dim=action_dim,
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

    # In this script, encoder forward should return tensor z directly.
    # CausalityAwareEncoder returns dict only when return_slots=True.
    if isinstance(z, dict):
        z = z["z"]

    pred_delta = dyn_head(z, action)
    pred_subgoal = sub_head(z)

    assert z.shape == (batch_size, z_dim), f"{name}: bad z shape {z.shape}"
    assert pred_delta.shape == (batch_size, 3), (
        f"{name}: bad pred_delta shape {pred_delta.shape}"
    )
    assert pred_subgoal.shape == (batch_size, 3), (
        f"{name}: bad pred_subgoal shape {pred_subgoal.shape}"
    )

    losses = total_high_level_loss(
        pred_delta=pred_delta,
        target_delta=target_delta,
        pred_subgoal=pred_subgoal,
        target_subgoal=target_subgoal,
        lambda_subgoal=1.0,
    )

    assert torch.isfinite(losses["loss"]).item(), f"{name}: loss is not finite"

    optimizer.zero_grad()
    losses["loss"].backward()

    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += p.grad.detach().norm().item()

    assert grad_norm > 0.0, f"{name}: no gradients found"

    optimizer.step()

    print(f"[{name}] input:", x.shape)
    print(f"[{name}] z:", z.shape)
    print(f"[{name}] pred_delta:", pred_delta.shape)
    print(f"[{name}] pred_subgoal:", pred_subgoal.shape)
    print(f"[{name}] loss:", float(losses["loss"].detach()))
    print(f"[{name}] grad_norm:", grad_norm)
    print(f"[{name}] backward ok")
    print("-" * 80)


def check_causal_slots():
    """
    Extra diagnostic check for CausalityAwareEncoder.

    Verifies:
        z:             [B, 256]
        z_stable:      [B, 32]
        z_dynamics:    [B, 32]
        z_affordance:  [B, 32]
        z_nuisance:    [B, 32]
    """
    torch.manual_seed(42)

    batch_size = 4

    x, _, _, _ = make_dummy_batch(
        batch_size=batch_size,
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        action_dim=2,
    )

    causal = CausalityAwareEncoder(
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        d_model=128,
        transformer_layers=2,
        nhead=4,
        ffn_dim=256,
        gru_hidden=256,
        slot_dim=32,
    )

    out = causal(x, return_slots=True)

    assert isinstance(out, dict), "Causal encoder should return dict when return_slots=True"

    assert out["z"].shape == (batch_size, 256), out["z"].shape
    assert out["z_stable"].shape == (batch_size, 32), out["z_stable"].shape
    assert out["z_dynamics"].shape == (batch_size, 32), out["z_dynamics"].shape
    assert out["z_affordance"].shape == (batch_size, 32), out["z_affordance"].shape
    assert out["z_nuisance"].shape == (batch_size, 32), out["z_nuisance"].shape

    for key, value in out.items():
        assert torch.isfinite(value).all().item(), f"{key} contains non-finite values"

    print("[causality_aware] slot shapes ok")
    print("[causality_aware] z:", out["z"].shape)
    print("[causality_aware] z_stable:", out["z_stable"].shape)
    print("[causality_aware] z_dynamics:", out["z_dynamics"].shape)
    print("[causality_aware] z_affordance:", out["z_affordance"].shape)
    print("[causality_aware] z_nuisance:", out["z_nuisance"].shape)
    print("-" * 80)


def main():
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
        dropout=0.1,
    )

    causal = CausalityAwareEncoder(
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        d_model=128,
        transformer_layers=2,
        nhead=4,
        ffn_dim=256,
        gru_hidden=256,
        slot_dim=32,
        dropout=0.1,
    )

    run_one_encoder("flat", flat)
    run_one_encoder("object_centric", obj)
    run_one_encoder("causality_aware", causal)

    check_causal_slots()


if __name__ == "__main__":
    main()