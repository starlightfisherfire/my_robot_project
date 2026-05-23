#!/usr/bin/env python3
"""
check_model_interfaces.py — Verify Paper 1 model interface contracts.

Checks:
    1. All three encoders produce same output shape [B, z_dim]
    2. DynamicsHead output shape [B, 3]
    3. SubgoalHead output shape [B, 3]
    4. RIGWorldModel unified interface works for all variants
    5. LearnedRolloutModel forward_step and rollout_sequence work

Usage:
    python scripts/check_model_interfaces.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.rig_world import RIGWorldModel
from src.models.encoders import FlatEncoder, ObjectCentricEncoder, CausalityAwareEncoder
from src.models.heads import DynamicsHead, SubgoalHead


def check_encoders():
    """Check all three encoders produce same output shape."""
    print("=" * 60)
    print("CHECK 1: Encoder output shapes")
    print("=" * 60)

    B, H, N, D = 2, 6, 6, 16
    x = torch.randn(B, H, N, D)
    # Set valid flags (index 15) so ObjectCentric/CausalityAware encoders work
    x[:, :, :, 15] = 1.0
    z_dim = 256

    encoders = {
        "flat": FlatEncoder(history_len=H, num_tokens=N, raw_token_dim=D, gru_hidden=z_dim),
        "object_centric": ObjectCentricEncoder(history_len=H, num_tokens=N, raw_token_dim=D, gru_hidden=z_dim),
        "causality_aware": CausalityAwareEncoder(history_len=H, num_tokens=N, raw_token_dim=D, gru_hidden=z_dim),
    }

    all_pass = True
    for name, enc in encoders.items():
        enc.eval()
        with torch.no_grad():
            z = enc(x)
        ok = z.shape == (B, z_dim)
        status = "✅" if ok else "❌"
        print(f"  {status} {name}: z.shape = {tuple(z.shape)} (expected [{B}, {z_dim}])")
        if not ok:
            all_pass = False

    return all_pass


def check_heads():
    """Check dynamics and subgoal heads."""
    print("\n" + "=" * 60)
    print("CHECK 2: Head output shapes")
    print("=" * 60)

    B, z_dim, action_dim = 2, 256, 2
    z = torch.randn(B, z_dim)
    action = torch.randn(B, action_dim)

    dyn_head = DynamicsHead(z_dim=z_dim, action_dim=action_dim)
    sub_head = SubgoalHead(z_dim=z_dim)

    dyn_head.eval()
    sub_head.eval()

    with torch.no_grad():
        delta = dyn_head(z, action)
        subgoal = sub_head(z)

    ok_delta = delta.shape == (B, 3)
    ok_subgoal = subgoal.shape == (B, 3)

    print(f"  {'✅' if ok_delta else '❌'} DynamicsHead: delta.shape = {tuple(delta.shape)} (expected [{B}, 3])")
    print(f"  {'✅' if ok_subgoal else '❌'} SubgoalHead: subgoal.shape = {tuple(subgoal.shape)} (expected [{B}, 3])")

    return ok_delta and ok_subgoal


def check_rig_world_model():
    """Check RIGWorldModel unified interface for all variants."""
    print("\n" + "=" * 60)
    print("CHECK 3: RIGWorldModel unified interface")
    print("=" * 60)

    B, H, N, D, action_dim = 2, 6, 6, 16, 2
    x = torch.randn(B, H, N, D)
    x[:, :, :, 15] = 1.0  # Set valid flags
    action = torch.randn(B, action_dim)

    all_pass = True
    for model_type in ["flat", "object_centric", "causality_aware"]:
        model = RIGWorldModel(model_type=model_type, action_dim=action_dim)
        model.eval()
        with torch.no_grad():
            out = model(x, action)

        ok_z = out["z"].shape == (B, 256)
        ok_delta = out["pred_delta"].shape == (B, 3)
        ok_subgoal = out["pred_subgoal"].shape == (B, 3)
        ok = ok_z and ok_delta and ok_subgoal

        print(f"  {'✅' if ok else '❌'} {model_type}:")
        print(f"      z.shape = {tuple(out['z'].shape)}")
        print(f"      pred_delta.shape = {tuple(out['pred_delta'].shape)}")
        print(f"      pred_subgoal.shape = {tuple(out['pred_subgoal'].shape)}")

        if model_type == "causality_aware":
            model.eval()
            with torch.no_grad():
                out_slots = model(x, action, return_slots=True)
            for slot in ["z_stable", "z_dynamics", "z_affordance", "z_nuisance"]:
                print(f"      {slot}.shape = {tuple(out_slots[slot].shape)}")

        if not ok:
            all_pass = False

    return all_pass


def check_learned_rollout():
    """Check LearnedRolloutModel forward_step and rollout_sequence."""
    print("\n" + "=" * 60)
    print("CHECK 4: LearnedRolloutModel")
    print("=" * 60)

    from src.planners.rollout_model import LearnedRolloutModel

    H, N, D, action_dim = 6, 6, 16, 2
    state = np.random.randn(H, N, D).astype(np.float32)
    # Set valid flags
    state[:, :, 15] = 1.0

    model = RIGWorldModel(model_type="flat", action_dim=action_dim)
    model.eval()

    rollout = LearnedRolloutModel(model=model, device="cpu")

    # Test forward_step
    action = np.array([0.01, -0.01], dtype=np.float32)
    step_result = rollout.forward_step(state, action)
    print(f"  ✅ forward_step:")
    print(f"      delta_pose.shape = {step_result.delta_pose.shape}")
    print(f"      pred_object_pose.shape = {step_result.pred_object_pose.shape}")

    # Test rollout_sequence
    horizon = 10
    action_seq = np.random.randn(horizon, action_dim).astype(np.float32) * 0.01
    result = rollout.rollout_sequence(state, action_seq)
    print(f"  ✅ rollout_sequence (horizon={horizon}):")
    print(f"      object_traj.shape = {result.object_traj.shape}")
    print(f"      delta_traj.shape = {result.delta_traj.shape}")
    print(f"      ee_traj.shape = {result.ee_traj.shape}")

    # Verify shapes
    ok_obj = result.object_traj.shape == (horizon + 1, 3)
    ok_delta = result.delta_traj.shape == (horizon, 3)
    ok_ee = result.ee_traj.shape == (horizon + 1, 2)
    ok = ok_obj and ok_delta and ok_ee

    print(f"  {'✅' if ok else '❌'} Shape verification: {'PASS' if ok else 'FAIL'}")

    return ok


def check_batch_rollout():
    """Check batched rollout."""
    print("\n" + "=" * 60)
    print("CHECK 5: Batched rollout")
    print("=" * 60)

    from src.planners.rollout_model import LearnedRolloutModel

    B, H, N, D, action_dim = 3, 6, 6, 16, 2
    state = np.random.randn(B, H, N, D).astype(np.float32)
    state[:, :, :, 15] = 1.0

    model = RIGWorldModel(model_type="flat", action_dim=action_dim)
    model.eval()

    rollout = LearnedRolloutModel(model=model, device="cpu")

    horizon = 5
    action_seq = np.random.randn(B, horizon, action_dim).astype(np.float32) * 0.01
    result = rollout.rollout_sequence(state, action_seq)

    ok_obj = result.object_traj.shape == (B, horizon + 1, 3)
    ok_delta = result.delta_traj.shape == (B, horizon, 3)
    ok_ee = result.ee_traj.shape == (B, horizon + 1, 2)
    ok = ok_obj and ok_delta and ok_ee

    print(f"  {'✅' if ok else '❌'} Batch rollout (B={B}, horizon={horizon}):")
    print(f"      object_traj.shape = {result.object_traj.shape}")
    print(f"      delta_traj.shape = {result.delta_traj.shape}")
    print(f"      ee_traj.shape = {result.ee_traj.shape}")

    return ok


def main():
    print("Paper 1 Model Interface Check")
    print("=" * 60)

    results = {
        "encoders": check_encoders(),
        "heads": check_heads(),
        "rig_world": check_rig_world_model(),
        "rollout_single": check_learned_rollout(),
        "rollout_batch": check_batch_rollout(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n🎉 ALL CHECKS PASSED")
    else:
        print("\n⚠️ SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
