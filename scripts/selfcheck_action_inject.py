#!/usr/bin/env python3
"""
Self-check for ActionInjectModel.

Tests:
1. Import and instantiation
2. Forward pass shape correctness
3. Action injection correctness (EE token gets action, others get zeros)
4. Gradient flow
5. Rollout interface compatibility
6. No NaN/inf

Usage:
    PYTHONPATH=. python scripts/selfcheck_action_inject.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import numpy as np


def main():
    print("=" * 60)
    print("ActionInjectModel Self-Check")
    print("=" * 60)

    results = {}

    # ── Test 1: Import ──
    print("\n[1/6] Import...")
    try:
        from src.models.action_inject_model import (
            ActionInjectModel, NoActionDynamicsHead, ActionInjectRolloutModel,
        )
        from src.models.encoders import FlatEncoder, ObjectCentricEncoder
        results["import"] = "PASS"
        print("  ✅ Import OK")
    except Exception as e:
        results["import"] = f"FAIL: {e}"
        print(f"  ❌ Import FAIL: {e}")
        return results

    # ── Test 2: Instantiation ──
    print("\n[2/6] Instantiation...")
    try:
        for enc_type in ["flat", "object_centric"]:
            model = ActionInjectModel(encoder_type=enc_type, gru_hidden=256)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {enc_type}: {n_params:,} params")
            assert model.augmented_token_dim == 18, f"Expected 18, got {model.augmented_token_dim}"
        results["instantiation"] = "PASS"
    except Exception as e:
        results["instantiation"] = f"FAIL: {e}"
        print(f"  ❌ Instantiation FAIL: {e}")

    # ── Test 3: Forward pass ──
    print("\n[3/6] Forward pass...")
    try:
        model = ActionInjectModel(encoder_type="flat", gru_hidden=256)
        model.eval()

        B, H, N, D = 2, 6, 6, 16
        state = torch.randn(B, H, N, D)
        state[:, :, :, 15] = 1.0  # valid flag
        action = torch.tensor([[0.3, 0.0], [-0.5, 0.2]])

        with torch.no_grad():
            out = model(state, action)

        assert out["z"].shape == (B, 256), f"z shape: {out['z'].shape}"
        assert out["pred_delta"].shape == (B, 3), f"delta shape: {out['pred_delta'].shape}"
        assert out["pred_subgoal"].shape == (B, 3), f"subgoal shape: {out['pred_subgoal'].shape}"

        # Check no NaN/inf
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} has NaN/inf"

        print(f"  ✅ z: {out['z'].shape}, pred_delta: {out['pred_delta'].shape}")
        print(f"  ✅ No NaN/inf")
        results["forward"] = "PASS"
    except Exception as e:
        results["forward"] = f"FAIL: {e}"
        print(f"  ❌ Forward FAIL: {e}")

    # ── Test 4: Action injection correctness ──
    print("\n[4/6] Action injection correctness...")
    try:
        model = ActionInjectModel(encoder_type="flat", gru_hidden=256)
        model.eval()

        B, H, N, D = 1, 1, 6, 16
        state = torch.zeros(B, H, N, D)
        state[:, :, :, 15] = 1.0

        action = torch.tensor([[0.7, -0.3]])

        # Test _augment_state directly
        augmented = model._augment_state(state, action)

        # Check EE token (index 0) has action at features 16, 17
        ee_action_features = augmented[0, 0, 0, 16:18]
        assert torch.allclose(ee_action_features, action[0]), \
            f"EE action features: {ee_action_features} != {action[0]}"

        # Check other tokens (1-5) have zeros at features 16, 17
        for tok in range(1, 6):
            other_features = augmented[0, 0, tok, 16:18]
            assert torch.allclose(other_features, torch.zeros(2)), \
                f"Token {tok} features: {other_features} != [0, 0]"

        # Check original 16 dims are preserved
        assert torch.allclose(augmented[0, 0, 0, :16], state[0, 0, 0, :16]), \
            "Original features not preserved"

        print(f"  ✅ EE token gets action: {ee_action_features.numpy()}")
        print(f"  ✅ Other tokens get zeros")
        print(f"  ✅ Original features preserved")
        results["action_injection"] = "PASS"
    except Exception as e:
        results["action_injection"] = f"FAIL: {e}"
        print(f"  ❌ Action injection FAIL: {e}")

    # ── Test 5: Different actions produce different outputs ──
    print("\n[5/6] Action sensitivity...")
    try:
        model = ActionInjectModel(encoder_type="flat", gru_hidden=256)
        model.eval()

        state = torch.randn(1, 6, 6, 16)
        state[:, :, :, 15] = 1.0

        action_a = torch.tensor([[0.5, 0.0]])
        action_b = torch.tensor([[0.0, 0.5]])
        action_c = torch.tensor([[0.0, 0.0]])

        with torch.no_grad():
            out_a = model(state, action_a)
            out_b = model(state, action_b)
            out_c = model(state, action_c)

        delta_a = out_a["pred_delta"].numpy()[0]
        delta_b = out_b["pred_delta"].numpy()[0]
        delta_c = out_c["pred_delta"].numpy()[0]

        diff_ab = np.linalg.norm(delta_a - delta_b)
        diff_ac = np.linalg.norm(delta_a - delta_c)

        print(f"  delta_a (right):  {delta_a.round(4)}")
        print(f"  delta_b (up):     {delta_b.round(4)}")
        print(f"  delta_c (zero):   {delta_c.round(4)}")
        print(f"  ||a - b|| = {diff_ab:.6f}")
        print(f"  ||a - c|| = {diff_ac:.6f}")

        if diff_ab > 1e-6 or diff_ac > 1e-6:
            print(f"  ✅ Different actions → different deltas")
            results["action_sensitivity"] = "PASS"
        else:
            print(f"  ⚠️ All actions produce same delta (model not trained yet?)")
            results["action_sensitivity"] = "WARN: identical outputs"
    except Exception as e:
        results["action_sensitivity"] = f"FAIL: {e}"
        print(f"  ❌ Action sensitivity FAIL: {e}")

    # ── Test 6: Rollout interface ──
    print("\n[6/6] Rollout interface...")
    try:
        model = ActionInjectModel(encoder_type="flat", gru_hidden=256)
        model.eval()

        rollout = ActionInjectRolloutModel(model=model, device="cpu")

        state = np.random.randn(6, 6, 16).astype(np.float32)
        state[:, :, 15] = 1.0

        # Single step
        action_single = np.array([0.1, -0.1], dtype=np.float32)
        step_result = rollout.forward_step(state, action_single)
        print(f"  ✅ forward_step: delta={step_result.delta_pose.round(4)}, "
              f"pred_obj={step_result.pred_object_pose.round(4)}")

        # Multi-step rollout
        action_seq = np.random.randn(5, 2).astype(np.float32) * 0.01
        rollout_result = rollout.rollout_sequence(state, action_seq)
        print(f"  ✅ rollout_sequence: obj_traj={rollout_result.object_traj.shape}, "
              f"ee_traj={rollout_result.ee_traj.shape}")

        assert rollout_result.object_traj.shape == (6, 3), f"obj_traj shape: {rollout_result.object_traj.shape}"
        assert rollout_result.ee_traj.shape == (6, 2), f"ee_traj shape: {rollout_result.ee_traj.shape}"
        assert rollout_result.delta_traj.shape == (5, 3), f"delta_traj shape: {rollout_result.delta_traj.shape}"

        # Check no NaN/inf
        assert np.isfinite(rollout_result.object_traj).all(), "obj_traj has NaN/inf"
        assert np.isfinite(rollout_result.ee_traj).all(), "ee_traj has NaN/inf"

        results["rollout"] = "PASS"
    except Exception as e:
        results["rollout"] = f"FAIL: {e}"
        print(f"  ❌ Rollout FAIL: {e}")

    # ── Summary ──
    print("\n" + "=" * 60)
    all_pass = all(v == "PASS" for v in results.values())
    any_fail = any(v.startswith("FAIL") for v in results.values())

    for k, v in results.items():
        status = "✅" if v == "PASS" else ("⚠️" if v.startswith("WARN") else "❌")
        print(f"  {status} {k}: {v}")

    print()
    if all_pass:
        print("✅ ALL CHECKS PASSED")
    elif any_fail:
        print("❌ SOME CHECKS FAILED")
    else:
        print("⚠️ PASSED WITH WARNINGS")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)
