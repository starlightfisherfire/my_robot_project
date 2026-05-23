#!/usr/bin/env python3
"""Gate 1: Code compilation + interface self-check."""

import argparse, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def compile_check(files):
    for f in files:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(REPO_ROOT / f)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[FAIL] py_compile: {f}")
            print(result.stderr)
            return False
        print(f"[PASS] py_compile: {f}")
    return True


def interface_check():
    import torch
    from src.models.rig_world import RIGWorldModel

    torch.manual_seed(42)
    B, H, N, D = 2, 6, 6, 16
    x = torch.randn(B, H, N, D)
    x[..., 15] = 1.0  # valid_flag
    action = torch.randn(B, 2)

    for mtype in ["flat", "object_centric", "causality_aware"]:
        model = RIGWorldModel(model_type=mtype)
        out = model(x, action)

        assert out["z"].shape == (B, 256), f"{mtype}: z shape {out['z'].shape}"
        assert out["pred_delta"].shape == (B, 3), f"{mtype}: delta shape {out['pred_delta'].shape}"
        assert out["pred_subgoal"].shape == (B, 3), f"{mtype}: subgoal shape {out['pred_subgoal'].shape}"
        assert torch.isfinite(out["z"]).all(), f"{mtype}: z has NaN/Inf"
        assert torch.isfinite(out["pred_delta"]).all(), f"{mtype}: delta has NaN/Inf"
        assert torch.isfinite(out["pred_subgoal"]).all(), f"{mtype}: subgoal has NaN/Inf"

        # Backward check
        loss = out["pred_delta"].mean() + out["pred_subgoal"].mean()
        loss.backward()

        # Slot check for causal
        if mtype == "causality_aware":
            out2 = model(x, action, return_slots=True)
            assert "z_stable" in out2, "Missing z_stable"
            assert out2["z_stable"].shape == (B, 32)

        print(f"[PASS] interface: {mtype} (forward+backward)")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/train_state16_poc.yaml")
    args = parser.parse_args()

    print("=== Gate 1: Code Compilation + Interface Self-Check ===\n")

    files_to_check = [
        "src/models/encoders.py",
        "src/models/rig_world.py",
        "src/models/heads.py",
        "src/models/losses.py",
        "src/data/episode_loader.py",
        "src/data/state_normalizer.py",
        "scripts/train_high_level.py",
        "scripts/eval_high_level.py",
    ]

    ok1 = compile_check(files_to_check)
    ok2 = interface_check()

    if ok1 and ok2:
        print("\n=== Gate 1: PASS ===")
        sys.exit(0)
    else:
        print("\n=== Gate 1: FAIL ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
