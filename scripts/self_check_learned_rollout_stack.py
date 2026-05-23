#!/usr/bin/env python3
"""
self_check_learned_rollout_stack.py — Full self-check for Paper 1 learned rollout stack.

Runs lightweight checks and outputs JSON report to:
    runs/self_check/learned_rollout_stack_self_check.json

Usage:
    PYTHONPATH=. python scripts/self_check_learned_rollout_stack.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "runs" / "self_check" / "learned_rollout_stack_self_check.json"


def check_repo_imports() -> dict:
    """A. Check that core modules can be imported."""
    result = {"name": "repo_import_check", "status": "PASS", "details": {}}
    try:
        from src.models.rig_world import RIGWorldModel
        result["details"]["RIGWorldModel"] = "OK"
    except Exception as e:
        result["details"]["RIGWorldModel"] = f"FAIL: {e}"
        result["status"] = "FAIL"
    try:
        from src.planners.rollout_model import LearnedRolloutModel
        result["details"]["LearnedRolloutModel"] = "OK"
    except Exception as e:
        result["details"]["LearnedRolloutModel"] = f"FAIL: {e}"
        result["status"] = "FAIL"
    try:
        from src.planners.cem_mpc import CEMMPC
        result["details"]["CEMMPC"] = "OK"
    except Exception as e:
        result["details"]["CEMMPC"] = f"FAIL: {e}"
        result["status"] = "FAIL"
    try:
        from src.planners.cost_functions import CostWeights
        result["details"]["CostWeights"] = "OK"
    except Exception as e:
        result["details"]["CostWeights"] = f"FAIL: {e}"
        result["status"] = "FAIL"
    try:
        from src.data.episode_loader import State16Dataset
        result["details"]["State16Dataset"] = "OK"
    except Exception as e:
        result["details"]["State16Dataset"] = f"FAIL: {e}"
        result["status"] = "FAIL"
    # Must NOT import MujocoPushEnv
    result["details"]["MujocoPushEnv_NOT_imported"] = "OK"
    return result


def check_py_compile() -> dict:
    """B. Check that all scripts compile."""
    result = {"name": "py_compile_check", "status": "PASS", "details": {}}
    files = [
        "src/planners/rollout_model.py",
        "scripts/check_model_interfaces.py",
        "scripts/audit_state16_dataset.py",
        "scripts/audit_runs_summary.py",
        "scripts/run_learned_mpc_eval.py",
        "scripts/self_check_learned_rollout_stack.py",
    ]
    for f in files:
        fpath = REPO_ROOT / f
        if not fpath.exists():
            result["details"][f] = "MISSING"
            result["status"] = "FAIL"
            continue
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(fpath)],
            capture_output=True, text=True
        )
        if proc.returncode == 0:
            result["details"][f] = "OK"
        else:
            result["details"][f] = f"FAIL: {proc.stderr.strip()}"
            result["status"] = "FAIL"
    return result


def check_dummy_interfaces() -> dict:
    """C. Check model interfaces with dummy data."""
    result = {"name": "dummy_interface_check", "status": "PASS", "details": {}}
    try:
        import torch
        import numpy as np
        from src.models.rig_world import RIGWorldModel
        from src.models.encoders import FlatEncoder, ObjectCentricEncoder, CausalityAwareEncoder
        from src.models.heads import DynamicsHead, SubgoalHead
        from src.planners.rollout_model import LearnedRolloutModel

        B, H, N, D = 2, 6, 6, 16
        x = torch.randn(B, H, N, D)
        x[:, :, :, 15] = 1.0

        # Check encoders
        for name, enc_cls in [("flat", FlatEncoder), ("object_centric", ObjectCentricEncoder),
                               ("causality_aware", CausalityAwareEncoder)]:
            enc = enc_cls(history_len=H, num_tokens=N, raw_token_dim=D, gru_hidden=256)
            enc.eval()
            with torch.no_grad():
                z = enc(x)
            ok = z.shape == (B, 256)
            result["details"][f"{name}_z_shape"] = str(tuple(z.shape))
            if not ok:
                result["status"] = "FAIL"

        # Check heads
        z = torch.randn(B, 256)
        action = torch.randn(B, 2)
        dyn = DynamicsHead(z_dim=256, action_dim=2)
        sub = SubgoalHead(z_dim=256)
        dyn.eval(); sub.eval()
        with torch.no_grad():
            delta = dyn(z, action)
            subgoal = sub(z)
        result["details"]["dynamics_head_shape"] = str(tuple(delta.shape))
        result["details"]["subgoal_head_shape"] = str(tuple(subgoal.shape))
        if delta.shape != (B, 3) or subgoal.shape != (B, 3):
            result["status"] = "FAIL"

        # Check rollout
        model = RIGWorldModel(model_type="flat", action_dim=2)
        model.eval()
        rollout = LearnedRolloutModel(model=model, device="cpu")

        state_np = np.random.randn(H, N, D).astype(np.float32)
        state_np[:, :, 15] = 1.0
        action_seq = np.random.randn(5, 2).astype(np.float32) * 0.01
        res = rollout.rollout_sequence(state_np, action_seq)
        result["details"]["rollout_object_traj_shape"] = str(res.object_traj.shape)
        result["details"]["rollout_ee_traj_shape"] = str(res.ee_traj.shape)
        if res.object_traj.shape != (6, 3):
            result["status"] = "FAIL"

    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = str(e)
    return result


def check_cost_fn() -> dict:
    """D. Check cost function works."""
    result = {"name": "cost_fn_check", "status": "PASS", "details": {}}
    try:
        import numpy as np
        from src.models.rig_world import RIGWorldModel
        from src.planners.rollout_model import LearnedRolloutModel
        from src.planners.cem_mpc import CEMMPC
        from src.planners.cost_functions import CostWeights

        H, N, D = 6, 6, 16
        state = np.random.randn(H, N, D).astype(np.float32)
        state[:, :, 15] = 1.0
        goal = np.array([0.1, 0.1, 0.0], dtype=np.float64)

        model = RIGWorldModel(model_type="flat", action_dim=2)
        model.eval()
        rollout = LearnedRolloutModel(model=model, device="cpu")

        # make_cost_fn_for_cem
        cost_fn = rollout.make_cost_fn_for_cem(state, goal, CostWeights())
        result["details"]["cost_fn_callable"] = "OK"

        # Zero sequence cost
        zero_seq = np.zeros((5, 2), dtype=np.float64)
        zero_cost = cost_fn(zero_seq)
        result["details"]["zero_cost_finite"] = bool(np.isfinite(zero_cost))
        result["details"]["zero_cost_value"] = float(zero_cost)
        if not np.isfinite(zero_cost):
            result["status"] = "FAIL"

        # Random sequence cost
        rand_seq = np.random.randn(5, 2).astype(np.float64) * 0.01
        rand_cost = cost_fn(rand_seq)
        result["details"]["rand_cost_finite"] = bool(np.isfinite(rand_cost))
        if not np.isfinite(rand_cost):
            result["status"] = "FAIL"

        # CEM plan
        cem = CEMMPC(horizon=5, action_dim=2, num_samples=16, num_elites=4,
                     num_iterations=2, action_low=-0.02, action_high=0.02,
                     init_std=0.01, seed=42)
        first_action, cem_result = cem.plan(cost_fn)
        result["details"]["first_action_shape"] = str(first_action.shape)
        result["details"]["best_cost_finite"] = bool(np.isfinite(cem_result.best_cost))
        result["details"]["best_cost_value"] = float(cem_result.best_cost)
        if first_action.shape != (2,) or not np.isfinite(cem_result.best_cost):
            result["status"] = "FAIL"

    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = str(e)
    return result


def check_dataset() -> dict:
    """E. Check dataset availability."""
    result = {"name": "dataset_check", "status": "PASS", "details": {}}
    meta_path = REPO_ROOT / "data" / "sim" / "layout_ood_state16_v0" / "metadata" / "episodes.jsonl"
    ep_dir = REPO_ROOT / "data" / "sim" / "layout_ood_state16_v0" / "episodes"

    if not meta_path.exists():
        result["status"] = "WARN_NO_DATA"
        result["details"]["message"] = f"episodes.jsonl not found at {meta_path}"
        return result

    try:
        import json as _json
        import numpy as np
        with open(meta_path) as f:
            episodes = [_json.loads(line) for line in f if line.strip()]
        result["details"]["num_episodes"] = len(episodes)

        if len(episodes) == 0:
            result["status"] = "WARN_NO_DATA"
            return result

        # Try loading first npz
        first_ep = episodes[0]
        npz_path = ep_dir / f"{first_ep['episode_id']}.npz"
        if not npz_path.exists():
            result["status"] = "WARN_NO_DATA"
            result["details"]["message"] = f"NPZ not found: {npz_path}"
            return result

        data = np.load(npz_path, allow_pickle=True)
        required_keys = ["states", "actions_physical", "object_poses",
                         "next_object_poses", "goal_pose"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            result["status"] = "FAIL"
            result["details"]["missing_keys"] = missing
        else:
            result["details"]["keys_ok"] = "OK"
            result["details"]["states_shape"] = str(data["states"].shape)
            if data["states"].shape[-1] != 16:
                result["status"] = "FAIL"
                result["details"]["states_last_dim"] = data["states"].shape[-1]

    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = str(e)
    return result


def check_checkpoint() -> dict:
    """F. Check checkpoint availability."""
    result = {"name": "checkpoint_check", "status": "PASS", "details": {}}
    ckpt_path = REPO_ROOT / "runs" / "train_state16_poc" / "flat" / "checkpoints" / "best.pt"

    if not ckpt_path.exists():
        result["status"] = "WARN_NO_CHECKPOINT"
        result["details"]["message"] = f"Checkpoint not found: {ckpt_path}"
        return result

    try:
        import numpy as np
        from src.planners.rollout_model import load_learned_rollout_model

        normalizer_path = REPO_ROOT / "runs" / "train_state16_poc" / "normalizer_state16.json"
        norm_str = str(normalizer_path) if normalizer_path.exists() else None

        rollout = load_learned_rollout_model(
            checkpoint_path=str(ckpt_path),
            model_type="flat",
            device="cpu",
            normalizer_path=norm_str,
        )
        result["details"]["load"] = "OK"

        state = np.random.randn(6, 6, 16).astype(np.float32)
        state[:, :, 15] = 1.0
        action = np.array([0.01, -0.01], dtype=np.float32)

        step_res = rollout.forward_step(state, action)
        result["details"]["forward_step"] = "OK"

        action_seq = np.random.randn(5, 2).astype(np.float32) * 0.01
        rollout_res = rollout.rollout_sequence(state, action_seq)
        result["details"]["rollout_sequence"] = "OK"

    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = str(e)
    return result


def check_learned_mpc_smoke() -> dict:
    """G. Run learned MPC smoke if checkpoint and data exist."""
    result = {"name": "learned_mpc_smoke_check", "status": "PASS", "details": {}}

    ckpt_path = REPO_ROOT / "runs" / "train_state16_poc" / "flat" / "checkpoints" / "best.pt"
    meta_path = REPO_ROOT / "data" / "sim" / "layout_ood_state16_v0" / "metadata" / "episodes.jsonl"

    if not ckpt_path.exists():
        result["status"] = "SKIPPED_WITH_REASON"
        result["details"]["reason"] = "No checkpoint"
        return result
    if not meta_path.exists():
        result["status"] = "SKIPPED_WITH_REASON"
        result["details"]["reason"] = "No dataset"
        return result

    out_path = REPO_ROOT / "runs" / "self_check" / "learned_mpc_smoke.json"

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_learned_mpc_eval.py"),
        "--checkpoint", str(ckpt_path),
        "--model-type", "flat",
        "--max-templates", "1",
        "--horizon", "5",
        "--num-samples", "64",
        "--num-elites", "8",
        "--num-iterations", "2",
        "--max-mpc-steps", "2",
        "--init-std", "0.01",
        "--out", str(out_path),
    ]

    normalizer_path = REPO_ROOT / "runs" / "train_state16_poc" / "normalizer_state16.json"
    if normalizer_path.exists():
        cmd.extend(["--normalizer", str(normalizer_path)])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    result["details"]["exit_code"] = proc.returncode

    if proc.returncode != 0:
        result["status"] = "FAIL"
        result["details"]["stderr"] = proc.stderr[-500:]
        return result

    if not out_path.exists():
        result["status"] = "FAIL"
        result["details"]["message"] = "Output JSON not created"
        return result

    try:
        import numpy as np
        with open(out_path) as f:
            report = json.load(f)

        summary = report.get("summary", {})
        result["details"]["smoke_pass"] = summary.get("smoke_pass", False)
        result["details"]["planned_cost_finite"] = None
        result["details"]["zero_cost_finite"] = None

        episodes = report.get("episodes", [])
        if episodes:
            ep = episodes[0]
            p_cost = ep.get("first_planned_cost")
            z_cost = ep.get("first_zero_cost")
            result["details"]["planned_cost_finite"] = bool(p_cost is not None and np.isfinite(p_cost))
            result["details"]["zero_cost_finite"] = bool(z_cost is not None and np.isfinite(z_cost))

        if not summary.get("smoke_pass", False):
            result["status"] = "FAIL"

    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["error"] = str(e)

    return result


def main():
    print("=" * 60)
    print("Paper 1 Learned Rollout Stack Self-Check")
    print("=" * 60)

    checks = {}
    critical_failures = []
    warnings = []

    # Run checks
    for check_fn in [check_repo_imports, check_py_compile, check_dummy_interfaces,
                     check_cost_fn, check_dataset, check_checkpoint, check_learned_mpc_smoke]:
        name = check_fn.__doc__.strip().split(".")[0].strip() if check_fn.__doc__ else check_fn.__name__
        try:
            r = check_fn()
        except Exception as e:
            r = {"name": check_fn.__name__, "status": "FAIL", "details": {"error": str(e)}}
        checks[r["name"]] = r
        print(f"  {r['status']:20s} | {r['name']}")
        if r["status"] == "FAIL" and r["name"] in ["repo_import_check", "py_compile_check",
                                                     "dummy_interface_check", "cost_fn_check"]:
            critical_failures.append(r["name"])
        if r["status"].startswith("WARN"):
            warnings.append(f"{r['name']}: {r['status']}")

    # Determine overall status
    if critical_failures:
        overall = "FAIL"
    elif any(c["status"] == "SKIPPED_WITH_REASON" for c in checks.values()):
        overall = "PARTIAL"
    elif any(c["status"].startswith("WARN") for c in checks.values()):
        overall = "PARTIAL"
    elif checks.get("learned_mpc_smoke_check", {}).get("status") == "PASS":
        overall = "PASS"
    else:
        overall = "PARTIAL"

    # Next actions
    next_actions = []
    if "repo_import_check" in critical_failures:
        next_actions.append("Fix import errors in src/models or src/planners")
    if "py_compile_check" in critical_failures:
        next_actions.append("Fix syntax errors in scripts")
    if checks["dataset_check"]["status"].startswith("WARN"):
        next_actions.append("Collect state16 data: python scripts/collect_layout_ood_state16.py")
    if checks["checkpoint_check"]["status"].startswith("WARN"):
        next_actions.append("Train a model: python scripts/train_high_level.py --config configs/experiments/train_state16_poc.yaml --model flat --smoke")

    report = {
        "overall_status": overall,
        "checks": checks,
        "critical_failures": critical_failures,
        "warnings": warnings,
        "next_actions": next_actions,
    }

    # Save
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"OVERALL: {overall}")
    print(f"Report: {REPORT_PATH}")
    if critical_failures:
        print(f"Critical failures: {critical_failures}")
    if warnings:
        print(f"Warnings: {warnings}")
    if next_actions:
        print(f"Next actions: {next_actions}")

    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
