#!/usr/bin/env python3
"""
self_check_mppi_stage2c_pilot_ready.py — Preflight check for mppi_stage2c_state16 pilot.

Output: runs/pilot_state16_mppi_stage2c/preflight_ready.json

Usage:
    PYTHONPATH=. python scripts/self_check_mppi_stage2c_pilot_ready.py
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "runs" / "pilot_state16_mppi_stage2c" / "preflight_ready.json"


def check_file_exist() -> dict:
    r = {"name": "file_exist_check", "status": "PASS", "details": {}}
    files = [
        "src/models/encoders.py", "src/models/heads.py", "src/models/rig_world.py",
        "src/models/losses.py", "src/data/state_normalizer.py", "src/data/episode_loader.py",
        "src/planners/rollout_model.py", "src/planners/cem_mpc.py", "src/planners/cost_functions.py",
        "scripts/train_high_level.py", "scripts/run_learned_mpc_eval.py",
        "configs/train/rig_world_shared.yaml", "configs/train/flat_high_level.yaml",
        "configs/train/object_centric_noncausal.yaml", "configs/train/causality_aware.yaml",
        "data/sim/mppi_stage2c_state16/metadata/episodes.jsonl",
    ]
    for f in files:
        if not (REPO_ROOT / f).exists():
            r["details"][f] = "MISSING"
            r["status"] = "FAIL"
        else:
            r["details"][f] = "OK"
    return r


def check_py_compile() -> dict:
    r = {"name": "py_compile_check", "status": "PASS", "details": {}}
    files = [
        "src/models/encoders.py", "src/models/heads.py", "src/models/rig_world.py",
        "src/models/losses.py", "src/data/state_normalizer.py", "src/data/episode_loader.py",
        "src/planners/rollout_model.py", "scripts/train_high_level.py",
        "scripts/run_learned_mpc_eval.py", "scripts/create_mppi_stage2c_pilot_split.py",
    ]
    for f in files:
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(REPO_ROOT / f)],
                              capture_output=True, text=True)
        r["details"][f] = "OK" if proc.returncode == 0 else f"FAIL: {proc.stderr.strip()[:100]}"
        if proc.returncode != 0:
            r["status"] = "FAIL"
    return r


def check_data_schema() -> dict:
    r = {"name": "data_schema_check", "status": "PASS", "details": {}}
    try:
        import numpy as np
        meta = REPO_ROOT / "data" / "sim" / "mppi_stage2c_state16" / "metadata" / "episodes.jsonl"
        ep_dir = REPO_ROOT / "data" / "sim" / "mppi_stage2c_state16" / "episodes"
        with open(meta) as f:
            eps = [json.loads(l) for l in f if l.strip()]

        r["details"]["num_episodes"] = len(eps)

        # Check 5 random npz
        import random
        random.seed(42)
        sample_eps = random.sample(eps, min(5, len(eps)))

        for ep in sample_eps:
            npz = ep_dir / f"{ep['episode_id']}.npz"
            d = np.load(npz, allow_pickle=True)

            # Check shapes
            states = d["states"]
            if states.ndim != 3 or states.shape[1] != 6 or states.shape[2] != 16:
                r["details"][f"{ep['episode_id']}_states_shape"] = str(states.shape)
                r["status"] = "FAIL"
            elif np.any(np.isnan(states)) or np.any(np.isinf(states)):
                r["details"][f"{ep['episode_id']}_nan_inf"] = True
                r["status"] = "FAIL"
            else:
                r["details"][f"{ep['episode_id']}_states_shape"] = str(states.shape)
                r["details"][f"{ep['episode_id']}_T"] = states.shape[0]

            # Check required keys
            for k in ["actions_physical", "object_poses", "next_object_poses", "goal_pose"]:
                if k not in d:
                    r["details"][f"{ep['episode_id']}_{k}_missing"] = True
                    r["status"] = "FAIL"
    except Exception as e:
        r["status"] = "FAIL"
        r["details"]["error"] = str(e)
    return r


def check_split() -> dict:
    r = {"name": "split_check", "status": "PASS", "details": {}}
    split_path = REPO_ROOT / "configs" / "splits" / "mppi_stage2c_state16_pilot.yaml"
    if not split_path.exists():
        r["status"] = "FAIL"
        r["details"]["message"] = f"Split file not found: {split_path}"
        return r

    with open(split_path) as f:
        split_cfg = yaml.safe_load(f)

    for split_name, split_data in split_cfg.get("splits", {}).items():
        all_ids = []
        for part, ids in split_data.items():
            r["details"][f"{split_name}/{part}_count"] = len(ids)
            all_ids.extend(ids)

        # Check no overlap
        if len(all_ids) != len(set(all_ids)):
            r["status"] = "FAIL"
            r["details"][f"{split_name}_overlap"] = True
        else:
            r["details"][f"{split_name}_no_overlap"] = True

    return r


def check_dataset_loader() -> dict:
    r = {"name": "dataset_loader_check", "status": "PASS", "details": {}}
    try:
        import numpy as np
        import torch
        from src.data.episode_loader import State16Dataset

        meta = str(REPO_ROOT / "data" / "sim" / "mppi_stage2c_state16" / "metadata" / "episodes.jsonl")
        ep_root = str(REPO_ROOT / "data" / "sim" / "mppi_stage2c_state16" / "episodes")
        split_path = str(REPO_ROOT / "configs" / "splits" / "mppi_stage2c_state16_pilot.yaml")

        with open(split_path) as f:
            split_cfg = yaml.safe_load(f)

        train_ids = split_cfg["splits"]["random_episode_split"]["train"]

        ds = State16Dataset(
            metadata_path=meta, episode_root=ep_root,
            history_len=6, token_count=6, state_dim=16,
            split="train", split_episode_ids=train_ids,
        )
        r["details"]["train_dataset_len"] = len(ds)

        if len(ds) == 0:
            r["status"] = "FAIL"
            r["details"]["message"] = "Empty dataset"
            return r

        # Get one batch
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=lambda batch: (
            torch.stack([b["history"] for b in batch]),
            torch.stack([b["action"] for b in batch]),
            torch.stack([b["dynamics_target"] for b in batch]),
            torch.stack([b["subgoal_target"] for b in batch]),
        ))
        history, action, dyn_target, sub_target = next(iter(loader))
        r["details"]["history_shape"] = str(tuple(history.shape))
        r["details"]["action_shape"] = str(tuple(action.shape))
        r["details"]["dyn_target_shape"] = str(tuple(dyn_target.shape))
        r["details"]["sub_target_shape"] = str(tuple(sub_target.shape))

        if history.shape[1:] != (6, 6, 16):
            r["status"] = "FAIL"
    except Exception as e:
        r["status"] = "FAIL"
        r["details"]["error"] = str(e)
    return r


def check_normalizer() -> dict:
    r = {"name": "normalizer_check", "status": "PASS", "details": {}}
    try:
        import numpy as np
        from src.data.state_normalizer import StateNormalizer

        normalizer = StateNormalizer()
        # Dummy fit
        dummy = np.random.randn(100, 16).astype(np.float32)
        dummy[:, 15] = 1.0
        normalizer.fit(dummy)
        transformed = normalizer.transform(dummy)
        if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
            r["status"] = "FAIL"
            r["details"]["nan_inf_after_transform"] = True
        else:
            r["details"]["transform_ok"] = True
        # Invalid tokens should be 0
        dummy[0, 15] = 0.0
        transformed = normalizer.transform(dummy)
        if np.all(transformed[0] == 0.0):
            r["details"]["invalid_token_zeroed"] = True
        else:
            r["details"]["invalid_token_zeroed"] = False
    except Exception as e:
        r["status"] = "FAIL"
        r["details"]["error"] = str(e)
    return r


def check_model_forward() -> dict:
    r = {"name": "model_forward_check", "status": "PASS", "details": {}}
    try:
        import torch
        import numpy as np
        from src.models.rig_world import RIGWorldModel

        B, H, N, D = 4, 6, 6, 16
        x = torch.randn(B, H, N, D)
        x[:, :, :, 15] = 1.0
        action = torch.randn(B, 2)

        for model_type in ["flat", "object_centric", "causality_aware"]:
            model = RIGWorldModel(model_type=model_type, action_dim=2)
            model.eval()
            with torch.no_grad():
                out = model(x, action)
            ok = (out["z"].shape == (B, 256) and
                  out["pred_delta"].shape == (B, 3) and
                  out["pred_subgoal"].shape == (B, 3) and
                  torch.isfinite(out["z"]).all() and
                  torch.isfinite(out["pred_delta"]).all())
            r["details"][model_type] = "PASS" if ok else "FAIL"
            if not ok:
                r["status"] = "FAIL"
    except Exception as e:
        r["status"] = "FAIL"
        r["details"]["error"] = str(e)
    return r


def check_loss_backward() -> dict:
    r = {"name": "loss_backward_check", "status": "PASS", "details": {}}
    try:
        import torch
        from src.models.rig_world import RIGWorldModel
        from src.models.losses import total_high_level_loss

        B = 4
        x = torch.randn(B, 6, 6, 16)
        x[:, :, :, 15] = 1.0
        action = torch.randn(B, 2)
        dyn_target = torch.randn(B, 3)
        sub_target = torch.randn(B, 3)

        for model_type in ["flat", "object_centric", "causality_aware"]:
            model = RIGWorldModel(model_type=model_type, action_dim=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            out = model(x, action)
            losses = total_high_level_loss(out["pred_delta"], dyn_target, out["pred_subgoal"], sub_target)
            loss = losses["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ok = torch.isfinite(loss)
            r["details"][model_type] = f"PASS loss={loss.item():.4f}"
            if not ok:
                r["status"] = "FAIL"
    except Exception as e:
        r["status"] = "FAIL"
        r["details"]["error"] = str(e)
    return r


def check_learned_mpc_entry() -> dict:
    r = {"name": "learned_mpc_eval_entry_check", "status": "PASS", "details": {}}
    try:
        # Check that run_learned_mpc_eval.py supports --dataset-dir
        with open(REPO_ROOT / "scripts" / "run_learned_mpc_eval.py") as f:
            content = f.read()
        if "--dataset-dir" in content:
            r["details"]["dataset_dir_arg"] = "OK"
        else:
            r["status"] = "FAIL"
            r["details"]["dataset_dir_arg"] = "MISSING"

        # Check no hardcoded layout_ood_state16_v0 in critical paths
        if "layout_ood_state16_v0" in content and "default" in content.split("layout_ood_state16_v0")[0][-50:]:
            r["details"]["old_default_path"] = "Still present but as default fallback"
        else:
            r["details"]["old_default_path"] = "OK"
    except Exception as e:
        r["status"] = "FAIL"
        r["details"]["error"] = str(e)
    return r


def main():
    print("=" * 60)
    print("mppi_stage2c_state16 Pilot Preflight Check")
    print("=" * 60)

    checks = []
    for fn in [check_file_exist, check_py_compile, check_data_schema,
               check_split, check_dataset_loader, check_normalizer,
               check_model_forward, check_loss_backward, check_learned_mpc_entry]:
        r = fn()
        checks.append(r)
        print(f"  {r['status']:10s} | {r['name']}")

    critical = ["file_exist_check", "py_compile_check", "data_schema_check",
                "model_forward_check", "loss_backward_check"]
    critical_failures = [c["name"] for c in checks if c["status"] == "FAIL" and c["name"] in critical]
    warnings = [c["name"] for c in checks if c["status"] in ("PARTIAL", "WARN")]

    if critical_failures:
        overall = "FAIL"
    elif any(c["status"] == "FAIL" for c in checks):
        overall = "PARTIAL"
    else:
        overall = "PASS"

    next_actions = []
    if "file_exist_check" in [c["name"] for c in checks if c["status"] == "FAIL"]:
        next_actions.append("Check missing files")
    if "data_schema_check" in [c["name"] for c in checks if c["status"] == "FAIL"]:
        next_actions.append("Re-run convert_mppi_to_state16.py")
    if "split_check" in [c["name"] for c in checks if c["status"] == "FAIL"]:
        next_actions.append("Run create_mppi_stage2c_pilot_split.py")

    report = {
        "overall_status": overall,
        "checks": {c["name"]: c for c in checks},
        "critical_failures": critical_failures,
        "warnings": warnings,
        "next_actions": next_actions,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {overall}")
    print(f"Report: {REPORT_PATH}")
    if critical_failures:
        print(f"Critical failures: {critical_failures}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
