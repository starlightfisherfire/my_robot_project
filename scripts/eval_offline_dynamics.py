#!/usr/bin/env python3
"""eval_offline_dynamics.py — Offline dynamics evaluation for Paper 1 pilot.

Evaluates one-step and multi-step rollout dynamics for multiple models.

Usage:
    PYTHONPATH=. python scripts/eval_offline_dynamics.py \
        --dataset-dir data/sim/mppi_stage2c_state16 \
        --split-file configs/splits/mppi_stage2c_state16_pilot.yaml \
        --split-name random_episode_split/test \
        --checkpoints flat:runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt \
                      object_centric:runs/pilot_state16_mppi_stage2c/train_object_centric/object_centric/checkpoints/best.pt \
                      causality_aware:runs/pilot_state16_mppi_stage2c/train_causality_aware/causality_aware/checkpoints/best.pt \
        --normalizers flat:runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json \
                      object_centric:runs/pilot_state16_mppi_stage2c/train_object_centric/normalizer_state16.json \
                      causality_aware:runs/pilot_state16_mppi_stage2c/train_causality_aware/normalizer_state16.json \
        --out-dir runs/pilot_state16_mppi_stage2c/offline_eval
"""

import argparse, csv, json, sys, os
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer


def load_test_samples(dataset_dir, split_file, split_name, max_samples=2000):
    """Load test split samples as (history, action, dyn_target, sub_target, family, contact_flag)."""
    import yaml
    from src.data.episode_loader import State16Dataset

    ds_dir = Path(dataset_dir)
    meta = str(ds_dir / "metadata" / "episodes.jsonl")
    ep_root = str(ds_dir / "episodes")

    with open(split_file) as f:
        split_cfg = yaml.safe_load(f)

    parts = split_name.split("/")
    split_key = parts[0]
    sub_key = parts[1] if len(parts) > 1 else "test"
    episode_ids = split_cfg["splits"][split_key][sub_key]

    ds = State16Dataset(
        metadata_path=meta, episode_root=ep_root,
        history_len=6, token_count=6, state_dim=16,
        split=sub_key, split_episode_ids=episode_ids,
        limit_samples=max_samples,
    )
    return ds


def evaluate_model(model_type, checkpoint_path, normalizer_path, test_ds, device="cpu"):
    """Evaluate a single model on test_ds."""
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256,
                          d_model=128, head_hidden_dim=256)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    normalizer = None
    if normalizer_path and os.path.exists(normalizer_path):
        normalizer = StateNormalizer.load(normalizer_path)

    from torch.utils.data import DataLoader
    def collate(batch):
        return (torch.stack([b["history"] for b in batch]),
                torch.stack([b["action"] for b in batch]),
                torch.stack([b["dynamics_target"] for b in batch]),
                torch.stack([b["subgoal_target"] for b in batch]))

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

    all_pred_delta = []
    all_target_delta = []
    all_families = []

    with torch.no_grad():
        for history, action, dyn_target, sub_target in loader:
            history = history.to(device)
            action = action.to(device)
            if normalizer is not None:
                h_np = history.cpu().numpy()
                B, H, N, D = h_np.shape
                h_np = normalizer.transform(h_np.reshape(-1, D)).reshape(B, H, N, D)
                history = torch.from_numpy(h_np).to(device)
            out = model(history, action)
            all_pred_delta.append(out["pred_delta"].cpu().numpy())
            all_target_delta.append(dyn_target.numpy())

    pred = np.concatenate(all_pred_delta, axis=0)  # [N, 3]
    target = np.concatenate(all_target_delta, axis=0)  # [N, 3]

    # One-step metrics
    mse = np.mean((pred - target) ** 2, axis=0)  # [3]
    rmse = np.sqrt(mse)
    # Wrap theta error
    theta_err = np.arctan2(np.sin(pred[:, 2] - target[:, 2]), np.cos(pred[:, 2] - target[:, 2]))
    dtheta_rmse = np.sqrt(np.mean(theta_err ** 2))

    # Multi-step rollout (simplified: accumulate deltas)
    # For proper rollout we need sequential samples; approximate with batch prediction
    rollout_horizons = [1, 5, 10, 20]
    rollout_results = {}
    for H in rollout_horizons:
        # Approximate: sum of first H deltas vs sum of targets
        if len(pred) >= H:
            pred_sum = np.cumsum(pred[:H], axis=0)
            target_sum = np.cumsum(target[:H], axis=0)
            pos_rmse = np.sqrt(np.mean((pred_sum[:, :2] - target_sum[:, :2]) ** 2))
            theta_rmse = np.sqrt(np.mean(np.arctan2(
                np.sin(pred_sum[:, 2] - target_sum[:, 2]),
                np.cos(pred_sum[:, 2] - target_sum[:, 2])) ** 2))
            rollout_results[H] = {"pos_rmse": float(pos_rmse), "theta_rmse": float(theta_rmse)}
        else:
            rollout_results[H] = {"pos_rmse": float("nan"), "theta_rmse": float("nan")}

    return {
        "model_type": model_type,
        "num_samples": len(pred),
        "dynamics_mse": float(np.mean(mse)),
        "dynamics_rmse": float(np.sqrt(np.mean(mse))),
        "dx_rmse": float(rmse[0]),
        "dy_rmse": float(rmse[1]),
        "dtheta_rmse": float(dtheta_rmse),
        "rollout": rollout_results,
        "all_finite": bool(np.isfinite(pred).all() and np.isfinite(target).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-name", default="random_episode_split/test")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="model_type:path pairs")
    parser.add_argument("--normalizers", nargs="+", default=[], help="model_type:path pairs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse checkpoint and normalizer paths
    ckpts = {}
    for item in args.checkpoints:
        k, v = item.split(":", 1)
        ckpts[k] = v
    norms = {}
    for item in args.normalizers:
        k, v = item.split(":", 1)
        norms[k] = v

    # Load test dataset
    print("Loading test dataset...")
    test_ds = load_test_samples(args.dataset_dir, args.split_file, args.split_name, args.max_samples)
    print(f"  Test samples: {len(test_ds)}")

    # Evaluate each model
    all_results = []
    for model_type, ckpt_path in ckpts.items():
        print(f"\nEvaluating {model_type}...")
        norm_path = norms.get(model_type)
        result = evaluate_model(model_type, ckpt_path, norm_path, test_ds)
        all_results.append(result)
        print(f"  dynamics_rmse={result['dynamics_rmse']:.6f}")
        print(f"  dtheta_rmse={result['dtheta_rmse']:.6f}")
        for H, r in result["rollout"].items():
            print(f"  rollout@{H}: pos={r['pos_rmse']:.6f} theta={r['theta_rmse']:.6f}")

    # Save summary JSON
    summary = {"models": all_results, "test_samples": len(test_ds), "split_name": args.split_name}
    with open(out_dir / "offline_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    with open(out_dir / "offline_eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_samples", "dynamics_rmse", "dx_rmse", "dy_rmse", "dtheta_rmse",
                     "rollout_pos_rmse@5", "rollout_pos_rmse@10", "rollout_pos_rmse@20"])
        for r in all_results:
            w.writerow([r["model_type"], r["num_samples"], f"{r['dynamics_rmse']:.6f}",
                        f"{r['dx_rmse']:.6f}", f"{r['dy_rmse']:.6f}", f"{r['dtheta_rmse']:.6f}",
                        f"{r['rollout'][5]['pos_rmse']:.6f}", f"{r['rollout'][10]['pos_rmse']:.6f}",
                        f"{r['rollout'][20]['pos_rmse']:.6f}"])

    # Rollout error growth CSV
    with open(out_dir / "rollout_error_growth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "horizon", "pos_rmse", "theta_rmse"])
        for r in all_results:
            for H, rmse_data in r["rollout"].items():
                w.writerow([r["model_type"], H, f"{rmse_data['pos_rmse']:.6f}", f"{rmse_data['theta_rmse']:.6f}"])

    print(f"\nSaved to {out_dir}")

if __name__ == "__main__":
    main()
