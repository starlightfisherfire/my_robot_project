#!/usr/bin/env python3
"""eval_high_level.py — Evaluate trained Paper 1 model (one-step only)."""

import argparse, csv, json, sys
from pathlib import Path

import torch
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_loader import State16Dataset
from src.data.state_normalizer import StateNormalizer
from src.models.rig_world import RIGWorldModel
from src.models.losses import dynamics_loss, subgoal_loss


def collate_fn(batch):
    history = torch.stack([b["history"] for b in batch])
    action = torch.stack([b["action"] for b in batch])
    dyn_target = torch.stack([b["dynamics_target"] for b in batch])
    sub_target = torch.stack([b["subgoal_target"] for b in batch])
    return history, action, dyn_target, sub_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["flat", "object_centric", "causality_aware"])
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dc, mc = cfg["data"], cfg["models"]

    # Load val dataset
    val_ds = State16Dataset(
        metadata_path=dc["metadata"],
        episode_root=f"{dc['root']}/episodes",
        history_len=dc["history_len"],
        token_count=dc["token_count"],
        state_dim=dc["state_dim"],
        split="val",
        train_ratio=dc["train_ratio_if_no_split"],
        val_ratio=dc["val_ratio_if_no_split"],
        seed=cfg["experiment"]["seed"],
    )

    # Normalizer
    nc = cfg["normalizer"]
    normalizer = None
    if nc["enabled"]:
        normalizer = StateNormalizer.load(nc["save_path"])

    # Model
    model = RIGWorldModel(
        model_type=args.model,
        history_len=dc["history_len"],
        num_tokens=dc["token_count"],
        raw_token_dim=dc["state_dim"],
        action_dim=2,
        gru_hidden=mc["gru_hidden"],
        d_model=mc["d_model"],
        head_hidden_dim=mc["z_dim"],
        frame_embed_dim=128,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate
    batch_size = min(128, len(val_ds))
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    all_dyn_mse = []
    all_sub_mse = []
    pred_samples = []

    with torch.no_grad():
        for history, action, dyn_target, sub_target in loader:
            history = history.to(device)
            action = action.to(device)
            dyn_target = dyn_target.to(device)
            sub_target = sub_target.to(device)

            if normalizer is not None:
                history_np = history.cpu().numpy()
                B, H, N, D = history_np.shape
                history_np = normalizer.transform(history_np.reshape(-1, D)).reshape(B, H, N, D)
                history = torch.from_numpy(history_np).to(device)

            out = model(history, action)
            pred_delta = out["pred_delta"]
            pred_subgoal = out["pred_subgoal"]

            dyn_l = dynamics_loss(pred_delta, dyn_target)
            sub_l = subgoal_loss(pred_subgoal, sub_target)

            all_dyn_mse.append(dyn_l.item())
            all_sub_mse.append(sub_l.item())

            for i in range(pred_delta.size(0)):
                pred_samples.append({
                    "pred_dx": pred_delta[i, 0].item(),
                    "pred_dy": pred_delta[i, 1].item(),
                    "pred_dtheta": pred_delta[i, 2].item(),
                    "target_dx": dyn_target[i, 0].item(),
                    "target_dy": dyn_target[i, 1].item(),
                    "target_dtheta": dyn_target[i, 2].item(),
                    "pred_sgx": pred_subgoal[i, 0].item(),
                    "pred_sgy": pred_subgoal[i, 1].item(),
                    "pred_sgtheta": pred_subgoal[i, 2].item(),
                    "target_sgx": sub_target[i, 0].item(),
                    "target_sgy": sub_target[i, 1].item(),
                    "target_sgtheta": sub_target[i, 2].item(),
                })

    dyn_mse = np.mean(all_dyn_mse)
    dyn_rmse = np.sqrt(dyn_mse)
    sub_mse = np.mean(all_sub_mse)
    sub_rmse = np.sqrt(sub_mse)

    # Save metrics
    eval_dir = Path(cfg["outputs"]["run_dir"]) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / f"{args.model}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["dynamics_mse", dyn_mse])
        w.writerow(["dynamics_rmse", dyn_rmse])
        w.writerow(["subgoal_mse", sub_mse])
        w.writerow(["subgoal_rmse", sub_rmse])

    samples_path = eval_dir / f"{args.model}_pred_samples.csv"
    with open(samples_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_samples[0].keys())
        w.writeheader()
        w.writerows(pred_samples)

    print(f"[EVAL] {args.model}: dyn_rmse={dyn_rmse:.6f}, sub_rmse={sub_rmse:.6f}")
    print(f"[EVAL] Metrics: {metrics_path}")
    print(f"[EVAL] Samples: {samples_path}")


if __name__ == "__main__":
    main()
