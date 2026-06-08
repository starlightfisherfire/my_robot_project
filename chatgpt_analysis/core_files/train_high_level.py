#!/usr/bin/env python3
"""train_high_level.py — Train Paper 1 high-level representation models."""

import argparse, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_loader import State16Dataset
from src.data.state_normalizer import StateNormalizer
from src.models.rig_world import RIGWorldModel
from src.models.losses import total_high_level_loss


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset-dir", default=None, help="Override data.root")
    parser.add_argument("--split-file", default=None, help="YAML with episode_id splits")
    parser.add_argument("--split-name", default=None, help="Split name from split-file (e.g. random_episode_split/train)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--out", default=None, help="Override outputs.run_dir")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.dataset_dir:
        cfg.setdefault("data", {})["root"] = args.dataset_dir
        cfg["data"]["metadata"] = f"{args.dataset_dir}/metadata/episodes.jsonl"
    if args.out:
        cfg.setdefault("outputs", {})["run_dir"] = args.out

    set_seed(cfg["experiment"]["seed"])

    # Device
    device_str = cfg["train"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[INFO] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # Paths
    run_dir = Path(cfg["outputs"]["run_dir"]) / args.model
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"

    # Data
    dc = cfg["data"]
    nc = cfg["normalizer"]
    tc = cfg["train"]
    mc = cfg["models"]

    # Load split file if provided
    split_episode_ids = None
    split_name_train = "train"
    split_name_val = "val"
    if args.split_file and args.split_name:
        with open(args.split_file) as f:
            split_cfg = yaml.safe_load(f)
        # split_name format: "random_episode_split" or "random_episode_split/train"
        parts = args.split_name.split("/")
        split_key = parts[0]
        if split_key in split_cfg.get("splits", {}):
            sp = split_cfg["splits"][split_key]
            if len(parts) > 1:
                # e.g. random_episode_split/train
                split_episode_ids = sp.get(parts[1], [])
                split_name_train = parts[1]
            else:
                # Use all sub-splits
                split_episode_ids = None  # will use train/val from config
        print(f"[INFO] Split file: {args.split_file}, key: {args.split_name}")

    train_ds = State16Dataset(
        metadata_path=dc["metadata"],
        episode_root=f"{dc['root']}/episodes",
        history_len=dc["history_len"],
        token_count=dc["token_count"],
        state_dim=dc["state_dim"],
        split="train",
        train_ratio=dc["train_ratio_if_no_split"],
        val_ratio=dc["val_ratio_if_no_split"],
        seed=cfg["experiment"]["seed"],
        split_episode_ids=split_episode_ids if args.split_name and "/" in args.split_name else None,
    )
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

    import numpy as np
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Normalizer
    if nc["enabled"]:
        normalizer = StateNormalizer()
        # Fit on train only: collect all history states from train dataset
        all_train_states = []
        for i in range(min(len(train_ds), 2000)):
            sample = train_ds[i]
            all_train_states.append(sample["history"].numpy())
        all_train_states = np.array(all_train_states).reshape(-1, train_ds.state_dim)
        # Use only valid tokens
        valid = all_train_states[:, normalizer.valid_flag_index] > 0.5
        if valid.sum() > 0:
            normalizer.fit(all_train_states)
            print(f"[INFO] Normalizer fitted on train tokens")
        else:
            print("[WARN] No valid train tokens for normalizer, skipping fit")
        
        # Save
        save_path = Path(nc["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(str(save_path))
    else:
        normalizer = None

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model {args.model}: {n_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )

    # Dataloaders
    def make_loader(ds, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=tc["num_workers"],
        )

    epochs = args.epochs if args.epochs else (tc["epochs_smoke"] if args.smoke else tc["epochs_poc"])
    batch_size = min(tc["batch_size"], len(train_ds) // 2)
    if batch_size < 2:
        batch_size = 2

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    best_val_loss = float("inf")
    log_entries = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for history, action, dyn_target, sub_target in train_loader:
            history = history.to(device)
            action = action.to(device)
            dyn_target = dyn_target.to(device)
            sub_target = sub_target.to(device)

            # Normalize if enabled
            if normalizer is not None:
                history_np = history.cpu().numpy()
                B, H, N, D = history_np.shape
                history_np = normalizer.transform(history_np.reshape(-1, D)).reshape(B, H, N, D)
                history = torch.from_numpy(history_np).to(device)

            out = model(history, action)
            losses = total_high_level_loss(
                out["pred_delta"], dyn_target,
                out["pred_subgoal"], sub_target,
                lambda_dyn=tc["w_dynamics"],
                lambda_subgoal=tc["w_subgoal"],
            )

            optimizer.zero_grad()
            losses["loss"].backward()
            if tc["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"])
            optimizer.step()

            train_loss_sum += losses["loss"].item() * history.size(0)
            train_n += history.size(0)

        avg_train_loss = train_loss_sum / max(train_n, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for history, action, dyn_target, sub_target in val_loader:
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
                losses = total_high_level_loss(
                    out["pred_delta"], dyn_target,
                    out["pred_subgoal"], sub_target,
                    lambda_dyn=tc["w_dynamics"],
                    lambda_subgoal=tc["w_subgoal"],
                )
                val_loss_sum += losses["loss"].item() * history.size(0)
                val_n += history.size(0)

        avg_val_loss = val_loss_sum / max(val_n, 1)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "timestamp": time.time(),
        }
        log_entries.append(log_entry)

        # Check for NaN
        if not torch.isfinite(torch.tensor(avg_train_loss)):
            print(f"[ERROR] NaN train_loss at epoch {epoch}")
            break

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, ckpt_dir / "best.pt")

        print(f"[Epoch {epoch:3d}/{epochs}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")

    # Save final log
    with open(log_path, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[DONE] Best val_loss={best_val_loss:.6f}, log saved to {log_path}")
    print(f"[DONE] Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
