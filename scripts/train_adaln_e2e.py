#!/usr/bin/env python3
"""
End-to-end training of AdaLN world model.

Trains encoder + AdaLN dynamics head jointly (not frozen encoder).
Saves to a new directory, does NOT touch existing checkpoints.

Usage:
    conda activate lerobot
    python3 scripts/train_adaln_e2e.py --epochs 100 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.models.rig_world import RIGWorldModel
from src.data.episode_loader import State16Dataset

METADATA = "data/sim/mppi_stage2c_state16/metadata/episodes.jsonl"
EPISODE_ROOT = "data/sim/mppi_stage2c_state16/episodes"


def train_e2e(
    checkpoint_path: str | None = None,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    output_dir: str | None = None,
    max_train: int | None = None,
    max_val: int = 10000,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create AdaLN model
    model = RIGWorldModel(model_type="flat", dynamics_head_type="adaln")

    # Optionally initialize encoder from old checkpoint
    if checkpoint_path:
        print(f"\nInitializing encoder from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        # Load encoder weights only
        encoder_keys = {k: v for k, v in state_dict.items() if "dynamics_head" not in k}
        missing, unexpected = model.load_state_dict(encoder_keys, strict=False)
        print(f"  Loaded encoder, skipped dynamics_head ({len([m for m in missing if 'dynamics_head' in m])} new head keys)")

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Load data
    print("\nLoading data...")
    train_ds = State16Dataset(
        metadata_path=METADATA, episode_root=EPISODE_ROOT,
        split="train", limit_samples=max_train,
    )
    val_ds = State16Dataset(
        metadata_path=METADATA, episode_root=EPISODE_ROOT,
        split="val", limit_samples=max_val,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # Optimizer with differential LR
    # Encoder gets lower LR, head gets full LR
    encoder_params = [p for n, p in model.named_parameters() if "dynamics_head" not in n]
    head_params = [p for n, p in model.named_parameters() if "dynamics_head" in n]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr, "weight_decay": weight_decay},
    ])

    # LR schedule: warmup + cosine
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output
    if output_dir is None:
        output_dir = f"runs/adaln_e2e_{time.strftime('%Y%m%d_%H%M%S')}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\nTraining AdaLN E2E for {epochs} epochs...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_delta = 0.0
        train_loss_subgoal = 0.0
        train_count = 0

        for batch in train_loader:
            states = batch["history"].to(device)
            actions = batch["action"].to(device)
            target_delta = batch["dynamics_target"].to(device)
            target_subgoal = batch["subgoal_target"].to(device)

            optimizer.zero_grad()
            out = model(states, actions)

            loss_delta = nn.MSELoss()(out["pred_delta"], target_delta)
            loss_subgoal = nn.MSELoss()(out["pred_subgoal"], target_subgoal)
            loss = loss_delta + 0.2 * loss_subgoal  # subgoal is auxiliary

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_delta += loss_delta.item() * len(states)
            train_loss_subgoal += loss_subgoal.item() * len(states)
            train_count += len(states)

        train_loss_delta /= train_count
        train_loss_subgoal /= train_count

        # Val
        model.eval()
        val_loss_delta = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["history"].to(device)
                actions = batch["action"].to(device)
                target_delta = batch["dynamics_target"].to(device)

                out = model(states, actions)
                loss_delta = nn.MSELoss()(out["pred_delta"], target_delta)
                val_loss_delta += loss_delta.item() * len(states)
                val_count += len(states)

        val_loss_delta /= val_count
        scheduler.step()

        # Save best
        if val_loss_delta < best_val_loss:
            best_val_loss = val_loss_delta
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss_delta,
                "head_type": "adaln",
                "e2e": True,
            }, out_path / "best.pt")

        history.append({
            "epoch": epoch,
            "train_loss_delta": train_loss_delta,
            "train_loss_subgoal": train_loss_subgoal,
            "val_loss_delta": val_loss_delta,
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_d={train_loss_delta:.6f} train_s={train_loss_subgoal:.6f} "
                  f"val_d={val_loss_delta:.6f} "
                  f"lr_enc={optimizer.param_groups[0]['lr']:.6f} lr_head={optimizer.param_groups[1]['lr']:.6f}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs - 1,
        "val_loss": val_loss_delta,
        "head_type": "adaln",
        "e2e": True,
    }, out_path / "final.pt")

    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ E2E training complete: best_val={best_val_loss:.6f}")
    print(f"   Saved to: {out_path}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/retrain_action_embed_mass_50ep/flat/checkpoints/best.pt",
                        help="Init encoder from this checkpoint (optional)")
    parser.add_argument("--no_init", action="store_true", help="Train from scratch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=10000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    train_e2e(
        checkpoint_path=None if args.no_init else args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_val=args.max_val,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
