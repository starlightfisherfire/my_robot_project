#!/usr/bin/env python3
"""
Train AdaLN dynamics head vs concat baseline.

Freezes encoder, re-trains only dynamics head on existing data.

Usage:
    conda activate lerobot
    python3 scripts/train_adaln_head.py --head_type adaln --epochs 50
    python3 scripts/train_adaln_head.py --head_type concat --epochs 50
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


def train_head(
    head_type: str,
    checkpoint_path: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    freeze_encoder: bool = True,
    output_dir: str | None = None,
    max_train: int = 50000,
    max_val: int = 5000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = RIGWorldModel(model_type="flat", dynamics_head_type=head_type)

    # Load encoder weights only (dynamics_head has different architecture)
    state_dict = ckpt["model_state_dict"]
    encoder_keys = {k: v for k, v in state_dict.items() if "dynamics_head" not in k}
    missing, unexpected = model.load_state_dict(encoder_keys, strict=False)
    print(f"  Loaded encoder weights, skipped dynamics_head")
    if missing:
        print(f"  Missing keys (expected): {[k for k in missing if 'dynamics_head' in k]}")
    model.to(device)

    # Freeze encoder
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "dynamics_head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen encoder: {trainable:,} / {total:,} params trainable")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  All params trainable: {trainable:,}")

    # Load data
    print("\nLoading data...")
    train_ds = State16Dataset(
        metadata_path=METADATA, episode_root=EPISODE_ROOT, split="train",
        limit_samples=max_train,
    )
    val_ds = State16Dataset(
        metadata_path=METADATA, episode_root=EPISODE_ROOT, split="val",
        limit_samples=max_val,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Output
    if output_dir is None:
        output_dir = f"runs/adaln_train_{head_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\nTraining {head_type} head for {epochs} epochs...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            states = batch["history"].to(device)
            actions = batch["action"].to(device)
            targets = batch["dynamics_target"].to(device)

            optimizer.zero_grad()
            out = model(states, actions)
            loss = nn.MSELoss()(out["pred_delta"], targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(states)
            train_count += len(states)

        train_loss /= train_count

        # Val
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["history"].to(device)
                actions = batch["action"].to(device)
                targets = batch["dynamics_target"].to(device)

                out = model(states, actions)
                loss = nn.MSELoss()(out["pred_delta"], targets)
                val_loss += loss.item() * len(states)
                val_count += len(states)

        val_loss /= val_count
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "head_type": head_type,
            }, out_path / "best.pt")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs - 1, "val_loss": val_loss, "head_type": head_type,
    }, out_path / "final.pt")

    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete: best_val={best_val_loss:.6f}")
    print(f"   Saved to: {out_path}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_type", choices=["concat", "adaln"], required=True)
    parser.add_argument("--checkpoint", default="runs/retrain_action_embed_mass_50ep/flat/checkpoints/best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_train", type=int, default=50000)
    parser.add_argument("--max_val", type=int, default=5000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    train_head(
        head_type=args.head_type,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_val=args.max_val,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
