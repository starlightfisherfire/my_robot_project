#!/usr/bin/env python3
"""Train ActionInjectModel (3 encoder types) for 50 epochs.

Action is injected at input level (EE token features 16,17)
instead of latent-space concatenation.

Uses 32 CPU cores for data loading. GPU reserved for GPT-2.

Output: runs/retrain_action_inject_50ep/
"""

import sys, os, json, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ['OMP_NUM_THREADS'] = '4'

from src.models.action_inject_model import ActionInjectModel
from src.models.losses import total_high_level_loss
from src.data.episode_loader import State16Dataset

MODEL_TYPES = ['flat', 'object_centric', 'causality_aware']
METADATA = 'data/sim/mppi_stage2c_state16/metadata/episodes.jsonl'
EPISODE_ROOT = 'data/sim/mppi_stage2c_state16/episodes'
OUT_DIR = REPO / 'runs' / 'retrain_action_inject_50ep'
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 128
LR = 3e-4
NUM_WORKERS = 8  # 32 cores / 4 parallel models ≈ 8 workers each


def train_model(model_type, device='cpu'):
    model = ActionInjectModel(
        encoder_type=model_type,
        action_dim=2,
        history_len=6,
        num_tokens=6,
        raw_token_dim=16,
        gru_hidden=256,
        d_model=128,
        head_hidden_dim=256,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n=== Training {model_type} ({n_params/1e6:.2f}M params) ===')
    print(f'    augmented_token_dim={model.augmented_token_dim}')

    # Dataset
    dataset = State16Dataset(
        metadata_path=REPO / METADATA,
        episode_root=REPO / EPISODE_ROOT,
        history_len=6, token_count=6, state_dim=16, action_dim=2,
        split='train', seed=42,
    )
    val_dataset = State16Dataset(
        metadata_path=REPO / METADATA,
        episode_root=REPO / EPISODE_ROOT,
        history_len=6, token_count=6, state_dim=16, action_dim=2,
        split='val', seed=42,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    model_dir = OUT_DIR / model_type / 'checkpoints'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / model_type / 'train_log.jsonl'

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            history = batch['history'].to(device)
            action = batch['action'].to(device)
            dyn_target = batch['dynamics_target'].to(device)
            sub_target = batch['subgoal_target'].to(device)

            optimizer.zero_grad()
            out = model(history, action)
            losses = total_high_level_loss(
                out['pred_delta'], dyn_target,
                out['pred_subgoal'], sub_target,
            )
            losses['loss'].backward()
            optimizer.step()
            train_loss += losses['loss'].item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                action = batch['action'].to(device)
                dyn_target = batch['dynamics_target'].to(device)
                sub_target = batch['subgoal_target'].to(device)
                out = model(history, action)
                losses = total_high_level_loss(
                    out['pred_delta'], dyn_target,
                    out['pred_subgoal'], sub_target,
                )
                val_loss += losses['loss'].item()
                n_val += 1

        val_loss /= max(n_val, 1)

        # ── Early stopping ──
        status = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_type': 'action_inject',
                'encoder_type': model_type,
            }, model_dir / 'best.pt')
            status = ' ← BEST'
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        print(f'  epoch {epoch:2d}: train={train_loss:.6f} val={val_loss:.6f} '
              f'{elapsed:.0f}s{status}')

        with open(log_file, 'a') as f:
            json.dump({'epoch': epoch, 'train_loss': train_loss,
                       'val_loss': val_loss, 'timestamp': time.time()}, f)
            f.write('\n')

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break

    elapsed_total = time.time() - t0
    print(f'  Done: best_epoch={best_epoch} val_loss={best_val_loss:.6f} '
          f'total={elapsed_total/3600:.1f}h')
    del model, optimizer
    return best_val_loss


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cpu'  # GPU busy with GPT-2

    print(f'ActionInjectModel training')
    print(f'  Output: {OUT_DIR}')
    print(f'  Device: {device}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Batch: {BATCH_SIZE}')
    print(f'  Workers: {NUM_WORKERS}')
    print(f'  Models: {MODEL_TYPES}')

    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump({
            'model_class': 'ActionInjectModel',
            'encoder_types': MODEL_TYPES,
            'epochs': EPOCHS,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'num_workers': NUM_WORKERS,
            'action_inject': 'input_level (EE token features 16,17)',
        }, f, indent=2)

    results = {}
    for mt in MODEL_TYPES:
        results[mt] = train_model(mt, device)
        import gc; gc.collect()

    print(f'\n{"="*60}')
    print('Training summary:')
    for mt, val in results.items():
        print(f'  {mt}: val_loss={val:.6f}')

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
