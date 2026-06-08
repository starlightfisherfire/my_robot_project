#!/usr/bin/env python3
"""Retrain causality_aware 50 epochs without mass/friction. Fast version."""
import sys, os, json, time, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/brucewu/my_robot_project')

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

from src.models.rig_world import RIGWorldModel
from src.models.losses import total_high_level_loss
from src.data.episode_loader import State16Dataset

MODEL_TYPE = 'causality_aware'
METADATA = 'data/sim/mppi_stage2c_state16/metadata/episodes.jsonl'
EPISODE_ROOT = 'data/sim/mppi_stage2c_state16/episodes'
OUT_DIR = Path('runs/retrain_nomass_50ep')
MASKED_INDICES = [12, 13]
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 128
LR = 3e-4
NUM_WORKERS = 8

device = 'cpu'
model = RIGWorldModel(model_type=MODEL_TYPE, action_dim=2, history_len=6,
                      num_tokens=6, raw_token_dim=16, gru_hidden=256,
                      d_model=128, head_hidden_dim=256)
model.to(device)

dataset = State16Dataset(metadata_path=METADATA, episode_root=EPISODE_ROOT,
                         history_len=6, token_count=6, state_dim=16, action_dim=2,
                         split='train', seed=42)
val_dataset = State16Dataset(metadata_path=METADATA, episode_root=EPISODE_ROOT,
                             history_len=6, token_count=6, state_dim=16, action_dim=2,
                             split='val', seed=42)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0

model_dir = OUT_DIR / MODEL_TYPE / 'checkpoints'
model_dir.mkdir(parents=True, exist_ok=True)
log_file = OUT_DIR / MODEL_TYPE / 'train_log_v2.jsonl'

n_params = sum(p.numel() for p in model.parameters())
print(f'=== Training {MODEL_TYPE} ({n_params/1e6:.2f}M params) ===')
print(f'  threads={torch.get_num_threads()} workers={NUM_WORKERS}')
print(f'  train={len(dataset)} val={len(val_dataset)} batches={len(train_loader)}')
print()

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        history = batch['history'].to(device)
        action = batch['action'].to(device)
        dyn_target = batch['dynamics_target'].to(device)
        sub_target = batch['subgoal_target'].to(device)
        history = history.clone()
        history[..., MASKED_INDICES] = 0.0
        optimizer.zero_grad()
        out = model(history, action)
        losses = total_high_level_loss(out['pred_delta'], dyn_target,
                                       out['pred_subgoal'], sub_target)
        losses['loss'].backward()
        optimizer.step()
        train_loss += losses['loss'].item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            history = batch['history'].to(device)
            action = batch['action'].to(device)
            dyn_target = batch['dynamics_target'].to(device)
            sub_target = batch['subgoal_target'].to(device)
            history = history.clone()
            history[..., MASKED_INDICES] = 0.0
            out = model(history, action)
            losses = total_high_level_loss(out['pred_delta'], dyn_target,
                                           out['pred_subgoal'], sub_target)
            val_loss += losses['loss'].item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss, 'masked_indices': MASKED_INDICES},
                   model_dir / 'best.pt')
    else:
        patience_counter += 1

    elapsed = time.time() - t0
    status = ' ← BEST' if epoch == best_epoch else ''
    print(f'  epoch {epoch:2d}: train={train_loss:.6f} val={val_loss:.6f} total={elapsed:.0f}s{status}')

    with open(log_file, 'a') as f:
        json.dump({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                   'timestamp': time.time()}, f)
        f.write('\n')

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f'  Early stopping at epoch {epoch}')
        break

print(f'\n  Best: epoch={best_epoch} val_loss={best_val_loss:.6f}')
print(f'  Total: {time.time()-t0:.0f}s')
