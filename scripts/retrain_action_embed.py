#!/usr/bin/env python3
"""Retrain all 3 models with action embedding fix (50 epochs, no mass/friction)."""
import sys, os, json, time, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/brucewu/my_robot_project')

torch.set_num_threads(16)
torch.set_num_interop_threads(2)

from src.models.rig_world import RIGWorldModel
from src.models.losses import total_high_level_loss
from src.data.episode_loader import State16Dataset

MODEL_TYPES = ['flat', 'object_centric', 'causality_aware']
METADATA = 'data/sim/mppi_stage2c_state16/metadata/episodes.jsonl'
EPISODE_ROOT = 'data/sim/mppi_stage2c_state16/episodes'
OUT_DIR = Path('runs/retrain_action_embed_50ep')
MASKED_INDICES = [12, 13]
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 128
LR = 3e-4
NUM_WORKERS = 8

def train_model(model_type, device='cpu'):
    model = RIGWorldModel(model_type=model_type, action_dim=2, history_len=6,
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
    
    model_dir = OUT_DIR / model_type / 'checkpoints'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / model_type / 'train_log.jsonl'
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n=== Training {model_type} ({n_params:,} params, +action_embed) ===')
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
                        'val_loss': val_loss, 'architecture': 'action_embed_v1'},
                       model_dir / 'best.pt')
        else:
            patience_counter += 1
        
        status = ' ← BEST' if epoch == best_epoch else ''
        elapsed = time.time() - t0
        print(f'  epoch {epoch:2d}: train={train_loss:.6f} val={val_loss:.6f} {elapsed:.0f}s{status}')
        
        with open(log_file, 'a') as f:
            json.dump({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}, f)
            f.write('\n')
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break
    
    print(f'  Best: epoch={best_epoch} val_loss={best_val_loss:.6f}')
    del model, optimizer
    return best_val_loss


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump({'architecture': 'action_embed_v1', 'epochs': EPOCHS,
                   'action_embed_dim': 64, 'masked_indices': MASKED_INDICES}, f, indent=2)
    
    results = {}
    for mt in MODEL_TYPES:
        results[mt] = train_model(mt, 'cpu')
        import gc; gc.collect()
    
    print(f'\n===== Summary =====')
    for mt, val in results.items():
        print(f'  {mt}: val_loss={val:.6f}')
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
