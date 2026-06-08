#!/usr/bin/env python3
"""Retrain 3 models WITH mass/friction + action_embed, GPU-optimized for speed.

Key optimizations:
- torch.compile (reduce-overhead) 
- AMP (automatic mixed precision)
- Large batch size (512) — RTX 5090D 32GB
- pin_memory + prefetch for dataloader
- No MASKED_INDICES — keeps mass/friction

Output: runs/retrain_action_embed_mass_50ep/
"""

import sys, os, json, time, torch, numpy as np, gc
from pathlib import Path

sys.path.insert(0, '/home/brucewu/my_robot_project')

from src.models.rig_world import RIGWorldModel
from src.models.losses import total_high_level_loss
from src.data.episode_loader import State16Dataset

MODEL_TYPES = ['flat', 'object_centric', 'causality_aware']
METADATA = 'data/sim/mppi_stage2c_state16/metadata/episodes.jsonl'
EPISODE_ROOT = 'data/sim/mppi_stage2c_state16/episodes'
OUT_DIR = Path('runs/retrain_action_embed_mass_50ep')
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 512  # GPU has 32GB, this is safe
LR = 3e-4
NUM_WORKERS = 4

device = torch.device('cuda')


def train_model(model_type):
    model = RIGWorldModel(model_type=model_type, action_dim=2, history_len=6,
                          num_tokens=6, raw_token_dim=16, gru_hidden=256,
                          d_model=128, head_hidden_dim=256, use_action_embed=True)
    model.to(device)
    model = torch.compile(model, mode='reduce-overhead')
    
    dataset = State16Dataset(metadata_path=METADATA, episode_root=EPISODE_ROOT,
                             history_len=6, token_count=6, state_dim=16, action_dim=2,
                             split='train', seed=42)
    val_dataset = State16Dataset(metadata_path=METADATA, episode_root=EPISODE_ROOT,
                                 history_len=6, token_count=6, state_dim=16, action_dim=2,
                                 split='val', seed=42)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    model_dir = OUT_DIR / model_type / 'checkpoints'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / model_type / 'train_log.jsonl'
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n=== Training {model_type} ({n_params:,} params, +action_embed, +mass/friction) ===')
    t0 = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            history = batch['history'].to(device, non_blocking=True)
            action = batch['action'].to(device, non_blocking=True)
            dyn_target = batch['dynamics_target'].to(device, non_blocking=True)
            sub_target = batch['subgoal_target'].to(device, non_blocking=True)
            # NO masking — keep mass/friction
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(history, action)
                losses = total_high_level_loss(out['pred_delta'], dyn_target,
                                               out['pred_subgoal'], sub_target)
            scaler.scale(losses['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += losses['loss'].item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device, non_blocking=True)
                action = batch['action'].to(device, non_blocking=True)
                dyn_target = batch['dynamics_target'].to(device, non_blocking=True)
                sub_target = batch['subgoal_target'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(history, action)
                    losses = total_high_level_loss(out['pred_delta'], dyn_target,
                                                   out['pred_subgoal'], sub_target)
                val_loss += losses['loss'].item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model._orig_mod.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'architecture': 'action_embed_mass_v1'},
                       model_dir / 'best.pt')
        else:
            patience_counter += 1
        
        status = ' BEST' if epoch == best_epoch else ''
        elapsed = time.time() - t0
        print(f'  epoch {epoch:2d}: train={train_loss:.6f} val={val_loss:.6f} {elapsed:.0f}s{status}')
        
        with open(log_file, 'a') as f:
            json.dump({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}, f)
            f.write('\n')
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break
    
    print(f'  Best: epoch={best_epoch} val_loss={best_val_loss:.6f}')
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_loss


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        'architecture': 'action_embed_mass_v1',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'action_embed_dim': 64,
        'masked_indices': None,
        'mass_friction': 'included',
        'optimizations': ['torch.compile', 'AMP', 'pin_memory'],
        'device': str(device),
    }
    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    
    results = {}
    for mt in MODEL_TYPES:
        results[mt] = train_model(mt)
    
    print(f'\n===== Summary =====')
    for mt, val in results.items():
        print(f'  {mt}: val_loss={val:.6f}')
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('ALL DONE')


if __name__ == '__main__':
    main()
