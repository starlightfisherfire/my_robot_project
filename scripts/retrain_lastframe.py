#!/usr/bin/env python3
"""Train 3 models with LAST-FRAME-ONLY encoder (no GRU).

Hypothesis: GRU in encoder encodes velocity/trend into z, making action redundant.
Fix: use only the last frame's representation as z. No temporal info in z.

Key changes vs retrain_action_embed_mass.py:
- Encoder forward: last frame only, no GRU
- Add projection layer (d_model -> gru_hidden) to replace GRU
- Keep mass/friction (no masking)
- action_embed=True
- 30 epochs, early stopping patience=7

Output: runs/retrain_lastframe_30ep/
"""

import sys, os, json, time, torch, numpy as np, gc
from pathlib import Path

sys.path.insert(0, '/home/brucewu/my_robot_project')

from src.models.rig_world import RIGWorldModel
from src.models.encoders import FlatEncoder, ObjectCentricEncoder, CausalityAwareEncoder
from src.models.losses import total_high_level_loss
from src.data.episode_loader import State16Dataset

MODEL_TYPES = ['flat', 'object_centric', 'causality_aware']
METADATA = 'data/sim/mppi_stage2c_state16/metadata/episodes.jsonl'
EPISODE_ROOT = 'data/sim/mppi_stage2c_state16/episodes'
OUT_DIR = Path('runs/retrain_lastframe_30ep')
EPOCHS = 30
EARLY_STOP_PATIENCE = 7
BATCH_SIZE = 512
LR = 3e-4
NUM_WORKERS = 4
DEVICE = torch.device('cuda')


def make_lastframe_encoder(encoder):
    """Replace encoder.forward with last-frame-only (no GRU).
    Adds encoder._lf_proj as a registered submodule so it receives gradients.
    For CausalityAwareEncoder, patches the backbone recursively.
    """
    if isinstance(encoder, CausalityAwareEncoder):
        make_lastframe_encoder(encoder.object_backbone)
        return

    if isinstance(encoder, FlatEncoder):
        frame_dim = encoder.gru.input_size  # = frame_embed_dim
        gru_hidden = encoder.gru.hidden_size
    elif isinstance(encoder, ObjectCentricEncoder):
        frame_dim = encoder.d_model
        gru_hidden = encoder.gru_hidden
    else:
        raise ValueError(f"Unknown encoder: {type(encoder)}")

    proj = torch.nn.Linear(frame_dim, gru_hidden)
    torch.nn.init.xavier_uniform_(proj.weight)
    torch.nn.init.zeros_(proj.bias)
    encoder.add_module('_lf_proj', proj)

    def lastframe_forward(x, mask=None, return_slots=False):
        if x.dim() != 4:
            raise ValueError(f"Expected [B,H,N,D], got {x.shape}")
        b, h, n, d = x.shape
        x_last = x[:, -1, :, :]  # [B, N, D]

        if isinstance(encoder, FlatEncoder):
            x_flat = x_last.reshape(b, n * d)
            frame_emb = encoder.frame_mlp(x_flat)
            z = encoder._lf_proj(frame_emb)
            return z

        # ObjectCentricEncoder path (also used by CA's backbone)
        if mask is not None:
            valid_flat = mask[:, -1, :].bool()
        else:
            valid_flat = (x_last[:, :, encoder.valid_flag_index] > 0.5)

        token_emb = encoder.token_mlp(x_last)
        n_tok = min(n, len(encoder.token_type_ids))
        type_ids = encoder.token_type_ids[:n_tok].unsqueeze(0).expand(b, n_tok)
        token_emb[:, :n_tok] = token_emb[:, :n_tok] + encoder.type_embedding(type_ids)

        key_padding_mask = ~valid_flat
        token_out = encoder.transformer(token_emb, src_key_padding_mask=key_padding_mask)

        valid_float = valid_flat.float().unsqueeze(-1)
        pooled = (token_out * valid_float).sum(dim=1)
        denom = valid_float.sum(dim=1).clamp(min=1.0)
        frame_emb = pooled / denom

        z = encoder._lf_proj(frame_emb)
        return z

    encoder.forward = lastframe_forward


def train_model(model_type):
    model = RIGWorldModel(
        model_type=model_type, action_dim=2, history_len=6,
        num_tokens=6, raw_token_dim=16, gru_hidden=256,
        d_model=128, head_hidden_dim=256, use_action_embed=True
    )

    # Patch encoder BEFORE compiling
    make_lastframe_encoder(model.encoder)
    model.to(DEVICE)
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
    print(f'\n=== Training {model_type} ({n_params:,} params, lastframe+action_embed+mass) ===')
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            history = batch['history'].to(DEVICE, non_blocking=True)
            action = batch['action'].to(DEVICE, non_blocking=True)
            dyn_target = batch['dynamics_target'].to(DEVICE, non_blocking=True)
            sub_target = batch['subgoal_target'].to(DEVICE, non_blocking=True)

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
                history = batch['history'].to(DEVICE, non_blocking=True)
                action = batch['action'].to(DEVICE, non_blocking=True)
                dyn_target = batch['dynamics_target'].to(DEVICE, non_blocking=True)
                sub_target = batch['subgoal_target'].to(DEVICE, non_blocking=True)
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
            state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch, 'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'architecture': 'lastframe_action_embed_mass',
            }, model_dir / 'best.pt')
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
        'architecture': 'lastframe_action_embed_mass',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'action_embed_dim': 64,
        'masked_indices': None,
        'mass_friction': 'included',
        'encoder': 'last_frame_only (no GRU)',
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'optimizations': ['torch.compile', 'AMP', 'pin_memory'],
        'device': str(DEVICE),
    }
    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'Encoder: LAST FRAME ONLY (no GRU) + action_embed + mass/friction')

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
