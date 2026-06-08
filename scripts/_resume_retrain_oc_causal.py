#!/usr/bin/env python3
"""Resume retrain: object_centric + causality_aware with action embedding."""
import sys, json, time, torch, gc
from pathlib import Path
sys.path.insert(0, '/home/brucewu/my_robot_project')
torch.set_num_threads(8); torch.set_num_interop_threads(2)
from src.models.rig_world import RIGWorldModel
from src.models.losses import total_high_level_loss
from src.data.episode_loader import State16Dataset

MODELS = ['object_centric', 'causality_aware']
METADATA = 'data/sim/mppi_stage2c_state16/metadata/episodes.jsonl'
EPISODE_ROOT = 'data/sim/mppi_stage2c_state16/episodes'
OUT_DIR = Path('runs/retrain_action_embed_50ep')
MASK = [12, 13]; EPOCHS = 50; PATIENCE = 10; BS = 128; LR = 3e-4; NW = 8

for mt in MODELS:
    model = RIGWorldModel(model_type=mt, action_dim=2, history_len=6, num_tokens=6,
                          raw_token_dim=16, gru_hidden=256, d_model=128, head_hidden_dim=256)
    model.to('cpu')
    ds = State16Dataset(metadata_path=METADATA, episode_root=EPISODE_ROOT,
                        history_len=6, token_count=6, state_dim=16, action_dim=2,
                        split='train', seed=42)
    vds = State16Dataset(metadata_path=METADATA, episode_root=EPISODE_ROOT,
                         history_len=6, token_count=6, state_dim=16, action_dim=2,
                         split='val', seed=42)
    tl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=NW)
    vl = torch.utils.data.DataLoader(vds, batch_size=BS, num_workers=NW)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float('inf'); best_ep = 0; pc = 0
    md = OUT_DIR / mt / 'checkpoints'; md.mkdir(parents=True, exist_ok=True)
    lf = OUT_DIR / mt / 'train_log.jsonl'
    np_ = sum(p.numel() for p in model.parameters())
    print(f'\n=== Training {mt} ({np_:,} params) ===')
    t0 = time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); tl_ = 0.0
        for b in tl:
            h = b['history'].to('cpu'); a = b['action'].to('cpu')
            dt = b['dynamics_target'].to('cpu'); st = b['subgoal_target'].to('cpu')
            h = h.clone(); h[..., MASK] = 0.0
            opt.zero_grad()
            o = model(h, a)
            L = total_high_level_loss(o['pred_delta'], dt, o['pred_subgoal'], st)
            L['loss'].backward(); opt.step(); tl_ += L['loss'].item()
        tl_ /= len(tl)
        model.eval(); vl_ = 0.0
        with torch.no_grad():
            for b in vl:
                h = b['history'].to('cpu'); a = b['action'].to('cpu')
                dt = b['dynamics_target'].to('cpu'); st = b['subgoal_target'].to('cpu')
                h = h.clone(); h[..., MASK] = 0.0
                o = model(h, a)
                L = total_high_level_loss(o['pred_delta'], dt, o['pred_subgoal'], st)
                vl_ += L['loss'].item()
        vl_ /= len(vl)
        if vl_ < best_val:
            best_val = vl_; best_ep = ep; pc = 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'val_loss': vl_, 'architecture': 'action_embed_v1'},
                       md / 'best.pt')
        else:
            pc += 1
        st = ' BEST' if ep == best_ep else (' EARLY' if pc >= PATIENCE else '')
        print(f'  epoch {ep:2d}: train={tl_:.6f} val={vl_:.6f} {time.time()-t0:.0f}s{st}')
        with open(lf, 'a') as f:
            json.dump({'epoch': ep, 'train_loss': tl_, 'val_loss': vl_}, f)
            f.write('\n')
        if pc >= PATIENCE:
            print(f'  Early stopping at epoch {ep}')
            break
    print(f'  Best: epoch={best_ep} val_loss={best_val:.6f}')
    del model, opt, ds, vds; gc.collect()

print('\nALL DONE - retrain complete')
