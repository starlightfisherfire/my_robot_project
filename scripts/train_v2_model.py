#!/usr/bin/env python3
"""train_v2_model.py — Train v2 world model with profile ablation.

Usage:
    PYTHONPATH=. python scripts/train_v2_model.py \
        --dataset-dir data/sim/visual_state_v2_pilot \
        --split-file configs/splits/visual_state_v2_pilot.yaml \
        --split-name random_episode_split \
        --profile visual_object_relation_state_v2 \
        --epochs 10 \
        --out runs/v2_pilot/train_relation
"""

import argparse, json, os, sys, time, yaml
from pathlib import Path

import torch
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.v2_dataset import V2Dataset
from src.data.state_normalizer import StateNormalizer
from src.models.v2_encoder import V2WorldModel
from src.models.losses import total_high_level_loss


class SimpleNormalizer:
    """Simple z-score normalizer without valid_flag check."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, states):
        self.mean_ = states.mean(axis=0).astype(np.float32)
        std = states.std(axis=0).astype(np.float32)
        self.std_ = np.where(std < 1e-6, 1.0, std)
    def transform(self, states):
        return (states - self.mean_) / self.std_
    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({'mean': self.mean_.tolist(), 'std': self.std_.tolist()}, f)
    @classmethod
    def load(cls, path):
        import json
        obj = cls()
        with open(path) as f:
            d = json.load(f)
        obj.mean_ = np.array(d['mean'], dtype=np.float32)
        obj.std_ = np.array(d['std'], dtype=np.float32)
        return obj


def collate_fn(batch):
    history = torch.stack([b['history'] for b in batch])
    action = torch.stack([b['action'] for b in batch])
    dyn_target = torch.stack([b['dynamics_target'] for b in batch])
    sub_target = torch.stack([b['subgoal_target'] for b in batch])
    return history, action, dyn_target, sub_target


def get_profile_features(profile_name, schema_path):
    """Get feature list for a profile from schema."""
    with open(schema_path) as f:
        profiles = yaml.safe_load(f)

    profile = profiles.get('profiles', {}).get(profile_name, {})
    includes = profile.get('includes', [])

    # Parse semicolon-separated features
    features = []
    for entry in includes:
        # Check if it's a profile reference
        if entry in profiles.get('profiles', {}):
            # Recursively expand
            ref_features = get_profile_features(entry, schema_path)
            features.extend(ref_features)
        else:
            for part in entry.replace(',', ';').split(';'):
                part = part.strip()
                if part and part not in profiles.get('profiles', {}):
                    features.append(part)

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--split-file', default=None)
    parser.add_argument('--split-name', default='random_episode_split')
    parser.add_argument('--profile', default='visual_object_relation_state_v2',
                        help='Profile name from state_profiles.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    # Load profile features
    schema_path = REPO / 'configs/state_schema/state_profiles.yaml'
    profile_features = get_profile_features(args.profile, str(schema_path))
    print(f'Profile {args.profile}: {len(profile_features)} features')

    # Load split
    episode_ids = None
    if args.split_file:
        with open(args.split_file) as f:
            split_cfg = yaml.safe_load(f)
        parts = args.split_name.split('/')
        split_key = parts[0]
        sub_key = parts[1] if len(parts) > 1 else 'train'
        episode_ids = split_cfg['splits'][split_key][sub_key]
        print(f'Split: {args.split_name} → {len(episode_ids)} episodes')

    ds_dir = Path(args.dataset_dir)
    meta = str(ds_dir / 'metadata' / 'episodes.jsonl')
    ep_root = str(ds_dir / 'episodes')

    train_ds = V2Dataset(
        metadata_path=meta, episode_root=ep_root,
        profile_features=profile_features, split='train',
        split_episode_ids=episode_ids, seed=42,
    )
    val_ds = V2Dataset(
        metadata_path=meta, episode_root=ep_root,
        profile_features=profile_features, split='val',
        split_episode_ids=episode_ids, seed=42,
    )

    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')
    feature_dim = train_ds.feature_dim
    print(f'Feature dim: {feature_dim}')

    # Model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = V2WorldModel(feature_dim=feature_dim, gru_hidden=256, head_hidden_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Normalizer (simple z-score, no valid_flag check for v2)
    normalizer = SimpleNormalizer()
    all_train_states = []
    for i in range(min(len(train_ds), 2000)):
        sample = train_ds[i]
        all_train_states.append(sample['history'].numpy())
    all_train_states = np.array(all_train_states).reshape(-1, feature_dim)
    normalizer.fit(all_train_states)

    # Training
    epochs = 2 if args.smoke else args.epochs
    batch_size = min(args.batch_size, max(2, len(train_ds) // 2))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    best_val = float('inf')
    log_entries = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for history, action, dyn_target, sub_target in train_loader:
            history = history.to(device)
            action = action.to(device)
            dyn_target = dyn_target.to(device)
            sub_target = sub_target.to(device)

            # Normalize
            h_np = history.cpu().numpy()
            B, H, D = h_np.shape
            h_np = normalizer.transform(h_np.reshape(-1, D)).reshape(B, H, D)
            history = torch.from_numpy(h_np).to(device)

            out = model(history, action)
            losses = total_high_level_loss(out['pred_delta'], dyn_target, out['pred_subgoal'], sub_target)
            optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += losses['loss'].item() * history.size(0)
            train_n += history.size(0)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for history, action, dyn_target, sub_target in val_loader:
                history = history.to(device)
                action = action.to(device)
                dyn_target = dyn_target.to(device)
                sub_target = sub_target.to(device)
                h_np = history.cpu().numpy()
                B, H, D = h_np.shape
                h_np = normalizer.transform(h_np.reshape(-1, D)).reshape(B, H, D)
                history = torch.from_numpy(h_np).to(device)
                out = model(history, action)
                losses = total_high_level_loss(out['pred_delta'], dyn_target, out['pred_subgoal'], sub_target)
                val_loss_sum += losses['loss'].item() * history.size(0)
                val_n += history.size(0)

        avg_train = train_loss_sum / max(train_n, 1)
        avg_val = val_loss_sum / max(val_n, 1)
        log_entries.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val})

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': avg_val},
                       ckpt_dir / 'best.pt')

        print(f'[Epoch {epoch}/{epochs}] train={avg_train:.6f} val={avg_val:.6f}')

    # Save
    normalizer.save(str(out_dir / 'normalizer.json'))
    with open(out_dir / 'train_log.jsonl', 'w') as f:
        for e in log_entries:
            f.write(json.dumps(e) + '\n')

    # Save profile info
    with open(out_dir / 'profile_info.json', 'w') as f:
        json.dump({'profile': args.profile, 'feature_dim': feature_dim,
                   'features': profile_features, 'n_params': n_params}, f, indent=2)

    print(f'\nDone: best_val={best_val:.6f}')
    print(f'Checkpoint: {ckpt_dir / "best.pt"}')


if __name__ == '__main__':
    main()
