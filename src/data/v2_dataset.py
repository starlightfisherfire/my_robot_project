# src/data/v2_dataset.py
"""Dataset loader for visual_structured_state_v2 episodes."""

from __future__ import annotations
import json, random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _wrap_pose_delta(delta):
    out = delta.copy()
    out[..., 2] = _wrap_to_pi(out[..., 2])
    return out


class V2Dataset(Dataset):
    """Load v2 episodes and produce supervised samples for dynamics learning."""

    def __init__(
        self,
        metadata_path: str,
        episode_root: str,
        history_len: int = 6,
        profile_features: Optional[list[str]] = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        limit_samples: Optional[int] = None,
        seed: int = 0,
        split_episode_ids: Optional[list[str]] = None,
    ):
        self.history_len = history_len
        self.profile_features = profile_features

        meta_path = Path(metadata_path)
        with meta_path.open() as f:
            all_episodes = [json.loads(line) for line in f if line.strip()]

        # Split
        if split_episode_ids is not None:
            id_set = set(split_episode_ids)
            episodes = [ep for ep in all_episodes if ep['episode_id'] in id_set]
            # Apply train/val/test ratio within the given episode set
            rng = random.Random(seed)
            rng.shuffle(episodes)
            n = len(episodes)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            if split == "train":
                episodes = episodes[:n_train]
            elif split == "val":
                episodes = episodes[n_train:n_train + n_val]
            elif split == "test":
                episodes = episodes[n_train + n_val:]
            else:
                episodes = []
        else:
            rng = random.Random(seed)
            rng.shuffle(all_episodes)
            n = len(all_episodes)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio)) if n > 1 else 1
            if split == "train":
                episodes = all_episodes[:n_train]
            elif split == "val":
                episodes = all_episodes[n_train:n_train + n_val]
            else:
                episodes = []

        self.episodes = episodes
        self.episode_root = Path(episode_root)

        # Build sample index (use metadata for T, don't load npz)
        self.samples = []
        for ep in episodes:
            npz_path = self.episode_root / f"{ep['episode_id']}.npz"
            if not npz_path.exists():
                continue
            T = ep.get('num_transitions', 0)
            if T == 0:
                continue
            for t in range(history_len, T - 1):
                self.samples.append({
                    'episode_id': ep['episode_id'],
                    't': t,
                    'npz_path': str(npz_path),
                    'family': ep.get('family', 'unknown'),
                })

        if limit_samples and limit_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, limit_samples)

        # Cache feature keys from first episode
        self._feature_keys_cache = None
        self._feature_dim = 0
        if self.samples:
            data = np.load(self.samples[0]['npz_path'], allow_pickle=True)
            self._feature_keys_cache = self._get_feature_keys(data)
            self._feature_dim = len(self._feature_keys_cache)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = np.load(s['npz_path'], allow_pickle=True)
        t = s['t']
        H = self.history_len

        feature_keys = self._feature_keys_cache or self._get_feature_keys(data)
        N_features = len(feature_keys)

        history = np.zeros((H, N_features), dtype=np.float32)
        for hi, t_idx in enumerate(range(t - H, t)):
            for fi, key in enumerate(feature_keys):
                arr = data[key]
                if t_idx < len(arr):
                    history[hi, fi] = arr[t_idx]

        action = data['actions_physical'][t].astype(np.float32)

        obj_pose = data['object_poses'][t].astype(np.float32)
        next_obj_pose = data['next_object_poses'][t].astype(np.float32)
        goal_pose = data['goal_pose'].astype(np.float32)

        dynamics_target = _wrap_pose_delta(next_obj_pose - obj_pose)
        subgoal_target = _wrap_pose_delta(goal_pose - obj_pose)

        return {
            'history': torch.from_numpy(history),
            'action': torch.from_numpy(action),
            'dynamics_target': torch.from_numpy(dynamics_target),
            'subgoal_target': torch.from_numpy(subgoal_target),
        }

    def _get_feature_keys(self, data):
        """Get ordered list of numeric feature keys from npz."""
        if self.profile_features is not None:
            keys = [k for k in self.profile_features if k in data
                    and isinstance(data[k], np.ndarray)
                    and data[k].dtype.kind in 'fi'
                    and data[k].ndim == 1]
        else:
            keys = sorted([k for k in data.keys()
                          if isinstance(data[k], np.ndarray)
                          and data[k].dtype.kind in 'fi'
                          and data[k].ndim == 1
                          and k != 'schema_version'])
        return keys

    @property
    def feature_dim(self):
        return self._feature_dim
