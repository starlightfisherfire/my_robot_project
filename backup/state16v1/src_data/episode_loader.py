# src/data/episode_loader.py
"""Dataset loader for Paper 1 state16 proof-of-concept."""

from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _wrap_pose_delta(delta: np.ndarray) -> np.ndarray:
    """Wrap theta component (index 2) to [-pi, pi]."""
    out = delta.copy()
    out[..., 2] = _wrap_to_pi(out[..., 2])
    return out


class State16Dataset(Dataset):
    """Loads .npz episodes, constructs supervised samples for high-level model."""

    def __init__(
        self,
        metadata_path: str,
        episode_root: str,
        history_len: int = 6,
        token_count: int = 6,
        state_dim: int = 16,
        action_dim: int = 2,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        limit_samples: Optional[int] = None,
        seed: int = 0,
        split_episode_ids: Optional[list[str]] = None,
    ):
        self.history_len = history_len
        self.token_count = token_count
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Read metadata
        meta_path = Path(metadata_path)
        with meta_path.open("r") as f:
            all_episodes = [json.loads(line) for line in f if line.strip()]

        # If split_episode_ids provided, filter by those IDs
        if split_episode_ids is not None:
            id_set = set(split_episode_ids)
            episodes = [ep for ep in all_episodes if ep["episode_id"] in id_set]
        else:
            # Check for split info
            has_split = any("split_name" in ep for ep in all_episodes)
            
            if has_split:
                episodes = [ep for ep in all_episodes if ep.get("split_name") == split]
                if not episodes:
                    has_split = False
            
            if not has_split:
                rng = random.Random(seed)
                rng.shuffle(all_episodes)
                n = len(all_episodes)
                n_train = max(1, int(n * train_ratio))
                n_val = max(1, int(n * val_ratio)) if n > 1 else 1
                if split == "train":
                    episodes = all_episodes[:n_train]
                elif split == "val":
                    start = n_train
                    end = min(n_train + n_val, n)
                    episodes = all_episodes[start:end]
                else:
                    episodes = []

        self.episodes = episodes
        self.episode_root = Path(episode_root)

        # Build sample index
        self.samples = []
        for ep in episodes:
            npz_path = self.episode_root / f"{ep['episode_id']}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            n_steps = len(data["states"])
            for t in range(history_len, n_steps - 1):
                self.samples.append({
                    "episode_id": ep["episode_id"],
                    "t": t,
                    "npz_path": str(npz_path),
                    "family": ep.get("family", "unknown"),
                    "split_name": ep.get("split_name", split),
                })

        if limit_samples and limit_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, limit_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = np.load(s["npz_path"], allow_pickle=True)
        t = s["t"]
        H, N, D = self.history_len, self.token_count, self.state_dim

        # History states [H, N, D]
        history = data["states"][t - H : t].astype(np.float32)

        # Action [2]
        action = data["actions_physical"][t].astype(np.float32)

        # Targets
        current_pose = data["object_poses"][t].astype(np.float32)
        next_pose = data["next_object_poses"][t].astype(np.float32)
        goal_pose = data["goal_pose"].astype(np.float32)

        dynamics_target = _wrap_pose_delta(next_pose - current_pose)
        subgoal_target = _wrap_pose_delta(goal_pose - current_pose)

        return {
            "history": torch.from_numpy(history),
            "action": torch.from_numpy(action),
            "dynamics_target": torch.from_numpy(dynamics_target),
            "subgoal_target": torch.from_numpy(subgoal_target),
            "episode_id": s["episode_id"],
            "t": t,
            "family": s["family"],
        }
