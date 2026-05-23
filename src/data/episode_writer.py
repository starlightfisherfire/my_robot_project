#!/usr/bin/env python3
"""Episode writer for MPPI parameter sweep — saves transition-level npz + metadata jsonl."""
from __future__ import annotations

import json
import uuid
import time
from pathlib import Path
from typing import Optional

import numpy as np


class EpisodeWriter:
    """Accumulates per-step transition data and writes episode .npz + metadata .jsonl."""

    def __init__(self, output_dir: str | Path, run_root: str = ""):
        self.output_dir = Path(output_dir)
        self.episodes_dir = self.output_dir / "episodes"
        self.metadata_dir = self.output_dir / "metadata"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.run_root = run_root

        # Per-episode accumulators
        self._reset_episode()

    def _reset_episode(self):
        self.states: list[np.ndarray] = []
        self.actions_norm: list[np.ndarray] = []
        self.actions_physical: list[np.ndarray] = []
        self.next_states: list[np.ndarray] = []
        self.object_poses: list[np.ndarray] = []
        self.next_object_poses: list[np.ndarray] = []
        self.ee_positions: list[np.ndarray] = []
        self.next_ee_positions: list[np.ndarray] = []
        self.contact_flags: list[bool] = []
        self.collision_flags: list[bool] = []
        self.actual_ee_velocities: list[np.ndarray] = []
        self.actual_object_velocities: list[np.ndarray] = []
        self.timestamps: list[float] = []

    def add_step(
        self,
        state: np.ndarray,
        action_norm: np.ndarray,
        action_physical: np.ndarray,
        next_state: np.ndarray,
        object_pose: np.ndarray,
        next_object_pose: np.ndarray,
        ee_position: np.ndarray,
        next_ee_position: np.ndarray,
        contact: bool,
        collision: bool,
        actual_ee_velocity: np.ndarray | None = None,
        actual_object_velocity: np.ndarray | None = None,
    ):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions_norm.append(np.asarray(action_norm, dtype=np.float32))
        self.actions_physical.append(np.asarray(action_physical, dtype=np.float32))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.object_poses.append(np.asarray(object_pose, dtype=np.float32))
        self.next_object_poses.append(np.asarray(next_object_pose, dtype=np.float32))
        self.ee_positions.append(np.asarray(ee_position, dtype=np.float32))
        self.next_ee_positions.append(np.asarray(next_ee_position, dtype=np.float32))
        self.contact_flags.append(contact)
        self.collision_flags.append(collision)
        if actual_ee_velocity is not None:
            self.actual_ee_velocities.append(np.asarray(actual_ee_velocity, dtype=np.float32))
        if actual_object_velocity is not None:
            self.actual_object_velocities.append(np.asarray(actual_object_velocity, dtype=np.float32))
        self.timestamps.append(time.time())

    def save_episode(
        self,
        goal_pose: np.ndarray,
        obstacle_features: np.ndarray | None = None,
        metadata: dict | None = None,
        candidate_data: dict | None = None,
    ) -> str:
        """Save accumulated episode data as .npz and metadata as .jsonl line.
        Returns episode_id.
        """
        episode_id = str(uuid.uuid4())[:12]
        episode_dir = self.episodes_dir
        episode_path = episode_dir / f"{episode_id}.npz"

        np.savez_compressed(
            episode_path,
            states=np.array(self.states, dtype=np.float32),
            actions_norm=np.array(self.actions_norm, dtype=np.float32),
            actions_physical=np.array(self.actions_physical, dtype=np.float32),
            next_states=np.array(self.next_states, dtype=np.float32),
            object_poses=np.array(self.object_poses, dtype=np.float32),
            next_object_poses=np.array(self.next_object_poses, dtype=np.float32),
            ee_positions=np.array(self.ee_positions, dtype=np.float32),
            next_ee_positions=np.array(self.next_ee_positions, dtype=np.float32),
            contact_flags=np.array(self.contact_flags, dtype=np.bool_),
            collision_flags=np.array(self.collision_flags, dtype=np.bool_),
            actual_ee_velocities=(
                np.array(self.actual_ee_velocities, dtype=np.float32)
                if self.actual_ee_velocities
                else np.array([])
            ),
            actual_object_velocities=(
                np.array(self.actual_object_velocities, dtype=np.float32)
                if self.actual_object_velocities
                else np.array([])
            ),
            goal_pose=np.asarray(goal_pose, dtype=np.float32),
            obstacle_features=(
                np.asarray(obstacle_features, dtype=np.float32)
                if obstacle_features is not None
                else np.array([])
            ),
        )

        # Metadata jsonl
        meta = {
            "episode_id": episode_id,
            "source": "mppi_param_sweep_v1",
            "num_transitions": len(self.states),
            "episode_path": str(episode_path),
        }
        if metadata:
            meta.update(metadata)

        # Thread/process-safe append via fcntl advisory lock
        import fcntl
        meta_path = self.metadata_dir / "episodes.jsonl"
        with open(meta_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Candidate rollouts if provided
        if candidate_data:
            cand_path = (
                self.output_dir
                / "candidate_rollouts"
                / f"{episode_id}_candidates.npz"
            )
            cand_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cand_path, **candidate_data)

        self._reset_episode()
        return episode_id

    @property
    def step_count(self) -> int:
        return len(self.states)
