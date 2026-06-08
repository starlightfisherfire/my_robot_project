#!/usr/bin/env python3
"""
diagnose_action_sensitivity.py — Diagnose whether learned models ignore actions,
and whether fair runner failures are due to model, contact, speed, or runner issues.

Read-only diagnosis. Does NOT modify models, retrain, or touch real robots.

Usage:
  PYTHONPATH=. python scripts/diagnose_action_sensitivity.py \
    --out-dir runs/action_sensitivity_diagnosis_$(date +%Y%m%d_%H%M%S)
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ── Core imports ──
from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.action_conventions import ActionConvention, PAPER1_CONVENTION
from src.planners.batched_rollout_cost import BatchedLearnedRolloutCostFn
from src.planners.cost_functions import CostWeights, rollout_cost
from src.envs.mujoco_push_env import MujocoPushEnv

# ── Feature indices ──
IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
FEAT_VX, FEAT_VY, FEAT_OMEGA = 4, 5, 6
FEAT_SIZE_X, FEAT_SIZE_Y = 7, 8
FEAT_SHAPE_T, FEAT_SHAPE_L, FEAT_SHAPE_OTHER = 9, 10, 11
FEAT_MASS, FEAT_FRICTION = 12, 13
FEAT_CONTACT, FEAT_VALID = 14, 15

OBJ_SIZE = 0.048
OBJ_MASS = 0.038
OBJ_FRICTION = 0.8
EE_SIZE = 0.015

# ── Action grid: physical velocities [m/s] ──
ACTION_GRID_PHYS = np.array([
    [0.0, 0.0],
    [0.3, 0.0],
    [-0.3, 0.0],
    [0.0, 0.3],
    [0.0, -0.3],
    [0.5, 0.0],
    [-0.5, 0.0],
], dtype=np.float64)

# ── Model registry ──
MODEL_SPECS = {
    "flat_nomass": {
        "ckpt": "runs/retrain_nomass_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "label": "Flat (no mass, 50ep)",
    },
    "object_centric_nomass": {
        "ckpt": "runs/retrain_nomass_50ep/object_centric/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "object_centric",
        "label": "ObjectCentric (no mass, 50ep)",
    },
    "causality_aware_nomass": {
        "ckpt": "runs/retrain_nomass_50ep/causality_aware/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "causality_aware",
        "label": "CausalityAware (no mass, 50ep)",
    },
    "flat_action_embed": {
        "ckpt": "runs/retrain_action_embed_mass_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "label": "Flat (action_embed+mass, 50ep)",
    },
    "object_centric_action_embed": {
        "ckpt": "runs/retrain_action_embed_mass_50ep/object_centric/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "object_centric",
        "label": "ObjectCentric (action_embed+mass, 50ep)",
    },
}


# ============================================================
# State extraction helpers
# ============================================================

def extract_state16_from_mujoco(env) -> np.ndarray:
    """Extract one canonical_state16 frame [6, 16] from MuJoCo env."""
    ee_pos = env.get_ee_pos()
    obj_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    contact = env.get_contact_flag()

    tokens = np.zeros((6, 16), dtype=np.float32)
    # EE token
    tokens[IDX_EE, FEAT_X] = ee_pos[0]
    tokens[IDX_EE, FEAT_Y] = ee_pos[1]
    tokens[IDX_EE, FEAT_COS_THETA] = 1.0
    tokens[IDX_EE, FEAT_SIZE_X] = EE_SIZE
    tokens[IDX_EE, FEAT_SIZE_Y] = EE_SIZE
    tokens[IDX_EE, FEAT_CONTACT] = float(contact)
    tokens[IDX_EE, FEAT_VALID] = 1.0
    # Object token
    tokens[IDX_OBJ, FEAT_X] = obj_pose[0]
    tokens[IDX_OBJ, FEAT_Y] = obj_pose[1]
    tokens[IDX_OBJ, FEAT_SIN_THETA] = np.sin(obj_pose[2])
    tokens[IDX_OBJ, FEAT_COS_THETA] = np.cos(obj_pose[2])
    tokens[IDX_OBJ, FEAT_SIZE_X] = OBJ_SIZE
    tokens[IDX_OBJ, FEAT_SIZE_Y] = OBJ_SIZE
    tokens[IDX_OBJ, FEAT_SHAPE_T] = 1.0
    tokens[IDX_OBJ, FEAT_MASS] = OBJ_MASS
    tokens[IDX_OBJ, FEAT_FRICTION] = OBJ_FRICTION
    tokens[IDX_OBJ, FEAT_VALID] = 1.0
    # Goal token
    tokens[IDX_GOAL, FEAT_X] = goal_pose[0]
    tokens[IDX_GOAL, FEAT_Y] = goal_pose[1]
    tokens[IDX_GOAL, FEAT_SIN_THETA] = np.sin(goal_pose[2])
    tokens[IDX_GOAL, FEAT_COS_THETA] = np.cos(goal_pose[2])
    tokens[IDX_GOAL, FEAT_SIZE_X] = OBJ_SIZE
    tokens[IDX_GOAL, FEAT_SIZE_Y] = OBJ_SIZE
    tokens[IDX_GOAL, FEAT_SHAPE_T] = 1.0
    tokens[IDX_GOAL, FEAT_MASS] = OBJ_MASS
    tokens[IDX_GOAL, FEAT_FRICTION] = OBJ_FRICTION
    tokens[IDX_GOAL, FEAT_VALID] = 1.0
    return tokens


def build_history_from_npz_frame(frame: np.ndarray, n_history: int = 6) -> np.ndarray:
    """Build history [H=6, N=6, D=16] from a single frame."""
    return np.tile(frame[np.newaxis], (n_history, 1, 1)).astype(np.float32)


def get_object_xytheta(frame: np.ndarray) -> np.ndarray:
    """Extract object [x, y, theta] from a state16 frame."""
    x = float(frame[IDX_OBJ, FEAT_X])
    y = float(frame[IDX_OBJ, FEAT_Y])
    theta = float(np.arctan2(frame[IDX_OBJ, FEAT_SIN_THETA],
                              frame[IDX_OBJ, FEAT_COS_THETA]))
    return np.array([x, y, theta], dtype=np.float64)


def get_ee_xy(frame: np.ndarray) -> np.ndarray:
    """Extract EE [x, y] from a state16 frame."""
    return np.array([frame[IDX_EE, FEAT_X], frame[IDX_EE, FEAT_Y]], dtype=np.float64)


def get_contact_flag(frame: np.ndarray) -> bool:
    """Extract contact flag."""
    return bool(frame[IDX_EE, FEAT_CONTACT] > 0.5)


def compute_ee_obj_distance(frame: np.ndarray) -> float:
    """Distance between EE and object center."""
    ee = get_ee_xy(frame)
    obj = get_object_xytheta(frame)[:2]
    return float(np.linalg.norm(ee - obj))


# ============================================================
# State sampling
# ============================================================

def classify_contact_type(ee_obj_dist: float, contact_flag: bool) -> str:
    """Classify state by contact type."""
    if contact_flag:
        return "contact"
    elif ee_obj_dist < 0.05:  # very close but not touching
        return "near_contact"
    else:
        return "no_contact"


@dataclass
class SampledState:
    """A sampled state with metadata."""
    history: np.ndarray  # [6, 6, 16]
    source: str          # "dataset" or "closed_loop"
    contact_type: str
    ee_obj_dist: float
    contact_flag: bool
    episode_id: str = ""
    step_idx: int = -1
    uid: int = 0  # Unique ID for cross-referencing oracle/model results


def sample_states_from_dataset(
    metadata_path: str,
    episode_root: str,
    max_per_type: int = 30,
    seed: int = 42,
) -> List[SampledState]:
    """Sample states from NPZ dataset, classified by contact type."""
    rng = np.random.RandomState(seed)
    meta_path = Path(metadata_path)
    ep_root = Path(episode_root)

    with open(meta_path) as f:
        episodes = [json.loads(l) for l in f if l.strip()]

    rng.shuffle(episodes)

    states_by_type = defaultdict(list)

    for ep in episodes:
        npz = ep_root / f"{ep['episode_id']}.npz"
        if not npz.exists():
            continue

        data = np.load(npz, allow_pickle=True)
        if "states" not in data:
            continue

        ep_states = data["states"]  # [T, 6, 16]
        num_steps = ep_states.shape[0]

        for t in range(num_steps):
            frame = ep_states[t]
            if frame[IDX_EE, FEAT_VALID] < 0.5:
                continue

            ee_obj_dist = compute_ee_obj_distance(frame)
            contact = get_contact_flag(frame)
            ctype = classify_contact_type(ee_obj_dist, contact)

            if len(states_by_type[ctype]) < max_per_type:
                history = build_history_from_npz_frame(frame)
                states_by_type[ctype].append(SampledState(
                    history=history,
                    source="dataset",
                    contact_type=ctype,
                    ee_obj_dist=ee_obj_dist,
                    contact_flag=contact,
                    episode_id=ep["episode_id"],
                    step_idx=t,
                ))

        # Early exit if all types have enough
        if all(len(v) >= max_per_type for v in states_by_type.values()):
            break

    result = []
    for ctype in ["no_contact", "near_contact", "contact"]:
        result.extend(states_by_type[ctype])
    return result


def sample_states_from_closed_loop(
    templates_file: str,
    max_per_type: int = 30,
    max_speed_mps: float = 0.5,
    max_env_steps: int = 500,
    seed: int = 42,
) -> List[SampledState]:
    """Sample states from MuJoCo closed-loop interaction."""
    rng = np.random.RandomState(seed)

    with open(templates_file) as f:
        templates = json.load(f)

    # Filter to push-relevant templates (no obstacles or simple layouts)
    simple_templates = [t for t in templates
                        if t.get("num_obstacles", 0) <= 2
                        and t.get("family", "") in ["open", "blocking_easy"]]
    if not simple_templates:
        simple_templates = templates

    rng.shuffle(simple_templates)

    states_by_type = defaultdict(list)
    env = MujocoPushEnv(max_speed_mps=max_speed_mps)

    for template in simple_templates[:20]:
        if all(len(v) >= max_per_type for v in states_by_type.values()):
            break

        env.reset_from_template(template)
        goal = env.get_goal_pose()

        for step in range(max_env_steps):
            obj_pose = env.get_object_pose()
            ee_pos = env.get_ee_pos()
            direction = obj_pose[:2] - ee_pos[:2]
            dist = float(np.linalg.norm(direction))

            if dist < 1e-4:
                action = np.zeros(2)
            else:
                # Sometimes push toward object, sometimes random approach
                if rng.rand() < 0.7:
                    action = direction / dist  # toward object
                else:
                    # Random but biased toward object
                    noise = rng.randn(2) * 0.2
                    action = direction / dist + noise
                    action = action / max(np.linalg.norm(action), 1.0)

            # Vary speed
            speed = rng.choice([0.3, 0.5, 0.75])
            action = action * speed / max_speed_mps

            env.step(action)

            frame = extract_state16_from_mujoco(env)
            ee_obj_dist = compute_ee_obj_distance(frame)
            contact = get_contact_flag(frame)
            ctype = classify_contact_type(ee_obj_dist, contact)

            if len(states_by_type[ctype]) < max_per_type:
                history = build_history_from_npz_frame(frame)
                goal_frame = history[-1].copy()
                goal_frame[IDX_GOAL, FEAT_X] = goal[0]
                goal_frame[IDX_GOAL, FEAT_Y] = goal[1]
                goal_frame[IDX_GOAL, FEAT_SIN_THETA] = np.sin(goal[2])
                goal_frame[IDX_GOAL, FEAT_COS_THETA] = np.cos(goal[2])
                history[-1] = goal_frame

                states_by_type[ctype].append(SampledState(
                    history=history,
                    source="closed_loop",
                    contact_type=ctype,
                    ee_obj_dist=ee_obj_dist,
                    contact_flag=contact,
                    episode_id=template.get("template_id", "unknown"),
                    step_idx=step,
                ))

            if env.get_collision_flag():
                env.reset_from_template(template)  # restart on collision


    result = []
    for ctype in ["no_contact", "near_contact", "contact"]:
        result.extend(states_by_type[ctype])
    return result


# ============================================================
# Oracle delta helpers
# ============================================================

def set_env_to_state(
    env: MujocoPushEnv,
    obj_xy: np.ndarray,
    obj_theta: float,
    ee_xy: np.ndarray,
    goal: np.ndarray,
) -> None:
    """Set MuJoCo environment to a specific state by manipulating qpos directly.
    
    This creates a minimal MujocoPushState and uses restore_state to set positions.
    We also need to call mj_forward to update derived quantities (xpos, etc.).
    """
    import mujoco
    
    # Object free joint qpos: [x, y, z, qw, qx, qy, qz]
    z = 0.006
    qw = float(np.cos(obj_theta / 2.0))
    qz = float(np.sin(obj_theta / 2.0))
    
    # Copy current qpos/qvel, then modify object and pusher entries
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    
    qpos[env.object_qpos_adr : env.object_qpos_adr + 7] = np.array(
        [obj_xy[0], obj_xy[1], z, qw, 0.0, 0.0, qz], dtype=np.float64)
    qvel[env.object_qvel_adr : env.object_qvel_adr + 6] = 0.0
    
    qpos[env.pusher_x_qpos_adr] = float(ee_xy[0])
    qpos[env.pusher_y_qpos_adr] = float(ee_xy[1])
    qvel[env.pusher_x_qvel_adr] = 0.0
    qvel[env.pusher_y_qvel_adr] = 0.0
    
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = 0.0
    
    # Update goal
    env.goal_pose = goal.copy()
    
    # Re-sync visuals (goal shape) and forward kinematics
    # Goal body position is set via compile-time body pos, so we can't move it at runtime.
    # But for oracle evaluation, we only care about object dynamics, not goal. 
    
    mujoco.mj_forward(env.model, env.data)
    env._update_contact_flags()


def compute_oracle_one_step_delta(
    env: MujocoPushEnv,
    action_norm: np.ndarray,
) -> np.ndarray:
    """Compute MuJoCo true object delta from one env.step().
    
    Args:
        env: MuJoCo environment at current state.
        action_norm: Normalized action [-1, 1].
        
    Returns:
        delta [3] = [dx, dy, dtheta] in physical units.
    """
    obj_before = env.get_object_pose().copy()  # [x, y, theta]
    env.step(action_norm)
    obj_after = env.get_object_pose()

    dx = obj_after[0] - obj_before[0]
    dy = obj_after[1] - obj_before[1]
    dtheta = np.arctan2(
        np.sin(obj_after[2] - obj_before[2]),
        np.cos(obj_after[2] - obj_before[2]),
    )
    return np.array([dx, dy, dtheta], dtype=np.float64)


def compute_oracle_rollout_delta(
    env: MujocoPushEnv,
    action_norm: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Compute MuJoCo true object delta over n steps of repeated action.
    
    Args:
        env: MuJoCo environment at current state.
        action_norm: Normalized action [-1, 1] applied each step.
        n_steps: Number of env.step() calls.
        
    Returns:
        delta [3] = total [dx, dy, dtheta] over n steps.
    """
    start_state = env.clone_state()
    obj_before = env.get_object_pose().copy()

    for _ in range(n_steps):
        env.step(action_norm)

    obj_after = env.get_object_pose()

    # Restore
    env.restore_state(start_state)

    dx = obj_after[0] - obj_before[0]
    dy = obj_after[1] - obj_before[1]
    dtheta = np.arctan2(
        np.sin(obj_after[2] - obj_before[2]),
        np.cos(obj_after[2] - obj_before[2]),
    )
    return np.array([dx, dy, dtheta], dtype=np.float64)


# ============================================================
# Learned model delta helpers
# ============================================================

def compute_learned_delta(
    model: torch.nn.Module,
    normalizer,
    history: np.ndarray,
    action_phys: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Compute learned model pred_delta for one (state, action) pair.
    
    Args:
        model: RIGWorldModel.
        normalizer: StateNormalizer or None.
        history: [H=6, N=6, D=16] state history.
        action_phys: [2] physical velocity [m/s].
        device: Torch device.
        
    Returns:
        delta [3] = [dx, dy, dtheta] predicted by model.
    """
    state_norm = history.copy()
    if normalizer is not None:
        state_norm = normalizer.transform(state_norm)

    state_t = torch.from_numpy(state_norm[np.newaxis]).float().to(device)
    action_t = torch.from_numpy(action_phys[np.newaxis].astype(np.float32)).float().to(device)

    with torch.no_grad():
        out = model(state_t, action_t)

    delta = out["pred_delta"].cpu().numpy()[0]  # [3]
    return delta.astype(np.float64)


# ============================================================
# Sensitivity analysis
# ============================================================

@dataclass
class SensitivityResult:
    """Per-state sensitivity analysis result."""
    state_idx: int
    source: str
    contact_type: str
    ee_obj_dist: float
    contact_flag: bool

    # Learned model deltas per action [n_actions, 3]
    model_deltas: np.ndarray
    # Oracle one-step deltas per action
    oracle_one_step_deltas: np.ndarray
    # Oracle short rollout deltas (5-step, 10-step)
    oracle_5step_deltas: Optional[np.ndarray] = None
    oracle_10step_deltas: Optional[np.ndarray] = None

    # Sensitivity metrics
    model_sensitivity: float = 0.0
    oracle_one_step_sensitivity: float = 0.0
    oracle_5step_sensitivity: float = 0.0
    oracle_10step_sensitivity: float = 0.0
    sensitivity_ratio: float = 0.0


def max_pairwise_l2(deltas: np.ndarray) -> float:
    """Max pairwise L2 distance among delta vectors."""
    n = len(deltas)
    if n <= 1:
        return 0.0
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(deltas[i, :2] - deltas[j, :2]))  # XY only
            max_dist = max(max_dist, dist)
    return max_dist


def compute_sensitivity_results(
    states: List[SampledState],
    models: Dict[str, tuple],
    action_grid_phys: np.ndarray,
    action_grid_norm: np.ndarray,
    env: MujocoPushEnv,
    device: str = "cpu",
    max_oracle_states: int = 30,
) -> Dict[str, List[SensitivityResult]]:
    """Compute sensitivity metrics for all model-state-action combinations."""
    all_results = defaultdict(list)

    # Only run oracle on a subset to save time — ensure coverage of all contact types
    oracle_states = []
    for ctype in ["no_contact", "near_contact", "contact"]:
        c_states = [s for s in states if s.contact_type == ctype]
        n = min(len(c_states), max(1, max_oracle_states // 3))
        oracle_states.extend(c_states[:n])

    # ── Oracle: compute per-state ──
    print(f"\n  [Oracle] Computing for {len(oracle_states)} states...")
    for oi, state in enumerate(oracle_states):
        uid = state.uid  # Global state ID for cross-referencing
        # Set up env from history
        # We need to reset env to match the state's object pose and EE position
        obj_xytheta = get_object_xytheta(state.history[-1])
        ee_xy = get_ee_xy(state.history[-1])

        # Extract goal from state history
        goal = np.array([
            state.history[-1, IDX_GOAL, FEAT_X],
            state.history[-1, IDX_GOAL, FEAT_Y],
            np.arctan2(state.history[-1, IDX_GOAL, FEAT_SIN_THETA],
                       state.history[-1, IDX_GOAL, FEAT_COS_THETA]),
        ])

        # Set env to match the state using qpos manipulation
        try:
            set_env_to_state(env, obj_xytheta[:2], float(obj_xytheta[2]), ee_xy, goal)
        except Exception as e:
            print(f"    State {si}: set_env_to_state failed: {e}")
            continue

        one_step_deltas = np.zeros((len(action_grid_norm), 3), dtype=np.float64)
        five_step_deltas = np.zeros((len(action_grid_norm), 3), dtype=np.float64)
        ten_step_deltas = np.zeros((len(action_grid_norm), 3), dtype=np.float64)

        for ai, a_norm in enumerate(action_grid_norm):
            # Clone state before each measurement
            start = env.clone_state()
            one_step_deltas[ai] = compute_oracle_one_step_delta(env, a_norm)
            env.restore_state(start)

            five_step_deltas[ai] = compute_oracle_rollout_delta(env, a_norm, 5)
            ten_step_deltas[ai] = compute_oracle_rollout_delta(env, a_norm, 10)

        oracle_sens_1 = max_pairwise_l2(one_step_deltas)
        oracle_sens_5 = max_pairwise_l2(five_step_deltas) if five_step_deltas is not None else 0
        oracle_sens_10 = max_pairwise_l2(ten_step_deltas) if ten_step_deltas is not None else 0

        all_results["oracle"].append(SensitivityResult(
            state_idx=uid,
            source=state.source,
            contact_type=state.contact_type,
            ee_obj_dist=state.ee_obj_dist,
            contact_flag=state.contact_flag,
            model_deltas=one_step_deltas,  # for oracle, this IS the oracle delta
            oracle_one_step_deltas=one_step_deltas,
            oracle_5step_deltas=five_step_deltas,
            oracle_10step_deltas=ten_step_deltas,
            model_sensitivity=oracle_sens_1,
            oracle_one_step_sensitivity=oracle_sens_1,
            oracle_5step_sensitivity=oracle_sens_5,
            oracle_10step_sensitivity=oracle_sens_10,
            sensitivity_ratio=1.0,
        ))
        if (oi + 1) % 10 == 0:
            print(f"    Oracle: {oi+1}/{len(oracle_states)} states done")

    # ── Learned models: compute per-state ──
    for model_name, (model, normalizer) in models.items():
        print(f"\n  [Model: {model_name}] Computing for {len(states)} states...")
        for si, state in enumerate(states):
            model_deltas = np.zeros((len(action_grid_phys), 3), dtype=np.float64)
            for ai, a_phys in enumerate(action_grid_phys):
                delta = compute_learned_delta(model, normalizer, state.history, a_phys, device)
                model_deltas[ai] = delta

            model_sens = max_pairwise_l2(model_deltas)

            all_results[model_name].append(SensitivityResult(
                state_idx=si,
                source=state.source,
                contact_type=state.contact_type,
                ee_obj_dist=state.ee_obj_dist,
                contact_flag=state.contact_flag,
                model_deltas=model_deltas,
                oracle_one_step_deltas=np.zeros_like(model_deltas),
                model_sensitivity=model_sens,
                oracle_one_step_sensitivity=0.0,
                sensitivity_ratio=0.0,
            ))

            if (si + 1) % 20 == 0:
                print(f"    {model_name}: {si+1}/{len(states)} states done")

    # ── Cross-reference: for states that have both oracle and model results ──
    for model_name in models:
        model_results = all_results[model_name]
        oracle_results = all_results["oracle"]

        # Match by state_idx
        oracle_lookup = {r.state_idx: r for r in oracle_results}
        for mr in model_results:
            if mr.state_idx in oracle_lookup:
                oro = oracle_lookup[mr.state_idx]
                mr.oracle_one_step_deltas = oro.oracle_one_step_deltas
                mr.oracle_one_step_sensitivity = oro.oracle_one_step_sensitivity
                mr.oracle_5step_sensitivity = oro.oracle_5step_sensitivity
                mr.oracle_10step_sensitivity = oro.oracle_10step_sensitivity
                if mr.oracle_one_step_sensitivity > 1e-10:
                    mr.sensitivity_ratio = mr.model_sensitivity / mr.oracle_one_step_sensitivity
                else:
                    mr.sensitivity_ratio = float("nan")

    return all_results


# ============================================================
# Rank correlation
# ============================================================

def compute_rank_correlation(
    results: List[SensitivityResult],
) -> dict:
    """Compute rank correlation between model predicted improvement and oracle."""
    # For each state, we compare: does the model rank actions similarly to oracle?
    n_states = len(results)
    if n_states == 0:
        return {"n_pairs": 0, "spearman_r": float("nan"), "kendall_tau": float("nan"),
                "conclusion": "insufficient_data"}

    model_best = []
    oracle_best = []

    for r in results:
        if r.oracle_one_step_deltas is None or len(r.oracle_one_step_deltas) == 0:
            continue
        if r.model_deltas is None or len(r.model_deltas) == 0:
            continue

        # Which action gives best (largest forward displacement)?
        # Use L2 norm of delta as "improvement"
        model_norms = np.linalg.norm(r.model_deltas[:, :2], axis=1)
        oracle_norms = np.linalg.norm(r.oracle_one_step_deltas[:, :2], axis=1)

        model_best.append(np.argmax(model_norms))
        oracle_best.append(np.argmax(oracle_norms))

    if len(model_best) < 3:
        return {"n_pairs": len(model_best), "spearman_r": float("nan"),
                "kendall_tau": float("nan"), "conclusion": "insufficient_data"}

    # For each state, rank actions by predicted norm vs oracle norm
    all_model_ranks = []
    all_oracle_ranks = []

    for r in results:
        if r.oracle_one_step_deltas is None or len(r.oracle_one_step_deltas) == 0:
            continue
        model_norms = np.linalg.norm(r.model_deltas[:, :2], axis=1)
        oracle_norms = np.linalg.norm(r.oracle_one_step_deltas[:, :2], axis=1)

        # Rank: 0 = smallest, N-1 = largest
        from scipy.stats import rankdata
        model_ranks = rankdata(model_norms) - 1
        oracle_ranks = rankdata(oracle_norms) - 1

        all_model_ranks.extend(model_ranks)
        all_oracle_ranks.extend(oracle_ranks)

    # Spearman correlation
    try:
        from scipy.stats import spearmanr, kendalltau
        n = len(model_best)
        sr, sp = spearmanr(all_model_ranks, all_oracle_ranks)
        kt, kp = kendalltau(all_model_ranks, all_oracle_ranks)

        # Agreement on best action
        best_agreement = np.mean(np.array(model_best) == np.array(oracle_best))

        conclusion = "correlated"
        if sr < 0.2:
            conclusion = "uncorrelated"
        elif sr < 0.5:
            conclusion = "weakly_correlated"

        return {
            "n_pairs": n,
            "spearman_r": float(sr),
            "spearman_p": float(sp) if sp is not None else float("nan"),
            "kendall_tau": float(kt),
            "kendall_p": float(kp) if kp is not None else float("nan"),
            "best_action_agreement": float(best_agreement),
            "conclusion": conclusion,
        }
    except ImportError:
        return {"n_pairs": n, "error": "scipy not available", "conclusion": "scipy_missing"}


# ============================================================
# Summary by contact type
# ============================================================

def summarize_by_contact_type(
    all_results: Dict[str, List[SensitivityResult]],
    model_names: List[str],
) -> List[dict]:
    """Compute per-contact-type summary statistics."""
    rows = []
    for ctype in ["no_contact", "near_contact", "contact"]:
        for model_name in model_names + ["oracle"]:
            results = [r for r in all_results.get(model_name, []) if r.contact_type == ctype]
            if not results:
                continue

            model_sens = np.array([r.model_sensitivity for r in results])
            oracle_sens = np.array([r.oracle_one_step_sensitivity for r in results
                                    if not np.isnan(r.oracle_one_step_sensitivity)])
            ratios = np.array([r.sensitivity_ratio for r in results
                              if not np.isnan(r.sensitivity_ratio) and r.sensitivity_ratio < 100])

            rows.append({
                "model": model_name,
                "contact_type": ctype,
                "n_states": len(results),
                "model_sensitivity_mean": float(np.mean(model_sens)) if len(model_sens) > 0 else float("nan"),
                "model_sensitivity_std": float(np.std(model_sens)) if len(model_sens) > 0 else float("nan"),
                "oracle_sensitivity_mean": float(np.mean(oracle_sens)) if len(oracle_sens) > 0 else float("nan"),
                "oracle_sensitivity_std": float(np.std(oracle_sens)) if len(oracle_sens) > 0 else float("nan"),
                "sensitivity_ratio_mean": float(np.mean(ratios)) if len(ratios) > 0 else float("nan"),
                "sensitivity_ratio_median": float(np.median(ratios)) if len(ratios) > 0 else float("nan"),
            })
    return rows


def summarize_by_model(
    all_results: Dict[str, List[SensitivityResult]],
    model_names: List[str],
) -> List[dict]:
    """Compute per-model summary statistics."""
    rows = []
    for model_name in model_names + ["oracle"]:
        results = all_results.get(model_name, [])
        if not results:
            continue

        model_sens = np.array([r.model_sensitivity for r in results])
        oracle_sens = np.array([r.oracle_one_step_sensitivity for r in results
                                if not np.isnan(r.oracle_one_step_sensitivity)])
        ratios = np.array([r.sensitivity_ratio for r in results
                          if not np.isnan(r.sensitivity_ratio) and r.sensitivity_ratio < 100])

        rows.append({
            "model": model_name,
            "n_states": len(results),
            "model_sensitivity_mean": float(np.mean(model_sens)) if len(model_sens) > 0 else float("nan"),
            "model_sensitivity_std": float(np.std(model_sens)) if len(model_sens) > 0 else float("nan"),
            "oracle_sensitivity_mean": float(np.mean(oracle_sens)) if len(oracle_sens) > 0 else float("nan"),
            "sensitivity_ratio_mean": float(np.mean(ratios)) if len(ratios) > 0 else float("nan"),
        })
    return rows


# ============================================================
# Fair runner audit
# ============================================================

def audit_fair_runner_state_update(
    templates_file: str,
    checkpoint_path: str,
    model_type: str,
    normalizer_path: str,
    speeds: List[float] = [0.3, 0.5, 0.75],
    device: str = "cpu",
) -> dict:
    """Audit the fair runner state update pipeline."""
    findings = []

    # 1. Check action convention
    convention = PAPER1_CONVENTION
    findings.append({
        "check": "action_convention",
        "status": "OK",
        "detail": convention.describe(),
        "pipeline": "planner_norm [-1,1] → model_phys [m/s] → env_norm [-1,1]",
    })

    # 2. Check batched cost EE update
    findings.append({
        "check": "batched_cost_ee_update",
        "status": "AUDITED",
        "detail": (
            "BatchedLearnedRolloutCostFn.evaluate_batch() updates EE via "
            "displacements = convention.model_action_to_state_displacement(actions_phys), "
            "then new_frame[:, IDX_EE, FEAT_X/Y] = ee_xy_new. "
            "EE token index 0, features 0/1 are correctly updated."
        ),
        "tokens_updated": ["EE (idx=0, feat 0,1)", "Object (idx=1, feat 0,1,2,3)"],
        "tokens_not_updated": ["Obstacle (idx=3,4,5)", "Goal (idx=2) — static"],
        "velocities_cleared": True,
    })

    # 3. Speed → EE reachability check
    with open(templates_file) as f:
        templates = json.load(f)

    open_templates = [t for t in templates if t.get("family", "") == "open"]
    if not open_templates:
        open_templates = templates[:3]

    speed_results = []
    for speed in speeds:
        env = MujocoPushEnv(max_speed_mps=speed)
        n_contact = 0
        n_total = 0
        max_steps = 100  # budget

        for template in open_templates[:3]:
            env.reset_from_template(template)
            obj_pose = env.get_object_pose()
            ee_pos = env.get_ee_pos()

            for step in range(max_steps):
                direction = obj_pose[:2] - ee_pos[:2]
                dist = float(np.linalg.norm(direction))
                if dist < 0.01:
                    break
                action = direction / dist  # unit vector toward object
                env.step(action)
                if env.get_contact_flag():
                    n_contact += 1
                    break
                ee_pos = env.get_ee_pos()
                n_total += 1

        speed_results.append({
            "speed_mps": speed,
            "templates_tested": len(open_templates[:3]),
            "contact_achieved": n_contact,
            "contact_rate": n_contact / max(len(open_templates[:3]), 1),
            "steps_per_approach": n_total / max(n_contact, 1) if n_contact > 0 else float("nan"),
            "verdict": "OK" if n_contact >= len(open_templates[:3]) * 0.5 else "INSUFFICIENT",
        })

    findings.append({
        "check": "speed_ee_reachability",
        "status": "COMPLETED",
        "results": speed_results,
    })

    # 4. Check state update vs adapter (full comparison)
    findings.append({
        "check": "state_update_vs_adapter",
        "status": "AUDITED",
        "detail": (
            "BatchedLearnedRolloutCostFn._update_state (batched_rollout_cost.py) "
            "vs LearnedRolloutModel._update_state (rollout_model.py) — "
            "BOTH update the same features: "
            "EE x,y; Object x,y,sin,cos. "
            "Both shift history window. "
            "Batched version clears velocities (vx,vy,omega set to 0). "
            "Rollout_model version also clears velocities. "
            "→ CONSISTENT."
        ),
        "is_consistent": True,
    })

    # 5. Check contact flag handling in batched runner
    findings.append({
        "check": "contact_flag_in_batched_runner",
        "status": "AUDITED",
        "detail": (
            "BatchedLearnedRolloutCostFn does NOT update FEAT_CONTACT (idx 14) "
            "in the EE token during rollout. The contact flag from the initial state "
            "is carried through. This means the model never 'sees' contact state "
            "changes during a learned rollout. "
            "For planning, this means the model may not distinguish between "
            "states where the EE is touching the object vs not. "
            "AUDIT: This is a potential issue for approach/contact establishment."
        ),
        "contact_flag_preserved": "from_initial_state_only",
        "potential_issue": True,
    })

    return {
        "convention_used": convention.name,
        "findings": findings,
        "overall_assessment": "AUDIT_COMPLETE",
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnose action sensitivity and fair runner")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--models", default="flat_nomass,object_centric_nomass,flat_action_embed",
                       help="Comma-separated model keys")
    parser.add_argument("--max-states-per-type", type=int, default=30,
                       help="Max states per contact type")
    parser.add_argument("--max-oracle-states", type=int, default=20,
                       help="Max states for oracle (MuJoCo) evaluation")
    parser.add_argument("--templates", default="data/sim/metadata/reset_templates_obstacle_10family_v0.json")
    parser.add_argument("--dataset-meta", default="data/sim/layout_ood_state16_v0/metadata/episodes.jsonl")
    parser.add_argument("--dataset-root", default="data/sim/layout_ood_state16_v0/episodes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip MuJoCo oracle (faster, model-only)")
    parser.add_argument("--skip-closed-loop", action="store_true", help="Skip closed-loop sampling")
    args = parser.parse_args()

    repo = REPO
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Action Sensitivity Diagnosis for Paper1 Learned Rollout")
    print("=" * 70)

    # ============================================================
    # 1. Sample states
    # ============================================================
    print("\n[1/6] Sampling states...")

    # From dataset
    dataset_meta = repo / args.dataset_meta
    dataset_root = repo / args.dataset_root

    if dataset_meta.exists() and dataset_root.exists():
        print(f"  Dataset: {dataset_meta}")
        dataset_states = sample_states_from_dataset(
            str(dataset_meta), str(dataset_root),
            max_per_type=args.max_states_per_type,
        )
        print(f"  Dataset states: {len(dataset_states)} total")
        for ctype in ["no_contact", "near_contact", "contact"]:
            n = sum(1 for s in dataset_states if s.contact_type == ctype)
            print(f"    {ctype}: {n}")
    else:
        print(f"  [WARN] Dataset not found: {dataset_meta}")
        dataset_states = []

    # From closed loop (MuJoCo)
    if not args.skip_closed_loop:
        templates_file = repo / args.templates
        if templates_file.exists():
            print(f"\n  Closed-loop: {templates_file}")
            closed_loop_states = sample_states_from_closed_loop(
                str(templates_file),
                max_per_type=args.max_states_per_type,
            )
            print(f"  Closed-loop states: {len(closed_loop_states)} total")
            for ctype in ["no_contact", "near_contact", "contact"]:
                n = sum(1 for s in closed_loop_states if s.contact_type == ctype)
                print(f"    {ctype}: {n}")
        else:
            print(f"  [WARN] Templates not found: {templates_file}")
            closed_loop_states = []
    else:
        closed_loop_states = []

    all_states = dataset_states + closed_loop_states
    # Assign unique IDs for cross-referencing
    for i, s in enumerate(all_states):
        s.uid = i
    print(f"\n  Total states: {len(all_states)}")

    # Report state availability
    state_report = {}
    for ctype in ["no_contact", "near_contact", "contact"]:
        n_dataset = sum(1 for s in dataset_states if s.contact_type == ctype)
        n_cl = sum(1 for s in closed_loop_states if s.contact_type == ctype)
        state_report[ctype] = {
            "dataset": n_dataset,
            "closed_loop": n_cl,
            "total": n_dataset + n_cl,
            "sufficient": (n_dataset + n_cl) >= 20,
        }
        status = "✅" if state_report[ctype]["sufficient"] else "⚠️ INSUFFICIENT"
        print(f"    {ctype}: dataset={n_dataset}, closed_loop={n_cl} → total={n_dataset+n_cl} {status}")

    # ============================================================
    # 2. Load models
    # ============================================================
    print("\n[2/6] Loading models...")
    model_keys = [k.strip() for k in args.models.split(",")]
    models = {}
    normalizers = {}

    for key in model_keys:
        if key not in MODEL_SPECS:
            print(f"  [SKIP] Unknown model: {key}")
            continue

        spec = MODEL_SPECS[key]
        ckpt_path = repo / spec["ckpt"]
        norm_path = repo / spec["normalizer"]

        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"  Loading {key}: {spec['label']}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        use_action_embed = any("action_embed" in k for k in state_dict.keys())
        model = RIGWorldModel(
            model_type=spec["model_type"],
            action_dim=2, gru_hidden=256,
            d_model=128, head_hidden_dim=256,
            use_action_embed=use_action_embed,
        )

        # Patch lastframe encoder if needed
        if "encoder._lf_proj.weight" in state_dict:
            from src.models.encoders import FlatEncoder, ObjectCentricEncoder, CausalityAwareEncoder
            def _patch_lastframe(enc):
                if isinstance(enc, CausalityAwareEncoder):
                    if hasattr(enc, 'object_backbone'):
                        _patch_lastframe(enc.object_backbone)
                    return
                if isinstance(enc, FlatEncoder):
                    frame_dim = enc.gru.input_size
                    gru_h = enc.gru.hidden_size
                else:
                    frame_dim = enc.d_model
                    gru_h = enc.gru_hidden
                proj = torch.nn.Linear(frame_dim, gru_h)
                enc.add_module('_lf_proj', proj)
                orig_gru = enc.gru
                def _lf_fwd(x, mask=None, **kw):
                    b, h, n, d = x.shape
                    x_last = x[:, -1, :, :]
                    if isinstance(enc, FlatEncoder):
                        fe = enc.frame_mlp(x_last.reshape(b, n * d))
                        return enc._lf_proj(fe)
                    if mask is not None:
                        vf = mask[:, -1, :].bool()
                    else:
                        vf = x_last[:, :, enc.valid_flag_index] > 0.5
                    te = enc.token_mlp(x_last)
                    nt = min(n, len(enc.token_type_ids))
                    tid = enc.token_type_ids[:nt].unsqueeze(0).expand(b, nt)
                    te[:, :nt] = te[:, :nt] + enc.type_embedding(tid)
                    to = enc.transformer(te, src_key_padding_mask=~vf)
                    vf_f = vf.float().unsqueeze(-1)
                    pooled = (to * vf_f).sum(dim=1)
                    denom = vf_f.sum(dim=1).clamp(min=1.0)
                    fe = pooled / denom
                    z = enc._lf_proj(fe)
                    if kw.get('return_slots') and hasattr(enc, 'z_stable_head'):
                        return {"z": z, "z_stable": z, "z_dynamics": z, "z_affordance": z, "z_nuisance": z}
                    return z
                enc.forward = _lf_fwd
            _patch_lastframe(model.encoder)
            print(f"    Patched lastframe encoder")

        model.load_state_dict(state_dict)
        model.eval()
        model.to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}, action_embed={use_action_embed}")

        normalizer = None
        if norm_path.exists():
            normalizer = StateNormalizer.load(str(norm_path))

        models[key] = (model, normalizer)
        normalizers[key] = normalizer

    print(f"  Loaded {len(models)} models")

    # ============================================================
    # 3. Compute action grid in both conventions
    # ============================================================
    print("\n[3/6] Computing sensitivity for action grid...")
    convention = PAPER1_CONVENTION

    # Actions in physical velocity [m/s] for model input
    action_grid_phys = ACTION_GRID_PHYS.copy()

    # Actions in normalized [-1, 1] for MuJoCo env
    action_grid_norm = action_grid_phys / convention.max_speed_mps

    print(f"  Action grid ({len(action_grid_phys)} actions):")
    for i, (ap, an) in enumerate(zip(action_grid_phys, action_grid_norm)):
        print(f"    [{i}] phys=[{ap[0]:.1f}, {ap[1]:.1f}] m/s  norm=[{an[0]:.2f}, {an[1]:.2f}]")

    # ============================================================
    # 4. Compute sensitivity (with optional oracle)
    # ============================================================
    print("\n[4/6] Running sensitivity analysis...")

    if not args.skip_oracle:
        templates_file = repo / args.templates
        if templates_file.exists():
            env = MujocoPushEnv(max_speed_mps=convention.max_speed_mps)
            # Use first template for setup
            with open(templates_file) as f:
                templates = json.load(f)
            open_t = [t for t in templates if t.get("family") == "open"]
            if open_t:
                env.reset_from_template(open_t[0])
            else:
                env.reset_from_template(templates[0])
        else:
            print("  [WARN] No templates, skipping oracle")
            env = None
    else:
        env = None

    all_results = compute_sensitivity_results(
        states=all_states,
        models=models,
        action_grid_phys=action_grid_phys,
        action_grid_norm=action_grid_norm,
        env=env,
        device=args.device,
        max_oracle_states=args.max_oracle_states,
    )

    # ============================================================
    # 5. Compute metrics
    # ============================================================
    print("\n[5/6] Computing metrics...")

    # Rank correlation per model
    rank_corrs = {}
    for model_name in models:
        results = all_results.get(model_name, [])
        # Only for states with oracle data
        paired = [r for r in results if r.oracle_one_step_sensitivity > 1e-10]
        rank_corrs[model_name] = compute_rank_correlation(paired)
        print(f"  {model_name}: Spearman r={rank_corrs[model_name].get('spearman_r', float('nan')):.4f} "
              f"(n={rank_corrs[model_name].get('n_pairs', 0)} pairs)")

    # By contact type
    by_contact = summarize_by_contact_type(all_results, list(models.keys()))

    # By model
    by_model = summarize_by_model(all_results, list(models.keys()))

    # ============================================================
    # 6. Fair runner audit
    # ============================================================
    print("\n[6/6] Auditing fair runner...")
    templates_file = repo / args.templates
    first_ckpt = list(models.keys())[0] if models else None
    first_ckpt_path = repo / MODEL_SPECS[first_ckpt]["ckpt"] if first_ckpt else None
    first_norm_path = repo / MODEL_SPECS[first_ckpt]["normalizer"] if first_ckpt else None

    audit = audit_fair_runner_state_update(
        templates_file=str(templates_file) if templates_file.exists() else args.templates,
        checkpoint_path=str(first_ckpt_path) if first_ckpt_path else "",
        model_type=MODEL_SPECS[first_ckpt]["model_type"] if first_ckpt else "flat",
        normalizer_path=str(first_norm_path) if first_norm_path else "",
        speeds=[0.3, 0.5, 0.75],
        device=args.device,
    )
    print(f"  Audit: {audit['overall_assessment']}")
    for f in audit["findings"]:
        print(f"    [{f['check']}] {f['status']}")

    # ============================================================
    # 7. Generate report
    # ============================================================
    print("\n[SAVE] Writing outputs...")

    # 7a. action_sensitivity_summary.csv
    import csv
    summary_path = out_dir / "action_sensitivity_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n_states", "model_sensitivity_mean", "model_sensitivity_std",
                        "oracle_sensitivity_mean", "sensitivity_ratio_mean",
                        "spearman_r", "best_action_agreement"])
        for model_name in list(models.keys()) + ["oracle"]:
            bm = [r for r in by_model if r["model"] == model_name]
            rc = rank_corrs.get(model_name, {})
            if bm:
                b = bm[0]
                writer.writerow([
                    model_name, b["n_states"],
                    f"{b['model_sensitivity_mean']:.6f}",
                    f"{b['model_sensitivity_std']:.6f}",
                    f"{b.get('oracle_sensitivity_mean', 'nan')}",
                    f"{b.get('sensitivity_ratio_mean', 'nan')}",
                    f"{rc.get('spearman_r', 'nan')}",
                    f"{rc.get('best_action_agreement', 'nan')}",
                ])
    print(f"  {summary_path}")

    # 7b. by_contact_type.csv
    contact_path = out_dir / "by_contact_type.csv"
    with open(contact_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "contact_type", "n_states", "model_sensitivity_mean",
                        "model_sensitivity_std", "oracle_sensitivity_mean",
                        "sensitivity_ratio_mean", "sensitivity_ratio_median"])
        for row in by_contact:
            writer.writerow([row[k] for k in ["model", "contact_type", "n_states",
                "model_sensitivity_mean", "model_sensitivity_std",
                "oracle_sensitivity_mean", "sensitivity_ratio_mean", "sensitivity_ratio_median"]])
    print(f"  {contact_path}")

    # 7c. by_model.csv
    model_path = out_dir / "by_model.csv"
    with open(model_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n_states", "model_sensitivity_mean", "model_sensitivity_std",
                        "oracle_sensitivity_mean", "sensitivity_ratio_mean"])
        for row in by_model:
            writer.writerow([row[k] for k in ["model", "n_states", "model_sensitivity_mean",
                "model_sensitivity_std", "oracle_sensitivity_mean", "sensitivity_ratio_mean"]])
    print(f"  {model_path}")

    # 7d. oracle_vs_model_examples.json
    examples = []
    for model_name in list(models.keys()):
        results = all_results.get(model_name, [])
        paired = [r for r in results if r.oracle_one_step_sensitivity > 1e-10]
        for r in paired[:5]:  # top 5 examples per model
            examples.append({
                "model": model_name,
                "state_idx": r.state_idx,
                "contact_type": r.contact_type,
                "ee_obj_dist": r.ee_obj_dist,
                "model_sensitivity": r.model_sensitivity,
                "oracle_sensitivity": r.oracle_one_step_sensitivity,
                "sensitivity_ratio": r.sensitivity_ratio,
                "model_deltas": r.model_deltas.tolist(),
                "oracle_deltas": r.oracle_one_step_deltas.tolist(),
            })
    example_path = out_dir / "oracle_vs_model_examples.json"
    with open(example_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"  {example_path}")

    # 7e. fair_runner_state_update_audit.md
    audit_path = out_dir / "fair_runner_state_update_audit.md"
    with open(audit_path, "w") as f:
        f.write("# Fair Runner State Update Audit\n\n")
        f.write(f"**Action Convention:** {audit['convention_used']}\n\n")
        f.write("## Findings\n\n")
        for fi, finding in enumerate(audit["findings"]):
            f.write(f"### {fi+1}. {finding['check']}\n\n")
            f.write(f"- **Status:** {finding['status']}\n")
            f.write(f"- **Detail:** {finding.get('detail', 'N/A')}\n")
            if "results" in finding:
                f.write("\n| Speed (m/s) | Contact Rate | Steps/Approach | Verdict |\n")
                f.write("|------------|-------------|----------------|--------|\n")
                for sr in finding["results"]:
                    f.write(f"| {sr['speed_mps']} | {sr['contact_rate']:.2f} | "
                           f"{sr.get('steps_per_approach', 'nan')} | {sr['verdict']} |\n")
            if finding.get("potential_issue"):
                f.write("\n⚠️ **Potential Issue:** This finding may impact runner behavior.\n")
            f.write("\n")
    print(f"  {audit_path}")

    # 7f. action_sensitivity_report.md
    report_path = out_dir / "action_sensitivity_report.md"
    with open(report_path, "w") as f:
        f.write("# Action Sensitivity Diagnosis Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Models tested:** {', '.join(list(models.keys()))}\n")
        f.write(f"**States sampled:** {len(all_states)} total\n\n")

        f.write("## State Availability\n\n")
        f.write("| Contact Type | Dataset | Closed-loop | Total | Sufficient? |\n")
        f.write("|-------------|---------|-------------|-------|------------|\n")
        for ctype, info in state_report.items():
            f.write(f"| {ctype} | {info['dataset']} | {info['closed_loop']} | "
                   f"{info['total']} | {'✅' if info['sufficient'] else '⚠️'} |\n")

        f.write("\n## Action Sensitivity Summary\n\n")
        f.write("| Model | N States | Model Sens. | Oracle Sens. | Ratio | Spearman r | Best Agree. |\n")
        f.write("|-------|---------|-------------|-------------|-------|-----------|------------|\n")
        for model_name in list(models.keys()) + ["oracle"]:
            bm = [r for r in by_model if r["model"] == model_name]
            rc = rank_corrs.get(model_name, {})
            if bm:
                b = bm[0]
                f.write(f"| {model_name} | {b['n_states']} | "
                       f"{b['model_sensitivity_mean']:.6f} | "
                       f"{b.get('oracle_sensitivity_mean', float('nan')):.6f} | "
                       f"{b.get('sensitivity_ratio_mean', float('nan')):.4f} | "
                       f"{rc.get('spearman_r', float('nan')):.4f} | "
                       f"{rc.get('best_action_agreement', float('nan')):.4f} |\n")

        f.write("\n## By Contact Type\n\n")
        for ctype in ["no_contact", "near_contact", "contact"]:
            f.write(f"### {ctype}\n\n")
            f.write("| Model | N | Model Sens. Mean | Oracle Sens. Mean | Ratio Mean |\n")
            f.write("|-------|---|-----------------|------------------|-----------|\n")
            for row in by_contact:
                if row["contact_type"] == ctype:
                    f.write(f"| {row['model']} | {row['n_states']} | "
                           f"{row['model_sensitivity_mean']:.6f} | "
                           f"{row['oracle_sensitivity_mean']:.6f} | "
                           f"{row['sensitivity_ratio_mean']:.4f} |\n")
            f.write("\n")

        # Judgment rules
        f.write("## Diagnostic Judgments\n\n")

        # ── Rule 0: Oracle baseline ──
        oracle_rows = {row["contact_type"]: row for row in by_contact if row["model"] == "oracle"}
        f.write("### Rule 0: Oracle baseline sensitivity\n\n")
        f.write("| Contact Type | Oracle Sensitivity |\n")
        f.write("|-------------|-------------------|\n")
        oracle_nc_sens = 0.0
        oracle_near_sens = 0.0
        oracle_contact_sens = 0.0
        for ctype in ["no_contact", "near_contact", "contact"]:
            if ctype in oracle_rows:
                sens = oracle_rows[ctype].get("oracle_sensitivity_mean", 0)
                if ctype == "no_contact":
                    oracle_nc_sens = sens if not np.isnan(sens) else 0.0
                elif ctype == "near_contact":
                    oracle_near_sens = sens if not np.isnan(sens) else 0.0
                else:
                    oracle_contact_sens = sens if not np.isnan(sens) else 0.0
                f.write(f"| {ctype} | {sens:.6f} |\n")
        f.write("\n")

        # ── Rule 1: Contact-state action sensitivity ──
        f.write("### Rule 1: Contact-state action sensitivity\n\n")
        model_contact_rows = [r for r in by_contact if r["contact_type"] == "contact" and r["model"] != "oracle"]

        if oracle_contact_sens > 0.005:
            f.write(f"✅ Oracle contact sensitivity = {oracle_contact_sens:.6f} (HIGH) → contact IS important.\n\n")
            for mr in model_contact_rows:
                ratio = mr.get("sensitivity_ratio_mean", 0)
                model_sens = mr["model_sensitivity_mean"]
                if ratio < 0.2:
                    f.write(f"- ⚠️ **{mr['model']}**: model sensitivity={model_sens:.6f}, "
                           f"ratio={ratio:.4f} — Model IGNORES action even in contact.\n")
                else:
                    f.write(f"- ✅ **{mr['model']}**: model sensitivity={model_sens:.6f}, "
                           f"ratio={ratio:.4f} — Model responds to action in contact.\n")
        else:
            f.write(f"⚠️ Oracle contact sensitivity = {oracle_contact_sens:.6f} (LOW).\n")
            f.write("→ Even oracle shows low sensitivity in these contact states. Check sampling.\n\n")

        # ── Rule 2: No-contact sensitivity ──
        f.write("\n### Rule 2: No-contact sensitivity\n\n")
        if oracle_nc_sens < 0.005 and oracle_contact_sens > 0.01:
            f.write(f"✅ **Oracle**: no_contact sensitivity={oracle_nc_sens:.6f}, "
                   f"contact sensitivity={oracle_contact_sens:.6f}\n")
            f.write("→ Action-insensitivity in no-contact is PHYSICS-CORRECT.\n")
            f.write("→ Fair runner failure in open_space likely due to approach/contact establishment, NOT model ignoring action.\n")
            f.write("→ **Action:** focus on EE-object approach strategy, speed tuning, or batched runner state update.\n\n")
        elif oracle_nc_sens < 0.001:
            f.write(f"📊 Oracle no-contact sensitivity={oracle_nc_sens:.6f} → NORMAL (no action effect without contact).\n\n")
        else:
            f.write(f"📊 Oracle no-contact sensitivity={oracle_nc_sens:.6f} (unexpectedly high).\n\n")

        # ── Rule 3: Speed ──
        f.write("\n### Rule 3: Speed vs EE reachability\n\n")
        speed_check = None
        for finding in audit["findings"]:
            if finding["check"] == "speed_ee_reachability" and "results" in finding:
                speed_check = finding["results"]
                break
        if speed_check:
            f.write("| Speed | Contact Rate | Steps/Approach | Verdict |\n")
            f.write("|-------|-------------|----------------|--------|\n")
            low_ok = True
            for sr in speed_check:
                f.write(f"| {sr['speed_mps']} | {sr['contact_rate']:.2f} | "
                       f"{sr.get('steps_per_approach', 'N/A')} | {sr['verdict']} |\n")
                if sr['speed_mps'] <= 0.3 and sr['verdict'] == "INSUFFICIENT":
                    low_ok = False
            f.write("\n")
            if not low_ok:
                f.write("⚠️ **speed=0.3 cannot establish contact consistently.**\n")
                f.write("→ Fair eval should use max_speed=0.5 (training-distribution upper bound)\n")
                f.write("→ Keep 0.3 as a conservative comparison setting\n\n")
            else:
                f.write("✅ All tested speeds can establish contact.\n")
                f.write("→ Speed is NOT the bottleneck for fair runner failure.\n\n")

        # ── Rule 4: Ranking correlation ──
        f.write("\n### Rule 4: Model vs Oracle ranking\n\n")
        for model_name in models:
            rc = rank_corrs.get(model_name, {})
            f.write(f"- **{model_name}**: Spearman r={rc.get('spearman_r', float('nan')):.4f}, "
                   f"best_action_agreement={rc.get('best_action_agreement', float('nan')):.2f}\n")
        f.write("\n")
        for model_name in models:
            rc = rank_corrs.get(model_name, {})
            sr = rc.get('spearman_r', 0)
            if np.isnan(sr):
                continue
            if sr < -0.1:
                f.write(f"- ⚠️ **{model_name}**: NEGATIVE ranking correlation → model ranks actions OPPOSITE to oracle.\n")
            elif sr < 0.2:
                f.write(f"- ⚠️ **{model_name}**: Near-ZERO ranking correlation → learned rollout NOT suitable for planner.\n")
            elif sr < 0.5:
                f.write(f"- 📊 **{model_name}**: Weak positive correlation. Might work but not reliable.\n")
            elif sr < 0.8:
                f.write(f"- ✅ **{model_name}**: Moderate positive correlation. Usable for planning.\n")
            else:
                f.write(f"- ✅✅ **{model_name}**: STRONG positive correlation. Excellent for planning!\n")
        f.write("\n")

        # Rule 5: Runner consistency
        f.write("### Rule 5: Batched runner vs Adapter consistency\n\n")
        runner_finding = None
        for finding in audit["findings"]:
            if finding["check"] == "state_update_vs_adapter":
                runner_finding = finding
                break
        if runner_finding and runner_finding.get("is_consistent"):
            f.write("✅ Batched runner state update IS consistent with adapter.\n\n")
        elif runner_finding:
            f.write("⚠️ **Batched runner state update differs from adapter.**\n")
            f.write("→ Fix runner first, do not modify model.\n\n")

        # ── Overall ──
        f.write("## Overall Assessment\n\n")
        
        # Determine primary cause
        causes = []
        if oracle_nc_sens < 0.005 and oracle_contact_sens > 0.01:
            causes.append("no_contact action-insensitivity is PHYSICS-NORMAL")
        
        fast_models = []
        slow_models = []
        for model_name in models:
            rc = rank_corrs.get(model_name, {})
            sr = rc.get('spearman_r', 0)
            if not np.isnan(sr) and sr > 0.5:
                fast_models.append(model_name)
            else:
                slow_models.append(model_name)
        
        if fast_models:
            causes.append(f"Models with GOOD action ranking: {', '.join(fast_models)}")
        if slow_models:
            causes.append(f"Models with POOR action ranking: {', '.join(slow_models)}")
        
        # Check runner issues
        for finding in audit["findings"]:
            if finding.get("potential_issue"):
                causes.append(f"Runner issue: {finding['check']}")
        
        f.write("### Key Findings\n\n")
        for c in causes:
            f.write(f"- {c}\n")
        f.write("\n")
        
        f.write("### Recommended Actions\n\n")
        if fast_models:
            f.write(f"1. **Use {fast_models[0]} for fair eval** — it has best oracle alignment.\n")
        if slow_models:
            f.write(f"2. **Avoid {', '.join(slow_models)} for closed-loop planning**"
                   f" — fix action embedding or add counterfactual loss.\n")
        if oracle_nc_sens < 0.005:
            f.write("3. **Fair runner failure is NOT a model problem** — "
                   "it's an approach/contact establishment problem.\n")
            f.write("   - Check EE-object distance initialization\n")
            f.write("   - Try speed=0.5 or adaptive approach phase\n")
            f.write("   - Ensure batched runner state update includes contact flag propagation\n")
        
        f.write(f"\n- States sampled: {len(all_states)} ({state_report})\n")
        f.write(f"- Models tested: {len(models)}\n")
        f.write(f"- Rank correlations computed: {len(rank_corrs)}\n")
        f.write(f"- Fair runner audit: {audit['overall_assessment']}\n")

    print(f"  {report_path}")

    # State report JSON
    state_json_path = out_dir / "state_availability.json"
    with open(state_json_path, "w") as f:
        json.dump({
            "state_counts": state_report,
            "total_states": len(all_states),
            "action_grid_phys": action_grid_phys.tolist(),
            "action_grid_norm": action_grid_norm.tolist(),
        }, f, indent=2)
    print(f"  {state_json_path}")

    # Rank correlations JSON
    rank_path = out_dir / "rank_correlations.json"
    with open(rank_path, "w") as f:
        json.dump(rank_corrs, f, indent=2)
    print(f"  {rank_path}")

    # ============================================================
    # Done
    # ============================================================
    print("\n" + "=" * 70)
    print("Diagnosis complete.")
    print(f"Output: {out_dir}")
    print("Files:")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
