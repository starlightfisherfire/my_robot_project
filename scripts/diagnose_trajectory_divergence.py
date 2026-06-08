#!/usr/bin/env python3
"""
diagnose_trajectory_divergence.py — Hypotheses C & D diagnostic.

C: Learned rollout trajectory accumulation error
   → Compare learned trajectory vs oracle trajectory at each timestep
   → If divergence grows with horizon → accumulation error is the issue

D: Cost function robustness to learned trajectories
   → Compare cost RANKING of action sequences: oracle vs learned
   → If rankings disagree → cost function is unreliable for learned rollouts

Models tested: flat_action_embed (best single-step) vs flat_nomass (worst)

Usage:
  PYTHONPATH=. python scripts/diagnose_trajectory_divergence.py \
    --out-dir runs/trajectory_divergence_$(date +%Y%m%d_%H%M%S)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.action_conventions import ActionConvention, PAPER1_CONVENTION
from src.planners.rollout_model import LearnedRolloutModel, load_learned_rollout_model
from src.planners.mujoco_oracle_rollout import rollout_action_sequence_mujoco
from src.planners.cost_functions import CostWeights, rollout_cost
from src.envs.mujoco_push_env import MujocoPushEnv

# ── Feature indices ──
IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
FEAT_VX, FEAT_VY, FEAT_OMEGA = 4, 5, 6
FEAT_VALID = 15
EE_SIZE = 0.015
OBJ_SIZE = 0.048
OBJ_MASS = 0.038
OBJ_FRICTION = 0.8
FEAT_SIZE_X, FEAT_SIZE_Y = 7, 8
FEAT_SHAPE_T, FEAT_SHAPE_L, FEAT_SHAPE_OTHER = 9, 10, 11
FEAT_MASS, FEAT_FRICTION = 12, 13
FEAT_CONTACT = 14


# ============================================================
# Model registry
# ============================================================
MODEL_SPECS = {
    "flat_action_embed": {
        "ckpt": "runs/retrain_action_embed_mass_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "label": "Flat (action_embed+mass, 50ep)",
    },
    "flat_nomass": {
        "ckpt": "runs/retrain_nomass_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "label": "Flat (no mass, 50ep)",
    },
}


# ============================================================
# Helpers
# ============================================================
def extract_state16_from_mujoco(env) -> np.ndarray:
    """Extract one canonical_state16 frame [6, 16] from MuJoCo env."""
    ee_pos = env.get_ee_pos()
    obj_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    contact = env.get_contact_flag()

    tokens = np.zeros((6, 16), dtype=np.float32)
    tokens[IDX_EE, FEAT_X] = ee_pos[0]
    tokens[IDX_EE, FEAT_Y] = ee_pos[1]
    tokens[IDX_EE, FEAT_COS_THETA] = 1.0
    tokens[IDX_EE, FEAT_SIZE_X] = EE_SIZE
    tokens[IDX_EE, FEAT_SIZE_Y] = EE_SIZE
    tokens[IDX_EE, FEAT_CONTACT] = float(contact)
    tokens[IDX_EE, FEAT_VALID] = 1.0

    tokens[IDX_OBJ, FEAT_X] = obj_pose[0]
    tokens[IDX_OBJ, FEAT_Y] = obj_pose[1]
    tokens[IDX_OBJ, FEAT_SIN_THETA] = np.sin(obj_pose[2])
    tokens[IDX_OBJ, FEAT_COS_THETA] = np.cos(obj_pose[2])
    tokens[IDX_OBJ, FEAT_SIZE_X] = OBJ_SIZE
    tokens[IDX_OBJ, FEAT_SIZE_Y] = OBJ_SIZE
    tokens[IDX_OBJ, FEAT_MASS] = OBJ_MASS
    tokens[IDX_OBJ, FEAT_FRICTION] = OBJ_FRICTION
    tokens[IDX_OBJ, FEAT_VALID] = 1.0

    tokens[IDX_GOAL, FEAT_X] = goal_pose[0]
    tokens[IDX_GOAL, FEAT_Y] = goal_pose[1]
    tokens[IDX_GOAL, FEAT_SIN_THETA] = np.sin(goal_pose[2])
    tokens[IDX_GOAL, FEAT_COS_THETA] = np.cos(goal_pose[2])
    tokens[IDX_GOAL, FEAT_SIZE_X] = OBJ_SIZE
    tokens[IDX_GOAL, FEAT_SIZE_Y] = OBJ_SIZE
    tokens[IDX_GOAL, FEAT_MASS] = OBJ_MASS
    tokens[IDX_GOAL, FEAT_FRICTION] = OBJ_FRICTION
    tokens[IDX_GOAL, FEAT_VALID] = 1.0
    return tokens


def build_history(frame: np.ndarray, n: int = 6) -> np.ndarray:
    """Tile a single frame into [n, 6, 16] history."""
    return np.tile(frame[np.newaxis], (n, 1, 1)).astype(np.float32)


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def compute_velocity(prev: np.ndarray, curr: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Finite-difference velocity features."""
    state = curr.copy()
    state[IDX_EE, FEAT_VX] = (curr[IDX_EE, FEAT_X] - prev[IDX_EE, FEAT_X]) / dt
    state[IDX_EE, FEAT_VY] = (curr[IDX_EE, FEAT_Y] - prev[IDX_EE, FEAT_Y]) / dt
    state[IDX_OBJ, FEAT_VX] = (curr[IDX_OBJ, FEAT_X] - prev[IDX_OBJ, FEAT_X]) / dt
    state[IDX_OBJ, FEAT_VY] = (curr[IDX_OBJ, FEAT_Y] - prev[IDX_OBJ, FEAT_Y]) / dt
    dtheta = np.arctan2(
        np.sin(curr[IDX_OBJ, FEAT_SIN_THETA] - prev[IDX_OBJ, FEAT_SIN_THETA]),
        np.cos(curr[IDX_OBJ, FEAT_COS_THETA] - prev[IDX_OBJ, FEAT_COS_THETA]),
    )
    state[IDX_OBJ, FEAT_OMEGA] = dtheta / dt
    return state


# ============================================================
# Generate action sequences
# ============================================================
def generate_action_sequences(
    horizon: int,
    n_random: int = 30,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray]]:
    """Generate diverse action sequences for comparison.
    
    Returns list of (label, action_seq) where action_seq is [H, 2] normalized.
    """
    rng = np.random.RandomState(seed)
    sequences = []

    # 1. Zero action
    sequences.append(("zero", np.zeros((horizon, 2))))

    # 2. Constant push right
    seq = np.zeros((horizon, 2))
    seq[:, 0] = 0.5
    sequences.append(("push_right_0.5", seq))

    # 3. Constant push left
    seq = np.zeros((horizon, 2))
    seq[:, 0] = -0.5
    sequences.append(("push_left_0.5", seq))

    # 4. Constant push up
    seq = np.zeros((horizon, 2))
    seq[:, 1] = 0.5
    sequences.append(("push_up_0.5", seq))

    # 5. Constant push down
    seq = np.zeros((horizon, 2))
    seq[:, 1] = -0.5
    sequences.append(("push_down_0.5", seq))

    # 6. Diagonal push
    seq = np.zeros((horizon, 2))
    seq[:, 0] = 0.35
    seq[:, 1] = 0.35
    sequences.append(("push_diag_0.35", seq))

    # 7. Alternating left-right
    seq = np.zeros((horizon, 2))
    for t in range(horizon):
        seq[t, 0] = 0.5 if t % 2 == 0 else -0.5
    sequences.append(("alternating_lr", seq))

    # 8. Ramp up
    seq = np.zeros((horizon, 2))
    for t in range(horizon):
        seq[t, 0] = 0.8 * (t / horizon)
    sequences.append(("ramp_right", seq))

    # 9-N. Random sequences
    for i in range(n_random):
        seq = rng.randn(horizon, 2) * 0.3
        seq = np.clip(seq, -1.0, 1.0)
        sequences.append((f"random_{i:02d}", seq))

    return sequences


# ============================================================
# Oracle rollout wrapper
# ============================================================
def oracle_rollout(
    env: MujocoPushEnv,
    initial_state: np.ndarray,
    action_seq_norm: np.ndarray,
    convention: ActionConvention = PAPER1_CONVENTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MuJoCo oracle rollout from initial_state with action_seq.
    
    Returns:
        object_traj: [H+1, 3] object poses
        ee_traj: [H+1, 2] EE positions
    """
    # Set env to match initial_state
    obj_xytheta = np.array([
        initial_state[-1, IDX_OBJ, FEAT_X],
        initial_state[-1, IDX_OBJ, FEAT_Y],
        np.arctan2(initial_state[-1, IDX_OBJ, FEAT_SIN_THETA],
                   initial_state[-1, IDX_OBJ, FEAT_COS_THETA]),
    ], dtype=np.float64)
    ee_xy = np.array([
        initial_state[-1, IDX_EE, FEAT_X],
        initial_state[-1, IDX_EE, FEAT_Y],
    ], dtype=np.float64)
    goal = np.array([
        initial_state[-1, IDX_GOAL, FEAT_X],
        initial_state[-1, IDX_GOAL, FEAT_Y],
        np.arctan2(initial_state[-1, IDX_GOAL, FEAT_SIN_THETA],
                   initial_state[-1, IDX_GOAL, FEAT_COS_THETA]),
    ], dtype=np.float64)

    # Set env state
    import mujoco
    z = 0.006
    theta = float(obj_xytheta[2])
    qw = float(np.cos(theta / 2.0))
    qz = float(np.sin(theta / 2.0))
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    qpos[env.object_qpos_adr:env.object_qpos_adr + 7] = np.array(
        [obj_xytheta[0], obj_xytheta[1], z, qw, 0.0, 0.0, qz], dtype=np.float64)
    qvel[env.object_qvel_adr:env.object_qvel_adr + 6] = 0.0
    qpos[env.pusher_x_qpos_adr] = float(ee_xy[0])
    qpos[env.pusher_y_qpos_adr] = float(ee_xy[1])
    qvel[env.pusher_x_qvel_adr] = 0.0
    qvel[env.pusher_y_qvel_adr] = 0.0
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = 0.0
    env.goal_pose = goal.copy()
    mujoco.mj_forward(env.model, env.data)
    env._update_contact_flags()

    # Rollout
    object_poses = [env.get_object_pose().copy()]
    ee_positions = [env.get_ee_pos().copy()]

    for action_norm in action_seq_norm:
        env.step(action_norm)
        object_poses.append(env.get_object_pose().copy())
        ee_positions.append(env.get_ee_pos().copy())

    return np.array(object_poses), np.array(ee_positions)


# ============================================================
# Learned rollout wrapper
# ============================================================
def learned_rollout(
    rollout_model: LearnedRolloutModel,
    initial_state: np.ndarray,
    action_seq_norm: np.ndarray,
    convention: ActionConvention = PAPER1_CONVENTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run learned rollout from initial_state with action_seq.
    
    Args:
        action_seq_norm: [H, 2] normalized actions
        
    Returns:
        object_traj: [H+1, 3]
        ee_traj: [H+1, 2]
    """
    # Convert normalized → physical velocity for model input
    action_seq_phys = convention.planner_to_model_action(
        action_seq_norm.reshape(-1, 2)
    ).reshape(action_seq_norm.shape)

    result = rollout_model.rollout_sequence(initial_state, action_seq_phys)
    return result.object_traj, result.ee_traj


# ============================================================
# Per-trial result
# ============================================================
@dataclass
class TrajectoryComparison:
    """One action-sequence comparison between oracle and learned."""
    action_label: str
    horizon: int
    
    # [H+1, 3] trajectories
    oracle_obj_traj: np.ndarray
    learned_obj_traj: np.ndarray
    oracle_ee_traj: np.ndarray
    learned_ee_traj: np.ndarray
    
    # Per-step divergence [H]
    obj_l2_per_step: np.ndarray
    obj_theta_per_step: np.ndarray
    ee_l2_per_step: np.ndarray
    
    # Cost comparison
    oracle_cost: float
    learned_cost: float
    
    # Final metrics
    final_obj_l2: float
    final_ee_l2: float
    max_obj_l2: float


# ============================================================
# Main comparison
# ============================================================
def compare_trajectories(
    env: MujocoPushEnv,
    rollout_models: Dict[str, LearnedRolloutModel],
    initial_state: np.ndarray,
    action_sequences: List[Tuple[str, np.ndarray]],
    convention: ActionConvention = PAPER1_CONVENTION,
    goal_pose: Optional[np.ndarray] = None,
) -> Dict[str, List[TrajectoryComparison]]:
    """Compare oracle vs learned trajectories for all models and action sequences."""
    
    all_results = defaultdict(list)
    horizon = len(action_sequences[0][1])
    
    for label, action_seq in action_sequences:
        # Oracle rollout
        oracle_obj, oracle_ee = oracle_rollout(env, initial_state, action_seq, convention)
        
        # Oracle cost
        if goal_pose is not None:
            oracle_cost = rollout_cost(
                oracle_obj, oracle_ee, action_seq, goal_pose, CostWeights())
        else:
            oracle_cost = float('nan')
        
        for model_name, rollout_model in rollout_models.items():
            learned_obj, learned_ee = learned_rollout(
                rollout_model, initial_state, action_seq, convention)
            
            # Learned cost
            if goal_pose is not None:
                learned_cost = rollout_cost(
                    learned_obj, learned_ee, action_seq, goal_pose, CostWeights())
            else:
                learned_cost = float('nan')
            
            # Per-step divergence
            n_steps = min(len(oracle_obj), len(learned_obj))
            obj_l2 = np.linalg.norm(
                oracle_obj[:n_steps, :2] - learned_obj[:n_steps, :2], axis=1)
            obj_theta = np.abs(np.arctan2(
                np.sin(oracle_obj[:n_steps, 2] - learned_obj[:n_steps, 2]),
                np.cos(oracle_obj[:n_steps, 2] - learned_obj[:n_steps, 2]),
            ))
            ee_l2 = np.linalg.norm(
                oracle_ee[:n_steps] - learned_ee[:n_steps], axis=1)
            
            all_results[model_name].append(TrajectoryComparison(
                action_label=label,
                horizon=horizon,
                oracle_obj_traj=oracle_obj,
                learned_obj_traj=learned_obj,
                oracle_ee_traj=oracle_ee,
                learned_ee_traj=learned_ee,
                obj_l2_per_step=obj_l2,
                obj_theta_per_step=obj_theta,
                ee_l2_per_step=ee_l2,
                oracle_cost=oracle_cost,
                learned_cost=learned_cost,
                final_obj_l2=float(obj_l2[-1]),
                final_ee_l2=float(ee_l2[-1]),
                max_obj_l2=float(np.max(obj_l2)),
            ))
    
    return all_results


# ============================================================
# Aggregate metrics
# ============================================================
def compute_divergence_metrics(results: List[TrajectoryComparison]) -> dict:
    """Compute aggregate divergence metrics across action sequences."""
    if not results:
        return {}
    
    final_l2s = [r.final_obj_l2 for r in results]
    max_l2s = [r.max_obj_l2 for r in results]
    
    # Trajectory L2 growth rate (fit linear trend)
    growth_rates = []
    for r in results:
        steps = np.arange(len(r.obj_l2_per_step))
        if len(steps) > 1:
            coeffs = np.polyfit(steps, r.obj_l2_per_step, 1)
            growth_rates.append(float(coeffs[0]))
    
    # Cost ranking correlation
    oracle_costs = [r.oracle_cost for r in results if np.isfinite(r.oracle_cost)]
    learned_costs = [r.learned_cost for r in results if np.isfinite(r.learned_cost)]
    
    cost_rank_corr = float('nan')
    if len(oracle_costs) >= 3:
        from scipy.stats import spearmanr
        try:
            cost_rank_corr, _ = spearmanr(oracle_costs, learned_costs)
        except:
            pass
    
    return {
        "n_sequences": len(results),
        "final_obj_l2_mean": float(np.mean(final_l2s)),
        "final_obj_l2_std": float(np.std(final_l2s)),
        "final_obj_l2_max": float(np.max(final_l2s)),
        "max_obj_l2_mean": float(np.mean(max_l2s)),
        "divergence_growth_rate_mean": float(np.mean(growth_rates)) if growth_rates else float('nan'),
        "divergence_growth_rate_std": float(np.std(growth_rates)) if growth_rates else float('nan'),
        "cost_rank_correlation": cost_rank_corr,
        "oracle_cost_mean": float(np.mean(oracle_costs)) if oracle_costs else float('nan'),
        "learned_cost_mean": float(np.mean(learned_costs)) if learned_costs else float('nan'),
    }


# ============================================================
# Smart loader (auto-detects action_embed, patches lastframe)
# ============================================================
def load_learned_rollout_smart(
    ckpt_path: str,
    model_type: str,
    device: str,
    normalizer_path: Optional[str] = None,
) -> LearnedRolloutModel:
    """Load model with auto-detected use_action_embed and lastframe patch."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    use_action_embed = any("action_embed" in k for k in state_dict.keys())

    model = RIGWorldModel(
        model_type=model_type,
        history_len=6, num_tokens=6, raw_token_dim=16,
        action_dim=2, gru_hidden=256, d_model=128,
        head_hidden_dim=256, use_action_embed=use_action_embed,
    )

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
                return enc._lf_proj(pooled / denom)
            enc.forward = _lf_fwd
        _patch_lastframe(model.encoder)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    normalizer = None
    if normalizer_path and Path(normalizer_path).exists():
        normalizer = StateNormalizer.load(normalizer_path)

    return LearnedRolloutModel(model=model, device=device, normalizer=normalizer)


# ============================================================
# Self-check
# ============================================================
def self_check(repo: Path) -> dict:
    """Run self-checks before diagnosis."""
    checks = {}
    
    # 1. Check imports
    try:
        from src.models.rig_world import RIGWorldModel
        from src.planners.rollout_model import LearnedRolloutModel
        from src.planners.mujoco_oracle_rollout import rollout_action_sequence_mujoco
        from src.planners.cost_functions import rollout_cost
        from src.envs.mujoco_push_env import MujocoPushEnv
        checks["imports"] = "PASS"
    except Exception as e:
        checks["imports"] = f"FAIL: {e}"
    
    # 2. Check models exist
    for name, spec in MODEL_SPECS.items():
        p = repo / spec["ckpt"]
        checks[f"checkpoint_{name}"] = "PASS" if p.exists() else f"MISSING: {p}"
    
    # 3. Check templates
    templates_file = repo / "data/sim/metadata/reset_templates_obstacle_10family_v0.json"
    checks["templates"] = "PASS" if templates_file.exists() else f"MISSING: {templates_file}"
    
    # 4. Quick env + oracle test
    history = None
    action_seq = None
    oracle_obj = None
    oracle_ee = None
    try:
        env = MujocoPushEnv(max_speed_mps=0.5)
        templates = json.load(open(templates_file))
        open_t = [t for t in templates if t.get("family") == "open"]
        if open_t:
            env.reset_from_template(open_t[0])
        frame = extract_state16_from_mujoco(env)
        history = build_history(frame)
        
        action_seq = np.zeros((5, 2), dtype=np.float64)
        action_seq[:, 0] = 0.3
        oracle_obj, oracle_ee = oracle_rollout(env, history, action_seq)
        
        checks["oracle_rollout"] = "PASS" if oracle_obj.shape == (6, 3) else f"SHAPE: {oracle_obj.shape}"
        checks["oracle_no_nan"] = "PASS" if np.isfinite(oracle_obj).all() else "FAIL: NaN/inf"
    except Exception as e:
        checks["oracle_rollout"] = f"FAIL: {e}"
    
    # 5. Quick learned rollout test
    try:
        if history is None or action_seq is None:
            checks["learned_rollout"] = "SKIP (oracle failed)"
        else:
            spec = list(MODEL_SPECS.values())[0]
            ckpt_path = str(repo / spec["ckpt"])
            norm_path = str(repo / spec["normalizer"]) if (repo / spec["normalizer"]).exists() else None
            rollout_m = load_learned_rollout_smart(ckpt_path, spec["model_type"], "cpu", norm_path)
            
            learned_obj, learned_ee = learned_rollout(rollout_m, history, action_seq)
            checks["learned_rollout"] = "PASS" if learned_obj.shape == (6, 3) else f"SHAPE: {learned_obj.shape}"
            checks["learned_no_nan"] = "PASS" if np.isfinite(learned_obj).all() else "FAIL: NaN/inf"
    except Exception as e:
        checks["learned_rollout"] = f"FAIL: {e}"
    
    # 6. Cost function test
    try:
        if oracle_obj is None or oracle_ee is None:
            checks["cost_function"] = "SKIP (oracle failed)"
        else:
            goal = np.array([0.4, 0.2, 0.0])
            cost = rollout_cost(oracle_obj, oracle_ee, action_seq, goal, CostWeights())
            checks["cost_function"] = "PASS" if np.isfinite(cost) else f"FAIL: {cost}"
    except Exception as e:
        checks["cost_function"] = f"FAIL: {e}"
    
    all_pass = all(v == "PASS" for v in checks.values())
    return {"all_pass": all_pass, "checks": checks}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Diagnose trajectory divergence (Hypotheses C & D)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default="flat_action_embed,flat_nomass",
                       help="Comma-separated model keys")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--max-templates", type=int, default=5)
    parser.add_argument("--templates", default="data/sim/metadata/reset_templates_obstacle_10family_v0.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo = REPO
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Self-check
    # ============================================================
    print("=" * 70)
    print("Self-check...")
    print("=" * 70)
    sc = self_check(repo)
    for k, v in sc["checks"].items():
        status = "✅" if v == "PASS" else "❌"
        print(f"  {status} {k}: {v}")
    
    if not sc["all_pass"]:
        print("\n❌ Self-check FAILED. Fix issues before running diagnosis.")
        sys.exit(1)
    print("\n✅ Self-check PASSED.\n")

    # ============================================================
    # Load models
    # ============================================================
    print("[1/5] Loading models...")
    model_keys = [k.strip() for k in args.models.split(",")]
    rollout_models = {}
    
    for key in model_keys:
        if key not in MODEL_SPECS:
            print(f"  [SKIP] Unknown model: {key}")
            continue
        spec = MODEL_SPECS[key]
        ckpt_path = str(repo / spec["ckpt"])
        norm_path = str(repo / spec["normalizer"]) if (repo / spec["normalizer"]).exists() else None
        
        if not Path(ckpt_path).exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue
        
        rollout_m = load_learned_rollout_smart(ckpt_path, spec["model_type"], args.device, norm_path)
        rollout_models[key] = rollout_m
        print(f"  ✅ {key}: {spec['label']}")
    
    if not rollout_models:
        print("ERROR: No models loaded")
        sys.exit(1)

    # ============================================================
    # Load templates
    # ============================================================
    print(f"\n[2/5] Loading templates (max {args.max_templates})...")
    templates_file = repo / args.templates
    with open(templates_file) as f:
        all_templates = json.load(f)
    
    open_templates = [t for t in all_templates if t.get("family") == "open"]
    if not open_templates:
        open_templates = all_templates
    templates = open_templates[:args.max_templates]
    print(f"  Selected {len(templates)} open templates")

    # ============================================================
    # Generate action sequences
    # ============================================================
    print(f"\n[3/5] Generating action sequences (horizon={args.horizon}, n_random={args.n_random})...")
    action_sequences = generate_action_sequences(args.horizon, args.n_random)
    print(f"  Generated {len(action_sequences)} sequences")

    # ============================================================
    # Run comparisons
    # ============================================================
    print(f"\n[4/5] Running trajectory comparisons...")
    convention = PAPER1_CONVENTION
    env = MujocoPushEnv(max_speed_mps=convention.max_speed_mps)
    
    all_template_results = {}
    
    for ti, template in enumerate(templates):
        print(f"\n  Template {ti+1}/{len(templates)}: {template.get('template_id', 'unknown')}")
        env.reset_from_template(template)
        
        # Build initial state
        frame = extract_state16_from_mujoco(env)
        history = build_history(frame)
        goal = env.get_goal_pose()
        
        # Compare trajectories
        results = compare_trajectories(
            env, rollout_models, history, action_sequences, convention, goal)
        
        for model_name, model_results in results.items():
            metrics = compute_divergence_metrics(model_results)
            all_template_results.setdefault(model_name, []).append({
                "template_id": template.get("template_id", "unknown"),
                "metrics": metrics,
                "per_sequence": [{
                    "label": r.action_label,
                    "final_obj_l2": r.final_obj_l2,
                    "max_obj_l2": r.max_obj_l2,
                    "oracle_cost": r.oracle_cost,
                    "learned_cost": r.learned_cost,
                    "divergence_per_step": r.obj_l2_per_step.tolist(),
                } for r in model_results],
            })
            
            print(f"    {model_name}: final_l2_mean={metrics['final_obj_l2_mean']:.4f}m, "
                  f"growth={metrics['divergence_growth_rate_mean']:.6f}m/step, "
                  f"cost_rank_corr={metrics['cost_rank_correlation']:.3f}")
    
    # ============================================================
    # Aggregate across templates
    # ============================================================
    print(f"\n[5/5] Aggregating results...")
    
    summary_rows = []
    for model_name in rollout_models:
        template_results = all_template_results.get(model_name, [])
        if not template_results:
            continue
        
        # Average metrics across templates
        all_final_l2 = []
        all_growth = []
        all_cost_corr = []
        for tr in template_results:
            m = tr["metrics"]
            all_final_l2.append(m["final_obj_l2_mean"])
            all_growth.append(m["divergence_growth_rate_mean"])
            if np.isfinite(m["cost_rank_correlation"]):
                all_cost_corr.append(m["cost_rank_correlation"])
        
        summary_rows.append({
            "model": model_name,
            "n_templates": len(template_results),
            "final_obj_l2_mean": float(np.mean(all_final_l2)),
            "final_obj_l2_std": float(np.std(all_final_l2)),
            "growth_rate_mean": float(np.mean(all_growth)),
            "growth_rate_std": float(np.std(all_growth)),
            "cost_rank_corr_mean": float(np.mean(all_cost_corr)) if all_cost_corr else float('nan'),
            "cost_rank_corr_std": float(np.std(all_cost_corr)) if all_cost_corr else float('nan'),
        })
        
        print(f"  {model_name}:")
        print(f"    Final obj L2: {np.mean(all_final_l2):.4f} ± {np.std(all_final_l2):.4f} m")
        print(f"    Growth rate:  {np.mean(all_growth):.6f} ± {np.std(all_growth):.6f} m/step")
        print(f"    Cost rank r:  {np.mean(all_cost_corr):.3f} ± {np.std(all_cost_corr):.3f}" 
              if all_cost_corr else "    Cost rank r:  N/A")

    # ============================================================
    # Save outputs
    # ============================================================
    print(f"\n[SAVE] Writing to {out_dir}...")
    
    # summary.csv
    import csv
    summary_path = out_dir / "trajectory_divergence_summary.csv"
    with open(summary_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"  {summary_path}")
    
    # detailed_results.json
    detail_path = out_dir / "detailed_results.json"
    with open(detail_path, "w") as f:
        # Convert numpy arrays to lists for JSON
        def to_jsonable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json.dump(all_template_results, f, indent=2, default=to_jsonable)
    print(f"  {detail_path}")
    
    # report.md
    report_path = out_dir / "trajectory_divergence_report.md"
    with open(report_path, "w") as f:
        f.write("# Trajectory Divergence Diagnosis (Hypotheses C & D)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Horizon:** {args.horizon}\n")
        f.write(f"**Action sequences:** {len(action_sequences)}\n")
        f.write(f"**Templates:** {len(templates)}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Model | Final Obj L2 (m) | Growth Rate (m/step) | Cost Rank r |\n")
        f.write("|-------|-----------------|---------------------|------------|\n")
        for row in summary_rows:
            corr_str = f"{row['cost_rank_corr_mean']:.3f}" if np.isfinite(row['cost_rank_corr_mean']) else "N/A"
            f.write(f"| {row['model']} | {row['final_obj_l2_mean']:.4f} ± {row['final_obj_l2_std']:.4f} | "
                   f"{row['growth_rate_mean']:.6f} ± {row['growth_rate_std']:.6f} | {corr_str} |\n")
        
        f.write("\n## Interpretation\n\n")
        
        for row in summary_rows:
            f.write(f"### {row['model']}\n\n")
            final_l2 = row['final_obj_l2_mean']
            growth = row['growth_rate_mean']
            corr = row['cost_rank_corr_mean']
            
            # Hypothesis C
            f.write("**Hypothesis C (Accumulation Error):**\n")
            if final_l2 > 0.1:
                f.write(f"- ⚠️ Final trajectory divergence = {final_l2:.4f}m (HIGH)\n")
                f.write(f"- Growth rate = {growth:.6f}m/step → divergence accumulates over time\n")
                f.write("- **Conclusion:** Learned rollout has significant accumulation error.\n")
            elif final_l2 > 0.02:
                f.write(f"- 📊 Final trajectory divergence = {final_l2:.4f}m (MODERATE)\n")
                f.write(f"- Growth rate = {growth:.6f}m/step\n")
                f.write("- **Conclusion:** Some accumulation error, may be acceptable for short horizons.\n")
            else:
                f.write(f"- ✅ Final trajectory divergence = {final_l2:.4f}m (LOW)\n")
                f.write("- **Conclusion:** Accumulation error is minimal.\n")
            f.write("\n")
            
            # Hypothesis D
            f.write("**Hypothesis D (Cost Function Robustness):**\n")
            if np.isfinite(corr):
                if corr > 0.7:
                    f.write(f"- ✅ Cost rank correlation = {corr:.3f} (STRONG)\n")
                    f.write("- **Conclusion:** Cost function works well with learned trajectories.\n")
                elif corr > 0.4:
                    f.write(f"- 📊 Cost rank correlation = {corr:.3f} (MODERATE)\n")
                    f.write("- **Conclusion:** Cost function partially works; some action rankings disagree.\n")
                else:
                    f.write(f"- ⚠️ Cost rank correlation = {corr:.3f} (WEAK)\n")
                    f.write("- **Conclusion:** Cost function is unreliable for learned rollouts.\n")
                    f.write("- **Action:** Consider retraining with multi-step loss or counterfactual loss.\n")
            else:
                f.write("- N/A (cost correlation not computed)\n")
            f.write("\n")
        
        f.write("## Self-Check Results\n\n")
        for k, v in sc["checks"].items():
            f.write(f"- {k}: {v}\n")
    
    print(f"  {report_path}")
    
    # self_check.json
    sc_path = out_dir / "self_check.json"
    with open(sc_path, "w") as f:
        json.dump(sc, f, indent=2)
    print(f"  {sc_path}")
    
    print(f"\n{'='*70}")
    print("Diagnosis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
