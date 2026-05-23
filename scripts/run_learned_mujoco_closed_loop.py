#!/usr/bin/env python3
"""run_learned_mujoco_closed_loop.py — MuJoCo closed-loop learned MPC smoke.

Uses trained learned rollout model to plan actions in real MuJoCo.

Action convention:
  MuJoCo env.step expects normalized action in [-1, 1].
  Internally: velocity_cmd = action * max_speed_mps.
  Training data actions_physical = normalized_action * max_speed_mps.
  So we plan in normalized [-1,1] and let MuJoCo handle the scaling.

Usage:
    PYTHONPATH=. python scripts/run_learned_mujoco_closed_loop.py \
        --checkpoint runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt \
        --model-type flat \
        --normalizer runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json \
        --max-templates 3 \
        --out runs/pilot_state16_mppi_stage2c/closed_loop_smoke/flat_smoke.json
"""

import argparse, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights, rollout_cost


# === State extraction from MuJoCo ===

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


def extract_state16_from_mujoco(env) -> np.ndarray:
    """Extract one canonical_state16 frame [6, 16] from MuJoCo env."""
    ee_pos = env.get_ee_pos()
    obj_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    contact = env.get_contact_flag()

    tokens = np.zeros((6, 16), dtype=np.float32)

    # Token 0: EE
    tokens[IDX_EE, FEAT_X] = ee_pos[0]
    tokens[IDX_EE, FEAT_Y] = ee_pos[1]
    tokens[IDX_EE, FEAT_COS_THETA] = 1.0
    tokens[IDX_EE, FEAT_SIZE_X] = EE_SIZE
    tokens[IDX_EE, FEAT_SIZE_Y] = EE_SIZE
    tokens[IDX_EE, FEAT_CONTACT] = float(contact)
    tokens[IDX_EE, FEAT_VALID] = 1.0

    # Token 1: Object
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

    # Token 2: Goal
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

    # Obstacles from template (if available)
    # For now, leave tokens 3-5 as zeros (no obstacles in open_space)

    return tokens


def compute_velocity(prev_state: np.ndarray, curr_state: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Compute velocity by finite difference. Updates vx,vy,omega for EE and Object."""
    state = curr_state.copy()
    # EE velocity
    state[IDX_EE, FEAT_VX] = (curr_state[IDX_EE, FEAT_X] - prev_state[IDX_EE, FEAT_X]) / dt
    state[IDX_EE, FEAT_VY] = (curr_state[IDX_EE, FEAT_Y] - prev_state[IDX_EE, FEAT_Y]) / dt
    # Object velocity
    state[IDX_OBJ, FEAT_VX] = (curr_state[IDX_OBJ, FEAT_X] - prev_state[IDX_OBJ, FEAT_X]) / dt
    state[IDX_OBJ, FEAT_VY] = (curr_state[IDX_OBJ, FEAT_Y] - prev_state[IDX_OBJ, FEAT_Y]) / dt
    dtheta = np.arctan2(
        np.sin(curr_state[IDX_OBJ, FEAT_SIN_THETA] - prev_state[IDX_OBJ, FEAT_SIN_THETA]),
        np.cos(curr_state[IDX_OBJ, FEAT_COS_THETA] - prev_state[IDX_OBJ, FEAT_COS_THETA])
    )  # approximate
    state[IDX_OBJ, FEAT_OMEGA] = dtheta / dt
    return state


# === Learned rollout for CEM ===

class LearnedRolloutCostFn:
    """Cost function using learned rollout for CEM planning."""

    def __init__(self, model, normalizer, initial_state, goal_pose, device="cpu"):
        self.model = model
        self.normalizer = normalizer
        self.initial_state = initial_state
        self.goal_pose = goal_pose
        self.device = device
        self.weights = CostWeights()

    def __call__(self, action_sequence: np.ndarray) -> float:
        """Evaluate cost of an action sequence using learned rollout."""
        action_sequence = np.asarray(action_sequence, dtype=np.float64)
        horizon = len(action_sequence)

        # Build rollout
        current_state = self.initial_state.copy()
        obj_xy = current_state[-1, IDX_OBJ, [FEAT_X, FEAT_Y]].copy()
        obj_theta = np.arctan2(current_state[-1, IDX_OBJ, FEAT_SIN_THETA],
                               current_state[-1, IDX_OBJ, FEAT_COS_THETA])
        object_traj = [np.array([obj_xy[0], obj_xy[1], obj_theta])]
        ee_traj = [current_state[-1, IDX_EE, [FEAT_X, FEAT_Y]].copy()]

        for t in range(horizon):
            # Normalize state for model
            state_norm = current_state.copy()
            if self.normalizer is not None:
                state_norm = self.normalizer.transform(state_norm)

            # Model prediction
            state_t = torch.from_numpy(state_norm[np.newaxis]).float().to(self.device)
            action_t = torch.from_numpy(action_sequence[t:t+1].astype(np.float32)).float().to(self.device)

            with torch.no_grad():
                out = self.model(state_t, action_t)

            delta = out["pred_delta"].cpu().numpy()[0]  # [dx, dy, dtheta]

            # Update object pose
            obj_xy = object_traj[-1][:2] + delta[:2]
            obj_theta = np.arctan2(
                np.sin(object_traj[-1][2] + delta[2]),
                np.cos(object_traj[-1][2] + delta[2])
            )
            object_traj.append(np.array([obj_xy[0], obj_xy[1], obj_theta]))

            # Update EE
            ee_xy = ee_traj[-1] + action_sequence[t]
            ee_traj.append(ee_xy)

            # Update state for next step
            new_state = current_state.copy()
            # Shift history
            new_state[:-1] = new_state[1:]
            # Update newest frame
            new_state[-1, IDX_OBJ, FEAT_X] = obj_xy[0]
            new_state[-1, IDX_OBJ, FEAT_Y] = obj_xy[1]
            new_state[-1, IDX_OBJ, FEAT_SIN_THETA] = np.sin(obj_theta)
            new_state[-1, IDX_OBJ, FEAT_COS_THETA] = np.cos(obj_theta)
            new_state[-1, IDX_EE, FEAT_X] = ee_xy[0]
            new_state[-1, IDX_EE, FEAT_Y] = ee_xy[1]
            current_state = new_state

        object_traj = np.array(object_traj)
        ee_traj = np.array(ee_traj)

        return rollout_cost(
            predicted_object_poses=object_traj,
            ee_positions=ee_traj,
            action_sequence=action_sequence,
            goal_pose=self.goal_pose,
            weights=self.weights,
        )


# === Main closed-loop ===

def run_one_template(env, model, normalizer, template, cem, max_mpc_steps, device="cpu"):
    """Run closed-loop on one template."""
    env.reset_from_template(template)

    # Build initial history (6 frames, repeat first)
    first_state = extract_state16_from_mujoco(env)
    history = np.tile(first_state[np.newaxis], (6, 1, 1))  # [6, 6, 16]

    goal_pose = env.get_goal_pose()
    best_dist = float("inf")
    final_dist = float("inf")
    total_steps = 0
    contact_count = 0
    collision_count = 0

    prev_state = first_state.copy()

    for mpc_step in range(max_mpc_steps):
        # Update velocity in history
        curr_state = extract_state16_from_mujoco(env)
        if mpc_step > 0:
            curr_state = compute_velocity(prev_state, curr_state, env.control_dt)
        # Update history
        history[:-1] = history[1:]
        history[-1] = curr_state

        # Build cost function
        cost_fn = LearnedRolloutCostFn(model, normalizer, history.copy(), goal_pose, device)

        # Plan
        first_action, cem_result = cem.plan(cost_fn)

        # Execute in MuJoCo (normalized action)
        env_state = env.step(first_action)

        # Track metrics
        obj_pose = env.get_object_pose()
        dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
        best_dist = min(best_dist, dist)
        final_dist = dist
        total_steps += 1
        contact_count += int(env.get_contact_flag())
        collision_count += int(env.get_collision_flag())

        prev_state = curr_state.copy()

        # Check success
        theta_err = abs(np.arctan2(
            np.sin(obj_pose[2] - goal_pose[2]),
            np.cos(obj_pose[2] - goal_pose[2])
        ))
        if dist < 0.02 and np.degrees(theta_err) < 10.0:
            break

    return {
        "final_pos_dist_m": float(final_dist),
        "best_pos_dist_m": float(best_dist),
        "total_steps": total_steps,
        "contact_count": contact_count,
        "collision_count": collision_count,
        "success": bool(final_dist < 0.02 and np.degrees(theta_err) < 10.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", required=True, choices=["flat", "object_centric", "causality_aware"])
    parser.add_argument("--normalizer", default=None)
    parser.add_argument("--max-templates", type=int, default=3)
    parser.add_argument("--max-mpc-steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-elites", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--max-speed", type=float, default=0.3, help="MuJoCo max_speed_mps")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_type}")
    model = RIGWorldModel(model_type=args.model_type, action_dim=2, gru_hidden=256,
                          d_model=128, head_hidden_dim=256)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = None
    if args.normalizer and Path(args.normalizer).exists():
        normalizer = StateNormalizer.load(args.normalizer)

    # Load templates
    template_path = REPO / "data/sim/metadata/reset_templates_v0.json"
    if not template_path.exists():
        # Try obstacle templates
        template_path = REPO / "data/sim/metadata/reset_templates_obstacle_10family_v0.json"
    if not template_path.exists():
        print("No templates found!")
        sys.exit(1)

    with open(template_path) as f:
        all_templates = json.load(f)

    # Filter to open_space / simple templates
    open_templates = [t for t in all_templates if t.get("family", "") in ("open_space", "open", "mild_offset")]
    if not open_templates:
        open_templates = all_templates[:10]  # fallback

    templates = open_templates[:args.max_templates]
    print(f"Using {len(templates)} templates")

    # Create env
    from src.envs.mujoco_push_env import MujocoPushEnv
    env = MujocoPushEnv(shape_type="T", control_dt=0.1, max_speed_mps=args.max_speed)

    # Create CEM planner
    cem = CEMMPC(
        horizon=args.horizon,
        action_dim=2,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        action_low=-1.0,
        action_high=1.0,
        init_std=args.init_std,
        smoothing=0.2,
        seed=42,
    )

    # Run closed-loop
    results = []
    for i, template in enumerate(templates):
        t0 = time.time()
        try:
            result = run_one_template(env, model, normalizer, template, cem,
                                      args.max_mpc_steps, "cpu")
            elapsed = time.time() - t0
            result["template_idx"] = i
            result["family"] = template.get("family", "unknown")
            result["runtime_sec"] = elapsed
            result["error"] = None
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "template_idx": i, "family": template.get("family", "unknown"),
                "runtime_sec": elapsed, "error": str(e),
                "success": False, "final_pos_dist_m": float("nan"),
                "best_pos_dist_m": float("nan"),
            }
        results.append(result)

        status = "✅" if result.get("success") else "❌"
        dist = result.get("final_pos_dist_m", float("nan"))
        best = result.get("best_pos_dist_m", float("nan"))
        print(f"  [{i+1}/{len(templates)}] {status} dist={dist:.4f}m best={best:.4f}m "
              f"steps={result.get('total_steps','-')} time={elapsed:.1f}s")

    # Summary
    successes = sum(1 for r in results if r.get("success"))
    final_dists = [r["final_pos_dist_m"] for r in results if np.isfinite(r.get("final_pos_dist_m", float("nan")))]
    best_dists = [r["best_pos_dist_m"] for r in results if np.isfinite(r.get("best_pos_dist_m", float("nan")))]

    summary = {
        "eval_type": "mujoco_closed_loop_learned_mpc",
        "note": "This is real MuJoCo closed-loop evaluation with learned rollout planning.",
        "model_type": args.model_type,
        "planner_backend": "CEM_FALLBACK",
        "num_templates": len(results),
        "success_count": successes,
        "success_rate": successes / len(results) if results else 0,
        "mean_final_dist": float(np.mean(final_dists)) if final_dists else float("nan"),
        "mean_best_dist": float(np.mean(best_dists)) if best_dists else float("nan"),
        "min_best_dist": float(np.min(best_dists)) if best_dists else float("nan"),
        "mean_runtime": float(np.mean([r["runtime_sec"] for r in results])),
        "cem_config": {
            "horizon": args.horizon, "num_samples": args.num_samples,
            "num_elites": args.num_elites, "num_iterations": args.num_iterations,
            "init_std": args.init_std,
        },
        "max_speed_mps": args.max_speed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)

    print(f"\nSummary: success={successes}/{len(results)}, "
          f"mean_final={summary['mean_final_dist']:.4f}m, "
          f"mean_best={summary['mean_best_dist']:.4f}m")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
