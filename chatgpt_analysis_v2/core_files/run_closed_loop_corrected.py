#!/usr/bin/env python3
"""run_closed_loop_corrected.py — Corrected MuJoCo closed-loop with learned MPC.

Fixes from 2026-05-24 first attempt:
1. Action convention: planner → physical velocity → model
2. EE update: displacement = a_phys * control_dt
3. CEM config: increased horizon/samples/iterations
4. MPPI support (if available)

Usage:
    PYTHONPATH=. python scripts/run_closed_loop_corrected.py \
        --model-type flat \
        --checkpoint runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt \
        --normalizer runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json \
        --planner cem \
        --max-templates 3 \
        --out runs/closed_loop_action_planner_fix/open_space/cem/flat.json
"""

import argparse, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.action_conventions import PAPER1_CONVENTION, get_convention
from src.planners.learned_planner_adapter import CEMLearnedPlanner, PlannerResult


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

    return tokens


def compute_velocity(prev_state: np.ndarray, curr_state: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Compute velocity by finite difference."""
    state = curr_state.copy()
    state[IDX_EE, FEAT_VX] = (curr_state[IDX_EE, FEAT_X] - prev_state[IDX_EE, FEAT_X]) / dt
    state[IDX_EE, FEAT_VY] = (curr_state[IDX_EE, FEAT_Y] - prev_state[IDX_EE, FEAT_Y]) / dt
    state[IDX_OBJ, FEAT_VX] = (curr_state[IDX_OBJ, FEAT_X] - prev_state[IDX_OBJ, FEAT_X]) / dt
    state[IDX_OBJ, FEAT_VY] = (curr_state[IDX_OBJ, FEAT_Y] - prev_state[IDX_OBJ, FEAT_Y]) / dt
    return state


# === Main closed-loop ===

def run_one_template(env, planner, template, max_mpc_steps, convention):
    """Run closed-loop on one template."""
    env.reset_from_template(template)

    # Build initial history (6 frames, repeat first)
    first_state = extract_state16_from_mujoco(env)
    history = np.tile(first_state[np.newaxis], (6, 1, 1))  # [6, 6, 16]

    goal_pose = env.get_goal_pose()
    initial_dist = float(np.linalg.norm(
        env.get_object_pose()[:2] - goal_pose[:2]
    ))
    
    best_dist = initial_dist
    final_dist = initial_dist
    total_steps = 0
    contact_count = 0
    collision_count = 0

    prev_state = first_state.copy()
    
    # Track action norms
    first_action_norms = []
    model_action_norms = []
    state_disp_norms = []
    planned_costs = []
    zero_costs = []

    for mpc_step in range(max_mpc_steps):
        # Update velocity in history
        curr_state = extract_state16_from_mujoco(env)
        if mpc_step > 0:
            curr_state = compute_velocity(prev_state, curr_state, env.control_dt)
        history[:-1] = history[1:]
        history[-1] = curr_state

        # Plan using learned rollout
        result: PlannerResult = planner.plan(history.copy(), goal_pose)
        
        # Record metrics
        first_action_norms.append(float(np.linalg.norm(result.first_action_norm)))
        model_action_norms.append(float(np.linalg.norm(result.first_action_phys)))
        disp = convention.model_action_to_state_displacement(result.first_action_phys)
        state_disp_norms.append(float(np.linalg.norm(disp)))
        planned_costs.append(result.planned_cost)
        zero_costs.append(result.zero_cost)

        # Execute in MuJoCo (normalized action)
        env_state = env.step(result.first_action_norm)

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
        "success": bool(final_dist < 0.02 and np.degrees(theta_err) < 10.0),
        "initial_dist_m": float(initial_dist),
        "final_dist_m": float(final_dist),
        "best_dist_m": float(best_dist),
        "improved": bool(best_dist < initial_dist),
        "total_steps": total_steps,
        "contact_count": contact_count,
        "contact_rate": contact_count / max(total_steps, 1),
        "collision_count": collision_count,
        "collision_rate": collision_count / max(total_steps, 1),
        "mean_first_action_norm": float(np.mean(first_action_norms)) if first_action_norms else 0,
        "mean_model_action_norm": float(np.mean(model_action_norms)) if model_action_norms else 0,
        "mean_state_disp_norm": float(np.mean(state_disp_norms)) if state_disp_norms else 0,
        "mean_planned_cost": float(np.mean(planned_costs)) if planned_costs else 0,
        "mean_zero_cost": float(np.mean(zero_costs)) if zero_costs else 0,
        "cost_improvement": float(np.mean(zero_costs) - np.mean(planned_costs)) if planned_costs else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", required=True, choices=["flat", "object_centric", "causality_aware"])
    parser.add_argument("--normalizer", default=None)
    parser.add_argument("--planner", default="cem", choices=["cem"])
    parser.add_argument("--max-templates", type=int, default=3)
    parser.add_argument("--max-mpc-steps", type=int, default=30)
    parser.add_argument("--template-family", default="open_space",
                        help="Template family to use (open_space, blocking, passage, etc.)")
    parser.add_argument("--max-speed", type=float, default=0.5,
                        help="max_speed_mps (default: 0.5 from MPPI best config)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Get convention
    convention = get_convention("paper1_push_v1")
    convention.max_speed_mps = args.max_speed
    
    print(f"Action convention: {convention.describe()}")

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
        template_path = REPO / "data/sim/metadata/reset_templates_obstacle_10family_v0.json"
    if not template_path.exists():
        print("No templates found!")
        sys.exit(1)

    with open(template_path) as f:
        all_templates = json.load(f)

    # Filter by family (try both 'family' and 'layout_family' keys)
    family_templates = [t for t in all_templates if t.get("family", "") == args.template_family]
    if not family_templates:
        family_templates = [t for t in all_templates if t.get("layout_family", "") == args.template_family]
    if not family_templates:
        # Try partial match
        family_templates = [t for t in all_templates if args.template_family in t.get("family", "") or args.template_family in t.get("layout_family", "")]
    if not family_templates:
        print(f"No templates found for family: {args.template_family}")
        available = set(t.get('family', t.get('layout_family', 'unknown')) for t in all_templates)
        print(f"Available families: {available}")
        sys.exit(1)

    templates = family_templates[:args.max_templates]
    print(f"Using {len(templates)} templates from family: {args.template_family}")

    # Create env
    from src.envs.mujoco_push_env import MujocoPushEnv
    env = MujocoPushEnv(shape_type="T", control_dt=convention.control_dt, 
                        max_speed_mps=convention.max_speed_mps)

    # Create planner
    if args.planner == "cem":
        planner = CEMLearnedPlanner(
            model=model,
            normalizer=normalizer,
            convention=convention,
            horizon=20,
            num_samples=512,
            num_elites=64,
            num_iterations=5,
            init_std=0.3,
            device="cpu",
        )
    else:
        raise ValueError(f"Unknown planner: {args.planner}")

    # Run closed-loop
    results = []
    for i, template in enumerate(templates):
        t0 = time.time()
        try:
            result = run_one_template(env, planner, template, args.max_mpc_steps, convention)
            elapsed = time.time() - t0
            result["template_idx"] = i
            result["template_id"] = template.get("template_id", template.get("reset_template_id", f"template_{i}"))
            result["family"] = template.get("family", template.get("layout_family", args.template_family))
            result["runtime_sec"] = elapsed
            result["error"] = None
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "template_idx": i,
                "template_id": template.get("template_id", template.get("reset_template_id", f"template_{i}")),
                "family": template.get("family", template.get("layout_family", args.template_family)),
                "runtime_sec": elapsed,
                "error": str(e),
                "success": False,
                "final_dist_m": float("nan"),
                "best_dist_m": float("nan"),
            }
        results.append(result)

        status = "✅" if result.get("success") else "❌"
        improved = "📈" if result.get("improved") else "📉"
        dist = result.get("final_dist_m", float("nan"))
        best = result.get("best_dist_m", float("nan"))
        init_d = result.get("initial_dist_m", float("nan"))
        print(f"  [{i+1}/{len(templates)}] {status} {improved} "
              f"init={init_d:.4f}m final={dist:.4f}m best={best:.4f}m "
              f"steps={result.get('total_steps','-')} time={elapsed:.1f}s")

    # Summary
    successes = sum(1 for r in results if r.get("success"))
    improved = sum(1 for r in results if r.get("improved"))
    final_dists = [r["final_dist_m"] for r in results if np.isfinite(r.get("final_dist_m", float("nan")))]
    best_dists = [r["best_dist_m"] for r in results if np.isfinite(r.get("best_dist_m", float("nan")))]
    init_dists = [r["initial_dist_m"] for r in results if np.isfinite(r.get("initial_dist_m", float("nan")))]

    summary = {
        "eval_type": "mujoco_closed_loop_corrected",
        "note": "Corrected action convention + improved CEM config",
        "model_type": args.model_type,
        "planner_backend": "CEM_BEST_LEARNED_ROLLOUT",
        "planner_config": planner.config,
        "convention": convention.describe(),
        "template_family": args.template_family,
        "num_templates": len(results),
        "success_count": successes,
        "success_rate": successes / len(results) if results else 0,
        "improved_count": improved,
        "improved_rate": improved / len(results) if results else 0,
        "mean_initial_dist": float(np.mean(init_dists)) if init_dists else float("nan"),
        "mean_final_dist": float(np.mean(final_dists)) if final_dists else float("nan"),
        "mean_best_dist": float(np.mean(best_dists)) if best_dists else float("nan"),
        "min_best_dist": float(np.min(best_dists)) if best_dists else float("nan"),
        "mean_distance_improvement": float(np.mean(init_dists) - np.mean(best_dists)) if init_dists and best_dists else float("nan"),
        "mean_contact_rate": float(np.mean([r.get("contact_rate", 0) for r in results])),
        "mean_collision_rate": float(np.mean([r.get("collision_rate", 0) for r in results])),
        "mean_runtime": float(np.mean([r["runtime_sec"] for r in results])),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary: success={successes}/{len(results)}, improved={improved}/{len(results)}")
    print(f"Mean: initial={summary['mean_initial_dist']:.4f}m → "
          f"final={summary['mean_final_dist']:.4f}m, best={summary['mean_best_dist']:.4f}m")
    print(f"Mean distance improvement: {summary['mean_distance_improvement']:.4f}m")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
