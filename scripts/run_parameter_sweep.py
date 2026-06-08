#!/usr/bin/env python3
"""
Parameter sweep: old-style (successful) vs new-style (failed) parameters.

Tests different combinations of:
- horizon: [10, 30, 50, 100]
- num_samples: [32, 128, 512]
- execute_steps: [1, 5, 10]
- cost_mode: [simple, full]

Uses pilot_10ep model (flat, no action_embed, 10 epochs).

Usage:
    PYTHONUNBUFFERED=1 PYTHONPATH=. python -u scripts/run_parameter_sweep.py \
        --out-dir runs/parameter_sweep_$(date +%Y%m%d_%H%M%S) \
        --device cuda
"""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.cem_mpc import CEMMPC
from src.planners.batched_rollout_cost import BatchedLearnedRolloutCostFn
from src.planners.cost_functions import CostWeights, rollout_cost
from src.envs.mujoco_push_env import MujocoPushEnv
from scripts.run_learned_fair_closed_loop import (
    extract_state16_from_mujoco, compute_velocity
)

# ============================================================
# Sweep configurations
# ============================================================

# Old-style parameters (from successful videos)
OLD_STYLE = {
    "horizon": 10,
    "num_samples": 32,
    "num_elites": 8,
    "num_iterations": 5,
    "execute_steps": 1,
    "max_steps": 30,
    "cost_mode": "simple",
}

# New-style parameters (from failed comparison)
NEW_STYLE = {
    "horizon": 100,
    "num_samples": 512,
    "num_elites": 64,
    "num_iterations": 5,
    "execute_steps": 10,
    "max_steps": 1000,
    "cost_mode": "full",
}

# Sweep grid
SWEEP_CONFIGS = [
    # Old-style (should succeed)
    {"horizon": 10, "num_samples": 32, "num_elites": 8, "execute_steps": 1, "max_steps": 30, "cost_mode": "simple", "label": "old_style"},
    
    # New-style (should fail)
    {"horizon": 100, "num_samples": 512, "num_elites": 64, "execute_steps": 10, "max_steps": 1000, "cost_mode": "full", "label": "new_style"},
    
    # Hybrid: old params + new cost
    {"horizon": 10, "num_samples": 32, "num_elites": 8, "execute_steps": 1, "max_steps": 30, "cost_mode": "full", "label": "old_params_new_cost"},
    
    # Hybrid: new params + old cost
    {"horizon": 100, "num_samples": 512, "num_elites": 64, "execute_steps": 10, "max_steps": 1000, "cost_mode": "simple", "label": "new_params_old_cost"},
    
    # Medium: moderate params
    {"horizon": 30, "num_samples": 128, "num_elites": 16, "execute_steps": 1, "max_steps": 100, "cost_mode": "simple", "label": "medium_simple"},
    {"horizon": 30, "num_samples": 128, "num_elites": 16, "execute_steps": 1, "max_steps": 100, "cost_mode": "full", "label": "medium_full"},
    
    # Short horizon + many samples
    {"horizon": 10, "num_samples": 512, "num_elites": 64, "execute_steps": 1, "max_steps": 30, "cost_mode": "simple", "label": "short_horizon_many_samples"},
    
    # Long horizon + few samples
    {"horizon": 100, "num_samples": 32, "num_elites": 8, "execute_steps": 1, "max_steps": 100, "cost_mode": "simple", "label": "long_horizon_few_samples"},
]


def get_simple_cost_weights():
    """Old-style simple cost weights (3 terms)."""
    return CostWeights(
        w_pos=10.0,
        w_theta=2.0,
        w_reach=0.0,
        w_no_contact=0.0,
        w_push_alignment=0.0,
        w_collision=0.0,
        w_collision_step=0.0,
        w_proximity=0.0,
        w_early_contact=0.0,
        w_persistent_contact=0.0,
        w_action=0.05,
        w_smooth=0.0,
        w_subgoal=0.0,
    )


def get_full_cost_weights():
    """New-style full cost weights (11 terms)."""
    return CostWeights()


def run_episode(env, model, normalizer, template, cfg, device="cpu"):
    """Run one episode with given config."""
    env.reset_from_template(template)
    goal_pose = env.get_goal_pose()
    
    first_state = extract_state16_from_mujoco(env)
    history = np.tile(first_state[np.newaxis], (6, 1, 1))
    initial_dist = float(np.linalg.norm(first_state[1, :2] - goal_pose[:2]))
    
    cem = CEMMPC(
        horizon=cfg["horizon"],
        action_dim=2,
        num_samples=cfg["num_samples"],
        num_elites=cfg["num_elites"],
        num_iterations=5,
        action_low=-1.0,
        action_high=1.0,
        init_std=0.3,
        smoothing=0.2,
        seed=42,
    )
    
    weights = get_simple_cost_weights() if cfg["cost_mode"] == "simple" else get_full_cost_weights()
    
    best_dist = float("inf")
    total_steps = 0
    contact_count = 0
    prev_state = first_state.copy()
    
    t0 = time.time()
    
    max_mpc_steps = cfg["max_steps"] // cfg["execute_steps"]
    
    for mpc_step in range(max_mpc_steps):
        if total_steps >= cfg["max_steps"]:
            break
        
        curr_state = extract_state16_from_mujoco(env)
        if mpc_step > 0:
            curr_state = compute_velocity(prev_state, curr_state, env.control_dt)
        history[:-1] = history[1:]
        history[-1] = curr_state
        
        try:
            cost_fn = BatchedLearnedRolloutCostFn(
                model, normalizer, history.copy(), goal_pose,
                device=device, weights=weights
            )
            result = cem.optimize(cost_fn)
            action_sequence = result.action_sequence
        except Exception as e:
            return {"error": str(e)}
        
        for step_idx in range(min(cfg["execute_steps"], len(action_sequence))):
            if total_steps >= cfg["max_steps"]:
                break
            
            action = action_sequence[step_idx]
            env.step(action)
            total_steps += 1
            
            obj_pose = env.get_object_pose()
            dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
            if dist < best_dist:
                best_dist = dist
            
            contact_count += int(env.get_contact_flag())
            
            theta_err = abs(np.arctan2(
                np.sin(obj_pose[2] - goal_pose[2]),
                np.cos(obj_pose[2] - goal_pose[2])
            ))
            if dist < 0.002 and np.degrees(theta_err) < 10:
                return {
                    "final_pos_dist_m": dist,
                    "best_pos_dist_m": best_dist,
                    "initial_dist_m": initial_dist,
                    "total_steps": total_steps,
                    "contact_rate": contact_count / max(1, total_steps),
                    "success": True,
                    "runtime_sec": time.time() - t0,
                }
        
        prev_state = curr_state.copy()
    
    obj_pose = env.get_object_pose()
    final_dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
    
    return {
        "final_pos_dist_m": final_dist,
        "best_pos_dist_m": best_dist,
        "initial_dist_m": initial_dist,
        "total_steps": total_steps,
        "contact_rate": contact_count / max(1, total_steps),
        "success": best_dist < 0.05,
        "runtime_sec": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-per-family", type=int, default=2)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (pilot_10ep)
    print("Loading pilot_10ep model...")
    ckpt_path = REPO / "runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt"
    norm_path = REPO / "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RIGWorldModel(model_type="flat", action_dim=2, gru_hidden=256, d_model=128,
                          head_hidden_dim=256, use_action_embed=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(args.device)
    normalizer = StateNormalizer.load(str(norm_path))
    
    # Load templates
    templates_file = REPO / "data/sim/metadata/reset_templates_obstacle_10family_v0.json"
    with open(templates_file) as f:
        all_templates = json.load(f)
    
    families = ["open", "blocking_easy", "passage_direct_narrow"]
    selected = []
    for fam in families:
        fam_tmpls = [t for t in all_templates if t.get("family", "") == fam]
        selected.extend(fam_tmpls[:args.max_per_family])
    
    print(f"Selected {len(selected)} templates from {families}")
    print(f"Running {len(SWEEP_CONFIGS)} configurations × {len(selected)} templates = {len(SWEEP_CONFIGS) * len(selected)} trials")
    
    env = MujocoPushEnv(shape_type="T", control_dt=0.1, max_speed_mps=0.75)
    
    all_results = []
    total_trials = len(SWEEP_CONFIGS) * len(selected)
    trial_idx = 0
    
    for cfg in SWEEP_CONFIGS:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"  horizon={cfg['horizon']}, samples={cfg['num_samples']}, "
              f"execute_steps={cfg['execute_steps']}, max_steps={cfg['max_steps']}, "
              f"cost={cfg['cost_mode']}")
        print(f"{'='*60}")
        
        cfg_results = []
        for template in selected:
            trial_idx += 1
            template_id = template.get("reset_template_id", "unknown")
            family = template.get("family", "unknown")
            
            print(f"  [{trial_idx}/{total_trials}] {family}: {template_id[:40]}...", end=" ", flush=True)
            
            try:
                result = run_episode(env, model, normalizer, template, cfg, args.device)
                result["config"] = label
                result["template_id"] = template_id
                result["family"] = family
                result.update(cfg)
                
                cfg_results.append(result)
                all_results.append(result)
                
                success = "✅" if result.get("success") else "❌"
                print(f"{success} best={result['best_pos_dist_m']:.4f}m "
                      f"contact={result['contact_rate']:.2f} "
                      f"steps={result['total_steps']} "
                      f"({result['runtime_sec']:.1f}s)")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                all_results.append({"config": label, "template_id": template_id, "family": family, "error": str(e)})
        
        # Config summary
        valid = [r for r in cfg_results if "error" not in r]
        if valid:
            success_count = sum(1 for r in valid if r.get("success"))
            mean_best = np.mean([r["best_pos_dist_m"] for r in valid])
            mean_contact = np.mean([r["contact_rate"] for r in valid])
            print(f"\n  Summary: {success_count}/{len(valid)} success, "
                  f"mean_best={mean_best:.4f}m, mean_contact={mean_contact:.2f}")
    
    # Save results
    results_path = out_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<30} {'Success':>10} {'Best Dist':>12} {'Contact':>10} {'Steps':>8}")
    print(f"{'='*80}")
    
    for cfg in SWEEP_CONFIGS:
        label = cfg["label"]
        cfg_results = [r for r in all_results if r.get("config") == label and "error" not in r]
        if not cfg_results:
            continue
        
        success_count = sum(1 for r in cfg_results if r.get("success"))
        mean_best = np.mean([r["best_pos_dist_m"] for r in cfg_results])
        mean_contact = np.mean([r["contact_rate"] for r in cfg_results])
        mean_steps = np.mean([r["total_steps"] for r in cfg_results])
        
        print(f"{label:<30} {success_count/len(cfg_results)*100:>9.1f}% {mean_best:>11.4f}m {mean_contact:>9.2f} {mean_steps:>7.0f}")
    
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
