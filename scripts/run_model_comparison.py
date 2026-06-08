#!/usr/bin/env python3
"""
Model Comparison Experiment — Compare different learned model versions.

Uses full MPPI/CEM parameters with the current cost function (11-term).
Evaluates on open, blocking_easy, and passage_direct_narrow families.

Usage:
    PYTHONPATH=. python scripts/run_model_comparison.py \
        --out-dir runs/model_comparison_$(date +%Y%m%d_%H%M%S) \
        --max-per-family 2
"""

import argparse, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.batched_rollout_cost import BatchedLearnedRolloutCostFn
from src.planners.cem_mpc import CEMMPC
from src.planners.mppi import MPPI
from src.planners.cost_functions import CostWeights
from src.envs.mujoco_push_env import MujocoPushEnv
from scripts.run_learned_fair_closed_loop import (
    extract_state16_from_mujoco, compute_velocity, validate_fair_config
)

# ============================================================
# Model definitions
# ============================================================

MODELS = {
    "pilot_10ep": {
        "checkpoint": "runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "description": "Initial training (10 epochs)",
        "use_action_embed": False,
    },
    "nomass_50ep": {
        "checkpoint": "runs/retrain_nomass_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "description": "No mass/friction (50 epochs)",
        "use_action_embed": False,
    },
    "action_embed_50ep": {
        "checkpoint": "runs/retrain_action_embed_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "description": "Action embed (50 epochs)",
        "use_action_embed": True,
    },
    "action_embed_mass_50ep": {
        "checkpoint": "runs/retrain_action_embed_mass_50ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "description": "Action embed + mass (50 epochs)",
        "use_action_embed": True,
    },
    "gru_action_embed_30ep": {
        "checkpoint": "runs/retrain_gru_action_embed_30ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "description": "GRU + action embed (30 epochs)",
        "use_action_embed": True,
    },
    "lastframe_30ep": {
        "checkpoint": "runs/retrain_lastframe_30ep/flat/checkpoints/best.pt",
        "normalizer": "runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json",
        "model_type": "flat",
        "description": "LastFrame encoder (30 epochs)",
        "use_action_embed": True,
    },
}

# ============================================================
# Fair config (CEM)
# ============================================================

CEM_FAIR_CONFIG = {
    "backend": "CEM_FAIR",
    "horizon": 100,
    "num_samples": 512,
    "num_elites": 64,
    "num_iterations": 5,
    "init_std": 0.3,
    "execute_steps": 10,
    "max_mpc_steps": 100,
    "total_budget": 1000,
    "early_stop": True,
    "stop_pos_threshold": 0.002,
    "stop_theta_threshold_deg": 10,
    "max_speed_mps": 0.75,
    "cost_mode": "full",
}

# ============================================================
# Helper functions
# ============================================================

def load_model(model_cfg, device="cpu"):
    """Load model and normalizer."""
    ckpt_path = REPO / model_cfg["checkpoint"]
    norm_path = REPO / model_cfg["normalizer"]
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"  Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Auto-detect action_embed
    has_action_embed = any("action_embed" in k for k in ckpt["model_state_dict"].keys())
    use_action_embed = model_cfg.get("use_action_embed", has_action_embed)
    
    model = RIGWorldModel(
        model_type=model_cfg["model_type"],
        action_dim=2,
        gru_hidden=256,
        d_model=128,
        head_hidden_dim=256,
        use_action_embed=use_action_embed,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    
    normalizer = StateNormalizer.load(str(norm_path))
    
    return model, normalizer


def run_episode(env, model, normalizer, template, cfg, device="cpu"):
    """Run one episode with learned model planner."""
    env.reset_from_template(template)
    goal_pose = env.get_goal_pose()
    
    # Initialize history
    first_state = extract_state16_from_mujoco(env)
    history = np.tile(first_state[np.newaxis], (6, 1, 1))
    initial_dist = float(np.linalg.norm(first_state[1, :2] - goal_pose[:2]))
    
    # Create planner
    cem = CEMMPC(
        horizon=cfg["horizon"],
        action_dim=2,
        num_samples=cfg["num_samples"],
        num_elites=cfg["num_elites"],
        num_iterations=cfg["num_iterations"],
        action_low=-1.0,
        action_high=1.0,
        init_std=cfg["init_std"],
        smoothing=0.2,
        seed=42,
    )
    
    # Run episode
    best_dist = float("inf")
    best_step = 0
    total_mpc_steps = 0
    contact_count = 0
    collision_count = 0
    prev_state = first_state.copy()
    execute_steps = cfg["execute_steps"]
    
    for mpc_step in range(cfg["max_mpc_steps"]):
        if total_mpc_steps >= cfg["total_budget"]:
            break
        
        # Update history
        curr_state = extract_state16_from_mujoco(env)
        if mpc_step > 0:
            curr_state = compute_velocity(prev_state, curr_state, env.control_dt)
        history[:-1] = history[1:]
        history[-1] = curr_state
        
        # Plan
        try:
            cost_fn = BatchedLearnedRolloutCostFn(
                model=model,
                normalizer=normalizer,
                initial_state=history.copy(),
                goal_pose=goal_pose,
                device=device,
                weights=CostWeights(),
            )
            result = cem.optimize(cost_fn)
            action_sequence = result.action_sequence
        except Exception as e:
            return {"error": str(e), "final_pos_dist_m": float("nan"), "best_pos_dist_m": float("nan")}
        
        # Execute multiple steps
        for step_idx in range(min(execute_steps, len(action_sequence))):
            if total_mpc_steps >= cfg["total_budget"]:
                break
            
            action = action_sequence[step_idx]
            env.step(action)
            total_mpc_steps += 1
            
            # Metrics
            obj_pose = env.get_object_pose()
            dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
            
            if dist < best_dist:
                best_dist = dist
                best_step = total_mpc_steps
            
            contact_count += int(env.get_contact_flag())
            collision_count += int(env.get_collision_flag())
            
            # Early stop
            theta_err = abs(np.arctan2(
                np.sin(obj_pose[2] - goal_pose[2]),
                np.cos(obj_pose[2] - goal_pose[2])
            ))
            if dist < cfg["stop_pos_threshold"] and np.degrees(theta_err) < cfg["stop_theta_threshold_deg"]:
                return {
                    "final_pos_dist_m": dist,
                    "best_pos_dist_m": best_dist,
                    "initial_dist_m": initial_dist,
                    "best_step": best_step,
                    "total_mpc_steps": total_mpc_steps,
                    "contact_rate": contact_count / total_mpc_steps,
                    "collision_rate": collision_count / total_mpc_steps,
                    "success": True,
                    "early_stop": True,
                }
        
        prev_state = curr_state.copy()
    
    # Final metrics
    obj_pose = env.get_object_pose()
    final_dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
    
    return {
        "final_pos_dist_m": final_dist,
        "best_pos_dist_m": best_dist,
        "initial_dist_m": initial_dist,
        "best_step": best_step,
        "total_mpc_steps": total_mpc_steps,
        "contact_rate": contact_count / max(1, total_mpc_steps),
        "collision_rate": collision_count / max(1, total_mpc_steps),
        "success": best_dist < 0.05,  # 5cm threshold
        "early_stop": False,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-per-family", type=int, default=2)
    parser.add_argument("--templates-file", default="data/sim/metadata/reset_templates_obstacle_10family_v0.json")
    parser.add_argument("--families", default="open,blocking_easy,passage_direct_narrow")
    parser.add_argument("--models", default=None, help="Comma-separated model names to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load templates
    with open(args.templates_file) as f:
        all_templates = json.load(f)
    
    families = args.families.split(",")
    selected_templates = []
    for fam in families:
        fam_tmpls = [t for t in all_templates if t.get("family", "") == fam]
        selected_templates.extend(fam_tmpls[:args.max_per_family])
    
    print(f"Selected {len(selected_templates)} templates from families: {families}")
    
    # Select models
    if args.models:
        model_names = args.models.split(",")
    else:
        model_names = list(MODELS.keys())
    
    print(f"Evaluating models: {model_names}")
    
    # Validate config
    cfg = CEM_FAIR_CONFIG.copy()
    failures = validate_fair_config(cfg)
    if failures:
        print(f"❌ Config validation failed: {failures}")
        sys.exit(1)
    
    # Create environment
    env = MujocoPushEnv(shape_type="T", control_dt=0.1, max_speed_mps=cfg["max_speed_mps"])
    
    # Run experiments
    all_results = []
    total_trials = len(model_names) * len(selected_templates)
    trial_idx = 0
    
    for model_name in model_names:
        if model_name not in MODELS:
            print(f"⚠️ Unknown model: {model_name}, skipping")
            continue
        
        model_cfg = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name} — {model_cfg['description']}")
        print(f"{'='*60}")
        
        try:
            model, normalizer = load_model(model_cfg, args.device)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
        
        model_results = []
        for template in selected_templates:
            trial_idx += 1
            template_id = template.get("reset_template_id", "unknown")
            family = template.get("family", "unknown")
            
            print(f"  [{trial_idx}/{total_trials}] {family}: {template_id[:40]}...", end=" ", flush=True)
            
            try:
                t0 = time.time()
                result = run_episode(env, model, normalizer, template, cfg, args.device)
                elapsed = time.time() - t0
                
                result["model"] = model_name
                result["template_id"] = template_id
                result["family"] = family
                result["runtime_sec"] = elapsed
                
                model_results.append(result)
                all_results.append(result)
                
                success = "✅" if result.get("success") else "❌"
                print(f"{success} dist={result['best_pos_dist_m']:.4f}m contact={result['contact_rate']:.2f} ({elapsed:.0f}s)")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    "model": model_name,
                    "template_id": template_id,
                    "family": family,
                    "error": str(e),
                })
        
        # Model summary
        valid_results = [r for r in model_results if "error" not in r]
        if valid_results:
            success_count = sum(1 for r in valid_results if r.get("success"))
            mean_best_dist = np.mean([r["best_pos_dist_m"] for r in valid_results])
            mean_contact = np.mean([r["contact_rate"] for r in valid_results])
            print(f"\n  Summary: {success_count}/{len(valid_results)} success, "
                  f"mean_best_dist={mean_best_dist:.4f}m, mean_contact={mean_contact:.2f}")
    
    # Save results
    results_path = out_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    summary = {}
    for model_name in model_names:
        model_results = [r for r in all_results if r.get("model") == model_name and "error" not in r]
        if not model_results:
            continue
        
        success_count = sum(1 for r in model_results if r.get("success"))
        mean_best_dist = np.mean([r["best_pos_dist_m"] for r in model_results])
        mean_final_dist = np.mean([r["final_pos_dist_m"] for r in model_results])
        mean_contact = np.mean([r["contact_rate"] for r in model_results])
        mean_runtime = np.mean([r["runtime_sec"] for r in model_results])
        
        summary[model_name] = {
            "total": len(model_results),
            "success": success_count,
            "success_rate": success_count / len(model_results),
            "mean_best_dist": mean_best_dist,
            "mean_final_dist": mean_final_dist,
            "mean_contact_rate": mean_contact,
            "mean_runtime": mean_runtime,
        }
        
        print(f"\n{model_name} ({MODELS[model_name]['description']}):")
        print(f"  Success rate: {success_count}/{len(model_results)} ({success_count/len(model_results)*100:.1f}%)")
        print(f"  Mean best dist: {mean_best_dist:.4f}m")
        print(f"  Mean final dist: {mean_final_dist:.4f}m")
        print(f"  Mean contact rate: {mean_contact:.2f}")
        print(f"  Mean runtime: {mean_runtime:.1f}s")
    
    # Save summary
    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Success':>10} {'Best Dist':>12} {'Contact':>10} {'Runtime':>10}")
    print(f"{'='*80}")
    for model_name in model_names:
        if model_name in summary:
            s = summary[model_name]
            print(f"{model_name:<25} {s['success_rate']*100:>9.1f}% {s['mean_best_dist']:>11.4f}m {s['mean_contact_rate']:>9.2f} {s['mean_runtime']:>9.1f}s")
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
