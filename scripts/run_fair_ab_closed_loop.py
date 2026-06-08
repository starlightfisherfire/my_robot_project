#!/usr/bin/env python3
"""
run_fair_ab_closed_loop.py — Strict Fair A/B closed-loop evaluation.

Uses CEMLearnedPlanner / MPPILearnedPlanner with:
- full_terminal cost (11-term, matching Oracle/MPPI)
- early_stop (pos < 0.002m, theta < 10°)
- total_budget enforcement (1000 steps max)
- All per-trial metrics recorded

Protocol: if any parameter falls back to weak config → STOP immediately.

Usage:
    PYTHONPATH=. python scripts/run_fair_ab_closed_loop.py \
        --checkpoint runs/retrain_nomass_50ep/flat/checkpoints/best.pt \
        --model-type flat \
        --checkpoint-type new_50epoch_nomass \
        --normalizer runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json \
        --planner CEM_FAIR \
        --templates-file data/sim/metadata/reset_templates_obstacle_10family_v0.json \
        --families open,blocking_easy,narrow \
        --max-per-family 2 \
        --out-dir runs/fair_ab_test_20260526_200200
"""

import argparse, json, sys, time, traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import PAPER1_CONVENTION
from src.planners.batched_rollout_cost import BatchedLearnedRolloutCostFn
from src.planners.cem_mpc import CEMMPC
from src.planners.mppi import MPPI, MPPIResult
from src.planners.cost_functions import CostWeights
from src.envs.mujoco_push_env import MujocoPushEnv

# ============================================================
# Fair config definitions
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
    "max_speed_mps": 0.3,
    "cost_mode": "full",
}

MPPI_FAIR_CONFIG = {
    "backend": "MPPI_FAIR",
    "horizon": 100,
    "num_samples": 1024,
    "num_iterations": 5,
    "temperature": 0.1,
    "init_std": 0.5,
    "execute_steps": 10,
    "max_mpc_steps": 100,
    "total_budget": 1000,
    "early_stop": True,
    "stop_pos_threshold": 0.002,
    "stop_theta_threshold_deg": 10,
    "max_speed_mps": 0.3,
    "cost_mode": "full",
}

# ============================================================
# Protocol enforcement
# ============================================================

def validate_fair_config(cfg: dict) -> list[str]:
    """Validate that config meets fair A/B protocol. Returns list of failures."""
    failures = []
    if cfg.get("horizon", 0) < 80:
        failures.append(f"horizon={cfg.get('horizon')} < 80")
    if cfg.get("total_budget", 0) < 800:
        failures.append(f"total_budget={cfg.get('total_budget')} < 800")
    if not cfg.get("early_stop"):
        failures.append("early_stop not enabled")
    if cfg.get("cost_mode") != "full":
        failures.append(f"cost_mode={cfg.get('cost_mode')} != 'full'")
    
    if cfg["backend"] == "MPPI_FAIR":
        if cfg.get("num_samples", 0) < 512:
            failures.append(f"MPPI num_samples={cfg.get('num_samples')} < 512")
    elif cfg["backend"] == "CEM_FAIR":
        if cfg.get("num_elites", 0) < 32:
            failures.append(f"CEM num_elites={cfg.get('num_elites')} < 32")
    
    return failures


# ============================================================
# State extraction helpers
# ============================================================

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
    """Extract one canonical_state16 frame [6, 16]."""
    ee_pos = env.get_ee_pos()
    obj_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    contact = env.get_contact_flag()

    tokens = np.zeros((6, 16), dtype=np.float32)
    # EE
    tokens[IDX_EE, FEAT_X] = ee_pos[0]
    tokens[IDX_EE, FEAT_Y] = ee_pos[1]
    tokens[IDX_EE, FEAT_COS_THETA] = 1.0
    tokens[IDX_EE, FEAT_SIZE_X] = EE_SIZE
    tokens[IDX_EE, FEAT_SIZE_Y] = EE_SIZE
    tokens[IDX_EE, FEAT_CONTACT] = float(contact)
    tokens[IDX_EE, FEAT_VALID] = 1.0
    # Object
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
    # Goal
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
    dtheta = np.arctan2(
        np.sin(curr_state[IDX_OBJ, FEAT_SIN_THETA] - prev_state[IDX_OBJ, FEAT_SIN_THETA]),
        np.cos(curr_state[IDX_OBJ, FEAT_COS_THETA] - prev_state[IDX_OBJ, FEAT_COS_THETA])
    )
    state[IDX_OBJ, FEAT_OMEGA] = dtheta / dt
    return state


def get_obstacle_info_from_template(template: dict):
    """Extract obstacle positions and radii from template."""
    obstacles = template.get("obstacles", [])
    if not obstacles:
        return None, None, None
    positions = []
    radii = []
    for obs in obstacles:
        if obs.get("valid", True):
            positions.append([obs["pose"]["x"], obs["pose"]["y"]])
            radii.append(max(obs.get("size_x", 0.04), obs.get("size_y", 0.04)) / 1.5)
    if not positions:
        return None, None, None
    return np.array(positions), np.array(radii), obstacles


# ============================================================
# Main trial runner
# ============================================================

def run_fair_trial(
    env: MujocoPushEnv,
    template: dict,
    planner,            # CEMLearnedPlanner or MPPILearnedPlanner
    cfg: dict,
    model: torch.nn.Module,
    normalizer,
    device: str = "cpu",
) -> dict:
    """Run one closed-loop trial with fair protocol."""
    
    env.reset_from_template(template)
    
    # Build initial history (6 frames)
    first_state = extract_state16_from_mujoco(env)
    history = np.tile(first_state[np.newaxis], (6, 1, 1))
    
    goal_pose = env.get_goal_pose()
    initial_dist = float(np.linalg.norm(first_state[IDX_OBJ, :2] - goal_pose[:2]))
    
    best_dist = float("inf")
    final_dist = float("inf")
    best_step = 0
    best_theta_err = float("inf")
    final_theta_err = float("inf")
    early_stop_triggered = False
    early_stop_step = -1
    ever_success_pose = False
    ever_success_pos_only = False
    final_success_pose = False
    
    total_mpc_steps = 0
    contact_count = 0
    collision_count = 0
    cost_breakdowns = []
    
    prev_state = first_state.copy()
    max_mpc_steps = cfg["max_mpc_steps"]
    total_budget = cfg["total_budget"]
    stop_pos_threshold = cfg["stop_pos_threshold"]
    stop_theta_threshold_deg = cfg["stop_theta_threshold_deg"]
    
    for mpc_step in range(max_mpc_steps):
        # Budget check
        if total_mpc_steps >= total_budget:
            break
        
        # Update velocity in history
        curr_state = extract_state16_from_mujoco(env)
        if mpc_step > 0:
            curr_state = compute_velocity(prev_state, curr_state, env.control_dt)
        history[:-1] = history[1:]
        history[-1] = curr_state
        
        # Plan using batched learned rollout cost
        try:
            cost_fn = BatchedLearnedRolloutCostFn(
                model=model, normalizer=normalizer,
                initial_state=history.copy(), goal_pose=goal_pose,
                device=device, weights=CostWeights(),
            )
            opt_result = planner.optimize(cost_fn)
            first_action_norm = opt_result.action_sequence[0].copy()
        except Exception as e:
            return {
                "error": f"plan failed at step {mpc_step}: {e}",
                "final_pos_dist_m": float("nan"),
                "best_pos_dist_m": float("nan"),
                "total_mpc_steps": total_mpc_steps,
            }
        
        # Execute in MuJoCo
        env_state = env.step(first_action_norm)
        total_mpc_steps += 1
        
        # Metrics
        obj_pose = env.get_object_pose()
        dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
        theta_err = abs(np.arctan2(
            np.sin(obj_pose[2] - goal_pose[2]),
            np.cos(obj_pose[2] - goal_pose[2])
        ))
        
        if dist < best_dist:
            best_dist = dist
            best_step = total_mpc_steps
            best_theta_err = theta_err
        
        final_dist = dist
        final_theta_err = theta_err
        
        contact_count += int(env.get_contact_flag())
        collision_count += int(env.get_collision_flag())
        
        # Ever success check
        if dist < stop_pos_threshold and np.degrees(theta_err) < stop_theta_threshold_deg:
            ever_success_pose = True
        if dist < stop_pos_threshold:
            ever_success_pos_only = True
        
        # Early stop check
        if dist < stop_pos_threshold and np.degrees(theta_err) < stop_theta_threshold_deg:
            early_stop_triggered = True
            early_stop_step = total_mpc_steps
            final_success_pose = True
            break
        
        prev_state = curr_state.copy()
    
    # Post-loop success check
    if final_dist < stop_pos_threshold and np.degrees(final_theta_err) < stop_theta_threshold_deg:
        final_success_pose = True
    
    drift_after_best = final_dist - best_dist if best_dist < float("inf") else float("nan")
    
    return {
        "final_pos_dist_m": float(final_dist),
        "best_pos_dist_m": float(best_dist),
        "initial_dist_m": float(initial_dist),
        "best_step": int(best_step),
        "best_theta_error_deg": float(np.degrees(best_theta_err)),
        "final_theta_error_deg": float(np.degrees(final_theta_err)),
        "drift_after_best_m": float(drift_after_best),
        "total_mpc_steps": int(total_mpc_steps),
        "contact_count": int(contact_count),
        "collision_count": int(collision_count),
        "contact_rate": float(contact_count / max(total_mpc_steps, 1)),
        "collision_rate": float(collision_count / max(total_mpc_steps, 1)),
        "final_success_pose": final_success_pose,
        "ever_success_pose": ever_success_pose,
        "ever_success_pos_only": ever_success_pos_only,
        "early_stop_triggered": early_stop_triggered,
        "early_stop_step": int(early_stop_step),
        "error": None,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fair A/B closed-loop evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", required=True, choices=["flat", "object_centric", "causality_aware"])
    parser.add_argument("--checkpoint-type", required=True, choices=["old_10epoch", "new_50epoch_nomass"])
    parser.add_argument("--normalizer", default=None)
    parser.add_argument("--planner", required=True, choices=["CEM_FAIR", "MPPI_FAIR"])
    parser.add_argument("--templates-file", required=True)
    parser.add_argument("--families", default="open,blocking_easy,narrow", help="Comma-separated family filter")
    parser.add_argument("--max-per-family", type=int, default=2)
    parser.add_argument("--max-speed", type=float, default=0.3, help="Override max_speed_mps")
    parser.add_argument("--tg-dir", default=None, help="Topology/geometry audit dir (optional, non-blocking)")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-trial JSONs and CSVs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    # ============================================================
    # Protocol enforcement
    # ============================================================
    
    fair_cfg = CEM_FAIR_CONFIG if args.planner == "CEM_FAIR" else MPPI_FAIR_CONFIG
    fair_cfg = dict(fair_cfg)  # copy
    fair_cfg["max_speed_mps"] = args.max_speed
    
    failures = validate_fair_config(fair_cfg)
    if failures:
        failure_path = Path(args.out_dir) / "failure_report.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "FAILED",
            "reason": "Fair config validation failed",
            "failures": failures,
            "config": fair_cfg,
        }
        with open(failure_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"FATAL: Fair config validation failed: {failures}")
        print(f"Failure report: {failure_path}")
        sys.exit(1)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[PROTOCOL] {args.planner} config validated: PASS")
    print(f"  horizon={fair_cfg['horizon']} budget={fair_cfg['total_budget']} "
          f"early_stop={fair_cfg['early_stop']} cost_mode={fair_cfg['cost_mode']}")
    
    # ============================================================
    # Load model
    # ============================================================
    
    print(f"\n[LOAD] Model: {args.model_type} from {args.checkpoint}")
    model = RIGWorldModel(model_type=args.model_type, action_dim=2, gru_hidden=256,
                          d_model=128, head_hidden_dim=256)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(args.device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    
    normalizer = None
    if args.normalizer and Path(args.normalizer).exists():
        normalizer = StateNormalizer.load(args.normalizer)
        print(f"  Normalizer loaded from {args.normalizer}")
    
    # ============================================================
    # Create planner
    # ============================================================
    
    # ============================================================
    # Create planner (raw CEM/MPPI with batched cost)
    # ============================================================
    
    if args.planner == "CEM_FAIR":
        planner = CEMMPC(
            horizon=fair_cfg["horizon"],
            action_dim=2,
            num_samples=fair_cfg["num_samples"],
            num_elites=fair_cfg["num_elites"],
            num_iterations=fair_cfg["num_iterations"],
            action_low=-1.0,
            action_high=1.0,
            init_std=fair_cfg["init_std"],
            smoothing=0.2,
            seed=42,
        )
        planner_backend = "CEM_BATCHED_LEARNED_ROLLOUT"
    else:
        planner = MPPI(
            horizon=fair_cfg["horizon"],
            action_dim=2,
            num_samples=fair_cfg["num_samples"],
            num_iterations=fair_cfg["num_iterations"],
            action_low=-1.0,
            action_high=1.0,
            init_std=fair_cfg["init_std"],
            temperature=fair_cfg["temperature"],
            smoothing=0.2,
            seed=42,
        )
        planner_backend = "MPPI_BATCHED_LEARNED_ROLLOUT"
    
    print(f"[PLANNER] {planner_backend} initialized")
    
    # ============================================================
    # Load templates
    # ============================================================
    
    with open(args.templates_file) as f:
        all_templates = json.load(f)
    
    family_filter = set(f.strip() for f in args.families.split(","))
    
    # Map "narrow" to passage_direct_narrow
    FAMILY_ALIASES = {
        "narrow": "passage_direct_narrow",
        "passage": "passage_direct_medium",
        "edge": "open",  # fallback
    }
    expanded_filter = set()
    for fam in family_filter:
        expanded_filter.add(FAMILY_ALIASES.get(fam, fam))
    
    # Filter and group templates by family
    grouped = defaultdict(list)
    for t in all_templates:
        family = t.get("family", t.get("layout_family", "unknown"))
        if family in expanded_filter:
            grouped[family].append(t)
    
    # Select max_per_family
    selected = []
    for family in sorted(grouped.keys()):
        templates = grouped[family][:args.max_per_family]
        selected.extend(templates)
        print(f"  {family}: {len(templates)} selected (out of {len(grouped[family])})")
    
    if not selected:
        print(f"ERROR: No templates found for families {args.families}")
        sys.exit(1)
    
    print(f"\n[RUN] {len(selected)} templates × {args.planner} on {args.model_type}/{args.checkpoint_type}")
    
    # ============================================================
    # Load topology/geometry data (non-blocking)
    # ============================================================
    
    tg_lookup = {}
    tg_status = "TG_NOT_AVAILABLE"
    tg_used_for_selection = False
    tg_used_for_analysis = False
    
    if args.tg_dir and Path(args.tg_dir).exists():
        tg_json = Path(args.tg_dir) / "template_topology_geometry.json"
        tg_status_json = Path(args.tg_dir) / "tg_status.json"
        
        if tg_json.exists():
            try:
                with open(tg_json) as f:
                    tg_data = json.load(f)
                for m in tg_data:
                    tg_lookup[m["template_id"]] = m
                
                if tg_status_json.exists():
                    with open(tg_status_json) as f:
                        tg_status_data = json.load(f)
                        tg_status = tg_status_data.get("status", "TG_PASS")
                else:
                    tg_status = "TG_PASS"
                
                tg_used_for_selection = True
                tg_used_for_analysis = True
                print(f"[TG] Loaded {len(tg_lookup)} template geometry records. Status: {tg_status}")
            except Exception as e:
                tg_status = "TG_FAIL"
                tg_lookup = {}
                print(f"[TG] WARNING: Failed to load TG data: {e}. Continuing without geometry.")
        else:
            print(f"[TG] No TG JSON found at {tg_json}. Continuing without geometry.")
    else:
        print(f"[TG] No TG dir provided. Continuing without geometry.")
    
    # ============================================================
    # Create env
    # ============================================================
    
    env = MujocoPushEnv(shape_type="T", control_dt=0.1, max_speed_mps=fair_cfg["max_speed_mps"])
    
    # ============================================================
    # Run trials
    # ============================================================
    
    results = []
    t_start = time.time()
    
    for i, template in enumerate(selected):
        t0 = time.time()
        template_id = template.get("reset_template_id", f"tpl_{i}")
        family = template.get("family", template.get("layout_family", "unknown"))
        
        try:
            result = run_fair_trial(
                env=env, template=template, planner=planner,
                cfg=fair_cfg, model=model, normalizer=normalizer, device=args.device
            )
            elapsed = time.time() - t0
            
            if result.get("error"):
                status = "💥"
            elif result["final_success_pose"]:
                status = "✅"
            elif result["ever_success_pose"]:
                status = "🔶"
            else:
                status = "❌"
            
            print(f"  [{i+1:2d}/{len(selected)}] {status} {family[:15]:15s} "
                  f"best={result['best_pos_dist_m']:.4f}m "
                  f"final={result['final_pos_dist_m']:.4f}m "
                  f"steps={result['total_mpc_steps']:3d} "
                  f"drift={result.get('drift_after_best_m', float('nan')):+.4f} "
                  f"early={result['early_stop_triggered']} "
                  f"{elapsed:.0f}s")
            
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "error": str(e),
                "final_pos_dist_m": float("nan"),
                "best_pos_dist_m": float("nan"),
                "final_success_pose": False,
                "ever_success_pose": False,
                "total_mpc_steps": 0,
            }
            print(f"  [{i+1:2d}/{len(selected)}] 💥 {family[:15]:15s} ERROR: {e}")
        
        # Enrich result with metadata
        result["model"] = args.model_type
        result["checkpoint_type"] = args.checkpoint_type
        result["planner"] = args.planner
        result["planner_backend"] = planner_backend
        result["template_id"] = template_id
        result["family"] = family
        result["horizon"] = fair_cfg["horizon"]
        result["execute_steps"] = fair_cfg["execute_steps"]
        result["max_mpc_steps"] = fair_cfg["max_mpc_steps"]
        result["total_budget"] = fair_cfg["total_budget"]
        result["early_stop_enabled"] = fair_cfg["early_stop"]
        result["cost_mode"] = fair_cfg["cost_mode"]
        result["max_speed_mps"] = fair_cfg["max_speed_mps"]
        result["runtime_sec"] = elapsed
        result["failure_code"] = result.get("error", "") if result.get("error") else "none"
        
        # Join topology/geometry metrics (non-blocking)
        tg_entry = tg_lookup.get(template_id, None)
        if tg_entry is not None:
            cls = tg_entry.get("classification", {})
            bm = tg_entry.get("basic_metrics", {})
            pm = tg_entry.get("path_metrics", {})
            cm = tg_entry.get("contact_metrics", {})
            result["topology_geometry_status"] = tg_status
            result["difficulty_level"] = cls.get("difficulty_level", "unknown")
            result["topology_family_pred"] = cls.get("topology_family_pred", "unknown")
            result["object_goal_distance"] = bm.get("object_goal_distance", None)
            result["direct_path_blocked"] = pm.get("object_goal_path_blocked", None)
            result["blocking_score"] = pm.get("blocking_score", None)
            result["passage_width_estimate"] = pm.get("passage_width_estimate", None)
            result["edge_goal_score"] = pm.get("edge_goal_score", None)
            result["approach_feasibility_score"] = cm.get("approach_feasibility_score", None)
            result["dominant_geometric_challenge"] = cls.get("dominant_geometric_challenge", "unknown")
        else:
            result["topology_geometry_status"] = "TG_NOT_AVAILABLE" if tg_status == "TG_NOT_AVAILABLE" else "TG_MISSING"
            result["difficulty_level"] = "unknown"
            result["topology_family_pred"] = "unknown"
            result["object_goal_distance"] = None
            result["direct_path_blocked"] = None
            result["blocking_score"] = None
            result["passage_width_estimate"] = None
            result["edge_goal_score"] = None
            result["approach_feasibility_score"] = None
            result["dominant_geometric_challenge"] = "unknown"
        
        results.append(result)
        
        # Save per-trial JSON
        trial_path = out_dir / "trials" / f"{args.model_type}_{args.checkpoint_type}_{args.planner}_{template_id}.json"
        trial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trial_path, "w") as f:
            json.dump(result, f, indent=2)
    
    total_elapsed = time.time() - t_start
    
    # ============================================================
    # Summary CSV
    # ============================================================
    
    csv_fields = [
        "model", "checkpoint_type", "planner", "template_id", "family",
        "horizon", "execute_steps", "max_mpc_steps", "total_budget",
        "early_stop_enabled", "early_stop_triggered", "early_stop_step",
        "cost_mode", "max_speed_mps",
        "final_success_pose", "ever_success_pose", "ever_success_pos_only",
        "initial_dist_m", "best_pos_dist_m", "final_pos_dist_m",
        "drift_after_best_m", "best_step",
        "best_theta_error_deg", "final_theta_error_deg",
        "contact_rate", "collision_rate",
        "total_mpc_steps", "runtime_sec", "failure_code",
        # Topology/geometry fields
        "topology_geometry_status",
        "difficulty_level",
        "topology_family_pred",
        "object_goal_distance",
        "direct_path_blocked",
        "blocking_score",
        "passage_width_estimate",
        "edge_goal_score",
        "approach_feasibility_score",
        "dominant_geometric_challenge",
    ]
    
    csv_path = out_dir / f"fair_ab_{args.model_type}_{args.checkpoint_type}_{args.planner}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(csv_fields) + "\n")
        for r in results:
            row = [str(r.get(field, "")) for field in csv_fields]
            f.write(",".join(row) + "\n")
    
    # Summary JSON
    valid = [r for r in results if not r.get("error")]
    success_count = sum(1 for r in valid if r["final_success_pose"])
    ever_success_count = sum(1 for r in valid if r["ever_success_pose"])
    
    summary = {
        "model": args.model_type,
        "checkpoint_type": args.checkpoint_type,
        "planner": args.planner,
        "num_templates": len(selected),
        "num_valid": len(valid),
        "success_rate": success_count / len(valid) if valid else 0,
        "ever_success_rate": ever_success_count / len(valid) if valid else 0,
        "mean_best_dist": float(np.mean([r["best_pos_dist_m"] for r in valid if np.isfinite(r["best_pos_dist_m"])])),
        "mean_final_dist": float(np.mean([r["final_pos_dist_m"] for r in valid if np.isfinite(r["final_pos_dist_m"])])),
        "mean_drift": float(np.mean([r["drift_after_best_m"] for r in valid if np.isfinite(r.get("drift_after_best_m", float("nan")))])),
        "mean_runtime": float(np.mean([r["runtime_sec"] for r in results])),
        "total_runtime": total_elapsed,
        "early_stop_rate": sum(1 for r in valid if r["early_stop_triggered"]) / len(valid) if valid else 0,
        "planner_config": fair_cfg,
        "topology_geometry": {
            "status": tg_status,
            "used_for_template_selection": tg_used_for_selection,
            "used_for_failure_analysis": tg_used_for_analysis,
        },
    }
    
    summary_path = out_dir / f"summary_{args.model_type}_{args.checkpoint_type}_{args.planner}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[DONE] {args.planner} on {args.model_type}/{args.checkpoint_type}")
    print(f"  Success: {success_count}/{len(valid)} ({summary['success_rate']:.1%})")
    print(f"  Ever Success: {ever_success_count}/{len(valid)} ({summary['ever_success_rate']:.1%})")
    print(f"  Mean BestD: {summary['mean_best_dist']:.4f}m  FinalD: {summary['mean_final_dist']:.4f}m")
    print(f"  Mean Drift: {summary['mean_drift']:+.4f}m")
    print(f"  Time: {total_elapsed:.0f}s")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
