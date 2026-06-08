#!/usr/bin/env python3
"""
Quick cost structure ablation — current vs staged_full.
Reduced MPPI params for speed.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.envs.toy_push_env import ToyPushEnv
from src.planners.mppi import MPPI
from src.planners.oracle_rollout import rollout_action_sequence
from src.planners.cost_modes import rollout_cost_with_mode

MPPI_CFG = {"temperature": 0.2, "num_samples": 256, "horizon": 20, "init_std": 0.5, "smoothing": 0.2}
SUCCESS_THRESH = 0.10
MAX_STEPS = 40

SCENARIOS = [
    {"obj": [0.20, 0.18, 0.0], "goal": [0.50, 0.18, 0.0], "ee": [0.10, 0.18], "desc": "open_straight"},
    {"obj": [0.20, 0.18, 0.0], "goal": [0.50, 0.40, 0.0], "ee": [0.10, 0.18], "desc": "open_diagonal"},
    {"obj": [0.10, 0.18, 0.0], "goal": [0.55, 0.18, 0.0], "ee": [0.02, 0.18], "desc": "open_long"},
    {"obj": [0.40, 0.30, 0.0], "goal": [0.10, 0.10, 0.0], "ee": [0.50, 0.30], "desc": "far_push_reverse"},
    {"obj": [0.20, 0.20, 0.0], "goal": [0.20, 0.50, 0.0], "ee": [0.10, 0.20], "desc": "open_lateral"},
    {"obj": [0.20, 0.20, 0.0], "goal": [0.45, 0.40, 0.5], "ee": [0.10, 0.20], "desc": "open_rotation"},
]

def run_episode(env, planner, cost_mode, obj_init, goal, ee_init, max_steps=MAX_STEPS):
    env.reset(object_pose=np.array(obj_init), goal_pose=np.array(goal), ee_pos=np.array(ee_init))
    goal_arr = np.array(goal)
    init_obj = env.clone_state().object_pose.copy()
    contact_flags, obj_poses = [], [init_obj.copy()]
    for _ in range(max_steps):
        def cost_fn(s):
            r = rollout_action_sequence(env, s, restore_state=True)
            return rollout_cost_with_mode(predicted_object_poses=r.predicted_object_poses,
                ee_positions=r.ee_positions, action_sequence=s, goal_pose=goal_arr,
                cost_mode=cost_mode, contact_flags=r.contact_flags, collision_flags=r.collision_flags)
        res = planner.optimize(cost_fn)
        state = env.step(res.action_sequence[0])
        obj_poses.append(state.object_pose.copy())
        contact_flags.append(float(state.last_contact))
        if np.linalg.norm(state.object_pose[:2] - goal_arr[:2]) < SUCCESS_THRESH:
            break
    obj_poses = np.array(obj_poses)
    contact_flags = np.array(contact_flags)
    init_d = float(np.linalg.norm(init_obj[:2] - goal_arr[:2]))
    final_d = float(np.linalg.norm(obj_poses[-1,:2] - goal_arr[:2]))
    best_d = float(np.min(np.linalg.norm(obj_poses[:,:2] - goal_arr[:2], axis=-1)))
    cr = float(np.mean(contact_flags>0.5)) if len(contact_flags)>0 else 0.0
    ci = np.where(contact_flags>0.5)[0]
    return {"contact_rate": round(cr,4), "first_contact_step": int(ci[0]) if len(ci)>0 else -1,
        "object_progress": round(init_d-final_d,4), "drift": round(max(0.,final_d-best_d),4),
        "final_success": bool(final_d<SUCCESS_THRESH), "init_dist": round(init_d,4),
        "final_dist": round(final_d,4), "best_dist": round(best_d,4), "n_steps": len(contact_flags)}

def main():
    run_dir = Path(__file__).resolve().parent.parent / "runs" / f"cost_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    env = ToyPushEnv()
    planner = MPPI(horizon=MPPI_CFG["horizon"], num_samples=MPPI_CFG["num_samples"],
        temperature=MPPI_CFG["temperature"], init_std=MPPI_CFG["init_std"],
        smoothing=MPPI_CFG["smoothing"], seed=42)
    modes = ["current", "staged_full"]
    total = len(SCENARIOS)*len(modes)
    trial = 0
    for s in SCENARIOS:
        for m in modes:
            trial+=1
            print(f"[{trial}/{total}] {s['desc']}/{m} ...", end=" ", flush=True)
            try:
                met = run_episode(env, planner, m, s["obj"], s["goal"], s["ee"])
                r = {"trial":trial,"scenario":s["desc"],"cost_mode":m,**met}
                all_results.append(r)
                print(f"✅ p={met['object_progress']:.3f} c={met['contact_rate']:.2f} d={met['drift']:.3f} s={met['final_success']}")
            except Exception as e:
                print(f"❌ {e}")
                import traceback; traceback.print_exc()
                all_results.append({"trial":trial,"scenario":s["desc"],"cost_mode":m,"error":str(e)})
    # Save
    json_path = run_dir / "ablation_summary.json"
    with open(json_path,"w") as f: json.dump(all_results,f,indent=2,default=str)
    valid = [r for r in all_results if "error" not in r]
    if valid:
        csv_path = run_dir / "ablation_summary.csv"
        with open(csv_path,"w") as f:
            keys = list(valid[0].keys()); f.write(",".join(keys)+"\n")
            for r in valid: f.write(",".join(str(r[k]) for k in keys)+"\n")
    # Print summary
    print("\n"+"="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    for mode in modes:
        mr = [r for r in all_results if r.get("cost_mode")==mode and "error" not in r]
        if not mr: continue
        print(f"\n  {mode}:")
        print(f"    trials: {len(mr)}")
        print(f"    avg_progress: {np.mean([r['object_progress'] for r in mr]):.4f}")
        print(f"    avg_contact:  {np.mean([r['contact_rate'] for r in mr]):.4f}")
        print(f"    avg_drift:    {np.mean([r['drift'] for r in mr]):.4f}")
        print(f"    success_rate: {np.mean([r['final_success'] for r in mr]):.2%}")
    print(f"\n{'─'*70}")
    print(f"{'scenario':<22} {'metric':<18} {'current':>9} {'staged':>9} {'delta':>9}")
    print(f"{'─'*70}")
    for s in SCENARIOS:
        cur=next((r for r in all_results if r.get("scenario")==s["desc"] and r.get("cost_mode")=="current"),None)
        stg=next((r for r in all_results if r.get("scenario")==s["desc"] and r.get("cost_mode")=="staged_full"),None)
        if not cur or not stg or "error" in cur or "error" in stg: continue
        for metric in ["object_progress","contact_rate","drift","final_success"]:
            cv,sv=cur.get(metric,0),stg.get(metric,0)
            if isinstance(cv,bool): cv,sv=int(cv),int(sv)
            d=sv-cv; m="↑" if d>0 else ("↓" if d<0 else "─")
            print(f"  {s['desc']:<20} {metric:<18} {cv:>9.4f} {sv:>9.4f} {d:>+9.4f} {m}")
    print(f"\nResults: {run_dir}")

if __name__=="__main__":
    main()
