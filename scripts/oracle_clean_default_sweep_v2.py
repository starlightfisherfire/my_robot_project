#!/usr/bin/env python3
"""Clean default cost sweep - one trial per invocation."""
import sys, os, json, time, random, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))
import numpy as np

CEM_CONFIGS = {
    "CEM_s01_H100": {"horizon":100,"num_samples":512,"num_elites":64,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "CEM_s01_H140": {"horizon":140,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "CEM_s03_H100": {"horizon":100,"num_samples":512,"num_elites":64,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "CEM_s05_H100": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "CEM_s05_H140_exec20": {"horizon":140,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.5},
    "CEM_s075_exec20": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75},
}
MPPI_CONFIGS = {
    "MPPI_s01_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "MPPI_s03_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "MPPI_s05_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "MPPI_s075_H100": {"horizon":100,"num_samples":2048,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.75},
    "MPPI_s03_H140": {"horizon":140,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "MPPI_s05_H140": {"horizon":140,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "MPPI_s075_exec20": {"horizon":100,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75},
}


def run_one_trial(args):
    planner_name, config_name, cost_mode, family, seed, out_dir = args
    out_dir = Path(out_dir)
    trial_id = f"{planner_name}_{config_name}_{cost_mode}_{family}_{seed}"
    try:
        from src.envs.mujoco_push_env import MujocoPushEnv
        from src.planners.cem_mpc import CEMMPC
        from src.planners.mppi import MPPI
        from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
        from src.planners.cost_functions import CostWeights, make_staged_cost_weights, make_clean_default_cost_weights, staged_contact_obstacle_goal_cost
        from src.data.template_generator import generate_template, is_template_valid

        all_cfgs = {**CEM_CONFIGS, **MPPI_CONFIGS}
        cfg = all_cfgs[config_name]

        rng = random.Random(seed)
        for _ in range(50):
            template = generate_template(family, rng)
            if is_template_valid(template):
                break
        else:
            return {"status":"failed","reason":"no_valid_template","trial_id":trial_id}

        env = MujocoPushEnv(max_speed_mps=cfg["max_speed_mps"])
        env.reset_from_template(template)
        goal = env.get_goal_pose()

        obs_list = template.get("obstacles", [])
        obs_pos = np.array([[o["pose"]["x"], o["pose"]["y"]] for o in obs_list]) if obs_list else None
        obs_rad = np.array([max(o["size_x"], o["size_y"])/2.0 for o in obs_list]) if obs_list else None

        if cost_mode == "staged_contact_obstacle_goal":
            weights = make_staged_cost_weights()
        elif cost_mode == "clean_default":
            weights = make_clean_default_cost_weights()
        else:
            weights = CostWeights()

        def cost_fn(action_seq):
            return mujoco_oracle_rollout_cost(
                env=env, action_sequence=action_seq, weights=weights,
                restore_state=True, obstacle_positions=obs_pos, obstacle_radii=obs_rad,
                cost_mode=cost_mode,
            )

        if planner_name == "CEM":
            planner = CEMMPC(
                horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                num_elites=cfg["num_elites"], num_iterations=cfg["num_iterations"],
            )
        else:
            planner = MPPI(
                horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                temperature=cfg["temperature"], init_std=cfg["init_std"],
                smoothing=cfg["smoothing"],
            )

        execute_steps = cfg["execute_steps"]
        total_steps = cfg["total_env_steps"]
        max_mpc_replans = total_steps // execute_steps

        qpos_trace = [env.data.qpos.copy()]
        qvel_trace = [env.data.qvel.copy()]
        ctrl_trace = []
        obj_arr = [env.get_object_pose().copy()]
        ee_arr = [env.get_ee_pos().copy()]
        act_arr_list = []
        cf_arr_list = []
        colf_arr_list = []
        costs_list = []
        planner_runtimes = []
        best_dist_trace = []
        zero_action_costs = []
        cost_breakdowns = []
        prev_mean = None

        best_dist_ever = float(np.linalg.norm(obj_arr[0][:2] - goal[:2]))
        first_contact_step = -1
        post_contact_loss_count = 0
        post_contact_total = 0
        contact_happened = False

        for replan_i in range(max_mpc_replans):
            rt0 = time.time()
            result = planner.optimize(cost_fn, init_mean=prev_mean)
            planner_runtimes.append(time.time() - rt0)
            costs_list.append(result.best_cost)

            if hasattr(result, 'best_action_sequence') and result.best_action_sequence is not None:
                act_seq = np.array(result.best_action_sequence[:execute_steps])
                if hasattr(result, 'elite_mean') and result.elite_mean is not None:
                    prev_mean = np.roll(result.elite_mean, -execute_steps, axis=0)
            else:
                act_seq = np.zeros((execute_steps, 2))

            if len(act_seq) < execute_steps:
                pad = np.zeros((execute_steps - len(act_seq), 2))
                act_seq = np.concatenate([act_seq, pad])

            for step_i in range(execute_steps):
                action = act_seq[step_i]
                env.step(action)
                qpos_trace.append(env.data.qpos.copy())
                qvel_trace.append(env.data.qvel.copy())
                ctrl_trace.append(action)
                obj_pose = env.get_object_pose().copy()
                ee_pos = env.get_ee_pos().copy()
                obj_arr.append(obj_pose)
                ee_arr.append(ee_pos)
                act_arr_list.append(action)

                cf = env.get_contact_flag()
                colf = env.get_collision_flag()
                cf_arr_list.append(cf)
                colf_arr_list.append(colf)

                cur_dist = float(np.linalg.norm(obj_pose[:2] - goal[:2]))
                if cur_dist < best_dist_ever:
                    best_dist_ever = cur_dist

                if cf > 0.5 and not contact_happened:
                    first_contact_step = len(cf_arr_list) - 1
                    contact_happened = True

                best_dist_trace.append(best_dist_ever)

            # Early stop
            theta_err_now = abs(float(np.arctan2(np.sin(obj_arr[-1][2]-goal[2]), np.cos(obj_arr[-1][2]-goal[2]))))
            cur_dist_now = float(np.linalg.norm(obj_arr[-1][:2] - goal[:2]))
            if cur_dist_now < 0.005 and theta_err_now < 0.15:
                break

        obj_arr = np.array(obj_arr)
        ee_arr = np.array(ee_arr)
        act_arr = np.array(act_arr_list)
        cf_arr = np.array(cf_arr_list)
        colf_arr = np.array(colf_arr_list)
        qpos_trace = np.array(qpos_trace)
        qvel_trace = np.array(qvel_trace)
        ctrl_trace = np.array(ctrl_trace)

        final_pose = obj_arr[-1]
        final_dist = float(np.linalg.norm(final_pose[:2] - goal[:2]))
        final_theta_err = abs(float(np.arctan2(np.sin(final_pose[2]-goal[2]), np.cos(final_pose[2]-goal[2]))))
        success = final_dist < 0.01 and final_theta_err < 0.2

        contact_rate = float(np.mean(cf_arr > 0.5))
        collision_rate = float(np.mean(colf_arr > 0.5))

        drift_after_best = final_dist - best_dist_ever
        total_steps_done = len(act_arr)
        runtime = sum(planner_runtimes)

        post_contact_loss_rate = 0.0
        if first_contact_step >= 0:
            post_contact_flags = cf_arr[first_contact_step:]
            if len(post_contact_flags) > 0:
                post_contact_loss_rate = float(np.mean(post_contact_flags < 0.5))

        # Save
        ep_dir = out_dir / "episodes" / trial_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        distance_to_goal_trace = np.linalg.norm(obj_arr[:,:2] - goal[:2], axis=1)
        theta_error_trace = np.array([
            abs(float(np.arctan2(np.sin(obj_arr[i,2]-goal[2]), np.cos(obj_arr[i,2]-goal[2]))))
            for i in range(len(obj_arr))
        ])

        np.savez_compressed(ep_dir / "episode.npz",
            object_poses=obj_arr, ee_positions=ee_arr, goal_pose=goal,
            obstacle_positions=obs_pos if obs_pos is not None else np.array([]),
            obstacle_radii=obs_rad if obs_rad is not None else np.array([]),
            actions_physical=act_arr * cfg["max_speed_mps"],
            actions_env=act_arr, actions_planner=act_arr,
            contact_flags=cf_arr, collision_flags=colf_arr,
            best_dist_trace=np.array(best_dist_trace),
            distance_to_goal_trace=distance_to_goal_trace,
            theta_error_trace=theta_error_trace,
        )

        np.savez_compressed(ep_dir / "replay.npz",
            qpos=qpos_trace, qvel=qvel_trace, ctrl=ctrl_trace,
            initial_qpos=qpos_trace[0], initial_qvel=qvel_trace[0], reset_seed=seed,
        )

        git_commit = ""
        try: git_commit = os.popen("git rev-parse --short HEAD").read().strip()
        except: pass

        meta = {
            "episode_id": trial_id, "schema_version": "oracle_multimodal_v1_minimal",
            "planner": planner_name, "cost_mode": cost_mode,
            "planner_config_name": config_name,
            "template_id": f"{family}_{seed}", "family": family,
            "success": success, "final_dist": final_dist, "best_dist": best_dist_ever,
            "drift_after_best": drift_after_best,
            "first_contact_step": first_contact_step, "contact_rate": contact_rate,
            "post_contact_loss_rate": post_contact_loss_rate,
            "collision_rate": collision_rate, "object_collision_rate": collision_rate,
            "ee_collision_rate": 0.0, "runtime_sec": runtime,
            "total_steps": total_steps_done, "mpc_replans": len(costs_list),
            "video_status": "deferred", "git_commit": git_commit,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"status":"ok", "trial_id":trial_id, "planner":planner_name,
                "config":config_name, "cost_mode":cost_mode, "family":family,
                "success":success, "final_dist":final_dist, "best_dist":best_dist_ever,
                "contact_rate":contact_rate, "collision_rate":collision_rate,
                "runtime_sec":runtime, "total_steps":total_steps_done}

    except Exception as e:
        return {"status":"failed","reason":traceback.format_exc()[-200:],"trial_id":trial_id}


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/oracle_clean_default_sweep")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers

    tasks = []
    seed_base = 20260601

    for cfg_name in CEM_CONFIGS:
        for family in ["open", "blocking", "passage"]:
            for trial in range(3):
                seed = seed_base + hash(f"CEM_{cfg_name}_clean_default_{family}_{trial}") % 100000
                tasks.append(("CEM", cfg_name, "clean_default", family, seed, str(out_dir)))

    for cfg_name in MPPI_CONFIGS:
        for family in ["open", "blocking", "passage"]:
            for trial in range(3):
                seed = seed_base + hash(f"MPPI_{cfg_name}_clean_default_{family}_{trial}") % 100000
                tasks.append(("MPPI", cfg_name, "clean_default", family, seed, str(out_dir)))

    for family in ["open", "blocking", "passage"]:
        for trial in range(3):
            seed = seed_base + hash(f"MPPI_baseline_clean_default_{family}_{trial}") % 100000
            tasks.append(("MPPI", "MPPI_s03_H100", "clean_default", family, seed, str(out_dir)))

    print(f"=== Clean Default Cost Sweep ===", flush=True)
    print(f"  Total trials: {len(tasks)}, Workers: {workers}", flush=True)
    print(flush=True)

    t0 = time.time()
    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one_trial, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            elapsed = time.time() - t0
            if r.get("status") == "ok":
                results.append(r)
                s = "✓" if r["success"] else "✗"
                print(f"  [{i+1}/{len(tasks)}] {s} {r['trial_id']} dist={r['final_dist']:.4f} ({elapsed:.0f}s)", flush=True)
            else:
                failed.append(r)
                print(f"  [{i+1}/{len(tasks)}] FAILED {r.get('trial_id','?')} {r.get('reason','?')[:60]}", flush=True)

    summary = {
        "total_trials": len(tasks), "ok": len(results), "failed": len(failed),
        "success_count": sum(1 for r in results if r.get("success")),
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: {len(results)}/{len(tasks)} OK, {summary['success_count']} success, {summary['elapsed_sec']:.0f}s", flush=True)

    if results:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            groups[(r["planner"], r["config"], r["family"])].append(r)

        print(f"\n{'Planner':<5} {'Config':<25} {'Family':<10} {'N':>3} {'Succ':>5} {'MFD':>8} {'MCR':>6} {'MColR':>6} {'MRt':>6}", flush=True)
        for (pl, cfg, fam), rs in sorted(groups.items()):
            n = len(rs)
            succ = sum(1 for r in rs if r["success"])
            mfd = np.mean([r["final_dist"] for r in rs])
            mcr = np.mean([r["contact_rate"] for r in rs])
            mcolr = np.mean([r["collision_rate"] for r in rs])
            mrt = np.mean([r["runtime_sec"] for r in rs])
            print(f"{pl:<5} {cfg:<25} {fam:<10} {n:>3} {succ:>5} {mfd:>8.4f} {mcr:>6.3f} {mcolr:>6.3f} {mrt:>6.1f}", flush=True)

    print(f"\nSaved: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
