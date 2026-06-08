#!/usr/bin/env python3
"""
Full Oracle CEM/MPPI staged-cost parameter sweep.

1000 env steps per trial, speed ablation, comprehensive data saving.
"""
from __future__ import annotations
import os, sys, json, time, random, traceback, uuid, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import fcntl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

# ── Planner configs ──
CEM_CONFIGS = {
    "CEM_s01_H100": {"horizon":100,"num_samples":512,"num_elites":64,"num_iterations":5,
                     "execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "CEM_s01_H140": {"horizon":140,"num_samples":1024,"num_elites":128,"num_iterations":5,
                     "execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "CEM_s03_H100": {"horizon":100,"num_samples":512,"num_elites":64,"num_iterations":5,
                     "execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "CEM_s05_H100": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,
                     "execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "CEM_s05_H140_exec20": {"horizon":140,"num_samples":1024,"num_elites":128,"num_iterations":5,
                            "execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.5},
    "CEM_s075_exec20": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,
                        "execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75},
}

MPPI_CONFIGS = {
    "MPPI_s01_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,
                      "smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "MPPI_s03_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,
                      "smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "MPPI_s05_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,
                      "smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "MPPI_s075_H100": {"horizon":100,"num_samples":2048,"temperature":0.1,"init_std":0.5,
                       "smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.75},
    "MPPI_s03_H140": {"horizon":140,"num_samples":2048,"temperature":0.2,"init_std":0.5,
                      "smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "MPPI_s05_H140": {"horizon":140,"num_samples":2048,"temperature":0.2,"init_std":0.5,
                      "smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "MPPI_s075_exec20": {"horizon":100,"num_samples":2048,"temperature":0.2,"init_std":0.5,
                         "smoothing":0.2,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75},
}

FAMILIES = ["open", "blocking", "passage"]
TRIALS_PER = 3
COST_MODES = ["current", "staged_contact_obstacle_goal"]
WORKERS = int(os.environ.get("SWEEP_WORKERS", "13"))
OUT_DIR = Path(os.environ.get("SWEEP_OUT_DIR", "runs/oracle_staged_full_sweep"))


def run_one_trial(args):
    """Run one planner × cost_mode × template trial with 1000 env steps."""
    planner_name, config_name, cost_mode, family, seed, out_dir = args
    out_dir = Path(out_dir)
    trial_id = f"{planner_name}_{config_name}_{cost_mode}_{family}_{seed}"

    try:
        from src.envs.mujoco_push_env import MujocoPushEnv
        from src.planners.cem_mpc import CEMMPC
        from src.planners.mppi import MPPI
        from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
        from src.planners.cost_functions import CostWeights, make_staged_cost_weights, staged_contact_obstacle_goal_cost
        from src.data.template_generator import generate_template, is_template_valid

        # Get config
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
                init_std=0.5, smoothing=0.2,
            )
        else:
            planner = MPPI(
                horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                temperature=cfg["temperature"], init_std=cfg["init_std"],
                smoothing=cfg["smoothing"],
            )

        # Data accumulators
        object_poses = [env.get_object_pose().copy()]
        ee_positions = [env.get_ee_pos().copy()]
        contact_flags = [env.get_contact_flag()]
        collision_flags = [env.get_collision_flag()]
        actions_list = []
        qpos_trace = [env.data.qpos.copy()]
        qvel_trace = [env.data.qvel.copy()]
        ctrl_trace = [env.data.ctrl.copy()]
        costs_list = []
        best_dist_trace = []
        planner_runtimes = []
        zero_action_costs = []
        cost_breakdowns = []

        best_dist_ever = float("inf")
        first_contact_step = -1
        total_steps = 0
        prev_mean = None
        t0 = time.time()

        execute_steps = cfg["execute_steps"]
        total_env_steps = cfg["total_env_steps"]
        max_replans = total_env_steps // execute_steps

        for replan_i in range(max_replans):
            rt0 = time.time()
            result = planner.optimize(cost_fn, init_mean=prev_mean)
            planner_runtimes.append(time.time() - rt0)

            costs_list.append(result.best_cost)

            # Zero-action cost
            if replan_i == 0:
                try:
                    zc = mujoco_oracle_rollout_cost(
                        env=env, action_sequence=np.zeros((cfg["horizon"],2)),
                        weights=weights, restore_state=True,
                        obstacle_positions=obs_pos, obstacle_radii=obs_rad,
                        cost_mode=cost_mode,
                    )
                    zero_action_costs.append(zc)
                except:
                    zero_action_costs.append(float("nan"))

            # Cost breakdown for first replan
            if replan_i == 0 and cost_mode == "staged_contact_obstacle_goal":
                try:
                    from src.planners.mujoco_oracle_rollout import rollout_action_sequence_mujoco
                    r = rollout_action_sequence_mujoco(env, result.action_sequence, restore_state=True)
                    bd = staged_contact_obstacle_goal_cost(
                        r.predicted_object_poses, r.ee_positions, result.action_sequence,
                        goal, weights=weights, contact_flags=r.contact_flags,
                        collision_flags=r.collision_flags,
                        obstacle_positions=obs_pos, obstacle_radii=obs_rad,
                        return_breakdown=True,
                    )
                    cost_breakdowns.append(bd)
                except:
                    pass

            early_stop = False
            for exec_i in range(execute_steps):
                if exec_i >= len(result.action_sequence):
                    break
                action = result.action_sequence[exec_i]
                env.step(action)
                total_steps += 1

                obj = env.get_object_pose().copy()
                ee = env.get_ee_pos().copy()
                cf = env.get_contact_flag()
                colf = env.get_collision_flag()

                object_poses.append(obj)
                ee_positions.append(ee)
                contact_flags.append(cf)
                collision_flags.append(colf)
                actions_list.append(action.copy())
                qpos_trace.append(env.data.qpos.copy())
                qvel_trace.append(env.data.qvel.copy())
                ctrl_trace.append(env.data.ctrl.copy())

                dist = float(np.linalg.norm(obj[:2] - goal[:2]))
                if dist < best_dist_ever:
                    best_dist_ever = dist
                best_dist_trace.append(best_dist_ever)

                if cf > 0.5 and first_contact_step < 0:
                    first_contact_step = total_steps

                # Early stop
                theta_err = abs(float(np.arctan2(np.sin(obj[2]-goal[2]), np.cos(obj[2]-goal[2]))))
                if dist < 0.005 and theta_err < 0.15:
                    early_stop = True
                    break

            if early_stop:
                break

            # Warm start
            prev_mean = np.zeros_like(result.mean)
            shift = execute_steps
            if shift < len(result.mean):
                prev_mean[:-shift] = result.mean[shift:]

        runtime = time.time() - t0

        # Compute metrics
        obj_arr = np.array(object_poses)
        ee_arr = np.array(ee_positions)
        cf_arr = np.array(contact_flags)
        colf_arr = np.array(collision_flags)
        act_arr = np.array(actions_list)

        final_pose = obj_arr[-1]
        final_dist = float(np.linalg.norm(final_pose[:2] - goal[:2]))
        final_theta_err = abs(float(np.arctan2(np.sin(final_pose[2]-goal[2]), np.cos(final_pose[2]-goal[2]))))
        success = final_dist < 0.01 and final_theta_err < 0.2

        contact_rate = float(np.mean(cf_arr > 0.5))
        collision_rate = float(np.mean(colf_arr > 0.5))

        # Post-contact loss rate
        contact_indices = np.where(cf_arr > 0.5)[0]
        if len(contact_indices) > 0:
            first_t = int(contact_indices[0])
            post_contact = cf_arr[first_t:]
            post_contact_loss_rate = float(np.mean(post_contact < 0.5))
        else:
            post_contact_loss_rate = 1.0

        drift_after_best = final_dist - best_dist_ever

        # Theta error trace
        theta_error_trace = np.array([
            abs(float(np.arctan2(np.sin(obj_arr[i,2]-goal[2]), np.cos(obj_arr[i,2]-goal[2]))))
            for i in range(len(obj_arr))
        ])

        distance_to_goal_trace = np.linalg.norm(obj_arr[:,:2] - goal[:2], axis=1)

        # ── Save episode ──
        ep_dir = out_dir / "episodes" / trial_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        # episode.npz
        np.savez_compressed(
            ep_dir / "episode.npz",
            object_poses=obj_arr, ee_positions=ee_arr, goal_pose=goal,
            obstacle_positions=obs_pos if obs_pos is not None else np.array([]),
            obstacle_radii=obs_rad if obs_rad is not None else np.array([]),
            actions_physical=act_arr * cfg["max_speed_mps"],
            actions_env=act_arr, actions_planner=act_arr,
            contact_flags=cf_arr, collision_flags=colf_arr,
            best_dist_trace=np.array(best_dist_trace),
            distance_to_goal_trace=distance_to_goal_trace,
            theta_error_trace=theta_error_trace,
            cost_breakdown_trace=json.dumps(cost_breakdowns) if cost_breakdowns else "",
        )

        # replay.npz
        np.savez_compressed(
            ep_dir / "replay.npz",
            qpos=np.array(qpos_trace), qvel=np.array(qvel_trace),
            ctrl=np.array(ctrl_trace),
            initial_qpos=qpos_trace[0], initial_qvel=qvel_trace[0],
            reset_seed=seed,
        )

        # planner_trace.npz
        np.savez_compressed(
            ep_dir / "planner_trace.npz",
            selected_costs=np.array(costs_list),
            executed_actions=act_arr,
            planner_runtime_per_replan=np.array(planner_runtimes),
            zero_action_costs=np.array(zero_action_costs),
            selected_action_sequences=act_arr[:len(costs_list)*execute_steps].reshape(-1, execute_steps, 2) if len(act_arr) >= len(costs_list)*execute_steps else np.array([]),
        )

        # metadata.json
        git_commit = ""
        try:
            git_commit = os.popen("git rev-parse --short HEAD").read().strip()
        except:
            pass

        meta = {
            "episode_id": trial_id, "schema_version": "oracle_multimodal_v1_minimal",
            "planner": planner_name, "cost_mode": cost_mode,
            "planner_config_name": config_name,
            "template_id": f"{family}_{seed}", "family": family,
            "success": success,
            "final_dist": final_dist, "best_dist": best_dist_ever,
            "drift_after_best": drift_after_best,
            "first_contact_step": first_contact_step,
            "contact_rate": contact_rate,
            "post_contact_loss_rate": post_contact_loss_rate,
            "collision_rate": collision_rate,
            "object_collision_rate": collision_rate,  # same as collision for now
            "ee_collision_rate": 0.0,  # not separated yet
            "runtime_sec": runtime, "total_steps": total_steps,
            "mpc_replans": len(costs_list),
            "video_status": "deferred",
            "replay_path": str(ep_dir / "replay.npz"),
            "episode_path": str(ep_dir / "episode.npz"),
            "git_commit": git_commit,
            "worker_id": mp.current_process().name,
            "pid": os.getpid(),
            "planner_topk_status": "NOT_IMPLEMENTED",
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # render_later_command.txt
        cmd = (f"MUJOCO_GL=egl PYTHONPATH=. "
               f"/home/brucewu/miniconda3/envs/lerobot/bin/python "
               f"scripts/render_episode_from_replay.py --episode_dir {ep_dir} --height 224 --width 224")
        with open(ep_dir / "render_later_command.txt", "w") as f:
            f.write(cmd + "\n")

        return {
            "status":"ok", "trial_id":trial_id,
            "planner":planner_name, "config":config_name, "cost_mode":cost_mode,
            "family":family, "success":success,
            "final_dist":final_dist, "best_dist":best_dist_ever,
            "drift_after_best":drift_after_best,
            "first_contact_step":first_contact_step,
            "contact_rate":contact_rate,
            "post_contact_loss_rate":post_contact_loss_rate,
            "collision_rate":collision_rate,
            "runtime_sec":runtime, "total_steps":total_steps,
            "mpc_replans":len(costs_list),
        }

    except Exception as e:
        traceback.print_exc()
        return {"status":"failed","reason":str(e),"trial_id":trial_id}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or WORKERS

    # Generate tasks
    tasks = []
    seed_base = 20260601

    # CEM staged
    for cfg_name in CEM_CONFIGS:
        for family in FAMILIES:
            for trial in range(TRIALS_PER):
                seed = seed_base + hash(f"CEM_{cfg_name}_staged_{family}_{trial}") % 100000
                tasks.append(("CEM", cfg_name, "staged_contact_obstacle_goal", family, seed, str(out_dir)))

    # MPPI staged
    for cfg_name in MPPI_CONFIGS:
        for family in FAMILIES:
            for trial in range(TRIALS_PER):
                seed = seed_base + hash(f"MPPI_{cfg_name}_staged_{family}_{trial}") % 100000
                tasks.append(("MPPI", cfg_name, "staged_contact_obstacle_goal", family, seed, str(out_dir)))

    # Baseline: MPPI_s03_H100 + current
    for family in FAMILIES:
        for trial in range(TRIALS_PER):
            seed = seed_base + hash(f"MPPI_baseline_current_{family}_{trial}") % 100000
            tasks.append(("MPPI", "MPPI_s03_H100", "current", family, seed, str(out_dir)))

    print(f"=== Oracle Staged Full Sweep ===")
    print(f"  CEM configs: {len(CEM_CONFIGS)}")
    print(f"  MPPI configs: {len(MPPI_CONFIGS)}")
    print(f"  Families: {FAMILIES}")
    print(f"  Trials per: {TRIALS_PER}")
    print(f"  Total trials: {len(tasks)}")
    print(f"  Workers: {workers}")
    print(f"  Output: {out_dir}")
    print()

    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(run_one_trial, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            results.append(r)
            status = r["status"]
            tid = r.get("trial_id", "?")[:50]
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(tasks) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(tasks)}] {status} {tid} ({elapsed:.0f}s, ETA {eta:.0f}s)")

            # Progress update every 10 trials
            if (i+1) % 10 == 0:
                ok = sum(1 for r in results if r["status"] == "ok")
                fail = sum(1 for r in results if r["status"] != "ok")
                succ = sum(1 for r in results if r.get("success"))
                print(f"    Progress: {ok} ok, {fail} failed, {succ} success")

    elapsed = time.time() - t0

    # ── Save results ──
    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] != "ok"]

    # summary.csv
    csv_path = out_dir / "summary.csv"
    if ok_results:
        keys = ["trial_id","planner","config","cost_mode","family","success",
                "final_dist","best_dist","drift_after_best","first_contact_step",
                "contact_rate","post_contact_loss_rate","collision_rate",
                "runtime_sec","total_steps","mpc_replans"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in ok_results:
                w.writerow({k: r.get(k,"") for k in keys})

    # summary.json
    summary = {
        "total_trials": len(tasks), "ok": len(ok_results), "failed": len(failed_results),
        "success_count": sum(1 for r in ok_results if r.get("success")),
        "elapsed_sec": elapsed, "workers": workers,
        "results": ok_results, "failures": failed_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # episodes.jsonl
    with open(out_dir / "episodes.jsonl", "w") as f:
        for r in ok_results:
            ep_dir = out_dir / "episodes" / r["trial_id"]
            r_copy = dict(r)
            r_copy["episode_path"] = str(ep_dir / "episode.npz")
            r_copy["replay_path"] = str(ep_dir / "replay.npz")
            f.write(json.dumps(r_copy) + "\n")

    # render_later_commands.sh
    render_sh = out_dir / "render_later_commands.sh"
    with open(render_sh, "w") as f:
        f.write("#!/bin/bash\n# Render all episodes\n\n")
        for r in ok_results:
            ep_dir = out_dir / "episodes" / r["trial_id"]
            f.write(f"MUJOCO_GL=egl PYTHONPATH=. /home/brucewu/miniconda3/envs/lerobot/bin/python "
                    f"scripts/render_episode_from_replay.py --episode_dir {ep_dir} --height 224 --width 224\n")
    os.chmod(render_sh, 0o755)

    # Print summary
    print(f"\n{'='*100}")
    print(f"SUMMARY: {len(ok_results)}/{len(tasks)} OK, {sum(1 for r in ok_results if r.get('success'))} success, {elapsed:.0f}s")
    print(f"{'='*100}")

    if ok_results:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in ok_results:
            groups[(r["planner"], r["config"], r["cost_mode"], r["family"])].append(r)

        print(f"\n{'Planner':<5} {'Config':<25} {'CostMode':<35} {'Family':<10} {'N':>3} {'Succ':>5} {'MFD':>8} {'MCR':>6} {'MColR':>6} {'MRt':>6}")
        for (pl, cfg, cm, fam), rs in sorted(groups.items()):
            n = len(rs)
            succ = sum(1 for r in rs if r["success"])
            mfd = np.mean([r["final_dist"] for r in rs])
            mcr = np.mean([r["contact_rate"] for r in rs])
            mcolr = np.mean([r["collision_rate"] for r in rs])
            mrt = np.mean([r["runtime_sec"] for r in rs])
            cm_short = cm.replace("staged_contact_obstacle_goal", "staged")
            print(f"{pl:<5} {cfg:<25} {cm_short:<35} {fam:<10} {n:>3} {succ:>5} {mfd:>8.4f} {mcr:>6.3f} {mcolr:>6.3f} {mrt:>6.1f}")

    if failed_results:
        print(f"\nFailed: {len(failed_results)}")
        for r in failed_results[:5]:
            print(f"  {r.get('trial_id','?')}: {r.get('reason','?')[:80]}")

    print(f"\nSaved: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
