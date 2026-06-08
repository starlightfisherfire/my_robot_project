#!/usr/bin/env python3
"""
Minimal Oracle staged-cost sanity sweep.

Runs CEM + MPPI × current + staged × open/blocking/passage templates.
Saves episode data, replay, metadata, render_later commands.
"""
from __future__ import annotations
import os, sys, json, time, random, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# ── Config ──
CEM_CFG = {
    "horizon": 80, "num_samples": 256, "num_elites": 32,
    "num_iterations": 3, "execute_steps": 10, "max_mpc_steps": 30,
    "max_speed_mps": 0.5, "init_std": 0.5, "smoothing": 0.2,
}
MPPI_CFG = {
    "horizon": 80, "num_samples": 512, "temperature": 0.1,
    "init_std": 0.5, "smoothing": 0.2, "execute_steps": 10,
    "max_mpc_steps": 30, "max_speed_mps": 0.5,
}
FAMILIES = ["open", "blocking", "passage"]
TRIALS_PER = 2  # 2 per family
COST_MODES = ["current", "staged_contact_obstacle_goal"]
WORKERS = 4

OUT_DIR = Path(os.environ.get("SWEEP_OUT_DIR", "runs/oracle_staged_sweep_sanity"))


def run_one_trial(args):
    """Run one planner × cost_mode × template trial."""
    planner_name, cost_mode, family, seed, trial_idx, out_dir = args
    out_dir = Path(out_dir)

    try:
        from src.envs.mujoco_push_env import MujocoPushEnv
        from src.planners.cem_mpc import CEMMPC
        from src.planners.mppi import MPPI
        from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
        from src.planners.cost_functions import CostWeights, make_staged_cost_weights
        from src.data.template_generator import generate_template, is_template_valid

        rng = random.Random(seed)

        # Generate valid template
        for _ in range(50):
            template = generate_template(family, rng)
            if is_template_valid(template):
                break
        else:
            return {"status": "failed", "reason": "no_valid_template"}

        # Setup env
        cfg = CEM_CFG if planner_name == "CEM" else MPPI_CFG
        env = MujocoPushEnv(max_speed_mps=cfg["max_speed_mps"])
        env.reset_from_template(template)
        goal = env.get_goal_pose()

        # Obstacle info for cost
        obs_list = template.get("obstacles", [])
        obs_pos = np.array([[o["pose"]["x"], o["pose"]["y"]] for o in obs_list]) if obs_list else None
        obs_rad = np.array([max(o["size_x"], o["size_y"]) / 2.0 for o in obs_list]) if obs_list else None

        # Cost function
        if cost_mode == "staged_contact_obstacle_goal":
            weights = make_staged_cost_weights()
        else:
            weights = CostWeights()

        def cost_fn(action_seq):
            return mujoco_oracle_rollout_cost(
                env=env, action_sequence=action_seq,
                weights=weights, restore_state=True,
                obstacle_positions=obs_pos, obstacle_radii=obs_rad,
                cost_mode=cost_mode,
            )

        # Planner
        if planner_name == "CEM":
            planner = CEMMPC(
                horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                num_elites=cfg["num_elites"], num_iterations=cfg["num_iterations"],
                init_std=cfg["init_std"], smoothing=cfg["smoothing"],
            )
        else:
            planner = MPPI(
                horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                temperature=cfg["temperature"], init_std=cfg["init_std"],
                smoothing=cfg["smoothing"],
            )

        # Run MPC loop
        prev_mean = None
        all_object_poses = [env.get_object_pose().copy()]
        all_ee_pos = [env.get_ee_pos().copy()]
        all_contact = [env.get_contact_flag()]
        all_collision = [env.get_collision_flag()]
        all_actions = []
        all_costs = []
        all_qpos = [env.data.qpos.copy()]
        all_qvel = [env.data.qvel.copy()]
        all_ctrl = [env.data.ctrl.copy()]

        best_dist = float("inf")
        best_dist_ever = float("inf")
        first_contact_step = -1
        mpc_steps = 0
        t0 = time.time()

        for mpc_i in range(cfg["max_mpc_steps"]):
            mpc_steps += 1
            result = planner.optimize(cost_fn, init_mean=prev_mean)
            all_costs.append(result.best_cost)

            for exec_i in range(cfg["execute_steps"]):
                if exec_i >= len(result.action_sequence):
                    break
                action = result.action_sequence[exec_i]
                env.step(action)

                obj_pose = env.get_object_pose().copy()
                ee = env.get_ee_pos().copy()
                contact = env.get_contact_flag()
                collision = env.get_collision_flag()

                all_object_poses.append(obj_pose)
                all_ee_pos.append(ee)
                all_contact.append(contact)
                all_collision.append(collision)
                all_actions.append(action.copy())
                all_qpos.append(env.data.qpos.copy())
                all_qvel.append(env.data.qvel.copy())
                all_ctrl.append(env.data.ctrl.copy())

                dist = float(np.linalg.norm(obj_pose[:2] - goal[:2]))
                if dist < best_dist_ever:
                    best_dist_ever = dist
                if contact > 0.5 and first_contact_step < 0:
                    first_contact_step = len(all_object_poses) - 1

            # Early stop
            cur_obj = env.get_object_pose()
            cur_dist = float(np.linalg.norm(cur_obj[:2] - goal[:2]))
            cur_theta_err = abs(float(np.arctan2(np.sin(cur_obj[2] - goal[2]), np.cos(cur_obj[2] - goal[2]))))
            if cur_dist < 0.005 and cur_theta_err < 0.15:
                break

            # Warm start
            prev_mean = np.zeros_like(result.mean)
            shift = cfg["execute_steps"]
            if shift < len(result.mean):
                prev_mean[:-shift] = result.mean[shift:]

        runtime = time.time() - t0

        # Compute metrics
        object_poses = np.array(all_object_poses)
        ee_positions = np.array(all_ee_pos)
        contact_flags = np.array(all_contact)
        collision_flags = np.array(all_collision)
        actions_arr = np.array(all_actions)

        final_pose = object_poses[-1]
        final_dist = float(np.linalg.norm(final_pose[:2] - goal[:2]))
        final_theta_err = abs(float(np.arctan2(np.sin(final_pose[2] - goal[2]), np.cos(final_pose[2] - goal[2]))))
        success = final_dist < 0.01 and final_theta_err < 0.2
        contact_rate = float(np.mean(contact_flags > 0.5))
        collision_rate = float(np.mean(collision_flags > 0.5))
        drift_after_best = final_dist - best_dist_ever

        # Save episode
        episode_id = f"{planner_name}_{cost_mode}_{family}_{seed}"
        ep_dir = out_dir / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        # episode.npz
        np.savez_compressed(
            ep_dir / "episode.npz",
            object_poses=object_poses,
            ee_positions=ee_positions,
            goal_pose=goal,
            obstacle_positions=obs_pos if obs_pos is not None else np.array([]),
            obstacle_radii=obs_rad if obs_rad is not None else np.array([]),
            actions_physical=actions_arr * cfg["max_speed_mps"],
            actions_env=actions_arr,
            actions_planner=actions_arr,
            contact_flags=contact_flags,
            collision_flags=collision_flags,
        )

        # replay.npz
        np.savez_compressed(
            ep_dir / "replay.npz",
            qpos=np.array(all_qpos),
            qvel=np.array(all_qvel),
            ctrl=np.array(all_ctrl),
            initial_qpos=all_qpos[0],
            initial_qvel=all_qvel[0],
        )

        # planner_trace.npz
        np.savez_compressed(
            ep_dir / "planner_trace.npz",
            selected_costs=np.array(all_costs),
            executed_actions=actions_arr,
        )

        # metadata.json
        meta = {
            "episode_id": episode_id,
            "schema_version": "oracle_multimodal_v1_minimal",
            "planner": planner_name,
            "cost_mode": cost_mode,
            "planner_config_name": f"{planner_name}_SANITY",
            "template_id": f"{family}_{seed}",
            "family": family,
            "success": success,
            "final_dist": final_dist,
            "best_dist": best_dist_ever,
            "drift_after_best": drift_after_best,
            "first_contact_step": first_contact_step,
            "contact_rate": contact_rate,
            "collision_rate": collision_rate,
            "runtime_sec": runtime,
            "mpc_steps": mpc_steps,
            "total_steps": len(all_object_poses) - 1,
            "video_status": "deferred",
            "replay_path": str(ep_dir / "replay.npz"),
            "episode_path": str(ep_dir / "episode.npz"),
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # render_later_command.txt
        render_cmd = (
            f"MUJOCO_GL=egl PYTHONPATH=. /home/brucewu/miniconda3/envs/lerobot/bin/python "
            f"scripts/render_episode_from_replay.py --episode_dir {ep_dir} --height 224 --width 224"
        )
        with open(ep_dir / "render_later_command.txt", "w") as f:
            f.write(render_cmd + "\n")

        return {
            "status": "ok", "episode_id": episode_id,
            "planner": planner_name, "cost_mode": cost_mode, "family": family,
            "success": success, "final_dist": final_dist, "best_dist": best_dist_ever,
            "drift_after_best": drift_after_best,
            "first_contact_step": first_contact_step,
            "contact_rate": contact_rate, "collision_rate": collision_rate,
            "runtime_sec": runtime, "episode_path": str(ep_dir),
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "reason": str(e), "planner": planner_name,
                "cost_mode": cost_mode, "family": family}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"runs/oracle_staged_sweep_sanity_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Generate tasks
    tasks = []
    seed_base = 20260601
    for planner in ["CEM", "MPPI"]:
        for cost_mode in COST_MODES:
            for family in FAMILIES:
                for trial in range(TRIALS_PER):
                    seed = seed_base + hash(f"{planner}_{cost_mode}_{family}_{trial}") % 100000
                    tasks.append((planner, cost_mode, family, seed, trial, str(out_dir)))

    print(f"Total trials: {len(tasks)}")
    print(f"Workers: {WORKERS}")
    print()

    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(run_one_trial, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            results.append(r)
            status = r["status"]
            eid = r.get("episode_id", "?")
            print(f"  [{i+1}/{len(tasks)}] {status} {eid}")

    elapsed = time.time() - t0

    # Save summary
    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] != "ok"]

    # summary.csv
    import csv
    csv_path = out_dir / "summary.csv"
    if ok_results:
        keys = ["episode_id", "planner", "cost_mode", "family", "success",
                "final_dist", "best_dist", "drift_after_best",
                "first_contact_step", "contact_rate", "collision_rate", "runtime_sec"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in ok_results:
                w.writerow({k: r.get(k, "") for k in keys})

    # summary.json
    summary = {
        "total_trials": len(tasks),
        "ok": len(ok_results),
        "failed": len(failed_results),
        "elapsed_sec": elapsed,
        "results": ok_results,
        "failures": failed_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY ({len(ok_results)}/{len(tasks)} OK, {elapsed:.0f}s)")
    print(f"{'='*80}")
    if ok_results:
        print(f"{'Planner':<6} {'CostMode':<35} {'Family':<10} {'Trials':<7} {'Success':<8} {'MeanFD':<8} {'MeanCR':<8}")
        from collections import defaultdict
        groups = defaultdict(list)
        for r in ok_results:
            groups[(r["planner"], r["cost_mode"], r["family"])].append(r)
        for (pl, cm, fam), rs in sorted(groups.items()):
            n = len(rs)
            succ = sum(1 for r in rs if r["success"])
            mfd = np.mean([r["final_dist"] for r in rs])
            mcr = np.mean([r["contact_rate"] for r in rs])
            print(f"{pl:<6} {cm:<35} {fam:<10} {n:<7} {succ:<8} {mfd:<8.4f} {mcr:<8.3f}")

    if failed_results:
        print(f"\nFailed trials:")
        for r in failed_results:
            print(f"  {r.get('planner')}/{r.get('cost_mode')}/{r.get('family')}: {r.get('reason')}")

    # Generate render_later_commands.sh
    render_sh = out_dir / "render_later_commands.sh"
    with open(render_sh, "w") as f:
        f.write("#!/bin/bash\n# Render all episodes from replay\n\n")
        for r in ok_results:
            ep_dir = r.get("episode_path", "")
            if ep_dir:
                f.write(
                    f"MUJOCO_GL=egl PYTHONPATH=. /home/brucewu/miniconda3/envs/lerobot/bin/python "
                    f"scripts/render_episode_from_replay.py --episode_dir {ep_dir} --height 224 --width 224\n"
                )
    os.chmod(render_sh, 0o755)

    print(f"\nSaved: {out_dir}")
    print(f"  summary.csv: {csv_path}")
    print(f"  summary.json: {out_dir / 'summary.json'}")
    print(f"  render_later_commands.sh: {render_sh}")

    return 0 if not failed_results else 1


if __name__ == "__main__":
    sys.exit(main())
