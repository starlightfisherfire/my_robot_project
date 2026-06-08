#!/usr/bin/env python3
"""Run a single clean_default trial. Usage: python run_single_clean_trial.py <planner> <config> <family> <seed> <out_dir>"""
import sys, os, json, time, random, traceback
from pathlib import Path
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

CEM_CONFIGS = {
    "CEM_s01_H100": {"horizon":100,"num_samples":512,"num_elites":64,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "CEM_s01_H140": {"horizon":140,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "CEM_s03_H100": {"horizon":100,"num_samples":512,"num_elites":64,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "CEM_s05_H100": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "CEM_s05_H140_exec20": {"horizon":140,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.5},
    "CEM_s075_exec20": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75},
    "CEM_s03_exec20": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.3},
}
MPPI_CONFIGS = {
    "MPPI_s01_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.1},
    "MPPI_s03_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "MPPI_s05_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "MPPI_s075_H100": {"horizon":100,"num_samples":2048,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.75},
    "MPPI_s03_H140": {"horizon":140,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.3},
    "MPPI_s05_H140": {"horizon":140,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5},
    "MPPI_s075_exec20": {"horizon":100,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75},
    "MPPI_s03_exec20": {"horizon":100,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.3},
}

# ─── Warm-Start configs ───
# These use the WarmStartCEM / WarmStartMPPI planners.
# warm_std_factor: 0.3 = trust previous plan a lot (30% of base std)
# reoptimize_std_factor: 1.0 = normal exploration at horizon frontier
WARM_START_CEM_CONFIGS = {
    "WS_CEM_s05_H100": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
    "WS_CEM_s075_exec20": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
    "WS_CEM_s05_H100_tight": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5,"warm_std_factor":0.15,"reoptimize_std_factor":0.8},
    "WS_CEM_s03_exec20": {"horizon":100,"num_samples":1024,"num_elites":128,"num_iterations":5,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.3,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
}
WARM_START_MPPI_CONFIGS = {
    "WS_MPPI_s05_H100": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
    "WS_MPPI_s075_exec20": {"horizon":100,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.75,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
    "WS_MPPI_s075_H100": {"horizon":100,"num_samples":2048,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.75,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
    "WS_MPPI_s05_H100_tight": {"horizon":100,"num_samples":1024,"temperature":0.1,"init_std":0.5,"smoothing":0.2,"execute_steps":10,"total_env_steps":1000,"max_speed_mps":0.5,"warm_std_factor":0.15,"reoptimize_std_factor":0.8},
    "WS_MPPI_s03_exec20": {"horizon":100,"num_samples":2048,"temperature":0.2,"init_std":0.5,"smoothing":0.2,"execute_steps":20,"total_env_steps":1000,"max_speed_mps":0.3,"warm_std_factor":0.3,"reoptimize_std_factor":1.0},
}

def main():
    planner_name = sys.argv[1]
    config_name = sys.argv[2]
    family = sys.argv[3]
    seed = int(sys.argv[4])
    out_dir = Path(sys.argv[5])
    cost_mode = "clean_default"

    from src.envs.mujoco_push_env import MujocoPushEnv
    from src.planners.cem_mpc import CEMMPC
    from src.planners.mppi import MPPI
    from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
    from src.planners.cost_functions import make_clean_default_cost_weights
    from src.data.template_generator import generate_template, is_template_valid

    trial_id = f"{planner_name}_{config_name}_{cost_mode}_{family}_{seed}"
    all_cfgs = {**CEM_CONFIGS, **MPPI_CONFIGS, **WARM_START_CEM_CONFIGS, **WARM_START_MPPI_CONFIGS}
    cfg = all_cfgs[config_name]

    rng = random.Random(seed)
    for _ in range(50):
        template = generate_template(family, rng)
        if is_template_valid(template):
            break
    else:
        print(json.dumps({"status":"failed","reason":"no_valid_template","trial_id":trial_id}))
        return

    env = MujocoPushEnv(max_speed_mps=cfg["max_speed_mps"])
    env.reset_from_template(template)
    goal = env.get_goal_pose()

    obs_list = template.get("obstacles", [])
    obs_pos = np.array([[o["pose"]["x"], o["pose"]["y"]] for o in obs_list]) if obs_list else None
    obs_rad = np.array([max(o["size_x"], o["size_y"])/2.0 for o in obs_list]) if obs_list else None

    weights = make_clean_default_cost_weights()

    def cost_fn(action_seq):
        return mujoco_oracle_rollout_cost(
            env=env, action_sequence=action_seq, weights=weights,
            restore_state=True, obstacle_positions=obs_pos, obstacle_radii=obs_rad,
            cost_mode=cost_mode,
        )

    from src.planners.warm_start_planner import WarmStartCEM, WarmStartMPPI

    if planner_name == "CEM":
        planner = CEMMPC(horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                         num_elites=cfg["num_elites"], num_iterations=cfg["num_iterations"])
    elif planner_name == "MPPI":
        planner = MPPI(horizon=cfg["horizon"], num_samples=cfg["num_samples"],
                       temperature=cfg["temperature"], init_std=cfg["init_std"], smoothing=cfg["smoothing"])
    elif planner_name == "WS_CEM":
        planner = WarmStartCEM(
            execute_steps=cfg["execute_steps"],
            warm_std_factor=cfg.get("warm_std_factor", 0.3),
            reoptimize_std_factor=cfg.get("reoptimize_std_factor", 1.0),
            horizon=cfg["horizon"], num_samples=cfg["num_samples"],
            num_elites=cfg["num_elites"], num_iterations=cfg["num_iterations"],
        )
    elif planner_name == "WS_MPPI":
        planner = WarmStartMPPI(
            execute_steps=cfg["execute_steps"],
            warm_std_factor=cfg.get("warm_std_factor", 0.3),
            reoptimize_std_factor=cfg.get("reoptimize_std_factor", 1.0),
            horizon=cfg["horizon"], num_samples=cfg["num_samples"],
            temperature=cfg["temperature"], init_std=cfg["init_std"], smoothing=cfg["smoothing"],
        )
    else:
        raise ValueError(f"Unknown planner: {planner_name}")

    execute_steps = cfg["execute_steps"]
    max_mpc_replans = cfg["total_env_steps"] // execute_steps

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
    prev_mean = None

    best_dist_ever = float(np.linalg.norm(obj_arr[0][:2] - goal[:2]))
    first_contact_step = -1
    contact_happened = False
    is_warm_start = planner_name.startswith("WS_")

    for replan_i in range(max_mpc_replans):
        rt0 = time.time()
        result = planner.optimize(cost_fn, init_mean=prev_mean)
        planner_runtimes.append(time.time() - rt0)
        costs_list.append(result.best_cost)

        act_seq = np.array(result.action_sequence[:execute_steps]) if hasattr(result, 'action_sequence') and result.action_sequence is not None else np.zeros((execute_steps, 2))
        if is_warm_start:
            # Warm-start planner: update its internal state
            planner.update_after_execute(result)
            # Also keep prev_mean as fallback (not strictly needed for WS, but harmless)
            if hasattr(result, 'mean') and result.mean is not None:
                prev_mean = np.roll(result.mean, -execute_steps, axis=0)
        else:
            # Original: shift mean for next replan
            if hasattr(result, 'mean') and result.mean is not None:
                prev_mean = np.roll(result.mean, -execute_steps, axis=0)
        if len(act_seq) < execute_steps:
            act_seq = np.concatenate([act_seq, np.zeros((execute_steps - len(act_seq), 2))])

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

        theta_err_now = abs(float(np.arctan2(np.sin(obj_arr[-1][2]-goal[2]), np.cos(obj_arr[-1][2]-goal[2]))))
        cur_dist_now = float(np.linalg.norm(obj_arr[-1][:2] - goal[:2]))
        if cur_dist_now < 0.005 and theta_err_now < 0.15:
            break

    obj_arr = np.array(obj_arr)
    ee_arr = np.array(ee_arr)
    act_arr = np.array(act_arr_list)
    cf_arr = np.array(cf_arr_list)
    colf_arr = np.array(colf_arr_list)

    final_pose = obj_arr[-1]
    final_dist = float(np.linalg.norm(final_pose[:2] - goal[:2]))
    final_theta_err = abs(float(np.arctan2(np.sin(final_pose[2]-goal[2]), np.cos(final_pose[2]-goal[2]))))
    success = final_dist < 0.01 and final_theta_err < 0.2
    contact_rate = float(np.mean(cf_arr > 0.5))
    collision_rate = float(np.mean(colf_arr > 0.5))
    drift_after_best = final_dist - best_dist_ever
    runtime = sum(planner_runtimes)

    post_contact_loss_rate = 0.0
    if first_contact_step >= 0:
        post_cf = cf_arr[first_contact_step:]
        if len(post_cf) > 0:
            post_contact_loss_rate = float(np.mean(post_cf < 0.5))

    ep_dir = out_dir / "episodes" / trial_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    distance_to_goal_trace = np.linalg.norm(obj_arr[:,:2] - goal[:2], axis=1)
    theta_error_trace = np.array([abs(float(np.arctan2(np.sin(obj_arr[i,2]-goal[2]), np.cos(obj_arr[i,2]-goal[2])))) for i in range(len(obj_arr))])

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
        qpos=np.array(qpos_trace), qvel=np.array(qvel_trace), ctrl=np.array(ctrl_trace),
        initial_qpos=qpos_trace[0], initial_qvel=qvel_trace[0], reset_seed=seed,
    )

    git_commit = ""
    try: git_commit = os.popen("git rev-parse --short HEAD").read().strip()
    except: pass

    meta = {
        "episode_id": trial_id, "planner": planner_name, "cost_mode": cost_mode,
        "planner_config_name": config_name, "family": family,
        "success": success, "final_dist": final_dist, "best_dist": best_dist_ever,
        "drift_after_best": drift_after_best, "first_contact_step": first_contact_step,
        "contact_rate": contact_rate, "post_contact_loss_rate": post_contact_loss_rate,
        "collision_rate": collision_rate, "runtime_sec": runtime,
        "total_steps": len(act_arr), "mpc_replans": len(costs_list),
        "video_status": "deferred", "git_commit": git_commit,
    }
    with open(ep_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    s = "SUCCESS" if success else "FAIL"
    print(f"{s} {trial_id} dist={final_dist:.4f} contact={contact_rate:.3f} runtime={runtime:.0f}s")
    print(json.dumps(meta))

if __name__ == "__main__":
    main()
