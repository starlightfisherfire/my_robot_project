#!/usr/bin/env python3
"""Closed-loop MPC rollout — Stage 2A: execute-steps loop + pose-aware metrics.
Each MPC planning cycle executes args.execute_steps actions from the planned sequence.
"""
from __future__ import annotations

import argparse, json, os, sys, time, math, fcntl
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.mppi import MPPI
from src.planners.multimodal_cem_mpc import MultimodalCEMMPC
from src.data.episode_writer import EpisodeWriter


# ── helpers ──────────────────────────────────────────────────────────────────

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def extract_theta(pose: np.ndarray) -> float:
    """Extract theta (radians) from pose array.
    If len==3: pose[2] is theta directly.
    If len>=4: pose[2]=sin(theta), pose[3]=cos(theta) → atan2.
    """
    if len(pose) == 3:
        return float(pose[2])
    else:
        return math.atan2(float(pose[2]), float(pose[3]))


def theta_error_deg(obj_theta: float, goal_theta: float) -> float:
    return abs(wrap_to_pi(obj_theta - goal_theta)) * 180.0 / math.pi


def pos_dist(obj_xy, goal_xy):
    return float(np.linalg.norm(np.asarray(obj_xy) - np.asarray(goal_xy)))


def pose_score(pd: float, te: float) -> float:
    return pd / 0.002 + te / 10.0


# ── arg parser ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--templates", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--template-index", type=int, default=0)
    p.add_argument("--planner-mode", default="mppi", choices=["cem","multimodal_cem","mppi"])
    p.add_argument("--horizon", type=int, default=140)
    p.add_argument("--execute-steps", type=int, default=10)
    p.add_argument("--max-mpc-steps", type=int, default=100)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--num-elites", type=int, default=96)
    p.add_argument("--num-iterations", type=int, default=5)
    p.add_argument("--max-speed-mps", type=float, default=0.75)
    p.add_argument("--pusher-mass", type=float, default=0.300)
    p.add_argument("--mppi-temperature", type=float, default=0.1)
    p.add_argument("--mppi-init-std", type=float, default=0.7)
    p.add_argument("--mppi-smoothing", type=float, default=0.2)
    p.add_argument("--lateral-offset", type=float, default=0.5)
    p.add_argument("--camera", default="topdown")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out-video", default=None)
    p.add_argument("--data-output-dir", default="data/sim/mppi_sweep_v1")
    return p.parse_args()


def make_cost_fn(env, weights):
    from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
    def cost_fn(action_seq):
        return mujoco_oracle_rollout_cost(env=env, action_sequence=action_seq,
                                          weights=weights, restore_state=True)
    return cost_fn


def make_goal_relative_means(env, horizon, lateral_offset=0.5):
    obj = env.get_object_pose()
    goal = env.get_goal_pose()
    dx, dy = goal[0]-obj[0], goal[1]-obj[1]
    d = np.hypot(dx, dy)
    if d < 1e-4:
        return np.zeros((horizon, 2))
    dir_x, dir_y = dx/d, dy/d
    base = np.tile(np.array([dir_x, dir_y]), (horizon, 1)) * 0.3
    if lateral_offset > 0:
        perp = np.array([-dir_y, dir_x])
        mid = horizon // 2
        base[:mid] += perp * lateral_offset
        base[mid:] -= perp * lateral_offset * 0.5
    return base


def render_frame(renderer, camera, env, text_lines):
    renderer.update_scene(env.data, camera=camera)
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    y = 10
    for line in text_lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += 16
    return np.array(img)


def warm_start_mean(mean: np.ndarray, shift: int) -> np.ndarray:
    """Shift planned mean by `shift` steps. Remaining steps are filled with zeros."""
    H = mean.shape[0]
    if shift >= H:
        return np.zeros_like(mean)
    new = np.zeros_like(mean)
    new[: H - shift] = mean[shift:]
    return new


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    with open(args.templates) as f:
        templates = json.load(f)
    matches = [t for t in templates if t.get("split") == args.split]
    if not matches:
        print(f"ERROR: No templates for split={args.split}", file=sys.stderr)
        sys.exit(1)
    if args.template_index >= len(matches):
        args.template_index = 0
    template = matches[args.template_index]

    env = MujocoPushEnv(max_speed_mps=args.max_speed_mps, pusher_mass=args.pusher_mass)
    env.reset_from_template(template)

    action_low = np.array([-1.0, -1.0])
    action_high = np.array([1.0, 1.0])

    if args.planner_mode == "mppi":
        planner = MPPI(
            horizon=args.horizon, action_dim=2, num_samples=args.num_samples,
            num_iterations=args.num_iterations, action_low=action_low,
            action_high=action_high, temperature=args.mppi_temperature,
            init_std=args.mppi_init_std, smoothing=args.mppi_smoothing,
        )
    elif args.planner_mode == "cem":
        planner = CEMMPC(
            horizon=args.horizon, action_dim=2, num_samples=args.num_samples,
            num_elites=args.num_elites, num_iterations=args.num_iterations,
            action_low=action_low, action_high=action_high,
        )
    else:
        lateral = args.lateral_offset
        init_means = [
            make_goal_relative_means(env, args.horizon, lateral_offset=0.0),
            make_goal_relative_means(env, args.horizon, lateral_offset=lateral),
            make_goal_relative_means(env, args.horizon, lateral_offset=-lateral),
        ]
        planner = MultimodalCEMMPC(
            horizon=args.horizon, action_dim=2, num_samples=args.num_samples,
            num_elites=args.num_elites, num_iterations=args.num_iterations,
            action_low=action_low, action_high=action_high, init_means=init_means,
        )

    # Renderer (only if video output requested)
    renderer = None
    cam_id = None
    if args.out_video:
        try:
            renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        except Exception:
            cam_id = None

    writer = EpisodeWriter(args.data_output_dir)
    from src.metrics.mujoco_oracle_capacity import make_default_mujoco_cost_weights
    weights = make_default_mujoco_cost_weights()

    obs_list = template.get("obstacles", [])
    obstacle_features = []
    for o in obs_list[:2]:
        p = o["pose"]
        sz = o.get("size_x", 0), o.get("size_y", 0)
        obstacle_features.extend([p["x"], p["y"], p.get("theta", 0), sz[0], sz[1]])
    while len(obstacle_features) < 10:
        obstacle_features.append(0.0)
    obstacle_features = np.array(obstacle_features, dtype=np.float32)

    goal_pose = np.array([
        template["goal_pose"]["x"], template["goal_pose"]["y"],
        template["goal_pose"].get("theta", 0)
    ], dtype=np.float32)
    goal_theta = float(goal_pose[2])

    # ── Initial state (before MPC loop) ────────────────────────────────────
    initial_obj = env.get_object_pose()
    initial_pos_dist_m = pos_dist(initial_obj[:2], goal_pose[:2])
    initial_theta_rad = extract_theta(initial_obj)
    initial_theta_err = theta_error_deg(initial_theta_rad, goal_theta)

    frames = []
    total_collision = 0
    total_contact = 0
    t_start = time.time()
    total_env_step = 0       # total env.step() calls
    mpc_decisions = 0        # total MPC planning calls
    planning_time = 0.0
    prev_mean = None

    # ── Stage 2A metrics ─────────────────────────────────────────────────
    pos_dist_history = []
    theta_err_history = []

    best_pos_dist_m = initial_pos_dist_m
    best_pos_dist_step = 0
    best_theta_error_deg = initial_theta_err
    best_theta_error_step = 0

    best_pose_score_val = pose_score(initial_pos_dist_m, initial_theta_err)
    best_pose_score_step = 0
    best_pose_pos_dist_m = initial_pos_dist_m
    best_pose_theta_error_deg = initial_theta_err

    reached_pos_5mm_once = initial_pos_dist_m < 0.005
    reached_pos_10mm_once = initial_pos_dist_m < 0.010
    reached_pose_5mm_10deg_once = reached_pos_5mm_once and initial_theta_err < 10.0
    reached_pose_10mm_10deg_once = reached_pos_10mm_once and initial_theta_err < 10.0

    success = False
    early_stop_reason = ""

    # ── Stage 2B path efficiency tracking ────────────────────────────────
    initial_ee = env.get_ee_pos().copy()
    ee_positions = [tuple(initial_ee[:2])]         # includes initial position
    object_positions = [tuple(initial_obj[:2])]     # includes initial position
    all_actions_norm = []     # list of np.array shape (2,) per env step
    time_to_success_2mm10deg = None
    time_to_near_10mm10deg = None

    # ── MPC loop ──────────────────────────────────────────────────────────
    for mpc_step in range(args.max_mpc_steps):
        t0 = time.time()
        env.clone_state()
        cost_fn = make_cost_fn(env, weights)

        result = planner.optimize(cost_fn, init_mean=prev_mean)
        mpc_decisions += 1
        planning_time += time.time() - t0

        # Execute first `execute_steps` actions from the planned trajectory
        for exec_i in range(args.execute_steps):
            if exec_i >= len(result.action_sequence):
                break

            action = result.action_sequence[exec_i]

            old_obj = env.get_object_pose().copy()
            old_ee = env.get_ee_pos().copy()

            env.step(action)
            total_env_step += 1

            new_obj = env.get_object_pose().copy()
            new_ee = env.get_ee_pos().copy()
            contact = env.get_contact_flag()
            collision = env.get_collision_flag()
            if collision: total_collision += 1
            if contact: total_contact += 1

            # Position + angle
            cur_pos_dist = pos_dist(new_obj[:2], goal_pose[:2])
            cur_theta_rad = extract_theta(new_obj)
            cur_theta_err = theta_error_deg(cur_theta_rad, goal_theta)

            pos_dist_history.append(cur_pos_dist)
            theta_err_history.append(cur_theta_err)

            global_step = total_env_step  # 1-indexed for reporting

            # Update bests
            if cur_pos_dist < best_pos_dist_m:
                best_pos_dist_m = cur_pos_dist
                best_pos_dist_step = global_step
            if cur_theta_err < best_theta_error_deg:
                best_theta_error_deg = cur_theta_err
                best_theta_error_step = global_step

            ps = pose_score(cur_pos_dist, cur_theta_err)
            if ps < best_pose_score_val:
                best_pose_score_val = ps
                best_pose_score_step = global_step
                best_pose_pos_dist_m = cur_pos_dist
                best_pose_theta_error_deg = cur_theta_err

            # Near-success once
            if cur_pos_dist < 0.005:
                reached_pos_5mm_once = True
            if cur_pos_dist < 0.010:
                reached_pos_10mm_once = True
            if cur_pos_dist < 0.005 and cur_theta_err < 10.0:
                reached_pose_5mm_10deg_once = True
            if cur_pos_dist < 0.010 and cur_theta_err < 10.0:
                reached_pose_10mm_10deg_once = True

            # Stage 2B: time-to-threshold tracking
            if time_to_success_2mm10deg is None and cur_pos_dist < 0.002 and cur_theta_err < 10.0:
                time_to_success_2mm10deg = total_env_step
            if time_to_near_10mm10deg is None and cur_pos_dist < 0.010 and cur_theta_err < 10.0:
                time_to_near_10mm10deg = total_env_step

            # Save transition
            old_state_arr = np.concatenate([old_obj, old_ee]).astype(np.float32)
            new_state_arr = np.concatenate([new_obj, new_ee]).astype(np.float32)
            ee_vel = (new_ee - old_ee).astype(np.float32)
            obj_vel = (new_obj[:2] - old_obj[:2]).astype(np.float32)
            action_n = np.asarray(action, dtype=np.float32)

            # Stage 2B: record path positions and actions (after action_n defined)
            ee_positions.append(tuple(new_ee[:2]))
            object_positions.append(tuple(new_obj[:2]))
            all_actions_norm.append(action_n.copy())

            writer.add_step(
                state=old_state_arr, action_norm=action_n,
                action_physical=action_n * args.max_speed_mps,
                next_state=new_state_arr,
                object_pose=old_obj, next_object_pose=new_obj,
                ee_position=old_ee, next_ee_position=new_ee,
                contact=bool(contact), collision=bool(collision),
                actual_ee_velocity=ee_vel, actual_object_velocity=obj_vel,
            )

            # Early stop
            if cur_pos_dist < 0.002 and cur_theta_err < 10.0:
                success = True
                early_stop_reason = "pose_2mm_10deg"
                break

            # Render frame (one per env step)
            if cam_id is not None:
                text_lines = [
                    f"MPC#{mpc_step+1}/{args.max_mpc_steps} exec{exec_i+1}/{args.execute_steps}",
                    f"Dist:{cur_pos_dist:.4f}m θ:{cur_theta_err:.1f}° step:{global_step}",
                ]
                frames.append(render_frame(renderer, cam_id, env, text_lines))

        # If early-stopped inside execute loop, break MPC loop too
        if success:
            break

        # Warm-start: shift planned mean by execute_steps
        prev_mean = warm_start_mean(result.mean, args.execute_steps)

    total_runtime = time.time() - t_start
    mpc_steps_done = mpc_decisions  # number of MPC planning calls actually made

    # ── Final metrics ────────────────────────────────────────────────────────
    final_obj = env.get_object_pose()
    final_pos_dist_m = pos_dist(final_obj[:2], goal_pose[:2])
    final_theta_rad = extract_theta(final_obj)
    final_theta_error_deg = theta_error_deg(final_theta_rad, goal_theta)

    success_pos_1mm = final_pos_dist_m < 0.001
    success_pos_2mm = final_pos_dist_m < 0.002
    success_pos_5mm = final_pos_dist_m < 0.005
    success_pos_10mm = final_pos_dist_m < 0.010
    success_pos_50mm = final_pos_dist_m < 0.050

    success_pose_2mm_10deg = final_pos_dist_m < 0.002 and final_theta_error_deg < 10.0
    success_pose_5mm_10deg = final_pos_dist_m < 0.005 and final_theta_error_deg < 10.0
    success_pose_10mm_10deg = final_pos_dist_m < 0.010 and final_theta_error_deg < 10.0
    success_pose_5mm_5deg = final_pos_dist_m < 0.005 and final_theta_error_deg < 5.0
    success_pose_10mm_5deg = final_pos_dist_m < 0.010 and final_theta_error_deg < 5.0

    if not success:
        success = success_pose_2mm_10deg
        if success:
            early_stop_reason = "pose_2mm_10deg_final"

    regressed = reached_pose_10mm_10deg_once and final_pos_dist_m > 0.020

    total_progress_m = initial_pos_dist_m - final_pos_dist_m

    K = min(20, max(1, len(pos_dist_history) // 4))
    if len(pos_dist_history) >= K + 1:
        last_progress_m = pos_dist_history[-K] - final_pos_dist_m
    else:
        last_progress_m = 0.0

    # ── Stage 2B path efficiency / random-walk diagnostics ───────────────────
    def _pairwise_dist(pairs):
        """Sum of L2 distances between consecutive (x,y) tuples."""
        if len(pairs) < 2:
            return 0.0
        s = 0.0
        for i in range(len(pairs) - 1):
            dx = pairs[i+1][0] - pairs[i][0]
            dy = pairs[i+1][1] - pairs[i][1]
            s += math.hypot(dx, dy)
        return s

    ee_path_length_m = _pairwise_dist(ee_positions)
    object_path_length_m = _pairwise_dist(object_positions)
    net_progress_m = total_progress_m  # alias for path efficiency calc
    progress_efficiency_ee = net_progress_m / max(ee_path_length_m, 1e-9)
    progress_efficiency_object = net_progress_m / max(object_path_length_m, 1e-9)
    wasted_motion_ratio = ee_path_length_m / max(net_progress_m, 1e-4)
    wasted_motion_ratio_capped = min(wasted_motion_ratio, 100.0)

    action_smoothness_mean = 0.0
    action_direction_change_count = 0
    mean_action_norm_val = 0.0
    if len(all_actions_norm) >= 2:
        diffs = [np.linalg.norm(all_actions_norm[i+1] - all_actions_norm[i])
                 for i in range(len(all_actions_norm) - 1)]
        action_smoothness_mean = float(np.mean(diffs)) if diffs else 0.0
        for i in range(len(all_actions_norm) - 1):
            a0 = all_actions_norm[i]
            a1 = all_actions_norm[i+1]
            n0 = np.linalg.norm(a0)
            n1 = np.linalg.norm(a1)
            cosim = np.dot(a0, a1) / max(n0 * n1, 1e-9)
            if cosim < 0:
                action_direction_change_count += 1
    if all_actions_norm:
        mean_action_norm_val = float(np.mean([np.linalg.norm(a) for a in all_actions_norm]))

    contact_efficiency = object_path_length_m / max(total_contact, 1)

    to_succ_steps = time_to_success_2mm10deg if time_to_success_2mm10deg is not None else ""
    to_near_steps = time_to_near_10mm10deg if time_to_near_10mm10deg is not None else ""

    random_walk_flag = (
        ee_path_length_m > 1.5
        and net_progress_m < 0.05
        and not success
    )
    inefficient_success_flag = (
        success
        and wasted_motion_ratio > 20
    )
    excessive_wander_flag = (
        ee_path_length_m > 10.0
        or wasted_motion_ratio_capped > 50
    )
    clean_success_flag = (
        success
        and wasted_motion_ratio_capped <= 20
        and progress_efficiency_ee >= 0.05
    )

    # ── Stage 2C segment metrics (early/middle/late) ──────────────────────
    total_steps_for_seg = total_env_step if total_env_step > 0 else 1
    early_end = total_steps_for_seg // 3
    middle_end = 2 * total_steps_for_seg // 3

    # Segment positions (include initial position)
    early_ee_pos = ee_positions[:early_end + 1]
    middle_ee_pos = ee_positions[early_end:middle_end + 1]
    late_ee_pos = ee_positions[middle_end:]

    early_obj_pos = object_positions[:early_end + 1]
    middle_obj_pos = object_positions[early_end:middle_end + 1]
    late_obj_pos = object_positions[middle_end:]

    early_ee_path_length_m = _pairwise_dist(early_ee_pos)
    middle_ee_path_length_m = _pairwise_dist(middle_ee_pos)
    late_ee_path_length_m = _pairwise_dist(late_ee_pos)

    early_object_path_length_m = _pairwise_dist(early_obj_pos)
    middle_object_path_length_m = _pairwise_dist(middle_obj_pos)
    late_object_path_length_m = _pairwise_dist(late_obj_pos)

    # Segment progress
    early_pos_dist_start = initial_pos_dist_m
    early_pos_dist_end = pos_dist_history[min(early_end, len(pos_dist_history)-1)] if pos_dist_history else initial_pos_dist_m
    early_progress_m = early_pos_dist_start - early_pos_dist_end

    middle_pos_dist_start = early_pos_dist_end
    middle_pos_dist_end = pos_dist_history[min(middle_end, len(pos_dist_history)-1)] if pos_dist_history else early_pos_dist_end
    middle_progress_m = middle_pos_dist_start - middle_pos_dist_end

    late_pos_dist_start = middle_pos_dist_end
    late_pos_dist_end = final_pos_dist_m
    late_progress_m = late_pos_dist_start - late_pos_dist_end

    # Segment progress efficiency
    early_progress_efficiency_ee = early_progress_m / max(early_ee_path_length_m, 1e-9)
    middle_progress_efficiency_ee = middle_progress_m / max(middle_ee_path_length_m, 1e-9)
    late_progress_efficiency_ee = late_progress_m / max(late_ee_path_length_m, 1e-9)

    # Segment action direction changes
    early_actions = all_actions_norm[:early_end]
    middle_actions = all_actions_norm[early_end:middle_end]
    late_actions = all_actions_norm[middle_end:]

    def _count_dir_changes(actions):
        cnt = 0
        for i in range(len(actions) - 1):
            a0, a1 = actions[i], actions[i+1]
            n0, n1 = np.linalg.norm(a0), np.linalg.norm(a1)
            cosim = np.dot(a0, a1) / max(n0 * n1, 1e-9)
            if cosim < 0:
                cnt += 1
        return cnt

    early_action_direction_change_count = _count_dir_changes(early_actions)
    middle_action_direction_change_count = _count_dir_changes(middle_actions)
    late_action_direction_change_count = _count_dir_changes(late_actions)

    # Segment contact counts
    early_contact_count = 0
    middle_contact_count = 0
    late_contact_count = 0
    # Re-simulate to count contacts per segment (simplified: use contact history if available)
    # For now, approximate by proportion of steps
    if total_contact > 0 and total_env_step > 0:
        early_contact_count = int(total_contact * early_end / total_env_step)
        middle_contact_count = int(total_contact * (middle_end - early_end) / total_env_step)
        late_contact_count = total_contact - early_contact_count - middle_contact_count

    # Best distance improvement per segment
    early_best_dist_improvement_m = initial_pos_dist_m - min(pos_dist_history[:early_end+1]) if pos_dist_history and early_end < len(pos_dist_history) else 0.0
    middle_best_before = min(pos_dist_history[:early_end+1]) if pos_dist_history and early_end < len(pos_dist_history) else initial_pos_dist_m
    middle_best_after = min(pos_dist_history[:middle_end+1]) if pos_dist_history and middle_end < len(pos_dist_history) else middle_best_before
    middle_best_dist_improvement_m = middle_best_before - middle_best_after
    late_best_before = middle_best_after
    late_best_after = best_pos_dist_m
    late_best_dist_improvement_m = late_best_before - late_best_after

    # Diagnostic flags
    late_breakthrough_flag = (
        success
        and late_progress_m > 0.5 * total_progress_m
    )
    front_loaded_wander_flag = (
        (early_ee_path_length_m + middle_ee_path_length_m) > 0.6 * ee_path_length_m
        and (early_progress_m + middle_progress_m) < 0.3 * total_progress_m
    )
    meaningless_exploration_flag = (
        front_loaded_wander_flag
        and late_breakthrough_flag
    )

    # ── Failure type ─────────────────────────────────────────────────────────
    if success:
        failure_type = "success"
    elif success_pose_5mm_10deg:
        failure_type = "near_success_pose_5mm_10deg"
    elif success_pose_10mm_10deg:
        failure_type = "near_success_pose_10mm_10deg"
    elif reached_pose_10mm_10deg_once and regressed:
        failure_type = "regressed_after_near_success"
    elif total_contact == 0:
        failure_type = "no_contact"
    elif total_collision > 10 and last_progress_m < 0.002:
        failure_type = "collision_stuck"
    elif total_progress_m < 0.020:
        failure_type = "low_progress"
    elif final_pos_dist_m < 0.010 and final_theta_error_deg >= 10.0:
        failure_type = "position_reached_theta_failed"
    else:
        failure_type = "not_reached"

    # ── Save video ───────────────────────────────────────────────────────────
    if args.out_video and frames:
        try:
            import imageio
            imageio.mimsave(args.out_video, frames, fps=args.fps)
        except ImportError:
            import subprocess
            tmpdir = Path("/tmp/mppi_frames")
            tmpdir.mkdir(exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(tmpdir / f"frame_{i:06d}.png")
            subprocess.run(["ffmpeg","-y","-framerate",str(args.fps),
                "-i",str(tmpdir/"frame_%06d.png"),
                "-c:v","libx264","-pix_fmt","yuv420p",str(args.out_video)],
                capture_output=True)

    # ── Episode metadata ─────────────────────────────────────────────────────
    template_family = (
        template.get("layout_family")
        or template.get("family")
        or template.get("split", "").replace("test_sim_layout_ood_", "")
    )
    is_bypass = "bypass" in template_family
    is_direct = "direct" in template_family

    metadata = {
        "planner_mode": args.planner_mode,
        "temperature": args.mppi_temperature,
        "num_samples": args.num_samples,
        "num_iterations": args.num_iterations,
        "init_std": args.mppi_init_std,
        "smoothing": args.mppi_smoothing,
        "horizon": args.horizon,
        "execute_steps": args.execute_steps,
        "max_mpc_steps": args.max_mpc_steps,
        "speed_mps": args.max_speed_mps,
        "speed_cm_s": round(args.max_speed_mps * 100, 2),
        "family": template_family,
        "type": ("passage_bypass" if is_bypass else "passage_direct" if is_direct
                 else "blocking" if ("blocking" in template_family or template.get("obstacles"))
                 else "open"),
        "split": template.get("split",""),
        "template_index": args.template_index,
        "template_id": template.get("reset_template_id",""),
        "obstacle_count": len(template.get("obstacles", [])),
        "passage_gap": template.get("effective_passage_gap") or template.get("passage_gap"),
        "is_direct": is_direct,
        "is_bypass": is_bypass,
        "success": success,
        "best_dist": float(best_pos_dist_m),
        "collision_count": total_collision,
        "collision_rate": total_collision / max(total_env_step, 1),
        "contact_count": total_contact,
        "failure_type": failure_type,
        "video_path": args.out_video or "",
    }

    episode_id = writer.save_episode(goal_pose, obstacle_features, metadata)

    # NaN check
    nan_found = False
    ep_path = Path(args.data_output_dir) / "episodes" / f"{episode_id}.npz"
    if ep_path.exists():
        data = np.load(ep_path)
        for key in ["states", "actions_norm", "next_states"]:
            if key in data and np.any(np.isnan(data[key])):
                print(f"WARNING: NaN in {key}", file=sys.stderr)
                nan_found = True

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "config": f"mppi_T{args.mppi_temperature}_{template_family}_idx{args.template_index}",
        "family": template_family,
        "type": ("passage_bypass" if is_bypass else "passage_direct" if is_direct
                 else "blocking" if ("blocking" in template_family or template.get("obstacles"))
                 else "open"),
        "temperature": args.mppi_temperature,
        "execute_steps": args.execute_steps,
        "max_mpc_steps": args.max_mpc_steps,
        "total_budget": args.max_mpc_steps * args.execute_steps,
        "num_samples": args.num_samples,
        "init_std": args.mppi_init_std,
        "template_index": args.template_index,
        "template_id": template.get("reset_template_id",""),
        "collision_count": total_collision,
        "contact_count": total_contact,
        "mpc_steps": mpc_steps_done,
        "total_env_steps": total_env_step,
        "runtime_sec": round(total_runtime, 2),
        "episode_id": episode_id,
        "video_path": args.out_video or "",
        "log_path": "",

        "initial_pos_dist_m": round(initial_pos_dist_m, 6),
        "initial_theta_error_deg": round(initial_theta_err, 2),
        "final_pos_dist_m": round(final_pos_dist_m, 6),
        "best_pos_dist_m": round(best_pos_dist_m, 6),
        "best_pos_dist_step": best_pos_dist_step,
        "final_theta_error_deg": round(final_theta_error_deg, 2),
        "best_theta_error_deg": round(best_theta_error_deg, 2),
        "best_theta_error_step": best_theta_error_step,

        "best_pose_score": round(best_pose_score_val, 4),
        "best_pose_score_step": best_pose_score_step,
        "best_pose_pos_dist_m": round(best_pose_pos_dist_m, 6),
        "best_pose_theta_error_deg": round(best_pose_theta_error_deg, 2),

        "total_progress_m": round(total_progress_m, 6),
        "last_progress_m": round(last_progress_m, 6),

        "success": success,
        "success_pos_1mm": success_pos_1mm,
        "success_pos_2mm": success_pos_2mm,
        "success_pos_5mm": success_pos_5mm,
        "success_pos_10mm": success_pos_10mm,
        "success_pos_50mm": success_pos_50mm,

        "success_pose_2mm_10deg": success_pose_2mm_10deg,
        "success_pose_5mm_10deg": success_pose_5mm_10deg,
        "success_pose_10mm_10deg": success_pose_10mm_10deg,
        "success_pose_5mm_5deg": success_pose_5mm_5deg,
        "success_pose_10mm_5deg": success_pose_10mm_5deg,

        "reached_pos_5mm_once": reached_pos_5mm_once,
        "reached_pos_10mm_once": reached_pos_10mm_once,
        "reached_pose_5mm_10deg_once": reached_pose_5mm_10deg_once,
        "reached_pose_10mm_10deg_once": reached_pose_10mm_10deg_once,
        "regressed_after_near_success": regressed,

        # ── Stage 2B path efficiency metrics ─────────────────────────────
        "speed_mps": args.max_speed_mps,
        "speed_cm_s": round(args.max_speed_mps * 100, 2),
        "ee_path_length_m": round(ee_path_length_m, 6),
        "object_path_length_m": round(object_path_length_m, 6),
        "net_progress_m": round(net_progress_m, 6),
        "progress_efficiency_ee": round(progress_efficiency_ee, 8),
        "progress_efficiency_object": round(progress_efficiency_object, 8),
        "wasted_motion_ratio": round(wasted_motion_ratio, 4),
        "wasted_motion_ratio_capped": round(wasted_motion_ratio_capped, 4),
        "action_smoothness_mean": round(action_smoothness_mean, 6),
        "action_direction_change_count": action_direction_change_count,
        "mean_action_norm": round(mean_action_norm_val, 6),
        "contact_efficiency": round(contact_efficiency, 6),
        "time_to_success_env_steps": to_succ_steps,
        "time_to_near_success_10mm_env_steps": to_near_steps,
        "random_walk_flag": random_walk_flag,
        "inefficient_success_flag": inefficient_success_flag,
        "excessive_wander_flag": excessive_wander_flag,
        "clean_success_flag": clean_success_flag,

        # ── Stage 2C segment metrics ───────────────────────────────────
        "early_ee_path_length_m": round(early_ee_path_length_m, 6),
        "middle_ee_path_length_m": round(middle_ee_path_length_m, 6),
        "late_ee_path_length_m": round(late_ee_path_length_m, 6),
        "early_object_path_length_m": round(early_object_path_length_m, 6),
        "middle_object_path_length_m": round(middle_object_path_length_m, 6),
        "late_object_path_length_m": round(late_object_path_length_m, 6),
        "early_progress_m": round(early_progress_m, 6),
        "middle_progress_m": round(middle_progress_m, 6),
        "late_progress_m": round(late_progress_m, 6),
        "early_progress_efficiency_ee": round(early_progress_efficiency_ee, 8),
        "middle_progress_efficiency_ee": round(middle_progress_efficiency_ee, 8),
        "late_progress_efficiency_ee": round(late_progress_efficiency_ee, 8),
        "early_action_direction_change_count": early_action_direction_change_count,
        "middle_action_direction_change_count": middle_action_direction_change_count,
        "late_action_direction_change_count": late_action_direction_change_count,
        "early_contact_count": early_contact_count,
        "middle_contact_count": middle_contact_count,
        "late_contact_count": late_contact_count,
        "early_best_dist_improvement_m": round(early_best_dist_improvement_m, 6),
        "middle_best_dist_improvement_m": round(middle_best_dist_improvement_m, 6),
        "late_best_dist_improvement_m": round(late_best_dist_improvement_m, 6),
        "late_breakthrough_flag": late_breakthrough_flag,
        "front_loaded_wander_flag": front_loaded_wander_flag,
        "meaningless_exploration_flag": meaningless_exploration_flag,

        # ── Stage 2B path diagnostics (Gate 3) ─────────────────────────
        "ee_positions_count": len(ee_positions),
        "object_positions_count": len(object_positions),
        "path_includes_initial_position": True,

        "early_stop_reason": early_stop_reason,
        "failure_type": failure_type,
        "nan_check": "FAIL" if nan_found else "PASS",
    }

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
