#!/usr/bin/env python3
"""
Render MuJoCo oracle-MPC closed-loop execution to video with parallel CEM.

WARNING: This script parallelizes CEM candidate evaluation within each MPC
step using ProcessPoolExecutor.  Rendering stays in the main process only.
Worker processes do MuJoCo rollout cost evaluation, never render.

Differences from render_closed_loop_rollout.py:
  - Each CEM iteration distributes num_samples across --cem-workers subprocesses
  - Each worker process creates ONE MujocoPushEnv via initializer, reuses it
  - Main process handles CEM iteration logic, closed-loop execution, rendering

Usage (serial fallback, same as original):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --out-video runs/debug/videos/serial_baseline.mp4

Usage (12-core parallel CEM):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --parallel-cem --cem-workers 12 \
    --out-video runs/debug/videos/parallel_cem_12w.mp4

Usage (debug serial compare, quick exit):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --parallel-cem --cem-workers 12 \
    --debug-serial-compare --compare-only --compare-num-samples 64

Usage (12-core parallel CEM + custom speed):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --parallel-cem --cem-workers 12 \
    --max-speed-mps 0.075 \
    --out-video runs/debug/videos/parallel_cem_12w_fast.mp4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco is not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import imageio
except ImportError:
    print("ERROR: imageio is not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: PIL is not installed.", file=sys.stderr)
    sys.exit(1)

from src.envs.mujoco_push_env import MujocoPushEnv, MujocoPushState
from src.interventions.reset_template_loader import load_reset_templates
from src.metrics.mujoco_oracle_capacity import make_default_mujoco_cost_weights
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights, rollout_cost, wrap_angle
from src.planners.mujoco_oracle_rollout import (
    mujoco_oracle_rollout_cost,
    rollout_action_sequence_mujoco,
)
from src.planners.obstacle_utils import extract_obstacle_geometry


# ─── State serialization helpers ────────────────────────────────────────────

def state_to_payload(state: MujocoPushState) -> dict:
    """Serialize MujocoPushState to a plain dict for cross-process transfer."""
    return {
        "qpos": state.qpos.copy(),
        "qvel": state.qvel.copy(),
        "ctrl": state.ctrl.copy(),
        "goal_pose": state.goal_pose.copy(),
        "step_count": int(state.step_count),
        "last_contact": bool(state.last_contact),
        "last_collision": bool(state.last_collision),
    }


def payload_to_state(payload: dict) -> MujocoPushState:
    """Deserialize dict back to MujocoPushState."""
    return MujocoPushState(
        qpos=np.array(payload["qpos"], dtype=np.float64),
        qvel=np.array(payload["qvel"], dtype=np.float64),
        ctrl=np.array(payload["ctrl"], dtype=np.float64),
        goal_pose=np.array(payload["goal_pose"], dtype=np.float64),
        step_count=payload["step_count"],
        last_contact=payload["last_contact"],
        last_collision=payload["last_collision"],
    )


# ─── Worker initializer and top-level worker function ──────────────────────

_WORKER_ENV = None
_WORKER_WEIGHTS = None
_WORKER_TEMPLATE_ID = None
_WORKER_OBSTACLE_POSITIONS = None
_WORKER_OBSTACLE_RADII = None


def init_worker_env(template: dict, env_config: dict, cost_weights_dict: dict) -> None:
    """Initializer called once per worker process.

    Creates MujocoPushEnv and CostWeights in the worker's global state
    so that evaluate_action_chunk_worker does NOT recreate them per chunk.
    """
    global _WORKER_ENV, _WORKER_WEIGHTS, _WORKER_TEMPLATE_ID
    global _WORKER_OBSTACLE_POSITIONS, _WORKER_OBSTACLE_RADII
    _WORKER_ENV = MujocoPushEnv(
        control_dt=env_config.get("control_dt", 0.1),
        max_speed_mps=env_config.get("max_speed_mps", 0.05),
        pusher_radius=env_config.get("pusher_radius", 0.010),
        pusher_halfheight=env_config.get("pusher_halfheight", 0.014),
        pusher_z=env_config.get("pusher_z", 0.016),
        pusher_mass=env_config.get("pusher_mass", 0.05),
    )
    _WORKER_ENV.reset_from_template(template)
    _WORKER_WEIGHTS = CostWeights(**cost_weights_dict)
    _WORKER_TEMPLATE_ID = template.get("reset_template_id", "unknown")
    _WORKER_OBSTACLE_POSITIONS, _WORKER_OBSTACLE_RADII = extract_obstacle_geometry(template)


def evaluate_action_chunk_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a chunk of action sequences in a subprocess.

    Uses the global _WORKER_ENV created by init_worker_env.

    Args:
        args tuple with:
            indices: np.ndarray of int — original sample indices
            state_payload: dict from state_to_payload
            action_sequences: np.ndarray [chunk_size, horizon, action_dim]

    Returns:
        (indices, costs) where costs is np.ndarray [chunk_size]
    """
    indices, state_payload, action_sequences = args

    env = _WORKER_ENV
    weights = _WORKER_WEIGHTS
    state = payload_to_state(state_payload)
    goal_pose = env.get_goal_pose()

    costs = []
    for seq in action_sequences:
        env.restore_state(state)
        result = rollout_action_sequence_mujoco(
            env=env,
            action_sequence=seq,
            restore_state=False,
        )
        from src.planners.cost_functions import rollout_cost
        cost = rollout_cost(
            predicted_object_poses=result.predicted_object_poses,
            ee_positions=result.ee_positions,
            action_sequence=seq,
            goal_pose=goal_pose,
            weights=weights,
            contact_flags=result.contact_flags,
            collision_flags=result.collision_flags,
            obstacle_positions=_WORKER_OBSTACLE_POSITIONS,
            obstacle_radii=_WORKER_OBSTACLE_RADII,
        )
        costs.append(cost)

    return indices, np.asarray(costs, dtype=np.float64)


# ─── Parallel CEM planner ───────────────────────────────────────────────────

@dataclass
class ParallelCEMResult:
    action_sequence: np.ndarray
    best_cost: float
    mean: np.ndarray
    std: np.ndarray
    cost_history: list[float]
    plan_time_sec: float
    worker_eval_time_sec: float


def shift_warm_start(
    mean: np.ndarray, std: np.ndarray, execute_steps: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shift warm-start mean/std by execute_steps for MPC continuity.

    The first execute_steps of the previous plan have been executed,
    so we shift the remaining horizon-execute_steps steps forward
    and fill the tail with zeros/default std.
    """
    action_dim = mean.shape[1]
    shifted_mean = np.zeros((horizon, action_dim), dtype=np.float64)
    shifted_std = np.full((horizon, action_dim), 0.8, dtype=np.float64)

    if execute_steps < horizon:
        shifted_mean[: horizon - execute_steps] = mean[execute_steps:]
        shifted_std[: horizon - execute_steps] = std[execute_steps:]

    return shifted_mean, shifted_std


def parallel_cem_plan(
    env: MujocoPushEnv,
    template: dict,
    state_payload: dict,
    horizon: int,
    num_samples: int,
    num_elites: int,
    num_iterations: int,
    seed: int,
    num_workers: int,
    env_config: dict,
    cost_weights_dict: dict,
    mp_start_method: str = "spawn",
    init_mean: np.ndarray | None = None,
    init_std: np.ndarray | None = None,
) -> ParallelCEMResult:
    """Run CEM with parallel cost evaluation.

    The ProcessPoolExecutor is created ONCE and reused across all CEM
    iterations.  Worker processes initialize their MujocoPushEnv once
    via the initializer, then reuse it for every chunk.

    Returns ParallelCEMResult with best action sequence and diagnostics.
    """
    rng = np.random.default_rng(seed)
    action_dim = 2

    if init_mean is None:
        mean = np.zeros((horizon, action_dim), dtype=np.float64)
    else:
        mean = np.asarray(init_mean, dtype=np.float64).copy()

    if init_std is None:
        std = np.full((horizon, action_dim), 0.8, dtype=np.float64)
    else:
        std = np.asarray(init_std, dtype=np.float64).copy()

    best_sequence = mean.copy()
    best_cost = float("inf")
    cost_history = []

    total_worker_eval_time = 0.0

    # Pre-split indices for chunk assignment (stable across iterations)
    all_indices = np.arange(num_samples)
    index_chunks = np.array_split(all_indices, num_workers)
    # Filter empty chunks (when num_workers > num_samples)
    index_chunks = [c for c in index_chunks if len(c) > 0]
    actual_workers = len(index_chunks)

    ctx = mp.get_context(mp_start_method)

    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=ctx,
        initializer=init_worker_env,
        initargs=(template, env_config, cost_weights_dict),
    ) as executor:
        for iteration in range(num_iterations):
            # Sample candidates
            samples = rng.normal(
                loc=mean[None, :, :],
                scale=std[None, :, :],
                size=(num_samples, horizon, action_dim),
            )
            samples = np.clip(samples, -1.0, 1.0)

            # Build worker args: (indices, state_payload, action_sequences)
            worker_args = []
            for idx_chunk in index_chunks:
                worker_args.append((
                    idx_chunk,
                    state_payload,
                    samples[idx_chunk],
                ))

            # Dispatch to workers
            t_eval = time.time()
            all_costs = np.empty(num_samples, dtype=np.float64)

            for indices, chunk_costs in executor.map(
                evaluate_action_chunk_worker, worker_args
            ):
                all_costs[indices] = chunk_costs

            # Validate cost array
            assert all_costs.shape == (num_samples,), (
                f"Cost shape mismatch: {all_costs.shape} != ({num_samples},)"
            )
            assert np.isfinite(all_costs).all(), "Non-finite costs detected"
            assert num_elites <= num_samples, (
                f"num_elites={num_elites} > num_samples={num_samples}"
            )

            iter_eval_time = time.time() - t_eval
            total_worker_eval_time += iter_eval_time

            # Select elites
            elite_idx = np.argsort(all_costs)[:num_elites]
            elites = samples[elite_idx]

            elite_mean = elites.mean(axis=0)
            elite_std = elites.std(axis=0) + 1e-6

            mean = 0.2 * mean + 0.8 * elite_mean
            std = 0.2 * std + 0.8 * elite_std

            iter_best_idx = int(np.argmin(all_costs))
            iter_best_cost = float(all_costs[iter_best_idx])

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_sequence = samples[iter_best_idx].copy()

            cost_history.append(best_cost)

    plan_time = time.time() - t_start

    return ParallelCEMResult(
        action_sequence=best_sequence,
        best_cost=best_cost,
        mean=mean,
        std=std,
        cost_history=cost_history,
        plan_time_sec=plan_time,
        worker_eval_time_sec=total_worker_eval_time,
    )


# ─── Parallel MPPI planner ────────────────────────────────────────────────

@dataclass
class ParallelMPPIResult:
    action_sequence: np.ndarray
    best_cost: float
    mean: np.ndarray
    std: np.ndarray
    cost_history: list[float]
    plan_time_sec: float
    worker_eval_time_sec: float


def parallel_mppi_plan(
    env: MujocoPushEnv,
    template: dict,
    state_payload: dict,
    horizon: int,
    num_samples: int,
    num_iterations: int,
    temperature: float,
    seed: int,
    num_workers: int,
    env_config: dict,
    cost_weights_dict: dict,
    mp_start_method: str = "spawn",
    init_mean: np.ndarray | None = None,
    init_std: np.ndarray | None = None,
    smoothing: float = 0.2,
) -> ParallelMPPIResult:
    """MPPI with parallel cost evaluation via ProcessPoolExecutor."""
    rng = np.random.default_rng(seed)
    action_dim = 2

    if init_mean is None:
        mean = np.zeros((horizon, action_dim), dtype=np.float64)
    else:
        mean = np.asarray(init_mean, dtype=np.float64).copy()

    if init_std is None:
        std = np.full((horizon, action_dim), 0.8, dtype=np.float64)
    else:
        std = np.asarray(init_std, dtype=np.float64).copy()

    best_sequence = mean.copy()
    best_cost = float("inf")
    cost_history = []
    total_worker_eval_time = 0.0

    all_indices = np.arange(num_samples)
    index_chunks = [c for c in np.array_split(all_indices, num_workers) if len(c) > 0]
    actual_workers = len(index_chunks)

    ctx = mp.get_context(mp_start_method)
    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=ctx,
        initializer=init_worker_env,
        initargs=(template, env_config, cost_weights_dict),
    ) as executor:
        for iteration in range(num_iterations):
            samples = rng.normal(
                loc=mean[None, :, :],
                scale=std[None, :, :],
                size=(num_samples, horizon, action_dim),
            )
            samples = np.clip(samples, -1.0, 1.0)

            worker_args = [
                (idx_chunk, state_payload, samples[idx_chunk])
                for idx_chunk in index_chunks
            ]

            t_eval = time.time()
            all_costs = np.empty(num_samples, dtype=np.float64)
            for indices, chunk_costs in executor.map(
                evaluate_action_chunk_worker, worker_args
            ):
                all_costs[indices] = chunk_costs
            total_worker_eval_time += time.time() - t_eval

            assert np.isfinite(all_costs).all(), "Non-finite costs in MPPI"

            shifted = all_costs - np.min(all_costs)
            weights = np.exp(-shifted / temperature)
            w_sum = weights.sum()
            if w_sum < 1e-10:
                weights = np.ones_like(weights) / num_samples
            else:
                weights = weights / w_sum

            weighted_mean = np.einsum("n,nhd->hd", weights, samples)
            diff = samples - weighted_mean[None, :, :]
            weighted_std = np.sqrt(
                np.einsum("n,nhd->hd", weights, diff ** 2) + 1e-6
            )

            mean = smoothing * mean + (1.0 - smoothing) * weighted_mean
            std = smoothing * std + (1.0 - smoothing) * weighted_std

            iter_best_idx = int(np.argmin(all_costs))
            iter_best_cost = float(all_costs[iter_best_idx])
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_sequence = samples[iter_best_idx].copy()
            cost_history.append(best_cost)

    plan_time = time.time() - t_start
    return ParallelMPPIResult(
        action_sequence=best_sequence,
        best_cost=best_cost,
        mean=mean,
        std=std,
        cost_history=cost_history,
        plan_time_sec=plan_time,
        worker_eval_time_sec=total_worker_eval_time,
    )


# ─── Parallel Multimodal CEM planner ──────────────────────────────────────

@dataclass
class ParallelMultimodalCEMResult:
    action_sequence: np.ndarray
    best_cost: float
    mean: np.ndarray
    std: np.ndarray
    cost_history: list[float]
    plan_time_sec: float
    worker_eval_time_sec: float
    mode_costs: list[float]
    best_mode: int


def _make_goal_relative_lateral_means(
    env: MujocoPushEnv, horizon: int, lateral_offset: float,
) -> list[np.ndarray]:
    """Generate 3 init means relative to object->goal direction."""
    obj_pos = env.get_object_pose()[:2]
    goal_pos = env.get_goal_pose()[:2]
    push_dir = goal_pos - obj_pos
    push_dir_norm = float(np.linalg.norm(push_dir))
    if push_dir_norm > 1e-6:
        push_dir = push_dir / push_dir_norm
    else:
        push_dir = np.array([1.0, 0.0])

    perp_dir = np.array([-push_dir[1], push_dir[0]])
    straight = np.zeros((horizon, 2), dtype=np.float64)

    mid = horizon // 2
    left = np.zeros((horizon, 2), dtype=np.float64)
    left[:mid] = perp_dir * lateral_offset
    left[mid:] = -perp_dir * lateral_offset * 0.5

    right = np.zeros((horizon, 2), dtype=np.float64)
    right[:mid] = -perp_dir * lateral_offset
    right[mid:] = perp_dir * lateral_offset * 0.5

    return [straight, left, right]


def parallel_multimodal_cem_plan(
    env: MujocoPushEnv,
    template: dict,
    state_payload: dict,
    horizon: int,
    num_samples: int,
    num_elites: int,
    num_iterations: int,
    seed: int,
    num_workers: int,
    env_config: dict,
    cost_weights_dict: dict,
    mp_start_method: str = "spawn",
    init_std: np.ndarray | None = None,
    lateral_offset: float = 0.5,
) -> ParallelMultimodalCEMResult:
    """Run 3 CEM modes (straight/left/right) with parallel eval, pick best."""
    modes = _make_goal_relative_lateral_means(env, horizon, lateral_offset)
    t_start = time.time()
    total_worker_eval = 0.0

    results: list[ParallelCEMResult] = []
    for i, mode_mean in enumerate(modes):
        r = parallel_cem_plan(
            env=env, template=template, state_payload=state_payload,
            horizon=horizon, num_samples=num_samples, num_elites=num_elites,
            num_iterations=num_iterations, seed=seed + i * 10000,
            num_workers=num_workers, env_config=env_config,
            cost_weights_dict=cost_weights_dict,
            mp_start_method=mp_start_method,
            init_mean=mode_mean, init_std=init_std,
        )
        results.append(r)
        total_worker_eval += r.worker_eval_time_sec

    mode_costs = [r.best_cost for r in results]
    best_mode = int(np.argmin(mode_costs))
    best_r = results[best_mode]
    plan_time = time.time() - t_start

    return ParallelMultimodalCEMResult(
        action_sequence=best_r.action_sequence,
        best_cost=best_r.best_cost,
        mean=best_r.mean, std=best_r.std,
        cost_history=best_r.cost_history,
        plan_time_sec=plan_time,
        worker_eval_time_sec=total_worker_eval,
        mode_costs=mode_costs, best_mode=best_mode,
    )


def debug_serial_compare(
    env: MujocoPushEnv,
    template: dict,
    horizon: int,
    num_samples: int,
    seed: int,
    num_workers: int,
    env_config: dict,
    cost_weights_dict: dict,
    mp_start_method: str = "spawn",
) -> None:
    """Compare serial and parallel CEM cost on the same samples."""
    print(f"\n{'='*60}")
    print(f"DEBUG: Serial vs Parallel CEM comparison ({num_samples} samples)")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=0.0, scale=0.8, size=(num_samples, horizon, 2))
    samples = np.clip(samples, -1.0, 1.0)

    state = env.clone_state()
    state_payload = state_to_payload(state)
    goal_pose = env.get_goal_pose()

    # Serial evaluation
    weights = make_default_mujoco_cost_weights()
    t0 = time.time()
    serial_costs = []
    for seq in samples:
        cost = mujoco_oracle_rollout_cost(
            env=env,
            action_sequence=seq,
            weights=weights,
            restore_state=True,
        )
        serial_costs.append(cost)
    serial_costs = np.asarray(serial_costs, dtype=np.float64)
    serial_time = time.time() - t0

    # Parallel evaluation (same index-based dispatching)
    t0 = time.time()
    all_indices = np.arange(num_samples)
    index_chunks = np.array_split(all_indices, num_workers)
    index_chunks = [c for c in index_chunks if len(c) > 0]
    actual_workers = len(index_chunks)

    ctx = mp.get_context(mp_start_method)

    parallel_costs = np.empty(num_samples, dtype=np.float64)
    with ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=ctx,
        initializer=init_worker_env,
        initargs=(template, env_config, cost_weights_dict),
    ) as executor:
        worker_args = [
            (idx_chunk, state_payload, samples[idx_chunk])
            for idx_chunk in index_chunks
        ]
        for indices, chunk_costs in executor.map(
            evaluate_action_chunk_worker, worker_args
        ):
            parallel_costs[indices] = chunk_costs

    parallel_time = time.time() - t0

    # Compare
    max_abs_diff = float(np.max(np.abs(serial_costs - parallel_costs)))
    mean_abs_diff = float(np.mean(np.abs(serial_costs - parallel_costs)))

    print(f"  Serial:   {serial_time:.3f}s  costs range=[{serial_costs.min():.4f}, {serial_costs.max():.4f}]")
    print(f"  Parallel: {parallel_time:.3f}s  costs range=[{parallel_costs.min():.4f}, {parallel_costs.max():.4f}]")
    print(f"  Max abs diff:  {max_abs_diff:.2e}")
    print(f"  Mean abs diff: {mean_abs_diff:.2e}")
    ok = max_abs_diff <= 1e-6
    print(f"  Result: {'PASS' if ok else 'FAIL'} (threshold=1e-6)")
    if not ok:
        worst_idx = int(np.argmax(np.abs(serial_costs - parallel_costs)))
        print(f"  Worst sample #{worst_idx}: serial={serial_costs[worst_idx]:.8f} parallel={parallel_costs[worst_idx]:.8f}")
    print(f"{'='*60}\n")


# ─── Rendering helpers (from render_closed_loop_rollout.py) ─────────────────

def setup_renderer(env: MujocoPushEnv, width: int, height: int, camera_name: str = "topdown"):
    """Initialize MuJoCo renderer, preferring named camera over programmatic camera."""
    gl_backend = os.environ.get("MUJOCO_GL", "not set")
    print(f"MUJOCO_GL backend: {gl_backend}")

    if gl_backend == "not set":
        print("WARNING: MUJOCO_GL not set. Renderer may fail.", file=sys.stderr)

    try:
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        print(f"Renderer initialized: {width}x{height}")

        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id >= 0:
            print(f"Using named camera: {camera_name} (id={cam_id})")
            return renderer, cam_id

        print(f"WARNING: Camera '{camera_name}' not found, using programmatic top-down camera")
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.35, 0.25, 0.0]
        camera.distance = 0.8
        camera.elevation = -90
        camera.azimuth = 90
        return renderer, camera
    except Exception as e:
        print(f"Failed to initialize renderer: {e}", file=sys.stderr)
        sys.exit(1)


def add_text_to_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Add text overlay to frame using PIL."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    text_height = 20 * len(lines)
    draw.rectangle([(10, 10), (650, 10 + text_height)], fill=(0, 0, 0))

    y_offset = 15
    for line in lines:
        draw.text((15, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += 20

    return np.array(img)


def render_frame_with_overlay(
    renderer, camera, env, template_id, mpc_step, exec_step_in_chunk,
    total_env_step, current_dist, current_theta_error_deg, contact_flag,
    object_displacement, planned_contact_first_step, execute_steps, horizon,
    strict_stop_active=False, parallel_cem=False, cem_workers=1,
    cem_plan_time=0.0, cem_best_cost=0.0,
    planned_collision_count=0.0, planned_collision_rate=0.0,
    warm_start=False,
) -> np.ndarray:
    """Render one frame with text overlay."""
    renderer.update_scene(env.data, camera=camera)
    pixels = renderer.render()

    stop_str = "ON" if strict_stop_active else "off"
    cem_str = f"parallel({cem_workers}w)" if parallel_cem else "serial"
    ws_str = "on" if warm_start else "off"
    text_lines = [
        f"Template: {template_id}",
        f"MPC Step: {mpc_step}",
        f"Exec Step: {exec_step_in_chunk}/{execute_steps}",
        f"Total Env Step: {total_env_step}",
        f"Dist: {current_dist*1000:.2f}mm  Theta: {current_theta_error_deg:.1f}deg",
        f"Contact: {contact_flag:.0f}  Collision: {planned_collision_count:.0f} ({planned_collision_rate:.2f})",
        f"Object Disp: {object_displacement:.4f}m",
        f"Planned Contact@: {planned_contact_first_step}",
        f"Horizon: {horizon}  StrictStop:{stop_str}",
        f"CEM: {cem_str}  best={cem_best_cost:.2f}  plan={cem_plan_time:.1f}s",
        f"WarmStart: {ws_str}",
    ]

    frame_with_text = add_text_to_frame(pixels, text_lines)
    return frame_with_text


def find_first_contact_step(rollout_result) -> int:
    """Find the first step where contact occurs in a rollout."""
    for i, contact in enumerate(rollout_result.contact_flags):
        if contact > 0.5:
            return i
    return -1


# ─── Main closed-loop with rendering ────────────────────────────────────────

def run_closed_loop_with_rendering(
    env: MujocoPushEnv,
    renderer, camera,
    template: dict,
    planning_horizon: int,
    num_samples: int,
    num_elites: int,
    num_iterations: int,
    execute_steps: int,
    max_mpc_steps: int,
    seed: int,
    success_dist_threshold: float,
    success_pos_threshold: float = 0.05,
    success_theta_threshold_deg: float = 180.0,
    parallel_cem: bool = False,
    cem_workers: int = 1,
    max_speed_mps: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
    pusher_mass: float = 0.05,
    warm_start: bool = False,
    mp_start_method: str = "spawn",
    planner_mode: str = "cem",
    mppi_temperature: float = 0.1,
    lateral_offset: float = 0.5,
) -> tuple[list[np.ndarray], dict]:
    """Run closed-loop MPC and render each execution step.

    Returns:
        (frames, diagnostics)
    """
    frames = []
    diag = {
        "total_plan_time": 0.0,
        "total_render_time": 0.0,
        "total_worker_eval_time": 0.0,
        "mpc_step_times": [],
    }

    initial_object_pose = env.get_object_pose()
    initial_goal_pose = env.get_goal_pose()
    initial_dist = float(np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2]))

    strict_stop_active = (
        success_theta_threshold_deg < 180.0
        or success_pos_threshold < success_dist_threshold
    )

    print(f"\nTemplate: {template['reset_template_id']}")
    print(f"Initial distance: {initial_dist:.4f}m")
    print(f"Planning horizon: {planning_horizon}, Execute steps: {execute_steps}")
    print(f"Strict pose stop: pos<={success_pos_threshold*1000:.1f}mm AND theta<={success_theta_threshold_deg:.1f}deg  (active={strict_stop_active})")
    print(f"Parallel CEM: {parallel_cem}, workers={cem_workers}, mp_start={mp_start_method}")
    print(f"Warm start: {warm_start}")
    print(f"Planner mode: {planner_mode}")

    weights = make_default_mujoco_cost_weights()

    # Prepare env_config and cost_weights for workers
    env_config = {
        "control_dt": 0.1,
        "max_speed_mps": max_speed_mps,
        "pusher_radius": pusher_radius,
        "pusher_halfheight": pusher_halfheight,
        "pusher_z": pusher_z,
        "pusher_mass": pusher_mass,
    }
    cost_weights_dict = {
        "w_pos": weights.w_pos,
        "w_theta": weights.w_theta,
        "w_reach": weights.w_reach,
        "w_no_contact": weights.w_no_contact,
        "w_push_alignment": weights.w_push_alignment,
        "w_collision": weights.w_collision,
        "w_collision_step": weights.w_collision_step,
        "w_proximity": weights.w_proximity,
        "w_action": weights.w_action,
        "w_smooth": weights.w_smooth,
        "w_subgoal": weights.w_subgoal,
    }

    total_env_step = 0
    success = False
    best_dist = initial_dist

    initial_theta_error_deg = float(np.rad2deg(abs(wrap_angle(
        initial_object_pose[2] - initial_goal_pose[2]
    ))))

    # Initial frame
    frame = render_frame_with_overlay(
        renderer=renderer, camera=camera, env=env,
        template_id=template["reset_template_id"],
        mpc_step=0, exec_step_in_chunk=0, total_env_step=0,
        current_dist=initial_dist, current_theta_error_deg=initial_theta_error_deg,
        contact_flag=0.0, object_displacement=0.0,
        planned_contact_first_step=-1, execute_steps=execute_steps,
        horizon=planning_horizon, strict_stop_active=strict_stop_active,
        parallel_cem=parallel_cem, cem_workers=cem_workers,
        warm_start=warm_start,
    )
    frames.append(frame)

    prev_mean = None
    prev_std = None

    for mpc_step in range(max_mpc_steps):
        current_object_pose = env.get_object_pose()
        current_dist = float(np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2]))

        print(f"\nMPC Step {mpc_step + 1}/{max_mpc_steps}, dist={current_dist:.4f}m")

        # ─── Determine warm-start init_mean/init_std ────────────────────
        if warm_start and prev_mean is not None and prev_std is not None:
            init_mean, init_std = shift_warm_start(
                prev_mean, prev_std, execute_steps, planning_horizon
            )
        else:
            init_mean = None
            init_std = None

        # ─── Planning (dispatch by planner_mode) ──────────────────────
        t_plan_start = time.time()

        use_parallel = parallel_cem and cem_workers > 1

        if planner_mode == "mppi":
            if use_parallel:
                state_payload = state_to_payload(env.clone_state())
                mppi_result = parallel_mppi_plan(
                    env=env,
                    template=template,
                    state_payload=state_payload,
                    horizon=planning_horizon,
                    num_samples=num_samples,
                    num_iterations=num_iterations,
                    temperature=mppi_temperature,
                    seed=seed + mpc_step,
                    num_workers=cem_workers,
                    env_config=env_config,
                    cost_weights_dict=cost_weights_dict,
                    mp_start_method=mp_start_method,
                    init_mean=init_mean,
                    init_std=init_std,
                )
                best_action_seq = mppi_result.action_sequence
                best_cost = mppi_result.best_cost
                plan_time = mppi_result.plan_time_sec
                worker_eval_time = mppi_result.worker_eval_time_sec
                prev_mean = mppi_result.mean
                prev_std = mppi_result.std
            else:
                from src.planners import MPPI as MPPIPlanner
                planner = MPPIPlanner(
                    horizon=planning_horizon,
                    action_dim=2,
                    num_samples=num_samples,
                    num_iterations=num_iterations,
                    action_low=[-1.0, -1.0],
                    action_high=[1.0, 1.0],
                    init_std=0.8,
                    temperature=mppi_temperature,
                    smoothing=0.2,
                    seed=seed + mpc_step,
                )
                def cost_fn(action_sequence: np.ndarray) -> float:
                    return mujoco_oracle_rollout_cost(
                        env=env, action_sequence=action_sequence,
                        weights=weights, restore_state=True,
                    )
                first_action, mppi_obj = planner.plan(cost_fn, init_mean=init_mean, init_std=init_std)
                best_action_seq = mppi_obj.action_sequence
                best_cost = mppi_obj.best_cost
                plan_time = time.time() - t_plan_start
                worker_eval_time = 0.0
                prev_mean = mppi_obj.mean
                prev_std = mppi_obj.std

        elif planner_mode == "multimodal_cem":
            if use_parallel:
                state_payload = state_to_payload(env.clone_state())
                mm_result = parallel_multimodal_cem_plan(
                    env=env,
                    template=template,
                    state_payload=state_payload,
                    horizon=planning_horizon,
                    num_samples=num_samples,
                    num_elites=num_elites,
                    num_iterations=num_iterations,
                    seed=seed + mpc_step,
                    num_workers=cem_workers,
                    env_config=env_config,
                    cost_weights_dict=cost_weights_dict,
                    mp_start_method=mp_start_method,
                    init_std=np.full((planning_horizon, 2), 0.8) if init_std is None else init_std,
                    lateral_offset=lateral_offset,
                )
                best_action_seq = mm_result.action_sequence
                best_cost = mm_result.best_cost
                plan_time = mm_result.plan_time_sec
                worker_eval_time = mm_result.worker_eval_time_sec
                prev_mean = mm_result.mean
                prev_std = mm_result.std
                print(f"  MultimodalCEM best_mode={mm_result.best_mode} mode_costs={[f'{c:.2f}' for c in mm_result.mode_costs]}")
            else:
                from src.planners import MultimodalCEMMPC as MMCEM
                planner = MMCEM(
                    horizon=planning_horizon,
                    action_dim=2,
                    num_samples=num_samples,
                    num_elites=num_elites,
                    num_iterations=num_iterations,
                    action_low=[-1.0, -1.0],
                    action_high=[1.0, 1.0],
                    init_std=0.8,
                    smoothing=0.2,
                    lateral_offset=lateral_offset,
                    seed=seed + mpc_step,
                )
                def cost_fn(action_sequence: np.ndarray) -> float:
                    return mujoco_oracle_rollout_cost(
                        env=env, action_sequence=action_sequence,
                        weights=weights, restore_state=True,
                    )
                first_action, mm_obj = planner.plan(cost_fn, init_mean=init_mean, init_std=init_std)
                best_action_seq = mm_obj.action_sequence
                best_cost = mm_obj.best_cost
                plan_time = time.time() - t_plan_start
                worker_eval_time = 0.0
                prev_mean = mm_obj.mean
                prev_std = mm_obj.std
                print(f"  MultimodalCEM best_mode={mm_obj.best_mode} mode_costs={[f'{c:.2f}' for c in mm_obj.mode_costs]}")

        else:  # "cem" (default)
            if use_parallel:
                state_payload = state_to_payload(env.clone_state())
                cem_result = parallel_cem_plan(
                    env=env,
                    template=template,
                    state_payload=state_payload,
                    horizon=planning_horizon,
                    num_samples=num_samples,
                    num_elites=num_elites,
                    num_iterations=num_iterations,
                    seed=seed + mpc_step,
                    num_workers=cem_workers,
                    env_config=env_config,
                    cost_weights_dict=cost_weights_dict,
                    mp_start_method=mp_start_method,
                    init_mean=init_mean,
                    init_std=init_std,
                )
                best_action_seq = cem_result.action_sequence
                best_cost = cem_result.best_cost
                plan_time = cem_result.plan_time_sec
                worker_eval_time = cem_result.worker_eval_time_sec
                prev_mean = cem_result.mean
                prev_std = cem_result.std
            else:
                planner = CEMMPC(
                    horizon=planning_horizon,
                    action_dim=2,
                    num_samples=num_samples,
                    num_elites=num_elites,
                    num_iterations=num_iterations,
                    action_low=[-1.0, -1.0],
                    action_high=[1.0, 1.0],
                    init_std=0.8,
                    smoothing=0.2,
                    seed=seed + mpc_step,
                )
                def cost_fn(action_sequence: np.ndarray) -> float:
                    return mujoco_oracle_rollout_cost(
                        env=env, action_sequence=action_sequence,
                        weights=weights, restore_state=True,
                    )
                first_action, cem_obj = planner.plan(cost_fn, init_mean=init_mean, init_std=init_std)
                best_action_seq = cem_obj.action_sequence
                best_cost = cem_obj.best_cost
                plan_time = time.time() - t_plan_start
                worker_eval_time = 0.0
                prev_mean = cem_obj.mean
                prev_std = cem_obj.std

        diag["total_plan_time"] += plan_time
        diag["total_worker_eval_time"] += worker_eval_time

        # Planned rollout for diagnostics
        planned_rollout = rollout_action_sequence_mujoco(
            env=env,
            action_sequence=best_action_seq,
            restore_state=True,
        )
        planned_contact_first_step = find_first_contact_step(planned_rollout)
        planned_collision_flags = planned_rollout.collision_flags
        planned_collision_count = float(np.sum(planned_collision_flags))
        planned_collision_rate = float(np.mean(planned_collision_flags))

        print(f"  CEM best_cost: {best_cost:.4f}")
        print(f"  Parallel CEM: {parallel_cem}, workers={cem_workers}, mp_start={mp_start_method}")
        print(f"  Plan time: {plan_time:.2f}s  worker_eval: {worker_eval_time:.2f}s")
        print(f"  Planned contact@step: {planned_contact_first_step}")
        print(f"  Planned collision: count={planned_collision_count:.0f} rate={planned_collision_rate:.3f}")
        print(f"  Warm start: {warm_start}")

        # ─── Execute and render ─────────────────────────────────────────
        t_render_start = time.time()
        actions_to_execute = best_action_seq[:execute_steps]

        for exec_step_idx, action in enumerate(actions_to_execute):
            env.step(action)
            total_env_step += 1

            current_object_pose = env.get_object_pose()
            current_dist = float(np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2]))
            current_theta_error_deg = float(np.rad2deg(abs(wrap_angle(
                current_object_pose[2] - initial_goal_pose[2]
            ))))
            contact_flag = env.get_contact_flag()
            object_displacement = float(
                np.linalg.norm(current_object_pose[:2] - initial_object_pose[:2])
            )

            if current_dist < best_dist:
                best_dist = current_dist

            frame = render_frame_with_overlay(
                renderer=renderer, camera=camera, env=env,
                template_id=template["reset_template_id"],
                mpc_step=mpc_step + 1, exec_step_in_chunk=exec_step_idx + 1,
                total_env_step=total_env_step,
                current_dist=current_dist, current_theta_error_deg=current_theta_error_deg,
                contact_flag=contact_flag, object_displacement=object_displacement,
                planned_contact_first_step=planned_contact_first_step,
                execute_steps=execute_steps, horizon=planning_horizon,
                strict_stop_active=strict_stop_active,
                parallel_cem=parallel_cem, cem_workers=cem_workers,
                cem_plan_time=plan_time, cem_best_cost=best_cost,
                planned_collision_count=planned_collision_count,
                planned_collision_rate=planned_collision_rate,
                warm_start=warm_start,
            )
            frames.append(frame)

            # Strict pose early stop: both pos AND theta must be satisfied
            _pos_ok = current_dist <= success_pos_threshold
            _theta_ok = current_theta_error_deg <= success_theta_threshold_deg
            if _pos_ok and _theta_ok:
                success = True
                print(f"\n  STRICT POSE STOP at step {total_env_step}! dist={current_dist*1000:.2f}mm theta={current_theta_error_deg:.1f}deg")
                break
            # Legacy distance-only stop (only if strict stop not active)
            if not strict_stop_active and current_dist <= success_dist_threshold:
                success = True
                print(f"\n  SUCCESS at step {total_env_step}!")
                break

        render_time = time.time() - t_render_start
        diag["total_render_time"] += render_time
        diag["mpc_step_times"].append({
            "mpc_step": mpc_step + 1,
            "plan_time_sec": plan_time,
            "worker_eval_time_sec": worker_eval_time,
            "render_time_sec": render_time,
            "best_cost": best_cost,
            "dist": current_dist,
            "planned_collision_count": planned_collision_count,
            "planned_collision_rate": planned_collision_rate,
        })

        print(f"  Execute+Render time: {render_time:.2f}s")

        if success:
            break

    print(f"\nRendered {len(frames)} frames")
    print(f"Success: {success}, Best dist: {best_dist:.4f}m")

    return frames, diag


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render closed-loop MuJoCo oracle-MPC with parallel CEM"
    )

    parser.add_argument("--templates", type=str,
                        default="data/sim/metadata/reset_templates_v0.json")
    parser.add_argument("--split", type=str, default="train_sim_id")
    parser.add_argument("--template-index", type=int, default=0)

    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-mpc-steps", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=1536)
    parser.add_argument("--num-elites", type=int, default=128)
    parser.add_argument("--num-iterations", type=int, default=7)

    parser.add_argument("--out-video", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--success-dist-threshold", type=float, default=0.05)
    parser.add_argument("--success-pos-threshold", type=float, default=0.05)
    parser.add_argument("--stop-pos-threshold", type=float, default=None,
                        dest="success_pos_threshold")
    parser.add_argument("--success-theta-threshold-deg", type=float, default=180.0)
    parser.add_argument("--stop-theta-threshold-deg", type=float, default=None,
                        dest="success_theta_threshold_deg")
    parser.add_argument("--strict-pose-stop", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=str, default="topdown")

    parser.add_argument("--pusher-radius", type=float, default=None)
    parser.add_argument("--pusher-mass", type=float, default=0.05, help="Pusher mass in kg")
    parser.add_argument("--pusher-halfheight", type=float, default=None)
    parser.add_argument("--pusher-z", type=float, default=None)
    parser.add_argument("--max-speed-mps", type=float, default=0.05,
                        help="Pusher max speed m/s (default: 0.05).")

    # Parallel CEM options
    parser.add_argument("--parallel-cem", action="store_true", default=False,
                        help="Enable parallel CEM cost evaluation.")
    parser.add_argument("--cem-workers", type=int, default=1,
                        help="Number of CEM worker processes (default: 1 = serial).")
    parser.add_argument("--mp-start-method", type=str, default="spawn",
                        choices=["spawn", "forkserver", "fork"],
                        help="Multiprocessing start method (default: spawn).")

    # Planner mode selection
    parser.add_argument("--planner-mode", type=str, default="cem",
                        choices=["cem", "multimodal_cem", "mppi"],
                        help="Planner algorithm (default: cem).")
    parser.add_argument("--mppi-temperature", type=float, default=0.1,
                        help="MPPI temperature (default: 0.1). Lower=greedy, higher=exploratory.")
    parser.add_argument("--lateral-offset", type=float, default=0.5,
                        help="MultimodalCEM lateral offset in action space (default: 0.5).")

    # Warm start
    parser.add_argument("--warm-start", action="store_true", default=False,
                        help="Enable warm-start CEM (shift previous mean/std). Default: off.")

    # Debug compare
    parser.add_argument("--debug-serial-compare", action="store_true", default=False,
                        help="Run serial vs parallel comparison on first MPC step.")
    parser.add_argument("--compare-num-samples", type=int, default=64,
                        help="Number of samples for debug comparison (default: 64).")
    parser.add_argument("--compare-only", action="store_true", default=False,
                        help="Exit after serial/parallel compare (no rendering).")

    return parser.parse_args()


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Handle --strict-pose-stop
    if args.strict_pose_stop:
        if args.success_pos_threshold >= 0.05:
            args.success_pos_threshold = 0.0015
        if args.success_theta_threshold_deg >= 180.0:
            args.success_theta_threshold_deg = 3.0

    # Parameter guards
    if args.cem_workers < 1:
        raise ValueError(f"cem_workers must be >= 1, got {args.cem_workers}")
    if args.num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {args.num_samples}")
    if args.num_elites > args.num_samples:
        raise ValueError(
            f"num_elites ({args.num_elites}) must be <= num_samples ({args.num_samples})"
        )

    template_path = Path(args.templates)
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    templates = load_reset_templates(template_path)
    templates = [t for t in templates if t["split"] == args.split]
    if not templates:
        print(f"ERROR: No templates found for split={args.split}", file=sys.stderr)
        sys.exit(1)

    if args.template_index < 0 or args.template_index >= len(templates):
        print(f"ERROR: template-index {args.template_index} out of range", file=sys.stderr)
        sys.exit(1)

    template = templates[args.template_index]

    print(f"Selected template: {template['reset_template_id']}")
    print(f"  split: {template['split']}")
    print(f"  layout_family: {template['layout_family']}")
    print(f"  shape_family: {template['shape_family']}")

    pusher_radius = args.pusher_radius if args.pusher_radius is not None else 0.010
    pusher_halfheight = args.pusher_halfheight if args.pusher_halfheight is not None else 0.014
    pusher_z = args.pusher_z if args.pusher_z is not None else 0.016
    pusher_mass = args.pusher_mass if args.pusher_mass is not None else 0.05

    print(f"\nPusher geometry:")
    print(f"  radius: {pusher_radius:.4f}m")
    print(f"  halfheight: {pusher_halfheight:.4f}m")
    print(f"  z: {pusher_z:.4f}m")
    print(f"  max_speed: {args.max_speed_mps:.3f} m/s")

    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=args.max_speed_mps,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
        pusher_mass=args.pusher_mass,
    )
    env.reset_from_template(template)

    # ─── Debug compare-only mode: run before renderer init ──────────────
    if args.debug_serial_compare:
        weights = make_default_mujoco_cost_weights()
        env_config = {
            "pusher_mass": args.pusher_mass,
            "control_dt": 0.1,
            "max_speed_mps": args.max_speed_mps,
            "pusher_radius": pusher_radius,
            "pusher_halfheight": pusher_halfheight,
            "pusher_z": pusher_z,
        }
        cost_weights_dict = {
            "w_pos": weights.w_pos, "w_theta": weights.w_theta,
            "w_reach": weights.w_reach, "w_no_contact": weights.w_no_contact,
            "w_push_alignment": weights.w_push_alignment,
            "w_collision": weights.w_collision, "w_collision_step": weights.w_collision_step,
            "w_action": weights.w_action, "w_smooth": weights.w_smooth,
            "w_subgoal": weights.w_subgoal,
        }
        debug_serial_compare(
            env=env, template=template,
            horizon=args.horizon,
            num_samples=args.compare_num_samples,
            seed=args.seed,
            num_workers=args.cem_workers,
            env_config=env_config,
            cost_weights_dict=cost_weights_dict,
            mp_start_method=args.mp_start_method,
        )
        if args.compare_only:
            print("Compare-only mode: exiting after serial/parallel comparison.")
            return

    # ─── Normal rendering path ──────────────────────────────────────────
    renderer, camera = setup_renderer(env, args.width, args.height, camera_name=args.camera)

    parallel_cem = args.parallel_cem and args.cem_workers > 1

    print(f"\n{'='*60}")
    print(f"Starting closed-loop rendering")
    print(f"  planner_mode={args.planner_mode}")
    print(f"  parallel_cem={parallel_cem}")
    print(f"  cem_workers={args.cem_workers if parallel_cem else 1}")
    print(f"  mp_start_method={args.mp_start_method}")
    print(f"  warm_start={args.warm_start}")
    print(f"  num_samples={args.num_samples}")
    print(f"  num_iterations={args.num_iterations}")
    print(f"  max_mpc_steps={args.max_mpc_steps}")
    print(f"  max_speed_mps={args.max_speed_mps}")
    if args.planner_mode == "mppi":
        print(f"  mppi_temperature={args.mppi_temperature}")
    if args.planner_mode == "multimodal_cem":
        print(f"  lateral_offset={args.lateral_offset}")
    print(f"{'='*60}")

    t_total_start = time.time()

    frames, diag = run_closed_loop_with_rendering(
        env=env, renderer=renderer, camera=camera,
        template=template,
        planning_horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        execute_steps=args.execute_steps,
        max_mpc_steps=args.max_mpc_steps,
        seed=args.seed,
        success_dist_threshold=args.success_dist_threshold,
        success_pos_threshold=args.success_pos_threshold,
        success_theta_threshold_deg=args.success_theta_threshold_deg,
        parallel_cem=parallel_cem,
        cem_workers=args.cem_workers,
        max_speed_mps=args.max_speed_mps,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
        pusher_mass=args.pusher_mass,
        warm_start=args.warm_start,
        planner_mode=args.planner_mode,
        mppi_temperature=args.mppi_temperature,
        lateral_offset=args.lateral_offset,
    )

    t_total = time.time() - t_total_start

    # Save video
    if args.out_video is None:
        out_dir = Path("artifacts/videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "parallel_cem" if parallel_cem else "serial"
        out_video = out_dir / f"closed_loop_{mode}_{template['reset_template_id']}.mp4"
    else:
        out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving video to: {out_video}")
    imageio.mimsave(out_video, frames, fps=args.fps, codec="libx264", quality=8)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Video: {out_video}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Total runtime: {t_total:.1f}s")
    print(f"  Total plan time: {diag['total_plan_time']:.1f}s")
    print(f"  Total render time: {diag['total_render_time']:.1f}s")
    print(f"  Total worker eval time: {diag['total_worker_eval_time']:.1f}s")
    print(f"  Planner mode: {args.planner_mode}")
    print(f"  Speed config: max_speed_mps={args.max_speed_mps}")
    print(f"  Parallel CEM: {parallel_cem}, workers={args.cem_workers if parallel_cem else 1}")
    print(f"  MP start method: {args.mp_start_method}")
    print(f"  Warm start: {args.warm_start}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
