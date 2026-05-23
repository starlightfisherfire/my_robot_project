#!/usr/bin/env python3
"""
Generate obstacle difficulty ladder reset templates for Oracle-MPC capacity gate.

Produces:
  data/sim/metadata/reset_templates_obstacle_difficulty_v0.json

Groups (60 templates total, 10 per condition):
  - blocking_easy      (1 obstacle, 0.04 x 0.08 m)
  - blocking_medium    (1 obstacle, 0.05 x 0.10 m)
  - blocking_hard      (1 obstacle, 0.06 x 0.12 m)
  - passage_direct_wide       (2 obstacles, effective_passage_gap = 0.09 m)
  - passage_direct_medium     (2 obstacles, effective_passage_gap = 0.07 m)
  - passage_direct_narrow   (2 obstacles, effective_passage_gap = 0.05 m)

IMPORTANT: passage_gap = effective inner wall gap (NOT center-to-center).
  passage_center_distance = effective_passage_gap + obstacle_size_y
  passage_gap_definition = "inner_wall_gap"

Does NOT call MuJoCo or run MPC.
May overwrite the output JSON only when --overwrite is provided.
Does NOT modify data/sim/metadata/reset_templates_v0.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.interventions.shape_families import get_shape_spec
from src.interventions.sampling_rules import (
    save_reset_templates,
    summarize_reset_templates,
    validate_reset_templates,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_pose(x: float, y: float, theta: float = 0.0) -> dict:
    return {"x": float(x), "y": float(y), "theta": float(theta)}


def _make_obstacle(
    obstacle_id: str,
    x: float,
    y: float,
    size_x: float,
    size_y: float,
    theta: float = 0.0,
    shape: str = "box",
    valid: bool = True,
) -> dict:
    return {
        "obstacle_id": obstacle_id,
        "pose": _make_pose(x=x, y=y, theta=theta),
        "size_x": float(size_x),
        "size_y": float(size_y),
        "shape": shape,
        "valid": bool(valid),
    }


def _jitter(rng: random.Random, value: float, scale: float) -> float:
    return float(value + rng.uniform(-scale, scale))


def _safe_id(text: str) -> str:
    safe = text.strip().replace("/", "_").replace(" ", "_").replace("-", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


# ---------------------------------------------------------------------------
# blocking layout sampler (single obstacle near object->goal midpoint)
# ---------------------------------------------------------------------------

def _sample_blocking(
    rng: random.Random,
    size_x: float,
    size_y: float,
) -> dict:
    object_pose = _make_pose(
        x=_jitter(rng, 0.18, 0.025),
        y=_jitter(rng, 0.20, 0.025),
        theta=rng.uniform(-0.25, 0.25),
    )
    goal_pose = _make_pose(
        x=_jitter(rng, 0.48, 0.035),
        y=_jitter(rng, 0.30, 0.035),
        theta=rng.uniform(-0.25, 0.25),
    )

    mid_x = 0.5 * (object_pose["x"] + goal_pose["x"])
    mid_y = 0.5 * (object_pose["y"] + goal_pose["y"])

    obstacle = _make_obstacle(
        obstacle_id="obs_blocking_0",
        x=_jitter(rng, mid_x, 0.03),
        y=_jitter(rng, mid_y, 0.03),
        size_x=size_x,
        size_y=size_y,
        theta=rng.uniform(-0.5, 0.5),
    )

    ee_pose = _make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    return {
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": [obstacle],
    }


# ---------------------------------------------------------------------------
# passage layout sampler (two obstacles forming a channel)
# effective_passage_gap = inner wall gap (distance between inner faces)
# passage_center_distance = effective_passage_gap + obstacle_size_y
# ---------------------------------------------------------------------------

def _sample_passage(
    rng: random.Random,
    effective_passage_gap: float,
    obs_size_x: float = 0.08,
    obs_size_y: float = 0.06,
) -> dict:
    object_pose = _make_pose(
        x=_jitter(rng, 0.18, 0.025),
        y=_jitter(rng, 0.20, 0.025),
        theta=rng.uniform(-0.25, 0.25),
    )
    goal_pose = _make_pose(
        x=_jitter(rng, 0.50, 0.035),
        y=_jitter(rng, 0.20, 0.03),
        theta=rng.uniform(-0.25, 0.25),
    )

    # Passage centre along object->goal midpoint
    passage_x = 0.5 * (object_pose["x"] + goal_pose["x"])
    passage_y = 0.5 * (object_pose["y"] + goal_pose["y"])

    # Center-to-center distance = effective gap + obstacle thickness
    passage_center_distance = effective_passage_gap + obs_size_y
    half_center_dist = passage_center_distance / 2.0

    # Shared x jitter so top/bottom obstacles stay aligned
    passage_x_jittered = _jitter(rng, passage_x, 0.02)

    obstacles = [
        _make_obstacle(
            obstacle_id="obs_passage_top",
            x=passage_x_jittered,
            y=passage_y + half_center_dist,
            size_x=obs_size_x,
            size_y=obs_size_y,
            theta=0.0,
        ),
        _make_obstacle(
            obstacle_id="obs_passage_bottom",
            x=passage_x_jittered,
            y=passage_y - half_center_dist,
            size_x=obs_size_x,
            size_y=obs_size_y,
            theta=0.0,
        ),
    ]

    ee_pose = _make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    return {
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": obstacles,
        "_passage_gap": effective_passage_gap,
        "_effective_passage_gap": effective_passage_gap,
        "_passage_center_distance": passage_center_distance,
        "_passage_center": {"x": passage_x, "y": passage_y},
    }


# ---------------------------------------------------------------------------
# split specs
# ---------------------------------------------------------------------------

def _build_split_specs() -> list[dict]:
    return [
        # --- blocking difficulty ladder ---
        {
            "split": "test_sim_layout_ood_blocking_easy",
            "layout_family": "blocking_easy",
            "type": "blocking",
            "obstacle_size_x": 0.04,
            "obstacle_size_y": 0.08,
        },
        {
            "split": "test_sim_layout_ood_blocking_medium",
            "layout_family": "blocking_medium",
            "type": "blocking",
            "obstacle_size_x": 0.05,
            "obstacle_size_y": 0.10,
        },
        {
            "split": "test_sim_layout_ood_blocking_hard",
            "layout_family": "blocking_hard",
            "type": "blocking",
            "obstacle_size_x": 0.06,
            "obstacle_size_y": 0.12,
        },
        # --- passage difficulty ladder (effective inner wall gaps) ---
        {
            "split": "test_sim_layout_ood_passage_direct_wide",
            "layout_family": "passage_direct_wide",
            "type": "passage",
            "effective_passage_gap": 0.09,
            "obs_size_x": 0.08,
            "obs_size_y": 0.06,
        },
        {
            "split": "test_sim_layout_ood_passage_direct_medium",
            "layout_family": "passage_direct_medium",
            "type": "passage",
            "effective_passage_gap": 0.07,
            "obs_size_x": 0.08,
            "obs_size_y": 0.06,
        },
        {
            "split": "test_sim_layout_ood_passage_direct_narrow",
            "layout_family": "passage_direct_narrow",
            "type": "passage",
            "effective_passage_gap": 0.05,
            "obs_size_x": 0.08,
            "obs_size_y": 0.06,
        },
    ]


# ---------------------------------------------------------------------------
# template builder
# ---------------------------------------------------------------------------

def build_obstacle_difficulty_templates(
    num_per_condition: int = 10,
    base_seed: int = 20260513,
) -> list[dict]:
    specs = _build_split_specs()
    shape = get_shape_spec("T_shape")
    templates: list[dict] = []
    global_index = 0

    for spec in specs:
        split = spec["split"]
        layout_family = spec["layout_family"]

        for local_index in range(num_per_condition):
            seed = base_seed + global_index
            rng = random.Random(seed)

            if spec["type"] == "passage":
                layout = _sample_passage(
                    rng,
                    effective_passage_gap=spec["effective_passage_gap"],
                    obs_size_x=spec["obs_size_x"],
                    obs_size_y=spec["obs_size_y"],
                )
            else:
                layout = _sample_blocking(
                    rng,
                    size_x=spec["obstacle_size_x"],
                    size_y=spec["obstacle_size_y"],
                )

            reset_template_id = (
                f"{_safe_id(split)}__"
                f"{_safe_id(layout_family)}__"
                f"{_safe_id('T_shape')}__"
                f"{local_index:06d}"
            )

            template = {
                "schema_version": "reset_template_v0.1",
                "reset_template_id": reset_template_id,
                "domain": "sim",
                "split": split,
                "layout_family": layout_family,
                "shape_family": "T_shape",
                "seed": seed,
                "object_shape": shape["object_shape"],
                "object_size_x": shape["object_size_x"],
                "object_size_y": shape["object_size_y"],
                "object_mass": shape["object_mass"],
                "object_friction": shape["object_friction"],
                "object_initial_pose": layout["object_initial_pose"],
                "goal_pose": layout["goal_pose"],
                "ee_initial_pose": layout["ee_initial_pose"],
                "obstacles": layout["obstacles"],
            }

            # Per-type metadata fields
            if spec["type"] == "passage":
                template["passage_gap"] = layout["_passage_gap"]
                template["effective_passage_gap"] = layout["_effective_passage_gap"]
                template["passage_center_distance"] = layout["_passage_center_distance"]
                template["passage_gap_definition"] = "inner_wall_gap"
                template["passage_center"] = layout["_passage_center"]
                template["obstacle_size_x"] = spec["obs_size_x"]
                template["obstacle_size_y"] = spec["obs_size_y"]

                # Assertions
                assert template["effective_passage_gap"] == template["passage_gap"]
                assert template["passage_center_distance"] == (
                    template["effective_passage_gap"] + spec["obs_size_y"]
                )
                assert template["passage_gap_definition"] == "inner_wall_gap"
            else:
                template["blocking_difficulty"] = layout_family.replace(
                    "blocking_", ""
                )
                template["obstacle_size_x"] = spec["obstacle_size_x"]
                template["obstacle_size_y"] = spec["obstacle_size_y"]

            templates.append(template)
            global_index += 1

    # Passage gap assertions
    for t in templates:
        if t["layout_family"] == "passage_direct_wide":
            assert abs(t["effective_passage_gap"] - 0.09) < 1e-6, (
                f"passage_direct_wide effective_gap={t['effective_passage_gap']}"
            )
        elif t["layout_family"] == "passage_direct_medium":
            assert abs(t["effective_passage_gap"] - 0.07) < 1e-6, (
                f"passage_direct_medium effective_gap={t['effective_passage_gap']}"
            )
        elif t["layout_family"] == "passage_direct_narrow":
            assert abs(t["effective_passage_gap"] - 0.05) < 1e-6, (
                f"passage_direct_narrow effective_gap={t['effective_passage_gap']}"
            )

    validate_reset_templates(templates)
    return templates


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate obstacle difficulty ladder reset templates.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--num-per-condition",
        type=int,
        default=10,
        help="Number of templates per difficulty condition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260513,
        help="Base random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output file if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        print(f"ERROR: {out_path} already exists. Use --overwrite to replace.")
        raise SystemExit(1)

    templates = build_obstacle_difficulty_templates(
        num_per_condition=args.num_per_condition,
        base_seed=args.seed,
    )

    save_reset_templates(templates, out_path)

    summary = summarize_reset_templates(templates)

    print(f"Generated {summary['num_templates']} obstacle difficulty templates.")
    print(f"By split:")
    print(json.dumps(summary["by_split"], indent=2, ensure_ascii=False))
    print(f"By layout_family:")
    print(json.dumps(summary["by_layout_family"], indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")

    # Print passage details
    print("\n--- Passage Gap Summary ---")
    passage_families = ["passage_direct_wide", "passage_direct_medium", "passage_direct_narrow"]
    for fam in passage_families:
        fam_templates = [t for t in templates if t["layout_family"] == fam]
        if fam_templates:
            t0 = fam_templates[0]
            print(f"  {fam}:")
            print(f"    effective_passage_gap    = {t0['effective_passage_gap']:.3f} m")
            print(f"    passage_center_distance  = {t0['passage_center_distance']:.3f} m")
            print(f"    obstacle_size_y          = {t0['obstacle_size_y']:.3f} m")
            print(f"    passage_gap_definition   = {t0['passage_gap_definition']}")
            print(f"    count                    = {len(fam_templates)}")

    print("\n--- Blocking Summary ---")
    blocking_families = ["blocking_easy", "blocking_medium", "blocking_hard"]
    for fam in blocking_families:
        fam_templates = [t for t in templates if t["layout_family"] == fam]
        if fam_templates:
            t0 = fam_templates[0]
            print(f"  {fam}:")
            print(f"    obstacle_size = {t0['obstacle_size_x']:.3f} x {t0['obstacle_size_y']:.3f} m")
            print(f"    count         = {len(fam_templates)}")

    print("\nObstacle difficulty template generation ok.")


if __name__ == "__main__":
    main()
