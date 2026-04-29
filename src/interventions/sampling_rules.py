from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.interventions.layout_families import sample_layout_family
from src.interventions.shape_families import get_shape_spec


SPLIT_SPECS = [
    {
        "split": "train_sim_id",
        "domain": "sim",
        "layout_families": ["open_space", "mild_offset"],
        "shape_families": ["T_shape"],
    },
    {
        "split": "val_sim_id",
        "domain": "sim",
        "layout_families": ["open_space", "mild_offset"],
        "shape_families": ["T_shape"],
    },
    {
        "split": "test_sim_id",
        "domain": "sim",
        "layout_families": ["open_space", "mild_offset"],
        "shape_families": ["T_shape"],
    },
    {
        "split": "test_sim_layout_ood_blocking",
        "domain": "sim",
        "layout_families": ["blocking"],
        "shape_families": ["T_shape"],
    },
    {
        "split": "test_sim_layout_ood_narrow_passage",
        "domain": "sim",
        "layout_families": ["narrow_passage"],
        "shape_families": ["T_shape"],
    },
    {
        "split": "test_sim_layout_ood_edge_goal",
        "domain": "sim",
        "layout_families": ["edge_goal"],
        "shape_families": ["T_shape"],
    },
    {
        "split": "test_sim_shape_ood_L",
        "domain": "sim",
        "layout_families": ["open_space", "mild_offset"],
        "shape_families": ["L_shape"],
    },
]


def _safe_id(text: str) -> str:
    safe = (
        text.strip()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )

    while "__" in safe:
        safe = safe.replace("__", "_")

    return safe


def build_reset_template(
    split: str,
    domain: str,
    layout_family: str,
    shape_family: str,
    index: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    layout = sample_layout_family(layout_family=layout_family, rng=rng)
    shape = get_shape_spec(shape_family)

    reset_template_id = (
        f"{_safe_id(split)}__"
        f"{_safe_id(layout_family)}__"
        f"{_safe_id(shape_family)}__"
        f"{index:06d}"
    )

    template = {
        "schema_version": "reset_template_v0.1",
        "reset_template_id": reset_template_id,
        "domain": domain,
        "split": split,
        "layout_family": layout_family,
        "shape_family": shape_family,
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

    return template


def build_reset_templates(
    num_per_split: int = 20,
    base_seed: int = 42,
    split_specs: Iterable[dict] = SPLIT_SPECS,
) -> list[dict]:
    if num_per_split <= 0:
        raise ValueError(f"num_per_split must be positive, got {num_per_split}")

    templates: list[dict] = []
    global_index = 0

    for split_spec in split_specs:
        split = split_spec["split"]
        domain = split_spec["domain"]
        layout_families = split_spec["layout_families"]
        shape_families = split_spec["shape_families"]

        if not layout_families:
            raise ValueError(f"split={split} has empty layout_families")

        if not shape_families:
            raise ValueError(f"split={split} has empty shape_families")

        for local_index in range(num_per_split):
            layout_family = layout_families[local_index % len(layout_families)]
            shape_family = shape_families[local_index % len(shape_families)]

            seed = base_seed + global_index

            template = build_reset_template(
                split=split,
                domain=domain,
                layout_family=layout_family,
                shape_family=shape_family,
                index=local_index,
                seed=seed,
            )

            templates.append(template)
            global_index += 1

    validate_reset_templates(templates)
    return templates


def validate_reset_templates(templates: list[dict]) -> None:
    if not templates:
        raise ValueError("No reset templates generated.")

    ids = [t["reset_template_id"] for t in templates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reset_template_id found.")

    required_keys = [
        "schema_version",
        "reset_template_id",
        "domain",
        "split",
        "layout_family",
        "shape_family",
        "seed",
        "object_shape",
        "object_size_x",
        "object_size_y",
        "object_mass",
        "object_friction",
        "object_initial_pose",
        "goal_pose",
        "ee_initial_pose",
        "obstacles",
    ]

    for template in templates:
        for key in required_keys:
            if key not in template:
                raise ValueError(
                    f"Missing key={key} in template={template.get('reset_template_id')}"
                )

        if template["schema_version"] != "reset_template_v0.1":
            raise ValueError(
                f"Unsupported schema_version={template['schema_version']} "
                f"in template={template['reset_template_id']}"
            )

        if template["domain"] != "sim":
            raise ValueError(f"v0.1 only supports sim templates: {template['domain']}")

        if not template["split"]:
            raise ValueError(f"Empty split in template={template['reset_template_id']}")

        if not template["layout_family"]:
            raise ValueError(
                f"Empty layout_family in template={template['reset_template_id']}"
            )

        if not template["shape_family"]:
            raise ValueError(
                f"Empty shape_family in template={template['reset_template_id']}"
            )

        if not template["object_shape"]:
            raise ValueError(
                f"Empty object_shape in template={template['reset_template_id']}"
            )

        if template["object_size_x"] <= 0 or template["object_size_y"] <= 0:
            raise ValueError(
                f"Invalid object size in template={template['reset_template_id']}"
            )

        if template["object_mass"] is not None and template["object_mass"] <= 0:
            raise ValueError(
                f"Invalid object_mass in template={template['reset_template_id']}"
            )

        if template["object_friction"] is not None and template["object_friction"] < 0:
            raise ValueError(
                f"Invalid object_friction in template={template['reset_template_id']}"
            )

        for pose_key in ["object_initial_pose", "goal_pose", "ee_initial_pose"]:
            pose = template[pose_key]
            for field in ["x", "y", "theta"]:
                if field not in pose:
                    raise ValueError(
                        f"Missing pose field {field} in {pose_key} "
                        f"for template={template['reset_template_id']}"
                    )

        for obs in template["obstacles"]:
            if "obstacle_id" not in obs:
                raise ValueError(
                    f"Obstacle missing obstacle_id in template="
                    f"{template['reset_template_id']}"
                )

            if "pose" not in obs:
                raise ValueError(
                    f"Obstacle missing pose in template="
                    f"{template['reset_template_id']}"
                )

            if obs["size_x"] <= 0 or obs["size_y"] <= 0:
                raise ValueError(
                    f"Invalid obstacle size in template="
                    f"{template['reset_template_id']}"
                )


def save_reset_templates(templates: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)


def summarize_reset_templates(templates: list[dict]) -> dict:
    return {
        "num_templates": len(templates),
        "by_split": dict(Counter(t["split"] for t in templates)),
        "by_layout_family": dict(Counter(t["layout_family"] for t in templates)),
        "by_shape_family": dict(Counter(t["shape_family"] for t in templates)),
    }