from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.metadata_schema import (
    EpisodeMetadata,
    ObstacleMetadata,
    Pose2D,
)
from src.interventions.sampling_rules import validate_reset_templates


def load_reset_templates(path: str | Path) -> list[dict]:
    """
    Load reset templates from JSON and validate schema.

    Expected input:
        data/sim/metadata/reset_templates_v0.json
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        templates = json.load(f)

    if not isinstance(templates, list):
        raise ValueError(f"Expected a list of templates, got {type(templates)}")

    validate_reset_templates(templates)

    return templates


def summarize_templates(templates: list[dict]) -> dict:
    """
    Summarize reset templates by split, layout family, and shape family.
    """
    return {
        "num_templates": len(templates),
        "by_split": dict(Counter(t["split"] for t in templates)),
        "by_layout_family": dict(Counter(t["layout_family"] for t in templates)),
        "by_shape_family": dict(Counter(t["shape_family"] for t in templates)),
    }


def get_templates_by_split(templates: list[dict], split: str) -> list[dict]:
    """
    Return all templates from a given split.
    """
    selected = [t for t in templates if t["split"] == split]

    if not selected:
        raise ValueError(f"No reset templates found for split={split}")

    return selected


def get_template_by_id(templates: list[dict], reset_template_id: str) -> dict:
    """
    Return one template by reset_template_id.
    """
    matches = [t for t in templates if t["reset_template_id"] == reset_template_id]

    if len(matches) == 0:
        raise ValueError(f"No template found with id={reset_template_id}")

    if len(matches) > 1:
        raise ValueError(f"Duplicate template id={reset_template_id}")

    return matches[0]


def _pose_from_dict(data: dict) -> Pose2D:
    return Pose2D(
        x=float(data["x"]),
        y=float(data["y"]),
        theta=float(data.get("theta", 0.0)),
    )


def _obstacle_from_dict(data: dict) -> ObstacleMetadata:
    return ObstacleMetadata(
        obstacle_id=data["obstacle_id"],
        pose=_pose_from_dict(data["pose"]),
        size_x=float(data["size_x"]),
        size_y=float(data["size_y"]),
        shape=data.get("shape", "box"),
        valid=bool(data.get("valid", True)),
    )


def template_to_episode_metadata(
    template: dict,
    episode_id: str | None = None,
) -> EpisodeMetadata:
    """
    Convert one reset template into an EpisodeMetadata object.

    This is useful before MuJoCo is connected:

        reset template
        → planned episode metadata
        → validate / save
        → later fill rollout result fields
    """
    validate_reset_templates([template])

    reset_template_id = template["reset_template_id"]

    if episode_id is None:
        episode_id = f"episode__{reset_template_id}"

    obstacles = [_obstacle_from_dict(obs) for obs in template.get("obstacles", [])]

    metadata = EpisodeMetadata(
        episode_id=episode_id,
        schema_version="v0.1",
        domain=template["domain"],
        split=template["split"],
        layout_family=template["layout_family"],
        shape_family=template["shape_family"],
        reset_template_id=reset_template_id,
        seed=template.get("seed"),
        object_shape=template["object_shape"],
        object_initial_pose=_pose_from_dict(template["object_initial_pose"]),
        goal_pose=_pose_from_dict(template["goal_pose"]),
        ee_initial_pose=_pose_from_dict(template["ee_initial_pose"]),
        object_size_x=float(template["object_size_x"]),
        object_size_y=float(template["object_size_y"]),
        object_mass=template.get("object_mass"),
        object_friction=template.get("object_friction"),
        obstacles=obstacles,
        num_steps=0,
        control_dt=0.1,
        success=None,
        failure_code=None,
        notes="Generated from reset template before environment rollout.",
    )

    metadata.validate()

    return metadata