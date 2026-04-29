"""
Smoke test for reset templates.

This script verifies:

reset_templates_v0.json
→ load
→ validate
→ summarize
→ query by split
→ query by id
→ convert reset template to EpisodeMetadata
→ save metadata json

This script assumes reset_templates_v0.json already exists.
Generate it first with:

    PYTHONPATH=. python scripts/generate_reset_templates.py --num-per-split 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.metadata_schema import EpisodeMetadata
from src.interventions.reset_template_loader import (
    get_template_by_id,
    get_templates_by_split,
    load_reset_templates,
    summarize_templates,
    template_to_episode_metadata,
)


EXPECTED_SPLITS = [
    "train_sim_id",
    "val_sim_id",
    "test_sim_id",
    "test_sim_layout_ood_blocking",
    "test_sim_layout_ood_narrow_passage",
    "test_sim_layout_ood_edge_goal",
    "test_sim_shape_ood_L",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default="data/sim/metadata/reset_templates_v0.json",
        help="Path to reset template JSON file.",
    )

    parser.add_argument(
        "--save-metadata-path",
        type=str,
        default="runs/debug/fake_episode_from_reset_template.json",
        help="Path to save fake EpisodeMetadata converted from one reset template.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    template_path = Path(args.path)
    if not template_path.exists():
        raise FileNotFoundError(
            f"Reset template file does not exist: {template_path}\n"
            "Generate it first with:\n"
            "  PYTHONPATH=. python scripts/generate_reset_templates.py --num-per-split 3"
        )

    templates = load_reset_templates(template_path)
    summary = summarize_templates(templates)

    print("num_templates:", summary["num_templates"])
    print("by_split:")
    print(json.dumps(summary["by_split"], indent=2, ensure_ascii=False))
    print("by_layout_family:")
    print(json.dumps(summary["by_layout_family"], indent=2, ensure_ascii=False))
    print("by_shape_family:")
    print(json.dumps(summary["by_shape_family"], indent=2, ensure_ascii=False))

    for split in EXPECTED_SPLITS:
        selected = get_templates_by_split(templates, split)
        assert len(selected) > 0, f"No templates found for split={split}"

    train_templates = get_templates_by_split(templates, "train_sim_id")
    first_template = train_templates[0]

    reset_template_id = first_template["reset_template_id"]

    same_template = get_template_by_id(templates, reset_template_id)
    assert same_template["reset_template_id"] == reset_template_id

    metadata = template_to_episode_metadata(first_template)
    metadata.validate()

    save_path = Path(args.save_metadata_path)
    metadata.save_json(save_path)

    loaded = EpisodeMetadata.load_json(save_path)
    loaded.validate()

    assert loaded.episode_id == f"episode__{reset_template_id}"
    assert loaded.schema_version == "v0.1"
    assert loaded.reset_template_id == reset_template_id
    assert loaded.seed == first_template["seed"]

    assert loaded.split == first_template["split"]
    assert loaded.layout_family == first_template["layout_family"]
    assert loaded.shape_family == first_template["shape_family"]
    assert loaded.object_shape == first_template["object_shape"]

    assert loaded.object_initial_pose.x == first_template["object_initial_pose"]["x"]
    assert loaded.object_initial_pose.y == first_template["object_initial_pose"]["y"]
    assert loaded.object_initial_pose.theta == first_template["object_initial_pose"]["theta"]

    assert loaded.goal_pose.x == first_template["goal_pose"]["x"]
    assert loaded.goal_pose.y == first_template["goal_pose"]["y"]
    assert loaded.goal_pose.theta == first_template["goal_pose"]["theta"]

    assert loaded.ee_initial_pose.x == first_template["ee_initial_pose"]["x"]
    assert loaded.ee_initial_pose.y == first_template["ee_initial_pose"]["y"]
    assert loaded.ee_initial_pose.theta == first_template["ee_initial_pose"]["theta"]

    assert len(loaded.obstacles) == len(first_template["obstacles"])

    print("sample reset_template_id:", reset_template_id)
    print("converted episode_id:", loaded.episode_id)
    print("converted metadata reset_template_id:", loaded.reset_template_id)
    print("converted metadata seed:", loaded.seed)
    print("converted metadata save path:", save_path)
    print("reset template debug ok")


if __name__ == "__main__":
    main()