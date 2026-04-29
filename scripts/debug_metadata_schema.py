"""
Smoke test for Paper 1 metadata schema.

This script verifies:

fake episode metadata
→ validate
→ save json
→ load json
→ validate again

It also checks that invalid metadata raises errors.
"""

from __future__ import annotations

from pathlib import Path

from src.data.metadata_schema import (
    EpisodeMetadata,
    ObstacleMetadata,
    Pose2D,
)


def make_fake_metadata() -> EpisodeMetadata:
    metadata = EpisodeMetadata(
        episode_id="sim_train_000001",
        domain="sim",
        split="train_sim_id",
        layout_family="open_space",
        shape_family="T_shape",
        object_shape="T",
        object_initial_pose=Pose2D(x=0.20, y=0.10, theta=0.0),
        goal_pose=Pose2D(x=0.45, y=0.25, theta=0.0),
        ee_initial_pose=Pose2D(x=0.10, y=0.10, theta=0.0),
        object_size_x=0.08,
        object_size_y=0.08,
        schema_version="v0.1",
        reset_template_id="reset_train_open_space_000001",
        seed=42,
        object_mass=0.05,
        object_friction=None,
        obstacles=[
            ObstacleMetadata(
                obstacle_id="obs_0",
                pose=Pose2D(x=0.30, y=0.20, theta=0.0),
                size_x=0.05,
                size_y=0.10,
                shape="box",
                valid=True,
            )
        ],
        num_steps=80,
        control_dt=0.1,
        success=False,
        final_object_pose=Pose2D(x=0.35, y=0.18, theta=0.1),
        final_pos_error=0.12,
        final_theta_error=0.1,
        failure_code="timeout",
        future_delta_pose=Pose2D(x=0.03, y=0.02, theta=0.05),
        contact_side="left",
        push_direction_bin="right",
        rotation_required=False,
        state_path="data/sim/episodes/sim_train_000001_states.npy",
        action_path="data/sim/episodes/sim_train_000001_actions.npy",
        video_path="data/sim/videos/sim_train_000001.mp4",
        notes="Fake metadata for schema smoke test.",
    )

    return metadata


def check_save_load_roundtrip() -> None:
    metadata = make_fake_metadata()
    metadata.validate()

    save_path = Path("runs/debug/fake_episode_metadata.json")
    metadata.save_json(save_path)

    loaded = EpisodeMetadata.load_json(save_path)
    loaded.validate()

    assert loaded.episode_id == metadata.episode_id
    assert loaded.schema_version == metadata.schema_version
    assert loaded.domain == metadata.domain
    assert loaded.split == metadata.split
    assert loaded.layout_family == metadata.layout_family
    assert loaded.shape_family == metadata.shape_family
    assert loaded.reset_template_id == metadata.reset_template_id
    assert loaded.seed == metadata.seed
    assert loaded.object_shape == metadata.object_shape

    assert loaded.object_initial_pose.x == metadata.object_initial_pose.x
    assert loaded.object_initial_pose.y == metadata.object_initial_pose.y
    assert loaded.object_initial_pose.theta == metadata.object_initial_pose.theta

    assert loaded.goal_pose.x == metadata.goal_pose.x
    assert loaded.goal_pose.y == metadata.goal_pose.y
    assert loaded.goal_pose.theta == metadata.goal_pose.theta

    assert loaded.ee_initial_pose.x == metadata.ee_initial_pose.x
    assert loaded.ee_initial_pose.y == metadata.ee_initial_pose.y
    assert loaded.ee_initial_pose.theta == metadata.ee_initial_pose.theta

    assert len(loaded.obstacles) == len(metadata.obstacles)

    assert loaded.future_delta_pose is not None
    assert metadata.future_delta_pose is not None
    assert loaded.future_delta_pose.x == metadata.future_delta_pose.x
    assert loaded.future_delta_pose.y == metadata.future_delta_pose.y
    assert loaded.future_delta_pose.theta == metadata.future_delta_pose.theta

    assert loaded.final_object_pose is not None
    assert metadata.final_object_pose is not None
    assert loaded.final_object_pose.x == metadata.final_object_pose.x
    assert loaded.final_object_pose.y == metadata.final_object_pose.y
    assert loaded.final_object_pose.theta == metadata.final_object_pose.theta

    print("episode_id:", loaded.episode_id)
    print("schema_version:", loaded.schema_version)
    print("domain:", loaded.domain)
    print("split:", loaded.split)
    print("layout_family:", loaded.layout_family)
    print("shape_family:", loaded.shape_family)
    print("reset_template_id:", loaded.reset_template_id)
    print("seed:", loaded.seed)
    print("num_obstacles:", len(loaded.obstacles))
    print("future_delta_pose:", loaded.future_delta_pose)
    print("save path:", save_path)


def check_invalid_metadata() -> None:
    # Invalid domain should raise ValueError.
    bad_domain = make_fake_metadata()
    bad_domain.domain = "fake_domain"  # type: ignore[assignment]

    try:
        bad_domain.validate()
        raise AssertionError("Invalid domain did not raise ValueError.")
    except ValueError:
        pass

    # Negative object size should raise ValueError.
    bad_size = make_fake_metadata()
    bad_size.object_size_x = -0.1

    try:
        bad_size.validate()
        raise AssertionError("Negative object size did not raise ValueError.")
    except ValueError:
        pass

    # Empty episode_id should raise ValueError.
    bad_episode_id = make_fake_metadata()
    bad_episode_id.episode_id = ""

    try:
        bad_episode_id.validate()
        raise AssertionError("Empty episode_id did not raise ValueError.")
    except ValueError:
        pass

    # Negative seed should raise ValueError.
    bad_seed = make_fake_metadata()
    bad_seed.seed = -1

    try:
        bad_seed.validate()
        raise AssertionError("Negative seed did not raise ValueError.")
    except ValueError:
        pass

    # Negative final position error should raise ValueError.
    bad_final_pos_error = make_fake_metadata()
    bad_final_pos_error.final_pos_error = -0.01

    try:
        bad_final_pos_error.validate()
        raise AssertionError("Negative final_pos_error did not raise ValueError.")
    except ValueError:
        pass

    # Negative obstacle size should raise ValueError.
    bad_obstacle = make_fake_metadata()
    bad_obstacle.obstacles[0].size_x = -0.05

    try:
        bad_obstacle.validate()
        raise AssertionError("Negative obstacle size did not raise ValueError.")
    except ValueError:
        pass

    print("invalid metadata checks ok")


def main() -> None:
    check_save_load_roundtrip()
    check_invalid_metadata()
    print("metadata schema debug ok")


if __name__ == "__main__":
    main()