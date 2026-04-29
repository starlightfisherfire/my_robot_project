from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


Domain = Literal["sim", "real"]
Split = str
LayoutFamily = str
ShapeFamily = str


@dataclass
class Pose2D:
    """
    2D pose used for object, goal, end-effector, and obstacles.

    theta is in radians.
    """

    x: float
    y: float
    theta: float = 0.0


@dataclass
class Velocity2D:
    """
    2D planar velocity.

    omega is angular velocity in rad/s.

    v0.1 note:
        This class is currently optional. It is kept for future metadata
        fields such as object_initial_velocity or final_object_velocity.
    """

    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


@dataclass
class ObstacleMetadata:
    """
    Metadata for one obstacle in the pushing scene.
    """

    obstacle_id: str
    pose: Pose2D
    size_x: float
    size_y: float
    shape: str = "box"
    valid: bool = True


@dataclass
class EpisodeMetadata:
    """
    Episode-level metadata for Paper 1.

    This schema is intentionally simple and JSON-serializable.

    It records:
        - which split the episode belongs to;
        - whether it is sim or real;
        - layout / shape intervention family;
        - reset template identity;
        - initial and goal poses;
        - obstacles;
        - labels used for metrics, probes, and failure analysis.

    It does not store the full trajectory tensor.
    Full states/actions/videos should be saved separately.
    """

    episode_id: str

    # Dataset organization
    domain: Domain
    split: Split
    layout_family: LayoutFamily
    shape_family: ShapeFamily

    # Core scene information
    object_shape: str
    object_initial_pose: Pose2D
    goal_pose: Pose2D
    ee_initial_pose: Pose2D

    # Object metadata
    object_size_x: float
    object_size_y: float

    # Schema / provenance
    schema_version: str = "v0.1"
    reset_template_id: str | None = None
    seed: int | None = None

    # Optional physical metadata
    object_mass: float | None = None
    object_friction: float | None = None

    # Obstacles
    obstacles: list[ObstacleMetadata] = field(default_factory=list)

    # Episode info
    num_steps: int = 0
    control_dt: float = 0.1

    # Labels / metrics
    success: bool | None = None
    final_object_pose: Pose2D | None = None
    final_pos_error: float | None = None
    final_theta_error: float | None = None
    failure_code: str | None = None

    # Optional supervised labels / representation-probe labels
    future_delta_pose: Pose2D | None = None
    contact_side: str | None = None
    push_direction_bin: str | None = None
    rotation_required: bool | None = None

    # Optional data paths relative to project root
    state_path: str | None = None
    action_path: str | None = None
    video_path: str | None = None

    # Notes for debugging / manual annotation
    notes: str | None = None

    def validate(self) -> None:
        if self.domain not in {"sim", "real"}:
            raise ValueError(f"Invalid domain: {self.domain}")

        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")

        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")

        if not self.split:
            raise ValueError("split must be non-empty")

        if not self.layout_family:
            raise ValueError("layout_family must be non-empty")

        if not self.shape_family:
            raise ValueError("shape_family must be non-empty")

        if not self.object_shape:
            raise ValueError("object_shape must be non-empty")

        if self.object_size_x <= 0 or self.object_size_y <= 0:
            raise ValueError(
                f"Object size must be positive, got "
                f"{self.object_size_x}, {self.object_size_y}"
            )

        if self.object_mass is not None and self.object_mass <= 0:
            raise ValueError(f"object_mass must be positive, got {self.object_mass}")

        if self.object_friction is not None and self.object_friction < 0:
            raise ValueError(
                f"object_friction must be non-negative, got {self.object_friction}"
            )

        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        if self.num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {self.num_steps}")

        if self.control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {self.control_dt}")

        if self.final_pos_error is not None and self.final_pos_error < 0:
            raise ValueError(
                f"final_pos_error must be non-negative, got {self.final_pos_error}"
            )

        if self.final_theta_error is not None and self.final_theta_error < 0:
            raise ValueError(
                f"final_theta_error must be non-negative, got {self.final_theta_error}"
            )

        for obs in self.obstacles:
            if not obs.obstacle_id:
                raise ValueError("obstacle_id must be non-empty")

            if obs.size_x <= 0 or obs.size_y <= 0:
                raise ValueError(
                    f"Obstacle {obs.obstacle_id} has invalid size: "
                    f"{obs.size_x}, {obs.size_y}"
                )

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeMetadata":
        object_initial_pose = Pose2D(**data["object_initial_pose"])
        goal_pose = Pose2D(**data["goal_pose"])
        ee_initial_pose = Pose2D(**data["ee_initial_pose"])

        final_object_pose = data.get("final_object_pose")
        if final_object_pose is not None:
            final_object_pose = Pose2D(**final_object_pose)

        future_delta_pose = data.get("future_delta_pose")
        if future_delta_pose is not None:
            future_delta_pose = Pose2D(**future_delta_pose)

        obstacles = []
        for obs in data.get("obstacles", []):
            obstacles.append(
                ObstacleMetadata(
                    obstacle_id=obs["obstacle_id"],
                    pose=Pose2D(**obs["pose"]),
                    size_x=obs["size_x"],
                    size_y=obs["size_y"],
                    shape=obs.get("shape", "box"),
                    valid=obs.get("valid", True),
                )
            )

        obj = cls(
            episode_id=data["episode_id"],
            domain=data["domain"],
            split=data["split"],
            layout_family=data["layout_family"],
            shape_family=data["shape_family"],
            object_shape=data["object_shape"],
            object_initial_pose=object_initial_pose,
            goal_pose=goal_pose,
            ee_initial_pose=ee_initial_pose,
            object_size_x=data["object_size_x"],
            object_size_y=data["object_size_y"],
            schema_version=data.get("schema_version", "v0.1"),
            reset_template_id=data.get("reset_template_id"),
            seed=data.get("seed"),
            object_mass=data.get("object_mass"),
            object_friction=data.get("object_friction"),
            obstacles=obstacles,
            num_steps=data.get("num_steps", 0),
            control_dt=data.get("control_dt", 0.1),
            success=data.get("success"),
            final_object_pose=final_object_pose,
            final_pos_error=data.get("final_pos_error"),
            final_theta_error=data.get("final_theta_error"),
            failure_code=data.get("failure_code"),
            future_delta_pose=future_delta_pose,
            contact_side=data.get("contact_side"),
            push_direction_bin=data.get("push_direction_bin"),
            rotation_required=data.get("rotation_required"),
            state_path=data.get("state_path"),
            action_path=data.get("action_path"),
            video_path=data.get("video_path"),
            notes=data.get("notes"),
        )

        obj.validate()
        return obj

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> "EpisodeMetadata":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)