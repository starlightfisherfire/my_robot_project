from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.planners.cost_functions import wrap_angle


@dataclass
class ToyPushState:
    """
    Minimal state for a toy planar pushing environment.

    This is not MuJoCo.
    It is only a lightweight deterministic environment for debugging
    oracle rollout and CEM-MPC interfaces.
    """

    object_pose: np.ndarray  # [3] = x, y, theta
    goal_pose: np.ndarray    # [3] = x, y, theta
    ee_pos: np.ndarray       # [2] = x, y
    step_count: int = 0
    last_contact: bool = False
    last_collision: bool = False


class ToyPushEnv:
    """
    Toy planar pushing environment.

    Purpose:
        - debug oracle rollout
        - debug CEM-MPC integration
        - test state clone / restore behavior

    It is intentionally simple:
        - action is normalized planar EE velocity [vx, vy] in [-1, 1]
        - if EE is close to object, object moves with EE
        - workspace boundary is enforced by clipping
        - no real contact physics, no MuJoCo

    Important:
        The default max_speed_mps=0.50 is intentionally larger than the
        planned SO-101 / real-control value, e.g. around 0.05 m/s.
        This toy environment uses a larger speed only to make CEM debugging
        easier within a short planning horizon.

        Do not treat ToyPushEnv speed settings as the final real-robot
        control configuration.
    """

    def __init__(
        self,
        control_dt: float = 0.1,
        max_speed_mps: float = 0.50,
        contact_distance: float = 0.075,
        object_coupling: float = 0.85,
        rotation_gain: float = 0.10,
        workspace_bounds: dict[str, float] | None = None,
    ):
        if control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")

        if max_speed_mps <= 0:
            raise ValueError(f"max_speed_mps must be positive, got {max_speed_mps}")

        if contact_distance <= 0:
            raise ValueError(
                f"contact_distance must be positive, got {contact_distance}"
            )

        if not (0.0 < object_coupling <= 1.0):
            raise ValueError(
                f"object_coupling must be in (0, 1], got {object_coupling}"
            )

        self.control_dt = float(control_dt)
        self.max_speed_mps = float(max_speed_mps)
        self.contact_distance = float(contact_distance)
        self.object_coupling = float(object_coupling)
        self.rotation_gain = float(rotation_gain)

        if workspace_bounds is None:
            workspace_bounds = {
                "x_min": 0.0,
                "x_max": 0.70,
                "y_min": 0.0,
                "y_max": 0.50,
            }

        self.workspace_bounds = workspace_bounds
        self.state: ToyPushState | None = None

        self.reset()

    def reset(
        self,
        object_pose: np.ndarray | list[float] | None = None,
        goal_pose: np.ndarray | list[float] | None = None,
        ee_pos: np.ndarray | list[float] | None = None,
    ) -> ToyPushState:
        """
        Reset toy environment.

        Default task:
            EE starts left of the object.
            Goal is to the right of the object.
        """
        if object_pose is None:
            object_pose = [0.20, 0.18, 0.0]

        if goal_pose is None:
            goal_pose = [0.42, 0.18, 0.0]

        if ee_pos is None:
            ee_pos = [0.10, 0.18]

        self.state = ToyPushState(
            object_pose=np.asarray(object_pose, dtype=np.float64).copy(),
            goal_pose=np.asarray(goal_pose, dtype=np.float64).copy(),
            ee_pos=np.asarray(ee_pos, dtype=np.float64).copy(),
            step_count=0,
            last_contact=False,
            last_collision=False,
        )

        self._validate_state(self.state)
        return self.clone_state()

    def reset_from_template(self, template: dict) -> ToyPushState:
        """
        Reset from one reset template.

        Uses:
            object_initial_pose
            goal_pose
            ee_initial_pose

        Only x/y/theta fields are used.
        """
        object_pose = [
            template["object_initial_pose"]["x"],
            template["object_initial_pose"]["y"],
            template["object_initial_pose"].get("theta", 0.0),
        ]

        goal_pose = [
            template["goal_pose"]["x"],
            template["goal_pose"]["y"],
            template["goal_pose"].get("theta", 0.0),
        ]

        ee_pos = [
            template["ee_initial_pose"]["x"],
            template["ee_initial_pose"]["y"],
        ]

        return self.reset(
            object_pose=object_pose,
            goal_pose=goal_pose,
            ee_pos=ee_pos,
        )

    def clone_state(self) -> ToyPushState:
        if self.state is None:
            raise RuntimeError("ToyPushEnv has not been reset.")

        return ToyPushState(
            object_pose=self.state.object_pose.copy(),
            goal_pose=self.state.goal_pose.copy(),
            ee_pos=self.state.ee_pos.copy(),
            step_count=int(self.state.step_count),
            last_contact=bool(self.state.last_contact),
            last_collision=bool(self.state.last_collision),
        )

    def restore_state(self, state: ToyPushState) -> None:
        self._validate_state(state)
        self.state = ToyPushState(
            object_pose=state.object_pose.copy(),
            goal_pose=state.goal_pose.copy(),
            ee_pos=state.ee_pos.copy(),
            step_count=int(state.step_count),
            last_contact=bool(state.last_contact),
            last_collision=bool(state.last_collision),
        )

    def step(self, action: np.ndarray | list[float]) -> ToyPushState:
        """
        Apply one normalized planar velocity action.

        action:
            [2], clipped to [-1, 1]
        """
        if self.state is None:
            raise RuntimeError("ToyPushEnv has not been reset.")

        action = np.asarray(action, dtype=np.float64)

        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        if not np.isfinite(action).all():
            raise ValueError(f"Action contains non-finite values: {action}")

        action = np.clip(action, -1.0, 1.0)

        old_ee = self.state.ee_pos.copy()
        old_object_xy = self.state.object_pose[:2].copy()

        ee_delta = action * self.max_speed_mps * self.control_dt
        new_ee = old_ee + ee_delta

        new_ee, ee_boundary_collision = self._clip_point_to_workspace(new_ee)

        dist_before = float(np.linalg.norm(old_ee - old_object_xy))
        dist_after = float(np.linalg.norm(new_ee - old_object_xy))

        has_contact = min(dist_before, dist_after) <= self.contact_distance

        object_collision = False

        if has_contact:
            object_delta = self.object_coupling * ee_delta
            new_object_xy = self.state.object_pose[:2] + object_delta
            new_object_xy, object_collision = self._clip_point_to_workspace(
                new_object_xy
            )

            self.state.object_pose[:2] = new_object_xy
            self.state.object_pose[2] = wrap_angle(
                self.state.object_pose[2] + self.rotation_gain * float(ee_delta[1])
            )

        self.state.ee_pos = new_ee
        self.state.step_count += 1
        self.state.last_contact = bool(has_contact)
        self.state.last_collision = bool(ee_boundary_collision or object_collision)

        self._validate_state(self.state)
        return self.clone_state()

    def _clip_point_to_workspace(self, point: np.ndarray) -> tuple[np.ndarray, bool]:
        clipped = point.copy()

        x_before = float(clipped[0])
        y_before = float(clipped[1])

        clipped[0] = np.clip(
            clipped[0],
            self.workspace_bounds["x_min"],
            self.workspace_bounds["x_max"],
        )
        clipped[1] = np.clip(
            clipped[1],
            self.workspace_bounds["y_min"],
            self.workspace_bounds["y_max"],
        )

        collided = bool(
            not np.isclose(x_before, clipped[0])
            or not np.isclose(y_before, clipped[1])
        )

        return clipped, collided

    def _validate_state(self, state: ToyPushState) -> None:
        if state.object_pose.shape != (3,):
            raise ValueError(
                f"object_pose must have shape (3,), got {state.object_pose.shape}"
            )

        if state.goal_pose.shape != (3,):
            raise ValueError(
                f"goal_pose must have shape (3,), got {state.goal_pose.shape}"
            )

        if state.ee_pos.shape != (2,):
            raise ValueError(f"ee_pos must have shape (2,), got {state.ee_pos.shape}")

        if not (
            np.isfinite(state.object_pose).all()
            and np.isfinite(state.goal_pose).all()
            and np.isfinite(state.ee_pos).all()
        ):
            raise ValueError("ToyPushState contains non-finite values.")