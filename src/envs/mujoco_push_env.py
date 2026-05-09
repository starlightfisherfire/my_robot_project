from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mujoco is not installed or not available in this environment. "
        "Install/activate MuJoCo before using MujocoPushEnv."
    ) from exc

from src.planners.cost_functions import wrap_angle
from src.envs.object_shape_factory import ObjectShapeFactory


MINIMAL_PUSH_XML = """
<mujoco model="minimal_planar_push">
  <compiler angle="radian" coordinate="local"/>

  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>

  <default>
    <geom friction="1.0 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>

  <worldbody>
    <light name="top_light" pos="0.35 0.25 1.0" dir="0 0 -1"/>

    <geom
      name="floor"
      type="plane"
      pos="0.35 0.25 0.0"
      size="0.80 0.60 0.02"
      rgba="0.85 0.85 0.85 1"
      contype="1"
      conaffinity="1"
    />

    <body name="pusher" pos="0 0 0.035">
      <joint
        name="pusher_x"
        type="slide"
        axis="1 0 0"
        limited="true"
        range="0.0 0.70"
        damping="4.0"
      />
      <joint
        name="pusher_y"
        type="slide"
        axis="0 1 0"
        limited="true"
        range="0.0 0.50"
        damping="4.0"
      />
      <geom
        name="pusher_geom"
        type="sphere"
        size="0.025"
        mass="0.05"
        rgba="0.1 0.2 0.9 1"
        contype="1"
        conaffinity="1"
      />
    </body>

    <body name="object" pos="0.20 0.18 0.006">
      <freejoint name="object_free"/>
{object_geoms}
    </body>

    <site
      name="goal_site"
      pos="0.42 0.18 0.005"
      size="0.02"
      rgba="0.1 0.8 0.1 0.5"
    />
  </worldbody>

  <actuator>
    <velocity
      name="pusher_x_vel"
      joint="pusher_x"
      kv="10"
      ctrlrange="-0.50 0.50"
    />
    <velocity
      name="pusher_y_vel"
      joint="pusher_y"
      kv="10"
      ctrlrange="-0.50 0.50"
    />
  </actuator>
</mujoco>
"""


def _build_xml_with_shape(shape_type: str, use_simple_box: bool = False) -> str:
    """
    Build MuJoCo XML with object geometry from ObjectShapeFactory.

    Args:
        shape_type: Shape type (T, L, cross, bar, square, cylinder)
        use_simple_box: If True, use simple box instead of compound geoms

    Returns:
        Complete MuJoCo XML string
    """
    if use_simple_box:
        # Legacy single box object
        object_geoms = """      <geom
        name="object_geom"
        type="box"
        size="0.024 0.024 0.006"
        mass="0.01905"
        rgba="0.9 0.2 0.1 1"
        contype="1"
        conaffinity="1"
      />"""
    else:
        # Use ObjectShapeFactory for compound geoms
        factory = ObjectShapeFactory()
        object_geoms = factory.get_object_geoms_xml(
            shape_type=shape_type,
            rgba="0.9 0.2 0.1 1",
            contype=1,
            conaffinity=1,
        )

    return MINIMAL_PUSH_XML.format(object_geoms=object_geoms)


@dataclass
class MujocoPushState:
    """
    Snapshot of the minimal MuJoCo pushing environment.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    goal_pose: np.ndarray
    step_count: int
    last_contact: bool
    last_collision: bool


class MujocoPushEnv:
    """
    Minimal MuJoCo planar pushing environment for Paper 1 scaffolding.

    Current purpose:
        - verify MuJoCo import / XML loading
        - verify reset_from_template
        - verify step(action)
        - verify clone_state / restore_state
        - verify object pose / EE pose / contact extraction

    This is not yet the final SO-101-realistic environment.
    It is the first MuJoCo scaffold for Oracle-MPC capacity checks.

    v0.1 notes:
        - Obstacles from reset templates are not yet instantiated in MuJoCo.
        - goal_site is currently a visual XML site and is not dynamically updated.
        - last_collision is a placeholder and currently remains False.
    """

    def __init__(
        self,
        control_dt: float = 0.1,
        max_speed_mps: float = 0.05,
        shape_type: str = "T",
        use_simple_box_object: bool = False,
        xml: str | None = None,
    ):
        """
        Initialize MuJoCo pushing environment.

        Args:
            control_dt: Control timestep in seconds
            max_speed_mps: Maximum pusher speed in m/s
            shape_type: Object shape type (T, L, cross, bar, square, cylinder)
                TODO: This should be read from reset_template["object_shape"]
                      in reset_from_template(), but for now we use a default.
            use_simple_box_object: If True, use legacy single box object.
                                   If False, use ObjectShapeFactory compound geoms.
            xml: Optional custom XML string. If None, generates XML based on
                 shape_type and use_simple_box_object.
        """
        if control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")

        if max_speed_mps <= 0:
            raise ValueError(f"max_speed_mps must be positive, got {max_speed_mps}")

        self.control_dt = float(control_dt)
        self.max_speed_mps = float(max_speed_mps)
        self.shape_type = shape_type
        self.use_simple_box_object = use_simple_box_object

        # Generate XML if not provided
        if xml is None:
            xml = _build_xml_with_shape(shape_type, use_simple_box_object)

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.object_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "object",
        )
        self.pusher_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pusher",
        )

        self.object_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_free",
        )
        self.pusher_x_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "pusher_x",
        )
        self.pusher_y_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "pusher_y",
        )

        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_adr = int(self.model.jnt_dofadr[self.object_joint_id])

        self.pusher_x_qpos_adr = int(self.model.jnt_qposadr[self.pusher_x_joint_id])
        self.pusher_y_qpos_adr = int(self.model.jnt_qposadr[self.pusher_y_joint_id])

        self.pusher_x_qvel_adr = int(self.model.jnt_dofadr[self.pusher_x_joint_id])
        self.pusher_y_qvel_adr = int(self.model.jnt_dofadr[self.pusher_y_joint_id])

        self.goal_pose = np.array([0.42, 0.18, 0.0], dtype=np.float64)
        self.step_count = 0
        self.last_contact = False
        self.last_collision = False

        self.reset()

    def reset(
        self,
        object_pose: np.ndarray | list[float] | None = None,
        goal_pose: np.ndarray | list[float] | None = None,
        ee_pos: np.ndarray | list[float] | None = None,
    ) -> MujocoPushState:
        if object_pose is None:
            object_pose = [0.20, 0.18, 0.0]

        if goal_pose is None:
            goal_pose = [0.42, 0.18, 0.0]

        if ee_pos is None:
            ee_pos = [0.10, 0.18]

        object_pose = np.asarray(object_pose, dtype=np.float64)
        goal_pose = np.asarray(goal_pose, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)

        if object_pose.shape != (3,):
            raise ValueError(
                f"object_pose must have shape (3,), got {object_pose.shape}"
            )

        if goal_pose.shape != (3,):
            raise ValueError(
                f"goal_pose must have shape (3,), got {goal_pose.shape}"
            )

        if ee_pos.shape != (2,):
            raise ValueError(
                f"ee_pos must have shape (2,), got {ee_pos.shape}"
            )

        if not (
            np.isfinite(object_pose).all()
            and np.isfinite(goal_pose).all()
            and np.isfinite(ee_pos).all()
        ):
            raise ValueError(
                "reset received non-finite pose values: "
                f"object_pose={object_pose}, goal_pose={goal_pose}, ee_pos={ee_pos}"
            )

        mujoco.mj_resetData(self.model, self.data)

        self.goal_pose = goal_pose.copy()
        self.step_count = 0
        self.last_contact = False
        self.last_collision = False

        z = 0.006
        theta = float(object_pose[2])
        qw = float(np.cos(theta / 2.0))
        qz = float(np.sin(theta / 2.0))

        # Object free joint qpos:
        # [x, y, z, qw, qx, qy, qz]
        self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = np.array(
            [object_pose[0], object_pose[1], z, qw, 0.0, 0.0, qz],
            dtype=np.float64,
        )
        self.data.qvel[self.object_qvel_adr : self.object_qvel_adr + 6] = 0.0

        # Pusher slide joints.
        self.data.qpos[self.pusher_x_qpos_adr] = float(ee_pos[0])
        self.data.qpos[self.pusher_y_qpos_adr] = float(ee_pos[1])
        self.data.qvel[self.pusher_x_qvel_adr] = 0.0
        self.data.qvel[self.pusher_y_qvel_adr] = 0.0

        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._update_contact_flags()

        return self.clone_state()

    def reset_from_template(self, template: dict) -> MujocoPushState:
        """
        Reset from one reset template.

        Current v0.1 uses only:
            object_initial_pose
            goal_pose
            ee_initial_pose

        Obstacles are ignored in this first MuJoCo scaffold.
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

    def clone_state(self) -> MujocoPushState:
        return MujocoPushState(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            goal_pose=self.goal_pose.copy(),
            step_count=int(self.step_count),
            last_contact=bool(self.last_contact),
            last_collision=bool(self.last_collision),
        )

    def restore_state(self, state: MujocoPushState) -> None:
        self.data.qpos[:] = state.qpos.copy()
        self.data.qvel[:] = state.qvel.copy()
        self.data.ctrl[:] = state.ctrl.copy()

        self.goal_pose = state.goal_pose.copy()
        self.step_count = int(state.step_count)
        self.last_contact = bool(state.last_contact)
        self.last_collision = bool(state.last_collision)

        mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray | list[float]) -> MujocoPushState:
        """
        Apply one normalized planar velocity action.

        action:
            Shape (2,), clipped to [-1, 1].

        The internal actuator command is:
            action * max_speed_mps
        """
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        if not np.isfinite(action).all():
            raise ValueError(f"Action contains non-finite values: {action}")

        action = np.clip(action, -1.0, 1.0)
        velocity_cmd = action * self.max_speed_mps

        self.data.ctrl[0] = float(velocity_cmd[0])
        self.data.ctrl[1] = float(velocity_cmd[1])

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._update_contact_flags()

        return self.clone_state()

    def get_object_pose(self) -> np.ndarray:
        xpos = self.data.xpos[self.object_body_id].copy()
        xquat = self.data.xquat[self.object_body_id].copy()

        theta = self._yaw_from_quat_wxyz(xquat)

        return np.array([xpos[0], xpos[1], theta], dtype=np.float64)

    def get_ee_pos(self) -> np.ndarray:
        xpos = self.data.xpos[self.pusher_body_id].copy()

        return np.array([xpos[0], xpos[1]], dtype=np.float64)

    def get_goal_pose(self) -> np.ndarray:
        return self.goal_pose.copy()

    def get_contact_flag(self) -> float:
        return float(self.last_contact)

    def get_collision_flag(self) -> float:
        return float(self.last_collision)

    def _update_contact_flags(self) -> None:
        contact = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]

            geom1_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                con.geom1,
            )
            geom2_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                con.geom2,
            )

            names = {geom1_name, geom2_name}

            # Check if pusher_geom is in contact with any object geom
            # Object may have multiple geoms (e.g., object_geom_top, object_geom_stem)
            has_pusher = "pusher_geom" in names
            has_object = any("object_geom" in name for name in names if name)

            if has_pusher and has_object:
                contact = True
                break

        self.last_contact = bool(contact)

        # v0.1 placeholder:
        # Collision semantics will be defined later for obstacle / boundary checks.
        self.last_collision = False

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        """
        Convert MuJoCo quaternion [w, x, y, z] to planar yaw.
        """
        w, x, y, z = [float(v) for v in quat]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        return float(wrap_angle(np.arctan2(siny_cosp, cosy_cosp)))