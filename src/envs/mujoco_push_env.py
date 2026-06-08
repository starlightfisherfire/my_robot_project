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


# ---------------------------------------------------------------------------
# XML template: {obstacle_bodies} is filled at model-build time
# ---------------------------------------------------------------------------

MINIMAL_PUSH_XML = """
<mujoco model="minimal_planar_push">
  <compiler angle="radian" coordinate="local"/>

  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <geom friction="1.0 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>

  <worldbody>
    <light name="top_light" pos="0.35 0.25 1.0" dir="0 0 -1"/>

    <camera name="topdown" pos="0.35 0.25 0.85" xyaxes="1 0 0 0 1 0" fovy="45"/>

    <geom
      name="floor"
      type="plane"
      pos="0.35 0.25 0.0"
      size="0.80 0.60 0.02"
      rgba="0.85 0.85 0.85 1"
      contype="1"
      conaffinity="1"
    />

    <body name="pusher" pos="0 0 {pusher_z}">
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
{pusher_geom}
    </body>

    <body name="object" pos="0.20 0.18 0.006">
      <freejoint name="object_free"/>
{object_geoms}
    </body>

    <body name="goal_shape" pos="0.42 0.18 0.006">
{goal_geoms}
    </body>

{obstacle_bodies}
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


# ---------------------------------------------------------------------------
# Obstacle XML builder
# ---------------------------------------------------------------------------

MAX_OBSTACLES = 3
DEFAULT_OBSTACLE_HEIGHT = 0.050


def _build_obstacle_body_xml(
    slot: int,
    x: float,
    y: float,
    theta: float,
    size_x: float,
    size_y: float,
    half_h: float,
    active: bool,
) -> str:
    """Build XML for one obstacle body slot."""
    name = f"obstacle_{slot}"
    geom_name = f"obstacle_geom_{slot}"

    if active:
        qw = float(np.cos(theta / 2.0))
        qz = float(np.sin(theta / 2.0))
        return f"""    <body name="{name}" pos="{x:.6f} {y:.6f} {half_h:.6f}" quat="{qw:.6f} 0 0 {qz:.6f}">
      <geom
        name="{geom_name}"
        type="box"
        size="{size_x / 2.0:.6f} {size_y / 2.0:.6f} {half_h:.6f}"
        rgba="0.45 0.45 0.45 1"
        contype="1"
        conaffinity="1"
      />
    </body>"""
    else:
        return f"""    <body name="{name}" pos="0.35 0.25 -0.10">
      <geom
        name="{geom_name}"
        type="box"
        size="0.02 0.02 0.006"
        rgba="0.45 0.45 0.45 0"
        contype="0"
        conaffinity="0"
      />
    </body>"""


def _build_obstacle_bodies_xml(
    obstacles: list[dict] | None,
    max_obstacles: int = MAX_OBSTACLES,
    obstacle_height: float = DEFAULT_OBSTACLE_HEIGHT,
) -> str:
    """Build XML for all obstacle body slots.

    Active slots get real position/size/collision from template.
    Inactive slots are underground, invisible, no collision.
    """
    if obstacles is None:
        obstacles = []

    half_h = obstacle_height / 2.0
    parts: list[str] = []

    for i in range(max_obstacles):
        if i < len(obstacles):
            obs = obstacles[i]
            pose = obs["pose"]
            parts.append(_build_obstacle_body_xml(
                slot=i,
                x=float(pose["x"]),
                y=float(pose["y"]),
                theta=float(pose.get("theta", 0.0)),
                size_x=float(obs["size_x"]),
                size_y=float(obs["size_y"]),
                half_h=half_h,
                active=True,
            ))
        else:
            parts.append(_build_obstacle_body_xml(
                slot=i, x=0, y=0, theta=0, size_x=0, size_y=0,
                half_h=0, active=False,
            ))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# XML builder with shape + obstacles
# ---------------------------------------------------------------------------

def _build_pusher_geom_xml(
    pusher_radius: float,
    pusher_halfheight: float,
    pusher_mass: float,
) -> str:
    return f"""      <geom
        name="pusher_geom"
        type="cylinder"
        size="{pusher_radius:.6f} {pusher_halfheight:.6f}"
        mass="{pusher_mass:.6f}"
        rgba="0.1 0.2 0.9 1"
        contype="1"
        conaffinity="1"
      />"""


def _build_xml_with_shape(
    shape_type: str,
    use_simple_box: bool = False,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
    pusher_mass: float = 0.05,
    obstacles: list[dict] | None = None,
    obstacle_height: float = DEFAULT_OBSTACLE_HEIGHT,
) -> str:
    """Build complete MuJoCo XML with object geometry and compile-time obstacles."""
    if use_simple_box:
        object_geoms = """      <geom
        name="object_geom"
        type="box"
        size="0.024 0.024 0.006"
        mass="0.01905"
        rgba="0.9 0.2 0.1 1"
        contype="1"
        conaffinity="1"
      />"""
        goal_geoms = """      <geom
        name="goal_geom"
        type="box"
        size="0.024 0.024 0.006"
        rgba="0.1 0.8 0.1 0.25"
        contype="0"
        conaffinity="0"
      />"""
    else:
        factory = ObjectShapeFactory()
        object_geoms = factory.get_object_geoms_xml(
            shape_type=shape_type,
            rgba="0.9 0.2 0.1 1",
            contype=1,
            conaffinity=1,
        )
        goal_geoms = factory.get_goal_ghost_geoms_xml(
            shape_type=shape_type,
            rgba="0.1 0.8 0.1 0.25",
            name_prefix="goal_geom",
        )

    pusher_geom = _build_pusher_geom_xml(
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_mass=pusher_mass,
    )

    obstacle_bodies = _build_obstacle_bodies_xml(
        obstacles=obstacles,
        max_obstacles=MAX_OBSTACLES,
        obstacle_height=obstacle_height,
    )

    return MINIMAL_PUSH_XML.format(
        pusher_z=pusher_z,
        pusher_geom=pusher_geom,
        object_geoms=object_geoms,
        goal_geoms=goal_geoms,
        obstacle_bodies=obstacle_bodies,
    )


# ---------------------------------------------------------------------------
# Obstacle signature helper
# ---------------------------------------------------------------------------

def _obstacle_signature_from_template(template: dict) -> tuple:
    """Compute a stable signature for template obstacles.

    Includes x/y/theta/size_x/size_y/valid and obstacle_height.
    Floats are rounded to 1e-6 for stability.
    """
    obstacles = template.get("obstacles", [])
    sig_parts: list = []
    for obs in obstacles:
        pose = obs["pose"]
        sig_parts.append((
            round(float(pose["x"]), 6),
            round(float(pose["y"]), 6),
            round(float(pose.get("theta", 0.0)), 6),
            round(float(obs["size_x"]), 6),
            round(float(obs["size_y"]), 6),
            bool(obs.get("valid", True)),
        ))
    return (tuple(sig_parts), round(DEFAULT_OBSTACLE_HEIGHT, 6))


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

@dataclass
class MujocoPushState:
    """Snapshot of the minimal MuJoCo pushing environment."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    goal_pose: np.ndarray
    step_count: int
    last_contact: bool
    last_collision: bool


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MujocoPushEnv:
    """
    Minimal MuJoCo planar pushing environment for Paper 1 scaffolding.

    v0.3 notes:
        - Obstacles are compiled into the MuJoCo XML at model creation time.
        - reset_from_template() rebuilds the model when obstacle config changes.
        - last_collision detects object-obstacle AND pusher-obstacle collisions.
        - obstacle_height = 0.050m (wall-style barrier, prevents z-escape).
    """

    def __init__(
        self,
        control_dt: float = 0.1,
        max_speed_mps: float = 0.05,
        shape_type: str = "T",
        use_simple_box_object: bool = False,
        pusher_radius: float = 0.010,
        pusher_halfheight: float = 0.014,
        pusher_z: float = 0.016,
        pusher_mass: float = 0.05,
        xml: str | None = None,
    ):
        if control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")
        if max_speed_mps <= 0:
            raise ValueError(f"max_speed_mps must be positive, got {max_speed_mps}")

        self.control_dt = float(control_dt)
        self.max_speed_mps = float(max_speed_mps)
        self.shape_type = shape_type
        self.use_simple_box_object = use_simple_box_object
        self.pusher_radius = float(pusher_radius)
        self.pusher_halfheight = float(pusher_halfheight)
        self.pusher_z = float(pusher_z)
        self.pusher_mass = float(pusher_mass)
        self.obstacle_height = DEFAULT_OBSTACLE_HEIGHT
        self.max_obstacles = MAX_OBSTACLES
        self._current_obstacle_signature: tuple | None = None

        if xml is None:
            xml = _build_xml_with_shape(
                shape_type=shape_type,
                use_simple_box=use_simple_box_object,
                pusher_radius=pusher_radius,
                pusher_halfheight=pusher_halfheight,
                pusher_z=pusher_z,
                pusher_mass=pusher_mass,
            )

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self._cache_model_handles()

        self.goal_pose = np.array([0.42, 0.18, 0.0], dtype=np.float64)
        self.step_count = 0
        self.last_contact = False
        self.last_collision = False

        self.reset()

    # ------------------------------------------------------------------
    # Model handle caching
    # ------------------------------------------------------------------

    def _cache_model_handles(self) -> None:
        """Cache body/joint/geom IDs and qpos/qvel addresses from self.model.

        Must be called after every model rebuild.
        """
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object",
        )
        self.pusher_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher",
        )
        self.object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free",
        )
        self.pusher_x_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_x",
        )
        self.pusher_y_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_y",
        )

        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_adr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.pusher_x_qpos_adr = int(self.model.jnt_qposadr[self.pusher_x_joint_id])
        self.pusher_y_qpos_adr = int(self.model.jnt_qposadr[self.pusher_y_joint_id])
        self.pusher_x_qvel_adr = int(self.model.jnt_dofadr[self.pusher_x_joint_id])
        self.pusher_y_qvel_adr = int(self.model.jnt_dofadr[self.pusher_y_joint_id])

        self.goal_shape_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_shape",
        )

        self._obstacle_body_ids: list[int] = []
        self._obstacle_geom_ids: list[int] = []
        for i in range(self.max_obstacles):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}",
            )
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom_{i}",
            )
            self._obstacle_body_ids.append(body_id)
            self._obstacle_geom_ids.append(geom_id)

    # ------------------------------------------------------------------
    # Model rebuild (compile-time obstacles)
    # ------------------------------------------------------------------

    def _rebuild_model_with_obstacles(self, obstacles: list[dict] | None) -> None:
        """Rebuild the MuJoCo model with obstacles baked into the XML.

        This is the ONLY reliable way to change collision geometry in MuJoCo.
        Runtime modification of geom_contype/conaffinity/geom_size does NOT
        update the collision broadphase.
        """
        self._num_active_obstacles = len(obstacles) if obstacles else 0
        xml = _build_xml_with_shape(
            shape_type=self.shape_type,
            use_simple_box=self.use_simple_box_object,
            pusher_radius=self.pusher_radius,
            pusher_halfheight=self.pusher_halfheight,
            pusher_z=self.pusher_z,
            pusher_mass=self.pusher_mass,
            obstacles=obstacles,
            obstacle_height=self.obstacle_height,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self._cache_model_handles()

    # ------------------------------------------------------------------
    # Goal visuals
    # ------------------------------------------------------------------

    def _sync_goal_visuals(self) -> None:
        """Sync goal visualization (goal_shape) with goal_pose."""
        if self.goal_shape_body_id >= 0:
            theta = float(self.goal_pose[2])
            qw = float(np.cos(theta / 2.0))
            qz = float(np.sin(theta / 2.0))

            self.model.body_pos[self.goal_shape_body_id] = np.array(
                [self.goal_pose[0], self.goal_pose[1], 0.006],
                dtype=np.float64,
            )
            self.model.body_quat[self.goal_shape_body_id] = np.array(
                [qw, 0.0, 0.0, qz],
                dtype=np.float64,
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

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
            raise ValueError(f"object_pose must have shape (3,), got {object_pose.shape}")
        if goal_pose.shape != (3,):
            raise ValueError(f"goal_pose must have shape (3,), got {goal_pose.shape}")
        if ee_pos.shape != (2,):
            raise ValueError(f"ee_pos must have shape (2,), got {ee_pos.shape}")
        if not (np.isfinite(object_pose).all() and np.isfinite(goal_pose).all()
                and np.isfinite(ee_pos).all()):
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

        # Object free joint qpos: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = np.array(
            [object_pose[0], object_pose[1], z, qw, 0.0, 0.0, qz],
            dtype=np.float64,
        )
        self.data.qvel[self.object_qvel_adr : self.object_qvel_adr + 6] = 0.0

        # Pusher slide joints
        self.data.qpos[self.pusher_x_qpos_adr] = float(ee_pos[0])
        self.data.qpos[self.pusher_y_qpos_adr] = float(ee_pos[1])
        self.data.qvel[self.pusher_x_qvel_adr] = 0.0
        self.data.qvel[self.pusher_y_qvel_adr] = 0.0

        self.data.ctrl[:] = 0.0

        self._sync_goal_visuals()

        mujoco.mj_forward(self.model, self.data)
        self._update_contact_flags()

        return self.clone_state()

    def reset_from_template(self, template: dict) -> MujocoPushState:
        """Reset from one reset template.

        If the template's obstacle configuration differs from the current model,
        the MuJoCo model is rebuilt with compile-time obstacles.
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

        # Check if obstacle config changed -> rebuild model
        obstacles = template.get("obstacles", [])
        sig = _obstacle_signature_from_template(template)
        if sig != self._current_obstacle_signature:
            self._rebuild_model_with_obstacles(obstacles if obstacles else None)
            self._current_obstacle_signature = sig

        state = self.reset(
            object_pose=object_pose,
            goal_pose=goal_pose,
            ee_pos=ee_pos,
        )

        return state

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

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

        self._sync_goal_visuals()

        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray | list[float]) -> MujocoPushState:
        """Apply one normalized planar velocity action.

        action: Shape (2,), clipped to [-1, 1].
        The internal actuator command is action * max_speed_mps.
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

    # ------------------------------------------------------------------
    # State getters
    # ------------------------------------------------------------------

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
        """Returns 1.0 if object-obstacle OR pusher-obstacle contact detected."""
        return float(self.last_collision)

    # ------------------------------------------------------------------
    # Contact detection
    # ------------------------------------------------------------------

    def _update_contact_flags(self) -> None:
        contact = False
        collision = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]

            geom1_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1,
            )
            geom2_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2,
            )

            names = {geom1_name, geom2_name}

            has_pusher = "pusher_geom" in names
            has_object = any("object_geom" in name for name in names if name)
            has_obstacle = any("obstacle_geom" in name for name in names if name)

            if has_pusher and has_object:
                contact = True
            if has_object and has_obstacle:
                collision = True
            if has_pusher and has_obstacle:
                collision = True

        self.last_contact = bool(contact)
        self.last_collision = bool(collision)

    # ------------------------------------------------------------------
    # Deprecated helpers (no longer used in main path)
    # ------------------------------------------------------------------

    def _disable_all_obstacles(self) -> None:
        """DEPRECATED: Runtime obstacle modification does not work in MuJoCo.
        Kept for backwards compatibility only. Do not use in new code."""
        pass

    def _set_obstacle_slot(self, i: int, obs: dict) -> None:
        """DEPRECATED: Runtime obstacle modification does not work in MuJoCo.
        Kept for backwards compatibility only. Do not use in new code."""
        pass

    def _apply_obstacles_from_template(self, template: dict) -> None:
        """DEPRECATED: Runtime obstacle modification does not work in MuJoCo.
        Kept for backwards compatibility only. Do not use in new code."""
        pass

    def _refresh_model_constants(self) -> None:
        """DEPRECATED: Runtime obstacle modification does not work in MuJoCo.
        Kept for backwards compatibility only. Do not use in new code."""
        pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        """Convert MuJoCo quaternion [w, x, y, z] to planar yaw."""
        w, x, y, z = [float(v) for v in quat]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        return float(wrap_angle(np.arctan2(siny_cosp, cosy_cosp)))
