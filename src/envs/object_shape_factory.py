from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal

import numpy as np


ShapeType = Literal["T", "L", "cross", "bar", "square", "cylinder"]


class ObjectShapeFactory:
    """
    Generate MuJoCo compound geometries for pushable object shapes.

    Reads configs/object_specs.yaml and converts shape_type specifications
    into actual MuJoCo geom XML strings.

    All shapes (T, L, cross, bar, square, cylinder) are pushable manipulated objects.
    T and L shapes are decomposed into non-overlapping rectangles.
    """

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            repo_root = Path(__file__).parent.parent.parent
            config_path = repo_root / "configs" / "object_specs.yaml"

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Object specs config not found: {self.config_path}"
            )

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.spec_version = list(self.config.keys())[0]
        self.specs = self.config[self.spec_version]

    def get_object_geoms_xml(
        self,
        shape_type: ShapeType,
        rgba: str = "0.9 0.2 0.1 1",
        contype: int = 1,
        conaffinity: int = 1,
    ) -> str:
        """
        Generate MuJoCo geom XML for a given shape type.

        Returns a string containing one or more <geom> tags that form
        the compound geometry for the object.

        Args:
            shape_type: T, L, cross, bar, square, or cylinder
            rgba: color string for MuJoCo
            contype: collision type bitmask
            conaffinity: collision affinity bitmask

        Returns:
            XML string with geom definitions (no enclosing <body> tag)
        """
        if shape_type in ["T", "L", "cross", "bar"]:
            return self._get_manipulated_object_geoms(
                shape_type, rgba, contype, conaffinity
            )
        elif shape_type in ["square", "cylinder"]:
            return self._get_additional_object_geoms(
                shape_type, rgba, contype, conaffinity
            )
        else:
            raise ValueError(
                f"Unknown shape_type: {shape_type}. "
                f"Supported: T, L, cross, bar, square, cylinder"
            )

    def _get_manipulated_object_geoms(
        self,
        shape_type: str,
        rgba: str,
        contype: int,
        conaffinity: int,
    ) -> str:
        """Generate geoms for manipulated objects (T, L, cross, bar)."""
        obj_spec = self.specs["manipulated_objects"][shape_type]

        footprint_mm = obj_spec["footprint_mm"]
        thickness_mm = self.specs["thickness_mm"]
        arm_width_mm = obj_spec.get("arm_width_mm")
        mass_g = obj_spec["nominal_mass_g"]

        footprint_m = [f / 1000.0 for f in footprint_mm]
        thickness_m = thickness_mm / 1000.0
        mass_kg = mass_g / 1000.0

        if shape_type == "bar":
            return self._make_bar_geoms(
                footprint_m, thickness_m, mass_kg, rgba, contype, conaffinity
            )

        if arm_width_mm is None:
            raise ValueError(f"arm_width_mm is required for shape_type={shape_type}")

        arm_width_m = arm_width_mm / 1000.0

        if shape_type == "T":
            return self._make_T_geoms(
                footprint_m, arm_width_m, thickness_m, mass_kg, rgba, contype, conaffinity
            )
        elif shape_type == "L":
            return self._make_L_geoms(
                footprint_m, arm_width_m, thickness_m, mass_kg, rgba, contype, conaffinity
            )
        elif shape_type == "cross":
            return self._make_cross_geoms(
                footprint_m, arm_width_m, thickness_m, mass_kg, rgba, contype, conaffinity
            )
        else:
            raise ValueError(f"Unsupported manipulated object: {shape_type}")

    def _make_T_geoms(
        self,
        footprint_m: list[float],
        arm_width_m: float,
        thickness_m: float,
        mass_kg: float,
        rgba: str,
        contype: int,
        conaffinity: int,
    ) -> str:
        """
        T shape decomposition:
        - Top horizontal bar: width=footprint[0], height=arm_width
        - Vertical stem: width=arm_width, height=(footprint[1] - arm_width)

        Non-overlapping rectangles centered at origin.
        Mass is distributed by area ratio.
        """
        W, H = footprint_m
        w = arm_width_m
        half_thickness = thickness_m / 2.0

        top_bar_size_x = W / 2.0
        top_bar_size_y = w / 2.0
        top_bar_pos_y = (H - w) / 2.0

        stem_size_x = w / 2.0
        stem_size_y = (H - w) / 2.0
        stem_pos_y = -(w / 2.0)

        top_bar_area = W * w
        stem_area = w * (H - w)
        total_area = top_bar_area + stem_area

        mass_top_bar = mass_kg * (top_bar_area / total_area)
        mass_stem = mass_kg * (stem_area / total_area)

        geoms = f"""      <geom
        name="object_geom_top"
        type="box"
        pos="0 {top_bar_pos_y:.6f} 0"
        size="{top_bar_size_x:.6f} {top_bar_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_top_bar:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />
      <geom
        name="object_geom_stem"
        type="box"
        pos="0 {stem_pos_y:.6f} 0"
        size="{stem_size_x:.6f} {stem_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_stem:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />"""

        return geoms

    def _make_L_geoms(
        self,
        footprint_m: list[float],
        arm_width_m: float,
        thickness_m: float,
        mass_kg: float,
        rgba: str,
        contype: int,
        conaffinity: int,
    ) -> str:
        """
        L shape decomposition:
        - Horizontal bar: width=footprint[0], height=arm_width
        - Vertical bar: width=arm_width, height=(footprint[1] - arm_width)

        Non-overlapping rectangles centered at origin.
        Mass is distributed by area ratio.
        """
        W, H = footprint_m
        w = arm_width_m
        half_thickness = thickness_m / 2.0

        horiz_bar_size_x = W / 2.0
        horiz_bar_size_y = w / 2.0
        horiz_bar_pos_y = -(H - w) / 2.0

        vert_bar_size_x = w / 2.0
        vert_bar_size_y = (H - w) / 2.0
        vert_bar_pos_x = -(W - w) / 2.0
        vert_bar_pos_y = w / 2.0

        horiz_bar_area = W * w
        vert_bar_area = w * (H - w)
        total_area = horiz_bar_area + vert_bar_area

        mass_horiz_bar = mass_kg * (horiz_bar_area / total_area)
        mass_vert_bar = mass_kg * (vert_bar_area / total_area)

        geoms = f"""      <geom
        name="object_geom_horiz"
        type="box"
        pos="0 {horiz_bar_pos_y:.6f} 0"
        size="{horiz_bar_size_x:.6f} {horiz_bar_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_horiz_bar:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />
      <geom
        name="object_geom_vert"
        type="box"
        pos="{vert_bar_pos_x:.6f} {vert_bar_pos_y:.6f} 0"
        size="{vert_bar_size_x:.6f} {vert_bar_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_vert_bar:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />"""

        return geoms

    def _make_cross_geoms(
        self,
        footprint_m: list[float],
        arm_width_m: float,
        thickness_m: float,
        mass_kg: float,
        rgba: str,
        contype: int,
        conaffinity: int,
    ) -> str:
        """
        Cross shape: non-overlapping decomposition.
        - Central vertical bar: arm_width × footprint[1]
        - Left arm: arm_width × arm_width
        - Right arm: arm_width × arm_width

        Mass is distributed by area ratio.
        """
        W, H = footprint_m
        w = arm_width_m
        half_thickness = thickness_m / 2.0

        # Central vertical bar: full height, arm_width width
        vert_bar_size_x = w / 2.0
        vert_bar_size_y = H / 2.0
        vert_bar_pos_x = 0
        vert_bar_pos_y = 0

        # Left arm: arm_width × arm_width
        left_arm_size_x = w / 2.0
        left_arm_size_y = w / 2.0
        left_arm_pos_x = -(W - w) / 2.0
        left_arm_pos_y = 0

        # Right arm: arm_width × arm_width
        right_arm_size_x = w / 2.0
        right_arm_size_y = w / 2.0
        right_arm_pos_x = (W - w) / 2.0
        right_arm_pos_y = 0

        # Calculate areas and masses
        vert_bar_area = w * H
        left_arm_area = w * w
        right_arm_area = w * w
        total_area = vert_bar_area + left_arm_area + right_arm_area

        mass_vert_bar = mass_kg * (vert_bar_area / total_area)
        mass_left_arm = mass_kg * (left_arm_area / total_area)
        mass_right_arm = mass_kg * (right_arm_area / total_area)

        geoms = f"""      <geom
        name="object_geom_vert"
        type="box"
        pos="{vert_bar_pos_x:.6f} {vert_bar_pos_y:.6f} 0"
        size="{vert_bar_size_x:.6f} {vert_bar_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_vert_bar:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />
      <geom
        name="object_geom_left"
        type="box"
        pos="{left_arm_pos_x:.6f} {left_arm_pos_y:.6f} 0"
        size="{left_arm_size_x:.6f} {left_arm_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_left_arm:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />
      <geom
        name="object_geom_right"
        type="box"
        pos="{right_arm_pos_x:.6f} {right_arm_pos_y:.6f} 0"
        size="{right_arm_size_x:.6f} {right_arm_size_y:.6f} {half_thickness:.6f}"
        mass="{mass_right_arm:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />"""

        return geoms

    def _make_bar_geoms(
        self,
        footprint_m: list[float],
        thickness_m: float,
        mass_kg: float,
        rgba: str,
        contype: int,
        conaffinity: int,
    ) -> str:
        """Bar shape: single rectangular box."""
        W, H = footprint_m
        half_thickness = thickness_m / 2.0

        geoms = f"""      <geom
        name="object_geom"
        type="box"
        pos="0 0 0"
        size="{W/2.0:.6f} {H/2.0:.6f} {half_thickness:.6f}"
        mass="{mass_kg:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />"""

        return geoms

    def _get_additional_object_geoms(
        self,
        shape_type: str,
        rgba: str,
        contype: int,
        conaffinity: int,
    ) -> str:
        """Generate geoms for additional pushable object shapes (square, cylinder)."""
        obj_spec = self.specs["obstacles"][shape_type]

        if shape_type == "square":
            size_mm = obj_spec["size_mm"]
            mass_g = obj_spec["nominal_mass_g"]

            size_m = [s / 1000.0 for s in size_mm]
            mass_kg = mass_g / 1000.0

            geoms = f"""      <geom
        name="object_geom"
        type="box"
        pos="0 0 0"
        size="{size_m[0]/2.0:.6f} {size_m[1]/2.0:.6f} {size_m[2]/2.0:.6f}"
        mass="{mass_kg:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />"""

        elif shape_type == "cylinder":
            radius_mm = obj_spec["radius_mm"]
            height_mm = obj_spec["height_mm"]
            mass_g = obj_spec["nominal_mass_g"]

            radius_m = radius_mm / 1000.0
            half_height_m = (height_mm / 1000.0) / 2.0
            mass_kg = mass_g / 1000.0

            geoms = f"""      <geom
        name="object_geom"
        type="cylinder"
        pos="0 0 0"
        size="{radius_m:.6f} {half_height_m:.6f}"
        mass="{mass_kg:.6f}"
        rgba="{rgba}"
        contype="{contype}"
        conaffinity="{conaffinity}"
      />"""

        else:
            raise ValueError(f"Unsupported additional object shape: {shape_type}")

        return geoms
