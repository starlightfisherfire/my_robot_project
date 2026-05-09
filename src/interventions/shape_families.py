from __future__ import annotations

from copy import deepcopy


SHAPE_SPECS = {
    "T_shape": {
        "shape_family": "T_shape",
        "object_shape": "T",
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "object_friction": None,
    },
    "L_shape": {
        "shape_family": "L_shape",
        "object_shape": "L",
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "object_friction": None,
    },
}


def get_shape_spec(shape_family: str) -> dict:
    """
    Return object metadata for a shape family.

    v0.1:
        T_shape is ID.
        L_shape is shape OOD.

    v0.2 (2026-05-09):
        Updated to use real SO-101 physical object specifications:
        - Footprint: 48×48 mm (0.048 m)
        - Mass: 19.05 g (0.01905 kg)
        - Thickness: 12 mm
        - Volume: 15,360 mm³

        See configs/object_specs.yaml for full specifications.
    """
    if shape_family not in SHAPE_SPECS:
        raise ValueError(
            f"Unknown shape_family={shape_family}. "
            f"Available: {list(SHAPE_SPECS.keys())}"
        )

    return deepcopy(SHAPE_SPECS[shape_family])


def get_shape_families() -> list[str]:
    return list(SHAPE_SPECS.keys())