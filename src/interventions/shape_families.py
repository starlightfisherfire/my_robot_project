from __future__ import annotations

from copy import deepcopy


SHAPE_SPECS = {
    "T_shape": {
        "shape_family": "T_shape",
        "object_shape": "T",
        "object_size_x": 0.08,
        "object_size_y": 0.08,
        "object_mass": 0.05,
        "object_friction": None,
    },
    "L_shape": {
        "shape_family": "L_shape",
        "object_shape": "L",
        "object_size_x": 0.08,
        "object_size_y": 0.08,
        "object_mass": 0.05,
        "object_friction": None,
    },
}


def get_shape_spec(shape_family: str) -> dict:
    """
    Return object metadata for a shape family.

    v0.1:
        T_shape is ID.
        L_shape is shape OOD.

    Note:
        object_size_x / object_size_y / object_mass are placeholder values.
        Replace them with CAD or measured values when the real printed objects
        are finalized.
    """
    if shape_family not in SHAPE_SPECS:
        raise ValueError(
            f"Unknown shape_family={shape_family}. "
            f"Available: {list(SHAPE_SPECS.keys())}"
        )

    return deepcopy(SHAPE_SPECS[shape_family])


def get_shape_families() -> list[str]:
    return list(SHAPE_SPECS.keys())