from __future__ import annotations

import math

import numpy as np


def extract_obstacle_geometry(
    template: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extract obstacle positions [N,2] and enclosing-circle radii [N] from template.

    Box obstacles are converted to enclosing circles:
        radius = sqrt((size_x/2)^2 + (size_y/2)^2)

    Returns (None, None) if template has no valid obstacles.
    """
    obstacles = template.get("obstacles", [])
    if not obstacles:
        return None, None

    positions = []
    radii = []

    for obs in obstacles:
        if not obs.get("valid", True):
            continue

        pose = obs["pose"]
        positions.append([pose["x"], pose["y"]])

        sx = obs.get("size_x", 0.04)
        sy = obs.get("size_y", 0.04)
        radii.append(math.sqrt((sx / 2.0) ** 2 + (sy / 2.0) ** 2))

    if not positions:
        return None, None

    return (
        np.array(positions, dtype=np.float64),
        np.array(radii, dtype=np.float64),
    )
