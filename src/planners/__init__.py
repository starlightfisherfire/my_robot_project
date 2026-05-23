from .cem_mpc import CEMMPC, CEMResult
from .mppi import MPPI, MPPIResult
from .multimodal_cem_mpc import MultimodalCEMMPC, MultimodalCEMResult
from .obstacle_utils import extract_obstacle_geometry
from .cost_functions import (
    CostWeights,
    rollout_cost,
    obstacle_proximity_cost,
    pose_error,
    wrap_angle,
)

PLANNER_REGISTRY = {
    "cem": CEMMPC,
    "mppi": MPPI,
    "multimodal_cem": MultimodalCEMMPC,
}


def make_planner(name: str, **kwargs):
    """
    Factory for sweep selection.

    Usage:
        planner = make_planner("mppi", horizon=15, temperature=0.1)
        planner = make_planner("multimodal_cem", lateral_offset=0.6)
        planner = make_planner("cem")  # original baseline
    """
    if name not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner '{name}'. Choose from: {list(PLANNER_REGISTRY.keys())}"
        )
    return PLANNER_REGISTRY[name](**kwargs)


__all__ = [
    "CEMMPC",
    "CEMResult",
    "MPPI",
    "MPPIResult",
    "MultimodalCEMMPC",
    "MultimodalCEMResult",
    "CostWeights",
    "rollout_cost",
    "obstacle_proximity_cost",
    "extract_obstacle_geometry",
    "pose_error",
    "wrap_angle",
    "make_planner",
    "PLANNER_REGISTRY",
]
