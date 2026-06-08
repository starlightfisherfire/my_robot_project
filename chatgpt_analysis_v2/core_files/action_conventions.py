# src/planners/action_conventions.py
"""Action convention helper for Paper1 push manipulation.

This module provides a unified interface for converting between
different action representations:

- a_norm: normalized action [-1, 1] (planner/env space)
- a_phys: physical velocity [m/s] (model input space)
- disp: displacement [m] (state update space)

Convention:
    a_phys = a_norm * max_speed_mps
    disp = a_phys * control_dt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ActionConvention:
    """Action convention specification."""
    
    name: str
    action_dim: int = 2
    max_speed_mps: float = 0.5
    control_dt: float = 0.1
    model_action_type: Literal["physical_velocity", "physical_displacement", "normalized"] = "physical_velocity"
    env_action_type: Literal["normalized", "physical_velocity"] = "normalized"
    
    def describe(self) -> dict:
        """Return human-readable description."""
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "max_speed_mps": self.max_speed_mps,
            "control_dt": self.control_dt,
            "model_action_type": self.model_action_type,
            "env_action_type": self.env_action_type,
            "model_action_range": f"[{-self.max_speed_mps:.3f}, {self.max_speed_mps:.3f}] m/s",
            "env_action_range": "[-1.0, 1.0] (normalized)",
            "displacement_per_step": f"{self.max_speed_mps * self.control_dt:.4f} m",
        }
    
    def planner_to_model_action(self, a_norm: np.ndarray) -> np.ndarray:
        """Convert normalized planner action to model input action.
        
        Args:
            a_norm: Normalized action, shape [..., 2]
            
        Returns:
            Model action (physical velocity), shape [..., 2]
        """
        a_norm = np.asarray(a_norm, dtype=np.float64)
        self._check_shape(a_norm, "a_norm")
        self._check_finite(a_norm, "a_norm")
        
        if self.model_action_type == "physical_velocity":
            a_phys = a_norm * self.max_speed_mps
        elif self.model_action_type == "physical_displacement":
            a_phys = a_norm * self.max_speed_mps * self.control_dt
        elif self.model_action_type == "normalized":
            a_phys = a_norm.copy()
        else:
            raise ValueError(f"Unknown model_action_type: {self.model_action_type}")
        
        self._check_finite(a_phys, "a_phys")
        return a_phys
    
    def planner_to_env_action(self, a_norm: np.ndarray) -> np.ndarray:
        """Convert planner action to env.step() action.
        
        Args:
            a_norm: Normalized action, shape [..., 2]
            
        Returns:
            Env action, shape [..., 2]
        """
        a_norm = np.asarray(a_norm, dtype=np.float64)
        self._check_shape(a_norm, "a_norm")
        self._check_finite(a_norm, "a_norm")
        
        if self.env_action_type == "normalized":
            a_env = np.clip(a_norm, -1.0, 1.0)
        elif self.env_action_type == "physical_velocity":
            a_env = a_norm * self.max_speed_mps
        else:
            raise ValueError(f"Unknown env_action_type: {self.env_action_type}")
        
        self._check_finite(a_env, "a_env")
        return a_env
    
    def model_action_to_state_displacement(self, a_model: np.ndarray) -> np.ndarray:
        """Convert model action to state displacement.
        
        Args:
            a_model: Model action, shape [..., 2]
            
        Returns:
            Displacement in meters, shape [..., 2]
        """
        a_model = np.asarray(a_model, dtype=np.float64)
        self._check_shape(a_model, "a_model")
        self._check_finite(a_model, "a_model")
        
        if self.model_action_type == "physical_velocity":
            disp = a_model * self.control_dt
        elif self.model_action_type == "physical_displacement":
            disp = a_model.copy()
        elif self.model_action_type == "normalized":
            disp = a_model * self.max_speed_mps * self.control_dt
        else:
            raise ValueError(f"Unknown model_action_type: {self.model_action_type}")
        
        self._check_finite(disp, "disp")
        return disp
    
    def _check_shape(self, a: np.ndarray, name: str) -> None:
        """Check action shape."""
        if a.shape[-1] != self.action_dim:
            raise ValueError(
                f"{name} last dim must be {self.action_dim}, got {a.shape}"
            )
    
    def _check_finite(self, a: np.ndarray, name: str) -> None:
        """Check for NaN/inf."""
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains non-finite values")


# Pre-defined conventions
PAPER1_CONVENTION = ActionConvention(
    name="paper1_push_v1",
    action_dim=2,
    max_speed_mps=0.5,
    control_dt=0.1,
    model_action_type="physical_velocity",
    env_action_type="normalized",
)


def get_convention(name: str = "paper1_push_v1") -> ActionConvention:
    """Get pre-defined convention by name."""
    conventions = {
        "paper1_push_v1": PAPER1_CONVENTION,
    }
    if name not in conventions:
        raise ValueError(f"Unknown convention: {name}. Available: {list(conventions.keys())}")
    return conventions[name]
