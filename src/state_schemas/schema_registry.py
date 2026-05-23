# src/state_schemas/schema_registry.py
"""Schema and profile loading for visual_structured_state_v2."""

from pathlib import Path
from typing import Any

import yaml


def load_schema(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_profiles(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(profiles: dict, name: str) -> dict:
    prof = profiles.get("profiles", {}).get(name)
    if prof is None:
        raise KeyError(f"Profile '{name}' not found")
    return prof


def expand_profile_features(profiles: dict, name: str, visited: set | None = None) -> set[str]:
    """Recursively expand profile includes to a set of feature names."""
    if visited is None:
        visited = set()
    if name in visited:
        return set()
    visited.add(name)
    prof = profiles.get("profiles", {}).get(name, {})
    if prof.get("status") == "FORBIDDEN":
        return set()
    feats = set()
    for entry in prof.get("includes", []):
        if entry in profiles.get("profiles", {}):
            feats |= expand_profile_features(profiles, entry, visited)
        else:
            for part in entry.replace(",", ";").split(";"):
                part = part.strip()
                if part:
                    feats.add(part)
    return feats


def validate_schema_metadata(schema: dict) -> list[str]:
    """Check that every feature has required metadata fields."""
    issues = []
    required_fields = ["name", "source", "use_in_main", "use_in_ablation", "leakage_risk"]
    for sec_name in ["object_tokens", "proprio_action_tokens", "relation_tokens",
                     "temporal_tokens", "goal_features", "obstacle_features",
                     "visual_nuisance", "privileged_physics"]:
        for feat in schema.get(sec_name, {}).get("features", []):
            for field in required_fields:
                if field not in feat:
                    issues.append(f"Missing {field} in {sec_name}/{feat.get('name', '?')}")
    # Check constraints
    for feat in schema.get("privileged_physics", {}).get("features", []):
        if feat.get("use_in_main"):
            issues.append(f"PRIVILEGED LEAK: {feat['name']} has use_in_main=true")
    for feat in schema.get("visual_nuisance", {}).get("features", []):
        if feat.get("use_in_main"):
            issues.append(f"NUISANCE LEAK: {feat['name']} has use_in_main=true")
    return issues
