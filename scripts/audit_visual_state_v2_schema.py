#!/usr/bin/env python3
"""audit_visual_state_v2_schema.py — Validate v2 schema and profiles YAML files."""

import json, sys, yaml
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def check_schema(path):
    with open(path) as f:
        schema = yaml.safe_load(f)
    issues = []
    # Check top-level sections
    for sec in ["object_tokens", "proprio_action_tokens", "relation_tokens",
                "temporal_tokens", "goal_features", "obstacle_features",
                "visual_nuisance", "privileged_physics", "targets", "masks"]:
        if sec not in schema:
            issues.append(f"Missing section: {sec}")
    # Check each feature has required metadata
    for sec_name in ["object_tokens", "proprio_action_tokens", "relation_tokens",
                     "temporal_tokens", "goal_features", "obstacle_features",
                     "visual_nuisance", "privileged_physics"]:
        sec = schema.get(sec_name, {})
        feats = sec.get("features", [])
        for feat in feats:
            for field in ["name", "source", "use_in_main", "use_in_ablation", "leakage_risk"]:
                if field not in feat:
                    issues.append(f"{sec_name}/{feat.get('name','?')}: missing {field}")
    # Check privileged features NOT in main
    for feat in schema.get("privileged_physics", {}).get("features", []):
        if feat.get("use_in_main", True):
            issues.append(f"PRIVILEGED LEAK: {feat['name']} has use_in_main=true")
    # Check nuisance features NOT in main
    for feat in schema.get("visual_nuisance", {}).get("features", []):
        if feat.get("use_in_main", True):
            issues.append(f"NUISANCE LEAK: {feat['name']} has use_in_main=true")
    return issues

def check_profiles(path, schema_path):
    with open(path) as f:
        profiles = yaml.safe_load(f)
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    # Collect all feature names from schema
    all_feats = set()
    for sec_name in ["object_tokens", "proprio_action_tokens", "relation_tokens",
                     "temporal_tokens", "goal_features", "obstacle_features",
                     "visual_nuisance", "privileged_physics"]:
        for feat in schema.get(sec_name, {}).get("features", []):
            all_feats.add(feat["name"])

    issues = []
    for name, prof in profiles.get("profiles", {}).items():
        if "status" in prof and prof["status"] == "FORBIDDEN":
            continue
        for feat_name in prof.get("includes", []):
            # Skip "all" and references
            if feat_name in ("all",) or feat_name.startswith("visual_"):
                continue
            if feat_name not in all_feats:
                issues.append(f"Profile {name}: feature '{feat_name}' not in schema")
    return issues

def main():
    schema_path = REPO / "configs/state_schema/visual_structured_state_v2.yaml"
    profiles_path = REPO / "configs/state_schema/state_profiles.yaml"

    results = {}
    schema_issues = check_schema(schema_path)
    results["schema_check"] = {"status": "PASS" if not schema_issues else "FAIL", "issues": schema_issues}
    profile_issues = check_profiles(profiles_path, schema_path)
    results["profile_check"] = {"status": "PASS" if not profile_issues else "FAIL", "issues": profile_issues}

    overall = "PASS" if (not schema_issues and not profile_issues) else "FAIL"

    print(f"  {'PASS' if not schema_issues else 'FAIL'} | schema_check ({len(schema_issues)} issues)")
    for i in schema_issues[:5]:
        print(f"    ⚠ {i}")
    print(f"  {'PASS' if not profile_issues else 'FAIL'} | profile_check ({len(profile_issues)} issues)")
    for i in profile_issues[:5]:
        print(f"    ⚠ {i}")
    print(f"OVERALL: {overall}")

    out = REPO / "runs/self_check/visual_state_v2_schema_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"overall_status": overall, "checks": results}, f, indent=2)

    return 0 if overall == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
