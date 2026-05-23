# v2_dataset_card.md

## Dataset Identity
- Name: visual_state_v2
- Schema: configs/state_schema/visual_structured_state_v2.yaml
- Profiles: configs/state_schema/state_profiles.yaml
- Version: 0.1

## Data Structure
```
data/sim/visual_state_v2/
├── metadata/
│   ├── episodes.jsonl
│   └── profiles.json
├── episodes/
│   └── {episode_id}.npz
└── qc/
    └── rejected_episodes.jsonl
```

## Episode Contents (v2)
- object_tokens: dict of arrays
- relation_tokens: dict of arrays
- temporal_tokens: dict of arrays
- proprio_action_tokens: dict of arrays
- visual_nuisance: metadata dict
- privileged_physics: dict of arrays
- targets: dict of arrays
- masks: dict of arrays
- actions_physical: [T,2]
- object_poses: [T,3]
- next_object_poses: [T,3]
- goal_pose: [3]
- schema_version: string

## Quality Checks
- All required keys present
- No NaN/inf in main input arrays
- Masks are consistent with data
- Privileged fields use_in_main=false
