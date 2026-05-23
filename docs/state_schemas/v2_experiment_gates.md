# v2_experiment_gates.md

## Gate V2-0: Design Docs
- Status: 🔧 IN PROGRESS
- Output: docs/state_schemas/*.md

## Gate V2-1: Schema YAML
- Output: configs/state_schema/*.yaml
- Self-check: scripts/audit_visual_state_v2_schema.py

## Gate V2-2: Python Skeleton
- Output: src/state_schemas/*.py
- Self-check: py_compile + import

## Gate V2-3: Episode IO
- Output: src/data/v2_episode_writer.py, v2_episode_reader.py

## Gate V2-4: Export Smoke
- Output: scripts/export_visual_state_v2.py
- Exports 3 episodes from mppi_stage2c_state16

## Gate V2-5: Profile Mask
- Verifies: apply_profile_to_episode works for all profiles

## Gate V2-6: v2_to_state16 Adapter
- Verifies: v2 episode can convert back to canonical_state16
- Verifies: converted state16 can pass through RIGWorldModel.forward()

## Gate V2-7: Offline Dynamics
- Train v2 encoder on v2 data, compare with state16 baseline

## Gate V2-8: Nuisance Invariance
- Train on randomized nuisance, test invariance

## Gate V2-9: Privileged Oracle
- Upper bound using privileged physics

## Gate V2-10: Learned MPC Internal
- v2 encoder + learned rollout + CEM-MPC

## Gate V2-11: MuJoCo Closed-Loop
- Real MuJoCo eval with v2 encoder
