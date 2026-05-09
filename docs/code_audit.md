# Code Audit: File-Level Status Table

**Last updated:** 2026-05-09

This document provides a file-level audit of the Paper 1 codebase, tracking validation status and limitations.

---

## Reset / Split / Metadata Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/interventions/shape_families.py` | Data | Define shape families for OOD | `SHAPE_FAMILIES`, `get_shape_config()` | shape_name | shape config dict | generate_reset_templates.py | N/A | PASS | Shape families defined correctly | Not tested in actual MuJoCo env | None needed |
| `src/interventions/layout_families.py` | Data | Define layout families for OOD | `LAYOUT_FAMILIES`, `get_layout_config()` | layout_name | layout config dict | generate_reset_templates.py | N/A | PASS | Layout families defined correctly | Obstacles not instantiated in MuJoCo yet | Implement obstacles in MuJoCo |
| `src/interventions/sampling_rules.py` | Data | Validate reset templates | `validate_reset_templates()` | templates list | raises on error | reset_template_loader.py | N/A | PASS | Schema validation works | Does not validate geometric feasibility | None needed |
| `scripts/generate_reset_templates.py` | Data | Generate reset templates | `main()` | CLI args | reset_templates_v0.json | User | `PYTHONPATH=. python scripts/generate_reset_templates.py` | PASS | Generates valid templates | Does not guarantee task solvability | None needed |
| `data/sim/metadata/reset_templates_v0.json` | Data | Reset template storage | N/A | N/A | JSON templates | reset_template_loader.py | N/A | PASS | 140 templates generated | Obstacles not yet instantiated | None needed |
| `src/interventions/reset_template_loader.py` | Data | Load and query templates | `load_reset_templates()`, `get_templates_by_split()` | JSON path | templates list | check_mpc_capacity.py | `PYTHONPATH=. python scripts/debug_reset_templates.py` | PASS | Loading and querying works | N/A | None needed |
| `src/data/metadata_schema.py` | Data | Episode metadata schema | `EpisodeMetadata`, `Pose2D`, `ObstacleMetadata` | dict | dataclass | reset_template_loader.py | `PYTHONPATH=. python scripts/debug_metadata_schema.py` | PASS | Schema validation works | Not used in actual data collection yet | Use in data collection |

---

## Environment Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/envs/toy_push_env.py` | Env | Toy dynamics for interface testing | `ToyPushEnv` | action | next_state | oracle_rollout.py | `PYTHONPATH=. python scripts/debug_cem_mpc_toy.py` | PASS | Interface works, toy dynamics correct | Ignores obstacles, not MuJoCo, not real physics | None needed (toy only) |
| `src/envs/mujoco_push_env.py` | Env | MuJoCo dynamics scaffold | `MujocoPushEnv`, `MujocoPushState` | action | next_state | mujoco_oracle_rollout.py | `PYTHONPATH=. python scripts/debug_mujoco_env.py` | PARTIAL | reset/step/clone/restore works, contact detection works | **Obstacles not instantiated**, collision detection placeholder, goal_site not dynamic | Implement obstacles instantiation |
| `scripts/debug_mujoco_env.py` | Debug | Test MuJoCo env interface | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_mujoco_env.py` | PASS | Env interface validated | Does not test obstacles | None needed |

---

## Planner Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/planners/cost_functions.py` | Planner | Cost computation core | `rollout_cost()`, `CostWeights`, `wrap_angle()` | rollout result, weights | scalar cost | oracle_rollout.py, mujoco_oracle_rollout.py | N/A | PASS | Cost computation correct, angle wrapping safe | N/A | None needed |
| `src/planners/cem_mpc.py` | Planner | CEM optimizer (fixed) | `CEMMPC`, `CEMResult` | cost_fn | action_sequence | check_mpc_capacity.py | `PYTHONPATH=. python scripts/debug_cem_mpc_toy.py` | PASS | CEM optimization works | Not tuned for task success yet | Tune for task success |
| `src/planners/oracle_rollout.py` | Planner | Toy oracle rollout | `rollout_action_sequence()`, `oracle_rollout_cost()` | ToyPushEnv, action_sequence | rollout result, cost | toy_oracle_capacity.py | `PYTHONPATH=. python scripts/debug_oracle_rollout.py` | PASS | Toy oracle rollout works | Not MuJoCo, ignores obstacles | None needed (toy only) |
| `src/planners/mujoco_oracle_rollout.py` | Planner | MuJoCo oracle rollout | `rollout_action_sequence_mujoco()`, `mujoco_oracle_rollout_cost()` | MujocoPushEnv, action_sequence | rollout result, cost | mujoco_oracle_capacity.py | `PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py` | PASS | MuJoCo oracle rollout works, restore_state correct | Does not test with obstacles | Test with obstacles |
| `scripts/debug_oracle_rollout.py` | Debug | Test toy oracle rollout | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_oracle_rollout.py` | PASS | Toy oracle rollout validated | Not MuJoCo | None needed |
| `scripts/debug_mujoco_oracle_rollout.py` | Debug | Test MuJoCo oracle rollout | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py` | PASS | MuJoCo oracle rollout validated | Does not test with obstacles | Test with obstacles |
| `scripts/debug_cem_mpc_toy.py` | Debug | Test CEM with toy env | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_cem_mpc_toy.py` | PASS | CEM + toy oracle works | Not MuJoCo | None needed |

---

## Capacity Check Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/metrics/planner_capacity.py` | Metrics | State sanity checks | `run_state_sanity()` | templates | report | check_mpc_capacity.py | N/A | PASS | Geometric sanity validated | Does not run CEM | None needed |
| `src/metrics/toy_oracle_capacity.py` | Metrics | Toy oracle-MPC capacity | `run_toy_oracle_mpc_capacity()` | templates | report | check_mpc_capacity.py | N/A | PASS | Toy oracle-MPC works | Not MuJoCo, ignores obstacles | None needed (toy only) |
| `src/metrics/mujoco_oracle_capacity.py` | Metrics | MuJoCo oracle-MPC capacity | `run_mujoco_oracle_mpc_capacity()`, `evaluate_one_template_mujoco_oracle_mpc()` | templates | report | check_mpc_capacity.py | N/A | PARTIAL | Interface smoke test passed (5/5), cost improvement works | **success_rate=0**, task success not achieved, obstacles not tested | Tune for task success OR implement obstacles first |
| `scripts/check_mpc_capacity.py` | Metrics | Unified capacity check entry | `main()`, `run_state_sanity_mode()`, `run_toy_oracle_mpc_mode()`, `run_mujoco_oracle_mpc_mode()` | CLI args | report JSON | User | `PYTHONPATH=. python scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc` | PARTIAL | state_sanity: 140/140, toy_oracle_mpc: 20/20, mujoco_oracle_mpc: 5/5 interface | mujoco_oracle_mpc success_rate=0, task not solved | Tune or implement obstacles |

---

## Data Normalization Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/data/state_normalizer.py` | Data | State normalization (CRITICAL) | `StateNormalizer`, `fit()`, `transform()` | raw state | normalized state | RIGWorldModel (future) | `PYTHONPATH=. python scripts/debug_state_normalizer.py` | PASS | Interface works, fit/transform correct | **Not tested with real data**, **split leakage not validated** | Validate split protocol in training |
| `scripts/debug_state_normalizer.py` | Debug | Test normalizer | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_state_normalizer.py` | PASS | Normalizer interface validated | Dummy data only | None needed |

---

## Model Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/models/encoders.py` | Model | Three encoder variants | `FlatEncoder`, `ObjectCentricEncoder`, `CausalityAwareEncoder` | [B,H,N,D] state | [B,256] z | rig_world.py | `PYTHONPATH=. python scripts/debug_encoder_variants.py` | PASS | Forward/backward works, shapes correct | Not trained, not tested on real data | Train on real data |
| `src/models/heads.py` | Model | Prediction heads | `DynamicsHead`, `SubgoalHead` | z, action | pred_delta, pred_subgoal | rig_world.py | `PYTHONPATH=. python scripts/debug_rig_world_model.py` | PASS | Forward/backward works, shapes correct | Not trained, not tested on real data | Train on real data |
| `src/models/losses.py` | Model | Training losses | `dynamics_loss()`, `subgoal_loss()` | pred, target | scalar loss | train script (future) | N/A | PASS | Loss computation works | Not used in training yet | Use in training |
| `src/models/rig_world.py` | Model | Unified model interface | `RIGWorldModel` | state, action | z, pred_delta, pred_subgoal | train/eval scripts (future) | `PYTHONPATH=. python scripts/debug_rig_world_model.py` | PASS | Unified interface works, all variants work | Not trained, not tested on real data | Train on real data |
| `scripts/debug_encoder_variants.py` | Debug | Test encoders | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_encoder_variants.py` | PASS | All encoders validated | Dummy data only | None needed |
| `scripts/debug_rig_world_model.py` | Debug | Test RIGWorldModel | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_rig_world_model.py` | PASS | RIGWorldModel validated | Dummy data only | None needed |

---

## Learned Rollout Layer (Future)

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/planners/rollout_model.py` | Planner | Learned model rollout | `rollout_with_learned_model()` (future) | RIGWorldModel, action_sequence | predicted states | CEM-MPC (future) | N/A | NOT_RUN | N/A | Not implemented yet | Implement after model training |

---

## Config Files

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `configs/planner/cem_mpc.yaml` | Config | CEM-MPC config | N/A | N/A | config dict | train/eval scripts (future) | N/A | PARTIAL | Config defined | **horizon=10 is for learned eval, not oracle strong**, config policy not clear | Document config policy |
| `configs/planner/cost_weights.yaml` | Config | Cost weights | N/A | N/A | weights dict | cost_functions.py | N/A | PARTIAL | Weights defined | Not tuned for task success | Tune for task success |
| `configs/splits.yaml` | Config | Split definitions | N/A | N/A | split config | generate_reset_templates.py | N/A | PASS | Splits defined correctly | N/A | None needed |

---

## Summary Statistics

### By Status:
- **PASS**: 23 files (fully validated for their scope)
- **PARTIAL**: 4 files (interface works, but limitations exist)
- **NOT_RUN**: 1 file (not implemented yet)
- **FUTURE**: Multiple files (planned but not started)

### Critical Blockers:
1. **Obstacles not instantiated** in MujocoPushEnv → blocks layout OOD capacity
2. **Task success not achieved** (success_rate=0) → blocks learned model attribution
3. **Planner config policy unclear** → risks unfair comparison

### Next Critical Path:
1. **Decision:** Tune Oracle-MPC for task success OR implement obstacles first
2. **After decision:** Complete chosen path
3. **Then:** Collect sim data
4. **Then:** Train learned models
5. **Then:** Evaluate learned model + MPC
6. **Then:** Analyze OOD gap

---

## File Dependency Graph

```
reset_templates_v0.json
    ↓
reset_template_loader.py
    ↓
MujocoPushEnv.reset_from_template()
    ↓
mujoco_oracle_rollout.rollout_action_sequence_mujoco()
    ↓
cost_functions.rollout_cost()
    ↓
CEMMPC.plan(cost_fn)
    ↓
mujoco_oracle_capacity.evaluate_one_template_mujoco_oracle_mpc()
    ↓
check_mpc_capacity.py --mode mujoco_oracle_mpc
```

---

## Validation Commands Quick Reference

```bash
# State sanity
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode state_sanity

# Toy oracle-MPC
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode toy_oracle_mpc --split train_sim_id --max-templates 20

# MuJoCo oracle-MPC
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc --split train_sim_id --max-templates 5

# Model smoke tests
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_encoder_variants.py

# Environment tests
PYTHONPATH=. python scripts/debug_mujoco_env.py
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py

# Data tests
PYTHONPATH=. python scripts/debug_state_normalizer.py
PYTHONPATH=. python scripts/debug_reset_templates.py
PYTHONPATH=. python scripts/debug_metadata_schema.py
```
