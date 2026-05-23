# state_profile_ablation_plan.md

**Version:** 0.1  
**Date:** 2026-05-24  

---

## Purpose

Define profile-based ablation for visual_structured_state_v2. Each profile selects a subset of features from the full v2 schema. By training models on different profiles, we can ablate which structured visual information is necessary for dynamics learning.

## Profiles

### 1. state16_minimal_legacy
- Purpose: Baseline, equivalent to current canonical_state16
- Includes: object pose, EE position, goal, obstacle positions, velocity, mass, friction
- Excludes: relations, temporal history, nuisance, privileged contact
- Note: This is the Paper 1 legacy baseline

### 2. visual_object_state_v2
- Purpose: Object-centric visual state without relations
- Includes: object_tokens (pose, kinematics, geometry), EE, goal, obstacles
- Excludes: relation_tokens, temporal_tokens, nuisance, privileged physics

### 3. visual_object_relation_state_v2
- Purpose: Full visual structured state with relations
- Includes: object_tokens + relation_tokens + EE + goal + obstacles
- Excludes: temporal_tokens(optional), nuisance, privileged physics
- Note: Main v2 profile for comparison

### 4. history_dynamics_state_v2
- Purpose: Full v2 + temporal history
- Includes: profile 3 + temporal_tokens
- Excludes: nuisance, privileged physics

### 5. nuisance_randomized_state_v2
- Purpose: Test invariance to visual nuisance
- Includes: profile 3 + visual_nuisance values (randomized)
- Excludes: privileged physics
- Note: nuisance values are randomized, model should learn invariance

### 6. privileged_physics_oracle
- Purpose: Test upper bound with oracle physics access
- Includes: profile 3 + privileged_physics
- Excludes: nuisance
- Note: For upper-bound analysis only, NOT for main claim

### 7. solver_internal_forbidden
- Purpose: Document what is explicitly forbidden
- Includes: MuJoCo internal solver state
- Status: FORBIDDEN for all profiles
- Reason: Not recoverable from vision on real robot

## Ablation Matrix

| Profile | Object | Relations | Temporal | Nuisance | Privileged | Target |
|---------|--------|-----------|----------|----------|------------|--------|
| state16_legacy | 16D legacy | ❌ | v-t only | ❌ | mass+fric only | Δpose |
| visual_object | ✅ v2 | ❌ | ❌ | ❌ | ❌ | Δpose |
| visual_relation | ✅ v2 | ✅ | ❌ | ❌ | ❌ | Δpose |
| history_dynamics | ✅ v2 | ✅ | ✅ | ❌ | ❌ | Δpose |
| nuisance_randomized | ✅ v2 | ✅ | ❌ | ✅ (rand) | ❌ | Δpose |
| privileged_oracle | ✅ v2 | ✅ | ❌ | ❌ | ✅ (all) | Δpose |
| solver_forbidden | ❌ | ❌ | ❌ | ❌ | ✅ (solver) | N/A |

## Research Questions

1. Does explicit relation encoding reduce the OOD gap compared to implicit (state16)?
2. Does temporal history improve multi-step rollout stability?
3. Is the v2 representation invariant to visual nuisance?
4. How much does privileged physics (mass/friction/contact) improve the upper bound?
