# v2_transition_from_state16.md

## Transition Strategy

state16 is the current anchor. v2 is a framework expansion.

### Phase A: Coexistence (NOW)
- state16 pipeline unchanged
- v2 schema/docs/skeleton built in parallel
- v2_to_state16 adapter ensures backward compatibility

### Phase B: v2 Smoke (NEXT)
- Export 3 episodes from mppi_stage2c_state16 to v2 format
- Verify adapter round-trip
- Verify profile masks

### Phase C: v2 Training (AFTER 16D pilot)
- Collect v2 episodes from MuJoCo MPPI
- Train v2 encoder on profile 3 (visual_object_relation)
- Compare with state16 encoder on same task

### Phase D: Ablation (AFTER comparison)
- Run profile ablation matrix
- Nuisance invariance test
- Privileged physics oracle

## Backward Compatibility
- v2_to_state16_adapter ensures v2 data can feed into state16 models
- state16 models are not modified
- v2 evaluation uses the same CEM-MPC planner
