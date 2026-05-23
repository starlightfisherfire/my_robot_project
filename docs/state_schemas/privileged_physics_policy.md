# privileged_physics_policy.md

## Purpose
Document privileged physics variables and their usage constraints.

## Variables

| Variable | Source | Real Robot | Usage |
|----------|--------|------------|-------|
| object_mass | MuJoCo model | Weighable ✅ | Oracle ablation |
| object_friction | MuJoCo model | Hard ❌ | Oracle ablation |
| object_inertia | MuJoCo model | Hard ❌ | Oracle ablation |
| true_contact_flag | MuJoCo data | FT sensor ✅ | Oracle/label |
| contact_point | MuJoCo data | FT sensor ✅ | Oracle/label |
| contact_normal | MuJoCo data | FT sensor ✅ | Oracle/label |
| contact_force | MuJoCo data | FT sensor ✅ | Oracle/label |
| contact_mu | MuJoCo data | No ❌ | Oracle only |
| solref/solimp | MuJoCo model | No ❌ | Oracle only |

## Policy
1. ALLOWED: privileged_physics_oracle profile for upper-bound analysis
2. ALLOWED: as labels/probes in diagnostics
3. FORBIDDEN: as input to the main Paper 1 claim comparison
4. FORBIDDEN: in any profile used for primary encoder comparison

## Enforcement
- state_profiles.yaml must mark privileged features with use_in_main=false
- schema validation must check this constraint
