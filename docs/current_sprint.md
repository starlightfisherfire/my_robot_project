# Current Sprint: Paper 1 Minimal Runnable System
> Internal project-management note.  
> This file tracks the current implementation sprint and is not part of the public benchmark protocol.


## Sprint Goal

Build two minimal runnable loops:

1. Planner capacity loop:
   MuJoCo/toy environment → Oracle-MPC → success/failure metrics

2. Model forward loop:
   dummy structured state → normalizer → encoder → dynamics/subgoal heads → loss → backward

## Current Priority

1. Freeze repo with first commit.
2. Add MPC capacity check protocol.
3. Run dummy structured state through flat model.
4. Add object-centric and causal encoders only after flat path works.
5. Add MuJoCo oracle rollout only after the planner interface is clear.

## Not Doing Now

- C-JEPA
- RGB input
- language-conditioned world model
- diffusion policy
- topology analysis as main result
- real robot experiments

## Rule

Paper 1 first.  
Planner capacity first.  
Minimum runnable system first.