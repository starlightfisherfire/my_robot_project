# MPC Capacity Check

## Purpose

Before evaluating learned representations, we first perform planner capacity checks using oracle MuJoCo dynamics.

The purpose is to ensure that failures under learned models are not merely caused by:

- insufficient MPC capacity;
- infeasible reset conditions;
- poorly designed action primitives;
- bad cost weights;
- too short planning horizon;
- contact dynamics that CEM cannot search through.

## Core Question

Can CEM-MPC solve the push-to-pose task when it is given simulator ground-truth dynamics?

If Oracle-MPC fails, then failures cannot be attributed to representation alone.

If Oracle-MPC succeeds but learned-model MPC fails, then the problem is more likely caused by learned dynamics, representation quality, or rollout error.

## Planner Ladder

1. Random policy
2. Hand-designed heuristic push policy
3. Oracle dynamics + CEM-MPC
4. Oracle subgoal + CEM-MPC
5. Learned flat model + CEM-MPC
6. Learned object-centric model + CEM-MPC
7. Learned causal-aware model + CEM-MPC

## Main Criterion

Oracle-MPC should solve simple sim ID push-to-pose before learned representations are evaluated.

Initial pass criterion:

- Sim ID success >= 60% for early development
- Sim ID success >= 80% for main experiments
- Layout OOD success should remain meaningfully above heuristic baseline

## Interpretation

If Oracle-MPC succeeds:
- the task is feasible under current action primitives and cost function;
- learned model failures are meaningful.

If Oracle-MPC fails:
- fix planner horizon, action primitives, cost function, reset difficulty, or environment design before training representation models.