# Paper Claim

## Core Claim

Causality-aware object-level high-level representations improve structural OOD generalization under the same MPC planner.

## Main Task

Single-arm push-to-pose / planar pushing.

## Primary OOD Setting

Layout OOD.

The agent is trained on in-distribution layouts such as open-space and mild-offset pushing, and evaluated on held-out structural layout interventions such as blocking, narrow passage, and edge-goal settings.

## Secondary OOD Setting

Shape OOD.

The agent is trained on T-shaped objects and evaluated on held-out object shapes such as L-shaped objects.

## Main Comparison

We compare three high-level representation families under the same executor:

1. Flat non-causal high-level representation
2. Object-centric non-causal high-level representation
3. Causality-aware object-level high-level representation

## Fixed Executor

CEM-MPC.

The planner is held fixed across all methods. Any performance difference should primarily come from the high-level representation and learned predictive interface, not from changing the executor.

## Main Scientific Question

Does causality-aware object-level representation reduce structural OOD degradation compared with flat and object-centric non-causal representations?

## Evidence Chain

Simulation is used for:

- mechanism identification
- controlled structural interventions
- ablation
- representation comparison
- OOD gap measurement

Real robot experiments are used for:

- external validity
- real-world OOD validation
- sim-to-real transfer analysis

## Real Robot Protocols

We report two real-robot protocols.

### Protocol A: Zero-shot Sim-to-Real Transfer

Models are trained only on simulation ID data and directly evaluated on real robot ID and OOD splits.

This tests both:

- zero-shot sim-to-real transfer on real ID layouts
- zero-shot sim-to-real plus structural OOD on real OOD layouts and shapes

### Protocol B: Real-ID Adapted OOD Generalization

Models are first trained on simulation ID data and then adapted using a small real robot ID adaptation set.

The real adaptation set only contains:

- T-shaped objects
- open-space and mild-offset ID layouts

It does not contain:

- blocking layouts
- narrow-passage layouts
- edge-goal layouts
- L-shaped objects

Therefore, real-ID adaptation does not leak layout OOD or shape OOD test information.

## Main Paper Scope

Paper 1 focuses on:

- structured object-state input
- fixed CEM-MPC
- layout OOD as the primary setting
- shape OOD as the secondary setting
- MuJoCo simulation as the main mechanism testbed
- SO-101 real robot as the external validation platform

C-JEPA, Diffusion Policy, VLM/VLA, and LLM+graph MVP are auxiliary or future extensions, not the main scope of Paper 1.
