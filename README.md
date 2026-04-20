# My Robot Project

This project studies whether causality-aware object-level high-level representations improve structural OOD generalization in single-arm push-to-pose tasks.

## Core Idea

- Task: single-arm push-to-pose
- Main OOD: layout OOD
- Secondary OOD: shape OOD
- Executor: fixed CEM-MPC
- Main comparison:
  1. Flat non-causal high-level representation
  2. Object-centric non-causal high-level representation
  3. Causality-aware object-level representation

## Evidence Strategy

- MuJoCo simulation: mechanism identification, ablation, OOD protocol
- SO-101 real robot: external validity and real-world OOD validation

## Project Structure

- docs/: human-readable protocols and paper claim
- configs/: machine-readable experiment configs
- src/: reusable implementation
- scripts/: executable entry points
- data/: sim and real robot episodes
- runs/: experiment outputs
- papers/: paper draft
- benchmark/: benchmark documentation
