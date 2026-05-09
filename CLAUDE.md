# Claude Code Entry

This file is a tool-specific entry point for Claude Code.

It is NOT the full project memory.

Before any non-trivial code change, read the source-of-truth documents:

- docs/ai_handoff.md
- docs/current_sprint.md
- docs/code_audit.md
- docs/project_topology.md

Use those documents as the project truth.

---

## 1. Role

Claude Code is a repo-local engineering assistant.

Claude may:

- inspect files
- find references
- make small approved code changes
- run approved smoke tests
- summarize diffs
- help fix tracebacks, imports, and interface errors

Claude must not:

- decide research direction
- change the Paper 1 claim
- expand the project scope
- redesign architecture without approval
- auto-commit or push
- silently change schemas or public interfaces

The human user owns:

- research direction
- experiment design
- architecture decisions
- final judgment
- git commit / push decisions
- whether to enter the next stage

---

## 2. Current Project Scope

This repository is for Paper 1:

> Causality-aware object-level high-level representations improve structural OOD generalization under the same MPC planner.

Current scope:

- structured object-state input
- flat / object-centric / causality-aware encoder comparison
- fixed CEM-MPC planner
- MuJoCo first
- SO-101 real robot later
- layout OOD primary
- shape OOD secondary

Current priority:

```text
MuJoCo oracle rollout
→ MuJoCo oracle-MPC capacity
→ sim data collection
→ learned high-level model
→ learned model + MPC
→ OOD gap

Do not expand into:

RGB input
C-JEPA implementation
Diffusion Policy
Flow Policy
VLM / VLA
LLM planning
topology diagnostics
real robot integration
full obstacle-enabled MuJoCo

unless the user explicitly asks.

3. Non-Negotiable Interface Contracts

Structured state input:

x: [B, H, N, D_raw] = [B, 6, 6, 16]

Token order:

0 end-effector
1 manipulated object
2 goal
3 obstacle 1
4 obstacle 2
5 obstacle 3

Raw token schema:

0  x
1  y
2  sin(theta)
3  cos(theta)
4  vx
5  vy
6  omega
7  size_x
8  size_y
9  shape_T
10 shape_L
11 shape_other
12 mass_norm
13 friction_norm
14 contact_flag
15 valid_flag

Action contract:

action: [B, 2]
action_sequence: [H_plan, 2]
action = [vx, vy]
range = [-1, 1]

Encoder output contract:

z: [B, 256]

RIGWorldModel output contract:

z: [B, 256]
pred_delta: [B, 3]
pred_subgoal: [B, 3]

Causality-aware optional slots:

z_stable: [B, 32]
z_dynamics: [B, 32]
z_affordance: [B, 32]
z_nuisance: [B, 32]

Pose convention:

[x, y, theta]

Angle errors must be wrapped with:

atan2(sin(error), cos(error))
4. Modification Rules

Before changing code, Claude should state:

goal
files to inspect
files to modify
expected input/output contract
smoke test command

Claude must ask before changing:

token schema
state dimension
action_dim
CEMMPC API
reset template schema
metadata schema
split definitions
normalizer rules
planner / loss / model semantics

Claude must keep diffs minimal.

Do not refactor working code for style reasons.

Do not rename, move, or delete files unless explicitly asked.

5. Testing Rules

Every core change needs a smoke test.

Use:

PYTHONPATH=. python scripts/debug_state_normalizer.py
PYTHONPATH=. python scripts/debug_encoder_variants.py
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_metadata_schema.py
PYTHONPATH=. python scripts/generate_reset_templates.py --num-per-split 20
PYTHONPATH=. python scripts/debug_reset_templates.py
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode state_sanity
PYTHONPATH=. python scripts/debug_cem_mpc_toy.py
PYTHONPATH=. python scripts/debug_oracle_rollout.py
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode toy_oracle_mpc --split train_sim_id --max-templates 20
PYTHONPATH=. python scripts/debug_mujoco_env.py
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py

A smoke test passing means:

interface / shape / simple semantic sanity passed

It does not mean:

paper claim proven
OOD generalization proven
real robot transfer proven
6. Data Leakage Rules

StateNormalizer must fit only on train or allowed adaptation data.

Never fit on:

test splits
OOD splits
real OOD splits

Do not use OOD test data for:

normalizer fitting
hyperparameter tuning
model selection
early stopping

Planner and cost settings must remain shared across flat / object / causal variants.

Do not tune CEM-MPC separately per encoder.

7. Git and Safety Rules

Claude must not run unless explicitly asked:

git commit
git push
git reset --hard
git clean
git checkout .
git branch -D
gh pr create

Claude may run safe inspection commands:

git status --short
git diff --stat
git diff
git log --oneline -5

Never access, print, edit, or commit:

.env
API keys
credentials
SSH keys
tokens
private secrets
raw private real-robot data

Do not commit generated artifacts unless explicitly asked:

runs/
__pycache__/
*.pyc
large videos
checkpoints
raw datasets
temporary debug outputs
8. Reporting Format

After work, report:

Goal:
Files inspected:
Files changed:
Input/output contract:
Test command:
Test result:
Diff summary:
Known limitations:
Recommended next step:
Recommended commit message:

If a test fails, report:

Failure command:
Traceback summary:
Likely cause:
Minimal proposed fix:
Files affected:

Do not hide uncertainty.

Do not claim success if tests were not run.

9. Final Reminder

This repository is a controlled scientific pipeline.

Preserve the experiment.

Minimize changes.

Run tests.

Show diffs.

Do not commit.