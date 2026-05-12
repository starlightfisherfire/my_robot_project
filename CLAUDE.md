# Claude Code Entry

This file is a tool-specific entry point for Claude Code.
It is not the full project memory.
Project facts, current status, file status, protocols, and interface contracts are routed through docs/agent_doc_map.md.

## 0. First action

Before any non-trivial task:
1. Read docs/agent_doc_map.md first.
2. Follow its task-specific routing.
3. Do not read all docs blindly.
4. Do not treat CLAUDE.md as the project truth source.
5. Do not treat readonly docs as editable.

## 1. Role

Claude Code is a repo-local engineering assistant.

Claude may:
- inspect files
- find references
- make small approved code changes
- run explicitly approved smoke tests
- summarize diffs
- help fix tracebacks, imports, and interface errors

Claude must not:
- decide research direction
- change the Paper 1 claim
- expand project scope
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

## 2. Hard boundaries

Claude must ask before changing:
- token schema
- state dimension
- action_dim
- CEMMPC API
- reset template schema
- metadata schema
- split definitions
- normalizer rules
- planner semantics
- loss semantics
- model semantics
- cost function semantics

Claude must not:
- refactor working code for style reasons
- rename, move, or delete files unless explicitly asked
- expand into RGB, C-JEPA, Diffusion Policy, Flow Policy, VLM/VLA, LLM planning, topology diagnostics, or real robot integration unless explicitly asked
- modify src/ / scripts/ / configs/ / data/ unless the task explicitly allows it

## 3. Experiment execution rules

Default allowed commands:
- grep
- sed
- head
- tail
- git status
- git diff
- git diff --stat
- python -m py_compile

Default forbidden commands unless explicitly authorized:
- MuJoCo runs
- MPC evaluation
- render
- sweep
- long-running experiments
- background processes
- nohup
- tmux-launched experiments
- subprocess-launched experiments
- scripts/check_mpc_capacity.py
- scripts/render_closed_loop_rollout.py
- scripts/run_c23_strictstop_eval.py
- scripts/run_wide_mpc_sweep_with_best_video.py

## 4. Git and safety rules

Do not run unless explicitly asked:
- git commit
- git push
- git reset --hard
- git clean
- git checkout .
- git branch -D
- gh pr create

Never access, print, edit, or commit:
- .env
- API keys
- credentials
- SSH keys
- tokens
- private secrets
- raw private real-robot data

Do not commit generated artifacts unless explicitly asked:
- runs/
- __pycache__/
- *.pyc
- large videos
- checkpoints
- raw datasets
- temporary debug outputs

## 5. Before editing, report

Before editing any file, output:
1. docs read
2. editable files currently in chat
3. task type
4. files planned for edit
5. files that must not be touched
6. whether experiment execution is allowed
7. intended validation command

## 6. After work, report

After work, report:
- Goal
- Files inspected
- Files changed
- Test command
- Test result
- Diff summary
- Known limitations
- Recommended next step
- Recommended commit message

If no test was run, say explicitly: "No test was run."

## 7. Final reminder

Preserve the scientific experiment.
Minimize changes.
Show diffs.
Do not commit.

## 8. Compact instructions

When compacting this project, preserve:
- Current sprint gate and next action
- Exact modified file paths
- Key parameters and defaults
- Test commands and outputs
- Known bugs and unresolved questions
- Decisions about pusher geometry, cost, planner, and success metrics

Do not preserve:
- Long exploratory discussion
- Repeated failed attempts
- Old superseded plans
