<<'EOF'
# Paper 1 Project Rules for Cline

This is a project-specific rule file for Cline.

For detailed rules on project scope, architecture, data splits, testing discipline, 
and code style, see **CLAUDE.md** — it is the authoritative source.

## Document Reading Protocol（启动任务前必读）

**第一步：读 docs/agent_doc_map.md**，按其指引决定读哪些文档。

Cline must read these source-of-truth documents before non-trivial changes:

- `docs/ai_handoff.md`
- `docs/current_sprint.md`
- `docs/code_audit.md`
- `docs/known_issues.md`

**不要默认读取所有 docs。** 按任务类型读取相关协议文档（见 `docs/agent_doc_map.md`）。

**Staleness rule：** 如果 `docs/file_topology_map.md` 与 `current_sprint.md` / `code_audit.md` 冲突，以后者为准。

Use those documents as the project memory. Do not duplicate or override them here.

## Experiment Execution Rules

**默认只允许：**
- `py_compile`、`grep`、`git diff`、静态检查、读取文件

**不允许（除非用户明确授权）：**
- 运行 MuJoCo / MPC / render / sweep / 长时间实验
- 启动后台进程（tmux / nohup / &）
- 修改 `src/` / `scripts/` / `configs/` / `data/`

## Role

Cline is the repo-local coding assistant.

Cline may:
- inspect files
- search references
- modify code
- run smoke tests
- summarize diffs
- help fix traceback / import / interface errors

Cline must not:
- decide research direction
- expand the Paper 1 scope
- change the paper claim
- redesign architecture without approval
- auto-commit unless explicitly asked

## Current Paper 1 Scope

This repo is for:

structured object-state robot pushing
flat / object-centric / causality-aware high-level representations
fixed CEM-MPC planner
MuJoCo + SO-101
layout OOD primary
shape OOD secondary

Current priority:

MuJoCo oracle rollout
→ MuJoCo oracle-MPC capacity
→ sim data collection
→ learned high-level model
→ learned model + MPC
→ OOD gap

## Forbidden Scope Expansion

Do not expand into these unless the user explicitly asks:

- RGB input
- C-JEPA implementation
- Diffusion Policy
- Flow Policy
- VLM / VLA
- LLM + graph planner

## Code Modification Rules

Before changing code:

1. State the goal.
2. State files to inspect.
3. State files to modify.
4. State expected input/output contract.
5. State the smoke test command.
6. Ask for approval if the change touches schemas, APIs, splits, or planner interfaces.

Do not silently change:

- token schema
- state dimension
- action_dim
- CEMMPC API
- reset template schema
- metadata schema
- split definitions
- normalizer rules

## Testing Rules

Every core change must have a smoke test.

Common commands:

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

A smoke test passing means interface / shape / simple semantic sanity passed.
It does not mean the paper claim is proven.

## Git Rules

Do not auto-commit.

After a tested change, report:

- goal
- files changed
- test command
- test result
- diff summary
- known limitations
- recommended commit message

Use one commit per tested capability.

## Security Rules

Never write API keys, tokens, or secrets into repo files.
Never commit .env, local credentials, API keys, or provider configs.

## Report Format

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
