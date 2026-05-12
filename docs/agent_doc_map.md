# Agent Document Map

This file tells AI coding agents which project documents to read for each task type.
Do not read all docs blindly.
Do not treat every document as equally authoritative.

## 1. Core rule

- CLAUDE.md is only the tool entry point.
- This file is the routing map.
- Task-specific truth lives in the docs listed below.
- If documents conflict, follow the trust hierarchy.

## 2. Trust hierarchy

When documents conflict:

1. Current state / next action:
   - docs/current_sprint.md wins.

2. Code capability / file status / validation state:
   - docs/code_audit.md wins.

3. Known risks / do-not-misread items:
   - docs/known_issues.md wins.

4. Model architecture / token schema / state-action contracts:
   - docs/model_design.md wins.

5. Split rules / leakage rules:
   - docs/split_protocol.md wins.

6. Planner config policy:
   - docs/planner_config_policy.md wins.

7. Oracle-MPC capacity protocol:
   - docs/planner_capacity_protocol.md wins.

8. Paper claim / scope:
   - docs/paper_claim.md wins.

9. File location:
   - docs/file_topology_map.md is only a file index.
   - It is not current project truth.

10. New-agent context:
   - docs/ai_handoff.md is for quick handoff.
   - It does not override current_sprint.md or code_audit.md.

## 3. Always read first

For non-trivial tasks, read:

- docs/current_sprint.md
- docs/code_audit.md
- docs/known_issues.md

Use docs/ai_handoff.md only when:
- starting a fresh new agent session,
- the user asks for a project handoff,
- the agent lacks broad project context.

Do not read all docs by default.

## 4. Task-specific routing

### Oracle-MPC / planner / config23 / obstacle gate

Read:
- docs/current_sprint.md
- docs/code_audit.md
- docs/known_issues.md
- docs/planner_capacity_protocol.md
- docs/planner_config_policy.md
- docs/mpc_capacity_check.md
- docs/experiment_log.md

Use:
- docs/file_topology_map.md only to locate files.

### Dataset / reset templates / split

Read:
- docs/current_sprint.md
- docs/code_audit.md
- docs/split_protocol.md
- docs/experiment_conditions.md

If experiment_conditions.md is empty or placeholder, report that and continue with split_protocol.md.

### Learned dynamics / representation model

Read:
- docs/current_sprint.md
- docs/code_audit.md
- docs/model_design.md
- docs/paper_claim.md
- docs/split_protocol.md
- docs/experiment_gates.md

### Real robot / SO-101

Read:
- docs/current_sprint.md
- docs/known_issues.md
- docs/safety_ops.md
- docs/sim_real_parity.md
- docs/annotation_guideline.md

If safety_ops.md / sim_real_parity.md / annotation_guideline.md are empty placeholders, report that before proceeding.

### Paper writing

Read:
- docs/paper_claim.md
- docs/model_design.md
- docs/planner_capacity_protocol.md
- docs/experiment_log.md
- docs/current_sprint.md

### Documentation maintenance

Read:
- docs/agent_doc_map.md
- docs/current_sprint.md
- docs/code_audit.md
- docs/known_issues.md
- relevant target docs only

## 5. Editable / readonly policy

- Most docs are readonly context.
- Only files explicitly added by the user should be treated as editable.
- Do not ask the user to add all docs as editable.
- For code tasks: read docs as readonly, ask to add only the 2-5 target code files.
- For docs tasks: ask to add only the specific docs to be edited.
- Never modify files that were only read as context.
- If the chat contains many unintended editable files, stop and ask the user to clean the editable set.

## 6. Task start checklist

Before editing any file, output:

1. readonly docs read
2. editable files currently in chat
3. task type
4. files planned for edit
5. files that must not be touched
6. whether experiment execution is allowed
7. intended validation command

## 7. Update rules

After any meaningful task:

- update docs/current_sprint.md only if current stage or next steps changed
- update docs/code_audit.md only if code capability, validation status, or file role changed
- update docs/known_issues.md only if a new risk or pitfall was discovered
- update docs/experiment_log.md only if a new experiment was actually run
- update docs/file_topology_map.md only if file structure or file responsibility changed
- update docs/planner_capacity_protocol.md only if protocol or success criterion changed
- do not update all docs every time

## 8. Placeholder docs

Some docs may be placeholders. Do not delete them automatically.

If a placeholder doc is empty:
- do not treat it as missing project truth
- report that it is empty
- continue with the nearest canonical doc
- ask before deleting, archiving, or filling it

## 9. Staleness rule

If a doc has old current-status claims but current_sprint.md has newer status, trust current_sprint.md.

If a doc has old code-status claims but code_audit.md has newer status, trust code_audit.md.

If a doc contains project history, do not rewrite history unless explicitly asked.
