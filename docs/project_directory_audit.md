# Project Directory Audit

**Date:** 2026-05-17  
**Project:** ~/my_robot_project (Paper 1: Oracle-MPC / MPPI + learned world model)

---

## 1. Root-level loose files

| File | Size | Date | Action |
|------|------|------|--------|
| `boundary_refine_v1_20260511_213206_analysis_bundle.tar.gz` | 21 KB | May 11 | → archive/old_tarballs/ |
| `boundary_video_night2_20260512_001900_analysis_bundle.tar.gz` | 87 KB | May 12 | → archive/old_tarballs/ |
| `wide_overnight_v2_20260511_012735_analysis_bundle.tar.gz` | 1.6 MB | May 11 | → archive/old_tarballs/ |
| `cjepa_mpc_full_analysis.txt` | 77 KB | May 14 | → archive/root_junk/ (temp analysis dump) |
| `project_overview.pptx` | 43 KB | May 14 | → archive/old_exports/ |
| `0` | 0 B | — | → archive/root_junk/ (zero-byte junk) |
| `pyproject.toml` | 0 B | — | keep (required by project structure, zero-byte ok) |
| `requirements.txt` | 14 B | Apr 30 | keep (core project file) |
| `小龙虾启动命令.txt` | 1.1 KB | May 15 | → archive/root_junk/ (setup note, not core) |

## 2. Runs inventory

| Run Dir | Modified | Manifest | Logs | Videos | Size | Type | Status |
|---------|----------|----------|------|--------|------|------|--------|
| `planner_trio_300g_sweep_20260516_090909` | May 17 02:22 | 109 | 108 | 108 | 81 MB | CEM/MMCEM/MPPI trio sweep | **active** |
| `best_bh_sweep_20260516_182807` | May 16 20:58 | 11 | 10 | 10 | 7 MB | blocking_hard ablation | keep |
| `best_bh_sweep_20260516_140439` | May 16 14:04 | 0 | 0 | 0 | 1 MB | incomplete | archive_later |
| `heavy_pusher_250g_sweep_20260516_015635` | May 16 01:56 | 169 | 168 | 168 | 167 MB | heavy pusher sweep | keep |
| `heavy_pusher_700g_sweep_20260516_012126` | May 16 01:21 | 169 | 168 | 168 | 155 MB | heavy pusher sweep | keep |
| `heavy_pusher_500g_b1000_20260515_234649` | May 15 23:46 | 13 | 12 | 12 | 15 MB | heavy pusher config | keep |
| `heavy_pusher_500g_20260515_222952` | May 15 22:29 | 20 | 20 | 19 | 9 MB | heavy pusher config | keep |
| `best_config_ablation_speed005_to_100_20260515_181130` | May 15 22:26 | 225 | 224 | 54 | 27 MB | speed ablation | keep |
| `best_config_ablation_speed005_to_100_20260515_181059` | May 15 18:10 | 2 | 1 | 0 | 1 MB | incomplete ablation | archive_later |
| `horizon140_verify_20260515_164753` | May 15 16:47 | 20 | 20 | 19 | 13 MB | horizon verify | keep |
| `horizon140_verify_20260515_164557` | May 15 16:45 | 22 | 21 | 0 | 1 MB | incomplete verify | archive_later |
| `horizon140_sweep_20260515_141327` | May 15 14:13 | 19 | 18 | 18 | 17 MB | horizon sweep | keep |
| `dual_obstacle_speed_ablation_20260515_120428` | May 15 12:04 | 25 | 24 | 24 | 29 MB | dual obstacle ablation | keep |
| `speed_horizon_ablation_20260515_120253` | May 15 12:02 | 37 | 36 | 36 | 9 MB | speed×horizon ablation | keep |
| `mppi_blocking_t05_s02_b600` | May 16 12:23 | 0 | 3 | 3 | 3 MB | MPPI blocking test | keep |
| `mppi_blocking_h140` | May 16 11:50 | 0 | 3 | 3 | 3 MB | MPPI blocking test | keep |
| `mppi_blocking_demo` | May 16 11:13 | 1 | 3 | 3 | 3 MB | MPPI demo | keep |
| `mppi_param_sweep_checkpoint8h_v1_*` (×13) | May 17 13:36-52 | 1ea | 1ea | 1ea | ~1 MB ea | smoke test runs | archive_later (smoke artifacts) |
| `debug/` | May 15 03:50 | 0 | 0 | 9 | 90 MB | debug outputs | keep (reference) |
| `sweeps/` | May 11 21:32 | 0 | 0 | 0 | 20 MB | old sweep data | archive_later |
| `video_sweeps/` | May 12 11:27 | 0 | 0 | 0 | 10 MB | old video sweeps | archive_later |
| `obstacle_sweeps/` | May 13 18:07 | 0 | 0 | 0 | 1 MB | old obstacle sweeps | archive_later |
| `obstacle_videos/` | May 13 17:26 | 0 | 0 | 0 | 14 MB | old obstacle videos | archive_later |
| `videos/` | May 14 02:00 | 0 | 0 | 0 | 1 MB | old videos folder | archive_later |

## 3. Data inventory (data/sim/)

| Data Dir | Has Episodes | Has Metadata | Has NPZ | Related Experiment | Status |
|----------|-------------|-------------|---------|-------------------|--------|
| `metadata/` | — | — | — | template files (all sweeps) | **keep** |
| `mppi_sweep_v1/` | yes | yes (jsonl) | yes | mppi_param_sweep_checkpoint8h_v1 | **active** |
| `layout_ood_state16_v0/` | yes | yes | yes | layout OOD state16 experiment | **active** |
| `episodes/` | yes | — | yes | legacy episode data | keep |
| `qc/` | — | — | — | quality control | keep |
| `videos/` | — | — | — | legacy videos | archive_later |

## 4. Docs inventory

| Category | Files |
|----------|-------|
| **Design / Protocol** | `planner_capacity_protocol.md`, `planner_config_policy.md`, `model_design.md`, `split_protocol.md`, `paper_claim.md`, `paper1_oracle_levels.md`, `annotation_guideline.md` |
| **Experiment Summary** | `layout_ood_state16_v0_summary.md`, `best_config_ablation_speed005_to_100_summary.md`, `experiment_conditions.md`, `experiment_gates.md`, `experiment_log.md`, `mpc_capacity_check.md`, `success_rate_revise.md`, `sim_real_parity.md` |
| **Audit / Handoff** | `mppi_sweep_integration_audit.md`, `planner_trio_integration_audit.md`, `code_audit.md`, `debug_obstacle_cem_physics_audit_snapshot.md`, `ai_handoff.md`, `agent_doc_map.md`, `current_sprint.md`, `file_topology_map.md`, `known_issues.md` |
| **Resume / Personal** | `resume_boss.md`, `resume_brucewu.html`, `resume_brucewu.md` |
| **Misc** | `blocking_demo.mp4`, `safety_ops.md`, `papers/` |

## 5. Recommended future cleanup

- Clean up 13 smoke test runs in `runs/mppi_param_sweep_checkpoint8h_v1_20260517_13*` — keep only the latest successful one for reference
- Archive `runs/sweeps/`, `runs/video_sweeps/`, `runs/obstacle_sweeps/`, `runs/obstacle_videos/`, `runs/videos/` — old organizational directories
- Archive incomplete runs: `best_bh_sweep_20260516_140439`, `best_config_ablation_speed005_to_100_20260515_181059`, `horizon140_verify_20260515_164557`
- Archive `data/sim/videos/` (legacy)
- Consider splitting `docs/` into sub-categories (design/, experiments/, audits/) when there are >30 files
