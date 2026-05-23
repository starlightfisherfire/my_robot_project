# Planner Trio Integration Audit

Date: 2026-05-16
Scope: Can the new planner trio (CEM / MultimodalCEM / MPPI) be used in the existing MuJoCo Oracle-MPC obstacle sweep pipeline without modification?

## Verdict

**FAIL** — the new planners compile and have compatible interfaces, but the existing pipeline does NOT wire obstacle geometry into `rollout_cost()`, so `w_proximity` is dead code. Additionally, no `--planner-mode` CLI argument exists, so there is no way to select the new planners from the sweep script.

## Critical blockers

| # | Issue | Impact |
|---|-------|--------|
| 1 | `mujoco_oracle_rollout_cost()` never passes `obstacle_positions` / `obstacle_radii` to `rollout_cost()` | `w_proximity` is gated by `if obstacle_positions is not None` → **proximity cost is always 0**, regardless of weight value |
| 2 | `render_closed_loop_rollout_parallel_cem.py` does NOT use `make_planner()` — it directly instantiates `CEMMPC` (line 679) and has its own `parallel_cem_plan()` | No way to select MPPI or MultimodalCEM from CLI |
| 3 | No `--planner-mode` argument exists in `parse_args()` | Cannot sweep across planner modes |
| 4 | `cost_weights_dict` (line 598-609) does not include `w_proximity` key | Workers would use dataclass default (5.0) but it's moot since obstacle info is never passed |
| 5 | `evaluate_action_chunk_worker()` (line 175) calls `rollout_cost()` without obstacle args | Same as #1 but in the parallel path |

## Semantic risks

| # | Issue | Impact |
|---|-------|--------|
| A | Obstacles are **boxes** (`size_x`, `size_y`) but `obstacle_proximity_cost()` uses a single scalar radius per obstacle — pure circle approximation | For a box 0.04×0.08, any single radius either under-covers the long axis or over-covers the short axis. Effective margin is direction-dependent. |
| B | `MultimodalCEMMPC._make_lateral_means()` offsets only action dim index 1 (y-axis) | Assumes push direction is aligned with x-axis. If object→goal vector is diagonal or along y, the "left/right detour" seeds are wrong — they become "forward/backward" instead. |
| C | MPPI evaluates candidates serially: `[cost_fn(seq) for seq in samples]` | With horizon=80, num_samples=1024, num_iterations=5 → ~400k MuJoCo rollout steps per MPC step. At ~0.1ms/step that's ~40s per MPC step. Unusable without parallel eval. |
| D | `MultimodalCEMMPC` runs 3× full CEM (each with `num_samples`). With default 512 samples × 3 modes = 1536 total evaluations per iteration. | 3× slower than single CEM. Acceptable if `num_samples` is reduced per mode, but user must be aware. |
| E | `lateral_offset` is in normalized action space [-1, 1], not meters. A value of 0.5 means "half of max action" which maps to `0.5 * max_speed_mps * dt` per step — the actual spatial detour depends on horizon and speed. | Not a bug, but the relationship between `lateral_offset` and physical detour distance is non-obvious. |

## Minimal fixes needed before sweep

1. **Wire obstacle geometry into rollout_cost** — in `mujoco_oracle_rollout_cost()` and `evaluate_action_chunk_worker()`:
   - Extract obstacle positions/radii from the template dict
   - Convert box `(size_x, size_y)` to an enclosing radius: `r = sqrt(size_x² + size_y²) / 2`
   - Pass `obstacle_positions`, `obstacle_radii`, `proximity_margin` to `rollout_cost()`

2. **Add `--planner-mode` to CLI** — choices: `cem`, `multimodal_cem`, `mppi`
   - Route to `make_planner()` or equivalent instantiation
   - For MPPI: reuse the existing `parallel_cem_plan()` pattern (ProcessPoolExecutor) or accept serial-only for small sample counts

3. **Add `w_proximity` to `cost_weights_dict`** in both the main loop and worker initializer

4. **Fix lateral_offset to be goal-relative** (optional but recommended):
   - Compute pusher→goal direction vector
   - Generate lateral means perpendicular to that vector, not hardcoded to y-axis

5. **Add parallel eval to MPPI** (optional but needed for real sweeps):
   - Either wrap MPPI's cost eval in the same ProcessPoolExecutor pattern
   - Or accept that MPPI sweeps use fewer samples (e.g. 256) for feasibility

## Recommended smoke commands

All commands assume `MUJOCO_GL=egl PYTHONPATH=.` prefix. These are hypothetical — they require fixes #1-3 above to actually exercise the new features.

### CEM (baseline + proximity cost)

```bash
# After fix: obstacle info wired, w_proximity active
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
  --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
  --split test_sim_layout_ood_blocking_easy \
  --template-index 0 \
  --planner-mode cem \
  --horizon 80 --num-samples 1024 --num-iterations 5 \
  --max-speed-mps 0.05 \
  --out-video runs/debug/videos/smoke_cem_proximity.mp4
```

### MultimodalCEM

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
  --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
  --split test_sim_layout_ood_blocking_easy \
  --template-index 0 \
  --planner-mode multimodal_cem \
  --horizon 80 --num-samples 512 --num-iterations 5 \
  --lateral-offset 0.5 \
  --max-speed-mps 0.05 \
  --out-video runs/debug/videos/smoke_multimodal_cem.mp4
```

### MPPI

```bash
# Use fewer samples due to serial eval (until parallel MPPI is added)
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
  --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
  --split test_sim_layout_ood_blocking_easy \
  --template-index 0 \
  --planner-mode mppi \
  --horizon 80 --num-samples 256 --num-iterations 5 \
  --mppi-temperature 0.1 \
  --max-speed-mps 0.05 \
  --out-video runs/debug/videos/smoke_mppi.mp4
```

## Summary of answers to audit questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Script uses `make_planner()`? | **No** — directly instantiates `CEMMPC` |
| 2 | `--planner-mode` exists? | **No** |
| 3 | `obstacle_positions`/`obstacle_radii` passed to `rollout_cost()`? | **No** — neither serial nor parallel path |
| 4 | `w_proximity` effective? | **No** — dead code (gated by None check) |
| 5 | Proximity is circle-only for box obstacles? | **Yes** — single radius, no box geometry |
| 6 | `lateral_offset` is y-axis only? | **Yes** — hardcoded dim index 1 |
| 7 | MPPI uses parallel eval? | **No** — serial list comprehension. H=80, N=1024 will be very slow |
| 8 | Same `plan()` interface? | **Yes** — all three: `plan(cost_fn, init_mean, init_std) → (action, result)` |
| 9 | Breaking bugs? | No crash bugs, but **functional no-ops** (proximity never fires, no CLI path to new planners) |
