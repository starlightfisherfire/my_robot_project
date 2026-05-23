# Linked Paths — exp_20260517_layout_ood_state16_v0

**Updated:** 2026-05-17 19:27

| Resource | Path | Status |
|----------|------|--------|
| Config | `configs/experiments/layout_ood_state16_v0.yaml` | ✅ exists |
| Data | `data/sim/layout_ood_state16_v0/` | ✅ exists (10 episodes) |
| Runs | *(not yet created)* | TODO: will be `runs/layout_ood_state16_v0/` |
| Docs summary | `experiments/exp_20260517_layout_ood_state16_v0/summary.md` | ✅ this file |
| Original summary | `docs/layout_ood_state16_v0_summary.md` | ✅ original kept |
| Template file | `data/sim/metadata/reset_templates_obstacle_10family_v0.json` | ✅ 73 templates |
| Template generator | `src/data/template_generator.py` | ✅ |
| Collection script | `scripts/collect_layout_ood_state16.py` | ✅ |
| Model encoders | `src/models/encoders.py` | ✅ |
| RIGWorldModel | `src/models/rig_world.py` | ✅ |
| EpisodeWriter | `src/data/episode_writer.py` | ✅ |
| Training script | `scripts/train_high_level.py` | ❌ empty file |
| Eval script | `scripts/eval_policy.py` | ❌ empty file |
| Episode loader | `src/data/episode_loader.py` | ❌ empty file |

## Notes
- MPPI params need strengthening (horizon=8→12, samples=128→512, iterations=2→3)
- Training and eval scripts are placeholder files, need implementation
- Real robot data directory is empty (awaiting Paper 2 perception bridge)
