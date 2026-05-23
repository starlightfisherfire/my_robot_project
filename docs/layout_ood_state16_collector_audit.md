# Collector Audit — collect_layout_ood_state16.py

## Script Status

- File: `scripts/collect_layout_ood_state16.py` (475 lines)
- Exists: ✅
- Functional: Yes, collects live MuJoCo MPPI episodes

## Capabilities

| Feature | Supported |
|---------|-----------|
| Collect new episodes | ✅ (--smoke, --small-main, default) |
| Reorganize existing | ❌ |
| Source-root selection | ❌ (hardcoded output_dir) |
| Split config | ✅ (SMALL_MAIN_SPLITS dict) |
| Template families | open, blocking_easy/medium/hard, passage_direct_*, passage_bypass_* |
| Canonical state16 output | ✅ (uses EpisodeWriter) |
| Metadata preservation | ✅ (family, split, template_id, etc.) |

## Does it read existing data/sim/mppi_*?

No. It only collects new data via MuJoCo + MPPI.

## Can it convert compact rollout → state16?

No. It generates state16 from scratch using live MuJoCo simulation.

## Leakage Risk

Low. The script uses explicit split definitions (SMALL_MAIN_SPLITS) that separate train/val/test/OOD families.

## Recommendation

Use as-is for data collection. The script is ready for production use.
