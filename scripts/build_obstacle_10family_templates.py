#!/usr/bin/env python3
"""Phase 0.1: Build unified 10-family template file for MPPI sweep.

Scans all existing reset_templates*.json in data/sim/metadata/, merges families,
renames where needed (with provenance), and outputs a unified 10-family file.

Target families (10):
  1.  open                         (10 templates, 0 obstacles)
  2.  blocking_easy                (10 templates, 1 obstacle)
  3.  blocking_medium              (10 templates, 1 obstacle)
  4.  blocking_hard                (10 templates, 1 obstacle)
  5.  passage_direct_wide          (10 templates, 2 obstacles)
  6.  passage_direct_medium        (10 templates, 2 obstacles)
  7.  passage_direct_narrow        (10 templates, 2 obstacles)
  8.  passage_bypass_wide          (1 template,  2 obstacles)
  9.  passage_bypass_medium        (1 template,  2 obstacles)
  10. passage_bypass_narrow        (1 template,  2 obstacles)

Total expected: 73 templates
"""
from __future__ import annotations

import argparse, csv, json, os, sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "sim" / "metadata"
OUTPUT_FILE = METADATA_DIR / "reset_templates_obstacle_10family_v0.json"
INVENTORY_FILE = METADATA_DIR / "reset_templates_obstacle_10family_v0_inventory.csv"
PROVENANCE_FILE = METADATA_DIR / "reset_templates_obstacle_10family_v0_provenance.md"

# Family name mapping: old_name -> new_name (direct passage only)
FAMILY_RENAME = {
    "passage_wide": "passage_direct_wide",
    "passage_medium": "passage_direct_medium",
    "passage_hard_5cm": "passage_direct_narrow",
    "open_space": "open",
}

# Target families with expected counts
TARGET_FAMILIES: dict[str, dict[str, Any]] = OrderedDict([
    ("open",                      {"count": 10, "type": "open",            "obstacle_count": 0}),
    ("blocking_easy",             {"count": 10, "type": "blocking",        "obstacle_count": 1}),
    ("blocking_medium",           {"count": 10, "type": "blocking",        "obstacle_count": 1}),
    ("blocking_hard",             {"count": 10, "type": "blocking",        "obstacle_count": 1}),
    ("passage_direct_wide",       {"count": 10, "type": "passage_direct",  "obstacle_count": 2}),
    ("passage_direct_medium",     {"count": 10, "type": "passage_direct",  "obstacle_count": 2}),
    ("passage_direct_narrow",     {"count": 10, "type": "passage_direct",  "obstacle_count": 2}),
    ("passage_bypass_wide",       {"count": 1,  "type": "passage_bypass",  "obstacle_count": 2}),
    ("passage_bypass_medium",     {"count": 1,  "type": "passage_bypass",  "obstacle_count": 2}),
    ("passage_bypass_narrow",     {"count": 1,  "type": "passage_bypass",  "obstacle_count": 2}),
])

TOTAL_EXPECTED = sum(f["count"] for f in TARGET_FAMILIES.values())  # 73


def scan_template_files() -> list[Path]:
    """Find all reset_templates*.json files, excluding the 10-family output."""
    files = sorted(METADATA_DIR.glob("reset_templates*.json"))
    return [f for f in files if f.name != OUTPUT_FILE.name]


def load_all_templates(files: list[Path]) -> list[dict]:
    all_templates = []
    for fp in files:
        with open(fp) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for t in data:
                t["_source_file"] = fp.name
                all_templates.append(t)
        elif isinstance(data, dict):
            data["_source_file"] = fp.name
            all_templates.append(data)
    return all_templates


def normalize_family(template: dict) -> str:
    """Return canonical family name, applying rename map."""
    family = template.get("layout_family", "")
    return FAMILY_RENAME.get(family, family)


def adapt_open_template(t: dict, index: int) -> dict:
    """Adapt an open_space template to test-split open family."""
    new = dict(t)
    new["split"] = "test_sim_layout_ood_open"
    new["layout_family"] = "open"
    new["reset_template_id"] = (
        f"test_sim_layout_ood_open__open__{t.get('shape_family','T_shape')}__0000{index:02d}"
    )
    new["obstacles"] = []
    new["_source_file"] = t.get("_source_file", "reset_templates_v0.json")
    new["_original_family"] = "open_space"
    new["_original_template_id"] = t["reset_template_id"]
    new["_renamed"] = True
    return new


def enrich_template_metadata(template: dict, family: str, family_meta: dict) -> dict:
    """Add or normalize metadata fields per Section D.5."""
    enriched = dict(template)
    enriched["family"] = family
    enriched["type"] = family_meta["type"]

    # obstacle_count
    obstacles = template.get("obstacles", [])
    enriched["obstacle_count"] = len(obstacles)

    # Passage gap fields
    if "passage_gap" not in enriched:
        enriched["passage_gap"] = template.get("passage_gap", None)
    if "effective_passage_gap" not in enriched:
        enriched["effective_passage_gap"] = template.get("effective_passage_gap",
                                                          template.get("passage_gap", None))

    # is_direct / is_bypass
    enriched["is_direct"] = "direct" in family
    enriched["is_bypass"] = "bypass" in family

    # Obstacle size (first obstacle)
    if obstacles:
        enriched["obstacle_size_x"] = obstacles[0].get("size_x",
                                                         enriched.get("obstacle_size_x", 0))
        enriched["obstacle_size_y"] = obstacles[0].get("size_y",
                                                         enriched.get("obstacle_size_y", 0))

    # Provenance fields
    enriched["template_source_file"] = template.get("_source_file", "")
    if "_original_template_id" not in enriched:
        enriched["original_template_id"] = template["reset_template_id"]

    # Clean internal fields
    for key in ["_source_file", "_original_family", "_renamed"]:
        enriched.pop(key, None)

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Build 10-family template file")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    provenance_log: list[str] = [
        "# Template Merge Provenance",
        f"Generated by: {Path(__file__).name}",
        "",
        "## Source files scanned",
    ]

    # 1. Scan and load
    files = scan_template_files()
    provenance_log.append("")
    for f in files:
        provenance_log.append(f"- {f.name}")

    all_templates = load_all_templates(files)
    provenance_log.append(f"\nTotal templates scanned: {len(all_templates)}")

    # 2. Group by normalized family
    by_family: dict[str, list[dict]] = defaultdict(list)
    for t in all_templates:
        family = normalize_family(t)
        if family in TARGET_FAMILIES:
            by_family[family].append(t)

    # 3. Check for missing families
    missing = [f for f in TARGET_FAMILIES if f not in by_family or len(by_family[f]) == 0]
    if missing:
        print(f"ERROR: Missing families with no source templates: {missing}")
        print("Cannot proceed. Run inventory check first.")
        sys.exit(1)

    # 4. Build output
    output_templates: list[dict] = []
    family_summary: dict[str, dict] = {}
    provenance_log.append("\n## Family merge decisions")

    for family, meta in TARGET_FAMILIES.items():
        sources = by_family[family]
        expected = meta["count"]

        # All open-family templates need adapt (layout_family → "open", split → test)
        if family == "open":
            sources = [adapt_open_template(t, i) for i, t in enumerate(sources[:expected])]
            provenance_log.append(
                f"\n### {family}: adapted {len(sources)} from open_space → open (train split → test split)"
            )
            # Enrich and add directly, skip general enrichment below
            enriched = [enrich_template_metadata(t, family, meta) for t in sources]
            output_templates.extend(enriched)
            family_summary[family] = {
                "expected": expected, "actual": len(enriched),
                "type": meta["type"], "obstacle_count": meta["obstacle_count"],
            }
            continue
        else:
            rename_note = ""
            if family in FAMILY_RENAME.values():
                old_names = [k for k, v in FAMILY_RENAME.items() if v == family]
                rename_note = f" (renamed from {old_names})"
            provenance_log.append(f"\n### {family}: {len(sources)} available{rename_note}")

        # If we have more than needed, take first N (prefer test split)
        if len(sources) > expected:
            test_sources = [s for s in sources if "test" in s.get("split", "")]
            train_sources = [s for s in sources if "train" in s.get("split", "")]
            sources = (test_sources + train_sources)[:expected]
            provenance_log.append(f"  Selected {len(sources)} templates (test={len(test_sources)}, train={len(train_sources)})")

        # If fewer than expected, warn but proceed
        if len(sources) < expected:
            provenance_log.append(f"  ⚠ WARNING: only {len(sources)}/{expected} templates available")

        # Enrich and add
        enriched = [enrich_template_metadata(t, family, meta) for t in sources]
        output_templates.extend(enriched)

        family_summary[family] = {
            "expected": expected,
            "actual": len(enriched),
            "type": meta["type"],
            "obstacle_count": meta["obstacle_count"],
        }

    # 5. Validate
    total = len(output_templates)
    provenance_log.append(f"\n## Summary")
    provenance_log.append(f"Total templates: {total}/{TOTAL_EXPECTED}")
    for family, summary in family_summary.items():
        status = "✅" if summary["actual"] >= summary["expected"] else "⚠️"
        provenance_log.append(f"  {status} {family}: {summary['actual']}/{summary['expected']} "
                             f"(type={summary['type']}, obstacles={summary['obstacle_count']})")

    if total < TOTAL_EXPECTED:
        provenance_log.append(f"\n⚠ WARNING: Total {total} < expected {TOTAL_EXPECTED}")

    # 6. Output
    if args.dry_run:
        print("\n".join(provenance_log))
        print(f"\nDry run — no files written.")
        return

    # Write template JSON
    with open(OUTPUT_FILE, "w") as fh:
        json.dump(output_templates, fh, indent=2, ensure_ascii=False)
    print(f"✅ Written {total} templates to {OUTPUT_FILE}")

    # Write inventory CSV
    with open(INVENTORY_FILE, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "family", "type", "obstacle_count", "passage_gap", "effective_passage_gap",
            "is_direct", "is_bypass", "split", "template_index", "reset_template_id",
            "source_file", "original_template_id",
        ])
        for family, meta in TARGET_FAMILIES.items():
            family_templates = [t for t in output_templates if t["layout_family"] == family]
            for idx, t in enumerate(family_templates):
                writer.writerow([
                    family, meta["type"], len(t.get("obstacles", [])),
                    t.get("passage_gap", ""), t.get("effective_passage_gap", ""),
                    t.get("is_direct", False), t.get("is_bypass", False),
                    t.get("split", ""), idx, t["reset_template_id"],
                    t.get("template_source_file", ""), t.get("original_template_id", ""),
                ])
    print(f"✅ Written inventory to {INVENTORY_FILE}")

    # Write provenance
    with open(PROVENANCE_FILE, "w") as fh:
        fh.write("\n".join(provenance_log) + "\n")
    print(f"✅ Written provenance to {PROVENANCE_FILE}")

    # Final validation
    if total != TOTAL_EXPECTED:
        print(f"⚠ WARNING: Final count {total} != expected {TOTAL_EXPECTED}")
    else:
        print(f"✅ Count validation PASS: {total} == {TOTAL_EXPECTED}")


if __name__ == "__main__":
    main()
