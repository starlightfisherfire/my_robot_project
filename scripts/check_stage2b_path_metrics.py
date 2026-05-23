#!/usr/bin/env python3
"""Stage 2B Gate 3: Verify path metrics include initial positions."""
from __future__ import annotations

import csv, sys, argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to manifest.csv')
    parser.add_argument('--data-dir', help='Path to run directory (for reference)')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: manifest is empty")
        sys.exit(1)

    print(f"Checking {len(rows)} rows from {manifest_path}")
    print(f"Gate 3: Path Metrics Initial Point Check")
    print("=" * 60)

    all_pass = True
    for i, r in enumerate(rows):
        cfg = r.get('config', f'row{i}')
        total_env_steps = int(float(r.get('total_env_steps', -1)))
        ee_count = int(float(r.get('ee_positions_count', -1)))
        obj_count = int(float(r.get('object_positions_count', -1)))
        path_init = r.get('path_includes_initial_position', 'false')

        expected = total_env_steps + 1
        ee_ok = ee_count == expected
        obj_ok = obj_count == expected
        init_ok = str(path_init).lower() in ('true', '1', 'yes')

        print(f"\nRow {i}: {cfg}")
        print(f"  total_env_steps:          {total_env_steps}")
        print(f"  ee_positions_count:       {ee_count} (expected {expected}) {'✅' if ee_ok else '❌'}")
        print(f"  object_positions_count:   {obj_count} (expected {expected}) {'✅' if obj_ok else '❌'}")
        print(f"  path_includes_initial_position: {path_init} {'✅' if init_ok else '❌'}")

        if not ee_ok:
            print(f"  FAIL: ee_positions_count ({ee_count}) != total_env_steps+1 ({expected})")
            all_pass = False
        if not obj_ok:
            print(f"  FAIL: object_positions_count ({obj_count}) != total_env_steps+1 ({expected})")
            all_pass = False
        if not init_ok:
            print(f"  FAIL: path_includes_initial_position is not true")
            all_pass = False

    print(f"\n{'='*60}")
    print(f"Gate 3: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
