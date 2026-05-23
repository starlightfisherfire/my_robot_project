#!/usr/bin/env python3
"""Gate 3 & 4: Check path metrics and segment consistency for Stage 2C smoke manifest."""
import csv, sys, argparse
from pathlib import Path

def sf(r, key, default=0.0):
    try:
        v = r.get(key, default)
        if v == '' or v is None:
            return default
        return float(v)
    except (ValueError, TypeError):
        return default

def sb(r, key):
    v = str(r.get(key,'false')).lower()
    return v in ('true','1','yes','True')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to manifest.csv')
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

    print(f"Checking {len(rows)} rows in {manifest_path}")

    gate3_pass = True
    gate4_pass = True
    failures = []

    # Gate 3: Path metrics initial point
    print("\n=== Gate 3: Path Metrics Initial Point ===")
    for i, r in enumerate(rows):
        row_id = r.get('config', f'row_{i}')

        # Check path_includes_initial_position
        if not sb(r, 'path_includes_initial_position'):
            failures.append(f"  Gate 3 FAIL [{row_id}]: path_includes_initial_position != true")
            gate3_pass = False

        # Check ee_positions_count == total_env_steps + 1
        ee_cnt = sf(r, 'ee_positions_count', -1)
        total_steps = sf(r, 'total_env_steps', -1)
        if total_steps >= 0 and ee_cnt != total_steps + 1:
            failures.append(f"  Gate 3 FAIL [{row_id}]: ee_positions_count={ee_cnt} != total_env_steps+1={int(total_steps)+1}")
            gate3_pass = False

        # Check object_positions_count == total_env_steps + 1
        obj_cnt = sf(r, 'object_positions_count', -1)
        if total_steps >= 0 and obj_cnt != total_steps + 1:
            failures.append(f"  Gate 3 FAIL [{row_id}]: object_positions_count={obj_cnt} != total_env_steps+1={int(total_steps)+1}")
            gate3_pass = False

        # Check wasted_motion_ratio_capped <= 100
        wcap = sf(r, 'wasted_motion_ratio_capped', -1)
        if wcap > 100:
            failures.append(f"  Gate 3 FAIL [{row_id}]: wasted_motion_ratio_capped={wcap} > 100")
            gate3_pass = False

        # Check ee_path_length_m >= 0
        ee_path = sf(r, 'ee_path_length_m', -1)
        if ee_path < 0:
            failures.append(f"  Gate 3 FAIL [{row_id}]: ee_path_length_m={ee_path} < 0")
            gate3_pass = False

        # Check object_path_length_m >= 0
        obj_path = sf(r, 'object_path_length_m', -1)
        if obj_path < 0:
            failures.append(f"  Gate 3 FAIL [{row_id}]: object_path_length_m={obj_path} < 0")
            gate3_pass = False

    if gate3_pass:
        print("  ✅ Gate 3 PASS")
    else:
        print("  ❌ Gate 3 FAIL")

    # Gate 4: Segment metrics consistency
    print("\n=== Gate 4: Segment Metrics Consistency ===")
    for i, r in enumerate(rows):
        row_id = r.get('config', f'row_{i}')

        # Check segment path sums
        ee_total = sf(r, 'ee_path_length_m', 0)
        ee_early = sf(r, 'early_ee_path_length_m', 0)
        ee_middle = sf(r, 'middle_ee_path_length_m', 0)
        ee_late = sf(r, 'late_ee_path_length_m', 0)
        ee_sum = ee_early + ee_middle + ee_late
        if ee_total > 0 and abs(ee_sum - ee_total) > max(1e-5, 0.01 * ee_total):
            failures.append(f"  Gate 4 FAIL [{row_id}]: segment EE path sum={ee_sum:.6f} != total={ee_total:.6f}")
            gate4_pass = False

        # Check segment progress sums
        prog_total = sf(r, 'total_progress_m', 0)
        prog_early = sf(r, 'early_progress_m', 0)
        prog_middle = sf(r, 'middle_progress_m', 0)
        prog_late = sf(r, 'late_progress_m', 0)
        prog_sum = prog_early + prog_middle + prog_late
        if abs(prog_total) > 1e-6 and abs(prog_sum - prog_total) > max(1e-5, 0.01 * abs(prog_total)):
            failures.append(f"  Gate 4 FAIL [{row_id}]: segment progress sum={prog_sum:.6f} != total={prog_total:.6f}")
            gate4_pass = False

        # Check progress efficiency fields are finite
        for k in ['early_progress_efficiency_ee', 'middle_progress_efficiency_ee', 'late_progress_efficiency_ee']:
            v = sf(r, k, None)
            if v is not None and not (v == v):  # NaN check
                failures.append(f"  Gate 4 FAIL [{row_id}]: {k} is NaN")
                gate4_pass = False

        # Check boolean flags
        for k in ['late_breakthrough_flag', 'front_loaded_wander_flag', 'meaningless_exploration_flag']:
            v = str(r.get(k, '')).lower()
            if v and v not in ('true', 'false', '1', '0', 'yes', 'no', ''):
                failures.append(f"  Gate 4 FAIL [{row_id}]: {k}={v} is not boolean-like")
                gate4_pass = False

    if gate4_pass:
        print("  ✅ Gate 4 PASS")
    else:
        print("  ❌ Gate 4 FAIL")

    # Print all failures
    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(f)

    # Final decision
    print("\n=== Final Decision ===")
    if gate3_pass and gate4_pass:
        print("ALL GATES PASS: READY_TO_RUN_FULL_STAGE2C")
        sys.exit(0)
    else:
        print("GATE FAILURE: DO_NOT_RUN_FULL_STAGE2C")
        sys.exit(1)

if __name__ == "__main__":
    main()
