#!/usr/bin/env python3
"""
Warm-Start vs Original ablation sweep.

Compares warm-start planners against their original counterparts
on open / blocking / passage families, 3 trials each.

Usage: python scripts/run_warmstart_ablation.py [--workers N]
"""
import subprocess, sys, os, json, time, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_DIR = Path("runs/warmstart_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config pairs: (planner_name, config_name, label)
# label is for the summary table
TRIALS = [
    # ── Warm-Start CEM vs Original CEM ──
    ("WS_CEM", "WS_CEM_s05_H100",        "WS_CEM_s05"),
    ("CEM",    "CEM_s05_H100",            "CEM_s05"),
    ("WS_CEM", "WS_CEM_s05_H100_tight",   "WS_CEM_s05_tight"),
    # ── Warm-Start MPPI vs Original MPPI ──
    ("WS_MPPI", "WS_MPPI_s05_H100",       "WS_MPPI_s05"),
    ("MPPI",    "MPPI_s05_H100",          "MPPI_s05"),
    ("WS_MPPI", "WS_MPPI_s075_H100",      "WS_MPPI_s075"),
    ("MPPI",    "MPPI_s075_H100",         "MPPI_s075"),
    ("WS_MPPI", "WS_MPPI_s05_H100_tight", "WS_MPPI_s05_tight"),
]

FAMILIES = ["open", "blocking", "passage"]
SEEDS = [42, 137, 256]  # 3 trials each


def run_one_trial(args):
    planner, config, family, seed, out_dir, label = args
    trial_id = f"{planner}_{config}_{family}_{seed}"
    cmd = [
        sys.executable, "scripts/run_single_clean_trial.py",
        planner, config, family, str(seed), str(out_dir),
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0
        # Parse last line as JSON
        lines = result.stdout.strip().split("\n")
        meta = None
        for line in reversed(lines):
            try:
                meta = json.loads(line)
                break
            except:
                pass
        if meta is None:
            return {"trial_id": trial_id, "status": "parse_error", "stdout": result.stdout[-500:], "stderr": result.stderr[-500:], "elapsed": elapsed}
        meta["trial_id"] = trial_id
        meta["label"] = label
        meta["elapsed"] = elapsed
        return meta
    except subprocess.TimeoutExpired:
        return {"trial_id": trial_id, "status": "timeout", "elapsed": time.time() - t0}
    except Exception as e:
        return {"trial_id": trial_id, "status": "error", "error": str(e), "elapsed": time.time() - t0}


def main():
    workers = 26  # default
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        workers = int(sys.argv[idx + 1])

    # Build task list
    tasks = []
    for planner, config, label in TRIALS:
        for family in FAMILIES:
            for seed in SEEDS:
                tasks.append((planner, config, family, seed, str(OUT_DIR), label))

    print(f"=== Warm-Start Ablation Sweep ===")
    print(f"  Configs: {len(TRIALS)} ({len([t for t in TRIALS if t[0].startswith('WS')])} WS + {len([t for t in TRIALS if not t[0].startswith('WS')])} original)")
    print(f"  Families: {FAMILIES}")
    print(f"  Seeds per combo: {len(SEEDS)}")
    print(f"  Total trials: {len(tasks)}")
    print(f"  Workers: {workers}")
    print(f"  Output: {OUT_DIR}")
    print()

    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_one_trial, t): t for t in tasks}
        done = 0
        for future in as_completed(future_map):
            done += 1
            meta = future.result()
            status = "OK" if meta.get("success") else meta.get("status", "FAIL")
            dist = meta.get("final_dist", -1)
            elapsed = meta.get("elapsed", 0)
            print(f"  [{done}/{len(tasks)}] {status} {meta.get('trial_id','')} dist={dist:.4f} time={elapsed:.0f}s")
            results.append(meta)

    total_time = time.time() - t_start

    # Save raw results
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"SUMMARY ({total_time/3600:.1f}h)")
    print(f"{'='*90}")

    from collections import defaultdict
    by_combo = defaultdict(list)
    for r in results:
        label = r.get("label", r.get("planner_config_name", "?"))
        family = r.get("family", "?")
        by_combo[(label, family)].append(r)

    header = f"{'Config':<22} {'Scen':<10} {'N':>2} {'Succ':>5} {'Dist':>8} {'Contact':>8} {'FC_step':>8} {'PCLR':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))

    for label in [t[2] for t in TRIALS]:
        for family in FAMILIES:
            rs = by_combo.get((label, family), [])
            if not rs:
                continue
            n = len(rs)
            ok = sum(1 for r in rs if r.get("success"))
            dists = [r.get("final_dist", 999) for r in rs]
            contacts = [r.get("contact_rate", 0) for r in rs]
            fc_steps = [r.get("first_contact_step", -1) for r in rs if r.get("first_contact_step", -1) >= 0]
            pclr = [r.get("post_contact_loss_rate", 0) for r in rs]
            times = [r.get("elapsed", 0) for r in rs]
            avg_dist = sum(dists) / n
            avg_contact = sum(contacts) / n
            avg_fc = sum(fc_steps) / len(fc_steps) if fc_steps else -1
            avg_pclr = sum(pclr) / n
            avg_time = sum(times) / n
            print(f"{label:<22} {family:<10} {n:>2} {ok:>5} {avg_dist:>8.4f} {avg_contact:>8.3f} {avg_fc:>8.1f} {avg_pclr:>6.3f} {avg_time:>5.0f}s")

    # ── Pairwise comparison ──
    print(f"\n{'='*90}")
    print("PAIRWISE COMPARISON (Warm-Start vs Original)")
    print(f"{'='*90}")

    pairs = [
        ("WS_CEM_s05", "CEM_s05", "CEM s05"),
        ("WS_MPPI_s05", "MPPI_s05", "MPPI s05"),
        ("WS_MPPI_s075", "MPPI_s075", "MPPI s075"),
    ]
    for ws_label, orig_label, desc in pairs:
        print(f"\n--- {desc} ---")
        for family in FAMILIES:
            ws_rs = by_combo.get((ws_label, family), [])
            orig_rs = by_combo.get((orig_label, family), [])
            if not ws_rs or not orig_rs:
                continue
            ws_ok = sum(1 for r in ws_rs if r.get("success"))
            orig_ok = sum(1 for r in orig_rs if r.get("success"))
            ws_dist = sum(r.get("final_dist", 999) for r in ws_rs) / len(ws_rs)
            orig_dist = sum(r.get("final_dist", 999) for r in orig_rs) / len(orig_rs)
            ws_contact = sum(r.get("contact_rate", 0) for r in ws_rs) / len(ws_rs)
            orig_contact = sum(r.get("contact_rate", 0) for r in orig_rs) / len(orig_rs)
            ws_time = sum(r.get("elapsed", 0) for r in ws_rs) / len(ws_rs)
            orig_time = sum(r.get("elapsed", 0) for r in orig_rs) / len(orig_rs)
            speedup = orig_time / ws_time if ws_time > 0 else 0
            print(f"  {family:<10} succ: {ws_ok}/{len(ws_rs)} vs {orig_ok}/{len(orig_rs)}  "
                  f"dist: {ws_dist:.4f} vs {orig_dist:.4f}  "
                  f"contact: {ws_contact:.3f} vs {orig_contact:.3f}  "
                  f"time: {ws_time:.0f}s vs {orig_time:.0f}s ({speedup:.1f}x)")

    print(f"\nSaved: {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
