#!/usr/bin/env python3
"""
Fast Warm-Start vs Original ablation.
Uses exec_steps=20 and reduced samples for quicker turnaround.
Each trial ~5-15 min instead of 30-60 min.
"""
import subprocess, sys, os, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_DIR = Path("runs/warmstart_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS = [
    # (planner, config, label)
    ("WS_CEM", "WS_CEM_s075_exec20",  "WS_CEM"),
    ("CEM",    "CEM_s075_exec20",     "CEM"),
    ("WS_MPPI", "WS_MPPI_s075_exec20", "WS_MPPI"),
    ("MPPI",    "MPPI_s075_exec20",    "MPPI"),
]
FAMILIES = ["open", "blocking", "passage"]
SEEDS = [42, 137, 256]

def run_one(args):
    planner, config, family, seed, out_dir, label = args
    trial_id = f"{planner}_{config}_{family}_{seed}"
    cmd = [sys.executable, "scripts/run_single_clean_trial.py",
           planner, config, family, str(seed), str(out_dir)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        meta = None
        for line in reversed(r.stdout.strip().split("\n")):
            try: meta = json.loads(line); break
            except: pass
        if meta is None:
            return {"trial_id": trial_id, "status": "parse_error", "elapsed": elapsed}
        meta["trial_id"] = trial_id
        meta["label"] = label
        meta["elapsed"] = elapsed
        return meta
    except subprocess.TimeoutExpired:
        return {"trial_id": trial_id, "status": "timeout", "elapsed": time.time() - t0}
    except Exception as e:
        return {"trial_id": trial_id, "status": "error", "error": str(e), "elapsed": time.time() - t0}

def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 10
    tasks = []
    for planner, config, label in TRIALS:
        for family in FAMILIES:
            for seed in SEEDS:
                tasks.append((planner, config, family, seed, str(OUT_DIR), label))

    print(f"=== Fast Warm-Start Ablation ===")
    print(f"  Trials: {len(tasks)}, Workers: {workers}")
    print(f"  Output: {OUT_DIR}\n")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            done += 1
            m = f.result()
            s = "OK" if m.get("success") else m.get("status", "FAIL")
            print(f"  [{done}/{len(tasks)}] {s} {m.get('trial_id','')} dist={m.get('final_dist',-1):.4f} t={m.get('elapsed',0):.0f}s", flush=True)
            results.append(m)

    total = time.time() - t0
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    from collections import defaultdict
    by_combo = defaultdict(list)
    for r in results:
        by_combo[(r.get("label","?"), r.get("family","?"))].append(r)

    print(f"\n{'='*85}")
    print(f"SUMMARY ({total/3600:.1f}h)")
    print(f"{'='*85}")
    print(f"{'Config':<12} {'Scen':<10} {'N':>2} {'Succ':>5} {'Dist':>8} {'Contact':>8} {'FC_step':>8} {'PCLR':>6} {'Time':>6}")
    print("-" * 85)
    for label in [t[2] for t in TRIALS]:
        for family in FAMILIES:
            rs = by_combo.get((label, family), [])
            if not rs: continue
            n = len(rs)
            ok = sum(1 for r in rs if r.get("success"))
            d = sum(r.get("final_dist",999) for r in rs)/n
            c = sum(r.get("contact_rate",0) for r in rs)/n
            fc = [r["first_contact_step"] for r in rs if r.get("first_contact_step",-1)>=0]
            p = sum(r.get("post_contact_loss_rate",0) for r in rs)/n
            t_ = sum(r.get("elapsed",0) for r in rs)/n
            print(f"{label:<12} {family:<10} {n:>2} {ok:>5} {d:>8.4f} {c:>8.3f} {sum(fc)/len(fc) if fc else -1:>8.1f} {p:>6.3f} {t_:>5.0f}s")

    print(f"\n--- PAIRWISE ---")
    for ws, orig in [("WS_CEM","CEM"), ("WS_MPPI","MPPI")]:
        for family in FAMILIES:
            w = by_combo.get((ws, family), [])
            o = by_combo.get((orig, family), [])
            if not w or not o: continue
            wd = sum(r.get("final_dist",999) for r in w)/len(w)
            od = sum(r.get("final_dist",999) for r in o)/len(o)
            wc = sum(r.get("contact_rate",0) for r in w)/len(w)
            oc = sum(r.get("contact_rate",0) for r in o)/len(o)
            wt = sum(r.get("elapsed",0) for r in w)/len(w)
            ot = sum(r.get("elapsed",0) for r in o)/len(o)
            print(f"  {ws} vs {orig} @ {family}: dist {wd:.4f} vs {od:.4f} | contact {wc:.3f} vs {oc:.3f} | time {wt:.0f}s vs {ot:.0f}s ({ot/wt:.1f}x)")

    print(f"\nSaved: {OUT_DIR}/results.json")

if __name__ == "__main__":
    main()
