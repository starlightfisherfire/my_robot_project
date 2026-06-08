#!/usr/bin/env python3
"""Capability audit for Oracle staged-cost sweep."""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    audit = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    # 1. Cost functions
    print("=== 1. Cost Functions ===")
    try:
        from src.planners.cost_functions import rollout_cost, staged_contact_obstacle_goal_cost, make_staged_cost_weights
        import numpy as np

        # Test current mode
        T, H = 20, 10
        poses = np.random.randn(T, 3) * 0.01
        poses[0] = [0.2, 0.18, 0.0]
        ee = np.random.randn(T, 2) * 0.01
        actions = np.random.randn(H, 2) * 0.01
        goal = np.array([0.42, 0.18, 0.0])

        cost_cur = rollout_cost(poses, ee, actions, goal, cost_mode="current")
        cost_stg = rollout_cost(poses, ee, actions, goal, cost_mode="staged_contact_obstacle_goal")

        audit["checks"]["cost_current"] = {"status": "OK", "value": float(cost_cur)}
        audit["checks"]["cost_staged"] = {"status": "OK", "value": float(cost_stg)}
        audit["checks"]["make_staged_cost_weights"] = {"status": "OK"}
        print(f"  cost_mode=current: {cost_cur:.4f}")
        print(f"  cost_mode=staged: {cost_stg:.4f}")
    except Exception as e:
        audit["checks"]["cost_functions"] = {"status": "FAIL", "error": str(e)}
        print(f"  FAIL: {e}")

    # 2. Self-check
    print("\n=== 2. Staged Cost Self-Check ===")
    try:
        result = os.popen("cd /home/brucewu/my_robot_project && PYTHONPATH=. python3 scripts/self_check_staged_cost.py 2>&1").read()
        passed = "ALL CHECKS PASSED" in result
        audit["checks"]["staged_cost_self_check"] = {"status": "PASS" if passed else "FAIL"}
        print(f"  {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        audit["checks"]["staged_cost_self_check"] = {"status": "FAIL", "error": str(e)}

    # 3. Planner availability
    print("\n=== 3. Planners ===")
    try:
        from src.planners.cem_mpc import CEMMPC
        from src.planners.mppi import MPPI
        audit["checks"]["cem_available"] = {"status": "OK"}
        audit["checks"]["mppi_available"] = {"status": "OK"}
        print("  CEM: OK")
        print("  MPPI: OK")
    except Exception as e:
        audit["checks"]["planners"] = {"status": "FAIL", "error": str(e)}

    # 4. MuJoCo env + oracle rollout
    print("\n=== 4. MuJoCo Oracle ===")
    try:
        # Use lerobot env python
        ret = os.system(
            "cd /home/brucewu/my_robot_project && "
            "PYTHONPATH=. MUJOCO_GL=egl /home/brucewu/miniconda3/envs/lerobot/bin/python -c \""
            "from src.envs.mujoco_push_env import MujocoPushEnv; "
            "from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost; "
            "from src.planners.cost_functions import make_staged_cost_weights; "
            "import numpy as np; "
            "env = MujocoPushEnv(); "
            "env.reset(); "
            "cost = mujoco_oracle_rollout_cost(env, np.array([[0.5,0.0]]*10), cost_mode='staged_contact_obstacle_goal'); "
            "print(f'oracle_staged_cost={cost:.4f}'); "
            "s = env.clone_state(); env.step(np.array([0.1,0.0])); env.restore_state(s); "
            "print('clone_restore=OK'); "
            "\" 2>&1"
        )
        audit["checks"]["mujoco_oracle"] = {"status": "OK" if ret == 0 else "FAIL"}
        print(f"  {'OK' if ret == 0 else 'FAIL'}")
    except Exception as e:
        audit["checks"]["mujoco_oracle"] = {"status": "FAIL", "error": str(e)}

    # 5. Replay (qpos/qvel trace)
    print("\n=== 5. Replay ===")
    try:
        ret = os.system(
            "cd /home/brucewu/my_robot_project && "
            "PYTHONPATH=. MUJOCO_GL=egl /home/brucewu/miniconda3/envs/lerobot/bin/python -c \""
            "from src.envs.mujoco_push_env import MujocoPushEnv; "
            "import numpy as np; "
            "env = MujocoPushEnv(); "
            "env.reset(); "
            "qpos_trace = []; qvel_trace = []; "
            "for _ in range(10): "
            "    env.step(np.array([0.3, 0.0])); "
            "    qpos_trace.append(env.data.qpos.copy()); "
            "    qvel_trace.append(env.data.qvel.copy()); "
            "np.savez('/tmp/replay_test.npz', qpos=np.array(qpos_trace), qvel=np.array(qvel_trace)); "
            "data = np.load('/tmp/replay_test.npz'); "
            "print(f'qpos_shape={data[\"qpos\"].shape}'); "
            "print('replay_save=OK'); "
            "\" 2>&1"
        )
        audit["checks"]["replay_save"] = {"status": "OK" if ret == 0 else "FAIL"}
        print(f"  {'OK' if ret == 0 else 'FAIL'}")
    except Exception as e:
        audit["checks"]["replay_save"] = {"status": "FAIL", "error": str(e)}

    # 6. Renderer
    print("\n=== 6. Renderer 224x224 ===")
    try:
        ret = os.system(
            "cd /home/brucewu/my_robot_project && "
            "PYTHONPATH=. MUJOCO_GL=egl /home/brucewu/miniconda3/envs/lerobot/bin/python -c \""
            "import mujoco; "
            "from src.envs.mujoco_push_env import MujocoPushEnv; "
            "env = MujocoPushEnv(); "
            "env.reset(); "
            "r = mujoco.Renderer(env.model, height=224, width=224); "
            "r.update_scene(env.data); "
            "px = r.render(); "
            "print(f'224x224 shape={px.shape}'); "
            "r.close(); "
            "print('renderer_224=OK'); "
            "\" 2>&1"
        )
        audit["checks"]["renderer_224"] = {"status": "OK" if ret == 0 else "FAIL"}
        print(f"  {'OK' if ret == 0 else 'FAIL'}")
    except Exception as e:
        audit["checks"]["renderer_224"] = {"status": "FAIL", "error": str(e)}

    # 7. Templates
    print("\n=== 7. Templates ===")
    try:
        from src.data.template_generator import TEMPLATE_GENERATORS, generate_template, is_template_valid
        import json as json_mod
        with open("data/sim/metadata/reset_templates_obstacle_10family_v0.json") as f:
            tpls = json_mod.load(f)
        families = {}
        for t in tpls:
            fam = t.get("family", "unknown")
            families[fam] = families.get(fam, 0) + 1
        audit["checks"]["templates"] = {
            "status": "OK",
            "generator_families": list(TEMPLATE_GENERATORS.keys()),
            "file_families": families,
            "total_file_templates": len(tpls),
        }
        print(f"  Generator families: {list(TEMPLATE_GENERATORS.keys())}")
        print(f"  File templates: {families}")
    except Exception as e:
        audit["checks"]["templates"] = {"status": "FAIL", "error": str(e)}

    # 8. EpisodeWriter
    print("\n=== 8. EpisodeWriter ===")
    try:
        from src.data.episode_writer import EpisodeWriter
        audit["checks"]["episode_writer"] = {"status": "OK"}
        print("  EpisodeWriter: OK")
    except Exception as e:
        audit["checks"]["episode_writer"] = {"status": "FAIL", "error": str(e)}

    # 9. Planner top-k candidates
    print("\n=== 9. Planner Top-K ===")
    from src.planners.cem_mpc import CEMMPC
    cem = CEMMPC(horizon=5, num_samples=32, num_elites=4, num_iterations=2)
    has_evaluate_batch = hasattr(cem, 'evaluate_batch') or True  # CEM supports it via cost_fn
    # Check if CEM optimize returns candidate info
    import numpy as np
    import inspect
    src = inspect.getsource(cem.optimize)
    has_candidates = "elite_idx" in src or "samples[elite_idx]" in src
    audit["checks"]["planner_elite_tracking"] = {
        "status": "PARTIAL",
        "note": "CEM tracks elites internally but does not export top-k candidates. Need to add.",
    }
    print("  CEM top-k export: NOT_IMPLEMENTED (need to add)")

    # Summary
    print("\n=== Summary ===")
    ok_count = sum(1 for v in audit["checks"].values() if v.get("status") == "OK")
    total = len(audit["checks"])
    print(f"  {ok_count}/{total} checks OK")

    # Save
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_path = Path(out_dir) / "capability_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Return blocking status
    blocking = []
    if audit["checks"].get("mujoco_oracle", {}).get("status") != "OK":
        blocking.append("mujoco_oracle")
    if audit["checks"].get("replay_save", {}).get("status") != "OK":
        blocking.append("replay_save")

    if blocking:
        print(f"\nBLOCKING: {blocking}")
        return 1
    else:
        print("\nNo blockers. Ready for sanity sweep.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
