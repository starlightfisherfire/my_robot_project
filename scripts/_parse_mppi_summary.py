#!/usr/bin/env python3
"""Parse a render script JSON output into a CSV manifest row.
Called by run_mppi_param_sweep_checkpoint8h_v1.sh"""
import json, sys

def main():
    json_file = sys.argv[1]
    stage = sys.argv[2]
    priority = sys.argv[3]
    cfg = sys.argv[4]
    family = sys.argv[5]
    T = sys.argv[6]
    video = sys.argv[7]
    log = sys.argv[8]

    with open(json_file) as f:
        d = json.load(f)

    ess_mean = d.get("effective_sample_size_mean", "")
    ess_min = d.get("effective_sample_size_min", "")
    ent_mean = d.get("weight_entropy_mean", "")
    collapse = d.get("collapse_rate", "")
    tc_flag = d.get("temperature_collapse_flag", "")
    nan_check = d.get("nan_check", "")

    row = [
        stage, priority, cfg, family,
        str(d.get("template_family", "")),
        family, "mppi", str(T),
        str(d.get("num_samples", "")),
        str(d.get("num_iterations", "")),
        str(d.get("init_std", "")),
        "", "",  # smoothing, extra
        str(d.get("speed_mps", "")),
        "",  # speed_cm_s
        str(d.get("horizon", "")),
        str(d.get("execute_steps", "")),
        str(d.get("max_mpc_steps", "")),
        "", "",  # total_budget, template_file
        str(d.get("split", "")),
        str(d.get("template_index", "")),
        str(d.get("template_id", "")),
        str(d.get("obstacle_count", "")),
        str(d.get("passage_gap", "")),
        str(d.get("is_direct", "")),
        str(d.get("is_bypass", "")),
        "completed" if d.get("success") is not None else "failed",
        str(d.get("success", "")),
        str(d.get("best_dist", "")),
        str(d.get("avg_cost", "")),
        "",  # min_cost
        str(d.get("collision_count", "")),
        "", "",  # collision_rate, contact_count
        str(d.get("mpc_steps", "")),
        str(d.get("runtime_sec", "")),
        str(ess_mean), str(ess_min), str(ent_mean),
        str(collapse), str(tc_flag), str(nan_check),
        str(d.get("episode_id", "")),
        video, log, ""
    ]
    print(",".join(row))


if __name__ == "__main__":
    main()
