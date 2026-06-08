#!/bin/bash
# Clean default cost sweep using xargs for parallelism
WORKERS="${1:-10}"
OUT_DIR="${2:-runs/oracle_clean_default_sweep}"
SCRIPT="scripts/run_single_clean_trial.py"
PYTHON="/home/brucewu/miniconda3/envs/lerobot/bin/python"
SEED_BASE=20260601

mkdir -p "$OUT_DIR"

generate_tasks() {
    CEM_CONFIGS="CEM_s01_H100 CEM_s01_H140 CEM_s03_H100 CEM_s05_H100 CEM_s05_H140_exec20 CEM_s075_exec20"
    MPPI_CONFIGS="MPPI_s01_H100 MPPI_s03_H100 MPPI_s05_H100 MPPI_s075_H100 MPPI_s03_H140 MPPI_s05_H140 MPPI_s075_exec20"
    FAMILIES="open blocking passage"

    for cfg in $CEM_CONFIGS; do
        for fam in $FAMILIES; do
            for trial in 0 1 2; do
                seed=$((SEED_BASE + $(echo "CEM_${cfg}_clean_default_${fam}_${trial}" | cksum | cut -d' ' -f1) % 100000))
                echo "CEM $cfg $fam $seed $OUT_DIR"
            done
        done
    done

    for cfg in $MPPI_CONFIGS; do
        for fam in $FAMILIES; do
            for trial in 0 1 2; do
                seed=$((SEED_BASE + $(echo "MPPI_${cfg}_clean_default_${fam}_${trial}" | cksum | cut -d' ' -f1) % 100000))
                echo "MPPI $cfg $fam $seed $OUT_DIR"
            done
        done
    done

    for fam in $FAMILIES; do
        for trial in 0 1 2; do
            seed=$((SEED_BASE + $(echo "MPPI_baseline_clean_default_${fam}_${trial}" | cksum | cut -d' ' -f1) % 100000))
            echo "MPPI MPPI_s03_H100 $fam $seed $OUT_DIR"
        done
    done
}

run_trial() {
    local line="$1"
    local planner config family seed out_dir
    read -r planner config family seed out_dir <<< "$line"
    local trial_id="${planner}_${config}_clean_default_${family}_${seed}"
    local ep_dir="${out_dir}/episodes/${trial_id}"
    
    # Skip if already done
    if [ -f "$ep_dir/metadata.json" ]; then
        echo "SKIP $trial_id (exists)"
        return 0
    fi
    
    MUJOCO_GL=egl $PYTHON -u $SCRIPT "$planner" "$config" "$family" "$seed" "$out_dir" >> "$out_dir/trial_log.txt" 2>&1
}

export -f run_trial
export PYTHON SCRIPT OUT_DIR

TOTAL=$(generate_tasks | wc -l)
echo "=== Clean Default Cost Sweep ===" | tee "$OUT_DIR/sweep.log"
echo "  Workers: $WORKERS" | tee -a "$OUT_DIR/sweep.log"
echo "  Total trials: $TOTAL" | tee -a "$OUT_DIR/sweep.log"
echo "  Output: $OUT_DIR" | tee -a "$OUT_DIR/sweep.log"
echo "" | tee -a "$OUT_DIR/sweep.log"
echo "Started at $(date)" | tee -a "$OUT_DIR/sweep.log"

generate_tasks | xargs -P "$WORKERS" -I{} bash -c 'run_trial "$@"' _ {}

echo "Done at $(date)" | tee -a "$OUT_DIR/sweep.log"
