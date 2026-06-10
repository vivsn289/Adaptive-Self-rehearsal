#!/bin/bash
# Run adaptive SFT across 3 paired seeds and benchmark each.
# Aggressively cleans up model dirs after each seed to bound disk usage to ~7 GB.

set -e

SEEDS=(42 1337 2025)
EPOCHS=${EPOCHS:-4}
mkdir -p logs results/adaptive

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="./qwen-sft-seed${SEED}-ep${EPOCHS}-adaptive"
    BENCHMARK_OUT="results/adaptive/seed${SEED}-ep${EPOCHS}-adaptive"

    echo "================================================================"
    echo "[$(date)] Seed $SEED: ADAPTIVE SFT training"
    echo "  Disk free before:  $(df -h /home | awk 'NR==2 {print $4}')"
    echo "================================================================"
    SEED=$SEED EPOCHS=$EPOCHS OUTPUT_DIR_OVERRIDE="$OUTPUT_DIR" \
        python finetune_adaptive.py 2>&1 | tee "logs/finetune_adaptive_seed${SEED}.log"

    echo "================================================================"
    echo "[$(date)] Seed $SEED: benchmarking"
    echo "================================================================"
    MODEL="$OUTPUT_DIR" OUT="$BENCHMARK_OUT" \
        bash benchmark_instruct.sh 2>&1 | tee "logs/benchmark_adaptive_seed${SEED}.log"

    # Preserve probe trajectory in the benchmark output dir before deleting model
    if [ -f "$OUTPUT_DIR/probe_trajectory.json" ]; then
        cp "$OUTPUT_DIR/probe_trajectory.json" "$BENCHMARK_OUT/"
        echo "  Probe trajectory preserved → $BENCHMARK_OUT/probe_trajectory.json"
    fi

    # CRITICAL: delete model dir now that benchmarking is done
    # The benchmark results in BENCHMARK_OUT are what we need; the weights are not.
    echo "  Cleaning up: rm -rf $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"

    echo "[$(date)] Seed $SEED: COMPLETE"
    echo "  Disk free after cleanup: $(df -h /home | awk 'NR==2 {print $4}')"
done

echo ""
echo "All adaptive seeds complete."
echo "Final disk: $(df -h /home | awk 'NR==2 {print $4}') free"
