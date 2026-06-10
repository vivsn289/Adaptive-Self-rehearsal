#!/bin/bash
set -e
source env/bin/activate

SEEDS="42 1337 2025"
EPOCHS=4

for SEED in $SEEDS; do
    echo ""
    echo "========================================"
    echo "=== SEED $SEED (EPOCHS=$EPOCHS) ==="
    echo "========================================"
    OUTDIR="./qwen-sft-seed${SEED}-ep${EPOCHS}"
    RESULTS="results/instruct/seed${SEED}-ep${EPOCHS}"
    
    SEED=$SEED EPOCHS=$EPOCHS OUTPUT_DIR_OVERRIDE=$OUTDIR \
        python finetune.py 2>&1 | tee logs/finetune_seed${SEED}.log
    
    MODEL=$OUTDIR OUT=$RESULTS \
        bash benchmark_instruct.sh 2>&1 | tee logs/benchmark_instruct_seed${SEED}.log
    
    echo "=== Seed $SEED done ==="
done

echo ""
echo "All 3 seeds complete!"
ls -la results/instruct/
