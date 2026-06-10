#!/bin/bash
set -e
source env/bin/activate

MODEL="mistralai/Mistral-7B-v0.1"
OUT="results/base"

echo "Benchmarking BASE model: $MODEL"
echo "Logging to logs/benchmark_base.log"

# Fast multiple-choice benchmarks
lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks arc_challenge,arc_easy,hellaswag,boolq,mmlu,winogrande,piqa,openbookqa,copa,rte,lambada_openai \
    --batch_size 4 \
    --output_path $OUT 2>&1 | tee logs/benchmark_base_mc.log

# Generation-heavy benchmarks (slower, smaller batch)
lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks gsm8k,triviaqa \
    --batch_size 1 \
    --output_path $OUT 2>&1 | tee logs/benchmark_base_gen.log

echo "BASE benchmarks complete. Results in $OUT"
