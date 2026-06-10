#!/bin/bash
set -e
source env/bin/activate

MODEL="./mistral-self-instruct-sft"
OUT="results/instruct"

echo "Benchmarking FINE-TUNED model: $MODEL"

lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks arc_challenge,arc_easy,hellaswag,boolq,mmlu,winogrande,piqa,openbookqa,copa,rte,lambada_openai \
    --batch_size 4 \
    --output_path $OUT 2>&1 | tee logs/benchmark_instruct_mc.log

lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks gsm8k,triviaqa \
    --batch_size 1 \
    --output_path $OUT 2>&1 | tee logs/benchmark_instruct_gen.log

echo "FINE-TUNED benchmarks complete. Results in $OUT"
