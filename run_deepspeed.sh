#!/bin/bash

# Run DeepSpeed training with 4 GPUs using QLoRA for memory optimization
deepspeed --num_gpus=4 train.py \
    --dataset SQA \
    --model_name deepseek-ai/DeepSeek-Coder-V2-Lite-Base \
    --batch_size 1 \
    --bits 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --deepspeed_config ds_config.json \
    "$@"