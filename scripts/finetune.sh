#!/bin/bash

set -a
. ./.env
set -a

ORIGINAL_MODEL_DIR="/tmp/llama_8b_instruct/original"
OUTPUT_DIR="./tmp/llama_8b_lora"
CKPT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR=".$OUTPUT_DIR/logs"

if [ ! -d "$ORIGINAL_MODEL_DIR" ]; then
    tune download meta-llama/Llama-3.1-8B-Instruct \
      --output-dir $ORIGINAL_MODEL_DIR  \
      --hf-token $HF_API_TOKEN
fi

if [ ! -d "$CKPT_DIR" ]; then
  mkdir -p CKPT_DIR
fi

tune run lora_finetune_single_device \
  --config ./config/llama_8b_lora.yaml \
  output_dir=$OUTPUT_DIR \
  checkpointer.checkpoint_dir=$CKPT_DIR \
  checkpointer.output_dir=$CKPT_DIR 


