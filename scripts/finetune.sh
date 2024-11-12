#!/bin/bash

set -a
. ./.env
set -a

RUN_NAME="llama_3_1_8b_lora_induction_baseline"
ORIGINAL_MODEL_DIR="./tmp/models/llama_3_1_8b_instruct"
OUTPUT_DIR="./tmp/runs/$RUN_NAME"
echo "$OUTPUT_DIR"

CONFIG_FNAME="./config/llama_3_1_8b_lora_induction_baseline.yaml"

if [ ! -d "$ORIGINAL_MODEL_DIR" ]; then
    tune download meta-llama/Llama-3.1-8B-Instruct \
      --output-dir $ORIGINAL_MODEL_DIR  \
      --hf-token $HF_API_TOKEN
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "making $OUTPUT_DIR"
  mkdir -p OUTPUT_DIR
fi

tune run lora_finetune_single_device \
  --config $CONFIG_FNAME \
  output_dir=$OUTPUT_DIR \
  checkpointer.checkpoint_dir=$ORIGINAL_MODEL_DIR \
  checkpointer.output_dir=$OUTPUT_DIR  \
  tokenizer.path=$ORIGINAL_MODEL_DIR/tokenizer.model



