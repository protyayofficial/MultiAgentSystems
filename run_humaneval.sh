#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="Qwen/Qwen3-4B-Instruct-2507"
DOMAIN="humaneval"
AGENT_NAMES="Algorithm_Designer Bug_Fixer Programming_Expert Test_Analyst"
AGENT_NUMS="1 1 1 1"
BATCH_SIZE=2

TRAIN_DATASET_JSON="datasets/humaneval/humaneval-train.jsonl"
TEST_DATASET_JSON="datasets/humaneval/humaneval-test.jsonl"

OUTPUT_DIR="result_Qwen_Qwen3-4B-Instruct-2507/gtd_humaneval"

mkdir -p "$OUTPUT_DIR"

# == Phase 1: Generate initial dataset ==
echo "--- Running GTD Phase 1: Dataset Generation ---"
python3 -m experiments.run_humaneval \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --mode GTD \
  --gtd-generate-data \
  --gtd-datagen-limit 50 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_dataset_training.jsonl"

# == Phase 2: Train models ==
echo "--- Running GTD Phase 2: Model Training ---"
python3 -m experiments.run_humaneval \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --mode GTD \
  --gtd-train-models \
  --gtd-epochs 10 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_dataset_training.jsonl" \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval.pth"

# == Phase 3: Run inference ==
echo "--- Running GTD Phase 3: Inference ---"
python3 -m experiments.run_humaneval \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TEST_DATASET_JSON" \
  --mode GTD \
  --batch_size $BATCH_SIZE \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval.pth"

echo "--- Script finished ---"