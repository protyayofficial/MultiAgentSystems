#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="Qwen/Qwen3-4B-Instruct-2507"
DOMAIN="svamp"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2

# Separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/SVAMP/SVAMP_train.jsonl"  # For Phase 1 & 2: data generation + training
TEST_DATASET_JSON="datasets/SVAMP/SVAMP_test.jsonl"    # For Phase 3: evaluation

OUTPUT_DIR="result_Qwen_Qwen3-4B-Instruct-2507/gtd_svamp"

mkdir -p "$OUTPUT_DIR"

# --- GTD Mode ---

# == Phase 1: Generate initial dataset (using TRAINING set) ==
echo "--- Running GTD Phase 1: Dataset Generation for SVAMP (TRAIN set) ---"
python -m experiments.run_svamp \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --mode GTD \
  --gtd-generate-data \
  --gtd-datagen-limit 50 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_svamp_dataset_training.jsonl"


# == Phase 2: Train Proxy and Diffusion models (using Phase 1 generated data) ==
echo "--- Running GTD Phase 2: Model Training for SVAMP (TRAIN set) ---"
python -m experiments.run_svamp \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --mode GTD \
  --gtd-train-models \
  --gtd-epochs 10 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_svamp_dataset_training.jsonl" \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_svamp.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_svamp.pth"


# == Phase 3: Run inference (using TEST set - no data leakage) ==
echo "--- Running GTD Phase 3: Inference for SVAMP (TEST set) ---"
python -m experiments.run_svamp \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TEST_DATASET_JSON" \
  --mode GTD \
  --batch_size $BATCH_SIZE \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_svamp.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_svamp.pth"

echo "--- Script finished ---"
