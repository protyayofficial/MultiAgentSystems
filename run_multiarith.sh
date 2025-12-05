#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
# LLM_NAME="Qwen3-8B"
DOMAIN="multiarith"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2

# Separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/MultiArith/MultiArith_train.json"  # For Phase 1 & 2: data generation + training
TEST_DATASET_JSON="datasets/MultiArith/MultiArith_test.json"    # For Phase 3: evaluation

OUTPUT_DIR="result/gtd_multiarith" # All outputs will go here

mkdir -p "$OUTPUT_DIR"

BASE_CMD_COMMON="python -m experiments.run_multiarith \
  --llm_name $LLM_NAME \
  --domain $DOMAIN \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS"

# --- GTD Mode ---

# == Phase 1: Generate initial dataset (using TRAINING set) ==
echo "--- Running GTD Phase 1: Dataset Generation for MultiArith (TRAIN set) ---"
$BASE_CMD_COMMON \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --gtd-generate-data \
  --gtd-datagen-limit 10 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_multiarith_dataset_training.jsonl"


# == Phase 2: Train Proxy and Diffusion models (using Phase 1 generated data) ==
echo "--- Running GTD Phase 2: Model Training for MultiArith (TRAIN set) ---"
$BASE_CMD_COMMON \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --gtd-train-models \
  --gtd-epochs 10 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_multiarith_dataset_training.jsonl" \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_multiarith.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_multiarith.pth"


# == Phase 3: Run inference (using TEST set - no data leakage) ==
echo "--- Running GTD Phase 3: Inference for MultiArith (TEST set) ---"
$BASE_CMD_COMMON \
  --dataset_json "$TEST_DATASET_JSON" \
  --batch_size $BATCH_SIZE \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_multiarith.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_multiarith.pth"

echo "--- Script finished ---"
