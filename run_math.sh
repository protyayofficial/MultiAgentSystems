#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="Qwen/Qwen3-8B"
# If you want to try a larger Qwen model, you can switch to:
# LLM_NAME="Qwen/Qwen3-8B"
DOMAIN="math"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2

# Use separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/MATH/MATH_train.jsonl"      # For Phase 1 & 2: data generation + training
TEST_DATASET_JSON="datasets/MATH/MATH_test500.jsonl"     # For Phase 3: evaluation

OUTPUT_DIR="result_Qwen_Qwen3-8B/gtd_math"

mkdir -p "$OUTPUT_DIR"

# --- GTD Mode ---

# == Phase 1: Generate initial dataset (using TRAIN set) ==
echo "--- Running GTD Phase 1: Dataset Generation for MATH (TRAIN set) ---"
python -m experiments.run_math \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --mode GTD \
  --gtd-generate-data \
  --gtd-datagen-limit 50 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_math_dataset_training.jsonl"


# == Phase 2: Train Proxy and Diffusion models (using Phase 1 generated data) ==
echo "--- Running GTD Phase 2: Model Training for MATH (TRAIN set) ---"
python -m experiments.run_math \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET_JSON" \
  --mode GTD \
  --gtd-train-models \
  --gtd-epochs 10 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_math_dataset_training.jsonl" \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_math.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_math.pth"


# == Phase 3: Run inference (using TEST set) ==
echo "--- Running GTD Phase 3: Inference for MATH (TEST set) ---"
python -m experiments.run_math \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TEST_DATASET_JSON" \
  --mode GTD \
  --batch_size $BATCH_SIZE \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_math.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_math.pth"

echo "--- Script finished ---"
