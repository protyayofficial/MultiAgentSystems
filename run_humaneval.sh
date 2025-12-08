#!/bin/bash

# This script runs the Humaneval experiments with GTD (3 phases).
set -e
set -x

# --- Configuration ---
LLM_NAME="Qwen/Qwen3-4B-Thinking-2507"
DOMAIN="humaneval"
AGENT_NAMES="Algorithm_Designer Bug_Fixer Programming_Expert Test_Analyst"
AGENT_NUMS="1 1 1 1"  # Corresponds to the number of agents for each name
BATCH_SIZE=2

# Separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/humaneval/humaneval-train.jsonl"  # For Phase 1 & 2: data generation + training
TEST_DATASET_JSON="datasets/humaneval/humaneval-test.jsonl"    # For Phase 3: evaluation

OUTPUT_DIR="result_Qwen3_8B/gtd_humaneval"  # All outputs will go here

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# --- GTD Mode ---
# This mode uses a Graph Topology Diffusion model to generate agent communication graphs.

# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation for Humaneval (TRAIN set) ---"
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


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for Humaneval (TRAIN set) ---"
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


# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference for Humaneval (TEST set) ---"
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
