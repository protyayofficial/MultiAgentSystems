#!/bin/bash

set -e
set -x

# Configuration
LLM_NAME="Qwen/Qwen3-4B-Thinking-2507"
DOMAIN="multiarith"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2

TRAIN_DATASET="datasets/MultiArith/MultiArith_train.json"
TEST_DATASET="datasets/MultiArith/MultiArith_test.json"

OUTPUT_DIR="result/gtd_multiarith"
mkdir -p "$OUTPUT_DIR"

# Phase 1: Generate dataset
echo "=== Phase 1: Generating GTD dataset ==="
python -m experiments.run_multiarith \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET" \
  --gtd-generate-data \
  --gtd-datagen-limit 50 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_multiarith_dataset.jsonl"

# Phase 2: Train models
echo "=== Phase 2: Training GTD models ==="
python -m experiments.run_multiarith \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TRAIN_DATASET" \
  --gtd-train-models \
  --gtd-epochs 10 \
  --gtd-dataset-path "$OUTPUT_DIR/gtd_multiarith_dataset.jsonl" \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_multiarith.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_multiarith.pth"

# Phase 3: Inference
echo "=== Phase 3: GTD Inference ==="
python -m experiments.run_multiarith \
  --llm_name "$LLM_NAME" \
  --domain "$DOMAIN" \
  --agent_names $AGENT_NAMES \
  --agent_nums $AGENT_NUMS \
  --dataset_json "$TEST_DATASET" \
  --batch_size $BATCH_SIZE \
  --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_multiarith.pth" \
  --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_multiarith.pth"

echo "=== Complete ==="