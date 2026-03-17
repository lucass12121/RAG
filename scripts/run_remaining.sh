#!/bin/bash
set -e

echo "=== 1EP MERGE ==="
cd /workspace/LLaMA-Factory
llamafactory-cli export merge_1ep.yaml
echo "=== 1EP MERGE DONE ==="

echo "=== 1EP INFERENCE ==="
python3 /workspace/run_inference_ep.py 1ep /workspace/models/qwen25-7b-legal-merged-1ep
echo "=== 1EP INFERENCE DONE ==="

echo "=== RANK16 TRAINING ==="
llamafactory-cli train train_rank16.yaml
echo "=== RANK16 TRAINING DONE ==="

echo "=== RANK16 MERGE ==="
llamafactory-cli export merge_rank16.yaml
echo "=== RANK16 MERGE DONE ==="

echo "=== RANK16 INFERENCE ==="
python3 /workspace/run_inference_ep.py rank16 /workspace/models/qwen25-7b-legal-merged-rank16
echo "=== ALL DONE ==="
