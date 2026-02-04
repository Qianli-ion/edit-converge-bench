#!/bin/bash
# Run all models for 10 rounds on the benchmark

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

ROUNDS=10
DATA_DIR="data"

# List of models to run
MODELS=(
    "seedream"
    "grok"
    "qwen"
    "nano-banana-edit"
)

echo "========================================"
echo "ConvergeBench - Running ${#MODELS[@]} models"
echo "Rounds: $ROUNDS"
echo "========================================"

for model in "${MODELS[@]}"; do
    OUTPUT_DIR="results_${model}_${ROUNDS}rounds"
    echo ""
    echo "========================================"
    echo "Running: $model"
    echo "Output: $OUTPUT_DIR"
    echo "========================================"
    
    python eval/run_smart_benchmark.py \
        --model "$model" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --max-rounds "$ROUNDS"
    
    echo ""
    echo "✓ Completed: $model"
done

echo ""
echo "========================================"
echo "All models complete!"
echo "========================================"
