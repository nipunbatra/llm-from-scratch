#!/bin/bash
# Train all models with good quality settings
# Usage: ./train_all.sh [--quick]

set -e
cd "$(dirname "$0")/.."

if [ "$1" == "--quick" ]; then
    echo "=== QUICK MODE: Training smaller models for testing ==="

    echo -e "\n[1/4] Training name generator..."
    python scripts/train_names.py --epochs 1000 --hidden 256 --emb 16 --plot

    echo -e "\n[2/4] Training Shakespeare model..."
    python scripts/train_shakespeare.py --epochs 100 --d-model 128 --n-layers 3 --n-heads 4 --plot

    echo -e "\n[3/4] Training instruction model..."
    python scripts/train_instruction.py --epochs 200 --d-model 128 --n-layers 3 --n-heads 4 --plot

    echo -e "\n[4/4] Training code model..."
    python scripts/train_code.py --epochs 200 --d-model 128 --n-layers 3 --n-heads 4 --plot

else
    echo "=== FULL MODE: Training high-quality models ==="

    echo -e "\n[1/4] Training name generator..."
    python scripts/train_names.py --epochs 3000 --hidden 512 --emb 32 --plot

    echo -e "\n[2/4] Training Shakespeare model..."
    python scripts/train_shakespeare.py --epochs 300 --d-model 256 --n-layers 6 --n-heads 8 --plot

    echo -e "\n[3/4] Training instruction model..."
    python scripts/train_instruction.py --epochs 500 --d-model 256 --n-layers 6 --n-heads 8 --plot

    echo -e "\n[4/4] Training code model..."
    python scripts/train_code.py --epochs 500 --d-model 256 --n-layers 6 --n-heads 8 --plot
fi

echo -e "\n=== All models trained! ==="
echo "Models saved to models/"
ls -la models/*.pt
