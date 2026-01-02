#!/bin/bash
# Train all models with balanced settings (~15 mins total on GPU)
# Good quality without excessive training time

set -e
cd "$(dirname "$0")/.."

echo "=== BALANCED MODE: ~15 mins total on GPU ==="
echo "Better than quick, faster than full"
echo ""

# Names: ~2 mins
echo "[1/4] Training name generator (~2 mins)..."
python scripts/train_names.py \
    --epochs 2000 \
    --hidden 384 \
    --emb 24 \
    --batch-size 2048 \
    --lr 0.01 \
    --plot

# Shakespeare: ~4 mins
echo -e "\n[2/4] Training Shakespeare model (~4 mins)..."
python scripts/train_shakespeare.py \
    --epochs 150 \
    --d-model 192 \
    --n-layers 4 \
    --n-heads 6 \
    --block-size 96 \
    --batch-size 128 \
    --max-chars 200000 \
    --plot

# Instruction: ~4 mins
echo -e "\n[3/4] Training instruction model (~4 mins)..."
python scripts/train_instruction.py \
    --epochs 300 \
    --d-model 192 \
    --n-layers 4 \
    --n-heads 6 \
    --block-size 96 \
    --batch-size 64 \
    --plot

# Code: ~4 mins
echo -e "\n[4/4] Training code model (~4 mins)..."
python scripts/train_code.py \
    --epochs 300 \
    --d-model 192 \
    --n-layers 4 \
    --n-heads 6 \
    --block-size 96 \
    --batch-size 64 \
    --plot

echo -e "\n=== All models trained! ==="
echo "Models saved to models/"
ls -lh models/*.pt
echo ""
echo "Model sizes:"
for f in models/*.pt; do
    params=$(python -c "import torch; c=torch.load('$f', weights_only=False); print(f\"{sum(p.numel() for p in torch.nn.Module().parameters()) if 'model_state_dict' not in c else sum(v.numel() for v in c['model_state_dict'].values()):,}\")" 2>/dev/null || echo "?")
    echo "  $(basename $f): $params parameters"
done
