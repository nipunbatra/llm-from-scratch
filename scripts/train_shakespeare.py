#!/usr/bin/env python3
"""
Train a transformer language model on Shakespeare text.

Usage:
    python train_shakespeare.py --epochs 100 --d-model 128 --n-layers 3
    python train_shakespeare.py --epochs 500 --d-model 256 --n-layers 6 --n-heads 8  # Better
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch import nn
import requests
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.W_qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        return self.W_out(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, block_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.output(x)


def load_data(max_chars=None):
    """Load Shakespeare text."""
    data_path = "data/shakespeare.txt"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(data_path):
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(data_path, 'w') as f:
            f.write(response.text)

    with open(data_path, 'r') as f:
        text = f.read()

    if max_chars:
        text = text[:max_chars]
    return text


def build_dataset(text, block_size, stoi):
    """Build training dataset."""
    data = [stoi[ch] for ch in text]
    X, Y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i + block_size])
        Y.append(data[i + 1:i + block_size + 1])
    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def generate(model, stoi, itos, device, seed="ROMEO:\n", max_len=200, temperature=0.8):
    """Generate text from seed."""
    model.eval()
    block_size = model.block_size
    tokens = [stoi.get(ch, 0) for ch in seed]
    generated = list(seed)

    for _ in range(max_len):
        context = tokens[-block_size:] if len(tokens) >= block_size else tokens
        x = torch.tensor([context]).to(device)
        logits = model(x)[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        tokens.append(next_idx)
        generated.append(itos[next_idx])

    return ''.join(generated)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    text = load_data(args.max_chars)
    print(f"Dataset: {len(text):,} characters")

    # Build vocab
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    # Build dataset
    X, Y = build_dataset(text, args.block_size, stoi)
    X, Y = X.to(device), Y.to(device)
    print(f"Training examples: {len(X):,}")

    # Create model
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(X.shape[0])
        total_loss, n_batches = 0, 0

        for i in range(0, len(X), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x_batch, y_batch = X[idx], Y[idx]

            logits = model(x_batch)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': vocab_size,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'block_size': args.block_size,
        },
        'stoi': stoi,
        'itos': itos,
        'final_loss': losses[-1],
    }
    torch.save(checkpoint, args.output)
    print(f"\nModel saved to {args.output}")

    # Plot loss
    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (final: {losses[-1]:.4f})')
        plt.grid(True, alpha=0.3)
        plt.savefig('models/shakespeare_loss.png', dpi=150)
        print("Loss plot saved to models/shakespeare_loss.png")

    # Generate sample
    print(f"\n{'='*60}")
    print("Generated text:")
    print('='*60)
    print(generate(model, stoi, itos, device, seed="ROMEO:\n", max_len=300))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shakespeare transformer")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=64, help="Context window size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-chars", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--output", type=str, default="models/transformer_shakespeare.pt")
    parser.add_argument("--plot", action="store_true", help="Save loss plot")

    args = parser.parse_args()
    train(args)
