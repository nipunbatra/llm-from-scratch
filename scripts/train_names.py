#!/usr/bin/env python3
"""
Train a character-level language model on Indian names.

Usage:
    python train_names.py --epochs 2000 --hidden 256 --emb 16
    python train_names.py --epochs 5000 --hidden 512 --emb 32  # Better quality
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import requests
import matplotlib.pyplot as plt


class CharLM(nn.Module):
    """Character-level language model."""

    def __init__(self, vocab_size, block_size, emb_dim, hidden_size):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.hidden = nn.Linear(block_size * emb_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.tanh(self.hidden(x))
        return self.output(x)


def load_data():
    """Load and preprocess names dataset."""
    data_path = "data/names.csv"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(data_path):
        print("Downloading names dataset...")
        url = "https://raw.githubusercontent.com/balasahebgulave/Dataset-Indian-Names/master/Indian_Names.csv"
        response = requests.get(url)
        with open(data_path, 'w') as f:
            f.write(response.text)

    words = pd.read_csv(data_path)["Name"]
    words = words.str.lower().str.strip().str.replace(" ", "")
    words = words[words.str.len().between(3, 9)]
    words = words[words.apply(lambda x: x.isalpha())]
    words = words.sample(frac=1, random_state=42).reset_index(drop=True)
    return words.tolist()


def build_vocab(words):
    """Build character vocabulary."""
    chars = sorted(set(''.join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi['.'] = 0  # Start/end token
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size):
    """Convert words to training examples."""
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            target = stoi[ch]
            X.append(context.copy())
            Y.append(target)
            context = context[1:] + [target]
    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def generate(model, stoi, itos, block_size, device, temperature=0.8):
    """Generate a single name."""
    model.eval()
    context = [0] * block_size
    name = []
    for _ in range(15):
        x = torch.tensor([context]).to(device)
        logits = model(x)
        probs = F.softmax(logits / temperature, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        if next_idx == 0:
            break
        name.append(itos[next_idx])
        context = context[1:] + [next_idx]
    return ''.join(name)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    words = load_data()
    print(f"Loaded {len(words)} names")

    # Build vocab and dataset
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    X, Y = build_dataset(words, stoi, args.block_size)
    X, Y = X.to(device), Y.to(device)
    print(f"Training examples: {len(X)}")

    # Create model
    model = CharLM(vocab_size, args.block_size, args.emb_dim, args.hidden_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(X.shape[0])
        total_loss, n_batches = 0, 0

        for i in range(0, X.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            x_batch, y_batch = X[idx], Y[idx]

            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if epoch % 200 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")
            # Generate sample names
            samples = [generate(model, stoi, itos, args.block_size, device) for _ in range(3)]
            print(f"         Samples: {', '.join(samples)}")

    # Save model
    os.makedirs("models", exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'block_size': args.block_size,
        'emb_dim': args.emb_dim,
        'hidden_size': args.hidden_size,
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
        plt.savefig('models/names_loss.png', dpi=150)
        print("Loss plot saved to models/names_loss.png")

    # Generate final samples
    print(f"\n{'='*50}")
    print("Generated names:")
    print('='*50)
    for i in range(10):
        name = generate(model, stoi, itos, args.block_size, device, temperature=0.8)
        print(f"  {name.capitalize()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train character-level name generator")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=5, help="Context window size")
    parser.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--output", type=str, default="models/char_lm_names.pt", help="Output path")
    parser.add_argument("--plot", action="store_true", help="Save loss plot")

    args = parser.parse_args()
    train(args)
