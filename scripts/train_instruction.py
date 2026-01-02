#!/usr/bin/env python3
"""
Train an instruction-following model (SFT).

Usage:
    python train_instruction.py --epochs 200 --d-model 128 --n-layers 3
    python train_instruction.py --epochs 500 --d-model 256 --n-layers 6  # Better
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


# Instruction dataset
INSTRUCTION_DATA = [
    # General Q&A
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "What is the capital of India?", "response": "The capital of India is New Delhi."},
    {"instruction": "What is the capital of Japan?", "response": "The capital of Japan is Tokyo."},
    {"instruction": "What is the capital of Germany?", "response": "The capital of Germany is Berlin."},
    {"instruction": "What is the capital of Italy?", "response": "The capital of Italy is Rome."},
    {"instruction": "What is the capital of Spain?", "response": "The capital of Spain is Madrid."},
    {"instruction": "What is the capital of China?", "response": "The capital of China is Beijing."},
    {"instruction": "What is the capital of Brazil?", "response": "The capital of Brazil is Brasilia."},
    {"instruction": "What is the capital of Australia?", "response": "The capital of Australia is Canberra."},
    {"instruction": "What is the capital of Canada?", "response": "The capital of Canada is Ottawa."},

    # Math
    {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
    {"instruction": "What is 5 * 3?", "response": "5 * 3 equals 15."},
    {"instruction": "What is 10 - 7?", "response": "10 - 7 equals 3."},
    {"instruction": "What is 20 / 4?", "response": "20 / 4 equals 5."},
    {"instruction": "What is 3 + 3?", "response": "3 + 3 equals 6."},
    {"instruction": "What is 7 * 8?", "response": "7 * 8 equals 56."},
    {"instruction": "What is 100 / 10?", "response": "100 / 10 equals 10."},
    {"instruction": "What is 15 - 9?", "response": "15 - 9 equals 6."},
    {"instruction": "What is 6 * 7?", "response": "6 * 7 equals 42."},
    {"instruction": "What is 5 + 5?", "response": "5 + 5 equals 10."},

    # Basic instructions
    {"instruction": "Say hello.", "response": "Hello! How can I help you today?"},
    {"instruction": "Count to five.", "response": "1, 2, 3, 4, 5."},
    {"instruction": "Count to three.", "response": "1, 2, 3."},
    {"instruction": "Count to ten.", "response": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."},
    {"instruction": "Name three colors.", "response": "Red, blue, and green."},
    {"instruction": "List three fruits.", "response": "Apple, banana, and orange."},
    {"instruction": "Name three animals.", "response": "Dog, cat, and elephant."},
    {"instruction": "List three countries.", "response": "USA, India, and Japan."},

    # General knowledge
    {"instruction": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."},
    {"instruction": "What planet is closest to the sun?", "response": "Mercury is the planet closest to the sun."},
    {"instruction": "How many days are in a week?", "response": "There are 7 days in a week."},
    {"instruction": "What is H2O?", "response": "H2O is the chemical formula for water."},
    {"instruction": "How many hours in a day?", "response": "There are 24 hours in a day."},
    {"instruction": "How many months in a year?", "response": "There are 12 months in a year."},

    # Python basics
    {"instruction": "What is Python?", "response": "Python is a popular programming language known for its simplicity and readability."},
    {"instruction": "How do I print Hello World in Python?", "response": "Use print('Hello World') to print Hello World in Python."},
    {"instruction": "What is a variable in Python?", "response": "A variable is a name that stores a value. Example: x = 5 stores 5 in x."},
    {"instruction": "How do I create a list in Python?", "response": "Create a list with square brackets: my_list = [1, 2, 3]"},
    {"instruction": "What is a function in Python?", "response": "A function is reusable code defined with def. Example: def greet(): print('Hi')"},
    {"instruction": "How do I define a function in Python?", "response": "Use def keyword: def my_function(): followed by indented code."},
    {"instruction": "What is a loop in Python?", "response": "A loop repeats code. Use for to iterate: for i in range(5): print(i)"},
    {"instruction": "How do I use a for loop?", "response": "for item in collection: then indented code. Example: for i in range(3): print(i)"},
    {"instruction": "What is an if statement?", "response": "An if statement runs code conditionally: if x > 5: print('big')"},
    {"instruction": "How do I check if a number is even?", "response": "Use modulo: if num % 2 == 0: the number is even."},
    {"instruction": "What is a string in Python?", "response": "A string is text in quotes: name = 'Python' or name = \"Python\""},
    {"instruction": "How do I get the length of a list?", "response": "Use len(): length = len(my_list) returns the number of items."},
    {"instruction": "What is a dictionary in Python?", "response": "A dictionary stores key-value pairs: d = {'name': 'John', 'age': 25}"},
    {"instruction": "How do I add to a list?", "response": "Use append(): my_list.append(4) adds 4 to the end of my_list."},
    {"instruction": "What is range() in Python?", "response": "range(n) generates numbers 0 to n-1. range(5) gives 0,1,2,3,4."},
    {"instruction": "How do I read a file in Python?", "response": "Use open(): with open('file.txt', 'r') as f: content = f.read()"},
    {"instruction": "What is a class in Python?", "response": "A class is a blueprint for objects: class Dog: def bark(self): print('Woof')"},
    {"instruction": "How do I handle errors in Python?", "response": "Use try/except: try: risky_code() except: handle_error()"},
    {"instruction": "What is None in Python?", "response": "None represents no value or null. Example: x = None"},
    {"instruction": "How do I convert string to int?", "response": "Use int(): num = int('42') converts string '42' to integer 42."},
    {"instruction": "What is a tuple in Python?", "response": "A tuple is immutable list: t = (1, 2, 3) cannot be changed after creation."},
    {"instruction": "How do I join strings?", "response": "Use + or join: 'Hello' + ' World' or ' '.join(['Hello', 'World'])"},
    {"instruction": "What is import in Python?", "response": "import loads a module: import math then use math.sqrt(16) for square root."},
    {"instruction": "How do I create a set?", "response": "Use curly braces: s = {1, 2, 3} creates a set with unique values."},
    {"instruction": "What is self in Python?", "response": "self refers to the current object instance in a class method."},
    {"instruction": "How do I reverse a list?", "response": "Use reverse() method: my_list.reverse() or my_list[::-1] for new list."},
    {"instruction": "How do I reverse a string?", "response": "Use slicing: reversed_str = original_str[::-1]"},
    {"instruction": "What is pip?", "response": "pip is Python's package manager. Use pip install package_name to install."},
    {"instruction": "How do I check Python version?", "response": "Run python --version in terminal or import sys; print(sys.version)"},
    {"instruction": "What is a boolean?", "response": "Boolean is True or False. Used in conditions: if is_valid: do_something()"},
    {"instruction": "How do I sort a list?", "response": "Use sort(): my_list.sort() or sorted(my_list) for new sorted list."},

    # Data types
    {"instruction": "What data types exist in Python?", "response": "Python has int, float, str, list, dict, tuple, set, bool, and None."},
    {"instruction": "What is the difference between list and tuple?", "response": "Lists are mutable (changeable), tuples are immutable (fixed after creation)."},
    {"instruction": "How do I check type of a variable?", "response": "Use type(): type(x) returns the type, like <class 'int'> or <class 'str'>"},
    {"instruction": "What is a float in Python?", "response": "A float is a decimal number: x = 3.14 or y = 2.0"},

    # Algorithms
    {"instruction": "What is an algorithm?", "response": "An algorithm is a step-by-step procedure for solving a problem or task."},
    {"instruction": "What is Big O notation?", "response": "Big O describes algorithm efficiency. O(1) is constant, O(n) is linear time."},
    {"instruction": "What is a sorting algorithm?", "response": "A sorting algorithm arranges items in order. Examples: bubble sort, quicksort."},
    {"instruction": "What is recursion?", "response": "Recursion is when a function calls itself. Needs a base case to stop."},
    {"instruction": "What is binary search?", "response": "Binary search finds items in sorted list by halving. O(log n) time."},

    # Machine learning
    {"instruction": "What is machine learning?", "response": "Machine learning is AI where computers learn patterns from data to make predictions."},
    {"instruction": "What is deep learning?", "response": "Deep learning uses neural networks with many layers to learn complex patterns."},
    {"instruction": "What is a neural network?", "response": "A neural network is layers of connected nodes that learn from data."},
    {"instruction": "What is training in ML?", "response": "Training is teaching a model by showing it examples and adjusting weights."},
    {"instruction": "What is overfitting?", "response": "Overfitting is when a model memorizes training data but fails on new data."},
    {"instruction": "What is a loss function?", "response": "A loss function measures how wrong predictions are. Training minimizes it."},
    {"instruction": "What is PyTorch?", "response": "PyTorch is a Python library for deep learning with dynamic computation graphs."},
    {"instruction": "What is a tensor?", "response": "A tensor is a multi-dimensional array, like a matrix but with more dimensions."},
    {"instruction": "What is backpropagation?", "response": "Backpropagation computes gradients to update weights during training."},
    {"instruction": "What is a learning rate?", "response": "Learning rate controls how much to adjust weights. Too high = unstable, too low = slow."},
]


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


def format_example(example):
    """Format instruction-response pair."""
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}<|endoftext|>"""


def build_dataset(text, block_size, stoi):
    """Build training dataset."""
    data = [stoi[ch] for ch in text]
    X, Y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i + block_size])
        Y.append(data[i + 1:i + block_size + 1])
    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def ask(model, instruction, stoi, itos, device, max_tokens=100, temperature=0.7):
    """Ask the model a question."""
    model.eval()
    block_size = model.block_size

    prompt = f"""### Instruction:
{instruction}

### Response:
"""
    tokens = [stoi.get(ch, 0) for ch in prompt]
    generated = list(prompt)

    for _ in range(max_tokens):
        context = tokens[-block_size:] if len(tokens) >= block_size else tokens
        x = torch.tensor([context]).to(device)
        logits = model(x)[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        next_char = itos[next_idx]
        tokens.append(next_idx)
        generated.append(next_char)
        if '<|endoftext|>' in ''.join(generated[-15:]):
            break

    response = ''.join(generated)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
        response = response.replace("<|endoftext|>", "").strip()
    return response


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare training text
    training_text = "\n\n".join(format_example(ex) for ex in INSTRUCTION_DATA)
    print(f"Training data: {len(training_text):,} characters")
    print(f"Instruction pairs: {len(INSTRUCTION_DATA)}")

    # Build vocab
    chars = sorted(set(training_text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    # Build dataset
    X, Y = build_dataset(training_text, args.block_size, stoi)
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

        if epoch % 50 == 0 or epoch == args.epochs - 1:
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
        plt.savefig('models/instruction_loss.png', dpi=150)
        print("Loss plot saved to models/instruction_loss.png")

    # Test
    print(f"\n{'='*60}")
    print("Testing:")
    print('='*60)
    test_questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What is Python?",
        "What is recursion?",
        "Say hello.",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(model, q, stoi, itos, device)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train instruction-following model")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=64, help="Context window size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--output", type=str, default="models/instruction_tuned.pt")
    parser.add_argument("--plot", action="store_true", help="Save loss plot")

    args = parser.parse_args()
    train(args)
