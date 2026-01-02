#!/usr/bin/env python3
"""
Train a Python code generation model.

Usage:
    python train_code.py --epochs 200 --d-model 128 --n-layers 3
    python train_code.py --epochs 500 --d-model 256 --n-layers 6  # Better
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


# Python code training data
PYTHON_CODE = '''
# === BASIC FUNCTIONS ===

def hello_world():
    """Print hello world."""
    print("Hello, World!")

def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def square(x):
    """Return square of x."""
    return x * x

def cube(x):
    """Return cube of x."""
    return x * x * x

def power(base, exp):
    """Return base raised to exp."""
    return base ** exp

def absolute(x):
    """Return absolute value."""
    if x < 0:
        return -x
    return x

def maximum(a, b):
    """Return maximum of two numbers."""
    if a > b:
        return a
    return b

def minimum(a, b):
    """Return minimum of two numbers."""
    if a < b:
        return a
    return b

# === LIST OPERATIONS ===

def sum_list(numbers):
    """Sum all numbers in a list."""
    total = 0
    for num in numbers:
        total += num
    return total

def average(numbers):
    """Calculate average of numbers."""
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

def find_max(numbers):
    """Find maximum value in list."""
    if len(numbers) == 0:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

def find_min(numbers):
    """Find minimum value in list."""
    if len(numbers) == 0:
        return None
    min_val = numbers[0]
    for num in numbers:
        if num < min_val:
            min_val = num
    return min_val

def reverse_list(items):
    """Reverse a list."""
    return items[::-1]

def count_items(items):
    """Count items in list."""
    return len(items)

def first_element(items):
    """Get first element."""
    if len(items) == 0:
        return None
    return items[0]

def last_element(items):
    """Get last element."""
    if len(items) == 0:
        return None
    return items[-1]

def remove_duplicates(items):
    """Remove duplicates from list."""
    return list(set(items))

def flatten(nested_list):
    """Flatten a nested list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def contains(items, target):
    """Check if list contains target."""
    for item in items:
        if item == target:
            return True
    return False

def index_of(items, target):
    """Find index of target in list."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

# === STRING OPERATIONS ===

def reverse_string(s):
    """Reverse a string."""
    return s[::-1]

def is_palindrome(s):
    """Check if string is palindrome."""
    s = s.lower()
    return s == s[::-1]

def count_vowels(s):
    """Count vowels in string."""
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

def count_consonants(s):
    """Count consonants in string."""
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    count = 0
    for char in s:
        if char in consonants:
            count += 1
    return count

def count_words(s):
    """Count words in string."""
    words = s.split()
    return len(words)

def to_uppercase(s):
    """Convert to uppercase."""
    return s.upper()

def to_lowercase(s):
    """Convert to lowercase."""
    return s.lower()

def capitalize_words(s):
    """Capitalize each word."""
    return s.title()

def remove_spaces(s):
    """Remove all spaces."""
    return s.replace(" ", "")

def replace_char(s, old, new):
    """Replace character in string."""
    return s.replace(old, new)

def starts_with(s, prefix):
    """Check if string starts with prefix."""
    return s.startswith(prefix)

def ends_with(s, suffix):
    """Check if string ends with suffix."""
    return s.endswith(suffix)

# === NUMBER CHECKS ===

def is_even(n):
    """Check if number is even."""
    return n % 2 == 0

def is_odd(n):
    """Check if number is odd."""
    return n % 2 != 0

def is_positive(n):
    """Check if number is positive."""
    return n > 0

def is_negative(n):
    """Check if number is negative."""
    return n < 0

def is_zero(n):
    """Check if number is zero."""
    return n == 0

def is_prime(n):
    """Check if number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def is_perfect_square(n):
    """Check if n is perfect square."""
    if n < 0:
        return False
    root = int(n**0.5)
    return root * root == n

def is_divisible(a, b):
    """Check if a is divisible by b."""
    if b == 0:
        return False
    return a % b == 0

# === CLASSIC ALGORITHMS ===

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def factorial_iterative(n):
    """Calculate factorial iteratively."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def fibonacci(n):
    """Return nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    """Return nth Fibonacci iteratively."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_list(n):
    """Return first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fibs = [0, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def gcd(a, b):
    """Find greatest common divisor."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Find least common multiple."""
    return a * b // gcd(a, b)

def binary_search(arr, target):
    """Binary search for target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def linear_search(arr, target):
    """Linear search for target."""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def bubble_sort(arr):
    """Sort array using bubble sort."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection_sort(arr):
    """Sort array using selection sort."""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """Sort array using insertion sort."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """Sort array using merge sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# === DATA STRUCTURES ===

class Stack:
    """Stack data structure."""
    def __init__(self):
        self.items = []

    def push(self, item):
        """Add item to top."""
        self.items.append(item)

    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            return None
        return self.items.pop()

    def peek(self):
        """Return top item without removing."""
        if self.is_empty():
            return None
        return self.items[-1]

    def is_empty(self):
        """Check if stack is empty."""
        return len(self.items) == 0

    def size(self):
        """Return number of items."""
        return len(self.items)

class Queue:
    """Queue data structure."""
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """Add item to back."""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item."""
        if self.is_empty():
            return None
        return self.items.pop(0)

    def front(self):
        """Return front item without removing."""
        if self.is_empty():
            return None
        return self.items[0]

    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0

    def size(self):
        """Return number of items."""
        return len(self.items)

class Node:
    """Node for linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """Singly linked list."""
    def __init__(self):
        self.head = None

    def append(self, data):
        """Add node to end."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, data):
        """Add node to beginning."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        """Delete first node with data."""
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next

    def find(self, data):
        """Check if data exists in list."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def size(self):
        """Return number of nodes."""
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

# === FILE OPERATIONS ===

def read_file(filename):
    """Read contents of a file."""
    with open(filename, 'r') as f:
        return f.read()

def write_file(filename, content):
    """Write content to a file."""
    with open(filename, 'w') as f:
        f.write(content)

def append_file(filename, content):
    """Append content to a file."""
    with open(filename, 'a') as f:
        f.write(content)

def read_lines(filename):
    """Read file as list of lines."""
    with open(filename, 'r') as f:
        return f.readlines()

def count_lines(filename):
    """Count lines in a file."""
    with open(filename, 'r') as f:
        return len(f.readlines())

# === DICTIONARY OPERATIONS ===

def merge_dicts(dict1, dict2):
    """Merge two dictionaries."""
    result = dict1.copy()
    result.update(dict2)
    return result

def get_keys(d):
    """Get all keys from dictionary."""
    return list(d.keys())

def get_values(d):
    """Get all values from dictionary."""
    return list(d.values())

def invert_dict(d):
    """Swap keys and values."""
    return {v: k for k, v in d.items()}

def filter_dict(d, keys):
    """Filter dictionary by keys."""
    return {k: v for k, v in d.items() if k in keys}

# === LIST COMPREHENSIONS ===

def squares(n):
    """Return squares from 1 to n."""
    return [x**2 for x in range(1, n+1)]

def cubes(n):
    """Return cubes from 1 to n."""
    return [x**3 for x in range(1, n+1)]

def evens(n):
    """Return even numbers up to n."""
    return [x for x in range(n+1) if x % 2 == 0]

def odds(n):
    """Return odd numbers up to n."""
    return [x for x in range(n+1) if x % 2 != 0]

def filter_positive(numbers):
    """Filter positive numbers."""
    return [x for x in numbers if x > 0]

def filter_negative(numbers):
    """Filter negative numbers."""
    return [x for x in numbers if x < 0]

def double_all(numbers):
    """Double all numbers."""
    return [x * 2 for x in numbers]

def triple_all(numbers):
    """Triple all numbers."""
    return [x * 3 for x in numbers]

# === ERROR HANDLING ===

def safe_divide(a, b):
    """Safely divide with error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        return None

def safe_int(s):
    """Safely convert string to int."""
    try:
        return int(s)
    except ValueError:
        return None

def safe_float(s):
    """Safely convert string to float."""
    try:
        return float(s)
    except ValueError:
        return None

def safe_get(lst, index):
    """Safely get list element."""
    try:
        return lst[index]
    except IndexError:
        return None

def safe_key(d, key):
    """Safely get dictionary value."""
    try:
        return d[key]
    except KeyError:
        return None
'''


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


def build_dataset(text, block_size, stoi):
    """Build training dataset."""
    data = [stoi[ch] for ch in text]
    X, Y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i + block_size])
        Y.append(data[i + 1:i + block_size + 1])
    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def generate_code(model, prompt, stoi, itos, device, max_tokens=200, temperature=0.7):
    """Generate Python code."""
    model.eval()
    block_size = model.block_size

    tokens = [stoi.get(ch, 0) for ch in prompt]
    generated = list(prompt)

    for _ in range(max_tokens):
        context = tokens[-block_size:] if len(tokens) >= block_size else tokens
        x = torch.tensor([context]).to(device)
        logits = model(x)[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        tokens.append(next_idx)
        generated.append(itos[next_idx])
        if ''.join(generated[-3:]) == '\n\n\n':
            break

    return ''.join(generated)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    print(f"Training data: {len(PYTHON_CODE):,} characters")

    # Build vocab
    chars = sorted(set(PYTHON_CODE))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    # Build dataset
    X, Y = build_dataset(PYTHON_CODE, args.block_size, stoi)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
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

        scheduler.step()
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
        plt.savefig('models/code_loss.png', dpi=150)
        print("Loss plot saved to models/code_loss.png")

    # Test
    print(f"\n{'='*60}")
    print("Code generation test:")
    print('='*60)
    prompts = [
        'def is_even(n):\n    """',
        'def factorial(n):\n    """',
        'def reverse_string(s):\n    """',
    ]
    for prompt in prompts:
        print(f"\nPrompt: {repr(prompt)}")
        print(generate_code(model, prompt, stoi, itos, device, max_tokens=150))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Python code generation model")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=64, help="Context window size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--output", type=str, default="models/python_code_lm.pt")
    parser.add_argument("--plot", action="store_true", help="Save loss plot")

    args = parser.parse_args()
    train(args)
