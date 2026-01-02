# LLM from Scratch

A 6-part educational series that builds language models from the ground up. Every line of code is written from scratch and explained.

## The Journey

| Part | Notebook | What You'll Learn |
|------|----------|-------------------|
| 1 | [Character-Level LM](notebooks/part-1-char-lm.ipynb) | Embeddings, MLPs, training loops, generating names |
| 2 | [Shakespeare Pretraining](notebooks/part-2-shakespeare.ipynb) | Applying the same model to different data |
| 3 | [BPE Tokenizer](notebooks/part-3-bpe.ipynb) | Byte-pair encoding from scratch, subword tokenization |
| 4 | [Self-Attention](notebooks/part-4-attention.ipynb) | The transformer's core mechanism |
| 5 | [Instruction Tuning](notebooks/part-5-instruction.ipynb) | Making the model follow instructions |
| 6 | [DPO Alignment](notebooks/part-6-dpo.ipynb) | Aligning with human preferences |

## Philosophy

- **Everything from scratch**: No magic, no black boxes
- **Progressively complex**: Each part builds on the previous
- **Educational focus**: Understanding over performance

## Prerequisites

- Python basics
- PyTorch fundamentals (tensors, autograd)
- Basic calculus and linear algebra

## Getting Started

```bash
git clone https://github.com/nipunbatra/llm-from-scratch
cd llm-from-scratch
pip install -r requirements.txt
jupyter notebook notebooks/part-1-char-lm.ipynb
```

## The Complete LLM Pipeline

```
PRETRAINING (Part 1-2)
├── Character-level language modeling
├── Learn language patterns from raw text
└── Names dataset → Shakespeare

TOKENIZATION (Part 3)
├── BPE: Subword tokenization
├── Efficient representation
└── Handles any text

ARCHITECTURE (Part 4)
├── Self-attention mechanism
├── Transformer blocks
└── Positional encoding

INSTRUCTION TUNING (Part 5)
├── Teach instruction-following format
├── (instruction, response) pairs
└── Standard supervised learning

ALIGNMENT (Part 6)
├── Teach preferences with DPO
├── (instruction, chosen, rejected) triplets
└── Direct preference optimization
```

## Author

Nipun Batra
