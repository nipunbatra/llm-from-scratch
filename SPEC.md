# LLM from Scratch: Detailed Specification

## Vision

This repository is an educational resource that demystifies large language models by building one from the ground up. Every component—from tokenization to alignment—is implemented from scratch with extensive explanations. The goal is not to build a production-ready model, but to provide deep understanding of how modern LLMs work.

By the end of this series, readers will understand:
- How neural networks learn to predict text
- Why tokenization matters and how BPE works
- The mechanics of self-attention and transformers
- How instruction tuning transforms a text predictor into an assistant
- How preference learning (DPO) aligns models with human values

---

## Target Audience

- **ML students** who want to understand LLMs beyond API calls
- **Practitioners** who use LLMs but want deeper intuition
- **Educators** looking for teaching materials
- **Curious programmers** who learn by building

**Prerequisites:**
- Python proficiency
- Basic PyTorch (tensors, autograd, nn.Module)
- Linear algebra fundamentals (matrix multiplication, softmax)
- Calculus basics (gradients, chain rule)

---

## Part-by-Part Specification

### Part 1: Character-Level Language Model

**Objective:** Build the simplest possible language model that actually works.

**Dataset:** Indian names (~6,000 names)

**Key Concepts:**
1. **Vocabulary and Encoding**
   - Map characters to integers (stoi) and back (itos)
   - Special tokens: '.' for start/end of sequence
   - Why we need numerical representations

2. **Training Data Construction**
   - Sliding window approach
   - Context (X) → Target (Y) pairs
   - Block size / context length concept

3. **Embeddings**
   - Why one-hot encoding is inefficient
   - Learnable dense vectors
   - Embedding as a lookup table
   - Visualizing learned embeddings (PCA)

4. **Model Architecture**
   ```
   Input [batch, block_size]
     → Embedding [batch, block_size, emb_dim]
     → Flatten [batch, block_size * emb_dim]
     → Linear + Tanh [batch, hidden_size]
     → Linear [batch, vocab_size]
   ```

5. **Training Loop**
   - Cross-entropy loss explanation
   - Mini-batch gradient descent
   - AdamW optimizer
   - Loss interpretation (perplexity connection)

6. **Generation**
   - Autoregressive sampling
   - Temperature parameter
   - Nucleus sampling (optional)

**Exercises:**
- Vary block_size (3, 5, 8) and observe quality
- Experiment with hidden_size
- Add dropout for regularization
- Try different activation functions

---

### Part 2: Shakespeare Pretraining

**Objective:** Demonstrate that the same architecture generalizes to different domains.

**Dataset:** Tiny Shakespeare (~1MB of text)

**Key Concepts:**
1. **Domain Transfer**
   - Same model architecture, different data
   - Vocabulary differences (case, punctuation)
   - Longer context requirements

2. **Scaling Considerations**
   - Why we need more parameters for complex data
   - Relationship between data size and model size
   - Training time vs quality tradeoffs

3. **Qualitative Evaluation**
   - Generating from different prompts
   - Character names as seeds
   - Observing learned patterns

4. **Limitations of Character-Level Models**
   - No word-level understanding
   - Inefficient sequence lengths
   - Motivation for better tokenization

**Comparison Table:**
| Aspect | Names | Shakespeare |
|--------|-------|-------------|
| Vocab size | 27 | 65 |
| Context | 5 chars | 32 chars |
| Data size | ~50KB | ~1MB |
| Parameters | ~5K | ~50K |

---

### Part 3: BPE Tokenizer

**Objective:** Implement Byte-Pair Encoding from scratch.

**Key Concepts:**
1. **The Tokenization Problem**
   - Character-level: too granular
   - Word-level: can't handle new words
   - Subword: best of both worlds

2. **BPE Algorithm**
   ```
   1. Initialize vocabulary with characters
   2. Count all adjacent pairs
   3. Merge most frequent pair → new token
   4. Repeat until vocab_size reached
   ```

3. **Implementation Details**
   - Efficient pair counting
   - Merge ordering matters
   - Encoding: apply merges in learned order
   - Decoding: simple concatenation

4. **Compression Analysis**
   - Measure tokens per character
   - Compare across different vocab sizes
   - Tradeoffs: vocab size vs sequence length

5. **Language Model with BPE**
   - Same architecture, token-level instead of char-level
   - Effective context length increases
   - Quality improvements

**Extensions (discussed but not implemented):**
- Byte-level BPE (GPT-2 style)
- SentencePiece / Unigram
- Special tokens handling

---

### Part 4: Self-Attention and Transformers

**Objective:** Build the transformer architecture from scratch.

**Key Concepts:**
1. **Attention Intuition**
   - Dynamic weighting of context
   - Query-Key-Value framework
   - "What am I looking for?" / "What do I contain?" / "What do I provide?"

2. **Attention Mathematics**
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) V
   ```
   - Dot product similarity
   - Scaling factor explanation
   - Softmax for probability distribution

3. **Causal Masking**
   - Why we can't look at future tokens
   - Upper triangular mask
   - -inf before softmax trick

4. **Multi-Head Attention**
   - Multiple attention patterns in parallel
   - Head dimension vs model dimension
   - Concatenation and projection

5. **Positional Encoding**
   - Attention is permutation-invariant
   - Sinusoidal encodings (original paper)
   - Why sin/cos at different frequencies
   - Alternative: learned positions, RoPE, ALiBi

6. **Transformer Block**
   ```
   x → LayerNorm → MultiHeadAttention → + (residual)
     → LayerNorm → FFN → + (residual)
   ```
   - Pre-norm vs post-norm
   - Residual connections for gradient flow
   - Feed-forward expansion (4x)

7. **Complete Model**
   - Token embedding + positional encoding
   - Stack of transformer blocks
   - Final layer norm + output projection

**Visualizations:**
- Attention weight heatmaps
- Positional encoding patterns
- Layer-by-layer activations

---

### Part 5: Instruction Tuning (SFT)

**Objective:** Transform a text completion model into an instruction-following assistant.

**Key Concepts:**
1. **The Gap Problem**
   - Pretrained models complete text, not answer questions
   - "What is 2+2?" → "What is 2+3? What is..."
   - Need to teach the response format

2. **Instruction Data Format**
   ```
   ### Instruction:
   {user's question}

   ### Response:
   {assistant's answer}<|endoftext|>
   ```
   - Consistent formatting is crucial
   - Special tokens for boundaries
   - End-of-text for stopping

3. **Dataset Creation**
   - Human-written examples (expensive, high quality)
   - Model-generated (Alpaca approach)
   - Converted from existing NLP datasets
   - Our mini dataset: 20 examples

4. **Training Process**
   - Standard next-token prediction
   - Loss only on response tokens (optional)
   - Lower learning rate than pretraining
   - Few epochs to avoid forgetting

5. **Evaluation**
   - Format adherence
   - Response quality
   - Generalization to new instructions

**Real-World Context:**
- Alpaca: 52K examples, GPT-3.5 generated
- Dolly: 15K examples, human written
- OpenAssistant: 160K examples, crowdsourced

---

### Part 6: DPO Alignment

**Objective:** Teach the model to prefer good responses over bad ones.

**Key Concepts:**
1. **The Alignment Problem**
   - Instruction tuning teaches format, not quality
   - Model might give correct but unhelpful answers
   - Or helpful but harmful answers
   - Need to encode human preferences

2. **RLHF vs DPO**

   **RLHF Pipeline:**
   ```
   1. Collect preference data
   2. Train reward model
   3. RL (PPO) against reward model
   ```
   - Complex: 3 models, unstable training

   **DPO Pipeline:**
   ```
   1. Collect preference data
   2. Direct optimization with special loss
   ```
   - Simple: 2 models (policy + frozen reference)

3. **DPO Mathematics**
   ```
   L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
   ```
   - Implicit reward model
   - Reference model prevents drift
   - β controls preference strength

4. **Preference Data Types**
   - Helpfulness: detailed vs terse
   - Accuracy: correct vs incorrect
   - Safety: polite vs harmful
   - Tone: friendly vs rude

5. **Implementation**
   - Log probability computation
   - Reference model (frozen copy)
   - Loss function with metrics
   - Monitoring: accuracy, reward margin

6. **Evaluation**
   - Preference accuracy on held-out data
   - Qualitative comparison before/after
   - Safety prompt testing

---

## Progressive Complexity

```
Part 1: 27 vocab, 5 context, ~5K params, MLP
    ↓
Part 2: 65 vocab, 32 context, ~50K params, MLP
    ↓
Part 3: 300 vocab (BPE), 32 context, ~100K params, MLP
    ↓
Part 4: 65 vocab, 64 context, ~400K params, Transformer
    ↓
Part 5: Same architecture + instruction format
    ↓
Part 6: Same architecture + preference learning
```

---

## What We Don't Cover (and Why)

1. **Distributed Training**
   - Requires multiple GPUs
   - Focus is on understanding, not scale

2. **Flash Attention**
   - Optimization, not fundamental concept
   - Standard attention teaches the idea

3. **Mixture of Experts**
   - Advanced architecture variant
   - Core transformer is sufficient

4. **Quantization**
   - Deployment optimization
   - Not relevant for learning

5. **RLHF with PPO**
   - DPO is simpler and equally educational
   - PPO adds RL complexity without new LLM insights

6. **Multi-modal (Vision-Language)**
   - Requires image encoders
   - Text-only keeps focus clear

---

## Success Criteria

After completing this series, readers should be able to:

1. **Explain** how a character-level LM predicts the next token
2. **Implement** BPE tokenization from scratch
3. **Derive** the attention formula and explain each component
4. **Build** a transformer block with correct dimensions
5. **Design** an instruction-tuning dataset
6. **Implement** DPO loss and explain why it works
7. **Debug** common issues in LLM training

---

## Future Extensions (Potential Parts 7+)

- **Part 7: Efficient Fine-tuning (LoRA)**
  - Low-rank adaptation
  - Parameter-efficient training

- **Part 8: Long Context**
  - RoPE positional encoding
  - Sliding window attention

- **Part 9: Tool Use**
  - Function calling format
  - ReAct-style reasoning

- **Part 10: Retrieval Augmentation (RAG)**
  - Embedding models
  - Vector similarity search

---

## Technical Specifications

**Hardware Requirements:**
- CPU-only is fine for all parts
- GPU recommended for Part 4-6 (faster training)
- ~4GB RAM minimum

**Software Dependencies:**
```
torch>=2.0
matplotlib
pandas
numpy
scikit-learn (for PCA visualization)
requests (for data download)
```

**Notebook Execution Time (CPU):**
- Part 1: ~5 minutes
- Part 2: ~10 minutes
- Part 3: ~10 minutes
- Part 4: ~20 minutes
- Part 5: ~15 minutes
- Part 6: ~15 minutes

---

## Contributing

Contributions welcome:
- Bug fixes in notebooks
- Clearer explanations
- Additional exercises
- Translations

Please maintain the educational focus—clarity over cleverness.
