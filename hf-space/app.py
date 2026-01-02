"""
LLM From Scratch - Interactive Demo
A Gradio app showcasing models trained in the educational series.
"""

import torch
import torch.nn.functional as F
from torch import nn
import math
import gradio as gr
from huggingface_hub import hf_hub_download
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model Architectures (copied from notebooks)
# ============================================================

class CharLM(nn.Module):
    """Character-level language model (Parts 1-2)."""

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
        x = self.output(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

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
    """Multi-head self-attention."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=True):
        batch, seq_len, d_model = x.shape
        qkv = self.W_qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        return self.W_out(output)


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
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
    """Transformer language model (Parts 4-6)."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, block_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
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


# ============================================================
# Model Loading
# ============================================================

REPO_ID = "nipunbatra/llm-from-scratch"
MODELS = {}

def load_model_file(name, filename):
    """Try to load a model from HF Hub, fallback to local."""
    # Try local paths first (for development)
    local_paths = [
        f"../models/{filename}",
        f"models/{filename}",
        f"/home/nipun.batra/git/llm-from-scratch/models/{filename}",
    ]

    for local_path in local_paths:
        if os.path.exists(local_path):
            checkpoint = torch.load(local_path, map_location=device, weights_only=False)
            print(f"Loaded {name} from {local_path}")
            return checkpoint

    # Try HuggingFace Hub
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=f"models/{filename}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        print(f"Loaded {name} from HuggingFace Hub")
        return checkpoint
    except Exception as e:
        print(f"Could not load {name}: {e}")
        return None


def load_models():
    """Load all available models."""
    global MODELS

    model_files = {
        "names": "char_lm_names.pt",
        "shakespeare": "transformer_shakespeare.pt",
        "instruction": "instruction_tuned.pt",
    }

    for name, filename in model_files.items():
        checkpoint = load_model_file(name, filename)
        if checkpoint:
            MODELS[name] = checkpoint


def get_char_model(checkpoint):
    """Instantiate CharLM from checkpoint."""
    model = CharLM(
        vocab_size=checkpoint['vocab_size'],
        block_size=checkpoint['block_size'],
        emb_dim=checkpoint['emb_dim'],
        hidden_size=checkpoint['hidden_size']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_transformer_model(checkpoint):
    """Instantiate TransformerLM from checkpoint."""
    config = checkpoint['model_config']
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        block_size=config['block_size'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# ============================================================
# Generation Functions
# ============================================================

@torch.no_grad()
def generate_name(temperature=0.8, num_names=5):
    """Generate Indian names using the character-level model."""
    if "names" not in MODELS:
        return "Model not loaded. Please check the models are available."

    checkpoint = MODELS["names"]
    model = get_char_model(checkpoint)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    block_size = checkpoint['block_size']

    names = []
    for _ in range(num_names):
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

        names.append(''.join(name).capitalize())

    return '\n'.join(names)


@torch.no_grad()
def generate_shakespeare(prompt, max_length=200, temperature=0.8):
    """Generate Shakespeare-style text."""
    if "shakespeare" not in MODELS:
        return "Model not loaded. Please check the models are available."

    checkpoint = MODELS["shakespeare"]
    model = get_transformer_model(checkpoint)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    block_size = checkpoint['model_config']['block_size']

    # Handle unknown characters
    tokens = [stoi.get(ch, 0) for ch in prompt]
    generated = list(prompt)

    for _ in range(max_length):
        context = tokens[-block_size:] if len(tokens) >= block_size else tokens
        x = torch.tensor([context]).to(device)

        logits = model(x)
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        tokens.append(next_idx)
        generated.append(itos[next_idx])

    return ''.join(generated)


@torch.no_grad()
def answer_question(question, max_length=150, temperature=0.7):
    """Answer a question using the instruction-tuned model."""
    if "instruction" not in MODELS:
        return "Instruction model not loaded."

    checkpoint = MODELS["instruction"]
    model = get_transformer_model(checkpoint)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    block_size = checkpoint['model_config']['block_size']

    # Format as instruction
    prompt = f"Q: {question}\nA:"
    tokens = [stoi.get(ch, 0) for ch in prompt]
    generated = list(prompt)

    for _ in range(max_length):
        context = tokens[-block_size:] if len(tokens) >= block_size else tokens
        x = torch.tensor([context]).to(device)

        logits = model(x)
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        tokens.append(next_idx)
        char = itos[next_idx]
        generated.append(char)

        # Stop at newline (end of answer)
        if char == '\n' and len(generated) > len(prompt) + 10:
            break

    return ''.join(generated)


# ============================================================
# Gradio Interface
# ============================================================

def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="LLM From Scratch",
        theme=gr.themes.Base(
            primary_hue="neutral",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .container { max-width: 800px; margin: auto; }
        .title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0.5rem; }
        """
    ) as demo:

        gr.Markdown(
            """
            # LLM From Scratch

            Interactive demos of models trained in the [educational series](https://nipunbatra.github.io/llm-from-scratch/).
            These are tiny models (~600K parameters) trained for learning purposes.
            """,
            elem_classes=["title"]
        )

        with gr.Tabs():

            # Tab 1: Name Generation
            with gr.TabItem("Name Generator"):
                gr.Markdown(
                    """
                    ### Part 1: Character-Level Language Model
                    Generate Indian names using a simple neural network trained on 6,000+ names.
                    """
                )

                with gr.Row():
                    with gr.Column():
                        name_temp = gr.Slider(
                            minimum=0.3, maximum=1.5, value=0.8, step=0.1,
                            label="Temperature",
                            info="Lower = more common names, Higher = more creative"
                        )
                        name_count = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of names"
                        )
                        name_btn = gr.Button("Generate Names", variant="primary")

                    with gr.Column():
                        name_output = gr.Textbox(
                            label="Generated Names",
                            lines=10,
                            show_copy_button=True
                        )

                name_btn.click(
                    fn=generate_name,
                    inputs=[name_temp, name_count],
                    outputs=name_output
                )

            # Tab 2: Shakespeare Generation
            with gr.TabItem("Shakespeare"):
                gr.Markdown(
                    """
                    ### Part 4: Transformer Language Model
                    Generate Shakespeare-style text using a small transformer trained on the Bard's works.
                    """
                )

                with gr.Row():
                    with gr.Column():
                        shakespeare_prompt = gr.Textbox(
                            label="Prompt",
                            value="ROMEO:\n",
                            lines=2,
                            info="Start with a character name like ROMEO: or JULIET:"
                        )
                        shakespeare_length = gr.Slider(
                            minimum=50, maximum=500, value=200, step=50,
                            label="Max length"
                        )
                        shakespeare_temp = gr.Slider(
                            minimum=0.3, maximum=1.5, value=0.8, step=0.1,
                            label="Temperature"
                        )
                        shakespeare_btn = gr.Button("Generate Text", variant="primary")

                    with gr.Column():
                        shakespeare_output = gr.Textbox(
                            label="Generated Text",
                            lines=15,
                            show_copy_button=True
                        )

                shakespeare_btn.click(
                    fn=generate_shakespeare,
                    inputs=[shakespeare_prompt, shakespeare_length, shakespeare_temp],
                    outputs=shakespeare_output
                )

            # Tab 3: Q&A (Instruction Tuned)
            with gr.TabItem("Q&A"):
                gr.Markdown(
                    """
                    ### Part 5: Instruction-Tuned Model
                    Ask questions and get answers from a model fine-tuned on Q&A pairs.
                    Try Python, algorithms, or general knowledge questions.
                    """
                )

                with gr.Row():
                    with gr.Column():
                        qa_question = gr.Textbox(
                            label="Question",
                            value="What is a Python list?",
                            lines=2,
                            info="Try: 'What is recursion?', 'How do I reverse a string?'"
                        )
                        qa_temp = gr.Slider(
                            minimum=0.3, maximum=1.2, value=0.7, step=0.1,
                            label="Temperature"
                        )
                        qa_btn = gr.Button("Ask", variant="primary")

                    with gr.Column():
                        qa_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            show_copy_button=True
                        )

                gr.Examples(
                    examples=[
                        ["What is a Python list?"],
                        ["How do I reverse a string in Python?"],
                        ["What is recursion?"],
                        ["What is a for loop?"],
                        ["How do I read a file in Python?"],
                    ],
                    inputs=[qa_question],
                )

                qa_btn.click(
                    fn=answer_question,
                    inputs=[qa_question, gr.State(150), qa_temp],
                    outputs=qa_output
                )

            # Tab 4: About
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ## About This Demo

                    This app showcases models trained in the **LLM From Scratch** educational series.

                    ### The Series

                    | Part | Topic | Model |
                    |------|-------|-------|
                    | 1 | Character-Level LM | CharLM (~18K params) |
                    | 2 | Shakespeare | CharLM (~50K params) |
                    | 3 | BPE Tokenization | - |
                    | 4 | Transformers | TransformerLM (~600K params) |
                    | 5 | Instruction Tuning | InstructionLM |
                    | 6 | DPO Alignment | AlignedLM |
                    | 7 | Code Generation | CodeLM |

                    ### Model Scale

                    ```
                    These models:   ~600K parameters
                    GPT-2 Small:    124M parameters (200x larger)
                    GPT-4/Claude:   ~1T+ parameters (1,600,000x larger)
                    ```

                    Despite being tiny, these models demonstrate all the core concepts!

                    ### Links

                    - [Tutorial Series](https://nipunbatra.github.io/llm-from-scratch/)
                    - [GitHub Repository](https://github.com/nipunbatra/llm-from-scratch)
                    - [Author: Nipun Batra](https://nipunbatra.github.io/)
                    """
                )

        gr.Markdown(
            """
            ---
            Built with PyTorch and Gradio. Part of the [LLM From Scratch](https://github.com/nipunbatra/llm-from-scratch) series.
            """,
            elem_classes=["subtitle"]
        )

    return demo


if __name__ == "__main__":
    print(f"Using device: {device}")
    load_models()
    demo = create_demo()
    demo.launch()
