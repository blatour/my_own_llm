# My Own LLM - Build an LLM from Scratch 🚀

A learning-focused implementation of a Large Language Model (LLM) built from scratch in Python. This project is designed to help you understand how LLMs work, from tokenization to transformer architecture to text generation.

## 🎯 Purpose

This repository is a hands-on learning experience to understand:
- How text is converted to numbers (tokenization)
- How transformers process sequences (attention mechanism)
- How language models are trained (next token prediction)
- What architectural choices matter (model size, layers, heads)
- How to generate text (sampling strategies)

## 🏗️ Architecture Overview

This project implements a **GPT-style decoder-only transformer**, similar to GPT-2/GPT-3 but simplified for learning:

```
Input Text
    ↓
Tokenizer (Character or BPE)
    ↓
Token Embeddings + Positional Embeddings
    ↓
Transformer Blocks (x N)
├── Multi-Head Self-Attention
├── Layer Normalization
├── Feed-Forward Network
└── Residual Connections
    ↓
Output Projection → Vocabulary Logits
    ↓
Generated Text
```

### Key Components

1. **Tokenizer** (`src/llm/tokenizer.py`)
   - **CharacterTokenizer**: Simple character-level tokenization
   - **BPETokenizer**: Byte Pair Encoding for efficient subword tokenization

2. **Attention Mechanism** (`src/llm/attention.py`)
   - Multi-head self-attention
   - Causal masking for autoregressive generation
   - Scaled dot-product attention

3. **Model Architecture** (`src/llm/model.py`)
   - **GPTModel**: Complete transformer implementation
   - **TransformerBlock**: Attention + Feed-forward with residuals
   - **FeedForward**: Position-wise MLP

4. **Training** (`src/llm/trainer.py`)
   - Dataset preparation
   - Training loop with gradient clipping
   - Checkpointing

5. **Utilities** (`src/llm/utils.py`)
   - Text generation with temperature and top-k sampling
   - Model size estimation
   - Helper functions

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm
- regex

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/blatour/my_own_llm.git
cd my_own_llm

# Install dependencies
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Train on sample data (included in the script)
python examples/train.py
```

This will:
- Create sample training data
- Initialize a character-level tokenizer
- Create a small GPT model (~500K parameters)
- Train for 5 epochs
- Save the model to `checkpoints/`

### 3. Generate Text

```bash
# Generate text with the trained model
python examples/generate.py
```

This will:
- Load your trained model
- Show generation examples with different parameters
- Enter interactive mode for custom prompts

## 📚 Understanding the Code

### How Training Works

Language models are trained with a simple objective: **predict the next token**.

```python
# Example: "Hello World"
# Input:  [H, e, l, l, o, _, W, o, r]
# Target: [e, l, l, o, _, W, o, r, l, d]
```

The model learns to predict each next token, which teaches it:
- Grammar and syntax
- Word relationships
- Semantic meaning
- Style and patterns

### Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `d_model` | Model dimension (embedding size) | 128-768 (small), 1024-2048 (medium), 4096+ (large) |
| `num_heads` | Number of attention heads | 4-12 (must divide d_model) |
| `num_layers` | Number of transformer blocks | 4-12 (small), 12-24 (medium), 24+ (large) |
| `max_seq_len` | Maximum sequence length | 128-512 (small), 1024-2048 (medium), 4096+ (large) |
| `vocab_size` | Number of tokens in vocabulary | 50-1000 (char), 10K-50K (BPE) |
| `dropout` | Dropout rate for regularization | 0.1-0.2 |
| `learning_rate` | Optimizer learning rate | 1e-4 to 5e-4 |

### Scaling Laws

Model capability generally increases with:
1. **More parameters** (larger d_model, more layers)
2. **More data** (larger training corpus)
3. **More compute** (longer training)

Example model sizes:
- **Tiny** (~1M params): Learning/experimentation
- **Small** (~10M params): Simple text generation
- **Medium** (~100M params): Coherent paragraphs
- **Large** (~1B params): Strong language understanding
- **Very Large** (~10B+ params): GPT-3 class models

## 🎨 Customization Ideas

### 1. Use Your Own Data

Replace the sample text with your own dataset:

```python
# In examples/train.py, modify:
data_path = 'path/to/your/data.txt'
```

Ideas:
- Train on Shakespeare → Generate Shakespearean text
- Train on code → Generate code snippets
- Train on your writing → Mimic your style

### 2. Adjust Model Size

Make the model larger or smaller:

```python
config = {
    'd_model': 256,      # Increase for more capacity
    'num_heads': 8,      # More heads = more attention patterns
    'num_layers': 6,     # Deeper = more complex patterns
    'max_seq_len': 512,  # Longer context
}
```

### 3. Experiment with Generation

Try different sampling strategies:

```python
# Greedy (deterministic)
temperature=0.1

# Balanced
temperature=1.0

# Creative (random)
temperature=1.5

# Focused (top-k)
top_k=40
```

### 4. Try BPE Tokenization

For better efficiency on larger texts:

```python
from src.llm.tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(text, verbose=True)
```

## 🔍 What You'll Learn

### 1. Tokenization
- Why we need to convert text to numbers
- Trade-offs between character and subword tokenization
- How BPE builds a vocabulary

### 2. Embeddings
- Converting discrete tokens to continuous vectors
- Why positional information matters
- Learned vs. fixed positional encodings

### 3. Attention Mechanism
- How attention allows "looking at" other tokens
- Multi-head attention = multiple perspectives
- Causal masking for autoregressive generation

### 4. Transformer Architecture
- Why residual connections help training
- Layer normalization for stability
- Feed-forward networks for processing

### 5. Training Dynamics
- Next-token prediction objective
- Gradient clipping prevents instability
- Learning rate and batch size effects

### 6. Generation
- Autoregressive sampling
- Temperature controls randomness
- Top-k sampling balances diversity/quality

## 📖 Further Learning

### Recommended Papers
1. **"Attention Is All You Need"** (Vaswani et al., 2017) - Original Transformer
2. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019) - GPT-2
3. **"Language Models are Few-Shot Learners"** (Brown et al., 2020) - GPT-3

### Resources
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features (better tokenizers, optimizations)
- Improve documentation
- Share interesting experiments
- Report issues or suggestions

## 📝 License

MIT License - Feel free to use this for learning!

## 🙏 Acknowledgments

This implementation is inspired by:
- The original Transformer paper
- GPT-2 architecture
- Andrej Karpathy's educational materials
- The PyTorch community

---

**Happy Learning! 🎓** If you found this helpful, give it a ⭐!
