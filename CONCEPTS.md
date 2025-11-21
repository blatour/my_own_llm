# LLM Concepts Explained 🧠

This document explains the key concepts behind Large Language Models in simple terms.

## Table of Contents
1. [What is a Language Model?](#what-is-a-language-model)
2. [Tokenization](#tokenization)
3. [Embeddings](#embeddings)
4. [Attention Mechanism](#attention-mechanism)
5. [Transformer Architecture](#transformer-architecture)
6. [Training Process](#training-process)
7. [Text Generation](#text-generation)
8. [Why These Choices?](#why-these-choices)

---

## What is a Language Model?

A **language model** is a system that learns the statistical patterns of language. Given some text, it tries to predict what comes next.

### Simple Example
```
Input:  "The cat sat on the"
Output: "mat" (most likely)
        "floor", "chair", "table" (also possible)
```

### How It Works
1. Train on lots of text (books, websites, articles)
2. Learn patterns: which words follow which words
3. Use these patterns to generate new text or understand existing text

---

## Tokenization

**Problem**: Computers work with numbers, not text.

**Solution**: Convert text to numbers (tokens).

### Character-Level Tokenization
Simplest approach: each character gets a number.

```
"Hello" → [72, 101, 108, 108, 111]
  H        e     l     l     o
```

**Pros**: 
- Simple, small vocabulary
- Can represent any text

**Cons**: 
- Long sequences (one number per character)
- Doesn't capture word meaning

### Byte Pair Encoding (BPE)
Smarter approach: learn common subwords.

```
"unhappiness" → ["un", "happi", "ness"]
```

**How it works**:
1. Start with characters
2. Find most common adjacent pairs
3. Merge them into new tokens
4. Repeat until you have desired vocabulary size

**Pros**:
- Shorter sequences than characters
- Handles rare words (breaks into known parts)
- Balances vocabulary size and sequence length

**Cons**:
- More complex than character-level

### Why It Matters
- Smaller vocabulary = less parameters
- Shorter sequences = faster processing
- Better tokenization = better model performance

---

## Embeddings

**Problem**: Token IDs (integers) don't capture meaning.

**Solution**: Convert each token to a dense vector of numbers.

### Example
```
Token: "cat" (ID: 42)
Embedding: [0.2, -0.5, 0.8, 0.1, ..., 0.3]  # 768 numbers
```

### Why Vectors?
Vectors can capture:
- Semantic meaning (cat ≈ dog)
- Relationships (king - man + woman ≈ queen)
- Context (bank: money vs. river)

### Two Types

**1. Token Embeddings**
- Maps each token to a vector
- Learned during training
- Shape: (vocab_size, d_model)

**2. Positional Embeddings**
- Adds position information
- "cat" at position 1 vs. position 10 should be different
- Shape: (max_seq_len, d_model)

### Combined
```python
# For each position:
final_embedding = token_embedding + positional_embedding
```

---

## Attention Mechanism

**The "magic" of transformers!**

### The Problem
How does the model know which words are related?

```
"The animal didn't cross the street because it was too tired."
```

What does "it" refer to? The animal or the street?

### The Solution: Attention

Attention lets each word "look at" all other words and decide which are important.

```
"it" pays attention to → "animal" (high score)
"it" pays attention to → "street" (low score)
```

### How It Works

1. **Query, Key, Value** (Q, K, V)
   - Each word creates three vectors
   - Query: "what am I looking for?"
   - Key: "what do I contain?"
   - Value: "what information do I have?"

2. **Attention Scores**
   ```
   score = Query @ Key^T / sqrt(d_k)
   attention_weights = softmax(scores)
   ```
   
3. **Weighted Sum**
   ```
   output = attention_weights @ Values
   ```

### Multi-Head Attention

Run attention multiple times in parallel with different "perspectives":
- Head 1: Focuses on grammar
- Head 2: Focuses on semantics
- Head 3: Focuses on syntax
- etc.

Each head learns different patterns!

### Causal (Masked) Attention

For text generation, can't look at future words:

```
Position 0 can see: [0]
Position 1 can see: [0, 1]
Position 2 can see: [0, 1, 2]
Position 3 can see: [0, 1, 2, 3]
```

This makes generation autoregressive (one token at a time).

---

## Transformer Architecture

Combines attention with other components:

### Transformer Block
```
Input
  ↓
LayerNorm → Multi-Head Attention → Add (residual)
  ↓
LayerNorm → Feed-Forward Network → Add (residual)
  ↓
Output
```

### Key Components

**1. Layer Normalization**
- Stabilizes training
- Normalizes inputs to have mean=0, std=1
- Applied before each sub-layer (pre-norm)

**2. Residual Connections**
- `output = input + layer(input)`
- Helps gradients flow during training
- Allows deeper networks

**3. Feed-Forward Network**
- 2-layer MLP applied to each position
- Expansion: `d_model → 4*d_model → d_model`
- Processes the attended information

### Stack of Blocks

GPT models stack many blocks:
- More layers = more capacity to learn patterns
- But also harder to train and slower

```
Input Embeddings
  ↓
Block 1
  ↓
Block 2
  ↓
Block 3
  ↓
...
  ↓
Block N
  ↓
Output Projection
```

---

## Training Process

### Objective: Next Token Prediction

Train the model to predict the next token:

```
Given: "The cat sat on"
Predict: "the"

Given: "The cat sat on the"
Predict: "mat"
```

### Loss Function

Compare predictions to actual next tokens:
```python
loss = CrossEntropyLoss(predicted_logits, actual_tokens)
```

Lower loss = better predictions

### Training Loop

```python
for epoch in epochs:
    for batch in data:
        # Forward pass
        predictions = model(input_tokens)
        
        # Compute loss
        loss = criterion(predictions, target_tokens)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
```

### Why This Works

By learning to predict next tokens:
- Model learns grammar
- Model learns facts
- Model learns reasoning patterns
- Model learns style and tone

All from the simple objective of "predict the next word"!

### Hyperparameters

**Learning Rate**: How big are weight updates?
- Too high: Training unstable
- Too low: Training too slow
- Typical: 1e-4 to 5e-4

**Batch Size**: How many examples per update?
- Larger: More stable, but needs more memory
- Smaller: More updates, but noisier
- Typical: 4-32 (depends on model size)

**Gradient Clipping**: Prevent exploding gradients
- Clip gradient norm to max value (e.g., 1.0)
- Prevents training instability

---

## Text Generation

### Autoregressive Generation

Generate one token at a time:

```
1. Start: "The"
2. Predict next: "cat"  → "The cat"
3. Predict next: "sat"  → "The cat sat"
4. Predict next: "on"   → "The cat sat on"
5. ...continue...
```

### Sampling Strategies

**1. Greedy Sampling**
- Always pick the most likely token
- Deterministic but can be repetitive
```python
next_token = argmax(probabilities)
```

**2. Temperature Sampling**
- Control randomness with temperature
```python
logits = logits / temperature
probabilities = softmax(logits)
next_token = sample(probabilities)
```
- Temperature = 0.1: Very focused (almost greedy)
- Temperature = 1.0: Normal randomness
- Temperature = 2.0: Very random

**3. Top-k Sampling**
- Only sample from top k most likely tokens
```python
top_k_probs = get_top_k(probabilities, k=40)
next_token = sample(top_k_probs)
```
- Prevents sampling very unlikely tokens
- Balances quality and diversity

**4. Top-p (Nucleus) Sampling**
- Sample from smallest set of tokens with cumulative probability > p
- Adaptive: sometimes few tokens, sometimes many
- Used in GPT-3

---

## Why These Choices?

### Why Decoder-Only (GPT-style)?

**Other options**:
- Encoder-only (BERT): Good for understanding, not generation
- Encoder-Decoder (T5): Good for translation/summarization
- Decoder-only (GPT): Best for generation

**Why decoder-only?**
- Simple architecture
- Great for text generation
- Scales well with size
- Can also do understanding tasks

### Why Causal Attention?

Forces model to generate left-to-right:
- More challenging (can't cheat by looking ahead)
- Necessary for autoregressive generation
- Makes the model learn true language modeling

### Why Pre-Norm?

LayerNorm before layer (not after):
- More stable training
- Allows training deeper models
- Modern best practice

### Why Tied Embeddings?

Token embeddings and output projection share weights:
- Fewer parameters
- Better performance
- Makes sense: both map between tokens and vectors

### Why These Activation Functions?

**GELU (Gaussian Error Linear Unit)**
- Smoother than ReLU
- Better gradient flow
- Used in GPT-2/3

**Softmax**
- Converts logits to probabilities
- Ensures probabilities sum to 1

### Why AdamW Optimizer?

- Adaptive learning rates per parameter
- Weight decay for regularization
- Works well for transformers
- Standard choice in NLP

---

## Scaling Laws

Bigger models are generally better, but with diminishing returns:

**Model Size (parameters)**
```
1M    → Basic patterns
10M   → Simple coherence
100M  → Good paragraphs
1B    → Strong language understanding
10B+  → Near-human performance on many tasks
```

**Training Data**
```
1MB   → Very limited
1GB   → Basic language patterns
10GB  → Good language understanding
100GB → Strong performance
1TB+  → State-of-the-art (GPT-3 trained on ~500GB)
```

**Compute**
```
More compute = Better model (for same data/params)
But: Expensive and time-consuming
```

### The Trade-off

Balance three factors:
1. Model size (parameters)
2. Data size (training examples)
3. Compute (training time)

For learning: Small model, small data, small compute
For production: Large model, large data, large compute

---

## Further Reading

### Papers
- **Attention Is All You Need** - Original Transformer
- **GPT-2: Language Models are Unsupervised Multitask Learners**
- **GPT-3: Language Models are Few-Shot Learners**
- **Scaling Laws for Neural Language Models**

### Tutorials
- Andrej Karpathy's "Let's build GPT" (YouTube)
- The Illustrated Transformer (Jay Alammar)
- The Illustrated GPT-2 (Jay Alammar)

### Code
- Hugging Face Transformers library
- PyTorch tutorials
- This repository!

---

**Remember**: The best way to learn is by experimenting! Try:
- Changing hyperparameters
- Training on different data
- Adding new features
- Breaking things and fixing them

Happy learning! 🚀
