# Getting Started Guide 🚀

This guide will walk you through your first steps with building your own LLM!

## Prerequisites

- Python 3.8 or higher
- Basic understanding of Python
- (Optional) CUDA-capable GPU for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/blatour/my_own_llm.git
cd my_own_llm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **tqdm**: Progress bars
- **regex**: Advanced text processing

## Your First LLM

### Step 1: Train a Simple Model

Let's train a tiny model on sample text:

```bash
python examples/train.py
```

This will:
1. Create sample training data (repeated text about transformers and ML)
2. Train a small model (~500K parameters)
3. Save checkpoints every epoch
4. Save the final model to `checkpoints/`

**Expected time**: 2-5 minutes on CPU, < 1 minute on GPU

**Output**:
```
Training on cpu
Model parameters: 530,432
Training samples: 4,472
Batches per epoch: 559

Epoch 1/5
Training: 100%|████████████| 559/559
Average loss: 2.8234

...

Training complete!
Model saved to: checkpoints/final_model.pt
```

### Step 2: Generate Text

Now let's generate text with your trained model:

```bash
python examples/generate.py
```

This will:
1. Load your trained model
2. Show generation examples with different parameters
3. Enter interactive mode where you can type prompts

**Example output**:
```
### Example 1: Low temperature (0.5) - More focused
Prompt: 'The Transformer'

Generated:
The Transformer architecture has revolutionized natural language processing...

### Example 2: High temperature (1.2) - More creative
Prompt: 'Machine learning'

Generated:
Machine learning is the study of algorithms that improve through experience...
```

### Step 3: Explore the Model

Learn about the model's internals:

```bash
python examples/explore_model.py
```

Choose from:
1. **Model Architecture** - See how the model is structured
2. **Tokenization** - Understand how text becomes numbers
3. **Forward Pass** - Watch data flow through the model
4. **Embeddings** - Explore the embedding space

## What's Next?

### 1. Train on Your Own Text

Modify `examples/train.py` to use your own data:

```python
# Replace this line in train.py:
data_path = 'path/to/your/text.txt'
```

Try training on:
- **Books**: Classic literature (Project Gutenberg)
- **Code**: GitHub repositories
- **Your writing**: Personal notes, blog posts
- **Song lyrics**: Your favorite artist
- **Poetry**: Shakespeare, Wordsworth

### 2. Experiment with Model Size

Edit the config in `examples/train.py`:

```python
# Tiny model (fast, ~100K params)
config = {
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'max_seq_len': 128,
}

# Medium model (slower, ~10M params)
config = {
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 6,
    'max_seq_len': 512,
}
```

### 3. Play with Generation Parameters

In `examples/generate.py` or interactive mode:

```python
# More creative/random
temperature = 1.5

# More focused/deterministic
temperature = 0.5

# Use top-k sampling
top_k = 40
```

### 4. Try Different Tokenizers

Use BPE tokenizer for better efficiency:

```python
from src.llm.tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(text, verbose=True)
```

## Understanding the Results

### Loss

The loss measures how well the model predicts next tokens:
- **High loss (>3.0)**: Model is struggling, barely better than random
- **Medium loss (1.0-3.0)**: Model is learning patterns
- **Low loss (<1.0)**: Model is doing well on training data

### Generation Quality

What to expect from different model sizes:

**Tiny (100K-1M params)**
- Learns basic character patterns
- Can spell words correctly
- Limited coherence beyond a few words

**Small (1M-10M params)**
- Learns word patterns
- Can form simple sentences
- Limited long-range coherence

**Medium (10M-100M params)**
- Good sentence structure
- Can maintain topic for a paragraph
- Starts showing creativity

**Large (100M+ params)**
- Strong coherence
- Good grammar and style
- Can write multiple coherent paragraphs

## Common Issues

### Out of Memory

If you run out of memory:

1. **Reduce batch size**: Change `batch_size` from 8 to 4 or 2
2. **Smaller model**: Reduce `d_model`, `num_layers`
3. **Shorter sequences**: Reduce `max_seq_len`

### Slow Training

Training too slow?

1. **Use GPU**: Install CUDA and PyTorch with GPU support
2. **Smaller model**: Reduce parameters
3. **Less data**: Use subset of your dataset

### Poor Generation Quality

Model not generating good text?

1. **Train longer**: Increase `num_epochs`
2. **More data**: Use larger/better training corpus
3. **Larger model**: Increase `d_model`, `num_layers`
4. **Better tokenizer**: Try BPE instead of character-level

## Tips for Learning

### Start Small

Begin with tiny models to understand the concepts:
- Faster iteration
- Easier to debug
- Can run on any computer

### Read the Code

The code is heavily commented. Read:
1. `src/llm/tokenizer.py` - How text becomes numbers
2. `src/llm/attention.py` - The "magic" of transformers
3. `src/llm/model.py` - Complete architecture

### Experiment

Try changing things and see what happens:
- What if you remove attention?
- What if you use more heads?
- What if you change the activation function?

### Track Your Experiments

Keep notes:
```
Experiment 1: Tiny model on Shakespeare
- Config: d_model=64, layers=2
- Data: 1MB of Shakespeare plays
- Results: Loss 1.8, generates Shakespearean-style words

Experiment 2: ...
```

## Next Steps

1. **Read CONCEPTS.md** - Deep dive into how LLMs work
2. **Explore examples/** - More advanced usage
3. **Modify the code** - Add features, optimize
4. **Train bigger models** - If you have the compute
5. **Share your results** - What interesting things did you create?

## Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2

### Videos
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown: Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)

### Interactive
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)

## Get Help

- **Issues**: Open a GitHub issue
- **Questions**: Use GitHub Discussions
- **Bugs**: Report with minimal reproduction example

---

**Happy Learning!** 🎓

The best way to learn is by doing. Don't be afraid to break things - that's how you learn!
