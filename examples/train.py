"""
Example training script for the LLM.

This script demonstrates how to:
1. Load and prepare training data
2. Initialize a tokenizer
3. Create a model
4. Train the model
5. Save the trained model

Usage:
    python examples/train.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.llm.tokenizer import CharacterTokenizer
from src.llm.model import GPTModel
from src.llm.trainer import TextDataset, Trainer
from src.llm.utils import load_text_file, print_model_info


def main():
    # Configuration
    # For learning purposes, we use small model sizes
    config = {
        'd_model': 128,          # Small embedding dimension
        'num_heads': 4,          # 4 attention heads
        'num_layers': 4,         # 4 transformer blocks
        'max_seq_len': 256,      # Maximum sequence length
        'dropout': 0.1,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'num_epochs': 5,
        'seq_len': 128,          # Training sequence length
    }
    
    print("=" * 60)
    print("LLM Training Example")
    print("=" * 60)
    
    # Load training data
    # For this example, we'll use a sample text file
    # You can replace this with your own dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.txt')
    
    if not os.path.exists(data_path):
        print(f"\nCreating sample training data at {data_path}")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create a simple sample text for demonstration
        sample_text = """The Transformer architecture has revolutionized natural language processing.
It uses self-attention mechanisms to process sequences in parallel, making it much faster than recurrent networks.
The key innovation is the attention mechanism, which allows the model to focus on relevant parts of the input.
GPT models are decoder-only transformers that generate text autoregressively.
They are trained on large amounts of text data to predict the next token in a sequence.
This simple objective leads to surprisingly sophisticated language understanding and generation capabilities.
Machine learning is the study of algorithms that improve through experience.
Deep learning uses neural networks with multiple layers to learn hierarchical representations.
Language models learn statistical patterns in text data.
The more data and compute you have, the better your model can perform.
"""
        with open(data_path, 'w', encoding='utf-8') as f:
            # Repeat the text multiple times to have more training data
            f.write(sample_text * 50)
        print(f"Sample data created with {len(sample_text) * 50} characters")
    
    text = load_text_file(data_path)
    print(f"\nLoaded training data: {len(text)} characters")
    
    # Initialize tokenizer
    print("\nInitializing character-level tokenizer...")
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Tokenize the text
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids)}")
    
    # Create dataset
    dataset = TextDataset(token_ids, seq_len=config['seq_len'])
    print(f"Training examples: {len(dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    print_model_info(model)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        device=device
    )
    
    # Train
    print("\nStarting training...")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    trainer.train(
        num_epochs=config['num_epochs'],
        checkpoint_dir=checkpoint_dir
    )
    
    # Save final model
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    trainer.save_model(model_path)
    
    # Save tokenizer vocabulary
    import pickle
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved: {tokenizer_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print("\nNext steps:")
    print("1. Run examples/generate.py to generate text with your trained model")
    print("2. Experiment with different hyperparameters")
    print("3. Try training on your own text dataset")


if __name__ == '__main__':
    main()
