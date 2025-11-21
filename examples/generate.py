"""
Example text generation script.

This script demonstrates how to:
1. Load a trained model
2. Generate text with different parameters
3. Experiment with temperature and top-k sampling

Usage:
    python examples/generate.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pickle
from src.llm.model import GPTModel
from src.llm.utils import generate_text


def main():
    # Paths
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: No trained model found!")
        print(f"Expected model at: {model_path}")
        print("\nPlease run examples/train.py first to train a model.")
        return
    
    print("=" * 60)
    print("Text Generation Example")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model (must match training config)
    print("\nLoading model...")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    )
    
    # Load trained weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch cannot find a CUDA-enabled GPU.")
        print("Please ensure you have installed the correct PyTorch version with CUDA support and that your NVIDIA drivers are up to date.")


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Using device: {device}")
    
    # Generation examples
    prompts = [
        "The Transformer",
        "Machine learning",
        "Deep learning",
        "",  # Empty prompt - start from scratch
    ]
    
    print("\n" + "=" * 60)
    print("GENERATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Low temperature (more deterministic)
    print("\n### Example 1: Low temperature (0.5) - More focused")
    print("-" * 60)
    prompt = prompts[0]
    print(f"Prompt: '{prompt}'")
    print("\nGenerated:")
    text = generate_text(
        model, tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.5,
        device=device
    )
    print(text)
    
    # Example 2: High temperature (more random)
    print("\n### Example 2: High temperature (1.2) - More creative")
    print("-" * 60)
    prompt = prompts[1]
    print(f"Prompt: '{prompt}'")
    print("\nGenerated:")
    text = generate_text(
        model, tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=1.2,
        device=device
    )
    print(text)
    
    # Example 3: With top-k sampling
    print("\n### Example 3: Top-k sampling (k=40)")
    print("-" * 60)
    prompt = prompts[2]
    print(f"Prompt: '{prompt}'")
    print("\nGenerated:")
    text = generate_text(
        model, tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=1.0,
        top_k=40,
        device=device
    )
    print(text)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nYou can now generate text interactively!")
    print("Enter a prompt (or press Enter for empty prompt)")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            prompt = input("Prompt: ")
            
            if prompt.lower() == 'quit':
                break
            
            # Get generation parameters
            try:
                temp_input = input("Temperature (default 1.0): ").strip()
                temperature = float(temp_input) if temp_input else 1.0
            except ValueError:
                temperature = 1.0
            
            try:
                tokens_input = input("Max tokens (default 100): ").strip()
                max_tokens = int(tokens_input) if tokens_input else 100
            except ValueError:
                max_tokens = 100
            
            # Generate
            print("\nGenerating...")
            text = generate_text(
                model, tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=device
            )
            
            print("\nGenerated text:")
            print("-" * 60)
            print(text)
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")
    
    print("\nThank you for using the text generator!")


if __name__ == '__main__':
    main()
