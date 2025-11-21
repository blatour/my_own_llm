"""
Utility functions for data loading, generation, and evaluation.
"""

import torch
import os
from typing import Optional


def load_text_file(file_path: str) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Text content as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_text_file(text: str, file_path: str):
    """
    Save text to a file.
    
    Args:
        text: Text to save
        file_path: Path to save to
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def generate_text(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """
    Generate text using the model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer to encode/decode text
        prompt: Starting text (can be empty)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k most likely tokens
        device: Device to run on
        
    Returns:
        Generated text
    """
    model.eval()
    model.to(device)
    
    # Encode prompt
    if prompt:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        # Start with a random token or BOS token
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def count_tokens(text: str, tokenizer) -> int:
    """
    Count number of tokens in text.
    
    Args:
        text: Text to count tokens in
        tokenizer: Tokenizer to use
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def estimate_model_size(model) -> dict:
    """
    Estimate model size and memory usage.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'size_mb': size_mb,
        'size_gb': size_mb / 1024
    }


def print_model_info(model):
    """
    Print detailed model information.
    
    Args:
        model: PyTorch model
    """
    info = estimate_model_size(model)
    
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Total parameters: {info['parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Model size: {info['size_mb']:.2f} MB ({info['size_gb']:.4f} GB)")
    print("=" * 60)
    
    if hasattr(model, 'd_model'):
        print(f"Model dimension: {model.d_model}")
    if hasattr(model, 'num_layers'):
        print(f"Number of layers: {len(model.blocks) if hasattr(model, 'blocks') else 'N/A'}")
    if hasattr(model, 'vocab_size'):
        print(f"Vocabulary size: {model.vocab_size}")
    if hasattr(model, 'max_seq_len'):
        print(f"Max sequence length: {model.max_seq_len}")
    print("=" * 60)
