"""
Example model configurations.

This file contains preset configurations for different model sizes,
from tiny (for learning/testing) to large (for serious applications).

You can use these as starting points and adjust based on your needs
and computational resources.
"""

# Tiny model - for quick experimentation and learning
TINY_CONFIG = {
    'name': 'tiny',
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'max_seq_len': 128,
    'dropout': 0.1,
    'batch_size': 16,
    'learning_rate': 5e-4,
    'description': 'Tiny model for quick experiments (~200K params)',
}

# Small model - good for learning on limited hardware
SMALL_CONFIG = {
    'name': 'small',
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 4,
    'max_seq_len': 256,
    'dropout': 0.1,
    'batch_size': 8,
    'learning_rate': 3e-4,
    'description': 'Small model for learning (~800K params)',
}

# Medium model - balanced performance
MEDIUM_CONFIG = {
    'name': 'medium',
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 6,
    'max_seq_len': 512,
    'dropout': 0.1,
    'batch_size': 4,
    'learning_rate': 3e-4,
    'description': 'Medium model with decent capacity (~10M params)',
}

# Large model - requires good GPU
LARGE_CONFIG = {
    'name': 'large',
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 8,
    'max_seq_len': 1024,
    'dropout': 0.1,
    'batch_size': 2,
    'learning_rate': 2e-4,
    'description': 'Large model, needs GPU (~50M params)',
}

# GPT-2 Small style
GPT2_SMALL_CONFIG = {
    'name': 'gpt2-small',
    'd_model': 768,
    'num_heads': 12,
    'num_layers': 12,
    'max_seq_len': 1024,
    'dropout': 0.1,
    'batch_size': 1,
    'learning_rate': 2e-4,
    'description': 'Similar to GPT-2 Small (~117M params)',
}

# All configs
CONFIGS = {
    'tiny': TINY_CONFIG,
    'small': SMALL_CONFIG,
    'medium': MEDIUM_CONFIG,
    'large': LARGE_CONFIG,
    'gpt2-small': GPT2_SMALL_CONFIG,
}


def get_config(name: str) -> dict:
    """
    Get configuration by name.
    
    Args:
        name: Configuration name ('tiny', 'small', 'medium', 'large', 'gpt2-small')
        
    Returns:
        Configuration dictionary
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name].copy()


def print_all_configs():
    """Print all available configurations."""
    print("=" * 70)
    print("AVAILABLE MODEL CONFIGURATIONS")
    print("=" * 70)
    
    for name, config in CONFIGS.items():
        print(f"\n{config['name'].upper()}")
        print("-" * 70)
        print(f"Description: {config['description']}")
        print(f"Parameters:")
        print(f"  • Model dimension: {config['d_model']}")
        print(f"  • Attention heads: {config['num_heads']}")
        print(f"  • Layers: {config['num_layers']}")
        print(f"  • Max sequence length: {config['max_seq_len']}")
        print(f"  • Dropout: {config['dropout']}")
        print(f"  • Batch size: {config['batch_size']}")
        print(f"  • Learning rate: {config['learning_rate']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    print_all_configs()
    
    print("\n### Example Usage ###\n")
    print("from examples.configs import get_config")
    print("from src.llm.model import GPTModel")
    print()
    print("config = get_config('small')")
    print("model = GPTModel(")
    print("    vocab_size=1000,")
    print("    d_model=config['d_model'],")
    print("    num_heads=config['num_heads'],")
    print("    num_layers=config['num_layers'],")
    print("    max_seq_len=config['max_seq_len'],")
    print("    dropout=config['dropout']")
    print(")")
