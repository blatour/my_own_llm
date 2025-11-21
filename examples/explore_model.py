"""
Interactive model exploration script.

This script helps you understand what's inside the model:
1. Inspect model architecture
2. Visualize attention patterns (conceptually)
3. Understand token embeddings
4. Explore how the model processes text

Usage:
    python examples/explore_model.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.llm.model import GPTModel
from src.llm.tokenizer import CharacterTokenizer
from src.llm.utils import print_model_info


def explore_architecture():
    """Show model architecture in detail."""
    print("=" * 60)
    print("MODEL ARCHITECTURE EXPLORATION")
    print("=" * 60)
    
    # Create a small model for exploration
    model = GPTModel(
        vocab_size=100,
        d_model=128,
        num_heads=4,
        num_layers=3,
        max_seq_len=256
    )
    
    print("\n### Model Structure ###\n")
    print(model)
    
    print("\n### Detailed Parameter Count ###\n")
    print_model_info(model)
    
    print("\n### Layer-by-Layer Breakdown ###\n")
    
    # Token embedding
    token_emb_params = model.token_embedding.weight.numel()
    print(f"1. Token Embedding: {token_emb_params:,} parameters")
    print(f"   Shape: ({model.vocab_size}, {model.d_model})")
    print(f"   Purpose: Convert token IDs to dense vectors")
    
    # Position embedding
    pos_emb_params = model.position_embedding.weight.numel()
    print(f"\n2. Position Embedding: {pos_emb_params:,} parameters")
    print(f"   Shape: ({model.max_seq_len}, {model.d_model})")
    print(f"   Purpose: Add position information to embeddings")
    
    # Transformer blocks
    print(f"\n3. Transformer Blocks: {len(model.blocks)} blocks")
    for i, block in enumerate(model.blocks):
        block_params = sum(p.numel() for p in block.parameters())
        print(f"\n   Block {i + 1}: {block_params:,} parameters")
        
        # Attention
        attn_params = sum(p.numel() for p in block.attention.parameters())
        print(f"      - Attention: {attn_params:,} params")
        print(f"        • Q, K, V projections: {model.d_model} -> {model.d_model}")
        print(f"        • {model.num_heads} heads × {model.d_model // model.num_heads} dim each")
        
        # Feed-forward
        ff_params = sum(p.numel() for p in block.feed_forward.parameters())
        print(f"      - Feed-Forward: {ff_params:,} params")
        print(f"        • Expansion: {model.d_model} -> {model.d_model * 4} -> {model.d_model}")
    
    # Output projection
    lm_head_params = 0  # Tied with token embedding
    print(f"\n4. Output Projection: {lm_head_params} parameters (tied with token embedding)")
    print(f"   Shape: ({model.d_model}, {model.vocab_size})")
    print(f"   Purpose: Project to vocabulary logits")
    
    print("\n" + "=" * 60)


def explore_tokenization():
    """Demonstrate tokenization."""
    print("=" * 60)
    print("TOKENIZATION EXPLORATION")
    print("=" * 60)
    
    # Sample text
    text = "Hello, World! How are you?"
    
    print(f"\nOriginal text: '{text}'")
    print(f"Length: {len(text)} characters")
    
    # Character tokenization
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    
    print(f"\n### Character Tokenization ###")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {list(tokenizer.char_to_id.keys())}")
    
    tokens = tokenizer.encode(text)
    print(f"\nTokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Show character -> ID mapping
    print("\nCharacter to ID mapping:")
    for char, id in list(tokenizer.char_to_id.items())[:10]:
        print(f"  '{char}' -> {id}")
    
    # Decode back
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded: '{decoded}'")
    print(f"Match original: {decoded == text}")
    
    print("\n" + "=" * 60)


def explore_forward_pass():
    """Show what happens during a forward pass."""
    print("=" * 60)
    print("FORWARD PASS EXPLORATION")
    print("=" * 60)
    
    # Create small model
    vocab_size = 50
    d_model = 64
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        max_seq_len=128
    )
    
    # Sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape} (batch_size={batch_size}, seq_len={seq_len})")
    print(f"Sample input IDs:\n{input_ids[0].tolist()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"  (batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    
    # Show logits for first position
    print(f"\nLogits for position 0, token 0:")
    print(f"  Shape: {logits[0, 0, :].shape}")
    print(f"  Sample values: {logits[0, 0, :5].tolist()}")
    
    # Convert to probabilities
    probs = torch.softmax(logits[0, 0, :], dim=-1)
    print(f"\nProbabilities (after softmax):")
    print(f"  Shape: {probs.shape}")
    print(f"  Sum: {probs.sum().item():.4f} (should be ~1.0)")
    
    # Top 5 predictions
    top_probs, top_ids = torch.topk(probs, k=5)
    print(f"\nTop 5 predicted tokens for position 0:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_ids)):
        print(f"  {i + 1}. Token {idx.item():2d}: {prob.item():.4f} ({prob.item() * 100:.2f}%)")
    
    print("\n" + "=" * 60)


def explore_embeddings():
    """Explore embedding spaces."""
    print("=" * 60)
    print("EMBEDDING SPACE EXPLORATION")
    print("=" * 60)
    
    vocab_size = 26  # a-z
    d_model = 8  # Small for visualization
    
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=1,
        max_seq_len=10
    )
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d_model}")
    
    # Get embeddings for a few tokens
    token_ids = torch.tensor([0, 1, 2, 25])  # a, b, c, z
    embeddings = model.token_embedding(token_ids)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"  (num_tokens={len(token_ids)}, d_model={d_model})")
    
    print("\nSample embeddings:")
    for i, emb in enumerate(embeddings):
        print(f"  Token {token_ids[i]}: {emb.tolist()}")
    
    # Compute similarity between embeddings
    print("\n### Embedding Similarity ###")
    print("(Random initially, learns meaningful relations during training)")
    
    # Cosine similarity
    from torch.nn.functional import cosine_similarity
    
    sim_01 = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    sim_02 = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
    sim_03 = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[3].unsqueeze(0))
    
    print(f"\nCosine similarity:")
    print(f"  Token 0 vs Token 1: {sim_01.item():.4f}")
    print(f"  Token 0 vs Token 2: {sim_02.item():.4f}")
    print(f"  Token 0 vs Token 25: {sim_03.item():.4f}")
    
    print("\n" + "=" * 60)


def main():
    print("\n" + "=" * 60)
    print("LLM MODEL EXPLORATION")
    print("=" * 60)
    print("\nThis script helps you understand the internals of the model.")
    print("Each section explores a different aspect.\n")
    
    sections = [
        ("1", "Model Architecture", explore_architecture),
        ("2", "Tokenization", explore_tokenization),
        ("3", "Forward Pass", explore_forward_pass),
        ("4", "Embeddings", explore_embeddings),
    ]
    
    while True:
        print("\nAvailable sections:")
        for num, name, _ in sections:
            print(f"  {num}. {name}")
        print("  q. Quit")
        
        choice = input("\nSelect a section (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            print("\nExiting...")
            break
        
        # Find and run the selected section
        found = False
        for num, name, func in sections:
            if choice == num:
                print()
                func()
                found = True
                break
        
        if not found:
            print("\nInvalid choice. Please try again.")
    
    print("\nThank you for exploring!")


if __name__ == '__main__':
    main()
