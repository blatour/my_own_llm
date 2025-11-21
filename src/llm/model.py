"""
GPT-style transformer model implementation.

This module implements a decoder-only transformer, similar to GPT-2/GPT-3.
The architecture consists of:
1. Token embeddings + positional embeddings
2. Stack of transformer blocks (attention + feed-forward)
3. Output projection to vocabulary

This is a "decoder-only" model because it only uses causal (masked) attention,
making it suitable for autoregressive text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This is applied to each position independently after attention.
    It's a simple 2-layer MLP with expansion in the hidden dimension.
    
    Architecture:
        Linear(d_model -> 4*d_model) -> GELU -> Linear(4*d_model -> d_model) -> Dropout
    
    The 4x expansion is standard in transformers and gives the model capacity
    to process the attended information.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation (smoother than ReLU, used in GPT)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block.
    
    Structure:
        x = x + Attention(LayerNorm(x))      # Self-attention with residual
        x = x + FeedForward(LayerNorm(x))    # Feed-forward with residual
    
    Key concepts:
    - Residual connections (x + layer(x)): Help gradients flow during training
    - Layer normalization: Stabilizes training
    - Pre-norm (norm before layer): Modern practice, more stable than post-norm
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attention(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    GPT-style decoder-only transformer model.
    
    This is the complete language model architecture. It's similar to GPT-2/GPT-3
    but simplified for learning purposes.
    
    Architecture:
        1. Token Embedding: Convert token IDs to vectors
        2. Positional Embedding: Add position information
        3. Transformer Blocks: Stack of attention + feed-forward layers
        4. Layer Norm: Final normalization
        5. Output Projection: Map to vocabulary logits
    
    Args:
        vocab_size: Size of vocabulary (number of possible tokens)
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads per block
        num_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    
    Example:
        model = GPTModel(vocab_size=50257, d_model=768, num_heads=12, num_layers=12)
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings: map token IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings: add position information
        # (Learned embeddings, not sinusoidal - common in GPT models)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between token embedding and output projection
        # This is a common practice that reduces parameters and improves training
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # Get token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_emb = self.position_embedding(positions)  # (seq_len, d_model)
        
        # Combine embeddings
        x = self.dropout(token_emb + position_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds max length
            input_crop = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self(input_crop)  # (batch_size, seq_len, vocab_size)
            
            # Focus on last token
            logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
