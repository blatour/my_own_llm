"""
Attention mechanism - the core of transformer models.

The attention mechanism allows the model to "attend" to different parts of the input
when processing each token. This is what makes transformers so powerful!

Key concepts:
- Self-Attention: Each token attends to all other tokens in the sequence
- Multi-Head Attention: Multiple attention operations in parallel (different perspectives)
- Causal/Masked Attention: For language modeling, tokens can only attend to previous tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    This is the "magic" that makes transformers work! The attention mechanism
    allows each position in the sequence to attend to all positions and combine
    their information in a weighted way.
    
    How it works:
    1. Each token creates Query (Q), Key (K), Value (V) vectors
    2. Attention weights = softmax(Q @ K^T / sqrt(d_k))
    3. Output = Attention weights @ V
    4. Multiple "heads" learn different attention patterns in parallel
    
    Args:
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        dropout: Dropout probability
        is_causal: If True, use causal masking (for language modeling)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal
        
        # Linear transformations for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Generate Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if needed (for autoregressive language modeling)
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final output projection
        out = self.out_proj(out)
        
        return out
