"""
Tokenizer implementations for converting text to tokens and back.

This module contains two tokenizer implementations:
1. CharacterTokenizer: Simple character-level tokenization (good for learning)
2. BPETokenizer: Byte Pair Encoding tokenizer (more efficient, used in real LLMs)
"""

import regex as re
from typing import List, Dict, Tuple
from collections import Counter


class CharacterTokenizer:
    """
    Simple character-level tokenizer.
    
    Each character in the vocabulary gets a unique integer ID.
    This is the simplest form of tokenization and great for understanding
    the basics, but not efficient for large-scale models.
    
    Example:
        tokenizer = CharacterTokenizer()
        tokenizer.train("Hello World!")
        tokens = tokenizer.encode("Hello")  # [72, 101, 108, 108, 111]
        text = tokenizer.decode(tokens)     # "Hello"
    """
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
    def train(self, text: str):
        """Build vocabulary from training text."""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]
        
    def decode(self, ids: List[int]) -> str:
        """Convert list of token IDs back to text."""
        return ''.join([self.id_to_char.get(i, '') for i in ids])
        

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer.
    
    BPE is a subword tokenization algorithm that:
    1. Starts with individual characters
    2. Iteratively merges the most frequent adjacent pairs
    3. Builds a vocabulary of subword units
    
    This allows the model to:
    - Handle rare/unknown words by breaking them into subwords
    - Be more efficient than character-level tokenization
    - Balance vocabulary size with sequence length
    
    This is used in GPT-2, GPT-3, and many modern LLMs.
    """
    
    def __init__(self, vocab_size: int = 256):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (default 256 for basic ASCII)
        """
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
    def _get_stats(self, ids: List[int]) -> Counter:
        """Count frequency of adjacent pairs in the sequence."""
        counts = Counter()
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
        
    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Merge all occurrences of a pair into a new token ID."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
        
    def train(self, text: str, verbose: bool = False):
        """
        Train BPE tokenizer on text.
        
        Args:
            text: Training text
            verbose: Print progress during training
        """
        # Start with UTF-8 bytes
        tokens = list(text.encode('utf-8'))
        
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
                
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            if verbose and i % 50 == 0:
                print(f"Merge {i}/{num_merges}: {pair} -> {new_id} (count: {stats[pair]})")
            
            # Merge the pair
            tokens = self._merge(tokens, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using learned merges."""
        tokens = list(text.encode('utf-8'))
        
        # Apply merges
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            # Find the pair with lowest merge index (merged earliest)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            tokens = self._merge(tokens, pair, self.merges[pair])
            
        return tokens
        
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = b''.join([self.vocab.get(i, b'') for i in ids])
        return tokens.decode('utf-8', errors='replace')
