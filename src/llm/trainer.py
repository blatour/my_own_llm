"""
Training utilities for the LLM.

This module provides a simple training loop and utilities for training
a language model on text data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from tqdm import tqdm
import os


class TextDataset(Dataset):
    """
    Simple dataset for language modeling.
    
    Takes a long sequence of token IDs and creates training examples by
    sliding a window over it.
    
    Args:
        token_ids: List of token IDs from the training text
        seq_len: Length of each training sequence
    """
    
    def __init__(self, token_ids: list, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)
    
    def __getitem__(self, idx):
        """
        Return a chunk of token_ids.
        For language modeling, input is tokens[idx:idx+seq_len]
        and target is tokens[idx+1:idx+seq_len+1] (shifted by 1)
        """
        chunk = self.token_ids[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class Trainer:
    """
    Simple trainer for the language model.
    
    Handles the training loop with:
    - Forward pass and loss computation
    - Backpropagation and optimization
    - Progress tracking
    - Model checkpointing
    
    Args:
        model: The GPT model to train
        train_dataset: Training dataset
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        device: Device to train on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        
        # Create data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Optimizer (AdamW is standard for transformers)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, progress_callback: Optional[Callable] = None):
        """
        Train for one epoch.
        
        Args:
            progress_callback: Optional callback function called after each batch
                               with (batch_idx, loss) as arguments
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            logits = self.model(x)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for loss computation
            # CrossEntropyLoss expects (N, C) and (N,)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
                y.view(-1)  # (batch_size * seq_len,)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if progress_callback:
                progress_callback(batch_idx, loss.item())
        
        return total_loss / len(self.train_loader)
    
    def train(self, num_epochs: int, checkpoint_dir: Optional[str] = None):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            avg_loss = self.train_epoch()
            print(f"Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded: {path}")
