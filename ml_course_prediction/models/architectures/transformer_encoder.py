"""
Transformer Encoder Component
Handles long-term dependencies and attention patterns in AIS trajectories
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to understand temporal order
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output with positional encoding added
        """
        seq_len = x.size(1)
        d_model = x.size(2)
        
        # If sequence is longer than pre-computed positional encoding, compute it dynamically
        if seq_len > self.pe.size(1):
            # Generate positional encoding for the longer sequence
            device = x.device
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(seq_len, d_model, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
            x = x + pe
        else:
            # Use pre-computed positional encoding
            x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for capturing long-term dependencies in vessel trajectories.
    
    Uses self-attention to model relationships between all time steps in the sequence.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_size: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 2000  # Increased to handle variable-length sequences (dynamic encoding handles longer)
    ):
        """
        Initialize Transformer encoder
        
        Args:
            input_size: Number of input features per time step
            hidden_size: Model dimension (d_model)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            feedforward_size: Size of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection to match hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Transformer encoder
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask of shape (batch_size, seq_len)
                 True/1 for valid positions, False/0 for padding
        
        Returns:
            output: Transformer output of shape (batch_size, seq_len, hidden_size)
        """
        # Project input to hidden_size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if lengths are provided
        if mask is not None:
            # Convert mask to attention mask format
            # True/1 = valid, False/0 = padding (should be masked)
            # Transformer expects True = mask out, so invert
            src_key_padding_mask = ~mask.bool()  # Invert: True = mask out
        else:
            src_key_padding_mask = None
        
        # Forward through transformer
        output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
    
    def get_last_hidden(self, output: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract the last valid hidden state for each sequence
        
        Args:
            output: Transformer output of shape (batch_size, seq_len, hidden_size)
            lengths: Optional tensor of actual sequence lengths
        
        Returns:
            last_hidden: Last hidden state of shape (batch_size, hidden_size)
        """
        if lengths is not None:
            # Get last valid output for each sequence
            batch_size = output.size(0)
            last_hidden = output[torch.arange(batch_size), lengths - 1]
        else:
            # Use last timestep
            last_hidden = output[:, -1, :]
        
        return last_hidden

