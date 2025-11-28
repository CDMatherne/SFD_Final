"""
LSTM Encoder Component
Handles short-term temporal patterns in AIS trajectories
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for capturing short-term temporal patterns in vessel trajectories.
    
    Processes variable-length sequences (2-8 points) and extracts temporal features.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM encoder
        
        Args:
            input_size: Number of input features per time step
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output size depends on bidirectional flag
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM encoder
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional tensor of actual sequence lengths for variable-length sequences
                    Shape: (batch_size,)
        
        Returns:
            output: LSTM output of shape (batch_size, seq_len, output_size)
            hidden: Final hidden state of shape (num_layers * num_directions, batch_size, hidden_size)
        """
        # Pack sequence if lengths are provided (for variable-length sequences)
        if lengths is not None:
            # Sort sequences by length (descending) for efficiency
            lengths = lengths.cpu()
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            x_sorted = x[sorted_idx]
            
            # Pack padded sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted,
                sorted_lengths,
                batch_first=True,
                enforce_sorted=True
            )
            
            # Forward through LSTM
            output_packed, hidden = self.lstm(x_packed)
            
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed,
                batch_first=True,
                total_length=x.size(1)
            )
            
            # Check actual batch sizes
            original_batch_size = x.size(0)
            output_batch_size = output.size(0)
            # hidden is tuple of (h_n, c_n), each with shape (num_layers * directions, batch_size, hidden_size)
            # So we index dimension 1 for batch_size
            hidden_batch_size = hidden[0].size(1)
            
            # Debug: check dimensions
            # If hidden_batch_size != original_batch_size, we have a mismatch
            if hidden_batch_size != output_batch_size:
                raise RuntimeError(
                    f"Batch size mismatch after unpacking: hidden_batch_size={hidden_batch_size}, "
                    f"output_batch_size={output_batch_size}, original_batch_size={original_batch_size}. "
                    f"hidden[0] shape: {hidden[0].shape}, output shape: {output.shape}"
                )
            
            # Unsort to restore original order
            # sorted_idx maps: new_position -> original_position  
            # Example: if sorted_idx = [2, 1, 0, 3], it means:
            #   position 0 in sorted order was originally at position 2
            #   position 1 in sorted order was originally at position 1
            #   position 2 in sorted order was originally at position 0
            #   position 3 in sorted order was originally at position 3
            # To reverse this, we need unsort_idx such that:
            #   unsort_idx[original_position] = sorted_position
            # This is achieved by: unsort_idx = torch.argsort(sorted_idx)
            # Example: torch.argsort([2, 1, 0, 3]) = [2, 1, 0, 3] - wait, that's not right
            # Actually: torch.argsort([2, 1, 0, 3]) returns indices to sort [2, 1, 0, 3], which is [2, 1, 0, 3]
            # But we need: if sorted_idx[i] = original_pos, then unsort_idx[original_pos] = i
            # So unsort_idx should satisfy: unsort_idx[sorted_idx[i]] = i
            # We can create this with: unsort_idx = torch.zeros_like(sorted_idx); unsort_idx[sorted_idx] = torch.arange(len(sorted_idx))
            unsort_idx = torch.zeros_like(sorted_idx)
            unsort_idx[sorted_idx] = torch.arange(len(sorted_idx), dtype=sorted_idx.dtype)
            
            output = output[unsort_idx]
            # Ensure unsort_idx is on the same device as hidden states
            device = hidden[0].device
            unsort_idx = unsort_idx.to(device)
            
            # Unsort hidden states (index dimension 1, which is the batch dimension)
            hidden = tuple(h[:, unsort_idx, :] for h in hidden)
        else:
            # Standard forward pass for fixed-length sequences
            output, hidden = self.lstm(x)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, hidden
    
    def get_last_hidden(self, output: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract the last valid hidden state for each sequence
        
        Args:
            output: LSTM output of shape (batch_size, seq_len, output_size)
            lengths: Optional tensor of actual sequence lengths
        
        Returns:
            last_hidden: Last hidden state of shape (batch_size, output_size)
        """
        if lengths is not None:
            # Get last valid output for each sequence
            batch_size = output.size(0)
            last_hidden = output[torch.arange(batch_size), lengths - 1]
        else:
            # Use last timestep
            last_hidden = output[:, -1, :]
        
        return last_hidden

