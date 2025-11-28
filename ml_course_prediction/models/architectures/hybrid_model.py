"""
Hybrid LSTM-Transformer Model
Combines LSTM for short-term patterns and Transformer for long-term dependencies
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .lstm_encoder import LSTMEncoder
from .transformer_encoder import TransformerEncoder
from .uncertainty_head import UncertaintyHead


class HybridLSTMTransformer(nn.Module):
    """
    Hybrid model combining LSTM and Transformer for AIS course prediction.
    
    Architecture:
    1. LSTM encoder: Captures short-term temporal patterns
    2. Transformer encoder: Captures long-term dependencies via attention
    3. Feature fusion: Combines LSTM and Transformer outputs
    4. Output heads: Predicts position, speed, course with uncertainty
    """
    
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        lstm_bidirectional: bool = False,
        transformer_hidden_size: int = 256,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 8,
        transformer_feedforward_size: int = 1024,
        transformer_dropout: float = 0.1,
        fusion_hidden_size: int = 256,
        prediction_horizon: int = 48,
        uncertainty_method: str = "variational",
        include_speed_course: bool = True
    ):
        """
        Initialize hybrid LSTM-Transformer model
        
        Args:
            input_size: Number of input features per time step
            lstm_hidden_size: LSTM hidden state size
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: LSTM dropout probability
            lstm_bidirectional: Whether LSTM is bidirectional
            transformer_hidden_size: Transformer model dimension
            transformer_num_layers: Number of transformer layers
            transformer_num_heads: Number of attention heads
            transformer_feedforward_size: Transformer feedforward size
            transformer_dropout: Transformer dropout probability
            fusion_hidden_size: Size of fusion layer
            prediction_horizon: Hours to predict ahead (48)
            uncertainty_method: Uncertainty quantification method
            include_speed_course: Whether to predict speed and course
        """
        super(HybridLSTMTransformer, self).__init__()
        
        self.input_size = input_size
        self.prediction_horizon = prediction_horizon
        self.include_speed_course = include_speed_course
        
        # LSTM encoder for short-term patterns
        self.lstm_encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidirectional
        )
        lstm_output_size = self.lstm_encoder.output_size
        
        # Transformer encoder for long-term dependencies
        self.transformer_encoder = TransformerEncoder(
            input_size=input_size,
            hidden_size=transformer_hidden_size,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            feedforward_size=transformer_feedforward_size,
            dropout=transformer_dropout,
            max_seq_len=2000  # Increased to handle variable-length sequences (dynamic encoding handles longer)
        )
        
        # Feature fusion: Combine LSTM and Transformer outputs
        # Use last hidden states from both encoders
        fusion_input_size = lstm_output_size + transformer_hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.ReLU()
        )
        
        # Output heads
        # Position prediction (LAT, LON) with uncertainty
        self.position_head = UncertaintyHead(
            input_size=fusion_hidden_size,
            output_size=2,  # LAT, LON
            hidden_size=128,
            uncertainty_method=uncertainty_method
        )
        
        # Speed and course prediction (optional)
        if include_speed_course:
            self.speed_head = nn.Sequential(
                nn.Linear(fusion_hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # SOG (knots)
            )
            
            self.course_head = nn.Sequential(
                nn.Linear(fusion_hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # COG (degrees)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               seq_len varies from 2-8 points (AIS reports every 3 hours)
            lengths: Optional tensor of actual sequence lengths
        
        Returns:
            Dictionary with predictions:
                - 'position': Dict with 'mean', 'std', 'var' for LAT/LON
                - 'speed': Predicted SOG (if include_speed_course=True)
                - 'course': Predicted COG (if include_speed_course=True)
        """
        batch_size = x.size(0)
        
        # Create mask for transformer if lengths provided
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(
                batch_size, max_len
            ) < lengths.unsqueeze(1)
        
        # LSTM encoder
        lstm_output, lstm_hidden = self.lstm_encoder(x, lengths)
        lstm_last = self.lstm_encoder.get_last_hidden(lstm_output, lengths)
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(x, mask)
        transformer_last = self.transformer_encoder.get_last_hidden(
            transformer_output, lengths
        )
        
        # Fuse LSTM and Transformer features
        fused_features = torch.cat([lstm_last, transformer_last], dim=1)
        fused = self.fusion(fused_features)
        
        # Position prediction with uncertainty
        position_pred = self.position_head(fused)
        
        # Speed and course prediction (if enabled)
        results = {
            'position': position_pred
        }
        
        if self.include_speed_course:
            speed_pred = self.speed_head(fused)
            course_pred = self.course_head(fused)
            results['speed'] = speed_pred
            results['course'] = course_pred
        
        return results
    
    def predict(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with optional uncertainty
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional tensor of actual sequence lengths
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            Dictionary with predictions and optionally uncertainty
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, lengths)
            
            if return_uncertainty:
                # Add confidence intervals
                position_mean = outputs['position']['mean']
                position_std = outputs['position']['std']
                
                lower_68, upper_68 = self.position_head.get_confidence_interval(
                    position_mean, position_std, confidence=0.68
                )
                
                outputs['position']['lower_68'] = lower_68
                outputs['position']['upper_68'] = upper_68
        
        return outputs
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size information
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

