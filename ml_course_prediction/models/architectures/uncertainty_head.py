"""
Uncertainty Quantification Head
Provides uncertainty estimates for predictions (1σ confidence intervals)
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple
import math


class UncertaintyHead(nn.Module):
    """
    Uncertainty quantification head for variational inference.
    
    Predicts both mean and variance (uncertainty) for each output dimension.
    Enables 1σ confidence intervals for predictions.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 2,  # LAT, LON
        hidden_size: int = 128,
        uncertainty_method: str = "variational"
    ):
        """
        Initialize uncertainty head
        
        Args:
            input_size: Size of input features
            output_size: Number of output dimensions (2 for LAT/LON)
            hidden_size: Size of hidden layer
            uncertainty_method: Method for uncertainty ("variational", "monte_carlo", "ensemble")
        """
        super(UncertaintyHead, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.uncertainty_method = uncertainty_method
        
        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Mean prediction head
        self.mean_head = nn.Linear(hidden_size, output_size)
        
        # Variance (uncertainty) prediction head
        # Use log variance for numerical stability
        self.logvar_head = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small values for stability"""
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through uncertainty head
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Dictionary with:
                - 'mean': Predicted mean of shape (batch_size, output_size)
                - 'logvar': Predicted log variance of shape (batch_size, output_size)
                - 'var': Predicted variance of shape (batch_size, output_size)
                - 'std': Predicted standard deviation of shape (batch_size, output_size)
        """
        # Shared feature extraction
        shared_features = self.shared(x)
        
        # Predict mean
        mean = self.mean_head(shared_features)
        
        # Predict log variance (for numerical stability)
        logvar = self.logvar_head(shared_features)
        
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Convert to variance and standard deviation
        var = torch.exp(logvar)
        std = torch.sqrt(var + 1e-6)  # Add small epsilon for numerical stability
        
        return {
            'mean': mean,
            'logvar': logvar,
            'var': var,
            'std': std
        }
    
    def sample(self, mean: torch.Tensor, std: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from predicted distribution
        
        Args:
            mean: Predicted mean of shape (batch_size, output_size)
            std: Predicted standard deviation of shape (batch_size, output_size)
            num_samples: Number of samples to draw
        
        Returns:
            samples: Sampled values of shape (batch_size, num_samples, output_size)
        """
        # Sample from normal distribution
        samples = torch.normal(mean.unsqueeze(1).expand(-1, num_samples, -1),
                              std.unsqueeze(1).expand(-1, num_samples, -1))
        return samples
    
    def get_confidence_interval(self, mean: torch.Tensor, std: torch.Tensor, 
                               confidence: float = 0.68) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get confidence interval for predictions
        
        Args:
            mean: Predicted mean of shape (batch_size, output_size)
            std: Predicted standard deviation of shape (batch_size, output_size)
            confidence: Confidence level (0.68 for 1σ, 0.95 for 2σ)
        
        Returns:
            lower: Lower bound of shape (batch_size, output_size)
            upper: Upper bound of shape (batch_size, output_size)
        """
        # For normal distribution, 1σ = 68%, 2σ = 95%
        # Using approximate z-scores for common confidence levels
        if confidence == 0.68:
            z_score = 1.0
        elif confidence == 0.95:
            z_score = 2.0
        elif confidence == 0.90:
            z_score = 1.645
        elif confidence == 0.99:
            z_score = 2.576
        else:
            # Approximate z-score using inverse error function
            # For 68%: z ≈ 1.0, for 95%: z ≈ 2.0
            # Linear interpolation for other values
            z_score = 1.0 + (confidence - 0.68) * (2.0 - 1.0) / (0.95 - 0.68)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return lower, upper

