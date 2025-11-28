"""
Loss Functions for ML Course Prediction Model
Implements multi-objective loss with position, speed, course, and uncertainty components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class CoursePredictionLoss(nn.Module):
    """
    Multi-objective loss function for course prediction model.
    
    Components:
    - Position loss: MSE for LAT/LON predictions
    - Speed loss: MSE for SOG predictions
    - Course loss: Circular loss for COG predictions (von Mises)
    - Uncertainty loss: Negative log-likelihood of actual positions under predicted distribution
    - Physics constraint loss: Penalty for violating physical constraints
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        speed_weight: float = 0.5,
        course_weight: float = 0.5,
        uncertainty_weight: float = 0.3,
        physics_weight: float = 0.2,
        max_speed_knots: float = 50.0,
        max_turn_rate_deg_per_hour: float = 180.0
    ):
        """
        Initialize loss function with component weights
        
        Args:
            position_weight: Weight for position prediction loss
            speed_weight: Weight for speed prediction loss
            course_weight: Weight for course prediction loss
            uncertainty_weight: Weight for uncertainty calibration loss
            physics_weight: Weight for physics constraint violations
            max_speed_knots: Maximum allowed speed (for physics constraints)
            max_turn_rate_deg_per_hour: Maximum turn rate (for physics constraints)
        """
        super(CoursePredictionLoss, self).__init__()
        
        self.position_weight = position_weight
        self.speed_weight = speed_weight
        self.course_weight = course_weight
        self.uncertainty_weight = uncertainty_weight
        self.physics_weight = physics_weight
        
        self.max_speed_knots = max_speed_knots
        self.max_turn_rate_deg_per_hour = max_turn_rate_deg_per_hour
        
        # Constants for nautical miles conversion
        # 1 degree latitude ≈ 60 nautical miles
        self.DEG_TO_NM_LAT = 60.0
        # Longitude varies by latitude, but approximate
        self.DEG_TO_NM_LON_APPROX = 60.0
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        input_sequences: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and component losses
        
        Args:
            predictions: Dictionary with model predictions
                - 'position': Dict with 'mean' (B, 2) and 'std' (B, 2)
                - 'speed': (B, 1) optional
                - 'course': (B, 1) optional
            targets: Dictionary with target values
                - 'position': (B, 2) [LAT, LON]
                - 'speed': (B, 1) optional
                - 'course': (B, 1) optional
            input_sequences: (B, T, F) input sequences for physics constraints
        
        Returns:
            total_loss: Scalar total loss
            loss_components: Dictionary with individual loss components
        """
        loss_components = {}
        
        # Position loss (MSE in degrees, can be converted to nautical miles)
        position_loss = self._compute_position_loss(
            predictions['position'],
            targets['position']
        )
        loss_components['position'] = position_loss
        
        # Speed loss (if available)
        speed_loss = torch.tensor(0.0, device=position_loss.device)
        if 'speed' in predictions and 'speed' in targets:
            speed_loss = self._compute_speed_loss(
                predictions['speed'],
                targets['speed']
            )
        loss_components['speed'] = speed_loss
        
        # Course loss (if available) - circular loss
        course_loss = torch.tensor(0.0, device=position_loss.device)
        if 'course' in predictions and 'course' in targets:
            course_loss = self._compute_course_loss(
                predictions['course'],
                targets['course']
            )
        loss_components['course'] = course_loss
        
        # Uncertainty calibration loss (negative log-likelihood)
        uncertainty_loss = self._compute_uncertainty_loss(
            predictions['position'],
            targets['position']
        )
        loss_components['uncertainty'] = uncertainty_loss
        
        # Physics constraint loss (if input sequences provided)
        physics_loss = torch.tensor(0.0, device=position_loss.device)
        if input_sequences is not None and 'position' in predictions:
            physics_loss = self._compute_physics_loss(
                predictions,
                input_sequences
            )
        loss_components['physics'] = physics_loss
        
        # Weighted total loss
        total_loss = (
            self.position_weight * position_loss +
            self.speed_weight * speed_loss +
            self.course_weight * course_loss +
            self.uncertainty_weight * uncertainty_loss +
            self.physics_weight * physics_loss
        )
        
        return total_loss, loss_components
    
    def _compute_position_loss(
        self,
        position_pred: Dict[str, torch.Tensor],
        position_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute position prediction loss (MSE)
        
        Args:
            position_pred: Dict with 'mean' (B, 2) [LAT, LON]
            position_target: (B, 2) [LAT, LON]
        
        Returns:
            MSE loss
        """
        pred_mean = position_pred['mean']  # (B, 2)
        
        # MSE loss
        mse = F.mse_loss(pred_mean, position_target, reduction='mean')
        
        return mse
    
    def _compute_speed_loss(
        self,
        speed_pred: torch.Tensor,
        speed_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute speed prediction loss (MSE)
        
        Args:
            speed_pred: (B, 1) predicted SOG
            speed_target: (B, 1) target SOG
        
        Returns:
            MSE loss
        """
        return F.mse_loss(speed_pred, speed_target, reduction='mean')
    
    def _compute_course_loss(
        self,
        course_pred: torch.Tensor,
        course_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute course prediction loss using circular statistics.
        Uses von Mises distribution approach (1 - cos(difference))
        
        Args:
            course_pred: (B, 1) predicted COG in degrees
            course_target: (B, 1) target COG in degrees
        
        Returns:
            Circular loss
        """
        # Convert to radians
        pred_rad = torch.deg2rad(course_pred)
        target_rad = torch.deg2rad(course_target)
        
        # Circular difference
        diff_rad = pred_rad - target_rad
        
        # Normalize to [-pi, pi]
        diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))
        
        # Circular loss: 1 - cos(difference) ranges from 0 to 2
        # This gives higher penalty for larger angular differences
        loss = 1.0 - torch.cos(diff_rad)
        
        return loss.mean()
    
    def _compute_uncertainty_loss(
        self,
        position_pred: Dict[str, torch.Tensor],
        position_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty calibration loss (negative log-likelihood).
        Assumes bivariate normal distribution for position.
        
        Args:
            position_pred: Dict with 'mean' (B, 2) and 'std' (B, 2)
            position_target: (B, 2) actual positions
        
        Returns:
            Negative log-likelihood loss
        """
        pred_mean = position_pred['mean']  # (B, 2)
        pred_std = position_pred['std']  # (B, 2)
        
        # Ensure std is positive and not too small
        pred_std = torch.clamp(pred_std, min=1e-6)
        
        # Compute negative log-likelihood for independent Gaussian (approximation)
        # For each dimension (LAT, LON), compute NLL
        diff = position_target - pred_mean  # (B, 2)
        
        # NLL = 0.5 * log(2π * σ²) + 0.5 * (x - μ)² / σ²
        nll_lat = 0.5 * torch.log(2 * np.pi * pred_std[:, 0]**2) + \
                  0.5 * (diff[:, 0]**2) / (pred_std[:, 0]**2)
        
        nll_lon = 0.5 * torch.log(2 * np.pi * pred_std[:, 1]**2) + \
                  0.5 * (diff[:, 1]**2) / (pred_std[:, 1]**2)
        
        nll = nll_lat + nll_lon
        
        return nll.mean()
    
    def _compute_physics_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        input_sequences: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physics constraint violation loss.
        Penalizes predictions that violate physical constraints:
        - Maximum speed
        - Maximum turn rate
        - Impossible position changes
        
        Args:
            predictions: Model predictions dictionary
            input_sequences: (B, T, F) input sequences or None
        
        Returns:
            Physics constraint penalty
        """
        if input_sequences is None:
            # If no input sequences, only check speed constraints
            if 'speed' in predictions:
                speed_pred = predictions['speed'].squeeze(-1)  # (B,)
                speed_violations = torch.clamp(
                    speed_pred - self.max_speed_knots,
                    min=0.0
                )
                return speed_violations.mean()
            # Get device from position predictions if available
            if 'position' in predictions and 'mean' in predictions['position']:
                return torch.tensor(0.0, device=predictions['position']['mean'].device)
            return torch.tensor(0.0)
        
        physics_penalties = []
        
        # Check speed constraints
        if 'speed' in predictions:
            speed_pred = predictions['speed'].squeeze(-1)  # (B,)
            # Penalty for speeds exceeding maximum
            speed_violations = torch.clamp(
                speed_pred - self.max_speed_knots,
                min=0.0
            )
            physics_penalties.append(speed_violations.mean())
        
        # Check turn rate constraints (if we have course predictions and input sequences)
        if 'course' in predictions and input_sequences.shape[1] > 1:
            # Extract last course from input sequence (COG is typically feature index 3)
            if input_sequences.shape[2] >= 4:
                last_cog = input_sequences[:, -1, 3] * 180.0 / np.pi  # Convert to degrees if needed
                pred_cog = predictions['course'].squeeze(-1)  # (B,)
                
                # Compute turn rate (degrees per hour, assuming 48-hour prediction)
                turn_deg = torch.abs(pred_cog - last_cog)
                # Normalize to [0, 180] (smaller angle)
                turn_deg = torch.min(turn_deg, 360.0 - turn_deg)
                turn_rate = turn_deg / 48.0  # degrees per hour
                
                # Penalty for turn rates exceeding maximum
                turn_violations = torch.clamp(
                    turn_rate - self.max_turn_rate_deg_per_hour,
                    min=0.0
                )
                physics_penalties.append(turn_violations.mean())
        
        if len(physics_penalties) == 0:
            return torch.tensor(0.0, device=input_sequences.device)
        
        return sum(physics_penalties) / len(physics_penalties)


def haversine_distance_nm(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Haversine distance between two points in nautical miles.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in nautical miles
    """
    # Convert to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat / 2)**2 + \
        torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2)**2
    
    c = 2 * torch.asin(torch.sqrt(a))
    
    # Earth radius in nautical miles
    R_NM = 3440.0  # approximately
    
    distance_nm = R_NM * c
    
    return distance_nm

