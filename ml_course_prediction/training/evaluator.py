"""
Evaluation utilities for ML Course Prediction Model
Computes metrics and validates model performance
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .loss_functions import haversine_distance_nm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for course prediction model.
    Computes various metrics including position error, uncertainty calibration, etc.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize evaluator
        
        Args:
            device: Device to use for computations
        """
        self.device = device
    
    def evaluate(
        self,
        model: torch.nn.Module,
        sequences: List[Dict],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on a set of sequences
        
        Args:
            model: Trained model
            sequences: List of sequence dictionaries
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary of metric names to values
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        all_errors_nm = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Convert sequences to tensors
            batch_inputs, batch_targets = self._prepare_batch(batch_sequences)
            
            # Move to device
            x = batch_inputs['x'].to(self.device)
            lengths = batch_inputs['lengths'].to(self.device)
            batch_targets = {k: v.to(self.device) for k, v in batch_targets.items()}
            
            # Forward pass
            with torch.no_grad():
                predictions = model(x, lengths)
            
            # Extract predictions and targets
            pred_positions = predictions['position']['mean'].cpu()  # (B, 2)
            pred_stds = predictions['position']['std'].cpu()  # (B, 2)
            target_positions = batch_targets['position'].cpu()  # (B, 2)
            
            # Compute errors in nautical miles
            errors_nm = haversine_distance_nm(
                target_positions[:, 0],  # lat
                target_positions[:, 1],  # lon
                pred_positions[:, 0],    # pred_lat
                pred_positions[:, 1]     # pred_lon
            )
            
            all_predictions.append(pred_positions.numpy())
            all_targets.append(target_positions.numpy())
            all_uncertainties.append(pred_stds.numpy())
            all_errors_nm.append(errors_nm.numpy())
        
        # Concatenate all results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        all_errors_nm = np.concatenate(all_errors_nm, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_predictions,
            all_targets,
            all_uncertainties,
            all_errors_nm
        )
        
        return metrics
    
    def _prepare_batch(
        self,
        sequences: List[Dict]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare batch of sequences for model input
        
        Args:
            sequences: List of sequence dictionaries
        
        Returns:
            Tuple of (batch_inputs, batch_targets) dictionaries
        """
        batch_size = len(sequences)
        
        # Find max sequence length
        max_len = max(len(seq['input_sequence']) for seq in sequences)
        
        # Get feature dimension
        feature_dim = sequences[0]['input_sequence'].shape[1] if hasattr(
            sequences[0]['input_sequence'], 'shape'
        ) else len(sequences[0]['input_sequence'][0])
        
        # Initialize batch tensors
        batch_input = torch.zeros(batch_size, max_len, feature_dim)
        batch_lengths = torch.zeros(batch_size, dtype=torch.long)
        batch_target_pos = torch.zeros(batch_size, 2)
        batch_target_sog = torch.zeros(batch_size, 1)
        batch_target_cog = torch.zeros(batch_size, 1)
        
        # Fill batch
        for i, seq in enumerate(sequences):
            seq_len = len(seq['input_sequence'])
            seq_data = torch.tensor(seq['input_sequence'], dtype=torch.float32)
            batch_input[i, :seq_len, :] = seq_data
            batch_lengths[i] = seq_len
            
            # Targets
            if len(seq['target_positions']) > 0:
                batch_target_pos[i] = torch.tensor(
                    seq['target_positions'][0],  # First target position
                    dtype=torch.float32
                )
            
            if seq.get('target_sog') is not None:
                # Convert numpy array to scalar - target_seq has 1 row, so get first value
                target_sog_val = seq['target_sog']
                if isinstance(target_sog_val, np.ndarray):
                    target_sog_val = float(target_sog_val[0] if len(target_sog_val) > 0 else 0.0)
                elif isinstance(target_sog_val, (list, tuple)):
                    target_sog_val = float(target_sog_val[0] if len(target_sog_val) > 0 else 0.0)
                else:
                    target_sog_val = float(target_sog_val)
                batch_target_sog[i, 0] = target_sog_val
            
            if seq.get('target_cog') is not None:
                # Convert numpy array to scalar - target_seq has 1 row, so get first value
                target_cog_val = seq['target_cog']
                if isinstance(target_cog_val, np.ndarray):
                    target_cog_val = float(target_cog_val[0] if len(target_cog_val) > 0 else 0.0)
                elif isinstance(target_cog_val, (list, tuple)):
                    target_cog_val = float(target_cog_val[0] if len(target_cog_val) > 0 else 0.0)
                else:
                    target_cog_val = float(target_cog_val)
                batch_target_cog[i, 0] = target_cog_val
        
        batch_inputs = {
            'x': batch_input,
            'lengths': batch_lengths
        }
        
        batch_targets = {
            'position': batch_target_pos,
            'speed': batch_target_sog,
            'course': batch_target_cog
        }
        
        return batch_inputs, batch_targets
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
        errors_nm: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            predictions: (N, 2) predicted positions [LAT, LON]
            targets: (N, 2) target positions [LAT, LON]
            uncertainties: (N, 2) predicted standard deviations
            errors_nm: (N,) position errors in nautical miles
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Position error metrics (in nautical miles)
        metrics['mae_nm'] = float(np.mean(errors_nm))
        metrics['rmse_nm'] = float(np.sqrt(np.mean(errors_nm**2)))
        metrics['median_error_nm'] = float(np.median(errors_nm))
        metrics['p90_error_nm'] = float(np.percentile(errors_nm, 90))
        metrics['p95_error_nm'] = float(np.percentile(errors_nm, 95))
        
        # Uncertainty calibration
        # Check coverage: % of predictions within 1σ
        # Approximate 1σ ellipse with circular approximation
        # Use mean of LAT and LON std as radius
        mean_std_deg = np.mean(uncertainties, axis=1)  # (N,)
        # Convert to nautical miles (1 degree ≈ 60 nm)
        mean_std_nm = mean_std_deg * 60.0
        
        within_1sigma = errors_nm <= mean_std_nm
        metrics['coverage_1sigma'] = float(np.mean(within_1sigma))
        
        # Check coverage for 2σ
        within_2sigma = errors_nm <= (mean_std_nm * 2.0)
        metrics['coverage_2sigma'] = float(np.mean(within_2sigma))
        
        # Position error breakdown by dimension (in degrees)
        lat_errors_deg = np.abs(predictions[:, 0] - targets[:, 0])
        lon_errors_deg = np.abs(predictions[:, 1] - targets[:, 1])
        
        metrics['mae_lat_deg'] = float(np.mean(lat_errors_deg))
        metrics['mae_lon_deg'] = float(np.mean(lon_errors_deg))
        
        # Uncertainty statistics
        metrics['mean_uncertainty_lat_deg'] = float(np.mean(uncertainties[:, 0]))
        metrics['mean_uncertainty_lon_deg'] = float(np.mean(uncertainties[:, 1]))
        metrics['mean_uncertainty_nm'] = float(np.mean(mean_std_nm))
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
        """
        logger.info("=" * 60)
        logger.info("Evaluation Metrics")
        logger.info("=" * 60)
        
        logger.info("\nPosition Error (Nautical Miles):")
        logger.info(f"  MAE:      {metrics['mae_nm']:.2f} nm")
        logger.info(f"  RMSE:     {metrics['rmse_nm']:.2f} nm")
        logger.info(f"  Median:   {metrics['median_error_nm']:.2f} nm")
        logger.info(f"  90th %ile: {metrics['p90_error_nm']:.2f} nm")
        logger.info(f"  95th %ile: {metrics['p95_error_nm']:.2f} nm")
        
        logger.info("\nUncertainty Calibration:")
        logger.info(f"  1σ Coverage: {metrics['coverage_1sigma']*100:.1f}% (target: 68%)")
        logger.info(f"  2σ Coverage: {metrics['coverage_2sigma']*100:.1f}% (target: 95%)")
        logger.info(f"  Mean Uncertainty: {metrics['mean_uncertainty_nm']:.2f} nm")
        
        logger.info("\nPosition Error by Dimension (Degrees):")
        logger.info(f"  LAT MAE: {metrics['mae_lat_deg']:.4f}°")
        logger.info(f"  LON MAE: {metrics['mae_lon_deg']:.4f}°")
        
        logger.info("=" * 60)

