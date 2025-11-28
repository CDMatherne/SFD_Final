#!/usr/bin/env python3
"""
ML Course Prediction Integration Module for SFD Project

This module provides integration between the ML Course Prediction system
and the Advanced Analysis GUI.
"""

import os
import sys
import logging
import traceback
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Try to import ML dependencies
ML_PREDICTION_AVAILABLE = False
try:
    import torch
    from pathlib import Path
    
    # Add ml_course_prediction to path
    ml_course_prediction_path = Path(__file__).parent / "ml_course_prediction"
    if str(ml_course_prediction_path.parent) not in sys.path:
        sys.path.insert(0, str(ml_course_prediction_path.parent))
    
    from ml_course_prediction.models.model_factory import load_model
    from ml_course_prediction.utils.feature_engineering import FeatureEngineer
    from ml_course_prediction.utils.trajectory_utils import TrajectoryProcessor
    
    ML_PREDICTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML Course Prediction module not available: {e}")
    ML_PREDICTION_AVAILABLE = False


class MLPredictionError(Exception):
    """Custom exception for ML prediction errors"""
    pass


class MLPredictionIntegrator:
    """
    Integrates ML Course Prediction with the Advanced Analysis system.
    """
    
    def _detect_amd_gpu_hardware(self) -> bool:
        """
        Detect if AMD GPU hardware is present in the system.
        
        Returns:
            True if AMD GPU is detected, False otherwise
        """
        try:
            is_windows = platform.system() == 'Windows'
            
            if is_windows:
                # Windows: Use wmic to get video controller information
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if line and line.lower() != 'name':
                            gpu_name_lower = line.lower()
                            if 'amd' in gpu_name_lower or 'radeon' in gpu_name_lower:
                                logger.info(f"Detected AMD GPU hardware: {line.strip()}")
                                return True
            else:
                # Linux: Use lspci to get GPU information
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'vga' in line.lower() or 'display' in line.lower() or '3d' in line.lower():
                            line_lower = line.lower()
                            if 'amd' in line_lower or 'radeon' in line_lower or 'ati' in line_lower:
                                logger.info(f"Detected AMD GPU hardware: {line.strip()}")
                                return True
        except Exception as e:
            logger.debug(f"Error detecting AMD GPU hardware: {e}")
        
        return False
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the ML Prediction Integrator.
        
        Args:
            model_path: Path to trained model. If None, will search for best_model.pt
            device: Device to use ('cpu', 'cuda' for NVIDIA, or 'cuda' for AMD ROCm). 
                    If None, auto-detects (checks for CUDA/ROCm, falls back to CPU).
        """
        if not ML_PREDICTION_AVAILABLE:
            raise MLPredictionError("ML Course Prediction module is not available. "
                                  "Please install PyTorch and ensure ml_course_prediction module is available.")
        
        self.model = None
        # Auto-detect device: check for CUDA (NVIDIA), ROCm (AMD), or fallback to CPU
        if device:
            self.device = device
        else:
            # Check if PyTorch has ROCm support compiled in
            has_rocm_support = hasattr(torch.version, 'hip') and torch.version.hip is not None
            
            # First check PyTorch GPU availability
            if torch.cuda.is_available():
                self.device = 'cuda'
                # Log which backend is being used
                if has_rocm_support:
                    logger.info("Using AMD ROCm backend for GPU acceleration")
                    if torch.cuda.device_count() > 0:
                        try:
                            device_name = torch.cuda.get_device_name(0)
                            logger.info(f"GPU: {device_name}")
                        except:
                            pass
                else:
                    logger.info("Using NVIDIA CUDA backend for GPU acceleration")
                    if torch.cuda.device_count() > 0:
                        try:
                            device_name = torch.cuda.get_device_name(0)
                            logger.info(f"GPU: {device_name}")
                        except:
                            pass
            else:
                # PyTorch doesn't see GPU - check if AMD GPU hardware is present
                amd_gpu_detected = self._detect_amd_gpu_hardware()
                if amd_gpu_detected:
                    if has_rocm_support:
                        logger.warning(
                            "AMD GPU hardware detected and PyTorch has ROCm support, "
                            "but torch.cuda.is_available() returned False.\n"
                            "This may indicate:\n"
                            "  1. ROCm drivers are not properly installed\n"
                            "  2. GPU is not compatible with current ROCm version\n"
                            "  3. ROCm runtime is not accessible\n\n"
                            "Check ROCm installation: https://rocm.docs.amd.com/\n"
                            "Falling back to CPU for now."
                        )
                    else:
                        logger.warning(
                            "AMD GPU hardware detected, but PyTorch does not have ROCm support.\n"
                            "Standard PyTorch from PyPI does not support AMD GPUs.\n\n"
                            "To enable AMD GPU support:\n"
                            "  1. Uninstall current PyTorch: pip uninstall torch torchvision torchaudio\n"
                            "  2. Install PyTorch with ROCm support:\n"
                            "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7\n"
                            "  3. Install ROCm drivers: https://rocm.docs.amd.com/\n"
                            "  4. Note: ROCm on Windows is experimental - Linux is recommended\n\n"
                            "Falling back to CPU for now."
                        )
                else:
                    logger.info("No GPU hardware detected in system")
                self.device = 'cpu'
                logger.info("Using CPU for model inference")
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer()
        self.trajectory_processor = TrajectoryProcessor(
            max_gap_hours=6.0,
            min_trajectory_points=2,
            min_trajectory_hours=1.0
        )
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the trained ML model.
        
        Args:
            model_path: Path to model file. If None, searches for best_model.pt
        """
        if model_path:
            self.model_path = model_path
        else:
            # Search for model
            model_dir = Path(__file__).parent / "ml_course_prediction" / "models" / "trained"
            best_model = model_dir / "best_model.pt"
            
            if best_model.exists():
                self.model_path = str(best_model)
            else:
                # Try checkpoint files
                checkpoint_files = list(model_dir.glob("checkpoint_*.pt"))
                if checkpoint_files:
                    # Use the latest checkpoint
                    self.model_path = str(sorted(checkpoint_files)[-1])
                else:
                    raise MLPredictionError(
                        f"Trained model not found in {model_dir}. "
                        "Please train a model first or specify a model path."
                    )
        
        if not os.path.exists(self.model_path):
            raise MLPredictionError(f"Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading model from {self.model_path} on device {self.device}")
            # Try to find default config file
            model_file = Path(self.model_path)
            default_config = model_file.parent.parent / "configs" / "default_config.yaml"
            config_path = str(default_config) if default_config.exists() else None
            
            if config_path:
                logger.info(f"Using config file: {config_path}")
            
            self.model = load_model(self.model_path, config_path=config_path, device=self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            raise MLPredictionError(f"Failed to load model: {str(e)}")
    
    def prepare_vessel_data(self, df: pd.DataFrame, mmsi: int, 
                          hours_back: int = 24) -> pd.DataFrame:
        """
        Prepare vessel data for prediction.
        
        Args:
            df: DataFrame with AIS data
            mmsi: Vessel MMSI number
            hours_back: Number of hours of historical data to use
            
        Returns:
            DataFrame with prepared vessel trajectory
            
        Raises:
            MLPredictionError: If data preparation fails
        """
        if df.empty:
            raise MLPredictionError("Input DataFrame is empty")
        
        # Filter for the specific vessel
        if 'MMSI' in df.columns:
            # Ensure MMSI types match
            sample_mmsi = df['MMSI'].iloc[0] if len(df) > 0 else None
            if sample_mmsi is not None:
                if isinstance(sample_mmsi, (int, np.integer)):
                    mmsi = int(mmsi)
                elif isinstance(sample_mmsi, (str, np.str_)):
                    mmsi = str(mmsi)
            df['MMSI'] = df['MMSI'].astype(type(mmsi))
        
        vessel_data = df[df['MMSI'] == mmsi].copy()
        
        if vessel_data.empty:
            raise MLPredictionError(f"No data found for vessel {mmsi}")
        
        # Sort by time and get last N hours of data
        if 'BaseDateTime' in vessel_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(vessel_data['BaseDateTime']):
                vessel_data['BaseDateTime'] = pd.to_datetime(
                    vessel_data['BaseDateTime'], errors='coerce'
                )
            vessel_data = vessel_data.sort_values('BaseDateTime')
            
            # Get last N hours
            last_time = vessel_data['BaseDateTime'].max()
            time_window_start = last_time - pd.Timedelta(hours=hours_back)
            recent_data = vessel_data[vessel_data['BaseDateTime'] >= time_window_start].copy()
        else:
            # If no BaseDateTime, use all available data (take last 8 points max)
            recent_data = vessel_data.tail(8).copy()
        
        if len(recent_data) < 2:
            raise MLPredictionError(
                f"Insufficient data for prediction. Need at least 2 data points, "
                f"found {len(recent_data)}"
            )
        
        return recent_data
    
    def process_trajectory(self, vessel_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process vessel trajectory data.
        
        Args:
            vessel_data: DataFrame with vessel AIS data
            
        Returns:
            Processed trajectory DataFrame
            
        Raises:
            MLPredictionError: If trajectory processing fails
        """
        try:
            # Segment trajectories
            trajectories = self.trajectory_processor.segment_trajectories(
                vessel_data,
                mmsi_col='MMSI',
                time_col='BaseDateTime'
            )
            
            if not trajectories:
                # Provide detailed error message
                num_records = len(vessel_data)
                if num_records < 2:
                    raise MLPredictionError(
                        f"Insufficient data for prediction. Found {num_records} record(s), "
                        f"need at least 2 data points for trajectory processing."
                    )
                
                # Check if records have valid timestamps
                if 'BaseDateTime' in vessel_data.columns:
                    time_span = (vessel_data['BaseDateTime'].max() - vessel_data['BaseDateTime'].min()).total_seconds() / 3600
                    if time_span < 1.0:
                        raise MLPredictionError(
                            f"Trajectory time span too short: {time_span:.2f} hours. "
                            f"Need at least 1.0 hours between first and last data point. "
                            f"Found {num_records} records."
                        )
                    elif time_span > 6.0:
                        raise MLPredictionError(
                            f"Time gap too large: {time_span:.2f} hours between data points. "
                            f"Maximum allowed gap is 6.0 hours. Trajectory was split and segments "
                            f"did not meet minimum requirements (need at least 2 points per segment)."
                        )
                
                # Check for missing position data
                if 'LAT' in vessel_data.columns and 'LON' in vessel_data.columns:
                    missing_lat = vessel_data['LAT'].isna().sum()
                    missing_lon = vessel_data['LON'].isna().sum()
                    if missing_lat > 0 or missing_lon > 0:
                        raise MLPredictionError(
                            f"Missing position data: {missing_lat} missing LAT values, "
                            f"{missing_lon} missing LON values out of {num_records} records."
                        )
                
                raise MLPredictionError(
                    f"Failed to create valid trajectory from {num_records} records. "
                    f"Trajectory must have at least 2 points spanning at least 1.0 hours, "
                    f"with no gaps larger than 6.0 hours."
                )
            
            # Use the most recent trajectory
            trajectory = trajectories[-1]
            
            if len(trajectory) < 2:
                raise MLPredictionError(
                    f"Insufficient trajectory points. Need at least 2, found {len(trajectory)}"
                )
            
            return trajectory
            
        except Exception as e:
            raise MLPredictionError(f"Trajectory processing failed: {str(e)}")
    
    def extract_features(self, trajectory: pd.DataFrame) -> torch.Tensor:
        """
        Extract features from trajectory for model input.
        
        Args:
            trajectory: Processed trajectory DataFrame
            
        Returns:
            Input tensor for the model
            
        Raises:
            MLPredictionError: If feature extraction fails
        """
        try:
            sequence_data = self.feature_engineer.create_sequence_features(
                trajectory,
                sequence_length=24,
                prediction_horizon=48
            )
            
            if not sequence_data or 'input_sequence' not in sequence_data:
                raise MLPredictionError("Failed to extract features from trajectory")
            
            input_sequence = sequence_data['input_sequence']
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  # Add batch dimension
            
            return input_tensor
            
        except Exception as e:
            raise MLPredictionError(f"Feature extraction failed: {str(e)}")
    
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Generate predictions using the loaded model.
        Uses autoregressive prediction to generate 8 time steps (48 hours at 6-hour intervals).
        
        Args:
            input_tensor: Input tensor for the model
            
        Returns:
            Dictionary with prediction results (all arrays are numpy arrays):
            - position_mean: Predicted positions (8, 2) [lat, lon] for 48 hours
            - position_std: Position uncertainties (8, 2)
            - position_lower: Lower 68% confidence bounds (8, 2)
            - position_upper: Upper 68% confidence bounds (8, 2)
            - speed: Predicted speeds (8,) [optional]
            - course: Predicted courses (8,) [optional]
            
        Note:
            This method's output format is designed to align with _display_prediction_results()
            in advanced_analysis.py. All position arrays must have shape (8, 2) where:
            - First dimension: 8 time steps (6-hour intervals)
            - Second dimension: 2 (LAT, LON)
            
        Raises:
            MLPredictionError: If prediction fails
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Generate 8 time steps (48 hours at 6-hour intervals) using autoregressive prediction
            num_steps = 8
            device = input_tensor.device
            
            # Store predictions for each time step
            position_means = []
            position_stds = []
            position_lowers = []
            position_uppers = []
            # Check if model supports speed/course prediction
            test_output = self.model.forward(input_tensor)
            speeds = [] if 'speed' in test_output else None
            courses = [] if 'course' in test_output else None
            
            # Current input sequence (starts with the provided input)
            current_input = input_tensor.clone()
            
            with torch.no_grad():
                for step in range(num_steps):
                    # Get prediction for this time step
                    predictions = self.model.predict(current_input, return_uncertainty=True)
                    
                    # Extract position prediction
                    pos_mean = predictions['position']['mean']  # Shape: (batch, 2)
                    pos_std = predictions['position']['std']  # Shape: (batch, 2)
                    pos_lower = predictions['position']['lower_68']  # Shape: (batch, 2)
                    pos_upper = predictions['position']['upper_68']  # Shape: (batch, 2)
                    
                    # Store predictions
                    position_means.append(pos_mean.cpu().numpy())
                    position_stds.append(pos_std.cpu().numpy())
                    position_lowers.append(pos_lower.cpu().numpy())
                    position_uppers.append(pos_upper.cpu().numpy())
                    
                    # Extract optional predictions
                    if speeds is not None and 'speed' in predictions:
                        speeds.append(predictions['speed'].cpu().numpy())
                    if courses is not None and 'course' in predictions:
                        courses.append(predictions['course'].cpu().numpy())
                    
                    # For autoregressive prediction, append the predicted position to the input sequence
                    # This allows the model to use previous predictions to predict the next step
                    if step < num_steps - 1:  # Don't need to update for the last step
                        # Get the last position from the current input sequence
                        last_pos = current_input[:, -1, :2]  # Extract LAT, LON from last time step
                        
                        # Create a new time step with the predicted position
                        # We need to construct a feature vector for the next time step
                        # Use the predicted position and optionally predicted speed/course
                        next_features = pos_mean  # Start with predicted position
                        
                        # If we have speed/course predictions, add them
                        if speeds is not None and len(speeds) > 0:
                            # Use the most recent speed prediction
                            speed_val = speeds[-1] if isinstance(speeds[-1], np.ndarray) else speeds[-1]
                            if isinstance(speed_val, np.ndarray):
                                speed_val = torch.tensor(speed_val, device=device, dtype=torch.float32)
                            else:
                                speed_val = torch.tensor([[speed_val]], device=device, dtype=torch.float32)
                            next_features = torch.cat([next_features, speed_val], dim=1)
                        
                        if courses is not None and len(courses) > 0:
                            # Use the most recent course prediction
                            course_val = courses[-1] if isinstance(courses[-1], np.ndarray) else courses[-1]
                            if isinstance(course_val, np.ndarray):
                                course_val = torch.tensor(course_val, device=device, dtype=torch.float32)
                            else:
                                course_val = torch.tensor([[course_val]], device=device, dtype=torch.float32)
                            next_features = torch.cat([next_features, course_val], dim=1)
                        
                        # Pad or truncate to match input_size
                        input_size = current_input.shape[2]
                        if next_features.shape[1] < input_size:
                            # Pad with zeros or repeat last value
                            padding = torch.zeros((next_features.shape[0], input_size - next_features.shape[1]), 
                                                 device=device, dtype=torch.float32)
                            next_features = torch.cat([next_features, padding], dim=1)
                        elif next_features.shape[1] > input_size:
                            # Truncate to input_size
                            next_features = next_features[:, :input_size]
                        
                        # Append to current input sequence (sliding window - keep last sequence_length points)
                        # For simplicity, we'll append and keep a reasonable window size
                        current_input = torch.cat([current_input, next_features.unsqueeze(1)], dim=1)
                        # Keep only the last sequence_length points to avoid memory issues
                        max_seq_len = 24  # Keep last 24 points (sequence_length)
                        if current_input.shape[1] > max_seq_len:
                            current_input = current_input[:, -max_seq_len:, :]
            
            # Stack predictions into arrays
            # Each prediction is shape (batch, 2), so stacking gives (num_steps, batch, 2)
            # We want (num_steps, 2) for single batch or (batch, num_steps, 2) for multiple batches
            position_mean = np.stack(position_means, axis=0)  # Shape: (num_steps, batch, 2)
            position_std = np.stack(position_stds, axis=0)
            position_lower = np.stack(position_lowers, axis=0)
            position_upper = np.stack(position_uppers, axis=0)
            
            # Remove batch dimension if batch_size == 1, resulting in (num_steps, 2)
            if position_mean.shape[1] == 1:
                position_mean = position_mean[:, 0, :]  # (num_steps, 2)
                position_std = position_std[:, 0, :]
                position_lower = position_lower[:, 0, :]
                position_upper = position_upper[:, 0, :]
            else:
                # Multiple batches: transpose to (batch, num_steps, 2)
                position_mean = position_mean.transpose(1, 0, 2)
                position_std = position_std.transpose(1, 0, 2)
                position_lower = position_lower.transpose(1, 0, 2)
                position_upper = position_upper.transpose(1, 0, 2)
            
            logger.info(f"Generated {num_steps} prediction steps. Final position_mean shape: {position_mean.shape}, ndim: {position_mean.ndim}")
            
            # Final verification - should already be (num_steps, 2) after above processing
            if position_mean.ndim != 2:
                raise MLPredictionError(f"After processing, position_mean has {position_mean.ndim} dimensions, expected 2. Shape: {position_mean.shape}")
            if position_mean.shape[1] != 2:
                raise MLPredictionError(f"After processing, position_mean second dimension is {position_mean.shape[1]}, expected 2. Shape: {position_mean.shape}")
            if position_mean.shape[0] != num_steps:
                logger.warning(f"Expected {num_steps} time steps, but got {position_mean.shape[0]}. This may indicate an issue with autoregressive prediction.")
            
            logger.info(f"Final position_mean shape: {position_mean.shape} (expected ({num_steps}, 2))")
            
            # Ensure all arrays are numpy arrays with consistent shape (num_steps, 2)
            # Verify all arrays have the same shape
            assert position_mean.shape == position_std.shape == position_lower.shape == position_upper.shape, \
                f"Shape mismatch: mean={position_mean.shape}, std={position_std.shape}, lower={position_lower.shape}, upper={position_upper.shape}"
            
            result = {
                'position_mean': position_mean,  # Shape: (8, 2)
                'position_std': position_std,   # Shape: (8, 2)
                'position_lower': position_lower,  # Shape: (8, 2)
                'position_upper': position_upper,  # Shape: (8, 2)
            }
            
            # Add optional predictions if available
            if speeds is not None and len(speeds) > 0:
                speed_array = np.stack(speeds, axis=0)
                if speed_array.ndim > 1 and speed_array.shape[1] == 1:
                    speed_array = speed_array[:, 0]
                result['speed'] = speed_array
            if courses is not None and len(courses) > 0:
                course_array = np.stack(courses, axis=0)
                if course_array.ndim > 1 and course_array.shape[1] == 1:
                    course_array = course_array[:, 0]
                result['course'] = course_array
            
            return result
            
        except Exception as e:
            raise MLPredictionError(f"Prediction failed: {str(e)}")
    
    def predict_vessel_course(self, df: pd.DataFrame, mmsi: int, 
                             hours_back: int = 24) -> Dict[str, Any]:
        """
        Complete prediction pipeline for a vessel.
        
        Args:
            df: DataFrame with AIS data
            mmsi: Vessel MMSI number
            hours_back: Number of hours of historical data to use
            
        Returns:
            Dictionary with:
            - predictions: Prediction results (see predict() method)
            - trajectory: Processed trajectory DataFrame
            - last_position: Tuple of (lat, lon) for last known position
            - last_time: Last known timestamp
            
        Raises:
            MLPredictionError: If any step fails
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Prepare vessel data
        vessel_data = self.prepare_vessel_data(df, mmsi, hours_back)
        
        # Process trajectory
        trajectory = self.process_trajectory(vessel_data)
        
        # Extract features
        input_tensor = self.extract_features(trajectory)
        
        # Generate predictions
        predictions = self.predict(input_tensor)
        
        # Get last known position
        last_lat = float(trajectory['LAT'].iloc[-1])
        last_lon = float(trajectory['LON'].iloc[-1])
        last_time = None
        if 'BaseDateTime' in trajectory.columns:
            last_time = trajectory['BaseDateTime'].iloc[-1]
        
        return {
            'predictions': predictions,
            'trajectory': trajectory,
            'last_position': (last_lat, last_lon),
            'last_time': last_time,
            'mmsi': mmsi
        }


def is_available() -> bool:
    """Check if ML prediction is available."""
    return ML_PREDICTION_AVAILABLE


def create_integrator(model_path: Optional[str] = None, 
                     device: Optional[str] = None) -> MLPredictionIntegrator:
    """
    Factory function to create an MLPredictionIntegrator instance.
    
    Args:
        model_path: Path to trained model
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        MLPredictionIntegrator instance
        
    Raises:
        MLPredictionError: If module is not available
    """
    if not ML_PREDICTION_AVAILABLE:
        raise MLPredictionError("ML Course Prediction module is not available")
    
    return MLPredictionIntegrator(model_path=model_path, device=device)