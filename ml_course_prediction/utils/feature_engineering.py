"""
Feature Engineering for ML Course Prediction
Extracts features from AIS trajectories for model training and prediction.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers features from AIS trajectory data for course prediction.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def extract_trajectory_features(self, trajectory: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract trajectory-based features from vessel data
        
        Args:
            trajectory: Trajectory DataFrame sorted by time
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        if len(trajectory) < 2:
            return features
        
        # Position features
        if 'LAT' in trajectory.columns and 'LON' in trajectory.columns:
            features['last_lat'] = trajectory['LAT'].iloc[-1]
            features['last_lon'] = trajectory['LON'].iloc[-1]
            features['lat_mean'] = trajectory['LAT'].mean()
            features['lon_mean'] = trajectory['LON'].mean()
            features['lat_std'] = trajectory['LAT'].std()
            features['lon_std'] = trajectory['LON'].std()
        
        # Speed features
        if 'SOG' in trajectory.columns:
            features['sog_mean'] = trajectory['SOG'].mean()
            features['sog_std'] = trajectory['SOG'].std()
            features['sog_max'] = trajectory['SOG'].max()
            features['sog_min'] = trajectory['SOG'].min()
            features['sog_last'] = trajectory['SOG'].iloc[-1]
            features['sog_trend'] = self._calculate_trend(trajectory['SOG'])
        
        # Course features (circular statistics)
        if 'COG' in trajectory.columns:
            cog_features = self._extract_circular_features(trajectory['COG'])
            features.update({f'cog_{k}': v for k, v in cog_features.items()})
        
        # Heading features
        if 'Heading' in trajectory.columns:
            heading_features = self._extract_circular_features(trajectory['Heading'])
            features.update({f'heading_{k}': v for k, v in heading_features.items()})
        
        # Motion features
        if len(trajectory) > 1:
            motion_features = self._extract_motion_features(trajectory)
            features.update(motion_features)
        
        # Temporal features
        if 'BaseDateTime' in trajectory.columns:
            temporal_features = self._extract_temporal_features(trajectory)
            features.update(temporal_features)
        
        # Vessel features
        if 'VesselType' in trajectory.columns:
            features['vessel_type'] = trajectory['VesselType'].iloc[0]
        
        if 'Length' in trajectory.columns:
            features['vessel_length'] = trajectory['Length'].iloc[0]
        
        if 'Width' in trajectory.columns:
            features['vessel_width'] = trajectory['Width'].iloc[0]
        
        return features
    
    def _extract_circular_features(self, angles: pd.Series) -> Dict[str, float]:
        """
        Extract circular statistics for angles (COG, Heading)
        
        Args:
            angles: Series of angles in degrees (0-360)
            
        Returns:
            Dictionary of circular features
        """
        features = {}
        
        # Remove NaN values
        angles_clean = angles.dropna()
        if len(angles_clean) == 0:
            return features
        
        # Convert to radians
        angles_rad = np.radians(angles_clean)
        
        # Circular mean (using unit vectors)
        sin_mean = np.sin(angles_rad).mean()
        cos_mean = np.cos(angles_rad).mean()
        circular_mean = np.degrees(np.arctan2(sin_mean, cos_mean))
        if circular_mean < 0:
            circular_mean += 360
        
        features['mean'] = circular_mean
        features['last'] = angles_clean.iloc[-1]
        
        # Circular variance (1 - mean resultant length)
        mean_resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
        circular_variance = 1 - mean_resultant_length
        features['variance'] = circular_variance
        features['consistency'] = mean_resultant_length  # Higher = more consistent
        
        # Circular standard deviation
        circular_std = np.sqrt(-2 * np.log(mean_resultant_length))
        features['std'] = np.degrees(circular_std)
        
        return features
    
    def _extract_motion_features(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """
        Extract motion-related features (acceleration, rate of turn, etc.)
        
        Args:
            trajectory: Trajectory DataFrame
            
        Returns:
            Dictionary of motion features
        """
        features = {}
        
        if len(trajectory) < 2:
            return features
        
        # Calculate time differences
        if 'BaseDateTime' in trajectory.columns:
            time_diffs = trajectory['BaseDateTime'].diff().dt.total_seconds() / 3600  # hours
        else:
            return features
        
        # Acceleration (if SOG available)
        if 'SOG' in trajectory.columns:
            speed_diffs = trajectory['SOG'].diff()
            acceleration = speed_diffs / time_diffs.replace(0, np.nan)
            features['acceleration_mean'] = acceleration.mean()
            features['acceleration_max'] = acceleration.abs().max()
            features['acceleration_std'] = acceleration.std()
        
        # Rate of turn (if COG available)
        if 'COG' in trajectory.columns:
            cog_diffs = trajectory['COG'].diff()
            # Handle circular wrap-around
            cog_diffs = cog_diffs.apply(lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x))
            rate_of_turn = cog_diffs / time_diffs.replace(0, np.nan)
            features['rate_of_turn_mean'] = rate_of_turn.mean()
            features['rate_of_turn_max'] = rate_of_turn.abs().max()
            features['rate_of_turn_std'] = rate_of_turn.std()
        
        # Distance traveled
        if 'LAT' in trajectory.columns and 'LON' in trajectory.columns:
            distances = []
            for i in range(1, len(trajectory)):
                dist = self._haversine_distance(
                    trajectory.iloc[i-1]['LAT'], trajectory.iloc[i-1]['LON'],
                    trajectory.iloc[i]['LAT'], trajectory.iloc[i]['LON']
                )
                distances.append(dist)
            
            if distances:
                features['distance_total_nm'] = sum(distances)
                features['distance_mean_nm'] = np.mean(distances)
                features['distance_std_nm'] = np.std(distances)
        
        return features
    
    def _extract_temporal_features(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """
        Extract temporal features (time of day, day of week, etc.)
        
        Args:
            trajectory: Trajectory DataFrame with BaseDateTime column
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        if 'BaseDateTime' not in trajectory.columns:
            return features
        
        # Time of day (hour)
        hours = trajectory['BaseDateTime'].dt.hour
        features['hour_mean'] = hours.mean()
        features['hour_std'] = hours.std()
        features['hour_last'] = hours.iloc[-1]
        
        # Day of week (0=Monday, 6=Sunday)
        day_of_week = trajectory['BaseDateTime'].dt.dayofweek
        features['day_of_week_mean'] = day_of_week.mean()
        features['day_of_week_last'] = day_of_week.iloc[-1]
        
        # Time span
        time_span = (trajectory['BaseDateTime'].max() - trajectory['BaseDateTime'].min()).total_seconds() / 3600
        features['time_span_hours'] = time_span
        
        # Time since last position
        if len(trajectory) > 1:
            last_time_diff = (trajectory['BaseDateTime'].iloc[-1] - 
                            trajectory['BaseDateTime'].iloc[-2]).total_seconds() / 3600
            features['time_since_last_hours'] = last_time_diff
        
        return features
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """
        Calculate linear trend (slope) of a time series
        
        Args:
            series: Time series values
            
        Returns:
            Trend value (positive = increasing, negative = decreasing)
        """
        if len(series) < 2:
            return 0.0
        
        series_clean = series.dropna()
        if len(series_clean) < 2:
            return 0.0
        
        x = np.arange(len(series_clean))
        y = series_clean.values
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in nautical miles
        """
        R = 3440.065  # Earth radius in nautical miles
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def create_sequence_features(self, trajectory: pd.DataFrame,
                                sequence_length: int = 24,
                                prediction_horizon: int = 48) -> Dict[str, np.ndarray]:
        """
        Create sequence features for LSTM/Transformer input
        
        Args:
            trajectory: Trajectory DataFrame (already filtered to time window)
            sequence_length: Number of hours for the sequence (24 hours = max 8 points)
            prediction_horizon: Number of hours to predict ahead
            
        Returns:
            Dictionary with 'input_sequence' and 'feature_columns' arrays
            
        Note:
            AIS reports come every 3 hours, so a 24-hour window contains
            at most 8 points (24 / 3 = 8). Sequences will have 2-8 points
            depending on reporting frequency and data availability.
        """
        # AIS reports every 3 hours, so 24h window = max 8 points
        # With time-based windowing, trajectory can have 2-8 points
        # as long as it spans the sequence_length hours
        if len(trajectory) < 1:
            logger.warning(f"Trajectory is empty")
            return {}
        
        # Use all points in the trajectory (they're already filtered to the time window)
        # Expected: 2-8 points for a 24-hour window (AIS reports every 3 hours)
        input_traj = trajectory.copy()
        
        # Feature columns for input
        feature_cols = ['LAT', 'LON']
        if 'SOG' in trajectory.columns:
            feature_cols.append('SOG')
        if 'COG' in trajectory.columns:
            feature_cols.append('COG')
        if 'Heading' in trajectory.columns:
            feature_cols.append('Heading')
        
        # Extract input sequence
        input_sequence = input_traj[feature_cols].values
        
        # For training, we'd extract target sequence from future positions
        # For inference, target would be None
        result = {
            'input_sequence': input_sequence,
            'feature_columns': feature_cols
        }
        
        return result

