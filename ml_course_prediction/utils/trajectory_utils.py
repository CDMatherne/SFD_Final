"""
Trajectory Processing Utilities
Handles trajectory segmentation, gap detection, and preprocessing for AIS data.
"""
import gc
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class TrajectoryProcessor:
    """
    Processes AIS trajectories for ML course prediction.
    Handles segmentation, gap detection, and data quality checks.
    """
    
    def __init__(self, 
                 max_gap_hours: float = 6.0,
                 min_trajectory_points: int = 2,  # AIS reports every 3h, so min 2 points = 6h
                 min_trajectory_hours: float = 1.0):  # Lenient during segmentation
        """
        Initialize trajectory processor
        
        Args:
            max_gap_hours: Maximum time gap (hours) before splitting trajectory
            min_trajectory_points: Minimum number of points for valid trajectory
                Note: AIS reports every 3 hours, so max 8 points per 24h window
            min_trajectory_hours: Minimum duration (hours) for valid trajectory
        """
        self.max_gap_hours = max_gap_hours
        self.min_trajectory_points = min_trajectory_points
        self.min_trajectory_hours = min_trajectory_hours
        
    def segment_trajectories(self, df: pd.DataFrame, 
                            mmsi_col: str = 'MMSI',
                            time_col: str = 'BaseDateTime',
                            batch_size: int = 1000) -> List[pd.DataFrame]:
        """
        Segment vessel trajectories by MMSI and time gaps.
        
        Splits trajectories when:
        - Time gap exceeds max_gap_hours
        - Vessel changes (different MMSI)
        
        Processes vessels in batches to reduce memory usage.
        
        Args:
            df: DataFrame with AIS data
            mmsi_col: Column name for MMSI
            time_col: Column name for timestamp
            batch_size: Number of vessels to process before freeing memory
            
        Returns:
            List of trajectory DataFrames
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for segmentation")
            return []
        
        # Ensure time column is datetime (modify in place if needed)
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by MMSI and time (without deep copy to save memory)
        # Use sort=False in groupby to avoid re-sorting since we sort by MMSI first
        logger.info(f"Sorting {len(df):,} records by {mmsi_col} and {time_col}...")
        df_sorted = df.sort_values([mmsi_col, time_col])
        
        # Get unique MMSIs for batch processing
        unique_mmsis = df_sorted[mmsi_col].unique()
        num_vessels = len(unique_mmsis)
        logger.info(f"Segmenting trajectories for {num_vessels:,} vessels from {len(df_sorted):,} records...")
        
        trajectories = []
        
        # Process vessels in batches to reduce memory pressure
        for batch_start in range(0, num_vessels, batch_size):
            batch_end = min(batch_start + batch_size, num_vessels)
            batch_mmsis = unique_mmsis[batch_start:batch_end]
            
            if batch_start % (batch_size * 10) == 0 or batch_start == 0:
                logger.info(f"Processing vessels {batch_start+1:,}-{batch_end:,}/{num_vessels:,}...")
            
            # Filter to batch vessels only
            batch_mask = df_sorted[mmsi_col].isin(batch_mmsis)
            batch_df = df_sorted[batch_mask]
            
            # Process batch vessels
            for mmsi in batch_mmsis:
                vessel_df = batch_df[batch_df[mmsi_col] == mmsi]
                
                if len(vessel_df) < self.min_trajectory_points:
                    continue
                
                # Since we sorted by [MMSI, time_col], vessel_df is already sorted by time
                # No need to sort again - this saves memory by avoiding copies
                # Extract time column as Series (memory-efficient view) for diff calculation
                time_series = vessel_df[time_col].reset_index(drop=True)
                time_diffs = time_series.diff().dt.total_seconds() / 3600  # hours
                
                # Find split points (gaps > max_gap_hours)
                split_indices = [0]
                for idx, gap in enumerate(time_diffs):
                    if gap > self.max_gap_hours:
                        split_indices.append(idx)
                
                split_indices.append(len(vessel_df))
                
                # Create trajectory segments
                # Use iloc indexing and copy only when appending to trajectories list
                for i in range(len(split_indices) - 1):
                    start_idx = split_indices[i]
                    end_idx = split_indices[i + 1]
                    segment = vessel_df.iloc[start_idx:end_idx].copy()  # Copy needed for storage
                    
                    # Validate segment
                    if self._is_valid_trajectory(segment, time_col):
                        trajectories.append(segment)
            
            # Explicitly delete batch_df to free memory between batches
            del batch_df
            gc.collect()  # Force garbage collection to free memory
        
        logger.info(f"Segmented {len(df)} records into {len(trajectories)} trajectories")
        return trajectories
    
    def _is_valid_trajectory(self, segment: pd.DataFrame, time_col: str) -> bool:
        """
        Check if trajectory segment is valid
        
        Args:
            segment: Trajectory segment DataFrame
            time_col: Column name for timestamp
            
        Returns:
            True if valid, False otherwise
        """
        # Check minimum points
        if len(segment) < self.min_trajectory_points:
            return False
        
        # Check minimum duration
        time_span = (segment[time_col].max() - segment[time_col].min()).total_seconds() / 3600
        if time_span < self.min_trajectory_hours:
            return False
        
        # Check for required columns
        required_cols = ['LAT', 'LON']
        if not all(col in segment.columns for col in required_cols):
            return False
        
        # Check for valid positions
        if segment['LAT'].isna().all() or segment['LON'].isna().all():
            return False
        
        return True
    
    def detect_gaps(self, trajectory: pd.DataFrame,
                   time_col: str = 'BaseDateTime') -> List[Dict[str, Any]]:
        """
        Detect time gaps in trajectory
        
        Args:
            trajectory: Trajectory DataFrame
            time_col: Column name for timestamp
            
        Returns:
            List of gap dictionaries with start, end, duration
        """
        if len(trajectory) < 2:
            return []
        
        trajectory = trajectory.sort_values(time_col)
        time_diffs = trajectory[time_col].diff().dt.total_seconds() / 3600  # hours
        
        gaps = []
        for idx, gap_hours in enumerate(time_diffs):
            if gap_hours > self.max_gap_hours:
                gap_info = {
                    'start_time': trajectory.iloc[idx - 1][time_col],
                    'end_time': trajectory.iloc[idx][time_col],
                    'duration_hours': gap_hours,
                    'start_index': idx - 1,
                    'end_index': idx
                }
                gaps.append(gap_info)
        
        return gaps
    
    def filter_outliers(self, trajectory: pd.DataFrame,
                       max_speed_knots: float = 100.0,  # Changed from 50.0 to match preprocessing
                       max_acceleration: Optional[float] = None) -> pd.DataFrame:  # Removed acceleration filter
        """
        Filter physically implausible positions (outliers)
        
        Args:
            trajectory: Trajectory DataFrame
            max_speed_knots: Maximum reasonable speed (knots)
            max_acceleration: Maximum reasonable acceleration (knots/hour)
            
        Returns:
            Filtered trajectory DataFrame
        """
        if len(trajectory) < 2:
            return trajectory
        
        trajectory = trajectory.sort_values('BaseDateTime').copy()
        
        # Calculate speeds and distances
        trajectory['time_diff_hours'] = trajectory['BaseDateTime'].diff().dt.total_seconds() / 3600
        trajectory['lat_diff'] = trajectory['LAT'].diff()
        trajectory['lon_diff'] = trajectory['LON'].diff()
        
        # Calculate distance (nautical miles) using Haversine formula
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance in nautical miles"""
            R = 3440.065  # Earth radius in nautical miles
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        
        distances = []
        for i in range(1, len(trajectory)):
            dist = haversine_distance(
                trajectory.iloc[i-1]['LAT'], trajectory.iloc[i-1]['LON'],
                trajectory.iloc[i]['LAT'], trajectory.iloc[i]['LON']
            )
            distances.append(dist)
        
        trajectory['distance_nm'] = [0] + distances
        trajectory['speed_knots'] = trajectory['distance_nm'] / trajectory['time_diff_hours'].replace(0, np.nan)
        
        # Filter outliers
        initial_len = len(trajectory)
        
        # Filter by speed only (acceleration filter removed)
        if 'SOG' in trajectory.columns:
            # Use SOG if available, otherwise use calculated speed
            speed_col = 'SOG'
        else:
            speed_col = 'speed_knots'
        
        trajectory = trajectory[
            (trajectory[speed_col].fillna(0) <= max_speed_knots) |
            (trajectory[speed_col].isna())
        ]
        
        filtered_len = len(trajectory)
        if filtered_len < initial_len:
            logger.info(f"Filtered {initial_len - filtered_len} outlier points from trajectory")
        
        # Clean up temporary columns
        trajectory = trajectory.drop(columns=['time_diff_hours', 'lat_diff', 'lon_diff', 
                                             'distance_nm', 'speed_knots'], 
                                    errors='ignore')
        
        return trajectory
    
    def interpolate_trajectory(self, trajectory: pd.DataFrame,
                              target_interval_hours: float = 1.0,
                              time_col: str = 'BaseDateTime') -> pd.DataFrame:
        """
        Interpolate trajectory to regular time intervals
        
        Args:
            trajectory: Trajectory DataFrame
            target_interval_hours: Target time interval between points (hours)
            time_col: Column name for timestamp
            
        Returns:
            Interpolated trajectory DataFrame
        """
        if len(trajectory) < 2:
            return trajectory
        
        trajectory = trajectory.sort_values(time_col).copy()
        
        # Create regular time index
        start_time = trajectory[time_col].min()
        end_time = trajectory[time_col].max()
        time_range = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{target_interval_hours * 60:.0f}min'
        )
        
        # Set time as index for interpolation
        trajectory_indexed = trajectory.set_index(time_col)
        
        # Interpolate numeric columns
        numeric_cols = trajectory_indexed.select_dtypes(include=[np.number]).columns
        interpolated = trajectory_indexed.reindex(time_range)
        
        for col in numeric_cols:
            if col in ['LAT', 'LON', 'SOG', 'COG']:
                # Use linear interpolation for position and motion
                interpolated[col] = interpolated[col].interpolate(method='linear')
            else:
                # Forward fill for other numeric columns
                interpolated[col] = interpolated[col].interpolate(method='ffill')
        
        # Forward fill non-numeric columns
        non_numeric_cols = trajectory_indexed.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            interpolated[col] = interpolated[col].fillna(method='ffill')
        
        # Reset index
        interpolated = interpolated.reset_index()
        interpolated = interpolated.rename(columns={'index': time_col})
        
        logger.debug(f"Interpolated trajectory from {len(trajectory)} to {len(interpolated)} points")
        
        return interpolated

