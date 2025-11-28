"""
Data Loader for ML Course Prediction Training
Loads and preprocesses AIS data for model training.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os
import pickle
import hashlib
import json

# Add parent directory to path to import from existing backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from ..utils.trajectory_utils import TrajectoryProcessor
from ..utils.feature_engineering import FeatureEngineer
from ..utils.data_preprocessing import HistoricalDataPreprocessor

logger = logging.getLogger(__name__)

# Try to import backend modules (after logger is defined)
try:
    from backend.data_connector import AISDataConnector
    from backend.data_cache import AISDataCache
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logger.warning("Backend modules not available - data loading will be limited")


class CoursePredictionDataLoader:
    """
    Data loader for course prediction model training and evaluation.
    """
    
    def __init__(self, 
                 data_connector: Optional[Any] = None,
                 data_cache: Optional[Any] = None,
                 historical_data_path: Optional[str] = None,
                 max_gap_hours: float = 6.0,
                 sequence_length: int = 24,
                 prediction_horizon: int = 48):
        """
        Initialize data loader
        
        Args:
            data_connector: AISDataConnector instance (optional)
            data_cache: AISDataCache instance (optional)
            historical_data_path: Path to historical data directory (e.g., r"C:\\AIS_Data_Testing\\Historical\\2024")
            max_gap_hours: Maximum time gap before splitting trajectory
            sequence_length: Number of historical hours for input
            prediction_horizon: Number of hours to predict ahead
        """
        self.data_connector = data_connector
        self.data_cache = data_cache
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.max_gap_hours = max_gap_hours
        
        # Initialize historical data preprocessor if path provided
        if historical_data_path:
            # Cache directory will be set to training/cache/ by default
            self.data_preprocessor = HistoricalDataPreprocessor(
                data_path=historical_data_path,
                use_cache=True  # Enable caching by default
            )
            self.use_historical_data = True
            # Use same cache directory for sequences (training/cache/)
            self.sequences_cache_dir = self.data_preprocessor.cache_dir / 'sequences'
        else:
            self.data_preprocessor = None
            self.use_historical_data = False
            # Set up sequences cache directory relative to this module
            training_dir = Path(__file__).resolve().parent
            self.sequences_cache_dir = training_dir / 'cache' / 'sequences'
        
        # Create sequences cache directory
        self.sequences_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use lenient min_trajectory_hours during segmentation (1 hour)
        # The 24-hour check is applied later in sequence preparation
        # AIS reports every 3 hours, so max 8 points per 24h window
        self.trajectory_processor = TrajectoryProcessor(
            max_gap_hours=max_gap_hours,
            min_trajectory_points=2,  # Minimum 2 points (6 hours of data)
            min_trajectory_hours=1.0  # Lenient during segmentation, strict check later
        )
        self.feature_engineer = FeatureEngineer()
        
        # Generate cache version hash for sequence parameters
        # This allows invalidating cache if sequence generation logic changes
        cache_params = {
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'max_gap_hours': max_gap_hours,
            'version': '1.0'  # Increment if sequence generation logic changes
        }
        params_str = json.dumps(cache_params, sort_keys=True)
        self.sequence_cache_version_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
    async def load_training_data(self,
                                start_date: str,
                                end_date: str,
                                vessel_types: Optional[List[int]] = None,
                                mmsi_list: Optional[List[str]] = None,
                                preprocess: bool = True,
                                filter_unknown_vessel_types: bool = True) -> pd.DataFrame:
        """
        Load AIS data for training
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            vessel_types: List of vessel type codes to filter (optional)
            mmsi_list: List of MMSI to filter (optional)
            preprocess: Whether to preprocess data (default: True)
            filter_unknown_vessel_types: Whether to filter sequences after preparation based on 24h vessel type reporting (default: False)
            
        Returns:
            DataFrame with AIS data
        """
        logger.info(f"Loading training data from {start_date} to {end_date}")
        
        # Priority 1: Use historical data preprocessor if available
        if self.use_historical_data and self.data_preprocessor:
            try:
                logger.info("Loading from historical data directory...")
                # load_date_range now handles preprocessing day-by-day with caching
                df = self.data_preprocessor.load_date_range(
                    start_date, 
                    end_date, 
                    preprocess=preprocess
                )
                logger.info("Data loading complete")
                
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                df = pd.DataFrame()
        
        # Priority 2: Use data cache if available
        elif self.data_cache:
            try:
                df = await self.data_cache.get_data_range(start_date, end_date)
                logger.info(f"Loaded {len(df)} records from cache")
                
                if preprocess and not df.empty:
                    # Apply basic preprocessing
                    if 'BaseDateTime' in df.columns and df['BaseDateTime'].dtype == 'object':
                        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                    df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
                    
            except Exception as e:
                logger.error(f"Error loading from cache: {e}")
                df = pd.DataFrame()
        
        # Priority 3: Use data connector
        elif self.data_connector:
            try:
                # This would need to be implemented based on data_connector API
                logger.warning("Direct data_connector loading not yet implemented")
                df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading from connector: {e}")
                df = pd.DataFrame()
        else:
            logger.error("No data source available")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning("No data loaded")
            return df
        
        # Log initial state for debugging
        initial_count = len(df)
        logger.debug(f"Data loaded: {initial_count:,} records")
        if 'VesselType' in df.columns:
            valid_types = ((df['VesselType'] >= 20) & (df['VesselType'] <= 99)).sum()
            unknown_types = ((df['VesselType'] == 0) | (df['VesselType'] == 0.0) | (df['VesselType'].isna())).sum()
            logger.debug(f"VesselType breakdown: {valid_types:,} valid (20-99), {unknown_types:,} unknown/null")
        
        # NOTE: Vessel type filtering moved to AFTER sequence preparation (post stage 4)
        # Do NOT filter unknown vessel types here - will be done after sequence preparation
        # The filter_unknown_vessel_types parameter is passed to prepare_training_sequences instead
        # This allows all data to flow through preprocessing and sequence creation first
        
        # Filter by vessel types if specified (for loading specific types, not filtering unknown)
        if vessel_types and 'VesselType' in df.columns:
            initial_count = len(df)
            
            # Get available types BEFORE filtering (for error reporting)
            available_types_before = sorted(df['VesselType'].dropna().unique()[:20])
            
            # Convert vessel_types to same type as VesselType column for comparison
            df_vessel_type_dtype = df['VesselType'].dtype
            if df_vessel_type_dtype == 'float64':
                vessel_types_compare = [float(vt) for vt in vessel_types]
            else:
                vessel_types_compare = [int(vt) for vt in vessel_types]
            
            df = df[df['VesselType'].isin(vessel_types_compare)]
            removed_count = initial_count - len(df)
            logger.info(f"Filtered to {len(df):,} records for vessel types {vessel_types} "
                       f"({removed_count:,} removed)")
            
            # Safety check
            if len(df) == 0:
                logger.error(f"ERROR: All data filtered out when filtering for vessel types {vessel_types}. "
                           f"Check if these vessel types exist in the data.")
                # Show available vessel types (from before filtering)
                if initial_count > 0:
                    logger.info(f"Available vessel types in data (sample): {available_types_before}")
                    # Check if requested types are close to available types
                    requested_set = set(vessel_types_compare)
                    available_set = set(available_types_before)
                    if not requested_set.intersection(available_set):
                        logger.warning(f"None of the requested vessel types {vessel_types} match available types. "
                                     f"This is why all data was filtered out.")
        
        # Filter by MMSI if specified
        if mmsi_list and 'MMSI' in df.columns:
            initial_count = len(df)
            df = df[df['MMSI'].isin(mmsi_list)]
            logger.info(f"Filtered to {len(df):,} records for {len(mmsi_list)} vessels "
                       f"({initial_count - len(df):,} removed)")
        
        return df
    
    def prepare_training_sequences(self, df: pd.DataFrame, 
                                   filter_unknown_vessel_types: bool = False,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   vessel_types: Optional[List[int]] = None,
                                   use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Prepare training sequences from AIS data
        
        Creates sliding window sequences for training:
        - Input: Last N hours of trajectory
        - Target: Next M hours of trajectory (for training)
        
        Supports caching to avoid reprocessing on subsequent runs.
        
        Args:
            df: DataFrame with AIS data
            filter_unknown_vessel_types: Whether to filter sequences after preparation based on 24h vessel type reporting (default: False)
            start_date: Start date for cache key (optional)
            end_date: End date for cache key (optional)
            vessel_types: List of vessel types for cache key (optional)
            use_cache: Whether to use cache for loading/saving sequences (default: True)
            
        Returns:
            List of training sequence dictionaries
        """
        # Generate cache key (only if caching is enabled and cache directory is set up)
        cache_path = None
        if use_cache and hasattr(self, 'sequences_cache_dir') and self.sequences_cache_dir:
            try:
                cache_key = self._get_sequences_cache_key(df, start_date, end_date, vessel_types, filter_unknown_vessel_types)
                cache_path = self.sequences_cache_dir / f"sequences_{cache_key}.pkl"
                
                # Try to load from cache
                if cache_path.exists():
                    try:
                        logger.info(f"Loading sequences from cache: {cache_path.name}")
                        with open(cache_path, 'rb') as f:
                            sequences = pickle.load(f)
                        logger.info(f"Loaded {len(sequences):,} sequences from cache")
                        
                        # FIX: If cache has 0 sequences, it's from a failed run - regenerate
                        if len(sequences) == 0:
                            logger.warning(f"Cache contains 0 sequences (likely from failed run). Regenerating...")
                        else:
                            return sequences
                    except Exception as e:
                        logger.warning(f"Error loading sequences from cache: {e}. Regenerating...")
            except Exception as e:
                logger.debug(f"Could not generate cache key (will continue without cache): {e}")
                cache_path = None
        
        # Generate sequences
        logger.info("Preparing training sequences")
        logger.info(f"Input data: {len(df):,} records, {df['MMSI'].nunique():,} unique vessels")
        logger.info(f"Sequence length requirement: {self.sequence_length} hours")
        logger.info(f"Prediction horizon: {self.prediction_horizon} hours")
        
        # Segment trajectories
        logger.info("Segmenting trajectories...")
        trajectories = self.trajectory_processor.segment_trajectories(df)
        logger.info(f"Created {len(trajectories):,} trajectory segments")
        
        if len(trajectories) == 0:
            logger.warning("No trajectories created - check trajectory segmentation parameters")
            return []
        
        sequences = []
        skipped_short = 0
        skipped_no_target = 0
        skipped_no_features = 0
        skipped_too_long = 0  # Track sequences exceeding max length
        
        # Max sequence length limit (Recommendation 1: Add max length limit)
        MAX_SEQUENCE_LENGTH = 500  # Prevent extremely long sequences
        MAX_EXPECTED_LENGTH = 50   # Warn if sequence exceeds expected length (8 points for 24h, but allow buffer)
        
        # Progress logging
        total_trajectories = len(trajectories)
        log_interval = max(1, total_trajectories // 20)  # Log every 5%
        
        for traj_idx, trajectory in enumerate(trajectories):
            # Progress logging
            if traj_idx % log_interval == 0 or traj_idx == total_trajectories - 1:
                logger.info(f"Processing trajectory {traj_idx + 1}/{total_trajectories} "
                          f"({(traj_idx + 1) / total_trajectories * 100:.1f}%) - "
                          f"Created {len(sequences)} sequences so far...")
            
            # EARLY EXIT: Skip extremely long trajectories to avoid hangs
            # If a trajectory has > 100k points, it's likely invalid or will cause performance issues
            MAX_TRAJECTORY_POINTS = 100000
            if len(trajectory) > MAX_TRAJECTORY_POINTS:
                skipped_short += 1
                if skipped_short <= 5:
                    logger.debug(f"Skipping extremely long trajectory: {len(trajectory)} points (max {MAX_TRAJECTORY_POINTS} allowed)")
                continue
            
            # Ensure sorted by time (trajectories should already be sorted, but verify)
            if not trajectory['BaseDateTime'].is_monotonic_increasing:
                trajectory = trajectory.sort_values('BaseDateTime')
            
            # Check if trajectory has at least 2 points in any 24-hour window
            # AIS reports come every 3 hours, so max 8 points in 24 hours
            # Minimum 2 points ensures we have some movement data
            AIS_REPORTING_INTERVAL_HOURS = 3.0
            MAX_POINTS_PER_24H = 8  # 24 hours / 3 hours = 8 points max
            min_points_in_24h = 2  # Minimum 2 points (6 hours of data) for valid sequence
            
            # Check if trajectory has at least min_points_in_24h points within any 24-hour window
            # FIXED: Don't require trajectory to span 24 hours total - just check for valid windows
            has_valid_24h_window = False
            trajectory_times = trajectory['BaseDateTime'].values
            
            # Need at least min_points_in_24h points to have a valid window
            if len(trajectory) < min_points_in_24h:
                # Trajectory doesn't have enough points at all
                skipped_short += 1
                if skipped_short <= 20:  # Log first 20 for diagnosis
                    logger.warning(f"[{skipped_short}] Skipping trajectory: only {len(trajectory)} points (need at least {min_points_in_24h})")
                continue
            
            # Trajectory has enough points - check for valid 24h window
            if len(trajectory) >= min_points_in_24h:
                try:
                    # FIXED: Handle numpy timedelta64 from .values - convert to pandas for proper time arithmetic
                    # trajectory_times is a numpy array (from .values), so subtraction gives numpy.timedelta64
                    # Convert to pandas Timestamp first to get pandas Timedelta
                    time_start = pd.Timestamp(trajectory_times[0])
                    time_end = pd.Timestamp(trajectory_times[-1])
                    time_span = time_end - time_start
                    
                    # Convert to hours - now time_span is pandas Timedelta
                    time_span_hours = time_span.total_seconds() / 3600.0
                    
                    # FIXED: Check if trajectory spans enough time to create sequences
                    # Need: sequence_length (24h) + prediction_horizon (48h) = 72h minimum
                    min_trajectory_span = self.sequence_length + self.prediction_horizon
                    
                    if time_span_hours < min_trajectory_span:
                        # Trajectory doesn't span enough time to create sequences
                        # But allow trajectories that are shorter if they have enough points
                        # (We'll relax the prediction horizon requirement for shorter trajectories)
                        pass
                    
                    # FIXED: Check for valid 24h window regardless of total trajectory span
                    # A trajectory can have valid sequences even if shorter than 24 hours
                    
                    # Fast path: If trajectory has enough points and spans >= 24h, check first window
                    if time_span_hours >= 24.0:
                        # Check first 24-hour window
                        # Convert to pandas Timestamp for proper arithmetic
                        window_start = pd.Timestamp(trajectory_times[0])
                        window_end = window_start + pd.Timedelta(hours=24.0)
                        # Convert trajectory_times to pandas Timestamp for comparison
                        trajectory_times_ts = pd.to_datetime(trajectory_times)
                        window_mask = (trajectory_times_ts >= window_start) & (trajectory_times_ts < window_end)
                        if np.sum(window_mask) >= min_points_in_24h:
                            has_valid_24h_window = True
                    
                    # FIXED: For shorter trajectories (< 24h), check if ALL points fit in a 24h window
                    # This is valid because we can use a sequence shorter than 24h if it has enough points
                    if not has_valid_24h_window:
                        # If trajectory is shorter than 24h but has enough points, it's valid
                        # (We'll slide through it to create sequences)
                        if time_span_hours < 24.0 and len(trajectory) >= min_points_in_24h:
                            # All points are within a window smaller than 24h, which is valid
                            has_valid_24h_window = True
                        else:
                            # Trajectory is >= 24h but first window check failed
                            # Check a few strategic windows (first, middle, last)
                            check_indices = [0]
                            if len(trajectory) > 2:
                                check_indices.append(len(trajectory) // 2)
                                check_indices.append(len(trajectory) - 1)
                            
                            window_delta = pd.Timedelta(hours=24.0)
                            trajectory_times_ts = pd.to_datetime(trajectory_times)
                            for i in check_indices:
                                window_start = pd.Timestamp(trajectory_times[i])
                                window_end = window_start + window_delta
                                window_mask = (trajectory_times_ts >= window_start) & (trajectory_times_ts < window_end)
                                if np.sum(window_mask) >= min_points_in_24h:
                                    has_valid_24h_window = True
                                    break
                except Exception as e:
                    # If time calculation fails, log error and assume trajectory is invalid
                    if skipped_short <= 5:
                        logger.debug(f"Error checking 24h window for trajectory with {len(trajectory)} points: {e}")
                    has_valid_24h_window = False
            
            if not has_valid_24h_window:
                skipped_short += 1
                # Always log first 20 for detailed diagnosis
                if skipped_short <= 20:
                    try:
                        if len(trajectory) > 0:
                            # FIXED: Convert numpy datetime64 to pandas Timestamp for proper arithmetic
                            time_start = pd.Timestamp(trajectory_times[0])
                            time_end = pd.Timestamp(trajectory_times[-1])
                            time_span = time_end - time_start
                            time_span_hours = time_span.total_seconds() / 3600.0
                        else:
                            time_span_hours = 0
                        
                        # Detailed diagnostic info
                        if time_span_hours >= 24.0:
                            # Check why first window failed
                            window_start = pd.Timestamp(trajectory_times[0])
                            window_end = window_start + pd.Timedelta(hours=24.0)
                            trajectory_times_ts = pd.to_datetime(trajectory_times)
                            window_mask = (trajectory_times_ts >= window_start) & (trajectory_times_ts < window_end)
                            points_in_first_window = np.sum(window_mask)
                            logger.warning(f"[{skipped_short}] Skipping trajectory: {len(trajectory)} points, spans {time_span_hours:.1f}h, "
                                         f"first 24h window has {points_in_first_window} points (need >= {min_points_in_24h})")
                        else:
                            logger.warning(f"[{skipped_short}] Skipping trajectory: {len(trajectory)} points, spans {time_span_hours:.1f}h "
                                         f"(< 24h), but validation failed")
                    except Exception as e:
                        logger.warning(f"[{skipped_short}] Skipping trajectory: {len(trajectory)} points, validation error: {e}")
                continue
            
            # Filter outliers AFTER validation (only for trajectories we'll use)
            # Skip filtering for very long trajectories to avoid performance issues
            if len(trajectory) < 50000:  # Only filter outliers for smaller trajectories
                trajectory = self.trajectory_processor.filter_outliers(trajectory)
            # For large trajectories, outlier filtering could create huge date_ranges
            # The segmentation should have already filtered most outliers
            
            # Create sliding windows based on TIME, not point indices
            # Find all possible starting times that allow for sequence_length hours of data
            trajectory_start = trajectory['BaseDateTime'].iloc[0]
            trajectory_end = trajectory['BaseDateTime'].iloc[-1]
            time_span_hours = (trajectory_end - trajectory_start).total_seconds() / 3600
            
            # FIXED: If trajectory is shorter than sequence_length, we can still create sequences
            # by using the entire trajectory as the input sequence (even if < 24h)
            if time_span_hours < self.sequence_length:
                # Trajectory is shorter than required sequence length
                # Use entire trajectory as one sequence (if it has enough points)
                max_start_time = trajectory_start  # Only one sequence possible
            else:
                # Normal case: trajectory spans >= sequence_length hours
                max_start_time = trajectory_end - pd.Timedelta(hours=self.sequence_length)
            
            # OPTIMIZED: Limit sliding window iterations for very long trajectories
            # Calculate max iterations to prevent excessive processing
            slide_hours = 6
            if time_span_hours >= self.sequence_length:
                max_iterations = int((time_span_hours - self.sequence_length) / slide_hours) + 1
            else:
                # Short trajectory: only one sequence possible
                max_iterations = 1
            
            # Limit to reasonable number of sequences per trajectory (max 100)
            # This prevents one very long trajectory from dominating processing time
            if max_iterations > 100:
                # For very long trajectories, increase slide interval
                slide_hours = max(6, int(time_span_hours / 100))
                max_iterations = 100
                logger.debug(f"Long trajectory ({time_span_hours:.1f}h): using {slide_hours}h slide interval")
            
            # Slide by slide_hours through time
            current_start_time = trajectory_start
            iteration_count = 0
            
            while current_start_time <= max_start_time and iteration_count < max_iterations:
                iteration_count += 1
                
                # FIXED: For short trajectories (< sequence_length), use entire trajectory
                if time_span_hours < self.sequence_length:
                    # Use entire trajectory as input sequence
                    input_seq = trajectory.copy()
                    input_end_time = trajectory_end
                else:
                    # Normal case: Input sequence: sequence_length hours starting from current_start_time
                    input_end_time = current_start_time + pd.Timedelta(hours=self.sequence_length)
                    
                    # Find all points in the input window
                    input_seq = trajectory[
                        (trajectory['BaseDateTime'] >= current_start_time) &
                        (trajectory['BaseDateTime'] < input_end_time)
                    ].copy()
                
                # Need at least min_points_in_24h points in the input window
                if len(input_seq) < min_points_in_24h:
                    current_start_time += pd.Timedelta(hours=slide_hours)
                    continue
                
                # WARNING: Check for unexpectedly long sequences
                # AIS reports every 3 hours, so 24h window should have max 8 points
                # If sequence has many more points, log a warning
                if len(input_seq) > 50:
                    logger.debug(f"Warning: Sequence has {len(input_seq)} points in 24h window "
                               f"(expected max 8). Trajectory may have high-frequency data.")
                
                # Target sequence: prediction_horizon hours ahead (or closest available point)
                input_end_time = input_seq['BaseDateTime'].iloc[-1]
                
                # Find all points after the input window
                future_trajectory = trajectory[
                    trajectory['BaseDateTime'] > input_end_time  # Use > not >= to avoid using input_end_time itself
                ].copy()
                
                if len(future_trajectory) == 0:
                    skipped_no_target += 1
                    current_start_time += pd.Timedelta(hours=slide_hours)
                    continue
                
                # FIXED: Use available future data even if less than prediction_horizon
                # Calculate ideal target time
                target_time = input_end_time + pd.Timedelta(hours=self.prediction_horizon)
                
                # Find the point closest to target_time (or furthest available if trajectory ends early)
                time_diffs = (future_trajectory['BaseDateTime'] - target_time).abs()
                closest_idx = time_diffs.idxmin()
                
                # Get the target point
                target_seq = future_trajectory.loc[[closest_idx]].copy()
                
                # Calculate actual time difference between input end and target
                time_diff = (target_seq['BaseDateTime'].iloc[-1] - 
                           input_seq['BaseDateTime'].iloc[-1]).total_seconds() / 3600
                
                # FIXED: Accept sequences with future data, even if less than prediction_horizon
                # Minimum acceptable: at least 1 hour ahead (any future data is better than none)
                if time_diff <= 0:
                    skipped_no_target += 1
                    current_start_time += pd.Timedelta(hours=slide_hours)
                    continue
                
                # Warn if future data is significantly less than prediction_horizon
                if time_diff < self.prediction_horizon * 0.5:  # Less than 50% of desired horizon
                    if skipped_no_target < 10:  # Log first 10
                        logger.debug(f"Using future data {time_diff:.1f}h ahead (desired: {self.prediction_horizon}h) "
                                   f"for sequence ending at {input_end_time}")
                
                # Extract features
                input_features = self.feature_engineer.create_sequence_features(
                    input_seq, 
                    sequence_length=self.sequence_length
                )
                
                if not input_features:
                    skipped_no_features += 1
                    current_start_time += pd.Timedelta(hours=slide_hours)
                    continue
                
                # Recommendation 1: Check sequence length and apply max limit
                input_sequence = input_features['input_sequence']
                seq_len = len(input_sequence)
                
                # Skip sequences that are too long (memory/performance issue)
                if seq_len > MAX_SEQUENCE_LENGTH:
                    skipped_too_long += 1
                    if skipped_too_long <= 5:  # Log first 5
                        logger.warning(f"Skipping sequence with {seq_len} points (exceeds max {MAX_SEQUENCE_LENGTH}). "
                                     f"Expected max ~8 points for 24h window (AIS every 3h).")
                    current_start_time += pd.Timedelta(hours=slide_hours)
                    continue
                
                # Log warning for sequences longer than expected but within limit
                if seq_len > MAX_EXPECTED_LENGTH:
                    if skipped_too_long <= 10:  # Log first 10 warnings
                        logger.debug(f"Warning: Sequence has {seq_len} points in 24h window "
                                   f"(expected max ~8 points, AIS reports every 3h).")
                
                # Create sequence dictionary
                sequence = {
                    'mmsi': trajectory['MMSI'].iloc[0] if 'MMSI' in trajectory.columns else None,
                    'input_sequence': input_sequence,
                    'input_times': input_seq['BaseDateTime'].values[:len(input_sequence)] if len(input_seq['BaseDateTime'].values) > len(input_sequence) else input_seq['BaseDateTime'].values,
                    'target_positions': target_seq[['LAT', 'LON']].values,
                    'target_times': target_seq['BaseDateTime'].values,
                    'target_sog': target_seq['SOG'].values if 'SOG' in target_seq.columns else None,
                    'target_cog': target_seq['COG'].values if 'COG' in target_seq.columns else None,
                    'feature_columns': input_features['feature_columns'],
                    'last_known_time': input_seq['BaseDateTime'].iloc[-1],
                    'prediction_time': target_seq['BaseDateTime'].iloc[-1]
                }
                
                sequences.append(sequence)
                
                # Move to next window
                current_start_time += pd.Timedelta(hours=slide_hours)
        
        # Recommendation 1: Log sequence length statistics
        if len(sequences) > 0:
            seq_lengths = [len(seq['input_sequence']) for seq in sequences]
            # np is already imported at module level
            seq_lengths_arr = np.array(seq_lengths)
            
            logger.info("\nSequence Length Statistics:")
            logger.info(f"  Total sequences: {len(sequences):,}")
            logger.info(f"  Min length: {seq_lengths_arr.min()} points")
            logger.info(f"  Max length: {seq_lengths_arr.max()} points")
            logger.info(f"  Mean length: {seq_lengths_arr.mean():.2f} points")
            logger.info(f"  Median length: {np.median(seq_lengths_arr):.2f} points")
            logger.info(f"  Std dev: {seq_lengths_arr.std():.2f} points")
            
            # Count sequences by length ranges
            very_short = np.sum(seq_lengths_arr < 2)
            short = np.sum((seq_lengths_arr >= 2) & (seq_lengths_arr < 5))
            expected = np.sum((seq_lengths_arr >= 5) & (seq_lengths_arr <= 8))
            long = np.sum((seq_lengths_arr > 8) & (seq_lengths_arr <= 50))
            very_long = np.sum(seq_lengths_arr > 50)
            
            logger.info(f"  Length distribution:")
            logger.info(f"    < 2 points: {very_short} ({very_short/len(sequences)*100:.1f}%)")
            logger.info(f"    2-4 points: {short} ({short/len(sequences)*100:.1f}%)")
            logger.info(f"    5-8 points (expected): {expected} ({expected/len(sequences)*100:.1f}%)")
            logger.info(f"    9-50 points: {long} ({long/len(sequences)*100:.1f}%)")
            logger.info(f"    > 50 points: {very_long} ({very_long/len(sequences)*100:.1f}%)")
            
            if very_long > 0:
                logger.warning(f"  WARNING: {very_long} sequences exceed 50 points (expected max 8 for 24h window)")
            
            # Log feature dimensions (Recommendation 2)
            if sequences:
                sample_seq = sequences[0]
                if hasattr(sample_seq['input_sequence'], 'shape'):
                    feat_dim = sample_seq['input_sequence'].shape[1]
                else:
                    feat_dim = len(sample_seq['input_sequence'][0])
                
                # Verify all sequences have same feature dimension
                all_feat_dims = []
                for seq in sequences[:100]:  # Check first 100
                    if hasattr(seq['input_sequence'], 'shape'):
                        all_feat_dims.append(seq['input_sequence'].shape[1])
                    else:
                        all_feat_dims.append(len(seq['input_sequence'][0]))
                
                if len(set(all_feat_dims)) == 1:
                    logger.info(f"\nFeature Dimension: {feat_dim} ([OK] consistent across sequences)")
                else:
                    logger.warning(f"\nFeature Dimension: ⚠️ Varies! Found: {set(all_feat_dims)}")
        
        # Log summary (detailed stats)
        if skipped_short > 0:
            logger.info(f"\nSkipped {skipped_short} trajectories: too short")
        if skipped_no_target > 0:
            logger.info(f"Skipped {skipped_no_target} sequences: no future data")
        if skipped_no_features > 0:
            logger.info(f"Skipped {skipped_no_features} sequences: feature extraction failed")
        if skipped_too_long > 0:
            logger.warning(f"Skipped {skipped_too_long} sequences: too long (exceeded {MAX_SEQUENCE_LENGTH} points)")
        
        if len(sequences) == 0:
            logger.warning("WARNING: No training sequences created! Check filtering parameters.")
            logger.warning(f"  Trajectories created: {len(trajectories)}")
            logger.warning(f"  Sequence length required: {self.sequence_length} hours")
            logger.warning(f"  Prediction horizon: {self.prediction_horizon} hours")
        
        # Stage 4: Filter vessel types AFTER sequence preparation
        # Only filter if vessel's MMSI didn't have at least one vessel type report in a 24-hour period
        if filter_unknown_vessel_types and len(sequences) > 0:
            sequences = self._filter_sequences_by_vessel_type(df, sequences)
            logger.info(f"After vessel type filtering: {len(sequences)} sequences remaining")
        
        logger.info(f"Created {len(sequences)} training sequences")
        
        # Save to cache for future use (avoiding 12+ hour reprocessing on subsequent runs)
        # Only save if cache_path was generated and caching is enabled
        if use_cache and cache_path is not None:
            try:
                logger.info(f"Saving sequences to cache: {cache_path.name}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(sequences, f)
                logger.info(f"Cached {len(sequences):,} sequences for future use")
                logger.info(f"Cache location: {cache_path}")
            except Exception as e:
                logger.warning(f"Error saving sequences to cache: {e}")
        elif use_cache and cache_path is None:
            # Cache path wasn't generated - try to generate it now if we can
            try:
                if hasattr(self, 'sequences_cache_dir') and self.sequences_cache_dir:
                    cache_key = self._get_sequences_cache_key(df, start_date, end_date, vessel_types, filter_unknown_vessel_types)
                    cache_path = self.sequences_cache_dir / f"sequences_{cache_key}.pkl"
                    logger.info(f"Saving sequences to cache: {cache_path.name}")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(sequences, f)
                    logger.info(f"Cached {len(sequences):,} sequences for future use")
                    logger.info(f"Cache location: {cache_path}")
            except Exception as e:
                logger.debug(f"Could not save to cache: {e}")
        
        return sequences
    
    def _get_sequences_cache_key(self, df: pd.DataFrame, 
                                 start_date: Optional[str],
                                 end_date: Optional[str],
                                 vessel_types: Optional[List[int]],
                                 filter_unknown_vessel_types: bool) -> str:
        """
        Generate cache key for sequences based on input parameters
        
        Args:
            df: DataFrame with AIS data
            start_date: Start date (optional)
            end_date: End date (optional)
            vessel_types: List of vessel types (optional)
            filter_unknown_vessel_types: Whether filtering unknown types
            
        Returns:
            Cache key string (MD5 hash)
        """
        # Build cache parameters (validate required attributes first)
        # These attributes must exist (set in __init__), so we access them directly
        # If they don't exist, it indicates improper initialization
        try:
            sequence_length = self.sequence_length
            prediction_horizon = self.prediction_horizon
            max_gap_hours = self.max_gap_hours
        except AttributeError as e:
            logger.error(f"Missing required attribute: {e}")
            logger.error("CoursePredictionDataLoader must be initialized with sequence_length, prediction_horizon, and max_gap_hours")
            raise
        
        cache_params = {
            'cache_version': getattr(self, 'sequence_cache_version_hash', 'v1.0'),
            'filter_unknown_types': filter_unknown_vessel_types,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'max_gap_hours': max_gap_hours
        }
        
        # Add date range if provided
        if start_date and end_date:
            cache_params['start_date'] = start_date
            cache_params['end_date'] = end_date
        else:
            # Use data range from DataFrame if available
            if 'BaseDateTime' in df.columns and len(df) > 0:
                try:
                    min_date = df['BaseDateTime'].min()
                    max_date = df['BaseDateTime'].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        cache_params['data_start'] = str(min_date.date()) if hasattr(min_date, 'date') else str(min_date)
                        cache_params['data_end'] = str(max_date.date()) if hasattr(max_date, 'date') else str(max_date)
                except Exception:
                    pass  # Skip if date extraction fails
        
        # Add vessel types if provided
        if vessel_types:
            cache_params['vessel_types'] = sorted(vessel_types)
        else:
            # Use unique vessel types from data (limit to avoid huge keys)
            if 'VesselType' in df.columns:
                try:
                    unique_types = sorted(df['VesselType'].dropna().unique().tolist())
                    cache_params['vessel_types_data'] = unique_types[:20]  # Limit to avoid huge keys
                except Exception:
                    pass  # Skip if type extraction fails
        
        # Add data hash for uniqueness (based on number of records and vessels)
        try:
            data_hash = hashlib.md5(f"{len(df)}_{df['MMSI'].nunique() if 'MMSI' in df.columns else 0}".encode()).hexdigest()[:8]
            cache_params['data_hash'] = data_hash
        except Exception:
            pass  # Skip if hash generation fails
        
        # Generate cache key
        params_str = json.dumps(cache_params, sort_keys=True)
        cache_key = hashlib.md5(params_str.encode()).hexdigest()
        
        return cache_key
    
    def save_sequences_to_cache(self, 
                                 sequences: List[Dict[str, Any]], 
                                 df: pd.DataFrame,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 vessel_types: Optional[List[int]] = None,
                                 filter_unknown_vessel_types: bool = False) -> bool:
        """
        Standalone function to save sequences to cache after generation.
        This can be called after prepare_training_sequences completes, even if
        the original call didn't use caching.
        
        Args:
            sequences: List of training sequences to cache
            df: Original DataFrame used to generate sequences
            start_date: Start date for cache key (optional)
            end_date: End date for cache key (optional)
            vessel_types: List of vessel types for cache key (optional)
            filter_unknown_vessel_types: Whether unknown vessel types were filtered
            
        Returns:
            True if sequences were saved successfully, False otherwise
        """
        if not sequences:
            logger.warning("No sequences to save to cache")
            return False
        
        try:
            # Ensure cache directory exists
            if not hasattr(self, 'sequences_cache_dir') or not self.sequences_cache_dir:
                # Try to set up cache directory
                if hasattr(self, 'data_preprocessor') and self.data_preprocessor:
                    self.sequences_cache_dir = self.data_preprocessor.cache_dir / 'sequences'
                else:
                    training_dir = Path(__file__).resolve().parent
                    self.sequences_cache_dir = training_dir / 'cache' / 'sequences'
                
                # Create directory
                self.sequences_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache key
            cache_key = self._get_sequences_cache_key(df, start_date, end_date, vessel_types, filter_unknown_vessel_types)
            cache_path = self.sequences_cache_dir / f"sequences_{cache_key}.pkl"
            
            # Check if cache already exists
            if cache_path.exists():
                logger.info(f"Cache already exists for this configuration: {cache_path.name}")
                logger.info("Skipping cache save (cache is up to date)")
                return True
            
            # Save sequences to cache
            logger.info(f"Saving {len(sequences):,} sequences to cache: {cache_path.name}")
            with open(cache_path, 'wb') as f:
                pickle.dump(sequences, f)
            
            logger.info(f"[OK] Cached {len(sequences):,} sequences for future use")
            logger.info(f"Cache location: {cache_path}")
            
            # Get file size for info
            try:
                cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
                logger.info(f"Cache size: {cache_size_mb:.2f} MB")
            except Exception:
                pass  # File size check is optional
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving sequences to cache: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    def _filter_sequences_by_vessel_type(self, df: pd.DataFrame, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter sequences based on vessel type reporting.
        Only removes sequences if vessel's MMSI didn't have at least one vessel type report in a 24-hour period.
        
        Args:
            df: Original DataFrame with vessel data
            sequences: List of training sequences
            
        Returns:
            Filtered list of sequences
        """
        if not sequences or df.empty or 'VesselType' not in df.columns:
            return sequences
        
        logger.info("Filtering sequences by vessel type reporting (24-hour check)")
        
        # Group by MMSI and check for vessel type reports in 24-hour windows
        df_sorted = df.sort_values(['MMSI', 'BaseDateTime'])
        
        # For each MMSI, check if it has at least one valid vessel type in any 24-hour window
        mmsi_with_valid_type = set()
        
        for mmsi, vessel_df in df_sorted.groupby('MMSI'):
            # Check if vessel has any valid vessel type (20-99) in the data
            valid_types = vessel_df[
                (vessel_df['VesselType'] >= 20) & 
                (vessel_df['VesselType'] <= 99) &
                (vessel_df['VesselType'].notna())
            ]
            
            if len(valid_types) > 0:
                # Check if there's at least one valid type report in any 24-hour period
                vessel_df_sorted = vessel_df.sort_values('BaseDateTime')
                
                # Slide 24-hour window and check for valid types
                window_hours = 24
                has_valid_type_in_window = False
                
                for i in range(len(vessel_df_sorted)):
                    window_start = vessel_df_sorted.iloc[i]['BaseDateTime']
                    window_end = window_start + pd.Timedelta(hours=window_hours)
                    
                    window_data = vessel_df_sorted[
                        (vessel_df_sorted['BaseDateTime'] >= window_start) &
                        (vessel_df_sorted['BaseDateTime'] < window_end)
                    ]
                    
                    # Check if window has at least one valid vessel type
                    if len(window_data) > 0:
                        valid_in_window = window_data[
                            (window_data['VesselType'] >= 20) & 
                            (window_data['VesselType'] <= 99) &
                            (window_data['VesselType'].notna())
                        ]
                        if len(valid_in_window) > 0:
                            has_valid_type_in_window = True
                            break
                
                if has_valid_type_in_window:
                    mmsi_with_valid_type.add(mmsi)
        
        logger.info(f"Found {len(mmsi_with_valid_type):,} vessels with valid type reports in 24-hour windows")
        
        # Filter sequences to only include vessels with valid types
        filtered_sequences = []
        for seq in sequences:
            mmsi = seq.get('mmsi')
            if mmsi and mmsi in mmsi_with_valid_type:
                filtered_sequences.append(seq)
        
        removed_count = len(sequences) - len(filtered_sequences)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} sequences from vessels without valid type reports in 24h windows")
        
        return filtered_sequences
    
    def split_train_val_test(self, sequences: List[Dict[str, Any]],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """
        Split sequences into train/validation/test sets
        
        Args:
            sequences: List of sequence dictionaries
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_sequences, val_sequences, test_sequences)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
        
        # Shuffle sequences
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(sequences))
        sequences_shuffled = [sequences[i] for i in indices]
        
        # Calculate split indices
        n_total = len(sequences_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_sequences = sequences_shuffled[:n_train]
        val_sequences = sequences_shuffled[n_train:n_train + n_val]
        test_sequences = sequences_shuffled[n_train + n_val:]
        
        logger.info(f"Split sequences: {len(train_sequences)} train, "
                   f"{len(val_sequences)} val, {len(test_sequences)} test")
        
        return train_sequences, val_sequences, test_sequences

