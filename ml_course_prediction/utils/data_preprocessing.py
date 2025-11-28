"""
Data Preprocessing for Historical AIS Data
Handles loading and preprocessing of parquet files from historical data directory
Supports day-by-day processing and caching to avoid memory issues
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import hashlib
import json

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class HistoricalDataPreprocessor:
    """
    Preprocesses historical AIS data from parquet files for ML training.
    Handles data cleaning, validation, and transformation.
    """
    
    def __init__(self,
                 data_path: str = r"C:\\AIS_Data_Testing\\Historical\\2024",
                 max_speed_knots: float = 100.0,
                 min_speed_knots: float = 0.0,
                 valid_lat_range: Tuple[float, float] = (-90, 90),
                 valid_lon_range: Tuple[float, float] = (-180, 180),
                 valid_cog_range: Tuple[float, float] = (0, 360),
                 filter_unknown_vessel_types: bool = False,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to historical data directory
            max_speed_knots: Maximum valid speed (knots)
            min_speed_knots: Minimum valid speed (knots)
            valid_lat_range: Valid latitude range
            valid_lon_range: Valid longitude range
            valid_cog_range: Valid course over ground range
            filter_unknown_vessel_types: Whether to filter out unknown vessel types (VesselType=0) (default: False)
            cache_dir: Directory for cached preprocessed data (default: training/cache/)
            use_cache: Whether to use cache for loading/saving preprocessed data
        """
        self.data_path = Path(data_path)
        self.max_speed_knots = max_speed_knots
        self.min_speed_knots = min_speed_knots
        self.valid_lat_range = valid_lat_range
        self.valid_lon_range = valid_lon_range
        self.valid_cog_range = valid_cog_range
        self.filter_unknown_vessel_types = filter_unknown_vessel_types
        self.use_cache = use_cache
        
        # Set up cache directory (default to training/cache/)
        if cache_dir is None:
            # Default: training folder relative to this module
            # utils/data_preprocessing.py -> training/cache/
            # Get absolute path to ensure it works from any working directory
            utils_dir = Path(__file__).resolve().parent
            ml_course_prediction_dir = utils_dir.parent
            training_dir = ml_course_prediction_dir / 'training'
            self.cache_dir = training_dir / 'cache'
        else:
            self.cache_dir = Path(cache_dir).resolve()
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory: {self.cache_dir}")
        
        # Generate cache version hash based on preprocessing parameters
        # This allows invalidating cache if preprocessing logic changes
        cache_params = {
            'max_speed': max_speed_knots,
            'min_speed': min_speed_knots,
            'lat_range': valid_lat_range,
            'lon_range': valid_lon_range,
            'cog_range': valid_cog_range,
            'filter_unknown_types': filter_unknown_vessel_types,
            'version': '1.0'  # Increment if preprocessing logic changes
        }
        params_str = json.dumps(cache_params, sort_keys=True)
        self.cache_version_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
    def load_date_file(self, date: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load AIS data for a specific date
        
        Args:
            date: Date string in format 'YYYY-MM-DD' or 'YYYY_MM_DD'
            chunk_size: If specified, load in chunks (for very large files)
            
        Returns:
            DataFrame with AIS data
        """
        # Normalize date format
        date = date.replace('-', '_')
        
        # Construct filename
        filename = f"AIS_{date}.parquet"
        filepath = self.data_path / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        logger.info(f"Loading {filename}...")
        
        try:
            if chunk_size:
                # Load in chunks for very large files
                chunks = []
                for chunk in pd.read_parquet(filepath, chunksize=chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_parquet(filepath)
            
            logger.info(f"Loaded {len(df):,} records from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def _get_cache_path(self, date: datetime) -> Path:
        """Get cache file path for a specific date"""
        date_str = date.strftime('%Y_%m_%d')
        cache_filename = f"preprocessed_{date_str}_v{self.cache_version_hash}.parquet"
        return self.cache_dir / cache_filename
    
    def _load_cached_date(self, date: datetime) -> Optional[pd.DataFrame]:
        """Load preprocessed data from cache for a specific date"""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(date)
        if cache_path.exists():
            try:
                logger.debug(f"Loading cached data from {cache_path.name}")
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {len(df):,} records from cache")
                return df
            except Exception as e:
                logger.warning(f"Error loading cache file {cache_path}: {e}")
                return None
        return None
    
    def _save_cached_date(self, date: datetime, df: pd.DataFrame):
        """Save preprocessed data to cache for a specific date"""
        if not self.use_cache or df.empty:
            return
        
        cache_path = self._get_cache_path(date)
        try:
            df.to_parquet(cache_path, index=False, compression='snappy')
            logger.debug(f"Cached preprocessed data to {cache_path.name} ({len(df):,} records)")
        except Exception as e:
            logger.warning(f"Error saving cache file {cache_path}: {e}")
    
    def load_date_range(self, 
                       start_date: str,
                       end_date: str,
                       chunk_size: Optional[int] = None,
                       preprocess: bool = True) -> pd.DataFrame:
        """
        Load AIS data for a date range with day-by-day processing and caching
        
        Processes each day separately to avoid memory issues:
        - Checks cache for each day
        - If cached, loads from cache
        - If not cached, loads raw, preprocesses, and caches
        - Combines all preprocessed days
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_size: Chunk size for loading (optional, not used with caching)
            preprocess: Whether to preprocess data (default: True)
            
        Returns:
            Combined DataFrame for date range
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        logger.info(f"Loading data from {start_date} to {end_date} ({total_days} days, day-by-day with cache)")
        
        all_data = []
        current_date = start
        days_loaded = 0
        days_cached = 0
        days_processed = 0
        day_num = 0
        
        while current_date <= end:
            day_num += 1
            date_str = current_date.strftime('%Y_%m_%d')
            date_str_readable = current_date.strftime('%Y-%m-%d')
            
            # Try to load from cache first
            df = None
            if self.use_cache and preprocess:
                df = self._load_cached_date(current_date)
                if df is not None and not df.empty:
                    days_cached += 1
                    logger.info(f"  [{day_num}/{total_days}] {date_str_readable}: [OK] Loaded from cache ({len(df):,} records)")
            
            # If not in cache or not using cache, load and preprocess
            if df is None or df.empty:
                logger.info(f"  [{day_num}/{total_days}] {date_str_readable}: Processing...")
                df = self.load_date_file(date_str, chunk_size=chunk_size)
                
                if not df.empty:
                    if preprocess:
                        # Preprocess this day's data
                        df = self.preprocess_dataframe(df)
                        # Save to cache
                        if self.use_cache:
                            self._save_cached_date(current_date, df)
                        logger.info(f"  [{day_num}/{total_days}] {date_str_readable}: [OK] Processed and cached ({len(df):,} records)")
                        days_processed += 1
                    days_loaded += 1
                else:
                    logger.debug(f"  [{day_num}/{total_days}] {date_str_readable}: No data")
            
            if not df.empty:
                all_data.append(df)
            
            current_date += timedelta(days=1)
        
        if not all_data:
            logger.warning("No data loaded for date range")
            return pd.DataFrame()
        
        # Combine all preprocessed dataframes
        logger.info(f"Combining {len(all_data)} days of data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Combined {len(combined_df):,} total records from {len(all_data)} files")
        if preprocess and self.use_cache:
            logger.info(f"  Cached: {days_cached} days, Processed: {days_processed} days")
        
        return combined_df
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess AIS DataFrame: clean, validate, and transform
        
        Args:
            df: Raw AIS DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        logger.info(f"Preprocessing {initial_count:,} records...")
        
        if initial_count == 0:
            logger.warning("Empty DataFrame provided for preprocessing")
            return df
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Log initial state for debugging
        logger.debug(f"Initial record count: {initial_count:,}")
        if 'VesselType' in df.columns:
            valid_types = ((df['VesselType'] >= 20) & (df['VesselType'] <= 99)).sum()
            unknown_types = ((df['VesselType'] == 0) | (df['VesselType'] == 0.0) | (df['VesselType'].isna())).sum()
            logger.debug(f"Initial VesselType: {valid_types:,} valid, {unknown_types:,} unknown/null")
        
        # 1. Convert BaseDateTime to datetime
        if 'BaseDateTime' in df.columns:
            if df['BaseDateTime'].dtype == 'object':
                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
        
        # 2. Remove rows with invalid timestamps
        invalid_times = df['BaseDateTime'].isna()
        if invalid_times.sum() > 0:
            logger.warning(f"Removing {invalid_times.sum():,} rows with invalid timestamps")
            df = df[~invalid_times]
        
        # 3. Validate and filter positions
        df = self._filter_invalid_positions(df)
        
        # 4. Validate and filter speeds
        df = self._filter_invalid_speeds(df)
        
        # 5. Validate and filter courses
        df = self._filter_invalid_courses(df)
        
        # 6. Handle missing values in optional fields
        df = self._handle_missing_values(df)
        
        # 6a. Filter unknown vessel types if requested
        if self.filter_unknown_vessel_types and 'VesselType' in df.columns:
            # Check for both 0 and 0.0 (in case of float type), and also null/NaN
            unknown_mask = (
                (df['VesselType'] == 0) | 
                (df['VesselType'] == 0.0) | 
                (df['VesselType'].isna())
            )
            unknown_count = unknown_mask.sum()
            if unknown_count > 0:
                df = df[~unknown_mask]
                logger.info(f"Filtered out {unknown_count:,} records with unknown vessel types "
                           f"({len(df):,} records remaining)")
                
                # Safety check: warn if too much data was removed
                if len(df) == 0:
                    logger.error("ERROR: All data was filtered out! Check vessel type filtering logic.")
                elif unknown_count > len(df):
                    logger.warning(f"WARNING: More unknown vessel types ({unknown_count:,}) than remaining data. "
                                 f"Check data quality.")
        
        # 7. Sort by MMSI and time
        df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
        
        final_count = len(df)
        removed = initial_count - final_count
        removal_pct = (removed / initial_count * 100) if initial_count > 0 else 0
        
        logger.info(f"Preprocessing complete: {final_count:,} records ({removal_pct:.2f}% removed)")
        
        return df
    
    def _filter_invalid_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter invalid latitude/longitude values"""
        if 'LAT' not in df.columns or 'LON' not in df.columns:
            return df
        
        # Filter invalid lat/lon
        valid_lat = (df['LAT'] >= self.valid_lat_range[0]) & (df['LAT'] <= self.valid_lat_range[1])
        valid_lon = (df['LON'] >= self.valid_lon_range[0]) & (df['LON'] <= self.valid_lon_range[1])
        
        # Filter zero positions (often indicate missing data)
        not_zero = ~((df['LAT'] == 0) & (df['LON'] == 0))
        
        valid_positions = valid_lat & valid_lon & not_zero
        
        invalid_count = (~valid_positions).sum()
        if invalid_count > 0:
            logger.debug(f"Filtering {invalid_count:,} invalid positions")
            df = df[valid_positions]
        
        return df
    
    def _filter_invalid_speeds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter invalid speed values"""
        if 'SOG' not in df.columns:
            return df
        
        # Filter speeds outside valid range
        # Note: SOG can be NaN (stationary or missing), which is OK
        valid_speed = (
            df['SOG'].isna() | 
            ((df['SOG'] >= self.min_speed_knots) & (df['SOG'] <= self.max_speed_knots))
        )
        
        invalid_count = (~valid_speed).sum()
        if invalid_count > 0:
            logger.debug(f"Filtering {invalid_count:,} invalid speeds")
            df = df[valid_speed]
        
        return df
    
    def _filter_invalid_courses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter invalid course values"""
        if 'COG' not in df.columns:
            return df
        
        # Filter COG outside valid range
        # Note: COG can be NaN, which is OK
        valid_cog = (
            df['COG'].isna() | 
            ((df['COG'] >= self.valid_cog_range[0]) & (df['COG'] <= self.valid_cog_range[1]))
        )
        
        invalid_count = (~valid_cog).sum()
        if invalid_count > 0:
            logger.debug(f"Filtering {invalid_count:,} invalid courses")
            df = df[valid_cog]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in optional fields"""
        # VesselType: Fill with 0 (unknown) if missing
        # Note: AIS vessel type codes are 20-99, so 0 is used to mark unknown/missing
        if 'VesselType' in df.columns:
            # Convert to float first to handle any type issues, then fillna, then convert to int
            df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce')
            df['VesselType'] = df['VesselType'].fillna(0.0)
            # Convert to int, but keep as float if there are decimal values
            # Check if all values are whole numbers
            if (df['VesselType'] % 1 == 0).all():
                df['VesselType'] = df['VesselType'].astype(int)
            else:
                # If there are non-integer values, round them
                df['VesselType'] = df['VesselType'].round().astype(int)
        
        # Length, Width, Draft: Fill with 0 if missing (will be filtered later if needed)
        for col in ['Length', 'Width', 'Draft']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Heading: Can be NaN (valid for stationary vessels)
        # SOG, COG: Can be NaN (valid)
        # VesselName, IMO, CallSign: Can be NaN (optional fields)
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_records': len(df),
            'unique_vessels': df['MMSI'].nunique() if 'MMSI' in df.columns else 0,
            'date_range': {
                'start': str(df['BaseDateTime'].min()) if 'BaseDateTime' in df.columns else None,
                'end': str(df['BaseDateTime'].max()) if 'BaseDateTime' in df.columns else None
            },
            'vessel_types': df['VesselType'].value_counts().to_dict() if 'VesselType' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_quality': {
                'has_sog': 'SOG' in df.columns and df['SOG'].notna().any(),
                'has_cog': 'COG' in df.columns and df['COG'].notna().any(),
                'has_heading': 'Heading' in df.columns and df['Heading'].notna().any(),
                'has_vessel_type': 'VesselType' in df.columns and (df['VesselType'] != 0).any()
            }
        }
        
        return summary
    
    def clear_cache(self, date_range: Optional[Tuple[str, str]] = None):
        """
        Clear cached preprocessed data
        
        Args:
            date_range: Optional tuple of (start_date, end_date) to clear specific range.
                       If None, clears all cache files.
        """
        if not self.cache_dir.exists():
            logger.info("Cache directory does not exist, nothing to clear")
            return
        
        if date_range:
            start_date, end_date = date_range
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            current_date = start
            cleared = 0
            
            while current_date <= end:
                cache_path = self._get_cache_path(current_date)
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        cleared += 1
                    except Exception as e:
                        logger.warning(f"Error deleting cache file {cache_path}: {e}")
                current_date += timedelta(days=1)
            
            logger.info(f"Cleared {cleared} cache files for date range {start_date} to {end_date}")
        else:
            # Clear all cache files
            cache_files = list(self.cache_dir.glob("preprocessed_*.parquet"))
            cleared = 0
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
            
            logger.info(f"Cleared {cleared} cache files from {self.cache_dir}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached files
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                'cache_dir': str(self.cache_dir),
                'exists': False,
                'total_files': 0,
                'total_size_mb': 0
            }
        
        cache_files = list(self.cache_dir.glob("preprocessed_*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        
        return {
            'cache_dir': str(self.cache_dir),
            'exists': True,
            'total_files': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_version': self.cache_version_hash
        }

