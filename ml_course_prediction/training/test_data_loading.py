"""
Test script for data loading and preprocessing
Tests the historical data loading pipeline with all vessel types
Analyzes movement patterns for different vessel types
"""
import asyncio
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_course_prediction.training.data_loader import CoursePredictionDataLoader
from ml_course_prediction.utils.data_preprocessing import HistoricalDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Vessel type categories for analysis
VESSEL_TYPE_CATEGORIES = {
    'Fishing': [30],
    'Cargo': [70, 71, 72, 73, 74, 79],
    'Tanker': [80, 81, 82, 83, 84, 89],
    'Passenger': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    'Towing': [31, 32],
    'Special Purpose': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    'HSC': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'Other': [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
}


async def test_data_loading():
    """Test data loading from historical data directory"""
    
    # Initialize data loader with historical data path
    data_path = r"C:\AIS_Data_Testing\Historical\2024"
    
    logger.info("=" * 60)
    logger.info("Testing Historical Data Loading")
    logger.info("=" * 60)
    
    # Test 1: Load single day
    logger.info("\nTest 1: Loading single day (2024-01-01)")
    loader = CoursePredictionDataLoader(
        historical_data_path=data_path,
        sequence_length=24,
        prediction_horizon=48
    )
    
    df = await loader.load_training_data(
        start_date="2024-01-01",
        end_date="2024-01-01",
        preprocess=True
    )
    
    if not df.empty:
        logger.info(f"[OK] Successfully loaded {len(df):,} records")
        logger.info(f"  Unique vessels: {df['MMSI'].nunique():,}")
        logger.info(f"  Date range: {df['BaseDateTime'].min()} to {df['BaseDateTime'].max()}")
        logger.info(f"  Columns: {list(df.columns)}")
    else:
        logger.error("✗ Failed to load data")
        return
    
    # Test 2: Load date range
    logger.info("\nTest 2: Loading date range (2024-01-01 to 2024-01-03)")
    df_range = await loader.load_training_data(
        start_date="2024-01-01",
        end_date="2024-01-03",
        preprocess=True
    )
    
    if not df_range.empty:
        logger.info(f"[OK] Successfully loaded {len(df_range):,} records")
        logger.info(f"  Unique vessels: {df_range['MMSI'].nunique():,}")
        logger.info(f"  Date range: {df_range['BaseDateTime'].min()} to {df_range['BaseDateTime'].max()}")
    else:
        logger.error("✗ Failed to load date range")
        return
    
    # Test 3: Analyze all vessel types
    logger.info("\nTest 3: Analyzing all vessel types and their patterns")
    analyze_vessel_types(loader, df)
    
    # Test 3a: Load data for each major vessel type category
    logger.info("\nTest 3a: Loading data by vessel type category")
    vessel_type_results = {}
    
    for category, type_codes in VESSEL_TYPE_CATEGORIES.items():
        logger.info(f"\n  Testing {category} vessels (types {type_codes})")
        df_type = await loader.load_training_data(
            start_date="2024-01-01",
            end_date="2024-01-01",
            vessel_types=type_codes,
            preprocess=True
        )
        
        if not df_type.empty:
            vessel_type_results[category] = {
                'records': len(df_type),
                'unique_vessels': df_type['MMSI'].nunique(),
                'dataframe': df_type
            }
            logger.info(f"  [OK] {category}: {len(df_type):,} records, {df_type['MMSI'].nunique():,} vessels")
        else:
            vessel_type_results[category] = None
            logger.warning(f"  ⚠ No {category} vessels found")
    
    # Test 3b: Analyze movement patterns by vessel type
    logger.info("\nTest 3b: Analyzing movement patterns by vessel type")
    if vessel_type_results:
        analyze_movement_patterns(vessel_type_results)
    
    # Test 4: Data preprocessing
    logger.info("\nTest 4: Testing data preprocessing directly")
    preprocessor = HistoricalDataPreprocessor(data_path=data_path)
    
    # Load raw data
    raw_df = preprocessor.load_date_file("2024_01_01")
    if not raw_df.empty:
        logger.info(f"  Raw data: {len(raw_df):,} records")
        
        # Preprocess
        processed_df = preprocessor.preprocess_dataframe(raw_df)
        logger.info(f"  Processed data: {len(processed_df):,} records")
        
        # Get summary
        summary = preprocessor.get_data_summary(processed_df)
        logger.info(f"  Summary: {summary.get('unique_vessels', 0):,} unique vessels")
    
    # Test 5: Prepare training sequences for all vessel types
    logger.info("\nTest 5: Preparing training sequences for all vessel types")
    if not df.empty:
        # Don't filter vessel types during sequence preparation
        sequences = loader.prepare_training_sequences(df, filter_unknown_vessel_types=False)
        logger.info(f"[OK] Created {len(sequences)} total training sequences")
        
        if sequences:
            # Analyze sequences by vessel type
            sequences_by_type = analyze_sequences_by_vessel_type(df, sequences)
            
            logger.info(f"\n  Sequences by vessel type category:")
            for category, seq_count in sequences_by_type.items():
                logger.info(f"    {category}: {seq_count} sequences")
            
            # Show sample sequences from different types
            sample_seq = sequences[0]
            logger.info(f"\n  Sample sequence:")
            logger.info(f"    MMSI: {sample_seq.get('mmsi', 'N/A')}")
            logger.info(f"    Input shape: {sample_seq['input_sequence'].shape}")
            logger.info(f"    Target positions: {len(sample_seq['target_positions'])}")
            logger.info(f"    Feature columns: {sample_seq.get('feature_columns', [])}")
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


def analyze_vessel_types(loader: CoursePredictionDataLoader, df: pd.DataFrame):
    """
    Analyze vessel type distribution in the dataset
    
    Args:
        loader: Data loader instance
        df: Loaded DataFrame
    """
    if df.empty or 'VesselType' not in df.columns:
        logger.warning("Cannot analyze vessel types - no data or VesselType column")
        return
    
    logger.info("  Vessel type distribution:")
    
    # Get vessel type counts
    vessel_type_counts = df['VesselType'].value_counts().sort_index()
    total_vessels = df['MMSI'].nunique()
    
    # Group by category
    category_counts = defaultdict(int)
    category_vessels = defaultdict(set)
    
    for vessel_type, count in vessel_type_counts.items():
        # Find category
        category = None
        for cat, codes in VESSEL_TYPE_CATEGORIES.items():
            if int(vessel_type) in codes:
                category = cat
                break
        
        if category:
            category_counts[category] += count
            # Count unique vessels for this type
            type_vessels = df[df['VesselType'] == vessel_type]['MMSI'].unique()
            category_vessels[category].update(type_vessels)
        else:
            category_counts['Unknown'] += count
    
    # Log summary
    logger.info(f"  Total unique vessels: {total_vessels:,}")
    logger.info(f"\n  Records by category:")
    for category in sorted(category_counts.keys()):
        records = category_counts[category]
        vessels = len(category_vessels[category])
        pct_records = (records / len(df) * 100) if len(df) > 0 else 0
        pct_vessels = (vessels / total_vessels * 100) if total_vessels > 0 else 0
        logger.info(f"    {category:20s}: {records:>10,} records ({pct_records:>5.1f}%), "
                   f"{vessels:>6,} vessels ({pct_vessels:>5.1f}%)")
    
    # Top 10 individual vessel types
    logger.info(f"\n  Top 10 individual vessel types:")
    for vessel_type, count in vessel_type_counts.head(10).items():
        vessels = df[df['VesselType'] == vessel_type]['MMSI'].nunique()
        logger.info(f"    Type {int(vessel_type):3d}: {count:>10,} records, {vessels:>6,} vessels")


def analyze_movement_patterns(vessel_type_results: Dict[str, Any]):
    """
    Analyze movement patterns for different vessel types
    
    Args:
        vessel_type_results: Dictionary with vessel type data
    """
    logger.info("  Movement pattern analysis:")
    
    pattern_stats = {}
    
    for category, result in vessel_type_results.items():
        if result is None or result['dataframe'] is None:
            continue
        
        df = result['dataframe']
        stats = {}
        
        # Speed statistics
        if 'SOG' in df.columns:
            sog_valid = df['SOG'].dropna()
            if len(sog_valid) > 0:
                stats['speed_mean'] = sog_valid.mean()
                stats['speed_std'] = sog_valid.std()
                stats['speed_max'] = sog_valid.max()
                stats['speed_median'] = sog_valid.median()
        
        # Course consistency (circular statistics)
        if 'COG' in df.columns:
            cog_valid = df['COG'].dropna()
            if len(cog_valid) > 0:
                # Convert to radians for circular mean
                cog_rad = np.radians(cog_valid)
                sin_mean = np.sin(cog_rad).mean()
                cos_mean = np.cos(cog_rad).mean()
                mean_resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
                stats['course_consistency'] = mean_resultant_length  # 0-1, higher = more consistent
        
        # Position spread (geographic dispersion)
        if 'LAT' in df.columns and 'LON' in df.columns:
            lat_std = df['LAT'].std()
            lon_std = df['LON'].std()
            stats['geographic_spread_lat'] = lat_std
            stats['geographic_spread_lon'] = lon_std
        
        # Records per vessel (activity level)
        stats['records_per_vessel'] = len(df) / result['unique_vessels'] if result['unique_vessels'] > 0 else 0
        
        pattern_stats[category] = stats
    
    # Log pattern comparison
    logger.info(f"\n  Pattern comparison:")
    logger.info(f"    {'Category':<20} {'Mean Speed':<12} {'Speed Std':<12} {'Course Consist':<15} {'Rec/Vessel':<12}")
    logger.info(f"    {'-'*20} {'-'*12} {'-'*12} {'-'*15} {'-'*12}")
    
    for category, stats in sorted(pattern_stats.items()):
        speed_mean = stats.get('speed_mean', 0)
        speed_std = stats.get('speed_std', 0)
        course_consist = stats.get('course_consistency', 0)
        rec_per_vessel = stats.get('records_per_vessel', 0)
        
        logger.info(f"    {category:<20} {speed_mean:>10.2f} kt  {speed_std:>10.2f} kt  "
                   f"{course_consist:>13.3f}      {rec_per_vessel:>10.1f}")
    
    # Insights
    logger.info(f"\n  Pattern insights:")
    
    # Find fastest category
    fastest = max(pattern_stats.items(), 
                 key=lambda x: x[1].get('speed_mean', 0))
    logger.info(f"    Fastest average speed: {fastest[0]} ({fastest[1].get('speed_mean', 0):.2f} kt)")
    
    # Find most consistent course
    most_consistent = max(pattern_stats.items(),
                         key=lambda x: x[1].get('course_consistency', 0))
    logger.info(f"    Most consistent course: {most_consistent[0]} "
               f"({most_consistent[1].get('course_consistency', 0):.3f})")
    
    # Find most active (records per vessel)
    most_active = max(pattern_stats.items(),
                     key=lambda x: x[1].get('records_per_vessel', 0))
    logger.info(f"    Most active (records/vessel): {most_active[0]} "
               f"({most_active[1].get('records_per_vessel', 0):.1f} records/vessel)")


def analyze_sequences_by_vessel_type(df: pd.DataFrame, sequences: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Analyze training sequences by vessel type category
    
    Args:
        df: Original DataFrame with vessel data
        sequences: List of training sequences
        
    Returns:
        Dictionary mapping vessel type category to sequence count
    """
    if not sequences or df.empty:
        return {}
    
    # Create MMSI to vessel type mapping
    mmsi_to_type = df.groupby('MMSI')['VesselType'].first().to_dict()
    
    # Count sequences by category
    category_counts = defaultdict(int)
    
    for seq in sequences:
        mmsi = seq.get('mmsi')
        if mmsi and mmsi in mmsi_to_type:
            vessel_type = int(mmsi_to_type[mmsi])
            
            # Find category
            category = 'Unknown'
            for cat, codes in VESSEL_TYPE_CATEGORIES.items():
                if vessel_type in codes:
                    category = cat
                    break
            
            category_counts[category] += 1
        else:
            category_counts['Unknown'] += 1
    
    return dict(category_counts)


if __name__ == "__main__":
    asyncio.run(test_data_loading())

