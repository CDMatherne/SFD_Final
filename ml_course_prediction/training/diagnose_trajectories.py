"""
Diagnostic script to identify why no trajectories/sequences are being created
"""
import asyncio
import logging
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loader import CoursePredictionDataLoader
from utils.data_preprocessing import HistoricalDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def diagnose_trajectory_creation():
    """Diagnose why trajectories are not being created"""
    
    # Initialize components
    historical_data_path = r"C:\AIS_Data_Testing\Historical\2024"
    preprocessor = HistoricalDataPreprocessor(historical_data_path)
    loader = CoursePredictionDataLoader(
        sequence_length=24,
        prediction_horizon=48,
        historical_data_path=historical_data_path
    )
    
    # Load a small sample of data
    logger.info("Loading sample data...")
    df = await loader.load_training_data(
        start_date="2024-01-01",
        end_date="2024-01-02",  # Just one day for diagnosis
        preprocess=True,
        filter_unknown_vessel_types=False
    )
    
    if df.empty:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(df):,} records from {df['MMSI'].nunique():,} vessels")
    
    # Analyze data distribution
    logger.info("\n=== Data Analysis ===")
    if 'BaseDateTime' in df.columns:
        time_span = (df['BaseDateTime'].max() - df['BaseDateTime'].min()).total_seconds() / 3600
        logger.info(f"Time span: {time_span:.1f} hours")
    
    # Group by MMSI and analyze
    logger.info("\n=== Vessel Analysis ===")
    vessel_stats = []
    for mmsi, vessel_df in df.groupby('MMSI'):
        vessel_df_sorted = vessel_df.sort_values('BaseDateTime')
        time_span_hours = (vessel_df_sorted['BaseDateTime'].max() - 
                          vessel_df_sorted['BaseDateTime'].min()).total_seconds() / 3600
        num_points = len(vessel_df_sorted)
        
        vessel_stats.append({
            'MMSI': mmsi,
            'points': num_points,
            'time_span_hours': time_span_hours,
            'avg_points_per_hour': num_points / max(time_span_hours, 0.01)
        })
    
    vessel_df_stats = pd.DataFrame(vessel_stats)
    
    logger.info(f"\nVessel statistics:")
    logger.info(f"  Total vessels: {len(vessel_df_stats)}")
    logger.info(f"  Vessels with >= 3 points: {(vessel_df_stats['points'] >= 3).sum()}")
    logger.info(f"  Vessels with >= 24 hours span: {(vessel_df_stats['time_span_hours'] >= 24.0).sum()}")
    logger.info(f"  Vessels with both >= 3 points AND >= 24 hours: {((vessel_df_stats['points'] >= 3) & (vessel_df_stats['time_span_hours'] >= 24.0)).sum()}")
    
    logger.info(f"\nPoint count distribution:")
    logger.info(f"  Min: {vessel_df_stats['points'].min()}")
    logger.info(f"  Max: {vessel_df_stats['points'].max()}")
    logger.info(f"  Mean: {vessel_df_stats['points'].mean():.1f}")
    logger.info(f"  Median: {vessel_df_stats['points'].median():.1f}")
    logger.info(f"  25th percentile: {vessel_df_stats['points'].quantile(0.25):.1f}")
    logger.info(f"  75th percentile: {vessel_df_stats['points'].quantile(0.75):.1f}")
    
    logger.info(f"\nTime span distribution (hours):")
    logger.info(f"  Min: {vessel_df_stats['time_span_hours'].min():.2f}")
    logger.info(f"  Max: {vessel_df_stats['time_span_hours'].max():.2f}")
    logger.info(f"  Mean: {vessel_df_stats['time_span_hours'].mean():.2f}")
    logger.info(f"  Median: {vessel_df_stats['time_span_hours'].median():.2f}")
    logger.info(f"  25th percentile: {vessel_df_stats['time_span_hours'].quantile(0.25):.2f}")
    logger.info(f"  75th percentile: {vessel_df_stats['time_span_hours'].quantile(0.75):.2f}")
    
    # Check trajectory segmentation
    logger.info("\n=== Trajectory Segmentation Analysis ===")
    trajectories = loader.trajectory_processor.segment_trajectories(df)
    logger.info(f"Trajectories created: {len(trajectories)}")
    
    if len(trajectories) == 0:
        logger.warning("\n*** NO TRAJECTORIES CREATED ***")
        logger.info(f"Trajectory processor settings:")
        logger.info(f"  min_trajectory_points: {loader.trajectory_processor.min_trajectory_points}")
        logger.info(f"  min_trajectory_hours: {loader.trajectory_processor.min_trajectory_hours}")
        logger.info(f"  max_gap_hours: {loader.trajectory_processor.max_gap_hours}")
        
        # Check what's happening in segmentation
        logger.info("\nChecking segmentation logic...")
        for mmsi, vessel_df in df.groupby('MMSI'):
            vessel_df_sorted = vessel_df.sort_values('BaseDateTime')
            time_span_hours = (vessel_df_sorted['BaseDateTime'].max() - 
                             vessel_df_sorted['BaseDateTime'].min()).total_seconds() / 3600
            
            if len(vessel_df_sorted) < loader.trajectory_processor.min_trajectory_points:
                logger.debug(f"  MMSI {mmsi}: Skipped - only {len(vessel_df_sorted)} points (need {loader.trajectory_processor.min_trajectory_points})")
                continue
            
            if time_span_hours < loader.trajectory_processor.min_trajectory_hours:
                logger.debug(f"  MMSI {mmsi}: Skipped - only {time_span_hours:.2f}h span (need {loader.trajectory_processor.min_trajectory_hours}h)")
                continue
            
            # Check gap-based segmentation
            time_diffs = vessel_df_sorted['BaseDateTime'].diff().dt.total_seconds() / 3600
            gaps = time_diffs[time_diffs > loader.trajectory_processor.max_gap_hours]
            if len(gaps) > 0:
                logger.debug(f"  MMSI {mmsi}: Would be split at {len(gaps)} gaps")
            
            # Check if segments would pass validation
            segments_created = 0
            split_indices = [0]
            for idx, gap in enumerate(time_diffs):
                if gap > loader.trajectory_processor.max_gap_hours:
                    split_indices.append(idx)
            split_indices.append(len(vessel_df_sorted))
            
            for i in range(len(split_indices) - 1):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]
                segment = vessel_df_sorted.iloc[start_idx:end_idx].copy()
                seg_time_span = (segment['BaseDateTime'].max() - segment['BaseDateTime'].min()).total_seconds() / 3600
                
                if len(segment) >= loader.trajectory_processor.min_trajectory_points and \
                   seg_time_span >= loader.trajectory_processor.min_trajectory_hours:
                    segments_created += 1
                else:
                    logger.debug(f"  MMSI {mmsi} segment {i}: {len(segment)} points, {seg_time_span:.2f}h - REJECTED")
        
        # Show sample of vessels that should pass
        logger.info("\nSample vessels that should create trajectories:")
        sample_vessels = vessel_df_stats[
            (vessel_df_stats['points'] >= loader.trajectory_processor.min_trajectory_points) &
            (vessel_df_stats['time_span_hours'] >= loader.trajectory_processor.min_trajectory_hours)
        ].head(5)
        
        if len(sample_vessels) > 0:
            for _, row in sample_vessels.iterrows():
                logger.info(f"  MMSI {row['MMSI']}: {row['points']} points, {row['time_span_hours']:.2f}h span")
        else:
            logger.warning("  No vessels meet the criteria!")
    else:
        logger.info(f"\nTrajectory statistics:")
        traj_stats = []
        for traj in trajectories:
            time_span = (traj['BaseDateTime'].max() - traj['BaseDateTime'].min()).total_seconds() / 3600
            traj_stats.append({
                'points': len(traj),
                'time_span_hours': time_span
            })
        
        traj_df_stats = pd.DataFrame(traj_stats)
        logger.info(f"  Min points: {traj_df_stats['points'].min()}")
        logger.info(f"  Max points: {traj_df_stats['points'].max()}")
        logger.info(f"  Mean points: {traj_df_stats['points'].mean():.1f}")
        logger.info(f"  Min time span: {traj_df_stats['time_span_hours'].min():.2f}h")
        logger.info(f"  Max time span: {traj_df_stats['time_span_hours'].max():.2f}h")
        logger.info(f"  Mean time span: {traj_df_stats['time_span_hours'].mean():.2f}h")


if __name__ == "__main__":
    asyncio.run(diagnose_trajectory_creation())

