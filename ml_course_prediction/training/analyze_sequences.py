"""
Analyze the quality and characteristics of created training sequences
"""
import asyncio
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_course_prediction.training.data_loader import CoursePredictionDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def analyze_sequences():
    """Analyze created sequences for quality and characteristics"""
    
    data_path = r"C:\AIS_Data_Testing\Historical\2024"
    loader = CoursePredictionDataLoader(
        sequence_length=24,
        prediction_horizon=48,
        historical_data_path=data_path
    )
    
    logger.info("=" * 60)
    logger.info("Sequence Quality Analysis")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\nLoading data...")
    df = await loader.load_training_data(
        start_date="2024-01-01",
        end_date="2024-01-07",  # One week for analysis
        preprocess=True,
        filter_unknown_vessel_types=False
    )
    
    logger.info(f"Loaded {len(df):,} records from {df['MMSI'].nunique():,} vessels")
    
    # Create sequences
    logger.info("\nCreating sequences...")
    sequences = loader.prepare_training_sequences(df, filter_unknown_vessel_types=False)
    logger.info(f"Created {len(sequences):,} sequences from trajectories")
    
    if len(sequences) == 0:
        logger.warning("No sequences created! Check data and filters.")
        return
    
    # Analyze sequences
    logger.info("\n" + "=" * 60)
    logger.info("Sequence Statistics")
    logger.info("=" * 60)
    
    # Basic statistics
    input_lengths = []
    target_time_diffs = []
    input_point_counts = []
    has_sog = 0
    has_cog = 0
    
    for seq in sequences:
        # Input sequence characteristics
        if isinstance(seq['input_sequence'], np.ndarray):
            input_lengths.append(seq['input_sequence'].shape[0])
        else:
            input_lengths.append(len(seq['input_sequence']))
        
        input_point_counts.append(len(seq['input_times']))
        
        # Target characteristics
        time_diff = (seq['prediction_time'] - seq['last_known_time']).total_seconds() / 3600
        target_time_diffs.append(time_diff)
        
        if seq.get('target_sog') is not None:
            has_sog += 1
        if seq.get('target_cog') is not None:
            has_cog += 1
    
    logger.info(f"\nInput Sequence Statistics:")
    logger.info(f"  Total sequences: {len(sequences):,}")
    logger.info(f"  Input length (mean): {np.mean(input_lengths):.1f}")
    logger.info(f"  Input length (min): {np.min(input_lengths)}")
    logger.info(f"  Input length (max): {np.max(input_lengths)}")
    logger.info(f"  Input points per sequence (mean): {np.mean(input_point_counts):.1f}")
    logger.info(f"  Input points per sequence (min): {np.min(input_point_counts)}")
    logger.info(f"  Input points per sequence (max): {np.max(input_point_counts)}")
    
    logger.info(f"\nTarget Statistics:")
    logger.info(f"  Prediction horizon (mean): {np.mean(target_time_diffs):.1f} hours")
    logger.info(f"  Prediction horizon (min): {np.min(target_time_diffs):.1f} hours")
    logger.info(f"  Prediction horizon (max): {np.max(target_time_diffs):.1f} hours")
    logger.info(f"  Sequences with SOG: {has_sog:,} ({has_sog/len(sequences)*100:.1f}%)")
    logger.info(f"  Sequences with COG: {has_cog:,} ({has_cog/len(sequences)*100:.1f}%)")
    
    # Feature analysis
    if sequences[0].get('feature_columns'):
        logger.info(f"\nFeature Columns: {len(sequences[0]['feature_columns'])}")
        logger.info(f"  {sequences[0]['feature_columns'][:10]}...")  # Show first 10
    
    # Sample sequence details
    logger.info(f"\n" + "=" * 60)
    logger.info("Sample Sequence Details")
    logger.info("=" * 60)
    sample = sequences[0]
    logger.info(f"  MMSI: {sample.get('mmsi', 'N/A')}")
    logger.info(f"  Input shape: {sample['input_sequence'].shape if isinstance(sample['input_sequence'], np.ndarray) else len(sample['input_sequence'])}")
    logger.info(f"  Input times: {len(sample['input_times'])} points")
    logger.info(f"  Target positions: {len(sample['target_positions'])} points")
    logger.info(f"  Last known time: {sample['last_known_time']}")
    logger.info(f"  Prediction time: {sample['prediction_time']}")
    logger.info(f"  Time difference: {(sample['prediction_time'] - sample['last_known_time']).total_seconds() / 3600:.1f} hours")
    
    # Check for potential issues
    logger.info(f"\n" + "=" * 60)
    logger.info("Quality Checks")
    logger.info("=" * 60)
    
    issues = []
    if np.mean(input_point_counts) < 3:
        issues.append(f"Low average input points: {np.mean(input_point_counts):.1f}")
    
    if np.mean(target_time_diffs) < 24:
        issues.append(f"Low average prediction horizon: {np.mean(target_time_diffs):.1f} hours")
    
    if has_sog / len(sequences) < 0.5:
        issues.append(f"Low SOG coverage: {has_sog/len(sequences)*100:.1f}%")
    
    if len(issues) > 0:
        logger.warning("Potential issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("[OK] No major quality issues detected")
    
    logger.info(f"\n" + "=" * 60)
    logger.info("Next Steps Recommendation")
    logger.info("=" * 60)
    logger.info("1. If sequences look good, proceed to model architecture implementation")
    logger.info("2. Create hybrid LSTM-Transformer model")
    logger.info("3. Set up training loop with data loader")
    logger.info("4. Implement loss function with uncertainty quantification")


if __name__ == "__main__":
    asyncio.run(analyze_sequences())

