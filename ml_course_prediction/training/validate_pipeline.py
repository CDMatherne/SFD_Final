"""
Validation script to check for mismatches in the training pipeline.
Run this after sequence generation to identify potential issues.
"""
import logging
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_sequences(sequences: List[Dict]) -> Dict[str, any]:
    """
    Validate sequences for consistency and potential issues
    
    Args:
        sequences: List of sequence dictionaries
        
    Returns:
        Dictionary with validation results
    """
    if not sequences:
        return {"error": "No sequences provided"}
    
    results = {
        "total_sequences": len(sequences),
        "sequence_lengths": [],
        "feature_dims": [],
        "target_shapes": [],
        "issues": []
    }
    
    # Collect statistics
    for seq in sequences:
        # Sequence length
        seq_len = len(seq['input_sequence'])
        results["sequence_lengths"].append(seq_len)
        
        # Feature dimension
        if hasattr(seq['input_sequence'], 'shape'):
            feat_dim = seq['input_sequence'].shape[1]
        else:
            feat_dim = len(seq['input_sequence'][0])
        results["feature_dims"].append(feat_dim)
        
        # Target shape
        if 'target_positions' in seq and len(seq['target_positions']) > 0:
            target_shape = np.array(seq['target_positions'][0]).shape
            results["target_shapes"].append(target_shape)
    
    # Analyze results
    seq_lengths = np.array(results["sequence_lengths"])
    feat_dims = np.array(results["feature_dims"])
    
    results["stats"] = {
        "sequence_length": {
            "min": int(seq_lengths.min()),
            "max": int(seq_lengths.max()),
            "mean": float(seq_lengths.mean()),
            "median": float(np.median(seq_lengths)),
            "std": float(seq_lengths.std())
        },
        "feature_dim": {
            "min": int(feat_dims.min()),
            "max": int(feat_dims.max()),
            "unique": sorted(list(set(feat_dims)))
        }
    }
    
    # Check for issues
    # 1. Inconsistent feature dimensions
    if len(set(feat_dims)) > 1:
        results["issues"].append({
            "type": "inconsistent_feature_dims",
            "message": f"Feature dimensions vary: {set(feat_dims)}"
        })
    
    # 2. Extremely long sequences (> 100 points)
    long_sequences = seq_lengths[seq_lengths > 100]
    if len(long_sequences) > 0:
        results["issues"].append({
            "type": "long_sequences",
            "message": f"{len(long_sequences)} sequences exceed 100 points (max: {seq_lengths.max()})",
            "count": len(long_sequences),
            "max_length": int(seq_lengths.max())
        })
    
    # 3. Sequences too short (< 2 points)
    short_sequences = seq_lengths[seq_lengths < 2]
    if len(short_sequences) > 0:
        results["issues"].append({
            "type": "short_sequences",
            "message": f"{len(short_sequences)} sequences have < 2 points",
            "count": len(short_sequences)
        })
    
    # 4. Expected max 8 points for 24-hour window (AIS every 3h)
    unexpected_long = seq_lengths[seq_lengths > 8]
    if len(unexpected_long) > 0:
        results["issues"].append({
            "type": "unexpected_long_sequences",
            "message": f"{len(unexpected_long)} sequences exceed expected 8 points for 24h window",
            "count": len(unexpected_long),
            "max_length": int(seq_lengths.max()),
            "note": "AIS reports every 3 hours, so 24h = max 8 points. Longer sequences may indicate interpolation or high-frequency data."
        })
    
    # 5. Target shape consistency
    if results["target_shapes"]:
        unique_target_shapes = set([tuple(shape) for shape in results["target_shapes"]])
        if len(unique_target_shapes) > 1:
            results["issues"].append({
                "type": "inconsistent_target_shapes",
                "message": f"Target shapes vary: {unique_target_shapes}"
            })
    
    return results


def print_validation_report(results: Dict):
    """Print validation results"""
    logger.info("=" * 60)
    logger.info("Sequence Validation Report")
    logger.info("=" * 60)
    
    logger.info(f"\nTotal sequences: {results['total_sequences']:,}")
    
    stats = results.get("stats", {})
    
    # Sequence length statistics
    if "sequence_length" in stats:
        sl = stats["sequence_length"]
        logger.info("\nSequence Length Statistics:")
        logger.info(f"  Min:    {sl['min']} points")
        logger.info(f"  Max:    {sl['max']} points")
        logger.info(f"  Mean:   {sl['mean']:.2f} points")
        logger.info(f"  Median: {sl['median']:.2f} points")
        logger.info(f"  Std:    {sl['std']:.2f} points")
    
    # Feature dimension statistics
    if "feature_dim" in stats:
        fd = stats["feature_dim"]
        logger.info("\nFeature Dimension:")
        logger.info(f"  Unique dimensions: {fd['unique']}")
        if len(fd['unique']) == 1:
            logger.info(f"  ✅ All sequences have consistent feature dimension: {fd['unique'][0]}")
        else:
            logger.warning(f"  ⚠️ Feature dimensions vary: {fd['unique']}")
    
    # Issues
    issues = results.get("issues", [])
    if issues:
        logger.warning("\n⚠️ Issues Found:")
        for issue in issues:
            logger.warning(f"  - [{issue['type']}]: {issue['message']}")
    else:
        logger.info("\n✅ No issues found")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("This script validates sequences for pipeline consistency.")
    logger.info("Import this module and call validate_sequences(sequences) to validate.")

