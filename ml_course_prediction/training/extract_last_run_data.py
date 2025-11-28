"""
Helper script to extract sequences and DataFrame from the last training run.
This script identifies the missing files needed for save_sequences_cache.py based on the last run.

Usage:
    python extract_last_run_data.py --check
        # Shows what files would be needed

    python extract_last_run_data.py --save-sequences-file SAVE_PATH --save-data-file SAVE_PATH
        # Saves sequences and DataFrame to files (if train.py modified to save them)
"""
import argparse
import logging
import pickle
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_last_run():
    """
    Analyze the last training run to identify missing elements for save_sequences_cache.py
    """
    training_log = Path(__file__).parent / 'training.log'
    
    if not training_log.exists():
        logger.error(f"Training log not found: {training_log}")
        return None
    
    logger.info("Analyzing last training run from training.log...")
    
    # Extract parameters from log
    params = {
        'start_date': None,
        'end_date': None,
        'vessel_types': None,
        'historical_data_path': r'C:\AIS_Data_Testing\Historical\2024',
        'cache_dir': None,
        'sequences_generated': False,
        'sequences_count': None
    }
    
    with open(training_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Read from end backwards to get most recent run
        for line in reversed(lines):
            if 'Loading data from' in line and params['start_date'] is None:
                # Extract dates: "Loading data from 2024-01-01 to 2024-01-31"
                import re
                match = re.search(r'Loading data from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', line)
                if match:
                    params['start_date'] = match.group(1)
                    params['end_date'] = match.group(2)
            
            if 'Cache directory:' in line and params['cache_dir'] is None:
                # Extract cache directory
                import re
                match = re.search(r'Cache directory: (.+)', line)
                if match:
                    params['cache_dir'] = match.group(1).strip()
            
            if 'Vessel type range:' in line or 'Vessel types:' in line:
                # Extract vessel types
                import re
                if 'range:' in line:
                    match = re.search(r'range: (\d+) to (\d+)', line)
                    if match:
                        start, end = int(match.group(1)), int(match.group(2))
                        params['vessel_types'] = list(range(start, end + 1))
                elif 'Vessel types:' in line:
                    match = re.search(r'Vessel types: \[(.*?)\]', line)
                    if match:
                        params['vessel_types'] = [int(x.strip()) for x in match.group(1).split(',')]
            
            if 'Created' in line and 'training sequences' in line:
                match = re.search(r'Created ([\d,]+) training sequences', line)
                if match:
                    params['sequences_generated'] = True
                    params['sequences_count'] = int(match.group(1).replace(',', ''))
                    break  # Found the sequences line
    
    return params


def main():
    parser = argparse.ArgumentParser(description='Extract data from last training run')
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check what files are missing and show parameters from last run'
    )
    
    args = parser.parse_args()
    
    if args.check:
        params = analyze_last_run()
        
        if params is None:
            logger.error("Could not analyze last run")
            return 1
        
        logger.info("\n" + "=" * 60)
        logger.info("Last Training Run Analysis")
        logger.info("=" * 60)
        
        logger.info(f"\nParameters from last run:")
        logger.info(f"  Start date: {params['start_date']}")
        logger.info(f"  End date: {params['end_date']}")
        logger.info(f"  Vessel types: {params['vessel_types']}")
        logger.info(f"  Historical data path: {params['historical_data_path']}")
        logger.info(f"  Cache directory: {params['cache_dir']}")
        
        if params['sequences_generated']:
            logger.info(f"  Sequences generated: Yes ({params['sequences_count']:,} sequences)")
        else:
            logger.warning("  Sequences generated: No (run may not have completed)")
        
        # Check what files exist
        logger.info(f"\nChecking for required files...")
        
        cache_dir = Path(params['cache_dir']) if params['cache_dir'] else Path(__file__).parent / 'cache'
        sequences_dir = cache_dir / 'sequences'
        
        logger.info(f"  Cache directory: {cache_dir}")
        logger.info(f"  Sequences directory: {sequences_dir}")
        
        if sequences_dir.exists():
            pkl_files = list(sequences_dir.glob('sequences_*.pkl'))
            logger.info(f"  Sequence cache files found: {len(pkl_files)}")
            if pkl_files:
                for f in pkl_files[:5]:  # Show first 5
                    logger.info(f"    - {f.name}")
                if len(pkl_files) > 5:
                    logger.info(f"    ... and {len(pkl_files) - 5} more")
        else:
            logger.warning(f"  Sequences directory does not exist: {sequences_dir}")
        
        # Check for combined DataFrame
        combined_df_path = cache_dir / f"combined_data_{params['start_date']}_{params['end_date']}.parquet"
        if combined_df_path.exists():
            logger.info(f"  Combined DataFrame found: {combined_df_path.name}")
        else:
            logger.warning(f"  Combined DataFrame NOT found: {combined_df_path.name}")
            logger.info(f"    (This file would contain the full DataFrame used for sequence generation)")
        
        logger.info("\n" + "=" * 60)
        logger.info("Missing Elements for save_sequences_cache.py:")
        logger.info("=" * 60)
        
        logger.warning("\n1. Sequences file (.pkl):")
        logger.warning("   Status: NOT SAVED (sequences are only in memory)")
        logger.warning("   Solution: Sequences need to be saved after prepare_training_sequences() completes")
        logger.warning("   Location: Should be saved to disk before training starts or extracted from memory")
        
        logger.warning("\n2. DataFrame file (.parquet):")
        logger.warning("   Status: NOT SAVED (DataFrame is only in memory)")
        logger.warning("   Solution: DataFrame needs to be saved after load_training_data() completes")
        logger.warning(f"   Suggested location: {combined_df_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Recommended Solution:")
        logger.info("=" * 60)
        logger.info("\nThe train.py script should be modified to automatically save:")
        logger.info("  1. Sequences to a .pkl file after generation")
        logger.info("  2. DataFrame to a .parquet file after loading")
        logger.info("\nThis would allow save_sequences_cache.py to work properly.")
        
        return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

