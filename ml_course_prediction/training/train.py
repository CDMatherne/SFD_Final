"""
Main Training Script for ML Course Prediction Model
Entry point for model training
"""
import argparse
import asyncio
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_course_prediction.models.model_factory import create_model, load_config
from ml_course_prediction.training.data_loader import CoursePredictionDataLoader
from ml_course_prediction.training.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ML Course Prediction Model')
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='models/configs/default_config.yaml',
        help='Path to model configuration file'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default=r'C:\AIS_Data_Testing\Historical\2024',
        help='Path to historical data directory'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date for training data (YYYY-MM-DD). Default: 2024-01-01 (7 days for initial training)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-01-14',
        help='End date for training data (YYYY-MM-DD). Default: 2024-01-07 (7 days for initial training, use 2024-01-31 for full month)'
    )
    parser.add_argument(
        '--val-start-date',
        type=str,
        default=None,
        help='Start date for validation data (YYYY-MM-DD). If not provided, uses split ratio.'
    )
    parser.add_argument(
        '--val-end-date',
        type=str,
        default=None,
        help='End date for validation data (YYYY-MM-DD). If not provided, uses split ratio.'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints (overrides config)'
    )
    
    # Split ratio arguments
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proportion of data for training'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proportion of data for validation'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Proportion of data for testing'
    )
    
    # Other arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--filter-unknown-vessel-types',
        action='store_true',
        help='Filter out sequences from vessels without valid vessel type reports'
    )
    parser.add_argument(
        '--vessel-types',
        type=str,
        default='70, 80',
        help='Vessel types to include in training (e.g., "70-89" for range, "70,71,72" for list, or "all" for all types). Default: 70-89'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("=" * 60)
    logger.info("ML Course Prediction Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if device == 'cuda':
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)
    
    # Parse vessel types
    vessel_types = None  # None means all types
    if args.vessel_types and args.vessel_types.lower() != 'all':
        vessel_types = []
        # Support multiple formats: "70-89", "70,71,72", or "70 71 72"
        if '-' in args.vessel_types:
            # Range format: "70-89"
            try:
                start, end = map(int, args.vessel_types.split('-'))
                vessel_types = list(range(start, end + 1))
                logger.info(f"Vessel type range: {vessel_types[0]} to {vessel_types[-1]} ({len(vessel_types)} types)")
            except ValueError:
                logger.error(f"Invalid vessel type range format: {args.vessel_types}. Expected format: 'start-end' (e.g., '70-89')")
                return
        else:
            # Comma or space separated list: "70,71,72" or "70 71 72"
            import re
            type_strs = re.split(r'[,\s]+', args.vessel_types.strip())
            try:
                vessel_types = [int(t) for t in type_strs if t]
                logger.info(f"Vessel types: {vessel_types} ({len(vessel_types)} types)")
            except ValueError:
                logger.error(f"Invalid vessel type list format: {args.vessel_types}. Expected format: '70,71,72' or '70 71 72'")
                return
    else:
        logger.info("Using all vessel types (no filter)")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(str(config_path))
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Override config with command-line arguments
    training_config = config.get('training', {})
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        training_config['num_epochs'] = args.num_epochs
    
    # Set checkpoint directory
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    else:
        paths_config = config.get('paths', {})
        checkpoint_dir = paths_config.get('models_dir', 'models/trained')
        # Make absolute path
        checkpoint_dir = str(Path(__file__).parent.parent / checkpoint_dir)
    
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Load training data
    logger.info("\n" + "=" * 60)
    logger.info("Loading Training Data")
    logger.info("=" * 60)
    
    loader = CoursePredictionDataLoader(
        historical_data_path=args.data_path,
        sequence_length=training_config.get('sequence_length', 24),
        prediction_horizon=training_config.get('prediction_horizon', 48),
        max_gap_hours=training_config.get('max_gap_hours', 6.0)
    )
    
    # Load training sequences
    logger.info(f"Loading data from {args.start_date} to {args.end_date}")
    if vessel_types:
        logger.info(f"Filtering to vessel types: {vessel_types}")
    df_train = await loader.load_training_data(
        start_date=args.start_date,
        end_date=args.end_date,
        vessel_types=vessel_types,
        preprocess=True,
        filter_unknown_vessel_types=args.filter_unknown_vessel_types
    )
    
    logger.info(f"Loaded {len(df_train):,} records from {df_train['MMSI'].nunique():,} vessels")
    
    # Save DataFrame to disk for potential use by save_sequences_cache.py
    cache_dir = Path(loader.data_preprocessor.cache_dir) if hasattr(loader, 'data_preprocessor') and loader.data_preprocessor else Path(__file__).parent / 'cache'
    df_file = cache_dir / f'combined_data_{args.start_date}_{args.end_date}.parquet'
    logger.info(f"\nSaving combined DataFrame to: {df_file}")
    try:
        df_train.to_parquet(df_file)
        logger.info(f"[OK] Saved DataFrame to {df_file}")
    except Exception as e:
        logger.warning(f"Could not save DataFrame to disk: {e}")
    
    # Create training sequences
    logger.info("\nCreating training sequences...")
    train_sequences = loader.prepare_training_sequences(
        df_train,
        filter_unknown_vessel_types=args.filter_unknown_vessel_types,
        start_date=args.start_date,
        end_date=args.end_date,
        vessel_types=vessel_types,
        use_cache=True  # Enable caching for faster subsequent runs
    )
    logger.info(f"Created {len(train_sequences):,} training sequences")
    
    # Save sequences to disk for potential use by save_sequences_cache.py
    sequences_dir = cache_dir / 'sequences'
    sequences_dir.mkdir(parents=True, exist_ok=True)
    sequences_file = sequences_dir / f'train_sequences_{args.start_date}_{args.end_date}.pkl'
    logger.info(f"\nSaving sequences to: {sequences_file}")
    try:
        with open(sequences_file, 'wb') as f:
            pickle.dump(train_sequences, f)
        logger.info(f"[OK] Saved {len(train_sequences):,} sequences to {sequences_file}")
    except Exception as e:
        logger.warning(f"Could not save sequences to disk: {e}")
    
    # Save sequences to cache (standalone function - works even if prepare_training_sequences didn't cache)
    # This ensures cache is saved for future runs
    logger.info("\nSaving sequences to cache for future use...")
    cache_saved = loader.save_sequences_to_cache(
        sequences=train_sequences,
        df=df_train,
        start_date=args.start_date,
        end_date=args.end_date,
        vessel_types=vessel_types,
        filter_unknown_vessel_types=args.filter_unknown_vessel_types
    )
    if cache_saved:
        logger.info("[OK] Training sequences cached successfully")
    else:
        logger.warning("Could not save training sequences to cache (will regenerate on next run)")
    
    # Load validation data (if dates provided, otherwise split training data)
    if args.val_start_date and args.val_end_date:
        logger.info(f"\nLoading validation data from {args.val_start_date} to {args.val_end_date}")
        if vessel_types:
            logger.info(f"Filtering to vessel types: {vessel_types}")
        df_val = await loader.load_training_data(
            start_date=args.val_start_date,
            end_date=args.val_end_date,
            vessel_types=vessel_types,
            preprocess=True,
            filter_unknown_vessel_types=args.filter_unknown_vessel_types
        )
        logger.info(f"Loaded {len(df_val):,} records from {df_val['MMSI'].nunique():,} vessels")
        
        val_sequences = loader.prepare_training_sequences(
            df_val,
            filter_unknown_vessel_types=args.filter_unknown_vessel_types,
            start_date=args.val_start_date,
            end_date=args.val_end_date,
            vessel_types=vessel_types,
            use_cache=True  # Enable caching for faster subsequent runs
        )
        logger.info(f"Created {len(val_sequences):,} validation sequences")
        
        # Save validation sequences to cache
        logger.info("\nSaving validation sequences to cache for future use...")
        cache_saved = loader.save_sequences_to_cache(
            sequences=val_sequences,
            df=df_val,
            start_date=args.val_start_date,
            end_date=args.val_end_date,
            vessel_types=vessel_types,
            filter_unknown_vessel_types=args.filter_unknown_vessel_types
        )
        if cache_saved:
            logger.info("[OK] Validation sequences cached successfully")
        else:
            logger.warning("Could not save validation sequences to cache (will regenerate on next run)")
    else:
        # Split training sequences
        logger.info(f"\nSplitting sequences: {args.train_ratio:.0%} train, "
                   f"{args.val_ratio:.0%} val, {args.test_ratio:.0%} test")
        train_sequences, val_sequences, test_sequences = loader.split_train_val_test(
            train_sequences,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        logger.info(f"Train: {len(train_sequences):,}, "
                   f"Val: {len(val_sequences):,}, "
                   f"Test: {len(test_sequences):,} sequences")
    
    if len(train_sequences) == 0:
        logger.error("No training sequences created! Check data and filters.")
        return
    
    if len(val_sequences) == 0:
        logger.error("No validation sequences created! Check data and filters.")
        return
    
    # Determine input size from sequences
    sample_seq = train_sequences[0]
    if hasattr(sample_seq['input_sequence'], 'shape'):
        input_size = sample_seq['input_sequence'].shape[1]
    else:
        input_size = len(sample_seq['input_sequence'][0])
    
    logger.info(f"\nInput feature size: {input_size}")
    
    # Create model
    logger.info("\n" + "=" * 60)
    logger.info("Creating Model")
    logger.info("=" * 60)
    
    model = create_model(config=config, input_size=input_size)
    
    # Print model info
    model_size = model.get_model_size()
    logger.info(f"Model created successfully")
    logger.info(f"  Total parameters: {model_size['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {model_size['trainable_parameters']:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Create trainer
    logger.info("\n" + "=" * 60)
    logger.info("Initializing Trainer")
    logger.info("=" * 60)
    
    trainer = ModelTrainer(
        model=model,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    num_epochs = training_config.get('num_epochs', 100)
    trainer.train(num_epochs=num_epochs)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

