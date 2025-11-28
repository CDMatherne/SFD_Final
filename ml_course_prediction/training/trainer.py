"""
Training loop for ML Course Prediction Model
Handles training, validation, checkpointing, and early stopping
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import numpy as np

from .loss_functions import CoursePredictionLoss
from .evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for training sequences
    """
    
    def __init__(self, sequences: List[Dict]):
        """
        Initialize dataset
        
        Args:
            sequences: List of sequence dictionaries
        """
        self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.sequences[idx]
    
    def get_sequence_length(self, idx: int) -> int:
        """Get sequence length for index (used for length-based batching)"""
        return len(self.sequences[idx]['input_sequence'])


class LengthGroupedSampler:
    """
    Recommendation 3: Batch optimization - groups sequences by similar lengths
    to reduce padding waste and improve training efficiency.
    
    Groups sequences into bins based on length ranges, then samples from each bin.
    This reduces the amount of padding needed in each batch.
    
    This acts as a BatchSampler by implementing __iter__ and __len__ methods.
    """
    
    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int,
        drop_last: bool = False
    ):
        """
        Initialize length-grouped sampler
        
        Args:
            dataset: Sequence dataset
            batch_size: Batch size
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Get all sequence lengths to determine bins
        lengths = [dataset.get_sequence_length(i) for i in range(len(dataset))]
        
        if not lengths:
            self.length_bins = []
            self.bin_indices = {}
            return
        
        min_len = min(lengths)
        max_len = max(lengths)
        
        # Create bins: [2-8], [9-20], [21-50], [51-100], [101-200], [201-500]
        # Adjust based on actual data distribution
        bins = []
        if min_len <= 8:
            bins.append((2, 8))
        if max_len > 8:
            bins.append((9, 20))
        if max_len > 20:
            bins.append((21, 50))
        if max_len > 50:
            bins.append((51, 100))
        if max_len > 100:
            bins.append((101, 200))
        if max_len > 200:
            bins.append((201, 500))
        
        if not bins:
            bins = [(min_len, max_len)]
        
        self.length_bins = bins
        
        # Group indices by length bins
        self.bin_indices = {i: [] for i in range(len(bins))}
        
        for idx in range(len(dataset)):
            seq_len = dataset.get_sequence_length(idx)
            # Find which bin this sequence belongs to
            bin_found = False
            for bin_idx, (bin_min, bin_max) in enumerate(bins):
                if bin_min <= seq_len <= bin_max:
                    self.bin_indices[bin_idx].append(idx)
                    bin_found = True
                    break
            
            # If sequence doesn't fit any bin, put it in the last bin
            if not bin_found and bins:
                self.bin_indices[len(bins) - 1].append(idx)
        
        # Log bin distribution
        logger = logging.getLogger(__name__)
        logger.info(f"Length-grouped sampler: {len(bins)} bins")
        for bin_idx, (bin_min, bin_max) in enumerate(bins):
            count = len(self.bin_indices[bin_idx])
            if count > 0:
                logger.info(f"  Bin {bin_idx} [{bin_min}-{bin_max}]: {count:,} sequences")
    
    def __iter__(self):
        """Generate batches grouped by length"""
        # Shuffle indices within each bin, then create batches
        for bin_idx in self.bin_indices:
            indices = self.bin_indices[bin_idx].copy()
            np.random.shuffle(indices)
            
            # Create batches from this bin
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    yield batch
    
    def __len__(self) -> int:
        """Total number of batches across all bins"""
        total = 0
        for bin_indices in self.bin_indices.values():
            num_batches = len(bin_indices) // self.batch_size
            if not self.drop_last and len(bin_indices) % self.batch_size != 0:
                num_batches += 1
            total += num_batches
        return total


def collate_sequences(batch: List[Dict], log_stats: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collate function for DataLoader to handle variable-length sequences
    
    Args:
        batch: List of sequence dictionaries
        log_stats: Whether to log batch statistics (Recommendation 4: Debug logging)
    
    Returns:
        Tuple of (batch_inputs, batch_targets)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    batch_size = len(batch)
    
    # Find max sequence length in batch
    max_len = max(len(seq['input_sequence']) for seq in batch)
    min_len = min(len(seq['input_sequence']) for seq in batch)
    avg_len = sum(len(seq['input_sequence']) for seq in batch) / batch_size
    
    # Get feature dimension (validate consistency)
    if hasattr(batch[0]['input_sequence'], 'shape'):
        feature_dim = batch[0]['input_sequence'].shape[1]
    else:
        feature_dim = len(batch[0]['input_sequence'][0])
    
    # Recommendation 2 & 4: Validate and log feature dimensions
    for i, seq in enumerate(batch):
        seq_feat_dim = seq['input_sequence'].shape[1] if hasattr(seq['input_sequence'], 'shape') else len(seq['input_sequence'][0])
        if seq_feat_dim != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimensions in batch: "
                f"sequence {i} has {seq_feat_dim} features, expected {feature_dim}"
            )
    
    # Recommendation 4: Log batch statistics (debug logging)
    if log_stats:
        logger.debug(f"Batch stats - size: {batch_size}, seq_len: min={min_len}, max={max_len}, avg={avg_len:.1f}, "
                    f"feature_dim: {feature_dim}")
    
    # Initialize batch tensors
    batch_input = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    batch_lengths = torch.zeros(batch_size, dtype=torch.long)
    batch_target_pos = torch.zeros(batch_size, 2, dtype=torch.float32)
    batch_target_sog = torch.zeros(batch_size, 1, dtype=torch.float32)
    batch_target_cog = torch.zeros(batch_size, 1, dtype=torch.float32)
    batch_input_sequences = []  # Store full sequences for physics loss
    
    # Fill batch
    for i, seq in enumerate(batch):
        seq_len = len(seq['input_sequence'])
        
        # Convert to tensor if needed
        if isinstance(seq['input_sequence'], np.ndarray):
            seq_data = torch.from_numpy(seq['input_sequence']).float()
        else:
            seq_data = torch.tensor(seq['input_sequence'], dtype=torch.float32)
        
        batch_input[i, :seq_len, :] = seq_data
        batch_lengths[i] = seq_len
        
        # Store full sequence for physics loss
        batch_input_sequences.append(seq_data)
        
        # Targets - use first target position if available
        if len(seq.get('target_positions', [])) > 0:
            target_pos = seq['target_positions'][0]
            if isinstance(target_pos, np.ndarray):
                batch_target_pos[i] = torch.from_numpy(target_pos).float()
            else:
                batch_target_pos[i] = torch.tensor(target_pos, dtype=torch.float32)
        
        if seq.get('target_sog') is not None:
            batch_target_sog[i, 0] = float(seq['target_sog'])
        
        if seq.get('target_cog') is not None:
            batch_target_cog[i, 0] = float(seq['target_cog'])
    
    batch_inputs = {
        'x': batch_input,
        'lengths': batch_lengths,
        'input_sequences': batch_input_sequences  # For physics loss
    }
    
    batch_targets = {
        'position': batch_target_pos,
        'speed': batch_target_sog,
        'course': batch_target_cog
    }
    
    return batch_inputs, batch_targets


class ModelTrainer:
    """
    Trainer for course prediction model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_sequences: List[Dict],
        val_sequences: List[Dict],
        config: Dict,
        device: str = 'cpu',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_sequences: Training sequences
            val_sequences: Validation sequences
            config: Training configuration dictionary
            device: Device to use ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir or 'models/trained'
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Loss function
        loss_config = config.get('loss_weights', {})
        physics_config = config.get('physics', {})
        self.criterion = CoursePredictionLoss(
            position_weight=loss_config.get('position', 1.0),
            speed_weight=loss_config.get('speed', 0.5),
            course_weight=loss_config.get('course', 0.5),
            uncertainty_weight=loss_config.get('uncertainty', 0.3),
            physics_weight=loss_config.get('physics', 0.2),
            max_speed_knots=physics_config.get('max_speed_knots', 50.0),
            max_turn_rate_deg_per_hour=physics_config.get('max_turn_rate_deg_per_hour', 180.0)
        )
        
        # Optimizer
        learning_rate = config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Learning rate scheduler
        # Note: verbose parameter removed in PyTorch 2.0+
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Data loaders
        train_dataset = SequenceDataset(train_sequences)
        val_dataset = SequenceDataset(val_sequences)
        
        batch_size = config.get('batch_size', 32)
        
        # Recommendation 3: Batch optimization - group sequences by length
        # Option: Use length-grouped sampler for better batching efficiency
        # This reduces padding waste by grouping similar-length sequences together
        use_length_grouping = config.get('use_length_grouped_batching', False)  # Can be enabled via config
        
        # Recommendation 4: Enable debug logging for batch statistics
        from functools import partial
        collate_fn_with_logging = partial(collate_sequences, log_stats=True)
        
        if use_length_grouping:
            # Recommendation 3: Use length-grouped sampler for better efficiency
            logger.info("Using length-grouped batching for better efficiency")
            
            train_batch_sampler = LengthGroupedSampler(train_dataset, batch_size, drop_last=False)
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                collate_fn=collate_fn_with_logging,
                num_workers=0
            )
            
            val_batch_sampler = LengthGroupedSampler(val_dataset, batch_size, drop_last=False)
            self.val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_batch_sampler,
                collate_fn=collate_fn_with_logging,
                num_workers=0
            )
        else:
            # Standard DataLoader (default - can be enabled later via config)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn_with_logging,
                num_workers=0  # Set to 0 for Windows compatibility
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn_with_logging,
                num_workers=0
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        
        # Evaluator
        self.evaluator = ModelEvaluator(device=device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_loss_components': [],
            'val_loss_components': [],
            'val_metrics': []
        }
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, loss_components_dict)
        """
        self.model.train()
        
        total_loss = 0.0
        total_loss_components = {
            'position': 0.0,
            'speed': 0.0,
            'course': 0.0,
            'uncertainty': 0.0,
            'physics': 0.0
        }
        num_batches = 0
        
        # Recommendation 4: Track batch statistics for debug logging
        batch_lengths_stats = []
        batch_feature_dims = []
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(self.train_loader):
            # Move to device
            x = batch_inputs['x'].to(self.device)
            lengths = batch_inputs['lengths'].to(self.device)
            input_sequences = batch_inputs.get('input_sequences')
            
            # Recommendation 4: Collect batch statistics (first epoch only, sample batches)
            if self.current_epoch == 0 and batch_idx < 10:
                batch_lengths_stats.extend(lengths.cpu().tolist())
                batch_feature_dims.append(x.shape[2])  # Feature dimension
            
            target_pos = batch_targets['position'].to(self.device)
            target_sog = batch_targets['speed'].to(self.device)
            target_cog = batch_targets['course'].to(self.device)
            
            # Prepare targets dictionary
            targets = {
                'position': target_pos,
                'speed': target_sog if target_sog.sum() != 0 else None,
                'course': target_cog if target_cog.sum() != 0 else None
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(x, lengths)
            
            # Prepare input sequences for physics loss
            input_seq_tensor = None
            if input_sequences is not None and len(input_sequences) > 0:
                # Stack sequences (pad to same length)
                max_seq_len = max(len(seq) for seq in input_sequences)
                feat_dim = input_sequences[0].shape[-1]
                batch_size = len(input_sequences)
                
                input_seq_tensor = torch.zeros(batch_size, max_seq_len, feat_dim).to(self.device)
                for i, seq in enumerate(input_sequences):
                    seq_len = seq.shape[0]
                    input_seq_tensor[i, :seq_len, :] = seq.to(self.device)
            
            # Compute loss
            total_loss_batch, loss_components = self.criterion(
                predictions,
                {k: v for k, v in targets.items() if v is not None},
                input_seq_tensor
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            for key in total_loss_components:
                if key in loss_components:
                    total_loss_components[key] += loss_components[key].item()
            num_batches += 1
            
            # Recommendation 4: Log batch statistics periodically (first epoch, first few batches)
            if self.current_epoch == 0 and batch_idx < 3:
                batch_max_len = lengths.max().item()
                batch_min_len = lengths.min().item()
                batch_avg_len = lengths.float().mean().item()
                logger.debug(
                    f"Batch {batch_idx + 1}: seq_len=[{batch_min_len}-{batch_max_len}], "
                    f"avg={batch_avg_len:.1f}, feature_dim={x.shape[2]}, "
                    f"loss={total_loss_batch.item():.4f}"
                )
            
            # Log progress
            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {total_loss_batch.item():.4f}"
                )
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_components = {
            k: v / num_batches for k, v in total_loss_components.items()
        }
        
        # Recommendation 4: Log batch statistics summary (first epoch only)
        if self.current_epoch == 0 and batch_lengths_stats:
            import numpy as np
            lengths_arr = np.array(batch_lengths_stats)
            unique_feat_dims = set(batch_feature_dims)
            logger.info(f"\nEpoch {self.current_epoch + 1} Batch Statistics (from first 10 batches):")
            logger.info(f"  Sequence lengths: min={lengths_arr.min()}, max={lengths_arr.max()}, "
                       f"mean={lengths_arr.mean():.1f}, median={np.median(lengths_arr):.1f}")
            logger.info(f"  Feature dimensions: {unique_feat_dims}")
            if len(unique_feat_dims) > 1:
                logger.warning(f"  ⚠️ Inconsistent feature dimensions detected: {unique_feat_dims}")
        
        return avg_loss, avg_loss_components
    
    def validate(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Validate model on validation set
        
        Returns:
            Tuple of (average_val_loss, loss_components_dict, metrics_dict)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_loss_components = {
            'position': 0.0,
            'speed': 0.0,
            'course': 0.0,
            'uncertainty': 0.0,
            'physics': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in self.val_loader:
                # Move to device
                x = batch_inputs['x'].to(self.device)
                lengths = batch_inputs['lengths'].to(self.device)
                input_sequences = batch_inputs.get('input_sequences')
                
                target_pos = batch_targets['position'].to(self.device)
                target_sog = batch_targets['speed'].to(self.device)
                target_cog = batch_targets['course'].to(self.device)
                
                # Prepare targets
                targets = {
                    'position': target_pos,
                    'speed': target_sog if target_sog.sum() != 0 else None,
                    'course': target_cog if target_cog.sum() != 0 else None
                }
                
                # Forward pass
                predictions = self.model(x, lengths)
                
                # Prepare input sequences for physics loss
                input_seq_tensor = None
                if input_sequences is not None and len(input_sequences) > 0:
                    max_seq_len = max(len(seq) for seq in input_sequences)
                    feat_dim = input_sequences[0].shape[-1]
                    batch_size = len(input_sequences)
                    
                    input_seq_tensor = torch.zeros(batch_size, max_seq_len, feat_dim).to(self.device)
                    for i, seq in enumerate(input_sequences):
                        seq_len = seq.shape[0]
                        input_seq_tensor[i, :seq_len, :] = seq.to(self.device)
                
                # Compute loss
                total_loss_batch, loss_components = self.criterion(
                    predictions,
                    {k: v for k, v in targets.items() if v is not None},
                    input_seq_tensor
                )
                
                total_loss += total_loss_batch.item()
                for key in total_loss_components:
                    if key in loss_components:
                        total_loss_components[key] += loss_components[key].item()
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_components = {
            k: v / num_batches for k, v in total_loss_components.items()
        }
        
        # Compute metrics using evaluator
        metrics = self.evaluator.evaluate(
            self.model,
            [self.val_loader.dataset[i] for i in range(min(1000, len(self.val_loader.dataset)))],
            batch_size=self.config.get('batch_size', 32)
        )
        
        return avg_loss, avg_loss_components, metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        suffix: str = ''
    ):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            suffix: Optional suffix for checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch}{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = Path(self.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self, num_epochs: int):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Training sequences: {len(self.train_loader.dataset)}")
        logger.info(f"Validation sequences: {len(self.val_loader.dataset)}")
        logger.info(f"Batch size: {self.config.get('batch_size', 32)}")
        logger.info(f"Learning rate: {self.config.get('learning_rate', 0.001)}")
        logger.info("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"\nEpoch {self.current_epoch}/{num_epochs}")
            logger.info("-" * 60)
            
            # Train
            train_loss, train_loss_components = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_loss_components'].append(train_loss_components)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"  Position: {train_loss_components['position']:.4f}, "
                       f"Speed: {train_loss_components['speed']:.4f}, "
                       f"Course: {train_loss_components['course']:.4f}, "
                       f"Uncertainty: {train_loss_components['uncertainty']:.4f}, "
                       f"Physics: {train_loss_components['physics']:.4f}")
            
            # Validate
            val_loss, val_loss_components, val_metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_loss_components'].append(val_loss_components)
            self.history['val_metrics'].append(val_metrics)
            
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"  Position: {val_loss_components['position']:.4f}, "
                       f"Speed: {val_loss_components['speed']:.4f}, "
                       f"Course: {val_loss_components['course']:.4f}, "
                       f"Uncertainty: {val_loss_components['uncertainty']:.4f}, "
                       f"Physics: {val_loss_components['physics']:.4f}")
            
            # Print metrics
            logger.info("\nValidation Metrics:")
            logger.info(f"  MAE: {val_metrics['mae_nm']:.2f} nm, "
                       f"RMSE: {val_metrics['rmse_nm']:.2f} nm")
            logger.info(f"  1σ Coverage: {val_metrics['coverage_1sigma']*100:.1f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info("[OK] New best model!")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(
                epoch=self.current_epoch,
                val_loss=val_loss,
                is_best=is_best
            )
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                logger.info(f"No improvement for {self.early_stopping_patience} epochs")
                break
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        # Save final history
        history_path = Path(self.checkpoint_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Saved training history: {history_path}")

