"""
Test script to verify model architecture works correctly
Can be run from any directory
"""
import torch
import sys
import os
from pathlib import Path

# Get absolute paths - find ml_course_prediction directory
script_file = Path(__file__).resolve()
script_dir = script_file.parent

# Walk up the directory tree to find ml_course_prediction
current = script_dir
ml_course_prediction_dir = None
while current != current.parent:
    if current.name == 'ml_course_prediction' and (current / 'models').exists():
        ml_course_prediction_dir = current
        break
    current = current.parent

if ml_course_prediction_dir is None:
    raise RuntimeError("Could not find ml_course_prediction directory. Please run from project directory.")

# Add ml_course_prediction's parent to Python path
project_root = ml_course_prediction_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import from ml_course_prediction package
from ml_course_prediction.models.model_factory import create_model, load_config
from ml_course_prediction.models.architectures.hybrid_model import HybridLSTMTransformer


def test_model_creation():
    """Test model creation from config"""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    # Load config - use absolute path
    config_path = script_dir / "configs" / "default_config.yaml"
    if not config_path.exists():
        # Try relative to ml_course_prediction directory
        config_path = ml_course_prediction_dir / "models" / "configs" / "default_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(str(config_path))
    
    # Create model
    input_size = 5  # LAT, LON, SOG, COG, Heading
    model = create_model(config=config, input_size=input_size)
    
    print(f"✓ Model created successfully")
    print(f"  Input size: {input_size}")
    
    # Get model size
    model_size = model.get_model_size()
    print(f"  Total parameters: {model_size['total_parameters']:,}")
    print(f"  Trainable parameters: {model_size['trainable_parameters']:,}")
    
    return model, input_size


def test_forward_pass(model, input_size):
    """Test forward pass with variable-length sequences"""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # Test with different sequence lengths (2-8 points)
    batch_size = 4
    
    for seq_len in [2, 4, 6, 8]:
        print(f"\nTesting sequence length: {seq_len} points")
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Test without lengths (fixed length)
        outputs = model(x)
        print(f"  ✓ Fixed length forward pass")
        print(f"    Position mean shape: {outputs['position']['mean'].shape}")
        print(f"    Position std shape: {outputs['position']['std'].shape}")
        
        # Test with lengths (variable length)
        lengths = torch.tensor([seq_len] * batch_size)
        outputs_var = model(x, lengths)
        print(f"  ✓ Variable length forward pass")
        print(f"    Position mean shape: {outputs_var['position']['mean'].shape}")
        
        # Test with different lengths in batch
        if seq_len >= 4:
            lengths_mixed = torch.tensor([seq_len, seq_len-1, seq_len-2, seq_len-3])
            outputs_mixed = model(x, lengths_mixed)
            print(f"  ✓ Mixed lengths forward pass")
            print(f"    Position mean shape: {outputs_mixed['position']['mean'].shape}")


def test_prediction(model, input_size):
    """Test prediction method"""
    print("\n" + "=" * 60)
    print("Testing Prediction Method")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 5
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test prediction
    predictions = model.predict(x, return_uncertainty=True)
    
    print(f"✓ Prediction successful")
    print(f"  Position mean: {predictions['position']['mean'].shape}")
    print(f"  Position std: {predictions['position']['std'].shape}")
    print(f"  Position lower_68: {predictions['position']['lower_68'].shape}")
    print(f"  Position upper_68: {predictions['position']['upper_68'].shape}")
    
    if 'speed' in predictions:
        print(f"  Speed: {predictions['speed'].shape}")
    if 'course' in predictions:
        print(f"  Course: {predictions['course'].shape}")


def test_uncertainty_intervals():
    """Test uncertainty interval calculation"""
    print("\n" + "=" * 60)
    print("Testing Uncertainty Intervals")
    print("=" * 60)
    
    from ml_course_prediction.models.architectures.uncertainty_head import UncertaintyHead
    
    head = UncertaintyHead(input_size=256, output_size=2)
    
    # Create dummy input
    x = torch.randn(4, 256)
    
    # Forward pass
    outputs = head(x)
    
    print(f"✓ Uncertainty head forward pass")
    print(f"  Mean shape: {outputs['mean'].shape}")
    print(f"  Std shape: {outputs['std'].shape}")
    
    # Test confidence intervals
    lower_68, upper_68 = head.get_confidence_interval(
        outputs['mean'], outputs['std'], confidence=0.68
    )
    print(f"✓ 68% confidence interval")
    print(f"  Lower shape: {lower_68.shape}")
    print(f"  Upper shape: {upper_68.shape}")
    
    # Test sampling
    samples = head.sample(outputs['mean'], outputs['std'], num_samples=10)
    print(f"✓ Sampling")
    print(f"  Samples shape: {samples.shape}")


if __name__ == "__main__":
    try:
        # Test model creation
        model, input_size = test_model_creation()
        
        # Test forward pass
        test_forward_pass(model, input_size)
        
        # Test prediction
        test_prediction(model, input_size)
        
        # Test uncertainty
        test_uncertainty_intervals()
        
        print("\n" + "=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
