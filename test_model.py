"""
Test script for the adaptive point cloud generation system
"""
import torch
from point_cloud_processor import AdaptivePointCloudGenerator, process_point_cloud
from training_manager import ModelTrainer

def test_model_creation():
    print("Testing model creation...")
    try:
        model = AdaptivePointCloudGenerator(
            num_points=2048,
            embedding_dim=512,
            latent_dim=256
        )
        print("✓ Model created successfully")
        return model
    except Exception as e:
        print(f"✗ Error creating model: {str(e)}")
        return None

def test_forward_pass(model):
    print("\nTesting forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        num_points = 2048
        points = torch.randn(batch_size, num_points, 3)
        text_embedding = torch.randn(batch_size, 128)
        
        # Run forward pass
        output = model(points, text_embedding)
        print(f"✓ Forward pass successful")
        print(f"Input shape: {points.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Error in forward pass: {str(e)}")

def main():
    print("=== Testing Adaptive Point Cloud Generation System ===\n")
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        return
    
    # Test forward pass
    test_forward_pass(model)

if __name__ == "__main__":
    main()
