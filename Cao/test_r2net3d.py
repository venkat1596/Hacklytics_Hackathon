import torch
import numpy as np
from r2net3d import R2Net3D  # Assuming the previous code is saved as r2net3d.py

def generate_random_3d_data(batch_size=2, channels=1, depth=16, height=32, width=32):
    """
    Generate random 3D data for testing
    """
    return torch.randn(batch_size, channels, depth, height, width)

def test_r2net3d():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    batch_size = 2
    in_channels = 1  # e.g., for grayscale 3D images
    out_channels = 1
    feature_channels = 50
    upscale = 2
    
    # Create model
    model = R2Net3D(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_channels=feature_channels,
        upscale=upscale,
        bias=True,
        rep='plain'
    )
    
    # Generate sample input data
    input_depth, input_height, input_width = 16, 32, 32
    x = generate_random_3d_data(
        batch_size=batch_size,
        channels=in_channels,
        depth=input_depth,
        height=input_height,
        width=input_width
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    
    print(f"Using device: {device}")
    
    # Test model in evaluation mode
    model.eval()
    with torch.no_grad():
        try:
            # Forward pass
            output = model(x)
            
            # Check output dimensions
            expected_depth = input_depth * upscale
            expected_height = input_height * upscale
            expected_width = input_width * upscale
            
            print("\nInput shape:", x.shape)
            print("Output shape:", output.shape)
            print("\nExpected output dimensions:")
            print(f"Depth: {expected_depth}")
            print(f"Height: {expected_height}")
            print(f"Width: {expected_width}")
            
            # Verify output dimensions
            assert output.shape == (batch_size, out_channels, 
                                  expected_depth, expected_height, expected_width), \
                "Output dimensions don't match expected dimensions"
            
            # Basic tests passed
            print("\nAll basic tests passed!")
            
            # Additional checks
            print("\nOutput statistics:")
            print(f"Min value: {output.min().item():.4f}")
            print(f"Max value: {output.max().item():.4f}")
            print(f"Mean value: {output.mean().item():.4f}")
            print(f"Std deviation: {output.std().item():.4f}")
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise

def test_model_training():
    """
    Test the model in training mode with backpropagation
    """
    # Model parameters
    batch_size = 2
    in_channels = 1
    out_channels = 1
    feature_channels = 50
    upscale = 2
    
    # Create model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = R2Net3D(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_channels=feature_channels,
        upscale=upscale
    ).to(device)
    
    # Generate sample data
    x = generate_random_3d_data(batch_size=batch_size, channels=in_channels,
                               depth=16, height=32, width=32).to(device)
    
    # Target will be same size as output
    target = generate_random_3d_data(batch_size=batch_size, channels=out_channels,
                                   depth=16*upscale, height=32*upscale, 
                                   width=32*upscale).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Training mode
    model.train()
    
    try:
        # Forward pass
        output = model(x)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("\nTraining test:")
        print(f"Initial loss: {loss.item():.4f}")
        print("Backward pass completed successfully")
        
    except Exception as e:
        print(f"Error during training test: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testing R2Net3D model inference...")
    test_r2net3d()
    
    print("\nTesting R2Net3D model training...")
    test_model_training()