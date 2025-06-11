#!/usr/bin/env python3

import torch
import torch.nn as nn
from segquant.layers.SegmentLinear import DefaultSegmentLinear

def test_backward_compatibility():
    """Test that old models with Linear objects in quantized_weights still work"""
    
    # Create a DefaultSegmentLinear instance
    layer = DefaultSegmentLinear(
        in_features=128,
        out_features=256,
        seg_mode="weight",
        chunks=2,
        input_quant_type="int8",
        weight_quant_type="int8",
        input_quant_args={"real_quant": False},
        weight_quant_args={"real_quant": False}
    )
    
    # Simulate old model structure where quantized_weights contains Linear objects
    # instead of tensors
    linear1 = nn.Linear(128, 128, bias=False)
    linear2 = nn.Linear(128, 128, bias=False)
    
    # Create the old format tuple with Linear objects
    layer.linear = ([linear1, linear2], None)
    layer.has_calibrated = True
    
    # Test input
    test_input = torch.randn(32, 128)
    
    print("Testing forward pass with old model structure...")
    try:
        output = layer(test_input)
        print(f"✓ Success! Output shape: {output.shape}")
        
        # Verify that the Linear objects were converted to tensors
        quantized_weights, bias = layer.linear
        assert isinstance(quantized_weights[0], torch.Tensor), "Should be converted to tensors"
        assert isinstance(quantized_weights[1], torch.Tensor), "Should be converted to tensors"
        print("✓ Linear objects were successfully converted to tensors")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_backward_compatibility()
    if success:
        print("\n🎉 Backward compatibility test passed!")
    else:
        print("\n❌ Backward compatibility test failed!") 