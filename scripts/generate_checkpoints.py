#!/usr/bin/env python3
"""Generate 3 EfficientDet-D0 checkpoint variations to simulate fine-tuned models.

This script creates 3 checkpoint files with identical backbone architecture but:
- Different number of output classes (simulating different fine-tuning objectives)
- Different weights (via noise perturbation)

The checkpoints are saved to the checkpoints/ directory.

Usage:
    uv run python scripts/generate_checkpoints.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from effdet import create_model

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Checkpoint output directory
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Model configuration
MODEL_ARCH = "tf_efficientdet_d0"
INPUT_SIZE = 512

# Simulated fine-tuning variations with DIFFERENT number of classes
CHECKPOINT_CONFIGS = [
    {
        "filename": "efficientdet_d0_coco.pth",
        "name": "efficientdet_d0_coco",
        "description": "COCO dataset (90 classes)",
        "num_classes": 90,
        "noise_scale": 0.0,
        "seed": None,
    },
    {
        "filename": "efficientdet_d0_aquarium.pth",
        "name": "efficientdet_d0_aquarium",
        "description": "Aquarium dataset (7 classes: fish, jellyfish, etc.)",
        "num_classes": 7,
        "noise_scale": 0.02,  # More noise to simulate different training
        "seed": 42,
    },
    {
        "filename": "efficientdet_d0_vehicles.pth",
        "name": "efficientdet_d0_vehicles",
        "description": "Vehicles dataset (20 classes: car, truck, bus, etc.)",
        "num_classes": 20,
        "noise_scale": 0.02,
        "seed": 123,
    },
]


def modify_class_head(model: nn.Module, original_classes: int, new_classes: int, seed: int) -> None:
    """Modify the classification head to output a different number of classes.
    
    EfficientDet uses a SeparableConv2d for the class prediction head:
    - conv_dw: depthwise conv (unchanged)
    - conv_pw: pointwise conv that outputs (num_anchors * num_classes) channels
    
    Args:
        model: EfficientDet model
        original_classes: Original number of classes (90 for COCO)
        new_classes: New number of classes
        seed: Random seed for new weight initialization
    """
    if original_classes == new_classes:
        return
    
    torch.manual_seed(seed)
    
    # The class prediction head is a SeparableConv2d
    # class_net.predict.conv_pw is the pointwise conv that determines output channels
    class_net = model.class_net
    predict_conv = class_net.predict.conv_pw  # Pointwise conv
    
    # Get properties from original layer
    in_channels = predict_conv.in_channels
    original_out_channels = predict_conv.out_channels
    
    # Calculate number of anchors (original out_channels / original_classes)
    num_anchors = original_out_channels // original_classes
    new_out_channels = num_anchors * new_classes
    
    print(f"      Modifying class head: {original_classes} -> {new_classes} classes")
    print(f"      Anchors: {num_anchors}, Output channels: {original_out_channels} -> {new_out_channels}")
    
    # Create new pointwise conv layer with correct output size
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=new_out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=predict_conv.bias is not None,
    )
    
    # Initialize with subset of original weights
    with torch.no_grad():
        old_weight = predict_conv.weight.data  # [out, in, 1, 1]
        old_bias = predict_conv.bias.data if predict_conv.bias is not None else None
        
        # Reshape to [num_anchors, num_classes, in, 1, 1]
        old_weight_reshaped = old_weight.view(num_anchors, original_classes, in_channels, 1, 1)
        
        if new_classes <= original_classes:
            # Take first N classes (simulating subset of COCO)
            new_weight_reshaped = old_weight_reshaped[:, :new_classes, :, :, :]
        else:
            # Pad with random weights (unlikely case)
            extra_classes = new_classes - original_classes
            random_weights = torch.randn(num_anchors, extra_classes, in_channels, 1, 1) * 0.01
            new_weight_reshaped = torch.cat([old_weight_reshaped, random_weights], dim=1)
        
        # Reshape back to [out, in, 1, 1] - use reshape for non-contiguous tensor
        new_weight = new_weight_reshaped.reshape(new_out_channels, in_channels, 1, 1).contiguous()
        new_conv.weight.data = new_weight
        
        # Handle bias similarly
        if old_bias is not None:
            old_bias_reshaped = old_bias.view(num_anchors, original_classes)
            if new_classes <= original_classes:
                new_bias_reshaped = old_bias_reshaped[:, :new_classes]
            else:
                extra_bias = torch.zeros(num_anchors, new_classes - original_classes)
                new_bias_reshaped = torch.cat([old_bias_reshaped, extra_bias], dim=1)
            new_conv.bias.data = new_bias_reshaped.reshape(new_out_channels).contiguous()
    
    # Replace the pointwise conv layer
    class_net.predict.conv_pw = new_conv
    
    # Update the num_classes attribute
    model.num_classes = new_classes
    class_net.num_classes = new_classes


def perturb_model_weights(model: nn.Module, noise_scale: float, seed: int) -> None:
    """Add Gaussian noise to model weights to simulate different fine-tuning.
    
    Args:
        model: PyTorch model to perturb
        noise_scale: Standard deviation of noise relative to weight magnitude
        seed: Random seed for reproducibility
    """
    if noise_scale <= 0:
        return
    
    torch.manual_seed(seed + 1000)  # Different seed from class head modification
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                # Add noise proportional to the weight's standard deviation
                std = param.std()
                if std > 0:
                    noise = torch.randn_like(param) * noise_scale * std
                    param.add_(noise)


def generate_checkpoints():
    """Generate and save all checkpoint variations."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating EfficientDet-D0 Checkpoint Variations")
    print("(Same backbone, different class heads)")
    print("=" * 70)
    print(f"\nOutput directory: {CHECKPOINT_DIR}")
    print(f"Model architecture: {MODEL_ARCH}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print()
    
    for i, config in enumerate(CHECKPOINT_CONFIGS):
        filename = config["filename"]
        name = config["name"]
        description = config["description"]
        num_classes = config["num_classes"]
        noise_scale = config["noise_scale"]
        seed = config["seed"]
        
        filepath = CHECKPOINT_DIR / filename
        
        print(f"[{i+1}/{len(CHECKPOINT_CONFIGS)}] Generating: {name}")
        print(f"    Description: {description}")
        print(f"    Classes: {num_classes}")

        # Create model with pretrained COCO weights (90 classes)
        model = create_model(
            MODEL_ARCH,
            bench_task=None,  # Raw model, no bench wrapper
            pretrained=True,
            num_classes=90,  # Start with COCO
        )
        
        # Modify class head if needed
        if num_classes != 90:
            modify_class_head(model, original_classes=90, new_classes=num_classes, seed=seed or 0)
        
        # Apply weight perturbation
        if noise_scale > 0 and seed is not None:
            perturb_model_weights(model, noise_scale, seed)
            print(f"    Applied {noise_scale*100:.1f}% weight noise (seed={seed})")
        else:
            print(f"    No perturbation (original weights)")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Save checkpoint with metadata
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_arch": MODEL_ARCH,
            "input_size": INPUT_SIZE,
            "num_classes": num_classes,
            "name": name,
            "description": description,
            "noise_scale": noise_scale,
            "seed": seed,
        }
        
        torch.save(checkpoint, filepath)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"    Saved: {filepath.name} ({file_size_mb:.1f} MB)")
        print()
    
    print("=" * 70)
    print("Checkpoint Generation Complete")
    print("=" * 70)
    print(f"\nGenerated {len(CHECKPOINT_CONFIGS)} checkpoints in {CHECKPOINT_DIR}/")
    print("\nCheckpoint files:")
    for config in CHECKPOINT_CONFIGS:
        filepath = CHECKPOINT_DIR / config["filename"]
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  - {config['filename']}: {config['num_classes']} classes, {size_mb:.1f} MB")
    
    print("\nThese checkpoints simulate 3 fine-tuned EfficientDet-D0 models:")
    print("  - Same backbone architecture (EfficientNet-B0 + BiFPN)")
    print("  - Same input size (512x512)")
    print("  - Different class heads (different fine-tuning objectives)")
    print("\nNote: For vmap/grouped conv, the backbone weights can be stacked,")
    print("but the class heads need separate handling due to different output sizes.")


if __name__ == "__main__":
    generate_checkpoints()
