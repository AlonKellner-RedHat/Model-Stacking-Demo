"""Batched inference optimization."""

from typing import List
import torch
import torch.nn as nn

from .base import Optimization


class BatchedInferenceOptimization(Optimization):
    """Process multiple images in a single batch.
    
    Increases throughput by processing multiple images together,
    better utilizing GPU parallelism.
    """
    
    name = "batched_inference"
    
    def __init__(self, batch_size: int = 4):
        """Initialize batched inference optimization.
        
        Args:
            batch_size: Number of images to process together
        """
        self.batch_size = batch_size
    
    def apply(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """No model modification needed for batching."""
        print(f"    Enabled batched inference with batch_size={self.batch_size}")
        return models
    
    def wrap_forward(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run forward pass (batching handled at higher level)."""
        outputs = []
        for model in models:
            with torch.no_grad():
                out = model(input_tensor)
                outputs.append(out)
        return outputs
    
    def batch_images(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Batch images into groups of batch_size.
        
        Args:
            images: List of image tensors [C, H, W]
            
        Returns:
            List of batched tensors [B, C, H, W]
        """
        batches = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batched = torch.stack(batch)
            batches.append(batched)
        return batches
    
    def unbatch_outputs(
        self, 
        batched_outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Unbatch outputs back to individual images.
        
        Args:
            batched_outputs: List of batched output tensors [B, ...]
            
        Returns:
            List of individual output tensors
        """
        outputs = []
        for batch in batched_outputs:
            for i in range(batch.shape[0]):
                outputs.append(batch[i])
        return outputs
    
    def is_compatible_with(self, other: Optimization) -> bool:
        """Batching is compatible with most optimizations."""
        return True
