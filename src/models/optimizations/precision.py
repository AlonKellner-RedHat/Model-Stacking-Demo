"""Mixed precision optimization for faster inference."""

from typing import List
import torch
import torch.nn as nn

from .base import Optimization


class MixedPrecisionOptimization(Optimization):
    """Apply mixed precision (FP16/BF16) for faster inference.
    
    Uses automatic mixed precision to reduce memory bandwidth and
    leverage faster tensor cores on supported hardware.
    """
    
    name = "mixed_precision"
    
    def __init__(self, dtype: torch.dtype = torch.float16):
        """Initialize mixed precision optimization.
        
        Args:
            dtype: Target dtype (torch.float16 or torch.bfloat16)
        """
        self.dtype = dtype
        self._original_dtypes: List[torch.dtype] = []
    
    def apply(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """Convert models to target precision."""
        converted = []
        
        for i, model in enumerate(models):
            # Store original dtype
            first_param = next(model.parameters(), None)
            if first_param is not None:
                self._original_dtypes.append(first_param.dtype)
            
            # Check device compatibility
            if device.type == "mps" and self.dtype == torch.bfloat16:
                print(f"    Warning: MPS doesn't support bfloat16, using float16")
                self.dtype = torch.float16
            
            # Convert model to target dtype
            try:
                model = model.to(dtype=self.dtype)
                converted.append(model)
                print(f"    Converted model {i} to {self.dtype}")
            except Exception as e:
                print(f"    Warning: Could not convert model {i} to {self.dtype}: {e}")
                converted.append(model)
        
        return converted
    
    def wrap_forward(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run forward pass with autocast for mixed precision."""
        outputs = []
        
        device_type = input_tensor.device.type
        
        # Ensure input is correct dtype
        input_casted = input_tensor.to(dtype=self.dtype)
        
        for model in models:
            with torch.no_grad():
                # Use autocast for automatic precision handling
                with torch.autocast(device_type=device_type, dtype=self.dtype):
                    out = model(input_casted)
                outputs.append(out)
        
        return outputs
    
    def is_compatible_with(self, other: Optimization) -> bool:
        """Mixed precision is compatible with most optimizations."""
        return True


class Float16Optimization(MixedPrecisionOptimization):
    """FP16 mixed precision optimization."""
    
    name = "fp16"
    
    def __init__(self):
        super().__init__(dtype=torch.float16)


class BFloat16Optimization(MixedPrecisionOptimization):
    """BFloat16 mixed precision optimization."""
    
    name = "bf16"
    
    def __init__(self):
        super().__init__(dtype=torch.bfloat16)
