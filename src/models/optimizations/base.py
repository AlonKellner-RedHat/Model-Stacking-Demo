"""Base classes for composable optimizations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import torch
import torch.nn as nn


@dataclass
class OptimizationConfig:
    """Configuration for optimization stack."""
    
    # torch.compile options
    compile_enabled: bool = False
    compile_backend: str = "inductor"  # inductor, aot_eager, cudagraphs
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    compile_fullgraph: bool = False
    
    # Mixed precision options
    mixed_precision_enabled: bool = False
    dtype: torch.dtype = torch.float16
    
    # Batching options
    batched_inference_enabled: bool = False
    batch_size: int = 1
    
    # vmap backbone options
    vmap_backbone_enabled: bool = False
    
    def get_enabled_optimizations(self) -> List[str]:
        """Get list of enabled optimization names."""
        enabled = []
        if self.compile_enabled:
            enabled.append(f"compile({self.compile_backend})")
        if self.mixed_precision_enabled:
            enabled.append(f"mixed_precision({self.dtype})")
        if self.batched_inference_enabled:
            enabled.append(f"batched(bs={self.batch_size})")
        if self.vmap_backbone_enabled:
            enabled.append("vmap_backbone")
        return enabled if enabled else ["none"]
    
    def __str__(self) -> str:
        return "+".join(self.get_enabled_optimizations())


class Optimization(ABC):
    """Base class for composable optimizations."""
    
    name: str = "base"
    
    @abstractmethod
    def apply(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """Apply optimization to models.
        
        Args:
            models: List of PyTorch models to optimize
            device: Target device
            
        Returns:
            List of optimized models (may be same objects, modified in-place)
        """
        pass
    
    @abstractmethod
    def wrap_forward(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Wrap the forward pass with optimization.
        
        Args:
            models: List of models
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            List of output tensors from each model
        """
        pass
    
    def is_compatible_with(self, other: "Optimization") -> bool:
        """Check if this optimization is compatible with another."""
        return True  # Override in subclasses if needed


class OptimizationStack:
    """Stack of composable optimizations."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizations: List[Optimization] = []
        self._build_stack()
    
    def _build_stack(self) -> None:
        """Build optimization stack from config."""
        from .compile import TorchCompileOptimization
        from .precision import MixedPrecisionOptimization
        from .batching import BatchedInferenceOptimization
        from .vmap_backbone import VmapBackboneOptimization
        
        # Order matters: apply in this sequence
        if self.config.mixed_precision_enabled:
            self.optimizations.append(
                MixedPrecisionOptimization(dtype=self.config.dtype)
            )
        
        if self.config.compile_enabled:
            self.optimizations.append(
                TorchCompileOptimization(
                    backend=self.config.compile_backend,
                    mode=self.config.compile_mode,
                    fullgraph=self.config.compile_fullgraph,
                )
            )
        
        if self.config.vmap_backbone_enabled:
            self.optimizations.append(VmapBackboneOptimization())
        
        if self.config.batched_inference_enabled:
            self.optimizations.append(
                BatchedInferenceOptimization(batch_size=self.config.batch_size)
            )
    
    def apply_all(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """Apply all optimizations to models."""
        for opt in self.optimizations:
            models = opt.apply(models, device)
        return models
    
    def forward(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run optimized forward pass."""
        # For now, apply optimizations sequentially
        # More advanced implementations could fuse operations
        outputs = []
        for model in models:
            with torch.no_grad():
                if self.config.mixed_precision_enabled:
                    with torch.autocast(
                        device_type=input_tensor.device.type,
                        dtype=self.config.dtype,
                    ):
                        out = model(input_tensor)
                else:
                    out = model(input_tensor)
                outputs.append(out)
        return outputs
    
    def __str__(self) -> str:
        return str(self.config)
