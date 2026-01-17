"""torch.compile optimization for faster inference."""

from typing import List
import torch
import torch.nn as nn

from .base import Optimization


class TorchCompileOptimization(Optimization):
    """Apply torch.compile to models for optimized kernels.
    
    Uses PyTorch 2.x compilation to generate optimized code for the model.
    Can provide 1.5-3x speedup depending on model architecture.
    """
    
    name = "torch_compile"
    
    def __init__(
        self,
        backend: str = "inductor",
        mode: str = "default",
        fullgraph: bool = False,
    ):
        """Initialize torch.compile optimization.
        
        Args:
            backend: Compilation backend (inductor, aot_eager, cudagraphs)
            mode: Compilation mode (default, reduce-overhead, max-autotune)
            fullgraph: Whether to compile entire graph or allow breaks
        """
        self.backend = backend
        self.mode = mode
        self.fullgraph = fullgraph
        self._compiled_models: List[nn.Module] = []
    
    def apply(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """Apply torch.compile to each model."""
        compiled = []
        
        for i, model in enumerate(models):
            try:
                compiled_model = torch.compile(
                    model,
                    backend=self.backend,
                    mode=self.mode,
                    fullgraph=self.fullgraph,
                )
                compiled.append(compiled_model)
                print(f"    Compiled model {i} with {self.backend}/{self.mode}")
            except Exception as e:
                print(f"    Warning: Could not compile model {i}: {e}")
                compiled.append(model)
        
        self._compiled_models = compiled
        return compiled
    
    def wrap_forward(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run compiled forward pass."""
        outputs = []
        for model in models:
            with torch.no_grad():
                out = model(input_tensor)
                outputs.append(out)
        return outputs
    
    def is_compatible_with(self, other: Optimization) -> bool:
        """torch.compile is compatible with most optimizations."""
        # May have issues with vmap in some cases
        if other.name == "vmap_backbone":
            return False  # vmap and compile don't mix well currently
        return True
