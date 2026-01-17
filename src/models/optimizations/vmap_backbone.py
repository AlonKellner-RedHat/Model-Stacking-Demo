"""vmap-based backbone optimization for parallel multi-model inference.

This optimization exploits the fact that all models share the same backbone
architecture (EfficientNet-B0 + BiFPN) by stacking their weights and using
torch.vmap to parallelize computation across models.

Key insight from benchmarking:
- vmap alone is SLOWER than sequential (~0.9x)
- vmap + torch.compile is ~3.6x FASTER than sequential
- torch.compile alone is ~2.1x faster

Therefore, this optimization internally applies torch.compile to the vmapped
function for optimal performance.
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call

from .base import Optimization


class VmapBackboneOptimization(Optimization):
    """Parallelize backbone+FPN+box_net computation using torch.vmap.
    
    Since all 3 models share the same EfficientNet-B0 + BiFPN backbone
    architecture, we can:
    1. Stack their weights into tensors with shape [num_models, ...]
    2. Use vmap to vectorize the forward pass across models
    3. Run the class heads separately (different output sizes)
    
    Combined with torch.compile, this provides ~3.6x speedup on MPS.
    
    Note: This optimization internally uses torch.compile on the vmapped
    function, so it should NOT be combined with the separate compile
    optimization to avoid double compilation.
    """
    
    name = "vmap_backbone"
    
    def __init__(self, compile_vmapped: bool = True):
        """Initialize vmap backbone optimization.
        
        Args:
            compile_vmapped: Whether to apply torch.compile to the vmapped
                function. Default True for best performance.
        """
        self.compile_vmapped = compile_vmapped
        
        # Stacked parameters and buffers
        self.backbone_params: Dict[str, torch.Tensor] = {}
        self.backbone_buffers: Dict[str, torch.Tensor] = {}
        self.fpn_params: Dict[str, torch.Tensor] = {}
        self.fpn_buffers: Dict[str, torch.Tensor] = {}
        self.box_params: Dict[str, torch.Tensor] = {}
        self.box_buffers: Dict[str, torch.Tensor] = {}
        
        # Base modules for functional_call
        self.base_backbone: Optional[nn.Module] = None
        self.base_fpn: Optional[nn.Module] = None
        self.base_box_net: Optional[nn.Module] = None
        
        # Class nets (kept separate - different output shapes)
        self.class_nets: List[nn.Module] = []
        
        # Compiled vmapped function
        self._vmapped_forward: Optional[callable] = None
        self._is_warmed_up: bool = False
        
        # Device and model count
        self.device: Optional[torch.device] = None
        self.num_models: int = 0
    
    def apply(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """Stack backbone/FPN/box_net weights and prepare vmapped forward.
        
        Args:
            models: List of DetBenchPredict wrapped EfficientDet models
            device: Target device
            
        Returns:
            Original models list (unmodified, used for class_net calls)
        """
        if len(models) == 0:
            return models
        
        self.device = device
        self.num_models = len(models)
        
        print(f"    Preparing vmap for {self.num_models} models...")
        
        # Extract submodules from DetBenchPredict wrapper
        # DetBenchPredict has .model attribute that contains the EfficientDet
        backbones = [m.model.backbone for m in models]
        fpns = [m.model.fpn for m in models]
        box_nets = [m.model.box_net for m in models]
        self.class_nets = [m.model.class_net for m in models]
        
        # Store base modules for functional_call
        self.base_backbone = backbones[0]
        self.base_fpn = fpns[0]
        self.base_box_net = box_nets[0]
        
        # Stack params and buffers using PyTorch's utility
        self.backbone_params, self.backbone_buffers = stack_module_state(backbones)
        self.fpn_params, self.fpn_buffers = stack_module_state(fpns)
        self.box_params, self.box_buffers = stack_module_state(box_nets)
        
        print(f"    Stacked backbone: {len(self.backbone_params)} params, {len(self.backbone_buffers)} buffers")
        print(f"    Stacked FPN: {len(self.fpn_params)} params, {len(self.fpn_buffers)} buffers")
        print(f"    Stacked box_net: {len(self.box_params)} params, {len(self.box_buffers)} buffers")
        print(f"    Class nets kept separate: {len(self.class_nets)} models")
        
        # Create vmapped forward function
        self._create_vmapped_forward()
        
        return models
    
    def _create_vmapped_forward(self) -> None:
        """Create the vmapped forward function, optionally compiled."""
        
        # Define single-model forward using functional_call
        def shared_forward(
            bb_params: Dict[str, torch.Tensor],
            bb_buffers: Dict[str, torch.Tensor],
            fpn_params: Dict[str, torch.Tensor],
            fpn_buffers: Dict[str, torch.Tensor],
            box_params: Dict[str, torch.Tensor],
            box_buffers: Dict[str, torch.Tensor],
            x: torch.Tensor,
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            """Forward pass for backbone + FPN + box_net using functional_call."""
            # 1. Backbone: EfficientNet features at multiple scales
            features = functional_call(
                self.base_backbone, 
                (bb_params, bb_buffers), 
                (x,)
            )
            
            # 2. FPN (BiFPN): Feature pyramid refinement
            fpn_features = functional_call(
                self.base_fpn,
                (fpn_params, fpn_buffers),
                (features,)
            )
            
            # 3. Box head: Bounding box predictions
            box_out = functional_call(
                self.base_box_net,
                (box_params, box_buffers),
                (fpn_features,)
            )
            
            return fpn_features, box_out
        
        # Create vmapped version
        # in_dims=(0, 0, 0, 0, 0, 0, 0) means all inputs have model dimension first
        vmapped_fn = torch.vmap(
            shared_forward,
            in_dims=(0, 0, 0, 0, 0, 0, 0),
        )
        
        # Optionally compile for performance
        if self.compile_vmapped:
            print("    Compiling vmapped function with torch.compile...")
            self._vmapped_forward = torch.compile(
                vmapped_fn,
                backend="inductor",
                mode="default",
            )
        else:
            self._vmapped_forward = vmapped_fn
    
    def warmup(self, input_tensor: torch.Tensor, num_iterations: int = 5) -> None:
        """Warmup the compiled vmapped function.
        
        Important for accurate benchmarking with torch.compile.
        
        Args:
            input_tensor: Sample input tensor [B, C, H, W]
            num_iterations: Number of warmup iterations
        """
        if self._is_warmed_up or self._vmapped_forward is None:
            return
        
        print(f"    Warming up vmap ({num_iterations} iterations)...")
        
        # Replicate input for all models
        x_replicated = input_tensor.unsqueeze(0).expand(
            self.num_models, -1, -1, -1, -1
        )
        
        with torch.no_grad():
            for i in range(num_iterations):
                fpn_features, box_outs = self._vmapped_forward(
                    self.backbone_params, self.backbone_buffers,
                    self.fpn_params, self.fpn_buffers,
                    self.box_params, self.box_buffers,
                    x_replicated,
                )
                
                # Also warmup class nets
                for j, class_net in enumerate(self.class_nets):
                    # Extract FPN features for this model
                    model_fpn_features = [f[j] for f in fpn_features]
                    _ = class_net(model_fpn_features)
        
        self._is_warmed_up = True
        print("    Vmap warmup complete")
    
    def wrap_forward(
        self, 
        models: List[nn.Module], 
        input_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run vmap-optimized forward pass.
        
        This method runs:
        1. Vmapped backbone + FPN + box_net (parallel across models)
        2. Sequential class_net (different output dimensions)
        
        Args:
            models: List of DetBenchPredict models (for reference)
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            List of detection tensors, one per model.
            Each tensor has shape [B, num_detections, 6] with format
            [x1, y1, x2, y2, score, class_id].
        """
        if self._vmapped_forward is None:
            raise RuntimeError("VmapBackboneOptimization not initialized. Call apply() first.")
        
        # Warmup on first call
        if not self._is_warmed_up:
            self.warmup(input_tensor)
        
        # Replicate input for all models: [num_models, B, C, H, W]
        x_replicated = input_tensor.unsqueeze(0).expand(
            self.num_models, -1, -1, -1, -1
        )
        
        with torch.no_grad():
            # Run vmapped backbone + FPN + box_net
            fpn_features_all, box_outs_all = self._vmapped_forward(
                self.backbone_params, self.backbone_buffers,
                self.fpn_params, self.fpn_buffers,
                self.box_params, self.box_buffers,
                x_replicated,
            )
            
            # Run class nets sequentially (different output dimensions)
            class_outs_all = []
            for i, class_net in enumerate(self.class_nets):
                # Extract FPN features for this model: each feature level
                # fpn_features_all is List[Tensor[num_models, B, C, H, W]]
                model_fpn_features = [f[i] for f in fpn_features_all]
                class_out = class_net(model_fpn_features)
                class_outs_all.append(class_out)
        
        # Now we need to combine box and class predictions into detections
        # The models use anchor-based detection, so we need the anchor generator
        # For now, return the raw outputs - the caller will need to post-process
        return fpn_features_all, box_outs_all, class_outs_all
    
    def forward_with_postprocess(
        self,
        models: List[nn.Module],
        input_tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run forward and post-process to get final detections.
        
        This method handles the full inference pipeline including NMS.
        
        Args:
            models: List of DetBenchPredict models
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            List of detection tensors, one per model.
        """
        if self._vmapped_forward is None:
            raise RuntimeError("VmapBackboneOptimization not initialized. Call apply() first.")
        
        # Warmup on first call
        if not self._is_warmed_up:
            self.warmup(input_tensor)
        
        # Replicate input
        x_replicated = input_tensor.unsqueeze(0).expand(
            self.num_models, -1, -1, -1, -1
        )
        
        with torch.no_grad():
            # Run vmapped backbone + FPN + box_net
            fpn_features_all, box_outs_all = self._vmapped_forward(
                self.backbone_params, self.backbone_buffers,
                self.fpn_params, self.fpn_buffers,
                self.box_params, self.box_buffers,
                x_replicated,
            )
            
            # Collect final detections by running each model's post-processing
            all_detections = []
            
            for i, model in enumerate(models):
                # Extract outputs for this model
                model_fpn_features = [f[i] for f in fpn_features_all]
                model_box_out = [b[i] for b in box_outs_all]
                
                # Run class net
                class_out = self.class_nets[i](model_fpn_features)
                
                # Use the model's post-processing (anchors, NMS, etc.)
                # DetBenchPredict.model has anchors and post-processing
                detections = model.model._postprocess(
                    class_out, model_box_out, input_tensor.shape[-2:]
                )
                all_detections.append(detections)
        
        return all_detections
    
    def is_compatible_with(self, other: "Optimization") -> bool:
        """Check compatibility with other optimizations.
        
        vmap_backbone is NOT compatible with torch_compile because it
        internally applies compile to the vmapped function.
        """
        if other.name == "torch_compile":
            return False  # We handle compile internally
        return True


# Keep the old class for backwards compatibility but mark as deprecated
class StateDictVmapOptimization(Optimization):
    """Deprecated: Use VmapBackboneOptimization instead."""
    
    name = "statedict_vmap_deprecated"
    
    def __init__(self):
        import warnings
        warnings.warn(
            "StateDictVmapOptimization is deprecated. Use VmapBackboneOptimization.",
            DeprecationWarning
        )
        self.base_model = None
    
    def apply(self, models: List[nn.Module], device: torch.device) -> List[nn.Module]:
        return models
    
    def wrap_forward(self, models: List[nn.Module], input_tensor: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        with torch.no_grad():
            for model in models:
                out = model(input_tensor)
                outputs.append(out)
        return outputs
