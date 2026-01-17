"""Optimized implementation with composable optimizations."""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from effdet import create_model
from effdet.bench import DetBenchPredict
from PIL import Image
from torchvision import transforms

from .base import BaseModelImpl, DetectionOutput
from .optimizations.base import OptimizationConfig, OptimizationStack


# Model architecture constants
MODEL_ARCH = "tf_efficientdet_d0"
IMAGE_SIZE = 512

# Checkpoint files
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
CHECKPOINT_FILES = [
    "efficientdet_d0_coco.pth",
    "efficientdet_d0_aquarium.pth",
    "efficientdet_d0_vehicles.pth",
]


class OptimizedImpl(BaseModelImpl):
    """Optimized implementation with composable optimizations.
    
    Supports various optimization techniques that can be enabled independently
    for ablation studies:
    
    - torch.compile: JIT compilation for optimized kernels
    - mixed_precision: FP16/BF16 inference
    - batched_inference: Process multiple images together
    - vmap_backbone: Parallelize backbone computation
    
    Example:
        config = OptimizationConfig(
            compile_enabled=True,
            mixed_precision_enabled=True,
        )
        impl = OptimizedImpl(device="mps", optimization_config=config)
        impl.load()
        outputs = impl.predict(image)
    """

    def __init__(
        self, 
        device: Optional[str] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize optimized implementation.
        
        Args:
            device: Device to run inference on
            optimization_config: Configuration for optimizations
        """
        super().__init__(device)
        self.optimization_config = optimization_config or OptimizationConfig()
        self.optimization_stack: Optional[OptimizationStack] = None
        self.models: List[torch.nn.Module] = []
        self.model_names: List[str] = []
        self.model_num_classes: List[int] = []
        self._transform: Optional[transforms.Compose] = None
        self._warmup_done: bool = False

    def _checkpoints_exist(self) -> bool:
        """Check if all checkpoint files exist."""
        for filename in CHECKPOINT_FILES:
            if not (CHECKPOINT_DIR / filename).exists():
                return False
        return True

    def load(self) -> None:
        """Load models and apply optimizations."""
        if self._is_loaded:
            return

        print(f"Loading OptimizedImpl with: {self.optimization_config}")
        
        # Load base models
        if not self._checkpoints_exist():
            raise RuntimeError(
                "Checkpoints not found. Run: uv run python scripts/generate_checkpoints.py"
            )
        
        self._load_from_checkpoints()
        
        # Build and apply optimization stack
        self.optimization_stack = OptimizationStack(self.optimization_config)
        self.models = self.optimization_stack.apply_all(self.models, self.device)
        
        # Create super model if grouped optimization is enabled
        if self.optimization_config.grouped_super_model_enabled:
            self.optimization_stack.create_super_model(self.models, self.device)
        
        # Setup transform
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self._is_loaded = True
        print(f"Loaded {len(self.models)} optimized models on {self.device}")

    def _load_from_checkpoints(self) -> None:
        """Load models from checkpoint files."""
        print(f"Loading {len(CHECKPOINT_FILES)} models from checkpoints...")
        
        for i, filename in enumerate(CHECKPOINT_FILES):
            filepath = CHECKPOINT_DIR / filename
            
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
            num_classes = checkpoint.get("num_classes", 90)
            
            raw_model = create_model(
                checkpoint.get("model_arch", MODEL_ARCH),
                bench_task=None,
                pretrained=False,
                num_classes=num_classes,
            )
            
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            
            model = DetBenchPredict(raw_model)
            model.eval()
            model.to(self.device)
            
            self.models.append(model)
            self.model_names.append(checkpoint.get("name", f"model_{i}"))
            self.model_num_classes.append(num_classes)
            
            print(f"  [{i+1}] {self.model_names[-1]}: {num_classes} classes")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        tensor = self._transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Apply precision optimization if enabled
        if self.optimization_config.mixed_precision_enabled:
            tensor = tensor.to(dtype=self.optimization_config.dtype)
        
        return tensor

    def warmup(self, num_iterations: int = 3) -> None:
        """Warmup models with dummy inference.
        
        Important for accurate benchmarking, especially with torch.compile.
        """
        if self._warmup_done:
            return
        
        print(f"Warming up with {num_iterations} iterations...")
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        
        if self.optimization_config.mixed_precision_enabled:
            dummy_input = dummy_input.to(dtype=self.optimization_config.dtype)
        
        # If using vmap, warmup is handled by the vmap optimization
        if self.optimization_stack and self.optimization_stack.uses_vmap_forward:
            self.optimization_stack.vmap_optimization.warmup(dummy_input, num_iterations)
        else:
            for i in range(num_iterations):
                for model in self.models:
                    with torch.no_grad():
                        _ = model(dummy_input)
        
        self._warmup_done = True
        print("Warmup complete")

    def predict(self, image: Image.Image) -> List[DetectionOutput]:
        """Run optimized inference on a single image."""
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        input_tensor = self._preprocess(image)
        
        # Warmup on first call
        if not self._warmup_done:
            self.warmup()
        
        # Use grouped super model forward path if enabled
        if self.optimization_stack and self.optimization_stack.uses_grouped_forward:
            return self._predict_grouped(input_tensor)
        
        # Use vmap forward path if enabled
        if self.optimization_stack and self.optimization_stack.uses_vmap_forward:
            return self._predict_vmap(input_tensor)
        
        # Standard sequential forward path
        return self._predict_sequential(input_tensor)
    
    def _predict_vmap(self, input_tensor: torch.Tensor) -> List[DetectionOutput]:
        """Run inference using vmap-optimized forward pass."""
        from effdet.bench import _post_process, _batch_detection
        
        vmap_opt = self.optimization_stack.vmap_optimization
        
        # Synchronize for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        
        start_time = time.perf_counter()
        
        # Run vmapped forward (backbone + FPN + box_net parallel, class_net sequential)
        with torch.no_grad():
            if self.optimization_config.mixed_precision_enabled:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.optimization_config.dtype,
                ):
                    fpn_features_all, box_outs_all, class_outs_all = vmap_opt.wrap_forward(
                        self.models, input_tensor
                    )
            else:
                fpn_features_all, box_outs_all, class_outs_all = vmap_opt.wrap_forward(
                    self.models, input_tensor
                )
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        time_per_model = total_time_ms / len(self.models)
        
        # Post-process outputs for each model using effdet's post-processing
        outputs = []
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            bench = self.models[i]  # DetBenchPredict wrapper
            
            # Get outputs for this model
            # box_outs_all and class_outs_all are lists of [num_models, B, C, H, W] tensors
            model_box_out = [b[i] for b in box_outs_all]
            model_class_out = class_outs_all[i]  # class_outs_all is per-model list
            
            # Run effdet's post-processing
            class_out_pp, box_out_pp, indices, classes = _post_process(
                model_class_out,
                model_box_out,
                num_levels=bench.num_levels,
                num_classes=bench.num_classes,
                max_detection_points=bench.max_detection_points,
            )
            
            # Run NMS and detection decoding
            detections = _batch_detection(
                input_tensor.shape[0],  # batch size
                class_out_pp,
                box_out_pp,
                bench.anchors.boxes,
                indices,
                classes,
                img_scale=None,
                img_size=None,
                max_det_per_image=bench.max_det_per_image,
                soft_nms=bench.soft_nms,
            )
            
            # Parse detections [batch, max_det, 6]
            if detections is not None and len(detections) > 0:
                det = detections[0]  # First batch item
                # Filter valid detections (score > 0)
                valid_mask = det[:, 4] > 0
                det = det[valid_mask]
                
                if len(det) > 0:
                    boxes = det[:, :4].float()
                    scores = det[:, 4].float()
                    labels = det[:, 5].long()
                else:
                    boxes = torch.empty((0, 4), device=self.device)
                    scores = torch.empty((0,), device=self.device)
                    labels = torch.empty((0,), dtype=torch.long, device=self.device)
            else:
                boxes = torch.empty((0, 4), device=self.device)
                scores = torch.empty((0,), device=self.device)
                labels = torch.empty((0,), dtype=torch.long, device=self.device)
            
            outputs.append(DetectionOutput(
                boxes=boxes,
                scores=scores,
                labels=labels,
                model_name=model_name,
                inference_time_ms=time_per_model,
            ))
        
        return outputs
    
    def _predict_grouped(self, input_tensor: torch.Tensor) -> List[DetectionOutput]:
        """Run inference using grouped super model forward pass."""
        super_model = self.optimization_stack.super_model
        
        if super_model is None:
            raise RuntimeError("Super model not initialized. Call load() first.")
        
        # Synchronize for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if self.optimization_config.mixed_precision_enabled:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.optimization_config.dtype,
                ):
                    detections_list = super_model.detect(input_tensor, self.models)
            else:
                detections_list = super_model.detect(input_tensor, self.models)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        time_per_model = total_time_ms / len(self.models)
        
        outputs = []
        for i, (det, model_name) in enumerate(zip(detections_list, self.model_names)):
            # Filter valid detections
            if det is not None and len(det) > 0:
                valid_mask = det[:, 4] > 0
                det = det[valid_mask]
                
                if len(det) > 0:
                    boxes = det[:, :4].float()
                    scores = det[:, 4].float()
                    labels = det[:, 5].long()
                else:
                    boxes = torch.empty((0, 4), device=self.device)
                    scores = torch.empty((0,), device=self.device)
                    labels = torch.empty((0,), dtype=torch.long, device=self.device)
            else:
                boxes = torch.empty((0, 4), device=self.device)
                scores = torch.empty((0,), device=self.device)
                labels = torch.empty((0,), dtype=torch.long, device=self.device)
            
            outputs.append(DetectionOutput(
                boxes=boxes,
                scores=scores,
                labels=labels,
                model_name=model_name,
                inference_time_ms=time_per_model,
            ))
        
        return outputs
    
    def _predict_sequential(self, input_tensor: torch.Tensor) -> List[DetectionOutput]:
        """Run inference using standard sequential forward pass."""
        outputs = []
        
        for model, model_name in zip(self.models, self.model_names):
            # Synchronize for accurate timing
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                if self.optimization_config.mixed_precision_enabled:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.optimization_config.dtype,
                    ):
                        detections = model(input_tensor)
                else:
                    detections = model(input_tensor)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Parse detections
            if detections is not None and len(detections) > 0:
                det = detections[0]
                boxes = det[:, :4].float()
                scores = det[:, 4].float()
                labels = det[:, 5].long()
            else:
                boxes = torch.empty((0, 4), device=self.device)
                scores = torch.empty((0,), device=self.device)
                labels = torch.empty((0,), dtype=torch.long, device=self.device)
            
            outputs.append(DetectionOutput(
                boxes=boxes,
                scores=scores,
                labels=labels,
                model_name=model_name,
                inference_time_ms=inference_time_ms,
            ))

        return outputs

    def predict_batch(self, images: List[Image.Image]) -> List[List[DetectionOutput]]:
        """Run inference on multiple images."""
        return [self.predict(image) for image in images]

    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about enabled optimizations."""
        return {
            "config": str(self.optimization_config),
            "enabled": self.optimization_config.get_enabled_optimizations(),
            "device": str(self.device),
            "num_models": len(self.models),
        }

    @property
    def name(self) -> str:
        """Return implementation name including optimizations."""
        return f"optimized_{self.optimization_config}"

    @property
    def num_models(self) -> int:
        return len(CHECKPOINT_FILES)

    @property
    def architecture(self) -> str:
        return MODEL_ARCH

    @property
    def input_size(self) -> int:
        return IMAGE_SIZE


# Convenience factory functions for common optimization configurations

def create_baseline() -> OptimizedImpl:
    """Create baseline implementation (no optimizations)."""
    return OptimizedImpl(optimization_config=OptimizationConfig())


def create_compiled(device: str = "mps") -> OptimizedImpl:
    """Create implementation with torch.compile."""
    config = OptimizationConfig(
        compile_enabled=True,
        compile_backend="inductor",
        compile_mode="default",
    )
    return OptimizedImpl(device=device, optimization_config=config)


def create_fp16(device: str = "mps") -> OptimizedImpl:
    """Create implementation with FP16 precision."""
    config = OptimizationConfig(
        mixed_precision_enabled=True,
        dtype=torch.float16,
    )
    return OptimizedImpl(device=device, optimization_config=config)


def create_compiled_fp16(device: str = "mps") -> OptimizedImpl:
    """Create implementation with torch.compile + FP16."""
    config = OptimizationConfig(
        compile_enabled=True,
        compile_backend="inductor",
        mixed_precision_enabled=True,
        dtype=torch.float16,
    )
    return OptimizedImpl(device=device, optimization_config=config)


def create_vmap_backbone(device: str = "mps") -> OptimizedImpl:
    """Create implementation with vmap backbone optimization.
    
    This uses torch.vmap to parallelize backbone+FPN+box_net computation
    across all 3 models, with torch.compile internally applied.
    
    Expected speedup: ~3.6x on MPS vs baseline.
    """
    config = OptimizationConfig(
        vmap_backbone_enabled=True,
    )
    return OptimizedImpl(device=device, optimization_config=config)


def create_all_optimizations(device: str = "mps") -> OptimizedImpl:
    """Create implementation with all optimizations enabled.
    
    Note: vmap_backbone includes compile internally, so we use vmap
    instead of separate compile for best performance.
    """
    config = OptimizationConfig(
        vmap_backbone_enabled=True,  # Includes compile internally
    )
    return OptimizedImpl(device=device, optimization_config=config)
