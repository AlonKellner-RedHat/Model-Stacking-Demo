"""Custom TorchServe handler for EfficientDet multi-model inference.

This handler wraps the existing OptimizedImpl/BaselineImpl to serve
inference requests via TorchServe. It supports:
- Eager PyTorch models (current)
- TorchScript models (future)
- ONNX/TensorRT models (future)

The handler reuses all existing model loading, preprocessing, and
inference logic from the src/models package.
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

# TorchServe imports
try:
    from ts.torch_handler.base_handler import BaseHandler
    from ts.context import Context
    TORCHSERVE_AVAILABLE = True
except ImportError:
    # Allow module to be imported without TorchServe for testing
    TORCHSERVE_AVAILABLE = False
    BaseHandler = object
    Context = Any

logger = logging.getLogger(__name__)


class EfficientDetHandler(BaseHandler):
    """TorchServe handler for multi-model EfficientDet inference.
    
    This handler loads 3 EfficientDet-D0 models and runs them on each
    request, returning detection results from all models.
    
    Configuration is read from model-config.yaml or environment variables:
    - OPTIMIZATION: baseline | optimized | vmap_backbone | grouped_super_model
    - DEVICE: cpu | cuda | mps
    - MODEL_FORMAT: eager | torchscript | onnx (future)
    
    Example model-config.yaml:
        optimization: vmap_backbone
        device: cuda
        model_format: eager
    """
    
    def __init__(self):
        super().__init__()
        self.impl = None
        self.device = None
        self.optimization = "baseline"
        self.model_format = "eager"
        self.initialized = False
        self._warmup_done = False
    
    def initialize(self, context: "Context") -> None:
        """Initialize the handler with model and configuration.
        
        Called once when TorchServe loads the model.
        
        Args:
            context: TorchServe context containing model artifacts
        """
        logger.info("Initializing EfficientDetHandler...")
        
        # Get configuration from context or environment
        properties = context.system_properties if context else {}
        model_dir = properties.get("model_dir", ".")
        
        # Read model-config.yaml if present
        config_path = Path(model_dir) / "model-config.yaml"
        config = self._load_config(config_path)
        
        # Get settings from config or environment
        self.optimization = config.get(
            "optimization",
            os.environ.get("OPTIMIZATION", "baseline")
        )
        self.model_format = config.get(
            "model_format",
            os.environ.get("MODEL_FORMAT", "eager")
        )
        
        # Determine device
        device_str = config.get("device", os.environ.get("DEVICE", "auto"))
        self.device = self._resolve_device(device_str)
        
        logger.info(f"Configuration: optimization={self.optimization}, "
                   f"device={self.device}, format={self.model_format}")
        
        # Load the model implementation
        self._load_model_impl()
        
        self.initialized = True
        logger.info("EfficientDetHandler initialized successfully")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)
    
    def _load_model_impl(self) -> None:
        """Load the appropriate model implementation."""
        # Add src to path if needed (for when running within TorchServe)
        src_path = Path(__file__).parent.parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        if self.model_format == "eager":
            self._load_eager_model()
        else:
            raise NotImplementedError(
                f"Model format '{self.model_format}' not yet supported. "
                "Use 'eager' for now. TorchScript/ONNX coming soon."
            )
    
    def _load_eager_model(self) -> None:
        """Load eager PyTorch model using existing implementation."""
        from src.models.optimizations.base import OptimizationConfig
        from src.models.optimized import OptimizedImpl
        from src.models.baseline import BaselineImpl
        
        device_str = str(self.device)
        
        if self.optimization == "baseline":
            logger.info("Loading baseline implementation...")
            self.impl = BaselineImpl(device=device_str)
        else:
            # Map optimization name to config
            config = self._get_optimization_config(self.optimization)
            logger.info(f"Loading optimized implementation: {config}")
            self.impl = OptimizedImpl(device=device_str, optimization_config=config)
        
        self.impl.load()
        logger.info(f"Loaded {self.impl.num_models} models on {self.device}")
    
    def _get_optimization_config(self, name: str) -> "OptimizationConfig":
        """Get OptimizationConfig for a named configuration."""
        from src.models.optimizations.base import OptimizationConfig
        
        configs = {
            "optimized": OptimizationConfig(),  # No optimizations, just wrapper
            "compile": OptimizationConfig(
                compile_enabled=True,
                compile_backend="inductor",
                compile_mode="default",
            ),
            "vmap_backbone": OptimizationConfig(
                vmap_backbone_enabled=True,
            ),
            "grouped_super_model": OptimizationConfig(
                grouped_super_model_enabled=True,
            ),
        }
        
        if name not in configs:
            logger.warning(f"Unknown optimization '{name}', using baseline config")
            return OptimizationConfig()
        
        return configs[name]
    
    def preprocess(self, data: List[Dict[str, Any]]) -> List[Image.Image]:
        """Preprocess request data into PIL Images.
        
        Args:
            data: List of request dictionaries, each containing image data
            
        Returns:
            List of PIL Image objects
        """
        images = []
        
        for request in data:
            # TorchServe sends data in various formats
            # Handle common cases: body (bytes), data (dict), etc.
            image_data = None
            
            if isinstance(request, dict):
                # Check for body (raw bytes from HTTP request)
                if "body" in request:
                    image_data = request["body"]
                # Check for data key
                elif "data" in request:
                    image_data = request["data"]
                # Check for image key (base64 or bytes)
                elif "image" in request:
                    image_data = request["image"]
            elif isinstance(request, (bytes, bytearray)):
                image_data = request
            
            if image_data is None:
                raise ValueError(f"Could not extract image data from request: {type(request)}")
            
            # Convert to PIL Image
            if isinstance(image_data, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                # Assume base64 encoded
                import base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            images.append(image)
        
        return images
    
    def inference(self, images: List[Image.Image]) -> List[List["DetectionOutput"]]:
        """Run inference on preprocessed images.
        
        Supports dynamic batching from TorchServe - when multiple requests
        arrive within max_batch_delay, they are batched together here.
        
        Args:
            images: List of PIL Image objects (may be batched by TorchServe)
            
        Returns:
            List of detection outputs (one list per image, containing outputs from all models)
        """
        if not self.initialized or self.impl is None:
            raise RuntimeError("Handler not initialized. Call initialize() first.")
        
        # Warmup on first call
        if not self._warmup_done:
            logger.info("Running warmup inference...")
            if hasattr(self.impl, "warmup"):
                self.impl.warmup()
            else:
                # Manual warmup
                _ = self.impl.predict(images[0])
            self._warmup_done = True
            logger.info("Warmup complete")
        
        batch_size = len(images)
        if batch_size > 1:
            logger.debug(f"Processing batch of {batch_size} images (dynamic batching)")
        
        # Run inference on each image
        # TODO: True batched inference would stack images into single tensor
        # For now, we process sequentially but benefit from amortized overhead
        all_outputs = []
        for image in images:
            outputs = self.impl.predict(image)
            all_outputs.append(outputs)
        
        return all_outputs
    
    def postprocess(
        self, 
        inference_outputs: List[List["DetectionOutput"]]
    ) -> List[Dict[str, Any]]:
        """Convert detection outputs to JSON-serializable format.
        
        Args:
            inference_outputs: List of detection outputs per image
            
        Returns:
            List of JSON-serializable result dictionaries
        """
        results = []
        
        for image_outputs in inference_outputs:
            image_result = {
                "detections": [],
                "num_models": len(image_outputs),
            }
            
            total_time_ms = 0
            for output in image_outputs:
                detection = {
                    "model_name": output.model_name,
                    "boxes": output.boxes.cpu().tolist(),
                    "scores": output.scores.cpu().tolist(),
                    "labels": [int(l) for l in output.labels.cpu().tolist()],
                    "inference_time_ms": output.inference_time_ms,
                }
                image_result["detections"].append(detection)
                total_time_ms += output.inference_time_ms
            
            image_result["total_inference_time_ms"] = total_time_ms
            results.append(image_result)
        
        return results
    
    def handle(self, data: List[Dict], context: "Context") -> List[Dict]:
        """Main entry point for TorchServe requests.
        
        This method orchestrates the full inference pipeline:
        preprocess -> inference -> postprocess
        
        Args:
            data: List of request data
            context: TorchServe context
            
        Returns:
            List of response dictionaries
        """
        if not self.initialized:
            self.initialize(context)
        
        start_time = time.perf_counter()
        
        try:
            # Preprocess
            images = self.preprocess(data)
            
            # Inference
            outputs = self.inference(images)
            
            # Postprocess
            results = self.postprocess(outputs)
            
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Add timing to results
            for result in results:
                result["handler_time_ms"] = total_time_ms / len(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return [{"error": str(e)}] * len(data)


# For direct invocation (testing/embedded mode)
_handler_instance: Optional[EfficientDetHandler] = None


def get_handler() -> EfficientDetHandler:
    """Get or create the singleton handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = EfficientDetHandler()
    return _handler_instance
