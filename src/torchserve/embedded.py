"""Embedded TorchServe mode for direct handler invocation.

This module allows using the TorchServe handler directly without
running a TorchServe server. This is useful for:
- Benchmarking handler overhead vs direct model inference
- Testing handler preprocessing/postprocessing
- Local development without server setup

The interface matches TorchServeManager for easy switching between modes.

Usage:
    embedded = EmbeddedTorchServe(optimization="vmap_backbone")
    embedded.start()  # Initializes handler
    
    result = embedded.infer(image)
    
    embedded.stop()  # Cleanup
"""

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .handler import EfficientDetHandler


@dataclass
class MockContext:
    """Mock TorchServe context for embedded mode."""
    system_properties: Dict[str, Any]
    
    def __init__(
        self,
        model_dir: str = ".",
        gpu_id: Optional[int] = None,
    ):
        self.system_properties = {
            "model_dir": model_dir,
            "gpu_id": gpu_id,
        }


class EmbeddedTorchServe:
    """Embedded TorchServe for direct handler invocation.
    
    This class wraps the EfficientDetHandler to provide the same
    interface as TorchServeManager, but without HTTP overhead.
    
    Example:
        embedded = EmbeddedTorchServe(optimization="vmap_backbone")
        
        with embedded:  # Initializes handler
            result = embedded.infer(image_bytes)
            
    Comparison with TorchServeManager:
        - No network overhead (no HTTP serialization)
        - Same preprocessing/postprocessing as external mode
        - Useful for isolating handler overhead from network overhead
    """
    
    def __init__(
        self,
        optimization: str = "baseline",
        device: str = "auto",
        model_format: str = "eager",
        model_dir: Optional[str] = None,
    ):
        """Initialize embedded TorchServe.
        
        Args:
            optimization: Optimization configuration name
            device: Device string (auto, cpu, cuda, mps)
            model_format: Model format (eager, torchscript, onnx)
            model_dir: Model directory (for finding checkpoints)
        """
        self.optimization = optimization
        self.device = device
        self.model_format = model_format
        self.model_dir = model_dir or str(Path(__file__).parent.parent.parent)
        
        self._handler: Optional[EfficientDetHandler] = None
        self._started = False
    
    def __enter__(self) -> "EmbeddedTorchServe":
        """Context manager entry - initialize handler."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup."""
        self.stop()
    
    def start(self) -> None:
        """Initialize the handler."""
        if self._started:
            return
        
        import os
        
        # Set environment variables for handler configuration
        os.environ["OPTIMIZATION"] = self.optimization
        os.environ["DEVICE"] = self.device
        os.environ["MODEL_FORMAT"] = self.model_format
        
        # Create handler and initialize
        self._handler = EfficientDetHandler()
        
        context = MockContext(model_dir=self.model_dir)
        self._handler.initialize(context)
        
        self._started = True
    
    def stop(self) -> None:
        """Cleanup handler resources."""
        self._handler = None
        self._started = False
    
    def is_healthy(self) -> bool:
        """Check if handler is initialized."""
        return self._started and self._handler is not None
    
    def infer(
        self,
        image: Union[bytes, Image.Image, Path, str],
    ) -> Dict[str, Any]:
        """Run inference using the handler.
        
        Args:
            image: Image data (bytes, PIL Image, or path)
            
        Returns:
            Detection results dictionary (same format as TorchServeManager)
        """
        if not self._started or self._handler is None:
            raise RuntimeError("Handler not initialized. Call start() first.")
        
        # Convert image to bytes (matching TorchServe request format)
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image
        
        # Create request data (mimicking TorchServe format)
        request_data = [{"body": image_bytes}]
        
        # Time the full handler pipeline
        start_time = time.perf_counter()
        
        # Run through handler
        results = self._handler.handle(request_data, None)
        
        request_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Add timing info
        if results and len(results) > 0:
            result = results[0]
            result["request_time_ms"] = request_time_ms
            return result
        
        return {"error": "No results", "request_time_ms": request_time_ms}
    
    def infer_batch(
        self,
        images: List[Union[bytes, Image.Image, Path, str]],
    ) -> List[Dict[str, Any]]:
        """Run batch inference.
        
        Args:
            images: List of images
            
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            result = self.infer(image)
            results.append(result)
        return results
    
    def infer_raw(self, image: Image.Image) -> List[Any]:
        """Run inference and return raw DetectionOutput objects.
        
        This bypasses the postprocessing step for direct comparison
        with the underlying model implementation.
        
        Args:
            image: PIL Image
            
        Returns:
            List of DetectionOutput objects from all models
        """
        if not self._started or self._handler is None:
            raise RuntimeError("Handler not initialized. Call start() first.")
        
        # Preprocess
        images = self._handler.preprocess([{"body": self._image_to_bytes(image)}])
        
        # Inference (returns raw DetectionOutput objects)
        outputs = self._handler.inference(images)
        
        return outputs[0] if outputs else []
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to JPEG bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()
    
    @property
    def impl(self):
        """Get the underlying model implementation for direct access."""
        if self._handler is None:
            return None
        return self._handler.impl
