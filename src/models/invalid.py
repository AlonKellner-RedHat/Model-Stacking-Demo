"""Invalid implementation: Returns constant outputs without model computation.

This provides the lower bound on latency and serves as the maximum
difference reference for output comparison.
"""

import time
from typing import List, Optional

import torch
from PIL import Image

from .base import BaseModelImpl, DetectionOutput


class InvalidImpl(BaseModelImpl):
    """Invalid implementation that skips model computation entirely.
    
    Returns constant zero tensors matching the expected output format.
    This establishes:
    - Lower bound on latency (minimal processing)
    - Upper bound on output difference (maximum error vs reference)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        num_models: int = 3,
        num_dummy_detections: int = 100,
    ):
        """Initialize the invalid implementation.
        
        Args:
            device: Device to create tensors on. If None, uses CUDA if available.
            num_models: Number of "fake" models to simulate.
            num_dummy_detections: Number of dummy detections to return per model.
        """
        super().__init__(device)
        self._num_models = num_models
        self.num_dummy_detections = num_dummy_detections
        self.model_names = [f"invalid_model_{i}" for i in range(num_models)]

    def load(self) -> None:
        """No-op: No models to load."""
        self._is_loaded = True

    def predict(self, image: Image.Image) -> List[DetectionOutput]:
        """Return constant dummy outputs without any model computation.
        
        Args:
            image: PIL Image (only decoded to simulate minimal overhead).
            
        Returns:
            List of DetectionOutput with constant zero values.
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() first.")

        outputs = []
        
        # Minimal image processing to simulate request overhead
        # Just access image properties without actual computation
        _ = image.size
        _ = image.mode
        
        for model_name in self.model_names:
            start_time = time.perf_counter()
            
            # Create constant dummy outputs (zeros)
            boxes = torch.zeros(
                (self.num_dummy_detections, 4),
                device=self.device,
                dtype=torch.float32
            )
            scores = torch.zeros(
                (self.num_dummy_detections,),
                device=self.device,
                dtype=torch.float32
            )
            labels = torch.zeros(
                (self.num_dummy_detections,),
                device=self.device,
                dtype=torch.long
            )
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            outputs.append(DetectionOutput(
                boxes=boxes,
                scores=scores,
                labels=labels,
                model_name=model_name,
                inference_time_ms=inference_time_ms,
            ))

        return outputs

    def predict_batch(self, images: List[Image.Image]) -> List[List[DetectionOutput]]:
        """Return constant dummy outputs for a batch of images.
        
        Args:
            images: List of PIL Images.
            
        Returns:
            List of lists of DetectionOutput with constant values.
        """
        return [self.predict(image) for image in images]

    @property
    def name(self) -> str:
        """Return the name of this implementation."""
        return "invalid_constant"

    @property
    def num_models(self) -> int:
        """Return the number of simulated models."""
        return self._num_models
