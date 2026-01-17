"""Abstract base class for model implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image


@dataclass
class DetectionOutput:
    """Standardized detection output format for all implementations.
    
    Attributes:
        boxes: Detection bounding boxes, shape (N, 4) in xyxy format
        scores: Confidence scores, shape (N,)
        labels: Class labels, shape (N,)
        model_name: Name of the model that produced this output
        inference_time_ms: Time taken for inference in milliseconds
    """
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    model_name: str
    inference_time_ms: float

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "boxes": self.boxes.cpu().tolist(),
            "scores": self.scores.cpu().tolist(),
            "labels": self.labels.cpu().tolist(),
            "model_name": self.model_name,
            "inference_time_ms": self.inference_time_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict, device: str = "cpu") -> "DetectionOutput":
        """Create DetectionOutput from dictionary."""
        return cls(
            boxes=torch.tensor(data["boxes"], device=device),
            scores=torch.tensor(data["scores"], device=device),
            labels=torch.tensor(data["labels"], device=device),
            model_name=data["model_name"],
            inference_time_ms=data["inference_time_ms"],
        )


class BaseModelImpl(ABC):
    """Abstract base class for model implementations.
    
    All inference implementations (baseline, vmap, grouped conv, etc.)
    must inherit from this class and implement the required methods.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the model implementation.
        
        Args:
            device: Device to run inference on. If None, uses CUDA if available.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference.
        
        This method should:
        1. Load all required model checkpoints
        2. Move models to the target device
        3. Set models to eval mode
        4. Set self._is_loaded = True
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> List[DetectionOutput]:
        """Run inference on a single image.
        
        Args:
            image: PIL Image to run detection on.
            
        Returns:
            List of DetectionOutput, one per model.
        """
        pass

    @abstractmethod
    def predict_batch(self, images: List[Image.Image]) -> List[List[DetectionOutput]]:
        """Run inference on a batch of images.
        
        Args:
            images: List of PIL Images to run detection on.
            
        Returns:
            List of lists of DetectionOutput. Outer list is per-image,
            inner list is per-model.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this implementation."""
        pass

    @property
    @abstractmethod
    def num_models(self) -> int:
        """Return the number of models in this implementation."""
        pass

    def get_vram_usage(self) -> dict:
        """Get current VRAM usage statistics.
        
        Returns:
            Dictionary with VRAM usage info, or empty dict if not on CUDA.
        """
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return {}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024 / 1024,
        }

    def reset_vram_stats(self) -> None:
        """Reset VRAM peak statistics."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
