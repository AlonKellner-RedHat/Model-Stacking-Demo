"""Baseline implementation: Sequential inference with 3 EfficientDet models."""

import time
from typing import List, Optional

import torch
from effdet import create_model, DetBenchPredict
from PIL import Image
from torchvision import transforms

from .base import BaseModelImpl, DetectionOutput


# EfficientDet model configurations
MODEL_CONFIGS = [
    {"name": "tf_efficientdet_d0", "image_size": 512},
    {"name": "tf_efficientdet_d1", "image_size": 640},
    {"name": "tf_efficientdet_d2", "image_size": 768},
]


class BaselineImpl(BaseModelImpl):
    """Baseline implementation running 3 EfficientDet models sequentially.
    
    This provides the upper bound on latency and serves as the reference
    for output comparison with other implementations.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the baseline implementation.
        
        Args:
            device: Device to run inference on. If None, uses CUDA if available.
        """
        super().__init__(device)
        self.models: List[DetBenchPredict] = []
        self.model_names: List[str] = []
        self.transforms: List[transforms.Compose] = []

    def load(self) -> None:
        """Load all 3 EfficientDet models (D0, D1, D2)."""
        if self._is_loaded:
            return

        for config in MODEL_CONFIGS:
            model_name = config["name"]
            image_size = config["image_size"]

            # Create model with pretrained weights
            model = create_model(
                model_name,
                bench_task="predict",
                pretrained=True,
                num_classes=90,  # COCO classes
            )
            
            # Wrap in DetBenchPredict for inference
            bench = DetBenchPredict(model)
            bench.eval()
            bench.to(self.device)
            
            self.models.append(bench)
            self.model_names.append(model_name)
            
            # Create transform for this model's input size
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            self.transforms.append(transform)

        self._is_loaded = True

    def _preprocess(self, image: Image.Image, model_idx: int) -> torch.Tensor:
        """Preprocess image for a specific model.
        
        Args:
            image: PIL Image in RGB format.
            model_idx: Index of the model to preprocess for.
            
        Returns:
            Preprocessed tensor ready for inference.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        tensor = self.transforms[model_idx](image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> List[DetectionOutput]:
        """Run inference on a single image with all 3 models.
        
        Args:
            image: PIL Image to run detection on.
            
        Returns:
            List of 3 DetectionOutput, one per model.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        outputs = []
        
        for idx, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            # Preprocess for this model
            input_tensor = self._preprocess(image, idx)
            
            # Get original image size for scaling
            img_info = {
                "img_scale": torch.tensor([[1.0]], device=self.device),
                "img_size": torch.tensor([[image.height, image.width]], device=self.device),
            }
            
            # Run inference with timing
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                detections = model(input_tensor, img_info)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Parse detections: [batch, num_detections, 6]
            # Format: [x1, y1, x2, y2, score, class_id]
            if detections is not None and len(detections) > 0:
                det = detections[0]  # First batch item
                boxes = det[:, :4]
                scores = det[:, 4]
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
        """Run inference on a batch of images.
        
        Args:
            images: List of PIL Images.
            
        Returns:
            List of lists of DetectionOutput.
        """
        return [self.predict(image) for image in images]

    @property
    def name(self) -> str:
        """Return the name of this implementation."""
        return "baseline_sequential"

    @property
    def num_models(self) -> int:
        """Return the number of models."""
        return len(MODEL_CONFIGS)
