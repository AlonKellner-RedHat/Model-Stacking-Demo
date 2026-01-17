"""Baseline implementation: Sequential inference with 3 EfficientDet-D0 models.

All 3 models use the same backbone architecture (EfficientDet-D0, 512x512)
but have different class heads (different number of output classes) to simulate
models fine-tuned for different detection objectives.

Checkpoints must be generated first using: scripts/generate_checkpoints.py
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from effdet import create_model
from effdet.bench import DetBenchPredict
from PIL import Image
from torchvision import transforms

from .base import BaseModelImpl, DetectionOutput


# Model architecture constants
MODEL_ARCH = "tf_efficientdet_d0"
IMAGE_SIZE = 512

# Checkpoint files to load
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
CHECKPOINT_FILES = [
    "efficientdet_d0_coco.pth",
    "efficientdet_d0_aquarium.pth",
    "efficientdet_d0_vehicles.pth",
]


class BaselineImpl(BaseModelImpl):
    """Baseline implementation running 3 EfficientDet-D0 models sequentially.
    
    All models have the SAME backbone architecture but DIFFERENT class heads,
    simulating fine-tuned models for different detection tasks:
    
    - Model 1: COCO (90 classes)
    - Model 2: Aquarium (7 classes) 
    - Model 3: Vehicles (20 classes)
    
    Each model:
    - Architecture: EfficientDet-D0
    - Input: 512x512
    - Backbone parameters: ~3.8M (shared architecture)
    - Class head: varies by num_classes
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the baseline implementation.
        
        Args:
            device: Device to run inference on. If None, uses CUDA if available.
        """
        super().__init__(device)
        self.models: List[torch.nn.Module] = []
        self.model_names: List[str] = []
        self.model_descriptions: List[str] = []
        self.model_num_classes: List[int] = []
        self._transform: Optional[transforms.Compose] = None

    def _checkpoints_exist(self) -> bool:
        """Check if all checkpoint files exist."""
        for filename in CHECKPOINT_FILES:
            if not (CHECKPOINT_DIR / filename).exists():
                return False
        return True

    def load(self) -> None:
        """Load all 3 EfficientDet-D0 models from checkpoints."""
        if self._is_loaded:
            return

        # Check if checkpoints exist
        if not self._checkpoints_exist():
            print("Checkpoints not found. Generating on-the-fly...")
            self._generate_and_load()
        else:
            self._load_from_checkpoints()
        
        # Single shared transform since all models use same input size
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self._is_loaded = True
        print(f"Loaded {len(self.models)} models on {self.device}")
        for name, num_cls in zip(self.model_names, self.model_num_classes):
            print(f"  - {name}: {num_cls} classes")

    def _load_from_checkpoints(self) -> None:
        """Load models from pre-saved checkpoint files."""
        print(f"Loading {len(CHECKPOINT_FILES)} EfficientDet-D0 models from checkpoints...")
        
        for i, filename in enumerate(CHECKPOINT_FILES):
            filepath = CHECKPOINT_DIR / filename
            print(f"  [{i+1}/{len(CHECKPOINT_FILES)}] Loading {filepath.name}...")
            
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
            
            num_classes = checkpoint.get("num_classes", 90)
            
            # Create model architecture with correct number of classes
            raw_model = create_model(
                checkpoint.get("model_arch", MODEL_ARCH),
                bench_task=None,  # Raw model
                pretrained=False,  # Don't load pretrained weights
                num_classes=num_classes,
            )
            
            # Load saved weights
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            
            # Wrap in DetBenchPredict for inference
            model = DetBenchPredict(raw_model)
            model.eval()
            model.to(self.device)
            
            self.models.append(model)
            self.model_names.append(checkpoint.get("name", f"model_{i}"))
            self.model_descriptions.append(checkpoint.get("description", ""))
            self.model_num_classes.append(num_classes)
            
            print(f"      Name: {self.model_names[-1]}")
            print(f"      Classes: {num_classes}")

    def _generate_and_load(self) -> None:
        """Generate models on-the-fly if checkpoints don't exist."""
        print(f"Generating {len(CHECKPOINT_FILES)} EfficientDet-D0 model variations...")
        
        # Variation configurations matching the checkpoint generator
        variations = [
            {"name": "efficientdet_d0_coco", "description": "COCO (90 classes)", 
             "num_classes": 90, "noise": 0.0, "seed": None},
            {"name": "efficientdet_d0_aquarium", "description": "Aquarium (7 classes)", 
             "num_classes": 7, "noise": 0.02, "seed": 42},
            {"name": "efficientdet_d0_vehicles", "description": "Vehicles (20 classes)", 
             "num_classes": 20, "noise": 0.02, "seed": 123},
        ]
        
        for i, var in enumerate(variations):
            print(f"  [{i+1}/{len(variations)}] Creating {var['name']}...")
            
            # Create model with correct number of classes
            model = create_model(
                MODEL_ARCH,
                bench_task="predict",
                pretrained=True,
                num_classes=var["num_classes"],
            )
            
            # Apply weight perturbation
            if var["noise"] > 0 and var["seed"] is not None:
                self._perturb_weights(model.model, var["noise"], var["seed"])
            
            model.eval()
            model.to(self.device)
            
            self.models.append(model)
            self.model_names.append(var["name"])
            self.model_descriptions.append(var["description"])
            self.model_num_classes.append(var["num_classes"])

    def _perturb_weights(self, model: torch.nn.Module, noise_scale: float, seed: int) -> None:
        """Add noise to weights to simulate fine-tuning."""
        torch.manual_seed(seed)
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad and param.numel() > 0:
                    std = param.std()
                    if std > 0:
                        noise = torch.randn_like(param) * noise_scale * std
                        param.add_(noise)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference.
        
        Since all models use the same input size, we use a single transform.
        
        Args:
            image: PIL Image in RGB format.
            
        Returns:
            Preprocessed tensor ready for inference.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        tensor = self._transform(image)
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

        # Preprocess once - same for all models
        input_tensor = self._preprocess(image)
        
        outputs = []
        
        for model, model_name in zip(self.models, self.model_names):
            # Synchronize for accurate timing on GPU
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # Model returns [batch, num_detections, 6]
                # Format: [x1, y1, x2, y2, score, class_id]
                detections = model(input_tensor)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Parse detections
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

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about each loaded model.
        
        Returns:
            List of dictionaries with model info.
        """
        return [
            {
                "name": name,
                "description": desc,
                "num_classes": num_cls,
                "architecture": MODEL_ARCH,
                "input_size": IMAGE_SIZE,
            }
            for name, desc, num_cls in zip(
                self.model_names, self.model_descriptions, self.model_num_classes
            )
        ]

    def get_backbone_weights(self) -> dict:
        """Get backbone weights (shared architecture) for potential stacking.
        
        The backbone (EfficientNet-B0 + BiFPN) has identical architecture
        across all models and can be stacked for vmap optimization.
        
        Returns:
            Dictionary mapping parameter names to stacked tensors [num_models, ...].
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        
        stacked = {}
        
        # Get backbone and bifpn parameters (exclude class_net and box_net heads)
        for name, param in self.models[0].model.named_parameters():
            # Only stack backbone and bifpn weights (same shape across models)
            if not name.startswith("class_net") and not name.startswith("box_net"):
                try:
                    params = [dict(m.model.named_parameters())[name] for m in self.models]
                    if all(p.shape == params[0].shape for p in params):
                        stacked[name] = torch.stack(params)
                except Exception:
                    pass  # Skip if shapes don't match
        
        return stacked

    @property
    def name(self) -> str:
        """Return the name of this implementation."""
        return "baseline_sequential"

    @property
    def num_models(self) -> int:
        """Return the number of models."""
        return len(CHECKPOINT_FILES)
    
    @property
    def architecture(self) -> str:
        """Return the model architecture name."""
        return MODEL_ARCH
    
    @property
    def input_size(self) -> int:
        """Return the input image size."""
        return IMAGE_SIZE
