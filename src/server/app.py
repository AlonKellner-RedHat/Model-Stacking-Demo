"""FastAPI server for inference endpoints."""

import io
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from src.models import BaselineImpl, InvalidImpl, BaseModelImpl, DetectionOutput


# Global model registry
_models: Dict[str, BaseModelImpl] = {}


class DetectionResult(BaseModel):
    """API response model for detection results."""
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]
    model_name: str
    inference_time_ms: float


class InferenceResponse(BaseModel):
    """API response for inference endpoint."""
    implementation: str
    total_inference_time_ms: float
    detections: List[DetectionResult]


class VRAMStats(BaseModel):
    """VRAM usage statistics."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    device: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    cuda_available: bool
    device: str


def load_models() -> None:
    """Load all model implementations."""
    global _models
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load baseline implementation
    baseline = BaselineImpl(device=device)
    baseline.load()
    _models["baseline"] = baseline
    
    # Load invalid implementation
    invalid = InvalidImpl(device=device)
    invalid.load()
    _models["invalid"] = invalid


def get_model(impl_name: str) -> BaseModelImpl:
    """Get a model implementation by name."""
    if impl_name not in _models:
        raise HTTPException(
            status_code=404,
            detail=f"Implementation '{impl_name}' not found. Available: {list(_models.keys())}"
        )
    return _models[impl_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading."""
    print("Loading models...")
    load_models()
    print(f"Models loaded: {list(_models.keys())}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="EfficientDet Inference Benchmark Server",
    description="Benchmarking server for multi-model EfficientDet inference",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health and model status."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy",
        models_loaded={name: model.is_loaded for name, model in _models.items()},
        cuda_available=torch.cuda.is_available(),
        device=device,
    )


@app.get("/vram", response_model=Optional[VRAMStats])
async def get_vram_stats() -> Optional[VRAMStats]:
    """Get current VRAM usage statistics."""
    if not torch.cuda.is_available():
        return None
    
    device = torch.device("cuda")
    return VRAMStats(
        allocated_mb=torch.cuda.memory_allocated(device) / 1024 / 1024,
        reserved_mb=torch.cuda.memory_reserved(device) / 1024 / 1024,
        max_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        device=str(device),
    )


@app.post("/vram/reset")
async def reset_vram_stats() -> Dict[str, str]:
    """Reset VRAM peak statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return {"status": "reset"}


@app.post("/infer/{impl_name}", response_model=InferenceResponse)
async def infer(
    impl_name: str,
    file: UploadFile = File(...),
) -> InferenceResponse:
    """Run inference using the specified implementation.
    
    Args:
        impl_name: Implementation name ('baseline' or 'invalid')
        file: Image file to run detection on
        
    Returns:
        Detection results from all models in the implementation
    """
    model = get_model(impl_name)
    
    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    # Run inference
    start_time = time.perf_counter()
    outputs: List[DetectionOutput] = model.predict(image)
    total_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Convert to response format
    detections = [
        DetectionResult(
            boxes=out.boxes.cpu().tolist(),
            scores=out.scores.cpu().tolist(),
            labels=[int(l) for l in out.labels.cpu().tolist()],
            model_name=out.model_name,
            inference_time_ms=out.inference_time_ms,
        )
        for out in outputs
    ]
    
    return InferenceResponse(
        implementation=impl_name,
        total_inference_time_ms=total_time_ms,
        detections=detections,
    )


@app.get("/implementations")
async def list_implementations() -> Dict[str, List[str]]:
    """List available implementations."""
    return {
        "implementations": list(_models.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
