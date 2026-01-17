"""Shared fixtures for test suite."""

import io
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Image Fixtures
# ============================================================================

@pytest.fixture
def sample_image() -> Image.Image:
    """Create a synthetic 512x512 RGB test image."""
    # Create a simple gradient image for testing
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for i in range(512):
        for j in range(512):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
    return img


@pytest.fixture
def sample_image_small() -> Image.Image:
    """Create a smaller 128x128 test image for faster tests."""
    return Image.new("RGB", (128, 128), color=(128, 128, 128))


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Convert sample image to JPEG bytes for HTTP requests."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytes_small(sample_image_small: Image.Image) -> bytes:
    """Convert small sample image to JPEG bytes."""
    buffer = io.BytesIO()
    sample_image_small.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@pytest.fixture
def invalid_image_bytes() -> bytes:
    """Return invalid bytes that cannot be decoded as an image."""
    return b"not a valid image file content"


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_data_dir(temp_dir: Path, sample_image: Image.Image) -> Path:
    """Create a temporary data directory with test images."""
    # Create COCO-like structure
    coco_dir = temp_dir / "coco" / "val2017"
    coco_dir.mkdir(parents=True)
    
    # Save some test images
    for i in range(5):
        img_path = coco_dir / f"test_image_{i:04d}.jpg"
        sample_image.save(img_path, format="JPEG")
    
    # Create Roboflow Aquarium-like structure
    aquarium_dir = temp_dir / "roboflow" / "aquarium" / "valid"
    aquarium_dir.mkdir(parents=True)
    
    for i in range(3):
        img_path = aquarium_dir / f"aquarium_{i:04d}.jpg"
        sample_image.save(img_path, format="JPEG")
    
    return temp_dir


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = temp_dir / "outputs"
    output_dir.mkdir(parents=True)
    return output_dir


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_detection_output() -> torch.Tensor:
    """Create a mock detection output tensor.
    
    Format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, score, class_id]
    """
    # Create 10 mock detections
    detections = torch.zeros(1, 10, 6)
    for i in range(10):
        # Random-ish bounding boxes
        x1, y1 = i * 50, i * 40
        x2, y2 = x1 + 100, y1 + 80
        score = 0.9 - i * 0.08
        class_id = i % 10
        detections[0, i] = torch.tensor([x1, y1, x2, y2, score, class_id])
    return detections


@pytest.fixture
def mock_effdet_model(mock_detection_output: torch.Tensor):
    """Create a mock EfficientDet model for testing without GPU."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.return_value = mock_detection_output
    mock_model.__call__ = MagicMock(return_value=mock_detection_output)
    return mock_model


@pytest.fixture
def mock_det_bench_predict(mock_detection_output: torch.Tensor):
    """Create a mock DetBenchPredict wrapper."""
    mock_bench = MagicMock()
    mock_bench.eval.return_value = mock_bench
    mock_bench.to.return_value = mock_bench
    mock_bench.__call__ = MagicMock(return_value=mock_detection_output)
    return mock_bench


# ============================================================================
# Detection Output Fixtures
# ============================================================================

@pytest.fixture
def sample_detection_output():
    """Create a sample DetectionOutput for testing."""
    from src.models.base import DetectionOutput
    
    return DetectionOutput(
        boxes=torch.tensor([[10, 20, 100, 120], [50, 60, 150, 180]]),
        scores=torch.tensor([0.95, 0.85]),
        labels=torch.tensor([1, 2]),
        model_name="test_model",
        inference_time_ms=15.5,
    )


@pytest.fixture
def sample_detection_output_zeros():
    """Create a zero-valued DetectionOutput for comparison testing."""
    from src.models.base import DetectionOutput
    
    return DetectionOutput(
        boxes=torch.zeros((2, 4)),
        scores=torch.zeros(2),
        labels=torch.zeros(2, dtype=torch.long),
        model_name="zero_model",
        inference_time_ms=0.1,
    )


@pytest.fixture
def sample_detection_output_empty():
    """Create an empty DetectionOutput."""
    from src.models.base import DetectionOutput
    
    return DetectionOutput(
        boxes=torch.empty((0, 4)),
        scores=torch.empty(0),
        labels=torch.empty(0, dtype=torch.long),
        model_name="empty_model",
        inference_time_ms=0.05,
    )


# ============================================================================
# Server Fixtures
# ============================================================================

@pytest.fixture
def mock_baseline_impl():
    """Create a mock BaselineImpl that doesn't require GPU."""
    from src.models.base import DetectionOutput
    
    mock_impl = MagicMock()
    mock_impl.name = "baseline_sequential"
    mock_impl.num_models = 3
    mock_impl.is_loaded = True
    mock_impl.device = torch.device("cpu")
    
    def mock_predict(image):
        outputs = []
        for i in range(3):
            outputs.append(DetectionOutput(
                boxes=torch.tensor([[10 + i * 10, 20, 100, 120]]),
                scores=torch.tensor([0.9 - i * 0.1]),
                labels=torch.tensor([i]),
                model_name=f"tf_efficientdet_d{i}",
                inference_time_ms=10.0 + i * 5,
            ))
        return outputs
    
    mock_impl.predict = mock_predict
    mock_impl.get_vram_usage.return_value = {}
    return mock_impl


@pytest.fixture
def mock_invalid_impl():
    """Create a mock InvalidImpl."""
    from src.models.base import DetectionOutput
    
    mock_impl = MagicMock()
    mock_impl.name = "invalid_constant"
    mock_impl.num_models = 3
    mock_impl.is_loaded = True
    mock_impl.device = torch.device("cpu")
    
    def mock_predict(image):
        outputs = []
        for i in range(3):
            outputs.append(DetectionOutput(
                boxes=torch.zeros((100, 4)),
                scores=torch.zeros(100),
                labels=torch.zeros(100, dtype=torch.long),
                model_name=f"invalid_model_{i}",
                inference_time_ms=0.1,
            ))
        return outputs
    
    mock_impl.predict = mock_predict
    mock_impl.get_vram_usage.return_value = {}
    return mock_impl


@pytest.fixture
def test_client(mock_baseline_impl, mock_invalid_impl):
    """Create a FastAPI TestClient with mocked models."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from src.server.app import app, _models, load_models
    
    # Patch load_models to inject mocked models instead of real ones
    def mock_load_models():
        _models.clear()
        _models["baseline"] = mock_baseline_impl
        _models["invalid"] = mock_invalid_impl
    
    with patch("src.server.app.load_models", mock_load_models):
        with TestClient(app) as client:
            yield client
    
    _models.clear()


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture
def sample_benchmark_result():
    """Create a sample BenchmarkResult for testing."""
    from src.benchmark.metrics import BenchmarkResult, ComparisonMetrics
    
    return BenchmarkResult(
        implementation="baseline",
        num_requests=100,
        duration_seconds=30.0,
        throughput_rps=3.33,
        latency_mean_ms=150.0,
        latency_p50_ms=140.0,
        latency_p90_ms=200.0,
        latency_p99_ms=350.0,
        vram_peak_mb=2048.0,
        comparison_metrics=ComparisonMetrics(
            boxes_mse=0.0,
            scores_mse=0.0,
            labels_accuracy=1.0,
            num_detections_diff=0,
            iou_mean=1.0,
        ),
        errors=0,
    )


@pytest.fixture
def sample_comparison_metrics():
    """Create sample ComparisonMetrics for testing."""
    from src.benchmark.metrics import ComparisonMetrics
    
    return ComparisonMetrics(
        boxes_mse=125.5,
        scores_mse=0.45,
        labels_accuracy=0.8,
        num_detections_diff=5,
        iou_mean=0.65,
    )


# ============================================================================
# Utility Functions
# ============================================================================

def create_test_images(directory: Path, count: int = 5, size: tuple = (256, 256)) -> List[Path]:
    """Helper to create multiple test images in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        img = Image.new("RGB", size, color=(i * 50 % 256, 100, 150))
        path = directory / f"image_{i:04d}.jpg"
        img.save(path, format="JPEG")
        paths.append(path)
    return paths
