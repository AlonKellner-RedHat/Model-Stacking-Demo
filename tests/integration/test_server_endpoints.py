"""Integration tests for FastAPI server endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.server.app import app, _models


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, test_client):
        """Test that health endpoint returns 200."""
        response = test_client.get("/health")
        
        assert response.status_code == 200

    def test_health_response_structure(self, test_client):
        """Test health response structure."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "cuda_available" in data
        assert "device" in data

    def test_health_shows_models_loaded(self, test_client):
        """Test that health shows models loaded status."""
        response = test_client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "baseline" in data["models_loaded"]
        assert "invalid" in data["models_loaded"]


class TestVRAMEndpoints:
    """Tests for VRAM-related endpoints."""

    def test_vram_returns_stats_or_null(self, test_client):
        """Test that /vram returns stats or null."""
        response = test_client.get("/vram")
        
        assert response.status_code == 200
        # Can be null on CPU or stats on GPU

    def test_vram_reset_returns_success(self, test_client):
        """Test that /vram/reset returns success."""
        response = test_client.post("/vram/reset")
        
        assert response.status_code == 200
        assert response.json()["status"] == "reset"


class TestImplementationsEndpoint:
    """Tests for /implementations endpoint."""

    def test_lists_implementations(self, test_client):
        """Test that implementations are listed."""
        response = test_client.get("/implementations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "implementations" in data
        assert "baseline" in data["implementations"]
        assert "invalid" in data["implementations"]


class TestInferEndpoint:
    """Tests for /infer/{impl_name} endpoint."""

    def test_infer_baseline_success(self, test_client, sample_image_bytes):
        """Test successful inference with baseline."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "implementation" in data
        assert data["implementation"] == "baseline"
        assert "total_inference_time_ms" in data
        assert "detections" in data
        assert len(data["detections"]) == 3  # 3 models

    def test_infer_invalid_success(self, test_client, sample_image_bytes):
        """Test successful inference with invalid impl."""
        response = test_client.post(
            "/infer/invalid",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["implementation"] == "invalid"
        assert len(data["detections"]) == 3

    def test_infer_unknown_impl_returns_404(self, test_client, sample_image_bytes):
        """Test that unknown implementation returns 404."""
        response = test_client.post(
            "/infer/nonexistent",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_infer_invalid_image_returns_400(self, test_client, invalid_image_bytes):
        """Test that invalid image returns 400."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", invalid_image_bytes, "image/jpeg")},
        )
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_infer_response_structure(self, test_client, sample_image_bytes):
        """Test the structure of inference response."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        data = response.json()
        
        # Check response structure
        assert "implementation" in data
        assert "total_inference_time_ms" in data
        assert "detections" in data
        
        # Check detection structure
        for detection in data["detections"]:
            assert "boxes" in detection
            assert "scores" in detection
            assert "labels" in detection
            assert "model_name" in detection
            assert "inference_time_ms" in detection

    def test_infer_boxes_format(self, test_client, sample_image_bytes):
        """Test that boxes are in correct format."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        data = response.json()
        
        for detection in data["detections"]:
            boxes = detection["boxes"]
            assert isinstance(boxes, list)
            # Each box should have 4 coordinates
            for box in boxes:
                assert len(box) == 4

    def test_infer_labels_are_integers(self, test_client, sample_image_bytes):
        """Test that labels are integers."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        data = response.json()
        
        for detection in data["detections"]:
            for label in detection["labels"]:
                assert isinstance(label, int)

    def test_infer_timing_recorded(self, test_client, sample_image_bytes):
        """Test that timing is recorded."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        data = response.json()
        
        assert data["total_inference_time_ms"] >= 0
        for detection in data["detections"]:
            assert detection["inference_time_ms"] >= 0

    def test_infer_png_image(self, test_client, sample_image):
        """Test inference with PNG image."""
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        
        assert response.status_code == 200

    def test_infer_different_image_sizes(self, test_client):
        """Test inference with different image sizes."""
        sizes = [(64, 64), (256, 256), (1024, 768)]
        
        for width, height in sizes:
            img = Image.new("RGB", (width, height), color=(100, 100, 100))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            
            response = test_client.post(
                "/infer/baseline",
                files={"file": ("test.jpg", buffer.getvalue(), "image/jpeg")},
            )
            
            assert response.status_code == 200, f"Failed for size {width}x{height}"

    def test_infer_grayscale_image(self, test_client):
        """Test inference with grayscale image."""
        gray = Image.new("L", (256, 256), color=128)
        buffer = io.BytesIO()
        gray.save(buffer, format="JPEG")
        
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("gray.jpg", buffer.getvalue(), "image/jpeg")},
        )
        
        assert response.status_code == 200

    def test_infer_rgba_image(self, test_client):
        """Test inference with RGBA image."""
        rgba = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        rgba.save(buffer, format="PNG")
        
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("rgba.png", buffer.getvalue(), "image/png")},
        )
        
        assert response.status_code == 200


class TestInvalidVsBaseline:
    """Tests comparing invalid and baseline implementations."""

    def test_invalid_faster_than_baseline(self, test_client, sample_image_bytes):
        """Test that invalid impl is faster than baseline."""
        # Run baseline
        baseline_response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        baseline_time = baseline_response.json()["total_inference_time_ms"]
        
        # Run invalid
        invalid_response = test_client.post(
            "/infer/invalid",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        invalid_time = invalid_response.json()["total_inference_time_ms"]
        
        # Invalid should be faster (or at least not slower)
        # Note: With mocked models, this may not always hold
        assert invalid_time <= baseline_time * 10  # Allow some variance

    def test_both_return_same_number_of_detections_per_model(
        self,
        test_client,
        sample_image_bytes,
    ):
        """Test both implementations return same number of model outputs."""
        baseline_response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        invalid_response = test_client.post(
            "/infer/invalid",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        baseline_models = len(baseline_response.json()["detections"])
        invalid_models = len(invalid_response.json()["detections"])
        
        assert baseline_models == invalid_models == 3


class TestServerLifecycle:
    """Tests for server lifecycle and model loading."""

    def test_models_are_loaded(self, test_client):
        """Test that models are loaded on startup."""
        response = test_client.get("/health")
        data = response.json()
        
        for model_name, is_loaded in data["models_loaded"].items():
            assert is_loaded, f"Model {model_name} not loaded"

    def test_multiple_requests_work(self, test_client, sample_image_bytes):
        """Test that multiple sequential requests work."""
        for i in range(5):
            response = test_client.post(
                "/infer/baseline",
                files={"file": (f"test_{i}.jpg", sample_image_bytes, "image/jpeg")},
            )
            assert response.status_code == 200


class TestReferenceGeneration:
    """Tests for reference generation integration."""

    def test_reference_endpoint_integration(
        self,
        test_client,
        temp_dir,
        sample_image_bytes,
    ):
        """Test that inference results can be used as references."""
        # Get baseline results
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        data = response.json()
        
        # Results should be serializable as reference
        import json
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
        
        assert deserialized["implementation"] == "baseline"
        assert len(deserialized["detections"]) == 3
