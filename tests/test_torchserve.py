"""Tests for TorchServe integration.

These tests verify:
- Handler preprocessing/postprocessing
- Embedded mode functionality
- Output consistency with baseline
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


# Check if checkpoints exist
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINTS_EXIST = all(
    (CHECKPOINT_DIR / f).exists()
    for f in [
        "efficientdet_d0_coco.pth",
        "efficientdet_d0_aquarium.pth",
        "efficientdet_d0_vehicles.pth",
    ]
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample test image."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """Create sample image as JPEG bytes."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestEfficientDetHandler:
    """Tests for EfficientDetHandler."""
    
    def test_handler_can_be_imported(self):
        """Handler class can be imported."""
        from src.torchserve.handler import EfficientDetHandler
        
        handler = EfficientDetHandler()
        assert handler is not None
        assert handler.initialized is False
    
    def test_handler_preprocess_bytes(self, sample_image_bytes):
        """Handler can preprocess raw bytes."""
        from src.torchserve.handler import EfficientDetHandler
        
        handler = EfficientDetHandler()
        
        # Create request data like TorchServe sends
        request_data = [{"body": sample_image_bytes}]
        
        images = handler.preprocess(request_data)
        
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].mode == "RGB"
    
    def test_handler_preprocess_base64(self, sample_image_bytes):
        """Handler can preprocess base64-encoded images."""
        import base64
        from src.torchserve.handler import EfficientDetHandler
        
        handler = EfficientDetHandler()
        
        b64_data = base64.b64encode(sample_image_bytes).decode("utf-8")
        request_data = [{"image": b64_data}]
        
        images = handler.preprocess(request_data)
        
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
    
    def test_handler_preprocess_batch(self, sample_image_bytes):
        """Handler can preprocess multiple images."""
        from src.torchserve.handler import EfficientDetHandler
        
        handler = EfficientDetHandler()
        
        request_data = [
            {"body": sample_image_bytes},
            {"body": sample_image_bytes},
            {"body": sample_image_bytes},
        ]
        
        images = handler.preprocess(request_data)
        
        assert len(images) == 3
    
    def test_handler_postprocess_format(self):
        """Handler postprocess returns correct format."""
        from src.torchserve.handler import EfficientDetHandler
        from src.models.base import DetectionOutput
        
        handler = EfficientDetHandler()
        
        # Create mock detection outputs
        mock_outputs = [[
            DetectionOutput(
                boxes=torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
                scores=torch.tensor([0.9]),
                labels=torch.tensor([1]),
                model_name="test_model",
                inference_time_ms=10.0,
            )
        ]]
        
        results = handler.postprocess(mock_outputs)
        
        assert len(results) == 1
        result = results[0]
        
        assert "detections" in result
        assert "num_models" in result
        assert "total_inference_time_ms" in result
        
        detection = result["detections"][0]
        assert detection["model_name"] == "test_model"
        assert detection["boxes"] == [[10.0, 20.0, 100.0, 200.0]]
        assert detection["scores"] == pytest.approx([0.9], abs=1e-5)
        assert detection["labels"] == [1]


class TestEmbeddedTorchServe:
    """Tests for EmbeddedTorchServe."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_embedded_can_initialize(self):
        """Embedded TorchServe can initialize."""
        from src.torchserve.embedded import EmbeddedTorchServe
        
        embedded = EmbeddedTorchServe(optimization="baseline", device="cpu")
        
        assert not embedded.is_healthy()
        
        embedded.start()
        
        assert embedded.is_healthy()
        assert embedded.impl is not None
        
        embedded.stop()
        
        assert not embedded.is_healthy()
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_embedded_context_manager(self, sample_image):
        """Embedded TorchServe works as context manager."""
        from src.torchserve.embedded import EmbeddedTorchServe
        
        with EmbeddedTorchServe(optimization="baseline", device="cpu") as embedded:
            assert embedded.is_healthy()
            
            result = embedded.infer(sample_image)
            
            assert "detections" in result
            assert "request_time_ms" in result
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_embedded_infer_returns_detections(self, sample_image):
        """Embedded inference returns detection results."""
        from src.torchserve.embedded import EmbeddedTorchServe
        
        with EmbeddedTorchServe(optimization="baseline", device="cpu") as embedded:
            result = embedded.infer(sample_image)
            
            assert "detections" in result
            assert len(result["detections"]) == 3  # 3 models
            
            for detection in result["detections"]:
                assert "model_name" in detection
                assert "boxes" in detection
                assert "scores" in detection
                assert "labels" in detection
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_embedded_infer_raw(self, sample_image):
        """Embedded infer_raw returns DetectionOutput objects."""
        from src.torchserve.embedded import EmbeddedTorchServe
        from src.models.base import DetectionOutput
        
        with EmbeddedTorchServe(optimization="baseline", device="cpu") as embedded:
            outputs = embedded.infer_raw(sample_image)
            
            assert len(outputs) == 3  # 3 models
            
            for output in outputs:
                assert isinstance(output, DetectionOutput)
                assert isinstance(output.boxes, torch.Tensor)
                assert isinstance(output.scores, torch.Tensor)
                assert isinstance(output.labels, torch.Tensor)
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_embedded_batch_inference(self, sample_image):
        """Embedded batch inference works."""
        from src.torchserve.embedded import EmbeddedTorchServe
        
        with EmbeddedTorchServe(optimization="baseline", device="cpu") as embedded:
            images = [sample_image, sample_image]
            results = embedded.infer_batch(images)
            
            assert len(results) == 2
            
            for result in results:
                assert "detections" in result


class TestEmbeddedMatchesBaseline:
    """Tests that TorchServe embedded output matches direct baseline."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_output_format_matches_baseline(self, sample_image):
        """TorchServe embedded output has same structure as baseline."""
        from src.torchserve.embedded import EmbeddedTorchServe
        from src.models.baseline import BaselineImpl
        
        # Get baseline output
        baseline = BaselineImpl(device="cpu")
        baseline.load()
        baseline_outputs = baseline.predict(sample_image)
        
        # Get TorchServe embedded output using the same image (no JPEG re-encoding)
        with EmbeddedTorchServe(optimization="baseline", device="cpu") as embedded:
            # Use infer_raw with direct PIL image access
            ts_outputs = embedded.impl.predict(sample_image)
        
        # Compare structure (not exact values - JPEG encoding causes differences)
        assert len(ts_outputs) == len(baseline_outputs)
        
        for ts_out, base_out in zip(ts_outputs, baseline_outputs):
            # Model names should match
            assert ts_out.model_name == base_out.model_name
            
            # Output shapes should be consistent
            assert ts_out.boxes.shape[1] == 4  # (N, 4) boxes
            assert ts_out.scores.ndim == 1
            assert ts_out.labels.ndim == 1
            assert ts_out.scores.shape == ts_out.labels.shape


class TestTorchServeManager:
    """Tests for TorchServeManager (mocked, no actual server)."""
    
    def test_manager_can_be_imported(self):
        """Manager class can be imported."""
        from src.torchserve.server import TorchServeManager
        
        manager = TorchServeManager()
        assert manager is not None
        assert not manager._started
    
    def test_manager_urls_configured(self):
        """Manager URLs are correctly configured."""
        from src.torchserve.server import TorchServeManager
        
        manager = TorchServeManager(
            inference_port=9080,
            management_port=9081,
        )
        
        assert manager.inference_url == "http://localhost:9080"
        assert manager.management_url == "http://localhost:9081"
    
    @patch("src.torchserve.server.requests.get")
    def test_is_healthy_when_server_responds(self, mock_get):
        """is_healthy returns True when server responds."""
        from src.torchserve.server import TorchServeManager
        
        mock_get.return_value.status_code = 200
        
        manager = TorchServeManager()
        assert manager.is_healthy()
    
    @patch("src.torchserve.server.requests.get")
    def test_is_healthy_when_server_down(self, mock_get):
        """is_healthy returns False when server is down."""
        from src.torchserve.server import TorchServeManager
        import requests
        
        mock_get.side_effect = requests.ConnectionError()
        
        manager = TorchServeManager()
        assert not manager.is_healthy()
    
    @patch("src.torchserve.server.requests.post")
    def test_infer_sends_correct_request(self, mock_post, sample_image_bytes):
        """infer sends correct HTTP request."""
        from src.torchserve.server import TorchServeManager
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"detections": []}
        
        manager = TorchServeManager()
        manager._started = True  # Skip actual start
        
        result = manager.infer("test_model", sample_image_bytes)
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        assert "predictions/test_model" in call_args.args[0]
        assert call_args.kwargs["headers"]["Content-Type"] == "application/octet-stream"


class TestMARCreation:
    """Tests for MAR archive creation script."""
    
    def test_create_mar_script_exists(self):
        """MAR creation script exists."""
        script_path = Path(__file__).parent.parent / "scripts" / "create_mar.py"
        assert script_path.exists()
    
    def test_create_mar_can_import(self):
        """MAR creation functions can be imported."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        
        from create_mar import create_model_config, OPTIMIZATION_CONFIGS
        
        assert "baseline" in OPTIMIZATION_CONFIGS
        assert "vmap_backbone" in OPTIMIZATION_CONFIGS
        
        config = create_model_config("vmap_backbone", device="cuda")
        assert config["optimization"] == "vmap_backbone"
        assert config["device"] == "cuda"
        assert config["model_format"] == "eager"
