"""Unit tests for BaselineImpl."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.models.baseline import BaselineImpl, MODEL_CONFIGS


class TestBaselineImplUnit:
    """Unit tests for BaselineImpl without loading real models."""

    def test_initialization(self):
        """Test BaselineImpl initialization."""
        impl = BaselineImpl(device="cpu")
        
        assert impl.device.type == "cpu"
        assert not impl.is_loaded
        assert impl.models == []
        assert impl.model_names == []
        assert impl.transforms == []

    def test_name_property(self):
        """Test name property."""
        impl = BaselineImpl(device="cpu")
        assert impl.name == "baseline_sequential"

    def test_num_models_property(self):
        """Test num_models property."""
        impl = BaselineImpl(device="cpu")
        assert impl.num_models == 3

    def test_model_configs(self):
        """Test that MODEL_CONFIGS has expected structure."""
        assert len(MODEL_CONFIGS) == 3
        
        for config in MODEL_CONFIGS:
            assert "name" in config
            assert "image_size" in config
            assert config["image_size"] > 0

    def test_predict_without_load_raises(self, sample_image):
        """Test that predict raises if models not loaded."""
        impl = BaselineImpl(device="cpu")
        
        with pytest.raises(RuntimeError, match="Models not loaded"):
            impl.predict(sample_image)

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_load_creates_models(self, mock_bench_class, mock_create_model):
        """Test that load creates the expected number of models."""
        # Setup mocks
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        assert impl.is_loaded
        assert len(impl.models) == 3
        assert len(impl.model_names) == 3
        assert len(impl.transforms) == 3
        
        # Verify create_model was called for each config
        assert mock_create_model.call_count == 3

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_load_idempotent(self, mock_bench_class, mock_create_model):
        """Test that calling load twice doesn't reload models."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        impl.load()  # Second call should be no-op
        
        # Should only be called 3 times total (not 6)
        assert mock_create_model.call_count == 3

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_predict_returns_list(
        self,
        mock_bench_class,
        mock_create_model,
        sample_image,
        mock_detection_output,
    ):
        """Test that predict returns a list of DetectionOutput."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = mock_detection_output
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        
        for output in outputs:
            assert hasattr(output, "boxes")
            assert hasattr(output, "scores")
            assert hasattr(output, "labels")
            assert hasattr(output, "model_name")
            assert hasattr(output, "inference_time_ms")

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_predict_batch_processes_all_images(
        self,
        mock_bench_class,
        mock_create_model,
        sample_image,
        mock_detection_output,
    ):
        """Test that predict_batch processes all images."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = mock_detection_output
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        images = [sample_image, sample_image, sample_image]
        outputs = impl.predict_batch(images)
        
        assert len(outputs) == 3
        for img_outputs in outputs:
            assert len(img_outputs) == 3  # 3 models per image

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_predict_handles_rgba_image(
        self,
        mock_bench_class,
        mock_create_model,
        mock_detection_output,
    ):
        """Test that predict handles RGBA images by converting to RGB."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = mock_detection_output
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        # Create RGBA image
        rgba_image = Image.new("RGBA", (512, 512), color=(255, 0, 0, 128))
        
        # Should not raise
        outputs = impl.predict(rgba_image)
        assert len(outputs) == 3

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_predict_handles_grayscale_image(
        self,
        mock_bench_class,
        mock_create_model,
        mock_detection_output,
    ):
        """Test that predict handles grayscale images."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = mock_detection_output
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        # Create grayscale image
        gray_image = Image.new("L", (512, 512), color=128)
        
        # Should convert to RGB and not raise
        outputs = impl.predict(gray_image)
        assert len(outputs) == 3

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_inference_time_recorded(
        self,
        mock_bench_class,
        mock_create_model,
        sample_image,
        mock_detection_output,
    ):
        """Test that inference time is recorded for each model."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = mock_detection_output
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        for output in outputs:
            assert output.inference_time_ms >= 0

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_model_names_set_correctly(
        self,
        mock_bench_class,
        mock_create_model,
        sample_image,
        mock_detection_output,
    ):
        """Test that model names are set correctly in outputs."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = mock_detection_output
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        expected_names = [config["name"] for config in MODEL_CONFIGS]
        actual_names = [output.model_name for output in outputs]
        
        assert actual_names == expected_names

    @patch("src.models.baseline.create_model")
    @patch("src.models.baseline.DetBenchPredict")
    def test_handles_empty_detections(
        self,
        mock_bench_class,
        mock_create_model,
        sample_image,
    ):
        """Test handling when model returns no detections."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        # Return empty detections
        empty_detection = torch.empty((1, 0, 6))
        
        mock_bench = MagicMock()
        mock_bench.eval.return_value = mock_bench
        mock_bench.to.return_value = mock_bench
        mock_bench.return_value = empty_detection
        mock_bench_class.return_value = mock_bench
        
        impl = BaselineImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        # Should handle gracefully
        assert len(outputs) == 3
        for output in outputs:
            assert output.boxes.shape[0] == 0
            assert output.scores.shape[0] == 0
            assert output.labels.shape[0] == 0
