"""Unit tests for InvalidImpl."""

import time

import pytest
import torch
from PIL import Image

from src.models.invalid import InvalidImpl


class TestInvalidImpl:
    """Tests for the InvalidImpl stub implementation."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        impl = InvalidImpl(device="cpu")
        
        assert impl.device.type == "cpu"
        assert impl.num_models == 3
        assert impl.num_dummy_detections == 100
        assert len(impl.model_names) == 3

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        impl = InvalidImpl(
            device="cpu",
            num_models=5,
            num_dummy_detections=50,
        )
        
        assert impl.num_models == 5
        assert impl.num_dummy_detections == 50
        assert len(impl.model_names) == 5

    def test_name_property(self):
        """Test name property."""
        impl = InvalidImpl(device="cpu")
        assert impl.name == "invalid_constant"

    def test_load_is_noop(self):
        """Test that load is a no-op but sets is_loaded."""
        impl = InvalidImpl(device="cpu")
        
        assert not impl.is_loaded
        impl.load()
        assert impl.is_loaded

    def test_predict_without_load_raises(self, sample_image):
        """Test that predict raises if not loaded."""
        impl = InvalidImpl(device="cpu")
        
        with pytest.raises(RuntimeError, match="Call load"):
            impl.predict(sample_image)

    def test_predict_returns_constant_outputs(self, sample_image):
        """Test that predict returns constant zero outputs."""
        impl = InvalidImpl(device="cpu", num_models=3, num_dummy_detections=100)
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        assert len(outputs) == 3
        
        for output in outputs:
            # Boxes should be all zeros
            assert output.boxes.shape == (100, 4)
            assert torch.all(output.boxes == 0)
            
            # Scores should be all zeros
            assert output.scores.shape == (100,)
            assert torch.all(output.scores == 0)
            
            # Labels should be all zeros
            assert output.labels.shape == (100,)
            assert torch.all(output.labels == 0)

    def test_predict_has_minimal_latency(self, sample_image):
        """Test that predict has minimal latency compared to real inference."""
        impl = InvalidImpl(device="cpu")
        impl.load()
        
        start = time.perf_counter()
        outputs = impl.predict(sample_image)
        total_time_ms = (time.perf_counter() - start) * 1000
        
        # Should be very fast (less than 100ms)
        assert total_time_ms < 100
        
        # Individual inference times should be very small
        for output in outputs:
            assert output.inference_time_ms < 10

    def test_predict_records_inference_time(self, sample_image):
        """Test that inference time is recorded."""
        impl = InvalidImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        for output in outputs:
            assert output.inference_time_ms >= 0

    def test_predict_sets_model_names(self, sample_image):
        """Test that model names are set correctly."""
        impl = InvalidImpl(device="cpu", num_models=3)
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        expected_names = ["invalid_model_0", "invalid_model_1", "invalid_model_2"]
        actual_names = [output.model_name for output in outputs]
        
        assert actual_names == expected_names

    def test_predict_batch(self, sample_image):
        """Test predict_batch processes all images."""
        impl = InvalidImpl(device="cpu", num_models=2)
        impl.load()
        
        images = [sample_image, sample_image, sample_image]
        outputs = impl.predict_batch(images)
        
        assert len(outputs) == 3
        for img_outputs in outputs:
            assert len(img_outputs) == 2

    def test_predict_with_different_image_modes(self):
        """Test predict handles different image modes."""
        impl = InvalidImpl(device="cpu")
        impl.load()
        
        # RGB
        rgb_image = Image.new("RGB", (512, 512))
        outputs_rgb = impl.predict(rgb_image)
        assert len(outputs_rgb) == 3
        
        # RGBA
        rgba_image = Image.new("RGBA", (512, 512))
        outputs_rgba = impl.predict(rgba_image)
        assert len(outputs_rgba) == 3
        
        # Grayscale
        gray_image = Image.new("L", (512, 512))
        outputs_gray = impl.predict(gray_image)
        assert len(outputs_gray) == 3

    def test_predict_with_different_image_sizes(self, sample_image):
        """Test predict handles different image sizes."""
        impl = InvalidImpl(device="cpu")
        impl.load()
        
        # Small image
        small_image = Image.new("RGB", (64, 64))
        outputs_small = impl.predict(small_image)
        assert len(outputs_small) == 3
        
        # Large image
        large_image = Image.new("RGB", (1920, 1080))
        outputs_large = impl.predict(large_image)
        assert len(outputs_large) == 3

    def test_outputs_are_on_correct_device(self, sample_image):
        """Test that outputs are on the correct device."""
        impl = InvalidImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        for output in outputs:
            assert output.boxes.device.type == "cpu"
            assert output.scores.device.type == "cpu"
            assert output.labels.device.type == "cpu"

    def test_outputs_have_correct_dtypes(self, sample_image):
        """Test that outputs have correct data types."""
        impl = InvalidImpl(device="cpu")
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        for output in outputs:
            assert output.boxes.dtype == torch.float32
            assert output.scores.dtype == torch.float32
            assert output.labels.dtype == torch.long

    def test_consistency_across_calls(self, sample_image):
        """Test that outputs are consistent across multiple calls."""
        impl = InvalidImpl(device="cpu", num_models=2, num_dummy_detections=10)
        impl.load()
        
        outputs1 = impl.predict(sample_image)
        outputs2 = impl.predict(sample_image)
        
        for o1, o2 in zip(outputs1, outputs2):
            assert torch.equal(o1.boxes, o2.boxes)
            assert torch.equal(o1.scores, o2.scores)
            assert torch.equal(o1.labels, o2.labels)

    def test_zero_dummy_detections(self, sample_image):
        """Test with zero dummy detections."""
        impl = InvalidImpl(device="cpu", num_models=1, num_dummy_detections=0)
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        assert len(outputs) == 1
        assert outputs[0].boxes.shape == (0, 4)
        assert outputs[0].scores.shape == (0,)
        assert outputs[0].labels.shape == (0,)

    def test_single_model(self, sample_image):
        """Test with single model configuration."""
        impl = InvalidImpl(device="cpu", num_models=1)
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        assert len(outputs) == 1
        assert outputs[0].model_name == "invalid_model_0"

    def test_many_models(self, sample_image):
        """Test with many models (simulating 15-model scenario)."""
        impl = InvalidImpl(device="cpu", num_models=15)
        impl.load()
        
        outputs = impl.predict(sample_image)
        
        assert len(outputs) == 15
        for i, output in enumerate(outputs):
            assert output.model_name == f"invalid_model_{i}"
