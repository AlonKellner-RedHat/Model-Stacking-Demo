"""Unit tests for DetectionOutput and BaseModelImpl."""

import pytest
import torch

from src.models.base import BaseModelImpl, DetectionOutput


class TestDetectionOutput:
    """Tests for the DetectionOutput dataclass."""

    def test_creation(self):
        """Test basic DetectionOutput creation."""
        output = DetectionOutput(
            boxes=torch.tensor([[10, 20, 100, 120]]),
            scores=torch.tensor([0.95]),
            labels=torch.tensor([1]),
            model_name="test_model",
            inference_time_ms=10.5,
        )
        
        assert output.model_name == "test_model"
        assert output.inference_time_ms == 10.5
        assert output.boxes.shape == (1, 4)
        assert output.scores.shape == (1,)
        assert output.labels.shape == (1,)

    def test_to_dict(self, sample_detection_output):
        """Test serialization to dictionary."""
        result = sample_detection_output.to_dict()
        
        assert isinstance(result, dict)
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert "model_name" in result
        assert "inference_time_ms" in result
        
        # Check values are converted to lists
        assert isinstance(result["boxes"], list)
        assert isinstance(result["scores"], list)
        assert isinstance(result["labels"], list)
        assert result["model_name"] == "test_model"
        assert result["inference_time_ms"] == 15.5

    def test_to_dict_values(self, sample_detection_output):
        """Test that to_dict preserves values correctly."""
        result = sample_detection_output.to_dict()
        
        # Check box values
        assert len(result["boxes"]) == 2
        assert result["boxes"][0] == [10.0, 20.0, 100.0, 120.0]
        
        # Check scores
        assert len(result["scores"]) == 2
        assert abs(result["scores"][0] - 0.95) < 0.01
        
        # Check labels
        assert result["labels"] == [1, 2]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "boxes": [[10, 20, 100, 120], [50, 60, 150, 180]],
            "scores": [0.95, 0.85],
            "labels": [1, 2],
            "model_name": "restored_model",
            "inference_time_ms": 25.0,
        }
        
        output = DetectionOutput.from_dict(data)
        
        assert output.model_name == "restored_model"
        assert output.inference_time_ms == 25.0
        assert output.boxes.shape == (2, 4)
        assert output.scores.shape == (2,)
        assert output.labels.shape == (2,)

    def test_from_dict_device(self):
        """Test from_dict with specified device."""
        data = {
            "boxes": [[10, 20, 100, 120]],
            "scores": [0.95],
            "labels": [1],
            "model_name": "test",
            "inference_time_ms": 10.0,
        }
        
        output = DetectionOutput.from_dict(data, device="cpu")
        
        assert output.boxes.device.type == "cpu"
        assert output.scores.device.type == "cpu"
        assert output.labels.device.type == "cpu"

    def test_roundtrip(self, sample_detection_output):
        """Test that to_dict -> from_dict preserves data."""
        original = sample_detection_output
        data = original.to_dict()
        restored = DetectionOutput.from_dict(data)
        
        assert restored.model_name == original.model_name
        assert restored.inference_time_ms == original.inference_time_ms
        assert torch.allclose(restored.boxes, original.boxes.cpu())
        assert torch.allclose(restored.scores, original.scores.cpu())
        assert torch.equal(restored.labels, original.labels.cpu())

    def test_empty_output(self, sample_detection_output_empty):
        """Test handling of empty detection output."""
        output = sample_detection_output_empty
        
        assert output.boxes.shape == (0, 4)
        assert output.scores.shape == (0,)
        assert output.labels.shape == (0,)
        
        # Test serialization of empty output
        result = output.to_dict()
        assert result["boxes"] == []
        assert result["scores"] == []
        assert result["labels"] == []


class TestBaseModelImpl:
    """Tests for the BaseModelImpl abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseModelImpl cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelImpl()

    def test_device_auto_detection(self):
        """Test that device is auto-detected."""
        # Create a concrete subclass for testing
        class ConcreteImpl(BaseModelImpl):
            def load(self):
                self._is_loaded = True
            
            def predict(self, image):
                return []
            
            def predict_batch(self, images):
                return [self.predict(img) for img in images]
            
            @property
            def name(self):
                return "concrete"
            
            @property
            def num_models(self):
                return 1
        
        impl = ConcreteImpl()
        # Should be either cuda or cpu
        assert impl.device.type in ("cuda", "cpu")

    def test_explicit_device(self):
        """Test setting explicit device."""
        class ConcreteImpl(BaseModelImpl):
            def load(self):
                self._is_loaded = True
            
            def predict(self, image):
                return []
            
            def predict_batch(self, images):
                return []
            
            @property
            def name(self):
                return "concrete"
            
            @property
            def num_models(self):
                return 1
        
        impl = ConcreteImpl(device="cpu")
        assert impl.device.type == "cpu"

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        class ConcreteImpl(BaseModelImpl):
            def load(self):
                self._is_loaded = True
            
            def predict(self, image):
                return []
            
            def predict_batch(self, images):
                return []
            
            @property
            def name(self):
                return "concrete"
            
            @property
            def num_models(self):
                return 1
        
        impl = ConcreteImpl(device="cpu")
        assert not impl.is_loaded
        
        impl.load()
        assert impl.is_loaded

    def test_get_vram_usage_cpu(self):
        """Test VRAM usage returns empty dict on CPU."""
        class ConcreteImpl(BaseModelImpl):
            def load(self):
                pass
            
            def predict(self, image):
                return []
            
            def predict_batch(self, images):
                return []
            
            @property
            def name(self):
                return "concrete"
            
            @property
            def num_models(self):
                return 1
        
        impl = ConcreteImpl(device="cpu")
        vram = impl.get_vram_usage()
        
        assert vram == {}

    def test_reset_vram_stats_cpu(self):
        """Test reset_vram_stats doesn't error on CPU."""
        class ConcreteImpl(BaseModelImpl):
            def load(self):
                pass
            
            def predict(self, image):
                return []
            
            def predict_batch(self, images):
                return []
            
            @property
            def name(self):
                return "concrete"
            
            @property
            def num_models(self):
                return 1
        
        impl = ConcreteImpl(device="cpu")
        # Should not raise
        impl.reset_vram_stats()

    def test_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""
        # Missing predict_batch
        with pytest.raises(TypeError):
            class IncompleteImpl(BaseModelImpl):
                def load(self):
                    pass
                
                def predict(self, image):
                    return []
                
                @property
                def name(self):
                    return "incomplete"
                
                @property
                def num_models(self):
                    return 1
            
            IncompleteImpl()
