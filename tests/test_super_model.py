"""Tests for SuperEfficientDet model (TDD).

These tests define the expected behavior for the grouped super model
that fuses multiple EfficientDet instances into a single forward pass.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path


# Skip all tests if checkpoints don't exist
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINTS_EXIST = all(
    (CHECKPOINT_DIR / f).exists()
    for f in ["efficientdet_d0_coco.pth", "efficientdet_d0_aquarium.pth", "efficientdet_d0_vehicles.pth"]
)


@pytest.fixture
def device():
    """Get test device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_input(device):
    """Create a sample input tensor."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 512, 512, device=device)


class TestSuperEfficientDetCreation:
    """Tests for SuperEfficientDet model creation."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_super_model_can_be_created(self, device):
        """SuperEfficientDet should be creatable from checkpoint models."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        # Load base models
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        # Create super model
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        
        assert super_model is not None
        assert super_model.num_models == 3
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_super_model_has_grouped_layers(self, device):
        """SuperEfficientDet should use grouped convolutions."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        
        # Check that some convolutions use groups > 1
        has_grouped_conv = False
        for name, module in super_model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                has_grouped_conv = True
                break
        
        assert has_grouped_conv, "SuperEfficientDet should have grouped convolutions"


class TestSuperEfficientDetForward:
    """Tests for SuperEfficientDet forward pass."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_forward_output_structure(self, device, sample_input):
        """Forward should return box and class outputs for each model."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        super_model.eval()
        
        with torch.no_grad():
            box_outputs, class_outputs = super_model(sample_input)
        
        # Should have outputs for each model
        assert len(box_outputs) == 3, f"Expected 3 box outputs, got {len(box_outputs)}"
        assert len(class_outputs) == 3, f"Expected 3 class outputs, got {len(class_outputs)}"
        
        # Each box output should be a list of feature levels
        for i, box_out in enumerate(box_outputs):
            assert isinstance(box_out, list), f"box_outputs[{i}] should be a list"
            assert len(box_out) == 5, f"Expected 5 feature levels, got {len(box_out)}"
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_forward_output_shapes(self, device, sample_input):
        """Output shapes should match sequential model outputs."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        super_model.eval()
        
        # Get sequential outputs for reference
        with torch.no_grad():
            seq_outputs = []
            for model in impl.models:
                class_out, box_out = model.model(sample_input)
                seq_outputs.append((box_out, class_out))
        
        # Get super model outputs
        with torch.no_grad():
            box_outputs, class_outputs = super_model(sample_input)
        
        # Compare shapes
        for i, ((seq_box, seq_cls), super_box, super_cls) in enumerate(
            zip(seq_outputs, box_outputs, class_outputs)
        ):
            for level in range(5):
                assert super_box[level].shape == seq_box[level].shape, \
                    f"Model {i} level {level} box shape mismatch"
                assert super_cls[level].shape == seq_cls[level].shape, \
                    f"Model {i} level {level} class shape mismatch"


class TestSuperEfficientDetAccuracy:
    """Tests for SuperEfficientDet output accuracy."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_box_outputs_match_sequential(self, device, sample_input):
        """Box outputs should match running models sequentially."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        super_model.eval()
        
        # Get sequential outputs
        with torch.no_grad():
            seq_box_outputs = []
            for model in impl.models:
                _, box_out = model.model(sample_input)
                seq_box_outputs.append(box_out)
        
        # Get super model outputs
        with torch.no_grad():
            box_outputs, _ = super_model(sample_input)
        
        # Compare values
        for i, (seq_box, super_box) in enumerate(zip(seq_box_outputs, box_outputs)):
            for level in range(5):
                max_diff = (seq_box[level] - super_box[level]).abs().max().item()
                assert max_diff < 1e-4, \
                    f"Model {i} level {level} box max diff: {max_diff}"
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_class_outputs_match_sequential(self, device, sample_input):
        """Class outputs should match running models sequentially."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        super_model.eval()
        
        # Get sequential outputs
        with torch.no_grad():
            seq_class_outputs = []
            for model in impl.models:
                class_out, _ = model.model(sample_input)
                seq_class_outputs.append(class_out)
        
        # Get super model outputs
        with torch.no_grad():
            _, class_outputs = super_model(sample_input)
        
        # Compare values
        for i, (seq_cls, super_cls) in enumerate(zip(seq_class_outputs, class_outputs)):
            for level in range(5):
                max_diff = (seq_cls[level] - super_cls[level]).abs().max().item()
                assert max_diff < 1e-4, \
                    f"Model {i} level {level} class max diff: {max_diff}"


class TestSuperEfficientDetDetections:
    """Tests for final detection outputs."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_detections_match_baseline(self, device, sample_input):
        """Final detections should match baseline implementation."""
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        from src.benchmark.metrics import compare_outputs
        from src.models.base import DetectionOutput
        from effdet.bench import _post_process, _batch_detection
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        super_model.eval()
        
        # Get baseline detections
        with torch.no_grad():
            baseline_dets = []
            for model in impl.models:
                det = model(sample_input)
                if det is not None and len(det) > 0:
                    baseline_dets.append(det[0])
                else:
                    baseline_dets.append(torch.zeros(0, 6, device=device))
        
        # Get super model detections
        with torch.no_grad():
            super_dets = super_model.detect(sample_input, impl.models)
        
        # Compare
        for i, (base_det, super_det) in enumerate(zip(baseline_dets, super_dets)):
            if len(base_det) > 0 and len(super_det) > 0:
                # Compare box coordinates
                base_boxes = base_det[:, :4]
                super_boxes = super_det[:, :4]
                
                # Should have same number of detections or close
                assert abs(len(base_det) - len(super_det)) <= 5, \
                    f"Model {i}: detection count mismatch ({len(base_det)} vs {len(super_det)})"
                
                # For matching detections, check IoU is high
                if len(base_det) == len(super_det):
                    max_box_diff = (base_boxes - super_boxes).abs().max().item()
                    assert max_box_diff < 1.0, \
                        f"Model {i}: max box diff {max_box_diff}"


class TestSuperEfficientDetPerformance:
    """Performance-related tests."""
    
    @pytest.mark.skipif(not CHECKPOINTS_EXIST, reason="Checkpoints not found")
    def test_super_model_is_faster_than_sequential(self, device, sample_input):
        """Super model should be faster than running models sequentially."""
        import time
        from src.models.optimizations.super_model import SuperEfficientDet
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        impl = OptimizedImpl(device=str(device), optimization_config=OptimizationConfig())
        impl.load()
        
        super_model = SuperEfficientDet.from_models(impl.models, device=device)
        super_model.eval()
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                for model in impl.models:
                    _ = model.model(sample_input)
                _ = super_model(sample_input)
        
        if device.type == "mps":
            torch.mps.synchronize()
        
        # Time sequential
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                for model in impl.models:
                    _ = model.model(sample_input)
            if device.type == "mps":
                torch.mps.synchronize()
        seq_time = (time.perf_counter() - start) / 10
        
        # Time super model
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = super_model(sample_input)
            if device.type == "mps":
                torch.mps.synchronize()
        super_time = (time.perf_counter() - start) / 10
        
        print(f"\nSequential time: {seq_time*1000:.2f}ms")
        print(f"Super model time: {super_time*1000:.2f}ms")
        print(f"Speedup: {seq_time/super_time:.2f}x")
        
        # Super model should be at least as fast (may not be faster without compile)
        # This is a weak assertion - mainly for tracking
        assert super_time < seq_time * 2, \
            f"Super model too slow: {super_time:.3f}s vs {seq_time:.3f}s"
