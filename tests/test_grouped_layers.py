"""Tests for grouped layer implementations (TDD).

These tests define the expected behavior for GroupedConv2d and GroupedBatchNorm2d,
which are used to fuse multiple models into a single "super model".
"""

import pytest
import torch
import torch.nn as nn


class TestGroupedConv2d:
    """Tests for GroupedConv2d layer wrapper."""
    
    def test_grouped_conv_output_shape(self):
        """Output should be [B, out_channels*N, H, W] for N models."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        # Create 3 individual convs
        convs = [
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
            for _ in range(3)
        ]
        
        # Create grouped conv from them
        grouped = GroupedConv2d.from_convs(convs)
        
        # Input for single model
        x = torch.randn(2, 64, 8, 8)
        
        # Grouped input (replicated)
        x_grouped = x.repeat(1, 3, 1, 1)  # [2, 192, 8, 8]
        
        # Forward pass
        y = grouped(x_grouped)
        
        # Output should be [B, 128*3, H, W]
        assert y.shape == (2, 128 * 3, 8, 8), f"Expected (2, 384, 8, 8), got {y.shape}"
    
    def test_grouped_conv_isolates_models(self):
        """Each group should process independently (no cross-talk between models)."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        # Create 3 convs with different constant weights
        convs = []
        for i in range(3):
            conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
            nn.init.constant_(conv.weight, float(i + 1))  # 1.0, 2.0, 3.0
            convs.append(conv)
        
        grouped = GroupedConv2d.from_convs(convs)
        
        # Create input where each model's channels have different values
        x = torch.ones(1, 64, 4, 4)
        x_grouped = torch.cat([
            x * 1.0,  # Model 1 input
            x * 10.0,  # Model 2 input
            x * 100.0,  # Model 3 input
        ], dim=1)
        
        y = grouped(x_grouped)
        
        # Extract outputs for each model
        y1 = y[:, :128]
        y2 = y[:, 128:256]
        y3 = y[:, 256:]
        
        # Each model's output should depend only on its own input and weights
        # Model 1: input=1, weight=1 -> distinct pattern
        # Model 2: input=10, weight=2 -> different pattern
        # Model 3: input=100, weight=3 -> different pattern
        
        # Check they're all different (no cross-talk)
        assert not torch.allclose(y1, y2), "Model 1 and 2 outputs should differ"
        assert not torch.allclose(y2, y3), "Model 2 and 3 outputs should differ"
        assert not torch.allclose(y1, y3), "Model 1 and 3 outputs should differ"
        
        # Verify the scaling relationship
        # y1 uses weight=1, input=1 -> base output
        # y2 uses weight=2, input=10 -> should be 20x y1 (2*10 vs 1*1)
        ratio_y2_y1 = (y2.mean() / y1.mean()).item()
        assert abs(ratio_y2_y1 - 20.0) < 0.1, f"Expected ratio ~20, got {ratio_y2_y1}"
    
    def test_grouped_conv_matches_sequential(self):
        """Output should exactly match running N convs sequentially."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        torch.manual_seed(42)
        
        # Create 3 convs with random weights
        convs = [
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
            for _ in range(3)
        ]
        
        # Initialize with different random weights
        for i, conv in enumerate(convs):
            torch.manual_seed(i * 100)
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        grouped = GroupedConv2d.from_convs(convs)
        
        # Random input
        torch.manual_seed(999)
        x = torch.randn(2, 64, 8, 8)
        
        # Sequential outputs
        y_seq = [conv(x) for conv in convs]
        
        # Grouped output
        x_grouped = x.repeat(1, 3, 1, 1)
        y_grouped = grouped(x_grouped)
        
        # Split grouped output
        y_grouped_split = [
            y_grouped[:, i*128:(i+1)*128]
            for i in range(3)
        ]
        
        # Compare
        for i, (y_s, y_g) in enumerate(zip(y_seq, y_grouped_split)):
            assert torch.allclose(y_s, y_g, atol=1e-5), \
                f"Model {i} output mismatch: max diff = {(y_s - y_g).abs().max()}"
    
    def test_grouped_conv_with_bias(self):
        """Should correctly handle biases."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        convs = [
            nn.Conv2d(32, 64, kernel_size=1, bias=True)
            for _ in range(3)
        ]
        
        # Set distinct biases
        for i, conv in enumerate(convs):
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, float(i + 1))  # 1, 2, 3
        
        grouped = GroupedConv2d.from_convs(convs)
        
        x = torch.zeros(1, 32, 4, 4)
        x_grouped = x.repeat(1, 3, 1, 1)
        
        y = grouped(x_grouped)
        
        # With zero input and zero weights, output should equal bias
        assert torch.allclose(y[:, :64], torch.ones(1, 64, 4, 4) * 1.0)
        assert torch.allclose(y[:, 64:128], torch.ones(1, 64, 4, 4) * 2.0)
        assert torch.allclose(y[:, 128:], torch.ones(1, 64, 4, 4) * 3.0)
    
    def test_grouped_conv_different_kernel_sizes(self):
        """Should work with various kernel sizes."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        for kernel_size in [1, 3, 5]:
            convs = [
                nn.Conv2d(32, 64, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
                for _ in range(3)
            ]
            
            grouped = GroupedConv2d.from_convs(convs)
            
            x = torch.randn(1, 32, 8, 8)
            x_grouped = x.repeat(1, 3, 1, 1)
            
            y = grouped(x_grouped)
            assert y.shape == (1, 64 * 3, 8, 8), f"Failed for kernel_size={kernel_size}"
    
    def test_grouped_conv_stride(self):
        """Should correctly handle stride."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        convs = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
            for _ in range(3)
        ]
        
        grouped = GroupedConv2d.from_convs(convs)
        
        x = torch.randn(1, 32, 8, 8)
        x_grouped = x.repeat(1, 3, 1, 1)
        
        y = grouped(x_grouped)
        
        # With stride=2, spatial dims should be halved
        assert y.shape == (1, 64 * 3, 4, 4)


class TestGroupedBatchNorm2d:
    """Tests for GroupedBatchNorm2d layer wrapper."""
    
    def test_grouped_bn_output_shape(self):
        """Output should be [B, features*N, H, W]."""
        from src.models.optimizations.grouped_layers import GroupedBatchNorm2d
        
        bns = [nn.BatchNorm2d(64) for _ in range(3)]
        grouped = GroupedBatchNorm2d.from_batchnorms(bns)
        
        x = torch.randn(2, 64 * 3, 8, 8)
        y = grouped(x)
        
        assert y.shape == (2, 64 * 3, 8, 8)
    
    def test_grouped_bn_running_stats(self):
        """Running mean/var should be stacked correctly."""
        from src.models.optimizations.grouped_layers import GroupedBatchNorm2d
        
        bns = []
        for i in range(3):
            bn = nn.BatchNorm2d(64)
            # Set distinct running stats
            bn.running_mean.fill_(float(i))
            bn.running_var.fill_(float(i + 1))
            bns.append(bn)
        
        grouped = GroupedBatchNorm2d.from_batchnorms(bns)
        
        # Check running_mean is stacked correctly
        assert grouped.running_mean[:64].mean().item() == pytest.approx(0.0, abs=1e-5)
        assert grouped.running_mean[64:128].mean().item() == pytest.approx(1.0, abs=1e-5)
        assert grouped.running_mean[128:].mean().item() == pytest.approx(2.0, abs=1e-5)
    
    def test_grouped_bn_matches_sequential(self):
        """Output should match running N batchnorms sequentially."""
        from src.models.optimizations.grouped_layers import GroupedBatchNorm2d
        
        torch.manual_seed(42)
        
        bns = [nn.BatchNorm2d(64) for _ in range(3)]
        
        # Set to eval mode for deterministic behavior
        for bn in bns:
            bn.eval()
        
        grouped = GroupedBatchNorm2d.from_batchnorms(bns)
        grouped.eval()
        
        # Random input
        x = torch.randn(2, 64, 8, 8)
        
        # Sequential outputs
        y_seq = [bn(x) for bn in bns]
        
        # Grouped output
        x_grouped = x.repeat(1, 3, 1, 1)
        y_grouped = grouped(x_grouped)
        
        # Split and compare
        for i, y_s in enumerate(y_seq):
            y_g = y_grouped[:, i*64:(i+1)*64]
            assert torch.allclose(y_s, y_g, atol=1e-5), \
                f"Model {i} BN output mismatch"
    
    def test_grouped_bn_affine_params(self):
        """Should correctly handle weight and bias (gamma/beta)."""
        from src.models.optimizations.grouped_layers import GroupedBatchNorm2d
        
        bns = []
        for i in range(3):
            bn = nn.BatchNorm2d(32, affine=True)
            nn.init.constant_(bn.weight, float(i + 1))  # gamma = 1, 2, 3
            nn.init.constant_(bn.bias, float(i * 10))   # beta = 0, 10, 20
            bn.eval()
            bns.append(bn)
        
        grouped = GroupedBatchNorm2d.from_batchnorms(bns)
        grouped.eval()
        
        # Check weight (gamma) is stacked
        assert grouped.weight[:32].mean().item() == pytest.approx(1.0, abs=1e-5)
        assert grouped.weight[32:64].mean().item() == pytest.approx(2.0, abs=1e-5)
        assert grouped.weight[64:].mean().item() == pytest.approx(3.0, abs=1e-5)
        
        # Check bias (beta) is stacked
        assert grouped.bias[:32].mean().item() == pytest.approx(0.0, abs=1e-5)
        assert grouped.bias[32:64].mean().item() == pytest.approx(10.0, abs=1e-5)
        assert grouped.bias[64:].mean().item() == pytest.approx(20.0, abs=1e-5)


class TestGroupedDepthwiseConv2d:
    """Tests for depthwise separable convolutions (used in EfficientNet)."""
    
    def test_grouped_depthwise_conv(self):
        """Depthwise conv (groups=in_channels) should work when grouped."""
        from src.models.optimizations.grouped_layers import GroupedConv2d
        
        # Depthwise conv: groups = in_channels
        convs = [
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
            for _ in range(3)
        ]
        
        grouped = GroupedConv2d.from_convs(convs)
        
        x = torch.randn(1, 64, 8, 8)
        x_grouped = x.repeat(1, 3, 1, 1)
        
        y = grouped(x_grouped)
        
        assert y.shape == (1, 64 * 3, 8, 8)
        
        # Verify matches sequential
        y_seq = torch.cat([conv(x) for conv in convs], dim=1)
        assert torch.allclose(y, y_seq, atol=1e-5)


class TestGroupedSeparableConv2d:
    """Tests for separable convolutions (depthwise + pointwise)."""
    
    def test_grouped_separable_conv(self):
        """Separable conv (DW + PW) should work when grouped."""
        from src.models.optimizations.grouped_layers import GroupedSeparableConv2d
        
        # Create 3 separable conv pairs (depthwise + pointwise)
        dw_convs = [
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
            for _ in range(3)
        ]
        pw_convs = [
            nn.Conv2d(64, 128, kernel_size=1, bias=True)
            for _ in range(3)
        ]
        
        grouped = GroupedSeparableConv2d.from_convs(dw_convs, pw_convs)
        
        x = torch.randn(1, 64, 8, 8)
        x_grouped = x.repeat(1, 3, 1, 1)
        
        y = grouped(x_grouped)
        
        assert y.shape == (1, 128 * 3, 8, 8)
        
        # Verify matches sequential
        y_seq = []
        for dw, pw in zip(dw_convs, pw_convs):
            y_seq.append(pw(dw(x)))
        y_seq = torch.cat(y_seq, dim=1)
        
        assert torch.allclose(y, y_seq, atol=1e-5)
