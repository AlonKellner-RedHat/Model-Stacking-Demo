"""Grouped layer implementations for fusing multiple models.

These layers enable running N identical-architecture models as a single
"super model" using grouped convolutions and stacked batch normalization.

Key insight: With groups=N in Conv2d, each group processes independently,
allowing N models to share the same layer with no cross-talk.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedConv2d(nn.Module):
    """A Conv2d that fuses N separate Conv2d layers using groups.
    
    When created from N Conv2d layers with:
    - in_channels=C_in, out_channels=C_out
    
    The grouped version has:
    - in_channels=C_in*N, out_channels=C_out*N, groups=N
    
    This ensures each "model slot" processes independently.
    
    Example:
        convs = [nn.Conv2d(64, 128, 3) for _ in range(3)]
        grouped = GroupedConv2d.from_convs(convs)
        
        x = torch.randn(1, 64, 8, 8)
        x_grouped = x.repeat(1, 3, 1, 1)  # [1, 192, 8, 8]
        y = grouped(x_grouped)  # [1, 384, 8, 8]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        num_models: int = 1,
        original_groups: int = 1,
    ):
        """Initialize grouped conv.
        
        Args:
            in_channels: Input channels per model
            out_channels: Output channels per model
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            dilation: Convolution dilation
            groups: Base groups in original conv (e.g., 64 for depthwise)
            bias: Whether to use bias
            num_models: Number of models being fused
            original_groups: Groups in the original convs
        """
        super().__init__()
        
        self.num_models = num_models
        self.in_channels_per_model = in_channels
        self.out_channels_per_model = out_channels
        self.original_groups = original_groups
        
        # Total groups = original_groups * num_models
        # For standard conv (groups=1), total_groups = num_models
        # For depthwise conv (groups=in_channels), total_groups = in_channels * num_models
        total_groups = original_groups * num_models
        
        self.conv = nn.Conv2d(
            in_channels=in_channels * num_models,
            out_channels=out_channels * num_models,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=total_groups,
            bias=bias,
        )
    
    @classmethod
    def from_convs(cls, convs: List[nn.Conv2d]) -> "GroupedConv2d":
        """Create a GroupedConv2d from a list of Conv2d layers.
        
        Args:
            convs: List of N Conv2d layers with identical architecture
            
        Returns:
            GroupedConv2d that fuses all N convs
        """
        if len(convs) == 0:
            raise ValueError("Need at least one conv")
        
        # Verify all convs have same architecture
        ref = convs[0]
        for i, conv in enumerate(convs[1:], 1):
            assert conv.in_channels == ref.in_channels, \
                f"Conv {i} in_channels mismatch: {conv.in_channels} vs {ref.in_channels}"
            assert conv.out_channels == ref.out_channels, \
                f"Conv {i} out_channels mismatch"
            assert conv.kernel_size == ref.kernel_size, \
                f"Conv {i} kernel_size mismatch"
            assert conv.groups == ref.groups, \
                f"Conv {i} groups mismatch: {conv.groups} vs {ref.groups}"
        
        # Handle padding - could be int or tuple
        padding = ref.padding
        if isinstance(padding, tuple):
            padding = padding[0]  # Assume square padding
        
        kernel_size = ref.kernel_size
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]  # Assume square kernel
        
        # Create grouped conv
        grouped = cls(
            in_channels=ref.in_channels,
            out_channels=ref.out_channels,
            kernel_size=kernel_size,
            stride=ref.stride[0] if isinstance(ref.stride, tuple) else ref.stride,
            padding=padding,
            dilation=ref.dilation[0] if isinstance(ref.dilation, tuple) else ref.dilation,
            bias=ref.bias is not None,
            num_models=len(convs),
            original_groups=ref.groups,
        )
        
        # Stack weights from all convs
        # Weight shape for each conv: [out_channels, in_channels/groups, H, W]
        # For grouped, we concatenate along out_channels dimension
        with torch.no_grad():
            stacked_weight = torch.cat([conv.weight for conv in convs], dim=0)
            grouped.conv.weight.copy_(stacked_weight)
            
            if ref.bias is not None:
                stacked_bias = torch.cat([conv.bias for conv in convs], dim=0)
                grouped.conv.bias.copy_(stacked_bias)
        
        return grouped
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_channels * num_models, H, W]
            
        Returns:
            Output tensor [B, out_channels * num_models, H, W]
        """
        return self.conv(x)


class GroupedBatchNorm2d(nn.Module):
    """A BatchNorm2d that fuses N separate BatchNorm2d layers.
    
    Stacks the running statistics and affine parameters from N batchnorms
    into a single batchnorm with N× the features.
    """
    
    def __init__(
        self,
        num_features: int,
        num_models: int = 1,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        
        self.num_models = num_models
        self.num_features_per_model = num_features
        
        self.bn = nn.BatchNorm2d(
            num_features=num_features * num_models,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    
    @classmethod
    def from_batchnorms(cls, bns: List[nn.BatchNorm2d]) -> "GroupedBatchNorm2d":
        """Create a GroupedBatchNorm2d from a list of BatchNorm2d layers."""
        if len(bns) == 0:
            raise ValueError("Need at least one batchnorm")
        
        ref = bns[0]
        for i, bn in enumerate(bns[1:], 1):
            assert bn.num_features == ref.num_features, \
                f"BN {i} num_features mismatch"
        
        grouped = cls(
            num_features=ref.num_features,
            num_models=len(bns),
            eps=ref.eps,
            momentum=ref.momentum,
            affine=ref.affine,
            track_running_stats=ref.track_running_stats,
        )
        
        # Stack parameters and buffers
        with torch.no_grad():
            if ref.track_running_stats:
                grouped.bn.running_mean.copy_(
                    torch.cat([bn.running_mean for bn in bns])
                )
                grouped.bn.running_var.copy_(
                    torch.cat([bn.running_var for bn in bns])
                )
                grouped.bn.num_batches_tracked.copy_(bns[0].num_batches_tracked)
            
            if ref.affine:
                grouped.bn.weight.copy_(
                    torch.cat([bn.weight for bn in bns])
                )
                grouped.bn.bias.copy_(
                    torch.cat([bn.bias for bn in bns])
                )
        
        return grouped
    
    @property
    def running_mean(self) -> Optional[torch.Tensor]:
        return self.bn.running_mean
    
    @property
    def running_var(self) -> Optional[torch.Tensor]:
        return self.bn.running_var
    
    @property
    def weight(self) -> Optional[torch.Tensor]:
        return self.bn.weight
    
    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.bn.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
    
    def eval(self):
        self.bn.eval()
        return super().eval()
    
    def train(self, mode: bool = True):
        self.bn.train(mode)
        return super().train(mode)


class GroupedSeparableConv2d(nn.Module):
    """A separable conv (depthwise + pointwise) that fuses N instances.
    
    Used in EfficientNet and EfficientDet for efficient convolutions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        num_models: int = 1,
    ):
        super().__init__()
        
        self.num_models = num_models
        
        # Depthwise: groups = in_channels * num_models
        self.conv_dw = nn.Conv2d(
            in_channels=in_channels * num_models,
            out_channels=in_channels * num_models,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels * num_models,  # Depthwise for all models
            bias=False,
        )
        
        # Pointwise: groups = num_models (each model's pointwise is independent)
        self.conv_pw = nn.Conv2d(
            in_channels=in_channels * num_models,
            out_channels=out_channels * num_models,
            kernel_size=1,
            groups=num_models,
            bias=bias,
        )
    
    @classmethod
    def from_convs(
        cls,
        dw_convs: List[nn.Conv2d],
        pw_convs: List[nn.Conv2d],
    ) -> "GroupedSeparableConv2d":
        """Create from lists of depthwise and pointwise convs."""
        if len(dw_convs) != len(pw_convs):
            raise ValueError("Must have same number of DW and PW convs")
        if len(dw_convs) == 0:
            raise ValueError("Need at least one conv pair")
        
        ref_dw = dw_convs[0]
        ref_pw = pw_convs[0]
        
        padding = ref_dw.padding
        if isinstance(padding, tuple):
            padding = padding[0]
        
        kernel_size = ref_dw.kernel_size
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        
        grouped = cls(
            in_channels=ref_dw.in_channels,
            out_channels=ref_pw.out_channels,
            kernel_size=kernel_size,
            stride=ref_dw.stride[0] if isinstance(ref_dw.stride, tuple) else ref_dw.stride,
            padding=padding,
            bias=ref_pw.bias is not None,
            num_models=len(dw_convs),
        )
        
        # Stack weights
        with torch.no_grad():
            # Depthwise weights: [in_channels, 1, H, W] -> stack along dim 0
            grouped.conv_dw.weight.copy_(
                torch.cat([conv.weight for conv in dw_convs], dim=0)
            )
            
            # Pointwise weights: [out_channels, in_channels, 1, 1]
            # Need to stack properly for grouped conv
            grouped.conv_pw.weight.copy_(
                torch.cat([conv.weight for conv in pw_convs], dim=0)
            )
            
            if ref_pw.bias is not None:
                grouped.conv_pw.bias.copy_(
                    torch.cat([conv.bias for conv in pw_convs], dim=0)
                )
        
        return grouped
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x
