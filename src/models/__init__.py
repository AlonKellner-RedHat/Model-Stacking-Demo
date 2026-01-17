"""Model implementations for inference benchmarking."""

from .base import BaseModelImpl, DetectionOutput
from .baseline import BaselineImpl
from .invalid import InvalidImpl

__all__ = ["BaseModelImpl", "DetectionOutput", "BaselineImpl", "InvalidImpl"]
