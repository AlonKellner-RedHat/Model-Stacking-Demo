"""TorchServe integration for EfficientDet multi-model inference."""

from .handler import EfficientDetHandler
from .embedded import EmbeddedTorchServe
from .server import TorchServeManager

__all__ = [
    "EfficientDetHandler",
    "EmbeddedTorchServe",
    "TorchServeManager",
]
