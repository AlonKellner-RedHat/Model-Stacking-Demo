"""Locust load test definition for inference benchmarking."""

import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional

from locust import HttpUser, between, events, task
from locust.env import Environment


# Configuration from environment variables
DATA_DIR = Path(os.environ.get("BENCHMARK_DATA_DIR", "data"))
DATASET = os.environ.get("BENCHMARK_DATASET", "coco_val2017")  # or "roboflow_aquarium"
MAX_IMAGES = int(os.environ.get("BENCHMARK_MAX_IMAGES", "100"))
IMPLEMENTATION = os.environ.get("BENCHMARK_IMPLEMENTATION", "baseline")  # or "invalid"


class ImagePool:
    """Pool of images for load testing."""
    
    _instance: Optional["ImagePool"] = None
    _images: List[tuple] = []  # List of (filename, bytes)
    
    @classmethod
    def get_instance(cls) -> "ImagePool":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_images()
        return cls._instance
    
    def _load_images(self) -> None:
        """Load images from the configured dataset."""
        if DATASET == "coco_val2017":
            image_dir = DATA_DIR / "coco" / "val2017"
        elif DATASET == "roboflow_aquarium":
            image_dir = DATA_DIR / "roboflow" / "aquarium"
            # Search recursively for aquarium
            image_files = list(image_dir.rglob("*.jpg"))[:MAX_IMAGES]
            for img_path in image_files:
                with open(img_path, "rb") as f:
                    self._images.append((img_path.name, f.read()))
            print(f"Loaded {len(self._images)} images from {DATASET}")
            return
        else:
            image_dir = DATA_DIR / DATASET
        
        if not image_dir.exists():
            print(f"Warning: Dataset directory not found: {image_dir}")
            print("Using synthetic test image")
            # Create a minimal valid JPEG for testing
            self._create_synthetic_images()
            return
        
        image_files = sorted(image_dir.glob("*.jpg"))[:MAX_IMAGES]
        for img_path in image_files:
            with open(img_path, "rb") as f:
                self._images.append((img_path.name, f.read()))
        
        print(f"Loaded {len(self._images)} images from {DATASET}")
    
    def _create_synthetic_images(self) -> None:
        """Create synthetic test images if no dataset available."""
        from io import BytesIO
        try:
            from PIL import Image
            # Create a simple test image
            img = Image.new("RGB", (512, 512), color=(128, 128, 128))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            self._images.append(("synthetic.jpg", buffer.getvalue()))
        except ImportError:
            # Minimal valid JPEG if PIL not available
            # This is a 1x1 red pixel JPEG
            minimal_jpeg = bytes([
                0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
                0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
                0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
                0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
                0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
                0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
                0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
                0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
                0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
                0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
                0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
                0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
                0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
                0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
                0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
                0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
                0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
                0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
                0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
                0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
                0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
                0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
                0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x45, 0x10,
                0x28, 0xA2, 0x80, 0x28, 0xA2, 0x80, 0xFF, 0xD9
            ])
            self._images.append(("synthetic.jpg", minimal_jpeg))
    
    def get_random_image(self) -> tuple:
        """Get a random image (filename, bytes)."""
        return random.choice(self._images)


class InferenceBenchmarkUser(HttpUser):
    """Locust user for inference benchmarking."""
    
    # Wait between 0.1 and 0.5 seconds between requests
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Initialize the user."""
        self.image_pool = ImagePool.get_instance()
        self.implementation = IMPLEMENTATION
        
        # Warmup request
        filename, image_bytes = self.image_pool.get_random_image()
        self.client.post(
            f"/infer/{self.implementation}",
            files={"file": (filename, image_bytes, "image/jpeg")},
            name=f"/infer/{self.implementation} (warmup)",
        )
    
    @task(10)
    def infer_baseline(self):
        """Run inference on the baseline implementation."""
        if self.implementation != "baseline":
            return
        
        filename, image_bytes = self.image_pool.get_random_image()
        
        with self.client.post(
            "/infer/baseline",
            files={"file": (filename, image_bytes, "image/jpeg")},
            name="/infer/baseline",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Optionally validate response
                if "detections" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(10)
    def infer_invalid(self):
        """Run inference on the invalid implementation."""
        if self.implementation != "invalid":
            return
        
        filename, image_bytes = self.image_pool.get_random_image()
        
        with self.client.post(
            "/infer/invalid",
            files={"file": (filename, image_bytes, "image/jpeg")},
            name="/infer/invalid",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "detections" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def check_health(self):
        """Check server health."""
        self.client.get("/health", name="/health")
    
    @task(1)
    def check_vram(self):
        """Check VRAM usage."""
        self.client.get("/vram", name="/vram")


class ComparativeUser(HttpUser):
    """User that tests both implementations for comparison."""
    
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Initialize the user."""
        self.image_pool = ImagePool.get_instance()
    
    @task(5)
    def infer_baseline(self):
        """Run inference on baseline."""
        filename, image_bytes = self.image_pool.get_random_image()
        self.client.post(
            "/infer/baseline",
            files={"file": (filename, image_bytes, "image/jpeg")},
            name="/infer/baseline",
        )
    
    @task(5)
    def infer_invalid(self):
        """Run inference on invalid."""
        filename, image_bytes = self.image_pool.get_random_image()
        self.client.post(
            "/infer/invalid",
            files={"file": (filename, image_bytes, "image/jpeg")},
            name="/infer/invalid",
        )


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log request metrics."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("Inference Benchmark Load Test Starting")
    print(f"  Dataset: {DATASET}")
    print(f"  Implementation: {IMPLEMENTATION}")
    print(f"  Max Images: {MAX_IMAGES}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Called when test stops."""
    print("=" * 60)
    print("Inference Benchmark Load Test Complete")
    print("=" * 60)
