#!/usr/bin/env python3
"""Create synthetic test images for benchmarking."""

import sys
from pathlib import Path

from PIL import Image
import numpy as np

def create_synthetic_images(output_dir: Path, num_images: int = 100):
    """Create synthetic test images for benchmarking.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to create
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} synthetic images in {output_dir}")
    
    for i in range(num_images):
        # Create varied images with different patterns
        size = (512, 512)
        
        # Create base image with gradient
        img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(size[1]):
            for x in range(size[0]):
                img_array[y, x, 0] = (x + i * 10) % 256  # R
                img_array[y, x, 1] = (y + i * 5) % 256   # G
                img_array[y, x, 2] = ((x + y) // 2 + i * 15) % 256  # B
        
        # Add some random rectangles to simulate objects
        np.random.seed(i)
        for _ in range(np.random.randint(3, 10)):
            x1 = np.random.randint(0, size[0] - 50)
            y1 = np.random.randint(0, size[1] - 50)
            x2 = x1 + np.random.randint(30, 150)
            y2 = y1 + np.random.randint(30, 150)
            x2 = min(x2, size[0])
            y2 = min(y2, size[1])
            
            color = (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )
            img_array[y1:y2, x1:x2] = color
        
        img = Image.fromarray(img_array)
        img.save(output_dir / f"synthetic_{i:04d}.jpg", format="JPEG", quality=85)
    
    print(f"Created {num_images} images")


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Create synthetic dataset
    synthetic_dir = data_dir / "synthetic" / "images"
    create_synthetic_images(synthetic_dir, num_images=100)
    
    print(f"\nSynthetic dataset ready at: {synthetic_dir}")


if __name__ == "__main__":
    main()
