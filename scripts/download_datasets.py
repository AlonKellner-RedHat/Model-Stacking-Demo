#!/usr/bin/env python3
"""Download benchmark datasets (COCO val2017 and Roboflow Aquarium)."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.download import download_all_datasets


def main():
    """Download all datasets to the data directory."""
    # Default data directory is project_root/data
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    print(f"Downloading datasets to: {data_dir}")
    print()
    
    paths = download_all_datasets(data_dir)
    
    print()
    print("Dataset paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
