"""Dataset download utilities for COCO and Roboflow datasets."""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from tqdm import tqdm


# Dataset URLs
COCO_VAL2017_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ROBOFLOW_AQUARIUM_URL = "https://public.roboflow.com/ds/xKLV72TaXH?key=fk2C8aB2Ps"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        desc: Description for progress bar
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, output_path, reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total = len(zip_ref.namelist())
        for file in tqdm(zip_ref.namelist(), total=total, desc="Extracting"):
            zip_ref.extract(file, extract_to)


def download_coco_val2017(data_dir: Path, force: bool = False) -> Path:
    """Download COCO val2017 images.
    
    Args:
        data_dir: Base data directory
        force: If True, re-download even if exists
        
    Returns:
        Path to the extracted images directory
    """
    coco_dir = data_dir / "coco"
    images_dir = coco_dir / "val2017"
    zip_path = coco_dir / "val2017.zip"
    
    # Check if already downloaded
    if images_dir.exists() and not force:
        num_images = len(list(images_dir.glob("*.jpg")))
        if num_images > 0:
            print(f"COCO val2017 already exists with {num_images} images")
            return images_dir
    
    # Download
    print("Downloading COCO val2017 images (~1GB)...")
    download_file(COCO_VAL2017_IMAGES_URL, zip_path, desc="COCO val2017")
    
    # Extract
    print("Extracting COCO val2017...")
    extract_zip(zip_path, coco_dir)
    
    # Cleanup zip
    zip_path.unlink()
    
    num_images = len(list(images_dir.glob("*.jpg")))
    print(f"COCO val2017 ready: {num_images} images")
    
    return images_dir


def download_roboflow_aquarium(data_dir: Path, force: bool = False) -> Path:
    """Download Roboflow Aquarium dataset.
    
    Args:
        data_dir: Base data directory
        force: If True, re-download even if exists
        
    Returns:
        Path to the extracted dataset directory
    """
    roboflow_dir = data_dir / "roboflow"
    aquarium_dir = roboflow_dir / "aquarium"
    zip_path = roboflow_dir / "aquarium.zip"
    
    # Check if already downloaded
    if aquarium_dir.exists() and not force:
        # Check for images in any subdirectory
        image_count = sum(1 for _ in aquarium_dir.rglob("*.jpg"))
        if image_count > 0:
            print(f"Roboflow Aquarium already exists with {image_count} images")
            return aquarium_dir
    
    # Download
    print("Downloading Roboflow Aquarium dataset...")
    roboflow_dir.mkdir(parents=True, exist_ok=True)
    download_file(ROBOFLOW_AQUARIUM_URL, zip_path, desc="Aquarium")
    
    # Extract
    print("Extracting Roboflow Aquarium...")
    extract_zip(zip_path, aquarium_dir)
    
    # Cleanup zip
    zip_path.unlink()
    
    image_count = sum(1 for _ in aquarium_dir.rglob("*.jpg"))
    print(f"Roboflow Aquarium ready: {image_count} images")
    
    return aquarium_dir


def download_all_datasets(data_dir: Path, force: bool = False) -> dict:
    """Download all datasets.
    
    Args:
        data_dir: Base data directory
        force: If True, re-download even if exists
        
    Returns:
        Dictionary mapping dataset names to their paths
    """
    paths = {}
    
    print("=" * 60)
    print("Downloading datasets...")
    print("=" * 60)
    
    paths["coco_val2017"] = download_coco_val2017(data_dir, force)
    print()
    paths["roboflow_aquarium"] = download_roboflow_aquarium(data_dir, force)
    
    print()
    print("=" * 60)
    print("All datasets ready!")
    print("=" * 60)
    
    return paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store datasets"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--coco-only",
        action="store_true",
        help="Only download COCO dataset"
    )
    parser.add_argument(
        "--aquarium-only",
        action="store_true",
        help="Only download Aquarium dataset"
    )
    
    args = parser.parse_args()
    
    if args.coco_only:
        download_coco_val2017(args.data_dir, args.force)
    elif args.aquarium_only:
        download_roboflow_aquarium(args.data_dir, args.force)
    else:
        download_all_datasets(args.data_dir, args.force)
