"""Generate and store reference outputs from baseline implementation."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets import DatasetLoader
from src.models import BaselineImpl, DetectionOutput


def generate_reference_outputs(
    data_dir: Path,
    output_dir: Path,
    max_images: int = 100,
    device: Optional[str] = None,
) -> Dict[str, Path]:
    """Generate reference outputs from baseline implementation.
    
    Args:
        data_dir: Directory containing datasets
        output_dir: Directory to store reference outputs
        max_images: Maximum number of images to process
        device: Device to run inference on
        
    Returns:
        Dictionary mapping image IDs to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load baseline model
    print("Loading baseline model...")
    baseline = BaselineImpl(device=device)
    baseline.load()
    print(f"Loaded {baseline.num_models} models on {baseline.device}")
    
    # Load datasets
    print("Loading datasets...")
    loader = DatasetLoader(data_dir)
    counts = loader.add_all_datasets(max_images_per_dataset=max_images)
    print(f"Loaded datasets: {counts}")
    
    if len(loader) == 0:
        raise RuntimeError("No images found. Run download_datasets.py first.")
    
    # Generate references
    print(f"Generating reference outputs for {len(loader)} images...")
    
    references = {}
    manifest = []
    
    for item in tqdm(loader, desc="Generating references"):
        # Run inference
        outputs: List[DetectionOutput] = baseline.predict(item.image)
        
        # Create unique ID
        ref_id = f"{item.dataset_name}_{item.image_id}"
        
        # Save outputs
        ref_data = {
            "image_id": item.image_id,
            "dataset_name": item.dataset_name,
            "image_path": str(item.path),
            "outputs": [out.to_dict() for out in outputs],
        }
        
        ref_file = output_dir / f"{ref_id}.json"
        with open(ref_file, "w") as f:
            json.dump(ref_data, f)
        
        references[ref_id] = ref_file
        manifest.append({
            "ref_id": ref_id,
            "image_id": item.image_id,
            "dataset_name": item.dataset_name,
            "image_path": str(item.path),
            "ref_file": str(ref_file),
        })
    
    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "num_references": len(manifest),
            "model_name": baseline.name,
            "num_models": baseline.num_models,
            "references": manifest,
        }, f, indent=2)
    
    print(f"Generated {len(references)} reference outputs")
    print(f"Manifest saved to: {manifest_file}")
    
    return references


def load_reference(ref_file: Path) -> dict:
    """Load a reference output file.
    
    Args:
        ref_file: Path to reference JSON file
        
    Returns:
        Dictionary with reference data
    """
    with open(ref_file, "r") as f:
        return json.load(f)


def load_manifest(output_dir: Path) -> dict:
    """Load the reference manifest.
    
    Args:
        output_dir: Directory containing reference outputs
        
    Returns:
        Manifest dictionary
    """
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "r") as f:
        return json.load(f)


def main():
    """Generate reference outputs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reference outputs")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reference"),
        help="Directory to store reference outputs"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    generate_reference_outputs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        device=args.device,
    )


if __name__ == "__main__":
    main()
