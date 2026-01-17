#!/usr/bin/env python3
"""Create TorchServe Model Archive (.mar) files.

This script packages the EfficientDet handler with model checkpoints
into a .mar file that can be deployed to TorchServe.

Usage:
    uv run python scripts/create_mar.py [--optimization OPTIMIZATION] [--output-dir DIR]
    
Examples:
    # Create baseline MAR
    uv run python scripts/create_mar.py --optimization baseline
    
    # Create vmap_backbone optimized MAR
    uv run python scripts/create_mar.py --optimization vmap_backbone
    
    # Create all configurations
    uv run python scripts/create_mar.py --all
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# Available optimization configurations
OPTIMIZATION_CONFIGS = [
    "baseline",
    "compile",
    "vmap_backbone",
    "grouped_super_model",
]

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
HANDLER_PATH = PROJECT_ROOT / "src" / "torchserve" / "handler.py"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_STORE_DIR = PROJECT_ROOT / "model_store"
SRC_DIR = PROJECT_ROOT / "src"


def create_model_config(optimization: str, device: str = "auto") -> dict:
    """Create model configuration dictionary."""
    return {
        "optimization": optimization,
        "device": device,
        "model_format": "eager",
    }


def create_mar(
    optimization: str = "baseline",
    output_dir: Path = MODEL_STORE_DIR,
    version: str = "1.0",
    force: bool = False,
) -> Path:
    """Create a TorchServe Model Archive (.mar) file.
    
    Args:
        optimization: Optimization configuration name
        output_dir: Directory to save the .mar file
        version: Model version string
        force: Overwrite existing .mar file
        
    Returns:
        Path to the created .mar file
    """
    model_name = f"efficientdet_{optimization}"
    mar_path = output_dir / f"{model_name}.mar"
    
    # Check if already exists
    if mar_path.exists() and not force:
        print(f"MAR already exists: {mar_path}")
        print("Use --force to overwrite")
        return mar_path
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for packaging
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create model-config.yaml
        config = create_model_config(optimization)
        config_path = tmpdir / "model-config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Collect extra files:
        # - model-config.yaml
        # - src/ directory (for imports)
        # - checkpoints/ (model weights)
        extra_files = [str(config_path)]
        
        # Copy src directory to temp (TorchServe needs it as extra-files)
        src_copy = tmpdir / "src"
        shutil.copytree(SRC_DIR, src_copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        extra_files.append(str(src_copy))
        
        # Copy checkpoints
        if CHECKPOINTS_DIR.exists():
            ckpt_copy = tmpdir / "checkpoints"
            shutil.copytree(CHECKPOINTS_DIR, ckpt_copy)
            extra_files.append(str(ckpt_copy))
        else:
            print(f"Warning: Checkpoints directory not found: {CHECKPOINTS_DIR}")
            print("The MAR will be created but models won't load without checkpoints.")
        
        # Build torch-model-archiver command
        cmd = [
            sys.executable, "-m", "torch_model_archiver",
            "--model-name", model_name,
            "--version", version,
            "--handler", str(HANDLER_PATH),
            "--extra-files", ",".join(extra_files),
            "--export-path", str(output_dir),
            "--force",  # Always force in archiver, we check above
        ]
        
        print(f"Creating MAR: {model_name}")
        print(f"  Optimization: {optimization}")
        print(f"  Handler: {HANDLER_PATH}")
        print(f"  Output: {mar_path}")
        
        # Run archiver
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"  Created: {mar_path}")
            return mar_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating MAR: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise


def create_all_mars(output_dir: Path = MODEL_STORE_DIR, force: bool = False) -> list:
    """Create MAR files for all optimization configurations."""
    created = []
    
    for optimization in OPTIMIZATION_CONFIGS:
        try:
            mar_path = create_mar(
                optimization=optimization,
                output_dir=output_dir,
                force=force,
            )
            created.append(mar_path)
        except Exception as e:
            print(f"Failed to create {optimization} MAR: {e}")
    
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Create TorchServe Model Archive (.mar) files"
    )
    parser.add_argument(
        "--optimization",
        choices=OPTIMIZATION_CONFIGS,
        default="baseline",
        help="Optimization configuration to package",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_STORE_DIR,
        help="Output directory for .mar files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create MAR files for all optimization configurations",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .mar files",
    )
    parser.add_argument(
        "--version",
        default="1.0",
        help="Model version string",
    )
    
    args = parser.parse_args()
    
    # Verify handler exists
    if not HANDLER_PATH.exists():
        print(f"Error: Handler not found: {HANDLER_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("TorchServe Model Archiver")
    print("=" * 60)
    
    if args.all:
        created = create_all_mars(output_dir=args.output_dir, force=args.force)
        print(f"\nCreated {len(created)} MAR files")
    else:
        create_mar(
            optimization=args.optimization,
            output_dir=args.output_dir,
            version=args.version,
            force=args.force,
        )
    
    print("\nTo deploy with TorchServe:")
    print(f"  torchserve --start --model-store {args.output_dir} --models all")


if __name__ == "__main__":
    main()
