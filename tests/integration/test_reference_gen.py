"""Integration tests for reference generation workflow."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.benchmark.reference import (
    generate_reference_outputs,
    load_reference,
    load_manifest,
)
from src.models.base import DetectionOutput


class TestGenerateReferenceOutputs:
    """Tests for generate_reference_outputs function."""

    @patch("src.benchmark.reference.BaselineImpl")
    def test_generates_references(
        self,
        mock_baseline_class,
        temp_data_dir,
        temp_output_dir,
        sample_detection_output,
    ):
        """Test that references are generated correctly."""
        # Setup mock
        mock_baseline = MagicMock()
        mock_baseline.num_models = 3
        mock_baseline.device = torch.device("cpu")
        mock_baseline.name = "baseline_sequential"
        mock_baseline.predict.return_value = [
            sample_detection_output,
            sample_detection_output,
            sample_detection_output,
        ]
        mock_baseline_class.return_value = mock_baseline
        
        references = generate_reference_outputs(
            data_dir=temp_data_dir,
            output_dir=temp_output_dir,
            max_images=5,
            device="cpu",
        )
        
        assert len(references) > 0
        
        # Check that manifest was created
        manifest_file = temp_output_dir / "manifest.json"
        assert manifest_file.exists()

    @patch("src.benchmark.reference.BaselineImpl")
    def test_reference_files_created(
        self,
        mock_baseline_class,
        temp_data_dir,
        temp_output_dir,
        sample_detection_output,
    ):
        """Test that individual reference files are created."""
        mock_baseline = MagicMock()
        mock_baseline.num_models = 3
        mock_baseline.device = torch.device("cpu")
        mock_baseline.name = "baseline_sequential"
        mock_baseline.predict.return_value = [sample_detection_output]
        mock_baseline_class.return_value = mock_baseline
        
        references = generate_reference_outputs(
            data_dir=temp_data_dir,
            output_dir=temp_output_dir,
            max_images=2,
            device="cpu",
        )
        
        # Check that reference files exist
        for ref_id, ref_path in references.items():
            assert ref_path.exists()
            
            # Check file content
            with open(ref_path) as f:
                data = json.load(f)
            
            assert "image_id" in data
            assert "dataset_name" in data
            assert "outputs" in data

    @patch("src.benchmark.reference.BaselineImpl")
    def test_manifest_structure(
        self,
        mock_baseline_class,
        temp_data_dir,
        temp_output_dir,
        sample_detection_output,
    ):
        """Test manifest file structure."""
        mock_baseline = MagicMock()
        mock_baseline.num_models = 3
        mock_baseline.device = torch.device("cpu")
        mock_baseline.name = "baseline_sequential"
        mock_baseline.predict.return_value = [sample_detection_output]
        mock_baseline_class.return_value = mock_baseline
        
        generate_reference_outputs(
            data_dir=temp_data_dir,
            output_dir=temp_output_dir,
            max_images=3,
            device="cpu",
        )
        
        manifest = load_manifest(temp_output_dir)
        
        assert "num_references" in manifest
        assert "model_name" in manifest
        assert "num_models" in manifest
        assert "references" in manifest
        assert manifest["model_name"] == "baseline_sequential"

    @patch("src.benchmark.reference.BaselineImpl")
    def test_no_images_raises(
        self,
        mock_baseline_class,
        temp_dir,
        temp_output_dir,
    ):
        """Test that error is raised when no images found."""
        mock_baseline = MagicMock()
        mock_baseline_class.return_value = mock_baseline
        
        with pytest.raises(RuntimeError, match="No images found"):
            generate_reference_outputs(
                data_dir=temp_dir,  # Empty directory
                output_dir=temp_output_dir,
                max_images=5,
                device="cpu",
            )


class TestLoadReference:
    """Tests for load_reference function."""

    def test_loads_reference_file(self, temp_dir, sample_detection_output):
        """Test loading a reference file."""
        ref_data = {
            "image_id": "test_001",
            "dataset_name": "coco_val2017",
            "image_path": "/path/to/image.jpg",
            "outputs": [sample_detection_output.to_dict()],
        }
        
        ref_file = temp_dir / "test_ref.json"
        with open(ref_file, "w") as f:
            json.dump(ref_data, f)
        
        loaded = load_reference(ref_file)
        
        assert loaded["image_id"] == "test_001"
        assert loaded["dataset_name"] == "coco_val2017"
        assert len(loaded["outputs"]) == 1


class TestLoadManifest:
    """Tests for load_manifest function."""

    def test_loads_manifest(self, temp_dir):
        """Test loading a manifest file."""
        manifest_data = {
            "num_references": 10,
            "model_name": "baseline_sequential",
            "num_models": 3,
            "references": [
                {"ref_id": "test_1", "image_id": "001"},
                {"ref_id": "test_2", "image_id": "002"},
            ],
        }
        
        manifest_file = temp_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)
        
        loaded = load_manifest(temp_dir)
        
        assert loaded["num_references"] == 10
        assert loaded["model_name"] == "baseline_sequential"
        assert len(loaded["references"]) == 2
