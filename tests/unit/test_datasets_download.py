"""Unit tests for dataset download utilities."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.datasets.download import (
    download_file,
    extract_zip,
    download_coco_val2017,
    download_roboflow_aquarium,
    download_all_datasets,
    DownloadProgressBar,
)


class TestDownloadProgressBar:
    """Tests for DownloadProgressBar."""

    def test_update_to(self):
        """Test update_to method."""
        pbar = DownloadProgressBar(total=100)
        
        # Simulate download progress
        pbar.update_to(1, 10, 100)  # 10 bytes
        assert pbar.n == 10
        
        pbar.update_to(2, 10, 100)  # 20 bytes total
        assert pbar.n == 20
        
        pbar.close()

    def test_update_to_sets_total(self):
        """Test that update_to sets total from tsize."""
        pbar = DownloadProgressBar()
        
        pbar.update_to(1, 10, 1000)
        assert pbar.total == 1000
        
        pbar.close()


class TestDownloadFile:
    """Tests for download_file function."""

    @patch("src.datasets.download.urlretrieve")
    def test_creates_parent_directory(self, mock_urlretrieve, temp_dir):
        """Test that parent directories are created."""
        output_path = temp_dir / "subdir" / "nested" / "file.zip"
        
        download_file("http://example.com/file.zip", output_path)
        
        assert output_path.parent.exists()

    @patch("src.datasets.download.urlretrieve")
    def test_calls_urlretrieve(self, mock_urlretrieve, temp_dir):
        """Test that urlretrieve is called correctly."""
        output_path = temp_dir / "file.zip"
        url = "http://example.com/file.zip"
        
        download_file(url, output_path)
        
        mock_urlretrieve.assert_called_once()
        call_args = mock_urlretrieve.call_args
        assert call_args[0][0] == url
        assert call_args[0][1] == output_path


class TestExtractZip:
    """Tests for extract_zip function."""

    def test_extracts_zip_file(self, temp_dir):
        """Test that zip files are extracted correctly."""
        # Create a test zip file
        zip_path = temp_dir / "test.zip"
        extract_dir = temp_dir / "extracted"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")
        
        extract_zip(zip_path, extract_dir)
        
        assert (extract_dir / "file1.txt").exists()
        assert (extract_dir / "subdir" / "file2.txt").exists()
        assert (extract_dir / "file1.txt").read_text() == "content1"

    def test_creates_extract_directory(self, temp_dir):
        """Test that extraction directory is created."""
        zip_path = temp_dir / "test.zip"
        extract_dir = temp_dir / "new_dir" / "nested"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        extract_zip(zip_path, extract_dir)
        
        assert extract_dir.exists()


class TestDownloadCocoVal2017:
    """Tests for download_coco_val2017 function."""

    def test_skips_if_exists(self, temp_dir, sample_image):
        """Test that download is skipped if images exist."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)
        sample_image.save(coco_dir / "test.jpg")
        
        result = download_coco_val2017(temp_dir, force=False)
        
        assert result == coco_dir

    @patch("src.datasets.download.download_file")
    @patch("src.datasets.download.extract_zip")
    def test_downloads_and_extracts(self, mock_extract, mock_download, temp_dir):
        """Test download and extraction flow."""
        # Setup mock to create the expected structure and the zip file
        def mock_download_side_effect(url, output_path, desc=None):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
        
        def mock_extract_side_effect(zip_path, extract_dir):
            val_dir = extract_dir / "val2017"
            val_dir.mkdir(parents=True, exist_ok=True)
            (val_dir / "test.jpg").touch()
        
        mock_download.side_effect = mock_download_side_effect
        mock_extract.side_effect = mock_extract_side_effect
        
        result = download_coco_val2017(temp_dir, force=True)
        
        mock_download.assert_called_once()
        mock_extract.assert_called_once()

    def test_force_redownload(self, temp_dir, sample_image):
        """Test that force=True triggers redownload."""
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)
        sample_image.save(coco_dir / "test.jpg")
        
        with patch("src.datasets.download.download_file") as mock_download:
            with patch("src.datasets.download.extract_zip") as mock_extract:
                def mock_download_side_effect(url, output_path, desc=None):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.touch()
                
                def mock_extract_side_effect(zip_path, extract_dir):
                    val_dir = extract_dir / "val2017"
                    val_dir.mkdir(parents=True, exist_ok=True)
                    (val_dir / "test.jpg").touch()
                
                mock_download.side_effect = mock_download_side_effect
                mock_extract.side_effect = mock_extract_side_effect
                
                download_coco_val2017(temp_dir, force=True)
                
                mock_download.assert_called_once()


class TestDownloadRoboflowAquarium:
    """Tests for download_roboflow_aquarium function."""

    def test_skips_if_exists(self, temp_dir, sample_image):
        """Test that download is skipped if images exist."""
        aquarium_dir = temp_dir / "roboflow" / "aquarium" / "valid"
        aquarium_dir.mkdir(parents=True)
        sample_image.save(aquarium_dir / "test.jpg")
        
        result = download_roboflow_aquarium(temp_dir, force=False)
        
        assert result == temp_dir / "roboflow" / "aquarium"

    @patch("src.datasets.download.download_file")
    @patch("src.datasets.download.extract_zip")
    def test_downloads_and_extracts(self, mock_extract, mock_download, temp_dir):
        """Test download and extraction flow."""
        def mock_download_side_effect(url, output_path, desc=None):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
        
        def mock_extract_side_effect(zip_path, extract_dir):
            valid_dir = extract_dir / "valid"
            valid_dir.mkdir(parents=True, exist_ok=True)
            (valid_dir / "test.jpg").touch()
        
        mock_download.side_effect = mock_download_side_effect
        mock_extract.side_effect = mock_extract_side_effect
        
        result = download_roboflow_aquarium(temp_dir, force=True)
        
        mock_download.assert_called_once()
        mock_extract.assert_called_once()


class TestDownloadAllDatasets:
    """Tests for download_all_datasets function."""

    @patch("src.datasets.download.download_coco_val2017")
    @patch("src.datasets.download.download_roboflow_aquarium")
    def test_calls_both_downloaders(self, mock_aquarium, mock_coco, temp_dir):
        """Test that both dataset downloaders are called."""
        mock_coco.return_value = temp_dir / "coco" / "val2017"
        mock_aquarium.return_value = temp_dir / "roboflow" / "aquarium"
        
        result = download_all_datasets(temp_dir)
        
        mock_coco.assert_called_once_with(temp_dir, False)
        mock_aquarium.assert_called_once_with(temp_dir, False)
        
        assert "coco_val2017" in result
        assert "roboflow_aquarium" in result

    @patch("src.datasets.download.download_coco_val2017")
    @patch("src.datasets.download.download_roboflow_aquarium")
    def test_passes_force_flag(self, mock_aquarium, mock_coco, temp_dir):
        """Test that force flag is passed to downloaders."""
        mock_coco.return_value = temp_dir / "coco"
        mock_aquarium.return_value = temp_dir / "aquarium"
        
        download_all_datasets(temp_dir, force=True)
        
        mock_coco.assert_called_once_with(temp_dir, True)
        mock_aquarium.assert_called_once_with(temp_dir, True)
