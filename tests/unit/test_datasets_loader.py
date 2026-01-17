"""Unit tests for DatasetLoader and ImageItem."""

import pytest
from pathlib import Path

from src.datasets.loader import DatasetLoader, ImageItem


class TestImageItem:
    """Tests for ImageItem dataclass."""

    def test_creation(self, temp_data_dir):
        """Test basic ImageItem creation."""
        coco_dir = temp_data_dir / "coco" / "val2017"
        img_path = list(coco_dir.glob("*.jpg"))[0]
        
        item = ImageItem(
            path=img_path,
            dataset_name="test_dataset",
            image_id="test_001",
        )
        
        assert item.path == img_path
        assert item.dataset_name == "test_dataset"
        assert item.image_id == "test_001"

    def test_lazy_image_loading(self, temp_data_dir):
        """Test that image is loaded lazily."""
        coco_dir = temp_data_dir / "coco" / "val2017"
        img_path = list(coco_dir.glob("*.jpg"))[0]
        
        item = ImageItem(
            path=img_path,
            dataset_name="test",
            image_id="test_001",
        )
        
        # Internal image should be None initially
        assert item._image is None
        
        # Accessing .image should load it
        img = item.image
        assert img is not None
        assert item._image is not None

    def test_image_conversion_to_rgb(self, temp_dir, sample_image):
        """Test that images are converted to RGB."""
        # Save as grayscale
        gray_path = temp_dir / "gray.jpg"
        gray_image = sample_image.convert("L")
        gray_image.save(gray_path)
        
        item = ImageItem(
            path=gray_path,
            dataset_name="test",
            image_id="gray",
        )
        
        img = item.image
        assert img.mode == "RGB"

    def test_load_bytes(self, temp_data_dir):
        """Test loading image as bytes."""
        coco_dir = temp_data_dir / "coco" / "val2017"
        img_path = list(coco_dir.glob("*.jpg"))[0]
        
        item = ImageItem(
            path=img_path,
            dataset_name="test",
            image_id="test_001",
        )
        
        img_bytes = item.load_bytes()
        
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        # JPEG magic bytes
        assert img_bytes[:2] == b'\xff\xd8'


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_initialization(self, temp_dir):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader(temp_dir)
        
        assert loader.data_dir == temp_dir
        assert len(loader) == 0
        assert loader.datasets == {}

    def test_add_dataset(self, temp_data_dir):
        """Test adding a dataset."""
        loader = DatasetLoader(temp_data_dir)
        
        coco_dir = temp_data_dir / "coco" / "val2017"
        count = loader.add_dataset(
            name="test_coco",
            path=coco_dir,
            extensions=(".jpg",),
            recursive=False,
        )
        
        assert count == 5  # We created 5 test images
        assert len(loader) == 5
        assert "test_coco" in loader.datasets

    def test_add_dataset_recursive(self, temp_data_dir):
        """Test adding a dataset with recursive search."""
        loader = DatasetLoader(temp_data_dir)
        
        aquarium_dir = temp_data_dir / "roboflow" / "aquarium"
        count = loader.add_dataset(
            name="test_aquarium",
            path=aquarium_dir,
            extensions=(".jpg",),
            recursive=True,
        )
        
        assert count == 3  # We created 3 test images in subdirectory
        assert len(loader) == 3

    def test_add_dataset_max_images(self, temp_data_dir):
        """Test limiting number of images."""
        loader = DatasetLoader(temp_data_dir)
        
        coco_dir = temp_data_dir / "coco" / "val2017"
        count = loader.add_dataset(
            name="limited",
            path=coco_dir,
            extensions=(".jpg",),
            max_images=2,
        )
        
        assert count == 2
        assert len(loader) == 2

    def test_add_dataset_nonexistent_raises(self, temp_dir):
        """Test that adding nonexistent dataset raises."""
        loader = DatasetLoader(temp_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.add_dataset(
                name="nonexistent",
                path=temp_dir / "does_not_exist",
            )

    def test_add_coco_val2017(self, temp_data_dir):
        """Test add_coco_val2017 convenience method."""
        loader = DatasetLoader(temp_data_dir)
        count = loader.add_coco_val2017()
        
        assert count == 5
        assert "coco_val2017" in loader.datasets

    def test_add_roboflow_aquarium(self, temp_data_dir):
        """Test add_roboflow_aquarium convenience method."""
        loader = DatasetLoader(temp_data_dir)
        count = loader.add_roboflow_aquarium()
        
        assert count == 3
        assert "roboflow_aquarium" in loader.datasets

    def test_add_all_datasets(self, temp_data_dir):
        """Test add_all_datasets method."""
        loader = DatasetLoader(temp_data_dir)
        counts = loader.add_all_datasets()
        
        assert counts["coco_val2017"] == 5
        assert counts["roboflow_aquarium"] == 3
        assert len(loader) == 8

    def test_add_all_datasets_with_limit(self, temp_data_dir):
        """Test add_all_datasets with per-dataset limit."""
        loader = DatasetLoader(temp_data_dir)
        counts = loader.add_all_datasets(max_images_per_dataset=2)
        
        assert counts["coco_val2017"] == 2
        assert counts["roboflow_aquarium"] == 2
        assert len(loader) == 4

    def test_iteration(self, temp_data_dir):
        """Test iterating through loader."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()
        
        items = list(loader)
        
        assert len(items) == 5
        for item in items:
            assert isinstance(item, ImageItem)

    def test_getitem(self, temp_data_dir):
        """Test indexing loader."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()
        
        item = loader[0]
        
        assert isinstance(item, ImageItem)
        assert item.dataset_name == "coco_val2017"

    def test_sample(self, temp_data_dir):
        """Test sampling random images."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()
        
        samples = loader.sample(3, seed=42)
        
        assert len(samples) == 3
        for item in samples:
            assert isinstance(item, ImageItem)

    def test_sample_with_seed_reproducible(self, temp_data_dir):
        """Test that sampling with seed is reproducible."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()
        
        samples1 = loader.sample(3, seed=42)
        samples2 = loader.sample(3, seed=42)
        
        assert [s.image_id for s in samples1] == [s.image_id for s in samples2]

    def test_sample_more_than_available(self, temp_data_dir):
        """Test sampling more images than available."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()  # 5 images
        
        samples = loader.sample(100)
        
        assert len(samples) == 5  # Should return all available

    def test_get_by_dataset(self, temp_data_dir):
        """Test filtering by dataset name."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_all_datasets()
        
        coco_items = loader.get_by_dataset("coco_val2017")
        aquarium_items = loader.get_by_dataset("roboflow_aquarium")
        
        assert len(coco_items) == 5
        assert len(aquarium_items) == 3
        
        for item in coco_items:
            assert item.dataset_name == "coco_val2017"

    def test_shuffle(self, temp_data_dir):
        """Test shuffling images."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()
        
        original_order = [item.image_id for item in loader]
        
        loader.shuffle(seed=42)
        shuffled_order = [item.image_id for item in loader]
        
        # Order should be different (very unlikely to be same with 5 items)
        # But with seed, should be reproducible
        loader2 = DatasetLoader(temp_data_dir)
        loader2.add_coco_val2017()
        loader2.shuffle(seed=42)
        shuffled_order2 = [item.image_id for item in loader2]
        
        assert shuffled_order == shuffled_order2

    def test_datasets_property(self, temp_data_dir):
        """Test datasets property returns copy."""
        loader = DatasetLoader(temp_data_dir)
        loader.add_coco_val2017()
        
        datasets1 = loader.datasets
        datasets2 = loader.datasets
        
        # Should be equal but different objects
        assert datasets1 == datasets2
        assert datasets1 is not datasets2

    def test_multiple_extensions(self, temp_dir, sample_image):
        """Test loading images with multiple extensions."""
        img_dir = temp_dir / "mixed"
        img_dir.mkdir()
        
        sample_image.save(img_dir / "test1.jpg")
        sample_image.save(img_dir / "test2.jpeg")
        sample_image.save(img_dir / "test3.png")
        
        loader = DatasetLoader(temp_dir)
        count = loader.add_dataset(
            name="mixed",
            path=img_dir,
            extensions=(".jpg", ".jpeg", ".png"),
        )
        
        assert count == 3

    def test_uppercase_extensions(self, temp_dir, sample_image):
        """Test that uppercase extensions are handled."""
        img_dir = temp_dir / "upper"
        img_dir.mkdir()
        
        sample_image.save(img_dir / "test1.JPG")
        sample_image.save(img_dir / "test2.JPEG")
        
        loader = DatasetLoader(temp_dir)
        count = loader.add_dataset(
            name="upper",
            path=img_dir,
            extensions=(".jpg", ".jpeg"),
        )
        
        assert count == 2

    def test_empty_directory(self, temp_dir):
        """Test adding an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        loader = DatasetLoader(temp_dir)
        count = loader.add_dataset(
            name="empty",
            path=empty_dir,
        )
        
        assert count == 0
        assert len(loader) == 0

    def test_add_all_datasets_missing_one(self, temp_dir, sample_image):
        """Test add_all_datasets when one dataset is missing."""
        # Only create COCO
        coco_dir = temp_dir / "coco" / "val2017"
        coco_dir.mkdir(parents=True)
        sample_image.save(coco_dir / "test.jpg")
        
        loader = DatasetLoader(temp_dir)
        counts = loader.add_all_datasets()
        
        assert counts["coco_val2017"] == 1
        assert counts["roboflow_aquarium"] == 0
        assert len(loader) == 1
