"""Unified dataset loading interface for benchmark image iteration."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

from PIL import Image


@dataclass
class ImageItem:
    """A single image item from a dataset.
    
    Attributes:
        path: Path to the image file
        image: Loaded PIL Image (lazy loaded if not accessed)
        dataset_name: Name of the source dataset
        image_id: Unique identifier within the dataset
    """
    path: Path
    dataset_name: str
    image_id: str
    _image: Optional[Image.Image] = None
    
    @property
    def image(self) -> Image.Image:
        """Lazy load the image."""
        if self._image is None:
            self._image = Image.open(self.path)
            if self._image.mode != "RGB":
                self._image = self._image.convert("RGB")
        return self._image
    
    def load_bytes(self) -> bytes:
        """Load image as raw bytes for HTTP requests."""
        with open(self.path, "rb") as f:
            return f.read()


class DatasetLoader:
    """Unified dataset loader for benchmark image iteration.
    
    Supports loading from multiple dataset directories and iterating
    through images for benchmarking.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the dataset loader.
        
        Args:
            data_dir: Base directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self._items: List[ImageItem] = []
        self._datasets: dict = {}

    def add_dataset(
        self,
        name: str,
        path: Union[str, Path],
        extensions: tuple = (".jpg", ".jpeg", ".png"),
        recursive: bool = True,
        max_images: Optional[int] = None,
    ) -> int:
        """Add a dataset directory to the loader.
        
        Args:
            name: Name identifier for this dataset
            path: Path to dataset directory
            extensions: File extensions to include
            recursive: If True, search subdirectories
            max_images: Maximum number of images to load (None for all)
            
        Returns:
            Number of images added
        """
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all images
        if recursive:
            image_files = []
            for ext in extensions:
                image_files.extend(dataset_path.rglob(f"*{ext}"))
                image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        else:
            image_files = []
            for ext in extensions:
                image_files.extend(dataset_path.glob(f"*{ext}"))
                image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        # Sort for reproducibility
        image_files = sorted(set(image_files))
        
        # Limit if specified
        if max_images is not None:
            image_files = image_files[:max_images]
        
        # Create ImageItems
        dataset_items = []
        for img_path in image_files:
            item = ImageItem(
                path=img_path,
                dataset_name=name,
                image_id=img_path.stem,
            )
            dataset_items.append(item)
        
        self._items.extend(dataset_items)
        self._datasets[name] = {
            "path": dataset_path,
            "count": len(dataset_items),
        }
        
        return len(dataset_items)

    def add_coco_val2017(self, max_images: Optional[int] = None) -> int:
        """Add COCO val2017 dataset.
        
        Args:
            max_images: Maximum number of images to load
            
        Returns:
            Number of images added
        """
        coco_path = self.data_dir / "coco" / "val2017"
        return self.add_dataset(
            name="coco_val2017",
            path=coco_path,
            extensions=(".jpg",),
            recursive=False,
            max_images=max_images,
        )

    def add_roboflow_aquarium(self, max_images: Optional[int] = None) -> int:
        """Add Roboflow Aquarium dataset.
        
        Args:
            max_images: Maximum number of images to load
            
        Returns:
            Number of images added
        """
        aquarium_path = self.data_dir / "roboflow" / "aquarium"
        return self.add_dataset(
            name="roboflow_aquarium",
            path=aquarium_path,
            extensions=(".jpg", ".jpeg", ".png"),
            recursive=True,
            max_images=max_images,
        )

    def add_all_datasets(self, max_images_per_dataset: Optional[int] = None) -> dict:
        """Add all available datasets.
        
        Args:
            max_images_per_dataset: Maximum images per dataset
            
        Returns:
            Dictionary with counts per dataset
        """
        counts = {}
        
        try:
            counts["coco_val2017"] = self.add_coco_val2017(max_images_per_dataset)
        except FileNotFoundError:
            counts["coco_val2017"] = 0
            print("Warning: COCO val2017 not found, skipping")
        
        try:
            counts["roboflow_aquarium"] = self.add_roboflow_aquarium(max_images_per_dataset)
        except FileNotFoundError:
            counts["roboflow_aquarium"] = 0
            print("Warning: Roboflow Aquarium not found, skipping")
        
        return counts

    def __len__(self) -> int:
        """Return total number of images."""
        return len(self._items)

    def __getitem__(self, idx: int) -> ImageItem:
        """Get image by index."""
        return self._items[idx]

    def __iter__(self) -> Iterator[ImageItem]:
        """Iterate through all images."""
        return iter(self._items)

    def sample(self, n: int, seed: Optional[int] = None) -> List[ImageItem]:
        """Sample n random images.
        
        Args:
            n: Number of images to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled ImageItems
        """
        if seed is not None:
            random.seed(seed)
        return random.sample(self._items, min(n, len(self._items)))

    def get_by_dataset(self, dataset_name: str) -> List[ImageItem]:
        """Get all images from a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of ImageItems from that dataset
        """
        return [item for item in self._items if item.dataset_name == dataset_name]

    @property
    def datasets(self) -> dict:
        """Get dataset information."""
        return self._datasets.copy()

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the image order.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._items)
