"""TorchServe server management for external mode.

This module provides utilities to start, stop, and interact with
a TorchServe server instance. It's used for production-like benchmarking
where TorchServe runs as a separate process.

Usage:
    manager = TorchServeManager(model_store="model_store")
    manager.start()
    
    # Send inference request
    result = manager.infer("efficientdet_baseline", image_bytes)
    
    manager.stop()
"""

import io
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class TorchServeManager:
    """Manager for TorchServe server lifecycle and inference.
    
    This class handles:
    - Starting/stopping TorchServe server process
    - Model registration via management API
    - Inference requests via predictions API
    - Health checks and status monitoring
    
    Example:
        manager = TorchServeManager(
            model_store=Path("model_store"),
            config_file=Path("src/torchserve/config.properties"),
        )
        
        with manager:  # Starts server, stops on exit
            result = manager.infer("efficientdet_baseline", image_bytes)
    """
    
    def __init__(
        self,
        model_store: Union[str, Path] = "model_store",
        config_file: Optional[Union[str, Path]] = None,
        inference_port: int = 8080,
        management_port: int = 8081,
        metrics_port: int = 8082,
        startup_timeout: float = 60.0,
        log_location: Optional[Union[str, Path]] = None,
    ):
        """Initialize TorchServe manager.
        
        Args:
            model_store: Path to model store directory
            config_file: Path to config.properties file
            inference_port: Port for inference API
            management_port: Port for management API
            metrics_port: Port for metrics API
            startup_timeout: Max seconds to wait for server startup
            log_location: Directory for TorchServe logs
        """
        self.model_store = Path(model_store)
        self.config_file = Path(config_file) if config_file else None
        self.inference_port = inference_port
        self.management_port = management_port
        self.metrics_port = metrics_port
        self.startup_timeout = startup_timeout
        self.log_location = Path(log_location) if log_location else None
        
        self.inference_url = f"http://localhost:{inference_port}"
        self.management_url = f"http://localhost:{management_port}"
        self.metrics_url = f"http://localhost:{metrics_port}"
        
        self._process: Optional[subprocess.Popen] = None
        self._started = False
    
    def __enter__(self) -> "TorchServeManager":
        """Context manager entry - start server."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop server."""
        self.stop()
    
    def start(self, models: str = "all") -> None:
        """Start TorchServe server.
        
        Args:
            models: Models to load ("all" or comma-separated list)
        """
        if self._started:
            logger.warning("TorchServe already started")
            return
        
        # Verify model store exists
        if not self.model_store.exists():
            raise FileNotFoundError(f"Model store not found: {self.model_store}")
        
        # Build command
        cmd = [
            "torchserve",
            "--start",
            "--model-store", str(self.model_store),
            "--models", models,
            "--ncs",  # No config snapshots
        ]
        
        if self.config_file and self.config_file.exists():
            cmd.extend(["--ts-config", str(self.config_file)])
        
        if self.log_location:
            self.log_location.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--log-config", str(self.log_location)])
        
        logger.info(f"Starting TorchServe: {' '.join(cmd)}")
        
        # Start server process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait for server to be ready
        if not self._wait_for_healthy():
            self.stop()
            raise RuntimeError("TorchServe failed to start within timeout")
        
        self._started = True
        logger.info("TorchServe started successfully")
    
    def stop(self) -> None:
        """Stop TorchServe server."""
        if not self._started and self._process is None:
            return
        
        logger.info("Stopping TorchServe...")
        
        # Try graceful shutdown first
        try:
            subprocess.run(
                ["torchserve", "--stop"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            pass
        
        # Force kill if process still running
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        
        self._process = None
        self._started = False
        logger.info("TorchServe stopped")
    
    def _wait_for_healthy(self) -> bool:
        """Wait for server to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            if self.is_healthy():
                return True
            time.sleep(1.0)
        
        return False
    
    def is_healthy(self) -> bool:
        """Check if server is healthy and responding."""
        try:
            response = requests.get(
                f"{self.inference_url}/ping",
                timeout=2.0,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List registered models via management API."""
        try:
            response = requests.get(
                f"{self.management_url}/models",
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def register_model(
        self,
        model_name: str,
        mar_file: Optional[str] = None,
        initial_workers: int = 1,
    ) -> bool:
        """Register a model via management API.
        
        Args:
            model_name: Name of the model
            mar_file: Path to .mar file (relative to model_store)
            initial_workers: Number of workers to start
            
        Returns:
            True if registration succeeded
        """
        url = f"{self.management_url}/models"
        params = {
            "url": mar_file or f"{model_name}.mar",
            "initial_workers": initial_workers,
            "synchronous": "true",
        }
        
        try:
            response = requests.post(url, params=params, timeout=60.0)
            response.raise_for_status()
            logger.info(f"Registered model: {model_name}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            return False
    
    def unregister_model(self, model_name: str) -> bool:
        """Unregister a model via management API."""
        url = f"{self.management_url}/models/{model_name}"
        
        try:
            response = requests.delete(url, timeout=30.0)
            response.raise_for_status()
            logger.info(f"Unregistered model: {model_name}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to unregister model {model_name}: {e}")
            return False
    
    def infer(
        self,
        model_name: str,
        image: Union[bytes, Image.Image, Path, str],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Send inference request to TorchServe.
        
        Args:
            model_name: Name of the model to use
            image: Image data (bytes, PIL Image, or path)
            timeout: Request timeout in seconds
            
        Returns:
            Detection results dictionary
        """
        # Convert image to bytes
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image
        
        # Send request
        url = f"{self.inference_url}/predictions/{model_name}"
        
        try:
            start_time = time.perf_counter()
            
            response = requests.post(
                url,
                data=image_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=timeout,
            )
            response.raise_for_status()
            
            request_time_ms = (time.perf_counter() - start_time) * 1000
            
            result = response.json()
            result["request_time_ms"] = request_time_ms
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Inference request failed: {e}")
            raise
    
    def infer_batch(
        self,
        model_name: str,
        images: List[Union[bytes, Image.Image, Path, str]],
        timeout: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """Send batch inference requests.
        
        Note: This sends sequential requests. For true batching,
        configure TorchServe's batch_size and max_batch_delay.
        
        Args:
            model_name: Name of the model
            images: List of images
            timeout: Request timeout
            
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            result = self.infer(model_name, image, timeout=timeout)
            results.append(result)
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get TorchServe metrics."""
        try:
            response = requests.get(
                f"{self.metrics_url}/metrics",
                timeout=5.0,
            )
            response.raise_for_status()
            return {"raw": response.text}
        except requests.RequestException as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
