#!/usr/bin/env python3
"""Load testing script using Locust programmatic API.

This script runs controlled load tests against different inference configurations,
measuring throughput (RPS), latency percentiles, and error rates.

Two testing modes:
1. Server Mode: Runs against FastAPI server via HTTP
2. Embedded Mode: Calls inference directly (no HTTP overhead)

Usage:
    # Embedded mode (direct function calls, recommended for benchmarking)
    uv run python scripts/run_load_test.py --mode embedded --config vmap_backbone

    # Server mode (via HTTP)  
    uv run python scripts/run_load_test.py --mode server --config baseline

    # All configurations
    uv run python scripts/run_load_test.py --all --duration 60

    # Custom concurrency
    uv run python scripts/run_load_test.py --users 4 --spawn-rate 2 --duration 60
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np


# Configuration
DEFAULT_DURATION = 60  # seconds
DEFAULT_USERS = 1
DEFAULT_SPAWN_RATE = 1
DEFAULT_IMAGE_SIZE = 512


class LoadTestConfig:
    """Configuration for load tests."""
    
    # Available configurations for testing
    CONFIGS = {
        "baseline": {
            "optimization": "baseline",
            "description": "Sequential inference (no optimizations)",
        },
        "compile": {
            "optimization": "compile",
            "description": "torch.compile enabled",
        },
        "vmap_backbone": {
            "optimization": "vmap_backbone",
            "description": "vmap backbone optimization",
        },
        "baseline_batch4": {
            "optimization": "baseline",
            "batch_size": 4,
            "description": "Baseline with batch_size=4",
        },
        "vmap_batch4": {
            "optimization": "vmap_backbone",
            "batch_size": 4,
            "description": "vmap with batch_size=4",
        },
        "torchserve_baseline": {
            "optimization": "baseline",
            "torchserve": True,
            "description": "TorchServe embedded (baseline)",
        },
        "torchserve_vmap": {
            "optimization": "vmap_backbone",
            "torchserve": True,
            "description": "TorchServe embedded (vmap)",
        },
    }
    
    # Configurations that support concurrent users (for dynamic batching tests)
    CONCURRENT_CONFIGS = {
        "baseline_concurrent": {
            "optimization": "baseline",
            "concurrent_users": 4,
            "description": "Baseline with 4 concurrent users",
        },
        "vmap_concurrent": {
            "optimization": "vmap_backbone",
            "concurrent_users": 4,
            "description": "vmap with 4 concurrent users",
        },
    }


def create_test_image(size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Create a random test image."""
    data = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


class LoadTestRunner:
    """Runs load tests by measuring throughput directly.
    
    Instead of using Locust's complex gevent-based approach, this runner
    uses a simple continuous loop to measure pure inference throughput.
    """
    
    def __init__(
        self,
        users: int = DEFAULT_USERS,
        spawn_rate: float = DEFAULT_SPAWN_RATE,
        duration: int = DEFAULT_DURATION,
        device: str = "auto",
    ):
        self.users = users
        self.spawn_rate = spawn_rate
        self.duration = duration
        self.device = self._resolve_device(device)
        
        self.results: List[Dict[str, Any]] = []
        self.test_image = create_test_image()
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        import torch
        
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
    
    def _run_continuous_load_test(
        self,
        predict_fn: Callable,
        config_name: str,
        mode: str = "embedded",
        print_result: bool = True,
    ) -> Dict[str, Any]:
        """Run continuous load test measuring throughput.
        
        Args:
            predict_fn: Function to call for inference (takes no args)
            config_name: Configuration name for reporting
            mode: Test mode description
            
        Returns:
            Test results dictionary
        """
        print(f"\nStarting load test:")
        print(f"  Duration: {self.duration}s")
        print(f"  Device: {self.device}")
        
        latencies_ms: List[float] = []
        errors = 0
        
        # Run continuous inference for specified duration
        start_time = time.perf_counter()
        end_time = start_time + self.duration
        request_count = 0
        last_report_time = start_time
        last_report_count = 0
        
        print(f"\nRunning for {self.duration} seconds...")
        
        while time.perf_counter() < end_time:
            try:
                # Measure single inference
                inference_start = time.perf_counter()
                predict_fn()
                inference_time_ms = (time.perf_counter() - inference_start) * 1000
                
                latencies_ms.append(inference_time_ms)
                request_count += 1
                
            except Exception as e:
                errors += 1
                print(f"Error: {e}")
            
            # Print progress every 10 seconds
            current_time = time.perf_counter()
            if current_time - last_report_time >= 10:
                elapsed = current_time - start_time
                current_rps = (request_count - last_report_count) / (current_time - last_report_time)
                print(f"  [{elapsed:.0f}s] Requests: {request_count}, Current RPS: {current_rps:.1f}")
                last_report_time = current_time
                last_report_count = request_count
        
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        if latencies_ms:
            latencies_sorted = sorted(latencies_ms)
            n = len(latencies_sorted)
            
            result = {
                "config": config_name,
                "mode": mode,
                "users": 1,  # Single-threaded for pure throughput
                "spawn_rate": self.spawn_rate,
                "duration_s": round(total_time, 2),
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                
                # Throughput
                "total_requests": request_count,
                "total_failures": errors,
                "rps": round(request_count / total_time, 2),
                "fail_ratio": round((errors / max(1, request_count + errors)) * 100, 2),
                
                # Latency (ms)
                "latency_mean_ms": round(sum(latencies_ms) / n, 2),
                "latency_min_ms": round(latencies_sorted[0], 2),
                "latency_max_ms": round(latencies_sorted[-1], 2),
                "latency_p50_ms": round(latencies_sorted[int(n * 0.5)], 2),
                "latency_p90_ms": round(latencies_sorted[int(n * 0.9)], 2),
                "latency_p95_ms": round(latencies_sorted[int(n * 0.95)], 2),
                "latency_p99_ms": round(latencies_sorted[min(int(n * 0.99), n - 1)], 2),
            }
        else:
            result = {
                "config": config_name,
                "mode": mode,
                "error": "No successful requests",
            }
        
        if print_result:
            self._print_result(result)
        self.results.append(result)
        
        return result
    
    def run_embedded_test(
        self,
        config_name: str,
        optimization: str,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Run embedded mode load test.
        
        Args:
            config_name: Configuration name for reporting
            optimization: Optimization type
            batch_size: Number of images per inference call
            
        Returns:
            Test results dictionary
        """
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {config_name} (embedded mode, batch_size={batch_size})")
        print(f"{'='*60}")
        
        # Create optimization config
        if optimization == "baseline":
            config = OptimizationConfig()
        elif optimization == "compile":
            config = OptimizationConfig(compile_enabled=True)
        elif optimization == "vmap_backbone":
            config = OptimizationConfig(vmap_backbone_enabled=True)
        else:
            config = OptimizationConfig()
        
        print(f"Loading model with config: {config}")
        impl = OptimizedImpl(device=self.device, optimization_config=config)
        impl.load()
        impl.warmup()
        
        # Create batch of test images
        if batch_size > 1:
            test_images = [create_test_image() for _ in range(batch_size)]
            
            def predict_fn():
                # Batched inference - process multiple images
                return impl.predict_batch(test_images)
            
            mode = f"embedded_batch{batch_size}"
        else:
            def predict_fn():
                return impl.predict(self.test_image)
            mode = "embedded"
        
        # Run load test (don't print yet for batched tests)
        result = self._run_continuous_load_test(predict_fn, config_name, mode, print_result=(batch_size == 1))
        
        # For batched tests, adjust the metrics to be per-image
        if batch_size > 1:
            result["batch_size"] = batch_size
            result["images_per_second"] = round(result["rps"] * batch_size, 2)
            result["latency_per_image_ms"] = round(result["latency_mean_ms"] / batch_size, 2)
            # Now print with full batch info
            self._print_result(result)
        
        return result
    
    def run_torchserve_embedded_test(
        self,
        config_name: str,
        optimization: str,
    ) -> Dict[str, Any]:
        """Run TorchServe embedded mode load test.
        
        Args:
            config_name: Configuration name for reporting
            optimization: Optimization type
            
        Returns:
            Test results dictionary
        """
        from src.torchserve.embedded import EmbeddedTorchServe
        
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {config_name} (TorchServe embedded)")
        print(f"{'='*60}")
        
        print(f"Loading TorchServe handler with optimization: {optimization}")
        embedded = EmbeddedTorchServe(
            optimization=optimization,
            device=self.device,
        )
        embedded.start()
        
        # Create predict function
        def predict_fn():
            return embedded.infer(self.test_image)
        
        try:
            # Run load test
            return self._run_continuous_load_test(predict_fn, config_name, "torchserve_embedded")
        finally:
            # Cleanup
            embedded.stop()
    
    def run_concurrent_test(
        self,
        config_name: str,
        optimization: str,
        num_workers: int = 4,
    ) -> Dict[str, Any]:
        """Run concurrent load test with multiple worker threads.
        
        This simulates real-world scenarios where multiple clients send
        requests simultaneously. For MPS/CUDA, we need to serialize access
        to the model, but this still measures queueing behavior.
        
        NOTE: MPS does not support concurrent GPU access from multiple threads.
        For true concurrent GPU inference, use CUDA with proper stream management
        or run multiple model instances.
        
        Args:
            config_name: Configuration name for reporting
            optimization: Optimization type
            num_workers: Number of concurrent worker threads
            
        Returns:
            Test results dictionary
        """
        import threading
        import queue
        from src.models.optimized import OptimizedImpl
        from src.models.optimizations.base import OptimizationConfig
        
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {config_name} (concurrent, {num_workers} workers)")
        print(f"{'='*60}")
        
        # Create optimization config
        if optimization == "baseline":
            config = OptimizationConfig()
        elif optimization == "compile":
            config = OptimizationConfig(compile_enabled=True)
        elif optimization == "vmap_backbone":
            config = OptimizationConfig(vmap_backbone_enabled=True)
        else:
            config = OptimizationConfig()
        
        print(f"Loading model with config: {config}")
        impl = OptimizedImpl(device=self.device, optimization_config=config)
        impl.load()
        impl.warmup()
        
        # Shared state for workers
        results_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        model_lock = threading.Lock()  # Serialize model access for MPS
        
        # Create test images for each worker
        test_images = [create_test_image() for _ in range(num_workers)]
        
        def worker(worker_id: int, test_image: Image.Image):
            """Worker function that continuously runs inference."""
            while not stop_event.is_set():
                try:
                    start = time.perf_counter()
                    # Serialize access to model (required for MPS)
                    with model_lock:
                        impl.predict(test_image)
                    latency_ms = (time.perf_counter() - start) * 1000
                    results_queue.put(("success", latency_ms))
                except Exception as e:
                    results_queue.put(("error", str(e)))
        
        print(f"\nStarting {num_workers} concurrent workers (serialized for MPS)...")
        print(f"Duration: {self.duration}s")
        print("NOTE: MPS requires serialized access. Latency includes queue time.")
        
        # Start workers
        threads = []
        for i in range(num_workers):
            t = threading.Thread(target=worker, args=(i, test_images[i]))
            t.daemon = True
            threads.append(t)
        
        start_time = time.perf_counter()
        for t in threads:
            t.start()
        
        # Wait for duration
        last_report = start_time
        while time.perf_counter() - start_time < self.duration:
            time.sleep(0.1)
            if time.perf_counter() - last_report >= 10:
                elapsed = time.perf_counter() - start_time
                size = results_queue.qsize()
                print(f"  [{elapsed:.0f}s] Requests: {size}")
                last_report = time.perf_counter()
        
        # Stop workers
        print("\nStopping workers...")
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        
        total_time = time.perf_counter() - start_time
        
        # Collect results
        latencies_ms = []
        errors = 0
        while not results_queue.empty():
            status, value = results_queue.get_nowait()
            if status == "success":
                latencies_ms.append(value)
            else:
                errors += 1
        
        if latencies_ms:
            latencies_sorted = sorted(latencies_ms)
            n = len(latencies_sorted)
            
            result = {
                "config": config_name,
                "mode": f"concurrent_{num_workers}_workers",
                "users": num_workers,
                "spawn_rate": self.spawn_rate,
                "duration_s": round(total_time, 2),
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                
                # Throughput
                "total_requests": len(latencies_ms),
                "total_failures": errors,
                "rps": round(len(latencies_ms) / total_time, 2),
                "fail_ratio": round((errors / max(1, len(latencies_ms) + errors)) * 100, 2),
                
                # Latency (ms) - includes queue wait time for concurrent tests
                "latency_mean_ms": round(sum(latencies_ms) / n, 2),
                "latency_min_ms": round(latencies_sorted[0], 2),
                "latency_max_ms": round(latencies_sorted[-1], 2),
                "latency_p50_ms": round(latencies_sorted[int(n * 0.5)], 2),
                "latency_p90_ms": round(latencies_sorted[int(n * 0.9)], 2),
                "latency_p95_ms": round(latencies_sorted[int(n * 0.95)], 2),
                "latency_p99_ms": round(latencies_sorted[min(int(n * 0.99), n - 1)], 2),
            }
        else:
            result = {
                "config": config_name,
                "mode": f"concurrent_{num_workers}_workers",
                "error": "No successful requests",
            }
        
        self._print_result(result)
        self.results.append(result)
        
        return result
    
    def _print_result(self, result: Dict[str, Any]) -> None:
        """Print formatted test result."""
        print(f"\n{'='*60}")
        print(f"LOAD TEST RESULTS: {result['config']} ({result['mode']} mode)")
        print(f"{'='*60}")
        print(f"Duration:         {result['duration_s']}s")
        print(f"Concurrent Users: {result['users']}")
        print(f"Spawn Rate:       {result['spawn_rate']}/s")
        print(f"Device:           {result['device']}")
        
        # Show batch info if present
        if "batch_size" in result:
            print(f"Batch Size:       {result['batch_size']}")
        
        print()
        print("Throughput:")
        print(f"  Total Requests:  {result['total_requests']:,}")
        print(f"  Actual RPS:      {result['rps']}")
        
        # Show images/second for batched tests
        if "images_per_second" in result:
            print(f"  Images/second:   {result['images_per_second']}")
        
        print(f"  Failures:        {result['total_failures']} ({result['fail_ratio']}%)")
        print()
        print("Latency (ms):")
        print(f"  Mean:            {result['latency_mean_ms']}")
        
        # Show per-image latency for batched tests
        if "latency_per_image_ms" in result:
            print(f"  Per-image:       {result['latency_per_image_ms']}")
        
        print(f"  Min:             {result['latency_min_ms']}")
        print(f"  Max:             {result['latency_max_ms']}")
        print(f"  Median (p50):    {result['latency_p50_ms']}")
        print(f"  p90:             {result['latency_p90_ms']}")
        print(f"  p95:             {result['latency_p95_ms']}")
        print(f"  p99:             {result['latency_p99_ms']}")
        print(f"{'='*60}")
    
    def run_config(self, config_name: str) -> Dict[str, Any]:
        """Run load test for a specific configuration.
        
        Args:
            config_name: Name of configuration to test
            
        Returns:
            Test results
        """
        # Check both CONFIGS and CONCURRENT_CONFIGS
        all_configs = {**LoadTestConfig.CONFIGS, **LoadTestConfig.CONCURRENT_CONFIGS}
        
        if config_name not in all_configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(all_configs.keys())}")
        
        config = all_configs[config_name]
        batch_size = config.get("batch_size", 1)
        concurrent_users = config.get("concurrent_users", 0)
        
        if concurrent_users > 0:
            return self.run_concurrent_test(
                config_name,
                config["optimization"],
                num_workers=concurrent_users,
            )
        elif config.get("torchserve"):
            return self.run_torchserve_embedded_test(
                config_name,
                config["optimization"],
            )
        else:
            return self.run_embedded_test(
                config_name,
                config["optimization"],
                batch_size=batch_size,
            )
    
    def run_all_configs(self, include_concurrent: bool = False) -> List[Dict[str, Any]]:
        """Run load tests for all configurations.
        
        Args:
            include_concurrent: Whether to include concurrent tests
        
        Returns:
            List of all test results
        """
        print("\n" + "="*60)
        print("RUNNING ALL CONFIGURATIONS")
        print("="*60)
        
        configs_to_run = list(LoadTestConfig.CONFIGS.keys())
        if include_concurrent:
            configs_to_run.extend(LoadTestConfig.CONCURRENT_CONFIGS.keys())
        
        for config_name in configs_to_run:
            try:
                self.run_config(config_name)
            except Exception as e:
                print(f"\nERROR running {config_name}: {e}")
                self.results.append({
                    "config": config_name,
                    "error": str(e),
                })
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate text report of all results.
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append("LOAD TEST SUMMARY")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Device: {self.device}")
        lines.append(f"Duration per test: {self.duration}s")
        lines.append(f"Concurrent users: {self.users}")
        lines.append("")
        
        # Summary table
        lines.append(f"{'Configuration':<25} {'RPS':>8} {'p50 (ms)':>10} {'p99 (ms)':>10} {'Errors':>8}")
        lines.append("-" * 70)
        
        for result in self.results:
            if "error" in result:
                lines.append(f"{result.get('config', 'unknown'):<25} {'ERROR':>8} - {result.get('error', 'Unknown error')[:30]}")
            else:
                lines.append(
                    f"{result.get('config', 'unknown'):<25} "
                    f"{result.get('rps', 0):>8.1f} "
                    f"{result.get('latency_p50_ms', 0):>10.1f} "
                    f"{result.get('latency_p99_ms', 0):>10.1f} "
                    f"{result.get('fail_ratio', 0):>7.1f}%"
                )
        
        lines.append("-" * 70)
        lines.append("")
        
        # Analysis
        successful = [r for r in self.results if "error" not in r]
        if len(successful) >= 2:
            baseline = next((r for r in successful if r["config"] == "baseline"), None)
            if baseline and baseline.get("rps", 0) > 0:
                lines.append("Speedup vs Baseline:")
                for result in successful:
                    if result["config"] != "baseline" and result.get("rps", 0) > 0:
                        speedup = result["rps"] / baseline["rps"]
                        lines.append(f"  {result['config']}: {speedup:.2f}x")
        
        lines.append("")
        return "\n".join(lines)
    
    def save_results(
        self,
        output_dir: Path = Path("outputs/load_test"),
    ) -> None:
        """Save results to files.
        
        Args:
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / "load_test_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Save text report
        report_path = output_dir / "load_test_report.txt"
        with open(report_path, "w") as f:
            f.write(self.generate_report())
        print(f"Report saved to: {report_path}")


def main():
    # Combine all configs for choices
    all_config_names = list(LoadTestConfig.CONFIGS.keys()) + list(LoadTestConfig.CONCURRENT_CONFIGS.keys())
    
    parser = argparse.ArgumentParser(
        description="Run load tests on inference implementations"
    )
    parser.add_argument(
        "--mode",
        choices=["embedded", "server"],
        default="embedded",
        help="Testing mode (default: embedded)",
    )
    parser.add_argument(
        "--config",
        choices=all_config_names,
        help="Configuration to test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all standard configurations",
    )
    parser.add_argument(
        "--include-concurrent",
        action="store_true",
        help="Include concurrent configurations when using --all",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=DEFAULT_USERS,
        help=f"Number of concurrent users (default: {DEFAULT_USERS})",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=DEFAULT_SPAWN_RATE,
        help=f"Users spawned per second (default: {DEFAULT_SPAWN_RATE})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/load_test"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    if not args.all and not args.config:
        parser.error("Either --config or --all must be specified")
    
    # Create runner
    runner = LoadTestRunner(
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        device=args.device,
    )
    
    # Run tests
    if args.all:
        runner.run_all_configs(include_concurrent=args.include_concurrent)
    else:
        runner.run_config(args.config)
    
    # Generate and print report
    print(runner.generate_report())
    
    # Save results
    runner.save_results(args.output_dir)


if __name__ == "__main__":
    main()
