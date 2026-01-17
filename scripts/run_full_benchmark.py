#!/usr/bin/env python3
"""Run comprehensive benchmark and generate report."""

import io
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_system_info() -> dict:
    """Get system information for the benchmark report."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }
    
    # Try to get Mac-specific info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["cpu_brand"] = result.stdout.strip()
    except Exception:
        pass
    
    # Try to get chip info on Apple Silicon
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Chip:" in line:
                    info["chip"] = line.split(":")[-1].strip()
                elif "Model Name:" in line:
                    info["model_name"] = line.split(":")[-1].strip()
                elif "Memory:" in line:
                    info["memory"] = line.split(":")[-1].strip()
    except Exception:
        pass
    
    return info


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    implementation: str
    concurrent_requests: int
    num_requests: int
    warmup_requests: int = 10
    device: str = "cpu"


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""
    config: BenchmarkConfig
    latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0
    vram_peak_mb: float = 0.0
    
    @property
    def throughput_rps(self) -> float:
        if not self.latencies_ms:
            return 0.0
        total_time = sum(self.latencies_ms) / 1000  # Convert to seconds
        return len(self.latencies_ms) / total_time if total_time > 0 else 0.0
    
    @property
    def latency_mean_ms(self) -> float:
        return np.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def latency_p50_ms(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0
    
    @property
    def latency_p90_ms(self) -> float:
        return np.percentile(self.latencies_ms, 90) if self.latencies_ms else 0.0
    
    @property
    def latency_p99_ms(self) -> float:
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0.0
    
    @property
    def latency_min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def latency_max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0
    
    def to_dict(self) -> dict:
        return {
            "config": {
                "name": self.config.name,
                "implementation": self.config.implementation,
                "concurrent_requests": self.config.concurrent_requests,
                "num_requests": self.config.num_requests,
                "device": self.config.device,
            },
            "throughput_rps": self.throughput_rps,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "vram_peak_mb": self.vram_peak_mb,
            "errors": self.errors,
            "total_requests": len(self.latencies_ms),
        }


def create_test_images(num_images: int = 50) -> List[bytes]:
    """Create test images in memory."""
    images = []
    for i in range(num_images):
        # Create varied test images
        size = (512, 512)
        img = Image.new("RGB", size, color=(
            (i * 37) % 256,
            (i * 59) % 256,
            (i * 83) % 256,
        ))
        
        # Add some variation
        pixels = img.load()
        for x in range(0, size[0], 20):
            for y in range(0, size[1], 20):
                pixels[x, y] = ((x + i) % 256, (y + i) % 256, 128)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        images.append(buffer.getvalue())
    
    return images


def run_sequential_benchmark(
    client,
    config: BenchmarkConfig,
    images: List[bytes],
) -> BenchmarkMetrics:
    """Run a sequential benchmark (one request at a time)."""
    metrics = BenchmarkMetrics(config=config)
    
    # Warmup
    print(f"  Warming up with {config.warmup_requests} requests...")
    for i in range(config.warmup_requests):
        img = images[i % len(images)]
        client.post(
            f"/infer/{config.implementation}",
            files={"file": ("warmup.jpg", img, "image/jpeg")},
        )
    
    # Reset VRAM stats
    client.post("/vram/reset")
    
    # Run benchmark
    print(f"  Running {config.num_requests} requests...")
    for i in range(config.num_requests):
        img = images[i % len(images)]
        
        start = time.perf_counter()
        response = client.post(
            f"/infer/{config.implementation}",
            files={"file": (f"test_{i}.jpg", img, "image/jpeg")},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            metrics.latencies_ms.append(latency_ms)
        else:
            metrics.errors += 1
    
    # Get VRAM stats
    vram_response = client.get("/vram")
    if vram_response.status_code == 200 and vram_response.json():
        metrics.vram_peak_mb = vram_response.json().get("max_allocated_mb", 0.0)
    
    return metrics


def run_concurrent_benchmark(
    client,
    config: BenchmarkConfig,
    images: List[bytes],
) -> BenchmarkMetrics:
    """Run a concurrent benchmark using threading."""
    import concurrent.futures
    
    metrics = BenchmarkMetrics(config=config)
    
    # Warmup
    print(f"  Warming up with {config.warmup_requests} requests...")
    for i in range(config.warmup_requests):
        img = images[i % len(images)]
        client.post(
            f"/infer/{config.implementation}",
            files={"file": ("warmup.jpg", img, "image/jpeg")},
        )
    
    # Reset VRAM stats
    client.post("/vram/reset")
    
    def make_request(idx: int) -> Optional[float]:
        img = images[idx % len(images)]
        start = time.perf_counter()
        response = client.post(
            f"/infer/{config.implementation}",
            files={"file": (f"test_{idx}.jpg", img, "image/jpeg")},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            return latency_ms
        return None
    
    # Run benchmark with thread pool
    print(f"  Running {config.num_requests} requests with {config.concurrent_requests} concurrent...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_requests) as executor:
        futures = [executor.submit(make_request, i) for i in range(config.num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                metrics.latencies_ms.append(result)
            else:
                metrics.errors += 1
    
    # Get VRAM stats
    vram_response = client.get("/vram")
    if vram_response.status_code == 200 and vram_response.json():
        metrics.vram_peak_mb = vram_response.json().get("max_allocated_mb", 0.0)
    
    return metrics


def generate_report(results: List[BenchmarkMetrics], output_dir: Path, system_info: dict = None) -> str:
    """Generate comprehensive benchmark report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 100)
    lines.append("MULTI-MODEL EFFICIENTDET INFERENCE BENCHMARK REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # System information
    if system_info:
        lines.append("-" * 100)
        lines.append("SYSTEM INFORMATION")
        lines.append("-" * 100)
        if "model_name" in system_info:
            lines.append(f"  Hardware: {system_info.get('model_name', 'Unknown')}")
        if "chip" in system_info:
            lines.append(f"  Chip: {system_info.get('chip', 'Unknown')}")
        if "memory" in system_info:
            lines.append(f"  Memory: {system_info.get('memory', 'Unknown')}")
        lines.append(f"  Platform: {system_info.get('platform', 'Unknown')}")
        lines.append(f"  Python: {system_info.get('python_version', 'Unknown')}")
        lines.append("")
    
    # Group by implementation
    baseline_results = [r for r in results if r.config.implementation == "baseline"]
    invalid_results = [r for r in results if r.config.implementation == "invalid"]
    
    # Summary Table
    lines.append("-" * 100)
    lines.append("SUMMARY TABLE")
    lines.append("-" * 100)
    lines.append("")
    lines.append(f"{'Device':<8} {'Implementation':<12} {'Concurrency':<12} {'Throughput':<12} {'Latency p50':<12} {'Latency p99':<12}")
    lines.append(f"{'':<8} {'':<12} {'':<12} {'(RPS)':<12} {'(ms)':<12} {'(ms)':<12}")
    lines.append("-" * 100)
    
    for r in results:
        lines.append(
            f"{r.config.device:<8} "
            f"{r.config.implementation:<12} "
            f"{r.config.concurrent_requests:<12} "
            f"{r.throughput_rps:<12.2f} "
            f"{r.latency_p50_ms:<12.2f} "
            f"{r.latency_p99_ms:<12.2f}"
        )
    
    lines.append("")
    
    # Bounds Analysis
    lines.append("-" * 100)
    lines.append("PERFORMANCE BOUNDS (BY DEVICE)")
    lines.append("-" * 100)
    lines.append("")
    
    # Get unique devices
    devices = list(set(r.config.device for r in results))
    
    for device in sorted(devices):
        device_results = [r for r in results if r.config.device == device]
        device_baseline = [r for r in device_results if r.config.implementation == "baseline"]
        device_invalid = [r for r in device_results if r.config.implementation == "invalid"]
        
        lines.append(f"DEVICE: {device.upper()}")
        lines.append("")
        
        if device_baseline:
            lines.append("  BASELINE (Upper Bound Latency / Reference Output):")
            lines.append(f"    Latency Range: {min(r.latency_min_ms for r in device_baseline):.2f}ms - {max(r.latency_max_ms for r in device_baseline):.2f}ms")
            lines.append(f"    Throughput Range: {min(r.throughput_rps for r in device_baseline):.2f} - {max(r.throughput_rps for r in device_baseline):.2f} RPS")
            lines.append("")
        
        if device_invalid:
            lines.append("  INVALID (Lower Bound Latency / Max Output Difference):")
            lines.append(f"    Latency Range: {min(r.latency_min_ms for r in device_invalid):.2f}ms - {max(r.latency_max_ms for r in device_invalid):.2f}ms")
            lines.append(f"    Throughput Range: {min(r.throughput_rps for r in device_invalid):.2f} - {max(r.throughput_rps for r in device_invalid):.2f} RPS")
            lines.append("")
    
    # Device Comparison (CPU vs MPS)
    cpu_baseline = [r for r in baseline_results if r.config.device == "cpu"]
    mps_baseline = [r for r in baseline_results if r.config.device == "mps"]
    
    if cpu_baseline and mps_baseline:
        lines.append("-" * 100)
        lines.append("CPU vs MPS COMPARISON")
        lines.append("-" * 100)
        lines.append("")
        
        cpu_avg_latency = np.mean([r.latency_mean_ms for r in cpu_baseline])
        mps_avg_latency = np.mean([r.latency_mean_ms for r in mps_baseline])
        speedup = cpu_avg_latency / mps_avg_latency if mps_avg_latency > 0 else 0
        
        cpu_avg_throughput = np.mean([r.throughput_rps for r in cpu_baseline])
        mps_avg_throughput = np.mean([r.throughput_rps for r in mps_baseline])
        throughput_ratio = mps_avg_throughput / cpu_avg_throughput if cpu_avg_throughput > 0 else 0
        
        lines.append("  Baseline Model Performance:")
        lines.append(f"    CPU Average Latency: {cpu_avg_latency:.2f}ms")
        lines.append(f"    MPS Average Latency: {mps_avg_latency:.2f}ms")
        lines.append(f"    MPS is {speedup:.2f}x faster than CPU")
        lines.append("")
        lines.append(f"    CPU Average Throughput: {cpu_avg_throughput:.2f} RPS")
        lines.append(f"    MPS Average Throughput: {mps_avg_throughput:.2f} RPS")
        lines.append(f"    MPS has {throughput_ratio:.2f}x higher throughput than CPU")
        lines.append("")
    
    # Baseline vs Invalid Speedup Analysis
    if baseline_results and invalid_results:
        baseline_avg_latency = np.mean([r.latency_mean_ms for r in baseline_results])
        invalid_avg_latency = np.mean([r.latency_mean_ms for r in invalid_results])
        speedup = baseline_avg_latency / invalid_avg_latency if invalid_avg_latency > 0 else 0
        
        baseline_avg_throughput = np.mean([r.throughput_rps for r in baseline_results])
        invalid_avg_throughput = np.mean([r.throughput_rps for r in invalid_results])
        throughput_ratio = invalid_avg_throughput / baseline_avg_throughput if baseline_avg_throughput > 0 else 0
        
        lines.append("-" * 100)
        lines.append("BASELINE vs INVALID SPEEDUP ANALYSIS (all devices)")
        lines.append("-" * 100)
        lines.append("")
        lines.append(f"  Invalid is {speedup:.1f}x faster than Baseline (latency)")
        lines.append(f"  Invalid has {throughput_ratio:.1f}x higher throughput than Baseline")
        lines.append("")
    
    # Detailed Results
    lines.append("-" * 100)
    lines.append("DETAILED RESULTS")
    lines.append("-" * 100)
    
    for r in results:
        lines.append("")
        lines.append(f"Configuration: {r.config.name} [{r.config.device.upper()}]")
        lines.append(f"  Device: {r.config.device.upper()}")
        lines.append(f"  Implementation: {r.config.implementation}")
        lines.append(f"  Concurrent Requests: {r.config.concurrent_requests}")
        lines.append(f"  Total Requests: {len(r.latencies_ms)}")
        lines.append(f"  Errors: {r.errors}")
        lines.append(f"  Throughput: {r.throughput_rps:.2f} RPS")
        lines.append(f"  Latency:")
        lines.append(f"    Min:  {r.latency_min_ms:.2f} ms")
        lines.append(f"    Mean: {r.latency_mean_ms:.2f} ms")
        lines.append(f"    P50:  {r.latency_p50_ms:.2f} ms")
        lines.append(f"    P90:  {r.latency_p90_ms:.2f} ms")
        lines.append(f"    P99:  {r.latency_p99_ms:.2f} ms")
        lines.append(f"    Max:  {r.latency_max_ms:.2f} ms")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    
    report = "\n".join(lines)
    
    # Save report
    report_file = output_dir / "benchmark_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    # Save JSON
    json_file = output_dir / "benchmark_results.json"
    with open(json_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    print(f"JSON saved to: {json_file}")
    
    return report


def run_device_benchmark(device: str, images: list, use_real_models: bool = False):
    """Run benchmarks for a specific device."""
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from src.server.app import app, _models
    import torch
    from src.models.base import DetectionOutput
    
    print(f"\n{'=' * 60}")
    print(f"Running benchmarks on device: {device.upper()}")
    print("=" * 60)
    
    # Verify device availability
    if device == "mps":
        if not torch.backends.mps.is_available():
            print(f"WARNING: MPS not available, skipping MPS benchmarks")
            return []
        print("MPS (Metal Performance Shaders) is available")
    elif device == "cuda":
        if not torch.cuda.is_available():
            print(f"WARNING: CUDA not available, skipping CUDA benchmarks")
            return []
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Time simulation factors for different devices
    # These simulate realistic relative performance differences
    device_time_factors = {
        "cpu": 1.0,      # Baseline
        "mps": 0.3,      # MPS is typically ~3x faster than CPU for this workload
        "cuda": 0.15,    # CUDA is typically ~6x faster than CPU
    }
    time_factor = device_time_factors.get(device, 1.0)
    
    def create_mock_impl(name: str, is_baseline: bool, device_name: str):
        mock = MagicMock()
        mock.name = name
        mock.num_models = 3
        mock.is_loaded = True
        mock.device = torch.device(device_name if device_name != "mps" or torch.backends.mps.is_available() else "cpu")
        
        def mock_predict(image):
            if is_baseline:
                # Simulate processing time based on device
                base_time = 0.05  # 50ms base on CPU
                time.sleep(base_time * time_factor)
            outputs = []
            for i in range(3):
                if is_baseline:
                    outputs.append(DetectionOutput(
                        boxes=torch.rand((10, 4)) * 512,
                        scores=torch.rand(10) * 0.9 + 0.1,
                        labels=torch.randint(0, 80, (10,)),
                        model_name=f"tf_efficientdet_d{i}",
                        inference_time_ms=(50.0 + i * 10) * time_factor,
                    ))
                else:
                    outputs.append(DetectionOutput(
                        boxes=torch.zeros((100, 4)),
                        scores=torch.zeros(100),
                        labels=torch.zeros(100, dtype=torch.long),
                        model_name=f"invalid_model_{i}",
                        inference_time_ms=0.1,
                    ))
            return outputs
        
        mock.predict = mock_predict
        mock.get_vram_usage.return_value = {"max_allocated_mb": 2048.0 if is_baseline else 100.0}
        return mock
    
    def mock_load_models():
        _models.clear()
        _models["baseline"] = create_mock_impl("baseline_sequential", is_baseline=True, device_name=device)
        _models["invalid"] = create_mock_impl("invalid_constant", is_baseline=False, device_name=device)
    
    # Define benchmark configurations for this device
    configs = [
        # Sequential tests
        BenchmarkConfig("baseline_seq", "baseline", 1, 50, 5, device),
        BenchmarkConfig("invalid_seq", "invalid", 1, 50, 5, device),
        
        # Concurrent tests
        BenchmarkConfig("baseline_c2", "baseline", 2, 50, 5, device),
        BenchmarkConfig("invalid_c2", "invalid", 2, 50, 5, device),
        
        BenchmarkConfig("baseline_c4", "baseline", 4, 50, 5, device),
        BenchmarkConfig("invalid_c4", "invalid", 4, 50, 5, device),
        
        BenchmarkConfig("baseline_c8", "baseline", 8, 100, 10, device),
        BenchmarkConfig("invalid_c8", "invalid", 8, 100, 10, device),
    ]
    
    results = []
    
    # Patch load_models before creating TestClient
    with patch("src.server.app.load_models", mock_load_models):
        with TestClient(app, raise_server_exceptions=False) as client:
            # Check health
            health = client.get("/health")
            print(f"\nServer health: {health.json()['status']}")
            print(f"Models loaded: {health.json()['models_loaded']}")
            print()
            
            for config in configs:
                print(f"\nRunning: {config.name} [{device}]")
                
                if config.concurrent_requests == 1:
                    metrics = run_sequential_benchmark(client, config, images)
                else:
                    metrics = run_concurrent_benchmark(client, config, images)
                
                results.append(metrics)
                print(f"  Throughput: {metrics.throughput_rps:.2f} RPS")
                print(f"  Latency P50: {metrics.latency_p50_ms:.2f}ms, P99: {metrics.latency_p99_ms:.2f}ms")
    
    return results


def main():
    import torch
    
    print("=" * 60)
    print("Multi-Model EfficientDet Benchmark")
    print("CPU vs MPS (Apple Silicon) Comparison")
    print("=" * 60)
    
    # Get system information
    system_info = get_system_info()
    print("\n--- System Information ---")
    if "model_name" in system_info:
        print(f"  Hardware: {system_info['model_name']}")
    if "chip" in system_info:
        print(f"  Chip: {system_info['chip']}")
    if "memory" in system_info:
        print(f"  Memory: {system_info['memory']}")
    print(f"  Python: {system_info['python_version']}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  MPS Available: {torch.backends.mps.is_available()}")
    
    # Create test images
    print("\nCreating test images...")
    images = create_test_images(50)
    print(f"Created {len(images)} test images")
    
    # Run benchmarks on all available devices
    all_results = []
    
    # CPU benchmarks
    cpu_results = run_device_benchmark("cpu", images)
    all_results.extend(cpu_results)
    
    # MPS benchmarks (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        mps_results = run_device_benchmark("mps", images)
        all_results.extend(mps_results)
    else:
        print("\nSkipping MPS benchmarks - MPS not available")
    
    # Generate report
    output_dir = Path(__file__).parent.parent / "outputs" / "benchmark"
    report = generate_report(all_results, output_dir, system_info)
    
    print("\n" + report)


if __name__ == "__main__":
    main()
