#!/usr/bin/env python3
"""Orchestrate benchmark runs for all implementations."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.metrics import BenchmarkResult, ComparisonMetrics, generate_report


def wait_for_server(host: str, timeout: float = 60.0) -> bool:
    """Wait for the server to be ready.
    
    Args:
        host: Server host URL
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if server is ready, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{host}/health", timeout=5.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def run_locust_benchmark(
    host: str,
    implementation: str,
    users: int,
    spawn_rate: int,
    duration: int,
    data_dir: Path,
    output_dir: Path,
) -> Optional[BenchmarkResult]:
    """Run a Locust benchmark.
    
    Args:
        host: Server host URL
        implementation: Implementation to benchmark
        users: Number of concurrent users
        spawn_rate: Users spawned per second
        duration: Test duration in seconds
        data_dir: Dataset directory
        output_dir: Output directory for results
        
    Returns:
        BenchmarkResult or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = output_dir / f"{implementation}_u{users}"
    
    # Set environment variables for locustfile
    env = {
        "BENCHMARK_DATA_DIR": str(data_dir),
        "BENCHMARK_IMPLEMENTATION": implementation,
        "BENCHMARK_MAX_IMAGES": "100",
    }
    
    # Build locust command
    cmd = [
        sys.executable, "-m", "locust",
        "-f", "src/benchmark/locustfile.py",
        "--host", host,
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "-t", f"{duration}s",
        "--csv", str(csv_prefix),
        "--csv-full-history",
    ]
    
    print(f"\nRunning benchmark: {implementation} with {users} users for {duration}s")
    print(f"Command: {' '.join(cmd)}")
    
    # Run locust
    import os
    full_env = os.environ.copy()
    full_env.update(env)
    
    result = subprocess.run(
        cmd,
        env=full_env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    
    if result.returncode != 0:
        print(f"Locust failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return None
    
    # Parse CSV results
    stats_file = Path(f"{csv_prefix}_stats.csv")
    if not stats_file.exists():
        print(f"Stats file not found: {stats_file}")
        return None
    
    # Read stats CSV
    import csv
    with open(stats_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find the aggregated row
    agg_row = None
    for row in rows:
        if row.get("Name") == "Aggregated":
            agg_row = row
            break
    
    if not agg_row:
        print("No aggregated stats found")
        return None
    
    # Get VRAM stats from server
    vram_peak = 0.0
    try:
        response = httpx.get(f"{host}/vram", timeout=5.0)
        if response.status_code == 200 and response.json():
            vram_peak = response.json().get("max_allocated_mb", 0.0)
    except Exception:
        pass
    
    # Build result
    return BenchmarkResult(
        implementation=implementation,
        num_requests=int(agg_row.get("Request Count", 0)),
        duration_seconds=duration,
        throughput_rps=float(agg_row.get("Requests/s", 0)),
        latency_mean_ms=float(agg_row.get("Average Response Time", 0)),
        latency_p50_ms=float(agg_row.get("50%", 0)),
        latency_p90_ms=float(agg_row.get("90%", 0)),
        latency_p99_ms=float(agg_row.get("99%", 0)),
        vram_peak_mb=vram_peak,
        errors=int(agg_row.get("Failure Count", 0)),
    )


def run_direct_comparison(
    host: str,
    data_dir: Path,
    num_images: int = 50,
) -> dict:
    """Run direct output comparison between implementations.
    
    Args:
        host: Server host URL
        data_dir: Dataset directory
        num_images: Number of images to compare
        
    Returns:
        Dictionary with comparison results
    """
    from src.datasets import DatasetLoader
    from src.benchmark.metrics import compare_all_outputs, aggregate_comparison_metrics
    from src.models.base import DetectionOutput
    
    print(f"\nRunning output comparison with {num_images} images...")
    
    # Load images
    loader = DatasetLoader(data_dir)
    loader.add_all_datasets(max_images_per_dataset=num_images)
    
    if len(loader) == 0:
        print("No images found for comparison")
        return {}
    
    baseline_outputs = []
    invalid_outputs = []
    
    for item in loader:
        image_bytes = item.load_bytes()
        
        # Get baseline output
        try:
            response = httpx.post(
                f"{host}/infer/baseline",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                timeout=60.0,
            )
            if response.status_code == 200:
                data = response.json()
                outputs = [
                    DetectionOutput.from_dict({
                        "boxes": d["boxes"],
                        "scores": d["scores"],
                        "labels": d["labels"],
                        "model_name": d["model_name"],
                        "inference_time_ms": d["inference_time_ms"],
                    })
                    for d in data["detections"]
                ]
                baseline_outputs.append(outputs)
        except Exception as e:
            print(f"Failed to get baseline output: {e}")
            continue
        
        # Get invalid output
        try:
            response = httpx.post(
                f"{host}/infer/invalid",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                timeout=60.0,
            )
            if response.status_code == 200:
                data = response.json()
                outputs = [
                    DetectionOutput.from_dict({
                        "boxes": d["boxes"],
                        "scores": d["scores"],
                        "labels": d["labels"],
                        "model_name": d["model_name"],
                        "inference_time_ms": d["inference_time_ms"],
                    })
                    for d in data["detections"]
                ]
                invalid_outputs.append(outputs)
        except Exception as e:
            print(f"Failed to get invalid output: {e}")
            continue
    
    # Compare outputs
    all_metrics = []
    for baseline, invalid in zip(baseline_outputs, invalid_outputs):
        # Compare each model's output
        for b, i in zip(baseline, invalid):
            from src.benchmark.metrics import compare_outputs
            metrics = compare_outputs(i, b)
            all_metrics.append(metrics)
    
    if all_metrics:
        aggregated = aggregate_comparison_metrics(all_metrics)
        print(f"Output comparison complete:")
        print(f"  Boxes MSE: {aggregated.boxes_mse:.6f}")
        print(f"  Scores MSE: {aggregated.scores_mse:.6f}")
        print(f"  Labels Accuracy: {aggregated.labels_accuracy:.4f}")
        return aggregated.to_dict()
    
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="Server host URL"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Concurrent user counts to test"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration per configuration (seconds)"
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=2,
        help="User spawn rate per second"
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup phase"
    )
    parser.add_argument(
        "--compare-outputs",
        action="store_true",
        help="Run output comparison between implementations"
    )
    
    args = parser.parse_args()
    
    # Check server
    print(f"Checking server at {args.host}...")
    if not wait_for_server(args.host, timeout=10.0):
        print("Server not ready. Please start the server first:")
        print("  uv run uvicorn src.server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print("Server is ready!")
    
    # Warmup
    if not args.skip_warmup:
        print("\nRunning warmup...")
        try:
            # Create a simple test image
            from PIL import Image
            from io import BytesIO
            img = Image.new("RGB", (512, 512), color=(128, 128, 128))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            for _ in range(10):
                httpx.post(
                    f"{args.host}/infer/baseline",
                    files={"file": ("warmup.jpg", image_bytes, "image/jpeg")},
                    timeout=60.0,
                )
                httpx.post(
                    f"{args.host}/infer/invalid",
                    files={"file": ("warmup.jpg", image_bytes, "image/jpeg")},
                    timeout=60.0,
                )
            print("Warmup complete!")
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    # Reset VRAM stats
    try:
        httpx.post(f"{args.host}/vram/reset", timeout=5.0)
    except Exception:
        pass
    
    # Run benchmarks
    results: List[BenchmarkResult] = []
    
    implementations = ["baseline", "invalid"]
    
    for impl in implementations:
        for users in args.users:
            result = run_locust_benchmark(
                host=args.host,
                implementation=impl,
                users=users,
                spawn_rate=args.spawn_rate,
                duration=args.duration,
                data_dir=args.data_dir,
                output_dir=args.output_dir / impl,
            )
            if result:
                results.append(result)
    
    # Run output comparison
    if args.compare_outputs and len(results) > 0:
        comparison = run_direct_comparison(args.host, args.data_dir)
        # Add comparison metrics to invalid results
        if comparison:
            for result in results:
                if result.implementation == "invalid":
                    result.comparison_metrics = ComparisonMetrics(**comparison)
    
    # Generate report
    if results:
        report = generate_report(results, args.output_dir / "report.txt")
        print("\n" + report)
    else:
        print("No benchmark results collected")


if __name__ == "__main__":
    main()
