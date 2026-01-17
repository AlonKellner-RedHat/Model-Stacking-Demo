#!/usr/bin/env python3
"""Run ablation study on optimization configurations.

Tests various combinations of optimizations to measure their individual
and combined effects on inference performance.

Usage:
    uv run python scripts/run_ablation.py
"""

import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.optimizations.base import OptimizationConfig
from src.models.optimized import OptimizedImpl


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    config_name: str
    config: OptimizationConfig
    device: str
    latencies_ms: List[float] = field(default_factory=list)
    
    @property
    def mean_latency_ms(self) -> float:
        return np.mean(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def p50_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0
    
    @property
    def p99_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0
    
    @property
    def throughput_rps(self) -> float:
        if not self.latencies_ms:
            return 0
        total_time = sum(self.latencies_ms) / 1000
        return len(self.latencies_ms) / total_time if total_time > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "config": str(self.config),
            "device": self.device,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_rps": self.throughput_rps,
            "num_requests": len(self.latencies_ms),
        }


def create_test_images(num_images: int = 20) -> List[bytes]:
    """Create synthetic test images."""
    images = []
    for i in range(num_images):
        np.random.seed(i)
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        images.append(buf.getvalue())
    return images


def get_ablation_configs() -> Dict[str, OptimizationConfig]:
    """Get all optimization configurations for ablation study."""
    configs = {
        # Baseline (no optimizations)
        "baseline": OptimizationConfig(),
        
        # Individual optimizations
        "compile_default": OptimizationConfig(
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="default",
        ),
        "compile_reduce_overhead": OptimizationConfig(
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="reduce-overhead",
        ),
        "fp16": OptimizationConfig(
            mixed_precision_enabled=True,
            dtype=torch.float16,
        ),
        
        # Combinations
        "compile+fp16": OptimizationConfig(
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="default",
            mixed_precision_enabled=True,
            dtype=torch.float16,
        ),
        "compile_ro+fp16": OptimizationConfig(
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="reduce-overhead",
            mixed_precision_enabled=True,
            dtype=torch.float16,
        ),
    }
    
    return configs


def run_ablation_benchmark(
    config_name: str,
    config: OptimizationConfig,
    device: str,
    images: List[bytes],
    num_warmup: int = 5,
    num_requests: int = 20,
) -> AblationResult:
    """Run benchmark for a single configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Config: {config}")
    print(f"Device: {device}")
    print("=" * 60)
    
    result = AblationResult(
        config_name=config_name,
        config=config,
        device=device,
    )
    
    try:
        # Create and load implementation
        impl = OptimizedImpl(device=device, optimization_config=config)
        impl.load()
        
        # Convert image bytes to PIL Images
        pil_images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in images]
        
        # Warmup
        print(f"Warming up with {num_warmup} requests...")
        for i in range(num_warmup):
            img = pil_images[i % len(pil_images)]
            impl.predict(img)
        
        # Benchmark
        print(f"Running {num_requests} requests...")
        for i in range(num_requests):
            img = pil_images[i % len(pil_images)]
            
            # Time the full prediction (all 3 models)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            
            start = time.perf_counter()
            outputs = impl.predict(img)
            
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            result.latencies_ms.append(latency_ms)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{num_requests}")
        
        print(f"\nResults:")
        print(f"  Mean latency: {result.mean_latency_ms:.2f} ms")
        print(f"  P50 latency:  {result.p50_latency_ms:.2f} ms")
        print(f"  P99 latency:  {result.p99_latency_ms:.2f} ms")
        print(f"  Throughput:   {result.throughput_rps:.2f} RPS")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def generate_ablation_report(results: List[AblationResult], output_dir: Path) -> str:
    """Generate ablation study report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("ABLATION STUDY: OPTIMIZATION CONFIGURATIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary table
    lines.append("-" * 80)
    lines.append("SUMMARY TABLE")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"{'Config':<25} {'Device':<8} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P99 (ms)':<12} {'RPS':<10}")
    lines.append("-" * 80)
    
    # Sort by mean latency
    sorted_results = sorted(results, key=lambda r: r.mean_latency_ms if r.mean_latency_ms > 0 else float('inf'))
    
    baseline_latency = None
    for r in sorted_results:
        if r.config_name == "baseline" and r.mean_latency_ms > 0:
            baseline_latency = r.mean_latency_ms
            break
    
    for r in sorted_results:
        if r.mean_latency_ms > 0:
            speedup = f"({baseline_latency/r.mean_latency_ms:.2f}x)" if baseline_latency else ""
            lines.append(
                f"{r.config_name:<25} {r.device:<8} "
                f"{r.mean_latency_ms:<12.2f} {r.p50_latency_ms:<12.2f} "
                f"{r.p99_latency_ms:<12.2f} {r.throughput_rps:<10.2f} {speedup}"
            )
        else:
            lines.append(f"{r.config_name:<25} {r.device:<8} {'FAILED':<12}")
    
    lines.append("")
    
    # Speedup analysis
    if baseline_latency:
        lines.append("-" * 80)
        lines.append("SPEEDUP vs BASELINE")
        lines.append("-" * 80)
        lines.append("")
        
        for r in sorted_results:
            if r.mean_latency_ms > 0 and r.config_name != "baseline":
                speedup = baseline_latency / r.mean_latency_ms
                improvement = (1 - r.mean_latency_ms / baseline_latency) * 100
                lines.append(f"  {r.config_name}: {speedup:.2f}x faster ({improvement:.1f}% improvement)")
        lines.append("")
    
    report = "\n".join(lines)
    
    # Save report
    report_file = output_dir / "ablation_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    # Save JSON
    json_file = output_dir / "ablation_results.json"
    with open(json_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    print(f"JSON saved to: {json_file}")
    
    return report


def main():
    print("=" * 60)
    print("Ablation Study: Optimization Configurations")
    print("=" * 60)
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Using device: CPU")
    
    # Create test images
    print("\nCreating test images...")
    images = create_test_images(20)
    print(f"Created {len(images)} test images")
    
    # Get configurations
    configs = get_ablation_configs()
    print(f"\nTesting {len(configs)} configurations:")
    for name in configs:
        print(f"  - {name}")
    
    # Run benchmarks
    results = []
    for config_name, config in configs.items():
        result = run_ablation_benchmark(
            config_name=config_name,
            config=config,
            device=device,
            images=images,
            num_warmup=5,
            num_requests=20,
        )
        results.append(result)
    
    # Generate report
    output_dir = Path(__file__).parent.parent / "outputs" / "ablation"
    report = generate_ablation_report(results, output_dir)
    print("\n" + report)


if __name__ == "__main__":
    main()
