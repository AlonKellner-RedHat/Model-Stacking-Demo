#!/usr/bin/env python3
"""Run ablation study on optimization configurations.

Tests various combinations of optimizations to measure their individual
and combined effects on inference performance AND output correctness.

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
from src.models.base import DetectionOutput
from src.benchmark.metrics import (
    ComparisonMetrics, 
    compare_outputs, 
    aggregate_comparison_metrics
)


@dataclass
class OutputComparisonResult:
    """Aggregated output comparison metrics."""
    boxes_mse: float = 0.0
    scores_mse: float = 0.0
    labels_accuracy: float = 1.0
    iou_mean: float = 1.0
    num_detections_diff: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "boxes_mse": self.boxes_mse,
            "scores_mse": self.scores_mse,
            "labels_accuracy": self.labels_accuracy,
            "iou_mean": self.iou_mean,
            "num_detections_diff": self.num_detections_diff,
        }


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    config_name: str
    config: OptimizationConfig
    device: str
    latencies_ms: List[float] = field(default_factory=list)
    output_comparison: Optional[OutputComparisonResult] = None
    
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
        result = {
            "config_name": self.config_name,
            "config": str(self.config),
            "device": self.device,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_rps": self.throughput_rps,
            "num_requests": len(self.latencies_ms),
        }
        if self.output_comparison:
            result["output_comparison"] = self.output_comparison.to_dict()
        return result


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
        
        # Vmap backbone optimization (includes compile internally)
        # Parallelizes backbone+FPN+box_net across models using torch.vmap
        "vmap_backbone": OptimizationConfig(
            vmap_backbone_enabled=True,
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
        # Vmap with FP16 (likely slow on MPS due to FP16 overhead)
        "vmap+fp16": OptimizationConfig(
            vmap_backbone_enabled=True,
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
    reference_outputs: Optional[Dict[int, List[DetectionOutput]]] = None,
    num_warmup: int = 5,
    num_requests: int = 20,
) -> AblationResult:
    """Run benchmark for a single configuration.
    
    Args:
        config_name: Name of the configuration
        config: Optimization configuration
        device: Device to run on
        images: List of image bytes
        reference_outputs: Dict mapping image index to reference outputs (for comparison)
        num_warmup: Number of warmup requests
        num_requests: Number of benchmark requests
    """
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
        
        # Benchmark with output collection for comparison
        print(f"Running {num_requests} requests...")
        all_comparison_metrics = []
        
        for i in range(num_requests):
            img_idx = i % len(pil_images)
            img = pil_images[img_idx]
            
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
            
            # Compare outputs to reference (if available)
            if reference_outputs and img_idx in reference_outputs:
                ref_outputs = reference_outputs[img_idx]
                for pred, ref in zip(outputs, ref_outputs):
                    metrics = compare_outputs(pred, ref)
                    all_comparison_metrics.append(metrics)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{num_requests}")
        
        # Aggregate comparison metrics
        if all_comparison_metrics:
            agg = aggregate_comparison_metrics(all_comparison_metrics)
            result.output_comparison = OutputComparisonResult(
                boxes_mse=agg.boxes_mse,
                scores_mse=agg.scores_mse,
                labels_accuracy=agg.labels_accuracy,
                iou_mean=agg.iou_mean,
                num_detections_diff=agg.num_detections_diff,
            )
        
        print(f"\nResults:")
        print(f"  Mean latency: {result.mean_latency_ms:.2f} ms")
        print(f"  P50 latency:  {result.p50_latency_ms:.2f} ms")
        print(f"  P99 latency:  {result.p99_latency_ms:.2f} ms")
        print(f"  Throughput:   {result.throughput_rps:.2f} RPS")
        
        if result.output_comparison:
            print(f"  Output Comparison (vs baseline):")
            print(f"    Boxes MSE:     {result.output_comparison.boxes_mse:.6f}")
            print(f"    Scores MSE:    {result.output_comparison.scores_mse:.6f}")
            print(f"    Labels Acc:    {result.output_comparison.labels_accuracy:.4f}")
            print(f"    Mean IoU:      {result.output_comparison.iou_mean:.4f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def generate_ablation_report(results: List[AblationResult], output_dir: Path) -> str:
    """Generate ablation study report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 100)
    lines.append("ABLATION STUDY: OPTIMIZATION CONFIGURATIONS")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Performance Summary table
    lines.append("-" * 100)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 100)
    lines.append("")
    lines.append(f"{'Config':<25} {'Device':<8} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P99 (ms)':<12} {'RPS':<10}")
    lines.append("-" * 100)
    
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
    
    # Output Difference Summary
    lines.append("-" * 100)
    lines.append("OUTPUT DIFFERENCE (vs Baseline Reference)")
    lines.append("-" * 100)
    lines.append("")
    lines.append(f"{'Config':<25} {'Boxes MSE':<15} {'Scores MSE':<15} {'Labels Acc':<15} {'Mean IoU':<15}")
    lines.append("-" * 100)
    
    for r in sorted_results:
        if r.output_comparison:
            oc = r.output_comparison
            lines.append(
                f"{r.config_name:<25} "
                f"{oc.boxes_mse:<15.6f} "
                f"{oc.scores_mse:<15.6f} "
                f"{oc.labels_accuracy:<15.4f} "
                f"{oc.iou_mean:<15.4f}"
            )
        else:
            lines.append(f"{r.config_name:<25} {'N/A':<15}")
    
    lines.append("")
    lines.append("Note: Baseline compared to itself should have 0 MSE and 1.0 accuracy/IoU")
    lines.append("      Non-zero values for other configs indicate numerical differences from optimizations")
    lines.append("")
    
    # Speedup analysis
    if baseline_latency:
        lines.append("-" * 100)
        lines.append("SPEEDUP vs BASELINE")
        lines.append("-" * 100)
        lines.append("")
        
        for r in sorted_results:
            if r.mean_latency_ms > 0 and r.config_name != "baseline":
                speedup = baseline_latency / r.mean_latency_ms
                improvement = (1 - r.mean_latency_ms / baseline_latency) * 100
                
                # Add output quality note
                quality_note = ""
                if r.output_comparison:
                    if r.output_comparison.iou_mean >= 0.99:
                        quality_note = " [exact match]"
                    elif r.output_comparison.iou_mean >= 0.95:
                        quality_note = " [near-exact]"
                    elif r.output_comparison.iou_mean >= 0.8:
                        quality_note = " [minor diff]"
                    else:
                        quality_note = " [significant diff!]"
                
                lines.append(f"  {r.config_name}: {speedup:.2f}x faster ({improvement:.1f}% improvement){quality_note}")
        lines.append("")
    
    # Key findings
    lines.append("-" * 100)
    lines.append("KEY FINDINGS")
    lines.append("-" * 100)
    lines.append("")
    
    # Find best performing config with acceptable output quality
    best_config = None
    for r in sorted_results:
        if r.mean_latency_ms > 0 and r.config_name != "baseline":
            if r.output_comparison and r.output_comparison.iou_mean >= 0.95:
                best_config = r
                break
    
    if best_config:
        speedup = baseline_latency / best_config.mean_latency_ms if baseline_latency else 1.0
        lines.append(f"  Best config (with quality): {best_config.config_name}")
        lines.append(f"    Latency: {best_config.mean_latency_ms:.2f}ms ({speedup:.2f}x faster)")
        lines.append(f"    Output Quality: IoU={best_config.output_comparison.iou_mean:.4f}")
    
    lines.append("")
    lines.append("=" * 100)
    
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


def generate_reference_outputs(
    device: str, 
    images: List[bytes]
) -> Dict[int, List[DetectionOutput]]:
    """Generate reference outputs using baseline (no optimizations).
    
    Returns:
        Dict mapping image index to list of DetectionOutput (one per model)
    """
    print("\n" + "=" * 60)
    print("Generating reference outputs (baseline, no optimizations)")
    print("=" * 60)
    
    impl = OptimizedImpl(device=device, optimization_config=OptimizationConfig())
    impl.load()
    
    pil_images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in images]
    
    reference_outputs = {}
    for i, img in enumerate(pil_images):
        outputs = impl.predict(img)
        reference_outputs[i] = outputs
        if (i + 1) % 5 == 0:
            print(f"  Generated reference for {i+1}/{len(pil_images)} images")
    
    print(f"Generated reference outputs for {len(reference_outputs)} images")
    return reference_outputs


def main():
    print("=" * 60)
    print("Ablation Study: Optimization Configurations")
    print("With Output Difference Metrics (MSE, IoU)")
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
    
    # Generate reference outputs from baseline
    reference_outputs = generate_reference_outputs(device, images)
    
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
            reference_outputs=reference_outputs,
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
