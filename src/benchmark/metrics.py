"""Metrics utilities for VRAM monitoring, output comparison, and reporting."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.base import DetectionOutput


@dataclass
class ComparisonMetrics:
    """Metrics comparing predicted outputs to reference outputs.
    
    Attributes:
        boxes_mse: Mean Squared Error between bounding boxes
        scores_mse: Mean Squared Error between confidence scores
        labels_accuracy: Accuracy of label predictions
        num_detections_diff: Difference in number of detections
        iou_mean: Mean Intersection over Union (if boxes overlap)
    """
    boxes_mse: float
    scores_mse: float
    labels_accuracy: float
    num_detections_diff: int
    iou_mean: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "boxes_mse": self.boxes_mse,
            "scores_mse": self.scores_mse,
            "labels_accuracy": self.labels_accuracy,
            "num_detections_diff": self.num_detections_diff,
            "iou_mean": self.iou_mean,
        }


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.
    
    Attributes:
        implementation: Name of the implementation tested
        num_requests: Total number of requests
        duration_seconds: Total duration of the test
        throughput_rps: Requests per second
        latency_mean_ms: Mean latency in milliseconds
        latency_p50_ms: 50th percentile latency
        latency_p90_ms: 90th percentile latency
        latency_p99_ms: 99th percentile latency
        vram_peak_mb: Peak VRAM usage in MB
        comparison_metrics: Output comparison metrics (if available)
    """
    implementation: str
    num_requests: int
    duration_seconds: float
    throughput_rps: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    vram_peak_mb: float = 0.0
    comparison_metrics: Optional[ComparisonMetrics] = None
    errors: int = 0
    
    def to_dict(self) -> dict:
        result = {
            "implementation": self.implementation,
            "num_requests": self.num_requests,
            "duration_seconds": self.duration_seconds,
            "throughput_rps": self.throughput_rps,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "vram_peak_mb": self.vram_peak_mb,
            "errors": self.errors,
        }
        if self.comparison_metrics:
            result["comparison_metrics"] = self.comparison_metrics.to_dict()
        return result


def compute_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes in xyxy format.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        Intersection over Union value
    """
    # Intersection
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1.item() + area2.item() - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compare_outputs(
    predicted: DetectionOutput,
    reference: DetectionOutput,
    score_threshold: float = 0.1,
) -> ComparisonMetrics:
    """Compare predicted outputs to reference outputs.
    
    Args:
        predicted: Predicted detection output
        reference: Reference detection output
        score_threshold: Minimum score to consider a detection
        
    Returns:
        ComparisonMetrics with MSE and accuracy values
    """
    # Filter by score threshold
    pred_mask = predicted.scores >= score_threshold
    ref_mask = reference.scores >= score_threshold
    
    pred_boxes = predicted.boxes[pred_mask]
    pred_scores = predicted.scores[pred_mask]
    pred_labels = predicted.labels[pred_mask]
    
    ref_boxes = reference.boxes[ref_mask]
    ref_scores = reference.scores[ref_mask]
    ref_labels = reference.labels[ref_mask]
    
    num_pred = len(pred_boxes)
    num_ref = len(ref_boxes)
    
    # Handle empty cases
    if num_pred == 0 and num_ref == 0:
        return ComparisonMetrics(
            boxes_mse=0.0,
            scores_mse=0.0,
            labels_accuracy=1.0,
            num_detections_diff=0,
            iou_mean=1.0,
        )
    
    if num_pred == 0 or num_ref == 0:
        return ComparisonMetrics(
            boxes_mse=float('inf'),
            scores_mse=float('inf'),
            labels_accuracy=0.0,
            num_detections_diff=abs(num_pred - num_ref),
            iou_mean=0.0,
        )
    
    # Pad to same length for comparison
    max_len = max(num_pred, num_ref)
    
    def pad_tensor(t: torch.Tensor, target_len: int, fill_value: float = 0.0) -> torch.Tensor:
        if len(t) >= target_len:
            return t[:target_len]
        padding = torch.full(
            (target_len - len(t),) + t.shape[1:],
            fill_value,
            dtype=t.dtype,
            device=t.device
        )
        return torch.cat([t, padding], dim=0)
    
    # Pad tensors
    pred_boxes_padded = pad_tensor(pred_boxes, max_len)
    ref_boxes_padded = pad_tensor(ref_boxes, max_len)
    pred_scores_padded = pad_tensor(pred_scores, max_len)
    ref_scores_padded = pad_tensor(ref_scores, max_len)
    pred_labels_padded = pad_tensor(pred_labels.float(), max_len).long()
    ref_labels_padded = pad_tensor(ref_labels.float(), max_len).long()
    
    # Compute MSE for boxes and scores
    boxes_mse = torch.mean((pred_boxes_padded - ref_boxes_padded) ** 2).item()
    scores_mse = torch.mean((pred_scores_padded - ref_scores_padded) ** 2).item()
    
    # Compute label accuracy (on overlapping length)
    min_len = min(num_pred, num_ref)
    if min_len > 0:
        labels_match = (pred_labels[:min_len] == ref_labels[:min_len]).float()
        labels_accuracy = labels_match.mean().item()
    else:
        labels_accuracy = 0.0
    
    # Compute mean IoU for matched boxes
    ious = []
    for i in range(min(num_pred, num_ref)):
        iou = compute_box_iou(pred_boxes[i], ref_boxes[i])
        ious.append(iou)
    iou_mean = np.mean(ious) if ious else 0.0
    
    return ComparisonMetrics(
        boxes_mse=boxes_mse,
        scores_mse=scores_mse,
        labels_accuracy=labels_accuracy,
        num_detections_diff=abs(num_pred - num_ref),
        iou_mean=iou_mean,
    )


def compare_all_outputs(
    predicted_list: List[DetectionOutput],
    reference_list: List[DetectionOutput],
    score_threshold: float = 0.1,
) -> List[ComparisonMetrics]:
    """Compare multiple predicted outputs to references.
    
    Args:
        predicted_list: List of predicted outputs (one per model)
        reference_list: List of reference outputs (one per model)
        score_threshold: Minimum score threshold
        
    Returns:
        List of ComparisonMetrics, one per model
    """
    if len(predicted_list) != len(reference_list):
        raise ValueError(
            f"Mismatched number of outputs: {len(predicted_list)} vs {len(reference_list)}"
        )
    
    metrics = []
    for pred, ref in zip(predicted_list, reference_list):
        m = compare_outputs(pred, ref, score_threshold)
        metrics.append(m)
    
    return metrics


def aggregate_comparison_metrics(metrics_list: List[ComparisonMetrics]) -> ComparisonMetrics:
    """Aggregate multiple comparison metrics into a single summary.
    
    Args:
        metrics_list: List of ComparisonMetrics to aggregate
        
    Returns:
        Aggregated ComparisonMetrics with mean values
    """
    if not metrics_list:
        return ComparisonMetrics(
            boxes_mse=0.0,
            scores_mse=0.0,
            labels_accuracy=1.0,
            num_detections_diff=0,
            iou_mean=1.0,
        )
    
    return ComparisonMetrics(
        boxes_mse=np.mean([m.boxes_mse for m in metrics_list if not np.isinf(m.boxes_mse)]),
        scores_mse=np.mean([m.scores_mse for m in metrics_list if not np.isinf(m.scores_mse)]),
        labels_accuracy=np.mean([m.labels_accuracy for m in metrics_list]),
        num_detections_diff=int(np.mean([m.num_detections_diff for m in metrics_list])),
        iou_mean=np.mean([m.iou_mean for m in metrics_list]),
    )


def load_reference_outputs(ref_file: Path) -> List[DetectionOutput]:
    """Load reference outputs from a JSON file.
    
    Args:
        ref_file: Path to reference JSON file
        
    Returns:
        List of DetectionOutput objects
    """
    with open(ref_file, "r") as f:
        data = json.load(f)
    
    outputs = []
    for out_data in data["outputs"]:
        outputs.append(DetectionOutput.from_dict(out_data))
    
    return outputs


def generate_report(
    results: List[BenchmarkResult],
    output_file: Optional[Path] = None,
) -> str:
    """Generate a benchmark report.
    
    Args:
        results: List of BenchmarkResult objects
        output_file: Optional file to write report to
        
    Returns:
        Report as a string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary table header
    lines.append(f"{'Implementation':<20} {'Throughput':>12} {'Latency (ms)':>15} {'VRAM (MB)':>12}")
    lines.append(f"{'':<20} {'(RPS)':>12} {'p50 / p99':>15} {'Peak':>12}")
    lines.append("-" * 80)
    
    for result in results:
        latency_str = f"{result.latency_p50_ms:.1f} / {result.latency_p99_ms:.1f}"
        lines.append(
            f"{result.implementation:<20} "
            f"{result.throughput_rps:>12.2f} "
            f"{latency_str:>15} "
            f"{result.vram_peak_mb:>12.1f}"
        )
    
    lines.append("")
    lines.append("-" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("-" * 80)
    
    for result in results:
        lines.append("")
        lines.append(f"Implementation: {result.implementation}")
        lines.append(f"  Requests: {result.num_requests}")
        lines.append(f"  Duration: {result.duration_seconds:.2f}s")
        lines.append(f"  Throughput: {result.throughput_rps:.2f} RPS")
        lines.append(f"  Latency:")
        lines.append(f"    Mean: {result.latency_mean_ms:.2f}ms")
        lines.append(f"    P50:  {result.latency_p50_ms:.2f}ms")
        lines.append(f"    P90:  {result.latency_p90_ms:.2f}ms")
        lines.append(f"    P99:  {result.latency_p99_ms:.2f}ms")
        lines.append(f"  VRAM Peak: {result.vram_peak_mb:.1f}MB")
        lines.append(f"  Errors: {result.errors}")
        
        if result.comparison_metrics:
            m = result.comparison_metrics
            lines.append(f"  Output Comparison:")
            lines.append(f"    Boxes MSE: {m.boxes_mse:.6f}")
            lines.append(f"    Scores MSE: {m.scores_mse:.6f}")
            lines.append(f"    Labels Accuracy: {m.labels_accuracy:.4f}")
            lines.append(f"    Detection Count Diff: {m.num_detections_diff}")
            lines.append(f"    Mean IoU: {m.iou_mean:.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)
        
        # Also save JSON version
        json_file = output_file.with_suffix(".json")
        with open(json_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    
    return report
