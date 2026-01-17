"""Unit tests for benchmark metrics calculations and reporting."""

import json
from pathlib import Path

import pytest
import torch

from src.benchmark.metrics import (
    ComparisonMetrics,
    BenchmarkResult,
    compute_box_iou,
    compare_outputs,
    compare_all_outputs,
    aggregate_comparison_metrics,
    generate_report,
    load_reference_outputs,
)
from src.models.base import DetectionOutput


class TestComparisonMetrics:
    """Tests for ComparisonMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        metrics = ComparisonMetrics(
            boxes_mse=10.5,
            scores_mse=0.25,
            labels_accuracy=0.9,
            num_detections_diff=3,
            iou_mean=0.75,
        )
        
        assert metrics.boxes_mse == 10.5
        assert metrics.scores_mse == 0.25
        assert metrics.labels_accuracy == 0.9
        assert metrics.num_detections_diff == 3
        assert metrics.iou_mean == 0.75

    def test_to_dict(self, sample_comparison_metrics):
        """Test serialization to dict."""
        result = sample_comparison_metrics.to_dict()
        
        assert isinstance(result, dict)
        assert "boxes_mse" in result
        assert "scores_mse" in result
        assert "labels_accuracy" in result
        assert "num_detections_diff" in result
        assert "iou_mean" in result


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = BenchmarkResult(
            implementation="test",
            num_requests=100,
            duration_seconds=30.0,
            throughput_rps=3.33,
            latency_mean_ms=100.0,
            latency_p50_ms=90.0,
            latency_p90_ms=150.0,
            latency_p99_ms=200.0,
        )
        
        assert result.implementation == "test"
        assert result.num_requests == 100
        assert result.errors == 0  # default

    def test_to_dict_without_comparison(self):
        """Test serialization without comparison metrics."""
        result = BenchmarkResult(
            implementation="test",
            num_requests=100,
            duration_seconds=30.0,
            throughput_rps=3.33,
            latency_mean_ms=100.0,
            latency_p50_ms=90.0,
            latency_p90_ms=150.0,
            latency_p99_ms=200.0,
        )
        
        data = result.to_dict()
        
        assert data["implementation"] == "test"
        assert "comparison_metrics" not in data or data["comparison_metrics"] is None

    def test_to_dict_with_comparison(self, sample_benchmark_result):
        """Test serialization with comparison metrics."""
        data = sample_benchmark_result.to_dict()
        
        assert "comparison_metrics" in data
        assert data["comparison_metrics"]["boxes_mse"] == 0.0


class TestComputeBoxIoU:
    """Tests for compute_box_iou function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        box = torch.tensor([10.0, 20.0, 100.0, 120.0])
        
        iou = compute_box_iou(box, box)
        
        assert abs(iou - 1.0) < 0.001

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([20.0, 20.0, 30.0, 30.0])
        
        iou = compute_box_iou(box1, box2)
        
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([5.0, 5.0, 15.0, 15.0])
        
        iou = compute_box_iou(box1, box2)
        
        # Intersection is 5x5=25, union is 100+100-25=175
        expected = 25.0 / 175.0
        assert abs(iou - expected) < 0.001

    def test_one_inside_other(self):
        """Test IoU when one box is inside another."""
        outer = torch.tensor([0.0, 0.0, 100.0, 100.0])
        inner = torch.tensor([25.0, 25.0, 75.0, 75.0])
        
        iou = compute_box_iou(outer, inner)
        
        # Intersection is 50x50=2500, outer area is 10000, inner is 2500
        # Union is 10000+2500-2500=10000
        expected = 2500.0 / 10000.0
        assert abs(iou - expected) < 0.001

    def test_touching_boxes(self):
        """Test IoU of touching boxes is 0.0."""
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([10.0, 0.0, 20.0, 10.0])
        
        iou = compute_box_iou(box1, box2)
        
        assert iou == 0.0


class TestCompareOutputs:
    """Tests for compare_outputs function."""

    def test_identical_outputs(self, sample_detection_output):
        """Test comparing identical outputs."""
        metrics = compare_outputs(sample_detection_output, sample_detection_output)
        
        assert metrics.boxes_mse == 0.0
        assert metrics.scores_mse == 0.0
        assert metrics.labels_accuracy == 1.0
        assert metrics.num_detections_diff == 0

    def test_different_outputs(
        self,
        sample_detection_output,
        sample_detection_output_zeros,
    ):
        """Test comparing different outputs."""
        metrics = compare_outputs(
            sample_detection_output_zeros,
            sample_detection_output,
            score_threshold=0.0,
        )
        
        assert metrics.boxes_mse > 0
        assert metrics.scores_mse > 0

    def test_empty_outputs(self, sample_detection_output_empty):
        """Test comparing empty outputs."""
        metrics = compare_outputs(
            sample_detection_output_empty,
            sample_detection_output_empty,
        )
        
        assert metrics.boxes_mse == 0.0
        assert metrics.labels_accuracy == 1.0
        assert metrics.num_detections_diff == 0

    def test_one_empty_output(
        self,
        sample_detection_output,
        sample_detection_output_empty,
    ):
        """Test comparing one empty output with non-empty."""
        metrics = compare_outputs(
            sample_detection_output_empty,
            sample_detection_output,
            score_threshold=0.0,
        )
        
        assert metrics.boxes_mse == float('inf')
        assert metrics.labels_accuracy == 0.0

    def test_score_threshold_filtering(self):
        """Test that score threshold filters detections."""
        high_score = DetectionOutput(
            boxes=torch.tensor([[10, 20, 100, 120]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([1]),
            model_name="high",
            inference_time_ms=10.0,
        )
        
        low_score = DetectionOutput(
            boxes=torch.tensor([[10, 20, 100, 120]]),
            scores=torch.tensor([0.05]),  # Below default threshold
            labels=torch.tensor([1]),
            model_name="low",
            inference_time_ms=10.0,
        )
        
        # With default threshold (0.1), low score is filtered
        metrics = compare_outputs(high_score, low_score, score_threshold=0.1)
        
        # One has detection, other doesn't after filtering
        assert metrics.num_detections_diff > 0

    def test_different_number_of_detections(self):
        """Test comparing outputs with different detection counts."""
        few = DetectionOutput(
            boxes=torch.tensor([[10, 20, 100, 120]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([1]),
            model_name="few",
            inference_time_ms=10.0,
        )
        
        many = DetectionOutput(
            boxes=torch.tensor([[10, 20, 100, 120], [50, 60, 150, 180], [200, 200, 300, 300]]),
            scores=torch.tensor([0.9, 0.8, 0.7]),
            labels=torch.tensor([1, 2, 3]),
            model_name="many",
            inference_time_ms=10.0,
        )
        
        metrics = compare_outputs(few, many, score_threshold=0.0)
        
        assert metrics.num_detections_diff == 2


class TestCompareAllOutputs:
    """Tests for compare_all_outputs function."""

    def test_compares_all_pairs(self, sample_detection_output):
        """Test that all pairs are compared."""
        predicted = [sample_detection_output, sample_detection_output]
        reference = [sample_detection_output, sample_detection_output]
        
        metrics = compare_all_outputs(predicted, reference)
        
        assert len(metrics) == 2
        for m in metrics:
            assert m.boxes_mse == 0.0

    def test_mismatched_lengths_raises(self, sample_detection_output):
        """Test that mismatched lengths raise error."""
        predicted = [sample_detection_output]
        reference = [sample_detection_output, sample_detection_output]
        
        with pytest.raises(ValueError, match="Mismatched"):
            compare_all_outputs(predicted, reference)


class TestAggregateComparisonMetrics:
    """Tests for aggregate_comparison_metrics function."""

    def test_aggregates_multiple_metrics(self):
        """Test aggregation of multiple metrics."""
        metrics = [
            ComparisonMetrics(boxes_mse=10, scores_mse=0.1, labels_accuracy=0.8, num_detections_diff=2, iou_mean=0.7),
            ComparisonMetrics(boxes_mse=20, scores_mse=0.2, labels_accuracy=0.9, num_detections_diff=4, iou_mean=0.8),
        ]
        
        result = aggregate_comparison_metrics(metrics)
        
        assert result.boxes_mse == pytest.approx(15.0)
        assert result.scores_mse == pytest.approx(0.15)
        assert result.labels_accuracy == pytest.approx(0.85)
        assert result.num_detections_diff == 3
        assert result.iou_mean == pytest.approx(0.75)

    def test_empty_list(self):
        """Test aggregation of empty list."""
        result = aggregate_comparison_metrics([])
        
        assert result.boxes_mse == 0.0
        assert result.labels_accuracy == 1.0

    def test_handles_inf_values(self):
        """Test that inf values are handled."""
        metrics = [
            ComparisonMetrics(boxes_mse=float('inf'), scores_mse=0.1, labels_accuracy=0.8, num_detections_diff=2, iou_mean=0.7),
            ComparisonMetrics(boxes_mse=10, scores_mse=0.2, labels_accuracy=0.9, num_detections_diff=4, iou_mean=0.8),
        ]
        
        result = aggregate_comparison_metrics(metrics)
        
        # Should only average the non-inf value
        assert result.boxes_mse == 10.0


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generates_string(self, sample_benchmark_result):
        """Test that report is generated as string."""
        results = [sample_benchmark_result]
        
        report = generate_report(results)
        
        assert isinstance(report, str)
        assert "BENCHMARK REPORT" in report
        assert "baseline" in report

    def test_includes_all_results(self):
        """Test that all results are included."""
        results = [
            BenchmarkResult(
                implementation="impl1",
                num_requests=100,
                duration_seconds=30.0,
                throughput_rps=3.33,
                latency_mean_ms=100.0,
                latency_p50_ms=90.0,
                latency_p90_ms=150.0,
                latency_p99_ms=200.0,
            ),
            BenchmarkResult(
                implementation="impl2",
                num_requests=200,
                duration_seconds=30.0,
                throughput_rps=6.66,
                latency_mean_ms=50.0,
                latency_p50_ms=45.0,
                latency_p90_ms=75.0,
                latency_p99_ms=100.0,
            ),
        ]
        
        report = generate_report(results)
        
        assert "impl1" in report
        assert "impl2" in report

    def test_saves_to_file(self, sample_benchmark_result, temp_dir):
        """Test saving report to file."""
        results = [sample_benchmark_result]
        output_file = temp_dir / "report.txt"
        
        generate_report(results, output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "BENCHMARK REPORT" in content
        
        # JSON file should also be created
        json_file = output_file.with_suffix(".json")
        assert json_file.exists()

    def test_json_output(self, sample_benchmark_result, temp_dir):
        """Test JSON output format."""
        results = [sample_benchmark_result]
        output_file = temp_dir / "report.txt"
        
        generate_report(results, output_file)
        
        json_file = output_file.with_suffix(".json")
        with open(json_file) as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["implementation"] == "baseline"

    def test_includes_comparison_metrics(self, sample_benchmark_result):
        """Test that comparison metrics are included in report."""
        results = [sample_benchmark_result]
        
        report = generate_report(results)
        
        assert "Output Comparison" in report
        assert "Boxes MSE" in report
        assert "Scores MSE" in report


class TestLoadReferenceOutputs:
    """Tests for load_reference_outputs function."""

    def test_loads_from_file(self, temp_dir, sample_detection_output):
        """Test loading reference outputs from file."""
        # Create reference file
        ref_data = {
            "image_id": "test_001",
            "dataset_name": "test",
            "image_path": "/path/to/image.jpg",
            "outputs": [sample_detection_output.to_dict()],
        }
        
        ref_file = temp_dir / "ref.json"
        with open(ref_file, "w") as f:
            json.dump(ref_data, f)
        
        outputs = load_reference_outputs(ref_file)
        
        assert len(outputs) == 1
        assert isinstance(outputs[0], DetectionOutput)
        assert outputs[0].model_name == "test_model"

    def test_loads_multiple_outputs(self, temp_dir, sample_detection_output):
        """Test loading multiple outputs from file."""
        ref_data = {
            "image_id": "test_001",
            "dataset_name": "test",
            "image_path": "/path/to/image.jpg",
            "outputs": [
                sample_detection_output.to_dict(),
                sample_detection_output.to_dict(),
                sample_detection_output.to_dict(),
            ],
        }
        
        ref_file = temp_dir / "ref.json"
        with open(ref_file, "w") as f:
            json.dump(ref_data, f)
        
        outputs = load_reference_outputs(ref_file)
        
        assert len(outputs) == 3
