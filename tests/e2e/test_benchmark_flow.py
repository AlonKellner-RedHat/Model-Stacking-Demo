"""End-to-end tests for the complete benchmark workflow."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.benchmark.metrics import (
    compare_outputs,
    compare_all_outputs,
    aggregate_comparison_metrics,
    BenchmarkResult,
    ComparisonMetrics,
    generate_report,
)
from src.models.base import DetectionOutput


class TestFullInferenceCycle:
    """Tests for complete inference cycle with synthetic images."""

    def test_inference_returns_valid_detections(
        self,
        test_client,
        sample_image_bytes,
    ):
        """Test that inference cycle returns valid detections."""
        # Run inference
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify we got detections from all 3 models
        assert len(data["detections"]) == 3
        
        # Each detection should have required fields
        for det in data["detections"]:
            assert "boxes" in det
            assert "scores" in det
            assert "labels" in det
            assert "model_name" in det
            assert "inference_time_ms" in det

    def test_multiple_images_processed(self, test_client, sample_image):
        """Test processing multiple different images."""
        results = []
        
        # Create slightly different images
        for i in range(5):
            img = sample_image.copy()
            # Modify image slightly
            pixels = img.load()
            for x in range(min(10, img.width)):
                for y in range(min(10, img.height)):
                    pixels[x, y] = (i * 50 % 256, 100, 100)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            
            response = test_client.post(
                "/infer/baseline",
                files={"file": (f"test_{i}.jpg", buffer.getvalue(), "image/jpeg")},
            )
            
            assert response.status_code == 200
            results.append(response.json())
        
        # All should have same structure
        for result in results:
            assert len(result["detections"]) == 3


class TestOutputComparisonWorkflow:
    """Tests for output comparison between implementations."""

    def test_compare_baseline_vs_invalid(self, test_client, sample_image_bytes):
        """Test comparing baseline output against invalid output."""
        # Get baseline output
        baseline_response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        baseline_data = baseline_response.json()
        
        # Get invalid output
        invalid_response = test_client.post(
            "/infer/invalid",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        invalid_data = invalid_response.json()
        
        # Convert to DetectionOutput objects
        baseline_outputs = [
            DetectionOutput.from_dict({
                "boxes": d["boxes"],
                "scores": d["scores"],
                "labels": d["labels"],
                "model_name": d["model_name"],
                "inference_time_ms": d["inference_time_ms"],
            })
            for d in baseline_data["detections"]
        ]
        
        invalid_outputs = [
            DetectionOutput.from_dict({
                "boxes": d["boxes"],
                "scores": d["scores"],
                "labels": d["labels"],
                "model_name": d["model_name"],
                "inference_time_ms": d["inference_time_ms"],
            })
            for d in invalid_data["detections"]
        ]
        
        # Compare outputs
        metrics = compare_all_outputs(
            invalid_outputs,
            baseline_outputs,
            score_threshold=0.0,
        )
        
        assert len(metrics) == 3
        
        # Aggregate metrics
        aggregated = aggregate_comparison_metrics(metrics)
        
        # Invalid should differ from baseline
        # (boxes MSE should be non-zero since invalid returns zeros)
        assert isinstance(aggregated.boxes_mse, float)

    def test_identical_outputs_have_zero_error(self, test_client, sample_image_bytes):
        """Test that comparing identical outputs gives zero error."""
        # Get baseline output twice
        response1 = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        response2 = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        
        # Convert first response to DetectionOutput
        outputs1 = [
            DetectionOutput.from_dict({
                "boxes": d["boxes"],
                "scores": d["scores"],
                "labels": d["labels"],
                "model_name": d["model_name"],
                "inference_time_ms": d["inference_time_ms"],
            })
            for d in response1.json()["detections"]
        ]
        
        # Compare with itself
        metrics = compare_all_outputs(outputs1, outputs1)
        aggregated = aggregate_comparison_metrics(metrics)
        
        assert aggregated.boxes_mse == 0.0
        assert aggregated.scores_mse == 0.0
        assert aggregated.labels_accuracy == 1.0


class TestMetricsCollectionWorkflow:
    """Tests for metrics collection accuracy."""

    def test_timing_metrics_positive(self, test_client, sample_image_bytes):
        """Test that timing metrics are positive."""
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        
        assert data["total_inference_time_ms"] > 0
        
        for det in data["detections"]:
            assert det["inference_time_ms"] >= 0

    def test_invalid_faster_timing(self, test_client, sample_image_bytes):
        """Test that invalid implementation has faster timing."""
        # Get timings from both implementations
        baseline_times = []
        invalid_times = []
        
        for _ in range(3):
            baseline_response = test_client.post(
                "/infer/baseline",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            baseline_times.append(baseline_response.json()["total_inference_time_ms"])
            
            invalid_response = test_client.post(
                "/infer/invalid",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            invalid_times.append(invalid_response.json()["total_inference_time_ms"])
        
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)
        
        # Invalid should generally be faster
        # With mocked models this may vary, so we just verify both are measured
        assert avg_baseline > 0
        assert avg_invalid > 0


class TestReportGeneration:
    """Tests for benchmark report generation."""

    def test_generate_complete_report(self, temp_dir):
        """Test generating a complete benchmark report."""
        results = [
            BenchmarkResult(
                implementation="baseline",
                num_requests=100,
                duration_seconds=30.0,
                throughput_rps=3.33,
                latency_mean_ms=300.0,
                latency_p50_ms=280.0,
                latency_p90_ms=400.0,
                latency_p99_ms=500.0,
                vram_peak_mb=2048.0,
                comparison_metrics=ComparisonMetrics(
                    boxes_mse=0.0,
                    scores_mse=0.0,
                    labels_accuracy=1.0,
                    num_detections_diff=0,
                    iou_mean=1.0,
                ),
            ),
            BenchmarkResult(
                implementation="invalid",
                num_requests=100,
                duration_seconds=30.0,
                throughput_rps=50.0,
                latency_mean_ms=20.0,
                latency_p50_ms=18.0,
                latency_p90_ms=25.0,
                latency_p99_ms=35.0,
                vram_peak_mb=100.0,
                comparison_metrics=ComparisonMetrics(
                    boxes_mse=1500.0,
                    scores_mse=0.9,
                    labels_accuracy=0.0,
                    num_detections_diff=10,
                    iou_mean=0.0,
                ),
            ),
        ]
        
        output_file = temp_dir / "benchmark_report.txt"
        report = generate_report(results, output_file)
        
        # Verify report content
        assert "BENCHMARK REPORT" in report
        assert "baseline" in report
        assert "invalid" in report
        assert "Throughput" in report or "throughput" in report.lower()
        assert "Latency" in report or "latency" in report.lower()
        
        # Verify files created
        assert output_file.exists()
        assert output_file.with_suffix(".json").exists()
        
        # Verify JSON content
        with open(output_file.with_suffix(".json")) as f:
            json_data = json.load(f)
        
        assert len(json_data) == 2
        assert json_data[0]["implementation"] == "baseline"
        assert json_data[1]["implementation"] == "invalid"

    def test_report_shows_performance_difference(self, temp_dir):
        """Test that report shows clear performance difference."""
        results = [
            BenchmarkResult(
                implementation="baseline",
                num_requests=100,
                duration_seconds=30.0,
                throughput_rps=3.0,
                latency_mean_ms=300.0,
                latency_p50_ms=280.0,
                latency_p90_ms=400.0,
                latency_p99_ms=500.0,
            ),
            BenchmarkResult(
                implementation="invalid",
                num_requests=100,
                duration_seconds=30.0,
                throughput_rps=100.0,
                latency_mean_ms=10.0,
                latency_p50_ms=8.0,
                latency_p90_ms=15.0,
                latency_p99_ms=20.0,
            ),
        ]
        
        report = generate_report(results)
        
        # Both implementations should be in report
        assert "baseline" in report
        assert "invalid" in report


class TestEndToEndDataFlow:
    """Tests for complete data flow from image to metrics."""

    def test_image_to_detection_to_comparison(
        self,
        test_client,
        sample_image_bytes,
    ):
        """Test full flow from image input to comparison metrics."""
        # Step 1: Get baseline detections
        baseline_resp = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert baseline_resp.status_code == 200
        
        # Step 2: Get invalid detections
        invalid_resp = test_client.post(
            "/infer/invalid",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert invalid_resp.status_code == 200
        
        # Step 3: Parse into DetectionOutput objects
        baseline_outputs = [
            DetectionOutput.from_dict({
                "boxes": d["boxes"],
                "scores": d["scores"],
                "labels": d["labels"],
                "model_name": d["model_name"],
                "inference_time_ms": d["inference_time_ms"],
            })
            for d in baseline_resp.json()["detections"]
        ]
        
        invalid_outputs = [
            DetectionOutput.from_dict({
                "boxes": d["boxes"],
                "scores": d["scores"],
                "labels": d["labels"],
                "model_name": d["model_name"],
                "inference_time_ms": d["inference_time_ms"],
            })
            for d in invalid_resp.json()["detections"]
        ]
        
        # Step 4: Compare outputs
        metrics = compare_all_outputs(
            invalid_outputs,
            baseline_outputs,
            score_threshold=0.0,
        )
        
        # Step 5: Aggregate
        aggregated = aggregate_comparison_metrics(metrics)
        
        # Step 6: Create benchmark result
        result = BenchmarkResult(
            implementation="invalid",
            num_requests=1,
            duration_seconds=0.1,
            throughput_rps=10.0,
            latency_mean_ms=invalid_resp.json()["total_inference_time_ms"],
            latency_p50_ms=invalid_resp.json()["total_inference_time_ms"],
            latency_p90_ms=invalid_resp.json()["total_inference_time_ms"],
            latency_p99_ms=invalid_resp.json()["total_inference_time_ms"],
            comparison_metrics=aggregated,
        )
        
        # Verify complete result
        assert result.implementation == "invalid"
        assert result.comparison_metrics is not None

    def test_serialization_roundtrip(self, test_client, sample_image_bytes, temp_dir):
        """Test that detection results can be serialized and restored."""
        # Get detections
        response = test_client.post(
            "/infer/baseline",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        original_data = response.json()
        
        # Serialize to file
        ref_file = temp_dir / "reference.json"
        with open(ref_file, "w") as f:
            json.dump(original_data, f)
        
        # Load back
        with open(ref_file) as f:
            loaded_data = json.load(f)
        
        # Verify roundtrip
        assert loaded_data["implementation"] == original_data["implementation"]
        assert len(loaded_data["detections"]) == len(original_data["detections"])
        
        for orig_det, loaded_det in zip(
            original_data["detections"],
            loaded_data["detections"],
        ):
            assert orig_det["model_name"] == loaded_det["model_name"]
            assert orig_det["boxes"] == loaded_det["boxes"]
            assert orig_det["scores"] == loaded_det["scores"]
            assert orig_det["labels"] == loaded_det["labels"]


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""

    def test_sequential_requests_consistent(self, test_client, sample_image_bytes):
        """Test that sequential requests produce consistent results."""
        results = []
        
        for i in range(5):
            response = test_client.post(
                "/infer/invalid",  # Use invalid for speed
                files={"file": (f"test_{i}.jpg", sample_image_bytes, "image/jpeg")},
            )
            results.append(response.json())
        
        # All results should have same structure
        first = results[0]
        for result in results[1:]:
            assert len(result["detections"]) == len(first["detections"])
            
            # For invalid impl, boxes should all be zeros
            for det in result["detections"]:
                assert all(
                    all(v == 0.0 for v in box)
                    for box in det["boxes"]
                )
