# Multi-Model EfficientDet Inference Benchmarking

A benchmarking setup for evaluating inference strategies for multiple EfficientDet models running simultaneously.

## Overview

This project provides:

1. **Baseline Implementation** - Sequential inference with 3 pretrained EfficientDet models (D0, D1, D2)
2. **Invalid Implementation** - Constant output stub for establishing performance lower bounds
3. **Locust Load Testing** - Configurable RPS and concurrent users benchmarking
4. **Metrics Collection** - Throughput, latency (p50/p90/p99), VRAM usage, output comparison (MSE)

## Quick Start

### 1. Setup Environment

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### 2. Download Datasets

```bash
# Download COCO val2017 (~1GB) and Roboflow Aquarium datasets
uv run python scripts/download_datasets.py
```

### 3. Start the Inference Server

```bash
uv run uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

### 4. Run Benchmarks

**Option A: Using the benchmark script**

```bash
# Run full benchmark suite
uv run python scripts/run_benchmark.py --users 1 4 8 --duration 30 --compare-outputs
```

**Option B: Using Locust directly**

```bash
# Web UI mode (interactive)
uv run locust -f src/benchmark/locustfile.py --host http://localhost:8000

# Headless mode
uv run locust -f src/benchmark/locustfile.py --host http://localhost:8000 \
    --headless -u 10 -r 2 -t 60s --csv=outputs/benchmark_results
```

### 5. Generate Reference Outputs

```bash
# Generate reference outputs from baseline for comparison
uv run python -m src.benchmark.reference --max-images 100
```

## Project Structure

```
model-stacking-demo/
├── pyproject.toml              # Dependencies via uv
├── src/
│   ├── models/
│   │   ├── base.py             # Abstract base class for implementations
│   │   ├── baseline.py         # Valid: Sequential EfficientDet D0, D1, D2
│   │   └── invalid.py          # Invalid: Constant output stub
│   ├── server/
│   │   └── app.py              # FastAPI server with /infer endpoints
│   ├── datasets/
│   │   ├── loader.py           # Unified dataset loading interface
│   │   └── download.py         # Dataset download utilities
│   └── benchmark/
│       ├── locustfile.py       # Locust load test definition
│       ├── metrics.py          # VRAM monitoring, output comparison
│       └── reference.py        # Generate/store reference outputs
├── data/                       # Downloaded datasets
├── outputs/                    # Benchmark results and references
└── scripts/
    ├── download_datasets.py    # Download COCO val2017 + Roboflow datasets
    └── run_benchmark.py        # Orchestrate benchmark runs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/vram` | GET | Current VRAM usage statistics |
| `/vram/reset` | POST | Reset VRAM peak statistics |
| `/infer/{impl}` | POST | Run inference (impl: `baseline` or `invalid`) |
| `/implementations` | GET | List available implementations |

## Metrics Collected

| Metric | Description |
|--------|-------------|
| **Throughput (RPS)** | Requests processed per second |
| **Latency (p50/p90/p99)** | Response time percentiles |
| **VRAM Usage** | Peak GPU memory usage in MB |
| **Boxes MSE** | Mean squared error of bounding boxes vs reference |
| **Scores MSE** | Mean squared error of confidence scores vs reference |
| **Labels Accuracy** | Accuracy of predicted class labels |

## Extending with New Implementations

To add a new implementation (e.g., `torch.vmap` or grouped convolutions):

1. Create a new file in `src/models/` that inherits from `BaseModelImpl`
2. Implement `load()`, `predict()`, `predict_batch()`, and properties
3. Register the implementation in `src/server/app.py`
4. Run benchmarks to compare against baseline

```python
from src.models.base import BaseModelImpl, DetectionOutput

class VmapImpl(BaseModelImpl):
    def load(self) -> None:
        # Load stacked weights and prepare vmap
        ...
    
    def predict(self, image: Image.Image) -> List[DetectionOutput]:
        # Run parallel inference with vmap
        ...
```

## Expected Results

| Implementation | Throughput | Latency (p99) | VRAM | Output Error |
|----------------|------------|---------------|------|--------------|
| **Baseline** | ~X RPS | High | High | 0 (reference) |
| **Invalid** | ~10x baseline | Very low | Minimal | High MSE |
| **vmap** (future) | ~3-5x baseline | Medium | Optimal | ~0 |
| **Grouped Conv** (future) | ~4-6x baseline | Low | Low | ~0 |

## Configuration

Environment variables for Locust:

- `BENCHMARK_DATA_DIR` - Dataset directory (default: `data`)
- `BENCHMARK_DATASET` - Dataset to use: `coco_val2017` or `roboflow_aquarium`
- `BENCHMARK_MAX_IMAGES` - Maximum images to load (default: `100`)
- `BENCHMARK_IMPLEMENTATION` - Implementation to test: `baseline` or `invalid`

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~2GB disk space for datasets
- ~4GB VRAM for 3 EfficientDet models
