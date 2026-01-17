# Benchmark Results: Multi-Model EfficientDet Inference

## Test Environment

| Component | Specification |
|-----------|--------------|
| **Hardware** | MacBook Pro |
| **Chip** | Apple M4 Pro |
| **Memory** | 24 GB |
| **Platform** | macOS 26.2 (arm64) |
| **Python** | 3.12.8 |
| **PyTorch** | 2.9.1 |

## Executive Summary

This benchmark measures **real inference performance** of 3 EfficientDet-D0 models with different fine-tuning objectives, running sequentially on each request. Comparing **CPU vs MPS (Metal Performance Shaders)** on Apple Silicon.

### Key Finding: MPS is 11.3x faster than CPU

| Device | Latency (p50) | Throughput | Speedup vs CPU |
|--------|---------------|------------|----------------|
| **CPU** | 1,020 ms | 0.98 RPS | 1.0x (baseline) |
| **MPS** | 88 ms | 11.08 RPS | **11.3x faster** |

---

## Models Under Test

Each inference request runs **3 EfficientDet-D0 checkpoints** sequentially, simulating models fine-tuned for different detection objectives:

| Model | Classes | Description | Parameters |
|-------|---------|-------------|------------|
| `efficientdet_d0_coco` | 90 | COCO dataset (reference) | 3.88M |
| `efficientdet_d0_aquarium` | 7 | Aquarium dataset (fish, jellyfish, etc.) | 3.83M |
| `efficientdet_d0_vehicles` | 20 | Vehicles dataset (car, truck, bus, etc.) | 3.84M |

**Key characteristics:**
- **Same backbone architecture**: EfficientNet-B0 + BiFPN
- **Same input size**: 512×512
- **Different class heads**: Varying output dimensions (90, 7, 20 classes)
- **Total**: ~11.5M parameters across 3 models

This architecture enables future optimization with `torch.vmap` (backbone can be stacked).

---

## Performance Bounds

### Upper Bound: CPU Baseline (Worst Case)

Running 3 EfficientDet-D0 models sequentially on CPU represents the **maximum latency** and **minimum throughput** for valid inference.

| Metric | Value |
|--------|-------|
| **Latency p50** | 1,019.78 ms |
| **Latency p99** | 1,038.65 ms |
| **Latency Range** | 1,005 - 1,039 ms |
| **Throughput** | 0.98 RPS |

**Breakdown per model (estimated ~340 ms each):**
- EfficientDet-D0 COCO (90 classes)
- EfficientDet-D0 Aquarium (7 classes)
- EfficientDet-D0 Vehicles (20 classes)

### Lower Bound: MPS Baseline (Best Case with GPU)

Running the same 3 models on MPS (Apple Silicon GPU) represents the **minimum latency** achievable with real model inference.

| Metric | Value |
|--------|-------|
| **Latency p50** | 87.99 ms |
| **Latency p99** | 103.02 ms |
| **Latency Range** | 85 - 104 ms |
| **Throughput** | 11.08 RPS |

**Breakdown per model (estimated ~30 ms each):**
- EfficientDet-D0 COCO
- EfficientDet-D0 Aquarium  
- EfficientDet-D0 Vehicles

### Theoretical Lower Bound: Invalid Implementation

The invalid implementation (constant zero outputs) shows the pure I/O overhead without any model computation:

| Device | Latency | Throughput |
|--------|---------|------------|
| CPU | 0.01 ms | 119,763 RPS |
| MPS | 0.01 ms | 67,521 RPS |

---

## CPU vs MPS Comparison

### Detailed Comparison

| Metric | CPU | MPS | MPS Advantage |
|--------|-----|-----|---------------|
| **Average Latency** | 1,019.15 ms | 90.26 ms | **11.29x faster** |
| **Latency p50** | 1,019.78 ms | 87.99 ms | **11.59x faster** |
| **Latency p99** | 1,038.65 ms | 103.02 ms | **10.08x faster** |
| **Throughput** | 0.98 RPS | 11.08 RPS | **11.29x higher** |
| **Latency Variance** | 34 ms | 19 ms | 1.8x more stable |

### Analysis

1. **MPS provides 11.3x speedup** - The Apple M4 Pro's GPU dramatically accelerates inference

2. **Consistent speedup** - Speedup ranges from 10x (p99) to 11.6x (p50)

3. **Lower variance on MPS** - GPU execution is more deterministic

4. **Per-model inference time**:
   - CPU: ~340 ms per model
   - MPS: ~30 ms per model

### Performance Gap Analysis

| Comparison | Gap | Optimization Opportunity |
|------------|-----|--------------------------|
| CPU Baseline → MPS Baseline | 11.3x | Use GPU acceleration ✅ |
| MPS Baseline → Invalid | 6,100x | Room for batching, fusion |
| CPU Baseline → Invalid | 69,000x | Maximum theoretical |

---

## Why 3× EfficientDet-D0 is Faster than D0+D1+D2

Previous benchmarks used D0, D1, D2 (different architectures, different input sizes). The new setup uses 3× D0:

| Configuration | CPU Latency | MPS Latency | MPS Speedup |
|---------------|-------------|-------------|-------------|
| **D0+D1+D2** (mixed) | 1,461 ms | 167 ms | 8.7x |
| **3× D0** (uniform) | 1,019 ms | 88 ms | 11.3x |
| **Improvement** | 30% faster | 47% faster | +2.6x better |

**Why the improvement:**
- All models use 512×512 input (vs 512/640/768)
- Same architecture allows better GPU utilization
- Single preprocessing step (shared transform)

---

## Implications for Production

### Latency Requirements

| Use Case | Latency Target | CPU | MPS |
|----------|----------------|-----|-----|
| Real-time video (30 FPS) | < 33 ms | ❌ | ❌ (need vmap/fusion) |
| Interactive (< 100 ms) | < 100 ms | ❌ | ✅ (88 ms) |
| Near-realtime (< 200 ms) | < 200 ms | ❌ | ✅ |
| Batch processing (< 2s) | < 2,000 ms | ✅ | ✅ |

### Throughput Requirements

| Requests/sec | CPU | MPS |
|--------------|-----|-----|
| < 1 RPS | ✅ (0.98) | ✅ |
| 1-10 RPS | ❌ | ✅ (11.08) |
| 10-30 RPS | ❌ | ⚠️ (need batching) |
| > 30 RPS | ❌ | ❌ (need CUDA/optimization) |

---

## Optimization Roadmap

Based on these benchmarks with uniform architecture:

### Phase 1: Device Selection (Current) ✅
- **CPU**: 1,019 ms
- **MPS**: 88 ms → **11.3x improvement**

### Phase 2: Backbone Stacking with vmap (Next)
Since all 3 models share the same backbone architecture, we can:
- Stack backbone weights: `[3, ...weight_shape...]`
- Use `torch.vmap` for parallel backbone computation
- Expected: **2-3x improvement** → target ~30-40 ms on MPS

### Phase 3: Quantization
- FP16 inference on MPS
- Expected: **1.5-2x improvement** → target ~20-30 ms

### Phase 4: Full Fusion
- Grouped convolutions for parallel class heads
- Core ML conversion
- Target: **< 20 ms** (enabling 30+ FPS)

---

## Running the Benchmark

```bash
# 1. Install dependencies
uv sync

# 2. Generate checkpoints (required once)
uv run python scripts/generate_checkpoints.py

# 3. Run benchmark with REAL models
uv run python scripts/run_full_benchmark.py

# Optional: Run with mocked models (fast, for testing)
uv run python scripts/run_full_benchmark.py --mock

# Results saved to:
# - outputs/benchmark/benchmark_report.txt
# - outputs/benchmark/benchmark_results.json
```

---

## Raw Data

### CPU Baseline (20 requests, 3× EfficientDet-D0)

```
Mean: 1,019.15 ms
Std:  11.2 ms
Min:  1,005.11 ms
Max:  1,038.75 ms
p50:  1,019.78 ms
p99:  1,038.65 ms
```

### MPS Baseline (20 requests, 3× EfficientDet-D0)

```
Mean: 90.26 ms
Std:  6.3 ms
Min:  84.71 ms
Max:  103.63 ms
p50:  87.99 ms
p99:  103.02 ms
```

---

## Conclusion

1. **Real EfficientDet inference with 3 uniform D0 models** takes **1,019 ms on CPU** and **88 ms on MPS**

2. **MPS provides 11.3x speedup** - significantly better than the 8.7x with mixed architectures

3. **Uniform architecture unlocks optimization potential** - backbone weights can now be stacked for `vmap`

4. **MPS achieves interactive latency** - 88 ms enables near-realtime applications

5. **Real-time still requires optimization** - need vmap/fusion to achieve 30 FPS (33 ms)

### Performance Bounds Summary

| Bound | CPU | MPS |
|-------|-----|-----|
| **Upper (Real Model p50)** | 1,020 ms | 88 ms |
| **Lower (Invalid p50)** | 0.01 ms | 0.01 ms |
| **Optimization Gap** | 102,000x | 8,800x |

### Model Configuration

| Model | Classes | Purpose |
|-------|---------|---------|
| `efficientdet_d0_coco` | 90 | General object detection |
| `efficientdet_d0_aquarium` | 7 | Marine life detection |
| `efficientdet_d0_vehicles` | 20 | Vehicle detection |

All models share EfficientDet-D0 backbone (512×512 input, ~3.85M params each).
