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

This benchmark establishes **upper and lower bounds** for multi-model inference latency and throughput, comparing a valid baseline implementation against an invalid stub implementation, across **CPU and MPS (Metal Performance Shaders)** devices.

### CPU vs MPS Performance Comparison

| Device | Avg Latency (Baseline) | Avg Throughput (Baseline) | Speedup vs CPU |
|--------|------------------------|---------------------------|----------------|
| **CPU** | 201.74 ms | 8.49 RPS | 1.0x (baseline) |
| **MPS** | 69.59 ms | 24.46 RPS | **2.9x faster** |

### Performance Bounds by Device

| Metric | CPU Baseline | CPU Invalid | MPS Baseline | MPS Invalid |
|--------|--------------|-------------|--------------|-------------|
| **Latency (p50)** | 56 - 440 ms | 0.7 - 5.0 ms | 20 - 150 ms | 0.8 - 5.2 ms |
| **Throughput** | 2.4 - 18 RPS | 193 - 1,401 RPS | 6.9 - 52 RPS | 186 - 1,184 RPS |

## Benchmark Configuration

- **Baseline Implementation**: 3 EfficientDet models (D0, D1, D2) running sequentially
- **Invalid Implementation**: Constant zero-tensor outputs (no model computation)
- **Test Images**: 50 synthetic 512x512 RGB images
- **Concurrency Levels**: 1, 2, 4, 8 concurrent requests
- **Devices Tested**: CPU, MPS (Apple Metal)

---

## CPU Results

### 1. Sequential Requests (Concurrency = 1)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 17.90 | 56.31 ms | 59.55 ms | 51.76 ms | 60.65 ms |
| **Invalid** | 1,400.95 | 0.69 ms | 1.12 ms | 0.53 ms | 1.23 ms |

**Speedup**: Invalid is **82x faster** (latency), **78x higher throughput**

### 2. Low Concurrency (Concurrency = 2)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 9.04 | 111.37 ms | 170.11 ms | 52.28 ms | 171.10 ms |
| **Invalid** | 590.74 | 1.64 ms | 2.85 ms | 0.66 ms | 2.91 ms |

**Speedup**: Invalid is **68x faster** (latency), **65x higher throughput**

### 3. Medium Concurrency (Concurrency = 4)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 4.65 | 220.84 ms | 278.13 ms | 57.04 ms | 287.35 ms |
| **Invalid** | 318.47 | 3.02 ms | 4.64 ms | 1.51 ms | 4.77 ms |

**Speedup**: Invalid is **73x faster** (latency), **69x higher throughput**

### 4. High Concurrency (Concurrency = 8)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 2.35 | 439.59 ms | 547.72 ms | 56.59 ms | 557.25 ms |
| **Invalid** | 193.14 | 4.97 ms | 8.37 ms | 2.15 ms | 10.43 ms |

**Speedup**: Invalid is **88x faster** (latency), **82x higher throughput**

---

## MPS (Metal Performance Shaders) Results

### 1. Sequential Requests (Concurrency = 1)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 51.55 | 19.71 ms | 22.17 ms | 16.81 ms | 22.52 ms |
| **Invalid** | 1,183.59 | 0.78 ms | 1.27 ms | 0.63 ms | 1.32 ms |

**Speedup**: Invalid is **25x faster** (latency), **23x higher throughput**

### 2. Low Concurrency (Concurrency = 2)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 26.03 | 39.23 ms | 60.47 ms | 17.55 ms | 60.62 ms |
| **Invalid** | 581.09 | 1.69 ms | 3.05 ms | 0.71 ms | 3.09 ms |

**Speedup**: Invalid is **23x faster** (latency), **22x higher throughput**

### 3. Medium Concurrency (Concurrency = 4)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 13.39 | 76.23 ms | 115.59 ms | 18.11 ms | 132.27 ms |
| **Invalid** | 300.60 | 3.18 ms | 6.11 ms | 1.32 ms | 6.99 ms |

**Speedup**: Invalid is **24x faster** (latency), **22x higher throughput**

### 4. High Concurrency (Concurrency = 8)

| Implementation | Throughput (RPS) | Latency p50 | Latency p99 | Latency Min | Latency Max |
|----------------|------------------|-------------|-------------|-------------|-------------|
| **Baseline** | 6.86 | 149.83 ms | 224.91 ms | 18.69 ms | 263.23 ms |
| **Invalid** | 185.89 | 5.23 ms | 8.95 ms | 3.14 ms | 9.05 ms |

**Speedup**: Invalid is **29x faster** (latency), **27x higher throughput**

---

## CPU vs MPS Comparison Analysis

### Baseline Model Performance

| Metric | CPU | MPS | MPS Advantage |
|--------|-----|-----|---------------|
| **Average Latency** | 201.74 ms | 69.59 ms | **2.90x faster** |
| **Average Throughput** | 8.49 RPS | 24.46 RPS | **2.88x higher** |
| **Latency Range (min-max)** | 51.76 - 557.25 ms | 16.81 - 263.23 ms | ~2x lower |
| **Throughput Range** | 2.35 - 17.90 RPS | 6.86 - 51.55 RPS | ~3x higher |

### Key Findings

1. **MPS provides ~3x speedup** over CPU for the baseline model workload on Apple M4 Pro
2. **Consistent speedup across concurrency levels** - MPS advantage holds even at high concurrency
3. **Invalid implementation shows similar performance** on both devices (I/O bound, not compute bound)
4. **Optimization headroom is lower on MPS** (~25x gap vs ~80x gap on CPU) because the baseline is already faster

### Performance Bounds Summary

#### CPU Performance Bounds

| Concurrency | Baseline Latency Range | Invalid Latency Range | Baseline Throughput | Invalid Throughput |
|-------------|------------------------|----------------------|--------------------|--------------------|
| 1 | 51.76 - 60.65 ms | 0.53 - 1.23 ms | 17.90 RPS | 1,400.95 RPS |
| 2 | 52.28 - 171.10 ms | 0.66 - 2.91 ms | 9.04 RPS | 590.74 RPS |
| 4 | 57.04 - 287.35 ms | 1.51 - 4.77 ms | 4.65 RPS | 318.47 RPS |
| 8 | 56.59 - 557.25 ms | 2.15 - 10.43 ms | 2.35 RPS | 193.14 RPS |

#### MPS Performance Bounds

| Concurrency | Baseline Latency Range | Invalid Latency Range | Baseline Throughput | Invalid Throughput |
|-------------|------------------------|----------------------|--------------------|--------------------|
| 1 | 16.81 - 22.52 ms | 0.63 - 1.32 ms | 51.55 RPS | 1,183.59 RPS |
| 2 | 17.55 - 60.62 ms | 0.71 - 3.09 ms | 26.03 RPS | 581.09 RPS |
| 4 | 18.11 - 132.27 ms | 1.32 - 6.99 ms | 13.39 RPS | 300.60 RPS |
| 8 | 18.69 - 263.23 ms | 3.14 - 9.05 ms | 6.86 RPS | 185.89 RPS |

---

## Implications for Optimization

### Device-Specific Gap Analysis

| Device | Baseline-to-Invalid Gap | Optimization Target |
|--------|-------------------------|---------------------|
| **CPU** | ~75-82x | High potential for improvement |
| **MPS** | ~23-29x | Already accelerated, moderate headroom |

### Expected Performance with Optimizations

| Optimization | CPU Expected | MPS Expected |
|--------------|-------------|--------------|
| `torch.vmap` | 3-5x speedup (~15-25 ms) | 2-3x speedup (~7-10 ms) |
| Grouped Conv + TensorRT | 4-6x speedup (~10-15 ms) | N/A (Metal already optimized) |
| Optimal (near invalid) | ~0.7 ms | ~0.8 ms |

---

## Output Comparison (MSE)

| Implementation | Boxes MSE | Scores MSE | Labels Accuracy | Detection Count Diff |
|----------------|-----------|------------|-----------------|---------------------|
| **Baseline** | 0.0 (reference) | 0.0 (reference) | 100% | 0 |
| **Invalid** | Very High | Very High | 0% | Many |

The invalid implementation returns constant zero outputs, providing the **maximum possible output difference** from the baseline reference.

---

## Running the Benchmark

```bash
# Install dependencies
uv sync

# Run benchmark with CPU and MPS comparison
uv run python scripts/run_full_benchmark.py

# Results saved to:
# - outputs/benchmark/benchmark_report.txt
# - outputs/benchmark/benchmark_results.json
```

---

## Conclusion

### Key Takeaways

1. **MPS (Apple Silicon GPU) provides significant speedup** - approximately **2.9x faster** than CPU for the EfficientDet baseline on the M4 Pro

2. **CPU has larger optimization headroom** - The ~75-82x gap between baseline and invalid on CPU suggests more room for improvement through techniques like `torch.vmap` and TensorRT

3. **MPS is already well-optimized** - The ~23-29x gap indicates Metal's GPU acceleration is providing substantial benefits out of the box

4. **I/O overhead is similar across devices** - The invalid implementation shows similar performance on both CPU and MPS, indicating the remaining overhead is I/O bound

5. **Concurrency impacts both devices similarly** - Latency increases linearly with concurrency as requests queue up, regardless of device

### Performance Bounds Established

| Bound | CPU | MPS |
|-------|-----|-----|
| **Upper (Baseline p50)** | 56 - 440 ms | 20 - 150 ms |
| **Lower (Invalid p50)** | 0.7 - 5.0 ms | 0.8 - 5.2 ms |
| **Throughput Upper** | 1,401 RPS | 1,184 RPS |
| **Throughput Lower** | 2.4 RPS | 6.9 RPS |

Any future implementation should fall between these bounds, with throughput and latency metrics that approach the invalid implementation while maintaining output accuracy matching the baseline.
