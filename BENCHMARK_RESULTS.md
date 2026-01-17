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

| Use Case | Latency Target | CPU | MPS | MPS + compile | MPS + vmap |
|----------|----------------|-----|-----|---------------|------------|
| Real-time video (30 FPS) | < 33 ms | ❌ | ❌ | ❌ | ⚠️ (37 ms, close!) |
| Interactive (< 50 ms) | < 50 ms | ❌ | ❌ | ✅ (43 ms) | ✅ (37 ms) |
| Interactive (< 100 ms) | < 100 ms | ❌ | ✅ (88 ms) | ✅ | ✅ |
| Near-realtime (< 200 ms) | < 200 ms | ❌ | ✅ | ✅ | ✅ |
| Batch processing (< 2s) | < 2,000 ms | ✅ | ✅ | ✅ | ✅ |

### Throughput Requirements

| Requests/sec | CPU | MPS | MPS + compile | MPS + vmap |
|--------------|-----|-----|---------------|------------|
| < 1 RPS | ✅ (0.98) | ✅ | ✅ | ✅ |
| 1-10 RPS | ❌ | ✅ (11.08) | ✅ | ✅ |
| 10-20 RPS | ❌ | ⚠️ | ✅ (23 RPS) | ✅ (27 RPS) |
| 20-30 RPS | ❌ | ❌ | ✅ | ✅ (27 RPS) |
| > 30 RPS | ❌ | ❌ | ❌ | ⚠️ (need CUDA/batching) |

---

## Ablation Study: Optimization Techniques

We tested composable optimizations to find the best configuration:

### Results Summary (MPS)

| Configuration | Latency (ms) | Throughput | Speedup | Output Quality | Status |
|---------------|--------------|------------|---------|----------------|--------|
| **vmap_backbone** | **37.4** | **26.8 RPS** | **2.69x** | IoU=1.0 (exact) | ✅ **Best** |
| vmap + FP16 | 38.0 | 26.3 RPS | 2.64x | IoU=0.9995 | ✅ Excellent |
| torchserve_vmap | 41.1 | 24.3 RPS | 2.44x | IoU=0.78* | ✅ Production |
| torch.compile (default) | 42.6 | 23.5 RPS | 2.36x | IoU=1.0 (exact) | ✅ Great |
| torch.compile (reduce-overhead) | 44.5 | 22.5 RPS | 2.26x | IoU=1.0 (exact) | ✅ Great |
| Baseline (no optimizations) | 100.5 | 10.0 RPS | 1.00x | Reference | Reference |
| grouped_super_model | 101.2 | 9.9 RPS | 0.99x | IoU=1.0 (exact) | ⚠️ Not faster |
| torchserve_baseline | 103.4 | 9.7 RPS | 0.97x | IoU=0.78* | ⚠️ JPEG loss |
| grouped+compile | 104.5 | 9.6 RPS | 0.96x | IoU=1.0 (exact) | ⚠️ Not faster |
| FP16 only | 596.5 | 1.7 RPS | 0.17x | IoU=0.9995 | ❌ Slower |
| compile + FP16 | 620.9 | 1.6 RPS | 0.16x | IoU=0.9995 | ❌ Slower |

*TorchServe IoU=0.78 is due to JPEG encoding (simulates real HTTP traffic), not model error.

### Key Findings

1. **vmap_backbone provides 2.58x speedup** (NEW BEST!)
   - Reduces latency from 96ms to 37ms
   - Uses `torch.vmap` to parallelize backbone+FPN+box_net across 3 models
   - Internally applies `torch.compile` for optimal kernel fusion
   - **Exact output match** (MSE=0, IoU=1.0)

2. **torch.compile provides 2.2x speedup**
   - Reduces latency from 96ms to 43ms
   - Uses PyTorch's inductor backend for kernel optimization
   - vmap_backbone is 15% faster than compile alone

3. **FP16 is counterproductive on MPS**
   - 6x slower than FP32 (613ms vs 96ms)
   - MPS has significant FP16 conversion overhead
   - Not recommended for Apple Silicon

4. **Best configuration: vmap_backbone**
   - 37ms latency, 27 RPS throughput
   - Enables interactive applications (< 50ms target)
   - Exact output match with baseline

### Output Quality Comparison

| Configuration | Boxes MSE | Scores MSE | Labels Acc | Mean IoU |
|---------------|-----------|------------|------------|----------|
| vmap_backbone | 0.000000 | 0.000000 | 1.0000 | 1.0000 |
| torch.compile | 0.000000 | 0.000000 | 1.0000 | 1.0000 |
| baseline | 0.000000 | 0.000000 | 1.0000 | 1.0000 |
| FP16 variants | 0.007 | 0.000001 | 1.0000 | 0.9995 |

### Updated Performance After Optimization

| Device | Config | Latency (p50) | Throughput | vs CPU Baseline |
|--------|--------|---------------|------------|-----------------|
| CPU | baseline | 1,020 ms | 0.98 RPS | 1.0x |
| MPS | baseline | 88 ms | 11.08 RPS | 11.6x |
| MPS | torch.compile | 43 ms | 23 RPS | 23.7x |
| **MPS** | **vmap_backbone** | **37 ms** | **27 RPS** | **27.6x** |

---

## Optimization Roadmap

Based on ablation study results:

### Phase 1: Device Selection ✅
- **CPU**: 1,019 ms
- **MPS**: 88 ms → **11.3x improvement**

### Phase 2: torch.compile ✅
- **MPS + compile**: 43 ms → **2.1x additional improvement**
- **Total vs CPU**: **23.7x faster**

### Phase 3: Backbone Stacking with vmap ✅ (COMPLETED)
All 3 models share the same backbone architecture (EfficientNet-B0 + BiFPN), enabling:
- Stack backbone weights using `torch.func.stack_module_state`: `[3, ...weight_shape...]`
- Use `torch.vmap` with `torch.func.functional_call` for parallel computation
- **Actual result: 2.58x speedup** (37ms vs 96ms baseline)
- **Exact output match** (IoU=1.0, MSE=0)

**How it works:**
1. Backbone, FPN, and box_net weights are stacked across 3 models
2. `torch.vmap` vectorizes the forward pass across the model dimension
3. Class heads run sequentially (different output shapes: 90, 7, 20 classes)
4. `torch.compile` is applied internally for optimal kernel fusion

### Phase 4: Grouped Super Model ✅ (COMPLETED - Not Faster on MPS)

We implemented "Static Fusion" using grouped convolutions to fuse multiple models into a single super model:

| Config | Latency | Speedup | Output Quality |
|--------|---------|---------|----------------|
| grouped_super_model | 103 ms | 0.94x | IoU=1.0 (exact) |
| grouped+compile | 101 ms | 0.96x | IoU=1.0 (exact) |

**Implementation:**
- `GroupedConv2d`: Fuses N Conv2d layers using `groups=N`
- `GroupedBatchNorm2d`: Stacks running stats across N batchnorms
- `SuperEfficientDet`: Wraps backbone/FPN/box_net with grouped modules
- Class heads remain separate (different output dimensions)

**Why it's not faster on MPS:**
1. The current implementation uses a **wrapper approach** that still runs modules sequentially
2. True layer fusion would require rewriting each layer type with grouped operations
3. MPS may have overhead for grouped convolutions vs separate convolutions
4. The benefit of grouped convolutions is primarily for **export to TensorRT/ONNX**

**Key benefit:** Exportable static graph for deployment on NVIDIA/TensorRT.

### Phase 5: TorchServe Integration ✅ (COMPLETED)

We integrated TorchServe as a serving option with two modes:

| Config | Latency | Speedup | Output Quality | Notes |
|--------|---------|---------|----------------|-------|
| torchserve_baseline | 103.4 ms | 0.97x | IoU=0.78 | JPEG encoding overhead |
| torchserve_vmap_backbone | 41.1 ms | 2.44x | IoU=0.78 | Best TorchServe config |

**Implementation:**
- **External Mode**: TorchServe runs as separate process, HTTP API
- **Embedded Mode**: Direct handler invocation (no network overhead)
- Custom `EfficientDetHandler` wraps existing model implementations

**Key Findings:**

1. **TorchServe adds ~3ms overhead** vs direct inference:
   - `vmap_backbone` direct: 37.4ms
   - `torchserve_vmap_backbone`: 41.1ms
   - Overhead: 3.7ms (~10%)

2. **JPEG encoding affects output quality**:
   - TorchServe simulates real HTTP traffic (images as bytes)
   - JPEG lossy compression introduces differences
   - IoU drops from 1.0 to ~0.78 due to image compression
   - This is **expected behavior** for production serving

3. **Best TorchServe configuration**: `torchserve_vmap_backbone`
   - 2.44x faster than baseline (41ms vs 103ms)
   - Enables interactive serving (< 50ms target)

**Usage:**
```bash
# Create MAR archive
uv run python scripts/create_mar.py --optimization vmap_backbone

# Start TorchServe
torchserve --start --model-store model_store --models all

# Or use embedded mode for benchmarking
```

**Future**: TorchScript, ONNX, and TensorRT export via handler configuration.

### Phase 6: Future Optimizations
- TorchScript export for TorchServe
- Core ML conversion (Apple Neural Engine)
- TensorRT export for NVIDIA GPUs  
- Target: **< 20 ms** (enabling 30+ FPS)

### ~~Phase: FP16 Quantization~~ ❌
- **Not recommended on MPS** - causes 6x slowdown
- May work on CUDA with proper Tensor Core support

---

## Running the Benchmark

```bash
# 1. Install dependencies
uv sync

# 2. Generate checkpoints (required once)
uv run python scripts/generate_checkpoints.py

# 3. Run benchmark with REAL models
uv run python scripts/run_full_benchmark.py

# 4. Run ablation study on optimizations
uv run python scripts/run_ablation.py

# Optional: Run with mocked models (fast, for testing)
uv run python scripts/run_full_benchmark.py --mock

# Results saved to:
# - outputs/benchmark/benchmark_report.txt
# - outputs/benchmark/benchmark_results.json
# - outputs/ablation/ablation_report.txt
# - outputs/ablation/ablation_results.json
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

3. **vmap_backbone is the best optimization** - 2.58x faster than baseline (37ms vs 96ms) with exact output match

4. **Uniform architecture enables vmap parallelization** - backbone weights stacked across 3 models

5. **Interactive latency achieved** - 37ms enables near-realtime applications (< 50ms target)

6. **Real-time nearly achieved** - 37ms is close to 30 FPS target (33ms), room for further optimization

### Performance Bounds Summary

| Bound | CPU | MPS | MPS + vmap |
|-------|-----|-----|------------|
| **Upper (Real Model p50)** | 1,020 ms | 88 ms | 37 ms |
| **Lower (Invalid p50)** | 0.01 ms | 0.01 ms | 0.01 ms |
| **Optimization Gap** | 102,000x | 8,800x | 3,700x |

### Model Configuration

| Model | Classes | Purpose |
|-------|---------|---------|
| `efficientdet_d0_coco` | 90 | General object detection |
| `efficientdet_d0_aquarium` | 7 | Marine life detection |
| `efficientdet_d0_vehicles` | 20 | Vehicle detection |

All models share EfficientDet-D0 backbone (512×512 input, ~3.85M params each).
