#!/usr/bin/env python3
"""Analyze inference bottlenecks: model forward vs post-processing."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.optimized import OptimizedImpl, IMAGE_SIZE
from src.models.optimizations.base import OptimizationConfig


def create_test_image(size: int = IMAGE_SIZE) -> Image.Image:
    """Create a random test image."""
    data = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


def analyze_baseline(impl: OptimizedImpl, input_tensor: torch.Tensor, num_iterations: int = 50):
    """Analyze baseline (sequential) inference timing breakdown."""
    print("\n" + "="*70)
    print("BASELINE (Sequential) - Timing Breakdown")
    print("="*70)
    
    preprocess_times = []
    forward_times = []
    postprocess_times = []
    total_times = []
    
    for i in range(num_iterations):
        # Total time
        total_start = time.perf_counter()
        
        # Forward pass for each model
        forward_start = time.perf_counter()
        raw_outputs = []
        for model in impl.models:
            with torch.no_grad():
                out = model(input_tensor)
                raw_outputs.append(out)
        
        if impl.device.type == "mps":
            torch.mps.synchronize()
        elif impl.device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - forward_start
        
        # Post-processing (parsing detections)
        postprocess_start = time.perf_counter()
        for det in raw_outputs:
            if det is not None and len(det) > 0:
                d = det[0]
                boxes = d[:, :4].float()
                scores = d[:, 4].float()
                labels = d[:, 5].long()
        postprocess_time = time.perf_counter() - postprocess_start
        
        total_time = time.perf_counter() - total_start
        
        if i >= 5:  # Skip warmup
            forward_times.append(forward_time * 1000)
            postprocess_times.append(postprocess_time * 1000)
            total_times.append(total_time * 1000)
    
    print(f"\nResults ({num_iterations - 5} iterations after warmup):")
    print(f"  Forward pass (3 models):  {np.mean(forward_times):>8.2f} ms ± {np.std(forward_times):.2f}")
    print(f"  Post-processing:          {np.mean(postprocess_times):>8.2f} ms ± {np.std(postprocess_times):.2f}")
    print(f"  Total:                    {np.mean(total_times):>8.2f} ms ± {np.std(total_times):.2f}")
    print(f"\n  Forward % of total:       {100 * np.mean(forward_times) / np.mean(total_times):.1f}%")
    print(f"  Post-process % of total:  {100 * np.mean(postprocess_times) / np.mean(total_times):.1f}%")
    
    return {
        "forward_ms": np.mean(forward_times),
        "postprocess_ms": np.mean(postprocess_times),
        "total_ms": np.mean(total_times),
    }


def analyze_vmap(impl: OptimizedImpl, input_tensor: torch.Tensor, num_iterations: int = 50):
    """Analyze vmap inference timing breakdown."""
    from effdet.bench import _post_process, _batch_detection
    
    print("\n" + "="*70)
    print("VMAP_BACKBONE - Timing Breakdown")
    print("="*70)
    
    vmap_opt = impl.optimization_stack.vmap_optimization
    
    forward_times = []
    class_net_times = []
    postprocess_times = []
    nms_times = []
    total_times = []
    
    for i in range(num_iterations):
        total_start = time.perf_counter()
        
        # Vmapped forward (backbone + FPN + box_net)
        forward_start = time.perf_counter()
        with torch.no_grad():
            fpn_features_all, box_outs_all, class_outs_all = vmap_opt.wrap_forward(
                impl.models, input_tensor
            )
        if impl.device.type == "mps":
            torch.mps.synchronize()
        forward_time = time.perf_counter() - forward_start
        
        # Post-processing for each model
        postprocess_start = time.perf_counter()
        for model_idx, model in enumerate(impl.models):
            bench = model
            model_box_out = [b[model_idx] for b in box_outs_all]
            model_class_out = class_outs_all[model_idx]
            
            # Post-process
            class_out_pp, box_out_pp, indices, classes = _post_process(
                model_class_out,
                model_box_out,
                num_levels=bench.num_levels,
                num_classes=bench.num_classes,
                max_detection_points=bench.max_detection_points,
            )
            
            # NMS
            detections = _batch_detection(
                input_tensor.shape[0],
                class_out_pp,
                box_out_pp,
                bench.anchors.boxes,
                indices,
                classes,
                img_scale=None,
                img_size=None,
                max_det_per_image=bench.max_det_per_image,
                soft_nms=bench.soft_nms,
            )
            
            # Parse
            if detections is not None and len(detections) > 0:
                det = detections[0]
                valid_mask = det[:, 4] > 0
                det = det[valid_mask]
        
        postprocess_time = time.perf_counter() - postprocess_start
        total_time = time.perf_counter() - total_start
        
        if i >= 5:  # Skip warmup
            forward_times.append(forward_time * 1000)
            postprocess_times.append(postprocess_time * 1000)
            total_times.append(total_time * 1000)
    
    print(f"\nResults ({num_iterations - 5} iterations after warmup):")
    print(f"  Vmapped forward (backbone+FPN+box): {np.mean(forward_times):>8.2f} ms ± {np.std(forward_times):.2f}")
    print(f"  Post-processing (3 models):         {np.mean(postprocess_times):>8.2f} ms ± {np.std(postprocess_times):.2f}")
    print(f"  Total:                              {np.mean(total_times):>8.2f} ms ± {np.std(total_times):.2f}")
    print(f"\n  Forward % of total:                 {100 * np.mean(forward_times) / np.mean(total_times):.1f}%")
    print(f"  Post-process % of total:            {100 * np.mean(postprocess_times) / np.mean(total_times):.1f}%")
    
    return {
        "forward_ms": np.mean(forward_times),
        "postprocess_ms": np.mean(postprocess_times),
        "total_ms": np.mean(total_times),
    }


def analyze_detailed_forward(impl: OptimizedImpl, input_tensor: torch.Tensor, num_iterations: int = 30):
    """Analyze per-component timing within forward pass."""
    print("\n" + "="*70)
    print("DETAILED FORWARD PASS BREAKDOWN (per model)")
    print("="*70)
    
    # Get raw model (unwrap DetBenchPredict)
    model = impl.models[0]
    raw_model = model.model
    
    backbone_times = []
    fpn_times = []
    class_net_times = []
    box_net_times = []
    
    for i in range(num_iterations):
        with torch.no_grad():
            # Backbone
            start = time.perf_counter()
            features = raw_model.backbone(input_tensor)
            if impl.device.type == "mps":
                torch.mps.synchronize()
            backbone_time = time.perf_counter() - start
            
            # FPN
            start = time.perf_counter()
            fpn_features = raw_model.fpn(features)
            if impl.device.type == "mps":
                torch.mps.synchronize()
            fpn_time = time.perf_counter() - start
            
            # Class net
            start = time.perf_counter()
            class_out = raw_model.class_net(fpn_features)
            if impl.device.type == "mps":
                torch.mps.synchronize()
            class_net_time = time.perf_counter() - start
            
            # Box net
            start = time.perf_counter()
            box_out = raw_model.box_net(fpn_features)
            if impl.device.type == "mps":
                torch.mps.synchronize()
            box_net_time = time.perf_counter() - start
        
        if i >= 5:  # Skip warmup
            backbone_times.append(backbone_time * 1000)
            fpn_times.append(fpn_time * 1000)
            class_net_times.append(class_net_time * 1000)
            box_net_times.append(box_net_time * 1000)
    
    total = np.mean(backbone_times) + np.mean(fpn_times) + np.mean(class_net_times) + np.mean(box_net_times)
    
    print(f"\nPer-model component timing ({num_iterations - 5} iterations):")
    print(f"  Backbone (EfficientNet-B0):  {np.mean(backbone_times):>6.2f} ms ({100*np.mean(backbone_times)/total:.1f}%)")
    print(f"  FPN (BiFPN):                 {np.mean(fpn_times):>6.2f} ms ({100*np.mean(fpn_times)/total:.1f}%)")
    print(f"  Class Net:                   {np.mean(class_net_times):>6.2f} ms ({100*np.mean(class_net_times)/total:.1f}%)")
    print(f"  Box Net:                     {np.mean(box_net_times):>6.2f} ms ({100*np.mean(box_net_times)/total:.1f}%)")
    print(f"  Total per model:             {total:>6.2f} ms")
    print(f"  Estimated 3 models:          {total * 3:>6.2f} ms")
    
    return {
        "backbone_ms": np.mean(backbone_times),
        "fpn_ms": np.mean(fpn_times),
        "class_net_ms": np.mean(class_net_times),
        "box_net_ms": np.mean(box_net_times),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create test image and preprocess
    test_image = create_test_image()
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # Load baseline model
    print("\nLoading baseline model...")
    baseline_impl = OptimizedImpl(device=device, optimization_config=OptimizationConfig())
    baseline_impl.load()
    baseline_impl.warmup()
    
    # Analyze baseline
    baseline_results = analyze_baseline(baseline_impl, input_tensor)
    
    # Detailed forward breakdown
    detailed_results = analyze_detailed_forward(baseline_impl, input_tensor)
    
    # Load vmap model
    print("\nLoading vmap_backbone model...")
    vmap_impl = OptimizedImpl(
        device=device, 
        optimization_config=OptimizationConfig(vmap_backbone_enabled=True)
    )
    vmap_impl.load()
    vmap_impl.warmup()
    
    # Analyze vmap
    vmap_results = analyze_vmap(vmap_impl, input_tensor)
    
    # Summary
    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n1. BASELINE (Sequential):")
    print(f"   Forward pass:    {baseline_results['forward_ms']:.1f} ms ({100*baseline_results['forward_ms']/baseline_results['total_ms']:.0f}%)")
    print(f"   Post-processing: {baseline_results['postprocess_ms']:.1f} ms ({100*baseline_results['postprocess_ms']/baseline_results['total_ms']:.0f}%)")
    
    print("\n2. VMAP_BACKBONE:")
    print(f"   Forward pass:    {vmap_results['forward_ms']:.1f} ms ({100*vmap_results['forward_ms']/vmap_results['total_ms']:.0f}%)")
    print(f"   Post-processing: {vmap_results['postprocess_ms']:.1f} ms ({100*vmap_results['postprocess_ms']/vmap_results['total_ms']:.0f}%)")
    
    print("\n3. PER-MODEL COMPONENT BREAKDOWN:")
    total_component = detailed_results['backbone_ms'] + detailed_results['fpn_ms'] + detailed_results['class_net_ms'] + detailed_results['box_net_ms']
    print(f"   Backbone:  {detailed_results['backbone_ms']:.1f} ms ({100*detailed_results['backbone_ms']/total_component:.0f}%)")
    print(f"   FPN:       {detailed_results['fpn_ms']:.1f} ms ({100*detailed_results['fpn_ms']/total_component:.0f}%)")
    print(f"   Class Net: {detailed_results['class_net_ms']:.1f} ms ({100*detailed_results['class_net_ms']/total_component:.0f}%)")
    print(f"   Box Net:   {detailed_results['box_net_ms']:.1f} ms ({100*detailed_results['box_net_ms']/total_component:.0f}%)")
    
    print("\n4. KEY FINDINGS:")
    if baseline_results['forward_ms'] > baseline_results['postprocess_ms'] * 5:
        print("   ✅ Model forward pass is the PRIMARY bottleneck")
        print("   ✅ Post-processing is negligible (<5% of time)")
    else:
        pct = 100 * baseline_results['postprocess_ms'] / baseline_results['total_ms']
        print(f"   ⚠️ Post-processing takes {pct:.0f}% of time - significant overhead")
    
    speedup = baseline_results['total_ms'] / vmap_results['total_ms']
    print(f"\n   vmap speedup: {speedup:.2f}x")
    print(f"   vmap forward speedup: {baseline_results['forward_ms'] / vmap_results['forward_ms']:.2f}x")


if __name__ == "__main__":
    main()
