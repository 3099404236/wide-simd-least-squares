# Bottleneck Analysis Protocol

Based on the Meta KernelAgent and Roofline Model methodology.

## 1. Metric Classification
Classify the kernel's primary limitation into one of three regimes:
- **Memory-Bound**: Memory throughput (DRAM/HBM/GM) is close to peak, while compute arithmetic units are idle.
  - *Remedies*: Increase tile size, maximize temporal reuse in SRAM/SPAD, vectorize loads, eliminate redundant transfers.
- **Compute-Bound**: Vector/Matrix execution units are near 100% saturation.
  - *Remedies*: Use higher-throughput intrinsic instructions, reduce redundant math operations, balance FP32 vs BF16/FP16 pipelines.
- **Underutilized / Latency-Bound**: Neither memory nor compute reaches saturation (< 60%).
  - *Remedies*: High synchronization barrier stalls, warp/thread divergence, small problem size unable to saturate multi-cores, or pipeline bubbles.

## 2. Analytical Checklist Before Coding
- [ ] Working set fits inside available on-chip SRAM/SPAD.
- [ ] Vector loads are aligned to native hardware width (e.g. 32-byte / 512-bit).
- [ ] No unpadded power-of-two strides that cause bank conflicts.
- [ ] Tail and non-aligned dimensions handled by masking without scalar fallback loops.
