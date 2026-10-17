# Macro-Sketch Exploration: Multi-Paradigm Kernel Architecture

When approaching an accelerator or SIMD kernel optimization task, do NOT jump directly into tweaking tile parameters or unroll factors. Parameter tuning will only climb the closest local peak.

To discover the global optimum, the AI agent must first define and explore **3 fundamentally orthogonal structural paradigms (Macro-Sketches)**.

---

## The 3-Paradigm Architecture Framework

### 1. Sketch A: Resident Scratchpad / On-Chip State (Memory-Reuse Paradigm)
- **Principle**: Keep intermediate matrices, Gram accumulators, or tridiagonal reduction bands persistently inside local on-chip SRAM/SPAD across loop iterations.
- **Goal**: Minimize DDR/HBM round-trips. Maximize arithmetic intensity (FLOPs / Byte).
- **Bottleneck Check**: Does the active working set fit within hardware on-chip capacity without register spilling or bank conflicts?

### 2. Sketch B: Asynchronous Streaming Pipeline (Latency-Hiding Paradigm)
- **Principle**: Decouple memory transfer (DMA/cp.async/MTE) from vector/matrix compute using double-buffering (ping-pong) or multi-stage pipelines.
- **Goal**: Fully overlap communication latency with compute units.
- **Bottleneck Check**: Is computation time per tile greater than or equal to data transfer time? Are synchronization barriers minimized?

### 3. Sketch C: Algorithmic Reformulation (Mathematical-Structure Paradigm)
- **Principle**: Re-formulate the mathematics to trade raw FLOP count for sequential efficiency or SIMD regularity.
  - *Example (Least Squares)*: Normal Equations with compensated accumulation ($A^T A x = A^T b$) vs Householder QR streaming vs Incremental Block Gram-Schmidt.
  - *Example (Symmetric Eigen)*: Cyclic Jacobi sweep vs Tridiagonalization (Householder) + Divide-and-Conquer / QR iteration.
- **Goal**: Avoid irregular data dependencies and irregular branch divergence.

---

## Agent Step-by-Step Prompt

```markdown
You are an expert HPC & Accelerator Kernel Architect.

Before writing implementation code for [OPERATOR_NAME]:
1. Propose 3 fundamentally orthogonal Macro-Sketches (Resident SPAD, Double-Buffered Streaming, and Algorithmic Reformulation).
2. For each sketch, provide an analytical breakdown:
   - Arithmetic intensity (FLOPs / Byte).
   - On-chip memory footprint per core/thread-block.
   - Theoretical ceiling and primary hardware bottleneck.
3. Select the most promising candidate as the primary branch to implement.
```
