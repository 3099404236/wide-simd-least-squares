# Early Abandonment Protocol (The 3-Try Rule)

## Principle: Hyperband / Successive Halving for Kernel Engineering

A primary pitfall of AI coding agents is getting trapped in a **local minimum**—endlessly modifying tile strides, loop bounds, or unroll pragmas on an inherently flawed memory or pipeline architecture.

### The 3-Evaluation Rule
For any chosen architectural sketch:
1. **Attempt 1 (Functional Baseline)**:
   - Compile, run verification test suite.
   - If functional or numerical failure occurs, apply exact numerical/boundary compensation.
2. **Attempt 2 (Pipeline & Resource Tuning)**:
   - Align memory accesses to hardware cache-line/vector-width boundaries (e.g., 32-byte / 512-bit).
   - Insert ping-pong double buffers.
   - Re-evaluate performance & hardware utilization.
3. **Attempt 3 (Decision Point)**:
   - **Keep & Promote**: If performance reaches >= 75% of roofline ceiling OR shows steep descent slope (> 30% improvement per attempt).
   - **ABANDON IMMEDIATELY**: If performance plateaus, memory stalls dominate (> 50% idle time), or on-chip memory limits prevent scaling.

### Pruning & Jumping Protocol
When abandoning a branch:
1. Log the exact failure mode in `reflexion.json` (e.g., "Memory bank serialization cannot be hidden", "PipeBarrier latency dominates loop body").
2. Do NOT touch micro-parameters anymore on this branch.
3. Jump to an orthogonal Macro-Sketch from the archive.
