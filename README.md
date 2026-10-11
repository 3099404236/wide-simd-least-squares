# 40-Core Wide-SIMD Compute Routine: Matrix Least Squares (`matrix_least_squares`)

## 1. Routine Overview
This compute routine solves the linear least squares problem:
$$\min_X \|A X - B\|_F$$
for a tall coefficient matrix $A \in \mathbb{R}^{m \times n}$ ($m \ge n$) and observation matrix $B \in \mathbb{R}^{m \times k}$.

When $m > n$, it also computes the squared 2-norm residuals for each right-hand side column:
$$\text{residuals}[:, j] = \|A X[:, j] - B[:, j]\|_2^2$$

When $m = n$ and $A$ is invertible, $X = A^{-1} B$, and `residuals` is an empty tensor (size 0).

---

## 2. Input and Output Tensors

| Tensor | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `A` | `[*, m, n]` | `float32` | Coefficient matrix (`*` denotes 0 or more batch dimensions). Column full-rank. |
| `B` | `[*, m, k]` | `float32` | Observation matrix (or vector `[*, m]`). |
| `solution` | `[*, n, k]` | `float32` | Least-squares solution matrix (or `[*, n]`). |
| `residuals` | `[*, k]` | `float32` | Sum of squared residuals $\|AX - B\|_2^2$. **Only output when $m > n$**. Empty when $m = n$. |

---

## 3. Dimension Constraints & Boundary Guarantees

- **Matrix Rows ($m$)**: $m \in [1, 1024]$.
- **Matrix Columns ($n$)**: $n \in [1, \min(m, 512)]$.
- **Right-Hand Columns ($k$)**: $k \in [1, 64]$.
- **Batch Dimension**: Batch size $\in [1, 32]$.
- **CRITICAL GUARANTEE**: **$m \ge n$ ALWAYS holds.**
  - There are NO underdetermined ($m < n$) systems.
  - Do NOT implement minimum-norm solvers or augmented system solvers for $m < n$.
- **Condition Number**: The input data guarantees that $A$ has full column rank and bounded condition number:
  $$\text{cond}_2(A) \le 10^3$$

---

## 4. Precision Acceptance Criteria

- Elements of `solution` and `residuals` (when non-empty) are verified element-wise:
  $$\text{atol} \le 10^{-4}, \quad \text{rtol} \le 10^{-4}$$
- Ground truth reference is standard LAPACK `gels` (`torch.linalg.lstsq(..., driver='gels')` or `numpy.linalg.lstsq`).

---

## 5. Project Layout

```text
wide-simd-least-squares/
├── README.md                              # This specification
├── specs/
│   └── least_squares_spec.md              # Detailed mathematical formulation & notes
├── include/
│   └── simd_machine.hpp                   # SIMD machine API definition (192KB SPAD), CPU simulator
├── kernel/
│   ├── matrix_least_squares.cpp           # Kernel implementation
│   ├── matrix_least_squares_layout.h      # Layout/Tiling configuration struct
│   └── layout_key_matrix_least_squares.h  # Template argument selector
├── host/
│   └── matrix_least_squares.cpp           # Host layout & workspace calculation
└── test/
    ├── verify_local.py                    # Local reference validation script
    ├── sim_harness.cpp                    # Runs the kernel on the CPU simulator, checks results and costs
    ├── trace_checker.hpp                  # Fence-hazard checker and per-phase profiler for the harness
    ├── stubs/                             # Host-build stub of the device template-argument header
    └── Makefile
```

---

## 6. Implementation Recommendations
- **Recommended Algorithm**: Householder QR Decomposition followed by backward substitution:
  $$A = Q R = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$$
  $$R_1 X = Q_1^T B$$
  $$\text{residuals}[:, j] = \|Q_2^T B[:, j]\|_2^2$$
- **Precision**: single-precision Householder QR alone does **not** meet the tolerance near $\text{cond}_2(A) = 10^3$.
  Least squares with a nonzero residual has an error term $\approx \text{cond}^2 \, \varepsilon \, \|r\| / \|A\|$, about
  $0.06\,\|r\|$ for float32, and backward-stable float32 QR (compensated or not) reaches it: measured errors are 3 to
  130 times the $10^{-4}$ tolerance (1024x512, cond $10^3$). One step of refinement with $r = B - AX$ and $A^T r$ in
  double-float brings them to about $10^{-3}$ of the tolerance (section 7).

---

## 7. Kernel Implementation (`kernel/matrix_least_squares.cpp`)

**Accuracy.** QR in float32 gives $x_0$; then one corrected-semi-normal-equations step
$x = x_0 + R^{-1} R^{-T} A^T (B - A x_0)$. $r = B - A x_0$ and $g = A^T r$ are computed with Dot2 steps (Dekker's exact
TwoProd from 12-bit Veltkamp halves, error-free TwoSum, about 45 bits), and $r$ is passed to $g$ as an unevaluated
sum $r_h + r_l$: the error of $g$ reaches $x$ multiplied by $\text{cond}^2$, and rounding $r$ to float32 first would
leave $A^+ (r - \text{fl}(r))$, up to $\text{cond} \cdot \varepsilon \, \|r\|$. Worst observed error: 0.008 of the tolerance
for linearly spaced singular values, 0.04 with all but one singular value at $10^{-3}$.

**Three ways through**, chosen per shape:
- *resident* (working set fits in SPAD, at most 256 rows): $[A | B]$ is loaded once and factored in place by a
  row-oriented QR (reductions across rows: aligned, no lane sums, one scalar read per row and step); all later phases
  read SPAD. About 6 block copies per matrix.
- *medium* ($[A | B]$ fits in SPAD and $m < 8(n + k)$): the same row-oriented QR, then $R$ and $Q^T B$ go to the
  workspace and the streaming solves and refinement follow.
- *streaming*: column-oriented, left-looking blocked QR on $A^T$ in the workspace, reflectors streamed back in
  double-buffered groups; dot products are halved to 8 lane partials (the last levels over a compact buffer) that the
  scalar unit adds.

**Parallelism.** Batches go out in tiles whose solution and residual outputs are whole 32 B blocks, so cores never
write the same block. When tiles leave cores idle and $k$ is a multiple of 8, up to $k/8$ cores share a tile: each
redoes the QR of $A$ and solves its own groups of 8 right-hand sides (layout field `helpers`; the host gives every
working core its own workspace slot).

**Synchronization.** A SIMD fence separates dependent vector instructions and precedes scalar reads of vector results;
a global fence separates block copies from the accesses that depend on them and follows scalar writes read by vector
code or copies. All SPAD accesses go through small wrappers that report to the harness's hazard checker when built
with `MLS_TRACE` (compiled out otherwise).

---

## 8. Testing on the CPU Simulator

```sh
make -C test run                              # built-in suite (27 shapes)
./test/sim_harness sweep 80 7                 # 80 random shapes, seed 7
./test/sim_harness 1000 500 64 1              # one shape: m n k batch [cond seed b_has_batch cores spectrum]
```

For each case the harness builds float32 inputs with a prescribed condition number (singular values linear,
geometric or clustered), sizes the workspace exactly as `host/` does, runs all 40 simulated cores, and reports:
- `ratio_x`, `ratio_r`: worst $|out - ref| / (10^{-4} + 10^{-4}|ref|)$ against a float64 solve of the same float32
  inputs (pass: $\le 1$);
- `faults`: simulator faults (alignment, out-of-bounds, SPAD overflow, two cores writing one 32 B block);
- `hazards`: accesses the device's asynchronous units could reorder (missing fences), and partially overlapping
  vector operands;
- `cyc/core`: estimated cycles of the busiest core (the device critical path), from the simulator counters:
  1 per 256 B vector repeat, 4 per vector instruction and per scalar SPAD access, 20 per SIMD fence, 100 per global
  fence, 500 per block copy plus 1 per 64 B. Single-shape runs also print the counters per phase.

Estimated critical path (cycles per core), first correct-and-accurate version vs. current:

| m x n x k, batch | before | after | speedup |
| :--- | ---: | ---: | ---: |
| 8 x 4 x 1, 1 | 29.7K | 10.1K | 2.9x |
| 16 x 8 x 2, 4 | 197K | 73K | 2.7x |
| 64 x 32 x 4, 8 | 414K | 220K | 1.9x |
| 128 x 64 x 8, 4 | 627K | 360K | 1.7x |
| 100 x 100 x 7, 3 | 1.87M | 0.88M | 2.1x |
| 256 x 128 x 16, 2 | 2.30M | 1.20M | 1.9x |
| 300 x 300 x 2, 1 | 5.22M | 4.39M | 1.2x |
| 1024 x 64 x 4, 2 | 2.68M | 2.12M | 1.3x |
| 512 x 256 x 32, 1 | 11.2M | 5.1M | 2.2x |
| 1000 x 500 x 64, 1 | 66.2M | 23.6M | 2.8x |
| 200 x 100 x 1, 32 | 6.59M | 4.41M | 1.5x |
| 50 x 20 x 3, 16 | 1.00M | 0.51M | 2.0x |
