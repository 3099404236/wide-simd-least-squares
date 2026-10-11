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
│   └── simd_machine.hpp                   # SIMD machine API definition (192KB SPAD)
├── kernel/
│   ├── matrix_least_squares.cpp           # Clean kernel implementation
│   ├── matrix_least_squares_layout.h      # Layout/Tiling configuration struct
│   └── layout_key_matrix_least_squares.h  # Template argument selector
├── host/
│   └── matrix_least_squares.cpp           # Host layout & workspace calculation
└── test/
    └── verify_local.py                    # Local reference validation script
```

---

## 6. Implementation Recommendations
- **Recommended Algorithm**: Householder QR Decomposition followed by backward substitution:
  $$A = Q R = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$$
  $$R_1 X = Q_1^T B$$
  $$\text{residuals}[:, j] = \|Q_2^T B[:, j]\|_2^2$$
- **Precision**: Since $\text{cond}_2(A) \le 10^3$, standard single-precision Householder QR with compensated accumulation (or Dekker double-float dot products for Householder vector accumulation) easily achieves $< 10^{-4}$ tolerance without complex iterative refinement.
