# Mathematical Specification: Linear Least Squares

## Problem Statement
Given $A \in \mathbb{R}^{m \times n}$ with $m \ge n$ and full column rank, and $B \in \mathbb{R}^{m \times k}$:
Find $X \in \mathbb{R}^{n \times k}$ minimizing:
$$f(X) = \|A X - B\|_F^2 = \sum_{j=1}^k \|A X_j - B_j\|_2^2$$

## Output Requirements
1. **Solution Matrix**: $X = (A^T A)^{-1} A^T B \in \mathbb{R}^{n \times k}$
2. **Residuals**:
   - If $m > n$: $r_j = \|A X_j - B_j\|_2^2 \in \mathbb{R}$ for each column $j \in \{0, \dots, k-1\}$.
   - If $m = n$: Empty tensor with 0 elements.

## Numerical Properties
- Column rank is always full ($\text{rank}(A) = n$).
- Condition number is bounded: $\kappa_2(A) = \sigma_{\max}(A) / \sigma_{\min}(A) \le 1000$.
- No rank-deficient matrices.
- No underdetermined ($m < n$) matrices.
