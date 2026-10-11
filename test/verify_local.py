#!/usr/bin/env python3
"""Local reference validation helper for matrix_least_squares."""
import numpy as np

def generate_case(m=32, n=16, k=4, batch_size=1, cond=50.0, seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(batch_size):
        # Generate matrix with controlled condition number
        U, _ = np.linalg.qr(rng.standard_normal((m, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.linspace(1.0, 1.0 / cond, n)
        A = (U * s) @ V
        B = rng.standard_normal((m, k))
        
        # Reference solution using gels
        sol, res_sum, rank, _ = np.linalg.lstsq(A, B, rcond=None)
        if m > n:
            res_direct = np.sum((A @ sol - B)**2, axis=0)
        else:
            res_direct = np.zeros((0,), dtype=np.float32)
        cases.append((A.astype(np.float32), B.astype(np.float32), sol.astype(np.float32), res_direct.astype(np.float32)))
    return cases

if __name__ == "__main__":
    cases = generate_case(m=64, n=32, k=4, batch_size=2, cond=100.0)
    print(f"Generated {len(cases)} test cases successfully.")
    for i, (A, B, sol, res) in enumerate(cases):
        print(f"Case {i+1}: A={A.shape}, B={B.shape}, sol={sol.shape}, res={res.shape}")
