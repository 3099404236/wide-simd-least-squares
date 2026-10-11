// Wide-SIMD Compute Kernel: Matrix Least Squares
#include "simd_machine.hpp"

#include "matrix_least_squares_layout.h"
#include "layout_key_matrix_least_squares.h"

template <class DT_A>
class KernelMatrixLeastSquares {
public:
    SIMD_FN inline KernelMatrixLeastSquares() {}

    SIMD_FN inline void Init(MemAddr A, MemAddr B, MemAddr solution, MemAddr residuals, MemAddr workspace,
                             const MatrixLeastSquaresLayoutData &layout) {
        this->m = layout.m;
        this->n = layout.n;
        this->k = layout.k;
        this->batch_size = layout.batch_size;
        this->has_residuals = layout.has_residuals;
        this->b_has_batch = layout.b_has_batch;

        aMem.Bind((SIMD_MEM float *)A);
        bMem.Bind((SIMD_MEM float *)B);
        solMem.Bind((SIMD_MEM float *)solution);
        resMem.Bind((SIMD_MEM float *)residuals);
        wsMem.Bind((SIMD_MEM float *)workspace);
    }

    SIMD_FN inline void Process() {
        // TODO: Implement least squares solver min_X ||A X - B||_F
        // Constraints:
        //   m >= n always (never underdetermined)
        //   cond_2(A) <= 1e3 (well-conditioned)
        //   residuals output only when m > n (has_residuals == 1)
    }

private:
    uint32_t m = 0, n = 0, k = 0, batch_size = 0, has_residuals = 0, b_has_batch = 0;
    MemRef<float> aMem, bMem, solMem, resMem, wsMem;
};

template <typename DT_A>
SIMD_ENTRY SIMD_FN void matrix_least_squares(MemAddr A, MemAddr B, MemAddr solution, MemAddr residuals, MemAddr workspace, MemAddr tiling) {
    REGISTER_LAYOUT_DEFAULT(MatrixLeastSquaresLayoutData);
    ACQUIRE_LAYOUT_STRUCT(MatrixLeastSquaresLayoutData, layout_data, tiling);
    KernelMatrixLeastSquares<DT_A> op;
    op.Init(A, B, solution, residuals, workspace, layout_data);
    op.Process();
}
