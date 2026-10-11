#pragma once
#include "simd_runtime/layout/template_argument.hpp"

SIMD_TPL_ARGS_DECL(MatrixLeastSquares,
    SIMD_TPL_DTYPE_DECL(DT_A, C_DT_FLOAT),
);

SIMD_TPL_SEL(
    SIMD_TPL_ARGS_SEL(
        SIMD_TPL_DTYPE_SEL(DT_A, C_DT_FLOAT),
    ),
);