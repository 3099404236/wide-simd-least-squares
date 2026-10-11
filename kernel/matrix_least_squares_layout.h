#pragma once
#include <cstdint>

struct MatrixLeastSquaresLayoutData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t batch_size;
    uint32_t has_residuals;
    uint32_t b_has_batch;
};