#pragma once
#include <cstdint>

struct MatrixLeastSquaresLayoutData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t batch_size;
    uint32_t has_residuals;
    uint32_t b_has_batch;
    // Cores per batch tile (>= 1). Above 1, each of them redoes the QR of A and takes a share of the right-hand
    // sides (k is then a multiple of 8, so the shares own whole 32 B blocks of both outputs).
    uint32_t helpers;
};