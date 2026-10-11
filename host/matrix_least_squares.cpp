// Host侧Tiling实现 —— MatrixLeastSquares
#include "interface/operator_registry.hpp"
#include "interface/platform_hardware.hpp"

#include "../device_kernel/matrix_least_squares_layout.h"
#include "../device_kernel/layout_key_matrix_least_squares.h"

namespace kernel_layout {
    static hpc_rt::hpc_status_t ComputeLayout(simd_rt::LayoutContext *context) {
        auto platform = target_platform::TargetPlatform(context->GetHardwareProfile());
        int32_t num_simd_cores = platform.GetSimdUnitCount();
        if (num_simd_cores <= 0) num_simd_cores = 24;

        uint32_t m = 1;
        uint32_t n = 1;
        uint32_t k = 1;
        uint32_t batch_size = 1;
        uint32_t b_has_batch = 0;

        const simd_rt::BufferShape *shape_A_storage = context->GetArrayShape(0);
        const simd_rt::BufferShape *shape_B_storage = context->GetArrayShape(1);
        if (shape_A_storage != nullptr && shape_B_storage != nullptr) {
            const auto &shape_A = shape_A_storage->GetBufferShape();
            const auto &shape_B = shape_B_storage->GetBufferShape();
            size_t dimA = shape_A.GetArrayRank();
            size_t dimB = shape_B.GetArrayRank();
            int64_t total_A = shape_A.GetArrayElementCount();

            int64_t total_B = shape_B.GetArrayElementCount();

            if (dimA >= 2) {
                m = static_cast<uint32_t>(shape_A.GetArrayDimension(dimA - 2));
                n = static_cast<uint32_t>(shape_A.GetArrayDimension(dimA - 1));
                if (total_A > 0 && m > 0 && n > 0) {
                    batch_size = static_cast<uint32_t>(total_A / (m * n));
                }
            }
            if (dimB == dimA) {
                // B is batched matrix: [*, m, k]
                k = static_cast<uint32_t>(shape_B.GetArrayDimension(dimB - 1));
                b_has_batch = (batch_size > 1 && total_B > m * k) ? 1 : 0;
            } else if (dimB == dimA - 1 && dimA > 2 && shape_B.GetArrayDimension(dimB - 1) == m && total_B == batch_size * m) {
                // B is batched vector: [*, m]
                k = 1;
                b_has_batch = (batch_size > 1) ? 1 : 0;
            } else if (dimB == 2) {
                // B is non-batched matrix: [m, k]
                k = static_cast<uint32_t>(shape_B.GetArrayDimension(1));
                b_has_batch = 0;
            } else if (dimB == 1) {
                // B is non-batched vector: [m]
                k = 1;
                b_has_batch = 0;
            } else {
                if (total_B == m) {
                    k = 1;
                    b_has_batch = 0;
                } else if (total_B == batch_size * m) {
                    k = 1;
                    b_has_batch = (batch_size > 1) ? 1 : 0;
                } else if (total_B % (batch_size * m) == 0 && total_B > 0) {
                    k = static_cast<uint32_t>(total_B / (batch_size * m));
                    b_has_batch = (batch_size > 1) ? 1 : 0;
                } else if (total_B % m == 0 && total_B > 0) {
                    k = static_cast<uint32_t>(total_B / m);
                    b_has_batch = 0;
                } else {
                    k = static_cast<uint32_t>((dimB >= 1) ? shape_B.GetArrayDimension(dimB - 1) : 1);
                    b_has_batch = (batch_size > 1) ? 1 : 0;
                }
            }
        }

        const simd_rt::ArrayRef *mat_A = context->GetMandatoryInput(0);
        hpc_rt::DataType dtype_A = (mat_A != nullptr) ? mat_A->GetDataType() : hpc_rt::DTYPE_F32;
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);
        SIMD_TPL_SELECT_PARAM(context, DT_A);

        MatrixLeastSquaresLayoutData *tiling = context->GetLayoutConfig<MatrixLeastSquaresLayoutData>();
        tiling->m = m;
        tiling->n = n;
        tiling->k = k;
        tiling->batch_size = batch_size;
        tiling->has_residuals = (m > n) ? 1 : 0;
        tiling->b_has_batch = b_has_batch;

        context->SetParallelUnits(num_simd_cores);

        // The kernel gives each working core a workspace slot: A^T and its factor (N rows of Pad8(M) floats each, with
        // M = max(m, n), N = min(m, n)), then B^T (k rows of Pad8(m)) and X^T (k rows of Pad8(n)). Batches go out in
        // tiles whose outputs are whole 32 B blocks; a tile is solved by `helpers` cores (see below).
        const size_t big_dim = (m >= n) ? m : n;
        const size_t small_dim = (m >= n) ? n : m;
        const size_t main_slot = 2 * small_dim * (((big_dim + 7) / 8) * 8) + static_cast<size_t>(k) * (((m + 7) / 8) * 8) +
                                 static_cast<size_t>(k) * (((n + 7) / 8) * 8);
        // The smallest tile the kernel can choose (its residual output may be unbound), for the most cores it can use.
        uint32_t unit = 1;
        while ((static_cast<uint64_t>(unit) * n * k) % 8 != 0) ++unit;
        const size_t tiles = (static_cast<size_t>(batch_size) + unit - 1) / unit;
        // When there are fewer tiles than cores and the right-hand sides split into groups of 8 (whole 32 B blocks of
        // both outputs), the idle cores help: each core of a tile redoes the QR of A and solves a share of the
        // right-hand sides. Every working core gets its own workspace slot.
        uint32_t helpers = 1;
        if (k % 8 == 0 && tiles < static_cast<size_t>(num_simd_cores)) {
            const size_t perTile = static_cast<size_t>(num_simd_cores) / tiles;
            helpers = static_cast<uint32_t>((perTile < k / 8) ? perTile : k / 8);
            if (helpers < 1) helpers = 1;
        }
        tiling->helpers = helpers;
        const size_t cells = tiles * helpers;
        const size_t working = (cells < static_cast<size_t>(num_simd_cores)) ? cells : static_cast<size_t>(num_simd_cores);
        size_t ws_floats = working * main_slot;
        // The legacy kernel runs when KernelMatrixLeastSquares::Plan() finds no room in SPAD (same test: more right-hand
        // sides than Pad8(M), or 13 vectors of Pad8(M) plus 10 of Pad8(N) and 1472 floats of staging over 46000 floats).
        // It keeps the m rows of [A | B] (n + k floats each, padded to 8) per core, then per batch the m rows of its
        // factor and the m residual rows of its refinement.
        const size_t big_pad = ((big_dim + 7) / 8) * 8;
        const size_t small_pad = ((small_dim + 7) / 8) * 8;
        const bool legacy = (k > big_pad) || (13 * big_pad + 10 * small_pad + 1472 > 46000);
        if (legacy) {
            uint32_t s_stride = ((n + k + 7) / 8) * 8;
            size_t legacy_floats = static_cast<size_t>(num_simd_cores) * m * s_stride +
                                   static_cast<size_t>(batch_size) * n * s_stride;
            const size_t r_stride = static_cast<size_t>(((k + 7) / 8) * 8);
            const size_t per_batch_slots = static_cast<size_t>(batch_size) * m * (s_stride + r_stride);
            if (legacy_floats < per_batch_slots) legacy_floats = per_batch_slots;
            if (ws_floats < legacy_floats) ws_floats = legacy_floats;
        }
        if (ws_floats < static_cast<size_t>(2 * 1024 * 1024)) ws_floats = static_cast<size_t>(2 * 1024 * 1024);
        size_t *currentWorkspace = context->GetWorkspaceRequirements(1);
        currentWorkspace[0] = ws_floats * sizeof(float);
        return hpc_rt::HPC_STATUS_SUCCESS;
    }
}  // namespace kernel_layout

namespace hpc_rt {
    static hpc_status_t DeduceShape(simd_rt::ShapeDeduceContext *context) {
        const simd_rt::Shape *shape_A = context->GetArrayShape(0);
        const simd_rt::Shape *shape_B = context->GetArrayShape(1);
        simd_rt::Shape *shape_sol = context->GetResultShape(0);
        simd_rt::Shape *shape_res = context->GetResultShape(1);
        if (shape_A == nullptr || shape_B == nullptr || shape_sol == nullptr) {
            return HPC_STATUS_SUCCESS;
        }

        size_t dimA = shape_A->GetArrayRank();
        size_t dimB = shape_B->GetArrayRank();
        if (dimA < 2) return HPC_STATUS_FAILED;

        int64_t m = shape_A->GetArrayDimension(dimA - 2);
        int64_t n = shape_A->GetArrayDimension(dimA - 1);
        int64_t total_A = shape_A->GetArrayElementCount();
        int64_t total_B = shape_B->GetArrayElementCount();
        int64_t batch_size = (total_A > 0 && m > 0 && n > 0) ? (total_A / (m * n)) : 1;
        bool b_is_vector = (dimB == 1) || (dimB == dimA - 1 && shape_B->GetArrayDimension(dimB - 1) == m && total_B == batch_size * m);
        int64_t k = b_is_vector ? 1 : ((dimB >= 1) ? shape_B->GetArrayDimension(dimB - 1) : 1);

        if (b_is_vector) {
            if (dimA == 2) {
                shape_sol->SetDimNum(1);
                shape_sol->SetDim(0, n);
            } else {
                shape_sol->SetDimNum(dimA - 1);
                for (size_t i = 0; i < dimA - 2; ++i) {
                    shape_sol->SetDim(i, shape_A->GetArrayDimension(i));
                }
                shape_sol->SetDim(dimA - 2, n);
            }
        } else {
            shape_sol->SetDimNum(dimA);
            for (size_t i = 0; i < dimA - 2; ++i) {
                shape_sol->SetDim(i, shape_A->GetArrayDimension(i));
            }
            shape_sol->SetDim(dimA - 2, n);
            shape_sol->SetDim(dimA - 1, k);
        }

        if (shape_res != nullptr) {
            if (m > n) {
                if (dimA == 2) {
                    shape_res->SetDimNum(1);
                    shape_res->SetDim(0, k);
                } else {
                    shape_res->SetDimNum(dimA - 1);
                    for (size_t i = 0; i < dimA - 2; ++i) {
                        shape_res->SetDim(i, shape_A->GetArrayDimension(i));
                    }
                    shape_res->SetDim(dimA - 2, k);
                }
            } else {
                shape_res->SetDimNum(1);
                shape_res->SetDim(0, 0);
            }
        }
        return HPC_STATUS_SUCCESS;
    }

    static hpc_status_t DeduceDtype(simd_rt::DtypeDeduceContext *context) {
        context->SetResultDtype(0, hpc_rt::DTYPE_F32);
        context->SetResultDtype(1, hpc_rt::DTYPE_F32);
        return hpc_rt::HPC_STATUS_SUCCESS;
    }
}  // namespace hpc_rt

namespace ops {
    class MatrixLeastSquares : public OpDescriptor {
    public:
        explicit MatrixLeastSquares(const char *name) : OpDescriptor(name) {
            this->Input("A")
                .ArgRequirement(REQ_MANDATORY)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Input("B")
                .ArgRequirement(REQ_MANDATORY)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Output("solution")
                .ArgRequirement(REQ_MANDATORY)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Output("residuals")
                .ArgRequirement(REQ_OPTIONAL)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Attr("driver").AttrType(REQ_OPTIONAL).String("gels");
            this->Attr("rcond").AttrType(REQ_OPTIONAL).Float(-1.0);
            this->SetShapeDeduction(hpc_rt::DeduceShape).SetDtypeDeduction(hpc_rt::DeduceDtype);
            this->SimdCore()
                .ConfigureLayout(kernel_layout::ComputeLayout)
                .TargetProfile("simd_target_v2");
        }
    };
    REGISTER_OPERATOR(MatrixLeastSquares);
}  // namespace ops
