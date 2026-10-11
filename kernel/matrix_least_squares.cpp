// Wide-SIMD Compute Kernel: Matrix Least Squares
//
// min_X ||A X - B||_F for A [m, n] (m >= n, full column rank, cond_2(A) <= 1e3) and B [m, k], batched.
//
// Method (per matrix):
//   1. Householder QR of A in single precision, on the column-major copy A^T, left-looking in blocks of columns
//      that fit in SPAD; the right-hand sides ride along as extra columns, giving c = Q^T B.
//   2. x0 = R^{-1} c[0:n].
//   3. One step of iterative refinement (corrected semi-normal equations) with the residual in double-float:
//        r = B - A x0                 (exact products, error-free sums, rounded to float32 at the end)
//        g = A^T r                    (same)
//        x = x0 + R^{-1} R^{-T} g
//      Single-precision QR alone misses the 1e-4 tolerance for cond(A) near 1e3 (the cond^2 * eps * ||r|| term of
//      least squares); the refinement step brings the error down to the rounding of the result.
//   4. residuals = ||r||^2 per right-hand side (m > n).
//
// Data layout: SPAD vector operands are 32 B aligned and vector instructions are element-wise, so lanes (index mod 8)
// never mix in vector code. Column vectors are padded to M8 = Pad8(m) floats with zeros. Dot products are reduced by
// halving to 8 lane partials, which the scalar unit adds. Transposes (A -> A^T, B -> B^T, R -> R^T) go through
// EXTMEM with 4-byte block copies.
//
// Synchronization: a SIMD fence separates dependent vector instructions and precedes scalar reads of vector results;
// a global fence separates block copies from the vector or scalar accesses that depend on them (both ways) and
// follows scalar writes that vector code or block copies read.
//
// Work split: batches go out in tiles whose solution and residual outputs are whole 32 B blocks, so no two cores ever
// write the same block; one core solves every batch of a tile in turn.
//
// Workspace slot of a working core (floats), as sized by host/matrix_least_squares.cpp:
//   AT [n][M8]  A^T, later R^T rows ([n][N8])
//   VF [n][M8]  factored columns: R above the diagonal, the Householder vector from the diagonal down
//   BT [k][M8]  B^T, then A^T r (first N8 floats)
//   XT [k][N8]  Q^T B (first n rows), then x0, then x
#include "simd_machine.hpp"

#include "matrix_least_squares_layout.h"
#include "layout_key_matrix_least_squares.h"

using namespace hpc;

// Host-build instrumentation (test/sim_harness.cpp): every SPAD access, fence and block copy of the kernel is reported
// to a hazard checker and profiler. Compiled out unless MLS_TRACE is defined.
#ifdef MLS_TRACE
namespace mls_trace {
void Vec(uint32_t dst, uint32_t count, uint32_t src0, uint32_t src1);
void Scalar(uint32_t off, bool write);
void Fence(bool all);
void DmaIn(uint32_t dst, const void *src, const BlockCopyParams &prm);
void DmaOut(const void *dst, uint32_t src, const BlockCopyParams &prm);
void Phase(const char *name);
}  // namespace mls_trace
#define MLS_TRACE_HOOK(call) mls_trace::call
#else
#define MLS_TRACE_HOOK(call) ((void)0)
#endif

namespace mls {

constexpr uint32_t kSpadBytes = 195584;
constexpr uint32_t kSpadFloats = kSpadBytes / 4;
constexpr uint32_t kMaxBlockCols = 64;  // columns per QR block (bounded by the scalar scratch below)
constexpr uint32_t kMaxRhsGroup = 64;   // right-hand sides per refinement group
constexpr uint32_t kNone = 0xFFFFFFFFu;
// Transposes of at most this many rows go through the scalar unit (2 accesses of a few cycles per element, 8 * rows
// cycles per column) rather than 4-byte block copies (a spread copy per column and a compacting copy per batch).
constexpr uint32_t kScalarTransposeRows = 64;
// Problems with at most this many rows whose working set fits in SPAD take the SPAD-resident path.
constexpr uint32_t kResidentMaxRows = 256;
// Row-oriented QR in SPAD (resident path, and the medium path whose [A | B] fits in SPAD) for at most this many rows.
constexpr uint32_t kRowQrMaxRows = 640;
constexpr float kSplit = 4097.0f;  // 2^12 + 1: Veltkamp split of a float into two 12-bit halves

// Persistent SPAD area (float offsets).
constexpr uint32_t kMaskHi = 0;    // 8 blocks: maskHi[q][l] = (l >= q)
constexpr uint32_t kMaskLo = 64;   // 8 blocks: maskLo[q][l] = (l < q)
constexpr uint32_t kOneHot = 128;  // 8 blocks: oneHot[q][l] = (l == q)
constexpr uint32_t kTmp8 = 192;    // one scratch block
constexpr uint32_t kRes = 200;     // residuals of the current batch (k <= 64)
constexpr uint32_t kAlpha = 264;   // R diagonal (N8 floats), then tau (N8 floats)

SIMD_FN inline uint32_t Pad8(uint32_t x) { return (x + 7u) & ~7u; }
SIMD_FN inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace mls

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
        if (m == 0 || n == 0 || k == 0 || batch_size == 0) return;
        const uint32_t core = CoreId();
        const uint32_t cores = CoreCount();
        // Smallest tile whose outputs are whole 32 B blocks.
        uint32_t unit = 1;
        while ((unit * n * k) % 8 != 0 || (has_residuals != 0 && (unit * k) % 8 != 0)) ++unit;
        const uint32_t tiles = (batch_size + unit - 1) / unit;
        if (core >= tiles) return;

        M8 = mls::Pad8(m);
        N8 = mls::Pad8(n);
        const uint64_t slot = 2ull * n * M8 + static_cast<uint64_t>(k) * M8 + static_cast<uint64_t>(k) * N8;
        wsAT = static_cast<uint64_t>(core) * slot;
        wsVF = wsAT + static_cast<uint64_t>(n) * M8;
        wsBT = wsVF + static_cast<uint64_t>(n) * M8;
        wsXT = wsBT + static_cast<uint64_t>(k) * M8;
        kTau = mls::kAlpha + N8;
        kFree = kTau + N8;

        pipe.ReserveBlock(spadBlk, mls::kSpadBytes);
        sp = spadBlk.Get<float>();
        InitConstants();

        const bool resident = PlanResident();
        const bool medium = !resident && PlanMedium();
        for (uint32_t t = core; t < tiles; t += cores) {
            const uint32_t b1 = mls::Min((t + 1) * unit, batch_size);
            for (uint32_t b = t * unit; b < b1; ++b) {
                if (resident) {
                    SolveResident(b);
                } else if (medium) {
                    SolveMedium(b);
                } else {
                    SolveOne(b);
                }
            }
        }
    }

private:
    // ------------------------------------------------------------------ instruction wrappers (SPAD float offsets)

    SIMD_FN inline void VAdd(uint32_t d, uint32_t a, uint32_t b, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, a, b));
        Add(sp[d], sp[a], sp[b], n_);
    }
    SIMD_FN inline void VSub(uint32_t d, uint32_t a, uint32_t b, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, a, b));
        Sub(sp[d], sp[a], sp[b], n_);
    }
    SIMD_FN inline void VMul(uint32_t d, uint32_t a, uint32_t b, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, a, b));
        Mul(sp[d], sp[a], sp[b], n_);
    }
    SIMD_FN inline void VMax(uint32_t d, uint32_t a, uint32_t b, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, a, b));
        Max(sp[d], sp[a], sp[b], n_);
    }
    SIMD_FN inline void VMulS(uint32_t d, uint32_t a, float s, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, a, mls::kNone));
        MulScalar(sp[d], sp[a], s, n_);
    }
    SIMD_FN inline void VAbs(uint32_t d, uint32_t a, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, a, mls::kNone));
        Abs(sp[d], sp[a], n_);
    }
    SIMD_FN inline void VFill(uint32_t d, float v, uint32_t n_) {
        MLS_TRACE_HOOK(Vec(d, n_, mls::kNone, mls::kNone));
        Fill(sp[d], v, n_);
    }
    SIMD_FN inline float SRd(uint32_t off) {
        MLS_TRACE_HOOK(Scalar(off, false));
        return sp.ScalarRead(off);
    }
    SIMD_FN inline void SWr(uint32_t off, float v) {
        MLS_TRACE_HOOK(Scalar(off, true));
        sp.ScalarWrite(off, v);
    }
    SIMD_FN inline void FenceV() {
        MLS_TRACE_HOOK(Fence(false));
        Fence<Unit_Simd>();
    }
    SIMD_FN inline void FenceA() {
        MLS_TRACE_HOOK(Fence(true));
        Fence<Unit_All>();
    }

    // ------------------------------------------------------------------ data movement

    SIMD_FN inline void CopyIn(uint32_t dst, const MemRef<float> &src, const BlockCopyParams &prm, bool pad) {
        MLS_TRACE_HOOK(DmaIn(dst, src.Ptr(), prm));
        if (pad) {
            BlockCopyPad<float> zero{true, 0, 0, 0.0f};
            BlockCopyPartial(sp[dst], src, prm, zero);
        } else {
            BlockCopyPartial(sp[dst], src, prm);
        }
    }

    SIMD_FN inline void CopyOut(const MemRef<float> &dst, uint32_t src, const BlockCopyParams &prm) {
        MLS_TRACE_HOOK(DmaOut(dst.Ptr(), src, prm));
        BlockCopyPartial(dst, sp[src], prm);
    }

    // EXTMEM -> SPAD: `rows` rows of `len` floats, source rows `srcStride` floats apart, destination rows `dstStride`
    // floats apart (a multiple of 8, at least Pad8(len)); the tail of each destination row up to Pad8(len) is zeroed.
    SIMD_FN inline void Load(uint32_t dst, const MemRef<float> &src, uint32_t rows, uint32_t len, uint64_t srcStride,
                             uint32_t dstStride) {
        BlockCopyParams prm;
        prm.blockCount = static_cast<uint16_t>(rows);
        prm.blockLen = len * 4;
        prm.srcStride = (rows > 1) ? static_cast<uint32_t>((srcStride - len) * 4) : 0;
        prm.dstStride = (rows > 1) ? (dstStride - mls::Pad8(len)) / 8 : 0;
        prm.rsv = 0;
        CopyIn(dst, src, prm, true);
    }

    // SPAD -> EXTMEM: `rows` rows of `len` floats, source rows `srcStride` floats apart (multiple of 8), destination
    // rows `dstStride` floats apart.
    SIMD_FN inline void Store(const MemRef<float> &dst, uint32_t src, uint32_t rows, uint32_t len, uint64_t dstStride,
                              uint32_t srcStride) {
        BlockCopyParams prm;
        prm.blockCount = static_cast<uint16_t>(rows);
        prm.blockLen = len * 4;
        prm.srcStride = (rows > 1) ? (srcStride - mls::Pad8(len)) / 8 : 0;
        prm.dstStride = (rows > 1) ? static_cast<uint32_t>((dstStride - len) * 4) : 0;
        prm.rsv = 0;
        CopyOut(dst, src, prm);
    }

    // EXTMEM -> SPAD: `count` floats `stride` floats apart, each into lane 0 of consecutive 32 B blocks.
    SIMD_FN inline void Spread(uint32_t dst, const MemRef<float> &src, uint32_t count, uint64_t stride) {
        BlockCopyParams prm;
        prm.blockCount = static_cast<uint16_t>(count);
        prm.blockLen = 4;
        prm.srcStride = static_cast<uint32_t>((stride - 1) * 4);
        prm.dstStride = 0;
        prm.rsv = 0;
        CopyIn(dst, src, prm, false);
    }

    // SPAD -> EXTMEM: lane 0 of `count` consecutive 32 B blocks into `count` contiguous floats.
    SIMD_FN inline void Compact(const MemRef<float> &dst, uint32_t src, uint32_t count) {
        BlockCopyParams prm;
        prm.blockCount = static_cast<uint16_t>(count);
        prm.blockLen = 4;
        prm.srcStride = 0;
        prm.dstStride = 0;
        prm.rsv = 0;
        CopyOut(dst, src, prm);
    }

    // Transposes `cols` columns of a row-major matrix (`rows` x `ld`, starting at element `src`) into rows of `len`
    // floats (len >= rows, zero tail) at `dst`, `dstStride` floats apart. Short columns go through SPAD and the scalar
    // unit (2 rows * cols accesses and two block copies); long ones through EXTMEM with 4-byte-block copies: each
    // column is spread into lane 0 of 32 B blocks in SPAD, and a batch of them is compacted back in one copy.
    SIMD_FN inline void TransposeToRows(const MemRef<float> &dstMem, uint64_t dst, uint64_t dstStride,
                                        const MemRef<float> &srcMem, uint64_t src, uint32_t rows, uint32_t cols,
                                        uint32_t ld, uint32_t len, uint32_t buf) {
        const uint32_t ldp = mls::Pad8(cols);
        if (rows <= mls::kScalarTransposeRows && rows * ldp + cols * len <= mls::kSpadFloats - buf) {
            const uint32_t out = buf + rows * ldp;
            FenceA();
            Load(buf, srcMem[src], rows, cols, ld, ldp);
            VFill(out, 0.0f, cols * len);
            FenceA();
            for (uint32_t c = 0; c < cols; ++c) {
                for (uint32_t i = 0; i < rows; ++i) SWr(out + c * len + i, SRd(buf + i * ldp + c));
            }
            FenceA();
            Store(dstMem[dst], out, cols, len, dstStride, len);
            FenceA();
            return;
        }
        // Batches of wt columns: one spread copy per column into consecutive slots of 8 * len floats (zero beyond
        // `rows` blocks), one compacting copy per batch; two batch buffers alternate.
        const uint32_t slot = 8 * len;
        uint32_t wt = (mls::kSpadFloats - buf) / (2 * slot);
        if (wt > cols) wt = cols;
        if (wt < 1) wt = 1;
        const bool batched = (dstStride == len);
        if (!batched) wt = 1;
        FenceA();
        VFill(buf, 0.0f, 2 * wt * slot);
        FenceA();
        for (uint32_t c0 = 0, t = 0; c0 < cols; c0 += wt, ++t) {
            const uint32_t cnt = mls::Min(wt, cols - c0);
            const uint32_t bt = buf + (t & 1) * wt * slot;
            for (uint32_t c = 0; c < cnt; ++c) Spread(bt + c * slot, srcMem[src + c0 + c], rows, ld);
            FenceA();
            Compact(dstMem[dst + static_cast<uint64_t>(c0) * dstStride], bt, cnt * len);
        }
        FenceA();
    }

    // ------------------------------------------------------------------ vector building blocks

    // Reduces rows[q] = buf + q * stride + off (q < count), each `len` floats (multiple of 8), to 8 lane partials at
    // the start of the row, by addition or by maximum. Ends with a SIMD fence.
    SIMD_FN inline void Halve(uint32_t buf, uint32_t stride, uint32_t off, uint32_t count, uint32_t len,
                              bool useMax = false) {
        while (len > 8) {
            const uint32_t h = ((len / 8 + 1) / 2) * 8;
            for (uint32_t q = 0; q < count; ++q) {
                const uint32_t row = buf + q * stride + off;
                if (useMax) {
                    VMax(row, row, row + h, len - h);
                } else {
                    VAdd(row, row, row + h, len - h);
                }
            }
            FenceV();
            len = h;
        }
    }

    SIMD_FN inline float LaneSum(uint32_t off) {
        const float s0 = SRd(off) + SRd(off + 1);
        const float s1 = SRd(off + 2) + SRd(off + 3);
        const float s2 = SRd(off + 4) + SRd(off + 5);
        const float s3 = SRd(off + 6) + SRd(off + 7);
        return (s0 + s1) + (s2 + s3);
    }

    SIMD_FN inline float LaneMax(uint32_t off) {
        float r = SRd(off);
        for (uint32_t l = 1; l < 8; ++l) {
            const float v = SRd(off + l);
            if (v > r) r = v;
        }
        return r;
    }

    // Applies H = I - tau v v^T (v at `v`, zero outside [a, m)) to columns blk + q * M8, q in [q0, q1), using the
    // product rows prod + q * M8 as scratch. Long products are halved per row down to 128 floats, the next level
    // writes 64 floats per row into the compact buffer `cmp` (two halves of 64 floats per row), and the last three
    // levels each take one instruction over all rows.
    SIMD_FN inline void ApplyReflector(uint32_t v, float tau, uint32_t a, uint32_t blk, uint32_t prod, uint32_t cmp,
                                       uint32_t q0, uint32_t q1) {
        uint32_t len = M8 - a;
        const uint32_t nq = q1 - q0;
        for (uint32_t q = q0; q < q1; ++q) VMul(prod + q * M8 + a, v + a, blk + q * M8 + a, len);
        FenceV();
        if (len > 128) {
            while (len > 128) {
                uint32_t h = 64;
                while (2 * h < len) h *= 2;
                for (uint32_t q = q0; q < q1; ++q) {
                    const uint32_t row = prod + q * M8 + a;
                    VAdd(row, row, row + h, len - h);
                }
                FenceV();
                len = h;
            }
            const uint32_t ca = cmp, cb = cmp + nq * 64, total = nq * 64;
            for (uint32_t q = q0; q < q1; ++q) {
                const uint32_t row = prod + q * M8 + a;
                VAdd(ca + (q - q0) * 64, row, row + 64, 64);
            }
            FenceV();
            VAdd(cb, ca, ca + 32, total - 32);
            FenceV();
            VAdd(ca, cb, cb + 16, total - 16);
            FenceV();
            VAdd(cb, ca, ca + 8, total - 8);
            FenceV();
            for (uint32_t q = q0; q < q1; ++q) scal[q] = tau * LaneSum(cb + (q - q0) * 64);
        } else {
            Halve(prod + q0 * M8, M8, a, nq, len);
            for (uint32_t q = q0; q < q1; ++q) scal[q] = tau * LaneSum(prod + q * M8 + a);
        }
        len = M8 - a;
        for (uint32_t q = q0; q < q1; ++q) VMulS(prod + q * M8 + a, v + a, scal[q], len);
        FenceV();
        for (uint32_t q = q0; q < q1; ++q) VSub(blk + q * M8 + a, blk + q * M8 + a, prod + q * M8 + a, len);
        FenceV();
    }

    // ------------------------------------------------------------------ QR

    // Factors column p (block column q): builds its Householder vector in `vcl` (and in place, from the diagonal
    // down), records R[p][p] and tau, and applies the reflector to block columns (q, wb).
    SIMD_FN inline void FactorColumn(uint32_t p, uint32_t q, uint32_t wb, uint32_t blk, uint32_t prod, uint32_t vcl,
                                     uint32_t cmp) {
        const uint32_t a = p & ~7u;
        const uint32_t len = M8 - a;
        const uint32_t col = blk + q * M8;
        const uint32_t pr = prod + q * M8;
        VMul(vcl + a, col + a, mls::kMaskHi + (p - a) * 8, 8);
        if (len > 8) VMulS(vcl + a + 8, col + a + 8, 1.0f, len - 8);
        FenceV();
        VMul(pr + a, vcl + a, vcl + a, len);
        FenceV();
        Halve(pr, M8, a, 1, len);
        const float nrm = hpc::sqrt(LaneSum(pr + a));
        const float xp = SRd(col + p);
        const float alpha = (xp >= 0.0f) ? -nrm : nrm;
        const float axp = (xp >= 0.0f) ? xp : -xp;
        const float tau = 1.0f / (nrm * (nrm + axp));
        SWr(mls::kAlpha + p, alpha);
        SWr(kTau + p, tau);
        // v[p] = x[p] - alpha, in the clean copy and in the stored column.
        VMulS(mls::kTmp8, mls::kOneHot + (p - a) * 8, alpha, 8);
        FenceV();
        VSub(vcl + a, vcl + a, mls::kTmp8, 8);
        VSub(col + a, col + a, mls::kTmp8, 8);
        FenceV();
        if (q + 1 < wb) ApplyReflector(vcl, tau, a, blk, prod, cmp, q + 1, wb);
    }

    // Left-looking blocked Householder QR of [A | B] (A^T rows in AT, B^T rows in BT). Writes the factored columns to
    // VF and the first N8 entries of Q^T b to XT.
    SIMD_FN inline void FactorAndApply() {
        const uint32_t total = n + k;
        uint32_t g = 4;
        const uint32_t avail = mls::kSpadFloats - kFree;
        uint32_t w = 0;
        while (true) {
            const uint32_t fixed = (1 + 2 * g) * M8;
            w = (avail > fixed) ? (avail - fixed) / (2 * M8 + 128) : 0;
            if (w >= 2 || g == 1) break;
            --g;
        }
        w = mls::Min(w, mls::kMaxBlockCols);
        w = mls::Min(w, total);
        const uint32_t vcl = kFree;
        const uint32_t vbuf = vcl + M8;
        const uint32_t blk = vbuf + 2 * g * M8;
        const uint32_t prod = blk + w * M8;
        const uint32_t cmp = prod + w * M8;  // 2 x w x 64 floats

        for (uint32_t c0 = 0; c0 < total; c0 += w) {
            const uint32_t c1 = mls::Min(c0 + w, total);
            const uint32_t wb = c1 - c0;
            const uint32_t na = (c0 < n) ? mls::Min(c1, n) - c0 : 0;
            const uint32_t nb = wb - na;
            if (na > 0) Load(blk, wsMem[wsAT + static_cast<uint64_t>(c0) * M8], 1, na * M8, na * M8, na * M8);
            if (nb > 0) {
                const uint32_t r0 = c0 + na - n;
                Load(blk + na * M8, wsMem[wsBT + static_cast<uint64_t>(r0) * M8], 1, nb * M8, nb * M8, nb * M8);
            }
            FenceA();

            // Reflectors of earlier blocks, g at a time, double-buffered.
            const uint32_t prev = mls::Min(c0, n);
            if (prev > 0) {
                const uint32_t groups = (prev + g - 1) / g;
                Load(vbuf, wsMem[wsVF], 1, mls::Min(g, prev) * M8, M8, M8);
                for (uint32_t gi = 0; gi < groups; ++gi) {
                    const uint32_t p0 = gi * g;
                    const uint32_t gp = mls::Min(g, prev - p0);
                    const uint32_t cur = vbuf + (gi & 1) * g * M8;
                    FenceA();
                    if (gi + 1 < groups) {
                        const uint32_t p1 = p0 + g;
                        const uint32_t gn = mls::Min(g, prev - p1);
                        Load(vbuf + ((gi + 1) & 1) * g * M8, wsMem[wsVF + static_cast<uint64_t>(p1) * M8], 1, gn * M8,
                             gn * M8, gn * M8);
                    }
                    // Zero R entries left of the diagonal in the reflector's first block.
                    for (uint32_t j = 0; j < gp; ++j) {
                        const uint32_t p = p0 + j;
                        const uint32_t a = p & ~7u;
                        const uint32_t v = cur + j * M8;
                        VMul(v + a, v + a, mls::kMaskHi + (p - a) * 8, 8);
                    }
                    FenceV();
                    for (uint32_t j = 0; j < gp; ++j) {
                        const uint32_t p = p0 + j;
                        ApplyReflector(cur + j * M8, SRd(kTau + p), p & ~7u, blk, prod, cmp, 0, wb);
                    }
                }
            }

            for (uint32_t q = 0; q < na; ++q) FactorColumn(c0 + q, q, wb, blk, prod, vcl, cmp);

            FenceA();
            if (na > 0) Store(wsMem[wsVF + static_cast<uint64_t>(c0) * M8], blk, 1, na * M8, na * M8, na * M8);
            if (nb > 0) {
                const uint32_t r0 = c0 + na - n;
                Store(wsMem[wsXT + static_cast<uint64_t>(r0) * N8], blk + na * M8, nb, N8, N8, M8);
            }
            FenceA();
        }
    }

    // ------------------------------------------------------------------ triangular solves

    // Solves R x = c: c rows at `cv` (cnt rows, N8 apart) are consumed, x rows are written to `xv` (N8 apart, zero
    // beyond n). R columns come from VF; `rbuf` holds `gr` of them.
    SIMD_FN inline void BackSolve(uint32_t cv, uint32_t xv, uint32_t tmp, uint32_t cnt, uint32_t rbuf, uint32_t gr) {
        VFill(xv, 0.0f, cnt * N8);
        uint32_t lo = n;  // R columns [lo, hi) are in rbuf
        for (uint32_t jj = n; jj-- > 0;) {
            if (jj < lo) {
                const uint32_t hi = jj + 1;
                lo = (hi > gr) ? hi - gr : 0;
                FenceA();
                Load(rbuf, wsMem[wsVF + static_cast<uint64_t>(lo) * M8], hi - lo, N8, M8, N8);
                FenceA();
                for (uint32_t j = lo; j < hi; ++j) {
                    if ((j & 7u) != 0) {
                        const uint32_t blkOff = rbuf + (j - lo) * N8 + (j & ~7u);
                        VMul(blkOff, blkOff, mls::kMaskLo + (j & 7u) * 8, 8);
                    }
                }
            }
            FenceV();
            const float ajj = SRd(mls::kAlpha + jj);
            for (uint32_t r = 0; r < cnt; ++r) {
                scal[r] = SRd(cv + r * N8 + jj) / ajj;
                SWr(xv + r * N8 + jj, scal[r]);
            }
            if (jj == 0) break;
            const uint32_t len = mls::Pad8(jj);
            const uint32_t rc = rbuf + (jj - lo) * N8;
            for (uint32_t r = 0; r < cnt; ++r) VMulS(tmp + r * N8, rc, scal[r], len);
            FenceV();
            for (uint32_t r = 0; r < cnt; ++r) VSub(cv + r * N8, cv + r * N8, tmp + r * N8, len);
        }
        FenceA();
    }

    // Solves R^T h = g: g rows at `gv` are consumed, h rows written to `hv`. Rows of R come either from the R^T area
    // (AT, rows N8 apart, zero left of and on the diagonal) or, when rRowsInVf is set, from VF row i past its diagonal
    // (rows M8 apart; the first used block is masked).
    SIMD_FN inline void ForwardSolve(uint32_t gv, uint32_t hv, uint32_t tmp, uint32_t cnt, uint32_t rbuf, uint32_t gr) {
        VFill(hv, 0.0f, cnt * N8);
        uint32_t hi = 0;  // R rows [lo, hi) are in rbuf
        uint32_t lo = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (i >= hi) {
                lo = i;
                hi = mls::Min(n, i + gr);
                FenceA();
                if (rRowsInVf) {
                    Load(rbuf, wsMem[wsVF + static_cast<uint64_t>(lo) * M8], hi - lo, N8, M8, N8);
                    FenceA();
                    for (uint32_t j = lo; j < hi; ++j) {
                        const uint32_t a = (j + 1) & ~7u;
                        if (((j + 1) & 7u) != 0 && a < N8) {
                            const uint32_t blkOff = rbuf + (j - lo) * N8 + a;
                            VMul(blkOff, blkOff, mls::kMaskHi + ((j + 1) - a) * 8, 8);
                        }
                    }
                } else {
                    Load(rbuf, wsMem[wsAT + static_cast<uint64_t>(lo) * N8], 1, (hi - lo) * N8, N8, N8);
                    FenceA();
                }
            }
            FenceV();
            const float aii = SRd(mls::kAlpha + i);
            for (uint32_t r = 0; r < cnt; ++r) {
                scal[r] = SRd(gv + r * N8 + i) / aii;
                SWr(hv + r * N8 + i, scal[r]);
            }
            if (i + 1 == n) break;
            const uint32_t a = (i + 1) & ~7u;
            const uint32_t len = N8 - a;
            const uint32_t rr = rbuf + (i - lo) * N8 + a;
            for (uint32_t r = 0; r < cnt; ++r) VMulS(tmp + r * N8 + a, rr, scal[r], len);
            FenceV();
            for (uint32_t r = 0; r < cnt; ++r) VSub(gv + r * N8 + a, gv + r * N8 + a, tmp + r * N8 + a, len);
        }
        FenceA();
    }

    // ------------------------------------------------------------------ double-float accumulation

    // One Dot2 step for r < cnt: (Sh_r, Sl_r) += a * (x_r + y_r), with x_r = dx[r] (12-bit halves in dxh[r], dxl[r])
    // and an optional small y_r = dy[r] (|y_r| <= 2^-24 |x_r|; skipped when withY is false). a * x_r is split exactly
    // into P + E (Dekker's TwoProd, from the 12-bit halves of a at ah / al), P enters Sh through an error-free TwoSum,
    // and E, a * y_r and the TwoSum error enter Sl, so Sl only ever holds terms below 2^-24 of the products and the
    // pair keeps about 45 bits; folding the 2^-12 partial products into Sl instead would stop near 36 bits, which is
    // not enough for A^T r (its error reaches x multiplied by cond(A)^2).
    // Accumulators are stored by field: field f of all cnt right-hand sides is the contiguous run
    // base + f * cnt * len (row r at + r * len), so only the products by per-row scalars take one instruction per
    // right-hand side. Fields: 0 Sh, 1 S2 (the two swap roles through `flip`), 2 Sl, 3 P, 4 E, 5 T, 6 W, 7 Q (with y).
    SIMD_FN inline void Dot2(uint32_t a, uint32_t ah, uint32_t al, uint32_t base, uint32_t cnt, uint32_t len,
                             bool withY) {
        const uint32_t f = cnt * len;
        const uint32_t sh = base + (flip ? f : 0);
        const uint32_t s2 = base + (flip ? 0 : f);
        const uint32_t sl = base + 2 * f, pp = base + 3 * f, ee = base + 4 * f, tt = base + 5 * f, ww = base + 6 * f;
        const uint32_t qq = base + 7 * f;
        for (uint32_t r = 0; r < cnt; ++r) {
            VMulS(pp + r * len, a, dx[r], len);    // P = fl(a x)
            VMulS(ee + r * len, ah, dxh[r], len);  // ah xh
            VMulS(tt + r * len, ah, dxl[r], len);  // ah xl
            if (withY) VMulS(qq + r * len, a, dy[r], len);
        }
        FenceV();
        VSub(ee, ee, pp, f);  // E = ah xh - P
        VAdd(s2, sh, pp, f);  // s = Sh + P
        if (withY) VAdd(sl, sl, qq, f);
        FenceV();
        VAdd(ee, ee, tt, f);  // E += ah xl
        VSub(ww, s2, sh, f);  // bb = s - Sh
        FenceV();
        for (uint32_t r = 0; r < cnt; ++r) VMulS(tt + r * len, al, dxh[r], len);  // al xh
        VSub(pp, pp, ww, f);                                                     // P - bb
        FenceV();
        VAdd(ee, ee, tt, f);  // E += al xh
        VSub(ww, s2, ww, f);  // s - bb
        FenceV();
        for (uint32_t r = 0; r < cnt; ++r) VMulS(tt + r * len, al, dxl[r], len);  // al xl
        VSub(ww, sh, ww, f);                                                     // Sh - (s - bb)
        VAdd(sl, sl, pp, f);                                                     // Sl += P - bb
        FenceV();
        VAdd(ee, ee, tt, f);  // E += al xl: the exact error of P
        VAdd(sl, sl, ww, f);  // Sl += Sh - (s - bb)
        FenceV();
        VAdd(sl, sl, ee, f);  // Sl += E
        FenceV();
        flip = !flip;
    }

    // Splits the `len` floats at `src` into 12-bit halves at `hi` and `lo`.
    SIMD_FN inline void SplitVec(uint32_t src, uint32_t hi, uint32_t lo, uint32_t len) {
        VMulS(hi, src, mls::kSplit, len);
        FenceV();
        VSub(lo, hi, src, len);
        FenceV();
        VSub(hi, hi, lo, len);
        FenceV();
        VSub(lo, src, hi, len);
        FenceV();
    }

    SIMD_FN inline void SetMultiplier(uint32_t r, float x, float y) {
        const float t = x * mls::kSplit;
        dx[r] = x;
        dxh[r] = t - (t - x);
        dxl[r] = x - dxh[r];
        dy[r] = y;
    }

    // Refinement products for every right-hand side, in groups of `kg`:
    //   r = B - A x0 over M8-long vectors (A^T rows from AT, B^T rows from BT, x0 rows from XT), kept in SPAD as
    //       rh + rl; ||rh||^2 goes to the residual slots;
    //   g = A^T (rh + rl) over N8-long vectors (A rows from the input), written over the B^T rows.
    // Both in Dot2 double-float. Rounding r to float32 before forming g would leave an error A^+ (r - fl(r)) in x, up
    // to cond(A) * eps * ||r||, so r stays in two parts.
    // SPAD of a group of c right-hand sides. Residual: the accumulators (7 fields of c x M8) and the x0 rows (c x N8)
    // at the end of SPAD, two blocks of ra A^T rows and their 12-bit halves from the start of the free area, where the
    // r rows (rh and rl, 2c x M8) go once the last A^T row is used. Gradient: the r rows, the accumulators (8 fields of
    // c x N8), two blocks of rc A rows and their 12-bit halves.
    SIMD_FN inline void RefinePass(uint64_t aOff) {
        const uint32_t avail = mls::kSpadFloats - kFree;
        uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
        uint32_t ra = 1, rc = 1;
        for (; kg >= 1; --kg) {
            const uint32_t needR = 7 * kg * M8 + kg * N8;  // accumulators and x0 rows; A^T blocks go over the r rows
            const uint32_t needG = 2 * kg * M8 + 8 * kg * N8;
            if (needR + 4 * M8 > avail || 2 * kg * M8 + needR > avail || needG + 4 * N8 > avail) continue;
            ra = mls::Min(mls::Min(n, 16), (avail - needR) / (4 * M8));
            rc = mls::Min(mls::Min(m, 64), (avail - needG) / (4 * N8));
            if (kg == 1 || (ra >= mls::Min(n, 4) && rc >= mls::Min(m, 8))) break;
        }
        for (uint32_t r0 = 0; r0 < k; r0 += kg) {
            const uint32_t cnt = mls::Min(kg, k - r0);
            const uint32_t rArr = kFree;  // rh[r] at rArr + r * M8, rl[r] at rArr + (cnt + r) * M8
            const uint32_t xb = mls::kSpadFloats - cnt * N8;
            const uint32_t acc = xb - 7 * cnt * M8;

            // --- r = B - A x0.
            {
                const uint32_t raw = kFree;  // two blocks of ra A^T rows, then their 12-bit halves
                const uint32_t ahv = raw + 2 * ra * M8;
                const uint32_t alv = ahv + ra * M8;
                FenceA();
                VFill(acc, 0.0f, 7 * cnt * M8);
                FenceA();
                Load(acc, wsMem[wsBT + static_cast<uint64_t>(r0) * M8], cnt, M8, M8, M8);
                Load(xb, wsMem[wsXT + static_cast<uint64_t>(r0) * N8], 1, cnt * N8, N8, N8);
                Load(raw, wsMem[wsAT], 1, mls::Min(ra, n) * M8, M8, M8);
                flip = false;
                for (uint32_t c0 = 0, t = 0; c0 < n; c0 += ra, ++t) {
                    const uint32_t rows = mls::Min(ra, n - c0);
                    const uint32_t cur = raw + (t & 1) * ra * M8;
                    FenceA();
                    if (c0 + ra < n) {
                        const uint32_t nxt = mls::Min(ra, n - c0 - ra);
                        Load(raw + ((t + 1) & 1) * ra * M8, wsMem[wsAT + static_cast<uint64_t>(c0 + ra) * M8], 1,
                             nxt * M8, M8, M8);
                    }
                    SplitVec(cur, ahv, alv, rows * M8);
                    for (uint32_t j = 0; j < rows; ++j) {
                        for (uint32_t r = 0; r < cnt; ++r) SetMultiplier(r, -SRd(xb + r * N8 + c0 + j), 0.0f);
                        Dot2(cur + j * M8, ahv + j * M8, alv + j * M8, acc, cnt, M8, false);
                    }
                }
            }

            // --- rh + rl = Sh + Sl (TwoSum), ||rh||^2.
            {
                const uint32_t f = cnt * M8;
                const uint32_t sh = acc + (flip ? f : 0), sl = acc + 2 * f, pp = acc + 3 * f, ee = acc + 4 * f;
                const uint32_t tt = acc + 5 * f;
                const uint32_t rh = rArr, rl = rArr + f;
                VAdd(rh, sh, sl, f);
                FenceV();
                VSub(pp, rh, sh, f);  // bb = rh - Sh
                VMul(tt, rh, rh, f);  // rh^2
                FenceV();
                VSub(ee, rh, pp, f);  // rh - bb
                VSub(sl, sl, pp, f);  // Sl - bb
                FenceV();
                if (has_residuals != 0) Halve(tt, M8, 0, cnt, M8);
                VSub(ee, sh, ee, f);  // Sh - (rh - bb)
                FenceV();
                if (has_residuals != 0) {
                    for (uint32_t r = 0; r < cnt; ++r) SWr(mls::kRes + r0 + r, LaneSum(tt + r * M8));
                }
                VAdd(rl, ee, sl, f);
                FenceV();
            }

            // --- g = A^T (rh + rl), accumulators after the r rows.
            {
                const uint32_t gacc = rArr + 2 * cnt * M8;
                const uint32_t raw = gacc + 8 * cnt * N8;  // two blocks of rc A rows
                const uint32_t ahv = raw + 2 * rc * N8;
                const uint32_t alv = ahv + rc * N8;
                VFill(gacc, 0.0f, 8 * cnt * N8);
                FenceA();
                Load(raw, aMem[aOff], mls::Min(rc, m), n, n, N8);
                flip = false;
                for (uint32_t i0 = 0, t = 0; i0 < m; i0 += rc, ++t) {
                    const uint32_t rows = mls::Min(rc, m - i0);
                    const uint32_t cur = raw + (t & 1) * rc * N8;
                    FenceA();
                    if (i0 + rc < m) {
                        const uint32_t nxt = mls::Min(rc, m - i0 - rc);
                        Load(raw + ((t + 1) & 1) * rc * N8, aMem[aOff + static_cast<uint64_t>(i0 + rc) * n], nxt, n, n,
                             N8);
                    }
                    SplitVec(cur, ahv, alv, rows * N8);
                    for (uint32_t i = 0; i < rows; ++i) {
                        for (uint32_t r = 0; r < cnt; ++r) {
                            SetMultiplier(r, SRd(rArr + r * M8 + i0 + i), SRd(rArr + (cnt + r) * M8 + i0 + i));
                        }
                        Dot2(cur + i * N8, ahv + i * N8, alv + i * N8, gacc, cnt, N8, true);
                    }
                }
                const uint32_t f = cnt * N8;
                VAdd(gacc + 3 * f, gacc + (flip ? f : 0), gacc + 2 * f, f);  // g = Sh + Sl into the P field
                FenceA();
                Store(wsMem[wsBT + static_cast<uint64_t>(r0) * M8], gacc + 3 * f, cnt, N8, M8, N8);
                FenceA();
            }
        }
    }

    // Right-hand sides per group for triangular solves that keep `rows` vectors of N8 floats per right-hand side and
    // stream R in blocks through the rest of SPAD, `solves` times per group: each group pays two fences per step and
    // each block a pair of global fences and a copy, so this trades group count against block count.
    SIMD_FN inline uint32_t SolveGroup(uint32_t rows, uint32_t solves) {
        const uint32_t avail = mls::kSpadFloats - kFree;
        uint32_t best = 1;
        uint32_t bestCost = 0xFFFFFFFFu;
        for (uint32_t kg = 1; kg <= mls::Min(k, mls::kMaxRhsGroup); ++kg) {
            if ((rows * kg + 1) * N8 > avail) break;
            const uint32_t gr = mls::Min(n, (avail - rows * kg * N8) / N8);
            const uint32_t groups = (k + kg - 1) / kg;
            const uint32_t blocks = (n + gr - 1) / gr;
            const uint32_t cost = groups * solves * (2 * n * 20 + blocks * 700);
            if (cost < bestCost) {
                bestCost = cost;
                best = kg;
            }
        }
        return best;
    }

    // ------------------------------------------------------------------ one matrix

    // ------------------------------------------------------------------ SPAD-resident path

    // Small problems keep everything in SPAD: [A | B] is loaded once, factored in place by a row-oriented Householder
    // QR (reductions run across rows, so they need no lane sums and only one scalar read per row and step), and every
    // later phase reads SPAD instead of the workspace.
    // Layout (float offsets): aug [m][Wr] = [A | B] rows (Wr = N8 + K8), arow [m][N8] = A, brow [m][K8] = B, then
    // during the QR its scratch qrT [m][Wr] and wrow [Wr], afterwards at [n][M8] = A^T, bt [k][M8] = B^T,
    // rcol [n][N8] = R columns (zero from the diagonal down), the k-row vector sets xa, x0, xb, tmp ([k][N8] each),
    // and the refinement working set.
    SIMD_FN inline bool PlanResident() {
        K8 = mls::Pad8(k);
        Wr = N8 + K8;
        if (m > mls::kResidentMaxRows) return false;
        rAug = kFree;
        rArow = rAug + m * Wr;
        rBrow = rArow + m * N8;
        rQrT = rBrow + m * K8;
        rWrow = rQrT + m * Wr;
        if (rWrow + Wr > mls::kSpadFloats) return false;
        rAt = rQrT;
        rBt = rAt + n * M8;
        rRcol = rBt + k * M8;
        rXa = rRcol + n * N8;
        rX0 = rXa + k * N8;
        rXb = rX0 + k * N8;
        rTmp = rXb + k * N8;
        rRef = rTmp + k * N8;
        if (rRef + mls::Pad8(n * k) > mls::kSpadFloats) return false;
        rSbT = mls::Min(n, 16);
        rSbA = mls::Min(m, 16);
        for (rKg = mls::Min(k, mls::kMaxRhsGroup); rKg >= 1; --rKg) {
            const uint32_t needR = 9 * rKg * M8 + 2 * rSbT * M8;
            const uint32_t needG = 2 * rKg * M8 + 8 * rKg * N8 + 2 * rSbA * N8;
            if (rRef + (needR > needG ? needR : needG) <= mls::kSpadFloats) break;
        }
        return rKg >= 1;
    }

    // Row-oriented Householder QR of the first n columns of the m x wr row-major matrix at `aug`, applied to all of
    // it. Products for w = v^T aug go through `T` (`chunk` rows of wr floats) and are halved across rows, partial
    // sums of several chunks collect in `wacc`; `wrow` holds tau * w. R ends up in rows 0..n-1 (diagonal in kAlpha).
    SIMD_FN inline void RowQR(uint32_t aug, uint32_t wr, uint32_t T, uint32_t chunk, uint32_t wacc, uint32_t wrow) {
        for (uint32_t p = 0; p < n; ++p) {
            const uint32_t len0 = m - p;
            const uint32_t ap = p & ~7u;
            const uint32_t wa = wr - ap;
            FenceV();
            float nrm2 = 0.0f;
            for (uint32_t i = 0; i < len0; ++i) {
                const float x = SRd(aug + (p + i) * wr + p);
                vsc[i] = x;
                nrm2 += x * x;
            }
            const float nrm = hpc::sqrt(nrm2);
            const float xp = vsc[0];
            const float alpha = (xp >= 0.0f) ? -nrm : nrm;
            const float axp = (xp >= 0.0f) ? xp : -xp;
            const float tau = 1.0f / (nrm * (nrm + axp));
            vsc[0] = xp - alpha;
            SWr(mls::kAlpha + p, alpha);
            // w = v^T aug[p:m, ap:wr): products per row, halved across rows.
            const bool single = (len0 <= chunk);
            for (uint32_t i0 = 0; i0 < len0; i0 += chunk) {
                const uint32_t cl = mls::Min(chunk, len0 - i0);
                for (uint32_t i = 0; i < cl; ++i) {
                    VMulS(T + i * wr + ap, aug + (p + i0 + i) * wr + ap, vsc[i0 + i], wa);
                }
                FenceV();
                for (uint32_t len = cl; len > 1;) {
                    const uint32_t h = (len + 1) / 2;
                    VAdd(T + ap, T + ap, T + h * wr + ap, (len - h) * wr - ap);
                    FenceV();
                    len = h;
                }
                if (!single) {
                    if (i0 == 0) {
                        VMulS(wacc + ap, T + ap, 1.0f, wa);
                    } else {
                        VAdd(wacc + ap, wacc + ap, T + ap, wa);
                    }
                    FenceV();
                }
            }
            // aug[p:m, :] -= v (tau w)^T over whole rows (tau w is zero left of ap).
            if (ap > 0) VFill(wrow, 0.0f, ap);
            VMulS(wrow + ap, single ? T + ap : wacc + ap, tau, wa);
            FenceV();
            for (uint32_t i0 = 0; i0 < len0; i0 += chunk) {
                const uint32_t cl = mls::Min(chunk, len0 - i0);
                for (uint32_t i = 0; i < cl; ++i) VMulS(T + i * wr, wrow, vsc[i0 + i], wr);
                FenceV();
                VSub(aug + (p + i0) * wr, aug + (p + i0) * wr, T, cl * wr);
                if (i0 + cl < len0) FenceV();
            }
        }
        FenceV();
    }

    // Solves R x = c with R columns resident at rRcol (zero from the diagonal down): c rows at `cv` are consumed,
    // x rows written to `xv`.
    SIMD_FN inline void BackSolveResident(uint32_t cv, uint32_t xv, uint32_t tmp, uint32_t cnt) {
        VFill(xv, 0.0f, cnt * N8);
        for (uint32_t jj = n; jj-- > 0;) {
            FenceV();
            const float ajj = SRd(mls::kAlpha + jj);
            for (uint32_t r = 0; r < cnt; ++r) {
                scal[r] = SRd(cv + r * N8 + jj) / ajj;
                SWr(xv + r * N8 + jj, scal[r]);
            }
            if (jj == 0) break;
            const uint32_t len = mls::Pad8(jj);
            for (uint32_t r = 0; r < cnt; ++r) VMulS(tmp + r * N8, rRcol + jj * N8, scal[r], len);
            FenceV();
            for (uint32_t r = 0; r < cnt; ++r) VSub(cv + r * N8, cv + r * N8, tmp + r * N8, len);
        }
        FenceA();
    }

    // Solves R^T h = g with R rows resident in aug (their entries left of and on the diagonal zeroed first): g rows at
    // `gv` are consumed, h rows written to `hv`.
    SIMD_FN inline void ForwardSolveResident(uint32_t gv, uint32_t hv, uint32_t tmp, uint32_t cnt) {
        VFill(hv, 0.0f, cnt * N8);
        for (uint32_t i = 0; i < n; ++i) {
            FenceV();
            const float aii = SRd(mls::kAlpha + i);
            for (uint32_t r = 0; r < cnt; ++r) {
                scal[r] = SRd(gv + r * N8 + i) / aii;
                SWr(hv + r * N8 + i, scal[r]);
            }
            if (i + 1 == n) break;
            const uint32_t a = (i + 1) & ~7u;
            const uint32_t len = N8 - a;
            const uint32_t rr = rAug + i * Wr + a;
            for (uint32_t r = 0; r < cnt; ++r) VMulS(tmp + r * N8 + a, rr, scal[r], len);
            FenceV();
            for (uint32_t r = 0; r < cnt; ++r) VSub(gv + r * N8 + a, gv + r * N8 + a, tmp + r * N8 + a, len);
        }
        FenceA();
    }

    // Refinement products as in RefinePass, from the resident A^T (at), B^T (bt), x0 (x0) and A rows (arow); g goes to
    // the rows at xb.
    SIMD_FN inline void RefineResident() {
        for (uint32_t r0 = 0; r0 < k; r0 += rKg) {
            const uint32_t cnt = mls::Min(rKg, k - r0);
            const uint32_t rArr = rRef;  // rh[r] at rArr + r * M8, rl[r] at rArr + (cnt + r) * M8
            const uint32_t acc = rArr + 2 * cnt * M8;
            {
                const uint32_t ahv = acc + 7 * cnt * M8;
                const uint32_t alv = ahv + rSbT * M8;
                FenceV();
                VMulS(acc, rBt + r0 * M8, 1.0f, cnt * M8);  // Sh = b
                VFill(acc + 2 * cnt * M8, 0.0f, cnt * M8);  // Sl = 0
                FenceV();
                flip = false;
                for (uint32_t c0 = 0; c0 < n; c0 += rSbT) {
                    const uint32_t rows = mls::Min(rSbT, n - c0);
                    SplitVec(rAt + c0 * M8, ahv, alv, rows * M8);
                    for (uint32_t j = 0; j < rows; ++j) {
                        for (uint32_t r = 0; r < cnt; ++r) {
                            SetMultiplier(r, -SRd(rX0 + (r0 + r) * N8 + c0 + j), 0.0f);
                        }
                        Dot2(rAt + (c0 + j) * M8, ahv + j * M8, alv + j * M8, acc, cnt, M8, false);
                    }
                }
            }
            {
                const uint32_t f = cnt * M8;
                const uint32_t sh = acc + (flip ? f : 0), sl = acc + 2 * f, pp = acc + 3 * f, ee = acc + 4 * f;
                const uint32_t tt = acc + 5 * f;
                const uint32_t rh = rArr, rl = rArr + f;
                VAdd(rh, sh, sl, f);
                FenceV();
                VSub(pp, rh, sh, f);
                VMul(tt, rh, rh, f);
                FenceV();
                VSub(ee, rh, pp, f);
                VSub(sl, sl, pp, f);
                FenceV();
                if (has_residuals != 0) Halve(tt, M8, 0, cnt, M8);
                VSub(ee, sh, ee, f);
                FenceV();
                if (has_residuals != 0) {
                    for (uint32_t r = 0; r < cnt; ++r) SWr(mls::kRes + r0 + r, LaneSum(tt + r * M8));
                }
                VAdd(rl, ee, sl, f);
                FenceV();
            }
            {
                const uint32_t gacc = rArr + 2 * cnt * M8;
                const uint32_t ahv = gacc + 8 * cnt * N8;
                const uint32_t alv = ahv + rSbA * N8;
                VFill(gacc, 0.0f, 3 * cnt * N8);  // Sh, S2, Sl
                FenceV();
                flip = false;
                for (uint32_t i0 = 0; i0 < m; i0 += rSbA) {
                    const uint32_t rows = mls::Min(rSbA, m - i0);
                    SplitVec(rArow + i0 * N8, ahv, alv, rows * N8);
                    for (uint32_t i = 0; i < rows; ++i) {
                        for (uint32_t r = 0; r < cnt; ++r) {
                            SetMultiplier(r, SRd(rArr + r * M8 + i0 + i), SRd(rArr + (cnt + r) * M8 + i0 + i));
                        }
                        Dot2(rArow + (i0 + i) * N8, ahv + i * N8, alv + i * N8, gacc, cnt, N8, true);
                    }
                }
                const uint32_t f = cnt * N8;
                VAdd(rXb + r0 * N8, gacc + (flip ? f : 0), gacc + 2 * f, f);
                FenceV();
            }
        }
    }

    SIMD_FN inline void SolveResident(uint32_t b) {
        const uint64_t aOff = static_cast<uint64_t>(b) * m * n;
        const uint64_t bOff = (b_has_batch != 0) ? static_cast<uint64_t>(b) * m * k : 0;

        MLS_TRACE_HOOK(Phase("r.load"));
        // Long columns of A and B are transposed through the workspace while SPAD past the inputs is still free.
        const bool viaWs = (m > mls::kScalarTransposeRows);
        if (viaWs) {
            TransposeToRows(wsMem, wsAT, M8, aMem, aOff, m, n, n, M8, rQrT);
            TransposeToRows(wsMem, wsBT, M8, bMem, bOff, m, k, k, M8, rQrT);
        }
        FenceA();
        Load(rAug, aMem[aOff], m, n, n, Wr);
        Load(rAug + N8, bMem[bOff], m, k, k, Wr);
        Load(rArow, aMem[aOff], m, n, n, N8);
        Load(rBrow, bMem[bOff], m, k, k, K8);
        FenceA();

        MLS_TRACE_HOOK(Phase("r.qr"));
        RowQR(rAug, Wr, rQrT, m, mls::kNone, rWrow);

        // A^T, B^T, R columns and c = (Q^T B)[0:n] as rows, on the scalar unit.
        MLS_TRACE_HOOK(Phase("r.setup"));
        if (viaWs) {
            VFill(rRcol, 0.0f, rRef - rRcol);
            Load(rAt, wsMem[wsAT], 1, n * M8, M8, M8);
            Load(rBt, wsMem[wsBT], 1, k * M8, M8, M8);
            FenceA();
        } else {
            VFill(rAt, 0.0f, rRef - rAt);
            FenceV();
            for (uint32_t c = 0; c < n; ++c) {
                for (uint32_t i = 0; i < m; ++i) SWr(rAt + c * M8 + i, SRd(rArow + i * N8 + c));
            }
            for (uint32_t r = 0; r < k; ++r) {
                for (uint32_t i = 0; i < m; ++i) SWr(rBt + r * M8 + i, SRd(rBrow + i * K8 + r));
            }
        }
        for (uint32_t j = 1; j < n; ++j) {
            for (uint32_t i = 0; i < j; ++i) SWr(rRcol + j * N8 + i, SRd(rAug + i * Wr + j));
        }
        for (uint32_t r = 0; r < k; ++r) {
            for (uint32_t j = 0; j < n; ++j) SWr(rXa + r * N8 + j, SRd(rAug + j * Wr + N8 + r));
        }
        // R rows for the forward solve: zero the entries left of and on the diagonal in each row's first used block.
        for (uint32_t i = 0; i + 1 < n; ++i) {
            const uint32_t a = (i + 1) & ~7u;
            if (((i + 1) & 7u) != 0) VMul(rAug + i * Wr + a, rAug + i * Wr + a, mls::kMaskHi + ((i + 1) - a) * 8, 8);
        }
        FenceA();

        MLS_TRACE_HOOK(Phase("r.solve0"));
        BackSolveResident(rXa, rX0, rTmp, k);

        MLS_TRACE_HOOK(Phase("r.refine"));
        RefineResident();

        MLS_TRACE_HOOK(Phase("r.solve1"));
        ForwardSolveResident(rXb, rXa, rTmp, k);
        BackSolveResident(rXa, rXb, rTmp, k);
        VAdd(rX0, rX0, rXb, k * N8);
        FenceA();

        MLS_TRACE_HOOK(Phase("r.output"));
        const uint64_t xOff = static_cast<uint64_t>(b) * n * k;
        if (k == 1) {
            Store(solMem[xOff], rX0, 1, n, n, N8);
        } else {
            for (uint32_t c = 0; c < n; ++c) {
                for (uint32_t r = 0; r < k; ++r) SWr(rRef + c * k + r, SRd(rX0 + r * N8 + c));
            }
            FenceA();
            Store(solMem[xOff], rRef, 1, n * k, n * k, mls::Pad8(n * k));
        }
        if (has_residuals != 0) Store(resMem[static_cast<uint64_t>(b) * k], mls::kRes, 1, k, k, mls::Pad8(k));
        FenceA();
    }

    SIMD_FN inline void SolveOne(uint32_t b) {
        const uint64_t aOff = static_cast<uint64_t>(b) * m * n;
        const uint64_t bOff = (b_has_batch != 0) ? static_cast<uint64_t>(b) * m * k : 0;

        MLS_TRACE_HOOK(Phase("transpose"));
        TransposeToRows(wsMem, wsAT, M8, aMem, aOff, m, n, n, M8, kFree);
        TransposeToRows(wsMem, wsBT, M8, bMem, bOff, m, k, k, M8, kFree);

        MLS_TRACE_HOOK(Phase("qr"));
        FactorAndApply();
        rRowsInVf = false;
        FinishStreaming(b, aOff);
    }

    // Medium problems: [A | B] fits in SPAD for a row-oriented QR (cheaper than the column-oriented one unless the
    // matrix is much taller than wide), the rest streams like SolveOne. VF row j receives R column j left of the
    // diagonal and R row j right of it, so no R^T pass is needed.
    SIMD_FN inline bool PlanMedium() {
        K8 = mls::Pad8(k);
        Wr = N8 + K8;
        if (m > mls::kRowQrMaxRows || m >= 8 * (n + k)) return false;
        const uint32_t avail = mls::kSpadFloats - kFree;
        const uint32_t augF = m * Wr;
        if (augF + 2 * Wr + 8 * Wr > avail || augF + 8 * N8 + k * N8 > avail) return false;
        mChunk = mls::Min(m, (avail - augF - 2 * Wr) / Wr);
        return mChunk >= mls::Min(m, 8);
    }

    SIMD_FN inline void SolveMedium(uint32_t b) {
        const uint64_t aOff = static_cast<uint64_t>(b) * m * n;
        const uint64_t bOff = (b_has_batch != 0) ? static_cast<uint64_t>(b) * m * k : 0;
        const uint32_t aug = kFree;
        const uint32_t T = aug + m * Wr;
        const uint32_t wacc = T + mChunk * Wr;
        const uint32_t wrow = wacc + Wr;

        MLS_TRACE_HOOK(Phase("m.qr"));
        FenceA();
        Load(aug, aMem[aOff], m, n, n, Wr);
        Load(aug + N8, bMem[bOff], m, k, k, Wr);
        FenceA();
        RowQR(aug, Wr, T, mChunk, wacc, wrow);

        // VF rows (R column j left of the diagonal, R row j right of it) in chunks, and the c rows of XT.
        MLS_TRACE_HOOK(Phase("m.export"));
        const uint32_t ct = T;
        const uint32_t vfb = ct + k * N8;
        const uint32_t vfRows = mls::Min(n, (mls::kSpadFloats - vfb) / N8);
        VFill(ct, 0.0f, k * N8);
        FenceV();
        for (uint32_t r = 0; r < k; ++r) {
            for (uint32_t j = 0; j < n; ++j) SWr(ct + r * N8 + j, SRd(aug + j * Wr + N8 + r));
        }
        FenceA();
        Store(wsMem[wsXT], ct, 1, k * N8, N8, N8);
        for (uint32_t j0 = 0; j0 < n; j0 += vfRows) {
            const uint32_t j1 = mls::Min(n, j0 + vfRows);
            FenceA();
            for (uint32_t j = j0; j < j1; ++j) VMulS(vfb + (j - j0) * N8, aug + j * Wr, 1.0f, N8);
            FenceV();
            for (uint32_t j = j0; j < j1; ++j) {
                for (uint32_t i = 0; i < j; ++i) SWr(vfb + (j - j0) * N8 + i, SRd(aug + i * Wr + j));
            }
            FenceA();
            Store(wsMem[wsVF + static_cast<uint64_t>(j0) * M8], vfb, j1 - j0, N8, M8, N8);
        }
        FenceA();

        MLS_TRACE_HOOK(Phase("transpose"));
        TransposeToRows(wsMem, wsAT, M8, aMem, aOff, m, n, n, M8, kFree);
        TransposeToRows(wsMem, wsBT, M8, bMem, bOff, m, k, k, M8, kFree);
        rRowsInVf = true;
        FinishStreaming(b, aOff);
    }

    // Solves, refinement and outputs of the streaming paths, from R in VF (+ diagonal), c in XT, A^T in AT, B^T in BT.
    SIMD_FN inline void FinishStreaming(uint32_t b, uint64_t aOff) {
        // x0 = R^{-1} c, in groups of right-hand sides.
        MLS_TRACE_HOOK(Phase("solve0"));
        const uint32_t avail = mls::kSpadFloats - kFree;
        {
            const uint32_t kg = SolveGroup(3, 1);
            const uint32_t gr = mls::Min(n, (avail - 3 * kg * N8) / N8);
            const uint32_t cv = kFree, xv = cv + kg * N8, tmp = xv + kg * N8, rbuf = tmp + kg * N8;
            for (uint32_t r0 = 0; r0 < k; r0 += kg) {
                const uint32_t cnt = mls::Min(kg, k - r0);
                FenceA();
                Load(cv, wsMem[wsXT + static_cast<uint64_t>(r0) * N8], 1, cnt * N8, N8, N8);
                FenceA();
                BackSolve(cv, xv, tmp, cnt, rbuf, gr);
                Store(wsMem[wsXT + static_cast<uint64_t>(r0) * N8], xv, 1, cnt * N8, N8, N8);
            }
            FenceA();
        }

        MLS_TRACE_HOOK(Phase("refine"));
        RefinePass(aOff);

        // R^T rows (zero left of and on the diagonal) into AT, N8 apart: row i takes VF[j][i] for j > i.
        MLS_TRACE_HOOK(Phase("rtranspose"));
        if (rRowsInVf) {
            // R rows are already in VF.
        } else if (n <= mls::kScalarTransposeRows && 2 * n * N8 <= avail) {
            const uint32_t rc = kFree, rt = rc + n * N8;
            FenceA();
            Load(rc, wsMem[wsVF], n, N8, M8, N8);
            VFill(rt, 0.0f, n * N8);
            FenceA();
            for (uint32_t i = 0; i + 1 < n; ++i) {
                for (uint32_t j = i + 1; j < n; ++j) SWr(rt + i * N8 + j, SRd(rc + j * N8 + i));
            }
            FenceA();
            Store(wsMem[wsAT], rt, 1, n * N8, N8, N8);
            FenceA();
        } else {
            // Batches of wt rows, descending; row i is spread into blocks (i, n) of its slot of 8 * N8 floats. A slot
            // only ever holds rows of decreasing index, so the blocks left of a row's first entry stay zero.
            const uint32_t slot = 8 * N8;
            const uint32_t buf = kFree;
            uint32_t wt = (mls::kSpadFloats - buf) / (2 * slot);
            if (wt > n) wt = n;
            if (wt < 1) wt = 1;
            FenceA();
            VFill(buf, 0.0f, 2 * wt * slot);
            FenceA();
            for (uint32_t hi = n, t = 0; hi > 0; ++t) {
                const uint32_t lo = (hi > wt) ? hi - wt : 0;
                const uint32_t bt = buf + (t & 1) * wt * slot;
                for (uint32_t i = lo; i < hi; ++i) {
                    if (i + 1 < n) {
                        Spread(bt + (i - lo) * slot + 8 * (i + 1), wsMem[wsVF + static_cast<uint64_t>(i + 1) * M8 + i],
                               n - 1 - i, M8);
                    }
                }
                FenceA();
                Compact(wsMem[wsAT + static_cast<uint64_t>(lo) * N8], bt, (hi - lo) * N8);
                hi = lo;
            }
            FenceA();
        }

        // x = x0 + R^{-1} R^{-T} g.
        MLS_TRACE_HOOK(Phase("solve1"));
        {
            const uint32_t kg = SolveGroup(4, 2);
            const uint32_t gr = mls::Min(n, (avail - 4 * kg * N8) / N8);
            const uint32_t gv = kFree, hv = gv + kg * N8, tmp = hv + kg * N8, xv = tmp + kg * N8, rbuf = xv + kg * N8;
            for (uint32_t r0 = 0; r0 < k; r0 += kg) {
                const uint32_t cnt = mls::Min(kg, k - r0);
                FenceA();
                Load(gv, wsMem[wsBT + static_cast<uint64_t>(r0) * M8], cnt, N8, M8, N8);
                FenceA();
                ForwardSolve(gv, hv, tmp, cnt, rbuf, gr);
                BackSolve(hv, gv, tmp, cnt, rbuf, gr);  // dx into gv
                Load(xv, wsMem[wsXT + static_cast<uint64_t>(r0) * N8], 1, cnt * N8, N8, N8);
                FenceA();
                VAdd(xv, xv, gv, cnt * N8);
                FenceA();
                Store(wsMem[wsXT + static_cast<uint64_t>(r0) * N8], xv, 1, cnt * N8, N8, N8);
            }
            FenceA();
        }
        MLS_TRACE_HOOK(Phase("output"));
        WriteOutputs(b);
    }

    // Solution [n][k] from the x rows in XT, and the residuals of the batch.
    SIMD_FN inline void WriteOutputs(uint32_t b) {
        const uint64_t xOff = static_cast<uint64_t>(b) * n * k;
        const uint32_t avail = mls::kSpadFloats - kFree;
        if (k == 1) {
            const uint32_t buf = kFree;
            FenceA();
            Load(buf, wsMem[wsXT], 1, N8, N8, N8);
            FenceA();
            Store(solMem[xOff], buf, 1, n, n, N8);
        } else {
            // Chunks of solution rows: load the matching columns of the x rows, transpose on the scalar unit.
            uint32_t rows = 8;
            while (rows * 2 <= N8 && 2 * k * (rows * 2) + 8 <= avail) rows *= 2;
            const uint32_t xin = kFree;            // k rows of `rows` floats
            const uint32_t xout = xin + k * rows;  // rows * k floats, contiguous
            for (uint32_t c0 = 0; c0 < n; c0 += rows) {
                const uint32_t cr = mls::Min(rows, n - c0);
                FenceA();
                Load(xin, wsMem[wsXT + c0], k, cr, N8, rows);
                FenceA();
                for (uint32_t c = 0; c < cr; ++c) {
                    for (uint32_t r = 0; r < k; ++r) SWr(xout + c * k + r, SRd(xin + r * rows + c));
                }
                FenceA();
                Store(solMem[xOff + static_cast<uint64_t>(c0) * k], xout, 1, cr * k, cr * k, mls::Pad8(cr * k));
            }
        }
        if (has_residuals != 0) {
            FenceA();
            Store(resMem[static_cast<uint64_t>(b) * k], mls::kRes, 1, k, k, mls::Pad8(k));
        }
        FenceA();
    }

    SIMD_FN inline void InitConstants() {
        VFill(0, 0.0f, 192);
        FenceA();
        for (uint32_t q = 0; q < 8; ++q) {
            for (uint32_t l = 0; l < 8; ++l) {
                if (l >= q) SWr(mls::kMaskHi + q * 8 + l, 1.0f);
                if (l < q) SWr(mls::kMaskLo + q * 8 + l, 1.0f);
            }
            SWr(mls::kOneHot + q * 8 + q, 1.0f);
        }
        FenceA();
    }

    uint32_t m = 0, n = 0, k = 0, batch_size = 0, has_residuals = 0, b_has_batch = 0;
    MemRef<float> aMem, bMem, solMem, resMem, wsMem;

    uint32_t M8 = 0, N8 = 0, K8 = 0, Wr = 0, kTau = 0, kFree = 0;
    uint32_t rAug = 0, rArow = 0, rBrow = 0, rQrT = 0, rWrow = 0, rAt = 0, rBt = 0, rRcol = 0, rXa = 0, rX0 = 0, rXb = 0;
    uint32_t rTmp = 0, rRef = 0, rSbT = 0, rSbA = 0, rKg = 0, mChunk = 0;
    bool rRowsInVf = false;
    uint64_t wsAT = 0, wsVF = 0, wsBT = 0, wsXT = 0;
    bool flip = false;
    ScratchArena pipe;
    ScratchBlock<BufferLocation::WORK> spadBlk;
    Scratch<float> sp;
    float scal[mls::kMaxRhsGroup > mls::kMaxBlockCols ? mls::kMaxRhsGroup : mls::kMaxBlockCols];
    float dx[mls::kMaxRhsGroup], dxh[mls::kMaxRhsGroup], dxl[mls::kMaxRhsGroup], dy[mls::kMaxRhsGroup];
    float vsc[mls::kRowQrMaxRows];
};

template <typename DT_A>
SIMD_ENTRY SIMD_FN void matrix_least_squares(MemAddr A, MemAddr B, MemAddr solution, MemAddr residuals, MemAddr workspace, MemAddr tiling) {
    REGISTER_LAYOUT_DEFAULT(MatrixLeastSquaresLayoutData);
    ACQUIRE_LAYOUT_STRUCT(MatrixLeastSquaresLayoutData, layout_data, tiling);
    KernelMatrixLeastSquares<DT_A> op;
    op.Init(A, B, solution, residuals, workspace, layout_data);
    op.Process();
}
