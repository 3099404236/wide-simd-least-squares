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
// Transposes of at most this many rows go through the scalar unit (2 accesses of a few cycles per element) rather than
// 4-byte block copies (two copies and a global fence per column, several hundred cycles).
constexpr uint32_t kScalarTransposeRows = 128;
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

        for (uint32_t t = core; t < tiles; t += cores) {
            const uint32_t b1 = mls::Min((t + 1) * unit, batch_size);
            for (uint32_t b = t * unit; b < b1; ++b) SolveOne(b);
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
    // unit (2 rows * cols accesses and two block copies); long ones through EXTMEM, one column at a time, with two
    // 4-byte-block copies each (2 spread buffers of 8 * len floats at `buf`).
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
        const uint32_t spreadLen = 8 * len;
        FenceA();
        VFill(buf, 0.0f, 2 * spreadLen);
        FenceA();
        for (uint32_t c = 0; c <= cols; ++c) {
            if (c < cols) Spread(buf + (c & 1) * spreadLen, srcMem[src + c], rows, ld);
            FenceA();
            if (c < cols) Compact(dstMem[dst + static_cast<uint64_t>(c) * dstStride], buf + (c & 1) * spreadLen, len);
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
    // product rows prod + q * M8 as scratch.
    SIMD_FN inline void ApplyReflector(uint32_t v, float tau, uint32_t a, uint32_t blk, uint32_t prod, uint32_t q0,
                                       uint32_t q1) {
        const uint32_t len = M8 - a;
        for (uint32_t q = q0; q < q1; ++q) VMul(prod + q * M8 + a, v + a, blk + q * M8 + a, len);
        FenceV();
        Halve(prod + q0 * M8, M8, a, q1 - q0, len);
        for (uint32_t q = q0; q < q1; ++q) scal[q] = tau * LaneSum(prod + q * M8 + a);
        for (uint32_t q = q0; q < q1; ++q) VMulS(prod + q * M8 + a, v + a, scal[q], len);
        FenceV();
        for (uint32_t q = q0; q < q1; ++q) VSub(blk + q * M8 + a, blk + q * M8 + a, prod + q * M8 + a, len);
        FenceV();
    }

    // ------------------------------------------------------------------ QR

    // Factors column p (block column q): builds its Householder vector in `vcl` (and in place, from the diagonal
    // down), records R[p][p] and tau, and applies the reflector to block columns (q, wb).
    SIMD_FN inline void FactorColumn(uint32_t p, uint32_t q, uint32_t wb, uint32_t blk, uint32_t prod, uint32_t vcl) {
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
        if (q + 1 < wb) ApplyReflector(vcl, tau, a, blk, prod, q + 1, wb);
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
            w = (avail > fixed) ? (avail - fixed) / (2 * M8) : 0;
            if (w >= 2 || g == 1) break;
            --g;
        }
        w = mls::Min(w, mls::kMaxBlockCols);
        w = mls::Min(w, total);
        const uint32_t vcl = kFree;
        const uint32_t vbuf = vcl + M8;
        const uint32_t blk = vbuf + 2 * g * M8;
        const uint32_t prod = blk + w * M8;

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
                        ApplyReflector(cur + j * M8, SRd(kTau + p), p & ~7u, blk, prod, 0, wb);
                    }
                }
            }

            for (uint32_t q = 0; q < na; ++q) FactorColumn(c0 + q, q, wb, blk, prod, vcl);

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

    // Solves R^T h = g: g rows at `gv` are consumed, h rows written to `hv`. Rows of R (zero left of and on the
    // diagonal) come from the R^T area (AT, rows N8 apart).
    SIMD_FN inline void ForwardSolve(uint32_t gv, uint32_t hv, uint32_t tmp, uint32_t cnt, uint32_t rbuf, uint32_t gr) {
        VFill(hv, 0.0f, cnt * N8);
        uint32_t hi = 0;  // R rows [lo, hi) are in rbuf
        uint32_t lo = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (i >= hi) {
                lo = i;
                hi = mls::Min(n, i + gr);
                FenceA();
                Load(rbuf, wsMem[wsAT + static_cast<uint64_t>(lo) * N8], 1, (hi - lo) * N8, N8, N8);
                FenceA();
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
    // SPAD of a group of c right-hand sides: rh and rl rows (2c x M8), then either the residual accumulators (7 fields
    // of c x M8), two blocks of ra A^T rows, their 12-bit halves and the x0 rows (c x N8), or the gradient accumulators
    // (8 fields of c x N8), two blocks of rc A rows and their 12-bit halves.
    SIMD_FN inline void RefinePass(uint64_t aOff) {
        const uint32_t avail = mls::kSpadFloats - kFree;
        uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
        uint32_t ra = 1, rc = 1;
        for (; kg >= 1; --kg) {
            const uint32_t needR = 9 * kg * M8 + kg * N8;
            const uint32_t needG = 2 * kg * M8 + 8 * kg * N8;
            if (needR + 4 * M8 > avail || needG + 4 * N8 > avail) continue;
            ra = mls::Min(mls::Min(n, 16), (avail - needR) / (4 * M8));
            rc = mls::Min(mls::Min(m, 64), (avail - needG) / (4 * N8));
            if (kg == 1 || (ra >= mls::Min(n, 2) && rc >= mls::Min(m, 4))) break;
        }
        for (uint32_t r0 = 0; r0 < k; r0 += kg) {
            const uint32_t cnt = mls::Min(kg, k - r0);
            const uint32_t rArr = kFree;  // rh[r] at rArr + r * M8, rl[r] at rArr + (cnt + r) * M8
            const uint32_t acc = rArr + 2 * cnt * M8;

            // --- r = B - A x0.
            {
                const uint32_t raw = acc + 7 * cnt * M8;  // two blocks of ra A^T rows
                const uint32_t ahv = raw + 2 * ra * M8;
                const uint32_t alv = ahv + ra * M8;
                const uint32_t xb = alv + ra * M8;
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

            // --- g = A^T (rh + rl).
            {
                const uint32_t raw = acc + 8 * cnt * N8;  // two blocks of rc A rows
                const uint32_t ahv = raw + 2 * rc * N8;
                const uint32_t alv = ahv + rc * N8;
                VFill(acc, 0.0f, 8 * cnt * N8);
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
                        Dot2(cur + i * N8, ahv + i * N8, alv + i * N8, acc, cnt, N8, true);
                    }
                }
                const uint32_t f = cnt * N8;
                VAdd(acc + 3 * f, acc + (flip ? f : 0), acc + 2 * f, f);  // g = Sh + Sl into the P field
                FenceA();
                Store(wsMem[wsBT + static_cast<uint64_t>(r0) * M8], acc + 3 * f, cnt, N8, M8, N8);
                FenceA();
            }
        }
    }

    // ------------------------------------------------------------------ one matrix

    SIMD_FN inline void SolveOne(uint32_t b) {
        const uint64_t aOff = static_cast<uint64_t>(b) * m * n;
        const uint64_t bOff = (b_has_batch != 0) ? static_cast<uint64_t>(b) * m * k : 0;

        MLS_TRACE_HOOK(Phase("transpose"));
        TransposeToRows(wsMem, wsAT, M8, aMem, aOff, m, n, n, M8, kFree);
        TransposeToRows(wsMem, wsBT, M8, bMem, bOff, m, k, k, M8, kFree);

        MLS_TRACE_HOOK(Phase("qr"));
        FactorAndApply();

        // x0 = R^{-1} c, in groups of right-hand sides.
        MLS_TRACE_HOOK(Phase("solve0"));
        const uint32_t avail = mls::kSpadFloats - kFree;
        {
            uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
            while (kg > 1 && 3 * kg * N8 + N8 > avail) --kg;
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
        if (n <= mls::kScalarTransposeRows && 2 * n * N8 <= avail) {
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
            const uint32_t spreadLen = 8 * N8;
            const uint32_t buf = kFree;
            FenceA();
            VFill(buf, 0.0f, 2 * spreadLen);
            FenceA();
            // Descending rows: each buffer only ever gains nonzero blocks at or right of the rows it already held.
            for (uint32_t t = 0; t <= n; ++t) {
                if (t < n) {
                    const uint32_t i = n - 1 - t;
                    if (i + 1 < n) {
                        Spread(buf + (t & 1) * spreadLen + 8 * (i + 1), wsMem[wsVF + static_cast<uint64_t>(i + 1) * M8 + i],
                               n - 1 - i, M8);
                    }
                }
                FenceA();
                if (t < n) {
                    const uint32_t i = n - 1 - t;
                    Compact(wsMem[wsAT + static_cast<uint64_t>(i) * N8], buf + (t & 1) * spreadLen, N8);
                }
            }
            FenceA();
        }

        // x = x0 + R^{-1} R^{-T} g.
        MLS_TRACE_HOOK(Phase("solve1"));
        {
            uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
            while (kg > 1 && 4 * kg * N8 + N8 > avail) --kg;
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

    uint32_t M8 = 0, N8 = 0, kTau = 0, kFree = 0;
    uint64_t wsAT = 0, wsVF = 0, wsBT = 0, wsXT = 0;
    bool flip = false;
    ScratchArena pipe;
    ScratchBlock<BufferLocation::WORK> spadBlk;
    Scratch<float> sp;
    float scal[mls::kMaxRhsGroup > mls::kMaxBlockCols ? mls::kMaxRhsGroup : mls::kMaxBlockCols];
    float dx[mls::kMaxRhsGroup], dxh[mls::kMaxRhsGroup], dxl[mls::kMaxRhsGroup], dy[mls::kMaxRhsGroup];
};

template <typename DT_A>
SIMD_ENTRY SIMD_FN void matrix_least_squares(MemAddr A, MemAddr B, MemAddr solution, MemAddr residuals, MemAddr workspace, MemAddr tiling) {
    REGISTER_LAYOUT_DEFAULT(MatrixLeastSquaresLayoutData);
    ACQUIRE_LAYOUT_STRUCT(MatrixLeastSquaresLayoutData, layout_data, tiling);
    KernelMatrixLeastSquares<DT_A> op;
    op.Init(A, B, solution, residuals, workspace, layout_data);
    op.Process();
}
