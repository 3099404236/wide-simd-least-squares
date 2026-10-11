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
// Work split: batches go out in tiles whose solution and residual outputs are whole 32 B blocks, so no two cores ever
// write the same block; one core solves every batch of a tile in turn.
//
// Workspace slot of a working core (floats), as sized by host/matrix_least_squares.cpp:
//   AT [n][M8]  A^T, later R^T rows ([n][N8])
//   VF [n][M8]  factored columns: R above the diagonal, the Householder vector from the diagonal down
//   BT [k][M8]  B^T, later A^T r
//   XT [k][N8]  Q^T B (first n rows), later x0
#include "simd_machine.hpp"

#include "matrix_least_squares_layout.h"
#include "layout_key_matrix_least_squares.h"

using namespace hpc;

namespace mls {

constexpr uint32_t kSpadBytes = 195584;
constexpr uint32_t kSpadFloats = kSpadBytes / 4;
constexpr uint32_t kMaxBlockCols = 64;  // columns per QR block (bounded by the scalar scratch below)
constexpr uint32_t kMaxRhsGroup = 64;   // right-hand sides per refinement group
constexpr float kSplit = 4097.0f;       // 2^12 + 1: Veltkamp split of a float into two 12-bit halves

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
    // ------------------------------------------------------------------ data movement

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
        BlockCopyPad<float> pad{true, 0, 0, 0.0f};
        BlockCopyPartial(sp[dst], src, prm, pad);
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
        BlockCopyPartial(dst, sp[src], prm);
    }

    // EXTMEM -> SPAD: `count` floats `stride` floats apart, each into lane 0 of consecutive 32 B blocks.
    SIMD_FN inline void Spread(uint32_t dst, const MemRef<float> &src, uint32_t count, uint64_t stride) {
        BlockCopyParams prm;
        prm.blockCount = static_cast<uint16_t>(count);
        prm.blockLen = 4;
        prm.srcStride = static_cast<uint32_t>((stride - 1) * 4);
        prm.dstStride = 0;
        prm.rsv = 0;
        BlockCopyPartial(sp[dst], src, prm);
    }

    // SPAD -> EXTMEM: lane 0 of `count` consecutive 32 B blocks into `count` contiguous floats.
    SIMD_FN inline void Compact(const MemRef<float> &dst, uint32_t src, uint32_t count) {
        BlockCopyParams prm;
        prm.blockCount = static_cast<uint16_t>(count);
        prm.blockLen = 4;
        prm.srcStride = 0;
        prm.dstStride = 0;
        prm.rsv = 0;
        BlockCopyPartial(dst, sp[src], prm);
    }

    // Transposes `cols` columns of a row-major matrix (`rows` x `ld`, starting at element `src`) into rows of `len`
    // floats (len >= rows, zero tail) at `dst`, `dstStride` floats apart. Uses 2 spread buffers of 8 * len floats.
    SIMD_FN inline void TransposeToRows(const MemRef<float> &dstMem, uint64_t dst, uint64_t dstStride,
                                        const MemRef<float> &srcMem, uint64_t src, uint32_t rows, uint32_t cols,
                                        uint32_t ld, uint32_t len, uint32_t buf) {
        const uint32_t spreadLen = 8 * len;
        Fill(sp[buf], 0.0f, 2 * spreadLen);
        Fence<Unit_All>();
        for (uint32_t c = 0; c <= cols; ++c) {
            if (c < cols) Spread(buf + (c & 1) * spreadLen, srcMem[src + c], rows, ld);
            Fence<Unit_All>();
            if (c < cols) Compact(dstMem[dst + static_cast<uint64_t>(c) * dstStride], buf + (c & 1) * spreadLen, len);
        }
        Fence<Unit_All>();
    }

    // ------------------------------------------------------------------ vector building blocks

    // Reduces rows[q] = buf + q * stride + off (q < count), each `len` floats (multiple of 8), to 8 lane partials at
    // the start of the row. Ends with a SIMD fence.
    SIMD_FN inline void Halve(uint32_t buf, uint32_t stride, uint32_t off, uint32_t count, uint32_t len) {
        while (len > 8) {
            const uint32_t h = ((len / 8 + 1) / 2) * 8;
            for (uint32_t q = 0; q < count; ++q) {
                const uint32_t row = buf + q * stride + off;
                Add(sp[row], sp[row], sp[row + h], len - h);
            }
            Fence<Unit_Simd>();
            len = h;
        }
    }

    SIMD_FN inline float LaneSum(uint32_t off) {
        const float s0 = sp.ScalarRead(off) + sp.ScalarRead(off + 1);
        const float s1 = sp.ScalarRead(off + 2) + sp.ScalarRead(off + 3);
        const float s2 = sp.ScalarRead(off + 4) + sp.ScalarRead(off + 5);
        const float s3 = sp.ScalarRead(off + 6) + sp.ScalarRead(off + 7);
        return (s0 + s1) + (s2 + s3);
    }

    // Applies H = I - tau v v^T (v at `v`, zero outside [a, m)) to columns blk + q * M8, q in [q0, q1), using the
    // product rows prod + q * M8 as scratch.
    SIMD_FN inline void ApplyReflector(uint32_t v, float tau, uint32_t a, uint32_t blk, uint32_t prod, uint32_t q0,
                                       uint32_t q1) {
        const uint32_t len = M8 - a;
        for (uint32_t q = q0; q < q1; ++q) Mul(sp[prod + q * M8 + a], sp[v + a], sp[blk + q * M8 + a], len);
        Fence<Unit_Simd>();
        Halve(prod + q0 * M8, M8, a, q1 - q0, len);
        for (uint32_t q = q0; q < q1; ++q) scal[q] = tau * LaneSum(prod + q * M8 + a);
        for (uint32_t q = q0; q < q1; ++q) MulScalar(sp[prod + q * M8 + a], sp[v + a], scal[q], len);
        Fence<Unit_Simd>();
        for (uint32_t q = q0; q < q1; ++q) Sub(sp[blk + q * M8 + a], sp[blk + q * M8 + a], sp[prod + q * M8 + a], len);
        Fence<Unit_Simd>();
    }

    // ------------------------------------------------------------------ QR

    // Factors column p (block column q): builds its Householder vector in `vcl` (and in place, from the diagonal
    // down), records R[p][p] and tau, and applies the reflector to block columns (q, wb).
    SIMD_FN inline void FactorColumn(uint32_t p, uint32_t q, uint32_t wb, uint32_t blk, uint32_t prod, uint32_t vcl) {
        const uint32_t a = p & ~7u;
        const uint32_t len = M8 - a;
        const uint32_t col = blk + q * M8;
        const uint32_t pr = prod + q * M8;
        Mul(sp[vcl + a], sp[col + a], sp[mls::kMaskHi + (p - a) * 8], 8);
        if (len > 8) MulScalar(sp[vcl + a + 8], sp[col + a + 8], 1.0f, len - 8);
        Fence<Unit_Simd>();
        Mul(sp[pr + a], sp[vcl + a], sp[vcl + a], len);
        Fence<Unit_Simd>();
        Halve(pr, M8, a, 1, len);
        const float nrm = hpc::sqrt(LaneSum(pr + a));
        const float xp = sp.ScalarRead(col + p);
        const float alpha = (xp >= 0.0f) ? -nrm : nrm;
        const float axp = (xp >= 0.0f) ? xp : -xp;
        const float tau = 1.0f / (nrm * (nrm + axp));
        sp.ScalarWrite(mls::kAlpha + p, alpha);
        sp.ScalarWrite(kTau + p, tau);
        // v[p] = x[p] - alpha, in the clean copy and in the stored column.
        MulScalar(sp[mls::kTmp8], sp[mls::kOneHot + (p - a) * 8], alpha, 8);
        Fence<Unit_Simd>();
        Sub(sp[vcl + a], sp[vcl + a], sp[mls::kTmp8], 8);
        Sub(sp[col + a], sp[col + a], sp[mls::kTmp8], 8);
        Fence<Unit_Simd>();
        if (q + 1 < wb) ApplyReflector(vcl, tau, a, blk, prod, q + 1, wb);
    }

    // Left-looking blocked Householder QR of [A | B] (A^T rows in AT, B^T rows in BT). Writes the factored columns to
    // VF and the first N8 entries of Q^T b to XT.
    SIMD_FN inline void FactorAndApply() {
        const uint32_t total = n + k;
        uint32_t g = 4;
        uint32_t avail = mls::kSpadFloats - kFree;
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
            Fence<Unit_All>();

            // Reflectors of earlier blocks, g at a time, double-buffered.
            const uint32_t prev = mls::Min(c0, n);
            if (prev > 0) {
                const uint32_t groups = (prev + g - 1) / g;
                Load(vbuf, wsMem[wsVF], 1, mls::Min(g, prev) * M8, M8, M8);
                for (uint32_t gi = 0; gi < groups; ++gi) {
                    const uint32_t p0 = gi * g;
                    const uint32_t gp = mls::Min(g, prev - p0);
                    const uint32_t cur = vbuf + (gi & 1) * g * M8;
                    Fence<Unit_All>();
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
                        Mul(sp[v + a], sp[v + a], sp[mls::kMaskHi + (p - a) * 8], 8);
                    }
                    Fence<Unit_Simd>();
                    for (uint32_t j = 0; j < gp; ++j) {
                        const uint32_t p = p0 + j;
                        ApplyReflector(cur + j * M8, sp.ScalarRead(kTau + p), p & ~7u, blk, prod, 0, wb);
                    }
                }
            }

            for (uint32_t q = 0; q < na; ++q) FactorColumn(c0 + q, q, wb, blk, prod, vcl);

            Fence<Unit_All>();
            if (na > 0) Store(wsMem[wsVF + static_cast<uint64_t>(c0) * M8], blk, 1, na * M8, na * M8, na * M8);
            if (nb > 0) {
                const uint32_t r0 = c0 + na - n;
                Store(wsMem[wsXT + static_cast<uint64_t>(r0) * N8], blk + na * M8, nb, N8, N8, M8);
            }
            Fence<Unit_All>();
        }
    }

    // ------------------------------------------------------------------ triangular solves

    // Solves R x = c in place of nothing: c rows at `cv` (cnt rows, N8 apart) are consumed, x rows are written to
    // `xv` (N8 apart, zero beyond n). R columns come from VF; `rbuf` holds `gr` of them.
    SIMD_FN inline void BackSolve(uint32_t cv, uint32_t xv, uint32_t tmp, uint32_t cnt, uint32_t rbuf, uint32_t gr) {
        Fill(sp[xv], 0.0f, cnt * N8);
        uint32_t lo = n;  // R columns [lo, hi) are in rbuf
        for (uint32_t jj = n; jj-- > 0;) {
            if (jj < lo) {
                const uint32_t hi = jj + 1;
                lo = (hi > gr) ? hi - gr : 0;
                Fence<Unit_All>();
                Load(rbuf, wsMem[wsVF + static_cast<uint64_t>(lo) * M8], hi - lo, N8, M8, N8);
                Fence<Unit_All>();
                for (uint32_t j = lo; j < hi; ++j) {
                    if ((j & 7u) != 0) {
                        const uint32_t blkOff = rbuf + (j - lo) * N8 + (j & ~7u);
                        Mul(sp[blkOff], sp[blkOff], sp[mls::kMaskLo + (j & 7u) * 8], 8);
                    }
                }
            }
            Fence<Unit_Simd>();
            const float ajj = sp.ScalarRead(mls::kAlpha + jj);
            for (uint32_t r = 0; r < cnt; ++r) {
                scal[r] = sp.ScalarRead(cv + r * N8 + jj) / ajj;
                sp.ScalarWrite(xv + r * N8 + jj, scal[r]);
            }
            if (jj == 0) break;
            const uint32_t len = mls::Pad8(jj);
            const uint32_t rc = rbuf + (jj - lo) * N8;
            for (uint32_t r = 0; r < cnt; ++r) MulScalar(sp[tmp + r * N8], sp[rc], scal[r], len);
            Fence<Unit_Simd>();
            for (uint32_t r = 0; r < cnt; ++r) Sub(sp[cv + r * N8], sp[cv + r * N8], sp[tmp + r * N8], len);
        }
        Fence<Unit_All>();
    }

    // Solves R^T h = g: g rows at `gv` are consumed, h rows written to `hv`. Rows of R (zero left of and on the
    // diagonal) come from the R^T area (AT, rows N8 apart).
    SIMD_FN inline void ForwardSolve(uint32_t gv, uint32_t hv, uint32_t tmp, uint32_t cnt, uint32_t rbuf, uint32_t gr) {
        Fill(sp[hv], 0.0f, cnt * N8);
        uint32_t hi = 0;  // R rows [lo, hi) are in rbuf
        uint32_t lo = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (i >= hi) {
                lo = i;
                hi = mls::Min(n, i + gr);
                Fence<Unit_All>();
                Load(rbuf, wsMem[wsAT + static_cast<uint64_t>(lo) * N8], 1, (hi - lo) * N8, N8, N8);
                Fence<Unit_All>();
            }
            Fence<Unit_Simd>();
            const float aii = sp.ScalarRead(mls::kAlpha + i);
            for (uint32_t r = 0; r < cnt; ++r) {
                scal[r] = sp.ScalarRead(gv + r * N8 + i) / aii;
                sp.ScalarWrite(hv + r * N8 + i, scal[r]);
            }
            if (i + 1 == n) break;
            const uint32_t a = (i + 1) & ~7u;
            const uint32_t len = N8 - a;
            const uint32_t rr = rbuf + (i - lo) * N8 + a;
            for (uint32_t r = 0; r < cnt; ++r) MulScalar(sp[tmp + r * N8 + a], sp[rr], scal[r], len);
            Fence<Unit_Simd>();
            for (uint32_t r = 0; r < cnt; ++r) Sub(sp[gv + r * N8 + a], sp[gv + r * N8 + a], sp[tmp + r * N8 + a], len);
        }
        Fence<Unit_All>();
    }

    // ------------------------------------------------------------------ double-float accumulation

    // For r < cnt: (Sh_r, Sl_r) += av * s_r, where av = ah + al (12-bit halves) and s_r = sh_r + sl_r (scalar halves
    // in scal / scal2): all four partial products are exact, the large one is added with an error-free TwoSum. Buffers of each accumulator, `len` floats:
    // base + r * 7 * len + {0: Sh, 1: T0, 2: T1, 3: T2, 4: P, 5: Q, 6: Sl}; the Sh/T0 roles swap through `flip`.
    SIMD_FN inline void ExtAxpy(uint32_t ah, uint32_t al, uint32_t base, uint32_t cnt, uint32_t len) {
        const uint32_t sh = flip ? 1 : 0;
        const uint32_t t0 = flip ? 0 : 1;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            MulScalar(sp[b + 4 * len], sp[ah], scal[r], len);   // P = ah * sh
            MulScalar(sp[b + 5 * len], sp[ah], scal2[r], len);  // Q = ah * sl
        }
        Fence<Unit_Simd>();
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            Add(sp[b + t0 * len], sp[b + sh * len], sp[b + 4 * len], len);  // s = Sh + P
            Add(sp[b + 6 * len], sp[b + 6 * len], sp[b + 5 * len], len);    // Sl += Q
        }
        Fence<Unit_Simd>();
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            Sub(sp[b + 2 * len], sp[b + t0 * len], sp[b + sh * len], len);  // bb = s - Sh
            MulScalar(sp[b + 5 * len], sp[al], scal[r], len);               // Q = al * sh
        }
        Fence<Unit_Simd>();
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            Sub(sp[b + 3 * len], sp[b + t0 * len], sp[b + 2 * len], len);  // s - bb
            Sub(sp[b + 4 * len], sp[b + 4 * len], sp[b + 2 * len], len);   // P - bb
            Add(sp[b + 6 * len], sp[b + 6 * len], sp[b + 5 * len], len);   // Sl += Q
        }
        Fence<Unit_Simd>();
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            Sub(sp[b + 3 * len], sp[b + sh * len], sp[b + 3 * len], len);  // Sh - (s - bb)
            MulScalar(sp[b + 5 * len], sp[al], scal2[r], len);             // Q = al * sl
        }
        Fence<Unit_Simd>();
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            Add(sp[b + 3 * len], sp[b + 3 * len], sp[b + 4 * len], len);  // e = (Sh - (s - bb)) + (P - bb)
            Add(sp[b + 6 * len], sp[b + 6 * len], sp[b + 5 * len], len);  // Sl += Q
        }
        Fence<Unit_Simd>();
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t b = base + r * 7 * len;
            Add(sp[b + 6 * len], sp[b + 6 * len], sp[b + 3 * len], len);  // Sl += e
        }
        Fence<Unit_Simd>();
        flip = !flip;
    }

    // Splits the `len` floats at `src` into 12-bit halves at `hi` and `lo`.
    SIMD_FN inline void SplitVec(uint32_t src, uint32_t hi, uint32_t lo, uint32_t len) {
        MulScalar(sp[hi], sp[src], mls::kSplit, len);
        Fence<Unit_Simd>();
        Sub(sp[lo], sp[hi], sp[src], len);
        Fence<Unit_Simd>();
        Sub(sp[hi], sp[hi], sp[lo], len);
        Fence<Unit_Simd>();
        Sub(sp[lo], sp[src], sp[hi], len);
        Fence<Unit_Simd>();
    }

    SIMD_FN inline void SplitScalar(float x, float &hi, float &lo) {
        const float t = x * mls::kSplit;
        hi = t - (t - x);
        lo = x - hi;
    }

    // ------------------------------------------------------------------ one matrix

    SIMD_FN inline void SolveOne(uint32_t b) {
        const uint64_t aOff = static_cast<uint64_t>(b) * m * n;
        const uint64_t bOff = (b_has_batch != 0) ? static_cast<uint64_t>(b) * m * k : 0;

        // A^T and B^T into the workspace.
        TransposeToRows(wsMem, wsAT, M8, aMem, aOff, m, n, n, M8, kFree);
        TransposeToRows(wsMem, wsBT, M8, bMem, bOff, m, k, k, M8, kFree);

        FactorAndApply();

        // x0 = R^{-1} c, in groups of right-hand sides.
        const uint32_t avail = mls::kSpadFloats - kFree;
        {
            uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
            while (kg > 1 && 3 * kg * N8 + N8 > avail) --kg;
            const uint32_t gr = mls::Min(n, (avail - 3 * kg * N8) / N8);
            const uint32_t cv = kFree, xv = cv + kg * N8, tmp = xv + kg * N8, rbuf = tmp + kg * N8;
            for (uint32_t r0 = 0; r0 < k; r0 += kg) {
                const uint32_t cnt = mls::Min(kg, k - r0);
                Load(cv, wsMem[wsXT + static_cast<uint64_t>(r0) * N8], 1, cnt * N8, N8, N8);
                Fence<Unit_All>();
                BackSolve(cv, xv, tmp, cnt, rbuf, gr);
                Store(wsMem[wsXT + static_cast<uint64_t>(r0) * N8], xv, 1, cnt * N8, N8, N8);
                Fence<Unit_All>();
            }
        }

        // Refinement: r = B - A x0 and g = A^T r in double-float, per group of right-hand sides.
        {
            const uint32_t rowsA = 8;  // rows of A per chunk in the A^T r pass
            uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
            while (kg > 1 && (7 * kg * M8 + 4 * M8 + kg * N8 > avail ||
                              kg * M8 + 7 * kg * N8 + 3 * rowsA * N8 > avail)) {
                --kg;
            }
            for (uint32_t r0 = 0; r0 < k; r0 += kg) {
                const uint32_t cnt = mls::Min(kg, k - r0);
                // --- r = B - A x0 over M8-long vectors.
                const uint32_t acc = kFree;
                const uint32_t abuf = acc + 7 * cnt * M8;  // two A^T rows
                const uint32_t ahv = abuf + 2 * M8, alv = ahv + M8;
                const uint32_t xbuf = alv + M8;
                Fill(sp[acc], 0.0f, 7 * cnt * M8);
                Fence<Unit_All>();
                for (uint32_t r = 0; r < cnt; ++r) {
                    Load(acc + r * 7 * M8, wsMem[wsBT + static_cast<uint64_t>(r0 + r) * M8], 1, M8, M8, M8);
                }
                Load(xbuf, wsMem[wsXT + static_cast<uint64_t>(r0) * N8], 1, cnt * N8, N8, N8);
                Load(abuf, wsMem[wsAT], 1, M8, M8, M8);
                flip = false;
                for (uint32_t c = 0; c < n; ++c) {
                    const uint32_t cur = abuf + (c & 1) * M8;
                    Fence<Unit_All>();
                    if (c + 1 < n) Load(abuf + ((c + 1) & 1) * M8, wsMem[wsAT + static_cast<uint64_t>(c + 1) * M8], 1, M8, M8, M8);
                    SplitVec(cur, ahv, alv, M8);
                    for (uint32_t r = 0; r < cnt; ++r) {
                        float hi, lo;
                        SplitScalar(-sp.ScalarRead(xbuf + r * N8 + c), hi, lo);
                        scal[r] = hi;
                        scal2[r] = lo;
                    }
                    ExtAxpy(ahv, alv, acc, cnt, M8);
                }
                // r = Sh + Sl, into the slot of P (offset 4).
                const uint32_t sh = flip ? 1 : 0;
                for (uint32_t r = 0; r < cnt; ++r) {
                    const uint32_t bb = acc + r * 7 * M8;
                    Add(sp[bb + 4 * M8], sp[bb + sh * M8], sp[bb + 6 * M8], M8);
                }
                Fence<Unit_Simd>();
                if (has_residuals != 0) {
                    for (uint32_t r = 0; r < cnt; ++r) {
                        const uint32_t bb = acc + r * 7 * M8;
                        Mul(sp[bb + 5 * M8], sp[bb + 4 * M8], sp[bb + 4 * M8], M8);
                    }
                    Fence<Unit_Simd>();
                    Halve(acc + 5 * M8, 7 * M8, 0, cnt, M8);
                    for (uint32_t r = 0; r < cnt; ++r) sp.ScalarWrite(mls::kRes + r0 + r, LaneSum(acc + r * 7 * M8 + 5 * M8));
                }
                // Move the residual vectors to the front: rv[r] (M8 floats each).
                const uint32_t rv = kFree;
                for (uint32_t r = 0; r < cnt; ++r) MulScalar(sp[rv + r * M8], sp[acc + r * 7 * M8 + 4 * M8], 1.0f, M8);
                Fence<Unit_Simd>();

                // --- g = A^T r over N8-long vectors, A rows from the input.
                const uint32_t gacc = rv + cnt * M8;
                const uint32_t arows = gacc + 7 * cnt * N8;
                const uint32_t arh = arows + rowsA * N8, arl = arh + rowsA * N8;
                Fill(sp[gacc], 0.0f, 7 * cnt * N8);
                Fence<Unit_All>();
                flip = false;
                for (uint32_t i0 = 0; i0 < m; i0 += rowsA) {
                    const uint32_t rows = mls::Min(rowsA, m - i0);
                    Fence<Unit_All>();
                    Load(arows, aMem[aOff + static_cast<uint64_t>(i0) * n], rows, n, n, N8);
                    Fence<Unit_All>();
                    SplitVec(arows, arh, arl, rows * N8);
                    for (uint32_t i = 0; i < rows; ++i) {
                        for (uint32_t r = 0; r < cnt; ++r) {
                            float hi, lo;
                            SplitScalar(sp.ScalarRead(rv + r * M8 + i0 + i), hi, lo);
                            scal[r] = hi;
                            scal2[r] = lo;
                        }
                        ExtAxpy(arh + i * N8, arl + i * N8, gacc, cnt, N8);
                    }
                }
                const uint32_t gh = flip ? 1 : 0;
                for (uint32_t r = 0; r < cnt; ++r) {
                    const uint32_t bb = gacc + r * 7 * N8;
                    Add(sp[bb + 4 * N8], sp[bb + gh * N8], sp[bb + 6 * N8], N8);
                }
                Fence<Unit_All>();
                for (uint32_t r = 0; r < cnt; ++r) {
                    Store(wsMem[wsBT + static_cast<uint64_t>(r0 + r) * M8], gacc + r * 7 * N8 + 4 * N8, 1, N8, N8, N8);
                }
                Fence<Unit_All>();
            }
        }

        // R^T rows (zero left of and on the diagonal) into AT, N8 apart: row i takes VF[j][i] for j > i.
        {
            const uint32_t spreadLen = 8 * N8;
            const uint32_t buf = kFree;
            Fill(sp[buf], 0.0f, 2 * spreadLen);
            Fence<Unit_All>();
            // Descending rows: each buffer only ever gains nonzero blocks at or right of the rows it already held.
            for (uint32_t t = 0; t <= n; ++t) {
                if (t < n) {
                    const uint32_t i = n - 1 - t;
                    if (i + 1 < n) {
                        Spread(buf + (t & 1) * spreadLen + 8 * (i + 1), wsMem[wsVF + static_cast<uint64_t>(i + 1) * M8 + i],
                               n - 1 - i, M8);
                    }
                }
                Fence<Unit_All>();
                if (t < n) {
                    const uint32_t i = n - 1 - t;
                    Compact(wsMem[wsAT + static_cast<uint64_t>(i) * N8], buf + (t & 1) * spreadLen, N8);
                }
            }
            Fence<Unit_All>();
        }

        // x = x0 + R^{-1} R^{-T} g, then the outputs.
        {
            uint32_t kg = mls::Min(k, mls::kMaxRhsGroup);
            while (kg > 1 && 4 * kg * N8 + N8 > avail) --kg;
            const uint32_t gr = mls::Min(n, (avail - 4 * kg * N8) / N8);
            const uint32_t gv = kFree, hv = gv + kg * N8, tmp = hv + kg * N8, xv = tmp + kg * N8, rbuf = xv + kg * N8;
            for (uint32_t r0 = 0; r0 < k; r0 += kg) {
                const uint32_t cnt = mls::Min(kg, k - r0);
                for (uint32_t r = 0; r < cnt; ++r) {
                    Load(gv + r * N8, wsMem[wsBT + static_cast<uint64_t>(r0 + r) * M8], 1, N8, N8, N8);
                }
                Fence<Unit_All>();
                ForwardSolve(gv, hv, tmp, cnt, rbuf, gr);
                BackSolve(hv, gv, tmp, cnt, rbuf, gr);  // dx into gv
                Load(xv, wsMem[wsXT + static_cast<uint64_t>(r0) * N8], 1, cnt * N8, N8, N8);
                Fence<Unit_All>();
                Add(sp[xv], sp[xv], sp[gv], cnt * N8);
                Fence<Unit_All>();
                Store(wsMem[wsXT + static_cast<uint64_t>(r0) * N8], xv, 1, cnt * N8, N8, N8);
                Fence<Unit_All>();
            }
        }
        WriteOutputs(b);
    }

    // Solution [n][k] from the x rows in XT, and the residuals of the batch.
    SIMD_FN inline void WriteOutputs(uint32_t b) {
        const uint64_t xOff = static_cast<uint64_t>(b) * n * k;
        const uint32_t avail = mls::kSpadFloats - kFree;
        if (k == 1) {
            const uint32_t buf = kFree;
            Load(buf, wsMem[wsXT], 1, N8, N8, N8);
            Fence<Unit_All>();
            Store(solMem[xOff], buf, 1, n, n, N8);
        } else {
            // Chunks of solution rows: load the matching columns of the x rows, transpose on the scalar unit.
            uint32_t rows = 8;
            while (rows * 2 <= N8 && 2 * k * (rows * 2) + 8 <= avail) rows *= 2;
            const uint32_t xin = kFree;            // k rows of `rows` floats
            const uint32_t xout = xin + k * rows;  // rows * k floats, contiguous
            for (uint32_t c0 = 0; c0 < n; c0 += rows) {
                const uint32_t cr = mls::Min(rows, n - c0);
                Fence<Unit_All>();
                Load(xin, wsMem[wsXT + c0], k, cr, N8, rows);
                Fence<Unit_All>();
                for (uint32_t c = 0; c < cr; ++c) {
                    for (uint32_t r = 0; r < k; ++r) sp.ScalarWrite(xout + c * k + r, sp.ScalarRead(xin + r * rows + c));
                }
                Fence<Unit_All>();
                Store(solMem[xOff + static_cast<uint64_t>(c0) * k], xout, 1, cr * k, cr * k, mls::Pad8(cr * k));
            }
        }
        if (has_residuals != 0) {
            Fence<Unit_All>();
            Store(resMem[static_cast<uint64_t>(b) * k], mls::kRes, 1, k, k, mls::Pad8(k));
        }
        Fence<Unit_All>();
    }

    SIMD_FN inline void InitConstants() {
        Fill(sp[0], 0.0f, 192);
        Fence<Unit_All>();
        for (uint32_t q = 0; q < 8; ++q) {
            for (uint32_t l = 0; l < 8; ++l) {
                if (l >= q) sp.ScalarWrite(mls::kMaskHi + q * 8 + l, 1.0f);
                if (l < q) sp.ScalarWrite(mls::kMaskLo + q * 8 + l, 1.0f);
            }
            sp.ScalarWrite(mls::kOneHot + q * 8 + q, 1.0f);
        }
        Fence<Unit_All>();
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
    float scal2[mls::kMaxRhsGroup];
};

template <typename DT_A>
SIMD_ENTRY SIMD_FN void matrix_least_squares(MemAddr A, MemAddr B, MemAddr solution, MemAddr residuals, MemAddr workspace, MemAddr tiling) {
    REGISTER_LAYOUT_DEFAULT(MatrixLeastSquaresLayoutData);
    ACQUIRE_LAYOUT_STRUCT(MatrixLeastSquaresLayoutData, layout_data, tiling);
    KernelMatrixLeastSquares<DT_A> op;
    op.Init(A, B, solution, residuals, workspace, layout_data);
    op.Process();
}
