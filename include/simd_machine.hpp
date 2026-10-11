// Host-side simulator of the wide-SIMD kernel API, for checking kernels on an ordinary CPU.
//
// It implements the subset of the device API that the kernels in kernels/ use, with the same arithmetic (IEEE float32,
// no FMA contraction) and the documented machine limits:
//  - each core owns a 195584-byte SPAD; ScratchArena::ReserveBlock places blocks on 32 B boundaries and stops the run
//    when the SPAD is exhausted;
//  - BlockCopyPartial needs a 32 B aligned SPAD address; vector instructions need 32 B aligned operands;
//  - SPAD contents start as NaN garbage, so reads of data that was never written show up in the results.
// Cores run one after another (core 0 .. CoreCount()-1) on shared EXTMEM buffers. Two different cores writing the same
// 32 B block of EXTMEM is reported, because the device writes whole blocks and such writes race.
// Every scalar access, vector instruction, fence and block copy is counted, for a rough cost model.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define SIMD_FN
#define SIMD_ENTRY
#define SIMD_MEM
#define REGISTER_LAYOUT_DEFAULT(T)
#define ACQUIRE_LAYOUT_STRUCT(T, name, ptr) \
    T name;                                  \
    std::memcpy(&name, reinterpret_cast<const void *>(ptr), sizeof(T))

using MemAddr = uint8_t *;

namespace simemu {

constexpr size_t kSpadBytes = 195584;
constexpr size_t kVecBytes = 256;
// Limits the verified kernels respect on the device, enforced here as faults: vector instructions bounded by SPAD size
// (matrix_linear_solve: "单次向量 mask 安全上限"), no single block copy over 32 KB.
constexpr size_t kVecMaxElems = 65536;
constexpr size_t kDmaMaxBytes = 196608;

struct Counters {
    uint64_t scalar_rd = 0;
    uint64_t scalar_wr = 0;
    uint64_t vec_calls = 0;
    uint64_t vec_repeats = 0;
    uint64_t dma_calls = 0;
    uint64_t dma_bytes = 0;
    uint64_t fence_simd = 0;
    uint64_t fence_all = 0;
};

struct Faults {
    uint64_t spad_misalign = 0;   // block copy with a SPAD address off a 32 B boundary
    uint64_t gm_misalign = 0;     // block copy with an EXTMEM address off a 32 B boundary (allowed, counted)
    uint64_t vec_misalign = 0;    // vector operand off a 32 B boundary
    uint64_t spad_oob = 0;        // access outside the reserved block
    uint64_t gm_oob = 0;          // block copy outside a registered EXTMEM buffer
    uint64_t write_conflict = 0;  // two cores wrote the same 32 B block of EXTMEM
    uint64_t spad_overflow = 0;   // ReserveBlock past the SPAD capacity
    uint64_t vec_long = 0;        // vector instruction over kVecMaxElems elements
    uint64_t dma_large = 0;       // block copy over kDmaMaxBytes
    uint64_t total() const {
        return spad_misalign + vec_misalign + spad_oob + gm_oob + write_conflict + spad_overflow + vec_long + dma_large;
    }
};

struct Region {
    uint8_t *base;
    size_t bytes;  // logical size of the tensor
    size_t cap;    // allocated size (the logical size rounded up to 512 B, as device allocations are)
    std::string name;
    std::vector<int16_t> owner;  // core that wrote each 32 B block, -1 if none
};

inline Counters g_cnt;
inline Faults g_faults;
inline int g_core_id = 0;
inline int g_core_count = 1;
inline std::vector<uint8_t> g_spad(kSpadBytes);
inline size_t g_spad_used = 0;
inline size_t g_spad_peak = 0;
inline std::vector<Region> g_regions;
inline int g_reports = 0;

inline void Report(const char *kind, const char *detail) {
    if (g_reports < 20) std::fprintf(stderr, "[simemu core %d] %s: %s\n", g_core_id, kind, detail);
    ++g_reports;
}

// Allocates an EXTMEM tensor of `bytes` bytes (512 B aligned and padded, filled with NaN garbage) and registers it.
inline uint8_t *AllocRegion(size_t bytes, const char *name) {
    const size_t cap = (bytes + 511) / 512 * 512 + 512;
    uint8_t *p = static_cast<uint8_t *>(std::aligned_alloc(512, cap));
    std::memset(p, 0xFF, cap);
    Region r;
    r.base = p;
    r.bytes = bytes;
    r.cap = cap;
    r.name = name;
    r.owner.assign(cap / 32, -1);
    g_regions.push_back(r);
    return p;
}

inline void FreeRegions() {
    for (auto &r : g_regions) std::free(r.base);
    g_regions.clear();
}

inline Region *FindRegion(const uint8_t *p) {
    for (auto &r : g_regions) {
        if (p >= r.base && p < r.base + r.cap) return &r;
    }
    return nullptr;
}

// Starts the run of one core: empty SPAD filled with NaN garbage.
inline void BeginCore(int id, int count) {
    g_core_id = id;
    g_core_count = count;
    std::memset(g_spad.data(), 0xFF, g_spad.size());
    g_spad_used = 0;
}

inline size_t SpadOffset(const void *p) {
    return static_cast<size_t>(static_cast<const uint8_t *>(p) - g_spad.data());
}

inline bool InSpad(const void *p) {
    const uint8_t *q = static_cast<const uint8_t *>(p);
    return q >= g_spad.data() && q <= g_spad.data() + g_spad.size();
}

}  // namespace simemu

namespace hpc {

using f16 = _Float16;

enum class BufferLocation { WORK, VECIN, VECOUT, VECCALC };
enum class Rounding { ROUND_NONE, ROUND_NEAREST, ROUND_FLOOR, ROUND_CEIL, ROUND_TRUNC };
enum FenceUnit { Unit_Simd, Unit_All, Unit_Dma };

template <FenceUnit U>
inline void Fence() {
    if (U == Unit_Simd) {
        ++simemu::g_cnt.fence_simd;
    } else {
        ++simemu::g_cnt.fence_all;
    }
}

inline uint32_t CoreId() { return static_cast<uint32_t>(simemu::g_core_id); }
inline uint32_t CoreCount() { return static_cast<uint32_t>(simemu::g_core_count); }

// Scalar square root of the scalar unit (IEEE float32).
inline float sqrt(float x) { return std::sqrt(x); }

template <typename T>
class Scratch {
public:
    T *p = nullptr;
    T *lo = nullptr;
    T *hi = nullptr;

    Scratch() = default;
    Scratch(T *p_, T *lo_, T *hi_) : p(p_), lo(lo_), hi(hi_) {}

    bool InBlock(uint64_t i, uint64_t count) const { return p + i >= lo && p + i + count <= hi; }

    T ScalarRead(uint32_t i) const {
        ++simemu::g_cnt.scalar_rd;
        if (!InBlock(i, 1)) {
            ++simemu::g_faults.spad_oob;
            simemu::Report("spad_oob", "ScalarRead outside the block");
            if (!simemu::InSpad(p + i)) std::abort();
        }
        return p[i];
    }

    void ScalarWrite(uint32_t i, T v) const {
        ++simemu::g_cnt.scalar_wr;
        if (!InBlock(i, 1)) {
            ++simemu::g_faults.spad_oob;
            simemu::Report("spad_oob", "ScalarWrite outside the block");
            if (!simemu::InSpad(p + i)) std::abort();
        }
        p[i] = v;
    }

    Scratch operator[](uint32_t off) const { return Scratch(p + off, lo, hi); }
};

template <BufferLocation L>
class ScratchBlock {
public:
    uint8_t *base = nullptr;
    size_t bytes = 0;

    template <typename T>
    Scratch<T> Get() const {
        return Scratch<T>(reinterpret_cast<T *>(base), reinterpret_cast<T *>(base),
                          reinterpret_cast<T *>(base + bytes));
    }
};

class ScratchArena {
public:
    template <BufferLocation L>
    void ReserveBlock(ScratchBlock<L> &blk, size_t bytes) {
        const size_t off = (simemu::g_spad_used + 31) / 32 * 32;
        const size_t sz = (bytes + 31) / 32 * 32;
        if (off + sz > simemu::kSpadBytes) {
            ++simemu::g_faults.spad_overflow;
            char msg[128];
            std::snprintf(msg, sizeof(msg), "ReserveBlock of %zu bytes at offset %zu exceeds the SPAD", bytes, off);
            simemu::Report("spad_overflow", msg);
            std::fprintf(stderr, "fatal: SPAD overflow\n");
            std::exit(3);
        }
        blk.base = simemu::g_spad.data() + off;
        blk.bytes = sz;
        simemu::g_spad_used = off + sz;
        if (simemu::g_spad_used > simemu::g_spad_peak) simemu::g_spad_peak = simemu::g_spad_used;
    }
};

template <typename T>
class MemRef {
public:
    T *base = nullptr;
    uint64_t off = 0;

    void Bind(T *p) {
        base = p;
        off = 0;
    }
    MemRef operator[](uint64_t i) const {
        MemRef r;
        r.base = base;
        r.off = off + i;
        return r;
    }
    T *Ptr() const { return base + off; }
};

struct BlockCopyParams {
    uint16_t blockCount;
    uint32_t blockLen;   // bytes
    uint32_t srcStride;  // gap between blocks
    uint32_t dstStride;
    uint32_t rsv;
};

template <typename T>
struct BlockCopyPad {
    bool isPad;
    uint8_t leftPadding;
    uint8_t rightPadding;
    T paddingValue;
};

namespace detail {

template <typename T>
inline void CheckSpadOperand(const Scratch<T> &s, uint64_t count, bool dma) {
    if (simemu::SpadOffset(s.p) % 32 != 0) {
        if (dma) {
            ++simemu::g_faults.spad_misalign;
            simemu::Report("spad_misalign", "block copy with a SPAD address off a 32 B boundary");
        } else {
            ++simemu::g_faults.vec_misalign;
            simemu::Report("vec_misalign", "vector operand off a 32 B boundary");
        }
    }
    if (!s.InBlock(0, count)) {
        ++simemu::g_faults.spad_oob;
        simemu::Report("spad_oob", dma ? "block copy outside the SPAD block" : "vector op outside the SPAD block");
        if (!simemu::InSpad(s.p) || !simemu::InSpad(s.p + count)) std::abort();
    }
}

inline void CheckGm(const uint8_t *p, uint64_t bytes, bool write) {
    if (bytes > simemu::kDmaMaxBytes) {
        ++simemu::g_faults.dma_large;
        simemu::Report("dma_large", "block copy over 32 KB");
    }
    simemu::Region *r = simemu::FindRegion(p);
    if (r == nullptr || p + bytes > r->base + r->cap) {
        ++simemu::g_faults.gm_oob;
        simemu::Report("gm_oob", "block copy outside a registered EXTMEM buffer");
        std::abort();
    }
    if (p + bytes > r->base + r->bytes) {
        // Past the logical end, inside the allocation padding: harmless for a read, corrupts a neighbour for a write.
        if (write) {
            ++simemu::g_faults.gm_oob;
            simemu::Report("gm_oob", "block copy writes past the end of an EXTMEM tensor");
        }
    }
    const uint64_t off = static_cast<uint64_t>(p - r->base);
    if (off % 32 != 0) ++simemu::g_faults.gm_misalign;
    if (write && bytes > 0) {
        for (uint64_t blk = off / 32; blk <= (off + bytes - 1) / 32; ++blk) {
            int16_t &o = r->owner[blk];
            if (o >= 0 && o != simemu::g_core_id) {
                ++simemu::g_faults.write_conflict;
                char msg[160];
                std::snprintf(msg, sizeof(msg), "block %llu of %s written by cores %d and %d",
                              static_cast<unsigned long long>(blk), r->name.c_str(), o, simemu::g_core_id);
                simemu::Report("write_conflict", msg);
            }
            o = static_cast<int16_t>(simemu::g_core_id);
        }
    }
}

template <typename T>
inline void CountVec(uint64_t count) {
    if (count > simemu::kVecMaxElems) {
        ++simemu::g_faults.vec_long;
        simemu::Report("vec_long", "vector instruction over limit");
    }
    ++simemu::g_cnt.vec_calls;
    simemu::g_cnt.vec_repeats += (count * sizeof(T) + simemu::kVecBytes - 1) / simemu::kVecBytes;
}

}  // namespace detail

// EXTMEM -> SPAD.
template <typename T>
inline void BlockCopyPartial(const Scratch<T> &dst, const MemRef<T> &src, const BlockCopyParams &prm,
                             const BlockCopyPad<T> &pad) {
    const uint64_t len = prm.blockLen;
    const uint64_t dst_blk = (len + 31) / 32 * 32;
    const uint64_t total_dst = prm.blockCount * dst_blk + (prm.blockCount ? (prm.blockCount - 1) : 0) * prm.dstStride * 32;
    detail::CheckSpadOperand(dst, total_dst / sizeof(T), true);
    ++simemu::g_cnt.dma_calls;
    simemu::g_cnt.dma_bytes += prm.blockCount * len;
    const uint8_t *s = reinterpret_cast<const uint8_t *>(src.Ptr());
    uint8_t *d = reinterpret_cast<uint8_t *>(dst.p);
    for (uint32_t b = 0; b < prm.blockCount; ++b) {
        const uint8_t *sb = s + b * (len + prm.srcStride);
        uint8_t *db = d + b * (dst_blk + prm.dstStride * 32);
        detail::CheckGm(sb, len, false);
        std::memcpy(db, sb, len);
        T *tail = reinterpret_cast<T *>(db + len);
        const uint64_t tail_n = (dst_blk - len) / sizeof(T);
        for (uint64_t i = 0; i < tail_n; ++i) {
            if (pad.isPad) {
                tail[i] = pad.paddingValue;
            } else {
                std::memset(&tail[i], 0xFF, sizeof(T));
            }
        }
    }
}

template <typename T>
inline void BlockCopyPartial(const Scratch<T> &dst, const MemRef<T> &src, const BlockCopyParams &prm) {
    BlockCopyPad<T> pad{false, 0, 0, T(0)};
    BlockCopyPartial(dst, src, prm, pad);
}

// SPAD -> EXTMEM: exactly blockLen bytes per block.
template <typename T>
inline void BlockCopyPartial(const MemRef<T> &dst, const Scratch<T> &src, const BlockCopyParams &prm) {
    const uint64_t len = prm.blockLen;
    const uint64_t src_blk = (len + 31) / 32 * 32;
    const uint64_t total_src = prm.blockCount * src_blk + (prm.blockCount ? (prm.blockCount - 1) : 0) * prm.srcStride * 32;
    detail::CheckSpadOperand(src, total_src / sizeof(T), true);
    ++simemu::g_cnt.dma_calls;
    simemu::g_cnt.dma_bytes += prm.blockCount * len;
    uint8_t *d = reinterpret_cast<uint8_t *>(dst.Ptr());
    const uint8_t *s = reinterpret_cast<const uint8_t *>(src.p);
    for (uint32_t b = 0; b < prm.blockCount; ++b) {
        uint8_t *db = d + b * (len + prm.dstStride);
        const uint8_t *sb = s + b * (src_blk + prm.srcStride * 32);
        detail::CheckGm(db, len, true);
        std::memcpy(db, sb, len);
    }
}

// Whole-block copies (count elements, 32 B multiples).
template <typename T>
inline void BlockCopy(const Scratch<T> &dst, const MemRef<T> &src, uint32_t count) {
    BlockCopyParams prm{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    BlockCopyPartial(dst, src, prm);
}

template <typename T>
inline void BlockCopy(const MemRef<T> &dst, const Scratch<T> &src, uint32_t count) {
    BlockCopyParams prm{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    BlockCopyPartial(dst, src, prm);
}

// ----- vector instructions (count elements, 256 B per repeat) -----

template <typename T, typename S>
inline void Fill(const Scratch<T> &dst, S value, uint32_t count) {
    detail::CheckSpadOperand(dst, count, false);
    detail::CountVec<T>(count);
    const T v = static_cast<T>(value);
    for (uint32_t i = 0; i < count; ++i) dst.p[i] = v;
}

#define SIMEMU_BINARY(NAME, EXPR)                                                                        \
    template <typename T>                                                                                \
    inline void NAME(const Scratch<T> &dst, const Scratch<T> &a, const Scratch<T> &b, uint32_t count) { \
        detail::CheckSpadOperand(dst, count, false);                                                     \
        detail::CheckSpadOperand(a, count, false);                                                       \
        detail::CheckSpadOperand(b, count, false);                                                       \
        detail::CountVec<T>(count);                                                                      \
        for (uint32_t i = 0; i < count; ++i) {                                                           \
            const T x = a.p[i];                                                                          \
            const T y = b.p[i];                                                                          \
            dst.p[i] = (EXPR);                                                                           \
        }                                                                                                \
    }

SIMEMU_BINARY(Add, x + y)
SIMEMU_BINARY(Sub, x - y)
SIMEMU_BINARY(Mul, x * y)
SIMEMU_BINARY(Div, x / y)
SIMEMU_BINARY(Max, (x > y) ? x : y)
SIMEMU_BINARY(Min, (x < y) ? x : y)
#undef SIMEMU_BINARY

#define SIMEMU_SCALAR(NAME, EXPR)                                                                    \
    template <typename T, typename S>                                                                \
    inline void NAME(const Scratch<T> &dst, const Scratch<T> &a, S scalar, uint32_t count) {        \
        detail::CheckSpadOperand(dst, count, false);                                                 \
        detail::CheckSpadOperand(a, count, false);                                                   \
        detail::CountVec<T>(count);                                                                  \
        const T y = static_cast<T>(scalar);                                                          \
        for (uint32_t i = 0; i < count; ++i) {                                                       \
            const T x = a.p[i];                                                                      \
            dst.p[i] = (EXPR);                                                                       \
        }                                                                                            \
    }

SIMEMU_SCALAR(MulScalar, x * y)
SIMEMU_SCALAR(AddScalar, x + y)
SIMEMU_SCALAR(MaxScalar, (x > y) ? x : y)
SIMEMU_SCALAR(MinScalar, (x < y) ? x : y)
#undef SIMEMU_SCALAR

#define SIMEMU_UNARY(NAME, EXPR)                                                    \
    template <typename T>                                                           \
    inline void NAME(const Scratch<T> &dst, const Scratch<T> &a, uint32_t count) { \
        detail::CheckSpadOperand(dst, count, false);                                \
        detail::CheckSpadOperand(a, count, false);                                  \
        detail::CountVec<T>(count);                                                 \
        for (uint32_t i = 0; i < count; ++i) {                                      \
            const T x = a.p[i];                                                     \
            dst.p[i] = (EXPR);                                                      \
        }                                                                           \
    }

SIMEMU_UNARY(Sqrt, static_cast<T>(std::sqrt(x)))
SIMEMU_UNARY(Abs, (x < T(0)) ? -x : x)
SIMEMU_UNARY(Exp, static_cast<T>(std::exp(x)))
#undef SIMEMU_UNARY

template <typename D, typename S>
inline void Convert(const Scratch<D> &dst, const Scratch<S> &src, Rounding, uint32_t count) {
    detail::CheckSpadOperand(dst, count, false);
    detail::CheckSpadOperand(src, count, false);
    detail::CountVec<float>(count);
    for (uint32_t i = 0; i < count; ++i) dst.p[i] = static_cast<D>(src.p[i]);
}

}  // namespace hpc
