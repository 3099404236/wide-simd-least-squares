// Fence-hazard checker and phase profiler for kernels built with MLS_TRACE (see kernel/matrix_least_squares.cpp).
//
// The simulator executes every instruction to completion in program order, so a missing fence never changes its
// results. On the device the units run concurrently: vector instructions are pipelined, the two block-copy queues
// (EXTMEM -> SPAD and SPAD -> EXTMEM) and the scalar unit run asynchronously. This checker replays the kernel's
// accesses against that model and reports every access that could observe a stale or not-yet-written value:
//   - an access is complete once a fence covering its unit follows it: Fence<Unit_Simd> covers vector instructions,
//     Fence<Unit_All> covers every unit;
//   - copies of one queue are ordered among themselves, and the scalar unit is ordered with itself;
//   - everything else (vector after vector, copy after vector or scalar, vector or scalar after copy, vector after a
//     scalar write, opposite copy queues on EXTMEM) must be separated by a fence that completes the earlier access.
// Read-after-write, write-after-write and write-after-read on SPAD floats and EXTMEM floats are checked, and so are
// vector instructions whose destination partially overlaps a source.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "simd_machine.hpp"

namespace tracechk {

enum Unit : int8_t { kV = 0, kDI = 1, kDO = 2, kS = 3 };
const char *const kUnitName[4] = {"vector", "copy-in", "copy-out", "scalar"};

struct Cell {
    int8_t wUnit = -1;
    uint32_t wGen = 0;
    uint32_t rGen[4] = {0, 0, 0, 0};  // 0: no pending read by that unit
};

struct ExtMemShadow {
    const uint8_t *base = nullptr;
    size_t floats = 0;
    std::vector<uint32_t> wGen;  // pending SPAD -> EXTMEM write
    std::vector<uint32_t> rGen;  // pending EXTMEM -> SPAD read
};

struct State {
    uint32_t gen[4] = {1, 1, 1, 1};
    std::vector<Cell> spad;
    std::vector<ExtMemShadow> ext_regions;
    uint64_t hazards = 0;
    int reports = 0;
    std::string phase;
    std::map<std::string, simemu::Counters> phaseCost;
    simemu::Counters phaseStart;
};

inline State &St() {
    static State s;
    return s;
}

inline bool Complete(int u, uint32_t g) { return St().gen[u] > g; }

// Same-queue accesses are ordered without a fence, except vector instructions (pipelined).
inline bool Ordered(int earlier, int later) { return earlier == later && earlier != kV; }

inline void Report(const char *what, int earlier, int later, const char *where, size_t idx) {
    State &s = St();
    ++s.hazards;
    if (s.reports < 12) {
        std::fprintf(stderr, "[hazard] %s: %s access then %s access of %s[%zu] without a fence (phase %s, core %d)\n",
                     what, kUnitName[earlier], kUnitName[later], where, idx, s.phase.c_str(), simemu::g_core_id);
    }
    ++s.reports;
}

inline void AccumulatePhase() {
    State &s = St();
    if (s.phase.empty()) return;
    const simemu::Counters &now = simemu::g_cnt;
    simemu::Counters &acc = s.phaseCost[s.phase];
    acc.scalar_rd += now.scalar_rd - s.phaseStart.scalar_rd;
    acc.scalar_wr += now.scalar_wr - s.phaseStart.scalar_wr;
    acc.vec_calls += now.vec_calls - s.phaseStart.vec_calls;
    acc.vec_repeats += now.vec_repeats - s.phaseStart.vec_repeats;
    acc.dma_calls += now.dma_calls - s.phaseStart.dma_calls;
    acc.dma_bytes += now.dma_bytes - s.phaseStart.dma_bytes;
    acc.fence_simd += now.fence_simd - s.phaseStart.fence_simd;
    acc.fence_all += now.fence_all - s.phaseStart.fence_all;
}

// Resets the per-core state (call before each core runs).
inline void BeginCore() {
    State &s = St();
    for (auto &g : s.gen) g = 1;
    s.spad.assign(simemu::kSpadBytes / 4, Cell());
    s.ext_regions.clear();
    for (const auto &r : simemu::g_regions) {
        ExtMemShadow g;
        g.base = r.base;
        g.floats = r.cap / 4;
        g.wGen.assign(g.floats, 0);
        g.rGen.assign(g.floats, 0);
        s.ext_regions.push_back(std::move(g));
    }
    s.phase.clear();
    s.phaseCost.clear();
}

inline void EndCore() {
    AccumulatePhase();
    St().phase.clear();
}

// Checks the reads [r0, r0 + rn), [r1, r1 + rn) and the writes [w, w + wn) of one SPAD access by unit u, then records
// them. kNone ranges are skipped.
inline void Spad(int u, uint32_t w, uint32_t wn, uint32_t r0, uint32_t r1, uint32_t rn) {
    State &s = St();
    const uint32_t none = 0xFFFFFFFFu;
    auto checkRead = [&](uint32_t i) {
        const Cell &c = s.spad[i];
        if (c.wUnit >= 0 && !Complete(c.wUnit, c.wGen) && !Ordered(c.wUnit, u)) Report("read-after-write", c.wUnit, u, "SPAD", i);
    };
    auto checkWrite = [&](uint32_t i) {
        const Cell &c = s.spad[i];
        if (c.wUnit >= 0 && !Complete(c.wUnit, c.wGen) && !Ordered(c.wUnit, u)) Report("write-after-write", c.wUnit, u, "SPAD", i);
        for (int q = 0; q < 4; ++q) {
            if (c.rGen[q] != 0 && !Complete(q, c.rGen[q]) && !Ordered(q, u)) Report("write-after-read", q, u, "SPAD", i);
        }
    };
    if (r0 != none) for (uint32_t i = r0; i < r0 + rn; ++i) checkRead(i);
    if (r1 != none) for (uint32_t i = r1; i < r1 + rn; ++i) checkRead(i);
    if (w != none) for (uint32_t i = w; i < w + wn; ++i) checkWrite(i);
    // Scalar reads finish before the scalar unit moves on, so they never leave a pending read behind.
    if (u != kS) {
        if (r0 != none) for (uint32_t i = r0; i < r0 + rn; ++i) s.spad[i].rGen[u] = s.gen[u];
        if (r1 != none) for (uint32_t i = r1; i < r1 + rn; ++i) s.spad[i].rGen[u] = s.gen[u];
    }
    if (w != none) {
        for (uint32_t i = w; i < w + wn; ++i) {
            Cell &c = s.spad[i];
            c.wUnit = static_cast<int8_t>(u);
            c.wGen = s.gen[u];
            for (auto &g : c.rGen) g = 0;
        }
    }
}

inline ExtMemShadow *FindExtMem(const uint8_t *p) {
    for (auto &g : St().ext_regions) {
        if (p >= g.base && p < g.base + g.floats * 4) return &g;
    }
    return nullptr;
}

inline void TraceExtMem(int u, const uint8_t *p, uint32_t bytes, bool write) {
    ExtMemShadow *g = FindExtMem(p);
    if (g == nullptr) return;
    const size_t f0 = static_cast<size_t>(p - g->base) / 4;
    const size_t f1 = (static_cast<size_t>(p - g->base) + bytes + 3) / 4;
    for (size_t i = f0; i < f1 && i < g->floats; ++i) {
        if (write) {
            if (g->rGen[i] != 0 && !Complete(kDI, g->rGen[i])) Report("write-after-read", kDI, u, "EXTMEM", i);
            g->wGen[i] = St().gen[u];
            g->rGen[i] = 0;
        } else {
            if (g->wGen[i] != 0 && !Complete(kDO, g->wGen[i])) Report("read-after-write", kDO, u, "EXTMEM", i);
            g->rGen[i] = St().gen[u];
        }
    }
}

}  // namespace tracechk

namespace mls_trace {

void Vec(uint32_t dst, uint32_t count, uint32_t src0, uint32_t src1) {
    // A vector instruction may update its operand in place (same start), but a destination that partially overlaps a
    // source is not a supported operand layout on the device.
    for (uint32_t src : {src0, src1}) {
        if (src == 0xFFFFFFFFu || src == dst) continue;
        if (src < dst + count && dst < src + count) {
            tracechk::State &s = tracechk::St();
            ++s.hazards;
            if (s.reports++ < 12) {
                std::fprintf(stderr, "[hazard] partially overlapping vector operands: dst %u, src %u, count %u (phase %s)\n",
                             dst, src, count, s.phase.c_str());
            }
        }
    }
    tracechk::Spad(tracechk::kV, dst, count, src0, src1, count);
}

void Scalar(uint32_t off, bool write) {
    const uint32_t none = 0xFFFFFFFFu;
    if (write) {
        tracechk::Spad(tracechk::kS, off, 1, none, none, 0);
    } else {
        tracechk::Spad(tracechk::kS, none, 0, off, none, 1);
    }
}

void Fence(bool all) {
    tracechk::State &s = tracechk::St();
    if (all) {
        for (auto &g : s.gen) ++g;
    } else {
        ++s.gen[tracechk::kV];
    }
}

void DmaIn(uint32_t dst, const void *src, const BlockCopyParams &prm) {
    const uint32_t len = prm.blockLen;
    const uint32_t dstBlk = (len + 31) / 32 * 32;
    const uint8_t *s = static_cast<const uint8_t *>(src);
    for (uint32_t b = 0; b < prm.blockCount; ++b) {
        tracechk::TraceExtMem(tracechk::kDI, s + static_cast<uint64_t>(b) * (len + prm.srcStride), len, false);
        const uint32_t d = dst + b * (dstBlk + prm.dstStride * 32) / 4;
        tracechk::Spad(tracechk::kDI, d, dstBlk / 4, 0xFFFFFFFFu, 0xFFFFFFFFu, 0);
    }
}

void DmaOut(const void *dst, uint32_t src, const BlockCopyParams &prm) {
    const uint32_t len = prm.blockLen;
    const uint32_t srcBlk = (len + 31) / 32 * 32;
    const uint8_t *d = static_cast<const uint8_t *>(dst);
    for (uint32_t b = 0; b < prm.blockCount; ++b) {
        const uint32_t s = src + b * (srcBlk + prm.srcStride * 32) / 4;
        tracechk::Spad(tracechk::kDO, 0xFFFFFFFFu, 0, s, 0xFFFFFFFFu, (len + 3) / 4);
        tracechk::TraceExtMem(tracechk::kDO, d + static_cast<uint64_t>(b) * (len + prm.dstStride), len, true);
    }
}

void Phase(const char *name) {
    tracechk::State &s = tracechk::St();
    tracechk::AccumulatePhase();
    s.phase = name;
    s.phaseStart = simemu::g_cnt;
}

}  // namespace mls_trace
