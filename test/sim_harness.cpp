// Host-side test harness for kernel/matrix_least_squares.cpp on the CPU simulator (include/simd_machine.hpp).
//
// For every case it builds float32 inputs with a prescribed condition number, sizes the workspace exactly as
// host/matrix_least_squares.cpp does, runs every core of the simulated machine, and checks
//   - the solution and residuals against a float64 least-squares solve of the same float32 inputs
//     (element-wise |out - ref| <= 1e-4 + 1e-4 * |ref|; the "ratio" column is the worst left/right quotient),
//   - the simulator fault counters (alignment, out-of-bounds, cross-core write conflicts, SPAD overflow, ...),
// and reports the cost counters of the busiest core (the critical path on the device) plus a rough cycle estimate.
//
// Usage:
//   sim_harness                      run the built-in suite
//   sim_harness sweep count [seed]   run `count` random shapes
//   sim_harness m n k batch [cond] [seed] [b_has_batch] [cores] [spectrum]   one case, with a per-phase cost breakdown
//                                    (spectrum: 0 linear, 1 geometric, 2 clustered at 1/cond, 3 clustered at 1)
//
// The kernel is built with MLS_TRACE: test/trace_checker.hpp checks its fence placement against the device's
// asynchronous units and splits its cost counters by phase.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "simd_machine.hpp"
using namespace hpc;
#define MLS_TRACE
#include "trace_checker.hpp"
#include "../kernel/matrix_least_squares.cpp"

namespace {

// ---------------------------------------------------------------- random inputs

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
    uint64_t Next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double Uniform() { return (static_cast<double>(Next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double Normal() {
        const double u = Uniform(), v = Uniform();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
    }
};

// Orthonormalizes the columns of the column-major rows x cols matrix q (two passes of modified Gram-Schmidt).
void Orthonormalize(std::vector<double> &q, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        double *qj = &q[static_cast<size_t>(j) * rows];
        for (int pass = 0; pass < 2; ++pass) {
            for (int p = 0; p < j; ++p) {
                const double *qp = &q[static_cast<size_t>(p) * rows];
                double d = 0;
                for (int i = 0; i < rows; ++i) d += qp[i] * qj[i];
                for (int i = 0; i < rows; ++i) qj[i] -= d * qp[i];
            }
        }
        double nrm = 0;
        for (int i = 0; i < rows; ++i) nrm += qj[i] * qj[i];
        nrm = std::sqrt(nrm);
        for (int i = 0; i < rows; ++i) qj[i] /= nrm;
    }
}

// Singular value p of n for condition number `cond`: spectrum 0 is linear from 1 to 1/cond (test/verify_local.py),
// 1 geometric, 2 one value at 1 and the rest at 1/cond, 3 one value at 1/cond and the rest at 1.
double Singular(int p, int n, double cond, int spectrum) {
    if (n == 1) return 1.0;
    const double t = static_cast<double>(p) / (n - 1);
    switch (spectrum) {
        case 1: return std::pow(cond, -t);
        case 2: return p == 0 ? 1.0 : 1.0 / cond;
        case 3: return p == n - 1 ? 1.0 / cond : 1.0;
        default: return 1.0 + (1.0 / cond - 1.0) * t;
    }
}

// A = U diag(s) V^T with random orthonormal U, V. Row-major m x n float32.
std::vector<float> MakeMatrix(int m, int n, double cond, int spectrum, Rng &rng) {
    std::vector<double> u(static_cast<size_t>(m) * n), v(static_cast<size_t>(n) * n);
    for (auto &x : u) x = rng.Normal();
    for (auto &x : v) x = rng.Normal();
    Orthonormalize(u, m, n);
    Orthonormalize(v, n, n);
    std::vector<float> a(static_cast<size_t>(m) * n);
    std::vector<double> row(n);
    for (int i = 0; i < m; ++i) {
        std::fill(row.begin(), row.end(), 0.0);
        for (int p = 0; p < n; ++p) {
            const double s = Singular(p, n, cond, spectrum);
            const double up = u[static_cast<size_t>(p) * m + i] * s;
            const double *vp = &v[static_cast<size_t>(p) * n];  // column p of V
            for (int c = 0; c < n; ++c) row[c] += up * vp[c];
        }
        for (int c = 0; c < n; ++c) a[static_cast<size_t>(i) * n + c] = static_cast<float>(row[c]);
    }
    return a;
}

// ---------------------------------------------------------------- float64 reference

// Least-squares solution (n x k, row-major) and squared residual norms of min ||A X - B|| for row-major float inputs,
// by Householder QR in double precision.
void Reference(const float *a, const float *b, int m, int n, int k, std::vector<double> &x, std::vector<double> &res) {
    std::vector<double> c(static_cast<size_t>(n + k) * m);  // column-major [A | B]
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) c[static_cast<size_t>(j) * m + i] = a[static_cast<size_t>(i) * n + j];
        for (int j = 0; j < k; ++j) c[static_cast<size_t>(n + j) * m + i] = b[static_cast<size_t>(i) * k + j];
    }
    std::vector<double> diag(n);
    for (int j = 0; j < n; ++j) {
        double *cj = &c[static_cast<size_t>(j) * m];
        double nrm = 0;
        for (int i = j; i < m; ++i) nrm += cj[i] * cj[i];
        nrm = std::sqrt(nrm);
        const double alpha = (cj[j] >= 0) ? -nrm : nrm;
        cj[j] -= alpha;
        const double vtv = 2.0 * alpha * (alpha - (cj[j] + alpha));
        for (int q = j + 1; q < n + k; ++q) {
            double *cq = &c[static_cast<size_t>(q) * m];
            double d = 0;
            for (int i = j; i < m; ++i) d += cj[i] * cq[i];
            d *= 2.0 / vtv;
            for (int i = j; i < m; ++i) cq[i] -= d * cj[i];
        }
        diag[j] = alpha;
    }
    x.assign(static_cast<size_t>(n) * k, 0.0);
    res.assign(k, 0.0);
    for (int r = 0; r < k; ++r) {
        const double *y = &c[static_cast<size_t>(n + r) * m];
        for (int j = n - 1; j >= 0; --j) {
            double s = y[j];
            for (int q = j + 1; q < n; ++q) s -= c[static_cast<size_t>(q) * m + j] * x[static_cast<size_t>(q) * k + r];
            x[static_cast<size_t>(j) * k + r] = s / diag[j];
        }
        // Residual from the definition, so that it does not depend on the factorization.
        double rr = 0;
        for (int i = 0; i < m; ++i) {
            double t = b[static_cast<size_t>(i) * k + r];
            for (int j = 0; j < n; ++j) t -= static_cast<double>(a[static_cast<size_t>(i) * n + j]) * x[static_cast<size_t>(j) * k + r];
            rr += t * t;
        }
        res[r] = rr;
    }
}

// ---------------------------------------------------------------- host workspace size (mirrors host/matrix_least_squares.cpp)

size_t HostWorkspaceFloats(uint32_t m, uint32_t n, uint32_t k, uint32_t batch_size, uint32_t num_simd_cores) {
    const size_t big_dim = (m >= n) ? m : n;
    const size_t small_dim = (m >= n) ? n : m;
    const size_t main_slot = 2 * small_dim * (((big_dim + 7) / 8) * 8) + static_cast<size_t>(k) * (((m + 7) / 8) * 8) +
                             static_cast<size_t>(k) * (((n + 7) / 8) * 8);
    uint32_t unit = 1;
    while ((static_cast<uint64_t>(unit) * n * k) % 8 != 0) ++unit;
    const size_t tiles = (static_cast<size_t>(batch_size) + unit - 1) / unit;
    const size_t working = (tiles < static_cast<size_t>(num_simd_cores)) ? tiles : static_cast<size_t>(num_simd_cores);
    size_t ws_floats = working * main_slot;
    const size_t big_pad = ((big_dim + 7) / 8) * 8;
    const size_t small_pad = ((small_dim + 7) / 8) * 8;
    const bool legacy = (k > big_pad) || (13 * big_pad + 10 * small_pad + 1472 > 46000);
    if (legacy) {
        uint32_t s_stride = ((n + k + 7) / 8) * 8;
        size_t legacy_floats = static_cast<size_t>(num_simd_cores) * m * s_stride + static_cast<size_t>(batch_size) * n * s_stride;
        const size_t r_stride = static_cast<size_t>(((k + 7) / 8) * 8);
        const size_t per_batch_slots = static_cast<size_t>(batch_size) * m * (s_stride + r_stride);
        if (legacy_floats < per_batch_slots) legacy_floats = per_batch_slots;
        if (ws_floats < legacy_floats) ws_floats = legacy_floats;
    }
    if (ws_floats < static_cast<size_t>(2 * 1024 * 1024)) ws_floats = static_cast<size_t>(2 * 1024 * 1024);
    return ws_floats;
}

// ---------------------------------------------------------------- cost model

// Rough cycle estimate of one core: vector throughput of 256 B per cycle, a few cycles to issue an instruction or to
// touch SPAD from the scalar unit, a pipeline drain per SIMD fence, a full drain per global fence, and a fixed latency
// plus bandwidth per block copy. Only the ratios matter; the raw counters are printed next to it.
double Cycles(const simemu::Counters &c) {
    return 1.0 * c.vec_repeats + 4.0 * c.vec_calls + 4.0 * (c.scalar_rd + c.scalar_wr) + 20.0 * c.fence_simd +
           100.0 * c.fence_all + 500.0 * c.dma_calls + c.dma_bytes / 64.0;
}

simemu::Counters Diff(const simemu::Counters &a, const simemu::Counters &b) {
    simemu::Counters d;
    d.scalar_rd = a.scalar_rd - b.scalar_rd;
    d.scalar_wr = a.scalar_wr - b.scalar_wr;
    d.vec_calls = a.vec_calls - b.vec_calls;
    d.vec_repeats = a.vec_repeats - b.vec_repeats;
    d.dma_calls = a.dma_calls - b.dma_calls;
    d.dma_bytes = a.dma_bytes - b.dma_bytes;
    d.fence_simd = a.fence_simd - b.fence_simd;
    d.fence_all = a.fence_all - b.fence_all;
    return d;
}

struct Case {
    int m, n, k, batch;
    double cond;
    uint64_t seed;
    int b_has_batch;  // -1: batched B when batch > 1
    int cores;
    int spectrum = 0;
};

struct Result {
    double ratio_sol = 0, ratio_res = 0;
    uint64_t faults = 0;
    uint64_t hazards = 0;
    std::map<std::string, simemu::Counters> phases;  // of the most expensive core
    simemu::Counters worst;  // counters of the most expensive core
    double worst_cycles = 0, total_cycles = 0;
    int busy_cores = 0;
    size_t spad_peak = 0;
};

Result RunCase(const Case &cs, bool verbose) {
    const int m = cs.m, n = cs.n, k = cs.k, batch = cs.batch;
    const int b_has_batch = (cs.b_has_batch < 0) ? (batch > 1 ? 1 : 0) : cs.b_has_batch;
    const bool has_res = m > n;
    Rng rng(cs.seed);
    std::vector<float> a(static_cast<size_t>(batch) * m * n);
    const int b_count = b_has_batch ? batch : 1;
    std::vector<float> b(static_cast<size_t>(b_count) * m * k);
    for (int t = 0; t < batch; ++t) {
        std::vector<float> at = MakeMatrix(m, n, cs.cond, cs.spectrum, rng);
        std::copy(at.begin(), at.end(), a.begin() + static_cast<size_t>(t) * m * n);
    }
    for (auto &x : b) x = static_cast<float>(rng.Normal());

    simemu::g_faults = simemu::Faults();
    simemu::g_cnt = simemu::Counters();
    simemu::g_spad_peak = 0;
    simemu::g_reports = 0;
    uint8_t *ga = simemu::AllocRegion(a.size() * 4, "A");
    uint8_t *gb = simemu::AllocRegion(b.size() * 4, "B");
    uint8_t *gx = simemu::AllocRegion(static_cast<size_t>(batch) * n * k * 4, "solution");
    uint8_t *gr = has_res ? simemu::AllocRegion(static_cast<size_t>(batch) * k * 4, "residuals") : nullptr;
    const size_t ws = HostWorkspaceFloats(m, n, k, batch, cs.cores);
    uint8_t *gw = simemu::AllocRegion(ws * 4, "workspace");
    std::memcpy(ga, a.data(), a.size() * 4);
    std::memcpy(gb, b.data(), b.size() * 4);
    MatrixLeastSquaresLayoutData layout{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                                        static_cast<uint32_t>(batch), has_res ? 1u : 0u,
                                        static_cast<uint32_t>(b_has_batch)};
    Result r;
    tracechk::St().hazards = 0;
    tracechk::St().reports = 0;
    for (int c = 0; c < cs.cores; ++c) {
        simemu::BeginCore(c, cs.cores);
        tracechk::BeginCore();
        const simemu::Counters before = simemu::g_cnt;
        matrix_least_squares<float>(ga, gb, gx, gr, gw, reinterpret_cast<MemAddr>(&layout));
        tracechk::EndCore();
        const simemu::Counters d = Diff(simemu::g_cnt, before);
        const double cy = Cycles(d);
        r.total_cycles += cy;
        if (d.vec_calls + d.dma_calls + d.scalar_rd > 0) ++r.busy_cores;
        if (cy > r.worst_cycles) {
            r.worst_cycles = cy;
            r.worst = d;
            r.phases = tracechk::St().phaseCost;
        }
    }
    r.faults = simemu::g_faults.total();
    r.hazards = tracechk::St().hazards;
    r.spad_peak = simemu::g_spad_peak;

    std::vector<double> xr, rr;
    for (int t = 0; t < batch; ++t) {
        const float *at = a.data() + static_cast<size_t>(t) * m * n;
        const float *bt = b.data() + (b_has_batch ? static_cast<size_t>(t) * m * k : 0);
        Reference(at, bt, m, n, k, xr, rr);
        const float *xo = reinterpret_cast<const float *>(gx) + static_cast<size_t>(t) * n * k;
        double worst = 0;
        int wi = 0;
        for (int i = 0; i < n * k; ++i) {
            const double e = std::fabs(static_cast<double>(xo[i]) - xr[i]) / (1e-4 + 1e-4 * std::fabs(xr[i]));
            if (!(e <= r.ratio_sol)) r.ratio_sol = std::isnan(e) ? INFINITY : e;
            if (!(e <= worst)) {
                worst = std::isnan(e) ? INFINITY : e;
                wi = i;
            }
        }
        if (verbose) {
            double xmax = 0;
            for (int i = 0; i < n * k; ++i) xmax = std::max(xmax, std::fabs(xr[i]));
            std::printf("    batch %d: ratio %.3g at x[%d][%d] = %.9g (ref %.9g), max|x| %.3g\n", t, worst, wi / k, wi % k,
                        xo[wi], xr[wi], xmax);
        }
        if (has_res) {
            const float *ro = reinterpret_cast<const float *>(gr) + static_cast<size_t>(t) * k;
            for (int i = 0; i < k; ++i) {
                const double e = std::fabs(static_cast<double>(ro[i]) - rr[i]) / (1e-4 + 1e-4 * std::fabs(rr[i]));
                if (!(e <= r.ratio_res)) r.ratio_res = std::isnan(e) ? INFINITY : e;
            }
        }
    }
    if (verbose) {
        const auto &w = r.worst;
        std::printf("    worst core: vec %llu (repeats %llu)  scalar r/w %llu/%llu  dma %llu (%llu B)  fence simd/all %llu/%llu\n",
                    (unsigned long long)w.vec_calls, (unsigned long long)w.vec_repeats, (unsigned long long)w.scalar_rd,
                    (unsigned long long)w.scalar_wr, (unsigned long long)w.dma_calls, (unsigned long long)w.dma_bytes,
                    (unsigned long long)w.fence_simd, (unsigned long long)w.fence_all);
        std::printf("    %-11s %10s %9s %10s %9s %9s %7s %8s %6s\n", "phase", "cycles", "vec", "repeats", "scal_rd",
                    "scal_wr", "dma", "f_simd", "f_all");
        for (const auto &ph : r.phases) {
            const auto &c = ph.second;
            std::printf("    %-11s %10.0f %9llu %10llu %9llu %9llu %7llu %8llu %6llu\n", ph.first.c_str(), Cycles(c),
                        (unsigned long long)c.vec_calls, (unsigned long long)c.vec_repeats,
                        (unsigned long long)c.scalar_rd, (unsigned long long)c.scalar_wr,
                        (unsigned long long)c.dma_calls, (unsigned long long)c.fence_simd,
                        (unsigned long long)c.fence_all);
        }
    }
    simemu::FreeRegions();
    return r;
}

void PrintHeader() {
    std::printf("%5s %4s %3s %3s %5s %6s | %9s %9s %6s %7s | %5s %12s %12s\n", "m", "n", "k", "bat", "Bbat", "cond",
                "ratio_x", "ratio_r", "faults", "hazards", "cores", "cyc/core", "cyc_total");
}

bool PrintRow(const Case &cs, const Result &r) {
    const bool ok = r.faults == 0 && r.hazards == 0 && r.ratio_sol <= 1.0 && r.ratio_res <= 1.0;
    std::printf("%5d %4d %3d %3d %5d %6.0f | %9.3g %9.3g %6llu %7llu | %5d %12.0f %12.0f %s\n", cs.m, cs.n, cs.k,
                cs.batch, cs.b_has_batch, cs.cond, r.ratio_sol, r.ratio_res, (unsigned long long)r.faults,
                (unsigned long long)r.hazards, r.busy_cores, r.worst_cycles, r.total_cycles, ok ? "ok" : "FAIL");
    std::fflush(stdout);
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    std::vector<Case> cases;
    bool verbose = false;
    if (argc >= 3 && std::strcmp(argv[1], "sweep") == 0) {
        const int count = std::atoi(argv[2]);
        Rng pick(argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 7);
        for (int i = 0; i < count; ++i) {
            // Mostly small and medium shapes, some large ones.
            const int mmax = (pick.Next() % 4 == 0) ? 1024 : 160;
            const int m = 1 + static_cast<int>(pick.Next() % mmax);
            const int nmax = std::min(m, 512);
            const int n = 1 + static_cast<int>(pick.Next() % nmax);
            const int k = 1 + static_cast<int>(pick.Next() % ((pick.Next() % 3 == 0) ? 64 : 8));
            const int batch = 1 + static_cast<int>(pick.Next() % ((pick.Next() % 3 == 0) ? 32 : 4));
            const int bb = (pick.Next() % 5 == 0) ? 0 : -1;
            const double cond = (pick.Next() % 2) ? 1000.0 : 1.0 + static_cast<double>(pick.Next() % 1000);
            cases.push_back(Case{m, n, k, batch, cond, pick.Next() % 100000, bb, 40});
        }
    } else if (argc >= 5) {
        Case c{std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), 1000.0, 1, -1, 40};
        if (argc > 5) c.cond = std::atof(argv[5]);
        if (argc > 6) c.seed = std::strtoull(argv[6], nullptr, 10);
        if (argc > 7) c.b_has_batch = std::atoi(argv[7]);
        if (argc > 8) c.cores = std::atoi(argv[8]);
        if (argc > 9) c.spectrum = std::atoi(argv[9]);
        cases.push_back(c);
        verbose = true;
    } else {
        const int list[][4] = {
            {1, 1, 1, 1},     {2, 1, 1, 1},      {3, 2, 1, 3},     {5, 3, 2, 7},     {8, 8, 1, 1},      {4, 4, 3, 5},
            {9, 9, 9, 2},     {16, 16, 3, 5},    {17, 5, 64, 3},   {8, 4, 64, 2},    {20, 5, 1, 32},    {33, 7, 3, 32},
            {40, 39, 2, 9},   {64, 32, 4, 2},    {100, 100, 7, 3}, {128, 64, 8, 4},  {127, 61, 13, 5},  {200, 3, 5, 6},
            {256, 128, 16, 3}, {300, 300, 2, 2}, {511, 255, 9, 2}, {1024, 64, 4, 2}, {1024, 1, 1, 4},   {1000, 500, 64, 1},
        };
        uint64_t seed = 11;
        for (const auto &e : list) cases.push_back(Case{e[0], e[1], e[2], e[3], 1000.0, seed++, -1, 40});
        cases.push_back(Case{64, 16, 4, 8, 1000.0, seed++, 0, 40});
        cases.push_back(Case{48, 48, 5, 4, 1000.0, seed++, 0, 40});
        cases.push_back(Case{96, 40, 6, 32, 1.0, seed++, -1, 40});
    }
    PrintHeader();
    int fails = 0;
    for (const auto &c : cases) {
        const auto t0 = std::chrono::steady_clock::now();
        const Result r = RunCase(c, verbose);
        if (!PrintRow(c, r)) ++fails;
        if (verbose) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("    wall %.2f s\n", s);
        }
    }
    std::printf("%s: %d of %zu cases failed\n", fails ? "FAIL" : "PASS", fails, cases.size());
    return fails ? 1 : 0;
}
