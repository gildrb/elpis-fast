#!/usr/bin/env python3
"""Finite link between the Bend model (hgemm_wide.bend, via HGEMM_WIDE_TABLE.bend) and the CUDA source of ext
patch 3011 (exllamav3_ext/hgemm_f16acc_wide.cuh and hgemm_f16acc.cu, post-patch).

1. Every source line quoted below must occur verbatim (whitespace-trimmed) in the patched files.
2. A C++ program is built from those very lines (the index expressions, loop headers, the route table and the
   wide_route / wide_enabled bodies extracted whole from hgemm_f16acc.cu) and prints the same table the Bend
   model prints: raster blocks of 7 grids, all 16384 in-tile stores, the A / B stage writes of 128 threads x 4
   iterations, the per-element partial order of both kernels at K = 64, 128, 320, 80 route decisions (4 switch
   values), grid_m at 7 values of M.
3. HGEMM_WIDE_TABLE.bend is compiled with the pinned bend and run; the two tables must be identical.

usage: hgemm_wide_diff.py [EXT_DIR]   (EXT_DIR: patched exllamav3_ext; default: the scratch pin tree)
Exit 0 and "hgemm_wide_diff: OK" iff all three steps pass."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEND = os.environ.get("BEND", "/nix/store/kqhwjzdm96d14fvzblb4jz9m73cr3i0j-bend-2.0.34/bin/bend")
EXT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kernel-work/PrefillMap/gemm/pins.tree/exllamav3_ext")

HDR_QUOTES = [
    "if constexpr (A_CPR == 4) pc = chunk ^ ((row >> 1) & 3);",
    "return row * BK + pc * 8;",
    "return row * BN + (chunk ^ (row & 7)) * 8;",
    "const int bid = blockIdx.y * grid_n + blockIdx.x;",
    "const int group_size = GROUP_M * grid_n;",
    "const int group = bid / group_size;",
    "const int first_m = group * GROUP_M;",
    "const int gm_eff = min(GROUP_M, grid_m - first_m);",
    "const int in_group = bid - group * group_size;",
    "const int bm = (first_m + in_group % gm_eff) * BM;",
    "const int bn = (in_group / gm_eff) * BN;",
    "const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;",
    "const int wm = warp / WARPS_N, wn = warp % WARPS_N;",
    "for (int i = 0; i < BM * A_CPR / THREADS; ++i)",
    "for (int i = 0; i < BK * B_CPR / THREADS; ++i)",
    "int c = tid + i * THREADS;",
    "int row = c / A_CPR, chunk = c % A_CPR;",
    "int row = c / B_CPR, chunk = c % B_CPR;",
    "cp_async16(as + a_off<A_CPR, BK>(row, chunk), src, pred);",
    "cp_async16(bs + b_off<BN>(row, chunk), src, true);",
    "const int KT = K / BK;",
    "for (int kt = 0; kt < KT; ++kt)",
    "for (int kk = 0; kk < BK / 16; kk += 2)",
    "for (int s = 0; s < 2; ++s)",
    "ldsm_x4(af[s][i], as + a_off<A_CPR, BK>(a_row + i * 16, (kk + s) * 2 + a_ch));",
    "uint32_t h[2] = {0u, 0u};",
    "mma_h(h, af[0][i], bf[0][j]);",
    "mma_h(h, af[1][i], bf[1][j]);",
    "acc[i][j][0] += f0.x; acc[i][j][1] += f0.y;",
    "const int g = lane >> 2, t = lane & 3;",
    "int col = bn + wn * WN + j * 8 + t * 2;",
    "int row = bm + wm * WM + i * 16 + g + h * 8;",
    "if (row >= M) continue;",
    "for (int i = 0; i < MT; ++i)",
    "for (int j = 0; j < NT; ++j)",
    "for (int h = 0; h < 2; ++h)",
]
CU_QUOTES = [
    "using Wide = f16acc_wide::Cfg<128, 128, 32, 2, 2, 3,  8, 2>;",
    "dim3 grid(N / Wide::BN, (M + Wide::BM - 1) / Wide::BM, 1);",
    "if (wide_route(a, b, c)) launch_wide<OUT_F32>(a, b, c, stream);",
    # served kernel, untuned layout (K order reference)
    "constexpr int BM = 128, BN = 128, BK = 64, PAD = 8;",
    "constexpr int KK = BK / 16;",
    "for (int kk = 0; kk < KK; ++kk)",
    "mma_f16(hacc[i][j], af[cur][i], bf[cur][j]);",
    "if (kk & 1) flush();     // every 32 of K",
    "hacc[i][j][0] = 0u; hacc[i][j][1] = 0u;",
]
CFG_QUOTES = [  # Cfg<BM_, BN_, BK_, WARPS_M_, WARPS_N_, ...> derived constants (header)
    "static constexpr int THREADS = WARPS_M * WARPS_N * 32;",
    "static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N, MT = WM / 16, NT = WN / 8;",
    "static constexpr int A_CPR = BK / 8, B_CPR = BN / 8;",
]


def lines_of(p):
    return [l.strip() for l in p.read_text().splitlines()]


def need(quotes, lines, name):
    missing = [q for q in quotes if q not in lines]
    if missing:
        print(f"hgemm_wide_diff: FAIL: {name} lacks quoted lines:\n  " + "\n  ".join(missing))
        sys.exit(1)


def extract(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def main():
    hdr = EXT / "hgemm_f16acc_wide.cuh"
    cu = EXT / "hgemm_f16acc.cu"
    hl, cl = lines_of(hdr), lines_of(cu)
    need(HDR_QUOTES + CFG_QUOTES, hl, hdr.name)
    need(CU_QUOTES, cl, cu.name)
    Q = {q: q for q in HDR_QUOTES + CU_QUOTES}
    cutext = cu.read_text()
    route_src = extract(cutext, "struct WideShape", "static_assert(Wide::BN == 128")
    hdrtext = hdr.read_text()
    off_src = extract(hdrtext, "template <int A_CPR, int BK> __device__ __forceinline__ int a_off", "template <bool OUT_F32, class CF>")
    cfg_src = extract(hdrtext, "template <int BM_, int BN_, int BK_", "// Swizzled element offsets")
    prog = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#define __device__
#define __forceinline__ inline
typedef uint16_t half;
struct Dim { int x, y; };
static Dim blockIdx, gridDim, threadIdx;
static int min(int a, int b) { return a < b ? a : b; }
struct Ten;
namespace at { using Tensor = ::Ten; enum ScalarType { kHalf, kFloat }; struct Props { int major, minor; }; static Props g_props;
  namespace cuda { static const Props* getDeviceProperties(int) { return &g_props; } } }
struct Dev { int index() const { return 0; } };
struct Ten { int d; int64_t s0, r, k; at::ScalarType t;
  int dim() const { return d; } int64_t size(int i) const { return i == 0 && d == 3 ? s0 : (i == -2 ? r : k); }
  Dev device() const { return Dev{}; } at::ScalarType dtype() const { return t; } };
#define TORCH_CHECK(c, m) do { if (!(c)) throw std::runtime_error(m); } while (0)
namespace f16acc_wide {
''' + cfg_src + off_src + r'''
}
using Wide = f16acc_wide::Cfg<128, 128, 32, 2, 2, 3,  8, 2>;
''' + route_src + r'''
static std::string route_of(const char* env, int major, int minor, int batch, int64_t M, int64_t K, int64_t N, bool f32) {
  if (env) setenv("EXL3_HGEMM_F16ACC_WIDE", env, 1); else unsetenv("EXL3_HGEMM_F16ACC_WIDE");
  at::g_props.major = major; at::g_props.minor = minor;
  Ten a{batch == 1 ? 2 : 3, batch, M, K, at::kHalf}, b{2, 1, K, N, at::kHalf}, c{2, 1, M, N, f32 ? at::kFloat : at::kHalf};
  try { return wide_route(a, b, c) ? "wide" : "served"; } catch (const std::exception&) { return "error"; }
}
int main() {
  constexpr int BM = Wide::BM, BN = Wide::BN, BK = Wide::BK, THREADS = Wide::THREADS, GROUP_M_DEFAULT = Wide::GROUP_M;
  constexpr int WARPS_N = Wide::WARPS_N, WM = Wide::WM, WN = Wide::WN, MT = Wide::MT, NT = Wide::NT;
  constexpr int A_CPR = Wide::A_CPR, B_CPR = Wide::B_CPR;
  (void) GROUP_M_DEFAULT;
  // raster
  int grids[7][3] = {{8, 1, 1}, {8, 3, 5}, {8, 8, 9}, {8, 5, 17}, {8, 2, 20}, {3, 4, 7}, {1, 3, 4}};
  for (auto& gg : grids) {
    const int GROUP_M = gg[0]; gridDim.x = gg[1]; gridDim.y = gg[2];
    for (int by = 0; by < gridDim.y; ++by) for (int bx = 0; bx < gridDim.x; ++bx) {
      blockIdx.x = bx; blockIdx.y = by;
      const int grid_n = gridDim.x, grid_m = gridDim.y;
      ''' + "\n      ".join(Q[q] for q in HDR_QUOTES[3:11]) + r'''
      (void) grid_m;
      printf("R %d %d %d %d %d %d\n", GROUP_M, grid_n, grid_m, bid, bm / BM, bn / BN);
    }
  }
  // stores (bm = bn = 0; M large, so no row is dropped)
  { const int bm = 0, bn = 0, M = 1 << 30;
    for (threadIdx.x = 0; threadIdx.x < THREADS; ++threadIdx.x) {
      ''' + Q["const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;"] + r'''
      ''' + Q["const int wm = warp / WARPS_N, wn = warp % WARPS_N;"] + r'''
      ''' + Q["const int g = lane >> 2, t = lane & 3;"] + r'''
      ''' + Q["for (int i = 0; i < MT; ++i)"] + r'''
        ''' + Q["for (int j = 0; j < NT; ++j)"] + r'''
        {
          ''' + Q["int col = bn + wn * WN + j * 8 + t * 2;"] + r'''
          ''' + Q["for (int h = 0; h < 2; ++h)"] + r'''
          {
            ''' + Q["int row = bm + wm * WM + i * 16 + g + h * 8;"] + r'''
            ''' + Q["if (row >= M) continue;"] + r'''
            for (int e = 0; e < 2; ++e) printf("S %d %d %d %d %d %d %d\n", tid, i, j, h, e, row, col + e);
          }
        }
    }
  }
  // stage writes (offsets in halves relative to the stage base)
  for (threadIdx.x = 0; threadIdx.x < THREADS; ++threadIdx.x) {
    const int tid = threadIdx.x;
    std::vector<int> av, bv;
    ''' + Q["for (int i = 0; i < BM * A_CPR / THREADS; ++i)"] + r'''
    {
      ''' + Q["int c = tid + i * THREADS;"] + r'''
      ''' + Q["int row = c / A_CPR, chunk = c % A_CPR;"] + r'''
      av.push_back(f16acc_wide::a_off<A_CPR, BK>(row, chunk));
    }
    ''' + Q["for (int i = 0; i < BK * B_CPR / THREADS; ++i)"] + r'''
    {
      ''' + Q["int c = tid + i * THREADS;"] + r'''
      ''' + Q["int row = c / B_CPR, chunk = c % B_CPR;"] + r'''
      bv.push_back(f16acc_wide::b_off<BN>(row, chunk));
    }
    for (int i = 0; i < 4; ++i) { printf("A %d %d %d\nB %d %d %d\n", tid, i, av[i], tid, i, bv[i]); }
  }
  // accumulation order of one element: partials added to the fp32 accumulator
  for (int K : {64, 128, 320}) {
    { printf("K wide %d", K);
      ''' + Q["const int KT = K / BK;"] + r'''
      ''' + Q["for (int kt = 0; kt < KT; ++kt)"] + r'''
        ''' + Q["for (int kk = 0; kk < BK / 16; kk += 2)"] + r'''
        {
          int af[2];
          ''' + Q["for (int s = 0; s < 2; ++s)"] + r''' af[s] = kt * BK + (kk + s) * 16;
          std::vector<int> hp;   // uint32_t h[2] = {0u, 0u};
          hp.push_back(af[0]);   // mma_h(h, af[0][i], bf[0][j]);
          hp.push_back(af[1]);   // mma_h(h, af[1][i], bf[1][j]);
          for (int x : hp) printf(" %d", x); printf(" ;");   // acc += h
        }
      printf("\n"); }
    { printf("K served %d", K);
      constexpr int BK = 64;
      ''' + Q["constexpr int KK = BK / 16;"] + r'''
      std::vector<int> hacc;
      auto flush = [&]() { for (int x : hacc) printf(" %d", x); printf(" ;"); hacc.clear(); };
      const int KT = K / BK;
      for (int kt = 0; kt < KT; ++kt)
        ''' + Q["for (int kk = 0; kk < KK; ++kk)"] + r'''
        {
          hacc.push_back(kt * BK + kk * 16);   // mma_f16(hacc[i][j], af[cur][i], bf[cur][j]);  (af[cur]: k16 step kk)
          ''' + Q["if (kk & 1) flush();     // every 32 of K"] + r'''
        }
      printf("\n"); }
  }
  // route
  struct Case { int ma, mi, ba; int64_t M, K, N; bool f; };
  Case cases[] = {{8,6,1,2048,5120,1024,false},{8,6,1,2048,5120,6144,true},{8,6,1,2048,5120,10240,true},{8,6,1,2048,5120,12288,false},
    {8,6,1,1792,5120,17408,false},{8,6,1,1024,6144,5120,true},{8,6,1,4096,17408,5120,true},{8,6,1,2048,25600,5120,false},
    {8,6,1,1023,5120,17408,false},{8,6,1,247,5120,17408,false},{8,6,2,2048,5120,17408,false},{8,9,1,2048,5120,17408,false},
    {12,0,1,2048,5120,17408,false},{8,0,1,2048,5120,17408,false},{8,6,1,2048,5120,17408,true},{8,6,1,2048,5120,1024,true},
    {8,6,1,2048,4096,4096,false},{8,6,1,2048,5120,17536,false},{8,6,1,2048,5248,5120,true},{8,6,1,2048,25600,5120,true}};
  const char* envs[4] = {nullptr, "1", "0", "x"};
  const char* names[4] = {"unset", "1", "0", "x"};
  for (int e = 0; e < 4; ++e) for (auto& c : cases)
    printf("Q %s %d %d %d %lld %lld %lld %d %s\n", names[e], c.ma, c.mi, c.ba, (long long) c.M, (long long) c.K, (long long) c.N,
           c.f ? 1 : 0, route_of(envs[e], c.ma, c.mi, c.ba, c.M, c.K, c.N, c.f).c_str());
  // launch grid rows
  for (int M : {1, 127, 128, 129, 1100, 2048, 262143}) printf("M %d %d\n", M, (M + Wide::BM - 1) / Wide::BM);
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "hw_ref.cpp"
        src.write_text(prog)
        exe = Path(td) / "hw_ref"
        r = subprocess.run(["c++", "-std=c++17", "-O1", "-o", str(exe), str(src)], capture_output=True, text=True)
        if r.returncode:
            print("hgemm_wide_diff: FAIL: reference program does not compile:\n" + r.stderr[-3000:])
            sys.exit(1)
        ref = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
        tb = Path(td) / "hw_table"
        r = subprocess.run([BEND, str(HERE / "HGEMM_WIDE_TABLE.bend"), "-o", str(tb)], capture_output=True, text=True, cwd=HERE)
        if r.returncode:
            print("hgemm_wide_diff: FAIL: bend table build failed:\n" + (r.stdout + r.stderr)[-3000:])
            sys.exit(1)
        got = subprocess.run([str(tb)], capture_output=True, text=True, check=True).stdout
    a, b = ref.strip().splitlines(), got.strip().splitlines()
    if a != b:
        for k, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"hgemm_wide_diff: MISMATCH at line {k + 1}:\n  C++  : {x}\n  Bend : {y}")
                break
        else:
            print(f"hgemm_wide_diff: MISMATCH: {len(a)} C++ lines vs {len(b)} Bend lines")
        sys.exit(1)
    kinds = {}
    for l in a:
        kinds[l.split()[0]] = kinds.get(l.split()[0], 0) + 1
    print(f"hgemm_wide_diff: {len(a)} lines identical {kinds}")
    print("hgemm_wide_diff: OK")


if __name__ == "__main__":
    main()
