#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite link between the Bend model and the CUDA source of ext patch 3011.

The model is hgemm_wide.bend, via HGEMM_WIDE_TABLE.bend; the source is
exllamav3_ext/hgemm_f16acc_wide.cuh and hgemm_f16acc.cu, post-patch.

1. Every source line quoted below must occur verbatim (whitespace-trimmed) in
   the patched files.
2. A C++ program is built from those very lines (the index expressions, loop
   headers, the route table and the wide_route / wide_enabled bodies extracted
   whole from hgemm_f16acc.cu) and prints the same table the Bend model
   prints: raster blocks of 7 grids, all 16384 in-tile stores, the A / B stage
   writes of 128 threads x 4 iterations, the per-element partial order of both
   kernels at K = 64, 128, 320, 80 route decisions (4 switch values), grid_m
   at 7 values of M.
3. HGEMM_WIDE_TABLE.bend is compiled with the pinned bend and run; the two
   tables must be identical.

usage: hgemm_wide_diff.py EXT_DIR
  (EXT_DIR: exllamav3_ext of OUT/patched of bend/engine_trees.py)
Exit 0 and "hgemm_wide_diff: OK" iff all three steps pass.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

HERE = Path(__file__).resolve().parent
ARGC = 2
TAIL = 3000

USAGE = (
    "Finite link between the Bend model (hgemm_wide.bend, via "
    "HGEMM_WIDE_TABLE.bend) and the CUDA source of ext\n"
    "patch 3011 (exllamav3_ext/hgemm_f16acc_wide.cuh and "
    "hgemm_f16acc.cu, post-patch).\n"
    "\n"
    "1. Every source line quoted below must occur verbatim "
    "(whitespace-trimmed) in the patched files.\n"
    "2. A C++ program is built from those very lines (the index "
    "expressions, loop headers, the route table and the\n"
    "   wide_route / wide_enabled bodies extracted whole from "
    "hgemm_f16acc.cu) and prints the same table the Bend\n"
    "   model prints: raster blocks of 7 grids, all 16384 in-tile "
    "stores, the A / B stage writes of 128 threads x 4\n"
    "   iterations, the per-element partial order of both kernels at K "
    "= 64, 128, 320, 80 route decisions (4 switch\n"
    "   values), grid_m at 7 values of M.\n"
    "3. HGEMM_WIDE_TABLE.bend is compiled with the pinned bend and "
    "run; the two tables must be identical.\n"
    "\n"
    "usage: hgemm_wide_diff.py EXT_DIR   (EXT_DIR: exllamav3_ext of "
    "OUT/patched of bend/engine_trees.py)\n"
    'Exit 0 and "hgemm_wide_diff: OK" iff all three steps pass.'
)

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
# Cfg<BM_, BN_, BK_, WARPS_M_, WARPS_N_, ...> derived constants (header)
CFG_QUOTES = [
    "static constexpr int THREADS = WARPS_M * WARPS_N * 32;",
    (
        "static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N, "
        "MT = WM / 16, NT = WN / 8;"
    ),
    "static constexpr int A_CPR = BK / 8, B_CPR = BN / 8;",
]


def lines_of(p: Path) -> list[str]:
    """Read a file as whitespace-trimmed lines.

    Args:
        p: The file.

    Returns:
        Its lines, stripped.

    """
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]


def need(quotes: list[str], lines: list[str], name: str) -> None:
    """Exit 1 unless every quoted line occurs in lines.

    Args:
        quotes: The quoted source lines.
        lines: The file's trimmed lines.
        name: The file name for the message.

    """
    missing = [q for q in quotes if q not in lines]
    if missing:
        sys.stdout.write(
            f"hgemm_wide_diff: FAIL: {name} lacks quoted lines:\n  "
            + "\n  ".join(missing)
            + "\n"
        )
        sys.exit(1)


def extract(text: str, start: str, end: str) -> str:
    """Return text from the first start up to the next end.

    Args:
        text: The source text.
        start: The opening anchor (included).
        end: The closing anchor (excluded).

    Returns:
        The slice.

    """
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def program(cfg_src: str, off_src: str, route_src: str) -> str:
    """Build the C++ reference program from the quoted and extracted source.

    Args:
        cfg_src: The header's Cfg template.
        off_src: The header's swizzled offset functions.
        route_src: The .cu file's route table and functions.

    Returns:
        The program text.

    """
    q_map = {q: q for q in HDR_QUOTES + CU_QUOTES}
    return (
        r"""
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
namespace at { using Tensor = ::Ten; enum ScalarType { kHalf, kFloat }; struct"""
        r""" Props { int major, minor; }; static Props g_props;
  namespace cuda { static const Props* getDeviceProperties(int) { return &g_props; } } }
struct Dev { int index() const { return 0; } };
struct Ten { int d; int64_t s0, r, k; at::ScalarType t;
  int dim() const { return d; } int64_t size(int i) const { return i == 0 && d"""
        r""" == 3 ? s0 : (i == -2 ? r : k); }
  Dev device() const { return Dev{}; } at::ScalarType dtype() const { return t; } };
#define TORCH_CHECK(c, m) do { if (!(c)) throw std::runtime_error(m); } while (0)
namespace f16acc_wide {
"""
        + cfg_src
        + off_src
        + r"""
}
using Wide = f16acc_wide::Cfg<128, 128, 32, 2, 2, 3,  8, 2>;
"""
        + route_src
        + r"""
static std::string route_of(const char* env, int major, int minor, int batch,"""
        r""" int64_t M, int64_t K, int64_t N, bool f32) {
  if (env) setenv("EXL3_HGEMM_F16ACC_WIDE", env, 1); else"""
        r""" unsetenv("EXL3_HGEMM_F16ACC_WIDE");
  at::g_props.major = major; at::g_props.minor = minor;
  Ten a{batch == 1 ? 2 : 3, batch, M, K, at::kHalf}, b{2, 1, K, N, at::kHalf},"""
        r""" c{2, 1, M, N, f32 ? at::kFloat : at::kHalf};
  try { return wide_route(a, b, c) ? "wide" : "served"; } catch (const"""
        r""" std::exception&) { return "error"; }
}
int main() {
  constexpr int BM = Wide::BM, BN = Wide::BN, BK = Wide::BK, THREADS ="""
        r""" Wide::THREADS, GROUP_M_DEFAULT = Wide::GROUP_M;
  constexpr int WARPS_N = Wide::WARPS_N, WM = Wide::WM, WN = Wide::WN, MT ="""
        r""" Wide::MT, NT = Wide::NT;
  constexpr int A_CPR = Wide::A_CPR, B_CPR = Wide::B_CPR;
  (void) GROUP_M_DEFAULT;
  // raster
  int grids[7][3] = {{8, 1, 1}, {8, 3, 5}, {8, 8, 9}, {8, 5, 17}, {8, 2, 20},"""
        r""" {3, 4, 7}, {1, 3, 4}};
  for (auto& gg : grids) {
    const int GROUP_M = gg[0]; gridDim.x = gg[1]; gridDim.y = gg[2];
    for (int by = 0; by < gridDim.y; ++by) for (int bx = 0; bx < gridDim.x; ++bx) {
      blockIdx.x = bx; blockIdx.y = by;
      const int grid_n = gridDim.x, grid_m = gridDim.y;
      """
        + "\n      ".join(q_map[q] for q in HDR_QUOTES[3:11])
        + r"""
      (void) grid_m;
      printf("R %d %d %d %d %d %d\n", GROUP_M, grid_n, grid_m, bid, bm / BM, bn / BN);
    }
  }
  // stores (bm = bn = 0; M large, so no row is dropped)
  { const int bm = 0, bn = 0, M = 1 << 30;
    for (threadIdx.x = 0; threadIdx.x < THREADS; ++threadIdx.x) {
      """
        + q_map["const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;"]
        + r"""
      """
        + q_map["const int wm = warp / WARPS_N, wn = warp % WARPS_N;"]
        + r"""
      """
        + q_map["const int g = lane >> 2, t = lane & 3;"]
        + r"""
      """
        + q_map["for (int i = 0; i < MT; ++i)"]
        + r"""
        """
        + q_map["for (int j = 0; j < NT; ++j)"]
        + r"""
        {
          """
        + q_map["int col = bn + wn * WN + j * 8 + t * 2;"]
        + r"""
          """
        + q_map["for (int h = 0; h < 2; ++h)"]
        + r"""
          {
            """
        + q_map["int row = bm + wm * WM + i * 16 + g + h * 8;"]
        + r"""
            """
        + q_map["if (row >= M) continue;"]
        + r"""
            for (int e = 0; e < 2; ++e) printf("S %d %d %d %d %d %d %d\n", tid,"""
        r""" i, j, h, e, row, col + e);
          }
        }
    }
  }
  // stage writes (offsets in halves relative to the stage base)
  for (threadIdx.x = 0; threadIdx.x < THREADS; ++threadIdx.x) {
    const int tid = threadIdx.x;
    std::vector<int> av, bv;
    """
        + q_map["for (int i = 0; i < BM * A_CPR / THREADS; ++i)"]
        + r"""
    {
      """
        + q_map["int c = tid + i * THREADS;"]
        + r"""
      """
        + q_map["int row = c / A_CPR, chunk = c % A_CPR;"]
        + r"""
      av.push_back(f16acc_wide::a_off<A_CPR, BK>(row, chunk));
    }
    """
        + q_map["for (int i = 0; i < BK * B_CPR / THREADS; ++i)"]
        + r"""
    {
      """
        + q_map["int c = tid + i * THREADS;"]
        + r"""
      """
        + q_map["int row = c / B_CPR, chunk = c % B_CPR;"]
        + r"""
      bv.push_back(f16acc_wide::b_off<BN>(row, chunk));
    }
    for (int i = 0; i < 4; ++i) { printf("A %d %d %d\nB %d %d %d\n", tid, i,"""
        r""" av[i], tid, i, bv[i]); }
  }
  // accumulation order of one element: partials added to the fp32 accumulator
  for (int K : {64, 128, 320}) {
    { printf("K wide %d", K);
      """
        + q_map["const int KT = K / BK;"]
        + r"""
      """
        + q_map["for (int kt = 0; kt < KT; ++kt)"]
        + r"""
        """
        + q_map["for (int kk = 0; kk < BK / 16; kk += 2)"]
        + r"""
        {
          int af[2];
          """
        + q_map["for (int s = 0; s < 2; ++s)"]
        + r""" af[s] = kt * BK + (kk + s) * 16;
          std::vector<int> hp;   // uint32_t h[2] = {0u, 0u};
          hp.push_back(af[0]);   // mma_h(h, af[0][i], bf[0][j]);
          hp.push_back(af[1]);   // mma_h(h, af[1][i], bf[1][j]);
          for (int x : hp) printf(" %d", x); printf(" ;");   // acc += h
        }
      printf("\n"); }
    { printf("K served %d", K);
      constexpr int BK = 64;
      """
        + q_map["constexpr int KK = BK / 16;"]
        + r"""
      std::vector<int> hacc;
      auto flush = [&]() { for (int x : hacc) printf(" %d", x); printf(" ;");"""
        r""" hacc.clear(); };
      const int KT = K / BK;
      for (int kt = 0; kt < KT; ++kt)
        """ + q_map["for (int kk = 0; kk < KK; ++kk)"] + r"""
        {
          hacc.push_back(kt * BK + kk * 16);   // mma_f16(hacc[i][j],"""
        r""" af[cur][i], bf[cur][j]);  (af[cur]: k16 step kk)
          """ + q_map["if (kk & 1) flush();     // every 32 of K"] + r"""
        }
      printf("\n"); }
  }
  // route
  struct Case { int ma, mi, ba; int64_t M, K, N; bool f; };
  Case cases[] = {"""
        r"""{8,6,1,2048,5120,1024,false},{8,6,1,2048,5120,6144,true},{8,6,1,2048,5120,10240,true},{8,6,1,2048,5120,12288,false}"""
        r""",
    {8,6,1,1792,5120,17408,false},{8,6,1,1024,6144,5120,true},{8,6,1,4096,17408,5120,true},{8,6,1,2048,25600,5120,false}"""
        r""",
    {8,6,1,1023,5120,17408,false},{8,6,1,247,5120,17408,false},{8,6,2,2048,5120,17408,false},{8,9,1,2048,5120,17408,false}"""
        r""",
    {12,0,1,2048,5120,17408,false},{8,0,1,2048,5120,17408,false},{8,6,1,2048,5120,17408,true},{8,6,1,2048,5120,1024,true}"""
        r""",
    {8,6,1,2048,4096,4096,false},{8,6,1,2048,5120,17536,false},{8,6,1,2048,5248,5120,true},{8,6,1,2048,25600,5120,true}}"""
        r""";
  const char* envs[4] = {nullptr, "1", "0", "x"};
  const char* names[4] = {"unset", "1", "0", "x"};
  for (int e = 0; e < 4; ++e) for (auto& c : cases)
    printf("Q %s %d %d %d %lld %lld %lld %d %s\n", names[e], c.ma, c.mi, c.ba,"""
        r""" (long long) c.M, (long long) c.K, (long long) c.N,
           c.f ? 1 : 0, route_of(envs[e], c.ma, c.mi, c.ba, c.M, c.K, c.N,"""
        r""" c.f).c_str());
  // launch grid rows
  for (int M : {1, 127, 128, 129, 1100, 2048, 262143}) printf("M %d %d\n", M,"""
        r""" (M + Wide::BM - 1) / Wide::BM);
  return 0;
}
"""
    )


def build_and_run(prog: str) -> tuple[str, str]:
    """Compile and run the C++ reference and the Bend table.

    Args:
        prog: The C++ reference program.

    Returns:
        The stdout of the reference and of the Bend table.

    """
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "hw_ref.cpp"
        src.write_text(prog)
        exe = Path(td) / "hw_ref"
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
            source_link.locked(["c++", "-std=c++17", "-O1", "-o", str(exe), str(src)]),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode:
            sys.stdout.write(
                "hgemm_wide_diff: FAIL: reference program does not compile:\n"
                + r.stderr[-TAIL:]
                + "\n"
            )
            sys.exit(1)
        ref = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(exe)], capture_output=True, text=True, check=True
        ).stdout
        tb = Path(td) / "hw_table"
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
            source_link.locked([
                source_link.bend(),
                str(HERE / "HGEMM_WIDE_TABLE.bend"),
                "-o",
                str(tb),
            ]),
            capture_output=True,
            text=True,
            cwd=HERE,
            check=False,
        )
        if r.returncode:
            sys.stdout.write(
                "hgemm_wide_diff: FAIL: bend table build failed:\n"
                + (r.stdout + r.stderr)[-TAIL:]
                + "\n"
            )
            sys.exit(1)
        got = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(tb)], capture_output=True, text=True, check=True
        ).stdout
    return ref, got


def compare(ref: str, got: str) -> None:
    """Exit 1 unless the two tables are identical; print the summary.

    Args:
        ref: The C++ reference table.
        got: The Bend table.

    """
    a, b = ref.strip().splitlines(), got.strip().splitlines()
    if a != b:
        for k, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                sys.stdout.write(
                    f"hgemm_wide_diff: MISMATCH at line {k + 1}:\n"
                    f"  C++  : {x}\n  Bend : {y}\n"
                )
                break
        else:
            sys.stdout.write(
                f"hgemm_wide_diff: MISMATCH: {len(a)} C++ lines vs "
                f"{len(b)} Bend lines\n"
            )
        sys.exit(1)
    kinds: dict[str, int] = {}
    for line in a:
        kinds[line.split()[0]] = kinds.get(line.split()[0], 0) + 1
    sys.stdout.write(f"hgemm_wide_diff: {len(a)} lines identical {kinds}\n")
    sys.stdout.write("hgemm_wide_diff: OK\n")


def main() -> None:
    """Run the three steps on the EXT_DIR argument."""
    if len(sys.argv) != ARGC:
        sys.exit(USAGE)
    ext = Path(sys.argv[1])
    hdr = ext / "hgemm_f16acc_wide.cuh"
    cu = ext / "hgemm_f16acc.cu"
    hl, cl = lines_of(hdr), lines_of(cu)
    need(HDR_QUOTES + CFG_QUOTES, hl, hdr.name)
    need(CU_QUOTES, cl, cu.name)
    cutext = cu.read_text()
    route_src = extract(cutext, "struct WideShape", "static_assert(Wide::BN == 128")
    hdrtext = hdr.read_text()
    off_src = extract(
        hdrtext,
        "template <int A_CPR, int BK> __device__ __forceinline__ int a_off",
        "template <bool OUT_F32, class CF>",
    )
    cfg_src = extract(
        hdrtext, "template <int BM_, int BN_, int BK_", "// Swizzled element offsets"
    )
    ref, got = build_and_run(program(cfg_src, off_src, route_src))
    compare(ref, got)


if __name__ == "__main__":
    main()
