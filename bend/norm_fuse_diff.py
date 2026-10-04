#!/usr/bin/env python3
"""
Finite differential check of bend/norm_fuse.bend (fused residual add + RMSNorm, ext patch 7001)
and its unfused reference bend/norm_fuse_spec.bend against the pinned norm.cu kernels.

The pristine engine package is copied and the pinned patches/exl3 and patches/exl3-ext series are
applied with the repository's own strict applier, once without and once with 7001 (the post-7001
files must match exl3-ext.json). From each norm.cu the device helpers sum_sq4, apply4, apply4_nw,
reduce_dyn, the RES_* defines, rms_norm_kernel and the launcher's `threads` line are extracted
verbatim and compiled for the CPU (clang++) under a small CUDA shim: one std::thread per CUDA
thread, a block barrier for __syncthreads, a warp barrier for __shfl_xor_sync, and `float` is a
hash algebra H (every value is a U32 hash of the operation tree that produced it; add and mul are
commutative, fma commutes in its two factors, nothing is associative). The post-7001
rms_norm_kernel<RES_IN, float, half, half, float> is the fused launch; torch `x += y` followed by
the pre-7001 rms_norm_kernel<RES_NONE, float, half, half, float> is the unfused sequence. Every
vector store (buffer, address, value, thread, program order) is recorded, formatted exactly like
bend/NORM_FUSE_TABLE.bend, and compared byte for byte with the Bend program's output. The C side
also checks that every element of y and of the residual is stored exactly once, by thread
(e / 4) % blockDim.x of block row, for rows = 1..16 at the served dim 5120.

This is differential evidence on finite instances (dim 96 x 2 rows, dim 5120 x 16 rows, fp32
residual), not a proof of equivalence. `--mutate NAME` applies a deliberate kernel mutation that
the check must reject.

Usage: python3 bend/norm_fuse_diff.py [--mutate NAME] PRISTINE_ENGINE_ROOT
  PRISTINE_ENGINE_ROOT: OUT/stock of bend/engine_trees.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
TABLE = "bend/NORM_FUSE_TABLE.bend"
PATCH = "7001-norm-residual-fuse.patch"
PINNED = ("exllamav3_ext/norm.cu", "modules/transformer.py", "model/model.py", "model/model_ls.py")
SHAPES = ((96, 2), (5120, 16))          # NORM_FUSE_TABLE.bend main
SERVED_DIM = 5120
SERVED_ROWS = range(1, 17)

MUTATIONS = {
    # swap the two accumulation steps of every thread: the sum loop runs i = 1 before i = 0
    # (same columns, same registers, reversed fma chain)
    "swap_reduction_steps": (
        "        float sum = 0.0f;\n        #pragma unroll\n        for (int i = 0; i < REG_COLS; ++i)\n",
        "        float sum = 0.0f;\n        #pragma unroll\n        for (int i = REG_COLS - 1; i >= 0; --i)\n",
    ),
    # off-by-one on the output loop's column guard: the last column is never stored
    "output_guard_off_by_one": (
        "            if (column < columns)\n                apply_out(x4[i], w4[i], column, rmf);",
        "            if (column + 1 < columns)\n                apply_out(x4[i], w4[i], column, rmf);",
    ),
}


def fail(msg: str):
    raise SystemExit(f"norm_fuse_diff: {msg}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trees(pristine: Path, work: Path) -> tuple[Path, Path]:
    """Pre-7001 and post-7001 engine trees, applied with the repository's pinned applier."""
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("ext", REPO / "patches/exl3-ext/ext.py")
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    from patches.exl3.apply import apply_file, parse_patch

    manifest = json.loads((REPO / "patches/exl3-ext/exl3-ext.json").read_bytes())
    series = []
    for line in (REPO / "patches/exl3-ext/series").read_text().splitlines():
        digest, name = line.split()
        if sha(REPO / "patches/exl3-ext" / name) != digest:
            fail(f"patch hash mismatch: {name}")
        series.append(name)
    if PATCH not in series:
        fail(f"{PATCH} is not in the series")
    at = series.index(PATCH)
    trees = []
    for label, names in (("pre", series[:at]), ("post", series[: at + 1])):
        root = ext.engine_copy(pristine, work / label)
        for name in names:
            text = (REPO / "patches/exl3-ext" / name).read_text(encoding="utf-8")
            for relative, creates, hunks in parse_patch(text):
                apply_file(root / relative, creates=creates, hunks=hunks)
        trees.append(root)
    pre, post = trees
    # A file no later patch touches keeps its post-7001 image as the series' pinned post-image
    later = set()
    for name in series[at + 1 :]:
        text = (REPO / "patches/exl3-ext" / name).read_text(encoding="utf-8")
        later |= {relative for relative, _, _ in parse_patch(text)}
    for relative in PINNED:
        if relative in later:
            print(f"note: {relative} is patched again after {PATCH}; its final pin does not apply")
        elif sha(post / relative) != manifest["files"][relative]["post"]:
            fail(f"post-7001 {relative} differs from exl3-ext.json")
    return pre, post


def block(text: str, marker: str) -> tuple[str, int, int]:
    """Verbatim text from the line holding marker through the brace that closes its body."""
    start = text.find(marker)
    if start < 0 or text.find(marker, start + 1) >= 0:
        fail(f"marker not unique: {marker!r}")
    start = text.rfind("\n", 0, start) + 1
    open_at = text.find("{", text.find(")", start))
    depth, i = 0, open_at
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    end = i + 1
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, end) + 1
    return text[start:end], first, last


def line(text: str, marker: str) -> tuple[str, int]:
    start = text.find(marker)
    if start < 0 or text.find(marker, start + 1) >= 0:
        fail(f"marker not unique: {marker!r}")
    start = text.rfind("\n", 0, start) + 1
    end = text.find("\n", start)
    return text[start:end], text.count("\n", 0, start) + 1


def extract(norm: Path) -> dict[str, tuple[str, int, int]]:
    text = norm.read_text(encoding="utf-8")
    out = {}
    out["sum_sq4"] = block(text, "__device__ inline float sum_sq4(")
    out["apply4"] = block(text, "__device__ inline void apply4(")
    out["apply4_nw"] = block(text, "__device__ inline void apply4_nw(")
    out["reduce_dyn"] = block(text, "__device__ inline float reduce_dyn(")
    kernel, first, last = block(text, "template <int res_mode, typename input_t, typename output_t,")
    if "void rms_norm_kernel" not in kernel.split("{", 1)[0]:
        fail("kernel extraction did not land on rms_norm_kernel")
    out["kernel"] = (kernel, first, last)
    defs = []
    for name in ("RES_NONE", "RES_POST", "RES_IN"):
        src, no = line(text, f"#define {name} ")
        defs.append(src)
    out["defines"] = ("\n".join(defs), no - 2, no)
    threads, no = line(text, "int threads = MIN(NUM_THREADS, ")
    out["threads"] = (threads, no, no)
    nt, no = line(text, "#define NUM_THREADS ")
    out["num_threads"] = (nt, no, no)
    return out


SHIM = r"""
#include <algorithm>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

// ---- hash algebra (identical to NORM_FUSE_TABLE.bend) ----
static inline uint32_t f15(uint32_t z) { return z ^ (z >> 15); }
static inline uint32_t st(uint32_t h, uint32_t v) { return f15((h ^ v) * 2654435761u + 1013904223u); }
static inline uint32_t n1(uint32_t t, uint32_t a) { return st(t, a); }
static inline uint32_t n2(uint32_t t, uint32_t a, uint32_t b) { return st(st(t, a), b); }
static inline uint32_t n3(uint32_t t, uint32_t a, uint32_t b, uint32_t c) { return st(st(st(t, a), b), c); }
static inline uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct H
{
    uint32_t v;
    H() : v(0xA5A5A5A5u) {}
    H(float f) : v(n1(4, fbits(f))) {}
    H(double f) : v(n1(4, fbits((float) f))) {}
    H(int i) : v(n1(4, fbits((float) i))) {}
    static H raw(uint32_t x) { H h; h.v = x; return h; }
};
static inline H operator+(H a, H b) { return H::raw(n2(1, std::min(a.v, b.v), std::max(a.v, b.v))); }
static inline H operator*(H a, H b) { return H::raw(n2(2, std::min(a.v, b.v), std::max(a.v, b.v))); }
static inline H operator/(H a, H b) { return H::raw(n2(5, a.v, b.v)); }
static inline H& operator+=(H& a, H b) { a = a + b; return a; }
static inline H& operator*=(H& a, H b) { a = a * b; return a; }
static inline bool operator!=(H a, H b) { return a.v != b.v; }
static inline bool operator==(H a, H b) { return a.v == b.v; }
static inline H fma(H a, H b, H c) { return H::raw(n3(3, std::min(a.v, b.v), std::max(a.v, b.v), c.v)); }
static inline H rsqrtf(H a) { return H::raw(n1(6, a.v)); }
static inline H hcast(H a) { return H::raw(n1(7, a.v)); }

struct float4 { H x, y, z, w; };
struct half { H v; };
struct half2 { half x, y; };
struct half4 { half2 x, y; half4() {} half4(half2 a, half2 b) : x(a), y(b) {} };
struct bfloat16 { H v; };
struct bfloat162 { bfloat16 x, y; };
struct bfloat164 { bfloat162 x, y; };
static inline half __float2half_rn(H f) { return half{hcast(f)}; }
static inline half2 __halves2half2(half a, half b) { return half2{a, b}; }
#define LOW_TO_FLOAT(h2) ((h2).x.v)
#define HIGH_TO_FLOAT(h2) ((h2).y.v)

// ---- CUDA execution shim ----
struct Dim { unsigned x; };
static thread_local Dim threadIdx, blockIdx;
static Dim blockDim;
static std::barrier<>* block_bar;
static std::barrier<>* warp_bar[32];
static H warp_slot[32][32];
static inline void __syncthreads() { block_bar->arrive_and_wait(); }
static inline H __shfl_xor_sync(unsigned, H v, int off)
{
    unsigned t = threadIdx.x, w = t / 32, l = t % 32;
    warp_slot[w][l] = v;
    warp_bar[w]->arrive_and_wait();
    H r = warp_slot[w][l ^ (unsigned) off];
    warp_bar[w]->arrive_and_wait();
    return r;
}
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(x)
#define __restrict__
#define __shared__ static
#define NUM_THREADS_PLACEHOLDER

// ---- memory with store recording ----
struct Rec { int buf; long elem; H v[4]; unsigned thread; long seq; };
struct Buf { const void* base; long elems; int id; };
static std::vector<Buf> bufs;
static std::mutex rec_mu;
static std::vector<Rec> recs;
static thread_local long seq_no;
static void record(int kind, const void* addr, const float4& f4, bool halfs)
{
    for (auto& b : bufs)
    {
        const char* lo = (const char*) b.base;
        long unit = halfs ? (long) sizeof(half) : (long) sizeof(H);
        if ((const char*) addr >= lo && (const char*) addr < lo + b.elems * unit)
        {
            Rec r{b.id, ((const char*) addr - lo) / unit, {f4.x, f4.y, f4.z, f4.w}, threadIdx.x, seq_no++};
            std::lock_guard<std::mutex> g(rec_mu);
            recs.push_back(r);
            return;
        }
    }
    std::fprintf(stderr, "store outside every buffer\n");
    std::abort();
}
static inline void read_float4(float4& f4, const float4* addr) { f4 = *addr; }
template <bool dummy> static inline void read_half4(float4& f4, const half4* addr)
{
    f4.x = addr->x.x.v; f4.y = addr->x.y.v; f4.z = addr->y.x.v; f4.w = addr->y.y.v;
}
static inline void read_bfloat164(float4& f4, const bfloat164* addr)
{
    f4.x = addr->x.x.v; f4.y = addr->x.y.v; f4.z = addr->y.x.v; f4.w = addr->y.y.v;
}
static inline void write_float4(const float4& f4, float4* addr) { record(0, addr, f4, false); *addr = f4; }
static inline void write_half4(const float4& f4, half4* addr)
{
    float4 c{hcast(f4.x), hcast(f4.y), hcast(f4.z), hcast(f4.w)};
    record(0, addr, c, true);
    addr->x.x.v = c.x; addr->x.y.v = c.y; addr->y.x.v = c.z; addr->y.y.v = c.w;
}
#define WRITE64(addr, h4) do { float4 c_{(h4).x.x.v, (h4).x.y.v, (h4).y.x.v, (h4).y.y.v}; \
    record(0, (addr), c_, true); *(addr) = (h4); } while (0)
#define WRITE128(addr, f4) write_float4((f4), (addr))
"""

DRIVER = r"""
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEIL_DIVIDE(a, b) (((a) + (b) - 1) / (b))
static int launch_threads(int dim)
{
@@THREADS@@
    return threads;
}

template <typename K>
static void launch(K kernel, int rows, int threads)
{
    blockDim.x = threads;
    for (int row = 0; row < rows; ++row)
    {
        std::barrier<> bb(threads);
        block_bar = &bb;
        std::vector<std::unique_ptr<std::barrier<>>> wb;
        for (int w = 0; w < threads / 32; ++w) { wb.emplace_back(new std::barrier<>(32)); warp_bar[w] = wb.back().get(); }
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t, row] { threadIdx.x = t; blockIdx.x = row; seq_no = 0; kernel(); });
        for (auto& th : ts) th.join();
    }
}

static H leaf(uint32_t tag, uint32_t row, uint32_t e) { return H::raw(n2(tag, row, e)); }

struct Run { std::vector<Rec> r, y, u; std::vector<H> torch; };

static Run run(int dim, int rows)
{
    int threads = launch_threads(dim);
    long n = (long) rows * dim;
    std::vector<H> x(n), r(n), rt(n);
    std::vector<half> w(dim), y(n), yu(n);
    for (int row = 0; row < rows; ++row)
        for (int e = 0; e < dim; ++e)
        {
            x[(long) row * dim + e] = leaf(8, row, e);
            r[(long) row * dim + e] = leaf(9, row, e);
        }
    for (int e = 0; e < dim; ++e) w[e].v = H::raw(n1(10, e));
    H eps(1e-6f), bias(1.0f), scale(1.0f);
    Run out;

    // fused: post-7001 rms_norm_kernel<RES_IN, float, half, half, float>, x = MLP output, r = stream
    bufs = {{r.data(), n, 1}, {y.data(), n, 2}};
    recs.clear();
    launch([&] { post7001::rms_norm_kernel<RES_IN, H, half, half, H>(x.data(), w.data(), y.data(), r.data(), eps, rows, dim, bias, scale, 1); }, rows, threads);
    for (auto& q : recs) (q.buf == 1 ? out.r : out.y).push_back(q);

    // unfused: torch `x += y` (residual r + MLP output x, stored in fp32), then the pre-7001
    // rms_norm_kernel<RES_NONE, float, half, half, float> over the updated stream
    for (long i = 0; i < n; ++i) rt[i] = leaf(9, i / dim, i % dim) + leaf(8, i / dim, i % dim);
    bufs = {{yu.data(), n, 3}};
    recs.clear();
    launch([&] { pre7001::rms_norm_kernel<RES_NONE, H, half, half, H>(rt.data(), w.data(), yu.data(), (H*) nullptr, eps, rows, dim, bias, scale, 1); }, rows, threads);
    out.u = recs;
    out.torch = rt;
    return out;
}

static void sort_by_thread(std::vector<Rec>& v)
{
    std::stable_sort(v.begin(), v.end(), [](const Rec& a, const Rec& b) {
        long ra = a.elem, rb = b.elem;
        (void) ra; (void) rb;
        return a.thread != b.thread ? a.thread < b.thread : a.seq < b.seq; });
}

static void print_stores(const char* tag, std::vector<Rec> v, int dim, int row)
{
    std::vector<Rec> mine;
    for (auto& q : v) if (q.elem / dim == row) mine.push_back(q);
    sort_by_thread(mine);
    for (auto& q : mine)
        std::printf("%s row=%ld t=%u c=%ld %u %u %u %u\n", tag, q.elem / dim, q.thread, (q.elem % dim) / 4,
            q.v[0].v, q.v[1].v, q.v[2].v, q.v[3].v);
}

// Exactly once, by thread (e / 4) % blockDim.x; every store lands in its own block's row.
static long ownership(const std::vector<Rec>& v, int dim, int rows, int threads)
{
    std::vector<int> count((long) rows * dim, 0);
    long bad = 0;
    for (auto& q : v)
        for (int l = 0; l < 4; ++l)
        {
            long e = q.elem + l;
            count[e]++;
            if ((unsigned) (((e % dim) / 4) % threads) != q.thread) bad++;
        }
    for (int c : count) if (c != 1) bad++;
    return bad;
}

int main(int argc, char** argv)
{
    std::string mode = argc > 1 ? argv[1] : "table";
    if (mode == "table")
    {
        for (auto [dim, rows] : {@@SHAPES@@})
        {
            int threads = launch_threads(dim);
            Run o = run(dim, rows);
            std::printf("shape dim=%d rows=%d W=%d\n", dim, rows, threads / 32);
            for (int row = 0; row < rows; ++row)
            {
                print_stores("R", o.r, dim, row);
                print_stores("Y", o.y, dim, row);
                for (int c = 0; c < dim / 4; ++c)
                {
                    long b = (long) row * dim + 4 * c;
                    std::printf("T row=%d c=%d %u %u %u %u\n", row, c, o.torch[b].v, o.torch[b + 1].v, o.torch[b + 2].v, o.torch[b + 3].v);
                }
                print_stores("U", o.u, dim, row);
                std::printf("\n");
            }
        }
        return 0;
    }
    // ownership: served dim, rows 1..16
    int dim = @@SERVED_DIM@@;
    long total_bad = 0;
    for (int rows = 1; rows <= 16; ++rows)
    {
        int threads = launch_threads(dim);
        Run o = run(dim, rows);
        long br = ownership(o.r, dim, rows, threads), by = ownership(o.y, dim, rows, threads), bu = ownership(o.u, dim, rows, threads);
        std::printf("rows=%d threads=%d residual_bad=%ld y_bad=%ld unfused_y_bad=%ld stores=%zu/%zu/%zu\n", rows, threads, br, by, bu, o.r.size(), o.y.size(), o.u.size());
        total_bad += br + by + bu;
    }
    std::printf("ownership_bad_total=%ld\n", total_bad);
    return total_bad ? 1 : 0;
}
"""


def harness(pre: dict, post: dict) -> str:
    def ns(name: str, parts: dict) -> str:
        return (
            f"namespace {name} {{\n#define float H\n"
            + "\n\n".join(parts[k][0] for k in ("sum_sq4", "apply4", "apply4_nw", "reduce_dyn", "kernel"))
            + "\n#undef float\n}\n"
        )
    defines = post["defines"][0]
    if defines != pre["defines"][0]:
        fail("RES_* defines differ pre/post")
    shapes = ", ".join(f"std::pair{{{d}, {r}}}" for d, r in SHAPES)
    threads = post["threads"][0]
    if threads != pre["threads"][0]:
        fail("launcher threads line differs pre/post")
    nt = post["num_threads"][0]
    return (
        SHIM + nt + "\n" + defines + "\n" + ns("pre7001", pre) + ns("post7001", post)
        + DRIVER.replace("@@THREADS@@", threads).replace("@@SHAPES@@", shapes)
        .replace("@@SERVED_DIM@@", str(SERVED_DIM))
    )


def main(argv: list[str]) -> None:
    mutate = None
    args = argv[1:]
    if args[:1] == ["--mutate"]:
        if len(args) < 2 or args[1] not in MUTATIONS:
            fail(f"--mutate needs one of {sorted(MUTATIONS)}")
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail(__doc__)
    pristine = Path(args[0])
    with tempfile.TemporaryDirectory(prefix="norm-fuse-diff-") as scratch:
        work = Path(scratch)
        pre_tree, post_tree = build_trees(pristine, work)
        pre = extract(pre_tree / "exllamav3_ext/norm.cu")
        post = extract(post_tree / "exllamav3_ext/norm.cu")
        for k in ("sum_sq4", "apply4", "apply4_nw", "reduce_dyn"):
            if pre[k][0] != post[k][0]:
                fail(f"{k} differs pre/post 7001")
        print("pinned: post-7001 norm.cu", sha(post_tree / "exllamav3_ext/norm.cu"),
              "pre-7001 norm.cu", sha(pre_tree / "exllamav3_ext/norm.cu"))
        for k, (_, a, b) in post.items():
            print(f"verbatim post norm.cu:{a}-{b} {k}")
        print(f"verbatim pre norm.cu:{pre['kernel'][1]}-{pre['kernel'][2]} kernel (unfused)")
        print("byte-identical pre/post: sum_sq4 apply4 apply4_nw reduce_dyn, RES_* defines, launcher threads line")
        if mutate:
            old, new = MUTATIONS[mutate]
            kernel = post["kernel"][0]
            if kernel.count(old) != 1:
                fail(f"mutation {mutate} does not apply")
            post["kernel"] = (kernel.replace(old, new), post["kernel"][1], post["kernel"][2])
            print(f"MUTATION {mutate} applied to the post-7001 kernel")
        src = work / "norm_fuse_harness.cpp"
        src.write_text(harness(pre, post))
        exe = work / "harness"
        r = subprocess.run(source_link.locked(["clang++", "-std=c++20", "-O1", "-pthread", "-w", str(src), "-o", str(exe)]),
                           capture_output=True, text=True)
        if r.returncode:
            fail("harness compile failed:\n" + r.stderr[-4000:])
        c_table = subprocess.run([str(exe), "table"], capture_output=True, text=True, check=True).stdout
        own = subprocess.run([str(exe), "ownership"], capture_output=True, text=True)
        exe_b = work / "table_bin"
        r = subprocess.run(source_link.locked([source_link.bend(), str(REPO / TABLE), "-o", str(exe_b)]),
                           capture_output=True, text=True,
                           cwd=REPO)
        if r.returncode:
            fail("Bend table compile failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        b_table = subprocess.run([str(exe_b)], capture_output=True, text=True, check=True).stdout
    print(own.stdout.strip())
    lines_c, lines_b = c_table.splitlines(), b_table.splitlines()
    ok_own = own.returncode == 0
    same = c_table == b_table
    fused_eq_unfused = all(
        a.split(" ", 1)[1] == b.split(" ", 1)[1]
        for a, b in zip([l for l in lines_c if l.startswith("Y ")], [l for l in lines_c if l.startswith("U ")])
    )
    print(f"table lines: C {len(lines_c)}, Bend {len(lines_b)}; sha256 C {hashlib.sha256(c_table.encode()).hexdigest()}"
          f" Bend {hashlib.sha256(b_table.encode()).hexdigest()}")
    print(f"C fused y == C unfused y (thread, column, hash) on every store: {fused_eq_unfused}")
    if not same:
        for i, (a, b) in enumerate(zip(lines_c, lines_b)):
            if a != b:
                print(f"first difference at line {i + 1}:\n  C:    {a}\n  Bend: {b}")
                break
        else:
            print("tables differ in length")
    verdict = same and ok_own and fused_eq_unfused
    print("RESULT:", "MATCH" if verdict else "MISMATCH")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main(sys.argv)
