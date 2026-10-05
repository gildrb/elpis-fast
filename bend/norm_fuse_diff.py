# Copyright (c) 2026 Gil Rodrigues
"""Check bend/norm_fuse.bend against the pinned norm.cu kernels on finite instances.

Finite differential check of the fused residual add + RMSNorm (ext patch 7001) and
its unfused reference bend/norm_fuse_spec.bend: the verbatim pre/post-7001 norm.cu
kernels are compiled for the CPU under a hash-algebra CUDA shim and their store
tables are compared byte for byte with bend/NORM_FUSE_TABLE.bend. See USAGE for the
full description.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

if TYPE_CHECKING:
    from types import ModuleType

HERE = Path(__file__).resolve().parent
REPO = source_link.REPO
TABLE = "bend/NORM_FUSE_TABLE.bend"
PATCH = "7001-norm-residual-fuse.patch"
PINNED = (
    "exllamav3_ext/norm.cu",
    "modules/transformer.py",
    "model/model.py",
    "model/model_ls.py",
)
SHAPES = ((96, 2), (5120, 16))  # NORM_FUSE_TABLE.bend main
SERVED_DIM = 5120
SERVED_ROWS = range(1, 17)
# `--mutate NAME` occupies two arguments
MUTATE_ARGS = 2
HELPERS = ("sum_sq4", "apply4", "apply4_nw", "reduce_dyn")
# The usage message: the module docstring of the pre-lint revision, verbatim.
USAGE = (
    "\n"
    "Finite differential check of bend/norm_fuse.bend (fused residual add + RMSNorm, "
    "ext patch 7001)\n"
    "and its unfused reference bend/norm_fuse_spec.bend against the pinned norm.cu "
    "kernels.\n"
    "\n"
    "The pristine engine package is copied and the pinned patches/exl3 and "
    "patches/exl3-ext series are\n"
    "applied with the repository's own strict applier, once without and once with 7001 "
    "(the post-7001\n"
    "files must match exl3-ext.json). From each norm.cu the device helpers sum_sq4, "
    "apply4, apply4_nw,\n"
    "reduce_dyn, the RES_* defines, rms_norm_kernel and the launcher's `threads` line "
    "are extracted\n"
    "verbatim and compiled for the CPU (clang++) under a small CUDA shim: one "
    "std::thread per CUDA\n"
    "thread, a block barrier for __syncthreads, a warp barrier for __shfl_xor_sync, "
    "and `float` is a\n"
    "hash algebra H (every value is a U32 hash of the operation tree that produced it; "
    "add and mul are\n"
    "commutative, fma commutes in its two factors, nothing is associative). The "
    "post-7001\n"
    "rms_norm_kernel<RES_IN, float, half, half, float> is the fused launch; torch `x "
    "+= y` followed by\n"
    "the pre-7001 rms_norm_kernel<RES_NONE, float, half, half, float> is the unfused "
    "sequence. Every\n"
    "vector store (buffer, address, value, thread, program order) is recorded, "
    "formatted exactly like\n"
    "bend/NORM_FUSE_TABLE.bend, and compared byte for byte with the Bend program's "
    "output. The C side\n"
    "also checks that every element of y and of the residual is stored exactly once, "
    "by thread\n"
    "(e / 4) % blockDim.x of block row, for rows = 1..16 at the served dim 5120.\n"
    "\n"
    "This is differential evidence on finite instances (dim 96 x 2 rows, dim 5120 x 16 "
    "rows, fp32\n"
    "residual), not a proof of equivalence. `--mutate NAME` applies a deliberate "
    "kernel mutation that\n"
    "the check must reject.\n"
    "\n"
    "Usage: python3 bend/norm_fuse_diff.py [--mutate NAME] PRISTINE_ENGINE_ROOT\n"
    "  PRISTINE_ENGINE_ROOT: OUT/stock of bend/engine_trees.py.\n"
)

MUTATIONS = {
    # swap the two accumulation steps of every thread: the sum loop runs i = 1
    # before i = 0 (same columns, same registers, reversed fma chain)
    "swap_reduction_steps": (
        (
            "        float sum = 0.0f;\n        #pragma unroll\n"
            "        for (int i = 0; i < REG_COLS; ++i)\n"
        ),
        (
            "        float sum = 0.0f;\n        #pragma unroll\n"
            "        for (int i = REG_COLS - 1; i >= 0; --i)\n"
        ),
    ),
    # off-by-one on the output loop's column guard: the last column is never stored
    "output_guard_off_by_one": (
        (
            "            if (column < columns)\n"
            "                apply_out(x4[i], w4[i], column, rmf);"
        ),
        (
            "            if (column + 1 < columns)\n"
            "                apply_out(x4[i], w4[i], column, rmf);"
        ),
    ),
}

Parts = dict[str, tuple[str, int, int]]


def fail(msg: str) -> NoReturn:
    """Exit with a norm_fuse_diff error message.

    Args:
        msg: The error text.

    Raises:
        SystemExit: Always.

    """
    text = f"norm_fuse_diff: {msg}"
    raise SystemExit(text)


def say(*parts: object) -> None:
    """Write parts to stdout separated by spaces, with a trailing newline.

    Args:
        *parts: The values to write.

    """
    sys.stdout.write(" ".join(map(str, parts)) + "\n")


def sha(path: Path) -> str:
    """Hash a file.

    Args:
        path: The file to hash.

    Returns:
        The sha256 hex digest of the file's bytes.

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ext() -> ModuleType:
    """Import patches/exl3-ext/ext.py, the repository's pinned patch applier.

    Its import also loads patches/exl3/apply.py, whose apply_file and parse_patch it
    re-exports.

    Returns:
        The ext module.

    """
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "ext", REPO / "patches/exl3-ext/ext.py"
    )
    if spec is None or spec.loader is None:
        fail("cannot load patches/exl3-ext/ext.py")
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext


def patch_text(name: str) -> str:
    """Read a patch of the exl3-ext series.

    Args:
        name: The patch file name.

    Returns:
        The patch text.

    """
    return (REPO / "patches/exl3-ext" / name).read_text(encoding="utf-8")


def read_series() -> list[str]:
    """Read patches/exl3-ext/series, checking every patch hash.

    Returns:
        The patch names in series order.

    """
    series = []
    for entry in (REPO / "patches/exl3-ext/series").read_text().splitlines():
        digest, name = entry.split()
        if sha(REPO / "patches/exl3-ext" / name) != digest:
            fail(f"patch hash mismatch: {name}")
        series.append(name)
    return series


def build_tree(
    ext: ModuleType, pristine: Path, destination: Path, names: list[str]
) -> Path:
    """Copy the pristine engine and apply the exl3 series plus the named patches.

    Args:
        ext: The loaded patches/exl3-ext/ext.py module.
        pristine: The pristine engine package directory.
        destination: Where the tree is created.
        names: The exl3-ext patches to apply, in order.

    Returns:
        The patched package directory.

    """
    root = ext.engine_copy(pristine, destination)
    if not isinstance(root, Path):
        fail("ext.engine_copy did not return a path")
    for name in names:
        for relative, creates, hunks in ext.parse_patch(patch_text(name)):
            ext.apply_file(root / relative, creates=creates, hunks=hunks)
    return root


def check_pins(post: Path, later: set[str], manifest: dict[str, object]) -> None:
    """Check the PINNED files of the post-7001 tree against exl3-ext.json.

    A file no later patch touches keeps its post-7001 image as the series' pinned
    post-image.

    Args:
        post: The post-7001 package directory.
        later: The files patched again after PATCH.
        manifest: The parsed exl3-ext.json.

    """
    for relative in PINNED:
        if relative in later:
            say(
                f"note: {relative} is patched again after {PATCH}; "
                "its final pin does not apply"
            )
            continue
        digest = sha(post / relative)
        files = manifest["files"]
        if not isinstance(files, dict):
            fail("exl3-ext.json has no files table")
        if digest != files[relative]["post"]:
            fail(f"post-7001 {relative} differs from exl3-ext.json")


def build_trees(pristine: Path, work: Path) -> tuple[Path, Path]:
    """Build the pre-7001 and post-7001 engine trees with the pinned applier.

    Args:
        pristine: The pristine engine package directory.
        work: The scratch directory the trees are created in.

    Returns:
        The pre-7001 and post-7001 package directories.

    """
    ext = load_ext()
    manifest = json.loads((REPO / "patches/exl3-ext/exl3-ext.json").read_bytes())
    series = read_series()
    if PATCH not in series:
        fail(f"{PATCH} is not in the series")
    at = series.index(PATCH)
    pre = build_tree(ext, pristine, work / "pre", series[:at])
    post = build_tree(ext, pristine, work / "post", series[: at + 1])
    later: set[str] = set()
    for name in series[at + 1 :]:
        later |= {relative for relative, _, _ in ext.parse_patch(patch_text(name))}
    check_pins(post, later, manifest)
    return pre, post


def block(text: str, marker: str) -> tuple[str, int, int]:
    """Cut the text from the line holding marker through the brace closing its body.

    Args:
        text: The source text.
        marker: A string that occurs exactly once in text.

    Returns:
        The verbatim block and its first and last line numbers.

    """
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
    """Cut the line holding marker.

    Args:
        text: The source text.
        marker: A string that occurs exactly once in text.

    Returns:
        The verbatim line (without its newline) and its line number.

    """
    start = text.find(marker)
    if start < 0 or text.find(marker, start + 1) >= 0:
        fail(f"marker not unique: {marker!r}")
    start = text.rfind("\n", 0, start) + 1
    end = text.find("\n", start)
    return text[start:end], text.count("\n", 0, start) + 1


def extract(norm: Path) -> Parts:
    """Extract the linked pieces of a norm.cu verbatim.

    Args:
        norm: The norm.cu file.

    Returns:
        Each piece's text with its first and last line numbers.

    """
    text = norm.read_text(encoding="utf-8")
    out = {}
    out["sum_sq4"] = block(text, "__device__ inline float sum_sq4(")
    out["apply4"] = block(text, "__device__ inline void apply4(")
    out["apply4_nw"] = block(text, "__device__ inline void apply4_nw(")
    out["reduce_dyn"] = block(text, "__device__ inline float reduce_dyn(")
    kernel, first, last = block(
        text, "template <int res_mode, typename input_t, typename output_t,"
    )
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


SHIM = (
    r"""
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
static inline uint32_t st(uint32_t h, uint32_t v) { return f15((h ^ v) * 2654435761u"""
    r""" + 1013904223u); }
static inline uint32_t n1(uint32_t t, uint32_t a) { return st(t, a); }
static inline uint32_t n2(uint32_t t, uint32_t a, uint32_t b) { return st(st(t, a),"""
    r""" b); }
static inline uint32_t n3(uint32_t t, uint32_t a, uint32_t b, uint32_t c) { return"""
    r""" st(st(st(t, a), b), c); }
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
static inline H operator+(H a, H b) { return H::raw(n2(1, std::min(a.v, b.v),"""
    r""" std::max(a.v, b.v))); }
static inline H operator*(H a, H b) { return H::raw(n2(2, std::min(a.v, b.v),"""
    r""" std::max(a.v, b.v))); }
static inline H operator/(H a, H b) { return H::raw(n2(5, a.v, b.v)); }
static inline H& operator+=(H& a, H b) { a = a + b; return a; }
static inline H& operator*=(H& a, H b) { a = a * b; return a; }
static inline bool operator!=(H a, H b) { return a.v != b.v; }
static inline bool operator==(H a, H b) { return a.v == b.v; }
static inline H fma(H a, H b, H c) { return H::raw(n3(3, std::min(a.v, b.v),"""
    r""" std::max(a.v, b.v), c.v)); }
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
            Rec r{b.id, ((const char*) addr - lo) / unit, {f4.x, f4.y, f4.z, f4.w},"""
    r""" threadIdx.x, seq_no++};
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
static inline void write_float4(const float4& f4, float4* addr) { record(0, addr, f4,"""
    r""" false); *addr = f4; }
static inline void write_half4(const float4& f4, half4* addr)
{
    float4 c{hcast(f4.x), hcast(f4.y), hcast(f4.z), hcast(f4.w)};
    record(0, addr, c, true);
    addr->x.x.v = c.x; addr->x.y.v = c.y; addr->y.x.v = c.z; addr->y.y.v = c.w;
}
#define WRITE64(addr, h4) do { float4 c_{(h4).x.x.v, (h4).x.y.v, (h4).y.x.v,"""
    r""" (h4).y.y.v}; \
    record(0, (addr), c_, true); *(addr) = (h4); } while (0)
#define WRITE128(addr, f4) write_float4((f4), (addr))
"""
)

DRIVER = (
    r"""
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
        for (int w = 0; w < threads / 32; ++w) { wb.emplace_back(new"""
    r""" std::barrier<>(32)); warp_bar[w] = wb.back().get(); }
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t, row] { threadIdx.x = t; blockIdx.x = row; seq_no ="""
    r""" 0; kernel(); });
        for (auto& th : ts) th.join();
    }
}

static H leaf(uint32_t tag, uint32_t row, uint32_t e) { return H::raw(n2(tag, row,"""
    r""" e)); }

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

    // fused: post-7001 rms_norm_kernel<RES_IN, float, half, half, float>, x = MLP"""
    r""" output, r = stream
    bufs = {{r.data(), n, 1}, {y.data(), n, 2}};
    recs.clear();
    launch([&] { post7001::rms_norm_kernel<RES_IN, H, half, half, H>(x.data(),"""
    r""" w.data(), y.data(), r.data(), eps, rows, dim, bias, scale, 1); }, rows,"""
    r""" threads);
    for (auto& q : recs) (q.buf == 1 ? out.r : out.y).push_back(q);

    // unfused: torch `x += y` (residual r + MLP output x, stored in fp32), then the"""
    r""" pre-7001
    // rms_norm_kernel<RES_NONE, float, half, half, float> over the updated stream
    for (long i = 0; i < n; ++i) rt[i] = leaf(9, i / dim, i % dim) + leaf(8, i / dim,"""
    r""" i % dim);
    bufs = {{yu.data(), n, 3}};
    recs.clear();
    launch([&] { pre7001::rms_norm_kernel<RES_NONE, H, half, half, H>(rt.data(),"""
    r""" w.data(), yu.data(), (H*) nullptr, eps, rows, dim, bias, scale, 1); }, rows,"""
    r""" threads);
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
        std::printf("%s row=%ld t=%u c=%ld %u %u %u %u\n", tag, q.elem / dim,"""
    r""" q.thread, (q.elem % dim) / 4,
            q.v[0].v, q.v[1].v, q.v[2].v, q.v[3].v);
}

// Exactly once, by thread (e / 4) % blockDim.x; every store lands in its own block's"""
    r""" row.
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
                    std::printf("T row=%d c=%d %u %u %u %u\n", row, c, o.torch[b].v,"""
    r""" o.torch[b + 1].v, o.torch[b + 2].v, o.torch[b + 3].v);
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
        long br = ownership(o.r, dim, rows, threads), by = ownership(o.y, dim, rows,"""
    r""" threads), bu = ownership(o.u, dim, rows, threads);
        std::printf("rows=%d threads=%d residual_bad=%ld y_bad=%ld unfused_y_bad=%ld"""
    r""" stores=%zu/%zu/%zu\n", rows, threads, br, by, bu, o.r.size(), o.y.size(),"""
    r""" o.u.size());
        total_bad += br + by + bu;
    }
    std::printf("ownership_bad_total=%ld\n", total_bad);
    return total_bad ? 1 : 0;
}
"""
)


def harness(pre: Parts, post: Parts) -> str:
    """Assemble the C++ harness program.

    Args:
        pre: The pieces of the pre-7001 norm.cu.
        post: The pieces of the post-7001 norm.cu (possibly mutated).

    Returns:
        The program text.

    """

    def ns(name: str, parts: Parts) -> str:
        return (
            f"namespace {name} {{\n#define float H\n"
            + "\n\n".join(parts[k][0] for k in (*HELPERS, "kernel"))
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
        SHIM
        + nt
        + "\n"
        + defines
        + "\n"
        + ns("pre7001", pre)
        + ns("post7001", post)
        + DRIVER
        .replace("@@THREADS@@", threads)
        .replace("@@SHAPES@@", shapes)
        .replace("@@SERVED_DIM@@", str(SERVED_DIM))
    )


def parse_args(argv: list[str]) -> tuple[str | None, Path]:
    """Parse the command line.

    Args:
        argv: The full argv, program name first.

    Returns:
        The mutation name (or None) and the pristine engine root.

    """
    mutate = None
    args = argv[1:]
    if args[:1] == ["--mutate"]:
        if len(args) < MUTATE_ARGS or args[1] not in MUTATIONS:
            fail(f"--mutate needs one of {sorted(MUTATIONS)}")
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail(USAGE)
    return mutate, Path(args[0])


def mutated(kernel: tuple[str, int, int], mutate: str) -> tuple[str, int, int]:
    """Apply a MUTATIONS entry to the post-7001 kernel.

    Args:
        kernel: The kernel text with its first and last line numbers.
        mutate: The MUTATIONS key.

    Returns:
        The mutated kernel with the same line numbers.

    """
    old, new = MUTATIONS[mutate]
    text = kernel[0]
    if text.count(old) != 1:
        fail(f"mutation {mutate} does not apply")
    return (text.replace(old, new), kernel[1], kernel[2])


def prepare(pristine: Path, work: Path, mutate: str | None) -> tuple[Parts, Parts]:
    """Build both trees, extract and report the linked pieces, apply the mutation.

    Args:
        pristine: The pristine engine package directory.
        work: The scratch directory.
        mutate: The MUTATIONS key, or None.

    Returns:
        The pre-7001 and post-7001 pieces.

    """
    pre_tree, post_tree = build_trees(pristine, work)
    pre = extract(pre_tree / "exllamav3_ext/norm.cu")
    post = extract(post_tree / "exllamav3_ext/norm.cu")
    for k in HELPERS:
        if pre[k][0] != post[k][0]:
            fail(f"{k} differs pre/post 7001")
    say(
        "pinned: post-7001 norm.cu",
        sha(post_tree / "exllamav3_ext/norm.cu"),
        "pre-7001 norm.cu",
        sha(pre_tree / "exllamav3_ext/norm.cu"),
    )
    for k, (_, a, b) in post.items():
        say(f"verbatim post norm.cu:{a}-{b} {k}")
    say(f"verbatim pre norm.cu:{pre['kernel'][1]}-{pre['kernel'][2]} kernel (unfused)")
    say(
        "byte-identical pre/post: sum_sq4 apply4 apply4_nw reduce_dyn, RES_* defines, "
        "launcher threads line"
    )
    if mutate:
        post["kernel"] = mutated(post["kernel"], mutate)
        say(f"MUTATION {mutate} applied to the post-7001 kernel")
    return pre, post


def compile_harness(work: Path, program: str) -> Path:
    """Compile the harness program for the CPU.

    Args:
        work: The scratch directory.
        program: The harness source.

    Returns:
        The harness executable.

    """
    src = work / "norm_fuse_harness.cpp"
    src.write_text(program)
    exe = work / "harness"
    r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
        source_link.locked([
            "clang++",
            "-std=c++20",
            "-O1",
            "-pthread",
            "-w",
            str(src),
            "-o",
            str(exe),
        ]),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode:
        fail("harness compile failed:\n" + r.stderr[-4000:])
    return exe


def bend_table(work: Path) -> str:
    """Compile and run bend/NORM_FUSE_TABLE.bend.

    Args:
        work: The scratch directory.

    Returns:
        The Bend program's stdout.

    """
    exe_b = work / "table_bin"
    r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
        source_link.locked([
            source_link.bend(),
            str(REPO / TABLE),
            "-o",
            str(exe_b),
        ]),
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    if r.returncode:
        fail("Bend table compile failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
        [str(exe_b)], capture_output=True, text=True, check=True
    ).stdout


def fused_matches_unfused(lines_c: list[str]) -> bool:
    """Compare the fused (Y) and unfused (U) stores of the C table pairwise.

    Args:
        lines_c: The C table lines.

    Returns:
        Whether every paired store has the same thread, column and hash.

    """
    return all(
        a.split(" ", 1)[1] == b.split(" ", 1)[1]
        for a, b in zip(
            [entry for entry in lines_c if entry.startswith("Y ")],
            [entry for entry in lines_c if entry.startswith("U ")],
            strict=False,
        )
    )


def report_difference(lines_c: list[str], lines_b: list[str]) -> None:
    """Report the first differing line of the two tables.

    Args:
        lines_c: The C table lines.
        lines_b: The Bend table lines.

    """
    for i, (a, b) in enumerate(zip(lines_c, lines_b, strict=False)):
        if a != b:
            say(f"first difference at line {i + 1}:\n  C:    {a}\n  Bend: {b}")
            break
    else:
        say("tables differ in length")


def report(c_table: str, b_table: str, own: subprocess.CompletedProcess[str]) -> bool:
    """Print the comparison of the C and Bend tables and the ownership check.

    Args:
        c_table: The harness's table output.
        b_table: The Bend program's output.
        own: The finished harness ownership run.

    Returns:
        Whether the tables match and both checks hold.

    """
    say(own.stdout.strip())
    lines_c, lines_b = c_table.splitlines(), b_table.splitlines()
    ok_own = own.returncode == 0
    same = c_table == b_table
    fused_eq_unfused = fused_matches_unfused(lines_c)
    say(
        f"table lines: C {len(lines_c)}, Bend {len(lines_b)}; "
        f"sha256 C {hashlib.sha256(c_table.encode()).hexdigest()}"
        f" Bend {hashlib.sha256(b_table.encode()).hexdigest()}"
    )
    say(
        "C fused y == C unfused y (thread, column, hash) on every store: "
        f"{fused_eq_unfused}"
    )
    if not same:
        report_difference(lines_c, lines_b)
    return same and ok_own and fused_eq_unfused


def main(argv: list[str]) -> None:
    """Run the differential check and exit 0 on a match, 1 otherwise.

    Args:
        argv: The full argv, program name first.

    """
    mutate, pristine = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="norm-fuse-diff-") as scratch:
        work = Path(scratch)
        pre, post = prepare(pristine, work, mutate)
        exe = compile_harness(work, harness(pre, post))
        c_table = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(exe), "table"], capture_output=True, text=True, check=True
        ).stdout
        own = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(exe), "ownership"], capture_output=True, text=True, check=False
        )
        b_table = bend_table(work)
    verdict = report(c_table, b_table, own)
    say("RESULT:", "MATCH" if verdict else "MISMATCH")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main(sys.argv)
