#!/usr/bin/env python3
"""
Finite source link of bend/pattn8i_int.bend (ext 3021c int8 QK^T integer core) to the patched engine text, plus host C
evidence for the fp32 facts the Bend model takes as hypotheses. Text and test evidence, not a proof.

1. Every kernel line the model transcribes (3-bit code, f, kq, Q clamp, the s32.s8.s8.s32 IMMA asm string, the g < 8 /
   pp < 2 loop with two IMMAs per pp, i2f_exact, the four sc lines, the k_out_scales line, kscale) occurs exactly once
   in exllamav3_ext/pattn8i_kernel.cuh, in the order the model assumes.
2. The constants the model writes structurally match the kernel's: 0x4B400000 == 150 * 2^23 + 2^22, 12582912 ==
   3 * 2^22 (the model's h + (h + h), h = 2^22), the requantisation factor is 18 (|kq| <= 7 * 18 = 126), the Q clamp
   is 127; and the products bend/pattn8i_int_laws.bend writes for the big constants are those values: h =
   Nat.mul(2048n, 2048n) = 2^22, Nat.mul(4096n, 4096n) = 2^24, Nat.mul(32768n, 65536n) = 2^31 (INT32_MAX + 1), the
   partial bound P = Nat.mul(32n, Nat.mul(127n, 126n)) = 32 * 127 * (7 * factor) = 512064 with 8 P < 2^22, and the laws
   use no decimal literal of 2^17 or more (the largest factor is 65536).
3. Host C (the dev shell's c++, -O2 -ffp-contract=off, no fast-math, default round-to-nearest; run via
   source_link.locked), built from the constants extracted from the kernel text:
   - i2f_exact(x) == x for every x in [-2^22, 2^22) (memcpy bit casts, fp32 subtraction); at x = 2^22 the exponent
     field is 151 (the model's precondition, field 150, fails) although the value is still 2^22; the first wrong
     results are at 2^22 + 1 and -2^22 - 1;
   - for every code -7..7 (odd) and every pair of finite fp16 scales 0 <= s <= smax (smax > 0):
     f = RN(RN(18 s) / smax) <= 18, kq = rint(RN(code * f)) has |kq| <= 126, and kq == 18 code when s == smax;
   - the finding instance s = 3, smax = 5, code 7: code * f = 75.6 in exact arithmetic, kq = 76.
   The kq check evaluates f for every pair and then every code on each distinct f bit pattern (kq depends on (code, f)
   only), plus every s == smax pair directly.
`--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/pattn8i_int_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link  # noqa: E402

KERNEL = "exllamav3_ext/pattn8i_kernel.cuh"

# (model def, kernel line), in kernel order
LINES = [
    ("Impl.mma (asm)", '"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"'),
    ("Impl.i2f_exact", "return __fsub_rn(__int_as_float(x + 0x4B400000), 12582912.0f);"),
    ("rounds QRotate", "v[i] = __fadd_rn(a, c); v[i + s] = __fsub_rn(a, c);"),
    ("rounds QRotate", "v[i] = (lane & s) ? __fsub_rn(o, v[i]) : __fadd_rn(v[i], o);"),
    ("rounds QQuant", "const float inv = amax > 0.f ? __fdiv_rn(127.f, amax) : 0.f;"),
    ("Impl.clamp", "for (int i = 0; i < 8; ++i) qi[i] = max(-127, min(127, __float2int_rn(__fmul_rn(v[i], inv))));"),
    ("rounds RowFactor", "if (lane == 0) qsc[row] = __fmul_rn(__fdiv_rn(amax, 127.f), scale_log2);"),
    ("Impl.chain (acc = 0)", "for (int e = 0; e < 4; ++e) ai[nt][e] = 0;"),
    ("Impl.g_loop", "for (int g = 0; g < 8; ++g)"),
    ("Impl.pp_loop", "for (int pp = 0; pp < 2; ++pp)"),
    ("Impl.pp_loop", "imma(ai[2 * pp], a, bk[0], bk[1]);"),
    ("Impl.pp_loop", "imma(ai[2 * pp + 1], a, bk[2], bk[3]);"),
    ("rounds ScaleProd/ScoreMul", "sc[nt][0] = __fmul_rn(i2f_exact(ai[nt][0]), __fmul_rn(rs0, sk.x));"),
    ("rounds ScaleProd/ScoreMul", "sc[nt][1] = __fmul_rn(i2f_exact(ai[nt][1]), __fmul_rn(rs0, sk.y));"),
    ("rounds ScaleProd/ScoreMul", "sc[nt][2] = __fmul_rn(i2f_exact(ai[nt][2]), __fmul_rn(rs1, sk.x));"),
    ("rounds ScaleProd/ScoreMul", "sc[nt][3] = __fmul_rn(i2f_exact(ai[nt][3]), __fmul_rn(rs1, sk.y));"),
    ("rounds KeyScale", "constexpr float kscale = 0.17677669529663688110f / 144.0f;   // r32 / 8 / 18"),
    ("f (s)", "const float s = __half2float(k_in_scales[gi]);"),
    ("f (smax over the 8 groups)", "for (int x = 4; x < 32; x <<= 1) smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, x));"),
    ("Impl.kq_rounds (f)", "const float f = smax > 0.f ? __fdiv_rn(__fmul_rn(18.f, s), smax) : 0.f;"),
    ("Impl.field / Impl.code", "int code = 2 * (int) ((((hi >> (2 * i)) & 3) << 1) | ((lo >> i) & 1)) - 7;"),
    ("Impl.kq_rounds (kq)", "kq[i] = __float2int_rn(__fmul_rn((float) code, f));"),
    ("rounds KeyScale", "if (lane == 0) k_out_scales[(dstp * n_kvh + kvh) * PA_PAGE + tok % PA_PAGE] = __fmul_rn(smax, kscale);"),
]

MUTATIONS = {
    "kq18": ("__fmul_rn(18.f, s)", "__fmul_rn(19.f, s)"),
    "magic": ("x + 0x4B400000", "x + 0x4B000000"),
    "clamp": ("max(-127, min(127,", "max(-128, min(127,"),
}

C_SRC = r"""
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
static_assert(FLT_EVAL_METHOD == 0, "fp32 arithmetic must be evaluated in fp32");
static float as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t as_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float i2f(int32_t x) { return as_float((uint32_t) x + (uint32_t) MAGIC) - (float) SUBC; }
static float h2f(uint16_t h) {   // finite non-negative fp16, exact
    int e = (h >> 10) & 31, m = h & 1023;
    return e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (1024 + m), e - 25);
}
int main() {
    int bad = 0;
    if (std::fegetround() != FE_TONEAREST) { std::printf("rounding mode is not RN\n"); return 1; }
    long n = 0, wrong = 0;
    for (int32_t x = -(1 << 22); x < (1 << 22); ++x, ++n) if (i2f(x) != (float) x || (double) i2f(x) != (double) x) ++wrong;
    std::printf("i2f exact on [-2^22, 2^22): %ld values, %ld wrong\n", n, wrong);
    if (wrong) bad = 1;
    uint32_t u22 = (uint32_t) (1 << 22) + (uint32_t) MAGIC;
    std::printf("x = 2^22: exponent field %u, i2f = %.1f\n", (u22 >> 23) & 255u, (double) i2f(1 << 22));
    std::printf("x = 2^22 + 1: i2f = %.1f; x = -2^22 - 1: i2f = %.1f\n", (double) i2f((1 << 22) + 1), (double) i2f(-(1 << 22) - 1));
    if (((u22 >> 23) & 255u) != 151u || i2f((1 << 22) + 1) == (float) ((1 << 22) + 1) || i2f(-(1 << 22) - 1) == (float) (-(1 << 22) - 1)) bad = 1;
    long pairs = 0, fbad = 0, kbad = 0, ebad = 0, maxk = 0, distinct = 0;
    const uint32_t LIM = 0x42000000u;   // bit patterns of f in [0, 32)
    std::vector<uint8_t> seen(LIM / 8, 0);
    auto check = [&](float f, bool top) {
        for (int c = 0; c < 8; ++c) {
            int code = 2 * c - 7;
            long kq = (long) std::rint((float) code * f);
            long ak = kq < 0 ? -kq : kq;
            if (ak > maxk) maxk = ak;
            if (ak > 126) ++kbad;
            if (top && kq != 18 * code) ++ebad;
        }
    };
    for (uint16_t hs = 1; hs <= 0x7BFF; ++hs) {
        float smax = h2f(hs);
        for (uint16_t ss = 0; ss <= hs; ++ss) {
            float s = h2f(ss);
            float f = smax > 0.f ? (F18 * s) / smax : 0.f;
            ++pairs;
            if (!(f <= 18.f)) ++fbad;
            if (ss == hs) { check(f, true); continue; }
            uint32_t b = as_bits(f);
            if (b >= LIM) { check(f, false); continue; }
            if (!(seen[b >> 3] & (1u << (b & 7)))) { seen[b >> 3] |= (uint8_t) (1u << (b & 7)); ++distinct; check(f, false); }
        }
    }
    std::printf("kq: %ld fp16 pairs 0 <= s <= smax (%ld distinct f) x 8 codes: f > 18 in %ld, |kq| > 126 in %ld, s == smax and kq != 18 code in %ld, max |kq| %ld\n",
                pairs, distinct, fbad, kbad, ebad, maxk);
    if (fbad || kbad || ebad) bad = 1;
    float f35 = (F18 * h2f(0x4200)) / h2f(0x4500);
    long k35 = (long) std::rint(7.f * f35);
    std::printf("finding: s = 3, smax = 5, code 7: f = %.9g (bits 0x%08x), code * f = 75.6 exact, kq = %ld\n", (double) f35, as_bits(f35), k35);
    if (k35 != 76) bad = 1;
    return bad;
}
"""


def fail(msg: str) -> None:
    raise SystemExit(f"pattn8i_int_diff: FAIL: {msg}")


def main(argv: list[str]) -> None:
    args = argv[1:]
    mutate = None
    if len(args) == 3 and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: pattn8i_int_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    src = (Path(args[0]) / KERNEL).read_text()
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r}")
        old, new = MUTATIONS[mutate]
        if src.count(old) != 1:
            fail(f"mutation anchor {old!r}")
        src = src.replace(old, new)
        print(f"pattn8i_int_diff: applied mutation {mutate}")
    errors: list[str] = []
    pos = []
    for what, line in LINES:
        n = src.count(line)
        if n != 1:
            errors.append(f"{KERNEL}: {line!r} ({what}) occurs {n} times (want 1)")
        else:
            pos.append(src.index(line))
    if not errors and pos != sorted(pos):
        errors.append(f"{KERNEL}: transcribed lines out of order: {pos}")
    print(f"pattn8i_int_diff: {len(LINES) - len(errors)} of {len(LINES)} transcribed kernel lines found once")
    # constants, extracted from the (possibly mutated) text
    m = re.search(r"__int_as_float\(x \+ (0x[0-9A-Fa-f]+)\), ([0-9.]+)f\)", src)
    f18 = re.search(r"__fdiv_rn\(__fmul_rn\(([0-9.]+)f, s\), smax\)", src)
    clamp = re.search(r"qi\[i\] = max\(-(\d+), min\((\d+), __float2int_rn", src)
    if not (m and f18 and clamp):
        fail("cannot extract the i2f / f / clamp constants")
    magic, subc, fac = int(m[1], 16), float(m[2]), float(f18[1])
    h = 1 << 22
    for ok, msg in ((magic == h + 150 * (h + h), f"0x{magic:08X} != 150 * 2^23 + 2^22 (model bits constant)"),
                    (subc == h + (h + h), f"{subc} != 3 * 2^22 (model fsub constant)"),
                    (fac == 18.0, f"requantisation factor {fac} != 18 (model kq bound 7 * 18 = 126)"),
                    (clamp[1] == clamp[2] == "127", f"Q clamp [-{clamp[1]}, {clamp[2]}] != [-127, 127]")):
        if not ok:
            errors.append(msg)
    print(f"pattn8i_int_diff: constants magic 0x{magic:08X}, fsub {subc:.0f}, factor {fac:g}, clamp {clamp[1]}")
    laws = (Path(__file__).resolve().parent / "pattn8i_int_laws.bend").read_text()
    # the named powers 2^22 / 2^24 / 2^31, and P = 32 * (127 * 126) = 512064, the IMMA partial bound (K = 126 = 7 * 18,
    # Q = 127 from the clamp, 32 terms per m16n8k32 IMMA), with 8 P < 2^22
    prods = {(2048, 2048): 1 << 22, (4096, 4096): 1 << 24, (32768, 65536): 1 << 31}
    found = {(int(a), int(b)) for a, b in re.findall(r"Nat\.mul\((\d+)n, (\d+)n\)", laws) if int(a) * int(b) >= 1 << 21}
    for pair in found:
        if pair not in prods or pair[0] * pair[1] != prods[pair]:
            errors.append(f"pattn8i_int_laws.bend: Nat.mul({pair[0]}n, {pair[1]}n) is not 2^22 / 2^24 / 2^31")
    if set(prods) - found:
        errors.append(f"pattn8i_int_laws.bend: missing the products {sorted(set(prods) - found)}")
    pform = "Nat.mul(32n, Nat.mul(127n, 126n))"
    if laws.count(pform) < 5 or 32 * 127 * 126 != 512064 or 32 * 127 * int(fac) * 7 != 512064 or not 8 * 512064 < h:
        errors.append(f"pattn8i_int_laws.bend: the partial bound {pform} (= 32 * 127 * 7 * 18 = 512064, 8 * it < 2^22) is not the laws' bound")
    code = "\n".join(l.split("#", 1)[0] for l in laws.splitlines())
    big_lits = [x for x in re.findall(r"\b(\d+)n\b", code) if int(x) >= 1 << 17]
    if big_lits:
        errors.append(f"pattn8i_int_laws.bend: decimal literals >= 2^17 {big_lits} (unary values too deep for the full contract)")
    if (32768 * 65536 - 1) != 0x7FFFFFFF or h != 2048 * 2048:
        errors.append("INT32_MAX / 2^22 products")
    print(f"pattn8i_int_diff: laws constants {sorted(found)} = 2^22, 2^24, 2^31; P = {pform} = 512064, 8 P < 2^22; "
          "no decimal literal >= 2^17")
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "i2f.cpp"
        c.write_text(C_SRC)
        exe = Path(td) / "i2f"
        r = subprocess.run(source_link.locked(["c++", "-O2", "-msse4.1", "-std=c++17", "-ffp-contract=off", "-fno-fast-math",
                                               f"-DMAGIC={magic}u", f"-DSUBC={subc!r}f", f"-DF18={fac!r}f", "-o", str(exe), str(c)]),
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail("C harness does not compile:\n" + r.stderr[-2000:])
        cres = subprocess.run(source_link.locked([str(exe)]), capture_output=True, text=True)
    print(cres.stdout.rstrip())
    if cres.returncode != 0:
        errors.append(f"host C evidence failed (exit {cres.returncode})")
    if errors:
        fail("; ".join(errors))
    print("pattn8i_int_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
