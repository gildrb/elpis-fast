#!/usr/bin/env python3
"""
Differential check of the parameters ext 2106's laws are instantiated with.

Runs the Bend emitter bend/M16_DIET_TABLE.bend (config defs of bend/m16_diet.bend, phase shapes of
bend/tail_m16_sched.bend / bend/mlp_m16_sched.bend) and compares every printed value with the patched
extension sources of TREE (exllamav3_ext/quant/):
  1. Exl3M16Cfg<MT> (exl3_gemm_m16_kernel.cuh): PF, FOLD, XR, W_STAGE = EXL3_M16_TILES_W * 128,
     X_ITER = MT * 256, for every MT the three DIET kernels are instantiated with (exl3_gemm_m16g.cu,
     exl3_mlp_m16.cu: MT 1 and 2; exl3_tail_m16_kernel.cuh: MT = 1);
  2. in each DIET kernel (m16g, 8201 MLP, 8202 tail), the exact bookkeeping expressions the Bend model
     transcribes (masks RING_MASK = PF * W_STAGE - 1 and XBUF_MASK = 2 * XR * X_ITER - 1, the x-chunk
     and fold tests, the ring / activation offset steps, the `while (--left > 0)` countdown with
     jc = seg_end - left, the issue guards and cursor updates) and the power-of-two / cadence
     static_asserts; the numeric masks, the first m16g issue offset (PF - 1) * W_STAGE, the power-of-two
     flags and the cadence are then recomputed from the extracted constants;
  3. the tail / 8201 phase k-tile counts KT = K / 16 from exl3_m16_wsched.h (K0, K1, K2) and
     exl3_tail_m16_sched.h (EXL3_TAIL_SCHED_KT0).
With --patch PATCH (default: the 2106 patch if present) it prints the patch sha256 and requires
PATCH_SHA256. Exit status 0 iff every comparison is IDENTICAL.

Usage: python3 -B bend/m16_diet_diff.py [--tree TREE] [--patch PATCH]
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent
BEND = "/nix/store/kqhwjzdm96d14fvzblb4jz9m73cr3i0j-bend-2.0.34/bin/bend"
TABLE = "bend/M16_DIET_TABLE.bend"
DEFAULT_TREE = "/tmp/kernel-work/M16gEff/diet/tree"
DEFAULT_PATCH = "/tmp/kernel-work/M16gEff/diet/2106-m16-diet-on8205b.patch"
PATCH_SHA256 = "39da72c8b933109d13100b8b4e7048431230b2eb2a6f16a9c5c811cddc7de57b"
KERNELS = {
    "m16g": "exl3_gemm_m16g_kernel.cuh",
    "mlp": "exl3_mlp_m16_kernel.cuh",
    "tail": "exl3_tail_m16_kernel.cuh",
}
# Expressions every DIET kernel must contain verbatim (whitespace-normalised): what bend/m16_diet.bend
# transcribes (dring, dxnext, dchunk, dfold, dinner / douter) and the facts its `mod` reading needs.
COMMON = [
    "static_assert(((PF * W_STAGE) & (PF * W_STAGE - 1)) == 0 && ((2 * XR * X_ITER) & (2 * XR * X_ITER - 1)) == 0, \"ring sizes\");",
    "static_assert((2 * XR) % Cfg::FOLD == 0, \"fold cadence divides the activation ring\");",
    "constexpr uint32_t RING_MASK = PF * W_STAGE - 1;",
    "constexpr uint32_t XBUF_MASK = 2 * XR * X_ITER - 1;",
    "if ((x_off & (XR * X_ITER - 1)) == 0)",
    "const int jc = seg_end - left;",
    "if (jc > 0) __syncthreads();",
    "iss_off = (iss_off + W_STAGE) & RING_MASK;",
    "cur_off = (cur_off + W_STAGE) & RING_MASK;",
    "x_off = (x_off + X_ITER) & XBUF_MASK;",
    "if ((x_off & (Cfg::FOLD * X_ITER - 1)) == 0) fold();",
    "while (--left > 0);",
    "j = seg_end;",
    "uint32_t x_off = 0;",
]
# dt_step / dt_end / dt_seek / d_exit (tail, 8201) and dg_step / dg_end / dg_src / dg_init (m16g).
PER_KERNEL = {
    "m16g": [
        "uint32_t cur_off = 0;",
        "uint32_t iss_off = (PF - 1) * W_STAGE;",
        "int iss_left = min(KT - iss_kt, n_iter - iss_j);",
        "if (iss_left > 0)",
        "if (--iss_left == 0)",
        "iss_j += KT - iss_kt;",
        "iss_left = min(KT, n_iter - iss_j);",
    ],
    "mlp": [
        "uint32_t cur_off = cur_slot * W_STAGE;",
        "uint32_t iss_off = iss_slot * W_STAGE;",
        "if constexpr (DIET) iss_j += iss_rem;",
        "if (iss_rem > 0)",
        "if (--iss_rem == 0)",
        "iss_j += iss_rem;",
        "iss_j -= iss_rem;",
        "cur_slot = cur_off / W_STAGE;",
        "iss_slot = iss_off / W_STAGE;",
    ],
}
PER_KERNEL["tail"] = PER_KERNEL["mlp"]


def fail(msg: str) -> NoReturn:
    raise SystemExit(f"m16_diet_diff: FAIL: {msg}")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def take_opt(argv: list[str], key: str) -> str | None:
    return argv[argv.index(key) + 1] if key in argv else None


def one(pattern: str, text: str, what: str) -> int:
    found = re.findall(pattern, text)
    if len(found) != 1:
        fail(f"{what}: expected exactly one match of {pattern!r}, found {len(found)}")
    return int(found[0])


def bend_table() -> tuple[dict[int, dict[str, list[int]]], list[int]]:
    proc = subprocess.run([BEND, TABLE], cwd=REPO, capture_output=True, text=True, timeout=1800, check=False)
    if proc.returncode != 0:
        fail(f"{BEND} {TABLE} exited {proc.returncode}: {proc.stderr.strip()[:400]}")
    cfgs: dict[int, dict[str, list[int]]] = {}
    kts: list[int] | None = None
    for line in proc.stdout.splitlines():
        if m := re.fullmatch(r"cfg mt (\d+): (.*)", line):
            fields: dict[str, list[int]] = {}
            key = ""
            for tok in m.group(2).split():
                if tok.isdigit():
                    if not key:
                        fail(f"TABLE value before a field name: {line!r}")
                    fields[key].append(int(tok))
                else:
                    key = tok
                    fields[key] = []
            cfgs[int(m.group(1))] = fields
        elif m := re.fullmatch(r"kt tail (\d+) (\d+) (\d+) mlp (\d+) (\d+)", line):
            kts = [int(x) for x in m.groups()]
        elif line.strip():
            fail(f"unexpected TABLE line {line!r}")
    if not cfgs or kts is None:
        fail(f"TABLE printed no cfg / kt lines: {proc.stdout[:200]!r}")
    return cfgs, kts


def pow2(n: int) -> int:
    return int(n > 0 and n & (n - 1) == 0)


def source_values(quant: Path) -> tuple[dict[int, dict[str, list[int]]], list[int], set[int]]:
    m16 = (quant / "exl3_gemm_m16_kernel.cuh").read_text()
    cfg = re.search(r"template <int MT>\s*struct Exl3M16Cfg\s*\{(.*?)\n\};", m16, re.DOTALL)
    if not cfg:
        fail("Exl3M16Cfg<MT> not found in exl3_gemm_m16_kernel.cuh")
    body = cfg.group(1)
    tiles_w = one(r"#define EXL3_M16_TILES_W (\d+)\n", m16, "EXL3_M16_TILES_W")
    pf = one(r"static constexpr int PF = (\d+);", body, "Exl3M16Cfg::PF")
    fold = one(r"static constexpr int FOLD = (\d+);", body, "Exl3M16Cfg::FOLD")
    xr = one(r"static constexpr int XR = (\d+);", body, "Exl3M16Cfg::XR")
    ws_mul = one(r"static constexpr int W_STAGE = EXL3_M16_TILES_W \* (\d+);", body, "Exl3M16Cfg::W_STAGE")
    xi_mul = one(r"static constexpr int X_ITER = MT \* (\d+);", body, "Exl3M16Cfg::X_ITER")

    # MT instances of the DIET kernels.
    mts: set[int] = set()
    for host in ("exl3_gemm_m16g.cu", "exl3_mlp_m16.cu"):
        text = (quant / host).read_text()
        sel = re.search(r"void\* kernel_ptr_d\(int MT, bool \w+\)\s*\{(.*?)\n\}", text, re.DOTALL)
        if not sel:
            fail(f"{host}: kernel_ptr_d(MT, ...) not found")
        found = {int(x) for x in re.findall(r"kernel_ptr<(\d+), \w+, DIET>", sel.group(1))}
        if not found:
            fail(f"{host}: no kernel_ptr<MT, ..., DIET> instances")
        mts |= found
    tail = (quant / KERNELS["tail"]).read_text()
    mts.add(one(r"constexpr int MT = (\d+);", tail, "tail MT"))

    for name, fname in KERNELS.items():
        text = norm((quant / fname).read_text())
        for expr in COMMON + PER_KERNEL[name]:
            if norm(expr) not in text:
                fail(f"{fname}: DIET expression not found: {expr}")
        if "DIET" not in text:
            fail(f"{fname}: no DIET template parameter")
        print(f"{fname}: {len(COMMON) + len(PER_KERNEL[name])} DIET expressions present")

    ws = tiles_w * ws_mul
    vals: dict[int, dict[str, list[int]]] = {}
    for mt in sorted(mts):
        xi = mt * xi_mul
        ring, xbuf, xch, foldp = pf * ws, 2 * xr * xi, xr * xi, fold * xi
        vals[mt] = {
            "pf": [pf], "fold": [fold], "xr": [xr], "tiles_w": [tiles_w], "w_stage": [ws], "x_iter": [xi],
            "ring_mask": [ring - 1], "xbuf_mask": [xbuf - 1], "xchunk_mask": [xch - 1], "fold_mask": [foldp - 1],
            "iss_off0": [(pf - 1) * ws], "pow2": [pow2(ring), pow2(xbuf), pow2(xch), pow2(foldp)],
            "cadence": [int((2 * xr) % fold == 0)],
        }

    ws_h = (quant / "exl3_m16_wsched.h").read_text()
    k0 = one(r"\bK0 = (\d+),", ws_h, "exl3_m16_wsched.h K0")
    k1 = one(r"\bK1 = (\d+),", ws_h, "exl3_m16_wsched.h K1")
    k2 = one(r"\bK2 = (\d+),", ws_h, "exl3_m16_wsched.h K2")
    kt0_baked = one(r"#define EXL3_TAIL_SCHED_KT0 (\d+)\n", (quant / "exl3_tail_m16_sched.h").read_text(), "EXL3_TAIL_SCHED_KT0")
    for fname, decl in ((KERNELS["tail"], ("KT0 = K0 / 16", "KT1 = K1 / 16", "KT2 = K2 / 16")),
                        (KERNELS["mlp"], ("KT1 = K1 / 16", "KT2 = K2 / 16"))):
        text = (quant / fname).read_text()
        for d in decl:
            if f"constexpr int {d};" not in text:
                fail(f"{fname}: `constexpr int {d};` not found")
    if kt0_baked != k0 // 16:
        fail(f"EXL3_TAIL_SCHED_KT0 {kt0_baked} != K0 / 16 = {k0 // 16}")
    kts = [k0 // 16, k1 // 16, k2 // 16, k1 // 16, k2 // 16]
    return vals, kts, mts


def main(argv: list[str]) -> int:
    tree = Path(take_opt(argv, "--tree") or DEFAULT_TREE)
    patch = take_opt(argv, "--patch") or (DEFAULT_PATCH if Path(DEFAULT_PATCH).exists() else None)
    if patch:
        digest = hashlib.sha256(Path(patch).read_bytes()).hexdigest()
        print(f"patch {patch} sha256 {digest}")
        if digest != PATCH_SHA256:
            fail(f"patch sha256 {digest} != {PATCH_SHA256}")
    quant = tree / "exllamav3_ext" / "quant"
    bend_cfgs, bend_kts = bend_table()
    src_cfgs, src_kts, mts = source_values(quant)
    if set(bend_cfgs) != mts:
        fail(f"TABLE MT instances {sorted(bend_cfgs)} != the kernels' {sorted(mts)}")
    for mt in sorted(mts):
        if bend_cfgs[mt] != src_cfgs[mt]:
            diff = {k: (bend_cfgs[mt].get(k), v) for k, v in src_cfgs[mt].items() if bend_cfgs[mt].get(k) != v}
            fail(f"MT {mt}: Bend != sources (Bend, sources): {diff}")
        if bend_cfgs[mt]["pow2"] != [1, 1, 1, 1] or bend_cfgs[mt]["cadence"] != [1]:
            fail(f"MT {mt}: a mask modulus is not a power of two or FOLD does not divide 2 XR")
        print(f"cfg mt {mt}: Bend == {tree} " + " ".join(f"{k} {','.join(map(str, v))}" for k, v in src_cfgs[mt].items())
              + " IDENTICAL")
    if bend_kts != src_kts:
        fail(f"phase KTs: Bend {bend_kts} != sources {src_kts}")
    if min(src_kts) < 1:
        fail(f"phase KTs {src_kts}: the laws need KT >= 1")
    print(f"kt tail {src_kts[0]} {src_kts[1]} {src_kts[2]} mlp {src_kts[3]} {src_kts[4]}: Bend == {tree} IDENTICAL")
    print("m16_diet_diff: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
