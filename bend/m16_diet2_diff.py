#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Differential check of ext 2107's decode model against the patched sources.

The model is bend/m16_diet2.bend with bend/m16_diet2_spec.bend. Runs the Bend
emitter bend/M16_DIET2_TABLE.bend and, independently, evaluates the C sources of
TREE (exllamav3_ext/) on the same inputs:
  1. extraction: every statement of dq8_aligned_4bits (quant/exl3_dq.cuh, the DIET
     0 / 1 decode) and of dq8_m16_regs (quant/exl3_m16_decode.cuh, DIET 2) is parsed
     (funnel shifts, FSHF_IMM, BFE16_IMM, masks, __byte_perm with its selector; any
     unrecognised statement fails) and evaluated on the TABLE's marker word pairs;
     every window w0 .. w7 must equal the Bend ref / diet / spec windows. The decode
     pairing (frag0[0] = decode(w0, w1), ...) must be the model's in both functions,
     and no prmt selector may use the sign-replicate bit;
  2. addresses: the hoisted dw0 / dw1 expressions of the three DIET kernels
     (evaluated by bend/pysubset.py) and the m16_lds<OFF> immediates of
     m16_tile_diet, and the reference mma_tile call
     dq8_aligned_4bits(st + t * 128, lane * 8) with dq8's i1 / i0 word indices, are
     evaluated for lanes 0 .. 31 and tiles 0 .. 3 and compared with the TABLE's
     diet / ref offsets; the tile-step calls, the A-fragment order, the ldmatrix
     instructions and the MMA loop of m16_tile_diet must be those of every kernel's
     mma_tile;
  3. the switch: exl3_m16_diet's parse / set lines, each launcher's instance
     selector and 3-entry cache, and the DIET guards of the kernels are matched,
     evaluated and compared with the TABLE's mode / set / inst lines;
  4. the hoisted L2 policy is created with the same createpolicy instruction
     cp_async_stream uses per call, and cp_async_stream_pol issues the same
     cp.async.
With --patch PATCH (default: the tracked patches/exl3-ext/2107 patch) it prints the
patch sha256 and requires PATCH_SHA256. Exit status 0 iff every comparison is
IDENTICAL.

Usage: python3 -B bend/m16_diet2_diff.py --tree TREE [--patch PATCH]
  TREE: OUT/patched of bend/engine_trees.py (full series: the checks need the FO
  parameter of ext 3023).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

REPO = source_link.REPO
TABLE = "bend/M16_DIET2_TABLE.bend"
DEFAULT_PATCH = REPO / "patches/exl3-ext/2107-m16-decode-diet-on2106.patch"
PATCH_SHA256 = "0584196dab3f0cdd2e255d383dfd23c6a76c10ec7c3c151fabcbd9418c1b2ae9"
KERNELS = (
    "exl3_gemm_m16g_kernel.cuh",
    "exl3_mlp_m16_kernel.cuh",
    "exl3_tail_m16_kernel.cuh",
)
LAUNCHERS = ("exl3_gemm_m16g.cu", "exl3_mlp_m16.cu", "exl3_tail_m16.cu")
PAIRS = [
    "frag0[0]=decode_3inst_2<cb>(w0,w1)",
    "frag0[1]=decode_3inst_2<cb>(w2,w3)",
    "frag1[0]=decode_3inst_2<cb>(w4,w5)",
    "frag1[1]=decode_3inst_2<cb>(w6,w7)",
]
AFRAG = "a[0]=f0[0];a[1]=f1[0];a[2]=f0[1];a[3]=f1[1];"
M32 = 0xFFFFFFFF
BEND_TIMEOUT = 1800
WINDOWS = 8
TILES = 4
LANES = 32
MIN_EXTRACT_LINES = 4
DIET_GUARDS = 3  # `if constexpr (DIET >= 2)`: hoist, cp.async, tile
MODE_MAX = 2  # exl3_m16_diet: `mode = set > 2 ? 2 : set`
LDSM_FORMS = 2  # m16_tile_diet's ldmatrix: the x4 and the x2 form
PRMT_SIGN = 8  # prmt selector nibble bit 3: sign-replicate mode
SWITCH_LINES = (
    "if (!env || !env[0]) return 2;",
    (
        "TORCH_CHECK((env[0] == '0' || env[0] == '1' || env[0] == '2') && !env[1], "
        '"EXL3_M16_DIET must be 0, 1 or 2, got \'", env, "\'");'
    ),
    "return env[0] - '0';",
    "if (set >= 0) mode = set > 2 ? 2 : set;",
    'const char* env = std::getenv("EXL3_M16_DIET");',
)
# Each launcher's 3-way instance selector: mode d -> DIET instance
SELECTOR = (
    r"void\* k = diet == 2 \? ([^;]*?<[^;]*?(\d)>[^;]*?) : "
    r"diet == 1 \? ([^;]*?(\d)>[^;]*?) : ([^;]*?(\d)>[^;]*?);"
)


def fail(msg: str) -> NoReturn:
    """Stop with a FAIL message.

    Args:
        msg: The failure description.

    Raises:
        SystemExit: Always.

    """
    text = f"m16_diet2_diff: FAIL: {msg}"
    raise SystemExit(text)


def squash(text: str) -> str:
    """Return the text without whitespace.

    Args:
        text: The text.

    Returns:
        The squashed text.

    """
    return re.sub(r"\s+", "", text)


def take_opt(argv: list[str], key: str) -> str | None:
    """Return the value following option key, if present.

    Args:
        argv: The arguments.
        key: The option.

    Returns:
        The value, or None.

    """
    return argv[argv.index(key) + 1] if key in argv else None


def body(text: str, head: str, what: str) -> str:
    """Return the brace-balanced body following the first match of regex head.

    Args:
        text: The source.
        head: The regular expression of the text before the body.
        what: The body's name, for the failure message.

    Returns:
        The body without its braces.

    """
    m = re.search(head, text)
    if not m:
        fail(f"{what}: {head!r} not found")
    i = text.index("{", m.start())
    depth = 0
    for k in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[i + 1 : k]
    fail(f"{what}: unbalanced body")


def funnel(lo: int, hi: int, n: int) -> int:
    """Return the 32-bit funnel shift right of hi:lo by n mod 32.

    Args:
        lo: The low word.
        hi: The high word.
        n: The shift.

    Returns:
        The shifted word.

    """
    return (((hi << 32) | lo) >> (n & 31)) & M32


def prmt(x: int, y: int, sel: int) -> int:
    """Return __byte_perm(x, y, sel) without the sign-replicate mode.

    Args:
        x: The low word.
        y: The high word.
        sel: The selector.

    Returns:
        The permuted word.

    """
    both = (y << 32) | x
    out = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        if s & PRMT_SIGN:
            fail(f"prmt selector {sel:#x} uses the sign-replicate mode")
        out |= ((both >> (8 * s)) & 0xFF) << (8 * i)
    return out


def run_statement(st: str, stmt: str, env: dict[str, int], what: str) -> bool:
    """Evaluate one squashed extraction statement into env.

    Args:
        st: The squashed statement.
        stmt: The statement as written, for the failure message.
        env: The word values, updated.
        what: The function's name, for the failure message.

    Returns:
        True for a decode pairing statement (env unchanged), else False.

    """
    if (
        m := re.fullmatch(
            r"(?:constuint32_t)?(\w+)=__funnelshift_r\((\w+),(\w+),(\d+)\)", st
        )
    ) or (m := re.fullmatch(r"FSHF_IMM\((\w+),(\w+),(\w+),(\d+)\)", st)):
        env[m[1]] = funnel(env[m[2]], env[m[3]], int(m[4]))
    elif m := re.fullmatch(r"BFE16_IMM\((\w+),(\w+),(\d+)\)", st):
        env[m[1]] = (env[m[2]] >> int(m[3])) & 0xFFFF
    elif m := re.fullmatch(r"(?:constuint32_t)?(\w+)=(\w+)&0xffff", st):
        env[m[1]] = env[m[2]] & 0xFFFF
    elif m := re.fullmatch(
        r"(?:constuint32_t)?(\w+)=__byte_perm\((\w+),(\w+),(0x[0-9a-fA-F]+)\)", st
    ):
        env[m[1]] = prmt(env[m[2]], env[m[3]], int(m[4], 16))
    elif re.fullmatch(r"frag[01]\[[01]\]=decode_3inst_2<cb>\(w\d,w\d\)", st):
        return True
    else:
        fail(f"{what}: unrecognised statement {stmt!r}")
    return False


def run_extract(
    src: str, what: str, a: int, b: int
) -> tuple[dict[str, int], list[str]]:
    """Evaluate an extraction function body on words a (i0) and b (i1).

    Args:
        src: The function body.
        what: The function's name, for the failure message.
        a: Word i0.
        b: Word i1.

    Returns:
        The values of its words and its decode pairing statements.

    """
    env: dict[str, int] = {"a": a, "b": b, "0": 0}
    pairs: list[str] = []
    for stmt in (s.strip() for s in src.split(";")):
        st = squash(stmt)
        if not st or re.fullmatch(r"uint32_t(\w+,)*\w+", st):
            continue
        if st in {"i1=t_offset>>3", "i0=(i1+31)&31", "a=ptr[i0]", "b=ptr[i1]"}:
            continue
        if run_statement(st, stmt, env, what):
            pairs.append(st)
    return env, pairs


def bend_table() -> list[str]:
    """Run the Bend table.

    Returns:
        Its non-blank output lines.

    """
    bend = source_link.bend()
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
        [bend, TABLE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=BEND_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        fail(f"{bend} {TABLE} exited {proc.returncode}: {proc.stderr.strip()[:400]}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def c_parse(env: str | None) -> str:
    """Return exl3_m16_diet's mode for an EXL3_M16_DIET value.

    exl3_m16_diet (matched line by line below):
      if (!env || !env[0]) return 2;
      TORCH_CHECK((env[0] == '0' || env[0] == '1' || env[0] == '2') && !env[1], ...);
      return env[0] - '0';

    Args:
        env: The variable's value, or None if unset.

    Returns:
        The mode, or "bad" where the check fails.

    """
    if env is None or not env:
        return "2"
    if env[0] in "012" and len(env) == 1:
        return str(ord(env[0]) - ord("0"))
    return "bad"


def c_set(mode: int, x: int | None) -> int:
    """Return the mode after exl3_m16_diet(x).

    `if (set >= 0) mode = set > 2 ? 2 : set;` (query: set < 0)

    Args:
        mode: The current mode.
        x: The set argument, or None for a query.

    Returns:
        The new mode.

    """
    if x is None:
        return mode
    return min(x, MODE_MAX)


def check_extraction(table: list[str], ref_src: str, diet_src: str) -> None:
    """Check both extraction functions against the TABLE's windows (section 1).

    Args:
        table: The TABLE lines.
        ref_src: dq8_aligned_4bits's body.
        diet_src: dq8_m16_regs's body.

    """
    n_ext = 0
    for line in table:
        m = re.fullmatch(
            r"extract (\w{8}) (\w{8}): ref (.*) \| diet (.*) \| spec (.*)", line
        )
        if not m:
            continue
        a, b = int(m[1], 16), int(m[2], 16)
        want_ref, want_diet, want_spec = (
            [int(x, 16) for x in m[k].split()] for k in (3, 4, 5)
        )
        for name, src, want in (
            ("dq8_aligned_4bits", ref_src, want_ref),
            ("dq8_m16_regs", diet_src, want_diet),
        ):
            env, pairs = run_extract(src, name, a, b)
            got = [env[f"w{i}"] for i in range(WINDOWS)]
            if got != want:
                fail(
                    f"{name} on {a:08X} {b:08X}: C {[f'{x:08X}' for x in got]} != "
                    f"Bend {[f'{x:08X}' for x in want]}"
                )
            if pairs != PAIRS:
                fail(f"{name}: decode pairing {pairs} != {PAIRS}")
        both = (a << 32) | b
        spec = [(both >> (4 * (7 - i))) & 0xFFFF for i in range(WINDOWS)]
        if want_spec != spec:
            fail(
                f"Bend spec windows on {a:08X} {b:08X} != bits 4 (7 - i) .. + 15 "
                f"of {{a:b}}"
            )
        sys.stdout.write(
            f"extract {a:08X} {b:08X}: dq8_aligned_4bits == dq8_m16_regs == Bend ref "
            "== Bend diet == window(4 (7 - i)) IDENTICAL\n"
        )
        n_ext += 1
    if n_ext < MIN_EXTRACT_LINES:
        fail(f"TABLE printed {n_ext} extract lines")


def tile_loads(tile: str) -> dict[int, dict[str, int]]:
    """Return m16_tile_diet's m16_lds<OFF> immediates by tile and word.

    Args:
        tile: m16_tile_diet's body.

    Returns:
        The offsets by tile, then "a" (w0) / "b" (w1).

    """
    offs: dict[int, dict[str, int]] = {}
    for m in re.finditer(r"w([ab])\[(\d)\] = m16_lds<(\d+)>\(w([01])\);", tile):
        if {"a": "0", "b": "1"}[m[1]] != m[4]:
            fail(f"m16_tile_diet: w{m[1]}[{m[2]}] reads w{m[4]}")
        offs.setdefault(int(m[2]), {})[m[1]] = int(m[3])
    if sorted(offs) != [0, 1, 2, 3] or any(
        sorted(v) != ["a", "b"] for v in offs.values()
    ):
        fail(f"m16_tile_diet: loads {offs}")
    return offs


def check_tile_step(fname: str, k: str, ldsm: list[str], mma_diet: str) -> None:
    """Check a kernel's mma_tile and tile-step calls against m16_tile_diet.

    Args:
        fname: The kernel file name.
        k: The kernel source.
        ldsm: m16_tile_diet's ldmatrix forms.
        mma_diet: m16_tile_diet's squashed A fragment / MMA loop.

    """
    mt = body(
        k,
        r"auto mma_tile = \[&\] \(uint32_t sa, const uint8_t\* st\)\s*\{",
        fname + " mma_tile",
    )
    cb = "cb" if "m16g" in fname else "2"
    if (
        f"dq8_aligned_4bits<{cb}>((const uint32_t*) (st + t * 128), lane * 8, f0, f1);"
        not in mt
    ):
        fail(f"{fname}: mma_tile's dq8_aligned_4bits call not found")
    # m16_tile_diet: `if constexpr (MT == 2)` x4 else x2; the tail (MT = 1) keeps only
    # the x2 form
    ref_ldsm = re.findall(r'asm volatile \("(ldmatrix[^"]*)"', mt)
    if not (
        ref_ldsm == ldsm or (ref_ldsm == ldsm[1:] and "constexpr int MT = 1;" in k)
    ):
        fail(f"{fname}: mma_tile's ldmatrix {ref_ldsm} != m16_tile_diet's {ldsm}")
    if squash(mt[mt.index("FragA a;") :]) != mma_diet:
        fail(f"{fname}: the A fragment / MMA loop of mma_tile != m16_tile_diet's")
    if AFRAG not in squash(mt) or AFRAG not in mma_diet:
        fail(f"{fname}: A-fragment order")
    call = (
        f"if constexpr (DIET >= 2) m16_tile_diet<MT, {cb}>(dxs + x_off, "
        "dw0 + cur_off, dw1 + cur_off, hacc);"
    )
    if call not in k or "else mma_tile(xbuf_sa + x_off, ring + cur_off);" not in k:
        fail(f"{fname}: the DIET 2 / reference tile-step calls not found")
    if k.count("if constexpr (DIET >= 2)") != DIET_GUARDS:
        fail(
            f"{fname}: expected 3 `if constexpr (DIET >= 2)` guards (hoist, cp.async, "
            f"tile), found {k.count('if constexpr (DIET >= 2)')}"
        )
    if (
        "cp_async_stream_pol(ring_lane + iss_off, iss_ptr, dpol);" not in k
        or "else cp_async_stream(ring_lane + iss_off, iss_ptr);" not in k
    ):
        fail(f"{fname}: the weight cp.async pair not found")
    if not re.search(r"template <[^>]*int DIET = 0>", k):
        fail(f"{fname}: `int DIET = 0` template parameter not found")
    sys.stdout.write(
        f"{fname}: hoist, tile-step calls, mma_tile (ldmatrix, "
        f"dq8_aligned_4bits<{cb}>(st + t * 128, lane * 8), A order, MMA loop) matched\n"
    )


def kernel_hoists(quant: Path, tile: str) -> tuple[str, str]:
    """Check every DIET kernel's hoist and tile step; return its dw0 / dw1 offsets.

    Args:
        quant: The tree's exllamav3_ext/quant directory.
        tile: m16_tile_diet's body.

    Returns:
        The hoisted dw0 and dw1 expressions after `rs + `.

    """
    if "dq8_m16_regs<cb>(wa[t], wb[t], f0, f1);" not in tile:
        fail("m16_tile_diet: dq8_m16_regs<cb>(wa[t], wb[t], f0, f1) not found")
    ldsm = re.findall(r'asm volatile \("(ldmatrix[^"]*)"', tile)
    if (
        len(ldsm) != LDSM_FORMS
        or "x4" not in ldsm[0]
        or "x2" not in ldsm[1]
        or "if constexpr (MT == 2)" not in tile
    ):
        fail(f"m16_tile_diet: ldmatrix forms {ldsm}")
    mma_diet = squash(tile[tile.index("FragA a;") :])
    hoist: tuple[str, str] | None = None
    for fname in KERNELS:
        k = (quant / fname).read_text()
        h = body(
            k, r"if constexpr \(DIET >= 2\)\s*\{\s*const uint32_t rs", fname + " hoist"
        )
        m0 = re.search(r"dw0 = rs \+ ([^;]+);", h)
        m1 = re.search(r"dw1 = rs \+ ([^;]+);", h)
        if not (
            m0
            and m1
            and "uint32_t dw0 = 0, dw1 = 0, dxs = xbuf_sa;" in k
            and "dpol = m16_stream_policy();" in h
        ):
            fail(f"{fname}: hoisted dw0 / dw1 / dxs / dpol not found")
        found = (str(m0[1]), str(m1[1]))
        if hoist is None:
            hoist = found
        elif found != hoist:
            fail(f"{fname}: dw0 / dw1 differ between kernels")
        check_tile_step(fname, k, ldsm, mma_diet)
    if hoist is None:
        fail("no hoisted addresses")
    return hoist


def check_lds(
    table: list[str], hoist: tuple[str, str], offs: dict[int, dict[str, int]]
) -> None:
    """Compare the hoisted and reference lds offsets with the TABLE's (section 2).

    Args:
        table: The TABLE lines.
        hoist: The hoisted dw0 / dw1 expressions.
        offs: m16_tile_diet's m16_lds immediates.

    """
    for expr in hoist:
        if not re.fullmatch(r"[\s\w()+&*]+", expr) or set(
            re.findall(r"[A-Za-z_]\w*", expr)
        ) != {"lane"}:
            fail(f"unexpected hoist expression {expr!r}")
    # checked above: lane, digits, + & * ( )
    dw0, dw1 = (pysubset.compile_expr(e) for e in hoist)
    n_lds = 0
    for line in table:
        m = re.fullmatch(r"lds lane (\d+): diet ([\d ]+) \| ref ([\d ]+)", line)
        if not m:
            continue
        lane = int(m[1])
        d0, d1 = dw0({"lane": lane}), dw1({"lane": lane})
        if not isinstance(d0, int) or not isinstance(d1, int):
            fail(f"lane {lane}: hoisted offsets {d0!r}, {d1!r} are not integers")
        diet = [x for t in range(TILES) for x in (d0 + offs[t]["a"], d1 + offs[t]["b"])]
        i1 = (lane * 8) >> 3
        i0 = (i1 + 31) & 31
        ref = [x for t in range(TILES) for x in (t * 128 + 4 * i0, t * 128 + 4 * i1)]
        if (
            [int(x) for x in m[2].split()] != diet
            or [int(x) for x in m[3].split()] != ref
            or diet != ref
        ):
            fail(f"lane {lane}: C diet {diet} / C ref {ref} / Bend {m[2]} | {m[3]}")
        n_lds += 1
    if n_lds != LANES:
        fail(f"TABLE printed {n_lds} lds lines")
    imm = ",".join(str(offs[t]["a"]) for t in range(TILES))
    sys.stdout.write(
        f"lds lanes 0..31, tiles 0..3: C hoisted (dw0 = rs + {hoist[0]}, "
        f"dw1 = rs + {hoist[1]}, m16_lds<{imm}>) == C mma_tile / dq8 == Bend diet == "
        "Bend ref IDENTICAL\n"
    )


def check_addresses(table: list[str], quant: Path, dec: str, ref_src: str) -> None:
    """Check the addresses and the tile step (section 2).

    Args:
        table: The TABLE lines.
        quant: The tree's exllamav3_ext/quant directory.
        dec: exl3_m16_decode.cuh.
        ref_src: dq8_aligned_4bits's body.

    """
    tile = body(
        dec,
        r"__device__ __forceinline__ void m16_tile_diet\(uint32_t sa, uint32_t w0, "
        r"uint32_t w1, Acc& hacc\)\s*\{",
        "m16_tile_diet",
    )
    offs = tile_loads(tile)
    hoist = kernel_hoists(quant, tile)
    dq_idx = [squash(x) for x in ref_src.split(";")]
    if "i1=t_offset>>3" not in dq_idx or "i0=(i1+31)&31" not in dq_idx:
        fail("dq8_aligned_4bits: word indices")
    check_lds(table, hoist, offs)


def check_modes(table: list[str], quant: Path) -> None:
    """Check exl3_m16_diet's parse / set lines against the TABLE's.

    Args:
        table: The TABLE lines.
        quant: The tree's exllamav3_ext/quant directory.

    """
    gm = (quant / "exl3_gemm_m16.cu").read_text()
    sw = body(gm, r"int exl3_m16_diet\(int set\)\s*\{", "exl3_m16_diet")
    for want in SWITCH_LINES:
        if want not in sw:
            fail(f"exl3_m16_diet: `{want}` not found")
    envs = {
        "unset": None,
        '""': "",
        '"0"': "0",
        '"1"': "1",
        '"2"': "2",
        '"3"': "3",
        '"00"': "00",
        '"a"': "a",
    }
    sets = {"q": None, "0": 0, "1": 1, "2": 2, "3": 3, "7": 7}
    got_modes = {
        m[1]: m[2] for line in table if (m := re.fullmatch(r"mode (\S+): (\S+)", line))
    }
    got_sets = {
        m[1]: int(m[2])
        for line in table
        if (m := re.fullmatch(r"set (\S+): (\d+)", line))
    }
    want_modes = {k: c_parse(v) for k, v in envs.items()}
    if got_modes != want_modes:
        fail(f"modes: Bend {got_modes} != C {want_modes}")
    want_sets = {k: c_set(1, v) for k, v in sets.items()}
    if got_sets != want_sets:
        fail(f"set: Bend {got_sets} != C {want_sets}")
    modes = ", ".join(f"{k} -> {v}" for k, v in got_modes.items())
    diets = ", ".join(f"{k} -> {v}" for k, v in got_sets.items())
    sys.stdout.write(
        f"EXL3_M16_DIET {modes}; diet(x) from mode 1 {diets}: Bend == C IDENTICAL\n"
    )


def launcher_instances(quant: Path) -> dict[int, int]:
    """Return the DIET instance each mode selects, the same in every launcher.

    Args:
        quant: The tree's exllamav3_ext/quant directory.

    Returns:
        The instance by mode.

    """
    insts: dict[str, dict[int, int]] = {}
    for fname in LAUNCHERS:
        text = (quant / fname).read_text()
        m = re.search(SELECTOR, text)
        if not m:
            fail(f"{fname}: the 3-way instance selector not found")
        sel = {2: int(m[2]), 1: int(m[4]), 0: int(m[6])}
        cache = re.search(
            r"static \w+ cache\[MAX_DEVICES\](?:\[2\])*\[3\] = \{\};", text
        )
        index = re.search(r"cache\[device\](?:\[[^\]]+\])*\[diet\];", text)
        if not (cache and index and "const int diet = exl3_m16_diet();" in text):
            fail(f"{fname}: the [3] cache indexed by exl3_m16_diet() not found")
        insts[fname] = sel
    first = next(iter(insts.values()))
    if any(v != first for v in insts.values()):
        fail(f"launchers disagree: {insts}")
    return first


def check_switch(table: list[str], quant: Path) -> None:
    """Check the switch, the launchers' selectors and the DIET guards (section 3).

    Args:
        table: The TABLE lines.
        quant: The tree's exllamav3_ext/quant directory.

    """
    check_modes(table, quant)
    first = launcher_instances(quant)
    got_inst = {
        int(m[1]): (int(m[2]), m[3], m[4])
        for line in table
        if (m := re.fullmatch(r"inst (\d): (\d) ([01]) ([01])", line))
    }
    want_inst = {
        d: (first[d], str(int(first[d] != 0)), str(int(first[d] >= MODE_MAX)))
        for d in (0, 1, 2)
    }
    if got_inst != want_inst:
        fail(f"inst: Bend {got_inst} != C {want_inst}")
    for fname in KERNELS:
        k = (quant / fname).read_text()
        if not (("if constexpr (DIET)" in k) or ("if constexpr (!DIET)" in k)):
            fail(f"{fname}: 2106's DIET guard not found")
    sys.stdout.write(
        f"launchers {', '.join(LAUNCHERS)}: mode d -> instance DIET {first}, "
        "cache[3]; DIET guards (bookkeeping DIET != 0, "
        "tile DIET >= 2): Bend == C IDENTICAL\n"
    )


def check_policy(ext: Path, dec: str) -> None:
    """Check the hoisted L2 policy and cp.async against cp_async_stream (section 4).

    Args:
        ext: The tree's exllamav3_ext directory.
        dec: exl3_m16_decode.cuh.

    """
    ptx = (ext / "ptx.cuh").read_text()
    stream = body(
        ptx,
        r"__device__ inline void cp_async_stream\(void\* smem_ptr, "
        r"const void\* glob_ptr\)\s*\{",
        "cp_async_stream",
    )
    pol = body(
        dec,
        r"__device__ __forceinline__ uint64_t m16_stream_policy\(\)\s*\{",
        "m16_stream_policy",
    )
    polc = body(
        dec,
        r"__device__ __forceinline__ void cp_async_stream_pol\(void\* smem_ptr, "
        r"const void\* glob_ptr, uint64_t pol\)\s*\{",
        "cp_async_stream_pol",
    )
    if (
        "createpolicy.fractional.L2::evict_first.b64 p, 1.0;" not in stream
        or "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" not in pol
    ):
        fail("createpolicy instructions differ")
    if (
        "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;" not in stream
        or '"n"(bytes)' not in stream
        or "const int bytes = 16;" not in stream
        or "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" not in polc
    ):
        fail("cp.async instructions differ")
    sys.stdout.write(
        "L2 policy: createpolicy.fractional.L2::evict_first.b64 .., 1.0 and "
        "cp.async.cg ... L2::cache_hint 16 B in both IDENTICAL\n"
    )


def main(argv: list[str]) -> int:
    """Run the source link.

    Args:
        argv: The arguments after the program name.

    Returns:
        The exit status.

    """
    tree_opt = take_opt(argv, "--tree")
    if tree_opt is None:
        fail("usage: m16_diet2_diff.py --tree TREE [--patch PATCH]")
    tree = Path(tree_opt)
    patch = take_opt(argv, "--patch") or str(DEFAULT_PATCH)
    if patch:
        digest = hashlib.sha256(Path(patch).read_bytes()).hexdigest()
        sys.stdout.write(f"patch {patch} sha256 {digest}\n")
        if digest != PATCH_SHA256:
            fail(f"patch sha256 {digest} != {PATCH_SHA256}")
    ext = tree / "exllamav3_ext"
    quant = ext / "quant"
    table = bend_table()

    # ---- 1. extraction ----
    dq = (quant / "exl3_dq.cuh").read_text()
    dec = (quant / "exl3_m16_decode.cuh").read_text()
    ref_src = body(
        dq,
        r"void dq8_aligned_4bits\(const uint32_t\* ptr, int t_offset, FragB& frag0, "
        r"FragB& frag1\)\s*\{",
        "dq8_aligned_4bits",
    )
    diet_src = body(
        dec,
        r"void dq8_m16_regs\(uint32_t a, uint32_t b, FragB& frag0, FragB& frag1\)\s*\{",
        "dq8_m16_regs",
    )
    check_extraction(table, ref_src, diet_src)
    check_addresses(table, quant, dec, ref_src)
    check_switch(table, quant)
    check_policy(ext, dec)
    sys.stdout.write("m16_diet2_diff: PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
