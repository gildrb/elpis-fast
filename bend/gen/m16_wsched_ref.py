#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Reference generator of the 8201 / 8202 schedule tables (ext 8205).

Usage text: USAGE (printed on bad arguments).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wpart

USAGE = (
    "Reference generator of the 8201 fused-MLP and 8202 layer-tail "
    "schedule tables under the shared slot-weighted\n"
    "split-K partition (ext 8205; partition formulas: "
    "bend/gen/wpart.py, imported). ONE generator for both kernels.\n"
    "\n"
    "Origin: the TailV2b work copied this file (gen_sched.py) into "
    "the repo without change of logic. It imports\n"
    "wpart from its own directory only. Role: "
    "bend/m16_wsched_diff.py compares flat() with the Bend emitter.\n"
    "\n"
    "  tables(wt) -> dict with the 8201 block/pair/wait tables, the "
    "tail block table, the per-group first-contributor /\n"
    "                contributor-count tables of phases 0/1/2, "
    "MC0/MC1/MC2 and a validity verdict.\n"
    "  wt = ((w0, w1) of phase 0, of phase 1, of phase 2); (w, w) in "
    "every phase = v2's uniform partition, and then\n"
    "  mlp_header(wt) / tail_header(wt) are byte-identical to "
    "bend/MLP_M16_SCHED_TABLE.bend / TAIL_M16_SCHED_TABLE.bend\n"
    "  (checked by `m16_wsched_ref.py check <quant dir>` against the "
    "baked headers).\n"
    "  flat(wt, kind) -> the exact uint16 array the C++ port "
    "(quant/exl3_m16_wsched.cpp) uploads, for cross-checking.\n"
    "\n"
    "CLI: m16_wsched_ref.py check QUANT_DIR | m16_wsched_ref.py flat "
    "KIND P0W0 P0W1 P1W0 P1W1 P2W0 P2W1 [KQ0 KQ2] |\n"
    "     m16_wsched_ref.py summary P0W0 P0W1 P1W0 P1W1 P2W0 P2W1 "
    "[KQ0 KQ2]   (ext 8207: KQ = k-tiles per iteration, 1 or 4)\n"
)

G, NSM, PF, GW, W = 164, 82, 8, 512, 8
K0, N0 = 6144, 5120  # tail phase 0 (o_proj)
K1, N1 = 5120, 17408  # 8201 phase 1 (gate, up)
K2, N2 = 17408, 5120  # 8201 phase 2 (down)
KT0, NG0 = K0 // 16, N0 // GW
KT1, NG1 = K1 // 16, N1 // GW
KT2, NG2 = K2 // 16, N2 // GW
T0, T1, T2 = NG0 * KT0, 2 * NG1 * KT1, NG2 * KT2
NP, PKT, NCH = NG1, 32, N0 // 128
UNIFORM = ((1, 1), (1, 1), (1, 1))
U16_MAX = 0xFFFF
FLAT_KQ0_ARGC = 10  # argv length with the optional KQ0 of `flat`
FLAT_KQ2_ARGC = 11
SUMMARY_KQ0_ARGC = 9
SUMMARY_KQ2_ARGC = 10
KQ_ALLOWED = {1, 2, 4}

Weights = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


class Table(TypedDict, total=False):
    """Schedule tables of one weight set (only valid / errors when invalid)."""

    valid: bool
    errors: list[str]
    block: list[list[int]]
    pair: list[list[int]]
    waits: list[list[int]]
    tail: list[list[int]]
    fc0: list[int]
    nc0: list[int]
    fc1: list[int]
    nc1: list[int]
    fc2: list[int]
    nc2: list[int]
    mc0: int
    mc1: int
    mc2: int
    nwait: int
    kq0: int
    kq2: int


def _grid(w: tuple[int, int]) -> wpart.Grid:
    return wpart.Grid(G, NSM, w[0], w[1])


def start(b: int, total: int, w: tuple[int, int]) -> int:
    """Return the first iteration of block b.

    Returns:
        The weighted start.

    """
    return wpart.start(b, total, _grid(w))


def owner(x: int, total: int, w: tuple[int, int]) -> int:
    """Return the block owning iteration x.

    Returns:
        The owning block.

    """
    return wpart.owner_closed(x, total, _grid(w))


def groups_fc_nc(
    ngroups: int, kt: int, total: int, w: tuple[int, int]
) -> tuple[list[int], list[int]]:
    """Return the first contributor and contributor count of each group.

    Returns:
        (fc, nc).

    """
    fc = [owner(g * kt, total, w) for g in range(ngroups)]
    nc = [owner((g + 1) * kt - 1, total, w) - fc[g] + 1 for g in range(ngroups)]
    return fc, nc


# ext 8207: kq0 / kq2 = k-tiles per block iteration in phase 0 / phase 2
# (1 = 8205's 512-column groups; 4 = 128-column groups, one iteration =
# (group, 4 consecutive k-tiles), same 4 KB per iteration, so T0 / T2 and every
# slice (s, n) are unchanged; only the group decomposition, the down pair of an
# iteration and the tables derived from them change). Phase 1 always keeps
# 512-column groups.
def shape(kq0: int, kq2: int) -> dict[str, int]:
    """Return the group shape of phases 0 and 2.

    Returns:
        ng0, kt0, ng2, kt2 and pkt.

    Raises:
        AssertionError: kq0 or kq2 is not 1, 2 or 4.

    """
    if kq0 not in KQ_ALLOWED:
        raise AssertionError
    if kq2 not in KQ_ALLOWED:
        raise AssertionError
    return {
        "ng0": N0 * kq0 // GW,
        "kt0": KT0 // kq0,
        "ng2": N2 * kq2 // GW,
        "kt2": KT2 // kq2,
        "pkt": PKT // kq2,
    }


def _slices(total: int, w: tuple[int, int]) -> tuple[list[int], list[int]]:
    s = [start(b, total, w) for b in range(G + 1)]
    return s, [s[b + 1] - s[b] for b in range(G)]


def _finishers(
    s2: list[int], n2: list[int], kt2: int, pkt: int
) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    fin_pair = [(s2[b] % kt2) // pkt for b in range(G)]
    fin_rank = [
        sum(1 for c in range(b) if fin_pair[c] == fin_pair[b]) for b in range(G)
    ]
    fin_count = [sum(1 for b in range(G) if fin_pair[b] == p) for p in range(NP)]
    waits = []
    for b in range(G):
        seen: list[int] = []
        for it in range(s2[b], s2[b] + n2[b]):
            p = (it % kt2) // pkt
            if p not in seen:
                seen.append(p)
        waits.append(seen)
    return fin_pair, fin_rank, fin_count, waits


def _mlp_rows(
    sn1: tuple[list[int], list[int]],
    sn2: tuple[list[int], list[int]],
    fin: tuple[list[int], list[int], list[int], list[list[int]]],
    nc1: list[int],
) -> tuple[list[list[int]], list[list[int]], int]:
    (s1, n1), (s2, n2) = sn1, sn2
    fin_pair, fin_rank, fin_count, waits = fin
    block: list[list[int]] = []
    off = 0
    for b in range(G):
        block.append([
            s1[b],
            n1[b],
            b,
            s2[b],
            n2[b],
            n1[b] % PF,
            fin_pair[b],
            fin_rank[b],
            fin_count[fin_pair[b]],
            off,
            len(waits[b]),
        ])
        off += len(waits[b])
    pair = [
        [
            8 * nc1[p],
            8 * nc1[NP + p],
            8 * fin_count[p],
            8 * fin_count[p],
            sum(1 for b in range(G) if p in waits[b]),
        ]
        for p in range(NP)
    ]
    return block, pair, off


def _domain_errors(wt: Weights) -> list[str]:
    return [
        f"{name}: weights {w} outside the domain (empty slice)"
        for name, total, w in (("p0", T0, wt[0]), ("p1", T1, wt[1]), ("p2", T2, wt[2]))
        if not wpart.valid(total, _grid(w))
    ]


def tables(wt: Weights, kq0: int = 1, kq2: int = 1) -> Table:
    """Build every schedule table for the per-phase weights wt.

    Returns:
        The tables, MC0/MC1/MC2 and the validity verdict.

    Raises:
        AssertionError: the shape does not cover T0 / T2.

    """
    sh = shape(kq0, kq2)
    if sh["ng0"] * sh["kt0"] != T0:
        raise AssertionError
    if sh["ng2"] * sh["kt2"] != T2:
        raise AssertionError
    errs = _domain_errors(wt)
    if errs:
        return {"valid": False, "errors": errs}
    s0, n0 = _slices(T0, wt[0])
    sn1 = _slices(T1, wt[1])
    sn2 = _slices(T2, wt[2])
    fin = _finishers(*sn2, sh["kt2"], sh["pkt"])
    t: Table = {"valid": True, "errors": errs, "waits": fin[3]}
    t["fc0"], t["nc0"] = groups_fc_nc(sh["ng0"], sh["kt0"], T0, wt[0])
    t["fc1"], t["nc1"] = groups_fc_nc(2 * NG1, KT1, T1, wt[1])
    t["fc2"], t["nc2"] = groups_fc_nc(sh["ng2"], sh["kt2"], T2, wt[2])
    t["block"], t["pair"], t["nwait"] = _mlp_rows(sn1, sn2, fin, t["nc1"])
    n1 = sn1[1]
    t["tail"] = [[s0[b], n0[b], n0[b] % PF, (n0[b] + n1[b]) % PF] for b in range(G)]
    errs.extend(_value_errors(t, fin[2]))
    t["valid"] = not errs
    t["mc0"], t["mc1"], t["mc2"] = max(t["nc0"]), max(t["nc1"]), max(t["nc2"])
    t["kq0"], t["kq2"] = kq0, kq2
    return t


def _value_errors(t: Table, fin_count: list[int]) -> list[str]:
    errs = []
    if min(fin_count) < 1:
        errs.append(
            "pairs without a finisher block: "
            f"{[p for p in range(NP) if fin_count[p] == 0]}"
        )
    flat_all = (
        [x for row in t["block"] for x in row]
        + [x for row in t["pair"] for x in row]
        + [x for row in t["waits"] for x in row]
    )
    if max(flat_all + [x for r in t["tail"] for x in r]) > U16_MAX:
        errs.append("table value exceeds 16 bits")
    return errs


def cells(xs: list[int]) -> str:
    """Return xs as C initializer cells.

    Returns:
        "x0,x1,...,".

    """
    return "".join(f"{x}," for x in xs)


def _define(prefix: str, n: str, v: object) -> str:
    return f"#define {prefix}_{n} {v}\n"


def mlp_header(t: Table) -> str:
    """Return the 8201 header text of tables t.

    Returns:
        The header, as bend/MLP_M16_SCHED_TABLE.bend bakes it.

    """
    d = "EXL3_MLP_SCHED"
    return (
        "// Generated by bend/MLP_M16_SCHED_TABLE.bend "
        "(ext 8201 fused MLP schedule); do not edit.\n"
        + _define(d, "G", G)
        + _define(d, "PF", PF)
        + _define(d, "K1", K1)
        + _define(d, "N1", N1)
        + _define(d, "K2", K2)
        + _define(d, "N2", N2)
        + _define(d, "PAIRS", NP)
        + _define(d, "BF", 11)
        + _define(d, "PFLD", 5)
        + _define(d, "NWAIT", t["nwait"])
        + _define(d, "MC1", t["mc1"])
        + _define(d, "MC2", t["mc2"])
        + f"static const unsigned short exl3_mlp_sched_block[{G} * 11] = {{\n"
        + "".join(cells(r) + "\n" for r in t["block"])
        + "};\n"
        + f"static const unsigned short exl3_mlp_sched_pair[{NP} * 5] = {{\n"
        + "".join(cells(r) + "\n" for r in t["pair"])
        + "};\n"
        + f"static const unsigned short exl3_mlp_sched_wait[{t['nwait']}] = {{\n"
        + "".join(cells(r) + "\n" for r in t["waits"])
        + "};\n"
    )


def tail_header(t: Table) -> str:
    """Return the 8202 tail header text of tables t.

    Returns:
        The header, as bend/TAIL_M16_SCHED_TABLE.bend bakes it.

    """
    d = "EXL3_TAIL_SCHED"
    return (
        "// Generated by bend/TAIL_M16_SCHED_TABLE.bend "
        "(ext 8202 layer tail schedule); do not edit.\n"
        + _define(d, "G", G)
        + _define(d, "PF", PF)
        + _define(d, "K0", K0)
        + _define(d, "N0", N0)
        + _define(d, "NG0", NG0)
        + _define(d, "KT0", KT0)
        + _define(d, "T0", T0)
        + _define(d, "MC0", t["mc0"])
        + _define(d, "NCH", NCH)
        + _define(d, "BF", 4)
        + f"static const unsigned short exl3_tail_sched_block[{G} * 4] = {{\n"
        + "".join(cells(r) + "\n" for r in t["tail"])
        + "};\n"
    )


# Layout of the device table the C++ port builds (exl3_m16_wsched.h), uint16:
#   header[8]: mc0, mc1, mc2, nwait, kind, kq0 - 1, kq2 - 1, 0
#              (0, 0 = 8205's 512-column tables)
#   mlp block[G * 11] | pair[NP * 5] | wait[nwait]
#   grp1: fc1[2 NG1] nc1[2 NG1] | grp2: fc2[NG2] nc2[NG2]
#   kind 1 (tail) only: tail block[G * 4] | grp0: fc0[NG0] nc0[NG0]
#   (NG0 / NG2 of the kq shape)
def flat(wt: Weights, kind: int, kq0: int = 1, kq2: int = 1) -> list[int]:
    """Return the uint16 array the C++ port uploads.

    Returns:
        The flat table.

    Raises:
        AssertionError: the weights are invalid (message: the errors).

    """
    if not kind:
        kq0 = 1
    t = tables(wt, kq0, kq2)
    if not t["valid"]:
        raise AssertionError(t["errors"])
    out = [
        t["mc0"] if kind else 0,
        t["mc1"],
        t["mc2"],
        t["nwait"],
        kind,
        kq0 - 1,
        kq2 - 1,
        0,
    ]
    out += (
        [x for r in t["block"] for x in r]
        + [x for r in t["pair"] for x in r]
        + [x for r in t["waits"] for x in r]
    )
    out += t["fc1"] + t["nc1"] + t["fc2"] + t["nc2"]
    if kind:
        out += [x for r in t["tail"] for x in r] + t["fc0"] + t["nc0"]
    return out


def _weights(v: list[int]) -> Weights:
    return ((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))


def _check(q: str) -> None:
    t = tables(UNIFORM)
    ok = True
    for name, text in (
        ("exl3_mlp_m16_sched.h", mlp_header(t)),
        ("exl3_tail_m16_sched.h", tail_header(t)),
    ):
        baked = (Path(q) / name).read_text(encoding="utf-8")
        same = baked == text
        ok = ok and same
        sys.stdout.write(
            f"{name}: generator at uniform weights == baked (Bend) header "
            f"byte for byte: {same}\n"
        )
    sys.exit(0 if ok else 1)


def _span(xs: list[int]) -> str:
    return f"{min(xs)}-{max(xs)}"


def _summary() -> None:
    v = [int(x) for x in sys.argv[2:8]]
    kq0 = int(sys.argv[8]) if len(sys.argv) > SUMMARY_KQ0_ARGC - 1 else 1
    kq2 = int(sys.argv[9]) if len(sys.argv) > SUMMARY_KQ2_ARGC - 1 else 1
    t = tables(_weights(v), kq0, kq2)
    if not t["valid"]:
        sys.stdout.write(f"INVALID {t['errors']}\n")
        sys.exit(1)
    n0 = [r[1] for r in t["tail"]]
    n1 = [r[1] for r in t["block"]]
    n2 = [r[4] for r in t["block"]]
    nw = [len(w) for w in t["waits"]]
    fin = [r[2] // 8 for r in t["pair"]]
    sys.stdout.write(
        f"kq {kq0},{kq2} mc0 {t['mc0']} mc1 {t['mc1']} mc2 {t['mc2']} "
        f"nwait {t['nwait']} (per block {_span(nw)}) | "
        f"n0 {_span(n0[:NSM])} / {_span(n0[NSM:])} | "
        f"n1 {_span(n1[:NSM])} / {_span(n1[NSM:])} | "
        f"n2 {_span(n2[:NSM])} / {_span(n2[NSM:])} | fin_cnt {_span(fin)}\n"
    )


def main() -> None:
    """Run the CLI (see USAGE)."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        _check(sys.argv[2])
    if cmd == "flat":
        kind = int(sys.argv[2])
        v = [int(x) for x in sys.argv[3:9]]
        kq0 = int(sys.argv[9]) if len(sys.argv) > FLAT_KQ0_ARGC - 1 else 1
        kq2 = int(sys.argv[10]) if len(sys.argv) > FLAT_KQ2_ARGC - 1 else 1
        values = flat(_weights(v), kind, kq0, kq2)
        sys.stdout.write(",".join(str(x) for x in values) + "\n")
        return
    if cmd == "summary":
        _summary()
        return
    sys.exit(USAGE)


if __name__ == "__main__":
    main()
