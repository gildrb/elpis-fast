#!/usr/bin/env python3
"""Reference generator of the 8201 fused-MLP and 8202 layer-tail schedule tables under the shared slot-weighted
split-K partition (ext 8205; partition formulas: bend/gen/wpart.py, imported). ONE generator for both kernels.

Origin: the TailV2b work copied this file (gen_sched.py) into the repo without change of logic. It imports
wpart from its own directory only. Role: bend/m16_wsched_diff.py compares flat() with the Bend emitter.

  tables(wt) -> dict with the 8201 block/pair/wait tables, the tail block table, the per-group first-contributor /
                contributor-count tables of phases 0/1/2, MC0/MC1/MC2 and a validity verdict.
  wt = ((w0, w1) of phase 0, of phase 1, of phase 2); (w, w) in every phase = v2's uniform partition, and then
  mlp_header(wt) / tail_header(wt) are byte-identical to bend/MLP_M16_SCHED_TABLE.bend / TAIL_M16_SCHED_TABLE.bend
  (checked by `m16_wsched_ref.py check <quant dir>` against the baked headers).
  flat(wt, kind) -> the exact uint16 array the C++ port (quant/exl3_m16_wsched.cpp) uploads, for cross-checking.

CLI: m16_wsched_ref.py check QUANT_DIR | m16_wsched_ref.py flat KIND P0W0 P0W1 P1W0 P1W1 P2W0 P2W1 [KQ0 KQ2] |
     m16_wsched_ref.py summary P0W0 P0W1 P1W0 P1W1 P2W0 P2W1 [KQ0 KQ2]   (ext 8207: KQ = k-tiles per iteration, 1 or 4)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wpart  # noqa: E402

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


def start(b, total, w):
    return wpart.start(b, total, G, NSM, w[0], w[1])


def owner(x, total, w):
    return wpart.owner_closed(x, total, G, NSM, w[0], w[1])


def groups_fc_nc(ngroups, KT, total, w):
    fc = [owner(g * KT, total, w) for g in range(ngroups)]
    nc = [owner((g + 1) * KT - 1, total, w) - fc[g] + 1 for g in range(ngroups)]
    return fc, nc


# ext 8207: kq0 / kq2 = k-tiles per block iteration in phase 0 / phase 2 (1 = 8205's 512-column groups; 4 =
# 128-column groups, one iteration = (group, 4 consecutive k-tiles), same 4 KB per iteration, so T0 / T2 and every
# slice (s, n) are unchanged; only the group decomposition, the down pair of an iteration and the tables derived
# from them change). Phase 1 always keeps 512-column groups.
def shape(kq0, kq2):
    assert kq0 in (1, 2, 4) and kq2 in (1, 2, 4)
    return {
        "ng0": N0 * kq0 // GW,
        "kt0": KT0 // kq0,
        "ng2": N2 * kq2 // GW,
        "kt2": KT2 // kq2,
        "pkt": PKT // kq2,
    }


def tables(wt, kq0=1, kq2=1):
    sh = shape(kq0, kq2)
    kt0, ng0, kt2, ng2, pkt = sh["kt0"], sh["ng0"], sh["kt2"], sh["ng2"], sh["pkt"]
    assert ng0 * kt0 == T0 and ng2 * kt2 == T2
    w0, w1, w2 = wt
    errs = []
    for name, total, w in (("p0", T0, w0), ("p1", T1, w1), ("p2", T2, w2)):
        if not wpart.valid(total, G, NSM, w[0], w[1]):
            errs.append(f"{name}: weights {w} outside the domain (empty slice)")
    if errs:
        return {"valid": False, "errors": errs}
    s0 = [start(b, T0, w0) for b in range(G + 1)]
    s1 = [start(b, T1, w1) for b in range(G + 1)]
    s2 = [start(b, T2, w2) for b in range(G + 1)]
    n0 = [s0[b + 1] - s0[b] for b in range(G)]
    n1 = [s1[b + 1] - s1[b] for b in range(G)]
    n2 = [s2[b + 1] - s2[b] for b in range(G)]
    fin_pair = [(s2[b] % kt2) // pkt for b in range(G)]
    fin_rank = [
        sum(1 for c in range(b) if fin_pair[c] == fin_pair[b]) for b in range(G)
    ]
    fin_count = [sum(1 for b in range(G) if fin_pair[b] == p) for p in range(NP)]
    waits = []
    for b in range(G):
        seen = []
        for it in range(s2[b], s2[b] + n2[b]):
            p = (it % kt2) // pkt
            if p not in seen:
                seen.append(p)
        waits.append(seen)
    fc0, nc0 = groups_fc_nc(ng0, kt0, T0, w0)
    fc1, nc1 = groups_fc_nc(2 * NG1, KT1, T1, w1)
    fc2, nc2 = groups_fc_nc(ng2, kt2, T2, w2)
    block, off = [], 0
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
    tail = [[s0[b], n0[b], n0[b] % PF, (n0[b] + n1[b]) % PF] for b in range(G)]
    if min(fin_count) < 1:
        errs.append(
            f"pairs without a finisher block: {[p for p in range(NP) if fin_count[p] == 0]}"
        )
    flat_all = (
        [x for row in block for x in row]
        + [x for row in pair for x in row]
        + [x for l in waits for x in l]
    )
    if max(flat_all + [x for r in tail for x in r]) > 0xFFFF:
        errs.append("table value exceeds 16 bits")
    return {
        "valid": not errs,
        "errors": errs,
        "block": block,
        "pair": pair,
        "waits": waits,
        "tail": tail,
        "fc0": fc0,
        "nc0": nc0,
        "fc1": fc1,
        "nc1": nc1,
        "fc2": fc2,
        "nc2": nc2,
        "mc0": max(nc0),
        "mc1": max(nc1),
        "mc2": max(nc2),
        "nwait": off,
        "kq0": kq0,
        "kq2": kq2,
    }


def cells(xs):
    return "".join(f"{x}," for x in xs)


def mlp_header(t):
    d = lambda n, v: f"#define EXL3_MLP_SCHED_{n} {v}\n"
    return (
        "// Generated by bend/MLP_M16_SCHED_TABLE.bend (ext 8201 fused MLP schedule); do not edit.\n"
        + d("G", G)
        + d("PF", PF)
        + d("K1", K1)
        + d("N1", N1)
        + d("K2", K2)
        + d("N2", N2)
        + d("PAIRS", NP)
        + d("BF", 11)
        + d("PFLD", 5)
        + d("NWAIT", t["nwait"])
        + d("MC1", t["mc1"])
        + d("MC2", t["mc2"])
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


def tail_header(t):
    d = lambda n, v: f"#define EXL3_TAIL_SCHED_{n} {v}\n"
    return (
        "// Generated by bend/TAIL_M16_SCHED_TABLE.bend (ext 8202 layer tail schedule); do not edit.\n"
        + d("G", G)
        + d("PF", PF)
        + d("K0", K0)
        + d("N0", N0)
        + d("NG0", NG0)
        + d("KT0", KT0)
        + d("T0", T0)
        + d("MC0", t["mc0"])
        + d("NCH", NCH)
        + d("BF", 4)
        + f"static const unsigned short exl3_tail_sched_block[{G} * 4] = {{\n"
        + "".join(cells(r) + "\n" for r in t["tail"])
        + "};\n"
    )


# Layout of the device table the C++ port builds (exl3_m16_wsched.h), uint16:
#   header[8]: mc0, mc1, mc2, nwait, kind, kq0 - 1, kq2 - 1, 0     (0, 0 = 8205's 512-column tables)
#   mlp block[G * 11] | pair[NP * 5] | wait[nwait]
#   grp1: fc1[2 NG1] nc1[2 NG1] | grp2: fc2[NG2] nc2[NG2]
#   kind 1 (tail) only: tail block[G * 4] | grp0: fc0[NG0] nc0[NG0]      (NG0 / NG2 of the kq shape)
def flat(wt, kind, kq0=1, kq2=1):
    if not kind:
        kq0 = 1
    t = tables(wt, kq0, kq2)
    assert t["valid"], t["errors"]
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
        + [x for l in t["waits"] for x in l]
    )
    out += t["fc1"] + t["nc1"] + t["fc2"] + t["nc2"]
    if kind:
        out += [x for r in t["tail"] for x in r] + t["fc0"] + t["nc0"]
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        q = sys.argv[2]
        t = tables(UNIFORM)
        ok = True
        for name, text in (
            ("exl3_mlp_m16_sched.h", mlp_header(t)),
            ("exl3_tail_m16_sched.h", tail_header(t)),
        ):
            baked = open(os.path.join(q, name)).read()
            same = baked == text
            ok = ok and same
            print(
                f"{name}: generator at uniform weights == baked (Bend) header byte for byte: {same}"
            )
        sys.exit(0 if ok else 1)
    if cmd == "flat":
        kind = int(sys.argv[2])
        v = [int(x) for x in sys.argv[3:9]]
        kq0 = int(sys.argv[9]) if len(sys.argv) > 9 else 1
        kq2 = int(sys.argv[10]) if len(sys.argv) > 10 else 1
        print(
            ",".join(
                str(x)
                for x in flat(
                    ((v[0], v[1]), (v[2], v[3]), (v[4], v[5])), kind, kq0, kq2
                )
            )
        )
        return
    if cmd == "summary":
        v = [int(x) for x in sys.argv[2:8]]
        kq0 = int(sys.argv[8]) if len(sys.argv) > 8 else 1
        kq2 = int(sys.argv[9]) if len(sys.argv) > 9 else 1
        t = tables(((v[0], v[1]), (v[2], v[3]), (v[4], v[5])), kq0, kq2)
        if not t["valid"]:
            print("INVALID", t["errors"])
            sys.exit(1)
        n0 = [r[1] for r in t["tail"]]
        n1 = [r[1] for r in t["block"]]
        n2 = [r[4] for r in t["block"]]
        nw = [len(w) for w in t["waits"]]
        print(
            f"kq {kq0},{kq2} mc0 {t['mc0']} mc1 {t['mc1']} mc2 {t['mc2']} nwait {t['nwait']} (per block {min(nw)}-{max(nw)}) | "
            f"n0 {min(n0[:NSM])}-{max(n0[:NSM])} / "
            f"{min(n0[NSM:])}-{max(n0[NSM:])} | n1 {min(n1[:NSM])}-{max(n1[:NSM])} / {min(n1[NSM:])}-{max(n1[NSM:])} | "
            f"n2 {min(n2[:NSM])}-{max(n2[:NSM])} / {min(n2[NSM:])}-{max(n2[NSM:])} | fin_cnt "
            f"{min(r[2] for r in t['pair']) // 8}-{max(r[2] for r in t['pair']) // 8}"
        )
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
