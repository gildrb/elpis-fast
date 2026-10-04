#!/usr/bin/env python3
"""
Finite differential check of bend/gdn_ba_ksplit.bend (ext 5106: the k-split b/a GEMV of
gdn_conv_rule_norm_kernel<..., KS, STAMP>) against the shipped kernel source.

Quoted verbatim from the patched engine tree (argument = the exllamav3 package directory),
exllamav3_ext/gdn.cu:
  - the constants GR_KS_ITERS, GR_KS_ROWS, GR_WARPS, GR_KS_DEFAULT and the kernel's THREADS;
    warp_id / lane_id; k2
  - ks_load: the i loop header, the element index `const int j = ...` (also in the FMA loop: both
    must be the same expression), the `if (j < k2)` guard, the ba_w row bases (b row h, a row H + h),
    the x index
  - the FMA loop: the i and s loop headers, `if (s < S)`, the four fmaf lines (their order and the
    x / w components they pair)
  - the reduction: the offset loop header, the two __shfl_down_sync lines, `if (lane_id == 0)`, the two
    ks_part stores (write slots)
  - the thread-t read: `bv = 0.0f; av = 0.0f;`, the w loop header, the two ks_part loads (read slots),
    the two bias lines
  - the launcher's fallback line `if (S > GR_KS_ROWS || k / 2 > 512 * GR_KS_ITERS) ks = 0;`
  - the block event order: the positions of the `if constexpr (KS ...)` state / ks_load lines, the
    norm-operand prefetch, the conv, the GEMV branches, the first __syncthreads() after them, the
    conv_sync arrival, the v-window write-back, the b/a read and gdn_rule_tokens (whose own l2-norm /
    __syncthreads / m.dot2 order is quoted from its body); no other m.dot2 / ks_part store between
    the partial stores and the token loop
Python evaluates the quoted index expressions and re-runs the quoted loops symbolically (an fp32
value is the tree of the fmaf / add operations that produced it; shfl_down lanes past 31 read their
own value, as on the GPU). The result is printed in GDN_BA_KSPLIT_TABLE.bend's format and compared
byte for byte with that table's output (compiled with the pinned bend).

Differential evidence on finite instances, not a proof. `--mutate NAME` applies a deliberate source
mutation that the check must reject.

Usage: python3 bend/gdn_ba_ksplit_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
  ENGINE_PACKAGE_DIR: OUT/patched of bend/engine_trees.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
TABLE = "bend/GDN_BA_KSPLIT_TABLE.bend"

# name: (original, replacement, occurrences)
MUTATIONS = {
    "stride256": ("lane_id + i * THREADS;", "lane_id + i * 256;", 2),
    "warps_reversed": ("for (int w = 0; w < GR_WARPS; ++w)", "for (int w = GR_WARPS - 1; w >= 0; --w)", 1),
    "no_barrier": ("        gdn_wstamp<STAMP>(wstamps, 2, pb[0] + pa[0]);\n    }\n    __syncthreads();\n",
                   "        gdn_wstamp<STAMP>(wstamps, 2, pb[0] + pa[0]);\n    }\n", 1),
    "fallback16": ("if (S > GR_KS_ROWS || k / 2 > 512 * GR_KS_ITERS) ks = 0;",
                   "if (S > 16 || k / 2 > 512 * GR_KS_ITERS) ks = 0;", 1),
    "slot_collide": ("ks_part[warp_id * 2 * GR_KS_ROWS + GR_KS_ROWS + s] = pa[s];",
                     "ks_part[warp_id * 2 * GR_KS_ROWS + s] = pa[s];", 1),
    "fma_order": ("pb[s] = fmaf(xf.x, wbf.x, pb[s]);\n                        pb[s] = fmaf(xf.y, wbf.y, pb[s]);",
                  "pb[s] = fmaf(xf.y, wbf.y, pb[s]);\n                        pb[s] = fmaf(xf.x, wbf.x, pb[s]);", 1),
}


def fail(msg: str):
    raise SystemExit(f"gdn_ba_ksplit_diff: {msg}")


def grab(text: str, pattern: str, count: int = 1) -> list:
    m = re.findall(pattern, text)
    if len(m) != count:
        fail(f"pattern {pattern!r}: {len(m)} matches, expected {count}")
    return m


def one(text: str, pattern: str):
    return grab(text, pattern)[0]


def cexpr(expr: str, env: dict) -> int:
    """Evaluate a C integer expression over the non-negative ints of env (only + - * / % and names)."""
    if not re.fullmatch(r"[\w\s+*/%()\-]+", expr):
        fail(f"unexpected C expression {expr!r}")
    py = re.sub(r"(?<![/])/(?![/])", "//", expr)
    return int(eval(py, {"__builtins__": {}}, dict(env)))


def kernel_body(src: str) -> str:
    start = src.index("void gdn_conv_rule_norm_kernel\n")
    end = src.index("\n}\n", start)
    return src[start:end]


# ---- symbolic values, rendered as GDN_BA_KSPLIT_TABLE.bend's `ex`
def show(e) -> str:
    tag = e[0]
    if tag == "0":
        return "0"
    if tag == "F":
        return f"F({e[1]},{e[2]},{e[3]},{show(e[4])})"
    if tag == "A":
        return f"A({show(e[1])},{show(e[2])})"
    if tag == "B":
        return f"B({show(e[1])},{e[2]})"
    if tag == "S":
        return f"S{e[1]}"
    fail(f"bad value {e!r}")


def main(argv: list[str]) -> None:
    mutate = None
    if len(argv) >= 3 and argv[1] == "--mutate":
        mutate = argv[2]
        argv = [argv[0]] + argv[3:]
    if len(argv) != 2:
        fail(__doc__)
    src = (Path(argv[1]) / "exllamav3_ext/gdn.cu").read_text()
    if mutate:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate}; known: {', '.join(MUTATIONS)}")
        a, b, n = MUTATIONS[mutate]
        if src.count(a) != n:
            fail(f"mutation {mutate} does not apply ({src.count(a)} occurrences, expected {n})")
        src = src.replace(a, b)
    body = kernel_body(src)

    # ---- constants
    C = {"GR_KS_ITERS": int(one(src, r"#define GR_KS_ITERS (\d+)\n")),
         "GR_KS_ROWS": int(one(src, r"#define GR_KS_ROWS (\d+)\n")),
         "GR_WARPS": int(one(src, r"#define GR_WARPS (\d+)\n")),
         "THREADS": int(one(body, r"constexpr int THREADS = (\d+);"))}
    ks_default = int(one(src, r"#define GR_KS_DEFAULT (\d+)"))
    warp_e = one(body, r"const int warp_id = (t / \d+);")
    lane_e = one(body, r"const int lane_id = (t % \d+);")
    k2_e = grab(body, r"const int k2 = (k / 2);", 3)[0]   # ks_load, KS 0, KS > 0

    # ---- element index (ks_load and the FMA loop: the same expression)
    jx = grab(body, r"const int j = ([^;]+);\n\s+if \(j < k2\)", 2)
    if jx[0] != jx[1]:
        fail(f"ks_load and the FMA loop index elements differently: {jx}")
    j_e = jx[0]
    iloop = grab(body, r"for \(int i = 0; i < (GR_KS_ITERS); \+\+i\)", 2)
    wb_e = one(body, r"const half2\* wb = \(const half2\*\) \(ba_w \+ \(size_t\) ([^;]+) \* k\);")
    wa_e = one(body, r"const half2\* wa = \(const half2\*\) \(ba_w \+ \(size_t\) ([^;]+) \* k\);")
    x_e = one(body, r"if \(s < S\) ks_x\[s\]\[i\] = x2\[\(size_t\) (s \* k2 \+ j)\];")
    grab(body, r"for \(int s = 0; s < (GR_KS_ROWS); \+\+s\)", 5)
    fmas = grab(body, r"(p[ab])\[s\] = fmaf\(xf\.([xy]), w([ab])f\.([xy]), p[ab]\[s\]\);", 4)
    for acc, xc, wr, wc in fmas:
        if xc != wc or acc[1] != wr:
            fail(f"fmaf pairs x.{xc} with w{wr}.{wc} into {acc}")

    # ---- reduction and slots
    off0 = int(one(body, r"for \(int offset = (\d+); offset > 0; offset >>= 1\)\n\s+\{\n\s+pb\[s\] \+= "))
    shfl = grab(body, r"(p[ab])\[s\] \+= __shfl_down_sync\(0xffffffff, (p[ab])\[s\], offset\);", 2)
    if any(a != b for a, b in shfl) or [a for a, _ in shfl] != ["pb", "pa"]:
        fail(f"shfl lines {shfl}")
    wslots = grab(body, r"if \(lane_id == 0\)\n\s+\{\n\s+ks_part\[([^\]]+)\] = pb\[s\];\n\s+ks_part\[([^\]]+)\] = pa\[s\];", 1)[0]
    grab(body, r"bv = 0\.0f;\n\s+av = 0\.0f;", 1)
    wl = one(body, r"for \(int w = ([^;]+); w (<|>=) ([^;]+); (\+\+w|--w)\)\n\s+\{\n\s+bv \+= ks_part\[")
    rslots = grab(body, r"bv \+= ks_part\[([^\]]+)\];\n\s+av \+= ks_part\[([^\]]+)\];", 1)[0]
    bias = grab(body, r"if \(ba_bias\)\n\s+\{\n\s+bv \+= __half2float\(ba_bias\[([^\]]+)\]\);\n\s+av \+= __half2float\(ba_bias\[([^\]]+)\]\);", 1)[0]
    fb = one(src, r"if \((S > [\w ]+ \|\| k / 2 > [\w *]+)\) ks = 0;")

    # ---- event order (positions in the kernel body)
    def pos(pattern: str) -> int:
        m = list(re.finditer(pattern, body))
        if len(m) != 1:
            fail(f"event pattern {pattern!r}: {len(m)} matches")
        return m[0].start()
    gemv0 = pos(r"if constexpr \(KS == 0\)\n\s+\{\n\s+// One \(row, output\) per warp")
    fma_p = pos(r"pb\[s\] = fmaf\(xf\.[xy], wbf\.[xy], pb\[s\]\);\n\s+pb")
    tree_p = pos(r"pb\[s\] \+= __shfl_down_sync")
    partw_p = pos(r"ks_part\[[^\]]+\] = pb\[s\];")
    read0_p = pos(r"bv = m\.ba\[t\];")
    tok_p = pos(r"gdn_rule_tokens<RULE_VERIFY, 4>\(")
    bar_p = [m.start() for m in re.finditer(r"__syncthreads\(\);", body)
             if max(gemv0, partw_p) < m.start() < read0_p]
    ev_all = [  # (position, event, condition on (ks, owner))
        (pos(r"if constexpr \(KS != 2\) gdn_rule_load_state\("), "state", lambda ks, o: ks != 2),
        (pos(r"if constexpr \(KS == 2\) ks_load\(\);"), "ks_load", lambda ks, o: ks == 2),
        (pos(r"gdn_rule_norm_prefetch\(w4, g4"), "norm_ops", lambda ks, o: True),
        (pos(r"if \(t < 3 \* HD\)"), "conv", lambda ks, o: True),
        (pos(r"if constexpr \(KS == 1\) ks_load\(\);"), "ks_load", lambda ks, o: ks == 1),
        (pos(r"if constexpr \(KS == 2\) gdn_rule_load_state\("), "state", lambda ks, o: ks == 2),
        (gemv0, "gemv", lambda ks, o: ks == 0),
        (fma_p, "ks_fma", lambda ks, o: ks != 0),
        (tree_p, "ks_tree", lambda ks, o: ks != 0),
        (partw_p, "part_w", lambda ks, o: ks != 0),
        *[(p, "bar", lambda ks, o: True) for p in bar_p],
        (pos(r"if \(!owner && t == 0\)\n\s+\{\n\s+__threadfence\(\);\n\s+atomicAdd\(conv_sync"), "arrive", lambda ks, o: not o),
        (pos(r"state_v\[\(size_t\) c \* state_size"), "win_v", lambda ks, o: True),
        (read0_p, "part_r", lambda ks, o: True),
    ]
    rt = src[src.index("void gdn_rule_tokens("):]
    rt = rt[:rt.index("\n}\n")]
    l2, sync1, d2 = rt.find("gdn_rule_l2norm_warp("), rt.find("__syncthreads();"), rt.find("m.dot2[")
    if not (0 <= l2 < sync1 < d2):
        fail("gdn_rule_tokens: l2 norms, __syncthreads, m.dot2 store not in that order")
    if not (read0_p < tok_p):
        fail("gdn_rule_tokens runs before the b/a read")
    # no other store to the partial area between the partial stores and the token loop
    mid = body[partw_p:tok_p]
    stores = re.findall(r"(?:m\.dot2|ks_part)\[[^\]]+\]\s*=(?!=)", mid)
    if len(stores) != 2:
        fail(f"stores to m.dot2 / ks_part between the partial stores and the token loop: {stores}")
    ev_all.sort(key=lambda e: e[0])

    def prog(ks: int, own: bool) -> list[str]:
        return [e for _, e, c in ev_all if c(ks, own)] + ["l2", "bar", "dot2_w"]

    # ---- the table
    env0 = dict(C)
    out = []
    for t in range(512):
        e = dict(env0, t=t)
        e["warp_id"] = cexpr(warp_e, e)
        e["lane_id"] = cexpr(lane_e, e)
        js = [cexpr(j_e, dict(e, i=i)) for i in range(C[iloop[0]])]
        if len(js) != 5:
            fail(f"{len(js)} iterations per thread")
        out.append(f"J {t}:" + ",".join(map(str, js)))
    for q in range(256):
        w, ab, s = q // 16, (q % 16) // 8, q % 8
        e = dict(env0, warp_id=w, w=w, s=s, t=s)
        wslot = cexpr(wslots[ab], e)
        rslot = cexpr(rslots[ab], e)
        out.append(f"W {w} {s} {ab}:{wslot},{rslot}")
    for ks in range(3):
        for S, k in ((1, 5120), (8, 5120), (9, 5120), (16, 5120), (8, 5121), (8, 5122), (1, 2), (3, 6144)):
            cond = fb.replace("||", " or ")
            lhs, rhs = [x.strip() for x in cond.split(" or ")]
            big = cexpr(lhs.split(">")[0], dict(env0, S=S, k=k)) > cexpr(lhs.split(">")[1], dict(env0, S=S, k=k)) or \
                cexpr(rhs.split(">")[0], dict(env0, S=S, k=k)) > cexpr(rhs.split(">")[1], dict(env0, S=S, k=k))
            out.append(f"V {ks} {S} {k}:{0 if big else ks}")
    for ks in range(3):
        out.append(f"P {ks} owner: " + " ".join(prog(ks, True)))
        out.append(f"P {ks} other: " + " ".join(prog(ks, False)))

    def lane_tree(vals: list):
        v = list(vals)
        off = off0
        while off > 0:
            v = [("A", v[l], v[l + off] if l + off < 32 else v[l]) for l in range(32)]
            off >>= 1
        return v[0]
    out.append("L " + show(lane_tree([("S", l) for l in range(32)])))

    def task(S: int, trow: int, ab: int, k2: int, h: int, H: int, has_bias: bool):
        env = dict(env0, h=h, H=H, k=2 * k2)
        if cexpr(k2_e, env) != k2:
            fail("k2")
        # every thread's accumulators, FMA loop as quoted
        part = {}
        for w in range(C["GR_WARPS"]):
            lanes = {0: [], 1: []}
            for l in range(32):
                e = dict(env, t=32 * w + l)
                e["warp_id"] = cexpr(warp_e, e)
                e["lane_id"] = cexpr(lane_e, e)
                p = {("pb", s): ("0",) for s in range(C["GR_KS_ROWS"])}
                p.update({("pa", s): ("0",) for s in range(C["GR_KS_ROWS"])})
                for i in range(C["GR_KS_ITERS"]):
                    j = cexpr(j_e, dict(e, i=i))
                    if j < k2:
                        for s in range(C["GR_KS_ROWS"]):
                            if s < S:
                                x2 = cexpr(x_e, dict(e, s=s, j=j, k2=k2))
                                for acc, comp, wr, _ in fmas:
                                    row = cexpr(wb_e if wr == "b" else wa_e, e)
                                    p[(acc, s)] = ("F", x2, row * k2 + j, 0 if comp == "x" else 1, p[(acc, s)])
                lanes[0].append(p[("pb", trow)] if trow < S else None)
                lanes[1].append(p[("pa", trow)] if trow < S else None)
            for s in range(C["GR_KS_ROWS"]):
                if s < S:
                    e = dict(env, warp_id=w, s=s)
                    for abx, acc in ((0, "pb"), (1, "pa")):
                        if s == trow:
                            part[cexpr(wslots[abx], e)] = lane_tree(lanes[abx])
        e = dict(env, t=trow)
        start, cmp_, stop, step = wl
        w = cexpr(start, e)
        v = ("0",)
        while (w < cexpr(stop, e)) if cmp_ == "<" else (w >= cexpr(stop, e)):
            q = cexpr(rslots[ab], dict(e, w=w))
            v = ("A", v, part.get(q, ("S", q)))
            w = w + 1 if step == "++w" else w - 1
        if has_bias:
            v = ("B", v, cexpr(bias[ab], e))
        return v
    out.append("T 2 1 1 520 1 2 1:" + show(task(2, 1, 1, 520, 1, 2, True)))
    out.append("T 1 0 0 3 0 1 0:" + show(task(1, 0, 0, 3, 0, 1, False)))
    ctext = "\n".join(out) + "\n"

    bend = source_link.bend()
    bres = subprocess.run(source_link.locked([bend, TABLE]), cwd=REPO, capture_output=True, text=True, timeout=1800)
    if bres.returncode != 0:
        fail(f"{bend} {TABLE} exited {bres.returncode}: {bres.stderr.strip()[-500:]}")
    btext = bres.stdout
    if btext.endswith("\n\n"):
        btext = btext[:-1]                  # IO.print's own newline after the table's last "\n"
    same = ctext == btext
    print(f"constants {C}, default KS {ks_default}, fallback `{fb}`")
    print(f"table lines: source {len(ctext.splitlines())}, Bend {len(btext.splitlines())}; byte-identical: {same}")
    if not same:
        for i, (x, y) in enumerate(zip(ctext.splitlines(), btext.splitlines())):
            if x != y:
                print(f"first difference at line {i + 1}:\n  source: {x[:300]}\n  Bend:   {y[:300]}")
                break
        fail("MISMATCH")
    print("gdn_ba_ksplit_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
