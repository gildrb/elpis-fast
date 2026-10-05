# Copyright (c) 2026 Gil Rodrigues
"""Shared slot-weighted split-K partition: reference and header generator.

Usage text: USAGE (printed on bad arguments).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrand

USAGE = (
    "Shared slot-weighted split-K partition (m16g fix and 8205). "
    "Reference + header generator.\n"
    "\n"
    "Origin: the M16gEff work copied this file (wpart.py) into the "
    "repo without change of logic.\n"
    "Role: bend/gen/m16_wsched_ref.py imports it for the partition "
    "formulas.\n"
    "\n"
    "Blocks b < n_sm have weight w0, blocks b >= n_sm have weight w1 "
    "(integers >= 1); G = grid (n_sm <= G <= 2 n_sm):\n"
    "    W(b)     = w0 * min(b, n_sm) + w1 * max(b - n_sm, 0)\n"
    "    start(b) = (total * W(b)) // W(G)                      "
    "(int64; (1, 1) is v2's b * total // G)\n"
    "    owner(x) = max{b : start(b) <= x}                       "
    "(block owning iteration x, 0 <= x < total)\n"
    "Domain: w0, w1 >= 1; every slice nonempty, i.e. start(b + 1) > "
    "start(b) for all b < G (checked by `valid`).\n"
    "owner in closed form (what the CUDA side implements): with s1 = "
    "start(n_sm),\n"
    "    x < s1:  b = ((x + 1) * W(G) + total - 1) // total - 1 over "
    "weights w0      -> b = (ceil((x+1) * WG / total) - 1) // w0\n"
    "    x >= s1: same for the w1 segment, offset by n_sm.\n"
    "`owner_closed` is that expression; `self_test` checks it "
    "against the definition.\n"
    "\n"
    "    python3 bend/gen/wpart.py self-test\n"
    "    python3 bend/gen/wpart.py header NAME total G n_sm w0 w1    "
    "  # prints #defines (starts are recomputed on the device)\n"
)
SAMPLE = 4000  # self-test: iterations checked per configuration (all when fewer)


class Grid(NamedTuple):
    """Grid G, SM count n_sm and the block weights w0 (b < n_sm) / w1 (b >= n_sm)."""

    grid: int
    n_sm: int
    w0: int
    w1: int


def weight(b: int, g: Grid) -> int:
    """Return W(b), the weight prefix of blocks 0 .. b - 1.

    Returns:
        The weight prefix.

    """
    return g.w0 * min(b, g.n_sm) + g.w1 * max(b - g.n_sm, 0)


def start(b: int, total: int, g: Grid) -> int:
    """Return the first iteration of block b.

    Returns:
        start(b) = total * W(b) // W(G).

    """
    return total * weight(b, g) // weight(g.grid, g)


def owner(x: int, total: int, g: Grid) -> int:
    """Return the block owning iteration x, by binary search on start.

    Returns:
        max{b : start(b) <= x}.

    """
    lo, hi = 0, g.grid - 1  # max b with start(b) <= x (start monotone, start(0) = 0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if start(mid, total, g) <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def owner_closed(x: int, total: int, g: Grid) -> int:
    """Return the block owning iteration x, in closed form.

    Returns:
        The owning block.

    """
    wg = weight(g.grid, g)
    # the smallest weight prefix P with total * P // WG > x is
    # ((x + 1) * WG + total - 1) // total; the owner is the block
    # whose prefix range [W(b), W(b + 1)) contains P - 1
    p = ((x + 1) * wg + total - 1) // total - 1
    if g.w0 * g.n_sm > p:
        return p // g.w0
    return g.n_sm + (p - g.w0 * g.n_sm) // g.w1


def valid(total: int, g: Grid) -> bool:
    """Check the domain: positive weights, n_sm <= G <= 2 n_sm, nonempty slices.

    Returns:
        True when the configuration is in the domain.

    """
    return (
        g.w0 >= 1
        and g.w1 >= 1
        and g.n_sm <= g.grid <= 2 * g.n_sm
        and all(start(b + 1, total, g) > start(b, total, g) for b in range(g.grid))
    )


def _check(total: int, g: Grid, rnd: detrand.SplitMix64) -> None:
    """Check one valid configuration against the definitions.

    Raises:
        AssertionError: a property fails.

    """
    if start(0, total, g) != 0:
        raise AssertionError
    if start(g.grid, total, g) != total:
        raise AssertionError
    if g.w0 == g.w1 and not all(
        start(b, total, g) == b * total // g.grid for b in range(g.grid + 1)
    ):
        msg = "(w, w) must be v2"
        raise AssertionError(msg)
    xs = range(total) if total <= SAMPLE else rnd.sample(range(total), SAMPLE)
    for x in xs:
        b = owner_closed(x, total, g)
        if b != owner(x, total, g):
            msg = (total, g.grid, g.n_sm, g.w0, g.w1, x)
            raise AssertionError(msg)
        if not start(b, total, g) <= x < start(b + 1, total, g):
            raise AssertionError


def self_test() -> None:
    """Check owner_closed and the (w, w) == v2 identity on fixed and random cases."""
    rnd = detrand.SplitMix64(8205)
    cases = [
        (32 * 320, Grid(164, 82, 100, 93)),
        (28 * 320, Grid(164, 82, 100, 93)),
        (10 * 384, Grid(164, 82, 100, 80)),
        (34 * 320 * 2, Grid(164, 82, 100, 97)),
        (32 * 320, Grid(164, 82, 1, 1)),
        (32 * 320, Grid(82, 82, 1, 1)),
    ]
    for _ in range(300):
        n_sm = rnd.randint(1, 100)
        grid = rnd.randint(n_sm, 2 * n_sm)
        total = rnd.randint(grid, 20000)
        cases.append((
            total,
            Grid(grid, n_sm, rnd.randint(1, 120), rnd.randint(1, 120)),
        ))
    n = 0
    for total, g in cases:
        if not valid(total, g):
            continue
        n += 1
        _check(total, g, rnd)
    sys.stdout.write(
        f"self-test OK: {n} valid configurations; "
        "owner_closed == max{b: start(b) <= x}; (w, w) == v2\n"
    )


def header(name: str, total: int, g: Grid) -> str:
    """Return the #define header of partition name.

    Returns:
        The header text.

    Raises:
        AssertionError: the weights are outside the domain.

    """
    if not valid(total, g):
        msg = "weights outside the domain (a slice would be empty)"
        raise AssertionError(msg)
    return (
        f"// Generated by wpart.py; slot-weighted partition {name}: "
        f"total {total}, G {g.grid}, n_sm {g.n_sm}\n"
        f"#define {name}_W0 {g.w0}\n#define {name}_W1 {g.w1}\n"
        f"#define {name}_NSM {g.n_sm}\n"
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["self-test"]:
        self_test()
    elif sys.argv[1:2] == ["header"]:
        total, *rest = map(int, sys.argv[3:8])
        sys.stdout.write(header(sys.argv[2], total, Grid(*rest)))
    else:
        sys.exit(USAGE)
