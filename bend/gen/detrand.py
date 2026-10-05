# Copyright (c) 2026 Gil Rodrigues
"""Deterministic test-case generator: SplitMix64 with explicit derived draws.

random.Random's derived methods (randint, choice, sample) are not guaranteed to
draw the same values across Python versions, and Ruff (S311) flags it. This
module spells out every step, so a seed gives the same cases on any Python. It
is not a cryptographic generator; the links use it for reproducible test cases
only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    """SplitMix64 (Steele, Lea, Flood 2014) and draws derived from it."""

    def __init__(self, seed: int) -> None:
        """Start the stream at seed (taken modulo 2**64)."""
        self.state: int = seed & MASK64

    def next64(self) -> int:
        """Return the next 64-bit output.

        Returns:
            An integer in [0, 2**64).

        """
        self.state = (self.state + GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Return a uniform integer in [0, n), by rejection (no modulo bias).

        Returns:
            An integer in [0, n).

        Raises:
            ValueError: n is not in [1, 2**64].

        """
        if not 1 <= n <= 1 << 64:
            msg = f"below({n}): n must be in [1, 2**64]"
            raise ValueError(msg)
        limit = (1 << 64) - (1 << 64) % n
        while True:
            r = self.next64()
            if r < limit:
                return r % n

    def randint(self, a: int, b: int) -> int:
        """Return a uniform integer in [a, b], both ends included.

        Returns:
            An integer in [a, b].

        """
        return a + self.below(b - a + 1)

    def choice[T](self, seq: Sequence[T]) -> T:
        """Return a uniform element of the non-empty seq.

        Returns:
            One element of seq.

        """
        return seq[self.below(len(seq))]

    def sample[T](self, population: Sequence[T], k: int) -> list[T]:
        """Return k elements of population at distinct positions (Fisher-Yates).

        Returns:
            k elements of population, in draw order.

        """
        pool = list(population)
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]
