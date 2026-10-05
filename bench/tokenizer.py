# Copyright (c) 2026 inference contributors.
"""The served target tokenizer for the C1 and prefill lanes.

Kept apart from ``bench.exl3`` so that module stays importable by host Python
without the prepared evaluator's ``tokenizers`` wheel.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from tokenizers import Tokenizer

if TYPE_CHECKING:
    from pathlib import Path


class RawTokenizer:
    """The served target tokenizer, loaded from bytes already bound to identity."""

    def __init__(self, path: Path, sha256: str) -> None:
        """Load exactly the hash-pinned tokenizer.json bytes.

        Args:
            path: The host tokenizer.json path.
            sha256: The pinned SHA256 of its bytes.

        Raises:
            ValueError: If the tokenizer bytes differ from the pin.

        """
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            msg = "Tokenizer bytes changed"
            raise ValueError(msg)
        self.tokenizer: Tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        self.sha256 = sha256

    def count(self, content: str) -> int:
        """Count raw-content tokens without special tokens or a chat template.

        Args:
            content: The raw content.

        Returns:
            The token count.

        """
        return len(self.tokenizer.encode(content, add_special_tokens=False).ids)
