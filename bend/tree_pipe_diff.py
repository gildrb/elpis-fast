#!/usr/bin/env python3
"""Source pin of bend/tree_pipe.bend: tree_pipe_diff.py <engine tree with ext 9601 applied>.

The Bend model transcribes, by hand, the stage kernel's per-thread writes (exllamav3_ext/tree_pipe.cu), the host's
expected upload (generator/tree_round.py verify_bytes, the Embedding host path), the readback comparison
(generator/tree_pipe.py) and the decision after the launch (generator/generator.py _tree_pipe_settle: a failed
check discards and raises, an equal readback keeps the speculative verify, a different one discards and recomputes
from the host upload). This check fails unless the sources still contain exactly those expressions
(whitespace-normalized), so an edit the model no longer describes is caught. It is a conformance check of the
transcription, not a proof of the CUDA or Python code: the laws are theorems about the model only.
"""
import re
import sys
from pathlib import Path

PINS = {
    "exllamav3_ext/tree_pipe.cu": [
        "#define TREE_PIPE_ROWS 8",
        "#define TREE_PIPE_TOKEN_BYTES 64",
        "#define TREE_PIPE_DESC_BYTES 128",
        "#define TREE_PIPE_DEPTH 16",
        "#define TREE_PIPE_DEV_BYTES (TREE_PIPE_DESC_BYTES + 4 * TREE_PIPE_ROWS)",
        # copy_desc: thread i < 128 writes dev[i] = desc[i]
        "if (t < TREE_PIPE_DESC_BYTES) dev[t] = rec[TREE_PIPE_TOKEN_BYTES + t];",
        # write_pos: thread r < 8 writes base + dp[r], dp = desc + 16
        (
            "if (t < TREE_PIPE_ROWS) reinterpret_cast<int*>(dev + TREE_PIPE_DESC_BYTES)[t] = "
            "base + (int) rec[TREE_PIPE_TOKEN_BYTES + TREE_PIPE_DEPTH + t];"
        ),
        # gather: row r of x and rows = the table row of token r, widened exactly
        "const int64_t id = reinterpret_cast<const int64_t*>(rec)[r];",
        "const float v = valid ? tree_pipe_widen<in_t>(table[id * (int64_t) hidden + c]) : 0.0f;",
        "x[(int64_t) r * hidden + c] = v;",
        "rows[(int64_t) r * hidden + c] = v;",
        "dim3 blocks((unsigned) CEIL_DIVIDE(hidden, (int64_t) TREE_PIPE_THREADS), TREE_PIPE_ROWS);",
        'TORCH_CHECK(base >= 0 && base + 255 < ((int64_t) 1 << 31), "tree_verify_stage: position base ", base, " out of range");',
    ],
    "generator/tree_round.py": [
        "DESC_BYTES = 128",
        "OUT_BYTES = 64 + DESC_BYTES",
        "DEV_BYTES = DESC_BYTES + 4 * ROWS",
        "_DEPTH = 16",
        # expect / upload: the TreeDesc, then base + depth[r]
        'return desc + struct.pack("<8i", *[base + d for d in depth])',
        "depth = list(desc[_DEPTH:_DEPTH + ROWS])",
    ],
    "generator/tree_pipe.py": [
        # same: bytes, then rows, bit for bit
        "expect = tree_round.verify_bytes(rnd.desc, rnd.depth, staged.base)",
        "if self.dev_pinned.numpy().tobytes() != expect: return False",
        "rows = self.emb.forward(torch.tensor([rnd.tokens], dtype = torch.long), {})",
        (
            "return torch.equal(rows.view(tree_round.ROWS, self.hidden).view(torch.int32), "
            "self.rows_pinned.view(torch.int32))"
        ),
        "if not (0 <= base and base + 255 < (1 << 31)):",
        # the readback is taken on a side stream after the stage, in stream order
        "staged.record(main)",
        "self.stream.wait_event(staged)",
        "self.dev_pinned.copy_(self.tree_dev, non_blocking = True)",
        "self.rows_pinned.copy_(self.rows, non_blocking = True)",
    ],
    "generator/generator.py": [
        # decide: check, then the comparison
        "staged.checked.synchronize()",
        "rnd, base = self._tree_check_record(mode)",
        "same = self.tree_pipe.same(staged, rnd)",
        "except BaseException: self.tree_pipe.discard(snap) raise",
        "if same: return rnd, batch_logits, params",
        (
            "self.tree_pipe.discard(snap) self._tree_upload(rnd, base, staging_set) "
            "batch_logits, params = verify_forward(self._batch_ids(input_ids_list), None, True)"
        ),
        # the sequential round: check, then upload, then verify
        "rnd, base = self._tree_check_record(mode) self._tree_upload(rnd, base, staging_set) return rnd",
    ],
    "modules/embedding.py": [
        # the host path the comparison reproduces: plain rows widened to float32
        "x = self.embedding.forward(x)",
        "x = to2(x, out_dtype, self.out_dtype)",
    ],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


root = Path(sys.argv[1])
missing = 0
total = 0
for rel, pins in PINS.items():
    src = norm((root / rel).read_text())
    for p in pins:
        total += 1
        ok = norm(p) in src
        missing += not ok
        print(("ok      " if ok else "MISSING ") + f"{rel}: {p}")
if missing:
    raise SystemExit(f"FAIL: {missing} transcribed expression(s) not in the source")
print(f"all {total} transcribed expressions present")
