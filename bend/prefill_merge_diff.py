#!/usr/bin/env python3
"""
Finite source link of bend/prefill_merge.bend (ext 5110g merged prefill pieces) and bend/prefill_merge_spec.bend to the
patched engine text: every expression the two models transcribe occurs verbatim (once where it names one site), in the
order the models assume (prefill end, cached-page skip, merge, last-page cut; prefill() before recurrent_checkpoint()
in iterate()), the guard constant is 131072, and the GDN b/a projections of a merged forward are split back into
2048-row calls. Text evidence, not a proof. `--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/prefill_merge_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

JOB = [
    "M4096_MAX_PROMPT = 131072",
    "prefill_end = seq.kv_position + self.generator.max_chunk_size",
    "prefill_end = (prefill_end // PAGE_SIZE) * PAGE_SIZE",
    "prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)",
    "last_tok = len(seq.sequence_ids) - 1",
    "last_tok <= M4096_MAX_PROMPT and",
    "prefill_start == seq.kv_position and prefill_start % (2 * mc) == 0 and",
    "prefill_end - prefill_start == mc and prefill_end + mc <= last_tok - 2 * mc and mc % PAGE_SIZE == 0",
    "e2 = prefill_end + mc",
    "prefill_end = e2",
    "seqlen = len(seq.sequence_ids) - 1",
    "last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE",
    "if prefill_start < last_page_b <= prefill_end:",
    "prefill_end = last_page_b",
    "seq.kv_position = prefill_end",
    "elif seq_pos >= prompt_len - self.generator.max_chunk_size * 2:",
    "return (seq_pos - self.cached_pages * PAGE_SIZE) % self.generator.recurrent_checkpoint_interval_pp == 0",
]
ORDER = [
    "prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)",
    "if prefill_end <= prefill_start:",
    "e2 = prefill_end + mc",
    "if prefill_start < last_page_b <= prefill_end:",
    "seq.kv_position = prefill_end",
]
GEN = [
    "job.prefill(results)",
    "self.recurrent_checkpoint()",
    "job.maybe_stash_recurrent(self.recurrent_cache)",
    "recurrent_checkpoint_interval_pp: int = 32768,",
    "self.recurrent_checkpoint_interval_pp = ceil_span(recurrent_checkpoint_interval_pp, self.max_chunk_size)",
]
GDN = [
    "if bsz == 1 and seqlen > 2048:",
    "b = torch.cat([self.b_proj.forward(x[:, i:i + 2048], params) for i in range(0, seqlen, 2048)], 1)",
    "a = torch.cat([self.a_proj.forward(x[:, i:i + 2048], params) for i in range(0, seqlen, 2048)], 1)",
]
MUTATIONS = {
    "guard_262k": ("M4096_MAX_PROMPT = 131072", "M4096_MAX_PROMPT = 262144"),
    "margin": ("prefill_end + mc <= last_tok - 2 * mc", "prefill_end + mc <= last_tok - mc"),
    "cut_first": ("            # Ext 5110 (m4096)", "            if prefill_start < last_page_b <= prefill_end: pass\n            # Ext 5110 (m4096)"),
}


def fail(msg: str) -> None:
    raise SystemExit(f"prefill_merge_diff: FAIL: {msg}")


def main(argv: list[str]) -> None:
    args = argv[1:]
    mutate = None
    if len(args) == 3 and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: prefill_merge_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    root = Path(args[0])
    job = (root / "generator/job.py").read_text()
    gen = (root / "generator/generator.py").read_text()
    gdn = (root / "modules/gated_delta_net.py").read_text()
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r}")
        old, new = MUTATIONS[mutate]
        if job.count(old) != 1:
            fail(f"mutation anchor {old!r}")
        job = job.replace(old, new)
        print(f"prefill_merge_diff: applied mutation {mutate}")
    body = job[job.index("    def prefill(self, results: list):"):job.index("    def allocate_pages(self):")]
    for e in JOB:
        where = job if e.startswith(("M4096", "elif seq_pos", "return (seq_pos")) else body
        if where.count(e) != 1:
            fail(f"job.py: {e!r} occurs {where.count(e)} times (want 1)")
    pos = [body.index(e) for e in ORDER]
    if pos != sorted(pos):
        fail(f"job.py prefill(): order of {ORDER} is {pos}")
    it = gen[gen.index("def iterate(self)"):gen.index("def on_queue_drained")]
    if not it.index("job.prefill(results)") < it.index("self.recurrent_checkpoint()"):
        fail("generator.py: prefill() must precede recurrent_checkpoint() in iterate()")
    for e in GEN:
        if gen.count(e) < 1:
            fail(f"generator.py: {e!r} missing")
    for e in GDN:
        if gdn.count(e) != 1:
            fail(f"gated_delta_net.py: {e!r} occurs {gdn.count(e)} times (want 1)")
    print(f"prefill_merge_diff: {len(JOB) + len(GEN) + len(GDN)} expressions found, prefill() order {ORDER} holds")
    print("prefill_merge_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
