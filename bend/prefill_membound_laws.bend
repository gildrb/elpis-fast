# Obligations for ext 5110h (prefill_membound.bend: 5110g's merge decision plus the staging-scratch page bound
# s = m4096_stage_pages(len(seq.allocated_pages)) <= B = M4096_MAX_STAGE_PAGES, and M4096_MAX_PROMPT = 262143) against
# the 2048-row prefill schedule (prefill_merge_spec.bend). Domain: as prefill_merge_laws.bend (served text-only
# recurrent model, no prompt-cache page hit in the pieces involved; law 1 at page-aligned k = a P with mc = c P, c >= 1,
# P >= 1); s and B are any Nats, constant over a job's schedule (len(seq.allocated_pages) is fixed between
# allocate_pages and deallocate_pages, see prefill_membound.bend). What they do NOT cover: the memory use itself (the
# 1024-page bound is a measurement, 0081zr, not a theorem), floating point of the forwards, the empty_cache call, and
# whether the constants in the source are 1024 / 262143 (prefill_membound_diff.py checks the text).
import Base
import ./prefill_membound.bend as Impl
import ./prefill_merge.bend as M
import ./prefill_merge_spec.bend as Spec

# 1. A 5110h merged step is exactly two 2048-row steps: the merged forward ends where the second of the two unmerged
# pieces ends (k + 2 mc), the first unmerged piece ends at k + mc, and neither is cut by the last-page split.
law merge_two_steps:
  for +a: Nat
  for +c: Nat
  for +T: Nat
  for +P: Nat
  for +G: Nat
  for +s: Nat
  for +B: Nat
  for pp: {Nat.is_lt(0n, P) == True{} : Bool}
  for cp: {Nat.is_lt(0n, c) == True{} : Bool}
  for merged: {Impl.merge_ok(True{}, Nat.mul(a, P), T, Nat.mul(c, P), P, G, s, B) == True{} : Bool}
  {Spec.step(Nat.mul(a, P), T, Nat.mul(c, P), P) == Nat.add(Nat.mul(a, P), Nat.mul(c, P)) : Nat} &
  {Spec.step(Spec.step(Nat.mul(a, P), T, Nat.mul(c, P), P), T, Nat.mul(c, P), P) == Nat.add(Nat.add(Nat.mul(a, P), Nat.mul(c, P)), Nat.mul(c, P)) : Nat} &
  {Impl.step(True{}, Nat.mul(a, P), T, Nat.mul(c, P), P, G, s, B) == Nat.add(Nat.add(Nat.mul(a, P), Nat.mul(c, P)), Nat.mul(c, P)) : Nat}

# 2. Otherwise the step is the 2048-row step.
law unmerged_same:
  for en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for +s: Nat
  for +B: Nat
  for no: {Impl.merge_ok(en, k, T, mc, P, G, s, B) == False{} : Bool}
  {Impl.step(en, k, T, mc, P, G, s, B) == Spec.step(k, T, mc, P) : Nat}

# 3. The guard: a merge happens only with last_tok <= G and a staging scratch of at most B pages.
law guard:
  for en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for +s: Nat
  for +B: Nat
  for merged: {Impl.merge_ok(en, k, T, mc, P, G, s, B) == True{} : Bool}
  {Nat.is_le(T, G) == True{} : Bool} & {Nat.is_le(s, B) == True{} : Bool}

# 4. Above either bound the whole schedule (every kv position the prefill visits, fuel-bounded) is the 2048-row one.
law above_stage_schedule:
  for fuel: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for +s: Nat
  for +B: Nat
  for above: {Nat.is_le(s, B) == False{} : Bool}
  {Impl.visits(fuel, en, k, T, mc, P, G, s, B) == Spec.visits(fuel, k, T, mc, P) : List<&2, Nat>}

law above_guard_schedule:
  for fuel: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for +s: Nat
  for +B: Nat
  for above: {Nat.is_le(T, G) == False{} : Bool}
  {Impl.visits(fuel, en, k, T, mc, P, G, s, B) == Spec.visits(fuel, k, T, mc, P) : List<&2, Nat>}

# 5. Within the bound 5110h is 5110g: with s <= B the decision, the step and the schedule equal prefill_merge.bend's
# (same G); so 5110g's laws (merged steps, checkpoint positions) carry over unchanged.
law within_bound_is_5110g:
  for fuel: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for +s: Nat
  for +B: Nat
  for within: {Nat.is_le(s, B) == True{} : Bool}
  {Impl.merge_ok(en, k, T, mc, P, G, s, B) == M.merge_ok(en, k, T, mc, P, G) : Bool} &
  {Impl.step(en, k, T, mc, P, G, s, B) == M.step(en, k, T, mc, P, G) : Nat} &
  {Impl.visits(fuel, en, k, T, mc, P, G, s, B) == M.visits(fuel, en, k, T, mc, P, G) : List<&2, Nat>}

# 6. The served stage function (job.py m4096_stage_pages, power of two) at the bound: a 1024-page block table stages
# 1024 pages (<= 1024: merge-eligible), a 1025-page one stages 2048 (> 1024: never merged, so by law 4 its schedule is
# the 2048-row one); the served constants are B = 1024, G = 262143.
law stage_pow2_bound:
  {Impl.stage_pages(1024n) == 1024n : Nat} &
  {Nat.is_le(Impl.stage_pages(1024n), Impl.MAX_STAGE_PAGES()) == True{} : Bool} &
  {Impl.stage_pages(1025n) == 2048n : Nat} &
  {Nat.is_le(Impl.stage_pages(1025n), Impl.MAX_STAGE_PAGES()) == False{} : Bool} &
  {Impl.MAX_STAGE_PAGES() == 1024n : Nat} & {Impl.MAX_PROMPT() == 262143n : Nat}

law stage_1025_unmerged:
  for fuel: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  {Impl.visits(fuel, en, k, T, mc, P, Impl.MAX_PROMPT(), Impl.stage_pages(1025n), Impl.MAX_STAGE_PAGES()) == Spec.visits(fuel, k, T, mc, P) : List<&2, Nat>}
