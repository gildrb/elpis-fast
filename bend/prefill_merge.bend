# Ext 5110g: one prefill() step of generator/job.py with the merged 4096-row piece, transcribed over Nat for the served
# text-only recurrent model (recurrent_cache present: the partial-page copy J:1324 is skipped and the last-page cut J:1363
# runs) in a region without prompt-cache page hits (the skip loops J:1256-1275 and the merge's own page check 5110g:51-56
# find no full cached page). J = generator/job.py of the served tree (exl3-ext.json post hash 33b9bfec), X =
# patches/exl3-ext/5110g-prefill-m4096-guard.patch. Names: k = seq.kv_position = prefill_start, T = last_tok =
# len(seq.sequence_ids) - 1, mc = max_chunk_size, P = PAGE_SIZE, G = M4096_MAX_PROMPT. Reference: prefill_merge_spec.bend
# (the step without the merge); laws: prefill_merge_laws.bend.
import Base

# J:1245-1247 prefill_end = kv_position + max_chunk_size; prefill_end = (prefill_end // PAGE_SIZE) * PAGE_SIZE;
# prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)
def base_end(+k: Nat, +T: Nat, +mc: Nat, +P: Nat) -> Nat:
  Nat.min(Nat.mul(Nat.div(Nat.add(k, mc), P), P), T)

# X:44-47 (J:1290-1295): recurrent_state and not embeddings and switch on (en), last_tok <= M4096_MAX_PROMPT,
# prefill_start == kv_position (no page skipped: domain), prefill_start % (2 * mc) == 0, prefill_end - prefill_start == mc,
# prefill_end + mc <= last_tok - 2 * mc, mc % PAGE_SIZE == 0. (Python ints: last_tok - 2 mc may be negative; with
# mc > 0 the Nat truncation gives the same verdict, since prefill_end + mc > 0.)
def merge_ok(en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat) -> Bool:
  Bool.and(en, Bool.and(Nat.is_le(T, G), Bool.and(Nat.is_eq(Nat.mod(k, Nat.mul(2n, mc)), 0n),
    Bool.and(Nat.is_eq(Nat.sub(base_end(k, T, mc, P), k), mc),
      Bool.and(Nat.is_le(Nat.add(base_end(k, T, mc, P), mc), Nat.sub(T, Nat.mul(2n, mc))), Nat.is_eq(Nat.mod(mc, P), 0n))))))

# X:49, 57-58: e2 = prefill_end + mc; prefill_end = e2 when merging
def widen(m: Bool, +e: Nat, +mc: Nat) -> Nat:
  match m:
    case True{}:
      Nat.add(e, mc)
    case False{}:
      e

# J:1364-1367 seqlen = len(seq.sequence_ids) - 1; last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE;
# if prefill_start < last_page_b <= prefill_end: prefill_end = last_page_b
def cut.pick(c: Bool, +lpb: Nat, +e: Nat) -> Nat:
  match c:
    case True{}:
      lpb
    case False{}:
      e

def cut(+k: Nat, +T: Nat, +P: Nat, +e: Nat) -> Nat:
  cut.pick(Bool.and(Nat.is_lt(k, Nat.mul(Nat.div(T, P), P)), Nat.is_le(Nat.mul(Nat.div(T, P), P), e)), Nat.mul(Nat.div(T, P), P), e)

# J:1501 seq.kv_position = prefill_end: the next step starts where this forward ended; the forward runs rows k .. end - 1
def step(en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat) -> Nat:
  cut(k, T, P, widen(merge_ok(en, k, T, mc, P, G), base_end(k, T, mc, P), mc))

# The kv positions the prefill reaches from k, one per prefill() call (J:1238 one piece per call; generator.py:632-640
# calls prefill() and then recurrent_checkpoint() once per iteration, so a recurrent checkpoint can be stashed only at
# a position listed here, generator.py:690-692, J:1660-1681); fuel-bounded, ends once kv_position >= T (J:1522-1523)
def visits.go(fuel: Nat, done: Bool, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat) -> List<&2, Nat>:
  match fuel:
    case 0n:
      []
    case 1n+f:
      match done:
        case True{}:
          []
        case False{}:
          step(en, k, T, mc, P, G) <> visits.go(f, Nat.is_le(T, step(en, k, T, mc, P, G)), en, step(en, k, T, mc, P, G), T, mc, P, G)

def visits(fuel: Nat, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat) -> List<&2, Nat>:
  visits.go(fuel, Nat.is_le(T, k), en, k, T, mc, P, G)
