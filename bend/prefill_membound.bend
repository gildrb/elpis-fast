# Ext 5110h: the 5110g merge decision of one prefill() step of generator/job.py with the memory bound on the attention
# staging scratch, transcribed over Nat. Same domain as prefill_merge.bend (served text-only recurrent model, no
# prompt-cache page hit in the pieces involved); the 5110g lines this model still uses (prefill end, merge widening, the
# last-page cut, the per-call visits) are prefill_merge.bend's (imported as M, unchanged by 5110h). J = generator/job.py
# of the tree made through patches/exl3-ext/5110h-prefill-m4096-membound.patch (H); line numbers are that tree's.
# Names as prefill_merge.bend plus s = m4096_stage_pages(len(seq.allocated_pages)) and B = M4096_MAX_STAGE_PAGES.
# s is a per-job constant: seq.allocated_pages is assigned only by Sequence.allocate_pages (generator/pagetable.py:289,
# one entry per page of prompt + max_new_tokens) before prefill() runs and emptied only by Job.deallocate_pages
# (J:1641); prefill and the rewind / tree / dedup paths in between replace or read entries, never change its length
# (J:716 replaces one page in place). So s and B are parameters fixed for the whole visits schedule of a job.
# Reference: prefill_merge_spec.bend (the 2048-row schedule); laws: prefill_membound_laws.bend.
import Base
import ./prefill_merge.bend as M

# Python int.bit_length() of a Nat m (the number of binary digits; 0 for 0), by halving; fuel m suffices (m halves)
def bl.go(f: Nat, m: Nat) -> Nat:
  match f:
    case 0n:
      0n
    case 1n+g:
      match m:
        case 0n:
          0n
        case 1n+p:
          1n+bl.go(g, Nat.div(1n+p, 2n))

def bit_length(+m: Nat) -> Nat:
  bl.go(m, m)

# (n - 1).bit_length() for a Python int n >= 0: n = 0 gives (-1).bit_length() = 1 (Python's bit_length of a negative
# int is that of its absolute value)
def bit_length_dec(n: Nat) -> Nat:
  match n:
    case 0n:
      1n
    case 1n+p:
      bit_length(p)

# H:28 (J:39) return max(1, 1 << (block_table_pages - 1).bit_length())   (1 << b = 2 ** b)
def stage_pages(n: Nat) -> Nat:
  Nat.max(1n, Nat.pow(2n, bit_length_dec(n)))

# H:22-23 (J:33-34) the served constants
def MAX_STAGE_PAGES() -> Nat:
  1024n

def MAX_PROMPT() -> Nat:
  262143n

# H:33-38 (J:1323-1328), conjuncts in source order: recurrent_state and not embeddings and switch on (en),
# last_tok <= M4096_MAX_PROMPT, m4096_stage_pages(len(seq.allocated_pages)) <= M4096_MAX_STAGE_PAGES (s <= B, the 5110h
# conjunct), prefill_start == kv_position (no page skipped: domain), prefill_start % (2 * mc) == 0,
# prefill_end - prefill_start == mc, prefill_end + mc <= last_tok - 2 * mc, mc % PAGE_SIZE == 0
def merge_ok(en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat) -> Bool:
  Bool.and(en, Bool.and(Nat.is_le(T, G), Bool.and(Nat.is_le(s, B), Bool.and(Nat.is_eq(Nat.mod(k, Nat.mul(2n, mc)), 0n),
    Bool.and(Nat.is_eq(Nat.sub(M.base_end(k, T, mc, P), k), mc),
      Bool.and(Nat.is_le(Nat.add(M.base_end(k, T, mc, P), mc), Nat.sub(T, Nat.mul(2n, mc))), Nat.is_eq(Nat.mod(mc, P), 0n)))))))

# J:1330, 1339 widen when merging; J:1399-1402 last-page cut; J:1547 seq.kv_position = prefill_end
def step(en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat) -> Nat:
  M.cut(k, T, P, M.widen(merge_ok(en, k, T, mc, P, G, s, B), M.base_end(k, T, mc, P), mc))

# The kv positions the prefill reaches from k, one per prefill() call (as prefill_merge.bend visits; s, B fixed per job)
def visits.go(fuel: Nat, done: Bool, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat) -> List<&2, Nat>:
  match fuel:
    case 0n:
      []
    case 1n+f:
      match done:
        case True{}:
          []
        case False{}:
          step(en, k, T, mc, P, G, s, B) <> visits.go(f, Nat.is_le(T, step(en, k, T, mc, P, G, s, B)), en,
            step(en, k, T, mc, P, G, s, B), T, mc, P, G, s, B)

def visits(fuel: Nat, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat) -> List<&2, Nat>:
  visits.go(fuel, Nat.is_le(T, k), en, k, T, mc, P, G, s, B)
