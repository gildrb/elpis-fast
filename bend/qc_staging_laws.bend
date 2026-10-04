# Obligations for ext 9503f + 9503b (qc_staging.bend): the staging scratch qc_staging_pages(n) for a block-table span of
# n pages, against the reference qc_staging_spec.bend, and its use as the 5110h stage bound (prefill_membound.bend) with
# M4096_MAX_STAGE_PAGES = 1088. Domain: every Nat n (laws 4 and 6b need n >= 1; law 9 pins n = 0, where Python's
# (-1).bit_length() = 1 makes pow2(0) = 2). What they do NOT cover: that the scratch rows the kernels write and read stay
# below bsz * npps_w pages (the 9503f comment's argument about dequant_cache_paged_window / pattn_stage_int8k / the
# compact block table; law 1 gives only pages_alloc >= bsz * npps_w, the TORCH_CHECK in exllamav3_ext/pattn.cu), the
# memory use (1088 pages is a measurement), and that the source text is this model (qc_staging_diff.py checks it and
# evaluates the engine's own function for n = 0..4096).
import Base
import ./qc_staging.bend as Impl
import ./qc_staging_spec.bend as S
import ./prefill_membound.bend as MB
import ./prefill_merge.bend as M
import ./prefill_merge_spec.bend as Spec

# 1. The scratch covers the block-table span: round64(n) >= n (so pattn_stage_int8k's TORCH_CHECK
# pages >= bsz * pps and dequant_cache_paged_window's k_out.size(0) >= bsz * width hold for pages_alloc).
law covers:
  for +n: Nat
  {Nat.is_le(n, Impl.round64(n)) == True{} : Bool}

# 2. Never larger than the 5110h-era power of two.
law never_larger:
  for +n: Nat
  {Nat.is_le(Impl.round64(n), Impl.pow2(n)) == True{} : Bool}

# 3. Tight: less than 64 pages of slack.
law tight:
  for +n: Nat
  {Nat.is_lt(Impl.round64(n), Nat.add(n, 64n)) == True{} : Bool}

# 4. Short jobs unchanged: for 1 <= n <= 64 the scratch is the power of two.
law short_unchanged:
  for +n: Nat
  for lo: {Nat.is_le(1n, n) == True{} : Bool}
  for hi: {Nat.is_le(n, 64n) == True{} : Bool}
  {Impl.round64(n) == Impl.pow2(n) : Nat}

# 5. Monotone in the span.
law monotone:
  for +n: Nat
  for +m: Nat
  for nm: {Nat.is_le(n, m) == True{} : Bool}
  {Nat.is_le(Impl.round64(n), Impl.round64(m)) == True{} : Bool}

# 6. The model is the reference: the multiple-of-64 rounding and round64 for every n; the power of two, and the whole
# env switch (pages or the ValueError), for n >= 1. The power of two is also 5110h's m4096_stage_pages.
law impl_is_spec:
  for +n: Nat
  {Impl.ceil64(n) == S.smul64(n) : Nat} & {Impl.round64(n) == S.sround64(n) : Nat} & {Impl.pow2(n) == MB.stage_pages(n) : Nat}

law impl_is_spec_pos:
  for env: S.QcEnv
  for +n: Nat
  for lo: {Nat.is_le(1n, n) == True{} : Bool}
  {Impl.pow2(n) == S.spow2(n) : Nat} & {Impl.stage(env, n) == S.sstage(env, n) : S.QcOut}

# 7. The served numbers: a 1025-page table (262144 tokens + the 8-token draft margin) stages 1088 pages, within the
# 9503b bound 1088; with EXL3_QC_STAGING_ROUND64=0 it stages 2048 pages, above it; 1024 pages stage 1024; constants.
law served_numbers:
  {Impl.round64(1025n) == 1088n : Nat} &
  {Nat.is_le(Impl.round64(1025n), Impl.MAX_STAGE_PAGES()) == True{} : Bool} &
  {Impl.pow2(1025n) == 2048n : Nat} &
  {Nat.is_le(Impl.pow2(1025n), Impl.MAX_STAGE_PAGES()) == False{} : Bool} &
  {Impl.round64(1024n) == 1024n : Nat} &
  {Impl.stage(S.QcZero{}, 1025n) == S.QcPages{2048n} : S.QcOut} &
  {Impl.MAX_STAGE_PAGES() == 1088n : Nat} & {Impl.ROUND_PAGES() == 64n : Nat} & {Impl.MAX_PROMPT() == 262143n : Nat}

# 8. Every served block table of at most 1025 pages passes the 1088 stage conjunct, so under the default switch
# (unset or "1") the 9503 merge decision, step and schedule are exactly 5110g's with G = M4096_MAX_PROMPT = 262143.
law within_1088:
  for +n: Nat
  for small: {Nat.is_le(n, 1025n) == True{} : Bool}
  {Nat.is_le(Impl.round64(n), Impl.MAX_STAGE_PAGES()) == True{} : Bool}

law served_is_5110g:
  for fuel: Nat
  for env: S.QcEnv
  for +n: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for small: {Nat.is_le(n, 1025n) == True{} : Bool}
  for default: {S.rounds64(env) == True{} : Bool}
  {Impl.stage(env, n) == S.QcPages{Impl.round64(n)} : S.QcOut} &
  {MB.merge_ok(en, k, T, mc, P, Impl.MAX_PROMPT(), Impl.round64(n), Impl.MAX_STAGE_PAGES()) == M.merge_ok(en, k, T, mc, P, 262143n) : Bool} &
  {MB.step(en, k, T, mc, P, Impl.MAX_PROMPT(), Impl.round64(n), Impl.MAX_STAGE_PAGES()) == M.step(en, k, T, mc, P, 262143n) : Nat} &
  {MB.visits(fuel, en, k, T, mc, P, Impl.MAX_PROMPT(), Impl.round64(n), Impl.MAX_STAGE_PAGES()) == M.visits(fuel, en, k, T, mc, P, 262143n) : List<&2, Nat>}

# Kill switch "0": the 1025-page table stages 2048 pages, past 1088, so its whole schedule is the 2048-row one.
law killswitch_1025_unmerged:
  for fuel: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  {MB.visits(fuel, en, k, T, mc, P, Impl.MAX_PROMPT(), Impl.pow2(1025n), Impl.MAX_STAGE_PAGES()) == Spec.visits(fuel, k, T, mc, P) : List<&2, Nat>}

# 9. Pinned discrepancy at n = 0 (not reachable from the served callers, which pass >= 1 page): Python's
# (0 - 1).bit_length() is 1, so pow2(0) = 2 pages, not the smallest power of two >= 0 (1), and the switch "0" stages 2
# where the reference stages 1; round64(0) = 0 (= the reference), so law 4 fails at n = 0 (0 != 2).
law zero_pages:
  {Impl.pow2(0n) == 2n : Nat} & {S.spow2(0n) == 1n : Nat} & {Impl.round64(0n) == 0n : Nat} &
  {Impl.stage(S.QcZero{}, 0n) == S.QcPages{2n} : S.QcOut} & {S.sstage(S.QcZero{}, 0n) == S.QcPages{1n} : S.QcOut}
