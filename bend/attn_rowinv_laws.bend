# Obligations for the row-invariant verify-attention partition (ext patch 3005) over the Nat model
# attn_rowinv.bend against the absolute-position reference attn_rowinv_spec.bend. Domain: every
# round start c, every q_len <= SPLITS * BLOCK_N (the served target: SPLITS 39, BLOCK_N 32,
# q_len 8), every row r < q_len, every split k, every MIN_SPLIT (B2 states the tile alignment for
# MIN_SPLIT = m * BLOCK_N). SPLITS >= 1 and BLOCK_N >= 1 follow from r < q_len <= SPLITS * BLOCK_N.
# Hypotheses a law does not need are left out (a law without them is stronger).
import Base
import ./attn_rowinv.bend as Impl
import ./attn_rowinv_spec.bend as Spec
import ./attn_split.bend as Split

# B1: the split length and every chunk [lo, hi) the kernel uses for row r of the round starting at
# c equal the reference partition of the row's absolute position p = c + r.
law row_partition_absolute:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +k: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  {Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split) ==
    Spec.row_len(1n+Nat.add(c, r), splits, block_n, min_split) : Nat} &
  {Impl.chunk_lo(c, q_len, r, k, splits, block_n, min_split) ==
    Spec.chunk_lo(Nat.add(c, r), k, splits, block_n, min_split) : Nat} &
  {Impl.chunk_hi(c, q_len, r, k, splits, block_n, min_split) ==
    Spec.chunk_hi(Nat.add(c, r), k, splits, block_n, min_split) : Nat}

# B2: every row's chunks fit in the SPLITS launched programs (cdiv(n, s(n)) <= SPLITS for n >= 1
# keys) and s(n) is a whole number of BLOCK_N tiles when MIN_SPLIT = m * BLOCK_N.
law coverage:
  for +n: Nat
  for +splits: Nat
  for +block_n: Nat
  for +m: Nat
  for npos: {Nat.is_lt(0n, n) == True{} : Bool}
  for spos: {Nat.is_lt(0n, splits) == True{} : Bool}
  for bpos: {Nat.is_lt(0n, block_n) == True{} : Bool}
  {Nat.is_le(Spec.ceil_div(n, Impl.row_split_len(n, splits, block_n, Nat.mul(m, block_n))), splits) == True{} : Bool} &
  {Nat.mod(Impl.row_split_len(n, splits, block_n, Nat.mul(m, block_n)), block_n) == 0n : Nat}

# B3: within a round every row's s(p + 1) is sA or sB, sA <= sB, and no row is assigned to pass B
# unless pass B is launched: two passes compute every row with its own partition.
law two_passes_suffice:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  {Bool.or(Nat.is_eq(Impl.row_split_len(1n+Nat.add(c, r), splits, block_n, min_split), Impl.pass_a_len(c, splits, block_n, min_split)),
    Nat.is_eq(Impl.row_split_len(1n+Nat.add(c, r), splits, block_n, min_split), Impl.pass_b_len(c, q_len, splits, block_n, min_split))) == True{} : Bool} &
  {Nat.is_le(Impl.pass_a_len(c, splits, block_n, min_split), Impl.pass_b_len(c, q_len, splits, block_n, min_split)) == True{} : Bool} &
  {Bool.and(Impl.in_pass_b(c, r, splits, block_n, min_split), Bool.not(Impl.runs_pass_b(c, q_len, splits, block_n, min_split))) == False{} : Bool}

# B4 (headline): the same absolute row computed in two different rounds (c1 + r1 = c2 + r2) gets
# the identical split length and identical chunks.
law round_independence:
  for +c1: Nat
  for +q1: Nat
  for +r1: Nat
  for +c2: Nat
  for +q2: Nat
  for +r2: Nat
  for +k: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits1: {Nat.is_le(q1, Nat.mul(splits, block_n)) == True{} : Bool}
  for row1: {Nat.is_lt(r1, q1) == True{} : Bool}
  for fits2: {Nat.is_le(q2, Nat.mul(splits, block_n)) == True{} : Bool}
  for row2: {Nat.is_lt(r2, q2) == True{} : Bool}
  for same: {Nat.add(c1, r1) == Nat.add(c2, r2) : Nat}
  {Impl.kernel_row_split_len(c1, q1, r1, splits, block_n, min_split) ==
    Impl.kernel_row_split_len(c2, q2, r2, splits, block_n, min_split) : Nat} &
  {Impl.chunk_lo(c1, q1, r1, k, splits, block_n, min_split) == Impl.chunk_lo(c2, q2, r2, k, splits, block_n, min_split) : Nat} &
  {Impl.chunk_hi(c1, q1, r1, k, splits, block_n, min_split) == Impl.chunk_hi(c2, q2, r2, k, splits, block_n, min_split) : Nat}

# B5: for every merge op with e as right identity (the kernel's acc + 0 * w = acc, stated as a
# hypothesis) and every partial function P, the combine of row r over all SPLITS splits equals the
# left fold, from e, over the row's own chunks 0 .. cdiv(p + 1, s(p + 1)) - 1 of the reference: the
# trailing splits past the row's last chunk are identities, so the result depends on p alone.
law combine_absolute:
  for ~A: Data
  for ~merge: A -> A -> A
  for ~P: Nat -> Nat -> A
  for ~e: A
  for rid: @x:A -> {merge(x, e) == x : A}
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  {Impl.combine(~A, ~merge, ~P, ~e, c, q_len, r, splits, block_n, min_split) ==
    Spec.row_result(~A, ~merge, ~P, ~e, Nat.add(c, r), splits, block_n, min_split) : A}

# B6 (the bug 3005 fixes): with ext 3002's round-relative s_len(c + q_len) the same absolute row
# gets different partitions in different rounds. Served target (SPLITS 39, BLOCK_N 32, MIN_SPLIT
# 128, q_len 8): position p = 121 is row 0 of the round at c = 121 (total 129: live 2, s_len 96,
# chunks [0, 96) [96, 122)) and row 1 of the round at c = 120 (total 128: live 1, s_len 128,
# chunk [0, 122)); the 3005 kernel gives it s = 128 in both rounds.
law old_round_relative_differs:
  {Impl.old_row_split_len(121n, 8n, 0n, 39n, 32n, 128n) == 96n : Nat} &
  {Impl.old_row_split_len(120n, 8n, 1n, 39n, 32n, 128n) == 128n : Nat} &
  {Impl.old_chunk_hi(121n, 8n, 0n, 0n, 39n, 32n, 128n) == 96n : Nat} &
  {Impl.old_chunk_hi(120n, 8n, 1n, 0n, 39n, 32n, 128n) == 122n : Nat} &
  {Impl.kernel_row_split_len(121n, 8n, 0n, 39n, 32n, 128n) == 128n : Nat} &
  {Impl.kernel_row_split_len(120n, 8n, 1n, 39n, 32n, 128n) == 128n : Nat}

# B7 (every key exactly once): for row r of the round starting at c, key_valid(k, j) is the
# kernel's visit-and-mask predicate for key j in split k of the row's pass. (a) every causal key
# j <= c + r is valid in split j / s (s the row's split length), a launched split (< SPLITS);
# (b) no key is valid in two splits; (c) only causal keys are valid. So the n = c + r + 1 keys of
# the row are each read by exactly one launched split, and no other key is.
law key_covered:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +j: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  for causal: {Nat.is_le(j, Nat.add(c, r)) == True{} : Bool}
  {Nat.is_lt(Nat.div(j, Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split)), splits) == True{} : Bool} &
  {Impl.key_valid(c, q_len, r, Nat.div(j, Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split)), j,
    splits, block_n, min_split) == True{} : Bool}

law key_unique:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +k1: Nat
  for +k2: Nat
  for +j: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  for v1: {Impl.key_valid(c, q_len, r, k1, j, splits, block_n, min_split) == True{} : Bool}
  for v2: {Impl.key_valid(c, q_len, r, k2, j, splits, block_n, min_split) == True{} : Bool}
  {k1 == k2 : Nat}

law key_causal:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +k: Nat
  for +j: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for v: {Impl.key_valid(c, q_len, r, k, j, splits, block_n, min_split) == True{} : Bool}
  {Nat.is_le(j, Nat.add(c, r)) == True{} : Bool}

# B8 (every row exactly once): pass A computes and stores the rows with in_pass_b False (rows
# ~bX, triton_paged.py:1354), pass B the rows with in_pass_b True and only when it is launched
# (s_b != s_a, 1364), so exactly one launched pass computes row r. With B3 an 8-row round splits
# into at most two groups of equal split length and the two passes cover every row once. The
# kernel's extra guard q_len > 1 (1363) changes nothing: with one row pass B is never launched.
law pass_exactly_once:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  {Nat.add(Split.bit(Bool.not(Impl.in_pass_b(c, r, splits, block_n, min_split))),
    Split.bit(Bool.and(Impl.in_pass_b(c, r, splits, block_n, min_split), Impl.runs_pass_b(c, q_len, splits, block_n, min_split)))) == 1n : Nat}

law pass_b_single_row:
  for +c: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  {Impl.runs_pass_b(c, 1n, splits, block_n, min_split) == False{} : Bool}

# B9: the number of launched splits holding a non-empty chunk of row r is the reference chunk
# count of its absolute position c + r (depends on n = c + r + 1 alone).
law row_chunk_count:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  {Impl.chunk_count(c, q_len, r, splits, block_n, min_split) == Spec.chunks(Nat.add(c, r), splits, block_n, min_split) : Nat}

# B10: the shipped accumulator-store guard store_acc (triton_paged.py:1351, from pass A's s_a) is
# true for every split block holding a non-empty chunk of any row of the round (s_a <= the row's
# split length): the combine never reads a stale accumulator row of a live split.
law acc_stored:
  for +c: Nat
  for +q_len: Nat
  for +r: Nat
  for +k: Nat
  for +splits: Nat
  for +block_n: Nat
  for +min_split: Nat
  for fits: {Nat.is_le(q_len, Nat.mul(splits, block_n)) == True{} : Bool}
  for row: {Nat.is_lt(r, q_len) == True{} : Bool}
  for live: {Nat.is_lt(Impl.chunk_lo(c, q_len, r, k, splits, block_n, min_split),
    Impl.chunk_hi(c, q_len, r, k, splits, block_n, min_split)) == True{} : Bool}
  {Impl.store_acc(c, q_len, k, splits, block_n, min_split) == True{} : Bool}
