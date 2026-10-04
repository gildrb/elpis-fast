# Obligations for the integer core of ext 3021c's int8 QK^T (pattn8i_int.bend, kernel exllamav3_ext/pattn8i_kernel.cuh
# of the patched tree) against the exact reference (pattn8i_int_spec.bend). Integers are offset pairs Spec.Z{p, n} =
# p - n over Nat. Covered: the 3-bit K code; the Q int8 clamp; the K int8 requantisation bound and its exact case;
# the IMMA issue pattern, each 32-term partial and the 8-IMMA int32 chain (no wrap: equal to the exact 256-dim
# integer dot product); i2f_exact as an IEEE single decoding (exact on [-2^22, 2^22), where its precondition stops
# holding, and what it returns just outside); the finding that K is rounded (law 9); the score's rounding points.
# NOT covered: the values of the fp32 steps (Q rotation and quantisation, f, the K product and rint, rs, skk, rs * sk,
# the final multiply), which are modelled only as relations (adjacent-integer rounding) or listed as rounding points;
# the mapping of head dims to IMMA fragments (ldmatrix lane layout); the CQ cache codes' own accuracy. The host C
# evidence in pattn8i_int_diff.py checks the fp32 facts the model takes as hypotheses (f <= 18; RN of a representable
# difference is exact) exhaustively over fp16 scale pairs and over the int32 range.
import Base
import ./pattn8i_int.bend as Impl
import ./pattn8i_int_spec.bend as Spec

# 1. The 3-bit field is below 8 for every hi / lo word and lane i, and its code 2 c - 7 is one of the odd integers
# -7, -5, ..., 7; the map c -> 2 c - 7 on 0..7 lists them in ascending order (so it is a bijection onto them).
law code_field:
  for +hi: Nat
  for +lo: Nat
  for +i: Nat
  {Nat.is_lt(Impl.field(hi, lo, i), 8n) == True{} : Bool} &
  {Spec.member(Impl.code(Impl.field(hi, lo, i)), Spec.odd_codes()) == True{} : Bool}

law code_bijection:
  {[Spec.norm(Impl.code(0n)), Spec.norm(Impl.code(1n)), Spec.norm(Impl.code(2n)), Spec.norm(Impl.code(3n)),
    Spec.norm(Impl.code(4n)), Spec.norm(Impl.code(5n)), Spec.norm(Impl.code(6n)), Spec.norm(Impl.code(7n))]
    == Spec.odd_codes() : List<&2, Spec.Z>} &
  {Spec.ascending(Spec.odd_codes()) == True{} : Bool}

# 2. Q: every pre-clamp integer x gives |q8| <= 127 (canonical, so mass = |q8|); x itself when |x| <= 127.
law clamp_bound:
  for +xp: Nat
  for +xn: Nat
  {Nat.is_le(Spec.mass(Impl.clamp(Spec.Z{xp, xn})), 127n) == True{} : Bool}

law clamp_keeps:
  for +xp: Nat
  for +xn: Nat
  for small: {Nat.is_le(Spec.abs(Spec.Z{xp, xn}), 127n) == True{} : Bool}
  Spec.zeq(Impl.clamp(Spec.Z{xp, xn}), Spec.Z{xp, xn})

# 3. K: for f = a / b with 0 <= f <= 18 (hypothesis f18: in the kernel f = RN(RN(18 s) / smax) with 0 <= s <= smax;
# 18 s is exact in fp32 (an fp16 s has 11 significant bits, 18 = 9 * 2 needs 4), 18 s / smax <= 18, and RN is
# monotone with 18 representable, so f <= 18; checked over all fp16 pairs by pattn8i_int_diff.py), every adjacent
# integer kq of code * f has |kq| <= 126. On the key's largest-scale group (s = smax, f = 18 exactly) kq = 18 code,
# and 126 is reached (code 7: field 7 is hi = 3, lo = 1, i = 0).
law kq_bound:
  for +hi: Nat
  for +lo: Nat
  for +i: Nat
  for +kp: Nat
  for +kn: Nat
  for +a: Nat
  for +b: Nat
  for bpos: {Nat.is_lt(0n, b) == True{} : Bool}
  for f18: {Nat.is_le(a, Nat.mul(b, 18n)) == True{} : Bool}
  for rn: {Impl.kq_rounds(Spec.Z{kp, kn}, hi, lo, i, a, b) == True{} : Bool}
  {Nat.is_le(Spec.abs(Spec.Z{kp, kn}), 126n) == True{} : Bool}

law kq_max_group_exact:
  for +hi: Nat
  for +lo: Nat
  for +i: Nat
  for +kp: Nat
  for +kn: Nat
  for +b: Nat
  for bpos: {Nat.is_lt(0n, b) == True{} : Bool}
  for rn: {Impl.kq_rounds(Spec.Z{kp, kn}, hi, lo, i, Nat.mul(b, 18n), b) == True{} : Bool}
  Spec.zeq(Spec.Z{kp, kn}, Spec.zscale(Impl.code(Impl.field(hi, lo, i)), 18n))

law kq_bound_reached:
  {Impl.field(3n, 1n, 0n) == 7n : Nat} &
  {Impl.kq_rounds(Spec.Z{126n, 0n}, 3n, 1n, 0n, 18n, 1n) == True{} : Bool}

# 4. The IMMA issue pattern: per g iteration the pp loop feeds accumulators 0, 1, 2, 3 once each, so over the g loop
# each of the 4 accumulators receives exactly 8 IMMAs (one per k32 slice) and no other index is fed.
law imma_issue:
  {Impl.pp_loop(2n, 0n) == [0n, 1n, 2n, 3n] : List<&2, Nat>} &
  {Spec.count(0n, Impl.g_loop(8n)) == 8n : Nat} &
  {Spec.count(1n, Impl.g_loop(8n)) == 8n : Nat} &
  {Spec.count(2n, Impl.g_loop(8n)) == 8n : Nat} &
  {Spec.count(3n, Impl.g_loop(8n)) == 8n : Nat} &
  {List.length(&2, Nat, Impl.g_loop(8n)) == 32n : Nat}

# Big constants are written as products of small factors and never as decimal literals: 2^22 = 2048 * 2048 (4194304),
# 2^24 = 4096 * 4096 (16777216), 2^31 = 32768 * 65536 (2147483648), INT32_MAX = 2^31 - 1, and the IMMA partial bound
# P = 32 * (127 * 126) (512064). The checker's Nat is unary: the proven kernel of --verdict cannot evaluate past about 2^21
# steps and the full contract's check overflows its stack on values near 2^19, so the proofs derive every fact about these
# constants from evaluations of at most 127 * 126 = 16002 steps and from monotonicity (pattn8i_int_diff.py checks the
# products against the kernel's decimal / hex constants).

# 5. One IMMA: 32 products of |q| <= 127 and |k| <= 126 sum to |partial| <= P = 32 * (127 * 126) = 512064 < 2^24, and the
# instruction's int32 result from a zero accumulator is that exact partial. P is written as the product (its unary value is
# about 2^19 steps deep, too deep for the full contract's check), 127 * 126 = 16002 is evaluated.
law imma_partial:
  for +q: List<&2, Spec.Z>
  for +k: List<&2, Spec.Z>
  for +ok: {Spec.zip_ok(q, k, 127n, 126n) == True{} : Bool}
  for +n32: {Nat.is_eq(Spec.len(k), 32n) == True{} : Bool}
  {Nat.is_le(Spec.abs(Spec.dot(q, k)), Nat.mul(32n, Nat.mul(127n, 126n))) == True{} : Bool} &
  {Nat.mul(127n, 126n) == 16002n : Nat} &
  {Nat.is_lt(Nat.mul(32n, Nat.mul(127n, 126n)), Nat.mul(4096n, 4096n)) == True{} : Bool} &
  Spec.zeq(Impl.mma(Spec.Z{0n, 0n}, q, k, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Spec.dot(q, k))

# 6. int32 wrap (m = INT32_MAX = 2^31 - 1) is the identity on |z| < 2^31 (and is not beyond: 2^31 wraps to -2^31).
law wrap_identity:
  for +p: Nat
  for +n: Nat
  for small: {Nat.is_lt(Spec.abs(Spec.Z{p, n}), Nat.mul(32768n, 65536n)) == True{} : Bool}
  Spec.zeq(Impl.wrap(Spec.Z{p, n}, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Spec.Z{p, n})

law wrap_at_2_31:
  for +n: Nat
  {Impl.wrap(Spec.Z{Nat.add(n, Nat.mul(32768n, 65536n)), n}, Nat.sub(Nat.mul(32768n, 65536n), 1n))
    == Spec.Z{0n, Nat.mul(32768n, 65536n)} : Spec.Z}

# 7. The chain of g + 1 <= 8 IMMAs (every prefix of the 8): the int32 accumulator equals the exact integer dot product
# over the covered 32 (g + 1) dims, |acc| <= P (g + 1) <= 8 P = 4096512 < 2^22 < 2^31.
law imma_chain_exact:
  for +ch: List<&2, Spec.Chunk>
  for +ok: {Spec.chunks_ok(ch, 127n, 126n) == True{} : Bool}
  for +le8: {Nat.is_le(Spec.chunks_len(ch), 8n) == True{} : Bool}
  Spec.zeq(Impl.chain(ch, Spec.Z{0n, 0n}, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Spec.dot(Spec.flat_q(ch), Spec.flat_k(ch))) &
  {Nat.is_le(Spec.abs(Impl.chain(ch, Spec.Z{0n, 0n}, Nat.sub(Nat.mul(32768n, 65536n), 1n))),
    Nat.mul(Spec.chunks_len(ch), Nat.mul(32n, Nat.mul(127n, 126n)))) == True{} : Bool}

# 7b. The constants of 7: 8 P < 2^22 < 2^31.
law chain_bound_2_22:
  {Nat.is_lt(Nat.mul(8n, Nat.mul(32n, Nat.mul(127n, 126n))), Nat.mul(2048n, 2048n)) == True{} : Bool}

law bound_2_22_lt_2_31:
  {Nat.is_lt(Nat.mul(2048n, 2048n), Nat.mul(32768n, 65536n)) == True{} : Bool}

# 8. i2f_exact (with its constants in terms of h = 2^22 = 2048 * 2048, see pattn8i_int.bend): for -2^22 <= x < 2^22 the
# pattern x + 0x4B400000 is a positive single with exponent field 150 and fraction 2^22 + x, i.e. the integer
# 2^23 + 2^22 + x, and the RN subtraction of 12582912 is exact (|x| <= 2^24): the result is x. The precondition is
# tight: for 2^22 <= x < 3 * 2^22 the exponent field is 151 (the value there is 2^24 + 2 (x - 2^22), so the result is
# 2 x - 2^22: exact only at x = 2^22; the first wrong results, at 2^22 + 1 and -2^22 - 1, are shown by the host C
# evidence of pattn8i_int_diff.py, not by a law).
law i2f_exact_range:
  for +xp: Nat
  for +xn: Nat
  for +lo: {Nat.is_le(xn, Nat.add(xp, Nat.mul(2048n, 2048n))) == True{} : Bool}
  for +hi: {Nat.is_lt(xp, Nat.add(xn, Nat.mul(2048n, 2048n))) == True{} : Bool}
  {Impl.exp_field(Impl.bits(Spec.Z{xp, xn}, Nat.mul(2048n, 2048n)), Nat.mul(2048n, 2048n)) == 150n : Nat} &
  {Impl.frac(Impl.bits(Spec.Z{xp, xn}, Nat.mul(2048n, 2048n)), Nat.mul(2048n, 2048n)) == Nat.sub(Nat.add(xp, Nat.mul(2048n, 2048n)), xn) : Nat} &
  {Impl.i2f_exact(Spec.Z{xp, xn}, Nat.mul(2048n, 2048n)) == Some{Spec.norm(Spec.Z{xp, xn})} : Maybe<&2, Spec.Z>}

# Tightness: at x = 2^22 (every pair naming it) the pattern is 151 * 2^23, exponent field 151, so the precondition
# (field 150) fails.
law i2f_at_2_22:
  for +xn: Nat
  {Impl.exp_field(Impl.bits(Spec.Z{Nat.add(xn, Nat.mul(2048n, 2048n)), xn}, Nat.mul(2048n, 2048n)), Nat.mul(2048n, 2048n)) == 151n : Nat}

# 9. FINDING (ext 3021c differs from the per-group exact-K design): K is rounded. For code 7 and s / smax = 3 / 5
# (f = 54 / 5, code * f = 75.6) both 75 and 76 are adjacent integers, and no integer kq satisfies 5 kq = 7 * 54, so the
# kernel's kq is not code * f; the IMMA partial over int8 K reaches P = 32 * 127 * 126 = 512064 > 32 * 127 * 7 = 28448
# (the per-group design's bound) on admissible operands (q = 127, k = 126 = 18 * 7 on the largest-scale group).
law finding_k_rounded:
  {Impl.kq_rounds(Spec.Z{75n, 0n}, 3n, 1n, 0n, 54n, 5n) == True{} : Bool} &
  {Impl.kq_rounds(Spec.Z{76n, 0n}, 3n, 1n, 0n, 54n, 5n) == True{} : Bool} &
  {Nat.mul(7n, 54n) == 378n : Nat}

law finding_no_exact_kq:
  for +kp: Nat
  for +kn: Nat
  for same: Spec.zeq(Spec.zscale(Spec.Z{kp, kn}, 5n), Spec.zscale(Impl.code(Impl.field(3n, 1n, 0n)), 54n))
  Empty

law finding_partial_bound:
  {Spec.zip_ok(Spec.rep(32n, Spec.Z{127n, 0n}), Spec.rep(32n, Spec.Z{126n, 0n}), 127n, 126n) == True{} : Bool} &
  Spec.zeq(Spec.dot(Spec.rep(32n, Spec.Z{127n, 0n}), Spec.rep(32n, Spec.Z{126n, 0n})), Spec.Z{Nat.mul(32n, Nat.mul(127n, 126n)), 0n}) &
  {Nat.is_lt(Nat.mul(32n, Nat.mul(127n, 7n)), Nat.mul(32n, Nat.mul(127n, 126n))) == True{} : Bool}

# 10. The rounding points of one score, in kernel order: Q rotation, Q quantisation, K requantisation, the row factor
# rs, the key scale skk, rs * sk and the final multiply; the integer core (IMMA chain, i2f_exact) has none (laws 5-8).
law rounding_points:
  {Impl.rounding_points(Spec.score_steps()) ==
    [Spec.QRotate{}, Spec.QQuant{}, Spec.KRequant{}, Spec.RowFactor{}, Spec.KeyScale{}, Spec.ScaleProd{}, Spec.ScoreMul{}]
    : List<&2, Spec.Step>} &
  {Impl.exact_points(Spec.score_steps()) == [Spec.ImmaChain{}, Spec.ToFloat{}] : List<&2, Spec.Step>}
