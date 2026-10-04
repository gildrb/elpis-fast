# Ext 3021c (patches/exl3-ext/3021c-prefill-pattn8-int8qk.patch): the integer core of pattn8i's QK^T, transcribed over
# Nat with integers as offset pairs (Spec.Z{p, n} = p - n). K = exllamav3_ext/pattn8i_kernel.cuh of the patched tree
# (OUT/patched). Covers: the 3-bit K code (stage pass pattn_stage_int8k_kernel), the int8 requantisation of K and Q as
# relations / clamps, the m16n8k32 s32.s8.s8.s32 IMMA chain over the 256 head dims with int32 wrap, and the i2f_exact
# magic conversion as an IEEE single decoding. Floating-point steps other than i2f_exact are not modelled (they are
# the score's rounding points, listed by `rounds`). Reference: pattn8i_int_spec.bend; laws: pattn8i_int_laws.bend;
# source link: pattn8i_int_diff.py.
import Base
import ./pattn8i_int_spec.bend as Spec

def pickz(c: Bool, x: Spec.Z, y: Spec.Z) -> Spec.Z:
  match c:
    case True{}:
      x
    case False{}:
      y

# ---------------------------------------------------------------------------------------------
# K code (stage pass)

# K:478 int code = 2 * (int) ((((hi >> (2 * i)) & 3) << 1) | ((lo >> i) & 1)) - 7;
# The 3-bit field: x >> s = x / 2^s, x & 3 = x mod 4, x & 1 = x mod 2, y << 1 = 2 y; the operands of | have disjoint
# bits (an even number and one bit), so | is +.
def field(hi: Nat, lo: Nat, +i: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.mod(Nat.div(hi, Nat.pow(2n, Nat.mul(2n, i))), 4n), 2n), Nat.mod(Nat.div(lo, Nat.pow(2n, i)), 2n))

# K:478 ... 2 * c - 7 (int arithmetic: the offset pair (2 c, 7))
def code(c: Nat) -> Spec.Z:
  Spec.Z{Nat.mul(c, 2n), 7n}

# K:479 kq[i] = __float2int_rn(__fmul_rn((float) code, f)); with f = a / b, K:473
# f = smax > 0.f ? __fdiv_rn(__fmul_rn(18.f, s), smax) : 0.f. Both roundings are round-to-nearest; every value in
# [floor(code f), ceil(code f)] that RN can produce is an adjacent integer, so the model is the relation
# Spec.adjacent(kq, code, a, b) (fmul_rn(code, f) lies between the integers floor and ceil of code f, both
# representable singles of magnitude <= 127, since RN is monotone; rint of a value in that range stays in it).
def kq_rounds(+r: Spec.Z, hi: Nat, lo: Nat, +i: Nat, +a: Nat, +b: Nat) -> Bool:
  Spec.adjacent(r, code(field(hi, lo, i)), a, b)

# ---------------------------------------------------------------------------------------------
# Q quantisation clamp

# CUDA int min / max: min(a, b) = b < a ? b : a, max(a, b) = a < b ? b : a
def imin(+a: Spec.Z, +b: Spec.Z) -> Spec.Z:
  pickz(Spec.zlt(b, a), b, a)

def imax(+a: Spec.Z, +b: Spec.Z) -> Spec.Z:
  pickz(Spec.zlt(a, b), b, a)

# K:176 qi[i] = max(-127, min(127, __float2int_rn(__fmul_rn(v[i], inv))));  x = the rounded pre-clamp integer; the
# result is an int register value, so canonical (Spec.norm).
def clamp(+x: Spec.Z) -> Spec.Z:
  Spec.norm(imax(Spec.Z{0n, 127n}, imin(Spec.Z{127n, 0n}, x)))

# ---------------------------------------------------------------------------------------------
# IMMA chain (int32)

# int32 two's complement, written in terms of m = INT32_MAX = 2^31 - 1 (so 2^31 = m + 1, 2^32 - 1 = m + 2^31 and
# 2^32 = 2^31 + 2^31; the checker's Nat is unary, so the constants are never evaluated): the bit pattern of p - n is
# (p + (2^32 - 1) n) mod 2^32 (p + (2^32 - 1) n = p - n + 2^32 n); a pattern u decodes to u if u < 2^31, else to u - 2^32.
def enc(z: Spec.Z, +m: Nat) -> Nat:
  match z:
    case Spec.Z{p, n}:
      Nat.mod(Nat.add(p, Nat.mul(n, Nat.add(m, 1n+m))), Nat.add(1n+m, 1n+m))

def dec(+u: Nat, +m: Nat) -> Spec.Z:
  pickz(Nat.is_lt(u, 1n+m), Spec.Z{u, 0n}, Spec.Z{0n, Nat.sub(Nat.add(1n+m, 1n+m), u)})

# int32 wrap of z; the int32 registers use m = INT32_MAX = 2^31 - 1, written Nat.sub(Nat.mul(32768n, 65536n), 1n) in the laws
def wrap(z: Spec.Z, +m: Nat) -> Spec.Z:
  dec(enc(z, m), m)

# K:43 "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
# per output element: d = c + sum of the 32 products a_i b_i, in int32 (PTX: s32 accumulate without .satfinite wraps
# modulo 2^32); the in-instruction order is immaterial for a sum taken modulo 2^32.
def mma_sum(a: List<&2, Spec.Z>, b: List<&2, Spec.Z>) -> Spec.Z:
  match a b:
    case Nil{} _:
      Spec.Z{0n, 0n}
    case _ Nil{}:
      Spec.Z{0n, 0n}
    case x <> xs y <> ys:
      Spec.zadd(Spec.zmul(x, y), mma_sum(xs, ys))

# m = INT32_MAX (2^31 - 1) for the s32 accumulator
def mma(c: Spec.Z, a: List<&2, Spec.Z>, b: List<&2, Spec.Z>, +m: Nat) -> Spec.Z:
  wrap(Spec.zadd(c, mma_sum(a, b)), m)

# K:205 ai[nt][e] = 0; K:207 for (int g = 0; g < 8; ++g): one IMMA per k32 slice g (Q chunk 2g, 2g + 1 of the row,
# K:210; key chunk 2g + (mi & 1), K:216), chained through the accumulator ai[nt].
def chain(ch: List<&2, Spec.Chunk>, acc: Spec.Z, +m: Nat) -> Spec.Z:
  match ch:
    case Nil{}:
      acc
    case Spec.Chunk{q, k} <> t:
      chain(t, mma(acc, q, k, m), m)

# K:212-218 for (int pp = 0; pp < 2; ++pp) { ... imma(ai[2 * pp], ...); imma(ai[2 * pp + 1], ...); }: the accumulator
# indices one g iteration feeds, in issue order.
def pp_loop(left: Nat, +pp: Nat) -> List<&2, Nat>:
  match left:
    case 0n:
      []
    case 1n+l:
      Nat.mul(pp, 2n) <> Nat.add(Nat.mul(pp, 2n), 1n) <> pp_loop(l, 1n+pp)

# K:207 for (int g = 0; g < 8; ++g) around the pp loop
def g_loop(left: Nat) -> List<&2, Nat>:
  match left:
    case 0n:
      []
    case 1n+l:
      List.append(&2, Nat, pp_loop(2n, 0n), g_loop(l))

# ---------------------------------------------------------------------------------------------
# i2f_exact (K:49-52), written in terms of h = 2^22 (4194304; Nat.mul(2048n, 2048n) in the laws) so that no proof step
# evaluates a constant: 2^23 = h + h, 0x4B400000 = 150 * 2^23 + 2^22 = h + 150 (h + h), 12582912.0f = 1.5 * 2^23 = h + (h + h),
# 2^24 = (h + h) + (h + h).

# K:51 __int_as_float(x + 0x4B400000): the int32 sum (no overflow in the domain |x| <= 2^22), as a bit pattern
def bits(x: Spec.Z, +h: Nat) -> Nat:
  match x:
    case Spec.Z{p, n}:
      Nat.sub(Nat.add(p, Nat.add(h, Nat.mul(150n, Nat.add(h, h)))), n)

# IEEE binary32 fields of a pattern u < 2^32: fraction bits 22..0, exponent field bits 30..23, sign bit 31
def frac(u: Nat, +h: Nat) -> Nat:
  Nat.mod(u, Nat.add(h, h))

def exp_field(u: Nat, +h: Nat) -> Nat:
  Nat.mod(Nat.div(u, Nat.add(h, h)), 256n)

def sign_bit(u: Nat, +h: Nat) -> Nat:
  Nat.div(Nat.div(u, Nat.add(h, h)), 256n)

def pickm(c: Bool, x: Maybe<&2, Nat>) -> Maybe<&2, Nat>:
  match c:
    case True{}:
      x
    case False{}:
      None{}

# The value of a positive normal single with exponent field e >= 150: (2^23 + fraction) * 2^(e - 150), an integer.
# Other patterns (negative, or e < 150: not on the integer grid) are not modelled (None).
def f32_int(+u: Nat, +h: Nat) -> Maybe<&2, Nat>:
  pickm(Bool.and(Nat.is_eq(sign_bit(u, h), 0n), Nat.is_le(150n, exp_field(u, h))),
    Some{Nat.mul(Nat.pow(2n, Nat.sub(exp_field(u, h), 150n)), Nat.add(frac(u, h), Nat.add(h, h)))})

def pickr(c: Bool, x: Spec.Z) -> Maybe<&2, Spec.Z>:
  match c:
    case True{}:
      Some{x}
    case False{}:
      None{}

# K:51 __fsub_rn(., 12582912.0f) of an integer-valued single v: the exact difference v - c is an integer; when
# |v - c| <= 2^24 it is a single, so round-to-nearest returns it unchanged (canonical pair). Larger differences are not
# modelled (None).
def fsub_rn_int(+v: Nat, +c: Nat, +h: Nat) -> Maybe<&2, Spec.Z>:
  pickr(Nat.is_le(Spec.abs(Spec.Z{v, c}), Nat.add(Nat.add(h, h), Nat.add(h, h))), Spec.norm(Spec.Z{v, c}))

# K:49-52 float i2f_exact(int x) { return __fsub_rn(__int_as_float(x + 0x4B400000), 12582912.0f); }
def i2f_exact.go(m: Maybe<&2, Nat>, +h: Nat) -> Maybe<&2, Spec.Z>:
  match m:
    case None{}:
      None{}
    case Some{v}:
      fsub_rn_int(v, Nat.add(h, Nat.add(h, h)), h)

def i2f_exact(x: Spec.Z, +h: Nat) -> Maybe<&2, Spec.Z>:
  i2f_exact.go(f32_int(bits(x, h), h), h)

# ---------------------------------------------------------------------------------------------
# Rounding points of one score (sc[nt][e], K:225-228)

# True where the kernel step rounds (an fp32 result that need not equal the exact real value):
# - QRotate: K:158, K:166 __fadd_rn / __fsub_rn butterflies of the H32 rotation of Q
# - QQuant: K:173 inv = __fdiv_rn(127.f, amax); K:176 __float2int_rn(__fmul_rn(v[i], inv))
# - KRequant: K:473 f = __fdiv_rn(__fmul_rn(18.f, s), smax); K:479 __float2int_rn(__fmul_rn((float) code, f))
# - RowFactor: K:179 qsc[row] = __fmul_rn(__fdiv_rn(amax, 127.f), scale_log2)
# - KeyScale: K:448 kscale = 0.17677669529663688110f / 144.0f (an fp32 constant); K:483 __fmul_rn(smax, kscale)
# - ScaleProd: K:225 __fmul_rn(rs0, sk.x) (and the other three sc lines)
# - ImmaChain: K:217-218 int32 IMMA chain: exact (laws imma_partial, imma_chain_exact)
# - ToFloat: K:51 i2f_exact: exact on the chain's range (law i2f_exact_range)
# - ScoreMul: K:225 __fmul_rn(i2f_exact(ai[nt][0]), ...)
def rounds(s: Spec.Step) -> Bool:
  match s:
    case Spec.QRotate{}:
      True{}
    case Spec.QQuant{}:
      True{}
    case Spec.KRequant{}:
      True{}
    case Spec.RowFactor{}:
      True{}
    case Spec.KeyScale{}:
      True{}
    case Spec.ScaleProd{}:
      True{}
    case Spec.ImmaChain{}:
      False{}
    case Spec.ToFloat{}:
      False{}
    case Spec.ScoreMul{}:
      True{}

def keep(c: Bool, s: Spec.Step, t: List<&2, Spec.Step>) -> List<&2, Spec.Step>:
  match c:
    case True{}:
      s <> t
    case False{}:
      t

def rounding_points(xs: List<&2, Spec.Step>) -> List<&2, Spec.Step>:
  match xs:
    case Nil{}:
      []
    case +s <> t:
      keep(rounds(s), s, rounding_points(t))

def exact_points(xs: List<&2, Spec.Step>) -> List<&2, Spec.Step>:
  match xs:
    case Nil{}:
      []
    case +s <> t:
      keep(Bool.not(rounds(s)), s, exact_points(t))
