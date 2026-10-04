# Proofs of qc_staging_laws.bend (ext 9503f / 9503b staging scratch). bit_length: m < 2^bl(m) (with enough fuel) and
# m < 2^b => bl(m) <= b; from these pow2 is the smallest power of two >= n (n >= 1), so it equals the search spec; the
# multiple-of-64 rounding is the smallest multiple >= n by floor-division facts. Nat lemmas from
# gemm_m16_partition_proof.bend (Ar).
import Base
import ./qc_staging.bend as Impl
import ./qc_staging_spec.bend as S
import ./qc_staging_laws.bend as Laws
import ./prefill_membound.bend as MB
import ./prefill_membound_laws.bend as MBL
import ./prefill_membound_proof.bend as MBP
import ./prefill_merge.bend as M
import ./prefill_merge_spec.bend as Spec
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------------------------
# Small helpers

def qs_le_cong(-x: Nat, -x2: Nat, -y: Nat, -y2: Nat, ex: {x == x2 : Nat}, ey: {y == y2 : Nat})
  -> {Nat.is_le(x, y) == Nat.is_le(x2, y2) : Bool}:
  Equal.trans(Bool, Nat.is_le(x, y), Nat.is_le(x2, y), Nat.is_le(x2, y2),
    Equal.cong(Nat, Bool, z => Nat.is_le(z, y), x, x2, ex), Equal.cong(Nat, Bool, z => Nat.is_le(x2, z), y, y2, ey))

def qs_lt_cong(-x: Nat, -x2: Nat, -y: Nat, -y2: Nat, ex: {x == x2 : Nat}, ey: {y == y2 : Nat})
  -> {Nat.is_lt(x, y) == Nat.is_lt(x2, y2) : Bool}:
  Equal.trans(Bool, Nat.is_lt(x, y), Nat.is_lt(x2, y), Nat.is_lt(x2, y2),
    Equal.cong(Nat, Bool, z => Nat.is_lt(z, y), x, x2, ex), Equal.cong(Nat, Bool, z => Nat.is_lt(x2, z), y, y2, ey))

# rewrite a True fact along an equality of Bools
def qs_tr(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool}) -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

# x < y gives x + 1 <= y, and back
def qs_lt_le(+x: Nat, +y: Nat, h: {Nat.is_lt(x, y) == True{} : Bool}) -> {Nat.is_le(1n+x, y) == True{} : Bool}:
  qs_tr(Nat.is_lt(x, y), Nat.is_le(1n+x, y), Ar.ar_lt_succ_le(x, y), h)

def qs_le_lt(+x: Nat, +y: Nat, h: {Nat.is_le(1n+x, y) == True{} : Bool}) -> {Nat.is_lt(x, y) == True{} : Bool}:
  Ar.ard_tr(Nat.is_lt(x, y), Nat.is_le(1n+x, y), Ar.ar_lt_succ_le(x, y), h)

def qs_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      qs_min_le_l(ap, bp)

def qs_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      Ar.ar_h_le_zero(b)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      qs_min_le_r(ap, bp)

def qs_le_min(+x: Nat, +a: Nat, +b: Nat, ha: {Nat.is_le(x, a) == True{} : Bool}, hb: {Nat.is_le(x, b) == True{} : Bool})
  -> {Nat.is_le(x, Nat.min(a, b)) == True{} : Bool}:
  match x a b:
    case 0n _ _:
      Ar.ar_h_le_zero(Nat.min(a, b))
    case 1n+xp 0n _:
      Empty.absurd({Nat.is_le(1n+xp, Nat.min(0n, b)) == True{} : Bool}, Ar.ar_false_true(ha))
    case 1n+xp 1n+ap 0n:
      Empty.absurd({Nat.is_le(1n+xp, Nat.min(1n+ap, 0n)) == True{} : Bool}, Ar.ar_false_true(hb))
    case 1n+xp 1n+ap 1n+bp:
      qs_le_min(xp, ap, bp, ha, hb)

# 0 < 2^k
def qs_pow_pos(+k: Nat) -> {Nat.is_lt(0n, Nat.pow(2n, k)) == True{} : Bool}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      Ar.ar_lt_le_trans(0n, Nat.pow(2n, j), Nat.add(Nat.pow(2n, j), Nat.mul(1n, Nat.pow(2n, j))), qs_pow_pos(j),
        Ar.ar_le_add_r(Nat.pow(2n, j), Nat.mul(1n, Nat.pow(2n, j))))

# 2^(1 + k) = 2^k * 2
def qs_pow_succ(+k: Nat) -> {Nat.pow(2n, 1n+k) == Nat.mul(Nat.pow(2n, k), 2n) : Nat}:
  Ar.ar_mul_comm(2n, Nat.pow(2n, k))

# a <= b gives 2^a <= 2^b
def qs_pow_mono(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_le(Nat.pow(2n, a), Nat.pow(2n, b)) == True{} : Bool}:
  match a b:
    case 0n _:
      qs_lt_le(0n, Nat.pow(2n, b), qs_pow_pos(b))
    case 1n+ap 0n:
      Empty.absurd({Nat.is_le(Nat.pow(2n, 1n+ap), Nat.pow(2n, 0n)) == True{} : Bool}, Ar.ar_false_true(h))
    case 1n+ap 1n+bp:
      qs_tr(Nat.is_le(Nat.mul(Nat.pow(2n, ap), 2n), Nat.mul(Nat.pow(2n, bp), 2n)), Nat.is_le(Nat.pow(2n, 1n+ap), Nat.pow(2n, 1n+bp)),
        qs_le_cong(Nat.mul(Nat.pow(2n, ap), 2n), Nat.pow(2n, 1n+ap), Nat.mul(Nat.pow(2n, bp), 2n), Nat.pow(2n, 1n+bp),
          Equal.sym(Nat, Nat.pow(2n, 1n+ap), Nat.mul(Nat.pow(2n, ap), 2n), qs_pow_succ(ap)),
          Equal.sym(Nat, Nat.pow(2n, 1n+bp), Nat.mul(Nat.pow(2n, bp), 2n), qs_pow_succ(bp))),
        Ar.ar_le_mul_mono(Nat.pow(2n, ap), Nat.pow(2n, bp), 2n, qs_pow_mono(ap, bp, h)))

# max(1, x) = x for x >= 1
def qs_max1(+x: Nat, pos: {Nat.is_lt(0n, x) == True{} : Bool}) -> {Nat.max(1n, x) == x : Nat}:
  match x:
    case 0n:
      Empty.absurd({Nat.max(1n, 0n) == 0n : Nat}, Ar.ar_false_true(pos))
    case 1n+y:
      {==}

# ---------------------------------------------------------------------------------------------
# bit_length

# (1 + p) / 2 <= p
def qs_half_le(+p: Nat) -> {Nat.is_le(Nat.div(1n+p, 2n), p) == True{} : Bool}:
  +two = Ar.ar_mul_comm(1n+p, 2n)
  +Y = Nat.add(p, 0n)
  +h = qs_tr(Nat.is_le(p, Nat.add(p, Y)), Nat.is_le(1n+p, Nat.add(p, 1n+Y)),
    Equal.cong(Nat, Bool, z => Nat.is_le(1n+p, z), 1n+Nat.add(p, Y), Nat.add(p, 1n+Y), Equal.sym(Nat, Nat.add(p, 1n+Y), 1n+Nat.add(p, Y), Ar.ar_add_succ_r(p, Y))),
    Ar.ar_le_add_r(p, Y))
  +big = Equal.trans(Bool, Nat.is_lt(1n+p, Nat.mul(1n+p, 2n)), Nat.is_lt(1n+p, Nat.mul(2n, 1n+p)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(1n+p, z), Nat.mul(1n+p, 2n), Nat.mul(2n, 1n+p), two),
    qs_le_lt(p, Nat.add(p, 1n+Y), h))
  +lt = Ar.ard_tr(Nat.is_lt(Nat.div(1n+p, 2n), 1n+p), Nat.is_lt(1n+p, Nat.mul(1n+p, 2n)), Ar.ar_div_lt(1n+p, 1n+p, 2n, {==}), big)
  qs_lt_le(Nat.div(1n+p, 2n), 1n+p, lt)

# (1 + p) < 2^(1 + b) gives (1 + p) / 2 < 2^b, and back
def qs_half_lt(+p: Nat, +b: Nat, h: {Nat.is_lt(1n+p, Nat.pow(2n, 1n+b)) == True{} : Bool})
  -> {Nat.is_lt(Nat.div(1n+p, 2n), Nat.pow(2n, b)) == True{} : Bool}:
  Ar.ard_tr(Nat.is_lt(Nat.div(1n+p, 2n), Nat.pow(2n, b)), Nat.is_lt(1n+p, Nat.mul(Nat.pow(2n, b), 2n)), Ar.ar_div_lt(Nat.pow(2n, b), 1n+p, 2n, {==}),
    qs_tr(Nat.is_lt(1n+p, Nat.pow(2n, 1n+b)), Nat.is_lt(1n+p, Nat.mul(Nat.pow(2n, b), 2n)),
      Equal.cong(Nat, Bool, z => Nat.is_lt(1n+p, z), Nat.pow(2n, 1n+b), Nat.mul(Nat.pow(2n, b), 2n), qs_pow_succ(b)), h))

def qs_half_up(+p: Nat, +b: Nat, h: {Nat.is_lt(Nat.div(1n+p, 2n), Nat.pow(2n, b)) == True{} : Bool})
  -> {Nat.is_lt(1n+p, Nat.pow(2n, 1n+b)) == True{} : Bool}:
  qs_tr(Nat.is_lt(1n+p, Nat.mul(Nat.pow(2n, b), 2n)), Nat.is_lt(1n+p, Nat.pow(2n, 1n+b)),
    Equal.cong(Nat, Bool, z => Nat.is_lt(1n+p, z), Nat.mul(Nat.pow(2n, b), 2n), Nat.pow(2n, 1n+b),
      Equal.sym(Nat, Nat.pow(2n, 1n+b), Nat.mul(Nat.pow(2n, b), 2n), qs_pow_succ(b))),
    qs_tr(Nat.is_lt(Nat.div(1n+p, 2n), Nat.pow(2n, b)), Nat.is_lt(1n+p, Nat.mul(Nat.pow(2n, b), 2n)), Ar.ar_div_lt(Nat.pow(2n, b), 1n+p, 2n, {==}), h))

# m <= f gives m < 2^bl.go(f, m)
def qs_bl_upper(+f: Nat, +m: Nat, hf: {Nat.is_le(m, f) == True{} : Bool})
  -> {Nat.is_lt(m, Nat.pow(2n, MB.bl.go(f, m))) == True{} : Bool}:
  match f m:
    case 0n 0n:
      {==}
    case 0n 1n+p:
      Empty.absurd({Nat.is_lt(1n+p, Nat.pow(2n, MB.bl.go(0n, 1n+p))) == True{} : Bool}, Ar.ar_false_true(hf))
    case 1n+g 0n:
      {==}
    case 1n+g 1n+p:
      +le = Ar.ar_le_trans(Nat.div(1n+p, 2n), p, g, qs_half_le(p), hf)
      qs_half_up(p, MB.bl.go(g, Nat.div(1n+p, 2n)), qs_bl_upper(g, Nat.div(1n+p, 2n), le))

# m < 2^b gives bl.go(f, m) <= b (any fuel)
def qs_bl_lower(+b: Nat, +f: Nat, +m: Nat, h: {Nat.is_lt(m, Nat.pow(2n, b)) == True{} : Bool})
  -> {Nat.is_le(MB.bl.go(f, m), b) == True{} : Bool}:
  match b f m:
    case _ 0n _:
      Ar.ar_h_le_zero(b)
    case _ 1n+g 0n:
      Ar.ar_h_le_zero(b)
    case 0n 1n+g 1n+p:
      Empty.absurd({Nat.is_le(MB.bl.go(1n+g, 1n+p), 0n) == True{} : Bool},
        Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(p, 0n), True{}, Equal.sym(Bool, Nat.is_lt(p, 0n), False{}, Ar.ar_h_lt_zero(p)), h)))
    case 1n+c 1n+g 1n+p:
      qs_bl_lower(c, g, Nat.div(1n+p, 2n), qs_half_lt(p, c, h))

# bl.go(f, m) <= f
def qs_bl_fuel(+f: Nat, +m: Nat) -> {Nat.is_le(MB.bl.go(f, m), f) == True{} : Bool}:
  match f m:
    case 0n _:
      {==}
    case 1n+g 0n:
      {==}
    case 1n+g 1n+p:
      qs_bl_fuel(g, Nat.div(1n+p, 2n))

# ---------------------------------------------------------------------------------------------
# pow2: the smallest power of two >= n (n >= 1)

def qs_pow2_eq(+n: Nat) -> {Impl.pow2(n) == Nat.pow(2n, MB.bit_length_dec(n)) : Nat}:
  qs_max1(Nat.pow(2n, MB.bit_length_dec(n)), qs_pow_pos(MB.bit_length_dec(n)))

def qs_pow2_ge(+n: Nat) -> {Nat.is_le(n, Impl.pow2(n)) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Impl.pow2(1n+p), Nat.pow(2n, MB.bl.go(p, p)), qs_pow2_eq(1n+p)) : {Nat.is_le(1n+p, _) == True{} : Bool}
      qs_lt_le(p, Nat.pow(2n, MB.bl.go(p, p)), qs_bl_upper(p, p, Ar.ar_le_refl(p)))

# 1 + p <= 2^b gives pow2(1 + p) <= 2^b
def qs_pow2_min(+p: Nat, +b: Nat, h: {Nat.is_le(1n+p, Nat.pow(2n, b)) == True{} : Bool})
  -> {Nat.is_le(Impl.pow2(1n+p), Nat.pow(2n, b)) == True{} : Bool}:
  %Equal.sym(Nat, Impl.pow2(1n+p), Nat.pow(2n, MB.bl.go(p, p)), qs_pow2_eq(1n+p)) : {Nat.is_le(_, Nat.pow(2n, b)) == True{} : Bool}
  qs_pow_mono(MB.bl.go(p, p), b, qs_bl_lower(b, p, p, qs_le_lt(p, Nat.pow(2n, b), h)))

def qs_pow2_mono(+n: Nat, +m: Nat, lo: {Nat.is_le(1n, n) == True{} : Bool}, nm: {Nat.is_le(n, m) == True{} : Bool})
  -> {Nat.is_le(Impl.pow2(n), Impl.pow2(m)) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({Nat.is_le(Impl.pow2(0n), Impl.pow2(m)) == True{} : Bool}, Ar.ar_false_true(lo))
    case 1n+p:
      +e = qs_pow2_eq(m)
      +ge = qs_tr(Nat.is_le(m, Impl.pow2(m)), Nat.is_le(m, Nat.pow(2n, MB.bit_length_dec(m))),
        Equal.cong(Nat, Bool, z => Nat.is_le(m, z), Impl.pow2(m), Nat.pow(2n, MB.bit_length_dec(m)), e), qs_pow2_ge(m))
      %Equal.sym(Nat, Impl.pow2(m), Nat.pow(2n, MB.bit_length_dec(m)), e) : {Nat.is_le(Impl.pow2(1n+p), _) == True{} : Bool}
      qs_pow2_min(p, MB.bit_length_dec(m), Ar.ar_le_trans(1n+p, m, Nat.pow(2n, MB.bit_length_dec(m)), nm, ge))

# ---------------------------------------------------------------------------------------------
# ceil64: the smallest multiple of 64 >= n

def qs_Q(+n: Nat) -> Nat:
  Nat.div(Nat.add(n, 63n), 64n)

# n + 63 <= x + 63 iff n <= x
def qs_add63(+n: Nat, +x: Nat) -> {Nat.is_le(Nat.add(n, 63n), Nat.add(x, 63n)) == Nat.is_le(n, x) : Bool}:
  %Ar.ar_add_comm(63n, n) : {Nat.is_le(_, Nat.add(x, 63n)) == Nat.is_le(n, x) : Bool}
  %Ar.ar_add_comm(63n, x) : {Nat.is_le(Nat.add(63n, n), _) == Nat.is_le(n, x) : Bool}
  Ar.ar_le_add_cancel(63n, n, x)

def qs_c64_ge(+n: Nat) -> {Nat.is_le(n, Impl.ceil64(n)) == True{} : Bool}:
  +C = Impl.ceil64(n)
  +r = Nat.mod(Nat.add(n, 63n), 64n)
  +rle = qs_lt_le(r, 64n, Ar.ard_dm_lt(Nat.add(n, 63n), 64n, {==}))
  +a = qs_tr(Nat.is_le(r, 63n), Nat.is_le(Nat.add(C, r), Nat.add(C, 63n)), Equal.sym(Bool, Nat.is_le(Nat.add(C, r), Nat.add(C, 63n)), Nat.is_le(r, 63n), Ar.ar_le_add_cancel(C, r, 63n)), rle)
  +b = qs_tr(Nat.is_le(Nat.add(C, r), Nat.add(C, 63n)), Nat.is_le(Nat.add(n, 63n), Nat.add(C, 63n)),
    Equal.cong(Nat, Bool, z => Nat.is_le(z, Nat.add(C, 63n)), Nat.add(C, r), Nat.add(n, 63n), Ar.ard_dm_eq(Nat.add(n, 63n), 64n, {==})), a)
  qs_tr(Nat.is_le(Nat.add(n, 63n), Nat.add(C, 63n)), Nat.is_le(n, C), qs_add63(n, C), b)

def qs_c64_lt(+n: Nat) -> {Nat.is_lt(Impl.ceil64(n), Nat.add(n, 64n)) == True{} : Bool}:
  Ar.ar_le_lt_trans(Impl.ceil64(n), Nat.add(n, 63n), Nat.add(n, 64n), Ar.ard_le_mul_div(Nat.add(n, 63n), 64n, {==}),
    Ar.ard_tr(Nat.is_lt(Nat.add(n, 63n), Nat.add(n, 64n)), Nat.is_lt(63n, 64n), Ar.ar_lt_add_cancel(n, 63n, 64n), {==}))

def qs_c64_mono(+n: Nat, +m: Nat, nm: {Nat.is_le(n, m) == True{} : Bool}) -> {Nat.is_le(Impl.ceil64(n), Impl.ceil64(m)) == True{} : Bool}:
  +le = Ar.ard_tr(Nat.is_le(Nat.add(n, 63n), Nat.add(m, 63n)), Nat.is_le(n, m), qs_add63(n, m), nm)
  Ar.ar_le_mul_mono(qs_Q(n), qs_Q(m), 64n, Ar.ar_div_mono(Nat.add(n, 63n), Nat.add(m, 63n), 64n, {==}, le))

def qs_succ_comm63(+p: Nat) -> {Nat.add(1n+p, 63n) == Nat.add(Nat.mul(1n, 64n), p) : Nat}:
  Equal.cong(Nat, Nat, z => 1n+z, Nat.add(p, 63n), Nat.add(63n, p), Ar.ar_add_comm(p, 63n))

# 1 + p <= 64 gives ceil64(1 + p) = 64
def qs_c64_small(+p: Nat, h: {Nat.is_le(1n+p, 64n) == True{} : Bool}) -> {Impl.ceil64(1n+p) == 64n : Nat}:
  +lt = qs_le_lt(p, 64n, h)
  +e = qs_succ_comm63(p)
  +d = Ar.ard_fst({Nat.div(Nat.add(Nat.mul(1n, 64n), p), 64n) == 1n : Nat}, {Nat.mod(Nat.add(Nat.mul(1n, 64n), p), 64n) == p : Nat},
    Ar.ar_div_unique(1n, p, 64n, lt))
  +q = Equal.trans(Nat, Nat.div(Nat.add(1n+p, 63n), 64n), Nat.div(Nat.add(Nat.mul(1n, 64n), p), 64n), 1n,
    Equal.cong(Nat, Nat, z => Nat.div(z, 64n), Nat.add(1n+p, 63n), Nat.add(Nat.mul(1n, 64n), p), e), d)
  Equal.cong(Nat, Nat, z => Nat.mul(z, 64n), qs_Q(1n+p), 1n, q)

# ---------------------------------------------------------------------------------------------
# The reference searches

# k <= K, k != K gives 1 + k <= K
def qs_step_le(+k: Nat, +K: Nat, hk: {Nat.is_le(k, K) == True{} : Bool}, ne: {Nat.is_eq(k, K) == False{} : Bool})
  -> {Nat.is_le(1n+k, K) == True{} : Bool}:
  qs_lt_le(k, K, Ar.p3_le_ne_lt(k, K, hk, ne))

# K <= 1 + (g + k) gives K <= g + (1 + k)
def qs_fuel_step(+K: Nat, +g: Nat, +k: Nat, h: {Nat.is_le(K, 1n+Nat.add(g, k)) == True{} : Bool})
  -> {Nat.is_le(K, Nat.add(g, 1n+k)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(g, 1n+k), 1n+Nat.add(g, k), Ar.ar_add_succ_r(g, k)) : {Nat.is_le(K, _) == True{} : Bool}
  h

# a miss at k <= K = bl(p) (1 + p > 2^k) gives 1 + k <= K
def qs_sp_lt(+k: Nat, +p: Nat, hc: {Nat.is_le(1n+p, Nat.pow(2n, k)) == False{} : Bool}, hk: {Nat.is_le(k, MB.bl.go(p, p)) == True{} : Bool})
  -> {Nat.is_le(1n+k, MB.bl.go(p, p)) == True{} : Bool}:
  +K = MB.bl.go(p, p)
  +up = qs_lt_le(p, Nat.pow(2n, K), qs_bl_upper(p, p, Ar.ar_le_refl(p)))
  +ne = Ar.ar_eq_false(k, K, e => Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_le(1n+p, Nat.pow(2n, k)), True{},
    Equal.sym(Bool, Nat.is_le(1n+p, Nat.pow(2n, k)), False{}, hc),
    qs_tr(Nat.is_le(1n+p, Nat.pow(2n, K)), Nat.is_le(1n+p, Nat.pow(2n, k)),
      Equal.cong(Nat, Bool, z => Nat.is_le(1n+p, Nat.pow(2n, z)), K, k, Equal.sym(Nat, k, K, e)), up))))
  qs_step_le(k, K, hk, ne)

# with c = (1 + p <= 2^k), k <= K = bl(p) and K <= f + k, the power search from 2^k ends at 2^K
def qs_sp_go(+f: Nat, c: Bool, +k: Nat, +p: Nat,
    hc: {Nat.is_le(1n+p, Nat.pow(2n, k)) == c : Bool},
    hk: {Nat.is_le(k, MB.bl.go(p, p)) == True{} : Bool},
    hf: {Nat.is_le(MB.bl.go(p, p), Nat.add(f, k)) == True{} : Bool})
  -> {S.spow2.go(f, c, k, 1n+p) == Nat.pow(2n, MB.bl.go(p, p)) : Nat}:
  match f c:
    case 0n _:
      Equal.cong(Nat, Nat, z => Nat.pow(2n, z), k, MB.bl.go(p, p), Ar.ar_le_antisym(k, MB.bl.go(p, p), hk, hf))
    case 1n+g True{}:
      +lo = qs_bl_lower(k, p, p, qs_le_lt(p, Nat.pow(2n, k), hc))
      Equal.cong(Nat, Nat, z => Nat.pow(2n, z), k, MB.bl.go(p, p), Ar.ar_le_antisym(k, MB.bl.go(p, p), hk, lo))
    case 1n+g False{}:
      qs_sp_go(g, Nat.is_le(1n+p, Nat.pow(2n, 1n+k)), 1n+k, p, {==}, qs_sp_lt(k, p, hc, hk), qs_fuel_step(MB.bl.go(p, p), g, k, hf))

def qs_pow2_spec(+p: Nat) -> {Impl.pow2(1n+p) == S.spow2(1n+p) : Nat}:
  +K = MB.bl.go(p, p)
  +fu = Ar.ar_le_trans(K, p, 1n+Nat.add(p, 0n), qs_bl_fuel(p, p),
    qs_tr(Nat.is_le(p, 1n+p), Nat.is_le(p, 1n+Nat.add(p, 0n)),
      Equal.cong(Nat, Bool, z => Nat.is_le(p, 1n+z), p, Nat.add(p, 0n), Equal.sym(Nat, Nat.add(p, 0n), p, Ar.ar_add_zero_r(p))), Ar.ar_h_le_succ(p)))
  Equal.trans(Nat, Impl.pow2(1n+p), Nat.pow(2n, K), S.spow2(1n+p), qs_pow2_eq(1n+p),
    Equal.sym(Nat, S.spow2(1n+p), Nat.pow(2n, K), qs_sp_go(1n+p, Nat.is_le(1n+p, 1n), 0n, p, {==}, Ar.ar_h_le_zero(K), fu)))

# Q = ceil64(n) / 64 <= q when n <= 64 q
def qs_Q_le(+n: Nat, +q: Nat, h: {Nat.is_le(n, Nat.mul(q, 64n)) == True{} : Bool}) -> {Nat.is_le(qs_Q(n), q) == True{} : Bool}:
  +X = Nat.mul(q, 64n)
  +a = Ar.ard_tr(Nat.is_le(Nat.add(n, 63n), Nat.add(X, 63n)), Nat.is_le(n, X), qs_add63(n, X), h)
  +b = Ar.ar_le_lt_trans(Nat.add(n, 63n), Nat.add(X, 63n), Nat.add(X, 64n), a,
    Ar.ard_tr(Nat.is_lt(Nat.add(X, 63n), Nat.add(X, 64n)), Nat.is_lt(63n, 64n), Ar.ar_lt_add_cancel(X, 63n, 64n), {==}))
  +c = qs_tr(Nat.is_lt(Nat.add(n, 63n), Nat.add(X, 64n)), Nat.is_lt(Nat.add(n, 63n), Nat.mul(1n+q, 64n)),
    Equal.cong(Nat, Bool, z => Nat.is_lt(Nat.add(n, 63n), z), Nat.add(X, 64n), Nat.mul(1n+q, 64n), Ar.ar_add_comm(X, 64n)), b)
  +d = qs_tr(Nat.is_lt(Nat.add(n, 63n), Nat.mul(1n+q, 64n)), Nat.is_lt(qs_Q(n), 1n+q),
    Equal.sym(Bool, Nat.is_lt(qs_Q(n), 1n+q), Nat.is_lt(Nat.add(n, 63n), Nat.mul(1n+q, 64n)), Ar.ar_div_lt(1n+q, Nat.add(n, 63n), 64n, {==})), c)
  qs_lt_le(qs_Q(n), 1n+q, d)

# a miss at q <= Q (n > 64 q) gives 1 + q <= Q
def qs_sm_lt(+q: Nat, +n: Nat, hc: {Nat.is_le(n, Nat.mul(q, 64n)) == False{} : Bool}, hq: {Nat.is_le(q, qs_Q(n)) == True{} : Bool})
  -> {Nat.is_le(1n+q, qs_Q(n)) == True{} : Bool}:
  +Q = qs_Q(n)
  +ne = Ar.ar_eq_false(q, Q, e => Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_le(n, Nat.mul(q, 64n)), True{},
    Equal.sym(Bool, Nat.is_le(n, Nat.mul(q, 64n)), False{}, hc),
    qs_tr(Nat.is_le(n, Nat.mul(Q, 64n)), Nat.is_le(n, Nat.mul(q, 64n)),
      Equal.cong(Nat, Bool, z => Nat.is_le(n, Nat.mul(z, 64n)), Q, q, Equal.sym(Nat, q, Q, e)), qs_c64_ge(n)))))
  qs_step_le(q, Q, hq, ne)

# with c = (n <= 64 q), q <= Q and Q <= f + q, the multiple search from 64 q ends at 64 Q = ceil64(n)
def qs_sm_go(+f: Nat, c: Bool, +q: Nat, +n: Nat,
    hc: {Nat.is_le(n, Nat.mul(q, 64n)) == c : Bool},
    hq: {Nat.is_le(q, qs_Q(n)) == True{} : Bool},
    hf: {Nat.is_le(qs_Q(n), Nat.add(f, q)) == True{} : Bool})
  -> {S.smul64.go(f, c, q, n) == Impl.ceil64(n) : Nat}:
  match f c:
    case 0n _:
      Equal.cong(Nat, Nat, z => Nat.mul(z, 64n), q, qs_Q(n), Ar.ar_le_antisym(q, qs_Q(n), hq, hf))
    case 1n+g True{}:
      Equal.cong(Nat, Nat, z => Nat.mul(z, 64n), q, qs_Q(n), Ar.ar_le_antisym(q, qs_Q(n), hq, qs_Q_le(n, q, hc)))
    case 1n+g False{}:
      qs_sm_go(g, Nat.is_le(n, Nat.mul(1n+q, 64n)), 1n+q, n, {==}, qs_sm_lt(q, n, hc, hq), qs_fuel_step(qs_Q(n), g, q, hf))

# Q <= n
def qs_Q_fuel(+n: Nat) -> {Nat.is_le(qs_Q(n), Nat.add(n, 0n)) == True{} : Bool}:
  +X = Nat.mul(n, 64n)
  +nx = qs_tr(Nat.is_le(n, Nat.add(n, Nat.mul(63n, n))), Nat.is_le(n, X),
    Equal.cong(Nat, Bool, z => Nat.is_le(n, z), Nat.add(n, Nat.mul(63n, n)), X, Ar.ar_mul_comm(64n, n)), Ar.ar_le_add_r(n, Nat.mul(63n, n)))
  %Equal.sym(Nat, Nat.add(n, 0n), n, Ar.ar_add_zero_r(n)) : {Nat.is_le(qs_Q(n), _) == True{} : Bool}
  qs_Q_le(n, n, nx)

def qs_c64_spec(+n: Nat) -> {Impl.ceil64(n) == S.smul64(n) : Nat}:
  Equal.sym(Nat, S.smul64(n), Impl.ceil64(n), qs_sm_go(n, Nat.is_le(n, 0n), 0n, n, {==}, Ar.ar_h_le_zero(qs_Q(n)), qs_Q_fuel(n)))

# ---------------------------------------------------------------------------------------------
# Laws

def Laws.covers(n):
  qs_le_min(n, Impl.pow2(n), Impl.ceil64(n), qs_pow2_ge(n), qs_c64_ge(n))

def Laws.never_larger(n):
  qs_min_le_l(Impl.pow2(n), Impl.ceil64(n))

def Laws.tight(n):
  Ar.ar_le_lt_trans(Impl.round64(n), Impl.ceil64(n), Nat.add(n, 64n), qs_min_le_r(Impl.pow2(n), Impl.ceil64(n)), qs_c64_lt(n))

def qs_short(+n: Nat, lo: {Nat.is_le(1n, n) == True{} : Bool}, hi: {Nat.is_le(n, 64n) == True{} : Bool})
  -> {Impl.round64(n) == Impl.pow2(n) : Nat}:
  match n:
    case 0n:
      Empty.absurd({Impl.round64(0n) == Impl.pow2(0n) : Nat}, Ar.ar_false_true(lo))
    case 1n+p:
      +h = hi
      %Equal.sym(Nat, Impl.ceil64(1n+p), 64n, qs_c64_small(p, h)) : {Nat.min(Impl.pow2(1n+p), _) == Impl.pow2(1n+p) : Nat}
      Ar.ar_min_left(Impl.pow2(1n+p), 64n, qs_pow2_min(p, 6n, h))

def Laws.short_unchanged(n, lo, hi):
  qs_short(n, lo, hi)

def qs_mono(+n: Nat, +m: Nat, nm: {Nat.is_le(n, m) == True{} : Bool})
  -> {Nat.is_le(Impl.round64(n), Impl.round64(m)) == True{} : Bool}:
  match n:
    case 0n:
      Ar.ar_h_le_zero(Impl.round64(m))
    case 1n+p:
      +h = nm
      +r = Impl.round64(1n+p)
      qs_le_min(r, Impl.pow2(m), Impl.ceil64(m),
        Ar.ar_le_trans(r, Impl.pow2(1n+p), Impl.pow2(m), qs_min_le_l(Impl.pow2(1n+p), Impl.ceil64(1n+p)), qs_pow2_mono(1n+p, m, Ar.ar_h_le_zero(p), h)),
        Ar.ar_le_trans(r, Impl.ceil64(1n+p), Impl.ceil64(m), qs_min_le_r(Impl.pow2(1n+p), Impl.ceil64(1n+p)), qs_c64_mono(1n+p, m, h)))

def Laws.monotone(n, m, nm):
  qs_mono(n, m, nm)

def qs_round_spec(+n: Nat) -> {Impl.round64(n) == S.sround64(n) : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Impl.pow2(1n+p), S.spow2(1n+p), qs_pow2_spec(p)) : {Nat.min(_, Impl.ceil64(1n+p)) == S.sround64(1n+p) : Nat}
      %Equal.sym(Nat, Impl.ceil64(1n+p), S.smul64(1n+p), qs_c64_spec(1n+p)) : {Nat.min(S.spow2(1n+p), _) == S.sround64(1n+p) : Nat}
      {==}

def Laws.impl_is_spec(n):
  (qs_c64_spec(n), qs_round_spec(n), {==})

def qs_stage_spec(env: S.QcEnv, +n: Nat, ps: {Impl.pow2(n) == S.spow2(n) : Nat}, rs: {Impl.round64(n) == S.sround64(n) : Nat})
  -> {Impl.stage(env, n) == S.sstage(env, n) : S.QcOut}:
  match env:
    case S.QcUnset{}:
      Equal.cong(Nat, S.QcOut, z => S.QcPages{z}, Impl.round64(n), S.sround64(n), rs)
    case S.QcOne{}:
      Equal.cong(Nat, S.QcOut, z => S.QcPages{z}, Impl.round64(n), S.sround64(n), rs)
    case S.QcZero{}:
      Equal.cong(Nat, S.QcOut, z => S.QcPages{z}, Impl.pow2(n), S.spow2(n), ps)
    case S.QcOther{}:
      {==}

def qs_pos_spec(env: S.QcEnv, +n: Nat, lo: {Nat.is_le(1n, n) == True{} : Bool})
  -> {Impl.pow2(n) == S.spow2(n) : Nat} & {Impl.stage(env, n) == S.sstage(env, n) : S.QcOut}:
  match n:
    case 0n:
      Empty.absurd({Impl.pow2(0n) == S.spow2(0n) : Nat} & {Impl.stage(env, 0n) == S.sstage(env, 0n) : S.QcOut}, Ar.ar_false_true(lo))
    case 1n+p:
      +ps = qs_pow2_spec(p)
      +rs = qs_round_spec(1n+p)
      (ps, qs_stage_spec(env, 1n+p, ps, rs))

def Laws.impl_is_spec_pos(env, n, lo):
  qs_pos_spec(env, n, lo)

def Laws.served_numbers():
  ({==}, {==}, {==}, {==}, {==}, {==}, {==}, {==}, {==})

def Laws.within_1088(n, small):
  Ar.ar_le_trans(Impl.round64(n), Impl.round64(1025n), 1088n, qs_mono(n, 1025n, small), {==})

def qs_default(env: S.QcEnv, +n: Nat, d: {S.rounds64(env) == True{} : Bool}) -> {Impl.stage(env, n) == S.QcPages{Impl.round64(n)} : S.QcOut}:
  match env:
    case S.QcUnset{}:
      {==}
    case S.QcOne{}:
      {==}
    case S.QcZero{}:
      Empty.absurd({Impl.stage(S.QcZero{}, n) == S.QcPages{Impl.round64(n)} : S.QcOut}, Ar.ar_false_true(d))
    case S.QcOther{}:
      Empty.absurd({Impl.stage(S.QcOther{}, n) == S.QcPages{Impl.round64(n)} : S.QcOut}, Ar.ar_false_true(d))

def Laws.served_is_5110g(fuel, env, n, en, k, T, mc, P, small, default):
  +w = Laws.within_1088(n, small)
  (qs_default(env, n, default), MBL.within_bound_is_5110g(fuel, en, k, T, mc, P, 262143n, Impl.round64(n), 1088n, w))

def Laws.killswitch_1025_unmerged(fuel, en, k, T, mc, P):
  MBL.above_stage_schedule(fuel, en, k, T, mc, P, 262143n, Impl.pow2(1025n), 1088n, {==})

def Laws.zero_pages():
  ({==}, {==}, {==}, {==}, {==})
