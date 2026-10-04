# Proofs of the row-invariant verify-attention partition laws (attn_rowinv_laws.bend) about the Nat
# model attn_rowinv.bend. Structural evidence only: no unsafe definitions, axioms, holes, foreign
# proof code or answer tables. Arithmetic lemmas come from the proven m16 partition library, the
# attn_split proofs (ceil division) and the attn_bounds proofs.
import Base
import ./attn_rowinv.bend as Impl
import ./attn_rowinv_spec.bend as Spec
import ./attn_rowinv_laws.bend as Laws
import ./attn_split.bend as Split
import ./attn_split_proof.bend as AS
import ./attn_bounds_proof.bend as AB
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------- Nat / Bool helpers

def rb_le_add_r(+a: Nat, +b: Nat, +t: Nat, ab: {Nat.is_le(a, b) == True{} : Bool})
  -> {Nat.is_le(Nat.add(a, t), Nat.add(b, t)) == True{} : Bool}:
  %Ar.ar_add_comm(t, a) : {Nat.is_le(_, Nat.add(b, t)) == True{} : Bool}
  %Ar.ar_add_comm(t, b) : {Nat.is_le(Nat.add(t, a), _) == True{} : Bool}
  Ar.ard_tr(Nat.is_le(Nat.add(t, a), Nat.add(t, b)), Nat.is_le(a, b), Ar.ar_le_add_cancel(t, a, b), ab)

# (c + x) + y == (c + y) + x
def rb_swap_tail(+c: Nat, +x: Nat, +y: Nat) -> {Nat.add(Nat.add(c, x), y) == Nat.add(Nat.add(c, y), x) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.add(c, x), y), Nat.add(c, Nat.add(x, y)), Ar.ar_add_assoc(c, x, y)) : {_ == Nat.add(Nat.add(c, y), x) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(c, y), x), Nat.add(c, Nat.add(y, x)), Ar.ar_add_assoc(c, y, x)) : {Nat.add(c, Nat.add(x, y)) == _ : Nat}
  %Ar.ar_add_comm(x, y) : {Nat.add(c, Nat.add(x, y)) == Nat.add(c, _) : Nat}
  {==}

def rb_mul_mono_r(+a: Nat, +b: Nat, +c: Nat, h: {Nat.is_le(a, b) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(c, a), Nat.mul(c, b)) == True{} : Bool}:
  %Ar.ar_mul_comm(a, c) : {Nat.is_le(_, Nat.mul(c, b)) == True{} : Bool}
  %Ar.ar_mul_comm(b, c) : {Nat.is_le(Nat.mul(a, c), _) == True{} : Bool}
  Ar.ar_le_mul_mono(a, b, c, h)

def rb_max_ge_l(+a: Nat, +b: Nat) -> {Nat.is_le(a, Nat.max(a, b)) == True{} : Bool}:
  match a b:
    case 0n _:
      Ar.ar_h_le_zero(b)
    case 1n+ap 0n:
      Ar.ar_le_refl(ap)
    case 1n+ap 1n+bp:
      rb_max_ge_l(ap, bp)

def rb_max_ge_r(+a: Nat, +b: Nat) -> {Nat.is_le(b, Nat.max(a, b)) == True{} : Bool}:
  match a b:
    case 0n _:
      Ar.ar_le_refl(b)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      rb_max_ge_r(ap, bp)

def rb_max_right(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.max(a, b) == b : Nat}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      Empty.absurd({Nat.max(1n+ap, 0n) == 0n : Nat}, Ar.ar_false_true(h))
    case 1n+ap 1n+bp:
      Equal.cong(Nat, Nat, t => 1n+t, Nat.max(ap, bp), bp, rb_max_right(ap, bp, h))

def rb_max_left(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == False{} : Bool}) -> {Nat.max(a, b) == a : Nat}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.max(0n, 0n) == 0n : Nat}, Ar.ar_true_false(h))
    case 0n 1n+bp:
      Empty.absurd({Nat.max(0n, 1n+bp) == 0n : Nat}, Ar.ar_true_false(h))
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      Equal.cong(Nat, Nat, t => 1n+t, Nat.max(ap, bp), ap, rb_max_left(ap, bp, h))

def rb_or_true_r(+z: Bool) -> {Bool.or(z, True{}) == True{} : Bool}:
  match z:
    case False{}:
      {==}
    case True{}:
      {==}

def rb_not_inj(+x: Bool, +y: Bool, h: {Bool.not(x) == Bool.not(y) : Bool}) -> {x == y : Bool}:
  match x y:
    case False{} False{}:
      {==}
    case False{} True{}:
      Empty.absurd({False{} == True{} : Bool}, Ar.ar_true_false(h))
    case True{} False{}:
      Empty.absurd({True{} == False{} : Bool}, Ar.ar_false_true(h))
    case True{} True{}:
      {==}

# x < x + s for s >= 1
def rb_lt_add_pos(+x: Nat, +s: Nat, +pos: {Nat.is_lt(0n, s) == True{} : Bool})
  -> {Nat.is_lt(x, Nat.add(x, s)) == True{} : Bool}:
  match x:
    case 0n:
      pos
    case 1n+xp:
      rb_lt_add_pos(xp, s, pos)

# min(min(a, T), P) == min(a, P) when P <= T
def rb_min_min(+a: Nat, +T: Nat, +P: Nat, h: {Nat.is_le(P, T) == True{} : Bool})
  -> {Nat.min(Nat.min(a, T), P) == Nat.min(a, P) : Nat}:
  match a T P:
    case 0n _ _:
      {==}
    case 1n+ap 0n 0n:
      {==}
    case 1n+ap 0n 1n+pp:
      Empty.absurd({Nat.min(Nat.min(1n+ap, 0n), 1n+pp) == Nat.min(1n+ap, 1n+pp) : Nat}, Ar.ar_false_true(h))
    case 1n+ap 1n+tp 0n:
      {==}
    case 1n+ap 1n+tp 1n+pp:
      Equal.cong(Nat, Nat, t => 1n+t, Nat.min(Nat.min(ap, tp), pp), Nat.min(ap, pp), rb_min_min(ap, tp, pp, h))

def rb_pos_l(+a: Nat, +b: Nat, h: {Nat.is_lt(0n, Nat.mul(a, b)) == True{} : Bool}) -> {Nat.is_lt(0n, a) == True{} : Bool}:
  match a:
    case 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Ar.ar_false_true(h))
    case 1n+ap:
      {==}

def rb_pos_r(+a: Nat, +b: Nat, h: {Nat.is_lt(0n, Nat.mul(a, b)) == True{} : Bool}) -> {Nat.is_lt(0n, b) == True{} : Bool}:
  match b:
    case 0n:
      +z = Equal.cong(Nat, Bool, t => Nat.is_lt(0n, t), 0n, Nat.mul(a, 0n), Equal.sym(Nat, Nat.mul(a, 0n), 0n, Ar.ar_mul_zero_r(a)))
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool},
        Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(0n, Nat.mul(a, 0n)), True{}, z, h)))
    case 1n+bp:
      {==}

# r < q <= T gives T >= 1
def rb_tpos(+q: Nat, +r: Nat, +T: Nat, fits: {Nat.is_le(q, T) == True{} : Bool}, row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Nat.is_lt(0n, T) == True{} : Bool}:
  Ar.ar_lt_le_trans(0n, q, T, Ar.ar_le_lt_trans(0n, r, q, Ar.ar_h_le_zero(r), row), fits)

# r < q gives c + r + 1 <= c + q
def rb_row_le(+c: Nat, +r: Nat, +q: Nat, row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Nat.is_le(1n+Nat.add(c, r), Nat.add(c, q)) == True{} : Bool}:
  %Ar.ar_add_succ_r(c, r) : {Nat.is_le(_, Nat.add(c, q)) == True{} : Bool}
  Ar.ard_tr(Nat.is_le(Nat.add(c, 1n+r), Nat.add(c, q)), Nat.is_le(1n+r, q), Ar.ar_le_add_cancel(c, 1n+r, q),
    Ar.ard_rt(Nat.is_lt(r, q), Nat.is_le(1n+r, q), Ar.ar_lt_succ_le(r, q), row))

# ---------------------------------------------------------------- ceil division

def rb_cdiv_succ(+a: Nat, +tp: Nat) -> {Split.cdiv(a, 1n+tp) == Nat.div(Nat.add(a, tp), 1n+tp) : Nat}:
  Equal.cong(Nat, Nat, x => Nat.div(x, 1n+tp), Nat.sub(Nat.add(a, 1n+tp), 1n), Nat.add(a, tp), AS.as_pred(a, tp))

def rb_cdiv_mono(+a: Nat, +b: Nat, +T: Nat, +pos: {Nat.is_lt(0n, T) == True{} : Bool}, ab: {Nat.is_le(a, b) == True{} : Bool})
  -> {Nat.is_le(Split.cdiv(a, T), Split.cdiv(b, T)) == True{} : Bool}:
  match T:
    case 0n:
      Empty.absurd({Nat.is_le(Split.cdiv(a, 0n), Split.cdiv(b, 0n)) == True{} : Bool}, Ar.ar_false_true(pos))
    case 1n+tp:
      %Equal.sym(Nat, Split.cdiv(a, 1n+tp), Nat.div(Nat.add(a, tp), 1n+tp), rb_cdiv_succ(a, tp)) :
        {Nat.is_le(_, Split.cdiv(b, 1n+tp)) == True{} : Bool}
      %Equal.sym(Nat, Split.cdiv(b, 1n+tp), Nat.div(Nat.add(b, tp), 1n+tp), rb_cdiv_succ(b, tp)) :
        {Nat.is_le(Nat.div(Nat.add(a, tp), 1n+tp), _) == True{} : Bool}
      Ar.ar_div_mono(Nat.add(a, tp), Nat.add(b, tp), 1n+tp, pos, rb_le_add_r(a, b, tp, ab))

# q <= T: cdiv(c + q, T) <= cdiv(c + 1, T) + 1 (a round of q <= T rows spans at most one step)
def rb_cdiv_step(+c: Nat, +q: Nat, +T: Nat, +pos: {Nat.is_lt(0n, T) == True{} : Bool}, fits: {Nat.is_le(q, T) == True{} : Bool})
  -> {Nat.is_le(Split.cdiv(Nat.add(c, q), T), 1n+Split.cdiv(1n+c, T)) == True{} : Bool}:
  match T:
    case 0n:
      Empty.absurd({Nat.is_le(Split.cdiv(Nat.add(c, q), 0n), 1n+Split.cdiv(1n+c, 0n)) == True{} : Bool}, Ar.ar_false_true(pos))
    case 1n+tp:
      +T1 = {1n+tp : Nat}
      %Equal.sym(Nat, Split.cdiv(Nat.add(c, q), T1), Nat.div(Nat.add(Nat.add(c, q), tp), T1), rb_cdiv_succ(Nat.add(c, q), tp)) :
        {Nat.is_le(_, 1n+Split.cdiv(1n+c, T1)) == True{} : Bool}
      %Equal.sym(Nat, Split.cdiv(1n+c, T1), Nat.div(Nat.add(1n+c, tp), T1), rb_cdiv_succ(1n+c, tp)) :
        {Nat.is_le(Nat.div(Nat.add(Nat.add(c, q), tp), T1), 1n+_) == True{} : Bool}
      %Ar.ar_div_add_self(Nat.add(1n+c, tp), T1, pos) :
        {Nat.is_le(Nat.div(Nat.add(Nat.add(c, q), tp), T1), _) == True{} : Bool}
      # (c + q) + tp <= (c + T) + tp == (c + tp) + T <= 1 + ((c + tp) + T)
      +h1 = rb_le_add_r(Nat.add(c, q), Nat.add(c, T1), tp,
        Ar.ard_tr(Nat.is_le(Nat.add(c, q), Nat.add(c, T1)), Nat.is_le(q, T1), Ar.ar_le_add_cancel(c, q, T1), fits))
      +h2 = Ar.ard_rt(Nat.is_le(Nat.add(Nat.add(c, q), tp), Nat.add(Nat.add(c, T1), tp)),
        Nat.is_le(Nat.add(Nat.add(c, q), tp), Nat.add(Nat.add(c, tp), T1)),
        Equal.cong(Nat, Bool, x => Nat.is_le(Nat.add(Nat.add(c, q), tp), x), Nat.add(Nat.add(c, T1), tp),
          Nat.add(Nat.add(c, tp), T1), rb_swap_tail(c, T1, tp)), h1)
      +h3 = Ar.ar_le_trans(Nat.add(Nat.add(c, q), tp), Nat.add(Nat.add(c, tp), T1), 1n+Nat.add(Nat.add(c, tp), T1),
        h2, Ar.ar_h_le_succ(Nat.add(Nat.add(c, tp), T1)))
      Ar.ar_div_mono(Nat.add(Nat.add(c, q), tp), Nat.add(Nat.add(1n+c, tp), T1), T1, pos, h3)

# ceil(n / S) <= m iff n <= m * S (S >= 1)
def rb_cdiv_le(+n: Nat, +S: Nat, +m: Nat, +spos: {Nat.is_lt(0n, S) == True{} : Bool})
  -> {Nat.is_le(Split.cdiv(n, S), m) == Nat.is_le(n, Nat.mul(m, S)) : Bool}:
  rb_not_inj(Nat.is_le(Split.cdiv(n, S), m), Nat.is_le(n, Nat.mul(m, S)),
    Equal.trans(Bool, Bool.not(Nat.is_le(Split.cdiv(n, S), m)), Nat.is_lt(m, Split.cdiv(n, S)), Bool.not(Nat.is_le(n, Nat.mul(m, S))),
      Equal.sym(Bool, Nat.is_lt(m, Split.cdiv(n, S)), Bool.not(Nat.is_le(Split.cdiv(n, S), m)), Ar.ar_lt_not_le(m, Split.cdiv(n, S))),
      AS.as_live(m, n, S, spos)))

# i < ceil(n / (S * B)) iff i < ceil(ceil(n / S) / B)
def rb_nest_lt(+i: Nat, +n: Nat, +S: Nat, +B: Nat, +spos: {Nat.is_lt(0n, S) == True{} : Bool}, +bpos: {Nat.is_lt(0n, B) == True{} : Bool})
  -> {Nat.is_lt(i, Split.cdiv(n, Nat.mul(S, B))) == Nat.is_lt(i, Spec.ceil_div(Spec.ceil_div(n, S), B)) : Bool}:
  +SB = Nat.mul(S, B)
  +sbpos = AB.ab_mul_pos(S, B, spos, bpos)
  # i < ceil(n / SB) == not(n <= i * SB)
  %Equal.sym(Bool, Nat.is_lt(i, Split.cdiv(n, SB)), Bool.not(Nat.is_le(n, Nat.mul(i, SB))), AS.as_live(i, n, SB, sbpos)) :
    {_ == Nat.is_lt(i, Spec.ceil_div(Spec.ceil_div(n, S), B)) : Bool}
  # i < ceil(ceil(n / S) / B) == not(ceil(n / S) <= i * B) == not(n <= (i * B) * S)
  %Equal.sym(Bool, Nat.is_lt(i, Split.cdiv(Split.cdiv(n, S), B)), Bool.not(Nat.is_le(Split.cdiv(n, S), Nat.mul(i, B))),
    AS.as_live(i, Split.cdiv(n, S), B, bpos)) : {Bool.not(Nat.is_le(n, Nat.mul(i, SB))) == _ : Bool}
  %Equal.sym(Bool, Nat.is_le(Split.cdiv(n, S), Nat.mul(i, B)), Nat.is_le(n, Nat.mul(Nat.mul(i, B), S)),
    rb_cdiv_le(n, S, Nat.mul(i, B), spos)) : {Bool.not(Nat.is_le(n, Nat.mul(i, SB))) == Bool.not(_) : Bool}
  %Equal.sym(Nat, Nat.mul(Nat.mul(i, B), S), Nat.mul(i, Nat.mul(B, S)), Ar.ar_mul_assoc(i, B, S)) :
    {Bool.not(Nat.is_le(n, Nat.mul(i, SB))) == Bool.not(Nat.is_le(n, _)) : Bool}
  %Equal.sym(Nat, Nat.mul(B, S), Nat.mul(S, B), Ar.ar_mul_comm(B, S)) :
    {Bool.not(Nat.is_le(n, Nat.mul(i, SB))) == Bool.not(Nat.is_le(n, Nat.mul(i, _))) : Bool}
  {==}

# ceil(n / (S * B)) == ceil(ceil(n / S) / B)
def rb_nest(+n: Nat, +S: Nat, +B: Nat, +spos: {Nat.is_lt(0n, S) == True{} : Bool}, +bpos: {Nat.is_lt(0n, B) == True{} : Bool})
  -> {Split.cdiv(n, Nat.mul(S, B)) == Spec.ceil_div(Spec.ceil_div(n, S), B) : Nat}:
  +X = Spec.ceil_div(Spec.ceil_div(n, S), B)
  +Y = Split.cdiv(n, Nat.mul(S, B))
  # X < Y == X < X == False, so Y <= X
  +h1 = Equal.trans(Bool, Nat.is_lt(X, Y), Nat.is_lt(X, X), False{}, rb_nest_lt(X, n, S, B, spos, bpos), Ar.ar_lt_irrefl(X))
  +yx = AS.as_not_false(Nat.is_le(Y, X), Equal.trans(Bool, Bool.not(Nat.is_le(Y, X)), Nat.is_lt(X, Y), False{},
    Equal.sym(Bool, Nat.is_lt(X, Y), Bool.not(Nat.is_le(Y, X)), Ar.ar_lt_not_le(X, Y)), h1))
  # Y < X == Y < Y == False, so X <= Y
  +h2 = Equal.trans(Bool, Nat.is_lt(Y, X), Nat.is_lt(Y, Y), False{},
    Equal.sym(Bool, Nat.is_lt(Y, Y), Nat.is_lt(Y, X), rb_nest_lt(Y, n, S, B, spos, bpos)), Ar.ar_lt_irrefl(Y))
  +xy = AS.as_not_false(Nat.is_le(X, Y), Equal.trans(Bool, Bool.not(Nat.is_le(X, Y)), Nat.is_lt(Y, X), False{},
    Equal.sym(Bool, Nat.is_lt(Y, X), Bool.not(Nat.is_le(X, Y)), Ar.ar_lt_not_le(Y, X)), h2))
  Ar.ar_le_antisym(Y, X, yx, xy)

# The kernel's s(n) equals the reference length (S, B >= 1).
def rb_spec_eq(+n: Nat, +S: Nat, +B: Nat, +M: Nat, +spos: {Nat.is_lt(0n, S) == True{} : Bool}, +bpos: {Nat.is_lt(0n, B) == True{} : Bool})
  -> {Impl.row_split_len(n, S, B, M) == Spec.row_len(n, S, B, M) : Nat}:
  Equal.cong(Nat, Nat, t => Nat.max(M, Nat.mul(B, t)), Split.cdiv(n, Nat.mul(S, B)),
    Spec.ceil_div(Spec.ceil_div(n, S), B), rb_nest(n, S, B, spos, bpos))

# a = 0, x = 1 + xp, b = 1 + bp: xp <= bp <= 0
def rb_between0(+xp: Nat, +bp: Nat, xb: {Nat.is_le(xp, bp) == True{} : Bool}, b1: {Nat.is_le(bp, 0n) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(1n+xp, 0n), Nat.is_eq(1n+xp, 1n+bp)) == True{} : Bool}:
  match xp bp:
    case 0n 0n:
      {==}
    case 0n 1n+bq:
      Empty.absurd({Bool.or(Nat.is_eq(1n, 0n), Nat.is_eq(1n, 2n+bq)) == True{} : Bool}, Ar.ar_false_true(b1))
    case 1n+xq 0n:
      Empty.absurd({Bool.or(Nat.is_eq(2n+xq, 0n), Nat.is_eq(2n+xq, 1n)) == True{} : Bool}, Ar.ar_false_true(xb))
    case 1n+xq 1n+bq:
      Empty.absurd({Bool.or(Nat.is_eq(2n+xq, 0n), Nat.is_eq(2n+xq, 2n+bq)) == True{} : Bool}, Ar.ar_false_true(b1))

# ---------------------------------------------------------------- B3: two passes

# a <= x <= b <= a + 1 gives x == a or x == b
def rb_between(+a: Nat, +x: Nat, +b: Nat, ax: {Nat.is_le(a, x) == True{} : Bool},
    xb: {Nat.is_le(x, b) == True{} : Bool}, b1: {Nat.is_le(b, 1n+a) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(x, a), Nat.is_eq(x, b)) == True{} : Bool}:
  match a x b:
    case 0n 0n _:
      {==}
    case 0n 1n+xp 0n:
      Empty.absurd({Bool.or(Nat.is_eq(1n+xp, 0n), Nat.is_eq(1n+xp, 0n)) == True{} : Bool}, Ar.ar_false_true(xb))
    case 0n 1n+xp 1n+bp:
      rb_between0(xp, bp, xb, b1)
    case 1n+ap 0n _:
      Empty.absurd({Bool.or(Nat.is_eq(0n, 1n+ap), Nat.is_eq(0n, b)) == True{} : Bool}, Ar.ar_false_true(ax))
    case 1n+ap 1n+xp 0n:
      Empty.absurd({Bool.or(Nat.is_eq(1n+xp, 1n+ap), Nat.is_eq(1n+xp, 0n)) == True{} : Bool}, Ar.ar_false_true(xb))
    case 1n+ap 1n+xp 1n+bp:
      rb_between(ap, xp, bp, ax, xb, b1)

# Lift x in {a, b} through t -> max(M, B * t).
def rb_lift(+x: Nat, +a: Nat, +b: Nat, +B: Nat, +M: Nat, u: Bool, eu: {Nat.is_eq(x, a) == u : Bool},
    h: {Bool.or(u, Nat.is_eq(x, b)) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(Nat.max(M, Nat.mul(B, x)), Nat.max(M, Nat.mul(B, a))),
       Nat.is_eq(Nat.max(M, Nat.mul(B, x)), Nat.max(M, Nat.mul(B, b)))) == True{} : Bool}:
  match u:
    case True{}:
      %Equal.sym(Nat, x, a, Ar.ar_eq_true(x, a, eu)) :
        {Bool.or(Nat.is_eq(Nat.max(M, Nat.mul(B, _)), Nat.max(M, Nat.mul(B, a))),
          Nat.is_eq(Nat.max(M, Nat.mul(B, _)), Nat.max(M, Nat.mul(B, b)))) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_eq(Nat.max(M, Nat.mul(B, a)), Nat.max(M, Nat.mul(B, a))), True{}, Ar.ar_eq_refl(Nat.max(M, Nat.mul(B, a)))) :
        {Bool.or(_, Nat.is_eq(Nat.max(M, Nat.mul(B, a)), Nat.max(M, Nat.mul(B, b)))) == True{} : Bool}
      {==}
    case False{}:
      %Equal.sym(Nat, x, b, Ar.ar_eq_true(x, b, h)) :
        {Bool.or(Nat.is_eq(Nat.max(M, Nat.mul(B, _)), Nat.max(M, Nat.mul(B, a))),
          Nat.is_eq(Nat.max(M, Nat.mul(B, _)), Nat.max(M, Nat.mul(B, b)))) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_eq(Nat.max(M, Nat.mul(B, b)), Nat.max(M, Nat.mul(B, b))), True{}, Ar.ar_eq_refl(Nat.max(M, Nat.mul(B, b)))) :
        {Bool.or(Nat.is_eq(Nat.max(M, Nat.mul(B, b)), Nat.max(M, Nat.mul(B, a))), _) == True{} : Bool}
      rb_or_true_r(Nat.is_eq(Nat.max(M, Nat.mul(B, b)), Nat.max(M, Nat.mul(B, a))))

def rb_b3a(+c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(Impl.row_split_len(1n+Nat.add(c, r), S, B, M), Impl.pass_a_len(c, S, B, M)),
       Nat.is_eq(Impl.row_split_len(1n+Nat.add(c, r), S, B, M), Impl.pass_b_len(c, q, S, B, M))) == True{} : Bool}:
  +T = Nat.mul(S, B)
  +tpos = rb_tpos(q, r, T, fits, row)
  +tA = Split.cdiv(1n+c, T)
  +tn = Split.cdiv(1n+Nat.add(c, r), T)
  +tB = Split.cdiv(Nat.add(c, q), T)
  +ax = rb_cdiv_mono(1n+c, 1n+Nat.add(c, r), T, tpos, Ar.ar_le_add_r(c, r))
  +xb = rb_cdiv_mono(1n+Nat.add(c, r), Nat.add(c, q), T, tpos, rb_row_le(c, r, q, row))
  +b1 = rb_cdiv_step(c, q, T, tpos, fits)
  rb_lift(tn, tA, tB, B, M, Nat.is_eq(tn, tA), {==}, rb_between(tA, tn, tB, ax, xb, b1))

def rb_g_mono(+t1: Nat, +t2: Nat, +B: Nat, +M: Nat, h: {Nat.is_le(t1, t2) == True{} : Bool})
  -> {Nat.is_le(Nat.max(M, Nat.mul(B, t1)), Nat.max(M, Nat.mul(B, t2))) == True{} : Bool}:
  AB.ab_max_le(M, Nat.mul(B, t1), Nat.max(M, Nat.mul(B, t2)), rb_max_ge_l(M, Nat.mul(B, t2)),
    Ar.ar_le_trans(Nat.mul(B, t1), Nat.mul(B, t2), Nat.max(M, Nat.mul(B, t2)), rb_mul_mono_r(t1, t2, B, h),
      rb_max_ge_r(M, Nat.mul(B, t2))))

def rb_b3b(+c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Nat.is_le(Impl.pass_a_len(c, S, B, M), Impl.pass_b_len(c, q, S, B, M)) == True{} : Bool}:
  +T = Nat.mul(S, B)
  +tpos = rb_tpos(q, r, T, fits, row)
  rb_g_mono(Split.cdiv(1n+c, T), Split.cdiv(Nat.add(c, q), T), B, M,
    rb_cdiv_mono(1n+c, Nat.add(c, q), T, tpos,
      Ar.ar_le_trans(1n+c, 1n+Nat.add(c, r), Nat.add(c, q), Ar.ar_le_add_r(c, r), rb_row_le(c, r, q, row))))

def rb_pass(+sn: Nat, +sA: Nat, +sB: Nat, u: Bool, +eu: {Nat.is_eq(sn, sA) == u : Bool},
    h: {Bool.or(u, Nat.is_eq(sn, sB)) == True{} : Bool})
  -> {Bool.and(Bool.not(Nat.is_eq(sn, sA)), Bool.not(Bool.not(Nat.is_eq(sB, sA)))) == False{} : Bool}:
  match u:
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(sn, sA), True{}, eu) :
        {Bool.and(Bool.not(_), Bool.not(Bool.not(Nat.is_eq(sB, sA)))) == False{} : Bool}
      {==}
    case False{}:
      %Ar.ar_eq_true(sn, sB, h) : {Bool.and(Bool.not(Nat.is_eq(sn, sA)), Bool.not(Bool.not(Nat.is_eq(_, sA)))) == False{} : Bool}
      %Equal.sym(Bool, Nat.is_eq(sn, sA), False{}, eu) :
        {Bool.and(Bool.not(_), Bool.not(Bool.not(_))) == False{} : Bool}
      {==}

def rb_b3c(+c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Bool.and(Impl.in_pass_b(c, r, S, B, M), Bool.not(Impl.runs_pass_b(c, q, S, B, M))) == False{} : Bool}:
  +sn = Impl.row_split_len(1n+Nat.add(c, r), S, B, M)
  +sA = Impl.pass_a_len(c, S, B, M)
  rb_pass(sn, sA, Impl.pass_b_len(c, q, S, B, M), Nat.is_eq(sn, sA), {==}, rb_b3a(c, q, r, S, B, M, fits, row))

def Laws.two_passes_suffice(c, q_len, r, splits, block_n, min_split, fits_, row_):
  +fits = fits_
  +row = row_
  (rb_b3a(c, q_len, r, splits, block_n, min_split, fits, row),
    rb_b3b(c, q_len, r, splits, block_n, min_split, fits, row),
    rb_b3c(c, q_len, r, splits, block_n, min_split, fits, row))

# ---------------------------------------------------------------- B1 / B4: absolute partition

def rb_pick(+sn: Nat, +sA: Nat, +sB: Nat, u: Bool, +eu: {Nat.is_eq(sn, sA) == u : Bool},
    h: {Bool.or(u, Nat.is_eq(sn, sB)) == True{} : Bool})
  -> {Impl.pick(Bool.not(Nat.is_eq(sn, sA)), sB, sA) == sn : Nat}:
  match u:
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(sn, sA), True{}, eu) : {Impl.pick(Bool.not(_), sB, sA) == sn : Nat}
      Equal.sym(Nat, sn, sA, Ar.ar_eq_true(sn, sA, eu))
    case False{}:
      %Equal.sym(Bool, Nat.is_eq(sn, sA), False{}, eu) : {Impl.pick(Bool.not(_), sB, sA) == sn : Nat}
      Equal.sym(Nat, sn, sB, Ar.ar_eq_true(sn, sB, h))

def rb_b1_len(+c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.kernel_row_split_len(c, q, r, S, B, M) == Spec.row_len(1n+Nat.add(c, r), S, B, M) : Nat}:
  +sn = Impl.row_split_len(1n+Nat.add(c, r), S, B, M)
  +sA = Impl.pass_a_len(c, S, B, M)
  +tpos = rb_tpos(q, r, Nat.mul(S, B), fits, row)
  Equal.trans(Nat, Impl.kernel_row_split_len(c, q, r, S, B, M), sn, Spec.row_len(1n+Nat.add(c, r), S, B, M),
    rb_pick(sn, sA, Impl.pass_b_len(c, q, S, B, M), Nat.is_eq(sn, sA), {==}, rb_b3a(c, q, r, S, B, M, fits, row)),
    rb_spec_eq(1n+Nat.add(c, r), S, B, M, rb_pos_l(S, B, tpos), rb_pos_r(S, B, tpos)))

def rb_b1_lo(+c: Nat, +q: Nat, +r: Nat, +k: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.chunk_lo(c, q, r, k, S, B, M) == Spec.chunk_lo(Nat.add(c, r), k, S, B, M) : Nat}:
  Equal.cong(Nat, Nat, t => Nat.mul(k, t), Impl.kernel_row_split_len(c, q, r, S, B, M),
    Spec.row_len(1n+Nat.add(c, r), S, B, M), rb_b1_len(c, q, r, S, B, M, fits, row))

def rb_b1_hi(+c: Nat, +q: Nat, +r: Nat, +k: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.chunk_hi(c, q, r, k, S, B, M) == Spec.chunk_hi(Nat.add(c, r), k, S, B, M) : Nat}:
  +KSL = Impl.kernel_row_split_len(c, q, r, S, B, M)
  +SL = Spec.row_len(1n+Nat.add(c, r), S, B, M)
  %Equal.sym(Nat, KSL, SL, rb_b1_len(c, q, r, S, B, M, fits, row)) :
    {Nat.min(Nat.min(Nat.add(Nat.mul(k, _), _), Nat.add(c, q)), 1n+Nat.add(c, r)) == Spec.chunk_hi(Nat.add(c, r), k, S, B, M) : Nat}
  rb_min_min(Nat.add(Nat.mul(k, SL), SL), Nat.add(c, q), 1n+Nat.add(c, r), rb_row_le(c, r, q, row))

def Laws.row_partition_absolute(c, q_len, r, k, splits, block_n, min_split, fits_, row_):
  +fits = fits_
  +row = row_
  (rb_b1_len(c, q_len, r, splits, block_n, min_split, fits, row),
    rb_b1_lo(c, q_len, r, k, splits, block_n, min_split, fits, row),
    rb_b1_hi(c, q_len, r, k, splits, block_n, min_split, fits, row))

def Laws.round_independence(c1, q1, r1, c2, q2, r2, k, splits, block_n, min_split, fits1_, row1_, fits2_, row2_, same_):
  +fits1 = fits1_
  +row1 = row1_
  +fits2 = fits2_
  +row2 = row2_
  +same = same_
  +p1 = Nat.add(c1, r1)
  +p2 = Nat.add(c2, r2)
  (Equal.trans(Nat, Impl.kernel_row_split_len(c1, q1, r1, splits, block_n, min_split),
      Spec.row_len(1n+p1, splits, block_n, min_split), Impl.kernel_row_split_len(c2, q2, r2, splits, block_n, min_split),
      rb_b1_len(c1, q1, r1, splits, block_n, min_split, fits1, row1),
      Equal.trans(Nat, Spec.row_len(1n+p1, splits, block_n, min_split), Spec.row_len(1n+p2, splits, block_n, min_split),
        Impl.kernel_row_split_len(c2, q2, r2, splits, block_n, min_split),
        Equal.cong(Nat, Nat, t => Spec.row_len(1n+t, splits, block_n, min_split), p1, p2, same),
        Equal.sym(Nat, Impl.kernel_row_split_len(c2, q2, r2, splits, block_n, min_split), Spec.row_len(1n+p2, splits, block_n, min_split),
          rb_b1_len(c2, q2, r2, splits, block_n, min_split, fits2, row2)))),
    Equal.trans(Nat, Impl.chunk_lo(c1, q1, r1, k, splits, block_n, min_split),
      Spec.chunk_lo(p1, k, splits, block_n, min_split), Impl.chunk_lo(c2, q2, r2, k, splits, block_n, min_split),
      rb_b1_lo(c1, q1, r1, k, splits, block_n, min_split, fits1, row1),
      Equal.trans(Nat, Spec.chunk_lo(p1, k, splits, block_n, min_split), Spec.chunk_lo(p2, k, splits, block_n, min_split),
        Impl.chunk_lo(c2, q2, r2, k, splits, block_n, min_split),
        Equal.cong(Nat, Nat, t => Spec.chunk_lo(t, k, splits, block_n, min_split), p1, p2, same),
        Equal.sym(Nat, Impl.chunk_lo(c2, q2, r2, k, splits, block_n, min_split), Spec.chunk_lo(p2, k, splits, block_n, min_split),
          rb_b1_lo(c2, q2, r2, k, splits, block_n, min_split, fits2, row2)))),
    Equal.trans(Nat, Impl.chunk_hi(c1, q1, r1, k, splits, block_n, min_split),
      Spec.chunk_hi(p1, k, splits, block_n, min_split), Impl.chunk_hi(c2, q2, r2, k, splits, block_n, min_split),
      rb_b1_hi(c1, q1, r1, k, splits, block_n, min_split, fits1, row1),
      Equal.trans(Nat, Spec.chunk_hi(p1, k, splits, block_n, min_split), Spec.chunk_hi(p2, k, splits, block_n, min_split),
        Impl.chunk_hi(c2, q2, r2, k, splits, block_n, min_split),
        Equal.cong(Nat, Nat, t => Spec.chunk_hi(t, k, splits, block_n, min_split), p1, p2, same),
        Equal.sym(Nat, Impl.chunk_hi(c2, q2, r2, k, splits, block_n, min_split), Spec.chunk_hi(p2, k, splits, block_n, min_split),
          rb_b1_hi(c2, q2, r2, k, splits, block_n, min_split, fits2, row2)))))

def Laws.old_round_relative_differs():
  ({==}, {==}, {==}, {==}, {==}, {==})

# ---------------------------------------------------------------- B2: coverage and alignment

# n <= S * s(n)
def rb_cover_le(+n: Nat, +S: Nat, +B: Nat, +M: Nat, +tpos: {Nat.is_lt(0n, Nat.mul(S, B)) == True{} : Bool})
  -> {Nat.is_le(n, Nat.mul(S, Impl.row_split_len(n, S, B, M))) == True{} : Bool}:
  +T = Nat.mul(S, B)
  +t = Split.cdiv(n, T)
  +s = Impl.row_split_len(n, S, B, M)
  +e1 = Equal.trans(Nat, Nat.mul(t, T), Nat.mul(T, t), Nat.mul(S, Nat.mul(B, t)), Ar.ar_mul_comm(t, T), Ar.ar_mul_assoc(S, B, t))
  +h2 = Ar.ard_rt(Nat.is_le(n, Nat.mul(t, T)), Nat.is_le(n, Nat.mul(S, Nat.mul(B, t))),
    Equal.cong(Nat, Bool, x => Nat.is_le(n, x), Nat.mul(t, T), Nat.mul(S, Nat.mul(B, t)), e1), AS.as_up(n, T, tpos))
  Ar.ar_le_trans(n, Nat.mul(S, Nat.mul(B, t)), Nat.mul(S, s), h2, rb_mul_mono_r(Nat.mul(B, t), s, S, rb_max_ge_r(M, Nat.mul(B, t))))

# s(n) >= 1 for n >= 1
def rb_s_pos(+n: Nat, +S: Nat, +B: Nat, +M: Nat, +npos: {Nat.is_lt(0n, n) == True{} : Bool},
    +tpos: {Nat.is_lt(0n, Nat.mul(S, B)) == True{} : Bool}, +bpos: {Nat.is_lt(0n, B) == True{} : Bool})
  -> {Nat.is_lt(0n, Impl.row_split_len(n, S, B, M)) == True{} : Bool}:
  +T = Nat.mul(S, B)
  +t = Split.cdiv(n, T)
  +tp1 = AS.as_cpos(n, t, T, npos, AS.as_up(n, T, tpos))
  Ar.ar_lt_le_trans(0n, Nat.mul(B, t), Nat.max(M, Nat.mul(B, t)), AB.ab_mul_pos(B, t, bpos, tp1), rb_max_ge_r(M, Nat.mul(B, t)))

# ceil(n / s(n)) <= S for n >= 1
def rb_cover(+n: Nat, +S: Nat, +B: Nat, +M: Nat, +npos: {Nat.is_lt(0n, n) == True{} : Bool},
    +tpos: {Nat.is_lt(0n, Nat.mul(S, B)) == True{} : Bool}, +bpos: {Nat.is_lt(0n, B) == True{} : Bool})
  -> {Nat.is_le(Split.cdiv(n, Impl.row_split_len(n, S, B, M)), S) == True{} : Bool}:
  +s = Impl.row_split_len(n, S, B, M)
  +cd = Split.cdiv(n, s)
  +f = Equal.trans(Bool, Nat.is_lt(S, cd), Bool.not(Nat.is_le(n, Nat.mul(S, s))), False{},
    AS.as_live(S, n, s, rb_s_pos(n, S, B, M, npos, tpos, bpos)),
    Equal.cong(Bool, Bool, x => Bool.not(x), Nat.is_le(n, Nat.mul(S, s)), True{}, rb_cover_le(n, S, B, M, tpos)))
  AS.as_not_false(Nat.is_le(cd, S), Equal.trans(Bool, Bool.not(Nat.is_le(cd, S)), Nat.is_lt(S, cd), False{},
    Equal.sym(Bool, Nat.is_lt(S, cd), Bool.not(Nat.is_le(cd, S)), Ar.ar_lt_not_le(S, cd)), f))

def rb_mod_mul(+t: Nat, +B: Nat, +pos: {Nat.is_lt(0n, B) == True{} : Bool}) -> {Nat.mod(Nat.mul(t, B), B) == 0n : Nat}:
  %Ar.ar_add_zero_r(Nat.mul(t, B)) : {Nat.mod(_, B) == 0n : Nat}
  Ar.ard_snd({Nat.div(Nat.add(Nat.mul(t, B), 0n), B) == t : Nat}, {Nat.mod(Nat.add(Nat.mul(t, B), 0n), B) == 0n : Nat},
    Ar.ar_div_unique(t, 0n, B, pos))

def rb_mod_mul_l(+B: Nat, +t: Nat, +pos: {Nat.is_lt(0n, B) == True{} : Bool}) -> {Nat.mod(Nat.mul(B, t), B) == 0n : Nat}:
  %Ar.ar_mul_comm(t, B) : {Nat.mod(_, B) == 0n : Nat}
  rb_mod_mul(t, B, pos)

def rb_maxmod(+a: Nat, +b: Nat, +B: Nat, u: Bool, eu: {Nat.is_le(a, b) == u : Bool},
    ha: {Nat.mod(a, B) == 0n : Nat}, hb: {Nat.mod(b, B) == 0n : Nat}) -> {Nat.mod(Nat.max(a, b), B) == 0n : Nat}:
  match u:
    case True{}:
      %Equal.sym(Nat, Nat.max(a, b), b, rb_max_right(a, b, eu)) : {Nat.mod(_, B) == 0n : Nat}
      hb
    case False{}:
      %Equal.sym(Nat, Nat.max(a, b), a, rb_max_left(a, b, eu)) : {Nat.mod(_, B) == 0n : Nat}
      ha

def Laws.coverage(n, splits, block_n, m, npos_, spos_, bpos_):
  +npos = npos_
  +bpos = bpos_
  +tpos = AB.ab_mul_pos(splits, block_n, spos_, bpos)
  +M = Nat.mul(m, block_n)
  +t = Split.cdiv(n, Nat.mul(splits, block_n))
  (rb_cover(n, splits, block_n, M, npos, tpos, bpos),
    rb_maxmod(M, Nat.mul(block_n, t), block_n, Nat.is_le(M, Nat.mul(block_n, t)), {==},
      rb_mod_mul(m, block_n, bpos), rb_mod_mul_l(block_n, t, bpos)))

# ---------------------------------------------------------------- B5: combine

# chunk k of position p is non-empty iff k < chunks(p)
def rb_nonempty(+p: Nat, +k: Nat, +SL: Nat, +spos: {Nat.is_lt(0n, SL) == True{} : Bool})
  -> {Nat.is_lt(Nat.mul(k, SL), Nat.min(Nat.add(Nat.mul(k, SL), SL), 1n+p)) == Nat.is_lt(k, Spec.ceil_div(1n+p, SL)) : Bool}:
  +x = Nat.mul(k, SL)
  %Equal.sym(Bool, Nat.is_lt(x, Nat.min(Nat.add(x, SL), 1n+p)), Bool.and(Nat.is_lt(x, Nat.add(x, SL)), Nat.is_lt(x, 1n+p)),
    Ar.ar_min_lt(x, Nat.add(x, SL), 1n+p)) : {_ == Nat.is_lt(k, Spec.ceil_div(1n+p, SL)) : Bool}
  %Equal.sym(Bool, Nat.is_lt(x, Nat.add(x, SL)), True{}, rb_lt_add_pos(x, SL, spos)) :
    {Bool.and(_, Nat.is_lt(x, 1n+p)) == Nat.is_lt(k, Spec.ceil_div(1n+p, SL)) : Bool}
  Equal.trans(Bool, Nat.is_lt(x, 1n+p), Bool.not(Nat.is_le(1n+p, x)), Nat.is_lt(k, Spec.ceil_div(1n+p, SL)),
    Ar.ar_lt_not_le(x, 1n+p), Equal.sym(Bool, Nat.is_lt(k, Split.cdiv(1n+p, SL)), Bool.not(Nat.is_le(1n+p, x)), AS.as_live(k, 1n+p, SL, spos)))

def rb_sl_pos(+p: Nat, +S: Nat, +B: Nat, +M: Nat, +tpos: {Nat.is_lt(0n, Nat.mul(S, B)) == True{} : Bool})
  -> {Nat.is_lt(0n, Spec.row_len(1n+p, S, B, M)) == True{} : Bool}:
  +sp = rb_pos_l(S, B, tpos)
  +bp = rb_pos_r(S, B, tpos)
  Ar.ard_rt(Nat.is_lt(0n, Impl.row_split_len(1n+p, S, B, M)), Nat.is_lt(0n, Spec.row_len(1n+p, S, B, M)),
    Equal.cong(Nat, Bool, x => Nat.is_lt(0n, x), Impl.row_split_len(1n+p, S, B, M), Spec.row_len(1n+p, S, B, M),
      rb_spec_eq(1n+p, S, B, M, sp, bp)),
    rb_s_pos(1n+p, S, B, M, {==}, tpos, bp))

# The kernel's partial for split k of row r is the reference partial of chunk k when k < chunks(p),
# else the identity.
def rb_kpart(~A: Data, ~P: Nat -> Nat -> A, ~e: A, +c: Nat, +q: Nat, +r: Nat, +k: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.partial(~A, ~P, ~e, Nat.is_lt(Impl.chunk_lo(c, q, r, k, S, B, M), Impl.chunk_hi(c, q, r, k, S, B, M)),
        Impl.chunk_lo(c, q, r, k, S, B, M), Impl.chunk_hi(c, q, r, k, S, B, M)) ==
      Impl.partial(~A, ~P, ~e, Nat.is_lt(k, Spec.chunks(Nat.add(c, r), S, B, M)),
        Spec.chunk_lo(Nat.add(c, r), k, S, B, M), Spec.chunk_hi(Nat.add(c, r), k, S, B, M)) : A}:
  +p = Nat.add(c, r)
  +ILO = Impl.chunk_lo(c, q, r, k, S, B, M)
  +IHI = Impl.chunk_hi(c, q, r, k, S, B, M)
  +SLO = Spec.chunk_lo(p, k, S, B, M)
  +SHI = Spec.chunk_hi(p, k, S, B, M)
  +RHS = Impl.partial(~A, ~P, ~e, Nat.is_lt(k, Spec.chunks(p, S, B, M)), SLO, SHI)
  %Equal.sym(Nat, ILO, SLO, rb_b1_lo(c, q, r, k, S, B, M, fits, row)) :
    {Impl.partial(~A, ~P, ~e, Nat.is_lt(_, IHI), _, IHI) == RHS : A}
  %Equal.sym(Nat, IHI, SHI, rb_b1_hi(c, q, r, k, S, B, M, fits, row)) :
    {Impl.partial(~A, ~P, ~e, Nat.is_lt(SLO, _), SLO, _) == RHS : A}
  %Equal.sym(Bool, Nat.is_lt(SLO, SHI), Nat.is_lt(k, Spec.chunks(p, S, B, M)),
    rb_nonempty(p, k, Spec.row_len(1n+p, S, B, M), rb_sl_pos(p, S, B, M, rb_tpos(q, r, Nat.mul(S, B), fits, row)))) :
    {Impl.partial(~A, ~P, ~e, _, SLO, SHI) == RHS : A}
  {==}

def rb_sub_zero_nlt(+a: Nat, +k: Nat, h: {Nat.sub(a, k) == 0n : Nat}) -> {Nat.is_lt(k, a) == False{} : Bool}:
  match a k:
    case 0n 0n:
      {==}
    case 0n 1n+kp:
      {==}
    case 1n+ap 0n:
      Empty.absurd({Nat.is_lt(0n, 1n+ap) == False{} : Bool}, Ar.ar_zero_succ(ap, Equal.sym(Nat, 1n+ap, 0n, h)))
    case 1n+ap 1n+kp:
      rb_sub_zero_nlt(ap, kp, h)

def rb_sub_zero_succ(+a: Nat, +k: Nat, h: {Nat.sub(a, k) == 0n : Nat}) -> {Nat.sub(a, 1n+k) == 0n : Nat}:
  match a k:
    case 0n 0n:
      {==}
    case 0n 1n+kp:
      {==}
    case 1n+ap 0n:
      Empty.absurd({Nat.sub(1n+ap, 1n) == 0n : Nat}, Ar.ar_zero_succ(ap, Equal.sym(Nat, 1n+ap, 0n, h)))
    case 1n+ap 1n+kp:
      rb_sub_zero_succ(ap, kp, h)

def rb_sub_succ_lt(+a: Nat, +k: Nat, -d: Nat, h: {Nat.sub(a, k) == 1n+d : Nat}) -> {Nat.is_lt(k, a) == True{} : Bool}:
  match a k:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Ar.ar_zero_succ(d, h))
    case 0n 1n+kp:
      Empty.absurd({Nat.is_lt(1n+kp, 0n) == True{} : Bool}, Ar.ar_zero_succ(d, h))
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+kp:
      rb_sub_succ_lt(ap, kp, d, h)

def rb_sub_succ_pred(+a: Nat, +k: Nat, +d: Nat, h: {Nat.sub(a, k) == 1n+d : Nat}) -> {Nat.sub(a, 1n+k) == d : Nat}:
  match a k:
    case 0n 0n:
      Empty.absurd({Nat.sub(0n, 1n) == d : Nat}, Ar.ar_zero_succ(d, h))
    case 0n 1n+kp:
      Empty.absurd({Nat.sub(0n, 2n+kp) == d : Nat}, Ar.ar_zero_succ(d, h))
    case 1n+ap 0n:
      Equal.trans(Nat, Nat.sub(ap, 0n), ap, d, Ar.ar_sub_zero_r(ap), Ar.ar_succ_inj(ap, d, h))
    case 1n+ap 1n+kp:
      rb_sub_succ_pred(ap, kp, d, h)

# Past the row's last chunk (chunks - k == 0) every remaining split merges the identity, so the
# combine over `left` splits leaves acc unchanged. h is the one instance merge(acc, e) == acc of
# the right-identity hypothesis that this tail needs (acc does not change along it).
def rb_tail(~A: Data, ~merge: A -> A -> A, ~P: Nat -> Nat -> A, ~e: A,
    +left: Nat, +k: Nat, +c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat, +acc: A,
    +h: {merge(acc, e) == acc : A},
    +hd: {Nat.sub(Spec.chunks(Nat.add(c, r), S, B, M), k) == 0n : Nat},
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.combine.go(~A, ~merge, ~P, ~e, left, k, c, q, r, S, B, M, acc) == acc : A}:
  match left:
    case 0n:
      {==}
    case 1n+m:
      +live = Spec.chunks(Nat.add(c, r), S, B, M)
      +ILO = Impl.chunk_lo(c, q, r, k, S, B, M)
      +IHI = Impl.chunk_hi(c, q, r, k, S, B, M)
      +SLO = Spec.chunk_lo(Nat.add(c, r), k, S, B, M)
      +SHI = Spec.chunk_hi(Nat.add(c, r), k, S, B, M)
      %Equal.sym(A, Impl.partial(~A, ~P, ~e, Nat.is_lt(ILO, IHI), ILO, IHI), Impl.partial(~A, ~P, ~e, Nat.is_lt(k, live), SLO, SHI),
        rb_kpart(~A, ~P, ~e, c, q, r, k, S, B, M, fits, row)) :
        {Impl.combine.go(~A, ~merge, ~P, ~e, m, 1n+k, c, q, r, S, B, M, merge(acc, _)) == acc : A}
      %Equal.sym(Bool, Nat.is_lt(k, live), False{}, rb_sub_zero_nlt(live, k, hd)) :
        {Impl.combine.go(~A, ~merge, ~P, ~e, m, 1n+k, c, q, r, S, B, M, merge(acc, Impl.partial(~A, ~P, ~e, _, SLO, SHI))) == acc : A}
      %Equal.sym(A, merge(acc, e), acc, h) :
        {Impl.combine.go(~A, ~merge, ~P, ~e, m, 1n+k, c, q, r, S, B, M, _) == acc : A}
      rb_tail(~A, ~merge, ~P, ~e, m, 1n+k, c, q, r, S, B, M, acc, h, rb_sub_zero_succ(live, k, hd), fits, row)

# The combine over `left` splits from split k equals the reference fold over the d = chunks - k
# remaining own chunks (d <= left); splits past the last chunk merge the identity. rid is used
# once on every path (passed on, or instantiated at the final acc for rb_tail).
def rb_fold(~A: Data, ~merge: A -> A -> A, ~P: Nat -> Nat -> A, ~e: A, rid: @x:A -> {merge(x, e) == x : A},
    +left: Nat, +k: Nat, +c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat, +acc: A, +d: Nat,
    +hd: {Nat.sub(Spec.chunks(Nat.add(c, r), S, B, M), k) == d : Nat}, dle: {Nat.is_le(d, left) == True{} : Bool},
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.combine.go(~A, ~merge, ~P, ~e, left, k, c, q, r, S, B, M, acc) ==
      Spec.fold.go(~A, ~merge, ~P, d, k, Nat.add(c, r), S, B, M, acc) : A}:
  match left d:
    case 0n 0n:
      {==}
    case 0n 1n+dp:
      Empty.absurd({Impl.combine.go(~A, ~merge, ~P, ~e, 0n, k, c, q, r, S, B, M, acc) ==
        Spec.fold.go(~A, ~merge, ~P, 1n+dp, k, Nat.add(c, r), S, B, M, acc) : A}, Ar.ar_false_true(dle))
    case 1n+m 0n:
      rb_tail(~A, ~merge, ~P, ~e, 1n+m, k, c, q, r, S, B, M, acc, rid(acc), hd, fits, row)
    case 1n+m 1n+dp:
      +live = Spec.chunks(Nat.add(c, r), S, B, M)
      +ILO = Impl.chunk_lo(c, q, r, k, S, B, M)
      +IHI = Impl.chunk_hi(c, q, r, k, S, B, M)
      +SLO = Spec.chunk_lo(Nat.add(c, r), k, S, B, M)
      +SHI = Spec.chunk_hi(Nat.add(c, r), k, S, B, M)
      +RHS = Spec.fold.go(~A, ~merge, ~P, 1n+dp, k, Nat.add(c, r), S, B, M, acc)
      %Equal.sym(A, Impl.partial(~A, ~P, ~e, Nat.is_lt(ILO, IHI), ILO, IHI), Impl.partial(~A, ~P, ~e, Nat.is_lt(k, live), SLO, SHI),
        rb_kpart(~A, ~P, ~e, c, q, r, k, S, B, M, fits, row)) :
        {Impl.combine.go(~A, ~merge, ~P, ~e, m, 1n+k, c, q, r, S, B, M, merge(acc, _)) == RHS : A}
      %Equal.sym(Bool, Nat.is_lt(k, live), True{}, rb_sub_succ_lt(live, k, dp, hd)) :
        {Impl.combine.go(~A, ~merge, ~P, ~e, m, 1n+k, c, q, r, S, B, M, merge(acc, Impl.partial(~A, ~P, ~e, _, SLO, SHI))) == RHS : A}
      rb_fold(~A, ~merge, ~P, ~e, rid, m, 1n+k, c, q, r, S, B, M, merge(acc, P(SLO, SHI)), dp,
        rb_sub_succ_pred(live, k, dp, hd), dle, fits, row)

def Laws.combine_absolute(A, merge, P, e, rid, c, q_len, r, splits, block_n, min_split, fits_, row_):
  +fits = fits_
  +row = row_
  +p = Nat.add(c, r)
  +live = Spec.chunks(p, splits, block_n, min_split)
  +tpos = rb_tpos(q_len, r, Nat.mul(splits, block_n), fits, row)
  +bp = rb_pos_r(splits, block_n, tpos)
  +si = Impl.row_split_len(1n+p, splits, block_n, min_split)
  +sl = Spec.row_len(1n+p, splits, block_n, min_split)
  +cov = Ar.ard_rt(Nat.is_le(Spec.ceil_div(1n+p, si), splits), Nat.is_le(live, splits),
    Equal.cong(Nat, Bool, x => Nat.is_le(Spec.ceil_div(1n+p, x), splits), si, sl,
      rb_spec_eq(1n+p, splits, block_n, min_split, rb_pos_l(splits, block_n, tpos), bp)),
    rb_cover(1n+p, splits, block_n, min_split, {==}, tpos, bp))
  rb_fold(~A, ~merge, ~P, ~e, rid, splits, 0n, c, q_len, r, splits, block_n, min_split, e, live,
    Ar.ar_sub_zero_r(live), cov, fits, row)

# ---------------------------------------------------------------- B7: every key exactly once

# The row's split length is positive.
def rb_ks_pos(+c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Nat.is_lt(0n, Impl.kernel_row_split_len(c, q, r, S, B, M)) == True{} : Bool}:
  +KSL = Impl.kernel_row_split_len(c, q, r, S, B, M)
  +SL = Spec.row_len(1n+Nat.add(c, r), S, B, M)
  Ar.ard_tr(Nat.is_lt(0n, KSL), Nat.is_lt(0n, SL),
    Equal.cong(Nat, Bool, x => Nat.is_lt(0n, x), KSL, SL, rb_b1_len(c, q, r, S, B, M, fits, row)),
    rb_sl_pos(Nat.add(c, r), S, B, M, rb_tpos(q, r, Nat.mul(S, B), fits, row)))

# k * s <= j < k * s + s gives j / s == k (s >= 1)
def rb_div_eq(+k: Nat, +s: Nat, +j: Nat, +pos: {Nat.is_lt(0n, s) == True{} : Bool},
    lo: {Nat.is_le(Nat.mul(k, s), j) == True{} : Bool}, hi: {Nat.is_lt(j, Nat.add(Nat.mul(k, s), s)) == True{} : Bool})
  -> {Nat.div(j, s) == k : Nat}:
  +x = Nat.div(j, s)
  +ks = Nat.mul(k, s)
  +kle = Ar.ard_rt(Nat.is_le(ks, j), Nat.is_le(k, x), Ar.ar_div_le(k, j, s, pos), lo)
  # k * s + s == (1 + k) * s
  +hi2 = Ar.ard_rt(Nat.is_lt(j, Nat.add(ks, s)), Nat.is_lt(j, Nat.mul(1n+k, s)),
    Equal.cong(Nat, Bool, t => Nat.is_lt(j, t), Nat.add(ks, s), Nat.add(s, ks), Ar.ar_add_comm(ks, s)), hi)
  +xlt = Ar.ard_tr(Nat.is_lt(x, 1n+k), Nat.is_lt(j, Nat.mul(1n+k, s)), Ar.ar_div_lt(1n+k, j, s, pos), hi2)
  +xle = Ar.ard_rt(Nat.is_lt(x, 1n+k), Nat.is_le(x, k), Ar.ar_lt_succ_le(x, 1n+k), xlt)
  Ar.ar_le_antisym(x, k, xle, kle)

# key_valid(k, j) gives k * s <= j
def rb_kv_lo(+c: Nat, +q: Nat, +r: Nat, +k: Nat, +j: Nat, +S: Nat, +B: Nat, +M: Nat,
    v: {Impl.key_valid(c, q, r, k, j, S, B, M) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(k, Impl.kernel_row_split_len(c, q, r, S, B, M)), j) == True{} : Bool}:
  +s = Impl.kernel_row_split_len(c, q, r, S, B, M)
  +ns = Nat.mul(k, s)
  Ar.ar_and_l(Nat.is_le(ns, j), Bool.and(Nat.is_lt(j, Nat.min(Nat.add(ns, s), Nat.add(c, q))), Nat.is_le(j, Nat.add(c, r))), v)

# key_valid(k, j) gives j < k * s + s
def rb_kv_hi(+c: Nat, +q: Nat, +r: Nat, +k: Nat, +j: Nat, +S: Nat, +B: Nat, +M: Nat,
    v: {Impl.key_valid(c, q, r, k, j, S, B, M) == True{} : Bool})
  -> {Nat.is_lt(j, Nat.add(Nat.mul(k, Impl.kernel_row_split_len(c, q, r, S, B, M)), Impl.kernel_row_split_len(c, q, r, S, B, M))) == True{} : Bool}:
  +s = Impl.kernel_row_split_len(c, q, r, S, B, M)
  +ns = Nat.mul(k, s)
  +T = Nat.add(c, q)
  +rest = Ar.ar_and_r(Nat.is_le(ns, j), Bool.and(Nat.is_lt(j, Nat.min(Nat.add(ns, s), T)), Nat.is_le(j, Nat.add(c, r))), v)
  +ltmin = Ar.ar_and_l(Nat.is_lt(j, Nat.min(Nat.add(ns, s), T)), Nat.is_le(j, Nat.add(c, r)), rest)
  Ar.ar_and_l(Nat.is_lt(j, Nat.add(ns, s)), Nat.is_lt(j, T),
    Ar.ard_rt(Nat.is_lt(j, Nat.min(Nat.add(ns, s), T)), Bool.and(Nat.is_lt(j, Nat.add(ns, s)), Nat.is_lt(j, T)),
      Ar.ar_min_lt(j, Nat.add(ns, s), T), ltmin))

def Laws.key_causal(c, q_len, r, k, j, splits, block_n, min_split, v):
  +s = Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split)
  +ns = Nat.mul(k, s)
  +T = Nat.add(c, q_len)
  Ar.ar_and_r(Nat.is_lt(j, Nat.min(Nat.add(ns, s), T)), Nat.is_le(j, Nat.add(c, r)),
    Ar.ar_and_r(Nat.is_le(ns, j), Bool.and(Nat.is_lt(j, Nat.min(Nat.add(ns, s), T)), Nat.is_le(j, Nat.add(c, r))), v))

def Laws.key_unique(c, q_len, r, k1, k2, j, splits, block_n, min_split, fits_, row_, v1_, v2_):
  +fits = fits_
  +row = row_
  +v1 = v1_
  +v2 = v2_
  +s = Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split)
  +pos = rb_ks_pos(c, q_len, r, splits, block_n, min_split, fits, row)
  +e1 = rb_div_eq(k1, s, j, pos, rb_kv_lo(c, q_len, r, k1, j, splits, block_n, min_split, v1),
    rb_kv_hi(c, q_len, r, k1, j, splits, block_n, min_split, v1))
  +e2 = rb_div_eq(k2, s, j, pos, rb_kv_lo(c, q_len, r, k2, j, splits, block_n, min_split, v2),
    rb_kv_hi(c, q_len, r, k2, j, splits, block_n, min_split, v2))
  Equal.trans(Nat, k1, Nat.div(j, s), k2, Equal.sym(Nat, Nat.div(j, s), k1, e1), e2)

def Laws.key_covered(c, q_len, r, j, splits, block_n, min_split, fits_, row_, causal_):
  +fits = fits_
  +row = row_
  +causal = causal_
  +p = Nat.add(c, r)
  +T = Nat.add(c, q_len)
  +s = Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split)
  +RS = Impl.row_split_len(1n+p, splits, block_n, min_split)
  +x = Nat.div(j, s)
  +xs = Nat.mul(x, s)
  +pos = rb_ks_pos(c, q_len, r, splits, block_n, min_split, fits, row)
  +tpos = rb_tpos(q_len, r, Nat.mul(splits, block_n), fits, row)
  # s == s(p + 1)
  +srs = Equal.trans(Nat, s, Spec.row_len(1n+p, splits, block_n, min_split), RS,
    rb_b1_len(c, q_len, r, splits, block_n, min_split, fits, row),
    Equal.sym(Nat, RS, Spec.row_len(1n+p, splits, block_n, min_split),
      rb_spec_eq(1n+p, splits, block_n, min_split, rb_pos_l(splits, block_n, tpos), rb_pos_r(splits, block_n, tpos))))
  # j < p + 1 <= SPLITS * s
  +jp = Ar.ard_tr(Nat.is_lt(j, 1n+p), Nat.is_le(j, p), Ar.ar_lt_succ_le(j, 1n+p), causal)
  +cov = Ar.ard_tr(Nat.is_le(1n+p, Nat.mul(splits, s)), Nat.is_le(1n+p, Nat.mul(splits, RS)),
    Equal.cong(Nat, Bool, t => Nat.is_le(1n+p, Nat.mul(splits, t)), s, RS, srs),
    rb_cover_le(1n+p, splits, block_n, min_split, tpos))
  +xlt = Ar.ard_tr(Nat.is_lt(x, splits), Nat.is_lt(j, Nat.mul(splits, s)), Ar.ar_div_lt(splits, j, s, pos),
    Ar.ar_lt_le_trans(j, 1n+p, Nat.mul(splits, s), jp, cov))
  # x * s <= j < x * s + s, j <= p < c + q_len
  +lo = Ar.ard_le_mul_div(j, s, pos)
  +hi = Ar.ard_rt(Nat.is_lt(j, Nat.add(s, xs)), Nat.is_lt(j, Nat.add(xs, s)),
    Equal.cong(Nat, Bool, t => Nat.is_lt(j, t), Nat.add(s, xs), Nat.add(xs, s), Ar.ar_add_comm(s, xs)), Ar.ard_lt_next(j, s, pos))
  +jT = Ar.ar_le_lt_trans(j, p, T, causal, Ar.ard_tr(Nat.is_lt(p, T), Nat.is_lt(r, q_len), Ar.ar_lt_add_cancel(c, r, q_len), row))
  +minlt = Ar.ard_tr(Nat.is_lt(j, Nat.min(Nat.add(xs, s), T)), Bool.and(Nat.is_lt(j, Nat.add(xs, s)), Nat.is_lt(j, T)),
    Ar.ar_min_lt(j, Nat.add(xs, s), T), Ar.ar_and_intro(Nat.is_lt(j, Nat.add(xs, s)), Nat.is_lt(j, T), hi, jT))
  (xlt, Ar.ar_and_intro(Nat.is_le(xs, j), Bool.and(Nat.is_lt(j, Nat.min(Nat.add(xs, s), T)), Nat.is_le(j, p)), lo,
    Ar.ar_and_intro(Nat.is_lt(j, Nat.min(Nat.add(xs, s), T)), Nat.is_le(j, p), minlt, causal)))

# ---------------------------------------------------------------- B8: every row exactly once

def rb_once(+x: Bool, +y: Bool, h: {Bool.and(x, Bool.not(y)) == False{} : Bool})
  -> {Nat.add(Split.bit(Bool.not(x)), Split.bit(Bool.and(x, y))) == 1n : Nat}:
  match x y:
    case False{} False{}:
      {==}
    case False{} True{}:
      {==}
    case True{} False{}:
      Empty.absurd({Nat.add(Split.bit(False{}), Split.bit(False{})) == 1n : Nat}, Ar.ar_true_false(h))
    case True{} True{}:
      {==}

def Laws.pass_exactly_once(c, q_len, r, splits, block_n, min_split, fits_, row_):
  +fits = fits_
  +row = row_
  rb_once(Impl.in_pass_b(c, r, splits, block_n, min_split), Impl.runs_pass_b(c, q_len, splits, block_n, min_split),
    rb_b3c(c, q_len, r, splits, block_n, min_split, fits, row))

def Laws.pass_b_single_row(c, splits, block_n, min_split):
  +X = Impl.row_split_len(1n+c, splits, block_n, min_split)
  +e = Equal.trans(Nat, Nat.add(c, 1n), 1n+Nat.add(c, 0n), 1n+c, Ar.ar_add_succ_r(c, 0n),
    Equal.cong(Nat, Nat, t => 1n+t, Nat.add(c, 0n), c, Ar.ar_add_zero_r(c)))
  %Equal.sym(Nat, Nat.add(c, 1n), 1n+c, e) :
    {Bool.not(Nat.is_eq(Impl.row_split_len(_, splits, block_n, min_split), X)) == False{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(X, X), True{}, Ar.ar_eq_refl(X)) : {Bool.not(_) == False{} : Bool}
  {==}

# ---------------------------------------------------------------- B9: chunk count

# chunk k of row r is non-empty iff k < chunks(c + r)
def rb_ne(+c: Nat, +q: Nat, +r: Nat, +k: Nat, +S: Nat, +B: Nat, +M: Nat,
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Nat.is_lt(Impl.chunk_lo(c, q, r, k, S, B, M), Impl.chunk_hi(c, q, r, k, S, B, M)) ==
      Nat.is_lt(k, Spec.chunks(Nat.add(c, r), S, B, M)) : Bool}:
  +p = Nat.add(c, r)
  +ILO = Impl.chunk_lo(c, q, r, k, S, B, M)
  +IHI = Impl.chunk_hi(c, q, r, k, S, B, M)
  +SLO = Spec.chunk_lo(p, k, S, B, M)
  +SHI = Spec.chunk_hi(p, k, S, B, M)
  +RHS = Nat.is_lt(k, Spec.chunks(p, S, B, M))
  %Equal.sym(Nat, ILO, SLO, rb_b1_lo(c, q, r, k, S, B, M, fits, row)) : {Nat.is_lt(_, IHI) == RHS : Bool}
  %Equal.sym(Nat, IHI, SHI, rb_b1_hi(c, q, r, k, S, B, M, fits, row)) : {Nat.is_lt(SLO, _) == RHS : Bool}
  rb_nonempty(p, k, Spec.row_len(1n+p, S, B, M), rb_sl_pos(p, S, B, M, rb_tpos(q, r, Nat.mul(S, B), fits, row)))

# The count over `left` splits from k is d = chunks - k (d <= left).
def rb_count(+left: Nat, +k: Nat, +c: Nat, +q: Nat, +r: Nat, +S: Nat, +B: Nat, +M: Nat, +d: Nat,
    +hd: {Nat.sub(Spec.chunks(Nat.add(c, r), S, B, M), k) == d : Nat}, dle: {Nat.is_le(d, left) == True{} : Bool},
    +fits: {Nat.is_le(q, Nat.mul(S, B)) == True{} : Bool}, +row: {Nat.is_lt(r, q) == True{} : Bool})
  -> {Impl.chunk_count.go(left, k, c, q, r, S, B, M) == d : Nat}:
  match left d:
    case 0n 0n:
      {==}
    case 0n 1n+dp:
      Empty.absurd({Impl.chunk_count.go(0n, k, c, q, r, S, B, M) == 1n+dp : Nat}, Ar.ar_false_true(dle))
    case 1n+m 0n:
      +live = Spec.chunks(Nat.add(c, r), S, B, M)
      +ILO = Impl.chunk_lo(c, q, r, k, S, B, M)
      +IHI = Impl.chunk_hi(c, q, r, k, S, B, M)
      +REST = Impl.chunk_count.go(m, 1n+k, c, q, r, S, B, M)
      %Equal.sym(Bool, Nat.is_lt(ILO, IHI), Nat.is_lt(k, live), rb_ne(c, q, r, k, S, B, M, fits, row)) :
        {Nat.add(Split.bit(_), REST) == 0n : Nat}
      %Equal.sym(Bool, Nat.is_lt(k, live), False{}, rb_sub_zero_nlt(live, k, hd)) :
        {Nat.add(Split.bit(_), REST) == 0n : Nat}
      rb_count(m, 1n+k, c, q, r, S, B, M, 0n, rb_sub_zero_succ(live, k, hd), Ar.ar_h_le_zero(m), fits, row)
    case 1n+m 1n+dp:
      +live = Spec.chunks(Nat.add(c, r), S, B, M)
      +ILO = Impl.chunk_lo(c, q, r, k, S, B, M)
      +IHI = Impl.chunk_hi(c, q, r, k, S, B, M)
      +REST = Impl.chunk_count.go(m, 1n+k, c, q, r, S, B, M)
      %Equal.sym(Bool, Nat.is_lt(ILO, IHI), Nat.is_lt(k, live), rb_ne(c, q, r, k, S, B, M, fits, row)) :
        {Nat.add(Split.bit(_), REST) == 1n+dp : Nat}
      %Equal.sym(Bool, Nat.is_lt(k, live), True{}, rb_sub_succ_lt(live, k, dp, hd)) :
        {Nat.add(Split.bit(_), REST) == 1n+dp : Nat}
      Equal.cong(Nat, Nat, t => 1n+t, REST, dp,
        rb_count(m, 1n+k, c, q, r, S, B, M, dp, rb_sub_succ_pred(live, k, dp, hd), dle, fits, row))

def Laws.row_chunk_count(c, q_len, r, splits, block_n, min_split, fits_, row_):
  +fits = fits_
  +row = row_
  +p = Nat.add(c, r)
  +live = Spec.chunks(p, splits, block_n, min_split)
  +tpos = rb_tpos(q_len, r, Nat.mul(splits, block_n), fits, row)
  +bp = rb_pos_r(splits, block_n, tpos)
  +si = Impl.row_split_len(1n+p, splits, block_n, min_split)
  +sl = Spec.row_len(1n+p, splits, block_n, min_split)
  +cov = Ar.ard_rt(Nat.is_le(Spec.ceil_div(1n+p, si), splits), Nat.is_le(live, splits),
    Equal.cong(Nat, Bool, x => Nat.is_le(Spec.ceil_div(1n+p, x), splits), si, sl,
      rb_spec_eq(1n+p, splits, block_n, min_split, rb_pos_l(splits, block_n, tpos), bp)),
    rb_cover(1n+p, splits, block_n, min_split, {==}, tpos, bp))
  rb_count(splits, 0n, c, q_len, r, splits, block_n, min_split, live, Ar.ar_sub_zero_r(live), cov, fits, row)

# ---------------------------------------------------------------- B10: accumulator store guard

def rb_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      Ar.ar_h_le_zero(b)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      rb_min_le_r(ap, bp)

# sA <= the split length of either pass
def rb_pick_ge(b: Bool, +sA: Nat, +sB: Nat, h: {Nat.is_le(sA, sB) == True{} : Bool})
  -> {Nat.is_le(sA, Impl.pick(b, sB, sA)) == True{} : Bool}:
  match b:
    case True{}:
      h
    case False{}:
      Ar.ar_le_refl(sA)

def Laws.acc_stored(c, q_len, r, k, splits, block_n, min_split, fits_, row_, live):
  +fits = fits_
  +row = row_
  +T = Nat.add(c, q_len)
  +sA = Impl.pass_a_len(c, splits, block_n, min_split)
  +s = Impl.kernel_row_split_len(c, q_len, r, splits, block_n, min_split)
  +ILO = Impl.chunk_lo(c, q_len, r, k, splits, block_n, min_split)
  +IHI = Impl.chunk_hi(c, q_len, r, k, splits, block_n, min_split)
  +sa_le = rb_pick_ge(Impl.in_pass_b(c, r, splits, block_n, min_split), sA, Impl.pass_b_len(c, q_len, splits, block_n, min_split),
    rb_b3b(c, q_len, r, splits, block_n, min_split, fits, row))
  +hi_le = Ar.ar_le_trans(IHI, 1n+Nat.add(c, r), T,
    rb_min_le_r(Nat.min(Nat.add(ILO, s), T), 1n+Nat.add(c, r)), rb_row_le(c, r, q_len, row))
  Ar.ar_le_lt_trans(Nat.mul(k, sA), ILO, T, rb_mul_mono_r(sA, s, k, sa_le), Ar.ar_lt_le_trans(ILO, IHI, T, live, hi_le))
